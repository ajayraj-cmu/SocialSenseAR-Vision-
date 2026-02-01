"""SAM3 text-prompted segmentation engine — fully optimized.

Optimization tiers:
1. PyTorch optimized (no TRT engines needed):
   - Batched decoder (all prompts in one forward pass)
   - Fast image preprocessing (bypasses PIL + HF processor in hot path)
   - torch.compile on vision encoder AND decoder
   - FP16 inference
   - Pre-tokenized and padded prompts for efficient batching

2. TensorRT (when engines exist in project root):
   - Direct TRT inference for vision encoder (~15-25ms vs ~350ms PyTorch)
   - Direct TRT inference for decoder (~3-8ms vs ~83ms PyTorch per prompt)
   - Pre-allocated CUDA tensors, high-priority stream
   - Expected: 30-45+ FPS

Performance tiers on RTX 3060 Laptop:
- Baseline (no optimizations):  ~0.5 FPS (1920ms)
- v1 (FP16 + vision cache):     ~1.4 FPS (690ms)
- v2 (batched + compile):       ~3-5 FPS (200-350ms)
- v3 (TensorRT):                ~30-45 FPS (22-33ms)

Same interface as FastSAMSegmenter:
    initialize(), segment_frame(frame_bgr) -> list[SegmentData], shutdown(), .last_person_mask
"""

import os
import time
import logging
import cv2
import numpy as np

from server.vision.segment_data import SegmentData

logger = logging.getLogger(__name__)

# Direct debug file for bypassing buffered logging
_DEBUG_FILE = os.path.join(os.path.expanduser("~"), "sam3_debug.log")
def _dbg(msg):
    """Write debug message directly to file (bypasses Python log buffering)."""
    try:
        with open(_DEBUG_FILE, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass

# All categories to detect. "person" is always queried; others rotate.
ALL_PROMPTS = [
    "person", "face", "chair", "table", "desk", "couch", "monitor",
    "laptop", "lamp", "wall", "floor", "door", "window",
]

# Map labels -> asset classes
_ASSET_CLASS_MAP = {
    "person": "person", "face": "person",
    "chair": "furniture", "table": "furniture", "desk": "furniture",
    "couch": "furniture",
    "monitor": "electronics", "laptop": "electronics",
    "lamp": "lighting", "light": "lighting",
    "wall": "structure", "floor": "structure",
    "ceiling": "structure", "door": "structure", "window": "structure",
}

# Project root (for finding engine files)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SAM3Segmenter:
    """Text-prompted SAM3 segmenter with multi-tier optimization.

    Automatically uses TensorRT engines if found, otherwise falls back
    to optimized PyTorch with batched decoding and torch.compile.
    Same interface as FastSAMSegmenter so the orchestrator can swap them.
    """

    def __init__(self, config):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._initialized = False
        self._torch = None

        # Public: expose person_mask for pipeline
        self.last_person_mask: np.ndarray | None = None

        # Pre-tokenized prompts: prompt -> {input_ids, attention_mask} (padded to max_len)
        self._tokenized: dict[str, dict] = {}
        self._max_token_len = 0

        # Round-robin for non-person prompts
        self._rotate_index = 0
        self._prompts_per_frame = config.sam3_prompts_per_frame
        self._confidence_threshold = config.sam3_confidence_threshold

        # Mask cache: prompt -> (mask_u8, timestamp) for non-queried prompts
        self._mask_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._cache_ttl = config.sam3_cache_ttl

        self._frame_count = 0
        self._decoder_skipped = False  # True if last frame used fully cached decoder

        # Frame similarity check for vision skip (Phase 4)
        self._last_thumb = None       # 64x64 thumbnail of previous frame
        self._vision_skip_count = 0   # how many consecutive frames skipped vision

        # TRT engine backend (legacy .engine files)
        self._use_trt = False
        self._trt = None

        # Native TRT decoder backend (built .engine file)
        self._use_trt_decoder = False  # hybrid: PyTorch vision + TRT decoder
        self._trt_topk = False         # True if using top-k engine (best_mask+best_score outputs)
        self._trt_context = None       # TRT execution context
        self._trt_engine = None        # TRT engine
        self._trt_stream = None        # CUDA stream for TRT

        # TRT vision encoder backend
        self._use_trt_vision = False
        self._trt_vis_context = None
        self._trt_vis_engine = None
        self._trt_vis_stream = None
        self._trt_vis_outputs = {}     # name -> pre-allocated output tensor

        # Async pipeline: double-buffered vision for overlapping vision+decoder
        self._pipeline_state = "IDLE"  # IDLE | VISION_RUNNING
        self._vis_buffers = [None, None]  # two sets of vision output tensors
        self._vis_buf_write = 0   # index vision TRT writes to
        self._vis_buf_read = 0    # index decoder reads from
        self._vis_done_event = None  # torch.cuda.Event for async completion

        # Whether TRT engines are available (set in initialize())
        self._trt_vision_available = False
        self._trt_decoder_available = False

        # TRT metadata (shapes from sam3_meta.json, resolution-flexible)
        self._trt_meta = None
        self._dec_fpn_shapes = None  # {name: [B,C,H,W]} for decoder input shapes
        self._dec_mask_hw = (288, 288)  # mask output (H,W), updated from meta

        # Fast preprocessing (discovered from processor during init)
        self._img_target_h = 0
        self._img_target_w = 0
        self._img_mean = None  # (1, 3, 1, 1) tensor on device
        self._img_std = None
        self._do_rescale = True
        self._fast_preprocess_ready = False

        # Compiled forward for decoder (may be same as model.forward if compile fails)
        self._compiled_forward = None

        # Pre-computed text embeddings: prompt -> (text_embeds, attention_mask)
        # Bypasses CLIP-24L text encoder entirely during inference
        self._text_embeds: dict[str, dict] = {}
        self._text_embeds_ready = False

        # Batching support
        self._batch_supported = True  # set False if batched call fails

        # Cached vision embeds (current frame, for decoder)
        self._cached_vision_embeds = None

        # Cached FP32 FPN features for TRT (avoid FP16→FP32 every frame)
        self._cached_fpn_fp32 = None  # (fpn_hs_0, fpn_hs_1, fpn_hs_2, fpn_pe_2) or None

        # Pre-computed FP32 text embeddings for TRT decoder
        self._text_embeds_fp32 = {}  # prompt -> (text_embeds_fp32, attn_mask_i64)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Load SAM3 — PyTorch for vision, TRT engine for decoder (if available).

        Initialization order is VRAM-aware:
        1. Load PyTorch model (text encoder + vision encoder + decoder)
        2. Pre-tokenize + pre-compute text embeddings (uses text encoder only)
        3. If TRT vision engine exists: skip torch.compile on vision_encoder,
           delete it to free ~400 MB VRAM, then load TRT vision engine
        4. Load TRT decoder engine
        5. Warmup (TRT vision if available, else PyTorch vision)
        """
        import torch
        self._torch = torch
        self._device = self.config.device

        # Detect TRT engines early (before init) so we can skip unnecessary VRAM usage
        # Resolution-aware: look for _<res> suffixed files first, then unsuffixed fallback
        res = self.config.sam3_resolution
        logger.info(f"    SAM3 resolution: {res}x{res}")

        def _find_engine(base_name):
            """Find engine file: try _<res> suffix first, then unsuffixed."""
            suffixed = os.path.join(_PROJECT_ROOT, f"{base_name}_{res}.engine")
            unsuffixed = os.path.join(_PROJECT_ROOT, f"{base_name}.engine")
            if os.path.exists(suffixed):
                return suffixed
            if os.path.exists(unsuffixed):
                return unsuffixed
            return None

        def _find_meta():
            """Find meta JSON: try _<res> suffix first, then unsuffixed."""
            suffixed = os.path.join(_PROJECT_ROOT, f"sam3_meta_{res}.json")
            unsuffixed = os.path.join(_PROJECT_ROOT, "sam3_meta.json")
            if os.path.exists(suffixed):
                return suffixed
            if os.path.exists(unsuffixed):
                return unsuffixed
            return None

        # Prefer INT8 engine over FP16 (better perf on Ampere+)
        vis_engine_int8 = _find_engine("sam3_vision_int8")
        vis_engine_fp16 = _find_engine("sam3_vision")
        meta_path = _find_meta()
        if vis_engine_int8 and meta_path:
            vis_engine = vis_engine_int8
            logger.info(f"    Using INT8 vision engine: {os.path.basename(vis_engine)}")
        else:
            vis_engine = vis_engine_fp16
        self._trt_vision_available = vis_engine is not None and meta_path is not None
        if vis_engine:
            logger.info(f"    Vision engine: {os.path.basename(vis_engine)}")
        if meta_path:
            logger.info(f"    Meta: {os.path.basename(meta_path)}")

        topk_path = _find_engine("sam3_topk_decoder")
        full_path = _find_engine("sam3_decoder")
        dec_engine_path = topk_path or full_path
        self._trt_decoder_available = dec_engine_path is not None

        # Init PyTorch (skips torch.compile + warmup for components with TRT replacements)
        self._init_pytorch()

        # --- TRT Vision Loading ---
        # Must load AFTER deleting PyTorch vision_encoder (892 MB engine needs VRAM headroom)
        if self._trt_vision_available:
            # Free PyTorch vision encoder FIRST to make room for TRT engine
            if hasattr(self._model, 'vision_encoder'):
                del self._model.vision_encoder
                self._torch.cuda.empty_cache()
                logger.info("    Freed PyTorch vision_encoder to make room for TRT vision")
            try:
                self._init_trt_vision(vis_engine)
            except Exception as e:
                logger.warning(f"TRT vision init failed, reloading PyTorch vision: {e}")
                import traceback
                traceback.print_exc()
                # Reload PyTorch vision encoder as fallback
                self._init_pytorch_model_only()

        # --- TRT Decoder Loading ---
        if self._trt_decoder_available:
            try:
                self._init_trt_decoder(dec_engine_path)
            except Exception as e:
                logger.warning(f"TRT decoder init failed, using PyTorch decoder: {e}")
                import traceback
                traceback.print_exc()

        # --- Free unused PyTorch components to reclaim VRAM ---
        freed = []
        # Text encoder: only needed during init for _precompute_text_embeds()
        if self._text_embeds_ready and hasattr(self._model, 'text_encoder'):
            del self._model.text_encoder
            freed.append("text_encoder")
        # Decoder components: only needed if PyTorch decoder fallback is active
        if self._use_trt_decoder:
            for name in ["detr_encoder", "detr_decoder", "mask_decoder"]:
                if hasattr(self._model, name):
                    delattr(self._model, name)
                    freed.append(name)
        if freed:
            self._torch.cuda.empty_cache()
            logger.info(f"    Freed PyTorch components: {', '.join(freed)}")

        self._initialized = True

    def _init_pytorch(self):
        """Load SAM3 from HuggingFace with all PyTorch-level optimizations."""
        torch = self._torch
        t0 = time.perf_counter()

        from transformers import Sam3Model, Sam3Processor

        t_import = time.perf_counter()
        logger.info(f"    import transformers: {(t_import - t0)*1000:.0f}ms")

        model_name = self.config.sam3_model
        logger.info(f"    Loading SAM3 ({model_name}) on {self._device}")

        self._processor = Sam3Processor.from_pretrained(model_name)
        t_proc = time.perf_counter()
        logger.info(f"    SAM3 processor load: {(t_proc - t_import)*1000:.0f}ms")

        self._model = Sam3Model.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        t_model = time.perf_counter()
        logger.info(f"    SAM3 model load: {(t_model - t_proc)*1000:.0f}ms")

        # FP16
        if self._device == "cuda":
            self._model = self._model.half()
            logger.info("    SAM3 using FP16")

        # --- Discover fast preprocessing params from processor ---
        self._setup_fast_preprocess()

        # --- Pre-tokenize all prompts with padding for batched decoding ---
        self._pretokenize_padded()

        # --- Pre-compute text embeddings (bypass CLIP-24L during inference) ---
        self._precompute_text_embeds()

        # --- torch.compile: only for components WITHOUT TRT replacements ---
        # "default" mode only — reduce-overhead/max-autotune use CUDA graphs
        # which conflict with SAM3's FPN neck (outputs get overwritten on replay)
        compile_targets = []
        if not self._trt_vision_available:
            compile_targets.append("vision_encoder")
        else:
            logger.info("    Skipping torch.compile on vision_encoder (TRT vision available)")
        if not self._trt_decoder_available:
            compile_targets.extend(["detr_encoder", "detr_decoder", "mask_decoder"])
        else:
            logger.info("    Skipping torch.compile on decoder submodules (TRT decoder available)")
        for name in compile_targets:
            try:
                setattr(self._model, name, torch.compile(
                    getattr(self._model, name), mode="default"))
                logger.info(f"    torch.compile: {name}")
            except Exception as e:
                logger.warning(f"    torch.compile {name} failed: {e}")

        # Use uncompiled model.forward (it dispatches to compiled submodules)
        self._compiled_forward = self._model.forward

        # --- Warmup: compile kernels + stabilize ---
        self._warmup_pytorch(t_model)

        logger.info(f"    SAM3Segmenter ready ({(time.perf_counter() - t0)*1000:.0f}ms total)")

    def _init_pytorch_model_only(self):
        """Reload just the PyTorch model (for vision fallback after failed TRT vision)."""
        torch = self._torch
        from transformers import Sam3Model
        self._model = Sam3Model.from_pretrained(self.config.sam3_model)
        self._model.to(self._device).eval()
        if self._device == "cuda":
            self._model = self._model.half()
        self._compiled_forward = self._model.forward
        logger.info("    PyTorch model reloaded for vision fallback")

    def _setup_fast_preprocess(self):
        """Extract preprocessing params from HF processor for fast CV2+torch path."""
        try:
            ip = self._processor.image_processor
            # Get target size
            if hasattr(ip, 'size'):
                size = ip.size
                if isinstance(size, dict):
                    if 'height' in size and 'width' in size:
                        self._img_target_h = size['height']
                        self._img_target_w = size['width']
                    elif 'longest_edge' in size:
                        # Will use longest_edge logic at frame time
                        self._img_target_h = size['longest_edge']
                        self._img_target_w = size['longest_edge']
                elif isinstance(size, (list, tuple)):
                    self._img_target_h, self._img_target_w = size[0], size[1]
                else:
                    self._img_target_h = self._img_target_w = int(size)

            # Get normalization params
            mean = getattr(ip, 'image_mean', [0.485, 0.456, 0.406])
            std = getattr(ip, 'image_std', [0.229, 0.224, 0.225])
            self._do_rescale = getattr(ip, 'do_rescale', True)

            torch = self._torch
            self._img_mean = torch.tensor(mean, device=self._device, dtype=torch.float16 if self._device == "cuda" else torch.float32).view(1, 3, 1, 1)
            self._img_std = torch.tensor(std, device=self._device, dtype=torch.float16 if self._device == "cuda" else torch.float32).view(1, 3, 1, 1)

            if self._img_target_h > 0 and self._img_target_w > 0:
                self._fast_preprocess_ready = True
                logger.info(f"    Fast preprocess: {self._img_target_h}x{self._img_target_w}, "
                            f"mean={mean}, std={std}")
            else:
                logger.info("    Fast preprocess: could not determine target size, using processor")
        except Exception as e:
            logger.warning(f"    Fast preprocess setup failed: {e}")

    def _pretokenize_padded(self):
        """Pre-tokenize all prompts and pad to uniform length for batched decoding."""
        from PIL import Image
        dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        torch = self._torch

        raw_tokens = {}
        max_len = 0
        for prompt in ALL_PROMPTS:
            inputs = self._processor(images=dummy_img, text=prompt, return_tensors="pt")
            ids = inputs["input_ids"].to(self._device)
            attn = inputs["attention_mask"].to(self._device)
            raw_tokens[prompt] = (ids, attn)
            max_len = max(max_len, ids.shape[1])

        self._max_token_len = max_len

        # Pad all to max_len
        for prompt in ALL_PROMPTS:
            ids, attn = raw_tokens[prompt]
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=0)
                attn = torch.nn.functional.pad(attn, (0, pad_len), value=0)
            self._tokenized[prompt] = {
                "input_ids": ids,       # (1, max_len)
                "attention_mask": attn,  # (1, max_len)
            }

        logger.info(f"    Pre-tokenized {len(ALL_PROMPTS)} prompts (padded to {max_len} tokens)")

    def _precompute_text_embeds(self):
        """Pre-compute text embeddings for all fixed prompts (bypass CLIP-24L at inference)."""
        torch = self._torch
        t0 = time.perf_counter()

        with torch.no_grad():
            for prompt in ALL_PROMPTS:
                tokens = self._tokenized[prompt]
                text_out = self._model.get_text_features(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    return_dict=True,
                )
                # pooler_output is (1, seq_len, 256) — the projected text features
                self._text_embeds[prompt] = {
                    "text_embeds": text_out.pooler_output.detach(),
                    "attention_mask": tokens["attention_mask"],
                }

        self._text_embeds_ready = True
        logger.info(f"    Pre-computed text embeddings for {len(ALL_PROMPTS)} prompts "
                     f"({(time.perf_counter() - t0)*1000:.0f}ms)")

    def _warmup_pytorch(self, t_model):
        """Warmup compiled vision encoder + decoder with text_embeds.

        If TRT vision is available, skips vision encoder warmup entirely
        (TRT vision has its own warmup in _init_trt_vision, and the PyTorch
        vision_encoder will be deleted to free VRAM).
        """
        torch = self._torch
        try:
            from PIL import Image
            dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            inputs = self._processor(images=dummy_img, text="person", return_tensors="pt")
            dummy_pv = inputs["pixel_values"].to(self._device)
            if self._device == "cuda":
                dummy_pv = dummy_pv.half()

            # Skip all warmup if both TRT engines are available
            if self._trt_vision_available and self._trt_decoder_available:
                logger.info("    Skipping PyTorch warmup (TRT vision + decoder available)")
                logger.info(f"    Full warm-up done: {(time.perf_counter() - t_model)*1000:.0f}ms")
                return

            if self._trt_vision_available:
                logger.info("    Skipping vision warmup (TRT vision will be used)")
                # Generate dummy vision embeddings for decoder warmup only
                with torch.no_grad():
                    vis = self._model.get_vision_features(pixel_values=dummy_pv)
            else:
                logger.info("    Warming up (torch.compile first run — may take 60-90s)...")
                # Vision encoder warmup (triggers compile)
                with torch.no_grad():
                    vis = self._model.get_vision_features(pixel_values=dummy_pv)
                logger.info(f"    Vision compile+warmup: {(time.perf_counter() - t_model)*1000:.0f}ms")

                for _ in range(3):
                    with torch.no_grad():
                        self._model.get_vision_features(pixel_values=dummy_pv)

            # Decoder warmup using pre-computed text_embeds (no CLIP)
            test_prompts = ["person", "chair", "table"]
            B = len(test_prompts)

            batch_te = torch.cat([self._text_embeds[p]["text_embeds"] for p in test_prompts], dim=0)
            batch_attn = torch.cat([self._text_embeds[p]["attention_mask"] for p in test_prompts], dim=0)
            vis_batch = self._expand_vision_output(vis, B)

            # Try batched
            try:
                with torch.no_grad():
                    self._compiled_forward(
                        vision_embeds=vis_batch,
                        text_embeds=batch_te,
                        attention_mask=batch_attn,
                    )
                self._batch_supported = True
                logger.info("    Batched decoder warmup OK (text_embeds)")
                # Stabilize
                for _ in range(2):
                    with torch.no_grad():
                        self._compiled_forward(
                            vision_embeds=vis_batch,
                            text_embeds=batch_te,
                            attention_mask=batch_attn,
                        )
            except Exception as e:
                logger.warning(f"    Batched decoder failed: {e}")
                self._batch_supported = False
                # Try sequential
                try:
                    with torch.no_grad():
                        self._compiled_forward(
                            vision_embeds=vis,
                            text_embeds=self._text_embeds["person"]["text_embeds"],
                            attention_mask=self._text_embeds["person"]["attention_mask"],
                        )
                    logger.info("    Sequential decoder warmup OK (text_embeds)")
                except Exception as e:
                    logger.warning(f"    Decoder warmup failed: {e}")

            logger.info(f"    Full warm-up done: {(time.perf_counter() - t_model)*1000:.0f}ms")
        except Exception as e:
            logger.warning(f"    SAM3 warm-up failed: {e}")
            self._compiled_forward = self._model.forward

    def _init_trt_decoder(self, engine_path):
        """Initialize native TRT decoder from pre-built .engine file.

        Supports both full decoder (200 masks) and top-k decoder (1 mask).
        Auto-detects based on output tensor names.
        """
        import torch as _torch

        t0 = time.perf_counter()

        # Ensure TRT DLLs are on PATH
        sp_dir = os.path.dirname(os.path.dirname(_torch.__file__))
        trt_libs = os.path.join(sp_dir, 'tensorrt_libs')
        if os.path.isdir(trt_libs):
            current_path = os.environ.get('PATH', '')
            if trt_libs not in current_path:
                os.environ['PATH'] = trt_libs + ';' + current_path
                logger.info(f"    Added to PATH: {trt_libs}")

        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        logger.info(f"    Loading TRT decoder engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self._trt_engine = runtime.deserialize_cuda_engine(f.read())

        ctx_mem = self._trt_engine.device_memory_size
        logger.info(f"    TRT context memory: {ctx_mem / 1024 / 1024:.0f}MB")

        self._trt_context = self._trt_engine.create_execution_context()
        self._trt_stream = _torch.cuda.Stream()

        # Detect engine type from output tensor names
        output_names = set()
        for i in range(self._trt_engine.num_io_tensors):
            name = self._trt_engine.get_tensor_name(i)
            if self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_names.add(name)
        self._trt_topk = "best_mask" in output_names

        # Read FPN shapes from meta (resolution-flexible) or use defaults
        if self._trt_meta and "decoder_input_shapes" in self._trt_meta:
            dec_in = self._trt_meta["decoder_input_shapes"]
            fpn0_shape = dec_in.get("fpn_hs_0", [1, 256, 288, 288])
            fpn1_shape = dec_in.get("fpn_hs_1", [1, 256, 144, 144])
            fpn2_shape = dec_in.get("fpn_hs_2", [1, 256, 72, 72])
        else:
            fpn0_shape = [1, 256, 288, 288]
            fpn1_shape = [1, 256, 144, 144]
            fpn2_shape = [1, 256, 72, 72]

        # Store for hot path (set_input_shape calls)
        self._dec_fpn_shapes = {
            "fpn_hs_0": fpn0_shape,
            "fpn_hs_1": fpn1_shape,
            "fpn_hs_2": fpn2_shape,
            "fpn_pe_2": fpn2_shape,  # same spatial dims as fpn_hs_2
            "text_embeds": [1, 32, 256],
            "attention_mask": [1, 32],
        }
        mask_h, mask_w = fpn0_shape[2], fpn0_shape[3]
        self._dec_mask_hw = (mask_h, mask_w)
        logger.info(f"    Decoder FPN shapes: {fpn0_shape[2:]}/"
                     f"{fpn1_shape[2:]}/{fpn2_shape[2:]}, mask={mask_h}x{mask_w}")

        if self._trt_topk:
            logger.info("    Engine type: top-k (best_mask + best_score)")
            self._trt_out_mask = _torch.empty(1, 1, mask_h, mask_w, dtype=_torch.float32, device=self._device)
            self._trt_out_score = _torch.empty(1, dtype=_torch.float32, device=self._device)
        else:
            logger.info("    Engine type: full (pred_masks + pred_logits + presence_logits)")
            self._trt_out_mask = _torch.empty(1, 200, mask_h, mask_w, dtype=_torch.float32, device=self._device)
            self._trt_out_logits = _torch.empty(1, 200, dtype=_torch.float32, device=self._device)
            self._trt_out_presence = _torch.empty(1, 1, dtype=_torch.float32, device=self._device)

        # Warmup with batch=1 (shapes from meta)
        ctx = self._trt_context
        dummy_vis = _torch.zeros(*fpn0_shape, dtype=_torch.float32, device=self._device)
        dummy_vis2 = _torch.zeros(*fpn1_shape, dtype=_torch.float32, device=self._device)
        dummy_vis3 = _torch.zeros(*fpn2_shape, dtype=_torch.float32, device=self._device)
        dummy_te = _torch.zeros(1, 32, 256, dtype=_torch.float32, device=self._device)
        dummy_am = _torch.zeros(1, 32, dtype=_torch.int64, device=self._device)

        for name, shape in self._dec_fpn_shapes.items():
            ctx.set_input_shape(name, shape)

        ctx.set_tensor_address("fpn_hs_0", dummy_vis.data_ptr())
        ctx.set_tensor_address("fpn_hs_1", dummy_vis2.data_ptr())
        ctx.set_tensor_address("fpn_hs_2", dummy_vis3.data_ptr())
        ctx.set_tensor_address("fpn_pe_2", dummy_vis3.data_ptr())
        ctx.set_tensor_address("text_embeds", dummy_te.data_ptr())
        ctx.set_tensor_address("attention_mask", dummy_am.data_ptr())

        if self._trt_topk:
            ctx.set_tensor_address("best_mask", self._trt_out_mask.data_ptr())
            ctx.set_tensor_address("best_score", self._trt_out_score.data_ptr())
        else:
            ctx.set_tensor_address("pred_masks", self._trt_out_mask.data_ptr())
            ctx.set_tensor_address("pred_logits", self._trt_out_logits.data_ptr())
            ctx.set_tensor_address("presence_logits", self._trt_out_presence.data_ptr())

        ok = ctx.execute_async_v3(self._trt_stream.cuda_stream)
        self._trt_stream.synchronize()
        if not ok:
            raise RuntimeError("TRT decoder warmup failed")

        # Pre-convert text embeddings to FP32 + contiguous for TRT
        for prompt in ALL_PROMPTS:
            te = self._text_embeds[prompt]
            self._text_embeds_fp32[prompt] = (
                te["text_embeds"].float().contiguous(),
                te["attention_mask"].long().contiguous(),
            )

        # Pre-allocate persistent output buffers (max batch = len(ALL_PROMPTS))
        max_B = len(ALL_PROMPTS)
        if self._trt_topk:
            self._trt_out_mask_persistent = _torch.empty(max_B, 1, mask_h, mask_w, dtype=_torch.float32, device=self._device)
            self._trt_out_score_persistent = _torch.empty(max_B, dtype=_torch.float32, device=self._device)

        self._trt_call_count = 0  # track calls for proactive context rebuild

        engine_type = "top-k" if self._trt_topk else "full"
        self._use_trt_decoder = True
        logger.info(f"    TRT decoder ready: {engine_type} ({(time.perf_counter() - t0)*1000:.0f}ms)")

    def _init_trt_vision(self, engine_path):
        """Initialize TRT vision encoder from pre-built .engine file.

        Replaces the PyTorch vision encoder (~280ms) with TRT (~25ms).
        Outputs are FP32 tensors matching Sam3VisionEncoderOutput structure.
        """
        import torch as _torch
        t0 = time.perf_counter()

        # Ensure TRT DLLs on PATH
        sp_dir = os.path.dirname(os.path.dirname(_torch.__file__))
        trt_libs = os.path.join(sp_dir, 'tensorrt_libs')
        if os.path.isdir(trt_libs):
            current_path = os.environ.get('PATH', '')
            if trt_libs not in current_path:
                os.environ['PATH'] = trt_libs + ';' + current_path

        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        logger.info(f"    Loading TRT vision engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            self._trt_vis_engine = runtime.deserialize_cuda_engine(f.read())

        ctx_mem = self._trt_vis_engine.device_memory_size
        logger.info(f"    TRT vision context memory: {ctx_mem / 1024 / 1024:.0f}MB")

        self._trt_vis_context = self._trt_vis_engine.create_execution_context()
        self._trt_vis_stream = _torch.cuda.Stream()

        # Read output shapes from meta.json
        meta_path = os.path.join(_PROJECT_ROOT, "sam3_meta.json")
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        self._trt_meta = meta  # Store for decoder init + resolution flexibility

        vis_shapes = meta["vision_output_shapes"]
        vis_names = meta["vision_output_names"]

        # Pre-allocate double-buffered output tensors for async pipeline
        self._trt_vis_outputs = {}
        for name in vis_names:
            shape = vis_shapes[name]
            self._trt_vis_outputs[name] = _torch.empty(
                shape, dtype=_torch.float32, device=self._device)

        # Buffer 0 = the original outputs, Buffer 1 = second copy
        self._vis_buffers[0] = self._trt_vis_outputs
        self._vis_buffers[1] = {}
        for name in vis_names:
            shape = vis_shapes[name]
            self._vis_buffers[1][name] = _torch.empty(
                shape, dtype=_torch.float32, device=self._device)
        self._vis_buf_write = 0
        self._vis_buf_read = 0
        self._vis_done_event = _torch.cuda.Event()
        self._pipeline_state = "IDLE"
        buf1_mb = sum(t.nelement() * 4 for t in self._vis_buffers[1].values()) / 1024 / 1024
        logger.info(f"    Double-buffered vision outputs (+{buf1_mb:.0f} MB)")

        # Override preprocessing target to match TRT engine's input resolution
        pv_shape = meta["pixel_values_shape"]  # e.g. [1, 3, 784, 784]
        self._img_target_h = pv_shape[2]
        self._img_target_w = pv_shape[3]
        logger.info(f"    TRT vision input resolution: {pv_shape[2]}x{pv_shape[3]}")

        # Set input shape (from meta, resolution-flexible)
        self._trt_vis_context.set_input_shape("pixel_values", pv_shape)

        # Warmup with dummy input on vision stream
        dummy_pv = _torch.zeros(*pv_shape, dtype=_torch.float32, device=self._device)
        self._trt_vis_context.set_tensor_address("pixel_values", dummy_pv.data_ptr())
        for name, tensor in self._vis_buffers[0].items():
            self._trt_vis_context.set_tensor_address(name, tensor.data_ptr())

        _torch.cuda.synchronize()
        ok = self._trt_vis_context.execute_async_v3(self._trt_vis_stream.cuda_stream)
        self._trt_vis_stream.synchronize()
        if not ok:
            raise RuntimeError("TRT vision warmup failed")

        self._use_trt_vision = True
        logger.info(f"    TRT vision ready ({(time.perf_counter() - t0)*1000:.0f}ms)")

    def _build_vision_output(self, buf_idx):
        """Build a mock Sam3VisionEncoderOutput from a buffer index."""
        buf = self._vis_buffers[buf_idx]

        class _VisionOutput:
            pass

        out = _VisionOutput()
        out.last_hidden_state = buf["last_hidden_state"]
        out.fpn_hidden_states = [buf["fpn_hs_0"], buf["fpn_hs_1"],
                                  buf["fpn_hs_2"], buf["fpn_hs_3"]]
        out.fpn_position_encoding = [buf["fpn_pe_0"], buf["fpn_pe_1"],
                                      buf["fpn_pe_2"], buf["fpn_pe_3"]]

        def _keys():
            return ["last_hidden_state", "fpn_hidden_states", "fpn_position_encoding"]
        out.keys = _keys
        out.__getitem__ = lambda key: getattr(out, key)
        return out

    def _run_vision_trt_sync(self, pixel_values):
        """Run vision encoder via TRT synchronously. Used for first frame."""
        torch = self._torch
        t0 = time.perf_counter()

        pv_f32 = pixel_values.float().contiguous()
        ctx = self._trt_vis_context
        ctx.set_tensor_address("pixel_values", pv_f32.data_ptr())

        # Write to buffer 0 (read buffer for first frame)
        for name, tensor in self._vis_buffers[0].items():
            ctx.set_tensor_address(name, tensor.data_ptr())

        torch.cuda.synchronize()
        ok = ctx.execute_async_v3(self._trt_vis_stream.cuda_stream)
        self._trt_vis_stream.synchronize()

        if not ok:
            if not hasattr(self._model, 'vision_encoder'):
                raise RuntimeError("TRT vision failed and PyTorch vision_encoder was freed")
            logger.warning("TRT vision sync failed, falling back to PyTorch")
            self._use_trt_vision = False
            with torch.no_grad():
                return self._model.get_vision_features(pixel_values=pixel_values)

        self._vis_buf_read = 0
        self._vis_buf_write = 1

        vis_ms = (time.perf_counter() - t0) * 1000
        _dbg(f"TRT vision sync: {vis_ms:.1f}ms")
        return self._build_vision_output(0)

    def _start_vision_async(self, pixel_values):
        """Launch vision TRT on stream_vis asynchronously. Non-blocking."""
        torch = self._torch

        pv_f32 = pixel_values.float().contiguous()
        ctx = self._trt_vis_context
        ctx.set_tensor_address("pixel_values", pv_f32.data_ptr())

        # Bind write-buffer outputs
        buf = self._vis_buffers[self._vis_buf_write]
        for name, tensor in buf.items():
            ctx.set_tensor_address(name, tensor.data_ptr())

        # Ensure preprocessing op (.float()) completes before TRT reads
        self._trt_vis_stream.wait_stream(torch.cuda.current_stream())

        ok = ctx.execute_async_v3(self._trt_vis_stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT vision async launch failed")

        # Record event for efficient polling
        self._vis_done_event.record(self._trt_vis_stream)
        self._pipeline_state = "VISION_RUNNING"

    def _finish_vision_async(self, force_wait=False):
        """Check/wait for async vision completion. Swap buffers if done. Returns True if done."""
        if self._pipeline_state != "VISION_RUNNING":
            return True

        if force_wait:
            self._vis_done_event.synchronize()
        elif not self._vis_done_event.query():
            return False  # Still running

        # Vision complete — swap read/write buffers
        self._vis_buf_read = self._vis_buf_write
        self._vis_buf_write = 1 - self._vis_buf_write

        # Invalidate FPN cache (decoder will rebuild from new read buffer)
        self._cached_fpn_fp32 = None

        # Update cached vision embeds to point to new read buffer
        self._cached_vision_embeds = self._build_vision_output(self._vis_buf_read)

        self._pipeline_state = "IDLE"

        if self._frame_count <= 10 or self._frame_count % 30 == 0:
            _dbg(f"TRT vision async complete (buf={self._vis_buf_read})")
        return True

    def _run_vision_trt(self, pixel_values):
        """Run vision encoder via TRT synchronously. Legacy compat wrapper."""
        return self._run_vision_trt_sync(pixel_values)

    def _init_trt(self, vision_path, decoder_path, meta_path):
        """Initialize TensorRT backend (legacy .engine files)."""
        from server.vision.sam3_trt import SAM3TRT

        t0 = time.perf_counter()
        self._trt = SAM3TRT(vision_path, decoder_path, meta_path)
        self._use_trt = True

        # Still need processor for post-processing
        from transformers import Sam3Processor
        self._processor = Sam3Processor.from_pretrained(self.config.sam3_model)

        # Pre-tokenize (needed for decoder input)
        self._pretokenize_padded()

        logger.info(f"    SAM3 TRT init: {(time.perf_counter() - t0)*1000:.0f}ms")
        logger.info(f"    SAM3Segmenter ready (TRT backend)")

    # ------------------------------------------------------------------
    # Frame similarity check (Phase 4 — vision skip)
    # ------------------------------------------------------------------

    def _frame_changed(self, frame_bgr: np.ndarray) -> bool:
        """Compare 64x64 thumbnail to detect real motion vs webcam noise.

        Returns True if the frame has changed enough to warrant re-running
        the vision encoder. Cost: ~0.1ms (negligible vs 195ms vision encode).
        """
        thumb = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        if self._last_thumb is None:
            self._last_thumb = thumb
            return True
        diff = np.mean(np.abs(thumb.astype(np.int16) - self._last_thumb.astype(np.int16)))
        changed = diff > 12.0  # webcam noise ~5-11, deliberate motion 12+
        if self._frame_count <= 30 or self._frame_count % 20 == 0:
            _dbg(f"  frame_diff={diff:.1f} changed={changed}")
        if changed:
            self._last_thumb = thumb  # only update on change to avoid drift
        return changed

    # ------------------------------------------------------------------
    # Fast image preprocessing (bypass PIL + HF processor)
    # ------------------------------------------------------------------

    def _preprocess_fast(self, frame_bgr: np.ndarray):
        """Preprocess frame using cv2+torch — ~2ms vs ~15ms through PIL+processor."""
        torch = self._torch
        h, w = frame_bgr.shape[:2]

        # Resize to model's expected input size
        resized = cv2.resize(frame_bgr, (self._img_target_w, self._img_target_h),
                             interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # To tensor: (H, W, 3) uint8 -> (1, 3, H, W) float16
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=torch.float16 if self._device == "cuda" else torch.float32, non_blocking=True)

        # Rescale 0-255 -> 0-1
        if self._do_rescale:
            tensor = tensor.float().div_(255.0)
            if self._device == "cuda":
                tensor = tensor.half()

        # Normalize with ImageNet mean/std
        tensor = (tensor - self._img_mean) / self._img_std

        return tensor

    def _preprocess_slow(self, frame_bgr: np.ndarray):
        """Fallback: use HF processor (PIL path). ~15ms."""
        from PIL import Image
        torch = self._torch

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        img_inputs = self._processor(images=pil_image, text="person", return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(self._device)
        if self._device == "cuda":
            pixel_values = pixel_values.half()
        return pixel_values

    # ------------------------------------------------------------------
    # Vision embedding cache (Tier 2)
    # ------------------------------------------------------------------

    # (_frame_thumbnail removed — no longer needed without async vision caching)


    # (Async vision encoding removed — pipeline is now fully synchronous.
    #  TRT vision runs in ~200ms, making async unnecessary.)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment_frame(self, frame_bgr: np.ndarray) -> list[SegmentData]:
        """Run SAM3 on one BGR frame — synchronous pipeline with vision skip.

        Flow: check frame similarity → (skip or run) vision encode →
              select prompts → decode → build segments.
        Vision encode (~195ms) skipped when frame is near-identical to previous.
        """
        if not self._initialized:
            self.initialize()

        torch = self._torch
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        now = time.time()
        self._frame_count += 1

        try:
            return self._segment_frame_inner(frame_bgr, torch, t0, h, w, now)
        except Exception as e:
            import traceback
            _dbg(f"CRASH in segment_frame #{self._frame_count}: {e}\n{traceback.format_exc()}")
            raise

    def _segment_frame_inner(self, frame_bgr, torch, t0, h, w, now):
        # --- Step 1: Check frame similarity (Phase 4 — vision skip) ---
        frame_changed = self._frame_changed(frame_bgr)
        force_refresh = (self._frame_count % 60 == 0)
        need_vision = frame_changed or self._cached_vision_embeds is None or force_refresh

        # --- Step 2: Collect previous async vision result (if any) ---
        if self._pipeline_state == "VISION_RUNNING":
            # Always force-wait: async pipelining doesn't overlap well on mid-range
            # GPUs (3060) due to SM/bandwidth contention. Wait for result, then
            # run decoder without contention for correct 26ms/prompt timing.
            self._finish_vision_async(force_wait=True)

        # --- Step 3: Preprocess (only if vision needed) ---
        t_pre = time.perf_counter()
        pixel_values = None
        if need_vision:
            if self._use_trt:
                pixel_values = self._trt.preprocess(frame_bgr)
            elif self._fast_preprocess_ready:
                pixel_values = self._preprocess_fast(frame_bgr)
            else:
                pixel_values = self._preprocess_slow(frame_bgr)
        pre_ms = (time.perf_counter() - t_pre) * 1000

        # --- Step 4: Vision encode ---
        t_vis = time.perf_counter()
        if need_vision:
            if self._use_trt_vision:
                vision_embeds = self._run_vision_trt_sync(pixel_values)
                self._cached_vision_embeds = vision_embeds
            elif self._use_trt:
                vision_embeds = self._trt.encode_vision(pixel_values)
                self._cached_vision_embeds = vision_embeds
            else:
                with torch.no_grad():
                    vision_embeds = self._model.get_vision_features(pixel_values=pixel_values)
                vision_embeds = self._clone_vision_output(vision_embeds)
                self._cached_vision_embeds = vision_embeds

            self._cached_fpn_fp32 = None
            self._vision_skip_count = 0
        else:
            vision_embeds = self._cached_vision_embeds
            self._vision_skip_count += 1
        vis_ms = (time.perf_counter() - t_vis) * 1000

        # --- Step 5: Select prompts ---
        n = min(self._prompts_per_frame, len(ALL_PROMPTS))
        prompts_this_frame = []
        for i in range(n):
            idx = (self._rotate_index + i) % len(ALL_PROMPTS)
            prompts_this_frame.append(ALL_PROMPTS[idx])
        self._rotate_index = (self._rotate_index + n) % len(ALL_PROMPTS)

        # --- Step 6: Decode prompts ---
        # On motion frames, only decode person + current rotating prompt(s).
        # Other cached masks stay valid and refresh via rotation cycle.
        # This keeps motion-frame decoder to 2 prompts × ~26ms = ~52ms.
        t_dec = time.perf_counter()
        if need_vision:
            prompts_to_decode = list(prompts_this_frame)
            if "person" not in prompts_to_decode:
                prompts_to_decode.insert(0, "person")
            new_masks = self._run_decoder(vision_embeds, prompts_to_decode, h, w)
        else:
            # Scene unchanged — skip decoder for prompts with fresh cache
            prompts_to_decode = []
            for p in prompts_this_frame:
                if p in self._mask_cache:
                    _, cache_time, _ = self._mask_cache[p]
                    if now - cache_time < self._cache_ttl:
                        continue
                prompts_to_decode.append(p)
            if prompts_to_decode:
                new_masks = self._run_decoder(vision_embeds, prompts_to_decode, h, w)
            else:
                new_masks = {}
        dec_ms = (time.perf_counter() - t_dec) * 1000

        # Update mask cache
        for prompt, mask_u8 in new_masks.items():
            self._mask_cache[prompt] = (mask_u8, now, self._frame_count)

        self._decoder_skipped = (len(new_masks) == 0)

        # --- Step 7: Build segments from cache ---
        segments = self._build_segments(h, w, now)

        total_ms = (time.perf_counter() - t0) * 1000
        vis_tag = "SKIP" if not need_vision else f"{vis_ms:.0f}"
        dec_tag = "SKIP" if self._decoder_skipped else f"{dec_ms:.0f}"
        if self._frame_count <= 20 or self._frame_count % 10 == 0:
            n_decoded = len(new_masks)
            _dbg(f"SAM3 #{self._frame_count}: {total_ms:.0f}ms "
                 f"(pre={pre_ms:.0f} vis={vis_tag} dec={dec_tag}) "
                 f"{n_decoded}p {len(segments)}s "
                 f"vskip={self._vision_skip_count}")
        if self._frame_count % 30 == 0 or self._frame_count <= 3:
            backend = "TRT-pipe" if (self._use_trt_vision and self._use_trt_decoder) else \
                      "TRT-dec" if self._use_trt_decoder else \
                      ("TRT" if self._use_trt else ("batched" if self._batch_supported else "seq"))
            logger.info(
                f"SAM3 #{self._frame_count}: {total_ms:.0f}ms "
                f"(pre={pre_ms:.0f}, vis={vis_tag}, dec={dec_ms:.0f}) | "
                f"{len(segments)} segs | {backend} | vskip={self._vision_skip_count}"
            )

        return segments

    # ------------------------------------------------------------------
    # Decoder: batched or sequential
    # ------------------------------------------------------------------

    def _run_decoder(self, vision_embeds, prompts: list[str], h: int, w: int) -> dict[str, np.ndarray | None]:
        """Run decoder for all prompts. Tries TRT, then batched PyTorch, then sequential."""
        if self._use_trt_decoder:
            return self._run_decoder_trt_native(vision_embeds, prompts, h, w)

        if self._use_trt:
            return self._run_decoder_trt(vision_embeds, prompts, h, w)

        if self._batch_supported and len(prompts) > 1:
            try:
                return self._run_decoder_batched(vision_embeds, prompts, h, w)
            except Exception as e:
                logger.warning(f"Batched decoder failed, disabling: {e}")
                self._batch_supported = False

        return self._run_decoder_sequential(vision_embeds, prompts, h, w)

    def _run_decoder_trt_native(self, vision_embeds, prompts: list[str], h: int, w: int) -> dict[str, np.ndarray | None]:
        """Run decoder using native TRT engine. Supports top-k and full engines."""
        if self._trt_topk:
            return self._run_decoder_trt_topk(vision_embeds, prompts, h, w)
        return self._run_decoder_trt_full(vision_embeds, prompts, h, w)

    def _get_fpn_fp32(self, vision_embeds):
        """Get FP32 FPN features, using cache if available."""
        if self._cached_fpn_fp32 is not None:
            return self._cached_fpn_fp32

        fpn_hs = vision_embeds.fpn_hidden_states
        fpn_pe = vision_embeds.fpn_position_encoding
        # TRT vision outputs are already FP32; PyTorch outputs are FP16
        result = (
            fpn_hs[0].float().contiguous() if fpn_hs[0].dtype != self._torch.float32 else fpn_hs[0].contiguous(),
            fpn_hs[1].float().contiguous() if fpn_hs[1].dtype != self._torch.float32 else fpn_hs[1].contiguous(),
            fpn_hs[2].float().contiguous() if fpn_hs[2].dtype != self._torch.float32 else fpn_hs[2].contiguous(),
            fpn_pe[2].float().contiguous() if fpn_pe[2].dtype != self._torch.float32 else fpn_pe[2].contiguous(),
        )
        self._cached_fpn_fp32 = result
        return result

    def _run_decoder_trt_topk(self, vision_embeds, prompts: list[str], h: int, w: int) -> dict[str, np.ndarray | None]:
        """Run top-k TRT decoder: sequential B=1 per prompt, optimized overhead.

        Optimizations over naive per-prompt loop:
        - Input shapes set ONCE (not per-prompt) — saves ~1.8ms
        - FPN addresses set ONCE (not per-prompt) — saves ~1.5ms
        - Single pre-sync before loop (not per-prompt) — saves ~2.5ms
        - Each prompt writes to different persistent buffer slot
        - Mask post-processing batched after all prompts — saves ~10ms
        """
        torch = self._torch
        n = len(prompts)

        # Get cached FP32 FPN features (B=1)
        fpn_hs_0, fpn_hs_1, fpn_hs_2, fpn_pe_2 = self._get_fpn_fp32(vision_embeds)

        ctx = self._trt_context

        # Set input shapes ONCE — all B=1, identical across prompts (from meta)
        for name, shape in self._dec_fpn_shapes.items():
            ctx.set_input_shape(name, shape)

        # Set FPN addresses ONCE — same vision features for all prompts
        ctx.set_tensor_address("fpn_hs_0", fpn_hs_0.data_ptr())
        ctx.set_tensor_address("fpn_hs_1", fpn_hs_1.data_ptr())
        ctx.set_tensor_address("fpn_hs_2", fpn_hs_2.data_ptr())
        ctx.set_tensor_address("fpn_pe_2", fpn_pe_2.data_ptr())

        # Decoder reads from the READ buffer; async vision writes to the WRITE buffer.
        # No cross-stream sync needed — different buffers, no data race.
        # Only sync with default stream for _get_fpn_fp32() FP32 conversion.
        self._trt_stream.wait_stream(torch.cuda.current_stream())

        all_scores = []

        for i, p in enumerate(prompts):
            te, am = self._text_embeds_fp32[p]
            self._trt_call_count += 1

            # Per-prompt: only text embeddings + output buffer addresses change
            ctx.set_tensor_address("text_embeds", te.data_ptr())
            ctx.set_tensor_address("attention_mask", am.data_ptr())
            ctx.set_tensor_address("best_mask", self._trt_out_mask_persistent[i:i+1].data_ptr())
            ctx.set_tensor_address("best_score", self._trt_out_score_persistent[i:i+1].data_ptr())

            # Execute on decoder stream (separate from vision stream for pipelining)
            ok = ctx.execute_async_v3(self._trt_stream.cuda_stream)
            self._trt_stream.synchronize()  # Must sync before reading score / changing addresses

            if not ok:
                logger.warning(f"TRT decoder failed for '{p}', falling back to PyTorch")
                self._use_trt_decoder = False
                if self._batch_supported:
                    return self._run_decoder_batched(vision_embeds, prompts, h, w)
                return self._run_decoder_sequential(vision_embeds, prompts, h, w)

            score = self._trt_out_score_persistent[i].item()
            all_scores.append(score)

            # NaN check per prompt — reactive safety net
            if np.isnan(score):
                _dbg(f"!!! NaN for prompt '{p}' at TRT call #{self._trt_call_count}, rebuilding context")
                try:
                    self._trt_context = self._trt_engine.create_execution_context()
                except Exception:
                    self._use_trt_decoder = False
                self._cached_fpn_fp32 = None
                if self._batch_supported:
                    return self._run_decoder_batched(vision_embeds, prompts, h, w)
                return self._run_decoder_sequential(vision_embeds, prompts, h, w)

        # Log scores
        score_info = ", ".join(f"{p}={all_scores[i]:.4f}" for i, p in enumerate(prompts))
        _dbg(f"TRT-topk F#{self._frame_count} scores: [{score_info}] calls={self._trt_call_count}")

        # --- Batch post-process: threshold + resize + CPU transfer ---
        result = {}
        passing = []  # (buffer_index, prompt_name)
        for i, p in enumerate(prompts):
            if all_scores[i] < self._confidence_threshold:
                result[p] = None
            else:
                passing.append((i, p))

        if passing:
            # Gather masks that passed confidence threshold from persistent buffers
            indices = [idx for idx, _ in passing]
            masks_raw = self._trt_out_mask_persistent[indices]  # (K, 1, Hm, Wm)

            # Batch: threshold → resize → transfer (1 kernel + 1 D2H instead of K each)
            masks_bool = (masks_raw[:, 0] > 0.0).to(torch.uint8)  # (K, Hm, Wm)
            masks_resized = torch.nn.functional.interpolate(
                masks_bool.unsqueeze(1).float(), size=(h, w), mode='nearest'
            ).squeeze(1).to(torch.uint8) * 255  # (K, h, w)
            masks_cpu = masks_resized.cpu().numpy()

            for k, (idx, p) in enumerate(passing):
                mask_u8 = masks_cpu[k]
                nz = cv2.countNonZero(mask_u8)
                if nz < self.config.mask_min_area:
                    result[p] = None
                else:
                    result[p] = mask_u8

        return result

    def _run_decoder_trt_full(self, vision_embeds, prompts: list[str], h: int, w: int) -> dict[str, np.ndarray | None]:
        """Run full TRT decoder (200 masks per prompt). Sequential batch=1."""
        torch = self._torch

        fpn_hs_0, fpn_hs_1, fpn_hs_2, fpn_pe_2 = self._get_fpn_fp32(vision_embeds)

        ctx = self._trt_context
        for name, shape in self._dec_fpn_shapes.items():
            ctx.set_input_shape(name, shape)

        ctx.set_tensor_address("fpn_hs_0", fpn_hs_0.data_ptr())
        ctx.set_tensor_address("fpn_hs_1", fpn_hs_1.data_ptr())
        ctx.set_tensor_address("fpn_hs_2", fpn_hs_2.data_ptr())
        ctx.set_tensor_address("fpn_pe_2", fpn_pe_2.data_ptr())

        # Collect results across prompts
        all_masks_list = []
        all_logits_list = []
        all_presence_list = []

        for p in prompts:
            te_f32, am_i64 = self._text_embeds_fp32[p]
            ctx.set_tensor_address("text_embeds", te_f32.data_ptr())
            ctx.set_tensor_address("attention_mask", am_i64.data_ptr())
            ctx.set_tensor_address("pred_masks", self._trt_out_mask.data_ptr())
            ctx.set_tensor_address("pred_logits", self._trt_out_logits.data_ptr())
            ctx.set_tensor_address("presence_logits", self._trt_out_presence.data_ptr())

            ok = ctx.execute_async_v3(self._trt_stream.cuda_stream)
            self._trt_stream.synchronize()

            if not ok:
                logger.warning("TRT decoder execution failed, falling back to PyTorch")
                self._use_trt_decoder = False
                return self._run_decoder_batched(vision_embeds, prompts, h, w)

            all_masks_list.append(self._trt_out_mask.clone())
            all_logits_list.append(self._trt_out_logits.clone())
            all_presence_list.append(self._trt_out_presence.clone())

        class _Out:
            pass
        outputs = _Out()
        outputs.pred_masks = torch.cat(all_masks_list, dim=0)
        outputs.pred_logits = torch.cat(all_logits_list, dim=0)
        outputs.presence_logits = torch.cat(all_presence_list, dim=0)

        return self._extract_masks_gpu(outputs, prompts, h, w)

    def _run_decoder_batched(self, vision_embeds, prompts: list[str], h: int, w: int) -> dict[str, np.ndarray | None]:
        """Run all prompts through the decoder in a single batched forward pass."""
        torch = self._torch
        B = len(prompts)

        vis_batch = self._expand_vision_output(vision_embeds, B)

        # Use pre-computed text_embeds (bypass CLIP) or fall back to input_ids
        if self._text_embeds_ready:
            batch_te = torch.cat([self._text_embeds[p]["text_embeds"] for p in prompts], dim=0)
            batch_attn = torch.cat([self._text_embeds[p]["attention_mask"] for p in prompts], dim=0)
            with torch.no_grad():
                outputs = self._compiled_forward(
                    vision_embeds=vis_batch,
                    text_embeds=batch_te,
                    attention_mask=batch_attn,
                )
        else:
            batch_ids = torch.cat([self._tokenized[p]["input_ids"] for p in prompts], dim=0)
            batch_attn = torch.cat([self._tokenized[p]["attention_mask"] for p in prompts], dim=0)
            with torch.no_grad():
                outputs = self._compiled_forward(
                    vision_embeds=vis_batch,
                    input_ids=batch_ids,
                    attention_mask=batch_attn,
                )

        # GPU-side mask extraction (avoids transferring (B,200,H,W) to CPU)
        return self._extract_masks_gpu(outputs, prompts, h, w)

    def _run_decoder_sequential(self, vision_embeds, prompts: list[str], h: int, w: int) -> dict[str, np.ndarray | None]:
        """Run decoder one prompt at a time (fallback)."""
        torch = self._torch
        result = {}

        for prompt in prompts:
            try:
                # Use pre-computed text_embeds (bypass CLIP) or fall back
                if self._text_embeds_ready and prompt in self._text_embeds:
                    te = self._text_embeds[prompt]
                    with torch.no_grad():
                        outputs = self._compiled_forward(
                            vision_embeds=vision_embeds,
                            text_embeds=te["text_embeds"],
                            attention_mask=te["attention_mask"],
                        )
                else:
                    tokens = self._tokenized.get(prompt)
                    if tokens is None:
                        result[prompt] = None
                        continue
                    with torch.no_grad():
                        outputs = self._compiled_forward(
                            vision_embeds=vision_embeds,
                            input_ids=tokens["input_ids"],
                            attention_mask=tokens["attention_mask"],
                        )

                # GPU-side mask extraction for single prompt
                masks = self._extract_masks_gpu(outputs, [prompt], h, w)
                result[prompt] = masks[prompt]

            except Exception as e:
                logger.warning(f"SAM3 decoder '{prompt}' failed: {e}")
                result[prompt] = None

        return result

    def _run_decoder_trt(self, vision_embeds, prompts: list[str], h: int, w: int) -> dict[str, np.ndarray | None]:
        """Run decoder using TRT engine."""
        torch = self._torch
        B = len(prompts)

        batch_ids = torch.cat([self._tokenized[p]["input_ids"] for p in prompts], dim=0)
        batch_attn = torch.cat([self._tokenized[p]["attention_mask"] for p in prompts], dim=0)

        # TRT batched decoder
        raw_outputs = self._trt.decode_batch(vision_embeds, batch_ids, batch_attn, B)

        # Post-process with HF processor
        outputs_cpu = {}
        for k, v in raw_outputs.items():
            if hasattr(v, 'cpu'):
                outputs_cpu[k] = v.cpu().float()
            else:
                outputs_cpu[k] = v

        target_sizes = [(h, w)] * B
        try:
            results_list = self._processor.post_process_instance_segmentation(
                outputs_cpu,
                target_sizes=target_sizes,
                threshold=self._confidence_threshold,
            )
        except Exception:
            results_list = []
            for i in range(B):
                single_out = {}
                for k, v in outputs_cpu.items():
                    if hasattr(v, 'shape') and len(v.shape) > 0 and v.shape[0] == B:
                        single_out[k] = v[i:i+1]
                    else:
                        single_out[k] = v
                res = self._processor.post_process_instance_segmentation(
                    single_out,
                    target_sizes=[(h, w)],
                    threshold=self._confidence_threshold,
                )
                results_list.append(res[0])

        result = {}
        for i, prompt in enumerate(prompts):
            result[prompt] = self._extract_mask(results_list[i], h, w)

        return result

    # ------------------------------------------------------------------
    # Vision output cloning (prevent CUDA graph overwrites)
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_vision_output(vision_output):
        """Deep-clone a Sam3VisionEncoderOutput to prevent CUDA graph overwrites."""
        import torch
        if isinstance(vision_output, torch.Tensor):
            return vision_output.clone()
        try:
            fields = {}
            for key in vision_output.keys():
                val = vision_output[key]
                if isinstance(val, torch.Tensor):
                    fields[key] = val.clone()
                elif isinstance(val, (list, tuple)):
                    fields[key] = type(val)(
                        v.clone() if isinstance(v, torch.Tensor) else v for v in val
                    )
                else:
                    fields[key] = val
            return type(vision_output)(**fields)
        except Exception as e:
            logger.warning(f"    Vision output clone via type() failed ({type(vision_output).__name__}): {e}, using manual clone")
            _dbg(f"Clone fallback: type={type(vision_output).__name__} err={e}")
            # Manual clone: create a simple namespace with cloned tensors
            # Use getattr() instead of vision_output[key] because mock objects
            # from _run_vision_trt don't support subscript access
            class _VisionClone:
                pass
            clone = _VisionClone()
            for key in vision_output.keys():
                val = getattr(vision_output, key)
                if isinstance(val, torch.Tensor):
                    setattr(clone, key, val.clone())
                elif isinstance(val, (list, tuple)):
                    setattr(clone, key, type(val)(
                        v.clone() if isinstance(v, torch.Tensor) else v for v in val
                    ))
                else:
                    setattr(clone, key, val)
            # Provide keys() for expand/fpn helpers
            stored_keys = list(vision_output.keys())
            clone.keys = lambda: stored_keys
            clone.__getitem__ = lambda k: getattr(clone, k)
            return clone

    # ------------------------------------------------------------------
    # Vision output expansion (for batched decoding)
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_vision_output(vision_output, batch_size):
        """Expand all tensors in a Sam3VisionEncoderOutput to batch_size.

        The decoder expects the full output object (with .fpn_hidden_states etc.),
        not a raw tensor. For batching, we expand each tensor inside it.
        """
        import torch

        # Already a raw tensor — simple expand
        if isinstance(vision_output, torch.Tensor):
            return vision_output.expand(batch_size, *vision_output.shape[1:])

        # Dataclass / ModelOutput — expand each field
        def _expand_val(val):
            if isinstance(val, torch.Tensor):
                return val.expand(batch_size, *val.shape[1:])
            elif isinstance(val, (list, tuple)):
                expanded = [_expand_val(v) for v in val]
                return type(val)(expanded)
            return val

        # Reconstruct as same type with expanded fields
        try:
            fields = {}
            for key in vision_output.keys():
                fields[key] = _expand_val(vision_output[key])
            return type(vision_output)(**fields)
        except Exception:
            # Fallback: try dict-style construction
            fields = {}
            for key in dir(vision_output):
                if key.startswith('_'):
                    continue
                val = getattr(vision_output, key, None)
                if val is not None and (isinstance(val, torch.Tensor) or isinstance(val, (list, tuple))):
                    fields[key] = _expand_val(val)
            return type(vision_output)(**fields)

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    def _extract_masks_gpu(self, outputs, prompts: list[str], h: int, w: int) -> dict[str, np.ndarray | None]:
        """Extract best mask per prompt entirely on GPU. Avoids HF postprocess + massive CPU transfer.

        Instead of transferring (B, 200, 1008, 1008) float to CPU, we:
        1. Score queries on GPU: final_score = pred_logits.sigmoid() * presence_logits.sigmoid()
        2. Pick best query per batch element (argmax)
        3. Threshold that single mask on GPU
        4. Resize to (h, w) on GPU
        5. Transfer only (B, h, w) uint8 to CPU
        """
        torch = self._torch
        B = len(prompts)

        pred_masks = outputs.pred_masks      # (B, 200, Hm, Wm)
        pred_logits = outputs.pred_logits    # (B, 200)
        presence = getattr(outputs, 'presence_logits', None)  # (B, 1) or None

        if pred_masks is None or pred_logits is None:
            return {p: None for p in prompts}

        # Score each query: logit * presence
        scores = pred_logits.sigmoid()  # (B, 200)
        if presence is not None:
            scores = scores * presence.sigmoid()  # broadcast (B, 1)

        # Best query per batch element
        best_idx = scores.argmax(dim=1)       # (B,)
        best_score = scores.gather(1, best_idx.unsqueeze(1)).squeeze(1)  # (B,)

        result = {}
        for i, prompt in enumerate(prompts):
            if best_score[i].item() < self._confidence_threshold:
                result[prompt] = None
                continue

            # Extract single mask: (Hm, Wm)
            mask_logits = pred_masks[i, best_idx[i]]
            mask_bool = (mask_logits > 0.0)  # sigmoid threshold at 0.5 = logit > 0

            # Resize to target (h, w) using nearest (fast)
            mask_u8_gpu = mask_bool.to(torch.uint8).unsqueeze(0).unsqueeze(0)  # (1,1,Hm,Wm)
            mask_resized = torch.nn.functional.interpolate(
                mask_u8_gpu.float(), size=(h, w), mode='nearest'
            ).squeeze().to(torch.uint8) * 255  # (h, w)

            mask_u8 = mask_resized.cpu().numpy()
            if cv2.countNonZero(mask_u8) < self.config.mask_min_area:
                result[prompt] = None
            else:
                result[prompt] = mask_u8

        return result

    @staticmethod
    def _outputs_to_cpu(outputs):
        """Move model outputs to CPU float32, preserving ModelOutput type for HF processor."""
        cpu_fields = {}
        for k in outputs.keys():
            v = outputs[k]
            if hasattr(v, 'cpu'):
                cpu_fields[k] = v.cpu().float()
            else:
                cpu_fields[k] = v
        try:
            return type(outputs)(**cpu_fields)
        except Exception:
            return cpu_fields

    def _extract_mask(self, results: dict, h: int, w: int) -> np.ndarray | None:
        """Extract and merge instance masks from post-processing results (CPU fallback)."""
        torch = self._torch
        masks = results.get("masks")
        scores = results.get("scores")

        if masks is None or scores is None or len(scores) == 0:
            return None

        merged = torch.zeros((h, w), dtype=torch.uint8)
        for i in range(len(scores)):
            merged = torch.maximum(merged, masks[i].to(torch.uint8) * 255)

        mask_u8 = merged.numpy()
        if cv2.countNonZero(mask_u8) < self.config.mask_min_area:
            return None

        return mask_u8

    def _build_segments(self, h: int, w: int, now: float) -> list[SegmentData]:
        """Build output segments from mask cache (same logic as before)."""
        segments: list[SegmentData] = []
        person_mask = None

        # Debug: log cache state on first frames and periodically
        if self._frame_count <= 20 or self._frame_count % 100 == 0:
            cache_info = {p: (m is not None, f"age={now-ts:.1f}s", f"gen={g}")
                         for p, (m, ts, g) in self._mask_cache.items()}
            _dbg(f"_build_segments F#{self._frame_count}: cache={cache_info}")

        # Person first (priority)
        if "person" in self._mask_cache:
            pmask, pts, _gen = self._mask_cache["person"]
            if pmask is not None and now - pts <= self._cache_ttl:
                person_mask = pmask
                seg = self._mask_to_segment(pmask, "person", h, w)
                if seg is not None:
                    segments.append(seg)

        self.last_person_mask = person_mask

        # Used pixels from person
        used_pixels = np.zeros((h, w), dtype=bool)
        if person_mask is not None:
            used_pixels |= (person_mask > 127)

        # Other categories from cache
        expired = []
        for prompt, (mask_u8, ts, _gen) in self._mask_cache.items():
            if prompt == "person":
                continue
            if now - ts > self._cache_ttl:
                expired.append(prompt)
                continue
            if mask_u8 is None:
                continue

            clean = mask_u8.copy()
            clean[used_pixels] = 0
            area = cv2.countNonZero(clean)
            if area < self.config.mask_min_area:
                continue

            seg = self._mask_to_segment(clean, prompt, h, w)
            if seg is not None:
                segments.append(seg)
                used_pixels |= (clean > 127)

        for k in expired:
            del self._mask_cache[k]

        return segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_segment(mask_u8: np.ndarray, label: str, h: int, w: int) -> SegmentData | None:
        """Convert a uint8 mask to SegmentData."""
        area = cv2.countNonZero(mask_u8)
        if area < 200:
            return None

        M = cv2.moments(mask_u8)
        if M["m00"] <= 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, bw, bh = cv2.boundingRect(mask_u8)

        seg = SegmentData(mask=mask_u8, label=label, confidence=0.8)
        seg.center_x = cx / w
        seg.center_y = cy / h
        seg.mask_width = w
        seg.mask_height = h
        seg.bbox = (x / w, y / h, (x + bw) / w, (y + bh) / h)
        seg.asset_class = _ASSET_CLASS_MAP.get(label, "unknown")

        return seg

    @property
    def decoder_skipped(self) -> bool:
        """True if last segment_frame() returned cached masks (no new decode)."""
        return self._decoder_skipped

    def shutdown(self):
        """Release all resources. Waits for in-flight async vision before cleanup."""
        # Wait for any in-flight async vision to finish before freeing buffers
        if self._pipeline_state == "VISION_RUNNING" and self._vis_done_event is not None:
            try:
                self._vis_done_event.synchronize()
            except Exception:
                pass
            self._pipeline_state = "IDLE"

        self._model = None
        self._processor = None
        self._compiled_forward = None
        self._tokenized.clear()
        self._mask_cache.clear()
        self._trt_context = None
        self._trt_engine = None
        self._trt_vis_context = None
        self._trt_vis_engine = None
        self._vis_buffers = [None, None]
        self._vis_done_event = None
        self._cached_vision_embeds = None
        self._cached_fpn_fp32 = None
        if self._trt is not None:
            self._trt.shutdown()
            self._trt = None
        self._initialized = False
        logger.info("SAM3Segmenter shutdown")
