"""SAM3 text-prompted segmentation engine.

SAM3 handles both segmentation AND labeling via text prompts —
no need for a separate Gemini labeler.

Optimizations applied:
1. Vision embedding caching — ViT runs once per frame (350ms), decoder reuses it (83ms/prompt)
2. Pre-tokenized prompts — processor overhead eliminated from hot path
3. FP16 inference — halved memory bandwidth
4. No round-robin needed — cached ViT makes per-prompt decoder cheap enough to run many/frame

Performance: ~690ms/frame (1.4 FPS) with 4 categories on RTX 3060 Laptop.
Baseline without caching was ~1920ms (0.5 FPS).

Same interface as FastSAMSegmenter:
    initialize(), segment_frame(frame_bgr) -> list[SegmentData], shutdown(), .last_person_mask
"""

import time
import logging
import cv2
import numpy as np

from server.vision.segment_data import SegmentData

logger = logging.getLogger(__name__)

# All categories to detect. "person" is always queried; others rotate.
ALL_PROMPTS = [
    "person", "chair", "table", "desk", "couch", "monitor",
    "laptop", "lamp", "wall", "floor", "door", "window",
]

# Map labels -> asset classes
_ASSET_CLASS_MAP = {
    "person": "person",
    "chair": "furniture", "table": "furniture", "desk": "furniture",
    "couch": "furniture",
    "monitor": "electronics", "laptop": "electronics",
    "lamp": "lighting", "light": "lighting",
    "wall": "structure", "floor": "structure",
    "ceiling": "structure", "door": "structure", "window": "structure",
}


class SAM3Segmenter:
    """Text-prompted SAM3 segmenter with vision embedding caching.

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

        # Pre-tokenized prompts: prompt -> {input_ids, attention_mask} on device
        self._tokenized: dict[str, dict] = {}

        # Round-robin for non-person prompts
        self._rotate_index = 0
        self._prompts_per_frame = config.sam3_prompts_per_frame
        self._confidence_threshold = config.sam3_confidence_threshold

        # Mask cache: prompt -> (mask_u8, timestamp) for non-queried prompts
        self._mask_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._cache_ttl = config.sam3_cache_ttl

        self._frame_count = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Load SAM3 model from HuggingFace and pre-tokenize all prompts."""
        import torch
        self._torch = torch

        t0 = time.perf_counter()

        from transformers import Sam3Model, Sam3Processor

        t_import = time.perf_counter()
        logger.info(f"    import transformers: {(t_import - t0)*1000:.0f}ms")

        model_name = self.config.sam3_model
        self._device = self.config.device

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

        # Pre-tokenize all prompts (eliminates processor overhead in hot path)
        from PIL import Image
        dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        for prompt in ALL_PROMPTS:
            inputs = self._processor(images=dummy_img, text=prompt, return_tensors="pt")
            self._tokenized[prompt] = {
                "input_ids": inputs["input_ids"].to(self._device),
                "attention_mask": inputs["attention_mask"].to(self._device),
            }
        logger.info(f"    Pre-tokenized {len(ALL_PROMPTS)} prompts")

        # Warmup: vision encoder + one decoder pass
        try:
            dummy_pv = inputs["pixel_values"].to(self._device)
            if self._device == "cuda":
                dummy_pv = dummy_pv.half()
            with torch.no_grad():
                vis = self._model.get_vision_features(pixel_values=dummy_pv)
                self._model(
                    vision_embeds=vis,
                    input_ids=self._tokenized["person"]["input_ids"],
                    attention_mask=self._tokenized["person"]["attention_mask"],
                )
            logger.info(f"    SAM3 warm-up: {(time.perf_counter() - t_model)*1000:.0f}ms")
        except Exception as e:
            logger.warning(f"    SAM3 warm-up failed: {e}")

        self._initialized = True
        logger.info(f"    SAM3Segmenter ready ({(time.perf_counter() - t0)*1000:.0f}ms total)")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment_frame(self, frame_bgr: np.ndarray) -> list[SegmentData]:
        """Run SAM3 on one BGR frame.

        1. Vision encoder runs ONCE (350ms) — cached for all prompts.
        2. Decoder runs per prompt with cached vision (83ms each).
        3. "person" always queried + N rotating categories.
        """
        if not self._initialized:
            self.initialize()

        torch = self._torch
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        now = time.time()
        self._frame_count += 1

        # Preprocess image — run through processor for pixel_values only
        from PIL import Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        img_inputs = self._processor(images=pil_image, text="person", return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(self._device)
        if self._device == "cuda":
            pixel_values = pixel_values.half()

        # --- Step 1: Vision encoder ONCE per frame ---
        t_vis = time.perf_counter()
        with torch.no_grad():
            vision_embeds = self._model.get_vision_features(pixel_values=pixel_values)
        vis_ms = (time.perf_counter() - t_vis) * 1000

        # --- Step 2: Decoder per prompt (using cached vision) ---
        # Build prompt list: "person" always + rotating others
        object_prompts = [p for p in ALL_PROMPTS if p != "person"]
        prompts_this_frame = ["person"]
        for i in range(self._prompts_per_frame):
            idx = (self._rotate_index + i) % len(object_prompts)
            prompts_this_frame.append(object_prompts[idx])
        self._rotate_index = (self._rotate_index + self._prompts_per_frame) % len(object_prompts)

        t_dec = time.perf_counter()
        for prompt in prompts_this_frame:
            mask_u8 = self._run_decoder(vision_embeds, prompt, h, w)
            if mask_u8 is not None:
                self._mask_cache[prompt] = (mask_u8, now)
            else:
                self._mask_cache.pop(prompt, None)
        dec_ms = (time.perf_counter() - t_dec) * 1000

        # --- Step 3: Build segments from cache ---
        segments: list[SegmentData] = []
        person_mask = None

        # Person first (priority)
        if "person" in self._mask_cache:
            pmask, pts = self._mask_cache["person"]
            if now - pts <= self._cache_ttl:
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
        for prompt, (mask_u8, ts) in self._mask_cache.items():
            if prompt == "person":
                continue
            if now - ts > self._cache_ttl:
                expired.append(prompt)
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

        total_ms = (time.perf_counter() - t0) * 1000
        if self._frame_count % 30 == 0 or self._frame_count <= 3:
            logger.info(
                f"SAM3 #{self._frame_count}: {total_ms:.0f}ms "
                f"(vis={vis_ms:.0f}ms, dec={dec_ms:.0f}ms) | "
                f"{len(segments)} segs | {prompts_this_frame}"
            )

        return segments

    # ------------------------------------------------------------------
    # Decoder pass (uses cached vision embeddings)
    # ------------------------------------------------------------------

    def _run_decoder(self, vision_embeds, prompt: str, h: int, w: int) -> np.ndarray | None:
        """Run SAM3 decoder with cached vision embeddings for one text prompt."""
        torch = self._torch

        tokens = self._tokenized.get(prompt)
        if tokens is None:
            return None

        try:
            with torch.no_grad():
                outputs = self._model(
                    vision_embeds=vision_embeds,
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )

            # Move to CPU for post-processing
            for k in list(outputs.keys()):
                v = outputs[k]
                if hasattr(v, 'cpu'):
                    outputs[k] = v.cpu().float()

            results = self._processor.post_process_instance_segmentation(
                outputs,
                target_sizes=[(h, w)],
                threshold=self._confidence_threshold,
            )[0]

            masks = results["masks"]
            scores = results["scores"]

            if len(scores) == 0:
                return None

            # Merge all instances into one mask for this prompt
            merged = torch.zeros((h, w), dtype=torch.uint8)
            for i in range(len(scores)):
                merged = torch.maximum(merged, masks[i].to(torch.uint8) * 255)

            mask_u8 = merged.numpy()

            if cv2.countNonZero(mask_u8) < self.config.mask_min_area:
                return None

            return mask_u8

        except Exception as e:
            logger.warning(f"SAM3 decoder '{prompt}' failed: {e}")
            return None

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

    def shutdown(self):
        """Release all resources."""
        self._model = None
        self._processor = None
        self._tokenized.clear()
        self._mask_cache.clear()
        self._initialized = False
        logger.info("SAM3Segmenter shutdown")
