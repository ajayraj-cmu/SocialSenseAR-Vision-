"""SAM3 text-prompted segmentation engine.

SAM3 handles both segmentation AND labeling via text prompts —
no need for a separate Gemini labeler.

Round-robin prompt scheduling (can't run all prompts every frame):
- Every frame: "person" (always queried)
- Rotating: N additional prompts per frame from object categories
- Cache: masks from non-queried categories persist up to TTL

Same interface as FastSAMSegmenter:
    initialize(), segment_frame(frame_bgr) -> list[SegmentData], shutdown(), .last_person_mask
"""

import time
import logging
import cv2
import numpy as np

from server.vision.segment_data import SegmentData

logger = logging.getLogger(__name__)

# Object categories to rotate through (person is always queried)
OBJECT_PROMPTS = [
    "chair", "table", "desk", "couch", "monitor",
    "laptop", "lamp", "light", "wall", "floor",
    "ceiling", "door", "window",
]

# Map labels -> asset classes (matches SemanticLabeler convention)
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
    """Text-prompted SAM3 segmenter.

    Same interface as FastSAMSegmenter so the orchestrator can swap them.
    """

    def __init__(self, config):
        self.config = config
        self._model = None
        self._processor = None
        self._device = None
        self._initialized = False

        # Public: expose person_mask for pipeline
        self.last_person_mask: np.ndarray | None = None

        # Round-robin state
        self._rotate_index = 0
        self._prompts_per_frame = config.sam3_prompts_per_frame

        # Mask cache: prompt -> (mask_u8, timestamp)
        self._mask_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._cache_ttl = config.sam3_cache_ttl
        self._confidence_threshold = config.sam3_confidence_threshold

        self._frame_count = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Load SAM3 model from HuggingFace."""
        t0 = time.perf_counter()

        import torch  # noqa: F401
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

        # FP16 for speed on CUDA (near-zero quality loss)
        if self._device == "cuda":
            self._model = self._model.half()
            logger.info("    SAM3 using FP16")

        # Warmup
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._run_prompt(dummy, "person")
            logger.info(f"    SAM3 warm-up: {(time.perf_counter() - t_model)*1000:.0f}ms")
        except Exception as e:
            logger.warning(f"    SAM3 warm-up failed: {e}")

        self._initialized = True
        logger.info(f"    SAM3Segmenter ready ({(time.perf_counter() - t0)*1000:.0f}ms total)")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment_frame(self, frame_bgr: np.ndarray) -> list[SegmentData]:
        """Run SAM3 on one BGR frame with round-robin prompt scheduling.

        Returns pre-labeled SegmentData (labels come from text prompts).
        """
        if not self._initialized:
            self.initialize()

        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        now = time.time()
        self._frame_count += 1

        # Convert BGR -> RGB for model
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Always query "person"
        prompts_this_frame = ["person"]

        # Add rotating prompts
        for i in range(self._prompts_per_frame):
            idx = (self._rotate_index + i) % len(OBJECT_PROMPTS)
            prompts_this_frame.append(OBJECT_PROMPTS[idx])
        self._rotate_index = (self._rotate_index + self._prompts_per_frame) % len(OBJECT_PROMPTS)

        # Run SAM3 for each prompt this frame
        for prompt in prompts_this_frame:
            mask_u8 = self._run_prompt(frame_rgb, prompt)
            if mask_u8 is not None:
                self._mask_cache[prompt] = (mask_u8, now)
            else:
                # No detection — clear cache for this prompt
                self._mask_cache.pop(prompt, None)

        # Build segments from cache (including non-queried prompts within TTL)
        segments: list[SegmentData] = []
        person_mask = None

        # Person first (priority for overlap resolution)
        if "person" in self._mask_cache:
            pmask, pts = self._mask_cache["person"]
            if now - pts <= self._cache_ttl:
                person_mask = pmask
                seg = self._mask_to_segment(pmask, "person", h, w)
                if seg is not None:
                    segments.append(seg)

        self.last_person_mask = person_mask

        # Used pixels from person mask (for overlap resolution)
        used_pixels = np.zeros((h, w), dtype=bool)
        if person_mask is not None:
            used_pixels |= (person_mask > 127)

        # Other categories (subtract person pixels)
        expired = []
        for prompt, (mask_u8, ts) in self._mask_cache.items():
            if prompt == "person":
                continue
            if now - ts > self._cache_ttl:
                expired.append(prompt)
                continue

            # Subtract claimed pixels
            clean = mask_u8.copy()
            clean[used_pixels] = 0

            area = cv2.countNonZero(clean)
            if area < self.config.mask_min_area:
                continue

            seg = self._mask_to_segment(clean, prompt, h, w)
            if seg is not None:
                segments.append(seg)
                used_pixels |= (clean > 127)

        # Clean expired entries
        for k in expired:
            del self._mask_cache[k]

        total_ms = (time.perf_counter() - t0) * 1000
        if self._frame_count % 30 == 0 or self._frame_count <= 3:
            logger.info(
                f"SAM3 frame #{self._frame_count}: {total_ms:.1f}ms | "
                f"{len(segments)} segs | prompts={prompts_this_frame}"
            )

        return segments

    # ------------------------------------------------------------------
    # SAM3 inference
    # ------------------------------------------------------------------

    def _run_prompt(self, frame_rgb: np.ndarray, prompt: str) -> np.ndarray | None:
        """Run SAM3 with a single text prompt. Returns uint8 mask [0,255] or None."""
        import torch
        from PIL import Image

        try:
            pil_image = Image.fromarray(frame_rgb)
            inputs = self._processor(
                images=pil_image,
                text=prompt,
                return_tensors="pt",
            )
            # Move to device
            inputs = {k: v.to(self._device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            if self._device == "cuda":
                # FP16 inputs
                inputs = {
                    k: v.half() if hasattr(v, 'half') and v.dtype == torch.float32 else v
                    for k, v in inputs.items()
                }

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process masks
            masks = self._processor.post_process_masks(
                outputs.pred_masks,
                inputs.get("original_sizes", [pil_image.size[::-1]]),
                inputs.get("reshaped_input_sizes", None),
            )

            # Get scores and filter by confidence
            scores = outputs.iou_scores  # (batch, num_masks)
            if scores is not None:
                scores_np = scores.cpu().float().numpy().flatten()
                masks_np = masks[0].cpu().float().numpy()  # (num_masks, H, W)

                # Keep best mask above threshold
                best_idx = scores_np.argmax()
                if scores_np[best_idx] < self._confidence_threshold:
                    return None

                mask = masks_np[best_idx]
            else:
                masks_np = masks[0].cpu().float().numpy()
                if len(masks_np) == 0:
                    return None
                mask = masks_np[0]

            # Convert to uint8
            mask_u8 = ((mask > 0.5).astype(np.uint8)) * 255

            h, w = frame_rgb.shape[:2]
            if mask_u8.shape != (h, w):
                mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

            # Check minimum area
            if cv2.countNonZero(mask_u8) < self.config.mask_min_area:
                return None

            return mask_u8

        except Exception as e:
            logger.warning(f"SAM3 prompt '{prompt}' failed: {e}")
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
        self._mask_cache.clear()
        self._initialized = False
        logger.info("SAM3Segmenter shutdown")
