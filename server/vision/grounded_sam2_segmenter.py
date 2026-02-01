"""Grounded SAM2 segmentation engine.

GroundingDINO detects objects by text prompt → bounding boxes + labels.
SAM2 converts those boxes into precise masks.
Combined: text-prompted instance segmentation without API keys.

Same interface as FastSAMSegmenter / SAM3Segmenter:
    initialize(), segment_frame(frame_bgr) -> list[SegmentData], shutdown(), .last_person_mask
"""

import time
import logging
import cv2
import numpy as np
import torch
from PIL import Image

from server.vision.segment_data import SegmentData

logger = logging.getLogger(__name__)

# All categories in a single GroundingDINO query (period-separated)
DETECT_PROMPT = "person. chair. table. desk. couch. monitor. laptop. lamp. wall. floor. door. window."

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


class GroundedSAM2Segmenter:
    """GroundingDINO + SAM2 segmenter.

    GroundingDINO handles all categories in one forward pass (fast).
    SAM2 generates precise masks from detected boxes.
    """

    def __init__(self, config):
        self.config = config
        self._gdino_model = None
        self._gdino_proc = None
        self._sam2_model = None
        self._sam2_proc = None
        self._device = None
        self._initialized = False

        # Public: expose person_mask for pipeline
        self.last_person_mask: np.ndarray | None = None

        self._frame_count = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Load GroundingDINO + SAM2 from HuggingFace."""
        t0 = time.perf_counter()
        self._device = self.config.device

        # GroundingDINO
        from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

        gdino_name = self.config.gdino_model
        logger.info(f"    Loading GroundingDINO ({gdino_name}) on {self._device}")
        self._gdino_proc = GroundingDinoProcessor.from_pretrained(gdino_name)
        self._gdino_model = GroundingDinoForObjectDetection.from_pretrained(gdino_name)
        self._gdino_model.to(self._device).eval()
        t_gdino = time.perf_counter()
        logger.info(f"    GroundingDINO loaded: {(t_gdino - t0)*1000:.0f}ms")

        # SAM2
        from transformers import Sam2Model, Sam2Processor

        sam2_name = self.config.sam2_model
        logger.info(f"    Loading SAM2 ({sam2_name}) on {self._device}")
        self._sam2_proc = Sam2Processor.from_pretrained(sam2_name)
        self._sam2_model = Sam2Model.from_pretrained(sam2_name)
        self._sam2_model.to(self._device).eval()
        t_sam2 = time.perf_counter()
        logger.info(f"    SAM2 loaded: {(t_sam2 - t_gdino)*1000:.0f}ms")

        # FP16 for speed
        if self._device == "cuda":
            self._gdino_model = self._gdino_model.half()
            self._sam2_model = self._sam2_model.half()
            logger.info("    Using FP16 for both models")

        # Warmup
        try:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self._detect_and_segment(dummy)
            logger.info(f"    Warm-up: {(time.perf_counter() - t_sam2)*1000:.0f}ms")
        except Exception as e:
            logger.warning(f"    Warm-up failed: {e}")

        self._initialized = True
        logger.info(f"    GroundedSAM2Segmenter ready ({(time.perf_counter() - t0)*1000:.0f}ms total)")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment_frame(self, frame_bgr: np.ndarray) -> list[SegmentData]:
        """Run GroundingDINO + SAM2 on one BGR frame."""
        if not self._initialized:
            self.initialize()

        t0 = time.perf_counter()
        self._frame_count += 1
        h, w = frame_bgr.shape[:2]

        detections = self._detect_and_segment(frame_bgr)

        # Build segments, person first for overlap resolution
        segments: list[SegmentData] = []
        person_mask = None
        used_pixels = np.zeros((h, w), dtype=bool)

        # Sort: person first, then by area descending
        detections.sort(key=lambda d: (0 if d[1] == "person" else 1, -d[2]))

        for mask_u8, label, area, box in detections:
            if label == "person":
                if person_mask is None:
                    person_mask = mask_u8
                # Merge multiple person detections
                elif person_mask is not None:
                    person_mask = np.maximum(person_mask, mask_u8)

            # Subtract already-claimed pixels
            clean = mask_u8.copy()
            clean[used_pixels] = 0
            clean_area = cv2.countNonZero(clean)
            if clean_area < self.config.mask_min_area:
                continue

            seg = self._mask_to_segment(clean, label, h, w)
            if seg is not None:
                segments.append(seg)
                used_pixels |= (clean > 127)

        self.last_person_mask = person_mask

        total_ms = (time.perf_counter() - t0) * 1000
        if self._frame_count % 30 == 0 or self._frame_count <= 3:
            logger.info(
                f"GroundedSAM2 #{self._frame_count}: {total_ms:.1f}ms | "
                f"{len(segments)} segs"
            )

        return segments

    # ------------------------------------------------------------------
    # Detection + Segmentation
    # ------------------------------------------------------------------

    def _detect_and_segment(self, frame_bgr: np.ndarray) -> list:
        """Run GroundingDINO → SAM2 pipeline. Returns [(mask_u8, label, area, box), ...]."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # --- GroundingDINO: detect all categories in one pass ---
        gdino_inputs = self._gdino_proc(
            images=pil_image, text=DETECT_PROMPT, return_tensors="pt"
        )
        gdino_inputs = {
            k: v.to(self._device) if hasattr(v, 'to') else v
            for k, v in gdino_inputs.items()
        }
        if self._device == "cuda":
            gdino_inputs = {
                k: v.half() if hasattr(v, 'half') and v.is_floating_point() else v
                for k, v in gdino_inputs.items()
            }

        with torch.no_grad():
            gdino_out = self._gdino_model(**gdino_inputs)

        results = self._gdino_proc.post_process_grounded_object_detection(
            gdino_out,
            threshold=self.config.gdino_box_threshold,
            text_threshold=self.config.gdino_text_threshold,
            input_ids=gdino_inputs.get("input_ids"),
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"]  # tensor (N, 4) in xyxy format
        labels = results["labels"]  # list of strings
        scores = results["scores"]  # tensor (N,)

        if len(boxes) == 0:
            return []

        # --- SAM2: generate masks from detected boxes ---
        detections = []
        boxes_list = boxes.cpu().numpy()

        for i in range(len(boxes_list)):
            box = boxes_list[i].tolist()  # [x1, y1, x2, y2]
            label = labels[i].strip().lower()

            try:
                sam_inputs = self._sam2_proc(
                    images=pil_image,
                    input_boxes=[[[box]]],
                    return_tensors="pt",
                )
                sam_inputs = {
                    k: v.to(self._device) if hasattr(v, 'to') else v
                    for k, v in sam_inputs.items()
                }
                if self._device == "cuda":
                    sam_inputs = {
                        k: v.half() if hasattr(v, 'half') and v.is_floating_point() else v
                        for k, v in sam_inputs.items()
                    }

                with torch.no_grad():
                    sam_out = self._sam2_model(**sam_inputs)

                # Post-process mask
                masks = self._sam2_proc.post_process_masks(
                    sam_out.pred_masks,
                    sam_inputs["original_sizes"],
                    sam_inputs["reshaped_input_sizes"],
                )
                mask_np = masks[0][0].cpu().float().numpy()  # (num_masks, H, W)

                # Pick highest-scoring mask
                if hasattr(sam_out, 'iou_scores') and sam_out.iou_scores is not None:
                    iou_scores = sam_out.iou_scores.cpu().float().numpy().flatten()
                    best_idx = iou_scores.argmax()
                else:
                    best_idx = 0

                mask = mask_np[best_idx] if len(mask_np.shape) == 3 else mask_np
                mask_u8 = ((mask > 0.5).astype(np.uint8)) * 255

                if mask_u8.shape != (h, w):
                    mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

                area = cv2.countNonZero(mask_u8)
                if area >= self.config.mask_min_area:
                    detections.append((mask_u8, label, area, box))

            except Exception as e:
                logger.warning(f"SAM2 mask for '{label}' failed: {e}")
                continue

        return detections

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
        self._gdino_model = None
        self._gdino_proc = None
        self._sam2_model = None
        self._sam2_proc = None
        self._initialized = False
        logger.info("GroundedSAM2Segmenter shutdown")
