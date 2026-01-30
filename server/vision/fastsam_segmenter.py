"""FastSAM segmentation engine — orchestrates MediaPipe + FastSAM + refinement + labeling.

Uses modular components:
- mediapipe_detector.py  — person/face/hands/pose detection
- mask_refinement.py     — bilateral + GrabCut mask refinement
- semantic_labeler.py    — position/size heuristic labels + persistence

The segmentation logic is identical to sam_gemini_voice.py _update_masks().
Only the file organization has changed — every threshold, kernel, and code path is preserved.

SAFE TO EDIT: Changes here affect SAM inference and the orchestration flow.
For detection, refinement, or labeling changes, edit the respective module.
"""

import time
import logging
import cv2
import numpy as np

from server.vision.mediapipe_detector import MediaPipeDetector
from server.vision.mask_refinement import refine_mask_edges, fallback_morphology
from server.vision.semantic_labeler import SemanticLabeler

logger = logging.getLogger(__name__)


class SegmentData:
    """One detected segment."""
    __slots__ = (
        "mask", "label", "asset_class", "confidence",
        "bbox", "center_x", "center_y",
        "mask_width", "mask_height", "rle_mask",
        "track_id", "emotion",
    )

    def __init__(self, mask: np.ndarray, label: str = "", confidence: float = 0.0):
        self.mask = mask
        self.label = label
        self.asset_class = ""
        self.confidence = confidence
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.center_x = 0.0
        self.center_y = 0.0
        self.mask_width = 0
        self.mask_height = 0
        self.rle_mask = b""
        self.track_id = ""
        self.emotion = None


class FastSAMSegmenter:
    """Combined MediaPipe + FastSAM segmenter.

    Same logic as sam_gemini_voice.py EnvironmentController, now split across:
    - MediaPipeDetector: selfie/face/hands/pose
    - mask_refinement: bilateral + GrabCut
    - SemanticLabeler: position heuristic + persistence
    - This class: FastSAM inference + orchestration
    """

    def __init__(self, config):
        self.config = config
        self._sam = None
        self._initialized = False

        # Modular components
        self._mp_detector = MediaPipeDetector(config)
        self._labeler = SemanticLabeler()

        # Last result (used as fallback on error only)
        self._cached_segments: list = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Load FastSAM + MediaPipe models."""
        t0 = time.perf_counter()

        from ultralytics import FastSAM
        t_import = time.perf_counter()
        logger.info(f"    import ultralytics: {(t_import - t0)*1000:.0f}ms")

        device = self.config.fastsam_device
        model_path = self.config.fastsam_model

        logger.info(f"    Loading FastSAM ({model_path}) on {device}")
        self._sam = FastSAM(model_path)
        t_load = time.perf_counter()
        logger.info(f"    FastSAM model load: {(t_load - t_import)*1000:.0f}ms")

        # Warm up
        if "cuda" in device:
            try:
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                self._sam(dummy, device=device, imgsz=320, conf=0.5, verbose=False)
                t_warmup = time.perf_counter()
                logger.info(f"    CUDA warm-up: {(t_warmup - t_load)*1000:.0f}ms")
            except Exception as e:
                logger.warning(f"    CUDA warm-up failed: {e}")

        # MediaPipe
        t_mp_start = time.perf_counter()
        self._mp_detector.initialize()
        t_mp = time.perf_counter()
        logger.info(f"    MediaPipe init: {(t_mp - t_mp_start)*1000:.0f}ms")

        self._initialized = True
        logger.info(f"    FastSAMSegmenter ready ({(t_mp - t0)*1000:.0f}ms total)")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment_frame(self, frame_bgr: np.ndarray, sam_conf: float = None) -> list:
        """Run full segmentation on one BGR frame.

        1. MediaPipe body parts (person, face, hands, pose)
        2. FastSAM object masks with person exclusion
        3. Mask refinement (bilateral + GrabCut)
        4. Semantic labeling + area validation + unique labels
        5. Convert to SegmentData list

        Args:
            frame_bgr: BGR uint8 image (H x W x 3).
            sam_conf:  Override SAM confidence.

        Returns:
            List[SegmentData]
        """
        if not self._initialized:
            self.initialize()

        t_total = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ============================================================
        # STEP 1: MediaPipe body parts
        # ============================================================
        t_mp = time.perf_counter()
        body_masks, person_mask = self._mp_detector.detect(frame_bgr, rgb, h, w)
        mp_ms = (time.perf_counter() - t_mp) * 1000

        # ============================================================
        # STEP 2: FastSAM + refinement + labeling
        # ============================================================
        masks_out = self._update_masks(frame_bgr, h, w, body_masks, person_mask, sam_conf)

        # ============================================================
        # STEP 3: Convert to SegmentData
        # ============================================================
        segments: list[SegmentData] = []
        for mask, label, center in masks_out:
            seg = SegmentData(mask=mask, label=label, confidence=0.7)
            seg.center_x = float(center[0]) / w if center[0] > 0 else 0.0
            seg.center_y = float(center[1]) / h if center[1] > 0 else 0.0
            seg.mask_width = w
            seg.mask_height = h
            # Compute bbox
            ys, xs = np.where(mask > 0.5)
            if len(xs) > 0:
                seg.bbox = (
                    int(xs.min()) / w, int(ys.min()) / h,
                    int(xs.max()) / w, int(ys.max()) / h,
                )
            # Assign asset_class from label
            seg.asset_class = SemanticLabeler.label_to_asset_class(label)
            segments.append(seg)

        total_ms = (time.perf_counter() - t_total) * 1000
        self._cached_segments = segments

        if self._mp_detector.frame_count % 10 == 0:
            logger.info(
                f"Frame: mediapipe={mp_ms:.1f}ms total={total_ms:.1f}ms "
                f"| {len(segments)} segs"
            )

        return segments

    # ------------------------------------------------------------------
    # _update_masks — same flow as original, using modular components
    # ------------------------------------------------------------------

    def _update_masks(self, frame, h, w, body_masks, person_mask, sam_conf_override=None):
        """Run FastSAM + refine + label. Same flow as original lines 2273-2347."""
        masks = list(body_masks)  # Start with MediaPipe body parts
        self._labeler.reset_frame()

        # ==========================================
        # SAM masks
        # ==========================================
        t_sam = time.perf_counter()
        used_pixels = np.zeros((h, w), dtype=bool)
        try:
            sam_conf = sam_conf_override if sam_conf_override is not None else self.config.fastsam_conf

            # Adaptive based on frame size — exact same as original line 2285
            target_size = min(512, max(320, min(h, w)))

            sam_results = self._sam(
                frame,
                device=self.config.fastsam_device,
                retina_masks=True,       # exact same as original
                imgsz=target_size,
                conf=sam_conf,
                verbose=False,
            )
            sam_ms = (time.perf_counter() - t_sam) * 1000

            if sam_results and sam_results[0].masks is not None:
                if person_mask is not None:
                    used_pixels |= (person_mask > 0.5)

                t_refine_total = time.perf_counter()
                t_label_total = time.perf_counter()
                refine_ms_acc = 0.0
                label_ms_acc = 0.0

                for mask_data in sam_results[0].masks.data.cpu().numpy():
                    # cv2.resize with DEFAULT interpolation (INTER_LINEAR) — same as original
                    mask = cv2.resize(mask_data.astype(np.float32), (w, h))
                    mask_binary = mask > 0.5

                    # Remove overlap
                    clean_mask = mask_binary & ~used_pixels

                    if np.sum(clean_mask) < 500:
                        continue

                    # ==========================================
                    # MASK REFINEMENT
                    # ==========================================
                    t_ref = time.perf_counter()
                    clean_mask_float = clean_mask.astype(np.float32)
                    refined_mask = refine_mask_edges(frame, clean_mask_float)

                    # Ensure refined mask is valid
                    if refined_mask is not None and np.sum(refined_mask > 0.5) >= 500:
                        clean_mask = refined_mask
                    else:
                        # Fallback — exact same: np.ones kernel, MORPH_CLOSE only
                        clean_mask = fallback_morphology(clean_mask)
                    refine_ms_acc += (time.perf_counter() - t_ref) * 1000

                    used_pixels |= (clean_mask > 0.5)

                    # ==========================================
                    # SEMANTIC LABELING
                    # ==========================================
                    t_lbl = time.perf_counter()

                    # Get label from semantic analysis
                    label, _ = self._labeler.get_label(clean_mask, h, w, frame)

                    # Validate suspicious labels — exact same as original lines 2327-2332
                    mask_area = np.sum(clean_mask > 0.3)
                    area_ratio = mask_area / (h * w)
                    if label in ["person", "backpack", "handbag"] and area_ratio > 0.12:
                        label, _ = self._labeler.get_label(clean_mask, h, w, frame)

                    # Make label unique — same as original
                    label = self._labeler.unique_label(label)
                    label_ms_acc += (time.perf_counter() - t_lbl) * 1000

                    center = _mask_center(clean_mask)
                    masks.append((clean_mask, label, center))

                if self._mp_detector.frame_count % 10 == 0:
                    logger.info(
                        f"  SAM={sam_ms:.1f}ms refine={refine_ms_acc:.1f}ms "
                        f"label={label_ms_acc:.1f}ms "
                        f"| {len(masks) - len(body_masks)} SAM masks"
                    )

        except Exception as e:
            logger.error(f"SAM error: {e}")

        return masks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_geometry(seg: SegmentData, mask_f: np.ndarray, h: int, w: int):
        """Populate bbox, centre, mask dims on a SegmentData."""
        ys, xs = np.where(mask_f > 0.5)
        if len(xs) == 0:
            return
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        seg.bbox = (x_min / w, y_min / h, x_max / w, y_max / h)
        seg.center_x = float(np.mean(xs)) / w
        seg.center_y = float(np.mean(ys)) / h
        seg.mask_width = w
        seg.mask_height = h

    def shutdown(self):
        """Release all resources."""
        self._sam = None
        self._mp_detector.shutdown()
        self._initialized = False
        self._cached_segments.clear()
        logger.info("FastSAMSegmenter shutdown")


def _mask_center(mask: np.ndarray) -> tuple[int, int]:
    """Compute mask centroid. Same as sam_gemini_voice.py line 3299."""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return (0, 0)
    return (int(np.mean(xs)), int(np.mean(ys)))
