"""FastSAM segmentation engine — high-performance pipeline.

Architecture for 40+ FPS:
- FastSAM uses DIRECT TensorRT inference (bypasses ultralytics overhead)
- MediaPipe runs in a SEPARATE PROCESS (eliminates GIL contention entirely)
- No mask refinement (bilateral filter removed — SAM masks are good enough)
- Fast bbox/centroid via cv2 instead of np.where
- Vectorized mask post-processing in uint8

Performance: ~10ms per frame (100+ FPS) on RTX 3060 Laptop with TRT engine.
Falls back to ultralytics if no TRT engine found.

SAFE TO EDIT: Changes here affect SAM inference and the orchestration flow.
"""

import os
import time
import logging
import cv2
import numpy as np

from server.vision.mediapipe_worker import MediaPipeWorker
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
    """High-performance MediaPipe + FastSAM segmenter.

    Uses direct TensorRT inference when engine is available (151 FPS raw).
    MediaPipe runs in a SEPARATE PROCESS — zero GIL contention with SAM.
    Falls back to ultralytics if no TRT engine exists.
    """

    # CLAHE for contrast enhancement in poor lighting (~0.3ms overhead)
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __init__(self, config):
        self.config = config
        self._sam = None  # FastSAMTRT or ultralytics FastSAM
        self._use_direct_trt = False  # True if using FastSAMTRT
        self._initialized = False

        # Modular components
        self._mp_worker = MediaPipeWorker(config)
        self._labeler = SemanticLabeler()

        # Public: expose person_mask for pipeline
        self.last_person_mask: np.ndarray | None = None

        self._cached_segments: list = []
        self._model_path: str = ""
        self._last_logged_mp_fc: int = -1  # Prevent duplicate person_mask logs

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Load FastSAM (direct TRT or ultralytics fallback) + start MediaPipe process."""
        t0 = time.perf_counter()

        device = self.config.fastsam_device
        model_path = self.config.fastsam_model
        engine_path = model_path.replace(".pt", ".engine")

        # Try direct TRT first (bypasses ultralytics, 151 FPS)
        if os.path.exists(engine_path):
            try:
                from server.vision.fastsam_trt import FastSAMTRT
                self._sam = FastSAMTRT(engine_path, imgsz=self.config.fastsam_imgsz)
                self._use_direct_trt = True
                self._model_path = engine_path
                t_load = time.perf_counter()
                logger.info(f"    Direct TRT engine loaded: {engine_path} ({(t_load-t0)*1000:.0f}ms)")
            except Exception as e:
                logger.warning(f"    Direct TRT failed: {e}. Falling back to ultralytics.")
                self._use_direct_trt = False

        # Fallback to ultralytics
        if not self._use_direct_trt:
            from ultralytics import FastSAM
            t_import = time.perf_counter()
            logger.info(f"    import ultralytics: {(t_import - t0)*1000:.0f}ms")

            if os.path.exists(engine_path):
                model_path = engine_path
            self._model_path = model_path
            logger.info(f"    Loading FastSAM ({model_path}) on {device}")
            self._sam = FastSAM(model_path)
            t_load = time.perf_counter()
            logger.info(f"    FastSAM model load: {(t_load - t_import)*1000:.0f}ms")

            # Warmup
            try:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                imgsz = self.config.fastsam_imgsz if model_path.endswith(".engine") else 512
                for _ in range(3):
                    self._sam(dummy, device=device, imgsz=imgsz, conf=0.5,
                              retina_masks=True, verbose=False)
                logger.info(f"    CUDA warm-up: {(time.perf_counter()-t_load)*1000:.0f}ms")
            except Exception as e:
                logger.warning(f"    CUDA warm-up failed: {e}")

        # MediaPipe — start in separate process (no GIL contention)
        t_mp_start = time.perf_counter()
        self._mp_worker.start()
        t_mp = time.perf_counter()
        logger.info(f"    MediaPipe subprocess: {(t_mp - t_mp_start)*1000:.0f}ms")

        self._initialized = True
        logger.info(f"    FastSAMSegmenter ready ({(t_mp - t0)*1000:.0f}ms total) "
                     f"[{'direct TRT' if self._use_direct_trt else 'ultralytics'}]")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment_frame(self, frame_bgr: np.ndarray, sam_conf: float = None) -> list:
        """Run segmentation on one BGR frame.

        MediaPipe results come from a separate process (zero GIL contention).
        Only FastSAM inference + lightweight post-processing runs here.
        """
        if not self._initialized:
            self.initialize()

        t_total = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        # CLAHE contrast enhancement — improves segmentation in poor/uneven lighting
        # Applied BEFORE MediaPipe so both SAM and person detection benefit
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Feed enhanced frame to MediaPipe process (non-blocking)
        self._mp_worker.send_frame(frame_enhanced)

        # Get cached MediaPipe results (non-blocking poll from subprocess)
        body_masks, person_mask = self._mp_worker.get_results()

        self.last_person_mask = person_mask

        # Debug: log person_mask coverage once per MP update (every ~50 MP frames)
        mp_fc = self._mp_worker.frame_count
        if mp_fc > 0 and mp_fc % 50 == 0 and mp_fc != self._last_logged_mp_fc:
            self._last_logged_mp_fc = mp_fc
            if person_mask is not None:
                coverage = float(np.count_nonzero(person_mask > 0.5)) / max(1, h * w)
                n_body = len([m for m, _, _ in body_masks if m is not None])
                logger.info(f"person_mask: {coverage*100:.0f}% of frame, {n_body} body parts, mp_fc={mp_fc}")
            else:
                logger.info(f"person_mask: None, mp_fc={mp_fc}")

        # Run FastSAM + post-processing on contrast-enhanced frame
        masks_out = self._update_masks(frame_enhanced, h, w, body_masks, person_mask, sam_conf)

        # Convert to SegmentData (fast path)
        segments: list[SegmentData] = []
        for item in masks_out:
            if len(item) == 4:
                mask, label, center, bbox = item
                x, y, bw, bh = bbox
            else:
                # Body masks from MediaPipe use 3-tuple (no bbox pre-computed)
                mask, label, center = item
                mask_u8 = (mask > 0.5).astype(np.uint8)
                x, y, bw, bh = cv2.boundingRect(mask_u8)

            # Reject thin edge strips (letterbox bars, frame borders)
            # These span nearly the full width/height but are very thin
            if bw > 0 and bh > 0:
                aspect = max(bw, bh) / min(bw, bh)
                spans_width = bw > w * 0.8
                spans_height = bh > h * 0.8
                if aspect > 8 and (spans_width or spans_height):
                    continue

            seg = SegmentData(mask=mask, label=label, confidence=0.7)
            seg.center_x = float(center[0]) / w if center[0] > 0 else 0.0
            seg.center_y = float(center[1]) / h if center[1] > 0 else 0.0
            seg.mask_width = w
            seg.mask_height = h
            if bw > 0 and bh > 0:
                seg.bbox = (x / w, y / h, (x + bw) / w, (y + bh) / h)
            seg.asset_class = SemanticLabeler.label_to_asset_class(label)
            segments.append(seg)

        total_ms = (time.perf_counter() - t_total) * 1000
        self._cached_segments = segments

        if self._mp_worker.frame_count % 30 == 0:
            logger.info(f"segment_frame: {total_ms:.1f}ms | {len(segments)} segs")

        return segments

    # ------------------------------------------------------------------
    # _update_masks — uses direct TRT or ultralytics
    # ------------------------------------------------------------------

    def _update_masks(self, frame, h, w, body_masks, person_mask, sam_conf_override=None):
        """Run FastSAM + post-processing.

        TRT path: entire pipeline (resize + filter) runs on GPU via process_full().
        Ultralytics path: CPU-based fallback via _process_masks().
        """
        masks = []

        # Add person silhouette as a segment (so person gets an outline)
        if person_mask is not None and np.any(person_mask > 0.5):
            pm_u8 = ((person_mask > 0.5).astype(np.uint8)) * 255
            center = _fast_center(pm_u8)
            bbox = cv2.boundingRect(pm_u8)
            masks.append((pm_u8, "person", center, bbox))

        # Add individual body part masks (face, hands, arms, torso, legs)
        for body_mask, label, center in body_masks:
            if body_mask is not None:
                bm_u8 = ((body_mask > 0.5).astype(np.uint8)) * 255
                area = cv2.countNonZero(bm_u8)
                if area > 200:  # Skip tiny noise
                    bbox = cv2.boundingRect(bm_u8)
                    masks.append((bm_u8, label, center, bbox))

        # Build used_pixels from specific body parts only (not full silhouette).
        # Using person_mask as exclusion hides objects held by the person (bottle, phone).
        # Body part masks are small enough to exclude without losing held objects.
        used_pixels = np.zeros((h, w), dtype=bool)
        for body_mask, _, _ in body_masks:
            if body_mask is not None:
                used_pixels |= (body_mask > 0.5)

        sam_conf = sam_conf_override if sam_conf_override is not None else self.config.fastsam_conf

        try:
            if self._use_direct_trt:
                # GPU-accelerated path: infer + resize + filter all on GPU
                results = self._sam.process_full(
                    frame, h, w, used_pixels,
                    conf=sam_conf, max_masks=8,
                    min_area=self.config.mask_min_area,
                )
                self._labeler.reset_frame()
                # Downscaled connected components: find largest component at 1/4 res
                qh, qw = h // 4, w // 4
                for mask_u8, center, bbox in results:
                    small = cv2.resize(mask_u8, (qw, qh), interpolation=cv2.INTER_NEAREST)
                    n_labels, labels_s, stats, _ = cv2.connectedComponentsWithStats(small)
                    if n_labels > 2:
                        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        keep_small = (labels_s == largest).astype(np.uint8) * 255
                        keep_full = cv2.resize(keep_small, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_u8 = mask_u8 & keep_full
                    label, _ = self._labeler.get_label(mask_u8, h, w, frame)
                    label = self._labeler.unique_label(label)
                    center = _fast_center(mask_u8)
                    bbox = cv2.boundingRect(mask_u8)
                    masks.append((mask_u8, label, center, bbox))
            else:
                # Ultralytics fallback — CPU post-processing
                self._labeler.reset_frame()
                target_size = self.config.fastsam_imgsz if self._model_path.endswith(".engine") else min(512, max(320, min(h, w)))
                sam_results = self._sam(
                    frame, device=self.config.fastsam_device,
                    retina_masks=True, imgsz=target_size,
                    conf=sam_conf, verbose=False,
                )
                if sam_results and sam_results[0].masks is not None:
                    all_masks = sam_results[0].masks.data.cpu().numpy()
                    if len(all_masks) > 0:
                        # Sort by area (largest first) and limit — same as TRT path
                        areas = np.count_nonzero(all_masks.reshape(len(all_masks), -1) > 0.5, axis=1)
                        top_indices = np.argsort(areas)[::-1][:8]
                        all_masks = all_masks[top_indices]
                        self._process_masks(all_masks, h, w, used_pixels, masks, frame)

        except Exception as e:
            logger.error(f"SAM error: {e}")

        return masks

    def _process_masks(self, all_masks, h, w, used_pixels, masks, frame):
        """Vectorized mask post-processing shared by both TRT and ultralytics paths.

        Works in uint8 throughout to avoid expensive float32 resize (4x data).
        """
        n_raw = len(all_masks)
        sam_h, sam_w = all_masks.shape[1], all_masks.shape[2]

        # Downscale used_pixels for early rejection at mask resolution
        if sam_h != h or sam_w != w:
            used_small = cv2.resize(
                used_pixels.astype(np.uint8), (sam_w, sam_h),
                interpolation=cv2.INTER_NEAREST
            )
            min_area_small = max(1, int(self.config.mask_min_area * (sam_h * sam_w) / (h * w)))
        else:
            used_small = used_pixels.astype(np.uint8)
            min_area_small = self.config.mask_min_area

        # Binarize all masks at SAM resolution (uint8)
        all_binary = (all_masks > 0.5).astype(np.uint8)

        # Batch filter at mask resolution (vectorized — no per-mask loop)
        all_clean = all_binary & ~used_small[None, :, :]
        areas = np.count_nonzero(all_clean.reshape(n_raw, -1), axis=1)
        passed_indices = np.where(areas >= min_area_small)[0]

        morph_kernel = np.ones((3, 3), np.uint8)
        used_u8 = used_pixels.astype(np.uint8)

        for idx in passed_indices:
            # Resize uint8 binary mask (faster than float32 resize)
            mask_u8 = cv2.resize(all_binary[idx], (w, h),
                                 interpolation=cv2.INTER_NEAREST)
            clean_u8 = mask_u8 & ~used_u8

            area = cv2.countNonZero(clean_u8)
            if area < self.config.mask_min_area:
                continue

            # Morphological close (needs 0/255 range)
            clean_255 = clean_u8 * 255
            clean_255 = cv2.morphologyEx(clean_255, cv2.MORPH_CLOSE, morph_kernel)

            # Update used pixels
            used_u8 |= (clean_255 > 0).astype(np.uint8)

            # Convert to float only for output (required by RLE encoder downstream)
            clean_mask = clean_255.astype(np.float32) / 255.0

            label, _ = self._labeler.get_label(clean_mask, h, w, frame)
            label = self._labeler.unique_label(label)

            center = _fast_center(clean_255)
            bbox = cv2.boundingRect(clean_255)
            masks.append((clean_mask, label, center, bbox))

        # Write back used_pixels (bool)
        used_pixels |= used_u8.astype(bool)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_geometry(seg: SegmentData, mask_f: np.ndarray, h: int, w: int):
        """Populate bbox, centre, mask dims on a SegmentData."""
        mask_u8 = (mask_f > 0.5).astype(np.uint8)
        x, y, bw, bh = cv2.boundingRect(mask_u8)
        if bw > 0:
            seg.bbox = (x / w, y / h, (x + bw) / w, (y + bh) / h)
        M = cv2.moments(mask_u8)
        if M["m00"] > 0:
            seg.center_x = (M["m10"] / M["m00"]) / w
            seg.center_y = (M["m01"] / M["m00"]) / h
        seg.mask_width = w
        seg.mask_height = h

    def shutdown(self):
        """Release all resources."""
        self._mp_worker.shutdown()
        self._sam = None
        self._initialized = False
        self._cached_segments.clear()
        logger.info("FastSAMSegmenter shutdown")


def _fast_center(mask_u8: np.ndarray) -> tuple[int, int]:
    """Compute mask centroid via cv2.moments (10x faster than np.where)."""
    M = cv2.moments(mask_u8)
    if M["m00"] > 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return (0, 0)


def _mask_center(mask: np.ndarray) -> tuple[int, int]:
    """Compute mask centroid."""
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    return _fast_center(mask_u8)


