"""MediaPipe person/face/hands/pose detection.

Extracted from fastsam_segmenter.py (originally sam_gemini_voice.py).
Owns all MediaPipe models, timestamp counter, frame counter, and person_mask state.
Every parameter, kernel size, and detection threshold is identical to the original.

SAFE TO EDIT: Changes here only affect person/body detection, not SAM or labeling.
"""

import os
import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _mask_center(mask: np.ndarray) -> tuple[int, int]:
    """Compute mask centroid. Same as sam_gemini_voice.py line 3299."""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return (0, 0)
    return (int(np.mean(xs)), int(np.mean(ys)))


class MediaPipeDetector:
    """Detects person, face, hands, and body parts using MediaPipe.

    State:
        person_mask: float32 array, updated every 2 frames by selfie segmenter.
        frame_count: controls selfie every-2-frames cadence.
        _mp_timestamp_ms: monotonically increasing, shared across all MediaPipe calls.
    """

    def __init__(self, config):
        self.config = config

        # Models (loaded in initialize())
        self._selfie = None
        self._face_mesh = None
        self._hands = None
        self._pose = None

        # Shared MediaPipe timestamp (must be monotonically increasing across all models)
        self._mp_timestamp_ms = 0

        # Person mask (persists across frames, updated every 2 frames)
        self.person_mask: np.ndarray | None = None
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Initialization — exact same options as sam_gemini_voice.py __init__
    # ------------------------------------------------------------------

    def initialize(self):
        """Load all MediaPipe models."""
        t0 = time.perf_counter()

        import mediapipe as mp
        t_import = time.perf_counter()
        logger.info(f"      import mediapipe: {(t_import - t0)*1000:.0f}ms")

        models_dir = self.config.mediapipe_models_dir
        _BaseOptions = mp.tasks.BaseOptions
        _VisionRunningMode = mp.tasks.vision.RunningMode

        selfie_path = os.path.join(models_dir, "selfie_segmenter.tflite")
        if os.path.exists(selfie_path):
            t = time.perf_counter()
            self._selfie = mp.tasks.vision.ImageSegmenter.create_from_options(
                mp.tasks.vision.ImageSegmenterOptions(
                    base_options=_BaseOptions(model_asset_path=selfie_path),
                    running_mode=_VisionRunningMode.VIDEO,
                    output_category_mask=True,
                )
            )
            logger.info(f"      selfie segmenter: {(time.perf_counter() - t)*1000:.0f}ms")
        else:
            logger.warning(f"      selfie model not found: {selfie_path}")

        face_path = os.path.join(models_dir, "face_landmarker.task")
        if os.path.exists(face_path):
            t = time.perf_counter()
            self._face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=_BaseOptions(model_asset_path=face_path),
                    running_mode=_VisionRunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            logger.info(f"      face landmarker: {(time.perf_counter() - t)*1000:.0f}ms")
        else:
            logger.warning(f"      face model not found: {face_path}")

        hand_path = os.path.join(models_dir, "hand_landmarker.task")
        if os.path.exists(hand_path):
            t = time.perf_counter()
            self._hands = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    base_options=_BaseOptions(model_asset_path=hand_path),
                    running_mode=_VisionRunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            logger.info(f"      hand landmarker: {(time.perf_counter() - t)*1000:.0f}ms")
        else:
            logger.warning(f"      hand model not found: {hand_path}")

        pose_path = os.path.join(models_dir, "pose_landmarker_full.task")
        if os.path.exists(pose_path):
            t = time.perf_counter()
            self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=_BaseOptions(model_asset_path=pose_path),
                    running_mode=_VisionRunningMode.VIDEO,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            logger.info(f"      pose landmarker: {(time.perf_counter() - t)*1000:.0f}ms")
        else:
            logger.warning(f"      pose model not found: {pose_path}")

        self._mp_timestamp_ms = 0
        logger.info(f"      MediaPipe total: {(time.perf_counter() - t0)*1000:.0f}ms")

    # ------------------------------------------------------------------
    # Main detection — returns body masks + current person_mask
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray, rgb: np.ndarray | None, h: int, w: int) -> tuple[list, np.ndarray | None]:
        """Run all MediaPipe detectors on one frame.

        Handles selfie segmenter every-2-frames cadence internally.

        Args:
            frame_bgr: BGR uint8 image.
            rgb: Pre-computed RGB image, or None (will be computed).
            h, w: Frame dimensions.

        Returns:
            (body_masks, person_mask) where:
                body_masks: list[(mask, label, center)] for person/face/hands/body parts.
                person_mask: float32 person mask (or None). Use for SAM used_pixels.
        """
        import mediapipe as mp

        t_total = time.perf_counter()
        masks = []

        # Ensure RGB
        if rgb is None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ============================================================
        # Selfie segmenter (every 2 frames) — line 1735-1745
        # ============================================================
        t_selfie = time.perf_counter()
        if self.frame_count % 2 == 0:
            if self._selfie is not None:
                self._mp_timestamp_ms += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    selfie_result = self._selfie.segment_for_video(
                        mp_image, self._mp_timestamp_ms
                    )
                    if selfie_result.category_mask is not None:
                        # np.squeeze — exact same as original line 1743
                        self.person_mask = np.squeeze(
                            selfie_result.category_mask.numpy_view()
                        ).astype(np.float32)
                    else:
                        self.person_mask = np.zeros((h, w), dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Selfie segmenter error: {e}")
        self.frame_count += 1
        selfie_ms = (time.perf_counter() - t_selfie) * 1000

        # Person mask is used for SAM exclusion and person-awareness,
        # but NOT added as a display segment (it overlaps all other body parts).

        # ============================================================
        # Face from FaceLandmarker
        # ============================================================
        t_face = time.perf_counter()
        if self._face_mesh is not None:
            self._mp_timestamp_ms += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                face_result = self._face_mesh.detect_for_video(mp_image, self._mp_timestamp_ms)
                if face_result.face_landmarks:
                    face_mask = np.zeros((h, w), dtype=np.float32)
                    pts = np.array([[int(lm.x * w), int(lm.y * h)]
                                   for lm in face_result.face_landmarks[0]], dtype=np.int32)
                    hull = cv2.convexHull(pts)
                    cv2.fillConvexPoly(face_mask, hull, 1.0)
                    face_mask = cv2.GaussianBlur(face_mask, (5, 5), 0)
                    center = _mask_center(face_mask)
                    masks.append((face_mask, "face", center))
            except Exception as e:
                logger.debug(f"Face landmarker error: {e}")
        face_ms = (time.perf_counter() - t_face) * 1000

        # ============================================================
        # Hands from HandLandmarker
        # ============================================================
        t_hands = time.perf_counter()
        if self._hands is not None:
            self._mp_timestamp_ms += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                hands_result = self._hands.detect_for_video(mp_image, self._mp_timestamp_ms)
                if hands_result.hand_landmarks and hands_result.handedness:
                    for hand_lm, handedness in zip(hands_result.hand_landmarks, hands_result.handedness):
                        hand_mask = np.zeros((h, w), dtype=np.float32)
                        pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_lm], dtype=np.int32)
                        hull = cv2.convexHull(pts)
                        cv2.fillConvexPoly(hand_mask, hull, 1.0)
                        # Expand hand region
                        kernel = np.ones((9, 9), np.uint8)
                        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
                        hand_mask = cv2.GaussianBlur(hand_mask, (5, 5), 0)

                        # Label based on handedness (mirrored in selfie view)
                        label_side = handedness[0].category_name
                        if label_side == "Left":
                            hand_label = "right_hand"  # Mirrored
                        else:
                            hand_label = "left_hand"

                        center = _mask_center(hand_mask)
                        masks.append((hand_mask, hand_label, center))
            except Exception as e:
                logger.debug(f"Hand landmarker error: {e}")
        hands_ms = (time.perf_counter() - t_hands) * 1000

        # ============================================================
        # Body parts from PoseLandmarker
        # ============================================================
        t_pose = time.perf_counter()
        if self._pose is not None:
            self._mp_timestamp_ms += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                pose_result = self._pose.detect_for_video(mp_image, self._mp_timestamp_ms)
                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks[0]

                    body_parts = {
                        "left_arm": [11, 13, 15],
                        "right_arm": [12, 14, 16],
                        "torso": [11, 12, 24, 23],
                        "left_leg": [23, 25, 27, 31],
                        "right_leg": [24, 26, 28, 32],
                    }

                    for part_name, indices in body_parts.items():
                        pts = []
                        valid = True
                        for idx in indices:
                            if idx < len(lm) and (lm[idx].visibility or 0) > 0.5:
                                pts.append([int(lm[idx].x * w), int(lm[idx].y * h)])
                            else:
                                valid = False
                                break

                        if valid and len(pts) >= 3:
                            part_mask = np.zeros((h, w), dtype=np.float32)
                            pts_arr = np.array(pts, dtype=np.int32)

                            if "arm" in part_name or "leg" in part_name:
                                for i in range(len(pts) - 1):
                                    cv2.line(part_mask, tuple(pts[i]), tuple(pts[i+1]), 1.0, 25)
                            else:
                                hull = cv2.convexHull(pts_arr)
                                cv2.fillConvexPoly(part_mask, hull, 1.0)

                            # Smooth
                            kernel = np.ones((7, 7), np.uint8)
                            part_mask = cv2.dilate(part_mask, kernel, iterations=1)
                            part_mask = cv2.GaussianBlur(part_mask, (7, 7), 0)

                            if np.any(part_mask > 0.3):
                                center = _mask_center(part_mask)
                                masks.append((part_mask, part_name, center))
            except Exception as e:
                logger.debug(f"Pose landmarker error: {e}")
        pose_ms = (time.perf_counter() - t_pose) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000
        logger.debug(
            f"MediaPipe: selfie={selfie_ms:.1f}ms face={face_ms:.1f}ms "
            f"hands={hands_ms:.1f}ms pose={pose_ms:.1f}ms total={total_ms:.1f}ms "
            f"| {len(masks)} body parts"
        )

        return masks, self.person_mask

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Release MediaPipe models."""
        self._selfie = None
        self._face_mesh = None
        self._hands = None
        self._pose = None
        self.person_mask = None
        logger.info("MediaPipeDetector shutdown")
