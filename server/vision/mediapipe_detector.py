"""MediaPipe detector — face, hands, pose, holistic.

GitHub: https://github.com/google-ai-edge/mediapipe
Install: pip install mediapipe>=0.10.0

Outputs keypoints for face (468 landmarks), pose (33 landmarks),
left hand (21 landmarks), right hand (21 landmarks).

Config keys (in cv_models.yaml):
  mediapipe:
    tasks: [pose, face, hands]   # which sub-detectors to run
    pose_complexity: 1           # 0=lite, 1=full, 2=heavy
    min_detection_confidence: 0.5
    min_tracking_confidence: 0.5
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

from server.vision.base import CVModelBase, RawOutput

logger = logging.getLogger(__name__)

# Pose skeleton (MediaPipe BlazePose 33 keypoints)
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_SKELETON = [
    0, 1, 1, 2, 2, 3, 3, 7,
    0, 4, 4, 5, 5, 6, 6, 8,
    9, 10,
    11, 12,
    11, 13, 13, 15, 15, 17, 17, 19, 19, 15,
    12, 14, 14, 16, 16, 18, 18, 20, 20, 16,
    11, 23, 12, 24, 23, 24,
    23, 25, 25, 27, 27, 29, 29, 31,
    24, 26, 26, 28, 28, 30, 30, 32,
]

POSE_KP_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HAND_SKELETON = [
    0, 1, 1, 2, 2, 3, 3, 4,
    0, 5, 5, 6, 6, 7, 7, 8,
    0, 9, 9, 10, 10, 11, 11, 12,
    0, 13, 13, 14, 14, 15, 15, 16,
    0, 17, 17, 18, 18, 19, 19, 20,
]

HAND_KP_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


class MediaPipeDetector(CVModelBase):
    """MediaPipe pose, face, and hand landmark detection.

    GitHub: https://github.com/google-ai-edge/mediapipe
    """

    output_type = "keypoints"
    model_id = "mediapipe"

    def __init__(self, config=None, tasks: Optional[list] = None,
                 pose_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        super().__init__(config)
        self.tasks = tasks or ["pose"]  # e.g. ["pose", "hands", "face"]
        self.pose_complexity = pose_complexity
        self.min_det_conf = min_detection_confidence
        self.min_track_conf = min_tracking_confidence
        self._pose = None
        self._hands = None
        self._face_mesh = None

    def _initialize(self) -> None:
        import mediapipe as mp  # https://github.com/google-ai-edge/mediapipe

        if "pose" in self.tasks:
            self._pose = mp.solutions.pose.Pose(
                model_complexity=self.pose_complexity,
                min_detection_confidence=self.min_det_conf,
                min_tracking_confidence=self.min_track_conf,
                enable_segmentation=False,
                static_image_mode=False,
            )
            logger.info("MediaPipe Pose ready (complexity=%d)", self.pose_complexity)

        if "hands" in self.tasks:
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=self.min_det_conf,
                min_tracking_confidence=self.min_track_conf,
                static_image_mode=False,
            )
            logger.info("MediaPipe Hands ready")

        if "face" in self.tasks:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=4,
                min_detection_confidence=self.min_det_conf,
                min_tracking_confidence=self.min_track_conf,
                refine_landmarks=True,
                static_image_mode=False,
            )
            logger.info("MediaPipe FaceMesh ready")

    def process(self, frame_bgr: np.ndarray) -> RawOutput:
        import mediapipe as mp
        H, W = frame_bgr.shape[:2]
        raw = RawOutput(output_type="keypoints", model_id=self.model_id, frame_shape=(H, W))

        frame_rgb = frame_bgr[:, :, ::-1]  # BGR → RGB
        all_group_sets = []

        # --- Pose ---
        if self._pose is not None:
            result = self._pose.process(frame_rgb)
            if result.pose_landmarks:
                points = []
                for idx, lm in enumerate(result.pose_landmarks.landmark):
                    points.append({
                        "x": lm.x, "y": lm.y,
                        "visibility": lm.visibility,
                        "name": POSE_KP_NAMES[idx] if idx < len(POSE_KP_NAMES) else str(idx),
                    })
                all_group_sets.append({
                    "groups": [{
                        "label": "body",
                        "points": points,
                        "skeleton_edges": POSE_SKELETON,
                    }]
                })

        # --- Hands ---
        if self._hands is not None:
            result = self._hands.process(frame_rgb)
            if result.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    handedness = "left_hand"
                    if result.multi_handedness and hand_idx < len(result.multi_handedness):
                        label = result.multi_handedness[hand_idx].classification[0].label
                        handedness = "left_hand" if label == "Left" else "right_hand"
                    points = []
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        points.append({
                            "x": lm.x, "y": lm.y,
                            "visibility": 1.0,
                            "name": HAND_KP_NAMES[idx] if idx < len(HAND_KP_NAMES) else str(idx),
                        })
                    all_group_sets.append({
                        "groups": [{
                            "label": handedness,
                            "points": points,
                            "skeleton_edges": HAND_SKELETON,
                        }]
                    })

        # --- Face ---
        if self._face_mesh is not None:
            result = self._face_mesh.process(frame_rgb)
            if result.multi_face_landmarks:
                for face_idx, face_landmarks in enumerate(result.multi_face_landmarks):
                    points = []
                    for idx, lm in enumerate(face_landmarks.landmark):
                        points.append({
                            "x": lm.x, "y": lm.y,
                            "visibility": 1.0,
                            "name": f"face_{idx}",
                        })
                    # Face mesh has too many edges to enumerate; send empty skeleton
                    all_group_sets.append({
                        "groups": [{
                            "label": f"face_{face_idx}",
                            "points": points,
                            "skeleton_edges": [],
                        }]
                    })

        raw.keypoint_groups = all_group_sets
        return raw

    def shutdown(self) -> None:
        for attr in ("_pose", "_hands", "_face_mesh"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.close()
                setattr(self, attr, None)
        super().shutdown()
