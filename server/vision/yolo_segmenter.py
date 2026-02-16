"""YOLO segmenter using Ultralytics.

GitHub: https://github.com/ultralytics/ultralytics
Install: pip install ultralytics

Supports:
  - Detection:    YOLOv8n/s/m/l/x  (model_variant="yolov8n.pt")
  - Segmentation: YOLOv8n-seg etc  (model_variant="yolov8n-seg.pt")
  - Pose:         YOLOv8n-pose etc (model_variant="yolov8n-pose.pt")

Config keys (in cv_models.yaml):
  yolo:
    model_variant: yolov8n-seg.pt   # auto-downloaded on first run
    confidence: 0.35
    iou: 0.45
    device: cuda
    output_type: segmentation       # or detection, keypoints
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

from server.vision.base import CVModelBase, RawOutput

logger = logging.getLogger(__name__)

# COCO pose skeleton edges (pairs of keypoint indices)
COCO_SKELETON = [
    0, 1, 0, 2, 1, 3, 2, 4,           # head
    5, 6,                               # shoulders
    5, 7, 7, 9,                        # left arm
    6, 8, 8, 10,                       # right arm
    5, 11, 6, 12,                      # torso
    11, 12,                            # hips
    11, 13, 13, 15,                    # left leg
    12, 14, 14, 16,                    # right leg
]

COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class YOLOSegmenter(CVModelBase):
    """Ultralytics YOLO for detection, segmentation, or pose estimation.

    GitHub: https://github.com/ultralytics/ultralytics
    """

    model_id = "yolo"

    def __init__(self, config=None, model_variant: str = "yolov8n-seg.pt",
                 confidence: float = 0.35, iou: float = 0.45,
                 device: str = "cuda", output_type: str = "segmentation"):
        super().__init__(config)
        self.model_variant = model_variant
        self.confidence = confidence
        self.iou = iou
        self.device = device
        # Infer output_type from model name if not explicitly set
        if "pose" in model_variant:
            self.output_type = "keypoints"
        elif "seg" in model_variant:
            self.output_type = "segmentation"
        else:
            self.output_type = output_type if output_type in ("segmentation", "detection") else "detection"
        self._model = None

    def _initialize(self) -> None:
        from ultralytics import YOLO  # https://github.com/ultralytics/ultralytics
        logger.info("Loading YOLO model: %s (output_type=%s)", self.model_variant, self.output_type)
        self._model = YOLO(self.model_variant)
        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(dummy, verbose=False, device=self.device)
        logger.info("YOLO model ready")

    def process(self, frame_bgr: np.ndarray) -> RawOutput:
        H, W = frame_bgr.shape[:2]
        raw = RawOutput(output_type=self.output_type, model_id=self.model_id, frame_shape=(H, W))

        results = self._model(
            frame_bgr,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        if not results:
            return raw

        result = results[0]

        if self.output_type == "keypoints" and result.keypoints is not None:
            raw.keypoint_groups.extend(self._parse_keypoints(result, H, W))
        else:
            self._parse_detection_seg(result, raw, H, W)

        return raw

    def _parse_detection_seg(self, result, raw: RawOutput, H: int, W: int) -> None:
        if result.boxes is None:
            return
        boxes_xyxy = result.boxes.xyxyn.cpu().numpy()  # normalized
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        masks_list = []
        if result.masks is not None:
            import torch
            # masks.data shape: (N, H_mask, W_mask)
            import torch.nn.functional as F
            mask_data = result.masks.data.cpu()
            # Resize to frame size
            resized = F.interpolate(
                mask_data.unsqueeze(1).float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).numpy()
            masks_list = [(m > 0.5).astype(np.uint8) for m in resized]

        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            raw.boxes.append((float(x1), float(y1), float(x2), float(y2)))
            raw.labels.append(names[cls_ids[i]] if cls_ids[i] in names else str(cls_ids[i]))
            raw.scores.append(float(confs[i]))
            raw.masks.append(masks_list[i] if i < len(masks_list) else None)

    def _parse_keypoints(self, result, H: int, W: int) -> list[dict]:
        """Parse YOLO pose results into canonical keypoint group dicts."""
        kp_data = result.keypoints.xyn.cpu().numpy()  # (N, 17, 2) normalized
        conf_data = result.keypoints.conf
        if conf_data is not None:
            conf_data = conf_data.cpu().numpy()  # (N, 17)
        else:
            conf_data = np.ones((kp_data.shape[0], kp_data.shape[1]))

        groups = []
        for person_idx in range(kp_data.shape[0]):
            points = []
            for kp_idx in range(kp_data.shape[1]):
                x, y = kp_data[person_idx, kp_idx]
                vis = float(conf_data[person_idx, kp_idx]) if conf_data is not None else 1.0
                name = COCO_KP_NAMES[kp_idx] if kp_idx < len(COCO_KP_NAMES) else str(kp_idx)
                points.append({"x": float(x), "y": float(y), "visibility": vis, "name": name})

            groups.append({
                "groups": [{
                    "label": "body",
                    "points": points,
                    "skeleton_edges": COCO_SKELETON,
                }]
            })
        return groups

    def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        super().shutdown()
