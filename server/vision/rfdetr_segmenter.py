"""RF-DETR segmenter using Roboflow's RF-DETR model.

GitHub: https://github.com/roboflow/rf-detr
Install: pip install rfdetr

RF-DETR is a real-time transformer-based detector that supports:
  - Object detection (bounding boxes + labels)
  - Instance segmentation (with mask head variant)

Config keys (in cv_models.yaml):
  rf-detr:
    model_variant: rfdetr_base     # rfdetr_base or rfdetr_large
    confidence: 0.35
    device: cuda
"""

from __future__ import annotations

import logging
import numpy as np

from server.vision.base import CVModelBase, RawOutput

logger = logging.getLogger(__name__)

# COCO class names (RF-DETR defaults to COCO 80-class)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class RFDETRSegmenter(CVModelBase):
    """RF-DETR real-time object detection.

    GitHub: https://github.com/roboflow/rf-detr
    Install: pip install rfdetr
    """

    output_type = "detection"
    model_id = "rf-detr"

    def __init__(self, config=None, model_variant: str = "rfdetr_base",
                 confidence: float = 0.35, device: str = "cuda"):
        super().__init__(config)
        self.model_variant = model_variant
        self.confidence = confidence
        self.device = device
        self._model = None

    def _initialize(self) -> None:
        # https://github.com/roboflow/rf-detr
        from rfdetr import RFDETRBase, RFDETRLarge  # pip install rfdetr

        logger.info("Loading RF-DETR model variant: %s", self.model_variant)
        if "large" in self.model_variant.lower():
            self._model = RFDETRLarge()
        else:
            self._model = RFDETRBase()

        # Warm up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._run_inference(dummy)
        logger.info("RF-DETR model ready")

    def _run_inference(self, frame_bgr: np.ndarray):
        """Run RF-DETR inference and return raw detections."""
        from PIL import Image
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        detections = self._model.predict(pil_img, threshold=self.confidence)
        return detections

    def process(self, frame_bgr: np.ndarray) -> RawOutput:
        H, W = frame_bgr.shape[:2]
        raw = RawOutput(output_type="detection", model_id=self.model_id, frame_shape=(H, W))

        try:
            detections = self._run_inference(frame_bgr)
        except Exception as e:
            logger.warning("RF-DETR inference error: %s", e)
            return raw

        # detections is a supervision Detections object
        # Fields: xyxy (N,4), confidence (N,), class_id (N,)
        if detections is None or len(detections) == 0:
            return raw

        for i in range(len(detections)):
            xyxy = detections.xyxy[i]  # pixel coords
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else 0

            x1 = float(xyxy[0]) / W
            y1 = float(xyxy[1]) / H
            x2 = float(xyxy[2]) / W
            y2 = float(xyxy[3]) / H

            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)

            raw.boxes.append((x1, y1, x2, y2))
            raw.labels.append(label)
            raw.scores.append(conf)
            raw.masks.append(None)  # detection only, no instance masks

        return raw

    def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        super().shutdown()
