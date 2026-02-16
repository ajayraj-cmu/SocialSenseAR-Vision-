"""CV Model Registry — central catalogue of all supported models.

GitHub sources:
  YOLO:           https://github.com/ultralytics/ultralytics
  SAM3:           https://github.com/facebookresearch/sam3  (HF: facebook/sam3)
  Grounded SAM2:  https://github.com/IDEA-Research/Grounded-SAM-2
  MediaPipe:      https://github.com/google-ai-edge/mediapipe
  RF-DETR:        https://github.com/roboflow/rf-detr
  Depth Anything: https://github.com/LiheYoung/Depth-Anything-V2
  Sapiens:        https://github.com/facebookresearch/sapiens

Usage:
    from server.vision.registry import load_models

    models = load_models(["sam3", "mediapipe"], config)
    primary = models["sam3"]
    supplementary = [models["mediapipe"]]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.config import ServerConfig
    from server.vision.base import CVModelBase

logger = logging.getLogger(__name__)

# Registry: model_id → (class_path, output_type, install_note)
# class_path is a dotted import string resolved lazily so heavy deps aren't
# imported unless the model is actually activated.
CV_MODEL_REGISTRY: dict[str, dict] = {
    "sam3": {
        "module": "server.vision.sam3_segmenter",
        "class": "SAM3Segmenter",
        "output_type": "segmentation",
        "github": "https://github.com/facebookresearch/sam3",
        "install": "HF gated — set HF_SAM_TOKEN and login via huggingface-hub",
    },
    "grounded-sam2": {
        "module": "server.vision.grounded_sam2_segmenter",
        "class": "GroundedSAM2Segmenter",
        "output_type": "segmentation",
        "github": "https://github.com/IDEA-Research/Grounded-SAM-2",
        "install": "See docs/BUILD.md — GroundingDINO + SAM2 from HuggingFace",
    },
    "yolo": {
        "module": "server.vision.yolo_segmenter",
        "class": "YOLOSegmenter",
        "output_type": "segmentation",  # or detection/keypoints depending on variant
        "github": "https://github.com/ultralytics/ultralytics",
        "install": "pip install ultralytics",
    },
    "yolo-pose": {
        "module": "server.vision.yolo_segmenter",
        "class": "YOLOSegmenter",
        "output_type": "keypoints",
        "github": "https://github.com/ultralytics/ultralytics",
        "install": "pip install ultralytics",
        "kwargs": {"model_variant": "yolov8n-pose.pt", "output_type": "keypoints"},
    },
    "mediapipe": {
        "module": "server.vision.mediapipe_detector",
        "class": "MediaPipeDetector",
        "output_type": "keypoints",
        "github": "https://github.com/google-ai-edge/mediapipe",
        "install": "pip install mediapipe>=0.10.0",
    },
    "rf-detr": {
        "module": "server.vision.rfdetr_segmenter",
        "class": "RFDETRSegmenter",
        "output_type": "detection",
        "github": "https://github.com/roboflow/rf-detr",
        "install": "pip install rfdetr",
    },
    "depth-anything": {
        "module": "server.vision.depth_anything",
        "class": "DepthAnythingEstimator",
        "output_type": "depth",
        "github": "https://github.com/LiheYoung/Depth-Anything-V2",
        "install": "pip install depth-anything-v2  # or clone repo",
    },
    "sapiens": {
        "module": "server.vision.sapiens_estimator",
        "class": "SapiensEstimator",
        "output_type": "keypoints",
        "github": "https://github.com/facebookresearch/sapiens",
        "install": "pip install sapiens  # or HuggingFace: facebook/sapiens",
    },
}


def list_models() -> list[str]:
    """Return registered model IDs."""
    return list(CV_MODEL_REGISTRY.keys())


def get_model_info(model_id: str) -> dict:
    """Return metadata for a model without instantiating it."""
    if model_id not in CV_MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_id}'. Available: {list_models()}")
    return CV_MODEL_REGISTRY[model_id]


def load_model(model_id: str, config: "ServerConfig") -> "CVModelBase":
    """Instantiate and initialize a single model by ID."""
    if model_id not in CV_MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_id}'. Available: {list_models()}")

    entry = CV_MODEL_REGISTRY[model_id]
    module_path = entry["module"]
    class_name = entry["class"]
    kwargs = entry.get("kwargs", {})

    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    instance: "CVModelBase" = cls(config=config, **kwargs)
    instance.initialize()
    logger.info("Loaded model '%s' (output_type=%s)", model_id, entry["output_type"])
    return instance


def load_models(model_ids: list[str], config: "ServerConfig") -> dict[str, "CVModelBase"]:
    """Load and initialize multiple models. Returns {model_id: instance}."""
    loaded = {}
    for mid in model_ids:
        try:
            loaded[mid] = load_model(mid, config)
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", mid, exc)
            raise
    return loaded
