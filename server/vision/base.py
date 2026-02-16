"""Base class and interfaces for all CV models.

All CV models must subclass CVModelBase and implement:
  - initialize(): load weights, warm up
  - process(frame_bgr): run inference, return RawOutput
  - shutdown(): release GPU memory

GitHub sources for supported models:
  YOLO:           https://github.com/ultralytics/ultralytics
  SAM3:           https://github.com/facebookresearch/sam3 (via HF: facebook/sam3)
  Grounded SAM2:  https://github.com/IDEA-Research/Grounded-SAM-2
  MediaPipe:      https://github.com/google-ai-edge/mediapipe
  RF-DETR:        https://github.com/roboflow/rf-detr
  Depth Anything: https://github.com/LiheYoung/Depth-Anything-V2
  Sapiens:        https://github.com/facebookresearch/sapiens
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

OutputType = Literal["segmentation", "detection", "keypoints", "depth"]


@dataclass
class RawOutput:
    """Canonical raw output from any CV model before adapter conversion."""

    output_type: OutputType

    # segmentation / detection
    masks: list[np.ndarray] = field(default_factory=list)          # HxW bool/uint8
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # (x1,y1,x2,y2) norm 0-1
    labels: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    # keypoints: list of KeypointGroup dicts
    # [{label, points: [{x,y,visibility,name}], skeleton_edges: [int]}]
    keypoint_groups: list[dict] = field(default_factory=list)

    # depth
    depth_map: Optional[np.ndarray] = None  # HxW float32

    # source metadata
    model_id: str = ""
    frame_shape: tuple[int, int] = (0, 0)  # (H, W)


class CVModelBase(abc.ABC):
    """Abstract base class for all CV models."""

    output_type: OutputType = "segmentation"
    model_id: str = "base"

    def __init__(self, config=None):
        self.config = config
        self._initialized = False

    def initialize(self) -> None:
        """Load weights and prepare for inference. Called once at startup."""
        self._initialize()
        self._initialized = True
        logger.info("Model '%s' initialized", self.model_id)

    @abc.abstractmethod
    def _initialize(self) -> None: ...

    @abc.abstractmethod
    def process(self, frame_bgr: np.ndarray) -> RawOutput:
        """Run inference on a single BGR frame. Must return RawOutput."""
        ...

    def shutdown(self) -> None:
        """Release GPU/CPU resources."""
        logger.info("Model '%s' shut down", self.model_id)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model_id={self.model_id!r} output={self.output_type!r}>"
