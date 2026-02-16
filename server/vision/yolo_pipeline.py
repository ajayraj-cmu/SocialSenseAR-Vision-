"""YOLOPipeline — wraps YOLOSegmenter to match the segmenter interface
expected by PipelineOrchestrator (same contract as SAM3Segmenter / FastSAMSegmenter).

Interface:
    pipeline.initialize()                          -> None
    pipeline.segment_frame(frame_bgr, fw, fh)     -> list[SegmentData]
    pipeline.process_frame_sync(frame_bgr)        -> list[SegmentData]  (alias)
    pipeline.shutdown()                            -> None
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class YOLOPipeline:
    """Thin orchestrator-compatible wrapper around YOLOSegmenter.

    Provides the segment_frame() / initialize() interface that
    PipelineOrchestrator expects from any segmenter.
    """

    def __init__(self, config):
        self.config = config
        self._yolo = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        if self._initialized:
            return
        from server.vision.yolo_segmenter import YOLOSegmenter
        from server.vision.adapters import SegmentationAdapter, DetectionAdapter

        device = getattr(self.config, "device", "cpu")
        # Default to seg model; pose model can be set via cv_models_config
        model_variant = "yolov8n-seg.pt"
        confidence = 0.35
        iou = 0.45

        # Try to read from cv_models.yaml if present
        try:
            import yaml, os
            cfg_path = getattr(self.config, "cv_models_config", None)
            if cfg_path and os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cv_cfg = yaml.safe_load(f)
                yolo_cfg = cv_cfg.get("models", {}).get("yolo", {})
                model_variant = yolo_cfg.get("model_variant", model_variant)
                confidence = yolo_cfg.get("confidence", confidence)
                iou = yolo_cfg.get("iou", iou)
        except Exception as e:
            logger.debug("Could not load cv_models.yaml for YOLO: %s", e)

        self._yolo = YOLOSegmenter(
            config=self.config,
            model_variant=model_variant,
            confidence=confidence,
            iou=iou,
            device=device,
        )
        self._yolo.initialize()

        # Choose adapter based on output type
        if self._yolo.output_type == "segmentation":
            from server.vision.adapters import SegmentationAdapter
            self._adapter = SegmentationAdapter()
        else:
            from server.vision.adapters import DetectionAdapter
            self._adapter = DetectionAdapter()

        self._initialized = True
        logger.info("YOLOPipeline initialized (model=%s, device=%s)", model_variant, device)

    def shutdown(self) -> None:
        if self._yolo is not None:
            self._yolo.shutdown()
            self._yolo = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Inference — matches SAM3Segmenter / FastSAMSegmenter interface
    # ------------------------------------------------------------------

    def segment_frame(self, frame_bgr: np.ndarray,
                      frame_width: Optional[int] = None,
                      frame_height: Optional[int] = None):
        """Run YOLO on frame_bgr; return list[SegmentData]."""
        if not self._initialized:
            self.initialize()
        raw = self._yolo.process(frame_bgr)
        H, W = frame_bgr.shape[:2]
        return self._adapter.convert(raw, frame_shape=(H, W))

    # Alias used by some code paths
    def process_frame_sync(self, frame_bgr: np.ndarray):
        return self.segment_frame(frame_bgr)

    # ------------------------------------------------------------------
    # Prompt / label management stubs (no-op for YOLO — uses class names)
    # ------------------------------------------------------------------

    def get_active_prompts(self) -> set:
        return set()

    def set_active_prompts(self, prompts) -> None:
        pass  # YOLO detects all COCO classes by default

    def add_prompt(self, prompt: str) -> None:
        pass

    def remove_prompt(self, prompt: str) -> None:
        pass
