"""
ML Processor module.

Provides different ML models for frame processing (YOLO, MediaPipe, etc.)

To add a new processor:
1. Create a new file in this directory (e.g., mediapipe_pose.py)
2. Implement a class inheriting from BaseProcessor
3. Register it in PROCESSORS dict below

Example:
    # In processors/mediapipe_pose.py
    from .base import BaseProcessor, ProcessorResult

    class MediaPipePoseProcessor(BaseProcessor):
        def process(self, frame):
            # Submit frame for processing
            ...

        def get_result(self):
            return ProcessorResult(masks=..., boxes=..., keypoints=...)

    # Then add to PROCESSORS:
    # "mediapipe": MediaPipePoseProcessor,
"""
from typing import Optional
from .base import BaseProcessor, ProcessorResult
from .yolo_segmentation import YOLOSegmentationProcessor

# Registry of available processors
PROCESSORS = {
    "yolo": YOLOSegmentationProcessor,
    # Add more processors here:
    # "mediapipe": MediaPipePoseProcessor,
    # "densepose": DensePoseProcessor,
}


def get_processor(name: str, config=None) -> BaseProcessor:
    """Get a processor instance by name.

    Args:
        name: Processor type name (e.g., "yolo", "mediapipe")
        config: Configuration object

    Returns:
        Initialized processor instance

    Raises:
        ValueError: If processor name is not registered
    """
    if name not in PROCESSORS:
        available = ", ".join(PROCESSORS.keys())
        raise ValueError(f"Unknown processor '{name}'. Available: {available}")

    return PROCESSORS[name](config)


def list_processors() -> list:
    """List available processor names."""
    return list(PROCESSORS.keys())


__all__ = ['BaseProcessor', 'ProcessorResult', 'YOLOSegmentationProcessor',
           'get_processor', 'list_processors']
