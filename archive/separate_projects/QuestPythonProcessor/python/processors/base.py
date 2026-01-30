"""
Base class for ML processors.

To add a new ML processor:
1. Create a new file in the processors/ directory
2. Inherit from BaseProcessor
3. Implement all abstract methods
4. Register in processors/__init__.py PROCESSORS dict

Example implementation:
    class MyProcessor(BaseProcessor):
        def __init__(self, config):
            self.model = load_my_model()
            self._fps = 0.0

        def start(self):
            # Start background processing thread
            self.thread = threading.Thread(target=self._worker)
            self.thread.start()

        def stop(self):
            self.running = False
            self.thread.join()

        def process(self, frame):
            self.queue.put(frame)

        def get_result(self):
            return ProcessorResult(
                left_mask=self.left_mask,
                right_mask=self.right_mask,
                left_box=self.left_box,
                right_box=self.right_box
            )

        @property
        def fps(self):
            return self._fps
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class ProcessorResult:
    """Result from a processor containing masks, boxes, and other data.

    Attributes:
        left_mask: Mask for left eye/view (H, W) uint8, 0-255
        right_mask: Mask for right eye/view (H, W) uint8, 0-255
        left_box: Bounding box for left (x1, y1, x2, y2) normalized 0-1
        right_box: Bounding box for right (x1, y1, x2, y2) normalized 0-1
        left_center: Center point of detection (x, y) normalized 0-1
        right_center: Center point of detection (x, y) normalized 0-1
        keypoints: Optional keypoints data (for pose estimation)
        extra: Optional dict for processor-specific data
    """
    left_mask: Optional[np.ndarray] = None
    right_mask: Optional[np.ndarray] = None
    left_box: Optional[Tuple[float, float, float, float]] = None
    right_box: Optional[Tuple[float, float, float, float]] = None
    left_center: Tuple[float, float] = (0.5, 0.5)
    right_center: Tuple[float, float] = (0.5, 0.5)
    keypoints: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)

    @property
    def has_detection(self) -> bool:
        """Check if any detection was found."""
        return self.left_mask is not None or self.right_mask is not None


class BaseProcessor(ABC):
    """Abstract base class for ML processors.

    Processors run ML models on video frames to extract information
    like segmentation masks, bounding boxes, pose keypoints, etc.

    Processors typically run asynchronously in a background thread
    to avoid blocking the main render loop.

    Attributes:
        fps: Current processing frames per second
    """

    @abstractmethod
    def start(self) -> None:
        """Start the processor (initialize model, start worker thread).

        Called once before processing begins.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the processor and cleanup resources.

        Called when shutting down.
        """
        pass

    @abstractmethod
    def process(self, frame: np.ndarray) -> None:
        """Submit a frame for processing.

        This should be non-blocking. The frame is queued for async processing.
        For stereo frames, the processor handles splitting into left/right.

        Args:
            frame: RGB frame as numpy array (H, W, 3)
        """
        pass

    @abstractmethod
    def get_result(self) -> ProcessorResult:
        """Get the latest processing result.

        Returns the most recent result with temporal smoothing applied.
        This should be non-blocking.

        Returns:
            ProcessorResult with masks, boxes, etc.
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Get current processing FPS.

        Returns:
            Frames processed per second
        """
        pass

    def warmup(self) -> None:
        """Warm up the model (optional).

        Run a few inference passes to initialize CUDA context, etc.
        """
        pass
