"""
Base class for visual effects.

To add a new visual effect:
1. Create a new file in the effects/ directory
2. Inherit from BaseEffect
3. Implement apply() and apply_transition()
4. Register in effects/__init__.py EFFECTS dict

Example implementation:
    class BlurBackgroundEffect(BaseEffect):
        def __init__(self, config):
            self.blur_strength = 25

        def apply(self, frame, result):
            if result.left_mask is None:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Create blurred version
            blurred = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)

            # Blend based on mask
            # ... implementation ...

            return output

        def apply_transition(self, frame, result, progress):
            # Animate blur strength based on progress
            current_blur = int(self.blur_strength * progress)
            # ... implementation ...
"""
from abc import ABC, abstractmethod
import numpy as np

# Import ProcessorResult for type hints
from processors.base import ProcessorResult


class BaseEffect(ABC):
    """Abstract base class for visual effects.

    Effects transform video frames based on processor results (masks, boxes, etc.)
    They must support both steady-state rendering and animated transitions.

    The apply() method is called when the effect is fully active.
    The apply_transition() method is called during on/off animations.

    All methods should return BGR frames (OpenCV format) for display.
    """

    @abstractmethod
    def apply(self, frame: np.ndarray, result: ProcessorResult) -> np.ndarray:
        """Apply the effect to a frame (steady state).

        Called when the effect is fully active (not transitioning).

        Args:
            frame: RGB input frame (H, W, 3)
            result: ProcessorResult with masks, boxes, etc.

        Returns:
            BGR output frame for display
        """
        pass

    @abstractmethod
    def apply_transition(self, frame: np.ndarray, result: ProcessorResult,
                         progress: float) -> np.ndarray:
        """Apply the effect with transition animation.

        Called during effect on/off transitions.
        Progress goes from 0.0 (off) to 1.0 (fully on).

        Args:
            frame: RGB input frame (H, W, 3)
            result: ProcessorResult with masks, boxes, etc.
            progress: Transition progress from 0.0 to 1.0

        Returns:
            BGR output frame for display
        """
        pass

    def no_effect(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with no effect applied (just color conversion).

        Args:
            frame: RGB input frame

        Returns:
            BGR output frame
        """
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
