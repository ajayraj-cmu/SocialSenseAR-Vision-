"""
Focus/grayscale effect.

Makes the background grayscale while keeping the detected person in color.
During transitions, the color "collapses" from the screen edges to the person.
"""
import numpy as np
import cv2

from .base import BaseEffect
from .gpu_kernels import apply_steady_gpu, apply_transition_gpu, CUPY_AVAILABLE
from processors.base import ProcessorResult


# Transition effect resolution (balances quality and speed)
TRANSITION_PROCESS_WIDTH = 480


class FocusGrayscaleEffect(BaseEffect):
    """Focus effect that highlights the detected person.

    The detected person stays in color while the background becomes grayscale.
    During transitions, the effect animates from screen edges toward the person.

    Features:
        - GPU-accelerated rendering via CuPy
        - Smooth distance-based transition animation
        - Optimized for real-time performance
    """

    def __init__(self, config=None):
        """Initialize focus effect.

        Args:
            config: Configuration object with effect_scale, mask_blur, etc.
        """
        self.config = config
        self.effect_scale = 1.0
        if config:
            self.effect_scale = getattr(config, 'effect_scale', 1.0)

    def apply(self, frame: np.ndarray, result: ProcessorResult) -> np.ndarray:
        """Apply focus effect (steady state).

        Args:
            frame: RGB input frame
            result: ProcessorResult with masks

        Returns:
            BGR output frame
        """
        if self.effect_scale <= 0:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return apply_steady_gpu(frame, result.left_mask, result.right_mask)

    def apply_transition(self, frame: np.ndarray, result: ProcessorResult,
                         progress: float) -> np.ndarray:
        """Apply focus effect with animated transition.

        The color "collapses" from screen edges toward the person silhouette.
        Progress 0.0 = full grayscale, 1.0 = full color on person.

        Args:
            frame: RGB input frame
            result: ProcessorResult with masks
            progress: Transition progress 0.0 to 1.0

        Returns:
            BGR output frame
        """
        frame_h, frame_w = frame.shape[:2]
        half_w = frame_w // 2

        # Process at reduced resolution for speed
        proc_w = TRANSITION_PROCESS_WIDTH
        proc_h = int(frame_h * proc_w / half_w)

        # Animation: progress 0->1 means color collapses FROM full screen TO person
        inverse_progress = 1.0 - progress

        # Precompute constants
        max_dist = np.sqrt(proc_w**2 + proc_h**2) * 1.2
        threshold = inverse_progress * max_dist
        soft_edge = max(10.0, max_dist * 0.15)
        inv_soft_edge = 1.0 / soft_edge

        # Compute weight mask at low resolution
        weight_small = np.zeros((proc_h, proc_w * 2), dtype=np.float32)

        for mask, x_start, x_end in [(result.left_mask, 0, proc_w),
                                      (result.right_mask, proc_w, proc_w * 2)]:
            region = weight_small[:, x_start:x_end]

            if mask is None:
                region[:] = inverse_progress
                continue

            # Downscale mask
            mask_small = cv2.resize(mask, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

            if mask_small.max() <= 127:
                region[:] = inverse_progress
                continue

            # Distance transform for expansion animation
            binary = (mask_small > 127).astype(np.uint8)
            dist_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)

            # Expansion from person silhouette
            expansion_weight = np.clip((threshold - dist_outside) * inv_soft_edge, 0, 1)
            mask_weight = mask_small.astype(np.float32) * (1.0 / 255.0)

            np.maximum(mask_weight, expansion_weight, out=region)

        # Convert to uint8 at low res, then upscale
        weight_small_u8 = (weight_small * 255).astype(np.uint8)
        weight_u8 = cv2.resize(weight_small_u8, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        # GPU-accelerated blending
        return apply_transition_gpu(frame, weight_u8)
