"""Visual Transformation: brightness, saturation, color temperature, blur, and texture effects."""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from loguru import logger
from src.core.contracts import (
    SegmentedObject, VisualOperation, TransformParameters,
    TransformResult, SafetyConstraints,
)


class VisualTransformer:
    """Visual transformation engine with safety constraints and temporal easing."""

    def __init__(
        self,
        min_brightness: float = 0.1,
        max_blur_radius: float = 30.0,
        default_transition_frames: int = 15,
    ):
        self.min_brightness = min_brightness
        self.max_blur_radius = max_blur_radius
        self.default_transition_frames = default_transition_frames
        self._current_params: Dict[str, TransformParameters] = {}
        self._transition_progress: Dict[str, float] = {}

    def apply_brightness_attenuation(
        self, frame: NDArray[np.uint8], mask: NDArray[np.uint8],
        factor: float, depth_map: Optional[NDArray[np.float32]] = None,
    ) -> NDArray[np.uint8]:
        factor = max(factor, self.min_brightness)
        soft_mask = self._create_soft_mask(mask, feather_radius=5)
        frame_float = frame.astype(np.float32)
        if depth_map is not None:
            mask_region = mask > 0
            if np.any(mask_region):
                depth_region = depth_map[mask_region]
                depth_normalized = np.clip(
                    (depth_region - depth_region.min()) / (depth_region.max() - depth_region.min() + 1e-6), 0, 1
                )
                adjusted = frame_float * factor
        else:
            adjusted = frame_float * factor
        soft_mask_3d = soft_mask[:, :, np.newaxis]
        result = frame_float * (1 - soft_mask_3d) + adjusted * soft_mask_3d
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_saturation_reduction(self, frame: NDArray[np.uint8], mask: NDArray[np.uint8], factor: float) -> NDArray[np.uint8]:
        soft_mask = self._create_soft_mask(mask, feather_radius=5)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        desaturated = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        soft_mask_3d = soft_mask[:, :, np.newaxis]
        result = frame.astype(np.float32) * (1 - soft_mask_3d) + desaturated.astype(np.float32) * soft_mask_3d
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_color_temperature_shift(self, frame: NDArray[np.uint8], mask: NDArray[np.uint8], shift: float) -> NDArray[np.uint8]:
        soft_mask = self._create_soft_mask(mask, feather_radius=5)
        frame_float = frame.astype(np.float32)
        if shift > 0:
            adjusted = frame_float.copy()
            adjusted[:, :, 0] = adjusted[:, :, 0] * (1 + shift * 0.2)
            adjusted[:, :, 2] = adjusted[:, :, 2] * (1 - shift * 0.15)
        else:
            adjusted = frame_float.copy()
            adjusted[:, :, 0] = adjusted[:, :, 0] * (1 + shift * 0.15)
            adjusted[:, :, 2] = adjusted[:, :, 2] * (1 - shift * 0.2)
        soft_mask_3d = soft_mask[:, :, np.newaxis]
        result = frame_float * (1 - soft_mask_3d) + adjusted * soft_mask_3d
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_edge_preserving_blur(
        self, frame: NDArray[np.uint8], mask: NDArray[np.uint8],
        radius: float, sigma_color: float = 75, sigma_space: float = 75,
    ) -> NDArray[np.uint8]:
        radius = min(radius, self.max_blur_radius)
        if radius < 1:
            return frame
        soft_mask = self._create_soft_mask(mask, feather_radius=3)
        d = int(radius * 2) | 1
        blurred = cv2.bilateralFilter(frame, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        soft_mask_3d = soft_mask[:, :, np.newaxis]
        result = frame.astype(np.float32) * (1 - soft_mask_3d) + blurred.astype(np.float32) * soft_mask_3d
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_texture_simplification(self, frame: NDArray[np.uint8], mask: NDArray[np.uint8], strength: float) -> NDArray[np.uint8]:
        if strength <= 0:
            return frame
        soft_mask = self._create_soft_mask(mask, feather_radius=5)
        sigma_s = int(10 + strength * 50)
        sigma_r = 0.1 + strength * 0.3
        simplified = cv2.bilateralFilter(frame, d=-1, sigmaColor=int(sigma_r * 255), sigmaSpace=sigma_s)
        if strength > 0.5:
            levels = max(8, int(32 * (1 - strength)))
            simplified = (simplified // levels) * levels
        soft_mask_3d = soft_mask[:, :, np.newaxis]
        result = frame.astype(np.float32) * (1 - soft_mask_3d) + simplified.astype(np.float32) * soft_mask_3d
        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_transform(
        self, frame: NDArray[np.uint8], obj: SegmentedObject, operation: VisualOperation,
        parameters: TransformParameters, safety: SafetyConstraints,
        depth_map: Optional[NDArray[np.float32]] = None,
    ) -> TransformResult:
        mask = obj.smoothed_mask if obj.smoothed_mask is not None else obj.mask
        if mask is None or frame.shape[:2] != mask.shape:
            return TransformResult(modified_frame=frame, success=False, error_message="Invalid mask")

        was_constrained = False
        constraint_details = None

        try:
            if operation == VisualOperation.BRIGHTNESS_ATTENUATION:
                safe_factor = max(parameters.brightness_factor, safety.min_brightness)
                if safe_factor != parameters.brightness_factor:
                    was_constrained = True
                    constraint_details = f"Brightness clamped to minimum {safety.min_brightness}"
                result_frame = self.apply_brightness_attenuation(frame, mask, safe_factor, depth_map)
            elif operation == VisualOperation.SATURATION_REDUCTION:
                result_frame = self.apply_saturation_reduction(frame, mask, parameters.saturation_factor)
            elif operation == VisualOperation.COLOR_TEMPERATURE_SHIFT:
                result_frame = self.apply_color_temperature_shift(frame, mask, parameters.color_temperature_shift)
            elif operation == VisualOperation.EDGE_PRESERVING_BLUR:
                safe_radius = min(parameters.blur_radius, safety.max_blur_radius)
                if safe_radius != parameters.blur_radius:
                    was_constrained = True
                    constraint_details = f"Blur radius clamped to {safety.max_blur_radius}px"
                result_frame = self.apply_edge_preserving_blur(frame, mask, safe_radius)
            elif operation == VisualOperation.TEXTURE_SIMPLIFICATION:
                result_frame = self.apply_texture_simplification(frame, mask, parameters.texture_simplification)
            else:
                logger.warning(f"Unsupported visual operation: {operation}")
                return TransformResult(modified_frame=frame, success=False, error_message=f"Unsupported operation: {operation}")

            return TransformResult(
                modified_frame=result_frame,
                was_constrained=was_constrained,
                constraint_details=constraint_details,
                success=True,
            )
        except Exception as e:
            logger.error(f"Visual transform failed: {e}")
            return TransformResult(modified_frame=frame, success=False, error_message=str(e))

    def _create_soft_mask(self, mask: NDArray[np.uint8], feather_radius: int = 5) -> NDArray[np.float32]:
        if feather_radius <= 0:
            return mask.astype(np.float32)
        kernel_size = feather_radius * 2 + 1
        soft = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), feather_radius / 3)
        return np.clip(soft, 0, 1)

    def ease_parameter(self, current: float, target: float, progress: float, easing: str = "ease_out_quad") -> float:
        if easing == "linear":
            t = progress
        elif easing == "ease_out_quad":
            t = 1 - (1 - progress) ** 2
        elif easing == "ease_in_out_quad":
            if progress < 0.5:
                t = 2 * progress ** 2
            else:
                t = 1 - (-2 * progress + 2) ** 2 / 2
        else:
            t = progress
        return current + (target - current) * t

    def reset(self):
        self._current_params.clear()
        self._transition_progress.clear()
        logger.debug("Visual transformer reset")
