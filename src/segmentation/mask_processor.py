"""Mask Processing: morphological cleaning, temporal smoothing, and edge refinement."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from loguru import logger


class MaskProcessor:
    """Processor for segmentation masks with temporal smoothing and morphological cleaning."""

    def __init__(
        self,
        smoothing_kernel_size: int = 5,
        morphological_iterations: int = 2,
        min_area_threshold: int = 100,
    ):
        self.smoothing_kernel_size = smoothing_kernel_size
        self.morphological_iterations = morphological_iterations
        self.min_area_threshold = min_area_threshold
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (smoothing_kernel_size, smoothing_kernel_size)
        )

    def clean_mask(self, mask: NDArray[np.uint8], remove_small_regions: bool = True) -> NDArray[np.uint8]:
        if mask is None or mask.size == 0:
            return mask
        mask = (mask > 0).astype(np.uint8) * 255
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=self.morphological_iterations)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._morph_kernel, iterations=self.morphological_iterations)
        if remove_small_regions:
            closed = self._remove_small_components(closed)
        return (closed > 127).astype(np.uint8)

    def _remove_small_components(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        output = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area_threshold:
                output[labels == i] = 255
        return output

    def temporal_smooth(
        self, current_mask: NDArray[np.uint8], previous_mask: NDArray[np.uint8], alpha: float = 0.7,
    ) -> NDArray[np.uint8]:
        if previous_mask is None or previous_mask.shape != current_mask.shape:
            return current_mask
        blended = alpha * current_mask.astype(np.float32) + (1 - alpha) * previous_mask.astype(np.float32)
        return (blended > 0.5).astype(np.uint8)

    def refine_edges(self, mask: NDArray[np.uint8], frame: NDArray[np.uint8], edge_width: int = 10) -> NDArray[np.uint8]:
        if mask is None or frame is None:
            return mask
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        mask_dilated = cv2.dilate(mask * 255, self._morph_kernel, iterations=edge_width // self.smoothing_kernel_size)
        mask_eroded = cv2.erode(mask * 255, self._morph_kernel, iterations=edge_width // self.smoothing_kernel_size)
        edge_band = ((mask_dilated > 0) & (mask_eroded == 0)).astype(np.uint8)
        refined = mask.copy()
        strong_edges = (edges > 100) & (edge_band > 0)
        refined_float = cv2.GaussianBlur(refined.astype(np.float32) * 255, (5, 5), 0)
        refined = (refined_float > 127).astype(np.uint8)
        return refined

    def create_soft_mask(self, mask: NDArray[np.uint8], feather_radius: int = 5) -> NDArray[np.float32]:
        if feather_radius <= 0:
            return mask.astype(np.float32)
        kernel_size = feather_radius * 2 + 1
        soft = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), feather_radius / 3)
        return np.clip(soft, 0, 1)

    def combine_masks(self, masks: list[NDArray[np.uint8]], mode: str = "union") -> NDArray[np.uint8]:
        if not masks:
            return None
        if len(masks) == 1:
            return masks[0]
        result = masks[0].astype(np.uint8)
        for mask in masks[1:]:
            if mode == "union":
                result = np.maximum(result, mask)
            elif mode == "intersection":
                result = np.minimum(result, mask)
            elif mode == "xor":
                result = np.logical_xor(result > 0, mask > 0).astype(np.uint8)
        return result

    def mask_to_bbox(self, mask: NDArray[np.uint8]) -> Optional[Tuple[int, int, int, int]]:
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if not rows.any() or not cols.any():
            return None
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def calculate_mask_overlap(self, mask1: NDArray[np.uint8], mask2: NDArray[np.uint8]) -> float:
        if mask1.shape != mask2.shape:
            return 0.0
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        union = np.sum((mask1 > 0) | (mask2 > 0))
        if union == 0:
            return 0.0
        return intersection / union
