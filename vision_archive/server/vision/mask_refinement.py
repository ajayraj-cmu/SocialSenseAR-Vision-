"""Mask edge refinement — bilateral filter + morphology + GrabCut.

Extracted from fastsam_segmenter.py (originally sam_gemini_voice.py line 3180).
All functions are stateless — take frame + mask, return refined mask.
Every parameter, kernel size, and threshold is identical to the original.

SAFE TO EDIT: Changes here only affect mask quality, not the protobuf contract.
"""

import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def refine_mask_edges(frame: np.ndarray, mask: np.ndarray, use_grabcut: bool = True) -> np.ndarray | None:
    """Refine mask edges using bilateral filtering and morphological operations.

    Exact copy from sam_gemini_voice.py line 3180.

    Args:
        frame: BGR uint8 image.
        mask: float32 mask (0.0-1.0).
        use_grabcut: Whether to run GrabCut refinement (expensive, ~400ms per mask).

    Returns:
        Refined float32 mask, or None on failure.
    """
    if mask is None or frame is None:
        return mask

    try:
        t0 = time.perf_counter()
        mask_u8 = (mask * 255).astype(np.uint8)

        # Bilateral filter
        t_bilateral = time.perf_counter()
        mask_filtered = cv2.bilateralFilter(mask_u8, 5, 50, 50)
        bilateral_ms = (time.perf_counter() - t_bilateral) * 1000

        # Morphological opening + closing
        t_morph = time.perf_counter()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_opened = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel)
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
        morph_ms = (time.perf_counter() - t_morph) * 1000

        # GrabCut refinement (optional — ~400ms per mask, too slow for realtime)
        grabcut_ms = 0.0
        if use_grabcut:
            t_gc = time.perf_counter()
            mask_closed = _grabcut_refinement(frame, mask_closed)
            grabcut_ms = (time.perf_counter() - t_gc) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Refine: bilateral={bilateral_ms:.1f}ms morph={morph_ms:.1f}ms "
            f"grabcut={grabcut_ms:.1f}ms total={total_ms:.1f}ms"
        )

        return mask_closed.astype(np.float32) / 255.0

    except Exception:
        return None


def _grabcut_refinement(frame: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """GrabCut-based mask refinement.

    Exact copy from sam_gemini_voice.py line 3210.

    Args:
        frame: BGR uint8 image.
        mask_u8: uint8 mask (0-255).

    Returns:
        Refined uint8 mask (0 or 255).
    """
    try:
        h, w = mask_u8.shape

        grabcut_mask = np.zeros(mask_u8.shape, dtype=np.uint8)

        # Core region
        core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core_mask = cv2.erode(mask_u8, core_kernel, iterations=2)
        grabcut_mask[core_mask > 128] = 1

        # Edge region
        edge_mask = mask_u8 - core_mask
        grabcut_mask[edge_mask > 64] = 3

        # Background region
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated_mask = cv2.dilate(mask_u8, bg_kernel, iterations=2)
        grabcut_mask[dilated_mask == 0] = 0

        if np.sum(grabcut_mask == 1) < 100:
            return mask_u8

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(frame, grabcut_mask, None, bgd_model, fgd_model,
                        1, cv2.GC_INIT_WITH_MASK)
            refined_mask = np.where(
                (grabcut_mask == 1) | (grabcut_mask == 3), 255, 0
            ).astype(np.uint8)
            return refined_mask
        except Exception:
            return mask_u8

    except Exception:
        return mask_u8


def fallback_morphology(mask: np.ndarray) -> np.ndarray:
    """Fallback refinement when GrabCut fails.

    Exact copy: np.ones((3,3)) kernel + MORPH_CLOSE only.

    Args:
        mask: binary mask (bool or float32).

    Returns:
        float32 mask (0.0-1.0).
    """
    mask_u8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8.astype(np.float32) / 255.0
