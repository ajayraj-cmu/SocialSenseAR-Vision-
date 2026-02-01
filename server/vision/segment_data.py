"""Shared segment data class used by all segmenters."""

import numpy as np


class SegmentData:
    """One detected segment."""
    __slots__ = (
        "mask", "label", "asset_class", "confidence",
        "bbox", "center_x", "center_y",
        "mask_width", "mask_height", "rle_mask",
        "track_id", "emotion",
    )

    def __init__(self, mask: np.ndarray, label: str = "", confidence: float = 0.0):
        self.mask = mask
        self.label = label
        self.asset_class = ""
        self.confidence = confidence
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.center_x = 0.0
        self.center_y = 0.0
        self.mask_width = 0
        self.mask_height = 0
        self.rle_mask = b""
        self.track_id = ""
        self.emotion = None
