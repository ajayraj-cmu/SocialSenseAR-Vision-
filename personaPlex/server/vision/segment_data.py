"""Shared segment data class used by all segmenters."""

import numpy as np
from typing import Optional


class EffectData:
    """Effect metadata for a segment (voice agent effects)."""
    __slots__ = ("effect_type", "intensity", "color_hex", "params")

    def __init__(
        self,
        effect_type: str = "none",
        intensity: float = 1.0,
        color_hex: str = "",
        params: Optional[dict] = None,
    ):
        self.effect_type = effect_type  # "blur", "dim", "pixelate", "highlight", "outline", "none"
        self.intensity = intensity  # 0.0-1.0
        self.color_hex = color_hex  # hex color for outline/highlight
        self.params = params or {}  # additional parameters


class SegmentData:
    """One detected segment."""
    __slots__ = (
        "mask", "label", "asset_class", "confidence",
        "bbox", "center_x", "center_y",
        "mask_width", "mask_height", "rle_mask",
        "track_id", "emotion", "effect",
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
        self.effect: Optional[EffectData] = None  # voice agent effects
