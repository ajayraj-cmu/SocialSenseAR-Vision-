"""
Segmentation module using SAM (Segment Anything Model).

Responsibilities:
- Real-time object segmentation
- Mask temporal smoothing
- Confidence scoring
- Morphological cleaning
"""

from .sam_auto_segmenter import SAMAutoSegmenter
from .mask_processor import MaskProcessor


