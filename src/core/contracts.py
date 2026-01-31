"""Core data contracts for the Perceptual Modulation Engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray

class Modality(Enum):
    VISUAL = auto()
    AUDIO = auto()
    BOTH = auto()

class VisualOperation(Enum):
    BRIGHTNESS_ATTENUATION = "brightness_attenuation"
    COLOR_TEMPERATURE_SHIFT = "color_temperature_shift"
    SATURATION_REDUCTION = "saturation_reduction"
    TEXTURE_SIMPLIFICATION = "texture_simplification"
    EDGE_PRESERVING_BLUR = "edge_preserving_blur"
    OBJECT_REMOVAL = "object_removal"
    GEOMETRY_DISTORTION = "geometry_distortion"

class AudioOperation(Enum):
    SELECTIVE_MUTING = "selective_muting"
    VOLUME_ATTENUATION = "volume_attenuation"
    FREQUENCY_FILTERING = "frequency_filtering"
    DIRECTIONAL_DAMPENING = "directional_dampening"

class SpatialRelation(Enum):
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far"
    CENTER = "center"
    ABOVE = "above"
    BELOW = "below"

class Magnitude(Enum):
    SLIGHT = 0.25
    MODERATE = 0.5
    STRONG = 0.75
    MAXIMUM = 1.0

class ObjectClass(Enum):
    PERSON = "person"
    FACE = "face"
    SCREEN = "screen"
    LIGHT_SOURCE = "light_source"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    TEXT = "text"
    HAND = "hand"
    UNKNOWN = "unknown"

@dataclass
class BoundingRegion:
    """Bounding region for a segmented object."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    @property
    def area(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, other: BoundingRegion) -> float:
        """Calculate Intersection over Union with another region."""
        x_left = max(self.x_min, other.x_min)
        y_top = max(self.y_min, other.y_min)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

@dataclass
class SegmentedObject:
    """A segmented object from the scene."""
    stable_id: str
    mask: NDArray[np.uint8]
    confidence: float
    bounding_region: BoundingRegion
    class_label: ObjectClass = ObjectClass.UNKNOWN
    depth_estimate: Optional[float] = None
    saliency_score: float = 0.5
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    frames_tracked: int = 1
    smoothed_mask: Optional[NDArray[np.uint8]] = None

    def __post_init__(self):
        if self.smoothed_mask is None:
            self.smoothed_mask = self.mask.copy()

@dataclass
class AudioSource:
    """An identified audio source in the scene."""
    source_id: str
    azimuth_degrees: float
    elevation_degrees: float
    distance_estimate: Optional[float] = None
    confidence: float = 0.5
    frequency_profile: Optional[NDArray[np.float32]] = None
    bound_object_id: Optional[str] = None

@dataclass
class AudioVisualBinding:
    """Binding between an audio source and a visual entity."""
    audio_source_id: str
    visual_object_id: str
    confidence: float
    binding_type: str = "spatial"
    angular_offset_degrees: float = 0.0

@dataclass
class TransformParameters:
    """Parameters for a transformation operation."""
    brightness_factor: float = 1.0
    saturation_factor: float = 1.0
    color_temperature_shift: float = 0.0
    blur_radius: float = 0.0
    texture_simplification: float = 0.0
    volume_factor: float = 1.0
    frequency_filter: Optional[Dict[str, Any]] = None
    directional_dampen: float = 0.0
    transition_duration_seconds: float = 0.5

@dataclass
class TransformOperation:
    """A structured transformation operation."""
    operation_id: str
    target_ids: List[str]
    modality: Modality
    operation: VisualOperation | AudioOperation
    parameters: TransformParameters
    transition_time_seconds: float = 0.5
    start_timestamp: float = 0.0
    is_active: bool = True
    progress: float = 0.0
    original_state: Optional[Dict[str, Any]] = None

@dataclass
class SafetyConstraints:
    """Safety constraints for sensory-safe operation."""
    max_brightness_delta: float = 0.15
    max_volume_delta: float = 0.2
    max_saturation_delta: float = 0.15
    max_blur_delta: float = 5.0
    min_transition_seconds: float = 0.3
    min_brightness: float = 0.1
    min_volume: float = 0.05
    max_blur_radius: float = 30.0
    current_sensory_load: float = 0.0
    sensory_load_threshold: float = 0.7

@dataclass
class UserIntent:
    """Parsed user intent from natural language command."""
    raw_command: str
    target_description: str
    target_spatial_relation: Optional[SpatialRelation] = None
    target_class: Optional[ObjectClass] = None
    resolved_object_ids: List[str] = field(default_factory=list)
    modality: Modality = Modality.VISUAL
    operation_type: Optional[str] = None
    attribute: Optional[str] = None
    magnitude: Magnitude = Magnitude.MODERATE
    is_valid: bool = True
    is_safe: bool = True
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None
    rejection_reason: Optional[str] = None

@dataclass
class FrameData:
    """Synchronized frame data for a single pipeline iteration."""
    frame_id: int
    timestamp_ms: float
    rgb_frame: NDArray[np.uint8]
    audio_chunk: Optional[NDArray[np.float32]] = None
    segmented_objects: List[SegmentedObject] = field(default_factory=list)
    depth_map: Optional[NDArray[np.float32]] = None
    audio_sources: List[AudioSource] = field(default_factory=list)
    bindings: List[AudioVisualBinding] = field(default_factory=list)
    pending_operations: List[TransformOperation] = field(default_factory=list)

@dataclass
class PipelineState:
    """Complete state of the perceptual modulation pipeline."""
    current_frame_id: int = 0
    current_timestamp_ms: float = 0.0
    object_registry: Dict[str, SegmentedObject] = field(default_factory=dict)
    active_operations: List[TransformOperation] = field(default_factory=list)
    active_bindings: List[AudioVisualBinding] = field(default_factory=list)
    safety: SafetyConstraints = field(default_factory=SafetyConstraints)
    is_recording: bool = False
    is_processing_recording: bool = False
    last_frame_latency_ms: float = 0.0
    last_audio_latency_ms: float = 0.0
    frames_processed: int = 0
    consecutive_failures: int = 0
    last_failure_reason: Optional[str] = None

@dataclass
class SegmentationResult:
    """Result from the segmentation module."""
    objects: List[SegmentedObject]
    inference_time_ms: float
    model_confidence: float
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class TrackingResult:
    """Result from the object tracking module."""
    updated_objects: List[SegmentedObject]
    new_object_ids: List[str]
    lost_object_ids: List[str]
    reassigned_ids: List[Tuple[str, str]]
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class DepthResult:
    """Result from the depth estimation module."""
    depth_map: NDArray[np.float32]
    source: str
    inference_time_ms: float
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class TransformResult:
    """Result from applying a transformation."""
    modified_frame: Optional[NDArray[np.uint8]] = None
    modified_audio: Optional[NDArray[np.float32]] = None
    was_constrained: bool = False
    constraint_details: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class PipelineOutput:
    """Final output from a single pipeline iteration."""
    frame_id: int
    timestamp_ms: float
    output_frame: NDArray[np.uint8]
    output_audio: Optional[NDArray[np.float32]] = None
    total_latency_ms: float = 0.0
    video_latency_ms: float = 0.0
    audio_latency_ms: float = 0.0
    latency_budget_exceeded: bool = False
    safety_constraints_applied: bool = False
    success: bool = True
    fallback_to_passthrough: bool = False
    error_message: Optional[str] = None
