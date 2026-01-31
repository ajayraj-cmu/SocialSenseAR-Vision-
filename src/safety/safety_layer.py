"""Safety Layer - Core Guardrails."""
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import time
from loguru import logger
from src.core.contracts import SafetyConstraints, TransformOperation, TransformParameters, VisualOperation, AudioOperation

@dataclass
class ParameterHistory:
    """Tracks parameter changes over time."""
    parameter_name: str
    values: List[Tuple[float, float]] = field(default_factory=list)
    max_history: int = 100

    def add(self, value: float, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        self.values.append((timestamp, value))
        if len(self.values) > self.max_history:
            self.values.pop(0)

    def get_rate_of_change(self, window_seconds: float = 1.0) -> float:
        """Calculate rate of change over the last window_seconds."""
        if len(self.values) < 2:
            return 0.0
        current_time = time.time()
        cutoff = current_time - window_seconds
        recent = [(t, v) for t, v in self.values if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        time_diff = recent[-1][0] - recent[0][0]
        if time_diff <= 0:
            return 0.0
        value_diff = abs(recent[-1][1] - recent[0][1])
        return value_diff / time_diff

class SafetyLayer:
    """Core safety layer for sensory-safe operation."""

    def __init__(self, constraints: Optional[SafetyConstraints] = None):
        self.constraints = constraints or SafetyConstraints()
        self._param_history: Dict[str, ParameterHistory] = {}
        self._emergency_revert_active: bool = False
        self._last_safe_state: Optional[Dict] = None
        self._violation_count: int = 0
        self._last_violation_time: float = 0.0
        self._violation_log: List[str] = []

    def validate_operation(self, operation: TransformOperation) -> Tuple[bool, Optional[str], Optional[TransformOperation]]:
        """Validate a transform operation against safety constraints."""
        if isinstance(operation.operation, VisualOperation):
            if operation.operation in [VisualOperation.OBJECT_REMOVAL, VisualOperation.GEOMETRY_DISTORTION]:
                return (False, f"Operation '{operation.operation.value}' is not allowed for safety reasons.", None)
        if operation.transition_time_seconds < self.constraints.min_transition_seconds:
            constrained = self._copy_operation(operation)
            constrained.transition_time_seconds = self.constraints.min_transition_seconds
            logger.info(f"Transition time constrained from {operation.transition_time_seconds}s to {self.constraints.min_transition_seconds}s")
            return (True, None, constrained)
        params_valid, param_reason, constrained_params = self._validate_parameters(operation.parameters)
        if not params_valid:
            return (False, param_reason, None)
        if constrained_params != operation.parameters:
            constrained = self._copy_operation(operation)
            constrained.parameters = constrained_params
            return (True, None, constrained)
        return (True, None, None)

    def _validate_parameters(self, params: TransformParameters) -> Tuple[bool, Optional[str], TransformParameters]:
        """Validate and potentially constrain transform parameters."""
        constrained = TransformParameters(
            brightness_factor=params.brightness_factor, saturation_factor=params.saturation_factor,
            color_temperature_shift=params.color_temperature_shift, blur_radius=params.blur_radius,
            texture_simplification=params.texture_simplification, volume_factor=params.volume_factor,
            frequency_filter=params.frequency_filter, directional_dampen=params.directional_dampen,
            transition_duration_seconds=params.transition_duration_seconds,
        )
        was_constrained = False
        if params.brightness_factor < self.constraints.min_brightness:
            constrained.brightness_factor = self.constraints.min_brightness
            was_constrained = True
            logger.info(f"Brightness constrained to minimum {self.constraints.min_brightness}")
        if params.volume_factor < self.constraints.min_volume:
            constrained.volume_factor = self.constraints.min_volume
            was_constrained = True
            logger.info(f"Volume constrained to minimum {self.constraints.min_volume}")
        if params.blur_radius > self.constraints.max_blur_radius:
            constrained.blur_radius = self.constraints.max_blur_radius
            was_constrained = True
            logger.info(f"Blur radius constrained to maximum {self.constraints.max_blur_radius}px")
        if params.transition_duration_seconds < self.constraints.min_transition_seconds:
            constrained.transition_duration_seconds = self.constraints.min_transition_seconds
            was_constrained = True
        return (True, None, constrained if was_constrained else params)

    def check_rate_limits(self, param_name: str, current_value: float, target_value: float, delta_time_seconds: float) -> Tuple[float, bool]:
        """Check if a parameter change respects rate limits."""
        max_delta_per_second = self._get_max_delta_for_param(param_name)
        if max_delta_per_second is None:
            return (target_value, False)
        max_change = max_delta_per_second * delta_time_seconds
        actual_change = target_value - current_value
        if abs(actual_change) <= max_change:
            return (target_value, False)
        if actual_change > 0:
            safe_target = current_value + max_change
        else:
            safe_target = current_value - max_change
        self._log_violation(f"Rate limit exceeded for {param_name}: requested {actual_change:.3f}, allowed {max_change:.3f}")
        return (safe_target, True)

    def _get_max_delta_for_param(self, param_name: str) -> Optional[float]:
        """Get maximum change rate for a parameter."""
        rates = {
            'brightness': self.constraints.max_brightness_delta,
            'brightness_factor': self.constraints.max_brightness_delta,
            'volume': self.constraints.max_volume_delta,
            'volume_factor': self.constraints.max_volume_delta,
            'saturation': self.constraints.max_saturation_delta,
            'saturation_factor': self.constraints.max_saturation_delta,
            'blur': self.constraints.max_blur_delta,
            'blur_radius': self.constraints.max_blur_delta,
        }
        return rates.get(param_name)

    def trigger_emergency_revert(self, reason: str):
        """Trigger emergency revert to unmodified passthrough."""
        self._emergency_revert_active = True
        self._log_violation(f"EMERGENCY REVERT: {reason}")
        logger.warning(f"Emergency revert triggered: {reason}")

    def clear_emergency_revert(self):
        """Clear emergency revert state."""
        self._emergency_revert_active = False
        logger.info("Emergency revert cleared")

    @property
    def is_emergency_revert_active(self) -> bool:
        return self._emergency_revert_active

    def update_sensory_load(self, load: float):
        """Update current sensory load metric."""
        self.constraints.current_sensory_load = load
        if load > self.constraints.sensory_load_threshold:
            logger.warning(f"Sensory load ({load:.2f}) exceeds threshold ({self.constraints.sensory_load_threshold})")

    def _log_violation(self, message: str):
        """Log a constraint violation."""
        self._violation_count += 1
        self._last_violation_time = time.time()
        self._violation_log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if len(self._violation_log) > 100:
            self._violation_log.pop(0)
        logger.warning(f"Safety violation: {message}")
        if self._violation_count > 10:
            recent_violations = sum(1 for t in [self._last_violation_time] if time.time() - t < 5.0)
            if recent_violations > 5:
                self.trigger_emergency_revert("Too many violations in short time")

    def _copy_operation(self, operation: TransformOperation) -> TransformOperation:
        """Create a copy of an operation."""
        return TransformOperation(
            operation_id=operation.operation_id, target_ids=operation.target_ids.copy(),
            modality=operation.modality, operation=operation.operation, parameters=operation.parameters,
            transition_time_seconds=operation.transition_time_seconds, start_timestamp=operation.start_timestamp,
            is_active=operation.is_active, progress=operation.progress, original_state=operation.original_state,
        )

    def get_violation_log(self) -> List[str]:
        """Get recent violation log."""
        return self._violation_log.copy()

    def reset(self):
        """Reset safety layer state."""
        self._param_history.clear()
        self._emergency_revert_active = False
        self._violation_count = 0
        self._violation_log.clear()
        logger.info("Safety layer reset")
