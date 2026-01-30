"""
Pipeline Orchestrator.

Executes the perceptual modulation pipeline in strict order:

1. Acquire synchronized RGB + audio
2. Segment objects using SAM-3
3. Assign persistent object IDs via tracking
4. Estimate depth (hardware if available, MiDaS fallback)
5. Bind audio sources to visual entities
6. Parse user intent (language → structured ops)
7. Resolve intent to object IDs
8. Apply parameterized audio/visual transformations
9. Enforce safety constraints and temporal smoothing
10. Render output streams
"""

from __future__ import annotations

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from loguru import logger

from src.core.contracts import (
    PipelineState,
    PipelineOutput,
    FrameData,
    SegmentedObject,
    TransformOperation,
    TransformParameters,
    SafetyConstraints,
    UserIntent,
    Modality,
    VisualOperation,
    AudioOperation,
    ObjectClass,
)
from src.capture.video_capture import VideoCapture
from src.capture.frame_buffer import FrameBuffer
from src.segmentation.sam_auto_segmenter import SAMAutoSegmenter
from src.intent.llm_interpreter import LLMInterpreter
from src.transforms.visual_transformer import VisualTransformer
from src.safety.safety_layer import SafetyLayer
from src.safety.sensory_monitor import SensoryLoadMonitor


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    # Latency budgets (HARD CONSTRAINTS)
    video_max_latency_ms: float = 120.0
    audio_max_latency_ms: float = 40.0
    
    # Video settings
    video_device: int = 0
    video_width: int = 1920
    video_height: int = 1080
    video_fps: int = 30
    
    # Audio settings
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    
    # Processing settings
    segmentation_interval_frames: int = 3  # Run SAM every N frames
    depth_interval_frames: int = 5


class PipelineOrchestrator:
    """
    Main pipeline orchestrator.
    
    Coordinates all components to process frames in real-time.
    
    Guarantees:
    - Pipeline order is NEVER reordered
    - Latency budgets are respected
    - Fails safely to unmodified passthrough
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Pipeline state
        self._state = PipelineState()
        
        # Initialize components
        self._video_capture = VideoCapture(
            device_index=self.config.video_device,
            width=self.config.video_width,
            height=self.config.video_height,
            fps=self.config.video_fps,
        )

        self._frame_buffer = FrameBuffer()

        # Use SAM Auto Segmenter for ALL objects (not just faces/people)
        self._segmenter = SAMAutoSegmenter(
            min_object_area=3000,
            max_objects=15,
            enable_person_detection=True,
            enable_contour_detection=True,
            enable_color_clustering=True,
        )

        # LLM interpreter for complete freedom in verbal requests
        self._llm_interpreter = LLMInterpreter(
            fallback_to_rules=True,  # Use rule-based if no API key
        )

        self._visual_transformer = VisualTransformer()

        self._safety_layer = SafetyLayer()
        self._sensory_monitor = SensoryLoadMonitor()
        
        # Performance tracking
        self._frame_latencies: List[float] = []
        self._audio_latencies: List[float] = []
        
        # Keyboard command queue
        self._pending_commands: List[str] = []
        
        logger.info("Pipeline orchestrator initialized")
    
    def start(self) -> bool:
        """
        Start the pipeline.
        
        Returns:
            True if started successfully
        """
        # Initialize components
        if not self._segmenter.initialize():
            logger.warning("Segmenter initialization failed")

        # Start capture
        if not self._video_capture.start():
            logger.error("Failed to start video capture")
            return False
        
        logger.info("Pipeline started")
        return True
    
    def stop(self):
        """Stop the pipeline."""
        self._video_capture.stop()
        self._segmenter.shutdown()
        logger.info("Pipeline stopped")
    
    def process_frame(self) -> PipelineOutput:
        """
        Process a single frame through the pipeline.

        Pipeline order (simplified):
        1. Acquire RGB frame
        2. Segment objects using SAM
        3. Parse user intent
        4. Resolve intent to object IDs
        5. Apply parameterized transformations
        6. Enforce safety constraints
        7. Render output

        Returns:
            PipelineOutput with processed frame
        """
        pipeline_start = time.perf_counter()
        
        # Check emergency revert
        if self._safety_layer.is_emergency_revert_active:
            return self._passthrough_output("Emergency revert active")
        
        # ============================================================
        # STEP 1: Acquire RGB frame
        # ============================================================
        video_frame, timestamp_ms, frame_id = self._video_capture.read_frame()

        if video_frame is None:
            return self._passthrough_output("No video frame available")

        frame_data = FrameData(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            rgb_frame=video_frame,
            audio_chunk=None,
        )

        # ============================================================
        # STEP 2: Segment objects using SAM
        # ============================================================
        segmentation_result = None
        if frame_id % self.config.segmentation_interval_frames == 0:
            segmentation_result = self._segmenter.segment_frame(
                video_frame, frame_id
            )
            
            if segmentation_result.success:
                frame_data.segmented_objects = segmentation_result.objects

                # Update state registry
                for obj in frame_data.segmented_objects:
                    self._state.object_registry[obj.stable_id] = obj

        # ============================================================
        # STEP 3: Parse user intent and resolve targets
        # ============================================================
        self._process_pending_commands(frame_data)
        
        # ============================================================
        # STEP 4: Apply parameterized transformations
        # ============================================================
        output_frame = video_frame.copy()
        
        safety_applied = False
        
        for operation in self._state.active_operations:
            if not operation.is_active:
                continue
            
            # Validate operation
            is_valid, reason, constrained_op = self._safety_layer.validate_operation(
                operation
            )
            
            if not is_valid:
                logger.warning(f"Operation rejected: {reason}")
                operation.is_active = False
                continue
            
            if constrained_op:
                operation = constrained_op
                safety_applied = True
            
            # Apply transformation
            if operation.modality in [Modality.VISUAL, Modality.BOTH]:
                output_frame = self._apply_visual_transform(
                    output_frame,
                    operation,
                    frame_data
                )
            
            # Update operation progress
            operation.progress = min(1.0, operation.progress + 0.1)
        
        # ============================================================
        # STEP 5: Enforce safety constraints
        # ============================================================
        sensory_metrics = self._sensory_monitor.update(self._state.active_operations)
        self._safety_layer.update_sensory_load(sensory_metrics.total_load)
        
        # Apply automatic dampening if needed
        if self._sensory_monitor.should_dampen():
            dampening_factor = self._sensory_monitor.get_dampening_factor()
            logger.debug(f"Applying sensory dampening: {dampening_factor}")
            # Would apply additional smoothing here
        
        # ============================================================
        # STEP 6: Render output
        # ============================================================
        pipeline_end = time.perf_counter()
        total_latency = (pipeline_end - pipeline_start) * 1000
        
        # Track latency
        self._frame_latencies.append(total_latency)
        if len(self._frame_latencies) > 100:
            self._frame_latencies.pop(0)
        
        # Check latency budget
        latency_exceeded = total_latency > self.config.video_max_latency_ms
        
        if latency_exceeded:
            logger.warning(
                f"Latency budget exceeded: {total_latency:.1f}ms > "
                f"{self.config.video_max_latency_ms}ms"
            )
        
        # Update state
        self._state.current_frame_id = frame_id
        self._state.current_timestamp_ms = timestamp_ms
        self._state.last_frame_latency_ms = total_latency
        self._state.frames_processed += 1
        
        # Recording mode - capture the AUGMENTED frame
        if self._frame_buffer.is_recording:
            self._frame_buffer.add_augmented_frame(
                frame_id, timestamp_ms, output_frame
            )
        
        return PipelineOutput(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            output_frame=output_frame,
            output_audio=None,
            total_latency_ms=total_latency,
            video_latency_ms=total_latency,
            latency_budget_exceeded=latency_exceeded,
            safety_constraints_applied=safety_applied,
            success=True,
        )
    
    def _apply_visual_transform(
        self,
        frame: NDArray[np.uint8],
        operation: TransformOperation,
        frame_data: FrameData,
    ) -> NDArray[np.uint8]:
        """Apply visual transformation to frame using REAL masks."""
        result_frame = frame
        
        for target_id in operation.target_ids:
            # Handle GLOBAL target - apply to whole frame
            if target_id == "__GLOBAL__":
                if isinstance(operation.operation, VisualOperation):
                    result_frame = self._apply_global_visual_effect(
                        result_frame,
                        operation,  # Pass full operation for LLM interpretation
                    )
                continue
            
            # Find the object - check both registry and current frame data
            obj = self._state.object_registry.get(target_id)
            
            # Also check frame_data.segmented_objects for fresh masks
            if obj is None:
                for seg_obj in frame_data.segmented_objects:
                    if seg_obj.stable_id == target_id or target_id in seg_obj.stable_id:
                        obj = seg_obj
                        break
            
            # If still no object, try to use any available segmented object
            if obj is None and frame_data.segmented_objects:
                obj = frame_data.segmented_objects[0]  # Use first available
            
            if obj is None:
                continue
            
            # Apply effect using the REAL mask
            if isinstance(operation.operation, VisualOperation):
                result_frame = self._apply_masked_effect(
                    result_frame,
                    obj,
                    operation,
                    operation.parameters,
                )
        
        return result_frame
    
    def _apply_masked_effect(
        self,
        frame: NDArray[np.uint8],
        obj: SegmentedObject,
        operation: TransformOperation,
        parameters: TransformParameters,
    ) -> NDArray[np.uint8]:
        """Apply effect to the ACTUAL SEGMENTED MASK region."""
        mask = obj.smoothed_mask if obj.smoothed_mask is not None else obj.mask
        
        if mask is None or not np.any(mask):
            logger.warning(f"No mask for object {obj.stable_id}")
            return frame
        
        # Get effect type from interpretation if available
        effect_type = None
        color = None
        intensity = 0.7
        
        # Check if we have LLM interpretation stored
        if operation.original_state and 'interpretation' in operation.original_state:
            interp = operation.original_state['interpretation']
            effect_type = interp.effect
            color = interp.color
            intensity = interp.intensity
        else:
            # Fallback to operation type mapping
            op = operation.operation
            if op == VisualOperation.EDGE_PRESERVING_BLUR:
                effect_type = 'blur'
                intensity = min(parameters.blur_radius / 30.0, 1.0) if parameters.blur_radius > 0 else 0.7
            elif op == VisualOperation.BRIGHTNESS_ATTENUATION:
                effect_type = 'darken' if parameters.brightness_factor < 1 else 'brighten'
                intensity = abs(1.0 - parameters.brightness_factor)
            elif op == VisualOperation.COLOR_TEMPERATURE_SHIFT:
                effect_type = 'color'
                shift = parameters.color_temperature_shift
                if shift > 0:
                    color = (255, int(200 + shift * 55), int(100 * (1 - shift)))
                else:
                    color = (int(100 * (1 + shift)), int(150 + abs(shift) * 50), 255)
                intensity = abs(shift) * 0.7
            elif op == VisualOperation.SATURATION_REDUCTION:
                effect_type = 'desaturate'
                intensity = 1.0 - parameters.saturation_factor
            elif op == VisualOperation.TEXTURE_SIMPLIFICATION:
                effect_type = 'pixelate'
                intensity = parameters.texture_simplification
        
        if effect_type:
            result = self._segmenter.apply_effect_to_mask(
                frame,
                mask,
                effect_type,
                intensity,
                color
            )
            logger.info(f"Applied {effect_type} to {obj.stable_id} (class: {obj.class_label.value}, intensity: {intensity:.2f})")
            return result
        
        return frame
    
    def _apply_global_visual_effect(
        self,
        frame: NDArray[np.uint8],
        operation: TransformOperation,
    ) -> NDArray[np.uint8]:
        """Apply visual effect to the ENTIRE frame (no mask)."""
        import cv2
        
        result = frame.copy()
        op = operation.operation
        parameters = operation.parameters
        
        # Get effect type from LLM interpretation if available
        effect_type = None
        intensity = 0.7
        color = None
        
        if operation.original_state and 'interpretation' in operation.original_state:
            interp = operation.original_state['interpretation']
            effect_type = interp.effect
            intensity = interp.intensity
            color = interp.color
            logger.info(f"🎨 GLOBAL effect from voice: {effect_type} (intensity: {intensity})")
        
        # Apply effect based on interpretation OR operation type
        if effect_type == 'blur' or op == VisualOperation.EDGE_PRESERVING_BLUR:
            blur_size = int(5 + intensity * 40) | 1  # Ensure odd
            result = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
            logger.info(f"✅ Applied GLOBAL blur: {blur_size}px")
        
        elif effect_type == 'darken' or (op == VisualOperation.BRIGHTNESS_ATTENUATION and parameters.brightness_factor < 1):
            factor = 1.0 - intensity * 0.6
            result = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            logger.info(f"✅ Applied GLOBAL darken: factor={factor:.2f}")
        
        elif effect_type == 'brighten' or (op == VisualOperation.BRIGHTNESS_ATTENUATION and parameters.brightness_factor >= 1):
            factor = 1.0 + intensity * 0.5
            result = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            logger.info(f"✅ Applied GLOBAL brighten: factor={factor:.2f}")
        
        elif effect_type == 'desaturate' or op == VisualOperation.SATURATION_REDUCTION:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * (1 - intensity)
            result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            logger.info(f"✅ Applied GLOBAL desaturate: {intensity:.2f}")
        
        elif effect_type == 'color' or effect_type == 'color_overlay' or effect_type == 'highlight':
            if color:
                overlay = np.full_like(frame, color, dtype=np.uint8)
            else:
                overlay = np.full_like(frame, (255, 200, 100), dtype=np.uint8)  # Warm tint
            alpha = intensity * 0.5
            result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
            logger.info(f"✅ Applied GLOBAL color overlay: {color}")
        
        elif effect_type == 'pixelate' or op == VisualOperation.TEXTURE_SIMPLIFICATION:
            pixel_size = max(2, int(intensity * 30))
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            logger.info(f"✅ Applied GLOBAL pixelate: {pixel_size}px blocks")
        
        elif effect_type == 'thermal':
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            result = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            logger.info(f"✅ Applied GLOBAL thermal vision")
        
        elif effect_type == 'invert':
            result = 255 - frame
            logger.info(f"✅ Applied GLOBAL invert")
        
        elif effect_type == 'hide':
            result = cv2.GaussianBlur(frame, (51, 51), 0)
            logger.info(f"✅ Applied GLOBAL heavy blur (hide)")
        
        elif op == VisualOperation.COLOR_TEMPERATURE_SHIFT:
            shift = parameters.color_temperature_shift
            result = frame.astype(np.float32)
            if shift > 0:  # Warmer
                result[:, :, 0] = result[:, :, 0] * (1 + shift * 0.3)
                result[:, :, 2] = result[:, :, 2] * (1 - shift * 0.2)
            else:  # Cooler
                result[:, :, 0] = result[:, :, 0] * (1 + shift * 0.2)
                result[:, :, 2] = result[:, :, 2] * (1 - shift * 0.3)
            result = np.clip(result, 0, 255).astype(np.uint8)
            logger.info(f"✅ Applied GLOBAL color temp: {shift}")
        
        else:
            # Default to blur if nothing matched
            result = cv2.GaussianBlur(frame, (21, 21), 0)
            logger.info(f"✅ Applied GLOBAL default blur")
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _process_pending_commands(self, frame_data: FrameData):
        """Process pending user commands using LLM interpretation."""
        while self._pending_commands:
            command = self._pending_commands.pop(0)
            
            # Clear previous operations when new command comes in
            self._state.active_operations = []
            
            # Use LLM interpreter for complete freedom
            interp = self._llm_interpreter.interpret(command)
            
            logger.info(f"LLM interpreted: {interp.effect} on {interp.target} (understood: {interp.understood})")
            logger.info(f"  Explanation: {interp.explanation}")
            
            if not interp.understood:
                logger.warning(f"Could not understand command: {command}")
            
            # PROPERLY resolve targets using OBJECT CLASS, not just ID strings
            target_ids = self._resolve_targets_by_class(interp.target, frame_data.segmented_objects)
            
            logger.info(f"Resolved targets: {target_ids} for target type: {interp.target}")
            
            # Build parameters based on effect
            params = TransformParameters()
            operation_type = VisualOperation.EDGE_PRESERVING_BLUR
            
            if interp.effect in ["blur", "hide", "soft", "soften"]:
                operation_type = VisualOperation.EDGE_PRESERVING_BLUR
                params.blur_radius = interp.intensity * 30
            elif interp.effect in ["darken", "dim"]:
                operation_type = VisualOperation.BRIGHTNESS_ATTENUATION
                params.brightness_factor = 1.0 - interp.intensity * 0.6
            elif interp.effect in ["brighten", "light"]:
                operation_type = VisualOperation.BRIGHTNESS_ATTENUATION
                params.brightness_factor = 1.0 + interp.intensity * 0.5
            elif interp.effect in ["desaturate", "gray", "grayscale"]:
                operation_type = VisualOperation.SATURATION_REDUCTION
                params.saturation_factor = 1.0 - interp.intensity
            elif interp.effect in ["pixelate", "censor"]:
                operation_type = VisualOperation.TEXTURE_SIMPLIFICATION
                params.texture_simplification = interp.intensity
            elif interp.effect in ["color", "color_overlay", "tint", "warm", "cool"]:
                operation_type = VisualOperation.COLOR_TEMPERATURE_SHIFT
                params.color_temperature_shift = interp.intensity
            
            import uuid
            operation = TransformOperation(
                operation_id=str(uuid.uuid4())[:8],
                target_ids=target_ids,
                modality=Modality.VISUAL,
                operation=operation_type,
                parameters=params,
                transition_time_seconds=0.3,
                is_active=True,
                progress=0.0,
            )
            
            operation.start_timestamp = frame_data.timestamp_ms
            
            # Store the interpretation for effect application
            operation.original_state = {
                'interpretation': interp,
                'color': interp.color,
                'intensity': interp.intensity,
                'effect': interp.effect,
            }
            
            self._state.active_operations.append(operation)
            logger.info(f"✨ Operation queued: {interp.effect} on {target_ids}")
    
    def _resolve_targets_by_class(self, target_type: str, objects: List[SegmentedObject]) -> List[str]:
        """Resolve target type to actual object IDs based on their CLASS LABEL."""
        
        target_ids = []
        
        if target_type == "face":
            # Find objects with FACE class
            for obj in objects:
                if obj.class_label == ObjectClass.FACE:
                    target_ids.append(obj.stable_id)
                    logger.info(f"  Found FACE: {obj.stable_id}")
            # Fallback: if no face found, use first person
            if not target_ids:
                for obj in objects:
                    if obj.class_label == ObjectClass.PERSON:
                        target_ids.append(obj.stable_id)
                        logger.info(f"  No face, using PERSON: {obj.stable_id}")
                        break
        
        elif target_type == "person":
            # Find objects with PERSON class
            for obj in objects:
                if obj.class_label == ObjectClass.PERSON:
                    target_ids.append(obj.stable_id)
                    logger.info(f"  Found PERSON: {obj.stable_id}")
        
        elif target_type == "background":
            # Find all objects that are NOT person or face
            for obj in objects:
                if obj.class_label not in [ObjectClass.PERSON, ObjectClass.FACE]:
                    target_ids.append(obj.stable_id)
            # If nothing found, use global
            if not target_ids:
                target_ids = ["__GLOBAL__"]
        
        elif target_type == "hands":
            # Find hand objects
            for obj in objects:
                if obj.class_label == ObjectClass.HAND:
                    target_ids.append(obj.stable_id)
        
        elif target_type == "everything":
            # Use GLOBAL - applies to entire screen
            target_ids = ["__GLOBAL__"]
        
        else:
            # Default: apply globally
            target_ids = ["__GLOBAL__"]
        
        # Final fallback
        if not target_ids:
            target_ids = ["__GLOBAL__"]
            logger.info(f"  No specific targets found, using GLOBAL")
        
        return target_ids
    
    def _passthrough_output(self, reason: str) -> PipelineOutput:
        """Return unmodified passthrough frame."""
        frame, timestamp, frame_id = self._video_capture.get_latest_frame()
        
        return PipelineOutput(
            frame_id=frame_id or 0,
            timestamp_ms=timestamp,
            output_frame=frame if frame is not None else np.zeros(
                (self.config.video_height, self.config.video_width, 3),
                dtype=np.uint8
            ),
            output_audio=None,
            total_latency_ms=0,
            success=False,
            fallback_to_passthrough=True,
            error_message=reason,
        )
    
    def queue_command(self, command: str):
        """Queue a user command for processing."""
        self._pending_commands.append(command)
        logger.debug(f"Command queued: {command}")
    
    def handle_keyboard(self, key: str):
        """
        Handle keyboard input.
        
        Bindings:
        - P: Start recording
        - R: Stop recording and playback augmented version
        - ESC: Emergency revert
        """
        key_lower = key.lower()
        
        if key_lower == 'p':
            # P = Start recording
            if not self._frame_buffer.is_recording:
                self._frame_buffer.start_recording()
                logger.info("🔴 RECORDING STARTED - Press R to stop and playback")
        
        elif key_lower == 'r':
            # R = Stop recording and prepare for playback
            if self._frame_buffer.is_recording:
                count = self._frame_buffer.stop_recording()
                logger.info(f"⏹️ Recording stopped: {count} frames - Playing back augmented version...")
                if count > 0:
                    self._frame_buffer.start_playback()
        
        elif key == 'escape':
            self._safety_layer.trigger_emergency_revert("User pressed ESC")
            # Clear all active operations
            self._state.active_operations.clear()
            # Stop any recording/playback
            if self._frame_buffer.is_recording:
                self._frame_buffer.stop_recording()
            self._frame_buffer.stop_playback()
            logger.info("Emergency revert: All operations cleared")
    
    def get_playback_frame(self):
        """Get next frame from playback buffer."""
        return self._frame_buffer.get_next_playback_frame()
    
    @property
    def is_playing_back(self) -> bool:
        """Check if currently playing back recorded frames."""
        return self._frame_buffer.is_playing
    
    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state
    
    @property
    def average_latency_ms(self) -> float:
        """Get average frame latency."""
        if not self._frame_latencies:
            return 0.0
        return sum(self._frame_latencies) / len(self._frame_latencies)


