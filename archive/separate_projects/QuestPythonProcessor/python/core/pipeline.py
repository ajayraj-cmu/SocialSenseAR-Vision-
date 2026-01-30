"""
Main processing pipeline.

Orchestrates the video capture, ML processing, effect rendering, and display.
"""
import signal
import time
from typing import Optional
from collections import deque
import cv2
import numpy as np

from .transition import TransitionEffect
from sources.base import BaseSource
from processors.base import BaseProcessor
from effects.base import BaseEffect
from controls.base import BaseControl
from controls.keyboard import KeyboardControl
from controls.quest_pinch import QuestPinchControl
from ui.base import BaseUI
from ui.overlay_panels import render_overlay


class PipelineProfiler:
    """Detailed profiling for each pipeline stage."""

    def __init__(self, window_size=60):
        self.window_size = window_size
        self.timings = {
            'frame_wait': deque(maxlen=window_size),
            'yolo_process': deque(maxlen=window_size),
            'effect_apply': deque(maxlen=window_size),
            'ui_send': deque(maxlen=window_size),
            'total': deque(maxlen=window_size),
        }
        self.last_log = time.time()
        self.frame_count = 0

    def record(self, stage: str, duration_ms: float):
        self.timings[stage].append(duration_ms)

    def log_if_ready(self, interval=2.0):
        self.frame_count += 1
        now = time.time()
        if now - self.last_log >= interval:
            avgs = {}
            for stage, times in self.timings.items():
                if times:
                    avgs[stage] = sum(times) / len(times)
                else:
                    avgs[stage] = 0

            fps = self.frame_count / interval
            print(f"\n[PIPELINE] wait:{avgs['frame_wait']:.1f}ms | yolo:{avgs['yolo_process']:.1f}ms | effect:{avgs['effect_apply']:.1f}ms | send:{avgs['ui_send']:.1f}ms | total:{avgs['total']:.1f}ms | {fps:.1f}fps")

            # Identify bottleneck
            bottleneck = max(avgs.items(), key=lambda x: x[1] if x[0] != 'total' else 0)
            if bottleneck[1] > 5:
                print(f"[BOTTLENECK] {bottleneck[0]}: {bottleneck[1]:.1f}ms")

            self.last_log = now
            self.frame_count = 0


class Pipeline:
    """Main video processing pipeline.

    Coordinates all components:
    - Source: Video input (Quest, webcam, etc.)
    - Processor: ML model (YOLO, etc.)
    - Effect: Visual effect (focus, blur, etc.)
    - UI: Display backend (OpenCV, etc.)
    - Controls: Input handling (keyboard, gestures)

    Usage:
        pipeline = Pipeline(source, processor, effect, ui, config)
        pipeline.run()
    """

    def __init__(self, source: BaseSource, processor: BaseProcessor,
                 effect: BaseEffect, ui: BaseUI, config):
        """Initialize pipeline with components.

        Args:
            source: Video source
            processor: ML processor
            effect: Visual effect
            ui: UI backend
            config: Configuration object
        """
        self.source = source
        self.processor = processor
        self.effect = effect
        self.ui = ui
        self.config = config

        # Wire up Quest TCP source and UI if both are used
        if hasattr(ui, 'set_source') and hasattr(source, 'send_processed_frame'):
            ui.set_source(source)

        # State
        self.transition = TransitionEffect(
            duration=getattr(config, 'transition_duration', 1.0)
        )
        self.running = False
        self.profiling = getattr(config, 'show_profiling', True)

        # Controls
        self.controls = []
        self.keyboard = KeyboardControl(self._on_toggle)

        # Stats
        self.frame_count = 0
        self.start_time = 0

        # Detailed profiler
        self.profiler = PipelineProfiler()

    def _on_toggle(self) -> None:
        """Callback when toggle is triggered."""
        if self.transition.toggle():
            state = "ON" if self.transition.direction == 1 else "OFF"
            print(f"\nProcessing: {state} (animating...)")

    def run(self) -> None:
        """Run the main processing loop."""
        print("Quest Processor - Modular Pipeline")
        print("-" * 40)
        print()

        headless = getattr(self.config, 'headless', False)

        # Start source FIRST so we can show video quickly
        if not self.source.start():
            print("ERROR: Failed to start video source")
            return

        # Setup UI immediately so window appears
        self.ui.setup("Quest Processor (q=quit, p=toggle)")

        if not headless:
            # Create a window immediately, even if the first camera frame is slow.
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Starting camera...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            self.ui.show(placeholder, {'status': 'Starting camera...'})
            cv2.waitKey(1)

        # Show first frame immediately while model loads
        first_frame = None
        for _ in range(10):  # Try a few times to get first frame
            first_frame = self.source.get_frame()
            if first_frame is not None:
                break
            time.sleep(0.05)

        if first_frame is not None:
            # Show raw frame immediately
            display = self.effect.no_effect(first_frame)
            self.ui.show(display, {'status': 'Loading AI model...'})
            cv2.waitKey(1)  # Force window to appear

        # NOW load the processor (this is the slow part)
        print("Loading AI model...")
        self.processor.start()
        print("Model loaded!")

        # Initialize and start audio AFTER window and model are ready
        self.audio_manager = None
        if hasattr(self, 'audio_enabled') and self.audio_enabled and hasattr(self, 'AudioManagerClass'):
            print("Initializing audio processing...")
            self.audio_manager = self.AudioManagerClass(self.config)
            self.audio_manager.start()
            print("Audio processing started!")

        # Start controls (only use Quest pinch control for Quest sources)
        source_name = getattr(self.config, 'source', '')
        if source_name.startswith('quest'):
            pinch = QuestPinchControl(self._on_toggle)
            pinch.start()
            self.controls.append(pinch)

        # Auto-start processing if configured
        if getattr(self.config, 'auto_start', True):
            self.transition.force_on()
            print("Auto-processing enabled")

        # Setup signal handler
        self.running = True
        def signal_handler(_sig, _frame):
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)

        self.frame_count = 0
        self.start_time = time.time()
        last_frame = None
        profile_interval = getattr(self.config, 'profile_interval', 30)
        process_skip = getattr(self.config, 'process_skip', 1)
        display_skip = getattr(self.config, 'display_skip', 1)
        print()
        print("Press 'q' to quit, 'p' to toggle processing, 't' to toggle profiling")
        if source_name.startswith('quest'):
            print("Or PINCH your fingers on the Quest to toggle!")
        print()

        try:
            while self.running:
                t_start = time.time()

                # Get frame (measure wait time)
                t_wait_start = time.time()
                frame_rgb = self.source.get_frame()
                if frame_rgb is None or frame_rgb is last_frame:
                    time.sleep(0.0001)
                    continue
                last_frame = frame_rgb
                t_wait_end = time.time()
                self.profiler.record('frame_wait', (t_wait_end - t_wait_start) * 1000)

                # Update transition
                self.transition.update()

                # Process and apply effect
                t_yolo_start = time.time()
                if self.transition.is_active():
                    if self.frame_count % process_skip == 0:
                        self.processor.process(frame_rgb)
                    result = self.processor.get_result()
                else:
                    result = None
                t_yolo_end = time.time()
                self.profiler.record('yolo_process', (t_yolo_end - t_yolo_start) * 1000)

                # Apply effect
                t_effect_start = time.time()
                if self.transition.is_active():
                    if self.transition.transitioning:
                        frame = self.effect.apply_transition(
                            frame_rgb, result, self.transition.progress
                        )
                    else:
                        frame = self.effect.apply(frame_rgb, result)
                else:
                    frame = self.effect.no_effect(frame_rgb)

                # Note: Overlay is now rendered in UI after scaling for crisp text
                # frame = render_overlay(frame)

                t_effect_end = time.time()
                self.profiler.record('effect_apply', (t_effect_end - t_effect_start) * 1000)

                self.frame_count += 1

                # Display and send to Quest
                t_ui_start = time.time()
                if not headless and self.frame_count % display_skip == 0:
                    # Build stats for UI
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    if self.transition.active:
                        status = "Processing"
                        effect_name = "Focus (Active)"
                    elif self.transition.transitioning:
                        status = "Transitioning"
                        effect_name = f"Focus ({int(self.transition.progress * 100)}%)"
                    else:
                        status = "Idle"
                        effect_name = "None"

                    # Check if person is being tracked (ProcessorResult has left_mask/right_mask)
                    person_tracked = False
                    head_x, head_y = 0.5, 0.3  # Default head position
                    if result is not None and result.has_detection:
                        person_tracked = True
                        h, w = frame.shape[:2]

                        # Use left_center (normalized 0-1) for head position
                        # In stereo mode, left_center is relative to left half
                        head_x = result.left_center[0]
                        head_y = max(0.1, result.left_center[1] - 0.15)  # Above center

                        # Extract region for emotion detection - pass whole person bbox
                        # and let fer's MTCNN find the actual face
                        if self.audio_manager is not None:
                            try:
                                # Check for stereo mode
                                aspect = w / h
                                is_stereo = aspect > 2.0
                                eye_w = w // 2 if is_stereo else w

                                # Get the full person bounding box
                                if result.left_box is not None:
                                    # Box is normalized 0-1, convert to pixels
                                    bx1 = int(result.left_box[0] * eye_w)
                                    by1 = int(result.left_box[1] * h)
                                    bx2 = int(result.left_box[2] * eye_w)
                                    by2 = int(result.left_box[3] * h)

                                    # Add some padding and pass the FULL person region
                                    # fer's MTCNN will find the face within this region
                                    pad = int((bx2 - bx1) * 0.1)
                                    x1 = max(0, bx1 - pad)
                                    y1 = max(0, by1)
                                    x2 = min(eye_w, bx2 + pad)
                                    y2 = min(h, by2)
                                else:
                                    # Fallback: use large center region
                                    cx = int(result.left_center[0] * eye_w)
                                    cy = int(result.left_center[1] * h)
                                    region_w = int(eye_w * 0.5)
                                    region_h = int(h * 0.7)
                                    x1 = max(0, cx - region_w // 2)
                                    y1 = max(0, cy - region_h // 2)
                                    x2 = min(eye_w, cx + region_w // 2)
                                    y2 = min(h, cy + region_h // 2)

                                if x2 > x1 + 50 and y2 > y1 + 50:
                                    # Extract from left eye region only
                                    face_bgr = cv2.cvtColor(frame_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
                                    self.audio_manager.submit_face(face_bgr, (x1, y1, x2-x1, y2-y1))
                            except Exception as e:
                                pass  # Don't let face extraction errors break the pipeline

                    stats = {
                        'fps': fps,
                        'status': status,
                        'effect': effect_name,
                        'yolo_fps': self.processor.fps if hasattr(self.processor, 'fps') else 0,
                        'person_tracked': person_tracked,
                        'head_x': head_x,
                        'head_y': head_y,
                    }
                    self.ui.show(frame, stats)
                t_ui_end = time.time()
                self.profiler.record('ui_send', (t_ui_end - t_ui_start) * 1000)

                # Total time
                t_end = time.time()
                self.profiler.record('total', (t_end - t_start) * 1000)

                # Log profiling
                self.profiler.log_if_ready()

                # Handle input
                if not headless:
                    key = self.ui.poll_input()
                    if key is not None:
                        action = self.keyboard.poll(key)
                        if action == "quit":
                            break
                        elif action == "profiling":
                            self.profiling = not self.profiling
                            print(f"\nProfiling: {'ON' if self.profiling else 'OFF'}")

        finally:
            # Cleanup
            for control in self.controls:
                control.stop()
            self.processor.stop()
            self.ui.cleanup()
            self.source.stop()

            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"\n\nTotal: {self.frame_count} frames, {fps:.1f} FPS")
