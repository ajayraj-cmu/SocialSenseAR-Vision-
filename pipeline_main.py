#!/usr/bin/env python3
"""
Perceptual Modulation Engine for Meta Quest 3 Passthrough

Main entry point for the real-time perceptual modulation system.

Usage:
    python main.py [--config CONFIG_PATH] [--device DEVICE_INDEX]

Keyboard Controls:
    R     - Toggle recording
    P     - Process last recording
    ESC   - Emergency revert (restore unmodified passthrough)
    Q     - Quit

Example Commands (voice or text):
    "dim the person to my right"
    "mute the screen on the left"
    "blur the bright light"
    "make everything slightly less saturated"
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from loguru import logger
from pynput import keyboard

from src.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig
from src.core.contracts import SafetyConstraints
from src.voice.voice_commander import VoiceCommander


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging."""
    logger.remove()  # Remove default handler
    
    # Console output with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True,
    )
    
    # File output
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{line} | {message}",
            rotation="10 MB",
            retention="7 days",
        )


class KeyboardHandler:
    """Handles keyboard input for the application."""
    
    def __init__(self, pipeline: PipelineOrchestrator, on_playback_callback=None):
        self.pipeline = pipeline
        self._quit_requested = False
        self._command_mode = False
        self._current_command = ""
        self._on_playback = on_playback_callback
        
    def on_press(self, key):
        """Handle key press events."""
        try:
            # Check for special keys
            if key == keyboard.Key.esc:
                self.pipeline.handle_keyboard('escape')
                return
            
            if hasattr(key, 'char'):
                char = key.char
                
                if char == 'q':
                    self._quit_requested = True
                elif char == 'r':
                    # R = Stop recording and playback augmented version
                    self.pipeline.handle_keyboard('r')
                    if self._on_playback:
                        self._on_playback()
                elif char == 'p':
                    # P = Start recording
                    self.pipeline.handle_keyboard('p')
                elif char == '/':
                    # Enter command mode
                    self._command_mode = True
                    self._current_command = ""
                    logger.info("Command mode: Type command and press Enter")
                elif self._command_mode:
                    self._current_command += char
                    
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release events."""
        if key == keyboard.Key.enter and self._command_mode:
            if self._current_command:
                self.pipeline.queue_command(self._current_command)
                logger.info(f"Command sent: {self._current_command}")
            self._command_mode = False
            self._current_command = ""
    
    @property
    def quit_requested(self) -> bool:
        return self._quit_requested


class OutputRenderer:
    """Renders output to display and optional virtual camera."""
    
    def __init__(
        self,
        window_name: str = "Perceptual Modulation Engine",
        display_fps: bool = True,
        display_objects: bool = True,
    ):
        self.window_name = window_name
        self.display_fps = display_fps
        self.display_objects = display_objects
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def render(
        self,
        frame: np.ndarray,
        fps: float = 0,
        latency_ms: float = 0,
        object_count: int = 0,
        recording: bool = False,
        emergency_mode: bool = False,
        playback_mode: bool = False,
        active_effects: int = 0,
        last_command: str = "",
    ):
        """Render frame with overlay information."""
        # Convert RGB to BGR for OpenCV display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add overlay information
        if self.display_fps:
            self._draw_info_overlay(
                display_frame, fps, latency_ms, object_count, recording, 
                emergency_mode, playback_mode, active_effects, last_command
            )
        
        cv2.imshow(self.window_name, display_frame)
    
    def _draw_info_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        object_count: int,
        recording: bool,
        emergency_mode: bool,
        playback_mode: bool = False,
        active_effects: int = 0,
        last_command: str = "",
    ):
        """Draw information overlay on frame."""
        h, w = frame.shape[:2]
        
        # Background for text - larger to fit more info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text color
        if playback_mode:
            color = (255, 165, 0)  # Orange for playback
        elif emergency_mode:
            color = (0, 0, 255)  # Red for emergency
        elif active_effects > 0:
            color = (255, 255, 0)  # Yellow when effects active
        else:
            color = (0, 255, 0)  # Green normal
        
        # FPS and latency
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
        )
        cv2.putText(
            frame, f"Latency: {latency_ms:.1f}ms", (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
        )
        cv2.putText(
            frame, f"Objects: {object_count}", (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
        )
        
        # Recording indicator
        if recording:
            cv2.circle(frame, (350, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                frame, "REC", (310, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        
        # Effects active indicator - BIG AND VISIBLE
        if active_effects > 0:
            cv2.putText(
                frame, f"FX ACTIVE: {active_effects}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
        
        # Last voice command - show what was understood
        if last_command:
            display_cmd = last_command[:40] + "..." if len(last_command) > 40 else last_command
            cv2.putText(
                frame, f'CMD: "{display_cmd}"', (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1
            )
        
        # Playback indicator
        if playback_mode:
            cv2.putText(
                frame, ">>> PLAYBACK >>>", (w // 2 - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2
            )
        
        # Emergency mode indicator
        if emergency_mode:
            cv2.putText(
                frame, "EMERGENCY REVERT", (w // 2 - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
        
        # Voice indicator (only when not in playback)
        if not playback_mode and active_effects == 0:
            cv2.putText(
                frame, "VOICE ACTIVE", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
        
        # Help text at bottom
        help_text = "VOICE | P:Start Rec  R:Stop+Playback  ESC:Revert  Q:Quit"
        cv2.putText(
            frame, help_text, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
    
    def close(self):
        """Close the renderer."""
        cv2.destroyAllWindows()


class PerceptualModulationEngine:
    """Main application class."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Create pipeline config
        pipeline_config = PipelineConfig(
            video_device=self.config.get('video', {}).get('device_index', 0),
            video_width=self.config.get('video', {}).get('width', 1920),
            video_height=self.config.get('video', {}).get('height', 1080),
            video_fps=self.config.get('video', {}).get('fps', 30),
            audio_sample_rate=self.config.get('audio', {}).get('sample_rate', 48000),
            audio_channels=self.config.get('audio', {}).get('channels', 2),
            video_max_latency_ms=self.config.get('latency', {}).get('video_max_ms', 120),
            audio_max_latency_ms=self.config.get('latency', {}).get('audio_max_ms', 40),
        )
        
        self.pipeline = PipelineOrchestrator(pipeline_config)
        self.keyboard_handler = KeyboardHandler(
            self.pipeline, 
            on_playback_callback=self._trigger_playback
        )
        self.renderer = OutputRenderer()
        
        # Voice commander - sends commands directly to pipeline
        self.voice_commander = VoiceCommander(
            on_command=self._on_voice_command
        )
        
        self._running = False
        self._voice_enabled = True
        self._playback_requested = False
        self._recorded_frames = []
        self._last_voice_command = ""
    
    def _on_voice_command(self, command: str):
        """Handle voice command."""
        logger.info(f"🎤 Voice command received: \"{command}\"")
        self._last_voice_command = command
        self.pipeline.queue_command(command)
    
    def _trigger_playback(self):
        """Trigger playback of recorded augmented frames."""
        self._playback_requested = True
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Try default location
        default_path = Path(__file__).parent / "config" / "settings.yaml"
        if default_path.exists():
            with open(default_path) as f:
                return yaml.safe_load(f)
        
        return {}
    
    def run(self):
        """Run the main application loop."""
        logger.info("Starting Perceptual Modulation Engine")
        logger.info("Press Q to quit, ESC for emergency revert")
        
        if not self.pipeline.start():
            logger.error("Failed to start pipeline")
            return
        
        # Start keyboard listener
        listener = keyboard.Listener(
            on_press=self.keyboard_handler.on_press,
            on_release=self.keyboard_handler.on_release,
        )
        listener.start()
        
        # Start voice commander
        if self._voice_enabled:
            if self.voice_commander.start():
                logger.info("🎤 VOICE COMMANDS ACTIVE - Speak naturally!")
                logger.info("   Try: 'dim the screen' or 'blur the person on the left'")
            else:
                logger.warning("Voice commands unavailable - using keyboard only")
        
        self._running = True
        frame_count = 0
        start_time = time.time()
        
        try:
            while self._running and not self.keyboard_handler.quit_requested:
                # Check if we're in playback mode
                if self.pipeline.is_playing_back:
                    playback_frame = self.pipeline.get_playback_frame()
                    if playback_frame:
                        # Show the recorded (augmented) frame
                        self.renderer.render(
                            playback_frame.video_frame,
                            fps=30,
                            latency_ms=0,
                            object_count=0,
                            recording=False,
                            emergency_mode=False,
                            playback_mode=True,
                            active_effects=len(self.pipeline.state.active_operations),
                            last_command=self._last_voice_command,
                        )
                        # Slow down playback to real-time
                        time.sleep(1/30)
                    else:
                        # Playback finished
                        logger.info("▶️ Playback complete!")
                    
                    # Handle OpenCV window events during playback
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    continue
                
                # Normal processing mode
                output = self.pipeline.process_frame()
                
                # If recording, store the augmented frame
                if self.pipeline.state.is_recording:
                    self._recorded_frames.append(output.output_frame.copy())
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Render output
                active_ops = len([op for op in self.pipeline.state.active_operations if op.is_active])
                self.renderer.render(
                    output.output_frame,
                    fps=fps,
                    latency_ms=output.total_latency_ms,
                    object_count=len(self.pipeline.state.object_registry),
                    recording=self.pipeline.state.is_recording,
                    emergency_mode=output.fallback_to_passthrough,
                    active_effects=active_ops,
                    last_command=self._last_voice_command,
                )
                
                # Handle OpenCV window events
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._running = False
            listener.stop()
            if self._voice_enabled:
                self.voice_commander.stop()
            self.pipeline.stop()
            self.renderer.close()
            logger.info("Engine stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Perceptual Modulation Engine for Meta Quest 3 Passthrough",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=0,
        help="Video device index (default: 0)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/perceptual_engine.log",
        help="Log file path (default: logs/perceptual_engine.log)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Create and run engine
    engine = PerceptualModulationEngine(args.config)
    engine.run()


if __name__ == "__main__":
    main()

