"""
Video file source - reads frames from a video file.

Use this to process pre-recorded videos through the full pipeline
with real audio processing (Whisper, GPT, emotion detection).
"""
import time
import threading
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BaseSource


class VideoFileSource(BaseSource):
    """Video source that reads from a file.

    Plays back at the original framerate (or custom rate) to simulate
    real-time input for the audio/video processing pipeline.
    Also plays audio through speakers for mic pickup.
    """

    def __init__(self, config):
        """Initialize video file source.

        Args:
            config: Configuration object with video_file path
        """
        self.config = config
        self.video_path = getattr(config, 'video_file', None)
        self.playback_speed = getattr(config, 'playback_speed', 1.0)
        self.loop = getattr(config, 'loop_video', False)
        self.play_audio = getattr(config, 'play_audio', True)  # Play audio by default

        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30
        self.frame_time = 1.0 / 30
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.duration = 0

        self.current_frame: Optional[np.ndarray] = None
        self.frame_idx = 0
        self.start_time = 0
        self.running = False

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._audio_process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Start reading from video file."""
        if not self.video_path:
            print("[VIDEO] Error: No video file specified")
            print("[VIDEO] Use --video-file path/to/video.mp4")
            return False

        path = Path(self.video_path)
        if not path.exists():
            print(f"[VIDEO] Error: File not found: {path}")
            return False

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            print(f"[VIDEO] Error: Could not open video: {path}")
            return False

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_time = 1.0 / (self.fps * self.playback_speed)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        print(f"\n{'='*60}")
        print(f"Video File Source")
        print(f"{'='*60}")
        print(f"File: {path.name}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Duration: {self.duration:.1f}s ({self.frame_count} frames @ {self.fps:.1f}fps)")
        print(f"Playback speed: {self.playback_speed}x")
        print(f"{'='*60}\n")

        # Read first frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_idx = 1

        self.running = True
        self.start_time = time.time()

        # Start audio playback
        if self.play_audio:
            self._start_audio_playback(path)

        # Start playback thread
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

        return True

    def _start_audio_playback(self, video_path: Path) -> None:
        """Start audio playback using ffplay."""
        # Find ffplay
        ffplay = shutil.which('ffplay')
        if not ffplay:
            # Try common locations
            common_paths = [
                r"C:\ffmpeg\bin\ffplay.exe",
                r"C:\Program Files\ffmpeg\bin\ffplay.exe",
                "/usr/bin/ffplay",
                "/usr/local/bin/ffplay",
            ]
            for p in common_paths:
                if Path(p).exists():
                    ffplay = p
                    break

        if not ffplay:
            print("[VIDEO] Warning: ffplay not found, audio will not play")
            print("[VIDEO] Install ffmpeg and add to PATH for audio playback")
            return

        try:
            # Build ffplay command
            cmd = [
                ffplay,
                '-nodisp',           # No video display (we handle that)
                '-autoexit',         # Exit when done
                '-loglevel', 'quiet',  # Suppress output
                str(video_path)
            ]

            # Add speed filter if not 1.0
            if self.playback_speed != 1.0:
                cmd = [
                    ffplay,
                    '-nodisp',
                    '-autoexit',
                    '-loglevel', 'quiet',
                    '-af', f'atempo={self.playback_speed}',
                    str(video_path)
                ]

            print(f"[VIDEO] Starting audio playback...")
            self._audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[VIDEO] Could not start audio playback: {e}")

    def _playback_loop(self):
        """Background thread for frame playback at correct rate."""
        next_frame_time = time.time()

        while self.running:
            now = time.time()

            if now >= next_frame_time:
                ret, frame = self.cap.read()

                if not ret:
                    if self.loop:
                        # Loop back to start
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frame_idx = 0
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                    else:
                        # End of video
                        print("\n[VIDEO] End of video reached")
                        self.running = False
                        break

                with self._lock:
                    self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_idx += 1

                next_frame_time += self.frame_time
            else:
                # Sleep until next frame
                time.sleep(max(0, next_frame_time - now - 0.001))

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame.

        Returns:
            RGB frame as numpy array, or None if not available
        """
        with self._lock:
            return self.current_frame

    def stop(self) -> None:
        """Stop reading and release resources."""
        self.running = False

        # Stop audio playback
        if self._audio_process:
            try:
                self._audio_process.terminate()
                self._audio_process.wait(timeout=1.0)
            except:
                pass
            self._audio_process = None

        if self._thread:
            self._thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        print("[VIDEO] Video source stopped")

    def get_progress(self) -> tuple:
        """Get playback progress.

        Returns:
            (current_frame, total_frames, elapsed_seconds, total_seconds)
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        return (self.frame_idx, self.frame_count, elapsed, self.duration)
