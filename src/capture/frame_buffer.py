"""Frame Buffer for Recording and Playback."""
from __future__ import annotations
import threading
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from loguru import logger

@dataclass
class RecordedFrame:
    """A single recorded frame with audio."""
    frame_id: int
    timestamp_ms: float
    video_frame: NDArray[np.uint8]
    audio_chunk: Optional[NDArray[np.float32]] = None
    transforms_applied: List[str] = field(default_factory=list)

class FrameBuffer:
    """Circular frame buffer for recording and playback."""

    def __init__(self, max_duration_seconds: float = 300.0, fps: int = 30):
        self.max_duration_seconds = max_duration_seconds
        self.fps = fps
        self.max_frames = int(max_duration_seconds * fps)
        self._frames: List[RecordedFrame] = []
        self._lock = threading.Lock()
        self._is_recording = False
        self._recording_start_time: float = 0.0
        self._playback_index: int = 0
        self._is_playing = False

    def start_recording(self):
        """Start recording frames."""
        with self._lock:
            self._frames.clear()
            self._is_recording = True
            self._recording_start_time = 0.0
            logger.info("Recording started")

    def stop_recording(self) -> int:
        """Stop recording and return frame count."""
        with self._lock:
            self._is_recording = False
            count = len(self._frames)
            logger.info(f"Recording stopped: {count} frames")
            return count

    def add_frame(self, frame_id: int, timestamp_ms: float, video_frame: NDArray[np.uint8], audio_chunk: Optional[NDArray[np.float32]] = None):
        """Add a frame to the buffer during recording."""
        if not self._is_recording:
            return
        with self._lock:
            if len(self._frames) >= self.max_frames:
                self._is_recording = False
                logger.warning("Recording buffer full, stopping")
                return
            if self._recording_start_time == 0.0:
                self._recording_start_time = timestamp_ms
            recorded = RecordedFrame(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms - self._recording_start_time,
                video_frame=video_frame.copy(),
                audio_chunk=audio_chunk.copy() if audio_chunk is not None else None,
            )
            self._frames.append(recorded)

    def add_augmented_frame(self, frame_id: int, timestamp_ms: float, augmented_frame: NDArray[np.uint8]):
        """Add an augmented (processed) frame to the buffer."""
        self.add_frame(frame_id, timestamp_ms, augmented_frame, None)

    def get_frame(self, index: int) -> Optional[RecordedFrame]:
        """Get a frame by index."""
        with self._lock:
            if 0 <= index < len(self._frames):
                return self._frames[index]
            return None

    def get_frame_count(self) -> int:
        """Get number of recorded frames."""
        with self._lock:
            return len(self._frames)

    def start_playback(self, start_index: int = 0):
        """Start playback from index."""
        with self._lock:
            self._playback_index = max(0, min(start_index, len(self._frames) - 1))
            self._is_playing = True
            logger.info(f"Playback started at frame {self._playback_index}")

    def stop_playback(self):
        """Stop playback."""
        self._is_playing = False
        logger.info("Playback stopped")

    def get_next_playback_frame(self) -> Optional[RecordedFrame]:
        """Get next frame during playback."""
        with self._lock:
            if not self._is_playing or self._playback_index >= len(self._frames):
                self._is_playing = False
                return None
            frame = self._frames[self._playback_index]
            self._playback_index += 1
            return frame

    def update_frame(self, index: int, video_frame: NDArray[np.uint8], transform_id: str):
        """Update a recorded frame with transformed video."""
        with self._lock:
            if 0 <= index < len(self._frames):
                self._frames[index].video_frame = video_frame.copy()
                self._frames[index].transforms_applied.append(transform_id)

    def get_all_frames(self) -> List[RecordedFrame]:
        """Get all recorded frames."""
        with self._lock:
            return self._frames.copy()

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._frames.clear()
            self._is_recording = False
            self._is_playing = False
            self._playback_index = 0
            logger.info("Frame buffer cleared")

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def duration_seconds(self) -> float:
        """Get recording duration in seconds."""
        with self._lock:
            if not self._frames:
                return 0.0
            return self._frames[-1].timestamp_ms / 1000.0
