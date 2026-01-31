"""
Video Stream Manager

Manages video frame ingestion from AR headset clients and processed frame delivery.
Handles frame buffering, encoding, and latency tracking.
"""

import asyncio
import time
from typing import Optional, Dict
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

from .codec_manager import VideoEncoder, VideoDecoder, FrameBuffer
from ..config import get_config, get_video_config


@dataclass
class StreamMetrics:
    """Metrics for a video stream."""

    session_id: str
    frames_received: int = 0
    frames_processed: int = 0
    frames_sent: int = 0
    frames_dropped: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    last_frame_time: float = field(default_factory=time.time)
    stream_start_time: float = field(default_factory=time.time)

    def update_latency(self, latency_ms: float):
        """Update latency metrics."""
        self.total_latency_ms += latency_ms
        samples = max(self.frames_processed, 1)
        self.avg_latency_ms = self.total_latency_ms / samples

    def get_fps(self) -> float:
        """Calculate current FPS."""
        elapsed = time.time() - self.stream_start_time
        if elapsed > 0:
            return self.frames_processed / elapsed
        return 0.0

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "session_id": self.session_id,
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_sent": self.frames_sent,
            "frames_dropped": self.frames_dropped,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "fps": round(self.get_fps(), 2),
            "uptime_seconds": round(time.time() - self.stream_start_time, 2),
        }


class VideoStreamManager:
    """
    Manages bidirectional video streaming for a single session.

    Handles:
    - Incoming raw frames from AR headset
    - Frame buffering and queuing for processing pipeline
    - Outgoing processed frames to AR headset
    - Latency tracking and adaptive quality
    """

    def __init__(self, session_id: str):
        """
        Initialize stream manager.

        Args:
            session_id: Unique session identifier.
        """
        self.session_id = session_id
        self.config = get_config()
        self.video_config = get_video_config()

        # Frame buffers
        self.incoming_buffer = FrameBuffer(max_size=self.config.FRAME_BUFFER_SIZE)
        self.outgoing_buffer = FrameBuffer(max_size=self.config.FRAME_BUFFER_SIZE)

        # Encoder/Decoder
        self.encoder = VideoEncoder(self.video_config)
        self.decoder = VideoDecoder(self.video_config)

        # Metrics
        self.metrics = StreamMetrics(session_id=session_id)

        # State
        self.is_active = False
        self.processing_task: Optional[asyncio.Task] = None

        logger.info(f"VideoStreamManager initialized for session {session_id}")

    async def start(self):
        """Start the stream manager."""
        if self.is_active:
            logger.warning(f"Stream {self.session_id} already active")
            return

        # Initialize encoder
        if not self.encoder.initialize():
            raise RuntimeError("Failed to initialize video encoder")

        self.is_active = True
        logger.success(f"Stream {self.session_id} started")

    async def stop(self):
        """Stop the stream manager and release resources."""
        self.is_active = False

        # Cancel processing task if running
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Close encoder/decoder
        self.encoder.close()
        self.decoder.close()

        # Clear buffers
        self.incoming_buffer.clear()
        self.outgoing_buffer.clear()

        logger.info(f"Stream {self.session_id} stopped")

    async def receive_frame(self, frame_data: bytes, timestamp: float) -> bool:
        """
        Receive an encoded frame from the client.

        Args:
            frame_data: Raw frame data (could be encoded or raw pixels).
            timestamp: Client-side timestamp.

        Returns:
            True if frame was accepted, False if dropped.
        """
        if not self.is_active:
            return False

        self.metrics.frames_received += 1

        try:
            # For now, assume we receive raw numpy arrays serialized
            # In production, you'd decode H.264 here
            # For simplicity, we'll expect numpy arrays
            import io
            frame = np.load(io.BytesIO(frame_data), allow_pickle=False)

            # Add to incoming buffer
            success = self.incoming_buffer.put(
                frame, timestamp, drop_if_full=self.config.ADAPTIVE_QUALITY
            )

            if not success:
                self.metrics.frames_dropped += 1

            return success

        except Exception as e:
            logger.error(f"Error receiving frame: {e}")
            return False

    async def get_next_frame_for_processing(
        self, timeout: float = 0.1
    ) -> Optional[tuple[np.ndarray, float]]:
        """
        Get the next frame that needs processing.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Tuple of (frame, timestamp) or None.
        """
        return self.incoming_buffer.get(timeout=timeout)

    async def send_processed_frame(self, frame: np.ndarray, original_timestamp: float) -> bool:
        """
        Send a processed frame back to the client.

        Args:
            frame: Processed frame.
            original_timestamp: Original client timestamp for latency calculation.

        Returns:
            True if frame was queued for sending.
        """
        if not self.is_active:
            return False

        current_time = time.time()
        latency_ms = (current_time - original_timestamp) * 1000

        # Update metrics
        self.metrics.frames_processed += 1
        self.metrics.update_latency(latency_ms)
        self.metrics.last_frame_time = current_time

        # Log high latency
        if latency_ms > self.config.MAX_LATENCY_MS:
            logger.warning(
                f"High latency detected: {latency_ms:.1f}ms "
                f"(target: {self.config.MAX_LATENCY_MS}ms)"
            )

        # Add to outgoing buffer
        success = self.outgoing_buffer.put(
            frame, current_time, drop_if_full=self.config.ADAPTIVE_QUALITY
        )

        if success:
            self.metrics.frames_sent += 1

        return success

    async def get_encoded_frame_for_client(self) -> Optional[bytes]:
        """
        Get the next encoded frame to send to client.

        Returns:
            Encoded frame data or None.
        """
        # Get latest frame (discard old ones for low latency)
        result = self.outgoing_buffer.get_latest()

        if not result:
            return None

        frame, timestamp = result

        # Encode frame
        encoded = self.encoder.encode_frame(frame)

        return encoded

    def get_metrics(self) -> dict:
        """Get current stream metrics."""
        return {
            **self.metrics.to_dict(),
            "incoming_buffer": self.incoming_buffer.get_stats(),
            "outgoing_buffer": self.outgoing_buffer.get_stats(),
        }

    def should_drop_frames(self) -> bool:
        """
        Determine if frames should be dropped based on buffer state.

        Returns:
            True if frames should be dropped to reduce latency.
        """
        incoming_size = self.incoming_buffer.size()
        threshold = self.config.FRAME_DROP_THRESHOLD

        return incoming_size > threshold


class StreamRegistry:
    """
    Global registry for all active video streams.

    Manages stream lifecycle and provides lookup by session ID.
    """

    def __init__(self):
        """Initialize stream registry."""
        self.streams: Dict[str, VideoStreamManager] = {}
        self._lock = asyncio.Lock()

    async def create_stream(self, session_id: str) -> VideoStreamManager:
        """
        Create a new stream for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            VideoStreamManager instance.

        Raises:
            ValueError: If stream already exists for this session.
        """
        async with self._lock:
            if session_id in self.streams:
                raise ValueError(f"Stream already exists for session {session_id}")

            stream = VideoStreamManager(session_id)
            await stream.start()
            self.streams[session_id] = stream

            logger.info(f"Created stream for session {session_id}")
            return stream

    async def get_stream(self, session_id: str) -> Optional[VideoStreamManager]:
        """
        Get stream by session ID.

        Args:
            session_id: Session identifier.

        Returns:
            VideoStreamManager or None if not found.
        """
        async with self._lock:
            return self.streams.get(session_id)

    async def remove_stream(self, session_id: str) -> bool:
        """
        Remove and stop a stream.

        Args:
            session_id: Session identifier.

        Returns:
            True if stream was removed, False if not found.
        """
        async with self._lock:
            stream = self.streams.pop(session_id, None)

            if stream:
                await stream.stop()
                logger.info(f"Removed stream for session {session_id}")
                return True

            return False

    async def get_all_metrics(self) -> dict:
        """
        Get metrics for all active streams.

        Returns:
            Dictionary mapping session_id to metrics.
        """
        async with self._lock:
            return {
                session_id: stream.get_metrics()
                for session_id, stream in self.streams.items()
            }

    async def cleanup_inactive_streams(self, timeout_seconds: int = 300):
        """
        Remove streams that have been inactive for too long.

        Args:
            timeout_seconds: Inactivity timeout in seconds.
        """
        current_time = time.time()
        inactive_sessions = []

        async with self._lock:
            for session_id, stream in self.streams.items():
                time_since_last_frame = current_time - stream.metrics.last_frame_time
                if time_since_last_frame > timeout_seconds:
                    inactive_sessions.append(session_id)

        # Remove inactive streams
        for session_id in inactive_sessions:
            logger.warning(f"Removing inactive stream: {session_id}")
            await self.remove_stream(session_id)

    def get_active_stream_count(self) -> int:
        """Get number of active streams."""
        return len(self.streams)


# Global stream registry
_stream_registry: Optional[StreamRegistry] = None


def get_stream_registry() -> StreamRegistry:
    """Get the global stream registry instance."""
    global _stream_registry
    if _stream_registry is None:
        _stream_registry = StreamRegistry()
    return _stream_registry
