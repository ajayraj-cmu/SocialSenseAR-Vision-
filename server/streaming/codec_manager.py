"""
Video Codec Manager

Handles H.264 encoding/decoding with hardware acceleration (NVENC).
Optimized for low-latency streaming to AR headsets.
"""

import time
from typing import Optional, Tuple
import numpy as np
import av
import cv2
from loguru import logger

from ..config import VideoConfig, get_config


class VideoEncoder:
    """H.264 video encoder with GPU acceleration."""

    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize video encoder.

        Args:
            config: Video configuration. If None, uses default from server config.
        """
        self.config = config or VideoConfig()
        self.container: Optional[av.container.OutputContainer] = None
        self.stream: Optional[av.video.stream.VideoStream] = None
        self.frame_count = 0
        self.is_initialized = False

        server_config = get_config()
        self.use_gpu = self.config.use_gpu_encoding and server_config.DEBUG is False

    def initialize(self) -> bool:
        """
        Initialize the encoder with configured settings.

        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Create in-memory container for streaming
            self.container = av.open("pipe:", mode="w", format="h264")

            # Determine codec
            codec_name = self.config.gpu_encoder if self.use_gpu else "libx264"

            logger.info(
                f"Initializing video encoder: {codec_name} "
                f"({self.config.width}x{self.config.height} @ {self.config.fps}fps)"
            )

            # Create video stream
            self.stream = self.container.add_stream(codec_name, rate=self.config.fps)
            self.stream.width = self.config.width
            self.stream.height = self.config.height
            self.stream.pix_fmt = self.config.pixel_format
            self.stream.bit_rate = self.config.bitrate
            self.stream.gop_size = self.config.gop_size

            # Codec-specific options for low latency
            if self.use_gpu:
                # NVENC options
                self.stream.options = {
                    "preset": "p1",  # Fastest preset
                    "tune": "ll",  # Low latency
                    "delay": "0",  # Zero delay
                    "zerolatency": "1",
                    "rc": "cbr",  # Constant bitrate for predictable latency
                }
            else:
                # libx264 options
                self.stream.options = {
                    "preset": self.config.preset,
                    "tune": self.config.tune,
                    "crf": str(self.config.crf),
                    "g": str(self.config.gop_size),
                }

            self.is_initialized = True
            logger.success("Video encoder initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize video encoder: {e}")
            self.is_initialized = False
            return False

    def encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Encode a single video frame to H.264.

        Args:
            frame: Input frame as numpy array (BGR format from OpenCV).

        Returns:
            Encoded frame data as bytes, or None if encoding fails.
        """
        if not self.is_initialized:
            logger.warning("Encoder not initialized, attempting to initialize")
            if not self.initialize():
                return None

        try:
            start_time = time.perf_counter()

            # Resize frame if needed
            if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
                frame = cv2.resize(frame, (self.config.width, self.config.height))

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create AVFrame
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            av_frame.pts = self.frame_count
            av_frame.time_base = self.stream.time_base

            # Encode
            packets = self.stream.encode(av_frame)

            # Collect encoded data
            encoded_data = b""
            for packet in packets:
                encoded_data += bytes(packet)

            self.frame_count += 1

            encode_time_ms = (time.perf_counter() - start_time) * 1000
            if encode_time_ms > 10:  # Log if encoding takes > 10ms
                logger.warning(f"Slow encoding: {encode_time_ms:.1f}ms")

            return encoded_data if encoded_data else None

        except Exception as e:
            logger.error(f"Frame encoding failed: {e}")
            return None

    def flush(self) -> bytes:
        """
        Flush remaining packets from encoder.

        Returns:
            Remaining encoded data.
        """
        if not self.stream:
            return b""

        try:
            packets = self.stream.encode(None)
            encoded_data = b""
            for packet in packets:
                encoded_data += bytes(packet)
            return encoded_data
        except Exception as e:
            logger.error(f"Flush failed: {e}")
            return b""

    def close(self):
        """Close encoder and release resources."""
        try:
            if self.stream:
                self.flush()
            if self.container:
                self.container.close()
            self.is_initialized = False
            logger.info("Video encoder closed")
        except Exception as e:
            logger.error(f"Error closing encoder: {e}")


class VideoDecoder:
    """H.264 video decoder."""

    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Initialize video decoder.

        Args:
            config: Video configuration.
        """
        self.config = config or VideoConfig()
        self.container: Optional[av.container.InputContainer] = None
        self.stream: Optional[av.video.stream.VideoStream] = None
        self.codec_context: Optional[av.codec.context.CodecContext] = None
        self.is_initialized = False

    def initialize(self, initial_data: bytes) -> bool:
        """
        Initialize decoder with initial encoded data.

        Args:
            initial_data: Initial H.264 data to parse codec parameters.

        Returns:
            True if successful.
        """
        try:
            # Create container from bytes
            self.container = av.open(io.BytesIO(initial_data), mode="r", format="h264")
            self.stream = self.container.streams.video[0]
            self.codec_context = self.stream.codec_context

            self.is_initialized = True
            logger.success("Video decoder initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize decoder: {e}")
            return False

    def decode_frame(self, encoded_data: bytes) -> Optional[np.ndarray]:
        """
        Decode H.264 data to frame.

        Args:
            encoded_data: Encoded H.264 data.

        Returns:
            Decoded frame as numpy array (BGR format), or None if decoding fails.
        """
        try:
            # For server-side, we typically don't decode incoming frames
            # This is primarily for testing or if we need to inspect client frames
            import io

            container = av.open(io.BytesIO(encoded_data), mode="r", format="h264")

            for frame in container.decode(video=0):
                # Convert to numpy array
                img = frame.to_ndarray(format="rgb24")
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                container.close()
                return img_bgr

            container.close()
            return None

        except Exception as e:
            logger.error(f"Frame decoding failed: {e}")
            return None

    def close(self):
        """Close decoder and release resources."""
        try:
            if self.container:
                self.container.close()
            self.is_initialized = False
            logger.info("Video decoder closed")
        except Exception as e:
            logger.error(f"Error closing decoder: {e}")


class FrameBuffer:
    """
    Thread-safe circular buffer for video frames.

    Implements triple buffering strategy for GPU <-> CPU transfers.
    """

    def __init__(self, max_size: int = 30):
        """
        Initialize frame buffer.

        Args:
            max_size: Maximum number of frames to buffer.
        """
        import queue

        self.max_size = max_size
        self.buffer: queue.Queue = queue.Queue(maxsize=max_size)
        self.dropped_frames = 0
        self.total_frames = 0

    def put(self, frame: np.ndarray, timestamp: float, drop_if_full: bool = True) -> bool:
        """
        Add frame to buffer.

        Args:
            frame: Video frame.
            timestamp: Frame timestamp.
            drop_if_full: If True, drop frame if buffer is full. If False, block.

        Returns:
            True if frame was added, False if dropped.
        """
        self.total_frames += 1

        try:
            if drop_if_full:
                self.buffer.put_nowait((frame, timestamp))
            else:
                self.buffer.put((frame, timestamp), timeout=0.1)
            return True

        except queue.Full:
            self.dropped_frames += 1
            if self.dropped_frames % 10 == 0:  # Log every 10 drops
                drop_rate = (self.dropped_frames / self.total_frames) * 100
                logger.warning(
                    f"Frame buffer full, dropped {self.dropped_frames} frames "
                    f"({drop_rate:.1f}% drop rate)"
                )
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get frame from buffer.

        Args:
            timeout: Timeout in seconds. None = block indefinitely.

        Returns:
            Tuple of (frame, timestamp) or None if timeout.
        """
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the latest frame, discarding all older frames.

        Returns:
            Latest frame and timestamp, or None if buffer is empty.
        """
        latest = None
        discarded = 0

        while not self.buffer.empty():
            try:
                latest = self.buffer.get_nowait()
                discarded += 1
            except queue.Empty:
                break

        if discarded > 1:
            logger.debug(f"Skipped {discarded - 1} frames to get latest")

        return latest

    def clear(self):
        """Clear all frames from buffer."""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break

    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.buffer.full()

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "size": self.size(),
            "max_size": self.max_size,
            "dropped_frames": self.dropped_frames,
            "total_frames": self.total_frames,
            "drop_rate": (
                (self.dropped_frames / self.total_frames) * 100
                if self.total_frames > 0
                else 0.0
            ),
        }
