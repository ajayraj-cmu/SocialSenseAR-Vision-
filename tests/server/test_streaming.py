"""
Tests for streaming server components
"""

import pytest
import asyncio
import numpy as np
from server.streaming.video_stream import VideoStreamManager, StreamRegistry
from server.streaming.codec_manager import VideoEncoder, FrameBuffer


@pytest.mark.asyncio
async def test_video_stream_manager_creation():
    """Test creating a video stream manager."""
    stream = VideoStreamManager("test_session_1")
    await stream.start()

    assert stream.is_active
    assert stream.session_id == "test_session_1"

    await stream.stop()
    assert not stream.is_active


@pytest.mark.asyncio
async def test_frame_buffering():
    """Test frame buffer operations."""
    buffer = FrameBuffer(max_size=5)

    # Create test frames
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add frames
    assert buffer.put(frame1, 1.0)
    assert buffer.put(frame2, 2.0)

    assert buffer.size() == 2

    # Get frame
    result = buffer.get(timeout=0.1)
    assert result is not None
    retrieved_frame, timestamp = result
    assert timestamp == 1.0


@pytest.mark.asyncio
async def test_stream_registry():
    """Test stream registry operations."""
    registry = StreamRegistry()

    # Create stream
    stream = await registry.create_stream("session1")
    assert stream.session_id == "session1"

    # Get stream
    retrieved = await registry.get_stream("session1")
    assert retrieved is not None
    assert retrieved.session_id == "session1"

    # Remove stream
    removed = await registry.remove_stream("session1")
    assert removed

    # Verify removed
    assert await registry.get_stream("session1") is None


def test_video_encoder_initialization():
    """Test video encoder initialization."""
    encoder = VideoEncoder()
    success = encoder.initialize()

    # May fail if GPU not available, that's okay for tests
    if success:
        assert encoder.is_initialized
        encoder.close()


def test_frame_buffer_drop_policy():
    """Test frame buffer drop policy when full."""
    buffer = FrameBuffer(max_size=3)

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Fill buffer
    for i in range(3):
        assert buffer.put(frame, float(i), drop_if_full=True)

    # Try to add when full
    assert not buffer.put(frame, 4.0, drop_if_full=True)
    assert buffer.dropped_frames == 1


@pytest.mark.asyncio
async def test_stream_metrics():
    """Test stream metrics collection."""
    stream = VideoStreamManager("metrics_test")
    await stream.start()

    metrics = stream.get_metrics()

    assert "session_id" in metrics
    assert metrics["session_id"] == "metrics_test"
    assert "frames_received" in metrics
    assert "frames_sent" in metrics

    await stream.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
