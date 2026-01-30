"""
Quest VR headset video source via scrcpy.

Connects to Meta Quest headset over ADB and captures the display
as a side-by-side stereo video stream.

Requirements:
    - adbutils
    - myscrcpy
    - Quest connected via USB or WiFi ADB
"""
from typing import Optional, Tuple
import threading
import time

import numpy as np

from .base import BaseSource


class FrameBuffer:
    """Async frame buffer that prefetches frames in a background thread.

    Runs a background thread that continuously fetches frames from the
    video adapter, keeping the latest frame available for immediate access.
    """

    def __init__(self, video_adapter, buffer_size: int = 2):
        self.va = video_adapter
        self.buffer_size = buffer_size
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()

    def start(self) -> None:
        """Start the background frame fetching thread."""
        self.running = True
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()

    def _fetch_loop(self) -> None:
        """Background loop that continuously fetches frames."""
        while self.running:
            frame = self.va.get_frame()
            if frame is not None:
                with self.lock:
                    self.latest_frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (non-blocking).

        Returns:
            Latest RGB frame or None if not available
        """
        with self.lock:
            return self.latest_frame

    def stop(self) -> None:
        """Stop the background thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)


class QuestSource(BaseSource):
    """Quest VR headset video source.

    Connects to Meta Quest via scrcpy and provides stereo video frames.
    The frames are side-by-side (left eye | right eye) in RGB format.

    Usage:
        source = QuestSource(config)
        if source.start():
            while True:
                frame = source.get_frame()
                if frame is not None:
                    process(frame)
        source.stop()
    """

    def __init__(self, config=None):
        """Initialize Quest source.

        Args:
            config: Configuration object with capture settings
        """
        self.config = config
        self.session = None
        self.frame_buffer = None
        self.device = None
        self._resolution = (0, 0)

    def start(self) -> bool:
        """Connect to Quest and start video capture.

        Returns:
            True if connected successfully
        """
        try:
            # Import here to avoid loading if not needed
            import adbutils
            from adbutils import adb
            from myscrcpy.core import Session, VideoArgs

            # Find Quest device
            devices = adb.device_list()
            if not devices:
                print("ERROR: No ADB devices found")
                return False

            self.device = devices[0]
            print(f"Found device: {self.device.serial}")

            # Get capture settings from config
            max_size = 0
            fps = 60
            if self.config:
                max_size = getattr(self.config, 'capture_max_size', 0)
                fps = getattr(self.config, 'capture_fps', 60)

            # Start scrcpy session
            print("Starting scrcpy session...")
            self.session = Session(
                self.device,
                video_args=VideoArgs(max_size=max_size, fps=fps),
            )

            print("Waiting for connection...")
            time.sleep(2)

            if self.session.va is None:
                print("ERROR: Video adapter not initialized")
                self.stop()
                return False

            print("Connected!")

            # Get initial frame to determine resolution
            frame = self.session.va.get_frame()
            if frame is None:
                print("ERROR: Could not get initial frame")
                self.stop()
                return False

            self._resolution = (frame.shape[1], frame.shape[0])
            print(f"Resolution: {self._resolution[0]}x{self._resolution[1]}")

            # Start async frame buffer
            self.frame_buffer = FrameBuffer(self.session.va)
            self.frame_buffer.latest_frame = frame
            self.frame_buffer.start()

            return True

        except ImportError as e:
            print(f"ERROR: Missing dependency - {e}")
            print("Install with: pip install adbutils myscrcpy")
            return False
        except Exception as e:
            print(f"ERROR: Failed to connect - {e}")
            return False

    def stop(self) -> None:
        """Disconnect from Quest and cleanup."""
        if self.frame_buffer:
            self.frame_buffer.stop()
            self.frame_buffer = None

        if self.session:
            self.session.disconnect()
            self.session = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from Quest.

        Returns:
            RGB frame as numpy array (stereo side-by-side), or None
        """
        if self.frame_buffer:
            return self.frame_buffer.get_frame()
        return None

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get video resolution (full stereo width x height)."""
        return self._resolution

    @property
    def is_stereo(self) -> bool:
        """Quest always provides stereo video."""
        return True

    def get_device_info(self) -> dict:
        """Get Quest device information."""
        if self.device:
            return {
                "device": "Quest",
                "serial": self.device.serial,
                "resolution": self._resolution,
            }
        return {"device": "Quest (not connected)"}
