"""
Webcam video source for testing without Quest.

Simple OpenCV webcam capture for local testing.
"""
from typing import Optional, Tuple
import os
import cv2
import numpy as np

from .base import BaseSource


class WebcamSource(BaseSource):
    """Webcam video source using OpenCV.

    Use for testing the pipeline without a Quest headset.
    """

    def __init__(self, config=None):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index = getattr(config, 'camera_index', 0) if config else 0
        self._resolution = (0, 0)

    def start(self) -> bool:
        """Start webcam capture."""
        print(f"Opening webcam {self.camera_index}...")
        backend = getattr(self.config, 'camera_backend', None)
        if backend is None and os.name == 'nt':
            backend = cv2.CAP_DSHOW
        if backend is not None:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"ERROR: Could not open webcam {self.camera_index}")
            return False

        # Keep the capture buffer small to reduce startup lag.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try to set higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Read actual resolution
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._resolution = (w, h)

        print(f"Webcam opened: {w}x{h} @ {fps:.0f}fps")
        return True

    def stop(self) -> None:
        """Release webcam."""
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Webcam released")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get frame from webcam.

        Returns:
            RGB stereo frame (duplicated side-by-side for pipeline compatibility)
        """
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Duplicate to create side-by-side stereo for pipeline compatibility
        stereo_frame = np.hstack([frame_rgb, frame_rgb])
        return stereo_frame

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get webcam resolution (stereo width)."""
        return (self._resolution[0] * 2, self._resolution[1])

    @property
    def is_stereo(self) -> bool:
        """Output is stereo (duplicated)."""
        return True

    def get_device_info(self) -> dict:
        """Get webcam info."""
        return {
            "device": f"Webcam {self.camera_index}",
            "resolution": f"{self._resolution[0]}x{self._resolution[1]}",
        }
