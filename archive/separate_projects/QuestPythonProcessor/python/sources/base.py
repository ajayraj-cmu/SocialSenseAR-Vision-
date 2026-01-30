"""
Base class for video input sources.

To add a new video source:
1. Create a new file in the sources/ directory
2. Inherit from BaseSource
3. Implement all abstract methods
4. Register in sources/__init__.py SOURCES dict

Example implementation:
    class WebcamSource(BaseSource):
        def __init__(self, config):
            self.cap = None
            self.config = config

        def start(self) -> bool:
            self.cap = cv2.VideoCapture(0)
            return self.cap.isOpened()

        def stop(self) -> None:
            if self.cap:
                self.cap.release()

        def get_frame(self) -> Optional[np.ndarray]:
            if self.cap:
                ret, frame = self.cap.read()
                return frame if ret else None
            return None

        @property
        def resolution(self) -> tuple:
            if self.cap:
                return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            return (0, 0)

        @property
        def is_stereo(self) -> bool:
            return False  # Webcam is mono
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseSource(ABC):
    """Abstract base class for video input sources.

    All video sources (Quest, webcam, file, etc.) must implement this interface.
    Sources handle connecting to the video feed and providing frames.

    Attributes:
        resolution: Tuple of (width, height) of the video
        is_stereo: Whether source provides stereo (side-by-side) video
    """

    @abstractmethod
    def start(self) -> bool:
        """Initialize and start the video source.

        Should handle all setup: device connection, stream initialization, etc.

        Returns:
            True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop and cleanup the video source.

        Should release all resources, close connections, etc.
        """
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest video frame.

        This should be non-blocking. Returns None if no frame is available.
        For Quest/VR sources, returns side-by-side stereo frame (left|right).

        Returns:
            RGB numpy array of shape (H, W, 3), or None if unavailable
        """
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Get the video resolution.

        Returns:
            Tuple of (width, height)
        """
        pass

    @property
    @abstractmethod
    def is_stereo(self) -> bool:
        """Check if source provides stereo video.

        Stereo sources provide side-by-side frames (left eye | right eye).

        Returns:
            True for stereo/VR sources, False for mono sources
        """
        pass

    def get_device_info(self) -> dict:
        """Get information about the connected device.

        Returns:
            Dict with device info (serial, name, etc.)
        """
        return {"device": "unknown"}
