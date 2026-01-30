"""
OpenCV window UI backend.

Simple UI using OpenCV's imshow for display.
Renders overlay AFTER scaling for crisp text.
"""
from typing import Optional
import cv2
import numpy as np

from .base import BaseUI
from .overlay_panels import render_overlay


class OpenCVUI(BaseUI):
    """OpenCV-based UI using imshow.

    Simple and portable UI that works on most platforms.
    Uses cv2.imshow() for display and cv2.waitKey() for input.
    Renders overlay after scaling for crisp text.
    """

    def __init__(self, config=None):
        """Initialize OpenCV UI.

        Args:
            config: Configuration object with display settings
        """
        self.config = config
        self.window_name = "Quest Processor"
        self.max_width = 1920
        self.render_overlay = True  # Render overlay in UI for crisp text
        self._frame_count = 0  # Skip overlay for first few frames to show window quickly
        if config:
            self.max_width = getattr(config, 'display_max_width', 1920)

    def setup(self, title: str = "Quest Processor") -> None:
        """Initialize the display window.

        Args:
            title: Window title
        """
        self.window_name = title
        # Window will be created on first show()

    def show(self, frame: np.ndarray, stats: Optional[dict] = None) -> None:
        """Display a frame.

        Args:
            frame: BGR frame to display
            stats: Optional stats dict (not used currently, could add overlay)
        """
        display_frame = frame

        # Downscale if too wide (use INTER_AREA for quality)
        h, w = display_frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            display_frame = cv2.resize(
                display_frame, None,
                fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA
            )

        # Show window first before any overlay processing
        cv2.imshow(self.window_name, display_frame)

        # Render overlay AFTER window is shown (non-blocking for first frames)
        if self.render_overlay and self._frame_count > 5:
            try:
                person_tracked = stats.get('person_tracked', True) if stats else True
                head_x = stats.get('head_x', 0.5) if stats else 0.5
                head_y = stats.get('head_y', 0.3) if stats else 0.3
                display_frame = render_overlay(display_frame, head_x=head_x, head_y=head_y,
                                               person_tracked=person_tracked)
                cv2.imshow(self.window_name, display_frame)
            except Exception as e:
                print(f"Overlay error: {e}")
                self.render_overlay = False

        self._frame_count += 1

    def poll_input(self) -> Optional[int]:
        """Poll for keyboard input.

        Returns:
            Key code (0-255) or None if no key pressed
        """
        key = cv2.waitKey(1) & 0xFF
        if key == 255:  # No key pressed
            return None
        return key

    def cleanup(self) -> None:
        """Destroy the window."""
        cv2.destroyAllWindows()
