"""
Keyboard input control.

Handles keyboard input via OpenCV window events.
"""
from typing import Optional, Callable
from .base import BaseControl


class KeyboardControl(BaseControl):
    """Keyboard input handler.

    Uses OpenCV's waitKey() for input detection.
    Must be polled in the main loop.

    Key bindings:
        q - quit
        p - toggle processing
        t - toggle profiling
    """

    def __init__(self, callback: Optional[Callable] = None):
        """Initialize keyboard control.

        Args:
            callback: Function to call when toggle key is pressed
        """
        super().__init__(callback)

    def start(self) -> None:
        """No startup needed for keyboard."""
        pass

    def stop(self) -> None:
        """No cleanup needed for keyboard."""
        pass

    def poll(self, key: int) -> Optional[str]:
        """Process a key press.

        Args:
            key: Key code from cv2.waitKey()

        Returns:
            Action string: "quit", "toggle", "profiling", or None
        """
        if key == ord('q'):
            return "quit"
        elif key == ord('p'):
            self.trigger()
            return "toggle"
        elif key == ord('t'):
            return "profiling"

        return None
