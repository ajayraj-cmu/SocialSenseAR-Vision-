"""
Base class for UI backends.

To add a new UI backend:
1. Create a new file in the ui/ directory
2. Inherit from BaseUI
3. Implement setup(), show(), poll_input(), cleanup()
4. Register in ui/__init__.py
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseUI(ABC):
    """Abstract base class for UI backends.

    UIs handle displaying video frames and receiving user input.
    Different UIs can be used for different platforms (desktop, web, etc.)
    """

    @abstractmethod
    def setup(self, title: str = "Quest Processor") -> None:
        """Initialize the UI.

        Args:
            title: Window/page title
        """
        pass

    @abstractmethod
    def show(self, frame: np.ndarray, stats: Optional[dict] = None) -> None:
        """Display a frame.

        Args:
            frame: BGR frame to display
            stats: Optional dict with fps, status, etc. for overlay
        """
        pass

    @abstractmethod
    def poll_input(self) -> Optional[int]:
        """Poll for user input.

        Returns:
            Key code or None if no input
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup UI resources."""
        pass
