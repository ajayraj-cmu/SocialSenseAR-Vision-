"""
Base class for input controls.

To add a new input control:
1. Create a new file in the controls/ directory
2. Inherit from BaseControl
3. Implement start(), stop(), poll()
4. Register in controls/__init__.py
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable


class BaseControl(ABC):
    """Abstract base class for input controls.

    Controls detect user input (keyboard, gestures, voice, etc.)
    and trigger callbacks or return actions.
    """

    def __init__(self, callback: Optional[Callable] = None):
        """Initialize control with optional callback.

        Args:
            callback: Function to call when control is activated
        """
        self.callback = callback

    @abstractmethod
    def start(self) -> None:
        """Start listening for input.

        Called once at startup.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and cleanup.

        Called at shutdown.
        """
        pass

    def poll(self) -> Optional[str]:
        """Poll for input (optional).

        Some controls use polling instead of callbacks.

        Returns:
            Action string or None
        """
        return None

    def trigger(self) -> None:
        """Trigger the callback if set."""
        if self.callback:
            self.callback()
