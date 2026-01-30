"""
UI module.

Provides different UI backends (OpenCV, Qt, Web, Headless, etc.)

To add a new UI:
1. Create a new file in this directory
2. Implement a class inheriting from BaseUI
3. Register it in UIS dict below

The UI is responsible for:
- Displaying processed video frames
- Handling window/display setup
- Providing input polling (for keyboard, etc.)
"""
from typing import Optional
from .base import BaseUI
from .opencv_window import OpenCVUI
from .quest_tcp import QuestTCPUI

# Registry of available UIs
UIS = {
    "opencv": OpenCVUI,
    "headless": lambda config: HeadlessUI(config),
    "webview": lambda config: _get_webview_ui(config),
    "quest_tcp": QuestTCPUI,
}


def _get_webview_ui(config):
    """Lazy import WebViewUI to avoid import errors if pywebview not installed."""
    from .webview_ui import WebViewUI
    return WebViewUI(config)


class HeadlessUI(BaseUI):
    """Headless UI - no display, just processing."""

    def __init__(self, config=None):
        pass

    def setup(self, title: str = "") -> None:
        pass

    def show(self, frame, stats: dict = None) -> None:
        pass

    def poll_input(self) -> Optional[int]:
        return None

    def cleanup(self) -> None:
        pass


def get_ui(name: str, config=None) -> BaseUI:
    """Get a UI instance by name.

    Args:
        name: UI type name (e.g., "opencv", "headless")
        config: Configuration object

    Returns:
        Initialized UI instance
    """
    if name not in UIS:
        available = ", ".join(UIS.keys())
        raise ValueError(f"Unknown UI '{name}'. Available: {available}")

    return UIS[name](config)


def list_uis() -> list:
    """List available UI names."""
    return list(UIS.keys())


__all__ = ['BaseUI', 'OpenCVUI', 'HeadlessUI', 'get_ui', 'list_uis', 'WebViewUI']
