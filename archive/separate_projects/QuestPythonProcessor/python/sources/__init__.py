"""
Video source module.

Provides different video input sources (Quest, webcam, file, etc.)

To add a new source:
1. Create a new file in this directory (e.g., webcam.py)
2. Implement a class inheriting from BaseSource
3. Register it in SOURCES dict below
"""
from typing import Optional
from .base import BaseSource
from .quest_scrcpy import QuestSource
from .quest_tcp import QuestTCPSource
from .webcam import WebcamSource
from .video_file import VideoFileSource

# Registry of available sources
SOURCES = {
    "quest": QuestSource,
    "quest_tcp": QuestTCPSource,
    "webcam": WebcamSource,
    "file": VideoFileSource,
}


def get_source(name: str, config=None) -> BaseSource:
    """Get a source instance by name.

    Args:
        name: Source type name (e.g., "quest", "webcam")
        config: Configuration object

    Returns:
        Initialized source instance

    Raises:
        ValueError: If source name is not registered
    """
    if name not in SOURCES:
        available = ", ".join(SOURCES.keys())
        raise ValueError(f"Unknown source '{name}'. Available: {available}")

    return SOURCES[name](config)


def list_sources() -> list:
    """List available source names."""
    return list(SOURCES.keys())


__all__ = ['BaseSource', 'QuestSource', 'get_source', 'list_sources']
