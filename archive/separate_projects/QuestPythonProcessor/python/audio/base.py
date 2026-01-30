"""
Base class for audio services.

Audio services run in parallel with the video pipeline and produce
state updates that are consumed by the overlay renderer.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import json
import time


class BaseAudioService(ABC):
    """Abstract base class for audio services."""

    def __init__(self, config, state_dir: Path):
        """Initialize audio service.

        Args:
            config: Configuration object with audio settings
            state_dir: Directory for state output files
        """
        self.config = config
        self.state_dir = state_dir
        self.running = False

    @abstractmethod
    def start(self) -> bool:
        """Start the audio service.

        Returns:
            True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the audio service and cleanup resources."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state for overlay rendering.

        Returns:
            Dictionary with current state data
        """
        pass

    def write_state(self, state: Dict[str, Any], filename: str = "latest_state.json") -> None:
        """Write state to JSON file for overlay consumption.

        Args:
            state: State dictionary to write
            filename: Output filename (default: latest_state.json)
        """
        state_path = self.state_dir / filename

        # Add timestamp if not present
        if "timestamp" not in state:
            state["timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Write atomically by writing to temp then renaming
        temp_path = state_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(state, f, indent=2)
            temp_path.replace(state_path)
            print(f"\n{'='*60}")
            print(f"[STATE UPDATED] {state_path}")
            print(f"  Summary: {state.get('convo_state_summary', '--')[:100]}")
            print(f"  Question: {state.get('question', '--')[:50]}")
            print(f"  Recent: {state.get('recent_utterance', '--')[:50]}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"[AUDIO] Error writing state: {e}")
            if temp_path.exists():
                temp_path.unlink()
