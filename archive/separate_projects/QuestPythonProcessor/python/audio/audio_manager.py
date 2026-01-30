"""
Audio Manager - Orchestrates all audio services.

Manages starting/stopping audio services alongside the video pipeline.
Provides integration between emotion detection and context services.
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .base import BaseAudioService


class AudioManager:
    """Manages audio services that run parallel to the video pipeline."""

    def __init__(self, config, services: Optional[List[str]] = None):
        """Initialize audio manager.

        Args:
            config: Configuration object with audio settings
            services: List of service names to enable, or None for defaults
        """
        self.config = config
        self.services: Dict[str, BaseAudioService] = {}
        self.running = False

        # Get state directory from config
        state_dir = getattr(config, 'audio_state_dir', '~/Downloads/Nex/conve_context')
        self.state_dir = Path(state_dir).expanduser()

        # Initialize requested services (include emotion by default for conversation helper)
        requested = services or getattr(config, 'audio_services', ['context', 'emotion'])
        self._init_services(requested)

    def _init_services(self, service_names: List[str]) -> None:
        """Initialize requested audio services.

        Args:
            service_names: List of service names to initialize
        """
        from . import AUDIO_SERVICES

        for name in service_names:
            if name in AUDIO_SERVICES:
                try:
                    service = AUDIO_SERVICES[name](self.config, self.state_dir)
                    self.services[name] = service
                    print(f"[AUDIO] Initialized {name} service")
                except Exception as e:
                    print(f"[AUDIO] Failed to initialize {name} service: {e}")
            else:
                print(f"[AUDIO] Unknown service: {name}")

    def start(self) -> None:
        """Start all audio services."""
        if self.running:
            return

        self.running = True

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        for name, service in self.services.items():
            try:
                if service.start():
                    print(f"[AUDIO] Started {name} service")
                else:
                    print(f"[AUDIO] Failed to start {name} service")
            except Exception as e:
                print(f"[AUDIO] Error starting {name} service: {e}")

    def stop(self) -> None:
        """Stop all audio services."""
        if not self.running:
            return

        self.running = False

        for name, service in self.services.items():
            try:
                service.stop()
                print(f"[AUDIO] Stopped {name} service")
            except Exception as e:
                print(f"[AUDIO] Error stopping {name} service: {e}")

    def get_state(self, service_name: str = "context") -> dict:
        """Get current state from a service.

        Args:
            service_name: Name of the service to query

        Returns:
            State dictionary, or empty dict if service not found
        """
        if service_name in self.services:
            return self.services[service_name].get_state()
        return {}

    def submit_face(self, face_image: np.ndarray, bbox=None) -> None:
        """Submit a face image to the emotion service for detection.

        Call this from the video pipeline with the cropped face of the
        person being tracked.

        Args:
            face_image: BGR face image (cropped from frame)
            bbox: Optional bounding box (x, y, w, h) for context
        """
        if 'emotion' in self.services:
            self.services['emotion'].submit_face(face_image, bbox)

        # Sync emotion to context service (emotion detection runs async)
        self._sync_emotion_to_context()

    def _sync_emotion_to_context(self) -> None:
        """Sync current emotion state from emotion service to context service."""
        if 'emotion' in self.services and 'context' in self.services:
            emotion_state = self.services['emotion'].get_state()
            if emotion_state:
                emotion = emotion_state.get('emotion', 'neutral')
                emotion_display = emotion_state.get('emotion_display', 'Neutral')
                self.services['context'].set_emotion(emotion, emotion_display)

    def get_speaking_state(self) -> dict:
        """Get current speaking state from context service.

        Returns:
            Dict with 'is_other_speaking' and 'is_user_speaking' booleans
        """
        if 'context' in self.services:
            state = self.services['context'].get_state()
            return {
                'is_other_speaking': state.get('is_other_speaking', False),
                'is_user_speaking': state.get('is_user_speaking', False)
            }
        return {'is_other_speaking': False, 'is_user_speaking': False}

    def get_emotion(self) -> dict:
        """Get current emotion from emotion service.

        Returns:
            Dict with emotion info or empty dict
        """
        if 'emotion' in self.services:
            return self.services['emotion'].get_state()
        return {}

    def list_microphones(self) -> None:
        """Print available microphones for configuration."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()

            print("\n" + "=" * 60)
            print("AVAILABLE MICROPHONES")
            print("=" * 60)

            default_index = p.get_default_input_device_info()['index']

            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    default = " (DEFAULT)" if i == default_index else ""
                    print(f"  [{i}] {info['name']}{default}")

            print("=" * 60)
            print("Use --mic1 and --mic2 to specify microphone indices")
            print()

            p.terminate()
        except Exception as e:
            print(f"[AUDIO] Could not list microphones: {e}")
