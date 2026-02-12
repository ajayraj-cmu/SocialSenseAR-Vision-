"""Whisper transcriber — sends audio to OpenAI Whisper API.

Extracted from QuestPythonProcessor/python/audio/context_service.py.
Takes raw PCM16 bytes, writes a temp WAV, sends to Whisper, returns text.
"""

import os
import wave
import tempfile
import logging

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Transcribes speech audio using OpenAI Whisper API."""

    def __init__(self, api_key: str = None, model: str = "whisper-1"):
        self._client = None
        self._model = model
        self._api_key = api_key
        self._available = False

    def initialize(self):
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OpenAI API key — transcriber disabled")
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._available = True
            logger.info(f"Whisper transcriber ready (model={self._model})")
        except Exception as e:
            logger.error(f"Whisper init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def transcribe(self, pcm16_bytes: bytes, sample_rate: int = 16000, mode: str = "listening") -> str:
        """Transcribe PCM16 audio bytes to text.

        Args:
            pcm16_bytes: Raw PCM16 mono audio data.
            sample_rate: Sample rate of the audio (default 16000).

        Returns:
            Transcribed text, or empty string on failure.
        """
        if not self._available:
            return ""

        tmp_path = None
        try:
            # Write to temp WAV (closed immediately so Windows doesn't lock it)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16_bytes)

            with open(tmp_path, "rb") as audio_file:
                transcript = self._client.audio.transcriptions.create(
                    model=self._model,
                    file=audio_file,
                    language="en",
                )

            text = transcript.text.strip()
            if text:
                logger.debug(f"Whisper: \"{text[:80]}\"")
            return text

        except Exception as e:
            logger.warning(f"Whisper error: {e}")
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def shutdown(self):
        self._client = None
        self._available = False
