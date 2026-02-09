"""Local speech-to-text using faster-whisper (CTranslate2).

Zero network latency, zero API cost. Runs Whisper models locally.
Uses 'tiny.en' for listening phase (fast wake word detection)
and 'base.en' for recording phase (accurate command transcription).
"""

import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class LocalTranscriber:
    """Local speech-to-text using faster-whisper.

    Same interface as WhisperTranscriber but runs entirely on-device.
    Two models loaded: a fast one for listening (wake word detection)
    and an accurate one for recording (command transcription).
    """

    # Vocabulary hint so Whisper biases toward our domain words
    INITIAL_PROMPT = "vibe blur pixelate dim person laptop phone screen"

    def __init__(
        self,
        listening_model: str,
        recording_model: str,
        beam_size: int = 5,
        device: str = "auto",
    ):
        self._listening_model_name = listening_model
        self._recording_model_name = recording_model
        self._beam_size = beam_size
        self._device = device
        self._listening_model = None
        self._recording_model = None
        self._available = False

    def initialize(self):
        """Load Whisper models. Auto-detects CUDA vs CPU."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper not installed: pip install faster-whisper")
            return

        # Auto-detect device
        device = self._device
        compute_type = "float16"
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device == "cpu":
            compute_type = "int8"

        try:
            t0 = time.perf_counter()
            self._listening_model = WhisperModel(
                self._listening_model_name,
                device=device,
                compute_type=compute_type,
            )
            t1 = time.perf_counter()
            logger.info(
                f"Listening model '{self._listening_model_name}' loaded "
                f"on {device} ({compute_type}) in {(t1-t0)*1000:.0f}ms"
            )

            # Only load recording model if different from listening model
            if self._recording_model_name != self._listening_model_name:
                self._recording_model = WhisperModel(
                    self._recording_model_name,
                    device=device,
                    compute_type=compute_type,
                )
                t2 = time.perf_counter()
                logger.info(
                    f"Recording model '{self._recording_model_name}' loaded "
                    f"on {device} ({compute_type}) in {(t2-t1)*1000:.0f}ms"
                )
            else:
                self._recording_model = self._listening_model
                logger.info("Recording model = listening model (same)")

            self._available = True
        except Exception as e:
            logger.error(f"Local transcriber init failed: {e}", exc_info=True)

    @property
    def available(self) -> bool:
        return self._available

    def transcribe(
        self,
        pcm16_bytes: bytes,
        sample_rate: int = 16000,
        mode: str = "listening",
    ) -> str:
        """Transcribe PCM16 audio bytes to text.

        Args:
            pcm16_bytes: Raw PCM16 mono audio data.
            sample_rate: Sample rate (default 16000).
            mode: "listening" uses fast model, "recording" uses accurate model.

        Returns:
            Transcribed text, or empty string on failure.
        """
        if not self._available:
            return ""

        try:
            # Convert PCM16 bytes → float32 numpy array (faster-whisper expects this)
            samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample if needed (faster-whisper expects 16kHz)
            if sample_rate != 16000:
                ratio = 16000 / sample_rate
                new_len = int(len(samples) * ratio)
                samples = np.interp(
                    np.linspace(0, len(samples) - 1, new_len),
                    np.arange(len(samples)),
                    samples,
                )

            model = self._recording_model if mode == "recording" else self._listening_model

            t0 = time.perf_counter()
            segments, info = model.transcribe(
                samples,
                language="en",
                beam_size=self._beam_size,
                initial_prompt=self.INITIAL_PROMPT,
                hotwords="vibe blur pixelate dim person laptop phone screen clear",
                vad_filter=False,  # We have our own energy gate — don't let VAD discard speech
            )

            # Collect all segment texts
            text = " ".join(seg.text.strip() for seg in segments).strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if text:
                logger.debug(f"Local [{mode}] ({elapsed_ms:.0f}ms): \"{text[:80]}\"")

            return text

        except Exception as e:
            logger.warning(f"Local transcription error: {e}")
            return ""

    def shutdown(self):
        """Free models."""
        self._listening_model = None
        self._recording_model = None
        self._available = False
