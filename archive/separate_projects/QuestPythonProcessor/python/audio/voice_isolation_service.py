"""
Voice Isolation Service - optional real-time noise suppression.

Uses DeepFilterNet when available, otherwise falls back to a simple
audio dampening pass-through.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseAudioService

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from df.enhance import enhance, init_df
    DEEPFILTER_AVAILABLE = True
except Exception:
    DEEPFILTER_AVAILABLE = False


class VoiceIsolationService(BaseAudioService):
    """Real-time voice isolation using DeepFilterNet (optional)."""

    def __init__(self, config, state_dir):
        super().__init__(config, state_dir)

        self.available = SOUNDDEVICE_AVAILABLE
        self.use_deepfilter = SOUNDDEVICE_AVAILABLE and TORCH_AVAILABLE and DEEPFILTER_AVAILABLE
        self.voice_isolation_enabled = getattr(config, 'audio_isolation_enabled', True)

        self.input_device_index: Optional[int] = getattr(
            config, 'audio_isolation_input_index', None
        )
        self.output_device_index: Optional[int] = getattr(
            config, 'audio_isolation_output_index', None
        )
        if self.input_device_index is None:
            self.input_device_index = getattr(config, 'audio_mic1_index', None)

        self.sample_rate = 48000 if self.use_deepfilter else 44100
        self.blocksize = int(self.sample_rate * 0.1)

        self.stream = None
        self.last_error: Optional[str] = None

        self.input_queue: Optional[queue.Queue] = None
        self.output_queue: Optional[queue.Queue] = None
        self._worker_thread: Optional[threading.Thread] = None

        self.model = None
        self.df_state = None

    def start(self) -> bool:
        if not self.available:
            print("[AUDIO] Voice isolation unavailable (sounddevice not installed)")
            return False

        if self.running:
            return True

        self.running = True

        if self.use_deepfilter:
            try:
                self.model, self.df_state, _ = init_df()
                self.model.eval()
                self.input_queue = queue.Queue(maxsize=3)
                self.output_queue = queue.Queue(maxsize=3)
                self._worker_thread = threading.Thread(
                    target=self._process_audio_worker, daemon=True
                )
                self._worker_thread.start()
            except Exception as exc:
                self.last_error = f"DeepFilterNet init failed: {exc}"
                print(f"[AUDIO] {self.last_error}")
                self.use_deepfilter = False

        try:
            device = None
            if self.input_device_index is not None or self.output_device_index is not None:
                device = (self.input_device_index, self.output_device_index)

            self.stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                latency="low",
                device=device,
            )
            self.stream.start()
            self.write_state(self.get_state())
            print("[AUDIO] Voice isolation stream started")
            return True
        except Exception as exc:
            self.last_error = f"Failed to start voice isolation: {exc}"
            print(f"[AUDIO] {self.last_error}")
            self.stop()
            return False

    def stop(self) -> None:
        self.running = False

        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.use_deepfilter and self.input_queue is not None:
            try:
                self.input_queue.put_nowait(None)
            except Exception:
                pass

        self.write_state(self.get_state())
        print("[AUDIO] Voice isolation stopped")

    def get_state(self) -> Dict[str, Any]:
        return {
            "voice_isolation_enabled": self.voice_isolation_enabled,
            "active": bool(self.stream is not None and self.running),
            "use_deepfilter": self.use_deepfilter,
            "sample_rate": self.sample_rate,
            "blocksize": self.blocksize,
            "last_error": self.last_error or "",
        }

    def _process_audio_worker(self) -> None:
        if not self.use_deepfilter or self.input_queue is None or self.output_queue is None:
            return

        while self.running:
            try:
                audio_data = self.input_queue.get(timeout=0.1)
                if audio_data is None:
                    break

                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
                with torch.no_grad():
                    enhanced = enhance(self.model, self.df_state, audio_tensor)
                enhanced_np = enhanced.squeeze().cpu().numpy()

                try:
                    self.output_queue.put_nowait(enhanced_np)
                except queue.Full:
                    try:
                        self.output_queue.get_nowait()
                        self.output_queue.put_nowait(enhanced_np)
                    except queue.Empty:
                        pass
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                self.last_error = f"Audio processing error: {exc}"
                continue

    def _audio_callback(self, indata, outdata, frames, _time, status) -> None:
        if status:
            self.last_error = f"Audio stream status: {status}"

        if not self.voice_isolation_enabled:
            outdata[:, 0] = indata[:, 0]
            return

        if self.use_deepfilter and self.input_queue is not None and self.output_queue is not None:
            audio_out = indata[:, 0].copy()
            try:
                self.input_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

            try:
                enhanced = self.output_queue.get_nowait()
                if len(enhanced) >= frames:
                    audio_out = enhanced[:frames]
                else:
                    audio_out = np.pad(enhanced, (0, frames - len(enhanced)), mode="constant")
            except queue.Empty:
                pass

            outdata[:, 0] = audio_out
            return

        # Simple dampening fallback
        audio_in = indata[:, 0].copy()
        dampened = audio_in * 0.3
        if len(dampened) > 1:
            smoothed = np.convolve(dampened, np.ones(3) / 3, mode="same")
            outdata[:, 0] = smoothed[:frames] if len(smoothed) >= frames else dampened[:frames]
        else:
            outdata[:, 0] = dampened[:frames]
