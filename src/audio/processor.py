"""Real-time audio processing with DeepFilterNet for voice isolation."""
import threading
import queue
import numpy as np

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from df.enhance import enhance, init_df
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False


class AudioProcessor:
    """Real-time audio processing with DeepFilterNet for voice isolation."""

    def __init__(self):
        self.available = AUDIO_AVAILABLE
        self.use_deepfilter = AUDIO_AVAILABLE and TORCH_AVAILABLE and DEEPFILTER_AVAILABLE
        self.is_active = False
        self.voice_isolation_enabled = False
        self.audio_stream = None
        self.model = None
        self.df_state = None
        self.fs = 48000 if self.use_deepfilter else 44100
        self.blocksize = int(self.fs * 0.1)

        if self.use_deepfilter:
            self.input_queue = queue.Queue(maxsize=3)
            self.output_queue = queue.Queue(maxsize=3)
            self.processing_lock = threading.Lock()

        if self.use_deepfilter:
            try:
                print("  🎵 Loading DeepFilterNet model...")
                self.model, self.df_state, _ = init_df()
                self.model.eval()
                print("  ✅ Audio processor ready (DeepFilterNet)")
                self.worker_thread = threading.Thread(target=self._process_audio_worker, daemon=True)
                self.worker_thread.start()
            except Exception as e:
                print(f"  ⚠️ DeepFilterNet init failed: {e}")
                self.use_deepfilter = False
                print("  🎵 Using simple audio dampening (no DeepFilterNet)")
        elif self.available:
            print("  🎵 Audio processor ready (simple dampening mode)")

    def _process_audio_worker(self):
        """Worker thread that processes audio asynchronously (DeepFilterNet only)."""
        if not self.use_deepfilter:
            return

        while True:
            try:
                audio_data = self.input_queue.get(timeout=0.1)
                if audio_data is None:
                    break

                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)

                try:
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
                except Exception as e:
                    print(f"⚠️ Audio enhance error: {e}")

                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Audio processing error: {e}")
                continue

    def _audio_callback(self, indata, outdata, frames, time, status):
        """Audio stream callback - processes or passes through audio."""
        if not self.voice_isolation_enabled:
            outdata[:, 0] = indata[:, 0]
            return

        if self.use_deepfilter:
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
        else:
            audio_in = indata[:, 0].copy()
            dampened = audio_in * 0.3

            if len(dampened) > 1:
                smoothed = np.convolve(dampened, np.ones(3)/3, mode='same')
                outdata[:, 0] = smoothed[:frames] if len(smoothed) >= frames else dampened[:frames]
            else:
                outdata[:, 0] = dampened[:frames]

    def start_audio_stream(self):
        """Start the audio processing stream."""
        if not self.available or self.is_active:
            return

        try:
            self.audio_stream = sd.Stream(
                samplerate=self.fs,
                blocksize=self.blocksize,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                latency="low"
            )
            self.audio_stream.start()
            self.is_active = True
            print("  🎵 Audio stream started")
        except Exception as e:
            print(f"  ⚠️ Failed to start audio stream: {e}")

    def stop_audio_stream(self):
        """Stop the audio processing stream."""
        if self.audio_stream and self.is_active:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.is_active = False
                print("  🎵 Audio stream stopped")
            except Exception as e:
                print(f"  ⚠️ Error stopping audio stream: {e}")

    def set_voice_isolation(self, enabled):
        """Enable or disable voice isolation (noise suppression)."""
        self.voice_isolation_enabled = enabled
        if enabled:
            print("  🎵 Voice isolation ENABLED (surrounding audio dampened)")
        else:
            print("  🎵 Voice isolation DISABLED (full audio passthrough)")

    def close(self):
        """Clean up resources."""
        self.stop_audio_stream()
        if self.use_deepfilter and hasattr(self, 'input_queue'):
            self.input_queue.put(None)
