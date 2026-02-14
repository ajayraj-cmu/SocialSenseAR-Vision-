"""PersonaPlex Bridge — WebSocket client connecting vision server to PersonaPlex.

Bridges audio between the vision server (PCM16 16kHz) and PersonaPlex (Opus 24kHz).
Handles the PersonaPlex binary protocol:
  0x00 = handshake (server ready after system prompts)
  0x01 = audio (Opus-encoded)
  0x02 = text token (UTF-8)

Audio pipeline:
  Unity (PCM16 16kHz) → resample to 24kHz → Opus encode → PersonaPlex
  PersonaPlex → Opus decode → resample to 16kHz → Unity (PCM16 16kHz)
"""

import asyncio
import logging
import threading
import time
from typing import Callable, Optional
from urllib.parse import urlencode

import numpy as np

logger = logging.getLogger(__name__)

# PersonaPlex protocol message types
_MSG_HANDSHAKE = 0x00
_MSG_AUDIO = 0x01
_MSG_TEXT = 0x02

# Audio config
_PERSONAPLEX_SAMPLE_RATE = 24000  # Mimi codec sample rate
_UNITY_SAMPLE_RATE = 16000        # Quest mic sample rate


def _resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample PCM audio using linear interpolation (preserves float32)."""
    if src_rate == dst_rate:
        return pcm
    ratio = dst_rate / src_rate
    n_out = int(len(pcm) * ratio)
    indices = np.arange(n_out, dtype=np.float32) / np.float32(ratio)
    indices = np.clip(indices, 0, len(pcm) - 1)
    idx_floor = indices.astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, len(pcm) - 1)
    frac = (indices - idx_floor.astype(np.float32)).astype(np.float32)
    return pcm[idx_floor] * (np.float32(1.0) - frac) + pcm[idx_ceil] * frac


class PersonaPlexBridge:
    """WebSocket client that connects to PersonaPlex and bridges audio.

    Thread-safe: send_audio() can be called from any thread.
    Callbacks are invoked from the asyncio event loop thread.
    """

    def __init__(
        self,
        url: str,
        text_prompt: str,
        voice_prompt: str = "NATF0.pt",
        on_command: Optional[Callable] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_audio: Optional[Callable[[bytes, int], None]] = None,
        text_temperature: float = 0.7,
        text_topk: int = 25,
        audio_temperature: float = 0.8,
        audio_topk: int = 250,
    ):
        self._base_url = url
        self._text_prompt = text_prompt
        self._voice_prompt = voice_prompt
        self._on_command = on_command
        self._on_text = on_text
        self._on_audio = on_audio
        self._text_temperature = text_temperature
        self._text_topk = text_topk
        self._audio_temperature = audio_temperature
        self._audio_topk = audio_topk

        self._ws = None
        self._ready = False  # True after handshake received
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._send_queue: asyncio.Queue = asyncio.Queue()

        # Diagnostics
        self._audio_frames_sent = 0
        self._audio_msgs_recv = 0
        self._text_tokens_recv = 0
        self._last_diag_time = 0.0

        # Opus encoder/decoder (lazy init — requires sphn)
        self._opus_writer = None
        self._opus_reader = None

        # Opus frame buffering (must send exact frame sizes)
        self._opus_frame_size = 960  # 40ms at 24kHz — standard Opus frame
        self._pcm_buffer = np.empty(0, dtype=np.float32)

        # Text token accumulation
        self._text_buffer = ""

        # Audio response buffer (PCM16 bytes at 16kHz for Unity)
        self._audio_response_lock = threading.Lock()
        self._audio_response_buffer = bytearray()

    def _build_url(self) -> str:
        """Build WebSocket URL with query parameters."""
        params = {
            "text_prompt": self._text_prompt,
            "voice_prompt": self._voice_prompt,
        }
        sep = "&" if "?" in self._base_url else "?"
        return self._base_url + sep + urlencode(params)

    async def connect(self):
        """Connect to PersonaPlex WebSocket server."""
        import aiohttp

        url = self._build_url()
        logger.info(f"Connecting to PersonaPlex: {self._base_url}")

        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(
                url,
                max_msg_size=10 * 1024 * 1024,
                heartbeat=20,
            )
            self._connected = True
            logger.info("PersonaPlex WebSocket connected, waiting for handshake...")
        except Exception as e:
            logger.error(f"PersonaPlex connection failed: {e}")
            if hasattr(self, '_session'):
                await self._session.close()
            raise

    async def _recv_loop(self):
        """Receive messages from PersonaPlex."""
        import aiohttp
        import sphn

        if self._opus_reader is None:
            self._opus_reader = sphn.OpusStreamReader(_PERSONAPLEX_SAMPLE_RATE)

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = msg.data
                    if not data:
                        continue

                    kind = data[0]

                    if kind == _MSG_HANDSHAKE:
                        self._ready = True
                        self._last_diag_time = time.time()
                        logger.info("PersonaPlex handshake received — ready for audio")

                    elif kind == _MSG_AUDIO:
                        # Opus audio response from PersonaPlex
                        self._audio_msgs_recv += 1
                        opus_bytes = data[1:]
                        self._opus_reader.append_bytes(opus_bytes)

                        # Decode available PCM
                        pcm_24k = self._opus_reader.read_pcm()
                        if pcm_24k.shape[-1] > 0:
                            if self._audio_msgs_recv == 1:
                                logger.info(f"PersonaPlex: first audio response ({pcm_24k.shape[-1]} samples)")
                            # Resample 24kHz → 16kHz
                            pcm_16k = _resample_linear(pcm_24k, _PERSONAPLEX_SAMPLE_RATE, _UNITY_SAMPLE_RATE)
                            # Convert to PCM16 bytes
                            pcm16_bytes = (pcm_16k * 32767).astype(np.int16).tobytes()

                            # Buffer for Unity
                            with self._audio_response_lock:
                                self._audio_response_buffer.extend(pcm16_bytes)

                            # Callback
                            if self._on_audio:
                                try:
                                    self._on_audio(pcm16_bytes, _UNITY_SAMPLE_RATE)
                                except Exception as e:
                                    logger.error(f"on_audio callback error: {e}")

                    elif kind == _MSG_TEXT:
                        # Text token from PersonaPlex
                        self._text_tokens_recv += 1
                        text_token = data[1:].decode("utf-8", errors="replace")
                        self._text_buffer += text_token
                        if self._text_tokens_recv <= 5:
                            logger.info(f"PersonaPlex text token #{self._text_tokens_recv}: '{text_token}' (accumulated: '{self._text_buffer[:100]}')")

                        if self._on_text:
                            try:
                                self._on_text(text_token)
                            except Exception as e:
                                logger.error(f"on_text callback error: {e}")

                    else:
                        logger.warning(f"Unknown PersonaPlex message kind: {kind}")

                    # Periodic diagnostics
                    now = time.time()
                    if now - self._last_diag_time >= 10.0:
                        self._last_diag_time = now
                        logger.info(f"PersonaPlex diag: sent={self._audio_frames_sent} frames, "
                                    f"recv_audio={self._audio_msgs_recv}, recv_text={self._text_tokens_recv}")

                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                                  aiohttp.WSMsgType.ERROR):
                    break

        except Exception as e:
            logger.error(f"PersonaPlex recv loop error: {e}")
        finally:
            self._ready = False
            self._connected = False
            logger.info("PersonaPlex recv loop ended")

    async def _send_loop(self):
        """Send queued audio to PersonaPlex."""
        while self._connected:
            try:
                data = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)
                if self._ws and not self._ws.closed:
                    await self._ws.send_bytes(data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"PersonaPlex send error: {e}")
                break

    _audio_drop_count = 0

    def send_audio(self, pcm16_bytes: bytes, sample_rate: int):
        """Send PCM16 audio to PersonaPlex (thread-safe).

        Encodes PCM16 → float32 → resample to 24kHz → Opus → send.
        """
        if not self._ready:
            self._audio_drop_count += 1
            if self._audio_drop_count == 1:
                logger.warning("PersonaPlex bridge not ready — dropping audio (will log every 500)")
            elif self._audio_drop_count % 500 == 0:
                logger.warning(f"PersonaPlex bridge not ready — {self._audio_drop_count} audio chunks dropped")
            return

        import sphn

        if self._opus_writer is None:
            self._opus_writer = sphn.OpusStreamWriter(_PERSONAPLEX_SAMPLE_RATE)

        # PCM16 bytes → float32 numpy (keep float32 dtype — sphn requires it)
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / np.float32(32768.0)

        # Resample to 24kHz if needed
        if sample_rate != _PERSONAPLEX_SAMPLE_RATE:
            samples = _resample_linear(samples, sample_rate, _PERSONAPLEX_SAMPLE_RATE)

        # Append to buffer and send complete Opus frames
        self._pcm_buffer = np.concatenate([self._pcm_buffer, samples])
        frame_size = self._opus_frame_size

        while len(self._pcm_buffer) >= frame_size:
            frame = np.ascontiguousarray(self._pcm_buffer[:frame_size], dtype=np.float32)
            self._pcm_buffer = self._pcm_buffer[frame_size:]

            self._opus_writer.append_pcm(frame)
            opus_bytes = self._opus_writer.read_bytes()

            if len(opus_bytes) > 0:
                self._audio_frames_sent += 1
                if self._audio_frames_sent == 1:
                    logger.info(f"PersonaPlex: sending first Opus frame ({len(opus_bytes)}B)")
                msg = bytes([_MSG_AUDIO]) + opus_bytes
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(self._send_queue.put_nowait, msg)
                else:
                    if self._audio_frames_sent <= 3:
                        logger.warning("PersonaPlex: _loop is None, audio frame dropped")

    def get_audio_response(self) -> Optional[bytes]:
        """Get and clear buffered audio response (PCM16 16kHz mono).

        Returns None if no audio available.
        """
        with self._audio_response_lock:
            if not self._audio_response_buffer:
                return None
            data = bytes(self._audio_response_buffer)
            self._audio_response_buffer.clear()
            return data

    def get_and_clear_text(self) -> str:
        """Get accumulated text tokens and clear the buffer."""
        text = self._text_buffer
        self._text_buffer = ""
        return text

    def is_ready(self) -> bool:
        """True after handshake received from PersonaPlex."""
        return self._ready

    def is_connected(self) -> bool:
        """True if WebSocket is connected."""
        return self._connected

    async def disconnect(self):
        """Close the WebSocket connection."""
        self._connected = False
        self._ready = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if hasattr(self, '_session') and self._session:
            await self._session.close()
        logger.info("PersonaPlex bridge disconnected")

    def start(self):
        """Start the bridge in a background thread with its own event loop."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception as e:
            logger.error(f"PersonaPlex bridge loop error: {e}")
        finally:
            self._loop.close()
            self._loop = None

    async def _run(self):
        """Main async loop: connect, run recv/send, reconnect on failure."""
        while True:
            try:
                await self.connect()

                # Run recv and send concurrently
                recv_task = asyncio.create_task(self._recv_loop())
                send_task = asyncio.create_task(self._send_loop())

                done, pending = await asyncio.wait(
                    [recv_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            except Exception as e:
                logger.error(f"PersonaPlex bridge error: {e}")

            self._ready = False
            self._connected = False

            # Reconnect after delay
            logger.info("PersonaPlex disconnected, reconnecting in 5s...")
            await asyncio.sleep(5)

    def shutdown(self):
        """Stop the bridge."""
        self._connected = False
        self._ready = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("PersonaPlex bridge shut down")
