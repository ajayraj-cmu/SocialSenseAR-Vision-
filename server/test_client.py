"""Test client — pipelined webcam sender + async response receiver.

Architecture:
  FrameGrabber thread  -> always has the latest webcam frame ready
  sender loop          -> encodes JPEG + sends protobuf (fire-and-forget)
  receiver() task      -> collects server responses, tracks round-trip latency
  CommandInput thread  -> reads console commands (blur face, etc.)

All display is handled by the server dashboard window.

Commands (type in console or speak via Whisper):
    blur <object>       — blur person, face, wall, etc.
    unblur <object>     — stop blurring that object
    <object> blur       — same as "blur <object>"
    clear               — remove all blur effects
    list                — show active blur targets
    help                — show available commands

Usage:
    python -m server.main --device cuda
    python -m server.test_client
    python -m server.test_client --whisper   # enable voice commands
"""

import asyncio
import argparse
import collections
import queue
import threading
import time
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32" and __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np
import websockets

sys.path.insert(0, ".")
from server.proto import socialsense_pb2 as pb
from server.commands import parse_command, KNOWN_LABELS, LABEL_ALIASES, CommandInput



class WhisperInput:
    """Background thread: listens to microphone, transcribes speech as commands.

    Uses openai-whisper (tiny.en) on CPU so it doesn't contend with SAM on GPU.
    Requires: pip install openai-whisper sounddevice
    """

    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._available = False

    def start(self):
        try:
            import whisper  # noqa: F401
            import sounddevice as sd  # noqa: F401
            self._available = True
            self._thread.start()
            print("Whisper: listening on microphone (tiny.en model)")
        except ImportError as e:
            print(f"Whisper unavailable: {e}")
            print("  Install with: pip install openai-whisper sounddevice")
            self._available = False

    @property
    def available(self):
        return self._available

    def _run(self):
        import whisper
        import sounddevice as sd

        model = whisper.load_model("tiny.en", device="cpu")
        sr = 16000
        chunk_duration = 0.5  # seconds per recording chunk
        silence_threshold = 0.015  # RMS energy threshold
        min_speech_chunks = 2  # minimum ~1s of speech
        max_speech_chunks = 10  # maximum ~5s

        while self._running:
            try:
                audio_chunks = []
                silence_count = 0
                recording = False

                # Listen for speech
                while self._running:
                    chunk = sd.rec(
                        int(sr * chunk_duration), samplerate=sr,
                        channels=1, dtype="float32",
                    )
                    sd.wait()
                    energy = float(np.sqrt(np.mean(chunk ** 2)))

                    if energy > silence_threshold:
                        audio_chunks.append(chunk)
                        silence_count = 0
                        recording = True
                    elif recording:
                        silence_count += 1
                        if silence_count >= 2 or len(audio_chunks) >= max_speech_chunks:
                            break
                    # Not recording and no speech — loop back

                if not self._running:
                    break

                if len(audio_chunks) >= min_speech_chunks:
                    audio = np.concatenate(audio_chunks).flatten()
                    result = model.transcribe(
                        audio, language="en", fp16=False,
                        no_speech_threshold=0.5,
                    )
                    text = result["text"].strip().strip(".")
                    if text and len(text) > 1:
                        print(f"  [Whisper] heard: \"{text}\"")
                        self._queue.put(text)

            except Exception as e:
                print(f"  [Whisper] error: {e}")
                time.sleep(1)

    def get_commands(self) -> list[str]:
        """Non-blocking: return all queued transcriptions."""
        cmds = []
        while not self._queue.empty():
            try:
                cmds.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def stop(self):
        self._running = False


class FrameGrabber:
    """Reads webcam frames in a background thread.

    Always keeps the latest frame available. Camera read never blocks
    the async event loop.
    """

    def __init__(self, camera_idx: int, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(camera_idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60fps from webcam
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def opened(self):
        return self._cap.isOpened()

    def start(self):
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self._thread.start()

    def _run(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1

    def get(self):
        """Return (frame_bgr, frame_id). frame may be None initially."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy(), self._frame_id
            return None, 0

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


def _mask_fingerprint(resp) -> str:
    """Content fingerprint of segment positions + masks. Ignores frame_id/timestamp."""
    if not resp or not resp.segments:
        return ""
    parts = []
    for seg in resp.segments:
        # Position (4 decimal places) + label + first 32 bytes of mask
        parts.append(f"{seg.center_x:.4f},{seg.center_y:.4f},{seg.label}")
        if seg.rle_mask:
            parts.append(seg.rle_mask[:32].hex())
    return "|".join(parts)


async def run_client(url: str, target_fps: int, camera: int,
                     whisper_enabled: bool = False):
    # Check protobuf backend
    try:
        from google.protobuf.internal import api_implementation
        impl = api_implementation.Type()
        print(f"Protobuf backend: {impl}")
        if impl == "python":
            print("WARNING: Pure Python protobuf — expect slow parse/serialize.")
            print("         pip install --upgrade protobuf  (needs C extension)")
    except Exception:
        pass

    # Start camera thread
    grabber = FrameGrabber(camera)
    if not grabber.opened:
        print(f"ERROR: Cannot open camera {camera}")
        return
    grabber.start()

    # Wait for first frame
    for _ in range(200):
        frame, _ = grabber.get()
        if frame is not None:
            break
        await asyncio.sleep(0.025)
    else:
        print("ERROR: No camera frames after 5s")
        grabber.stop()
        return

    h, w = frame.shape[:2]
    print(f"Camera ready: {w}x{h}")
    print(f"Connecting to {url}...")

    try:
        async with websockets.connect(
            url,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            print(f"Connected! Target {target_fps} fps. Ctrl+C to quit.")
            print("  Display is on the SERVER window.")
            print("Commands (type in console):")
            print("  blur <object>  — blur person, face, wall, monitor, etc.")
            print("  unblur <object> — stop blurring that object")
            print("  clear          — remove all blur effects")
            print("  list           — show active blur targets")
            print("  help           — show available labels")

            # --- Command input ---
            cmd_input = CommandInput()
            cmd_input.start()
            whisper_input = None
            if whisper_enabled:
                whisper_input = WhisperInput()
                whisper_input.start()

            # --- Blur state (sent to server, local tracking for console feedback) ---
            blur_targets: set[str] = set()

            # --- Shared state ---
            send_count = 0
            recv_count = 0
            latest_resp = None
            # FIFO queue for RTT: sender pushes timestamp, receiver pops.
            # Works because server responds in order (1 response per frame).
            send_deque: collections.deque[float] = collections.deque()
            rtts: list[float] = []
            t_start = time.perf_counter()
            interval = 1.0 / target_fps
            last_sent_fid = -1
            last_jpeg = None  # cached JPEG for re-sending between webcam frames
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            # SAM FPS tracking (detect actual SAM updates from server)
            last_raw_response = None  # raw bytes of last response (changes when SAM updates)
            sam_update_times: list[float] = []

            # --- Receiver coroutine (runs concurrently) ---
            async def receiver():
                nonlocal recv_count, latest_resp, last_raw_response
                try:
                    async for raw in ws:
                        t_recv = time.perf_counter()
                        resp = pb.ServerMessage()
                        resp.ParseFromString(raw)
                        latest_resp = resp
                        recv_count += 1

                        # Detect REAL mask updates by comparing segment content
                        # (positions + masks), ignoring frame_id/timestamp that
                        # change every frame and inflated the old metric.
                        mask_fp = _mask_fingerprint(resp)
                        if mask_fp != last_raw_response:
                            last_raw_response = mask_fp
                            sam_update_times.append(t_recv)
                            if len(sam_update_times) > 120:
                                sam_update_times.pop(0)

                        # Round-trip time (FIFO — server responds in send order)
                        if send_deque:
                            sent_at = send_deque.popleft()
                            rtt_ms = (t_recv - sent_at) * 1000
                            rtts.append(rtt_ms)
                except websockets.exceptions.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receiver())

            # --- Main send loop ---
            stats_interval = max(target_fps * 2, 30)

            try:
                while True:
                    t_loop = time.perf_counter()

                    # Grab latest camera frame
                    frame, fid = grabber.get()
                    if frame is None:
                        await asyncio.sleep(0.001)
                        continue
                    if fid != last_sent_fid:
                        # New frame from webcam — encode it
                        last_sent_fid = fid
                        frame_send = cv2.flip(frame, 0)
                        _, jpeg = cv2.imencode(".jpg", frame_send, encode_params)
                        last_jpeg = jpeg
                    elif last_jpeg is None:
                        await asyncio.sleep(0.001)
                        continue
                    # Re-send latest JPEG (maintains high frame rate even at 30fps webcam)
                    jpeg = last_jpeg

                    # Build protobuf (use send_count as frame_id so server sees unique frames)
                    msg = pb.ClientMessage()
                    msg.frame_id = send_count + 1
                    msg.timestamp_ms = time.time() * 1000
                    msg.frame.jpeg_data = jpeg.tobytes()
                    msg.frame.width = w
                    msg.frame.height = h
                    msg.frame.quality = 70
                    data = msg.SerializeToString()

                    # Send (fire-and-forget — don't wait for response)
                    send_deque.append(time.perf_counter())
                    await ws.send(data)
                    send_count += 1

                    # --- Process console + whisper commands ---
                    all_cmds = cmd_input.get_commands()
                    if whisper_input and whisper_input.available:
                        all_cmds.extend(whisper_input.get_commands())
                    for raw_cmd in all_cmds:
                        action, target = parse_command(raw_cmd)
                        if action == "blur" and target:
                            blur_targets.add(target)
                            print(f"  + Blur: {target}  (active: {', '.join(sorted(blur_targets))})")
                        elif action == "unblur" and target:
                            blur_targets.discard(target)
                            print(f"  - Unblur: {target}  (active: {', '.join(sorted(blur_targets)) or 'none'})")
                        elif action == "clear":
                            blur_targets.clear()
                            print("  Cleared all blur targets.")
                        elif action == "list":
                            if blur_targets:
                                print(f"  Active blur targets: {', '.join(sorted(blur_targets))}")
                            else:
                                print("  No active blur targets.")
                        elif action == "help":
                            print("  Available labels: " + ", ".join(sorted(KNOWN_LABELS)))
                            print("  Aliases: " + ", ".join(f"{k}->{v}" for k, v in sorted(LABEL_ALIASES.items())))
                            print("  Commands: blur <obj>, unblur <obj>, clear, list, help")

                        # Send command to server to update SAM3 active prompts
                        if action in ("blur", "unblur", "clear"):
                            ctrl = pb.ClientMessage()
                            ctrl.control.command = raw_cmd.strip()
                            await ws.send(ctrl.SerializeToString())

                    # Print stats periodically
                    if send_count % stats_interval == 0 and send_count > 0:
                        elapsed = time.perf_counter() - t_start
                        sfps = send_count / elapsed
                        rfps = recv_count / elapsed
                        avg_rtt = np.mean(rtts[-120:]) if rtts else 0
                        p95_rtt = np.percentile(rtts[-120:], 95) if len(rtts) >= 2 else 0
                        n_segs = len(latest_resp.segments) if latest_resp else 0
                        if len(sam_update_times) >= 2:
                            sam_dt = sam_update_times[-1] - sam_update_times[0]
                            sam_fps = (len(sam_update_times) - 1) / sam_dt if sam_dt > 0 else 0
                        else:
                            sam_fps = 0
                        print(
                            f"Client: {rfps:5.1f} fps | "
                            f"Mask: {sam_fps:4.1f} fps | "
                            f"RTT avg: {avg_rtt:5.1f}ms p95: {p95_rtt:5.1f}ms | "
                            f"Segs: {n_segs:2d} | "
                            f"Frames: {send_count}"
                        )

                    # Throttle to target FPS (busy-wait to avoid Windows timer granularity)
                    target_time = t_loop + interval
                    while time.perf_counter() < target_time:
                        await asyncio.sleep(0)

            except KeyboardInterrupt:
                print("\nStopped by user")
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
                cmd_input.stop()
                if whisper_input:
                    whisper_input.stop()

    except (ConnectionRefusedError, OSError) as e:
        print(f"ERROR: Cannot connect to {url} ({e})")
        print("Make sure the server is running: python -m server.main")
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        grabber.stop()


def main():
    parser = argparse.ArgumentParser(description="SocialSenseAR test client")
    parser.add_argument("--url", default="ws://localhost:8765", help="Server URL")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--whisper", action="store_true",
                        help="Enable voice commands via Whisper (requires openai-whisper + sounddevice)")
    args = parser.parse_args()
    asyncio.run(run_client(args.url, args.fps, args.camera, args.whisper))


if __name__ == "__main__":
    main()
