"""Test client — pipelined webcam sender + async response receiver.

Architecture:
  FrameGrabber thread  -> always has the latest webcam frame ready
  AudioStreamer thread  -> captures mic PCM16, sends to server for voice agent
  sender loop          -> encodes JPEG + sends protobuf (fire-and-forget)
  receiver() task      -> collects server responses, tracks round-trip latency
  CommandInput thread  -> reads console commands (blur face, etc.)

All display is handled by the server dashboard window.

Commands (type in console or speak via voice):
    blur <object>       — blur person, face, wall, etc.
    unblur <object>     — stop blurring that object
    <object> blur       — same as "blur <object>"
    clear               — remove all blur effects
    list                — show active blur targets
    help                — show available commands

Voice commands (say "Hey Vibe" to start, "Thank you" to end):
    "Hey Vibe, blur the laptop, thank you"
    "Hey Vibe, it's too bright, dim everything, thank you"
    "Hey Vibe, stop blurring the monitor, thank you"

Usage:
    python -m server.main --device cuda
    python -m server.test_client
    python -m server.test_client --whisper   # enable audio streaming to server
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
from server.commands import CommandInput
from server.dashboard import Dashboard



class AudioStreamer:
    """Background thread: captures PCM16 audio from mic and queues for server.

    Instead of running local Whisper, sends raw audio to the server where
    VoiceAgent handles wake word detection, transcription, and command planning.
    Requires: pip install sounddevice
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.5):
        self._sample_rate = sample_rate
        self._chunk_duration = chunk_duration
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._available = False

    def start(self):
        try:
            import sounddevice as sd  # noqa: F401
            self._available = True
            self._thread.start()
            print(f"Audio streaming: mic -> server (PCM16 {self._sample_rate}Hz)")
            print("  Voice commands: say 'Hey Vibe' to start, 'Thank you' to end")
        except ImportError as e:
            print(f"Audio streaming unavailable: {e}")
            print("  Install with: pip install sounddevice")
            self._available = False

    @property
    def available(self):
        return self._available

    def _run(self):
        import sounddevice as sd

        sr = self._sample_rate
        chunk_samples = int(sr * self._chunk_duration)

        # Accumulator for continuous stream → fixed-size chunks
        buf = np.zeros(0, dtype=np.float32)

        def callback(indata, frames, time_info, status):
            nonlocal buf
            buf = np.append(buf, indata[:, 0])
            while len(buf) >= chunk_samples:
                segment = buf[:chunk_samples]
                buf = buf[chunk_samples:]
                pcm16 = (segment * 32767).astype(np.int16)
                self._queue.put(pcm16.tobytes())

        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                blocksize=chunk_samples, callback=callback):
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"  [Audio] error: {e}")
            time.sleep(1)

    def get_audio_chunks(self) -> list[tuple[bytes, int, int]]:
        """Non-blocking: return all queued (pcm16_bytes, sample_rate, num_samples)."""
        chunks = []
        while not self._queue.empty():
            try:
                pcm_bytes = self._queue.get_nowait()
                num_samples = len(pcm_bytes) // 2  # 2 bytes per PCM16 sample
                chunks.append((pcm_bytes, self._sample_rate, num_samples))
            except queue.Empty:
                break
        return chunks

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
                     whisper_enabled: bool = False, show_dashboard: bool = False):
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
            if show_dashboard:
                print("  Dashboard: LOCAL window")
            else:
                print("  Display is on the SERVER window.")
            print("Commands (type in console or dashboard):")
            print("  blur <object>  — blur person, face, wall, monitor, etc.")
            print("  unblur <object> — stop blurring that object")
            print("  clear          — remove all blur effects")
            print("  list           — show active blur targets")
            print("  help           — show available labels")

            # --- Command input ---
            cmd_input = CommandInput()
            cmd_input.start()
            audio_streamer = None
            if whisper_enabled:
                audio_streamer = AudioStreamer()
                audio_streamer.start()

            # --- Dashboard (local display when --show) ---
            dashboard = None
            pending_commands: list[str] = []
            # Active prompts/effects derived from server response
            server_prompts: set[str] = set()
            server_effects: dict[str, dict] = {}

            def on_dashboard_command(cmd: str):
                pending_commands.append(cmd)

            if show_dashboard:
                dashboard = Dashboard(
                    window_name="SocialSenseAR Client",
                    on_command=on_dashboard_command,
                    get_active_prompts=lambda: server_prompts,
                    get_effects=lambda: server_effects,
                )

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
            _recv_diag_interval = 60  # Log every 60th message
            _recv_unique_fps = set()   # Unique fingerprints seen

            async def receiver():
                nonlocal recv_count, latest_resp, last_raw_response
                nonlocal server_prompts, server_effects
                try:
                    async for raw in ws:
                        try:
                            t_recv = time.perf_counter()
                            resp = pb.ServerMessage()
                            resp.ParseFromString(raw)
                            latest_resp = resp
                            recv_count += 1

                            # --- Receiver diagnostic: log raw + parsed data ---
                            if recv_count % _recv_diag_interval == 0:
                                seg_info = []
                                for s in resp.segments:
                                    seg_info.append(
                                        f"cx={s.center_x:.4f} cy={s.center_y:.4f} "
                                        f"rle={len(s.rle_mask)}B lbl={s.label}"
                                    )
                                print(
                                    f"  [Recv#{recv_count}] raw={len(raw)}B "
                                    f"fid={resp.frame_id} mfid={resp.masks_frame_id} "
                                    f"segs={len(resp.segments)} | {seg_info}"
                                )

                            # Extract active prompts/effects from server response
                            prompts = set()
                            effects = {}
                            for seg in resp.segments:
                                if seg.effect and seg.effect.effect_type and seg.effect.effect_type != "none":
                                    prompts.add(seg.label)
                                    effects[seg.label] = {
                                        "type": seg.effect.effect_type,
                                        "intensity": seg.effect.intensity,
                                        "invert": seg.effect.params.get("invert", "") == "true",
                                    }
                            server_prompts = prompts
                            server_effects = effects

                            # Detect REAL mask updates by comparing segment content
                            mask_fp = _mask_fingerprint(resp)
                            _recv_unique_fps.add(mask_fp)
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
                        except Exception as e:
                            import traceback
                            print(f"  [Recv] ERROR on msg #{recv_count}: {type(e).__name__}: {e}")
                            traceback.print_exc()
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    import traceback
                    print(f"  [Recv] FATAL: {type(e).__name__}: {e}")
                    traceback.print_exc()

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

                    # --- Update dashboard if enabled (show webcam even before server responds) ---
                    if dashboard:
                        display_frame, _ = grabber.get()
                        if display_frame is not None:
                            # No flip needed - webcam is already right-side up for display
                            # Use segments from response if available, otherwise empty
                            resp_ref = latest_resp  # snapshot reference
                            segments = list(resp_ref.segments) if resp_ref else []
                            # Log what dashboard will use (every 120th frame)
                            if send_count % 120 == 0 and resp_ref:
                                dash_info = [(f"cx={s.center_x:.4f}", f"rle={len(s.rle_mask)}B")
                                             for s in segments]
                                print(f"  [DashSnap#{send_count}] resp_id={id(resp_ref)} "
                                      f"fid={resp_ref.frame_id} mfid={resp_ref.masks_frame_id} "
                                      f"segs={dash_info}")
                            # Extract voice agent state if present
                            va_state = None
                            if latest_resp:
                                va = latest_resp.conversation.voice_agent
                                if va.listening or va.recording or va.last_response:
                                    va_state = {
                                        'listening': va.listening,
                                        'recording': va.recording,
                                        'partial_transcript': va.partial_transcript,
                                        'last_command': va.last_command,
                                        'last_response': va.last_response,
                                    }
                            dashboard.update(display_frame, segments, va_state)
                            # Yield after blocking dashboard render so receiver can drain messages
                            await asyncio.sleep(0)

                    # --- Stream audio to server (voice agent handles it) ---
                    if audio_streamer and audio_streamer.available:
                        for pcm_bytes, sr, n_samples in audio_streamer.get_audio_chunks():
                            audio_msg = pb.ClientMessage()
                            audio_msg.audio.pcm16_data = pcm_bytes
                            audio_msg.audio.sample_rate = sr
                            audio_msg.audio.num_samples = n_samples
                            await ws.send(audio_msg.SerializeToString())

                    # --- Process console + dashboard commands ---
                    all_cmds = cmd_input.get_commands()
                    all_cmds.extend(pending_commands)
                    pending_commands.clear()
                    for raw_cmd in all_cmds:
                        raw_cmd = raw_cmd.strip()
                        if not raw_cmd:
                            continue
                        # Send raw text to server — server runs it through Gemini NLP
                        ctrl = pb.ClientMessage()
                        ctrl.control.command = raw_cmd
                        await ws.send(ctrl.SerializeToString())
                        print(f"  > Sent: {raw_cmd}")

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
                            f"UniqFP: {len(_recv_unique_fps)} | "
                            f"Frames: {send_count}"
                        )

                    # Throttle to target FPS
                    target_time = t_loop + interval
                    while time.perf_counter() < target_time:
                        await asyncio.sleep(0)
                    # Always yield at least once so receiver coroutine can run
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
                if audio_streamer:
                    audio_streamer.stop()
                if dashboard:
                    dashboard.close()

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
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio streaming (audio is enabled by default)")
    parser.add_argument("--show", action="store_true",
                        help="Show local dashboard (required for remote servers like Modal)")
    args = parser.parse_args()

    # Auto-enable --show for remote (non-localhost) URLs
    show = args.show
    if not show and "localhost" not in args.url and "127.0.0.1" not in args.url:
        print("Remote URL detected, enabling local dashboard (--show)")
        show = True

    audio = not args.no_audio
    asyncio.run(run_client(args.url, args.fps, args.camera, audio, show))


if __name__ == "__main__":
    main()
