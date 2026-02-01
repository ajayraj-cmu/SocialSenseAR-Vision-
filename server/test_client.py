"""Test client — pipelined webcam sender + async response receiver.

Architecture:
  FrameGrabber thread  -> always has the latest webcam frame ready
  sender loop          -> encodes JPEG + sends protobuf (fire-and-forget)
  receiver() task      -> collects server responses, tracks round-trip latency
  display (optional)   -> cv2 overlay with latest response
  CommandInput thread  -> reads console commands (blur face, etc.)

Commands (type in console or speak via Whisper):
    blur <object>       — blur person, face, wall, etc.
    unblur <object>     — stop blurring that object
    <object> blur       — same as "blur <object>"
    clear               — remove all blur effects
    list                — show active blur targets
    help                — show available commands

Usage:
    python -m server.main --device cuda
    python -m server.test_client --show
    python -m server.test_client --show --whisper   # enable voice commands
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
from server.encoding.rle import decode_rle

# Known object labels that SAM3 can detect
KNOWN_LABELS = {
    "person", "face", "chair", "table", "desk", "couch", "monitor",
    "laptop", "lamp", "wall", "floor", "door", "window",
    "head", "hair", "torso", "body", "shirt", "arm", "hand",
    "shoulder", "skin", "pants", "leg", "human",
}

# Aliases: map common words to SAM3 labels
LABEL_ALIASES = {
    "computer": "monitor", "screen": "monitor", "display": "monitor",
    "sofa": "couch", "settee": "couch",
    "light": "lamp", "ceiling light": "lamp",
    "ground": "floor", "carpet": "floor",
    "background": "wall",  # blur wall = blur background
    "people": "person", "human": "person", "man": "person", "woman": "person",
    "hands": "hand", "arms": "arm", "legs": "leg",
    "furniture": "chair",  # best effort
}


def parse_command(text: str) -> tuple[str, str | None]:
    """Parse a command string into (action, target).

    Supports:
        "blur face"      -> ("blur", "face")
        "face blur"      -> ("blur", "face")
        "unblur person"  -> ("unblur", "person")
        "person unblur"  -> ("unblur", "person")
        "clear"          -> ("clear", None)
        "list"           -> ("list", None)
        "help"           -> ("help", None)
    """
    text = text.strip().lower()
    if not text:
        return ("", None)

    # Single-word commands
    if text in ("clear", "reset", "clearall"):
        return ("clear", None)
    if text in ("list", "ls", "show", "status"):
        return ("list", None)
    if text in ("help", "?", "commands"):
        return ("help", None)

    words = text.split()

    # Two-word: "blur face" or "face blur"
    action = None
    target = None
    for w in words:
        if w in ("blur", "blr", "blue"):  # handle typos
            action = "blur"
        elif w in ("unblur", "unblr", "remove", "stop", "off"):
            action = "unblur"
        else:
            target = w

    if action is None:
        # Default: "blur" if just a label name
        action = "blur"
        target = text.split()[0]

    if target:
        # Resolve aliases
        target = LABEL_ALIASES.get(target, target)

    return (action, target)


class CommandInput:
    """Background thread that reads console commands (stdin)."""

    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                line = input()
                if line.strip():
                    self._queue.put(line.strip())
            except EOFError:
                break
            except Exception:
                break

    def get_commands(self) -> list[str]:
        """Non-blocking: return all queued commands."""
        cmds = []
        while not self._queue.empty():
            try:
                cmds.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def stop(self):
        self._running = False


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


# Matches bright_colors from sam_gemini_voice.py
BRIGHT_COLORS = [
    (0, 255, 255),      # Cyan
    (255, 0, 255),      # Magenta
    (0, 255, 0),        # Bright Green
    (255, 255, 0),      # Yellow
    (255, 128, 0),      # Orange
    (128, 0, 255),      # Purple
    (0, 128, 255),      # Light Blue
    (255, 0, 128),      # Pink
    (0, 200, 100),      # Emerald
    (100, 255, 200),    # Mint
]


def _rects_overlap(rect, placed):
    """Check if rect overlaps any already-placed label rectangle."""
    x1, y1, x2, y2 = rect
    for px1, py1, px2, py2 in placed:
        if x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1:
            return True
    return False


def apply_blur_effects(frame, resp, blur_targets: set[str]):
    """Apply Gaussian blur to segments matching blur_targets.

    For each segment whose label matches a blur target, blur the masked region.
    Returns the modified frame.
    """
    if not blur_targets or not resp or not resp.segments:
        return frame

    h, w = frame.shape[:2]
    # Pre-compute a single heavy blur of the entire frame
    blurred = cv2.GaussianBlur(frame, (51, 51), 30)

    for seg in resp.segments:
        if not seg.rle_mask or seg.mask_width <= 0 or seg.mask_height <= 0:
            continue

        label = (seg.label or "").lower().lstrip("~")
        asset_class = (seg.asset_class or "").lower()

        # Check if this segment matches any blur target
        matched = False
        for target in blur_targets:
            if target == label:
                matched = True
                break
            if target == asset_class:
                matched = True
                break
            # Partial match: "person" matches "left_hand" via asset_class
            if target in label or label in target:
                matched = True
                break
        if not matched:
            continue

        # Decode mask and apply blur
        mask = decode_rle(seg.rle_mask, seg.mask_width, seg.mask_height)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask > 128 if mask.dtype == np.uint8 else mask > 0.5

        # Composite: blurred where mask, original elsewhere
        frame[mask_bool] = blurred[mask_bool]

    return frame


def draw_overlay_fast(frame, resp, debug_mode: bool):
    """Draw segment overlays with contour outlines + labels.

    Labels are nudged upward to avoid overlapping each other.
    debug_mode=False (default): contour outlines + center labels.
    debug_mode=True  (press P): adds bounding boxes + detailed label info.
    """
    h, w = frame.shape[:2]
    placed_rects = []  # Track placed label positions to avoid overlap

    for i, seg in enumerate(resp.segments):
        if not seg.rle_mask or seg.mask_width <= 0 or seg.mask_height <= 0:
            continue

        # Decode RLE mask
        mask = decode_rle(seg.rle_mask, seg.mask_width, seg.mask_height)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)

        # Find contours (masks are already smoothed during RLE encoding)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        label = seg.label or ""
        is_pending = label.startswith("~")
        display_label = label.lstrip("~") if is_pending else label

        # Stable color from track_id (consistent across frames)
        color_idx = hash(seg.track_id) if seg.track_id else i
        border_color = BRIGHT_COLORS[color_idx % len(BRIGHT_COLORS)]

        # Draw contour outlines with black border for contrast
        cv2.drawContours(frame, contours, -1, (0, 0, 0), 5)
        cv2.drawContours(frame, contours, -1, border_color, 2)

        # Draw label at segment center with collision avoidance
        if display_label:
            cx = int(seg.center_x * w)
            cy = int(seg.center_y * h)

            if is_pending:
                font_scale, thickness = 0.35, 1
                text_color = (150, 150, 150)
                bg_color = (50, 50, 50)
                pad = 2
            else:
                font_scale, thickness = 0.5, 1
                text_color = border_color
                bg_color = (0, 0, 0)
                pad = 3

            (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Nudge label upward if it overlaps an existing label
            cy_draw = cy
            for _ in range(5):
                label_rect = (cx - pad, cy_draw - th - pad, cx + tw + pad, cy_draw + pad)
                if not _rects_overlap(label_rect, placed_rects):
                    break
                cy_draw -= (th + pad * 2 + 2)
            else:
                label_rect = (cx - pad, cy_draw - th - pad, cx + tw + pad, cy_draw + pad)

            placed_rects.append(label_rect)
            cv2.rectangle(frame, (label_rect[0], label_rect[1]),
                         (label_rect[2], label_rect[3]), bg_color, -1)
            cv2.putText(frame, display_label, (cx, cy_draw),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        # Debug mode: bounding boxes + detailed info
        if debug_mode:
            x1, y1 = int(seg.bbox.x_min * w), int(seg.bbox.y_min * h)
            x2, y2 = int(seg.bbox.x_max * w), int(seg.bbox.y_max * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)

            detail = f"{seg.label} [{seg.asset_class}] {seg.confidence:.0%}"
            if seg.emotion.primary_emotion:
                detail += f" {seg.emotion.display_label}"
            (tw, th_), _ = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, y1 - th_ - 6), (x1 + tw + 4, y1), (30, 30, 30), -1)
            cv2.putText(frame, detail, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # Conversation
    conv = resp.conversation
    if conv.summary:
        cv2.putText(frame, f"Conv: {conv.summary}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if conv.vibe:
        cv2.putText(frame, f"Vibe: {conv.vibe}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)

    return frame


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


async def run_client(url: str, target_fps: int, show: bool, camera: int,
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
            print(f"Connected! Target {target_fps} fps. Ctrl+C or Q to quit.")
            if show:
                print("  P = toggle debug mode (bounding boxes + detailed labels)")
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

            # --- Blur state ---
            blur_targets: set[str] = set()

            # --- Shared state ---
            send_count = 0
            recv_count = 0
            latest_resp = None
            debug_mode = False  # P toggles: show all segments with bboxes
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

                    # Display
                    if show and latest_resp is not None:
                        # Apply blur effects before drawing overlay
                        display = apply_blur_effects(frame, latest_resp, blur_targets)
                        display = draw_overlay_fast(display, latest_resp, debug_mode)

                        # Compute SAM FPS from update timestamps
                        if len(sam_update_times) >= 2:
                            sam_dt = sam_update_times[-1] - sam_update_times[0]
                            sam_fps = (len(sam_update_times) - 1) / sam_dt if sam_dt > 0 else 0
                        else:
                            sam_fps = 0

                        elapsed = time.perf_counter() - t_start
                        sfps = send_count / elapsed if elapsed > 0 else 0
                        rfps = recv_count / elapsed if elapsed > 0 else 0
                        avg_rtt = np.mean(rtts[-60:]) if rtts else 0
                        n_segs = len(latest_resp.segments)

                        # HUD line 1: FPS (Mask = actual mask content updates)
                        cv2.putText(display,
                                    f"Client {rfps:.0f} fps | Mask {sam_fps:.1f} fps | RTT {avg_rtt:.0f}ms | {n_segs} segs",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)

                        # HUD line 2: mode indicator
                        mode_text = "[P] DEBUG: bboxes + detail" if debug_mode else "[P] contours + labels"
                        cv2.putText(display, mode_text,
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (180, 180, 180), 1, cv2.LINE_AA)

                        # HUD line 3: blur targets
                        if blur_targets:
                            blur_text = "BLUR: " + ", ".join(sorted(blur_targets))
                            cv2.putText(display, blur_text,
                                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        (0, 100, 255), 1, cv2.LINE_AA)

                        cv2.imshow("SocialSenseAR", display)

                        # Auto-save screenshot every 5s for visual iteration
                        _now = time.perf_counter()
                        if not hasattr(draw_overlay_fast, '_last_ss') or _now - draw_overlay_fast._last_ss >= 5.0:
                            draw_overlay_fast._last_ss = _now
                            import os
                            ss_path = os.path.expanduser("~/Downloads/client_debug.png")
                            cv2.imwrite(ss_path, display)
                            print(f"Screenshot saved: {ss_path}")

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        elif key == ord("p"):
                            debug_mode = not debug_mode
                            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

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
        if show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="SocialSenseAR test client")
    parser.add_argument("--url", default="ws://localhost:8765", help="Server URL")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    parser.add_argument("--show", action="store_true", help="Show overlay window")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--whisper", action="store_true",
                        help="Enable voice commands via Whisper (requires openai-whisper + sounddevice)")
    args = parser.parse_args()
    asyncio.run(run_client(args.url, args.fps, args.show, args.camera, args.whisper))


if __name__ == "__main__":
    main()
