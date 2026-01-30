"""Test client — pipelined webcam sender + async response receiver.

Architecture:
  FrameGrabber thread  -> always has the latest webcam frame ready
  sender loop          -> encodes JPEG + sends protobuf (fire-and-forget)
  receiver() task      -> collects server responses, tracks round-trip latency
  display (optional)   -> cv2 overlay with latest response

This mirrors how the Quest headset works: send frames as fast as the
camera produces them, process responses asynchronously.

Usage:
    python -m server.main --device cuda
    python -m server.test_client
    python -m server.test_client --url ws://localhost:8765 --fps 30 --show
"""

import asyncio
import argparse
import collections
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


class FrameGrabber:
    """Reads webcam frames in a background thread.

    Always keeps the latest frame available. Camera read never blocks
    the async event loop.
    """

    def __init__(self, camera_idx: int, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(camera_idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
    (255, 255, 255),    # White
    (100, 255, 200),    # Mint
]


def draw_overlay_fast(frame, resp, debug_mode: bool):
    """Draw segment overlays matching sam_gemini_voice.py visual style.

    Always shows all segments with contour outlines + labels at centers.
    debug_mode=False (default): contour outlines + center labels (like original).
    debug_mode=True  (press P): adds bounding boxes + detailed label info.
    """
    h, w = frame.shape[:2]

    for i, seg in enumerate(resp.segments):
        if not seg.rle_mask or seg.mask_width <= 0 or seg.mask_height <= 0:
            continue

        # Decode RLE mask
        mask = decode_rle(seg.rle_mask, seg.mask_width, seg.mask_height)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        label = seg.label or ""
        is_pending = label.startswith("~")
        display_label = label.lstrip("~") if is_pending else label

        # Border color + width (matches original logic)
        border_color = BRIGHT_COLORS[i % len(BRIGHT_COLORS)]
        border_width = 2

        # Draw contour outlines
        cv2.drawContours(frame, contours, -1, border_color, border_width)

        # Draw label at segment center
        if display_label:
            cx = int(seg.center_x * w)
            cy = int(seg.center_y * h)

            if is_pending:
                # Pending labels: smaller, dimmer (matches original)
                (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(frame, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2), (50, 50, 50), -1)
                cv2.putText(frame, display_label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            else:
                # Confirmed labels: bright text on black background
                (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (cx - 3, cy - th - 4), (cx + tw + 3, cy + 4), (0, 0, 0), -1)
                cv2.putText(frame, display_label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)

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


async def run_client(url: str, target_fps: int, show: bool, camera: int):
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
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            # SAM FPS tracking (detect actual segment changes)
            last_seg_signature = None  # fingerprint of current segments
            sam_update_times: list[float] = []

            # --- Receiver coroutine (runs concurrently) ---
            async def receiver():
                nonlocal recv_count, latest_resp, last_seg_signature
                try:
                    async for raw in ws:
                        t_recv = time.perf_counter()
                        resp = pb.ServerMessage()
                        resp.ParseFromString(raw)
                        latest_resp = resp
                        recv_count += 1

                        # Detect actual SAM updates by checking if segments changed.
                        # Server caches serialized bytes so metrics are stale —
                        # instead fingerprint segment count + first bbox.
                        n = len(resp.segments)
                        if n > 0:
                            s0 = resp.segments[0]
                            sig = (n, round(s0.bbox.x_min, 3), round(s0.bbox.y_min, 3),
                                   round(s0.center_x, 3), round(s0.center_y, 3))
                        else:
                            sig = (0,)
                        if sig != last_seg_signature:
                            last_seg_signature = sig
                            sam_update_times.append(t_recv)
                            if len(sam_update_times) > 60:
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
                    if frame is None or fid == last_sent_fid:
                        await asyncio.sleep(0.001)
                        continue
                    last_sent_fid = fid

                    # Encode JPEG
                    _, jpeg = cv2.imencode(".jpg", frame, encode_params)

                    # Build protobuf
                    msg = pb.ClientMessage()
                    msg.frame_id = fid
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

                    # Display
                    if show and latest_resp is not None:
                        display = draw_overlay_fast(frame, latest_resp, debug_mode)

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

                        # HUD line 1: FPS
                        cv2.putText(display,
                                    f"Client {rfps:.0f} fps | SAM {sam_fps:.0f} fps | RTT {avg_rtt:.0f}ms | {n_segs} segs",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)

                        # HUD line 2: mode indicator
                        mode_text = "[P] DEBUG: bboxes + detail" if debug_mode else "[P] contours + labels"
                        cv2.putText(display, mode_text,
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (180, 180, 180), 1, cv2.LINE_AA)

                        cv2.imshow("SocialSenseAR", display)
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
                            f"SAM: {sam_fps:4.1f} fps | "
                            f"RTT avg: {avg_rtt:5.1f}ms p95: {p95_rtt:5.1f}ms | "
                            f"Segs: {n_segs:2d} | "
                            f"Frames: {send_count}"
                        )

                    # Throttle to target FPS
                    loop_time = time.perf_counter() - t_loop
                    remaining = interval - loop_time
                    if remaining > 0:
                        await asyncio.sleep(remaining)

            except KeyboardInterrupt:
                print("\nStopped by user")
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

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
    args = parser.parse_args()
    asyncio.run(run_client(args.url, args.fps, args.show, args.camera))


if __name__ == "__main__":
    main()
