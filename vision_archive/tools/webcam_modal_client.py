"""Webcam client for Modal GPU offload testing.

Captures webcam frames, sends to Modal WebSocket, displays overlays locally.

This is essentially the same as server.test_client but:
1. Can connect to remote wss:// URLs (Modal deployment)
2. Has local RLE decode + OpenCV overlay rendering
3. Adds --show flag to display overlay window (optional)

Usage:
    # Connect to Modal deployment:
    python tools/webcam_modal_client.py --url wss://your-modal-url.modal.run/ws --show

    # Connect to local server:
    python tools/webcam_modal_client.py --url ws://localhost:8765 --show
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


class FrameGrabber:
    """Reads webcam frames in a background thread."""

    def __init__(self, camera_idx: int, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(camera_idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, 60)
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


def decode_and_overlay(frame_bgr: np.ndarray, segments: list) -> np.ndarray:
    """Decode RLE masks and overlay on frame.

    Args:
        frame_bgr: Original BGR frame
        segments: List of SceneSegment protobuf messages

    Returns:
        Frame with colored mask overlays
    """
    overlay = frame_bgr.copy()
    h, w = frame_bgr.shape[:2]

    # Color palette (bright colors for visibility)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 255, 0),    # Spring Green
        (255, 128, 0),    # Orange
    ]

    for i, seg in enumerate(segments):
        if not seg.rle_mask:
            continue

        # Decode RLE mask
        try:
            mask = decode_rle(seg.rle_mask, seg.mask_width, seg.mask_height)

            # Resize mask to frame size if needed
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Apply colored overlay
            color = colors[i % len(colors)]
            overlay[mask > 0] = (
                overlay[mask > 0] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)

            # Draw label at centroid
            cx = int(seg.center_x * w)
            cy = int(seg.center_y * h)
            label_text = f"{seg.label} ({seg.confidence:.2f})"

            # Background rectangle for label
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                overlay,
                (cx - 2, cy - th - 4),
                (cx + tw + 2, cy + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                overlay,
                label_text,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Warning: Failed to decode mask for segment {i}: {e}")
            continue

    return overlay


def _mask_fingerprint(resp) -> str:
    """Content fingerprint of segment positions + masks."""
    if not resp or not resp.segments:
        return ""
    parts = []
    for seg in resp.segments:
        parts.append(f"{seg.center_x:.4f},{seg.center_y:.4f},{seg.label}")
        if seg.rle_mask:
            parts.append(seg.rle_mask[:32].hex())
    return "|".join(parts)


async def run_client(
    url: str,
    target_fps: int,
    camera: int,
    show_overlay: bool = False,
):
    """Main client loop."""
    # Command input queue (from terminal thread)
    cmd_queue: queue.Queue = queue.Queue()

    def _terminal_input():
        """Read commands from terminal in background."""
        while True:
            try:
                line = input("cmd> ")
                if line.strip():
                    cmd_queue.put(line.strip())
            except EOFError:
                break
            except Exception:
                break

    input_thread = threading.Thread(target=_terminal_input, daemon=True)
    input_thread.start()

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
            print(f"✓ Connected to Modal GPU server!")
            print(f"Target {target_fps} fps. Ctrl+C to quit.")
            if show_overlay:
                print("Overlay window enabled (--show)")
            else:
                print("Overlay disabled (use --show to enable)")

            # Shared state
            send_count = 0
            recv_count = 0
            latest_resp = None
            send_deque: collections.deque[float] = collections.deque()
            rtts: list[float] = []
            t_start = time.perf_counter()
            interval = 1.0 / target_fps
            last_sent_fid = -1
            last_jpeg = None
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]

            # SAM FPS tracking
            last_raw_response = None
            sam_update_times: list[float] = []

            # --- Receiver coroutine ---
            async def receiver():
                nonlocal recv_count, latest_resp, last_raw_response
                try:
                    async for raw in ws:
                        t_recv = time.perf_counter()
                        resp = pb.ServerMessage()
                        resp.ParseFromString(raw)
                        latest_resp = resp
                        recv_count += 1

                        # Detect real mask updates
                        mask_fp = _mask_fingerprint(resp)
                        if mask_fp != last_raw_response:
                            last_raw_response = mask_fp
                            sam_update_times.append(t_recv)
                            if len(sam_update_times) > 120:
                                sam_update_times.pop(0)

                        # Round-trip time
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
                        # New frame — encode it
                        last_sent_fid = fid
                        frame_send = cv2.flip(frame, 0)  # Match server expectation
                        _, jpeg = cv2.imencode(".jpg", frame_send, encode_params)
                        last_jpeg = jpeg
                    elif last_jpeg is None:
                        await asyncio.sleep(0.001)
                        continue

                    jpeg = last_jpeg

                    # Build protobuf
                    msg = pb.ClientMessage()
                    msg.frame_id = send_count + 1
                    msg.timestamp_ms = time.time() * 1000
                    msg.frame.jpeg_data = jpeg.tobytes()
                    msg.frame.width = w
                    msg.frame.height = h
                    msg.frame.quality = 70
                    data = msg.SerializeToString()

                    # Send
                    send_deque.append(time.perf_counter())
                    await ws.send(data)
                    send_count += 1

                    # Display overlay if enabled
                    if show_overlay:
                        if latest_resp and latest_resp.segments:
                            overlay = decode_and_overlay(frame, latest_resp.segments)
                        else:
                            overlay = frame.copy()

                        # HUD: stats + help
                        n_segs = len(latest_resp.segments) if latest_resp else 0
                        hud_lines = [
                            f"Segs: {n_segs} | RTT: {np.mean(rtts[-30:]):.0f}ms" if rtts else "Connecting...",
                            "Keys: P=person  L=laptop  M=monitor  F=face",
                            "      W=wall  C=chair  T=table  D=door",
                            "      H=hand  B=bottle  X=clear  ESC=quit",
                            f"Cmd: type label + Enter in terminal",
                        ]
                        for idx_h, line in enumerate(hud_lines):
                            y_pos = 20 + idx_h * 18
                            cv2.putText(overlay, line, (8, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(overlay, line, (8, y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        (0, 255, 0), 1, cv2.LINE_AA)

                        cv2.imshow("SocialSenseAR - Modal GPU", overlay)
                        key = cv2.waitKey(1) & 0xFF

                        if key == 27:  # ESC to quit
                            break

                        # Keyboard shortcuts → send control commands
                        key_to_label = {
                            ord('p'): "person", ord('l'): "laptop",
                            ord('m'): "monitor", ord('f'): "face",
                            ord('w'): "wall", ord('c'): "chair",
                            ord('t'): "table", ord('d'): "door",
                            ord('h'): "hand", ord('b'): "bottle",
                            ord('g'): "glasses", ord('k'): "keyboard",
                            ord('s'): "shirt", ord('n'): "phone",
                        }
                        if key == ord('x'):
                            # Clear all
                            ctrl = pb.ClientMessage()
                            ctrl.frame_id = send_count
                            ctrl.timestamp_ms = time.time() * 1000
                            ctrl.control.command = "clear"
                            await ws.send(ctrl.SerializeToString())
                            print("  >> Sent: clear all prompts")
                        elif key in key_to_label:
                            label = key_to_label[key]
                            ctrl = pb.ClientMessage()
                            ctrl.frame_id = send_count
                            ctrl.timestamp_ms = time.time() * 1000
                            ctrl.control.command = f"blur {label}"
                            await ws.send(ctrl.SerializeToString())
                            print(f"  >> Sent: blur {label}")

                    # Also check terminal input queue
                    while not cmd_queue.empty():
                        try:
                            raw_cmd = cmd_queue.get_nowait()
                            ctrl = pb.ClientMessage()
                            ctrl.frame_id = send_count
                            ctrl.timestamp_ms = time.time() * 1000
                            ctrl.control.command = raw_cmd
                            await ws.send(ctrl.SerializeToString())
                            print(f"  >> Sent command: {raw_cmd}")
                        except Exception:
                            pass

                    # Print stats
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

                    # Throttle to target FPS
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
                if show_overlay:
                    cv2.destroyAllWindows()

    except (ConnectionRefusedError, OSError) as e:
        print(f"ERROR: Cannot connect to {url} ({e})")
        print("Make sure Modal app is deployed: modal deploy modal_app.py")
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        grabber.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Webcam client for Modal GPU server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to Modal deployment with overlay:
  python tools/webcam_modal_client.py --url wss://your-app.modal.run/ws --show

  # Connect to local server:
  python tools/webcam_modal_client.py --url ws://localhost:8765 --show

  # High FPS test (no overlay):
  python tools/webcam_modal_client.py --url wss://your-app.modal.run/ws --fps 60
        """,
    )
    parser.add_argument(
        "--url",
        required=True,
        help="WebSocket URL (ws://localhost:8765 or wss://your-app.modal.run/ws)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show overlay window with masks (adds latency)",
    )
    args = parser.parse_args()

    asyncio.run(run_client(args.url, args.fps, args.camera, args.show))


if __name__ == "__main__":
    main()
