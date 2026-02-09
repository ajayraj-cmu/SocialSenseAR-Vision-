"""Automated benchmark harness for SocialSenseAR vision pipeline.

Launches the server, connects as a client, sends webcam (or synthetic) frames
for a configurable duration, collects FPS/RTT/label-quality metrics, and writes
a JSON report.

Designed for AI-agent iteration: tweak code, run benchmark, compare results.

Usage:
    python -m server.benchmark                     # 30s webcam run
    python -m server.benchmark --duration 60       # 60s run
    python -m server.benchmark --no-camera         # synthetic frames
    python -m server.benchmark --output results/   # custom output dir
"""

import asyncio
import argparse
import collections
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import websockets

sys.path.insert(0, ".")
from server.proto import socialsense_pb2 as pb
from server.test_client import FrameGrabber

# Labels that indicate correct person identification
_PERSON_LABELS = frozenset({
    "person", "face", "head", "hair", "torso", "body", "shirt", "arm",
    "hand", "shoulder", "skin", "pants", "left_hand", "right_hand",
    "left_arm", "right_arm", "leg", "left_leg", "right_leg", "human",
})


def score_label(label: str) -> float:
    """Score a single label: 1.0 good, 0.5 pending/generic, 0.0 missing."""
    if not label:
        return 0.0
    if label.startswith("~"):
        return 0.5
    if label in ("object", "area"):
        return 0.5
    words = label.replace("_", " ").split()
    if len(words) > 2:
        return 0.7  # long label, penalize slightly
    return 1.0


def generate_synthetic_frame(width=640, height=480, frame_id=0):
    """Generate a synthetic test frame with colored shapes."""
    frame = np.full((height, width, 3), (200, 200, 200), dtype=np.uint8)
    # Moving rectangle (simulates object)
    x_offset = (frame_id * 3) % (width - 100)
    cv2.rectangle(frame, (x_offset, 100), (x_offset + 100, 300), (0, 100, 200), -1)
    # Static circle (simulates face-sized object)
    cv2.circle(frame, (width // 2, height // 3), 60, (180, 140, 100), -1)
    # Background wall-like region
    cv2.rectangle(frame, (0, height - 80), (width, height), (100, 100, 100), -1)
    return frame


async def run_benchmark(url, duration, use_camera, camera_idx, output_dir):
    """Run the benchmark and collect metrics."""
    grabber = None
    if use_camera:
        grabber = FrameGrabber(camera_idx)
        if not grabber.opened:
            print(f"ERROR: Cannot open camera {camera_idx}")
            return None
        grabber.start()
        # Wait for first frame
        for _ in range(200):
            frame, _ = grabber.get()
            if frame is not None:
                break
            await asyncio.sleep(0.025)
        else:
            print("ERROR: No camera frames")
            grabber.stop()
            return None
        h, w = frame.shape[:2]
    else:
        w, h = 640, 480

    print(f"Connecting to {url}...")
    try:
        async with websockets.connect(
            url, max_size=10 * 1024 * 1024, ping_interval=20, ping_timeout=60,
        ) as ws:
            print(f"Connected. Warming up server (torch.compile)...")

            # --- Warmup phase: send frames until first response ---
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            frame_id_counter = 0
            warmup_start = time.perf_counter()
            warmup_sent = 0
            while True:
                frame_id_counter += 1
                if use_camera and grabber:
                    frame, _ = grabber.get()
                    if frame is None:
                        frame = generate_synthetic_frame(w, h, frame_id_counter)
                else:
                    frame = generate_synthetic_frame(w, h, frame_id_counter)
                frame_send = cv2.flip(frame, 0)
                _, jpeg = cv2.imencode(".jpg", frame_send, encode_params)
                msg = pb.ClientMessage()
                msg.frame_id = frame_id_counter
                msg.timestamp_ms = time.time() * 1000
                msg.frame.jpeg_data = jpeg.tobytes()
                msg.frame.width = w
                msg.frame.height = h
                msg.frame.quality = 70
                await ws.send(msg.SerializeToString())
                warmup_sent += 1
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    # Got first response — server is warm
                    warmup_s = time.perf_counter() - warmup_start
                    print(f"  Server warm ({warmup_s:.1f}s, {warmup_sent} frames). "
                          f"Draining queue...")
                    # Drain any queued responses
                    drained = 1
                    while True:
                        try:
                            await asyncio.wait_for(ws.recv(), timeout=0.1)
                            drained += 1
                        except asyncio.TimeoutError:
                            break
                    print(f"  Drained {drained} queued responses.")
                    break
                except asyncio.TimeoutError:
                    pass  # Server still warming up, send another frame
                if time.perf_counter() - warmup_start > 120:
                    print("ERROR: Server warmup timed out after 120s")
                    return None

            print(f"Running benchmark for {duration}s...")

            # Metrics collection
            send_count = 0
            recv_count = 0
            send_deque: collections.deque[float] = collections.deque()
            rtts: list[float] = []
            last_raw_response = None
            sam_update_times: list[float] = []
            time_series: list[dict] = []
            all_label_scores: list[float] = []
            last_fid = -1
            latest_resp = None

            t_start = time.perf_counter()
            last_report = t_start

            # Receiver task
            async def receiver():
                nonlocal recv_count, latest_resp, last_raw_response
                try:
                    async for raw in ws:
                        t_recv = time.perf_counter()
                        resp = pb.ServerMessage()
                        resp.ParseFromString(raw)
                        latest_resp = resp
                        recv_count += 1

                        # Detect actual SAM updates: server sends identical cached
                        # bytes between SAM cycles, new bytes when SAM produces data.
                        if raw != last_raw_response:
                            last_raw_response = raw
                            sam_update_times.append(t_recv)

                        # RTT
                        if send_deque:
                            sent_at = send_deque.popleft()
                            rtts.append((t_recv - sent_at) * 1000)

                        # Score labels
                        for seg in resp.segments:
                            all_label_scores.append(score_label(seg.label))

                except websockets.exceptions.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receiver())

            # Pre-encode synthetic frames to eliminate per-frame JPEG encode overhead
            pre_encoded_jpegs = []
            if not use_camera:
                for i in range(120):  # 2 seconds of unique frames at 60fps
                    frame = generate_synthetic_frame(w, h, i)
                    frame_send = cv2.flip(frame, 0)
                    _, jpeg = cv2.imencode(".jpg", frame_send, encode_params)
                    pre_encoded_jpegs.append(jpeg.tobytes())
                print(f"  Pre-encoded {len(pre_encoded_jpegs)} synthetic frames")

            try:
                while True:
                    elapsed = time.perf_counter() - t_start
                    if elapsed >= duration:
                        break

                    # Get frame
                    if grabber:
                        frame, fid = grabber.get()
                        if frame is None or fid == last_fid:
                            await asyncio.sleep(0.001)
                            continue
                        last_fid = fid
                        frame_send = cv2.flip(frame, 0)
                        _, jpeg = cv2.imencode(".jpg", frame_send, encode_params)
                        jpeg_bytes = jpeg.tobytes()
                    else:
                        frame_id_counter += 1
                        fid = frame_id_counter
                        jpeg_bytes = pre_encoded_jpegs[fid % len(pre_encoded_jpegs)]

                    msg = pb.ClientMessage()
                    msg.frame_id = fid
                    msg.timestamp_ms = time.time() * 1000
                    msg.frame.jpeg_data = jpeg_bytes
                    msg.frame.width = w
                    msg.frame.height = h
                    msg.frame.quality = 70

                    send_deque.append(time.perf_counter())
                    await ws.send(msg.SerializeToString())
                    send_count += 1

                    # Periodic time-series snapshot (every 1s)
                    now_pc = time.perf_counter()
                    if now_pc - last_report >= 1.0:
                        last_report = now_pc
                        sec_elapsed = now_pc - t_start

                        # SAM FPS — rolling window (last 120 distinct updates)
                        if len(sam_update_times) >= 2:
                            window = sam_update_times[-120:]
                            sam_dt = window[-1] - window[0]
                            sam_fps = (len(window) - 1) / sam_dt if sam_dt > 0 else 0
                        else:
                            sam_fps = 0

                        client_fps = recv_count / sec_elapsed if sec_elapsed > 0 else 0
                        avg_rtt = float(np.mean(rtts[-60:])) if rtts else 0
                        p95_rtt = float(np.percentile(rtts[-60:], 95)) if len(rtts) >= 2 else 0
                        n_segs = len(latest_resp.segments) if latest_resp else 0
                        lq = float(np.mean(all_label_scores[-100:])) if all_label_scores else 0

                        snapshot = {
                            "t": round(sec_elapsed, 1),
                            "client_fps": round(client_fps, 1),
                            "sam_fps": round(sam_fps, 1),
                            "rtt_avg_ms": round(avg_rtt, 1),
                            "rtt_p95_ms": round(p95_rtt, 1),
                            "segments": n_segs,
                            "label_quality": round(lq, 3),
                        }
                        time_series.append(snapshot)
                        print(
                            f"  [{sec_elapsed:5.1f}s] "
                            f"client={client_fps:.0f}fps "
                            f"sam={sam_fps:.1f}fps "
                            f"rtt={avg_rtt:.0f}ms "
                            f"segs={n_segs} "
                            f"lq={lq:.2f}"
                        )

                    # Precise 60fps send rate (busy-wait avoids Windows timer granularity)
                    target_time = time.perf_counter() + 1.0 / 60
                    while time.perf_counter() < target_time:
                        await asyncio.sleep(0)  # yield to receiver task

            except KeyboardInterrupt:
                print("\nStopped early")
            finally:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

            # Build final report
            total_elapsed = time.perf_counter() - t_start

            if len(sam_update_times) >= 2:
                # Use last 120 samples for steady-state SAM fps
                window = sam_update_times[-120:]
                sam_dt = window[-1] - window[0]
                final_sam_fps = (len(window) - 1) / sam_dt if sam_dt > 0 else 0
            else:
                final_sam_fps = 0

            report = {
                "duration_s": round(total_elapsed, 1),
                "frames_sent": send_count,
                "frames_received": recv_count,
                "camera": "webcam" if use_camera else "synthetic",
                "summary": {
                    "client_fps": round(recv_count / total_elapsed, 1) if total_elapsed > 0 else 0,
                    "sam_fps": round(final_sam_fps, 1),
                    "rtt_avg_ms": round(float(np.mean(rtts)), 1) if rtts else 0,
                    "rtt_p95_ms": round(float(np.percentile(rtts, 95)), 1) if len(rtts) >= 2 else 0,
                    "label_quality": round(float(np.mean(all_label_scores)), 3) if all_label_scores else 0,
                },
                "time_series": time_series,
            }

            # Add git hash
            try:
                git_hash = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL, text=True
                ).strip()
                report["git_commit"] = git_hash
            except Exception:
                pass

            return report

    except (ConnectionRefusedError, OSError) as e:
        print(f"ERROR: Cannot connect to {url} ({e})")
        print("Make sure the server is running: python -m server.main")
        return None
    finally:
        if grabber:
            grabber.stop()


def main():
    parser = argparse.ArgumentParser(description="SocialSenseAR Benchmark")
    parser.add_argument("--url", default="ws://localhost:8765", help="Server URL")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration in seconds")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-camera", action="store_true", help="Use synthetic frames")
    parser.add_argument("--output", default="cache", help="Output directory for results (default: cache/)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "benchmark_results.json")

    report = asyncio.run(run_benchmark(
        args.url, args.duration, not args.no_camera, args.camera, args.output
    ))

    if report is None:
        print("Benchmark failed.")
        sys.exit(1)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    print()
    print("=" * 60)
    print(f"BENCHMARK RESULTS ({report['duration_s']}s, {report['camera']})")
    print("=" * 60)
    print(f"  Client FPS:    {s['client_fps']}")
    print(f"  SAM FPS:       {s['sam_fps']}")
    print(f"  RTT avg:       {s['rtt_avg_ms']}ms")
    print(f"  RTT p95:       {s['rtt_p95_ms']}ms")
    print(f"  Label quality: {s['label_quality']}")
    print(f"  Report saved:  {output_path}")
    if "git_commit" in report:
        print(f"  Git commit:    {report['git_commit']}")
    print()


if __name__ == "__main__":
    main()
