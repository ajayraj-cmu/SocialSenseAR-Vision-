"""Profile the full segment_frame() pipeline to find bottlenecks.

Usage: python -m server.pipeline_profiler
"""
import time
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import ServerConfig
from server.vision.fastsam_segmenter import FastSAMSegmenter


def profile_pipeline(iterations=100):
    # Capture a real frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("No webcam, using synthetic frame")
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    print(f"Frame: {frame.shape}")

    config = ServerConfig()
    seg = FastSAMSegmenter(config)
    seg.initialize()

    # Let MediaPipe thread warm up
    warmup = 30
    print(f"Warming up ({warmup} iterations + MP thread settle)...")
    for _ in range(warmup):
        seg.segment_frame(frame)
    time.sleep(1.0)  # Let MP thread fully settle

    # Profile
    print(f"\nProfiling {iterations} iterations of segment_frame()...\n")

    times = []
    seg_counts = []
    for i in range(iterations):
        # Feed a copy so MediaPipe thread sees a new id() each time
        frame_copy = frame.copy()
        t0 = time.perf_counter()
        segments = seg.segment_frame(frame_copy)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        seg_counts.append(len(segments))

        if i < 5 or i == iterations - 1:
            labels = [s.label for s in segments]
            print(f"  iter {i}: {elapsed:.1f}ms | {len(segments)} segs | {labels}")

    times = np.array(times)
    fps = 1000.0 / np.mean(times)

    print(f"\n{'='*60}")
    print(f"PIPELINE PROFILE ({iterations} iterations)")
    print(f"Model: {seg._model_path}")
    print(f"{'='*60}")
    print(f"Mean:   {np.mean(times):.1f}ms  ({fps:.1f} FPS)")
    print(f"Median: {np.median(times):.1f}ms  ({1000/np.median(times):.1f} FPS)")
    print(f"Min:    {np.min(times):.1f}ms  ({1000/np.min(times):.1f} FPS)")
    print(f"Max:    {np.max(times):.1f}ms")
    print(f"P95:    {np.percentile(times, 95):.1f}ms")
    print(f"Segments: mean={np.mean(seg_counts):.1f}")
    print(f"{'='*60}")

    target = 25.0  # 40 FPS
    if np.mean(times) <= target:
        print(f"\n*** TARGET MET: {fps:.1f} FPS >= 40 FPS ***")
    else:
        print(f"\nGap to 40 FPS: need to save {np.mean(times) - target:.1f}ms per frame")

    seg.shutdown()


if __name__ == "__main__":
    profile_pipeline()
