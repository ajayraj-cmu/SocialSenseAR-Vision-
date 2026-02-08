"""Quick standalone FPS benchmark for SAM3 segmenter.
Directly calls segment_frame() on synthetic frames, measures throughput.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import time
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stderr)
# Only show our segmenter logs at INFO level
logging.getLogger("server.vision.sam3_segmenter").setLevel(logging.INFO)

from server.config import ServerConfig
from server.vision.sam3_segmenter import SAM3Segmenter

def main():
    config = ServerConfig()
    config.device = "cuda"

    print("Initializing SAM3Segmenter...", flush=True)
    seg = SAM3Segmenter(config)
    seg.initialize()
    print("Initialized. Running FPS benchmark...", flush=True)

    # Synthetic 640x480 BGR frames (random noise simulates camera)
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
    # Also a "static" frame to test vision cache hits
    static_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup (first 3 frames are slow due to torch.compile)
    print("Warmup (3 frames)...", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        result = seg.segment_frame(frames[i % len(frames)])
        dt = time.perf_counter() - t0
        print(f"  warmup {i+1}: {dt*1000:.0f}ms, {len(result)} segs", flush=True)

    # Benchmark 1: FORCE vision re-encode every frame (no cache)
    print("\n--- Benchmark: UNCACHED (vision + decoder every frame) ---", flush=True)
    # Invalidate cache each iteration to force vision encoder re-run
    times_uncached = []
    for i in range(20):
        seg._cached_vision_embeds = None  # force re-encode
        seg._cached_frame_thumb = None
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        result = seg.segment_frame(frame)
        dt = time.perf_counter() - t0
        times_uncached.append(dt)
        if i < 3:
            print(f"  frame {i+1}: {dt*1000:.0f}ms", flush=True)

    avg_uc = np.mean(times_uncached) * 1000
    p50_uc = np.median(times_uncached) * 1000
    fps_uc = 1000.0 / avg_uc if avg_uc > 0 else 0
    print(f"  avg={avg_uc:.0f}ms  p50={p50_uc:.0f}ms  FPS={fps_uc:.1f}", flush=True)

    # Benchmark 2: CACHED (static frame, vision cache should hit)
    print("\n--- Benchmark: CACHED (decoder only, static frame) ---", flush=True)
    seg._cached_vision_embeds = None  # reset
    seg._cached_frame_thumb = None
    times_cached = []
    for i in range(30):
        noisy = static_frame.copy()
        noisy = np.clip(noisy.astype(np.int16) + np.random.randint(-3, 4, noisy.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        t0 = time.perf_counter()
        result = seg.segment_frame(noisy)
        dt = time.perf_counter() - t0
        times_cached.append(dt)

    avg_c = np.mean(times_cached[1:]) * 1000  # skip first (cache miss)
    p50_c = np.median(times_cached[1:]) * 1000
    fps_c = 1000.0 / avg_c if avg_c > 0 else 0
    print(f"  avg={avg_c:.0f}ms  p50={p50_c:.0f}ms  FPS={fps_c:.1f}", flush=True)

    # Summary
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"UNCACHED (vis+dec): {fps_uc:.1f} FPS ({avg_uc:.0f}ms avg)", flush=True)
    print(f"CACHED (dec only):  {fps_c:.1f} FPS ({avg_c:.0f}ms avg)", flush=True)
    print(f"Vision cache: hits={seg._vision_cache_hits} misses={seg._vision_cache_misses}", flush=True)

    seg.shutdown()

if __name__ == "__main__":
    main()
