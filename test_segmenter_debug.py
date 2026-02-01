"""Quick standalone test: load SAM3Segmenter, run on webcam frame, print debug info."""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

# Capture one webcam frame
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: Cannot read from camera")
    sys.exit(1)
h, w = frame.shape[:2]
print(f"Frame: {w}x{h}", flush=True)

# Init segmenter
from server.config import ServerConfig
config = ServerConfig()
config.device = "cuda"

from server.vision.sam3_segmenter import SAM3Segmenter
seg = SAM3Segmenter(config)

print("Initializing...", flush=True)
t0 = time.perf_counter()
seg.initialize()
print(f"Init done in {time.perf_counter()-t0:.1f}s", flush=True)

print(f"\n_use_trt_decoder = {seg._use_trt_decoder}", flush=True)
print(f"_trt_topk = {seg._trt_topk}", flush=True)
print(f"_use_trt_vision = {seg._use_trt_vision}", flush=True)
print(f"_use_trt = {seg._use_trt}", flush=True)

# Run 5 frames
print("\nRunning 5 frames...", flush=True)
for i in range(5):
    t1 = time.perf_counter()
    segments = seg.segment_frame(frame)
    ms = (time.perf_counter() - t1) * 1000
    labels = [s.label for s in segments]
    print(f"  Frame {i+1}: {ms:.0f}ms, {len(segments)} segs, labels={labels}", flush=True)

# Check mask cache
print(f"\nMask cache: {list(seg._mask_cache.keys())}", flush=True)
for k, (mask, ts) in seg._mask_cache.items():
    area = cv2.countNonZero(mask)
    print(f"  {k}: area={area}", flush=True)

seg.shutdown()
print("Done!", flush=True)
