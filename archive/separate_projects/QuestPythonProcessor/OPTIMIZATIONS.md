# YOLO Optimization Guide

Reference for applying similar optimizations to future models (SAM, etc.)

---

## 1. Model Format (Priority Order)

| Format | File | Speedup | Notes |
|--------|------|---------|-------|
| TensorRT | `.engine` | 2-3x | FP16 half precision, CUDA only |
| ONNX | `.onnx` | 1.5-2x | Cross-platform fallback |
| PyTorch | `.pt` | baseline | Most compatible |

**Auto-selection logic:**
```python
if cuda_available and tensorrt_exists:
    load_tensorrt()  # fastest
elif cuda_available and onnx_exists:
    load_onnx()      # fast
else:
    load_pytorch()   # fallback
```

**Export TensorRT:**
```bash
python export_tensorrt.py
```

---

## 2. GPU Warmup

Run dummy inferences at startup to initialize CUDA context:

```python
# Eliminates first-frame latency
for _ in range(3):
    model(dummy_frame, verbose=False)
```

---

## 3. Input Resolution Scaling

Downscale before inference, upscale results after:

```python
scale = processor_width / original_width
small = cv2.resize(frame, (new_w, new_h))
result = model(small)
mask = cv2.resize(result.mask, original_size)
```

---

## 4. Batch Processing

Process multiple frames in single inference call:

```python
# Better GPU utilization than sequential
batch = [left_frame, right_frame]
results = model(batch)
```

---

## 5. Frame Skipping

Run inference every Nth frame, reuse previous results:

```python
if frame_count % process_skip == 0:
    result = model(frame)
    cached_result = result
else:
    result = cached_result  # reuse
```

---

## 6. GPU Kernels (CuPy)

Custom CUDA kernels for post-processing:

```python
# 256 threads per block (optimal for modern GPUs)
threads = 256
blocks = (total_pixels + 255) // 256

# Preallocate buffers (avoid per-frame malloc)
class GPUBuffers:
    def __init__(self, shape):
        self.buffer = cp.empty(shape, dtype=cp.uint8)
```

---

## 7. Resolution Strategy

Process and convert at LOW resolution, upscale only for display:

```
Input (1920x1080)
    ↓ downscale
Process (480x270)
    ↓ compute mask
    ↓ convert to uint8 (at low res!)
    ↓ upscale
Output (1920x1080)
```

---

## 8. Math Optimizations

```python
# Precompute constants once
inv_soft_edge = 1.0 / soft_edge  # division → multiplication

# Stay in uint8 (avoid float conversions)
cv2.addWeighted(a, 0.7, b, 0.3, 0, dst=output, dtype=cv2.CV_8U)

# In-place operations
np.maximum(mask, threshold, out=mask)
```

---

## 9. Temporal Smoothing

Blend current with previous to reduce jitter:

```python
# Mask smoothing (30% new, 70% old)
smooth_factor = 0.3
mask = cv2.addWeighted(new_mask, smooth_factor,
                       prev_mask, 1 - smooth_factor, 0)

# Movement threshold for bounding boxes
if movement > 0.05:
    bbox = new_bbox
# else: keep previous
```

---

## 10. Async Frame Capture

Background thread fetches frames:

```python
class FrameBuffer:
    def __init__(self):
        self.latest_frame = None
        self.thread = Thread(target=self._capture_loop)

    def get_frame(self):
        return self.latest_frame  # non-blocking
```

---

## Quality Presets

| Preset | Width | Skip | Effect Scale | Target FPS |
|--------|-------|------|--------------|------------|
| QUALITY | 640 | 1 | 1.0 | ~9 |
| STREAM | 320 | 1 | 1.0 | ~55 |
| BALANCED | 320 | 2 | 0.5 | ~14 |
| FAST | 256 | 3 | 0.25 | ~25 |
| EXTREME | 160 | 4 | 0.25 | ~40+ |

```bash
python main.py --preset FAST
```

---

## Key Files

| File | Purpose |
|------|---------|
| `processors/yolo_segmentation.py` | Model loading, inference |
| `export_tensorrt.py` | TensorRT export script |
| `effects/gpu_kernels.py` | CuPy CUDA kernels |
| `effects/focus_grayscale.py` | Effect processing |
| `config.py` | Presets and settings |
| `core/pipeline.py` | Frame skip logic |
| `sources/quest_scrcpy.py` | Async capture |

---

## Checklist for New Models

- [ ] Export to TensorRT (FP16)
- [ ] Export to ONNX (fallback)
- [ ] Add GPU warmup (3 dummy inferences)
- [ ] Implement input downscaling
- [ ] Add batch processing support
- [ ] Implement frame skipping
- [ ] Write GPU kernels for post-processing
- [ ] Preallocate GPU buffers
- [ ] Add temporal smoothing
- [ ] Create quality presets
