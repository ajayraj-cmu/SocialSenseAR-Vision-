# SAM3 FPS Optimization

## Goal
10+ FPS real mask updates on RTX 3060 Laptop (6 GB VRAM, CUDA). MUST use SAM-family models. No quality sacrifice.

## Current State (2026-02-01)
- **Model**: facebook/sam3 (~530M params), text-prompted
- **Pipeline**: TRT vision encoder (FP16, 200ms) → TRT decoder (B=1, 28ms/prompt × 2) → sync
- **Effective rate**: ~3.8 fps (~258ms/frame), stable over 280+ frames
- **VRAM**: 3.4 GB stable

## Pre-built TRT Assets (project root)
| File | Size |
|------|------|
| `sam3_vision.engine` | 872 MB (FP16) |
| `sam3_topk_decoder.engine` | 50 MB |
| `sam3_vision.onnx` | 1770 MB |
| `sam3_topk_decoder.onnx` | 92 MB |
| `sam3_meta.json` | Tensor shapes + preprocessing params |

## Bottleneck
Vision encoder = 200ms (78% of frame time). ViT-32L, 1008x1008 input, compute-bound even with TRT FP16.

## Possible Next Steps
1. **Reduce vision input resolution** (504x504 = ~4x speedup) — must verify mask quality
2. **INT8 quantization** — needs calibration dataset, could halve latency
3. **Vision pruning/distillation** — reduce ViT layers while keeping SAM3 interface

## Quality Constraints
- NO MobileSAM, NanoSAM, or other lower-quality models
- NO resolution reduction that visibly hurts segmentation
- Allowed: TRT, ONNX RT, torch.compile, FP16, pipeline overhead elimination

---

## VRAM Budget (RTX 3060 Laptop — 6 GB)
| Component | Size |
|-----------|------|
| PyTorch SAM3 shell (no vision/decoder submodules) | ~200 MB |
| TRT vision engine + workspace | ~1072 MB |
| TRT decoder engine + workspace | ~250 MB |
| Pre-allocated TRT tensors | ~200 MB |
| Text embeddings (FP32) | ~10 MB |
| CUDA runtime | ~500 MB |
| **Total** | **~3.4 GB** (2.6 GB headroom) |

## VRAM Init Order (CRITICAL — wrong order = OOM)
1. Load PyTorch model (text encoder + processor)
2. Pre-compute text embeddings for all 13 prompts
3. **Delete** PyTorch vision_encoder (~400 MB freed)
4. Load TRT vision engine (872 MB — fits now)
5. Load TRT decoder engine (50 MB)
6. **Delete** text_encoder, detr_encoder, detr_decoder, mask_decoder
7. `torch.cuda.empty_cache()`

**Don't**: load TRT before deleting PyTorch vision (OOM), torch.compile with TRT decoder (wastes 1 GB), PyTorch decoder warmup with TRT available.

---

## Lessons Learned (Bugs & Failed Approaches)

### TRT Decoder NaN Bug (FIXED)
**Symptom**: NaN scores after ~5-7 vision cycles. FPN tensors valid — corruption inside TRT state.
**Root cause**: Dynamic batch (B=1 vs B=2) + cross-CUDA-stream races.
**Fix (all three required)**:
1. B=1 per prompt (sequential, not batched)
2. Default CUDA stream (`ctx.execute_async_v3(0)`)
3. Full `torch.cuda.synchronize()` before each TRT call

### TRT Context Rebuild (REMOVED — don't re-add)
`create_execution_context()` every N calls → 600-2700ms spikes + permanent 15x slowdown. Fresh contexts never complete warmup on RTX 3060. Removed entirely.

### TRT Vision Output Cloning (REMOVED — don't re-add)
`.clone()` on 9 tensors (~200+ MB/frame) → OOM at frame 64. Reference pre-allocated buffers directly (safe: sync pipeline).

### torch.compile + CUDA Graphs
`mode="reduce-overhead"` and `mode="max-autotune"` use CUDA graphs that corrupt SAM3's FPN neck (stale buffer replays). Only `mode="default"` is safe. Compile individual submodules, not top-level `model.forward()`.

### Async Vision Encoding (REMOVED — don't re-add)
Background thread held GPU → decoder perpetually skipped → masks updated every ~3s instead of 258ms. Synchronous pipeline is faster for TRT-speed vision.

### Optical Flow Tracking (REMOVED — don't re-add)
Inflated FPS counter (18fps shown, 1.3fps real). Quest handles motion tracking — server sends real SAM masks only.

### Frame Similarity / Motion Detection (REMOVED — don't re-add)
Quest handles motion detection. Server runs fresh vision every frame.

## Architecture Notes
- `segment_frame()`: preprocess → TRT vision (sync) → select prompts → TRT decode → mask cache → segments
- No tensor cloning — sync pipeline keeps TRT buffers valid through decoder
- `_cached_fpn_fp32`: caches FP32 conversion of FPN features (avoids re-conversion per prompt)
- Text embeddings pre-computed at init — CLIP text encoder never runs during inference
- Orchestrator background thread runs SAM continuously; `process_frame()` returns cached in <0.1ms
