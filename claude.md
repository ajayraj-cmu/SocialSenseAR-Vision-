# DIRECTIVE: SAM FPS Optimization (ACTIVE)

## Goal
Achieve 10+ FPS real mask update rate on the segmentation pipeline (SAM3-based).

## Rules
1. **MUST use SAM** — no switching away from SAM-family models entirely
2. **Target: 10+ FPS real mask updates** on the user's machine (Windows, RTX 3060 Laptop GPU, CUDA)
3. **DO NOT sacrifice segmentation quality** — same model, same input, just faster execution
4. **No motion detection on server** — the Quest VR headset handles motion detection on its end
5. **Always test on real server** — see "How to Test" below
6. **Batch changes before testing** — don't run server/client after every tiny change. Make all related code changes, then test once.

## CONSTRAINT: Quality Preservation
- NO switching to lower-quality models (no MobileSAM, NanoSAM, etc.)
- NO resolution reduction that hurts segmentation
- Only approaches that produce identical or near-identical masks:
  - TensorRT compilation of same model
  - ONNX Runtime with same model
  - torch.compile() JIT
  - FP16 (near-zero quality loss)
  - Pipeline/Python overhead elimination

## Current State (as of 2026-02-01)
- **Model**: facebook/sam3 (~530M params) — text-prompted segmentation
- **Architecture**: TRT vision encoder + TRT top-k decoder (fully TensorRT pipeline)
- **Vision encoder**: TRT FP16, ~200ms per encode (synchronous)
- **TRT decoder**: top-k engine, B=1 sequential per prompt, ~28ms/prompt, NaN-free
- **Prompts**: "person" every frame + 1 rotating object prompt (2 prompts/frame = ~56ms decode)
- **Pipeline**: Fully synchronous — preprocess → TRT vision → TRT decode → build segments
- **Effective mask update rate**: ~3.8 fps (~258ms/frame), rock-solid over 280+ frames
- **VRAM**: 3.4 GB stable (out of 6 GB)
- **No optical flow / motion detection** — removed, Quest handles this

### Pre-built Assets (project root)
| File | Size | Status |
|------|------|--------|
| `sam3_vision.engine` | 872 MB | Active, working (FP16) |
| `sam3_topk_decoder.engine` | 50 MB | Active, working |
| `sam3_vision.onnx` | 1770 MB | Source for vision engine |
| `sam3_topk_decoder.onnx` | 92 MB | Source for decoder engine |
| `sam3_meta.json` | metadata | Vision output shapes, preprocessing params |

## How to Test
```bash
# Terminal 1 — server (IMPORTANT: use -u for unbuffered output)
python -u -m server.main --device cuda

# Terminal 2 — client
python -u -m server.test_client --show
```
Watch "SAM X fps" / "Mask X.X fps" in top-left of client window.

## Key Files
- `server/vision/sam3_segmenter.py` — SAM3 segmenter (TRT vision + TRT decoder, synchronous pipeline)
- `server/pipeline/orchestrator.py` — Pipeline: SAM loop, tracking, Gemini labeling
- `server/test_client.py` — WebSocket client with camera, mask overlay, FPS counter, blur commands
- `server/config.py` — ServerConfig (sam3_model, confidence_threshold, cache_ttl, etc.)
- `server/vision/sam3_export.py` — ONNX export scripts for vision + decoder
- `sam3_meta.json` — Tensor shapes and preprocessing params for TRT runtime

## Bottleneck Analysis
The TRT vision encoder runs at ~200ms per frame on RTX 3060 Laptop. This is the dominant
bottleneck (78% of frame time). The ViT-32L encoder has ~400M params and processes 1008×1008
inputs — even with TRT FP16, this saturates the GPU's compute units.

### Possible next optimizations (to reach 10+ fps):
1. **Reduce vision encoder input resolution** — if SAM3 supports smaller inputs (e.g. 504×504),
   this could cut vision time by ~4x. Must verify mask quality first.
2. **INT8 quantization of vision encoder** — requires calibration dataset but could halve latency
3. **Vision encoder pruning/distillation** — reduce ViT layers or width while keeping SAM3 interface
4. **Skip vision on similar frames** — but Quest handles motion detection, so this is questionable
5. **Batch multiple frames** — amortize overhead (but adds latency per frame)

---

## LESSONS LEARNED (Bugs & Failed Approaches)

### TRT Decoder NaN Bug (FIXED)
**Symptom**: TRT top-k decoder produced NaN scores after ~5-7 vision encode cycles. Masks disappeared.
FPN tensor norms stayed valid throughout — the corruption was inside TRT engine state.

**Root Cause**: Combination of dynamic batch sizes (B=1 vs B=2) and cross-CUDA-stream races between
the async vision encode thread and the TRT decoder on the main thread.

**What DID NOT fix it**:
- Disabling torch.compile on vision_encoder (NaN still occurred)
- Switching from async to sync vision encoding alone (NaN still occurred)
- Cloning vision output tensors (already implemented, didn't help)

**What DID fix it (all required together)**:
1. **B=1 per prompt** — run TRT sequentially for each prompt instead of batched B=2
2. **Default CUDA stream (0)** — `ctx.execute_async_v3(0)` instead of custom stream
3. **Full `torch.cuda.synchronize()` before each TRT call** — prevents cross-stream races

**Key insight**: TRT execution contexts are NOT thread-safe and do NOT tolerate dynamic batch shape
changes well. If you ever re-enable batched TRT decode, test extensively for NaN.

### TRT Context Proactive Rebuild (REMOVED — don't re-add)
**Symptom**: `create_execution_context()` every N calls caused 600-2700ms spikes.
More critically, after the rebuild, the new context ran ~15x SLOWER permanently (885ms vs 58ms).

**Root Cause**: Fresh TRT contexts require internal warmup. On RTX 3060 Laptop,
the warmup never fully completes for the decoder — performance degrades permanently.

**Fix**: Removed proactive rebuild entirely. With B=1 + default stream + full sync,
NaN hasn't occurred in 280+ frame test runs. Reactive NaN check stays as safety net.

### TRT Vision Output Cloning (REMOVED — don't re-add)
**Symptom**: `.clone()` on 9 TRT vision output tensors (~200+ MB/frame) caused CUDA OOM at frame 64.
PyTorch allocated 9.62 GiB on a 6 GB GPU due to memory caching not freeing fast enough.

**Fix**: Reference pre-allocated TRT output buffers directly. Safe because synchronous pipeline
ensures buffers won't be overwritten until next frame's vision encode (after decoder finishes).

### VRAM-Aware Initialization (CRITICAL)
Loading order matters on 6 GB GPU. Wrong order causes OOM:

**Correct order (3.4 GB total)**:
1. Load PyTorch model (for text encoder + processor only)
2. Pre-compute text embeddings
3. Delete PyTorch vision_encoder (~400 MB)
4. Load TRT vision engine (~872 MB) — fits because vision_encoder was freed
5. Load TRT decoder engine (~50 MB)
6. Delete text_encoder, detr_encoder, detr_decoder, mask_decoder — not needed with TRT
7. torch.cuda.empty_cache()

**What NOT to do**:
- Don't torch.compile decoder submodules if TRT decoder is available (wastes ~1 GB VRAM)
- Don't load TRT vision before deleting PyTorch vision_encoder (OOM)
- Don't do PyTorch decoder warmup if TRT decoder is available (wastes time + VRAM)

### torch.compile + CUDA Graphs
- `mode="reduce-overhead"` and `mode="max-autotune"` use CUDA graphs which **conflict with SAM3's
  FPN neck** — the graph replays overwrite FPN output buffers, producing stale/corrupt data
- Only `mode="default"` is safe with SAM3
- torch.compile on individual submodules (vision_encoder, detr_encoder, detr_decoder, mask_decoder)
  works fine — just don't compile the top-level `model.forward()`

### Async Vision Encoding (REMOVED — don't re-add)
- Async vision encoding on background thread was removed in favor of synchronous pipeline
- With TRT vision at 200ms, async was counterproductive — decoder perpetually SKIPPED because
  `_vision_encode_pending=True` held the GPU, preventing decoder execution
- Masks only updated every ~3 seconds with async; 258ms per frame with synchronous
- If PyTorch vision fallback is ever needed (800ms), async might help again — but only with
  careful CUDA stream management (see NaN bug above)

### Optical Flow Tracking (REMOVED — don't re-add)
- Lucas-Kanade optical flow was added to track mask positions between SAM decoder updates
- It inflated the FPS counter (showed 18fps when real mask update was 1.3fps) because it counted
  shifted-but-unchanged masks as "new" updates
- The Quest VR headset handles motion/tracking on its end — server should only send real SAM masks
- Removed from orchestrator.py entirely

### Frame Similarity / Motion Detection (REMOVED — don't re-add)
- `_frames_similar()` compared frame thumbnails to skip vision re-encoding on similar frames
- Removed because the Quest handles motion detection
- Server now runs fresh vision encode every frame

### Windows-Specific Issues
1. **tqdm output buffering**: Model weight loading progress bar fills Python's TextIOWrapper buffer.
   Server appears frozen at ~27% but is actually running. Use `-u` flag.
2. **Process cleanup**: Use `wmic process call terminate` — `taskkill /F` often fails with CUDA.
   Always verify with `nvidia-smi --query-gpu=memory.used --format=csv,noheader`.
3. **Debug logging**: Python's stderr is block-buffered when redirected. Use `_dbg()` function in
   sam3_segmenter.py which writes directly to `~/sam3_debug.log` (unbuffered, bypasses logging).
4. **Bash tool quirks**: Windows cmd `for` loops and pipes don't work in the bash tool.
   Use Python one-liners instead: `python -c "..."`.
5. **Server startup**: Takes 15-25 seconds (model load, no torch.compile with TRT). Don't assume it hung.
   Check with `netstat -ano | findstr 8765` to see if port is listening.

### VRAM Budget (RTX 3060 Laptop — 6 GB)
- PyTorch SAM3 FP16 (model shell — no vision, no decoder submodules): ~200 MB
- TRT vision engine: ~872 MB + ~200 MB workspace
- TRT decoder engine: ~50 MB + ~200 MB workspace
- Pre-allocated TRT tensors (vision + decoder): ~200 MB
- Pre-computed text embeddings (FP32): ~10 MB
- CUDA runtime: ~500 MB
- **Total: ~3.4 GB stable, leaves 2.6 GB headroom**

## Architecture Notes
- `segment_frame()` flow: preprocess → TRT vision encode (sync) → select prompts → TRT decode → mask cache → build segments
- No cloning needed — synchronous pipeline ensures TRT buffers are valid through decoder execution
- `_cached_fpn_fp32` caches FP32 conversion of FPN features for TRT decoder (avoids re-conversion every prompt)
- Text embeddings are pre-computed at init for all 13 prompts — CLIP text encoder never runs during inference
- Orchestrator runs SAM in continuous background thread; `process_frame()` returns cached result in <0.1ms
