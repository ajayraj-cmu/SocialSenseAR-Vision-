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
- **Architecture**: ViT-32L vision encoder + CLIP text encoder (pre-computed at init) + TRT top-k decoder
- **Vision encoder**: PyTorch FP16 + torch.compile, runs async on background thread (~800ms per encode)
- **TRT decoder**: top-k engine, B=1 sequential per prompt, ~35ms/prompt, NaN-free
- **Prompts**: "person" every frame + 1 rotating object prompt (2 prompts/frame = ~70ms decode)
- **Effective mask update rate**: ~1.3 fps (bottlenecked by 800ms vision encode)
- **No optical flow / motion detection** — removed, Quest handles this

### Pre-built Assets (project root)
| File | Size | Status |
|------|------|--------|
| `sam3_vision.engine` | 872 MB | EXISTS but NOT loaded — see Phase 1 below |
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
- `server/vision/sam3_segmenter.py` — SAM3 segmenter (vision encode, TRT decoder, caching, async)
- `server/pipeline/orchestrator.py` — Pipeline: SAM loop, tracking, Gemini labeling
- `server/test_client.py` — WebSocket client with camera, mask overlay, FPS counter, blur commands
- `server/config.py` — ServerConfig (sam3_model, confidence_threshold, cache_ttl, etc.)
- `server/vision/sam3_export.py` — ONNX export scripts for vision + decoder
- `sam3_meta.json` — Tensor shapes and preprocessing params for TRT runtime

## Next Optimization: Enable TRT Vision Engine

The `sam3_vision.engine` already exists (872 MB) and code to load it exists in sam3_segmenter.py
(`_init_trt_vision` line ~524, `_run_vision_trt` line ~589) but is deliberately disabled in
`initialize()`. A previous comment claimed "TRT vision barely helps" but this was likely from
a bad test — PyTorch vision takes 800ms, TRT should do 15-25ms.

Steps:
1. Wire `_init_trt_vision()` into `initialize()` after decoder loading
2. Remove async vision encoding (unnecessary with fast TRT vision)
3. Make vision synchronous: preprocess → TRT vision → TRT decode → postprocess
4. `del self._model.vision_encoder` after TRT vision loads to free VRAM
5. Remove or increase proactive TRT context rebuild interval (currently every 4 calls = 600ms spikes)

Expected: ~25ms vision + ~70ms decoder = ~95ms/frame = ~10 fps

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
4. **Proactive context rebuild every 4 calls** — `self._trt_engine.create_execution_context()`

The context rebuild causes ~600ms spikes (TRT warmup on first execution after rebuild).
This interval should be increased or removed now that the core fix (B=1 + sync + default stream) is in place.

**Key insight**: TRT execution contexts are NOT thread-safe and do NOT tolerate dynamic batch shape
changes well. If you ever re-enable batched TRT decode, test extensively for NaN.

### torch.compile + CUDA Graphs
- `mode="reduce-overhead"` and `mode="max-autotune"` use CUDA graphs which **conflict with SAM3's
  FPN neck** — the graph replays overwrite FPN output buffers, producing stale/corrupt data
- Only `mode="default"` is safe with SAM3
- torch.compile on individual submodules (vision_encoder, detr_encoder, detr_decoder, mask_decoder)
  works fine — just don't compile the top-level `model.forward()`

### Async Vision Encoding Pitfalls
- `_queue_vision_encode()` MUST be called AFTER the decoder check, not before — otherwise
  `_vision_encode_pending=True` causes the decoder to skip every frame
- `_clone_vision_output()` is essential — compiled model can overwrite tensor storage on the next call
- `torch.cuda.synchronize()` in the background thread after vision encode is required to prevent
  TRT decoder contention on the main thread

### Optical Flow Tracking (REMOVED — don't re-add)
- Lucas-Kanade optical flow was added to track mask positions between SAM decoder updates
- It inflated the FPS counter (showed 18fps when real mask update was 1.3fps) because it counted
  shifted-but-unchanged masks as "new" updates
- The Quest VR headset handles motion/tracking on its end — server should only send real SAM masks
- Removed from orchestrator.py entirely

### Frame Similarity / Motion Detection (REMOVED — don't re-add)
- `_frames_similar()` compared frame thumbnails to skip vision re-encoding on similar frames
- Removed because the Quest handles motion detection
- Server now always queues a new vision encode when the previous one finishes

### Windows-Specific Issues
1. **tqdm output buffering**: Model weight loading progress bar fills Python's TextIOWrapper buffer.
   Server appears frozen at ~27% but is actually running. Use `-u` flag.
2. **Process cleanup**: Use `wmic process call terminate` — `taskkill /F` often fails with CUDA.
   Always verify with `nvidia-smi --query-gpu=memory.used --format=csv,noheader`.
3. **Debug logging**: Python's stderr is block-buffered when redirected. Use `_dbg()` function in
   sam3_segmenter.py which writes directly to `~/sam3_debug.log` (unbuffered, bypasses logging).
4. **Bash tool quirks**: Windows cmd `for` loops and pipes don't work in the bash tool.
   Use Python one-liners instead: `python -c "..."`.
5. **Server startup**: Takes 25-45 seconds (model load + torch.compile warmup). Don't assume it hung.
   Check with `netstat -ano | findstr 8765` to see if port is listening.

### VRAM Budget (RTX 3060 Laptop — 6 GB)
- PyTorch SAM3 FP16: ~1 GB model weights
- TRT decoder engine: ~50 MB + ~200 MB workspace
- TRT vision engine: ~872 MB + workspace (if enabled)
- Pre-allocated tensors: ~100 MB
- CUDA runtime: ~500 MB
- Total with vision TRT: ~2.5 GB, leaves ~3.5 GB for activations — should fit
- **After loading TRT vision, delete `model.vision_encoder` to reclaim ~400 MB**

## Architecture Notes
- `segment_frame()` flow: vision cache check → async encode queue → prompt selection → decoder (TRT or PyTorch) → mask cache → build segments
- `_clone_vision_output()` deep-clones HuggingFace ModelOutput to prevent CUDA graph buffer overwrites
- `_cached_fpn_fp32` caches FP16→FP32 conversion of FPN features for TRT decoder (avoids re-conversion every frame)
- Text embeddings are pre-computed at init for all 13 prompts — CLIP text encoder never runs during inference
- Orchestrator runs SAM in continuous background thread; `process_frame()` returns cached result in <0.1ms
