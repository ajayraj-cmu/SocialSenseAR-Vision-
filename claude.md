# DIRECTIVE: SAM FPS Optimization (ACTIVE)

## Goal
Achieve 30+ FPS on the segmentation pipeline (SAM3-based). DO NOT GIVE UP.

## Rules
1. **MUST use SAM** — no switching away from SAM-family models entirely
2. **Target: 30+ FPS** on the user's machine (Windows, RTX 3060 Laptop GPU, CUDA)
3. **Iterate continuously** — start server + client together, measure FPS, try next optimization, repeat
4. **DO NOT sacrifice segmentation quality** — same model, same input, just faster execution
5. **Do not enter plan mode** — just execute and test
6. **Do not stop** until 30+ FPS is confirmed
7. **Always test on real server** — `python -m server.main --device cuda` + `python -m server.test_client --show`

## CONSTRAINT: Quality Preservation
- NO switching to lower-quality models (no MobileSAM, NanoSAM, etc.)
- NO resolution reduction that hurts segmentation
- Only approaches that produce identical or near-identical masks:
  - TensorRT compilation of same model
  - ONNX Runtime with same model
  - torch.compile() JIT
  - FP16 (near-zero quality loss)
  - Pipeline/Python overhead elimination
  - Custom CUDA post-processing kernels

## Current State (SAM3)
- **Model**: facebook/sam3 (~530M params) — text-prompted segmentation
- **Current FPS**: ~12.3 FPS cached (81ms), ~1.9 FPS uncached (527ms)
- **Backend**: PyTorch vision encoder (torch.compile FP16) + TensorRT top-k decoder
- **Prompts**: "person" every frame + 1 rotating object prompt
- **Vision cache**: MSE threshold 500, ~80% hit rate on real camera
- Server: `python -m server.main --device cuda`
- Client: `python -m server.test_client --show`
- FPS displayed in client overlay top-left as "SAM X fps"
- Key file: `server/vision/sam3_segmenter.py`
- TRT export: `server/vision/sam3_export.py`
- TRT engine builder: `build_trt_engine.py`
- Benchmark: `test_fps.py`
- Config: `server/config.py`
- Pipeline: `server/pipeline/orchestrator.py`
- GPU: RTX 3060 Laptop (Ampere, compute 8.6, 6GB VRAM)

## Optimization Priority Order (remaining)
1. **Batch TRT prompts** — single batch=N call instead of N sequential calls
2. **Cache FP32 FPN features** — skip FP16→FP32 conversion on vision cache hits
3. **TensorRT vision encoder** — 280ms → 20-30ms (for uncached frames)
4. **Split DETR encoder from decoder** — run DETR encoder once, decoder per prompt
5. **Pipeline overlap** — decoder runs while next frame preprocesses
6. **Custom CUDA kernels** — for mask decode, post-processing

## How to Test
```bash
# Terminal 1 — server
python -m server.main --device cuda

# Terminal 2 — client
python -m server.test_client --show
```
Watch "SAM X fps" in top-left of client window. Must show 30+.

## Future Features (queued)
- Text input in client to modify server behavior (e.g. "make person blue", "hide screen")
