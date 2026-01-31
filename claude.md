# DIRECTIVE: SAM FPS Optimization (ACTIVE)

## Goal
Achieve 40+ FPS on the segmentation pipeline (SAM-based). DO NOT GIVE UP.

## Rules
1. **MUST use SAM** — no switching away from SAM-family models entirely
2. **Target: 40+ FPS** on the user's machine (Windows, RTX 3060 Laptop GPU, CUDA)
3. **Iterate continuously** — spawn server, measure FPS, try next optimization, repeat
4. **DO NOT sacrifice segmentation quality** — same model, same input, just faster execution
5. **Do not enter plan mode** — just execute and test
6. **Do not stop** until 40+ FPS is confirmed

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

## Current State
- FastSAM-s.pt runs at ~3 FPS (ultralytics YOLO-based, PyTorch)
- Server: `python -m server.main --device cuda`
- Client: `python -m server.test_client --show`
- FPS displayed in client overlay top-left as "SAM X fps"
- Key file: `server/vision/fastsam_segmenter.py`
- Config: `server/config.py`
- Pipeline: `server/pipeline/orchestrator.py`
- GPU: RTX 3060 Laptop (Ampere, compute 8.6, supports TRT/FP16/INT8)

## Optimization Priority Order
1. **TensorRT export** — compile FastSAM-s to TensorRT engine for 5-10x speedup (ZERO quality loss)
2. **ONNX Runtime + CUDA EP** — alternative runtime, zero quality loss
3. **torch.compile()** — PyTorch 2.x inductor backend, zero quality loss
4. **FP16 inference** — halve memory bandwidth, near-zero quality loss
5. **Pipeline optimization** — eliminate Python overhead in post-processing
6. **Custom CUDA kernels** — for mask decode, NMS, used_pixels
7. **C++ inference server** — eliminate GIL entirely

## How to Test
```bash
# Terminal 1 — server
python -m server.main --device cuda

# Terminal 2 — client
python -m server.test_client --show
```
Watch "SAM X fps" in top-left of client window. Must show 40+.
