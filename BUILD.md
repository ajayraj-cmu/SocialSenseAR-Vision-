# Building TensorRT Engines (Required Per Machine)

TensorRT engines are compiled for a specific GPU architecture, driver version, and TRT version. **You must rebuild on each machine.** Engines from one GPU will not work on another.

## Prerequisites

- Windows 10/11 with NVIDIA GPU (Ampere+ recommended: RTX 3060, 3070, 3080, 4060, 4070, 4080, 4090, etc.)
- NVIDIA driver 535+ installed
- Python dependencies installed (`pip install -r requirements.txt`)
- At least 8 GB system RAM free during export
- At least 6 GB VRAM (export temporarily uses ~5 GB)

## Quick Build (3 commands)

```bash
# Step 1: Export SAM3 model to ONNX (~3-5 min, downloads ~2 GB model on first run)
python -m server.vision.sam3_export

# Step 2: Build vision encoder TRT engine (~2-5 min)
python build_trt_engine.py sam3_vision.onnx

# Step 3: Build decoder TRT engine (~1-2 min)
python build_trt_engine.py sam3_topk_decoder.onnx
```

After building, you should have these files in the project root:

| File | Size | Description |
|------|------|-------------|
| `sam3_vision.onnx` | ~1.8 GB | Vision encoder ONNX (intermediate, can delete after build) |
| `sam3_topk_decoder.onnx` | ~90 MB | Decoder ONNX (intermediate, can delete after build) |
| `sam3_decoder.onnx` | ~90 MB | Full decoder ONNX (fallback, can delete) |
| `sam3_meta.json` | ~2 KB | Tensor shapes and preprocessing config (keep) |
| `sam3_vision.engine` | ~870 MB | TRT FP16 vision engine (keep) |
| `sam3_topk_decoder.engine` | ~50 MB | TRT FP16 decoder engine (keep) |

## Step-by-Step Details

### Step 1: ONNX Export

```bash
python -m server.vision.sam3_export
```

This downloads `facebook/sam3` from HuggingFace (gated model -- you need access), loads it in PyTorch, and exports three ONNX files plus `sam3_meta.json`.

**First run**: Downloads ~2 GB of model weights to your HuggingFace cache. Requires `huggingface-cli login` if the model is gated.

**VRAM note**: The export temporarily needs ~5 GB VRAM for FP32 tracing. Close other GPU-heavy apps first. If you get OOM errors, the script moves components to CPU automatically.

### Step 2: Build Vision Engine

```bash
python build_trt_engine.py sam3_vision.onnx
```

Builds a TensorRT FP16 engine from the vision ONNX. This is the largest engine (~870 MB) and takes the longest to build. TRT profiles different kernel implementations and picks the fastest for your specific GPU.

### Step 3: Build Decoder Engine

```bash
python build_trt_engine.py sam3_topk_decoder.onnx
```

Builds the top-k decoder engine (~50 MB). This is fast.

## Verification

Start the server and client:

```bash
# Terminal 1
python -u -m server.main --device cuda

# Terminal 2
python -u -m server.test_client --show
```

Check server startup logs for:
```
Loading TRT vision engine: ...sam3_vision.engine
TRT vision ready
Loading TRT decoder engine: ...sam3_topk_decoder.engine
TRT decoder ready
```

If TRT engines are missing or fail to load, the server falls back to PyTorch (slower but functional).

## Optional: INT8 Vision Engine

INT8 provides marginal improvement (~7%) on memory-bandwidth-bound ViTs. Only worth it if you want every last millisecond.

```bash
# Capture calibration frames (aims webcam at your typical scene)
python capture_calib_frames.py

# Build INT8 engine (~50 min due to calibration)
python build_trt_int8.py
```

This produces `sam3_vision_int8.engine`. The server auto-detects and prefers it over the FP16 engine.

## Rebuilding

You need to rebuild engines when:
- Moving to a different GPU
- Updating NVIDIA drivers (major versions)
- Updating the `tensorrt` Python package
- Re-exporting ONNX with different settings

You do NOT need to rebuild when:
- Changing server config (prompts, thresholds, cache TTL)
- Updating Python or other packages
- Changing client code

## Troubleshooting

**"No decoder ONNX found"**: Run Step 1 first (`python -m server.vision.sam3_export`).

**OOM during export**: Close Chrome, Discord, etc. The export needs ~5 GB VRAM temporarily.

**OOM during engine build**: `build_trt_engine.py` uses 2 GB workspace. Reduce if needed by editing `set_memory_pool_limit` in the script.

**Engine loads but crashes at runtime**: Delete `.engine` files and rebuild. Engine format changes between TRT versions.

**"TRT vision init failed, reloading PyTorch vision"**: Engine is incompatible with current TRT/GPU. Delete and rebuild.
