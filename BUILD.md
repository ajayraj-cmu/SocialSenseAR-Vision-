# Building TensorRT Engines (Required Per Machine)

TensorRT engines are compiled for a specific GPU architecture, driver version, and TRT version. **You must rebuild on each machine.** Engines from one GPU will not work on another.

## Prerequisites

- Windows 10/11 with NVIDIA GPU (Ampere+ recommended: RTX 3060, 3070, 3080, 4060, 4070, 4080, 4090, etc.)
- NVIDIA driver 535+ installed
- Python dependencies installed (`pip install -r requirements.txt`)
- At least 8 GB system RAM free during export
- At least 6 GB VRAM (export temporarily uses ~5 GB)

## Quick Build (3 commands per resolution)

The export script supports multiple resolutions. Files are saved with a `_<res>` suffix so they don't overwrite each other. The server loads the correct set based on `sam3_resolution` in `server/config.py`.

### Build 1008 (default, best quality)

```bash
# Step 1: Export ONNX at 1008x1008
python -m server.vision.sam3_export --resolution 1008

# Step 2: Build vision engine
python build_trt_engine.py sam3_vision_1008.onnx

# Step 3: Build decoder engine
python build_trt_engine.py sam3_topk_decoder_1008.onnx
```

### Build 784 (faster, lower quality)

```bash
# Step 1: Export ONNX at 784x784
python -m server.vision.sam3_export --resolution 784

# Step 2: Build vision engine
python build_trt_engine.py sam3_vision_784.onnx

# Step 3: Build decoder engine
python build_trt_engine.py sam3_topk_decoder_784.onnx
```

### Switching Resolutions

Set `sam3_resolution` in `server/config.py`:

```python
sam3_resolution: int = 1008   # or 784
```

The server looks for `sam3_vision_<res>.engine`, `sam3_topk_decoder_<res>.engine`, and `sam3_meta_<res>.json`. Falls back to unsuffixed filenames if resolution-specific files aren't found.

## Files After Building Both Resolutions

| File | Size | Description |
|------|------|-------------|
| `sam3_vision_1008.onnx` | ~1.8 GB | Vision ONNX at 1008 (intermediate) |
| `sam3_vision_784.onnx` | ~1.8 GB | Vision ONNX at 784 (intermediate) |
| `sam3_topk_decoder_1008.onnx` | ~90 MB | Decoder ONNX at 1008 (intermediate) |
| `sam3_topk_decoder_784.onnx` | ~90 MB | Decoder ONNX at 784 (intermediate) |
| `sam3_meta_1008.json` | ~2 KB | Shapes/config for 1008 (keep) |
| `sam3_meta_784.json` | ~2 KB | Shapes/config for 784 (keep) |
| `sam3_vision_1008.engine` | ~870 MB | TRT FP16 vision at 1008 (keep) |
| `sam3_vision_784.engine` | ~870 MB | TRT FP16 vision at 784 (keep) |
| `sam3_topk_decoder_1008.engine` | ~50 MB | TRT FP16 decoder at 1008 (keep) |
| `sam3_topk_decoder_784.engine` | ~50 MB | TRT FP16 decoder at 784 (keep) |

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
SAM3 resolution: 1008x1008
Vision engine: sam3_vision_1008.engine
TRT vision ready
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

**"No decoder ONNX found"**: Run the export step first.

**OOM during export**: Close Chrome, Discord, etc. The export needs ~5 GB VRAM temporarily.

**OOM during engine build**: `build_trt_engine.py` uses 2 GB workspace. Reduce if needed by editing `set_memory_pool_limit` in the script.

**Engine loads but crashes at runtime**: Delete `.engine` files and rebuild. Engine format changes between TRT versions.

**"TRT vision init failed, reloading PyTorch vision"**: Engine is incompatible with current TRT/GPU. Delete and rebuild.
