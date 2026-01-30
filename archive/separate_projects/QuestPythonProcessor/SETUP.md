# Quest Python Processor

Capture and process Quest passthrough video in Python.

## Requirements

- Python 3.8+
- Meta Quest (2/3/Pro) with Developer Mode enabled
- USB cable

## Setup

### Mac

```bash
# Install system dependencies
brew install portaudio

# Install Python dependencies
cd python
pip install -r requirements.txt
```

### Windows (with NVIDIA GPU)

```bash
# Install PyTorch with CUDA first (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
cd python
pip install -r requirements.txt

# Optional: Export TensorRT model for 2-3x faster inference
pip install tensorrt
python export_tensorrt.py
```

### Enable Developer Mode on Quest

1. Go to Settings → System → Developer on your Quest
2. Enable Developer Mode
3. Connect Quest to computer via USB
4. Put on Quest and accept "Allow USB debugging" prompt

## Usage

```bash
cd python
python fast_capture.py
```

This will:
1. Auto-detect GPU (CUDA on Windows, MPS on Mac, or CPU)
2. Auto-load TensorRT model if available (Windows)
3. Connect to Quest via MYScrcpy (scrcpy v3.2 protocol)
4. Capture frames at full resolution
5. Display the Quest screen in a window

**Controls:**
- Press `p` to toggle person highlighting (YOLO segmentation)
- Press `q` to quit

## Performance

| Platform | GPU | YOLO FPS | Notes |
|----------|-----|----------|-------|
| Mac M1/M2/M3 | MPS (Metal) | ~8 fps | |
| Windows | NVIDIA RTX 3060 | ~20-30 fps | |
| Windows | NVIDIA RTX 4080 | ~40-60 fps | |
| Windows | NVIDIA + TensorRT | ~60-100 fps | Run `export_tensorrt.py` first |
| Any | CPU only | ~1-3 fps | |

## TensorRT (Windows Only)

For maximum speed on NVIDIA GPUs, export the TensorRT model:

```bash
# Install TensorRT
pip install tensorrt

# Export model (run once, takes a few minutes)
python export_tensorrt.py
```

This creates `yolov8n-seg.engine` which is automatically loaded by `fast_capture.py`.

## Person Highlighting

When enabled (press `p`), the script:
1. Detects people using YOLOv8 segmentation
2. Keeps the person closest to center in color
3. Converts everything else to grayscale

Each eye is processed independently for accurate stereo vision.

## Backup: ADB Capture

If MYScrcpy has issues, use the slower but reliable ADB method:

```bash
python adb_capture.py
```

This captures screenshots directly from Quest (~5-10 FPS).
