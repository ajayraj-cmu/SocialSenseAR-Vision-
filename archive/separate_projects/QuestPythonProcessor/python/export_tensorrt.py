#!/usr/bin/env python3
"""
Export YOLOv8n-seg to TensorRT format for maximum inference speed on NVIDIA GPUs.
Run this once on your Windows machine with NVIDIA GPU.

Requirements:
- NVIDIA GPU with CUDA
- TensorRT installed (pip install tensorrt)
"""

import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA not available. TensorRT export requires NVIDIA GPU.")
    print("Make sure you have:")
    print("  1. NVIDIA GPU")
    print("  2. CUDA installed")
    print("  3. PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    exit(1)

print(f"CUDA available: {torch.cuda.get_device_name(0)}")
print()

from ultralytics import YOLO

print("Loading YOLOv8n-seg model...")
model = YOLO('yolov8n-seg.pt')

print("Exporting to TensorRT (this may take a few minutes)...")
print()

# Export with half precision for speed
# batch=2 for stereo (left + right eye processed together)
model.export(
    format='engine',
    half=True,  # FP16 for speed
    imgsz=256,  # Small size for maximum speed
    batch=2,    # Batch of 2 for stereo processing (left + right eye)
    simplify=True,
    workspace=4,  # GB of GPU memory for TensorRT optimization
)

print()
print("Done! TensorRT model saved as: yolov8n-seg.engine")
print("Supports batch=2 for stereo processing.")
print("Run: python main.py --source quest_tcp --ui quest_tcp")
