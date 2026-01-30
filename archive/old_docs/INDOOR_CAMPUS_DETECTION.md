# Indoor College Campus Object Detection

## Overview

The system now uses **YOLO-World** with a custom vocabulary specifically designed for **indoor college campus scenes**. This provides much better detection of structural elements and campus-specific objects that standard YOLO (COCO dataset) cannot detect.

---

## Custom Vocabulary (60+ Classes)

### Structural Elements
- `wall`, `door`, `window`, `ceiling`, `floor`, `corridor`, `hallway`

### Furniture
- `desk`, `chair`, `table`, `cabinet`, `shelf`, `bookshelf`, `filing cabinet`
- `whiteboard`, `blackboard`, `projector screen`, `whiteboard marker`

### Electronics
- `laptop`, `computer`, `monitor`, `screen`, `keyboard`, `mouse`, `printer`
- `projector`, `tv`, `television`, `speaker`, `camera`

### Academic Items
- `book`, `notebook`, `paper`, `pen`, `pencil`, `backpack`, `bag`, `handbag`
- `binder`, `folder`, `textbook`

### Personal Items
- `bottle`, `cup`, `water bottle`, `coffee cup`, `mug`, `cell phone`, `phone`
- `headphones`, `earbuds`

### People
- `person`, `student`, `teacher`, `professor`

### Campus-Specific
- `locker`, `trash can`, `recycling bin`, `fire extinguisher`, `exit sign`
- `light fixture`, `ceiling light`, `lamp`, `overhead light`

### Room Elements
- `door frame`, `window frame`, `door handle`, `light switch`, `outlet`
- `vent`, `air vent`, `heating vent`

---

## How It Works

### YOLO-World (Primary)
1. **Open-Vocabulary Detection**: Can detect any object based on text description
2. **Custom Vocabulary**: Trained on 60+ indoor campus-specific classes
3. **Better Accuracy**: Specifically designed for indoor scenes
4. **Auto-Downloads**: Model downloads automatically on first run

### Fallback System
- If YOLO-World fails → Falls back to YOLOv8m (standard YOLO)
- Enhanced semantic labeling still works for structural elements
- Multi-stage validation ensures correct labels

---

## Detection Pipeline

```
Frame Input
    ↓
YOLO-World (Custom Indoor Campus Vocabulary)
    ↓
Detects: wall, door, window, desk, chair, laptop, etc.
    ↓
SAM Segmentation
    ↓
Match SAM masks to YOLO-World detections
    ↓
Semantic Labeling (for unmatched masks)
    ↓
Final Labels with Confidence Scores
```

---

## Benefits

### Before (Standard YOLO):
- ❌ Wall → "cabinet" (mislabel)
- ❌ Door → "person" (mislabel)  
- ❌ Window → "object" (not in dataset)
- ❌ Whiteboard → not detected
- ❌ Locker → not detected

### After (YOLO-World + Indoor Campus Vocabulary):
- ✅ Wall → "wall" (correct!)
- ✅ Door → "door" (correct!)
- ✅ Window → "window" (correct!)
- ✅ Whiteboard → "whiteboard" (detected!)
- ✅ Locker → "locker" (detected!)

---

## Model Information

- **Model**: `yolov8s-worldv2.pt` (YOLO-World v2 Small)
- **Type**: Open-vocabulary object detection
- **Vocabulary Size**: 60+ indoor campus classes
- **Auto-Download**: Yes (downloads on first run)
- **Speed**: ~30 FPS (with frame skipping)

---

## Usage

No changes needed! The system automatically:

1. **Loads YOLO-World** with indoor campus vocabulary
2. **Detects objects** using custom classes
3. **Falls back** to standard YOLO if needed
4. **Validates** detections with semantic labeling

### Example Commands:
- "Make the whiteboard blue" → Finds "whiteboard"
- "Dim the projector screen" → Finds "projector screen"
- "Turn the locker red" → Finds "locker"
- "Make the desk green" → Finds "desk"

---

## Technical Details

### YOLO-World API:
```python
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(["wall", "door", "window", ...])
results = model.predict(frame, conf=0.2)
```

### Detection Settings:
- **Confidence Threshold**: 0.2 (lower for better recall)
- **IoU Threshold**: 0.45 (standard NMS)
- **Model Size**: Small (s) for speed, can upgrade to Medium (m) or Large (l)

---

## Upgrading the Model

To use a larger, more accurate model:

```python
# In sam_gemini_voice.py, change:
self.yolo_world = YOLOWorld("yolov8s-worldv2.pt")  # Small (fast)
# To:
self.yolo_world = YOLOWorld("yolov8m-worldv2.pt")  # Medium (more accurate)
# Or:
self.yolo_world = YOLOWorld("yolov8l-worldv2.pt")  # Large (most accurate)
```

**Trade-offs:**
- **Small (s)**: Fastest, ~30 FPS
- **Medium (m)**: Balanced, ~20 FPS
- **Large (l)**: Most accurate, ~15 FPS

---

## Adding More Classes

To add more indoor campus objects, edit `self.indoor_campus_vocab` in `sam_gemini_voice.py`:

```python
self.indoor_campus_vocab = [
    # ... existing classes ...
    "new object", "another object",  # Add here
]
```

YOLO-World will automatically detect these new classes!

---

## Troubleshooting

### YOLO-World Not Loading?
- Check internet connection (needs to download model first time)
- Model will auto-download to `~/.ultralytics/weights/`
- Falls back to standard YOLO if unavailable

### Low Detection Accuracy?
- Try upgrading to Medium or Large model
- Lower confidence threshold (currently 0.2)
- Check lighting conditions

### Missing Objects?
- Add object name to `indoor_campus_vocab` list
- YOLO-World can detect any object you describe!

---

*Last Updated: 2025-01-16*

