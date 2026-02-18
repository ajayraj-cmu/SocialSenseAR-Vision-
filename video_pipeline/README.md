# JSON-Driven SAM3 Video Pipeline

A standalone pipeline that processes a pre-recorded video using SAM3 segmentation,
driven entirely by a JSON instructions file — no audio/voice commands needed.

## Folder Structure

```
video_pipeline/
├── videos/                  ← Put your input videos here
├── overlay_images/          ← Put your overlay images here
│   └── RoyalPattern.png     ← Sample image (already included)
├── output/                  ← Rendered results go here (auto-created)
├── json_driven_processor.py ← Step 1: SAM3 mask extraction
├── json_driven_renderer.py  ← Step 2: Render effects onto video
├── example_instructions.json ← Copy this and fill in your events
└── README.md                ← This file
```

## Quick Start

### 1. Drop your files in place

- Put your video in `video_pipeline/videos/my_video.mp4`
- Put any overlay images in `video_pipeline/overlay_images/MyImage.png`
- Copy `example_instructions.json` → `video_pipeline/my_instructions.json` and edit it

### 2. Run Step 1: Process video with SAM3

```bash
# From SAMARSDK root, via Modal cloud (recommended — no GPU needed locally)
python -m video_pipeline.json_driven_processor \
    --input video_pipeline/videos/my_video.mp4 \
    --instructions video_pipeline/my_instructions.json \
    --output video_pipeline/output/ \
    --server-url wss://your-modal-app.modal.run/ws

# Via local SAM3 server (must be running on port 8765)
python -m video_pipeline.json_driven_processor \
    --input video_pipeline/videos/my_video.mp4 \
    --instructions video_pipeline/my_instructions.json \
    --output video_pipeline/output/ \
    --server-url ws://localhost:8765

# Locally without a server (requires CUDA GPU)
python -m video_pipeline.json_driven_processor \
    --input video_pipeline/videos/my_video.mp4 \
    --instructions video_pipeline/my_instructions.json \
    --output video_pipeline/output/ \
    --local --device cuda
```

Produces `video_pipeline/output/story.json` + `video_pipeline/output/masks.bin`.

### 3. Run Step 2: Render effects

```bash
python -m video_pipeline.json_driven_renderer \
    --story video_pipeline/output/ \
    --video video_pipeline/videos/my_video.mp4 \
    --output video_pipeline/output/rendered.mp4 \
    --overlay-dir video_pipeline/overlay_images/

# With live preview window
python -m video_pipeline.json_driven_renderer \
    --story video_pipeline/output/ \
    --output video_pipeline/output/rendered.mp4 \
    --show-preview
```

---

## JSON Instructions Format

Every event needs either `timestamp_s` (seconds) or `frame_idx` (frame number).

### Event Types

---

#### `segment_target`

Tell SAM3 which objects to detect and segment. You can fire this multiple times
at different points in the video to introduce new objects.

```json
{
  "type": "segment_target",
  "timestamp_s": 0.0,
  "targets": ["wall", "person", "laptop"]
}
```

---

#### `overlay_image`

Tile an image from `overlay_images/` over a SAM3-segmented mask using OpenCV.
The image repeats (tiles) to cover the entire mask area, regardless of shape.

```json
{
  "type": "overlay_image",
  "timestamp_s": 2.0,
  "subject": "wall",
  "image_name": "RoyalPattern",
  "opacity": 0.85,
  "invert": false,
  "end_timestamp_s": 10.0
}
```

| Field | Required | Description |
|---|---|---|
| `subject` | yes | SAM3 label to apply image to (e.g. `"wall"`) |
| `image_name` | yes | Filename in `overlay_images/` (extension optional) |
| `opacity` | no | 0.0–1.0 (default `1.0`) |
| `invert` | no | `true` = apply to everything EXCEPT the mask |
| `end_timestamp_s` | no | When to stop (omit = hold to end of video) |

---

#### `effect`

Apply a standard visual effect to a mask.

```json
{
  "type": "effect",
  "timestamp_s": 5.0,
  "subject": "person",
  "effect": "blur",
  "intensity": 0.8,
  "invert": false,
  "end_timestamp_s": 12.0
}
```

Available `effect` values:

| Value | Description |
|---|---|
| `blur` | Gaussian blur |
| `grayscale` | Desaturate to gray |
| `dim` | Darken |
| `highlight` | Brighten + warm tint |
| `color` | Color tint (requires `color_hex`) |
| `pixelate` | Pixelation |
| `frosted_glass` | Frosted blur |
| `redact` | Pixelate (identity redaction) |
| `outline` | Colored outline only |

```json
{
  "type": "effect",
  "timestamp_s": 8.0,
  "subject": "wall",
  "effect": "color",
  "color_hex": "#FF6B6B",
  "intensity": 0.6,
  "invert": false,
  "end_timestamp_s": 15.0
}
```

---

#### `conversation_mode`

Triggers the inverted-mask AR animation:
- The **nearest/largest detected person stays in full color**
- The **surroundings fade to grayscale + soft blur**
- A **cinematic sweep animation** converges inward from the edges toward the person

```json
{
  "type": "conversation_mode",
  "timestamp_s": 30.0,
  "subject": "person",
  "animation_duration_s": 1.5,
  "blur_intensity": 0.85,
  "end_timestamp_s": 45.0
}
```

| Field | Required | Description |
|---|---|---|
| `subject` | yes | Usually `"person"` — must match a SAM3 segment |
| `animation_duration_s` | no | Sweep-in duration in seconds (default `1.5`) |
| `blur_intensity` | no | Background blur strength 0.0–1.0 (default `1.0`) |
| `end_timestamp_s` | no | When to stop |

---

## Notes

- You can use `frame_idx` instead of `timestamp_s` for any event.
- Use `end_frame_idx` instead of `end_timestamp_s` if you prefer frames.
- `segment_target` can be repeated mid-video to introduce new objects.
- For `overlay_image`, `subject` must match what SAM3 detected. If SAM3 calls
  it `"wall"` you must write `"wall"` in `subject`.
- The `overlay_images/` directory already has `RoyalPattern.png` as a sample.
  Add any PNG/JPG/WEBP image there and reference it by name (no path, no extension needed).
