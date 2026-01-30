# SocialSenseAR - Vision Server

Real-time segmentation server for AR glasses. Runs FastSAM + MediaPipe on a GPU PC, streams results to Meta Quest 3 via WebSocket + protobuf.

## Architecture

```
Quest 3 (Unity)                    GPU PC (Python)
┌──────────────┐    WebSocket     ┌──────────────────┐
│ Camera JPEG  │ ────────────────>│ FastSAM + MediaPipe
│              │                  │ + Gemini Vision   │
│ Overlay RGBA │ <────────────────│ Protobuf + RLE    │
└──────────────┘                  └──────────────────┘
```

### Server modules (`server/`)

| File | Responsibility | Safe to edit? |
|------|---------------|---------------|
| `websocket_server.py` | WebSocket + protobuf serialization | CONTRACT (Unity depends on this) |
| `proto/socialsense_pb2.py` | Protobuf schema | CONTRACT |
| `encoding/rle.py` | RLE mask encoding (uint16 LE) | CONTRACT |
| `vision/fastsam_segmenter.py` | FastSAM inference + orchestration | Yes |
| `vision/mediapipe_detector.py` | Person/face/hands/pose detection | Yes |
| `vision/semantic_labeler.py` | Position/size heuristic labels | Yes |
| `vision/mask_refinement.py` | Bilateral + GrabCut refinement | Yes |
| `vision/gemini_labeler.py` | Gemini Vision API labeling | Yes |
| `pipeline/orchestrator.py` | Threading, caching, tracking | Yes |
| `config.py` | All tunables | Yes |
| `test_client.py` | Webcam test client (no headset) | Yes |

**CONTRACT** files: changing these requires updating the Unity C# side.
**Safe** files: edit freely without affecting the headset.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

### 3. Download models

Place these in the project root and `models/` directory:
- `FastSAM-s.pt` (root) - from [ultralytics](https://github.com/ultralytics/ultralytics)
- `models/selfie_segmenter.tflite` - from [MediaPipe](https://developers.google.com/mediapipe)
- `models/face_landmarker.task`
- `models/hand_landmarker.task`
- `models/pose_landmarker_full.task`

### 4. Run the server

```bash
python -m server.main --device cuda
```

### 5. Test without a headset

```bash
python -m server.test_client --show
```

## Server Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8765` | WebSocket port |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--gpu-id` | `0` | GPU index |
| `--no-audio` | | Disable audio pipeline |
| `--no-emotion` | | Disable emotion detection |
| `--stub` | | Run without ML models (for testing) |

SAM confidence and resolution are set in `server/config.py` (`fastsam_conf`, `fastsam_imgsz`).

## Debug Logging

Each vision module logs its own timing at DEBUG level. Set `LOG_LEVEL` to see per-stage breakdowns:

```bash
# Windows
set LOG_LEVEL=DEBUG && python -m server.main --device cuda

# Linux/Mac
LOG_LEVEL=DEBUG python -m server.main --device cuda
```

- **MediaPipe**: selfie, face, hands, pose timing per frame
- **Mask refinement**: bilateral, morphology, GrabCut per mask
- **Semantic labeler**: label decision + area/aspect/position per mask
- **FastSAM segmenter**: aggregated timing every 10th frame

## Project Structure

```
SocialSenseAR-Vision/
├── .env                    # API keys (not committed)
├── .gitignore
├── requirements.txt        # Python dependencies
├── FastSAM-s.pt           # FastSAM model (not committed)
├── models/                 # MediaPipe models (not committed)
│   ├── selfie_segmenter.tflite
│   ├── face_landmarker.task
│   ├── hand_landmarker.task
│   └── pose_landmarker_full.task
└── server/
    ├── main.py             # Entry point
    ├── config.py           # All tunables
    ├── websocket_server.py # WebSocket + protobuf
    ├── test_client.py      # Webcam test client
    ├── proto/
    │   └── socialsense_pb2.py
    ├── encoding/
    │   └── rle.py
    ├── pipeline/
    │   └── orchestrator.py
    ├── vision/
    │   ├── fastsam_segmenter.py
    │   ├── mediapipe_detector.py
    │   ├── semantic_labeler.py
    │   ├── mask_refinement.py
    │   └── gemini_labeler.py
    └── audio/
        ├── transcriber.py
        └── social_cue_detector.py
```
