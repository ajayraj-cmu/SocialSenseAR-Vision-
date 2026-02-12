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

## Features

- **SAM3 Text-Prompted Segmentation**: Segment only requested objects (no "scan everything")
- **Voice Agent Pipeline**: Natural language control via "Hey Vibe" wake word
- **Per-Object Effects**: Blur, dim, pixelate, highlight individual objects
- **Full-Screen Filters**: Global dim/warm/cool/night modes
- **Persistent Effects**: Objects remain tracked even when leaving/reentering view
- **Real-Time Dashboard**: Live visualization of effects + voice agent state

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# For voice agent (optional)
pip install sounddevice  # microphone capture for test client
```

### 2. Set API keys

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your-key-here       # Required for voice agent reasoning
OPENAI_API_KEY=your-key-here       # Required for voice transcription (Whisper)
```

### 3. Download models

Place these in the project root and `models/` directory:
- `FastSAM-s.pt` (root) - from [ultralytics](https://github.com/ultralytics/ultralytics)
- `models/selfie_segmenter.tflite` - from [MediaPipe](https://developers.google.com/mediapipe)
- `models/face_landmarker.task`
- `models/hand_landmarker.task`
- `models/pose_landmarker_full.task`

### 4. Run the server

**Standard mode (SAM3 + Voice Agent):**
```bash
python -u -m server.main --device cuda --pipeline sam3
```

**Legacy mode (FastSAM + MediaPipe + Gemini):**
```bash
python -u -m server.main --device cuda
```

### 5. Test without a headset

**With voice control:**
```bash
python -m server.test_client_voice
```
Say "Hey Vibe, blur the laptop, thank you" to test!

**Without voice (keyboard only):**
```bash
python -m server.test_client
```

## Voice Agent

Control segmentation and effects using natural language! See [VOICE_AGENT.md](./docs/VOICE_AGENT.md) for complete documentation.

**Quick example:**
```
You: "Hey Vibe, my environment is too bright and overstimulating, dim it, thank you"
Server: ✓ Applied dim filter
        ✓ Applied dim to lamp, window, screen
```

**Features:**
- Wake word: "Hey Vibe"
- End phrase: "Thank you"
- Natural language understanding (Gemini)
- On-demand scene analysis (Gemini Vision)
- Persistent effects across frames
- No "scan everything" — segments only requested objects

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
├── .env.example             # Env var template (copy to .env)
├── .gitignore
├── README.md
├── requirements.txt         # Python dependencies
├── modal_app.py             # Modal GPU deployment
├── modal_config.py          # Modal/config (GPU, SAM3, etc.)
├── config/                  # SAM3 metadata (TRT shapes, etc.)
│   ├── sam3_meta.json
│   ├── sam3_meta_784.json
│   └── sam3_meta_1008.json
├── docs/                    # All documentation
│   ├── RUNNING.md           # How to run (incl. Modal + secrets)
│   ├── BUILD.md             # TRT/ONNX build
│   ├── MODAL_*.md           # Modal setup & deployment
│   ├── VOICE_AGENT.md
│   └── ...
├── scripts/                 # Build and one-off scripts
│   ├── build_trt_engine.py
│   ├── build_trt_int8.py
│   └── sam_gemini_voice.py
├── tests/                   # Test and benchmark runners
│   ├── test_fps.py
│   ├── test_voice_agent_smoke.py
│   ├── test_local.bat
│   └── ...
├── tools/                   # Client and dev tools
│   ├── webcam_modal_client.py
│   ├── monitor_modal.sh
│   └── quest_recorder/
├── cache/                   # Generated (gitignored): benchmarks, TRT cache
├── server/                  # Core server package
│   ├── main.py
│   ├── config.py
│   ├── websocket_server.py
│   ├── test_client.py
│   ├── proto/, encoding/, pipeline/, vision/, audio/
│   └── ...
└── models/                  # MediaPipe/FastSAM models (not committed)
```
