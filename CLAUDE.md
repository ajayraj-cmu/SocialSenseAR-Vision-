# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SocialSenseAR is a real-time AR segmentation system for Meta Quest 3. The Quest captures passthrough camera frames and sends them via WebSocket to a Python GPU server running SAM3 (Segment Anything Model 3), which returns RLE-encoded masks. Unity renders the masks as overlays on passthrough using per-fragment pinhole projection. Users control effects via voice ("Hey Vibe, blur the laptop, thank you").

## Commands

### Running the Server

```bash
# Standard mode (SAM3 + Voice Agent) - always use -u for unbuffered output
python -u -m server.main --device cuda --pipeline sam3

# Legacy mode (FastSAM + MediaPipe + Gemini)
python -u -m server.main --device cuda --pipeline legacy

# Grounded-SAM2 mode (GroundingDINO + SAM2)
python -u -m server.main --device cuda --pipeline grounded-sam2

# Debug logging
LOG_LEVEL=DEBUG python -u -m server.main --device cuda
```

**Important**: The `-u` flag is critical on Windows to prevent buffered output that makes the server appear frozen.

**Server flags**: `--host`, `--port`, `--device cuda|cpu`, `--gpu-id`, `--no-audio`, `--no-emotion`, `--stub` (no ML models), `--pipeline sam3|grounded-sam2|legacy`.

### Testing Without Quest

```bash
# Test with webcam and voice control
python -m server.test_client_voice

# Test with webcam only (keyboard controls)
python -u -m server.test_client --show
```

Say "Hey Vibe, blur the laptop, thank you" to test voice control.

### Running Tests

```bash
# Voice agent smoke test
python -m pytest tests/test_voice_agent_smoke.py

# FPS benchmark
python -m pytest tests/test_fps.py

# SAM3 debug test
python -m pytest tests/test_sam3_debug.py

# TensorRT vs PyTorch comparison
python -m pytest tests/test_trt_vs_pytorch.py
```

### TensorRT Engine Build

```bash
# Build INT32 TensorRT engine (default 1008 resolution)
python scripts/build_trt_engine.py

# Build INT8 quantized engine
python scripts/build_trt_int8.py
```

Engines are built for specific resolutions (1008 or 784). The resolution must match `sam3_resolution` in `server/config.py`.

### Modal Cloud Deployment

```bash
# Deploy to Modal cloud
modal deploy modal_app.py

# Development mode with hot reload
modal serve modal_app.py

# Test Modal deployment with webcam
python tools/webcam_modal_client.py --url wss://<your-modal-url>/ws --show
```

Modal deployment requires a Modal secret named `socialsense-secrets` containing all required API keys.

### Dashboard

The dashboard is automatically available when the server runs:
```
http://localhost:8765/dashboard
```

Displays live FPS, active effects, tracked segments, and voice agent state.

## Architecture

```
Quest 3 (Unity)                     GPU Server (Python)
┌──────────────────┐   WebSocket    ┌──────────────────────────┐
│ Camera JPEG      │ ──────────────>│ websocket_server.py      │
│ + frame_id/pose  │    protobuf    │   → orchestrator.py      │
│                  │                │     → sam3_segmenter.py   │
│ Overlay render   │ <──────────────│     → voice_agent.py     │
│ (RLE masks)      │                │   → RLE + protobuf resp  │
└──────────────────┘                └──────────────────────────┘
```

### Server Pipeline (`server/`)

**Entry**: `main.py` → CLI args, config, starts WebSocket server

**Core flow**:
1. `websocket_server.py` - Async WebSocket server, protobuf serialization, response caching, serves dashboard at `/dashboard`
2. `pipeline/orchestrator.py` - SAM background thread (~3.8 FPS), tracking (centroid+IoU matching), RLE encoding at 75% resolution, Gemini labeling
3. `vision/sam3_segmenter.py` - SAM3 TRT or PyTorch inference
4. `vision/sam3_trt.py` - TensorRT engine wrapper
5. `encoding/rle.py` - uint16 LE run-length encoding (always starts with background run)
6. `proto/socialsense_pb2.py` - Generated protobuf messages

**Voice pipeline** (optional, enabled by default):
- `audio/voice_agent.py` - WakeWordGate ("Hey Vibe") → UtteranceAssembler → VoiceCommandPlanner (Gemini) → effect execution
- `audio/transcriber.py` - Whisper-based transcription (cloud or local)
- `audio/personaplex_voice_agent.py` + `audio/personaplex_bridge.py` - Alternative speech-to-speech via NVIDIA PersonaPlex-7B

**Config**: `server/config.py` - `ServerConfig` dataclass is the single source of truth for all tunables (SAM resolution, confidence thresholds, tracking params, RLE scaling, effect settings, etc.)

### Unity Client (`unity-client/Assets/Scripts/`)

- **SocialSenseClient.cs** - Camera capture, Y-flip blit, async GPU readback, JPEG encode, WebSocket send, stores pose per frame_id
- **OverlayRenderer.cs** - RLE decode (with Y-flip), composite RGBA32 texture, sphere geometry, inverse-pinhole label placement
- **Proto/SocialsenseMessages.cs** - Hand-written C# protobuf (not auto-generated)
- **Shaders/SegmentOverlay.shader** - Per-fragment pinhole projection using camera intrinsics, stereo support, edge fade

The Unity project is at `unity-client/`. Open in Unity Editor to modify client-side rendering, shaders, or protobuf messages.

## Contract Files (Unity ↔ Server)

Changing these requires updating **both** Python and C# sides:
- `server/proto/socialsense_pb2.py` - Protobuf schema (Python)
- `unity-client/Assets/Scripts/Proto/SocialsenseMessages.cs` - Protobuf messages (C#)
- `server/encoding/rle.py` - RLE mask format (uint16 LE)
- `server/websocket_server.py` - WebSocket protocol handler

All protobuf coordinates are normalized [0,1]. RLE masks are at 75% of frame resolution (configurable in `config.py`).

## The Y-Flip Chain (Critical for Debugging)

```
Quest camera (correct) → Client GPU blit Y-flip (upside-down JPEG sent)
→ Server cv2.flip(frame, 0) (corrects back) → SAM processes correct frame
→ RLE mask in OpenCV row order (0=top) → Unity decode: unityRow = h-1-rleRow
→ Shader: LensOffset has 180deg X flip built-in → Final: correct ✓
```

## Frame Timing (Critical for World-Locked Overlays)

- `frame_id` = latest frame received by server
- `masks_frame_id` = frame whose pixels SAM actually processed (lags 1-3 frames)

Unity **MUST** use `masks_frame_id` to look up capture-time head rotation for world-locked projection. Using current rotation causes angle-dependent drift.

**Rotation consistency rule**: `_CameraRotationMatrix` and `_LeftCameraWorldPos` MUST use the same head rotation from `masks_frame_id`. Mixing capture-time and current causes angle-dependent misalignment.

## Environment Variables

Required in `.env` (see `.env.example`):
```bash
GEMINI_API_KEY=...      # Voice agent reasoning + scene understanding
OPENAI_API_KEY=...      # Whisper transcription (cloud mode)
HF_SAM_TOKEN=...        # Gated access for facebook/sam3
```

Optional:
```bash
PERSONAPLEX_ENABLED=true           # Enable PersonaPlex speech-to-speech backend
HF_PERSONAPLEX_TOKEN=...           # Gated access for nvidia/personaplex-7b-v1
PERSONAPLEX_URL=ws://...           # PersonaPlex WebSocket endpoint
LOG_LEVEL=DEBUG                    # Enable debug logging
```

For Modal deployment, add these to a Modal secret named `socialsense-secrets`.

## PersonaPlex Integration

The `personaPlex/` directory contains an alternative implementation using NVIDIA PersonaPlex-7B for speech-to-speech voice interaction (instead of Gemini + Whisper). Same server architecture, different voice backend. Enable with `PERSONAPLEX_ENABLED=true` in `.env`.

## Key Documentation

- `docs/claude.md` - Detailed architecture + coordinate system guide (read this for vision/rendering work)
- `docs/VOICE_AGENT.md` - Voice control design and command examples
- `docs/BUILD.md` - TensorRT export instructions
- `docs/TUNING_GUIDE.md` - Performance optimization reference
- `docs/RUNNING.md` - Modal deployment setup
- `docs/OPTIMIZATIONS.md` - SAM3 optimization history (read before optimization work)
- `docs/BORDER_SMOOTHNESS_AND_LATENCY_SUGGESTIONS.md` - RLE mask smoothing tuning

## Directory Structure

```
SocialSenseAR/
├── server/                         # Python server package
│   ├── main.py                     # Entry point
│   ├── config.py                   # ServerConfig (all tunables)
│   ├── websocket_server.py         # WebSocket + dashboard
│   ├── dashboard.py                # Dashboard HTML/JS
│   ├── test_client.py              # Webcam test client
│   ├── pipeline/orchestrator.py    # SAM background loop + tracking
│   ├── vision/
│   │   ├── sam3_segmenter.py       # SAM3 inference
│   │   ├── sam3_trt.py             # TensorRT wrapper
│   │   └── segment_data.py         # SegmentData class
│   ├── audio/
│   │   ├── voice_agent.py          # Gemini-based voice agent
│   │   ├── personaplex_voice_agent.py
│   │   └── personaplex_bridge.py
│   ├── encoding/rle.py             # RLE encode/decode
│   └── proto/socialsense_pb2.py    # Generated protobuf
├── unity-client/Assets/Scripts/
│   ├── SocialSenseClient.cs        # Camera + WebSocket
│   ├── OverlayRenderer.cs          # RLE decode + rendering
│   ├── Proto/SocialsenseMessages.cs # C# protobuf
│   └── Shaders/SegmentOverlay.shader
├── tests/                          # Tests and benchmarks
├── scripts/                        # Build scripts (TRT, ONNX)
├── tools/                          # Client utilities
├── docs/                           # Technical documentation
├── config/                         # SAM3 metadata (TRT shapes)
├── personaPlex/                    # PersonaPlex alternative voice backend
├── modal_app.py                    # Modal deployment
└── requirements.txt                # Python dependencies
```

## Common Debugging

**Server not starting**: Check port 8765 is free
- Windows: `netstat -ano | findstr 8765`
- Linux/Mac: `lsof -i :8765`

**GPU memory not freed**: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
- Windows: Use `wmic process call terminate` not `taskkill /F` for CUDA processes

**Buffered output on Windows**: Always use `python -u` flag

**Unity overlay misalignment**: Verify `masks_frame_id` is used for pose lookup, not current frame. Check `_CameraRotationMatrix` and `_LeftCameraWorldPos` use same head rotation.

**Voice agent not responding**: Check `.env` has `GEMINI_API_KEY` and `OPENAI_API_KEY`. Try `LOG_LEVEL=DEBUG`.
