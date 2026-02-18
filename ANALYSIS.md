# ANALYSIS.md — SocialSenseAR Codebase Map

## System Diagram

```
User's CV Repo (GitHub)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Website (Flask, 2 pages)               │
│  Page 1: GitHub URL + API keys input                │
│  Page 2: Modal auth + deploy trigger                │
└────────────────────────┬────────────────────────────┘
                         │ deploys
                         ▼
┌─────────────────────────────────────────────────────┐
│           Modal GPU Container (A100)                │
│  ┌────────────────────────────────────────────┐     │
│  │   modal_deploy_wrapper.py                  │     │
│  │   - clones user's GitHub repo              │     │
│  │   - injects env vars                       │     │
│  │   - wraps user's CV model into our         │     │
│  │     existing backend pattern               │     │
│  └──────────────────────────┬─────────────────┘     │
│                             │                       │
│  ┌──────────────────────────▼─────────────────┐     │
│  │   FastAPI + WebSocket server (/ws)          │     │
│  │   (mirrors ServerBackend/modal_app.py)      │     │
│  │   - receives protobuf ClientMessage         │     │
│  │   - runs user's model on frame              │     │
│  │   - returns protobuf ServerMessage          │     │
│  │   - streams processed frames back           │     │
│  └─────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────┘
                         │ wss://...modal.run/ws
                         ▼
┌─────────────────────────────────────────────────────┐
│            Unity Client (generated)                 │
│   Assets/Scripts/SocialSenseClient.cs               │
│   - serverUrl = <Modal endpoint injected>           │
│   - inputMode = Webcam (default)                    │
│   - toggle to AR headset via Inspector              │
└─────────────────────────────────────────────────────┘
```

---

## 2.1 Current Architecture

### Server Backend — `ServerBackend/`

| File | Description |
|------|-------------|
| `server/main.py` | Entry point. Parses CLI args, loads `.env`, creates `ServerConfig`, starts `SocialSenseServer`. |
| `server/config.py` | `ServerConfig` dataclass — single source of truth for all tunables (host, port 8765, device, SAM3 model, GPU id, API keys, etc.). |
| `server/websocket_server.py` | `SocialSenseServer` class. Async WebSocket server via `websockets` library. Listens on `ws://0.0.0.0:8765`. Handles one client at a time. |
| `server/pipeline/orchestrator.py` | `PipelineOrchestrator`. SAM3 background thread. Centroid+IoU tracking. RLE encoding. Gemini labeling. |
| `server/vision/sam3_segmenter.py` | SAM3 PyTorch inference. Text-prompted ("person" + rotating prompts). |
| `server/vision/sam3_trt.py` | TensorRT engine wrapper for SAM3 (30-45+ fps). |
| `server/encoding/rle.py` | uint16 LE run-length encoding. Always starts with background run. |
| `server/proto/socialsense_pb2.py` | Generated protobuf (Python). `ClientMessage` / `ServerMessage`. |
| `modal_app.py` | Modal deployment. `SocialSenseGPU` class with `@modal.asgi_app()` exposing FastAPI at `/ws`. Identical protocol to local server. |
| `modal_config.py` | Modal infra config: GPU type (A100), volumes, secret name (`socialsense-secrets`), timeouts. |

### How Frames Enter the Pipeline

**AR (Headset) mode — Quest 3:**
- `SocialSenseClient.cs` captures via `PassthroughCameraAccess` (Meta XR SDK)
- Async GPU readback → JPEG encode → protobuf `ClientMessage.frame` → WebSocket send

**Webcam mode (default):**
- `SocialSenseClient.cs` captures via `WebCamTexture`
- Same path: JPEG encode → protobuf → WebSocket

**Both modes apply a Y-flip** on the client GPU before encoding (to correct camera orientation).

### Where Frame Processing Happens

```
websocket_server.py:_handle_message()
  → _process_frame_fast()
    → pipeline.process_frame(jpeg_data, width, height, frame_id)
      → orchestrator.py:PipelineOrchestrator.process_frame()
        → SAM3 background thread (sam3_segmenter.py / sam3_trt.py)
        → RLE encode (encoding/rle.py)
        → Gemini labeling (vision/gemini_scene_understanding.py)
```

### Where Streaming Back Happens

- `websocket_server.py:_process_frame_fast()` (line 127) — serializes `pb.ServerMessage` → `websocket.send(response_bytes)`
- In Modal: `modal_app.py:websocket_endpoint()` (line 305) — `await websocket.send_bytes(response.SerializeToString())`
- Response is cached: only rebuilt when `id(result.segments)` changes

### WebSocket Endpoints

| Mode | Endpoint |
|------|----------|
| Local | `ws://0.0.0.0:8765` (plain `websockets` library) |
| Modal | `wss://<app>.modal.run/ws` (FastAPI WebSocket via uvicorn) |

Protocol: binary protobuf. `ClientMessage` sends frames/audio/control. `ServerMessage` returns segments with RLE masks.

### Unity Client Connection

`unitySetUp/SocialSenseAR-Unity/Assets/Scripts/SocialSenseClient.cs`:
- Line 43: `public string serverUrl = "wss://ajraj2006--socialsense-ar-gpu-socialsensegpu-web.modal.run/ws";`
- Uses `NativeWebSocket` package
- Connects on `Start()`, sends frames at `targetFPS`

### Webcam vs Headset Toggle

`SocialSenseClient.cs:31-35`:
```csharp
public enum InputMode { Auto, AR, Webcam }

[Header("Input Mode")]
public InputMode inputMode = InputMode.Webcam;  // DEFAULT = Webcam
```

- `InputMode.Webcam` — uses `WebCamTexture`
- `InputMode.AR` — uses `PassthroughCameraAccess` (Quest headset)
- `InputMode.Auto` — defaults to Webcam
- Toggle via Unity Inspector at runtime

---

## 2.2 Coupling & Assumptions

### Hardcoded Values

| Location | Value | Notes |
|----------|-------|-------|
| `server/config.py:21` | `port = 8765` | Local default WebSocket port |
| `server/config.py:38` | `sam3_model = "facebook/sam3"` | Requires HF gated access |
| `modal_config.py:20` | `GPU_TYPE = "A100"` | Overridable via `MODAL_GPU` env var |
| `modal_config.py:53` | `SECRET_NAME = "socialsense-secrets"` | Modal secret containing API keys |
| `modal_config.py:87` | `APP_NAME = "socialsense-ar-gpu"` | Modal app name |
| `SocialSenseClient.cs:43` | `serverUrl = "wss://ajraj2006--..."` | Hardcoded Modal URL — THIS is what we inject |
| `SocialSenseClient.cs:35` | `inputMode = InputMode.Webcam` | Default webcam mode |

### Minimal Existing Config

- `.env` / `ServerBackend/.env` — API keys loaded by `main.py` via `python-dotenv`
- `server/config.py` — all runtime tunables in one dataclass
- `modal_config.py` — Modal infra settings
- Modal secret `socialsense-secrets` — API keys for cloud deployment

---

## 2.3 Integration Points We Reuse

1. **`modal_app.py`** — We mirror this pattern for user CV repos. Same `@app.cls` + `@modal.asgi_app()` structure. Same `/ws` WebSocket endpoint. Same protobuf protocol.

2. **`server/proto/socialsense_pb2.py`** + **`SocialsenseMessages.cs`** — We do NOT change the protocol. User's model output is adapted into existing `SegmentData` objects.

3. **`SocialSenseClient.cs:43`** — The single line we inject: `serverUrl = "<deployed-modal-url>/ws"`. This is the only change to the Unity client.

4. **`SocialSenseClient.cs:35`** — `inputMode = InputMode.Webcam` stays as-is (already default).

5. **`modal_config.py`** — GPU type, secret name, volume name reused as templates.

6. **`ServerBackend/.env.example`** — Template for required env vars (GEMINI_API_KEY, OPENAI_API_KEY, HF_SAM_TOKEN).

---

## Key Files Summary

```
SAMARSDK/
├── ServerBackend/
│   ├── server/main.py              # Entry point — CLI, config, server start
│   ├── server/config.py            # All tunables (port 8765, GPU, model paths)
│   ├── server/websocket_server.py  # WebSocket handler, frame processing, response caching
│   ├── server/pipeline/orchestrator.py  # SAM background thread, tracking, RLE
│   ├── server/vision/sam3_segmenter.py  # SAM3 inference (PyTorch)
│   ├── server/vision/sam3_trt.py   # TensorRT wrapper
│   ├── server/encoding/rle.py      # RLE encode/decode
│   ├── server/proto/socialsense_pb2.py  # Protobuf (Python)
│   ├── modal_app.py                # Modal deployment — FastAPI /ws endpoint
│   └── modal_config.py             # Modal infra config
└── unitySetUp/SocialSenseAR-Unity/
    └── Assets/Scripts/
        ├── SocialSenseClient.cs    # Camera capture, WebSocket, serverUrl field (line 43)
        ├── OverlayRenderer.cs      # RLE decode + sphere overlay render
        ├── AudioStreamer.cs        # PCM16 audio streaming
        ├── VoiceAgentHUD.cs        # Voice agent UI
        └── Proto/SocialsenseMessages.cs  # C# protobuf
```
