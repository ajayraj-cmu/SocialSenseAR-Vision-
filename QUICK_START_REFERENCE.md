# SAMARSDK Quick Reference Guide

## What is SAMARSDK?

A real-time AR segmentation system for Meta Quest 3:
- **Python GPU server** processes passthrough camera frames
- **SAM3 model** performs text-prompted image segmentation
- **Quest client** (Unity) renders masks as AR overlays
- **Optional voice control** via Gemini-based command parsing

---

## Key Concepts

### SAM3 (Segment Anything Model 3)
- **Text-prompted**: You specify WHAT to segment via text prompts
- **Not class-limited**: Can segment "person", "laptop", "wall", etc. or any description
- **Output**: Binary masks for each object + confidence scores

### Pipeline Architecture
```
Quest Camera Frame (JPEG)
    ↓ WebSocket
Python Server (SAM3)
    ↓ [Vision Encoder] → [Decoder for each prompt] → [RLE Encode]
Protobuf ServerMessage (segments + masks)
    ↓ WebSocket
Unity Client
    ↓ Render masks as overlays on passthrough
```

### Performance Tiers
- **TensorRT** (TRT engines): 30-50 FPS (20-33ms)
- **PyTorch** (no TRT): 3-5 FPS (200-350ms)
- **PyTorch + torch.compile**: 5-10 FPS (100-200ms)

---

## File Organization

```
ServerBackend/
├── server/
│   ├── main.py                         ← Entry point (CLI + config)
│   ├── config.py                       ← ServerConfig (all tunables)
│   ├── websocket_server.py             ← WebSocket handler
│   ├── vision/sam3_segmenter.py        ← SAM3 inference (1945 lines!)
│   ├── vision/sam3_trt.py              ← TensorRT wrapper
│   ├── vision/sam3_export.py           ← ONNX export script
│   ├── pipeline/orchestrator.py        ← Combines segmentation+audio
│   ├── encoding/rle.py                 ← Mask compression
│   ├── audio/voice_agent.py            ← Gemini-based commands
│   └── proto/socialsense.proto         ← Protocol Buffers
├── config/
│   ├── sam3_meta.json                  ← 1008x1008 shape metadata
│   ├── sam3_meta_784.json              ← 784x784 variant (faster)
├── scripts/
│   ├── build_trt_engine.py             ← Build TRT from ONNX
│   └── build_trt_int8.py               ← INT8 quantization
├── tools/
│   ├── video_processor.py              ← Process video → masks
│   ├── story_renderer.py               ← Render video with effects
│   └── story_schema.py                 ← Story JSON format
├── tests/
│   ├── test_sam3_debug.py
│   ├── test_trt_vs_pytorch.py
│   └── test_fps.py
└── requirements.txt
```

---

## Starting the Server

```bash
# Default: SAM3 + voice agent on CUDA
python -u -m server.main --device cuda --pipeline sam3

# Legacy: FastSAM + Gemini
python -u -m server.main --device cuda --pipeline legacy

# CPU mode (slow!)
python -u -m server.main --device cpu

# Stub mode (no ML, for testing)
python -u -m server.main --stub

# With metrics logging
python -u -m server.main --metrics-log metrics.jsonl
```

**Important**: Always use `-u` flag on Windows!

---

## SAM3 Segmenter Flow

### Initialization
1. Load SAM3 model + processor from HuggingFace
2. Pre-tokenize all prompts in `ALL_PROMPTS`
3. Pre-compute text embeddings
4. Check for TensorRT engines (if found, use TRT; else use PyTorch)
5. Warmup inference

### Per-Frame Processing
```
Input: BGR frame (any resolution, e.g., 1920x1080)
  ↓
Preprocess to 1008x1008 RGB (~2ms)
  ↓
Vision encoder → vision_embeds (9 tensors) (~15-25ms TRT, ~350ms PyTorch)
  ↓
For each active prompt:
  - Check mask_cache (skip if fresh)
  - Decoder → pred_masks (200, 288, 288), logits (~3-8ms TRT per prompt)
  - Select best query by max confidence
  - Extract mask, bbox, confidence
  - Create SegmentData
  ↓
Output: list[SegmentData]
```

### Key Optimizations
1. **Text embedding caching**: Precompute once, reuse forever
2. **Vision output caching**: One encode, multiple decodes
3. **Mask result caching**: Avoid decoder for static objects
4. **Batched decoding**: Process all prompts in one forward pass
5. **Fast preprocessing**: cv2.resize + GPU normalize (~2ms vs ~15ms)
6. **TensorRT**: Direct inference (eliminates PyTorch overhead)

---

## Configuration Tuning

### SAM3 Parameters (`server/config.py`)

```python
# Model
sam3_model: str = "facebook/sam3"
sam3_resolution: int = 1008  # 1008 or 784 (must match TRT engines!)

# Inference
sam3_prompts_per_frame: int = 1      # How many prompts per frame
sam3_cache_ttl: float = 4.0          # Cache mask for 4 seconds
sam3_confidence_threshold: float = 0.12  # Min confidence to include

# Tracking
track_max_age: float = 1.0           # Drop tracks after 1 sec no match
track_match_max_dist: float = 0.20   # Max centroid movement (% of diagonal)

# RLE Encoding (mask compression)
rle_scale: float = 1.0               # 1.0 = full res, 0.75 = 75%
rle_edge_blur_kernel: int = 7        # Smooth edges (5, 7, or 9)
rle_min_mask_area: int = 64          # Drop tiny masks (<64 pixels)

# Voice (optional)
audio_enabled: bool = True
personaplex_enabled: bool = False
gemini_vision_enabled: bool = False  # Disable vision API (faster)
```

---

## Working with Prompts

### Pre-defined Prompts
```python
ALL_PROMPTS = [
    "person", "face", "chair", "table", "desk", "couch", "monitor",
    "laptop", "lamp", "wall", "floor", "door", "window",
]
```

### Dynamic Prompt Control (Thread-safe)

```python
# Via voice agent (automatic)
User says: "blur the laptop"
→ Voice agent parses command
→ add_prompt("laptop")
→ apply_effect("laptop", "blur")

# Via API (programmatic)
segmenter.set_active_prompts({"person", "laptop", "wall"})
segmenter.add_prompt("chair")
segmenter.remove_prompt("floor")
```

### Asset Class Mapping
```
"person", "face"                     → "person"
"chair", "table", "desk", "couch"   → "furniture"
"monitor", "laptop"                 → "electronics"
"lamp", "light"                      → "lighting"
"wall", "floor", "door", "window"   → "structure"
```

---

## WebSocket Protocol

### Client → Server (Protobuf)

```protobuf
message ClientMessage {
  uint64 frame_id;                 // Frame sequence
  double timestamp_ms;             // Client timestamp
  oneof payload {
    FramePayload frame;            // JPEG frame
    AudioPayload audio;            // PCM16 audio
    ControlPayload control;        // "blur laptop", etc.
  }
}
```

### Server → Client (Protobuf)

```protobuf
message ServerMessage {
  uint64 frame_id;                 // Echo client frame_id
  uint64 masks_frame_id;           // Frame ID for masks (may lag)
  repeated SceneSegment segments;  // Detected objects
  ConversationState conversation;  // Voice agent state
}

message SceneSegment {
  string label;                    // Prompt text
  string asset_class;              // Semantic category
  float confidence;                // 0-1 SAM confidence
  BoundingBox bbox;
  bytes rle_mask;                  // RLE-encoded binary mask
  uint32 mask_width;
  uint32 mask_height;
  float center_x, center_y;        // Centroid
  string track_id;                 // Persistent ID
  EffectMetadata effect;           // Applied effects
}
```

---

## Video Processing Pipeline

### Process Video → Story

```bash
# Via local server (server must be running on port 8765)
python -u -m tools.video_processor \
    --input my_video.mp4 \
    --output story_output/ \
    --server-url ws://localhost:8765 \
    --transcribe-audio \
    --progress

# Via Modal cloud GPU
python -u -m tools.video_processor \
    --input my_video.mp4 \
    --output story_output/ \
    --server-url wss://your-modal-url.modal.run/ws \
    --progress

# Locally (no server, requires CUDA)
python -u -m tools.video_processor \
    --input my_video.mp4 \
    --output story_output/ \
    --local --device cuda \
    --prompts "person,laptop,chair"
```

### Output Files
- `story.json`: Metadata + frame timeline + effects
- `masks.bin`: Binary RLE-encoded mask data
- `my_video.mp4`: Original video (copied)

### Story JSON Format
```json
{
  "metadata": {
    "video_filename": "my_video.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30,
    "total_frames": 900
  },
  "frames": [
    {
      "frame_id": 0,
      "segments": [
        {
          "track_id": "T001",
          "label": "person",
          "asset_class": "person",
          "mask_offset": 0,
          "mask_size": 2048
        }
      ]
    }
  ],
  "effects_timeline": [
    {
      "frame_start": 50,
      "frame_end": 150,
      "track_id": "T001",
      "effect_type": "blur",
      "intensity": 1.0
    }
  ]
}
```

### Render Video with Effects
```bash
python -u -m tools.story_renderer \
    --story story_output/ \
    --output story_output/rendered.mp4 \
    --mode all
```

Modes: `outline`, `filled`, `effects`, `all`

---

## RLE Encoding (Mask Compression)

### Format
- Sequence of `uint16` (little-endian) run lengths
- Always starts with background run (0-count)
- Example: `[0, 2, 2, 2]` = 0 bg pixels, 2 fg, 2 bg, 2 fg

### Compression Ratio
- 300 KB raw mask (288x288) → 200-2000 bytes RLE
- **~150x-1500x compression!**

### Key Functions
```python
rle_bytes = encode_rle(mask)              # Mask → RLE bytes
mask = decode_rle(rle_bytes, 288, 288)    # RLE bytes → Mask
bbox = mask_to_bbox(mask)                 # → (x_min, y_min, x_max, y_max)
cx, cy = mask_centroid(mask)              # → (centroid_x, centroid_y)
```

---

## TensorRT Acceleration

### What is TensorRT?
- NVIDIA's inference engine (compiled, optimized)
- 10-50x faster than PyTorch
- Requires `.engine` files (built from ONNX models)

### Build TensorRT Engines

**Step 1: Export to ONNX**
```bash
python -m server.vision.sam3_export --resolution 1008
# Produces:
# - sam3_vision_1008.onnx  (~1.2 GB)
# - sam3_decoder_1008.onnx (~500 MB)
# - sam3_topk_decoder_1008.onnx (~500 MB, RECOMMENDED)
# - sam3_meta_1008.json
```

**Step 2: Build TensorRT Engines**
```bash
python scripts/build_trt_engine.py --resolution 1008
# Requires: trtexec command-line tool (NVIDIA TensorRT installation)
# Produces:
# - sam3_vision.engine
# - sam3_decoder.engine (or sam3_topk_decoder.engine)
```

### Resolution Options
- `1008x1008`: Default, full quality
- `784x784`: ~40% faster (fewer patches), recommended for real-time
- `896x896`: ~21% fewer patches

---

## Testing & Benchmarking

### Run Tests
```bash
# Basic SAM3 test
python -m pytest tests/test_sam3_debug.py -v

# Performance comparison (TRT vs PyTorch)
python -m pytest tests/test_trt_vs_pytorch.py -v

# FPS benchmark
python -m pytest tests/test_fps.py -v

# Voice agent test
python -m pytest tests/test_voice_agent_smoke.py -v
```

### Manual Testing with Webcam
```bash
# Terminal 1: Start server
python -u -m server.main --device cuda

# Terminal 2: Test with webcam + voice
python -m server.test_client_voice

# Say: "Hey Vibe, blur the person, thank you"
```

---

## Environment Setup

### `.env` File
```bash
# Required for voice agent
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key

# Required for SAM3 (gated HuggingFace model)
HF_SAM_TOKEN=your_hf_token

# Optional: PersonaPlex (speech-to-speech)
PERSONAPLEX_ENABLED=true
HF_PERSONAPLEX_TOKEN=your_personaplex_token
PERSONAPLEX_URL=ws://personaplex-server:5000

# Optional: Debugging
LOG_LEVEL=DEBUG
```

### Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# PyTorch + CUDA (separate step!)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# HuggingFace login (for gated models)
python -c "from huggingface_hub import login; login(token='hf_YOUR_TOKEN')"
```

---

## Key Metrics

### Latency (TensorRT, RTX 3060 Laptop)
- JPEG decode: ~2ms
- Vision encoder: ~15-25ms
- Decoder per prompt: ~3-8ms
- RLE encoding: ~5-10ms
- **Total**: ~20-33ms → 30-50 FPS

### Latency (PyTorch)
- Vision encoder: ~350ms
- Decoder per prompt: ~83ms
- **Total**: ~200-350ms → 3-5 FPS

### Memory
- Model weights: 1.25 GB (FP16)
- TRT engines: ~1.7 GB total
- Runtime buffers: ~512 MB

---

## Troubleshooting

### Server not starting
```bash
# Check if port 8765 is already in use
lsof -i :8765  # Mac/Linux
netstat -ano | findstr 8765  # Windows
```

### Buffered output on Windows
```bash
# ALWAYS use -u flag!
python -u -m server.main
```

### GPU out of memory
```bash
# Reduce resolution to 784
# Edit config.py: sam3_resolution = 784

# Or use CPU (slow!)
python -u -m server.main --device cpu
```

### TRT engines not loading
```bash
# Check if engines exist in project root
ls -la /Users/ajayraj/SAMARSDK/ServerBackend/sam3_*.engine

# If missing, build them:
python -m server.vision.sam3_export --resolution 1008
python scripts/build_trt_engine.py
```

### Voice agent not responding
```bash
# Check .env has API keys
cat .env | grep GEMINI_API_KEY

# Enable debug logging
LOG_LEVEL=DEBUG python -u -m server.main
```

---

## Quick Links

- **Full exploration**: `SAMARSDK_CODEBASE_EXPLORATION.md`
- **Architecture docs**: `ServerBackend/docs/claude.md`
- **Voice agent**: `ServerBackend/docs/VOICE_AGENT.md`
- **TRT build guide**: `ServerBackend/docs/BUILD.md`
- **Performance tuning**: `ServerBackend/docs/TUNING_GUIDE.md`
- **Running on Modal**: `ServerBackend/docs/RUNNING.md`

