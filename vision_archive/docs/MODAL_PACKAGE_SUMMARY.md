# SocialSenseAR Modal Deployment Package - Summary

**Date:** February 8, 2026
**Status:** ✅ Ready for Deployment
**Architecture:** Fully modularized with Modal GPU offload

---

## What's Been Done

Your SocialSenseAR workspace is now **completely packaged and modularized** for Modal deployment. All compute-intensive operations have been configured to run on Modal's GPU infrastructure.

### Files Created/Modified

#### 1. **`modal_config.py`** (NEW)
Centralized configuration file for all Modal deployment settings:
- GPU type selection (A10G, L4, T4, A100, H100)
- SAM3 model configuration (resolution, prompts, thresholds)
- Gemini configuration
- Container timeout settings
- Feature flags (audio, debug)
- Validation and warnings for suboptimal configurations

**Key Features:**
- Easy GPU switching (just change `GPU_TYPE = "A10G"`)
- Environment-specific deployments (staging/prod)
- Built-in validation warns about common issues
- All settings documented inline

#### 2. **`modal_app.py`** (ENHANCED)
Updated to use `modal_config.py` for all settings:
- Imports configuration from `modal_config.py`
- Fallback defaults if config file missing
- Uses config for GPU type, timeouts, model settings
- Adds config file to container image
- Better documentation and usage instructions

**Key Improvements:**
- Fully configurable without editing core app logic
- Supports multiple GPU types from config
- Container idle timeout configurable
- All paths use config constants

#### 3. **`MODAL_SETUP_INSTRUCTIONS.md`** (NEW)
Complete step-by-step guide from zero to deployed:
- Installing Modal CLI
- Authentication
- Creating secrets (HuggingFace, Gemini, OpenAI)
- Configuration walkthrough
- Testing locally before deployment
- Deployment (dev and prod modes)
- Testing with webcam and Quest
- Monitoring and cost management
- GPU selection guide with cost estimates
- Troubleshooting common issues

**Target Audience:** Someone who has never used Modal before.

#### 4. **`MODAL_DEPLOYMENT_GUIDE.md`** (EXISTS)
Already present in your repo with comprehensive info on:
- Architecture overview
- Performance expectations
- Quick start commands
- Cost management
- GPU selection details
- Monitoring and troubleshooting

---

## How It Works

### Architecture

```
Quest Camera (1280x960 JPEG)
        ↓
   [WebSocket]
        ↓
Modal GPU Container (A10G 24GB)
├─ SAM3 Segmentation (~200-350ms)
├─ Gemini Scene Understanding (optional)
└─ RLE Mask Encoding
        ↓
   [WebSocket]
        ↓
Quest Client (Overlay Rendering)
```

### Deployment Flow

1. **Local Development:**
   ```bash
   python -m server.main --device cuda  # Local GPU
   ```

2. **Modal Development:**
   ```bash
   modal serve modal_app.py  # Hot reload, free when idle
   ```

3. **Modal Production:**
   ```bash
   modal deploy modal_app.py  # Persistent deployment
   ```

### Compute Offloading

**What Runs on Modal GPU:**
- ✅ SAM3 vision encoding (200ms)
- ✅ SAM3 decoder (28ms × prompts)
- ✅ Gemini scene understanding (optional)
- ✅ Text embedding generation
- ✅ Mask processing and RLE encoding

**What Runs on Quest:**
- RLE decoding
- Texture compositing
- Shader rendering
- Label positioning
- UI rendering

**What Runs on Local Server (if not using Modal):**
- Everything above on local GPU

---

## GPU Requirements

### Recommended: A10G (Default)

**Specs:**
- VRAM: 24GB
- Architecture: Ampere (CUDA 12.x)
- Cost: ~$1.10/hour
- Performance: 3-5 FPS (PyTorch) or 30-45 FPS (TensorRT)

**VRAM Usage:**
- SAM3 model: ~6-8 GB
- Text embeddings: ~0.5 GB
- PyTorch overhead: ~2 GB
- Frame buffers: ~0.5 GB
- **Total:** ~9-11 GB
- **Headroom:** 13-15 GB (2.2x safety margin)

**Why A10G:**
- Perfect size for SAM3 (not underpowered, not overpowered)
- Best price/performance ratio
- Widely available on Modal
- Supports TensorRT optimization

### Alternative GPUs

| GPU | When to Use | Cost |
|-----|-------------|------|
| **L4** | Budget development, slightly slower OK | $0.70/hr |
| **T4** | Absolute minimum, may OOM | $0.50/hr |
| **A100** | Multiple clients, need <100ms latency | $3.70/hr |
| **H100** | Don't use (extreme overkill) | $8.00/hr |

### Changing GPU

Edit `modal_config.py`:
```python
GPU_TYPE = "L4"  # or "A10G", "A100", etc.
```

Then redeploy:
```bash
modal deploy modal_app.py
```

---

## Cost Estimates

### GPU Time (Only When Running)

| Usage Pattern | Hrs/Month | A10G Cost | L4 Cost |
|---------------|-----------|-----------|---------|
| Development (2 hrs/day) | 60 | $66 | $42 |
| Quest testing (1 hr/day) | 30 | $33 | $21 |
| Intermittent (10 hrs/week) | 40 | $44 | $28 |
| 24/7 continuous | 720 | $792 | $504 |

### Cost Optimization

**Current Configuration:**
- `CONTAINER_IDLE_TIMEOUT = 60` seconds
- Container shuts down after 60s of no requests
- Next request triggers cold start (~15-25s)

**Strategies:**
1. **Use `modal serve` during development** (doesn't bill when idle)
2. **Stop when not using:**
   ```bash
   modal app stop socialsense-ar-gpu
   ```
3. **Lower idle timeout** for cost-sensitive use:
   ```python
   CONTAINER_IDLE_TIMEOUT = 30  # 30 seconds
   ```
4. **Use L4 for development** (save $0.40/hr)

### Free Tier

Modal provides **$10-30/month free credits**:
- ~10-30 hours A10G
- ~40-70 hours L4
- Perfect for initial testing

---

## Setup Instructions (Quick Reference)

### 1. Install Modal
```bash
pip install modal
modal setup  # Opens browser for auth
```

### 2. Create Secret
```bash
# Via CLI:
modal secret create socialsense-secrets \
  HF_TOKEN=hf_your_token \
  GEMINI_API_KEY=your_key \
  OPENAI_API_KEY=your_key

# Or via dashboard: modal.com/secrets
```

**Required Keys:**
- `HF_TOKEN`: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - Must request access to [facebook/sam3](https://huggingface.co/facebook/sam3) first
- `GEMINI_API_KEY`: Get from [makersuite.google.com](https://makersuite.google.com/app/apikey) (optional)
- `OPENAI_API_KEY`: Get from [platform.openai.com](https://platform.openai.com/api-keys) (optional)

### 3. Configure
Edit `modal_config.py`:
```python
GPU_TYPE = "A10G"  # Your GPU choice
SAM3_RESOLUTION = 1008  # 1008 (recommended) or 784
AUDIO_ENABLED = False  # True if you want voice agent
```

### 4. Test
```bash
# Test pipeline on Modal GPU
modal run modal_app.py
```

### 5. Deploy
```bash
# Development (hot reload):
modal serve modal_app.py

# Production (persistent):
modal deploy modal_app.py
```

### 6. Test with Webcam
```bash
python tools/webcam_modal_client.py \
  --url wss://your-app-id--socialsense-ar-gpu-fastapi-app.modal.run/ws \
  --show
```

### 7. Connect Quest
Update `SocialSenseClient.cs`:
```csharp
private const string SERVER_URL = "wss://your-app-id--socialsense-ar-gpu-fastapi-app.modal.run/ws";
```

---

## What You Need to Do

### Immediate Actions

1. **Install Modal CLI** (if not already done):
   ```bash
   pip install modal
   ```

2. **Authenticate:**
   ```bash
   modal setup
   ```

3. **Create Secret** with your API keys:
   - HuggingFace token (required)
   - Gemini API key (recommended)
   - OpenAI API key (optional)

4. **Review Configuration:**
   - Open `modal_config.py`
   - Verify `GPU_TYPE = "A10G"` (recommended)
   - Adjust settings if needed

5. **Deploy:**
   ```bash
   modal deploy modal_app.py
   ```

6. **Test:**
   ```bash
   python tools/webcam_modal_client.py --url wss://your-url/ws --show
   ```

7. **Update Quest Client** with Modal URL

### Reference Documents

- **`MODAL_SETUP_INSTRUCTIONS.md`**: Complete step-by-step guide (START HERE)
- **`MODAL_DEPLOYMENT_GUIDE.md`**: Advanced topics, monitoring, optimization
- **`modal_config.py`**: All configuration settings with inline docs
- **`modal_app.py`**: Deployment script (shouldn't need to edit)

---

## Performance Expectations

### Current (PyTorch without TensorRT)

- **Cold Start:** 15-25 seconds (first request after idle)
- **Warm Start:** <2 seconds (cached models)
- **Processing Time:** 200-350ms per frame
- **Throughput:** 3-5 FPS per client
- **Latency:** Network RTT + 200-350ms

**This is perfectly adequate for AR glasses use case.**

### With TensorRT (Optional, Advanced)

- **Processing Time:** 22-35ms per frame
- **Throughput:** 30-45 FPS per client
- **Latency:** Network RTT + 25-35ms

**Requires pre-compiled TensorRT engines** (complex setup, optional).

---

## Troubleshooting Quick Reference

### Common Issues

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` (HuggingFace) | Request access to `facebook/sam3`, verify HF_TOKEN |
| `Out of Memory` | Use A10G (not T4), reduce `SAM3_RESOLUTION` to 784 |
| Slow cold start (>60s) | Normal on first run. Check volume is caching models |
| WebSocket connection failed | Verify URL: `wss://...--fastapi-app.modal.run/ws` |
| Low FPS (<2 FPS) | Check GPU type in logs, verify not using CPU |

### Debug Commands

```bash
# View logs
modal app logs socialsense-ar-gpu --follow

# Check status
modal app list

# Check volume
modal volume list

# Stop deployment
modal app stop socialsense-ar-gpu
```

---

## File Structure

```
SocialSenseAR/
├── modal_app.py              # Modal deployment script (ENHANCED)
├── modal_config.py            # Configuration file (NEW)
├── server/                    # Server package (deployed to Modal)
│   ├── main.py               # Local server entry point
│   ├── config.py             # Server configuration
│   ├── websocket_server.py   # WebSocket server
│   ├── pipeline/
│   │   └── orchestrator.py   # SAM3 pipeline orchestrator
│   ├── vision/
│   │   ├── sam3_segmenter.py # SAM3 inference engine
│   │   └── gemini_scene_understanding.py
│   └── proto/
│       └── socialsense_pb2.py # Protobuf messages
├── tools/
│   └── webcam_modal_client.py # Test client for Modal
├── requirements.txt           # Python dependencies
├── MODAL_SETUP_INSTRUCTIONS.md # Complete setup guide (NEW)
├── MODAL_DEPLOYMENT_GUIDE.md  # Advanced guide (EXISTS)
└── MODAL_PACKAGE_SUMMARY.md   # This file (NEW)
```

---

## Summary

✅ **Fully Modularized:** All configuration in `modal_config.py`, easy to customize
✅ **GPU Optimized:** Defaults to A10G (best value), easy to change
✅ **Cost Aware:** Configurable idle timeout, free tier friendly
✅ **Well Documented:** Complete setup guide + advanced deployment guide
✅ **Production Ready:** Same protobuf protocol, drop-in replacement
✅ **Easy Testing:** Webcam client included, test before Quest deployment

**Next Step:** Follow `MODAL_SETUP_INSTRUCTIONS.md` to deploy your first instance.

---

## Questions?

- **Setup:** See `MODAL_SETUP_INSTRUCTIONS.md`
- **Advanced:** See `MODAL_DEPLOYMENT_GUIDE.md`
- **Configuration:** Edit `modal_config.py`
- **Modal Docs:** [modal.com/docs](https://modal.com/docs)
- **Modal Support:** [modal.com/discord](https://modal.com/discord)

Enjoy your cloud-powered AR segmentation! 🚀
