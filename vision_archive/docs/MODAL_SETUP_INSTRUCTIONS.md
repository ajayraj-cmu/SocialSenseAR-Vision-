# Modal Setup Instructions for SocialSenseAR

**Complete step-by-step guide for setting up and deploying SocialSenseAR with Modal GPU offload.**

---

## Overview

This workspace is now fully packaged and modularized for Modal deployment. All compute-intensive operations (SAM3 segmentation, Gemini scene understanding) will run on Modal's GPU infrastructure.

**What's Been Done:**
- ✅ Modular `modal_app.py` with configuration support
- ✅ `modal_config.py` for easy customization
- ✅ Same protobuf protocol as local server (drop-in replacement)
- ✅ Persistent volume for model caching
- ✅ Multiple GPU options supported (A10G, L4, T4, A100, H100)

---

## Step 1: Install Modal CLI

```bash
# Install Modal
pip install modal

# Verify installation
modal --version
```

---

## Step 2: Authenticate with Modal

```bash
# This will open your browser for authentication
modal setup

# Follow the browser prompts to log in
# Your credentials will be saved locally
```

---

## Step 3: Create Modal Secret

Modal secrets securely store your API keys in the cloud.

### Option A: Via Modal Dashboard (Easiest)

1. Go to [modal.com/secrets](https://modal.com/secrets)
2. Click "New Secret"
3. Name it: `socialsense-secrets`
4. Add these key-value pairs:

   | Key | Value | Required? |
   |-----|-------|-----------|
   | `HF_TOKEN` | Your HuggingFace token | **Yes** |
   | `GEMINI_API_KEY` | Your Google/Gemini API key | Recommended |
   | `GOOGLE_API_KEY` | Alternative to GEMINI_API_KEY | Recommended |
   | `OPENAI_API_KEY` | Your OpenAI API key | Optional |

5. Click "Create"

### Option B: Via CLI

```bash
modal secret create socialsense-secrets \
  HF_TOKEN=hf_your_huggingface_token_here \
  GEMINI_API_KEY=your_gemini_api_key_here \
  OPENAI_API_KEY=sk-your_openai_key_here
```

### Getting API Keys

**HuggingFace Token (Required):**
1. Go to [huggingface.co](https://huggingface.co) and sign up/log in
2. Request access to [facebook/sam3](https://huggingface.co/facebook/sam3) (gated model)
3. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
4. Create a new token with "Read" permissions
5. Copy the token (starts with `hf_`)

**Gemini API Key (Recommended for scene understanding):**
1. Go to [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key

**OpenAI API Key (Optional, for voice agent):**
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Copy the key (starts with `sk-`)

---

## Step 4: Configure Your Deployment

Edit `modal_config.py` to customize your deployment:

```python
# GPU Selection (see GPU Requirements section below)
GPU_TYPE = "A10G"  # Recommended for SAM3

# SAM3 Configuration
SAM3_RESOLUTION = 1008  # 1008 (recommended) or 784
SAM3_PROMPTS_PER_FRAME = 1
SAM3_CONFIDENCE_THRESHOLD = 0.12

# Feature Flags
AUDIO_ENABLED = False  # Set to True if you want voice agent
DEBUG_VIEW = False  # Set to True for development debugging
```

**Don't modify** unless you know what you're doing:
- `PYTHON_VERSION`
- `TORCH_VERSION`
- `CUDA_VERSION`
- Volume/secret names

---

## Step 5: Test Locally (Optional but Recommended)

Before deploying, test that the pipeline works on Modal:

```bash
# From project root
modal run modal_app.py
```

This will:
- Upload your code to Modal
- Spin up an A10G GPU container
- Load SAM3 and process a test frame
- Report processing time

Expected output:
```
Creating test frame...
Test frame: 12345 bytes
Processing on Modal GPU...
Response: 3 segments
Processing time: 234.5ms
✓ Pipeline working!
```

**First run takes 15-25 seconds** (downloading models). Subsequent runs are <2 seconds (cached).

---

## Step 6: Deploy to Modal

### Development Mode (Hot Reload)

For active development with automatic redeployment on file changes:

```bash
modal serve modal_app.py
```

Keep this running in a terminal. It will:
- Watch for file changes
- Automatically redeploy when you save
- Print the WebSocket URL

### Production Mode

For stable production deployment:

```bash
modal deploy modal_app.py
```

This creates a persistent deployment that stays running until you stop it.

**Expected Output:**
```
✓ Created objects.
├── 🔨 Created mount /Users/.../server
├── 🔨 Created mount /Users/.../modal_app.py
├── 🔨 Created function SocialSenseGPU.setup
├── 🔨 Created function SocialSenseGPU.process_frame
├── 🔨 Created web function fastapi_app
└── 🌐 Web endpoint: https://your-app-id--socialsense-ar-gpu-fastapi-app.modal.run

View app at https://modal.com/apps/your-workspace/socialsense-ar-gpu
```

**SAVE THIS URL!** You'll need it to connect your clients.

---

## Step 7: Test the Deployment

### A. Browser Health Check

Visit your deployment URL in a browser:
```
https://your-app-id--socialsense-ar-gpu-fastapi-app.modal.run
```

You should see a status page showing the server is online.

### B. Test with Webcam Client

Use the included test client to verify everything works:

```bash
# From project root
python tools/webcam_modal_client.py \
  --url wss://your-app-id--socialsense-ar-gpu-fastapi-app.modal.run/ws \
  --show
```

This will:
- Capture frames from your webcam
- Send to Modal GPU via WebSocket
- Display segmentation masks overlaid on video
- Show FPS metrics

**Expected Performance:**
- Webcam FPS: 30-60 (your camera rate)
- Processing: 200-350ms per frame (PyTorch)
- Mask updates: 3-5 Hz

Press `q` to quit.

### C. Connect Quest Client

Update your Unity Quest client:

```csharp
// In QuestCameraKit/Unity-QuestVisionKit/Assets/Scripts/SocialSenseClient.cs

private const string SERVER_URL = "wss://your-app-id--socialsense-ar-gpu-fastapi-app.modal.run/ws";
```

Build and deploy to Quest. The protocol is identical to the local server.

---

## Step 8: Monitor Your Deployment

### View Real-Time Logs

```bash
# Stream logs
modal app logs socialsense-ar-gpu --follow

# View recent logs
modal app logs socialsense-ar-gpu
```

### Check Container Status

```bash
# List running apps
modal app list

# View container stats
modal app stats socialsense-ar-gpu

# List containers
modal container list
```

### Stop the Deployment

```bash
# Stop all containers (saves costs when not using)
modal app stop socialsense-ar-gpu

# Delete the deployment entirely
modal app delete socialsense-ar-gpu
```

---

## GPU Requirements & Selection

### Recommended: A10G (Default)

- **VRAM:** 24GB
- **Cost:** ~$1.10/hour
- **Performance:** Excellent for SAM3
- **Recommendation:** Best price/performance ratio

**This is the default in `modal_config.py` and is perfect for your use case.**

### Alternative Options

Edit `modal_config.py` and change `GPU_TYPE`:

| GPU | VRAM | Cost/hr | Use Case |
|-----|------|---------|----------|
| **A10G** | 24GB | $1.10 | **Recommended - best value** |
| L4 | 24GB | $0.70 | Budget option, slightly slower |
| T4 | 16GB | $0.50 | Minimum viable, tight VRAM (may OOM) |
| A100 | 40GB | $3.70 | Overkill but fastest |
| A100-80GB | 80GB | $4.50 | Unnecessary for SAM3 |
| H100 | 80GB | $8.00 | Extreme overkill |

### VRAM Usage Breakdown

Your SAM3 pipeline uses approximately:
- SAM3 model: ~6-8 GB
- Text embeddings: ~0.5 GB
- PyTorch overhead: ~2 GB
- Frame buffers: ~0.5 GB
- **Total: ~9-11 GB**

**A10G's 24GB provides comfortable headroom** (2.2x buffer) for peak usage.

### When to Upgrade

Consider A100 if:
- Running multiple concurrent clients (>5)
- Adding audio pipeline (voice agent)
- Adding emotion detection
- Need lower latency (<100ms)

**For single Quest client with SAM3 only: A10G is perfect.**

---

## Cost Estimates

### GPU Time Charges

Modal charges **only when containers are running**. With the default `container_idle_timeout=60s`, containers shut down after 60 seconds of inactivity.

**Typical Costs (A10G @ $1.10/hr):**

| Usage Pattern | Hours/Month | Cost/Month |
|---------------|-------------|------------|
| Development (2 hrs/day) | ~60 | ~$66 |
| Quest testing (1 hr/day) | ~30 | ~$33 |
| Intermittent use (10 hrs/week) | ~40 | ~$44 |
| Continuous 24/7 | 720 | ~$792 |

### Free Tier

Modal provides **$10-30/month in free credits**, enough for:
- ~10-30 hours of A10G usage
- Perfect for initial testing and development

### Cost Optimization Tips

1. **Use `modal serve` during development** (doesn't count against quota when idle)
2. **Lower `CONTAINER_IDLE_TIMEOUT`** in `modal_config.py` (default 60s is aggressive)
3. **Stop deployment when not using:**
   ```bash
   modal app stop socialsense-ar-gpu
   ```
4. **Use L4 for development** ($0.70/hr instead of $1.10/hr):
   ```python
   GPU_TYPE = "L4"  # in modal_config.py
   ```

---

## Performance Optimization (Optional)

### Current Performance (PyTorch)

- **Processing Time:** 200-350ms per frame
- **Throughput:** 3-5 FPS
- **Latency:** Network RTT + processing time

**This is perfectly adequate for AR glasses use case.**

### TensorRT Optimization (Advanced)

For maximum performance (~30-45 FPS), you can add pre-compiled TensorRT engines:

**Requirements:**
- Must be compiled on A10G GPU (same architecture)
- Resolution-specific (1008 or 784)
- Files: `sam3_vision_1008.engine`, `sam3_topk_decoder_1008.engine`, `sam3_meta_1008.json`

**Setup:**
1. Place engines in project root
2. Modal will auto-detect and use them
3. Expect ~10x speedup (25-35ms vs 200-350ms)

**Note:** TensorRT setup is complex and optional. PyTorch performance is sufficient for most use cases.

---

## Troubleshooting

### Issue: `401 Unauthorized` - HF Model Access

**Symptoms:**
```
401 Client Error: Unauthorized for url: https://huggingface.co/facebook/sam3
```

**Solution:**
1. Go to [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Click "Request Access" and wait for approval (usually instant)
3. Verify your HF_TOKEN in Modal secrets has read permissions
4. Redeploy: `modal deploy modal_app.py`

### Issue: `Out of Memory` (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Use A10G or larger (not T4)
2. Reduce `SAM3_RESOLUTION` to 784 in `modal_config.py`
3. Disable `AUDIO_ENABLED` if you enabled it

### Issue: Slow Cold Start (>60s)

**Cause:** Downloading models from HuggingFace on first run.

**Solution:** This is normal for the first cold start. The volume caches models, so subsequent starts are ~15-25s. If it's slow every time, check that the volume is mounted correctly:

```bash
modal volume list
# Should show: socialsense-cache
```

### Issue: WebSocket Connection Failed

**Symptoms:**
```
WebSocket connection failed: Could not connect
```

**Solutions:**
1. Verify deployment is running: `modal app list`
2. Check URL format:
   - Must start with `wss://` (not `ws://`)
   - Must end with `/ws`
   - Example: `wss://your-app--socialsense-ar-gpu-fastapi-app.modal.run/ws`
3. Check firewall/network allows WebSocket connections

### Issue: Low FPS or High Latency

**Expected Performance:**
- PyTorch: 3-5 FPS, 200-350ms latency
- TensorRT: 30-45 FPS, 25-35ms latency

**If worse than this:**
1. Check network latency: `ping api.modal.com`
2. Verify GPU type: Check logs for "A10G" not "CPU"
3. Monitor container: `modal app logs socialsense-ar-gpu`

### Getting Help

1. **Check logs:** `modal app logs socialsense-ar-gpu`
2. **Modal docs:** [modal.com/docs](https://modal.com/docs)
3. **Modal Discord:** [modal.com/discord](https://modal.com/discord)

---

## Summary

You have now:
- ✅ Installed Modal CLI
- ✅ Authenticated with Modal
- ✅ Created secrets for API keys
- ✅ Configured deployment (GPU, resolution, etc.)
- ✅ Deployed to Modal cloud
- ✅ Tested with webcam client
- ✅ Ready to connect Quest client

**All SAM3 segmentation and Gemini scene understanding now run on Modal's A10G GPU**, freeing your local machine and providing consistent, scalable performance.

**Your WebSocket URL:**
```
wss://your-app-id--socialsense-ar-gpu-fastapi-app.modal.run/ws
```

Use this URL in your Quest Unity client's `SocialSenseClient.cs`.

---

## Next Steps

1. **Connect Quest:** Update Unity client with Modal URL
2. **Test in VR:** Verify mask overlay and tracking
3. **Monitor Costs:** Check Modal dashboard after first day
4. **Optimize:** Adjust `CONTAINER_IDLE_TIMEOUT` based on usage patterns
5. **(Optional) TensorRT:** Compile engines for 10x speedup

Enjoy your cloud-powered AR segmentation! 🚀
