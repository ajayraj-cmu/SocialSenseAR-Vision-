# Modal GPU Deployment Guide

## Overview

This guide covers deploying SocialSenseAR's SAM3 segmentation pipeline to Modal's GPU infrastructure.

**Architecture:**
- **Modal A10G GPU** (24GB VRAM): Runs SAM3 + all ML inference
- **WebSocket endpoint**: Binary protobuf protocol (same as local server)
- **Persistent Volume**: HuggingFace + Torch model cache (~2-5GB)
- **API Keys**: Stored in Modal secrets (HF_TOKEN, OPENAI_API_KEY, GEMINI_API_KEY)

## Performance Expectations

### Current Setup (PyTorch Fallback)
- **Cold start**: 15-25 seconds (download + load models)
- **Warm start**: <2 seconds (cached models)
- **Inference speed**: ~3-5 FPS per client
- **Latency**: Network RTT + 200-350ms processing

### With TensorRT Optimization (Future)
- **Inference speed**: ~30-45 FPS per client
- **Latency**: Network RTT + 22-33ms processing
- **Requirements**: Pre-compiled `.engine` files for A10G GPU

## Prerequisites

✅ **Completed during setup:**
- Modal CLI installed (`modal --version`)
- Modal authenticated (`modal profile current`)
- Modal secrets created (`socialsense-secrets`)
- Modal volume created (`socialsense-cache`)

## Quick Start

### 1. Deploy to Modal

```bash
# Deploy (production):
modal deploy modal_app.py

# Or serve (development with hot reload):
modal serve modal_app.py
```

You'll see output like:
```
✓ Created objects.
├── 🔨 Created mount /Users/ajayraj/Meta x SocialSense (New)/server
├── 🔨 Created SocialSenseGPU => so-abc123...
└── 🌐 Created web function fastapi_app => https://your-workspace--socialsense-ar-gpu-fastapi-app.modal.run
```

**Copy the WebSocket URL**: `wss://your-workspace--socialsense-ar-gpu-fastapi-app.modal.run/ws`

### 2. Test with Webcam

```bash
# With overlay visualization:
python tools/webcam_modal_client.py \
    --url wss://your-workspace--socialsense-ar-gpu-fastapi-app.modal.run/ws \
    --show

# High FPS test (no overlay):
python tools/webcam_modal_client.py \
    --url wss://your-workspace--socialsense-ar-gpu-fastapi-app.modal.run/ws \
    --fps 60
```

### 3. Test with Quest

Update Unity client's `SocialSenseClient.cs`:

```csharp
// Replace localhost with Modal URL:
private const string SERVER_URL = "wss://your-workspace--socialsense-ar-gpu-fastapi-app.modal.run/ws";
```

## Cost Management

### GPU Pricing (as of Feb 2026)
- **A10G (24GB)**: ~$1.10/hour when running
- **Container idle timeout**: 5 minutes (keeps warm between requests)
- **Auto-shutdown**: After 5 minutes of no activity

### Cost Optimization Tips

1. **Use `modal serve` for development** (free, hot reload)
2. **Deploy for production** only when needed
3. **Monitor usage**: `modal app logs socialsense-ar-gpu`
4. **Set shorter idle timeout** if cost is concern (edit `container_idle_timeout` in `modal_app.py`)

### Estimated Costs
- **Active development** (2 hours/day): ~$2.20/day = $66/month
- **Quest testing** (1 hour/day): ~$1.10/day = $33/month
- **Production** (24/7, high load): ~$26/day = $792/month ⚠️

**Recommendation**: Use `modal serve` during development, deploy only for Quest testing sessions.

## GPU Selection

### Current: A10G (24GB VRAM)
- **Best for**: Development, testing, moderate load
- **Performance**: 3-5 FPS (PyTorch), 30-45 FPS (TRT)
- **Cost**: ~$1.10/hour
- **Max clients**: ~5-10 concurrent (PyTorch), ~20-30 (TRT)

### Upgrade to A100 (40GB/80GB)
Edit `modal_app.py`:
```python
@app.cls(
    gpu="A100",  # or "A100-80GB"
    # ... rest of config
)
```

**When to upgrade:**
- Need higher throughput (>5 FPS per client)
- Multiple concurrent clients
- Running additional models (emotion, MediaPipe)
- Latency spikes on A10G

## Performance Tuning

### 1. Increase Concurrent Clients

```python
@app.cls(
    allow_concurrent_inputs=20,  # default: 10
    # ... rest of config
)
```

### 2. Adjust Idle Timeout

```python
@app.cls(
    container_idle_timeout=60,  # 1 minute (lower cost)
    # or
    container_idle_timeout=600,  # 10 minutes (faster response)
    # ... rest of config
)
```

### 3. Enable TensorRT (Advanced)

**Prerequisites:**
- Compile `.engine` files on A10G GPU (same architecture as deployment)
- Add engines to Modal volume or image

**Steps:**
1. Export ONNX models: `python -m server.vision.sam3_export`
2. Compile with `trtexec` on A10G instance
3. Add to Modal image or volume
4. Restart deployment

**Expected improvement**: 6-10x faster (22-33ms vs 200-350ms)

## Monitoring

### View Logs
```bash
# Real-time logs:
modal app logs socialsense-ar-gpu --follow

# Recent logs:
modal app logs socialsense-ar-gpu
```

### Check Status
```bash
# List running apps:
modal app list

# Volume usage:
modal volume ls socialsense-cache
```

### Metrics
```bash
# Container stats:
modal stats socialsense-ar-gpu
```

## Troubleshooting

### Cold Start Takes Too Long (>60s)

**Cause**: Downloading 2GB+ models from HuggingFace
**Solution**: Models should cache after first run. If persistent, check volume mount.

### "Connection Refused" Error

**Cause**: App not deployed or URL incorrect
**Solution**:
1. Check deployment: `modal app list`
2. Verify URL includes `/ws` path
3. Use `wss://` not `ws://` for Modal URLs

### Low FPS (<2 FPS)

**Cause**: PyTorch fallback (no TRT engines)
**Expected**: 3-5 FPS without TRT, 30-45 FPS with TRT
**Solution**: Compile TRT engines (see "Enable TensorRT" section)

### High Latency (>500ms RTT)

**Causes**:
1. Geographic distance to Modal datacenter
2. Network congestion
3. GPU overload (too many concurrent clients)

**Solutions**:
1. Check network: `ping api.modal.com`
2. Reduce `allow_concurrent_inputs`
3. Upgrade to A100

### Out of Memory (OOM)

**Cause**: GPU VRAM exhausted
**Solution**:
1. Reduce `sam3_prompts_per_frame` in config
2. Disable audio pipeline (already disabled)
3. Upgrade to A100-80GB

## API Reference

### WebSocket Endpoint: `/ws`

**Protocol**: Binary protobuf (same as local server)

**Client → Server**: `ClientMessage`
```protobuf
message ClientMessage {
  uint64 frame_id = 1;
  double timestamp_ms = 2;
  oneof payload {
    FramePayload frame = 3;      // JPEG frame
    AudioPayload audio = 4;      // (not yet supported)
    ControlPayload control = 5;  // Commands (blur, etc.)
  }
}
```

**Server → Client**: `ServerMessage`
```protobuf
message ServerMessage {
  uint64 frame_id = 1;
  uint64 masks_frame_id = 5;
  bool masks_updated = 6;
  repeated SceneSegment segments = 10;
  // ... see server/proto/socialsense.proto
}
```

### Health Check: `/`

```bash
curl https://your-app.modal.run/
# Returns HTML status page
```

## Development Workflow

### Recommended Flow

1. **Local Development**:
   ```bash
   # Run server locally for fast iteration:
   python -u -m server.main --device cuda
   python -m server.test_client
   ```

2. **Modal Testing**:
   ```bash
   # Serve with hot reload (free):
   modal serve modal_app.py

   # Test:
   python tools/webcam_modal_client.py --url wss://... --show
   ```

3. **Quest Integration**:
   ```bash
   # Deploy (production):
   modal deploy modal_app.py

   # Update Quest client with URL
   # Build and test
   ```

### Making Changes

**To server code** (`server/` directory):
1. Edit files locally
2. Changes auto-sync to Modal with `modal serve`
3. For `modal deploy`, redeploy to update

**To Modal app** (`modal_app.py`):
1. Edit `modal_app.py`
2. Save
3. Modal auto-reloads (serve mode) or redeploy

## Next Steps

### Immediate (Day 1)
- [x] Deploy to Modal
- [x] Test with webcam
- [ ] Test with Quest
- [ ] Verify mask quality
- [ ] Measure latency

### Short-term (Week 1)
- [ ] Optimize network RTT (check datacenter location)
- [ ] Add voice commands support
- [ ] Monitor costs and adjust idle timeout
- [ ] Profile performance bottlenecks

### Long-term (Month 1)
- [ ] Compile TensorRT engines for A10G
- [ ] Add multi-user support
- [ ] Implement audio pipeline
- [ ] Add usage analytics
- [ ] Set up CI/CD for automated deployments

## Support

**Modal Docs**: https://modal.com/docs
**SocialSenseAR Issues**: [Your repo issues page]
**Modal Community**: https://modal.com/discord

## Files Reference

- `modal_app.py` - Modal deployment configuration
- `tools/webcam_modal_client.py` - Test client for Modal
- `server/` - Server package (deployed to Modal)
- `MODAL_OFFLOAD.md` - Original implementation spec
- `MODAL_DEPLOYMENT_GUIDE.md` - This file

---

**Status**: ✅ Ready for deployment
**Last Updated**: 2026-02-03
