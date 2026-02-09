# Modal GPU Monitoring Guide

Quick reference for monitoring your Modal GPU usage, costs, and performance.

---

## 🚀 Quick Start

### Interactive Dashboard (Recommended)
```bash
./tools/monitor_modal.sh
```

This launches an interactive menu with all monitoring options.

---

## 📊 Monitoring Commands

### 1. **Real-Time Logs** (Live Updates)
```bash
modal app logs socialsense-ar-gpu --follow
```
Press `Ctrl+C` to stop.

### 2. **Recent Logs** (Last Activity)
```bash
modal app logs socialsense-ar-gpu | tail -50
```

### 3. **App Status** (Running/Stopped)
```bash
modal app list | grep socialsense-ar-gpu
```

### 4. **Detailed Stats** (Web Dashboard)
Open in browser:
```
https://modal.com/apps/aadikrishna04/main/deployed/socialsense-ar-gpu
```

Or via CLI:
```bash
open https://modal.com/apps/aadikrishna04/main/deployed/socialsense-ar-gpu  # macOS
xdg-open https://modal.com/apps/aadikrishna04/main/deployed/socialsense-ar-gpu  # Linux
```

---

## 💰 Cost Monitoring

### Web Dashboard (Most Accurate)
```
https://modal.com/settings/billing
```

Shows:
- **Current month spending**
- **Usage breakdown by app**
- **GPU hours consumed**
- **Cost per function/container**
- **Historical usage trends**

### Quick Estimate
**A10G GPU Cost**: ~$1.10/hour when active

**Calculate your costs:**
```
Daily cost = (hours of active use) × $1.10
Monthly cost = daily cost × 30
```

**Example Usage:**
- 2 hours of development/day = $2.20/day = $66/month
- 1 hour of Quest testing/day = $1.10/day = $33/month
- Container auto-stops after 5 min idle = $0

---

## 🎯 Key Metrics to Watch

### In Modal Web Dashboard

1. **Container Status**
   - Green = Running (costing money)
   - Gray = Stopped (no cost)
   - Yellow = Initializing

2. **Request Count**
   - How many frames processed
   - WebSocket connections
   - API calls

3. **Latency**
   - P50, P95, P99 response times
   - Cold start duration
   - Processing time per frame

4. **GPU Utilization**
   - GPU memory used (out of 24GB on A10G)
   - GPU compute %
   - Time spent on GPU vs CPU

5. **Errors**
   - Failed requests
   - Timeouts
   - Exception logs

---

## 🔍 What to Look For

### ✅ Healthy Metrics
- **Cold start**: 15-25 seconds (first request after idle)
- **Warm start**: <2 seconds
- **Processing time**: 200-350ms per frame (PyTorch)
- **GPU memory**: ~8-12 GB used (SAM3)
- **Error rate**: <1%

### ⚠️ Warning Signs
- **Cold starts >60s**: Volume might not be caching properly
- **Processing >500ms**: GPU overload or inefficient code
- **GPU memory >20GB**: Potential memory leak
- **Error rate >5%**: Connection or pipeline issues
- **Costs >$5/day unexpectedly**: Container not shutting down

---

## 🛑 Cost Control

### Stop the App (Saves Money)
```bash
modal app stop socialsense-ar-gpu
```

No charges after stopping. Restart anytime with:
```bash
modal deploy modal_app.py
```

### Check Current Spending
```bash
# Open billing dashboard
open https://modal.com/settings/billing
```

### Set Budget Alerts
1. Go to https://modal.com/settings/billing
2. Click "Set Budget Alert"
3. Enter threshold (e.g., $50/month)
4. Get email when approaching limit

---

## 📈 Performance Monitoring

### Client-Side FPS (Your Computer)
When running webcam client, it displays:
```
Client: 30.0 fps | Mask: 3.8 fps | RTT avg: 245.2ms p95: 312.5ms | Segs: 5 | Frames: 1800
```

**What these mean:**
- **Client fps**: Frames sent to Modal per second
- **Mask fps**: SAM updates per second (actual segmentation rate)
- **RTT avg**: Average round-trip time (network + processing)
- **RTT p95**: 95th percentile latency (spikes)
- **Segs**: Number of segments detected
- **Frames**: Total frames processed

### Server-Side Logs
```bash
modal app logs socialsense-ar-gpu --follow
```

Look for:
```
WebSocket client connected: <ip>
Processed 120 frames from <ip>
Pipeline ready!
```

---

## 🔧 Troubleshooting

### Problem: High Costs
**Check:**
```bash
modal app list
```
If app shows "Running" but you're not using it:
```bash
modal app stop socialsense-ar-gpu
```

### Problem: Slow Performance
**Check logs for**:
```bash
modal app logs socialsense-ar-gpu | grep -i "error\|warning\|slow"
```

**Common causes:**
- Network latency (can't fix, use local server instead)
- GPU overload (upgrade to A100)
- Model not cached (wait for first request to complete)

### Problem: Connection Refused
**Check if deployed:**
```bash
modal app list | grep socialsense-ar-gpu
```

If not listed:
```bash
modal deploy modal_app.py
```

### Problem: Can't See Webcam Feed
**Make sure `--show` flag is used:**
```bash
python tools/webcam_modal_client.py --url wss://... --show
```

**Check camera permissions** (macOS):
```
System Settings → Privacy & Security → Camera
```

---

## 📱 Quick Reference Card

### Essential Commands
```bash
# Start monitoring dashboard
./tools/monitor_modal.sh

# Live logs
modal app logs socialsense-ar-gpu --follow

# Stop app (save money)
modal app stop socialsense-ar-gpu

# Redeploy (after code changes)
modal deploy modal_app.py

# Check billing
open https://modal.com/settings/billing

# Run webcam test
python tools/webcam_modal_client.py \
  --url wss://aadikrishna04--socialsense-ar-gpu-fastapi-app.modal.run/ws \
  --show
```

### Important URLs
```
App Dashboard: https://modal.com/apps/aadikrishna04/main/deployed/socialsense-ar-gpu
Billing: https://modal.com/settings/billing
WebSocket URL: wss://aadikrishna04--socialsense-ar-gpu-fastapi-app.modal.run/ws
Health Check: https://aadikrishna04--socialsense-ar-gpu-fastapi-app.modal.run/
```

---

## 💡 Pro Tips

1. **Use `modal serve` for development** (free, hot reload):
   ```bash
   modal serve modal_app.py
   ```
   Changes reload automatically without redeployment.

2. **Check logs after cold start** to see initialization time:
   ```bash
   modal app logs socialsense-ar-gpu | grep "Pipeline ready"
   ```

3. **Monitor first hour closely** to ensure auto-shutdown works:
   - Run a test
   - Wait 10 minutes
   - Check if container stopped: `modal app list`

4. **Set a daily reminder** to check billing dashboard (first week)

5. **Take baseline measurements**:
   - First cold start time
   - Typical FPS
   - Average RTT
   - GPU memory usage

   Compare weekly to detect performance degradation.

---

## 🆘 When to Stop Everything

**Stop immediately if:**
- Costs exceed $10/day unexpectedly
- Container won't shut down after 5 min idle
- Errors spike above 20%
- You're not actively developing

**How to stop:**
```bash
modal app stop socialsense-ar-gpu
```

**Then investigate** via web dashboard and logs.

---

## 📞 Getting Help

**Modal Documentation**: https://modal.com/docs
**Modal Discord**: https://modal.com/discord
**Billing Support**: support@modal.com

**Include in support requests:**
- Your workspace ID: `aadikrishna04`
- App name: `socialsense-ar-gpu`
- Recent logs: `modal app logs socialsense-ar-gpu | tail -100`
- Error messages

---

**Last Updated**: 2026-02-03
**Your Workspace**: `aadikrishna04`
**App Name**: `socialsense-ar-gpu`
**GPU**: A10G (24GB VRAM, ~$1.10/hr)
