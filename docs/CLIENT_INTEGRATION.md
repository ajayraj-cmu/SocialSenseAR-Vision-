# Client Integration Guide

Complete guide for integrating AR headsets and other clients with the SocialSenseAR server.

## Overview

This guide explains how to integrate your AR headset or client application with the SocialSenseAR compute offloading infrastructure. The server handles all heavy computation (vision processing, object detection, scene understanding) while your client only needs to stream video and render results.

---

## Architecture

```
┌─────────────────────────────┐
│     AR Headset (Client)     │
│                             │
│  ┌───────────────────────┐  │
│  │  Camera Capture       │  │
│  │  (30fps @ 720p)       │  │
│  └───────────┬───────────┘  │
│              ▼              │
│  ┌───────────────────────┐  │
│  │  H.264 Encode         │  │
│  │  (Hardware Accel)     │  │
│  └───────────┬───────────┘  │
│              ▼              │
│  ┌───────────────────────┐  │
│  │  WebSocket Send       │  │
│  └───────────┬───────────┘  │
└──────────────┼──────────────┘
               │
               ▼
        AWS SocialSenseAR Server
               │
               ▼
┌──────────────┼──────────────┐
│  ┌───────────▼───────────┐  │
│  │  WebSocket Receive    │  │
│  └───────────┬───────────┘  │
│              ▼              │
│  ┌───────────────────────┐  │
│  │  H.264 Decode         │  │
│  └───────────┬───────────┘  │
│              ▼              │
│  ┌───────────────────────┐  │
│  │  Display Render       │  │
│  │  (Passthrough Layer)  │  │
│  └───────────────────────┘  │
│     AR Headset (Client)     │
└─────────────────────────────┘
```

---

## Integration Steps

### 1. Obtain API Key

Contact your server administrator or check the server startup logs for an API key:

```
Demo API Key (save this!): sar_1234567890abcdef...
```

### 2. Install Client SDK

#### Python

```bash
pip install websockets numpy opencv-python
```

#### Unity (C#)

Use Unity's WebSocket package:
```bash
# In Unity Package Manager
Add package: com.unity.webrtc
```

#### Native (C++)

Use libwebsockets:
```bash
# Linux
sudo apt-get install libwebsockets-dev

# macOS
brew install libwebsockets
```

### 3. Implement Client

#### Python Example

See `client_sdk/python_example.py` for a complete reference implementation.

```python
import asyncio
import websockets
import numpy as np
import json
import base64
import time

class ARClient:
    def __init__(self, server_url, api_key):
        self.server_url = server_url
        self.api_key = api_key
        self.session_id = f"ar_headset_{int(time.time())}"

    async def connect(self):
        ws_url = f"{self.server_url}/ws/{self.session_id}"
        headers = {"X-API-Key": self.api_key}

        self.ws = await websockets.connect(ws_url, extra_headers=headers)
        print("Connected to server!")

    async def send_frame(self, frame):
        # Encode frame (use H.264 in production)
        import io
        buffer = io.BytesIO()
        np.save(buffer, frame, allow_pickle=False)

        # Send via WebSocket
        await self.ws.send(json.dumps({
            "type": "frame",
            "data": base64.b64encode(buffer.getvalue()).decode(),
            "timestamp": time.time()
        }))

    async def receive_frame(self):
        msg = await self.ws.recv()
        data = json.loads(msg)

        if data["type"] == "frame":
            # Decode and render (decode H.264 in production)
            print(f"Received frame, latency: {data['latency_ms']}ms")
            return data

# Usage
async def main():
    client = ARClient("ws://SERVER_IP:8000", "YOUR_API_KEY")
    await client.connect()

    # Main loop
    while True:
        frame = capture_camera_frame()  # Your camera capture
        await client.send_frame(frame)

        processed = await client.receive_frame()
        render_to_display(processed)  # Your render function

asyncio.run(main())
```

#### Unity Example (C#)

```csharp
using UnityEngine;
using System;
using System.Collections;
using NativeWebSocket;

public class SocialSenseARClient : MonoBehaviour
{
    private WebSocket websocket;
    private string serverUrl = "ws://SERVER_IP:8000";
    private string apiKey = "YOUR_API_KEY";
    private string sessionId;

    async void Start()
    {
        sessionId = $"unity_{DateTimeOffset.Now.ToUnixTimeSeconds()}";

        websocket = new WebSocket($"{serverUrl}/ws/{sessionId}");

        // Add API key header
        websocket.SetRequestHeader("X-API-Key", apiKey);

        websocket.OnMessage += OnMessageReceived;

        await websocket.Connect();
        Debug.Log("Connected to SocialSenseAR server");

        StartCoroutine(StreamFrames());
    }

    void OnMessageReceived(byte[] data)
    {
        string message = System.Text.Encoding.UTF8.GetString(data);
        var json = JsonUtility.FromJson<ServerMessage>(message);

        if (json.type == "frame")
        {
            // Decode and render processed frame
            Debug.Log($"Received frame, latency: {json.latency_ms}ms");
            RenderProcessedFrame(json.data);
        }
    }

    IEnumerator StreamFrames()
    {
        while (true)
        {
            // Capture frame from camera
            Texture2D frame = CaptureCamera();

            // Encode and send
            string encodedFrame = EncodeFrame(frame);
            var message = new ClientMessage
            {
                type = "frame",
                data = encodedFrame,
                timestamp = DateTimeOffset.Now.ToUnixTimeSeconds()
            };

            await websocket.SendText(JsonUtility.ToJson(message));

            yield return new WaitForSeconds(1f / 30f);  // 30 FPS
        }
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
        websocket?.DispatchMessageQueue();
        #endif
    }

    private async void OnApplicationQuit()
    {
        await websocket.Close();
    }
}

[Serializable]
public class ClientMessage
{
    public string type;
    public string data;
    public double timestamp;
}

[Serializable]
public class ServerMessage
{
    public string type;
    public string data;
    public double timestamp;
    public float latency_ms;
}
```

---

## Performance Optimization

### 1. Video Encoding

**Use Hardware Acceleration:**

- **Android:** MediaCodec H.264 encoder
- **iOS:** VideoToolbox H.264 encoder
- **Unity:** Use native plugins for hardware encoding

**Recommended Settings:**
```
Resolution: 1280x720 (720p)
FPS: 30
Bitrate: 2-5 Mbps
Codec: H.264
Preset: ultrafast (for low latency)
Profile: baseline
```

### 2. Network Optimization

**WebSocket Settings:**
```python
# Enable TCP_NODELAY for low latency
websocket.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# Set send/receive buffer sizes
websocket.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
websocket.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
```

**Heartbeat:**
- Send ping every 30 seconds to keep connection alive
- Monitor latency and adjust quality if needed

### 3. Latency Reduction

**Client-Side:**
- Use hardware encoding/decoding
- Minimize frame buffering (max 1-2 frames)
- Process frames immediately upon receipt
- Use zero-copy rendering where possible

**Monitoring:**
```python
# Track round-trip latency
send_time = time.time()
# ... send frame ...
# ... receive processed frame ...
rtt_latency = (time.time() - send_time) * 1000  # ms

print(f"Round-trip latency: {rtt_latency}ms")
```

---

## Error Handling

### Connection Errors

```python
async def connect_with_retry(client, max_retries=5):
    for attempt in range(max_retries):
        try:
            await client.connect()
            return
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Connection failed, retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    raise Exception("Failed to connect after retries")
```

### Dropped Frames

```python
# Detect dropped frames
expected_frame_id = 0

def on_frame_received(frame_id):
    global expected_frame_id

    if frame_id != expected_frame_id:
        dropped = frame_id - expected_frame_id
        print(f"Warning: {dropped} frames dropped")

    expected_frame_id = frame_id + 1
```

### Timeout Handling

```python
# Set timeout for receiving frames
try:
    frame = await asyncio.wait_for(
        client.receive_frame(),
        timeout=1.0  # 1 second timeout
    )
except asyncio.TimeoutError:
    print("Frame receive timeout - connection may be slow")
```

---

## Testing

### Local Testing

1. **Start server locally:**
```bash
cd /path/to/Meta X SocialSense
python -m uvicorn src.server.streaming_server:app --host 0.0.0.0 --port 8000
```

2. **Run test client:**
```bash
python client_sdk/python_example.py --server-url ws://localhost:8000 --api-key YOUR_API_KEY
```

### Network Testing

Test latency on your network:

```python
import asyncio
import websockets
import time

async def test_latency(server_url, api_key):
    latencies = []

    for i in range(100):
        start = time.time()

        # Send ping
        # ... (send frame or ping message) ...

        # Receive pong
        # ... (receive response) ...

        latency = (time.time() - start) * 1000
        latencies.append(latency)

    print(f"Average latency: {sum(latencies) / len(latencies):.2f}ms")
    print(f"Min: {min(latencies):.2f}ms, Max: {max(latencies):.2f}ms")
```

---

## Platform-Specific Notes

### Meta Quest 3

**Camera Access:**
```csharp
// Unity - access passthrough camera
OVRPassthroughLayer passthroughLayer = gameObject.AddComponent<OVRPassthroughLayer>();
```

**Performance:**
- Quest 3 has hardware H.264 encoder/decoder
- Target 72 FPS for smooth AR experience
- Use foveated rendering if needed

### HoloLens 2

**Camera Access:**
```csharp
// Access locatable camera
var mediaCapture = new MediaCapture();
await mediaCapture.InitializeAsync();
```

**Constraints:**
- Limited network bandwidth over WiFi
- Consider reducing resolution to 540p
- Use adaptive quality based on network conditions

### Apple Vision Pro

**Camera Access:**
```swift
// ARKit camera access
let session = ARSession()
session.run(ARWorldTrackingConfiguration())
```

**Performance:**
- Excellent hardware encoding (M2 chip)
- Can handle 4K if needed
- Very low latency over WiFi 6E

---

## Deployment Checklist

- [ ] Obtain production API key
- [ ] Configure server URL (use HTTPS/WSS in production)
- [ ] Implement hardware video encoding
- [ ] Add error handling and reconnection logic
- [ ] Test latency on target network
- [ ] Implement heartbeat mechanism
- [ ] Add telemetry/logging
- [ ] Test with multiple concurrent users
- [ ] Verify battery life impact
- [ ] Measure bandwidth usage

---

## Troubleshooting

### High Latency

**Symptoms:** Latency > 100ms consistently

**Solutions:**
1. Check network connection (WiFi signal strength)
2. Reduce video resolution (try 960x540)
3. Lower bitrate (2 Mbps instead of 5 Mbps)
4. Ensure server has GPU available
5. Check server `/api/v1/status` for GPU utilization

### Connection Drops

**Symptoms:** WebSocket disconnects frequently

**Solutions:**
1. Implement heartbeat (ping every 30s)
2. Add automatic reconnection
3. Check firewall rules (ports 8000, 3478, 49152-49200)
4. Increase WebSocket timeout on server

### Poor Video Quality

**Symptoms:** Blurry or pixelated output

**Solutions:**
1. Increase bitrate (up to 8 Mbps)
2. Increase resolution (up to 1080p if bandwidth allows)
3. Adjust H.264 CRF value (lower = better quality)
4. Check if adaptive quality is downgrading

---

## Support

For integration assistance:
- Review example client: `client_sdk/python_example.py`
- Check API reference: `docs/API_REFERENCE.md`
- Consult server logs for detailed error messages
- Monitor metrics via `/api/v1/metrics` endpoint
