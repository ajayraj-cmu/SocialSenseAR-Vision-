# SocialSenseAR API Reference

Complete API documentation for integrating with the SocialSenseAR streaming server.

## Table of Contents

- [Authentication](#authentication)
- [WebSocket API](#websocket-api)
- [REST API](#rest-api)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)

---

## Authentication

All API requests require authentication using an API key.

### API Key

Include your API key in the request header:

```
X-API-Key: sar_your_api_key_here
```

### Getting an API Key

API keys are generated when the server starts. Check the server logs for the demo API key, or contact your administrator to create a new one.

---

## WebSocket API

The WebSocket API provides real-time bidirectional video streaming.

### Connection

**Endpoint:** `ws://<SERVER_HOST>:8000/ws/{session_id}`

**Headers:**
```
X-API-Key: your_api_key
```

**Parameters:**
- `session_id` (string): Unique identifier for your session

### Message Format

All WebSocket messages are JSON objects with a `type` field.

#### Client → Server Messages

##### 1. Video Frame

Send a video frame for processing:

```json
{
  "type": "frame",
  "data": "<base64_encoded_frame>",
  "timestamp": 1234567890.123
}
```

**Fields:**
- `type`: Must be `"frame"`
- `data`: Base64-encoded frame data (or H.264 encoded video)
- `timestamp`: Client-side timestamp (Unix time)

##### 2. Command

Send a text or voice command:

```json
{
  "type": "command",
  "command": "blur the background"
}
```

**Fields:**
- `type`: Must be `"command"`
- `command`: Command text

##### 3. Heartbeat

Keep the connection alive:

```json
{
  "type": "ping"
}
```

#### Server → Client Messages

##### 1. Processed Frame

Receive a processed video frame:

```json
{
  "type": "frame",
  "data": "<base64_encoded_processed_frame>",
  "timestamp": 1234567890.456,
  "latency_ms": 45.2
}
```

**Fields:**
- `type`: `"frame"`
- `data`: Base64-encoded processed frame
- `timestamp`: Server-side timestamp
- `latency_ms`: Processing latency in milliseconds

##### 2. Command Acknowledgment

Confirmation that command was received:

```json
{
  "type": "command_ack",
  "command": "blur the background"
}
```

##### 3. Heartbeat Response

Response to ping:

```json
{
  "type": "pong"
}
```

### Example Connection (Python)

```python
import asyncio
import websockets
import json

async def connect():
    uri = "ws://localhost:8000/ws/my_session"
    headers = {"X-API-Key": "sar_your_api_key"}

    async with websockets.connect(uri, extra_headers=headers) as ws:
        # Send a frame
        await ws.send(json.dumps({
            "type": "frame",
            "data": "<encoded_frame>",
            "timestamp": time.time()
        }))

        # Receive processed frame
        response = await ws.recv()
        data = json.loads(response)
        print(f"Latency: {data['latency_ms']}ms")

asyncio.run(connect())
```

---

## REST API

The REST API provides session management and monitoring capabilities.

### Base URL

```
http://<SERVER_HOST>:8000/api/v1
```

### Endpoints

#### 1. Create Session

Start a new streaming session.

**Request:**
```
POST /api/v1/session/start
```

**Headers:**
```
X-API-Key: your_api_key
Content-Type: application/json
```

**Body:**
```json
{
  "session_id": "unique_session_id",
  "metadata": {
    "device_type": "AR_headset",
    "client_version": "1.0.0"
  }
}
```

**Response:** `200 OK`
```json
{
  "session_id": "unique_session_id",
  "status": "active",
  "created_at": 1234567890.123,
  "last_heartbeat": 1234567890.123,
  "metadata": {
    "device_type": "AR_headset",
    "client_version": "1.0.0"
  }
}
```

#### 2. Get Session

Retrieve session information.

**Request:**
```
GET /api/v1/session/{session_id}
```

**Headers:**
```
X-API-Key: your_api_key
```

**Response:** `200 OK`
```json
{
  "session_id": "unique_session_id",
  "status": "active",
  "created_at": 1234567890.123,
  "last_heartbeat": 1234567890.456,
  "metadata": {}
}
```

**Error:** `404 Not Found`
```json
{
  "detail": "Session not found"
}
```

#### 3. Stop Session

Terminate a session.

**Request:**
```
DELETE /api/v1/session/{session_id}/stop
```

**Headers:**
```
X-API-Key: your_api_key
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Session unique_session_id stopped"
}
```

#### 4. Send Text Command

Send a text command to a session.

**Request:**
```
POST /api/v1/command/text
```

**Headers:**
```
X-API-Key: your_api_key
Content-Type: application/json
```

**Body:**
```json
{
  "session_id": "unique_session_id",
  "command": "make the screen brighter"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Command queued: make the screen brighter",
  "session_id": "unique_session_id"
}
```

#### 5. Get System Status

Get server status and metrics.

**Request:**
```
GET /api/v1/status
```

**Headers:**
```
X-API-Key: your_api_key
```

**Response:** `200 OK`
```json
{
  "status": "running",
  "active_sessions": 5,
  "gpu_available": true,
  "gpu_utilization": 67.5,
  "uptime_seconds": 12345.67
}
```

#### 6. Get Stream Metrics

Get detailed metrics for all active streams.

**Request:**
```
GET /api/v1/metrics
```

**Headers:**
```
X-API-Key: your_api_key
```

**Response:** `200 OK`
```json
{
  "streams": {
    "session_1": {
      "session_id": "session_1",
      "frames_received": 1000,
      "frames_processed": 998,
      "frames_sent": 998,
      "frames_dropped": 2,
      "avg_latency_ms": 42.5,
      "fps": 29.8,
      "uptime_seconds": 33.5
    }
  },
  "timestamp": 1234567890.123
}
```

#### 7. Health Check

Simple health check (no authentication required).

**Request:**
```
GET /api/v1/health
```

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "gpu_available": true
}
```

---

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request format
- `401 Unauthorized`: Invalid or missing API key
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

#### Invalid API Key

```json
{
  "detail": "Invalid API key"
}
```

**Solution:** Check that your API key is correct and included in the `X-API-Key` header.

#### Session Not Found

```json
{
  "detail": "Session not found"
}
```

**Solution:** Ensure the session ID is correct and the session hasn't expired.

#### Session Already Exists

```json
{
  "detail": "Session already exists"
}
```

**Solution:** Use a different session ID or stop the existing session first.

---

## Rate Limits

### Default Limits

- **API Requests:** 60 requests per minute per API key
- **WebSocket Messages:** No hard limit, but excessive traffic may be throttled
- **Concurrent Sessions:** 10 sessions per API key (configurable)

### Rate Limit Headers

Rate limit information is included in response headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1234567920
```

### Exceeding Rate Limits

When rate limit is exceeded, you'll receive:

**Status:** `429 Too Many Requests`

**Response:**
```json
{
  "detail": "Rate limit exceeded. Try again in 30 seconds."
}
```

---

## Best Practices

### 1. Connection Management

- Reuse WebSocket connections for the entire session
- Implement automatic reconnection with exponential backoff
- Send heartbeat messages every 30 seconds

### 2. Frame Streaming

- Send frames at consistent intervals (e.g., 30 FPS)
- Include accurate timestamps for latency calculation
- Monitor latency metrics and adjust quality if needed

### 3. Error Handling

- Implement retry logic for transient errors
- Log all API errors for debugging
- Handle disconnections gracefully

### 4. Performance

- Use H.264 encoding for video frames (reduces bandwidth)
- Batch commands when possible
- Monitor GPU utilization via `/api/v1/status`

### 5. Security

- Keep API keys secure and never commit to version control
- Use HTTPS/WSS in production
- Rotate API keys regularly

---

## Support

For issues or questions:
- Check server logs for detailed error messages
- Review the client SDK examples in `client_sdk/`
- Consult the integration guide: `docs/CLIENT_INTEGRATION.md`
