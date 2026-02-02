# SocialSenseAR Codebase Guide

AR segmentation system: Quest 3 captures camera frames → Python server runs SAM3 → returns RLE masks → Unity renders overlay on passthrough.

> For SAM optimization work, read `OPTIMIZATIONS.md` first.

## How to Test
```bash
# Server (IMPORTANT: -u for unbuffered output, takes 15-25s to start)
python -u -m server.main --device cuda
# Client
python -u -m server.test_client --show
```

---

## Python Server (`server/`)

### Architecture
```
WebSocket (port 8765) receives JPEG frames from Quest
  → orchestrator.process_frame() returns cached result in <0.1ms
  → Background thread: SAM loop (continuous, ~3.8 fps)
      preprocess → TRT vision encode (200ms) → TRT decode (28ms/prompt × 2) → RLE encode
  → Protobuf response cached until segments change (by object identity)
```

### Key Files
| File | Purpose |
|------|---------|
| `server/main.py` | Entry point, CLI args |
| `server/config.py` | `ServerConfig` dataclass — all knobs |
| `server/websocket_server.py` | Async WS server, frame dispatch, response caching, debug view |
| `server/pipeline/orchestrator.py` | SAM background loop, tracking, RLE encoding, Gemini labeling |
| `server/vision/sam3_segmenter.py` | SAM3 TRT inference (vision + decoder engines) |
| `server/vision/sam3_trt.py` | TensorRT engine wrapper |
| `server/vision/segment_data.py` | `SegmentData` class shared across pipeline |
| `server/encoding/rle.py` | Vectorized RLE encode/decode, bbox, centroid helpers |
| `server/proto/socialsense_pb2.py` | Generated protobuf messages |
| `server/test_client.py` | Webcam test client with overlay + FPS counter |

### Orchestrator Quirks
- **Pre-decodes JPEG on WebSocket thread** (~2ms, overlaps with GPU work)
- **Quest sends vertically flipped frames** — `cv2.flip(frame, 0)` corrects in `_decode_jpeg()`
- **RLE at 75% resolution**: `mask_small = cv2.resize(mask, (fw*0.75, fh*0.75))` + morphology + blur
- **Tracking**: Cost-matrix matching (position + area + label bonus), IoU > 0.65 keeps old mask (stability), 1s TTL
- **Label persistence**: 2 consecutive Gemini votes required before label change accepted
- **`masks_frame_id`**: Frame ID when SAM's vision encoder last ran — Quest uses this to look up capture-time head pose
- **`masks_updated`**: True only when vision encoder produced fresh features (not when returning cached decoder result)

### Coordinate System (Server)
- **OpenCV convention**: origin top-left, Y-down
- **All protobuf coords normalized [0,1]**: bbox, centroid — relative to frame dimensions
- **center_x/center_y**: Centroid from `cv2.moments()`, normalized. Computed on the flip-corrected frame

### Segment Data Flow
```python
# In sam3_segmenter.py:
mask_u8 = (decoder_output > 0).astype(np.uint8) * 255
cx, cy = moments_centroid(mask_u8)
seg.center_x = cx / frame_width   # normalized [0,1]
seg.center_y = cy / frame_height  # normalized [0,1], OpenCV convention (0=top)

# In orchestrator.py (RLE encoding):
rle_w, rle_h = int(fw * 0.75), int(fh * 0.75)
mask_small = cv2.resize(mask, (rle_w, rle_h))
seg.rle_mask = encode_rle((mask_smooth > 128).astype(np.uint8))
```

### RLE Format
Binary uint16 little-endian run lengths. Always starts with background (0) run. Typical mask: 200-2000 bytes.

### Windows Gotchas
- **`python -u`** always — tqdm + logging buffer otherwise (appears frozen at 27%)
- **Process cleanup**: `wmic process call terminate` not `taskkill /F` (CUDA processes)
- **Verify GPU freed**: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
- **Debug log**: `_dbg()` in sam3_segmenter.py writes to `~/sam3_debug.log` (unbuffered, bypasses Python logging)
- **Startup check**: `netstat -ano | findstr 8765`

---

## Unity Client (`QuestCameraKit/Unity-QuestVisionKit/Assets/Scripts/`)

### Architecture
```
PassthroughCameraAccess (Meta.XR)
  → SocialSenseClient: GPU blit → async readback → JPEG → WebSocket → Server
  → Server response: protobuf with RLE masks + labels
  → OverlayRenderer: decode RLE → composite RGBA32 texture → sphere + shader
  → SegmentOverlay shader: pinhole projection with camera intrinsics per fragment
  → 3D TextMesh labels: inverse pinhole → billboard at segment centroids
```

### Key Files
| File | Purpose |
|------|---------|
| `SocialSenseClient.cs` | Camera capture, WebSocket client, pose storage |
| `OverlayRenderer.cs` | RLE decode, texture composite, sphere, labels, calibration |
| `Proto/SocialsenseMessages.cs` | Hand-written protobuf C# classes |
| `Shaders/SegmentOverlay.shader` | Intrinsics-based pinhole projection, stereo, edge fade |

### SocialSenseClient
- **Capture pipeline (Quest)**: Camera texture → `Graphics.DrawTexture` to scaled RT → `Graphics.Blit` with Y-flip `(1,-1)` → `AsyncGPUReadback` (max 2 in-flight) → JPEG encode → protobuf send
- **Y-flip on send**: `Graphics.Blit(_scaledRT, _flippedRT, new Vector2(1, -1), new Vector2(0, 1))`
- **Pose storage**: Stores `centerEyeAnchor.rotation` per `frame_id` in dict. On response, looks up rotation via `masks_frame_id` for world-locked reprojection
- **Reconnection**: 3s retry, guards against overlapping connection attempts with `_connecting` flag
- **NativeWebSocket**: Requires `_websocket.DispatchMessageQueue()` every Update

### OverlayRenderer

**Sphere**: Unity primitive sphere (default 1m radius) centered on `centerEyeAnchor.position`. Renders at queue "Overlay", ZTest Always, alpha blend.

**RLE Decoding — Y-FLIP**:
```csharp
int unityRow = h - 1 - rleRow;  // OpenCV top=0 → Unity bottom=0
_pixelBuffer[unityRow * w + col] = color;
```

**Camera Intrinsics Pipeline**:
1. Read `cameraAccess.Intrinsics` (focal length, principal point, sensor resolution, LensOffset)
2. Camera texture is 1280x960 center-crop of full 1280x1280 sensor
3. Adjust principal point: `adjustedCx = PP.x - (sensorW - actualW) / 2`
4. LensOffset rotation includes 180deg X flip (OpenCV→Unity) + physical tilt
5. Build `cameraProjectionRot = Quaternion.LookRotation(camForward, camUp)` from lens axes
6. Rotation matrix: `inverse(headRotation * cameraProjectionRot)` — sent to shader

**Stereo: Both Eyes from Left Camera**:
```csharp
// MUST use headRotation (capture-time), not centerEyeAnchor.rotation (current)
// Using current rotation causes angle-dependent drift between projection matrix and camera offset
Vector3 leftCamWorldPos = centerEyeAnchor.position + headRotation * lensOffset.position;
```
Both eyes project from `_LeftCameraWorldPos` → identical UVs → zero disparity (correct for monocular camera). Right eye gets optional pixel offset (`rightEyeOffset`, default 36px) for residual calibration.

**Labels — Inverse Pinhole**:
```csharp
float px = seg.CenterX * actualW;
float py = (1f - seg.CenterY) * actualH;  // FLIP Y: OpenCV (0=top) → shader convention
float dx = (px - PP.x) / focal.x;
float dy = (py - PP.y) / focal.y;
Vector3 localDir = new Vector3(dx, dy, 1f).normalized;
Vector3 worldDir = (headRot * cameraProjectionRot) * localDir;
Vector3 labelPos = headPos + worldDir * (sphereRadius * 0.93f);
```
Labels use `TextMesh` (not TextMeshPro — TMP requires importing Essential Resources). Render queue 4010 (after sphere at 4000).

**Calibration Mode**: A/B buttons + right stick adjust `rightEyeOffset` in real-time. Edge-triggered. Logs `[OverlayRenderer] CALIBRATE` via `adb logcat -s Unity`.

### SegmentOverlay Shader
```hlsl
// Per-fragment pinhole projection:
float3 diff = IN.worldPos - _LeftCameraWorldPos;
float3 localDir = mul(_CameraRotationMatrix, float4(diff, 0.0)).xyz;
float uPixel = scaledFocal.x * (localDir.x / localDir.z) + scaledPrincipal.x;
float vPixel = scaledFocal.y * (localDir.y / localDir.z) + scaledPrincipal.y;
finalUV = float2(uPixel / _TextureResolution.x, vPixel / _TextureResolution.y);
```
- Intrinsics scaled from sensor resolution to mask texture resolution
- Right eye offset applied when `unity_StereoEyeIndex == 1`
- Edge fade: `smoothstep` at texture borders hides rectangular FOV cutoff
- Debug mode: magenta = behind camera, yellow = out of bounds, UV gradient = projection test

### Meta/OVR Dependencies
```csharp
Meta.XR.PassthroughCameraAccess  // Camera texture + intrinsics + LensOffset
OVRInput                          // Controller buttons for calibration
```

---

## Connecting the Two: Coordinate Quirks

### The Y-Flip Chain
```
Quest camera (correct orientation)
  → Client GPU blit: Y-flip (upside down JPEG sent over wire)
  → Server orchestrator: cv2.flip(frame, 0) (corrects back)
  → SAM processes correct frame, computes center_x/center_y (OpenCV: 0=top)
  → RLE mask encoded in OpenCV row order (0=top)

  → Unity RLE decoder: unityRow = h-1-rleRow (flips to Unity: 0=bottom)
  → Shader: rotation matrix includes LensOffset's 180deg X flip
      → pinhole vPixel increases with localDir.y (up in Unity)
      → high vPixel → high UV.y → top of Unity texture → top of original image ✓
  → Labels: MUST use (1 - CenterY) before inverse pinhole
      → CenterY=0 (OpenCV top) → py = 1.0 * height → high pixel → correct upward direction
```

### Why `_RightEyeOffset` Can't Fix Periphery
Both eyes project from `_LeftCameraWorldPos` (left camera physical position). Right eye is ~63mm away, sees the sphere from a different angle. The parallax varies with viewing direction — a constant pixel offset only corrects center. The overlay is at sphere depth (1m); objects at other depths have different stereo parallax. This is fundamental to monocular overlay on stereo display.

### `masks_frame_id` vs `frame_id`
Server sends both. `frame_id` = latest frame received. `masks_frame_id` = the frame whose pixels SAM actually processed (may be several frames behind). Client MUST use `masks_frame_id` to look up capture-time head rotation for world-locked reprojection.

### Rotation Consistency Rule
`_CameraRotationMatrix` and `_LeftCameraWorldPos` MUST use the same head rotation (capture-time `headRotation`). Using current rotation for one and capture-time for the other causes angle-dependent misalignment that grows with head turn angle. This manifests as: "works at one angle, broken at another."

### Protobuf Field Mapping
| Server (Python) | Client (C#) | Notes |
|---|---|---|
| `seg.center_x` | `seg.CenterX` | Normalized [0,1], OpenCV Y convention |
| `seg.rle_mask` | `seg.RleMask` | Bytes, uint16 LE run lengths |
| `seg.mask_width/height` | `seg.MaskWidth/Height` | RLE mask dimensions (75% of frame) |
| `response.masks_frame_id` | `msg.MasksFrameId` | For pose lookup |
| `response.masks_updated` | `msg.MasksUpdated` | Only update overlay when True |
