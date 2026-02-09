# Unity Voice Agent & Effects Implementation Plan

## Context

The server-side voice agent pipeline is now fully wired: audio streams from clients as PCM16, the server handles wake word detection ("Hey Vibe"), Whisper transcription, Gemini-powered command planning, and SAM3 prompt sync. Effects metadata (blur, dim, pixelate, highlight, outline) and full-screen filters (dim, warm, cool, night, grayscale) are sent in every `ServerMessage`.

The Unity client currently receives and parses `ServerMessage` segments but **ignores** effect metadata, conversation state, and voice agent state entirely. It also never sends audio to the server.

This plan covers the changes needed in `SocialSenseAR-Unity` to complete the end-to-end pipeline on Quest.

---

## What Already Works in Unity

- Frame capture (Quest passthrough + editor webcam) with async GPU readback
- JPEG encoding and `ClientMessage` protobuf transmission
- `ServerMessage` deserialization (segments, conversation state, metrics)
- RLE mask decoding into RGBA32 texture (outline + filled modes)
- Pinhole camera projection shader with stereo support
- World-locked overlay via capture-time head rotation reprojection
- 3D billboard labels for segments

## What's Missing

1. **Protobuf classes** for `EffectMetadata`, `VoiceAgentState`, `FullScreenFilter`
2. **Audio capture and streaming** (Quest mic -> `AudioPayload` protobuf)
3. **Per-segment effect rendering** (blur, dim, pixelate, highlight, outline)
4. **Full-screen filter post-processing**
5. **Voice agent HUD** (listening/recording state, transcript, responses)

---

## Step 1: Add Missing Protobuf Message Types

**File:** `Assets/Scripts/Proto/SocialsenseMessages.cs`

The server already sends these fields but the C# protobuf classes don't define them. Add:

### EffectMetadata (inside SceneSegment)

```csharp
public class EffectMetadata : IMessage<EffectMetadata>
{
    public string EffectType { get; set; } = "";      // "blur", "dim", "pixelate", "highlight", "outline", "none"
    public float Intensity { get; set; }                // 0.0-1.0
    public string ColorHex { get; set; } = "";          // hex color for outline/highlight
    public Dictionary<string, string> Params { get; set; } = new();
    // ... MergeFrom, WriteTo, CalculateSize
}
```

### VoiceAgentState (inside ConversationState)

```csharp
public class VoiceAgentState : IMessage<VoiceAgentState>
{
    public bool Listening { get; set; }
    public bool Recording { get; set; }
    public string PartialTranscript { get; set; } = "";
    public string LastCommand { get; set; } = "";
    public string LastResponse { get; set; } = "";
    public double LastCommandTime { get; set; }
    public FullScreenFilter FullScreenFilter { get; set; }
    // ... MergeFrom, WriteTo, CalculateSize
}
```

### FullScreenFilter

```csharp
public class FullScreenFilter : IMessage<FullScreenFilter>
{
    public string FilterType { get; set; } = "";   // "dim", "warm", "cool", "night", "grayscale", "none"
    public float Intensity { get; set; }            // 0.0-1.0
    // ... MergeFrom, WriteTo, CalculateSize
}
```

### Wire into existing classes

- `SceneSegment`: Add `public EffectMetadata Effect { get; set; }` field + MergeFrom case for tag 16
- `ConversationState`: Add `public VoiceAgentState VoiceAgent { get; set; }` field + MergeFrom case for the appropriate tag
- `ClientMessage`: Add `AudioPayload` oneof case so the client can send audio

### AudioPayload (for sending mic audio)

```csharp
public class AudioPayload : IMessage<AudioPayload>
{
    public ByteString Pcm16Data { get; set; } = ByteString.Empty;
    public uint SampleRate { get; set; }
    public uint NumSamples { get; set; }
    // ... WriteTo, CalculateSize (send-only, no MergeFrom needed)
}
```

**Reference:** Match field numbers from `server/proto/socialsense.proto` exactly.

---

## Step 2: Audio Capture and Streaming

**New file:** `Assets/Scripts/AudioStreamer.cs`

Captures PCM16 from Quest mic and sends as `AudioPayload` over the existing WebSocket.

```
Architecture:
  Unity Microphone.Start() -> AudioClip circular buffer
  -> Every 0.5s, read samples -> convert float32 to int16
  -> Build ClientMessage with AudioPayload
  -> WebSocket.Send(bytes)
```

### Implementation

```csharp
public class AudioStreamer : MonoBehaviour
{
    [SerializeField] private SocialSenseClient client;  // reference to send via WebSocket
    [SerializeField] private int sampleRate = 16000;
    [SerializeField] private float chunkDuration = 0.5f;

    private AudioClip _micClip;
    private int _lastReadPos;
    private float _timer;
    private bool _streaming;

    public void StartStreaming()
    {
        // Request mic permission on Android/Quest
        _micClip = Microphone.Start(null, true, 10, sampleRate);
        _lastReadPos = 0;
        _streaming = true;
    }

    void Update()
    {
        if (!_streaming) return;
        _timer += Time.deltaTime;
        if (_timer < chunkDuration) return;
        _timer = 0f;

        int currentPos = Microphone.GetPosition(null);
        if (currentPos == _lastReadPos) return;

        // Read samples from circular AudioClip buffer
        int samplesToRead = currentPos >= _lastReadPos
            ? currentPos - _lastReadPos
            : (_micClip.samples - _lastReadPos) + currentPos;

        float[] samples = new float[samplesToRead];
        _micClip.GetData(samples, _lastReadPos);
        _lastReadPos = currentPos;

        // Convert float32 [-1,1] -> PCM16 bytes
        byte[] pcm16 = new byte[samplesToRead * 2];
        for (int i = 0; i < samplesToRead; i++)
        {
            short val = (short)(Mathf.Clamp(samples[i], -1f, 1f) * 32767);
            pcm16[i * 2] = (byte)(val & 0xFF);
            pcm16[i * 2 + 1] = (byte)((val >> 8) & 0xFF);
        }

        // Send via existing WebSocket
        client.SendAudio(pcm16, (uint)sampleRate, (uint)samplesToRead);
    }

    public void StopStreaming()
    {
        _streaming = false;
        Microphone.End(null);
    }
}
```

### SocialSenseClient changes

Add `SendAudio()` method (similar to `SendFrame()`):

```csharp
public async void SendAudio(byte[] pcm16Data, uint sampleRate, uint numSamples)
{
    if (_websocket?.State != WebSocketState.Open) return;

    var msg = new ClientMessage
    {
        TimestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
        Audio = new AudioPayload
        {
            Pcm16Data = ByteString.CopyFrom(pcm16Data),
            SampleRate = sampleRate,
            NumSamples = numSamples,
        }
    };

    byte[] data = msg.ToByteArray();
    await _websocket.Send(data);
}
```

### Quest Permissions

Add to `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Request at runtime in `Start()`:
```csharp
if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
    Permission.RequestUserPermission(Permission.Microphone);
```

---

## Step 3: Per-Segment Effect Rendering

**File:** `Assets/Scripts/OverlayRenderer.cs`

Currently, all segments are rendered as either solid color fills or outlines. Effects from the server should change how each segment's mask region is rendered.

### Read effect data from segments

In `UpdateOverlay()`, after decoding each segment's RLE mask, check `seg.Effect`:

```csharp
string effectType = seg.Effect?.EffectType ?? "none";
float intensity = seg.Effect?.Intensity ?? 0f;
```

### Per-effect rendering strategies

All 5 effects are **fully implemented on the server dashboard** (`server/dashboard.py`) and serve as the reference implementation. The Unity client must match this behavior.

| Effect | Server (OpenCV) | Unity Approach | Invert Support |
|--------|----------------|----------------|:-:|
| **blur** | `cv2.GaussianBlur(ksize=51*intensity)` applied to masked pixels | Kawase blur on RenderTexture, composite via stencil mask | Yes |
| **dim** | `frame * (1 - 0.7*intensity)` on masked pixels — darkens to ~30% brightness at full | Semi-transparent black overlay in mask region (`alpha = 0.7 * intensity`) | Yes |
| **pixelate** | Resize down (`w/block, h/block`) then back up with `INTER_NEAREST` — block size `max(4, 20*intensity)` | Sample passthrough at floored UV grid in shader — `floor(uv * blockCount) / blockCount` | Yes |
| **highlight** | `addWeighted(frame, 1.0, white, 0.3*intensity)` + warm yellow tint (+40 to R,G channels) | Brighten + warm tint: `color.rgb = lerp(color.rgb, float3(1,1,0.7), 0.3*intensity)` | Yes |
| **outline** | Thicker contours: `thickness = max(3, 8*intensity)` with black shadow, bright color fill | Existing `DrawOutlineFromMask()` with `outlineWidth = max(3, 8*intensity)`, use segment color or `ColorHex` | Yes |

### Invert behavior

All effects support `invert: true`. When inverted, the effect applies to everything **outside** the mask (e.g., "blur everything but person" = blur outside person's mask, person stays sharp). On the server: `mask_bool = ~mask_bool`. In Unity: flip the stencil test or invert the mask texture.

### Recommended approach: Hybrid CPU + Shader

**Simple effects (dim, highlight, outline)** — handle in the existing CPU pixel buffer:
```csharp
// In DecodeRleIntoBuffer, modify color based on effect:
if (effectType == "dim")
{
    // Match server: darken by 70% at full intensity
    float dim = 1f - (0.7f * intensity);
    color = new Color32(0, 0, 0, (byte)(178 * intensity));  // semi-transparent black overlay
}
else if (effectType == "highlight")
{
    // Match server: brighten + warm yellow tint
    color = new Color32(255, 255, 180, (byte)(76 * intensity));  // warm bright overlay
}
else if (effectType == "outline")
{
    // Thicker outline — scale width with intensity
    outlineWidth = Mathf.Max(3, Mathf.RoundToInt(8 * intensity));
}
```

**Complex effects (blur, pixelate)** — need shader-based approach:

Create a second overlay pass that uses the segment mask as a stencil:

1. Render segment masks into a stencil/mask texture (R channel = segment index, G = effect type encoded)
2. New shader `SegmentEffects.shader` samples the passthrough camera texture
3. Where mask is active: apply blur (multi-tap Gaussian) or pixelate (floor UV to grid)
4. Composite over the passthrough layer

This requires access to the passthrough camera texture, which Quest provides via `PassthroughCameraAccess`.

### New shader: `Assets/Scripts/Shaders/SegmentEffects.shader`

```hlsl
// Uniforms: _MaskTex, _EffectType (int), _Intensity (float), _BlockSize (float)
// Per-fragment:
half4 frag(v2f i) : SV_Target
{
    float mask = tex2D(_MaskTex, i.uv).r;
    if (mask < 0.5) discard;  // no effect outside mask

    half4 color = tex2D(_PassthroughTex, i.uv);

    if (_EffectType == 1) // blur
    {
        // 9-tap Gaussian (or Kawase multi-pass for quality)
        half4 sum = 0;
        float2 texelSize = _PassthroughTex_TexelSize.xy * _Intensity * 4.0;
        for (int x = -1; x <= 1; x++)
            for (int y = -1; y <= 1; y++)
                sum += tex2D(_PassthroughTex, i.uv + float2(x, y) * texelSize);
        color = sum / 9.0;
    }
    else if (_EffectType == 2) // pixelate
    {
        // Match server: block = max(4, 20*intensity), then floor UV
        float blockSize = max(4.0, 20.0 * _Intensity);
        float2 blockCount = _ScreenParams.xy / blockSize;
        float2 blockUV = floor(i.uv * blockCount) / blockCount;
        color = tex2D(_PassthroughTex, blockUV);
    }
    return color;
}
```

---

## Step 4: Full-Screen Filter Post-Processing

**New file:** `Assets/Scripts/FullScreenFilterEffect.cs`

Reads `VoiceAgentState.FullScreenFilter` from `ServerMessage.Conversation` and applies a full-screen post-process.

### Filter types

| Filter | Shader operation |
|--------|-----------------|
| **dim** | `color.rgb *= (1.0 - intensity)` |
| **warm** | Shift toward orange: `color.r += 0.1 * intensity; color.b -= 0.1 * intensity` |
| **cool** | Shift toward blue: `color.b += 0.1 * intensity; color.r -= 0.1 * intensity` |
| **night** | Red-only mode: `color.gb *= (1.0 - intensity * 0.8)` + overall darken |
| **grayscale** | `float lum = dot(color.rgb, float3(0.299, 0.587, 0.114)); color.rgb = lerp(color.rgb, lum, intensity)` |

### Implementation

Use URP Renderer Feature or `OnRenderImage` (legacy) / `ScriptableRenderPass` (URP):

```csharp
public class FullScreenFilterEffect : MonoBehaviour
{
    public Material filterMaterial;  // uses FullScreenFilter.shader

    private string _filterType = "none";
    private float _intensity = 0f;

    public void SetFilter(string filterType, float intensity)
    {
        _filterType = filterType;
        _intensity = intensity;
    }

    // Called from SocialSenseClient.Update() after parsing ServerMessage:
    // var fs = msg.Conversation?.VoiceAgent?.FullScreenFilter;
    // if (fs != null) filterEffect.SetFilter(fs.FilterType, fs.Intensity);
}
```

### New shader: `Assets/Scripts/Shaders/FullScreenFilter.shader`

Blit shader with `_FilterType` (int) and `_Intensity` (float) uniforms. Branches per filter type in fragment shader.

---

## Step 5: Voice Agent HUD

**New file:** `Assets/Scripts/VoiceAgentHUD.cs`

World-space Canvas attached to the user's head (or wrist) showing voice agent state.

### UI Layout

```
+----------------------------------+
|  [mic icon]  Listening...        |   <- state indicator
|  "blur the laptop and..."        |   <- partial transcript (green, while recording)
|  > Applied blur to laptop        |   <- last response
+----------------------------------+
```

### Implementation

```csharp
public class VoiceAgentHUD : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI stateText;
    [SerializeField] private TextMeshProUGUI transcriptText;
    [SerializeField] private TextMeshProUGUI responseText;
    [SerializeField] private Image micIcon;

    public void UpdateState(VoiceAgentState state)
    {
        if (state == null)
        {
            gameObject.SetActive(false);
            return;
        }

        gameObject.SetActive(true);

        if (state.Recording)
        {
            stateText.text = "Recording...";
            stateText.color = Color.green;
            micIcon.color = Color.red;  // recording indicator
            transcriptText.text = state.PartialTranscript;
            transcriptText.gameObject.SetActive(true);
        }
        else
        {
            stateText.text = "Say 'Hey Vibe'";
            stateText.color = Color.gray;
            micIcon.color = Color.white;
            transcriptText.gameObject.SetActive(false);
        }

        if (!string.IsNullOrEmpty(state.LastResponse))
            responseText.text = state.LastResponse;
    }
}
```

### Positioning

- **Option A: Head-locked** — Canvas child of `CenterEyeAnchor`, positioned slightly below center of view (subtle, always visible)
- **Option B: Wrist-locked** — Canvas attached to left controller anchor (less intrusive, look at wrist to check)
- **Option C: World-space floating** — Appears when recording, fades after response (cleanest UX)

Recommended: **Option C** — show only when active, fade after 3s of idle.

### Wire into SocialSenseClient

In `Update()`, after parsing `ServerMessage`:

```csharp
if (msg.Conversation?.VoiceAgent != null)
{
    voiceAgentHUD.UpdateState(msg.Conversation.VoiceAgent);
}
```

---

## Step 6: SocialSenseClient Integration

Tie everything together in `SocialSenseClient.cs`:

### New serialized fields

```csharp
[Header("Voice Agent")]
[SerializeField] private AudioStreamer audioStreamer;
[SerializeField] private VoiceAgentHUD voiceAgentHUD;
[SerializeField] private FullScreenFilterEffect fullScreenFilter;
[SerializeField] private bool enableVoiceAgent = true;
```

### Start() additions

```csharp
if (enableVoiceAgent)
{
    // Request mic permission
    #if UNITY_ANDROID
    if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        Permission.RequestUserPermission(Permission.Microphone);
    #endif
    audioStreamer?.StartStreaming();
}
```

### Update() additions (after ServerMessage parsing)

```csharp
// Voice agent state
if (msg.Conversation?.VoiceAgent != null)
{
    voiceAgentHUD?.UpdateState(msg.Conversation.VoiceAgent);

    // Full-screen filter
    var fs = msg.Conversation.VoiceAgent.FullScreenFilter;
    if (fs != null && !string.IsNullOrEmpty(fs.FilterType) && fs.FilterType != "none")
        fullScreenFilter?.SetFilter(fs.FilterType, fs.Intensity);
    else
        fullScreenFilter?.SetFilter("none", 0f);
}
```

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `Assets/Scripts/Proto/SocialsenseMessages.cs` | Modify | Add EffectMetadata, VoiceAgentState, FullScreenFilter, AudioPayload classes + wire into existing messages |
| `Assets/Scripts/AudioStreamer.cs` | New | Quest mic capture -> PCM16 -> WebSocket AudioPayload |
| `Assets/Scripts/SocialSenseClient.cs` | Modify | Add `SendAudio()`, wire AudioStreamer + VoiceAgentHUD + FullScreenFilter |
| `Assets/Scripts/OverlayRenderer.cs` | Modify | Read `seg.Effect` and apply per-segment rendering (dim/highlight CPU-side, blur/pixelate shader-side) |
| `Assets/Scripts/VoiceAgentHUD.cs` | New | World-space Canvas showing listening/recording state, transcript, response |
| `Assets/Scripts/FullScreenFilterEffect.cs` | New | URP post-process for dim/warm/cool/night/grayscale |
| `Assets/Scripts/Shaders/SegmentEffects.shader` | New | Per-segment blur/pixelate using mask as stencil |
| `Assets/Scripts/Shaders/FullScreenFilter.shader` | New | Blit shader for full-screen color grading |
| `AndroidManifest.xml` | Modify | Add `RECORD_AUDIO` permission |

## Implementation Order

```
Step 1 (Protobuf)          <- Everything else depends on this
    |
    +-- Step 2 (Audio)     <- Independent, can parallel with Step 3
    |
    +-- Step 3 (Effects)   <- Independent, can parallel with Step 2
    |
Step 4 (Full-screen)       <- Depends on Step 1 (needs FullScreenFilter class)
    |
Step 5 (HUD)               <- Depends on Step 1 (needs VoiceAgentState class)
    |
Step 6 (Integration)       <- Depends on all above
```

## Testing

### Server dashboard (reference — already working)

All 5 effects can be tested via the Python test client dashboard before touching Unity:
- `blur person` — Gaussian blur on person mask
- `dim laptop` — darken laptop region to ~30% brightness
- `pixelate phone` — mosaic/block effect on phone
- `highlight cup` — brighten + warm yellow tint on cup
- `outline person` — thick colored contour around person
- `blur everything but person` — blur everywhere except person (invert)
- `clear` — remove all effects and prompts

### Unity integration

1. **Editor mode**: Run server locally (`python -m server.main --device cuda`), connect Unity editor via WebSocket, verify audio streams to server and voice agent state appears in HUD
2. **Effect parity**: For each effect type, compare Unity rendering side-by-side with server dashboard to ensure visual match
3. **Quest build**: Deploy APK, connect to Modal server, say "Hey Vibe, blur the laptop, thank you" — verify:
   - Dashboard/HUD shows "Recording..." during speech
   - Laptop mask appears with blur effect overlay
   - Voice agent response displayed
4. **All effects on Quest**: Test each effect type via voice commands:
   - "Hey Vibe, dim the person, thank you" — person region darkens
   - "Hey Vibe, pixelate the laptop, thank you" — laptop gets mosaic effect
   - "Hey Vibe, highlight my phone, thank you" — phone brightens with warm tint
   - "Hey Vibe, outline the cup, thank you" — thick colored outline around cup
5. **Invert on Quest**: "Hey Vibe, blur everything but me, thank you" — everything except person blurs
6. **Full-screen filter**: Say "Hey Vibe, it's too bright, dim everything, thank you" — verify screen darkens
7. **Effect removal**: Say "Hey Vibe, stop blurring the laptop, thank you" — verify blur removed, segment still tracked
