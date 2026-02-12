# Voice Agent Pipeline — Natural Language Control of SAM3

Production-quality, low-latency voice control system for SocialSenseAR. Control object segmentation and visual effects using natural language.

## Architecture

```
Microphone → PCM16 Audio
    ↓
WebSocket (audio payload)
    ↓
PipelineOrchestrator.process_audio()
    ↓
VoiceAgent Pipeline:
    1. WakeWordGate: Detects "Hey Vibe"
    2. UtteranceAssembler: Collects speech until "Thank you"
    3. WhisperTranscriber: Transcribes via OpenAI API
    4. VoiceCommandPlanner: Maps intent → structured plan (Gemini)
    5. GeminiSceneUnderstanding: On-demand vision (what's visible?)
    ↓
Update State:
    - Known objects registry (persistent across frames)
    - Active effects {label: {type, intensity}}
    - Full-screen filters
    ↓
SAM3 Loop:
    - Segments ONLY requested objects
    - Applies effects to tracked segments
    - Persists effects when objects leave/reenter view
```

## Key Design Principles

### 1. **No "Scan Everything" Behavior**
- Default SAM3 prompt set is empty
- Objects are added to prompts ONLY when requested via voice
- Reduces latency and improves accuracy

### 2. **Persistence**
- **Known objects registry**: Tracks all objects ever requested
- **Active effects**: Persist across frames, even if object not visible
- Effects reapply automatically when object reenters view

### 3. **Asynchronous Processing**
- Audio transcription + command planning run on background threads
- Never blocks the 60fps vision pipeline
- Results apply atomically on next frame

### 4. **On-Demand Vision**
- Gemini Vision called ONLY when needed:
  - User makes vague request ("it's too bright")
  - User names object not in known registry
- Results cached (10s TTL) to avoid redundant API calls

## Voice UX

### Wake Word: "Hey Vibe"
Tolerates transcription errors:
- "Hey Vibes"
- "Hey Vive"
- "Hey Five"

### End Phrase: "Thank you"
Also accepts:
- "Thanks"
- "Thank ya"

### Command Structure
```
"Hey Vibe, [COMMAND], Thank you"
```

Commands execute after full utterance is collected (not partial).

## Voice Command Examples

### Explicit Object Requests
```
"Hey Vibe, blur the laptop, thank you"
→ Segments "laptop", applies blur effect

"Hey Vibe, dim the monitors, thank you"
→ Segments "monitor" objects, applies dim effect

"Hey Vibe, highlight the person, thank you"
→ Segments "person", applies highlight effect
```

### Implicit/Dynamic Requests
```
"Hey Vibe, my environment is too bright and overstimulating, dim it, thank you"
→ Gemini Vision analyzes scene
→ Identifies bright objects (lights, windows, screens)
→ Applies dim effect + global dim filter

"Hey Vibe, blur the background, thank you"
→ Gemini Vision identifies background objects (walls, floor, ceiling)
→ Applies blur to structural elements
```

### Effect Removal
```
"Hey Vibe, stop blurring the laptop, thank you"
→ Removes blur effect from "laptop"
→ Keeps "laptop" in known registry (persistent)

"Hey Vibe, clear everything, thank you"
→ Removes all effects
→ Clears full-screen filters
```

### Effect Changes
```
"Hey Vibe, change the laptop from blur to dim, thank you"
→ Updates effect type for "laptop"
```

## Supported Effects

### Per-Object Effects
- `blur`: Gaussian blur (default intensity 0.8)
- `dim`: Darken/reduce brightness
- `pixelate`: Mosaic pixelation
- `highlight`: Brighten/emphasize
- `outline`: Draw colored outline
- `none`: No effect (remove)

### Full-Screen Filters
Applied globally (not per-object):
- `dim`: Reduce overall brightness
- `warm`: Warm color temperature
- `cool`: Cool color temperature
- `night`: Night mode (high contrast)
- `grayscale`: Desaturate colors

## Setup & Requirements

### API Keys
```bash
# Required for voice transcription
export OPENAI_API_KEY="sk-..."

# Required for natural language understanding + scene analysis
export GEMINI_API_KEY="..."  # or GOOGLE_API_KEY
```

### Python Dependencies
```bash
pip install openai google-genai sounddevice
```

### Server Configuration
```python
# server/config.py
ServerConfig(
    pipeline_mode="sam3",        # MUST use SAM3 for text prompts
    audio_enabled=True,          # Enable voice agent
    openai_api_key="...",        # or via env var
    gemini_api_key="...",        # or via env var
)
```

## Usage

### 1. Start Server (SAM3 + Voice Agent)
```bash
python -u -m server.main --device cuda --pipeline sam3
```

Wait for:
```
Pipeline mode: SAM3 (text-prompted, no Gemini)
Voice agent pipeline created (will initialize on first frame)
Whisper transcriber ready (model=whisper-1)
Voice command planner ready (model=gemini-2.5-flash)
Gemini scene understanding ready (model=gemini-2.5-flash)
VoiceAgent started
```

### 2. Start Webcam Client (with Microphone)
```bash
python -m server.test_client_voice
```

The server dashboard window shows:
- **Camera feed** with mask overlays (left)
- **Active prompts** (objects being segmented)
- **Detected segments** with effects
- **Voice agent status** (listening/recording)
- **Last command + response**
- **Stats** (FPS, segment count)

### 3. Issue Voice Commands

**Example Session:**
```
You: "Hey Vibe, blur the laptop, thank you"
Server: ✓ Applied blur to laptop

[Laptop appears with blur overlay]

You: "Hey Vibe, it's too bright, dim it, thank you"
Server: ✓ Applied dim filter
        ✓ Applied dim to lamp, window

[Scene dims, lights/windows darkened]

You: "Hey Vibe, stop blurring the laptop, thank you"
Server: ✓ Removed effects from laptop

[Laptop overlay removed, dim filter persists]
```

## Dashboard Indicators

### Voice Agent Panel
```
VOICE AGENT
-----------
Listening               # Waiting for "Hey Vibe"
RECORDING...            # Active utterance
Applied blur to laptop  # Last response
```

### Active Prompts Panel
```
ACTIVE PROMPTS
--------------
> laptop
> monitor
> lamp
```
Only shows objects currently requested (not "everything").

### Detected Segments Panel
```
DETECTED
--------
laptop:blur (0.95)
monitor (0.87)
lamp:dim (0.92)
```
Effect type shown next to label.

## How It Works

### 1. Audio Ingestion
```python
# websocket_server.py receives audio protobuf
def _process_audio(self, msg: pb.ClientMessage):
    self.pipeline.process_audio(
        msg.audio.pcm16_data,
        msg.audio.sample_rate,
        msg.audio.num_samples,
    )
```

Audio is buffered and processed asynchronously by VoiceAgent.

### 2. Wake Word Detection
```python
# server/audio/voice_agent.py
class WakeWordGate:
    def detect(self, text: str) -> bool:
        # Regex matching: "hey vibe", "hey vibes", etc.
        return any(pattern.search(text) for pattern in self._patterns)
```

When detected:
- UtteranceAssembler starts recording
- Conversation state updates: `recording=True`

### 3. Utterance Assembly
```python
class UtteranceAssembler:
    def add_chunk(self, text: str) -> Optional[str]:
        self._chunks.append(text)
        # Check for "thank you" end phrase
        if self._end_detected(full_text):
            return self._strip_end_phrase(full_text)
        return None
```

Transcript chunks accumulate until "Thank you" detected.

### 4. Command Planning (Gemini Reasoning)
```python
# server/audio/voice_agent.py
planner.plan_command(
    utterance="blur the laptop",
    known_objects={"laptop", "monitor", "desk"},
    active_effects={"monitor": {"type": "dim", "intensity": 0.7}},
    visible_objects=["laptop", "desk", "chair"],  # from Gemini Vision
    scene_context="Office desk with laptop and monitor",
)
→ CommandPlan(
    targets=["laptop"],
    effect_type="blur",
    intensity=0.8,
    action="add",
    reasoning="User explicitly requested blur on laptop"
)
```

Gemini LLM maps natural language → structured actions.

### 5. Scene Understanding (On-Demand)
When user makes vague request or names unknown object:
```python
# server/vision/gemini_scene_understanding.py
snapshot = scene_understanding.snapshot(frame_bgr)
→ {
    "objects": ["laptop", "monitor", "desk", "lamp", "window", "wall"],
    "scene_description": "Office workspace with desk setup",
    "bright_objects": ["lamp", "window"],
    "background_objects": ["wall", "ceiling", "floor"]
}
```

Used to infer targets when not explicitly named.

### 6. State Update (Atomic)
```python
# Voice agent updates state under lock
with self._state_lock:
    for target in plan.targets:
        self._known_objects.add(target)
        self._active_effects[target] = {
            "type": plan.effect_type,
            "intensity": plan.intensity,
        }
```

State is thread-safe and updates atomically.

### 7. SAM3 Sync + Effect Application
```python
# orchestrator.py SAM loop
for seg in tracked:
    if seg.label:
        voice_agent.add_known_object(seg.label)

self._apply_voice_effects(tracked)
```

- Known objects registry updated as SAM3 segments them
- Effects applied to tracked segments on every frame
- Persistence: effects reapply when objects reappear

### 8. Protobuf Serialization
```python
# websocket_server.py
for seg in result.segments:
    proto_seg.effect.effect_type = seg.effect.effect_type
    proto_seg.effect.intensity = seg.effect.intensity

response.conversation.voice_agent.listening = conv_state["listening"]
response.conversation.voice_agent.recording = conv_state["recording"]
response.conversation.voice_agent.last_response = conv_state["last_response"]
```

Effects + voice state transmitted to client.

## Fallback Behavior (No API Keys)

### Without OpenAI Key
- Whisper transcription disabled
- Voice agent inactive
- Warning logged: `"No OpenAI API key — transcriber disabled"`

### Without Gemini Key
- Natural language understanding falls back to keyword matching
- Scene understanding disabled
- Still supports simple commands: "blur laptop", "dim monitor"

## Performance

### Latency Budget
```
Microphone → Server:         ~50ms (network)
Whisper API transcription:   ~500ms (OpenAI)
Gemini command planning:     ~300ms (Gemini LLM)
Gemini Vision (on-demand):   ~800ms (first call, then cached)
State update + SAM sync:     <1ms
Effect application:          <1ms per segment
--------------------------------------------
Total:                       ~1s (acceptable for voice UX)
```

### Caching Strategy
- **Gemini Vision snapshots**: 10s TTL (scene doesn't change rapidly)
- **Known objects registry**: Persistent (never expires)
- **Active effects**: Persistent until explicitly removed

### Frame Pipeline (Unaffected)
- Vision pipeline maintains 60fps (SAM runs continuously)
- Voice processing is fully asynchronous
- Effects apply without blocking frame processing

## Debugging

### Server Logs
```bash
# Voice agent logs
[VoiceAgent] Wake word detected!
[VoiceAgent] Complete utterance: "blur the laptop"
[VoiceCommandPlanner] Command plan: add blur to ['laptop']
[PipelineOrchestrator] Applied blur to laptop
```

### Dashboard HUD
Right panel shows:
- Voice agent status (listening/recording)
- Last command + response
- Active prompts (what SAM3 is segmenting)
- Detected segments with effects

### Test Without Webcam
```python
# Send audio directly via WebSocket client
import websockets
import asyncio
from server.proto import socialsense_pb2 as pb

async def test_audio():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as ws:
        msg = pb.ClientMessage()
        msg.audio.pcm16_data = audio_bytes  # PCM16 mono 16kHz
        msg.audio.sample_rate = 16000
        msg.audio.num_samples = len(audio_bytes) // 2
        await ws.send(msg.SerializeToString())

asyncio.run(test_audio())
```

## Limitations & Future Work

### Current Limitations
1. **Monaural camera**: Both eyes see identical overlay (no stereo disparity)
2. **SAM3 prompt rotation**: With many objects, some may segment less frequently
3. **English only**: Whisper configured for English (`language="en"`)
4. **Quest client rendering**: Effect visualization not yet fully implemented in Unity

### Future Enhancements
- Multi-language support
- Voice feedback (TTS responses)
- Gesture + voice multimodal control
- Per-eye stereo overlay (requires dual-camera setup)
- Advanced effects (shader-based blur, depth-aware dim)

## Troubleshooting

### "No OpenAI API key — transcriber disabled"
```bash
export OPENAI_API_KEY="sk-..."
```

### "No Gemini API key — scene understanding disabled"
```bash
export GEMINI_API_KEY="..."
```

### "Whisper unavailable" (test_client_voice)
```bash
pip install sounddevice
```

### Voice commands not executing
1. Check server logs for "Wake word detected"
2. Verify microphone is working (`test_client_voice` should show audio capture)
3. Speak clearly: "Hey Vibe ... Thank you" (don't forget end phrase)
4. Check Whisper transcription logs (should see `[WhisperTranscriber] Whisper: "..."`)

### SAM3 not segmenting requested objects
1. Check "ACTIVE PROMPTS" panel — object label should appear
2. Verify object is visible in camera view
3. SAM3 may need better prompt (try synonyms: "computer" → "monitor")
4. Check SAM3 confidence threshold (default 0.12 in config)

### Effects not appearing in dashboard
1. Check protobuf serialization (should see `effect.effect_type` in logs)
2. Verify voice agent state: `voice_agent.last_response` should confirm
3. Dashboard shows effect type next to label: `laptop:blur (0.95)`

## Code Structure

```
server/
├── audio/
│   ├── transcriber.py              # OpenAI Whisper API wrapper
│   └── voice_agent.py              # VoiceAgent, WakeWordGate, UtteranceAssembler, Planner
├── vision/
│   ├── segment_data.py             # SegmentData + EffectData models
│   └── gemini_scene_understanding.py  # On-demand Gemini Vision
├── pipeline/
│   └── orchestrator.py             # Wires voice agent into SAM loop
├── proto/
│   └── socialsense.proto           # Extended with EffectMetadata, VoiceAgentState
├── websocket_server.py             # Serializes voice state + effects to protobuf
├── test_client_voice.py            # Webcam + microphone test client
└── config.py                       # audio_enabled, API keys

VOICE_AGENT.md                      # This file
```

## API Reference

### VoiceAgent
```python
voice_agent = VoiceAgent(transcriber, planner, scene_understanding, config)
voice_agent.start()

# Feed audio (called from websocket thread)
voice_agent.ingest_audio(pcm16_data, sample_rate, num_samples)

# Query state (thread-safe)
known_objects = voice_agent.get_known_objects()
active_effects = voice_agent.get_active_effects()
full_screen_filter = voice_agent.get_full_screen_filter()
conversation_state = voice_agent.get_conversation_state()

# Update frame for Gemini Vision
voice_agent.update_frame(frame_bgr)

# Add object to known registry (called by SAM loop)
voice_agent.add_known_object("laptop")
```

### CommandPlan
```python
@dataclass
class CommandPlan:
    targets: list[str]              # ["laptop", "monitor"]
    effect_type: str                # "blur", "dim", "pixelate", ...
    intensity: float                # 0.0-1.0
    action: str                     # "add", "remove", "change"
    full_screen_filter: Optional[str]  # "dim", "warm", ...
    full_screen_intensity: float    # 0.0-1.0
    reasoning: str                  # Gemini's explanation
```

### EffectData
```python
class EffectData:
    effect_type: str                # "blur", "dim", "pixelate", ...
    intensity: float                # 0.0-1.0
    color_hex: str                  # "#FF0000" (for outline/highlight)
    params: dict                    # Additional parameters
```

## License & Credits

Part of SocialSenseAR Vision pipeline.
Voice agent implementation: Claude Code + SocialSenseAR team.

Uses:
- OpenAI Whisper API (transcription)
- Google Gemini 2.0 Flash (reasoning + vision)
- Meta SAM3 (text-prompted segmentation)
