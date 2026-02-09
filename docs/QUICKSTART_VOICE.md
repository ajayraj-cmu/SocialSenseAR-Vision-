# Voice Agent Quick Start

Get up and running with voice control in 5 minutes!

## Prerequisites

```bash
# 1. Install dependencies
pip install openai google-genai sounddevice

# 2. Set API keys
export OPENAI_API_KEY="sk-..."      # For Whisper transcription
export GEMINI_API_KEY="..."         # For natural language understanding
```

## Start Server

```bash
python -u -m server.main --device cuda --pipeline sam3
```

Wait for these messages:
```
✓ Pipeline mode: SAM3 (text-prompted, no Gemini)
✓ Whisper transcriber ready (model=whisper-1)
✓ Voice command planner ready (model=gemini-2.5-flash)
✓ Gemini scene understanding ready
✓ VoiceAgent started
✓ Server ready. Waiting for Quest connection...
```

## Start Voice Client

```bash
python -m server.test_client_voice
```

Should see:
```
✓ Webcam started
✓ Microphone started
✓ Connected to ws://localhost:8765
```

## Try It!

### Basic Commands

**Blur an object:**
```
You: "Hey Vibe, blur the laptop, thank you"
```

**Dim an object:**
```
You: "Hey Vibe, dim the monitor, thank you"
```

**Remove effect:**
```
You: "Hey Vibe, stop blurring the laptop, thank you"
```

### Advanced Commands

**Dynamic scene understanding:**
```
You: "Hey Vibe, my environment is too bright and overstimulating, dim it, thank you"
```
→ Gemini Vision analyzes scene, identifies bright objects (lights, windows, screens), applies effects

**Background blur:**
```
You: "Hey Vibe, blur the background, thank you"
```
→ Gemini Vision identifies background objects (walls, ceiling, floor), applies blur

## Watch the Dashboard

The server opens a window showing:

**Left side**: Camera feed with colored mask overlays

**Right panel**:
- **ACTIVE PROMPTS**: Objects being segmented
- **DETECTED**: Segments with effect types (e.g., "laptop:blur")
- **VOICE AGENT**: Status (Listening/RECORDING), last command/response
- **STATS**: FPS, segment count

## Voice UX Rules

1. **Always say "Hey Vibe" first** (wake word)
2. **Make your request** (e.g., "blur the laptop")
3. **End with "Thank you"** (command executes after this)

Example:
```
"Hey Vibe, blur the laptop and dim the monitor, thank you"
        ^         ^command goes here^              ^
     wake word                              end phrase
```

## Common Issues

### "No OpenAI API key — transcriber disabled"
```bash
export OPENAI_API_KEY="sk-..."
```

### "Whisper unavailable"
```bash
pip install sounddevice
```

### Server not responding to voice
1. Check microphone is working (client should show "✓ Microphone started")
2. Speak clearly: "Hey... Vibe... blur the laptop... Thank you"
3. Check server logs for "Wake word detected!"

### Object not segmented
1. Check "ACTIVE PROMPTS" panel — object label should appear
2. Make sure object is visible in camera
3. Try synonym: "computer" → "monitor", "light" → "lamp"

## Testing Without Voice

Type commands in the server dashboard window (click window first):

```
blur laptop     [Enter]
dim monitor     [Enter]
clear           [Enter]
```

## Performance Tips

**First command is slower** (~1.5s):
- Gemini Vision analyzes scene
- Results cached for 10s

**Subsequent commands are fast** (~0.8s):
- Uses cached scene + known objects
- No redundant API calls

## What's Happening Behind the Scenes

```
Your voice → Microphone → PCM16 audio
    ↓
WebSocket → Server
    ↓
Whisper API → Transcription → "blur the laptop"
    ↓
Wake word detected? → Start recording
    ↓
End phrase detected? → Process command
    ↓
Gemini LLM → Intent mapping → {targets: ["laptop"], effect: "blur"}
    ↓
Update state → Add "laptop" to known objects + active effects
    ↓
SAM3 segments "laptop" → Apply blur effect → Send to client
    ↓
Dashboard shows: laptop (0.95) with blur overlay
```

## Next Steps

Once working:
1. Read `VOICE_AGENT.md` for complete documentation
2. Try complex commands: "blur everything except the person"
3. Test persistence: blur object, look away, look back (effect persists!)
4. Experiment with effects: blur, dim, pixelate, highlight

## Cost Estimate

**OpenAI Whisper**:
- ~$0.006 per minute of audio
- Example: 5 min testing = $0.03

**Gemini 2.0 Flash**:
- Free tier: 15 requests per minute
- Vision: 1500 requests per day free
- Example: 20 commands = $0.00 (free tier)

## Full Command Reference

### Explicit Object Commands
```
blur <object>
dim <object>
pixelate <object>
highlight <object>
stop <effect>ing <object>
clear everything
```

### Dynamic Commands (require Gemini Vision)
```
dim everything, it's too bright
blur the background
make the environment less stimulating
highlight the important objects
```

### Supported Objects
Any common object SAM3 can segment:
- laptop, monitor, computer, screen
- desk, chair, table, couch
- lamp, light, window
- person, face, hand
- door, wall, floor, ceiling
- cup, bottle, phone, keyboard
- ...and many more!

## Debugging

Enable verbose logging:
```bash
LOG_LEVEL=DEBUG python -u -m server.main --device cuda --pipeline sam3
```

Shows:
- Whisper transcriptions
- Gemini Vision results
- Command planning reasoning
- Effect application

## Support

**Smoke test** (verifies installation):
```bash
python tests/test_voice_agent_smoke.py
```

Should see: `✓ All smoke tests passed!`

**Documentation**:
- `VOICE_AGENT.md` — Complete guide
- `README.md` — Project overview
- `IMPLEMENTATION_SUMMARY.md` — Technical details

**Have fun!** 🎤🤖
