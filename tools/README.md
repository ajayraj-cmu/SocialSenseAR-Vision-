# SocialSenseAR Tools

Development and debugging utilities for the SocialSenseAR pipeline.

## monitor_conversation.py

Real-time terminal monitor for PersonaPlex voice agent conversation flow.
Shows what was said, how it was translated into commands, and the agent's response.

### Quick Start

```bash
cd SocialSenseAR

# Connect to local server (default: ws://127.0.0.1:8765)
python tools/monitor_conversation.py

# Connect to Modal deployment
python tools/monitor_conversation.py --url wss://your-modal-url.modal.run/ws

# Verbose mode (periodic stats)
python tools/monitor_conversation.py --verbose
```

### What It Shows

```
  [14:23:45]  RECORDING  Wake word detected -- listening...
  [14:23:47]  Hearing: "Hey Vibe blur the laptop"

  [14:23:48]  COMMAND #1
      Said:       "[COMMAND:blur:laptop] Blurring the laptop."
      Translated: Blur -> laptop
      Response:   "Blurring the laptop."
      Active:     blur(laptop)

  [14:23:52]  LISTENING  Ready for wake word

  [14:24:10]  COMMAND #2
      Said:       "[COMMAND:dim:person:0.7] Dimming the person."
      Translated: Dim -> person @ 70%
      Response:   "Dimming the person."
      Active:     blur(laptop), dim(person)
```

### Typing Commands

You can type commands directly in the monitor terminal. They are sent to the
server as control messages (same as the dashboard text input or Unity typed commands).

```
> blur laptop
  [14:25:00]  SENT       "blur laptop"
```

### How It Works

1. Connects to the SocialSense WebSocket server
2. Sends tiny 2x2 heartbeat frames at 1 fps to trigger server responses
3. Parses `VoiceAgentState` from each `ServerMessage` protobuf response
4. Detects meaningful state changes (recording start, new command, listening)
5. Pretty-prints with ANSI colors and structured command breakdown

### Important Notes

- **Single client**: The server handles one client at a time. Running the
  monitor will replace any connected Unity client. Use this tool when Unity
  is not connected, or for standalone testing.
- **Heartbeat frames**: Requires `Pillow` or `opencv-python` to generate
  tiny JPEG heartbeat frames. If neither is installed, the monitor runs in
  listen-only mode (won't receive server responses).
- **Dependencies**: `websockets`, `protobuf` (same as the server).

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url URL` | `ws://127.0.0.1:8765` | WebSocket server URL |
| `--fps N` | `1.0` | Heartbeat frame rate (fps) |
| `--verbose` / `-v` | off | Show periodic message/segment stats |
