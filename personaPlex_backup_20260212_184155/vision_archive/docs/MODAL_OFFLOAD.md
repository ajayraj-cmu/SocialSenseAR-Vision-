## Modal GPU Offload (Run/Deploy From Your Laptop)

This document is written **for Claude Code** to follow end-to-end inside this workspace.  
Goal: make **Modal** the remote “compute server” (GPU) that runs SAM3 + other heavy ML, while your laptop (and later Quest) stays thin and only **streams frames/audio + renders overlays**.

---

## What you want (requirements)

- **Offload**: SAM3 inference + any major compute to **Modal GPU**.
- **Local**: webcam (for now) / Quest (later) captures frames + optionally audio; renders masks/effects client-side.
- **No “segment everything” scanning**: only segment what the user requests (voice-driven prompts). (This is a *pipeline behavior* task and can be implemented alongside Modal offload.)
- **Preserve existing infra**: keep the protobuf/WebSocket protocol and Quest pathway intact.
- **Setup automation**: Claude Code should do essentially everything: Modal auth, secrets, build image, deploy, give you an endpoint, then **ask you at the end** if you want to deploy and test.

---

## Important clarification: is `MODAL_TOKEN_SET` in `.env` enough?

**Not by itself.** Modal authentication is stored in Modal’s local config (typically `~/.modal.toml` / keychain).  
Having `MODAL_TOKEN_SET` inside `.env` is useful, but Claude Code still must **run a Modal auth command** to apply it.

Expected pattern:
- `.env` contains something like:
  - `MODAL_TOKEN_SET="ak-... --token-secret as-..."`
- Claude Code must parse out the token id + secret and run the appropriate command (Modal CLI has changed over time, so Claude should confirm the exact flags via `modal token --help`):
  - example (common pattern): `modal token set --token-id <ak-...> --token-secret <as-...>`

If the CLI does not support “token set”, Claude should fall back to a guided login step using `modal token new` and then continue.

---

## Inputs Claude Code should read from `.env`

Claude Code must read `.env` and use these values:
- `HF_TOKEN`: required for gated models like `facebook/sam3`
- `OPENAI_API_KEY`: needed for Whisper API (and any OpenAI usage)
- `GEMINI_API_KEY`: needed if/when Gemini Vision/reasoning is used
- `MODAL_TOKEN_SET`: optional convenience for non-interactive auth

If any of these are missing, Claude Code must stop and ask you for the missing ones.

---

## Target architecture (after offload)

### Modal (remote GPU)
- Hosts a **WebSocket endpoint** that accepts **the same protobuf messages** the current local server expects:
  - `ClientMessage(frame=...)`  → returns `ServerMessage(segments=..., masks_frame_id=...)`
  - `ClientMessage(audio=...)`  → (optional) processes audio and updates server state
  - `ClientMessage(control=...)` → updates prompts/effects state
- Runs:
  - `PipelineOrchestrator(config)` with `pipeline_mode="sam3"` (GPU)
  - SAM3 + (optionally) emotion detection / mediapipe / etc if desired
- Uses a **persistent Modal Volume** for HF + model caches (fast cold starts).

### Local machine / Quest
- Captures camera frames (webcam now; Quest later).
- Sends frames to remote Modal WebSocket.
- Renders overlays locally (Quest renderer or an OpenCV debug viewer).

---

## Claude Code runbook (do these steps in order)

### Step 0 — Safety / scope checks
- Confirm there is **no running local server** that would conflict.
- Confirm the repo runs locally (`python -m server.main --stub`) before offloading, if helpful.

### Step 1 — Install / verify Modal CLI
Claude Code should:
- `python -m pip install -U modal`
- verify: `modal --version`

### Step 2 — Authenticate Modal (use `.env` if possible)
Claude Code should:
1) Read `.env`, check `MODAL_TOKEN_SET`.
2) If present, attempt:
   - `modal token --help` to discover the correct command/flags
   - run the correct “set token” command using the parsed id/secret
3) Verify auth:
   - `modal whoami` (or `modal profile`) depending on CLI

If auth cannot be set non-interactively, Claude Code must ask you to run the one-time browser login and then continue automatically.

### Step 3 — Create Modal secrets (API keys)
Claude Code should create a secret (pick a consistent name), e.g. `socialsense-secrets` containing:
- `HF_TOKEN`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

Notes:
- Claude Code must check what’s already created and update if needed.
- Claude Code must **never print your key values** to logs or markdown.

### Step 4 — Create Modal Volume for model cache
Claude Code should create a persistent cache volume, e.g. `socialsense-model-cache`, mounted at:
- `/root/.cache/huggingface`
- optionally `/root/.cache/torch`
- optionally `/root/.cache/transformers`

### Step 5 — Add Modal server code to this repo
Claude Code must implement a new file at repo root:
- `modal_app.py`

#### `modal_app.py` requirements
- Defines a Modal `App`, builds an image with:
  - system deps for OpenCV/mediapipe (headless is fine server-side)
  - python deps: `server/requirements.txt` plus:
    - `fastapi`, `uvicorn` (or starlette) for WebSocket handling
    - any Modal-required packages
  - **CUDA torch** appropriate for Modal GPU runtime
- Adds the local `server/` package into the image (copy it in, don’t rely on host mounts for deployment).
- Exposes a **WebSocket endpoint** that:
  - accepts **binary** protobuf `ClientMessage` payloads
  - returns **binary** protobuf `ServerMessage` payloads
  - uses the existing `PipelineOrchestrator` to process frames
- Must NOT use the local cv2 “dashboard window” inside Modal (no GUI). Keep GUI only for local debug tools.
- Uses GPU:
  - default: **A10G** (good price/perf for SAM3)
  - allow easy switch to **A100** for faster throughput
- Uses the secrets + volume.

#### Suggested endpoint contract
- WebSocket path: `/ws`
- First message can be ignored; just handle streaming protobuf messages.

### Step 6 — Add a local webcam client that talks protobuf to Modal
Claude Code must ensure there is a working local test path **without Quest**:
- Add a script (suggested path): `tools/webcam_modal_client.py`
  - reads webcam frames
  - packages as protobuf `ClientMessage(frame=...)`
  - sends to the Modal WS URL
  - receives `ServerMessage`
  - overlays masks in an OpenCV window (for “I want to actually see it”)

Alternative:
- Adapt `server/test_client.py` to support a `--url wss://...` remote endpoint and a local overlay mode.

### Step 7 — Document + validate
Claude Code must:
- create/update docs (this file is the main doc)
- run a smoke test:
  - deploy Modal app
  - run local webcam client against the deployed WS endpoint
  - confirm segmentation masks are returned and rendered

---

## GPU recommendation

Start with:
- **`gpu="A10G"`**

Move to A100 if:
- you want higher mask FPS,
- you enable additional heavy models (emotion detection, multi-prompt SAM3 decode, etc.),
- you observe latency spikes.

---

## What should be offloaded to Modal (default)

**Must offload:**
- SAM3 model load + inference (`server/vision/sam3_segmenter.py`)
- All segmentation + tracking logic (`server/pipeline/orchestrator.py`)

**Optional offload (Claude decides based on perf):**
- MediaPipe body detection
- emotion detection
- Gemini Vision calls (if used)
- OpenAI Whisper transcription (if audio is sent to the server)

**Note:** If you want the headset to remain “dumb”, it’s fine to offload Whisper + Gemini reasoning too.  
If you want minimal cloud calls, you can keep Whisper local and send text commands instead.

---

## Final “ask the user” (Claude Code must do this)

After implementing + configuring everything, Claude Code must ask:

> “Modal setup is ready. Do you want me to deploy the Modal GPU server now and run a local webcam test against it?”

If you say yes, Claude Code proceeds to:
- deploy (`modal deploy modal_app.py`)
- print the WebSocket URL
- run the local webcam client pointed at that URL

If you say no, Claude Code stops after summarizing how to deploy later.

