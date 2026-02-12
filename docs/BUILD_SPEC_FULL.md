# SocialSenseAR Full Build Specification

**Purpose:** This document is the single source of truth for a major build that integrates PersonaPlex into the main branch, improves the voice agent with chain-of-thought reasoning, refines masking/color/transparency, smooths borders with minimal latency, and keeps Modal and monitoring options explicit. Use it as full context for any AI or developer working in this workspace.

**Workspace layout (relevant):**
- **Repo root:** `/Users/ajayraj/SAMARSDK`
- **Main app (target of integration):** `ServerBackend/` — server, Modal app, docs
- **PersonaPlex source:** `personaPlex/` — PersonaPlex integration (personaplex_voice_agent, personaplex_bridge, personaplex/moshi)
- **Unity client:** `unitySetUp/SocialSenseAR-Unity` — OverlayRenderer, SegmentOverlay, RLE decode

**Constraint:** All changes must be made **without compromising or worsening any existing functionality.**

---

## 1. PersonaPlex Integration & Chain-of-Thought Voice Agent

### 1.1 Integration into main branch

- **Goal:** Absorb PersonaPlex into the main branch pipeline so that it **replaces** FastWhisper/Whisper for speech input and command handling, while preserving the **exact** pipeline behavior (SAM3 segmentation, effect registry, prompts, RLE, WebSocket protocol, Unity contract).
- **Scope:**
  - **ServerBackend** remains the canonical server. Do **not** require the separate `personaPlex/` server to run; the main branch server should optionally use PersonaPlex as the voice/transcription backend.
  - **Transcription path:** Today the pipeline uses either:
    - `LocalTranscriber` (faster-whisper) or
    - `WhisperTranscriber` (OpenAI Whisper API).
  - **New path:** When PersonaPlex is enabled, use **PersonaPlex** for:
    - Wake word detection (“Hey Vibe” / “Vibe”)
    - Utterance capture (speech → text)
    - Optional spoken replies (speech-to-speech)
    - Command extraction (e.g. `[COMMAND:blur:laptop]` or structured plan)
  - The rest of the pipeline (VoiceCommandPlanner/Gemini for planning, GeminiSceneUnderstanding, SAM3 prompts, effect registry, `_on_voice_command`) should either:
    - Stay as-is when PersonaPlex is “off”, or
    - Be driven by PersonaPlex output (e.g. PersonaPlex emits command tags or a plan that is then executed the same way as today’s VoiceAgent → `_on_voice_command`).

- **Reference implementations to reuse/port:**
  - `personaPlex/server/audio/personaplex_voice_agent.py` — drop-in VoiceAgent using PersonaPlex
  - `personaPlex/server/audio/personaplex_bridge.py` — WebSocket client to PersonaPlex (audio resample, Opus, protocol)
  - PersonaPlex system prompt and `[COMMAND:...]` parsing already exist; align with main branch’s `CommandPlan` and `_on_voice_command(action, targets, effect_type, intensity, invert, color_hex)`.

### 1.2 Hugging Face tokens (env and code)

- **Two distinct tokens:**
  - **`HF_PERSONAPLEX_TOKEN`** — Used only for PersonaPlex (model `nvidia/personaplex-7b-v1`). All code that loads or talks to PersonaPlex (e.g. personaplex bridge, Moshi server client, or any `huggingface_hub` usage for PersonaPlex) must read this (with optional fallback to `HF_TOKEN` for backward compatibility).
  - **`HF_SAM_TOKEN`** — Used only for SAM3 (e.g. `facebook/sam3`). All code that loads SAM3 or other HuggingFace gated models for segmentation must use this (with optional fallback to `HF_TOKEN`).
- **Where to define:**
  - **`.env` (ServerBackend and workspace root):** Add and document:
    - `HF_PERSONAPLEX_TOKEN=`
    - `HF_SAM_TOKEN=`
  - **`.env.example`:** Include both with empty values and a short comment (PersonaPlex vs SAM3).
  - **`server/config.py`:** Add optional fields, e.g. `hf_personaplex_token`, `hf_sam_token`, loaded from env (e.g. `os.getenv("HF_PERSONAPLEX_TOKEN")` or `os.getenv("HF_TOKEN")` fallback for PersonaPlex; same for SAM).
  - **Orchestrator / transcriber wiring:** When creating the PersonaPlex bridge or any HF client for PersonaPlex, pass the PersonaPlex token. When initializing the SAM3 segmenter (or any HF download for SAM3), ensure the environment or config uses `HF_SAM_TOKEN` (e.g. set `HF_TOKEN` in the process only for the SAM3 load, or pass token into the loader).
  - **Modal:** Document that the Modal secret (e.g. `socialsense-secrets`) should contain both `HF_PERSONAPLEX_TOKEN` and `HF_SAM_TOKEN` (or at least one, with fallback to `HF_TOKEN`). In `modal_app.py`, when building the image or running the pipeline, inject the appropriate token for each component (PersonaPlex vs SAM3). Log “SET”/“MISSING” for each at startup so the operator can verify.
- **Dependencies:** Ensure `requirements.txt` and Modal image include any PersonaPlex-specific deps (e.g. `opus` lib, `huggingface-hub`) already required by the personaPlex folder; no duplicate or conflicting HF usage.

### 1.3 Chain-of-thought reasoning voice agent

- **Goal:** The voice agent should behave like a **reasoning** agent that asks follow-up questions when the user’s intent is ambiguous (especially for segmentation), and only proceeds when there is **reasonable confidence** — without forcing the user to be overly specific when the request is already clear.
- **Behavior:**
  - **Ambiguous segment request (e.g. “can you segment that out”):** The agent should **follow up** with clarifying questions (e.g. “Can you be more specific about which object you mean?”) and only trigger segmentation once the target is clear enough for Gemini/SAM3.
  - **Clear implicit request (e.g. “it’s too bright”):** **No** follow-up; interpret as “dim/blur lights, windows, screens” and apply the appropriate masks/effects (dim, blur, etc.) using existing logic (full-screen or object-based). Do not ask “what would you like to dim?” when the intent is obvious.
- **Implementation direction:**
  - If using PersonaPlex: Extend the PersonaPlex system prompt (and any structured output or command protocol) so that it can emit either:
    - A **command** (execute immediately), or
    - A **clarification question** (text/audio to the user, no pipeline command yet).
  - If using Gemini for planning: Add a “confidence” or “needs_clarification” branch: when the planner decides the request is ambiguous (e.g. “that”, “it”, no clear object list), return a short follow-up question and do **not** call `_on_voice_command` until a subsequent utterance resolves the target.
  - Ensure the conversation state (e.g. `get_conversation_state()`) can carry “pending clarification” and the follow-up text so the client (Unity/dashboard) can show or speak it.
- **Do not:** Require the user to be overly specific when the request is already actionable (e.g. “it’s too bright” → act; “segment that” → ask once, then act when they answer).

---

## 2. Pipeline: Correct Masks, Color, Transparency, Custom Masks, Full-Screen Filters

### 2.1 Right mask type and right objects

- **Goal:** The pipeline must apply the **appropriate effect type** to the **correct objects** implied by the user’s request (and by Gemini’s interpretation). No wrong object masking; no wrong effect for the request.
- **Current pieces:** SAM3 produces segments from text prompts; the effect registry maps labels to `effect_type`, `intensity`, `invert`, `color_hex`. `_on_voice_command` adds prompts and sets effects. Gemini (VoiceCommandPlanner, GeminiSceneUnderstanding) interprets natural language and identifies targets.
- **Improvements:**
  - Strengthen **natural language understanding** so that “blur the screen”, “dim the lamp”, “highlight the person”, “make the wall blue” map to the correct labels and effect types. Use existing `VoiceCommandPlanner` and scene understanding; improve prompts and parsing so that object lists and effect types are aligned.
  - **Gemini Flash Vision:** Use Gemini Vision (e.g. Gemini Flash) to **evaluate** whether the applied mask/effect “looks correct” when needed (e.g. after applying a color overlay, optionally send a thumbnail to Gemini: “Is this mask covering the intended object and is the color right?”). This can be a background or on-demand check to course-correct; do not block the main loop on it.
  - **Real-time course correction:** If Gemini Vision is used for verification, run it asynchronously and only adjust (e.g. prompt refinement, or intensity/color tweak) when the model indicates a clear mismatch; avoid flicker or constant re-planning.

### 2.2 Color and transparency

- **Color quality:** The system must **distinguish** between requests like “make it blue”, “dark blue”, and “very dark blue” with correct RGB and intensity. Current logic in `server/audio/voice_agent.py` and `server/commands.py` (e.g. `extract_color_from_text`, `adjust_color_brightness`, brightness modifiers) must be preserved and **improved** so that:
  - “Blue” → correct blue hex; “dark blue” / “very dark blue” → same hue with reduced brightness (e.g. darkness multipliers, not washed out).
  - Gemini planning prompt should include **explicit RGB rules and examples** (e.g. “very dark blue” → hex with low luminance; “bright red” → high luminance red).
- **Transparency:** Overlays are currently **too transparent**. Increase default opacity so that:
  - **Server:** Default intensity and any alpha-related defaults are raised where they influence the effect (e.g. color effect default intensity 1.0; dim/highlight defaults such that overlays are visibly stronger).
  - **Unity:** In `OverlayRenderer.cs`, increase the alpha values used for overlay drawing (e.g. dark base, light base, dim/highlight/global/segment alpha) so the default look is less translucent while still allowing intensity to scale down when the user says “slightly dim” or similar.
- **Gemini as full-context agent:** Gemini (planning + scene understanding) should have **full context** of:
  - The current workflow (listening vs recording, pending clarification, last command).
  - Available masks and effect types (blur, dim, pixelate, highlight, outline, color, full-screen filters).
  - Current prompts and effect registry (what’s active).
  - User’s goal (from the last utterance and history).
  Provide this context in the planner and scene-understanding prompts so that Gemini’s outputs align with the pipeline’s capabilities and state.

### 2.3 Custom masks and full-screen filters

- **Custom masks/filters:** Ensure **custom masks and filters** from the existing design (see `docs/CUSTOM_MASK_SUGGESTIONS.md`, `docs/EFFECT_SYSTEM_EXPANSION.md`) are **supported in the pipeline** and exposed to the voice agent (e.g. “frosted glass”, “redact”, “grayscale”, “spotlight”). If some are not yet implemented in Unity, document them as “voice-parsable but effect type X” and implement the Unity side as needed.
- **Full-screen filters:** When the request is clearly global (e.g. “dim everything”, “it’s too bright”), apply **full-screen** filters (dim, warm, cool, night, or custom color) as appropriate. Ensure the pipeline and Unity support full-screen filter parameters (e.g. `full_screen_filter`, `full_screen_intensity`, `full_screen_color`) and that the voice agent and PersonaPlex command mapping emit these when appropriate.

---

## 3. Borders: Smooth, Contiguous, Minimal Latency

### 3.1 Goals

- **Smooth borders:** Mask edges must be **smooth** (no jagged or sparse outlines). All masks should feel **contiguous** with **accurate** boundaries and **minimal latency** in the end-to-end process.
- **No sparse or broken masks:** Avoid scattered small fragments or “sparkly” masks; prefer coherent regions.

### 3.2 Where to act (additive, no regressions)

- **Server (ServerBackend):**
  - **RLE encoding** (`orchestrator._encode_rle_all`): Already uses resize, morphology (MORPH_CLOSE, MORPH_OPEN), Gaussian blur, threshold. Use existing config knobs `rle_edge_blur_kernel`, `rle_smooth_edges_only` (see `docs/BORDER_SMOOTHNESS_AND_LATENCY_SUGGESTIONS.md`). Tune defaults only if it improves smoothness without worsening latency or compatibility; otherwise keep as optional.
  - **Segmenter:** Ensure SAM3 (or legacy segmenter) output is not over-fragmented; minimum area and any mask refinement should reduce spurious small masks. Do not remove existing behavior; only tune parameters or add optional post-processing.
- **Unity:**
  - **OverlayRenderer:** Optional soft outline (distance-to-edge alpha) and optional overlay edge blur (see BORDER_SMOOTHNESS_AND_LATENCY_SUGGESTIONS.md). Defaults should keep current behavior unless explicitly changed for this build.
  - **RLE decode:** Keep format unchanged; optional improvements (e.g. worker-thread decode, copy to main thread) are additive.
- **Latency:** Avoid extra round-trips or heavy work in the hot path. Cache SAM results; keep pre-decode and cached result return as-is. Any new step (e.g. Gemini Vision check) must be async or batched so it does not block frame delivery.

---

## 4. Modal and Compute

- **Current:** Modal app in `ServerBackend/modal_app.py` uses `modal_config.py` (GPU type, e.g. A10G/A100, volume, secret name `socialsense-secrets`). Secrets currently expect `HF_TOKEN`, `OPENAI_API_KEY`, `GEMINI_API_KEY`.
- **After build:** Secrets should support `HF_PERSONAPLEX_TOKEN` and `HF_SAM_TOKEN` as above; document in `docs/RUNNING.md` and in `modal_config.py` comments.
- **Compute for lower latency:** If, after implementing the above, the team determines that **higher or different compute** is needed for lower latency (e.g. A100 instead of A10G, or different scaledown), **do not** change the Modal GPU or timeout by default. Instead, **output a clear recommendation** at the end of the build, e.g.:
  - “**Modal compute suggestion:** For lower latency, consider setting `MODAL_GPU=A100` (or L4/A10G) and/or increasing `CONTAINER_IDLE_TIMEOUT` so cold starts are less frequent. Approve and apply in `modal_config.py` and redeploy.”
  So the user can **approve** any compute change explicitly.

---

## 5. Omni-Monitoring the Visual Feed (Suggestions Only)

- **Goal:** Provide **suggestions** for how to set up a way for an AI (or automated system) to **monitor the visual feed** and make **dynamic edits** without the user having to say what to fix.
- **Possible approaches (to document, not necessarily implement in this build):**
  - **Periodic snapshot + Gemini Vision:** A background task (e.g. every N seconds) captures a frame, sends it to Gemini with a prompt like “Given the user’s preferences (e.g. reduce clutter, protect privacy), suggest or apply effect changes.” Results could drive prompt/effect updates (e.g. “blur screens”, “dim lamp”) via the same `set_effect` / `add_prompt` API.
  - **Dashboard / external monitor:** A small dashboard or external tool that shows the current overlay + passthrough and allows an operator (or another AI) to send text commands (e.g. “blur laptop”, “dim everything”) into the same WebSocket or REST API used by the voice agent.
  - **Preference profile:** Store user preferences (e.g. “always dim bright windows”, “blur all screens in meetings”); a background policy could apply these when the scene matches (e.g. when “screen” or “window” is detected and preference is set).
  - **Logging + replay:** Log frame IDs and effect decisions so that “what went wrong” can be replayed and tuned offline; this supports iterative improvement by an AI or developer without live user input.
- **Deliverable:** A short section in a doc (e.g. `docs/OMNI_MONITORING_SUGGESTIONS.md` or a subsection in this BUILD_SPEC_FULL.md) that lists 2–3 concrete options with one paragraph each, so the user can choose and implement later.

---

## 6. Implementation Checklist (Summary)

- [ ] **PersonaPlex in main branch:** Wire PersonaPlex as optional voice backend (transcriber path); keep faster-whisper/Whisper as fallback. Use `personaPlex/server/audio/personaplex_voice_agent.py` and `personaplex_bridge.py` as reference; ensure CommandPlan and `_on_voice_command` receive the same semantics.
- [ ] **HF tokens:** Introduce `HF_PERSONAPLEX_TOKEN` and `HF_SAM_TOKEN` in `.env`, `.env.example`, config, and all code that uses Hugging Face for PersonaPlex vs SAM3. Update Modal secret docs and startup logs.
- [ ] **Chain-of-thought:** Add clarification flow (follow-up questions for ambiguous “segment that”); no follow-up for clear requests like “it’s too bright”. Expose pending clarification in conversation state.
- [ ] **Masks and color:** Improve NL understanding and Gemini context; add optional Gemini Vision check for mask/color correctness; fix “blue” vs “dark blue” vs “very dark blue”; raise default transparency (server + Unity).
- [ ] **Custom masks & full-screen:** Ensure custom effect types and full-screen filters are in the pipeline and voice-parsable; Unity support where specified.
- [ ] **Borders and latency:** Apply additive smoothness options (blur kernel, soft outline, etc.) per BORDER_SMOOTHNESS_AND_LATENCY_SUGGESTIONS; no regressions; no extra blocking work in hot path.
- [ ] **Modal:** Document secret keys; at end of build, if needed, output a clear **Modal compute suggestion** for the user to approve.
- [ ] **Omni-monitoring:** Add a short “Omni-Monitoring Suggestions” section (or doc) with 2–3 options for dynamic visual-feed monitoring and edits.

---

## 7. File Reference (Key Paths)

| Area | Path |
|------|------|
| Server config | `ServerBackend/server/config.py` |
| Orchestrator | `ServerBackend/server/pipeline/orchestrator.py` |
| Voice agent (current) | `ServerBackend/server/audio/voice_agent.py` |
| PersonaPlex voice agent | `personaPlex/server/audio/personaplex_voice_agent.py` |
| PersonaPlex bridge | `personaPlex/server/audio/personaplex_bridge.py` |
| Local transcriber | `ServerBackend/server/audio/local_transcriber.py` |
| Commands / color | `ServerBackend/server/commands.py` |
| Scene understanding | `ServerBackend/server/vision/gemini_scene_understanding.py` |
| SAM3 segmenter | `ServerBackend/server/vision/sam3_segmenter.py` |
| Modal app | `ServerBackend/modal_app.py` |
| Modal config | `ServerBackend/modal_config.py` |
| Unity overlay | `unitySetUp/SocialSenseAR-Unity/Assets/Scripts/OverlayRenderer.cs` |
| Border/latency suggestions | `ServerBackend/docs/BORDER_SMOOTHNESS_AND_LATENCY_SUGGESTIONS.md` |
| Custom masks | `ServerBackend/docs/CUSTOM_MASK_SUGGESTIONS.md` |
| Effect expansion | `ServerBackend/docs/EFFECT_SYSTEM_EXPANSION.md` |
| Running / secrets | `ServerBackend/docs/RUNNING.md` |

---

*End of build specification. Use this document as full context when implementing the described changes in this workspace.*
