# Omni-Monitoring the Visual Feed — Suggestions

**Goal:** Enable an AI or automated system to monitor the visual feed and make dynamic edits without requiring the user to say what to fix. Below are three concrete options for implementation, ordered by complexity.

---

## Option 1: Periodic Snapshot + Gemini Vision (Recommended First Step)

A background task captures a frame every N seconds (e.g., 5-10s) and sends it to Gemini Vision with a context-aware prompt. The prompt includes the user's stored preferences (e.g., "reduce clutter", "protect privacy", "dim bright objects") and the current effect registry state.

**How it works:**
- A daemon thread in the orchestrator captures `self._latest_frame` on a timer (non-blocking, no extra GPU work).
- The frame is sent to `GeminiSceneUnderstanding.snapshot()` with an extended prompt that includes the user's preference profile.
- Gemini returns suggestions like `["blur screen", "dim lamp", "highlight speaker"]`.
- Suggestions are applied via the existing `set_effect()` / `add_prompt()` API — the same path as voice commands.
- A confidence threshold prevents flicker: only apply changes when Gemini's suggestion confidence > 0.7 and the suggestion differs from the current state.
- Rate-limited to avoid API costs (e.g., max 2 calls/minute).

**Pros:** Minimal code changes; uses existing Gemini infrastructure; no new dependencies.  
**Cons:** Latency (5-10s cycles); Gemini API costs; relies on Gemini's visual reasoning quality.

**Integration point:** Add a `_monitor_loop()` method to `PipelineOrchestrator` that runs alongside `_sam_loop()`. Wire it to a config flag `omni_monitor_enabled: bool = False` and `omni_monitor_interval: float = 8.0`.

---

## Option 2: Dashboard / External Monitor

A small web dashboard (or external tool) that shows the current overlay + passthrough composite and allows an operator (or another AI agent) to send text commands into the same WebSocket or REST API used by the voice agent.

**How it works:**
- Extend `websocket_server.py` (or the Modal FastAPI app) with a `/dashboard` HTML endpoint that shows:
  - A live JPEG stream of the composite frame (overlay + passthrough).
  - Current active prompts and effects (from `get_active_prompts()` and `get_effects()`).
  - A text input box that sends commands via the existing `process_text_command()` path.
- An external AI agent (e.g., a separate Gemini or GPT process) connects via WebSocket and periodically sends adjustment commands based on what it "sees" in the stream.
- The dashboard is read-only by default; operator commands are logged for audit.

**Pros:** Human-in-the-loop option; easy to debug and iterate; works with any external AI system.  
**Cons:** Requires a display/browser; adds network latency if the external monitor is remote; needs a compositing step to generate the preview frame.

**Integration point:** The `server/dashboard.py` file already exists with an OpenCV-based dashboard. Extend it with a web-based alternative using FastAPI + HTMX or a simple HTML page.

---

## Option 3: Preference Profile + Policy Engine

Store user preferences (e.g., "always dim bright windows", "blur all screens in meetings", "highlight the speaker") and apply them automatically when the scene matches.

**How it works:**
- Add a `preferences.json` file (or a config section) with rules like:
  ```json
  {
    "rules": [
      {"trigger": "screen", "effect": "blur", "intensity": 0.8, "context": "always"},
      {"trigger": "window", "condition": "bright", "effect": "dim", "intensity": 0.7},
      {"trigger": "person", "effect": "highlight", "intensity": 0.5, "context": "meeting"}
    ]
  }
  ```
- When SAM3 detects an object matching a trigger label, the policy engine checks if the rule's conditions are met (e.g., "bright" requires Gemini to confirm the object is bright) and applies the effect automatically.
- Rules can be updated via voice commands: "Vibe, always blur screens" → adds a rule.
- The policy engine runs in the SAM loop after segment detection, before RLE encoding — no extra latency.

**Pros:** Zero latency once configured; no API costs for rule-based decisions; personalized experience.  
**Cons:** Rules need to be set up; may not adapt to novel situations without Gemini fallback; requires careful UX to avoid unintended effects.

**Integration point:** Add a `PolicyEngine` class that reads `preferences.json` and is called from `_sam_loop()` after `_apply_effects()`. Add voice command parsing for "always [effect] [target]" → saves rule.

---

## Bonus: Logging + Replay for Offline Tuning

Log frame IDs, SAM3 segment labels, effect decisions, and Gemini reasoning to a structured JSONL file (the `metrics_log_path` config already supports this). This enables:
- **Replay:** Re-run Gemini on logged frames to test different prompts or preference rules offline.
- **Analysis:** Identify which objects are frequently detected, which effects are most used, and where Gemini's reasoning was wrong.
- **Training data:** Use logged frame-label-effect triplets to fine-tune a lightweight policy model that replaces Gemini for common decisions.

The existing `metrics_log` in the orchestrator already writes `{ts, sam_count, sam_ms, total_ms, segs, labels}`. Extend it with `{effects, gemini_reasoning, user_command}` for richer replay.

---

*Choose one option to implement first (Option 1 is recommended), and iterate based on real-world usage. All options are additive and compatible with each other.*
