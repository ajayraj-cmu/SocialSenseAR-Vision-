# Gemini Vision–Driven Pipeline Build Specification

**Purpose:** This document is a comprehensive build prompt for implementing a Gemini Vision–centric pipeline that elevates scene understanding, command interpretation, and visual verification. It preserves all existing functionality while making Gemini Vision the primary decision driver.

**Constraint:** Do not compromise or remove any existing behavior. All changes are additive or improve upon current logic.

---

## 1. Executive Summary

### Current Limitation

The pipeline today uses **Gemini text-only** for voice command planning. It receives:
- A list of objects SAM3 has already segmented (known_objects)
- Preloaded natural language patterns and examples
- No direct visual context of the scene

**Consequences:**
- Generic requests like "it's too bright" rely on hardcoded heuristics (dim light/screen/window) instead of dynamically identifying what is actually bright in *this* scene
- "Very dark green" / "darker green" collapse to near-black because there is no reference to the *current* on-screen color
- Color intensity and opacity choices are not verified against user intent
- No mechanism to discover and direct SAM3 to segment *additional* objects visible in the scene but not yet prompted

### Proposed Architecture

**Gemini Vision as the central decision driver:**

1. **Phase 1 (pre-command, initial state analysis):** When a voice command is received, run a **sufficient collection of Vision calls** to analyze the scene before any masking or filtering is applied. Take scene snapshots, send them to Gemini Vision, and let Gemini identify objects, infer targets, and produce a structured plan (targets, effects, colors, filters). Masking and filters are applied **only after** this informed plan is ready.
2. **Phase 2 (post-SAM3, verification and correction):** After SAM3 runs and overlays are rendered, take verification snapshots and send them to Gemini Vision. Gemini checks whether the result matches intent. **Continue correcting** (apply corrections, re-run SAM3 if needed, re-verify) until the persisted state is correct — but only for meaningful mismatches; avoid excessive refinement. Cap the correction loop (2–3 rounds) to balance correctness with cost.
3. **Trigger model:** Run Gemini Vision only **on command** — not every frame. SAM3 and the effect registry sustain state between commands.
4. **Persistent state:** Maintain a short-term memory of the last applied state (per-target effects, colors, opacities) so incremental commands like "make the chair darker green" can be resolved relative to prior "make the chair green".

---

## 2. Tool Inventory (Gemini Must Be Aware)

Gemini Vision must know the full set of tools available so it can reason about combinations and coordination.

### 2.1 Asset Masks (Per-Segment Effects)

| Effect | Description | Parameters |
|--------|-------------|------------|
| `blur` | Gaussian blur on passthrough | intensity (0.0–1.0) |
| `dim` | Semi-transparent black overlay | intensity |
| `pixelate` | Block/mosaic pattern | intensity |
| `highlight` | Warm yellow brightening | intensity |
| `outline` | Colored border | intensity |
| `color` | RGB color overlay | color_hex, intensity (opacity) |

**Color precision:** Gemini can specify exact RGB for Unity: `color_hex` as `#RRGGBB` or `#RRGGBBAA` (with alpha). Example: `#2D5016` for a specific dark green, `#2D5016CC` for 80% opacity.

**Opacity / translucency:**
- `intensity` (0.0–1.0) maps to overlay opacity (e.g., 0.3 = subtle, 1.0 = fully opaque)
- For color effects, `#RRGGBBAA` allows per-color alpha

### 2.2 Full-Screen Filters

| Filter | Description | Parameters |
|--------|-------------|------------|
| `dim` | Global darkening | intensity |
| `warm` | Warm color cast | intensity |
| `cool` | Cool color cast | intensity |
| `night` | Night-mode darkening | intensity |
| `grayscale` | Desaturate entire view | intensity |
| `color` | Tint entire view | color_hex, intensity |

**Coordination rule:** Full-screen filters and asset masks must **complement** each other. If a full-screen dim is active, asset masks (e.g., highlight on person) should remain visible and not be obscured. Gemini should avoid combinations that fully obscure important masked regions.

### 2.3 Inverted Effects

- **Invert:** Effect applies to everything *except* the target(s) (e.g., "blur everything but me" → blur on background, person stays clear).

---

## 3. Two-Phase Vision Flow

### 3.0 Design Principles: Initial Analysis and Corrective Iteration

**Initial state analysis (Phase 1):** Execute a **sufficient collection of Gemini Vision calls** before any masking or filtering is applied. The pipeline must not commit to effects until Vision has had adequate opportunity to analyze the scene — identifying visible objects, inferring targets from context (e.g., "it's too bright" → lamp, window, monitor), and producing a well-informed plan. A single comprehensive Phase 1 call is typically sufficient; for complex or ambiguous commands, consider a follow-up refinement call if the first plan is uncertain. The goal: masking and filters are driven by **informed decisions**, not guesswork.

**Verification and correction (Phase 2):** After SAM3 runs and effects are applied, use Gemini Vision to verify the result. **Continue correcting until the persisted state accurately reflects the user's intent** — but apply corrections with restraint. Only propose corrections when there is a **meaningful mismatch** (wrong targets, visibly incorrect color/intensity, filters obscuring important regions). Avoid nitpicking or excessive refinement that would trigger unnecessary API calls. Cap the correction loop (e.g., 2–3 rounds) to balance correctness with cost and latency; within that cap, iterate until verified or the cap is reached.

### 3.1 Phase 1: Pre-Command Scene Snapshot

**Trigger:** User utters a command (voice or typed) that requires scene understanding.

**Inputs:**
- Raw camera frame (BGR) at time of command
- User utterance
- Persistent state (last command, per-target effects, active full-screen filter)
- Known objects from SAM3 (if any) — for reference, not as sole source

**Gemini Vision prompt (conceptual):**

```
You see the user's current AR view. The user said: "{utterance}"

Current state:
- Active effects per object: {active_effects}
- Full-screen filter: {full_screen_filter}
- Last command: {last_command}

Your task:
1. Identify ALL visible objects in the scene (not limited to what SAM3 has segmented).
2. For generic requests ("it's too bright", "dim the room"), infer which objects in THIS scene should be targeted.
3. For color requests ("make the chair green", "darker green"), use the prior state to interpret relative terms:
   - "darker green" = same hue, lower luminance than current chair color
   - "very dark green" = significantly darkened green (provide exact hex, e.g. #1a3d0a)
4. Output exact color values when appropriate: suggest specific #RRGGBB or #RRGGBBAA for Unity.

Output JSON:
{
  "targets": ["chair", "desk"],
  "effect_type": "color",
  "color_hex": "#2D5016",
  "intensity": 0.85,
  "action": "add" | "remove" | "change",
  "invert": false,
  "full_screen_filter": "none" | "dim" | "warm" | ...,
  "full_screen_intensity": 0.5,
  "full_screen_color": "#...",
  "new_prompts_for_sam": ["chair", "desk"],
  "reasoning": "..."
}
```

**Key behaviors:**
- **New SAM3 prompts:** If Gemini identifies objects not yet in SAM3 prompts (e.g., "blur the lamp" but lamp wasn’t segmented), add them to `new_prompts_for_sam` so the pipeline can direct SAM3 to segment them.
- **Exact RGB:** For "dark green", "darker green", "forest green", etc., Gemini outputs a precise hex. No collapsing to near-black unless the user explicitly requests "extremely dark" or "black".
- **Full-screen + masks:** If applying a full-screen filter, ensure asset masks (e.g., highlight on person) remain visible; suggest intensity/filter type that complements rather than obscures.

### 3.2 Phase 2: Post-SAM3 Verification Snapshot

**Trigger:** After SAM3 has run, effects applied, and overlays rendered. Execute once per command, not per frame.

**Inputs:**
- Composite frame: camera view with overlays applied (as the user would see it)
- The plan that was executed (targets, effect_type, color_hex, intensity, full_screen_filter)
- User utterance

**Gemini Vision prompt (conceptual):**

```
You see the AR view AFTER the following was applied:
- User said: "{utterance}"
- Executed: {effect_type} on {targets}, color={color_hex}, intensity={intensity}, full_screen={full_screen_filter}

Does the result match the user's intent?
1. Are the correct objects covered?
2. Is the color/intensity/opacity appropriate? (e.g., "dark green" should look like dark green, not black)
3. Do full-screen filters and asset masks work well together (no accidental obscuring)?

Only flag corrections for meaningful mismatches (wrong targets, obviously wrong color/intensity, filters obscuring intent). Do not nitpick minor aesthetic differences.

If correct: output {"verified": true}.
If not: output {
  "verified": false,
  "corrections": {
    "targets": [...],
    "color_hex": "#...",
    "intensity": 0.8,
    "full_screen_filter": "none" | ...
  },
  "reasoning": "..."
}
```

**Self-correction:** When `verified: false`, the pipeline applies the correction plan (update effect registry, re-run SAM3 if targets changed, etc.) and runs verification again. **Iterate until** either (a) `verified: true` and the persisted state is correct, or (b) a maximum correction round cap (e.g., 2–3) is reached. Corrections should target **meaningful** mismatches only — wrong objects, obviously wrong colors, or filters that obscure intent. Avoid over-correction for minor visual differences.

---

## 4. Persistent State for Incremental Commands

### 4.1 State to Persist (Between Commands)

| Field | Description |
|-------|-------------|
| `last_utterance` | Last user command text |
| `per_target_state` | Map of label → {effect_type, color_hex, intensity, ...} |
| `full_screen_filter_state` | {type, intensity, color_hex} |
| `last_sam_prompts` | Prompts currently used for SAM3 |

### 4.2 Resolving Relative Terms

- **"darker green"** → Look up chair’s current `color_hex` (e.g. `#3d7a1f`), reduce luminance by a reasonable step (e.g. 20%), output `#2D5016`.
- **"more opaque"** → Increase `intensity` for that target.
- **"lighter"** → Increase luminance of current color or increase intensity.
- **"stop dimming"** / **"normal"** → Clear full-screen filter and/or remove effects.

### 4.3 Context Lifecycle

- **Persist:** After each command execution, save the resulting state (effects, colors, filters) into the persistent state store.
- **Clear reasoning context:** After each run, clear the Gemini conversation/reasoning buffer to avoid token bloat. Only the *structured state* (effects, colors) is kept.
- **Use in next command:** When the next utterance arrives, pass the persistent state into the Phase 1 prompt so Gemini can resolve "darker", "lighter", "more", etc.

---

## 5. Implementation Guidelines

### 5.1 Where to Integrate

| Component | Responsibility |
|-----------|----------------|
| `VoiceAgent` / `VoiceCommandPlanner` | Orchestrate two-phase flow: on utterance, request Phase 1 snapshot, then execute plan, then request Phase 2 verification |
| `GeminiSceneUnderstanding` | Extend with `plan_from_vision(utterance, frame_bgr, persistent_state)` (Phase 1) and `verify_and_correct(frame_with_overlay, plan, utterance)` (Phase 2) |
| `PipelineOrchestrator` | Ensure `new_prompts_for_sam` from Phase 1 are added to SAM3 prompts before next segment run |
| `Orchestrator` / `VoiceAgent` | Maintain persistent state store, clear reasoning context after each run |

### 5.2 Snapshot Sources

- **Phase 1:** Use the most recent decoded frame from the pipeline (same resolution as SAM3 input). No extra capture needed.
- **Phase 2:** Either (a) request a composite from the client (Unity renders and sends back), or (b) simulate composite server-side by overlaying masks on the frame (approximation). Prefer (a) for accuracy if feasible.

### 5.3 Async and Non-Blocking

- Phase 1 can block the command path briefly (user expects a short delay when giving a command).
- Phase 2 verification should run asynchronously. If a correction is needed, apply it and send updated state to the client; avoid blocking the main frame loop.
- **Correction loop:** Allow enough iterations (e.g., 2–3 rounds) so that SAM3 and effects converge to the correct persisted state. Cap at 2–3 to avoid excessive API usage; within the cap, iterate until verified or cap reached. Corrections should address meaningful mismatches only — not minor aesthetic differences.

### 5.4 Fallbacks

- If Gemini Vision is unavailable, fall back to the current text-only planner (regex fast-path + Gemini text).
- If verification fails or times out, keep the applied state; do not roll back unless the correction plan is confidently produced.

---

## 6. Exact RGB and Color Refinement

### 6.1 Gemini Output Format

Allow Gemini to output:

```json
{
  "color_hex": "#2D5016",
  "color_rgb": {"r": 45, "g": 80, "b": 22},
  "intensity": 0.85
}
```

Unity and the server should prefer `color_hex` when present. `color_rgb` can be used for logging or as a fallback to construct hex.

### 6.2 Color Modifier Rules (for Gemini)

| User phrase | Interpretation | Example output |
|-------------|----------------|----------------|
| "green" | Base green | #00FF00 or #228B22 (forest) |
| "dark green" | Reduced luminance | #006400, #1a3d0a |
| "darker green" | Relative to current; one step darker | If current #2D5016 → #1a300e |
| "very dark green" | Significantly dark | #0d260d, #1a3d0a |
| "extremely dark" | Near black but hue preserved | #051005 |
| "bright green" | High luminance | #00FF00, #7FFF00 |
| "pale green" | Desaturated, light | #98FB98, #90EE90 |

**Critical:** "Very dark green" must remain **visibly green**. Never collapse to #000000 unless the user says "black" or "invisible".

---

## 7. Full-Screen Filter and Mask Coordination

### 7.1 Coordination Rules

- If a full-screen dim is applied and the user has "highlight person", the person mask should stay visible (highlight is typically a brightening overlay).
- If full-screen grayscale is applied, colored asset masks can still show color on top (layer order).
- Gemini should avoid recommending full-screen filters that would make important masked regions hard to see (e.g., "dim everything" when the user wants to see a highlighted object clearly).

### 7.2 Expansion of Filters and Masks

Gemini Vision can suggest new effect types or combinations based on the scene. The build should:
- Document which effect types are supported in Unity and which are "voice-parsable but pending implementation".
- When Gemini suggests an unsupported effect, map it to the closest supported one and log the suggestion for future expansion.
- Allow the prompt to describe custom effects (e.g., "frosted glass", "redact") so that future Unity support can be wired without changing the prompt structure.

---

## 8. End-to-End Command Flow (Target State)

```
User: "Hey Vibe, it's too bright"
  → Phase 1 (initial state): Snapshot current frame
  → Gemini Vision: Analyze scene, identify bright sources (lamp, window, monitor)
  → Plan: full_screen_filter=dim, targets=[lamp, window, monitor], effect_type=dim
  → Add [lamp, window, monitor] to SAM3 prompts if not already present
  → SAM3 segments frame with new prompts
  → Apply effects
  → Phase 2 (verification): Snapshot frame with overlays
  → Gemini Vision: Verify dimming looks correct
  → If verified: Done. Persist state. Clear reasoning context.
  → If not: Apply correction, re-verify. Repeat until verified or cap (2–3 rounds).

User: "Make the chair green"
  → Phase 1: Snapshot, Gemini identifies chair
  → Plan: targets=[chair], effect_type=color, color_hex=#228B22, intensity=0.85
  → SAM3 segments chair, apply color
  → Phase 2: Verify
  → Persist: chair → {color_hex: #228B22, intensity: 0.85}

User: "Make the chair darker green"
  → Phase 1: Snapshot + persistent state (chair is #228B22)
  → Gemini: "darker" = reduce luminance, output #1a5c17 or similar
  → Plan: targets=[chair], effect_type=color, color_hex=#1a5c17, intensity=0.85
  → Apply, verify, persist
```

---

## 9. File Reference

| Area | Path |
|------|------|
| Voice agent | `ServerBackend/server/audio/voice_agent.py` |
| Scene understanding | `ServerBackend/server/vision/gemini_scene_understanding.py` |
| Orchestrator | `ServerBackend/server/pipeline/orchestrator.py` |
| Effect application | `Orchestrator._apply_effects`, voice agent `_active_effects` |
| Color parsing | `ServerBackend/server/commands.py` |
| Unity overlay | `unitySetUp/SocialSenseAR-Unity/Assets/Scripts/OverlayRenderer.cs` |
| Color utility | `unitySetUp/SocialSenseAR-Unity/Assets/Scripts/ColorUtility.cs` |
| Full-screen filter | `unitySetUp/SocialSenseAR-Unity/Assets/Scripts/FullScreenFilterEffect.cs` |
| Config | `ServerBackend/server/config.py` |

---

## 10. Checklist

- [ ] Extend `GeminiSceneUnderstanding` with `plan_from_vision(utterance, frame_bgr, persistent_state)` returning a structured plan (targets, effects, colors, full-screen filter, new_prompts_for_sam).
- [ ] Extend `GeminiSceneUnderstanding` with `verify_and_correct(frame_with_overlay, plan, utterance)` returning verified flag and optional correction plan.
- [ ] Add persistent state store (per-target effects, full-screen filter, last utterance) in `VoiceAgent` or `Orchestrator`.
- [ ] Ensure sufficient Phase 1 analysis before applying masks/filters: capture frame(s), run initial Vision call(s) to produce an informed plan, then apply (no effects until plan is ready).
- [ ] Integrate Phase 2 into command execution path: after effects applied, capture/simulate composite, call `verify_and_correct`, apply corrections, re-verify until persisted state is correct or cap (2–3 rounds) is reached.
- [ ] Update Gemini prompts with full tool inventory (masks, filters, opacities, invert).
- [ ] Add explicit RGB output to plan schema (`color_hex`, optional `color_rgb`).
- [ ] Implement relative term resolution ("darker green") using persistent state in the Phase 1 prompt.
- [ ] Clear reasoning context after each command run; persist only structured state.
- [ ] Preserve fallback to text-only planner when Gemini Vision is unavailable.
- [ ] Document coordination rules for full-screen filters and asset masks.
- [ ] Add tests for "darker green", "it's too bright" (dynamic targets), and verification correction flow.
- [ ] Implement rate-limit defenses: throttle by `gemini_min_interval`, cap RPM, retry with exponential backoff on 429, cap verification loop.
- [ ] On vision failure or rate limit after retries, fall back to text-only planner (never crash).

---

## 11. Implementation Notes for Claude / Developers

### 11.1 Phase 2 Composite Frame

For verification to be accurate, Gemini must see what the user sees. Two approaches:

1. **Server-side simulation:** Overlay RLE-decoded masks onto the frame using the same effect logic (blur, dim, color) as Unity. This is an approximation but avoids protocol changes.
2. **Client-side capture (preferred):** Add an optional "verification frame" message type: Unity captures a screenshot of the Game view (with overlays rendered), encodes as JPEG, and sends it to the server when a command completes. The server uses this for Phase 2. Requires a small protocol extension.

Start with (1) for simplicity; migrate to (2) if verification quality is insufficient.

### 11.2 Gemini API Usage

- Use `gemini-2.0-flash-exp` or `gemini-2.5-flash` for vision (or latest vision-capable model).
- Each command triggers 1–2 vision calls (Phase 1 + Phase 2). With correction, up to 3–4. This is acceptable (commands are infrequent).
- Resize frames to ≤1024px for vision calls to stay within token limits and reduce latency.
- Log vision call count and latency for tuning.

### 11.3 Preserving Existing Fast Paths

- Regex fast-paths ("blur laptop", "clear") should remain. If the utterance matches a fast-path, optionally skip Phase 1 (or run Phase 1 in parallel for richer targets) and proceed with the fast-path plan.
- The goal is to *enhance* planning, not replace fast-paths entirely. Fast-paths reduce latency for common, unambiguous commands.

---

## 12. Cost Estimation (Gemini Vision API)

**Models used:** `gemini-2.5-flash` (vision), `gemini-2.5-flash-lite` (planning). Pricing as of 2025 — verify at [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing).

### 12.1 Per-Call Token Estimates

| Component | Tokens | Notes |
|-----------|--------|-------|
| 1 scene image (resized ≤1024px) | ~250–560 | Depends on resolution; use ~400 for estimates |
| System prompt + tool inventory | ~600–1,000 | |
| User utterance + persistent state | ~100–300 | |
| Output (JSON plan / verification) | ~200–500 | |

### 12.2 Cost Per Vision Call (Gemini 2.5 Flash, Paid Tier)

| Tier | Input | Output |
|------|-------|--------|
| Paid | $0.30 / 1M tokens | $2.50 / 1M tokens |
| Free | Free | Free |

**Typical Phase 1 call:**
- Input: ~1,500 tokens (1 image + prompt) → $0.00045
- Output: ~300 tokens → $0.00075  
- **Total per Phase 1 call: ~\$0.0012**

**Typical Phase 2 call:** Same order of magnitude → **~\$0.0012 per call**

### 12.3 Estimated Cost Per Command

| Scenario | Vision calls | Est. cost (paid) |
|----------|--------------|------------------|
| Happy path (Phase 1 + Phase 2) | 2 | ~\$0.0024 |
| With one correction round | 3–4 | ~\$0.0036–\$0.0048 |
| Fast-path (regex) only, no vision | 0 | \$0 |

**Monthly ballpark (paid tier):** 1,000 commands/month ≈ $2.40–$4.80. With free tier, vision calls are free within quota.

---

## 13. Rate Limiting and Crash Prevention

Vision calls are **sparse** (only on user commands, not per-frame). With `gemini_max_calls_per_minute: 5` and `gemini_min_interval: 6.0`, we stay well below typical limits. Nevertheless, implement defensive measures so rate limits do not cause crashes.

### 13.1 API Limits (Reference)

| Tier | RPM (requests/min) | TPM (tokens/min) |
|------|--------------------|------------------|
| Free | 5–15 (model-dependent) | Varies |
| Paid Tier 1 | 150–300 | Higher |

**Config (`config.py`):** `gemini_max_calls_per_minute: 5`, `gemini_min_interval: 6.0` — enforces ≥6s between calls and caps at 5/min.

### 13.2 Implementation Requirements (Avoid Crashes)

1. **Request throttling:** Before each vision call, enforce `gemini_min_interval` since the last call. Use a simple last-call timestamp; if elapsed < min_interval, sleep or queue.
2. **RPM cap:** Track calls in a sliding 60s window. If at limit, either (a) queue the request and process when a slot frees, or (b) skip Phase 2 verification and apply the Phase 1 plan without verification (degraded but non-crashing).
3. **429 handling:** On `429 ResourceExhausted` or rate-limit errors, retry with exponential backoff (e.g., 2s, 4s, 8s, max 3 retries). Do **not** fail the command — either succeed after retry or fall back to text-only planner.
4. **Verification loop cap:** Limit Phase 2 correction rounds to 2–3. Enough to converge to the correct persisted state; prevents runaway loops that spike call count.
5. **Debounce rapid commands:** If the user fires multiple commands within a few seconds, consider coalescing or serializing (one command at a time) so bursts don’t overwhelm the limit.
6. **Graceful fallback:** If vision is rate-limited or fails after retries, fall back to the existing text-only planner. Never crash; log and continue.

### 13.3 Why Sparse Usage Is Safe

- Vision runs **only on command**, not every frame.
- Typical usage: 1–2 calls per command, a few commands per minute.
- With 5 RPM and 6s spacing, even back-to-back commands are throttled safely.
- Rate-limit crashes are usually caused by (a) unbounded retries, (b) no 429 handling, or (c) per-frame vision. This design avoids all three.

---

*End of Gemini Vision build prompt. Use this document as full context when implementing the described changes. Do not compromise existing functionality.*
