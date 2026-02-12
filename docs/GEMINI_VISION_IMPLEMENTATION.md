# Gemini Vision-Driven Pipeline — Implementation Guide

**Status:** ✅ **Phase 1 Complete** | 🚧 Phase 2 Verification (Future Enhancement)

This document describes the implementation of the Gemini Vision-centric pipeline enhancement that elevates scene understanding, command interpretation, and visual decision-making.

---

## Executive Summary

### What Changed

The pipeline now uses **Gemini Vision as the primary decision driver** for voice commands, enabling:

1. **Context-aware command interpretation**: "it's too bright" → Vision identifies what IS bright in the current scene (lamp, window, monitor) instead of using hardcoded heuristics
2. **Precise color control**: "very dark green" → Vision outputs exact RGB (#006600) that preserves hue while darkening
3. **Incremental commands**: "make chair green" → "darker green" → Vision uses persistent state to compute relative color adjustments
4. **Dynamic object discovery**: Vision identifies objects not yet segmented and directs SAM3 to add them

### What Was Preserved

**All existing functionality remains intact:**
- Fast-path regex commands (blur laptop, clear, etc.) — **zero latency**
- Text-only Gemini planner fallback when vision unavailable
- MediaPipe body detection
- SAM3 text-prompted segmentation
- Effect registry and full-screen filters
- Conversation state and clarification flow

**Design Principle:** Changes are **purely additive**. Vision enhances planning when available; text-only planning remains fully functional.

---

## Architecture Overview

### Two-Phase Vision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ USER COMMAND: "it's too bright"                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Pre-Command Scene Analysis (Vision-Driven Planning)   │
│                                                                 │
│ 1. Capture current camera frame (no masks applied yet)         │
│ 2. Get persistent state (last colors, effects, prompts)        │
│ 3. Call plan_from_vision(utterance, frame, state) → Gemini     │
│ 4. Vision identifies bright objects in THIS scene:             │
│    → ["lamp", "window", "monitor"]                             │
│ 5. Outputs structured plan:                                    │
│    {                                                            │
│      "targets": ["lamp", "window", "monitor"],                 │
│      "effect_type": "dim",                                      │
│      "intensity": 0.7,                                          │
│      "full_screen_filter": "dim",                              │
│      "new_prompts_for_sam": ["lamp", "window", "monitor"]      │
│    }                                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ EXECUTION                                                       │
│                                                                 │
│ 1. Add new_prompts_for_sam to SAM3 (lamp, window, monitor)    │
│ 2. Apply effects to targets                                    │
│ 3. Update persistent state for incremental commands            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Post-Execution Verification (Future Enhancement)      │
│                                                                 │
│ 1. Capture frame WITH overlays applied                         │
│ 2. Call verify_and_correct(frame, plan, utterance) → Gemini    │
│ 3. Vision checks: correct objects? appropriate color/intensity?│
│ 4. If mismatch: propose corrections, apply, re-verify          │
│ 5. Iterate until verified or max_corrections (2-3) reached     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. Enhanced GeminiSceneUnderstanding

**File:** `ServerBackend/server/vision/gemini_scene_understanding.py`

#### New Methods

**`plan_from_vision(utterance, frame_bgr, persistent_state) → dict`**

Phase 1 vision call that analyzes the scene BEFORE applying effects:
- Encodes frame as JPEG (resized to ≤1024px)
- Sends to Gemini Vision with full context:
  - Current active effects per object
  - Full-screen filter state
  - Last command (for relative terms)
  - Current SAM3 prompts
- Returns structured plan with targets, effects, colors, new prompts

**`verify_and_correct(frame_with_overlay, executed_plan, utterance, max_corrections) → dict`**

Phase 2 verification (infrastructure ready, not yet wired):
- Analyzes post-execution frame with overlays
- Checks if result matches intent
- Returns verified flag + corrections if needed

**`set_rate_limits(min_interval, max_calls_per_minute)`**

Configures rate limiting from ServerConfig.

#### Rate Limiting & Error Handling

**`_check_rate_limit() → bool`**

Enforces:
- Min interval (6s by default) between calls
- RPM limit (5 calls/min by default)

**`_call_gemini_vision_with_retry(prompt, image, max_retries=3)`**

Handles 429 errors with exponential backoff:
- Retry 1: wait 1s
- Retry 2: wait 2s
- Retry 3: wait 4s
- After retries exhausted: return None (triggers fallback)

---

### 2. VoiceAgent Persistent State

**File:** `ServerBackend/server/audio/voice_agent.py`

#### Persistent State Structure

```python
_persistent_state = {
    "per_target_state": {
        "chair": {
            "effect_type": "color",
            "color_hex": "#228B22",  # Forest green
            "intensity": 0.85,
            "invert": False,
        },
        # ... other targets
    },
    "full_screen_filter_state": {
        "type": "dim",
        "intensity": 0.5,
        "color_hex": "",
    },
    "last_utterance": "make the chair green",
    "last_sam_prompts": {"chair", "desk", "person"},
}
```

This enables incremental commands:
- **"make chair green"** → Stores `color_hex: #228B22`
- **"darker green"** → Vision looks up current `#228B22`, computes `#1a5c17`

#### New Methods

**`_update_persistent_state(utterance, plan)`**

Called after command execution to update:
- per_target_state for each modified target
- full_screen_filter_state
- last_utterance

**`_get_persistent_state_snapshot() → dict`**

Thread-safe copy for Phase 1 vision call.

**`sync_sam_prompts_to_persistent_state(prompts)`**

Called from orchestrator when prompts change (add/remove/set).

**`_vision_result_to_command_plan(vision_result) → CommandPlan`**

Converts vision API response to CommandPlan structure.

#### Modified Command Flow

**`_execute_command(utterance)`**

```python
# 1. Try fast-path regex (preserves low latency)
fast = self._planner._try_fast_parse(utterance)
if fast:
    plan = fast

# 2. If not fast-path, try vision-based planning
elif self._scene and self._scene.available and self._latest_frame:
    persistent_state = self._get_persistent_state_snapshot()
    vision_result = self._scene.plan_from_vision(
        utterance, self._latest_frame, persistent_state
    )
    plan = self._vision_result_to_command_plan(vision_result)

# 3. Fall back to text-only planner
else:
    plan = self._planner.plan_command(utterance, known_objs, active_fx)

# 4. Execute plan, update persistent state
self._update_persistent_state(utterance, plan)
```

---

### 3. Orchestrator Integration

**File:** `ServerBackend/server/pipeline/orchestrator.py`

#### Rate Limiting Setup

```python
self._scene_understanding = GeminiSceneUnderstanding(
    api_key=config.gemini_api_key,
    model=config.gemini_model,
)
# NEW: Configure rate limits from config
self._scene_understanding.set_rate_limits(
    min_interval=config.gemini_min_interval,        # 6.0s
    max_calls_per_minute=config.gemini_max_calls_per_minute,  # 5
)
```

#### SAM Prompt Synchronization

All prompt mutation methods now sync to VoiceAgent persistent state:

```python
def set_active_prompts(self, prompts: set[str]):
    self._segmenter.set_active_prompts(prompts)
    # NEW: Sync to VoiceAgent
    if self._voice_agent:
        self._voice_agent.sync_sam_prompts_to_persistent_state(prompts)

def add_prompt(self, prompt: str):
    self._segmenter.add_prompt(prompt)
    # NEW: Sync to VoiceAgent
    if self._voice_agent:
        self._voice_agent.sync_sam_prompts_to_persistent_state(
            self._segmenter.get_active_prompts()
        )

def remove_prompt(self, prompt: str):
    self._segmenter.remove_prompt(prompt)
    # NEW: Sync to VoiceAgent
    if self._voice_agent:
        self._voice_agent.sync_sam_prompts_to_persistent_state(
            self._segmenter.get_active_prompts()
        )
```

---

## Configuration

**File:** `ServerBackend/server/config.py`

Existing config parameters control vision behavior:

```python
@dataclass
class ServerConfig:
    # Vision model
    gemini_model: str = "gemini-2.5-flash"  # Vision-capable model

    # Rate limiting (prevents 429 errors)
    gemini_min_interval: float = 6.0       # Min seconds between vision calls
    gemini_max_calls_per_minute: int = 5   # Max vision calls per minute

    # Audio/Voice
    audio_enabled: bool = True
```

**No new config parameters required.** All vision enhancements use existing settings.

---

## Usage Examples

### Example 1: Context-Aware Dimming

**Before (text-only):**
```
User: "it's too bright"
→ Hardcoded heuristic: dim ["light", "screen", "window"]
→ May not match actual bright objects in scene
```

**After (vision-driven):**
```
User: "it's too bright"
→ Vision analyzes current frame
→ Identifies bright objects: ["lamp", "window", "monitor"]
→ Applies dim to actual bright sources in THIS scene
```

### Example 2: Precise Color Control

**Before (text-only):**
```
User: "make chair very dark green"
→ Gemini text-only: guesses #0d260d (near black, hue lost)
```

**After (vision-driven):**
```
User: "make chair very dark green"
→ Vision: "very dark green" = multiply #00FF00 by 0.40
→ Output: #006600 (clearly GREEN, appropriately dark)
```

### Example 3: Incremental Commands

**Before (text-only):**
```
User: "make chair green"
→ Applied #228B22

User: "darker green"
→ No context of current color
→ Gemini guesses a dark green (inconsistent)
```

**After (vision-driven):**
```
User: "make chair green"
→ Vision: #228B22
→ Persistent state: chair.color_hex = "#228B22"

User: "darker green"
→ Vision looks up current: #228B22
→ Reduces luminance by ~20%: #1a5c17
→ Consistent relative adjustment
```

### Example 4: Dynamic Object Discovery

**Before (text-only):**
```
User: "blur the lamp"
→ If "lamp" not in SAM3 prompts: no segment found
→ Effect not applied
```

**After (vision-driven):**
```
User: "blur the lamp"
→ Vision sees lamp in scene
→ new_prompts_for_sam: ["lamp"]
→ Orchestrator adds "lamp" to SAM3 prompts
→ SAM3 segments lamp
→ Effect applied
```

---

## Fallback Behavior

Vision enhancements are **gracefully degradable**:

1. **Fast-path regex always runs first** (zero vision calls)
   - "blur laptop" → instant, no API call
   - "clear" → instant

2. **Vision unavailable** → Text-only planner
   - No API key: falls back to text-only
   - Rate limited (429): falls back to text-only after retries
   - Network error: falls back to text-only

3. **No frame available** → Text-only planner
   - Vision needs a frame; if none, uses text-only

4. **Vision call fails** → Text-only planner
   - JSON parse error, timeout, etc.

**Result:** System never crashes, always provides a plan.

---

## Performance & Cost

### Vision Call Frequency

Vision runs **only on command**, not per-frame:
- Typical usage: 1-2 commands/minute
- Each command: 1 vision call (Phase 1)
- Fast-path commands: 0 vision calls

### Rate Limiting

Config enforces safe limits:
- `gemini_min_interval: 6.0` → ≥6s between calls
- `gemini_max_calls_per_minute: 5` → max 5 calls/min

Even with burst commands, rate limiter prevents 429 errors.

### Cost Estimation (Gemini 2.5 Flash, Paid Tier)

**Per vision call:**
- Input: ~1,500 tokens (image + prompt) → $0.00045
- Output: ~300 tokens → $0.00075
- **Total: ~$0.0012 per call**

**Monthly (1,000 commands):**
- 1,000 commands × $0.0012 = **$1.20/month**

**Free tier:** Vision calls are free within quota.

---

## Testing

### Manual Testing Checklist

**Color Commands:**
- [ ] "make chair green" → Apply green, store color in persistent state
- [ ] "darker green" → Vision uses stored color, outputs darker green
- [ ] "very dark blue" → Output looks BLUE (not black), e.g. #000066

**Context-Aware Commands:**
- [ ] "it's too bright" → Vision identifies bright objects in scene
- [ ] "dim the room" → Vision targets lights/windows based on scene

**Dynamic Discovery:**
- [ ] "blur the lamp" (lamp not in SAM3) → Vision adds "lamp" to prompts

**Fallback:**
- [ ] Disable API key → Commands fall back to text-only planner
- [ ] Fast-path commands ("blur laptop") → Zero latency, no vision call

**Incremental:**
- [ ] "make wall blue" → "more opaque" → Intensity increases
- [ ] "make screen red" → "lighter red" → Luminance increases

---

## Future Enhancements (Phase 2)

### Phase 2 Verification Loop

**Status:** Infrastructure complete, not yet wired.

**Design:**
1. After executing plan + applying effects, capture composite frame (server-side or client-side)
2. Call `verify_and_correct(frame_with_overlay, plan, utterance)`
3. If `verified: false`, apply corrections and re-run SAM3
4. Iterate until verified or max_corrections (2-3) reached

**Why not implemented yet:**
- Requires composite frame generation (server-side mask overlay simulation or Unity screenshot)
- Adds 1-2 extra vision calls per command (cost/latency trade-off)
- Phase 1 vision already provides significant improvement

**When to implement:**
- If user reports "effects look wrong" frequently
- If incremental commands drift over time
- If we add Unity screenshot capture to protocol

### Composite Frame Generation

**Option 1: Server-side simulation (no protocol changes)**
- Decode RLE masks
- Apply blur/dim/color effects using OpenCV
- Generate composite frame
- Pro: No Unity changes
- Con: Approximate (doesn't match Unity shader output exactly)

**Option 2: Client-side capture (preferred, requires protocol extension)**
- Unity captures screenshot of Game view after rendering overlays
- Encodes as JPEG, sends to server
- Pro: Accurate representation of what user sees
- Con: Requires new message type in protobuf

---

## Troubleshooting

### Vision calls fail with 429 errors

**Cause:** Rate limit exceeded.

**Fix:**
1. Check config: `gemini_min_interval` and `gemini_max_calls_per_minute`
2. Increase `gemini_min_interval` to 10s if using free tier
3. Verify retry logic is working (logs show exponential backoff)

### Commands always fall back to text-only

**Cause:** Vision not initializing.

**Check:**
1. API key set: `export GOOGLE_API_KEY=...` or `GEMINI_API_KEY=...`
2. Logs show: "Gemini scene understanding ready"
3. `_scene.available == True`

### "darker green" doesn't work incrementally

**Cause:** Persistent state not updating.

**Check:**
1. Logs show: "Update persistent state for incremental commands"
2. `_persistent_state["per_target_state"]["chair"]` has `color_hex`
3. Orchestrator calling `sync_sam_prompts_to_persistent_state`

### New objects not segmented

**Cause:** `new_prompts_for_sam` not synced to SAM3.

**Check:**
1. Vision output includes `new_prompts_for_sam: ["lamp"]`
2. Orchestrator adds prompts: `self.add_prompt(target)`
3. SAM3 receives new prompts (logs show active prompts list)

---

## File Reference

| Component | File |
|-----------|------|
| Vision planning (Phase 1 & 2) | `ServerBackend/server/vision/gemini_scene_understanding.py` |
| Persistent state + command flow | `ServerBackend/server/audio/voice_agent.py` |
| Rate limiting setup + prompt sync | `ServerBackend/server/pipeline/orchestrator.py` |
| Configuration | `ServerBackend/server/config.py` |
| Build specification | `ServerBackend/docs/GEMINI_VISION_BUILD_PROMPT.md` |
| Implementation guide | `ServerBackend/docs/GEMINI_VISION_IMPLEMENTATION.md` |

---

## Key Design Decisions

### 1. Vision is Opt-In, Not Required

- System works without vision (falls back to text-only)
- No breaking changes to existing deployments
- Free tier users can disable vision to save quota

### 2. Fast-Path Preserved

- Regex fast-path runs first (0ms latency)
- Vision only for complex/ambiguous commands
- "blur laptop" remains instant

### 3. Rate Limiting Built-In

- Min interval + RPM cap prevents 429 errors
- Exponential backoff on retries
- Graceful degradation on failure

### 4. Persistent State is Shallow

- Only stores per-target effects + filter state
- No conversation history (cleared after each command)
- Low memory footprint

### 5. Phase 2 Deferred

- Phase 1 provides 80% of value
- Phase 2 adds cost/latency for diminishing returns
- Infrastructure ready for future activation

---

## Summary

### What Was Implemented ✅

- [x] Phase 1: Vision-based scene analysis before applying effects
- [x] Persistent state store for incremental commands
- [x] Rate limiting + 429 retry logic
- [x] Fallback to text-only planner
- [x] SAM3 prompt synchronization with persistent state
- [x] Vision-driven color precision (RGB output)
- [x] Dynamic object discovery (new_prompts_for_sam)

### What Was Preserved ✅

- [x] Fast-path regex commands (zero latency)
- [x] Text-only Gemini planner
- [x] MediaPipe body detection
- [x] SAM3 text-prompted segmentation
- [x] All existing effects and filters
- [x] Conversation state and clarification

### Future Work 🚧

- [ ] Phase 2: Post-execution verification + correction loop
- [ ] Composite frame generation (server or client)
- [ ] Verification iteration logic (2-3 correction rounds)
- [ ] Metrics/telemetry for vision call frequency and success rate

---

**Implementation Complete:** Phase 1 Gemini Vision-Driven Pipeline

**Status:** Ready for testing and deployment

**Contact:** For questions or issues, refer to the build specification at `ServerBackend/docs/GEMINI_VISION_BUILD_PROMPT.md`
