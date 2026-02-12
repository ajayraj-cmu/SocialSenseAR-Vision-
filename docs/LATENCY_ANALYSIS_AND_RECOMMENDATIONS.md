# Latency Analysis & Recommendations

**Purpose:** Identify sources of latency between voice/typed command and mask appearance, with actionable recommendations. No code changes—analysis and suggestions only.

---

## 1. Executive Summary

Latency has increased since adding Gemini Vision because **non-fast-path commands** now block on:

1. **Rate-limit throttle** — up to **6 seconds** (`gemini_min_interval`) before the Gemini Vision call can run
2. **Gemini Vision API** — typically **1–4 seconds** for image + structured response
3. **Transcription** — 0.5–2 seconds (Whisper)
4. **SAM3 first mask** — one frame cycle after prompts are set

**Fast-path commands** (e.g. "blur lamp", "clear") skip Gemini Vision and are much faster; they only pay transcription + one frame cycle.

**Upping GPU alone will not fix the main bottleneck.** The largest delays come from API waits and rate limiting, not SAM3 inference.

---

## 2. Full Latency Breakdown

### 2.1 Command Flow (Vision Path)

```
User: "Hey Vibe, make the chair green, thank you"
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. TRANSCRIPTION (Whisper)                    ~0.5–2 s          │
│    - Recording until "thank you" / silence                       │
│    - OpenAI API or local faster-whisper                          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. FAST-PATH CHECK                             ~0 ms            │
│    - "make the chair green" does NOT match regex fast-path       │
│    - Falls through to Gemini Vision                              │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. RATE-LIMIT WAIT (gemini_min_interval=6.0)   0–6 s            │
│    - _wait_for_rate_limit() sleeps in 0.5s steps                 │
│    - If last vision call was <6s ago, blocks until 6s elapsed    │
│    - Worst case: command right after previous → 6s wait          │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. GEMINI VISION API (plan_from_vision)        1–4 s            │
│    - JPEG encode + resize to 1024px                              │
│    - Network: Modal ↔ Gemini API                                 │
│    - Model inference (image + long prompt)                       │
│    - JSON parse                                                  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. EXECUTION (callback → add prompts/effects)  ~0 ms            │
│    - _on_voice_command adds "chair" to SAM3 prompts              │
│    - Updates effect registry                                     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SAM3 LOOP (next frame)                      50–200 ms        │
│    - Waits for next frame from Quest                             │
│    - Vision encode (or reuse if frame similar)                   │
│    - Decoder for "chair"                                         │
│    - TRT on A100: ~50–100ms; PyTorch: ~100–200ms                 │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. NETWORK (Modal → Quest)                     ~20–80 ms        │
│    - RLE-encoded masks over WebSocket                            │
└─────────────────────────────────────────────────────────────────┘

TOTAL (vision path): 2–13+ seconds typical, 7–13s when rate-limited
```

### 2.2 Command Flow (Fast-Path)

```
"Hey Vibe, blur the lamp, thank you"
        │
        ▼
Transcription (~0.5–2 s) → Fast-path match ("blur" + "lamp") → Callback → SAM3 next frame → Network

TOTAL (fast-path): ~0.7–2.5 seconds
```

---

## 3. Latency Sources (Ranked by Impact)

| Rank | Source | Typical Latency | Mitigation |
|------|--------|-----------------|------------|
| 1 | **Rate-limit wait** (`gemini_min_interval=6.0`) | 0–6 s | Reduce to 2–3s or skip for first call |
| 2 | **Gemini Vision API** | 1–4 s | Smaller model, smaller image, shorter prompt |
| 3 | **Transcription** (Whisper) | 0.5–2 s | Local model, smaller chunks |
| 4 | **SAM3 inference** | 50–200 ms | Better GPU helps here |
| 5 | **Network** (Quest ↔ Modal) | 20–80 ms | Geographic proximity, keep container warm |
| 6 | **Frame pipeline** | 1–2 frame periods | Minimal; already efficient |

---

## 4. Recommendations (No Code Changes Yet)

### 4.1 High Impact

1. **Reduce `gemini_min_interval` from 6.0 to 2.0–3.0 seconds**
   - Current 6s is very conservative; free tier allows ~10 RPM.
   - With sparse commands, 2–3s is usually safe.
   - Saves 0–4 seconds per command when throttled.

2. **Skip rate-limit wait on first vision call**
   - If no vision call in the last 60s, skip `_wait_for_rate_limit` entirely.
   - First command after idle should feel instant (minus API time).

3. **Expand fast-path coverage**
   - Add more regex patterns for common phrases so more commands avoid Gemini Vision.
   - Examples: "blur that", "dim the screen", "highlight me" (→ person).
   - Each fast-path command avoids 1–10s of Vision + rate-limit.

4. **Use Gemini 2.5 Flash-Lite for planning**
   - `gemini_planning_model` is already Flash-Lite; ensure Vision uses it or a lighter variant.
   - Flash-Lite is cheaper and often 20–30% faster than Flash.

### 4.2 Medium Impact

5. **Resize frame more aggressively**
   - Current: 1024px max. Try 768px for Vision.
   - Fewer tokens → faster API response.

6. **Shorten Vision prompt**
   - Trim tool inventory and examples to essential parts.
   - Shorter prompt → lower input tokens → faster response.

7. **Warm Modal container**
   - Avoid scale-to-zero during active use (increase `CONTAINER_IDLE_TIMEOUT` or keep-alive).
   - Cold start adds 60–90s.

8. **Local / faster transcription**
   - Prefer faster-whisper over cloud Whisper when possible.
   - Consider streaming or smaller chunks to reduce time-to-first-token.

### 4.3 Lower Impact (GPU / Infrastructure)

9. **Upgrade GPU**
   - Current (A100) is already strong. A10G/L4 are slower but usually sufficient.
   - GPU mainly affects SAM3 (50–200ms); it is not the main bottleneck.
   - Upgrading helps if SAM3 is taking 200ms+ consistently.

10. **Regional Modal deployment**
    - Deploy Modal in a region close to Gemini API and to users.
    - Can shave 20–50ms on API and WebSocket round-trips.

---

## 5. Is Gemini Vision Actually Working?

### 5.1 How to Verify

1. **Log messages**
   - Vision path: `"Phase 1: Running vision-based scene analysis..."`
   - Success: `"Phase 1 (plan_from_vision): targets=[...] | effect=... | new_prompts_for_sam=[...]"`
   - Fast-path: `"Fast-path plan: add blur → [lamp]"`

2. **Commands that use Vision (non-fast-path)**
   - "make the chair green"
   - "darker green" (needs persistent state)
   - "dim the thing on the left"
   - "it's too bright" (generic; also has fast-path with hardcoded targets)
   - "blur that" (if "that" is ambiguous)

3. **Commands that skip Vision (fast-path)**
   - "blur lamp", "blur laptop", "blur the screen"
   - "clear", "remove all"
   - "blur everything but me"
   - "it's too bright" (has dedicated fast-path)

### 5.2 Common Failure Modes

| Symptom | Likely Cause |
|---------|--------------|
| Crashes on command | Vision unavailable (no GEMINI_API_KEY in Modal secret) |
| Always fast-path | GEMINI_API_KEY missing or `_scene.available == False` |
| Slow on every command | Vision path + rate limit; check `gemini_min_interval` |
| Masks never appear | `active_prompts` empty; command may not be invoking callback |

### 5.3 Potential Bug: `new_prompts_for_sam` Not Used

In `voice_agent.py`:

```python
new_prompts = vision_result.get("new_prompts_for_sam", [])
if new_prompts and self._on_command_callback:
    # Sync new prompts with orchestrator (done in callback)
    pass  # <-- Does nothing
```

The callback is called with `plan.targets`, not `new_prompts_for_sam`. If Vision returns `new_prompts_for_sam` as a superset of `targets` (e.g. segment lamp, window, monitor but only apply effect to lamp), those extra prompts are never added. For most commands, `targets` and `new_prompts_for_sam` match, but for "dim the room" / "it's too bright" they may differ. Worth verifying and fixing if needed.

---

## 6. Suggested Priority Order

1. **Immediate:** Reduce `gemini_min_interval` to 2.0–3.0 (config change).
2. **Short-term:** Skip rate-limit wait when no recent vision call (code change).
3. **Short-term:** Add more fast-path patterns for frequent commands.
4. **Medium-term:** Use 768px resize and shorter Vision prompt.
5. **Lower priority:** GPU upgrade (only if SAM3 is measured as the bottleneck).

---

## 7. Config Reference

| Config | Current | Suggested |
|--------|---------|-----------|
| `gemini_min_interval` | 6.0 | 2.0–3.0 |
| `gemini_max_calls_per_minute` | 5 | 5 (keep) |
| `gemini_model` | gemini-2.5-flash | gemini-2.5-flash or flash-lite |
| Frame resize for Vision | 1024px | 768px (experimental) |

---

*End of latency analysis. Implement changes incrementally and measure impact.*
