# Negation Prompts Code Analysis Report

## Overview
This document contains a comprehensive analysis of how negation prompts ("blur everything but X") are sent to the server, processed, and applied in the SocialSenseAR system.

---

## 1. NEGATION PROMPT PARSING

### Location: Server-side Command Parsing
**File:** `/Users/ajayraj/SAMARSDK/mainBranch/server/commands.py` (Lines 264-287)

```python
# Detect negation: "blur everything but X", "dim all except X", "blur everything around X"
invert = False
negation_match = re.search(
    r'(?:everything|all)\s+(?:but|except|around|other\s+than|besides)\s+(?:the\s+)?(\S+)',
    parse_text,
)
if negation_match:
    invert = True
    target_word = negation_match.group(1)
    # Detect action from the beginning
    action = "blur"  # default
    if parse_text.startswith(("dim", "darken")):
        action = "dim"
    elif parse_text.startswith("pixelate"):
        action = "pixelate"
    elif parse_text.startswith("highlight"):
        action = "highlight"
    elif color_hex or parse_text.startswith(("color", "paint", "make")):
        action = "color"
    # "me" / "myself" → person
    if target_word in ("me", "myself", "us"):
        target_word = "person"
    target_word = LABEL_ALIASES.get(target_word, target_word)
    return (action, target_word, True, color_hex, intensity)
```

### Key Patterns Recognized:
- "blur everything but laptop"
- "dim all except person"  
- "blur everything around X"
- "blur all other than X"
- "blur all besides X"

---

## 2. VOICE AGENT PROCESSING

### Location: Voice Command Planner
**File:** `/Users/ajayraj/SAMARSDK/mainBranch/server/audio/voice_agent.py` (Lines 456-465)

#### Fast-Path Regex Pattern for Inverted Effects:
```python
_RE_EFFECT_INVERT = re.compile(
    rf'^({_EFFECTS})\s+(?:every\s*thing|everything|all)\s+(?:but|except|nothing\s+but)\s+(.+)$', 
    re.IGNORECASE,
)

# Pattern 3: "<effect> everything/all but/except <target>"
m = self._RE_EFFECT_INVERT.match(text)
if m:
    target = self._extract_target(m.group(2))
    if target is None:
        return None
    return CommandPlan(
        targets=[target], 
        effect_type=m.group(1).lower(),
        intensity=0.9, 
        action="add", 
        invert=True,  # CRITICAL: Set invert=True for negation prompts
        reasoning=f"fast-path: {m.group(1)} everything but {target}",
    )
```

#### Data Structure:
**File:** `/Users/ajayraj/SAMARSDK/mainBranch/server/audio/voice_agent.py` (Lines 30-44)

```python
@dataclass
class CommandPlan:
    targets: list[str]
    effect_type: str
    intensity: float
    action: str
    invert: bool = False  # True = effect on everything EXCEPT targets
    color_hex: Optional[str] = None
    full_screen_filter: Optional[str] = None
    full_screen_intensity: float = 0.5
    full_screen_color: Optional[str] = None
    reasoning: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
```

---

## 3. INVERTED EFFECT HANDLING IN PIPELINE

### Location: Pipeline Orchestrator
**File:** `/Users/ajayraj/SAMARSDK/ServerBackend/server/pipeline/orchestrator.py`

#### Callback Handler (Lines 1019-1067):
```python
def _on_voice_command(self, action: str, targets: list[str], effect_type: str, 
                      intensity: float, invert: bool = False, color_hex: str | None = None):
    """Callback from VoiceAgent: sync SAM3 prompts with voice commands.
    
    Per spec: "We should always leave requested objects within the array" —
    prompts persist even after effect removal so SAM3 keeps segmenting them.
    
    If invert=True, SAM3 segments the targets but the effect applies to
    everything OUTSIDE those masks (e.g., "blur everything but laptop").
    color_hex: Optional hex color for "color" effect (e.g., "#FF0000")
    """
    # ...validation...
    
    if action in ("add", "change"):
        for target in valid_targets:
            self.add_prompt(target)
            # CRITICAL: Pass invert flag to set_effect
            self.set_effect(target, effect_type, intensity, invert=invert, color_hex=color_hex)
```

#### Effect Application (Lines 917-950):
```python
def _apply_effects(self, segments: list):
    """Apply effects to tracked segments.
    
    Combines effects from:
    1. Effect registry (typed commands: blur/unblur)
    2. Voice agent (if enabled)
    
    Sets EffectData on each segment.
    """
    from server.vision.segment_data import EffectData
    
    # Merge effect sources: registry + voice agent
    active_effects = self.get_effects()  # from typed commands
    
    # Voice agent effects override/extend typed commands
    if self._voice_agent is not None:
        voice_effects = self._voice_agent.get_active_effects()
        active_effects.update(voice_effects)
    
    if not active_effects:
        return
    
    for seg in segments:
        if seg.label and seg.label in active_effects:
            fx = active_effects[seg.label]
            params = {}
            if fx.get("invert", False):
                params["invert"] = "true"  # Store invert as string in params
            seg.effect = EffectData(
                effect_type=fx.get("type", "none"),
                intensity=fx.get("intensity", 1.0),
                color_hex=fx.get("color_hex", ""),
                params=params,
            )
```

#### Effect Registry Storage (Lines 318-335):
```python
def set_effect(self, label: str, effect_type: str, intensity: float = 1.0,
               invert: bool = False, color_hex: str | None = None):
    """Apply an effect to a label (e.g., blur laptop, color person blue).
    
    If invert=True, SAM3 segments the target but the effect applies to
    everything OUTSIDE the mask (e.g., "blur everything but laptop").
    color_hex: Optional hex color string for "color" effect (e.g., "#FF0000")
    """
    with self._effect_lock:
        self._effect_registry[label] = {
            "type": effect_type,
            "intensity": min(1.0, max(0.0, intensity)),
            "invert": invert,  # CRITICAL: Store invert flag
            "color_hex": color_hex,  # NEW: store color_hex
        }
```

---

## 4. PROTOBUF SERIALIZATION AND TRANSMISSION

### Location: WebSocket Server
**File:** `/Users/ajayraj/SAMARSDK/ServerBackend/server/websocket_server.py` (Lines 188-220)

```python
for seg in result.segments:
    proto_seg = response.segments.add()
    # Strip ~ prefix (internal "pending Gemini" marker)
    label = seg.label or ""
    proto_seg.label = label.lstrip("~")
    proto_seg.asset_class = seg.asset_class or ""
    proto_seg.confidence = seg.confidence
    proto_seg.bbox.x_min = seg.bbox[0]
    proto_seg.bbox.y_min = seg.bbox[1]
    proto_seg.bbox.x_max = seg.bbox[2]
    proto_seg.bbox.y_max = seg.bbox[3]
    proto_seg.rle_mask = seg.rle_mask or b""
    proto_seg.mask_width = seg.mask_width
    proto_seg.mask_height = seg.mask_height
    proto_seg.center_x = seg.center_x
    proto_seg.center_y = seg.center_y
    proto_seg.track_id = seg.track_id or ""
    
    # Voice agent effects
    if seg.effect is not None:
        proto_seg.effect.effect_type = seg.effect.effect_type
        proto_seg.effect.intensity = seg.effect.intensity
        # NEW: Pass color_hex directly to protobuf
        if seg.effect.color_hex:
            proto_seg.effect.color_hex = seg.effect.color_hex
        for k, v in seg.effect.params.items():
            proto_seg.effect.params[k] = v  # params includes "invert" flag
```

#### Full-Screen Filter State (Lines 241-250):
```python
# Full-screen filter state
if self.pipeline and hasattr(self.pipeline, '_voice_agent') and self.pipeline._voice_agent:
    fs_filter = self.pipeline._voice_agent.get_full_screen_filter()
    if fs_filter:
        response.conversation.voice_agent.full_screen_filter.filter_type = fs_filter.get("type", "none")
        response.conversation.voice_agent.full_screen_filter.intensity = fs_filter.get("intensity", 0.5)
        # NEW: Set color_hex for full-screen color filters
        color_hex = fs_filter.get("color_hex")
        if color_hex:
            response.conversation.voice_agent.full_screen_filter.color_hex = color_hex
```

---

## 5. UNITY CLIENT RECEPTION AND HANDLING

### Location: SocialSenseClient.cs
**File:** `/Users/ajayraj/SAMARSDK/unitySetUp/SocialSenseAR-Unity/Assets/Scripts/SocialSenseClient.cs`

#### Processing Voice Agent State (Lines 836-886):
```csharp
private void ProcessVoiceAgentState(ServerMessage msg)
{
    var voiceAgent = msg.Conversation?.VoiceAgent;
    
    // Update full-screen filter
    if (fullScreenFilter != null && voiceAgent != null)
    {
        var fs = voiceAgent.FullScreenFilter;
        if (fs != null && !string.IsNullOrEmpty(fs.FilterType) && fs.FilterType != "none")
            fullScreenFilter.SetFilter(fs.FilterType, fs.Intensity, fs.ColorHex);
        else
            fullScreenFilter.SetFilter("none", 0f);
    }
}
```

#### Critical Comment (Lines 735-740, 863-868):
```csharp
// Clear full-screen filter to prevent white screen
if (fullScreenFilter != null)
{
    fullScreenFilter.SetFilter("none", 0f);
    Debug.Log("[SocialSense] Full-screen filter cleared");
}
```

---

## 6. FULL-SCREEN FILTER EFFECT APPLICATION

### Location: FullScreenFilterEffect.cs
**File:** `/Users/ajayraj/SAMARSDK/unitySetUp/SocialSenseAR-Unity/Assets/Scripts/FullScreenFilterEffect.cs`

#### Color Parsing and Application (Lines 86-141):
```csharp
public void SetFilter(string filterType, float intensity, string colorHex = null)
{
    _filterType = filterType ?? "none";
    _intensity = Mathf.Clamp01(intensity);
    _colorHex = colorHex;  // NEW: store hex color
    ApplyFilter();
}

private void ApplyFilter()
{
    if (_quad == null || _instanceMaterial == null)
        return;
    
    bool active = _filterType != "none" && _intensity > 0f;
    _quad.SetActive(active);
    
    if (active)
    {
        _instanceMaterial.SetFloat(FilterTypeId, FilterTypeToIndex(_filterType));
        _instanceMaterial.SetFloat(IntensityId, _intensity);
        
        // NEW: Set color for "color" filter type
        if (_filterType == "color")
        {
            Color? parsedColor = ParseColorHex(_colorHex);
            Color filterColor = parsedColor ?? Color.white;  // Fallback to white
            _instanceMaterial.SetColor(FilterColorId, filterColor);
        }
    }
}

private Color? ParseColorHex(string hex)
{
    if (string.IsNullOrEmpty(hex))
        return null;
    
    hex = hex.Trim().TrimStart('#');
    if (hex.Length != 6)
        return null;
    
    try
    {
        byte r = Convert.ToByte(hex.Substring(0, 2), 16);
        byte g = Convert.ToByte(hex.Substring(2, 2), 16);
        byte b = Convert.ToByte(hex.Substring(4, 2), 16);
        return new Color(r / 255f, g / 255f, b / 255f, 1f);
    }
    catch
    {
        return null;
    }
}
```

---

## 7. POTENTIAL WHITE SCREEN ISSUES

### Root Causes Identified:

#### 1. Invalid Hex Color Parsing
**Issue:** If `color_hex` is malformed or not properly validated, `ParseColorHex()` returns `null`, causing fallback to `Color.white`

**Evidence (Line 111):**
```csharp
Color filterColor = parsedColor ?? Color.white;  // Fallback to white
```

**Risk Scenarios:**
- Server sends `color_hex = "#FFFFFF"` (pure white) with intensity 1.0
- Server sends `color_hex = ""` or `null`
- Invalid hex format like `"#FFF"` (too short) or `"white"` (not hex)

#### 2. Uninitialized Full-Screen Filter
**Issue:** If `fullScreenFilter` component is not assigned but voice agent sends filter data, it's silently ignored

**Evidence (Line 878):**
```csharp
if (fullScreenFilter != null && voiceAgent != null)
{
    var fs = voiceAgent.FullScreenFilter;
    // ...apply filter...
}
```

#### 3. Missing "none" Filter Fallback
**Issue:** If a full-screen color filter is applied but then the server doesn't explicitly send a "clear" command, the filter may persist or become invalid

**Critical Evidence (Lines 735-740):**
```csharp
// IMPORTANT: Always process EffectsCleared immediately, even if masks haven't changed.
// This ensures "clear" command works even when server returns cached response.
if (msg.EffectsCleared && overlayRenderer != null)
{
    Debug.Log("[SocialSense] EffectsCleared=true received, clearing overlay");
    overlayRenderer.UpdateOverlay(msg.Segments, null, true);
    // Clear full-screen filter to prevent white screen
    if (fullScreenFilter != null)
    {
        fullScreenFilter.SetFilter("none", 0f);
        Debug.Log("[SocialSense] Full-screen filter cleared");
    }
}
```

#### 4. Shader Invalid Color Issue
**Issue:** The shader receives a color value but shader parsing or color space conversion fails

**Risk:** If `_FilterColor` is never set in shader and defaults to white, rendering a full-screen white quad would create a white screen

---

## 8. DATA FLOW SUMMARY

```
User Command: "blur everything but laptop"
    ↓
[Client sends ControlPayload with command string]
    ↓
Server: parse_command() → ("blur", "laptop", True, None, None)
    ↓
Voice Agent: CommandPlan(targets=["laptop"], effect_type="blur", invert=True)
    ↓
Orchestrator._on_voice_command(invert=True)
    ↓
set_effect("laptop", "blur", 1.0, invert=True)
    ↓
_effect_registry["laptop"] = {
    "type": "blur",
    "intensity": 1.0,
    "invert": True,  ← CRITICAL FLAG
    "color_hex": None
}
    ↓
_apply_effects(segments):
    for seg in segments:
        if seg.label == "laptop":
            seg.effect = EffectData(
                effect_type="blur",
                intensity=1.0,
                params={"invert": "true"}  ← Serialized in protobuf
            )
    ↓
WebSocket sends ServerMessage with:
    segments[i].effect.effect_type = "blur"
    segments[i].effect.params["invert"] = "true"
    ↓
Unity Client receives and extracts invert flag from params
```

---

## 9. COLOR OVERLAY SYSTEM

### Full-Screen Color Filter Flow:

```
User Command: "make everything dark blue except laptop"
    ↓
Voice Agent Parsing:
    - Detects "dark blue" color modifier
    - Brightness multiplier: 0.40 (for "dark")
    - Base blue: #0000FF
    - Adjusted color: #000066 (preserves blue hue)
    
    - Detects "except laptop" negation
    - invert = True
    
    CommandPlan:
        targets=["laptop"]
        effect_type="color"
        color_hex="#000066"  ← Darkened blue
        invert=True
    ↓
Orchestrator._apply_effects():
    seg.effect = EffectData(
        effect_type="color",
        intensity=1.0,
        color_hex="#000066",
        params={"invert": "true"}
    )
    ↓
WebSocket Server serializes:
    proto_seg.effect.color_hex = "#000066"
    ↓
Unity FullScreenFilterEffect.SetFilter():
    ParseColorHex("#000066") → Color(0, 0, 102/255, 1)
    _instanceMaterial.SetColor(FilterColorId, color)
```

---

## 10. CRITICAL CONFIGURATION POINTS

### Server-side Intensity Hints (Lines 253-261):
```python
_INTENSITY_HINTS = [
    (r'\b(extremely|super|maximum|max|absolutely|completely|totally)\b', 1.0),
    (r'\b(very|really|highly|heavily)\b', 1.0),  # Increased from 0.9
    (r'\b(quite|pretty)\b', 0.9),
    (r'\b(moderately|somewhat)\b', 0.7),
    (r'\b(a bit|a little)\b', 0.5),
    (r'\b(slightly|barely|subtly|gently|minimally)\b', 0.35),
]
```

### Darkness Multipliers (Lines 262-270):
```python
_DARKNESS_HINTS = [
    (r'\b(pitch black|jet black|pure black)\b', 0.0),
    (r'\b(extremely dark)\b', 0.25),  # Preserves hue
    (r'\b(very dark)\b', 0.40),       # Preserves hue
    (r'\b(dark)\b', 0.55),            # Preserves hue
    (r'\b(deep)\b', 0.65),            # Preserves hue
]
```

---

## 11. RECOMMENDED SAFEGUARDS AGAINST WHITE SCREEN

1. **Always validate `color_hex` format**
   - Check it matches regex `^#[0-9A-Fa-f]{6}$`
   - Reject invalid formats before applying

2. **Implement fallback mechanism**
   ```csharp
   Color? parsed = ParseColorHex(colorHex);
   Color filterColor = parsed ?? new Color(0.5f, 0.5f, 0.5f, 1f);  // Gray instead of white
   ```

3. **Explicit "none" filter on disconnect/clear**
   ```csharp
   fullScreenFilter.SetFilter("none", 0f, null);
   ```

4. **Validate shader uniforms**
   - Ensure `_FilterColor` is always initialized in shader
   - Test with: `make everything white` to ensure it's not your fallback color

5. **Log color conversions for debugging**
   ```csharp
   Debug.Log($"[FullScreenFilter] Parsing hex={colorHex} → {filterColor}");
   ```

---

## Summary

**Negation prompts use a three-layer flag system:**
1. **Parse layer:** `invert=True` boolean from regex matching
2. **Processing layer:** Stored in effect registry with `invert` flag
3. **Transmission layer:** Serialized as `params["invert"] = "true"` in protobuf
4. **Rendering layer:** Applied by shader to invert mask regions

**White screen issues** primarily stem from:
- Invalid color hex parsing defaulting to white
- Uninitialized filter state
- Missing "none" filter fallback after color filter commands

