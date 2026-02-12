# SocialSenseAR Effect System Expansion Specification

## Overview

This document specifies the expansion of the SocialSenseAR visual effect system to support:
1. **Color overlays** (any RGB color) for object masking
2. **Enhanced full-screen filters** (dim, warm, cool, night, grayscale, custom colors)
3. **Improved voice agent mapping** to understand color requests and map them to appropriate effects

## Current State

### Supported Effects (v1)
- `blur` - Gaussian blur on passthrough
- `dim` - Semi-transparent black overlay
- `pixelate` - Block/mosaic pattern
- `highlight` - Warm yellow brightening
- `outline` - Colored border

### Limitations
- ❌ No color overlay support ("make person blue")
- ❌ Limited full-screen filters (only dim/warm/cool/night/grayscale)
- ❌ Voice agent doesn't recognize color names → effect mappings

---

## Proposed Expansion (v2)

### 1. Color Overlay Effects

#### New Effect Type: `color`

**Server-side (`server/config.py`):**
```python
EFFECT_TYPES = (
    "blur", "dim", "pixelate", "highlight", "outline",
    "color"  # NEW: RGB color overlay
)
```

**Effect Metadata Extension:**
- Add `color_hex: str` field to `EffectMetadata` protobuf (e.g., `"#FF0000"` for red)
- Add `color_rgb: (r, g, b)` tuple as alternative (0-255 per channel)
- Default: if `effect_type == "color"` but no `color_hex` provided, use segment's default color

**Unity Implementation (`OverlayRenderer.cs`):**
```csharp
case "color":
    // Extract RGB from EffectMetadata.color_hex or use segColor as fallback
    Color32 overlayColor = ParseColorHex(seg.Effect.ColorHex) ?? segColor;
    byte colorAlpha = (byte)(180 * intensity);  // Semi-transparent overlay
    overlayColor.a = colorAlpha;
    ApplyMaskToBuffer(w, h, overlayColor);
    break;
```

**Voice Command Examples:**
- `"make person blue"` → `effect_type: "color", color_hex: "#0000FF"`
- `"color laptop red"` → `effect_type: "color", color_hex: "#FF0000"`
- `"paint chair green"` → `effect_type: "color", color_hex: "#00FF00"`
- `"highlight monitor yellow"` → `effect_type: "color", color_hex: "#FFFF00"` (or use existing "highlight")

---

### 2. Color Name → Hex Mapping

**Create `server/commands.py` color dictionary:**

```python
COLOR_NAMES_TO_HEX = {
    # Primary colors
    "red": "#FF0000", "green": "#00FF00", "blue": "#0000FF",
    "yellow": "#FFFF00", "cyan": "#00FFFF", "magenta": "#FF00FF",
    "white": "#FFFFFF", "black": "#000000", "gray": "#808080", "grey": "#808080",
    
    # Extended palette
    "orange": "#FFA500", "purple": "#800080", "pink": "#FFC0CB",
    "brown": "#A52A2A", "tan": "#D2B48C", "beige": "#F5F5DC",
    "lime": "#00FF00", "teal": "#008080", "navy": "#000080",
    "maroon": "#800000", "olive": "#808000", "coral": "#FF7F50",
    "gold": "#FFD700", "silver": "#C0C0C0", "bronze": "#CD7F32",
    
    # Light variants
    "light blue": "#ADD8E6", "light green": "#90EE90", "light red": "#FFB6C1",
    "light yellow": "#FFFFE0", "light gray": "#D3D3D3",
    
    # Dark variants
    "dark blue": "#00008B", "dark green": "#006400", "dark red": "#8B0000",
    "dark gray": "#A9A9A9",
    
    # Common aliases
    "azure": "#007FFF", "violet": "#8A2BE2", "indigo": "#4B0082",
    "turquoise": "#40E0D0", "emerald": "#50C878", "crimson": "#DC143C",
}
```

**Usage in command parsing:**
- When Gemini/voice agent detects color name → map to hex
- Pass `color_hex` in `EffectMetadata` to Unity

---

### 3. Enhanced Full-Screen Filters

**Current:** `full_screen_filter` supports: `"dim"`, `"warm"`, `"cool"`, `"night"`, `"grayscale"`

**Expanded:** Add intensity control + custom color filters

**Server-side (`server/audio/voice_agent.py`):**
```python
# In CommandPlan dataclass, already has:
full_screen_filter: Optional[str] = None  # "dim", "warm", "cool", "night", "grayscale", "color"
full_screen_intensity: float = 0.5
full_screen_color: Optional[str] = None  # NEW: hex color for "color" filter type
```

**Unity Implementation (`FullScreenFilterEffect.cs`):**
- Add `SetColorFilter(colorHex, intensity)` method
- Apply as post-process overlay on entire screen
- Examples:
  - `"dim everything"` → `full_screen_filter: "dim", intensity: 0.6`
  - `"make screen blue"` → `full_screen_filter: "color", color_hex: "#0000FF", intensity: 0.3`
  - `"warm filter"` → `full_screen_filter: "warm", intensity: 0.4`

---

### 4. Voice Agent / Gemini Prompt Updates

**Update `VoiceCommandPlanner` prompt (`server/audio/voice_agent.py`):**

```python
prompt = f"""Parse this command into a structured action plan. Respond ONLY with JSON.

User said: "{utterance}"

Context:
- Known objects: {known_str}
- Active effects: {active_effects}

Output JSON with these fields:
- "targets": list of object labels. ALWAYS include any object the user names, even if not in the known set. SAM3 will search for it.
- "effect_type": "blur" | "dim" | "pixelate" | "highlight" | "outline" | "color"
- "color_hex": hex color string (e.g., "#FF0000") if effect_type is "color" or user requests a specific color
- "intensity": 0.0-1.0 (default 0.8 for effects, 0.5 for full-screen filters)
- "action": "add" | "remove" | "change"
- "invert": true if effect applies to everything EXCEPT targets (e.g. "blur everything but laptop")
- "full_screen_filter": "dim" | "warm" | "cool" | "night" | "grayscale" | "color" | null
- "full_screen_intensity": 0.0-1.0
- "full_screen_color": hex color string if full_screen_filter is "color"
- "reasoning": brief explanation

Rules:
- "make person blue" → targets: ["person"], effect_type: "color", color_hex: "#0000FF", action: "add"
- "color laptop red" → targets: ["laptop"], effect_type: "color", color_hex: "#FF0000", action: "add"
- "paint chair green" → targets: ["chair"], effect_type: "color", color_hex: "#00FF00", action: "add"
- "blur phone" → targets: ["phone"], effect_type: "blur", action: "add"
- "dim everything" → full_screen_filter: "dim", full_screen_intensity: 0.6, targets: []
- "make screen blue" → full_screen_filter: "color", full_screen_color: "#0000FF", full_screen_intensity: 0.3, targets: []
- "blur everything but me" → targets: ["person"], invert: true, effect_type: "blur"
- "stop blurring person" → targets: ["person"], action: "remove"
- Color names map to hex: "blue" → "#0000FF", "red" → "#FF0000", etc. (see COLOR_NAMES_TO_HEX dict)
- If user says "highlight X" without color, use effect_type: "highlight" (warm yellow)
- If user says "highlight X [color]", use effect_type: "color" with that color
- NEVER return empty targets if the user names an object
- If just an object name with no action ("person"), default to outline effect

{{
    "targets": ["person"],
    "effect_type": "color",
    "color_hex": "#0000FF",
    "intensity": 0.8,
    "action": "add",
    "invert": false,
    "full_screen_filter": null,
    "full_screen_intensity": 0.5,
    "reasoning": "User wants to color the person blue"
}}"""
```

---

### 5. Protobuf Schema Updates

**File: `server/proto/socialsense.proto`**

```protobuf
message EffectMetadata {
  string effect_type = 1;      // "blur", "dim", "pixelate", "highlight", "outline", "color"
  float intensity = 2;         // 0.0-1.0
  string color_hex = 3;        // NEW: hex color for "color" effect (e.g., "#FF0000")
  bool invert = 4;             // true = effect applies outside mask
  map<string, string> params = 5;  // Additional params (keep for backward compat)
}

message FullScreenFilter {
  string filter_type = 1;      // "dim", "warm", "cool", "night", "grayscale", "color"
  float intensity = 2;         // 0.0-1.0
  string color_hex = 3;        // NEW: hex color if filter_type is "color"
}
```

**After updating `.proto`, regenerate:**
```bash
cd server/proto
protoc --python_out=. socialsense.proto
# Unity: regenerate C# from same .proto (see Unity project's protobuf setup)
```

---

### 6. Unity OverlayRenderer Updates

**File: `unitySetUp/SocialSenseAR-Unity/Assets/Scripts/OverlayRenderer.cs`**

#### Add color parsing helper:
```csharp
private Color32? ParseColorHex(string hex)
{
    if (string.IsNullOrEmpty(hex)) return null;
    hex = hex.Trim().TrimStart('#');
    if (hex.Length != 6) return null;
    
    try
    {
        byte r = Convert.ToByte(hex.Substring(0, 2), 16);
        byte g = Convert.ToByte(hex.Substring(2, 2), 16);
        byte b = Convert.ToByte(hex.Substring(4, 2), 16);
        return new Color32(r, g, b, 255);
    }
    catch
    {
        return null;
    }
}
```

#### Update `RenderEffect()` method:
```csharp
private void RenderEffect(int w, int h, string effectType, float intensity, Color32 segColor, string colorHex = null)
{
    switch (effectType)
    {
        case "blur":
            // ... existing blur code ...
            break;

        case "color":
            // NEW: Color overlay effect
            Color32? parsedColor = ParseColorHex(colorHex);
            Color32 overlayColor = parsedColor ?? segColor;  // Fallback to segment color
            byte colorAlpha = (byte)(180 * intensity);  // Semi-transparent
            overlayColor.a = colorAlpha;
            ApplyMaskToBuffer(w, h, overlayColor);
            break;

        case "dim":
            // ... existing dim code ...
            break;

        // ... rest of cases ...
    }
}
```

#### Update `UpdateOverlay()` to pass color_hex:
```csharp
foreach (var seg in segments)
{
    // ... existing code ...
    string effectType = seg.Effect?.EffectType ?? "none";
    string colorHex = seg.Effect?.ColorHex ?? null;  // NEW
    float intensity = seg.Effect?.Intensity ?? 0f;
    
    // ... later in RenderEffect call ...
    RenderEffect(maskW, maskH, effectType, effectIntensity, color, colorHex);
}
```

---

### 7. Full-Screen Filter Unity Implementation

**File: `unitySetUp/SocialSenseAR-Unity/Assets/Scripts/FullScreenFilterEffect.cs`**

**Add color filter support:**
```csharp
public void SetFilter(string filterType, float intensity, string colorHex = null)
{
    switch (filterType.ToLower())
    {
        case "color":
            if (!string.IsNullOrEmpty(colorHex))
            {
                Color32? parsed = ParseColorHex(colorHex);
                if (parsed.HasValue)
                {
                    // Apply color overlay to entire screen
                    // Implementation: create full-screen quad with color overlay shader
                    ApplyColorFilter(parsed.Value, intensity);
                }
            }
            break;
        case "dim":
            // ... existing dim filter ...
            break;
        // ... other filters ...
    }
}
```

---

### 8. Command Parsing Updates

**File: `server/commands.py`**

**Add color extraction helper:**
```python
def extract_color_from_text(text: str) -> tuple[str | None, str]:
    """Extract color name and return (color_name, remaining_text).
    
    Returns:
        (color_name, text_without_color) if color found, else (None, text)
    """
    text_lower = text.lower()
    
    # Check for common color patterns
    for color_name, hex_val in COLOR_NAMES_TO_HEX.items():
        # Pattern: "make X [color]" or "[color] X" or "X [color]"
        patterns = [
            rf'\b{color_name}\b',
            rf'\b{color_name}\s+',
            rf'\s+{color_name}\b',
        ]
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Remove color word from text
                cleaned = re.sub(pattern, ' ', text_lower, count=1).strip()
                return color_name, cleaned
    
    return None, text
```

**Update `parse_command()` to handle color requests:**
```python
def parse_command(text: str) -> tuple[str, str | None, bool, str | None]:
    """Parse command. Returns (action, target, invert, color_hex).
    
    NEW: Returns color_hex if color detected.
    """
    # ... existing parsing ...
    
    # Check for color requests
    color_name, text_without_color = extract_color_from_text(text)
    if color_name:
        color_hex = COLOR_NAMES_TO_HEX.get(color_name)
        # If action is "blur" or "dim" but user said color, change to "color"
        if action in ("blur", "dim", "highlight") and color_name:
            action = "color"
        return (action, target, invert, color_hex)
    
    return (action, target, invert, None)
```

---

### 9. Server-Side Effect Application

**File: `server/pipeline/orchestrator.py`**

**Update `set_effect()` to accept color:**
```python
def set_effect(self, label: str, effect_type: str, intensity: float = 1.0, 
               invert: bool = False, color_hex: str | None = None):
    """Apply effect with optional color."""
    with self._effect_lock:
        self._effect_registry[label] = {
            "type": effect_type,
            "intensity": min(1.0, max(0.0, intensity)),
            "invert": invert,
            "color_hex": color_hex,  # NEW
        }
    # ... logging ...
```

**Update `_apply_effects()` to pass color_hex to segments:**
```python
def _apply_effects(self, segments: list):
    """Apply effects including color overlays."""
    # ... existing code ...
    for seg in segments:
        if seg.label and seg.label in active_effects:
            fx = active_effects[seg.label]
            params = {}
            if fx.get("invert", False):
                params["invert"] = "true"
            if fx.get("color_hex"):
                params["color_hex"] = fx["color_hex"]  # NEW
            seg.effect = EffectData(
                effect_type=fx.get("type", "none"),
                intensity=fx.get("intensity", 1.0),
                params=params,
            )
```

---

### 10. Testing Commands

**Voice/text commands that should work after implementation:**

```
✅ "make person blue"
✅ "color laptop red"
✅ "paint chair green"
✅ "highlight monitor yellow"
✅ "dim everything" (full-screen dim)
✅ "make screen blue" (full-screen color filter)
✅ "warm filter" (full-screen warm)
✅ "blur everything but person" (inverted blur)
✅ "color everything except laptop red" (inverted color)
✅ "stop coloring person" (remove color effect)
✅ "clear" (remove all effects)
```

---

## Implementation Checklist

### Phase 1: Server-Side
- [ ] Update `server/config.py`: Add `"color"` to `EFFECT_TYPES`
- [ ] Update `server/proto/socialsense.proto`: Add `color_hex` to `EffectMetadata` and `FullScreenFilter`
- [ ] Regenerate protobuf Python and C# code
- [ ] Update `server/commands.py`: Add `COLOR_NAMES_TO_HEX` dict and `extract_color_from_text()`
- [ ] Update `server/pipeline/orchestrator.py`: Add `color_hex` parameter to `set_effect()` and `_apply_effects()`
- [ ] Update `server/audio/voice_agent.py`: Update Gemini prompt to recognize color requests

### Phase 2: Unity-Side
- [ ] Regenerate C# protobuf from updated `.proto`
- [ ] Update `OverlayRenderer.cs`: Add `ParseColorHex()` helper
- [ ] Update `OverlayRenderer.cs`: Add `"color"` case to `RenderEffect()`
- [ ] Update `OverlayRenderer.cs`: Pass `color_hex` from `seg.Effect.ColorHex` to `RenderEffect()`
- [ ] Update `FullScreenFilterEffect.cs`: Add color filter support
- [ ] Test: Verify "make person blue" command works end-to-end

### Phase 3: Voice Agent Enhancement
- [ ] Update Gemini prompt with color examples
- [ ] Test: "make laptop red", "color chair green", etc.
- [ ] Verify color names map correctly (blue → #0000FF, etc.)

---

## Example Implementation Flow

**User says:** `"Hey Vibe, make the person blue, thank you"`

1. **Voice Agent** (`voice_agent.py`):
   - Transcribes: `"make the person blue"`
   - Gemini plans: `{"targets": ["person"], "effect_type": "color", "color_hex": "#0000FF", ...}`
   - Calls `_on_voice_command("add", ["person"], "color", 0.8, False)`

2. **Orchestrator** (`orchestrator.py`):
   - `set_effect("person", "color", 0.8, False, "#0000FF")`
   - Adds "person" to active prompts
   - SAM3 segments "person" → mask

3. **Effect Application** (`orchestrator.py`):
   - `_apply_effects()` sets `seg.effect.params["color_hex"] = "#0000FF"`

4. **Protobuf Serialization** (`websocket_server.py`):
   - `proto_seg.effect.color_hex = "#0000FF"`

5. **Unity Client** (`SocialSenseClient.cs`):
   - Receives `ServerMessage` with `seg.Effect.ColorHex = "#0000FF"`

6. **Overlay Renderer** (`OverlayRenderer.cs`):
   - `RenderEffect(..., "color", ..., "#0000FF")`
   - Parses hex → `Color32(0, 0, 255, 180)`
   - Applies blue overlay to person mask

---

## Color Palette Reference

**Common colors users might request:**
- **Primary**: red, green, blue, yellow, cyan, magenta
- **Extended**: orange, purple, pink, brown, tan, beige
- **Light**: light blue, light green, light yellow, light gray
- **Dark**: dark blue, dark green, dark red, dark gray
- **Special**: gold, silver, bronze, coral, emerald, turquoise

**Implementation note:** Use `COLOR_NAMES_TO_HEX` dictionary for all mappings. If user requests unknown color, Gemini should either:
1. Use closest match (e.g., "navy" → "dark blue")
2. Return error: "Unknown color. Try: red, blue, green, yellow, etc."

---

## Full-Screen Filter Examples

**Commands:**
- `"dim everything"` → `full_screen_filter: "dim", intensity: 0.6`
- `"make screen blue"` → `full_screen_filter: "color", color_hex: "#0000FF", intensity: 0.3`
- `"warm filter"` → `full_screen_filter: "warm", intensity: 0.4`
- `"night mode"` → `full_screen_filter: "night", intensity: 0.7`
- `"grayscale"` → `full_screen_filter: "grayscale", intensity: 1.0`

**Unity Implementation:**
- Apply as post-process overlay on entire camera feed
- Use separate shader/material for full-screen effects
- Intensity controls opacity/strength

---

## Backward Compatibility

- ✅ Existing effects (`blur`, `dim`, etc.) continue to work
- ✅ If `color_hex` is missing, use segment's default color
- ✅ Old clients (without color support) ignore `color_hex` field
- ✅ Default behavior: if effect_type is "color" but no color_hex, use segment color

---

## Testing Plan

1. **Unit tests**: Color name → hex mapping
2. **Integration tests**: "make person blue" → verify blue overlay appears
3. **Voice tests**: Test various color requests via voice commands
4. **Full-screen tests**: "dim everything" → verify screen dims
5. **Edge cases**: Unknown colors, malformed hex, missing color_hex

---

## Next Steps

1. Implement Phase 1 (server-side) changes
2. Implement Phase 2 (Unity-side) changes  
3. Update voice agent prompts
4. Test with real voice commands
5. Document new commands in `QUICKSTART_VOICE.md`

---

## Questions / Decisions Needed

- **Q1**: Should "highlight X" default to warm yellow (`highlight` effect) or allow color override?
  - **Proposal**: "highlight X" → warm yellow; "highlight X blue" → color overlay blue
  
- **Q2**: Should color effects support intensity for transparency, or always use fixed alpha?
  - **Proposal**: Use intensity for alpha: `alpha = 180 * intensity` (so intensity controls transparency)

- **Q3**: Should full-screen color filters blend with passthrough or replace it?
  - **Proposal**: Blend (semi-transparent overlay) so passthrough still visible

- **Q4**: Should we support color gradients or just solid colors?
  - **Proposal**: Start with solid colors; gradients can be Phase 2 enhancement
