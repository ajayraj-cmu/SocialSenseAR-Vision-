# Dynamic Color Intensity System

This document describes the enhanced color and intensity system that allows voice commands to control **both** the color AND the intensity/brightness of effects.

---

## Overview

The system now supports:
1. **Intensity modifiers** - "extremely", "very", "slightly", etc. control effect strength
2. **Brightness modifiers** - "dark", "bright", "pale" control color luminosity
3. **Dynamic alpha mapping** - Color overlay opacity scales from 39% to 100% based on intensity
4. **Gemini AI extraction** - Voice commands automatically extract intensity and brightness

---

## Problem Statement

### Before
- Requesting "make ceiling extremely dark black" would apply the same color as "make ceiling black"
- Color intensity was fixed at 0.8 (giving max 70% opacity via alpha = 180)
- Dark colors looked washed out because they couldn't reach full opacity
- No way to request subtle vs. strong color effects

### After
- "extremely dark black" → RGB(0,0,0) with intensity 1.0 → **100% opacity black**
- "slightly dim wall" → intensity 0.3 → 54% opacity
- "very bright red" → RGB(230,0,0) with intensity 0.9 → 96% opacity
- Full control over both color hue and effect strength

---

## System Architecture

### 1. Command Parsing (`commands.py`)

**New Functions:**

```python
extract_intensity_modifier(text: str) -> tuple[float | None, str]
```
Extracts intensity keywords and returns multiplier (0.0-1.0):

| Keyword | Intensity | Use Case |
|---------|-----------|----------|
| "extremely", "super", "maximum" | 1.0 | Maximum effect |
| "very", "really", "heavily" | 0.9 | Strong effect |
| "quite", "pretty" | 0.8 | Default strength |
| "moderately", "somewhat" | 0.6 | Moderate effect |
| "a bit", "a little" | 0.4 | Mild effect |
| "slightly", "barely", "subtly" | 0.3 | Minimal effect |
| (none) | 0.8 | Default |

```python
extract_brightness_modifier(text: str) -> tuple[float | None, str]
```
Extracts brightness keywords for color adjustment:

| Keyword | Brightness Multiplier | Example |
|---------|----------------------|---------|
| "extremely dark" | 0.1 | Red #FF0000 → #1A0000 |
| "very dark" | 0.2 | Red #FF0000 → #330000 |
| "dark" | 0.4 | Red #FF0000 → #660000 |
| "deep" | 0.5 | Red #FF0000 → #800000 |
| (normal) | 1.0 | Red #FF0000 → #FF0000 |
| "bright" | 0.8 + blend | Red #FF0000 → ~#FF3333 |
| "very bright" | 0.9 + blend | Red #FF0000 → ~#FF1A1A |
| "pale", "pastel" | 0.7 + blend | Red #FF0000 → ~#FF4D4D |

```python
adjust_color_brightness(hex_color: str, brightness: float) -> str
```
Applies brightness multiplier to RGB values:
- Multiplies each channel (R, G, B) by brightness
- Clamps to 0-255 range
- Returns adjusted hex string

**Example:**
```python
>>> adjust_color_brightness("#FF0000", 0.2)  # very dark red
"#330000"

>>> adjust_color_brightness("#000000", 0.1)  # extremely dark black
"#000000"  # already black, stays black
```

### 2. Voice Agent (`voice_agent.py`)

**Enhanced Gemini Prompt:**

The Gemini planning model now receives detailed instructions on:

**Intensity Extraction:**
```
"extremely blur" → intensity: 1.0
"slightly dim" → intensity: 0.3
"really blur background" → intensity: 0.9
```

**Brightness Extraction for Colors:**
```
"extremely dark black ceiling" → color_hex: "#000000", intensity: 1.0
"very bright red wall" → color_hex: "#E60000" (brightened), intensity: 0.9
"pale blue chair" → color_hex: "#B3D9F2" (lightened), intensity: 0.8
```

The model is explicitly instructed to:
1. Parse intensity modifiers and set `intensity` field accordingly
2. Parse brightness modifiers and **adjust the color_hex value**
3. Distinguish between effect intensity (opacity) and color brightness (RGB values)

### 3. Unity Rendering (`OverlayRenderer.cs`)

**New Alpha Mapping for Color Effects:**

```csharp
// Before: byte colorAlpha = (byte)(180 * intensity);  // Max 180/255 = 70%

// After:
byte colorAlpha = (byte)Mathf.Clamp(100f + 155f * intensity, 0f, 255f);
```

**Alpha Range:**

| Intensity | Alpha | Opacity | Use Case |
|-----------|-------|---------|----------|
| 0.0 | 100 | 39% | Subtle tint |
| 0.3 | 146 | 57% | Light overlay |
| 0.5 | 177 | 69% | Moderate overlay |
| 0.8 | 224 | 88% | Strong color |
| 1.0 | 255 | 100% | **Fully opaque** (critical for dark colors) |

**Why This Matters:**
- Dark colors like black (#000000) need 100% opacity to appear truly dark
- Previous max 70% opacity made "black" look dark gray
- New range allows full spectrum from subtle tint (39%) to complete coverage (100%)

---

## Examples & Test Cases

### Color + Brightness + Intensity

| Voice Command | Parsed Values | Result |
|---------------|---------------|--------|
| "make ceiling black" | color: #000000, intensity: 0.8 | Black at 88% opacity (dark gray-ish) |
| "make ceiling extremely dark black" | color: #000000, intensity: 1.0 | Black at **100% opacity** (true black) |
| "make ceiling very dark black" | color: #1A1A1A, intensity: 0.9 | Very dark gray at 96% opacity |
| "make wall bright red" | color: #FF3333 (brightened), intensity: 0.8 | Bright red at 88% opacity |
| "make wall very bright red" | color: #FF1A1A (very bright), intensity: 0.9 | Very bright red at 96% opacity |
| "make floor pale blue" | color: #B3D9F2 (lightened), intensity: 0.8 | Pale blue at 88% opacity |
| "slightly color chair green" | color: #00FF00, intensity: 0.3 | Green at 57% opacity (subtle tint) |

### Intensity-Only Effects

| Voice Command | Effect Type | Intensity | Result |
|---------------|-------------|-----------|--------|
| "blur laptop" | blur | 0.8 | Default blur |
| "extremely blur laptop" | blur | 1.0 | Maximum blur (sigma 180) |
| "slightly blur laptop" | blur | 0.3 | Light blur (sigma 96) |
| "very dim screen" | dim | 0.9 | Strong darkening |
| "a bit dim screen" | dim | 0.4 | Light darkening |

### Combined Modifiers

| Voice Command | Parsed | Explanation |
|---------------|--------|-------------|
| "extremely dark blue wall" | color: #00001A (darkened blue), intensity: 1.0 | Both "extremely" (intensity) and "dark" (brightness) apply |
| "very bright pale yellow" | color: ~#FFFFE6 (bright pale yellow), intensity: 0.9 | "very" → intensity, "bright pale" → brightness |
| "slightly dark red ceiling" | color: #660000 (darkened), intensity: 0.3 | "slightly" → intensity, "dark" → brightness |

---

## Implementation Details

### File Changes

**1. `/ServerBackend/server/commands.py`**
- Added `INTENSITY_MODIFIERS` dictionary (12 modifiers)
- Added `BRIGHTNESS_MODIFIERS` dictionary (10 modifiers)
- Added `extract_intensity_modifier()` function
- Added `extract_brightness_modifier()` function
- Added `adjust_color_brightness()` function
- Updated `extract_color_from_text()` to return brightness modifier
- Updated `parse_command()` to return intensity as 5th tuple element

**2. `/ServerBackend/server/audio/voice_agent.py`**
- Enhanced Gemini prompt with intensity extraction rules
- Enhanced Gemini prompt with brightness adjustment rules
- Added detailed examples showing intensity + brightness combinations
- Gemini now outputs adjusted color_hex values (not just base colors)

**3. `/unitySetUp/SocialSenseAR-Unity/Assets/Scripts/OverlayRenderer.cs`**
- Increased color alpha range from 0-180 to 100-255
- Formula: `alpha = 100 + 155 * intensity`
- Allows full opacity (255) at intensity 1.0
- Minimum 39% opacity ensures subtle colors are still visible

---

## Usage Guide

### For End Users (Voice Commands)

**Basic Colors:**
```
"make ceiling black"
"color wall red"
"paint floor blue"
```

**Dark Colors (High Intensity):**
```
"make ceiling extremely dark black"     ← Use this for true black
"make wall very dark gray"
"color screen deep blue"
```

**Bright Colors:**
```
"make lamp very bright yellow"
"paint chair bright red"
"color table extremely bright white"
```

**Soft/Pale Colors:**
```
"make wall pale blue"
"color chair pastel pink"
"paint ceiling light gray"
```

**Variable Intensity (Any Effect):**
```
"slightly blur laptop"                  ← Light blur
"extremely blur background"             ← Maximum blur
"very dim monitor"                      ← Strong dimming
"a bit pixelate face"                   ← Minimal pixelation
```

### For Developers (API)

**CommandPlan Structure:**
```python
@dataclass
class CommandPlan:
    targets: list[str]
    effect_type: str
    intensity: float  # 0.0-1.0, extracted from modifiers
    action: str
    invert: bool
    color_hex: Optional[str]  # Adjusted hex value with brightness applied
    # ...
```

**Example Plan Output:**
```json
{
  "targets": ["ceiling"],
  "effect_type": "color",
  "color_hex": "#000000",
  "intensity": 1.0,
  "action": "add",
  "invert": false,
  "reasoning": "Extremely dark black ceiling → max intensity + black color"
}
```

---

## Alpha Mapping Comparison

### Color Effect

| Intensity | Old Alpha | Old % | New Alpha | New % | Difference |
|-----------|-----------|-------|-----------|-------|------------|
| 0.0 | 0 | 0% | 100 | 39% | +39% (now visible) |
| 0.3 | 54 | 21% | 146 | 57% | +36% |
| 0.5 | 90 | 35% | 177 | 69% | +34% |
| 0.8 | 144 | 56% | 224 | 88% | +32% |
| 1.0 | 180 | **70%** | 255 | **100%** | **+30%** ⭐ |

**Key Improvement:** At maximum intensity, color overlays are now **fully opaque** instead of 70% transparent. This is critical for dark colors.

### Other Effects (Unchanged)

| Effect | Formula | Max Opacity |
|--------|---------|-------------|
| Blur mask | `240 + 15 * intensity` | 100% (255) |
| Dim | `200 * intensity` | 78% (200) |
| Highlight | `76 * intensity` | 30% (76) |
| Pixelate | `blockSize * intensity` | N/A (pattern-based) |

---

## Edge Cases & Notes

### Black Color Special Case
- Black (#000000) cannot be "darkened" further
- "extremely dark black" → same RGB but **intensity 1.0** → 100% opacity
- Previous system: black at 70% opacity looked like dark gray
- New system: black at 100% opacity looks truly black

### White Color Special Case
- White (#FFFFFF) is already maximum brightness
- "bright white" or "extremely bright white" → same RGB, high intensity
- Effect is more about **opacity** than brightness

### Overlapping Modifiers
- "very dark blue" → "very" affects intensity (0.9), "dark" affects brightness (0.4)
- "slightly bright red" → "slightly" affects intensity (0.3), "bright" affects brightness (0.8)
- Both are applied independently

### Gemini Fallback
- If Gemini is unavailable, fast-path parsing uses default intensity 0.8
- Brightness modifiers are still extracted by `commands.py` functions
- Voice agent emergency parser doesn't extract intensity (uses 0.8 default)

---

## Testing Recommendations

### Test Matrix

| Test Case | Expected Outcome |
|-----------|------------------|
| "make ceiling black" | Dark gray overlay (88% opacity) |
| "make ceiling extremely dark black" | True black (100% opacity) |
| "make ceiling extremely black" | True black (100% opacity, "extremely" as intensity) |
| "make wall very dark red" | Very dark red at 96% opacity |
| "make wall bright blue" | Bright blue at 88% opacity |
| "slightly blur laptop" | Light blur (sigma ~96) |
| "extremely blur laptop" | Maximum blur (sigma 180) |
| "very dim screen" | Strong darkening (90% intensity) |

### Visual Verification

**Ceiling Test (Dark Black):**
1. Say "make ceiling black" → Should see dark gray overlay
2. Say "make ceiling extremely dark black" → Should see **true black** (no gray)
3. Difference should be visually obvious

**Wall Test (Bright Colors):**
1. Say "make wall red" → Normal red
2. Say "make wall bright red" → Noticeably lighter/brighter red
3. Say "make wall pale red" → Pink-ish light red

**Intensity Test (Blur):**
1. Say "slightly blur laptop" → Subtle blur
2. Say "blur laptop" → Normal blur
3. Say "extremely blur laptop" → Maximum blur
4. Background should be progressively less recognizable

---

## Migration & Compatibility

### Backward Compatibility
- All existing voice commands continue to work
- "blur X", "dim X", "color X red" use default intensity 0.8
- New modifiers are **opt-in** (user must say them)

### Breaking Changes
- **None** - return signature of `parse_command()` changed but callers updated
- Color alpha range increased (may look more opaque, but this is the fix!)

### Server Updates Required
- Update `commands.py` (already done)
- Update `voice_agent.py` Gemini prompt (already done)
- Restart voice agent service to load new prompt

### Unity Client Updates Required
- Rebuild with updated `OverlayRenderer.cs` (already done)
- No configuration changes needed

---

## Troubleshooting

### "Dark black ceiling still looks gray"

**Check:**
1. Verify voice command includes "extremely" or intensity modifier
2. Check server logs for parsed intensity value (should be 1.0)
3. Verify Unity client rebuilt with new alpha formula
4. Test with debug: manually set intensity to 1.0 via dashboard

**Solution:**
- Use explicit intensity modifier: "extremely dark black ceiling"
- Ensure Gemini is parsing intensity correctly (check logs)

### "Bright colors don't look bright enough"

**Check:**
1. Verify brightness modifier in command ("bright", "very bright")
2. Check if color_hex was adjusted (should be lighter than base color)
3. Gemini logs should show reasoning about brightness adjustment

**Solution:**
- Use "very bright" or "extremely bright" for maximum effect
- Check that Gemini prompt includes brightness rules

### "Colors are too opaque now"

**Solution:**
- Use lower intensity modifiers: "slightly", "a bit", "moderately"
- For subtle tints, say "slightly color X [color]"
- Default intensity is 0.8 (88% opacity) - use "a little" for 40%

---

## Future Improvements

### Potential Enhancements
1. **Saturation control:** "vivid red", "muted blue"
2. **Gradient effects:** "dark red fading to light red"
3. **Color mixing:** "purple with a hint of blue"
4. **Temperature:** "warm white", "cool white"
5. **Per-channel control:** "red with green tint"

### Performance Optimizations
1. Cache brightness-adjusted colors to avoid recalculation
2. Precompute common color variants (dark/bright for all base colors)
3. Use lookup tables for intensity→alpha mapping

---

## Summary

The new color intensity system provides:

✅ **Dynamic intensity** - "extremely", "very", "slightly" control effect strength
✅ **Brightness control** - "dark", "bright", "pale" adjust color luminosity
✅ **Full opacity range** - Color effects can now be 100% opaque
✅ **True dark colors** - "extremely dark black" is now truly black
✅ **Natural language** - Gemini AI extracts modifiers automatically
✅ **Backward compatible** - Existing commands unchanged

**Before:** "make ceiling black" → dark gray
**After:** "make ceiling extremely dark black" → **true black**

This addresses the core issue where dark colors appeared washed out due to limited opacity range.
