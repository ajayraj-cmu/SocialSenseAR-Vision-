# Implementation Prompt: Expand SocialSenseAR Effect System

**Use this prompt with Claude (or another coding assistant) to implement the expanded effect pipeline.**

---

## Goal

Extend the SocialSenseAR pipeline so that:

1. **Color overlays** are supported: user can say "make [object] blue" (or any color) and the segmented object gets a semi-transparent color mask.
2. **Voice agent / Gemini** maps color requests and full-screen requests to the correct effect type and parameters.
3. **Full-screen filters** work (dim, warm, cool, night, grayscale, and optional color tint over entire screen).

## Reference Spec

**Read and implement according to:** `ServerBackend/docs/EFFECT_SYSTEM_EXPANSION.md`

That document defines:
- New effect type `color` with `color_hex` (or RGB)
- `COLOR_NAMES_TO_HEX` dictionary for mapping color names → hex
- Protobuf changes (`color_hex` on `EffectMetadata` and `FullScreenFilter`)
- Voice agent prompt updates so Gemini returns `effect_type: "color"` and `color_hex` for phrases like "make person blue"
- Unity `OverlayRenderer` changes: `ParseColorHex()`, new `"color"` case in `RenderEffect()`, pass `color_hex` from segment effect
- Full-screen filter enhancements (including optional color)

## Implementation Order

1. **Server:** Add `color` to `EFFECT_TYPES` in `server/config.py`.
2. **Protobuf:** Add `color_hex` to `EffectMetadata` (and `FullScreenFilter` if adding full-screen color). Regenerate Python and C#.
3. **Server commands:** Add `COLOR_NAMES_TO_HEX` and color-extraction logic in `server/commands.py`.
4. **Orchestrator:** In `server/pipeline/orchestrator.py`, support `color_hex` in `set_effect()` and when building `EffectData` for segments.
5. **Voice agent:** In `server/audio/voice_agent.py`, update the Gemini JSON prompt so it:
   - Includes `effect_type: "color"` and `color_hex` for color requests.
   - Maps "make X blue" → targets: ["X"], effect_type: "color", color_hex: "#0000FF".
   - Knows full-screen filters: "dim everything", "make screen blue", etc.
6. **Unity:** In `OverlayRenderer.cs`, add `ParseColorHex()`, handle `effect_type == "color"` in `RenderEffect()` using `seg.Effect.ColorHex`, and ensure C# protobuf has `ColorHex` on the effect message.
7. **Full-screen (optional):** In Unity `FullScreenFilterEffect`, support a "color" filter type using `full_screen_color` / `color_hex` if present in the conversation state.

## Success Criteria

- User says "make person blue" (or types it in the Editor HUD) → person segment gets a blue semi-transparent overlay.
- User says "color laptop red" → laptop gets red overlay.
- User says "dim everything" → full-screen dim filter applies.
- Existing effects (blur, dim, pixelate, highlight, outline) still work.
- All colors in `COLOR_NAMES_TO_HEX` (red, green, blue, yellow, orange, purple, etc.) are supported when user requests them.

## Constraints

- Do not break existing protobuf fields; add new optional fields only.
- Unity must handle missing `color_hex` (fall back to segment color).
- Voice agent must remain backward compatible: if Gemini does not return `color_hex`, treat as before.
