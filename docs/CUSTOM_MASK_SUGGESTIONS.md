# Custom Mask Suggestions for SocialSenseAR

Suggestions for **custom masks/effects** that fit the pipeline’s real use cases: **privacy**, **focus**, **accessibility**, and **comfort** in AR (e.g. Quest). These go beyond simple blur and are scoped so they can be implemented in Unity with existing or small extensions.

---

## Use Cases This Pipeline Serves

- **Privacy** – Hide or soften screens, documents, faces in shared or public spaces.
- **Focus** – Reduce distraction from background or specific objects (e.g. “dim everything but the whiteboard”).
- **Accessibility / comfort** – Reduce overstimulation, brightness, or clutter (“environment too bright”, “dim it”).
- **Social / situational** – Quickly turn on/off what’s visible (e.g. “blur all screens” in a meeting).

---

## 1. Privacy-Oriented Masks

### **Redaction bar**
- **What:** Solid black (or dark) horizontal/vertical bars over the masked region, like TV redaction.
- **Use case:** Hide screens or documents without blur; reads as “intentionally censored.”
- **Implementation:** Fill mask with black; optionally subdivide into 2–3 bars. No texture needed.
- **Voice:** “Redact the screen”, “Censor the laptop.”

### **Frosted glass**
- **What:** Blur + slight lightening/whiten so it reads as “frosted glass” rather than soft blur.
- **Use case:** Softer privacy than pixelate; good for faces or shared screens.
- **Implementation:** Same as blur path, then tint toward white (e.g. blend with 0.2 white overlay).
- **Voice:** “Frost the screen”, “Frosted glass on the monitor.”

### **Noise / static**
- **What:** TV-style static or fine noise inside the mask.
- **Use case:** Strong “content hidden” signal; useful for strict privacy.
- **Implementation:** Fill mask with random gray pixels (or tile a small noise texture). Can reuse pixelate-style block read but with random values per block.
- **Voice:** “Static on the screen”, “Noise the laptop.”

### **Placeholder / label**
- **What:** Replace the region with a solid color + optional short label (e.g. “Screen”, “Person”).
- **Use case:** Indicate “something is here” without showing content; good for UX and accessibility.
- **Implementation:** Fill mask with one color; optionally draw text at segment center (e.g. segment label). Reuse existing text/label infra if you have it.
- **Voice:** “Replace screen with placeholder”, “Show label on person.”

---

## 2. Focus & Attention Masks

### **Spotlight (inverted dim)**
- **What:** Darken everything *except* the selected object(s); soft falloff at boundary.
- **Use case:** “Focus on the whiteboard” / “dim everything but the person talking.”
- **Implementation:** You already have invert logic; ensure dim-with-invert uses a soft edge (e.g. alpha falloff at mask boundary) so it feels like a spotlight, not a hard cut.
- **Voice:** “Spotlight the whiteboard”, “Focus on the screen”, “Dim everything but me.”

### **Desaturate (grayscale)**
- **What:** Grayscale the masked region (or, inverted, grayscale everything except the target).
- **Use case:** Reduce visual load; make the “kept” object pop by being the only color.
- **Implementation:** In the mask region, sample passthrough, convert RGB to luminance, write gray. Or full-screen desaturate with invert so only the segment stays in color.
- **Voice:** “Desaturate the background”, “Grayscale everything but the person.”

### **Vignette**
- **What:** Darken or desaturate the *edges* of the frame, leaving center clearer.
- **Use case:** Reduce peripheral distraction; comfort in busy environments.
- **Implementation:** Full-screen effect: radial gradient from center (no segmentation). Intensity = strength of darkening at edges.
- **Voice:** “Vignette”, “Darken the edges”, “Less peripheral distraction.”

### **Outline only (minimal)**
- **What:** Only the contour of the object (no fill). You may already have this; treat as “minimal” mode.
- **Use case:** Show “where things are” without covering content; low visual weight.
- **Implementation:** Same as outline effect; ensure line is thin and optional.
- **Voice:** “Just outline the person”, “Outline only.”

---

## 3. Accessibility & Comfort Masks

### **Reduce contrast**
- **What:** Flatten contrast inside the mask (bring midtones toward gray).
- **Use case:** Light sensitivity; reduce glare from screens or windows.
- **Implementation:** In mask region: blend original with 50% gray by some factor (e.g. `lerp(sample, gray, 0.5)`). Simple and no extra textures.
- **Voice:** “Soften the screen”, “Reduce contrast on the window.”

### **Warm / cool tint (per-object)**
- **What:** Color tint overlay on the masked region (warm orange vs cool blue).
- **Use case:** Comfort (warm = cozy, cool = focus); can pair with “make X blue” color overlay.
- **Implementation:** Same as color overlay; add effect types `warm_tint` and `cool_tint` with fixed hex (e.g. warm #FFE4C4, cool #ADD8E6) and low alpha.
- **Voice:** “Warm tint on the lamp”, “Cool the screen.”

### **Soft glow**
- **What:** Subtle glow around the object (halo), not a solid fill.
- **Use case:** Highlight “this is important” without blocking the view (e.g. highlight a person or door).
- **Implementation:** Dilate mask by a few pixels, then draw with low-alpha color (or blur the dilated mask and use as alpha). Single pass, no complex shader.
- **Voice:** “Glow the door”, “Highlight the person with a glow.”

### **Pulse / breathe**
- **What:** Effect intensity or alpha slowly pulses (e.g. 0.6 ↔ 1.0 over ~1.5 s).
- **Use case:** Gentle “pay attention here” without being harsh.
- **Implementation:** In Update, `alpha = 0.8 + 0.2 * sin(time * 2π / period)`; apply to existing dim or color overlay.
- **Voice:** “Pulse the screen”, “Gentle pulse on the person.”

---

## 4. Informational & UX Masks

### **Pattern fill**
- **What:** Hatch or stripe pattern inside the mask (e.g. diagonal lines, dots).
- **Use case:** “Something is here” without showing content; good for diagrams or “under construction” feel.
- **Implementation:** Fill mask; in masked pixels, use `(x + y) % period` or similar to choose pattern color. No texture; procedural.
- **Voice:** “Hatch the screen”, “Stripes on the laptop.”

### **Blur + border**
- **What:** Blur inside + solid colored border (e.g. red/green) on the contour.
- **Use case:** “This is censored” + clear boundary; useful for compliance or demos.
- **Implementation:** Existing blur + outline; outline color could be a parameter (e.g. “red border”).
- **Voice:** “Blur the screen with a red border”, “Blur and outline the monitor.”

### **Mask-only (alpha)**
- **What:** Don’t render the segment at all; only use it for logic (e.g. “hide person from view”).
- **Use case:** Full removal of object from view (e.g. replace with passthrough background or black).
- **Implementation:** For “hide” effect: in mask region, don’t draw overlay (or draw black / passthrough). Requires one new effect type, e.g. `hide`.
- **Voice:** “Hide the screen”, “Remove the person from view.”

---

## 5. Full-Screen Filters (Already Partially There)

These apply to the **whole frame** and fit “comfort” and “focus”:

- **Dim** – Reduce overall brightness (you have this).
- **Warm / cool / night** – Color temperature or night shift (you have or can extend).
- **Grayscale** – Full-screen desaturate.
- **Color tint** – Full-screen overlay color at low alpha (e.g. “make the whole view slightly blue”).
- **Vignette** – Darken edges only (see above).

---

## Suggested Priority for Implementation

| Priority | Effect            | Use case      | Complexity | Notes                          |
|----------|-------------------|---------------|------------|---------------------------------|
| 1        | Redaction bar     | Privacy       | Low        | Solid fill, 2–3 bars            |
| 2        | Spotlight (soft) | Focus         | Low        | Invert + soft edge on dim       |
| 3        | Desaturate        | Focus/comfort | Medium     | Grayscale in/out of mask       |
| 4        | Frosted glass     | Privacy       | Low        | Blur + white tint               |
| 5        | Soft glow         | Accessibility | Medium     | Dilate + low-alpha draw         |
| 6        | Reduce contrast   | Comfort       | Low        | Blend with gray                 |
| 7        | Noise/static      | Privacy       | Low        | Random or tiled noise           |
| 8        | Pulse             | UX            | Low        | Animate intensity/alpha         |
| 9        | Pattern fill      | Informational | Low        | Procedural hatch/stripes        |
| 10       | Vignette          | Comfort       | Low        | Full-screen radial gradient     |

---

## Implementation Notes

- **No new segmentation:** All of these use the same segment masks (from SAM3); only the **rendering** of the mask changes.
- **Unity:** Prefer **procedural or simple shaders** (fill with color, blend with gray, dilate mask) so you don’t depend on many new textures.
- **Voice agent:** Extend the effect vocabulary (e.g. “redact”, “frost”, “spotlight”, “desaturate”, “glow”, “pulse”, “vignette”) and map them to these effect types in the same way you do “blur” and “dim.”
- **Protobuf:** Add new `effect_type` values (e.g. `redact`, `frost`, `spotlight`, `desaturate`, `glow`, `pulse`, `vignette`) and, where needed, a small set of parameters (e.g. `border_color_hex` for “blur with red border”).

If you tell me which 2–3 effects you want first (e.g. redaction, spotlight, desaturate), I can outline concrete changes in the server and Unity (effect type, payload, and render path) step by step.
