# Suggestions: Border Smoothness & Latency (No Existing Changes)

**Purpose:** Ideas to improve segmentation border smoothness and end-to-end latency **on top of** the current pipeline. Everything below is additive (new config options, optional code paths, or shader tweaks). Do **not** remove or alter existing behavior.

---

## Part 1: Border Smoothness (Segmentation Mask Edges)

### Where borders come from today

- **Server:** Masks are resized by `rle_scale` (0.75), then MORPH_CLOSE, MORPH_OPEN, GaussianBlur (5,5), then thresholded to binary and RLE-encoded. So edges are already smoothed once, but the **encoded mask is binary** (0 or 255).
- **Unity:** RLE is decoded into a **boolean** mask (`_maskTemp`). Outlines and fills are drawn from this binary mask, so edges are **hard** (no alpha gradient).

Rough edges can come from: (1) low RLE resolution, (2) binary threshold, (3) Unity drawing with no edge softening.

---

### 1.1 Server-side (optional config only)

**A. Stronger pre-encode blur (softer binary edge)**

- **Where:** `server/pipeline/orchestrator.py` → `_encode_rle_all()`.
- **Current:** `cv2.GaussianBlur(small_u8, (5, 5), 0)` then `(small_u8 > 128)`.
- **Suggestion:** Add a config knob, e.g. `rle_edge_blur_kernel: int = 5`. If set to 7 or 9, use `(rle_edge_blur_kernel, rle_edge_blur_kernel)` for the blur. **Default stays 5** so behavior is unchanged unless the user opts in.
- **Effect:** Slightly softer transition before threshold → slightly smoother binary mask, same RLE format and client logic.

**B. Optional higher RLE resolution**

- **Where:** `server/config.py` → `rle_scale`.
- **Current:** `rle_scale = 0.75`.
- **Suggestion:** Document that **increasing** `rle_scale` (e.g. 0.9 or 1.0) improves edge sharpness at the cost of larger payloads and a bit more encode time. No code change required; just a config/env choice for “quality vs latency.”

**C. Avoid extra morphology when not needed**

- **Where:** Same `_encode_rle_all()`.
- **Suggestion (optional):** Add a config flag, e.g. `rle_smooth_edges_only: bool = False`. When True, skip MORPH_CLOSE/MORPH_OPEN and only do GaussianBlur before threshold. Some masks look smoother with blur-only; others need morphology. **Default False** keeps current behavior.

---

### 1.2 Unity: optional edge softening after decode

**A. Soft outline (alpha by distance to edge)**

- **Where:** `OverlayRenderer.cs` → `DrawOutlineFromMask()`.
- **Current:** Pixels at the mask edge are drawn with full opacity; interior is not drawn (outline only).
- **Suggestion (additive):** Add an optional **soft-outline** path (e.g. gated by a serialized field `useSoftOutline` default false). In that path, for each pixel in a band around the mask boundary, compute a **distance-to-edge** (e.g. how many pixels to the nearest non-mask pixel). Map that to alpha (e.g. `alpha = clamp(distance / softWidth, 0, 1)`). Write `Color32(r, g, b, (byte)(255 * alpha))` into `_pixelBuffer`. Rest of the pipeline (same buffer, same texture) stays the same.
- **Effect:** Softer-looking outline; no change to existing behavior when the option is off.

**B. Optional 1px blur of the overlay texture**

- **Where:** After `_overlayTexture.SetPixels32(_pixelBuffer)` and `Apply(false)` in `UpdateOverlay()`.
- **Suggestion (additive):** Add an optional “edge soften” step: create a temporary RenderTexture of the same size, blit `_overlayTexture` through a simple 3×3 or 5×5 blur material (e.g. copy of your existing Gaussian blur with very small sigma), then copy the result back to `_overlayTexture` (or use the blurred RT as the material’s main texture). Gate this behind a field like `overlayEdgeSofteningBlurSize` (0 = off). When 0, do nothing.
- **Effect:** Slightly softer edges in the final overlay; default 0 keeps current behavior.

**C. Shader-level soft sampling (SegmentOverlay)**

- **Where:** `Assets/Scripts/Shaders/SegmentOverlay.shader` (or equivalent).
- **Current:** Fragment shader samples the overlay texture once.
- **Suggestion (additive):** Add an optional **2×2 or 3×3 box sample** (e.g. 4 or 9 taps with small offsets, average). Expose a keyword or float like `_SoftEdges` (0 = single sample, 1 = multi-sample). When 0, keep current single-sample. This softens aliasing at mask edges without changing server or RLE.

---

### 1.3 RLE decode (Unity) – keep format, optional quality

- **Current:** `DecodeRleIntoMask()` fills `_maskTemp` (bool) from RLE; outline/fill use this directly.
- **Suggestion:** Leave this as-is. If you later add a “soft mask” path (e.g. float array for alpha), that can be a **separate** code path filled from the same RLE and used only when a “soft edges” option is on. Default path continues to use `_maskTemp` only → no change to existing behavior.

---

## Part 2: Latency Suggestions

Again: only suggestions (config, optional paths, or documentation). No required changes to existing logic.

### 2.1 Server

- **rle_scale:** Lower (e.g. 0.5) → smaller RLE and faster encode/network; higher (0.9–1.0) → sharper edges, more bytes. Document this trade-off; default stays 0.75.
- **target FPS / batching:** If the client sends frames at 30 fps, the server already returns cached results when segments haven’t changed. Ensure you’re not doing redundant work in the hot path (e.g. avoid re-encoding when segment list identity hasn’t changed); your existing cache-by-`id(result.segments)` is in the right direction.
- **First-frame latency:** SAM warm-up on first connection dominates. Document that “first request after connect may take 1–2 seconds” and that subsequent frames are fast. Optional: expose a “warmup” endpoint that triggers pipeline init without a real frame so the first user frame is already warm.

### 2.2 Network

- **Message size:** RLE is already compact. If you add optional compression (e.g. deflate of the binary message), make it opt-in so existing clients remain unchanged.
- **WebSocket:** Keep a single connection; avoid extra round-trips. No change to your current design.

### 2.3 Unity client

- **RLE decode on worker thread:** Decoding RLE in `DecodeRleIntoMask()` can be moved to a background thread or Unity Job so the main thread doesn’t stall. **Additive:** decode into a temporary buffer, then on the main thread copy into `_maskTemp` and continue. Existing single-threaded path can remain the default.
- **Update overlay every N frames:** Add an option (e.g. “overlay update rate”) to apply `UpdateOverlay()` every 2nd or 3rd frame when latency is more important than smoothness. Default 1 (every frame) preserves current behavior.
- **Resolution:** You already have `resolutionScale` for capture. Document that lower resolution reduces server work and network; higher improves sharpness. No code change required.

### 2.4 Config summary (additive only)

| Knob | Where | Default | Effect |
|------|--------|--------|--------|
| `rle_scale` | server config | 0.75 | ↑ = sharper edges, more bytes; ↓ = faster, softer |
| `rle_edge_blur_kernel` | server config (new, optional) | 5 | 7 or 9 = softer binary edge before RLE |
| `rle_smooth_edges_only` | server config (new, optional) | false | true = blur-only, no morphology |
| `useSoftOutline` | Unity OverlayRenderer (new, optional) | false | true = outline alpha by distance to edge |
| `overlayEdgeSofteningBlurSize` | Unity OverlayRenderer (new, optional) | 0 | 3 or 5 = 1px blur on overlay texture |
| SegmentOverlay shader `_SoftEdges` | shader (new, optional) | 0 | 1 = multi-tap sample for softer sampling |

---

## Part 3: Implementation checklist (additive only)

**Border smoothness**

- [ ] Add optional `rle_edge_blur_kernel` (default 5) and use it in `_encode_rle_all`; leave 5 as default.
- [ ] Add optional `useSoftOutline` on OverlayRenderer; when true, draw outline with alpha from distance-to-edge; default false.
- [ ] Optionally add 1px blur pass on overlay texture when `overlayEdgeSofteningBlurSize` > 0; default 0.
- [ ] Optionally add multi-tap sampling in SegmentOverlay shader when `_SoftEdges` is set; default off.

**Latency**

- [ ] Document `rle_scale` and resolution vs latency in README or a short “Tuning” doc.
- [ ] Optionally move RLE decode to a worker/Job and copy result to `_maskTemp` on main thread; keep existing path as fallback.
- [ ] Optionally add “overlay update rate” (1 = every frame) so heavy devices can update overlay every 2nd frame.

**Do not**

- Change existing default values for `rle_scale`, blur kernel, or morphology.
- Remove or replace the current outline/fill logic; only add alternative paths behind flags.
- Change RLE format or protobuf; keep compatibility with existing clients.

---

## Summary

- **Borders:** Softer edges can come from (1) optional stronger blur before binary threshold on the server, (2) optional soft-outline (distance-to-edge alpha) in Unity, (3) optional 1px blur of the overlay texture, (4) optional multi-tap sampling in the overlay shader. All behind new options with safe defaults.
- **Latency:** Improvements are config (rle_scale, resolution), optional threading for RLE decode, and optional overlay update rate—all additive, with no change to current defaults or behavior.

Implementing only the items you want keeps the rest of the pipeline and existing functionality unchanged.
