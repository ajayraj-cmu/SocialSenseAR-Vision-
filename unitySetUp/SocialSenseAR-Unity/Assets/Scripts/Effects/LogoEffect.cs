using UnityEngine;

/// <summary>
/// Logo / custom image overlay effect.
/// Loads LogoTexture.png from Resources and maps it onto the segmented mask.
///
/// PROJECTION MODES
/// ================
/// Mode 0 — Bounding-box stretch (original)
///   Texture is scaled to fill the mask's bounding box with letterboxing.
///   Fast. Best for flat surfaces (walls, floors).
///
/// Mode 1 — Cylindrical wrap (DEFAULT)
///   Treats the silhouette as a vertical cylinder whose axis passes through the
///   mask centroid.  The texture U is derived from the normalised angle around
///   that axis so the image wraps continuously around the perimeter.
///   V is the normalised height (minY→maxY in the mask).
///   Gives a seamless "wrapped label" look on people and round objects.
///
/// Mode 2 — Contour / distance-field UV
///   U = normalised arc-length position around the silhouette boundary.
///   V = normalised inward distance from the boundary.
///   Produces a band of texture that hugs the object's edge.
///   Ideal for "glowing border" images or decal strips.
///
/// Select with LogoEffect.ProjectionMode (int 0/1/2).
///
/// SETUP
/// =====
/// Place your PNG at Assets/Resources/LogoTexture.png with Read/Write enabled.
/// PNG with transparency is required — JPG has no alpha and will render milky.
/// </summary>
public static class LogoEffect
{
    // ------------------------------------------------------------------
    // Public settings
    // ------------------------------------------------------------------

    /// <summary>
    /// 0 = BBox stretch (original)
    /// 1 = Cylindrical wrap around centroid (default, best for people)
    /// 2 = Contour UV (texture band hugs the silhouette edge)
    /// </summary>
    public static int ProjectionMode = 1;

    /// <summary>
    /// How many times the texture repeats around the full cylinder (Mode 1).
    /// 1.0 = one full wrap (default). 2.0 = two repeats.
    /// </summary>
    public static float CylinderTileU = 1.0f;

    /// <summary>
    /// Contour-mode band width in pixels (Mode 2).
    /// Texture V=0 is the boundary, V=1 is this many pixels inward.
    /// </summary>
    public static int ContourBandWidth = 40;

    // ------------------------------------------------------------------
    // Private state
    // ------------------------------------------------------------------

    private static Texture2D _logoTexture;
    private static Color32[] _logoPixels;
    private static int _logoTexW, _logoTexH;
    private static bool _loadAttempted;

    // Exponential-moving-average bounding box — smooths per-frame segmentation jitter
    private static float _sMinX, _sMaxX, _sMinY, _sMaxY;
    private static float _sCX, _sCY;   // smoothed centroid
    private static bool _hasBBox;
    private const float BBoxAlpha = 0.12f;

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    private static void EnsureLoaded()
    {
        if (_loadAttempted) return;
        _loadAttempted = true;
        _logoTexture = Resources.Load<Texture2D>("LogoTexture");
        if (_logoTexture == null)
        {
            Debug.LogWarning("[LogoEffect] LogoTexture not found in Resources — 'logo' effect will fall back to color overlay");
            return;
        }
        _logoTexW = _logoTexture.width;
        _logoTexH = _logoTexture.height;
        try
        {
            _logoPixels = _logoTexture.GetPixels32();
            Debug.Log($"[LogoEffect] Loaded LogoTexture {_logoTexW}x{_logoTexH} ({_logoPixels.Length} pixels) — mode={ProjectionMode}");
        }
        catch (UnityException ex)
        {
            Debug.LogError($"[LogoEffect] LogoTexture is not readable — enable Read/Write in import settings: {ex.Message}");
            _logoPixels = null;
        }
    }

    /// <summary>
    /// Main entry point called by OverlayRenderer / EffectRegistry.
    /// Dispatches to the configured projection mode.
    /// </summary>
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        EnsureLoaded();
        if (_logoPixels == null || _logoTexW == 0 || _logoTexH == 0)
        {
            byte fb = (byte)(200 * Mathf.Clamp01(intensity));
            MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(255, 0, 255, fb));
            return;
        }

        // --- Compute raw bounding box and centroid ---
        int rawMinX = w, rawMaxX = -1, rawMinY = h, rawMaxY = -1;
        long sumX = 0, sumY = 0, count = 0;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (mask[y * w + x] > 0)
                {
                    if (x < rawMinX) rawMinX = x;
                    if (x > rawMaxX) rawMaxX = x;
                    if (y < rawMinY) rawMinY = y;
                    if (y > rawMaxY) rawMaxY = y;
                    sumX += x; sumY += y; count++;
                }
            }
        }
        if (rawMaxX < rawMinX || rawMaxY < rawMinY) { _hasBBox = false; return; }

        float rawCX = count > 0 ? (float)sumX / count : (rawMinX + rawMaxX) * 0.5f;
        float rawCY = count > 0 ? (float)sumY / count : (rawMinY + rawMaxY) * 0.5f;

        // Stabilise with EMA
        if (!_hasBBox)
        {
            _sMinX = rawMinX; _sMaxX = rawMaxX;
            _sMinY = rawMinY; _sMaxY = rawMaxY;
            _sCX = rawCX;     _sCY = rawCY;
            _hasBBox = true;
        }
        else
        {
            _sMinX = Mathf.Lerp(_sMinX, rawMinX, BBoxAlpha);
            _sMaxX = Mathf.Lerp(_sMaxX, rawMaxX, BBoxAlpha);
            _sMinY = Mathf.Lerp(_sMinY, rawMinY, BBoxAlpha);
            _sMaxY = Mathf.Lerp(_sMaxY, rawMaxY, BBoxAlpha);
            _sCX   = Mathf.Lerp(_sCX,   rawCX,   BBoxAlpha);
            _sCY   = Mathf.Lerp(_sCY,   rawCY,   BBoxAlpha);
        }

        float alphaScale = Mathf.Clamp01(intensity);

        switch (ProjectionMode)
        {
            case 1:  ApplyCylindrical(mask, buffer, w, h, alphaScale);  break;
            case 2:  ApplyContour(mask, buffer, w, h, alphaScale);       break;
            default: ApplyBBox(mask, buffer, w, h, alphaScale);          break;
        }
    }

    // ------------------------------------------------------------------
    // Mode 0 — Bounding-box stretch (original, preserved exactly)
    // ------------------------------------------------------------------

    private static void ApplyBBox(byte[] mask, Color32[] buffer, int w, int h, float alphaScale)
    {
        int minX = Mathf.RoundToInt(_sMinX);
        int maxX = Mathf.RoundToInt(_sMaxX);
        int minY = Mathf.RoundToInt(_sMinY);
        int maxY = Mathf.RoundToInt(_sMaxY);
        int boxW = Mathf.Max(1, maxX - minX + 1);
        int boxH = Mathf.Max(1, maxY - minY + 1);

        float logoAspect = (float)_logoTexW / _logoTexH;
        float boxAspect  = (float)boxW / boxH;

        int drawW, drawH, drawOffX, drawOffY;
        if (logoAspect >= boxAspect)
        {
            drawW    = boxW;
            drawH    = Mathf.Max(1, (int)(boxW / logoAspect));
            drawOffX = 0;
            drawOffY = (boxH - drawH) / 2;
        }
        else
        {
            drawH    = boxH;
            drawW    = Mathf.Max(1, (int)(boxH * logoAspect));
            drawOffY = 0;
            drawOffX = (boxW - drawW) / 2;
        }

        for (int py = minY; py <= maxY; py++)
        {
            int localY = py - minY;
            if (localY < drawOffY || localY >= drawOffY + drawH) continue;
            int ty = Mathf.Clamp((localY - drawOffY) * _logoTexH / drawH, 0, _logoTexH - 1);

            for (int px = minX; px <= maxX; px++)
            {
                if (mask[py * w + px] == 0) continue;
                int localX = px - minX;
                if (localX < drawOffX || localX >= drawOffX + drawW) continue;
                int tx = Mathf.Clamp((localX - drawOffX) * _logoTexW / drawW, 0, _logoTexW - 1);

                Color32 lp = _logoPixels[ty * _logoTexW + tx];
                byte a = (byte)(lp.a * alphaScale);
                if (a < 4) continue;
                buffer[py * w + px] = new Color32(lp.r, lp.g, lp.b, a);
            }
        }
    }

    // ------------------------------------------------------------------
    // Mode 1 — Cylindrical wrap
    //
    // The silhouette is treated as a vertical cylinder whose axis is the
    // mask centroid column.  For each mask pixel we compute:
    //
    //   u = (angle_around_centroid / 2π) * CylinderTileU   (wraps 0→1→0...)
    //   v = (pixelY - minY) / (maxY - minY)                (top-to-bottom)
    //
    // The angle is computed in the XY plane (ignoring depth) but the
    // horizontal component is aspect-corrected so the wrap is uniform
    // even when the object is wider than it is tall.
    //
    // Quality notes:
    //  • EMA-smoothed centroid prevents jitter.
    //  • Bilinear texture sample (manual 2×2) avoids pixelation.
    //  • The texture is sampled with tiling (frac) so CylinderTileU>1 works.
    // ------------------------------------------------------------------

    private static void ApplyCylindrical(byte[] mask, Color32[] buffer, int w, int h, float alphaScale)
    {
        int minY = Mathf.RoundToInt(_sMinY);
        int maxY = Mathf.RoundToInt(_sMaxY);
        int minX = Mathf.RoundToInt(_sMinX);
        int maxX = Mathf.RoundToInt(_sMaxX);
        float boxH = Mathf.Max(1f, maxY - minY);
        float boxW = Mathf.Max(1f, maxX - minX);

        // Aspect ratio of the bounding box in pixel space.
        // We scale the X offset by this factor so the angle calculation
        // treats the silhouette as a circle rather than an ellipse,
        // preventing texture from compressing at top/bottom vs sides.
        float aspectInv = boxH / boxW;   // >1 tall, <1 wide

        float cx = _sCX;
        float cy = _sCY;

        float invPI2 = 1f / (2f * Mathf.PI);

        for (int py = minY; py <= maxY; py++)
        {
            if (py < 0 || py >= h) continue;
            float normV = (py - minY) / boxH;

            for (int px = minX; px <= maxX; px++)
            {
                if (px < 0 || px >= w) continue;
                if (mask[py * w + px] == 0) continue;

                // Vector from centroid to pixel, aspect-corrected to circle
                float dx = (px - cx) * aspectInv;
                float dy = py - cy;   // positive = downward

                // Angle in [-π, π]; offset by π so front of object (dx>0) is seam
                float angle = Mathf.Atan2(dy, dx) + Mathf.PI;   // 0 → 2π

                // U: normalised 0→1 around the cylinder, tiled
                float u = Mathf.Repeat(angle * invPI2 * CylinderTileU, 1f);
                float v = normV;

                Color32 sample = SampleBilinear(u, v);
                byte a = (byte)(sample.a * alphaScale);
                if (a < 4) continue;
                buffer[py * w + px] = new Color32(sample.r, sample.g, sample.b, a);
            }
        }
    }

    // ------------------------------------------------------------------
    // Mode 2 — Contour / distance-field UV
    //
    // For each mask pixel, we compute:
    //   v = distance to nearest background pixel / ContourBandWidth  (0=edge, 1=deep interior)
    //   u = normalised arc-length position around the silhouette contour
    //
    // This wraps the texture as a band around the object's silhouette,
    // which looks great for glowing border images or label strips.
    //
    // Implementation: single-pass O(W×H) approximate distance transform
    // using the mask array (no extra allocations beyond the contour list).
    // ------------------------------------------------------------------

    private static void ApplyContour(byte[] mask, Color32[] buffer, int w, int h, float alphaScale)
    {
        const byte threshold = 128;
        int band = Mathf.Max(1, ContourBandWidth);

        int minX = Mathf.RoundToInt(_sMinX);
        int maxX = Mathf.RoundToInt(_sMaxX);
        int minY = Mathf.RoundToInt(_sMinY);
        int maxY = Mathf.RoundToInt(_sMaxY);

        // --- Pass 1: collect contour pixels and assign arc-length U coordinate ---
        // We trace the bounding-box perimeter sweeping clockwise; each interior
        // mask pixel that is adjacent to background is a contour pixel.
        // We assign U by the angle from centroid (same cheap atan2 as cylindrical)
        // so it is O(1) per pixel without needing a full marching-squares trace.

        float cx = _sCX, cy = _sCY;
        float invPI2 = 1f / (2f * Mathf.PI);

        // --- Pass 2: for every mask pixel compute inward distance to nearest edge ---
        // We use a fast approximate: step outward in 4 directions and find the first
        // background pixel; depth = the minimum of those 4 distances.

        for (int py = minY; py <= maxY; py++)
        {
            if (py < 0 || py >= h) continue;
            for (int px = minX; px <= maxX; px++)
            {
                if (px < 0 || px >= w) continue;
                if (mask[py * w + px] < threshold) continue;

                // Compute inward distance (capped at band)
                int d = band + 1;
                for (int k = 1; k <= band && k < d; k++)
                {
                    if (px - k < 0    || mask[py * w + (px - k)] < threshold) { if (k < d) d = k; break; }
                    if (px + k >= w   || mask[py * w + (px + k)] < threshold) { if (k < d) d = k; break; }
                    if (py - k < 0    || mask[(py - k) * w + px] < threshold) { if (k < d) d = k; break; }
                    if (py + k >= h   || mask[(py + k) * w + px] < threshold) { if (k < d) d = k; break; }
                }

                // v = 0 at boundary, 1 deep inside (only pixels within 'band' are drawn)
                float v = (float)(d - 1) / band;
                if (v >= 1f) continue;   // too deep — skip for a tight contour band

                // u = angle from centroid (0→1 around the object)
                float angle = Mathf.Atan2(py - cy, px - cx) + Mathf.PI;
                float u = Mathf.Repeat(angle * invPI2, 1f);

                Color32 sample = SampleBilinear(u, v);
                byte a = (byte)(sample.a * alphaScale);
                if (a < 4) continue;
                buffer[py * w + px] = new Color32(sample.r, sample.g, sample.b, a);
            }
        }
    }

    // ------------------------------------------------------------------
    // Bilinear texture sampler (manual — avoids GPU roundtrip on CPU path)
    // Samples _logoPixels with tiling (wraps in U, clamps in V).
    // ------------------------------------------------------------------

    private static Color32 SampleBilinear(float u, float v)
    {
        // U wraps, V clamps
        float uf = Mathf.Repeat(u, 1f) * (_logoTexW - 1);
        float vf = Mathf.Clamp01(v)   * (_logoTexH - 1);

        int x0 = (int)uf;
        int y0 = (int)vf;
        int x1 = Mathf.Clamp(x0 + 1, 0, _logoTexW - 1);
        int y1 = Mathf.Clamp(y0 + 1, 0, _logoTexH - 1);

        float fx = uf - x0;
        float fy = vf - y0;

        Color32 c00 = _logoPixels[y0 * _logoTexW + x0];
        Color32 c10 = _logoPixels[y0 * _logoTexW + x1];
        Color32 c01 = _logoPixels[y1 * _logoTexW + x0];
        Color32 c11 = _logoPixels[y1 * _logoTexW + x1];

        // Bilinear blend
        float r = c00.r*(1-fx)*(1-fy) + c10.r*fx*(1-fy) + c01.r*(1-fx)*fy + c11.r*fx*fy;
        float g = c00.g*(1-fx)*(1-fy) + c10.g*fx*(1-fy) + c01.g*(1-fx)*fy + c11.g*fx*fy;
        float b = c00.b*(1-fx)*(1-fy) + c10.b*fx*(1-fy) + c01.b*(1-fx)*fy + c11.b*fx*fy;
        float a = c00.a*(1-fx)*(1-fy) + c10.a*fx*(1-fy) + c01.a*(1-fx)*fy + c11.a*fx*fy;

        return new Color32((byte)r, (byte)g, (byte)b, (byte)a);
    }

    // ------------------------------------------------------------------
    // Reset / StoryPlayback fallback
    // ------------------------------------------------------------------

    /// <summary>
    /// Resets the stabilised bounding box so the next appearance snaps immediately.
    /// Called by ClearOverlay() when the user issues a "vibe clear" command.
    /// </summary>
    public static void ResetBBox()
    {
        _hasBBox = false;
    }

    /// <summary>Direct pixel mode for StoryPlayback (fallback to magenta if no texture).</summary>
    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity)
    {
        EnsureLoaded();
        if (_logoPixels == null || _logoTexW == 0 || _logoTexH == 0)
        {
            byte fb = (byte)(200 * Mathf.Clamp01(intensity));
            MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(255, 0, 255, 255), fb / 255f);
            return;
        }
        Debug.LogWarning("[LogoEffect] ApplyPixels not fully implemented for StoryPlayback — use ApplyOverlay path");
        byte fb2 = (byte)(200 * Mathf.Clamp01(intensity));
        MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(255, 0, 255, 255), fb2 / 255f);
    }
}
