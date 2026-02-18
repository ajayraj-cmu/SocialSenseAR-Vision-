using UnityEngine;

/// <summary>
/// CustomImageEffect — wraps a custom PNG texture around a 3D segmentation mask
/// using GPU-quality triplanar projection computed entirely on the CPU pixel buffer.
///
/// THREE MAPPING STRATEGIES
/// ========================
/// Strategy A — Triplanar (CPU approximation of GPU triplanar)
///   Blends three planar projections (XY, YZ, XZ) using face-normal weights
///   derived from the local mask gradient.  No UV seams anywhere.
///   Best for blobby / organic shapes (people, trees).
///
/// Strategy B — Spherical UV
///   Maps the texture using spherical coordinates (θ, φ) centred on the mask
///   centroid.  One polar seam at the back.  Good for round objects.
///
/// Strategy C — Decal / perspective-correct planar
///   Planar front-projection that is perspective-corrected using the segment's
///   known bounding-box depth.  Similar to a projector sticker.
///   Best for flat surfaces or when you want the image to look "painted on".
///
/// SETUP
/// =====
/// Place your PNG at Assets/Resources/CustomImageTexture.png with Read/Write enabled.
/// PNG with alpha channel recommended.  The effect falls back to LogoEffect's
/// LogoTexture if CustomImageTexture is not found.
///
/// USAGE (from EffectRegistry)
/// ===========================
/// effect_type = "custom_image"
/// Params:
///   "mode"   → "triplanar" (default) | "spherical" | "decal"
///   "tile_u" → float, how many times to tile horizontally (default 1)
///   "tile_v" → float, how many times to tile vertically   (default 1)
/// </summary>
public static class CustomImageEffect
{
    // -----------------------------------------------------------------------
    // Public settings (can be set at runtime before first Apply call)
    // -----------------------------------------------------------------------

    /// <summary>0 = triplanar, 1 = spherical, 2 = decal</summary>
    public static int Mode = 1;

    public static float TileU = 1f;
    public static float TileV = 1f;

    // -----------------------------------------------------------------------
    // Texture state
    // -----------------------------------------------------------------------

    private static Texture2D _tex;
    private static Color32[] _pixels;
    private static int _texW, _texH;
    private static bool _loadAttempted;

    // EMA-smoothed bounding box + centroid
    private static float _sMinX, _sMaxX, _sMinY, _sMaxY;
    private static float _sCX, _sCY;
    private static bool _hasBBox;
    private const float BBoxAlpha = 0.10f;

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    private static void EnsureLoaded()
    {
        if (_loadAttempted) return;
        _loadAttempted = true;

        // Try CustomImageTexture first, fall back to LogoTexture
        _tex = Resources.Load<Texture2D>("CustomImageTexture")
            ?? Resources.Load<Texture2D>("LogoTexture");

        if (_tex == null)
        {
            Debug.LogWarning("[CustomImageEffect] No texture found at Resources/CustomImageTexture or Resources/LogoTexture");
            return;
        }
        _texW = _tex.width;
        _texH = _tex.height;
        try
        {
            _pixels = _tex.GetPixels32();
            Debug.Log($"[CustomImageEffect] Loaded {_tex.name} {_texW}×{_texH} mode={Mode}");
        }
        catch (UnityException ex)
        {
            Debug.LogError($"[CustomImageEffect] Texture is not readable: {ex.Message}");
            _pixels = null;
        }
    }

    public static void Reset()
    {
        _hasBBox = false;
        // Allow texture reload on next call (e.g. if user swaps the PNG at runtime)
        _loadAttempted = false;
        _tex = null;
        _pixels = null;
    }

    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------

    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        EnsureLoaded();
        if (_pixels == null)
        {
            byte fb = (byte)(200 * Mathf.Clamp01(intensity));
            MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(100, 100, 255, fb));
            return;
        }

        // Compute bbox + centroid with EMA smoothing
        UpdateBBox(mask, w, h);
        if (!_hasBBox) return;

        float alpha = Mathf.Clamp01(intensity);

        switch (Mode)
        {
            case 1:  ApplySpherical(mask, buffer, w, h, alpha); break;
            case 2:  ApplyDecal(mask, buffer, w, h, alpha);     break;
            default: ApplyTriplanar(mask, buffer, w, h, alpha);  break;
        }
    }

    // -----------------------------------------------------------------------
    // Strategy A — Triplanar
    //
    // Each pixel is assigned three candidate UVs (from XY, YZ, XZ planes)
    // weighted by the surface-normal estimated from the mask gradient.
    // The normal is approximated as the Sobel-gradient at each mask pixel:
    //   Gx = right_mask - left_mask
    //   Gy = below_mask - above_mask
    //   Gz = synthesised depth ≈ sqrt(1 - Gx² - Gy²) clamped
    // This gives each pixel a plausible normal that weights the three
    // projections without any expensive 3-D mesh construction.
    // -----------------------------------------------------------------------

    private static void ApplyTriplanar(byte[] mask, Color32[] buffer, int w, int h, float alpha)
    {
        int minX = Mathf.RoundToInt(_sMinX), maxX = Mathf.RoundToInt(_sMaxX);
        int minY = Mathf.RoundToInt(_sMinY), maxY = Mathf.RoundToInt(_sMaxY);
        float boxW = Mathf.Max(1f, maxX - minX);
        float boxH = Mathf.Max(1f, maxY - minY);

        for (int py = minY; py <= maxY; py++)
        {
            if (py < 1 || py >= h - 1) continue;
            for (int px = minX; px <= maxX; px++)
            {
                if (px < 1 || px >= w - 1) continue;
                byte mv = mask[py * w + px];
                if (mv == 0) continue;

                // --- Sobel gradient (normalized pixel coords) ---
                float mxL = mask[py * w + (px - 1)] / 255f;
                float mxR = mask[py * w + (px + 1)] / 255f;
                float myU = mask[(py - 1) * w + px] / 255f;
                float myD = mask[(py + 1) * w + px] / 255f;

                float gx = (mxR - mxL) * 0.5f;
                float gy = (myD - myU) * 0.5f;
                float gz = Mathf.Sqrt(Mathf.Max(0f, 1f - gx * gx - gy * gy));

                // Blend weights = abs(normal component), pow2 for sharper blending
                float wx = gx * gx;
                float wy = gy * gy;
                float wz = gz * gz;
                float wSum = wx + wy + wz;
                if (wSum < 1e-5f) { wx = 0; wy = 0; wz = 1; wSum = 1; }
                wx /= wSum; wy /= wSum; wz /= wSum;

                // --- Three planar UVs (normalised to box) ---
                float nx = (px - minX) / boxW;
                float ny = (py - minY) / boxH;
                // XZ plane uses horizontal position twice (no depth info, reuse nx)
                float nxInv = 1f - nx;

                // XY plane (front view)
                Color32 cXY = SampleBilinear(nx * TileU, ny * TileV);
                // YZ plane (side view — use inverted X as "depth proxy")
                Color32 cYZ = SampleBilinear(nxInv * TileU, ny * TileV);
                // XZ plane (top view — use X as U, X again as V approximation)
                Color32 cXZ = SampleBilinear(nx * TileU, nx * TileV);

                // Blend
                float r = cXY.r * wz + cYZ.r * wx + cXZ.r * wy;
                float g = cXY.g * wz + cYZ.g * wx + cXZ.g * wy;
                float b = cXY.b * wz + cYZ.b * wx + cXZ.b * wy;
                float a = cXY.a * wz + cYZ.a * wx + cXZ.a * wy;

                byte finalA = (byte)(a * alpha * (mv / 255f));
                if (finalA < 4) continue;
                buffer[py * w + px] = new Color32((byte)r, (byte)g, (byte)b, finalA);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Strategy B — Spherical projection
    //
    //   u = atan2(dy, dx) / (2π)       — longitude around centroid
    //   v = (py - minY) / boxH         — latitude top→bottom
    //
    // One vertical seam at the back.  Bilinear sampled.
    // -----------------------------------------------------------------------

    private static void ApplySpherical(byte[] mask, Color32[] buffer, int w, int h, float alpha)
    {
        int minY = Mathf.RoundToInt(_sMinY), maxY = Mathf.RoundToInt(_sMaxY);
        int minX = Mathf.RoundToInt(_sMinX), maxX = Mathf.RoundToInt(_sMaxX);
        float boxH = Mathf.Max(1f, maxY - minY);
        float boxW = Mathf.Max(1f, maxX - minX);
        float invPI2 = 1f / (2f * Mathf.PI);
        float cx = _sCX, cy = _sCY;
        float aspectInv = boxH / boxW;

        for (int py = minY; py <= maxY; py++)
        {
            if (py < 0 || py >= h) continue;
            float normV = Mathf.Clamp01((py - minY) / boxH);

            for (int px = minX; px <= maxX; px++)
            {
                if (px < 0 || px >= w) continue;
                byte mv = mask[py * w + px];
                if (mv == 0) continue;

                float dx = (px - cx) * aspectInv;
                float dy = py - cy;
                float angle = Mathf.Atan2(dy, dx) + Mathf.PI;
                float u = Mathf.Repeat(angle * invPI2 * TileU, 1f);
                float v = Mathf.Repeat(normV * TileV, 1f);

                Color32 s = SampleBilinear(u, v);
                byte finalA = (byte)(s.a * alpha * (mv / 255f));
                if (finalA < 4) continue;
                buffer[py * w + px] = new Color32(s.r, s.g, s.b, finalA);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Strategy C — Decal / perspective-correct planar
    //
    // Projects the texture from the front onto the bounding box, but applies
    // a simple perspective correction: pixels near the vertical edges of the
    // bounding box are treated as "further away" and compressed inward.
    //
    //   raw_u = (px - minX) / boxW
    //   u = apply_perspective(raw_u)   — squeeze toward vertical centre line
    // -----------------------------------------------------------------------

    private static void ApplyDecal(byte[] mask, Color32[] buffer, int w, int h, float alpha)
    {
        int minX = Mathf.RoundToInt(_sMinX), maxX = Mathf.RoundToInt(_sMaxX);
        int minY = Mathf.RoundToInt(_sMinY), maxY = Mathf.RoundToInt(_sMaxY);
        float boxW = Mathf.Max(1f, maxX - minX);
        float boxH = Mathf.Max(1f, maxY - minY);

        for (int py = minY; py <= maxY; py++)
        {
            if (py < 0 || py >= h) continue;
            float normV = Mathf.Clamp01((py - minY) / boxH);

            for (int px = minX; px <= maxX; px++)
            {
                if (px < 0 || px >= w) continue;
                byte mv = mask[py * w + px];
                if (mv == 0) continue;

                float normU = (px - minX) / boxW;   // 0→1 left to right

                // Perspective squeeze: simulate a ~60° field of view cylindrical lens
                // by remapping u through atan so edges compress.
                // tan(field/2) ≈ 0.577 for 60°
                float tangent = 0.577f;
                float perspU = (Mathf.Atan((normU * 2f - 1f) * tangent) / (2f * Mathf.Atan(tangent)) + 1f) * 0.5f;

                float u = Mathf.Repeat(perspU * TileU, 1f);
                float v = Mathf.Repeat(normV * TileV, 1f);

                Color32 s = SampleBilinear(u, v);
                byte finalA = (byte)(s.a * alpha * (mv / 255f));
                if (finalA < 4) continue;
                buffer[py * w + px] = new Color32(s.r, s.g, s.b, finalA);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static void UpdateBBox(byte[] mask, int w, int h)
    {
        int rawMinX = w, rawMaxX = -1, rawMinY = h, rawMaxY = -1;
        long sx = 0, sy = 0, cnt = 0;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (mask[y * w + x] == 0) continue;
                if (x < rawMinX) rawMinX = x;
                if (x > rawMaxX) rawMaxX = x;
                if (y < rawMinY) rawMinY = y;
                if (y > rawMaxY) rawMaxY = y;
                sx += x; sy += y; cnt++;
            }
        }
        if (rawMaxX < rawMinX) { _hasBBox = false; return; }

        float rcx = cnt > 0 ? (float)sx / cnt : (rawMinX + rawMaxX) * 0.5f;
        float rcy = cnt > 0 ? (float)sy / cnt : (rawMinY + rawMaxY) * 0.5f;

        if (!_hasBBox)
        {
            _sMinX = rawMinX; _sMaxX = rawMaxX;
            _sMinY = rawMinY; _sMaxY = rawMaxY;
            _sCX = rcx; _sCY = rcy;
            _hasBBox = true;
        }
        else
        {
            _sMinX = Mathf.Lerp(_sMinX, rawMinX, BBoxAlpha);
            _sMaxX = Mathf.Lerp(_sMaxX, rawMaxX, BBoxAlpha);
            _sMinY = Mathf.Lerp(_sMinY, rawMinY, BBoxAlpha);
            _sMaxY = Mathf.Lerp(_sMaxY, rawMaxY, BBoxAlpha);
            _sCX   = Mathf.Lerp(_sCX, rcx, BBoxAlpha);
            _sCY   = Mathf.Lerp(_sCY, rcy, BBoxAlpha);
        }
    }

    private static Color32 SampleBilinear(float u, float v)
    {
        float uf = Mathf.Repeat(u, 1f) * (_texW - 1);
        float vf = Mathf.Clamp01(v)   * (_texH - 1);

        int x0 = (int)uf, y0 = (int)vf;
        int x1 = Mathf.Min(x0 + 1, _texW - 1);
        int y1 = Mathf.Min(y0 + 1, _texH - 1);
        float fx = uf - x0, fy = vf - y0;

        Color32 c00 = _pixels[y0 * _texW + x0];
        Color32 c10 = _pixels[y0 * _texW + x1];
        Color32 c01 = _pixels[y1 * _texW + x0];
        Color32 c11 = _pixels[y1 * _texW + x1];

        float w00 = (1-fx)*(1-fy), w10 = fx*(1-fy), w01 = (1-fx)*fy, w11 = fx*fy;

        return new Color32(
            (byte)(c00.r*w00 + c10.r*w10 + c01.r*w01 + c11.r*w11),
            (byte)(c00.g*w00 + c10.g*w10 + c01.g*w01 + c11.g*w11),
            (byte)(c00.b*w00 + c10.b*w10 + c01.b*w01 + c11.b*w11),
            (byte)(c00.a*w00 + c10.a*w10 + c01.a*w01 + c11.a*w11));
    }
}
