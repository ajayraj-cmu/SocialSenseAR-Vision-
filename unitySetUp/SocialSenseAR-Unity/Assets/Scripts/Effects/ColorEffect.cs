using UnityEngine;

/// <summary>
/// Color overlay effect: applies a custom color with intensity-controlled opacity.
/// </summary>
public static class ColorEffect
{
    /// <summary>Overlay mode: color with intensity-scaled alpha.</summary>
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity, string colorHex)
    {
        Color32? parsed = EffectConfig.ParseHexColor(colorHex);
        Color32 color = parsed ?? new Color32(200, 200, 200, 255);

        // 8-char hex has explicit alpha — scale by intensity
        if (colorHex != null && colorHex.TrimStart('#').Length == 8)
            color.a = (byte)(color.a * intensity);
        else
            color.a = (byte)(255f * intensity);

        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, color);
    }

    /// <summary>Direct pixel mode: alpha-blend color onto video pixels.</summary>
    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity, string colorHex)
    {
        Color32 color = EffectConfig.ParseHexColorOrDefault(colorHex);
        MaskApplicator.BlendOntoPixels(mask, pixels, count, color, intensity);
    }
}
