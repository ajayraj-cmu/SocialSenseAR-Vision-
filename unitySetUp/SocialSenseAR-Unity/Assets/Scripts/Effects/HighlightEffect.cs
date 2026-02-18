using UnityEngine;

/// <summary>
/// Highlight effect: warm yellow semi-transparent overlay.
/// </summary>
public static class HighlightEffect
{
    private static readonly Color32 HighlightColor = new Color32(255, 255, 180, 255);

    /// <summary>Overlay mode: warm yellow with intensity-scaled alpha (80-255).</summary>
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        byte alpha = (byte)Mathf.Clamp(80f + 175f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(255, 255, 180, alpha));
    }

    /// <summary>Direct pixel mode: blend warm yellow onto pixels.</summary>
    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity)
    {
        MaskApplicator.BlendOntoPixels(mask, pixels, count, HighlightColor, (76f / 255f) * intensity);
    }
}
