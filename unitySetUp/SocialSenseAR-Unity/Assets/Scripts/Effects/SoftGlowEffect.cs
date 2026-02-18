using UnityEngine;

/// <summary>
/// Soft glow effect: warm yellowish semi-transparent overlay.
/// </summary>
public static class SoftGlowEffect
{
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        byte alpha = (byte)Mathf.Clamp(60f + 120f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(255, 245, 200, alpha));
    }

    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity)
    {
        MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(255, 245, 200, 255), intensity);
    }
}
