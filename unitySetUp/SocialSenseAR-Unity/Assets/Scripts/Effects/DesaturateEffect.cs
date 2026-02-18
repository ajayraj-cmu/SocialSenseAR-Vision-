using UnityEngine;

/// <summary>
/// Desaturate effect: muted gray overlay to reduce color saturation.
/// </summary>
public static class DesaturateEffect
{
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        byte alpha = (byte)Mathf.Clamp(100f + 120f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(160, 160, 160, alpha));
    }

    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity)
    {
        MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(160, 160, 160, 255), intensity);
    }
}
