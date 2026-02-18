using UnityEngine;

/// <summary>
/// Grayscale effect: neutral gray overlay to reduce color.
/// </summary>
public static class GrayscaleEffect
{
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        byte alpha = (byte)Mathf.Clamp(180f + 75f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(128, 128, 128, alpha));
    }

    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity)
    {
        MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(128, 128, 128, 255), intensity);
    }
}
