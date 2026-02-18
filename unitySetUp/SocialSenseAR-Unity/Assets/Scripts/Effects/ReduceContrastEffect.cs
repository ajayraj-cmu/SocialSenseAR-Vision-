using UnityEngine;

/// <summary>
/// Reduce contrast effect: neutral mid-gray semi-transparent overlay.
/// </summary>
public static class ReduceContrastEffect
{
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        byte alpha = (byte)Mathf.Clamp(60f + 100f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(140, 140, 140, alpha));
    }

    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity)
    {
        MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(140, 140, 140, 255), intensity);
    }
}
