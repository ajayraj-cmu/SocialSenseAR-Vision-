using UnityEngine;

/// <summary>
/// Cool tint effect: blue/cool color cast overlay.
/// </summary>
public static class CoolTintEffect
{
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        byte alpha = (byte)Mathf.Clamp(80f + 140f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(150, 200, 255, alpha));
    }

    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity)
    {
        MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(150, 200, 255, 255), intensity);
    }
}
