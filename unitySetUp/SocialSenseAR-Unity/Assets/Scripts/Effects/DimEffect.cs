using UnityEngine;

/// <summary>
/// Dim effect: semi-transparent black overlay for darkening.
/// </summary>
public static class DimEffect
{
    /// <summary>Overlay mode: black with intensity-scaled alpha (60-255).</summary>
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        byte alpha = (byte)Mathf.Clamp(60f + 195f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(0, 0, 0, alpha));
    }

    /// <summary>Direct pixel mode: blend black onto pixels.</summary>
    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count, float intensity)
    {
        MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(0, 0, 0, 255), (200f / 255f) * intensity);
    }
}
