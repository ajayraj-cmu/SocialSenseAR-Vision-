using UnityEngine;

/// <summary>
/// Frosted glass: semi-transparent white overlay + blur.
/// Overlay mode only.
/// </summary>
public static class FrostedGlassEffect
{
    /// <summary>Write frosted overlay + blur mask for the actual frosted glass blur.</summary>
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, Color32[] blurBuffer, int w, int h, float intensity)
    {
        byte frostAlpha = (byte)Mathf.Clamp(140f + 80f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(230, 235, 245, frostAlpha));

        byte frostBlurAlpha = (byte)Mathf.Clamp(200f + 55f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, blurBuffer, w, h, new Color32(255, 255, 255, frostBlurAlpha));
    }
}
