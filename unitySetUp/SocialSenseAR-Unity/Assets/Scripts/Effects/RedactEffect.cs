using UnityEngine;

/// <summary>
/// Redact effect: solid black censor bar over the masked area.
/// </summary>
public static class RedactEffect
{
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h)
    {
        MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(0, 0, 0, 255));
    }

    public static void ApplyPixels(byte[] mask, Color32[] pixels, int count)
    {
        MaskApplicator.BlendOntoPixels(mask, pixels, count, new Color32(0, 0, 0, 255), 1f);
    }
}
