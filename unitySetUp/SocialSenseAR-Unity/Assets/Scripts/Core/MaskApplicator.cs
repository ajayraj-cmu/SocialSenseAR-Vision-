using UnityEngine;

/// <summary>
/// Shared mask application helpers used by all effect types.
/// Provides two modes: overlay buffer writes (OverlayRenderer) and direct pixel blending (StoryPlayback).
/// </summary>
public static class MaskApplicator
{
    /// <summary>
    /// Overlay mode: write color into Color32[] buffer where mask > 0.
    /// Alpha is scaled by mask value for smooth edges.
    /// </summary>
    public static void ApplyToOverlayBuffer(byte[] mask, Color32[] buffer, int w, int h, Color32 color)
    {
        int total = w * h;
        for (int i = 0; i < total; i++)
        {
            byte maskAlpha = mask[i];
            if (maskAlpha == 0) continue;
            if (maskAlpha == 255)
            {
                buffer[i] = color;
            }
            else
            {
                buffer[i] = new Color32(color.r, color.g, color.b,
                    (byte)((color.a * maskAlpha + 127) / 255));
            }
        }
    }

    /// <summary>
    /// Direct pixel mode: alpha-blend a flat color onto video pixels where mask > 0.
    /// Combines mask alpha with effect intensity for smooth edge transitions.
    /// </summary>
    public static void BlendOntoPixels(byte[] mask, Color32[] pixels, int count, Color32 color, float intensity)
    {
        for (int i = 0; i < count; i++)
        {
            if (mask[i] == 0) continue;

            float maskAlpha = mask[i] / 255f;
            float a = maskAlpha * Mathf.Clamp01(intensity);
            float ia = 1f - a;

            pixels[i] = new Color32(
                (byte)(pixels[i].r * ia + color.r * a),
                (byte)(pixels[i].g * ia + color.g * a),
                (byte)(pixels[i].b * ia + color.b * a),
                255);
        }
    }

    /// <summary>
    /// Direct pixel mode: alpha-blend source pixels onto target pixels where mask > 0.
    /// Used for blur (blurred frame) and pixelate (block-averaged frame).
    /// </summary>
    public static void BlendPixelsOntoPixels(byte[] mask, Color32[] target, Color32[] source, int count)
    {
        for (int i = 0; i < count; i++)
        {
            int a = mask[i];
            if (a == 0) continue;

            if (a == 255)
            {
                target[i] = source[i];
            }
            else
            {
                int ia = 255 - a;
                target[i] = new Color32(
                    (byte)((source[i].r * a + target[i].r * ia + 127) / 255),
                    (byte)((source[i].g * a + target[i].g * ia + 127) / 255),
                    (byte)((source[i].b * a + target[i].b * ia + 127) / 255),
                    255);
            }
        }
    }
}
