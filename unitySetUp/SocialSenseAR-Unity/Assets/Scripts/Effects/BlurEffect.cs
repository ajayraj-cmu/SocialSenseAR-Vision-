using UnityEngine;

/// <summary>
/// Blur effect: GPU Gaussian blur applied where mask is active.
/// AR mode: writes to blur mask buffer (shader does actual passthrough blur).
/// Webcam/Story mode: GPU-blurs the source texture, blends blurred pixels into output.
/// </summary>
public static class BlurEffect
{
    /// <summary>
    /// Overlay mode (AR): write mask into blur buffer for shader-based passthrough blur.
    /// Alpha 140-220: defocus distractions while preserving spatial awareness.
    /// </summary>
    public static void ApplyToBlurBuffer(byte[] mask, Color32[] blurBuffer, int w, int h, float intensity)
    {
        byte blurMaskAlpha = (byte)Mathf.Clamp(140f + 80f * intensity, 0f, 255f);
        MaskApplicator.ApplyToOverlayBuffer(mask, blurBuffer, w, h, new Color32(255, 255, 255, blurMaskAlpha));
    }

    /// <summary>
    /// Overlay mode (webcam fallback): GPU-blur the webcam texture, write blurred pixels
    /// into overlay buffer where mask is active.
    /// </summary>
    public static void ApplyWebcamBlurToOverlay(
        byte[] mask, Color32[] buffer, int w, int h, float intensity,
        Color32[] blurredPixels, int blurW, int blurH)
    {
        if (blurredPixels == null)
        {
            // Fallback: frosted overlay if blur pipeline unavailable
            byte frostAlpha = (byte)Mathf.Clamp(200f + 55f * intensity, 0f, 255f);
            MaskApplicator.ApplyToOverlayBuffer(mask, buffer, w, h, new Color32(220, 225, 235, frostAlpha));
            return;
        }

        int total = w * h;
        for (int i = 0; i < total; i++)
        {
            byte maskAlpha = mask[i];
            if (maskAlpha == 0) continue;

            int mx = i % w;
            int my = i / w;
            int bx = mx * blurW / w;
            int by = my * blurH / h;
            int bi = by * blurW + bx;

            if (bi >= 0 && bi < blurredPixels.Length)
            {
                Color32 blurred = blurredPixels[bi];
                blurred.a = maskAlpha;
                buffer[i] = blurred;
            }
        }
    }

    /// <summary>
    /// Direct pixel mode: blend pre-computed blurred pixels where mask is active.
    /// </summary>
    public static void ApplyPixels(byte[] mask, Color32[] pixels, Color32[] blurredPixels, int count)
    {
        MaskApplicator.BlendPixelsOntoPixels(mask, pixels, blurredPixels, count);
    }
}
