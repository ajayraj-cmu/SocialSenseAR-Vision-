using System;
using UnityEngine;

/// <summary>
/// Pixelate effect: block-average pixelation within the masked region.
/// </summary>
public static class PixelateEffect
{
    /// <summary>
    /// Overlay mode (webcam): read webcam pixels, snap to block grid center,
    /// write pixelated result into overlay buffer.
    /// </summary>
    public static void ApplyWebcamToOverlay(
        byte[] mask, Color32[] buffer, int w, int h,
        Color32[] webcamPixels, int srcW, int srcH, float intensity)
    {
        int blockSize = Math.Max(4, (int)(20 * intensity));

        int total = w * h;
        for (int i = 0; i < total; i++)
        {
            byte maskAlpha = mask[i];
            if (maskAlpha == 0) continue;

            int mx = i % w;
            int my = i / w;
            int wx = mx * srcW / w;
            int wy = my * srcH / h;
            wx = (wx / blockSize) * blockSize + blockSize / 2;
            wy = (wy / blockSize) * blockSize + blockSize / 2;
            wx = Math.Min(wx, srcW - 1);
            wy = Math.Min(wy, srcH - 1);

            int wi = wy * srcW + wx;
            if (wi >= 0 && wi < webcamPixels.Length)
            {
                Color32 pixelated = webcamPixels[wi];
                pixelated.a = maskAlpha;
                buffer[i] = pixelated;
            }
        }
    }

    /// <summary>
    /// Overlay mode (AR): write mask into pixelate buffer for shader-based pixelation.
    /// </summary>
    public static void ApplyToPixelateBuffer(byte[] mask, Color32[] pixelateBuffer, int w, int h)
    {
        MaskApplicator.ApplyToOverlayBuffer(mask, pixelateBuffer, w, h, new Color32(255, 255, 255, 255));
    }

    /// <summary>
    /// Direct pixel mode: compute block averages and replace masked pixels.
    /// </summary>
    public static void ApplyPixels(byte[] mask, Color32[] pixels, int videoW, int videoH, float intensity)
    {
        int block = Math.Max(2, (int)(32 * intensity));
        int bw = (videoW + block - 1) / block;
        int bh = (videoH + block - 1) / block;

        int[] blockR = new int[bw * bh];
        int[] blockG = new int[bw * bh];
        int[] blockB = new int[bw * bh];
        int[] blockCounts = new int[bw * bh];

        for (int y = 0; y < videoH; y++)
        {
            int by = y / block;
            for (int x = 0; x < videoW; x++)
            {
                int bx = x / block;
                int bi = by * bw + bx;
                int pi = y * videoW + x;
                blockR[bi] += pixels[pi].r;
                blockG[bi] += pixels[pi].g;
                blockB[bi] += pixels[pi].b;
                blockCounts[bi]++;
            }
        }

        for (int y = 0; y < videoH; y++)
        {
            int by = y / block;
            for (int x = 0; x < videoW; x++)
            {
                int pi = y * videoW + x;
                int a = mask[pi];
                if (a == 0) continue;

                int bx = x / block;
                int bi = by * bw + bx;
                int c = blockCounts[bi];
                byte pr = (byte)(blockR[bi] / c);
                byte pg = (byte)(blockG[bi] / c);
                byte pb = (byte)(blockB[bi] / c);

                if (a == 255)
                {
                    pixels[pi] = new Color32(pr, pg, pb, 255);
                }
                else
                {
                    int ia = 255 - a;
                    pixels[pi] = new Color32(
                        (byte)((pr * a + pixels[pi].r * ia + 127) / 255),
                        (byte)((pg * a + pixels[pi].g * ia + 127) / 255),
                        (byte)((pb * a + pixels[pi].b * ia + 127) / 255),
                        255);
                }
            }
        }
    }
}
