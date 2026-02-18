using System;
using UnityEngine;

/// <summary>
/// Mask processing operations: smoothing, inversion, temporal smoothing.
/// Shared by OverlayRenderer and StoryPlayback.
/// </summary>
public static class MaskProcessor
{
    /// <summary>
    /// Smooth binary mask edges using:
    /// 1. 3x3 binary median filter (majority vote) to remove noisy specks
    /// 2. Double-pass separable box blur (triangle/tent filter approximation)
    /// Optionally applies sharp remap (narrow anti-aliasing band) at the end.
    /// </summary>
    public static void Smooth(byte[] mask, byte[] scratch, int w, int h, int radius, bool sharpRemap = true)
    {
        if (radius <= 0) return;

        int total = w * h;

        // Pre-pass: 3x3 binary median filter
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int count = 0;
                for (int dy = -1; dy <= 1; dy++)
                {
                    int ny = y + dy;
                    if (ny < 0 || ny >= h) continue;
                    int rowOff = ny * w;
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = x + dx;
                        if (nx < 0 || nx >= w) continue;
                        if (mask[rowOff + nx] >= 128) count++;
                    }
                }
                scratch[y * w + x] = count >= 4 ? (byte)255 : (byte)0;
            }
        }
        Array.Copy(scratch, mask, total);

        // Two iterations of separable box blur
        for (int pass = 0; pass < 2; pass++)
        {
            // Horizontal blur (mask → scratch)
            for (int y = 0; y < h; y++)
            {
                int rowStart = y * w;
                int sum = 0;
                int cnt = 0;

                for (int dx = 0; dx <= radius && dx < w; dx++)
                {
                    sum += mask[rowStart + dx];
                    cnt++;
                }

                for (int x = 0; x < w; x++)
                {
                    scratch[rowStart + x] = (byte)(sum / cnt);

                    int removeX = x - radius;
                    int addX = x + radius + 1;
                    if (removeX >= 0) { sum -= mask[rowStart + removeX]; cnt--; }
                    if (addX < w) { sum += mask[rowStart + addX]; cnt++; }
                }
            }

            // Vertical blur (scratch → mask)
            for (int x = 0; x < w; x++)
            {
                int sum = 0;
                int cnt = 0;

                for (int dy = 0; dy <= radius && dy < h; dy++)
                {
                    sum += scratch[dy * w + x];
                    cnt++;
                }

                for (int y = 0; y < h; y++)
                {
                    mask[y * w + x] = (byte)(sum / cnt);

                    int removeY = y - radius;
                    int addY = y + radius + 1;
                    if (removeY >= 0) { sum -= scratch[removeY * w + x]; cnt--; }
                    if (addY < h) { sum += scratch[addY * w + x]; cnt++; }
                }
            }
        }

        // Sharp remap: narrow anti-aliasing band (64→0, 160→255, linear ramp between)
        if (sharpRemap)
        {
            for (int i = 0; i < total; i++)
            {
                int v = mask[i];
                if (v <= 64) mask[i] = 0;
                else if (v >= 160) mask[i] = 255;
                else mask[i] = (byte)((v - 64) * 255 / 96);
            }
        }
    }

    /// <summary>Invert a byte mask (0↔255).</summary>
    public static void Invert(byte[] mask, int count)
    {
        for (int i = 0; i < count; i++)
            mask[i] = (byte)(255 - mask[i]);
    }

    /// <summary>
    /// Temporal alpha smoothing: 70% current + 30% previous frame.
    /// Reduces edge jitter for live rendering.
    /// </summary>
    public static void TemporalSmooth(Color32[] buffer, ref byte[] prevAlpha)
    {
        if (prevAlpha != null && prevAlpha.Length == buffer.Length)
        {
            for (int i = 0; i < buffer.Length; i++)
            {
                int prev = prevAlpha[i];
                int curr = buffer[i].a;
                int blended = (curr * 179 + prev * 76 + 127) / 255;
                buffer[i] = new Color32(buffer[i].r, buffer[i].g, buffer[i].b, (byte)blended);
                prevAlpha[i] = (byte)blended;
            }
        }
        else
        {
            prevAlpha = new byte[buffer.Length];
            for (int i = 0; i < buffer.Length; i++)
                prevAlpha[i] = buffer[i].a;
        }
    }
}
