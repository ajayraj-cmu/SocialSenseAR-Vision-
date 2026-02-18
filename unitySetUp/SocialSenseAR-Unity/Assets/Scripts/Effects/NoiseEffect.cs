using System;
using UnityEngine;

/// <summary>
/// Noise/static effect: pseudo-random block pattern for obfuscation.
/// Overlay mode only.
/// </summary>
public static class NoiseEffect
{
    private static readonly Color32[] BlockColors = new Color32[]
    {
        new Color32(40, 40, 50, 230),
        new Color32(80, 80, 90, 230),
        new Color32(60, 60, 70, 230),
        new Color32(100, 100, 110, 230),
    };

    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, float intensity)
    {
        int blockSize = Math.Max(4, (int)(16 * intensity));

        for (int y = 0; y < h; y++)
        {
            int blockY = y / blockSize;
            for (int x = 0; x < w; x++)
            {
                int idx = y * w + x;
                byte maskAlpha = mask[idx];
                if (maskAlpha == 0) continue;

                int blockX = x / blockSize;
                int colorIdx = ((blockX * 7) + (blockY * 13)) % BlockColors.Length;
                Color32 pixel = BlockColors[colorIdx];
                if (maskAlpha < 255)
                    pixel.a = (byte)((pixel.a * maskAlpha + 127) / 255);
                buffer[idx] = pixel;
            }
        }
    }
}
