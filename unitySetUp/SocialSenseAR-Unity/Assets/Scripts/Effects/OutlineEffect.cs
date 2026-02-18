using System;
using UnityEngine;

/// <summary>
/// Outline effect: draw colored outline at mask boundary.
/// Supports hard edges and soft gradient edges.
/// Overlay mode only — not used in story playback.
/// </summary>
public static class OutlineEffect
{
    /// <summary>Draw outline from mask into overlay buffer.</summary>
    public static void ApplyOverlay(byte[] mask, Color32[] buffer, int w, int h, Color32 color, int thickness, int softWidth)
    {
        const byte maskThreshold = 128;

        if (softWidth > 0)
        {
            DrawSoftOutline(mask, buffer, w, h, color, thickness, softWidth, maskThreshold);
        }
        else
        {
            DrawHardOutline(mask, buffer, w, h, color, thickness, maskThreshold);
        }
    }

    private static void DrawSoftOutline(byte[] mask, Color32[] buffer, int w, int h, Color32 color, int thickness, int softWidth, byte threshold)
    {
        int searchRadius = thickness + softWidth;

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int idx = y * w + x;
                if (mask[idx] < threshold) continue;

                int minDist = int.MaxValue;
                for (int dy = -searchRadius; dy <= searchRadius; dy++)
                {
                    for (int dx = -searchRadius; dx <= searchRadius; dx++)
                    {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h || mask[ny * w + nx] < threshold)
                        {
                            int dist = Math.Max(Math.Abs(dx), Math.Abs(dy));
                            if (dist < minDist) minDist = dist;
                        }
                    }
                }

                if (minDist <= thickness)
                {
                    buffer[idx] = color;
                }
                else if (minDist <= thickness + softWidth)
                {
                    float t = (float)(minDist - thickness) / softWidth;
                    byte alpha = (byte)(color.a * (1f - t));
                    buffer[idx] = new Color32(color.r, color.g, color.b, alpha);
                }
            }
        }
    }

    private static void DrawHardOutline(byte[] mask, Color32[] buffer, int w, int h, Color32 color, int thickness, byte threshold)
    {
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int idx = y * w + x;
                if (mask[idx] < threshold) continue;

                bool isEdge = false;
                for (int d = 1; d <= thickness && !isEdge; d++)
                {
                    if (x - d < 0 || x + d >= w || y - d < 0 || y + d >= h)
                    { isEdge = true; break; }
                    if (mask[(y - d) * w + x] < threshold || mask[(y + d) * w + x] < threshold ||
                        mask[y * w + (x - d)] < threshold || mask[y * w + (x + d)] < threshold)
                    { isEdge = true; }
                }

                if (isEdge) buffer[idx] = color;
            }
        }
    }
}
