using System;
using UnityEngine;

/// <summary>
/// Decodes uint16-LE run-length encoded masks from the server.
/// RLE alternates: background run, foreground run, background, ...
/// Y-flip converts OpenCV row order (0=top) to Unity texture order (0=bottom).
/// </summary>
public static class RleDecoder
{
    /// <summary>
    /// Decode RLE into a byte mask (0 or 255) with Y-flip.
    /// Used by both OverlayRenderer and StoryPlayback.
    /// </summary>
    public static void DecodeToMask(byte[] rleData, byte[] mask, int w, int h)
    {
        Array.Clear(mask, 0, mask.Length);
        int totalPixels = w * h;
        int pixelIndex = 0;
        bool isForeground = false;

        for (int i = 0; i + 1 < rleData.Length && pixelIndex < totalPixels; i += 2)
        {
            int runLength = rleData[i] | (rleData[i + 1] << 8);

            if (isForeground)
            {
                int end = Math.Min(pixelIndex + runLength, totalPixels);
                for (int p = pixelIndex; p < end; p++)
                {
                    int rleRow = p / w;
                    int col = p % w;
                    int unityRow = h - 1 - rleRow;
                    mask[unityRow * w + col] = 255;
                }
            }

            pixelIndex += runLength;
            isForeground = !isForeground;
        }
    }

    /// <summary>
    /// Decode RLE directly into a Color32 buffer with Y-flip (legacy path).
    /// </summary>
    public static void DecodeToBuffer(byte[] rleData, Color32[] buffer, int w, int h, Color32 color)
    {
        int totalPixels = w * h;
        int pixelIndex = 0;
        bool isForeground = false;

        for (int i = 0; i + 1 < rleData.Length && pixelIndex < totalPixels; i += 2)
        {
            int runLength = rleData[i] | (rleData[i + 1] << 8);

            if (isForeground)
            {
                int end = Math.Min(pixelIndex + runLength, totalPixels);
                for (int p = pixelIndex; p < end; p++)
                {
                    int rleRow = p / w;
                    int col = p % w;
                    int unityRow = h - 1 - rleRow;
                    buffer[unityRow * w + col] = color;
                }
            }

            pixelIndex += runLength;
            isForeground = !isForeground;
        }
    }
}
