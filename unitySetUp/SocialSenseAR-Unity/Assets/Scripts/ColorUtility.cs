using UnityEngine;
using System;

/// <summary>
/// Utility class for color operations, including hex color parsing.
/// Shared across multiple components to eliminate code duplication.
/// </summary>
public static class ColorUtility
{
    /// <summary>
    /// Parse hex color string (e.g., "#FF0000" or "FF0000") to Color32.
    /// Returns null if the hex string is invalid.
    /// </summary>
    public static Color32? ParseColorHex(string hex)
    {
        if (string.IsNullOrEmpty(hex))
            return null;

        hex = hex.Trim().TrimStart('#');
        if (hex.Length != 6 && hex.Length != 8)
            return null;

        try
        {
            byte r = Convert.ToByte(hex.Substring(0, 2), 16);
            byte g = Convert.ToByte(hex.Substring(2, 2), 16);
            byte b = Convert.ToByte(hex.Substring(4, 2), 16);
            byte a = hex.Length == 8 ? Convert.ToByte(hex.Substring(6, 2), 16) : (byte)255;
            return new Color32(r, g, b, a);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Parse hex color string (e.g., "#FF0000") to Unity Color (float-based).
    /// Returns null if the hex string is invalid.
    /// </summary>
    public static Color? ParseColorHexToColor(string hex)
    {
        Color32? c32 = ParseColorHex(hex);
        if (!c32.HasValue) return null;

        Color32 c = c32.Value;
        return new Color(c.r / 255f, c.g / 255f, c.b / 255f, 1f);
    }

    /// <summary>
    /// Calculate luminance of a color using standard RGB weights.
    /// Used for determining appropriate overlay alpha based on color brightness.
    /// </summary>
    public static float CalculateLuminance(Color32 color)
    {
        return (0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b) / 255f;
    }
}
