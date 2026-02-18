using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Shared effect configuration: asset colors, bright colors, color parsing.
/// Single source of truth for color constants used across the rendering pipeline.
/// </summary>
public static class EffectConfig
{
    private static readonly Dictionary<string, Color32> AssetColors = new Dictionary<string, Color32>
    {
        { "person",     new Color32(70, 130, 255, 255) },
        { "furniture",  new Color32(70, 200, 70, 255) },
        { "lighting",   new Color32(255, 220, 50, 255) },
        { "screen",     new Color32(50, 220, 220, 255) },
        { "structural", new Color32(160, 160, 160, 255) },
        { "object",     new Color32(180, 100, 255, 255) },
    };

    private static readonly Color32[] BrightColors = new Color32[]
    {
        new Color32(255, 255, 0, 255),
        new Color32(255, 0, 255, 255),
        new Color32(0, 255, 0, 255),
        new Color32(0, 255, 255, 255),
        new Color32(0, 128, 255, 255),
        new Color32(255, 0, 128, 255),
        new Color32(255, 128, 0, 255),
        new Color32(128, 0, 255, 255),
        new Color32(255, 255, 255, 255),
        new Color32(200, 255, 100, 255),
    };

    public static readonly Color32 DefaultColor = new Color32(200, 200, 200, 255);
    public static readonly Color32 ClearPixel = new Color32(0, 0, 0, 0);

    public static Color32 GetAssetColor(string assetClass, byte alpha)
    {
        if (string.IsNullOrEmpty(assetClass))
            return new Color32(DefaultColor.r, DefaultColor.g, DefaultColor.b, alpha);

        string key = assetClass.ToLowerInvariant();
        if (key == "screens") key = "screen";
        if (key == "objects") key = "object";

        if (AssetColors.TryGetValue(key, out Color32 c))
            return new Color32(c.r, c.g, c.b, alpha);

        return new Color32(DefaultColor.r, DefaultColor.g, DefaultColor.b, alpha);
    }

    public static Color32 GetBrightColor(int index)
    {
        return BrightColors[index % BrightColors.Length];
    }

    /// <summary>Parse hex color string (#RRGGBB or #RRGGBBAA). Returns null if invalid.</summary>
    public static Color32? ParseHexColor(string hex)
    {
        if (string.IsNullOrEmpty(hex)) return null;
        hex = hex.Trim().TrimStart('#');
        if (hex.Length != 6 && hex.Length != 8) return null;

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

    /// <summary>Parse hex to Color32 with fallback (never returns null).</summary>
    public static Color32 ParseHexColorOrDefault(string hex, Color32 fallback = default)
    {
        return ParseHexColor(hex) ?? (fallback.a == 0 && fallback.r == 0 ? new Color32(255, 255, 255, 255) : fallback);
    }

    /// <summary>Calculate luminance (0-1) using standard RGB weights.</summary>
    public static float CalculateLuminance(Color32 color)
    {
        return (0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b) / 255f;
    }
}
