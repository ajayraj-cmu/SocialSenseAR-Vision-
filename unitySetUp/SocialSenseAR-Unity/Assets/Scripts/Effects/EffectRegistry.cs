using UnityEngine;

/// <summary>
/// Central dispatch for all effect types.
/// Maps effect name string to the correct static effect class.
/// Replaces the giant switch statements in OverlayRenderer and StoryPlayback.
/// </summary>
public static class EffectRegistry
{
    /// <summary>
    /// Overlay mode: apply effect into overlay buffer (for OverlayRenderer).
    /// Some effects also write to blurBuffer or pixelateBuffer.
    /// </summary>
    public static void ApplyOverlay(
        string effectType, byte[] mask, Color32[] buffer, int w, int h,
        float intensity, Color32 segColor, string colorHex = null,
        Color32[] blurBuffer = null, Color32[] pixelateBuffer = null,
        // Webcam blur/pixelate fallback params
        Color32[] blurredPixels = null, int blurW = 0, int blurH = 0,
        Color32[] webcamPixels = null, int webcamW = 0, int webcamH = 0,
        bool useOverlayBlurFallback = false)
    {
        switch (effectType)
        {
            case "blur":
                if (useOverlayBlurFallback)
                    BlurEffect.ApplyWebcamBlurToOverlay(mask, buffer, w, h, intensity, blurredPixels, blurW, blurH);
                else if (blurBuffer != null)
                    BlurEffect.ApplyToBlurBuffer(mask, blurBuffer, w, h, intensity);
                break;
            case "color":
                ColorEffect.ApplyOverlay(mask, buffer, w, h, intensity, colorHex);
                break;
            case "dim":
                DimEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "highlight":
                HighlightEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "pixelate":
                if (useOverlayBlurFallback && webcamPixels != null)
                    PixelateEffect.ApplyWebcamToOverlay(mask, buffer, w, h, webcamPixels, webcamW, webcamH, intensity);
                else if (pixelateBuffer != null)
                    PixelateEffect.ApplyToPixelateBuffer(mask, pixelateBuffer, w, h);
                break;
            case "outline":
                segColor.a = 255;
                int thickness = System.Math.Max(3, (int)(8 * intensity));
                OutlineEffect.ApplyOverlay(mask, buffer, w, h, segColor, thickness, 2);
                break;
            case "frosted_glass":
                FrostedGlassEffect.ApplyOverlay(mask, buffer, blurBuffer, w, h, intensity);
                break;
            case "redact":
                RedactEffect.ApplyOverlay(mask, buffer, w, h);
                break;
            case "grayscale":
                GrayscaleEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "desaturate":
                DesaturateEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "noise":
                NoiseEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "soft_glow":
                SoftGlowEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "warm_tint":
                WarmTintEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "cool_tint":
                CoolTintEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "reduce_contrast":
                ReduceContrastEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
            case "logo":
                LogoEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;

            case "custom_image":
                // Parse optional mode / tile params from effect params dict (passed via colorHex convention)
                // Params encoding: "mode:triplanar|tile_u:2.0|tile_v:1.5" in the colorHex field for now.
                // Full EffectParams dict support can be added when protobuf is extended.
                if (colorHex != null)
                {
                    foreach (var part in colorHex.Split('|'))
                    {
                        var kv = part.Split(':');
                        if (kv.Length != 2) continue;
                        switch (kv[0].Trim())
                        {
                            case "mode":
                                CustomImageEffect.Mode = kv[1].Trim() switch
                                {
                                    "spherical" => 1,
                                    "decal"     => 2,
                                    _           => 0,  // triplanar default
                                };
                                break;
                            case "tile_u":
                                if (float.TryParse(kv[1], out float tu)) CustomImageEffect.TileU = tu;
                                break;
                            case "tile_v":
                                if (float.TryParse(kv[1], out float tv)) CustomImageEffect.TileV = tv;
                                break;
                        }
                    }
                }
                CustomImageEffect.ApplyOverlay(mask, buffer, w, h, intensity);
                break;
        }
    }

    /// <summary>
    /// Direct pixel mode: apply effect directly onto video pixels (for StoryPlayback).
    /// Blur and pixelate require pre-computed data passed as parameters.
    /// </summary>
    public static void ApplyPixels(
        string effectType, byte[] mask, Color32[] pixels, int w, int h,
        float intensity, string colorHex = null,
        Color32[] blurredPixels = null)
    {
        int count = w * h;
        switch (effectType)
        {
            case "blur":
                if (blurredPixels != null)
                    BlurEffect.ApplyPixels(mask, pixels, blurredPixels, count);
                break;
            case "color":
                ColorEffect.ApplyPixels(mask, pixels, count, intensity, colorHex);
                break;
            case "dim":
                DimEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "highlight":
                HighlightEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "pixelate":
                PixelateEffect.ApplyPixels(mask, pixels, w, h, intensity);
                break;
            case "redact":
                RedactEffect.ApplyPixels(mask, pixels, count);
                break;
            case "grayscale":
                GrayscaleEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "desaturate":
                DesaturateEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "soft_glow":
                SoftGlowEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "warm_tint":
                WarmTintEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "cool_tint":
                CoolTintEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "reduce_contrast":
                ReduceContrastEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "logo":
                LogoEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
            case "custom_image":
                // Use logo fallback path for StoryPlayback (pixel mode)
                LogoEffect.ApplyPixels(mask, pixels, count, intensity);
                break;
        }
    }

    /// <summary>
    /// Get the visual color for an effect type (used for edge sweep animation + out-of-bounds fill).
    /// </summary>
    public static Color32 GetEffectColor(string effectType, float intensity, Color32 segColor, string colorHex = null)
    {
        switch (effectType)
        {
            case "blur":
                return new Color32(128, 128, 130, (byte)(180 * intensity));
            case "color":
                Color32? parsed = EffectConfig.ParseHexColor(colorHex);
                Color32 c = parsed ?? segColor;
                c.a = (byte)(255f * intensity);
                return c;
            case "dim":
                return new Color32(0, 0, 0, (byte)(200 * intensity));
            case "highlight":
                return new Color32(255, 255, 180, (byte)(76 * intensity));
            case "pixelate":
                return new Color32(70, 70, 80, 230);
            case "outline":
                segColor.a = 255;
                return segColor;
            case "logo":
                return new Color32(255, 255, 255, (byte)(200 * intensity));
            case "custom_image":
                return new Color32(255, 255, 255, (byte)(200 * intensity));
            default:
                return new Color32(0, 0, 0, 0);
        }
    }
}
