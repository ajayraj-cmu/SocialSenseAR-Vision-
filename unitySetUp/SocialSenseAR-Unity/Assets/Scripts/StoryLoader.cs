using System;
using System.Collections.Generic;
using System.IO;
using Google.Protobuf;
using Google.Protobuf.Collections;
using Socialsense;
using UnityEngine;

/// <summary>
/// Loads a SocialSenseAR story (story.json + masks.bin) and provides
/// per-frame SceneSegment data ready for OverlayRenderer.UpdateOverlay().
/// </summary>
public class StoryLoader
{
    public StoryData Data { get; private set; }

    private byte[] _masksBin;

    // -----------------------------------------------------------
    // JSON data model (matches tools/story_schema.py)
    // -----------------------------------------------------------

    [Serializable]
    public class StoryData
    {
        public int version;
        public StoryMetadata metadata;
        public string[] prompts;
        public Dictionary<string, TrackInfo> tracks;
        public List<StoryFrame> frames;
        public List<VoiceEvent> voice_events;
        public List<EffectEntry> effects_timeline;
    }

    [Serializable]
    public class StoryMetadata
    {
        public string source_video;
        public float fps;
        public int frame_count;
        public int width;
        public int height;
        public int rle_width;
        public int rle_height;
        public float rle_scale;
        public string pipeline_mode;
        public float processing_time_s;
    }

    [Serializable]
    public class TrackInfo
    {
        public string label;
        public string asset_class;
        public int first_frame;
        public int last_frame;
    }

    [Serializable]
    public class StoryFrame
    {
        public int frame_idx;
        public float timestamp_s;
        public bool sam_ran;
        public List<StorySegment> segments;
    }

    [Serializable]
    public class StorySegment
    {
        public string track_id;
        public string label;
        public string asset_class;
        public float confidence;
        public float[] bbox;
        public float center_x;
        public float center_y;
        public int mask_offset;
        public int mask_length;
        public int mask_width;
        public int mask_height;
    }

    [Serializable]
    public class VoiceEvent
    {
        public float timestamp_s;
        public int frame_idx;
        public string transcript;
        public CommandData command;
    }

    [Serializable]
    public class CommandData
    {
        public string[] targets;
        public string effect_type;
        public float intensity;
        public string action;
        public bool invert;
        public string color_hex;
    }

    [Serializable]
    public class EffectEntry
    {
        public int start_frame;
        public int end_frame = -1;  // -1 or missing = until end
        public string target;
        public string effect_type;
        public float intensity = 1f;
        public string color_hex = "";
        public float[] color;  // RGB array [r,g,b] 0-255 (alternative to color_hex)
        public bool invert;
        public string source = "";

        /// <summary>Get color as hex string, converting from RGB array if needed.</summary>
        public string GetColorHex()
        {
            if (!string.IsNullOrEmpty(color_hex))
                return color_hex;
            if (color != null && color.Length >= 3)
                return $"#{(int)color[0]:X2}{(int)color[1]:X2}{(int)color[2]:X2}";
            return "";
        }
    }

    // -----------------------------------------------------------
    // Loading
    // -----------------------------------------------------------

    /// <summary>Load story from a directory containing story.json + masks.bin.</summary>
    public static StoryLoader Load(string storyDir)
    {
        var loader = new StoryLoader();

        string jsonPath = Path.Combine(storyDir, "story.json");
        string masksPath = Path.Combine(storyDir, "masks.bin");

        if (!File.Exists(jsonPath))
            throw new FileNotFoundException($"story.json not found in {storyDir}");

        // Parse JSON
        string jsonText = File.ReadAllText(jsonPath);
        loader.Data = JsonUtility.FromJson<StoryData>(jsonText);

        // Unity's JsonUtility doesn't handle Dictionary or nullable int well.
        // Parse tracks and effects manually from raw JSON.
        loader.ParseRawJson(jsonText);

        // Load masks binary
        if (File.Exists(masksPath))
            loader._masksBin = File.ReadAllBytes(masksPath);
        else
            loader._masksBin = new byte[0];

        Debug.Log($"[StoryLoader] Loaded story: {loader.Data.metadata.frame_count} frames, " +
                  $"{loader._masksBin.Length / 1024}KB masks");

        // Log effects timeline for verification
        if (loader.Data.effects_timeline != null)
        {
            foreach (var fx in loader.Data.effects_timeline)
            {
                string endStr = fx.end_frame > 0 ? fx.end_frame.ToString() : "END";
                string invertStr = fx.invert ? " invert=True" : "";
                string colorStr = !string.IsNullOrEmpty(fx.GetColorHex()) ? $" color={fx.GetColorHex()}" : "";
                Debug.Log($"[StoryLoader] Effect: {fx.effect_type} on '{fx.target}' " +
                          $"frames {fx.start_frame}-{endStr} intensity={fx.intensity}{invertStr}{colorStr}");
            }
        }

        return loader;
    }

    private void ParseRawJson(string jsonText)
    {
        // JsonUtility doesn't support Dictionary<string, T>.
        // Parse tracks manually using a simple approach.
        // For now, tracks are informational only (used for UI, not rendering).
        if (Data.tracks == null)
            Data.tracks = new Dictionary<string, TrackInfo>();

        // Parse effects_timeline — JsonUtility handles lists of serializable objects
        // but end_frame: null in JSON becomes 0 in C#. We handle this:
        // end_frame == 0 and end_frame == -1 both mean "until end of video"
        if (Data.effects_timeline != null)
        {
            foreach (var entry in Data.effects_timeline)
            {
                // JSON null → C# 0 for int fields; treat 0 as "no end"
                if (entry.end_frame <= 0)
                    entry.end_frame = -1;
            }
        }
    }

    // -----------------------------------------------------------
    // Per-frame segment access
    // -----------------------------------------------------------

    /// <summary>
    /// Get SceneSegments for a frame, ready for OverlayRenderer.UpdateOverlay().
    /// </summary>
    public RepeatedField<SceneSegment> GetSegmentsForFrame(int frameIdx)
    {
        var result = new RepeatedField<SceneSegment>();

        if (Data.frames == null || frameIdx < 0 || frameIdx >= Data.frames.Count)
            return result;

        StoryFrame frame = Data.frames[frameIdx];
        if (frame.segments == null)
            return result;

        // Resolve active effects
        var activeEffects = ResolveEffects(frameIdx);

        foreach (var storySeg in frame.segments)
        {
            if (storySeg.mask_length <= 0 || storySeg.mask_width <= 0 || storySeg.mask_height <= 0)
                continue;

            // Only include segments that have an active effect.
            // Without this, segments with no effect get the default blue/green fill
            // color in OverlayRenderer which doesn't match the Python rendering.
            string label = (storySeg.label ?? "").ToLower();
            if (!activeEffects.ContainsKey(label))
                continue;

            var seg = new SceneSegment
            {
                Label = storySeg.label ?? "",
                AssetClass = storySeg.asset_class ?? "",
                Confidence = storySeg.confidence,
                CenterX = storySeg.center_x,
                CenterY = storySeg.center_y,
                TrackId = storySeg.track_id ?? "",
                MaskWidth = (uint)storySeg.mask_width,
                MaskHeight = (uint)storySeg.mask_height,
            };

            // Set bounding box
            if (storySeg.bbox != null && storySeg.bbox.Length >= 4)
            {
                seg.Bbox = new BoundingBox
                {
                    XMin = storySeg.bbox[0],
                    YMin = storySeg.bbox[1],
                    XMax = storySeg.bbox[2],
                    YMax = storySeg.bbox[3],
                };
            }

            // Read RLE mask bytes from masks.bin
            if (storySeg.mask_offset >= 0 && storySeg.mask_length > 0 &&
                storySeg.mask_offset + storySeg.mask_length <= _masksBin.Length)
            {
                byte[] rleBytes = new byte[storySeg.mask_length];
                Buffer.BlockCopy(_masksBin, storySeg.mask_offset, rleBytes, 0, storySeg.mask_length);
                seg.RleMask = ByteString.CopyFrom(rleBytes);
            }

            // Attach effect metadata (we already confirmed this label has an active effect)
            EffectEntry fx = activeEffects[label];
            string fxColorHex = fx.GetColorHex();
            var effectMeta = new EffectMetadata
            {
                EffectType = fx.effect_type ?? "none",
                Intensity = fx.intensity,
                ColorHex = fxColorHex,
            };
            if (fx.invert)
                effectMeta.Params["invert"] = "true";
            seg.Effect = effectMeta;

            result.Add(seg);
        }

        return result;
    }

    // -----------------------------------------------------------
    // Effects timeline resolution
    // -----------------------------------------------------------

    /// <summary>
    /// Resolve which effects are active at a given frame index.
    /// Returns a dictionary of target label → EffectEntry.
    /// </summary>
    public Dictionary<string, EffectEntry> ResolveEffects(int frameIdx)
    {
        var active = new Dictionary<string, EffectEntry>();

        if (Data.effects_timeline == null)
            return active;

        foreach (var entry in Data.effects_timeline)
        {
            if (frameIdx < entry.start_frame)
                continue;
            if (entry.end_frame > 0 && frameIdx > entry.end_frame)
                continue;

            string target = entry.target ?? "";
            if (string.IsNullOrEmpty(target))
                continue;

            active[target] = entry;
        }

        return active;
    }

    /// <summary>Total number of frames in the story.</summary>
    public int FrameCount => Data?.frames?.Count ?? 0;

    /// <summary>Video FPS from metadata.</summary>
    public float Fps => Data?.metadata?.fps ?? 30f;

    /// <summary>Video dimensions from metadata.</summary>
    public Vector2Int VideoDimensions => new Vector2Int(
        Data?.metadata?.width ?? 0,
        Data?.metadata?.height ?? 0
    );
}
