using System.IO;
using Google.Protobuf.Collections;
using Socialsense;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using UnityEngine.Video;

/// <summary>
/// Plays a SocialSenseAR story: video + effects rendered using shared Core/ + Effects/.
/// Matches the Python story_renderer.py: GPU blur, mask decode, per-pixel effect application.
///
/// Press R to toggle frame capture. When video ends, ffmpeg encodes to MP4.
/// </summary>
public class StoryPlayback : MonoBehaviour
{
    [Header("Story")]
    public string storyPath = "";

    [Header("Video")]
    public VideoClip videoClip;
    public string videoFilePath = "";

    [Header("UI")]
    public RawImage videoDisplay;

    [Header("Capture")]
    public string outputPath = "demo_frames";
    [Range(1, 100)] public int jpegQuality = 95;
    public bool autoCapture = true;

    [Header("Mask Smoothing")]
    [Range(0, 3)]
    public int maskSmoothRadius = 2;

    [HideInInspector] public OverlayRenderer overlayRenderer; // backward compat

    private StoryLoader _story;
    private VideoPlayer _videoPlayer;
    private RenderTexture _videoRT;
    private int _currentFrame = -1;
    private int _totalFrames;
    private int _videoW, _videoH;

    // Rendering (shared pipelines)
    private GaussianBlurPipeline _blurPipeline;
    private Texture2D _compositeTex;
    private Texture2D _videoReadback;
    private Texture2D _blurReadback;
    private byte[] _maskScratch;

    // Capture
    private VideoExporter _exporter;

    void Start()
    {
        if (string.IsNullOrEmpty(storyPath))
        {
            Debug.LogError("[StoryPlayback] storyPath not set!");
            enabled = false;
            return;
        }

        _story = StoryLoader.Load(storyPath);
        _totalFrames = _story.FrameCount;

        Debug.Log($"[StoryPlayback] {_totalFrames} frames, " +
                  $"{_story.Data.metadata.rle_width}x{_story.Data.metadata.rle_height} masks, " +
                  $"{_story.Data.effects_timeline?.Count ?? 0} effects");

        _exporter = new VideoExporter(outputPath, jpegQuality);
        SetupVideo();
    }

    void SetupVideo()
    {
        _videoPlayer = gameObject.AddComponent<VideoPlayer>();
        _videoPlayer.playOnAwake = false;
        _videoPlayer.isLooping = false;
        _videoPlayer.skipOnDrop = false;
        _videoPlayer.audioOutputMode = VideoAudioOutputMode.Direct;
        _videoPlayer.renderMode = VideoRenderMode.RenderTexture;

        if (videoClip != null)
        {
            _videoPlayer.source = VideoSource.VideoClip;
            _videoPlayer.clip = videoClip;
        }
        else if (!string.IsNullOrEmpty(videoFilePath))
        {
            _videoPlayer.source = VideoSource.Url;
            _videoPlayer.url = videoFilePath;
        }
        else
        {
            string name = _story.Data?.metadata?.source_video ?? "";
            string path = Path.Combine(storyPath, name);
            if (File.Exists(path))
            {
                _videoPlayer.source = VideoSource.Url;
                _videoPlayer.url = path;
            }
            else
            {
                Debug.LogError($"[StoryPlayback] No video found: {path}");
                return;
            }
        }

        _videoW = _story.Data.metadata.width;
        _videoH = _story.Data.metadata.height;
        _videoRT = new RenderTexture(_videoW, _videoH, 0, RenderTextureFormat.ARGB32);
        _videoRT.Create();
        _videoPlayer.targetTexture = _videoRT;

        // Shared GPU blur pipeline
        _blurPipeline = new GaussianBlurPipeline();
        _blurPipeline.Init(_videoW, _videoH);
        _blurReadback = new Texture2D(_videoW, _videoH, TextureFormat.RGBA32, false);

        _videoReadback = new Texture2D(_videoW, _videoH, TextureFormat.RGBA32, false);
        _compositeTex = new Texture2D(_videoW, _videoH, TextureFormat.RGBA32, false);
        _maskScratch = new byte[_videoW * _videoH];

        if (videoDisplay != null)
        {
            videoDisplay.texture = _videoRT;
            videoDisplay.color = Color.white;
        }

        _videoPlayer.loopPointReached += vp => { if (_exporter.IsCapturing) StopAndExport(); };
        _videoPlayer.prepareCompleted += vp =>
        {
            Debug.Log($"[StoryPlayback] Video ready: {vp.width}x{vp.height}. Playing.");
            vp.Play();
            if (autoCapture) _exporter.StartCapture();
        };
        _videoPlayer.Prepare();
    }

    void Update()
    {
        if (_videoPlayer == null || !_videoPlayer.isPlaying) return;

        int frame = (int)_videoPlayer.frame;
        if (frame == _currentFrame || frame < 0 || frame >= _totalFrames) return;
        _currentFrame = frame;

        ProcessFrame(frame);

        if (Keyboard.current != null && Keyboard.current.rKey.wasPressedThisFrame)
        {
            if (_exporter.IsCapturing) StopAndExport();
            else _exporter.StartCapture();
        }
    }

    void ProcessFrame(int frameIdx)
    {
        Color32[] pixels = GaussianBlurPipeline.ReadRT(_videoRT, _videoReadback);
        RepeatedField<SceneSegment> segments = _story.GetSegmentsForFrame(frameIdx);

        if (segments.Count == 0)
        {
            UpdateDisplay(pixels);
            return;
        }

        // Pre-compute blurred frame if any blur effect is active
        Color32[] blurredPixels = null;
        float maxBlurIntensity = 0f;
        foreach (var seg in segments)
        {
            if (seg.Effect != null && seg.Effect.EffectType == "blur")
                maxBlurIntensity = Mathf.Max(maxBlurIntensity, seg.Effect.Intensity);
        }
        if (maxBlurIntensity > 0f && _blurPipeline.IsValid)
        {
            RenderTexture blurredRT = _blurPipeline.Blur(_videoRT, maxBlurIntensity);
            blurredPixels = GaussianBlurPipeline.ReadRT(blurredRT, _blurReadback);
        }

        // Apply each effect via shared Effects/ pipeline
        foreach (var seg in segments)
        {
            if (seg.RleMask == null || seg.RleMask.Length == 0) continue;
            if (seg.Effect == null || seg.Effect.EffectType == "none") continue;

            int maskW = (int)seg.MaskWidth;
            int maskH = (int)seg.MaskHeight;
            if (maskW != _videoW || maskH != _videoH) continue;

            // Decode + smooth + invert using shared Core/
            byte[] mask = new byte[maskW * maskH];
            RleDecoder.DecodeToMask(seg.RleMask.ToByteArray(), mask, maskW, maskH);
            MaskProcessor.Smooth(mask, _maskScratch, maskW, maskH, maskSmoothRadius, sharpRemap: false);

            if (seg.Effect.Invert)
                MaskProcessor.Invert(mask, mask.Length);

            // Dispatch to shared Effects/
            EffectRegistry.ApplyPixels(
                seg.Effect.EffectType, mask, pixels, _videoW, _videoH,
                seg.Effect.Intensity, seg.Effect.ColorHex,
                blurredPixels: blurredPixels);
        }

        UpdateDisplay(pixels);
    }

    void UpdateDisplay(Color32[] pixels)
    {
        _compositeTex.SetPixels32(pixels);
        _compositeTex.Apply(false);

        if (videoDisplay != null)
            videoDisplay.texture = _compositeTex;

        _exporter.SaveFrame(_compositeTex);
    }

    void StopAndExport()
    {
        _exporter.StopCapture();
        string sourceVideo = videoFilePath;
        if (string.IsNullOrEmpty(sourceVideo))
        {
            string name = _story.Data?.metadata?.source_video ?? "";
            sourceVideo = Path.Combine(storyPath, name);
        }
        _exporter.ExportAsync(_story.Fps, sourceVideo);
    }

    void OnDestroy()
    {
        if (_videoRT != null) { _videoRT.Release(); Destroy(_videoRT); }
        _blurPipeline?.Dispose();
        if (_compositeTex != null) Destroy(_compositeTex);
        if (_videoReadback != null) Destroy(_videoReadback);
        if (_blurReadback != null) Destroy(_blurReadback);
    }

    void OnGUI()
    {
        string status = _exporter.IsCapturing ? "<color=red>● REC</color>" : "R=capture";
        if (!string.IsNullOrEmpty(_exporter.ExportStatus))
            status = $"<color=yellow>EXPORT: {_exporter.ExportStatus}</color>";

        float fps = 1f / Time.deltaTime;
        GUI.Label(new Rect(10, 10, 800, 30),
            $"<size=18>Frame {_currentFrame}/{_totalFrames} | {fps:F0} fps | {status} | captured={_exporter.CapturedCount}</size>");
    }
}
