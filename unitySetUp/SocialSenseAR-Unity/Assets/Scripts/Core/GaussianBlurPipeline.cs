using System;
using UnityEngine;

/// <summary>
/// GPU two-pass separable Gaussian blur pipeline.
/// Manages shader materials and intermediate RenderTextures.
/// Shared by OverlayRenderer (webcam blur fallback) and StoryPlayback.
/// </summary>
public class GaussianBlurPipeline : IDisposable
{
    private Material _blurHMat;
    private Material _blurVMat;
    private RenderTexture _tempRT;
    private RenderTexture _resultRT;
    private int _width;
    private int _height;
    private bool _initialized;

    private static readonly int BlurSizeId = Shader.PropertyToID("_BlurSize");

    /// <summary>Whether the pipeline initialized successfully (shaders found).</summary>
    public bool IsValid => _initialized;

    /// <summary>
    /// Initialize the blur pipeline. Tries Resources first, then Shader.Find.
    /// </summary>
    public void Init(int width = 0, int height = 0)
    {
        _width = width;
        _height = height;

        Shader hShader = Resources.Load<Shader>("GaussianBlurHorizontal");
        if (hShader == null)
            hShader = Shader.Find("Hidden/SocialSense/GaussianBlurHorizontal");

        Shader vShader = Resources.Load<Shader>("GaussianBlurVertical");
        if (vShader == null)
            vShader = Shader.Find("Hidden/SocialSense/GaussianBlurVertical");

        if (hShader == null || vShader == null)
        {
            Debug.LogWarning("[GaussianBlurPipeline] Blur shaders not found — blur effects will be skipped.");
            return;
        }

        _blurHMat = new Material(hShader);
        _blurVMat = new Material(vShader);

        if (width > 0 && height > 0)
            EnsureRTs(width, height);

        _initialized = true;
    }

    private void EnsureRTs(int w, int h)
    {
        if (_tempRT != null && _tempRT.width == w && _tempRT.height == h)
            return;

        if (_tempRT != null) { _tempRT.Release(); UnityEngine.Object.Destroy(_tempRT); }
        if (_resultRT != null) { _resultRT.Release(); UnityEngine.Object.Destroy(_resultRT); }

        _tempRT = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
        _tempRT.Create();
        _resultRT = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
        _resultRT.Create();
        _width = w;
        _height = h;
    }

    /// <summary>
    /// Perform two-pass Gaussian blur on source texture.
    /// Sigma = 10 + intensity * 40 (same formula everywhere).
    /// Returns the blurred RenderTexture (owned by this pipeline, do not destroy).
    /// </summary>
    public RenderTexture Blur(Texture source, float intensity)
    {
        if (!_initialized || source == null) return null;

        EnsureRTs(source.width, source.height);

        float sigma = 10f + intensity * 40f;
        _blurHMat.SetFloat(BlurSizeId, sigma);
        _blurVMat.SetFloat(BlurSizeId, sigma);

        Graphics.Blit(source, _tempRT, _blurHMat);
        Graphics.Blit(_tempRT, _resultRT, _blurVMat);

        return _resultRT;
    }

    /// <summary>Read a RenderTexture back to CPU as Color32 array.</summary>
    public static Color32[] ReadRT(RenderTexture rt, Texture2D readback)
    {
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;
        readback.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
        readback.Apply(false);
        RenderTexture.active = prev;
        return readback.GetPixels32();
    }

    public void Dispose()
    {
        if (_tempRT != null) { _tempRT.Release(); UnityEngine.Object.Destroy(_tempRT); }
        if (_resultRT != null) { _resultRT.Release(); UnityEngine.Object.Destroy(_resultRT); }
        if (_blurHMat != null) UnityEngine.Object.Destroy(_blurHMat);
        if (_blurVMat != null) UnityEngine.Object.Destroy(_blurVMat);
        _initialized = false;
    }
}
