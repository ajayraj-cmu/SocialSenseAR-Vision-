using System;
using System.Collections.Generic;
using UnityEngine;
using Socialsense;
using Meta.XR;

/// <summary>
/// Decodes RLE masks from the server and composites them into a single RGBA32
/// overlay texture projected onto a sphere around the user's head.
///
/// The sphere is centered on centerEyeAnchor. The shader uses the head rotation
/// and camera intrinsics to map each sphere fragment to the correct mask pixel
/// via a pinhole projection. Edge fade prevents visible rectangular bounds.
///
/// Uses shared Core/ utilities (RleDecoder, MaskProcessor, EffectConfig, MaskApplicator)
/// and Effects/ (EffectRegistry) to eliminate duplication with StoryPlayback.
/// Delegates sphere management, intrinsics, webcam quad, and labels to Overlay/ companions.
/// </summary>
public class OverlayRenderer : MonoBehaviour
{
    [Header("Camera")]
    [Tooltip("PassthroughCameraAccess for intrinsics")]
    public PassthroughCameraAccess cameraAccess;

    [Tooltip("CenterEyeAnchor — sphere is centered here, left camera position derived from this + LensOffset")]
    public Transform centerEyeAnchor;

    [Header("Overlay Sphere")]
    [Tooltip("Radius of the overlay sphere (meters) — large values minimize stereo parallax error between camera and eyes")]
    public float sphereRadius = 50f;

    [Header("Appearance")]
    [Range(0f, 1f)]
    public float globalAlpha = 1.0f;

    [Range(0, 255)]
    public int segmentAlpha = 235;

    [Range(0f, 0.3f)]
    [Tooltip("Soft fade width at texture edges (hides rectangular FOV boundary)")]
    public float edgeFade = 0.12f;

    [Tooltip("Disable edge fade when inverted effects are active (blur everything but X)")]
    public bool disableEdgeFadeForInvertedEffects = true;

    [Header("Alignment")]
    [Tooltip("Manual pixel offset (X, Y) applied to BOTH eyes")]
    public Vector2 projectionOffset = Vector2.zero;

    [HideInInspector]
    public Vector2 rightEyeOffset = new Vector2(36f, 0f);

    [Tooltip("Multiplier for depth-based right eye shift. 1.0 = physically correct.")]
    public float rightEyeShiftMultiplier = 1.0f;

    [Range(0.9f, 1.1f)]
    [Tooltip("Scale factor for focal length. Values < 1 pull masks toward center.")]
    public float focalLengthScale = 1.095f;

    [Header("Calibration")]
    [Tooltip("Enable right-stick calibration mode")]
    public bool calibrationMode = true;

    [Range(0.5f, 10f)]
    public float calibrationStep = 2f;

    [Header("Debug")]
    [Tooltip("Show raw passthrough texture on blur sphere to verify texture is working")]
    public bool debugRawPassthrough = false;

    [System.NonSerialized]
    public int blurDebugMode = 0;

    [Header("Blur Fallback")]
    [Tooltip("Use frosted overlay fallback for blur (webcam mode). Set automatically by SetWebcamMode().")]
    public bool useOverlayBlurFallback = true;

    [Header("Labels")]
    [Tooltip("Show floating 3D text labels at each segment's center")]
    public bool showLabels = true;

    [Range(0.001f, 0.015f)]
    public float labelCharSize = 0.003f;

    [Header("Rendering")]
    [Tooltip("Draw outlines instead of filled masks")]
    public bool outlineMode = false;

    [Range(1, 5)]
    public int outlineWidth = 2;

    [Range(0, 5)]
    public int softOutlineWidth = 2;

    [Header("Mask Smoothing")]
    [Range(0, 3)]
    public int maskSmoothRadius = 2;

    [Header("Wall Anchor Mode")]
    [Tooltip("When enabled, each segment mask is immediately ray-cast into world space and " +
             "pinned to the nearest wall as a static quad. The quad never follows the user's " +
             "head — it stays exactly where it appeared when first generated. " +
             "Disable to revert to the standard head-following sphere projection.")]
    public bool wallAnchorMode = true;

    [Tooltip("WallAnchorManager component (auto-created if not assigned and wallAnchorMode is on).")]
    public WallAnchorManager wallAnchorManager;

    [Header("Legacy (optional)")]
    [Tooltip("If set, disables sphere and uses this renderer instead")]
    public Renderer overlayQuad;

    [System.NonSerialized]
    public float webcamAspectRatio = 4f / 3f;

    [System.NonSerialized]
    public Texture webcamSourceTexture;

    // --- Textures and buffers ---
    private Texture2D _overlayTexture;
    public Texture2D OverlayTexture => _overlayTexture;
    private Texture2D _blurMaskTexture;
    private Texture2D _pixelateMaskTexture;
    private Color32[] _pixelBuffer;
    private Color32[] _blurMaskBuffer;
    private Color32[] _pixelateMaskBuffer;
    // Logo persists independently of whether the wall is detected each frame.
    // Foreground segments write on top of it in _pixelBuffer so they naturally occlude it.
    private Color32[] _logoLayerBuffer;
    private byte[] _maskTemp;
    private byte[] _maskSmooth;
    private byte[] _prevOverlayAlpha;
    private byte[] _prevBlurAlpha;
    private byte[] _prevPixelateAlpha;
    private int _texWidth, _texHeight;

    // --- Material/renderer shortcuts (set in Start) ---
    private Material _material;
    private Renderer _activeRenderer;
    private bool _useIntrinsics;
    private bool _isWebcamMode;

    // --- Capture-time pose reprojection ---
    private Quaternion _captureRotation = Quaternion.identity;
    private bool _hasCaptureRotation;

    // --- Effect animation ---
    private Dictionary<string, float> _effectStartTimes = new Dictionary<string, float>();
    private const float EFFECT_FADE_DURATION = 0.6f;

    // --- Per-frame state (reset each UpdateOverlay) ---
    private bool _hasInvertedEffectThisFrame;
    private Color _currentInvertedEffectColor = Color.clear;
    private bool _hasBlurEffectThisFrame;
    private float _currentBlurIntensity;
    private bool _isBlurInverted;
    private bool _hasPixelateEffectThisFrame;
    private float _currentPixelateIntensity;
    private bool _isPixelateInverted;

    // --- Persistent blur state ---
    private bool _persistentInvertedBlur;
    private float _persistentBlurIntensity;
    private bool _blurFullyDisabled;

    // --- Depth sample UV ---
    private Vector2 _depthSampleUV = new Vector2(0.5f, 0.5f);

    // --- Webcam readback for blur/pixelate ---
    private Texture2D _blurredWebcamReadback;
    private Color32[] _blurredWebcamPixels;
    private int _blurredW, _blurredH;
    private Texture2D _webcamReadback;
    private Color32[] _webcamPixels;
    private int _webcamW, _webcamH;

    // --- Companion instances ---
    private SphereManager _sphereManager;
    private WebcamQuadManager _webcamQuadManager;
    private IntrinsicsProjector _intrinsicsProjector;
    private LabelManager _labelManager;

    // --- Shader property IDs ---
    private static readonly int GlobalAlphaId = Shader.PropertyToID("_GlobalAlpha");
    private static readonly int EdgeFadeId = Shader.PropertyToID("_EdgeFade");
    private static readonly int UseIntrinsicsId = Shader.PropertyToID("_UseIntrinsicsProjection");
    private static readonly int DebugModeId = Shader.PropertyToID("_DebugMode");
    private static readonly int OutOfBoundsColorId = Shader.PropertyToID("_OutOfBoundsColor");
    private static readonly int TextureResolutionId = Shader.PropertyToID("_TextureResolution");

    // --- Debug timers ---
    private float _debugLogTimer;
    private float _blurDebugTimer;

    // ----------------------------------------------------------------
    // Lifecycle
    // ----------------------------------------------------------------

    void Start()
    {
        _webcamQuadManager = new WebcamQuadManager();
        _intrinsicsProjector = new IntrinsicsProjector();
        _labelManager = new LabelManager { CharSize = labelCharSize, SphereRadius = sphereRadius };

        if (overlayQuad != null)
        {
            Debug.Log("[OverlayRenderer] MODE: Legacy quad");
            _activeRenderer = overlayQuad;
            _material = overlayQuad.material;
        }
        else
        {
            Debug.Log("[OverlayRenderer] MODE: Sphere");
            _sphereManager = new SphereManager();
            _sphereManager.CreateSpheres(sphereRadius);
            _activeRenderer = _sphereManager.OverlayRendererComponent;
            _material = _sphereManager.OverlayMaterial;
        }

        rightEyeOffset = new Vector2(36f, 0f);
        _useIntrinsics = !_isWebcamMode && (cameraAccess != null && centerEyeAnchor != null);

        if (!_useIntrinsics)
            useOverlayBlurFallback = true;

        if (_material != null)
        {
            _material.SetFloat(UseIntrinsicsId, _useIntrinsics ? 1f : 0f);
            _material.SetFloat(DebugModeId, 0f);
            _material.SetFloat(EdgeFadeId, edgeFade);
        }

        if (!_useIntrinsics)
            Debug.LogWarning("[OverlayRenderer] cameraAccess or centerEyeAnchor missing — falling back to UV mode");

        if (!_isWebcamMode)
            _intrinsicsProjector.InitializeDepth(centerEyeAnchor);

        // Wall anchor mode: auto-create WallAnchorManager if not already assigned
        if (wallAnchorMode && wallAnchorManager == null)
        {
            wallAnchorManager = gameObject.AddComponent<WallAnchorManager>();
            Debug.Log("[OverlayRenderer] Auto-created WallAnchorManager for wall-anchor mode");
        }
    }

    void Update()
    {
        if (_material == null) return;

        if (calibrationMode)
            HandleCalibrationInput();

        // Keep spheres centered on user's eye
        if (centerEyeAnchor != null)
            _sphereManager?.UpdatePositions(centerEyeAnchor.position);

        // GPU blur passthrough texture for blur sphere (AR mode only)
        if (!_isWebcamMode && cameraAccess != null && cameraAccess.IsPlaying && _sphereManager != null)
        {
            Texture passthrough = cameraAccess.GetTexture();
            if (passthrough != null)
            {
                bool needsBlur = !_blurFullyDisabled && (_hasBlurEffectThisFrame || _persistentInvertedBlur);
                float blurIntensity = _hasBlurEffectThisFrame ? _currentBlurIntensity : _persistentBlurIntensity;
                RenderTexture blurredRT = needsBlur ? _sphereManager.BlurTexture(passthrough, blurIntensity) : null;
                _sphereManager.UpdatePassthroughTextures(passthrough, blurredRT, debugRawPassthrough);
            }
        }

        // Intrinsics projection (AR mode only)
        if (!_useIntrinsics || cameraAccess == null || !cameraAccess.IsPlaying)
        {
            _material.SetFloat(UseIntrinsicsId, 0f);
            _sphereManager?.DisableIntrinsics();
        }
        else
        {
            Material[] mats = AllMaterials();
            bool ok = _intrinsicsProjector.UpdateProjection(
                cameraAccess, centerEyeAnchor,
                _captureRotation, _hasCaptureRotation,
                focalLengthScale, projectionOffset,
                rightEyeOffset, rightEyeShiftMultiplier,
                _depthSampleUV, mats, _texWidth, _texHeight);

            if (!ok)
                _material.SetFloat(UseIntrinsicsId, 0f);

            if (Time.time - _debugLogTimer > 3f)
            {
                _debugLogTimer = Time.time;
                Debug.Log($"[OverlayRenderer] EyePos={centerEyeAnchor.position} EyeFwd={centerEyeAnchor.forward:F3}" +
                          $" texSize={_texWidth}x{_texHeight} focalScale={focalLengthScale:F3}" +
                          $" rightEyeShift={rightEyeShiftMultiplier:F2}");
            }
        }
    }

    // ----------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------

    /// <summary>
    /// Called by SocialSenseClient when the input mode changes.
    /// Switches between AR mode (intrinsics + real blur) and webcam mode (flat quad + overlay blur fallback).
    /// </summary>
    public void SetWebcamMode(bool webcamMode)
    {
        _isWebcamMode = webcamMode;
        useOverlayBlurFallback = webcamMode;
        _useIntrinsics = !webcamMode && (cameraAccess != null && centerEyeAnchor != null);

        if (_material != null)
            _material.SetFloat(UseIntrinsicsId, _useIntrinsics ? 1f : 0f);

        _sphereManager?.SetWebcamMode(webcamMode);

        if (webcamMode)
        {
            Camera cam = FindCamera();
            _webcamQuadManager.EnsureQuad(cam);
        }
        else
        {
            _webcamQuadManager.SetActive(false);
        }

        Debug.Log($"[OverlayRenderer] SetWebcamMode({webcamMode}): useIntrinsics={_useIntrinsics}, blurFallback={useOverlayBlurFallback}");
    }

    public void UpdateOverlay(
        Google.Protobuf.Collections.RepeatedField<SceneSegment> segments,
        Quaternion? captureRotation = null, bool effectsCleared = false)
    {
        if (captureRotation.HasValue)
        {
            _captureRotation = captureRotation.Value;
            _hasCaptureRotation = true;
        }

        if (effectsCleared)
        {
            Debug.Log("[OverlayRenderer] Server sent effectsCleared=true, clearing all persistent effects");
            ClearOverlay();
            _labelManager.HideAll();
            return;
        }

        if (_activeRenderer == null || segments == null || segments.Count == 0)
        {
            if (_persistentInvertedBlur)
            {
                ClearVisualOverlays();
                _sphereManager?.UpdateBlurSphere(true, _persistentBlurIntensity, true,
                    globalAlpha, blurDebugMode, debugRawPassthrough, _blurMaskTexture, _texWidth, _texHeight);
            }
            else
            {
                ClearOverlay();
            }
            _labelManager.HideAll();
            return;
        }

        int maskW = (int)segments[0].MaskWidth;
        int maskH = (int)segments[0].MaskHeight;
        if (maskW == 0 || maskH == 0)
        {
            if (_persistentInvertedBlur)
            {
                ClearVisualOverlays();
                _sphereManager?.UpdateBlurSphere(true, _persistentBlurIntensity, true,
                    globalAlpha, blurDebugMode, debugRawPassthrough, _blurMaskTexture, _texWidth, _texHeight);
            }
            else
            {
                ClearOverlay();
            }
            _labelManager.HideAll();
            return;
        }

        EnsureTexture(maskW, maskH);

        Array.Fill(_pixelBuffer, EffectConfig.ClearPixel);
        Array.Fill(_blurMaskBuffer, EffectConfig.ClearPixel);
        Array.Fill(_pixelateMaskBuffer, EffectConfig.ClearPixel);

        // Stamp the persistent logo layer into _pixelBuffer first so the logo
        // is always visible — even when the wall segment is not detected this frame.
        // Foreground segments processed below will overwrite their pixels on top,
        // achieving correct depth ordering without any extra logic.
        CompositeLogoLayer();

        // Reset per-frame state
        var activeEffectIds = new HashSet<string>();
        _hasInvertedEffectThisFrame = false;
        _currentInvertedEffectColor = Color.clear;
        _hasBlurEffectThisFrame = false;
        _currentBlurIntensity = 0f;
        _isBlurInverted = false;
        _hasPixelateEffectThisFrame = false;
        _currentPixelateIntensity = 0f;
        _isPixelateInverted = false;

        // Pre-compute webcam readback data (once per frame, not per-segment)
        PrecomputeWebcamData(segments);

        float segCenterSumX = 0f, segCenterSumY = 0f;
        int segCenterCount = 0;
        int segIndex = 0;

        foreach (var seg in segments)
        {
            if (seg.CenterX > 0 && seg.CenterY > 0)
            {
                segCenterSumX += seg.CenterX;
                segCenterSumY += seg.CenterY;
                segCenterCount++;
            }

            Color32 segColor = EffectConfig.GetBrightColor(segIndex);
            string effectType = seg.Effect?.EffectType ?? "none";
            float rawIntensity = seg.Effect?.Intensity ?? 0f;
            string colorHex = seg.Effect?.ColorHex ?? null;
            bool hasEffect = effectType != "none" && effectType != "";
            float intensity = (hasEffect && rawIntensity <= 0f) ? 0.7f : rawIntensity;
            bool invert = seg.Effect?.Invert ?? false;

            bool hasMask = seg.RleMask != null && !seg.RleMask.IsEmpty &&
                           seg.MaskWidth == maskW && seg.MaskHeight == maskH;

            bool fullScreenFallback = false;

            // Inverted effect with no mask: apply to everything
            if (hasEffect && invert && !hasMask)
            {
                Array.Fill(_maskTemp, (byte)255);
                fullScreenFallback = true;
            }
            else if (!hasMask)
            {
                segIndex++;
                continue;
            }
            else
            {
                RleDecoder.DecodeToMask(seg.RleMask.ToByteArray(), _maskTemp, maskW, maskH);
                MaskProcessor.Smooth(_maskTemp, _maskSmooth, maskW, maskH, maskSmoothRadius);
            }

            if (hasEffect)
            {
                // Track animation start time
                string trackKey = seg.TrackId.Length > 0 ? seg.TrackId : $"seg_{segIndex}";
                string effectKey = invert ? $"__inverted__:{effectType}" : $"{trackKey}:{effectType}";
                activeEffectIds.Add(effectKey);

                if (!_effectStartTimes.ContainsKey(effectKey))
                    _effectStartTimes[effectKey] = Time.time;

                float elapsed = Time.time - _effectStartTimes[effectKey];
                float animProgress = Mathf.Clamp01(elapsed / EFFECT_FADE_DURATION);

                // Track blur state
                if (effectType == "blur")
                {
                    if (!useOverlayBlurFallback)
                    {
                        _blurFullyDisabled = false;
                        _hasBlurEffectThisFrame = true;
                        _currentBlurIntensity = Mathf.Max(_currentBlurIntensity, intensity);

                        if (invert)
                        {
                            _isBlurInverted = true;
                            _persistentInvertedBlur = true;
                            _persistentBlurIntensity = intensity;
                        }
                        else
                        {
                            _isBlurInverted = false;
                            _persistentInvertedBlur = false;
                            _persistentBlurIntensity = 0f;
                        }
                    }
                    else
                    {
                        _hasBlurEffectThisFrame = false;
                        _currentBlurIntensity = 0f;
                        _isBlurInverted = false;
                        _persistentInvertedBlur = false;
                        _persistentBlurIntensity = 0f;
                    }
                }

                // Track pixelate state
                if (effectType == "pixelate")
                {
                    _hasPixelateEffectThisFrame = true;
                    _currentPixelateIntensity = Mathf.Max(_currentPixelateIntensity, intensity);
                    if (invert) _isPixelateInverted = true;
                }

                // Invert mask
                if (invert)
                {
                    _hasInvertedEffectThisFrame = true;
                    if (!fullScreenFallback)
                        MaskProcessor.Invert(_maskTemp, maskW * maskH);

                    Color32 effectColor32 = EffectRegistry.GetEffectColor(effectType, intensity, segColor, colorHex);
                    effectColor32.a = (byte)(effectColor32.a * animProgress);
                    _currentInvertedEffectColor = new Color(
                        effectColor32.r / 255f, effectColor32.g / 255f,
                        effectColor32.b / 255f, effectColor32.a / 255f);
                }

                // Apply effect via shared Effects/ pipeline
                bool isTextureEffect = effectType == "logo" || effectType == "custom_image";
                float effectIntensity = (effectType == "blur" || isTextureEffect) ? intensity : intensity * animProgress;

                // Logo / custom_image: write into the persistent layer so they survive
                // frames where the wall/segment is not detected, then composite immediately.
                if (effectType == "logo")
                {
                    Array.Fill(_logoLayerBuffer, EffectConfig.ClearPixel);
                    LogoEffect.ApplyOverlay(_maskTemp, _logoLayerBuffer, maskW, maskH, effectIntensity);
                    CompositeLogoLayer();   // refresh _pixelBuffer with updated logo position
                }
                else if (effectType == "custom_image")
                {
                    Array.Fill(_logoLayerBuffer, EffectConfig.ClearPixel);
                    CustomImageEffect.ApplyOverlay(_maskTemp, _logoLayerBuffer, maskW, maskH, effectIntensity);
                    CompositeLogoLayer();
                }
                else
                {
                    EffectRegistry.ApplyOverlay(
                        effectType, _maskTemp, _pixelBuffer, maskW, maskH,
                        effectIntensity, segColor, colorHex,
                        blurBuffer: _blurMaskBuffer,
                        pixelateBuffer: _pixelateMaskBuffer,
                        blurredPixels: _blurredWebcamPixels, blurW: _blurredW, blurH: _blurredH,
                        webcamPixels: _webcamPixels, webcamW: _webcamW, webcamH: _webcamH,
                        useOverlayBlurFallback: useOverlayBlurFallback);
                }
            }
            else if (outlineMode)
            {
                segColor.a = 255;
                OutlineEffect.ApplyOverlay(_maskTemp, _pixelBuffer, maskW, maskH, segColor, outlineWidth, softOutlineWidth);
            }
            else
            {
                Color32 fillColor = EffectConfig.GetAssetColor(seg.AssetClass, (byte)segmentAlpha);
                MaskApplicator.ApplyToOverlayBuffer(_maskTemp, _pixelBuffer, maskW, maskH, fillColor);
            }

            segIndex++;
        }

        // Depth sample UV from segment centers
        if (segCenterCount > 0 && _useIntrinsics && _intrinsicsProjector.IsValid && centerEyeAnchor != null)
        {
            Camera cam = centerEyeAnchor.GetComponent<Camera>() ?? Camera.main;
            _depthSampleUV = _intrinsicsProjector.ComputeDepthSampleUV(
                segCenterSumX / segCenterCount, segCenterSumY / segCenterCount, cam);
        }

        // Purge stale effect start times
        var staleKeys = new List<string>();
        foreach (var key in _effectStartTimes.Keys)
            if (!activeEffectIds.Contains(key)) staleKeys.Add(key);
        foreach (var key in staleKeys)
            _effectStartTimes.Remove(key);

        // Clear persistent blur if we saw segments but none had blur
        if (segIndex > 0 && !_hasBlurEffectThisFrame)
        {
            _persistentInvertedBlur = false;
            _persistentBlurIntensity = 0f;
        }

        // Temporal alpha smoothing
        MaskProcessor.TemporalSmooth(_pixelBuffer, ref _prevOverlayAlpha);

        _overlayTexture.SetPixels32(_pixelBuffer);
        _overlayTexture.Apply(false);

        // --- Wall Anchor Mode ---
        // After the overlay texture is baked, anchor new segments to the nearest wall.
        // This fires once per unique track-ID; subsequent frames are ignored for that track.
        if (wallAnchorMode && wallAnchorManager != null && _useIntrinsics &&
            _intrinsicsProjector.IsValid && centerEyeAnchor != null)
        {
            Quaternion headRot = _hasCaptureRotation ? _captureRotation : centerEyeAnchor.rotation;
            wallAnchorManager.TryAnchorSegments(
                segments, _intrinsicsProjector, centerEyeAnchor, headRot, _overlayTexture);
        }

        // Upload blur mask
        if (_hasBlurEffectThisFrame && _blurMaskTexture != null)
        {
            MaskProcessor.TemporalSmooth(_blurMaskBuffer, ref _prevBlurAlpha);
            _blurMaskTexture.SetPixels32(_blurMaskBuffer);
            _blurMaskTexture.Apply(false);
        }

        // Upload pixelate mask
        if (_hasPixelateEffectThisFrame && _pixelateMaskTexture != null)
        {
            MaskProcessor.TemporalSmooth(_pixelateMaskBuffer, ref _prevPixelateAlpha);
            _pixelateMaskTexture.SetPixels32(_pixelateMaskBuffer);
            _pixelateMaskTexture.Apply(false);
        }

        // Set material properties
        if (_material != null)
        {
            _material.mainTexture = _overlayTexture;
            _material.SetFloat(GlobalAlphaId, globalAlpha);

            float effectiveEdgeFade = (_hasInvertedEffectThisFrame && disableEdgeFadeForInvertedEffects) ? 0f : edgeFade;
            _material.SetFloat(EdgeFadeId, effectiveEdgeFade);
            _material.SetColor(OutOfBoundsColorId, _currentInvertedEffectColor);
        }

        // Webcam overlay quad
        if (_isWebcamMode && _webcamQuadManager.Material != null)
        {
            _webcamQuadManager.Material.mainTexture = _overlayTexture;
            _webcamQuadManager.SetActive(true);
            Camera cam = FindCamera();
            _webcamQuadManager.UpdateSize(cam, webcamAspectRatio);
        }

        // Blur/pixelate spheres
        if (_sphereManager != null)
        {
            bool blurActive = !_blurFullyDisabled &&
                ((_hasBlurEffectThisFrame && _currentBlurIntensity > 0f) || _persistentInvertedBlur);
            bool blurInverted = _isBlurInverted || _persistentInvertedBlur;
            float effectiveBlurIntensity = _hasBlurEffectThisFrame ? _currentBlurIntensity : _persistentBlurIntensity;

            _sphereManager.UpdateBlurSphere(blurActive, effectiveBlurIntensity, blurInverted,
                globalAlpha, blurDebugMode, debugRawPassthrough, _blurMaskTexture, _texWidth, _texHeight);

            _sphereManager.UpdatePixelateSphere(_hasPixelateEffectThisFrame, _currentPixelateIntensity,
                _isPixelateInverted, globalAlpha, _pixelateMaskTexture, _texWidth, _texHeight);
        }

        // Labels
        if (showLabels && _intrinsicsProjector.IsValid && centerEyeAnchor != null)
        {
            _labelManager.CharSize = labelCharSize;
            Quaternion headRot = _hasCaptureRotation ? _captureRotation : centerEyeAnchor.rotation;
            _labelManager.UpdateLabels(segments,
                _intrinsicsProjector.FocalLength, _intrinsicsProjector.PrincipalPoint,
                _intrinsicsProjector.ActualW, _intrinsicsProjector.ActualH,
                _intrinsicsProjector.CameraProjectionRot,
                centerEyeAnchor, headRot);
        }
        else
        {
            _labelManager.HideAll();
        }
    }

    /// <summary>
    /// Fully clears all overlays AND resets persistent blur state.
    /// Called by "vibe clear" command to stop all effects.
    /// </summary>
    public void ClearOverlay()
    {
        _blurFullyDisabled = true;
        _persistentInvertedBlur = false;
        _persistentBlurIntensity = 0f;
        _isBlurInverted = false;
        _hasBlurEffectThisFrame = false;
        _currentBlurIntensity = 0f;

        // Clear persistent logo / custom_image so "vibe clear" removes textures from the wall.
        if (_logoLayerBuffer != null)
            Array.Fill(_logoLayerBuffer, EffectConfig.ClearPixel);
        LogoEffect.ResetBBox();
        CustomImageEffect.Reset();

        // Destroy all wall-anchored quads
        wallAnchorManager?.ClearAnchors();

        ClearVisualOverlays();

        _hasPixelateEffectThisFrame = false;
        _currentPixelateIntensity = 0f;
        _isPixelateInverted = false;

        if (_sphereManager != null)
        {
            _sphereManager.UpdateBlurSphere(false, 0, false, 0, 0, false, null, 0, 0);
            _sphereManager.UpdatePixelateSphere(false, 0, false, 0, null, 0, 0);

            if (_sphereManager.BlurMaterial != null)
            {
                _sphereManager.BlurMaterial.SetFloat(Shader.PropertyToID("_OutOfBoundsAlpha"), 0f);
                _sphereManager.BlurMaterial.SetFloat(GlobalAlphaId, 0f);
            }
        }

        if (_material != null)
            _material.SetColor(OutOfBoundsColorId, Color.clear);

        _webcamQuadManager?.SetActive(false);
        _labelManager?.HideAll();
    }

    // ----------------------------------------------------------------
    // Private helpers
    // ----------------------------------------------------------------

    /// <summary>
    /// Clears visual overlays but keeps persistent blur going if active.
    /// Called when segments are empty but we still want blur to continue.
    /// </summary>
    private void ClearVisualOverlays()
    {
        if (_overlayTexture != null && _pixelBuffer != null)
        {
            Array.Fill(_pixelBuffer, EffectConfig.ClearPixel);
            _overlayTexture.SetPixels32(_pixelBuffer);
            _overlayTexture.Apply(false);
        }

        // Persistent inverted blur: fill entire blur mask
        if (_persistentInvertedBlur)
        {
            if (_blurMaskTexture == null || _blurMaskBuffer == null)
                EnsureTexture(64, 64);

            byte blurAlpha = (byte)(255 * _persistentBlurIntensity);
            Array.Fill(_blurMaskBuffer, new Color32(255, 255, 255, blurAlpha));
            _blurMaskTexture.SetPixels32(_blurMaskBuffer);
            _blurMaskTexture.Apply(false);

            if (_sphereManager?.BlurMaterial != null)
            {
                _sphereManager.BlurMaterial.mainTexture = _blurMaskTexture;
                _sphereManager.BlurMaterial.SetVector(TextureResolutionId, new Vector4(_texWidth, _texHeight, 0, 0));
            }
        }
        else if (_blurMaskTexture != null && _blurMaskBuffer != null)
        {
            Array.Fill(_blurMaskBuffer, EffectConfig.ClearPixel);
            _blurMaskTexture.SetPixels32(_blurMaskBuffer);
            _blurMaskTexture.Apply(false);
        }

        if (!_persistentInvertedBlur && _material != null)
            _material.SetColor(OutOfBoundsColorId, Color.clear);

        _hasBlurEffectThisFrame = false;
        _currentBlurIntensity = 0f;
        _isBlurInverted = false;
        _currentInvertedEffectColor = Color.clear;
        _hasInvertedEffectThisFrame = false;

        if (_pixelateMaskTexture != null && _pixelateMaskBuffer != null)
        {
            Array.Fill(_pixelateMaskBuffer, EffectConfig.ClearPixel);
            _pixelateMaskTexture.SetPixels32(_pixelateMaskBuffer);
            _pixelateMaskTexture.Apply(false);
        }
        _hasPixelateEffectThisFrame = false;
        _currentPixelateIntensity = 0f;
        _isPixelateInverted = false;
    }

    /// <summary>
    /// Pre-compute GPU readback data for webcam blur and pixelate effects.
    /// Done once per frame before the segment loop (not per-segment).
    /// </summary>
    private void PrecomputeWebcamData(Google.Protobuf.Collections.RepeatedField<SceneSegment> segments)
    {
        _blurredWebcamPixels = null;
        _blurredW = 0;
        _blurredH = 0;
        _webcamPixels = null;
        _webcamW = 0;
        _webcamH = 0;

        if (!useOverlayBlurFallback || webcamSourceTexture == null || _sphereManager == null)
            return;

        // Scan segments for blur/pixelate effects
        float maxBlurIntensity = 0f;
        bool needsPixelateReadback = false;
        foreach (var seg in segments)
        {
            string et = seg.Effect?.EffectType ?? "none";
            float ei = seg.Effect?.Intensity ?? 0f;
            if (ei <= 0f) ei = 0.7f;
            if (et == "blur") maxBlurIntensity = Mathf.Max(maxBlurIntensity, ei);
            if (et == "pixelate") needsPixelateReadback = true;
        }

        // GPU blur the webcam texture
        if (maxBlurIntensity > 0f)
        {
            RenderTexture blurredRT = _sphereManager.BlurTexture(webcamSourceTexture, maxBlurIntensity);
            if (blurredRT != null)
            {
                _blurredW = blurredRT.width;
                _blurredH = blurredRT.height;
                EnsureReadbackTexture(ref _blurredWebcamReadback, _blurredW, _blurredH);
                _blurredWebcamPixels = GaussianBlurPipeline.ReadRT(blurredRT, _blurredWebcamReadback);
            }
        }

        // Readback raw webcam pixels for pixelate
        if (needsPixelateReadback)
        {
            int srcW = webcamSourceTexture.width;
            int srcH = webcamSourceTexture.height;
            EnsureReadbackTexture(ref _webcamReadback, srcW, srcH);

            RenderTexture tempRT = RenderTexture.GetTemporary(srcW, srcH, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(webcamSourceTexture, tempRT);
            _webcamPixels = GaussianBlurPipeline.ReadRT(tempRT, _webcamReadback);
            RenderTexture.ReleaseTemporary(tempRT);
            _webcamW = srcW;
            _webcamH = srcH;
        }
    }

    private static void EnsureReadbackTexture(ref Texture2D tex, int w, int h)
    {
        if (tex == null || tex.width != w || tex.height != h)
        {
            if (tex != null) Destroy(tex);
            tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        }
    }

    private void EnsureTexture(int w, int h)
    {
        if (_overlayTexture != null && _texWidth == w && _texHeight == h)
            return;

        if (_overlayTexture != null) Destroy(_overlayTexture);
        if (_blurMaskTexture != null) Destroy(_blurMaskTexture);
        if (_pixelateMaskTexture != null) Destroy(_pixelateMaskTexture);

        _texWidth = w;
        _texHeight = h;
        int total = w * h;

        _overlayTexture = CreateOverlayTexture(w, h);
        _blurMaskTexture = CreateOverlayTexture(w, h);
        _pixelateMaskTexture = CreateOverlayTexture(w, h);

        _pixelBuffer = new Color32[total];
        _blurMaskBuffer = new Color32[total];
        _pixelateMaskBuffer = new Color32[total];
        _logoLayerBuffer = new Color32[total];   // persistent logo layer
        _maskTemp = new byte[total];
        _maskSmooth = new byte[total];

        Debug.Log($"[OverlayRenderer] Created overlay textures {w}x{h}");
    }

    /// <summary>
    /// Stamps every non-transparent pixel from _logoLayerBuffer into _pixelBuffer.
    /// Called once per frame before the segment loop so the logo is always present,
    /// and again immediately after the logo effect is refreshed mid-loop.
    /// Foreground segment effects written after this call naturally overwrite logo
    /// pixels in _pixelBuffer — achieving correct depth ordering at zero extra cost.
    /// </summary>
    private void CompositeLogoLayer()
    {
        if (_logoLayerBuffer == null || _pixelBuffer == null) return;
        int len = _logoLayerBuffer.Length;
        for (int i = 0; i < len; i++)
        {
            if (_logoLayerBuffer[i].a > 0)
                _pixelBuffer[i] = _logoLayerBuffer[i];
        }
    }

    private static Texture2D CreateOverlayTexture(int w, int h)
    {
        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        tex.filterMode = FilterMode.Bilinear;
        tex.wrapMode = TextureWrapMode.Clamp;
        return tex;
    }

    private Material[] AllMaterials()
    {
        return new Material[]
        {
            _material,
            _sphereManager?.BlurMaterial,
            _sphereManager?.PixelateMaterial,
        };
    }

    private Camera FindCamera()
    {
        Camera cam = null;
        if (centerEyeAnchor != null)
            cam = centerEyeAnchor.GetComponentInChildren<Camera>();
        if (cam == null) cam = Camera.main;
        return cam;
    }

    // ----------------------------------------------------------------
    // Calibration (OVR controller input)
    // ----------------------------------------------------------------

    private bool _lBtnX_prev, _lBtnY_prev;
    private bool _lStickUp_prev, _lStickDown_prev, _lStickLeft_prev, _lStickRight_prev;
    private bool _btnA_prev, _btnB_prev;
    private bool _stickUp_prev, _stickDown_prev, _stickLeft_prev, _stickRight_prev;

    private void HandleCalibrationInput()
    {
        // LEFT CONTROLLER: adjusts projectionOffset (BOTH eyes)
        bool btnX = OVRInput.Get(OVRInput.Button.One, OVRInput.Controller.LTouch);
        bool btnY = OVRInput.Get(OVRInput.Button.Two, OVRInput.Controller.LTouch);
        Vector2 lStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);

        bool lStickRight = lStick.x > 0.5f;
        bool lStickLeft = lStick.x < -0.5f;
        bool lStickUp = lStick.y > 0.5f;
        bool lStickDown = lStick.y < -0.5f;

        if (btnX && !_lBtnX_prev) { projectionOffset.x += calibrationStep; LogCalibration("X: both +X", true); }
        if (btnY && !_lBtnY_prev) { projectionOffset.x -= calibrationStep; LogCalibration("Y: both -X", true); }
        if (lStickUp && !_lStickUp_prev) { projectionOffset.y -= calibrationStep; LogCalibration("LStickUp: both -Y", true); }
        if (lStickDown && !_lStickDown_prev) { projectionOffset.y += calibrationStep; LogCalibration("LStickDown: both +Y", true); }
        if (lStickRight && !_lStickRight_prev) { projectionOffset.x += calibrationStep; LogCalibration("LStickRight: both +X", true); }
        if (lStickLeft && !_lStickLeft_prev) { projectionOffset.x -= calibrationStep; LogCalibration("LStickLeft: both -X", true); }

        _lBtnX_prev = btnX; _lBtnY_prev = btnY;
        _lStickUp_prev = lStickUp; _lStickDown_prev = lStickDown;
        _lStickLeft_prev = lStickLeft; _lStickRight_prev = lStickRight;

        // RIGHT CONTROLLER: adjusts rightEyeOffset / rightEyeShiftMultiplier
        bool btnA = OVRInput.Get(OVRInput.Button.One, OVRInput.Controller.RTouch);
        bool btnB = OVRInput.Get(OVRInput.Button.Two, OVRInput.Controller.RTouch);
        Vector2 stick = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick);

        bool stickRight = stick.x > 0.5f;
        bool stickLeft = stick.x < -0.5f;
        bool stickUp = stick.y > 0.5f;
        bool stickDown = stick.y < -0.5f;

        if (btnA && !_btnA_prev) { rightEyeShiftMultiplier += 0.1f; Debug.Log($"[Calibrate] A: shiftMul={rightEyeShiftMultiplier:F2}"); }
        if (btnB && !_btnB_prev) { rightEyeShiftMultiplier -= 0.1f; Debug.Log($"[Calibrate] B: shiftMul={rightEyeShiftMultiplier:F2}"); }
        if (stickUp && !_stickUp_prev) { rightEyeOffset.y -= calibrationStep; LogCalibration("RStickUp: right -Y", false); }
        if (stickDown && !_stickDown_prev) { rightEyeOffset.y += calibrationStep; LogCalibration("RStickDown: right +Y", false); }
        if (stickRight && !_stickRight_prev) { rightEyeOffset.x += calibrationStep; LogCalibration("RStickRight: right +X", false); }
        if (stickLeft && !_stickLeft_prev) { rightEyeOffset.x -= calibrationStep; LogCalibration("RStickLeft: right -X", false); }

        _btnA_prev = btnA; _btnB_prev = btnB;
        _stickUp_prev = stickUp; _stickDown_prev = stickDown;
        _stickLeft_prev = stickLeft; _stickRight_prev = stickRight;
    }

    private void LogCalibration(string action, bool isBothEyes)
    {
        if (isBothEyes)
            Debug.Log($"[Calibrate] {action} => projectionOffset=({projectionOffset.x:F1}, {projectionOffset.y:F1})");
        else
            Debug.Log($"[Calibrate] {action} => rightEyeOffset=({rightEyeOffset.x:F1}, {rightEyeOffset.y:F1})");
    }

    // ----------------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------------

    void OnDestroy()
    {
        if (_overlayTexture != null) Destroy(_overlayTexture);
        if (_blurMaskTexture != null) Destroy(_blurMaskTexture);
        if (_pixelateMaskTexture != null) Destroy(_pixelateMaskTexture);
        if (_blurredWebcamReadback != null) Destroy(_blurredWebcamReadback);
        if (_webcamReadback != null) Destroy(_webcamReadback);

        _sphereManager?.Destroy();
        _webcamQuadManager?.Destroy();
        _labelManager?.DestroyAll();
    }

    return result;

}
