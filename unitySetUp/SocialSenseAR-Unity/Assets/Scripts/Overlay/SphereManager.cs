using System;
using UnityEngine;

/// <summary>
/// Creates and manages the overlay, blur, and pixelate spheres for AR mode.
/// The overlay sphere displays composited masks, blur sphere applies real
/// passthrough Gaussian blur, pixelate sphere applies UV-snapped pixelation.
/// </summary>
public class SphereManager
{
    public GameObject OverlaySphere { get; private set; }
    public GameObject BlurSphere { get; private set; }
    public GameObject PixelateSphere { get; private set; }

    public Material OverlayMaterial { get; private set; }
    public Material BlurMaterial { get; private set; }
    public Material PixelateMaterial { get; private set; }

    public Renderer OverlayRendererComponent { get; private set; }

    // GPU blur pipeline
    private GaussianBlurPipeline _blurPipeline;

    // Shader property IDs
    private static readonly int GlobalAlphaId = Shader.PropertyToID("_GlobalAlpha");
    private static readonly int OutOfBoundsAlphaId = Shader.PropertyToID("_OutOfBoundsAlpha");
    private static readonly int DebugModeId = Shader.PropertyToID("_DebugMode");
    private static readonly int DebugRawPassthroughId = Shader.PropertyToID("_DebugRawPassthrough");
    private static readonly int PassthroughTexId = Shader.PropertyToID("_PassthroughTex");
    private static readonly int TextureResolutionId = Shader.PropertyToID("_TextureResolution");
    private static readonly int PixelBlockSizeId = Shader.PropertyToID("_PixelBlockSize");
    private static readonly int UseIntrinsicsId = Shader.PropertyToID("_UseIntrinsicsProjection");

    /// <summary>
    /// Create all three spheres (overlay, blur, pixelate) and initialize the blur pipeline.
    /// </summary>
    public void CreateSpheres(float radius)
    {
        // Main overlay sphere (outlines, dim, highlight, etc.)
        OverlaySphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        OverlaySphere.name = "OverlaySphere";
        DestroyCollider(OverlaySphere);
        OverlaySphere.transform.localScale = Vector3.one * radius * 2f;
        OverlayRendererComponent = OverlaySphere.GetComponent<Renderer>();

        var shader = Shader.Find("SocialSense/SegmentOverlay");
        if (shader != null)
        {
            OverlayMaterial = new Material(shader);
            OverlayRendererComponent.material = OverlayMaterial;
            Debug.Log("[SphereManager] Created overlay sphere with SegmentOverlay shader");
        }
        else
        {
            Debug.LogError("[SphereManager] Cannot find SocialSense/SegmentOverlay shader!");
        }

        // Blur sphere (real passthrough blur) — slightly larger, renders after overlay
        BlurSphere = CreateEffectSphere("BlurSphere", radius * 2.02f,
            "PassthroughBlurMaterial", "SocialSense/PassthroughBlur", out var blurMat);
        BlurMaterial = blurMat;

        // Pixelate sphere (real passthrough pixelation) — between overlay and blur sphere
        PixelateSphere = CreateEffectSphere("PixelateSphere", radius * 2.01f,
            "PassthroughPixelateMaterial", "SocialSense/PassthroughPixelate", out var pixMat);
        PixelateMaterial = pixMat;

        // Initialize GPU blur pipeline
        _blurPipeline = new GaussianBlurPipeline();
        _blurPipeline.Init();
    }

    private GameObject CreateEffectSphere(string name, float scale,
        string resourceMaterialName, string shaderFallbackName, out Material material)
    {
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.name = name;
        DestroyCollider(sphere);
        sphere.transform.localScale = Vector3.one * scale;
        var renderer = sphere.GetComponent<Renderer>();

        // Try Resources first (works in builds), then Shader.Find (works in editor)
        var matAsset = Resources.Load<Material>(resourceMaterialName);
        if (matAsset != null)
        {
            material = new Material(matAsset);
            renderer.material = material;
            Debug.Log($"[SphereManager] Created {name} with {resourceMaterialName} from Resources");
        }
        else
        {
            var fallbackShader = Shader.Find(shaderFallbackName);
            if (fallbackShader != null)
            {
                material = new Material(fallbackShader);
                renderer.material = material;
                Debug.Log($"[SphereManager] Created {name} with {shaderFallbackName} (Shader.Find fallback)");
            }
            else
            {
                material = null;
                Debug.LogError($"[SphereManager] Cannot find {resourceMaterialName} or {shaderFallbackName}!");
            }
        }

        sphere.SetActive(false);
        return sphere;
    }

    private static void DestroyCollider(GameObject obj)
    {
        var col = obj.GetComponent<Collider>();
        if (col != null) UnityEngine.Object.Destroy(col);
    }

    /// <summary>Keep all spheres centered on the user's eye position.</summary>
    public void UpdatePositions(Vector3 eyePosition)
    {
        if (OverlaySphere != null) OverlaySphere.transform.position = eyePosition;
        if (BlurSphere != null) BlurSphere.transform.position = eyePosition;
        if (PixelateSphere != null) PixelateSphere.transform.position = eyePosition;
    }

    /// <summary>
    /// GPU two-pass Gaussian blur on a source texture.
    /// Returns the blurred RenderTexture (owned by the pipeline, do not destroy).
    /// </summary>
    public RenderTexture BlurTexture(Texture source, float intensity)
    {
        if (source == null || _blurPipeline == null || !_blurPipeline.IsValid)
            return null;
        return _blurPipeline.Blur(source, intensity);
    }

    /// <summary>Feed passthrough/blurred textures to blur and pixelate materials.</summary>
    public void UpdatePassthroughTextures(
        Texture passthrough, RenderTexture blurredRT, bool debugRawPassthrough)
    {
        if (BlurMaterial != null)
        {
            BlurMaterial.SetTexture(PassthroughTexId, blurredRT ?? passthrough);
            BlurMaterial.SetFloat(DebugRawPassthroughId, debugRawPassthrough ? 1f : 0f);
        }
        if (PixelateMaterial != null && passthrough != null)
            PixelateMaterial.SetTexture(PassthroughTexId, passthrough);
    }

    /// <summary>Enable/configure or disable the blur sphere.</summary>
    public void UpdateBlurSphere(
        bool blurActive, float intensity, bool isInverted,
        float globalAlpha, int debugMode, bool debugRawPassthrough,
        Texture2D blurMaskTexture, int texW, int texH)
    {
        if (BlurSphere == null || BlurMaterial == null) return;

        if (blurActive || debugRawPassthrough)
        {
            BlurSphere.SetActive(true);
            BlurMaterial.SetFloat(GlobalAlphaId, globalAlpha);
            BlurMaterial.SetFloat(OutOfBoundsAlphaId, isInverted ? 1f : 0f);
            BlurMaterial.SetFloat(DebugModeId, (float)debugMode);

            if (blurMaskTexture != null)
            {
                BlurMaterial.mainTexture = blurMaskTexture;
                BlurMaterial.SetVector(TextureResolutionId, new Vector4(texW, texH, 0, 0));
            }
        }
        else
        {
            BlurSphere.SetActive(false);
        }
    }

    /// <summary>Enable/configure or disable the pixelate sphere.</summary>
    public void UpdatePixelateSphere(
        bool pixelateActive, float intensity, bool isInverted,
        float globalAlpha, Texture2D pixelateMaskTexture, int texW, int texH)
    {
        if (PixelateSphere == null || PixelateMaterial == null) return;

        if (pixelateActive && intensity > 0f)
        {
            PixelateSphere.SetActive(true);
            int blockSize = Math.Max(4, (int)(32 * intensity));
            PixelateMaterial.SetFloat(PixelBlockSizeId, (float)blockSize);
            PixelateMaterial.SetFloat(GlobalAlphaId, globalAlpha);
            PixelateMaterial.SetFloat(OutOfBoundsAlphaId, isInverted ? 1f : 0f);

            if (pixelateMaskTexture != null)
            {
                PixelateMaterial.mainTexture = pixelateMaskTexture;
                PixelateMaterial.SetVector(TextureResolutionId, new Vector4(texW, texH, 0, 0));
            }
        }
        else
        {
            PixelateSphere.SetActive(false);
        }
    }

    /// <summary>Switch between AR mode (spheres) and webcam mode (no spheres).</summary>
    public void SetWebcamMode(bool webcamMode)
    {
        if (webcamMode)
        {
            if (OverlaySphere != null) OverlaySphere.SetActive(false);
            if (BlurSphere != null) BlurSphere.SetActive(false);
            if (PixelateSphere != null) PixelateSphere.SetActive(false);
        }
        else
        {
            if (OverlaySphere != null) OverlaySphere.SetActive(true);
        }
    }

    /// <summary>Disable intrinsics on blur/pixelate materials.</summary>
    public void DisableIntrinsics()
    {
        if (BlurMaterial != null) BlurMaterial.SetFloat(UseIntrinsicsId, 0f);
        if (PixelateMaterial != null) PixelateMaterial.SetFloat(UseIntrinsicsId, 0f);
    }

    public void Destroy()
    {
        _blurPipeline?.Dispose();
        if (OverlaySphere != null) UnityEngine.Object.Destroy(OverlaySphere);
        if (BlurSphere != null) UnityEngine.Object.Destroy(BlurSphere);
        if (PixelateSphere != null) UnityEngine.Object.Destroy(PixelateSphere);
        if (OverlayMaterial != null) UnityEngine.Object.Destroy(OverlayMaterial);
        if (BlurMaterial != null) UnityEngine.Object.Destroy(BlurMaterial);
        if (PixelateMaterial != null) UnityEngine.Object.Destroy(PixelateMaterial);
    }
}
