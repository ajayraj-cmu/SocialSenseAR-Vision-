using UnityEngine;

/// <summary>
/// Manages the flat overlay quad used in webcam mode.
/// The quad is parented to the camera and sized to match the webcam preview
/// for pixel-perfect overlay alignment (sphere UVs are equirectangular and
/// don't match a flat webcam image).
/// </summary>
public class WebcamQuadManager
{
    private GameObject _quad;
    private Renderer _renderer;
    private Material _material;

    public Material Material => _material;
    public bool IsActive => _quad != null && _quad.activeSelf;

    /// <summary>
    /// Create or re-enable the webcam overlay quad.
    /// Positioned slightly closer than the webcam preview (d=1.99) so the overlay draws on top.
    /// </summary>
    public void EnsureQuad(Camera cam)
    {
        if (_quad != null)
        {
            _quad.SetActive(true);
            return;
        }

        if (cam == null)
        {
            Debug.LogWarning("[WebcamQuadManager] No camera found for webcam overlay quad");
            return;
        }

        _quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _quad.name = "WebcamOverlayQuad";

        var col = _quad.GetComponent<Collider>();
        if (col != null) Object.Destroy(col);

        _quad.transform.SetParent(cam.transform, false);
        float d = Mathf.Max(cam.nearClipPlane + 0.15f, 1.99f);
        _quad.transform.localPosition = new Vector3(0f, 0f, d);
        _quad.transform.localRotation = Quaternion.identity;

        _renderer = _quad.GetComponent<MeshRenderer>();
        _renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        _renderer.receiveShadows = false;

        // Sprites/Default has built-in alpha blending (premultiplied)
        Shader shader = Shader.Find("Sprites/Default");
        if (shader == null) shader = Shader.Find("Unlit/Transparent");

        _material = new Material(shader);
        _material.renderQueue = 4000;
        _renderer.material = _material;

        // Start hidden — no overlay data yet, and Sprites/Default renders white with null texture
        _quad.SetActive(false);

        Debug.Log($"[WebcamQuadManager] Created webcam overlay quad (d={d:F3})");
    }

    /// <summary>
    /// Sizes the quad to match the webcam preview, preserving aspect ratio.
    /// Uses the same aspect-correction logic as SocialSenseClient.UpdateEditorPreviewQuadTransform().
    /// </summary>
    public void UpdateSize(Camera cam, float webcamAspect)
    {
        if (_quad == null || cam == null) return;

        float d = _quad.transform.localPosition.z;
        float frustumH = 2f * Mathf.Tan(cam.fieldOfView * Mathf.Deg2Rad * 0.5f) * d;
        float frustumW = frustumH * cam.aspect;

        float w, h;
        if (webcamAspect >= cam.aspect)
        {
            w = frustumW;
            h = w / webcamAspect;
        }
        else
        {
            h = frustumH;
            w = h * webcamAspect;
        }

        _quad.transform.localScale = new Vector3(w, h, 1f);
    }

    public void SetActive(bool active)
    {
        if (_quad != null) _quad.SetActive(active);
    }

    public void Destroy()
    {
        if (_quad != null) Object.Destroy(_quad);
        if (_material != null) Object.Destroy(_material);
        _quad = null;
        _material = null;
    }
}
