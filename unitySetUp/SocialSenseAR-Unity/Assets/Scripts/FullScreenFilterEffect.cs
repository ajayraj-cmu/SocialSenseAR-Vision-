using UnityEngine;
using System;

/// <summary>
/// Applies full-screen color filters (dim, warm, cool, night, grayscale, color)
/// driven by the voice agent's FullScreenFilter state.
/// Works on Quest/URP by rendering a full-screen quad in front of the camera.
/// </summary>
public class FullScreenFilterEffect : MonoBehaviour
{
    [Header("Shader")]
    [SerializeField] private Material filterMaterial;

    private string _filterType = "none";
    private float _intensity = 0f;
    private string _colorHex = null;  // NEW: hex color for "color" filter

    private GameObject _quad;
    private MeshRenderer _quadRenderer;
    private Material _instanceMaterial;

    // Shader property IDs
    private static readonly int FilterTypeId = Shader.PropertyToID("_FilterType");
    private static readonly int IntensityId = Shader.PropertyToID("_Intensity");
    private static readonly int FilterColorId = Shader.PropertyToID("_FilterColor");  // NEW

    void Start()
    {
        if (filterMaterial == null)
        {
            var shader = Shader.Find("SocialSense/FullScreenFilter");
            if (shader != null)
                filterMaterial = new Material(shader);
            else
            {
                Debug.LogWarning("[FullScreenFilter] Shader not found");
                return;
            }
        }

        CreateFullScreenQuad();
    }

    private void CreateFullScreenQuad()
    {
        Camera cam = GetComponent<Camera>();
        if (cam == null)
            cam = Camera.main;
        if (cam == null)
        {
            Debug.LogWarning("[FullScreenFilter] No camera found");
            return;
        }

        _quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _quad.name = "FullScreenFilterQuad";
        _quad.transform.SetParent(cam.transform, false);

        // Destroy the collider - we don't need it
        var collider = _quad.GetComponent<Collider>();
        if (collider != null)
            Destroy(collider);

        // Position just beyond the near clip plane so it covers the entire view
        float dist = cam.nearClipPlane + 0.01f;
        _quad.transform.localPosition = new Vector3(0, 0, dist);
        _quad.transform.localRotation = Quaternion.identity;

        // Scale to cover the full frustum at this distance
        float height = 2f * dist * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float width = height * cam.aspect;
        // Add a small margin to avoid edge gaps
        _quad.transform.localScale = new Vector3(width * 1.1f, height * 1.1f, 1f);

        // Set up material
        _quadRenderer = _quad.GetComponent<MeshRenderer>();
        _instanceMaterial = new Material(filterMaterial);
        _quadRenderer.material = _instanceMaterial;
        _quadRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        _quadRenderer.receiveShadows = false;

        // Start hidden
        _quad.SetActive(false);
    }

    public void SetFilter(string filterType, float intensity, string colorHex = null)
    {
        _filterType = filterType ?? "none";
        _intensity = Mathf.Clamp01(intensity);
        _colorHex = colorHex;  // NEW: store hex color
        ApplyFilter();
    }

    /// <summary>
    /// Explicitly clear/disable the filter. Useful for ensuring clean state.
    /// </summary>
    public void ClearFilter()
    {
        SetFilter("none", 0f, null);
        if (_quad != null)
        {
            _quad.SetActive(false);
        }
        Debug.Log("[FullScreenFilter] Filter explicitly cleared and disabled");
    }

    private void ApplyFilter()
    {
        if (_quad == null || _instanceMaterial == null)
            return;

        bool active = _filterType != "none" && _intensity > 0f;
        _quad.SetActive(active);

        if (active)
        {
            _instanceMaterial.SetFloat(FilterTypeId, FilterTypeToIndex(_filterType));
            _instanceMaterial.SetFloat(IntensityId, _intensity);

            // NEW: Set color for "color" filter type
            if (_filterType == "color")
            {
                Color? parsedColor = ColorUtility.ParseColorHexToColor(_colorHex);
                if (parsedColor.HasValue)
                {
                    _instanceMaterial.SetColor(FilterColorId, parsedColor.Value);
                }
                else
                {
                    Debug.LogWarning($"[FullScreenFilter] Invalid color hex: '{_colorHex}' - disabling filter to prevent white screen");
                    _quad.SetActive(false);
                }
            }
        }
    }


    private static int FilterTypeToIndex(string filterType)
    {
        switch (filterType)
        {
            case "dim": return 1;
            case "warm": return 2;
            case "cool": return 3;
            case "night": return 4;
            case "grayscale": return 5;
            case "color": return 6;  // NEW
            default: return 0;
        }
    }

    void OnDestroy()
    {
        if (_quad != null)
            Destroy(_quad);
        if (_instanceMaterial != null)
            Destroy(_instanceMaterial);
    }
}
