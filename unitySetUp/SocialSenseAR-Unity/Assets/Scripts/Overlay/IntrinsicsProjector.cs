using System;
using Meta.XR;
using Meta.XR.EnvironmentDepth;
using UnityEngine;

/// <summary>
/// Projects camera intrinsics onto overlay materials for pinhole-correct rendering.
/// Handles capture-time head rotation (world-locked reprojection), crop-adjusted
/// principal point, and depth-based stereo correction.
/// </summary>
public class IntrinsicsProjector
{
    // Cached values exposed for LabelManager and depth sampling
    public Vector2 FocalLength { get; private set; }
    public Vector2 PrincipalPoint { get; private set; }
    public float ActualW { get; private set; }
    public float ActualH { get; private set; }
    public Quaternion CameraProjectionRot { get; private set; } = Quaternion.identity;
    public Matrix4x4 CameraRotationMatrix { get; private set; } = Matrix4x4.identity;
    public Vector3 CamWorldPos { get; private set; }
    public bool IsValid { get; private set; }

    private EnvironmentDepthManager _depthManager;
    private Transform _leftEyeAnchor;
    private Transform _rightEyeAnchor;

    private float _debugLogTimer;

    // Shader property IDs
    private static readonly int UseIntrinsicsId = Shader.PropertyToID("_UseIntrinsicsProjection");
    private static readonly int FocalLengthId = Shader.PropertyToID("_FocalLength");
    private static readonly int PrincipalPointId = Shader.PropertyToID("_PrincipalPoint");
    private static readonly int SensorResolutionId = Shader.PropertyToID("_SensorResolution");
    private static readonly int TextureResolutionId = Shader.PropertyToID("_TextureResolution");
    private static readonly int CameraRotationMatrixId = Shader.PropertyToID("_CameraRotationMatrix");
    private static readonly int ProjectionOffsetId = Shader.PropertyToID("_ProjectionOffset");
    private static readonly int CameraWorldPosId = Shader.PropertyToID("_CameraWorldPos");
    private static readonly int RightEyeOffsetId = Shader.PropertyToID("_RightEyeOffset");
    private static readonly int DepthSampleUVId = Shader.PropertyToID("_DepthSampleUV");
    private static readonly int RightEyeWorldPosId = Shader.PropertyToID("_RightEyeWorldPos");
    private static readonly int UseDepthReprojectionId = Shader.PropertyToID("_UseDepthReprojection");
    private static readonly int RightEyeShiftMultiplierId = Shader.PropertyToID("_RightEyeShiftMultiplier");

    /// <summary>
    /// Initialize environment depth for stereo-correct mask reprojection.
    /// </summary>
    public void InitializeDepth(Transform centerEyeAnchor)
    {
        if (centerEyeAnchor != null && centerEyeAnchor.parent != null)
        {
            _leftEyeAnchor = centerEyeAnchor.parent.Find("LeftEyeAnchor");
            _rightEyeAnchor = centerEyeAnchor.parent.Find("RightEyeAnchor");
            if (_leftEyeAnchor != null && _rightEyeAnchor != null)
                Debug.Log("[IntrinsicsProjector] Found left/right eye anchors for stereo");
        }

        _depthManager = UnityEngine.Object.FindFirstObjectByType<EnvironmentDepthManager>();
        if (_depthManager == null && EnvironmentDepthManager.IsSupported)
        {
            var depthObj = new GameObject("EnvironmentDepthManager");
            _depthManager = depthObj.AddComponent<EnvironmentDepthManager>();
            _depthManager.OcclusionShadersMode = OcclusionShadersMode.None;
            _depthManager.RemoveHands = false;
            Debug.Log("[IntrinsicsProjector] Created EnvironmentDepthManager for depth reprojection");
        }
    }

    /// <summary>
    /// Update intrinsics projection on all provided materials.
    /// Returns false if intrinsics are unavailable.
    /// </summary>
    public bool UpdateProjection(
        PassthroughCameraAccess cameraAccess, Transform centerEyeAnchor,
        Quaternion captureRotation, bool hasCaptureRotation,
        float focalLengthScale, Vector2 projectionOffset,
        Vector2 rightEyeOffset, float rightEyeShiftMultiplier,
        Vector2 depthSampleUV, Material[] materials, int texWidth, int texHeight)
    {
        try
        {
            var intrinsics = cameraAccess.Intrinsics;
            if (intrinsics.SensorResolution.x == 0 || intrinsics.SensorResolution.y == 0)
                return false;

            // Camera rotation from LensOffset (includes 180° X flip for OpenCV→Unity)
            Pose lensOffset = intrinsics.LensOffset;
            Vector3 camForwardInHead = lensOffset.rotation * Vector3.forward;
            Vector3 camUpInHead = lensOffset.rotation * Vector3.up;
            CameraProjectionRot = Quaternion.LookRotation(camForwardInHead, camUpInHead);

            // Use capture-time head rotation for world-locked reprojection
            Quaternion headRotation = hasCaptureRotation ? captureRotation : centerEyeAnchor.rotation;
            CameraRotationMatrix = Matrix4x4.Rotate(Quaternion.Inverse(headRotation * CameraProjectionRot));

            Vector2 scaledFocal = intrinsics.FocalLength * focalLengthScale;

            // Adjust for center-crop (camera texture may be smaller than sensor)
            Texture camTex = cameraAccess.GetTexture();
            ActualW = camTex != null ? camTex.width : intrinsics.SensorResolution.x;
            ActualH = camTex != null ? camTex.height : intrinsics.SensorResolution.y;
            float sensorW = intrinsics.SensorResolution.x;
            float sensorH = intrinsics.SensorResolution.y;
            float cropX = (sensorW - ActualW) / 2f;
            float cropY = (sensorH - ActualH) / 2f;
            float adjustedCx = intrinsics.PrincipalPoint.x - cropX;
            float adjustedCy = intrinsics.PrincipalPoint.y - cropY;

            FocalLength = scaledFocal;
            PrincipalPoint = new Vector2(adjustedCx, adjustedCy);
            IsValid = true;

            // Camera world position (both eyes project from the left camera)
            CamWorldPos = centerEyeAnchor.position + headRotation * lensOffset.position;

            bool depthAvailable = _depthManager != null && _depthManager.IsDepthAvailable;
            Vector3 rightEyePos = _rightEyeAnchor != null ? _rightEyeAnchor.position : centerEyeAnchor.position;
            bool hasTexture = texWidth > 0 && texHeight > 0;

            // Set properties on all provided materials
            foreach (var mat in materials)
            {
                if (mat == null) continue;
                mat.SetMatrix(CameraRotationMatrixId, CameraRotationMatrix);
                mat.SetVector(FocalLengthId, scaledFocal);
                mat.SetVector(PrincipalPointId, PrincipalPoint);
                mat.SetVector(SensorResolutionId, new Vector4(ActualW, ActualH, 0, 0));
                mat.SetVector(ProjectionOffsetId, new Vector4(projectionOffset.x, projectionOffset.y, 0, 0));
                mat.SetVector(CameraWorldPosId, new Vector4(CamWorldPos.x, CamWorldPos.y, CamWorldPos.z, 0));
                mat.SetVector(RightEyeOffsetId, new Vector4(rightEyeOffset.x, rightEyeOffset.y, 0, 0));
                mat.SetFloat(UseDepthReprojectionId, depthAvailable ? 1f : 0f);
                mat.SetVector(DepthSampleUVId, new Vector4(depthSampleUV.x, depthSampleUV.y, 0, 0));
                mat.SetFloat(RightEyeShiftMultiplierId, rightEyeShiftMultiplier);
                if (depthAvailable)
                    mat.SetVector(RightEyeWorldPosId, new Vector4(rightEyePos.x, rightEyePos.y, rightEyePos.z, 0));
                mat.SetFloat(UseIntrinsicsId, hasTexture ? 1f : 0f);
                if (hasTexture)
                    mat.SetVector(TextureResolutionId, new Vector4(texWidth, texHeight, 0, 0));
            }

            // Debug logging
            if (Time.time - _debugLogTimer > 3f)
            {
                _debugLogTimer = Time.time;
                Vector3 lensEuler = lensOffset.rotation.eulerAngles;
                Debug.Log($"[IntrinsicsProjector] CamTex={ActualW}x{ActualH} " +
                          $"Sensor={sensorW}x{sensorH} Crop=({cropX},{cropY}) " +
                          $"PP=({adjustedCx:F1},{adjustedCy:F1}) Focal={intrinsics.FocalLength} " +
                          $"LensRot=({lensEuler.x:F1},{lensEuler.y:F1},{lensEuler.z:F1})");
            }

            return true;
        }
        catch (Exception e)
        {
            if (Time.time - _debugLogTimer > 3f)
            {
                _debugLogTimer = Time.time;
                Debug.LogWarning($"[IntrinsicsProjector] Intrinsics failed: {e.Message}");
            }
            IsValid = false;
            return false;
        }
    }

    /// <summary>Disable intrinsics on all materials (UV fallback mode).</summary>
    public void DisableIntrinsics(Material[] materials)
    {
        foreach (var mat in materials)
            if (mat != null) mat.SetFloat(UseIntrinsicsId, 0f);
    }

    /// <summary>
    /// Compute depth sample UV from average segment center using inverse pinhole projection.
    /// </summary>
    public Vector2 ComputeDepthSampleUV(float avgCenterX, float avgCenterY, Camera cam)
    {
        float px = avgCenterX * ActualW;
        float py = (1f - avgCenterY) * ActualH;
        float dx = (px - PrincipalPoint.x) / FocalLength.x;
        float dy = (py - PrincipalPoint.y) / FocalLength.y;
        Vector3 camLocalDir = new Vector3(dx, dy, 1f);

        Matrix4x4 camRotInv = CameraRotationMatrix.transpose;
        Vector3 worldDir = camRotInv.MultiplyVector(camLocalDir).normalized;
        Vector3 worldPoint = CamWorldPos + worldDir * 50f;

        if (cam != null)
        {
            Vector3 vp = cam.WorldToViewportPoint(worldPoint);
            return new Vector2(Mathf.Clamp01(vp.x), Mathf.Clamp01(vp.y));
        }
        return new Vector2(0.5f, 0.5f);
    }
}
