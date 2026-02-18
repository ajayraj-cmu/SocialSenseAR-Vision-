using System.Collections.Generic;
using Google.Protobuf.Collections;
using Socialsense;
using UnityEngine;
using Meta.XR.EnvironmentDepth;

/// <summary>
/// Wall-Anchor mode v2 — clean 2.5D object-shaped sticker anchored in world space.
///
/// Improvements over v1
/// ====================
///  • Live texture update every frame so the overlay always shows the freshest mask.
///  • Silhouette mask texture (_MaskTex) passed to shader — only the exact object
///    shape is rendered, not the full bounding-box rectangle.
///  • Smooth position / size lerp (lerpSpeed) damps depth-estimate jitter so the
///    quad never snaps or glitches.
///  • ZTest Always — the sticker always renders on top of real geometry, giving the
///    "2.5D sticker plastered on the wall" look.
///  • Quad normal always faces the current camera (billboard constraint on Up axis
///    only, so it stays level) for best parallax behaviour.
///  • First-frame anchor is held for stabilisationFrames before being shown, so the
///    depth estimate has time to converge.
/// </summary>
public class WallAnchorManager : MonoBehaviour
{
    [Tooltip("Default distance (metres) from camera when environment depth is unavailable.")]
    public float defaultAnchorDepth = 1.5f;

    [Tooltip("Layer mask for Physics.Raycast wall detection (fallback, AR floor/wall colliders).")]
    public LayerMask wallLayerMask = ~0;

    [Tooltip("Shader to use for anchored overlay quads.")]
    public Shader anchorShader;

    [Tooltip("How quickly the quad position and size lerp toward the live estimate. " +
             "Higher = snappier, lower = smoother. 6 is a good default.")]
    [Range(1f, 20f)]
    public float lerpSpeed = 6f;

    [Tooltip("Number of frames to accumulate depth samples before showing the anchor. " +
             "Reduces first-frame pop when depth is noisy.")]
    [Range(1, 10)]
    public int stabilisationFrames = 3;

    [Tooltip("If true the quad always soft-billboards toward the camera (yaw only). " +
             "Gives a flat-sticker look even on angled walls.")]
    public bool softBillboard = true;

    // -----------------------------------------------------------------------
    // Per-anchor runtime state
    // -----------------------------------------------------------------------
    private class AnchorState
    {
        public GameObject     go;
        public MeshRenderer   rend;
        public Material       mat;
        public Texture2D      overlayTex;   // colour+alpha crop, updated each frame
        public Texture2D      maskTex;      // silhouette (R8), updated each frame

        // Smoothed world-space transform targets
        public Vector3    targetPos;
        public Quaternion targetRot;
        public Vector2    targetSize;

        // Current smoothed values (what the quad actually uses)
        public Vector3    currentPos;
        public Quaternion currentRot;
        public Vector2    currentSize;

        public int stabilisationCount; // frames accumulated so far
        public bool visible;
    }

    private readonly Dictionary<string, AnchorState> _anchors =
        new Dictionary<string, AnchorState>();

    private EnvironmentDepthManager _depthManager;

    // Shader property IDs
    private static readonly int OverlayTexId = Shader.PropertyToID("_MainTex");
    private static readonly int MaskTexId    = Shader.PropertyToID("_MaskTex");
    private static readonly int GlobalAlphaId= Shader.PropertyToID("_GlobalAlpha");

    private void Awake()
    {
        _depthManager = FindFirstObjectByType<EnvironmentDepthManager>();

        if (anchorShader == null)
            anchorShader = Shader.Find("SocialSense/WallAnchorOverlay");
        if (anchorShader == null)
            anchorShader = Shader.Find("Unlit/Transparent");
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// <summary>
    /// Called every frame that new segments arrive from the server.
    /// Creates new anchors for unseen track-IDs; updates existing ones.
    /// </summary>
    public void TryAnchorSegments(
        RepeatedField<SceneSegment> segments,
        IntrinsicsProjector projector,
        Transform centerEyeAnchor,
        Quaternion headRotation,
        Texture2D overlayTexture)
    {
        if (projector == null || !projector.IsValid || centerEyeAnchor == null)
            return;

        int maskW = segments.Count > 0 ? (int)segments[0].MaskWidth  : 0;
        int maskH = segments.Count > 0 ? (int)segments[0].MaskHeight : 0;
        if (maskW == 0 || maskH == 0) return;

        foreach (var seg in segments)
        {
            if (string.IsNullOrEmpty(seg.TrackId)) continue;
            if (seg.RleMask == null || seg.RleMask.IsEmpty) continue;
            if (seg.CenterX <= 0f || seg.CenterY <= 0f) continue;
            if (seg.Bbox == null) continue;

            // ---- Compute live world pose ----
            if (!ComputeAnchorPose(seg, projector, centerEyeAnchor, headRotation,
                                   maskW, maskH, out Vector3 anchorPos, out Quaternion anchorRot))
                continue;

            float depth = Vector3.Distance(centerEyeAnchor.position, anchorPos);
            Vector2 quadSize = ComputeQuadSize(seg.Bbox, projector, depth, maskW, maskH);

            // ---- Decode mask for this segment ----
            byte[] maskBytes = DecodeRleMask(seg, maskW, maskH);

            if (!_anchors.TryGetValue(seg.TrackId, out AnchorState state))
            {
                // First time seeing this track-ID: create the anchor
                state = CreateAnchorState(seg, anchorPos, anchorRot, quadSize, overlayTexture, maskBytes, maskW, maskH);
                _anchors[seg.TrackId] = state;
            }
            else
            {
                // Subsequent frames: update target transform + refresh textures
                state.targetPos  = anchorPos;
                state.targetRot  = anchorRot;
                state.targetSize = quadSize;

                UpdateTextures(state, seg.Bbox, overlayTexture, maskBytes, maskW, maskH);

                // Count up stabilisation frames; reveal once stable
                if (!state.visible)
                {
                    state.stabilisationCount++;
                    if (state.stabilisationCount >= stabilisationFrames)
                    {
                        state.visible = true;
                        state.go.SetActive(true);
                        // Snap to target on first reveal to avoid slide-in
                        state.currentPos  = state.targetPos;
                        state.currentRot  = state.targetRot;
                        state.currentSize = state.targetSize;
                    }
                }
            }
        }

        // ---- Smooth all live anchors every frame ----
        SmoothAllAnchors(centerEyeAnchor);
    }

    /// <summary>Destroy all anchored quads and reset state.</summary>
    public void ClearAnchors()
    {
        foreach (var kv in _anchors)
        {
            if (kv.Value.go != null) Destroy(kv.Value.go);
            SafeDestroy(kv.Value.overlayTex);
            SafeDestroy(kv.Value.maskTex);
        }
        _anchors.Clear();
    }

    /// <summary>Remove a single anchor by track-ID.</summary>
    public void RemoveAnchor(string trackId)
    {
        if (_anchors.TryGetValue(trackId, out var state))
        {
            if (state.go != null) Destroy(state.go);
            SafeDestroy(state.overlayTex);
            SafeDestroy(state.maskTex);
            _anchors.Remove(trackId);
        }
    }

    // -----------------------------------------------------------------------
    // Smoothing
    // -----------------------------------------------------------------------

    private void SmoothAllAnchors(Transform centerEyeAnchor)
    {
        float t = Time.deltaTime * lerpSpeed;
        foreach (var kv in _anchors)
        {
            var state = kv.Value;
            if (!state.visible) continue;

            // Smooth position and size
            state.currentPos  = Vector3.Lerp(state.currentPos,  state.targetPos,  t);
            state.currentSize = Vector2.Lerp(state.currentSize, state.targetSize, t);

            // Smooth rotation (slerp)
            Quaternion targetRot = state.targetRot;
            if (softBillboard)
            {
                // Yaw-only billboard: face the camera but keep world up
                Vector3 toCam = (centerEyeAnchor.position - state.currentPos).normalized;
                toCam.y = 0f;
                if (toCam.sqrMagnitude > 0.001f)
                    targetRot = Quaternion.LookRotation(-toCam, Vector3.up);
            }
            state.currentRot = Quaternion.Slerp(state.currentRot, targetRot, t);

            // Apply to transform
            var tf = state.go.transform;
            tf.position   = state.currentPos;
            tf.rotation   = state.currentRot;
            tf.localScale = new Vector3(state.currentSize.x, state.currentSize.y, 1f);
        }
    }

    // -----------------------------------------------------------------------
    // Anchor creation
    // -----------------------------------------------------------------------

    private AnchorState CreateAnchorState(
        SceneSegment seg,
        Vector3 anchorPos, Quaternion anchorRot, Vector2 quadSize,
        Texture2D overlayTexture, byte[] maskBytes,
        int maskW, int maskH)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = $"WallAnchor_{seg.TrackId}_{seg.Label}";
        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);

        // Start hidden until stabilised
        go.SetActive(false);

        var mat = new Material(anchorShader);
        mat.SetFloat(GlobalAlphaId, 1f);

        var rend = go.GetComponent<MeshRenderer>();
        rend.material = mat;
        rend.shadowCastingMode  = UnityEngine.Rendering.ShadowCastingMode.Off;
        rend.receiveShadows = false;

        var state = new AnchorState
        {
            go            = go,
            rend          = rend,
            mat           = mat,
            targetPos     = anchorPos,
            targetRot     = anchorRot,
            targetSize    = quadSize,
            currentPos    = anchorPos,
            currentRot    = anchorRot,
            currentSize   = quadSize,
            stabilisationCount = 1,
            visible       = false,
        };

        // Create initial textures
        state.overlayTex = BakeOverlayCrop(seg.Bbox, overlayTexture, maskW, maskH);
        state.maskTex    = BakeSilhouetteMask(maskBytes, seg.Bbox, maskW, maskH);
        mat.SetTexture(OverlayTexId, state.overlayTex);
        mat.SetTexture(MaskTexId,    state.maskTex);

        Debug.Log($"[WallAnchorManager] Created anchor '{seg.Label}' (track={seg.TrackId}) " +
                  $"at {anchorPos:F2}, size={quadSize:F2}m");
        return state;
    }

    // -----------------------------------------------------------------------
    // Live texture update
    // -----------------------------------------------------------------------

    private void UpdateTextures(
        AnchorState state,
        BoundingBox bbox,
        Texture2D overlayTexture,
        byte[] maskBytes,
        int maskW, int maskH)
    {
        // Overlay colour crop ---------------------------------------------------
        if (overlayTexture != null)
        {
            int x0 = Mathf.Clamp(Mathf.FloorToInt(bbox.XMin * maskW), 0, maskW - 1);
            int y0 = Mathf.Clamp(Mathf.FloorToInt(bbox.YMin * maskH), 0, maskH - 1);
            int x1 = Mathf.Clamp(Mathf.CeilToInt(bbox.XMax  * maskW), 0, maskW);
            int y1 = Mathf.Clamp(Mathf.CeilToInt(bbox.YMax  * maskH), 0, maskH);
            int cropW = Mathf.Max(1, x1 - x0);
            int cropH = Mathf.Max(1, y1 - y0);

            // Resize if dimensions changed
            if (state.overlayTex == null ||
                state.overlayTex.width != cropW || state.overlayTex.height != cropH)
            {
                SafeDestroy(state.overlayTex);
                state.overlayTex = new Texture2D(cropW, cropH, TextureFormat.RGBA32, false);
                state.overlayTex.filterMode = FilterMode.Bilinear;
                state.overlayTex.wrapMode   = TextureWrapMode.Clamp;
                state.mat.SetTexture(OverlayTexId, state.overlayTex);
            }

            int flippedY0 = Mathf.Clamp(maskH - y1, 0, maskH - 1);
            try
            {
                Color[] pixels = overlayTexture.GetPixels(x0, flippedY0, cropW, cropH);
                state.overlayTex.SetPixels(pixels);
                state.overlayTex.Apply(false);
            }
            catch { /* source texture not readable in this frame — skip */ }
        }

        // Silhouette mask -------------------------------------------------------
        if (maskBytes != null && maskBytes.Length == maskW * maskH)
        {
            int x0 = Mathf.Clamp(Mathf.FloorToInt(bbox.XMin * maskW), 0, maskW - 1);
            int y0 = Mathf.Clamp(Mathf.FloorToInt(bbox.YMin * maskH), 0, maskH - 1);
            int x1 = Mathf.Clamp(Mathf.CeilToInt(bbox.XMax  * maskW), 0, maskW);
            int y1 = Mathf.Clamp(Mathf.CeilToInt(bbox.YMax  * maskH), 0, maskH);
            int cropW = Mathf.Max(1, x1 - x0);
            int cropH = Mathf.Max(1, y1 - y0);

            if (state.maskTex == null ||
                state.maskTex.width != cropW || state.maskTex.height != cropH)
            {
                SafeDestroy(state.maskTex);
                state.maskTex = new Texture2D(cropW, cropH, TextureFormat.R8, false);
                state.maskTex.filterMode = FilterMode.Bilinear;
                state.maskTex.wrapMode   = TextureWrapMode.Clamp;
                state.mat.SetTexture(MaskTexId, state.maskTex);
            }

            // Copy the crop region (flip Y: mask is top-left, Unity is bottom-left)
            Color[] maskPixels = new Color[cropW * cropH];
            for (int row = 0; row < cropH; row++)
            {
                int srcRow = (maskH - 1) - (y0 + row); // flip
                srcRow = Mathf.Clamp(srcRow, 0, maskH - 1);
                for (int col = 0; col < cropW; col++)
                {
                    int srcCol = x0 + col;
                    int srcIdx = srcRow * maskW + srcCol;
                    float v = maskBytes[srcIdx] > 0 ? 1f : 0f;
                    maskPixels[row * cropW + col] = new Color(v, v, v, v);
                }
            }
            state.maskTex.SetPixels(maskPixels);
            state.maskTex.Apply(false);
        }
    }

    // -----------------------------------------------------------------------
    // Pose computation
    // -----------------------------------------------------------------------

    private bool ComputeAnchorPose(
        SceneSegment seg,
        IntrinsicsProjector proj,
        Transform centerEyeAnchor,
        Quaternion headRot,
        int maskW, int maskH,
        out Vector3 worldPos,
        out Quaternion worldRot)
    {
        worldPos = Vector3.zero;
        worldRot = Quaternion.identity;

        // Pixel coordinates (OpenCV: CenterY 0 = top → flip for Unity)
        float px = seg.CenterX * proj.ActualW;
        float py = (1f - seg.CenterY) * proj.ActualH;

        // Inverse pinhole
        float dx = (px - proj.PrincipalPoint.x) / proj.FocalLength.x;
        float dy = (py - proj.PrincipalPoint.y) / proj.FocalLength.y;
        Vector3 localDir = new Vector3(dx, dy, 1f).normalized;
        Vector3 worldDir = (headRot * proj.CameraProjectionRot) * localDir;

        float depth = defaultAnchorDepth;
        bool gotDepth = false;

        // 1. Environment depth API
        if (_depthManager != null && _depthManager.IsDepthAvailable)
        {
            Vector2 depthUV = proj.ComputeDepthSampleUV(
                seg.CenterX, seg.CenterY,
                centerEyeAnchor.GetComponent<Camera>() ?? Camera.main);
            float envDepth = SampleEnvironmentDepth(depthUV);
            if (envDepth > 0.1f && envDepth < 30f)
            { depth = envDepth; gotDepth = true; }
        }

        // 2. Physics raycast
        if (!gotDepth)
        {
            if (Physics.Raycast(proj.CamWorldPos, worldDir, out var hit, 10f, wallLayerMask))
            { depth = hit.distance; gotDepth = true; }
        }

        worldPos = proj.CamWorldPos + worldDir * depth;

        Vector3 toCamera = (proj.CamWorldPos - worldPos).normalized;
        if (toCamera == Vector3.zero) return false;
        worldRot = Quaternion.LookRotation(-toCamera, Vector3.up);
        return true;
    }

    private Vector2 ComputeQuadSize(BoundingBox bbox, IntrinsicsProjector proj, float depth, int maskW, int maskH)
    {
        float pixW = (bbox.XMax - bbox.XMin) * proj.ActualW;
        float pixH = (bbox.YMax - bbox.YMin) * proj.ActualH;
        float worldW = Mathf.Clamp((pixW / proj.FocalLength.x) * depth, 0.05f, 5f);
        float worldH = Mathf.Clamp((pixH / proj.FocalLength.y) * depth, 0.05f, 5f);
        return new Vector2(worldW, worldH);
    }

    // -----------------------------------------------------------------------
    // Texture baking helpers (first-frame only)
    // -----------------------------------------------------------------------

    private static Texture2D BakeOverlayCrop(BoundingBox bbox, Texture2D src, int maskW, int maskH)
    {
        int x0 = Mathf.Clamp(Mathf.FloorToInt(bbox.XMin * maskW), 0, maskW - 1);
        int y0 = Mathf.Clamp(Mathf.FloorToInt(bbox.YMin * maskH), 0, maskH - 1);
        int x1 = Mathf.Clamp(Mathf.CeilToInt(bbox.XMax  * maskW), 0, maskW);
        int y1 = Mathf.Clamp(Mathf.CeilToInt(bbox.YMax  * maskH), 0, maskH);
        int cropW = Mathf.Max(1, x1 - x0);
        int cropH = Mathf.Max(1, y1 - y0);

        var tex = new Texture2D(cropW, cropH, TextureFormat.RGBA32, false);
        tex.filterMode = FilterMode.Bilinear;
        tex.wrapMode   = TextureWrapMode.Clamp;

        if (src != null)
        {
            int flippedY0 = Mathf.Clamp(maskH - y1, 0, maskH - 1);
            try
            {
                Color[] pixels = src.GetPixels(x0, flippedY0, cropW, cropH);
                tex.SetPixels(pixels);
            }
            catch { }
        }
        tex.Apply(false);
        return tex;
    }

    private static Texture2D BakeSilhouetteMask(byte[] maskBytes, BoundingBox bbox, int maskW, int maskH)
    {
        int x0 = Mathf.Clamp(Mathf.FloorToInt(bbox.XMin * maskW), 0, maskW - 1);
        int y0 = Mathf.Clamp(Mathf.FloorToInt(bbox.YMin * maskH), 0, maskH - 1);
        int x1 = Mathf.Clamp(Mathf.CeilToInt(bbox.XMax  * maskW), 0, maskW);
        int y1 = Mathf.Clamp(Mathf.CeilToInt(bbox.YMax  * maskH), 0, maskH);
        int cropW = Mathf.Max(1, x1 - x0);
        int cropH = Mathf.Max(1, y1 - y0);

        var tex = new Texture2D(cropW, cropH, TextureFormat.R8, false);
        tex.filterMode = FilterMode.Bilinear;
        tex.wrapMode   = TextureWrapMode.Clamp;

        if (maskBytes != null && maskBytes.Length == maskW * maskH)
        {
            Color[] pixels = new Color[cropW * cropH];
            for (int row = 0; row < cropH; row++)
            {
                int srcRow = Mathf.Clamp((maskH - 1) - (y0 + row), 0, maskH - 1);
                for (int col = 0; col < cropW; col++)
                {
                    float v = maskBytes[srcRow * maskW + x0 + col] > 0 ? 1f : 0f;
                    pixels[row * cropW + col] = new Color(v, v, v, v);
                }
            }
            tex.SetPixels(pixels);
        }
        tex.Apply(false);
        return tex;
    }

    // -----------------------------------------------------------------------
    // RLE decode
    // -----------------------------------------------------------------------

    /// <summary>
    /// Decode the server's uint16 little-endian RLE mask into a flat byte[] where
    /// 1 = object pixel, 0 = background pixel.
    /// Alternating runs: first run is background, then foreground, etc.
    /// </summary>
    private static byte[] DecodeRleMask(SceneSegment seg, int maskW, int maskH)
    {
        int total = maskW * maskH;
        byte[] result = new byte[total];

        var rle = seg.RleMask;
        if (rle == null || rle.IsEmpty) return result;

        // RleMask is a ByteString of packed uint16 LE values
        byte[] raw = rle.ToByteArray();
        int pos = 0;
        bool isFg = false; // first run is always background

        for (int i = 0; i + 1 < raw.Length && pos < total; i += 2)
        {
            int runLen = raw[i] | (raw[i + 1] << 8);
            if (isFg)
            {
                for (int j = 0; j < runLen && pos < total; j++)
                    result[pos++] = 1;
            }
            else
            {
                pos += runLen; // background: leave as 0
            }
            isFg = !isFg;
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------

    private static float SampleEnvironmentDepth(Vector2 uv)
    {
        // CPU-side depth readback requires async — use 0 to fall through to Physics raycast.
        return 0f;
    }

    private static void SafeDestroy(UnityEngine.Object obj)
    {
        if (obj != null) Destroy(obj);
    }

    private void OnDestroy()
    {
        ClearAnchors();
    }
}
