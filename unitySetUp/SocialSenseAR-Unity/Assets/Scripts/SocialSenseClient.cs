using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Android;
using UnityEngine.Rendering;
using NativeWebSocket;
using Google.Protobuf;
using Socialsense;
using Meta.XR;

/// <summary>
/// SocialSenseAR client: captures single-eye passthrough frames, sends over WebSocket
/// as protobuf, receives SAM segmentation results, and drives the overlay renderer.
///
/// Native passthrough stays ENABLED — overlay renders on top.
///
/// Camera capture reuses patterns from StereoPipelineClient (async GPU readback, JPEG).
///
/// INPUT MODE: Toggle between AR passthrough (Quest headset) and Webcam at runtime.
/// Both modes produce identical frame data for the server pipeline.
/// </summary>
public class SocialSenseClient : MonoBehaviour
{
    /// <summary>
    /// Runtime-switchable input mode. Change via Inspector or code at any time.
    /// AR: Uses PassthroughCameraAccess (Quest headset). Full intrinsics projection + real blur.
    /// Webcam: Uses WebCamTexture. UV-based overlay projection + overlay blur fallback.
    /// Auto: Webcam (default; avoids Quest passthrough/ReadPixels issues).
    /// </summary>
    public enum InputMode { Auto, AR, Webcam }

    [Header("Input Mode")]
    [Tooltip("Toggle between AR passthrough (headset) and Webcam at runtime. Auto = Webcam (default).")]
    public InputMode inputMode = InputMode.Webcam;

    [Header("Camera")]
    [Tooltip("PassthroughCameraAccess for a single eye (left recommended). Required for AR mode.")]
    public PassthroughCameraAccess cameraAccess;

    [Header("Network")]
    [Tooltip("Full WebSocket URL. Use wss:// for Modal, ws:// for local.\nModal example: wss://yourname--socialsense-ar-gpu.modal.run/ws\nLocal example: ws://127.0.0.1:8765")]
    public string serverUrl = "wss://ajraj2006--socialsense-ar-gpu-socialsensegpu-web.modal.run/ws";

    [Header("Performance")]
    [Range(5, 60)]
    public float targetFPS = 30f;

    [Range(30, 95)]
    public int jpegQuality = 70;

    [Range(0.25f, 1.0f)]
    public float resolutionScale = 0.5f;

    [Header("Overlay")]
    public OverlayRenderer overlayRenderer;

    [Header("Voice Agent")]
    [SerializeField] private AudioStreamer audioStreamer;
    [SerializeField] private VoiceAgentHUD voiceAgentHUD;
    [SerializeField] private FullScreenFilterEffect fullScreenFilter;
    [SerializeField] private bool enableVoiceAgent = true;

    [Header("Conversation Mode")]
    [Tooltip("ConversationModeEffect component for the cinematic greyscale/outline animation.")]
    [SerializeField] private ConversationModeEffect conversationModeEffect;
    private bool _conversationModeActive = false;

    [Header("Editor Testing")]
    public bool useWebcamInEditor = true;

    [Header("Webcam Preview")]
    [Tooltip("Draw webcam as a background quad in the Game view when in Webcam mode.")]
    [SerializeField] private bool enableWebcamPreview = true;
    [SerializeField] private float webcamPreviewDistance = 2.0f;
    [SerializeField] private bool showCommandHUD = true;
    [SerializeField] private string editorCommand = "blur person";
    [Tooltip("Optional horizontal mirror for front-facing webcams (preview only).")]
    [SerializeField] private bool editorMirrorHorizontal = false;

    private GameObject _editorPreviewQuad;
    private MeshRenderer _editorPreviewQuadRenderer;
    private Material _editorPreviewMaterial;
    private Camera _editorPreviewCamera;

    [Header("Debug")]
    public bool showDebugInfo = true;
    public UnityEngine.UI.Text debugText;

    // WebSocket
    private WebSocket _websocket;
    private bool _connected = false;
    private bool _connecting = false;
    private float _lastConnectAttempt = 0f;
    private int _connectCount = 0;
    private string _lastWsError = "";
    private int _lastWsCloseCode = 0;

    // Camera / Input Mode
    private bool _isWebcamMode = false;  // True when using webcam, false when using AR passthrough
    private bool _webcamInitialized = false;
    private bool _arInitialized = false;
    private InputMode _activeMode = InputMode.Auto;  // Currently active resolved mode
    private InputMode _lastCheckedInputMode = InputMode.Auto;  // For detecting Inspector changes at runtime
    private WebCamTexture _webcamTexture;
    private const string QUEST_CAMERA_PERMISSION = "horizonos.permission.HEADSET_CAMERA";

    // Constants
    private const float RECONNECT_DELAY_SECONDS = 3f;
    private const int MAX_READBACKS_IN_FLIGHT = 2;
    private const int MAX_STORED_ROTATIONS = 60;
    private const int DEFAULT_WEBCAM_WIDTH = 640;
    private const int DEFAULT_WEBCAM_HEIGHT = 480;
    private const int DEFAULT_WEBCAM_FPS = 30;
    private const float CAMERA_NEAR_CLIP_OFFSET = 0.2f;
    private const int GUI_PADDING = 10;
    private const int GUI_WIDTH = 560;
    private const int GUI_LINE_HEIGHT = 22;
    private const int GUI_BUTTON_WIDTH = 110;
    private const int LABEL_FONT_SIZE = 18;
    private const int HUD_FONT_SIZE = 13;
    private const int CONFIDENCE_FONT_SIZE = 12;
    private const float LABEL_BACKGROUND_ALPHA = 0.7f;
    private const float STATS_UPDATE_INTERVAL = 2f;
    private const float DEBUG_LOG_INTERVAL = 3f;

    // Legacy compat: _isEditorMode still works as alias for webcam mode
    private bool _isEditorMode { get { return _isWebcamMode; } }

    // GPU capture pipeline
    private RenderTexture _scaledRT;
    private RenderTexture _flippedRT;
    private Texture2D _readbackTexture;
    private int _scaledW, _scaledH;
    private int _readbacksInFlight = 0;

    // Frame timing
    private float _lastSendTime;
    private ulong _frameId = 0;
    private int _framesSent = 0;
    private int _framesReceived = 0;
    private float _statsTime;

    // Latest server response (set from OnMessage, read from Update)
    private volatile byte[] _pendingResponse;
    private ServerMessage _latestResponse;

    // Capture-time pose reprojection: store head rotation at capture time
    // so the overlay can be world-locked instead of head-locked
    private Dictionary<ulong, Quaternion> _captureRotations = new Dictionary<ulong, Quaternion>();

    // Last masks_frame_id to detect stale repeated responses
    private ulong _lastMasksFrameId = 0;

    void Start()
    {
        Debug.Log("[SocialSense] Starting...");

        // Validate required components
        ValidateComponents();

        // Clear all effects on startup (overlay, filters, conversation mode)
        ClearAllEffects();
        Debug.Log("[SocialSense] Cleared all effects on startup");

        // Resolve input mode
        ResolveInputMode();

        StartCoroutine(ConnectWhenReady());
    }

    /// <summary>
    /// Resolve the effective input mode from the InputMode enum.
    /// Can be called at runtime to switch modes dynamically.
    /// </summary>
    private void ResolveInputMode()
    {
        InputMode resolvedMode = inputMode;

        // Auto-resolve: Webcam by default (avoids Quest passthrough/ReadPixels issues). AR only when explicitly selected.
        if (resolvedMode == InputMode.Auto)
        {
            resolvedMode = InputMode.Webcam;
        }

        bool wasWebcam = _isWebcamMode;
        _isWebcamMode = (resolvedMode == InputMode.Webcam);
        _activeMode = resolvedMode;
        _lastCheckedInputMode = inputMode;

        Debug.Log($"[SocialSense] Input mode resolved: {inputMode} → {resolvedMode} (webcam={_isWebcamMode})");

        if (_isWebcamMode)
        {
            InitializeWebcam();
        }
        else
        {
            InitializeAR();
        }

        // Notify OverlayRenderer of the mode so it can adapt rendering
        if (overlayRenderer != null)
        {
            overlayRenderer.SetWebcamMode(_isWebcamMode);
        }
    }

    /// <summary>
    /// Initialize webcam capture. Can be called at any time.
    /// </summary>
    private void InitializeWebcam()
    {
        if (_webcamInitialized) return;

        if (WebCamTexture.devices.Length > 0)
        {
            if (_webcamTexture == null)
            {
                _webcamTexture = new WebCamTexture(DEFAULT_WEBCAM_WIDTH, DEFAULT_WEBCAM_HEIGHT, DEFAULT_WEBCAM_FPS);
            }
            if (!_webcamTexture.isPlaying)
            {
                _webcamTexture.Play();
            }
            _webcamInitialized = true;
            Debug.Log($"[SocialSense] Webcam initialized: {_webcamTexture.deviceName}");

            if (enableWebcamPreview)
            {
                EnsureEditorPreviewCamera();
                EnsureEditorPreviewQuad();
            }
        }
        else
        {
            Debug.LogError("[SocialSense] No webcam found — cannot use webcam mode");
        }
    }

    /// <summary>
    /// Initialize AR passthrough capture. Can be called at any time.
    /// </summary>
    private void InitializeAR()
    {
        if (_arInitialized) return;

#if UNITY_ANDROID && !UNITY_EDITOR
        if (!Permission.HasUserAuthorizedPermission(QUEST_CAMERA_PERMISSION))
        {
            Debug.Log("[SocialSense] Requesting camera permission...");
            Permission.RequestUserPermission(QUEST_CAMERA_PERMISSION);
        }

        // Request mic permission for voice agent
        if (enableVoiceAgent && !Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            Debug.Log("[SocialSense] Requesting microphone permission...");
            Permission.RequestUserPermission(Permission.Microphone);
        }
#endif

        if (cameraAccess == null)
        {
            Debug.LogWarning("[SocialSense] cameraAccess is null — AR mode may not work. Falling back to webcam.");
            _isWebcamMode = true;
            _activeMode = InputMode.Webcam;
            InitializeWebcam();
            return;
        }

        // Activate PassthroughCameraAccess GameObject (starts inactive when defaulting to webcam)
        cameraAccess.gameObject.SetActive(true);
        _arInitialized = true;
        Debug.Log("[SocialSense] AR passthrough initialized");
    }

    /// <summary>
    /// Call from code or UI to switch input mode at runtime.
    /// </summary>
    public void SwitchInputMode(InputMode newMode)
    {
        inputMode = newMode;
        _webcamInitialized = false;
        _arInitialized = false;
        ResolveInputMode();
        Debug.Log($"[SocialSense] Switched input mode to {newMode} → active={_activeMode}");
    }

    /// <summary>
    /// Returns true if currently using webcam mode.
    /// </summary>
    public bool IsWebcamMode => _isWebcamMode;

    /// <summary>
    /// Returns the currently active resolved input mode.
    /// </summary>
    public InputMode ActiveMode => _activeMode;

    /// <summary>
    /// Validate that required components are assigned in the Inspector.
    /// Logs warnings for missing optional components.
    /// </summary>
    private void ValidateComponents()
    {
        // Critical components
        if (overlayRenderer == null)
        {
            Debug.LogError("[SocialSense] OverlayRenderer not assigned! Overlay effects will not work.");
        }

        // Mode-specific components
        if (!_isWebcamMode && cameraAccess == null)
        {
            Debug.LogWarning("[SocialSense] PassthroughCameraAccess not assigned! AR mode will not work. Falling back to webcam.");
        }

        // Optional but recommended components
        if (enableVoiceAgent)
        {
            if (audioStreamer == null)
            {
                Debug.LogWarning("[SocialSense] Voice agent enabled but AudioStreamer not assigned! Voice commands will not work.");
            }
            if (voiceAgentHUD == null)
            {
                Debug.LogWarning("[SocialSense] VoiceAgentHUD not assigned! Voice feedback UI will not be shown.");
            }
        }

        if (fullScreenFilter == null)
        {
            Debug.LogWarning("[SocialSense] FullScreenFilterEffect not assigned! Full-screen filters will not work.");
        }
    }

    // ----------------------------------------------------------------
    // Webcam preview (works in both Editor and device for webcam mode)
    // ----------------------------------------------------------------

    private void UpdateEditorPreviewWebcamUV()
    {
        if (_editorPreviewMaterial == null || _webcamTexture == null) return;

        // WebCamTexture can be vertically mirrored and rotated depending on device/OS.
        // We correct the preview so "upright" matches what you expect to see in the Game view.
        float scaleX = editorMirrorHorizontal ? -1f : 1f;
        float scaleY = _webcamTexture.videoVerticallyMirrored ? -1f : 1f;
        _editorPreviewMaterial.mainTextureScale = new Vector2(scaleX, scaleY);
        _editorPreviewMaterial.mainTextureOffset = new Vector2(scaleX < 0 ? 1f : 0f, scaleY < 0 ? 1f : 0f);
    }

    private void EnsureEditorPreviewCamera()
    {
        if (_editorPreviewCamera != null) return;

        // Prefer the camera under the center eye anchor (XR rig), else fall back to Camera.main.
        if (overlayRenderer != null && overlayRenderer.centerEyeAnchor != null)
            _editorPreviewCamera = overlayRenderer.centerEyeAnchor.GetComponentInChildren<Camera>();
        if (_editorPreviewCamera == null)
            _editorPreviewCamera = Camera.main;

        if (_editorPreviewCamera == null)
            Debug.LogWarning("[SocialSense] WebcamPreview: could not find a Camera for preview quad.");
    }

    private void EnsureEditorPreviewQuad()
    {
        if (_editorPreviewQuad != null) return;
        if (_editorPreviewCamera == null) return;

        _editorPreviewQuad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _editorPreviewQuad.name = "WebcamPreview_Quad";
        _editorPreviewQuad.transform.SetParent(_editorPreviewCamera.transform, false);

        float d = Mathf.Max(_editorPreviewCamera.nearClipPlane + 0.2f, webcamPreviewDistance);
        _editorPreviewQuad.transform.localPosition = new Vector3(0f, 0f, d);
        _editorPreviewQuad.transform.localRotation = Quaternion.identity;

        // Remove collider (not needed)
        var col = _editorPreviewQuad.GetComponent<Collider>();
        if (col != null) Destroy(col);

        _editorPreviewQuadRenderer = _editorPreviewQuad.GetComponent<MeshRenderer>();
        _editorPreviewQuadRenderer.shadowCastingMode = ShadowCastingMode.Off;
        _editorPreviewQuadRenderer.receiveShadows = false;

        Shader shader = Shader.Find("Unlit/Texture");
        if (shader == null) shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) shader = Shader.Find("Sprites/Default");

        _editorPreviewMaterial = new Material(shader);
        _editorPreviewMaterial.name = "WebcamPreview_Mat (Runtime)";
        _editorPreviewQuadRenderer.material = _editorPreviewMaterial;

        UpdateEditorPreviewQuadTransform();
    }

    private void UpdateEditorPreviewQuadTransform()
    {
        if (_editorPreviewQuad == null || _editorPreviewCamera == null) return;

        float d = Mathf.Max(_editorPreviewCamera.nearClipPlane + 0.2f, webcamPreviewDistance);
        _editorPreviewQuad.transform.localPosition = new Vector3(0f, 0f, d);

        // Apply device-reported rotation so the preview looks correct.
        if (_webcamTexture != null)
        {
            _editorPreviewQuad.transform.localRotation = Quaternion.Euler(0f, 0f, -_webcamTexture.videoRotationAngle);
        }
        else
        {
            _editorPreviewQuad.transform.localRotation = Quaternion.identity;
        }

        // Size quad to fill the camera frustum while preserving the webcam's
        // native aspect ratio. This avoids horizontal stretching that makes the
        // feed look "zoomed in" when the Game view aspect doesn't match the webcam.
        float frustumH = 2f * Mathf.Tan(_editorPreviewCamera.fieldOfView * Mathf.Deg2Rad * 0.5f) * d;
        float frustumW = frustumH * _editorPreviewCamera.aspect;

        float webcamAspect = (_webcamTexture != null && _webcamTexture.width > 16)
            ? (float)_webcamTexture.width / _webcamTexture.height
            : _editorPreviewCamera.aspect;

        float w, h;
        if (webcamAspect >= _editorPreviewCamera.aspect)
        {
            // Webcam is wider than (or same as) Game view: fit to width, letterbox top/bottom
            w = frustumW;
            h = w / webcamAspect;
        }
        else
        {
            // Game view is wider than webcam: fit to height, pillarbox sides
            h = frustumH;
            w = h * webcamAspect;
        }

        _editorPreviewQuad.transform.localScale = new Vector3(w, h, 1f);
    }

    private void UpdateEditorPreviewTexture()
    {
        if (!_isWebcamMode || !enableWebcamPreview) return;
        if (_editorPreviewMaterial == null || _webcamTexture == null) return;
        if (_webcamTexture.width <= 16 || _webcamTexture.height <= 16) return;

        _editorPreviewMaterial.mainTexture = _webcamTexture;
        UpdateEditorPreviewWebcamUV();

        // Pass webcam texture and aspect ratio to OverlayRenderer
        if (overlayRenderer != null)
        {
            overlayRenderer.webcamAspectRatio = (float)_webcamTexture.width / _webcamTexture.height;
            overlayRenderer.webcamSourceTexture = _webcamTexture;
        }
    }

    /// <summary>
    /// Show/hide the webcam preview quad based on current mode.
    /// Called when switching modes at runtime.
    /// </summary>
    private void UpdatePreviewQuadVisibility()
    {
        if (_editorPreviewQuad != null)
        {
            bool visible = _isWebcamMode && enableWebcamPreview;
            _editorPreviewQuad.SetActive(visible);
        }
    }

    // ----------------------------------------------------------------
    // GUI overlay (segment labels + command HUD)
    // ----------------------------------------------------------------

    // Cached GUIStyles (created once in OnGUI to avoid per-frame alloc)
    private GUIStyle _labelBgStyle;
    private GUIStyle _labelTextStyle;
    private GUIStyle _hudBoxStyle;
    private GUIStyle _hudLabelStyle;
    private bool _guiStylesReady = false;

    private static readonly Color[] _segColors = new Color[]
    {
        new Color(0f, 1f, 1f),      // cyan
        new Color(1f, 0f, 1f),      // magenta
        new Color(0f, 1f, 0f),      // green
        new Color(1f, 1f, 0f),      // yellow
        new Color(1f, 0.5f, 0f),    // orange
        new Color(0.5f, 0f, 1f),    // purple
        new Color(0f, 0.5f, 1f),    // sky blue
        new Color(1f, 0f, 0.5f),    // hot pink
    };

    private void EnsureGUIStyles()
    {
        if (_guiStylesReady) return;

        // Semi-transparent black background for segment labels
        Texture2D bgTex = new Texture2D(1, 1);
        bgTex.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.7f));
        bgTex.Apply();

        _labelBgStyle = new GUIStyle(GUI.skin.box);
        _labelBgStyle.normal.background = bgTex;
        _labelBgStyle.padding = new RectOffset(8, 8, 4, 4);

        _labelTextStyle = new GUIStyle(GUI.skin.label);
        _labelTextStyle.fontSize = 18;
        _labelTextStyle.fontStyle = FontStyle.Bold;
        _labelTextStyle.alignment = TextAnchor.MiddleCenter;
        _labelTextStyle.normal.textColor = Color.white;

        _hudBoxStyle = new GUIStyle(GUI.skin.box);
        _hudLabelStyle = new GUIStyle(GUI.skin.label);
        _hudLabelStyle.fontSize = 13;

        _guiStylesReady = true;
    }

    private void OnGUI()
    {
        // Show GUI overlay in webcam mode (for both editor and device)
        if (!_isWebcamMode || !enableWebcamPreview) return;

        EnsureGUIStyles();

        // ── Draw segment labels directly on the video feed ──
        if (_latestResponse != null && _latestResponse.Segments.Count > 0)
        {
            float sw = Screen.width;
            float sh = Screen.height;

            int idx = 0;
            foreach (var seg in _latestResponse.Segments)
            {
                if (string.IsNullOrEmpty(seg.Label) && string.IsNullOrEmpty(seg.AssetClass))
                {
                    idx++;
                    continue;
                }

                string label = seg.Label;
                if (string.IsNullOrEmpty(label)) label = seg.AssetClass;

                // Effect annotation
                string effectStr = "";
                if (seg.Effect != null && !string.IsNullOrEmpty(seg.Effect.EffectType) && seg.Effect.EffectType != "none")
                    effectStr = $" [{seg.Effect.EffectType}]";

                string display = $"{label}{effectStr}";

                // Map server normalised coords (0‑1) to screen pixels
                float cx = seg.CenterX * sw;
                float cy = seg.CenterY * sh;

                Color col = _segColors[idx % _segColors.Length];
                _labelTextStyle.normal.textColor = col;

                Vector2 textSize = _labelTextStyle.CalcSize(new GUIContent(display));
                float boxW = textSize.x + 16;
                float boxH = textSize.y + 8;

                Rect bgRect = new Rect(cx - boxW * 0.5f, cy - boxH * 0.5f, boxW, boxH);
                GUI.Box(bgRect, GUIContent.none, _labelBgStyle);
                GUI.Label(bgRect, display, _labelTextStyle);

                // Small confidence tag below
                string confStr = $"{seg.Confidence:F2}";
                GUIStyle confStyle = new GUIStyle(_labelTextStyle);
                confStyle.fontSize = 12;
                confStyle.fontStyle = FontStyle.Normal;
                confStyle.normal.textColor = new Color(col.r, col.g, col.b, 0.8f);
                Vector2 confSize = confStyle.CalcSize(new GUIContent(confStr));
                Rect confRect = new Rect(cx - confSize.x * 0.5f, bgRect.yMax + 2, confSize.x, confSize.y);
                GUI.Label(confRect, confStr, confStyle);

                idx++;
            }
        }

        // ── HUD controls (top-left) ──
        if (!showCommandHUD) return;

        const int pad = 10;
        const int width = 560;
        const int lineH = 22;
        int x = pad;
        int y = pad;

        string modeLabel = _isWebcamMode ? "Webcam" : "AR Passthrough";
        GUI.Box(new Rect(x, y, width, 156), $"SocialSense ({modeLabel})");
        y += 26;

        string conn = _connected ? "<color=#00FF88>Connected</color>" : "<color=#FF4444>Disconnected</color>";
        int segs = _latestResponse != null ? _latestResponse.Segments.Count : 0;
        _hudLabelStyle.richText = true;
        GUI.Label(new Rect(x + 10, y, width - 20, lineH), $"Server: {conn} | Segs: {segs} | Mode: {modeLabel}", _hudLabelStyle);
        y += lineH;

        if (!_connected)
        {
            string err = string.IsNullOrEmpty(_lastWsError) ? "(none)" : _lastWsError;
            GUI.Label(new Rect(x + 10, y, width - 20, lineH), $"Close: {_lastWsCloseCode} | Err: {err}", _hudLabelStyle);
            y += lineH;
        }

        GUI.Label(new Rect(x + 10, y, 70, lineH), "Command:");
        editorCommand = GUI.TextField(new Rect(x + 85, y, 300, lineH), editorCommand ?? "");
        if (GUI.Button(new Rect(x + 395, y, 110, lineH), "Send"))
        {
            if (!string.IsNullOrWhiteSpace(editorCommand))
                SendControl(editorCommand.Trim());
        }
        y += lineH + 6;

        if (GUI.Button(new Rect(x + 10, y, 110, lineH), "Blur person")) SendControl("blur person");
        if (GUI.Button(new Rect(x + 130, y, 110, lineH), "Blur laptop")) SendControl("blur laptop");
        if (GUI.Button(new Rect(x + 250, y, 110, lineH), "Clear")) SendControl("clear");
        GUI.Label(new Rect(x + 370, y, 140, lineH), "Type or speak.");
        y += lineH + 6;

        // Mode toggle button — only show when AR hardware is actually available
        // (cameraAccess is null in editor/webcam-only, so switching to AR just falls back to webcam)
        if (cameraAccess != null)
        {
            string toggleLabel = _isWebcamMode ? "Switch to AR" : "Switch to Webcam";
            if (GUI.Button(new Rect(x + 10, y, 160, lineH), toggleLabel))
            {
                SwitchInputMode(_isWebcamMode ? InputMode.AR : InputMode.Webcam);
            }
        }
    }

    IEnumerator ConnectWhenReady()
    {
        // Wait for the active camera source to be ready
        if (_isWebcamMode)
        {
            while (_webcamTexture == null || !_webcamTexture.isPlaying)
                yield return new WaitForSeconds(0.5f);
        }
        else
        {
            // AR mode: wait for passthrough camera
            float timeout = 10f;
            float waited = 0f;
            while (cameraAccess != null && !cameraAccess.IsPlaying && waited < timeout)
            {
                yield return new WaitForSeconds(0.5f);
                waited += 0.5f;
            }
            if (cameraAccess == null || !cameraAccess.IsPlaying)
            {
                Debug.LogWarning($"[SocialSense] AR camera not ready after {waited}s. Proceeding anyway.");
            }
        }

        Debug.Log($"[SocialSense] Camera ready (mode={_activeMode}). Connecting to server...");
        ConnectWebSocket();
    }

    async void ConnectWebSocket()
    {
        if (_connecting)
        {
            Debug.Log("[SocialSense] Already connecting, skipping");
            return;
        }
        _connecting = true;
        _connectCount++;
        int attempt = _connectCount;

        // Close old socket if any
        if (_websocket != null)
        {
            Debug.Log($"[SocialSense] Closing old websocket (state={_websocket.State})");
            try
            {
                if (_websocket.State == WebSocketState.Open || _websocket.State == WebSocketState.Connecting)
                    await _websocket.Close();
            }
            catch (Exception) { }
            _websocket = null;
        }

        Debug.Log($"[SocialSense] Connecting to {serverUrl} (attempt #{attempt})");

        _websocket = new WebSocket(serverUrl);

        _websocket.OnOpen += () =>
        {
            Debug.Log($"[SocialSense] Connected! (attempt #{attempt})");
            _connected = true;
            _connecting = false;
            _lastWsError = "";
            _lastWsCloseCode = 0;

            // Start audio streaming after connection
            Debug.Log($"[SocialSense] Voice agent: enabled={enableVoiceAgent}, audioStreamer={(audioStreamer != null ? "assigned" : "NULL")}");
            if (enableVoiceAgent && audioStreamer != null)
            {
                audioStreamer.StartStreaming();
            }
        };

        _websocket.OnError += (e) =>
        {
            Debug.LogWarning($"[SocialSense] WebSocket error (attempt #{attempt}): {e}");
            _lastWsError = e ?? "";
        };

        _websocket.OnClose += (code) =>
        {
            Debug.Log($"[SocialSense] Disconnected (code={code}, attempt #{attempt})");
            _connected = false;
            _connecting = false;
            _lastWsCloseCode = (int)code;

            if (audioStreamer != null)
                audioStreamer.StopStreaming();
        };

        _websocket.OnMessage += (data) =>
        {
            _pendingResponse = data;
            _framesReceived++;
        };

        _lastConnectAttempt = Time.time;

        try
        {
            await _websocket.Connect();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[SocialSense] Connection failed (attempt #{attempt}): {e.Message}");
            _connected = false;
            _connecting = false;
        }
    }

    void Update()
    {
        // NativeWebSocket requires this to dispatch queued messages
        if (_websocket != null)
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            _websocket.DispatchMessageQueue();
#endif
        }

        // Reconnect if needed (guard against overlapping attempts)
        if (!_connected && !_connecting && Time.time - _lastConnectAttempt > RECONNECT_DELAY_SECONDS)
        {
            _lastConnectAttempt = Time.time;
            Debug.Log($"[SocialSense] Reconnecting... (prev attempts={_connectCount})");
            ConnectWebSocket();
        }

        if (!_connected) return;

        // Process any pending server response
        byte[] responseData = _pendingResponse;
        if (responseData != null)
        {
            _pendingResponse = null;
            try
            {
                var msg = new ServerMessage();
                msg.MergeFrom(new CodedInputStream(responseData));
                _latestResponse = msg;

                // masks_updated = true only when vision encoder ran (fresh features).
                // masks_frame_id = the frame_id whose vision features produced these masks.
                // Don't require masks_frame_id to differ — it's valid for vision to
                // re-run on a frame and produce updated masks.
                bool masksChanged = msg.MasksUpdated;

                // IMPORTANT: Always process EffectsCleared immediately, even if masks haven't changed.
                // This ensures "clear" command works even when server returns cached response.
                if (msg.EffectsCleared && overlayRenderer != null)
                {
                    Debug.Log("[SocialSense] EffectsCleared=true received, clearing all effects");
                    ClearAllEffects();
                    overlayRenderer.UpdateOverlay(msg.Segments, null, true);
                }
                else if (masksChanged)
                {
                    _lastMasksFrameId = msg.MasksFrameId;

                    // Look up capture rotation using masks_frame_id (the frame SAM processed),
                    // NOT msg.FrameId (the latest frame, which may have a very different head pose).
                    Quaternion? captureRot = null;
                    if (msg.MasksFrameId > 0)
                    {
                        Quaternion rot;
                        if (_captureRotations.TryGetValue(msg.MasksFrameId, out rot))
                            captureRot = rot;
                    }

                    if (overlayRenderer != null)
                    {
                        overlayRenderer.UpdateOverlay(msg.Segments, captureRot, false);
                    }

                    if (showDebugInfo)
                    {
                        float angle = captureRot.HasValue
                            ? Quaternion.Angle(captureRot.Value, overlayRenderer.centerEyeAnchor.rotation)
                            : -1f;
                        Debug.Log($"[SocialSense] NEW masks masksFrameId={msg.MasksFrameId} frameId={msg.FrameId} segs={msg.Segments.Count} rotDelta={angle:F1}°");
                    }
                }
                // When masks haven't changed and effects not cleared, don't update overlay —
                // keep existing world-locked texture and capture rotation.

                // Voice agent state — update HUD and full-screen filter
                ProcessVoiceAgentState(msg);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[SocialSense] Parse error: {e.Message}");
            }
        }

        // Detect runtime mode change (Inspector toggle of inputMode field)
        if (inputMode != _lastCheckedInputMode)
        {
            _lastCheckedInputMode = inputMode;
            _webcamInitialized = false;
            _arInitialized = false;
            ResolveInputMode();
            UpdatePreviewQuadVisibility();
        }

        // Capture and send at target rate
        bool cameraReady = _isWebcamMode
            ? (_webcamTexture != null && _webcamTexture.isPlaying)
            : (cameraAccess != null && cameraAccess.IsPlaying);

        if (!cameraReady) return;

        // Update webcam preview quad when in webcam mode
        if (_isWebcamMode && enableWebcamPreview)
        {
            if (_editorPreviewCamera == null) EnsureEditorPreviewCamera();
            if (_editorPreviewQuad == null) EnsureEditorPreviewQuad();
            UpdateEditorPreviewQuadTransform();
            UpdateEditorPreviewTexture();
        }

        float interval = 1f / targetFPS;
        if (Time.time - _lastSendTime >= interval)
        {
            CaptureAndSend();
            _lastSendTime = Time.time;
        }

        // Stats
        if (showDebugInfo && Time.time - _statsTime >= 2f)
        {
            float elapsed = Time.time - _statsTime;
            float sFps = _framesSent / elapsed;
            float rFps = _framesReceived / elapsed;
            int segs = _latestResponse != null ? _latestResponse.Segments.Count : 0;
            string stats = $"Send: {sFps:F0} fps | Recv: {rFps:F0} fps | Segs: {segs}";
            Debug.Log($"[SocialSense] {stats}");
            if (debugText != null) debugText.text = stats;
            _framesSent = 0;
            _framesReceived = 0;
            _statsTime = Time.time;
        }
    }

    // ----------------------------------------------------------------
    // Voice agent state processing
    // ----------------------------------------------------------------

    private string _lastProcessedCommand = "";

    private void ProcessVoiceAgentState(ServerMessage msg)
    {
        var voiceAgent = msg.Conversation?.VoiceAgent;

        // Debug: log voice agent state
        if (voiceAgent != null && (voiceAgent.Listening || voiceAgent.Recording ||
            !string.IsNullOrEmpty(voiceAgent.PartialTranscript) || !string.IsNullOrEmpty(voiceAgent.LastCommand)))
        {
            Debug.Log($"[SocialSense] VoiceAgent: listening={voiceAgent.Listening} recording={voiceAgent.Recording} partial=\"{voiceAgent.PartialTranscript}\" cmd=\"{voiceAgent.LastCommand}\"");
        }

        // Check if a new "clear" command was issued
        if (voiceAgent != null && !string.IsNullOrEmpty(voiceAgent.LastCommand))
        {
            string currentCmd = voiceAgent.LastCommand.ToLower();
            // Always re-process conversation-mode commands so toggling on/off
            // works even if the user repeats the same phrase.
            bool isRepeatAllowed = currentCmd.Contains("conversation") ||
                                   currentCmd.Contains("convo") ||
                                   currentCmd.Contains("focus mode");
            if (currentCmd != _lastProcessedCommand || isRepeatAllowed)
            {
                _lastProcessedCommand = currentCmd;

                // If the command contains "clear", force clear all effects including conversation mode
                if (currentCmd.Contains("clear"))
                {
                    Debug.Log($"[SocialSense] Detected CLEAR command: '{voiceAgent.LastCommand}' - clearing all effects");
                    ClearAllEffects();
                }

                // --- Conversation Mode: ONLY on explicit request (never auto-triggered) ---
                // Require full phrases to avoid false triggers (e.g. "have a conversation")
                bool isConvOn  = currentCmd.Contains("conversation mode") ||
                                 currentCmd.Contains("conversation  mode") ||
                                 currentCmd.Contains("convo mode") ||
                                 currentCmd.Contains("focus mode");
                bool isConvOff = currentCmd.Contains("exit conversation") ||
                                 currentCmd.Contains("leave conversation") ||
                                 currentCmd.Contains("end conversation") ||
                                 currentCmd.Contains("stop conversation");

                if (isConvOff && _conversationModeActive)
                {
                    Debug.Log("[SocialSense] Exiting Conversation Mode");
                    if (conversationModeEffect != null)
                        conversationModeEffect.Deactivate();
                    _conversationModeActive = false;
                }
                else if (isConvOn && !_conversationModeActive)
                {
                    Debug.Log("[SocialSense] Activating Conversation Mode");
                    ActivateConversationMode(msg);
                }
            }
        }

        // Update HUD
        if (voiceAgentHUD != null)
            voiceAgentHUD.UpdateState(voiceAgent);

        // Update full-screen filter
        if (fullScreenFilter != null && voiceAgent != null)
        {
            var fs = voiceAgent.FullScreenFilter;
            if (fs != null && !string.IsNullOrEmpty(fs.FilterType) && fs.FilterType != "none")
                fullScreenFilter.SetFilter(fs.FilterType, fs.Intensity, fs.ColorHex);
            else
                fullScreenFilter.SetFilter("none", 0f);
        }
    }

    /// <summary>
    /// Clear all visual effects: overlay, full-screen filter, and conversation mode.
    /// Use this for "clear"/"reset" commands, EffectsCleared, and startup.
    /// </summary>
    private void ClearAllEffects()
    {
        if (overlayRenderer != null)
            overlayRenderer.ClearOverlay();
        if (fullScreenFilter != null)
            fullScreenFilter.ClearFilter();
        if (conversationModeEffect != null && (conversationModeEffect.IsActive || _conversationModeActive))
        {
            conversationModeEffect.Deactivate();
            _conversationModeActive = false;
        }
    }

    /// <summary>
    /// Find the nearest (closest-to-camera) person segment in the current server message
    /// and start the Conversation Mode animation on it.
    ///
    /// "Nearest" is approximated by:
    ///   1. Largest bounding-box area (bigger silhouette = physically closer in a normal scene).
    ///   2. Segment label == "person" (asset class prioritised).
    ///
    /// We also decode the RLE mask so ConversationModeEffect can build the rim texture.
    /// </summary>
    private void ActivateConversationMode(ServerMessage msg)
    {
        if (conversationModeEffect == null)
        {
            // Auto-create: add the component to this GameObject
            conversationModeEffect = gameObject.AddComponent<ConversationModeEffect>();
            if (overlayRenderer != null)
                conversationModeEffect.overlayRenderer = overlayRenderer;
        }

        if (msg == null || msg.Segments == null || msg.Segments.Count == 0)
        {
            Debug.LogWarning("[SocialSense] ConversationMode: no segments in latest message — using screen centre");
            conversationModeEffect.Activate(0.5f, 0.5f, null, 0, 0);
            _conversationModeActive = true;
            return;
        }

        // Find the closest person: largest bbox area among "person" segments,
        // fall back to largest segment of any class if no person found.
        SceneSegment best      = null;
        float        bestScore = -1f;
        bool         foundPerson = false;

        foreach (var seg in msg.Segments)
        {
            if (seg.Bbox == null) continue;

            bool isPerson = (seg.AssetClass != null &&
                             seg.AssetClass.ToLower().Contains("person")) ||
                            (seg.Label != null &&
                             seg.Label.ToLower().Contains("person"));

            float bboxArea = (seg.Bbox.XMax - seg.Bbox.XMin) *
                             (seg.Bbox.YMax - seg.Bbox.YMin);

            // Prefer person over non-person
            if (isPerson && !foundPerson)
            {
                best        = seg;
                bestScore   = bboxArea;
                foundPerson = true;
            }
            else if (isPerson == foundPerson && bboxArea > bestScore)
            {
                best      = seg;
                bestScore = bboxArea;
            }
        }

        if (best == null)
        {
            Debug.LogWarning("[SocialSense] ConversationMode: no suitable segment found — using screen centre");
            conversationModeEffect.Activate(0.5f, 0.5f, null, 0, 0);
            _conversationModeActive = true;
            return;
        }

        // Decode the RLE mask for the rim texture builder
        byte[] maskBytes = null;
        int maskW = (int)best.MaskWidth;
        int maskH = (int)best.MaskHeight;
        if (best.RleMask != null && !best.RleMask.IsEmpty && maskW > 0 && maskH > 0)
        {
            maskBytes = DecodeRleMaskToBinary(best.RleMask.ToByteArray(), maskW, maskH);
        }

        Debug.Log($"[SocialSense] ConversationMode: locking to '{best.Label}' " +
                  $"center=({best.CenterX:F2},{best.CenterY:F2}) " +
                  $"bbox={bestScore:F4} person={foundPerson}");

        conversationModeEffect.Activate(best.CenterX, best.CenterY, maskBytes, maskW, maskH);
        _conversationModeActive = true;
    }

    /// <summary>
    /// Decode a uint16-LE RLE mask (as used by server/encoding/rle.py) into a flat
    /// binary byte array (0 = background, 255 = foreground) at maskW×maskH.
    ///
    /// Format: alternating background-run / foreground-run lengths as uint16 LE,
    /// starting with a background run (which may be 0).
    /// </summary>
    private static byte[] DecodeRleMaskToBinary(byte[] rle, int w, int h)
    {
        byte[] mask = new byte[w * h];
        int pos = 0;
        bool isFg = false;  // first run is always background

        for (int i = 0; i + 1 < rle.Length; i += 2)
        {
            int runLen = rle[i] | (rle[i + 1] << 8);
            byte val   = isFg ? (byte)255 : (byte)0;

            int end = System.Math.Min(pos + runLen, mask.Length);
            for (int j = pos; j < end; j++)
                mask[j] = val;

            pos  += runLen;
            isFg  = !isFg;

            if (pos >= mask.Length) break;
        }

        return mask;
    }

    void CaptureAndSend()
    {
        if (_isWebcamMode)
        {
            CaptureWebcam();
        }
        else
        {
            CaptureQuest();
        }
    }

    // ----------------------------------------------------------------
    // Webcam mode: webcam -> JPEG -> protobuf -> WebSocket
    // Works in both Editor and on-device.
    // ----------------------------------------------------------------

    void CaptureWebcam()
    {
        if (_webcamTexture == null || !_webcamTexture.isPlaying) return;

        int w = _webcamTexture.width;
        int h = _webcamTexture.height;

        if (_readbackTexture == null || _readbackTexture.width != w || _readbackTexture.height != h)
        {
            _readbackTexture = new Texture2D(w, h, TextureFormat.RGB24, false);
        }

        // IMPORTANT: The Python server assumes Quest frames arrive vertically flipped and
        // applies cv2.flip(frame_bgr, 0) on decode. WebCamTexture is NOT flipped that way,
        // so we pre-flip here so that server-side flip restores the correct orientation.
        Color[] src = _webcamTexture.GetPixels();
        Color[] flipped = new Color[src.Length];
        for (int y = 0; y < h; y++)
        {
            int srcRow = y * w;
            int dstRow = (h - 1 - y) * w;
            Array.Copy(src, srcRow, flipped, dstRow, w);
        }

        _readbackTexture.SetPixels(flipped);
        _readbackTexture.Apply();

        byte[] jpeg = _readbackTexture.EncodeToJPG(jpegQuality);
        SendFrame(jpeg, w, h);
    }

    // ----------------------------------------------------------------
    // Quest mode: passthrough camera -> GPU blit -> async readback -> JPEG
    // ----------------------------------------------------------------

    void CaptureQuest()
    {
        if (cameraAccess == null)
        {
            Debug.LogWarning("[SocialSense] cameraAccess is null in CaptureQuest");
            return;
        }

        Texture tex = cameraAccess.GetTexture();
        if (tex == null) return;

        int srcW = tex.width;
        int srcH = tex.height;

        int scaledW = Mathf.RoundToInt(srcW * resolutionScale);
        int scaledH = Mathf.RoundToInt(srcH * resolutionScale);
        scaledW = (scaledW / 2) * 2; // ensure even
        scaledH = (scaledH / 2) * 2;

        // Create RenderTextures if needed
        if (_scaledRT == null || _scaledRT.width != scaledW || _scaledRT.height != scaledH)
        {
            if (_scaledRT != null) _scaledRT.Release();
            if (_flippedRT != null) _flippedRT.Release();

            _scaledRT = new RenderTexture(scaledW, scaledH, 0, RenderTextureFormat.ARGB32);
            _scaledRT.Create();
            _flippedRT = new RenderTexture(scaledW, scaledH, 0, RenderTextureFormat.ARGB32);
            _flippedRT.Create();
            _readbackTexture = new Texture2D(scaledW, scaledH, TextureFormat.RGB24, false);
            _scaledW = scaledW;
            _scaledH = scaledH;
            Debug.Log($"[SocialSense] Capture: {srcW}x{srcH} -> {scaledW}x{scaledH}");
        }

        // GPU blit: source -> scaled -> flipped
        // Use DrawTexture for Quest passthrough OES textures (Blit fails on these)
        RenderTexture.active = _scaledRT;
        GL.PushMatrix();
        GL.LoadPixelMatrix(0, srcW, srcH, 0);
        Graphics.DrawTexture(new Rect(0, 0, srcW, srcH), tex);
        GL.PopMatrix();
        RenderTexture.active = null;

        // Scale + vertical flip
        Graphics.Blit(_scaledRT, _flippedRT, new Vector2(1, -1), new Vector2(0, 1));

        // Async GPU readback (non-blocking)
        if (_readbacksInFlight < MAX_READBACKS_IN_FLIGHT && SystemInfo.supportsAsyncGPUReadback)
        {
            _readbacksInFlight++;
            AsyncGPUReadback.Request(_flippedRT, 0, TextureFormat.RGB24, OnReadbackComplete);
        }
    }

    void OnReadbackComplete(AsyncGPUReadbackRequest request)
    {
        _readbacksInFlight = Mathf.Max(0, _readbacksInFlight - 1);

        if (request.hasError)
        {
            Debug.LogWarning("[SocialSense] GPU readback error");
            return;
        }

        // Get raw pixels and encode to JPEG
        if (_readbackTexture == null) return;

        _readbackTexture.LoadRawTextureData(request.GetData<byte>());
        _readbackTexture.Apply(false);

        byte[] jpeg = _readbackTexture.EncodeToJPG(jpegQuality);
        SendFrame(jpeg, _scaledW, _scaledH);
    }

    // ----------------------------------------------------------------
    // Send protobuf messages over WebSocket
    // ----------------------------------------------------------------

    async void SendFrame(byte[] jpegData, int width, int height)
    {
        if (_websocket == null || _websocket.State != WebSocketState.Open) return;

        _frameId++;

        // Store head rotation at capture time for pose reprojection
        if (overlayRenderer != null && overlayRenderer.centerEyeAnchor != null)
        {
            _captureRotations[_frameId] = overlayRenderer.centerEyeAnchor.rotation;

            // Prune old entries
            if (_captureRotations.Count > MAX_STORED_ROTATIONS)
            {
                ulong cutoff = _frameId > MAX_STORED_ROTATIONS ? _frameId - MAX_STORED_ROTATIONS : 0;
                var stale = new List<ulong>();
                foreach (var key in _captureRotations.Keys)
                    if (key < cutoff) stale.Add(key);
                foreach (var key in stale)
                    _captureRotations.Remove(key);
            }
        }

        var msg = new ClientMessage
        {
            FrameId = _frameId,
            TimestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            Frame = new FramePayload
            {
                JpegData = ByteString.CopyFrom(jpegData),
                Width = (uint)width,
                Height = (uint)height,
                Quality = (uint)jpegQuality,
            }
        };

        await SendClientMessage(msg);
        _framesSent++;
    }

    public async void SendAudio(byte[] pcm16Data, uint sampleRate, uint numSamples)
    {
        if (_websocket == null || _websocket.State != WebSocketState.Open)
        {
            Debug.LogWarning("[SocialSense] SendAudio: websocket not open");
            return;
        }

        var msg = new ClientMessage
        {
            TimestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            Audio = new AudioPayload
            {
                Pcm16Data = ByteString.CopyFrom(pcm16Data),
                SampleRate = sampleRate,
                NumSamples = numSamples,
            }
        };

        int size = msg.CalculateSize();
        Debug.Log($"[SocialSense] Sending audio msg: {numSamples} samples, size={size}B");
        await SendClientMessage(msg);
    }

    // Conversation mode is ONLY activated on explicit user request — never auto-triggered.
    // Patterns that are handled entirely client-side and must NEVER reach the server.
    // The server has no concept of conversation mode — if it receives these phrases
    // Gemini will hallucinate a colour/blur effect on the person.
    private static readonly System.Text.RegularExpressions.Regex _reConvModeOn =
        new System.Text.RegularExpressions.Regex(
            @"(?:enter|start|activate|begin\s+)?(?:conversation|convo|focus|cinema)\s+mode|conversation\s+(?:on|start|begin)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

    private static readonly System.Text.RegularExpressions.Regex _reConvModeOff =
        new System.Text.RegularExpressions.Regex(
            @"(?:exit|leave|end|stop|deactivate|quit)\s+(?:conversation|convo|focus|cinema)\s*(?:mode)?|conversation\s+(?:off|end|stop)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

    public async void SendControl(string command)
    {
        if (string.IsNullOrWhiteSpace(command)) return;

        string cmdLower = command.Trim().ToLower();

        // --- Client-side intercept for conversation mode ---
        // Handled entirely here; do NOT forward to server (server will apply blue mask).
        if (_reConvModeOn.IsMatch(cmdLower))
        {
            Debug.Log($"[SocialSense] Intercepted conversation-mode-ON command '{command}' — handling client-side");
            if (!_conversationModeActive)
            {
                // Use the most recent server message to find the nearest person
                ActivateConversationMode(_latestResponse);
            }
            return;   // ← never sent to server
        }

        if (_reConvModeOff.IsMatch(cmdLower))
        {
            Debug.Log($"[SocialSense] Intercepted conversation-mode-OFF command '{command}' — handling client-side");
            if (_conversationModeActive && conversationModeEffect != null)
            {
                conversationModeEffect.Deactivate();
                _conversationModeActive = false;
            }
            return;   // ← never sent to server
        }

        // "clear"/"reset" dismisses all effects including conversation mode
        if (cmdLower == "clear" || cmdLower == "reset")
        {
            ClearAllEffects();
            // still forward clear to server so it clears masks too
        }

        if (_websocket == null || _websocket.State != WebSocketState.Open) return;

        var msg = new ClientMessage
        {
            TimestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            Control = new ControlPayload { Command = command }
        };

        await SendClientMessage(msg);
    }

    private async System.Threading.Tasks.Task SendClientMessage(ClientMessage msg)
    {
        int size = msg.CalculateSize();
        byte[] buffer = new byte[size];
        var output = new CodedOutputStream(buffer);
        msg.WriteTo(output);
        output.Flush();

        try
        {
            await _websocket.Send(buffer);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[SocialSense] Send error: {e.Message}");
        }
    }

    // ----------------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------------

    void OnDisable()
    {
        // Only touch PassthroughCameraAccess when we actually used AR mode.
        // In webcam mode, leave it alone — disabling triggers NullReferenceException
        // in Meta.XR.PassthroughCameraAccess.Stop() / OnDisable().
        if (!_isWebcamMode && cameraAccess != null && cameraAccess.enabled)
        {
            try
            {
                // Safely disable without calling Stop() which might have null refs
                cameraAccess.enabled = false;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[SocialSense] Error disabling camera access: {e.Message}");
            }
        }
    }

    async void OnDestroy()
    {
        if (audioStreamer != null)
            audioStreamer.StopStreaming();

        if (_websocket != null && _websocket.State == WebSocketState.Open)
        {
            await _websocket.Close();
        }

        if (_scaledRT != null) _scaledRT.Release();
        if (_flippedRT != null) _flippedRT.Release();
        if (_readbackTexture != null) Destroy(_readbackTexture);
        if (_webcamTexture != null)
        {
            _webcamTexture.Stop();
            Destroy(_webcamTexture);
        }
        if (_editorPreviewQuad != null) Destroy(_editorPreviewQuad);
        if (_editorPreviewMaterial != null) Destroy(_editorPreviewMaterial);

        // Don't try to access cameraAccess here as it might already be destroyed
        cameraAccess = null;
    }

    void OnApplicationQuit()
    {
        if (audioStreamer != null)
            audioStreamer.StopStreaming();

        if (_websocket != null && _websocket.State == WebSocketState.Open)
        {
            _ = _websocket.Close();
        }
    }
}
