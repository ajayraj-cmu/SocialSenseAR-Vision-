using System.Collections;
using UnityEngine;
using Google.Protobuf.Collections;
using Socialsense;

/// <summary>
/// Conversation Mode: a cinematic full-screen animation that greyscales the entire
/// world EXCEPT the closest person, who remains in vivid color.
///
/// IMPORTANT: Never auto-starts. Only activated on explicit user request
/// ("conversation mode", "focus mode", etc.). Fully retractable via "clear",
/// "exit conversation", etc.
///
/// Animation Timeline
/// ==================
///  0.00 - 0.45 s  Phase 1 — Sonar Ping
///                  A sharp cyan ring expands outward from the person's screen-centre.
///                  Signals the system has locked onto the person.
///
///  0.45 - 1.60 s  Phase 2 — Greyscale Veil Sweep
///                  Greyscale "bleeds" inward from all four screen edges using a
///                  time-animated signed-distance threshold.  The world drains of color
///                  from the periphery inward.
///
///  1.60 - 2.70 s  Phase 3 — Electric Outline Trace
///                  A bright electric-arc rim traces around the person's silhouette.
///                  The arc travels around the contour via an animated angle-from-center
///                  sweep so it looks like electricity sparking around them.
///
///  2.70 - 3.40 s  Phase 4 — Settle
///                  The greyscale deepens, vignette corners darken, the electric outline
///                  fades to a soft steady glow, and the person "pops" back to full color.
///
///  3.40 s+        Phase 5 — Idle Loop
///                  A slow breathing pulse on the person's rim, subtle film-grain on the
///                  grey background, and a soft vignette. All parameters held steady with
///                  gentle oscillation.
///
/// How it works
/// ============
/// The animation drives a full-screen shader quad (ConversationModeFilter.shader)
/// that samples the screen's passthrough using the same approach as FullScreenFilterEffect.
/// A separate "person mask" overlay (greyscale-but-excluded) is built by
/// OverlayRenderer writing a special pass that preserves the person pixels.
///
/// Usage
/// =====
///   conversationModeEffect.Activate(personSegment, screenCenterX, screenCenterY);
///   conversationModeEffect.Deactivate();   // on "exit conversation mode"
/// </summary>
public class ConversationModeEffect : MonoBehaviour
{
    // -----------------------------------------------------------------------
    // Inspector
    // -----------------------------------------------------------------------

    [Header("Timing (seconds)")]
    [Tooltip("Duration of the sonar-ping ring expansion.")]
    public float phasePingDuration    = 0.45f;

    [Tooltip("Duration of the greyscale veil sweep.")]
    public float phaseVeilDuration    = 1.15f;

    [Tooltip("Duration of the electric outline trace.")]
    public float phaseTraceDuration   = 1.10f;

    [Tooltip("Duration of the settle phase.")]
    public float phaseSettleDuration  = 0.70f;

    [Header("Ping")]
    [Tooltip("Speed at which the sonar ring expands (0-1 UV space per second).")]
    public float pingSpeed       = 2.2f;
    [Tooltip("Thickness of the sonar ring in UV space.")]
    public float pingThickness   = 0.03f;
    [Tooltip("Brightness of the sonar ping ring.")]
    public float pingBrightness  = 4.5f;
    [Tooltip("Color of the sonar ping.")]
    public Color pingColor       = new Color(0.0f, 0.95f, 1.0f, 1.0f);   // cyan
    [Tooltip("Trailing ring offset behind main ring (UV units).")]
    public float pingTrailOffset = 0.12f;
    [Tooltip("Shockwave expansion multiplier (relative to ping radius).")]
    public float shockwaveScale  = 1.5f;
    [Tooltip("Shockwave strength (opacity).")]
    public float shockwaveStrength = 0.35f;

    [Header("Greyscale Veil")]
    [Tooltip("How far the veil sweeps inward from each edge (1.0 = reaches the screen centre).")]
    [Range(0f, 1f)]
    public float veilMaxCoverage = 1.0f;

    [Header("Electric Outline")]
    [Tooltip("Width in pixels (at mask resolution) of the inner electric rim.")]
    public int outlineThickness  = 5;
    [Tooltip("Brightness boost on the electric outline during trace phase.")]
    public float traceBrightness = 5.0f;
    [Tooltip("Color of the electric outline trace.")]
    public Color traceColor      = new Color(0.2f, 1.0f, 1.0f, 1.0f);   // electric cyan
    [Tooltip("Idle glow color after settle.")]
    public Color idleGlowColor   = new Color(0.4f, 0.9f, 1.0f, 1.0f);   // cool blue
    [Tooltip("Pulse frequency of the idle rim glow (Hz) — double-pulse heartbeat.")]
    public float idlePulseHz     = 1.2f;

    [Header("Vignette")]
    [Range(0f, 1f)]
    public float vignetteStrength = 0.75f;

    [Header("Film Grain")]
    [Range(0f, 1f)]
    public float grainStrength = 0.06f;

    [Header("Idle Enhancements")]
    [Tooltip("CRT scanline shimmer strength during idle.")]
    [Range(0f, 0.15f)]
    public float scanlineStrength = 0.08f;
    [Tooltip("Inner corona halo strength (soft aura inside person boundary).")]
    [Range(0f, 0.4f)]
    public float coronaStrength = 0.18f;
    [Tooltip("Inner corona radius in UV space.")]
    public float coronaRadius = 0.04f;
    [Tooltip("Veil tint. Use neutral grey for true greyscale; slight cool tint optional.")]
    public Color veilTint = new Color(0.48f, 0.48f, 0.52f, 1.0f);

    [Header("References")]
    [Tooltip("OverlayRenderer used to fetch the current person mask.")]
    public OverlayRenderer overlayRenderer;

    // -----------------------------------------------------------------------
    // Private state
    // -----------------------------------------------------------------------

    private enum Phase { Inactive, Ping, Veil, Trace, Settle, Idle }
    private Phase   _phase      = Phase.Inactive;
    private float   _phaseTimer = 0f;
    private float   _totalTimer = 0f;

    // Tracked person info (UV space, 0-1)
    private float _personCX = 0.5f;
    private float _personCY = 0.5f;

    // Person mask data (decoded from the SAM segment, same resolution as mask texture)
    private byte[] _personMask;
    private int    _maskW, _maskH;

    // Full-screen quad + material
    private GameObject _quad;
    private MeshRenderer _quadRenderer;
    private Material _mat;

    // Computed rim texture (updated during trace phase)
    private Texture2D _rimTex;
    private Color32[] _rimPixels;

    // Filled person silhouette texture — alpha=1 where person is, alpha=0 elsewhere.
    // Used by the shader to cut a pixel-perfect transparent hole in the grey veil.
    private Texture2D _personMaskTex;
    private Color32[] _personMaskPixels;

    // Shader property IDs
    private static readonly int PhaseId            = Shader.PropertyToID("_Phase");
    private static readonly int TimerId            = Shader.PropertyToID("_PhaseTimer");
    private static readonly int TotalTimerId       = Shader.PropertyToID("_TotalTimer");
    private static readonly int PersonCenterId     = Shader.PropertyToID("_PersonCenter");
    private static readonly int PingRadiusId       = Shader.PropertyToID("_PingRadius");
    private static readonly int PingRadius2Id      = Shader.PropertyToID("_PingRadius2");
    private static readonly int PingThicknessId    = Shader.PropertyToID("_PingThickness");
    private static readonly int PingColorId        = Shader.PropertyToID("_PingColor");
    private static readonly int PingBrightnessId   = Shader.PropertyToID("_PingBrightness");
    private static readonly int ShockwaveRadiusId  = Shader.PropertyToID("_ShockwaveRadius");
    private static readonly int ShockwaveStrengthId= Shader.PropertyToID("_ShockwaveStrength");
    private static readonly int VeilProgressId     = Shader.PropertyToID("_VeilProgress");
    private static readonly int VeilDarknessId     = Shader.PropertyToID("_VeilDarkness");
    private static readonly int VeilMaxId          = Shader.PropertyToID("_VeilMax");
    private static readonly int TraceProgressId    = Shader.PropertyToID("_TraceProgress");
    private static readonly int TraceColorId       = Shader.PropertyToID("_TraceColor");
    private static readonly int TraceBrightnessId  = Shader.PropertyToID("_TraceBrightness");
    private static readonly int IdleGlowColorId    = Shader.PropertyToID("_IdleGlowColor");
    private static readonly int IdleGlowId         = Shader.PropertyToID("_IdleGlow");
    private static readonly int CoronaRadiusId     = Shader.PropertyToID("_CoronaRadius");
    private static readonly int CoronaStrengthId   = Shader.PropertyToID("_CoronaStrength");
    private static readonly int ScanlineStrengthId = Shader.PropertyToID("_ScanlineStrength");
    private static readonly int VeilTintId         = Shader.PropertyToID("_VeilTint");
    private static readonly int VignetteId         = Shader.PropertyToID("_Vignette");
    private static readonly int GrainId            = Shader.PropertyToID("_Grain");
    private static readonly int GrainSeedId        = Shader.PropertyToID("_GrainSeed");
    private static readonly int RimTexId           = Shader.PropertyToID("_RimTex");
    private static readonly int UseRimTexId        = Shader.PropertyToID("_UseRimTex");
    private static readonly int GreyscaleId        = Shader.PropertyToID("_GreyscaleAmount");
    private static readonly int SettleId           = Shader.PropertyToID("_SettleProgress");
    private static readonly int PersonMaskTexId    = Shader.PropertyToID("_PersonMaskTex");
    private static readonly int UsePersonMaskId    = Shader.PropertyToID("_UsePersonMask");

    // -----------------------------------------------------------------------
    // Unity lifecycle
    // -----------------------------------------------------------------------

    private void Start()
    {
        BuildFullScreenQuad();
    }

    private void Update()
    {
        if (_phase == Phase.Inactive) return;

        float dt = Time.deltaTime;
        _phaseTimer += dt;
        _totalTimer += dt;

        AdvancePhase();
        PushUniforms();
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// <summary>
    /// Begin the Conversation Mode animation.
    /// personCX / personCY are the normalised (0-1) screen coordinates of the
    /// tracked person's centre (from SceneSegment.CenterX / CenterY).
    /// personMask is the decoded RLE mask at maskW×maskH resolution.
    /// </summary>
    public void Activate(float personCX, float personCY,
                         byte[] personMask, int maskW, int maskH)
    {
        _personCX   = personCX;
        _personCY   = 1f - personCY;    // flip Y: server is top-left, UV is bottom-left
        _personMask = personMask;
        _maskW      = maskW;
        _maskH      = maskH;

        _phase      = Phase.Ping;
        _phaseTimer = 0f;
        _totalTimer = 0f;

        // Bake the filled person silhouette for the shader's exclusion hole
        BuildPersonMaskTexture();

        // Pre-build the rim texture so it's ready for Phase 3
        BuildRimTexture(0f);   // arc-progress 0 → just the contour geometry, invisible yet

        if (_quad != null) _quad.SetActive(true);

        Debug.Log("[ConversationMode] Activated — starting Ping phase");
    }

    /// <summary>Deactivate and hide the effect. Fully retractable — resets shader to off state.</summary>
    public void Deactivate()
    {
        _phase = Phase.Inactive;
        _phaseTimer = 0f;
        _totalTimer = 0f;
        ResetShaderToOffState();
        if (_quad != null) _quad.SetActive(false);
        Debug.Log("[ConversationMode] Deactivated");
    }

    /// <summary>Reset shader uniforms to off/invisible so effect is fully retracted.</summary>
    private void ResetShaderToOffState()
    {
        if (_mat == null) return;
        _mat.SetFloat(GreyscaleId, 0f);
        _mat.SetFloat(VeilProgressId, 0f);
        _mat.SetFloat(VeilDarknessId, 0f);
        _mat.SetFloat(UseRimTexId, 0f);
        _mat.SetFloat(UsePersonMaskId, 0f);
    }

    public bool IsActive => _phase != Phase.Inactive;

    // -----------------------------------------------------------------------
    // Phase state machine
    // -----------------------------------------------------------------------

    private void AdvancePhase()
    {
        switch (_phase)
        {
            case Phase.Ping:
                if (_phaseTimer >= phasePingDuration)
                    TransitionTo(Phase.Veil);
                break;

            case Phase.Veil:
                if (_phaseTimer >= phaseVeilDuration)
                    TransitionTo(Phase.Trace);
                break;

            case Phase.Trace:
                // Update the rim texture arc progress every frame
                float arcProgress = Mathf.Clamp01(_phaseTimer / phaseTraceDuration);
                BuildRimTexture(arcProgress);
                if (_phaseTimer >= phaseTraceDuration)
                    TransitionTo(Phase.Settle);
                break;

            case Phase.Settle:
                if (_phaseTimer >= phaseSettleDuration)
                    TransitionTo(Phase.Idle);
                break;

            case Phase.Idle:
                // Stays here until Deactivate() is called
                // Rebuild rim texture fully complete (arc=1) so glow is always visible
                // Only rebuild once per second to save CPU
                if (Mathf.FloorToInt(_totalTimer) != Mathf.FloorToInt(_totalTimer - Time.deltaTime))
                    BuildRimTexture(1f);
                break;
        }
    }

    private void TransitionTo(Phase next)
    {
        Debug.Log($"[ConversationMode] Phase {_phase} → {next}");
        _phase      = next;
        _phaseTimer = 0f;
    }

    // -----------------------------------------------------------------------
    // Shader uniform push
    // -----------------------------------------------------------------------

    private void PushUniforms()
    {
        if (_mat == null) return;

        float t = Mathf.Clamp01(_phaseTimer);

        // --- Per-phase driven values ---
        float pingRadius       = 0f;
        float pingRadius2      = 0f;
        float shockwaveRadius  = 0f;
        float shockwaveStr     = 0f;
        float veilProgress     = 0f;
        float veilDarkness     = 0f;
        float traceProgress    = 0f;
        float greyAmount       = 0f;
        float settleProgress   = 0f;
        float idleGlow         = 0f;
        float coronaStr        = 0f;
        float scanlineStr      = 0f;
        bool  useRim           = false;

        float totalPhase = 0f;  // 0=Ping,1=Veil,2=Trace,3=Settle,4=Idle

        switch (_phase)
        {
            case Phase.Ping:
                totalPhase    = 0f;
                pingRadius    = (_phaseTimer / phasePingDuration) * pingSpeed;
                pingRadius2   = Mathf.Max(0f, pingRadius - pingTrailOffset);
                shockwaveRadius = pingRadius * shockwaveScale;
                shockwaveStr  = shockwaveStrength * (1f - _phaseTimer / phasePingDuration);
                veilProgress  = 0f;
                greyAmount    = 0f;
                break;

            case Phase.Veil:
                totalPhase    = 1f;
                pingRadius    = 999f;
                pingRadius2   = 999f;
                shockwaveRadius = 999f;
                float veilT = SmoothstepEaseInOut(_phaseTimer / phaseVeilDuration);
                veilProgress  = veilT;
                veilDarkness  = veilT;  // Full darkness ramp — background crushes to near-black
                greyAmount    = veilT;
                useRim        = false;
                break;

            case Phase.Trace:
                totalPhase    = 2f;
                veilProgress  = veilMaxCoverage;
                veilDarkness  = 1f;
                greyAmount    = veilMaxCoverage;
                traceProgress = SmoothstepEaseInOut(_phaseTimer / phaseTraceDuration);
                useRim        = true;
                break;

            case Phase.Settle:
                totalPhase    = 3f;
                veilProgress  = veilMaxCoverage;
                veilDarkness  = 1f;  // Max darkness — background fully crushed
                greyAmount    = 1f;
                settleProgress = SmoothstepEaseInOut(_phaseTimer / phaseSettleDuration);
                traceProgress = 1f;
                idleGlow      = settleProgress;
                coronaStr     = coronaStrength * settleProgress * 0.5f;
                scanlineStr   = scanlineStrength * settleProgress;
                useRim        = true;
                break;

            case Phase.Idle:
                totalPhase    = 4f;
                veilProgress  = veilMaxCoverage;
                veilDarkness  = 1f;
                greyAmount    = 1f;
                settleProgress = 1f;
                traceProgress = 1f;
                // Double-pulse heartbeat (two sharp peaks per cycle)
                float phase = Mathf.Repeat(_totalTimer * idlePulseHz, 1f);
                float p1 = Mathf.Exp(-Mathf.Pow((phase - 0.12f) * 14f, 2f));
                float p2 = Mathf.Exp(-Mathf.Pow((phase - 0.42f) * 14f, 2f));
                idleGlow  = 0.35f + 0.65f * (p1 + 0.55f * p2);
                coronaStr = coronaStrength;
                scanlineStr = scanlineStrength;
                useRim     = true;
                break;
        }

        _mat.SetFloat(PhaseId,            totalPhase);
        _mat.SetFloat(TimerId,            _phaseTimer);
        _mat.SetFloat(TotalTimerId,       _totalTimer);
        _mat.SetVector(PersonCenterId,    new Vector4(_personCX, _personCY, 0, 0));
        _mat.SetFloat(PingRadiusId,       pingRadius);
        _mat.SetFloat(PingRadius2Id,      pingRadius2);
        _mat.SetFloat(ShockwaveRadiusId,  shockwaveRadius);
        _mat.SetFloat(ShockwaveStrengthId, shockwaveStr);
        _mat.SetFloat(PingThicknessId,    pingThickness);
        _mat.SetColor(PingColorId,        pingColor);
        _mat.SetFloat(PingBrightnessId,   pingBrightness);
        _mat.SetFloat(VeilProgressId,     veilProgress);
        _mat.SetFloat(VeilDarknessId,     veilDarkness);
        _mat.SetFloat(VeilMaxId,        veilMaxCoverage);
        _mat.SetFloat(TraceProgressId,  traceProgress);
        _mat.SetColor(TraceColorId,     traceColor);
        _mat.SetFloat(TraceBrightnessId, traceBrightness);
        _mat.SetColor(IdleGlowColorId,  idleGlowColor);
        _mat.SetFloat(IdleGlowId,       idleGlow);
        _mat.SetFloat(VignetteId,       vignetteStrength * Mathf.Clamp01(greyAmount));
        _mat.SetFloat(GrainId,          grainStrength * Mathf.Clamp01(greyAmount));
        _mat.SetFloat(GrainSeedId,      _totalTimer * 137.5f);   // seed for frame-to-frame noise variation
        _mat.SetFloat(ScanlineStrengthId, scanlineStr);
        _mat.SetFloat(CoronaRadiusId,   coronaRadius);
        _mat.SetFloat(CoronaStrengthId, coronaStr);
        _mat.SetColor(VeilTintId,       veilTint);
        _mat.SetFloat(GreyscaleId,      greyAmount);
        _mat.SetFloat(SettleId,         settleProgress);
        _mat.SetFloat(UseRimTexId,      useRim ? 1f : 0f);

        if (useRim && _rimTex != null)
            _mat.SetTexture(RimTexId, _rimTex);

        // Always push person mask so shader can cut the exclusion hole
        bool hasPersonMask = _personMaskTex != null;
        _mat.SetFloat(UsePersonMaskId, hasPersonMask ? 1f : 0f);
        if (hasPersonMask)
            _mat.SetTexture(PersonMaskTexId, _personMaskTex);
    }

    // -----------------------------------------------------------------------
    // Rim texture builder
    // -----------------------------------------------------------------------

    /// <summary>
    /// Build the rim (outline) texture for the person at a given arc-progress (0-1).
    /// At progress=0 the texture is blank.  At progress=1 the full outline is lit.
    ///
    /// The arc is driven by sweeping an angular threshold from 0→360° around the
    /// person's centre pixel, using the angle of each outline pixel from centre.
    /// This gives the illusion of electricity tracing the contour.
    ///
    /// An extra "spark" 10° ahead of the leading edge adds a bright hot-spot.
    /// </summary>
    private void BuildRimTexture(float arcProgress)
    {
        if (_personMask == null || _maskW == 0 || _maskH == 0) return;

        // Ensure texture is allocated
        if (_rimTex == null || _rimTex.width != _maskW || _rimTex.height != _maskH)
        {
            if (_rimTex != null) Destroy(_rimTex);
            _rimTex    = new Texture2D(_maskW, _maskH, TextureFormat.RGBA32, false);
            _rimTex.filterMode = FilterMode.Bilinear;
            _rimTex.wrapMode   = TextureWrapMode.Clamp;
            _rimPixels = new Color32[_maskW * _maskH];
        }

        // Clear pixels
        System.Array.Fill(_rimPixels, new Color32(0, 0, 0, 0));

        // Compute outline pixels (pixels that are inside the mask but adjacent to outside)
        const byte threshold = 128;
        int thick = outlineThickness;

        // Centre of person in pixel space (CY already flipped when set)
        float cxPx = _personCX * _maskW;
        float cyPx = _personCY * _maskH;

        // Arc angle: 0→1 maps to -180→180° sweep (starting at "top" = -90° offset)
        float maxAngleDeg = arcProgress * 360f - 180f;   // leading edge angle (degrees)
        float sparkAngleDeg = maxAngleDeg + 12f;          // hot-spot just ahead

        for (int y = 0; y < _maskH; y++)
        {
            for (int x = 0; x < _maskW; x++)
            {
                if (_personMask[y * _maskW + x] < threshold) continue;

                // Check if this pixel is an outline pixel
                bool isOutline = false;
                for (int d = 1; d <= thick && !isOutline; d++)
                {
                    if (x - d < 0    || _personMask[y * _maskW + (x - d)] < threshold) { isOutline = true; break; }
                    if (x + d >= _maskW || _personMask[y * _maskW + (x + d)] < threshold) { isOutline = true; break; }
                    if (y - d < 0    || _personMask[(y - d) * _maskW + x] < threshold) { isOutline = true; break; }
                    if (y + d >= _maskH || _personMask[(y + d) * _maskW + x] < threshold) { isOutline = true; break; }
                }

                if (!isOutline) continue;

                // Compute angle of this pixel from person centre
                float dx = x - cxPx;
                float dy = y - cyPx;
                float angleDeg = Mathf.Atan2(dy, dx) * Mathf.Rad2Deg;   // -180..180

                // Is this pixel within the swept arc?
                // We need to compare in a circular fashion.
                // We sweep from -180 up to maxAngleDeg.
                bool inArc;
                if (arcProgress <= 0f)
                {
                    inArc = false;
                }
                else if (arcProgress >= 1f)
                {
                    inArc = true;
                }
                else
                {
                    // Normalize angle to -180..180 space
                    float startAngle = -180f;
                    float span       = (maxAngleDeg + 180f);   // 0..360

                    // Offset so start = 0
                    float a = angleDeg - startAngle;
                    if (a < 0) a += 360f;
                    inArc = a <= span;
                }

                if (!inArc) continue;

                // --- Color: spark hot-spot at leading edge ---
                float distToLeading = NormalizedAngularDist(angleDeg, maxAngleDeg);
                float distToSpark   = NormalizedAngularDist(angleDeg, sparkAngleDeg);

                // Hot-spot brightness: brighter near the leading edge
                float hot = Mathf.Exp(-distToLeading * 0.08f);

                // Base rim color interpolated between idle glow and trace color
                Color baseColor = Color.Lerp(idleGlowColor, traceColor, hot);

                // Spark: tiny patch ahead of leading edge (extra bright)
                float sparkFactor = arcProgress < 1f ? Mathf.Exp(-distToSpark * 0.05f) * 1.5f : 0f;

                float brightness  = (1f + sparkFactor) * Mathf.Lerp(1f, traceBrightness, hot);
                Color finalColor  = baseColor * brightness;
                finalColor.a      = 1f;

                // Soft distance falloff from outline centre for anti-aliasing
                float edgeDist    = 0f;
                for (int d = 1; d <= thick; d++)
                {
                    if ((x - d < 0 || _personMask[y * _maskW + (x - d)] < threshold) ||
                        (x + d >= _maskW || _personMask[y * _maskW + (x + d)] < threshold) ||
                        (y - d < 0 || _personMask[(y - d) * _maskW + x] < threshold) ||
                        (y + d >= _maskH || _personMask[(y + d) * _maskW + x] < threshold))
                    {
                        edgeDist = d / (float)thick;
                        break;
                    }
                }
                float alpha = Mathf.SmoothStep(1f, 0.2f, edgeDist);
                finalColor.a = alpha;

                _rimPixels[y * _maskW + x] = new Color32(
                    (byte)Mathf.Clamp(finalColor.r * 255f, 0, 255),
                    (byte)Mathf.Clamp(finalColor.g * 255f, 0, 255),
                    (byte)Mathf.Clamp(finalColor.b * 255f, 0, 255),
                    (byte)(alpha * 255f));
            }
        }

        _rimTex.SetPixels32(_rimPixels);
        _rimTex.Apply(false);
    }

    /// <summary>
    /// Bake the filled person silhouette into _personMaskTex.
    /// alpha=255 where the person mask is set, alpha=0 everywhere else.
    /// A soft 2-pixel feather is applied at the boundary so the veil edge
    /// blends cleanly without a hard visible rim-seam.
    /// This texture is passed to the shader as _PersonMaskTex so it can cut
    /// a pixel-perfect transparent hole in the grey veil over the person.
    /// </summary>
    private void BuildPersonMaskTexture()
    {
        if (_personMask == null || _maskW == 0 || _maskH == 0)
        {
            _personMaskTex = null;
            return;
        }

        if (_personMaskTex == null || _personMaskTex.width != _maskW || _personMaskTex.height != _maskH)
        {
            if (_personMaskTex != null) Destroy(_personMaskTex);
            _personMaskTex     = new Texture2D(_maskW, _maskH, TextureFormat.RGBA32, false);
            _personMaskTex.filterMode = FilterMode.Bilinear;
            _personMaskTex.wrapMode   = TextureWrapMode.Clamp;
            _personMaskPixels  = new Color32[_maskW * _maskH];
        }

        const byte threshold = 128;
        // Feather width in pixels — makes veil edge dissolve into person smoothly
        const int feather = 6;

        for (int i = 0; i < _maskW * _maskH; i++)
        {
            int x = i % _maskW;
            int y = i / _maskW;

            if (_personMask[i] < threshold)
            {
                _personMaskPixels[i] = new Color32(0, 0, 0, 0);
                continue;
            }

            // Compute approximate distance-to-edge of the mask for feathering
            int minDist = feather + 1;
            for (int d = 1; d <= feather; d++)
            {
                bool edgeFound =
                    (x - d < 0    || _personMask[y * _maskW + (x - d)] < threshold) ||
                    (x + d >= _maskW || _personMask[y * _maskW + (x + d)] < threshold) ||
                    (y - d < 0    || _personMask[(y - d) * _maskW + x] < threshold) ||
                    (y + d >= _maskH || _personMask[(y + d) * _maskW + x] < threshold);
                if (edgeFound) { minDist = d; break; }
            }

            // Full alpha inside, feathered at the border
            float a = Mathf.SmoothStep(0f, 1f, (float)minDist / feather);
            _personMaskPixels[i] = new Color32(255, 255, 255, (byte)(a * 255f));
        }

        _personMaskTex.SetPixels32(_personMaskPixels);
        _personMaskTex.Apply(false);
    }

    /// <summary>Angular distance (absolute degrees) between two angles in -180..180 space.</summary>
    private static float NormalizedAngularDist(float a, float b)
    {
        float d = Mathf.Abs(a - b) % 360f;
        return d > 180f ? 360f - d : d;
    }

    // -----------------------------------------------------------------------
    // Full-screen quad
    // -----------------------------------------------------------------------

    private void BuildFullScreenQuad()
    {
        Camera cam = GetComponent<Camera>() ?? Camera.main;
        if (cam == null) return;

        var shader = Shader.Find("SocialSense/ConversationModeFilter");
        if (shader == null)
        {
            Debug.LogError("[ConversationMode] Cannot find SocialSense/ConversationModeFilter shader!");
            return;
        }
        _mat = new Material(shader);

        _quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _quad.name = "ConversationModeQuad";
        _quad.transform.SetParent(cam.transform, false);

        var col = _quad.GetComponent<Collider>();
        if (col != null) Destroy(col);

        // Position just beyond near clip, covering full viewport
        float dist   = cam.nearClipPlane + 0.01f;
        float height = 2f * dist * Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float width  = height * cam.aspect;

        _quad.transform.localPosition  = new Vector3(0, 0, dist);
        _quad.transform.localRotation  = Quaternion.identity;
        _quad.transform.localScale     = new Vector3(width * 1.1f, height * 1.1f, 1f);

        _quadRenderer = _quad.GetComponent<MeshRenderer>();
        _quadRenderer.material = _mat;
        _quadRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        _quadRenderer.receiveShadows    = false;

        // Ensure quad is hidden — effect is off until user explicitly requests it
        _quad.SetActive(false);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Smoothstep with ease-in-out curve.
    private static float SmoothstepEaseInOut(float t)
    {
        t = Mathf.Clamp01(t);
        return t * t * (3f - 2f * t);
    }

    private void OnDestroy()
    {
        if (_quad != null)          Destroy(_quad);
        if (_mat != null)           Destroy(_mat);
        if (_rimTex != null)        Destroy(_rimTex);
        if (_personMaskTex != null) Destroy(_personMaskTex);
    }
}
