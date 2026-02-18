Shader "SocialSense/ConversationModeFilter"
{
    // =========================================================================
    // Conversation Mode full-screen overlay shader — v3.
    //
    // Goal: MAXIMUM contrast between the person (full vivid colour) and the
    // world (crushed near-black, desaturated, heavily vignetted).
    //
    // The effect:
    //   • World  → near-black dark veil (luminance ~0.06 in settle/idle)
    //              with heavy vignette, CRT scanlines, and film grain.
    //   • Person → 100% transparent passthrough — full vivid colour,
    //              pixel-perfect cutout from _PersonMaskTex.
    //   • Boundary → chromatic aberration lens-fringe + electric rim glow.
    //
    // Animation timeline:
    //   Phase 0 — Ping      (0.00–0.35s)
    //     Double sonar ring burst + energy shockwave expanding outward.
    //   Phase 1 — Veil      (0.35–1.30s)
    //     Dark veil slams in from ALL edges simultaneously, not a slow sweep.
    //     Veil crushes to near-black with a fast ease-in curve.
    //   Phase 2 — Trace     (1.30–2.30s)
    //     Neon electric arc traces the full silhouette; double-outline
    //     (bright inner + soft outer glow); spark hot-spot at leading edge.
    //   Phase 3 — Settle    (2.30–3.00s)
    //     Veil deepens to maximum darkness (0.96 alpha, 0.06 luminance).
    //     Heavy vignette crushes corners to pure black.
    //     Person "pops" — vivid, perfectly isolated.
    //   Phase 4 — Idle
    //     Double-pulse heartbeat on the rim (fast spike then slow decay).
    //     CRT scanline shimmer on the dark world.
    //     Inner soft corona ring just inside the person boundary.
    //     Subtle breathing vignette pulse.
    // =========================================================================

    Properties
    {
        _Phase            ("Phase", Float) = 0
        _PhaseTimer       ("Phase Timer", Float) = 0
        _TotalTimer       ("Total Timer", Float) = 0

        _PersonCenter     ("Person Center UV", Vector) = (0.5, 0.5, 0, 0)

        // --- Ping / shockwave ---
        _PingRadius       ("Ping Radius", Float) = 0
        _PingRadius2      ("Ping Radius 2 (trailing)", Float) = 0
        _PingThickness    ("Ping Thickness", Float) = 0.025
        _PingColor        ("Ping Color", Color) = (0, 0.95, 1, 1)
        _PingBrightness   ("Ping Brightness", Float) = 4.0
        _ShockwaveRadius  ("Shockwave Radius", Float) = 0
        _ShockwaveStrength("Shockwave Strength", Float) = 0

        // --- Veil (near-black) ---
        _VeilProgress     ("Veil Progress", Float) = 0
        _VeilMax          ("Veil Max", Float) = 1.0
        _GreyscaleAmount  ("Greyscale Amount", Float) = 0
        // Darkness: 0=mid-grey(0.48), 1=near-black(0.06)
        _VeilDarkness     ("Veil Darkness", Float) = 0

        // --- Electric rim ---
        _TraceProgress    ("Trace Progress", Float) = 0
        _TraceColor       ("Trace Color", Color) = (0.2, 1.0, 1.0, 1)
        _TraceBrightness  ("Trace Brightness", Float) = 5.0
        _UseRimTex        ("Use Rim Tex", Float) = 0
        [NoScaleOffset] _RimTex ("Rim Texture", 2D) = "black" {}

        // --- Idle glow / corona ---
        _IdleGlowColor    ("Idle Glow Color", Color) = (0.4, 0.9, 1.0, 1)
        _IdleGlow         ("Idle Glow", Float) = 0
        _CoronaRadius     ("Inner Corona Radius UV", Float) = 0.04
        _CoronaStrength   ("Inner Corona Strength", Float) = 0

        // --- Settle ---
        _SettleProgress   ("Settle Progress", Float) = 0

        // --- Post-processing ---
        _Vignette         ("Vignette", Float) = 0.75
        _Grain            ("Grain", Float) = 0.06
        _GrainSeed        ("Grain Seed", Float) = 0
        _ScanlineStrength ("Scanline Strength", Float) = 0

        // --- Person mask ---
        _UsePersonMask    ("Use Person Mask", Float) = 0
        [NoScaleOffset] _PersonMaskTex ("Person Mask", 2D) = "black" {}

        // --- Chromatic aberration ---
        _ChromaStrength   ("Chroma Aberration Strength", Range(0, 0.03)) = 0.009
        // Edge sharpness: 0=feathered, 1=crisp
        _EdgeSharpness    ("Edge Sharpness", Range(0, 1)) = 0.75
        // Veil tint (cool blue-black by default)
        _VeilTint         ("Veil Tint", Color) = (0.75, 0.82, 1.0, 1.0)
    }

    SubShader
    {
        Tags
        {
            "RenderType"     = "Transparent"
            "Queue"          = "Overlay+50"
            "RenderPipeline" = "UniversalPipeline"
        }

        LOD 100
        ZWrite Off
        ZTest Always
        Cull Off

        Pass
        {
            Name "ConversationMode"
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma vertex   vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // ---- Uniforms ----
            float  _Phase;
            float  _PhaseTimer;
            float  _TotalTimer;
            float4 _PersonCenter;

            float  _PingRadius;
            float  _PingRadius2;
            float  _PingThickness;
            float4 _PingColor;
            float  _PingBrightness;
            float  _ShockwaveRadius;
            float  _ShockwaveStrength;

            float  _VeilProgress;
            float  _VeilMax;
            float  _GreyscaleAmount;
            float  _VeilDarkness;

            float  _TraceProgress;
            float4 _TraceColor;
            float  _TraceBrightness;
            float  _UseRimTex;
            TEXTURE2D(_RimTex);
            SAMPLER(sampler_RimTex);

            float4 _IdleGlowColor;
            float  _IdleGlow;
            float  _CoronaRadius;
            float  _CoronaStrength;

            float  _SettleProgress;

            float  _Vignette;
            float  _Grain;
            float  _GrainSeed;
            float  _ScanlineStrength;

            float  _UsePersonMask;
            TEXTURE2D(_PersonMaskTex);
            SAMPLER(sampler_PersonMaskTex);

            float  _ChromaStrength;
            float  _EdgeSharpness;
            float4 _VeilTint;

            // ---- Structs ----
            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv         : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // ---- Vertex ----
            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv = IN.uv;
                return OUT;
            }

            // ================================================================
            // Helpers
            // ================================================================

            // 4-tap blue-noise grain — no banding, good temporal variation
            float Grain4(float2 uv, float seed)
            {
                float2 p0 = uv * float2(1127.1, 1234.5) + seed;
                float2 p1 = uv * float2(1017.3, 989.7)  + seed * 1.37;
                float2 p2 = uv * float2(873.1,  1451.3) + seed * 0.73;
                float2 p3 = uv * float2(1381.9, 761.3)  + seed * 2.11;
                float g0 = frac(sin(dot(p0, float2(127.1, 311.7))) * 43758.5453);
                float g1 = frac(sin(dot(p1, float2(269.5, 183.3))) * 43758.5453);
                float g2 = frac(sin(dot(p2, float2(419.2, 371.9))) * 43758.5453);
                float g3 = frac(sin(dot(p3, float2(193.1, 527.7))) * 43758.5453);
                return (g0 + g1 + g2 + g3) * 0.25 - 0.5;
            }

            // Min distance from any screen edge (0 at edge, 0.5 at centre)
            float EdgeDist(float2 uv)
            {
                float2 d = min(uv, 1.0 - uv);
                return min(d.x, d.y);
            }

            // Chromatic-split person mask sample
            float3 ChromaMask(float2 maskUV, float strength, float cx)
            {
                float2 dir = float2(maskUV.x - cx, 0.0) * strength;
                float mR = SAMPLE_TEXTURE2D(_PersonMaskTex, sampler_PersonMaskTex, maskUV - dir * 1.5).a;
                float mG = SAMPLE_TEXTURE2D(_PersonMaskTex, sampler_PersonMaskTex, maskUV).a;
                float mB = SAMPLE_TEXTURE2D(_PersonMaskTex, sampler_PersonMaskTex, maskUV + dir * 1.5).a;
                return float3(mR, mG, mB);
            }

            // Narrow glowing ring utility
            float Ring(float dist, float radius, float thickness, float brightness)
            {
                float d = abs(dist - radius);
                return smoothstep(thickness, 0.0, d) * brightness;
            }

            // ================================================================
            // Fragment
            // ================================================================
            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float2 uv = IN.uv;
                // Mask textures are top-left origin; quad UV is bottom-left
                float2 maskUV = float2(uv.x, 1.0 - uv.y);

                // ============================================================
                // 1. PERSON MASK  — pixel-perfect transparent cutout
                // ============================================================
                float feather = lerp(0.07, 0.006, _EdgeSharpness);

                float personMask = 0.0;
                float3 chromaSplit = float3(0, 0, 0);

                if (_UsePersonMask > 0.5)
                {
                    float rawMask = SAMPLE_TEXTURE2D(_PersonMaskTex, sampler_PersonMaskTex, maskUV).a;
                    personMask = smoothstep(0.5 - feather, 0.5 + feather, rawMask);

                    if (_ChromaStrength > 0.0001 && _GreyscaleAmount > 0.05)
                    {
                        chromaSplit = ChromaMask(maskUV, _ChromaStrength * _GreyscaleAmount, _PersonCenter.x);
                        chromaSplit = smoothstep(0.5 - feather, 0.5 + feather, chromaSplit);
                    }
                    else
                    {
                        chromaSplit = float3(personMask, personMask, personMask);
                    }
                }
                else
                {
                    float2 diff = (uv - _PersonCenter.xy) * float2(1.78, 1.0);
                    personMask = 1.0 - smoothstep(0.28, 0.36, length(diff));
                    chromaSplit = float3(personMask, personMask, personMask);
                }

                // ============================================================
                // 2. DARK VEIL  — near-total greyscale (reference: no discernible
                //    background color; shades of grey/black only)
                //
                // Luminance: veilDarkness=1 → 0.01 (near-absolute black)
                // Alpha max 0.99 — passthrough almost fully replaced.
                // ============================================================
                float edgeDist   = EdgeDist(uv);
                float veilFront  = _VeilProgress * _VeilMax * 0.5;
                float veilCoverage = 1.0 - smoothstep(veilFront - 0.02, veilFront + 0.02, edgeDist);
                veilCoverage *= (1.0 - personMask);

                // Base luminance: strong crush (0.01 = near-black)
                float baseLum  = lerp(0.48, 0.01, _VeilDarkness);
                float3 veilRGB = float3(baseLum, baseLum, baseLum) * _VeilTint.rgb;

                // Settle: further crush toward black
                veilRGB *= (1.0 - 0.55 * _SettleProgress);

                // Alpha: max 0.99 — background fully occluded / greyscale
                float maxAlpha  = lerp(0.82, 0.99, _VeilDarkness);
                float veilAlpha = veilCoverage * _GreyscaleAmount * maxAlpha;

                // ============================================================
                // 3. VIGNETTE  — heavy corner crush to pure black
                //    During settle, corners go completely black.
                // ============================================================
                float vignetteFactor = 1.0 - smoothstep(0.02, 0.42, edgeDist);
                // Strong corner darkening
                veilRGB *= (1.0 - vignetteFactor * (0.6 + 0.4 * _SettleProgress));
                // Extra alpha punch at corners during settle
                float vignetteAlpha = vignetteFactor * _Vignette * _GreyscaleAmount * (1.0 - personMask);
                float cornerAlpha   = vignetteFactor * _SettleProgress * 0.98 * (1.0 - personMask);
                veilAlpha = max(veilAlpha, max(vignetteAlpha * 0.8, cornerAlpha));

                // ============================================================
                // 4. SCANLINES  — CRT shimmer on dark world during idle/settle
                //    Thin horizontal bands every 4 pixels; pulsed by _TotalTimer.
                // ============================================================
                float scanline = 0.0;
                if (_ScanlineStrength > 0.001)
                {
                    // 4-pixel bands (at 720p scale)
                    float band = frac(uv.y * 180.0 + _TotalTimer * 0.4);
                    scanline = smoothstep(0.7, 0.85, band) * _ScanlineStrength
                               * veilCoverage * _GreyscaleAmount;
                    // Scanlines slightly lighten the dark veil
                    veilRGB += scanline * 0.12;
                    veilAlpha = max(veilAlpha, scanline * 0.15 * veilCoverage);
                }

                // ============================================================
                // 5. FILM GRAIN  — 4-tap blue-noise on dark world
                // ============================================================
                float grain      = Grain4(uv * float2(1280.0, 720.0), _GrainSeed);
                float grainScale = veilCoverage * _Grain;
                float3 grainCol  = grain > 0.0 ? float3(1, 1, 1) : float3(0, 0, 0);
                veilRGB = lerp(veilRGB, grainCol, abs(grain) * grainScale);

                // ============================================================
                // 6. PING — double-ring burst + energy shockwave
                //    Ring 1: leading fast ring
                //    Ring 2: trailing ring 0.12 behind, 60% brightness
                //    Shockwave: large fading disc ripple
                // ============================================================
                float2 pDiff  = (uv - _PersonCenter.xy) * float2(1.78, 1.0);
                float  pDist  = length(pDiff);

                // Ring 1
                float fadeFar  = 1.0 - smoothstep(0.0, 1.8, _PingRadius);
                float ring1    = Ring(pDist, _PingRadius,  _PingThickness, _PingBrightness * 0.5) * fadeFar;
                // Ring 2 (trailing)
                float ring2    = Ring(pDist, _PingRadius2, _PingThickness * 0.7, _PingBrightness * 0.28) * fadeFar;

                // Shockwave ripple: thin expanding translucent disc
                float swFade   = 1.0 - smoothstep(0.0, 2.2, _ShockwaveRadius);
                float shockwave = Ring(pDist, _ShockwaveRadius, 0.06, _ShockwaveStrength) * swFade;

                float pingAlpha  = saturate(ring1 + ring2 + shockwave * 0.4);
                pingAlpha       *= (1.0 - personMask);

                // ============================================================
                // 7. ELECTRIC RIM — sampled from C#-baked _RimTex
                //    + SOFT OUTER GLOW: a dilated version of the rim for bloom
                // ============================================================
                half4 rimSample = half4(0, 0, 0, 0);
                half4 outerGlow = half4(0, 0, 0, 0);

                if (_UseRimTex > 0.5)
                {
                    rimSample = SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV);
                    rimSample.a *= (1.0 - personMask);

                    // Outer glow: blur approximation — average 8 taps at radius 0.006
                    float2 ts = float2(0.006, 0.006);
                    half4 g = half4(0, 0, 0, 0);
                    g += SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV + float2( ts.x,  0));
                    g += SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV + float2(-ts.x,  0));
                    g += SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV + float2( 0,  ts.y));
                    g += SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV + float2( 0, -ts.y));
                    g += SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV + float2( ts.x,  ts.y) * 0.707);
                    g += SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV + float2(-ts.x,  ts.y) * 0.707);
                    g += SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV + float2( ts.x, -ts.y) * 0.707);
                    g += SAMPLE_TEXTURE2D(_RimTex, sampler_RimTex, maskUV + float2(-ts.x, -ts.y) * 0.707);
                    outerGlow = g * 0.125;
                    outerGlow.a *= (1.0 - personMask) * 0.55;   // softer than inner
                }

                // Heartbeat idle-glow multiplier (double-pulse driven by C#)
                float glowMult = 1.0 + _IdleGlow * 2.2;
                rimSample.rgb  *= glowMult;
                outerGlow.rgb  *= (1.0 + _IdleGlow * 1.1);

                // ============================================================
                // 8. INNER CORONA  — soft halo just inside the person boundary
                //    In idle phase a ring of light sits just inside the silhouette,
                //    giving the person a subtle "chosen" aura.
                // ============================================================
                float corona = 0.0;
                if (_CoronaStrength > 0.001 && _UsePersonMask > 0.5)
                {
                    // Sample mask at a slightly inset UV to get the inner border
                    float2 insetDir = normalize(maskUV - float2(_PersonCenter.x, 1.0 - _PersonCenter.y) + 0.0001);
                    float insetMask = SAMPLE_TEXTURE2D(_PersonMaskTex, sampler_PersonMaskTex,
                                         maskUV - insetDir * _CoronaRadius).a;
                    // Corona = rim between outer and inset mask
                    float coronaBand = personMask * (1.0 - smoothstep(0.4, 0.8, insetMask));
                    corona = coronaBand * _CoronaStrength;
                }

                // ============================================================
                // Composite
                // ============================================================
                float3 outColor = veilRGB;
                float  outAlpha = veilAlpha;

                // Ping rings — additive over veil
                outColor  = lerp(outColor, _PingColor.rgb, pingAlpha);
                outAlpha  = max(outAlpha, pingAlpha);

                // Outer rim glow — additive, wide soft bloom
                outColor += outerGlow.rgb * outerGlow.a * 0.7;
                outAlpha  = max(outAlpha, outerGlow.a);

                // Inner rim — sharp bright electric arc
                outColor += rimSample.rgb * rimSample.a;
                outAlpha  = max(outAlpha, rimSample.a);

                // Inner corona aura (on the person's inside edge)
                // Rendered as a colour tinted by IdleGlowColor
                outColor += _IdleGlowColor.rgb * corona;
                outAlpha  = max(outAlpha, corona * 0.9);

                // ============================================================
                // Chromatic aberration halo at grey→colour boundary
                // ============================================================
                float chromaEdgeR = chromaSplit.r;
                float chromaEdgeB = chromaSplit.b;
                float chromaAlpha = abs(chromaEdgeR - chromaEdgeB)
                                    * _ChromaStrength * 100.0 * _GreyscaleAmount;
                float3 chromaFringe = float3(chromaEdgeR * 1.1, 0.4, chromaEdgeB * 1.1);
                outColor = lerp(outColor, chromaFringe, saturate(chromaAlpha));

                // ============================================================
                // Final person exclusion — GREEN channel is unshifted centre
                // ============================================================
                float personExclusion = saturate(chromaSplit.g * 2.0 - 0.5);
                outAlpha *= (1.0 - personExclusion);

                return half4(outColor, outAlpha);
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
