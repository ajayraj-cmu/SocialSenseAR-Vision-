Shader "SocialSense/WallAnchorOverlay"
{
    // =========================================================================
    // Wall-Anchor Overlay v2 — 2.5D object-shaped sticker anchored in world space.
    //
    // Inputs
    // ------
    //   _MainTex   — RGBA colour+alpha crop baked from the overlay texture
    //   _MaskTex   — R8 silhouette mask: 1 = object pixel, 0 = background
    //                Bilinear-filtered so edges are anti-aliased.
    //
    // Visual design
    // -------------
    //   • Only object-shaped pixels are rendered (mask alpha cutout).
    //   • A configurable feather zone softens the silhouette boundary so the
    //     sticker looks like it naturally wraps around the object rather than
    //     having a hard cutout.
    //   • A thin edge-darkening rim gives the "sticker applied to surface" depth.
    //   • ZTest Always — sticker always renders on top of real geometry.
    //   • GlobalAlpha for fade-in / fade-out animations.
    // =========================================================================

    Properties
    {
        _MainTex      ("Baked Overlay Crop", 2D)      = "black" {}
        _MaskTex      ("Silhouette Mask (R8)", 2D)    = "white" {}
        _GlobalAlpha  ("Global Alpha", Range(0, 1))   = 1.0
        _EdgeFeather  ("Edge Feather", Range(0, 0.5)) = 0.12
        _EdgeDarken   ("Edge Darken Rim", Range(0, 1))= 0.35
        _Saturation   ("Colour Saturation", Range(0, 2)) = 1.15
    }

    SubShader
    {
        Tags
        {
            "RenderType"     = "Transparent"
            "Queue"          = "Transparent+100"
            "RenderPipeline" = "UniversalPipeline"
        }

        LOD 100

        Pass
        {
            Name "WallAnchorOverlay"
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest Always        // always render on top — 2.5D sticker feel
            Cull Back

            HLSLPROGRAM
            #pragma vertex   vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex);   SAMPLER(sampler_MainTex);
            TEXTURE2D(_MaskTex);   SAMPLER(sampler_MaskTex);

            float _GlobalAlpha;
            float _EdgeFeather;
            float _EdgeDarken;
            float _Saturation;

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

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv         = IN.uv;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float2 uv = IN.uv;

                // ---- 1. Sample colour and silhouette mask ----
                half4  col  = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, uv);
                float  mask = SAMPLE_TEXTURE2D(_MaskTex, sampler_MaskTex, uv).r;

                // ---- 2. Feathered silhouette cutout ----
                // mask is 0 outside object, 1 inside.
                // We remap: pixels with mask < _EdgeFeather fade out smoothly.
                float edgeFade = smoothstep(0.0, _EdgeFeather, mask);

                // ---- 3. Edge-darkening rim for 2.5D depth ----
                // Pixels near the silhouette boundary (mask ≈ _EdgeFeather) are
                // slightly darkened to give the impression the sticker has depth.
                float rimFactor  = 1.0 - smoothstep(_EdgeFeather, _EdgeFeather * 2.5, mask);
                col.rgb *= 1.0 - rimFactor * _EdgeDarken;

                // ---- 4. Slight saturation boost for vibrancy ----
                float luma = dot(col.rgb, float3(0.299, 0.587, 0.114));
                col.rgb    = lerp(float3(luma, luma, luma), col.rgb, _Saturation);

                // ---- 5. Combine: overlay alpha × silhouette × global ----
                // col.a comes from the overlay texture (effect color / opacity).
                col.a = col.a * edgeFade * _GlobalAlpha;

                // Hard discard pixels that are outside the silhouette entirely
                // (below half the feather threshold) to keep overdraw minimal.
                clip(col.a - 0.004);

                return col;
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
