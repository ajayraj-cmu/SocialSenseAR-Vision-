Shader "SocialSense/FullScreenFilter"
{
    Properties
    {
        _FilterType ("Filter Type", Float) = 0
        _Intensity ("Intensity", Range(0, 1)) = 0
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Overlay"
            "Queue" = "Overlay+100"
            "RenderPipeline" = "UniversalPipeline"
        }

        LOD 100
        ZWrite Off
        ZTest Always
        Cull Off

        Pass
        {
            Name "FullScreenFilter"
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            float _FilterType;
            float _Intensity;

            struct Attributes
            {
                float4 positionOS : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                int filterType = (int)_FilterType;

                // dim: black overlay
                if (filterType == 1)
                    return half4(0, 0, 0, _Intensity);

                // warm: orange tint overlay
                if (filterType == 2)
                    return half4(1.0, 0.6, 0.2, 0.15 * _Intensity);

                // cool: blue tint overlay
                if (filterType == 3)
                    return half4(0.2, 0.4, 1.0, 0.15 * _Intensity);

                // night: red-only, dark overlay
                if (filterType == 4)
                    return half4(0.3, 0, 0, 0.5 * _Intensity);

                // grayscale: desaturate via green-ish overlay (approximation)
                if (filterType == 5)
                    return half4(0.5, 0.5, 0.5, 0.4 * _Intensity);

                return half4(0, 0, 0, 0);
            }
            ENDHLSL
        }
    }
}
