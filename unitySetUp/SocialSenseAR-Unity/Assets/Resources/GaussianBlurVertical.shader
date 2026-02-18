Shader "Hidden/SocialSense/GaussianBlurVertical"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _BlurSize ("Blur Size (Sigma)", Float) = 10
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        LOD 100
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);
            float4 _MainTex_TexelSize;
            float _BlurSize;

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv = IN.uv;
                return OUT;
            }

            // Fixed max radius for loop unrolling
            #define MAX_RADIUS 64

            half4 frag(Varyings IN) : SV_Target
            {
                float sigma = max(1.0, _BlurSize);
                float2 texelSize = _MainTex_TexelSize.xy;

                // Reduce blur near edges to avoid sampling outside texture
                // This prevents streak artifacts from repeated edge pixels
                float distToEdgeY = min(IN.uv.y, 1.0 - IN.uv.y);
                float edgeBlend = saturate(distToEdgeY * 8.0); // Fade over ~12% of texture height
                float effectiveSigma = sigma * edgeBlend;
                effectiveSigma = max(1.0, effectiveSigma);

                // Calculate actual radius needed (clamped to MAX_RADIUS)
                float radiusF = min(effectiveSigma * 3.0, (float)MAX_RADIUS);

                half4 color = half4(0, 0, 0, 0);
                float totalWeight = 0.0;
                float twoSigmaSq = 2.0 * effectiveSigma * effectiveSigma;

                // Fixed loop bounds for compiler - skip samples outside actual radius
                [unroll(MAX_RADIUS * 2 + 1)]
                for (int y = -MAX_RADIUS; y <= MAX_RADIUS; y++)
                {
                    // Skip samples outside needed radius
                    if (abs(y) > radiusF) continue;

                    float weight = exp(-(float)(y * y) / twoSigmaSq);
                    float2 offset = float2(0.0, (float)y * texelSize.y);
                    float2 sampleUV = IN.uv + offset;

                    // Clamp to edge
                    sampleUV = clamp(sampleUV, 0.001, 0.999);

                    color += SAMPLE_TEXTURE2D_LOD(_MainTex, sampler_MainTex, sampleUV, 0) * weight;
                    totalWeight += weight;
                }

                return color / totalWeight;
            }
            ENDHLSL
        }
    }

    FallBack Off
}
