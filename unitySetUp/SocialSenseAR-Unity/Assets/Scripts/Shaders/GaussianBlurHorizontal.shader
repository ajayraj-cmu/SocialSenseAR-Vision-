Shader "Hidden/SocialSense/GaussianBlurHorizontal"
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

            half4 frag(Varyings IN) : SV_Target
            {
                float sigma = max(1.0, _BlurSize);
                float2 texelSize = _MainTex_TexelSize.xy;

                half4 color = half4(0, 0, 0, 0);
                float totalWeight = 0.0;

                // Gaussian blur: sample from -3*sigma to +3*sigma (covers 99.7%)
                // Use integer steps for clean sampling
                int radius = (int)ceil(sigma * 3.0);

                // Clamp radius to reasonable max to avoid shader timeout
                // Increased to 200 to support stronger privacy blur (sigma up to ~66)
                // Higher values may impact performance on lower-end devices
                radius = min(radius, 200);

                float twoSigmaSq = 2.0 * sigma * sigma;

                // Horizontal samples only
                for (int x = -radius; x <= radius; x++)
                {
                    float weight = exp(-(float)(x * x) / twoSigmaSq);
                    float2 offset = float2((float)x * texelSize.x, 0.0);
                    float2 sampleUV = IN.uv + offset;

                    // Clamp to edge (not repeat) for clean borders
                    sampleUV = clamp(sampleUV, 0.001, 0.999);

                    color += SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, sampleUV) * weight;
                    totalWeight += weight;
                }

                return color / totalWeight;
            }
            ENDHLSL
        }
    }

    FallBack Off
}
