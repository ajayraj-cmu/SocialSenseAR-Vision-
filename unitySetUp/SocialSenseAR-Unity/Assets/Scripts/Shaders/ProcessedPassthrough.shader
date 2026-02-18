Shader "Custom/ProcessedPassthrough"
{
    Properties
    {
        _MainTex ("Processed Video Texture", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 200

        Pass
        {
            Name "Processed Passthrough Display"
            Cull Off
            ZWrite On

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);
            float4 _MainTex_TexelSize;

            // Camera intrinsics (set from C# script)
            float2 _FocalLength;           // Focal length in pixels
            float2 _PrincipalPoint;        // Principal point in pixels (from top-left)
            float2 _SensorResolution;      // Original sensor resolution
            float2 _TextureResolution;     // Actual texture resolution (may be scaled)

            // Camera pose
            float3 _CameraPos;             // Camera world position
            float4x4 _CameraRotationMatrix; // World-to-camera rotation

            // Display settings
            float _UseIntrinsicsProjection; // 1.0 = use intrinsics, 0.0 = simple UV

            struct Attributes
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 clipPos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                OUT.clipPos = TransformObjectToHClip(IN.vertex.xyz);
                OUT.uv = IN.uv;

                // Get world position for intrinsics-based projection
                float4 worldPos = mul(unity_ObjectToWorld, IN.vertex);
                OUT.worldPos = worldPos.xyz;

                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float2 finalUV;

                if (_UseIntrinsicsProjection > 0.5)
                {
                    // Intrinsics-based projection (matches camera view)
                    float3 diff = IN.worldPos - _CameraPos;
                    float3 localPos = mul(_CameraRotationMatrix, float4(diff, 1.0)).xyz;

                    // Skip if behind camera
                    if (localPos.z < 0.001)
                        discard;

                    // Project to image plane using pinhole camera model
                    // Account for resolution scaling between sensor and texture
                    float scaleX = _TextureResolution.x / _SensorResolution.x;
                    float scaleY = _TextureResolution.y / _SensorResolution.y;

                    float2 scaledFocal = _FocalLength * float2(scaleX, scaleY);
                    float2 scaledPrincipal = _PrincipalPoint * float2(scaleX, scaleY);

                    // Compute pixel coordinates in texture space
                    float uPixel = scaledFocal.x * (localPos.x / localPos.z) + scaledPrincipal.x;
                    float vPixel = scaledFocal.y * (localPos.y / localPos.z) + scaledPrincipal.y;

                    // Normalize to [0,1] UV coordinates
                    finalUV = float2(uPixel / _TextureResolution.x, vPixel / _TextureResolution.y);

                    // Clamp to valid range
                    finalUV = saturate(finalUV);
                }
                else
                {
                    // Simple UV passthrough (fallback)
                    finalUV = IN.uv;
                }

                // Sample the processed texture
                half4 col = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, finalUV);

                return col;
            }
            ENDHLSL
        }
    }
    FallBack "Unlit/Texture"
}
