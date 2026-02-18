Shader "SocialSense/SegmentOverlay"
{
    Properties
    {
        _MainTex ("Overlay Texture", 2D) = "black" {}
        _GlobalAlpha ("Global Alpha", Range(0, 1)) = 0.4
        _EdgeFade ("Edge Fade Width", Range(0, 0.3)) = 0.12
        _DebugMode ("Debug Mode", Float) = 0
        _OutOfBoundsColor ("Out of Bounds Color", Color) = (0, 0, 0, 0)
        _SoftEdges ("Soft Edges (0=off, 1=2x2, 2=3x3)", Float) = 0
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Transparent"
            "Queue" = "Overlay"
            "RenderPipeline" = "UniversalPipeline"
        }

        LOD 100

        Pass
        {
            Name "SegmentOverlay"
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest Always
            Cull Off

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // Meta Environment Depth API
            #include "Packages/com.meta.xr.sdk.core/Shaders/EnvironmentDepth/URP/EnvironmentOcclusionURP.hlsl"

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            // Camera intrinsics
            float2 _FocalLength;
            float2 _PrincipalPoint;
            float2 _SensorResolution;
            float2 _TextureResolution;

            // Head rotation (world -> head-local, same for both eyes)
            float4x4 _CameraRotationMatrix;

            // Fine-tune offset (pixels in texture space)
            float2 _ProjectionOffset;

            // Camera world position (left passthrough camera).
            float3 _CameraWorldPos;

            // Right eye parallax
            float2 _RightEyeOffset;
            float _UseDepthReprojection;
            float2 _DepthSampleUV;
            float3 _RightEyeWorldPos;
            float _RightEyeShiftMultiplier;

            // Display settings
            float _GlobalAlpha;
            float _EdgeFade;
            float _UseIntrinsicsProjection;
            float _DebugMode;
            float4 _OutOfBoundsColor;  // Color for pixels outside camera FOV (for inverted effects)
            float _SoftEdges;  // 0=single sample, 1=2x2, 2=3x3

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(OUT);

                OUT.positionCS = TransformObjectToHClip(IN.positionOS.xyz);
                OUT.uv = IN.uv;
                OUT.worldPos = mul(unity_ObjectToWorld, IN.positionOS).xyz;

                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(IN);

                float2 finalUV;

                if (_UseIntrinsicsProjection > 0.5)
                {
                    // Baseline: direction from left camera to sphere fragment
                    float3 diff = IN.worldPos - _CameraWorldPos;
                    float3 localDir = mul(_CameraRotationMatrix, float4(diff, 0.0)).xyz;

                    // Behind camera = transparent
                    if (localDir.z < 0.001)
                    {
                        if (_DebugMode > 0.5)
                            return half4(1, 0, 1, 0.5);
                        return half4(0, 0, 0, 0);
                    }

                    // Scale intrinsics from sensor resolution to mask texture resolution
                    float scaleX = _TextureResolution.x / _SensorResolution.x;
                    float scaleY = _TextureResolution.y / _SensorResolution.y;

                    float2 scaledFocal = _FocalLength * float2(scaleX, scaleY);
                    float2 scaledPrincipal = _PrincipalPoint * float2(scaleX, scaleY);

                    // Pinhole projection: 3D direction -> 2D pixel coordinates
                    float uPixel = scaledFocal.x * (localDir.x / localDir.z) + scaledPrincipal.x;
                    float vPixel = scaledFocal.y * (localDir.y / localDir.z) + scaledPrincipal.y;

                    // Apply manual alignment offset (both eyes)
                    uPixel += _ProjectionOffset.x;
                    vPixel += _ProjectionOffset.y;

                    // Right-eye parallax: depth-based uniform shift sampled at segment center.
                    if (unity_StereoEyeIndex == 1)
                    {
                        bool offsetApplied = false;
                        if (_UseDepthReprojection > 0.5)
                        {
                            float objDepth = SampleEnvironmentDepthLinear(_DepthSampleUV);
                            if (objDepth > 0.2 && objDepth < 50.0)
                            {
                                float3 eyeOffset = _RightEyeWorldPos - _CameraWorldPos;
                                float3 localOffset = mul(_CameraRotationMatrix, float4(eyeOffset, 0.0)).xyz;
                                float shiftU = scaledFocal.x * localOffset.x / objDepth;
                                uPixel += clamp(shiftU * _RightEyeShiftMultiplier, -100.0, 100.0);
                                offsetApplied = true;
                            }
                        }
                        if (!offsetApplied)
                        {
                            uPixel += _RightEyeOffset.x;
                            vPixel += _RightEyeOffset.y;
                        }
                    }

                    // Normalize to [0,1] UV
                    finalUV = float2(uPixel / _TextureResolution.x, vPixel / _TextureResolution.y);

                    // Outside camera frustum - use out-of-bounds color if set, else transparent
                    if (finalUV.x < 0 || finalUV.x > 1 || finalUV.y < 0 || finalUV.y > 1)
                    {
                        if (_DebugMode > 0.5)
                            return half4(1, 1, 0, 0.8);
                        // If out-of-bounds color has alpha, use it (for inverted effects)
                        // Apply _GlobalAlpha to match inside effect alpha
                        if (_OutOfBoundsColor.a > 0.001)
                            return half4(_OutOfBoundsColor.rgb, _OutOfBoundsColor.a * _GlobalAlpha);
                        return half4(0, 0, 0, 0);
                    }

                    // Debug: show UV gradient
                    if (_DebugMode > 0.5)
                        return half4(finalUV.x, finalUV.y, 0.5, 0.8);
                }
                else
                {
                    // Fallback: simple UV passthrough (editor / no intrinsics)
                    finalUV = IN.uv;
                }

                // Sample overlay texture with optional soft edge multi-sampling
                half4 col;
                if (_SoftEdges > 1.5)
                {
                    // 3x3 box sampling for very soft edges
                    float2 texelSize = 1.0 / _TextureResolution;
                    half4 sum = half4(0, 0, 0, 0);
                    for (int y = -1; y <= 1; y++)
                    {
                        for (int x = -1; x <= 1; x++)
                        {
                            float2 offset = float2((float)x * texelSize.x, (float)y * texelSize.y);
                            sum += SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, finalUV + offset);
                        }
                    }
                    col = sum / 9.0;
                }
                else if (_SoftEdges > 0.5)
                {
                    // 2x2 box sampling for soft edges
                    float2 texelSize = 1.0 / _TextureResolution;
                    half4 s0 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, finalUV);
                    half4 s1 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, finalUV + float2(texelSize.x, 0));
                    half4 s2 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, finalUV + float2(0, texelSize.y));
                    half4 s3 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, finalUV + texelSize);
                    col = (s0 + s1 + s2 + s3) / 4.0;
                }
                else
                {
                    // Single sample (default, sharp edges)
                    col = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, finalUV);
                }

                // Soft fade near texture edges so the rectangular FOV boundary is invisible
                float fade = 1.0;
                if (_EdgeFade > 0.001)
                {
                    float e = _EdgeFade;
                    fade  = smoothstep(0.0, e, finalUV.x) * smoothstep(0.0, e, 1.0 - finalUV.x);
                    fade *= smoothstep(0.0, e, finalUV.y) * smoothstep(0.0, e, 1.0 - finalUV.y);
                }

                col.a *= _GlobalAlpha * fade;
                return col;
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
