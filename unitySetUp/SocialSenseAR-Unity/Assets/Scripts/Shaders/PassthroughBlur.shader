Shader "SocialSense/PassthroughBlur"
{
    Properties
    {
        _MainTex ("Mask Texture", 2D) = "black" {}
        _PassthroughTex ("Blurred Passthrough", 2D) = "black" {}
        _GlobalAlpha ("Global Alpha", Range(0, 1)) = 1.0
        _OutOfBoundsAlpha ("Out of Bounds Alpha", Range(0, 1)) = 0.0
        _DebugMode ("Debug Mode (0=off, 1=mask, 2=passthrough)", Float) = 0
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Transparent"
            "Queue" = "Overlay+10"
            "RenderPipeline" = "UniversalPipeline"
        }

        LOD 100

        Pass
        {
            Name "PassthroughBlur"
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
            TEXTURE2D(_PassthroughTex);
            SAMPLER(sampler_PassthroughTex);

            // Camera intrinsics (same as SegmentOverlay)
            float2 _FocalLength;
            float2 _PrincipalPoint;
            float2 _SensorResolution;
            float2 _TextureResolution;
            float4x4 _CameraRotationMatrix;
            float2 _ProjectionOffset;
            float3 _CameraWorldPos;
            float2 _RightEyeOffset;

            // Depth reprojection
            float _UseDepthReprojection;
            float2 _DepthSampleUV;
            float3 _RightEyeWorldPos;
            float _RightEyeShiftMultiplier;

            float _GlobalAlpha;
            float _UseIntrinsicsProjection;
            float _OutOfBoundsAlpha;
            float _DebugMode;

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

                    if (localDir.z < 0.001)
                        return half4(0, 0, 0, 0);

                    float scaleX = _TextureResolution.x / _SensorResolution.x;
                    float scaleY = _TextureResolution.y / _SensorResolution.y;

                    float2 scaledFocal = _FocalLength * float2(scaleX, scaleY);
                    float2 scaledPrincipal = _PrincipalPoint * float2(scaleX, scaleY);

                    float uPixel = scaledFocal.x * (localDir.x / localDir.z) + scaledPrincipal.x;
                    float vPixel = scaledFocal.y * (localDir.y / localDir.z) + scaledPrincipal.y;

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

                    finalUV = float2(uPixel / _TextureResolution.x, vPixel / _TextureResolution.y);
                }
                else
                {
                    finalUV = IN.uv;
                }

                // Clamp UV for sampling
                float2 clampedUV = clamp(finalUV, 0.001, 0.999);

                // Check if we're outside the camera FOV
                bool outOfBounds = finalUV.x < 0 || finalUV.x > 1 || finalUV.y < 0 || finalUV.y > 1;

                // Sample the mask texture
                half4 mask = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, clampedUV);

                // Determine blur alpha based on mode:
                float maskAlpha;
                bool isInvertedBlur = _OutOfBoundsAlpha > 0.5;

                if (isInvertedBlur)
                {
                    // INVERTED BLUR (blur everything except target):
                    // mask.a comes from the inverted+smoothed mask, so it already has
                    // soft edges (high alpha = blur here, low alpha = target, don't blur)
                    if (outOfBounds)
                        maskAlpha = _OutOfBoundsAlpha;
                    else
                        maskAlpha = mask.a * _OutOfBoundsAlpha;
                }
                else
                {
                    // NORMAL BLUR (blur only the target):
                    // Hard cutoff — any partial alpha composites blurred over sharp
                    // passthrough behind, creating a washed-out grey halo.
                    maskAlpha = (outOfBounds || mask.a < 0.5) ? 0.0 : 1.0;
                }

                // Debug mode 1: show mask alpha as red color
                if (_DebugMode > 0.5 && _DebugMode < 1.5)
                {
                    return half4(maskAlpha, 0, 0, 0.8);
                }

                if (maskAlpha < 0.01)
                    return half4(0, 0, 0, 0);  // No blur needed here

                // Debug mode 2: show raw passthrough
                if (_DebugMode > 1.5 && _DebugMode < 2.5)
                {
                    half4 raw = SAMPLE_TEXTURE2D(_PassthroughTex, sampler_PassthroughTex, clampedUV);
                    raw.a = maskAlpha;
                    return raw;
                }

                // Sample the PRE-BLURRED passthrough texture
                half4 blurred = SAMPLE_TEXTURE2D(_PassthroughTex, sampler_PassthroughTex, clampedUV);

                // Use full alpha from mask
                blurred.a = maskAlpha;

                return blurred;
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
