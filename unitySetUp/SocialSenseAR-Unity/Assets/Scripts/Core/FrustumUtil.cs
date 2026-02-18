using UnityEngine;

/// <summary>
/// Camera frustum utilities shared by OverlayRenderer (webcam quad),
/// SocialSenseClient (preview quad), and StoryPlayback (video quad).
/// </summary>
public static class FrustumUtil
{
    /// <summary>
    /// Compute quad scale to fill a perspective camera's frustum at a given distance,
    /// while preserving the content's aspect ratio (letterbox/pillarbox).
    /// Returns (width, height) for the quad's localScale.
    /// </summary>
    public static Vector2 ComputeQuadScale(Camera cam, float distance, float contentAspect)
    {
        float frustumH = 2f * Mathf.Tan(cam.fieldOfView * Mathf.Deg2Rad * 0.5f) * distance;
        float frustumW = frustumH * cam.aspect;

        float w, h;
        if (contentAspect >= cam.aspect)
        {
            w = frustumW;
            h = w / contentAspect;
        }
        else
        {
            h = frustumH;
            w = h * contentAspect;
        }

        return new Vector2(w, h);
    }
}
