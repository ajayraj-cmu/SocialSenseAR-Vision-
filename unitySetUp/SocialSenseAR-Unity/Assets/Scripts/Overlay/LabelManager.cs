using System.Collections.Generic;
using Google.Protobuf.Collections;
using Socialsense;
using UnityEngine;

/// <summary>
/// Manages 3D text labels placed at each segment's center via inverse pinhole projection.
/// Pooled for performance — labels are reused across frames.
/// </summary>
public class LabelManager
{
    private readonly List<GameObject> _pool = new List<GameObject>();

    public float CharSize { get; set; } = 0.003f;
    public float SphereRadius { get; set; } = 50f;

    /// <summary>
    /// Place labels at segment centers using inverse pinhole projection.
    /// Requires valid intrinsics data from IntrinsicsProjector.
    /// </summary>
    public void UpdateLabels(
        RepeatedField<SceneSegment> segments,
        Vector2 focalLength, Vector2 principalPoint,
        float actualW, float actualH,
        Quaternion cameraProjectionRot,
        Transform centerEyeAnchor, Quaternion headRotation)
    {
        if (centerEyeAnchor == null)
        {
            HideAll();
            return;
        }

        int labelIdx = 0;
        int segIndex = 0;
        foreach (var seg in segments)
        {
            string label = seg.Label;
            if (string.IsNullOrEmpty(label)) { segIndex++; continue; }

            // Inverse pinhole: normalized center → pixel → camera-local direction
            // CenterY is OpenCV convention (0=top), flip for Unity (0=bottom)
            float px = seg.CenterX * actualW;
            float py = (1f - seg.CenterY) * actualH;
            float dx = (px - principalPoint.x) / focalLength.x;
            float dy = (py - principalPoint.y) / focalLength.y;
            Vector3 localDir = new Vector3(dx, dy, 1f).normalized;

            // Camera-local to world direction (world-locked)
            Vector3 worldDir = (headRotation * cameraProjectionRot) * localDir;

            float labelDistance = Mathf.Min(SphereRadius * 0.93f, 2f);
            Vector3 labelPos = centerEyeAnchor.position + worldDir * labelDistance;

            GameObject labelObj = GetOrCreate(labelIdx);
            labelObj.SetActive(true);
            labelObj.transform.position = labelPos;
            labelObj.transform.rotation = Quaternion.LookRotation(
                labelPos - centerEyeAnchor.position, Vector3.up);

            var tm = labelObj.GetComponent<TextMesh>();
            tm.text = label;
            tm.characterSize = CharSize;
            Color32 c = EffectConfig.GetBrightColor(segIndex);
            tm.color = new Color(c.r / 255f, c.g / 255f, c.b / 255f, 1f);

            labelIdx++;
            segIndex++;
        }

        // Hide unused labels
        for (int i = labelIdx; i < _pool.Count; i++)
            _pool[i].SetActive(false);
    }

    public void HideAll()
    {
        foreach (var obj in _pool)
            if (obj != null) obj.SetActive(false);
    }

    public void DestroyAll()
    {
        foreach (var obj in _pool)
            if (obj != null) Object.Destroy(obj);
        _pool.Clear();
    }

    private GameObject GetOrCreate(int index)
    {
        if (index < _pool.Count)
            return _pool[index];

        var go = new GameObject($"SegLabel_{index}");
        var tm = go.AddComponent<TextMesh>();
        tm.alignment = TextAlignment.Center;
        tm.anchor = TextAnchor.MiddleCenter;
        tm.characterSize = CharSize;
        tm.fontSize = 48;

        // Render AFTER overlay sphere (queue 4000) so labels aren't occluded
        var mr = go.GetComponent<MeshRenderer>();
        if (mr != null && mr.material != null)
            mr.material.renderQueue = 4010;

        go.transform.localScale = Vector3.one;
        _pool.Add(go);
        return go;
    }
}
