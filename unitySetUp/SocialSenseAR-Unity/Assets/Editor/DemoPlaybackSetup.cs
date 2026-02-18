using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;

public class DemoPlaybackSetup
{
    [MenuItem("SocialSense/Create Demo Playback Scene")]
    public static void CreateScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // Camera
        var camObj = new GameObject("Main Camera");
        var cam = camObj.AddComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = 5f;
        cam.nearClipPlane = 0.01f;
        cam.farClipPlane = 100f;
        cam.backgroundColor = Color.black;
        cam.clearFlags = CameraClearFlags.SolidColor;
        camObj.transform.position = new Vector3(0, 0, -10);
        camObj.tag = "MainCamera";

        // Canvas — ScreenSpaceOverlay (no camera dependency, no 3D quad issues)
        var canvasObj = new GameObject("Canvas");
        var canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        var scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.matchWidthOrHeight = 0.5f;
        canvasObj.AddComponent<GraphicRaycaster>();

        // Video display (fills canvas)
        var videoObj = new GameObject("VideoDisplay");
        videoObj.transform.SetParent(canvasObj.transform, false);
        var videoImg = videoObj.AddComponent<RawImage>();
        videoImg.color = Color.white;
        var vrt = videoImg.rectTransform;
        vrt.anchorMin = Vector2.zero;
        vrt.anchorMax = Vector2.one;
        vrt.offsetMin = Vector2.zero;
        vrt.offsetMax = Vector2.zero;

        // StoryPlayback
        var playbackObj = new GameObject("StoryPlayback");
        var pb = playbackObj.AddComponent<StoryPlayback>();
        pb.storyPath = Application.streamingAssetsPath + "/story_dir";
        pb.videoFilePath = Application.streamingAssetsPath + "/story_dir/RAW_DEMO_VIDEO.mp4";
        pb.videoDisplay = videoImg;
        pb.outputPath = System.IO.Path.Combine(Application.dataPath, "..", "demo_frames");

        EditorSceneManager.SaveScene(scene, "Assets/DemoPlayback.unity");

        Debug.Log("=== Demo Playback Scene Ready ===");
        Debug.Log("1. Set Game view to 1920x1080 (landscape)");
        Debug.Log("2. Press Play — video + masks + effects");
        Debug.Log("3. Press R to toggle frame capture");

        Selection.activeGameObject = playbackObj;
    }
}
