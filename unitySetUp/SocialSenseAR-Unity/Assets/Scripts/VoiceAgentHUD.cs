using UnityEngine;
using UnityEngine.UI;
using Socialsense;

/// <summary>
/// World-space HUD showing voice agent state: listening indicator,
/// partial transcript during recording, and last response.
/// Appears when active, fades after idle.
/// </summary>
public class VoiceAgentHUD : MonoBehaviour
{
    [Header("UI References")]
    [SerializeField] private Text stateText;
    [SerializeField] private Text transcriptText;
    [SerializeField] private Text responseText;
    [SerializeField] private Image micIcon;
    [SerializeField] private CanvasGroup canvasGroup;

    [Header("Positioning")]
    [SerializeField] private Transform followTarget;  // CenterEyeAnchor
    [SerializeField] private float distanceFromHead = 1.5f;
    [SerializeField] private float verticalOffset = -0.3f;

    [Header("Timing")]
    [SerializeField] private float fadeDelay = 3f;
    [SerializeField] private float fadeSpeed = 2f;

    private float _lastActiveTime;
    private bool _wasActive;

    void Start()
    {
        if (canvasGroup == null)
            canvasGroup = GetComponentInChildren<CanvasGroup>();

        if (canvasGroup != null)
            canvasGroup.alpha = 0f;
    }

    public void UpdateState(VoiceAgentState state)
    {
        if (state == null)
        {
            return;
        }

        bool isActive = state.Recording || state.Listening;
        bool hasRecentResponse = !string.IsNullOrEmpty(state.LastResponse);

        if (state.Recording)
        {
            _lastActiveTime = Time.time;

            if (stateText != null)
            {
                stateText.text = "Recording...";
                stateText.color = Color.green;
            }

            if (micIcon != null)
                micIcon.color = Color.red;

            if (transcriptText != null)
            {
                transcriptText.gameObject.SetActive(true);
                transcriptText.text = state.PartialTranscript;
            }
        }
        else if (state.Listening)
        {
            if (hasRecentResponse)
                _lastActiveTime = Time.time;

            if (stateText != null)
            {
                stateText.text = "Say 'Hey Vibe'";
                stateText.color = Color.gray;
            }

            if (micIcon != null)
                micIcon.color = Color.white;

            if (transcriptText != null)
                transcriptText.gameObject.SetActive(false);
        }
        else
        {
            if (stateText != null)
                stateText.text = "";

            if (transcriptText != null)
                transcriptText.gameObject.SetActive(false);
        }

        if (responseText != null && hasRecentResponse)
        {
            responseText.text = state.LastResponse;
            _lastActiveTime = Time.time;
        }

        _wasActive = isActive || hasRecentResponse;
    }

    void Update()
    {
        // Position HUD in front of user
        if (followTarget != null)
        {
            Vector3 forward = followTarget.forward;
            forward.y = 0;
            forward.Normalize();

            transform.position = followTarget.position
                + forward * distanceFromHead
                + Vector3.up * verticalOffset;

            transform.rotation = Quaternion.LookRotation(
                transform.position - followTarget.position, Vector3.up);
        }

        // Fade logic
        if (canvasGroup != null)
        {
            float timeSinceActive = Time.time - _lastActiveTime;
            float targetAlpha = timeSinceActive < fadeDelay ? 1f : 0f;
            canvasGroup.alpha = Mathf.MoveTowards(canvasGroup.alpha, targetAlpha, fadeSpeed * Time.deltaTime);
        }
    }
}
