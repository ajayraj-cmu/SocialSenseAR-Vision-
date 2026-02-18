using System;
using UnityEngine;
using Google.Protobuf;
using Socialsense;

/// <summary>
/// Captures PCM16 audio from Quest microphone and sends over WebSocket
/// as AudioPayload protobuf messages at regular intervals.
/// </summary>
public class AudioStreamer : MonoBehaviour
{
    [SerializeField] private SocialSenseClient client;
    [SerializeField] private int sampleRate = 16000;
    [SerializeField] private float chunkDuration = 0.5f;

    private AudioClip _micClip;
    private int _lastReadPos;
    private float _timer;
    private bool _streaming;

    public bool IsStreaming => _streaming;

    public void StartStreaming()
    {
        if (_streaming) return;

#if UNITY_ANDROID && !UNITY_EDITOR
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Microphone))
        {
            Debug.LogWarning("[AudioStreamer] Microphone permission not granted");
            return;
        }
#endif

        string device = null; // default device
        if (Microphone.devices.Length == 0)
        {
            Debug.LogWarning("[AudioStreamer] No microphone found");
            return;
        }

        _micClip = Microphone.Start(device, true, 10, sampleRate);
        if (_micClip == null)
        {
            Debug.LogError("[AudioStreamer] Microphone.Start returned null!");
            return;
        }
        _lastReadPos = 0;
        _timer = 0f;
        _streaming = true;
        Debug.Log($"[AudioStreamer] Started streaming at {sampleRate}Hz, chunk={chunkDuration}s, devices={Microphone.devices.Length}, clip={_micClip.length}s");
    }

    public void StopStreaming()
    {
        if (!_streaming) return;
        _streaming = false;
        Microphone.End(null);
        Debug.Log("[AudioStreamer] Stopped streaming");
    }

    private float _debugTimer;

    void Update()
    {
        if (!_streaming || client == null) return;

        _timer += Time.deltaTime;
        _debugTimer += Time.deltaTime;

        // Debug log every 2 seconds
        if (_debugTimer >= 2f)
        {
            int pos = Microphone.GetPosition(null);
            bool recording = Microphone.IsRecording(null);
            Debug.Log($"[AudioStreamer] Debug: pos={pos} lastPos={_lastReadPos} recording={recording} micClip={((_micClip != null) ? "valid" : "NULL")}");
            _debugTimer = 0f;
        }

        if (_timer < chunkDuration) return;
        _timer = 0f;

        int currentPos = Microphone.GetPosition(null);
        if (currentPos == _lastReadPos) return;

        // Calculate samples to read from circular AudioClip buffer
        int totalSamples = _micClip.samples;
        int samplesToRead = currentPos >= _lastReadPos
            ? currentPos - _lastReadPos
            : (totalSamples - _lastReadPos) + currentPos;

        if (samplesToRead <= 0) return;

        float[] samples = new float[samplesToRead];
        _micClip.GetData(samples, _lastReadPos);
        _lastReadPos = currentPos;

        // Convert float32 [-1,1] -> PCM16 little-endian bytes
        byte[] pcm16 = new byte[samplesToRead * 2];
        for (int i = 0; i < samplesToRead; i++)
        {
            short val = (short)(Mathf.Clamp(samples[i], -1f, 1f) * 32767);
            pcm16[i * 2] = (byte)(val & 0xFF);
            pcm16[i * 2 + 1] = (byte)((val >> 8) & 0xFF);
        }

        Debug.Log($"[AudioStreamer] Sending {samplesToRead} samples ({pcm16.Length} bytes)");
        client.SendAudio(pcm16, (uint)sampleRate, (uint)samplesToRead);
    }

    void OnDestroy()
    {
        StopStreaming();
    }
}
