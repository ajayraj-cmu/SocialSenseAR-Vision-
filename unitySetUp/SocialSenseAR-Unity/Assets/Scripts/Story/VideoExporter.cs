using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// JPEG frame capture and ffmpeg video export.
/// Captures frames to disk, then encodes to H.264 MP4 on a background thread.
/// </summary>
public class VideoExporter
{
    public bool IsCapturing { get; private set; }
    public int CapturedCount { get; private set; }
    public string ExportStatus { get; private set; } = "";

    private string _outputPath;
    private int _jpegQuality;

    public VideoExporter(string outputPath, int jpegQuality)
    {
        _outputPath = outputPath;
        _jpegQuality = jpegQuality;
    }

    public void StartCapture()
    {
        if (IsCapturing) return;
        IsCapturing = true;
        CapturedCount = 0;
        if (!Directory.Exists(_outputPath)) Directory.CreateDirectory(_outputPath);
        Debug.Log($"[VideoExporter] REC ON → {Path.GetFullPath(_outputPath)}");
    }

    public void StopCapture()
    {
        if (!IsCapturing) return;
        IsCapturing = false;
        Debug.Log($"[VideoExporter] REC OFF — {CapturedCount} frames captured");
    }

    /// <summary>Save a frame as JPEG. Call each frame while capturing.</summary>
    public void SaveFrame(Texture2D tex)
    {
        if (!IsCapturing) return;
        byte[] jpg = tex.EncodeToJPG(_jpegQuality);
        File.WriteAllBytes(Path.Combine(_outputPath, $"frame_{CapturedCount:D5}.jpg"), jpg);
        CapturedCount++;
    }

    /// <summary>
    /// Encode captured frames to MP4 on a background thread using ffmpeg.
    /// Optionally muxes audio from a source video.
    /// </summary>
    public void ExportAsync(float fps, string sourceVideoPath = null)
    {
        if (CapturedCount == 0) return;

        string ffmpeg = FindFfmpeg();
        string absOutput = Path.GetFullPath(_outputPath);
        string pattern = Path.Combine(absOutput, "frame_%05d.jpg");
        string videoOut = Path.Combine(absOutput, "demo.mp4");
        string videoWithAudio = Path.Combine(absOutput, "demo_final.mp4");
        int count = CapturedCount;

        ExportStatus = "encoding...";
        Debug.Log($"[VideoExporter] Encoding {count} frames using {ffmpeg}...");

        new Thread(() =>
        {
            try
            {
                var encode = new ProcessStartInfo
                {
                    FileName = ffmpeg,
                    Arguments = $"-y -framerate {fps} -i \"{pattern}\" " +
                                $"-c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \"{videoOut}\"",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardError = true,
                };
                var proc = Process.Start(encode);
                proc.WaitForExit();

                if (proc.ExitCode != 0)
                {
                    string err = proc.StandardError.ReadToEnd();
                    ExportStatus = "FAILED";
                    Debug.LogError($"[VideoExporter] ffmpeg encode failed: {err.Substring(0, Math.Min(err.Length, 500))}");
                    return;
                }

                if (!string.IsNullOrEmpty(sourceVideoPath) && File.Exists(sourceVideoPath))
                {
                    ExportStatus = "muxing audio...";
                    var mux = new ProcessStartInfo
                    {
                        FileName = ffmpeg,
                        Arguments = $"-y -i \"{videoOut}\" -i \"{sourceVideoPath}\" " +
                                    $"-c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest \"{videoWithAudio}\"",
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardError = true,
                    };
                    var muxProc = Process.Start(mux);
                    muxProc.WaitForExit();
                    if (muxProc.ExitCode == 0)
                    {
                        File.Delete(videoOut);
                        File.Move(videoWithAudio, videoOut);
                    }
                }

                ExportStatus = $"DONE: {videoOut}";
                Debug.Log($"[VideoExporter] === EXPORT COMPLETE: {videoOut} ===");
            }
            catch (Exception e)
            {
                ExportStatus = "FAILED";
                Debug.LogError($"[VideoExporter] Export failed: {e.Message}");
            }
        }) { IsBackground = true }.Start();
    }

    private static string FindFfmpeg()
    {
        string[] candidates = {
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg",
            "ffmpeg",
        };
        foreach (var p in candidates)
            if (p == "ffmpeg" || File.Exists(p)) return p;
        return "ffmpeg";
    }
}
