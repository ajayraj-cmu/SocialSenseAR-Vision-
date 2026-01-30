#!/usr/bin/env python3
"""
Demo Video Renderer - Renders conversation helper overlay onto a video file.

Usage:
    python render_demo.py input.mp4 output.mp4
    python render_demo.py input.mp4 output.mp4 --audio  # Also process audio for transcription
    python render_demo.py input.mp4 output.mp4 --fps 30  # Output at specific FPS

This processes the video through YOLO segmentation and renders the glassmorphism
overlay panels, creating a polished demo video.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from processors import get_processor
from effects import get_effect
from ui.overlay_panels import render_overlay


def find_ffmpeg() -> str:
    """Find ffmpeg executable."""
    import shutil
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg

    # Common Windows locations
    common_paths = [
        r"C:\Program Files\ShareX\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in common_paths:
        if Path(path).exists():
            return path
    return None


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video using ffmpeg."""
    import subprocess
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[AUDIO] ffmpeg not found")
        return False

    try:
        cmd = [
            ffmpeg, '-y', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            str(audio_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"[AUDIO] Could not extract audio: {e}")
        return False


def mux_audio(video_path: Path, audio_source: Path, output_path: Path, fps: float = 30.0, width: int = None, height: int = None) -> bool:
    """Re-encode video with H.264 and mux audio from source using ffmpeg.

    This fixes jittering by using proper H.264 encoding with constant frame rate.

    Args:
        video_path: Not used directly (output_path is renamed to temp)
        audio_source: Original video to take audio from
        output_path: Final output path
        fps: Frame rate for output
        width: Video width (if None, will probe from temp file)
        height: Video height (if None, will probe from temp file)
    """
    import subprocess
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[AUDIO] ffmpeg not found, output will have no audio")
        return False

    if width is not None and height is not None:
        print(f"[FFMPEG] Re-encoding with H.264 and adding audio (preserving {width}x{height})...")
    else:
        print(f"[FFMPEG] Re-encoding with H.264 and adding audio...")
    temp_output = output_path.with_suffix('.temp.mp4')

    try:
        # Rename current output to temp
        if temp_output.exists():
            temp_output.unlink()
        video_path_to_mux = output_path
        video_path_to_mux.rename(temp_output)

        # Build the video filter to preserve exact dimensions and set square pixels
        vf_filters = ['setsar=1:1']  # Set sample aspect ratio to 1:1 (square pixels)

        # If dimensions provided, add scale filter to ensure exact match
        if width is not None and height is not None:
            # Use scale with exact dimensions, no padding
            vf_filters.insert(0, f'scale={width}:{height}:force_original_aspect_ratio=disable')

        vf_string = ','.join(vf_filters)

        cmd = [
            ffmpeg, '-y',
            '-i', str(temp_output),       # Rendered video (no audio)
            '-i', str(audio_source),      # Original video (for audio)
            '-c:v', 'libx264',            # Re-encode with H.264
            '-preset', 'fast',            # Encoding speed/quality tradeoff
            '-crf', '18',                 # Quality (lower = better, 18 is visually lossless)
            '-r', str(fps),               # Force constant frame rate
            '-vf', vf_string,             # Video filters: scale + setsar for exact dimensions
            '-pix_fmt', 'yuv420p',        # Compatibility
            '-c:a', 'aac',                # Encode audio as AAC
            '-b:a', '192k',               # Audio bitrate
            '-map', '0:v:0',              # Take video from first input
            '-map', '1:a:0',              # Take audio from second input
            '-shortest',                   # End when shortest stream ends
            '-movflags', '+faststart',    # Web-friendly
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[FFMPEG] error: {result.stderr}")
            # Restore original
            temp_output.rename(output_path)
            return False

        # Cleanup temp file
        if temp_output.exists():
            temp_output.unlink()

        print(f"[FFMPEG] Encoding complete!")
        return True

    except Exception as e:
        print(f"[FFMPEG] Could not encode video: {e}")
        # Try to restore original if it exists
        if temp_output.exists() and not output_path.exists():
            temp_output.rename(output_path)
        return False


def process_audio_for_demo(audio_path: Path, state_dir: Path, duration: float) -> list:
    """Process audio and return timed state updates.

    Returns list of (timestamp, state_dict) tuples.
    """
    # This is a simplified version - in practice you'd run the full
    # context service on the audio file
    print("[AUDIO] Processing audio for transcription...")

    # For now, return empty - the overlay will show "Listening..."
    # A full implementation would:
    # 1. Run VAD to find speech segments
    # 2. Transcribe each segment with Whisper
    # 3. Run GPT summarization
    # 4. Return timed state updates
    return []


def create_mock_state_timeline(duration: float, script_file: Path = None) -> list:
    """Create a timeline of mock states for demo purposes.

    Args:
        duration: Video duration in seconds
        script_file: Optional JSON file with scripted state changes

    Returns:
        List of (timestamp, state_dict) tuples
    """
    if script_file and script_file.exists():
        import json
        with open(script_file) as f:
            return [(item['time'], item['state']) for item in json.load(f)]

    # Default demo script - customize for your video
    timeline = [
        (0.0, {
            "convo_state_summary": "Starting conversation...",
            "recent_utterance": "",
            "question": "",
            "emotion": "neutral",
            "emotion_display": "Neutral",
            "is_other_speaking": False,
        }),
        (2.0, {
            "convo_state_summary": "Greeting and small talk",
            "recent_utterance": "Hey, how's it going?",
            "question": "",
            "emotion": "happy",
            "emotion_display": "Happy",
            "is_other_speaking": True,
        }),
        (5.0, {
            "convo_state_summary": "Discussing weekend plans",
            "recent_utterance": "I was thinking about going hiking",
            "question": "",
            "emotion": "happy",
            "emotion_display": "Happy",
            "is_other_speaking": True,
        }),
        (8.0, {
            "convo_state_summary": "Discussing weekend plans",
            "recent_utterance": "Want to come with us?",
            "question": "Want to come with us?",
            "emotion": "happy",
            "emotion_display": "Happy",
            "is_other_speaking": False,
            "social_cue": "They're inviting you to join",
            "social_cue_timestamp": 8.0,
        }),
        (12.0, {
            "convo_state_summary": "Making plans together, excited mood",
            "recent_utterance": "That sounds awesome!",
            "question": "",
            "emotion": "surprise",
            "emotion_display": "Surprised",
            "is_other_speaking": True,
        }),
        (15.0, {
            "convo_state_summary": "Finalizing hiking plans",
            "recent_utterance": "Let's meet at 8am Saturday",
            "question": "",
            "emotion": "happy",
            "emotion_display": "Happy",
            "is_other_speaking": True,
        }),
    ]

    # Filter to only include states within video duration
    return [(t, s) for t, s in timeline if t <= duration]


def write_state_file(state_dir: Path, state: dict):
    """Write state to the state file for overlay to read."""
    import json
    state_file = state_dir / "latest_state.json"
    state['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def render_demo_video(
    input_path: str,
    output_path: str,
    output_fps: int = None,
    process_audio: bool = False,
    show_preview: bool = False,
    script_file: str = None,
    use_mock_state: bool = True,
    effect_start: float = 0.0
):
    """Render a demo video with conversation helper overlay.

    Args:
        input_path: Path to input video file
        output_path: Path for output video
        output_fps: Output FPS (None = same as input)
        process_audio: Whether to extract and process audio
        show_preview: Show preview window while rendering
        script_file: Optional JSON file with scripted state changes
        use_mock_state: Use mock state timeline for demo (default True)
        effect_start: Time in seconds when focus effect starts (default 0.0)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False

    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return False

    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / input_fps if input_fps > 0 else 0

    print(f"\n{'='*60}")
    print(f"Demo Video Renderer")
    print(f"{'='*60}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {duration:.1f}s ({frame_count} frames @ {input_fps:.1f}fps)")
    print(f"{'='*60}\n")

    # Use input FPS if not specified
    if output_fps is None:
        output_fps = input_fps

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False

    # Initialize processor and effect
    print("Loading AI model...")
    config = Config(preset="QUALITY")
    processor = get_processor("yolo", config)
    effect = get_effect("focus", config)

    processor.start()
    print("Model loaded!\n")

    # Setup state directory
    state_dir = Path("~/Downloads/Nex/conve_context").expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)

    # Create state timeline
    if use_mock_state:
        script_path = Path(script_file) if script_file else None
        state_timeline = create_mock_state_timeline(duration, script_path)
        print(f"Using mock state timeline with {len(state_timeline)} state changes")
    elif process_audio:
        audio_path = input_path.with_suffix('.wav')
        if extract_audio(input_path, audio_path):
            state_timeline = process_audio_for_demo(audio_path, state_dir, duration)
        else:
            state_timeline = []
    else:
        state_timeline = []

    # Initialize with first state
    current_state_idx = 0
    if state_timeline:
        write_state_file(state_dir, state_timeline[0][1])

    # Process frames
    print("Rendering frames...")
    frame_idx = 0

    # Use tqdm for progress bar
    pbar = tqdm(total=frame_count, unit='frames', desc='Rendering')

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Calculate current timestamp
            current_time = frame_idx / input_fps

            # Update state if we've reached the next state change
            while (current_state_idx < len(state_timeline) - 1 and
                   current_time >= state_timeline[current_state_idx + 1][0]):
                current_state_idx += 1
                state = state_timeline[current_state_idx][1].copy()
                # Update social_cue_timestamp to be relative to render time
                if 'social_cue_timestamp' in state:
                    state['social_cue_timestamp'] = time.time()
                write_state_file(state_dir, state)

            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Process with YOLO
            processor.process(frame_rgb)
            result = processor.get_result()

            # Apply focus effect (only after effect_start time)
            effect_active = current_time >= effect_start

            if effect_active and result is not None and result.has_detection:
                frame_out = effect.apply(frame_rgb, result)
                person_tracked = True
                head_x = result.left_center[0]
                head_y = max(0.1, result.left_center[1] - 0.15)
            else:
                frame_out = effect.no_effect(frame_rgb)
                person_tracked = result is not None and result.has_detection if effect_active else False
                if person_tracked:
                    head_x = result.left_center[0]
                    head_y = max(0.1, result.left_center[1] - 0.15)
                else:
                    head_x, head_y = 0.5, 0.3

            # Render overlay (expects BGR, returns BGR)
            frame_out = render_overlay(
                frame_out,
                head_x=head_x,
                head_y=head_y,
                person_tracked=person_tracked
            )

            # frame_out is already BGR (effect.apply returns BGR, overlay expects/returns BGR)
            # Write frame directly
            out.write(frame_out)

            # Show preview if requested
            if show_preview:
                preview = cv2.resize(frame_out, (width // 2, height // 2))
                cv2.imshow('Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nPreview closed, stopping render...")
                    break

            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        out.release()
        processor.stop()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"Rendering complete!")
    print(f"Frames rendered: {frame_idx}")
    print(f"{'='*60}\n")

    # Re-encode with H.264 and mux audio from original video
    # Pass width and height to ensure output dimensions match input exactly
    mux_audio(output_path, input_path, output_path, fps=input_fps, width=width, height=height)

    print(f"Output saved to: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Render conversation helper demo video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python render_demo.py input.mp4 output.mp4
    python render_demo.py input.mp4 output.mp4 --preview
    python render_demo.py input.mp4 output.mp4 --script my_script.json
    python render_demo.py input.mp4 output.mp4 --no-mock  # No overlay text changes

Script JSON format:
    [
        {"time": 0.0, "state": {"convo_state_summary": "...", "emotion_display": "Happy", ...}},
        {"time": 5.0, "state": {"convo_state_summary": "...", ...}}
    ]
        """
    )

    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', help='Output video file path')
    parser.add_argument('--fps', type=int, default=None,
                        help='Output FPS (default: same as input)')
    parser.add_argument('--audio', action='store_true',
                        help='Process audio for transcription')
    parser.add_argument('--preview', action='store_true',
                        help='Show preview window while rendering')
    parser.add_argument('--script', type=str, default=None,
                        help='JSON file with scripted state changes')
    parser.add_argument('--no-mock', action='store_true',
                        help='Disable mock state (overlay shows static content)')
    parser.add_argument('--effect-start', type=float, default=0.0,
                        help='Time in seconds when focus effect starts (default: 0.0)')

    args = parser.parse_args()

    success = render_demo_video(
        args.input,
        args.output,
        output_fps=args.fps,
        process_audio=args.audio,
        show_preview=args.preview,
        script_file=args.script,
        use_mock_state=not args.no_mock,
        effect_start=args.effect_start
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
