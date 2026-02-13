"""Video Processor — extracts a "story" from a video file.

Sends video frames to a SocialSenseAR server (local or Modal) via WebSocket,
collects segmentation masks, and optionally transcribes voice commands.

Usage:
    # Process via Modal cloud GPU
    python -m tools.video_processor \\
        --input video.mp4 \\
        --output story_dir/ \\
        --server-url wss://your-modal-app.modal.run/ws \\
        --transcribe-audio \\
        --progress

    # Process via local server
    python -m tools.video_processor \\
        --input video.mp4 \\
        --output story_dir/ \\
        --server-url ws://localhost:8765 \\
        --progress

    # Process locally (no server, requires CUDA GPU)
    python -m tools.video_processor \\
        --input video.mp4 \\
        --output story_dir/ \\
        --local --device cuda \\
        --prompts "person,laptop,chair" \\
        --progress
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env so GEMINI_API_KEY, OPENAI_API_KEY etc. are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from tools.story_schema import (
    Story, StoryMetadata, make_timestamp, save_story, resolve_effects,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame processing — WebSocket mode (Modal or local server)
# ---------------------------------------------------------------------------

async def process_via_websocket(
    video_path: str,
    output_dir: Path,
    server_url: str,
    frame_skip: int = 1,
    jpeg_quality: int = 70,
    progress: bool = False,
    command_schedule: list[dict] | None = None,
):
    """Send video frames to server via WebSocket, collect story.

    command_schedule: list of {frame_idx, command_text} dicts.
        At each frame_idx, a ControlPayload is sent BEFORE the frame,
        so SAM3 adds/removes prompts dynamically (matching real-time flow).
    """
    import websockets
    from server.proto import socialsense_pb2 as pb

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Server: {server_url}")

    # Build frame→commands lookup from schedule
    cmd_by_frame = {}
    if command_schedule:
        for cmd in command_schedule:
            fi = cmd["frame_idx"]
            cmd_by_frame.setdefault(fi, []).append(cmd["command_text"])
        print(f"Command schedule: {len(command_schedule)} commands at {len(cmd_by_frame)} frames")

    # Open masks.bin for writing
    masks_path = output_dir / "masks.bin"
    masks_file = open(masks_path, "wb")
    mask_offset = 0

    story_frames = []
    tracks_seen = {}  # track_id -> {label, asset_class, first_frame, last_frame}
    last_segments_data = None  # For frame_skip reuse

    t_start = time.time()

    # Modal containers can take 30-90s to cold-start — retry with backoff
    ws = None
    for attempt in range(5):
        try:
            print(f"Connecting (attempt {attempt+1}/5)...")
            ws = await asyncio.wait_for(
                websockets.connect(
                    server_url,
                    max_size=10 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=120,
                    open_timeout=120,
                ),
                timeout=130,
            )
            break
        except (TimeoutError, OSError, ConnectionRefusedError) as e:
            if attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"  Connection failed ({e}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"Could not connect after 5 attempts: {e}") from e

    print(f"Connected to {server_url}")

    try:
        for frame_idx in range(total_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Inject voice commands at the right frame (before the frame)
            if frame_idx in cmd_by_frame:
                for cmd_text in cmd_by_frame[frame_idx]:
                    ctrl = pb.ClientMessage()
                    ctrl.frame_id = frame_idx
                    ctrl.timestamp_ms = time.time() * 1000
                    ctrl.control.command = cmd_text
                    await ws.send(ctrl.SerializeToString())
                    print(f"  >> Injected command @frame {frame_idx}: \"{cmd_text}\"")

            run_sam = (frame_idx % frame_skip == 0)

            if run_sam:
                # Y-flip: server expects upside-down JPEGs (Quest convention)
                # Recorded video is right-side-up, so flip before sending
                flipped = cv2.flip(frame_bgr, 0)
                success, jpeg_buf = cv2.imencode(
                    ".jpg", flipped,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not success:
                    continue

                # Build protobuf message
                msg = pb.ClientMessage()
                msg.frame_id = frame_idx
                msg.timestamp_ms = time.time() * 1000
                msg.frame.jpeg_data = jpeg_buf.tobytes()
                msg.frame.width = width
                msg.frame.height = height
                msg.frame.quality = jpeg_quality

                # Send and receive
                await ws.send(msg.SerializeToString())
                resp_bytes = await ws.recv()

                resp = pb.ServerMessage()
                resp.ParseFromString(resp_bytes)

                # Extract segments
                segments_data = []
                for seg in resp.segments:
                    rle_bytes = bytes(seg.rle_mask) if seg.rle_mask else b""

                    # Write mask to binary file
                    if rle_bytes:
                        masks_file.write(rle_bytes)

                    seg_entry = {
                        "track_id": seg.track_id or "",
                        "label": seg.label or "",
                        "asset_class": seg.asset_class or "",
                        "confidence": round(float(seg.confidence), 4),
                        "bbox": [
                            round(float(seg.bbox.x_min), 4),
                            round(float(seg.bbox.y_min), 4),
                            round(float(seg.bbox.x_max), 4),
                            round(float(seg.bbox.y_max), 4),
                        ] if seg.bbox else [0, 0, 0, 0],
                        "center_x": round(float(seg.center_x), 4),
                        "center_y": round(float(seg.center_y), 4),
                        "mask_offset": mask_offset,
                        "mask_length": len(rle_bytes),
                        "mask_width": int(seg.mask_width),
                        "mask_height": int(seg.mask_height),
                    }
                    mask_offset += len(rle_bytes)
                    segments_data.append(seg_entry)

                    # Track summary
                    tid = seg.track_id or ""
                    if tid:
                        if tid not in tracks_seen:
                            tracks_seen[tid] = {
                                "label": seg.label or "",
                                "asset_class": seg.asset_class or "",
                                "first_frame": frame_idx,
                                "last_frame": frame_idx,
                            }
                        else:
                            tracks_seen[tid]["last_frame"] = frame_idx
                            if seg.label:
                                tracks_seen[tid]["label"] = seg.label

                last_segments_data = segments_data
            else:
                # Reuse previous frame's segments (with same mask offsets)
                segments_data = last_segments_data or []

            frame_entry = {
                "frame_idx": frame_idx,
                "timestamp_s": round(frame_idx / fps, 4),
                "sam_ran": run_sam,
                "segments": segments_data,
            }
            story_frames.append(frame_entry)

            if progress and frame_idx % 50 == 0:
                elapsed = time.time() - t_start
                pct = (frame_idx + 1) / total_frames * 100
                fps_proc = (frame_idx + 1) / max(elapsed, 0.1)
                eta = (total_frames - frame_idx - 1) / max(fps_proc, 0.01)
                print(
                    f"  [{pct:5.1f}%] Frame {frame_idx+1}/{total_frames} "
                    f"| {fps_proc:.1f} fps | ETA {eta:.0f}s"
                )
    finally:
        await ws.close()

    masks_file.close()
    cap.release()

    processing_time = time.time() - t_start

    # Compute RLE dimensions from first segment found
    rle_w, rle_h = 0, 0
    for f in story_frames:
        for s in f.get("segments", []):
            if s.get("mask_width") and s.get("mask_height"):
                rle_w = s["mask_width"]
                rle_h = s["mask_height"]
                break
        if rle_w:
            break

    return story_frames, tracks_seen, {
        "fps": fps,
        "frame_count": total_frames,
        "width": width,
        "height": height,
        "rle_width": rle_w,
        "rle_height": rle_h,
        "processing_time_s": round(processing_time, 1),
    }


# ---------------------------------------------------------------------------
# Frame processing — Local mode (direct orchestrator, requires GPU)
# ---------------------------------------------------------------------------

def process_locally(
    video_path: str,
    output_dir: Path,
    device: str = "cuda",
    prompts: list[str] | None = None,
    frame_skip: int = 1,
    progress: bool = False,
):
    """Process video frames locally using PipelineOrchestrator."""
    from server.config import ServerConfig
    from server.pipeline.orchestrator import PipelineOrchestrator

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Processing locally on {device}")

    config = ServerConfig()
    config.device = device
    config.audio_enabled = False
    config.pipeline_mode = "sam3"

    orchestrator = PipelineOrchestrator(config)
    orchestrator.initialize()

    # Set prompts if specified
    if prompts:
        for p in prompts:
            orchestrator._segmenter.add_prompt(p)

    masks_path = output_dir / "masks.bin"
    masks_file = open(masks_path, "wb")
    mask_offset = 0

    story_frames = []
    tracks_seen = {}
    last_segments_data = None

    t_start = time.time()

    for frame_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        fh, fw = frame_bgr.shape[:2]
        run_sam = (frame_idx % frame_skip == 0)

        if run_sam:
            result = orchestrator.process_frame_sync(frame_bgr, fw, fh)

            segments_data = []
            for seg in result.segments:
                rle_bytes = seg.rle_mask or b""

                if rle_bytes:
                    masks_file.write(rle_bytes)

                seg_entry = {
                    "track_id": seg.track_id or "",
                    "label": seg.label or "",
                    "asset_class": seg.asset_class or "",
                    "confidence": round(float(seg.confidence), 4),
                    "bbox": [round(float(v), 4) for v in seg.bbox] if seg.bbox else [0, 0, 0, 0],
                    "center_x": round(float(seg.center_x), 4),
                    "center_y": round(float(seg.center_y), 4),
                    "mask_offset": mask_offset,
                    "mask_length": len(rle_bytes),
                    "mask_width": int(seg.mask_width),
                    "mask_height": int(seg.mask_height),
                }
                mask_offset += len(rle_bytes)
                segments_data.append(seg_entry)

                tid = seg.track_id or ""
                if tid:
                    if tid not in tracks_seen:
                        tracks_seen[tid] = {
                            "label": seg.label or "",
                            "asset_class": seg.asset_class or "",
                            "first_frame": frame_idx,
                            "last_frame": frame_idx,
                        }
                    else:
                        tracks_seen[tid]["last_frame"] = frame_idx
                        if seg.label:
                            tracks_seen[tid]["label"] = seg.label

            last_segments_data = segments_data
        else:
            segments_data = last_segments_data or []

        frame_entry = {
            "frame_idx": frame_idx,
            "timestamp_s": round(frame_idx / fps, 4),
            "sam_ran": run_sam,
            "segments": segments_data,
        }
        story_frames.append(frame_entry)

        if progress and frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            pct = (frame_idx + 1) / total_frames * 100
            fps_proc = (frame_idx + 1) / max(elapsed, 0.1)
            eta = (total_frames - frame_idx - 1) / max(fps_proc, 0.01)
            print(
                f"  [{pct:5.1f}%] Frame {frame_idx+1}/{total_frames} "
                f"| {fps_proc:.1f} fps | ETA {eta:.0f}s"
            )

    masks_file.close()
    cap.release()
    orchestrator.shutdown()

    processing_time = time.time() - t_start

    rle_w, rle_h = 0, 0
    for f in story_frames:
        for s in f.get("segments", []):
            if s.get("mask_width") and s.get("mask_height"):
                rle_w = s["mask_width"]
                rle_h = s["mask_height"]
                break
        if rle_w:
            break

    return story_frames, tracks_seen, {
        "fps": fps,
        "frame_count": total_frames,
        "width": width,
        "height": height,
        "rle_width": rle_w,
        "rle_height": rle_h,
        "processing_time_s": round(processing_time, 1),
    }


# ---------------------------------------------------------------------------
# Audio transcription — extract voice commands from video
# ---------------------------------------------------------------------------

def transcribe_audio(
    video_path: str,
    fps: float,
):
    """Extract audio from video and transcribe voice commands.

    Uses full-audio Whisper transcription with word-level timestamps for
    accurate recognition of fast, back-to-back commands.

    Returns list of voice_events and auto-generated effects_timeline entries.
    """
    import re
    from server.audio.voice_agent import VoiceCommandPlanner

    print("Extracting audio from video...")

    # Extract raw PCM16 audio via ffmpeg
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-v", "quiet",
                "pipe:1",
            ],
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0:
            print(f"Warning: ffmpeg failed (return code {proc.returncode}). Skipping audio.")
            return [], []
        pcm_data = proc.stdout
    except FileNotFoundError:
        print("Warning: ffmpeg not found. Install ffmpeg to enable audio transcription.")
        return [], []
    except subprocess.TimeoutExpired:
        print("Warning: ffmpeg timed out. Skipping audio.")
        return [], []

    if not pcm_data:
        print("No audio track found in video.")
        return [], []

    sample_rate = 16000
    total_samples = len(pcm_data) // 2
    total_duration = total_samples / sample_rate
    print(f"Audio: {total_duration:.1f}s, {total_samples} samples")

    # Convert to float32 for faster-whisper
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Full-audio transcription with word timestamps
    print("Loading Whisper model...")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Warning: faster-whisper not installed. Skipping audio.")
        return [], []

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel("medium.en", device=device, compute_type=compute_type)

    print("Transcribing full audio (word-level timestamps)...")
    segments_iter, info = model.transcribe(
        samples,
        language="en",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    # Collect all words with timestamps
    all_words = []
    for segment in segments_iter:
        if segment.words:
            for w in segment.words:
                all_words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

    if not all_words:
        print("No speech detected in audio.")
        return [], []

    full_text = " ".join(w["word"] for w in all_words)
    print(f"Transcript ({len(all_words)} words): {full_text}")

    # Find wake word positions and extract commands between them
    # Build text with word indices for pattern matching
    wake_patterns = [
        r'\bhey[\s,]+vibe\b', r'\bhey[\s,]+vibes\b', r'\bhey[\s,]+bye\b',
        r'\bhey[\s,]+vive\b', r'\bhey[\s,]+five\b', r'\bhey[\s,]+vi\b',
        r'\bvibe\b', r'\bvibes\b', r'\bvive\b',
    ]
    # End-of-command markers (only "thank you" / "thanks" — "clear" is a command)
    end_patterns = [
        r'\bthank\s*you\b', r'\bthanks\b',
    ]

    # Map character positions to word indices
    char_to_word = {}
    pos = 0
    for i, w in enumerate(all_words):
        for c in range(pos, pos + len(w["word"])):
            char_to_word[c] = i
        pos += len(w["word"]) + 1  # +1 for space

    # Find all wake word occurrences
    wake_hits = []
    full_lower = full_text.lower()
    for pat in wake_patterns:
        for m in re.finditer(pat, full_lower):
            # Find the word index AFTER the wake word
            end_char = m.end()
            # Skip to the next word after the wake phrase
            after_idx = None
            for ci in range(end_char, len(full_lower)):
                if ci in char_to_word:
                    after_idx = char_to_word[ci]
                    break
            if after_idx is not None:
                wake_hits.append((m.start(), after_idx))

    # Deduplicate and sort by position
    wake_hits = sorted(set(wake_hits), key=lambda x: x[0])

    if not wake_hits:
        print("No wake words detected. Trying without wake word (treating all speech as commands)...")
        # Fall back: split on pauses (>1s gaps between words) as command boundaries
        commands = []
        cmd_start = 0
        for i in range(1, len(all_words)):
            gap = all_words[i]["start"] - all_words[i-1]["end"]
            if gap > 1.0:
                cmd_text = " ".join(w["word"] for w in all_words[cmd_start:i])
                # Use END time of last word — effect starts when sentence finishes
                cmd_time = all_words[i-1]["end"]
                if len(cmd_text.split()) >= 2:  # skip single-word fragments
                    commands.append((cmd_time, cmd_text))
                cmd_start = i
        # Last command
        cmd_text = " ".join(w["word"] for w in all_words[cmd_start:])
        cmd_time = all_words[-1]["end"]
        if len(cmd_text.split()) >= 2:
            commands.append((cmd_time, cmd_text))
    else:
        # Extract command text between each wake word and the next wake word or end marker
        commands = []
        for hi, (_, word_idx) in enumerate(wake_hits):
            # Command starts after wake word
            # Ends at next wake word, end marker, or long pause
            next_wake_idx = wake_hits[hi + 1][1] if hi + 1 < len(wake_hits) else len(all_words)

            cmd_words = []
            cmd_end_time = all_words[word_idx]["end"] if word_idx < len(all_words) else 0

            for wi in range(word_idx, min(next_wake_idx, len(all_words))):
                w = all_words[wi]
                # Stop at end markers (thank you / thanks)
                if any(re.search(ep, w["word"].lower()) for ep in [r'thank', r'thanks']):
                    break
                # Stop at long pause (>2s)
                if cmd_words and w["start"] - all_words[wi-1]["end"] > 2.0:
                    break
                cmd_words.append(w["word"])
                cmd_end_time = w["end"]  # Track end time of last word

            if cmd_words:
                cmd_text = " ".join(cmd_words)
                # Use END time of last word — effect starts when sentence finishes
                commands.append((cmd_end_time, cmd_text))

    # Clean commands: strip wake word prefix + punctuation
    cleaned = []
    for t, txt in commands:
        # Strip leading "vibe," / "Vibe," / "vibe" prefix
        clean = re.sub(r'^(?:hey[\s,]+)?(?:vibe|vibes|vive|five|bye)[\s,]*', '', txt, flags=re.IGNORECASE).strip()
        # Strip trailing punctuation
        clean = clean.rstrip('.,!?;:')
        if not clean:
            clean = txt.rstrip('.,!?;:')  # fallback to original
        cleaned.append((t, clean))
    commands = cleaned

    print(f"\nFound {len(commands)} command segments:")
    for t, txt in commands:
        print(f"  @{t:.1f}s: \"{txt}\"")

    if not commands:
        return [], []

    # Initialize Gemini planner for command parsing
    planner = VoiceCommandPlanner(
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        model="gemini-2.0-flash",
    )
    planner.initialize()

    voice_events = []
    effects_timeline = []
    known_objects = set()
    active_effects = {}

    for cmd_time, cmd_text in commands:
        cmd_frame = int(cmd_time * fps)
        print(f"\n  Parsing @{cmd_time:.1f}s (frame {cmd_frame}): \"{cmd_text}\"")

        # Detect "clear" directly — no need for Gemini
        if re.match(r'^clear\b', cmd_text, re.IGNORECASE):
            print(f"    → clear all effects (direct match)")
            voice_events.append({
                "timestamp_s": round(cmd_time, 2),
                "frame_idx": cmd_frame,
                "transcript": cmd_text,
                "command": {
                    "targets": [],
                    "effect_type": "none",
                    "intensity": 0,
                    "action": "remove",
                    "invert": False,
                },
            })
            # End all open effects
            for entry in effects_timeline:
                if entry.get("end_frame") is None:
                    entry["end_frame"] = cmd_frame
            active_effects.clear()
            continue

        plan = planner.plan_command(
            cmd_text,
            known_objects=known_objects,
            active_effects=active_effects,
        )

        # Normalize "background" → "person" with invert=True
        # (same logic as voice_agent._execute_plan, lines 1438-1460)
        # Also handle: Gemini returns invert=True with empty targets for "blur background"
        if plan.invert and not plan.targets:
            plan.targets = ["person"]
            print(f"    (invert with no targets → default target 'person')")

        if plan.targets:
            mapped_background = False
            normalized = []
            for t in plan.targets:
                t = (t or "").strip().lower()
                if t == "background":
                    mapped_background = True
                    normalized.append("person")
                elif t:
                    normalized.append(t)
            if normalized:
                plan.targets = list(dict.fromkeys(normalized))
            if mapped_background and plan.action in ("add", "change"):
                plan.invert = True
                print(f"    (normalized 'background' → 'person' invert=True)")

        command_dict = {
            "targets": plan.targets,
            "effect_type": plan.effect_type,
            "intensity": plan.intensity,
            "action": plan.action,
            "invert": plan.invert,
        }
        if plan.color_hex:
            command_dict["color_hex"] = plan.color_hex
        if plan.full_screen_filter:
            command_dict["full_screen_filter"] = plan.full_screen_filter
            command_dict["full_screen_intensity"] = plan.full_screen_intensity

        voice_events.append({
            "timestamp_s": round(cmd_time, 2),
            "frame_idx": cmd_frame,
            "transcript": cmd_text,
            "command": command_dict,
        })

        print(f"    → {plan.action} {plan.effect_type} on {plan.targets} (invert={plan.invert})")

        # Auto-generate effects_timeline entries
        for target in plan.targets:
            known_objects.add(target)

            if plan.action == "remove":
                for entry in effects_timeline:
                    if (entry["target"] == target
                            and entry.get("end_frame") is None
                            and entry["effect_type"] == plan.effect_type):
                        entry["end_frame"] = cmd_frame
                        break
            else:
                effects_timeline.append({
                    "start_frame": cmd_frame,
                    "end_frame": None,
                    "target": target,
                    "effect_type": plan.effect_type,
                    "intensity": plan.intensity,
                    "color_hex": plan.color_hex or "",
                    "invert": plan.invert,
                    "source": "voice",
                })

            if plan.action == "remove":
                active_effects.pop(target, None)
            else:
                active_effects[target] = {
                    "type": plan.effect_type,
                    "intensity": plan.intensity,
                }

    print(f"\nFound {len(voice_events)} voice commands, {len(effects_timeline)} effect entries")
    return voice_events, effects_timeline


# ---------------------------------------------------------------------------
# Main — assemble story
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process video into a SocialSenseAR story (masks + voice events)",
    )
    parser.add_argument("--input", "-i", required=True, help="Input video file (MP4, MOV, AVI)")
    parser.add_argument("--output", "-o", required=True, help="Output story directory")
    parser.add_argument(
        "--server-url", default=None,
        help="WebSocket URL of SocialSenseAR server (e.g. wss://app.modal.run/ws)",
    )
    parser.add_argument("--local", action="store_true", help="Process locally (requires GPU)")
    parser.add_argument("--device", default="cuda", help="Device for local processing (cuda/cpu)")
    parser.add_argument("--prompts", default=None, help="Comma-separated SAM3 prompts")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality (1-100)")
    parser.add_argument("--transcribe-audio", action="store_true", help="Transcribe voice commands")
    parser.add_argument("--progress", action="store_true", help="Show progress bar")

    args = parser.parse_args()

    if not args.local and not args.server_url:
        parser.error("Must specify --server-url or --local")

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [p.strip() for p in args.prompts.split(",")] if args.prompts else None

    # Get video FPS for timestamp→frame conversion
    _cap = cv2.VideoCapture(str(input_path))
    video_fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    _cap.release()

    # Phase 1: Transcribe audio FIRST (need command schedule before processing)
    voice_events = []
    voice_effects = []
    command_schedule = None

    if args.transcribe_audio:
        print(f"\n{'='*60}")
        print(f"Phase 1: Transcribing voice commands")
        print(f"{'='*60}\n")

        voice_events, voice_effects = transcribe_audio(
            str(input_path),
            fps=video_fps,
        )

        # Build command schedule: send SAM3 PROMPT NAMES only
        # Effects are handled by the renderer via effects_timeline.
        # We only tell the server which objects to track and when.
        COMMAND_DELAY_S = 0.2
        command_schedule = []
        for ve in voice_events:
            cmd = ve.get("command", {})
            delay_frames = int(COMMAND_DELAY_S * video_fps)
            cmd_frame = ve["frame_idx"] + delay_frames
            action = cmd.get("action", "add")
            targets = cmd.get("targets", [])

            if action == "remove":
                # "clear" — don't clear prompts on server, just let
                # the renderer handle effect removal via effects_timeline
                print(f"  @frame {cmd_frame}: clear effects (renderer-only)")
                continue

            # Add each target as a SAM3 prompt
            # Also send synonym prompts for hard-to-detect objects
            PROMPT_SYNONYMS = {
                "face": ["face", "head"],
                "background": ["background"],
            }
            for target in targets:
                target = target.strip().lower()
                if not target or len(target) < 2:
                    continue
                prompts_to_send = PROMPT_SYNONYMS.get(target, [target])
                for prompt in prompts_to_send:
                    command_schedule.append({
                        "frame_idx": cmd_frame,
                        "command_text": prompt,
                    })
                print(f"  Scheduled SAM3 prompt @frame {cmd_frame} ({ve['timestamp_s'] + COMMAND_DELAY_S:.1f}s): \"{target}\" → {prompts_to_send}")

    # Phase 2: Process video frames (with commands injected at right moments)
    print(f"\n{'='*60}")
    print(f"Phase 2: Processing video frames")
    print(f"{'='*60}\n")

    if args.local:
        story_frames, tracks, video_info = process_locally(
            str(input_path), output_dir,
            device=args.device,
            prompts=prompts,
            frame_skip=args.frame_skip,
            progress=args.progress,
        )
    else:
        story_frames, tracks, video_info = asyncio.run(
            process_via_websocket(
                str(input_path), output_dir,
                server_url=args.server_url,
                frame_skip=args.frame_skip,
                jpeg_quality=args.jpeg_quality,
                progress=args.progress,
                command_schedule=command_schedule,
            )
        )

    print(f"\nProcessed {len(story_frames)} frames, found {len(tracks)} tracked objects")
    for tid, info in tracks.items():
        print(f"  {tid}: {info['label']} (frames {info['first_frame']}-{info['last_frame']})")

    # Phase 3: Build effects_timeline
    # No default effects — start blank, only voice commands create effects
    all_effects = voice_effects

    # Phase 4: Assemble and save story
    print(f"\n{'='*60}")
    print(f"Saving story")
    print(f"{'='*60}\n")

    rle_scale = 0.75
    if video_info["rle_width"] and video_info["width"]:
        rle_scale = round(video_info["rle_width"] / video_info["width"], 4)

    story = Story(
        version=1,
        metadata=StoryMetadata(
            source_video=str(input_path.name),
            fps=video_info["fps"],
            frame_count=video_info["frame_count"],
            width=video_info["width"],
            height=video_info["height"],
            rle_width=video_info["rle_width"],
            rle_height=video_info["rle_height"],
            rle_scale=rle_scale,
            pipeline_mode="sam3",
            processing_time_s=video_info["processing_time_s"],
            created_at=make_timestamp(),
        ),
        prompts=list(prompts) if prompts else [],
        tracks=tracks,
        frames=story_frames,
        voice_events=voice_events,
        effects_timeline=all_effects,
    )

    story_path = save_story(story, output_dir)
    masks_size = (output_dir / "masks.bin").stat().st_size

    print(f"Story saved to: {output_dir}")
    print(f"  story.json: {story_path.stat().st_size / 1024:.1f} KB")
    print(f"  masks.bin:  {masks_size / 1024:.1f} KB")
    print(f"  Frames:     {len(story_frames)}")
    print(f"  Tracks:     {len(tracks)}")
    print(f"  Voice cmds: {len(voice_events)}")
    print(f"  Effects:    {len(all_effects)}")
    print(f"\nDone! Edit effects_timeline in story.json, then render with story_renderer.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
