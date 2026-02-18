"""JSON-Driven Video Processor for SAM3.

Reads a user-provided JSON instruction file (no audio) and drives SAM3
to segment a pre-recorded video based purely on timed JSON directives.

Produces story.json + masks.bin in the output directory, which can then
be rendered by json_driven_renderer.py.

Usage:
    # Via Modal cloud GPU (recommended)
    python -m video_pipeline.json_driven_processor \\
        --input video_pipeline/videos/my_video.mp4 \\
        --instructions video_pipeline/my_instructions.json \\
        --output video_pipeline/output/ \\
        --server-url wss://your-modal-app.modal.run/ws

    # Via local server (must be running on port 8765)
    python -m video_pipeline.json_driven_processor \\
        --input video_pipeline/videos/my_video.mp4 \\
        --instructions video_pipeline/my_instructions.json \\
        --output video_pipeline/output/ \\
        --server-url ws://localhost:8765

    # Locally without server (requires CUDA GPU)
    python -m video_pipeline.json_driven_processor \\
        --input video_pipeline/videos/my_video.mp4 \\
        --instructions video_pipeline/my_instructions.json \\
        --output video_pipeline/output/ \\
        --local --device cuda

Instruction JSON format:
    See video_pipeline/example_instructions.json for full schema.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make the SAMARSDK workspace importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent / "invertedMaskAnimationRepo"
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction loader + validator
# ---------------------------------------------------------------------------

def load_instructions(instructions_path: str) -> dict:
    """Load and validate the user-provided JSON instructions file."""
    with open(instructions_path, "r") as f:
        data = json.load(f)

    # Basic validation
    if "events" not in data:
        raise ValueError("Instructions JSON must have a top-level 'events' list.")

    return data


def build_command_schedule(instructions: dict, fps: float) -> list[dict]:
    """Convert JSON events into a frame-indexed command schedule for SAM3.

    Only extracts SAM3 prompt directives (segment_target events).
    Image overlay and conversation_mode events are handled at render time.

    Returns list of {frame_idx, command_text} for the processor.
    """
    schedule = []
    events = instructions.get("events", [])

    for event in events:
        event_type = event.get("type", "")
        timestamp_s = event.get("timestamp_s", None)
        frame_idx = event.get("frame_idx", None)

        # Resolve frame index from timestamp if not provided
        if frame_idx is None and timestamp_s is not None:
            frame_idx = int(float(timestamp_s) * fps)
        elif frame_idx is None:
            logger.warning(f"Event missing both timestamp_s and frame_idx, skipping: {event}")
            continue

        frame_idx = int(frame_idx)

        if event_type == "segment_target":
            # Tell SAM3 what to segment
            targets = event.get("targets", [])
            for target in targets:
                schedule.append({
                    "frame_idx": frame_idx,
                    "command_text": target.strip().lower(),
                })
            print(f"  SAM3 prompt @frame {frame_idx} ({frame_idx/fps:.2f}s): {targets}")

        elif event_type in ("overlay_image", "conversation_mode", "effect"):
            # These are render-time only — no SAM3 prompt injection needed,
            # but we still need SAM3 to have a target for the mask.
            # Auto-inject a segment_target for the event's subject if provided.
            subject = event.get("subject", "")
            if subject and subject.lower() not in ("", "none"):
                schedule.append({
                    "frame_idx": frame_idx,
                    "command_text": subject.strip().lower(),
                })
                print(f"  SAM3 auto-prompt @frame {frame_idx} ({frame_idx/fps:.2f}s): '{subject}' (from {event_type})")

        else:
            logger.debug(f"Skipping non-SAM3 event type: {event_type}")

    return schedule


# ---------------------------------------------------------------------------
# WebSocket processing (Modal / local server)
# ---------------------------------------------------------------------------

async def process_via_websocket(
    video_path: str,
    output_dir: Path,
    server_url: str,
    command_schedule: list[dict],
    frame_skip: int = 1,
    jpeg_quality: int = 70,
    progress: bool = True,
) -> tuple[list, dict, dict]:
    """Stream video frames to SAM3 server, collect masks."""
    import websockets

    # Import protobuf from the repo
    sys.path.insert(0, str(_REPO_ROOT))
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
    print(f"Command schedule: {len(command_schedule)} SAM3 prompts")

    # Frame → commands lookup
    cmd_by_frame: dict[int, list[str]] = {}
    for cmd in command_schedule:
        fi = int(cmd["frame_idx"])
        cmd_by_frame.setdefault(fi, []).append(cmd["command_text"])

    masks_path = output_dir / "masks.bin"
    masks_file = open(masks_path, "wb")
    mask_offset = 0

    story_frames = []
    tracks_seen = {}
    last_segments_data = None

    t_start = time.time()

    # Connect with retry (Modal cold-start can take 30-90s)
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

    print("Connected.")

    try:
        for frame_idx in range(total_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Inject SAM3 prompts at scheduled frames
            if frame_idx in cmd_by_frame:
                for cmd_text in cmd_by_frame[frame_idx]:
                    ctrl = pb.ClientMessage()
                    ctrl.frame_id = frame_idx
                    ctrl.timestamp_ms = time.time() * 1000
                    ctrl.control.command = cmd_text
                    await ws.send(ctrl.SerializeToString())
                    print(f"  >> SAM3 prompt @frame {frame_idx}: \"{cmd_text}\"")

            run_sam = (frame_idx % frame_skip == 0)

            if run_sam:
                flipped = cv2.flip(frame_bgr, 0)
                success, jpeg_buf = cv2.imencode(
                    ".jpg", flipped,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                if not success:
                    continue

                msg = pb.ClientMessage()
                msg.frame_id = frame_idx
                msg.timestamp_ms = time.time() * 1000
                msg.frame.jpeg_data = jpeg_buf.tobytes()
                msg.frame.width = width
                msg.frame.height = height
                msg.frame.quality = jpeg_quality

                await ws.send(msg.SerializeToString())
                resp_bytes = await ws.recv()

                resp = pb.ServerMessage()
                resp.ParseFromString(resp_bytes)

                segments_data = []
                for seg in resp.segments:
                    rle_bytes = bytes(seg.rle_mask) if seg.rle_mask else b""
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

            story_frames.append({
                "frame_idx": frame_idx,
                "timestamp_s": round(frame_idx / fps, 4),
                "sam_ran": run_sam,
                "segments": segments_data,
            })

            if progress and frame_idx % 50 == 0:
                elapsed = time.time() - t_start
                pct = (frame_idx + 1) / total_frames * 100
                fps_proc = (frame_idx + 1) / max(elapsed, 0.1)
                eta = (total_frames - frame_idx - 1) / max(fps_proc, 0.01)
                print(f"  [{pct:5.1f}%] Frame {frame_idx+1}/{total_frames} | {fps_proc:.1f} fps | ETA {eta:.0f}s")
    finally:
        await ws.close()

    masks_file.close()
    cap.release()

    processing_time = time.time() - t_start

    rle_w, rle_h = 0, 0
    for f in story_frames:
        for s in f.get("segments", []):
            if s.get("mask_width") and s.get("mask_height"):
                rle_w, rle_h = s["mask_width"], s["mask_height"]
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
# Local processing (requires CUDA GPU)
# ---------------------------------------------------------------------------

def process_locally(
    video_path: str,
    output_dir: Path,
    command_schedule: list[dict],
    device: str = "cuda",
    frame_skip: int = 1,
    progress: bool = True,
) -> tuple[list, dict, dict]:
    """Process locally via PipelineOrchestrator."""
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

    # Pre-seed any prompts from schedule that start at frame 0
    cmd_by_frame: dict[int, list[str]] = {}
    for cmd in command_schedule:
        fi = int(cmd["frame_idx"])
        cmd_by_frame.setdefault(fi, []).append(cmd["command_text"])

    if 0 in cmd_by_frame:
        for prompt in cmd_by_frame[0]:
            orchestrator._segmenter.add_prompt(prompt)

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

        # Inject prompts at scheduled frames
        if frame_idx in cmd_by_frame and frame_idx > 0:
            for prompt in cmd_by_frame[frame_idx]:
                orchestrator._segmenter.add_prompt(prompt)
                print(f"  >> Local SAM3 prompt @frame {frame_idx}: \"{prompt}\"")

        run_sam = (frame_idx % frame_skip == 0)

        if run_sam:
            fh, fw = frame_bgr.shape[:2]
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

        story_frames.append({
            "frame_idx": frame_idx,
            "timestamp_s": round(frame_idx / fps, 4),
            "sam_ran": run_sam,
            "segments": segments_data,
        })

        if progress and frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            pct = (frame_idx + 1) / total_frames * 100
            fps_proc = (frame_idx + 1) / max(elapsed, 0.1)
            eta = (total_frames - frame_idx - 1) / max(fps_proc, 0.01)
            print(f"  [{pct:5.1f}%] Frame {frame_idx+1}/{total_frames} | {fps_proc:.1f} fps | ETA {eta:.0f}s")

    masks_file.close()
    cap.release()
    orchestrator.shutdown()

    processing_time = time.time() - t_start

    rle_w, rle_h = 0, 0
    for f in story_frames:
        for s in f.get("segments", []):
            if s.get("mask_width") and s.get("mask_height"):
                rle_w, rle_h = s["mask_width"], s["mask_height"]
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
# Story assembler
# ---------------------------------------------------------------------------

def assemble_and_save(
    instructions: dict,
    story_frames: list,
    tracks: dict,
    video_info: dict,
    output_dir: Path,
    input_path: Path,
):
    """Build story.json from frames + instruction events."""
    # Re-use story_schema from the repo
    sys.path.insert(0, str(_REPO_ROOT))
    from tools.story_schema import Story, StoryMetadata, make_timestamp, save_story

    fps = video_info["fps"]

    # Build effects_timeline from instruction events (for render time)
    effects_timeline = []
    for event in instructions.get("events", []):
        event_type = event.get("type", "")
        timestamp_s = event.get("timestamp_s", None)
        frame_idx = event.get("frame_idx", None)

        if frame_idx is None and timestamp_s is not None:
            frame_idx = int(float(timestamp_s) * fps)
        elif frame_idx is None:
            continue
        frame_idx = int(frame_idx)

        end_frame_idx = None
        end_ts = event.get("end_timestamp_s")
        end_fi = event.get("end_frame_idx")
        if end_fi is not None:
            end_frame_idx = int(end_fi)
        elif end_ts is not None:
            end_frame_idx = int(float(end_ts) * fps)

        if event_type == "effect":
            effects_timeline.append({
                "start_frame": frame_idx,
                "end_frame": end_frame_idx,
                "target": event.get("subject", ""),
                "effect_type": event.get("effect", "outline"),
                "intensity": float(event.get("intensity", 1.0)),
                "color_hex": event.get("color_hex", ""),
                "invert": bool(event.get("invert", False)),
                "source": "json",
            })

        elif event_type == "overlay_image":
            # Custom effect type handled by json_driven_renderer.py
            effects_timeline.append({
                "start_frame": frame_idx,
                "end_frame": end_frame_idx,
                "target": event.get("subject", ""),
                "effect_type": "overlay_image",
                "image_name": event.get("image_name", ""),
                "intensity": float(event.get("opacity", 1.0)),
                "invert": bool(event.get("invert", False)),
                "source": "json",
            })

        elif event_type == "conversation_mode":
            # Custom effect: grayscale converge on nearest person
            effects_timeline.append({
                "start_frame": frame_idx,
                "end_frame": end_frame_idx,
                "target": event.get("subject", "person"),
                "effect_type": "conversation_mode",
                "animation_duration_frames": int(event.get("animation_duration_s", 1.5) * fps),
                "blur_intensity": float(event.get("blur_intensity", 1.0)),
                "source": "json",
            })

    rle_scale = 0.75
    if video_info["rle_width"] and video_info["width"]:
        rle_scale = round(video_info["rle_width"] / video_info["width"], 4)

    story = Story(
        version=1,
        metadata=StoryMetadata(
            source_video=str(input_path.name),
            fps=fps,
            frame_count=video_info["frame_count"],
            width=video_info["width"],
            height=video_info["height"],
            rle_width=video_info["rle_width"],
            rle_height=video_info["rle_height"],
            rle_scale=rle_scale,
            pipeline_mode="sam3_json_driven",
            processing_time_s=video_info["processing_time_s"],
            created_at=make_timestamp(),
        ),
        prompts=[],
        tracks=tracks,
        frames=story_frames,
        voice_events=[],
        effects_timeline=effects_timeline,
    )

    story_path = save_story(story, output_dir)

    # Also save a copy of the instructions into the output dir for reference
    inst_copy = output_dir / "instructions_used.json"
    with open(inst_copy, "w") as f:
        json.dump(instructions, f, indent=2)

    return story_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JSON-Driven SAM3 Video Processor — no audio, instruction-based.",
    )
    parser.add_argument("--input", "-i", required=True, help="Input video file (MP4, MOV, AVI)")
    parser.add_argument("--instructions", "-j", required=True, help="Path to JSON instructions file")
    parser.add_argument("--output", "-o", default="video_pipeline/output/", help="Output directory")
    parser.add_argument("--server-url", default=None, help="SAM3 WebSocket server URL")
    parser.add_argument("--local", action="store_true", help="Run locally (requires GPU)")
    parser.add_argument("--device", default="cuda", help="Device for local mode (cuda/cpu)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality (1-100)")
    parser.add_argument("--progress", action="store_true", default=True, help="Show progress")

    args = parser.parse_args()

    if not args.local and not args.server_url:
        parser.error("Must specify --server-url or --local")

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input video not found: {input_path}")

    instructions_path = Path(args.instructions)
    if not instructions_path.exists():
        parser.error(f"Instructions file not found: {instructions_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"JSON-Driven SAM3 Video Processor")
    print(f"{'='*60}")
    print(f"Input:        {input_path}")
    print(f"Instructions: {instructions_path}")
    print(f"Output:       {output_dir}")
    print()

    # Load instructions
    instructions = load_instructions(str(instructions_path))
    print(f"Loaded {len(instructions['events'])} instruction events.")

    # Get video FPS first
    cap = cv2.VideoCapture(str(input_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Build SAM3 command schedule from instructions
    print(f"\nBuilding SAM3 command schedule (@ {video_fps:.1f} fps)...")
    command_schedule = build_command_schedule(instructions, video_fps)
    print(f"Schedule has {len(command_schedule)} SAM3 prompt injections.")

    # Process video
    print(f"\n{'='*60}")
    print(f"Processing video frames with SAM3...")
    print(f"{'='*60}\n")

    if args.local:
        story_frames, tracks, video_info = process_locally(
            str(input_path), output_dir, command_schedule,
            device=args.device,
            frame_skip=args.frame_skip,
            progress=args.progress,
        )
    else:
        story_frames, tracks, video_info = asyncio.run(
            process_via_websocket(
                str(input_path), output_dir, args.server_url,
                command_schedule,
                frame_skip=args.frame_skip,
                jpeg_quality=args.jpeg_quality,
                progress=args.progress,
            )
        )

    print(f"\nProcessed {len(story_frames)} frames, {len(tracks)} tracked objects.")
    for tid, info in tracks.items():
        print(f"  {tid}: {info['label']} (frames {info['first_frame']}-{info['last_frame']})")

    # Assemble story
    print(f"\n{'='*60}")
    print(f"Saving story...")
    print(f"{'='*60}\n")

    story_path = assemble_and_save(
        instructions, story_frames, tracks, video_info, output_dir, input_path,
    )

    masks_size = (output_dir / "masks.bin").stat().st_size
    print(f"story.json: {story_path.stat().st_size / 1024:.1f} KB")
    print(f"masks.bin:  {masks_size / 1024:.1f} KB")
    print(f"Frames:     {len(story_frames)}")
    print(f"Tracks:     {len(tracks)}")
    print(f"\nDone! Now render with:")
    print(f"  python -m video_pipeline.json_driven_renderer \\")
    print(f"      --story {output_dir} \\")
    print(f"      --video {input_path} \\")
    print(f"      --output {output_dir}/rendered.mp4")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
