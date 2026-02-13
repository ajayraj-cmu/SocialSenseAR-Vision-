"""Story Renderer — composites masks/effects onto video frames.

Quick Python/OpenCV renderer for previewing stories before Unity rendering.

Usage:
    # Preview with live window
    python -m tools.story_renderer \\
        --story story_dir/ \\
        --output preview.mp4 \\
        --mode outline \\
        --show-preview

    # Render with effects from timeline
    python -m tools.story_renderer \\
        --story story_dir/ \\
        --output demo.mp4 \\
        --mode all

    # Override effects from a separate file
    python -m tools.story_renderer \\
        --story story_dir/ \\
        --output demo.mp4 \\
        --effects-file custom_effects.json
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.encoding.rle import decode_rle
from tools.story_schema import load_story, read_mask_bytes, resolve_effects

logger = logging.getLogger(__name__)

# Color palette matching server/dashboard.py
BRIGHT_COLORS = [
    (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
    (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
    (0, 200, 100), (100, 255, 200),
]

# Asset class → color (matching Unity OverlayRenderer)
ASSET_COLORS = {
    "person": (255, 130, 70),     # blue-ish
    "furniture": (70, 200, 70),   # green
    "lighting": (50, 220, 255),   # yellow
    "electronics": (220, 220, 50),  # cyan
    "screen": (220, 220, 50),
    "structural": (160, 160, 160),  # grey
}


# ---------------------------------------------------------------------------
# Effect application (matches server/dashboard.py lines 152-204)
# ---------------------------------------------------------------------------

def apply_effect(frame, mask_bool, effect):
    """Apply a visual effect to the frame within the mask region."""
    effect_type = effect.get("effect_type", "")
    intensity = effect.get("intensity", 1.0)
    invert = effect.get("invert", False)

    if invert:
        mask_bool = ~mask_bool

    if effect_type == "blur":
        ksize = max(3, int(51 * intensity)) | 1
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        frame[mask_bool] = blurred[mask_bool]

    elif effect_type == "dim":
        dimmed = (frame.astype(np.float32) * (1.0 - 0.7 * intensity)).clip(0, 255).astype(np.uint8)
        frame[mask_bool] = dimmed[mask_bool]

    elif effect_type == "pixelate":
        block = max(4, int(20 * intensity))
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[mask_bool] = pixelated[mask_bool]

    elif effect_type == "highlight":
        bright = cv2.addWeighted(frame, 1.0, np.full_like(frame, 255), 0.3 * intensity, 0)
        tint = bright.copy()
        tint[:, :, 0] = np.clip(tint[:, :, 0].astype(np.int16) + int(40 * intensity), 0, 255).astype(np.uint8)
        tint[:, :, 2] = np.clip(tint[:, :, 2].astype(np.int16) + int(40 * intensity), 0, 255).astype(np.uint8)
        frame[mask_bool] = tint[mask_bool]

    elif effect_type == "color":
        # Support both color_hex="#RRGGBB" and color=[R,G,B] array
        color_rgb = effect.get("color")
        if color_rgb and isinstance(color_rgb, (list, tuple)) and len(color_rgb) == 3:
            r, g, b = int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])
        else:
            color_hex = effect.get("color_hex", "") or ""
            h_str = color_hex.lstrip("#")
            if len(h_str) < 6:
                h_str = "ADD8E6"  # default pastel blue
            r, g, b = int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16)
        tint = np.array([b, g, r], dtype=np.float32)
        blended = frame.astype(np.float32) * (1 - intensity) + tint * intensity
        frame[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)[mask_bool]

    elif effect_type == "frosted_glass":
        ksize = max(3, int(51 * intensity)) | 1
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        frame[mask_bool] = blurred[mask_bool]

    elif effect_type == "redact":
        block = max(4, int(20 * intensity))
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[mask_bool] = pixelated[mask_bool]

    elif effect_type in ("grayscale", "desaturate"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(gray_bgr, intensity, frame, 1 - intensity, 0)
        frame[mask_bool] = blended[mask_bool]

    return frame


def draw_outline(frame, mask_u8, seg_index, effect_type="", intensity=1.0):
    """Draw colored contours around the mask."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    color = BRIGHT_COLORS[seg_index % len(BRIGHT_COLORS)]

    if effect_type == "outline":
        thickness = max(3, int(8 * intensity))
        cv2.drawContours(frame, contours, -1, (0, 0, 0), thickness + 3)
        cv2.drawContours(frame, contours, -1, color, thickness)
    else:
        cv2.drawContours(frame, contours, -1, (0, 0, 0), 5)
        cv2.drawContours(frame, contours, -1, color, 2)

    return frame


def draw_filled(frame, mask_u8, asset_class, alpha=0.3):
    """Draw semi-transparent colored fill within the mask."""
    color = ASSET_COLORS.get(asset_class, (200, 200, 200))
    overlay = frame.copy()
    overlay[mask_u8 > 128] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_label(frame, seg, fw, fh, seg_index, effect=None):
    """Draw a label at the segment centroid."""
    cx = int(seg.get("center_x", 0.5) * fw)
    cy = int(seg.get("center_y", 0.5) * fh)
    label = seg.get("label", "") or seg.get("asset_class", "")

    if effect and effect.get("effect_type") and effect["effect_type"] != "none":
        label = f"{label} [{effect['effect_type']}]"

    if not label:
        return frame

    color = BRIGHT_COLORS[seg_index % len(BRIGHT_COLORS)]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (cx - 3, cy - th - 4), (cx + tw + 3, cy + 4), (0, 0, 0), -1)
    cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Audio muxing
# ---------------------------------------------------------------------------

def _mux_audio(source_video: str, rendered_video: str):
    """Mux audio from the original video into the rendered output via ffmpeg."""
    out_path = Path(rendered_video)
    tmp_path = out_path.with_stem(out_path.stem + "_tmp")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", rendered_video,
            "-i", source_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and tmp_path.exists():
            tmp_path.replace(out_path)
            print(f"Audio muxed from source video into: {rendered_video}")
        else:
            # ffmpeg failed — keep the video without audio
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"Warning: Could not mux audio (ffmpeg returned {result.returncode})")
            if result.stderr:
                print(f"  ffmpeg stderr: {result.stderr[:200]}")
    except FileNotFoundError:
        print("Warning: ffmpeg not found — output has no audio")
    except Exception as e:
        print(f"Warning: Audio mux failed: {e}")
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Rendering loop
# ---------------------------------------------------------------------------

def render_story(
    story_dir: str | Path,
    video_path: str | None,
    output_path: str,
    mode: str = "outline",
    effects_file: str | None = None,
    show_preview: bool = False,
    confidence_min: float = 0.10,
):
    """Render a story onto video frames."""
    story_dir = Path(story_dir)
    story = load_story(story_dir)
    meta = story.metadata

    # Use video from metadata if not specified
    if not video_path:
        video_path = meta.source_video
    if not Path(video_path).is_absolute():
        # Try relative to story dir, then CWD
        candidates = [story_dir / video_path, Path(video_path)]
        for c in candidates:
            if c.exists():
                video_path = str(c)
                break

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or meta.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Rendering: {width}x{height} @ {fps:.1f}fps, mode={mode}")

    # Load effects override
    effects_timeline = story.effects_timeline
    if effects_file:
        with open(effects_file, "r") as f:
            effects_timeline = json.load(f)
        print(f"Loaded effects from: {effects_file}")

    # Video writer — use ffmpeg pipe for H.264 (much better quality than mp4v)
    ffmpeg_proc = None
    out = None
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print("Using H.264 encoder (ffmpeg pipe)")
    except FileNotFoundError:
        print("Warning: ffmpeg not found, falling back to mp4v (lower quality)")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if show_preview:
        cv2.namedWindow("Story Preview", cv2.WINDOW_NORMAL)

    frame_idx = 0
    total_frames = len(story.frames)

    for frame_entry in story.frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        display = frame_bgr.copy()

        # Resolve effects for this frame
        active_effects = resolve_effects(effects_timeline, frame_entry["frame_idx"])

        # Decode all segment masks for this frame
        # Combine masks with same label (e.g. multiple "hand" segments)
        decoded_masks = {}  # label -> (mask_u8, mask_bool, seg, seg_index)
        for seg_i, seg in enumerate(frame_entry.get("segments", [])):
            conf = seg.get("confidence", 1.0)
            if conf < confidence_min:
                continue

            mask_len = seg.get("mask_length", 0)
            mask_w = seg.get("mask_width", 0)
            mask_h = seg.get("mask_height", 0)
            if mask_len <= 0 or mask_w <= 0 or mask_h <= 0:
                continue

            rle_bytes = read_mask_bytes(story_dir, seg["mask_offset"], mask_len)
            mask = decode_rle(rle_bytes, mask_w, mask_h)

            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

            mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)
            mask_bool = mask_u8 > 128
            label = seg.get("label", "")

            if label in decoded_masks:
                # Combine with existing mask (OR — union of all same-label segments)
                prev_u8, prev_bool, prev_seg, prev_i = decoded_masks[label]
                combined_u8 = np.maximum(prev_u8, mask_u8)
                combined_bool = prev_bool | mask_bool
                decoded_masks[label] = (combined_u8, combined_bool, seg, seg_i)
            else:
                decoded_masks[label] = (mask_u8, mask_bool, seg, seg_i)

        # Apply effects from timeline (handles invert and unmatched targets)
        # Synonym mapping: effect target → segment labels that match
        LABEL_SYNONYMS = {"face": ["face", "head"], "head": ["head", "face"]}

        if mode in ("effects", "all") and active_effects:
            for target, effect in active_effects.items():
                effect_type = effect.get("effect_type", "")
                if not effect_type or effect_type in ("none", "outline"):
                    continue

                # Find matching mask: try exact target, then synonyms
                matched_label = None
                if target in decoded_masks:
                    matched_label = target
                else:
                    for syn in LABEL_SYNONYMS.get(target, []):
                        if syn in decoded_masks:
                            matched_label = syn
                            break

                if matched_label:
                    _, mask_bool, _, _ = decoded_masks[matched_label]
                    display = apply_effect(display, mask_bool, effect)
                elif effect.get("invert", False) and decoded_masks:
                    # Invert on unknown target: treat as "everything except all masks"
                    combined = np.zeros((height, width), dtype=bool)
                    for _, (_, mb, _, _) in decoded_masks.items():
                        combined |= mb
                    bg_mask = ~combined
                    # Strip invert flag since we pre-computed the region
                    eff_copy = dict(effect)
                    eff_copy["invert"] = False
                    display = apply_effect(display, bg_mask, eff_copy)

        # Draw overlays on segments that have active effects or always in outline/all mode
        for label, (mask_u8, mask_bool, seg, seg_i) in decoded_masks.items():
            # Match effect by label or synonym
            effect = active_effects.get(label)
            if not effect:
                for eff_target, eff in active_effects.items():
                    if label in LABEL_SYNONYMS.get(eff_target, []):
                        effect = eff
                        break

            if mode in ("outline", "all"):
                # Only show fill/outline when there's an active effect on this segment
                if effect or any(e.get("invert") and label in decoded_masks for e in active_effects.values()):
                    display = draw_filled(display, mask_u8, seg.get("asset_class", ""), alpha=0.25)
                    et = effect.get("effect_type", "") if effect else "outline"
                    ei = effect.get("intensity", 1.0) if effect else 1.0
                    display = draw_outline(display, mask_u8, seg_i, et, ei)
            elif mode == "filled":
                display = draw_filled(display, mask_u8, seg.get("asset_class", ""))

            # Labels only when effect is active
            if mode != "effects" and (effect or any(e.get("invert") for e in active_effects.values())):
                display = draw_label(display, seg, width, height, seg_i, effect)

        # Show command text overlay for active voice commands
        if active_effects:
            y_offset = 60
            for target, effect in active_effects.items():
                et = effect.get("effect_type", "")
                inv = " (inverted)" if effect.get("invert") else ""
                text = f"{et} {target}{inv}"
                cv2.putText(display, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                y_offset += 35

        if ffmpeg_proc:
            ffmpeg_proc.stdin.write(display.tobytes())
        elif out:
            out.write(display)

        if show_preview:
            cv2.imshow("Story Preview", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  [{pct:.1f}%] Frame {frame_idx}/{total_frames}")

    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        stderr = ffmpeg_proc.stderr.read()
        ffmpeg_proc.wait()
        if ffmpeg_proc.returncode != 0:
            print(f"Warning: ffmpeg returned {ffmpeg_proc.returncode}")
            if stderr:
                print(f"  {stderr.decode(errors='replace')[-300:]}")
    elif out:
        out.release()
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    print(f"\nRendered {frame_idx} frames to: {output_path}")

    # Mux audio from original video into the output
    _mux_audio(str(video_path), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render a SocialSenseAR story to video")
    parser.add_argument("--story", "-s", required=True, help="Path to story directory")
    parser.add_argument("--video", "-v", default=None, help="Input video (default: from story metadata)")
    parser.add_argument("--output", "-o", default="preview.mp4", help="Output video path")
    parser.add_argument(
        "--mode", "-m", default="outline",
        choices=["outline", "filled", "effects", "all"],
        help="Render mode: outline, filled, effects, or all",
    )
    parser.add_argument("--effects-file", default=None, help="Override effects_timeline from JSON file")
    parser.add_argument("--show-preview", action="store_true", help="Show live preview window")
    parser.add_argument("--confidence-min", type=float, default=0.10, help="Min confidence to render")

    args = parser.parse_args()

    render_story(
        story_dir=args.story,
        video_path=args.video,
        output_path=args.output,
        mode=args.mode,
        effects_file=args.effects_file,
        show_preview=args.show_preview,
        confidence_min=args.confidence_min,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
