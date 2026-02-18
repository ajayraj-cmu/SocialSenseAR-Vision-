"""JSON-Driven Story Renderer.

Renders a processed story (story.json + masks.bin) onto the original video.
Supports two new special effect types on top of the standard SAM3 effects:

  1. overlay_image
     Tiles a named image (from video_pipeline/overlay_images/) over the
     SAM3-segmented mask region using OpenCV, covering the full mask area.

  2. conversation_mode
     Finds the nearest detected person, keeps them in full color, and applies
     a cinematic grayscale + blur-convergence animation on everything else
     (the surrounding environment).  The animation "converges in" from the
     edges toward the person over a configurable number of frames, mimicking
     the inverted-mask AR effect from invertedMaskAnimationRepo.

Standard SAM3 effects (blur, grayscale, dim, color, highlight, pixelate,
frosted_glass, redact, outline) also work, using the same logic as the
original story_renderer.py.

Usage:
    python -m video_pipeline.json_driven_renderer \\
        --story video_pipeline/output/ \\
        --video video_pipeline/videos/my_video.mp4 \\
        --output video_pipeline/output/rendered.mp4 \\
        --overlay-dir video_pipeline/overlay_images/ \\
        --show-preview
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent / "invertedMaskAnimationRepo"
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palettes (match upstream story_renderer)
# ---------------------------------------------------------------------------
BRIGHT_COLORS = [
    (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
    (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
    (0, 200, 100), (100, 255, 200),
]
ASSET_COLORS = {
    "person": (255, 130, 70),
    "furniture": (70, 200, 70),
    "lighting": (50, 220, 255),
    "electronics": (220, 220, 50),
    "screen": (220, 220, 50),
    "structural": (160, 160, 160),
}


# ---------------------------------------------------------------------------
# Image overlay cache
# ---------------------------------------------------------------------------
_image_cache: dict[str, np.ndarray | None] = {}

def load_overlay_image(image_name: str, overlay_dir: Path) -> np.ndarray | None:
    """Load an overlay image from overlay_dir, cache it."""
    if image_name in _image_cache:
        return _image_cache[image_name]

    candidates = [
        overlay_dir / image_name,
        overlay_dir / (image_name + ".png"),
        overlay_dir / (image_name + ".jpg"),
        overlay_dir / (image_name + ".jpeg"),
        overlay_dir / (image_name + ".webp"),
    ]
    for path in candidates:
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                _image_cache[image_name] = img
                print(f"Loaded overlay image: {path}")
                return img

    logger.warning(f"Overlay image not found: {image_name} in {overlay_dir}")
    _image_cache[image_name] = None
    return None


# ---------------------------------------------------------------------------
# Tile image over mask (core overlay logic)
# ---------------------------------------------------------------------------

def apply_image_overlay(
    frame: np.ndarray,
    mask_bool: np.ndarray,
    overlay_img: np.ndarray,
    opacity: float = 1.0,
) -> np.ndarray:
    """Tile overlay_img over frame within the mask region.

    The image is tiled to fill the entire frame, then composited only inside
    the mask so it perfectly covers the segmented region regardless of shape.
    Supports RGBA overlay images (uses alpha channel for blending).
    """
    fh, fw = frame.shape[:2]
    ih, iw = overlay_img.shape[:2]

    # Tile the overlay image to cover the full frame
    tile_rows = (fh + ih - 1) // ih
    tile_cols = (fw + iw - 1) // iw
    tiled = np.tile(overlay_img, (tile_rows, tile_cols, 1) if overlay_img.ndim == 3 else (tile_rows, tile_cols))
    tiled = tiled[:fh, :fw]

    has_alpha = (overlay_img.ndim == 3 and overlay_img.shape[2] == 4)

    if has_alpha:
        # Split RGBA
        tiled_rgb = tiled[:, :, :3]
        tiled_alpha = tiled[:, :, 3].astype(np.float32) / 255.0 * opacity
        # Blend per-pixel inside mask
        result = frame.astype(np.float32)
        for c in range(3):
            result[:, :, c] = np.where(
                mask_bool,
                result[:, :, c] * (1.0 - tiled_alpha) + tiled_rgb[:, :, c].astype(np.float32) * tiled_alpha,
                result[:, :, c],
            )
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        # No alpha — flat blend with opacity
        tiled_bgr = tiled[:, :, :3] if tiled.ndim == 3 else cv2.cvtColor(tiled, cv2.COLOR_GRAY2BGR)
        result = frame.copy().astype(np.float32)
        overlay_f = tiled_bgr.astype(np.float32)
        blended = result * (1.0 - opacity) + overlay_f * opacity
        out = result.copy()
        out[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)[mask_bool]
        return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Conversation Mode animation
# ---------------------------------------------------------------------------

def apply_conversation_mode(
    frame: np.ndarray,
    person_mask_bool: np.ndarray | None,
    anim_progress: float,  # 0.0 = start of animation, 1.0 = fully settled
    blur_intensity: float = 1.0,
) -> np.ndarray:
    """Apply conversation mode effect.

    - The nearest person stays in full color (no effect on their mask).
    - Everything else (the background / surroundings) gets:
        1. Grayscale filter
        2. Gaussian blur (soft bokeh)
        3. Subtle vignette darkening toward the edges
        4. An animated "converge-in" sweep that wipes from the frame edges
           inward, revealing the grayscale+blur as the animation progresses.

    anim_progress: float 0→1
        At 0: full color everywhere (animation hasn't started yet).
        At 1: full grayscale+blur on everything except the person.

    The sweep uses a radial distance field from the person's mask centroid
    (or frame center if no person detected), so the grayscale "converges
    in" toward the subject — exactly the style described.
    """
    fh, fw = frame.shape[:2]
    result = frame.copy()

    # Grayscale + blur for background
    ksize = max(3, int(61 * blur_intensity)) | 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Blend blurred + gray together for a "frozen focus" look
    bg_effect = cv2.addWeighted(gray_bgr, 0.7, blurred, 0.3, 0)

    # Add vignette darkening
    vignette = _make_vignette(fw, fh, strength=0.35)
    bg_effect = (bg_effect.astype(np.float32) * vignette).clip(0, 255).astype(np.uint8)

    # Find centroid of the person mask (for convergence origin)
    if person_mask_bool is not None and person_mask_bool.any():
        ys, xs = np.where(person_mask_bool)
        cx = int(xs.mean())
        cy = int(ys.mean())
    else:
        cx, cy = fw // 2, fh // 2

    # Radial distance field: 0 at centroid, 1 at max corner distance
    yy, xx = np.mgrid[0:fh, 0:fw]
    dist = np.sqrt((xx - cx) ** 2.0 + (yy - cy) ** 2.0).astype(np.float32)
    max_dist = float(np.sqrt(fw**2 + fh**2))
    dist_norm = dist / max_dist  # 0→1

    # Sweep threshold: at anim_progress=0, threshold=1 (nothing affected).
    # At anim_progress=1, threshold=0 (everything affected).
    # Pixels with dist_norm > threshold get bg_effect; others stay original.
    # This makes the effect sweep inward from edges toward the person.
    threshold = 1.0 - anim_progress

    # Soft edge on the sweep (feather zone = 10% of max_dist)
    feather = 0.10
    # alpha = how much bg_effect to blend: 0=original, 1=full effect
    alpha_map = np.clip((dist_norm - threshold) / feather, 0.0, 1.0).astype(np.float32)

    # Build per-pixel blend
    orig_f = frame.astype(np.float32)
    eff_f = bg_effect.astype(np.float32)
    alpha_3c = alpha_map[:, :, np.newaxis]

    blended = orig_f * (1.0 - alpha_3c) + eff_f * alpha_3c
    result = np.clip(blended, 0, 255).astype(np.uint8)

    # Person mask stays full color (overwrite effect with original)
    if person_mask_bool is not None:
        result[person_mask_bool] = frame[person_mask_bool]

    return result


def _make_vignette(w: int, h: int, strength: float = 0.4) -> np.ndarray:
    """Return a (h, w, 1) float32 vignette weight map, bright center → dark edges."""
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    vignette = 1.0 - np.clip(dist * strength, 0, strength)
    return vignette[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Standard effect dispatcher (matches upstream story_renderer)
# ---------------------------------------------------------------------------

def apply_standard_effect(frame: np.ndarray, mask_bool: np.ndarray, effect: dict) -> np.ndarray:
    """Apply a standard SAM3 effect (blur, grayscale, dim, color, etc.)."""
    effect_type = effect.get("effect_type", "")
    intensity = float(effect.get("intensity", 1.0))
    invert = bool(effect.get("invert", False))

    active_mask = ~mask_bool if invert else mask_bool

    if effect_type == "blur":
        ksize = max(3, int(51 * intensity)) | 1
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        frame[active_mask] = blurred[active_mask]

    elif effect_type == "dim":
        dimmed = (frame.astype(np.float32) * (1.0 - 0.7 * intensity)).clip(0, 255).astype(np.uint8)
        frame[active_mask] = dimmed[active_mask]

    elif effect_type == "pixelate":
        block = max(4, int(20 * intensity))
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (max(1, fw // block), max(1, fh // block)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
        frame[active_mask] = pixelated[active_mask]

    elif effect_type == "highlight":
        bright = cv2.addWeighted(frame, 1.0, np.full_like(frame, 255), 0.3 * intensity, 0)
        tint = bright.copy()
        tint[:, :, 0] = np.clip(tint[:, :, 0].astype(np.int16) + int(40 * intensity), 0, 255).astype(np.uint8)
        tint[:, :, 2] = np.clip(tint[:, :, 2].astype(np.int16) + int(40 * intensity), 0, 255).astype(np.uint8)
        frame[active_mask] = tint[active_mask]

    elif effect_type == "color":
        color_rgb = effect.get("color")
        if color_rgb and isinstance(color_rgb, (list, tuple)) and len(color_rgb) == 3:
            r, g, b = int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])
        else:
            h_str = (effect.get("color_hex", "") or "").lstrip("#")
            if len(h_str) < 6:
                h_str = "ADD8E6"
            r, g, b = int(h_str[0:2], 16), int(h_str[2:4], 16), int(h_str[4:6], 16)
        tint = np.array([b, g, r], dtype=np.float32)
        blended = frame.astype(np.float32) * (1 - intensity) + tint * intensity
        frame[active_mask] = np.clip(blended, 0, 255).astype(np.uint8)[active_mask]

    elif effect_type in ("frosted_glass",):
        ksize = max(3, int(51 * intensity)) | 1
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        frame[active_mask] = blurred[active_mask]

    elif effect_type in ("grayscale", "desaturate"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(gray_bgr, intensity, frame, 1 - intensity, 0)
        frame[active_mask] = blended[active_mask]

    elif effect_type == "redact":
        block = max(4, int(20 * intensity))
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (max(1, fw // block), max(1, fh // block)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
        frame[active_mask] = pixelated[active_mask]

    return frame


def draw_outline(frame, mask_u8, seg_index, intensity=1.0):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame
    color = BRIGHT_COLORS[seg_index % len(BRIGHT_COLORS)]
    thickness = max(3, int(8 * intensity))
    cv2.drawContours(frame, contours, -1, (0, 0, 0), thickness + 3)
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame


def draw_filled(frame, mask_u8, asset_class, alpha=0.3):
    color = ASSET_COLORS.get(asset_class, (200, 200, 200))
    overlay = frame.copy()
    overlay[mask_u8 > 128] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def render_story(
    story_dir: Path,
    video_path: str,
    output_path: str,
    overlay_dir: Path,
    mode: str = "effects",
    show_preview: bool = False,
    confidence_min: float = 0.10,
):
    from server.encoding.rle import decode_rle
    from tools.story_schema import load_story, read_mask_bytes, resolve_effects

    story = load_story(story_dir)
    meta = story.metadata

    if not video_path:
        video_path = meta.source_video
    if not Path(video_path).is_absolute():
        for candidate in [story_dir / video_path, Path(video_path)]:
            if candidate.exists():
                video_path = str(candidate)
                break

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or meta.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = len(story.frames)

    print(f"Rendering: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Overlay images dir: {overlay_dir}")

    # Setup video writer via ffmpeg pipe (H.264) or fallback
    ffmpeg_proc = None
    out = None
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", output_path,
        ]
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        print("Using H.264 encoder (ffmpeg pipe)")
    except FileNotFoundError:
        print("Warning: ffmpeg not found, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if show_preview:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

    LABEL_SYNONYMS = {"face": ["face", "head"], "head": ["head", "face"]}

    # Track animation state per active conversation_mode effect
    # key = start_frame, value = {"start_frame", "duration_frames"}
    conv_mode_state: dict[int, dict] = {}

    frame_idx = 0
    for frame_entry in story.frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        display = frame_bgr.copy()
        fi = frame_entry["frame_idx"]

        # Resolve active effects at this frame
        active_effects = resolve_effects(story.effects_timeline, fi)

        # ----------------------------------------------------------------
        # Decode all masks for this frame
        # ----------------------------------------------------------------
        decoded_masks: dict[str, tuple] = {}  # label -> (mask_u8, mask_bool, seg, seg_i)
        for seg_i, seg in enumerate(frame_entry.get("segments", [])):
            if seg.get("confidence", 1.0) < confidence_min:
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
                pu8, pb, ps, pi = decoded_masks[label]
                decoded_masks[label] = (np.maximum(pu8, mask_u8), pb | mask_bool, seg, seg_i)
            else:
                decoded_masks[label] = (mask_u8, mask_bool, seg, seg_i)

        # ----------------------------------------------------------------
        # Apply effects
        # ----------------------------------------------------------------
        conversation_mode_applied = False

        for target, effect in active_effects.items():
            effect_type = effect.get("effect_type", "")

            # --- conversation_mode ---
            if effect_type == "conversation_mode":
                conversation_mode_applied = True
                start_frame = effect.get("start_frame", fi)
                duration = int(effect.get("animation_duration_frames", int(fps * 1.5)))
                blur_int = float(effect.get("blur_intensity", 1.0))

                frames_since_start = fi - start_frame
                anim_progress = min(1.0, frames_since_start / max(duration, 1))

                # Find the "nearest" person mask
                # "nearest" = largest mask area (closest to camera = largest)
                person_mask = None
                best_area = -1
                for lbl, (mu8, mb, seg, _) in decoded_masks.items():
                    is_person = (
                        lbl in ("person", "people", "human", "body")
                        or seg.get("asset_class", "") == "person"
                        or lbl == target
                    )
                    if is_person:
                        area = int(mb.sum())
                        if area > best_area:
                            best_area = area
                            person_mask = mb

                display = apply_conversation_mode(
                    display, person_mask, anim_progress, blur_int,
                )
                continue

            # --- overlay_image ---
            if effect_type == "overlay_image":
                image_name = effect.get("image_name", "")
                opacity = float(effect.get("intensity", 1.0))

                # Find target mask
                matched_label = None
                if target in decoded_masks:
                    matched_label = target
                else:
                    for syn in LABEL_SYNONYMS.get(target, []):
                        if syn in decoded_masks:
                            matched_label = syn
                            break

                if matched_label and image_name:
                    _, mask_bool, _, _ = decoded_masks[matched_label]
                    invert = bool(effect.get("invert", False))
                    active_mask = ~mask_bool if invert else mask_bool

                    overlay_img = load_overlay_image(image_name, overlay_dir)
                    if overlay_img is not None:
                        display = apply_image_overlay(display, active_mask, overlay_img, opacity)
                continue

            # --- standard effects ---
            if not effect_type or effect_type in ("none",):
                continue

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
                display = apply_standard_effect(display, mask_bool, effect)
            elif effect.get("invert", False) and decoded_masks:
                combined = np.zeros((height, width), dtype=bool)
                for _, (_, mb, _, _) in decoded_masks.items():
                    combined |= mb
                bg_mask = ~combined
                eff_copy = dict(effect)
                eff_copy["invert"] = False
                display = apply_standard_effect(display, bg_mask, eff_copy)

        # ----------------------------------------------------------------
        # Outlines / fills in outline/all mode
        # ----------------------------------------------------------------
        if mode in ("outline", "all") and not conversation_mode_applied:
            for label, (mask_u8, mask_bool, seg, seg_i) in decoded_masks.items():
                effect = active_effects.get(label)
                if not effect:
                    for eff_target, eff in active_effects.items():
                        if label in LABEL_SYNONYMS.get(eff_target, []):
                            effect = eff
                            break
                if effect:
                    display = draw_filled(display, mask_u8, seg.get("asset_class", ""), alpha=0.25)
                    display = draw_outline(display, mask_u8, seg_i, effect.get("intensity", 1.0))

        # ----------------------------------------------------------------
        # HUD text overlay
        # ----------------------------------------------------------------
        if active_effects:
            y = 60
            for tgt, eff in active_effects.items():
                et = eff.get("effect_type", "")
                img_name = eff.get("image_name", "")
                label = f"{et}({img_name}) → {tgt}" if img_name else f"{et} → {tgt}"
                cv2.putText(display, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                y += 35

        # Write frame
        if ffmpeg_proc:
            ffmpeg_proc.stdin.write(display.tobytes())
        elif out:
            out.write(display)

        if show_preview:
            cv2.imshow("Preview", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  [{pct:.1f}%] Frame {frame_idx}/{total_frames}")

    # Cleanup
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

    # Mux original audio
    _mux_audio(str(video_path), output_path)


# ---------------------------------------------------------------------------
# Audio mux
# ---------------------------------------------------------------------------

def _mux_audio(source_video: str, rendered_video: str):
    out_path = Path(rendered_video)
    tmp_path = out_path.with_stem(out_path.stem + "_tmp")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", rendered_video, "-i", source_video,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and tmp_path.exists():
            tmp_path.replace(out_path)
            print(f"Audio muxed from source into: {rendered_video}")
        else:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"Warning: Audio mux failed (code {result.returncode})")
    except FileNotFoundError:
        print("Warning: ffmpeg not found — output has no audio")
    except Exception as e:
        print(f"Warning: Audio mux failed: {e}")
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="JSON-Driven SAM3 Story Renderer",
    )
    parser.add_argument("--story", "-s", required=True, help="Path to story directory (containing story.json + masks.bin)")
    parser.add_argument("--video", "-v", default=None, help="Input video file (default: from story metadata)")
    parser.add_argument("--output", "-o", default="video_pipeline/output/rendered.mp4", help="Output video path")
    parser.add_argument(
        "--overlay-dir", default="video_pipeline/overlay_images/",
        help="Directory containing overlay images (default: video_pipeline/overlay_images/)",
    )
    parser.add_argument(
        "--mode", "-m", default="effects",
        choices=["outline", "filled", "effects", "all"],
        help="Render mode",
    )
    parser.add_argument("--show-preview", action="store_true", help="Show live preview window")
    parser.add_argument("--confidence-min", type=float, default=0.10, help="Min mask confidence")

    args = parser.parse_args()

    story_dir = Path(args.story)
    overlay_dir = Path(args.overlay_dir)

    if not story_dir.exists():
        parser.error(f"Story directory not found: {story_dir}")
    if not overlay_dir.exists():
        print(f"Warning: overlay_dir not found ({overlay_dir}), image overlays will be skipped.")

    render_story(
        story_dir=story_dir,
        video_path=args.video,
        output_path=args.output,
        overlay_dir=overlay_dir,
        mode=args.mode,
        show_preview=args.show_preview,
        confidence_min=args.confidence_min,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
