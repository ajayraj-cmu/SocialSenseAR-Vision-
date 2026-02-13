"""Story schema — dataclasses, JSON+binary I/O, effect timeline resolver.

A "story" captures the full SAM3 segmentation of a video:
  story_dir/
    story.json  — metadata, per-frame segments, voice events, effects timeline
    masks.bin   — concatenated RLE mask bytes (same uint16 LE format as server/encoding/rle.py)

The effects_timeline is editable: change effects, timing, add/remove entries,
then re-render without re-running SAM3.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StoryMetadata:
    source_video: str
    fps: float
    frame_count: int
    width: int
    height: int
    rle_width: int
    rle_height: int
    rle_scale: float = 0.75
    pipeline_mode: str = "sam3"
    sam3_resolution: int = 1008
    processing_time_s: float = 0.0
    created_at: str = ""


@dataclass
class StorySegment:
    track_id: str
    label: str
    asset_class: str
    confidence: float
    bbox: list  # [x_min, y_min, x_max, y_max] normalized
    center_x: float
    center_y: float
    mask_offset: int
    mask_length: int
    mask_width: int
    mask_height: int


@dataclass
class StoryFrame:
    frame_idx: int
    timestamp_s: float
    sam_ran: bool
    segments: list  # list of StorySegment dicts (serialized)


@dataclass
class VoiceEvent:
    timestamp_s: float
    frame_idx: int
    transcript: str
    command: dict  # {targets, effect_type, intensity, action, invert, ...}


@dataclass
class EffectEntry:
    start_frame: int
    target: str
    effect_type: str
    end_frame: Optional[int] = None  # None = until end of video
    intensity: float = 1.0
    color_hex: str = ""
    invert: bool = False
    source: str = ""  # "voice" if auto-generated from voice_events


@dataclass
class Story:
    version: int = 1
    metadata: Optional[StoryMetadata] = None
    prompts: list = field(default_factory=list)
    tracks: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)  # list of StoryFrame dicts
    voice_events: list = field(default_factory=list)  # list of VoiceEvent dicts
    effects_timeline: list = field(default_factory=list)  # list of EffectEntry dicts


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_story(story: Story, output_dir: str | Path):
    """Write story.json + masks.bin to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    story_dict = _story_to_dict(story)
    story_path = output_dir / "story.json"
    with open(story_path, "w") as f:
        json.dump(story_dict, f, indent=2)

    # masks.bin is written incrementally by video_processor — don't overwrite
    masks_path = output_dir / "masks.bin"
    if not masks_path.exists():
        masks_path.touch()

    return story_path


def load_story(story_dir: str | Path) -> Story:
    """Load story.json from story_dir. Returns Story object."""
    story_dir = Path(story_dir)
    story_path = story_dir / "story.json"

    with open(story_path, "r") as f:
        data = json.load(f)

    story = Story()
    story.version = data.get("version", 1)

    md = data.get("metadata", {})
    story.metadata = StoryMetadata(
        source_video=md.get("source_video", ""),
        fps=md.get("fps", 30.0),
        frame_count=md.get("frame_count", 0),
        width=md.get("width", 0),
        height=md.get("height", 0),
        rle_width=md.get("rle_width", 0),
        rle_height=md.get("rle_height", 0),
        rle_scale=md.get("rle_scale", 0.75),
        pipeline_mode=md.get("pipeline_mode", "sam3"),
        sam3_resolution=md.get("sam3_resolution", 1008),
        processing_time_s=md.get("processing_time_s", 0.0),
        created_at=md.get("created_at", ""),
    )

    story.prompts = data.get("prompts", [])
    story.tracks = data.get("tracks", {})
    story.frames = data.get("frames", [])
    story.voice_events = data.get("voice_events", [])
    story.effects_timeline = data.get("effects_timeline", [])

    return story


def open_masks_bin(story_dir: str | Path, mode: str = "rb"):
    """Open the masks.bin file for reading or writing."""
    return open(Path(story_dir) / "masks.bin", mode)


def read_mask_bytes(story_dir: str | Path, offset: int, length: int) -> bytes:
    """Read RLE mask bytes from masks.bin at the given offset."""
    with open(Path(story_dir) / "masks.bin", "rb") as f:
        f.seek(offset)
        return f.read(length)


# ---------------------------------------------------------------------------
# Effect Timeline Resolver
# ---------------------------------------------------------------------------

def resolve_effects(effects_timeline: list, frame_idx: int) -> dict:
    """Resolve active effects at a given frame index.

    Returns: {target_label: {"effect_type": ..., "intensity": ..., ...}}
    """
    active = {}
    for entry in effects_timeline:
        start = entry.get("start_frame", 0)
        end = entry.get("end_frame")  # None = until end

        if frame_idx < start:
            continue
        if end is not None and frame_idx > end:
            continue

        target = entry.get("target", "")
        if not target:
            continue

        active[target] = {
            "effect_type": entry.get("effect_type", "outline"),
            "intensity": entry.get("intensity", 1.0),
            "color_hex": entry.get("color_hex", ""),
            "invert": entry.get("invert", False),
        }

    return active


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _story_to_dict(story: Story) -> dict:
    """Convert Story to a JSON-serializable dict."""
    d = {
        "version": story.version,
        "metadata": asdict(story.metadata) if story.metadata else {},
        "prompts": story.prompts,
        "tracks": story.tracks,
        "frames": story.frames,
        "voice_events": story.voice_events,
        "effects_timeline": story.effects_timeline,
    }
    return d


def make_timestamp() -> str:
    """ISO 8601 timestamp string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
