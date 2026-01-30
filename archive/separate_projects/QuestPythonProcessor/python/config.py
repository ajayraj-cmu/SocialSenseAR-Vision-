"""
Configuration module for Quest Processor.

This module contains all configuration options, presets, and settings.
Modify ACTIVE_CONFIG to change default behavior, or use command-line args.

To add a new preset:
1. Add entry to PRESETS dict with your settings
2. Optionally set as ACTIVE_PRESET default
"""
from dataclasses import dataclass, field
from typing import Optional, List
import argparse


# === QUALITY PRESETS ===
# Each preset balances quality vs performance differently
PRESETS = {
    "QUALITY": {
        "processor_width": 640,     # Input size for ML model
        "effect_scale": 1.0,        # Effect resolution multiplier
        "process_skip": 1,          # Process every Nth frame
        "display_skip": 1,          # Display every Nth frame
        "mask_blur": 15,            # Mask edge smoothing
    },
    "STREAM": {
        "processor_width": 320,
        "effect_scale": 1.0,
        "process_skip": 1,
        "display_skip": 3,
        "mask_blur": 15,
    },
    "BALANCED": {
        "processor_width": 320,
        "effect_scale": 0.5,
        "process_skip": 2,
        "display_skip": 2,
        "mask_blur": 11,
    },
    "FAST": {
        "processor_width": 256,
        "effect_scale": 0.25,
        "process_skip": 3,
        "display_skip": 3,
        "mask_blur": 7,
    },
    "EXTREME": {
        "processor_width": 160,
        "effect_scale": 0.25,
        "process_skip": 4,
        "display_skip": 4,
        "mask_blur": 5,
    },
}

# Default preset
ACTIVE_PRESET = "QUALITY"


@dataclass
class Config:
    """Main configuration for the processing pipeline.

    Attributes:
        source: Video source type ("quest", "webcam", "file")
        processor: ML processor type ("yolo", "mediapipe", etc.)
        effect: Visual effect type ("focus", "blur", etc.)
        ui: UI backend ("opencv", "qt", "headless")
        preset: Quality preset name
        auto_start: Start processing automatically
        show_profiling: Show performance stats
        display_max_width: Max display window width
        headless: Run without display
        capture_max_size: Max capture resolution (0 = full)
        transition_duration: Effect transition time in seconds
    """
    # Component selection
    source: str = "quest_tcp"
    processor: str = "yolo"
    effect: str = "focus"
    ui: str = "opencv"

    # Preset (overrides individual settings)
    preset: str = ACTIVE_PRESET

    # Behavior
    auto_start: bool = True
    show_profiling: bool = True
    headless: bool = False

    # Display
    display_max_width: int = 1920
    profile_interval: int = 30

    # Capture
    capture_max_size: int = 0  # 0 = full resolution
    capture_fps: int = 60

    # Transition
    transition_duration: float = 1.0

    # TCP streaming quality (0-100, higher = better quality but larger size)
    jpeg_quality: int = 95
    # Raw output mode: send uncompressed RGB24 instead of JPEG (faster encode, larger transfer)
    use_raw_output: bool = False

    # Audio processing
    audio_enabled: bool = True
    audio_services: List[str] = field(default_factory=lambda: ["context", "emotion"])
    audio_state_dir: str = "~/Downloads/Nex/conve_context"
    audio_mic1_index: Optional[int] = None  # User mic (None = auto-detect)
    audio_mic2_index: Optional[int] = None  # Other person mic (None = same as mic1)
    audio_isolation_enabled: bool = True
    audio_isolation_input_index: Optional[int] = None
    audio_isolation_output_index: Optional[int] = None

    # Video file source
    video_file: Optional[str] = None
    playback_speed: float = 1.0
    loop_video: bool = False

    # Computed from preset (set in __post_init__)
    processor_width: int = field(default=640, init=False)
    effect_scale: float = field(default=1.0, init=False)
    process_skip: int = field(default=1, init=False)
    display_skip: int = field(default=1, init=False)
    mask_blur: int = field(default=15, init=False)

    def __post_init__(self):
        """Apply preset settings."""
        if self.preset in PRESETS:
            preset = PRESETS[self.preset]
            self.processor_width = preset["processor_width"]
            self.effect_scale = preset["effect_scale"]
            self.process_skip = preset["process_skip"]
            self.display_skip = preset["display_skip"]
            self.mask_blur = preset["mask_blur"]


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """Load configuration from command-line args or defaults.

    Args:
        args: Parsed command-line arguments, or None for defaults

    Returns:
        Config object with all settings
    """
    if args is None:
        return Config()

    audio_services = getattr(args, 'audio_services', None)
    if audio_services:
        audio_services_list = [s.strip() for s in audio_services.split(',') if s.strip()]
    else:
        audio_services_list = ["context", "emotion"]

    if getattr(args, 'voice_isolation', False) and "voice_isolation" not in audio_services_list:
        audio_services_list.append("voice_isolation")

    return Config(
        source=getattr(args, 'source', 'quest_tcp'),
        processor=getattr(args, 'processor', 'yolo'),
        effect=getattr(args, 'effect', 'focus'),
        ui=getattr(args, 'ui', 'opencv'),
        preset=getattr(args, 'preset', ACTIVE_PRESET),
        auto_start=getattr(args, 'auto_start', True),
        headless=getattr(args, 'headless', False),
        use_raw_output=getattr(args, 'raw_output', False),
        audio_enabled=not getattr(args, 'no_audio', False),
        audio_services=audio_services_list,
        audio_mic1_index=getattr(args, 'mic1', None),
        audio_mic2_index=getattr(args, 'mic2', None),
        audio_isolation_enabled=not getattr(args, 'no_voice_isolation', False),
        audio_isolation_input_index=getattr(args, 'isolation_input', None),
        audio_isolation_output_index=getattr(args, 'isolation_output', None),
        video_file=getattr(args, 'video_file', None),
        playback_speed=getattr(args, 'playback_speed', 1.0),
        loop_video=getattr(args, 'loop', False),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Quest Processor - Modular ML video processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Default settings
  python main.py --preset FAST             # Use FAST preset
  python main.py --processor yolo          # Explicit processor
  python main.py --headless                # No display window
        """
    )

    parser.add_argument(
        '--source', '-s',
        choices=['quest', 'quest_tcp', 'webcam', 'file'],
        default='quest_tcp',
        help='Video input source (default: quest_tcp for Unity passthrough).'
    )

    parser.add_argument(
        '--processor', '-p',
        choices=['yolo'],  # Add more as implemented
        default='yolo',
        help='ML processor for frame analysis (default: yolo)'
    )

    parser.add_argument(
        '--effect', '-e',
        choices=['focus'],  # Add more as implemented
        default='focus',
        help='Visual effect to apply (default: focus)'
    )

    parser.add_argument(
        '--ui', '-u',
        choices=['opencv', 'headless', 'webview', 'quest_tcp'],
        default='opencv',
        help='UI backend (default: opencv). Use quest_tcp to send frames back to Unity.'
    )

    parser.add_argument(
        '--preset',
        choices=list(PRESETS.keys()),
        default=ACTIVE_PRESET,
        help=f'Quality preset (default: {ACTIVE_PRESET})'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without display window'
    )

    parser.add_argument(
        '--raw-output',
        action='store_true',
        help='Send raw RGB24 pixels instead of JPEG (faster encode, larger transfer)'
    )

    parser.add_argument(
        '--no-auto-start',
        action='store_true',
        help='Do not start processing automatically'
    )

    # Audio arguments
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Disable audio processing'
    )

    parser.add_argument(
        '--audio-services',
        type=str,
        default=None,
        help='Comma-separated audio services to enable (e.g., context,voice_isolation)'
    )

    parser.add_argument(
        '--voice-isolation',
        action='store_true',
        help='Enable voice isolation audio service'
    )

    parser.add_argument(
        '--no-voice-isolation',
        action='store_true',
        help='Disable voice isolation processing (passthrough if service is enabled)'
    )

    parser.add_argument(
        '--mic1',
        type=int,
        default=None,
        help='Microphone index for user (None = default device)'
    )

    parser.add_argument(
        '--mic2',
        type=int,
        default=None,
        help='Microphone index for other person (None = same as mic1)'
    )

    parser.add_argument(
        '--isolation-input',
        type=int,
        default=None,
        help='Input device index for voice isolation (None = default)'
    )

    parser.add_argument(
        '--isolation-output',
        type=int,
        default=None,
        help='Output device index for voice isolation (None = default)'
    )

    parser.add_argument(
        '--list-mics',
        action='store_true',
        help='List available microphones and exit'
    )

    # Video file arguments
    parser.add_argument(
        '--video-file', '-f',
        type=str,
        default=None,
        help='Path to video file (use with --source file)'
    )

    parser.add_argument(
        '--playback-speed',
        type=float,
        default=1.0,
        help='Video playback speed multiplier (default: 1.0)'
    )

    parser.add_argument(
        '--loop',
        action='store_true',
        help='Loop video playback'
    )

    return parser.parse_args()
