#!/usr/bin/env python3
"""
Quest Processor - Modular ML Video Processing Pipeline

A modular, plugin-based video processing system for Quest VR headsets.
Easily extensible with new ML models, visual effects, and UI backends.

Usage:
    python main.py                        # Default settings (QUALITY preset)
    python main.py --preset FAST          # Use FAST preset for speed
    python main.py --processor yolo       # Explicit processor selection
    python main.py --headless             # Run without display

To add a new component:
    - Processor: See processors/base.py
    - Effect: See effects/base.py
    - Source: See sources/base.py
    - UI: See ui/base.py

Architecture:
    main.py
    ├── config.py           # Configuration and presets
    ├── core/
    │   ├── pipeline.py     # Main processing loop
    │   └── transition.py   # Effect transition manager
    ├── sources/            # Video input sources
    ├── processors/         # ML models (YOLO, etc.)
    ├── effects/            # Visual effects
    ├── controls/           # Input controls
    └── ui/                 # Display backends
"""
from pathlib import Path
from dotenv import load_dotenv

from config import Config, parse_args, load_config
from core.pipeline import Pipeline
from sources import get_source
from processors import get_processor
from effects import get_effect
from ui import get_ui
from audio import AudioManager


def main():
    """Main entry point."""
    # Load environment variables from .env file
    # Look in the project root (parent of python directory)
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[CONFIG] Loaded environment from {env_file}")
    else:
        # Try current directory as fallback
        load_dotenv()

    # Parse command-line arguments
    args = parse_args()

    # Handle --list-mics: show available microphones and exit
    if getattr(args, 'list_mics', False):
        audio_manager = AudioManager.__new__(AudioManager)
        audio_manager.list_microphones()
        return

    # Load configuration
    config = load_config(args)

    # Handle headless mode
    if args.headless:
        config.headless = True
        config.ui = 'headless'

    # Auto-select quest_tcp UI when using quest_tcp source (for bidirectional communication)
    if config.source == 'quest_tcp' and config.ui == 'opencv':
        config.ui = 'quest_tcp'
        print("[AUTO] Using quest_tcp UI for bidirectional Quest communication")

    if args.no_auto_start:
        config.auto_start = False

    # Defer audio manager creation - will be initialized after window appears
    audio_manager = None
    audio_enabled = config.audio_enabled

    # Initialize components
    print(f"Initializing with preset: {config.preset}")
    print(f"  Source: {config.source}")
    print(f"  Processor: {config.processor}")
    print(f"  Effect: {config.effect}")
    print(f"  UI: {config.ui}")
    print(f"  Audio: {'enabled' if config.audio_enabled else 'disabled'}")
    print()

    source = get_source(config.source, config)
    processor = get_processor(config.processor, config)
    effect = get_effect(config.effect, config)
    ui = get_ui(config.ui, config)

    # Create pipeline
    pipeline = Pipeline(source, processor, effect, ui, config)

    # Pass audio config to pipeline - it will create and start AudioManager after window appears
    pipeline.audio_enabled = audio_enabled
    pipeline.AudioManagerClass = AudioManager if audio_enabled else None

    try:
        # Run video pipeline (audio starts inside after window appears)
        pipeline.run()
    finally:
        # Cleanup audio
        if hasattr(pipeline, 'audio_manager') and pipeline.audio_manager:
            pipeline.audio_manager.stop()


if __name__ == "__main__":
    main()
