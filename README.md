# SocialSenseAR - Vision

Real-time AR environment modifier with voice control, using SAM (Segment Anything Model), Gemini Vision, and sensory modulation features.

## Quick Start

### Main Application (Voice-Controlled)

```bash
python scripts/sam_gemini_voice.py
```

**Voice Commands:**
- Say **"hey vibe"** to start recording
- Say your command (e.g., "blur my face", "dim the ceiling")
- Say **"thanks"** to process

### Vision Pipeline (FastSAM + YOLO-World + Gemini)

```bash
python fast_sam_yolo_combo.py
```

**Controls:**
- D: Toggle Dev/Non-Dev mode
- Q: Quit
- S: Save screenshot

### Alternative Entry Point

```bash
python main.py
```

## Project Structure (Simplified)

```
Meta X SocialSense/
├── main.py                      # Perceptual modulation engine entry point
├── requirements.txt             # Python dependencies
├── .env                         # API keys (create from .env.example)
├── config/                      # Configuration files
├── src/                         # Core modular source code (SIMPLIFIED)
│   ├── capture/                 # Video capture and frame buffering
│   ├── core/                    # Core contracts and type definitions
│   ├── intent/                  # LLM-based intent parsing (Gemini)
│   ├── pipeline/                # Main pipeline orchestrator (6-stage)
│   ├── safety/                  # Safety layer and sensory monitoring
│   ├── segmentation/            # SAM auto-segmentation + mask processing
│   ├── transforms/              # Visual transformations
│   └── voice/                   # Voice command processing
├── scripts/                     # Standalone applications
│   └── sam_gemini_voice.py      # Main voice-controlled app (WORKING)
├── docs/                        # Documentation
└── archive/                     # Archived code and separate projects
    ├── separate_projects/
    │   └── QuestPythonProcessor/ # Quest VR pipeline (separate project)
    ├── Audio_Merge/             # Audio UI prototypes
    └── old_scripts/

**Removed (3,000+ lines of bloat):**
- ✗ audio/ subsystem (unused)
- ✗ tracking/ modules (over-engineered)
- ✗ depth/ estimation (not utilized)
- ✗ Duplicate segmentation implementations
- ✗ Duplicate intent parsers
```

## Features

- **Voice Control**: Wake word activation ("hey vibe" / "thanks")
- **Real-time Segmentation**: FastSAM for precise object segmentation
- **Smart Labeling**: Gemini Vision API for open-vocabulary detection
- **Multi-modal Pipeline**: YOLO-World + FastSAM + Gemini combo
- **Sensory Modulation**: Blur, brightness, color, motion dampening
- **Safety Layer**: Prevents excessive visual modifications
- **Simplified Architecture**: 6-stage pipeline (down from 10), ~60% less code

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `ultralytics` (FastSAM, YOLO)
- `google-generativeai` (Gemini API)
- `speech_recognition` (Voice commands)
- `mediapipe` (Body part segmentation)
- `opencv-python` (Video processing)

## Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_API_KEY=your-google-api-key  # Fallback
```

## Controls (Main App)

- **V** - Toggle clean/full view
- **C** - Clear all effects
- **L** - List all detected labels
- **S** - Screenshot
- **Q** - Quit

## Documentation

See `docs/` folder:
- `PIPELINE_DOCUMENTATION.md` - Full pipeline architecture
- `FEEDBACK_LOOP_DOCUMENTATION.md` - Self-correction system
- `USAGE_GUIDE.md` - Detailed usage instructions

## Architecture

### Modular Design

The codebase is organized into clean, modular components:

- **Core**: Type-safe contracts and interfaces
- **Pipeline**: Orchestrates all processing stages
- **Segmentation**: FastSAM and SAM-based object segmentation
- **Tracking**: Persistent object tracking with Kalman filters
- **Intent**: Natural language processing for commands
- **Transforms**: Visual and audio effect application
- **Safety**: Ensures modifications stay within safe bounds

### Vision Pipeline

The vision pipeline combines three powerful models:

1. **FastSAM**: Fast, accurate segmentation
2. **YOLO-World**: Open-vocabulary object detection
3. **Gemini Vision**: Label correction and unknown object identification

## License

MIT License
