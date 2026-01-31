# SocialSenseAR - Vision Pipeline

Real-time AR environment modifier with voice control, using FastSAM, YOLO-World, Gemini Vision, and advanced sensory modulation.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd "Meta X SocialSense"

# Install dependencies
pip install -r requirements/base.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Running the Application

**Main Application (Voice-Controlled):**
```bash
python sam_gemini_main.py
```

**Voice Commands:**
- Say **"hey vibe"** to record a video command
- Say **"hey vibe audio"** to record an audio command
- Speak your command (e.g., "blur my face", "dim the ceiling")
- Say **"thanks"** to process the command

**Keyboard Controls:**
- `A`: Activate autopilot stress-testing mode
- `C`: Toggle clean view (hide labels/masks)
- `Q`: Quit application

**Alternative Entry Points:**
```bash
# Modular pipeline with keyboard/voice control
python pipeline_main.py

# Experimental FastSAM + YOLO + Gemini demo
python experiments_demo.py
```

## Project Structure

```
Meta X SocialSense/
├── sam_gemini_main.py           # Main voice-controlled application
├── pipeline_main.py             # Alternative modular pipeline
├── experiments_demo.py          # FastSAM + YOLO demo
├── pyproject.toml              # Package configuration
├── .env                        # API keys (create from template)
│
├── src/                        # Core vision pipeline
│   ├── audio/                  # Audio processing and voice listening
│   ├── camera/                 # Async camera capture
│   ├── capture/                # Video capture and frame buffering
│   ├── control/                # Environment controller (SAM + MediaPipe)
│   ├── core/                   # Type definitions and contracts
│   ├── gemini/                 # Gemini AI agent integration
│   ├── intent/                 # LLM-based intent parsing
│   ├── pipeline/               # 6-stage pipeline orchestrator
│   ├── safety/                 # Safety layer and sensory monitoring
│   ├── segmentation/           # FastSAM auto-segmentation
│   ├── transforms/             # Visual transformations and effects
│   └── voice/                  # Voice command processing
│
├── server/                     # FastAPI streaming server (AWS deployment)
│   ├── api/                    # REST endpoints
│   ├── auth/                   # JWT + API key authentication
│   ├── latency/                # Adaptive quality control
│   ├── monitoring/             # CloudWatch integration
│   ├── session/                # Session management
│   └── streaming/              # WebSocket video streaming
│
├── models/                     # Model weights
│   ├── FastSAM-s.pt
│   ├── yolov8n.pt
│   └── yolov8s-worldv2.pt
│
├── requirements/               # Dependency management
│   ├── base.txt               # Core pipeline dependencies
│   ├── server.txt             # Server deployment dependencies
│   └── dev.txt                # Development and testing tools
│
├── config/                     # Configuration files
│   └── settings.yaml          # Pipeline settings
│
├── docs/                       # Documentation
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT.md          # AWS deployment guide
│   ├── SERVER.md              # Server documentation
│   └── ...
│
├── scripts/                    # Utility scripts
│   └── deployment/            # AWS deployment scripts
│
├── tests/                      # Test suite
│   ├── server/                # Server tests
│   ├── unit/                  # Unit tests (placeholder)
│   └── integration/           # Integration tests (placeholder)
│
├── examples/                   # Example code
│   └── client_example.py      # Client SDK example
│
└── infrastructure/             # Infrastructure as code
    └── terraform/             # Terraform configs for AWS
```

## Features

### Core Vision Pipeline
- **Real-time segmentation** using FastSAM
- **Body part detection** with MediaPipe (face, hands, arms, legs, torso)
- **Object detection** with YOLO-World
- **AI-powered labeling** using Gemini Vision
- **Visual effects** (blur, brightness, color, motion dampening)
- **Mask tracking** with persistence to prevent flickering

### Voice Control
- Wake word detection ("hey vibe", "hey vibe audio")
- Natural language command processing
- Real-time audio processing with DeepFilterNet
- Voice isolation and noise suppression

### Gemini AI Integration
- Vision-based object labeling
- Natural language command interpretation
- Feedback loop for self-optimization
- Label correction and validation
- Rate limiting for API compliance

### Server (AWS Deployment)
- FastAPI web server with WebSocket streaming
- JWT and API key authentication
- Adaptive quality control for bandwidth optimization
- CloudWatch monitoring and metrics
- Multi-session support with DynamoDB

## Configuration

### Environment Variables (.env)
```bash
GEMINI_API_KEY=your_key_here
# or
GOOGLE_API_KEY=your_key_here
```

### Pipeline Settings (config/settings.yaml)
- Detection thresholds
- Effect parameters
- Safety constraints
- Performance tuning

## Running as a Package

```bash
# Install in development mode
pip install -e .

# Run core pipeline
python -m src

# Or use the installed command
socialsensear
```

## Development

### Install Development Dependencies
```bash
pip install -r requirements/dev.txt
```

### Run Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/ server/

# Sort imports
isort src/ server/

# Type checking
mypy src/

# Linting
flake8 src/ server/
```

## Server Deployment

### Local Development
```bash
pip install -r requirements/server.txt
python server/streaming_server.py
```

### AWS Deployment
See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed AWS setup instructions.

```bash
# Quick deploy
./scripts/deployment/aws_deploy.sh
```

## Architecture

### 6-Stage Vision Pipeline
1. **Capture** - Video/frame acquisition
2. **Intent** - LLM command parsing
3. **Segmentation** - FastSAM object detection
4. **Safety** - Constraint validation
5. **Transform** - Visual effect application
6. **Output** - Rendered display

### Modular Design
Each component is independently testable and reusable:
- GeminiAgent - Portable AI integration
- AudioProcessor - Standalone audio processing
- EnvironmentController - Complete vision pipeline
- VoiceListener - Voice command handling

## Performance

- **Real-time processing** at 15-30 FPS (hardware dependent)
- **Optimized code** - 45% reduction from original codebase
- **Async architecture** - Non-blocking camera and audio
- **Adaptive quality** - Automatic bandwidth optimization (server mode)

## Recent Changes (2026-01-30)

The codebase was recently reorganized for better maintainability:
- Refactored 3,860-line monolithic script into clean modules
- Separated server code from core pipeline
- Optimized all modules (45% line reduction)
- Added proper package structure with pyproject.toml
- Consolidated requirements into organized files
- See [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md) for details

## Documentation

- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - AWS setup and deployment
- [Server Documentation](docs/SERVER.md) - Server architecture and usage
- [Usage Guide](docs/USAGE_GUIDE.md) - Detailed usage instructions
- [Pipeline Documentation](docs/PIPELINE_DOCUMENTATION.md) - Pipeline internals

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please ensure:
- Code passes all tests (`pytest tests/`)
- Code is formatted (`black`, `isort`)
- Type hints are included (`mypy`)
- Documentation is updated

## Support

For issues or questions, please open an issue on GitHub.
