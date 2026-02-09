"""Modal Deployment Configuration

Central configuration for Modal GPU deployment.
Modify these settings to customize your deployment.
"""

import os

# ============================================================
# GPU Configuration
# ============================================================

# GPU type to use. Options:
# - "A10G": 24GB VRAM, ~$1.10/hr (RECOMMENDED - best value)
# - "A100": 40GB VRAM, ~$3.70/hr (overkill for SAM3, but fastest)
# - "A100-80GB": 80GB VRAM, ~$4.50/hr (unnecessary for this workload)
# - "L4": 24GB VRAM, ~$0.70/hr (budget option, slightly slower)
# - "T4": 16GB VRAM, ~$0.50/hr (minimum viable, tight VRAM)
# - "H100": 80GB VRAM, ~$8/hr (extreme overkill, not recommended)
GPU_TYPE = os.environ.get("MODAL_GPU", "A100")

# GPU count (1 is sufficient for SAM3 pipeline)
GPU_COUNT = 1

# Container timeout (seconds) - max time before idle shutdown
CONTAINER_TIMEOUT = 600  # 10 minutes

# Scaledown window (seconds) - time before scaling to zero (renamed from container_idle_timeout)
SCALEDOWN_WINDOW = 60  # 1 minute
CONTAINER_IDLE_TIMEOUT = SCALEDOWN_WINDOW  # Backwards compatibility

# ============================================================
# Model Configuration
# ============================================================

# SAM3 configuration
SAM3_MODEL = "facebook/sam3"
SAM3_RESOLUTION = 1008  # Must match TRT engine if using TRT (1008 or 784)
SAM3_PROMPTS_PER_FRAME = 1  # How many prompts to decode per frame
SAM3_CACHE_TTL = 4.0  # Mask cache lifetime (seconds)
SAM3_CONFIDENCE_THRESHOLD = 0.12  # Minimum confidence for mask

# Gemini configuration
GEMINI_MODEL = "gemini-2.5-flash"  # Fast + accurate for scene understanding
GEMINI_MIN_INTERVAL = 6.0  # Min seconds between API calls
GEMINI_CACHE_TTL = 30.0  # Scene cache lifetime

# ============================================================
# Volume Configuration
# ============================================================

# Volume name for model cache (created automatically)
CACHE_VOLUME_NAME = "socialsense-cache"

# Volume mount path inside container
CACHE_MOUNT_PATH = "/cache"

# ============================================================
# Secret Configuration
# ============================================================

# Modal secret name containing API keys
# Required keys:
# - OPENAI_API_KEY: For Whisper transcription
# - GEMINI_API_KEY or GOOGLE_API_KEY: For scene understanding
# - HF_TOKEN: For gated HuggingFace models (facebook/sam3)
SECRET_NAME = "socialsense-secrets"

# ============================================================
# Feature Flags
# ============================================================

# Enable audio processing (voice agent)
AUDIO_ENABLED = True  # Server-side voice agent pipeline (wake word + Whisper + Gemini)

# Enable debug view (OpenCV window) - only works in modal serve
DEBUG_VIEW = False

# Enable metrics logging
METRICS_LOG_PATH = None  # Set to a path to enable JSONL metrics

# ============================================================
# Container Image Configuration
# ============================================================

# Python version
PYTHON_VERSION = "3.12"

# CUDA version (must match PyTorch wheel)
CUDA_VERSION = "cu121"  # CUDA 12.1 for PyTorch 2.5.1

# PyTorch version
TORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"

# ============================================================
# WebSocket Configuration
# ============================================================

# Maximum message size (bytes) - must accommodate stereo JPEG frames
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB

# WebSocket ping interval (seconds)
PING_INTERVAL = 20

# WebSocket ping timeout (seconds)
PING_TIMEOUT = 60

# ============================================================
# Deployment Mode
# ============================================================

# App name in Modal
APP_NAME = "socialsense-ar-gpu"

# Environment (for multiple deployments)
# Set via MODAL_ENV=staging or MODAL_ENV=prod
ENVIRONMENT = os.environ.get("MODAL_ENV", "prod")

# ============================================================
# Advanced: TensorRT Engine Configuration
# ============================================================

# If you have pre-built TensorRT engines, they will be auto-detected
# based on SAM3_RESOLUTION. Put them in project root with these names:
# - sam3_vision_int8_{resolution}.engine (preferred, INT8 quantized)
# - sam3_vision_{resolution}.engine (FP16 fallback)
# - sam3_topk_decoder_{resolution}.engine (top-k decoder, recommended)
# - sam3_decoder_{resolution}.engine (full decoder, slower)
# - sam3_meta_{resolution}.json (metadata for TRT engines)

# TRT engines are built on Modal GPU and stored on the cache volume.
# At container startup, modal_app.py copies them to /root/ for auto-detection.
TRT_ENABLED = True  # Set False to skip engine loading (PyTorch fallback)
TRT_ENGINE_DIR = f"/cache/trt_engines/{GPU_TYPE}"  # Per-GPU engines (not cross-compatible)

# ============================================================
# Helper Functions
# ============================================================

def get_gpu_spec():
    """Get Modal GPU specification string."""
    if GPU_COUNT > 1:
        return f"{GPU_TYPE}:{GPU_COUNT}"
    return GPU_TYPE

def get_app_name():
    """Get full app name with environment suffix."""
    if ENVIRONMENT != "prod":
        return f"{APP_NAME}-{ENVIRONMENT}"
    return APP_NAME

def get_cache_volume_name():
    """Get cache volume name with environment suffix."""
    if ENVIRONMENT != "prod":
        return f"{CACHE_VOLUME_NAME}-{ENVIRONMENT}"
    return CACHE_VOLUME_NAME

# ============================================================
# Validation
# ============================================================

def validate_config():
    """Validate configuration and warn about common issues."""
    issues = []

    # Check GPU type
    valid_gpus = ["A10G", "A100", "A100-80GB", "L4", "T4", "H100"]
    if GPU_TYPE not in valid_gpus:
        issues.append(f"Unknown GPU_TYPE '{GPU_TYPE}'. Valid: {valid_gpus}")

    # Warn about suboptimal choices
    if GPU_TYPE == "T4":
        issues.append("WARNING: T4 has only 16GB VRAM. SAM3 may fail with OOM. Recommend A10G or L4.")
    if GPU_TYPE in ["A100-80GB", "H100"]:
        issues.append(f"WARNING: {GPU_TYPE} is overkill for SAM3 (~$4-8/hr). A10G is recommended.")

    # Check resolution
    if SAM3_RESOLUTION not in [784, 1008]:
        issues.append(f"WARNING: SAM3_RESOLUTION {SAM3_RESOLUTION} is non-standard. Use 1008 or 784.")

    return issues

# Auto-validate on import
_validation_issues = validate_config()
if _validation_issues:
    import warnings
    for issue in _validation_issues:
        warnings.warn(issue)
