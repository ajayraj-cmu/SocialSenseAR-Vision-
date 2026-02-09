"""Server configuration."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    # Network
    host: str = "0.0.0.0"
    port: int = 8765
    max_message_size: int = 10 * 1024 * 1024  # 10MB for stereo JPEG frames

    # GPU
    gpu_id: int = 0
    device: str = "cuda"  # "cuda" or "cpu"

    # Pipeline mode: "sam3" (default), "grounded-sam2", or "legacy"
    pipeline_mode: str = "sam3"

    # Grounded SAM2 (GroundingDINO detection + SAM2 masks — no API keys needed)
    gdino_model: str = "IDEA-Research/grounding-dino-tiny"
    sam2_model: str = "facebook/sam2-hiera-large"
    gdino_box_threshold: float = 0.3
    gdino_text_threshold: float = 0.25

    # SAM3 (text-prompted segmentation — requires HF gated access)
    sam3_model: str = "facebook/sam3"
    sam3_resolution: int = 1008          # TRT engine resolution: 1008 or 784 (must match exported engines)
    sam3_prompts_per_frame: int = 1      # prompts per frame (all rotate equally, cached when static)
    sam3_cache_ttl: float = 4.0          # mask cache lifetime — must outlast full rotation
    sam3_confidence_threshold: float = 0.12

    # FastSAM (legacy mode — matches sam_gemini_voice.py defaults)
    fastsam_model: str = "FastSAM-s.pt"
    fastsam_conf: float = 0.50           # higher = fewer fragments on textured objects (paintings/screens)
    fastsam_imgsz: int = 512            # matches original adaptive max (was 320)
    fastsam_device: str = "cuda"

    # MediaPipe (person + body part detection)
    mediapipe_models_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )

    # Gemini Vision
    gemini_model: str = "gemini-2.5-flash"
    gemini_min_interval: float = 6.0    # seconds between API calls
    gemini_max_calls_per_minute: int = 5
    gemini_label_cache_ttl: float = 30.0  # seconds

    # Mask refinement
    mask_min_area: int = 300            # minimum pixel count — suppresses small fragments
    mask_refine_grabcut: bool = False   # OFF — GrabCut adds ~400ms per mask, too slow for realtime

    # Emotion detection
    emotion_enabled: bool = True
    emotion_frame_skip: int = 2
    emotion_smoothing_window: int = 4

    # Audio / Whisper
    audio_enabled: bool = True
    whisper_model: str = "whisper-1"

    # OpenAI
    openai_summary_model: str = "gpt-4o-mini"

    # Debug
    debug_view: bool = False  # Show Quest camera feed + overlays in cv2 window
    metrics_log_path: Optional[str] = None  # JSONL file for structured metrics

    # Env vars (loaded at runtime)
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
