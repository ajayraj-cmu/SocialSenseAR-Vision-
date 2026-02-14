"""Server configuration — single source of truth for all runtime defaults."""
import os
from dataclasses import dataclass
from typing import Optional

# Valid effect types (used by voice agent, dashboard, websocket server)
EFFECT_TYPES = ("blur", "dim", "pixelate", "highlight", "outline", "color")

# Expanded effect types for custom masks (voice-parsable, some Unity-side pending)
CUSTOM_EFFECT_TYPES = (
    "frosted_glass", "redact", "grayscale", "spotlight",
    "noise", "desaturate", "vignette", "soft_glow",
    "reduce_contrast", "warm_tint", "cool_tint",
)


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

    # Tracking & matching
    track_max_age: float = 1.0           # seconds before unmatched tracks expire
    track_match_max_dist: float = 0.20   # max centroid distance (fraction of frame diagonal)
    track_match_skip_dist: float = 0.25  # skip obviously impossible matches above this
    gemini_label_match_cost: float = 0.20  # max cost for Gemini label→track matching
    rle_scale: float = 1.0              # RLE encoding scale (1.0 = full resolution, no downscale)

    # RLE mask smoothing (optional quality tuning — see docs/BORDER_SMOOTHNESS_AND_LATENCY_SUGGESTIONS.md)
    rle_edge_blur_kernel: int = 7       # Gaussian blur kernel size before threshold (5, 7, or 9; higher = softer edges). Raised from 5 to 7 for smoother borders.
    rle_smooth_edges_only: bool = False  # If True, skip morphology and only blur (can look smoother on some masks)
    rle_morph_kernel_size: int = 5       # Morphology kernel size for MORPH_CLOSE/MORPH_OPEN (3, 5, or 7)
    rle_min_mask_area: int = 64          # Minimum mask pixel count after RLE resize; smaller masks are discarded (reduces sparse fragments)

    # Display confidence thresholds
    display_confidence_min: float = 0.75   # min confidence for client-side segment rendering
    dashboard_confidence_min: float = 0.10 # min confidence for debug dashboard display
    effect_confidence_min: float = 0.60    # min confidence for applying effects/masks to segments

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
    gemini_planning_model: str = "gemini-2.5-flash-lite"  # Fast model for voice command parsing
    gemini_min_interval: float = 6.0    # seconds between scene understanding API calls
    gemini_labeling_interval: float = 2.0  # seconds between Gemini labeling runs (legacy mode)
    gemini_max_calls_per_minute: int = 5
    gemini_label_cache_ttl: float = 30.0  # seconds

    # Mask refinement
    mask_min_area: int = 300            # minimum pixel count — suppresses small fragments
    mask_refine_grabcut: bool = False   # OFF — GrabCut adds ~400ms per mask, too slow for realtime

    # Emotion detection
    emotion_enabled: bool = True
    emotion_frame_skip: int = 2
    emotion_smoothing_window: int = 4

    # Audio / Voice Agent
    audio_enabled: bool = True
    transcriber_backend: str = "local"    # "local" (faster-whisper) or "cloud" (OpenAI Whisper API)
    whisper_model: str = "whisper-1"      # Cloud Whisper model (when backend="cloud")
    whisper_listening_model: str = "medium.en"
    whisper_recording_model: str = "medium.en"
    whisper_beam_size: int = 5            # Beam search width (higher = more accurate, slower)
    voice_energy_threshold: int = 150     # RMS silence threshold (PCM16)
    voice_listening_chunk_s: float = 3.0  # Chunk duration during listening phase
    voice_recording_chunk_s: float = 3.0  # Chunk duration during recording phase
    voice_assembler_timeout: float = 4.5  # Silence timeout before utterance auto-fires
    voice_wake_window: int = 3            # Sliding window size for wake word detection

    # OpenAI
    openai_summary_model: str = "gpt-4o-mini"

    # PersonaPlex (speech-to-speech voice agent via NVIDIA PersonaPlex-7B)
    personaplex_enabled: bool = False  # Off by default; enable to use PersonaPlex instead of faster-whisper
    personaplex_url: str = "ws://localhost:8998/api/chat"  # Override with Modal URL or remote host
    personaplex_voice_prompt: str = "NATF0.pt"
    personaplex_text_temperature: float = 0.7
    personaplex_text_topk: int = 25
    personaplex_audio_temperature: float = 0.8
    personaplex_audio_topk: int = 250

    # Debug
    debug_view: bool = False  # Show Quest camera feed + overlays in cv2 window
    metrics_log_path: Optional[str] = None  # JSONL file for structured metrics

    # Env vars (loaded at runtime)
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Hugging Face tokens — separate tokens for PersonaPlex vs SAM3
    # HF_PERSONAPLEX_TOKEN: gated access for nvidia/personaplex-7b-v1
    # HF_SAM_TOKEN: gated access for facebook/sam3
    # Both fall back to HF_TOKEN if the specific token is not set.
    hf_personaplex_token: Optional[str] = None
    hf_sam_token: Optional[str] = None
