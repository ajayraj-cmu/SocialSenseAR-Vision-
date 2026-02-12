"""SocialSenseAR Server — Entry point.

Usage:
    python -m server.main                          # Default (CUDA, port 8765)
    python -m server.main --port 9090              # Custom port
    python -m server.main --device cpu             # CPU mode
    python -m server.main --host 0.0.0.0 --port 8765  # AWS deployment
"""
import asyncio
import argparse
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from server.config import ServerConfig
from server.websocket_server import SocialSenseServer


def setup_logging():
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SocialSenseAR Server — ML pipeline for AR glasses"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Compute device")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio pipeline")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotion detection")
    parser.add_argument("--pipeline", default="sam3",
                        choices=["sam3", "grounded-sam2", "legacy"],
                        help="Pipeline mode: sam3 (default), grounded-sam2, or legacy")
    parser.add_argument("--legacy", action="store_true",
                        help="Shorthand for --pipeline legacy")
    parser.add_argument("--stub", action="store_true",
                        help="Run with stub pipeline (no ML models, for testing)")
    parser.add_argument("--debug-view", action="store_true",
                        help="Show Quest camera feed with mask overlays in a window")
    parser.add_argument("--metrics-log", type=str, default=None,
                        help="Path to JSONL file for structured metrics output")
    return parser.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger("server")

    # Load .env
    #
    # This repo is sometimes embedded inside a larger workspace (e.g. SAMARSDK/).
    # Support both:
    # - `ServerBackend/.env` (repo root)
    # - workspace root `.env` (common when users keep all secrets in one place)
    server_backend_root = Path(__file__).resolve().parent.parent
    workspace_root = server_backend_root.parent
    candidates = [server_backend_root / ".env", workspace_root / ".env"]

    loaded_any = False
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=False)
            logger.info(f"Loaded env from {p}")
            loaded_any = True
            break

    if not loaded_any:
        # Fall back to default dotenv behavior (search CWD / parents)
        load_dotenv()

    args = parse_args()

    config = ServerConfig(
        host=args.host,
        port=args.port,
        device=args.device,
        gpu_id=args.gpu_id,
        pipeline_mode="legacy" if args.legacy else args.pipeline,
        fastsam_device=args.device,
        audio_enabled=(not args.no_audio) and (os.getenv("AUDIO_ENABLED", "true").lower() in ("true", "1", "yes")),
        emotion_enabled=not args.no_emotion,
        debug_view=args.debug_view,
        metrics_log_path=args.metrics_log,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        # HF tokens — separate for PersonaPlex vs SAM3 (fallback to HF_TOKEN)
        hf_sam_token=os.getenv("HF_SAM_TOKEN") or os.getenv("HF_TOKEN"),
        hf_personaplex_token=os.getenv("HF_PERSONAPLEX_TOKEN") or os.getenv("HF_TOKEN"),
        # PersonaPlex (optional)
        personaplex_enabled=os.getenv("PERSONAPLEX_ENABLED", "").lower() in ("true", "1", "yes"),
        personaplex_url=os.getenv("PERSONAPLEX_URL", "ws://localhost:8998/api/chat"),
    )

    # Check protobuf backend
    pb_backend = "unknown"
    try:
        from google.protobuf.internal import api_implementation
        pb_backend = api_implementation.Type()
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("SocialSenseAR Server")
    logger.info("=" * 60)
    logger.info(f"  Host:     {config.host}")
    logger.info(f"  Port:     {config.port}")
    logger.info(f"  Device:   {config.device} (GPU {config.gpu_id})")
    logger.info(f"  Pipeline: {config.pipeline_mode}")
    logger.info(f"  Protobuf: {pb_backend}")
    logger.info(f"  Audio:    {'enabled' if config.audio_enabled else 'disabled'}")
    logger.info(f"  Voice:    {'PersonaPlex' if config.personaplex_enabled else 'faster-whisper + Gemini'}")
    logger.info(f"  Emotion:  {'enabled' if config.emotion_enabled else 'disabled'}")
    logger.info(f"  Gemini:   {'key set' if config.gemini_api_key else 'NO KEY'}")
    logger.info(f"  OpenAI:   {'key set' if config.openai_api_key else 'NO KEY'}")
    logger.info(f"  HF SAM:   {'SET' if config.hf_sam_token else 'MISSING'}")
    logger.info(f"  HF PP:    {'SET' if config.hf_personaplex_token else 'MISSING'}")
    if config.personaplex_enabled:
        logger.info(f"  PP URL:   {config.personaplex_url}")
    logger.info("")

    # Ensure HF_TOKEN is set for SAM3 model loading (huggingface_hub reads HF_TOKEN)
    if config.hf_sam_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = config.hf_sam_token
        logger.info("Set HF_TOKEN from HF_SAM_TOKEN for SAM3 model loading")

    if pb_backend == "python":
        logger.warning("Pure Python protobuf detected — this is slow!")
        logger.warning("pip install --upgrade protobuf  (for C/upb backend)")

    pipeline = None
    if not args.stub:
        try:
            t_init = time.perf_counter()
            from server.pipeline.orchestrator import PipelineOrchestrator
            t_import = time.perf_counter()
            logger.info(f"  Import orchestrator: {(t_import - t_init)*1000:.0f}ms")

            # IMPORTANT: do NOT eagerly call pipeline.initialize() here.
            # We want the WebSocket server to start listening immediately so clients
            # (Unity Editor, Quest) can connect even while models warm up.
            #
            # The orchestrator will call initialize() lazily on the first frame:
            # PipelineOrchestrator.process_frame() checks _initialized and runs initialize().
            pipeline = PipelineOrchestrator(config)
            t_ctor = time.perf_counter()
            logger.info(f"  Create orchestrator: {(t_ctor - t_import)*1000:.0f}ms")
            logger.info("  Pipeline will initialize on first incoming frame...")
        except ImportError as e:
            logger.warning(f"Pipeline not available ({e}), running in stub mode")
        except Exception as e:
            logger.error(f"Pipeline creation failed: {e}", exc_info=True)
            logger.warning("Falling back to stub mode")
    else:
        logger.info("Running in STUB mode (no ML models)")

    server = SocialSenseServer(config, pipeline=pipeline)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
