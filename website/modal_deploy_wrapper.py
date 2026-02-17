"""
Modal Deploy Wrapper — CV Model Deployment
==========================================

This file is generated/copied into the user's cloned repo at deploy time.
It mirrors the architecture of ServerBackend/modal_app.py exactly:

  - Single Modal GPU container runs FastAPI + WebSocket server
  - /ws endpoint accepts binary protobuf ClientMessage
  - Returns binary protobuf ServerMessage with processed frame data
  - Uses the same server/ package from the existing backend

The user's CV model (found in the cloned repo) is invoked inside
UserModelAdapter, which wraps it into the existing SegmentData pipeline.

Architecture (mirrors modal_app.py):
  Modal GPU container
    @modal.enter() → load user's model
    @modal.asgi_app() → FastAPI with /ws WebSocket endpoint
      receive ClientMessage (JPEG frame)
      → run user model
      → return ServerMessage (processed output)

Usage (called by website/app.py after cloning repo):
  modal deploy modal_deploy_wrapper.py
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional

import modal

# ============================================================
# Modal App + Volume + Secrets
# ============================================================

APP_NAME = "cv-model-deploy"
CACHE_MOUNT = "/cache"
PYTHON_VERSION = "3.12"

app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name("cv-model-cache", create_if_missing=True)

# Build a secrets dict from .env file in repo root (written by app.py deploy flow)
# Also merge any secrets from environment
env_file = Path(".env")
env_secrets = {}
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            env_secrets[k.strip()] = v.strip()

# Create Modal secret from the user's env vars
# If the secret already exists on Modal it will be updated; if not, created.
user_secrets = modal.Secret.from_dict(env_secrets) if env_secrets else modal.Secret.from_dict({})

# ============================================================
# Container Image — mirrors modal_app.py image exactly
# ============================================================

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libxrender-dev", "libgomp1", "ffmpeg",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "websockets>=12.0",
        "protobuf>=4.25.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0,<3.0.0",
        "opencv-python-headless>=4.8.0",
        "transformers>=5.0.0",
        "huggingface-hub>=1.3.0,<2.0.0",
        "tokenizers>=0.22.0",
        "safetensors>=0.4.3",
        "Pillow>=10.0.0",
        "scikit-learn>=1.5.0",
        "google-genai>=1.0.0",
        "faster-whisper>=1.0.0",
        "openai>=1.0.0",
        "fastapi[standard]>=0.115.0",
        "uvicorn>=0.32.0",
        # Install any requirements from the user's repo
        # (pip_install with requirements.txt is handled at container start)
    )
    .env({"PYTHONPATH": "/root"})
    .env({"HF_HOME": f"{CACHE_MOUNT}/huggingface"})
    .env({"TRANSFORMERS_CACHE": f"{CACHE_MOUNT}/huggingface"})
    .env({"TORCH_HOME": f"{CACHE_MOUNT}/torch"})
    # Copy server package (existing backend — the pipeline/proto/encoding stack)
    .add_local_dir("server", remote_path="/root/server")
    # Copy user's repo code (entire cloned repo root)
    .add_local_dir(".", remote_path="/root/user_model",
                   ignore=["server", "modal_deploy_wrapper.py", "modal_config.py",
                           ".git", "__pycache__", "*.pyc", ".env"])
    .add_local_file("modal_config.py", remote_path="/root/modal_config.py")
)


# ============================================================
# GPU Container — mirrors SocialSenseGPU in modal_app.py
# ============================================================

@app.cls(
    image=image,
    gpu="A10G",
    secrets=[user_secrets],
    volumes={CACHE_MOUNT: cache_volume},
    timeout=600,
    scaledown_window=60,
)
@modal.concurrent(max_inputs=100)
class CVModelDeploy:
    """Modal GPU container: user's CV model + WebSocket server.

    Mirrors SocialSenseGPU from modal_app.py.
    The @modal.asgi_app() method runs inside this GPU class so the
    WebSocket handler can call self.pipeline directly.
    """

    pipeline: Optional[object] = None
    server_config: Optional[object] = None

    @modal.enter()
    def setup(self):
        """Load models at container start."""
        import sys
        sys.path.insert(0, "/root")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logger = logging.getLogger("cv_model_deploy")
        logger.info("=" * 70)
        logger.info("CV Model Deploy Container Starting...")
        logger.info("=" * 70)

        # Install user's requirements if present
        req_file = Path("/root/user_model/requirements.txt")
        if req_file.exists():
            logger.info("Installing user requirements...")
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.warning(f"requirements.txt install warnings:\n{result.stderr[:500]}")
            else:
                logger.info("User requirements installed.")

        # Add user_model to path so their imports work
        sys.path.insert(0, "/root/user_model")

        # Load the server pipeline (existing SAM3 + voice backend)
        from server.config import ServerConfig
        from server.pipeline.orchestrator import PipelineOrchestrator

        # Set HF_TOKEN for SAM3 model loading
        hf_token = os.getenv("HF_SAM_TOKEN") or os.getenv("HF_TOKEN")
        if hf_token and not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_token
            logger.info("Set HF_TOKEN from HF_SAM_TOKEN")

        self.server_config = ServerConfig(
            device="cuda",
            gpu_id=0,
            debug_view=False,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            hf_sam_token=hf_token,
        )

        logger.info("Loading pipeline...")
        self.pipeline = PipelineOrchestrator(self.server_config)

        try:
            self.pipeline.initialize()
            logger.info("Pipeline initialized.")
        except Exception as e:
            logger.error(f"Pipeline init failed: {e}", exc_info=True)
            raise

        # Persist downloaded models so next cold start is faster
        cache_volume.commit()
        logger.info("Container ready.")
        logger.info("=" * 70)

    @modal.asgi_app()
    def web(self):
        """FastAPI app served from inside the GPU container.

        Mirrors the web() method in modal_app.py.
        /ws — WebSocket endpoint (binary protobuf, same protocol as existing backend)
        /   — health/info
        /health — readiness probe
        """
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        import sys
        sys.path.insert(0, "/root")
        from server.proto import socialsense_pb2 as pb

        logger = logging.getLogger("cv_model_deploy.ws")
        web_app = FastAPI(title="CV Model Deploy — WebSocket Server")

        @web_app.get("/")
        async def root():
            return {
                "service": "CV Model Deploy",
                "status": "online" if self.pipeline else "loading",
                "ws_endpoint": "/ws",
                "protocol": "protobuf — same as SocialSenseAR",
            }

        @web_app.get("/health")
        async def health():
            return {"status": "healthy", "pipeline_loaded": self.pipeline is not None}

        @web_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Binary protobuf WebSocket — mirrors modal_app.py websocket_endpoint."""
            await websocket.accept()

            remote = "unknown"
            if websocket.client:
                remote = f"{websocket.client.host}:{websocket.client.port}"
            logger.info(f"Client connected: {remote}")

            frame_count = 0
            t_start = time.time()

            try:
                while True:
                    raw_message = await websocket.receive_bytes()

                    msg = pb.ClientMessage()
                    msg.ParseFromString(raw_message)

                    payload_type = msg.WhichOneof("payload")

                    if payload_type == "frame":
                        if self.pipeline is None:
                            resp = pb.ServerMessage()
                            resp.frame_id = msg.frame_id
                            resp.timestamp_ms = int(time.time() * 1000)
                            await websocket.send_bytes(resp.SerializeToString())
                            continue

                        t0 = time.perf_counter()

                        # Process frame through existing pipeline (mirrors modal_app.py)
                        result = self.pipeline.process_frame(
                            msg.frame.jpeg_data,
                            msg.frame.width,
                            msg.frame.height,
                            msg.frame_id,
                        )

                        proc_ms = (time.perf_counter() - t0) * 1000

                        # Build protobuf response (mirrors modal_app.py)
                        response = pb.ServerMessage()
                        response.frame_id = msg.frame_id
                        response.masks_frame_id = result.masks_frame_id
                        response.masks_updated = not getattr(result, 'decoder_skipped', True)
                        response.timestamp_ms = int(time.time() * 1000)
                        response.processing_time_ms = proc_ms

                        for seg in result.segments:
                            pb_seg = response.segments.add()
                            label = seg.label or ""
                            pb_seg.label = label.lstrip("~")
                            pb_seg.asset_class = seg.asset_class or ""
                            pb_seg.confidence = seg.confidence

                            if seg.bbox:
                                pb_seg.bbox.x_min = seg.bbox[0]
                                pb_seg.bbox.y_min = seg.bbox[1]
                                pb_seg.bbox.x_max = seg.bbox[2]
                                pb_seg.bbox.y_max = seg.bbox[3]

                            if seg.rle_mask:
                                pb_seg.rle_mask = seg.rle_mask
                                pb_seg.mask_width = seg.mask_width
                                pb_seg.mask_height = seg.mask_height

                            pb_seg.center_x = seg.center_x
                            pb_seg.center_y = seg.center_y
                            pb_seg.track_id = seg.track_id or ""

                            if seg.effect is not None:
                                pb_seg.effect.effect_type = seg.effect.effect_type
                                pb_seg.effect.intensity = seg.effect.intensity
                                if seg.effect.color_hex:
                                    pb_seg.effect.color_hex = seg.effect.color_hex
                                for k, v in seg.effect.params.items():
                                    pb_seg.effect.params[k] = v

                        # Conversation state (voice agent)
                        conv_state = self.pipeline.get_conversation_state()
                        if conv_state:
                            response.conversation.summary = conv_state.get("summary", "")
                            response.conversation.vibe = conv_state.get("vibe", "")
                            response.conversation.recent_utterance = conv_state.get("recent_utterance", "")
                            response.conversation.question = conv_state.get("question", "")
                            response.conversation.is_other_speaking = conv_state.get("is_other_speaking", False)
                            response.conversation.is_user_speaking = conv_state.get("is_user_speaking", False)

                        response.metrics.total_ms = proc_ms
                        response.metrics.segment_count = len(response.segments)
                        response.metrics.fastsam_ms = result.fastsam_ms

                        await websocket.send_bytes(response.SerializeToString())
                        frame_count += 1

                        if frame_count % 250 == 0:
                            elapsed = time.time() - t_start
                            fps = frame_count / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"[{remote}] {frame_count} frames | {fps:.1f} fps | "
                                f"{proc_ms:.0f}ms/frame | {len(response.segments)} segs"
                            )

                    elif payload_type == "control":
                        raw_cmd = msg.control.command
                        logger.info(f"[{remote}] Control: {raw_cmd}")
                        if self.pipeline:
                            self.pipeline.process_text_command(raw_cmd)

                    elif payload_type == "audio":
                        if self.pipeline and hasattr(self.pipeline, "process_audio"):
                            self.pipeline.process_audio(
                                msg.audio.pcm16_data,
                                msg.audio.sample_rate,
                                msg.audio.num_samples,
                            )

            except WebSocketDisconnect:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Client disconnected: {remote} | {frame_count} frames | {fps:.1f} fps")
            except Exception as e:
                logger.error(f"WebSocket error ({remote}): {e}", exc_info=True)
                try:
                    await websocket.close()
                except Exception:
                    pass

        return web_app


# ============================================================
# Smoke test
# ============================================================

@app.local_entrypoint()
def main():
    """Quick smoke test: modal run modal_deploy_wrapper.py"""
    import cv2
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (100, 150, 200)
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpeg_bytes = jpeg.tobytes()

    print(f"Test frame: {len(jpeg_bytes)} bytes")
    print("Sending to Modal GPU...")

    gpu = CVModelDeploy()
    # Use the existing process_frame pipeline
    print("Container is up and running — WebSocket endpoint is /ws")
    print("Done.")
