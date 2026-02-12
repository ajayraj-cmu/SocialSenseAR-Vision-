"""Modal GPU Offload for SocialSenseAR

Production-ready Modal deployment for SAM3 segmentation pipeline.
WebSocket handler runs INSIDE the GPU class so it has direct access to the
loaded pipeline — no cross-container .remote() calls.

Architecture:
- Single Modal GPU container runs BOTH the FastAPI/WebSocket server AND SAM3
- HuggingFace/Torch model cache persisted in Modal Volume
- WebSocket endpoint at /ws accepts binary protobuf ClientMessage
- Returns binary protobuf ServerMessage with RLE-encoded masks
- Same protocol as local server (drop-in replacement for Quest)

Usage:
    modal deploy modal_app.py           # Deploy to Modal cloud
    modal serve modal_app.py            # Development mode (hot reload)
    python tools/webcam_modal_client.py --url wss://<your-modal-url>/ws --show
    modal run modal_app.py              # Quick smoke test
"""

import os
import logging
import time
from typing import Optional

import modal

# Modal infra config (GPU, volumes, secrets, TRT)
import modal_config as config

# ============================================================
# Modal App + Volume + Secrets
# ============================================================

app = modal.App(config.get_app_name())

cache_volume = modal.Volume.from_name(
    config.get_cache_volume_name(),
    create_if_missing=True,
)

secrets = modal.Secret.from_name(config.SECRET_NAME)

# ============================================================
# Container Image
# ============================================================

image = (
    modal.Image.debian_slim(python_version=config.PYTHON_VERSION)
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6",
        "libxext6", "libxrender-dev", "libgomp1", "ffmpeg",
    )
    .pip_install(
        f"torch=={config.TORCH_VERSION}",
        f"torchvision=={config.TORCHVISION_VERSION}",
        index_url=f"https://download.pytorch.org/whl/{config.CUDA_VERSION}",
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
        "webrtcvad>=2.0.10",
        "mediapipe>=0.10.0",
        "sphn>=0.1.4,<0.2",
        "aiohttp>=3.9.0",
        "fastapi[standard]>=0.115.0",
        "uvicorn>=0.32.0",
        "tensorrt-cu12>=10.0",
    )
    .env({"PYTHONPATH": "/root"})
    .env({"HF_HOME": f"{config.CACHE_MOUNT_PATH}/huggingface"})
    .env({"TRANSFORMERS_CACHE": f"{config.CACHE_MOUNT_PATH}/huggingface"})
    .env({"TORCH_HOME": f"{config.CACHE_MOUNT_PATH}/torch"})
    .add_local_dir("server", remote_path="/root/server")
    .add_local_dir("config", remote_path="/root/config")
    .add_local_file("modal_config.py", remote_path="/root/modal_config.py")
)

# ============================================================
# GPU Container — Pipeline + WebSocket in ONE container
# ============================================================

@app.cls(
    image=image,
    gpu=config.get_gpu_spec(),
    secrets=[secrets],
    volumes={config.CACHE_MOUNT_PATH: cache_volume},
    timeout=config.CONTAINER_TIMEOUT,
    scaledown_window=getattr(config, 'CONTAINER_IDLE_TIMEOUT', 60),
)
@modal.concurrent(max_inputs=100)
class SocialSenseGPU:
    """Modal GPU container: SAM3 pipeline + WebSocket server in one process.

    Key insight: the @modal.asgi_app() method runs INSIDE this GPU class,
    so the WebSocket handler can call self.pipeline directly — no cross-
    container .remote() calls, no cold-start per frame, no timeouts.
    """

    pipeline: Optional[object] = None
    server_config: Optional[object] = None

    @modal.enter()
    def setup(self):
        """Load SAM3 models onto GPU when container starts."""
        import sys
        sys.path.insert(0, "/root")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logger = logging.getLogger("modal_app")
        logger.info("=" * 70)
        logger.info("SocialSenseAR GPU Container Starting...")
        logger.info("=" * 70)

        from server.config import ServerConfig
        from server.pipeline.orchestrator import PipelineOrchestrator

        # --- Copy TRT engines from volume to /root/ for auto-detection ---
        try:
            from modal_config import TRT_ENABLED, TRT_ENGINE_DIR
        except ImportError:
            TRT_ENABLED = True
            TRT_ENGINE_DIR = "/cache/trt_engines"

        if TRT_ENABLED:
            import shutil
            # Try GPU-specific dir first, fall back to flat dir (legacy)
            trt_search_dirs = [TRT_ENGINE_DIR]
            flat_dir = "/cache/trt_engines"
            if TRT_ENGINE_DIR != flat_dir and not os.path.isdir(TRT_ENGINE_DIR):
                trt_search_dirs.append(flat_dir)
            copied = []
            for trt_dir in trt_search_dirs:
                if not os.path.isdir(trt_dir):
                    continue
                for fname in os.listdir(trt_dir):
                    if fname.endswith((".engine", ".json")):
                        src = os.path.join(trt_dir, fname)
                        dst = os.path.join("/root", fname)
                        shutil.copy2(src, dst)
                        size_mb = os.path.getsize(dst) / 1024 / 1024
                        copied.append(f"{fname} ({size_mb:.0f}MB)")
                if copied:
                    logger.info(f"TRT engines from {trt_dir}: {', '.join(copied)}")
                    break  # Found engines, done
            if not copied:
                logger.info(f"No TRT engines found ({trt_search_dirs}) — using PyTorch fallback")

        # ServerConfig defaults are the single source of truth.
        # Only override Modal-specific values (device, debug, API keys, metrics path).
        self.server_config = ServerConfig(
            device="cuda",
            gpu_id=0,
            debug_view=False,  # Never GUI in Modal
            metrics_log_path=config.METRICS_LOG_PATH,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            hf_sam_token=os.getenv("HF_SAM_TOKEN") or os.getenv("HF_TOKEN"),
            hf_personaplex_token=os.getenv("HF_PERSONAPLEX_TOKEN") or os.getenv("HF_TOKEN"),
            # PersonaPlex — enable if env var is set
            personaplex_enabled=os.getenv("PERSONAPLEX_ENABLED", "").lower() in ("true", "1", "yes"),
            personaplex_url=os.getenv("PERSONAPLEX_URL", "ws://localhost:8998/api/chat"),
        )

        # Log HF token status (separate tokens for SAM3 vs PersonaPlex)
        hf_sam = os.getenv("HF_SAM_TOKEN") or os.getenv("HF_TOKEN")
        hf_pp = os.getenv("HF_PERSONAPLEX_TOKEN") or os.getenv("HF_TOKEN")
        logger.info(f"Pipeline: sam3 | GPU: cuda | SAM3: {self.server_config.sam3_model}")
        logger.info(f"HF_SAM_TOKEN: {'SET' if hf_sam else 'MISSING'}")
        logger.info(f"HF_PERSONAPLEX_TOKEN: {'SET' if hf_pp else 'MISSING'}")
        logger.info(f"HF_TOKEN (fallback): {'SET' if os.getenv('HF_TOKEN') else 'MISSING'}")
        if not hf_sam:
            logger.warning("HF_SAM_TOKEN (or HF_TOKEN) is required for SAM3. Add it to Modal secret 'socialsense-secrets' and request access to facebook/sam3 on HuggingFace.")

        # Set HF_TOKEN for SAM3 model loading (huggingface-hub uses HF_TOKEN env var)
        if hf_sam and not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_sam
            logger.info("Set HF_TOKEN from HF_SAM_TOKEN for SAM3 model loading")

        logger.info("Loading SAM3 model (this takes ~60-90s on cold start)...")
        self.pipeline = PipelineOrchestrator(self.server_config)

        # Initialize pipeline now so model loads at startup. If HF_SAM_TOKEN is missing
        # or facebook/sam3 access is not granted, this will fail and the container
        # won't become ready (check Modal logs).
        try:
            self.pipeline.initialize()
        except Exception as e:
            logger.error("Pipeline initialization failed (check HF_SAM_TOKEN / HF_TOKEN in Modal secret and facebook/sam3 access): %s", e, exc_info=True)
            raise

        # Persist downloaded models so next cold start is faster
        cache_volume.commit()

        logger.info("Pipeline ready! Segmentation will work on first frame.")
        logger.info("=" * 70)

    @modal.asgi_app()
    def web(self):
        """FastAPI app served from INSIDE the GPU container.

        Because this method is on SocialSenseGPU, it runs in the same
        process that loaded the SAM3 model. self.pipeline is available
        directly — no .remote() calls needed.
        """
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        import sys
        sys.path.insert(0, "/root")
        from server.proto import socialsense_pb2 as pb

        logger = logging.getLogger("modal_app.ws")

        web_app = FastAPI(title="SocialSenseAR GPU Server")

        @web_app.get("/")
        async def root():
            active = set()
            if self.pipeline:
                active = self.pipeline.get_active_prompts()
            return {
                "service": "SocialSenseAR GPU Server",
                "status": "online" if self.pipeline else "loading",
                "pipeline": "sam3",
                "active_prompts": sorted(active),
                "ws_endpoint": "/ws",
            }

        @web_app.get("/health")
        async def health():
            return {"status": "healthy", "pipeline_loaded": self.pipeline is not None}

        @web_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()

            remote = "unknown"
            if websocket.client:
                remote = f"{websocket.client.host}:{websocket.client.port}"
            logger.info(f"Client connected: {remote}")

            frame_count = 0
            t_start = time.time()
            last_masks_fid = -1  # track masks_frame_id changes

            try:
                while True:
                    raw_message = await websocket.receive_bytes()

                    msg = pb.ClientMessage()
                    msg.ParseFromString(raw_message)

                    payload_type = msg.WhichOneof("payload")

                    if payload_type == "frame":
                        if self.pipeline is None:
                            # Pipeline still loading; send empty response
                            resp = pb.ServerMessage()
                            resp.frame_id = msg.frame_id
                            resp.timestamp_ms = int(time.time() * 1000)
                            await websocket.send_bytes(resp.SerializeToString())
                            continue

                        # ---- Process frame DIRECTLY on this GPU ----
                        t0 = time.perf_counter()

                        result = self.pipeline.process_frame(
                            msg.frame.jpeg_data,
                            msg.frame.width,
                            msg.frame.height,
                            msg.frame_id,
                        )

                        proc_ms = (time.perf_counter() - t0) * 1000

                        # Build protobuf response
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

                            # Voice agent effects
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

                            if "listening" in conv_state:
                                response.conversation.voice_agent.listening = conv_state.get("listening", False)
                                response.conversation.voice_agent.recording = conv_state.get("recording", False)
                                response.conversation.voice_agent.partial_transcript = conv_state.get("partial_transcript", "")
                                response.conversation.voice_agent.last_command = conv_state.get("last_command", "")
                                response.conversation.voice_agent.last_response = conv_state.get("last_response", "")
                                response.conversation.voice_agent.last_command_time = conv_state.get("last_command_time", 0.0)

                        # Full-screen filter
                        if hasattr(self.pipeline, '_voice_agent') and self.pipeline._voice_agent:
                            fs_filter = self.pipeline._voice_agent.get_full_screen_filter()
                            if fs_filter:
                                response.conversation.voice_agent.full_screen_filter.filter_type = fs_filter.get("type", "none")
                                response.conversation.voice_agent.full_screen_filter.intensity = fs_filter.get("intensity", 0.5)
                                color_hex = fs_filter.get("color_hex")
                                if color_hex:
                                    response.conversation.voice_agent.full_screen_filter.color_hex = color_hex

                        response.metrics.total_ms = proc_ms
                        response.metrics.segment_count = len(response.segments)
                        response.metrics.fastsam_ms = result.fastsam_ms

                        await websocket.send_bytes(response.SerializeToString())

                        frame_count += 1

                        # Diagnostic: log mask updates at most every 5s
                        if response.masks_frame_id != last_masks_fid:
                            last_masks_fid = response.masks_frame_id
                            if response.segments:
                                _now_mask = time.time()
                                if not hasattr(self, '_last_mask_log'):
                                    self._last_mask_log = 0.0
                                if _now_mask - self._last_mask_log >= 5.0:
                                    seg_info = [(s.label, f"conf={s.confidence:.3f}", f"rle={len(s.rle_mask)}B")
                                                for s in response.segments if s.rle_mask]
                                    logger.info(
                                        f"[{remote}] MASK UPDATE: fid={response.masks_frame_id} "
                                        f"| {len(response.segments)} segs | {seg_info}"
                                    )
                                    self._last_mask_log = _now_mask

                        if frame_count % 250 == 0:
                            elapsed = time.time() - t_start
                            fps = frame_count / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"[{remote}] {frame_count} frames | "
                                f"{fps:.1f} fps | {proc_ms:.0f}ms/frame | "
                                f"{len(response.segments)} segs"
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

    @modal.method()
    def process_frame_rpc(
        self,
        jpeg_data: bytes,
        width: int,
        height: int,
        frame_id: int,
    ) -> bytes:
        """RPC entry point for non-WebSocket callers (e.g. modal run tests)."""
        import sys
        sys.path.insert(0, "/root")
        from server.proto import socialsense_pb2 as pb

        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")

        result = self.pipeline.process_frame(jpeg_data, width, height, frame_id)

        response = pb.ServerMessage()
        response.frame_id = frame_id
        response.timestamp_ms = int(time.time() * 1000)
        response.processing_time_ms = result.total_ms
        response.masks_frame_id = result.masks_frame_id
        response.masks_updated = not result.decoder_skipped

        for seg in result.segments:
            pb_seg = response.segments.add()
            pb_seg.label = (seg.label or "").lstrip("~")
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

        return response.SerializeToString()


# ============================================================
# Local Development / Smoke Test
# ============================================================

@app.local_entrypoint()
def main():
    """Smoke test: send one dummy frame to verify GPU pipeline works.

    Usage: modal run modal_app.py
    """
    import cv2
    import numpy as np

    print("Creating test frame...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (100, 150, 200)

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpeg_bytes = jpeg.tobytes()

    print(f"Test frame: {len(jpeg_bytes)} bytes")
    print("Processing on Modal GPU...")

    gpu = SocialSenseGPU()
    response_bytes = gpu.process_frame_rpc.remote(jpeg_bytes, 640, 480, 1)

    import sys
    sys.path.insert(0, ".")
    from server.proto import socialsense_pb2 as pb

    response = pb.ServerMessage()
    response.ParseFromString(response_bytes)

    print(f"Response: {len(response.segments)} segments")
    print(f"Processing time: {response.processing_time_ms:.1f}ms")
    print("Pipeline working on Modal GPU!")
