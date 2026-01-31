"""
SocialSenseAR Streaming Server

Main WebSocket server for real-time video streaming from AR headsets.
Handles bidirectional communication, session management, and integration
with the vision processing pipeline.
"""

import asyncio
import time
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import numpy as np

from .config import get_config
from .auth.jwt_manager import get_jwt_manager
from .auth.api_key_validator import get_api_key_validator
from .session.session_store import get_session_store
from .streaming.video_stream import get_stream_registry
from .api import router as api_router


# Global state
class ServerState:
    """Global server state."""

    def __init__(self):
        self.config = get_config()
        self.jwt_manager = get_jwt_manager()
        self.api_key_validator = get_api_key_validator()
        self.session_store = get_session_store()
        self.stream_registry = get_stream_registry()
        self.active_connections: Dict[str, WebSocket] = {}
        self.cleanup_task: Optional[asyncio.Task] = None


server_state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting SocialSenseAR Streaming Server...")

    # Start background cleanup task
    server_state.cleanup_task = asyncio.create_task(cleanup_inactive_sessions())

    logger.success("Server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down server...")

    # Cancel cleanup task
    if server_state.cleanup_task:
        server_state.cleanup_task.cancel()
        try:
            await server_state.cleanup_task
        except asyncio.CancelledError:
            pass

    # Close all active connections
    for session_id, ws in list(server_state.active_connections.items()):
        try:
            await ws.close()
        except Exception:
            pass

    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="SocialSenseAR Streaming Server",
    description="Real-time AR vision processing server with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=server_state.config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router.router, prefix="/api/v1")


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """
    Dependency to verify API key from header.

    Args:
        x_api_key: API key from header.

    Returns:
        API key if valid.

    Raises:
        HTTPException: If API key is invalid.
    """
    if not server_state.api_key_validator.validate_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "SocialSenseAR Streaming Server",
        "version": "1.0.0",
        "status": "running",
        "active_sessions": len(server_state.active_connections),
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancer.

    Returns:
        Health status and metrics.
    """
    import torch

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions": len(server_state.active_connections),
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    WebSocket endpoint for bidirectional video streaming.

    Protocol:
    - Client sends: {"type": "frame", "data": <base64_encoded_frame>, "timestamp": <float>}
    - Server sends: {"type": "frame", "data": <base64_encoded_frame>, "timestamp": <float>}
    - Heartbeat: {"type": "ping"} / {"type": "pong"}

    Args:
        websocket: WebSocket connection.
        session_id: Unique session identifier.
        api_key: Validated API key.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for session {session_id}")

    # Register connection
    server_state.active_connections[session_id] = websocket

    # Create session
    session = await server_state.session_store.create_session(session_id)

    # Create video stream
    try:
        stream = await server_state.stream_registry.create_stream(session_id)
    except ValueError as e:
        await websocket.close(code=1008, reason=str(e))
        return

    # Start processing task
    processing_task = asyncio.create_task(process_stream(session_id, stream))

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            msg_type = data.get("type")

            if msg_type == "ping":
                # Respond to heartbeat
                await websocket.send_json({"type": "pong"})
                await server_state.session_store.update_heartbeat(session_id)

            elif msg_type == "frame":
                # Receive frame from client
                frame_data = data.get("data")  # Base64 encoded or numpy serialized
                timestamp = data.get("timestamp", time.time())

                if frame_data:
                    # Decode frame (from base64 or other format)
                    import base64
                    import io

                    try:
                        # Decode base64
                        decoded = base64.b64decode(frame_data)
                        # Load numpy array
                        frame = np.load(io.BytesIO(decoded), allow_pickle=False)

                        # Send to stream for processing
                        await stream.receive_frame(frame.tobytes(), timestamp)

                    except Exception as e:
                        logger.error(f"Error decoding frame: {e}")

            elif msg_type == "command":
                # Voice or text command from client
                command_text = data.get("command")
                if command_text:
                    logger.info(f"Received command from {session_id}: {command_text}")
                    # TODO: Process command through pipeline
                    # For now, just acknowledge
                    await websocket.send_json(
                        {"type": "command_ack", "command": command_text}
                    )

            else:
                logger.warning(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")

    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")

    finally:
        # Cleanup
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass

        server_state.active_connections.pop(session_id, None)
        await server_state.stream_registry.remove_stream(session_id)
        await server_state.session_store.delete_session(session_id)

        logger.info(f"Cleaned up session {session_id}")


async def process_stream(session_id: str, stream):
    """
    Background task to process video stream.

    This integrates with the existing pipeline to process incoming frames
    and send processed frames back to the client.

    Args:
        session_id: Session identifier.
        stream: VideoStreamManager instance.
    """
    # Import pipeline here to avoid circular imports
    from ..pipeline.orchestrator import PipelineOrchestrator, PipelineConfig
    import yaml
    from pathlib import Path

    # Load pipeline config
    config_path = Path(server_state.config.PIPELINE_CONFIG_PATH)
    if config_path.exists():
        with open(config_path) as f:
            pipeline_config_dict = yaml.safe_load(f)
        pipeline_config = PipelineConfig(**pipeline_config_dict)
    else:
        pipeline_config = PipelineConfig()

    # Create pipeline instance for this session
    # Note: In production, you might want to pool pipeline instances
    # to avoid creating too many GPU contexts
    pipeline = PipelineOrchestrator(config=pipeline_config)

    logger.info(f"Started processing pipeline for session {session_id}")

    try:
        while True:
            # Get next frame to process
            result = await stream.get_next_frame_for_processing(timeout=0.1)

            if not result:
                # No frame available, wait a bit
                await asyncio.sleep(0.01)
                continue

            frame, timestamp = result

            # Process frame through pipeline
            # Note: pipeline.process_frame() is synchronous, so run in executor
            loop = asyncio.get_event_loop()
            processed_frame = await loop.run_in_executor(
                None, pipeline.process_frame, frame
            )

            # Send processed frame back to client
            await stream.send_processed_frame(processed_frame, timestamp)

            # Send to client via WebSocket
            ws = server_state.active_connections.get(session_id)
            if ws:
                # Get encoded frame
                encoded = await stream.get_encoded_frame_for_client()

                if encoded:
                    import base64

                    # Send as base64
                    encoded_b64 = base64.b64encode(encoded).decode("utf-8")

                    await ws.send_json(
                        {
                            "type": "frame",
                            "data": encoded_b64,
                            "timestamp": time.time(),
                            "latency_ms": stream.metrics.avg_latency_ms,
                        }
                    )

    except asyncio.CancelledError:
        logger.info(f"Processing stopped for session {session_id}")
        pipeline.cleanup()
        raise

    except Exception as e:
        logger.error(f"Processing error for session {session_id}: {e}")
        pipeline.cleanup()


async def cleanup_inactive_sessions():
    """Background task to cleanup inactive sessions."""
    while True:
        try:
            await asyncio.sleep(60)  # Run every minute

            # Cleanup inactive streams
            await server_state.stream_registry.cleanup_inactive_streams(
                timeout_seconds=server_state.config.SESSION_TIMEOUT_SECONDS
            )

            # Cleanup expired sessions
            await server_state.session_store.cleanup_expired_sessions()

            logger.debug("Completed session cleanup")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")


if __name__ == "__main__":
    import uvicorn

    config = get_config()

    uvicorn.run(
        "src.server.streaming_server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info",
    )
