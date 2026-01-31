"""
API Router

REST API endpoints for session management, commands, and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, Dict, Any
from loguru import logger

from ..auth.api_key_validator import get_api_key_validator
from ..session.session_store import get_session_store
from ..streaming.video_stream import get_stream_registry


router = APIRouter()


# Request/Response models

class SessionCreateRequest(BaseModel):
    """Request to create a new session."""

    session_id: str
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Session information response."""

    session_id: str
    status: str
    created_at: float
    last_heartbeat: float
    metadata: Dict[str, Any]


class CommandRequest(BaseModel):
    """Voice or text command request."""

    session_id: str
    command: str
    command_type: str = "text"  # text or voice


class CommandResponse(BaseModel):
    """Command execution response."""

    success: bool
    message: str
    session_id: str


class StatusResponse(BaseModel):
    """System status response."""

    status: str
    active_sessions: int
    gpu_available: bool
    gpu_utilization: Optional[float]
    uptime_seconds: float


# Dependency for API key validation

async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Verify API key from header."""
    validator = get_api_key_validator()
    if not validator.validate_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# Session endpoints

@router.post("/session/start", response_model=SessionResponse)
async def start_session(
    request: SessionCreateRequest, api_key: str = Depends(verify_api_key)
):
    """
    Create a new streaming session.

    This endpoint should be called before establishing a WebSocket connection.
    """
    store = get_session_store()

    # Check if session already exists
    existing = await store.get_session(request.session_id)
    if existing:
        raise HTTPException(status_code=409, detail="Session already exists")

    # Create session
    session = await store.create_session(request.session_id, request.metadata)

    logger.info(f"Session created via API: {request.session_id}")

    return SessionResponse(
        session_id=session.session_id,
        status=session.status,
        created_at=session.created_at,
        last_heartbeat=session.last_heartbeat,
        metadata=session.metadata,
    )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, api_key: str = Depends(verify_api_key)):
    """Get session information."""
    store = get_session_store()
    session = await store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        session_id=session.session_id,
        status=session.status,
        created_at=session.created_at,
        last_heartbeat=session.last_heartbeat,
        metadata=session.metadata,
    )


@router.delete("/session/{session_id}/stop")
async def stop_session(session_id: str, api_key: str = Depends(verify_api_key)):
    """
    Stop and delete a session.

    This will close the WebSocket connection and cleanup all resources.
    """
    store = get_session_store()
    stream_registry = get_stream_registry()

    # Remove stream
    await stream_registry.remove_stream(session_id)

    # Delete session
    deleted = await store.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Session stopped via API: {session_id}")

    return {"success": True, "message": f"Session {session_id} stopped"}


# Command endpoints

@router.post("/command/text", response_model=CommandResponse)
async def send_text_command(
    request: CommandRequest, api_key: str = Depends(verify_api_key)
):
    """
    Send a text command to a session.

    This allows programmatic control of visual effects without voice input.
    """
    store = get_session_store()
    session = await store.get_session(request.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # TODO: Process command through pipeline
    # For now, just log it
    logger.info(f"Text command received for {request.session_id}: {request.command}")

    return CommandResponse(
        success=True,
        message=f"Command queued: {request.command}",
        session_id=request.session_id,
    )


@router.post("/command/voice", response_model=CommandResponse)
async def send_voice_command(
    request: CommandRequest, api_key: str = Depends(verify_api_key)
):
    """
    Send a pre-transcribed voice command to a session.

    Useful if client handles speech-to-text locally.
    """
    store = get_session_store()
    session = await store.get_session(request.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Voice command received for {request.session_id}: {request.command}")

    return CommandResponse(
        success=True,
        message=f"Voice command processed: {request.command}",
        session_id=request.session_id,
    )


# Status and monitoring endpoints

@router.get("/status", response_model=StatusResponse)
async def get_status(api_key: str = Depends(verify_api_key)):
    """
    Get server status and metrics.

    Provides information about active sessions, GPU status, and system health.
    """
    import time
    import torch

    store = get_session_store()
    session_count = await store.get_session_count()

    # Get GPU utilization if available
    gpu_utilization = None
    if torch.cuda.is_available():
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return StatusResponse(
        status="running",
        active_sessions=session_count,
        gpu_available=torch.cuda.is_available(),
        gpu_utilization=gpu_utilization,
        uptime_seconds=time.time(),  # TODO: Track actual uptime
    )


@router.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """
    Get detailed metrics for all active streams.

    Returns per-session metrics including latency, FPS, and buffer stats.
    """
    registry = get_stream_registry()
    metrics = await registry.get_all_metrics()

    return {"streams": metrics, "timestamp": __import__("time").time()}


@router.get("/health")
async def health():
    """
    Health check endpoint (no authentication required).

    Used by load balancers and monitoring systems.
    """
    import torch

    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
    }
