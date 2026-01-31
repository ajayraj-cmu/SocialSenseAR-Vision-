"""
Server Configuration

Central configuration for the streaming server with environment variable support.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Main server configuration."""

    # Server settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")
    DEBUG: bool = Field(default=False, description="Debug mode")

    # WebSocket settings
    WS_MAX_CONNECTIONS: int = Field(default=100, description="Max concurrent WebSocket connections")
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, description="WebSocket heartbeat interval (seconds)")
    WS_TIMEOUT: int = Field(default=300, description="WebSocket timeout (seconds)")

    # Video streaming settings
    VIDEO_WIDTH: int = Field(default=1280, description="Video frame width")
    VIDEO_HEIGHT: int = Field(default=720, description="Video frame height")
    VIDEO_FPS: int = Field(default=30, description="Target FPS")
    VIDEO_BITRATE: int = Field(default=5000000, description="Video bitrate (bps)")
    VIDEO_CODEC: str = Field(default="h264", description="Video codec")
    VIDEO_PRESET: str = Field(default="ultrafast", description="Encoding preset")

    # Frame buffer settings
    FRAME_BUFFER_SIZE: int = Field(default=30, description="Max frames in buffer")
    FRAME_DROP_THRESHOLD: int = Field(default=20, description="Drop frames if buffer exceeds this")

    # Latency optimization
    MAX_LATENCY_MS: int = Field(default=50, description="Target max latency (ms)")
    ADAPTIVE_QUALITY: bool = Field(default=True, description="Enable adaptive quality")
    LATENCY_WINDOW: int = Field(default=10, description="Latency measurement window")

    # Authentication
    JWT_SECRET_KEY: str = Field(default="", description="JWT secret key")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_EXPIRATION_MINUTES: int = Field(default=15, description="JWT token expiration")
    JWT_REFRESH_DAYS: int = Field(default=7, description="Refresh token expiration")
    API_KEY_HEADER: str = Field(default="X-API-Key", description="API key header name")

    # Session management
    SESSION_TIMEOUT_SECONDS: int = Field(default=300, description="Session idle timeout")
    SESSION_MAX_DURATION_SECONDS: int = Field(default=14400, description="Max session duration (4 hours)")

    # AWS settings
    AWS_REGION: str = Field(default="us-east-1", description="AWS region")
    DYNAMODB_TABLE_SESSIONS: str = Field(default="socialsensear-sessions", description="DynamoDB sessions table")
    DYNAMODB_TABLE_METRICS: str = Field(default="socialsensear-metrics", description="DynamoDB metrics table")

    # CloudWatch settings
    CLOUDWATCH_NAMESPACE: str = Field(default="SocialSenseAR", description="CloudWatch namespace")
    CLOUDWATCH_LOG_GROUP: str = Field(default="/socialsensear/server", description="CloudWatch log group")
    ENABLE_CLOUDWATCH: bool = Field(default=True, description="Enable CloudWatch integration")

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=60, description="Max requests per minute")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limit window (seconds)")

    # Pipeline settings (from main config)
    PIPELINE_CONFIG_PATH: str = Field(default="config/settings.yaml", description="Pipeline config path")
    ENABLE_VOICE_COMMANDS: bool = Field(default=True, description="Enable voice command processing")
    ENABLE_GEMINI_VISION: bool = Field(default=True, description="Enable Gemini Vision API")

    # Security
    ALLOWED_ORIGINS: list[str] = Field(default=["*"], description="CORS allowed origins")
    ENABLE_HTTPS: bool = Field(default=False, description="Enable HTTPS")
    SSL_CERT_PATH: Optional[str] = Field(default=None, description="SSL certificate path")
    SSL_KEY_PATH: Optional[str] = Field(default=None, description="SSL key path")

    # Monitoring
    METRICS_INTERVAL_SECONDS: int = Field(default=60, description="Metrics collection interval")
    ENABLE_PROMETHEUS: bool = Field(default=True, description="Enable Prometheus metrics")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class VideoConfig(BaseModel):
    """Video processing configuration."""

    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate: int = 5000000
    codec: str = "h264"
    preset: str = "ultrafast"
    gop_size: int = 30  # Keyframe interval
    pixel_format: str = "yuv420p"

    # Hardware acceleration
    use_gpu_encoding: bool = True
    gpu_encoder: str = "h264_nvenc"  # NVIDIA NVENC

    # Quality settings
    crf: int = 23  # Constant Rate Factor (0-51, lower = better quality)
    tune: str = "zerolatency"  # Optimize for low latency


class LatencyConfig(BaseModel):
    """Latency optimization configuration."""

    target_latency_ms: int = 50
    max_latency_ms: int = 100

    # Adaptive quality thresholds
    quality_degradation_threshold_ms: int = 60
    quality_upgrade_threshold_ms: int = 30

    # Quality levels (width, height, bitrate)
    quality_levels: dict[str, dict] = {
        "high": {"width": 1280, "height": 720, "bitrate": 5000000},
        "medium": {"width": 960, "height": 540, "bitrate": 3000000},
        "low": {"width": 640, "height": 480, "bitrate": 2000000},
    }

    # Frame dropping
    enable_frame_dropping: bool = True
    frame_drop_latency_threshold_ms: int = 80


# Global config instance
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """Get or create the global server configuration."""
    global _config
    if _config is None:
        _config = ServerConfig()

        # Generate JWT secret if not set
        if not _config.JWT_SECRET_KEY:
            import secrets
            _config.JWT_SECRET_KEY = secrets.token_urlsafe(32)

    return _config


def get_video_config() -> VideoConfig:
    """Get video configuration from server config."""
    config = get_config()
    return VideoConfig(
        width=config.VIDEO_WIDTH,
        height=config.VIDEO_HEIGHT,
        fps=config.VIDEO_FPS,
        bitrate=config.VIDEO_BITRATE,
        codec=config.VIDEO_CODEC,
        preset=config.VIDEO_PRESET,
    )


def get_latency_config() -> LatencyConfig:
    """Get latency optimization configuration."""
    config = get_config()
    return LatencyConfig(
        target_latency_ms=config.MAX_LATENCY_MS,
    )
