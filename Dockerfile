# Multi-stage build for SocialSenseAR with GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Audio dependencies
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    # Video codecs
    libx264-dev \
    # Utilities
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
COPY requirements-server.txt .

# Install PyTorch with CUDA support first (before other dependencies)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install server-specific dependencies
RUN pip install --no-cache-dir -r requirements-server.txt

# Download spaCy model if needed
RUN python -m spacy download en_core_web_sm || true

# Copy application code
COPY src/ ./src/
COPY server/ ./server/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create directories for runtime
RUN mkdir -p /app/logs /app/output /app/recordings

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Server mode environment variables (can be overridden)
ENV SERVER_MODE=true
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose ports for server mode
# 8000: HTTP/WebSocket server
# 3478: STUN server (for WebRTC)
# 49152-49200: RTP ports range (for WebRTC media)
EXPOSE 8000 3478 49152-49200/udp

# Health check for server mode
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run streaming server
CMD ["python", "-m", "uvicorn", "server.streaming_server:app", "--host", "0.0.0.0", "--port", "8000"]
