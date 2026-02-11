"""
PersonaPlex on Modal — A100 40GB deployment.

Setup (one-time):
    pip install modal
    modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_your_token_here

Pre-download model weights (one-time, avoids slow first startup):
    modal run deploy_modal.py::download_models

Run in dev mode (hot-reload, temporary URL):
    modal serve deploy_modal.py

Deploy to production (persistent URL):
    modal deploy deploy_modal.py
"""

import modal

app = modal.App("personaplex")

hf_cache = modal.Volume.from_name("personaplex-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libopus-dev", "libportaudio2", "pkg-config", "build-essential")
    .run_commands("pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124")
    .pip_install(
        "numpy>=1.26,<2.2",
        "safetensors>=0.4.0,<0.5",
        "huggingface-hub>=0.24,<0.25",
        "einops==0.7",
        "sentencepiece==0.2",
        "sounddevice==0.5",
        "sphn>=0.1.4,<0.2",
        "aiohttp>=3.10.5,<3.11",
    )
    .env({"NO_TORCH_COMPILE": "1"})
    .copy_local_dir("moshi", "/app/moshi")
    .run_commands("pip install /app/moshi")
)


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def download_models():
    """Pre-download model weights into the persistent volume."""
    from huggingface_hub import hf_hub_download

    repo = "nvidia/personaplex-7b-v1"
    files = [
        "model.safetensors",
        "tokenizer-e351c8d8-checkpoint125.safetensors",
        "tokenizer_spm_32k_3.model",
        "voices.tgz",
        "dist.tgz",
        "config.json",
    ]
    for name in files:
        print(f"Downloading {name}...")
        hf_hub_download(repo, name)
    hf_cache.commit()
    print("All model files cached.")


@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    container_idle_timeout=300,
)
@modal.web_server(port=8998, startup_timeout=600)
def serve():
    import subprocess
    import sys

    subprocess.Popen([
        sys.executable, "-m", "moshi.server",
        "--host", "0.0.0.0",
        "--port", "8998",
    ])
