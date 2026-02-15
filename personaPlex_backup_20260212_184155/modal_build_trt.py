"""Build TensorRT engines for SAM3 on Modal A10G GPU.

One-shot function: exports ONNX from PyTorch, builds TRT FP16 engines,
and saves everything to the Modal volume for the deployed app to load.

Usage:
    modal run modal_build_trt.py          # Build both vision + decoder engines
    modal run modal_build_trt.py --decoder-only  # Rebuild decoder only (faster)

After building, redeploy:
    modal deploy modal_app.py
"""

import os
import logging
import time

import modal

try:
    import modal_config as config
except ImportError:
    class config:
        APP_NAME = "socialsense-ar-gpu"
        GPU_TYPE = "A10G"
        CACHE_VOLUME_NAME = "socialsense-cache"
        SECRET_NAME = "socialsense-secrets"
        CACHE_MOUNT_PATH = "/cache"
        PYTHON_VERSION = "3.12"
        TORCH_VERSION = "2.5.1"
        TORCHVISION_VERSION = "0.20.1"
        CUDA_VERSION = "cu121"
        TRT_ENGINE_DIR = "/cache/trt_engines"
        TRT_ENABLED = True

        @staticmethod
        def get_app_name():
            return "socialsense-ar-gpu"

        @staticmethod
        def get_cache_volume_name():
            return "socialsense-cache"

        @staticmethod
        def get_gpu_spec():
            return "A10G"


app = modal.App("socialsense-trt-builder")

cache_volume = modal.Volume.from_name(
    config.get_cache_volume_name(),
    create_if_missing=True,
)

secrets = modal.Secret.from_name(config.SECRET_NAME)

# Build image: same base as main app + TensorRT + ONNX Runtime
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
        "protobuf>=4.25.0",
        "numpy>=1.24.0,<3.0.0",
        "opencv-python-headless>=4.8.0",
        "transformers>=5.0.0",
        "huggingface-hub>=1.3.0,<2.0.0",
        "tokenizers>=0.22.0",
        "safetensors>=0.4.3",
        "Pillow>=10.0.0",
        "scikit-learn>=1.5.0",
        "google-genai>=1.0.0",
        "openai>=1.0.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.17.0",
        "tensorrt-cu12>=10.0",
    )
    .env({"PYTHONPATH": "/root"})
    .env({"HF_HOME": f"{config.CACHE_MOUNT_PATH}/huggingface"})
    .env({"TRANSFORMERS_CACHE": f"{config.CACHE_MOUNT_PATH}/huggingface"})
    .env({"TORCH_HOME": f"{config.CACHE_MOUNT_PATH}/torch"})
    .add_local_dir("server", remote_path="/root/server")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_file("modal_config.py", remote_path="/root/modal_config.py")
)


@app.function(
    image=image,
    gpu=config.get_gpu_spec(),
    secrets=[secrets],
    volumes={config.CACHE_MOUNT_PATH: cache_volume},
    timeout=1800,  # 30 min — engine builds can take 15-20 min
)
def build_trt_engines(decoder_only: bool = False):
    """Export ONNX and build TRT FP16 engines for SAM3.

    Builds:
    - sam3_vision_<res>.engine (vision encoder, ~10 min build)
    - sam3_decoder_<res>.engine (full decoder, 200 queries — multi-instance)
    - sam3_topk_decoder_<res>.engine (top-k decoder, 1 query — faster fallback)
    - sam3_meta_<res>.json (metadata for runtime)

    All saved to volume at TRT_ENGINE_DIR.
    """
    import sys
    sys.path.insert(0, "/root")

    import torch
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("trt_builder")

    from server.config import ServerConfig
    server_config = ServerConfig()
    resolution = server_config.sam3_resolution
    trt_dir = getattr(config, "TRT_ENGINE_DIR", "/cache/trt_engines")
    os.makedirs(trt_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SAM3 TensorRT Engine Builder")
    logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  Output dir: {trt_dir}")
    logger.info(f"  Decoder only: {decoder_only}")
    logger.info("=" * 70)

    # ── Step 1: Load SAM3 model ─────────────────────────────────────
    from transformers import Sam3Model, Sam3Processor, Sam3Config

    sam3_model = server_config.sam3_model
    logger.info("Loading SAM3 model and processor...")
    t0 = time.perf_counter()
    processor = Sam3Processor.from_pretrained(sam3_model)

    if resolution != 1008:
        sam3_config = Sam3Config.from_pretrained(sam3_model)
        sam3_config.vision_config.image_size = resolution
        model = Sam3Model.from_pretrained(sam3_model, config=sam3_config)
    else:
        model = Sam3Model.from_pretrained(sam3_model)

    device = "cuda"
    model.to(device).eval().half()
    logger.info(f"Model loaded: {(time.perf_counter() - t0)*1000:.0f}ms")

    # Discover max token length across all prompts
    from PIL import Image
    from server.vision.sam3_segmenter import ALL_PROMPTS

    dummy_img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    max_token_len = 0
    for prompt in ALL_PROMPTS:
        inputs = processor(images=dummy_img, text=prompt, return_tensors="pt")
        max_token_len = max(max_token_len, inputs["input_ids"].shape[1])
    logger.info(f"Max token length: {max_token_len}")

    # ── Step 2: Export ONNX ─────────────────────────────────────────
    from server.vision.sam3_export import (
        export_vision_encoder, export_topk_decoder, save_metadata,
    )

    # Export vision encoder ONNX (needed for TRT build + metadata)
    logger.info("")
    logger.info("Exporting vision encoder to ONNX...")
    vis_onnx, pv_shape, vis_shapes, vis_names = export_vision_encoder(
        model, processor, device, trt_dir, resolution=resolution,
    )

    # Export top-k decoder ONNX
    logger.info("")
    logger.info("Exporting top-k decoder to ONNX...")
    topk_onnx = export_topk_decoder(
        model, processor, device, trt_dir, max_token_len, resolution=resolution,
    )

    # We also need the full decoder export to get decoder_input_shapes for metadata
    from server.vision.sam3_export import export_decoder
    logger.info("")
    logger.info("Exporting full decoder to ONNX (for metadata)...")
    dec_onnx, dec_info, dec_names, dec_input_shapes = export_decoder(
        model, processor, device, trt_dir, max_token_len, resolution=resolution,
    )

    # Save metadata
    save_metadata(
        trt_dir, processor, pv_shape, vis_shapes, vis_names,
        dec_info, dec_names, dec_input_shapes, max_token_len,
        resolution=resolution,
    )
    logger.info(f"Metadata saved: sam3_meta_{resolution}.json")

    # Free PyTorch model to give TRT builder maximum GPU memory
    del model
    torch.cuda.empty_cache()
    logger.info(f"Freed PyTorch model — {torch.cuda.memory_allocated()/1024**2:.0f}MB GPU in use")

    # ── Step 3: Build TRT engines ───────────────────────────────────
    # Import builder from scripts/
    sys.path.insert(0, "/root/scripts")
    from build_trt_engine import build_engine

    # Build full decoder engine (200 queries — supports multi-instance masks)
    full_decoder_engine = os.path.join(trt_dir, f"sam3_decoder_{resolution}.engine")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Building TRT full decoder engine (FP16, multi-instance)...")
    logger.info("=" * 70)
    t0 = time.perf_counter()
    build_engine(dec_onnx, full_decoder_engine)
    dec_build_s = time.perf_counter() - t0
    logger.info(f"Full decoder engine built in {dec_build_s:.0f}s")

    # Build top-k decoder engine (1 query — faster but single-instance only)
    topk_decoder_engine = os.path.join(trt_dir, f"sam3_topk_decoder_{resolution}.engine")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Building TRT top-k decoder engine (FP16, single-instance)...")
    logger.info("=" * 70)
    t0 = time.perf_counter()
    build_engine(topk_onnx, topk_decoder_engine)
    topk_build_s = time.perf_counter() - t0
    logger.info(f"Top-k decoder engine built in {topk_build_s:.0f}s")

    # Build vision encoder engine (195ms → 15-25ms)
    if not decoder_only:
        vision_engine = os.path.join(trt_dir, f"sam3_vision_{resolution}.engine")
        logger.info("")
        logger.info("=" * 70)
        logger.info("Building TRT vision engine (FP16)...")
        logger.info("=" * 70)
        t0 = time.perf_counter()
        build_engine(vis_onnx, vision_engine)
        vis_build_s = time.perf_counter() - t0
        logger.info(f"Vision engine built in {vis_build_s:.0f}s")

    # ── Step 4: Commit to volume ────────────────────────────────────
    cache_volume.commit()

    # List results
    logger.info("")
    logger.info("=" * 70)
    logger.info("BUILD COMPLETE — Engines on volume:")
    logger.info("=" * 70)
    for f in sorted(os.listdir(trt_dir)):
        fpath = os.path.join(trt_dir, f)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        logger.info(f"  {f}: {size_mb:.1f} MB")

    logger.info("")
    logger.info("Next: modal deploy modal_app.py")
    logger.info("The app will auto-copy engines from volume to /root/ at startup.")


@app.local_entrypoint()
def main(decoder_only: bool = False):
    """Build TRT engines on Modal GPU.

    Usage:
        modal run modal_build_trt.py
        modal run modal_build_trt.py --decoder-only
    """
    print("Starting TRT engine build on Modal A10G...")
    print("This will take ~15-20 minutes.")
    print()
    build_trt_engines.remote(decoder_only=decoder_only)
    print()
    print("Done! Now run: modal deploy modal_app.py")
