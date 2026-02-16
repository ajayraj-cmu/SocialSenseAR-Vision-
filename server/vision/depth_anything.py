"""Depth Anything V2 estimator — monocular depth estimation.

GitHub: https://github.com/LiheYoung/Depth-Anything-V2
HuggingFace: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

Install options:
  Option A (HuggingFace transformers):
    pip install transformers torch
    # Model auto-downloaded from HuggingFace

  Option B (clone repo):
    git clone https://github.com/LiheYoung/Depth-Anything-V2
    pip install -r Depth-Anything-V2/requirements.txt

Config keys (in cv_models.yaml):
  depth-anything:
    model_variant: small    # small, base, large
    device: cuda
    encoder: vits           # vits (small), vitb (base), vitl (large)
    max_dim: 518            # resize input to this max dimension
"""

from __future__ import annotations

import logging
import numpy as np

from server.vision.base import CVModelBase, RawOutput

logger = logging.getLogger(__name__)

HF_MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingEstimator(CVModelBase):
    """Monocular depth estimation via Depth Anything V2.

    GitHub: https://github.com/LiheYoung/Depth-Anything-V2
    HF:     https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
    """

    output_type = "depth"
    model_id = "depth-anything"

    def __init__(self, config=None, model_variant: str = "small",
                 device: str = "cuda", max_dim: int = 518):
        super().__init__(config)
        self.model_variant = model_variant
        self.device = device
        self.max_dim = max_dim
        self._pipe = None

    def _initialize(self) -> None:
        import torch
        from transformers import pipeline  # HuggingFace transformers

        hf_id = HF_MODEL_IDS.get(self.model_variant, HF_MODEL_IDS["small"])
        logger.info("Loading Depth Anything V2 (%s) from %s", self.model_variant, hf_id)

        device_str = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self._pipe = pipeline(
            task="depth-estimation",
            model=hf_id,
            device=device_str,
        )
        logger.info("Depth Anything V2 ready (device=%s)", device_str)

    def process(self, frame_bgr: np.ndarray) -> RawOutput:
        import cv2
        from PIL import Image

        H, W = frame_bgr.shape[:2]
        raw = RawOutput(output_type="depth", model_id=self.model_id, frame_shape=(H, W))

        # Resize to max_dim for speed
        scale = self.max_dim / max(H, W)
        if scale < 1.0:
            new_h, new_w = int(H * scale), int(W * scale)
            resized = cv2.resize(frame_bgr, (new_w, new_h))
        else:
            resized = frame_bgr

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        result = self._pipe(pil_img)
        depth_pil = result["depth"]  # PIL Image in grayscale

        # Convert to float32 numpy and resize back to original frame size
        depth_np = np.array(depth_pil, dtype=np.float32)
        if depth_np.shape != (H, W):
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)

        raw.depth_map = depth_np
        return raw

    def shutdown(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        super().shutdown()
