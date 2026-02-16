"""Sapiens estimator — Meta's human-centric foundation model.

GitHub: https://github.com/facebookresearch/sapiens
HuggingFace: https://huggingface.co/facebook/sapiens-pose-1b

Supports:
  - Pose estimation (133 COCO-WholeBody keypoints)
  - Body part segmentation
  - Depth estimation (human-centric)

Install:
  Option A (HuggingFace, recommended):
    pip install transformers torch torchvision
    # Models auto-downloaded from HuggingFace

  Option B (clone repo):
    git clone https://github.com/facebookresearch/sapiens
    # Follow sapiens/README.md for env setup

License: Apache 2.0 (check https://github.com/facebookresearch/sapiens/blob/main/LICENSE)

Config keys (in cv_models.yaml):
  sapiens:
    task: pose          # pose, seg, depth
    model_size: 0.3b    # 0.3b, 0.6b, 1b, 2b
    device: cuda
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

from server.vision.base import CVModelBase, RawOutput

logger = logging.getLogger(__name__)

# HuggingFace model IDs per task and size
HF_MODELS = {
    "pose": {
        "0.3b": "facebook/sapiens-pose-0.3b",
        "0.6b": "facebook/sapiens-pose-0.6b",
        "1b":   "facebook/sapiens-pose-1b",
    },
    "seg": {
        "0.3b": "facebook/sapiens-seg-0.3b",
        "0.6b": "facebook/sapiens-seg-0.6b",
        "1b":   "facebook/sapiens-seg-1b",
    },
    "depth": {
        "0.3b": "facebook/sapiens-depth-0.3b",
        "0.6b": "facebook/sapiens-depth-0.6b",
        "1b":   "facebook/sapiens-depth-1b",
    },
}

# COCO WholeBody skeleton edges (133 keypoints — pose task)
# Simplified subset: body (0-16), face (17-22), hands (91-132)
SAPIENS_BODY_EDGES = [
    0, 1, 0, 2, 1, 3, 2, 4,
    5, 6, 5, 7, 7, 9, 6, 8, 8, 10,
    5, 11, 6, 12, 11, 12,
    11, 13, 13, 15, 12, 14, 14, 16,
]


class SapiensEstimator(CVModelBase):
    """Meta Sapiens — human-centric pose, segmentation, depth.

    GitHub: https://github.com/facebookresearch/sapiens
    HF:     https://huggingface.co/facebook/sapiens-pose-0.3b
    """

    output_type = "keypoints"
    model_id = "sapiens"

    def __init__(self, config=None, task: str = "pose",
                 model_size: str = "0.3b", device: str = "cuda"):
        super().__init__(config)
        self.task = task  # "pose", "seg", "depth"
        self.model_size = model_size
        self.device = device
        self._model = None
        self._processor = None

        # Update output_type based on task
        if task == "depth":
            self.output_type = "depth"
        elif task == "seg":
            self.output_type = "segmentation"
        else:
            self.output_type = "keypoints"

    def _initialize(self) -> None:
        import torch

        hf_id = HF_MODELS.get(self.task, {}).get(self.model_size)
        if hf_id is None:
            raise ValueError(
                f"No Sapiens model for task={self.task!r} size={self.model_size!r}. "
                f"Available: {list(HF_MODELS.get(self.task, {}).keys())}"
            )

        logger.info("Loading Sapiens %s (%s) from %s", self.task, self.model_size, hf_id)

        try:
            # Attempt HuggingFace AutoModel loading
            from transformers import AutoProcessor, AutoModel
            device_str = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
            self._processor = AutoProcessor.from_pretrained(hf_id)
            self._model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float16 if device_str == "cuda" else torch.float32)
            self._model = self._model.to(device_str)
            self._model.eval()
            self._device_str = device_str
            logger.info("Sapiens %s ready (device=%s)", self.task, device_str)
        except Exception as exc:
            logger.warning(
                "Sapiens HuggingFace load failed (%s). "
                "Fall back to cloning https://github.com/facebookresearch/sapiens and following README.",
                exc,
            )
            raise

    def process(self, frame_bgr: np.ndarray) -> RawOutput:
        import cv2
        import torch
        from PIL import Image

        H, W = frame_bgr.shape[:2]
        raw = RawOutput(output_type=self.output_type, model_id=self.model_id, frame_shape=(H, W))

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device_str) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        if self.task == "pose":
            raw.keypoint_groups.extend(self._parse_pose(outputs, H, W))
        elif self.task == "depth":
            raw.depth_map = self._parse_depth(outputs, H, W)
        elif self.task == "seg":
            segs = self._parse_seg(outputs, H, W)
            raw.masks.extend([s.mask for s in segs])
            raw.labels.extend([s.label for s in segs])
            raw.scores.extend([s.confidence for s in segs])

        return raw

    def _parse_pose(self, outputs, H: int, W: int) -> list[dict]:
        """Parse pose heatmap outputs into keypoint groups."""
        # Sapiens outputs heatmaps; argmax gives (x, y) per keypoint
        # Shape typically: (1, num_kp, hm_h, hm_w)
        import torch
        heatmaps = outputs.get("heatmaps", None)
        if heatmaps is None:
            # Try logits or last_hidden_state fallback
            logger.debug("Sapiens pose: no 'heatmaps' key in output")
            return []

        hm = heatmaps[0].cpu().float()  # (num_kp, hm_h, hm_w)
        num_kp, hm_h, hm_w = hm.shape

        points = []
        for k in range(num_kp):
            flat_idx = hm[k].argmax().item()
            ky = flat_idx // hm_w
            kx = flat_idx % hm_w
            vis = float(hm[k].max().item())
            points.append({
                "x": float(kx) / hm_w,
                "y": float(ky) / hm_h,
                "visibility": min(vis, 1.0),
                "name": f"kp_{k}",
            })

        return [{
            "groups": [{
                "label": "body",
                "points": points[:17],   # first 17 = body keypoints
                "skeleton_edges": SAPIENS_BODY_EDGES,
            }]
        }]

    def _parse_depth(self, outputs, H: int, W: int) -> Optional[np.ndarray]:
        import cv2, torch
        depth = outputs.get("predicted_depth", None)
        if depth is None:
            return None
        depth_np = depth[0].cpu().float().numpy()
        if depth_np.shape != (H, W):
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)
        return depth_np

    def _parse_seg(self, outputs, H: int, W: int) -> list:
        """Parse segmentation logits into per-class masks."""
        import cv2, torch
        from server.vision.segment_data import SegmentData
        logits = outputs.get("logits", None)
        if logits is None:
            return []

        seg_map = logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        if seg_map.shape != (H, W):
            seg_map = cv2.resize(seg_map, (W, H), interpolation=cv2.INTER_NEAREST)

        segments = []
        for cls_id in np.unique(seg_map):
            if cls_id == 0:
                continue  # background
            mask = (seg_map == cls_id).astype(np.uint8)
            seg = SegmentData(mask=mask, label=f"body_part_{cls_id}", confidence=1.0)
            segments.append(seg)
        return segments

    def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        super().shutdown()
