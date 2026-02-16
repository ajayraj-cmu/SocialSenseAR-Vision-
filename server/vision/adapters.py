"""Output adapters: convert RawOutput from any CV model into SegmentData or overlay payloads.

Adapter map:
  segmentation  → SegmentationAdapter  → list[SegmentData]  (pass-through)
  detection     → DetectionAdapter     → list[SegmentData]  (bbox → rect mask)
  keypoints     → KeypointsAdapter     → (list[SegmentData], list[dict])
  depth         → DepthAdapter         → (list[SegmentData], bytes JPEG colormap)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from server.vision.base import RawOutput
from server.vision.segment_data import SegmentData


# ---------------------------------------------------------------------------
# Segmentation adapter (pass-through)
# ---------------------------------------------------------------------------

class SegmentationAdapter:
    """Converts mask-based RawOutput to SegmentData list. No transformation."""

    def convert(self, raw: RawOutput, frame_shape: tuple[int, int]) -> list[SegmentData]:
        H, W = frame_shape
        segments = []
        for i, mask in enumerate(raw.masks):
            label = raw.labels[i] if i < len(raw.labels) else ""
            score = raw.scores[i] if i < len(raw.scores) else 0.0
            seg = SegmentData(mask=mask.astype(np.uint8), label=label, confidence=score)
            _fill_bbox_center(seg, mask, W, H)
            segments.append(seg)
        return segments


# ---------------------------------------------------------------------------
# Detection adapter (bbox → rectangular mask)
# ---------------------------------------------------------------------------

class DetectionAdapter:
    """Converts bounding boxes into rectangular binary masks → SegmentData."""

    def convert(self, raw: RawOutput, frame_shape: tuple[int, int]) -> list[SegmentData]:
        H, W = frame_shape
        segments = []
        for i, box in enumerate(raw.boxes):
            label = raw.labels[i] if i < len(raw.labels) else ""
            score = raw.scores[i] if i < len(raw.scores) else 0.0

            x1, y1, x2, y2 = box
            px1 = int(x1 * W)
            py1 = int(y1 * H)
            px2 = int(x2 * W)
            py2 = int(y2 * H)

            mask = np.zeros((H, W), dtype=np.uint8)
            mask[py1:py2, px1:px2] = 1

            # Also use instance mask if provided
            if i < len(raw.masks) and raw.masks[i] is not None:
                mask = raw.masks[i].astype(np.uint8)

            seg = SegmentData(mask=mask, label=label, confidence=score)
            _fill_bbox_center(seg, mask, W, H)
            segments.append(seg)
        return segments


# ---------------------------------------------------------------------------
# Keypoints adapter
# ---------------------------------------------------------------------------

class KeypointsAdapter:
    """Converts keypoint groups to protobuf-ready dicts + optional hull masks."""

    def convert(
        self,
        raw: RawOutput,
        frame_shape: tuple[int, int],
        make_hull_masks: bool = False,
    ) -> tuple[list[SegmentData], list[dict]]:
        """Returns (segments, keypoint_groups_for_proto).

        segments may be empty unless make_hull_masks=True.
        keypoint_groups are already in the canonical proto format:
          [{model_id, groups: [{label, points:[{x,y,visibility,name}], skeleton_edges:[int]}]}]
        """
        H, W = frame_shape
        segments = []
        kp_payload = []

        for group_set in raw.keypoint_groups:
            proto_group = {
                "model_id": raw.model_id,
                "groups": [],
            }
            for grp in group_set.get("groups", []):
                proto_group["groups"].append({
                    "label": grp.get("label", ""),
                    "points": grp.get("points", []),
                    "skeleton_edges": grp.get("skeleton_edges", []),
                })

                if make_hull_masks:
                    pts = grp.get("points", [])
                    vis_pts = [
                        (int(p["x"] * W), int(p["y"] * H))
                        for p in pts if p.get("visibility", 1.0) > 0.3
                    ]
                    if len(vis_pts) >= 3:
                        hull = cv2.convexHull(np.array(vis_pts))
                        mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, hull, 1)
                        label = grp.get("label", "keypoints")
                        seg = SegmentData(mask=mask, label=label, confidence=1.0)
                        _fill_bbox_center(seg, mask, W, H)
                        segments.append(seg)

            kp_payload.append(proto_group)

        return segments, kp_payload


# ---------------------------------------------------------------------------
# Depth adapter
# ---------------------------------------------------------------------------

class DepthAdapter:
    """Converts depth map to JPEG-encoded colormap for the DepthOverlay proto message."""

    COLORMAPS = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }

    def convert(
        self,
        raw: RawOutput,
        frame_shape: tuple[int, int],
        colormap: str = "turbo",
        jpeg_quality: int = 75,
        max_dim: int = 512,
    ) -> Optional[dict]:
        """Returns a depth overlay dict ready for the proto, or None on failure."""
        if raw.depth_map is None:
            return None

        depth = raw.depth_map.astype(np.float32)

        # Normalize to 0-255
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            return None
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # Apply colormap
        cmap = self.COLORMAPS.get(colormap, cv2.COLORMAP_TURBO)
        depth_color = cv2.applyColorMap(depth_norm, cmap)

        # Downscale if needed
        h, w = depth_color.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            depth_color = cv2.resize(depth_color, (int(w * scale), int(h * scale)))

        # JPEG encode
        ok, buf = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return None

        out_h, out_w = depth_color.shape[:2]
        return {
            "model_id": raw.model_id,
            "depth_rgb": buf.tobytes(),
            "width": out_w,
            "height": out_h,
            "near_m": float(d_min),
            "far_m": float(d_max),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_bbox_center(seg: SegmentData, mask: np.ndarray, W: int, H: int) -> None:
    """Fill seg.bbox and seg.center_x/y from mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        seg.bbox = (0.0, 0.0, 0.0, 0.0)
        seg.center_x = 0.5
        seg.center_y = 0.5
        return
    x1, x2 = xs.min() / W, xs.max() / W
    y1, y2 = ys.min() / H, ys.max() / H
    seg.bbox = (x1, y1, x2, y2)
    seg.center_x = (x1 + x2) / 2
    seg.center_y = (y1 + y2) / 2
    seg.mask_width = W
    seg.mask_height = H
