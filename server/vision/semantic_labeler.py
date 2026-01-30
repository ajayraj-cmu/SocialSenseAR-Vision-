"""Semantic labeling — position/size heuristic + label persistence.

Extracted from fastsam_segmenter.py (originally sam_gemini_voice.py line 2428).
Owns all labeling state: label_counts (per-frame), persistent_labels (cross-frame).
Every threshold and heuristic path is identical to the original.

SAFE TO EDIT: Changes here only affect label text/asset_class, not masks or protobuf format.
"""

import time
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SemanticLabeler:
    """Assigns semantic labels to masks based on position, size, and aspect ratio.

    State:
        label_counts: reset each frame via reset_frame(). Powers unique_label().
        persistent_labels: persists across frames (10s TTL). Stabilizes labels.
    """

    def __init__(self):
        self.label_counts: dict[str, int] = {}
        self.persistent_labels: dict[str, tuple] = {}
        self.label_lock_threshold = 0.3
        self.label_persistence_time = 10.0
        self.label_change_threshold = 0.5

    def reset_frame(self):
        """Reset per-frame label counts. Call at start of each _update_masks."""
        self.label_counts = {}

    # ------------------------------------------------------------------
    # Main label function — exact copy from sam_gemini_voice.py line 2428
    # ------------------------------------------------------------------

    def get_label(self, mask: np.ndarray, h: int, w: int, frame: np.ndarray = None) -> tuple[str, None]:
        """Semantic labeling heuristic.

        Args:
            mask: float32 mask.
            h, w: frame dimensions.
            frame: BGR image (for window brightness check).

        Returns:
            (label, None) — second element kept for API compat with original.
        """
        t0 = time.perf_counter()

        ys, xs = np.where(mask > 0.5)

        if len(ys) == 0:
            return "area", None

        cy = np.mean(ys)
        cx = np.mean(xs)
        area = len(ys)
        total_area = h * w
        area_ratio = area / total_area

        # Region key for label persistence
        region_key = self.get_region_key(cx, cy, area, h, w)

        # ============================================
        # Semantic labeling for structures — exact same as original
        # ============================================

        bbox_h = ys.max() - ys.min() if len(ys) > 0 else 0
        bbox_w = xs.max() - xs.min() if len(xs) > 0 else 0
        aspect = bbox_w / (bbox_h + 1) if bbox_h > 0 else 1

        touches_top = np.any(ys < h * 0.05)
        touches_bottom = np.any(ys > h * 0.95)
        touches_left = np.any(xs < w * 0.05)
        touches_right = np.any(xs > w * 0.05)

        label = None

        # Very large areas = structural
        if area_ratio > 0.12:
            if touches_top and cy < h * 0.4:
                label = "ceiling"
            elif touches_bottom and cy > h * 0.6:
                label = "floor"
            elif (touches_left or touches_right) and area_ratio > 0.08:
                label = "wall"
            elif cy < h * 0.35:
                label = "ceiling"
            elif cy > h * 0.65:
                label = "floor"
            else:
                label = "wall"

        # Doors
        if label is None and 0.05 < area_ratio < 0.15:
            if aspect < 0.7 and (touches_left or touches_right or touches_bottom):
                if 0.3 < cy / h < 0.8:
                    label = "door"

        # Windows
        if label is None and 0.03 < area_ratio < 0.12 and 0.25 < cy / h < 0.75:
            if 1.2 < aspect < 3.5:
                if frame is not None:
                    try:
                        mask_region = frame[int(ys.min()):int(ys.max()), int(xs.min()):int(xs.max())]
                        if len(mask_region) > 0:
                            gray = cv2.cvtColor(mask_region, cv2.COLOR_BGR2GRAY)
                            brightness = np.mean(gray)
                            if brightness > 80:
                                label = "window"
                    except Exception:
                        pass
                if label is None:
                    label = "window"

        # Medium areas — furniture
        if label is None and area_ratio > 0.03:
            bbox_h2 = ys.max() - ys.min()
            bbox_w2 = xs.max() - xs.min()
            aspect2 = bbox_w2 / (bbox_h2 + 1)

            if cy > h * 0.55:
                if aspect2 > 2:
                    label = "table"
                elif aspect2 < 0.5:
                    label = "cabinet"
                else:
                    label = "furniture"
            elif cy < h * 0.4:
                if aspect2 > 1.5:
                    label = "shelf"
                else:
                    label = "cabinet"
            else:
                if aspect2 > 2:
                    label = "monitor"
                else:
                    label = "furniture"

        # Small objects
        if label is None and area_ratio > 0.005:
            bbox_h3 = ys.max() - ys.min()
            bbox_w3 = xs.max() - xs.min()
            aspect3 = bbox_w3 / (bbox_h3 + 1)

            if aspect3 > 1.5:
                label = "item_wide"
            elif aspect3 < 0.7:
                label = "item_tall"
            else:
                label = "item"

        if label is None:
            label = "small_item"

        label_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Label: '{label}' for region {region_key} "
            f"(area={area_ratio:.1%}, aspect={aspect:.1f}, cy={cy/h:.2f}) "
            f"in {label_ms:.2f}ms"
        )

        return label, None

    # ------------------------------------------------------------------
    # Unique label — exact copy from sam_gemini_voice.py line 2379
    # ------------------------------------------------------------------

    def unique_label(self, label: str) -> str:
        """Make label unique within the current frame (wall, wall_2, wall_3)."""
        if label not in self.label_counts:
            self.label_counts[label] = 1
            return label
        else:
            self.label_counts[label] += 1
            return f"{label}_{self.label_counts[label]}"

    # ------------------------------------------------------------------
    # Region key — exact copy from sam_gemini_voice.py line 2392
    # ------------------------------------------------------------------

    @staticmethod
    def get_region_key(cx: float, cy: float, area: int, h: int, w: int) -> str:
        """32x32 grid + area bucket → cache key for label persistence."""
        grid_x = int(cx / w * 32)
        grid_y = int(cy / h * 32)
        area_bucket = 0 if area < w * h * 0.05 else (1 if area < w * h * 0.2 else 2)
        return f"{grid_x}_{grid_y}_{area_bucket}"

    # ------------------------------------------------------------------
    # Persistent label — exact copy from sam_gemini_voice.py line 2401
    # ------------------------------------------------------------------

    def get_persistent_label(self, region_key: str, new_label: str, new_confidence: float) -> str:
        """Lock labels for 10s with 50% confidence improvement threshold."""
        current_time = time.time()

        stale_keys = [k for k, v in self.persistent_labels.items()
                      if current_time - v[2] > self.label_persistence_time]
        for k in stale_keys:
            del self.persistent_labels[k]

        if region_key in self.persistent_labels:
            existing_label, existing_conf, last_seen = self.persistent_labels[region_key]
            confidence_improvement = (new_confidence - existing_conf) / (existing_conf + 0.01)

            if new_label != existing_label and confidence_improvement > self.label_change_threshold:
                self.persistent_labels[region_key] = (new_label, new_confidence, current_time)
                return new_label
            else:
                self.persistent_labels[region_key] = (existing_label, existing_conf, current_time)
                return existing_label
        else:
            self.persistent_labels[region_key] = (new_label, new_confidence, current_time)
            return new_label

    # ------------------------------------------------------------------
    # Asset class mapping (for protobuf output)
    # ------------------------------------------------------------------

    @staticmethod
    def label_to_asset_class(label: str) -> str:
        """Map label to asset_class for protobuf."""
        lbl = label.lower().split("_")[0] if label else ""
        person_labels = {"person", "face", "left", "right", "torso"}
        if lbl in person_labels:
            return "person"
        structural = {"wall", "floor", "ceiling", "door", "window"}
        if lbl in structural:
            return "structural"
        furniture_labels = {"table", "chair", "desk", "cabinet", "shelf",
                            "furniture", "sofa", "couch", "bed"}
        if lbl in furniture_labels:
            return "furniture"
        lighting_labels = {"light", "lamp", "chandelier", "bulb"}
        if lbl in lighting_labels:
            return "lighting"
        screen_labels = {"monitor", "screen", "laptop", "tv", "display"}
        if lbl in screen_labels:
            return "screen"
        return "object"
