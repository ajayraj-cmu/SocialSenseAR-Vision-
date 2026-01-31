"""Semantic labeling — position/size heuristic + label persistence.

Extracted from fastsam_segmenter.py (originally sam_gemini_voice.py line 2428).
Owns all labeling state: label_counts (per-frame), persistent_labels (cross-frame).
Every threshold and heuristic path is identical to the original.

SAFE TO EDIT: Changes here only affect label text/asset_class, not masks or protobuf format.
"""

import logging
import numpy as np
import cv2

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
        """Semantic labeling heuristic — conservative, short labels.

        Only confidently labels very large structural elements touching
        frame edges. Everything else returns "~object" and waits for Gemini.
        Uses cv2 ops instead of np.where for speed (~10x faster).

        Returns:
            (label, None) — second element kept for API compat.
        """
        # Accept both float32 [0,1] and uint8 [0,255] masks
        if mask.dtype == np.uint8:
            mask_u8 = mask  # Already uint8 (0 or 255) — cv2 ops work directly
        else:
            mask_u8 = (mask > 0.5).astype(np.uint8)
        area = cv2.countNonZero(mask_u8)

        if area == 0:
            return "~object", None

        area_ratio = area / (h * w)

        # Only do expensive edge checks for large segments
        if area_ratio > 0.20:
            # Check edges via thin border slices (fast)
            top_row = int(h * 0.03) or 1
            bot_row = int(h * 0.97)
            left_col = int(w * 0.03) or 1
            right_col = int(w * 0.97)

            touches_top = np.any(mask_u8[:top_row, :])
            touches_bottom = np.any(mask_u8[bot_row:, :])
            touches_left = np.any(mask_u8[:, :left_col])
            touches_right = np.any(mask_u8[:, right_col:])
            edge_count = touches_top + touches_bottom + touches_left + touches_right

            if edge_count >= 2:
                M = cv2.moments(mask_u8)
                cy_norm = (M["m01"] / M["m00"]) / h if M["m00"] > 0 else 0.5
                if cy_norm < 0.35:
                    return "ceiling", None
                if cy_norm > 0.65:
                    return "floor", None
                return "wall", None

        # Everything else — pending Gemini (~ prefix = unconfirmed)
        return "~object", None

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
            return f"{label}{self.label_counts[label]}"

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
