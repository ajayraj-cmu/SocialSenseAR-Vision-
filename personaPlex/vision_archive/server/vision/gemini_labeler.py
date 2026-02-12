"""Gemini Vision labeler with spatial caching.

Ported from scripts/sam_gemini_voice.py GeminiAgent.label_all_segments().
Key changes:
- Extracted from monolithic class into standalone module
- Rate limiting preserved (5 calls/min, 6s interval)
- Spatial label cache preserved (8×8 grid, 30s TTL)
- Returns structured dicts instead of printing to console
"""

import os
import time
import json
import logging
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class GeminiLabeler:
    """Labels FastSAM segments using Gemini Vision API.

    Features:
    - Rate-limited vision API calls (free-tier safe)
    - 8×8 spatial cache with configurable TTL
    - Fallback heuristic labels when API unavailable
    - Asset-class taxonomy (lighting, screen, furniture, person, …)
    """

    def __init__(self, config):
        """
        Args:
            config: ServerConfig with gemini_model, gemini_api_key,
                    gemini_min_interval, gemini_max_calls_per_minute,
                    gemini_label_cache_ttl.
        """
        self.config = config
        self._model = None
        self._available = False

        # Rate limiting
        self._call_count = 0
        self._last_call_time = 0.0
        self._calls_this_minute = 0
        self._minute_start = time.time()

        # Spatial label cache: "gx_gy_size" → (label, asset_class, confidence, timestamp)
        self._spatial_cache: dict[str, tuple] = {}
        self._last_label_time = 0.0

        # Label corrections from feedback (current_label → corrected_label)
        self._corrections: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Configure Gemini and verify API key."""
        t0 = time.perf_counter()

        api_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key found — labeler disabled")
            return

        try:
            t_imp = time.perf_counter()
            import google.generativeai as genai
            logger.info(f"    import google.generativeai: {(time.perf_counter() - t_imp)*1000:.0f}ms")

            t_cfg = time.perf_counter()
            genai.configure(api_key=api_key)
            logger.info(f"    genai.configure: {(time.perf_counter() - t_cfg)*1000:.0f}ms")

            # Try configured model first, then fallbacks
            models_to_try = [
                self.config.gemini_model,
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-pro",
            ]
            # Deduplicate while preserving order
            seen = set()
            models_to_try = [m for m in models_to_try if m not in seen and not seen.add(m)]

            for model_name in models_to_try:
                try:
                    t_model = time.perf_counter()
                    self._model = genai.GenerativeModel(model_name)
                    self._available = True
                    logger.info(f"    Gemini model ({model_name}): {(time.perf_counter() - t_model)*1000:.0f}ms")
                    break
                except Exception:
                    logger.debug(f"Gemini model {model_name} not available, trying next")
            else:
                logger.error("No Gemini model available")

            logger.info(f"    Gemini total: {(time.perf_counter() - t0)*1000:.0f}ms")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")

    @property
    def available(self):
        return self._available

    # ------------------------------------------------------------------
    # Main entry — label a batch of segments
    # ------------------------------------------------------------------

    def label_segments(self, frame_bgr: np.ndarray, segments: list, person_mask=None) -> list:
        """Assign labels and asset_classes to segments using cache + Gemini API.

        Args:
            frame_bgr: BGR frame for Gemini Vision.
            segments: List[SegmentData] with mask, center_x, center_y already set.
            person_mask: Optional float32 mask from MediaPipe selfie segmenter.

        Returns:
            The same list, with .label and .asset_class populated.
        """
        h, w = frame_bgr.shape[:2]
        now = time.time()

        # 1. Apply cached / body-part labels first
        unlabeled_idxs = []
        for i, seg in enumerate(segments):
            # Keep body-part labels from MediaPipe
            if seg.label in _BODY_LABELS:
                seg.asset_class = "person"
                continue

            key = self._spatial_key(seg.center_x, seg.center_y, seg.mask, h, w)
            cached = self._spatial_cache.get(key) if key else None
            if cached and now - cached[3] < self.config.gemini_label_cache_ttl:
                seg.label = cached[0]
                seg.asset_class = cached[1]
                seg.confidence = cached[2]
            else:
                unlabeled_idxs.append(i)

        # 2. If there are unlabeled segments and rate limit allows, call Gemini
        if unlabeled_idxs and self._can_call():
            if now - self._last_label_time >= 3.0:  # min 3s between batch calls
                api_labels = self._call_gemini_vision(
                    frame_bgr, segments, unlabeled_idxs, person_mask=person_mask
                )
                if api_labels:
                    for idx, (label, asset_class) in api_labels.items():
                        seg = segments[idx]
                        seg.label = label
                        seg.asset_class = asset_class
                        seg.confidence = 0.85  # API-labelled
                        key = self._spatial_key(seg.center_x, seg.center_y, seg.mask, h, w)
                        if key:
                            self._spatial_cache[key] = (label, asset_class, 0.85, now)
                self._last_label_time = now

        # 3. Fallback for anything still unlabeled
        for i in unlabeled_idxs:
            seg = segments[i]
            if not seg.label or seg.label.startswith("~"):
                fb = self._heuristic_label(seg, h, w, person_mask)
                seg.label = fb[0]
                seg.asset_class = fb[1]

        # 4. Apply corrections
        for seg in segments:
            corrected = self._corrections.get(seg.label)
            if corrected:
                seg.label = corrected

        # 5. Expire old cache entries
        stale = [k for k, v in self._spatial_cache.items() if now - v[3] > self.config.gemini_label_cache_ttl]
        for k in stale:
            del self._spatial_cache[k]

        return segments

    def apply_corrections(self, corrections: dict[str, str]):
        """Merge label corrections (current → corrected) from feedback."""
        self._corrections.update(corrections)

    # ------------------------------------------------------------------
    # Gemini API call
    # ------------------------------------------------------------------

    # Colors for annotating segments on the image sent to Gemini
    _MARKER_COLORS = [
        (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
        (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
        (255, 255, 255), (100, 255, 200),
        (255, 100, 100), (100, 100, 255), (200, 255, 100), (255, 200, 100),
        (100, 255, 255), (255, 100, 200), (200, 100, 255), (100, 200, 100),
        (200, 200, 100), (100, 200, 200),
    ]

    def _call_gemini_vision(self, frame_bgr, segments, idxs, person_mask=None) -> dict:
        """Call Gemini Vision to label specific segment indices.

        Annotates the image with numbered markers and contour outlines
        so Gemini can see exactly which region each number refers to.
        Includes person-region hints so Gemini doesn't mislabel body parts.

        Returns: {segment_index: (label, asset_class)} or None.
        """
        from server.vision.semantic_labeler import SemanticLabeler

        try:
            h, w = frame_bgr.shape[:2]
            active_idxs = idxs[:20]

            # Detect which regions overlap the person mask
            person_regions = []
            for region_num, seg_idx in enumerate(active_idxs, start=1):
                seg = segments[seg_idx]
                if person_mask is not None and seg.mask is not None:
                    overlap = np.sum((seg.mask > 0.5) & (person_mask > 0.5))
                    seg_area = np.sum(seg.mask > 0.5)
                    if seg_area > 0 and overlap / seg_area > 0.3:
                        person_regions.append(str(region_num))

            # Draw contours and numbered markers on a copy of the frame
            annotated = frame_bgr.copy()
            for region_num, seg_idx in enumerate(active_idxs, start=1):
                seg = segments[seg_idx]
                color = self._MARKER_COLORS[(region_num - 1) % len(self._MARKER_COLORS)]

                # Draw contour outline so Gemini sees the segment boundary
                if seg.mask is not None:
                    mask_u8 = (seg.mask > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated, contours, -1, color, 2)

                # Draw numbered marker at segment center
                cx_px = int(seg.center_x * w)
                cy_px = int(seg.center_y * h)
                num_str = str(region_num)
                (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                radius = max(tw, th) // 2 + 6
                cv2.circle(annotated, (cx_px, cy_px), radius, (0, 0, 0), -1)
                cv2.circle(annotated, (cx_px, cy_px), radius, color, 2)
                cv2.putText(annotated, num_str,
                            (cx_px - tw // 2, cy_px + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((640, 480))

            # Build person hint
            person_hint = ""
            if person_regions:
                person_hint = f"\nRegion(s) {', '.join(person_regions)} overlap a detected person. Use body labels for these (face, hair, shirt, arm, hand, skin, torso, shoulder, etc). Do NOT label them as wall, floor, furniture, or window."

            prompt = f"""Image has {len(active_idxs)} numbered regions with colored outlines. Label what is INSIDE each outlined region. Use SHORT labels (1-2 words, no underscores).
{person_hint}
Labels: face, hair, head, hand, arm, shirt, torso, shoulder, skin, pants, body, wall, floor, ceiling, door, window, curtain, desk, chair, table, shelf, couch, lamp, light, monitor, laptop, tv, phone, plant, cup, book, keyboard, mouse, bag, clock, picture, speaker, bottle

Return ONLY JSON: [{{"r":1,"l":"desk"}}]"""

            self._record_call()
            response = self._model.generate_content([prompt, pil_image])
            text = response.text.strip()

            # Parse JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            if "[" in text:
                text = text[text.find("["):text.rfind("]") + 1]

            results = json.loads(text)

            # Structural labels that should never apply to person-overlapping segments
            _STRUCTURAL = {"wall", "floor", "ceiling", "door", "window", "furniture"}

            labels = {}
            for item in results:
                # Support both full keys and compact keys
                region_1based = item.get("r") or item.get("region", 0)
                region_0based = int(region_1based) - 1
                if 0 <= region_0based < len(active_idxs):
                    seg_idx = active_idxs[region_0based]
                    label = (item.get("l") or item.get("label", "unknown")).lower().strip()
                    # Truncate long labels to first 2 words
                    parts = label.replace("_", " ").split()
                    if len(parts) > 2:
                        label = " ".join(parts[:2])

                    # Post-process: reject structural labels on person regions
                    if str(region_0based + 1) in person_regions and label in _STRUCTURAL:
                        label = "body"

                    # Derive asset_class locally instead of trusting Gemini
                    asset_class = SemanticLabeler.label_to_asset_class(label)
                    labels[seg_idx] = (label, asset_class)

            logger.info(f"Gemini labelled {len(labels)} segments")
            return labels

        except json.JSONDecodeError:
            logger.warning("Gemini JSON parse error — using fallback")
            return None
        except Exception as e:
            logger.warning(f"Gemini labelling error: {str(e)[:80]}")
            return None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _can_call(self) -> bool:
        now = time.time()
        if now - self._minute_start > 60:
            self._calls_this_minute = 0
            self._minute_start = now
        if self._calls_this_minute >= self.config.gemini_max_calls_per_minute:
            return False
        if now - self._last_call_time < self.config.gemini_min_interval:
            return False
        return self._available

    def _record_call(self):
        self._call_count += 1
        self._calls_this_minute += 1
        self._last_call_time = time.time()

    # ------------------------------------------------------------------
    # Spatial cache key
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_key(cx_norm, cy_norm, mask, h, w):
        """8x8 grid + size bucket → cache key. Coarser grid tolerates centroid jitter."""
        if cx_norm <= 0 and cy_norm <= 0:
            return None
        gx = int(cx_norm * 8)
        gy = int(cy_norm * 8)
        area_frac = float(np.sum(mask > 0.5)) / (h * w) if mask is not None else 0
        bucket = 0 if area_frac < 0.02 else (1 if area_frac < 0.1 else 2)
        return f"{gx}_{gy}_{bucket}"

    # ------------------------------------------------------------------
    # Heuristic fallback (no API)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_label(seg, h, w, person_mask=None) -> tuple:
        """Conservative fallback labeling. Returns (label, asset_class).

        Person-mask-aware: segments overlapping a person get ~body.
        Only labels clearly structural elements (large + edge-touching).
        Everything else gets ~object (pending Gemini confirmation).
        """
        # Check person overlap first
        if person_mask is not None and seg.mask is not None:
            overlap = np.sum((seg.mask > 0.5) & (person_mask > 0.5))
            seg_area = np.sum(seg.mask > 0.5)
            if seg_area > 0 and overlap / seg_area > 0.3:
                return ("~body", "person")

        area = float(np.sum(seg.mask > 0.5)) / (h * w) if seg.mask is not None else 0

        # Only label structural for very large segments
        if area > 0.20 and seg.mask is not None:
            ys, xs = np.where(seg.mask > 0.5)
            if len(ys) > 0:
                touches_top = np.any(ys < h * 0.03)
                touches_bottom = np.any(ys > h * 0.97)
                touches_left = np.any(xs < w * 0.03)
                touches_right = np.any(xs > w * 0.97)
                edge_count = sum([touches_top, touches_bottom, touches_left, touches_right])
                if edge_count >= 2:
                    cy = seg.center_y
                    if cy < 0.35:
                        return ("ceiling", "structural")
                    if cy > 0.65:
                        return ("floor", "structural")
                    return ("wall", "structural")

        # Everything else: pending Gemini
        return ("~object", "object")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        self._model = None
        self._available = False
        self._spatial_cache.clear()
        logger.info("GeminiLabeler shutdown")


# Body-part labels that should not be re-labelled by Gemini
# NOTE: "person" is NOT included — we don't generate person silhouette segments.
# If Gemini says "person", it should be relabeled to something more specific.
_BODY_LABELS = frozenset({
    "face", "head", "torso", "body",
    "left_hand", "right_hand", "hand",
    "left_arm", "right_arm", "arm",
    "left_leg", "right_leg", "leg",
})
