"""Gemini Vision labeler with spatial caching.

Ported from scripts/sam_gemini_voice.py GeminiAgent.label_all_segments().
Key changes:
- Extracted from monolithic class into standalone module
- Rate limiting preserved (5 calls/min, 6s interval)
- Spatial label cache preserved (16×16 grid, 30s TTL)
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
    - 16×16 spatial cache with configurable TTL
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

    def label_segments(self, frame_bgr: np.ndarray, segments: list) -> list:
        """Assign labels and asset_classes to segments using cache + Gemini API.

        Args:
            frame_bgr: BGR frame for Gemini Vision.
            segments: List[SegmentData] with mask, center_x, center_y already set.

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
                api_labels = self._call_gemini_vision(frame_bgr, segments, unlabeled_idxs)
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
                fb = self._heuristic_label(seg, h, w, frame_bgr)
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

    def _call_gemini_vision(self, frame_bgr, segments, idxs) -> dict:
        """Call Gemini Vision to label specific segment indices.

        Returns: {segment_index: (label, asset_class)} or None.
        """
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((640, 480))

            h, w = frame_bgr.shape[:2]
            positions = []
            for region_num, seg_idx in enumerate(idxs[:20], start=1):
                seg = segments[seg_idx]
                cx_px = seg.center_x * w
                cy_px = seg.center_y * h
                horiz = "left-side" if cx_px < w * 0.33 else ("right-side" if cx_px > w * 0.67 else "center")
                vert = "upper" if cy_px < h * 0.33 else ("lower" if cy_px > h * 0.67 else "middle")
                positions.append(f"{region_num}: {vert} {horiz}")

            prompt = f"""Identify what object or surface is in EVERY numbered region of this image. Be accurate — label each region with what it actually is.

Regions:
{chr(10).join(positions)}

ASSET CLASSES (use exact asset_class string):
- "person": person, face, torso, head, hand, arm, leg, body
- "lighting": ceiling_light, table_lamp, floor_lamp, led_strip
- "screen": laptop_screen, monitor, tv_screen, tablet, phone_screen
- "furniture": desk, chair, table, shelf, cabinet, bed, couch
- "structural": wall, ceiling, floor, door, window_frame
- "object": plant, clock, picture_frame, book, keyboard, mouse, cup, bottle

Only use "person" for regions that clearly contain a human body part. Walls, furniture, floors, and other objects should NOT be labeled as person.

Return ONLY JSON array:
[{{"region":1,"label":"ceiling_light","asset_class":"lighting"}}]"""

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

            labels = {}
            for item in results:
                region_1based = item.get("region", 0)
                region_0based = region_1based - 1
                if 0 <= region_0based < len(idxs):
                    seg_idx = idxs[region_0based]
                    label = item.get("label", "unknown").lower().strip()
                    asset_class = item.get("asset_class", "object").lower().strip()
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
        """16×16 grid + size bucket → cache key."""
        if cx_norm <= 0 and cy_norm <= 0:
            return None
        gx = int(cx_norm * 16)
        gy = int(cy_norm * 16)
        area_frac = float(np.sum(mask > 0.5)) / (h * w) if mask is not None else 0
        bucket = 0 if area_frac < 0.02 else (1 if area_frac < 0.1 else 2)
        return f"{gx}_{gy}_{bucket}"

    # ------------------------------------------------------------------
    # Heuristic fallback (no API)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_label(seg, h, w, frame_bgr) -> tuple:
        """Position/brightness fallback labeling. Returns (label, asset_class).

        Conservative: only labels structural/furniture/lighting by position.
        Person detection left to Gemini — skin-tone heuristics have too many
        false positives (wood, warm walls, beige carpet, etc.).
        """
        cx = seg.center_x
        cy = seg.center_y
        area = float(np.sum(seg.mask > 0.5)) / (h * w) if seg.mask is not None else 0

        brightness = 128.0
        if seg.mask is not None and frame_bgr is not None:
            try:
                m = seg.mask > 0.5
                if np.any(m):
                    brightness = float(np.mean(frame_bgr[m]))
            except Exception:
                pass

        # Large segments are structural
        if area > 0.15:
            if cy < 0.35:
                return ("ceiling", "structural")
            if cy > 0.7:
                return ("floor", "structural")
            return ("wall", "structural")

        if brightness > 180 and cy < 0.5 and area < 0.05:
            return ("light", "lighting")
        if cy < 0.3 and area > 0.03:
            return ("ceiling", "structural")
        if cy > 0.5 and 0.02 < area < 0.15:
            return ("furniture", "furniture")
        if brightness > 150 and 0.01 < area < 0.08:
            return ("screen", "screen")
        if cx < 0.15 or cx > 0.85:
            return ("wall", "structural") if area > 0.05 else ("object", "object")
        if area > 0.05:
            return ("surface", "furniture")
        return ("object", "object")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        self._model = None
        self._available = False
        self._spatial_cache.clear()
        logger.info("GeminiLabeler shutdown")


# Body-part labels that should not be re-labelled by Gemini
_BODY_LABELS = frozenset({
    "person", "face", "head", "torso", "body",
    "left_hand", "right_hand", "hand",
    "left_arm", "right_arm", "arm",
    "left_leg", "right_leg", "leg",
    "human",
})
