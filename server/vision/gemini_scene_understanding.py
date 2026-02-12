"""Gemini Scene Understanding — on-demand vision reasoning for voice agent.

Provides:
- snapshot(frame) → list of visible objects + attributes
- identify_targets(utterance, visible_objects) → which objects to target
- cached results to avoid redundant API calls during request processing
"""

import time
import logging
import cv2
import numpy as np
from typing import Optional
import base64

logger = logging.getLogger(__name__)


class GeminiSceneUnderstanding:
    """On-demand Gemini Vision reasoning for voice agent.

    Used when user makes requests that need context about what's visible:
    - "blur the background"
    - "dim everything, it's too bright"
    - "highlight the laptop"

    Caches results for a short TTL to avoid redundant calls during a single request.
    """

    def __init__(self, api_key: str = None, model: str = None, cache_ttl: float = 10.0):
        self._api_key = api_key
        self._model = model
        self._cache_ttl = cache_ttl
        self._client = None
        self._available = False

        # Cache: frame snapshot results
        self._last_snapshot_time = 0.0
        self._last_snapshot_result: Optional[dict] = None

        # Rate limiting for vision calls (prevent 429 errors)
        self._last_vision_call_time = 0.0
        self._vision_call_times = []  # Sliding window for RPM tracking
        self._min_call_interval = 6.0  # Will be set from config
        self._max_calls_per_minute = 5  # Will be set from config

    def initialize(self):
        """Initialize Gemini API client."""
        import os
        api_key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key — scene understanding disabled")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._available = True
            logger.info(f"Gemini scene understanding ready (model={self._model})")
        except Exception as e:
            logger.error(f"Gemini scene understanding init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def snapshot(self, frame_bgr: np.ndarray, force_fresh: bool = False) -> dict:
        """Analyze current frame and return visible objects + scene description.

        Returns:
            {
                "objects": ["laptop", "monitor", "desk", "chair", "lamp", ...],
                "scene_description": "Office environment with desk, laptop, ...",
                "bright_objects": ["lamp", "window"],  # objects contributing to brightness
                "background_objects": ["wall", "ceiling", "floor"],
            }
        """
        if not self._available:
            return {"objects": [], "scene_description": "", "bright_objects": [], "background_objects": []}

        # Return cached result if recent
        now = time.time()
        if not force_fresh and self._last_snapshot_result and (now - self._last_snapshot_time) < self._cache_ttl:
            logger.debug(f"Using cached snapshot (age={now - self._last_snapshot_time:.1f}s)")
            return self._last_snapshot_result

        try:
            # Encode frame as JPEG for Gemini
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            # Resize if too large (Gemini has size limits)
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Create PIL Image for Gemini
            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(buffer.tobytes()))

            prompt = """Analyze this camera view and provide:
1. A list of all visible objects (be comprehensive - include furniture, electronics, structural elements, lights, etc.)
2. A brief scene description
3. Objects that might be considered "bright" or contributing to overstimulation (lights, windows, screens)
4. Objects that are part of the background/environment (walls, ceiling, floor, distant furniture)

Format your response as JSON:
{
    "objects": ["laptop", "desk", "chair", ...],
    "scene_description": "Brief description",
    "bright_objects": ["lamp", "window", ...],
    "background_objects": ["wall", "ceiling", ...]
}

Be thorough with object detection - include everything you can identify."""

            response = self._client.models.generate_content(
                model=self._model, contents=[prompt, pil_img]
            )
            text = response.text.strip()

            # Parse JSON response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            import json
            result = json.loads(text)

            # Cache the result
            self._last_snapshot_time = now
            self._last_snapshot_result = result

            logger.info(f"Snapshot: {len(result.get('objects', []))} objects detected")
            logger.debug(f"Objects: {result.get('objects', [])}")

            return result

        except Exception as e:
            logger.error(f"Gemini snapshot error: {e}", exc_info=True)
            return {"objects": [], "scene_description": "", "bright_objects": [], "background_objects": []}

    def identify_targets(
        self,
        utterance: str,
        visible_objects: list[str],
        scene_context: str = "",
    ) -> dict:
        """Use Gemini reasoning to identify which objects should be targeted for effects.

        Args:
            utterance: User's natural language command
            visible_objects: List of objects currently visible in scene
            scene_context: Optional scene description for context

        Returns:
            {
                "targets": ["laptop", "monitor"],  # objects to apply effect to
                "effect_type": "blur",
                "intensity": 0.8,
                "reasoning": "User wants to blur screens for focus"
            }
        """
        if not self._available:
            return {"targets": [], "effect_type": "none", "intensity": 1.0, "reasoning": "Gemini unavailable"}

        try:
            prompt = f"""Given this user request: "{utterance}"

Scene context: {scene_context or 'Unknown scene'}
Visible objects: {', '.join(visible_objects) if visible_objects else 'none'}

Determine:
1. Which specific objects should be targeted for the effect (from the visible objects list)
2. What type of effect should be applied (blur, dim, pixelate, highlight, outline, none)
3. What intensity (0.0-1.0)
4. Brief reasoning

If the request is general (e.g., "dim everything", "it's too bright"), infer which objects are appropriate targets based on the request context.
If the request mentions "background", target structural/distant objects.
If no appropriate targets exist, return empty targets list.

Format as JSON:
{{
    "targets": ["object1", "object2"],
    "effect_type": "blur",
    "intensity": 0.8,
    "reasoning": "explanation"
}}"""

            response = self._client.models.generate_content(
                model=self._model, contents=prompt
            )
            text = response.text.strip()

            # Parse JSON (handle markdown)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            import json
            result = json.loads(text)

            logger.info(f"Target identification: {result.get('targets', [])} ({result.get('effect_type', 'none')})")
            logger.debug(f"Reasoning: {result.get('reasoning', '')}")

            return result

        except Exception as e:
            logger.error(f"Gemini target identification error: {e}", exc_info=True)
            return {"targets": [], "effect_type": "none", "intensity": 1.0, "reasoning": f"Error: {e}"}

    def verify_mask_effect(
        self,
        frame_bgr: np.ndarray,
        applied_effect: str,
        target_object: str,
    ) -> dict:
        """Async-safe Gemini Vision check: does the mask/effect look correct?

        Used as an optional background check after applying effects.
        Does not block the main pipeline — call from a background thread.

        Args:
            frame_bgr: Current frame with overlay applied (composite or screenshot)
            applied_effect: Effect type that was applied (e.g., "blur", "color")
            target_object: Object the effect was applied to (e.g., "laptop")

        Returns:
            {
                "looks_correct": True/False,
                "confidence": 0.0-1.0,
                "suggestion": "optional adjustment suggestion" or "",
                "reasoning": "brief explanation"
            }
        """
        if not self._available:
            return {"looks_correct": True, "confidence": 0.0, "suggestion": "", "reasoning": "Gemini unavailable"}

        try:
            # Encode frame as JPEG for Gemini
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            if max(h, w) > 768:
                scale = 768 / max(h, w)
                frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])

            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(buffer.tobytes()))

            prompt = f"""Look at this AR view. A "{applied_effect}" effect was applied to "{target_object}".

Is the effect covering the correct object? Is the visual result appropriate?

Respond ONLY with JSON:
{{
    "looks_correct": true/false,
    "confidence": 0.0-1.0,
    "suggestion": "brief adjustment if needed, or empty string",
    "reasoning": "brief explanation"
}}"""

            response = self._client.models.generate_content(
                model=self._model, contents=[prompt, pil_img]
            )
            text = response.text.strip()

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            import json
            result = json.loads(text)

            logger.info(
                f"Mask verification: {applied_effect} on {target_object} → "
                f"{'OK' if result.get('looks_correct') else 'MISMATCH'} "
                f"(confidence={result.get('confidence', 0):.2f})"
            )
            return result

        except Exception as e:
            logger.warning(f"Gemini mask verification error: {e}")
            return {"looks_correct": True, "confidence": 0.0, "suggestion": "", "reasoning": f"Error: {e}"}

    def set_rate_limits(self, min_interval: float, max_calls_per_minute: int):
        """Configure rate limiting parameters from config."""
        self._min_call_interval = min_interval
        self._max_calls_per_minute = max_calls_per_minute

    def _check_rate_limit(self) -> bool:
        """Check if we can make a vision call without hitting rate limits.

        Returns True if call is allowed, False if we should wait.
        Enforces both min_interval (seconds between calls) and RPM limit.
        """
        now = time.time()

        # Check min interval since last call
        if now - self._last_vision_call_time < self._min_call_interval:
            return False

        # Clean old call times from sliding window (last 60 seconds)
        cutoff = now - 60.0
        self._vision_call_times = [t for t in self._vision_call_times if t > cutoff]

        # Check RPM limit
        if len(self._vision_call_times) >= self._max_calls_per_minute:
            return False

        return True

    def _wait_for_rate_limit(self):
        """Wait until rate limit allows a new call."""
        while not self._check_rate_limit():
            time.sleep(0.5)

    def _record_vision_call(self):
        """Record a vision call for rate limiting."""
        now = time.time()
        self._last_vision_call_time = now
        self._vision_call_times.append(now)

    def _call_gemini_vision_with_retry(self, prompt: str, image, max_retries: int = 3):
        """Call Gemini Vision with rate limiting and 429 retry logic.

        Args:
            prompt: Text prompt for Gemini
            image: PIL Image or None for text-only
            max_retries: Max retry attempts on 429 errors

        Returns:
            Response text or None on failure
        """
        if not self._available:
            return None

        self._wait_for_rate_limit()

        for attempt in range(max_retries):
            try:
                self._record_vision_call()

                if image is not None:
                    response = self._client.models.generate_content(
                        model=self._model, contents=[prompt, image]
                    )
                else:
                    response = self._client.models.generate_content(
                        model=self._model, contents=prompt
                    )

                return response.text.strip()

            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit errors (429)
                if '429' in error_str or 'resource' in error_str or 'quota' in error_str:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(f"Rate limit hit (429), retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exhausted after {max_retries} retries")
                        return None
                else:
                    logger.error(f"Gemini vision call error: {e}")
                    return None

        return None

    def plan_from_vision(
        self,
        utterance: str,
        frame_bgr: np.ndarray,
        persistent_state: dict,
    ) -> dict:
        """Phase 1: Pre-command scene analysis with Gemini Vision.

        Analyzes the scene BEFORE applying any masks/filters to produce an
        informed command plan. This is the PRIMARY decision driver.

        Args:
            utterance: User's voice command
            frame_bgr: Current camera frame (BGR)
            persistent_state: Dict with:
                - per_target_state: {label: {effect_type, color_hex, intensity, ...}}
                - full_screen_filter_state: {type, intensity, color_hex}
                - last_utterance: Last command text
                - last_sam_prompts: Current SAM3 prompts

        Returns:
            {
                "targets": ["chair", "desk"],
                "effect_type": "color",
                "color_hex": "#2D5016",
                "intensity": 0.85,
                "action": "add" | "remove" | "change",
                "invert": false,
                "full_screen_filter": "none" | "dim" | "warm" | ...,
                "full_screen_intensity": 0.5,
                "full_screen_color": "#...",
                "new_prompts_for_sam": ["chair", "desk"],  # Objects not yet segmented
                "reasoning": "..."
            }
        """
        if not self._available:
            raise RuntimeError(
                "Gemini Vision unavailable (no API key or not initialized). "
                "Pipeline configured to crash on vision unavailability."
            )

        try:
            # Encode frame as JPEG
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            # Resize to ≤1024px to stay within token limits
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])

            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(buffer.tobytes()))

            # Build context from persistent state
            per_target = persistent_state.get("per_target_state", {})
            full_screen = persistent_state.get("full_screen_filter_state", {})
            last_cmd = persistent_state.get("last_utterance", "")
            sam_prompts = persistent_state.get("last_sam_prompts", set())

            # Format active effects for prompt
            active_effects_str = ""
            if per_target:
                effects_list = [f"{label}: {state.get('effect_type', 'none')} @ {state.get('intensity', 1.0):.1f}"
                               f" (color={state.get('color_hex', 'none')})"
                               for label, state in per_target.items()]
                active_effects_str = ", ".join(effects_list)
            else:
                active_effects_str = "none"

            full_screen_str = "none"
            if full_screen:
                full_screen_str = f"{full_screen.get('type', 'none')} @ {full_screen.get('intensity', 0.5):.1f} (color={full_screen.get('color_hex', 'none')})"

            sam_prompts_str = ", ".join(sorted(sam_prompts)) if sam_prompts else "none"

            prompt = f"""You see the user's current AR view. The user said: "{utterance}"

Current state:
- Active effects per object: {active_effects_str}
- Full-screen filter: {full_screen_str}
- Last command: "{last_cmd}"
- Current SAM3 prompts (already segmented): {sam_prompts_str}

Your task:
1. Identify ALL visible objects in the scene (not limited to what SAM3 has segmented).
2. For generic requests ("it's too bright", "dim the room"), infer which objects in THIS scene should be targeted.
3. For color requests ("make the chair green", "darker green"), use the prior state to interpret relative terms:
   - "darker green" = same hue, lower luminance than current chair color
   - "very dark green" = significantly darkened green (provide exact hex, e.g. #1a3d0a)
   - CRITICAL: When darkening colors, PRESERVE the hue. "very dark blue" must still look BLUE, not black!
4. Output exact color values when appropriate: suggest specific #RRGGBB or #RRGGBBAA for Unity.
5. If objects are not yet in SAM3 prompts, add them to "new_prompts_for_sam" so SAM3 can segment them.

Available Tools:
Asset Masks (per-segment effects):
- blur (intensity 0.0-1.0): Gaussian blur on passthrough
- dim (intensity): Semi-transparent black overlay
- pixelate (intensity): Block/mosaic pattern
- highlight (intensity): Warm yellow brightening
- outline (intensity): Colored border
- color (color_hex, intensity): RGB color overlay with opacity

Full-Screen Filters:
- dim, warm, cool, night, grayscale, color (with color_hex)

Color Precision Rules:
- Specify exact RGB as #RRGGBB or #RRGGBBAA (with alpha)
- For "dark green": #006400, #1a3d0a (still GREEN, not black!)
- For "darker green" (relative to current): reduce luminance by ~20% while preserving hue
- For "very dark green": multiply RGB by 0.40 (e.g., #00FF00 → #006600, clearly green)
- intensity (0.0-1.0) maps to overlay opacity

Coordination Rule:
- Full-screen filters and asset masks must complement each other
- If full-screen dim is active, asset masks (e.g., highlight on person) should remain visible

Output JSON:
{{
  "targets": ["chair", "desk"],
  "effect_type": "color",
  "color_hex": "#2D5016",
  "intensity": 0.85,
  "action": "add" | "remove" | "change",
  "invert": false,
  "full_screen_filter": "none" | "dim" | "warm" | ...,
  "full_screen_intensity": 0.5,
  "full_screen_color": "#...",
  "new_prompts_for_sam": ["chair", "desk"],
  "reasoning": "..."
}}

IMPORTANT: Only include objects in "new_prompts_for_sam" if they are NOT already in the current SAM3 prompts list above."""

            # Call Gemini Vision with retry
            text = self._call_gemini_vision_with_retry(prompt, pil_img)
            if not text:
                raise RuntimeError(
                    "Gemini Vision call failed or rate limited after retries. "
                    "Pipeline configured to crash on vision unavailability."
                )

            # Parse JSON response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            import json
            result = json.loads(text)

            logger.info(f"Phase 1 (plan_from_vision): targets={result.get('targets', [])} | "
                       f"effect={result.get('effect_type', 'none')} | "
                       f"new_prompts={result.get('new_prompts_for_sam', [])}")

            return result

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"plan_from_vision error: {e}", exc_info=True)
            raise RuntimeError(
                f"Gemini Vision plan_from_vision failed: {e}. "
                "Pipeline configured to crash on vision unavailability."
            ) from e

    def verify_and_correct(
        self,
        frame_with_overlay: np.ndarray,
        executed_plan: dict,
        utterance: str,
        max_corrections: int = 2,
    ) -> dict:
        """Phase 2: Post-SAM3 verification with iterative correction.

        Verifies that the applied effects match user intent. If not, proposes
        corrections and iterates until verified or max_corrections reached.

        Args:
            frame_with_overlay: Composite frame showing effects as user sees them
            executed_plan: The plan that was executed (from Phase 1 or planner)
            utterance: Original user command
            max_corrections: Maximum correction rounds (default 2)

        Returns:
            {
                "verified": true/false,
                "corrections": {  # Only present if verified=false
                    "targets": [...],
                    "color_hex": "#...",
                    "intensity": 0.8,
                    "full_screen_filter": "none" | ...
                },
                "reasoning": "..."
            }
        """
        if not self._available:
            raise RuntimeError(
                "Gemini Vision unavailable for verification (no API key or not initialized). "
                "Pipeline configured to crash on vision unavailability."
            )

        try:
            # Encode frame as JPEG
            frame_rgb = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            if max(h, w) > 768:
                scale = 768 / max(h, w)
                frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 75])

            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(buffer.tobytes()))

            targets_str = ", ".join(executed_plan.get("targets", []))
            effect_type = executed_plan.get("effect_type", "none")
            color_hex = executed_plan.get("color_hex", "")
            intensity = executed_plan.get("intensity", 1.0)
            full_screen = executed_plan.get("full_screen_filter", "none")

            prompt = f"""You see the AR view AFTER the following was applied:
- User said: "{utterance}"
- Executed: {effect_type} on [{targets_str}], color={color_hex}, intensity={intensity:.2f}, full_screen={full_screen}

Does the result match the user's intent?
1. Are the correct objects covered?
2. Is the color/intensity/opacity appropriate? (e.g., "dark green" should look like dark green, not black)
3. Do full-screen filters and asset masks work well together (no accidental obscuring)?

IMPORTANT: Only flag corrections for MEANINGFUL mismatches:
- Wrong targets (effect on wrong object)
- Obviously wrong color/intensity (e.g., "green" looks black, or "subtle" is too strong)
- Filters obscuring important regions

Do NOT nitpick minor aesthetic differences or suggest corrections for acceptable results.

If correct: output {{"verified": true, "reasoning": "..."}}

If not correct: output {{
  "verified": false,
  "corrections": {{
    "targets": [...],
    "color_hex": "#...",
    "intensity": 0.8,
    "full_screen_filter": "none" | ...
  }},
  "reasoning": "..."
}}"""

            # Call Gemini Vision with retry
            text = self._call_gemini_vision_with_retry(prompt, pil_img)
            if not text:
                raise RuntimeError(
                    "Gemini Vision verification call failed or rate limited. "
                    "Pipeline configured to crash on vision unavailability."
                )

            # Parse JSON response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            import json
            result = json.loads(text)

            verified = result.get("verified", True)
            logger.info(f"Phase 2 (verify_and_correct): verified={verified} | "
                       f"reasoning={result.get('reasoning', '')}")

            return result

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"verify_and_correct error: {e}", exc_info=True)
            raise RuntimeError(
                f"Gemini Vision verify_and_correct failed: {e}. "
                "Pipeline configured to crash on vision unavailability."
            ) from e

    def shutdown(self):
        """Clean up resources."""
        self._client = None
        self._available = False
        self._last_snapshot_result = None
