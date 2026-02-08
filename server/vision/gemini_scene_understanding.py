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

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-exp", cache_ttl: float = 10.0):
        self._api_key = api_key
        self._model = model
        self._cache_ttl = cache_ttl
        self._client = None
        self._available = False

        # Cache: frame snapshot results
        self._last_snapshot_time = 0.0
        self._last_snapshot_result: Optional[dict] = None

    def initialize(self):
        """Initialize Gemini API client."""
        import os
        api_key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key — scene understanding disabled")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self._model)
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

            response = self._client.generate_content([prompt, pil_img])
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

            response = self._client.generate_content(prompt)
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

    def shutdown(self):
        """Clean up resources."""
        self._client = None
        self._available = False
        self._last_snapshot_result = None
