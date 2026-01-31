"""LLM-Powered Intent Interpreter using GPT/Claude."""
from __future__ import annotations
import json
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

from src.core.contracts import (
    UserIntent, TransformOperation, TransformParameters, Modality,
    VisualOperation, AudioOperation, SpatialRelation, ObjectClass, Magnitude,
)

SYSTEM_PROMPT = """You are a visual effects interpreter for an AR/XR system that can modify what the user sees in real-time.

The user will give you a verbal command about what they want to change in their view. Your job is to interpret this into a structured effect.

Available EFFECTS you can apply:
- blur: Blur/soften an object or area
- color_overlay: Apply a colored tint (specify RGB color)
- darken: Make something darker/dimmer
- brighten: Make something brighter
- pixelate: Pixelate/anonymize something
- desaturate: Remove color (grayscale)
- highlight: Add a colored outline/highlight
- hide: Heavily blur to obscure
- thermal: Apply thermal/heat vision effect
- invert: Invert colors

Available TARGETS (what to apply the effect to):
- face: The user's face
- person: The user's whole body
- background: Everything except the person
- hands: The user's hands
- object: A specific object (describe which one)
- region: A spatial region (left, right, top, bottom, center)
- everything: The entire view
- specific: User described something specific (extract description)

RESPOND WITH ONLY VALID JSON in this exact format:
{
    "effect": "effect_name",
    "target": "target_type",
    "target_description": "description if target is 'specific' or 'object'",
    "color": [R, G, B] or null,
    "intensity": 0.0 to 1.0,
    "spatial_region": "left/right/top/bottom/center" or null,
    "understood": true/false,
    "explanation": "brief explanation of what you understood"
}

If you don't understand the request, set "understood" to false and explain why.

Be LIBERAL in interpretation - try to find a reasonable effect even for unusual requests."""

@dataclass
class LLMInterpretation:
    """Result from LLM interpretation."""
    effect: str
    target: str
    target_description: Optional[str]
    color: Optional[Tuple[int, int, int]]
    intensity: float
    spatial_region: Optional[str]
    understood: bool
    explanation: str
    raw_response: Dict[str, Any]

class LLMInterpreter:
    """LLM-powered command interpreter."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", fallback_to_rules: bool = True):
        self.model = model
        self.fallback_to_rules = fallback_to_rules
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Optional[OpenAI] = None
        self._is_available = False
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self._client = OpenAI(api_key=self.api_key)
                self._is_available = True
                logger.info(f"LLM interpreter initialized with {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        else:
            logger.warning("LLM interpreter not available (no API key or OpenAI not installed)")

    def interpret(self, command: str) -> LLMInterpretation:
        """Interpret a verbal command using LLM."""
        if not self._is_available:
            return self._fallback_interpret(command)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": command}
                ],
                temperature=0.3,
                max_tokens=500,
            )
            content = response.choices[0].message.content
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                color = None
                if data.get("color") and isinstance(data["color"], list) and len(data["color"]) == 3:
                    color = tuple(data["color"])
                return LLMInterpretation(
                    effect=data.get("effect", "blur"),
                    target=data.get("target", "everything"),
                    target_description=data.get("target_description"),
                    color=color,
                    intensity=float(data.get("intensity", 0.5)),
                    spatial_region=data.get("spatial_region"),
                    understood=data.get("understood", True),
                    explanation=data.get("explanation", ""),
                    raw_response=data,
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                return self._fallback_interpret(command)
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return self._fallback_interpret(command)

    def _fallback_interpret(self, command: str) -> LLMInterpretation:
        """Fallback rule-based interpretation."""
        command_lower = command.lower().strip()
        if len(command_lower) < 2:
            return LLMInterpretation(
                effect="blur", target="everything", target_description=None,
                color=None, intensity=0.7, spatial_region=None,
                understood=False, explanation="Command too short", raw_response={},
            )
        effect = "blur"
        if any(w in command_lower for w in ["color", "yellow", "blue", "red", "green", "pink", "purple", "orange", "tint", "warm", "cool", "neon", "glow", "rgb", "hue"]):
            effect = "color_overlay"
        elif any(w in command_lower for w in ["dark", "dim", "shadow", "night", "low light", "shade", "reduce", "less", "lower", "decrease", "black"]):
            effect = "darken"
        elif any(w in command_lower for w in ["bright", "light", "brighter", "lighten", "increase", "more", "boost", "enhance", "white", "flash"]):
            effect = "brighten"
        elif any(w in command_lower for w in ["pixel", "censor", "anonymize", "mosaic", "block", "squares", "minecraft", "8-bit", "retro"]):
            effect = "pixelate"
        elif any(w in command_lower for w in ["gray", "grey", "desaturate", "black and white", "monochrome", "bw", "b&w", "noir", "colorless", "dull", "mute"]):
            effect = "desaturate"
        elif any(w in command_lower for w in ["hide", "remove", "obscure", "cover", "erase", "delete", "block out", "privacy", "censor", "invisible"]):
            effect = "hide"
        elif any(w in command_lower for w in ["highlight", "outline", "border", "frame", "edge", "contour"]):
            effect = "highlight"
        elif any(w in command_lower for w in ["thermal", "heat", "infrared", "ir", "night vision", "predator"]):
            effect = "thermal"
        elif any(w in command_lower for w in ["invert", "negative", "reverse", "opposite", "flip"]):
            effect = "invert"
        elif any(w in command_lower for w in ["blur", "soft", "soften", "fuzzy", "hazy", "out of focus", "smooth", "smudge", "unfocus", "defocus", "depth of field"]):
            effect = "blur"
        target = "everything"
        if any(w in command_lower for w in ["face", "head"]):
            target = "face"
        elif any(w in command_lower for w in ["person", "body", "me", "myself", "i am", "myself"]):
            target = "person"
        elif any(w in command_lower for w in ["background", "behind", "wall", "room", "back"]):
            target = "background"
        elif any(w in command_lower for w in ["hand", "finger", "palm"]):
            target = "hands"
        elif any(w in command_lower for w in ["object", "thing", "item", "that"]):
            target = "specific"
        color = None
        color_map = {
            "red": (255, 50, 50), "green": (50, 255, 50), "blue": (50, 100, 255),
            "yellow": (255, 255, 50), "orange": (255, 165, 0), "pink": (255, 105, 180),
            "purple": (180, 50, 255), "cyan": (50, 255, 255), "neon": (57, 255, 20),
            "white": (255, 255, 255), "warm": (255, 200, 150), "cool": (150, 180, 255),
            "gold": (255, 215, 0), "magenta": (255, 0, 255),
        }
        for color_name, rgb in color_map.items():
            if color_name in command_lower:
                color = rgb
                if effect == "blur":
                    effect = "color_overlay"
                break
        spatial = None
        if "left" in command_lower:
            spatial = "left"
        elif "right" in command_lower:
            spatial = "right"
        elif "top" in command_lower or "above" in command_lower:
            spatial = "top"
        elif "bottom" in command_lower or "below" in command_lower:
            spatial = "bottom"
        elif "center" in command_lower or "middle" in command_lower:
            spatial = "center"
        intensity = 0.7
        if any(w in command_lower for w in ["very", "super", "extreme", "maximum", "max", "a lot", "heavy"]):
            intensity = 0.95
        elif any(w in command_lower for w in ["slight", "little", "bit", "subtle", "soft", "gentle", "light"]):
            intensity = 0.4
        elif any(w in command_lower for w in ["medium", "moderate", "normal"]):
            intensity = 0.6
        logger.info(f"Interpreted '{command}' -> {effect} on {target} (intensity: {intensity})")
        return LLMInterpretation(
            effect=effect, target=target, target_description=command,
            color=color, intensity=intensity, spatial_region=spatial,
            understood=True, explanation=f"Voice command: '{command}' -> {effect} on {target}",
            raw_response={},
        )

    def interpretation_to_operation(self, interp: LLMInterpretation, available_object_ids: List[str]) -> Optional[TransformOperation]:
        """Convert LLM interpretation to a TransformOperation."""
        if not interp.understood:
            return None
        operation = None
        if interp.effect in ["blur", "hide", "soft", "soften"]:
            operation = VisualOperation.EDGE_PRESERVING_BLUR
        elif interp.effect in ["color_overlay", "color", "tint", "highlight"]:
            operation = VisualOperation.COLOR_TEMPERATURE_SHIFT
        elif interp.effect in ["darken", "dim", "shadow"]:
            operation = VisualOperation.BRIGHTNESS_ATTENUATION
        elif interp.effect in ["brighten", "light"]:
            operation = VisualOperation.BRIGHTNESS_ATTENUATION
        elif interp.effect in ["pixelate", "censor"]:
            operation = VisualOperation.TEXTURE_SIMPLIFICATION
        elif interp.effect in ["desaturate", "grayscale", "gray"]:
            operation = VisualOperation.SATURATION_REDUCTION
        elif interp.effect == "thermal":
            operation = VisualOperation.COLOR_TEMPERATURE_SHIFT
        else:
            operation = VisualOperation.EDGE_PRESERVING_BLUR
        params = TransformParameters()
        if operation == VisualOperation.EDGE_PRESERVING_BLUR:
            params.blur_radius = interp.intensity * 30
        elif operation == VisualOperation.BRIGHTNESS_ATTENUATION:
            if interp.effect == "brighten":
                params.brightness_factor = 1.0 + interp.intensity * 0.5
            else:
                params.brightness_factor = 1.0 - interp.intensity * 0.6
        elif operation == VisualOperation.SATURATION_REDUCTION:
            params.saturation_factor = 1.0 - interp.intensity
        elif operation == VisualOperation.COLOR_TEMPERATURE_SHIFT:
            if interp.color:
                r, g, b = interp.color
                if r > b:
                    params.color_temperature_shift = interp.intensity
                else:
                    params.color_temperature_shift = -interp.intensity
            else:
                params.color_temperature_shift = interp.intensity * 0.8
        elif operation == VisualOperation.TEXTURE_SIMPLIFICATION:
            params.texture_simplification = interp.intensity
        target_ids = []
        if interp.target == "everything" or interp.target == "background":
            target_ids = available_object_ids if available_object_ids else ["__GLOBAL__"]
        elif interp.target == "face":
            for obj_id in available_object_ids:
                if "face" in obj_id.lower():
                    target_ids.append(obj_id)
            if not target_ids:
                target_ids = available_object_ids[:1] if available_object_ids else ["__GLOBAL__"]
        elif interp.target == "person":
            for obj_id in available_object_ids:
                if "person" in obj_id.lower():
                    target_ids.append(obj_id)
            if not target_ids:
                target_ids = available_object_ids if available_object_ids else ["__GLOBAL__"]
        else:
            target_ids = available_object_ids if available_object_ids else ["__GLOBAL__"]
        import uuid
        return TransformOperation(
            operation_id=str(uuid.uuid4())[:8],
            target_ids=target_ids,
            modality=Modality.VISUAL,
            operation=operation,
            parameters=params,
            transition_time_seconds=0.3,
            is_active=True,
            progress=0.0,
        )

    @property
    def is_available(self) -> bool:
        return self._is_available
