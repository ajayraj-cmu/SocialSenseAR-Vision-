"""Voice Agent Pipeline — natural language control of SAM3 + effects.

Architecture:
1. WakeWordGate: Sliding window detection of "vibe" across recent transcripts
2. UtteranceAssembler: Collects speech until timeout (4s) or "thank you"
3. VoiceCommandPlanner: Uses Gemini to map intent → structured command plan

Design:
- Local faster-whisper transcription (no cloud API for wake word detection)
- 1s chunks during listening (fast), 2s during recording (accurate)
- Single-pass execution: "vibe blur laptop" executes immediately
- Audio processing runs asynchronously (does not block frame loop)
- Commands execute atomically after full utterance is collected
"""

import time
import logging
import threading
import queue
import re
from typing import Optional
from dataclasses import dataclass
import numpy as np

from server.config import EFFECT_TYPES

logger = logging.getLogger(__name__)


@dataclass
class CommandPlan:
    """Structured command plan from Gemini reasoning."""
    targets: list[str]  # object labels to target (SAM3 segments these)
    effect_type: str  # "blur", "dim", "pixelate", "color", etc.
    intensity: float  # 0.0-1.0
    action: str  # "add", "remove", "change"
    invert: bool = False  # True = effect on everything EXCEPT targets (targets are the "safe" objects)
    color_hex: Optional[str] = None  # NEW: hex color for "color" effect (e.g., "#FF0000")
    full_screen_filter: Optional[str] = None  # "dim", "warm", "color", etc.
    full_screen_intensity: float = 0.5
    full_screen_color: Optional[str] = None  # NEW: hex color for full-screen "color" filter
    reasoning: str = ""
    needs_clarification: bool = False  # Chain-of-thought: True when intent is ambiguous
    clarification_question: str = ""  # Follow-up question to ask the user


class WakeWordGate:
    """Sliding window wake word detection.

    Triggers on just "vibe" (single word) across recent transcripts.
    Sliding window catches wake words split across chunk boundaries.
    Returns remainder text after wake word for single-pass execution.
    """

    WAKE_PATTERNS = [
        r'\bhey[\s,]+vibe\b',    # Full phrase (highest priority)
        r'\bhey[\s,]+vibes\b',
        r'\bhey[\s,]+bye\b',     # Common Whisper mishearing of "hey vibe"
        r'\bhey[\s,]+vive\b',
        r'\bhey[\s,]+five\b',
        r'\bvibe\b',             # Single word
        r'\bvibes\b',
        r'\bvive\b',
        r'\bvybe\b',
        r'\bvine\b',
        r'\bvise\b',
    ]

    def __init__(self, window_size: int = 3):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.WAKE_PATTERNS]
        self._window: list[str] = []
        self._window_size = window_size

    def push_and_check(self, transcript: str) -> tuple[bool, str]:
        """Add transcript to sliding window, check combined text for wake word.

        Returns:
            (detected, remainder_text_after_wake_word)
        """
        self._window.append(transcript)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        combined = ' '.join(self._window)

        for pattern in self._patterns:
            match = pattern.search(combined)
            if match:
                remainder = combined[match.end():].strip()
                self._window.clear()
                return True, remainder

        return False, ""

    def reset(self):
        """Clear the sliding window."""
        self._window.clear()


class UtteranceAssembler:
    """Assembles utterance after wake word.

    Auto-completes after timeout (no end phrase required).
    "Thank you" / "thanks" is an optional early terminator.
    """

    END_PATTERNS = [
        r'\bthank\s+you\b',
        r'\bthanks\b',
        r'\bthank\s+ya\b',
    ]

    # Strip wake word leakage from recording chunks (Whisper re-transcribes from overlap)
    _WAKE_STRIP = re.compile(
        r'\b(?:hey\s+)?(?:vibe|vibes|vive|vybe|vine|vise|bye)\b[,\s]*',
        re.IGNORECASE,
    )

    def __init__(self, timeout: float = 4.5):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.END_PATTERNS]
        self._active = False
        self._chunks: list[str] = []
        self._timeout = timeout
        self._last_chunk_time: float = 0.0

    def start(self, initial_text: str = ""):
        """Start collecting utterance. initial_text is any command text
        that appeared in the same transcript as the wake word."""
        self._active = True
        self._chunks.clear()
        self._last_chunk_time = time.time()
        if initial_text.strip():
            self._chunks.append(initial_text.strip())

    def add_chunk(self, text: str) -> Optional[str]:
        """Add transcript chunk. Returns complete utterance if end phrase detected."""
        if not self._active:
            return None

        # Strip wake word that Whisper re-transcribes from overlapping audio
        text = self._WAKE_STRIP.sub('', text).strip()
        if not text:
            return None

        self._chunks.append(text)
        self._last_chunk_time = time.time()

        # Check for optional end phrase
        full_text = ' '.join(self._chunks)
        for pattern in self._patterns:
            if pattern.search(full_text):
                utterance = pattern.sub('', full_text, count=1).strip()
                self.reset()
                return utterance if utterance else None

        return None

    def check_timeout(self) -> Optional[str]:
        """Check if timeout has elapsed. Returns utterance if so."""
        if not self._active or not self._chunks:
            return None
        if time.time() - self._last_chunk_time >= self._timeout:
            full_text = ' '.join(self._chunks)
            # Strip any end phrases that might be partial
            for pattern in self._patterns:
                full_text = pattern.sub('', full_text).strip()
            self.reset()
            return full_text if full_text else None
        return None

    def has_content(self) -> bool:
        """True if there's at least some text collected."""
        return self._active and len(self._chunks) > 0

    def reset(self):
        """Reset state (ready for next wake word)."""
        self._active = False
        self._chunks.clear()

    @property
    def is_active(self) -> bool:
        return self._active

    def get_partial(self) -> str:
        """Get current partial utterance (for display)."""
        return ' '.join(self._chunks)


class VoiceCommandPlanner:
    """Uses Gemini reasoning to map natural language → structured command plan.

    Handles:
    - Explicit requests ("blur the laptop")
    - Implicit requests ("it's too bright" → dim lights/screens)
    - Effect changes ("stop blurring the laptop")
    - Full-screen filters ("dim everything")
    """

    def __init__(self, api_key: str = None, model: str = None):
        self._api_key = api_key
        self._model = model or "gemini-2.5-flash-lite"
        self._client = None
        self._available = False

    def initialize(self):
        """Initialize Gemini API client (google.genai SDK)."""
        import os
        api_key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key — voice planning disabled")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._available = True
            logger.info(f"Voice command planner ready (model={self._model})")
        except Exception as e:
            logger.error(f"Voice planner init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    # --- Fast-path regex patterns (built from EFFECT_TYPES + custom types) ---
    _EFFECTS = '|'.join(EFFECT_TYPES)  # "blur|dim|pixelate|highlight|outline|color"
    _CUSTOM_EFFECTS = "frosted_glass|frosted glass|redact|grayscale|spotlight|desaturate|noise|soft_glow|warm_tint|cool_tint"
    _ALL_EFFECTS = _EFFECTS + '|' + _CUSTOM_EFFECTS
    # Gerund/variant forms for "stop blurring", "remove dimming", etc.
    _EFFECTS_GERUND = (
        r'blur(?:ring)?|pixelat(?:e|ing)|dim(?:ming)?|highlight(?:ing)?|outlin(?:e|ing)|color(?:ing)?'
    )
    _FILLER = {'the', 'a', 'an', 'my', 'that', 'this'}
    _SELF_WORDS = {'me', 'myself', 'us'}
    _NOT_TARGETS = {'everything', 'every', 'all', 'anything'}  # incomplete phrases, not real objects

    _RE_CLEAR = re.compile(r'^(?:clear|reset|stop|off)$', re.IGNORECASE)
    _RE_EFFECT_INVERT = re.compile(
        rf'^({_EFFECTS})\s+(?:every\s*thing|everything|all)\s+(?:but|except|nothing\s+but)\s+(.+)$', re.IGNORECASE,
    )
    _RE_EFFECT_TARGET = re.compile(
        rf'^({_EFFECTS})\s+(.+)$', re.IGNORECASE,
    )
    _RE_REMOVE = re.compile(
        rf'^(?:stop|remove)\s+({_EFFECTS_GERUND})\s*(?:from\s+)?(.+)$', re.IGNORECASE,
    )
    _RE_UN_EFFECT = re.compile(
        rf'^un({_EFFECTS})\s+(.+)$', re.IGNORECASE,
    )

    # Environmental request patterns for full-screen filters
    _RE_TOO_BRIGHT = re.compile(
        r'^(?:it\'?s\s+)?(?:too|very|really|extremely)\s+bright', re.IGNORECASE,
    )
    _RE_TOO_DARK = re.compile(
        r'^(?:it\'?s\s+)?(?:too|very|really|extremely)\s+dark', re.IGNORECASE,
    )
    _RE_TOO_COLORFUL = re.compile(
        r'^(?:it\'?s\s+)?(?:too|very|really|so)\s+(?:many\s+)?color(?:s|ful)', re.IGNORECASE,
    )
    _RE_MAKE_DARKER = re.compile(
        r'^(?:make\s+(?:it|everything|the\s+room)\s+)?(?:darker|more\s+dark)', re.IGNORECASE,
    )
    _RE_MAKE_BRIGHTER = re.compile(
        r'^(?:make\s+(?:it|everything|the\s+room)\s+)?(?:brighter|more\s+bright|lighter)', re.IGNORECASE,
    )
    _RE_WARMER_TONE = re.compile(
        r'^(?:make\s+(?:it|everything)\s+)?(?:warmer|warm(?:er)?\s+tone)', re.IGNORECASE,
    )
    _RE_COOLER_TONE = re.compile(
        r'^(?:make\s+(?:it|everything)\s+)?(?:cooler|cool(?:er)?\s+tone)', re.IGNORECASE,
    )
    _RE_LESS_SATURATED = re.compile(
        r'^(?:make\s+(?:it|everything)\s+)?(?:less\s+saturated|desaturate|grayscale|black\s+and\s+white)', re.IGNORECASE,
    )
    # Filter removal patterns
    _RE_REMOVE_FILTER = re.compile(
        r'^(?:remove|clear|stop|undo|turn\s+off)\s+(?:the\s+)?(?:filter|dimming|dim|tint|tone|effect)', re.IGNORECASE,
    )
    _RE_NORMAL_LIGHTING = re.compile(
        r'^(?:normal|default|original|regular)\s+(?:lighting|brightness|view)', re.IGNORECASE,
    )
    _RE_REMOVE_DIM = re.compile(
        r'^(?:un|remove|stop)\s*dim(?:ming)?', re.IGNORECASE,
    )

    # Deterministic language hints so intensity/brightness obey user phrasing
    # even when LLM output is conservative.
    _INTENSITY_HINTS = [
        # Increased all values to make effects stronger by default
        (r'\b(extremely|super|maximum|max|absolutely|completely|totally)\b', 1.0),
        (r'\b(opaque|solid)\b', 1.0),
        (r'\b(very|really|highly|heavily)\b', 1.0),  # Increased from 0.9
        (r'\b(quite|pretty)\b', 0.9),                # Increased from 0.8
        (r'\b(moderately|somewhat)\b', 0.7),         # Increased from 0.6
        (r'\b(semi-transparent|semi transparent|translucent)\b', 0.5),
        (r'\b(a bit|a little)\b', 0.5),              # Increased from 0.4
        (r'\b(transparent|see-through|see through|sheer)\b', 0.35),
        (r'\b(slightly|barely|subtly|gently|minimally)\b', 0.35),  # Increased from 0.3
    ]
    _DARKNESS_HINTS = [
        # Adjusted multipliers to preserve color hue while darkening
        # "very dark blue" should still look blue, not black
        (r'\b(pitch black|jet black|pure black)\b', 0.0),
        (r'\b(extremely dark)\b', 0.25),  # Increased from 0.08 to preserve color
        (r'\b(very dark)\b', 0.40),       # Increased from 0.20 to preserve color
        (r'\b(dark)\b', 0.55),            # Increased from 0.35 to preserve color
        (r'\b(deep)\b', 0.65),            # Increased from 0.50 to preserve color
    ]

    def _extract_target(self, raw: str) -> Optional[str]:
        """Extract a single confident target from raw text.

        Returns the target noun, or None if we're not confident.
        """
        raw = raw.strip().lower()
        if not raw:
            return "person"

        # me/myself always → person
        if raw in self._SELF_WORDS:
            return "person"

        words = raw.split()

        # Single word → use directly (unless it's an incomplete phrase word)
        if len(words) == 1:
            if words[0] in self._NOT_TARGETS:
                return None  # "every", "everything" etc — incomplete, need more input
            return words[0]

        # Try stripping obvious fillers
        stripped = [w for w in words if w not in self._FILLER]

        # If exactly one word remains → confident
        if len(stripped) == 1:
            return "person" if stripped[0] in self._SELF_WORDS else stripped[0]

        # Still multiple words → not confident, let Gemini handle it
        return None

    def _normalize_effect(self, raw: str) -> str:
        """Map gerund/variant forms back to canonical effect name."""
        raw = raw.lower()
        if raw.startswith('blur'):
            return 'blur'
        if raw.startswith('pixelat'):
            return 'pixelate'
        if raw.startswith('dim'):
            return 'dim'
        if raw.startswith('highlight'):
            return 'highlight'
        if raw.startswith('outlin'):
            return 'outline'
        return 'blur'

    @staticmethod
    def _adjust_hex_brightness(hex_color: str, mult: float) -> str:
        """Darken a hex color by multiplier (0..1)."""
        if not hex_color or not isinstance(hex_color, str):
            return hex_color
        s = hex_color.strip()
        if not s:
            return s
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            return hex_color
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            r = max(0, min(255, int(r * mult)))
            g = max(0, min(255, int(g * mult)))
            b = max(0, min(255, int(b * mult)))
            return f"#{r:02X}{g:02X}{b:02X}"
        except Exception:
            return hex_color

    def _apply_language_modifiers(self, utterance: str, plan: CommandPlan) -> CommandPlan:
        """Post-process parsed plan with deterministic intensity/brightness hints."""
        text = utterance.lower()

        # Intensity hint from language
        hinted_intensity = None
        for pattern, value in self._INTENSITY_HINTS:
            if re.search(pattern, text):
                hinted_intensity = value
                break

        if hinted_intensity is not None and plan.action in ("add", "change"):
            plan.intensity = max(0.0, min(1.0, hinted_intensity))
            if plan.full_screen_filter:
                plan.full_screen_intensity = max(0.0, min(1.0, hinted_intensity))

        # Darkness/black handling for color effects
        dark_mult = None
        for pattern, value in self._DARKNESS_HINTS:
            if re.search(pattern, text):
                dark_mult = value
                break

        is_color_effect = (plan.effect_type == "color") or (plan.full_screen_filter == "color")
        if is_color_effect and dark_mult is not None:
            if "black" in text:
                # User explicitly asked for black; make it truly black and strong.
                plan.color_hex = "#000000" if plan.effect_type == "color" else plan.color_hex
                plan.full_screen_color = "#000000" if plan.full_screen_filter == "color" else plan.full_screen_color
                plan.intensity = max(plan.intensity, 1.0)
                if plan.full_screen_filter:
                    plan.full_screen_intensity = max(plan.full_screen_intensity, 1.0)
            else:
                # Darken the color while preserving hue
                if plan.effect_type == "color" and plan.color_hex:
                    plan.color_hex = self._adjust_hex_brightness(plan.color_hex, dark_mult)
                if plan.full_screen_filter == "color" and plan.full_screen_color:
                    plan.full_screen_color = self._adjust_hex_brightness(plan.full_screen_color, dark_mult)
                # Dark colors need high intensity to be visible
                plan.intensity = max(plan.intensity, 1.0)
                if plan.full_screen_filter:
                    plan.full_screen_intensity = max(plan.full_screen_intensity, 1.0)

        # Color effects without darkness modifiers should still use high intensity by default
        if is_color_effect and plan.action in ("add", "change") and hinted_intensity is None and dark_mult is None:
            plan.intensity = max(plan.intensity, 1.0)
            if plan.full_screen_filter:
                plan.full_screen_intensity = max(plan.full_screen_intensity, 1.0)

        return plan

    # Ambiguous phrases that need clarification (chain-of-thought reasoning)
    _AMBIGUOUS_PATTERNS = [
        re.compile(r'^(?:can you |could you |please )?(?:segment|do something|handle|work on|fix|change|modify|adjust)\s+(?:that|this|it|those|them)\.?$', re.IGNORECASE),
        re.compile(r'^(?:that|this|those|them|it)\s*\.?$', re.IGNORECASE),
        re.compile(r'^(?:can you |could you )?(?:do|help|fix|change)\s+(?:something|anything)\s*(?:with|about|to)?\s*(?:that|this|it|them)?\.?$', re.IGNORECASE),
        re.compile(r'^(?:segment|mask|overlay|effect)\s+(?:that|this|it|those|them)\s*(?:out)?\.?$', re.IGNORECASE),
    ]

    # Clear implicit requests that should NOT trigger clarification
    _CLEAR_IMPLICIT_PATTERNS = [
        re.compile(r'(?:it\'?s?\s+)?(?:too?\s+)?(?:bright|dark|loud|noisy|distracting|cluttered)', re.IGNORECASE),
        re.compile(r'(?:there\'?s?\s+)?(?:too?\s+)?(?:much\s+)?(?:light|glare|clutter|noise)', re.IGNORECASE),
        re.compile(r'(?:i\s+)?(?:can\'?t|cannot)\s+(?:see|focus|concentrate|read)', re.IGNORECASE),
        re.compile(r'(?:everything\s+is\s+)?(?:blinding|blurry|overwhelming)', re.IGNORECASE),
    ]

    def _needs_clarification(self, utterance: str) -> Optional[str]:
        """Check if utterance is ambiguous and needs a follow-up question.
        
        Returns clarification question string, or None if the request is clear.
        """
        text = utterance.strip().lower()
        
        # Clear implicit requests never need clarification
        for pattern in self._CLEAR_IMPLICIT_PATTERNS:
            if pattern.search(text):
                return None
        
        # Check for ambiguous patterns
        for pattern in self._AMBIGUOUS_PATTERNS:
            if pattern.search(text):
                if "segment" in text:
                    return "Which object would you like me to segment? Can you be more specific?"
                elif "fix" in text or "change" in text or "modify" in text or "adjust" in text:
                    return "What would you like me to change, and which object are you referring to?"
                else:
                    return "I'd be happy to help! Which object are you referring to, and what effect would you like?"
        
        return None

    def _try_fast_parse(self, utterance: str) -> Optional[CommandPlan]:
        """Regex fast-path for obvious commands. Returns None if not confident."""
        # Strip Whisper punctuation (periods, commas, exclamation marks)
        text = re.sub(r'[.,!?;:]+', '', utterance).strip()
        
        # Check for ambiguous requests that need clarification (chain-of-thought)
        clarification = self._needs_clarification(text)
        if clarification:
            return CommandPlan(
                targets=[], effect_type="none", intensity=0.0,
                action="add", needs_clarification=True,
                clarification_question=clarification,
                reasoning="chain-of-thought: ambiguous request, asking for clarification",
            )

        # Pattern 1: clear/reset/stop/off
        if self._RE_CLEAR.match(text):
            return CommandPlan(
                targets=[], effect_type="blur", intensity=0.0,
                action="remove", reasoning="fast-path: clear all",
            )

        # Pattern 3 (check before pattern 2 — more specific):
        # "<effect> everything/all but/except <target>"
        m = self._RE_EFFECT_INVERT.match(text)
        if m:
            target = self._extract_target(m.group(2))
            if target is None:
                return None
            return CommandPlan(
                targets=[target], effect_type=m.group(1).lower(),
                intensity=0.9, action="add", invert=True,  # Increased from 0.8
                reasoning=f"fast-path: {m.group(1)} everything but {target}",
            )

        # Pattern 4: "stop/remove <effect> <target>" or "un<effect> <target>"
        m = self._RE_REMOVE.match(text) or self._RE_UN_EFFECT.match(text)
        if m:
            target = self._extract_target(m.group(2))
            if target is None:
                return None
            return CommandPlan(
                targets=[target], effect_type=self._normalize_effect(m.group(1)),
                intensity=0.0, action="remove",
                reasoning=f"fast-path: remove {m.group(1)} from {target}",
            )

        # Pattern 2: "<effect> <target>"
        m = self._RE_EFFECT_TARGET.match(text)
        if m:
            target = self._extract_target(m.group(2))
            if target is None:
                return None
            return CommandPlan(
                targets=[target], effect_type=m.group(1).lower(),
                intensity=0.9, action="add",  # Increased from 0.8
                reasoning=f"fast-path: {m.group(1)} {target}",
            )

        # Pattern 5: Custom effect types (frosted glass, redact, spotlight, etc.)
        custom_match = re.match(
            rf'^(?:apply\s+)?({self._CUSTOM_EFFECTS})\s+(?:on\s+|to\s+)?(.+)$', text, re.IGNORECASE,
        )
        if custom_match:
            effect = custom_match.group(1).lower().replace(' ', '_')
            target = self._extract_target(custom_match.group(2))
            if target is not None:
                # "spotlight" is semantically "dim everything except target"
                if effect == "spotlight":
                    return CommandPlan(
                        targets=[target], effect_type="dim",
                        intensity=0.8, action="add", invert=True,
                        reasoning=f"fast-path: spotlight on {target} (inverted dim)",
                    )
                return CommandPlan(
                    targets=[target], effect_type=effect,
                    intensity=0.9, action="add",
                    reasoning=f"fast-path: {effect} on {target}",
                )

        # Pattern 6: Environmental requests with full-screen filters
        # "too bright" → dim filter + OUTLINE bright objects (so they're visible!)
        if self._RE_TOO_BRIGHT.match(text):
            intensity = self._apply_language_modifiers(text, 0.7)
            return CommandPlan(
                targets=["light", "screen", "window"],
                effect_type="outline",  # Changed from "dim" to "outline" for visibility
                intensity=1.0,  # High intensity so outline is very visible
                action="add",
                full_screen_filter="dim",
                full_screen_intensity=intensity,
                reasoning="fast-path: too bright → dim filter + outline bright objects",
            )

        # "too dark" → clear dim effects + brighten
        if self._RE_TOO_DARK.match(text):
            return CommandPlan(
                targets=[],
                effect_type="dim",
                intensity=0.0,
                action="remove",
                full_screen_filter="none",
                full_screen_intensity=0.0,
                reasoning="fast-path: too dark → remove dim effects",
            )

        # "too many colors" / "too colorful" → desaturate/grayscale filter
        if self._RE_TOO_COLORFUL.match(text):
            intensity = self._apply_language_modifiers(text, 0.6)
            return CommandPlan(
                targets=[],
                effect_type="none",
                intensity=0.0,
                action="add",
                full_screen_filter="grayscale",
                full_screen_intensity=intensity,
                reasoning="fast-path: too colorful → grayscale filter",
            )

        # "make it darker" → night filter + HIGHLIGHT lights (so they're visible!)
        if self._RE_MAKE_DARKER.match(text):
            intensity = self._apply_language_modifiers(text, 0.7)
            return CommandPlan(
                targets=["light", "screen"],
                effect_type="highlight",  # Changed from "dim" to "highlight" for visibility
                intensity=0.9,
                action="add",
                full_screen_filter="night",
                full_screen_intensity=intensity,
                reasoning="fast-path: darker → night filter + highlight lights",
            )

        # "remove the filter" / "clear the dimming" → clear filters and effects
        if self._RE_REMOVE_FILTER.match(text) or self._RE_NORMAL_LIGHTING.match(text) or self._RE_REMOVE_DIM.match(text):
            return CommandPlan(
                targets=[],
                effect_type="none",
                intensity=0.0,
                action="remove",
                full_screen_filter="none",
                full_screen_intensity=0.0,
                reasoning="fast-path: remove filter → clear all filters and effects",
            )

        # "make it brighter" → clear filters
        if self._RE_MAKE_BRIGHTER.match(text):
            return CommandPlan(
                targets=[],
                effect_type="none",
                intensity=0.0,
                action="remove",
                full_screen_filter="none",
                full_screen_intensity=0.0,
                reasoning="fast-path: brighter → clear filters",
            )

        # "warmer tone" → warm filter
        if self._RE_WARMER_TONE.match(text):
            intensity = self._apply_language_modifiers(text, 0.6)
            return CommandPlan(
                targets=[],
                effect_type="none",
                intensity=0.0,
                action="add",
                full_screen_filter="warm",
                full_screen_intensity=intensity,
                reasoning="fast-path: warmer tone → warm filter",
            )

        # "cooler tone" → cool filter
        if self._RE_COOLER_TONE.match(text):
            intensity = self._apply_language_modifiers(text, 0.6)
            return CommandPlan(
                targets=[],
                effect_type="none",
                intensity=0.0,
                action="add",
                full_screen_filter="cool",
                full_screen_intensity=intensity,
                reasoning="fast-path: cooler tone → cool filter",
            )

        # "less saturated" / "grayscale" → grayscale filter
        if self._RE_LESS_SATURATED.match(text):
            intensity = self._apply_language_modifiers(text, 0.7)
            return CommandPlan(
                targets=[],
                effect_type="none",
                intensity=0.0,
                action="add",
                full_screen_filter="grayscale",
                full_screen_intensity=intensity,
                reasoning="fast-path: desaturate → grayscale filter",
            )

        # Pattern 7: bare target (1-2 words, no keywords) → default to blur
        words = text.lower().split()
        if len(words) <= 2:
            target = self._extract_target(text)
            if target is not None:
                return CommandPlan(
                    targets=[target], effect_type="blur",
                    intensity=0.9, action="add",  # Increased from 0.8
                    reasoning=f"fast-path: bare target → blur {target}",
                )

        # Not confident → let Gemini handle it
        return None

    def plan_command(
        self,
        utterance: str,
        known_objects: set[str],
        active_effects: dict[str, dict],
        conversation_state: Optional[dict] = None,
    ) -> CommandPlan:
        """Map natural language utterance to structured command plan.

        Tries regex fast-path first (0ms). Falls back to Gemini for complex commands.
        Passes full pipeline context to Gemini for accurate command generation.
        """
        # Try fast regex parse first
        fast = self._try_fast_parse(utterance)
        if fast:
            fast = self._apply_language_modifiers(utterance, fast)
            if fast.needs_clarification:
                logger.info(f"Fast-path: needs clarification → \"{fast.clarification_question}\"")
            else:
                logger.info(
                    f"Fast-path plan: {fast.action} {fast.effect_type} → "
                    f"{fast.targets} (invert={fast.invert})"
                )
            return fast

        if not self._available:
            logger.warning(f"Gemini unavailable, extracting targets from text: {utterance}")
            return self._apply_language_modifiers(utterance, self._emergency_parse(utterance))

        try:
            known_str = ', '.join(sorted(known_objects)) if known_objects else 'none'
            active_str = ', '.join(f"{k}:{v['type']}@{v.get('intensity', 1.0):.1f}" for k, v in active_effects.items()) if active_effects else 'none'
            
            # Full pipeline context for Gemini (chain-of-thought reasoning)
            conv_state = conversation_state or {}
            pending_clarification = conv_state.get("pending_clarification", False)
            last_command = conv_state.get("last_command", "")

            prompt = f"""Parse this command into a structured action plan. Respond ONLY with JSON.

User said: "{utterance}"

Full Pipeline Context:
- Known objects in scene (SAM3 has detected these): {known_str}
- Currently active effects: {active_str}
- Available effect types: blur, dim, pixelate, highlight, outline, color
- Available custom masks: frosted_glass, redact, grayscale, spotlight, desaturate, noise
- Available full-screen filters: dim, warm, cool, night, grayscale, color
- Pipeline supports: inverted effects (apply to everything EXCEPT target), color overlays with hex colors
- Previous user command: "{last_command}"
- Was waiting for clarification: {pending_clarification}

Output JSON with these fields:
- "targets": list of object labels. ALWAYS include any object the user names, even if not in the known set. SAM3 will search for it.
- "effect_type": {' | '.join(f'"{e}"' for e in EFFECT_TYPES)}
- "color_hex": hex color string (e.g., "#FF0000") if effect_type is "color" or user requests a specific color. IMPORTANT: Adjust brightness based on modifiers (see brightness rules below).
- "intensity": 0.0-1.0 representing effect strength. Extract from intensity modifiers in the command (see intensity rules below). Default: 0.8 for normal effects, 0.5 for full-screen filters.
- "action": "add" | "remove" | "change"
- "invert": true if effect applies to everything EXCEPT targets (e.g. "blur everything but laptop")
- "full_screen_filter": "dim" | "warm" | "cool" | "night" | "grayscale" | "color" | null
- "full_screen_intensity": 0.0-1.0
- "full_screen_color": hex color string if full_screen_filter is "color"
- "needs_clarification": true ONLY if the user's request is AMBIGUOUS (e.g., "segment that", "do something with it") AND you cannot determine the target or effect. false for CLEAR requests.
- "clarification_question": short follow-up question to ask (only when needs_clarification is true)
- "reasoning": brief explanation

CRITICAL CHAIN-OF-THOUGHT RULES:
- If the user says something ambiguous like "segment that", "do something with it", "handle that" — set needs_clarification=true and provide a brief question.
- If the user says something clear and actionable like "it's too bright", "blur the laptop", "dim everything" — set needs_clarification=false and execute immediately. Do NOT ask for clarification on clear requests!
- "it's too bright" → CLEAR: apply dim filter immediately, NO clarification needed.
- "segment that out" → AMBIGUOUS: ask "Which object would you like me to segment?"

Intensity Rules (IMPORTANT - extract from user's language):
- "extremely", "super", "maximum": intensity = 1.0
- "very", "really", "highly", "heavily": intensity = 1.0 (changed to 1.0 for stronger effects)
- "quite", "pretty": intensity = 0.9 (increased from 0.8)
- "moderately", "somewhat": intensity = 0.7 (increased from 0.6)
- "a bit", "a little": intensity = 0.5 (increased from 0.4)
- "slightly", "barely", "subtly": intensity = 0.35 (increased from 0.3)
- No modifier for color effects: intensity = 1.0 (changed from 0.8 - color overlays need high opacity by default)
- No modifier for blur/dim: intensity = 0.9 (changed from 0.8 for stronger effects)
- No modifier for other effects: intensity = 0.8

Brightness Rules for Colors (CRITICAL - calculate exact RGB values):
IMPORTANT: When darkening a color, you MUST preserve its hue (color identity).
"very dark blue" should still clearly look BLUE, not black!

Step-by-step RGB calculation:
1. Start with base color hex (e.g., blue = #0000FF → R:0, G:0, B:255)
2. Apply brightness multiplier to EACH channel separately
3. Round to nearest integer and clamp to 0-255
4. Convert back to hex

Darkness multipliers (PRESERVE COLOR HUE):
- "pitch black", "jet black", "pure black": multiply by 0.0 → #000000 (only for black specifically)
- "extremely dark [color]": multiply RGB by 0.25 (e.g., "extremely dark blue" → #000040, still recognizable as dark blue)
- "very dark [color]": multiply RGB by 0.40 (e.g., "very dark blue" #0000FF → R:0, G:0, B:102 → #000066, clearly blue)
- "dark [color]": multiply RGB by 0.55 (e.g., "dark red" #FF0000 → R:140, G:0, B:0 → #8C0000, clearly red)
- "deep [color]": multiply RGB by 0.65 (e.g., "deep green" → #006B00, clearly green)
- Normal [color]: use base color unchanged (e.g., "blue" → "#0000FF")
- "bright [color]": multiply RGB by 1.2, clamp to 255 (e.g., "bright red" #FF0000 → stays #FF0000, already max)
- "very bright [color]": multiply RGB by 1.4, clamp to 255
- "pale [color]", "pastel [color]": blend 70% color + 30% white

EXAMPLE CALCULATIONS:
- "very dark blue":
  Base: #0000FF (R:0, G:0, B:255)
  Multiply by 0.40: R:0×0.4=0, G:0×0.4=0, B:255×0.4=102
  Result: #000066 ← This is clearly BLUE, not black!

- "dark red":
  Base: #FF0000 (R:255, G:0, B:0)
  Multiply by 0.55: R:255×0.55=140, G:0×0.55=0, B:0×0.55=0
  Result: #8C0000 ← This is clearly RED, not black!

- "extremely dark green":
  Base: #00FF00 (R:0, G:255, B:0)
  Multiply by 0.25: R:0×0.25=0, G:255×0.25=64, B:0×0.25=0
  Result: #004000 ← This is clearly GREEN, not black!

Examples with Intensity & Brightness:
- "make person blue" → targets: ["person"], effect_type: "color", color_hex: "#0000FF", intensity: 1.0, action: "add"
- "make wall very dark blue" → targets: ["wall"], effect_type: "color", color_hex: "#000066" (BLUE preserved!), intensity: 1.0, action: "add"
- "extremely dark black ceiling" → targets: ["ceiling"], effect_type: "color", color_hex: "#000000", intensity: 1.0
- "very bright red wall" → targets: ["wall"], effect_type: "color", color_hex: "#FF0000" (already max brightness), intensity: 1.0, action: "add"
- "slightly blur laptop" → targets: ["laptop"], effect_type: "blur", intensity: 0.3, action: "add"
- "really dim screen" → targets: ["screen"], effect_type: "dim", intensity: 0.9, action: "add"
- "blur phone" → targets: ["phone"], effect_type: "blur", intensity: 0.9, action: "add"
- "blur background" → targets: ["person"], invert: true, effect_type: "blur", intensity: 0.9
- "pixelate laptop" → targets: ["laptop"], effect_type: "pixelate", intensity: 0.9, action: "add"
- "stop blurring person" → targets: ["person"], action: "remove"
- "blur everything but me" → targets: ["person"], invert: true, effect_type: "blur", intensity: 0.9
- "dim everything" → full_screen_filter: "dim", full_screen_intensity: 0.7, targets: []
- "it's too bright" → targets: ["light", "screen", "window"], effect_type: "dim", intensity: 0.8, full_screen_filter: "dim", full_screen_intensity: 0.7
- "too many colors" → full_screen_filter: "grayscale", full_screen_intensity: 0.6, targets: []
- "make it darker" → targets: ["light", "screen"], effect_type: "dim", intensity: 0.8, full_screen_filter: "night", full_screen_intensity: 0.7
- "warmer tone" → full_screen_filter: "warm", full_screen_intensity: 0.6, targets: []
- "cooler tone" → full_screen_filter: "cool", full_screen_intensity: 0.6, targets: []
- "less saturated" → full_screen_filter: "grayscale", full_screen_intensity: 0.7, targets: []
- "make everything blue" → full_screen_filter: "color", full_screen_color: "#0000FF", full_screen_intensity: 0.6, targets: []
- "frosted glass on screen" → targets: ["screen"], effect_type: "frosted_glass", intensity: 0.9
- "redact laptop" → targets: ["laptop"], effect_type: "redact", intensity: 1.0
- "spotlight on me" → targets: ["person"], effect_type: "dim", intensity: 0.8, invert: true
- Base color hex: "blue" → "#0000FF", "red" → "#FF0000", "green" → "#00FF00", "yellow" → "#FFFF00", "orange" → "#FFA500", "purple" → "#800080", "pink" → "#FFC0CB", "black" → "#000000", "white" → "#FFFFFF", etc.
- If user says "highlight X" without color, use effect_type: "highlight" (warm yellow)
- If user says "highlight X [color]", use effect_type: "color" with that color
- NEVER return empty targets if the user names an object
- If just an object name with no action ("person"), default to outline effect

{{
    "targets": ["person"],
    "effect_type": "color",
    "color_hex": "#0000FF",
    "intensity": 0.8,
    "action": "add",
    "invert": false,
    "full_screen_filter": null,
    "full_screen_intensity": 0.5,
    "full_screen_color": null,
    "reasoning": "User wants to color the person blue"
}}"""

            response = self._client.models.generate_content(
                model=self._model, contents=prompt
            )
            text = response.text.strip()

            # Parse JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            import json
            data = json.loads(text)

            plan = CommandPlan(
                targets=data.get("targets", []),
                effect_type=data.get("effect_type") or "blur",
                intensity=float(data.get("intensity") or 0.8),
                action=data.get("action") or "add",
                invert=bool(data.get("invert", False)),
                color_hex=data.get("color_hex"),  # NEW
                full_screen_filter=data.get("full_screen_filter"),
                full_screen_intensity=float(data.get("full_screen_intensity") or 0.5),
                full_screen_color=data.get("full_screen_color"),  # NEW
                reasoning=data.get("reasoning") or "",
                needs_clarification=bool(data.get("needs_clarification", False)),
                clarification_question=data.get("clarification_question") or "",
            )
            plan = self._apply_language_modifiers(utterance, plan)

            if plan.needs_clarification:
                logger.info(f"Gemini: needs clarification → \"{plan.clarification_question}\"")
            else:
                logger.info(f"Gemini plan: {plan.action} {plan.effect_type} → {plan.targets} (invert={plan.invert})")
            return plan

        except Exception as e:
            logger.error(f"Gemini planning error: {e}")
            return self._apply_language_modifiers(utterance, self._emergency_parse(utterance))

    def _emergency_parse(self, utterance: str) -> CommandPlan:
        """Extract targets from text when Gemini is down. Not a full NLP parser —
        just strips obvious action/filler words and returns the object nouns."""
        text = utterance.lower().strip()

        # Detect effect type
        effect = "blur"
        for keyword, etype in [("dim", "dim"), ("darken", "dim"), ("pixelate", "pixelate"),
                                ("highlight", "highlight"), ("outline", "outline")]:
            if keyword in text:
                effect = etype
                break

        # Detect action
        action = "add"
        if any(w in text for w in ["stop", "remove", "unblur", "undim", "off"]):
            action = "remove"

        # Detect negation
        invert = False
        negation = re.search(
            r'(?:everything|all)\s+(?:but|except|around|other\s+than|besides)\s+(?:the\s+)?',
            text,
        )
        if negation:
            invert = True
            # Object is after the negation phrase
            text = text[negation.end():]

        # Strip action/filler words, keep object nouns
        skip = {
            "blur", "dim", "darken", "pixelate", "highlight", "outline",
            "unblur", "stop", "remove", "off", "undim", "clear",
            "everything", "all", "but", "except", "around", "the",
            "a", "an", "my", "that", "this", "please", "can", "you",
            "other", "than", "besides", "me", "it", "on",
        }
        targets = [w for w in text.split() if w not in skip]

        # "me"/"myself" → "person"
        targets = ["person" if t in ("me", "myself", "us") else t for t in targets]

        if not targets:
            targets = [text]  # last resort: use whole text

        logger.info(f"Emergency parse: {action} {effect} → {targets} (invert={invert})")
        return CommandPlan(
            targets=targets,
            effect_type=effect,
            intensity=0.9,  # Increased from 0.8
            action=action,
            invert=invert,
            reasoning="Gemini unavailable — emergency parse",
        )

    def shutdown(self):
        """Clean up."""
        self._client = None
        self._available = False


class VoiceAgent:
    """Complete voice agent pipeline: audio → transcript → command → effect.

    Thread-safe. Audio processing happens asynchronously.
    State updates (conversation_state) are atomic dict replacements.
    """

    def __init__(
        self,
        transcriber,
        planner: VoiceCommandPlanner,
        scene_understanding,
        config,
    ):
        self._transcriber = transcriber
        self._planner = planner
        self._scene = scene_understanding
        self._config = config

        self._wake_gate = WakeWordGate(
            window_size=getattr(config, 'voice_wake_window', 3),
        )
        self._assembler = UtteranceAssembler(
            timeout=getattr(config, 'voice_assembler_timeout', 4.5),
        )

        # Audio buffer
        self._audio_buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._sample_rate = 16000

        # State registries (accessed from pipeline orchestrator)
        self._known_objects: set[str] = set()  # labels we've seen
        self._active_effects: dict[str, dict] = {}  # {label: {"type": "blur", "intensity": 0.8}}
        self._full_screen_filter: Optional[dict] = None  # {"type": "dim", "intensity": 0.5}
        self._state_lock = threading.Lock()

        # Conversation state (for protobuf / dashboard)
        self._conversation_state: dict = {
            "listening": True,
            "recording": False,
            "partial_transcript": "",
            "last_command": "",
            "last_response": "Ready. Say 'Vibe' to start.",
            "last_command_time": 0.0,
        }

        # Audio transcription queue — only latest chunk matters, stale ones are dropped
        self._audio_queue: queue.Queue = queue.Queue()
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)

        # Text command queue — separate thread so typed commands never wait behind audio
        self._text_queue: queue.Queue = queue.Queue()
        self._text_thread = threading.Thread(target=self._text_loop, daemon=True)

        self._running = True

        # Callback for SAM3 prompt sync (set by orchestrator)
        self._on_command_callback = None

        # Cached frame for Gemini Vision
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    def set_on_command_callback(self, callback):
        """Set callback invoked when voice commands execute.

        Signature: callback(action, targets, effect_type, intensity, invert)
        Used by orchestrator to sync SAM3 prompts with voice agent targets.
        """
        self._on_command_callback = callback

    def process_text_command(self, text: str):
        """Process a typed text command through the same Gemini pipeline as voice.

        Skips wake word detection and utterance assembly — goes straight to
        command planning and execution. Runs on a separate thread from audio
        so typed commands are never blocked by audio transcription.
        """
        text = text.strip()
        if not text:
            return
        logger.info(f"Text command: {text}")
        self._text_queue.put(text)

    def start(self):
        """Start background processing threads."""
        self._audio_thread.start()
        self._text_thread.start()
        logger.info("VoiceAgent started (audio + text threads)")

    def ingest_audio(self, pcm16_data: bytes, sample_rate: int, num_samples: int):
        """Called from websocket thread to feed audio data.

        Buffers audio and emits chunks for background transcription.
        No stale chunk dropping — local transcription is cheap.
        """
        with self._buffer_lock:
            self._audio_buffer.extend(pcm16_data)
            self._sample_rate = sample_rate

            # Dynamic chunk size based on phase
            chunk_s = (
                self._config.voice_recording_chunk_s if self._assembler.is_active
                else self._config.voice_listening_chunk_s
            )
            min_bytes = int(sample_rate * chunk_s * 2)  # 2 bytes per PCM16 sample

            # Process all complete chunks (no dropping)
            while len(self._audio_buffer) >= min_bytes:
                chunk_bytes = bytes(self._audio_buffer[:min_bytes])
                self._audio_buffer = self._audio_buffer[min_bytes:]

                # Energy gate: skip silence
                samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
                if rms < self._config.voice_energy_threshold:
                    continue  # silence, skip

                # No stale chunk dropping — local transcription is fast and free
                self._audio_queue.put((chunk_bytes, sample_rate))

    def update_frame(self, frame_bgr: np.ndarray):
        """Update latest camera frame for Gemini Vision (called from pipeline orchestrator)."""
        with self._frame_lock:
            self._latest_frame = frame_bgr.copy()

    def get_known_objects(self) -> set[str]:
        """Get set of known object labels (for SAM3 prompt control)."""
        with self._state_lock:
            return self._known_objects.copy()

    def get_active_effects(self) -> dict[str, dict]:
        """Get current effect state {label: {"type": "blur", "intensity": 0.8}}."""
        with self._state_lock:
            return self._active_effects.copy()

    def get_full_screen_filter(self) -> Optional[dict]:
        """Get full-screen filter state."""
        with self._state_lock:
            return self._full_screen_filter.copy() if self._full_screen_filter else None

    def get_conversation_state(self) -> dict:
        """Get current conversation state (for protobuf)."""
        return self._conversation_state.copy()

    def clear_all_effects(self):
        """Clear all active effects and full-screen filters (called on new client connection)."""
        with self._state_lock:
            self._active_effects.clear()
            self._full_screen_filter = None
            self._known_objects.clear()
            logger.info("VoiceAgent: Cleared all effects, filters, and known objects")

    def add_known_object(self, label: str):
        """Add object to known registry (called when SAM3 segments it)."""
        with self._state_lock:
            self._known_objects.add(label)

    def _audio_loop(self):
        """Background thread: transcribe audio chunks."""
        while self._running:
            try:
                # Short timeout so we can check assembler timeout frequently
                try:
                    audio_bytes, sr = self._audio_queue.get(timeout=0.5)
                    self._handle_transcription(audio_bytes, sr)
                except queue.Empty:
                    pass

                # Check if assembler timed out (auto-complete without end phrase)
                utterance = self._assembler.check_timeout()
                if utterance:
                    logger.info(f"Utterance (auto-complete): \"{utterance}\"")
                    self._execute_command(utterance)
                    self._conversation_state.update({
                        "listening": True,
                        "recording": False,
                        "partial_transcript": "",
                    })

            except Exception as e:
                logger.error(f"Audio loop error: {e}", exc_info=True)

    def _text_loop(self):
        """Background thread: process typed text commands (never blocked by audio)."""
        while self._running:
            try:
                text = self._text_queue.get(timeout=0.5)
                self._execute_command(text)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Text command loop error: {e}", exc_info=True)

    def _handle_transcription(self, audio_bytes: bytes, sample_rate: int):
        """Transcribe audio chunk and process for wake word / command assembly.

        Uses fast model during listening (wake word detection) and
        accurate model during recording (command transcription).
        Supports single-pass execution: "vibe blur laptop" in one breath.
        """
        if not self._transcriber.available:
            return

        # Choose transcription model based on phase
        mode = "recording" if self._assembler.is_active else "listening"
        text = self._transcriber.transcribe(audio_bytes, sample_rate, mode=mode)
        if not text:
            return

        _now = time.monotonic()
        if not hasattr(self, '_last_transcript_log'):
            self._last_transcript_log = 0.0
        if mode == "recording" or _now - self._last_transcript_log >= 5.0:
            logger.info(f"Transcript [{mode}]: \"{text}\"")
            self._last_transcript_log = _now

        # --- LISTENING PHASE: check sliding window for wake word ---
        if not self._assembler.is_active:
            detected, remainder = self._wake_gate.push_and_check(text)
            if not detected:
                return

            logger.info(f"Wake word detected! remainder: \"{remainder}\"")

            # "vibe everything but X" → user means "blur everything but X"
            # The wake word consumed "vibe" but it also implied the default effect
            if remainder and re.match(r'(?:every\s*thing|everything|all)\s+(?:but|except)\s', remainder, re.IGNORECASE):
                remainder = f"blur {remainder}"
                logger.info(f"Injected default effect: \"{remainder}\"")

            # Always record after wake word — never single-pass.
            # 3s chunks split phrases too unpredictably for instant execution.
            # The assembler will collect chunks and fire on timeout (4.5s silence).
            self._assembler.start(initial_text=remainder)
            self._conversation_state.update({
                "listening": False,
                "recording": True,
                "partial_transcript": remainder,
                "last_response": "Listening...",
            })
            return

        # --- RECORDING PHASE: check for wake word as command boundary ---
        # If the user says "vibe" again mid-recording, fire current command and start new one
        wake_hit, wake_remainder = self._wake_gate.push_and_check(text)
        if wake_hit:
            # Fire whatever we've assembled so far
            current = self._assembler.get_partial()
            self._assembler.reset()
            if current.strip():
                logger.info(f"Wake boundary — firing current: \"{current}\"")
                self._execute_command(current.strip())

            # Inject default effect for "everything but X" pattern
            if wake_remainder and re.match(
                r'(?:every\s*thing|everything|all)\s+(?:but|except)\s',
                wake_remainder, re.IGNORECASE,
            ):
                wake_remainder = f"blur {wake_remainder}"

            # Start assembling the new command
            self._assembler.start(initial_text=wake_remainder)
            self._conversation_state.update({
                "partial_transcript": wake_remainder,
                "last_response": "Listening...",
            })
            return

        # Normal recording: assemble utterance
        complete_utterance = self._assembler.add_chunk(text)

        if complete_utterance is not None:
            logger.info(f"Complete utterance: \"{complete_utterance}\"")
            self._execute_command(complete_utterance)
            self._conversation_state.update({
                "listening": True,
                "recording": False,
                "partial_transcript": "",
            })
        else:
            self._conversation_state["partial_transcript"] = self._assembler.get_partial()

    # When user keeps talking and says "Hey Vibe" again mid-stream, use text after the last occurrence.
    _LAST_INTENT_SPLIT = re.compile(r'\bhey\s+vi(?:be)?s?\b', re.IGNORECASE)

    def _execute_command(self, utterance: str):
        """Execute command: Gemini plan → update state → apply effects.

        Chain-of-thought: if the planner returns needs_clarification=True,
        we don't execute any command — instead we send a follow-up question
        to the user and wait for their response in the next utterance.
        """
        # Normalize Whisper artifacts: strip punctuation, collapse whitespace
        utterance = re.sub(r'[.,!?;:]+', ' ', utterance).strip()
        utterance = re.sub(r'\s+', ' ', utterance)
        # "nothing but" / "not but" → "but" (Whisper splits "everything but" across sentences)
        utterance = re.sub(r'\bnothing\s+but\b', 'but', utterance, flags=re.IGNORECASE)

        # Long run-on with a repeated wake phrase: use only the text after the last "Hey Vibe" / "Hey Vi"
        if len(utterance) > 80:
            parts = self._LAST_INTENT_SPLIT.split(utterance)
            if len(parts) > 1 and parts[-1].strip():
                utterance = parts[-1].strip()
                logger.info(f"Using last intent from run-on: \"{utterance}\"")

        t0 = time.time()

        # Get current state
        with self._state_lock:
            known_objs = self._known_objects.copy()
            active_fx = self._active_effects.copy()

        # Plan command using Gemini reasoning (single API call — no scene snapshot)
        # Pass full conversation state for chain-of-thought context
        plan = self._planner.plan_command(
            utterance,
            known_objs,
            active_fx,
            conversation_state=self._conversation_state,
        )

        # --- Chain-of-thought: handle clarification requests ---
        if plan.needs_clarification:
            question = plan.clarification_question or "Can you be more specific about what you'd like me to do?"
            self._conversation_state.update({
                "last_command": utterance,
                "last_response": question,
                "last_command_time": time.time(),
                "pending_clarification": True,
                "clarification_question": question,
            })
            elapsed_ms = (time.time() - t0) * 1000
            logger.info(f"Clarification needed ({elapsed_ms:.0f}ms): \"{question}\"")
            return

        # Clear pending clarification since we're now executing
        self._conversation_state.pop("pending_clarification", None)
        self._conversation_state.pop("clarification_question", None)

        # Normalize "background" semantics for SAM3:
        # "blur/dim background" should mean "apply effect to everything except person".
        # Keep this centralized so fast-path, Gemini, and emergency parsing stay consistent.
        if plan.targets:
            mapped_background = False
            normalized_targets = []
            for target in plan.targets:
                t = (target or "").strip().lower()
                if t == "background":
                    mapped_background = True
                    normalized_targets.append("person")
                elif t:
                    normalized_targets.append(t)

            if normalized_targets:
                # Preserve order while deduplicating.
                plan.targets = list(dict.fromkeys(normalized_targets))

            if mapped_background and plan.action in ("add", "change"):
                plan.invert = True
                logger.info(
                    "Normalized target 'background' -> 'person' with invert=true"
                )

        # Execute plan: update state
        with self._state_lock:
            if plan.action == "add" or plan.action == "change":
                # Clean stale legacy key so "blur background" no longer leaves
                # a literal "background" effect active.
                self._active_effects.pop("background", None)
                # Add/update effects
                for target in plan.targets:
                    self._known_objects.add(target)
                    self._active_effects[target] = {
                        "type": plan.effect_type,
                        "intensity": plan.intensity,
                        "invert": plan.invert,
                        "color_hex": plan.color_hex,  # NEW: store color_hex
                    }
                    mode = "inverted" if plan.invert else "direct"
                    color_info = f" {plan.color_hex}" if plan.color_hex else ""
                    logger.info(f"Applied {plan.effect_type} ({mode}) to {target}{color_info}")

            elif plan.action == "remove":
                # Remove effects (but keep objects in known registry for persistence)
                for target in plan.targets:
                    if target in self._active_effects:
                        del self._active_effects[target]
                        logger.info(f"Removed effect from {target}")

            # Update full-screen filter
            if plan.full_screen_filter:
                self._full_screen_filter = {
                    "type": plan.full_screen_filter,
                    "intensity": plan.full_screen_intensity,
                    "color_hex": plan.full_screen_color,  # NEW: store full_screen_color
                }
                logger.info(f"Applied full-screen filter: {plan.full_screen_filter}")
            elif plan.action == "remove" and not plan.targets:
                # "Clear everything" command
                self._full_screen_filter = None
                self._active_effects.clear()
                logger.info("Cleared all effects")

        # Notify orchestrator to sync SAM3 prompts with voice command
        if self._on_command_callback:
            try:
                self._on_command_callback(
                    plan.action, plan.targets, plan.effect_type, plan.intensity, plan.invert, plan.color_hex,
                )
            except Exception as e:
                logger.error(f"on_command callback error: {e}")

        # Update conversation state
        response = self._generate_response(plan)
        self._conversation_state.update({
            "last_command": utterance,
            "last_response": response,
            "last_command_time": time.time(),
        })

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(f"Command executed in {elapsed_ms:.0f}ms: {response}")

    def _generate_response(self, plan: CommandPlan) -> str:
        """Generate friendly response message for user."""
        if plan.action == "add":
            if plan.targets:
                objs = ", ".join(plan.targets)
                if plan.invert:
                    return f"Applied {plan.effect_type} to everything except {objs}"
                return f"Applied {plan.effect_type} to {objs}"
            elif plan.full_screen_filter:
                return f"Applied {plan.full_screen_filter} filter"
            else:
                return "No matching objects found"
        elif plan.action == "remove":
            if plan.targets:
                objs = ", ".join(plan.targets)
                return f"Removed effects from {objs}"
            else:
                return "Cleared all effects"
        elif plan.action == "change":
            objs = ", ".join(plan.targets)
            return f"Changed {objs} to {plan.effect_type}"
        return "Done"

    def shutdown(self):
        """Stop processing and clean up."""
        self._running = False
        if self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2)
        if self._text_thread.is_alive():
            self._text_thread.join(timeout=2)
        logger.info("VoiceAgent shut down")
