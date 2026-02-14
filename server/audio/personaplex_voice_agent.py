"""PersonaPlex Voice Agent — drop-in replacement for VoiceAgent using PersonaPlex.

Uses PersonaPlex (NVIDIA PersonaPlex-7B, speech-to-speech) instead of:
- faster-whisper (transcription)
- WakeWordGate (wake word detection)
- UtteranceAssembler (utterance collection)
- VoiceCommandPlanner/Gemini (command planning)

PersonaPlex handles all of these via its system prompt:
- Wake word detection ("Hey Vibe") via prompt engineering
- Natural language understanding of commands
- Spoken audio response generation
- Command tags ([COMMAND:...]) in text output for effect execution

Same interface as VoiceAgent so the orchestrator needs minimal changes.
"""

import logging
import re
import threading
import time
from typing import Optional

import numpy as np

from server.audio.personaplex_bridge import PersonaPlexBridge
from server.audio.voice_agent import CommandPlan
from server.config import ServerConfig

logger = logging.getLogger(__name__)


PERSONAPLEX_SYSTEM_PROMPT = """You are Vibe, an AR assistant. You respond when the user says Vibe. Stay silent otherwise. When asked to apply an effect, say a short confirmation and include a command tag. Commands: [COMMAND:blur:target], [COMMAND:dim:target], [COMMAND:pixelate:target], [COMMAND:highlight:target], [COMMAND:color:target:#HEX], [COMMAND:clear], [COMMAND:invert_blur:target]. Example: user says blur the laptop, you say Sure, blurring it. [COMMAND:blur:laptop]. User says clear, you say Cleared. [COMMAND:clear]."""


# Regex for parsing [COMMAND:action:target] or [COMMAND:action] or [COMMAND:action:target:#HEX] from PersonaPlex text
_COMMAND_RE = re.compile(r'\[COMMAND:(\w+)(?::([^\]]+))?\]')

# Map PersonaPlex command names to (effect_type, action, invert) tuples
_COMMAND_MAP = {
    "blur": ("blur", "add", False),
    "dim": ("dim", "add", False),
    "pixelate": ("pixelate", "add", False),
    "highlight": ("highlight", "add", False),
    "outline": ("outline", "add", False),
    "color": ("color", "add", False),
    "clear": ("blur", "remove", False),  # clear all
    "remove": ("blur", "remove", False),  # remove from specific target
    "invert_blur": ("blur", "add", True),
    "invert_dim": ("dim", "add", True),
    "invert_pixelate": ("pixelate", "add", True),
    # Custom mask types
    "frosted_glass": ("frosted_glass", "add", False),
    "redact": ("redact", "add", False),
    "spotlight": ("dim", "add", True),  # spotlight = dim everything except target
    "grayscale": ("grayscale", "add", False),
    "desaturate": ("desaturate", "add", False),
}

_FILTER_COMMANDS = {
    "filter_dim": ("dim", 0.6),
    "filter_warm": ("warm", 0.6),
    "filter_cool": ("cool", 0.6),
    "filter_night": ("night", 0.6),
    "filter_grayscale": ("grayscale", 0.6),
}

# Regex to extract hex color from target string (e.g., "wall:#0000FF" -> ("wall", "#0000FF"))
_COLOR_TARGET_RE = re.compile(r'^(.+?):(#[0-9A-Fa-f]{6})$')


class PersonaPlexVoiceAgent:
    """Voice agent using PersonaPlex for speech-to-speech interaction.

    Drop-in replacement for VoiceAgent. Same interface for the orchestrator.
    """

    def __init__(self, bridge: PersonaPlexBridge, config: ServerConfig, planner=None):
        self._bridge = bridge
        self._config = config
        self._planner = planner  # VoiceCommandPlanner (Gemini) for typed text commands

        # State registries (same as VoiceAgent)
        self._known_objects: set[str] = set()
        self._active_effects: dict[str, dict] = {}
        self._full_screen_filter: Optional[dict] = None
        self._state_lock = threading.Lock()

        # Conversation state for protobuf
        self._conversation_state: dict = {
            "listening": True,
            "recording": False,
            "partial_transcript": "",
            "last_command": "",
            "last_response": "PersonaPlex ready. Say 'Hey Vibe' to start.",
            "last_command_time": 0.0,
        }

        # Callback for SAM3 prompt sync
        self._on_command_callback = None

        # Cached frame for potential future use
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Text accumulation for command parsing
        self._accumulated_text = ""
        self._text_lock = threading.Lock()

        # Wire up bridge callbacks
        self._bridge._on_text = self._on_text_token
        self._bridge._on_command = None  # We parse commands ourselves from text

    def start(self):
        """Start the PersonaPlex bridge and initialize Gemini planner."""
        self._bridge.start()
        if self._planner is not None:
            self._planner.initialize()
        logger.info("PersonaPlexVoiceAgent started")

    def ingest_audio(self, pcm16_data: bytes, sample_rate: int, num_samples: int):
        """Feed audio to PersonaPlex via the bridge."""
        self._bridge.send_audio(pcm16_data, sample_rate)

        # Update state based on bridge readiness
        if self._bridge.is_ready():
            self._conversation_state["listening"] = True
        else:
            self._conversation_state["listening"] = False

    def get_conversation_state(self) -> dict:
        """Get conversation state for protobuf."""
        return self._conversation_state.copy()

    def get_active_effects(self) -> dict[str, dict]:
        """Get current effect state."""
        with self._state_lock:
            return self._active_effects.copy()

    def get_full_screen_filter(self) -> Optional[dict]:
        """Get full-screen filter state."""
        with self._state_lock:
            return self._full_screen_filter.copy() if self._full_screen_filter else None

    def get_known_objects(self) -> set[str]:
        """Get set of known object labels."""
        with self._state_lock:
            return self._known_objects.copy()

    def get_audio_response(self) -> Optional[bytes]:
        """Get buffered audio response from PersonaPlex (PCM16 16kHz).

        Returns None if no audio available. Clears the buffer.
        """
        return self._bridge.get_audio_response()

    def set_on_command_callback(self, callback):
        """Set callback for SAM3 prompt sync.

        Signature: callback(action, targets, effect_type, intensity, invert)
        """
        self._on_command_callback = callback

    def clear_all_effects(self):
        """Clear all active effects and full-screen filters (called on new client connection)."""
        with self._state_lock:
            self._active_effects.clear()
            self._full_screen_filter = None
            self._known_objects.clear()
            logger.info("PersonaPlexVoiceAgent: Cleared all effects, filters, and known objects")

    def add_known_object(self, label: str):
        """Register a detected object label."""
        with self._state_lock:
            self._known_objects.add(label)

    def process_text_command(self, text: str):
        """Process a typed text command through Gemini NLP (same as VoiceAgent).

        PersonaPlex handles speech-to-speech, but typed commands go through
        VoiceCommandPlanner (Gemini) for proper natural language understanding.
        Falls back to regex parsing if Gemini is unavailable.
        """
        text = text.strip()
        if not text:
            return

        logger.info(f"Text command: {text}")

        # 1. If text already contains [COMMAND:...] tags, parse those directly
        plans = self._parse_text_for_commands(text)
        if plans:
            for plan in plans:
                self._execute_plan(plan)
            return

        # 2. Route through Gemini planner (same pipeline as VoiceAgent)
        if self._planner is not None:
            if not self._planner.available:
                self._planner.initialize()

            with self._state_lock:
                known = self._known_objects.copy()
                effects = self._active_effects.copy()

            plan = self._planner.plan_command(text, known, effects)
            logger.info(f"Gemini plan: {plan.action} {plan.effect_type} → {plan.targets} "
                        f"(invert={plan.invert}, reason={plan.reasoning})")
            self._execute_plan(plan)
            return

        # 3. Fallback: no planner available — basic regex
        logger.warning("No Gemini planner available — using basic regex fallback")
        parts = text.lower().split(None, 1)
        if len(parts) == 2 and parts[0] in _COMMAND_MAP:
            effect_type, action, invert = _COMMAND_MAP[parts[0]]
            self._execute_plan(CommandPlan(
                targets=[parts[1]], effect_type=effect_type,
                intensity=0.9, action=action, invert=invert,
            ))
        elif text.lower() in ("clear", "reset"):
            self._execute_plan(CommandPlan(
                targets=[], effect_type="blur", intensity=0.0, action="remove",
            ))

    def update_frame(self, frame_bgr):
        """Cache latest camera frame."""
        with self._frame_lock:
            self._latest_frame = frame_bgr.copy()

    def shutdown(self):
        """Stop the agent and bridge."""
        self._bridge.shutdown()
        logger.info("PersonaPlexVoiceAgent shut down")

    # ------------------------------------------------------------------
    # Internal: text token handling
    # ------------------------------------------------------------------

    def _on_text_token(self, token: str):
        """Called when PersonaPlex emits a text token."""
        with self._text_lock:
            self._accumulated_text += token

        # Update partial transcript for display
        self._conversation_state["partial_transcript"] = self._accumulated_text.strip()

        # Check for complete command tags in accumulated text
        self._check_for_commands()

    def _check_for_commands(self):
        """Parse accumulated text for [COMMAND:...] tags and execute them."""
        with self._text_lock:
            text = self._accumulated_text

        matches = list(_COMMAND_RE.finditer(text))
        if not matches:
            return

        for match in matches:
            action_name = match.group(1).lower()
            target = match.group(2)
            if target:
                target = target.strip().lower()

            plan = self._command_to_plan(action_name, target)
            if plan:
                self._execute_plan(plan)

                # Strip the command tag from display text
                clean_text = text.replace(match.group(0), "").strip()
                self._conversation_state.update({
                    "last_command": match.group(0),
                    "last_response": clean_text if clean_text else f"Applied {action_name}",
                    "last_command_time": time.time(),
                })

        # Clear accumulated text after processing commands
        with self._text_lock:
            # Remove processed command tags but keep any remaining text
            cleaned = _COMMAND_RE.sub("", self._accumulated_text).strip()
            # Reset accumulator for next utterance (keep partial non-command text)
            if not any(c.isalpha() for c in cleaned):
                self._accumulated_text = ""

    def _command_to_plan(self, action_name: str, target: Optional[str]) -> Optional[CommandPlan]:
        """Convert a parsed command tag to a CommandPlan."""
        # Full-screen filters
        if action_name in _FILTER_COMMANDS:
            filter_type, intensity = _FILTER_COMMANDS[action_name]
            return CommandPlan(
                targets=[],
                effect_type="blur",
                intensity=0.0,
                action="add",
                full_screen_filter=filter_type,
                full_screen_intensity=intensity,
            )

        # Full-screen color filter: filter_color:#HEXCOLOR
        if action_name == "filter_color" and target:
            color_hex = target.strip() if target.startswith("#") else f"#{target.strip()}"
            return CommandPlan(
                targets=[],
                effect_type="color",
                intensity=0.0,
                action="add",
                full_screen_filter="color",
                full_screen_intensity=0.6,
                full_screen_color=color_hex,
            )

        # Object-targeted commands
        if action_name in _COMMAND_MAP:
            effect_type, action, invert = _COMMAND_MAP[action_name]

            # Extract hex color from target if present (e.g., "wall:#0000FF")
            color_hex = None
            if target and effect_type == "color":
                cm = _COLOR_TARGET_RE.match(target)
                if cm:
                    target = cm.group(1).strip()
                    color_hex = cm.group(2)

            targets = [target] if target else []

            return CommandPlan(
                targets=targets,
                effect_type=effect_type,
                intensity=1.0 if effect_type == "color" else 0.9,
                action=action,
                invert=invert,
                color_hex=color_hex,
            )

        logger.warning(f"Unknown PersonaPlex command: {action_name}")
        return None

    def _parse_text_for_commands(self, text: str) -> list[CommandPlan]:
        """Parse text for [COMMAND:...] tags and return CommandPlans."""
        plans = []
        for match in _COMMAND_RE.finditer(text):
            action_name = match.group(1).lower()
            target = match.group(2)
            if target:
                target = target.strip().lower()
            plan = self._command_to_plan(action_name, target)
            if plan:
                plans.append(plan)
        return plans

    def _execute_plan(self, plan: CommandPlan):
        """Execute a command plan: update state registries and notify orchestrator."""
        with self._state_lock:
            if plan.action in ("add", "change"):
                for target in plan.targets:
                    self._known_objects.add(target)
                    self._active_effects[target] = {
                        "type": plan.effect_type,
                        "intensity": plan.intensity,
                        "invert": plan.invert,
                        "color_hex": plan.color_hex,
                    }
                    mode = "inverted" if plan.invert else "direct"
                    color_info = f" {plan.color_hex}" if plan.color_hex else ""
                    logger.info(f"PersonaPlex: applied {plan.effect_type} ({mode}) to {target}{color_info}")

            elif plan.action == "remove":
                if plan.targets:
                    for target in plan.targets:
                        if target in self._active_effects:
                            del self._active_effects[target]
                            logger.info(f"PersonaPlex: removed effect from {target}")
                else:
                    # Clear all
                    self._active_effects.clear()
                    self._full_screen_filter = None
                    logger.info("PersonaPlex: cleared all effects")

            # Full-screen filter
            if plan.full_screen_filter:
                self._full_screen_filter = {
                    "type": plan.full_screen_filter,
                    "intensity": plan.full_screen_intensity,
                    "color_hex": plan.full_screen_color,
                }
                logger.info(f"PersonaPlex: applied full-screen filter: {plan.full_screen_filter}")

        # Notify orchestrator to sync SAM3 prompts
        if self._on_command_callback:
            try:
                self._on_command_callback(
                    plan.action, plan.targets, plan.effect_type,
                    plan.intensity, plan.invert, plan.color_hex,
                )
            except Exception as e:
                logger.error(f"on_command callback error: {e}")
