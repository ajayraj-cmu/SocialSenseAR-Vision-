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


PERSONAPLEX_SYSTEM_PROMPT = """You are Vibe, an AR assistant built into a mixed reality headset. You help users control visual effects on objects they can see.

You ONLY respond when the user addresses you by saying "Hey Vibe", "Vibe", or similar. If they are not talking to you, stay completely silent.

When the user requests a visual effect, include a command tag in your response. Available commands:

[COMMAND:blur:target] - Blur an object (e.g., [COMMAND:blur:laptop])
[COMMAND:dim:target] - Dim an object
[COMMAND:pixelate:target] - Pixelate an object
[COMMAND:highlight:target] - Highlight an object
[COMMAND:outline:target] - Outline an object
[COMMAND:clear] - Clear all effects
[COMMAND:remove:target] - Remove effect from specific object
[COMMAND:invert_blur:target] - Blur everything EXCEPT the target
[COMMAND:filter_dim] - Apply full-screen dim filter
[COMMAND:filter_warm] - Apply full-screen warm filter
[COMMAND:filter_cool] - Apply full-screen cool filter
[COMMAND:filter_night] - Apply full-screen night filter

Examples:
- User: "Hey Vibe, blur the laptop" -> "Sure, blurring the laptop. [COMMAND:blur:laptop]"
- User: "Vibe, it's too bright" -> "Let me dim that for you. [COMMAND:filter_dim]"
- User: "Hey Vibe, clear everything" -> "All cleared. [COMMAND:clear]"
- User: "Vibe, blur everything except me" -> "Done, blurring everything but you. [COMMAND:invert_blur:person]"

Keep responses short and natural. Always include the command tag when performing an action."""


# Regex for parsing [COMMAND:action:target] or [COMMAND:action] from PersonaPlex text
_COMMAND_RE = re.compile(r'\[COMMAND:(\w+)(?::([^\]]+))?\]')

# Map PersonaPlex command names to (effect_type, action, invert) tuples
_COMMAND_MAP = {
    "blur": ("blur", "add", False),
    "dim": ("dim", "add", False),
    "pixelate": ("pixelate", "add", False),
    "highlight": ("highlight", "add", False),
    "outline": ("outline", "add", False),
    "clear": ("blur", "remove", False),  # clear all
    "remove": ("blur", "remove", False),  # remove from specific target
    "invert_blur": ("blur", "add", True),
    "invert_dim": ("dim", "add", True),
    "invert_pixelate": ("pixelate", "add", True),
}

_FILTER_COMMANDS = {
    "filter_dim": ("dim", 0.5),
    "filter_warm": ("warm", 0.5),
    "filter_cool": ("cool", 0.5),
    "filter_night": ("night", 0.5),
}


class PersonaPlexVoiceAgent:
    """Voice agent using PersonaPlex for speech-to-speech interaction.

    Drop-in replacement for VoiceAgent. Same interface for the orchestrator.
    """

    def __init__(self, bridge: PersonaPlexBridge, config: ServerConfig):
        self._bridge = bridge
        self._config = config

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
        """Start the PersonaPlex bridge."""
        self._bridge.start()
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

    def add_known_object(self, label: str):
        """Register a detected object label."""
        with self._state_lock:
            self._known_objects.add(label)

    def process_text_command(self, text: str):
        """Process a typed text command (bypass PersonaPlex, use regex parsing)."""
        text = text.strip()
        if not text:
            return

        logger.info(f"Text command (direct): {text}")
        plans = self._parse_text_for_commands(f"[COMMAND:blur:{text}]")
        if not plans:
            # Try to interpret as a raw command like "blur laptop"
            parts = text.lower().split(None, 1)
            if len(parts) == 2 and parts[0] in ("blur", "dim", "pixelate", "highlight", "outline"):
                plans = [CommandPlan(
                    targets=[parts[1]], effect_type=parts[0],
                    intensity=0.8, action="add",
                )]
            elif text.lower() in ("clear", "reset"):
                plans = [CommandPlan(
                    targets=[], effect_type="blur",
                    intensity=0.0, action="remove",
                )]

        for plan in plans:
            self._execute_plan(plan)

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

        # Object-targeted commands
        if action_name in _COMMAND_MAP:
            effect_type, action, invert = _COMMAND_MAP[action_name]
            targets = [target] if target else []

            return CommandPlan(
                targets=targets,
                effect_type=effect_type,
                intensity=0.8,
                action=action,
                invert=invert,
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
                    }
                    mode = "inverted" if plan.invert else "direct"
                    logger.info(f"PersonaPlex: applied {plan.effect_type} ({mode}) to {target}")

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
                }
                logger.info(f"PersonaPlex: applied full-screen filter: {plan.full_screen_filter}")

        # Notify orchestrator to sync SAM3 prompts
        if self._on_command_callback:
            try:
                self._on_command_callback(
                    plan.action, plan.targets, plan.effect_type,
                    plan.intensity, plan.invert,
                )
            except Exception as e:
                logger.error(f"on_command callback error: {e}")
