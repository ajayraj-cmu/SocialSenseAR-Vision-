"""Voice Agent Pipeline — natural language control of SAM3 + effects.

Architecture:
1. WakeWordGate: Detects "Hey Vibe" to start recording
2. UtteranceAssembler: Collects speech until "Thank you"
3. VoiceCommandPlanner: Uses Gemini to map intent → structured command plan

Design:
- Audio processing runs asynchronously (does not block frame loop)
- Commands execute atomically after full utterance is collected
- Maintains persistent "known objects" registry + active effects state
- Uses Gemini Vision on-demand for scene understanding
"""

import time
import logging
import threading
import queue
import re
from typing import Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CommandPlan:
    """Structured command plan from Gemini reasoning."""
    targets: list[str]  # object labels to target
    effect_type: str  # "blur", "dim", "pixelate", etc.
    intensity: float  # 0.0-1.0
    action: str  # "add", "remove", "change"
    full_screen_filter: Optional[str] = None  # "dim", "warm", etc.
    full_screen_intensity: float = 0.5
    reasoning: str = ""


class WakeWordGate:
    """Detects wake word ("Hey Vibe") in transcript stream.

    Tolerates:
    - Case variations
    - Punctuation
    - Common transcription errors (Hey Vibes, Hey Vive, etc.)
    """

    WAKE_PATTERNS = [
        r'\bhey\s+vibe\b',
        r'\bhey\s+vibes\b',
        r'\bhey\s+vive\b',
        r'\bhey\s+five\b',  # common misrecognition
    ]

    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.WAKE_PATTERNS]

    def detect(self, text: str) -> bool:
        """Check if text contains wake word."""
        return any(pattern.search(text) for pattern in self._patterns)

    def strip_wake_word(self, text: str) -> str:
        """Remove wake word from beginning of text."""
        for pattern in self._patterns:
            text = pattern.sub('', text, count=1)
        return text.strip()


class UtteranceAssembler:
    """Assembles full utterance from wake word to end phrase ("Thank you").

    Collects transcript chunks and detects end phrase.
    """

    END_PATTERNS = [
        r'\bthank\s+you\b',
        r'\bthanks\b',
        r'\bthank\s+ya\b',
    ]

    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.END_PATTERNS]
        self._active = False
        self._chunks: list[str] = []

    def start(self):
        """Start collecting utterance (called after wake word detected)."""
        self._active = True
        self._chunks.clear()

    def add_chunk(self, text: str) -> Optional[str]:
        """Add transcript chunk. Returns complete utterance if end phrase detected."""
        if not self._active:
            return None

        self._chunks.append(text)

        # Check for end phrase
        full_text = ' '.join(self._chunks)
        for pattern in self._patterns:
            if pattern.search(full_text):
                # Found end phrase — return utterance without end phrase
                utterance = pattern.sub('', full_text, count=1).strip()
                self.reset()
                return utterance

        return None

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

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash-exp"):
        self._api_key = api_key
        self._model = model
        self._client = None
        self._available = False

    def initialize(self):
        """Initialize Gemini API client."""
        import os
        api_key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key — voice planning disabled")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self._model)
            self._available = True
            logger.info(f"Voice command planner ready (model={self._model})")
        except Exception as e:
            logger.error(f"Voice planner init failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def plan_command(
        self,
        utterance: str,
        known_objects: set[str],
        active_effects: dict[str, dict],
        visible_objects: list[str] = None,
        scene_context: str = "",
    ) -> CommandPlan:
        """Map natural language utterance to structured command plan.

        Args:
            utterance: User's command text (wake word and end phrase already stripped)
            known_objects: Set of object labels we've seen before
            active_effects: Current effect state {label: {"type": "blur", "intensity": 0.8}}
            visible_objects: Objects currently visible (from Gemini Vision or SAM3)
            scene_context: Scene description from Gemini Vision

        Returns:
            CommandPlan with targets, effect, action, reasoning
        """
        if not self._available:
            # Fallback: simple keyword matching
            return self._fallback_plan(utterance, known_objects, active_effects)

        try:
            # Build context string
            known_str = ', '.join(sorted(known_objects)) if known_objects else 'none'
            visible_str = ', '.join(visible_objects) if visible_objects else 'unknown'
            active_str = ', '.join(f"{k}:{v['type']}" for k, v in active_effects.items()) if active_effects else 'none'

            prompt = f"""Parse this voice command and create a structured action plan.

User said: "{utterance}"

Context:
- Known objects (seen before): {known_str}
- Currently visible: {visible_str}
- Active effects: {active_str}
- Scene: {scene_context or 'Unknown'}

Determine:
1. **targets**: Which specific object labels to target (from known or visible objects)
2. **effect_type**: "blur", "dim", "pixelate", "highlight", "outline", or "none"
3. **intensity**: 0.0 (light) to 1.0 (strong)
4. **action**: "add" (apply new effect), "remove" (clear effect), or "change" (modify existing)
5. **full_screen_filter**: If request is global/environmental ("dim everything"), use "dim"/"warm"/"cool"/"night"/"grayscale", else null
6. **full_screen_intensity**: 0.0-1.0 for full-screen filter
7. **reasoning**: Brief explanation

Rules:
- If object named explicitly ("blur laptop"), use that object
- If object is in known set but not visible, still target it (persistent effects)
- If request is vague ("too bright"), infer appropriate targets (lights, screens, windows)
- For "stop X" or "unblur X", action is "remove"
- For "dim it" or "it's too bright", apply full_screen_filter="dim" + target bright objects
- If no objects match, return empty targets

Format as JSON:
{{
    "targets": ["laptop", "monitor"],
    "effect_type": "blur",
    "intensity": 0.8,
    "action": "add",
    "full_screen_filter": null,
    "full_screen_intensity": 0.5,
    "reasoning": "User wants to blur screens"
}}"""

            response = self._client.generate_content(prompt)
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
                effect_type=data.get("effect_type", "none"),
                intensity=float(data.get("intensity", 1.0)),
                action=data.get("action", "add"),
                full_screen_filter=data.get("full_screen_filter"),
                full_screen_intensity=float(data.get("full_screen_intensity", 0.5)),
                reasoning=data.get("reasoning", ""),
            )

            logger.info(f"Command plan: {plan.action} {plan.effect_type} to {plan.targets}")
            logger.debug(f"Reasoning: {plan.reasoning}")

            return plan

        except Exception as e:
            logger.error(f"Gemini planning error: {e}", exc_info=True)
            return self._fallback_plan(utterance, known_objects, active_effects)

    def _fallback_plan(
        self,
        utterance: str,
        known_objects: set[str],
        active_effects: dict[str, dict],
    ) -> CommandPlan:
        """Simple keyword-based planning when Gemini unavailable."""
        text_lower = utterance.lower()

        # Detect effect type
        effect = "blur"  # default
        if any(w in text_lower for w in ["dim", "darken", "darker"]):
            effect = "dim"
        elif "highlight" in text_lower:
            effect = "highlight"
        elif "pixelate" in text_lower:
            effect = "pixelate"

        # Detect action
        action = "add"
        if any(w in text_lower for w in ["stop", "remove", "unblur", "undim", "clear"]):
            action = "remove"

        # Extract targets from known objects
        targets = []
        for obj in known_objects:
            if obj.lower() in text_lower:
                targets.append(obj)

        # If no targets but "everything" or "background" mentioned
        full_screen = None
        if any(w in text_lower for w in ["everything", "all", "background", "too bright", "overstimulating"]):
            full_screen = "dim"
            # Also target common environmental objects if known
            env_keywords = {"wall", "floor", "ceiling", "light", "lamp", "window", "screen", "monitor"}
            targets.extend([obj for obj in known_objects if obj.lower() in env_keywords])

        return CommandPlan(
            targets=targets,
            effect_type=effect,
            intensity=0.7,
            action=action,
            full_screen_filter=full_screen,
            full_screen_intensity=0.5,
            reasoning="Fallback: keyword-based parsing (Gemini unavailable)",
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

        self._wake_gate = WakeWordGate()
        self._assembler = UtteranceAssembler()

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
            "last_response": "Ready. Say 'Hey Vibe' to start.",
            "last_command_time": 0.0,
        }

        # Processing queue (audio transcription + command planning happen on background thread)
        self._process_queue: queue.Queue = queue.Queue()
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._running = True

        # Cached frame for Gemini Vision
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

    def start(self):
        """Start background processing thread."""
        self._process_thread.start()
        logger.info("VoiceAgent started")

    def ingest_audio(self, pcm16_data: bytes, sample_rate: int, num_samples: int):
        """Called from websocket thread to feed audio data.

        Audio is buffered and processed asynchronously.
        """
        with self._buffer_lock:
            self._audio_buffer.extend(pcm16_data)
            self._sample_rate = sample_rate

            # Trigger transcription when we have enough audio (~2s chunks)
            min_samples = sample_rate * 2
            if len(self._audio_buffer) >= min_samples * 2:  # 2 bytes per PCM16 sample
                # Extract chunk
                chunk_bytes = bytes(self._audio_buffer[:min_samples * 2])
                self._audio_buffer = self._audio_buffer[min_samples * 2:]

                # Queue for async processing
                self._process_queue.put(("transcribe", chunk_bytes, sample_rate))

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

    def add_known_object(self, label: str):
        """Add object to known registry (called when SAM3 segments it)."""
        with self._state_lock:
            self._known_objects.add(label)

    def _process_loop(self):
        """Background thread: transcribe audio + execute commands."""
        while self._running:
            try:
                item = self._process_queue.get(timeout=0.5)
                if item is None:
                    break

                cmd_type = item[0]

                if cmd_type == "transcribe":
                    _, audio_bytes, sr = item
                    self._handle_transcription(audio_bytes, sr)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Voice agent process loop error: {e}", exc_info=True)

    def _handle_transcription(self, audio_bytes: bytes, sample_rate: int):
        """Transcribe audio chunk and process for wake word / command assembly."""
        if not self._transcriber.available:
            return

        text = self._transcriber.transcribe(audio_bytes, sample_rate)
        if not text:
            return

        logger.debug(f"Transcript: \"{text}\"")

        # Check for wake word
        if not self._assembler.is_active and self._wake_gate.detect(text):
            logger.info("Wake word detected!")
            self._assembler.start()
            self._conversation_state.update({
                "listening": False,
                "recording": True,
                "partial_transcript": "",
                "last_response": "Listening...",
            })
            return

        # Assemble utterance
        if self._assembler.is_active:
            complete_utterance = self._assembler.add_chunk(text)

            # Update partial transcript for dashboard
            if complete_utterance is None:
                self._conversation_state["partial_transcript"] = self._assembler.get_partial()
            else:
                # Complete command received
                logger.info(f"Complete utterance: \"{complete_utterance}\"")
                self._execute_command(complete_utterance)
                self._conversation_state.update({
                    "listening": True,
                    "recording": False,
                    "partial_transcript": "",
                })

    def _execute_command(self, utterance: str):
        """Execute complete voice command: plan → update state → apply effects."""
        t0 = time.time()

        # Get current state
        with self._state_lock:
            known_objs = self._known_objects.copy()
            active_fx = self._active_effects.copy()

        # Get visible objects from Gemini Vision (on-demand)
        visible_objects = []
        scene_context = ""
        with self._frame_lock:
            if self._latest_frame is not None and self._scene.available:
                snapshot = self._scene.snapshot(self._latest_frame)
                visible_objects = snapshot.get("objects", [])
                scene_context = snapshot.get("scene_description", "")

        # Plan command using Gemini reasoning
        plan = self._planner.plan_command(
            utterance,
            known_objs,
            active_fx,
            visible_objects,
            scene_context,
        )

        # Execute plan: update state
        with self._state_lock:
            if plan.action == "add" or plan.action == "change":
                # Add/update effects
                for target in plan.targets:
                    self._known_objects.add(target)
                    self._active_effects[target] = {
                        "type": plan.effect_type,
                        "intensity": plan.intensity,
                    }
                    logger.info(f"Applied {plan.effect_type} to {target}")

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
                }
                logger.info(f"Applied full-screen filter: {plan.full_screen_filter}")
            elif plan.action == "remove" and not plan.targets:
                # "Clear everything" command
                self._full_screen_filter = None
                self._active_effects.clear()
                logger.info("Cleared all effects")

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
        self._process_queue.put(None)
        if self._process_thread.is_alive():
            self._process_thread.join(timeout=2)
        logger.info("VoiceAgent shut down")
