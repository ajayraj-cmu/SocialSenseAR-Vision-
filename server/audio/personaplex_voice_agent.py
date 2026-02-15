"""PersonaPlex Voice Agent — hybrid architecture.

Whisper handles accurate transcription + command extraction.
PersonaPlex handles spoken audio responses (natural voice).

Pipeline:
  Mic audio → [Whisper] → text → WakeWordGate → UtteranceAssembler → Gemini → command
  Mic audio → [PersonaPlex] → spoken audio response → client speakers
  (mic muted while PersonaPlex is speaking to prevent feedback loop)

Same interface as VoiceAgent so the orchestrator needs minimal changes.
"""

import logging
import queue
import re
import threading
import time
from typing import Optional

import numpy as np

from server.audio.personaplex_bridge import PersonaPlexBridge
from server.audio.voice_agent import CommandPlan, WakeWordGate, UtteranceAssembler
from server.config import ServerConfig

logger = logging.getLogger(__name__)


# System prompt for PersonaPlex — controls spoken response style.
# IMPORTANT: Do NOT use the word "Vibe" here — the model associates it with
# a company name and generates call center greetings ("thanks for calling Vibe Labs").
PERSONAPLEX_SYSTEM_PROMPT = """You are a quiet assistant. Never greet the user. Never say hello. Never say "thanks for calling". Never introduce yourself. Never ask questions. Never offer help. Just listen silently. If someone speaks to you, reply with one or two words only like "OK" or "Got it" or "Done". Say nothing else."""


class PersonaPlexVoiceAgent:
    """Hybrid voice agent: Whisper for commands, PersonaPlex for spoken responses.

    Drop-in replacement for VoiceAgent. Same interface for the orchestrator.
    """

    def __init__(self, bridge: PersonaPlexBridge, config: ServerConfig,
                 planner=None, transcriber=None):
        self._bridge = bridge
        self._config = config
        self._planner = planner  # VoiceCommandPlanner (Gemini)
        self._transcriber = transcriber  # LocalTranscriber (faster-whisper)

        # Command pipeline (same as VoiceAgent)
        self._wake_gate = WakeWordGate(
            window_size=getattr(config, 'voice_wake_window', 3),
        )
        self._assembler = UtteranceAssembler(
            timeout=getattr(config, 'voice_assembler_timeout', 4.5),
        )

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

        # Cached frame
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Audio buffer for Whisper transcription
        self._audio_buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._sample_rate = 16000

        # Audio transcription queue
        self._audio_queue: queue.Queue = queue.Queue()
        self._audio_thread: Optional[threading.Thread] = None
        self._running = True

        # Mic mute: suppress sending audio to PersonaPlex AND Whisper while PersonaPlex speaks
        self._mic_mute_until = 0.0  # timestamp until which mic is muted

        # Don't feed PersonaPlex audio until after first wake word detection.
        # This prevents PersonaPlex from generating a greeting monologue on connect.
        self._wake_word_activated = False

        # Suppress PersonaPlex audio output until first command executes.
        # PersonaPlex always generates a greeting ("Hello, thanks for calling...")
        # the moment it receives audio. We discard that greeting silently.
        self._first_command_executed = False

        # Wire up bridge text callback for logging only
        self._bridge._on_text = self._on_text_token
        self._bridge._on_command = None

    def start(self):
        """Start the PersonaPlex bridge, transcriber, and Gemini planner."""
        self._bridge.start()
        if self._planner is not None:
            self._planner.initialize()
        if self._transcriber is not None:
            self._transcriber.initialize()

        # Start audio processing thread (Whisper transcription + command pipeline)
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._audio_thread.start()

        logger.info("PersonaPlexVoiceAgent started (hybrid: Whisper commands + PersonaPlex voice)")

    def ingest_audio(self, pcm16_data: bytes, sample_rate: int, num_samples: int):
        """Feed audio to both Whisper (for transcription) and PersonaPlex (for response generation)."""
        now = time.time()
        mic_muted = now < self._mic_mute_until

        # Only send to PersonaPlex after first wake word AND when not muted.
        # Before wake word activation, PersonaPlex gets no audio → stays silent.
        if self._wake_word_activated and not mic_muted:
            self._bridge.send_audio(pcm16_data, sample_rate)

        # Buffer for Whisper transcription — but skip during mute window
        # (prevents Whisper from transcribing PersonaPlex's own speaker audio)
        if mic_muted:
            # Drain audio buffer so stale audio doesn't queue up
            with self._buffer_lock:
                self._audio_buffer.clear()
        elif self._transcriber is not None and self._transcriber.available:
            with self._buffer_lock:
                self._audio_buffer.extend(pcm16_data)
                self._sample_rate = sample_rate

                # Dynamic chunk size based on phase
                chunk_s = (
                    self._config.voice_recording_chunk_s if self._assembler.is_active
                    else self._config.voice_listening_chunk_s
                )
                min_bytes = int(sample_rate * chunk_s * 2)  # 2 bytes per PCM16 sample

                while len(self._audio_buffer) >= min_bytes:
                    chunk_bytes = bytes(self._audio_buffer[:min_bytes])
                    self._audio_buffer = self._audio_buffer[min_bytes:]

                    # Energy gate: skip silence
                    samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
                    if rms < self._config.voice_energy_threshold:
                        continue

                    self._audio_queue.put((chunk_bytes, sample_rate))

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
        Also extends mic mute window when audio is being sent to prevent feedback.
        Discards audio until the first command has been executed (suppresses greeting).
        """
        data = self._bridge.get_audio_response()
        if data is None:
            return None

        # Discard PersonaPlex audio until first command executes
        # (suppresses "Hello, thanks for calling Vibe Labs" greeting)
        if not self._first_command_executed:
            return None

        # Extend mic mute: don't send mic audio to PersonaPlex/Whisper while it's speaking
        chunk_duration = len(data) / 2 / 16000  # PCM16 = 2 bytes/sample, 16kHz
        self._mic_mute_until = time.time() + chunk_duration + 0.5
        return data

    def set_on_command_callback(self, callback):
        """Set callback for SAM3 prompt sync.

        Signature: callback(action, targets, effect_type, intensity, invert, color_hex)
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
        """Process a typed text command through Gemini NLP."""
        text = text.strip()
        if not text:
            return

        logger.info(f"Text command: {text}")

        if self._planner is not None:
            if not self._planner.available:
                self._planner.initialize()

            with self._state_lock:
                known = self._known_objects.copy()
                effects = self._active_effects.copy()

            plan = self._planner.plan_command(text, known, effects)
            logger.info(f"Gemini plan: {plan.action} {plan.effect_type} -> {plan.targets} "
                        f"(invert={plan.invert}, reason={plan.reasoning})")
            self._execute_plan(plan)

    def update_frame(self, frame_bgr):
        """Cache latest camera frame."""
        with self._frame_lock:
            self._latest_frame = frame_bgr.copy()

    def shutdown(self):
        """Stop the agent and bridge."""
        self._running = False
        self._bridge.shutdown()
        if self._transcriber is not None:
            self._transcriber.shutdown()
        if self._audio_thread is not None and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2)
        logger.info("PersonaPlexVoiceAgent shut down")

    # ------------------------------------------------------------------
    # Internal: Whisper transcription + command pipeline (same as VoiceAgent)
    # ------------------------------------------------------------------

    def _audio_loop(self):
        """Background thread: transcribe audio via Whisper, detect wake word, assemble commands."""
        while self._running:
            try:
                try:
                    audio_bytes, sr = self._audio_queue.get(timeout=0.5)
                    self._handle_transcription(audio_bytes, sr)
                except queue.Empty:
                    pass

                # Check if assembler timed out
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

    def _handle_transcription(self, audio_bytes: bytes, sample_rate: int):
        """Transcribe audio chunk via Whisper and process for wake word / command assembly."""
        if self._transcriber is None or not self._transcriber.available:
            return

        mode = "recording" if self._assembler.is_active else "listening"
        text = self._transcriber.transcribe(audio_bytes, sample_rate, mode=mode)
        if not text:
            return

        _now = time.monotonic()
        if not hasattr(self, '_last_transcript_log'):
            self._last_transcript_log = 0.0
        if mode == "recording" or _now - self._last_transcript_log >= 5.0:
            logger.info(f"Whisper [{mode}]: \"{text}\"")
            self._last_transcript_log = _now

        # --- LISTENING PHASE: check sliding window for wake word ---
        if not self._assembler.is_active:
            detected, remainder = self._wake_gate.push_and_check(text)
            if not detected:
                return

            logger.info(f"Wake word detected! remainder: \"{remainder}\"")

            # Activate PersonaPlex audio feed on first wake word
            if not self._wake_word_activated:
                self._wake_word_activated = True
                logger.info("PersonaPlex activated — now receiving mic audio")

            # "vibe everything but X" → user means "blur everything but X"
            if remainder and re.match(r'(?:every\s*thing|everything|all)\s+(?:but|except)\s', remainder, re.IGNORECASE):
                remainder = f"blur {remainder}"

            self._assembler.start(initial_text=remainder)
            self._conversation_state.update({
                "listening": False,
                "recording": True,
                "partial_transcript": remainder,
                "last_response": "Listening...",
            })
            return

        # --- RECORDING PHASE: check for wake word as command boundary ---
        wake_hit, wake_remainder = self._wake_gate.push_and_check(text)
        if wake_hit:
            current = self._assembler.get_partial()
            self._assembler.reset()
            if current.strip():
                logger.info(f"Wake boundary — firing current: \"{current}\"")
                self._execute_command(current.strip())

            if wake_remainder and re.match(
                r'(?:every\s*thing|everything|all)\s+(?:but|except)\s',
                wake_remainder, re.IGNORECASE,
            ):
                wake_remainder = f"blur {wake_remainder}"

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

    # Clear commands that bypass Gemini
    _RE_INSTANT_CLEAR = re.compile(
        r'^(?:clear|reset|stop|off|normal|remove\s+(?:all|everything|filter(?:s)?))$',
        re.IGNORECASE,
    )
    _LAST_INTENT_SPLIT = re.compile(r'\bhey\s+vi(?:be)?s?\b', re.IGNORECASE)

    def _execute_command(self, utterance: str):
        """Execute command via Gemini VoiceCommandPlanner (same pipeline as VoiceAgent)."""
        # Normalize Whisper artifacts
        utterance = re.sub(r'[.,!?;:]+', ' ', utterance).strip()
        utterance = re.sub(r'\s+', ' ', utterance)
        utterance = re.sub(r'\bnothing\s+but\b', 'but', utterance, flags=re.IGNORECASE)

        # Long run-on: use only text after the last "Hey Vibe"
        if len(utterance) > 80:
            parts = self._LAST_INTENT_SPLIT.split(utterance)
            if len(parts) > 1 and parts[-1].strip():
                utterance = parts[-1].strip()

        t0 = time.time()

        # Instant clear/reset
        if self._RE_INSTANT_CLEAR.match(utterance):
            plan = CommandPlan(
                targets=[], effect_type="none", intensity=0.0,
                action="remove", reasoning="instant: clear all",
            )
            self._apply_plan(plan, utterance, t0)
            return

        # Route through Gemini planner
        if self._planner is not None:
            if not self._planner.available:
                self._planner.initialize()

            with self._state_lock:
                known = self._known_objects.copy()
                effects = self._active_effects.copy()

            try:
                plan = self._planner.plan_command(utterance, known, effects)
                logger.info(f"Gemini plan: {plan.action} {plan.effect_type} -> {plan.targets} "
                            f"(invert={plan.invert}, reason={plan.reasoning})")

                if plan.needs_clarification:
                    self._conversation_state.update({
                        "last_command": utterance,
                        "last_response": plan.clarification_question or "Can you be more specific?",
                        "last_command_time": time.time(),
                    })
                    return

                self._apply_plan(plan, utterance, t0)
            except Exception as e:
                logger.error(f"Gemini plan error for '{utterance}': {e}")
        else:
            logger.warning(f"No planner available for: '{utterance}'")

    def _apply_plan(self, plan: CommandPlan, utterance: str, t0: float):
        """Apply a resolved CommandPlan: update state registries and notify orchestrator."""
        # Normalize "background" → inverted person
        if plan.targets:
            normalized = []
            mapped_bg = False
            for t in plan.targets:
                tl = (t or "").strip().lower()
                if tl == "background":
                    mapped_bg = True
                    normalized.append("person")
                elif tl:
                    normalized.append(tl)
            if normalized:
                plan.targets = list(dict.fromkeys(normalized))
            if mapped_bg and plan.action in ("add", "change"):
                plan.invert = True

        with self._state_lock:
            if plan.action in ("add", "change"):
                self._active_effects.pop("background", None)
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
                    logger.info(f"Applied {plan.effect_type} ({mode}) to {target}{color_info}")

            elif plan.action == "remove":
                if plan.targets:
                    for target in plan.targets:
                        if target in self._active_effects:
                            del self._active_effects[target]
                            logger.info(f"Removed effect from {target}")
                else:
                    self._active_effects.clear()
                    self._full_screen_filter = None
                    logger.info("Cleared all effects")

            if plan.full_screen_filter:
                if plan.full_screen_filter == "none":
                    self._full_screen_filter = None
                else:
                    self._full_screen_filter = {
                        "type": plan.full_screen_filter,
                        "intensity": plan.full_screen_intensity,
                        "color_hex": plan.full_screen_color,
                    }
                    logger.info(f"Applied full-screen filter: {plan.full_screen_filter}")
            elif plan.action == "remove" and not plan.targets:
                self._full_screen_filter = None
                self._active_effects.clear()

        # Allow PersonaPlex audio to flow to client now that a real command executed
        if not self._first_command_executed:
            self._first_command_executed = True
            logger.info("First command executed — PersonaPlex audio output enabled")

        # Notify orchestrator
        if self._on_command_callback:
            try:
                self._on_command_callback(
                    plan.action, plan.targets, plan.effect_type,
                    plan.intensity, plan.invert, plan.color_hex,
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
        """Generate response message."""
        if plan.action == "add":
            if plan.targets:
                objs = ", ".join(plan.targets)
                if plan.invert:
                    return f"Applied {plan.effect_type} to everything except {objs}"
                return f"Applied {plan.effect_type} to {objs}"
            elif plan.full_screen_filter:
                return f"Applied {plan.full_screen_filter} filter"
            return "No matching objects found"
        elif plan.action == "remove":
            if plan.targets:
                return f"Removed effects from {', '.join(plan.targets)}"
            return "Cleared all effects"
        elif plan.action == "change":
            return f"Changed {', '.join(plan.targets)} to {plan.effect_type}"
        return "Done"

    # ------------------------------------------------------------------
    # Internal: PersonaPlex text tokens (logging only — no command extraction)
    # ------------------------------------------------------------------

    def _on_text_token(self, token: str):
        """Called when PersonaPlex emits a text token. Logged for diagnostics only."""
        # No command extraction from PersonaPlex text — Whisper handles that
        pass
