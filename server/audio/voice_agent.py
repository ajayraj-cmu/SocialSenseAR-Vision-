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

logger = logging.getLogger(__name__)


@dataclass
class CommandPlan:
    """Structured command plan from Gemini reasoning."""
    targets: list[str]  # object labels to target (SAM3 segments these)
    effect_type: str  # "blur", "dim", "pixelate", etc.
    intensity: float  # 0.0-1.0
    action: str  # "add", "remove", "change"
    invert: bool = False  # True = effect on everything EXCEPT targets (targets are the "safe" objects)
    full_screen_filter: Optional[str] = None  # "dim", "warm", etc.
    full_screen_intensity: float = 0.5
    reasoning: str = ""


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

    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash-lite"):
        self._api_key = api_key
        self._model = model
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

    # --- Fast-path regex patterns (compiled once) ---
    _EFFECTS = r'blur|pixelate|dim|highlight|outline'
    _EFFECTS_GERUND = (
        r'blur(?:ring)?|pixelat(?:e|ing)|dim(?:ming)?|highlight(?:ing)?|outlin(?:e|ing)'
    )
    _FILLER = {'the', 'a', 'an', 'my', 'that', 'this'}
    _SELF_WORDS = {'me', 'myself', 'us'}
    _NOT_TARGETS = {'everything', 'every', 'all', 'anything'}  # incomplete phrases, not real objects

    _RE_CLEAR = re.compile(r'^(?:clear|reset|stop|off)$', re.IGNORECASE)
    _RE_EFFECT_INVERT = re.compile(
        rf'^({_EFFECTS})\s+(?:every\s*thing|everything|all)\s+(?:but|except)\s+(.+)$', re.IGNORECASE,
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

    def _try_fast_parse(self, utterance: str) -> Optional[CommandPlan]:
        """Regex fast-path for obvious commands. Returns None if not confident."""
        text = utterance.strip()

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
                intensity=0.8, action="add", invert=True,
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
                intensity=0.8, action="add",
                reasoning=f"fast-path: {m.group(1)} {target}",
            )

        # Pattern 5: bare target (1-2 words, no keywords) → default to blur
        words = text.lower().split()
        if len(words) <= 2:
            target = self._extract_target(text)
            if target is not None:
                return CommandPlan(
                    targets=[target], effect_type="blur",
                    intensity=0.8, action="add",
                    reasoning=f"fast-path: bare target → blur {target}",
                )

        # Not confident → let Gemini handle it
        return None

    def plan_command(
        self,
        utterance: str,
        known_objects: set[str],
        active_effects: dict[str, dict],
    ) -> CommandPlan:
        """Map natural language utterance to structured command plan.

        Tries regex fast-path first (0ms). Falls back to Gemini for complex commands.
        """
        # Try fast regex parse first
        fast = self._try_fast_parse(utterance)
        if fast:
            logger.info(
                f"Fast-path plan: {fast.action} {fast.effect_type} → "
                f"{fast.targets} (invert={fast.invert})"
            )
            return fast

        if not self._available:
            logger.warning(f"Gemini unavailable, extracting targets from text: {utterance}")
            return self._emergency_parse(utterance)

        try:
            known_str = ', '.join(sorted(known_objects)) if known_objects else 'none'
            active_str = ', '.join(f"{k}:{v['type']}" for k, v in active_effects.items()) if active_effects else 'none'

            prompt = f"""Parse this command into a structured action plan. Respond ONLY with JSON.

User said: "{utterance}"

Context:
- Known objects: {known_str}
- Active effects: {active_str}

Output JSON with these fields:
- "targets": list of object labels. ALWAYS include any object the user names, even if not in the known set. SAM3 will search for it.
- "effect_type": "blur" | "dim" | "pixelate" | "highlight" | "outline"
- "intensity": 0.0-1.0 (default 0.8)
- "action": "add" | "remove" | "change"
- "invert": true if effect applies to everything EXCEPT targets (e.g. "blur everything but laptop")
- "full_screen_filter": "dim" | "warm" | "cool" | "night" | "grayscale" | null
- "full_screen_intensity": 0.0-1.0
- "reasoning": brief explanation

Rules:
- "blur phone" → targets: ["phone"], effect_type: "blur", action: "add"
- "pixelate laptop" → targets: ["laptop"], effect_type: "pixelate", action: "add"
- "stop blurring person" → targets: ["person"], action: "remove"
- "blur everything but me" → targets: ["person"], invert: true
- NEVER return empty targets if the user names an object
- If just an object name with no action ("person"), default to blur

{{
    "targets": ["phone"],
    "effect_type": "pixelate",
    "intensity": 0.8,
    "action": "add",
    "invert": false,
    "full_screen_filter": null,
    "full_screen_intensity": 0.5,
    "reasoning": "User wants to pixelate the phone"
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
                full_screen_filter=data.get("full_screen_filter"),
                full_screen_intensity=float(data.get("full_screen_intensity") or 0.5),
                reasoning=data.get("reasoning") or "",
            )

            logger.info(f"Gemini plan: {plan.action} {plan.effect_type} → {plan.targets} (invert={plan.invert})")
            return plan

        except Exception as e:
            logger.error(f"Gemini planning error: {e}")
            return self._emergency_parse(utterance)

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
            intensity=0.8,
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

    # Audio pipeline tuning
    _SILENCE_RMS_THRESHOLD = 150   # Was 300 — lower to catch quiet speech
    _LISTENING_CHUNK_S = 3.0       # 3s chunks — Whisper needs context for accuracy
    _RECORDING_CHUNK_S = 3.0       # 3s chunks for accurate command transcription

    def ingest_audio(self, pcm16_data: bytes, sample_rate: int, num_samples: int):
        """Called from websocket thread to feed audio data.

        Buffers audio and emits chunks for background transcription.
        Uses 1s chunks during listening (fast wake word) and 2s during recording (accurate).
        No stale chunk dropping — local transcription is cheap.
        """
        with self._buffer_lock:
            self._audio_buffer.extend(pcm16_data)
            self._sample_rate = sample_rate

            # Dynamic chunk size: shorter during listening for fast wake word detection
            chunk_s = self._RECORDING_CHUNK_S if self._assembler.is_active else self._LISTENING_CHUNK_S
            min_bytes = int(sample_rate * chunk_s * 2)  # 2 bytes per PCM16 sample

            # Process all complete chunks (no dropping)
            while len(self._audio_buffer) >= min_bytes:
                chunk_bytes = bytes(self._audio_buffer[:min_bytes])
                self._audio_buffer = self._audio_buffer[min_bytes:]

                # Energy gate: skip silence
                samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
                if rms < self._SILENCE_RMS_THRESHOLD:
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

            # Single-pass: execute immediately ONLY if fast-path is confident
            # e.g. "vibe blur the laptop" → remainder "blur laptop" → fast-path matches
            # but "vibe dim every" → fast-path returns None → record more input
            if remainder and len(remainder.split()) >= 2:
                fast = self._planner._try_fast_parse(remainder)
                if fast:
                    logger.info(f"Single-pass command: \"{remainder}\"")
                    self._conversation_state.update({
                        "listening": False,
                        "recording": True,
                        "partial_transcript": remainder,
                        "last_response": "Processing...",
                    })
                    self._execute_command(remainder)
                    self._conversation_state.update({
                        "listening": True,
                        "recording": False,
                        "partial_transcript": "",
                    })
                    return

            # Otherwise start recording for more input
            self._assembler.start(initial_text=remainder)
            self._conversation_state.update({
                "listening": False,
                "recording": True,
                "partial_transcript": remainder,
                "last_response": "Listening...",
            })
            return

        # --- RECORDING PHASE: assemble utterance ---
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

    def _execute_command(self, utterance: str):
        """Execute command: Gemini plan → update state → apply effects."""
        t0 = time.time()

        # Get current state
        with self._state_lock:
            known_objs = self._known_objects.copy()
            active_fx = self._active_effects.copy()

        # Plan command using Gemini reasoning (single API call — no scene snapshot)
        plan = self._planner.plan_command(
            utterance,
            known_objs,
            active_fx,
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
                        "invert": plan.invert,
                    }
                    mode = "inverted" if plan.invert else "direct"
                    logger.info(f"Applied {plan.effect_type} ({mode}) to {target}")

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

        # Notify orchestrator to sync SAM3 prompts with voice command
        if self._on_command_callback:
            try:
                self._on_command_callback(
                    plan.action, plan.targets, plan.effect_type, plan.intensity, plan.invert,
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
