"""Social Cue Detector — rule-based detection for neurodivergent users.

Ported from QuestPythonProcessor/python/audio/social_cue_detector.py.
Pure text analysis — no hardware dependencies.

Detects: sarcasm, rhetorical questions, conversation ending signals,
topic changes, polite disagreement, frustration, passive aggression,
humor, validation seeking, word-emotion mismatch.
"""

import re
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class CueType(Enum):
    SARCASM = "sarcasm"
    RHETORICAL = "rhetorical"
    ENDING = "ending"
    TOPIC_CHANGE = "topic_change"
    DISAGREEMENT = "disagreement"
    UNCERTAINTY = "uncertainty"
    FRUSTRATION = "frustration"
    PASSIVE_AGGRESSION = "passive_aggression"
    HUMOR = "humor"
    VALIDATION_SEEKING = "validation_seeking"
    DISENGAGEMENT = "disengagement"
    WAIT_YOUR_TURN = "wait_turn"


@dataclass
class SocialCue:
    cue_type: CueType
    message: str
    icon: str
    confidence: float
    timestamp: float
    source_text: str


class SocialCueDetector:
    """Detects social cues from transcribed speech."""

    # --- Pattern libraries (from original) ---

    SARCASM_PHRASES = [
        r"oh,?\s*(great|wonderful|fantastic|perfect)",
        r"yeah,?\s*right",
        r"sure,?\s*(thing|whatever)",
        r"as if",
        r"how\s+nice",
        r"that'?s?\s+just\s+(great|wonderful|perfect)",
        r"well,?\s+isn'?t\s+that\s+(nice|special|something)",
    ]

    RHETORICAL_PATTERNS = [
        r"who\s+(even\s+)?(does|cares|knows)\s+that",
        r"what\s+did\s+I\s+(just\s+)?tell\s+you",
        r"how\s+hard\s+(can|is)\s+(it|that)",
        r"are\s+you\s+(even\s+)?(listening|kidding|serious)",
        r"do\s+I\s+(really\s+)?(have\s+to|need\s+to|look\s+like)",
        r"why\s+would\s+(I|anyone)",
        r"can\s+you\s+believe",
    ]

    ENDING_PHRASES = [
        r"anyway,?\s+(I\s+should|I\s+gotta|I\s+need\s+to)",
        r"I\s+should\s+(probably\s+)?(go|get\s+going|head\s+out|run|leave)",
        r"(good|nice)\s+talking\s+to\s+you",
        r"take\s+care\s*$",
        r"catch\s+(you|ya)\s+later",
        r"(I\s+)?gotta\s+(go|run|head\s+out)",
        r"I('?ll|\s+will)\s+let\s+you\s+go",
        r"we('?ll|\s+will)\s+talk\s+(later|soon)",
        r"see\s+you\s+(later|around|soon)",
    ]

    TOPIC_CHANGE_PHRASES = [
        r"speaking\s+of",
        r"by\s+the\s+way",
        r"that\s+reminds\s+me",
        r"oh,?\s+(also|and)",
        r"anyway,?\s+so",
        r"(moving|getting)\s+on",
        r"on\s+another\s+note",
        r"while\s+(we'?re|I'm)\s+(at\s+it|here)",
        r"changing\s+(the\s+)?subject",
    ]

    DISAGREEMENT_PHRASES = [
        r"I\s+see\s+what\s+you\s+mean,?\s*but",
        r"that'?s?\s+interesting,?\s*(but|however)",
        r"with\s+(all\s+due\s+)?respect",
        r"I\s+(hear|understand)\s+you,?\s*but",
        r"not\s+to\s+disagree,?\s*but",
        r"I\s+(kind\s+of|kinda)\s+disagree",
        r"actually,?\s*I\s+think",
        r"well,?\s*not\s+(exactly|really|quite)",
        r"I'?m\s+not\s+(so\s+)?sure\s+(about\s+that|I\s+agree)",
    ]

    UNCERTAINTY_PHRASES = [
        r"I\s+guess\s+(so|not|that)",
        r"I\s+think\s+maybe",
        r"it'?s\s+(kind\s+of|kinda|sort\s+of|sorta)\s+(hard|difficult|complicated|unclear)",
        r"I\s+(don'?t|do\s+not)\s+(really\s+)?know\s*(if|what|how|why|whether)",
        r"probably\s+not",
        r"might\s+not\s+be",
        r"I'?m\s+not\s+(really\s+)?sure\s+(if|about|what|how|why)",
        r"could\s+be\s+(wrong|right|either)",
        r"^maybe\s+I\s+(should|shouldn'?t|could|can'?t)",
    ]

    FRUSTRATION_PHRASES = [
        r"^look,?\s+(I|you|we|this\s+is|it'?s\s+not)",
        r"I\s+already\s+(said|told|explained)",
        r"for\s+the\s+(last|third|fourth|fifth)\s+time",
        r"how\s+many\s+times\s+(do|have|did|must)",
        r"I'?ve\s+(already\s+)?(said|told|explained)\s+(you|this)",
        r"as\s+I\s+(already\s+)?(said|mentioned|explained)\s+(before|earlier|already)",
        r"(ugh|argh|jeez)(?:\s|$|!)",
        r".{10,}(seriously|honestly)\s*[!.]*$",
        r".{5,}come\s+on\s*[!.]*$",
    ]

    FRUSTRATION_INTENSIFIERS = {
        "already", "again", "still", "always", "never", "stop",
        "enough", "why", "can't", "won't", "don't",
    }

    PASSIVE_AGGRESSION_PHRASES = [
        r"^fine\.?\s*$",
        r"whatever\s+you\s+say",
        r"I'?m\s+not\s+(even\s+)?mad",
        r"(it'?s|that'?s)\s+fine,?\s*really",
        r"do\s+whatever\s+you\s+want",
        r"if\s+that'?s\s+what\s+you\s+(really\s+)?think",
        r"^no,?\s+it'?s\s+(okay|fine)\s*$",
        r"I\s+said\s+(it'?s|I'?m)\s+fine",
    ]

    HUMOR_PHRASES = [
        r"just\s+kidding",
        r"I'?m\s+(just\s+)?joking",
        r"(haha|hehe|lol|lmao)",
        r"I'?m\s+(just\s+)?messing\s+with\s+you",
        r"gotcha",
        r"pulling\s+your\s+leg",
    ]

    VALIDATION_PHRASES = [
        r"don'?t\s+you\s+think\s*\?",
        r"wouldn'?t\s+you\s+(say|agree)\s*\?",
        r"isn'?t\s+(it|that)\s+(right|true)\s*\?",
        r"(am|are)\s+I\s+(right|wrong)\s*(here|about\s+this)?\s*\?",
        r"makes\s+sense,?\s*right\s*\?",
        r"you\s+agree\s*,?\s*right\s*\?",
    ]

    MIN_CONFIDENCE_THRESHOLD = 0.7
    CUE_COOLDOWN_SECONDS = 10.0

    # --- Cue display info ---
    _CUE_INFO = {
        CueType.SARCASM: ("!", "May be sarcastic"),
        CueType.RHETORICAL: ("i", "No answer expected"),
        CueType.ENDING: ("->", "Wrapping up soon"),
        CueType.TOPIC_CHANGE: ("<>", "New topic"),
        CueType.DISAGREEMENT: ("*", "May disagree politely"),
        CueType.UNCERTAINTY: ("?", "They seem unsure"),
        CueType.FRUSTRATION: ("!!", "May be frustrated"),
        CueType.PASSIVE_AGGRESSION: ("!", "May not mean it"),
        CueType.HUMOR: (":)", "Probably joking"),
        CueType.VALIDATION_SEEKING: ("+", "They want agreement"),
        CueType.DISENGAGEMENT: ("-", "May be losing interest"),
        CueType.WAIT_YOUR_TURN: ("X", "Wait - they're still talking"),
    }

    def __init__(self):
        self.recent_cues: deque = deque(maxlen=10)
        self._last_cue_time: dict = {}

        # Pre-compile all regex patterns
        self._compiled_patterns = {
            CueType.SARCASM: [re.compile(p, re.IGNORECASE) for p in self.SARCASM_PHRASES],
            CueType.RHETORICAL: [re.compile(p, re.IGNORECASE) for p in self.RHETORICAL_PATTERNS],
            CueType.ENDING: [re.compile(p, re.IGNORECASE) for p in self.ENDING_PHRASES],
            CueType.TOPIC_CHANGE: [re.compile(p, re.IGNORECASE) for p in self.TOPIC_CHANGE_PHRASES],
            CueType.DISAGREEMENT: [re.compile(p, re.IGNORECASE) for p in self.DISAGREEMENT_PHRASES],
            CueType.UNCERTAINTY: [re.compile(p, re.IGNORECASE) for p in self.UNCERTAINTY_PHRASES],
            CueType.FRUSTRATION: [re.compile(p, re.IGNORECASE) for p in self.FRUSTRATION_PHRASES],
            CueType.PASSIVE_AGGRESSION: [re.compile(p, re.IGNORECASE) for p in self.PASSIVE_AGGRESSION_PHRASES],
            CueType.HUMOR: [re.compile(p, re.IGNORECASE) for p in self.HUMOR_PHRASES],
            CueType.VALIDATION_SEEKING: [re.compile(p, re.IGNORECASE) for p in self.VALIDATION_PHRASES],
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect(self, text: str, emotion: Optional[str] = None) -> List[SocialCue]:
        """Detect social cues in transcribed text.

        Args:
            text: Transcribed text to analyze.
            emotion: Current detected emotion (optional).

        Returns:
            List of detected social cues above confidence threshold.
        """
        detected = []
        text_lower = text.lower().strip()
        now = time.time()

        for cue_type, patterns in self._compiled_patterns.items():
            if self._is_on_cooldown(cue_type, now):
                continue

            for pattern in patterns:
                if pattern.search(text_lower):
                    if cue_type == CueType.FRUSTRATION:
                        confidence = self._frustration_confidence(text_lower, emotion)
                        if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                            continue
                    else:
                        confidence = 0.8

                    icon, message = self._CUE_INFO[cue_type]
                    message = self._adjust_message(cue_type, message, emotion)

                    detected.append(SocialCue(
                        cue_type=cue_type,
                        message=message,
                        icon=icon,
                        confidence=confidence,
                        timestamp=now,
                        source_text=text[:50],
                    ))
                    self._last_cue_time[cue_type] = now
                    break  # one per cue type

        # Word-emotion mismatch detection
        for cue in self._word_emotion_mismatch(text, emotion, now):
            if not self._is_on_cooldown(cue.cue_type, now):
                detected.append(cue)
                self._last_cue_time[cue.cue_type] = now

        return [c for c in detected if c.confidence >= self.MIN_CONFIDENCE_THRESHOLD]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, cue_type: CueType, now: float) -> bool:
        return (now - self._last_cue_time.get(cue_type, 0)) < self.CUE_COOLDOWN_SECONDS

    def _frustration_confidence(self, text_lower: str, emotion: Optional[str]) -> float:
        confidence = 0.5
        if emotion and emotion.lower() in ("angry", "disgust", "sad", "fear"):
            confidence += 0.3
        words = set(text_lower.split())
        n = len(words & self.FRUSTRATION_INTENSIFIERS)
        if n >= 2:
            confidence += 0.25
        elif n >= 1:
            confidence += 0.1
        if "!" in text_lower:
            confidence += 0.1
        if len(text_lower.split()) >= 8:
            confidence += 0.1
        return min(confidence, 1.0)

    def _adjust_message(self, cue_type: CueType, message: str, emotion: Optional[str]) -> str:
        if not emotion:
            return message
        e = emotion.lower()
        if cue_type == CueType.SARCASM and e in ("angry", "disgust", "sad"):
            return "Likely sarcastic (expression doesn't match)"
        if cue_type == CueType.PASSIVE_AGGRESSION and e in ("angry", "sad"):
            return "Words don't match expression"
        return message

    def _word_emotion_mismatch(self, text: str, emotion: Optional[str], now: float) -> List[SocialCue]:
        if not emotion:
            return []
        e = emotion.lower()
        tl = text.lower()
        out: list[SocialCue] = []

        positive = any(w in tl for w in ("great", "good", "fine", "happy", "love", "wonderful", "amazing"))
        if positive and e in ("angry", "disgust", "sad", "fear"):
            out.append(SocialCue(CueType.SARCASM, "Words don't match expression", "!", 0.9, now, text[:50]))

        fine_phrases = ("i'm fine", "im fine", "it's fine", "its fine", "it's okay", "its okay", "i'm okay", "im okay")
        if any(p in tl for p in fine_phrases) and e in ("sad", "angry"):
            out.append(SocialCue(CueType.PASSIVE_AGGRESSION, "They may not be fine", "!", 0.85, now, text[:50]))

        return out


# ------------------------------------------------------------------
# Module-level convenience
# ------------------------------------------------------------------

_detector: Optional[SocialCueDetector] = None


def get_detector() -> SocialCueDetector:
    global _detector
    if _detector is None:
        _detector = SocialCueDetector()
    return _detector


def detect_social_cues(text: str, emotion: Optional[str] = None) -> List[SocialCue]:
    return get_detector().detect(text, emotion)
