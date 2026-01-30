"""
Social Cue Detector - Rule-based detection of social cues for neurodivergent users.

Detects patterns in speech that may indicate:
- Sarcasm
- Rhetorical questions
- Conversation ending signals
- Topic changes
- Polite disagreement
- Disengagement
- And more...
"""
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class CueType(Enum):
    """Types of social cues we can detect."""
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
    """A detected social cue."""
    cue_type: CueType
    message: str
    icon: str
    confidence: float
    timestamp: float
    source_text: str


class SocialCueDetector:
    """Detects social cues from transcribed speech."""

    # Sarcasm patterns - positive words that may be sarcastic
    SARCASM_POSITIVE_WORDS = [
        "great", "wonderful", "fantastic", "amazing", "brilliant",
        "perfect", "lovely", "awesome", "terrific", "marvelous"
    ]
    SARCASM_PHRASES = [
        r"oh,?\s*(great|wonderful|fantastic|perfect)",
        r"yeah,?\s*right",
        r"sure,?\s*(thing|whatever)",
        r"as if",
        r"how\s+nice",
        r"that'?s?\s+just\s+(great|wonderful|perfect)",
        r"well,?\s+isn'?t\s+that\s+(nice|special|something)",
    ]

    # Rhetorical question patterns
    RHETORICAL_PATTERNS = [
        r"who\s+(even\s+)?(does|cares|knows)\s+that",
        r"what\s+did\s+I\s+(just\s+)?tell\s+you",
        r"how\s+hard\s+(can|is)\s+(it|that)",
        r"are\s+you\s+(even\s+)?(listening|kidding|serious)",
        r"do\s+I\s+(really\s+)?(have\s+to|need\s+to|look\s+like)",
        r"why\s+would\s+(I|anyone)",
        r"can\s+you\s+believe",
    ]

    # Conversation ending signals - tightened to avoid false positives
    ENDING_PHRASES = [
        # Removed standalone "anyway" - too common mid-conversation
        r"anyway,?\s+(I\s+should|I\s+gotta|I\s+need\s+to)",
        r"I\s+should\s+(probably\s+)?(go|get\s+going|head\s+out|run|leave)",
        r"(good|nice)\s+talking\s+to\s+you",
        r"take\s+care\s*$",
        r"catch\s+(you|ya)\s+later",
        r"(I\s+)?gotta\s+(go|run|head\s+out)",
        r"I('?ll|\s+will)\s+let\s+you\s+go",
        r"we('?ll|\s+will)\s+talk\s+(later|soon)",
        r"see\s+you\s+(later|around|soon)",
        # Removed standalone "well/okay/alright" - too common in normal speech
    ]

    # Topic change signals
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

    # Polite disagreement patterns
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

    # Uncertainty/hedging patterns - tightened to require more context
    # Avoid matching casual uses like "maybe later" or "I don't know his name"
    UNCERTAINTY_PHRASES = [
        r"I\s+guess\s+(so|not|that)",
        r"I\s+think\s+maybe",
        # Only match "kind of" / "sort of" when used as hedging, not as descriptors
        r"it'?s\s+(kind\s+of|kinda|sort\s+of|sorta)\s+(hard|difficult|complicated|unclear)",
        r"I\s+(don'?t|do\s+not)\s+(really\s+)?know\s*(if|what|how|why|whether)",
        r"probably\s+not",
        r"might\s+not\s+be",
        r"I'?m\s+not\s+(really\s+)?sure\s+(if|about|what|how|why)",
        r"could\s+be\s+(wrong|right|either)",
        r"^maybe\s+I\s+(should|shouldn'?t|could|can'?t)",
    ]

    # Frustration signals - tightened patterns to reduce false positives
    # Each pattern should be specific enough to avoid matching innocent phrases
    FRUSTRATION_PHRASES = [
        # "look" only when followed by frustrated context (not "look at this!")
        r"^look,?\s+(I|you|we|this\s+is|it'?s\s+not)",
        r"I\s+already\s+(said|told|explained)",
        r"for\s+the\s+(last|third|fourth|fifth)\s+time",
        r"how\s+many\s+times\s+(do|have|did|must)",
        r"I'?ve\s+(already\s+)?(said|told|explained)\s+(you|this)",
        r"as\s+I\s+(already\s+)?(said|mentioned|explained)\s+(before|earlier|already)",
        r"(ugh|argh|jeez)(?:\s|$|!)",
        # Only match at end of sentence with preceding content
        r".{10,}(seriously|honestly)\s*[!.]*$",
        r".{5,}come\s+on\s*[!.]*$",
    ]

    # Words that strengthen frustration detection when combined
    FRUSTRATION_INTENSIFIERS = [
        "already", "again", "still", "always", "never", "stop",
        "enough", "why", "can't", "won't", "don't"
    ]

    # Passive aggression patterns - tightened to reduce false positives
    PASSIVE_AGGRESSION_PHRASES = [
        r"^fine\.?\s*$",
        r"whatever\s+you\s+say",
        r"I'?m\s+not\s+(even\s+)?mad",
        r"(it'?s|that'?s)\s+fine,?\s*really",
        r"do\s+whatever\s+you\s+want",
        r"if\s+that'?s\s+what\s+you\s+(really\s+)?think",
        # Removed standalone "sure" and "okay, then" - too common in normal speech
        r"^no,?\s+it'?s\s+(okay|fine)\s*$",
        r"I\s+said\s+(it'?s|I'?m)\s+fine",
    ]

    # Humor/joke signals
    HUMOR_PHRASES = [
        r"just\s+kidding",
        r"I'?m\s+(just\s+)?joking",
        r"(haha|hehe|lol|lmao)",
        r"I'?m\s+(just\s+)?messing\s+with\s+you",
        r"gotcha",
        r"pulling\s+your\s+leg",
    ]

    # Validation seeking - tightened to avoid matching casual speech
    VALIDATION_PHRASES = [
        # Removed standalone "right?" and "you know?" - too common as filler
        r"don'?t\s+you\s+think\s*\?",
        r"wouldn'?t\s+you\s+(say|agree)\s*\?",
        r"isn'?t\s+(it|that)\s+(right|true)\s*\?",
        r"(am|are)\s+I\s+(right|wrong)\s*(here|about\s+this)?\s*\?",
        r"makes\s+sense,?\s*right\s*\?",
        r"you\s+agree\s*,?\s*right\s*\?",
    ]

    # Disengagement signals (short responses)
    SHORT_RESPONSES = [
        "uh huh", "mm hmm", "mmm", "yeah", "yep", "yup",
        "okay", "ok", "sure", "cool", "right", "interesting"
    ]

    # Minimum confidence threshold for showing cues
    MIN_CONFIDENCE_THRESHOLD = 0.7

    # Cooldown period in seconds - don't show same cue type within this window
    CUE_COOLDOWN_SECONDS = 10.0

    def __init__(self):
        """Initialize the social cue detector."""
        # Recent cues for deduplication
        self.recent_cues: deque = deque(maxlen=10)

        # Cooldown tracking: maps CueType -> last trigger timestamp
        self._last_cue_time: dict = {}

        # Compile regex patterns for performance
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

        # Compile frustration intensifiers for quick lookup
        self._frustration_intensifiers = set(word.lower() for word in self.FRUSTRATION_INTENSIFIERS)

        # Cue display info
        self._cue_info = {
            CueType.SARCASM: ("⚠️", "May be sarcastic"),
            CueType.RHETORICAL: ("ℹ️", "No answer expected"),
            CueType.ENDING: ("🚪", "Wrapping up soon"),
            CueType.TOPIC_CHANGE: ("🔄", "New topic"),
            CueType.DISAGREEMENT: ("⚡", "May disagree politely"),
            CueType.UNCERTAINTY: ("🤔", "They seem unsure"),
            CueType.FRUSTRATION: ("😤", "May be frustrated"),
            CueType.PASSIVE_AGGRESSION: ("⚠️", "May not mean it"),
            CueType.HUMOR: ("😄", "Probably joking"),
            CueType.VALIDATION_SEEKING: ("👍", "They want agreement"),
            CueType.DISENGAGEMENT: ("😐", "May be losing interest"),
            CueType.WAIT_YOUR_TURN: ("🛑", "Wait - they're still talking"),
        }

    def detect(self, text: str, emotion: Optional[str] = None,
               is_other_speaking: bool = False) -> List[SocialCue]:
        """Detect social cues in transcribed text.

        Args:
            text: Transcribed text to analyze
            emotion: Current detected emotion (optional, for combined analysis)
            is_other_speaking: Whether the other person is currently speaking

        Returns:
            List of detected social cues (filtered by confidence and cooldown)
        """
        detected = []
        text_lower = text.lower().strip()
        now = time.time()

        # Check each pattern type
        for cue_type, patterns in self._compiled_patterns.items():
            # Skip disengagement detection entirely - disabled
            if cue_type == CueType.DISENGAGEMENT:
                continue

            # Check cooldown - skip if we recently showed this cue type
            if self._is_on_cooldown(cue_type, now):
                continue

            for pattern in patterns:
                if pattern.search(text_lower):
                    # Special handling for frustration - require more context
                    if cue_type == CueType.FRUSTRATION:
                        confidence = self._calculate_frustration_confidence(text_lower, emotion)
                        if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                            continue
                    else:
                        confidence = 0.8

                    icon, message = self._cue_info[cue_type]

                    # Adjust message based on emotion context
                    message = self._adjust_message_with_emotion(cue_type, message, emotion)

                    cue = SocialCue(
                        cue_type=cue_type,
                        message=message,
                        icon=icon,
                        confidence=confidence,
                        timestamp=now,
                        source_text=text[:50]
                    )
                    detected.append(cue)
                    # Update cooldown tracker
                    self._last_cue_time[cue_type] = now
                    break  # Only report each type once per text

        # DISENGAGEMENT detection disabled - "May be losing interest" popup removed
        # The short response detection was causing false positives
        # if text_lower in self.SHORT_RESPONSES or len(text.split()) <= 2:
        #     # Only flag if we've seen multiple short responses recently
        #     self.recent_cues.append(("short", now))
        #     short_count = sum(1 for cue, ts in self.recent_cues
        #                       if cue == "short" and now - ts < 30)
        #     if short_count >= 3 and not self._is_on_cooldown(CueType.DISENGAGEMENT, now):
        #         icon, message = self._cue_info[CueType.DISENGAGEMENT]
        #         cue = SocialCue(
        #             cue_type=CueType.DISENGAGEMENT,
        #             message=message,
        #             icon=icon,
        #             confidence=0.7,
        #             timestamp=now,
        #             source_text=text[:50]
        #         )
        #         detected.append(cue)
        #         self._last_cue_time[CueType.DISENGAGEMENT] = now

        # Combined detection: words don't match emotion
        emotion_cues = self._detect_word_emotion_mismatch(text, emotion, now)
        for cue in emotion_cues:
            if not self._is_on_cooldown(cue.cue_type, now):
                detected.append(cue)
                self._last_cue_time[cue.cue_type] = now

        # Filter by minimum confidence threshold
        detected = [cue for cue in detected if cue.confidence >= self.MIN_CONFIDENCE_THRESHOLD]

        return detected

    def _is_on_cooldown(self, cue_type: CueType, now: float) -> bool:
        """Check if a cue type is on cooldown (recently shown)."""
        last_time = self._last_cue_time.get(cue_type, 0)
        return (now - last_time) < self.CUE_COOLDOWN_SECONDS

    def _calculate_frustration_confidence(self, text_lower: str, emotion: Optional[str]) -> float:
        """Calculate confidence for frustration detection.

        Requires either:
        - Negative emotion detected, OR
        - Multiple frustration indicators in the text

        Single word matches without supporting context get low confidence.
        """
        confidence = 0.5  # Base confidence for pattern match

        # Check for negative emotion - strong indicator
        if emotion and emotion.lower() in ['angry', 'disgust', 'sad', 'fear']:
            confidence += 0.3

        # Count frustration intensifiers in the text
        words = set(text_lower.split())
        intensifier_count = len(words & self._frustration_intensifiers)

        # Add confidence based on intensifier count
        if intensifier_count >= 2:
            confidence += 0.25
        elif intensifier_count >= 1:
            confidence += 0.1

        # Check for exclamation marks (frustration indicator)
        if '!' in text_lower:
            confidence += 0.1

        # Longer utterances with frustration patterns are more reliable
        word_count = len(text_lower.split())
        if word_count >= 8:
            confidence += 0.1

        return min(confidence, 1.0)

    def _adjust_message_with_emotion(self, cue_type: CueType, message: str,
                                      emotion: Optional[str]) -> str:
        """Adjust cue message based on emotion context."""
        if not emotion:
            return message

        emotion = emotion.lower()

        # Sarcasm is more likely with negative emotions
        if cue_type == CueType.SARCASM and emotion in ['angry', 'disgust', 'sad']:
            return "Likely sarcastic (expression doesn't match)"

        # Passive aggression with angry/sad face
        if cue_type == CueType.PASSIVE_AGGRESSION and emotion in ['angry', 'sad']:
            return "Words don't match expression"

        return message

    def _detect_word_emotion_mismatch(self, text: str, emotion: Optional[str],
                                       now: float) -> List[SocialCue]:
        """Detect when words don't match facial expression."""
        if not emotion:
            return []

        emotion = emotion.lower()
        text_lower = text.lower()
        detected = []

        # Positive words with negative expression
        positive_indicators = any(word in text_lower for word in
                                  ['great', 'good', 'fine', 'happy', 'love', 'wonderful', 'amazing'])
        negative_emotions = ['angry', 'disgust', 'sad', 'fear']

        if positive_indicators and emotion in negative_emotions:
            detected.append(SocialCue(
                cue_type=CueType.SARCASM,
                message="Words don't match expression",
                icon="⚠️",
                confidence=0.9,
                timestamp=now,
                source_text=text[:50]
            ))

        # "I'm fine" / "It's okay" with sad/angry expression
        fine_phrases = ['i\'m fine', 'im fine', 'it\'s fine', 'its fine', 'it\'s okay', 'its okay', 'i\'m okay', 'im okay']
        if any(phrase in text_lower for phrase in fine_phrases) and emotion in ['sad', 'angry']:
            detected.append(SocialCue(
                cue_type=CueType.PASSIVE_AGGRESSION,
                message="They may not be fine",
                icon="⚠️",
                confidence=0.85,
                timestamp=now,
                source_text=text[:50]
            ))

        return detected

    def create_interruption_warning(self) -> SocialCue:
        """Create a warning cue when user might be interrupting."""
        icon, message = self._cue_info[CueType.WAIT_YOUR_TURN]
        return SocialCue(
            cue_type=CueType.WAIT_YOUR_TURN,
            message=message,
            icon=icon,
            confidence=1.0,
            timestamp=time.time(),
            source_text=""
        )


# Global detector instance
_detector: Optional[SocialCueDetector] = None


def get_detector() -> SocialCueDetector:
    """Get or create the global social cue detector."""
    global _detector
    if _detector is None:
        _detector = SocialCueDetector()
    return _detector


def detect_social_cues(text: str, emotion: Optional[str] = None) -> List[SocialCue]:
    """Convenience function to detect social cues."""
    detector = get_detector()
    return detector.detect(text, emotion)
