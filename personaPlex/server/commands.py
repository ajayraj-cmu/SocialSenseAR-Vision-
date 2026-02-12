"""Shared command parsing for prompt control.

Used by both the WebSocket server and the test client to parse
commands like "blur laptop", "unblur person", "clear", etc.

Supports negation:
    "blur everything but laptop"  -> ("blur", "laptop", True)
    "blur all except person"      -> ("blur", "person", True)
"""
import re
import queue
import threading

# Common object labels (suggestions for help text).
# SAM3 accepts ANY text prompt — this list is not a restriction.
KNOWN_LABELS = {
    "person", "face", "chair", "table", "desk", "couch", "monitor",
    "laptop", "lamp", "wall", "floor", "door", "window",
    "head", "hair", "torso", "body", "shirt", "arm", "hand",
    "shoulder", "skin", "pants", "leg", "human",
    "bottle", "cup", "mug", "phone", "keyboard", "mouse",
    "book", "bag", "backpack", "shoe", "hat", "glasses",
    "painting", "poster", "picture", "clock", "plant", "flower",
    "pillow", "blanket", "curtain", "rug", "shelf", "cabinet",
    "tv", "speaker", "headphones", "microphone", "camera",
    "pen", "paper", "box", "toy", "food", "plate", "bowl",
    "car", "bicycle", "dog", "cat", "bird",
}

# Aliases: map common words to SAM3 labels
LABEL_ALIASES = {
    "computer": "monitor", "screen": "monitor", "display": "monitor",
    "sofa": "couch", "settee": "couch",
    "light": "lamp", "ceiling light": "lamp",
    "ground": "floor", "carpet": "floor",
    "background": "wall",  # blur wall = blur background
    "people": "person", "human": "person", "man": "person", "woman": "person",
    "hands": "hand", "arms": "arm", "legs": "leg",
    "furniture": "chair",  # best effort
}


def parse_command(text: str) -> tuple[str, str | None, bool]:
    """Parse a command string into (action, target, invert).

    Returns:
        (action, target, invert) where invert=True means effect applies to
        everything EXCEPT the target mask.

    Supports:
        "blur face"                    -> ("blur", "face", False)
        "face blur"                    -> ("blur", "face", False)
        "unblur person"                -> ("unblur", "person", False)
        "blur everything but laptop"   -> ("blur", "laptop", True)
        "blur all except person"       -> ("blur", "person", True)
        "dim everything around me"     -> ("dim", "person", True)
        "clear"                        -> ("clear", None, False)
        "list"                         -> ("list", None, False)
        "help"                         -> ("help", None, False)
    """
    text = text.strip().lower()
    if not text:
        return ("", None, False)

    # Single-word commands
    if text in ("clear", "reset", "clearall"):
        return ("clear", None, False)
    if text in ("list", "ls", "show", "status"):
        return ("list", None, False)
    if text in ("help", "?", "commands"):
        return ("help", None, False)

    # Detect negation: "blur everything but X", "dim all except X", "blur everything around X"
    invert = False
    negation_match = re.search(
        r'(?:everything|all)\s+(?:but|except|around|other\s+than|besides)\s+(?:the\s+)?(\S+)',
        text,
    )
    if negation_match:
        invert = True
        target_word = negation_match.group(1)
        # Detect action from the beginning
        action = "blur"  # default
        if text.startswith(("dim", "darken")):
            action = "dim"
        elif text.startswith("pixelate"):
            action = "pixelate"
        elif text.startswith("highlight"):
            action = "highlight"
        # "me" / "myself" → person
        if target_word in ("me", "myself", "us"):
            target_word = "person"
        target_word = LABEL_ALIASES.get(target_word, target_word)
        return (action, target_word, True)

    words = text.split()

    # Two-word: "blur face" or "face blur"
    action = None
    target = None
    for w in words:
        if w in ("blur", "blr", "blue"):  # handle typos
            action = "blur"
        elif w in ("dim", "darken"):
            action = "dim"
        elif w in ("pixelate",):
            action = "pixelate"
        elif w in ("highlight",):
            action = "highlight"
        elif w in ("unblur", "unblr", "remove", "stop", "off"):
            action = "unblur"
        else:
            target = w

    if action is None:
        # Default: "blur" if just a label name
        action = "blur"
        target = text.split()[0]

    if target:
        # Resolve aliases
        target = LABEL_ALIASES.get(target, target)

    return (action, target, invert)


class CommandInput:
    """Background thread that reads console commands (stdin)."""

    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                line = input()
                if line.strip():
                    self._queue.put(line.strip())
            except EOFError:
                break
            except Exception:
                break

    def get_commands(self) -> list[str]:
        """Non-blocking: return all queued commands."""
        cmds = []
        while not self._queue.empty():
            try:
                cmds.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def stop(self):
        self._running = False
