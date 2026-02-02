"""Shared command parsing for prompt control.

Used by both the WebSocket server and the test client to parse
commands like "blur laptop", "unblur person", "clear", etc.
"""
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


def parse_command(text: str) -> tuple[str, str | None]:
    """Parse a command string into (action, target).

    Supports:
        "blur face"      -> ("blur", "face")
        "face blur"      -> ("blur", "face")
        "unblur person"  -> ("unblur", "person")
        "person unblur"  -> ("unblur", "person")
        "clear"          -> ("clear", None)
        "list"           -> ("list", None)
        "help"           -> ("help", None)
    """
    text = text.strip().lower()
    if not text:
        return ("", None)

    # Single-word commands
    if text in ("clear", "reset", "clearall"):
        return ("clear", None)
    if text in ("list", "ls", "show", "status"):
        return ("list", None)
    if text in ("help", "?", "commands"):
        return ("help", None)

    words = text.split()

    # Two-word: "blur face" or "face blur"
    action = None
    target = None
    for w in words:
        if w in ("blur", "blr", "blue"):  # handle typos
            action = "blur"
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

    return (action, target)


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
