"""Shared command parsing for prompt control.

Used by both the WebSocket server and the test client to parse
commands like "blur laptop", "unblur person", "clear", etc.

Supports negation:
    "blur everything but laptop"  -> ("blur", "laptop", True)
    "blur all except person"      -> ("blur", "person", True)

Supports intensity modifiers:
    "extremely dark black" -> intensity 1.0
    "very bright red" -> intensity 0.9
    "slightly dim" -> intensity 0.3
"""
import re
import queue
import threading

# Color name to hex code mapping for color overlay effects
COLOR_NAMES_TO_HEX = {
    # Primary colors
    "red": "#FF0000", "green": "#00FF00", "blue": "#0000FF",
    "yellow": "#FFFF00", "cyan": "#00FFFF", "magenta": "#FF00FF",
    "white": "#FFFFFF", "black": "#000000", "gray": "#808080", "grey": "#808080",

    # Extended palette
    "orange": "#FFA500", "purple": "#800080", "pink": "#FFC0CB",
    "brown": "#A52A2A", "tan": "#D2B48C", "beige": "#F5F5DC",
    "lime": "#00FF00", "teal": "#008080", "navy": "#000080",
    "maroon": "#800000", "olive": "#808000", "coral": "#FF7F50",
    "gold": "#FFD700", "silver": "#C0C0C0", "bronze": "#CD7F32",

    # Light variants
    "light blue": "#ADD8E6", "light green": "#90EE90", "light red": "#FFB6C1",
    "light yellow": "#FFFFE0", "light gray": "#D3D3D3",

    # Dark variants
    "dark blue": "#00008B", "dark green": "#006400", "dark red": "#8B0000",
    "dark gray": "#A9A9A9",

    # Common aliases
    "azure": "#007FFF", "violet": "#8A2BE2", "indigo": "#4B0082",
    "turquoise": "#40E0D0", "emerald": "#50C878", "crimson": "#DC143C",
}

# Intensity modifiers: map words to intensity multipliers (0.0-1.0)
INTENSITY_MODIFIERS = {
    # Strong (high intensity)
    "extremely": 1.0, "super": 1.0, "maximum": 1.0, "max": 1.0,
    "very": 0.9, "really": 0.9, "highly": 0.9, "heavily": 0.9,
    "quite": 0.8, "pretty": 0.8,

    # Moderate (default range)
    "moderately": 0.6, "somewhat": 0.6,
    "a bit": 0.4, "a little": 0.4,

    # Weak (low intensity)
    "slightly": 0.3, "barely": 0.2, "minimally": 0.2,
    "subtly": 0.25, "gently": 0.3,
}

# Brightness modifiers for colors (adjusts RGB values)
# Tuned to preserve color hue: "very dark blue" should still look BLUE, not black
BRIGHTNESS_MODIFIERS = {
    # Darken — multipliers preserve hue while reducing luminance
    "pitch black": 0.0, "jet black": 0.0,
    "extremely dark": 0.25, "very dark": 0.40, "dark": 0.55,
    "deep": 0.65,

    # Lighten — blend toward white
    "extremely light": 0.95, "very light": 0.85, "light": 0.75,
    "bright": 0.80, "very bright": 0.90, "extremely bright": 1.0,

    # Pale/muted — reduced saturation
    "pale": 0.70, "pastel": 0.75, "muted": 0.60,
    "soft": 0.70, "faded": 0.60, "vivid": 1.0, "rich": 0.90,
}

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
    # "background" is handled specially - means "everything but person"
    "people": "person", "human": "person", "man": "person", "woman": "person",
    "hands": "hand", "arms": "arm", "legs": "leg",
    "furniture": "chair",  # best effort
}


def extract_intensity_modifier(text: str) -> tuple[float | None, str]:
    """Extract intensity modifier from text and return (intensity, remaining_text).

    Returns:
        (intensity, text_without_modifier) if modifier found, else (None, text)
    """
    text_lower = text.lower()

    # Check for multi-word modifiers first (e.g., "a bit", "a little")
    sorted_modifiers = sorted(INTENSITY_MODIFIERS.keys(), key=len, reverse=True)

    for modifier in sorted_modifiers:
        pattern = rf'\b{re.escape(modifier)}\b'
        if re.search(pattern, text_lower):
            cleaned = re.sub(pattern, ' ', text_lower, count=1).strip()
            cleaned = ' '.join(cleaned.split())
            return INTENSITY_MODIFIERS[modifier], cleaned

    return None, text


def adjust_color_brightness(hex_color: str, brightness: float) -> str:
    """Adjust a hex color's brightness.

    Args:
        hex_color: Hex color string (e.g., "#FF0000")
        brightness: Multiplier 0.0-1.0 (0.0=black, 0.5=half brightness, 1.0=full)

    Returns:
        Adjusted hex color string
    """
    if not hex_color or not hex_color.startswith("#"):
        return hex_color

    # Parse hex to RGB
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"#{hex_color}"

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Apply brightness multiplier
    r = int(r * brightness)
    g = int(g * brightness)
    b = int(b * brightness)

    # Clamp to 0-255
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f"#{r:02X}{g:02X}{b:02X}"


def extract_brightness_modifier(text: str) -> tuple[float | None, str]:
    """Extract brightness modifier from text (e.g., 'dark', 'bright', 'pale').

    Returns:
        (brightness_multiplier, text_without_modifier) if found, else (None, text)
    """
    text_lower = text.lower()

    # Check for brightness modifiers (prioritize longer matches)
    sorted_brightness = sorted(BRIGHTNESS_MODIFIERS.keys(), key=len, reverse=True)

    for modifier in sorted_brightness:
        pattern = rf'\b{re.escape(modifier)}\b'
        if re.search(pattern, text_lower):
            cleaned = re.sub(pattern, ' ', text_lower, count=1).strip()
            cleaned = ' '.join(cleaned.split())
            return BRIGHTNESS_MODIFIERS[modifier], cleaned

    return None, text


def extract_color_from_text(text: str) -> tuple[str | None, str, float | None]:
    """Extract color name with optional brightness modifier.

    Returns:
        (color_name, remaining_text, brightness_multiplier)
        If no brightness modifier, brightness_multiplier is None.
    """
    text_lower = text.lower()

    # First, check for brightness modifiers
    brightness, text_after_brightness = extract_brightness_modifier(text_lower)

    # Then check for color names in the modified text
    # Sort by length descending to match "light blue" before "blue"
    sorted_colors = sorted(COLOR_NAMES_TO_HEX.keys(), key=len, reverse=True)

    for color_name in sorted_colors:
        # Pattern: "make X [color]" or "[color] X" or "X [color]"
        pattern = rf'\b{re.escape(color_name)}\b'
        search_text = text_after_brightness if brightness else text_lower

        if re.search(pattern, search_text):
            # Remove color word from text
            cleaned = re.sub(pattern, ' ', search_text, count=1).strip()
            # Normalize whitespace
            cleaned = ' '.join(cleaned.split())
            return color_name, cleaned, brightness

    return None, text, None


def parse_command(text: str) -> tuple[str, str | None, bool, str | None, float | None]:
    """Parse a command string into (action, target, invert, color_hex, intensity).

    Returns:
        (action, target, invert, color_hex, intensity) where:
        - invert=True means effect applies to everything EXCEPT the target mask
        - color_hex is the adjusted hex code if a color was detected
        - intensity is the extracted intensity (0.0-1.0) or None for default

    Supports:
        "blur face"                        -> ("blur", "face", False, None, None)
        "very blur person"                 -> ("blur", "person", False, None, 0.9)
        "extremely dark black ceiling"     -> ("color", "ceiling", False, "#000000" (darkened), 1.0)
        "make person bright blue"          -> ("color", "person", False, "#0000FF" (brightened), None)
        "blur everything but laptop"       -> ("blur", "laptop", True, None, None)
        "slightly dim wall"                -> ("dim", "wall", False, None, 0.3)
        "clear"                            -> ("clear", None, False, None, None)
    """
    text = text.strip().lower()
    if not text:
        return ("", None, False, None, None)

    # Single-word commands
    if text in ("clear", "reset", "clearall"):
        return ("clear", None, False, None, None)
    if text in ("list", "ls", "show", "status"):
        return ("list", None, False, None, None)
    if text in ("help", "?", "commands"):
        return ("help", None, False, None, None)

    # Extract intensity modifier first
    intensity, text_after_intensity = extract_intensity_modifier(text)

    # Check for color requests with brightness modifiers
    color_name, text_without_color, brightness = extract_color_from_text(text_after_intensity)

    # Apply brightness adjustment if specified
    color_hex = None
    if color_name:
        base_hex = COLOR_NAMES_TO_HEX.get(color_name)
        if base_hex and brightness is not None:
            color_hex = adjust_color_brightness(base_hex, brightness)
        else:
            color_hex = base_hex

    # Use text without color for further parsing
    parse_text = text_without_color if color_name else text_after_intensity

    # Detect negation: "blur everything but X", "dim all except X", "blur everything around X"
    invert = False
    negation_match = re.search(
        r'(?:everything|all)\s+(?:but|except|around|other\s+than|besides)\s+(?:the\s+)?(\S+)',
        parse_text,
    )
    if negation_match:
        invert = True
        target_word = negation_match.group(1)
        # Detect action from the beginning
        action = "blur"  # default
        if parse_text.startswith(("dim", "darken")):
            action = "dim"
        elif parse_text.startswith("pixelate"):
            action = "pixelate"
        elif parse_text.startswith("highlight"):
            action = "highlight"
        elif color_hex or parse_text.startswith(("color", "paint", "make")):
            action = "color"
        # "me" / "myself" → person
        if target_word in ("me", "myself", "us"):
            target_word = "person"
        target_word = LABEL_ALIASES.get(target_word, target_word)
        return (action, target_word, True, color_hex, intensity)

    words = parse_text.split()

    # Two-word: "blur face" or "face blur" or "color laptop" / "make person"
    action = None
    target = None
    for w in words:
        if w in ("blur", "blr"):  # handle typos (removed "blue" since it's a color)
            action = "blur"
        elif w in ("dim", "darken"):
            action = "dim"
        elif w in ("pixelate",):
            action = "pixelate"
        elif w in ("highlight",):
            action = "highlight"
        elif w in ("color", "paint", "make"):
            action = "color" if color_hex else "blur"  # only "color" if we have a color
        elif w in ("unblur", "unblr", "remove", "stop", "off"):
            action = "unblur"
        else:
            target = w

    if action is None:
        # Default: "color" if color detected, else "blur"
        action = "color" if color_hex else "blur"
        target = parse_text.split()[0] if parse_text.split() else None

    if target:
        # Special case: "background" means "everything except person" (inverted)
        if target == "background":
            return (action, "person", True, color_hex, intensity)
        # Resolve aliases
        target = LABEL_ALIASES.get(target, target)

    return (action, target, invert, color_hex, intensity)


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
