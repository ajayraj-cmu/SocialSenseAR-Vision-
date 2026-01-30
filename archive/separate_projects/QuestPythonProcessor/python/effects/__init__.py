"""
Visual effects module.

Provides different visual effects (focus/grayscale, blur, etc.)

To add a new effect:
1. Create a new file in this directory (e.g., blur_background.py)
2. Implement a class inheriting from BaseEffect
3. Register it in EFFECTS dict below

Example:
    # In effects/blur_background.py
    from .base import BaseEffect

    class BlurBackgroundEffect(BaseEffect):
        def apply(self, frame, result):
            # Apply gaussian blur to background
            ...

        def apply_transition(self, frame, result, progress):
            # Animated transition
            ...

    # Then add to EFFECTS:
    # "blur": BlurBackgroundEffect,
"""
from typing import Optional
from .base import BaseEffect
from .focus_grayscale import FocusGrayscaleEffect

# Registry of available effects
EFFECTS = {
    "focus": FocusGrayscaleEffect,
    # Add more effects here:
    # "blur": BlurBackgroundEffect,
    # "bokeh": BokehEffect,
}


def get_effect(name: str, config=None) -> BaseEffect:
    """Get an effect instance by name.

    Args:
        name: Effect type name (e.g., "focus", "blur")
        config: Configuration object

    Returns:
        Initialized effect instance

    Raises:
        ValueError: If effect name is not registered
    """
    if name not in EFFECTS:
        available = ", ".join(EFFECTS.keys())
        raise ValueError(f"Unknown effect '{name}'. Available: {available}")

    return EFFECTS[name](config)


def list_effects() -> list:
    """List available effect names."""
    return list(EFFECTS.keys())


__all__ = ['BaseEffect', 'FocusGrayscaleEffect', 'get_effect', 'list_effects']
