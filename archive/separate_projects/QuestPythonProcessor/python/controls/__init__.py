"""
Input controls module.

Provides different input control methods (keyboard, Quest gestures, etc.)

To add a new control:
1. Create a new file in this directory
2. Implement a class inheriting from BaseControl
3. Register it in CONTROLS dict below
"""
from typing import List
from .base import BaseControl
from .keyboard import KeyboardControl
from .quest_pinch import QuestPinchControl

# Registry of available controls
CONTROLS = {
    "keyboard": KeyboardControl,
    "quest_pinch": QuestPinchControl,
}


def get_control(name: str, callback=None) -> BaseControl:
    """Get a control instance by name.

    Args:
        name: Control type name
        callback: Callback function for control events

    Returns:
        Initialized control instance
    """
    if name not in CONTROLS:
        available = ", ".join(CONTROLS.keys())
        raise ValueError(f"Unknown control '{name}'. Available: {available}")

    return CONTROLS[name](callback)


def get_default_controls(callback=None) -> List[BaseControl]:
    """Get default set of controls.

    Returns:
        List of control instances
    """
    return [
        QuestPinchControl(callback),
    ]


__all__ = ['BaseControl', 'KeyboardControl', 'QuestPinchControl',
           'get_control', 'get_default_controls']
