"""
Transition effect manager.

Handles smooth animated transitions when toggling effects on/off.
The transition animates from 0 (off) to 1 (fully on) over a configurable duration.
"""
import time


def smooth_ease(t: float) -> float:
    """Smooth ease-out cubic curve.

    Args:
        t: Progress value from 0.0 to 1.0

    Returns:
        Eased value from 0.0 to 1.0
    """
    if t >= 1.0:
        return 1.0
    if t <= 0.0:
        return 0.0
    return 1.0 - (1.0 - t) ** 3


class TransitionEffect:
    """Manages animated transitions for visual effects.

    The transition smoothly animates the effect on/off over time.
    During transition, `progress` goes from 0.0 (off) to 1.0 (on).

    Usage:
        transition = TransitionEffect(duration=1.0)

        # Toggle effect
        if transition.toggle():
            print("Transitioning...")

        # In render loop
        transition.update()
        if transition.is_active():
            effect.apply_transition(frame, result, transition.progress)

    Attributes:
        active: Whether effect is on (after transition completes)
        transitioning: Whether currently animating
        progress: Current transition progress (0.0 to 1.0)
        duration: Transition duration in seconds
    """

    def __init__(self, duration: float = 1.0):
        """Initialize transition manager.

        Args:
            duration: Time in seconds for full transition
        """
        self.active = False
        self.transitioning = False
        self.direction = 1  # 1 = turning on, -1 = turning off
        self.progress = 0.0  # 0.0 = off, 1.0 = fully on
        self.start_time = 0.0
        self.duration = duration
        self.person_center = (0.5, 0.5)  # normalized center position

    def toggle(self) -> bool:
        """Toggle the effect with animation.

        Ignores input while already transitioning (debounce).

        Returns:
            True if toggle was accepted, False if ignored (still transitioning)
        """
        if self.transitioning:
            return False

        self.transitioning = True
        self.start_time = time.time()

        if self.active:
            self.direction = -1  # turning off
        else:
            self.direction = 1  # turning on
            self.active = True

        return True

    def update(self) -> None:
        """Update transition progress. Call each frame."""
        if not self.transitioning:
            return

        elapsed = time.time() - self.start_time
        raw_progress = min(elapsed / self.duration, 1.0)

        if self.direction == 1:
            self.progress = smooth_ease(raw_progress)
        else:
            self.progress = 1.0 - smooth_ease(raw_progress)

        if raw_progress >= 1.0:
            self.transitioning = False
            if self.direction == -1:
                self.active = False
                self.progress = 0.0
            else:
                self.progress = 1.0

    def set_person_center(self, cx: float, cy: float) -> None:
        """Set the center point of detected person (normalized 0-1).

        Used by effects that animate from/to the person's position.

        Args:
            cx: Normalized x coordinate (0.0 to 1.0)
            cy: Normalized y coordinate (0.0 to 1.0)
        """
        self.person_center = (cx, cy)

    def is_active(self) -> bool:
        """Check if effect should be applied.

        Returns:
            True if effect is active or transitioning
        """
        return self.active or self.transitioning

    def force_on(self) -> None:
        """Force effect to on state immediately (no animation)."""
        self.active = True
        self.transitioning = False
        self.progress = 1.0

    def force_off(self) -> None:
        """Force effect to off state immediately (no animation)."""
        self.active = False
        self.transitioning = False
        self.progress = 0.0
