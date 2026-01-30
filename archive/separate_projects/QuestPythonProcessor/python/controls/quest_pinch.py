"""
Quest pinch gesture control.

Detects pinch gestures from Quest hand tracking via ADB logcat.
"""
from typing import Optional, Callable
import threading
import subprocess
import re
import time

from .base import BaseControl


class QuestPinchControl(BaseControl):
    """Quest hand tracking pinch gesture detector.

    Monitors ADB logcat for Quest hand tracking events.
    Triggers callback when pinch gesture is detected.

    Attributes:
        debounce_seconds: Minimum time between pinch triggers
        pointer_pos: Current hand pointer position (normalized 0-1)
    """

    def __init__(self, callback: Optional[Callable] = None):
        """Initialize pinch gesture monitor.

        Args:
            callback: Function to call when pinch is detected
        """
        super().__init__(callback)
        self.running = False
        self.thread = None
        self.last_pinch_time = 0
        self.debounce_seconds = 1.2
        self.last_gesture_state = {0: 0, 1: 0}

        # Hand pointer position
        self.pointer_pos = None
        self.pointer_lock = threading.Lock()

    def start(self) -> None:
        """Start monitoring for pinch gestures."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        print("Pinch gesture monitor started (pinch fingers to toggle effect)")

    def stop(self) -> None:
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_pointer_pos(self) -> Optional[tuple]:
        """Get current pointer position (normalized 0-1).

        Returns:
            Tuple (x, y) or None if not available
        """
        with self.pointer_lock:
            return self.pointer_pos

    def _monitor(self) -> None:
        """Background thread that monitors ADB logcat."""
        try:
            import adbutils
            adb_path = adbutils.adb_path()
            print(f"[PINCH] Using ADB at: {adb_path}")

            # Check if Quest is connected
            result = subprocess.run([adb_path, 'devices'], capture_output=True, text=True, timeout=5)
            print(f"[PINCH] ADB devices:\n{result.stdout}")

            # Clear logcat buffer first
            subprocess.run([adb_path, 'logcat', '-c'], capture_output=True, timeout=2)

            # Start monitoring logcat for hand tracking events
            print("[PINCH] Starting hand tracking monitor...")
            process = subprocess.Popen(
                [adb_path, 'logcat', '-v', 'time', 'TrexHandInputDataServerPlugin:I', '*:S'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Pattern: "HandInputData NEW MG changed for hand X Y"
            pattern = re.compile(r'MG changed for hand (\d+)\s+(\d+)')

            while self.running and process.poll() is None:
                line = process.stdout.readline()
                if not line:
                    continue

                match = pattern.search(line)
                if match:
                    hand = int(match.group(1))
                    gesture = int(match.group(2))

                    # Detect pinch: transition from 0 to 6 or 7
                    if gesture in [6, 7] and self.last_gesture_state.get(hand, 0) == 0:
                        now = time.time()
                        if now - self.last_pinch_time > self.debounce_seconds:
                            self.last_pinch_time = now
                            print(f"\n[PINCH - Hand {hand}]")
                            self.trigger()

                    self.last_gesture_state[hand] = gesture

            process.terminate()

        except FileNotFoundError:
            print("Warning: ADB not found, pinch detection disabled")
        except Exception as e:
            print(f"Warning: Pinch monitor error: {e}")
