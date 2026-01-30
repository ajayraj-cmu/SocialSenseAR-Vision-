#!/usr/bin/env python3
"""
Transition Indicator UI
Small circular indicator that changes color based on speech transition state:
- Green: User speaking
- Blue: Other person speaking
- Yellow: Other person stopped (2+ sec silence)
"""

import json
import os
from pathlib import Path
import webview

TRANSITION_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Transition Indicator</title>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

html, body {
    height: 100%;
    width: 100%;
    background: transparent;
    overflow: hidden;
}

body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* =========================
   Circular Indicator
   ========================= */
.indicator-circle {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    
    /* Default color (gray) */
    background: #6b7280;
    
    /* Smooth color transitions */
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
    
    /* Subtle glow effect */
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
}

/* Color states */
.indicator-circle.status-green {
    background: #22c55e;
    box-shadow: 0 0 25px rgba(34, 197, 94, 0.5);
}

.indicator-circle.status-blue {
    background: #3b82f6;
    box-shadow: 0 0 25px rgba(59, 130, 246, 0.5);
}

.indicator-circle.status-yellow {
    background: #eab308;
    box-shadow: 0 0 25px rgba(234, 179, 8, 0.5);
}

/* Pulse animation on color change */
@keyframes pulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.05);
    }
}

.indicator-circle.pulse {
    animation: pulse 0.4s ease;
}
</style>
</head>

<body>
    <div class="indicator-circle" id="indicator-circle"></div>

<script>
let lastUpdateTime = 0;
let lastStatus = null;

function updateIndicator(state) {
    if (!state) return;
    
    const circle = document.getElementById("indicator-circle");
    if (!circle) return;
    
    const status = state.status || null;
    const timestamp = state.timestamp || 0;
    
    // Only update if timestamp is newer or status changed
    if (timestamp > lastUpdateTime || status !== lastStatus) {
        lastUpdateTime = timestamp;
        
        // Remove all status classes
        circle.classList.remove("status-green", "status-blue", "status-yellow");
        
        // Add new status class
        if (status === "green") {
            circle.classList.add("status-green");
        } else if (status === "blue") {
            circle.classList.add("status-blue");
        } else if (status === "yellow") {
            circle.classList.add("status-yellow");
        }
        
        // Add pulse animation if status changed
        if (status !== lastStatus && lastStatus !== null) {
            circle.classList.add("pulse");
            setTimeout(() => circle.classList.remove("pulse"), 400);
        }
        
        lastStatus = status;
    }
}

function loadLatestState() {
    if (window.pywebview && window.pywebview.api) {
        window.pywebview.api.get_latest_state()
            .then(state => {
                if (state) {
                    updateIndicator(state);
                }
            })
            .catch(err => console.error(err));
    }
}

setInterval(loadLatestState, 300);
loadLatestState();

document.addEventListener("keydown", e => {
    if (e.key && e.key.toLowerCase() === "q" && window.pywebview) {
        pywebview.api.close_window();
    }
});
</script>
</body>
</html>
"""


class TransitionAPI:
    def __init__(self, window_ref):
        self._window_ref = window_ref
        
        # Look for transition_state.json in the same directory as this file
        # or in the parent directory (where speech_transition.py is)
        script_dir = Path(__file__).resolve().parent
        self.state_file = script_dir / "transition_state.json"
        
        # If not found, try parent directory
        if not self.state_file.exists():
            parent_guess = script_dir.parent / "transition_state.json"
            if parent_guess.exists():
                self.state_file = parent_guess

    def close_window(self):
        if self._window_ref[0]:
            self._window_ref[0].destroy()

    def get_latest_state(self):
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    return json.load(f)
            return None
        except Exception as e:
            print("State read error:", e)
            return None


def start_ui():
    """Start the transition indicator UI in a webview window"""
    window_ref = [None]
    api = TransitionAPI(window_ref)

    window = webview.create_window(
        "Transition Indicator",
        html=TRANSITION_HTML,
        js_api=api,
        width=120,
        height=120,
        resizable=False,
        frameless=True,
        easy_drag=True,
        transparent=True,
        background_color="#000000",
    )
    window_ref[0] = window
    webview.start(debug=False)


if __name__ == "__main__":
    start_ui()
