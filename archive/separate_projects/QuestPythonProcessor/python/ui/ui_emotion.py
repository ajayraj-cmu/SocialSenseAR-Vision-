#!/usr/bin/env python3
"""
Standalone test for glassmorphism webview UI
Single Emotion box, VR-neutral styling with calm blues:
- soft outline
- slightly transparent interior
- desaturated blue glass (low harsh contrast)
"""

import json
import os
from pathlib import Path
import webview

CONVO_HELPER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Emotion</title>

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
    padding: 16px;
    display: flex;
    align-items: center;
    justify-content: center;

    /* Soft, readable text */
    color: rgba(232, 240, 248, 0.94);
}

/* Hide scrollbars */
::-webkit-scrollbar { display: none; }
* { scrollbar-width: none; }

/* =========================
   VR-neutral glass card (Calm Blues)
   ========================= */
.glass-card {
    width: 100%;
    height: 100%;

    /* Calm, desaturated blue glass */
    background: rgba(86, 132, 196, 0.20);

    backdrop-filter: blur(34px) saturate(135%);
    -webkit-backdrop-filter: blur(34px) saturate(135%);

    /* Softer outline to reduce visual stress */
    border: 2px solid rgba(200, 220, 252, 0.72);

    border-radius: 18px;
    padding: 16px;

    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.26);

    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 10px;

    transition: background 0.25s ease, border-color 0.25s ease;
}

.glass-card:hover {
    background: rgba(96, 144, 208, 0.23);
    border-color: rgba(210, 228, 255, 0.78);
}

/* Label */
.section-label {
    font-size: 10px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 1.2px;

    color: rgba(208, 224, 244, 0.92);

    display: flex;
    align-items: center;
    gap: 8px;
}

.icon {
    width: 14px;
    height: 14px;
    opacity: 0.82;
}

/* Emotion text */
.emotion-text {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.2px;

    color: rgba(244, 250, 255, 0.96);
    text-shadow: 0 1px 2px rgba(12, 20, 32, 0.35);

    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(6px) scale(0.985);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.fade-in {
    animation: fadeIn 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
</style>
</head>

<body>
<div class="glass-card">
    <div class="section-label">
        <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 21s-7-4.35-7-11
                   a4 4 0 017-2
                   a4 4 0 017 2
                   c0 6.65-7 11-7 11z"/>
        </svg>
        Emotion
    </div>

    <div class="emotion-text" id="emotion-text">Calm</div>
</div>

<script>
let lastUpdateTime = 0;

function updateUI(state) {
    if (!state) return;
    const emotion = state.emotion || state.emotion_label || "Calm";
    const el = document.getElementById("emotion-text");

    if (el) {
        el.textContent = emotion;
        el.classList.add("fade-in");
        setTimeout(() => el.classList.remove("fade-in"), 350);
    }
}

function loadLatestState() {
    if (window.pywebview && window.pywebview.api) {
        window.pywebview.api.get_latest_state()
            .then(state => {
                if (!state) return;
                const ts = state.timestamp || 0;
                if (ts > lastUpdateTime || !lastUpdateTime) {
                    lastUpdateTime = ts;
                    updateUI(state);
                }
            })
            .catch(err => console.error(err));
    }
}

setInterval(loadLatestState, 500);
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


class TestAPI:
    def __init__(self, window_ref):
        self._window_ref = window_ref

        env_path = os.getenv("CONVO_STATE_PATH")
        if env_path:
            self.state_file = Path(env_path).expanduser()
        else:
            home_guess = Path("~/Downloads/Nex/conve_context/latest_state.json").expanduser()
            rel_guess = (
                Path(__file__).resolve().parent.parent.parent
                / "conve_context"
                / "latest_state.json"
            )
            self.state_file = home_guess if home_guess.exists() else rel_guess

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


def main():
    window_ref = [None]
    api = TestAPI(window_ref)

    window = webview.create_window(
        "Emotion",
        html=CONVO_HELPER_HTML,
        js_api=api,
        width=260,
        height=140,
        resizable=True,
        frameless=True,
        easy_drag=True,
        transparent=True,
        background_color="#000000",
    )
    window_ref[0] = window
    webview.start(debug=False)


if __name__ == "__main__":
    main()
