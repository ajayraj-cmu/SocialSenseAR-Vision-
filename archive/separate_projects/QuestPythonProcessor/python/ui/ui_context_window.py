#!/usr/bin/env python3
"""
Standalone test for glassmorphism webview UI (Context Window)
Palette tuned for neurodivergent-friendly calm blues:
- lower saturation
- soft contrast (readable but not harsh)
- consistent cool-blue glass across cards
"""

import webview
import json
from pathlib import Path

CONVO_HELPER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Conversation Helper</title>

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

    /* Softer, neurodivergent-friendly text (not pure white) */
    color: rgba(232, 240, 248, 0.94);
}

/* Hide scrollbars */
::-webkit-scrollbar { display: none; }
* { scrollbar-width: none; }

/* =========================
   Layout
   ========================= */
.main-container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* =========================
   Shared glass surface (Calm Blues)
   ========================= */
.glass-surface {
    /* Calm, desaturated blue glass */
    background: rgba(86, 132, 196, 0.20);

    backdrop-filter: blur(34px) saturate(135%);
    -webkit-backdrop-filter: blur(34px) saturate(135%);

    /* Softer border (avoid harsh bright outline) */
    border: 2px solid rgba(200, 220, 252, 0.72);

    border-radius: 18px;
    padding: 14px 16px;

    /* Subtle inner highlight, reduced glare */
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.26);

    transition: background 0.25s ease, border-color 0.25s ease;
}

.glass-surface:hover {
    background: rgba(96, 144, 208, 0.23);
    border-color: rgba(210, 228, 255, 0.78);
}

/* =========================
   Labels
   ========================= */
.section-label {
    font-size: 10px;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 1.2px;

    /* Muted label text (still readable) */
    color: rgba(208, 224, 244, 0.92);

    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}

.icon {
    width: 14px;
    height: 14px;

    /* Slightly softer icon */
    opacity: 0.82;
}

/* =========================
   Conversation summary
   ========================= */
.main-topic {
    font-size: 16px;
    font-weight: 650;
    line-height: 1.4;

    /* Soft near-white, avoids harshness */
    color: #fff !important;
    opacity: 1;

    /* Gentle depth without heavy contrast */
    text-shadow: 0 1px 2px rgba(12, 20, 32, 0.35);

    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* =========================
   Question (MATCHES CALM BLUES)
   ========================= */
.question-box {
    background: rgba(86, 132, 196, 0.20);

    backdrop-filter: blur(34px) saturate(135%);
    -webkit-backdrop-filter: blur(34px) saturate(135%);

    border: 2px solid rgba(200, 220, 252, 0.72);

    border-radius: 18px;
    padding: 14px 16px;

    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.26);
}

.question-text {
    font-size: 15px;
    font-weight: 600;
    line-height: 1.4;

    color: #fff !important;
    opacity: 1;
    text-shadow: 0 1px 2px rgba(12, 20, 32, 0.30);

    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.no-question {
    opacity: 0.60;
    font-style: italic;
}

/* =========================
   Utterance (lighter glass)
   ========================= */
.utterance-box {
    /* Lighter, less “blocky” and less contrasty */
    background: rgba(96, 144, 208, 0.14);

    backdrop-filter: blur(26px) saturate(125%);
    -webkit-backdrop-filter: blur(26px) saturate(125%);

    border: 1.5px solid rgba(200, 220, 252, 0.55);

    border-radius: 18px;
    padding: 12px 14px;
}

.utterance-text {
    font-size: 13px;
    line-height: 1.4;
    font-style: italic;

    color: #fff !important;
    opacity: 1;

    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;

    /* keep very subtle */
    text-shadow: 0 1px 2px rgba(12, 20, 32, 0.22);
}

/* =========================
   Animations
   ========================= */
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
<div class="main-container">

    <!-- Conversation Summary -->
    <div class="glass-surface">
        <div class="section-label">
            <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                      d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8
                         a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72
                         C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8
                         s9 3.582 9 8z"/>
            </svg>
            Conversation
        </div>
        <div class="main-topic" id="convo-summary">--</div>
    </div>

    <!-- Question -->
    <div class="question-box">
        <div class="section-label">
            <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                      d="M8.228 9c.549-1.165 2.03-2 3.772-2
                         2.21 0 4 1.343 4 3
                         0 1.4-1.278 2.575-3.006 2.907
                         -.542.104-.994.54-.994 1.093m0 3h.01"/>
            </svg>
            Question
        </div>
        <div class="question-text" id="question-text">--</div>
    </div>

    <!-- Utterance -->
    <div class="utterance-box">
        <div class="section-label">
            <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                      d="M7 8h10M7 12h4m1 8l-4-4H5
                         a2 2 0 01-2-2V6a2 2 0 012-2h14
                         a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"/>
            </svg>
            Just Said
        </div>
        <div class="utterance-text" id="utterance-text">--</div>
    </div>

</div>

<script>
let lastUpdateTime = 0;

function updateUI(state) {
    if (!state) return;

    const summary = document.getElementById("convo-summary");
    const question = document.getElementById("question-text");
    const utterance = document.getElementById("utterance-text");

    if (state.convo_state_summary) {
        summary.textContent = state.convo_state_summary;
        summary.classList.add("fade-in");
        setTimeout(() => summary.classList.remove("fade-in"), 350);
    }

    if (state.question) {
        question.textContent = state.question;
        question.classList.remove("no-question");
    } else {
        question.textContent = "No question detected";
        question.classList.add("no-question");
    }

    if (state.recent_utterance) {
        utterance.textContent = '"' + state.recent_utterance + '"';
        utterance.classList.add("fade-in");
        setTimeout(() => utterance.classList.remove("fade-in"), 350);
    }
}

function loadLatestState() {
    if (window.pywebview && window.pywebview.api) {
        window.pywebview.api.get_latest_state()
            .then(state => {
                if (!state) return;
                if ((state.timestamp || 0) > lastUpdateTime) {
                    lastUpdateTime = state.timestamp || Date.now();
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
        self.state_file = (
            Path(__file__).resolve().parent.parent.parent
            / "conve_context"
            / "latest_state.json"
        )

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
        "Conversation Helper",
        html=CONVO_HELPER_HTML,
        js_api=api,
        width=340,
        height=420,
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
