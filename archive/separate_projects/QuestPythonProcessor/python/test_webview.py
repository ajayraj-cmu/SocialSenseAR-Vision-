#!/usr/bin/env python3
"""
Standalone test for glassmorphism webview UI.

Run this to test the webview window separately from the OpenCV pipeline.
Usage: python test_webview.py
"""
import webview
import time
import threading

# Conversation helper UI for neurodivergent users
CONVO_HELPER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation Helper</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body {
            height: 100%;
            width: 100%;
            overflow: hidden;
            background: transparent;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
            color: white;
            -webkit-font-smoothing: antialiased;
        }

        /* Main glass container */
        .main-container {
            position: absolute;
            top: 8px;
            left: 8px;
            right: 8px;
            bottom: 8px;
            background: rgba(35, 35, 40, 0.92);
            backdrop-filter: blur(80px) saturate(150%);
            -webkit-backdrop-filter: blur(80px) saturate(150%);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow:
                0 25px 60px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Content - no scrollbar for headset use */
        .content {
            flex: 1;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            overflow: hidden;
        }

        /* Hide all scrollbars */
        ::-webkit-scrollbar {
            display: none;
        }

        * {
            scrollbar-width: none;
            -ms-overflow-style: none;
        }

        /* Context section */
        .context-section {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 10px 12px;
            flex-shrink: 0;
            overflow: hidden;
        }

        .section-label {
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(255, 255, 255, 0.35);
            margin-bottom: 4px;
        }

        /* Main topic */
        .main-topic {
            font-size: 15px;
            font-weight: 600;
            color: #fff !important;
            opacity: 1;
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        /* Sub context */
        .sub-context {
            font-size: 12px;
            color: #fff !important;
            opacity: 1;
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        /* Question highlight */
        .question-box {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 12px;
            padding: 10px 12px;
            flex-shrink: 0;
            overflow: hidden;
        }

        .question-label {
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: rgba(167, 139, 250, 0.7);
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .question-icon {
            width: 12px;
            height: 12px;
            flex-shrink: 0;
        }

        .question-text {
            font-size: 13px;
            font-weight: 500;
            color: #fff !important;
            opacity: 1;
            line-height: 1.3;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        .no-question {
            color: rgba(255, 255, 255, 0.3);
            font-style: italic;
        }

        /* Recent utterance */
        .utterance-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 8px 10px;
            flex-shrink: 0;
            overflow: hidden;
        }

        .utterance-text {
            font-size: 11px;
            color: #fff !important;
            opacity: 1;
            line-height: 1.35;
            font-style: italic;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="main-container">

        <div class="content">
            <!-- Main Topic -->
            <div class="context-section">
                <div class="section-label">Talking About</div>
                <div class="main-topic" id="main-topic">--</div>
            </div>

            <!-- Sub Context -->
            <div class="context-section">
                <div class="section-label">Right Now</div>
                <div class="sub-context" id="sub-context">--</div>
            </div>

            <!-- Current Question -->
            <div class="question-box">
                <div class="question-label">
                    <svg class="question-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    They Asked
                </div>
                <div class="question-text" id="question-text">
                    <span class="no-question">No question right now</span>
                </div>
            </div>

            <!-- Recent Utterance -->
            <div class="utterance-box">
                <div class="section-label">Just Said</div>
                <div class="utterance-text" id="utterance-text">--</div>
            </div>
        </div>

    </div>

    <script>
        // Test data - simulating conve_context states
        const testStates = [
            {
                main_idea: "Customer service interaction",
                sub_convo: "Customer requesting assistance",
                question: "",
                utterance: "Hi there, what can I get started for you today?"
            },
            {
                main_idea: "Customer service interaction",
                sub_convo: "Customer placing an order for coffee",
                question: "Would you like that hot or iced?",
                utterance: "Sure, take your time. OK, I'd like to order a coffee, a medium size."
            },
            {
                main_idea: "Customer service interaction",
                sub_convo: "Customer placing an order for coffee with modifications",
                question: "Would you like that hot or iced?",
                utterance: "Hot, please. And could I add one extra shot of espresso?"
            },
            {
                main_idea: "Customer service interaction",
                sub_convo: "Customer adding items to order",
                question: "What size would you like for the pastry?",
                utterance: "Absolutely, one extra shot. Anything else I can get for you?"
            },
            {
                main_idea: "Customer service interaction",
                sub_convo: "Finalizing the order",
                question: "Is that everything for today?",
                utterance: "Actually, could I also get a blueberry muffin?"
            },
            {
                main_idea: "Customer service interaction",
                sub_convo: "Payment and closing",
                question: "",
                utterance: "That'll be $7.50. You can tap or insert your card whenever you're ready."
            }
        ];

        let currentState = 0;

        function updateUI(state) {
            document.getElementById('main-topic').textContent = state.main_idea;
            document.getElementById('main-topic').classList.add('fade-in');

            document.getElementById('sub-context').textContent = state.sub_convo;

            const questionEl = document.getElementById('question-text');
            if (state.question) {
                questionEl.innerHTML = state.question;
            } else {
                questionEl.innerHTML = '<span class="no-question">No question right now</span>';
            }

            document.getElementById('utterance-text').textContent = '"' + state.utterance + '"';

            // Remove animation class after animation completes
            setTimeout(() => {
                document.getElementById('main-topic').classList.remove('fade-in');
            }, 300);
        }

        function nextState() {
            currentState = (currentState + 1) % testStates.length;
            updateUI(testStates[currentState]);
        }

        function prevState() {
            currentState = (currentState - 1 + testStates.length) % testStates.length;
            updateUI(testStates[currentState]);
        }

        // Arrow keys for testing, Q to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') nextState();
            if (e.key === 'ArrowLeft') prevState();
            if (e.key.toLowerCase() === 'q' && window.pywebview) {
                pywebview.api.close_window();
            }
        });

        // Initialize
        updateUI(testStates[0]);
    </script>
</body>
</html>
"""


class TestAPI:
    """API exposed to JavaScript for window control."""

    def __init__(self, window_ref):
        self._window_ref = window_ref

    def close_window(self):
        """Close the window."""
        if self._window_ref[0]:
            self._window_ref[0].destroy()

    def minimize_window(self):
        """Minimize the window."""
        if self._window_ref[0]:
            self._window_ref[0].minimize()

    def maximize_window(self):
        """Toggle maximize."""
        if self._window_ref[0]:
            self._window_ref[0].toggle_fullscreen()

    def send_command(self, command: str):
        """Handle commands from UI."""
        pass


def main():
    """Launch the conversation helper window."""
    # Create window reference holder
    window_ref = [None]
    api = TestAPI(window_ref)

    # Create frameless window with transparent background
    window = webview.create_window(
        'Conversation Helper',
        html=CONVO_HELPER_HTML,
        js_api=api,
        width=320,
        height=360,
        resizable=True,
        frameless=True,  # Remove window chrome for clean look
        easy_drag=True,  # Allow dragging from title bar
        transparent=True,  # Enable transparency for rounded corners
        background_color='#0a0a0f',  # Dark background fallback
    )
    window_ref[0] = window

    # Start the webview
    webview.start(debug=False)


if __name__ == "__main__":
    main()
