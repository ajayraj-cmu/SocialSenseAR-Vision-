"""
Pywebview UI backend with glassmorphism styling.

Modern web-based UI using pywebview for beautiful glass-effect panels.
"""
import base64
import threading
import queue
from typing import Optional
import numpy as np
import cv2

try:
    import webview
except ImportError:
    webview = None

from .base import BaseUI


class WebViewAPI:
    """API exposed to JavaScript."""

    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=2)
        self.input_queue = queue.Queue()
        self.stats = {}
        self._current_frame_b64 = None

    def get_frame(self):
        """Get the latest frame as base64 JPEG."""
        return self._current_frame_b64

    def get_stats(self):
        """Get current stats."""
        return self.stats

    def send_key(self, key_code: int):
        """Receive key press from JS."""
        self.input_queue.put(key_code)

    def send_command(self, command: str):
        """Receive command from UI buttons."""
        # Map commands to key codes
        key_map = {
            'quit': ord('q'),
            'toggle_effect': ord('e'),
            'cycle_mode': ord('m'),
            'reset': ord('r'),
        }
        if command in key_map:
            self.input_queue.put(key_map[command])


GLASS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quest Processor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: white;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }

        #video-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #1a1a2e 0%, #0a0a0f 100%);
        }

        #video-frame {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
            /* Crisp rendering - prevents blurry scaling */
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }

        .glass-panel {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }

        #stats-panel {
            position: absolute;
            top: 20px;
            left: 20px;
            padding: 20px 24px;
            min-width: 200px;
        }

        .stats-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: rgba(255, 255, 255, 0.5);
            margin-bottom: 12px;
        }

        .stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stat-row:last-child {
            border-bottom: none;
        }

        .stat-label {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.7);
        }

        .stat-value {
            font-size: 14px;
            font-weight: 600;
            color: #fff;
            font-variant-numeric: tabular-nums;
        }

        .stat-value.fps {
            color: #4ade80;
        }

        .stat-value.status {
            color: #60a5fa;
        }

        #controls-panel {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 16px 24px;
            display: flex;
            gap: 12px;
        }

        .glass-button {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 12px 20px;
            color: white;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .glass-button:hover {
            background: rgba(255, 255, 255, 0.2);
            border-color: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .glass-button:active {
            transform: translateY(0);
        }

        .glass-button.danger {
            background: rgba(239, 68, 68, 0.2);
            border-color: rgba(239, 68, 68, 0.3);
        }

        .glass-button.danger:hover {
            background: rgba(239, 68, 68, 0.3);
        }

        .glass-button svg {
            width: 16px;
            height: 16px;
        }

        #title-bar {
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
        }

        .title-text {
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.5px;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .recording-dot {
            width: 8px;
            height: 8px;
            background: #ef4444;
            border-radius: 50%;
            animation: pulse 1.5s ease-in-out infinite;
            display: inline-block;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div id="video-container">
        <img id="video-frame" src="" alt="Video Feed">
    </div>

    <div id="stats-panel" class="glass-panel">
        <div class="stats-title">Performance</div>
        <div class="stat-row">
            <span class="stat-label">FPS</span>
            <span class="stat-value fps" id="fps-value">--</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Resolution</span>
            <span class="stat-value" id="resolution-value">--</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Status</span>
            <span class="stat-value status" id="status-value">Starting...</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Effect</span>
            <span class="stat-value" id="effect-value">--</span>
        </div>
    </div>

    <div id="title-bar" class="glass-panel">
        <span class="recording-dot"></span>
        <span class="title-text">Quest Processor</span>
    </div>

    <div id="controls-panel" class="glass-panel">
        <button class="glass-button" onclick="sendCommand('toggle_effect')">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"></path>
            </svg>
            Toggle Effect
        </button>
        <button class="glass-button" onclick="sendCommand('cycle_mode')">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            Cycle Mode
        </button>
        <button class="glass-button" onclick="sendCommand('reset')">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            Reset
        </button>
        <button class="glass-button danger" onclick="sendCommand('quit')">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
            Quit
        </button>
    </div>

    <script>
        const videoFrame = document.getElementById('video-frame');
        const fpsValue = document.getElementById('fps-value');
        const resolutionValue = document.getElementById('resolution-value');
        const statusValue = document.getElementById('status-value');
        const effectValue = document.getElementById('effect-value');

        async function updateFrame() {
            try {
                const frame = await pywebview.api.get_frame();
                if (frame) {
                    videoFrame.src = 'data:image/png;base64,' + frame;
                }
            } catch (e) {
                console.error('Frame error:', e);
            }
            requestAnimationFrame(updateFrame);
        }

        async function updateStats() {
            try {
                const stats = await pywebview.api.get_stats();
                if (stats) {
                    if (stats.fps !== undefined) {
                        fpsValue.textContent = stats.fps.toFixed(1);
                    }
                    if (stats.resolution) {
                        resolutionValue.textContent = stats.resolution;
                    }
                    if (stats.status) {
                        statusValue.textContent = stats.status;
                    }
                    if (stats.effect) {
                        effectValue.textContent = stats.effect;
                    }
                }
            } catch (e) {
                console.error('Stats error:', e);
            }
            setTimeout(updateStats, 100);
        }

        function sendCommand(cmd) {
            pywebview.api.send_command(cmd);
        }

        document.addEventListener('keydown', (e) => {
            pywebview.api.send_key(e.keyCode);
        });

        // Start updates when pywebview is ready
        window.addEventListener('pywebviewready', () => {
            updateFrame();
            updateStats();
        });
    </script>
</body>
</html>
"""


class WebViewUI(BaseUI):
    """Pywebview-based UI with glassmorphism styling.

    Modern, beautiful UI using web technologies.
    Requires: pip install pywebview
    """

    def __init__(self, config=None):
        """Initialize WebView UI.

        Args:
            config: Configuration object with display settings
        """
        if webview is None:
            raise ImportError("pywebview not installed. Run: pip install pywebview")

        self.config = config
        self.api = WebViewAPI()
        self.window = None
        self._thread = None
        self._ready = threading.Event()
        # Higher quality for better text readability (default was 85)
        self.jpeg_quality = getattr(config, 'webview_jpeg_quality', 98) if config else 98

    def _run_webview(self, title: str):
        """Run webview in separate thread."""
        self.window = webview.create_window(
            title,
            html=GLASS_HTML,
            js_api=self.api,
            width=1280,
            height=800,
            resizable=True,
            frameless=False,
            easy_drag=False,
            background_color='#0a0a0f'
        )
        self._ready.set()
        webview.start()

    def setup(self, title: str = "Quest Processor") -> None:
        """Initialize the webview window.

        Args:
            title: Window title
        """
        self._thread = threading.Thread(target=self._run_webview, args=(title,), daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        # Give webview a moment to fully initialize
        import time
        time.sleep(0.5)

    def show(self, frame: np.ndarray, stats: Optional[dict] = None) -> None:
        """Display a frame.

        Args:
            frame: BGR frame to display
            stats: Optional stats dict for overlay
        """
        if self.window is None:
            return

        # Encode frame as PNG base64 (lossless for crisp text)
        _, buffer = cv2.imencode('.png', frame)
        self.api._current_frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Update stats
        if stats:
            self.api.stats = stats
            # Add resolution if not present
            if 'resolution' not in self.api.stats:
                h, w = frame.shape[:2]
                self.api.stats['resolution'] = f"{w}x{h}"

    def poll_input(self) -> Optional[int]:
        """Poll for keyboard/button input.

        Returns:
            Key code (0-255) or None if no input
        """
        try:
            return self.api.input_queue.get_nowait()
        except queue.Empty:
            return None

    def cleanup(self) -> None:
        """Destroy the window."""
        if self.window:
            try:
                self.window.destroy()
            except:
                pass
