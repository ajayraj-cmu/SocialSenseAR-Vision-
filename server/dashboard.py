"""Reusable dashboard UI for SocialSenseAR.

Used by:
- websocket_server.py: when running local server with Quest/webcam client
- test_client.py: when connecting to Modal (remote server, local display)

Shows camera feed + mask overlays + status panel + input bar.
"""

import time
import traceback
import cv2
import numpy as np
from typing import Optional, Callable
from server.encoding.rle import decode_rle


class Dashboard:
    """Live dashboard with camera feed, mask overlays, and command input.

    Args:
        window_name: OpenCV window title
        on_command: Callback when user types a command (receives raw command string)
        get_active_prompts: Callback to get active prompts set
        get_effects: Callback to get effects dict
    """

    _BRIGHT_COLORS = [
        (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
        (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
        (0, 200, 100), (100, 255, 200),
    ]

    _PANEL_W = 220       # right panel width
    _INPUT_BAR_H = 40    # bottom input bar height
    _PANEL_BG = (30, 30, 30)
    _INPUT_BG = (20, 20, 20)
    _HEADER_COLOR = (0, 200, 255)   # orange-yellow headers
    _TEXT_COLOR = (200, 200, 200)    # light grey text
    _ACTIVE_COLOR = (0, 255, 150)   # green for active prompts
    _CURSOR_COLOR = (0, 200, 255)

    def __init__(
        self,
        window_name: str = "SocialSenseAR",
        on_command: Optional[Callable[[str], None]] = None,
        get_active_prompts: Optional[Callable[[], set]] = None,
        get_effects: Optional[Callable[[], dict]] = None,
        confidence_min: float = 0.10,
    ):
        self.window_name = window_name
        self.on_command = on_command
        self.get_active_prompts = get_active_prompts or (lambda: set())
        self.get_effects = get_effects or (lambda: {})
        self._confidence_min = confidence_min

        self._input_text = ""
        self._input_log: list[str] = []
        self._start_time = time.time()
        self._frame_count = 0
        self._mask_update_times: list[float] = []
        self._last_mask_fingerprint = ""
        self._window_created = False
        self._render_error_count = 0
        self._last_seg_debug = 0.0
        self._show_segments = True

        # Create window with GUI_NORMAL for macOS title bar (red/yellow/green buttons)
        cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_NORMAL)
        self._window_created = True

    def update(
        self,
        frame: np.ndarray,
        segments: list,
        voice_agent_state: Optional[dict] = None,
    ):
        """Render dashboard with new frame and segments.

        Args:
            frame: BGR image (numpy array)
            segments: List of segment objects with .label, .rle_mask, .center_x, etc.
                     Can be protobuf SceneSegment or similar objects.
            voice_agent_state: Optional dict with keys: listening, recording, last_response
        """
        self._frame_count += 1

        if frame is None:
            return

        fh, fw = frame.shape[:2]
        display_frame = frame.copy()

        # Track mask updates
        fp = self._mask_fingerprint(segments)
        if fp != self._last_mask_fingerprint:
            self._last_mask_fingerprint = fp
            self._mask_update_times.append(time.perf_counter())
            if len(self._mask_update_times) > 120:
                self._mask_update_times.pop(0)

        # --- Debug: log segment data periodically ---
        now = time.time()
        if segments and now - self._last_seg_debug > 3.0:
            self._last_seg_debug = now
            for i, seg in enumerate(segments):
                rle = getattr(seg, 'rle_mask', b'')
                mw = getattr(seg, 'mask_width', 0)
                mh = getattr(seg, 'mask_height', 0)
                conf = getattr(seg, 'confidence', -1)
                cx = getattr(seg, 'center_x', -1)
                cy = getattr(seg, 'center_y', -1)
                lbl = getattr(seg, 'label', '?')
                print(f"  [Dashboard] seg[{i}]: label={lbl} conf={conf:.3f} "
                      f"rle={len(rle)}B mask={mw}x{mh} center=({cx:.3f},{cy:.3f}) "
                      f"frame={fh}x{fw}")

        # --- Draw mask overlays ---
        for i, seg in enumerate(segments):
            conf = getattr(seg, 'confidence', 1.0)
            if conf < self._confidence_min:
                continue

            rle_mask = getattr(seg, 'rle_mask', None)
            mask_w = getattr(seg, 'mask_width', 0)
            mask_h = getattr(seg, 'mask_height', 0)

            if not rle_mask or mask_w <= 0 or mask_h <= 0:
                continue

            try:
                mask = decode_rle(rle_mask, mask_w, mask_h)
                if mask.shape[:2] != (fh, fw):
                    mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_LINEAR)
                mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)

                # Apply visual effect if present
                effect = getattr(seg, 'effect', None)
                effect_type = ""
                is_inverted = False
                intensity = 1.0
                if effect:
                    effect_type = getattr(effect, 'effect_type', '')
                    params = getattr(effect, 'params', {})
                    is_inverted = params.get("invert", "") == "true" if params else False
                    try:
                        intensity = float(params.get("intensity", 1.0)) if params else 1.0
                    except (ValueError, TypeError):
                        intensity = 1.0

                if effect_type and effect_type != "none":
                    mask_bool = mask_u8 > 128
                    if is_inverted:
                        mask_bool = ~mask_bool

                    if effect_type == "blur":
                        ksize = max(3, int(51 * intensity)) | 1  # must be odd
                        blurred = cv2.GaussianBlur(display_frame, (ksize, ksize), 0)
                        display_frame[mask_bool] = blurred[mask_bool]

                    elif effect_type == "dim":
                        dimmed = (display_frame * (1.0 - 0.7 * intensity)).astype(np.uint8)
                        display_frame[mask_bool] = dimmed[mask_bool]

                    elif effect_type == "pixelate":
                        block = max(4, int(20 * intensity))
                        h, w = display_frame.shape[:2]
                        small = cv2.resize(display_frame, (w // block, h // block), interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                        display_frame[mask_bool] = pixelated[mask_bool]

                    elif effect_type == "highlight":
                        bright = cv2.addWeighted(display_frame, 1.0, np.full_like(display_frame, 255), 0.3 * intensity, 0)
                        tint = bright.copy()
                        tint[:, :, 0] = np.clip(tint[:, :, 0].astype(np.int16) + int(40 * intensity), 0, 255).astype(np.uint8)  # yellow-ish tint (add green)
                        tint[:, :, 2] = np.clip(tint[:, :, 2].astype(np.int16) + int(40 * intensity), 0, 255).astype(np.uint8)  # add red
                        display_frame[mask_bool] = tint[mask_bool]

                    elif effect_type == "outline":
                        pass  # outline is drawn below in contour section

                # Draw contours and labels (hidden with Ctrl+P)
                if self._show_segments or effect_type == "outline":
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        color = self._BRIGHT_COLORS[i % len(self._BRIGHT_COLORS)]
                        if effect_type == "outline":
                            thickness = max(3, int(8 * intensity))
                            cv2.drawContours(display_frame, contours, -1, (0, 0, 0), thickness + 3)
                            cv2.drawContours(display_frame, contours, -1, color, thickness)
                        elif self._show_segments:
                            cv2.drawContours(display_frame, contours, -1, (0, 0, 0), 5)
                            cv2.drawContours(display_frame, contours, -1, color, 2)

                        if self._show_segments:
                            label = getattr(seg, 'label', '') or getattr(seg, 'asset_class', '') or ''
                            if effect_type and effect_type != "none":
                                label = f"{label} [{effect_type}]"
                            if label:
                                cx = int(getattr(seg, 'center_x', 0.5) * fw)
                                cy = int(getattr(seg, 'center_y', 0.5) * fh)
                                (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(display_frame, (cx - 3, cy - th_ - 4), (cx + tw + 3, cy + 4), (0, 0, 0), -1)
                                cv2.putText(display_frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except Exception as e:
                self._render_error_count += 1
                if self._render_error_count <= 5:
                    print(f"  [Dashboard] RENDER ERROR #{self._render_error_count}: {e}")
                    traceback.print_exc()

        # --- Build canvas: frame | panel (top), input bar (bottom) ---
        pw = self._PANEL_W
        ibh = self._INPUT_BAR_H
        canvas_w = fw + pw
        canvas_h = fh + ibh
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Paste frame
        canvas[0:fh, 0:fw] = display_frame

        # --- Right panel ---
        panel = canvas[0:fh, fw:fw + pw]
        panel[:] = self._PANEL_BG

        y = 20
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Active prompts
        cv2.putText(panel, "ACTIVE PROMPTS", (8, y), font, 0.4, self._HEADER_COLOR, 1, cv2.LINE_AA)
        y += 5
        cv2.line(panel, (8, y), (pw - 8, y), (60, 60, 60), 1)
        y += 18

        active = self.get_active_prompts()
        effects = self.get_effects()
        if active:
            for p in sorted(active):
                fx = effects.get(p, {}).get("type", "")
                fx_str = f" [{fx}]" if fx else ""
                cv2.putText(panel, f"> {p}{fx_str}", (12, y), font, 0.4, self._ACTIVE_COLOR, 1, cv2.LINE_AA)
                y += 18
        else:
            cv2.putText(panel, "(none)", (12, y), font, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
            y += 18

        y += 10

        # Detected segments
        cv2.putText(panel, "DETECTED", (8, y), font, 0.4, self._HEADER_COLOR, 1, cv2.LINE_AA)
        y += 5
        cv2.line(panel, (8, y), (pw - 8, y), (60, 60, 60), 1)
        y += 18

        shown_segs = [s for s in segments if getattr(s, 'confidence', 1.0) >= self._confidence_min]
        if shown_segs:
            for i, seg in enumerate(shown_segs):
                label = getattr(seg, 'label', '') or getattr(seg, 'asset_class', '') or '?'
                conf = getattr(seg, 'confidence', 0.0)
                color = self._BRIGHT_COLORS[i % len(self._BRIGHT_COLORS)]
                text = f"{label} ({conf:.2f})"
                cv2.putText(panel, text, (12, y), font, 0.35, color, 1, cv2.LINE_AA)
                y += 16
        else:
            cv2.putText(panel, "(none)", (12, y), font, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
            y += 16

        y += 10

        # Voice Agent Status
        if voice_agent_state:
            listening = voice_agent_state.get('listening', False)
            recording = voice_agent_state.get('recording', False)
            last_resp = voice_agent_state.get('last_response', '')
            if listening or recording or last_resp:
                cv2.putText(panel, "VOICE AGENT", (8, y), font, 0.4, self._HEADER_COLOR, 1, cv2.LINE_AA)
                y += 5
                cv2.line(panel, (8, y), (pw - 8, y), (60, 60, 60), 1)
                y += 18

                if recording:
                    status = "RECORDING..."
                    status_color = (0, 255, 0)
                else:
                    status = "Listening"
                    status_color = (200, 200, 200)

                cv2.putText(panel, status, (12, y), font, 0.35, status_color, 1, cv2.LINE_AA)
                y += 16

                # Show partial transcript while recording
                if recording:
                    partial = voice_agent_state.get('partial_transcript', '')
                    if partial:
                        max_chars = 28
                        for i in range(0, len(partial), max_chars):
                            line = partial[i:i + max_chars]
                            cv2.putText(panel, f'"{line}"', (12, y), font, 0.3, (150, 255, 150), 1, cv2.LINE_AA)
                            y += 14

                # Show last command
                last_cmd = voice_agent_state.get('last_command', '')
                if last_cmd:
                    if len(last_cmd) > 25:
                        last_cmd = last_cmd[:25] + "..."
                    cv2.putText(panel, f'Cmd: {last_cmd}', (12, y), font, 0.3, (140, 200, 140), 1, cv2.LINE_AA)
                    y += 14

                # Show last response
                if last_resp:
                    if len(last_resp) > 25:
                        last_resp = last_resp[:25] + "..."
                    cv2.putText(panel, last_resp, (12, y), font, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
                    y += 14

                y += 10

        # Stats
        cv2.putText(panel, "STATS", (8, y), font, 0.4, self._HEADER_COLOR, 1, cv2.LINE_AA)
        y += 5
        cv2.line(panel, (8, y), (pw - 8, y), (60, 60, 60), 1)
        y += 18

        elapsed = time.time() - self._start_time
        ws_fps = self._frame_count / elapsed if elapsed > 0 else 0

        mts = self._mask_update_times
        if len(mts) >= 2:
            dt = mts[-1] - mts[0]
            mask_fps = (len(mts) - 1) / dt if dt > 0 else 0
        else:
            mask_fps = 0

        cv2.putText(panel, f"FPS: {ws_fps:.0f}", (12, y), font, 0.35, self._TEXT_COLOR, 1, cv2.LINE_AA)
        y += 16
        cv2.putText(panel, f"Mask: {mask_fps:.1f} fps", (12, y), font, 0.35, self._TEXT_COLOR, 1, cv2.LINE_AA)
        y += 16
        cv2.putText(panel, f"Segs: {len(shown_segs)}", (12, y), font, 0.35, self._TEXT_COLOR, 1, cv2.LINE_AA)
        y += 20

        # Recent commands log
        if self._input_log:
            cv2.putText(panel, "LOG", (8, y), font, 0.4, self._HEADER_COLOR, 1, cv2.LINE_AA)
            y += 5
            cv2.line(panel, (8, y), (pw - 8, y), (60, 60, 60), 1)
            y += 18
            for entry in self._input_log[-5:]:
                cv2.putText(panel, entry, (12, y), font, 0.3, (140, 140, 140), 1, cv2.LINE_AA)
                y += 14

        # --- Bottom input bar ---
        bar = canvas[fh:fh + ibh, 0:canvas_w]
        bar[:] = self._INPUT_BG
        cv2.line(canvas, (0, fh), (canvas_w, fh), (60, 60, 60), 1)

        cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
        input_display = f"> {self._input_text}{cursor}"
        cv2.putText(bar, input_display, (10, 27), font, 0.5, self._CURSOR_COLOR, 1, cv2.LINE_AA)

        seg_hint = "^P labels ON" if self._show_segments else "^P labels OFF"
        hint = f"Enter send | ESC clear | {seg_hint} | ^Q quit"
        (hw, _), _ = cv2.getTextSize(hint, font, 0.3, 1)
        cv2.putText(bar, hint, (canvas_w - hw - 10, 27), font, 0.3, (80, 80, 80), 1, cv2.LINE_AA)

        # Show window
        cv2.imshow(self.window_name, canvas)
        self._window_created = True

        # Poll keyboard
        self._poll_keys()

    def _mask_fingerprint(self, segments) -> str:
        """Content fingerprint for detecting mask changes."""
        if not segments:
            return ""
        parts = []
        for seg in segments:
            cx = getattr(seg, 'center_x', 0)
            cy = getattr(seg, 'center_y', 0)
            label = getattr(seg, 'label', '')
            parts.append(f"{cx:.4f},{cy:.4f},{label}")
            rle = getattr(seg, 'rle_mask', b'')
            if rle:
                parts.append(rle[:32].hex() if isinstance(rle, bytes) else str(rle[:32]))
        return "|".join(parts)

    def _poll_keys(self):
        """Handle keyboard input and window close."""
        for _ in range(20):
            key = cv2.waitKey(1) & 0xFF
            if key == 255 or key == 0:
                break
            self._process_key(key)

        # Check window close
        if self._window_created:
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    import os
                    os._exit(0)
            except cv2.error:
                pass

    def _process_key(self, key: int):
        """Handle a single keypress."""
        if key == 27:  # ESC — clear all prompts
            if self.on_command:
                self.on_command("clear")
            self._input_text = ""
        elif key == 16:  # Ctrl+P — toggle segment labels/outlines
            self._show_segments = not self._show_segments
            state = "ON" if self._show_segments else "OFF"
            self.log(f"Segments: {state}")
        elif key == 17:  # Ctrl+Q — quit
            self.close()
            import os
            os._exit(0)
        elif key == 13:  # Enter
            if self._input_text.strip():
                cmd = self._input_text.strip()
                self._input_log.append(cmd)
                if len(self._input_log) > 10:
                    self._input_log = self._input_log[-10:]
                if self.on_command:
                    self.on_command(cmd)
            self._input_text = ""
        elif key == 8 or key == 127:  # Backspace
            self._input_text = self._input_text[:-1]
        elif 32 <= key <= 126:  # Printable ASCII
            self._input_text += chr(key)

    def log(self, message: str):
        """Add a message to the command log."""
        self._input_log.append(message)
        if len(self._input_log) > 10:
            self._input_log = self._input_log[-10:]

    def close(self):
        """Close the dashboard window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
