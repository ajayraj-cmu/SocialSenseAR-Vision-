"""WebSocket server for Quest <-> Python communication.

Replaces the TCP server from QuestPythonProcessor.
Uses protobuf for structured messaging instead of raw frame bytes.

60fps optimizations:
- process_frame() called directly (no thread pool for cached frames)
- Serialized protobuf response cached and reused when segments unchanged
- Only rebuilds protobuf when pipeline produces new data

Dashboard:
- Always shows camera feed + mask overlays when client is connected
- Right panel: active prompts, detected segments, stats
- Bottom bar: type commands directly (blur/unblur/clear) — click window first
"""
import asyncio
import time
import logging
import websockets
from server.proto import socialsense_pb2 as pb
from server.config import ServerConfig

logger = logging.getLogger(__name__)


# Audio response messages use this prefix byte to distinguish from protobuf.
# Protobuf ServerMessage always starts with field tag 0x08+, so 0x01 is safe.
AUDIO_RESPONSE_PREFIX = b'\x01'


class SocialSenseServer:
    """Main WebSocket server.

    Handles one client at a time (the Quest headset).
    Receives JPEG frames + audio, runs ML pipeline, returns protobuf results.
    Shows a live dashboard window when frames are arriving.
    """

    def __init__(self, config: ServerConfig, pipeline=None):
        self.config = config
        self.pipeline = pipeline
        self._client = None
        self._frame_count = 0
        self._start_time = None

        # Cached serialized response (reused when segments haven't changed)
        self._cached_response_bytes: bytes = b""
        self._cached_response = None  # pb.ServerMessage for dashboard
        self._cached_segments_id: int = 0  # id() of the segments list
        self._last_screenshot_time: float = 0.0

        # Mask FPS tracking — timestamps of actual SAM mask updates
        self._mask_update_times: list[float] = []

        # Dashboard keyboard input state
        self._input_text: str = ""
        self._input_log: list[str] = []  # last N executed commands

    async def run(self):
        """Start the WebSocket server."""
        logger.info(f"Starting server on ws://{self.config.host}:{self.config.port}")

        async with websockets.serve(
            self._handle_client,
            self.config.host,
            self.config.port,
            max_size=self.config.max_message_size,
            ping_interval=20,
            ping_timeout=60,
        ):
            logger.info("Server ready. Waiting for Quest connection...")
            logger.info("Dashboard opens automatically when client connects.")
            if self.pipeline is not None:
                # Console fallback for when no cv2 window is focused
                logger.info("Console fallback: blur <obj>, unblur <obj>, clear, list, help")
                asyncio.create_task(self._console_command_loop())
            await asyncio.Future()  # run forever

    async def _handle_client(self, websocket):
        """Handle a single Quest client connection."""
        remote = websocket.remote_address
        logger.info(f"Client connected: {remote} (id={id(websocket):#x})")
        if self._client is not None:
            logger.warning(f"REPLACING existing client {id(self._client):#x} with new {id(websocket):#x}")
        self._client = websocket
        self._frame_count = 0
        self._start_time = time.time()
        self._cached_response_bytes = b""
        self._cached_response = None
        self._cached_segments_id = 0

        # Clear all effects and prompts from previous session
        if self.pipeline:
            self.pipeline.clear_effects()
            self.pipeline.set_active_prompts(set())
            # Also clear voice agent state if it exists
            if hasattr(self.pipeline, '_voice_agent') and self.pipeline._voice_agent:
                self.pipeline._voice_agent.clear_all_effects()
            logger.info("Cleared all effects, prompts, and voice agent state for new client session")

        # Start audio response sender (PersonaPlex → client)
        audio_task = asyncio.create_task(self._audio_response_loop(websocket))

        try:
            async for raw_message in websocket:
                await self._handle_message(websocket, raw_message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client disconnected: {e} code={e.code} reason='{e.reason}' frames={self._frame_count}")
        except Exception as e:
            logger.error(f"Client error: {e}", exc_info=True)
        finally:
            audio_task.cancel()
            self._client = None
            elapsed = time.time() - self._start_time
            logger.info(f"Client session ended: {self._frame_count} frames in {elapsed:.1f}s")

    async def _handle_message(self, websocket, raw_message: bytes):
        """Parse and dispatch a client message."""
        t_wall = time.perf_counter()

        msg = pb.ClientMessage()
        msg.ParseFromString(raw_message)

        payload_type = msg.WhichOneof("payload")

        if payload_type == "frame":
            response_bytes = self._process_frame_fast(msg, t_wall)
            await websocket.send(response_bytes)

        elif payload_type == "audio":
            self._process_audio(msg)

        elif payload_type == "control":
            self._handle_control(msg)

    def _process_frame_fast(self, msg: pb.ClientMessage, t_wall: float) -> bytes:
        """Process frame and return serialized protobuf bytes.

        Calls pipeline directly (no thread pool) since process_frame()
        returns cached results in <0.1ms. Caches serialized response
        when segments haven't changed.

        t_wall: perf_counter timestamp from when raw message was received.
        """
        self._frame_count += 1

        if self.pipeline is None:
            return self._empty_response(msg.frame_id)

        # Call pipeline directly — fast path returns cached result in ~0.1ms
        result = self.pipeline.process_frame(
            msg.frame.jpeg_data,
            msg.frame.width,
            msg.frame.height,
            msg.frame_id,
        )

        # Check if effects were explicitly cleared — must check BEFORE cache
        # so Unity gets the flag even when segments are unchanged
        effects_cleared = self.pipeline.consume_effects_cleared()
        if effects_cleared:
            logger.info("effects_cleared=True detected, will skip cache and send to Unity")

        # Check if segments changed (by object identity — pipeline creates new
        # list only when SAM produces new data)
        seg_id = id(result.segments)
        if seg_id == self._cached_segments_id and self._cached_response_bytes and not effects_cleared:
            # Segments unchanged and no clear command — reuse cached serialized bytes
            wall_ms = (time.perf_counter() - t_wall) * 1000
            if self._frame_count % 120 == 0:
                elapsed = time.time() - self._start_time
                fps = self._frame_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Frame {self._frame_count}: {wall_ms:.1f}ms wall (cached), "
                    f"{fps:.1f} fps"
                )
            # Show live feed even for cached frames (every 3rd frame to stay smooth)
            if self._frame_count % 3 == 0 and self._cached_response:
                self._show_dashboard(msg.frame.jpeg_data, self._cached_response)
            else:
                self._poll_keys()  # always service window + keyboard
            return self._cached_response_bytes

        # Segments changed — rebuild protobuf response
        t0 = time.perf_counter()
        response = pb.ServerMessage()
        response.frame_id = msg.frame_id
        response.masks_frame_id = result.masks_frame_id
        response.masks_updated = not getattr(result, 'decoder_skipped', True)
        response.timestamp_ms = time.time() * 1000

        # Set effects_cleared flag (already consumed above, before cache check)
        response.effects_cleared = effects_cleared
        if effects_cleared:
            logger.info("Sending effects_cleared=True to Unity")

        for seg in result.segments:
            proto_seg = response.segments.add()
            # Strip ~ prefix (internal "pending Gemini" marker)
            label = seg.label or ""
            proto_seg.label = label.lstrip("~")
            proto_seg.asset_class = seg.asset_class or ""
            proto_seg.confidence = seg.confidence
            proto_seg.bbox.x_min = seg.bbox[0]
            proto_seg.bbox.y_min = seg.bbox[1]
            proto_seg.bbox.x_max = seg.bbox[2]
            proto_seg.bbox.y_max = seg.bbox[3]
            proto_seg.rle_mask = seg.rle_mask or b""
            proto_seg.mask_width = seg.mask_width
            proto_seg.mask_height = seg.mask_height
            proto_seg.center_x = seg.center_x
            proto_seg.center_y = seg.center_y
            proto_seg.track_id = seg.track_id or ""

            if seg.emotion is not None:
                proto_seg.emotion.primary_emotion = seg.emotion.get("primary", "")
                proto_seg.emotion.display_label = seg.emotion.get("display", "")
                proto_seg.emotion.emoji = seg.emotion.get("emoji", "")
                proto_seg.emotion.confidence = seg.emotion.get("confidence", 0.0)

            # Voice agent effects
            if seg.effect is not None:
                proto_seg.effect.effect_type = seg.effect.effect_type
                proto_seg.effect.intensity = seg.effect.intensity
                # Only set color_hex if it's not empty (protobuf handles empty strings, but be explicit)
                if seg.effect.color_hex:
                    proto_seg.effect.color_hex = seg.effect.color_hex
                for k, v in seg.effect.params.items():
                    proto_seg.effect.params[k] = v

        # Conversation state
        conv_state = self.pipeline.get_conversation_state()
        if conv_state:
            response.conversation.summary = conv_state.get("summary", "")
            response.conversation.vibe = conv_state.get("vibe", "")
            response.conversation.recent_utterance = conv_state.get("recent_utterance", "")
            response.conversation.question = conv_state.get("question", "")
            response.conversation.is_other_speaking = conv_state.get("is_other_speaking", False)
            response.conversation.is_user_speaking = conv_state.get("is_user_speaking", False)

            # Voice agent state
            if "listening" in conv_state:
                response.conversation.voice_agent.listening = conv_state.get("listening", False)
                response.conversation.voice_agent.recording = conv_state.get("recording", False)
                response.conversation.voice_agent.partial_transcript = conv_state.get("partial_transcript", "")
                response.conversation.voice_agent.last_command = conv_state.get("last_command", "")
                response.conversation.voice_agent.last_response = conv_state.get("last_response", "")
                response.conversation.voice_agent.last_command_time = conv_state.get("last_command_time", 0.0)

        # Full-screen filter state
        if self.pipeline and hasattr(self.pipeline, '_voice_agent') and self.pipeline._voice_agent:
            fs_filter = self.pipeline._voice_agent.get_full_screen_filter()
            if fs_filter:
                response.conversation.voice_agent.full_screen_filter.filter_type = fs_filter.get("type", "none")
                response.conversation.voice_agent.full_screen_filter.intensity = fs_filter.get("intensity", 0.5)
                # NEW: Set color_hex for full-screen color filters
                color_hex = fs_filter.get("color_hex")
                if color_hex:
                    response.conversation.voice_agent.full_screen_filter.color_hex = color_hex

        # Metrics — report honest wall-clock time
        proto_ms = (time.perf_counter() - t0) * 1000
        wall_ms = (time.perf_counter() - t_wall) * 1000
        response.processing_time_ms = wall_ms
        response.metrics.total_ms = wall_ms
        response.metrics.segment_count = len(response.segments)
        response.metrics.fastsam_ms = result.fastsam_ms
        response.metrics.labeling_ms = result.gemini_ms

        # Serialize and cache
        response_bytes = response.SerializeToString()
        self._cached_response_bytes = response_bytes
        self._cached_response = response
        self._cached_segments_id = seg_id

        # Track mask update time for Mask FPS calculation
        now = time.perf_counter()
        self._mask_update_times.append(now)
        if len(self._mask_update_times) > 120:
            self._mask_update_times.pop(0)

        if self._frame_count % 120 == 0 or self._frame_count <= 2:
            elapsed = time.time() - self._start_time
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Frame {self._frame_count}: {wall_ms:.1f}ms wall "
                f"(pipeline={result.total_ms:.1f}ms + proto={proto_ms:.1f}ms), "
                f"{len(response.segments)} segs, {fps:.1f} fps"
            )

        self._show_dashboard(msg.frame.jpeg_data, response)

        return response_bytes

    # ------------------------------------------------------------------
    # Dashboard — always-on when client is connected
    # ------------------------------------------------------------------

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

    def _show_dashboard(self, jpeg_data: bytes, response: pb.ServerMessage):
        """Render full dashboard: camera + overlays + status panel + input bar."""
        import numpy as np
        import cv2
        from server.encoding.rle import decode_rle

        buf = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        frame = cv2.flip(frame, 0)

        fh, fw = frame.shape[:2]

        # --- Draw mask overlays on frame (skip low-confidence) ---
        for i, seg in enumerate(response.segments):
            if not seg.rle_mask or seg.mask_width <= 0 or seg.mask_height <= 0:
                continue
            if seg.confidence < self.config.display_confidence_min:
                continue
            mask = decode_rle(seg.rle_mask, seg.mask_width, seg.mask_height)
            if mask.shape[:2] != (fh, fw):
                mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_LINEAR)
            mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)

            # Apply visual effect if present
            effect_type = seg.effect.effect_type if seg.HasField("effect") else ""
            if effect_type == "blur":
                # Blur the region under the mask
                mask_bool = mask_u8 > 128
                blurred = cv2.GaussianBlur(frame, (51, 51), 0)
                frame[mask_bool] = blurred[mask_bool]

            contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            color = self._BRIGHT_COLORS[i % len(self._BRIGHT_COLORS)]
            cv2.drawContours(frame, contours, -1, (0, 0, 0), 5)
            cv2.drawContours(frame, contours, -1, color, 2)
            label = seg.label or seg.asset_class or ""
            # Show effect type in label
            if effect_type and effect_type != "none":
                label = f"{label} [{effect_type}]"
            if label:
                cx = int(seg.center_x * fw)
                cy = int(seg.center_y * fh)
                (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (cx - 3, cy - th_ - 4), (cx + tw + 3, cy + 4), (0, 0, 0), -1)
                cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- Build canvas: frame | panel (top), input bar (bottom) ---
        pw = self._PANEL_W
        ibh = self._INPUT_BAR_H
        canvas_w = fw + pw
        canvas_h = fh + ibh
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Paste frame
        canvas[0:fh, 0:fw] = frame

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

        active = set()
        effects = {}
        if self.pipeline is not None:
            active = self.pipeline.get_active_prompts()
            effects = self.pipeline.get_effects()
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

        shown_segs = [s for s in response.segments if s.confidence >= self.config.display_confidence_min]
        if shown_segs:
            for i, seg in enumerate(shown_segs):
                label = seg.label or seg.asset_class or "?"
                conf = seg.confidence
                color = self._BRIGHT_COLORS[i % len(self._BRIGHT_COLORS)]
                text = f"{label} ({conf:.2f})"
                cv2.putText(panel, text, (12, y), font, 0.35, color, 1, cv2.LINE_AA)
                y += 16
        else:
            cv2.putText(panel, "(none)", (12, y), font, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
            y += 16

        y += 10

        # Voice Agent Status
        if response.conversation.voice_agent.listening or response.conversation.voice_agent.recording:
            cv2.putText(panel, "VOICE AGENT", (8, y), font, 0.4, self._HEADER_COLOR, 1, cv2.LINE_AA)
            y += 5
            cv2.line(panel, (8, y), (pw - 8, y), (60, 60, 60), 1)
            y += 18

            if response.conversation.voice_agent.recording:
                status = "RECORDING..."
                status_color = (0, 255, 0)  # green
            else:
                status = "Listening"
                status_color = (200, 200, 200)

            cv2.putText(panel, status, (12, y), font, 0.35, status_color, 1, cv2.LINE_AA)
            y += 16

            if response.conversation.voice_agent.last_response:
                # Wrap long response text
                resp = response.conversation.voice_agent.last_response
                if len(resp) > 25:
                    resp = resp[:25] + "..."
                cv2.putText(panel, resp, (12, y), font, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
                y += 14

            y += 10

        # Stats
        cv2.putText(panel, "STATS", (8, y), font, 0.4, self._HEADER_COLOR, 1, cv2.LINE_AA)
        y += 5
        cv2.line(panel, (8, y), (pw - 8, y), (60, 60, 60), 1)
        y += 18

        elapsed = time.time() - self._start_time if self._start_time else 0
        ws_fps = self._frame_count / elapsed if elapsed > 0 else 0
        n_segs = len(response.segments)

        # Mask FPS from recent update timestamps
        mts = self._mask_update_times
        if len(mts) >= 2:
            dt = mts[-1] - mts[0]
            mask_fps = (len(mts) - 1) / dt if dt > 0 else 0
        else:
            mask_fps = 0

        cv2.putText(panel, f"WS: {ws_fps:.0f} fps", (12, y), font, 0.35, self._TEXT_COLOR, 1, cv2.LINE_AA)
        y += 16
        cv2.putText(panel, f"Mask: {mask_fps:.1f} fps", (12, y), font, 0.35, self._TEXT_COLOR, 1, cv2.LINE_AA)
        y += 16
        cv2.putText(panel, f"Segs: {n_segs}", (12, y), font, 0.35, self._TEXT_COLOR, 1, cv2.LINE_AA)
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

        # --- Bottom input bar (full width) ---
        bar = canvas[fh:fh + ibh, 0:canvas_w]
        bar[:] = self._INPUT_BG
        cv2.line(canvas, (0, fh), (canvas_w, fh), (60, 60, 60), 1)

        # Blinking cursor
        cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
        input_display = f"> {self._input_text}{cursor}"
        cv2.putText(bar, input_display, (10, 27), font, 0.5, self._CURSOR_COLOR, 1, cv2.LINE_AA)

        # Hint on right side of input bar
        hint = "Type command + Enter  |  ESC clear"
        (hw, _), _ = cv2.getTextSize(hint, font, 0.3, 1)
        cv2.putText(bar, hint, (canvas_w - hw - 10, 27), font, 0.3, (80, 80, 80), 1, cv2.LINE_AA)

        cv2.imshow("SocialSenseAR Server", canvas)

        # --- Drain all queued keyboard input (also processes window messages + close check) ---
        self._poll_keys()

        # Auto-save screenshot every 5 seconds
        now = time.time()
        if now - self._last_screenshot_time >= 5.0:
            self._last_screenshot_time = now
            import os
            ss_path = os.path.expanduser("~/Downloads/debug_view.png")
            cv2.imwrite(ss_path, canvas)

    def _poll_keys(self):
        """Drain all queued keyboard events and check for window close.

        Called on every frame (not just renders) so typing feels responsive
        and window X button is detected immediately.
        """
        import cv2
        for _ in range(20):  # cap to avoid infinite loop edge cases
            key = cv2.waitKey(1) & 0xFF
            if key == 255 or key == 0:
                break
            self._process_input_key(key)

        # Check if window was closed (X button)
        try:
            if cv2.getWindowProperty("SocialSenseAR Server", cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                import os
                os._exit(0)
        except cv2.error:
            import os
            os._exit(0)

    def _process_input_key(self, key: int):
        """Handle a single keypress from the dashboard cv2 window."""
        if key == 27:  # ESC — clear input
            self._input_text = ""
        elif key == 13:  # Enter — submit command
            if self._input_text.strip():
                self._execute_command(self._input_text.strip())
            self._input_text = ""
        elif key == 8 or key == 127:  # Backspace
            self._input_text = self._input_text[:-1]
        elif 32 <= key <= 126:  # printable ASCII
            self._input_text += chr(key)

    def _execute_command(self, raw_cmd: str):
        """Execute a typed command through the Gemini NLP pipeline."""
        if self.pipeline is None:
            return

        self._input_log.append(raw_cmd.strip())
        # Keep log bounded
        if len(self._input_log) > 10:
            self._input_log = self._input_log[-10:]

        self.pipeline.process_text_command(raw_cmd)

    # ------------------------------------------------------------------
    # Non-dashboard helpers
    # ------------------------------------------------------------------

    def _empty_response(self, frame_id: int) -> bytes:
        response = pb.ServerMessage()
        response.frame_id = frame_id
        response.timestamp_ms = time.time() * 1000
        return response.SerializeToString()

    _audio_msg_count = 0

    def _process_audio(self, msg: pb.ClientMessage):
        """Feed audio data to the audio pipeline."""
        if self.pipeline is not None and hasattr(self.pipeline, "process_audio"):
            self._audio_msg_count += 1
            if self._audio_msg_count == 1:
                logger.info(f"[Audio] First audio from client: {msg.audio.num_samples} samples, {msg.audio.sample_rate}Hz, {len(msg.audio.pcm16_data)}B")
            elif self._audio_msg_count % 500 == 0:
                logger.info(f"[Audio] {self._audio_msg_count} audio messages received from client")
            self.pipeline.process_audio(
                msg.audio.pcm16_data,
                msg.audio.sample_rate,
                msg.audio.num_samples,
            )

    async def _audio_response_loop(self, websocket):
        """Send PersonaPlex audio responses to client as separate WebSocket messages.

        Audio messages are prefixed with AUDIO_RESPONSE_PREFIX (0x01) to distinguish
        from protobuf ServerMessage responses (which start with field tag 0x08+).
        """
        audio_chunks_sent = 0
        total_bytes_sent = 0
        poll_count = 0
        try:
            while True:
                await asyncio.sleep(0.05)  # 50ms = 20 checks/sec
                poll_count += 1
                if self.pipeline is None:
                    continue
                audio_data = self.pipeline.get_audio_response()
                if audio_data:
                    audio_chunks_sent += 1
                    total_bytes_sent += len(audio_data)
                    if audio_chunks_sent == 1:
                        logger.info(f"[Audio] Sending first audio response to client: {len(audio_data)}B")
                    elif audio_chunks_sent % 100 == 0:
                        logger.info(f"[Audio] {audio_chunks_sent} chunks sent ({total_bytes_sent}B total)")
                    try:
                        await websocket.send(AUDIO_RESPONSE_PREFIX + audio_data)
                    except Exception:
                        break  # Connection closed
                elif poll_count == 200:  # Log once after ~10s if no audio seen
                    logger.info("[Audio] Response loop running but no audio received after 200 polls")
        except asyncio.CancelledError:
            logger.info(f"[Audio] Response loop ended: {audio_chunks_sent} chunks, {total_bytes_sent}B sent")

    def _handle_control(self, msg: pb.ClientMessage):
        """Handle control commands from the client via Gemini NLP pipeline."""
        raw = msg.control.command
        logger.info(f"Control command: {raw}")

        if self.pipeline is None:
            return

        # "reset" is a pipeline reset (not prompt clear)
        if raw.strip().lower() == "reset":
            self.pipeline.reset()
            return

        self.pipeline.process_text_command(raw)

    async def _console_command_loop(self):
        """Read commands from server console — fallback when cv2 window not focused."""
        from server.commands import CommandInput

        cmd_input = CommandInput()
        cmd_input.start()

        try:
            while True:
                await asyncio.sleep(0.2)
                for raw_cmd in cmd_input.get_commands():
                    self.pipeline.process_text_command(raw_cmd)
        except asyncio.CancelledError:
            cmd_input.stop()
