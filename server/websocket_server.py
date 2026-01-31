"""WebSocket server for Quest <-> Python communication.

Replaces the TCP server from QuestPythonProcessor.
Uses protobuf for structured messaging instead of raw frame bytes.

60fps optimizations:
- process_frame() called directly (no thread pool for cached frames)
- Serialized protobuf response cached and reused when segments unchanged
- Only rebuilds protobuf when pipeline produces new data
"""
import asyncio
import time
import logging
import websockets
from server.proto import socialsense_pb2 as pb
from server.config import ServerConfig

logger = logging.getLogger(__name__)


class SocialSenseServer:
    """Main WebSocket server.

    Handles one client at a time (the Quest headset).
    Receives JPEG frames + audio, runs ML pipeline, returns protobuf results.
    """

    def __init__(self, config: ServerConfig, pipeline=None):
        self.config = config
        self.pipeline = pipeline
        self._client = None
        self._frame_count = 0
        self._start_time = None

        # Cached serialized response (reused when segments haven't changed)
        self._cached_response_bytes: bytes = b""
        self._cached_response = None  # pb.ServerMessage for debug view
        self._cached_segments_id: int = 0  # id() of the segments list

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
            await asyncio.Future()  # run forever

    async def _handle_client(self, websocket):
        """Handle a single Quest client connection."""
        remote = websocket.remote_address
        logger.info(f"Client connected: {remote}")
        self._client = websocket
        self._frame_count = 0
        self._start_time = time.time()
        self._cached_response_bytes = b""
        self._cached_response = None
        self._cached_segments_id = 0

        try:
            async for raw_message in websocket:
                await self._handle_message(websocket, raw_message)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Client disconnected: {e}")
        except Exception as e:
            logger.error(f"Client error: {e}", exc_info=True)
        finally:
            self._client = None
            logger.info("Client session ended")

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

        # Check if segments changed (by object identity — pipeline creates new
        # list only when SAM produces new data)
        seg_id = id(result.segments)
        if seg_id == self._cached_segments_id and self._cached_response_bytes:
            # Segments unchanged — reuse cached serialized bytes
            wall_ms = (time.perf_counter() - t_wall) * 1000
            if self._frame_count % 60 == 0:
                elapsed = time.time() - self._start_time
                fps = self._frame_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Frame {self._frame_count}: {wall_ms:.1f}ms wall (cached), "
                    f"{fps:.1f} fps"
                )
            # Show live feed even for cached frames (every 3rd frame to stay smooth)
            if self.config.debug_view and self._frame_count % 3 == 0 and self._cached_response:
                self._show_debug_view(msg.frame.jpeg_data, self._cached_response)
            return self._cached_response_bytes

        # Segments changed — rebuild protobuf response
        t0 = time.perf_counter()
        response = pb.ServerMessage()
        response.frame_id = msg.frame_id
        response.timestamp_ms = time.time() * 1000

        for seg in result.segments:
            proto_seg = response.segments.add()
            proto_seg.label = seg.label or ""
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

        # Conversation state
        conv_state = self.pipeline.get_conversation_state()
        if conv_state:
            response.conversation.summary = conv_state.get("summary", "")
            response.conversation.vibe = conv_state.get("vibe", "")
            response.conversation.recent_utterance = conv_state.get("recent_utterance", "")
            response.conversation.question = conv_state.get("question", "")
            response.conversation.is_other_speaking = conv_state.get("is_other_speaking", False)
            response.conversation.is_user_speaking = conv_state.get("is_user_speaking", False)

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

        if self._frame_count % 60 == 0 or self._frame_count <= 2:
            elapsed = time.time() - self._start_time
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Frame {self._frame_count}: {wall_ms:.1f}ms wall "
                f"(pipeline={result.total_ms:.1f}ms + proto={proto_ms:.1f}ms), "
                f"{len(response.segments)} segs, {fps:.1f} fps"
            )

        if self.config.debug_view:
            self._show_debug_view(msg.frame.jpeg_data, response)

        return response_bytes

    _BRIGHT_COLORS = [
        (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
        (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
        (255, 255, 255), (100, 255, 200),
    ]

    def _show_debug_view(self, jpeg_data: bytes, response: pb.ServerMessage):
        """Decode Quest frame and draw mask overlays in a cv2 window."""
        import numpy as np
        import cv2
        from server.encoding.rle import decode_rle

        buf = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            return
        frame = cv2.flip(frame, 0)

        h, w = frame.shape[:2]
        for i, seg in enumerate(response.segments):
            if not seg.rle_mask or seg.mask_width <= 0 or seg.mask_height <= 0:
                continue
            mask = decode_rle(seg.rle_mask, seg.mask_width, seg.mask_height)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_u8 = mask if mask.dtype == np.uint8 else (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            color = self._BRIGHT_COLORS[i % len(self._BRIGHT_COLORS)]
            cv2.drawContours(frame, contours, -1, color, 2)
            label = seg.label or seg.asset_class or ""
            if label:
                cx = int(seg.center_x * w)
                cy = int(seg.center_y * h)
                (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (cx - 3, cy - th_ - 4), (cx + tw + 3, cy + 4), (0, 0, 0), -1)
                cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Quest Debug View", frame)
        cv2.waitKey(1)

    def _empty_response(self, frame_id: int) -> bytes:
        response = pb.ServerMessage()
        response.frame_id = frame_id
        response.timestamp_ms = time.time() * 1000
        return response.SerializeToString()

    def _process_audio(self, msg: pb.ClientMessage):
        """Feed audio data to the audio pipeline."""
        if self.pipeline is not None and hasattr(self.pipeline, "process_audio"):
            self.pipeline.process_audio(
                msg.audio.pcm16_data,
                msg.audio.sample_rate,
                msg.audio.num_samples,
            )

    def _handle_control(self, msg: pb.ClientMessage):
        """Handle control commands from the client."""
        cmd = msg.control.command
        logger.info(f"Control command: {cmd}")
        if cmd == "reset" and self.pipeline is not None:
            self.pipeline.reset()
