"""Pipeline orchestrator — ties FastSAM + Gemini + RLE + audio together.

60 fps design:
- SAM runs continuously in a background thread (as fast as GPU allows)
- Gemini labeling runs in background thread every 3.0s
- process_frame() NEVER blocks — always returns cached result (<0.1ms)
- JPEG decode happens only in the SAM thread (not on every frame)
- Main thread just stores latest raw JPEG for SAM to pick up

The segmenter now contains the EXACT logic from sam_gemini_voice.py:
MediaPipe selfie/face/hands/pose + FastSAM + mask refinement + semantic labeling.

Interface expected by websocket_server.py:
    result = pipeline.process_frame(jpeg_data, width, height, frame_id)
    result.segments -> list of SegmentData
    pipeline.get_conversation_state() -> dict or None
    pipeline.process_audio(...) -> None
    pipeline.reset() -> None
"""

import time
import json
import logging
import threading
import cv2
import numpy as np

from server.vision.fastsam_segmenter import FastSAMSegmenter, SegmentData
from server.vision.gemini_labeler import GeminiLabeler
from server.encoding.rle import encode_rle
from server.config import ServerConfig

logger = logging.getLogger(__name__)


class PipelineResult:
    """Return value of process_frame()."""
    __slots__ = ("segments", "fastsam_ms", "gemini_ms", "total_ms")

    def __init__(self):
        self.segments: list[SegmentData] = []
        self.fastsam_ms: float = 0
        self.gemini_ms: float = 0
        self.total_ms: float = 0


class PipelineOrchestrator:
    """Runs the full vision + audio pipeline for one Quest client.

    SAM runs continuously in a background thread — as fast as the GPU allows.
    process_frame() returns cached results in <0.1ms for 60fps.
    """

    def __init__(self, config: ServerConfig):
        self.config = config

        # Vision components
        self._segmenter = FastSAMSegmenter(config)
        self._labeler = GeminiLabeler(config)

        # --- Continuous SAM thread ---
        self._latest_jpeg: tuple | None = None  # (jpeg_data, width, height)
        self._latest_jpeg_lock = threading.Lock()
        self._sam_thread: threading.Thread | None = None
        self._sam_stop = threading.Event()
        self._sam_count = 0

        # --- Gemini labeling (background, every 2s) ---
        self._gemini_interval = 2.0
        self._last_gemini_time = 0.0

        # --- Tracked masks (position-based matching, no velocity) ---
        self._tracks: dict[str, dict] = {}
        self._next_track_id = 0
        self._max_frames_missing = 3  # ~1.5s at 2fps SAM

        # --- Cached output (returned on every frame) ---
        self._cached_result: PipelineResult | None = None

        # --- Person mask (from MediaPipe, for Gemini person-awareness) ---
        self._person_mask: np.ndarray | None = None

        # --- Frame dimensions (cached from last decode) ---
        self._fw = 0
        self._fh = 0

        # Conversation state (populated by audio pipeline)
        self._conversation_state: dict = {}

        self._frame_count = 0
        self._initialized = False

        # Structured metrics log (JSONL)
        self._metrics_log = None
        if config.metrics_log_path:
            try:
                self._metrics_log = open(config.metrics_log_path, "a")
                logger.info(f"Metrics logging to {config.metrics_log_path}")
            except Exception as e:
                logger.warning(f"Cannot open metrics log: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self):
        logger.info("Initializing pipeline...")
        t0 = time.perf_counter()

        self._segmenter.initialize()
        t_seg = time.perf_counter()
        logger.info(f"  Segmenter init: {(t_seg - t0)*1000:.0f}ms")

        self._labeler.initialize()
        t_lbl = time.perf_counter()
        logger.info(f"  Gemini labeler init: {(t_lbl - t_seg)*1000:.0f}ms")

        self._initialized = True
        logger.info(f"Pipeline ready ({(t_lbl - t0)*1000:.0f}ms)")

    def reset(self):
        self._tracks.clear()
        self._next_track_id = 0
        self._cached_result = None
        self._conversation_state.clear()
        self._last_gemini_time = 0
        self._frame_count = 0
        self._sam_count = 0
        logger.info("Pipeline state reset")

    def shutdown(self):
        self._sam_stop.set()
        if self._sam_thread and self._sam_thread.is_alive():
            self._sam_thread.join(timeout=5)
        self._segmenter.shutdown()
        self._labeler.shutdown()
        if self._metrics_log:
            self._metrics_log.close()
            self._metrics_log = None
        self._initialized = False
        logger.info("Pipeline shut down")

    # ------------------------------------------------------------------
    # Frame processing — NEVER blocks, always returns cached result
    # ------------------------------------------------------------------

    def process_frame(
        self,
        jpeg_data: bytes,
        width: int,
        height: int,
        frame_id: int,
    ) -> PipelineResult:
        if not self._initialized:
            self.initialize()

        t0 = time.perf_counter()
        self._frame_count += 1

        # Store latest JPEG for SAM thread (no decode on hot path)
        with self._latest_jpeg_lock:
            self._latest_jpeg = (jpeg_data, width, height)

        # First frame — run SAM synchronously, then start continuous loop
        if self._cached_result is None:
            result = self._run_sam_sync(jpeg_data, width, height)
            result.total_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"Frame {frame_id}: {result.total_ms:.0f}ms (first) | "
                f"SAM={result.fastsam_ms:.0f}ms | {len(result.segments)} segs"
            )
            # Start continuous SAM loop
            self._start_sam_loop()
            return result

        # Return cached result (<0.1ms)
        result = PipelineResult()
        result.segments = self._cached_result.segments
        result.fastsam_ms = 0
        result.gemini_ms = 0
        result.total_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Synchronous SAM (first frame only)
    # ------------------------------------------------------------------

    def _decode_jpeg(self, jpeg_data: bytes, width: int, height: int):
        """Decode JPEG bytes to BGR numpy array."""
        buf = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None, 0, 0
        # Quest client sends vertically flipped frames — correct here
        frame_bgr = cv2.flip(frame_bgr, 0)
        fh, fw = frame_bgr.shape[:2]
        if (fw, fh) != (width, height) and width > 0 and height > 0:
            frame_bgr = cv2.resize(frame_bgr, (width, height))
            fh, fw = height, width
        return frame_bgr, fw, fh

    def _run_sam_sync(self, jpeg_data: bytes, width: int, height: int) -> PipelineResult:
        """Run full pipeline synchronously. Only used for the very first frame."""
        result = PipelineResult()

        frame_bgr, fw, fh = self._decode_jpeg(jpeg_data, width, height)
        if frame_bgr is None:
            return result

        self._fw = fw
        self._fh = fh

        t1 = time.perf_counter()
        new_segments = self._segmenter.segment_frame(frame_bgr)
        result.fastsam_ms = (time.perf_counter() - t1) * 1000
        self._sam_count = 1
        self._person_mask = getattr(self._segmenter, 'last_person_mask', None)

        # Kick Gemini
        now = time.time()
        self._last_gemini_time = now
        threading.Thread(
            target=self._background_label,
            args=(frame_bgr.copy(), list(new_segments), self._person_mask),
            daemon=True,
        ).start()

        # Track with fresh SAM segments (labels come from tracks/Gemini)
        self._update_tracks(new_segments, now, fh, fw)
        tracked = self._get_tracked_output(now, fh, fw)
        self._encode_rle_all(tracked, fw, fh)

        result.segments = tracked
        self._cached_result = result
        return result

    # ------------------------------------------------------------------
    # Continuous SAM loop (runs forever in background thread)
    # ------------------------------------------------------------------

    def _start_sam_loop(self):
        """Start the continuous SAM worker thread."""
        self._sam_stop.clear()
        self._sam_thread = threading.Thread(target=self._sam_loop, daemon=True)
        self._sam_thread.start()
        logger.info("Continuous SAM thread started")

    def _sam_loop(self):
        """Grab latest frame -> run SAM -> update cache -> repeat forever."""
        last_jpeg_id = None  # avoid re-processing same frame

        while not self._sam_stop.is_set():
            # Grab latest JPEG
            with self._latest_jpeg_lock:
                jpeg_data_tuple = self._latest_jpeg

            if jpeg_data_tuple is None:
                time.sleep(0.005)
                continue

            # Skip if same JPEG (no new frame from client)
            jpeg_id = id(jpeg_data_tuple)
            if jpeg_id == last_jpeg_id:
                time.sleep(0.005)
                continue
            last_jpeg_id = jpeg_id

            jpeg_data, width, height = jpeg_data_tuple

            try:
                t0 = time.perf_counter()

                # 1. Decode JPEG (only in SAM thread, not on hot path)
                frame_bgr, fw, fh = self._decode_jpeg(jpeg_data, width, height)
                if frame_bgr is None:
                    continue
                self._fw = fw
                self._fh = fh

                # 2. Run FastSAM (now includes MediaPipe + full original pipeline)
                new_segments = self._segmenter.segment_frame(frame_bgr)
                sam_ms = (time.perf_counter() - t0) * 1000
                self._sam_count += 1
                self._person_mask = getattr(self._segmenter, 'last_person_mask', None)

                # 3. Kick Gemini if due
                now = time.time()
                if now - self._last_gemini_time >= self._gemini_interval:
                    self._last_gemini_time = now
                    threading.Thread(
                        target=self._background_label,
                        args=(frame_bgr.copy(), list(new_segments), self._person_mask),
                        daemon=True,
                    ).start()

                # 4. Update tracks with NEW SAM segments (not old labelled ones).
                # Labels transfer via tracks: tracks carry labels from Gemini,
                # _get_tracked_output copies them onto the new segments.
                self._update_tracks(new_segments, now, fh, fw)
                tracked = self._get_tracked_output(now, fh, fw)

                # 6. Encode RLE
                self._encode_rle_all(tracked, fw, fh)

                # 7. Atomic cache update
                result = PipelineResult()
                result.segments = tracked
                result.fastsam_ms = sam_ms
                self._cached_result = result

                total_ms = (time.perf_counter() - t0) * 1000
                if self._sam_count % 10 == 0 or self._sam_count <= 3:
                    logger.info(
                        f"SAM #{self._sam_count}: {total_ms:.0f}ms total "
                        f"({sam_ms:.0f}ms SAM), {len(tracked)} segs"
                    )

                # Write structured metrics
                if self._metrics_log:
                    try:
                        labels = [s.label or "" for s in tracked]
                        metric = {
                            "ts": time.time(),
                            "sam_count": self._sam_count,
                            "sam_ms": round(sam_ms, 1),
                            "total_ms": round(total_ms, 1),
                            "segs": len(tracked),
                            "labels": labels,
                        }
                        self._metrics_log.write(json.dumps(metric) + "\n")
                        self._metrics_log.flush()
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"SAM loop error: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("SAM loop stopped")

    # ------------------------------------------------------------------
    # RLE encoding
    # ------------------------------------------------------------------

    def _encode_rle_all(self, segments: list, fw: int, fh: int):
        """Encode all segment masks to RLE at half resolution."""
        rle_w = fw // 2
        rle_h = fh // 2
        for seg in segments:
            if seg.mask is not None:
                small = cv2.resize(seg.mask, (rle_w, rle_h), interpolation=cv2.INTER_NEAREST)
                seg.rle_mask = encode_rle(small)
                seg.mask_width = rle_w
                seg.mask_height = rle_h

    # ------------------------------------------------------------------
    # Background Gemini labeling
    # ------------------------------------------------------------------

    def _background_label(self, frame_bgr: np.ndarray, segments: list, person_mask=None):
        """Run Gemini labeling, then propagate labels to tracks by position."""
        try:
            labelled = self._labeler.label_segments(frame_bgr, segments, person_mask=person_mask)

            # Apply labels directly to tracks (matched by position)
            for seg in labelled:
                if not seg.label or seg.label.startswith("~"):
                    continue
                best_tid = None
                best_dist = 0.25
                for tid, track in self._tracks.items():
                    dx = seg.center_x - track["center_x"]
                    dy = seg.center_y - track["center_y"]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
                if best_tid is not None:
                    self._tracks[best_tid]["label"] = seg.label
                    self._tracks[best_tid]["asset_class"] = seg.asset_class

            logger.info(f"Gemini labels applied to tracks ({len(labelled)} segments)")
        except Exception as e:
            logger.warning(f"Background labeling error: {e}")

    # ------------------------------------------------------------------
    # Tracked masks (position-based matching, no velocity)
    # ------------------------------------------------------------------

    def _update_tracks(self, segments: list, now: float, h: int, w: int):
        """Match new segments to existing tracks by centroid distance."""
        matched_track_ids = set()

        for seg_idx, seg in enumerate(segments):
            best_dist = 0.25  # 25% of frame — tolerates moderate movement between SAM frames
            best_tid = None

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                dx = seg.center_x - track["center_x"]
                dy = seg.center_y - track["center_y"]
                dist = (dx * dx + dy * dy) ** 0.5

                # Prefer matching same label
                if seg.label and track["label"] and seg.label == track["label"]:
                    dist *= 0.5

                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                track = self._tracks[best_tid]
                track["center_x"] = seg.center_x
                track["center_y"] = seg.center_y
                track["seg"] = seg
                track["last_seen"] = now
                track["frames_missing"] = 0
                if seg.label and not seg.label.startswith("~"):
                    track["label"] = seg.label
                    track["asset_class"] = seg.asset_class

                seg.track_id = best_tid
                matched_track_ids.add(best_tid)
            else:
                tid = f"seg_{self._next_track_id}"
                self._next_track_id += 1
                seg.track_id = tid
                self._tracks[tid] = {
                    "seg": seg,
                    "center_x": seg.center_x,
                    "center_y": seg.center_y,
                    "last_seen": now,
                    "frames_missing": 0,
                    "label": seg.label,
                    "asset_class": seg.asset_class,
                }
                matched_track_ids.add(tid)

        dead_ids = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track["frames_missing"] += 1
                if track["frames_missing"] > self._max_frames_missing:
                    dead_ids.append(tid)

        for tid in dead_ids:
            del self._tracks[tid]

    def _get_tracked_output(self, now: float, h: int, w: int) -> list:
        """Build output segment list from tracked segments (no interpolation)."""
        output: list[SegmentData] = []

        for tid, track in self._tracks.items():
            seg = track["seg"]

            # Apply cached labels from Gemini
            if track["label"] and not track["label"].startswith("~"):
                seg.label = track["label"]
                seg.asset_class = track["asset_class"]

            seg.track_id = tid
            output.append(seg)

        return output

    # ------------------------------------------------------------------
    # Audio pipeline (stub — Phase 4)
    # ------------------------------------------------------------------

    def process_audio(self, pcm16_data: bytes, sample_rate: int, num_samples: int):
        pass

    def get_conversation_state(self) -> dict:
        return self._conversation_state if self._conversation_state else {}
