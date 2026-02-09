"""Pipeline orchestrator — ties FastSAM + Gemini + RLE + audio together.

60 fps design:
- SAM runs continuously in a background thread (as fast as GPU allows)
- Gemini labeling runs in background thread every 3.0s
- process_frame() pre-decodes JPEG (~2ms) then returns cached result
- JPEG decode overlaps with SAM GPU work (effectively free)
- SAM thread picks up pre-decoded BGR frames (no decode overhead)

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

from server.vision.segment_data import SegmentData
from server.encoding.rle import encode_rle
from server.config import ServerConfig

logger = logging.getLogger(__name__)


class PipelineResult:
    """Return value of process_frame()."""
    __slots__ = ("segments", "fastsam_ms", "gemini_ms", "total_ms",
                 "masks_frame_id", "decoder_skipped")

    def __init__(self):
        self.segments: list[SegmentData] = []
        self.fastsam_ms: float = 0
        self.gemini_ms: float = 0
        self.total_ms: float = 0
        self.masks_frame_id: int = 0  # frame_id when SAM decoder last ran
        self.decoder_skipped: bool = False  # True when SAM returned cached masks


class PipelineOrchestrator:
    """Runs the full vision + audio pipeline for one Quest client.

    SAM runs continuously in a background thread — as fast as the GPU allows.
    process_frame() returns cached results in <0.1ms for 60fps.
    """

    def __init__(self, config: ServerConfig):
        self.config = config

        # Vision components — conditional on pipeline mode
        if config.pipeline_mode == "grounded-sam2":
            from server.vision.grounded_sam2_segmenter import GroundedSAM2Segmenter
            self._segmenter = GroundedSAM2Segmenter(config)
            self._labeler = None  # GroundingDINO provides labels
            logger.info("Pipeline mode: Grounded SAM2 (GroundingDINO + SAM2)")
        elif config.pipeline_mode == "sam3":
            from server.vision.sam3_segmenter import SAM3Segmenter
            self._segmenter = SAM3Segmenter(config)
            self._labeler = None  # SAM3 text prompts ARE labels
            logger.info("Pipeline mode: SAM3 (text-prompted, no Gemini)")
        else:
            from server.vision.fastsam_segmenter import FastSAMSegmenter
            from server.vision.gemini_labeler import GeminiLabeler
            self._segmenter = FastSAMSegmenter(config)
            self._labeler = GeminiLabeler(config)
            logger.info("Pipeline mode: Legacy (FastSAM + MediaPipe + Gemini)")

        # Audio / Voice Agent components
        self._voice_agent = None
        if config.audio_enabled:
            from server.audio.voice_agent import VoiceAgent, VoiceCommandPlanner
            from server.vision.gemini_scene_understanding import GeminiSceneUnderstanding

            # Conditional transcriber: local (faster-whisper) or cloud (OpenAI)
            backend = getattr(config, 'transcriber_backend', 'local')
            if backend == "local":
                from server.audio.local_transcriber import LocalTranscriber
                self._transcriber = LocalTranscriber(
                    listening_model=getattr(config, 'whisper_listening_model', 'tiny.en'),
                    recording_model=getattr(config, 'whisper_recording_model', 'base.en'),
                )
                logger.info("Transcriber: local (faster-whisper)")
            else:
                from server.audio.transcriber import WhisperTranscriber
                self._transcriber = WhisperTranscriber(
                    api_key=config.openai_api_key,
                    model=config.whisper_model,
                )
                logger.info("Transcriber: cloud (OpenAI Whisper)")

            self._voice_planner = VoiceCommandPlanner(api_key=config.gemini_api_key)
            self._scene_understanding = GeminiSceneUnderstanding(api_key=config.gemini_api_key)

            self._voice_agent = VoiceAgent(
                transcriber=self._transcriber,
                planner=self._voice_planner,
                scene_understanding=self._scene_understanding,
                config=config,
            )
            self._voice_agent.set_on_command_callback(self._on_voice_command)
            logger.info("Voice agent pipeline created (will initialize on first frame)")
        else:
            logger.info("Audio disabled — voice agent not created")

        # --- Continuous SAM thread ---
        # Pre-decoded BGR frame from ws thread (overlaps decode with SAM GPU work)
        self._latest_frame: tuple | None = None  # (frame_bgr, fw, fh, frame_id)
        self._latest_frame_lock = threading.Lock()
        self._new_frame_event = threading.Event()  # Signal SAM loop on new frame
        self._sam_thread: threading.Thread | None = None
        self._sam_stop = threading.Event()
        self._sam_count = 0
        self._sam_total_ms_accum = 0.0
        self._sam_ms_accum = 0.0

        # --- Gemini labeling (background, every 2s) ---
        self._gemini_interval = 2.0
        self._last_gemini_time = 0.0

        # --- Tracked masks (position-based matching, no velocity) ---
        self._tracks: dict[str, dict] = {}
        self._next_track_id = 0
        self._track_max_age = 1.0  # seconds — tracks expire 1s after last match

        # --- Cached output (returned on every frame) ---
        self._cached_result: PipelineResult | None = None

        # --- Person mask (from MediaPipe, for Gemini person-awareness) ---
        self._person_mask: np.ndarray | None = None

        # --- Frame dimensions (cached from last decode) ---
        self._fw = 0
        self._fh = 0

        # Conversation state (populated by audio pipeline)
        self._conversation_state: dict = {}

        # --- Effect registry (works without voice agent) ---
        # Maps label -> {"type": "blur", "intensity": 1.0}
        self._effect_registry: dict[str, dict] = {}
        self._effect_lock = threading.Lock()

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

        if self._labeler is not None:
            self._labeler.initialize()
            t_lbl = time.perf_counter()
            logger.info(f"  Gemini labeler init: {(t_lbl - t_seg)*1000:.0f}ms")
        else:
            t_lbl = t_seg

        # Initialize voice agent components
        if self._voice_agent is not None:
            self._transcriber.initialize()
            self._voice_planner.initialize()
            self._scene_understanding.initialize()
            self._voice_agent.start()
            t_voice = time.perf_counter()
            logger.info(f"  Voice agent init: {(t_voice - t_lbl)*1000:.0f}ms")
        else:
            t_voice = t_lbl

        self._initialized = True
        logger.info(f"Pipeline ready ({(t_voice - t0)*1000:.0f}ms)")

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
        if self._labeler is not None:
            self._labeler.shutdown()
        if self._voice_agent is not None:
            self._voice_agent.shutdown()
            self._transcriber.shutdown()
            self._voice_planner.shutdown()
            self._scene_understanding.shutdown()
        if self._metrics_log:
            self._metrics_log.close()
            self._metrics_log = None
        self._initialized = False
        logger.info("Pipeline shut down")

    # ------------------------------------------------------------------
    # Dynamic prompt control
    # ------------------------------------------------------------------

    def set_active_prompts(self, prompts: set[str]):
        """Replace active prompts. Clears tracks for removed labels."""
        old = self._segmenter.get_active_prompts()
        self._segmenter.set_active_prompts(prompts)
        removed = old - prompts
        if removed:
            self._clear_tracks_for_labels(removed)

    def add_prompt(self, prompt: str):
        """Add a single prompt to the active set."""
        self._segmenter.add_prompt(prompt)

    def remove_prompt(self, prompt: str):
        """Remove a single prompt. Clears matching tracks."""
        self._segmenter.remove_prompt(prompt)
        self._clear_tracks_for_labels({prompt})

    def get_active_prompts(self) -> set[str]:
        """Return current active prompts."""
        return self._segmenter.get_active_prompts()

    def _clear_tracks_for_labels(self, labels: set[str]):
        """Remove tracks whose label matches any in the given set."""
        to_remove = [tid for tid, track in self._tracks.items()
                     if track.get("label") in labels]
        for tid in to_remove:
            del self._tracks[tid]
        if to_remove:
            # Replace cached result with empty segments instead of setting to None.
            # Setting to None would trigger _run_sam_sync_decoded + _start_sam_loop
            # on the next process_frame, spawning a SECOND SAM loop thread that
            # shares TRT contexts with the first — causing corruption.
            if self._cached_result is not None:
                result = PipelineResult()
                result.masks_frame_id = self._cached_result.masks_frame_id
                result.decoder_skipped = True
                self._cached_result = result
            logger.info(f"Cleared {len(to_remove)} tracks for removed prompts: {labels}")

    # ------------------------------------------------------------------
    # Effect registry (typed commands: blur/unblur)
    # ------------------------------------------------------------------

    def set_effect(self, label: str, effect_type: str, intensity: float = 1.0, invert: bool = False):
        """Apply an effect to a label (e.g., blur laptop).

        If invert=True, SAM3 segments the target but the effect applies to
        everything OUTSIDE the mask (e.g., "blur everything but laptop").
        """
        with self._effect_lock:
            self._effect_registry[label] = {
                "type": effect_type,
                "intensity": min(1.0, max(0.0, intensity)),
                "invert": invert,
            }
        mode = " (inverted)" if invert else ""
        logger.info(f"Effect set: {label} -> {effect_type} ({intensity:.1f}){mode}")

    def remove_effect(self, label: str):
        """Remove effect from a label."""
        with self._effect_lock:
            if label in self._effect_registry:
                del self._effect_registry[label]
                logger.info(f"Effect removed: {label}")

    def clear_effects(self):
        """Clear all effects."""
        with self._effect_lock:
            self._effect_registry.clear()
        logger.info("All effects cleared")

    def get_effects(self) -> dict[str, dict]:
        """Get copy of active effects."""
        with self._effect_lock:
            return self._effect_registry.copy()

    # ------------------------------------------------------------------
    # Frame processing — pre-decodes JPEG, returns cached result
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

        # Pre-decode JPEG on ws thread (overlaps with SAM GPU work on prev frame)
        frame_bgr, fw, fh = self._decode_jpeg(jpeg_data, width, height)
        if frame_bgr is None:
            if self._cached_result is not None:
                result = PipelineResult()
                result.segments = self._cached_result.segments
                result.masks_frame_id = self._cached_result.masks_frame_id
                result.total_ms = (time.perf_counter() - t0) * 1000
                return result
            return PipelineResult()

        with self._latest_frame_lock:
            self._latest_frame = (frame_bgr, fw, fh, frame_id)
        self._new_frame_event.set()  # Wake up SAM loop immediately

        # First frame — run SAM synchronously, then start continuous loop
        if self._cached_result is None:
            result = self._run_sam_sync_decoded(frame_bgr, fw, fh)
            result.masks_frame_id = frame_id
            result.total_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"Frame {frame_id}: {result.total_ms:.0f}ms (first) | "
                f"SAM={result.fastsam_ms:.0f}ms | {len(result.segments)} segs"
            )
            # Start continuous SAM loop (guard against duplicate threads)
            if self._sam_thread is None or not self._sam_thread.is_alive():
                self._start_sam_loop()
            return result

        # Return cached result (<0.1ms)
        result = PipelineResult()
        result.segments = self._cached_result.segments
        result.masks_frame_id = self._cached_result.masks_frame_id
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

    def _run_sam_sync_decoded(self, frame_bgr: np.ndarray, fw: int, fh: int) -> PipelineResult:
        """Run full pipeline synchronously on pre-decoded frame. First frame only."""
        result = PipelineResult()

        self._fw = fw
        self._fh = fh

        t1 = time.perf_counter()
        new_segments = self._segmenter.segment_frame(frame_bgr)
        result.fastsam_ms = (time.perf_counter() - t1) * 1000
        self._sam_count = 1
        self._person_mask = getattr(self._segmenter, 'last_person_mask', None)

        # Kick Gemini (legacy mode only — SAM3 segments arrive pre-labeled)
        now = time.time()
        if self._labeler is not None:
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
        """Grab latest pre-decoded frame -> run SAM -> update cache -> repeat."""
        last_frame_id = None  # avoid re-processing same frame

        while not self._sam_stop.is_set():
            # Grab latest pre-decoded frame (decoded on ws thread)
            with self._latest_frame_lock:
                frame_tuple = self._latest_frame

            if frame_tuple is None:
                self._new_frame_event.wait(timeout=0.1)
                self._new_frame_event.clear()
                continue

            # Skip if same frame (no new frame from client)
            frame_id = id(frame_tuple)
            if frame_id == last_frame_id:
                # Wait for new frame signal instead of polling with sleep
                self._new_frame_event.wait(timeout=0.1)
                self._new_frame_event.clear()
                continue
            last_frame_id = frame_id

            frame_bgr, fw, fh, sam_frame_id = frame_tuple

            try:
                t0 = time.perf_counter()

                self._fw = fw
                self._fh = fh

                # 1. Run SAM segmenter (frame already decoded on ws thread)
                new_segments = self._segmenter.segment_frame(frame_bgr)
                sam_ms = (time.perf_counter() - t0) * 1000
                self._sam_count += 1
                self._person_mask = getattr(self._segmenter, 'last_person_mask', None)

                # 2. Check if SAM ran fresh vision encoding.
                # decoder_skipped=True when vision didn't run — masks are from a
                # previous vision frame and should NOT trigger a Quest overlay update.
                # This prevents stale masks (decoded with old vision features) from
                # being projected with the wrong capture rotation.
                vision_ran = getattr(self._segmenter, '_vision_ran', False)
                decoder_skipped = not vision_ran

                # 3. Kick Gemini if due (legacy mode only — SAM3 is pre-labeled)
                now = time.time()
                if self._labeler is not None and now - self._last_gemini_time >= self._gemini_interval:
                    self._last_gemini_time = now
                    threading.Thread(
                        target=self._background_label,
                        args=(frame_bgr.copy(), list(new_segments), self._person_mask),
                        daemon=True,
                    ).start()

                # 4. Update tracks with segments (fresh or flow-tracked).
                self._update_tracks(new_segments, now, fh, fw)
                tracked = self._get_tracked_output(now, fh, fw)

                # 5. Apply effects (typed commands + voice agent) to tracked segments
                self._apply_effects(tracked)

                # 6. Sync voice agent known objects with SAM3 (register segments as known)
                if self._voice_agent is not None and self.config.pipeline_mode == "sam3":
                    for seg in tracked:
                        if seg.label and not seg.label.startswith("~"):
                            self._voice_agent.add_known_object(seg.label)

                    # Update voice agent with latest frame for Gemini Vision
                    self._voice_agent.update_frame(frame_bgr)

                # 7. Encode RLE
                self._encode_rle_all(tracked, fw, fh)

                # 8. Atomic cache update
                result = PipelineResult()
                result.segments = tracked
                result.fastsam_ms = sam_ms
                result.decoder_skipped = decoder_skipped
                # Only update masks_frame_id when vision encoder ran (fresh features).
                # When vision was skipped, masks correspond to the previous vision
                # frame — keep old ID so Quest uses the correct capture rotation.
                if vision_ran:
                    result.masks_frame_id = sam_frame_id
                elif self._cached_result is not None:
                    result.masks_frame_id = self._cached_result.masks_frame_id
                else:
                    result.masks_frame_id = sam_frame_id
                self._cached_result = result

                total_ms = (time.perf_counter() - t0) * 1000
                self._sam_total_ms_accum += total_ms
                self._sam_ms_accum += sam_ms

                # Frequent centroid log (every 30 iters ~4s) to diagnose mask movement
                if self._sam_count % 150 == 0 and self._sam_count > 0:
                    centroid_info = [(t.track_id, t.label,
                                      f"cx={t.center_x:.3f}", f"cy={t.center_y:.3f}",
                                      f"rle={len(t.rle_mask) if t.rle_mask else 0}B")
                                     for t in tracked]
                    logger.info(
                        f"SAM #{self._sam_count} mask-diag: {len(tracked)} segs | "
                        f"fid={sam_frame_id} | centroids={centroid_info}"
                    )

                if self._sam_count % 120 == 0 and self._sam_count > 0:
                    avg_total = self._sam_total_ms_accum / 120
                    avg_sam = self._sam_ms_accum / 120
                    self._sam_total_ms_accum = 0.0
                    self._sam_ms_accum = 0.0
                    track_info = [(t.track_id, t.label) for t in tracked]
                    logger.info(
                        f"SAM #{self._sam_count}: avg {avg_total:.0f}ms total "
                        f"({avg_sam:.0f}ms SAM), {len(tracked)} segs | {track_info}"
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
        """Encode all segment masks to RLE at 3/4 resolution with smooth edges."""
        rle_w = int(fw * 0.75)
        rle_h = int(fh * 0.75)
        for seg in segments:
            if seg.mask is not None:
                mask = seg.mask
                # Handle both uint8 [0,255] and float32 [0,1] masks
                if mask.dtype == np.uint8:
                    small_u8 = cv2.resize(mask, (rle_w, rle_h), interpolation=cv2.INTER_LINEAR)
                else:
                    small = cv2.resize(mask, (rle_w, rle_h), interpolation=cv2.INTER_LINEAR)
                    small_u8 = (small * 255).astype(np.uint8)
                # Smooth mask edges before encoding
                kernel = np.ones((5, 5), np.uint8)
                small_u8 = cv2.morphologyEx(small_u8, cv2.MORPH_CLOSE, kernel)
                small_u8 = cv2.morphologyEx(small_u8, cv2.MORPH_OPEN, kernel)
                small_u8 = cv2.GaussianBlur(small_u8, (5, 5), 0)
                seg.rle_mask = encode_rle((small_u8 > 128).astype(np.uint8))
                seg.mask_width = rle_w
                seg.mask_height = rle_h

    # ------------------------------------------------------------------
    # Background Gemini labeling
    # ------------------------------------------------------------------

    def _background_label(self, frame_bgr: np.ndarray, segments: list, person_mask=None):
        """Run Gemini labeling, then propagate labels to tracks by position + area."""
        try:
            labelled = self._labeler.label_segments(frame_bgr, segments, person_mask=person_mask)

            # Apply labels to tracks using cost-matrix matching (same as _update_tracks)
            valid_segs = [s for s in labelled if s.label and not s.label.startswith("~")]
            if valid_segs and self._tracks:
                track_ids = list(self._tracks.keys())
                pairs = []
                for si, seg in enumerate(valid_segs):
                    s_area = self._mask_area_norm(seg.mask)
                    for ti, tid in enumerate(track_ids):
                        track = self._tracks[tid]
                        dx = seg.center_x - track["center_x"]
                        dy = seg.center_y - track["center_y"]
                        dist = (dx * dx + dy * dy) ** 0.5

                        t_area = track.get("area", 0.0)
                        if t_area > 0 and s_area > 0:
                            area_ratio = min(s_area, t_area) / max(s_area, t_area)
                        else:
                            area_ratio = 0.5
                        cost = dist * (2.0 - area_ratio)
                        pairs.append((cost, si, ti))

                pairs.sort()
                matched_s = set()
                matched_t = set()
                for cost, si, ti in pairs:
                    if si in matched_s or ti in matched_t:
                        continue
                    if cost > 0.20:  # Slightly more lenient for Gemini (async, positions shift)
                        break
                    tid = track_ids[ti]
                    seg = valid_segs[si]
                    track = self._tracks[tid]
                    old_label = track.get("label", "")
                    new_label = seg.label

                    # MediaPipe body labels are ground truth — never overwrite
                    _BODY_LABELS = {"person", "face", "left_hand", "right_hand",
                                    "left_arm", "right_arm", "torso", "left_leg", "right_leg"}
                    if old_label in _BODY_LABELS:
                        continue

                    # Label persistence: once confirmed, require 2 consecutive
                    # votes for a DIFFERENT label before changing it
                    if old_label and not old_label.startswith("~") and new_label != old_label:
                        # Track how many times this new label was proposed
                        pending = track.get("_pending_label", "")
                        count = track.get("_pending_count", 0)
                        if new_label == pending:
                            count += 1
                        else:
                            pending = new_label
                            count = 1
                        track["_pending_label"] = pending
                        track["_pending_count"] = count
                        if count >= 2:
                            # Accept the new label after 2 consecutive votes
                            track["label"] = new_label
                            track["asset_class"] = seg.asset_class
                            track["_pending_label"] = ""
                            track["_pending_count"] = 0
                    else:
                        # First label or same label — accept immediately
                        track["label"] = new_label
                        track["asset_class"] = seg.asset_class
                        track["_pending_label"] = ""
                        track["_pending_count"] = 0

                    matched_s.add(si)
                    matched_t.add(ti)

            logger.info(f"Gemini labels applied to tracks ({len(labelled)} segments)")
        except Exception as e:
            logger.warning(f"Background labeling error: {e}")

    # ------------------------------------------------------------------
    # Tracked masks (position-based matching, no velocity)
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_area_norm(mask) -> float:
        """Compute normalized mask area (0-1) quickly."""
        if mask is None:
            return 0.0
        if mask.dtype == np.uint8:
            return cv2.countNonZero(mask) / max(1, mask.shape[0] * mask.shape[1])
        return float(np.count_nonzero(mask > 0.5)) / max(1, mask.shape[0] * mask.shape[1])

    def _update_tracks(self, segments: list, now: float, h: int, w: int):
        """Match new segments to existing tracks using cost-matrix matching.

        Uses global cost minimization instead of greedy per-segment matching
        to prevent order-dependent track swaps (the main cause of flickering).
        Cost = distance * area_penalty, with label bonus for same-label matches.
        """
        matched_seg_indices = set()
        matched_track_ids = set()

        if segments and self._tracks:
            track_ids = list(self._tracks.keys())

            # Compute areas for new segments
            seg_areas = [self._mask_area_norm(seg.mask) for seg in segments]

            # Build all (cost, raw_dist, seg_index, track_index) pairs
            # Raw distance gates acceptance; cost (with area/label adjustments) ranks matches
            pairs = []
            for si, seg in enumerate(segments):
                for ti, tid in enumerate(track_ids):
                    track = self._tracks[tid]
                    dx = seg.center_x - track["center_x"]
                    dy = seg.center_y - track["center_y"]
                    dist = (dx * dx + dy * dy) ** 0.5

                    # Skip obviously impossible matches early
                    if dist > 0.25:
                        continue

                    # Area similarity (for ranking only — not gating)
                    t_area = track.get("area", 0.0)
                    s_area = seg_areas[si]
                    if t_area > 0 and s_area > 0:
                        area_ratio = min(s_area, t_area) / max(s_area, t_area)
                    else:
                        area_ratio = 0.5

                    # Cost for ranking: distance + mild area weighting + label bonus
                    cost = dist * (1.5 - 0.5 * area_ratio)  # 1.0 (same size) to 1.5 (different)

                    # Label bonus — matching labels strongly prefer each other
                    if seg.label and track["label"] and seg.label == track["label"]:
                        cost *= 0.3

                    pairs.append((cost, dist, si, ti))

            # Sort by cost (lowest first) and greedily assign
            pairs.sort()
            max_dist = 0.20  # Gate on raw distance — 20% of frame diagonal

            for cost, dist, si, ti in pairs:
                if si in matched_seg_indices or ti in matched_track_ids:
                    continue
                if dist > max_dist:
                    continue  # Don't break — a later pair may have lower raw dist

                tid = track_ids[ti]
                seg = segments[si]
                track = self._tracks[tid]

                track["center_x"] = seg.center_x
                track["center_y"] = seg.center_y
                track["area"] = seg_areas[si]
                track["last_seen"] = now

                # Always use the fresh mask from SAM — no temporal smoothing.
                # Old logic kept the previous mask when IoU > 0.65, which prevented
                # jitter on static objects but made masks feel stuck during movement.
                track["seg"] = seg
                # Only set label if track has no confirmed label yet.
                # Gemini (_background_label) is the authority for label updates.
                if not track.get("label") or track["label"].startswith("~"):
                    if seg.label and not seg.label.startswith("~"):
                        track["label"] = seg.label
                        track["asset_class"] = seg.asset_class

                seg.track_id = tid
                matched_seg_indices.add(si)
                matched_track_ids.add(ti)

        # Create new tracks for unmatched segments
        for si, seg in enumerate(segments):
            if si in matched_seg_indices:
                continue
            tid = f"seg_{self._next_track_id}"
            self._next_track_id += 1
            seg.track_id = tid
            self._tracks[tid] = {
                "seg": seg,
                "center_x": seg.center_x,
                "center_y": seg.center_y,
                "area": self._mask_area_norm(seg.mask),
                "last_seen": now,
                "label": seg.label,
                "asset_class": seg.asset_class,
            }

        # Expire old tracks (time-based — independent of SAM framerate)
        dead_ids = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids and tid not in {s.track_id for s in segments}:
                if now - track["last_seen"] > self._track_max_age:
                    dead_ids.append(tid)

        for tid in dead_ids:
            del self._tracks[tid]

    def _get_tracked_output(self, now: float, h: int, w: int) -> list:
        """Build output segment list from tracked segments, sorted by track_id."""
        output: list[SegmentData] = []
        stale_threshold = 0.5  # Only show tracks seen within last 500ms

        for tid, track in self._tracks.items():
            # Skip stale tracks (keep them alive for re-matching, but don't render)
            if now - track["last_seen"] > stale_threshold:
                continue

            seg = track["seg"]

            # Apply cached labels from Gemini
            if track["label"] and not track["label"].startswith("~"):
                seg.label = track["label"]
                seg.asset_class = track["asset_class"]

            seg.track_id = tid
            output.append(seg)

        # Sort by track_id for consistent ordering (prevents color index jumps)
        output.sort(key=lambda s: s.track_id)
        return output

    # ------------------------------------------------------------------
    # Voice Agent Effects Application
    # ------------------------------------------------------------------

    def _apply_effects(self, segments: list):
        """Apply effects to tracked segments.

        Combines effects from:
        1. Effect registry (typed commands: blur/unblur)
        2. Voice agent (if enabled)

        Sets EffectData on each segment.
        """
        from server.vision.segment_data import EffectData

        # Merge effect sources: registry + voice agent
        active_effects = self.get_effects()  # from typed commands

        # Voice agent effects override/extend typed commands
        if self._voice_agent is not None:
            voice_effects = self._voice_agent.get_active_effects()
            active_effects.update(voice_effects)

        if not active_effects:
            return

        for seg in segments:
            if seg.label and seg.label in active_effects:
                fx = active_effects[seg.label]
                params = {}
                if fx.get("invert", False):
                    params["invert"] = "true"
                seg.effect = EffectData(
                    effect_type=fx.get("type", "none"),
                    intensity=fx.get("intensity", 1.0),
                    params=params,
                )

    # ------------------------------------------------------------------
    # Audio pipeline — Voice Agent
    # ------------------------------------------------------------------

    def process_audio(self, pcm16_data: bytes, sample_rate: int, num_samples: int):
        """Feed audio to voice agent for transcription + command processing.

        Called from websocket thread. Audio processing happens asynchronously.
        """
        if self._voice_agent is not None:
            self._voice_agent.ingest_audio(pcm16_data, sample_rate, num_samples)

    def get_conversation_state(self) -> dict:
        """Get conversation state for protobuf (includes voice agent state)."""
        base_state = self._conversation_state.copy() if self._conversation_state else {}

        # Merge voice agent state
        if self._voice_agent is not None:
            voice_state = self._voice_agent.get_conversation_state()
            base_state.update(voice_state)

        return base_state

    def process_text_command(self, text: str):
        """Process a typed command.

        Simple commands (clear, reset, list) are handled instantly.
        Everything else goes through Gemini NLP via the voice agent.
        """
        text = text.strip()
        if not text:
            return

        text_lower = text.lower()

        # Instant commands — no LLM needed, zero latency
        if text_lower in ("clear", "clearall", "reset"):
            self.set_active_prompts(set())
            self.clear_effects()
            logger.info("Cleared all prompts and effects")
            return
        if text_lower in ("list", "ls", "show", "status"):
            active = self.get_active_prompts()
            effects = self.get_effects()
            logger.info(f"Active prompts: {sorted(active)}, effects: {effects}")
            return

        # Everything else → Gemini NLP
        if self._voice_agent is not None:
            self._voice_agent.process_text_command(text)
        else:
            # Voice agent unavailable — direct prompt add as last resort
            # (no regex parsing — just treat the whole text as a label)
            self.add_prompt(text_lower)
            self.set_effect(text_lower, "blur", 1.0)
            logger.info(f"No voice agent — direct prompt: {text_lower}")

    def _on_voice_command(self, action: str, targets: list[str], effect_type: str, intensity: float, invert: bool = False):
        """Callback from VoiceAgent: sync SAM3 prompts with voice commands.

        Per spec: "We should always leave requested objects within the array" —
        prompts persist even after effect removal so SAM3 keeps segmenting them.

        If invert=True, SAM3 segments the targets but the effect applies to
        everything OUTSIDE those masks (e.g., "blur everything but laptop").
        """
        # Validate targets — filter garbage from bad transcriptions
        valid_targets = []
        for t in targets:
            t = t.strip().lower()
            if not t or len(t) < 2 or len(t) > 30:
                continue
            # Skip targets that are clearly not object names
            if any(c.isdigit() for c in t):
                continue
            valid_targets.append(t)

        if not valid_targets and action in ("add", "change"):
            logger.warning(f"No valid targets from voice command (raw: {targets}), skipping")
            return

        if action in ("add", "change"):
            for target in valid_targets:
                self.add_prompt(target)
                self.set_effect(target, effect_type, intensity, invert=invert)
            mode = " (inverted)" if invert else ""
            logger.info(f"Voice → SAM3: add prompts {valid_targets} with {effect_type}@{intensity}{mode}")
        elif action == "remove":
            if not valid_targets:
                # "clear" command — remove ALL prompts and effects
                self.set_active_prompts(set())
                self.clear_effects()
                logger.info("Voice → SAM3: cleared all prompts and effects")
            else:
                for target in valid_targets:
                    self.remove_effect(target)
                    self.remove_prompt(target)
                logger.info(f"Voice → SAM3: removed prompts + effects for {valid_targets}")
