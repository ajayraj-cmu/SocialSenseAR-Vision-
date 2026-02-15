"""MediaPipe subprocess worker — runs in a separate process to eliminate GIL contention.

Uses shared memory for frame transfer (zero-copy write, ~0.3ms) instead of
pickling through pipes (which blocks on large numpy arrays).

Communication:
  Frame:   shared memory (640x480x3 uint8 = 900KB) + Event signal
  Results: Pipe (small metadata + uint8 compressed masks ~100KB)
"""

import time
import logging
import multiprocessing
import multiprocessing.shared_memory as shm
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Fixed frame dimensions for shared memory pre-allocation
_MAX_H = 720
_MAX_W = 1280
_FRAME_SHM_SIZE = _MAX_H * _MAX_W * 3  # uint8 BGR

# Max body masks we can return via pipe
_MAX_MASKS = 10


def _mp_process_main(result_conn, shm_name, frame_shape_arr, frame_ready_event, stop_event, config_dict):
    """Entry point for the MediaPipe subprocess."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from server.vision.mediapipe_detector import MediaPipeDetector
    from server.config import ServerConfig

    config = ServerConfig(**config_dict)
    detector = MediaPipeDetector(config)
    detector.initialize()

    # Attach to shared memory
    frame_shm = shm.SharedMemory(name=shm_name)

    MP_INTERVAL = 0.30  # ~3 FPS max
    last_run = 0.0

    # Signal ready
    result_conn.send(("ready",))

    while not stop_event.is_set():
        # Wait for new frame
        if not frame_ready_event.wait(timeout=0.05):
            continue
        frame_ready_event.clear()

        now = time.perf_counter()
        if now - last_run < MP_INTERVAL:
            continue
        last_run = now

        try:
            # Read frame from shared memory
            h = frame_shape_arr[0]
            w = frame_shape_arr[1]
            if h <= 0 or w <= 0 or h > _MAX_H or w > _MAX_W:
                continue

            # Read from shared memory using the SAME layout as the writer (MAX_H x MAX_W)
            # then slice to actual frame dims. Direct (h,w,3) view has wrong strides.
            frame_full = np.ndarray((_MAX_H, _MAX_W, 3), dtype=np.uint8, buffer=frame_shm.buf)
            frame_bgr = frame_full[:h, :w, :].copy()

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            body_masks, person_mask = detector.detect(frame_bgr, rgb, h, w)

            # Forward debug info from detector through pipe
            if detector.debug_info:
                try:
                    result_conn.send(("debug", detector.debug_info))
                except Exception:
                    pass

            # Compress masks to uint8 for smaller IPC transfer
            compact_masks = []
            for mask, label, center in body_masks[:_MAX_MASKS]:
                mask_u8 = (mask * 255).astype(np.uint8)
                # RLE-compress: encode as JPEG (lossy but fast, ~1KB per mask)
                _, enc = cv2.imencode('.png', mask_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                compact_masks.append((enc.tobytes(), label, center))

            person_enc = None
            if person_mask is not None:
                pm_u8 = (person_mask * 255).astype(np.uint8)
                _, enc = cv2.imencode('.png', pm_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                person_enc = enc.tobytes()

            # Send compressed results via pipe (small: ~10-50KB total)
            result_conn.send(("result", compact_masks, person_enc, h, w))

        except Exception as e:
            try:
                result_conn.send(("error", str(e)))
            except Exception:
                pass
            time.sleep(0.1)

    frame_shm.close()
    detector.shutdown()


class MediaPipeWorker:
    """Manages a MediaPipe subprocess with shared-memory frame transfer."""

    def __init__(self, config):
        self._config = config
        self._process = None
        self._result_conn = None
        self._child_conn = None
        self._frame_shm = None
        self._frame_np = None
        self._frame_shape = None  # multiprocessing.Array for (h, w)
        self._frame_ready = None  # multiprocessing.Event
        self._stop_event = None

        # Cached results
        self.body_masks: list = []
        self.person_mask: np.ndarray | None = None
        self.frame_count = 0

    def start(self):
        """Spawn the MediaPipe subprocess."""
        # Create shared memory for frame data
        self._frame_shm = shm.SharedMemory(create=True, size=_FRAME_SHM_SIZE)
        self._frame_np = np.ndarray((_MAX_H, _MAX_W, 3), dtype=np.uint8, buffer=self._frame_shm.buf)
        self._frame_np[:] = 0

        # Shared array for frame dimensions
        self._frame_shape = multiprocessing.Array('i', [0, 0])
        self._frame_ready = multiprocessing.Event()
        self._stop_event = multiprocessing.Event()

        # Pipe for results (small compressed data)
        self._result_conn, self._child_conn = multiprocessing.Pipe()

        config_dict = {
            "mediapipe_models_dir": self._config.mediapipe_models_dir,
            "fastsam_model": self._config.fastsam_model,
            "fastsam_device": self._config.fastsam_device,
            "fastsam_conf": self._config.fastsam_conf,
            "fastsam_imgsz": self._config.fastsam_imgsz,
        }

        self._process = multiprocessing.Process(
            target=_mp_process_main,
            args=(
                self._child_conn,
                self._frame_shm.name,
                self._frame_shape,
                self._frame_ready,
                self._stop_event,
                config_dict,
            ),
            daemon=True,
        )
        self._process.start()

        # Wait for ready
        if self._result_conn.poll(30):
            msg = self._result_conn.recv()
            if msg[0] == "ready":
                logger.info("MediaPipe subprocess started and ready")
            else:
                logger.warning(f"MediaPipe subprocess: {msg}")
        else:
            logger.error("MediaPipe subprocess timeout (30s)")

    def send_frame(self, frame_bgr: np.ndarray):
        """Write frame to shared memory and signal the subprocess."""
        if self._frame_shm is None:
            return
        h, w = frame_bgr.shape[:2]
        if h > _MAX_H or w > _MAX_W:
            return
        # Write frame to shared memory (memcpy, ~0.3ms for 640x480)
        self._frame_np[:h, :w, :] = frame_bgr
        self._frame_shape[0] = h
        self._frame_shape[1] = w
        self._frame_ready.set()

    def get_results(self) -> tuple[list, np.ndarray | None]:
        """Non-blocking poll for new results from subprocess."""
        if self._result_conn is None:
            return list(self.body_masks), self.person_mask

        while self._result_conn.poll():
            try:
                msg = self._result_conn.recv()
                if msg[0] == "result":
                    compact_masks, person_enc, h, w = msg[1], msg[2], msg[3], msg[4]
                    # Decompress masks
                    self.body_masks = []
                    for enc_bytes, label, center in compact_masks:
                        buf = np.frombuffer(enc_bytes, dtype=np.uint8)
                        mask_u8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                        if mask_u8 is not None:
                            mask_f = mask_u8.astype(np.float32) / 255.0
                            self.body_masks.append((mask_f, label, center))
                    if person_enc is not None:
                        buf = np.frombuffer(person_enc, dtype=np.uint8)
                        pm_u8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                        if pm_u8 is not None:
                            self.person_mask = pm_u8.astype(np.float32) / 255.0
                    else:
                        self.person_mask = None
                    self.frame_count += 1
                elif msg[0] == "debug":
                    logger.info(f"MP subprocess: {msg[1]}")
                elif msg[0] == "error":
                    logger.warning(f"MediaPipe subprocess error: {msg[1]}")
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error polling MediaPipe results: {e}")
                break

        return list(self.body_masks), self.person_mask

    def shutdown(self):
        """Stop the subprocess and clean up shared memory."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._result_conn is not None:
            try:
                self._result_conn.close()
            except Exception:
                pass
        if self._child_conn is not None:
            try:
                self._child_conn.close()
            except Exception:
                pass
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()
        if self._frame_shm is not None:
            try:
                self._frame_shm.close()
                self._frame_shm.unlink()
            except Exception:
                pass
        self._process = None
        self._result_conn = None
        self._child_conn = None
        self._frame_shm = None
        logger.info("MediaPipe subprocess stopped")
