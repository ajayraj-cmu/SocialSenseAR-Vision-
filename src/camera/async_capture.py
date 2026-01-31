"""Async camera capture - smooth FPS independent of processing."""
import threading
import cv2


class AsyncCamera:
    """Async camera capture - smooth FPS independent of processing."""

    def __init__(self, src=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.frame_count = 0

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        """Continuously capture frames in background."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame
                    self.frame_count += 1

    def get_frame(self):
        """Get the latest frame (non-blocking)."""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
