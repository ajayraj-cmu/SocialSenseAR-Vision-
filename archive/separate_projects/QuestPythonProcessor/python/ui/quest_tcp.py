"""
Quest TCP UI - sends processed frames back to Quest over TCP.

Works with QuestTCPSource to complete the round-trip:
Quest -> TCP -> Python processing -> TCP -> Quest

Also shows a preview window on PC with overlay panels.
"""
from typing import Optional
import time
import cv2
import numpy as np

from .base import BaseUI
from .overlay_panels import render_overlay

# Profiling
PROFILE = True


class QuestTCPUI(BaseUI):
    """UI that sends processed frames back to Quest via TCP.

    Also displays a preview on the PC using OpenCV.
    """

    def __init__(self, config=None):
        self.config = config
        self.source = None  # Set by pipeline or config
        self.window_name = "Quest Preview"
        self.jpeg_quality = getattr(config, 'jpeg_quality', 95) if config else 95
        self.show_preview = getattr(config, 'show_preview', True) if config else True
        # Raw mode: send uncompressed RGB24 instead of JPEG (faster encode, larger transfer)
        self.use_raw = getattr(config, 'use_raw_output', False) if config else False

        # Stats
        self.frames_sent = 0

        # Profiling
        self.encode_times = []
        self.send_times = []
        self.last_profile_time = time.time()

    def set_source(self, source) -> None:
        """Set the TCP source (needed to send frames back)."""
        self.source = source
        print(f"[TCP-UI] Source set: {type(source).__name__}, has send_processed_frame: {hasattr(source, 'send_processed_frame')}")

    def setup(self, title: str = "") -> None:
        """Setup preview window."""
        if title:
            self.window_name = title
        if self.show_preview:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 480)  # Stereo side-by-side preview

    def show(self, frame: np.ndarray, stats: dict = None) -> None:
        """Send frame to Quest and show preview."""
        if frame is None:
            return

        # Frame is already BGR from the effect pipeline
        # Don't convert - effect.apply() returns BGR
        frame_bgr = frame

        # Render overlay panels onto frame before sending to Quest
        if stats:
            head_x = stats.get('head_x', 0.5)
            head_y = stats.get('head_y', 0.3)
            person_tracked = stats.get('person_tracked', False)
            frame_bgr = render_overlay(frame_bgr, head_x, head_y, person_tracked)

        t0 = time.time()

        # Send to Quest
        if self.source and hasattr(self.source, 'send_processed_frame'):
            success = self.source.send_processed_frame(frame_bgr, self.jpeg_quality, use_raw=self.use_raw)
            if success:
                self.frames_sent += 1
            elif self.frames_sent == 0:
                # Log first failure
                print(f"[TCP-UI] send_processed_frame failed! source.connected={getattr(self.source, 'connected', '?')}")
        elif self.frames_sent == 0:
            # Log if source not set
            print(f"[TCP-UI] WARNING: source not set or missing send_processed_frame! source={self.source}")

        t1 = time.time()

        # Profiling
        if PROFILE:
            self.send_times.append((t1 - t0) * 1000)

            if time.time() - self.last_profile_time >= 2.0:
                avg_send = sum(self.send_times) / len(self.send_times) if self.send_times else 0
                print(f"[TCP-OUT] send:{avg_send:.1f}ms | sent {self.frames_sent} frames")
                self.send_times.clear()
                self.frames_sent = 0
                self.last_profile_time = time.time()

        # Show preview on PC
        if self.show_preview:
            # Add stats overlay
            if stats:
                self._draw_stats(frame_bgr, stats)
            cv2.imshow(self.window_name, frame_bgr)

    def poll_input(self) -> Optional[int]:
        """Poll for keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return None
        return key

    def cleanup(self) -> None:
        """Cleanup preview window."""
        cv2.destroyAllWindows()
        print(f"Sent {self.frames_sent} frames to Quest")

    def _draw_stats(self, frame: np.ndarray, stats: dict) -> None:
        """Draw stats overlay on frame."""
        h, w = frame.shape[:2]

        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)

        # Stats text
        fps = stats.get('fps', 0)
        yolo_fps = stats.get('yolo_fps', 0)
        status = stats.get('status', '')

        text = f"Video: {fps:.1f} fps | YOLO: {yolo_fps:.1f} fps | {status}"
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
