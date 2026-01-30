"""
Emotion Service - Fast facial emotion detection with temporal smoothing.

Runs in a separate thread to avoid blocking the main video pipeline.
Uses fer (Facial Expression Recognition) for fast inference (~20-50ms per frame).
"""
import threading
import time
from collections import deque, Counter
from queue import Queue, Empty
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from .base import BaseAudioService

# Lazy import fer to avoid startup delay
_fer_detector = None


def get_fer_detector():
    """Lazy load the FER detector."""
    global _fer_detector
    if _fer_detector is None:
        try:
            # Try new import path first (fer >= 25.x)
            try:
                from fer.fer import FER
            except ImportError:
                # Fall back to old import path
                from fer import FER

            # Use mtcnn=True for better face detection accuracy
            _fer_detector = FER(mtcnn=True)
            print("[EMOTION] FER detector loaded (MTCNN backend)")

            # Quick self-test with a blank image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                _fer_detector.detect_emotions(test_img)
                print("[EMOTION] FER self-test passed")
            except Exception as e:
                print(f"[EMOTION] FER self-test warning: {e}")

        except ImportError as e:
            print(f"[EMOTION] WARNING: fer not installed: {e}")
            print("[EMOTION] Run: pip install fer tensorflow")
            print("[EMOTION] Falling back to placeholder emotion detection")
            _fer_detector = "placeholder"
        except Exception as e:
            print(f"[EMOTION] WARNING: FER init failed: {e}")
            import traceback
            traceback.print_exc()
            _fer_detector = "placeholder"
    return _fer_detector


class EmotionService(BaseAudioService):
    """Fast emotion detection service with temporal smoothing."""

    # Emotion detection settings - tuned for sensitivity
    FRAME_SKIP = 1  # Process every frame (most responsive)
    SMOOTHING_WINDOW = 4  # Shorter history for faster response
    STABILITY_THRESHOLD = 2  # Only need 2/4 agreement to change (very responsive)

    # Emotion mapping for display (no emojis for Windows compatibility)
    EMOTION_DISPLAY = {
        'angry': 'Frustrated',
        'disgust': 'Uncomfortable',
        'fear': 'Anxious',
        'happy': 'Happy',
        'sad': 'Sad',
        'surprise': 'Surprised',
        'neutral': 'Neutral'
    }

    def __init__(self, config, state_dir: Path):
        """Initialize emotion service."""
        super().__init__(config, state_dir)

        # Face frame queue (receives cropped faces from video pipeline)
        self.face_queue: Queue = Queue(maxsize=5)

        # Emotion history for smoothing
        self.emotion_history: deque = deque(maxlen=self.SMOOTHING_WINDOW)

        # Current stable emotion
        self.current_emotion = "neutral"
        self.current_emotion_display = "Neutral"
        self.emotion_confidence = 0.0

        # Processing thread
        self.thread: Optional[threading.Thread] = None

        # Frame counter for skipping
        self.frame_count = 0

        # Lock for thread-safe state access
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start emotion detection service."""
        try:
            # Pre-load the detector
            detector = get_fer_detector()
            if detector == "placeholder":
                print("[EMOTION] Running in placeholder mode (no fer installed)")

            self.running = True

            # Start processing thread
            self.thread = threading.Thread(
                target=self._emotion_processing_loop,
                daemon=True
            )
            self.thread.start()

            print("[EMOTION] Emotion service started")
            return True

        except Exception as e:
            print(f"[EMOTION] Failed to start: {e}")
            return False

    def stop(self) -> None:
        """Stop emotion detection service."""
        self.running = False

        # Clear queue to unblock thread
        try:
            while not self.face_queue.empty():
                self.face_queue.get_nowait()
        except:
            pass

        if self.thread:
            self.thread.join(timeout=1.0)

        print("[EMOTION] Emotion service stopped")

    def get_state(self) -> Dict[str, Any]:
        """Get current emotion state."""
        with self._lock:
            return {
                'emotion': self.current_emotion,
                'emotion_display': self.current_emotion_display,
                'emotion_confidence': self.emotion_confidence
            }

    def submit_face(self, face_image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Submit a face image for emotion detection.

        Call this from the video pipeline with the cropped face.
        Only processes every Nth frame for performance.

        Args:
            face_image: BGR face image (cropped from frame)
            bbox: Optional bounding box (x, y, w, h) for context
        """
        self.frame_count += 1

        # Skip frames for performance
        if self.frame_count % self.FRAME_SKIP != 0:
            return

        # Don't block if queue is full
        try:
            self.face_queue.put_nowait({
                'image': face_image,
                'bbox': bbox,
                'timestamp': time.time()
            })
        except:
            pass  # Queue full, skip this frame

    def _emotion_processing_loop(self) -> None:
        """Background thread for emotion detection."""
        print("[EMOTION] Processing loop started")
        detector = get_fer_detector()
        faces_received = 0
        faces_with_detection = 0
        last_status_time = time.time()

        while self.running:
            try:
                # Get face from queue with timeout
                face_data = self.face_queue.get(timeout=0.1)
                face_image = face_data['image']
                faces_received += 1

                # Log first face and save for debugging
                if faces_received == 1:
                    h, w = face_image.shape[:2]
                    print(f"[EMOTION] First face received: {w}x{h}")
                    # Save first face for debugging
                    try:
                        import cv2
                        debug_path = self.state_dir / "debug_face.jpg"
                        cv2.imwrite(str(debug_path), face_image)
                        print(f"[EMOTION] Saved debug face to {debug_path}")
                    except Exception as e:
                        print(f"[EMOTION] Could not save debug face: {e}")

                if detector == "placeholder":
                    # Placeholder mode - just use neutral
                    emotion = "neutral"
                    confidence = 0.5
                else:
                    # Detect emotion
                    emotion, confidence = self._detect_emotion(detector, face_image)
                    if emotion != "neutral" or confidence > 0:
                        faces_with_detection += 1

                # Update history
                self.emotion_history.append(emotion)

                # Check for stable emotion
                self._update_stable_emotion()

                # Periodic status update
                now = time.time()
                if now - last_status_time > 5.0:
                    last_status_time = now
                    history_counts = Counter(self.emotion_history)
                    print(f"[EMOTION] Status: {faces_received} faces, {faces_with_detection} with detection, history: {dict(history_counts)}")

            except Empty:
                continue
            except Exception as e:
                if self.running:
                    print(f"[EMOTION] Processing error: {e}")
                    import traceback
                    traceback.print_exc()

    def _detect_emotion(self, detector, face_image: np.ndarray) -> Tuple[str, float]:
        """Detect dominant emotion from face image."""
        try:
            # FER expects BGR image
            result = detector.detect_emotions(face_image)

            if result and len(result) > 0:
                # Get emotions dict from first (largest) face
                face_data = result[0]
                emotions = face_data.get('emotions', {})
                box = face_data.get('box', None)

                # Log detailed results periodically
                if hasattr(self, '_detect_count'):
                    self._detect_count += 1
                else:
                    self._detect_count = 1

                if self._detect_count <= 3 or self._detect_count % 50 == 0:
                    print(f"[EMOTION] Face found at {box}")
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                    print(f"[EMOTION] All emotions: {sorted_emotions}")

                if emotions:
                    # Find dominant emotion
                    dominant = max(emotions.items(), key=lambda x: x[1])
                    return dominant[0], dominant[1]
            else:
                # No face detected
                if hasattr(self, '_no_face_count'):
                    self._no_face_count += 1
                else:
                    self._no_face_count = 1

                if self._no_face_count <= 3 or self._no_face_count % 100 == 0:
                    h, w = face_image.shape[:2]
                    print(f"[EMOTION] No face detected in {w}x{h} image (count: {self._no_face_count})")

            return "neutral", 0.0

        except Exception as e:
            print(f"[EMOTION] Detection error: {e}")
            import traceback
            traceback.print_exc()
            return "neutral", 0.0

    def _update_stable_emotion(self) -> None:
        """Update the stable emotion based on history."""
        if len(self.emotion_history) < 3:
            return

        # Count emotions in history
        counts = Counter(self.emotion_history)
        top_emotion, count = counts.most_common(1)[0]

        # Only change if we have enough agreement
        if count >= self.STABILITY_THRESHOLD:
            with self._lock:
                if self.current_emotion != top_emotion:
                    print(f"[EMOTION] Changed: {self.current_emotion} -> {top_emotion} ({count}/{len(self.emotion_history)} agreement)")
                    self.current_emotion = top_emotion
                    self.current_emotion_display = self.EMOTION_DISPLAY.get(
                        top_emotion, top_emotion.title()
                    )
                    self.emotion_confidence = count / len(self.emotion_history)
