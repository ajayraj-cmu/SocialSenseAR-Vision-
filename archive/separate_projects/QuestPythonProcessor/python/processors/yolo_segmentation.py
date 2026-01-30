"""
YOLO person segmentation processor.

Uses YOLOv8 segmentation model to detect and segment people in video frames.
Supports TensorRT, ONNX, and PyTorch backends with automatic selection.

Requirements:
    - ultralytics
    - torch
    - (optional) tensorrt for maximum speed
"""
from typing import Optional, Tuple
import threading
from queue import Queue, Empty
import time
import os

import numpy as np
import cv2

from .base import BaseProcessor, ProcessorResult


class YOLOSegmentationProcessor(BaseProcessor):
    """YOLO-based person segmentation processor.

    Detects people in stereo video frames and produces segmentation masks.
    Runs asynchronously in a background thread for non-blocking operation.

    Features:
        - Automatic GPU detection (CUDA > MPS > CPU)
        - TensorRT/ONNX acceleration when available
        - Temporal smoothing for stable masks
        - Batch processing for stereo frames

    Usage:
        processor = YOLOSegmentationProcessor(config)
        processor.start()

        # In render loop:
        processor.process(frame)
        result = processor.get_result()
        if result.has_detection:
            apply_effect(frame, result.left_mask, result.right_mask)

        processor.stop()
    """

    def __init__(self, config=None):
        """Initialize YOLO processor.

        Args:
            config: Configuration object with processor_width, mask_blur, etc.
        """
        self.config = config
        self.model = None
        self.device = 'cpu'
        self.model_type = 'pytorch'

        # Processing settings
        self.process_width = 640
        if config:
            self.process_width = getattr(config, 'processor_width', 640)

        # Threading
        self.input_queue = Queue(maxsize=1)
        self.running = False
        self.thread = None

        # Results (thread-safe access via lock)
        self.lock = threading.Lock()
        self.left_mask = None
        self.right_mask = None
        self.left_center = (0.5, 0.5)
        self.right_center = (0.5, 0.5)
        self.left_box = None
        self.right_box = None
        self.frame_shape = None

        # Smoothing
        self.smooth_left_mask = None
        self.smooth_right_mask = None
        self.smooth_factor = 0.3
        self.smooth_left_box = None
        self.smooth_right_box = None
        self.box_smooth_factor = 0.15

        # FPS tracking
        self._fps = 0.0
        self._process_count = 0
        self._last_fps_time = time.time()

    def start(self) -> None:
        """Load model and start processing thread."""
        self._load_model()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _load_model(self) -> None:
        """Load YOLO model with best available backend."""
        import torch
        from ultralytics import YOLO

        # Detect device
        if torch.cuda.is_available():
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using CUDA (NVIDIA GPU: {gpu_name}) for YOLO")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using MPS (Apple Metal GPU) for YOLO")
        else:
            self.device = 'cpu'
            print("Using CPU for YOLO (no GPU detected)")

        # Try optimized formats in order: TensorRT > ONNX > PyTorch
        tensorrt_path = 'yolov8n-seg.engine'
        onnx_path = 'yolov8n-seg.onnx'

        if self.device == 'cuda' and os.path.exists(tensorrt_path):
            print(f"Loading TensorRT model ({tensorrt_path})...")
            self.model = YOLO(tensorrt_path)
            self.model_type = 'tensorrt'
            print("TensorRT model loaded! (Maximum speed)")
        elif self.device == 'cuda' and os.path.exists(onnx_path):
            print(f"Loading ONNX model ({onnx_path})...")
            self.model = YOLO(onnx_path)
            self.model_type = 'onnx'
            print("ONNX model loaded! (Fast inference)")
        else:
            self.model = YOLO('yolov8n-seg.pt')
            self.model.to(self.device)
            self.model_type = 'pytorch'

            if self.device == 'cuda':
                print("Model loaded! (CUDA/PyTorch)")
                if not os.path.exists(tensorrt_path) and not os.path.exists(onnx_path):
                    print("\nTip: Export TensorRT model for faster inference:")
                    print("  python -c \"from ultralytics import YOLO; YOLO('yolov8n-seg.pt').export(format='engine')\"")
            else:
                print("Model loaded!")

    def warmup(self) -> None:
        """Warm up CUDA context with dummy inference."""
        if self.device == 'cuda' and self.model:
            import torch
            torch.cuda.set_device(0)

            # Create dummy batch matching expected input size
            h = int(2208 * self.process_width / 2064)
            dummy = np.zeros((h, self.process_width, 3), dtype=np.uint8)
            batch = [dummy, dummy]

            for _ in range(3):
                self.model(batch, verbose=False, device=self.device)

            print(f"YOLO CUDA warmup complete ({self.process_width}x{h})")

    def process(self, frame: np.ndarray) -> None:
        """Submit frame for async processing.

        Args:
            frame: RGB stereo frame (left|right side-by-side)
        """
        try:
            # Clear queue and add new frame (drop old frames)
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except Empty:
                    break
            self.input_queue.put_nowait(frame)
        except:
            pass  # Queue full, skip this frame

    def get_result(self) -> ProcessorResult:
        """Get latest result with temporal smoothing."""
        with self.lock:
            # Apply temporal smoothing to masks
            alpha = self.smooth_factor
            beta = 1.0 - alpha

            if self.left_mask is not None:
                if self.smooth_left_mask is None:
                    self.smooth_left_mask = self.left_mask.copy()
                else:
                    cv2.addWeighted(self.left_mask, alpha, self.smooth_left_mask,
                                   beta, 0, self.smooth_left_mask)

            if self.right_mask is not None:
                if self.smooth_right_mask is None:
                    self.smooth_right_mask = self.right_mask.copy()
                else:
                    cv2.addWeighted(self.right_mask, alpha, self.smooth_right_mask,
                                   beta, 0, self.smooth_right_mask)

            # Smooth bounding boxes
            self.smooth_left_box = self._smooth_box(
                self.left_box, self.smooth_left_box, self.box_smooth_factor
            )
            self.smooth_right_box = self._smooth_box(
                self.right_box, self.smooth_right_box, self.box_smooth_factor
            )

            return ProcessorResult(
                left_mask=self.smooth_left_mask,
                right_mask=self.smooth_right_mask,
                left_box=self.smooth_left_box,
                right_box=self.smooth_right_box,
                left_center=self.left_center,
                right_center=self.right_center,
            )

    @property
    def fps(self) -> float:
        """Current processing FPS."""
        return self._fps

    def _smooth_box(self, new_box, smooth_box, factor, threshold=0.05):
        """Smooth bounding box with movement threshold."""
        if new_box is None:
            return smooth_box
        if smooth_box is None:
            return new_box

        dx = abs(new_box[0] - smooth_box[0]) + abs(new_box[2] - smooth_box[2])
        dy = abs(new_box[1] - smooth_box[1]) + abs(new_box[3] - smooth_box[3])

        if dx + dy < threshold:
            return smooth_box

        return tuple(
            smooth_box[i] * (1 - factor) + new_box[i] * factor
            for i in range(4)
        )

    def _worker(self) -> None:
        """Background worker thread."""
        # Warmup in worker thread's CUDA context
        self.warmup()

        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._process_frame(frame)

                # Track FPS
                self._process_count += 1
                if self._process_count % 10 == 0:
                    now = time.time()
                    self._fps = 10 / (now - self._last_fps_time)
                    self._last_fps_time = now

            except Exception as e:
                print(f"\n[YOLO ERROR] {e}")

    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a single stereo frame."""
        h, w = frame.shape[:2]
        half_w = w // 2

        # Split into left and right eye
        left_eye = frame[:, :half_w]
        right_eye = frame[:, half_w:]

        orig_h, orig_w = left_eye.shape[:2]

        # Downscale for faster processing
        scale = self.process_width / orig_w
        new_w = self.process_width
        new_h = int(orig_h * scale)

        left_small = cv2.resize(left_eye, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        right_small = cv2.resize(right_eye, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Batch process both eyes (retina_masks=True gives masks at input resolution)
        batch = [left_small, right_small]
        results = self.model(batch, verbose=False, classes=[0], device=self.device, retina_masks=True)

        # Extract results
        left_mask, left_center, left_box = self._extract_result(
            results[0], (orig_h, orig_w), (new_h, new_w)
        )
        right_mask, right_center, right_box = self._extract_result(
            results[1], (orig_h, orig_w), (new_h, new_w)
        )

        # Update shared state
        with self.lock:
            self.left_mask = left_mask
            self.right_mask = right_mask
            self.left_center = left_center
            self.right_center = right_center
            self.left_box = left_box
            self.right_box = right_box
            self.frame_shape = frame.shape

    def _extract_result(self, result, original_size, eye_size) -> Tuple:
        """Extract mask, center, box from YOLO result."""
        orig_h, orig_w = original_size
        eye_h, eye_w = eye_size

        if result.masks is None:
            return None, (0.5, 0.5), None

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            return None, (0.5, 0.5), None

        # Find person closest to center
        center_x, center_y = eye_w // 2, eye_h // 2
        min_dist = float('inf')
        closest_idx = 0
        closest_center = (0.5, 0.5)
        closest_box = None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            dist = ((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2) ** 0.5

            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                closest_center = (box_center_x / eye_w, box_center_y / eye_h)
                closest_box = (x1 / eye_w, y1 / eye_h, x2 / eye_w, y2 / eye_h)

        # Get mask at YOLO output resolution
        person_mask = masks[closest_idx]
        person_mask = (person_mask > 0.5).astype(np.uint8) * 255

        return person_mask, closest_center, closest_box
