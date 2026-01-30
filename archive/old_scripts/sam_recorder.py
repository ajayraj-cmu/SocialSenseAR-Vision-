#!/usr/bin/env python3
"""
SAM Recording & Processing Pipeline

Simple, working pipeline that:
1. Records video from webcam
2. Processes with SAM (Segment Anything Model) 
3. Shows actual segmentation masks
4. Plays back the processed result

Controls:
- R: Start/Stop Recording
- P: Process recorded video with SAM
- SPACE: Play processed video
- Q: Quit
"""

import cv2
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import torch

# Try to import SAM
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  segment-anything not installed. Run: pip install segment-anything")

# Paths
MODELS_DIR = Path(__file__).parent / "models"
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# SAM model checkpoint
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def download_sam_model():
    """Download SAM model if not present."""
    if SAM_CHECKPOINT.exists():
        print(f"✅ SAM model found: {SAM_CHECKPOINT}")
        return True
    
    print(f"📥 Downloading SAM model (vit_b, ~375MB)...")
    print(f"   From: {SAM_DOWNLOAD_URL}")
    print(f"   To: {SAM_CHECKPOINT}")
    
    try:
        import urllib.request
        urllib.request.urlretrieve(SAM_DOWNLOAD_URL, SAM_CHECKPOINT)
        print("✅ SAM model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download SAM model: {e}")
        print(f"   Please download manually from: {SAM_DOWNLOAD_URL}")
        print(f"   And place it in: {SAM_CHECKPOINT}")
        return False


class SAMProcessor:
    """Processes video frames with Segment Anything Model."""
    
    def __init__(self, checkpoint_path: str = str(SAM_CHECKPOINT), model_type: str = SAM_MODEL_TYPE):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.sam = None
        self.mask_generator = None
        self.predictor = None
        
    def initialize(self) -> bool:
        """Initialize SAM model."""
        if not SAM_AVAILABLE:
            print("❌ SAM not available")
            return False
            
        if not Path(self.checkpoint_path).exists():
            print(f"❌ SAM checkpoint not found: {self.checkpoint_path}")
            return False
        
        try:
            print(f"🔧 Loading SAM model on {self.device}...")
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)
            
            # Automatic mask generator - finds ALL objects
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=16,  # Fewer points = faster
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,  # No crop layers = faster
                min_mask_region_area=1000,  # Minimum mask size
            )
            
            # Predictor for point/box prompts
            self.predictor = SamPredictor(self.sam)
            
            print("✅ SAM model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load SAM: {e}")
            return False
    
    def generate_masks(self, frame: np.ndarray) -> List[dict]:
        """Generate all masks for a frame."""
        if self.mask_generator is None:
            return []
        
        # SAM expects RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
            
        masks = self.mask_generator.generate(rgb_frame)
        return masks
    
    def visualize_masks(self, frame: np.ndarray, masks: List[dict], alpha: float = 0.5) -> np.ndarray:
        """Overlay masks on frame with random colors."""
        if not masks:
            return frame
            
        result = frame.copy().astype(np.float32)
        
        # Sort by area (largest first)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        for i, mask_data in enumerate(sorted_masks):
            mask = mask_data['segmentation']
            
            # Generate a unique color for each mask
            np.random.seed(i * 42)
            color = np.random.randint(50, 255, 3).astype(np.float32)
            
            # Apply colored overlay where mask is True
            for c in range(3):
                result[:, :, c] = np.where(
                    mask,
                    result[:, :, c] * (1 - alpha) + color[c] * alpha,
                    result[:, :, c]
                )
            
            # Draw mask contour
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result.astype(np.uint8), contours, -1, color.tolist(), 2)
        
        return result.astype(np.uint8)
    
    def process_video(
        self, 
        input_frames: List[np.ndarray],
        progress_callback=None
    ) -> List[np.ndarray]:
        """Process all frames with SAM."""
        output_frames = []
        total = len(input_frames)
        
        print(f"\n🔄 Processing {total} frames with SAM...")
        
        for i, frame in enumerate(input_frames):
            start = time.time()
            
            # Generate masks
            masks = self.generate_masks(frame)
            
            # Visualize
            processed = self.visualize_masks(frame, masks)
            
            # Add info overlay
            self._add_info_overlay(processed, i, total, len(masks), time.time() - start)
            
            output_frames.append(processed)
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Print progress
            elapsed = time.time() - start
            print(f"   Frame {i+1}/{total}: {len(masks)} masks, {elapsed:.2f}s")
        
        print(f"✅ Processing complete!")
        return output_frames
    
    def _add_info_overlay(self, frame: np.ndarray, frame_num: int, total: int, mask_count: int, time_taken: float):
        """Add processing info to frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text
        cv2.putText(frame, f"SAM Segmentation", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_num+1}/{total}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Masks found: {mask_count}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, f"Time: {time_taken:.2f}s", (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


class VideoRecorder:
    """Records video from webcam."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.frames: List[np.ndarray] = []
        self.is_recording = False
        self.fps = 30.0
        
    def start_camera(self) -> bool:
        """Start camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ Cannot open camera {self.camera_id}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"✅ Camera started: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {self.fps}fps")
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame."""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def start_recording(self):
        """Start recording frames."""
        self.frames = []
        self.is_recording = True
        print("🔴 Recording started...")
    
    def stop_recording(self):
        """Stop recording."""
        self.is_recording = False
        print(f"⏹️  Recording stopped. {len(self.frames)} frames captured.")
    
    def record_frame(self, frame: np.ndarray):
        """Add frame to recording buffer."""
        if self.is_recording:
            self.frames.append(frame.copy())
    
    def save_video(self, filepath: str) -> bool:
        """Save recorded frames to video file."""
        if not self.frames:
            print("❌ No frames to save")
            return False
        
        h, w = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))
        
        for frame in self.frames:
            out.write(frame)
        
        out.release()
        print(f"💾 Saved video: {filepath}")
        return True
    
    def release(self):
        """Release camera."""
        if self.cap:
            self.cap.release()


def main():
    """Main application."""
    print("\n" + "="*60)
    print("  SAM Recording & Processing Pipeline")
    print("="*60)
    print("\nControls:")
    print("  R     - Start/Stop Recording")
    print("  P     - Process recorded video with SAM")
    print("  SPACE - Play processed video")
    print("  Q     - Quit")
    print("="*60 + "\n")
    
    # Check/download SAM model
    if not SAM_AVAILABLE:
        print("❌ Please install segment-anything: pip install segment-anything")
        return
    
    if not download_sam_model():
        return
    
    # Initialize SAM
    sam_processor = SAMProcessor()
    if not sam_processor.initialize():
        return
    
    # Initialize recorder
    recorder = VideoRecorder(camera_id=0)
    if not recorder.start_camera():
        return
    
    # State
    processed_frames: List[np.ndarray] = []
    playback_mode = False
    playback_index = 0
    
    cv2.namedWindow("SAM Recorder", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            if playback_mode and processed_frames:
                # Playback mode
                frame = processed_frames[playback_index]
                playback_index = (playback_index + 1) % len(processed_frames)
                
                # Add playback indicator
                cv2.putText(frame.copy(), ">>> PLAYBACK >>>", (frame.shape[1]//2 - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                cv2.imshow("SAM Recorder", frame)
                time.sleep(1/30)  # ~30fps playback
            else:
                # Live camera mode
                frame = recorder.get_frame()
                if frame is None:
                    continue
                
                # Record if active
                recorder.record_frame(frame)
                
                # Add status overlay
                display = frame.copy()
                status = "🔴 RECORDING" if recorder.is_recording else "⚪ READY"
                color = (0, 0, 255) if recorder.is_recording else (255, 255, 255)
                
                cv2.putText(display, status, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, f"Frames: {len(recorder.frames)}", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display, "[R]ecord [P]rocess [SPACE]Play [Q]uit", (20, display.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("SAM Recorder", display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('r'):
                if recorder.is_recording:
                    recorder.stop_recording()
                else:
                    recorder.start_recording()
                    
            elif key == ord('p'):
                if recorder.frames:
                    print(f"\n📊 Processing {len(recorder.frames)} frames...")
                    processed_frames = sam_processor.process_video(recorder.frames)
                    
                    # Save processed video
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = str(RECORDINGS_DIR / f"sam_processed_{timestamp}.mp4")
                    
                    # Save processed frames
                    h, w = processed_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, recorder.fps, (w, h))
                    for f in processed_frames:
                        out.write(f)
                    out.release()
                    print(f"💾 Saved processed video: {output_path}")
                    
                    print("✅ Processing complete! Press SPACE to play.")
                else:
                    print("❌ No frames recorded. Press R to record first.")
                    
            elif key == ord(' '):
                if processed_frames:
                    playback_mode = not playback_mode
                    playback_index = 0
                    if playback_mode:
                        print("▶️  Playback started")
                    else:
                        print("⏹️  Playback stopped")
                else:
                    print("❌ No processed video. Record and process first.")
                    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    
    finally:
        recorder.release()
        cv2.destroyAllWindows()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()

