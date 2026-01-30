#!/usr/bin/env python3
"""
FAST SAM Live Mask Viewer - Real-time segmentation!

Uses FastSAM which is 50x faster than regular SAM.
Shows ALL masks with no restrictions.

Press Q to quit
Press S to save frame
"""

import cv2
import numpy as np
import time
from pathlib import Path

# FastSAM via ultralytics
from ultralytics import FastSAM

def create_mask_overlay(frame: np.ndarray, results, alpha: float = 0.5) -> np.ndarray:
    """Create colorful overlay showing ALL detected masks."""
    result = frame.copy()
    
    if results is None or len(results) == 0:
        return frame
    
    # Get masks from results
    if results[0].masks is None:
        return frame
    
    masks = results[0].masks.data.cpu().numpy()
    
    # Generate distinct colors
    np.random.seed(42)
    
    for i, mask in enumerate(masks):
        # Resize mask to frame size
        mask_resized = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
        mask_bool = mask_resized > 0.5
        
        # Generate color using golden angle for distinct colors
        hue = int((i * 137.5) % 180)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        
        # Apply colored overlay
        overlay = result.copy()
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
        result = overlay.astype(np.uint8)
        
        # Draw contour
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color.tolist(), 2)
        
        # Add number at centroid
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, str(i+1), (cx-10, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(result, str(i+1), (cx-10, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return result


def main():
    print("\n" + "="*60)
    print("  FAST SAM - REAL-TIME SEGMENTATION")
    print("="*60)
    print("\nLoading FastSAM model...")
    
    # Load FastSAM (will auto-download ~140MB model)
    model = FastSAM("FastSAM-s.pt")  # Small model for speed
    
    print("✅ FastSAM loaded!")
    print("\nControls:")
    print("  Q - Quit")
    print("  S - Save frame")
    print("="*60 + "\n")
    
    # Camera - use lower resolution for speed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower res = faster
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera ready!")
    
    cv2.namedWindow("FastSAM Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FastSAM Live", 1280, 720)
    
    frame_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            start_time = time.time()
            
            # Run FastSAM - segment everything
            results = model(
                frame,
                device="cpu",  # Use CPU for stability
                retina_masks=True,
                imgsz=320,  # Smaller = faster
                conf=0.4,
                iou=0.9,
            )
            
            # Create visualization
            display = create_mask_overlay(frame, results)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Get mask count
            mask_count = 0
            if results and len(results) > 0 and results[0].masks is not None:
                mask_count = len(results[0].masks)
            
            # Add info overlay
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            cv2.putText(display, "FAST SAM - ALL MASKS", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Objects: {mask_count}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"FPS: {fps:.1f} ({elapsed*1000:.0f}ms)", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("FastSAM Live", display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"fastsam_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                print(f"💾 Saved: {filename}")
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

