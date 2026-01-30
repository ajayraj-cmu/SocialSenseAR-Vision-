#!/usr/bin/env python3
"""
SAM Live Mask Viewer - Shows ALL masks that SAM detects

NO RESTRICTIONS - Shows everything SAM can segment!

Press Q to quit
Press S to save current frame with masks
Press +/- to adjust detection sensitivity
"""

import cv2
import numpy as np
import time
import torch
from pathlib import Path

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Config
SAM_CHECKPOINT = Path(__file__).parent / "models" / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

def create_mask_overlay(frame: np.ndarray, masks: list, show_numbers: bool = True) -> np.ndarray:
    """Create colorful overlay showing ALL detected masks."""
    result = frame.copy().astype(np.float32)
    
    if not masks:
        return frame
    
    # Sort by area (largest first so smaller objects show on top)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Generate distinct colors
    np.random.seed(42)  # Consistent colors
    colors = []
    for i in range(len(sorted_masks)):
        # Use HSV for more distinct colors
        hue = (i * 137.5) % 360  # Golden angle for better distribution
        color = cv2.cvtColor(np.uint8([[[hue/2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(color.astype(np.float32))
    
    for i, mask_data in enumerate(sorted_masks):
        mask = mask_data['segmentation']
        color = colors[i]
        area = mask_data['area']
        score = mask_data.get('predicted_iou', 0)
        stability = mask_data.get('stability_score', 0)
        
        # Apply colored overlay (stronger alpha for visibility)
        alpha = 0.5
        for c in range(3):
            result[:, :, c] = np.where(
                mask,
                result[:, :, c] * (1 - alpha) + color[c] * alpha,
                result[:, :, c]
            )
        
        # Draw contour (thick border)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result.astype(np.uint8), contours, -1, color.tolist(), 3)
        
        # Add number label at centroid
        if show_numbers and contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw label background
                label = f"{i+1}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result.astype(np.uint8), 
                             (cx - tw//2 - 5, cy - th//2 - 5),
                             (cx + tw//2 + 5, cy + th//2 + 5),
                             (0, 0, 0), -1)
                cv2.putText(result.astype(np.uint8), label, 
                           (cx - tw//2, cy + th//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result.astype(np.uint8)


def main():
    print("\n" + "="*60)
    print("  SAM LIVE MASK VIEWER - ALL MASKS, NO RESTRICTIONS")
    print("="*60)
    print("\nLoading SAM model...")
    
    # Check model
    if not SAM_CHECKPOINT.exists():
        print(f"❌ SAM model not found at {SAM_CHECKPOINT}")
        print("Run sam_recorder.py first to download the model.")
        return
    
    # Device - use CPU for stability (MPS has float64 issues with SAM)
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"  # CPU is more stable for SAM on Mac
    print(f"Using device: {device}")
    
    # Load SAM
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    
    # AGGRESSIVE mask generator - detect EVERYTHING
    points_per_side = 16  # Start with 16, can increase with +
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.7,        # Lower = more masks (default 0.88)
        stability_score_thresh=0.8,  # Lower = more masks (default 0.95)
        crop_n_layers=0,             # No crop layers for speed
        min_mask_region_area=500,    # Minimum area
    )
    
    print("✅ SAM loaded!")
    print("\nControls:")
    print("  Q     - Quit")
    print("  S     - Save current frame")
    print("  +/-   - Adjust sensitivity")
    print("  N     - Toggle mask numbers")
    print("="*60 + "\n")
    
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera ready!")
    print("Processing frames... (this takes ~1-3 seconds per frame)")
    
    cv2.namedWindow("SAM Live Masks", cv2.WINDOW_NORMAL)
    
    show_numbers = True
    frame_count = 0
    sensitivity = 0.5  # 0-1, lower = more masks
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Process every frame (will be slow but shows everything)
            start_time = time.time()
            
            # Convert to RGB for SAM
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Generate ALL masks
            masks = mask_generator.generate(rgb_frame)
            
            process_time = time.time() - start_time
            
            # Create visualization
            display = create_mask_overlay(frame, masks, show_numbers)
            
            # Add info overlay
            h, w = display.shape[:2]
            
            # Background
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            # Info text
            cv2.putText(display, "SAM - ALL MASKS", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(display, f"Objects found: {len(masks)}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Process time: {process_time:.2f}s", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Points/side: {points_per_side}", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, "[Q]uit [S]ave [N]umbers [+/-]sensitivity", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Show each mask info at bottom
            y_offset = h - 20
            for i, m in enumerate(sorted(masks, key=lambda x: x['area'], reverse=True)[:10]):
                info = f"#{i+1}: area={m['area']}, iou={m.get('predicted_iou', 0):.2f}"
                cv2.putText(display, info, (20, y_offset - i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
            
            cv2.imshow("SAM Live Masks", display)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"sam_masks_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                print(f"💾 Saved: {filename}")
            elif key == ord('n'):
                show_numbers = not show_numbers
                print(f"Numbers: {'ON' if show_numbers else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                points_per_side = min(48, points_per_side + 4)
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=points_per_side,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.8,
                    crop_n_layers=0,
                    min_mask_region_area=500,
                )
                print(f"Points per side: {points_per_side} (more masks)")
            elif key == ord('-'):
                points_per_side = max(8, points_per_side - 4)
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=points_per_side,
                    pred_iou_thresh=0.7,
                    stability_score_thresh=0.8,
                    crop_n_layers=0,
                    min_mask_region_area=500,
                )
                print(f"Points per side: {points_per_side} (fewer masks)")
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

