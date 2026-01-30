#!/usr/bin/env python3
"""
SAM Click-to-Mask

Click on any object to turn it BLUE.
Release mouse to return to normal.

Shows:
- All SAM mask outlines
- YOLO object labels
- Click highlighting
"""

import cv2
import numpy as np
import time
from ultralytics import FastSAM, YOLO
import mediapipe as mp


class ClickMaskApp:
    def __init__(self):
        # Mouse state
        self.mouse_down = False
        self.mouse_pos = (0, 0)
        self.clicked_mask_idx = None
        
        # Models
        print("Loading models...")
        self.sam = FastSAM("FastSAM-s.pt")
        print("  ✓ FastSAM")
        self.yolo = YOLO("yolov8n.pt")
        print("  ✓ YOLO")
        self.selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        print("  ✓ MediaPipe")
        print("✅ Ready!")
        
        # Cached data
        self.masks = []  # List of (mask, label, bbox, center, priority)
        self.person_mask = None
        self.frame_count = 0
        
        # Store non-overlapping masks
        self.clean_masks = []
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            # Find which mask was clicked
            self.clicked_mask_idx = self._find_mask_at_point(x, y)
            if self.clicked_mask_idx is not None:
                print(f"🖱️ Clicked: {self.masks[self.clicked_mask_idx][1]}")
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            self.clicked_mask_idx = None
    
    def _find_mask_at_point(self, x, y):
        """Find the SMALLEST mask that contains the clicked point (most specific)."""
        candidates = []
        for i, (mask, label, bbox, center, priority) in enumerate(self.clean_masks):
            if mask is not None and mask.shape[0] > y and mask.shape[1] > x:
                if mask[y, x] > 0.5:
                    # Calculate mask area
                    area = np.sum(mask > 0.5)
                    candidates.append((i, area, priority))
        
        if not candidates:
            return None
        
        # Return the smallest mask (most specific to click point)
        candidates.sort(key=lambda x: (x[2], x[1]))  # Sort by priority, then area
        return candidates[0][0]
    
    def process_frame(self, frame):
        """Process frame and return display with masks."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display = frame.copy()
        
        self.frame_count += 1
        
        # Get person mask (every frame)
        selfie_result = self.selfie.process(rgb)
        self.person_mask = selfie_result.segmentation_mask if selfie_result.segmentation_mask is not None else np.zeros((h, w))
        
        # Update masks periodically (every 3 frames for speed)
        if self.frame_count % 3 == 0:
            self._update_masks(frame, h, w)
            self._create_clean_masks(h, w)
        
        # Draw all mask outlines and labels
        for i, (mask, label, bbox, center, priority) in enumerate(self.clean_masks):
            if mask is None:
                continue
            
            # Ensure mask is right size
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
                self.clean_masks[i] = (mask, label, bbox, center, priority)
            
            # Get contours for outline (tight fit)
            mask_u8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if this mask is being clicked
            is_clicked = (self.mouse_down and self.clicked_mask_idx == i)
            
            if is_clicked:
                # Apply BLUE color ONLY to this specific mask
                mask_binary = (mask > 0.5).astype(np.float32)
                mask_3d = np.stack([mask_binary, mask_binary, mask_binary], axis=2)
                blue_overlay = np.full_like(frame, (255, 50, 50), dtype=np.float32)  # Blue in BGR
                display = (display * (1 - mask_3d * 0.7) + blue_overlay * mask_3d * 0.7).astype(np.uint8)
                # Thick bright outline for clicked
                cv2.drawContours(display, contours, -1, (255, 255, 0), 2)
            else:
                # Normal thin outline - color based on type
                if label == "person":
                    color = (255, 255, 0)  # Cyan
                elif label in ["wall", "floor", "ceiling", "background"]:
                    color = (0, 200, 0)  # Green
                else:
                    color = (255, 255, 255)  # White
                cv2.drawContours(display, contours, -1, color, 1)
            
            # Draw label
            if center and center[0] > 0 and center[1] > 0:
                cx, cy = center
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (cx - 2, cy - th - 2), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
                cv2.putText(display, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw mouse cursor indicator
        cv2.circle(display, self.mouse_pos, 5, (0, 0, 255), -1)
        if self.mouse_down:
            cv2.circle(display, self.mouse_pos, 10, (0, 0, 255), 2)
        
        return display
    
    def _create_clean_masks(self, h, w):
        """Create non-overlapping masks by subtracting smaller from larger."""
        if not self.masks:
            self.clean_masks = []
            return
        
        # Sort masks by area (smallest first = highest priority)
        sorted_masks = []
        for mask, label, bbox, center in self.masks:
            if mask is not None:
                area = np.sum(mask > 0.5)
                # Priority: smaller = better (1), person always high priority (0)
                priority = 0 if label == "person" else 1
                sorted_masks.append((mask, label, bbox, center, area, priority))
        
        # Sort by priority then area
        sorted_masks.sort(key=lambda x: (x[5], x[4]))
        
        self.clean_masks = []
        used_pixels = np.zeros((h, w), dtype=bool)
        
        for mask, label, bbox, center, area, priority in sorted_masks:
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            # Create clean mask by removing already-used pixels
            mask_binary = mask > 0.5
            clean_mask = mask_binary & ~used_pixels
            
            # Apply morphological operations for tighter borders
            clean_u8 = (clean_mask * 255).astype(np.uint8)
            
            # Erode slightly to tighten borders
            kernel = np.ones((3, 3), np.uint8)
            clean_u8 = cv2.erode(clean_u8, kernel, iterations=1)
            # Then dilate back a bit
            clean_u8 = cv2.dilate(clean_u8, kernel, iterations=1)
            
            clean_mask = clean_u8.astype(np.float32) / 255.0
            
            # Skip if mask became too small
            if np.sum(clean_mask > 0.5) < 300:
                continue
            
            # Mark these pixels as used
            used_pixels |= (clean_mask > 0.5)
            
            # Recalculate center for clean mask
            new_center = self._mask_center(clean_mask)
            
            self.clean_masks.append((clean_mask, label, bbox, new_center, priority))
    
    def _update_masks(self, frame, h, w):
        """Update mask list from SAM and YOLO."""
        self.masks = []
        
        # Add person mask first (refined with morphology)
        if self.person_mask is not None and np.any(self.person_mask > 0.5):
            pm = self.person_mask.copy()
            if pm.shape != (h, w):
                pm = cv2.resize(pm, (w, h))
            
            # Refine person mask edges
            pm_u8 = (pm * 255).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            pm_u8 = cv2.morphologyEx(pm_u8, cv2.MORPH_CLOSE, kernel)
            pm_u8 = cv2.morphologyEx(pm_u8, cv2.MORPH_OPEN, kernel)
            pm = pm_u8.astype(np.float32) / 255.0
            
            center = self._mask_center(pm)
            self.masks.append((pm, "person", None, center))
        
        # Get YOLO detections for labels
        yolo_results = self.yolo(frame, verbose=False, conf=0.25)
        yolo_data = []
        if yolo_results and len(yolo_results) > 0:
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                label = self.yolo.names[int(box.cls[0])]
                conf = float(box.conf[0])
                yolo_data.append(((x1, y1, x2, y2), label, conf))
        
        # Get SAM masks
        try:
            sam_results = self.sam(frame, device="cpu", retina_masks=True,
                                  imgsz=320, conf=0.4, verbose=False)
            
            if sam_results and sam_results[0].masks is not None:
                for mask_data in sam_results[0].masks.data.cpu().numpy():
                    mask = cv2.resize(mask_data.astype(np.float32), (w, h))
                    
                    # SUBTRACT person from this mask to avoid overlap
                    if self.person_mask is not None:
                        person_binary = self.person_mask > 0.5
                        mask_binary = mask > 0.5
                        
                        # Check overlap
                        overlap = np.sum(mask_binary & person_binary)
                        mask_area = np.sum(mask_binary)
                        
                        # If mostly person, skip entirely
                        if mask_area > 0 and overlap > mask_area * 0.6:
                            continue
                        
                        # Remove person pixels from this mask
                        mask = mask * (1 - self.person_mask.astype(np.float32) * 0.9)
                    
                    # Skip tiny masks
                    if np.sum(mask > 0.5) < 400:
                        continue
                    
                    # Refine mask edges
                    mask_u8 = (mask * 255).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
                    mask = mask_u8.astype(np.float32) / 255.0
                    
                    # Get label from YOLO
                    label = self._get_label_for_mask(mask, yolo_data, h, w)
                    center = self._mask_center(mask)
                    bbox = self._mask_bbox(mask)
                    
                    self.masks.append((mask, label, bbox, center))
        except Exception as e:
            print(f"SAM error: {e}")
    
    def _get_label_for_mask(self, mask, yolo_data, h, w):
        """Get YOLO label for a mask based on overlap."""
        mask_bbox = self._mask_bbox(mask)
        
        best_iou = 0
        best_label = "object"
        
        for yolo_bbox, label, conf in yolo_data:
            iou = self._calc_iou(mask_bbox, yolo_bbox)
            if iou > best_iou and iou > 0.2:
                best_iou = iou
                best_label = label
        
        # Classify by position if no YOLO match
        if best_label == "object":
            best_label = self._classify_by_position(mask, h, w)
        
        return best_label
    
    def _mask_bbox(self, mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    
    def _mask_center(self, mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0)
        return (int(np.mean(xs)), int(np.mean(ys)))
    
    def _calc_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / (union + 1e-6)
    
    def _classify_by_position(self, mask, h, w):
        """Classify mask by position in frame."""
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return "object"
        
        cy = np.mean(ys)
        area = len(xs)
        
        # Large areas at edges = wall/floor/ceiling
        if area > h * w * 0.1:
            if cy < h * 0.25:
                return "ceiling"
            elif cy > h * 0.75:
                return "floor"
            else:
                return "wall"
        
        return "object"
    
    def close(self):
        self.selfie.close()


def main():
    print("\n" + "="*55)
    print("  🖱️ SAM CLICK-TO-MASK")
    print("="*55)
    print("\n👆 CLICK on any object to turn it BLUE")
    print("   RELEASE to return to normal")
    print("\n📋 Shows all object outlines + YOLO labels")
    print("\n⌨️  Q=quit S=save")
    print("="*55 + "\n")
    
    app = ClickMaskApp()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ No camera!")
        return
    
    cv2.namedWindow("Click to Mask")
    cv2.setMouseCallback("Click to Mask", app.mouse_callback)
    
    fps_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            t0 = time.time()
            display = app.process_frame(frame)
            elapsed = time.time() - t0
            
            # FPS
            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))
            
            # Info
            cv2.rectangle(display, (5, 5), (180, 45), (0, 0, 0), -1)
            cv2.putText(display, f"FPS: {fps:.0f} | Objects: {len(app.masks)}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display, "Click object to mask BLUE", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            cv2.imshow("Click to Mask", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fn = f"click_mask_{int(time.time())}.png"
                cv2.imwrite(fn, display)
                print(f"💾 {fn}")
                
    finally:
        app.close()
        cap.release()
        cv2.destroyAllWindows()
        print("👋")


if __name__ == "__main__":
    main()

