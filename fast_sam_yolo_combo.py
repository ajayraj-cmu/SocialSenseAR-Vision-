#!/usr/bin/env python3
"""
FastSAM + YOLO-World + Gemini Vision Combo - Advanced Object Detection & Segmentation

Features:
- FastSAM for precise segmentation
- YOLO-World for open-vocabulary object detection
- Gemini Vision for correcting labels and identifying unknowns
- Dev/Non-Dev mode toggle (press 'D')
- Real-time visualization

Controls:
- D: Toggle Dev/Non-Dev mode
- Q: Quit
- S: Save screenshot
"""

import cv2
import numpy as np
import time
import os
import sys
from PIL import Image
from ultralytics import FastSAM, YOLOWorld

# Force unbuffered output for immediate feedback
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Try new Gemini SDK first, fallback to old one
GEMINI_AVAILABLE = False
try:
    import google.genai as genai
    GEMINI_AVAILABLE = True
    GEMINI_NEW_SDK = True
    print("✓ Using new google.genai SDK")
except ImportError:
    try:
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
        GEMINI_NEW_SDK = False
        print("✓ Using legacy google.generativeai SDK")
    except ImportError:
        print("⚠️  No Gemini SDK available. Install with: pip install google-genai")

class FastSAMYOLOWorldGemini:
    def __init__(self):
        print("\n" + "="*60)
        print("  FastSAM + YOLO-World + Gemini Vision")
        print("="*60)
        print("\nLoading models...")

        # Load FastSAM
        self.sam = FastSAM("FastSAM-s.pt")
        print("  ✓ FastSAM loaded")

        # Load YOLO-World for open-vocabulary detection
        self.yolo = YOLOWorld("yolov8s-worldv2.pt")

        # Comprehensive indoor/campus vocabulary (from your main pipeline)
        self.vocab = [
            # STRUCTURAL ELEMENTS
            "wall", "door", "window", "ceiling", "floor", "corridor", "hallway",
            # FURNITURE
            "desk", "chair", "table", "cabinet", "shelf", "bookshelf", "sofa", "couch", "bed",
            # LIGHTING
            "light", "lamp", "desk lamp", "floor lamp", "ceiling light",
            # ELECTRONICS & SCREENS
            "laptop", "computer", "monitor", "screen", "keyboard", "mouse", "printer",
            "tv", "television", "speaker", "camera", "phone", "cell phone", "tablet",
            # DISPLAYS & BOARDS
            "whiteboard", "blackboard", "projector screen", "display",
            # PERSONAL ITEMS
            "book", "notebook", "paper", "pen", "pencil", "backpack", "bag",
            "bottle", "cup", "water bottle", "coffee cup", "mug", "glass",
            "headphones", "earbuds",
            # PEOPLE
            "person", "student", "teacher", "man", "woman",
            # ROOM ELEMENTS
            "door frame", "window frame", "door handle", "outlet", "vent",
            # APPLIANCES & FIXTURES
            "refrigerator", "fridge", "microwave", "oven", "sink", "mirror",
            # STORAGE & ORGANIZATION
            "locker", "drawer", "box", "container", "trash can", "recycling bin",
            # DECORATIVE & MISC
            "picture", "painting", "poster", "frame", "curtain", "rug", "pillow",
            # ADDITIONAL COMMON OBJECTS
            "plant", "vase", "clock", "calendar", "stapler", "scissors", "tape",
            "charger", "cable", "remote", "controller", "wallet", "keys", "watch",
            "sunglasses", "hat", "jacket", "shoes", "umbrella", "tissue box"
        ]

        # Set YOLO-World vocabulary
        self.yolo.set_classes(self.vocab)
        print(f"  ✓ YOLO-World loaded with {len(self.vocab)} object classes")

        # Initialize Gemini
        self.gemini_available = False
        self.gemini_client = None
        self.gemini_model = None
        self.last_gemini_call = 0
        self.gemini_interval = 1.5  # Call Gemini every 1.5 seconds for real-time corrections
        self.max_gemini_objects = 15  # Review max 15 objects at a time
        self.gemini_corrections = {}  # Track Gemini's label corrections

        if GEMINI_AVAILABLE:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                try:
                    if GEMINI_NEW_SDK:
                        # New SDK
                        self.gemini_client = genai.Client(api_key=api_key)
                        self.gemini_model = "gemini-2.0-flash-exp"
                    else:
                        # Legacy SDK - use models that actually exist
                        genai.configure(api_key=api_key)
                        # Try modern Gemini models that support vision
                        for model_name in ['gemini-2.0-flash', 'gemini-flash-latest', 'gemini-2.5-flash']:
                            try:
                                self.gemini_model = genai.GenerativeModel(model_name)
                                print(f"  ✓ Using {model_name} for vision")
                                break
                            except Exception as e:
                                print(f"  ⚠️ {model_name} failed: {e}")
                                continue

                        if not self.gemini_model:
                            raise Exception("No working Gemini vision model found")

                    self.gemini_available = True
                    print(f"  ✓ Gemini Vision loaded")
                except Exception as e:
                    print(f"  ⚠️  Gemini init failed: {e}")
            else:
                print("  ⚠️  No GEMINI_API_KEY found")

        print("✅ All models ready!\n")

        # Dev mode toggle
        self.dev_mode = True  # Start in dev mode

        # FPS tracking
        self.fps_times = []

        # Statistics
        self.frame_count = 0

    def process_frame(self, frame):
        """Process frame with FastSAM and YOLO-World."""
        h, w = frame.shape[:2]

        # Run YOLO-World for open-vocabulary detection with VERY LOW threshold
        yolo_results = self.yolo.predict(frame, verbose=False, conf=0.01, iou=0.5, max_det=50)
        yolo_detections = []

        if yolo_results and len(yolo_results) > 0:
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                label = self.yolo.names[cls_id]
                conf = float(box.conf[0])
                yolo_detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'label': label,
                    'confidence': conf
                })


        # Run FastSAM for segmentation with AGGRESSIVE parameters
        sam_results = self.sam(
            frame,
            device="cpu",
            retina_masks=True,
            imgsz=1024,  # High resolution
            conf=0.15,   # Low confidence to detect everything
            iou=0.5,     # Low IOU to prevent merging
            verbose=False
        )

        masks = []
        if sam_results and len(sam_results) > 0 and sam_results[0].masks is not None:
            masks_data = sam_results[0].masks.data.cpu().numpy()
            for mask_data in masks_data:
                mask_resized = cv2.resize(mask_data.astype(np.float32), (w, h))
                # Filter out very small masks (noise)
                mask_pixels = np.sum(mask_resized > 0.5)
                if mask_pixels > 100:  # At least 100 pixels
                    masks.append(mask_resized)

        self.frame_count += 1
        return yolo_detections, masks

    def match_masks_to_detections(self, masks, yolo_detections, h, w):
        """Match FastSAM masks to YOLO-World detections."""
        matched = []
        used_detections = set()

        for i, mask in enumerate(masks):
            # Get mask bbox and properties
            mask_bool = mask > 0.5
            ys, xs = np.where(mask_bool)
            if len(xs) == 0:
                continue

            mask_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            mask_center = (int(np.mean(xs)), int(np.mean(ys)))
            mask_area = len(xs)

            # Find best matching YOLO-World detection
            best_match = None
            best_score = 0
            best_iou = 0
            best_match_idx = None

            for det_idx, det in enumerate(yolo_detections):
                if det_idx in used_detections:
                    continue

                iou = self._calc_iou(mask_bbox, det['bbox'])
                # Score combines IOU and confidence
                score = iou * det['confidence']

                if score > best_score and iou > 0.05:  # Very low threshold for open vocab
                    best_score = score
                    best_iou = iou
                    best_match = det
                    best_match_idx = det_idx

            if best_match:
                used_detections.add(best_match_idx)

            # Check if we have a Gemini correction for this label
            label = best_match['label'] if best_match else None
            if label and label in self.gemini_corrections:
                label = self.gemini_corrections[label]

            matched.append({
                'mask': mask,
                'bbox': mask_bbox,
                'center': mask_center,
                'area': mask_area,
                'label': label,
                'confidence': best_match['confidence'] if best_match else 0.0,
                'yolo_match': best_match is not None,
                # Gemini reviews ALL objects, not just unknowns
                'needs_gemini': True,  # Review everything!
                'original_yolo_label': best_match['label'] if best_match else None
            })

        return matched

    def _calc_iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def label_with_gemini(self, frame, matched_objects):
        """Use Gemini to correct YOLO-World labels and identify unknowns."""
        current_time = time.time()

        # Rate limiting
        if not self.gemini_available:
            return matched_objects

        time_since_last = current_time - self.last_gemini_call
        if time_since_last < self.gemini_interval:
            return matched_objects

        # Find objects needing Gemini review - prioritize those without labels or low confidence
        needs_review = [obj for obj in matched_objects if obj['needs_gemini']]

        if not needs_review or len(needs_review) == 0:
            return matched_objects

        # Sort by priority: unknowns first, then low confidence
        needs_review.sort(key=lambda x: (0 if not x['label'] else 1, x['confidence']))

        # Limit to max objects to avoid overwhelming Gemini
        needs_review = needs_review[:self.max_gemini_objects]

        try:
            # Convert frame to PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            h, w = frame.shape[:2]

            # Create annotated image with numbered objects
            annotated = frame.copy()
            positions = []

            for idx, obj in enumerate(needs_review):
                cx, cy = obj['center']
                # Position description
                h_pos = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
                v_pos = "top" if cy < h/3 else "bottom" if cy > 2*h/3 else "middle"

                # Draw number on image
                cv2.putText(annotated, str(idx+1), (cx-15, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                # Build description
                if obj['yolo_match']:
                    positions.append(f"{idx+1}: {v_pos}-{h_pos} (YOLO says: '{obj['original_yolo_label']}')")
                else:
                    positions.append(f"{idx+1}: {v_pos}-{h_pos} (unknown object)")

            # Convert annotated image to PIL
            rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_annotated = Image.fromarray(rgb_annotated)

            # Gemini prompt - focused on CORRECTING labels
            prompt = f"""You are an expert object identification AI reviewing {len(needs_review)} detected objects.

CRITICAL: Your job is to VERIFY and CORRECT these labels. Many may be WRONG.

Detected objects (with current labels from YOLO):
{chr(10).join(positions)}

INSTRUCTIONS:
1. Look at each numbered object in the image
2. If the YOLO label is WRONG, provide the CORRECT label
3. If the YOLO label is CORRECT, confirm it
4. If unknown, identify what it actually is

Respond with ONLY a numbered list:
1: [correct label]
2: [correct label]
...

RULES:
- Be SPECIFIC: "laptop" not "electronics", "water bottle" not "bottle"
- CORRECT wrong labels (e.g., if YOLO says "hat" but it's a "person's head", say "person")
- Use lowercase, 1-3 words max
- NO explanations
- NO uncertain language like "possibly" or "might be"
- Just state what you SEE"""

            # Call Gemini
            if GEMINI_NEW_SDK:
                # New SDK
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=[prompt, pil_annotated]
                )
                response_text = response.text
            else:
                # Legacy SDK
                response = self.gemini_model.generate_content([prompt, pil_annotated])
                response_text = response.text

            self.last_gemini_call = current_time

            # Parse response
            lines = response_text.strip().split('\n')
            gemini_labels = {}

            for line in lines:
                line = line.strip()
                # Handle various formats: "1: label", "1. label", "1) label"
                for separator in [':', '.', ')']:
                    if separator in line:
                        try:
                            parts = line.split(separator, 1)
                            num_str = parts[0].strip()
                            # Remove any non-digit characters
                            num_str = ''.join(c for c in num_str if c.isdigit())
                            if num_str:
                                num = int(num_str)
                                label = parts[1].strip().lower()
                                # Clean up label
                                label = label.strip('.,;!-*[] ')
                                if label and len(label) > 0:
                                    gemini_labels[num] = label
                                    break
                        except Exception:
                            continue

            # Update labels and track corrections
            for idx, obj in enumerate(needs_review):
                if (idx + 1) in gemini_labels:
                    new_label = gemini_labels[idx + 1]
                    old_label = obj.get('original_yolo_label')

                    # Track correction if YOLO was wrong
                    if old_label and old_label != new_label:
                        self.gemini_corrections[old_label] = new_label

                    # Find object in matched_objects and update
                    for mo in matched_objects:
                        if mo is obj:
                            mo['label'] = new_label
                            mo['confidence'] = 0.8  # Gemini confidence
                            mo['source'] = 'gemini'
                            mo['needs_gemini'] = False

        except Exception as e:
            pass

        return matched_objects

    def draw_dev_mode(self, frame, matched_objects, yolo_detections):
        """Draw dev mode visualization with all debug info."""
        display = frame.copy()
        h, w = frame.shape[:2]

        # Draw all masks with colors
        np.random.seed(42)
        for i, obj in enumerate(matched_objects):
            mask = obj['mask']
            mask_bool = mask > 0.5

            # Generate color based on source
            if obj.get('source') == 'gemini':
                # Bright cyan for Gemini corrections/labels
                color_tuple = (255, 255, 0)
            elif obj['yolo_match']:
                # Green for YOLO-World matches
                color_tuple = (0, 200, 100)
            else:
                # Orange for unknown
                color_tuple = (0, 165, 255)

            # Draw mask overlay
            overlay = display.copy()
            overlay[mask_bool] = overlay[mask_bool] * 0.6 + np.array(color_tuple) * 0.4
            display = overlay.astype(np.uint8)

            # Draw contour
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, color_tuple, 2)

            # Draw label at center
            cx, cy = obj['center']

            if obj['label']:
                label = f"{i+1}: {obj['label']}"
                if obj.get('source') == 'gemini':
                    label += " 🤖"
                    label_color = (255, 255, 0)
                elif obj['yolo_match']:
                    label += f" ({obj['confidence']:.2f})"
                    label_color = (0, 255, 100)
                else:
                    label_color = (0, 200, 255)
            else:
                label = f"{i+1}: unknown"
                label_color = (0, 165, 255)

            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(display, (cx - 5, cy - th - 5), (cx + tw + 5, cy + 5), (0, 0, 0), -1)
            cv2.putText(display, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

        return display

    def draw_non_dev_mode(self, frame, matched_objects):
        """Draw clean non-dev mode visualization."""
        display = frame.copy()

        # Only draw subtle outlines for labeled objects
        for obj in matched_objects:
            if obj['label']:  # Only show labeled objects
                mask = obj['mask']
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Subtle white outline
                cv2.drawContours(display, contours, -1, (255, 255, 255), 1)

        return display

    def run(self):
        """Main loop."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("❌ Cannot open camera")
            return

        print("✅ Camera ready!")
        print("\nControls:")
        print("  D - Toggle Dev/Non-Dev mode")
        print("  Q - Quit")
        print("  S - Save screenshot")
        print("="*60 + "\n")

        cv2.namedWindow("FastSAM + YOLO-World + Gemini", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FastSAM + YOLO-World + Gemini", 1280, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                start_time = time.time()

                # Process frame
                yolo_detections, masks = self.process_frame(frame)

                # Match masks to detections
                h, w = frame.shape[:2]
                matched_objects = self.match_masks_to_detections(masks, yolo_detections, h, w)

                # Use Gemini to correct/label
                matched_objects = self.label_with_gemini(frame, matched_objects)

                # Draw based on mode
                if self.dev_mode:
                    display = self.draw_dev_mode(frame, matched_objects, yolo_detections)
                else:
                    display = self.draw_non_dev_mode(frame, matched_objects)

                # Calculate FPS
                elapsed = time.time() - start_time
                self.fps_times.append(elapsed)
                if len(self.fps_times) > 30:
                    self.fps_times.pop(0)
                avg_time = sum(self.fps_times) / len(self.fps_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0

                # Info overlay
                h, w = display.shape[:2]
                overlay = display.copy()
                cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

                mode_text = "DEV MODE" if self.dev_mode else "NON-DEV MODE"
                mode_color = (0, 255, 255) if self.dev_mode else (0, 255, 0)

                gemini_status = "✓" if self.gemini_available else "✗"
                cv2.putText(display, f"FastSAM+YOLOWorld+Gemini{gemini_status}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display, f"Mode: {mode_text}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

                # Count sources
                yolo_count = sum(1 for obj in matched_objects if obj['yolo_match'] and obj.get('source') != 'gemini')
                gemini_count = sum(1 for obj in matched_objects if obj.get('source') == 'gemini')
                unknown_count = len([obj for obj in matched_objects if not obj['label']])

                cv2.putText(display, f"Objects: {len(matched_objects)} | YOLO:{yolo_count} Gemini:{gemini_count} ?:{unknown_count}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(display, f"FPS: {fps:.1f} | Gemini corrections: {len(self.gemini_corrections)}", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(display, f"Vocab: {len(self.vocab)} classes", (20, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                cv2.imshow("FastSAM + YOLO-World + Gemini", display)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('d') or key == ord('D'):
                    self.dev_mode = not self.dev_mode
                    mode = "DEV" if self.dev_mode else "NON-DEV"
                    print(f"🔄 Switched to {mode} MODE")
                elif key == ord('s'):
                    filename = f"fastsam_yoloworld_gemini_{int(time.time())}.png"
                    cv2.imwrite(filename, display)
                    print(f"💾 Saved: {filename}")

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n👋 Done!")


def main():
    app = FastSAMYOLOWorldGemini()
    app.run()


if __name__ == "__main__":
    main()
