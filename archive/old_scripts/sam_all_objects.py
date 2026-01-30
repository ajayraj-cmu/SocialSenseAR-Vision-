#!/usr/bin/env python3
"""
SAM All Objects Controller

Identifies and labels ALL objects individually:
- FastSAM for segmentation
- YOLO for object identification  
- MediaPipe for person detection
- OpenCV for additional edge/contour analysis

Each object gets a label and can be targeted by name or number.

Commands:
- "make object 3 red"
- "color the chair blue"
- "dim the wall"
- "blur the person"
- "highlight object 5"
"""

import cv2
import numpy as np
import time
import threading
from collections import defaultdict
from ultralytics import FastSAM, YOLO
import mediapipe as mp

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# Colors
COLORS = {
    "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
    "yellow": (0, 255, 255), "orange": (0, 165, 255), "pink": (203, 192, 255),
    "purple": (255, 0, 128), "cyan": (255, 255, 0), "white": (255, 255, 255),
    "magenta": (255, 0, 255), "gold": (0, 215, 255),
}

# Object categories for targeting
OBJECT_SYNONYMS = {
    "person": ["person", "me", "myself", "human", "people", "man", "woman", "face", "body"],
    "wall": ["wall", "background", "behind", "back"],
    "floor": ["floor", "ground", "carpet"],
    "ceiling": ["ceiling", "top"],
    "chair": ["chair", "seat", "stool"],
    "desk": ["desk", "table", "surface"],
    "monitor": ["monitor", "screen", "display", "tv", "television"],
    "window": ["window", "glass"],
    "door": ["door", "doorway"],
    "plant": ["plant", "flower", "tree"],
    "book": ["book", "books"],
    "bottle": ["bottle", "cup", "mug", "glass"],
    "phone": ["phone", "cell", "mobile"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "lamp": ["lamp", "light"],
    "picture": ["picture", "frame", "poster", "art"],
}


class SegmentedObject:
    """Represents a single segmented object."""
    def __init__(self, id: int, mask: np.ndarray, label: str, confidence: float, bbox: tuple):
        self.id = id
        self.mask = mask  # Binary mask
        self.label = label  # e.g., "chair", "wall", "unknown"
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.center = self._calc_center()
        self.area = np.sum(mask > 0.5)
        
    def _calc_center(self):
        ys, xs = np.where(self.mask > 0.5)
        if len(xs) > 0:
            return (int(np.mean(xs)), int(np.mean(ys)))
        return (0, 0)


class ObjectIdentifier:
    """Identifies and tracks all objects in frame."""
    
    def __init__(self):
        print("Loading models...")
        
        # FastSAM for segmentation
        self.sam = FastSAM("FastSAM-s.pt")
        print("  ✓ FastSAM")
        
        # YOLO for object detection/classification
        self.yolo = YOLO("yolov8n.pt")  # Nano model for speed
        print("  ✓ YOLO")
        
        # MediaPipe for person
        mp_selfie = mp.solutions.selfie_segmentation
        self.selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)
        print("  ✓ MediaPipe")
        
        print("✅ All models loaded!")
        
        self.objects: list[SegmentedObject] = []
        
    def process_frame(self, frame: np.ndarray) -> list[SegmentedObject]:
        """Process frame and return list of identified objects."""
        h, w = frame.shape[:2]
        self.objects = []
        
        # 1. Get person mask from MediaPipe (most accurate for people)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = self.selfie_seg.process(rgb)
        person_mask = mp_result.segmentation_mask
        
        # Add person as object 1
        if np.any(person_mask > 0.5):
            self.objects.append(SegmentedObject(
                id=1,
                mask=person_mask,
                label="person",
                confidence=0.95,
                bbox=self._mask_to_bbox(person_mask)
            ))
        
        # 2. Get YOLO detections for object identification
        yolo_results = self.yolo(frame, verbose=False, conf=0.3)
        yolo_boxes = []
        yolo_labels = []
        if yolo_results and len(yolo_results) > 0:
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                label = self.yolo.names[cls_id]
                conf = float(box.conf[0])
                yolo_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                yolo_labels.append((label, conf))
        
        # 3. Get FastSAM masks for all objects
        sam_results = self.sam(frame, device="cpu", retina_masks=True,
                              imgsz=320, conf=0.35, verbose=False)
        
        if sam_results and sam_results[0].masks is not None:
            masks = sam_results[0].masks.data.cpu().numpy()
            
            for i, mask_data in enumerate(masks):
                # Resize mask
                mask = cv2.resize(mask_data.astype(np.float32), (w, h))
                
                # Skip if this is mostly the person (already added)
                if person_mask is not None:
                    overlap = np.sum((mask > 0.5) & (person_mask > 0.5))
                    mask_area = np.sum(mask > 0.5)
                    if mask_area > 0 and overlap / mask_area > 0.5:
                        continue
                
                # Try to identify what this object is using YOLO
                label = "object"
                confidence = 0.5
                mask_bbox = self._mask_to_bbox(mask)
                
                # Match with YOLO detection by IoU
                best_iou = 0
                for yolo_box, (yolo_label, yolo_conf) in zip(yolo_boxes, yolo_labels):
                    iou = self._calc_iou(mask_bbox, yolo_box)
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        label = yolo_label
                        confidence = yolo_conf
                
                # If no YOLO match, try to classify by position/size
                if label == "object":
                    label = self._classify_by_position(mask, h, w)
                
                obj_id = len(self.objects) + 1
                self.objects.append(SegmentedObject(
                    id=obj_id,
                    mask=mask,
                    label=label,
                    confidence=confidence,
                    bbox=mask_bbox
                ))
        
        return self.objects
    
    def _mask_to_bbox(self, mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    
    def _calc_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def _classify_by_position(self, mask, h, w):
        """Classify object by its position in frame."""
        bbox = self._mask_to_bbox(mask)
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        area = np.sum(mask > 0.5)
        
        # Large area at edges = wall/background
        if area > h * w * 0.15:
            if cy < h * 0.3:
                return "ceiling"
            elif cy > h * 0.7:
                return "floor"
            else:
                return "wall"
        
        return "object"
    
    def find_objects_by_name(self, name: str) -> list[SegmentedObject]:
        """Find objects matching a name/category."""
        name = name.lower()
        matches = []
        
        # Check synonyms
        for category, synonyms in OBJECT_SYNONYMS.items():
            if name in synonyms or category in name:
                for obj in self.objects:
                    if obj.label.lower() == category or category in obj.label.lower():
                        matches.append(obj)
        
        # Direct label match
        if not matches:
            for obj in self.objects:
                if name in obj.label.lower() or obj.label.lower() in name:
                    matches.append(obj)
        
        return matches
    
    def find_object_by_id(self, obj_id: int) -> SegmentedObject:
        """Find object by numeric ID."""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None
    
    def close(self):
        self.selfie_seg.close()


class VoiceController:
    def __init__(self, callback):
        self.callback = callback
        self.running = False
        
    def start(self):
        if not SPEECH_AVAILABLE:
            return False
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 200
            self.recognizer.pause_threshold = 0.5
            self.mic = sr.Microphone()
            
            print("🎤 Calibrating...")
            with self.mic as src:
                self.recognizer.adjust_for_ambient_noise(src, duration=1)
            
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()
            print("✅ Voice ready!")
            return True
        except Exception as e:
            print(f"❌ {e}")
            return False
    
    def stop(self):
        self.running = False
        
    def _loop(self):
        while self.running:
            try:
                with self.mic as src:
                    audio = self.recognizer.listen(src, timeout=3, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio).lower()
                print(f"\n🎤 \"{text}\"")
                self.callback(text)
            except:
                pass


class EffectController:
    def __init__(self, identifier: ObjectIdentifier):
        self.identifier = identifier
        self.effects = {}  # object_id -> (effect, params)
        self.last_cmd = ""
        
    def parse(self, text: str):
        self.last_cmd = text
        text = text.lower()
        
        # Clear
        if any(w in text for w in ["clear", "reset", "off", "remove all"]):
            self.effects = {}
            print("✨ Cleared!")
            return
        
        # Find target object(s)
        targets = []
        
        # Check for "object N" or "number N"
        import re
        num_match = re.search(r'(?:object|number)\s*(\d+)', text)
        if num_match:
            obj_id = int(num_match.group(1))
            obj = self.identifier.find_object_by_id(obj_id)
            if obj:
                targets.append(obj)
                print(f"🎯 Target: object {obj_id} ({obj.label})")
        
        # Check for object names
        if not targets:
            for category in OBJECT_SYNONYMS.keys():
                if category in text:
                    matches = self.identifier.find_objects_by_name(category)
                    targets.extend(matches)
                    if matches:
                        print(f"🎯 Target: {category} ({len(matches)} found)")
                    break
            
            # Check synonyms
            if not targets:
                for category, synonyms in OBJECT_SYNONYMS.items():
                    for syn in synonyms:
                        if syn in text:
                            matches = self.identifier.find_objects_by_name(category)
                            targets.extend(matches)
                            if matches:
                                print(f"🎯 Target: {category} ({len(matches)} found)")
                            break
                    if targets:
                        break
        
        # Default to all if "everything" or no specific target
        if not targets and ("everything" in text or "all" in text):
            targets = self.identifier.objects
            print("🎯 Target: everything")
        
        if not targets:
            print("❓ No target found. Try 'object 1' or 'the wall'")
            return
        
        # Find effect
        color = None
        for name, val in COLORS.items():
            if name in text:
                color = (name, val)
                break
        
        effect = None
        if color:
            effect = ("color", color[1])
            print(f"🎨 Effect: {color[0]}")
        elif any(w in text for w in ["dim", "dark"]):
            effect = ("dim", None)
            print("🌙 Effect: dim")
        elif any(w in text for w in ["bright"]):
            effect = ("bright", None)
            print("☀️ Effect: bright")
        elif any(w in text for w in ["blur"]):
            effect = ("blur", None)
            print("🔵 Effect: blur")
        elif any(w in text for w in ["pixel"]):
            effect = ("pixelate", None)
            print("🟩 Effect: pixelate")
        elif any(w in text for w in ["highlight", "outline"]):
            effect = ("highlight", None)
            print("✨ Effect: highlight")
        elif any(w in text for w in ["hide", "remove"]):
            effect = ("hide", None)
            print("🚫 Effect: hide")
        
        if effect:
            for obj in targets:
                self.effects[obj.id] = effect
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = frame.shape[:2]
        
        for obj in self.identifier.objects:
            if obj.id not in self.effects:
                continue
            
            effect, params = self.effects[obj.id]
            mask = obj.mask
            
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            mask_bool = mask > 0.5
            mask_3d = np.stack([mask_bool] * 3, axis=2)
            
            if effect == "color" and params:
                overlay = np.full_like(frame, params, dtype=np.uint8)
                blended = (frame * 0.35 + overlay * 0.65).astype(np.uint8)
                result = np.where(mask_3d, blended, result)
                
            elif effect == "dim":
                dimmed = (frame * 0.2).astype(np.uint8)
                result = np.where(mask_3d, dimmed, result)
                
            elif effect == "bright":
                bright = np.clip(frame * 2.0, 0, 255).astype(np.uint8)
                result = np.where(mask_3d, bright, result)
                
            elif effect == "blur":
                blurred = cv2.GaussianBlur(frame, (71, 71), 0)
                result = np.where(mask_3d, blurred, result)
                
            elif effect == "pixelate":
                small = cv2.resize(frame, (w//25, h//25))
                pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                result = np.where(mask_3d, pix, result)
                
            elif effect == "highlight":
                # Bright colored edge
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, (0, 255, 255), 4)
                
            elif effect == "hide":
                hidden = cv2.GaussianBlur(frame, (99, 99), 0)
                result = np.where(mask_3d, hidden, result)
        
        return result


def main():
    print("\n" + "="*65)
    print("  🎨 SAM ALL OBJECTS - Full Scene Understanding")
    print("="*65)
    print("\n📢 COMMANDS:")
    print('   "Make the wall red"')
    print('   "Color object 3 blue"')
    print('   "Dim the chair"')
    print('   "Blur the person"')
    print('   "Highlight object 5"')
    print('   "Clear"')
    print("\n⌨️  Q=quit C=clear L=list objects")
    print("="*65 + "\n")
    
    # Initialize
    identifier = ObjectIdentifier()
    effects = EffectController(identifier)
    voice = VoiceController(effects.parse)
    voice_ok = voice.start()
    
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ No camera!")
        return
    
    cv2.namedWindow("SAM All Objects", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SAM All Objects", 1280, 720)
    
    fps_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            t0 = time.time()
            
            # Identify all objects
            objects = identifier.process_frame(frame)
            
            # Apply effects
            display = effects.apply(frame)
            
            # Draw object labels and outlines
            for obj in objects:
                mask = obj.mask
                h, w = frame.shape[:2]
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h))
                
                # Draw outline
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Color based on whether effect is applied
                if obj.id in effects.effects:
                    color = (0, 255, 0)  # Green if effect active
                else:
                    color = (255, 255, 255)  # White otherwise
                
                cv2.drawContours(display, contours, -1, color, 2)
                
                # Draw label
                cx, cy = obj.center
                label = f"{obj.id}: {obj.label}"
                
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (cx - 5, cy - th - 5), (cx + tw + 5, cy + 5), (0, 0, 0), -1)
                cv2.putText(display, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS
            elapsed = time.time() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 20:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))
            
            # Info overlay
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (320, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            mic = "🎤" if voice_ok else "🔇"
            cv2.putText(display, f"{mic} SAM All Objects", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Objects: {len(objects)} | FPS: {fps:.1f}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if effects.last_cmd:
                cv2.putText(display, f'"{effects.last_cmd[:35]}"', (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)
            
            cv2.imshow("SAM All Objects", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                effects.effects = {}
                effects.last_cmd = ""
                print("✨ Cleared!")
            elif key == ord('l'):
                print("\n📋 Objects in scene:")
                for obj in objects:
                    eff = effects.effects.get(obj.id, (None, None))[0]
                    eff_str = f" [{eff}]" if eff else ""
                    print(f"  {obj.id}: {obj.label} (conf: {obj.confidence:.2f}){eff_str}")
            elif key == ord('s'):
                fn = f"objects_{int(time.time())}.png"
                cv2.imwrite(fn, display)
                print(f"💾 {fn}")
                
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
        identifier.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

