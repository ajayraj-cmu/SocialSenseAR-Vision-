#!/usr/bin/env python3
"""
SAM + Body Parts Segmentation

Fine-grained segmentation of:
- Face (using MediaPipe Face Mesh)
- Left/Right Hands (using MediaPipe Hands)
- Left/Right Arms
- Torso
- Left/Right Legs
- ALL other objects (using FastSAM + YOLO)

Commands:
- "make my face red"
- "color my hands blue"
- "blur my left hand"
- "dim the wall"
- "highlight my right arm"
"""

import cv2
import numpy as np
import time
import threading
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

# Body parts that can be targeted
BODY_PARTS = [
    "face", "left_hand", "right_hand", "hands",
    "left_arm", "right_arm", "arms",
    "torso", "chest", "body",
    "left_leg", "right_leg", "legs",
    "head", "person"
]

# Object synonyms
OBJECT_WORDS = {
    "wall": ["wall", "background", "behind"],
    "floor": ["floor", "ground"],
    "ceiling": ["ceiling"],
    "chair": ["chair", "seat"],
    "desk": ["desk", "table"],
    "monitor": ["monitor", "screen", "tv"],
}


class BodyPartSegmenter:
    """Creates masks for individual body parts using MediaPipe."""
    
    def __init__(self):
        # Face Mesh for detailed face
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Hands for detailed hand tracking
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Pose for body landmarks
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Selfie segmentation for full body mask
        self.selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        print("  ✓ MediaPipe Body Parts")
    
    def get_masks(self, frame: np.ndarray) -> dict:
        """Get masks for all body parts."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        masks = {
            "face": np.zeros((h, w), dtype=np.float32),
            "left_hand": np.zeros((h, w), dtype=np.float32),
            "right_hand": np.zeros((h, w), dtype=np.float32),
            "left_arm": np.zeros((h, w), dtype=np.float32),
            "right_arm": np.zeros((h, w), dtype=np.float32),
            "torso": np.zeros((h, w), dtype=np.float32),
            "left_leg": np.zeros((h, w), dtype=np.float32),
            "right_leg": np.zeros((h, w), dtype=np.float32),
            "person": np.zeros((h, w), dtype=np.float32),
        }
        
        # Full person mask from selfie segmentation
        selfie_result = self.selfie.process(rgb)
        if selfie_result.segmentation_mask is not None:
            masks["person"] = selfie_result.segmentation_mask
        
        # Face mesh for detailed face mask
        face_result = self.face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                # Create convex hull from face landmarks
                points = []
                for lm in face_landmarks.landmark:
                    points.append([int(lm.x * w), int(lm.y * h)])
                points = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(masks["face"], hull, 1.0)
        
        # Hands
        hands_result = self.hands.process(rgb)
        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for hand_landmarks, handedness in zip(hands_result.multi_hand_landmarks, 
                                                   hands_result.multi_handedness):
                # Determine which hand
                hand_label = handedness.classification[0].label.lower()
                # Note: MediaPipe returns mirrored, so left/right are swapped
                mask_key = "right_hand" if hand_label == "left" else "left_hand"
                
                # Create mask from hand landmarks
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * w), int(lm.y * h)])
                points = np.array(points, dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(masks[mask_key], hull, 1.0)
                
                # Expand hand mask slightly
                kernel = np.ones((15, 15), np.uint8)
                masks[mask_key] = cv2.dilate(masks[mask_key], kernel, iterations=1)
        
        # Pose for arms, torso, legs
        pose_result = self.pose.process(rgb)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            
            # Helper to get point
            def pt(idx):
                return (int(lm[idx].x * w), int(lm[idx].y * h))
            
            # Left arm: shoulder(11) -> elbow(13) -> wrist(15)
            left_arm_pts = np.array([pt(11), pt(13), pt(15)], dtype=np.int32)
            self._draw_limb_mask(masks["left_arm"], left_arm_pts, 40)
            
            # Right arm: shoulder(12) -> elbow(14) -> wrist(16)
            right_arm_pts = np.array([pt(12), pt(14), pt(16)], dtype=np.int32)
            self._draw_limb_mask(masks["right_arm"], right_arm_pts, 40)
            
            # Torso: shoulders and hips
            torso_pts = np.array([pt(11), pt(12), pt(24), pt(23)], dtype=np.int32)
            cv2.fillConvexPoly(masks["torso"], torso_pts, 1.0)
            
            # Left leg: hip(23) -> knee(25) -> ankle(27)
            left_leg_pts = np.array([pt(23), pt(25), pt(27)], dtype=np.int32)
            self._draw_limb_mask(masks["left_leg"], left_leg_pts, 35)
            
            # Right leg: hip(24) -> knee(26) -> ankle(28)
            right_leg_pts = np.array([pt(24), pt(26), pt(28)], dtype=np.int32)
            self._draw_limb_mask(masks["right_leg"], right_leg_pts, 35)
        
        return masks
    
    def _draw_limb_mask(self, mask: np.ndarray, points: np.ndarray, thickness: int):
        """Draw a thick polyline to create limb mask."""
        for i in range(len(points) - 1):
            cv2.line(mask, tuple(points[i]), tuple(points[i+1]), 1.0, thickness)
        # Fill with circles at joints
        for pt in points:
            cv2.circle(mask, tuple(pt), thickness // 2, 1.0, -1)
    
    def close(self):
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()
        self.selfie.close()


class ObjectSegmenter:
    """Segments non-body objects using FastSAM + YOLO."""
    
    def __init__(self):
        self.sam = FastSAM("FastSAM-s.pt")
        self.yolo = YOLO("yolov8n.pt")
        print("  ✓ FastSAM + YOLO")
        
    def get_objects(self, frame: np.ndarray, person_mask: np.ndarray) -> list:
        """Get all non-person objects."""
        h, w = frame.shape[:2]
        objects = []
        
        # YOLO detections
        yolo_results = self.yolo(frame, verbose=False, conf=0.3)
        yolo_data = []
        if yolo_results and len(yolo_results) > 0:
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                label = self.yolo.names[int(box.cls[0])]
                if label != "person":  # Skip person detections
                    yolo_data.append(((x1, y1, x2, y2), label))
        
        # FastSAM masks
        sam_results = self.sam(frame, device="cpu", retina_masks=True,
                              imgsz=320, conf=0.35, verbose=False)
        
        if sam_results and sam_results[0].masks is not None:
            for i, mask_data in enumerate(sam_results[0].masks.data.cpu().numpy()):
                mask = cv2.resize(mask_data.astype(np.float32), (w, h))
                
                # Skip if overlaps with person
                if person_mask is not None:
                    overlap = np.sum((mask > 0.5) & (person_mask > 0.5))
                    if overlap > np.sum(mask > 0.5) * 0.4:
                        continue
                
                # Try to identify with YOLO
                label = "object"
                mask_bbox = self._mask_bbox(mask)
                for yolo_bbox, yolo_label in yolo_data:
                    if self._iou(mask_bbox, yolo_bbox) > 0.3:
                        label = yolo_label
                        break
                
                # Classify by position if unknown
                if label == "object":
                    label = self._classify_position(mask, h, w)
                
                objects.append({
                    "mask": mask,
                    "label": label,
                    "id": len(objects) + 100  # Start object IDs at 100
                })
        
        return objects
    
    def _mask_bbox(self, mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (xs.min(), ys.min(), xs.max(), ys.max())
    
    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter + 1e-6)
    
    def _classify_position(self, mask, h, w):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return "object"
        cy = np.mean(ys)
        area = len(xs)
        if area > h * w * 0.15:
            if cy < h * 0.3:
                return "ceiling"
            elif cy > h * 0.7:
                return "floor"
            return "wall"
        return "object"


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
    def __init__(self):
        self.effects = {}  # target_name -> (effect, params)
        self.last_cmd = ""
    
    def parse(self, text: str, body_parts: list, objects: list):
        self.last_cmd = text
        text = text.lower()
        
        # Clear
        if any(w in text for w in ["clear", "reset", "off", "remove"]):
            self.effects = {}
            print("✨ Cleared!")
            return
        
        # Find target(s)
        targets = []
        
        # Check body parts first
        if "face" in text or "head" in text:
            targets.append("face")
        if "left hand" in text:
            targets.append("left_hand")
        elif "right hand" in text:
            targets.append("right_hand")
        elif "hand" in text or "hands" in text:
            targets.extend(["left_hand", "right_hand"])
        
        if "left arm" in text:
            targets.append("left_arm")
        elif "right arm" in text:
            targets.append("right_arm")
        elif "arm" in text or "arms" in text:
            targets.extend(["left_arm", "right_arm"])
        
        if "torso" in text or "chest" in text or "body" in text:
            targets.append("torso")
        
        if "left leg" in text:
            targets.append("left_leg")
        elif "right leg" in text:
            targets.append("right_leg")
        elif "leg" in text or "legs" in text:
            targets.extend(["left_leg", "right_leg"])
        
        # Check for "me" or "myself" = full person
        if ("me" in text.split() or "myself" in text) and not targets:
            targets.append("person")
        
        # Check objects
        for obj in objects:
            label = obj["label"].lower()
            if label in text:
                targets.append(f"obj_{obj['id']}")
        
        # Check object synonyms
        for category, synonyms in OBJECT_WORDS.items():
            for syn in synonyms:
                if syn in text:
                    # Find matching objects
                    for obj in objects:
                        if obj["label"].lower() == category or category in obj["label"].lower():
                            targets.append(f"obj_{obj['id']}")
                    # Also add category itself for position-based objects
                    targets.append(category)
        
        # Default
        if not targets and "everything" in text:
            targets = ["person"]
        
        if not targets:
            print("❓ No target. Try: 'face', 'hands', 'wall', etc.")
            return
        
        print(f"🎯 Targets: {targets}")
        
        # Find effect
        color = None
        for name, val in COLORS.items():
            if name in text:
                color = (name, val)
                break
        
        effect = None
        if color:
            effect = ("color", color[1])
            print(f"🎨 {color[0]}")
        elif any(w in text for w in ["dim", "dark"]):
            effect = ("dim", None)
        elif any(w in text for w in ["bright"]):
            effect = ("bright", None)
        elif any(w in text for w in ["blur"]):
            effect = ("blur", None)
        elif any(w in text for w in ["pixel"]):
            effect = ("pixelate", None)
        elif any(w in text for w in ["highlight"]):
            effect = ("highlight", None)
        elif any(w in text for w in ["hide"]):
            effect = ("hide", None)
        
        if effect:
            for t in targets:
                self.effects[t] = effect
                print(f"  ✓ {t} → {effect[0]}")
    
    def apply(self, frame: np.ndarray, body_masks: dict, objects: list) -> np.ndarray:
        result = frame.copy()
        h, w = frame.shape[:2]
        
        for target, (effect, params) in self.effects.items():
            # Get the mask for this target
            mask = None
            
            if target in body_masks:
                mask = body_masks[target]
            elif target.startswith("obj_"):
                obj_id = int(target.split("_")[1])
                for obj in objects:
                    if obj["id"] == obj_id:
                        mask = obj["mask"]
                        break
            elif target in ["wall", "floor", "ceiling"]:
                # Find object with this label
                for obj in objects:
                    if obj["label"] == target:
                        if mask is None:
                            mask = obj["mask"].copy()
                        else:
                            mask = np.maximum(mask, obj["mask"])
            
            if mask is None or not np.any(mask > 0.3):
                continue
            
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            mask_bool = mask > 0.3
            mask_3d = np.stack([mask_bool] * 3, axis=2)
            
            # Apply effect
            if effect == "color" and params:
                overlay = np.full_like(frame, params, dtype=np.uint8)
                blended = (frame * 0.3 + overlay * 0.7).astype(np.uint8)
                result = np.where(mask_3d, blended, result)
            elif effect == "dim":
                result = np.where(mask_3d, (frame * 0.2).astype(np.uint8), result)
            elif effect == "bright":
                result = np.where(mask_3d, np.clip(frame * 2, 0, 255).astype(np.uint8), result)
            elif effect == "blur":
                blurred = cv2.GaussianBlur(frame, (51, 51), 0)
                result = np.where(mask_3d, blurred, result)
            elif effect == "pixelate":
                small = cv2.resize(frame, (w//20, h//20))
                pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                result = np.where(mask_3d, pix, result)
            elif effect == "highlight":
                mask_u8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, (0, 255, 255), 4)
            elif effect == "hide":
                hidden = cv2.GaussianBlur(frame, (99, 99), 0)
                result = np.where(mask_3d, hidden, result)
        
        return result


def main():
    print("\n" + "="*65)
    print("  🎨 SAM + BODY PARTS - Fine-Grained Segmentation")
    print("="*65)
    print("\n📢 BODY PART COMMANDS:")
    print('   "make my face red"')
    print('   "color my left hand blue"')
    print('   "blur my hands"')
    print('   "dim my arms"')
    print('   "highlight my torso"')
    print("\n📢 OBJECT COMMANDS:")
    print('   "make the wall green"')
    print('   "blur the background"')
    print("\n⌨️  Q=quit C=clear S=save")
    print("="*65 + "\n")
    
    print("Loading models...")
    body_seg = BodyPartSegmenter()
    obj_seg = ObjectSegmenter()
    print("✅ All models loaded!")
    
    effects = EffectController()
    
    def on_voice(text):
        effects.parse(text, BODY_PARTS, current_objects)
    
    voice = VoiceController(on_voice)
    voice_ok = voice.start()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ No camera!")
        return
    
    cv2.namedWindow("Body Parts + Objects", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Body Parts + Objects", 1280, 720)
    
    fps_times = []
    current_objects = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            t0 = time.time()
            
            # Get body part masks
            body_masks = body_seg.get_masks(frame)
            
            # Get object masks (excluding person)
            current_objects = obj_seg.get_objects(frame, body_masks["person"])
            
            # Apply effects
            display = effects.apply(frame, body_masks, current_objects)
            
            # Draw body part outlines
            h, w = frame.shape[:2]
            outline_colors = {
                "face": (0, 255, 255),      # Cyan
                "left_hand": (255, 0, 0),    # Blue
                "right_hand": (255, 100, 0), # Light blue
                "left_arm": (0, 255, 0),     # Green
                "right_arm": (0, 200, 0),    # Dark green
                "torso": (0, 165, 255),      # Orange
                "left_leg": (255, 0, 255),   # Magenta
                "right_leg": (200, 0, 200),  # Purple
            }
            
            for part, color in outline_colors.items():
                mask = body_masks.get(part)
                if mask is not None and np.any(mask > 0.3):
                    mask_u8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, contours, -1, color, 2)
            
            # Draw object outlines
            for obj in current_objects:
                mask = obj["mask"]
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h))
                mask_u8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (255, 255, 255), 1)
                
                # Label
                ys, xs = np.where(mask > 0.5)
                if len(xs) > 0:
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    cv2.putText(display, obj["label"], (cx, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # FPS
            elapsed = time.time() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 20:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))
            
            # Info overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            mic = "🎤" if voice_ok else "🔇"
            cv2.putText(display, f"{mic} Body Parts + Objects", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, f"FPS: {fps:.1f} | Objects: {len(current_objects)}", (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Active effects
            y = 75
            for target, (eff, _) in list(effects.effects.items())[:3]:
                cv2.putText(display, f"{target}: {eff}", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y += 15
            
            if effects.last_cmd:
                cv2.putText(display, f'"{effects.last_cmd[:40]}"', (20, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 100), 1)
            
            cv2.imshow("Body Parts + Objects", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                effects.effects = {}
                effects.last_cmd = ""
                print("✨ Cleared!")
            elif key == ord('s'):
                fn = f"bodyparts_{int(time.time())}.png"
                cv2.imwrite(fn, display)
                print(f"💾 {fn}")
                
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
        body_seg.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

