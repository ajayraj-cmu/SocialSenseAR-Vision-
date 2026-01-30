#!/usr/bin/env python3
"""
SAM Fast & Clean - Optimized segmentation with tight masks

- FAST: Optimized for speed
- CLEAN: Tight mask borders, no chunky outlines
- SMART: Wall/background NEVER affects person

Commands:
- "wall red" / "background blue" - ONLY wall, NOT you
- "face green" / "hands blue" - specific body parts
- "clear"
"""




import cv2
import numpy as np
import time 
import threading
from ultralytics import FastSAM
import mediapipe as mp

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

COLORS = {
    "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
    "yellow": (0, 255, 255), "orange": (0, 165, 255), "pink": (203, 192, 255),
    "purple": (255, 0, 128), "cyan": (255, 255, 0), "white": (255, 255, 255),
}


class FastSegmenter:
    """Optimized segmentation - runs fast with clean masks."""
    
    def __init__(self):
        # MediaPipe for body parts (fast)
        self.selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)  # Faster model
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        
        # FastSAM for objects (only run periodically)
        self.sam = FastSAM("FastSAM-s.pt")
        
        self.cached_objects = []
        self.frame_count = 0
        
    def process(self, frame):
        """Get all masks - body parts and objects."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        masks = {}
        
        # Person mask (every frame - fast)
        selfie_result = self.selfie.process(rgb)
        person_mask = selfie_result.segmentation_mask if selfie_result.segmentation_mask is not None else np.zeros((h, w))
        # Smooth the mask edges
        person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)
        masks["person"] = person_mask
        
        # Face mask (every frame - fast)
        face_result = self.face_mesh.process(rgb)
        face_mask = np.zeros((h, w), dtype=np.float32)
        if face_result.multi_face_landmarks:
            pts = np.array([[int(lm.x * w), int(lm.y * h)] 
                           for lm in face_result.multi_face_landmarks[0].landmark], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(face_mask, hull, 1.0)
            face_mask = cv2.GaussianBlur(face_mask, (5, 5), 0)
        masks["face"] = face_mask
        
        # Hands (every frame - fast)
        hands_result = self.hands.process(rgb)
        left_hand = np.zeros((h, w), dtype=np.float32)
        right_hand = np.zeros((h, w), dtype=np.float32)
        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for hand_lm, handedness in zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness):
                pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_lm.landmark], dtype=np.int32)
                hull = cv2.convexHull(pts)
                hand_mask = np.zeros((h, w), dtype=np.float32)
                cv2.fillConvexPoly(hand_mask, hull, 1.0)
                # Expand slightly
                kernel = np.ones((7, 7), np.uint8)
                hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
                hand_mask = cv2.GaussianBlur(hand_mask, (5, 5), 0)
                
                label = handedness.classification[0].label
                if label == "Left":  # Mirrored
                    right_hand = np.maximum(right_hand, hand_mask)
                else:
                    left_hand = np.maximum(left_hand, hand_mask)
        
        masks["left_hand"] = left_hand
        masks["right_hand"] = right_hand
        masks["hands"] = np.maximum(left_hand, right_hand)
        
        # Background = NOT person (computed from person mask)
        masks["background"] = np.clip(1.0 - person_mask, 0, 1)
        masks["wall"] = masks["background"]  # Alias
        
        # Objects from SAM (every 5 frames for speed)
        self.frame_count += 1
        if self.frame_count % 5 == 0:
            try:
                sam_results = self.sam(frame, device="cpu", retina_masks=True,
                                      imgsz=256, conf=0.4, verbose=False)
                self.cached_objects = []
                if sam_results and sam_results[0].masks is not None:
                    for i, mask_data in enumerate(sam_results[0].masks.data.cpu().numpy()):
                        mask = cv2.resize(mask_data.astype(np.float32), (w, h))
                        # Exclude person area
                        mask = mask * (1 - person_mask)
                        if np.sum(mask > 0.3) > 1000:  # Min area
                            self.cached_objects.append({
                                "id": i,
                                "mask": mask,
                                "label": f"obj_{i}"
                            })
            except:
                pass
        
        # Add cached objects to masks
        for obj in self.cached_objects:
            masks[obj["label"]] = obj["mask"]
        
        return masks, self.cached_objects
    
    def close(self):
        self.selfie.close()
        self.face_mesh.close()
        self.hands.close()


class VoiceController:
    def __init__(self, callback):
        self.callback = callback
        self.running = False
        
    def start(self):
        if not SPEECH_AVAILABLE:
            return False
        try:
            self.rec = sr.Recognizer()
            self.rec.energy_threshold = 250
            self.rec.pause_threshold = 0.4
            self.mic = sr.Microphone()
            with self.mic as s:
                self.rec.adjust_for_ambient_noise(s, duration=0.5)
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()
            return True
        except:
            return False
    
    def stop(self):
        self.running = False
        
    def _loop(self):
        while self.running:
            try:
                with self.mic as s:
                    audio = self.rec.listen(s, timeout=2, phrase_time_limit=5)
                text = self.rec.recognize_google(audio).lower()
                print(f"🎤 {text}")
                self.callback(text)
            except:
                pass


class Effects:
    def __init__(self):
        self.active = {}  # target -> (effect, color)
        self.last = ""
    
    def parse(self, text):
        self.last = text
        
        if any(w in text for w in ["clear", "reset", "off"]):
            self.active = {}
            print("✨ Cleared")
            return
        
        # Find targets
        targets = []
        if "face" in text:
            targets.append("face")
        if "left hand" in text:
            targets.append("left_hand")
        elif "right hand" in text:
            targets.append("right_hand")
        elif "hand" in text:
            targets.append("hands")
        if any(w in text for w in ["wall", "background", "behind"]):
            targets.append("background")
        if "person" in text or (" me " in f" {text} " and not targets):
            targets.append("person")
        
        if not targets:
            targets = ["background"]  # Default
        
        # Find color/effect
        color = None
        for name, val in COLORS.items():
            if name in text:
                color = val
                break
        
        effect = None
        if color:
            effect = ("color", color)
        elif "dim" in text or "dark" in text:
            effect = ("dim", None)
        elif "blur" in text:
            effect = ("blur", None)
        elif "bright" in text:
            effect = ("bright", None)
        elif "pixel" in text:
            effect = ("pixel", None)
        elif "hide" in text:
            effect = ("hide", None)
        
        if effect:
            for t in targets:
                self.active[t] = effect
                print(f"  {t} → {effect[0]}")
    
    def apply(self, frame, masks, person_mask):
        result = frame.copy()
        h, w = frame.shape[:2]
        
        for target, (effect, param) in self.active.items():
            mask = masks.get(target)
            if mask is None:
                continue
            
            # IMPORTANT: If targeting background/wall, SUBTRACT person
            if target in ["background", "wall"]:
                mask = mask * (1 - person_mask)
            
            if not np.any(mask > 0.3):
                continue
            
            # Smooth mask for clean edges
            mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
            mask_3d = np.stack([mask, mask, mask], axis=2)
            
            if effect == "color" and param:
                overlay = np.full_like(frame, param, dtype=np.float32)
                result = (result * (1 - mask_3d * 0.7) + overlay * mask_3d * 0.7).astype(np.uint8)
            elif effect == "dim":
                result = (result * (1 - mask_3d * 0.7) + result * mask_3d * 0.2).astype(np.uint8)
            elif effect == "bright":
                bright = np.clip(frame * 1.8, 0, 255)
                result = (result * (1 - mask_3d) + bright * mask_3d).astype(np.uint8)
            elif effect == "blur":
                blurred = cv2.GaussianBlur(frame, (41, 41), 0)
                result = (result * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
            elif effect == "pixel":
                small = cv2.resize(frame, (w//15, h//15))
                pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                result = np.where(mask_3d > 0.5, pix, result).astype(np.uint8)
            elif effect == "hide":
                hidden = cv2.GaussianBlur(frame, (61, 61), 0)
                result = (result * (1 - mask_3d) + hidden * mask_3d).astype(np.uint8)
        
        return result


def main():
    print("\n" + "="*55)
    print("  🚀 SAM FAST & CLEAN")
    print("="*55)
    print('\n🎤 "wall red" "background blue" "face green"')
    print('   "hands blue" "dim background" "blur wall"')
    print('   "clear" to reset')
    print("\n⌨️  Q=quit C=clear")
    print("="*55 + "\n")
    
    seg = FastSegmenter()
    effects = Effects()
    voice = VoiceController(effects.parse)
    voice_ok = voice.start()
    print("✅ Ready!" if voice_ok else "⚠️ No voice")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cv2.namedWindow("SAM Fast", cv2.WINDOW_NORMAL)
    
    fps_t = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            t0 = time.time()
            
            masks, objects = seg.process(frame)
            display = effects.apply(frame, masks, masks["person"])
            
            # Draw THIN outlines only (1px)
            h, w = frame.shape[:2]
            
            # Person outline - thin cyan
            pm = (masks["person"] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (255, 255, 0), 1)
            
            # Face outline - thin yellow
            fm = (masks["face"] * 255).astype(np.uint8)
            contours, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 255), 1)
            
            # Hands - thin
            for hm in [masks["left_hand"], masks["right_hand"]]:
                hmu = (hm * 255).astype(np.uint8)
                contours, _ = cv2.findContours(hmu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (255, 0, 255), 1)
            
            # FPS
            elapsed = time.time() - t0
            fps_t.append(elapsed)
            if len(fps_t) > 30:
                fps_t.pop(0)
            fps = 1.0 / (sum(fps_t) / len(fps_t))
            
            # Minimal overlay
            cv2.rectangle(display, (5, 5), (200, 50), (0, 0, 0), -1)
            cv2.putText(display, f"FPS: {fps:.0f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            if effects.active:
                txt = " ".join([f"{k}:{v[0]}" for k, v in list(effects.active.items())[:2]])
                cv2.putText(display, txt[:30], (10, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("SAM Fast", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                effects.active = {}
                print("✨ Cleared")
                
    finally:
        voice.stop()
        seg.close()
        cap.release()
        cv2.destroyAllWindows()
        print("👋")


if __name__ == "__main__":
    main()

