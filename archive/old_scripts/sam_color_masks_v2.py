#!/usr/bin/env python3
"""
SAM Color Mask Controller v2

Uses MediaPipe for ACCURATE person detection, then applies effects correctly.

Say natural commands like:
- "make the wall red" (only affects background, NOT you)
- "change the background to blue"  
- "color me green" (only affects YOU)
- "dim the background"
- "clear" / "reset"
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


# Colors
COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255),
    "pink": (203, 192, 255),
    "purple": (255, 0, 128),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "magenta": (255, 0, 255),
    "teal": (128, 128, 0),
    "gold": (0, 215, 255),
}

BACKGROUND_WORDS = ["background", "wall", "behind", "back", "room", "floor", "ceiling", "around"]
PERSON_WORDS = ["person", "me", "myself", "face", "body", "self", "i"]


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
            
            print("🎤 Calibrating mic...")
            with self.mic as src:
                self.recognizer.adjust_for_ambient_noise(src, duration=1)
            
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()
            print("✅ Voice ready!")
            return True
        except Exception as e:
            print(f"❌ Voice error: {e}")
            return False
    
    def stop(self):
        self.running = False
        
    def _loop(self):
        while self.running:
            try:
                with self.mic as src:
                    audio = self.recognizer.listen(src, timeout=3, phrase_time_limit=6)
                text = self.recognizer.recognize_google(audio).lower()
                print(f"\n🎤 \"{text}\"")
                self.callback(text)
            except:
                pass


class EffectController:
    def __init__(self):
        self.effects = {}  # target -> (effect, color)
        self.last_cmd = ""
        
    def parse(self, text: str):
        self.last_cmd = text
        text = text.lower()
        
        # Clear
        if any(w in text for w in ["clear", "reset", "off", "normal", "remove all"]):
            self.effects = {}
            print("✨ Cleared!")
            return
        
        # Find target
        target = None
        if any(w in text for w in BACKGROUND_WORDS):
            target = "background"
        elif any(w in text for w in PERSON_WORDS):
            target = "person"
        else:
            # Default based on context
            if "everything" in text or "all" in text:
                target = "all"
            else:
                target = "background"  # Default to background for safety
        
        # Find color
        color = None
        for name, val in COLORS.items():
            if name in text:
                color = (name, val)
                break
        
        # Find effect
        effect = None
        if color:
            effect = "color"
        elif any(w in text for w in ["dim", "dark"]):
            effect = "dim"
        elif any(w in text for w in ["bright", "light"]):
            effect = "bright"
        elif any(w in text for w in ["blur", "blurry"]):
            effect = "blur"
        elif any(w in text for w in ["pixel"]):
            effect = "pixelate"
        elif any(w in text for w in ["gray", "grey"]):
            effect = "grayscale"
        elif any(w in text for w in ["thermal", "heat"]):
            effect = "thermal"
        
        if effect:
            if effect == "color" and color:
                self.effects[target] = ("color", color[1])
                print(f"🎨 {target.upper()} → {color[0]}")
            else:
                self.effects[target] = (effect, None)
                print(f"✨ {target.upper()} → {effect}")
    
    def apply(self, frame: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """Apply effects. person_mask is 1 where person is, 0 elsewhere."""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Ensure mask is right size
        if person_mask.shape[:2] != (h, w):
            person_mask = cv2.resize(person_mask, (w, h))
        
        # Background mask is inverse of person
        background_mask = 1.0 - person_mask
        
        for target, (effect, params) in self.effects.items():
            # Select correct mask
            if target == "person":
                mask = person_mask
            elif target == "background":
                mask = background_mask
            else:  # all
                mask = np.ones((h, w), dtype=np.float32)
            
            # Threshold mask
            mask_bool = mask > 0.5
            if not np.any(mask_bool):
                continue
            
            mask_3d = np.stack([mask_bool] * 3, axis=2)
            
            # Apply effect
            if effect == "color" and params:
                overlay = np.full_like(frame, params, dtype=np.uint8)
                blended = (frame * 0.4 + overlay * 0.6).astype(np.uint8)
                result = np.where(mask_3d, blended, result)
                
            elif effect == "dim":
                dimmed = (frame * 0.25).astype(np.uint8)
                result = np.where(mask_3d, dimmed, result)
                
            elif effect == "bright":
                bright = np.clip(frame * 1.8, 0, 255).astype(np.uint8)
                result = np.where(mask_3d, bright, result)
                
            elif effect == "blur":
                blurred = cv2.GaussianBlur(frame, (61, 61), 0)
                result = np.where(mask_3d, blurred, result)
                
            elif effect == "pixelate":
                small = cv2.resize(frame, (w//20, h//20))
                pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                result = np.where(mask_3d, pix, result)
                
            elif effect == "grayscale":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                result = np.where(mask_3d, gray3, result)
                
            elif effect == "thermal":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                therm = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                result = np.where(mask_3d, therm, result)
        
        return result


def main():
    print("\n" + "="*60)
    print("  🎨 SAM COLOR MASKS v2 - Accurate Person Detection")
    print("="*60)
    print("\n📢 COMMANDS:")
    print('   "Make the wall red"      - ONLY wall turns red')
    print('   "Background blue"        - ONLY background turns blue')
    print('   "Color me green"         - ONLY you turn green')
    print('   "Dim the background"     - Darkens wall, not you')
    print('   "Clear"                  - Remove all effects')
    print("\n⌨️  Q=quit C=clear S=save")
    print("="*60 + "\n")
    
    # MediaPipe for ACCURATE person segmentation
    print("Loading MediaPipe selfie segmentation...")
    mp_selfie = mp.solutions.selfie_segmentation
    selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)
    print("✅ MediaPipe loaded!")
    
    # FastSAM for object outlines (optional visual)
    print("Loading FastSAM...")
    sam_model = FastSAM("FastSAM-s.pt")
    print("✅ FastSAM loaded!")
    
    # Controllers
    effects = EffectController()
    voice = VoiceController(effects.parse)
    voice_ok = voice.start()
    
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ No camera!")
        return
    
    cv2.namedWindow("SAM Color Masks v2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SAM Color Masks v2", 1280, 720)
    
    fps_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            t0 = time.time()
            
            # Get ACCURATE person mask from MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_result = selfie_seg.process(rgb)
            person_mask = mp_result.segmentation_mask  # Float 0-1
            
            # Apply effects with accurate person/background separation
            display = effects.apply(frame, person_mask)
            
            # Optional: Run FastSAM for object outlines
            try:
                sam_results = sam_model(frame, device="cpu", retina_masks=True,
                                       imgsz=256, conf=0.4, verbose=False)
                if sam_results and sam_results[0].masks is not None:
                    h, w = frame.shape[:2]
                    for mask_data in sam_results[0].masks.data.cpu().numpy():
                        mask_resized = cv2.resize(mask_data, (w, h))
                        mask_uint8 = (mask_resized * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(display, contours, -1, (255, 255, 255), 1)
            except:
                pass
            
            # Draw person outline (cyan)
            person_uint8 = (person_mask * 255).astype(np.uint8)
            person_contours, _ = cv2.findContours(person_uint8, cv2.RETR_EXTERNAL, 
                                                   cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, person_contours, -1, (255, 255, 0), 2)
            
            # FPS
            elapsed = time.time() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))
            
            # Info overlay
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (380, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            mic = "🎤 ON" if voice_ok else "🔇"
            cv2.putText(display, f"{mic} SAM Color Masks v2", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            y = 58
            if effects.effects:
                for tgt, (eff, _) in effects.effects.items():
                    cv2.putText(display, f"  {tgt}: {eff}", (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y += 18
            else:
                cv2.putText(display, "  No effects", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                y += 18
            
            cv2.putText(display, f"FPS: {fps:.1f}", (20, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if effects.last_cmd:
                cmd = effects.last_cmd[:40]
                cv2.putText(display, f'"{cmd}"', (20, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)
            
            cv2.imshow("SAM Color Masks v2", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                effects.effects = {}
                effects.last_cmd = ""
                print("✨ Cleared!")
            elif key == ord('s'):
                fn = f"mask_{int(time.time())}.png"
                cv2.imwrite(fn, display)
                print(f"💾 {fn}")
                
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
        selfie_seg.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

