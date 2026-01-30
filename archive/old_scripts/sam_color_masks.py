#!/usr/bin/env python3
"""
SAM Color Mask Controller

Say natural commands like:
- "make the wall red"
- "change the background to blue"  
- "color the person green"
- "dim everything"
- "blur the background"
- "clear" or "reset"

Real-time color mask application to segmented objects!
"""

import cv2
import numpy as np
import time
import threading
from ultralytics import FastSAM

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False


# Color definitions
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
    "lime": (0, 255, 128),
    "gold": (0, 215, 255),
}

# Target keywords
BACKGROUND_WORDS = ["background", "wall", "behind", "back", "room", "floor", "ceiling", "environment"]
PERSON_WORDS = ["person", "me", "myself", "face", "body", "people", "human", "self"]
EVERYTHING_WORDS = ["everything", "all", "whole", "entire", "screen"]


class VoiceController:
    """Listens for voice commands."""
    
    def __init__(self, on_command):
        self.on_command = on_command
        self.running = False
        self.thread = None
        
    def start(self):
        if not SPEECH_AVAILABLE:
            print("❌ Speech recognition not available")
            return False
            
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 200
            self.recognizer.pause_threshold = 0.6
            self.recognizer.dynamic_energy_threshold = True
            self.mic = sr.Microphone()
            
            print("🎤 Calibrating...")
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            print("✅ Voice ready! Speak commands.")
            return True
        except Exception as e:
            print(f"❌ Voice error: {e}")
            return False
    
    def stop(self):
        self.running = False
        
    def _loop(self):
        while self.running:
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio).lower()
                print(f"\n🎤 HEARD: \"{text}\"")
                self.on_command(text)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                time.sleep(0.3)


class MaskController:
    """Controls which masks get which colors/effects."""
    
    def __init__(self):
        # Active effects: target -> (effect_type, color/params)
        self.effects = {}  # "background", "person", "all"
        self.last_command = ""
        
    def parse_command(self, text: str):
        """Parse natural language command."""
        self.last_command = text
        text = text.lower()
        
        # Clear commands
        if any(w in text for w in ["clear", "reset", "off", "remove", "normal", "none"]):
            self.effects = {}
            print("✨ All effects cleared!")
            return
        
        # Find target
        target = "all"
        if any(w in text for w in BACKGROUND_WORDS):
            target = "background"
        elif any(w in text for w in PERSON_WORDS):
            target = "person"
        elif any(w in text for w in EVERYTHING_WORDS):
            target = "all"
        
        # Find color
        found_color = None
        for color_name, color_val in COLORS.items():
            if color_name in text:
                found_color = (color_name, color_val)
                break
        
        # Find effect type
        effect_type = None
        if found_color:
            effect_type = "color"
        elif any(w in text for w in ["dim", "dark", "darken"]):
            effect_type = "dim"
        elif any(w in text for w in ["bright", "brighten", "light"]):
            effect_type = "bright"
        elif any(w in text for w in ["blur", "soft", "fuzzy", "blurry"]):
            effect_type = "blur"
        elif any(w in text for w in ["pixel", "pixelate"]):
            effect_type = "pixelate"
        elif any(w in text for w in ["gray", "grey", "black and white", "grayscale"]):
            effect_type = "grayscale"
        elif any(w in text for w in ["thermal", "heat"]):
            effect_type = "thermal"
        elif any(w in text for w in ["hide", "invisible", "remove"]):
            effect_type = "hide"
        
        if effect_type:
            if effect_type == "color" and found_color:
                self.effects[target] = ("color", found_color[1])
                print(f"🎨 {target.upper()} → {found_color[0]}")
            else:
                self.effects[target] = (effect_type, None)
                print(f"✨ {target.upper()} → {effect_type}")
        else:
            print(f"❓ Didn't understand effect. Try: 'make [target] [color/effect]'")
    
    def apply_to_frame(self, frame: np.ndarray, person_mask: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
        """Apply all active effects to frame."""
        result = frame.copy()
        
        for target, (effect_type, params) in self.effects.items():
            # Select mask
            if target == "person":
                mask = person_mask
            elif target == "background":
                mask = background_mask
            else:  # all
                mask = np.maximum(person_mask, background_mask)
            
            if mask is None or not np.any(mask > 0.3):
                continue
            
            mask_bool = mask > 0.3
            mask_3d = np.stack([mask_bool] * 3, axis=2)
            
            # Apply effect
            if effect_type == "color" and params is not None:
                # Color overlay with alpha blending
                overlay = np.full_like(frame, params, dtype=np.uint8)
                alpha = 0.6
                blended = (frame.astype(float) * (1 - alpha) + overlay.astype(float) * alpha)
                result = np.where(mask_3d, blended.astype(np.uint8), result)
                
            elif effect_type == "dim":
                dimmed = (frame * 0.3).astype(np.uint8)
                result = np.where(mask_3d, dimmed, result)
                
            elif effect_type == "bright":
                bright = np.clip(frame * 1.8, 0, 255).astype(np.uint8)
                result = np.where(mask_3d, bright, result)
                
            elif effect_type == "blur":
                blurred = cv2.GaussianBlur(frame, (51, 51), 0)
                result = np.where(mask_3d, blurred, result)
                
            elif effect_type == "pixelate":
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (w//20, h//20))
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                result = np.where(mask_3d, pixelated, result)
                
            elif effect_type == "grayscale":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                result = np.where(mask_3d, gray_3d, result)
                
            elif effect_type == "thermal":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                result = np.where(mask_3d, thermal, result)
                
            elif effect_type == "hide":
                # Heavy blur to hide
                hidden = cv2.GaussianBlur(frame, (99, 99), 0)
                result = np.where(mask_3d, hidden, result)
        
        return result


def classify_mask_as_person(mask: np.ndarray, frame_shape: tuple) -> float:
    """Score how likely a mask is a person (center, medium-large size)."""
    h, w = frame_shape[:2]
    mask_h, mask_w = mask.shape
    
    # Resize mask to frame size
    if mask_h != h or mask_w != w:
        mask = cv2.resize(mask, (w, h))
    
    # Check center presence
    center_y, center_x = h // 2, w // 2
    center_region = mask[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
    center_score = np.mean(center_region) if center_region.size > 0 else 0
    
    # Check size (person usually 10-60% of frame)
    total_area = np.sum(mask > 0.5)
    frame_area = h * w
    size_ratio = total_area / frame_area
    size_score = 1.0 if 0.05 < size_ratio < 0.7 else 0.3
    
    return center_score * size_score


def main():
    print("\n" + "="*65)
    print("  🎨 SAM COLOR MASK CONTROLLER")
    print("="*65)
    print("\n📢 SAY THINGS LIKE:")
    print('   "Make the wall red"')
    print('   "Change the background to blue"')
    print('   "Color me green"')
    print('   "Dim the background"')
    print('   "Blur everything"')
    print('   "Clear" or "Reset"')
    print("\n⌨️  KEYS: Q=quit, C=clear, S=save")
    print("="*65 + "\n")
    
    # Load model
    print("Loading FastSAM...")
    model = FastSAM("FastSAM-s.pt")
    print("✅ Model loaded!")
    
    # Controllers
    masks = MaskController()
    voice = VoiceController(masks.parse_command)
    voice_ok = voice.start()
    
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ No camera!")
        return
    
    cv2.namedWindow("SAM Color Masks", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SAM Color Masks", 1280, 720)
    
    fps_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            t0 = time.time()
            
            # Run segmentation
            results = model(frame, device="cpu", retina_masks=True, 
                          imgsz=320, conf=0.35, iou=0.9, verbose=False)
            
            # Build person and background masks
            h, w = frame.shape[:2]
            person_mask = np.zeros((h, w), dtype=np.float32)
            all_mask = np.zeros((h, w), dtype=np.float32)
            
            if results and results[0].masks is not None:
                for mask_data in results[0].masks.data.cpu().numpy():
                    mask_resized = cv2.resize(mask_data.astype(np.float32), (w, h))
                    all_mask = np.maximum(all_mask, mask_resized)
                    
                    # Classify as person
                    person_score = classify_mask_as_person(mask_resized, frame.shape)
                    if person_score > 0.3:
                        person_mask = np.maximum(person_mask, mask_resized * person_score)
            
            # Background = everything minus person
            background_mask = np.clip(all_mask - person_mask, 0, 1)
            
            # Apply effects
            display = masks.apply_to_frame(frame, person_mask, background_mask)
            
            # Draw mask outlines (subtle)
            if results and results[0].masks is not None:
                for i, mask_data in enumerate(results[0].masks.data.cpu().numpy()):
                    mask_resized = cv2.resize(mask_data, (w, h))
                    mask_uint8 = (mask_resized * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, contours, -1, (255, 255, 255), 1)
            
            # FPS
            elapsed = time.time() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = 1.0 / (sum(fps_times) / len(fps_times))
            
            # Overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            # Status text
            mic_icon = "🎤" if voice_ok else "🔇"
            cv2.putText(display, f"{mic_icon} SAM Color Masks", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show active effects
            y = 60
            if masks.effects:
                for target, (eff, _) in masks.effects.items():
                    cv2.putText(display, f"  {target}: {eff}", (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y += 18
            else:
                cv2.putText(display, "  No effects active", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                y += 18
            
            cv2.putText(display, f"FPS: {fps:.1f}", (20, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Last command
            if masks.last_command:
                cmd_display = masks.last_command[:45] + "..." if len(masks.last_command) > 45 else masks.last_command
                cv2.putText(display, f'"{cmd_display}"', (20, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)
            
            cv2.imshow("SAM Color Masks", display)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                masks.effects = {}
                masks.last_command = ""
                print("✨ Cleared!")
            elif key == ord('s'):
                fn = f"sam_mask_{int(time.time())}.png"
                cv2.imwrite(fn, display)
                print(f"💾 Saved: {fn}")
                
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

