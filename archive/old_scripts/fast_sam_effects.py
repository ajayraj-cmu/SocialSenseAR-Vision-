#!/usr/bin/env python3
"""
FastSAM with Voice-Controlled Effects

Real-time segmentation with voice commands to apply effects:
- "dim" / "darken" - darken all objects
- "blur" - blur all objects
- "red" / "blue" / "green" / etc - color overlay
- "highlight" - highlight edges
- "pixelate" - pixelate effect
- "thermal" - thermal vision
- "clear" / "reset" - remove all effects
- "background" - apply effect to background only
- "person" / "face" - apply effect to person/face only

Press Q to quit, M to toggle mic
"""

import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from ultralytics import FastSAM

# Voice recognition
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("⚠️ speech_recognition not available")


class VoiceListener:
    """Background voice listener."""
    
    def __init__(self, callback):
        self.callback = callback
        self.running = False
        self.thread = None
        self.recognizer = None
        self.mic = None
        
    def start(self):
        if not SPEECH_AVAILABLE:
            print("❌ Speech recognition not available")
            return False
        
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.pause_threshold = 0.5
            self.mic = sr.Microphone()
            
            # Calibrate
            print("🎤 Calibrating microphone...")
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            print("✅ Voice listening started!")
            return True
        except Exception as e:
            print(f"❌ Failed to start voice: {e}")
            return False
    
    def stop(self):
        self.running = False
        
    def _listen_loop(self):
        while self.running:
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                    
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"🎤 Heard: '{text}'")
                    self.callback(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"⚠️ Speech service error: {e}")
                    
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                time.sleep(0.5)


class EffectEngine:
    """Applies visual effects to masks."""
    
    def __init__(self):
        self.current_effect = None
        self.effect_color = None
        self.effect_intensity = 0.7
        self.target_mode = "all"  # "all", "background", "person"
        
        # Color map
        self.colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "pink": (203, 192, 255),
            "purple": (255, 0, 128),
            "cyan": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
    
    def parse_command(self, text: str):
        """Parse voice command and set effect."""
        text = text.lower().strip()
        
        # Clear/reset
        if any(w in text for w in ["clear", "reset", "off", "remove", "none", "normal"]):
            self.current_effect = None
            self.target_mode = "all"
            print("✨ Effects cleared")
            return
        
        # Target mode
        if "background" in text:
            self.target_mode = "background"
            print("🎯 Target: background only")
        elif any(w in text for w in ["person", "people", "face", "me"]):
            self.target_mode = "person"
            print("🎯 Target: person only")
        elif "all" in text or "everything" in text:
            self.target_mode = "all"
            print("🎯 Target: all objects")
        
        # Check for colors first
        for color_name, color_val in self.colors.items():
            if color_name in text:
                self.current_effect = "color"
                self.effect_color = color_val
                print(f"🎨 Effect: {color_name} overlay")
                return
        
        # Effect types
        if any(w in text for w in ["dim", "dark", "darken", "shadow"]):
            self.current_effect = "dim"
            print("🌙 Effect: dim/darken")
        elif any(w in text for w in ["bright", "light", "brighten"]):
            self.current_effect = "brighten"
            print("☀️ Effect: brighten")
        elif any(w in text for w in ["blur", "soft", "fuzzy"]):
            self.current_effect = "blur"
            print("🔵 Effect: blur")
        elif any(w in text for w in ["pixel", "pixelate", "minecraft"]):
            self.current_effect = "pixelate"
            print("🟩 Effect: pixelate")
        elif any(w in text for w in ["thermal", "heat", "infrared"]):
            self.current_effect = "thermal"
            print("🔥 Effect: thermal")
        elif any(w in text for w in ["highlight", "outline", "edge"]):
            self.current_effect = "highlight"
            print("✨ Effect: highlight edges")
        elif any(w in text for w in ["gray", "grey", "desaturate", "black and white"]):
            self.current_effect = "grayscale"
            print("⬜ Effect: grayscale")
        elif any(w in text for w in ["invert", "negative"]):
            self.current_effect = "invert"
            print("🔄 Effect: invert")
    
    def apply_effect(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply current effect to masked region."""
        if self.current_effect is None:
            return frame
        
        result = frame.copy()
        mask_bool = mask > 0.5
        mask_3d = np.stack([mask_bool, mask_bool, mask_bool], axis=2)
        
        if self.current_effect == "dim":
            darkened = (frame * 0.3).astype(np.uint8)
            result = np.where(mask_3d, darkened, frame)
            
        elif self.current_effect == "brighten":
            brightened = np.clip(frame * 1.5, 0, 255).astype(np.uint8)
            result = np.where(mask_3d, brightened, frame)
            
        elif self.current_effect == "color" and self.effect_color is not None:
            overlay = np.full_like(frame, self.effect_color, dtype=np.uint8)
            alpha = self.effect_intensity
            blended = (frame * (1 - alpha) + overlay * alpha).astype(np.uint8)
            result = np.where(mask_3d, blended, frame)
            
        elif self.current_effect == "blur":
            blurred = cv2.GaussianBlur(frame, (31, 31), 0)
            result = np.where(mask_3d, blurred, frame)
            
        elif self.current_effect == "pixelate":
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // 16, h // 16), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            result = np.where(mask_3d, pixelated, frame)
            
        elif self.current_effect == "thermal":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            result = np.where(mask_3d, thermal, frame)
            
        elif self.current_effect == "grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = np.where(mask_3d, gray_3d, frame)
            
        elif self.current_effect == "invert":
            inverted = 255 - frame
            result = np.where(mask_3d, inverted, frame)
            
        elif self.current_effect == "highlight":
            # Draw bright edges
            edges = cv2.Canny(frame, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[edges > 0] = [0, 255, 255]  # Yellow edges
            result = np.where(mask_3d & (edges_colored > 0), edges_colored, frame)
        
        return result


def main():
    print("\n" + "="*60)
    print("  FASTSAM + VOICE EFFECTS")
    print("="*60)
    print("\nVoice Commands:")
    print("  'dim' / 'darken'     - Darken objects")
    print("  'blur'               - Blur objects")
    print("  'red/blue/green/...' - Color overlay")
    print("  'pixelate'           - Pixelate effect")
    print("  'thermal'            - Thermal vision")
    print("  'grayscale'          - Black & white")
    print("  'highlight'          - Highlight edges")
    print("  'clear' / 'reset'    - Remove effects")
    print("  'background'         - Target background only")
    print("  'person'             - Target person only")
    print("\nKeys: Q=quit, M=toggle mic, C=clear effects")
    print("="*60 + "\n")
    
    # Load FastSAM
    print("Loading FastSAM...")
    model = FastSAM("FastSAM-s.pt")
    print("✅ FastSAM loaded!")
    
    # Effect engine
    effects = EffectEngine()
    
    # Voice listener
    voice = VoiceListener(effects.parse_command)
    voice_enabled = voice.start()
    
    # Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("✅ Camera ready!")
    
    cv2.namedWindow("FastSAM Effects", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FastSAM Effects", 1280, 720)
    
    frame_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            start_time = time.time()
            
            # Run FastSAM
            results = model(frame, device="cpu", retina_masks=True, imgsz=320, conf=0.4, iou=0.9, verbose=False)
            
            # Process masks and apply effects
            display = frame.copy()
            
            if results and len(results) > 0 and results[0].masks is not None:
                masks_data = results[0].masks.data.cpu().numpy()
                
                # Determine which masks to apply effects to
                all_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                person_mask = np.zeros_like(all_mask)
                
                for i, mask in enumerate(masks_data):
                    # Resize mask
                    mask_resized = cv2.resize(mask.astype(np.float32), 
                                             (frame.shape[1], frame.shape[0]))
                    all_mask = np.maximum(all_mask, mask_resized)
                    
                    # Simple heuristic: larger masks in center = person
                    h, w = mask_resized.shape
                    center_region = mask_resized[h//4:3*h//4, w//4:3*w//4]
                    if np.sum(center_region) > np.sum(mask_resized) * 0.3:
                        person_mask = np.maximum(person_mask, mask_resized)
                
                # Calculate background mask
                background_mask = (1 - person_mask) * all_mask
                
                # Choose target mask based on mode
                if effects.target_mode == "background":
                    target_mask = background_mask
                elif effects.target_mode == "person":
                    target_mask = person_mask
                else:
                    target_mask = all_mask
                
                # Apply effect to target mask
                if effects.current_effect:
                    display = effects.apply_effect(display, target_mask)
                
                # Draw outlines for all objects (subtle)
                for i, mask in enumerate(masks_data):
                    mask_resized = cv2.resize(mask.astype(np.float32), 
                                             (frame.shape[1], frame.shape[0]))
                    mask_uint8 = (mask_resized * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    # Subtle white outline
                    cv2.drawContours(display, contours, -1, (255, 255, 255), 1)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))
            
            # Info overlay
            h, w = display.shape[:2]
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
            
            # Text
            effect_text = effects.current_effect or "none"
            target_text = effects.target_mode
            mic_status = "🎤 ON" if voice_enabled else "🔇 OFF"
            
            cv2.putText(display, f"FastSAM Effects {mic_status}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Effect: {effect_text}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(display, f"Target: {target_text}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, f"FPS: {fps:.1f}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "[Q]uit [M]ic [C]lear", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            cv2.imshow("FastSAM Effects", display)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                if voice_enabled:
                    voice.stop()
                    voice_enabled = False
                    print("🔇 Mic OFF")
                else:
                    voice_enabled = voice.start()
            elif key == ord('c'):
                effects.current_effect = None
                effects.target_mode = "all"
                print("✨ Effects cleared")
            elif key == ord('s'):
                filename = f"effect_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                print(f"💾 Saved: {filename}")
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    
    finally:
        voice.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Done!")


if __name__ == "__main__":
    main()

