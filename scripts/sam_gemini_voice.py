#!/usr/bin/env python3
"""
SAM + Gemini Voice-Controlled Environment Modifier
ACCESSIBILITY AID FOR SENSORY REGULATION

VISUAL MODES:
1. Visual Noise Cancellation - Blur distracting elements (screens, lights, people)
2. Color Remapping - Change colors of objects to reduce stimulation
3. Motion Dampening - Reduce motion salience for predictability

USAGE:
- Say "hey vibe" to start recording, then speak your command
- End with "thanks" to process the command
- Example: "hey vibe" → "blur my face" → "thanks"
- Gemini Vision analyzes your environment and identifies the objects
- Effects are applied in real-time to detected objects

Requirements:
  pip install google-generativeai speechrecognition pyaudio
  
Set your API key in .env file:
  GEMINI_API_KEY=your-api-key
"""

import cv2
import numpy as np
import time
import os
import threading
import queue
import json
import base64
from collections import deque
from io import BytesIO
from PIL import Image

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file."""
    # Check multiple locations for .env file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, '.env'),  # scripts/.env
        os.path.join(script_dir, '..', '.env'),  # root/.env
    ]
    
    for env_path in possible_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Only set if not already in environment
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            break  # Stop after finding first .env file

# Load .env file before anything else
load_env_file()

# Models
from ultralytics import FastSAM, YOLO
import mediapipe as mp

# Speech recognition
import speech_recognition as sr

# Audio processing (optional)
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  sounddevice not installed. Audio features disabled. Run: pip install sounddevice")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  torch not installed. Audio processing disabled. Run: pip install torch")

try:
    from df.enhance import enhance, init_df
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False
    print("⚠️  DeepFilterNet not installed. Audio isolation disabled. Run: pip install deepfilternet")

# Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Run: pip install google-generativeai")


class GeminiAgent:
    """Handles Gemini text and vision processing."""
    
    def __init__(self):
        # Prioritize GEMINI_API_KEY, fallback to GOOGLE_API_KEY
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        self.scene_objects = []
        self.last_scene_analysis = None
        self.scene_context = {}  # Continuous scene context
        
        # ==========================================
        # GEMINI API RATE LIMITING (Conservative for free tier)
        # ==========================================
        self.vision_call_count = 0
        self.last_vision_call = 0
        self.min_vision_interval = 6.0  # Minimum 6 seconds between vision API calls
        self.max_vision_calls_per_minute = 5  # Limit vision calls per minute (free tier safe)
        self.vision_calls_this_minute = 0
        self.minute_start = time.time()
        
        # ==========================================
        # COMPREHENSIVE FEEDBACK LOOP SYSTEM
        # ==========================================
        self.detection_feedback = {}  # Store Gemini's validation of detections
        self.label_corrections = {}  # Store corrected labels (label -> corrected_label)
        self.confidence_adjustments = {}  # Store confidence adjustments (label -> multiplier)
        self.optimization_history = []  # Track optimizations over time
        self.last_feedback_update = 0
        self.feedback_interval = 10.0  # Update feedback every 10 seconds (conservative for free tier)
        
        # Detection quality metrics
        self.detection_accuracy = {}  # Track accuracy per object type
        self.missing_objects = []  # Objects Gemini sees but we don't detect
        self.false_positives = []  # Objects we detect but Gemini doesn't see
        self.label_confidence_scores = {}  # Track confidence over time
 
 #
        # ==========================================
        # ONLINE "TRAINING" LOOP (self-optimization)
        # Note: this does NOT fine-tune model weights; it optimizes the workflow in real-time.
        # ==========================================
        self.auto_train_enabled = os.environ.get("AUTO_TRAIN", "1") == "1"
        self.auto_train_target_acc = float(os.environ.get("AUTO_TRAIN_TARGET_ACC", "0.90"))
        self.auto_train_required_consecutive = int(os.environ.get("AUTO_TRAIN_CONSEC", "30"))
        self.auto_train_min_coverage = float(os.environ.get("AUTO_TRAIN_MIN_COVERAGE", "0.70"))
        self.auto_train_min_label_acc = float(os.environ.get("AUTO_TRAIN_MIN_LABEL_ACC", "0.75"))
        self._acc_history = deque(maxlen=max(50, self.auto_train_required_consecutive))
        self._train_complete = False
        self._train_complete_reason = ""
        
        # Self-optimization parameters
        self.optimal_yolo_threshold = 0.15  # Adaptive threshold
        self.optimal_sam_conf = 0.3  # Adaptive SAM confidence
        self.optimal_iou_threshold = 0.45  # Adaptive IoU threshold
        self.optimization_iterations = 0
        
        if not self.api_key:
            print("⚠️  No GEMINI_API_KEY or GOOGLE_API_KEY found. Set it with: export GEMINI_API_KEY='your-key'")
            self.available = False
            return
        
        if not GEMINI_AVAILABLE:
            self.available = False
            return
            
        genai.configure(api_key=self.api_key)
        
        # Text model for processing requests
        # Try different model names for compatibility with API version
        # The API version issue suggests we should use 'gemini-pro' instead of 'gemini-1.5-flash'
        model_name = 'gemini-pro'  # Use stable model name
        try:
            # Try gemini-pro first (most compatible)
            self.text_model = genai.GenerativeModel('gemini-pro')
            self.vision_model = genai.GenerativeModel('gemini-pro')
            model_name = 'gemini-pro'
        except Exception as e1:
            # Fallback to gemini-1.5-pro
            try:
                self.text_model = genai.GenerativeModel('gemini-1.5-pro')
                self.vision_model = genai.GenerativeModel('gemini-1.5-pro')
                model_name = 'gemini-1.5-pro'
            except Exception as e2:
                # Last resort: try gemini-1.5-flash (may fail with some API versions)
                try:
                    self.text_model = genai.GenerativeModel('gemini-1.5-flash')
                    self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
                    model_name = 'gemini-1.5-flash'
                except Exception as e3:
                    print(f"⚠️ Could not initialize any Gemini model: {e1}, {e2}, {e3}")
                    raise
        
        self.available = True
        print("✅ Gemini initialized")
    
    def _can_call_vision(self):
        """Check if we're within rate limits for vision API calls."""
        current_time = time.time()
        
        # Reset minute counter if new minute
        if current_time - self.minute_start > 60:
            self.vision_calls_this_minute = 0
            self.minute_start = current_time
        
        # Check per-minute limit
        if self.vision_calls_this_minute >= self.max_vision_calls_per_minute:
            return False
        
        # Check minimum interval
        if current_time - self.last_vision_call < self.min_vision_interval:
            return False
        
        return True
    
    def _record_vision_call(self):
        """Record a vision API call for rate limiting."""
        self.vision_call_count += 1
        self.vision_calls_this_minute += 1
        self.last_vision_call = time.time()
    
    def label_all_segments(self, frame, masks_with_centers):
        """Use Gemini Vision to label ALL SAM segments - ENHANCED OPEN VOCABULARY detection with asset class integration."""
        if not self.available or not self._can_call_vision():
            return None

        try:
            # Convert frame to PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((640, 480))  # Higher resolution for better identification

            # Build position descriptions with more context
            positions = []
            for idx, (mask, existing_label, center) in enumerate(masks_with_centers):
                if center and center[0] > 0:
                    cx, cy = center
                    # More descriptive positions
                    h = "left-side" if cx < 200 else "right-side" if cx > 400 else "center"
                    v = "upper" if cy < 150 else "lower" if cy > 300 else "middle"
                    # Include existing meaningful labels as hints
                    hint = f" (detected: {existing_label})" if existing_label and not existing_label.startswith("~") else ""
                    positions.append(f"{idx+1}: {v} {h}{hint}")

            prompt = f"""You are an EXPERT object identification system. Analyze this image carefully and identify EVERY numbered region with PRECISE, SPECIFIC labels.

Regions to label:
{chr(10).join(positions[:20])}

CRITICAL REQUIREMENTS:
1. Be SPECIFIC - Use exact object names, not generic terms
2. Identify ASSET CLASSES - Categorize objects for proper labeling
3. Include FUNCTIONAL attributes (e.g., "ceiling_light", "table_lamp", "desk_monitor")

ASSET CATEGORIES & EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIGHTING:
- "ceiling_light", "table_lamp", "floor_lamp", "led_strip", "desk_light"
- "window" (natural light source), "skylight", "light_fixture"

SCREENS:
- "laptop_screen", "desktop_monitor", "tv_screen", "tablet", "phone_screen"
- "projector_screen", "smart_display"

PERSON/BODY:
- "face", "left_hand", "right_hand", "left_arm", "right_arm", "torso", "head"
- "person", "body"

FURNITURE:
- "desk", "chair", "table", "shelf", "cabinet", "bed", "couch", "bookshelf"
- "drawer", "wardrobe"

STRUCTURAL:
- "wall", "ceiling", "floor", "door", "window_frame", "column", "corner"

OBJECTS:
- "plant", "clock", "picture_frame", "poster", "book", "keyboard", "mouse"
- "speaker", "headphones", "cup", "bottle", "bag"

Return ONLY this JSON format with SPECIFIC labels:
[{{"region":1,"label":"ceiling_light","asset_class":"lighting"}},
 {{"region":2,"label":"laptop_screen","asset_class":"screen"}},
 {{"region":3,"label":"desk","asset_class":"furniture"}}]"""

            self._record_vision_call()
            response = self.vision_model.generate_content([prompt, pil_image])
            text = response.text.strip()

            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            # Also try to find JSON array directly
            if "[" in text and "]" in text:
                start = text.find("[")
                end = text.rfind("]") + 1
                text = text[start:end]

            results = json.loads(text)

            # Convert to dict: region_index -> (label, asset_class)
            labels = {}
            label_list = []
            asset_classes = {}

            for item in results:
                region = item.get("region", 0) - 1  # Convert to 0-indexed
                label = item.get("label", "unknown").lower().strip()
                asset_class = item.get("asset_class", "object").lower().strip()

                if 0 <= region < len(masks_with_centers) and label:
                    labels[region] = label
                    asset_classes[region] = asset_class
                    label_list.append(f"{label}[{asset_class}]")

            # Store asset classes for semantic understanding
            if not hasattr(self, 'asset_class_map'):
                self.asset_class_map = {}
            self.asset_class_map.update({labels[i]: asset_classes.get(i, "object") for i in labels.keys()})

            if label_list:
                print(f"  🏷️ Gemini Vision ID: {', '.join(label_list[:6])}{'...' if len(label_list) > 6 else ''}")
            return labels

        except json.JSONDecodeError as e:
            print(f"  ⚠️ Gemini JSON parse error - using smart fallback")
            return self._smart_fallback_labels(masks_with_centers, frame)
        except Exception as e:
            if "404" not in str(e) and "Resource" not in str(e):
                print(f"  ⚠️ Gemini labeling: {str(e)[:50]} - using fallback")
            return self._smart_fallback_labels(masks_with_centers, frame)
    
    def _smart_fallback_labels(self, masks_with_centers, frame):
        """Smart fallback labeling based on mask position, size, and brightness."""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        labels = {}
        
        for idx, (mask, existing_label, center) in enumerate(masks_with_centers):
            if mask is None or center is None or center[0] <= 0:
                continue
            
            cx, cy = center
            mask_area = np.sum(mask > 0.5) / (h * w)
            
            # Keep MediaPipe labels (face, hand, arm, etc.)
            if existing_label and existing_label in ["face", "left_hand", "right_hand", "person", 
                                                       "left_arm", "right_arm", "torso", "left_leg", "right_leg"]:
                labels[idx] = existing_label
                continue
            
            # Get brightness of masked region
            try:
                mask_bool = mask > 0.5
                if np.any(mask_bool):
                    region = frame[mask_bool]
                    brightness = np.mean(region)
                else:
                    brightness = 128
            except:
                brightness = 128
            
            # Smart labeling based on position, size, and brightness
            # Very large areas = structural
            if mask_area > 0.15:
                if cy < h * 0.35:
                    labels[idx] = "ceiling"
                elif cy > h * 0.7:
                    labels[idx] = "floor"
                else:
                    labels[idx] = "wall"
            # Bright upper areas = lights
            elif brightness > 180 and cy < h * 0.5 and mask_area < 0.05:
                labels[idx] = "light"
            # Medium upper = ceiling or shelf
            elif cy < h * 0.3 and mask_area > 0.03:
                labels[idx] = "ceiling"
            # Medium lower = furniture
            elif cy > h * 0.5 and 0.02 < mask_area < 0.15:
                labels[idx] = "furniture"
            # Small bright rectangles = screen/monitor
            elif brightness > 150 and 0.01 < mask_area < 0.08:
                labels[idx] = "screen"
            # Edges = walls/doors
            elif cx < w * 0.15 or cx > w * 0.85:
                if mask_area > 0.05:
                    labels[idx] = "wall"
                else:
                    labels[idx] = "object"
            # Default based on size
            elif mask_area > 0.05:
                labels[idx] = "surface"
            else:
                labels[idx] = "object"
        
        if labels:
            print(f"  🔄 Fallback labels: {', '.join(list(labels.values())[:8])}")
        return labels
    
    def analyze_scene(self, frame, mask_labels):
        """Use Gemini Vision to understand the scene and available objects."""
        if not self.available:
            return mask_labels
        
        try:
            # Convert frame to PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            # Resize for faster processing
            pil_image.thumbnail((512, 512))
            
            prompt = f"""Analyze this image and list ALL visible objects/areas you can see.
            
Current detected masks: {mask_labels}

For each object, provide:
1. Object name (simple, like "wall", "ceiling", "chair", "person", "laptop", "window")
2. Location description (like "left side", "background", "center")
3. Approximate color

Return as JSON array:
[{{"name": "object_name", "location": "where", "color": "current_color"}}]

Only return the JSON, nothing else."""

            response = self.vision_model.generate_content([prompt, pil_image])
            
            # Parse response
            text = response.text.strip()
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            self.scene_objects = json.loads(text)
            self.last_scene_analysis = time.time()
            return self.scene_objects
            
        except Exception as e:
            print(f"Vision analysis error: {e}")
            return mask_labels
    
    def analyze_scene_continuous(self, frame, mask_labels, mask_centers):
        """Continuous scene analysis - runs every second for context."""
        if not self.available:
            return
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((640, 480))
            
            # Format labels with positions
            labels_with_pos = []
            for i, (label, center) in enumerate(zip(mask_labels, mask_centers)):
                if center:
                    cx, cy = center
                    h_pos = "left" if cx < frame.shape[1]//3 else "right" if cx > 2*frame.shape[1]//3 else "center"
                    v_pos = "top" if cy < frame.shape[0]//3 else "bottom" if cy > 2*frame.shape[0]//3 else "middle"
                    labels_with_pos.append(f"{label} ({h_pos}-{v_pos})")
                else:
                    labels_with_pos.append(label)
            
            prompt = f"""Analyze this scene continuously. Identify:
1. Light sources (windows, screens, lights)
2. Motion (moving objects, people)
3. High contrast areas
4. Bright or saturated regions
5. Potential sensory triggers

Detected objects: {', '.join(labels_with_pos)}

Return JSON with context:
{{
  "light_sources": ["window", "screen"],
  "motion_detected": true,
  "high_contrast_areas": ["screen", "window"],
  "bright_regions": ["window"],
  "sensory_triggers": ["bright sunlight", "moving people"]
}}"""

            response = self.vision_model.generate_content([prompt, pil_image])
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            scene_context = json.loads(text)
            self.scene_context = scene_context  # Store for natural language processing
            print(f"📊 Scene context: {scene_context.get('sensory_triggers', [])}")
            
        except Exception as e:
            error_str = str(e)
            # Suppress 404 model errors (non-fatal, API compatibility issue)
            if "404" in error_str and "models/" in error_str:
                # Only log occasionally to avoid spam
                if not hasattr(self, '_last_model_error_log'):
                    self._last_model_error_log = 0
                if time.time() - self._last_model_error_log > 60:  # Log once per minute
                    print(f"  ℹ️ Gemini Vision API model compatibility issue (non-fatal - using fallbacks)")
                    self._last_model_error_log = time.time()
            else:
                print(f"Continuous analysis error: {e}")
    
    def comprehensive_feedback_loop(self, frame, mask_labels, yolo_detections, mask_centers):
        """
        COMPREHENSIVE FEEDBACK LOOP: Gemini Vision analyzes scene and self-corrects everything.
        
        This is the core self-optimization system that:
        1. Analyzes what Gemini actually sees in the scene
        2. Compares with our detections (SAM/YOLO)
        3. Identifies errors and mismatches
        4. Automatically corrects labels, thresholds, and parameters
        5. Learns and improves over time
        """
        if not self.available:
            return
        
        current_time = time.time()
        if current_time - self.last_feedback_update < self.feedback_interval:
            return  # Throttle feedback updates
        
        self.last_feedback_update = current_time
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((640, 480))
            
            # Format current detections for comparison
            detected_info = []
            for i, (label, center) in enumerate(zip(mask_labels, mask_centers)):
                if center:
                    cx, cy = center
                    detected_info.append({
                        "label": label,
                        "position": f"({cx}, {cy})",
                        "index": i
                    })
            
            yolo_info = []
            for (x1, y1, x2, y2), label, conf in yolo_detections:
                yolo_info.append({
                    "label": label,
                    "bbox": f"({x1},{y1})-({x2},{y2})",
                    "confidence": conf
                })
            
            prompt = f"""You are a self-correction system analyzing a video feed in real-time.

CURRENT DETECTIONS FROM OUR SYSTEM:
SAM Masks: {json.dumps(detected_info, indent=2)}
YOLO Detections: {json.dumps(yolo_info, indent=2)}

YOUR TASK - Analyze this image and provide comprehensive feedback:

1. WHAT DO YOU ACTUALLY SEE?
   - List ALL visible objects with their correct names
   - Describe their positions (left/center/right, top/middle/bottom)
   - Note any objects we're missing

2. VALIDATE OUR DETECTIONS:
   - Are our labels correct? (e.g., is "wall" actually a wall?)
   - Are we detecting objects that don't exist? (false positives)
   - Are we missing objects you can see? (false negatives)

3. PROVIDE CORRECTIONS:
   - For each incorrect label, provide the correct label
   - Suggest confidence adjustments for uncertain detections
   - Identify objects we should detect but aren't

4. OPTIMIZATION SUGGESTIONS:
   - Should we adjust detection thresholds?
   - Are there segmentation issues?
   - Any improvements to object identification?

Return comprehensive JSON:
{{
  "actual_objects": [
    {{"name": "wall", "position": "background", "confidence": 0.95}},
    {{"name": "laptop", "position": "center", "confidence": 0.9}}
  ],
  "label_corrections": [
    {{"current_label": "item_1", "correct_label": "desk", "confidence": 0.9}},
    {{"current_label": "cabinet", "correct_label": "wall", "confidence": 0.8}}
  ],
  "missing_objects": [
    {{"name": "window", "position": "left side", "should_detect": true}}
  ],
  "false_positives": [
    {{"label": "person", "reason": "This is actually a wall", "confidence": 0.85}}
  ],
  "optimization_suggestions": {{
    "yolo_threshold": 0.15,
    "sam_confidence": 0.3,
    "iou_threshold": 0.45,
    "notes": "Lower YOLO threshold to catch windows"
  }},
  "detection_quality": {{
    "overall_accuracy": 0.75,
    "label_accuracy": 0.8,
    "coverage": 0.7
  }}
}}"""
            
            response = self.vision_model.generate_content([prompt, pil_image])
            text = response.text.strip()
            
            # Extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            feedback = json.loads(text)

            # ------------------------------------------------------------
            # Use "missing_objects" feedback to propose label corrections
            # (maps newly discovered objects onto existing generic SAM masks)
            # ------------------------------------------------------------
            try:
                missing = feedback.get("missing_objects", []) or []
                if missing and mask_centers and mask_labels:
                    h, w = frame.shape[:2]
                    generic_labels = {"item", "item_wide", "item_tall", "small_item", "furniture", "area"}

                    def _pos_score(pos_text: str, cx: float, cy: float) -> float:
                        t = (pos_text or "").lower()
                        score = 0.0
                        # Horizontal
                        if "left" in t:
                            score += 1.0 if cx < w * 0.40 else 0.0
                        if "right" in t:
                            score += 1.0 if cx > w * 0.60 else 0.0
                        if "center" in t or "middle" in t:
                            score += 1.0 if (w * 0.40 <= cx <= w * 0.60) else 0.0
                        # Vertical
                        if "top" in t or "upper" in t:
                            score += 1.0 if cy < h * 0.40 else 0.0
                        if "bottom" in t or "lower" in t:
                            score += 1.0 if cy > h * 0.60 else 0.0
                        if "mid" in t or "middle" in t:
                            score += 1.0 if (h * 0.40 <= cy <= h * 0.60) else 0.0
                        # Background-ish
                        if "background" in t:
                            score += 0.5
                        return score

                    # Build candidate list: (score, label, idx)
                    candidates = []
                    for idx, (lab, cen) in enumerate(zip(mask_labels, mask_centers)):
                        if not cen:
                            continue
                        base = (lab or "").split("_")[0]
                        if base in generic_labels:
                            candidates.append((lab, cen, idx))

                    # Convert missing objects into extra label corrections
                    for m in missing[:8]:
                        name = (m.get("name") or "").strip().lower()
                        pos = m.get("position") or m.get("location") or ""
                        should = m.get("should_detect", True)
                        if not name or not should:
                            continue
                        # Pick best candidate generic mask by position
                        best = None
                        best_s = -1.0
                        for lab, cen, idx in candidates:
                            cx, cy = float(cen[0]), float(cen[1])
                            s = _pos_score(pos, cx, cy)
                            if s > best_s:
                                best_s = s
                                best = (lab, idx)
                        if best and best_s >= 1.0:
                            # Add as a label correction (treat as "new asset" mapping)
                            feedback.setdefault("label_corrections", [])
                            feedback["label_corrections"].append({
                                "current_label": best[0],
                                "correct_label": name,
                                "confidence": 0.80
                            })
            except Exception as e:
                print(f"  ⚠️ Missing-object mapping error: {e}")
            
            # Apply corrections
            self._apply_feedback_corrections(feedback, mask_labels, yolo_detections)
            
            # Update optimization parameters
            self._update_optimization_parameters(feedback)
            
            # Log optimization
            self.optimization_iterations += 1
            
            # Always log feedback activity (not just every 10 iterations)
            corrections = feedback.get('label_corrections', [])
            missing = feedback.get('missing_objects', [])
            false_pos = feedback.get('false_positives', [])
            quality = feedback.get('detection_quality', {})
            
            if corrections or missing or false_pos or self.optimization_iterations % 10 == 0:
                print(f"\n🔄 Gemini Vision Feedback (Iteration {self.optimization_iterations}):")
                if quality.get('overall_accuracy'):
                    print(f"   📊 Overall Accuracy: {quality.get('overall_accuracy', 0):.0%}")
                if corrections:
                    print(f"   ✅ Label Corrections: {len(corrections)}")
                if missing:
                    print(f"   📋 Missing Objects: {len(missing)}")
                if false_pos:
                    print(f"   ⚠️ False Positives: {len(false_pos)}")
                if self.optimization_iterations % 10 == 0:
                    print(f"   🔧 Optimized Thresholds: YOLO={self.optimal_yolo_threshold:.2f}, SAM={self.optimal_sam_conf:.2f}")
            
        except Exception as e:
            error_str = str(e)
            # Suppress 404 model errors (non-fatal, API compatibility issue)
            if "404" in error_str and "models/" in error_str:
                # Only log occasionally to avoid spam
                if not hasattr(self, '_last_comprehensive_error_log'):
                    self._last_comprehensive_error_log = 0
                if time.time() - self._last_comprehensive_error_log > 60:  # Log once per minute
                    print(f"  ℹ️ Comprehensive feedback: Gemini API model compatibility (non-fatal)")
                    self._last_comprehensive_error_log = time.time()
            else:
                print(f"  ⚠️ Feedback loop error: {e}")
    
    def _apply_feedback_corrections(self, feedback, mask_labels, yolo_detections):
        """Apply label corrections and confidence adjustments from feedback."""
        # Apply label corrections
        corrections = feedback.get('label_corrections', [])
        correction_count = 0
        for correction in corrections:
            current = correction.get('current_label')
            correct = correction.get('correct_label')
            conf = correction.get('confidence', 0.8)
            
            if current and correct and conf > 0.7:
                if current not in self.label_corrections:
                    self.label_corrections[current] = {
                        'correct_label': correct,
                        'confidence': conf,
                        'applied_count': 0
                    }
                self.label_corrections[current]['applied_count'] += 1
                correction_count += 1
                print(f"  ✅ Gemini Vision corrected YOLO label: '{current}' → '{correct}' (conf: {conf:.0%})")
        
        # Track missing objects (objects Gemini sees but we don't detect)
        missing = feedback.get('missing_objects', [])
        if missing:
            missing_names = [m.get('name') for m in missing if m.get('should_detect', True)]
            if missing_names:
                print(f"  📋 Gemini Vision found {len(missing_names)} missing objects: {', '.join(missing_names[:5])}")
                print(f"     → These should be detected but aren't in current YOLO/SAM detections")
            self.missing_objects = missing[:10]  # Keep last 10
        
        # Track false positives (objects we detect but don't exist)
        false_pos = feedback.get('false_positives', [])
        if false_pos:
            fp_labels = [fp.get('label') for fp in false_pos]
            print(f"  ⚠️ Gemini Vision found {len(false_pos)} false positives: {', '.join(fp_labels[:5])}")
            print(f"     → These are incorrectly detected by YOLO/SAM")
            self.false_positives = false_pos[:10]  # Keep last 10
        
        # Log actual objects Gemini sees
        actual_objects = feedback.get('actual_objects', [])
        if actual_objects and len(actual_objects) > 0:
            seen_names = [obj.get('name', 'unknown') for obj in actual_objects[:8]]
            print(f"  👁️ Gemini Vision sees in scene: {', '.join(seen_names)}")
        
        if correction_count > 0:
            print(f"  🎯 Applied {correction_count} YOLO label correction(s) from Gemini Vision")
        
        # Log actual objects Gemini sees
        actual_objects = feedback.get('actual_objects', [])
        if actual_objects and len(actual_objects) > 0:
            print(f"  👁️ Gemini Vision sees: {', '.join([obj.get('name', 'unknown') for obj in actual_objects[:5]])}")
        
        if correction_count > 0:
            print(f"  🎯 Applied {correction_count} label correction(s) from Gemini Vision")
    
    def _update_optimization_parameters(self, feedback):
        """Update detection parameters based on feedback."""
        suggestions = feedback.get('optimization_suggestions', {})
        
        # Update YOLO threshold
        if 'yolo_threshold' in suggestions:
            new_threshold = suggestions['yolo_threshold']
            if 0.1 <= new_threshold <= 0.5:
                self.optimal_yolo_threshold = new_threshold
        
        # Update SAM confidence
        if 'sam_confidence' in suggestions:
            new_conf = suggestions['sam_confidence']
            if 0.2 <= new_conf <= 0.5:
                self.optimal_sam_conf = new_conf
        
        # Update IoU threshold
        if 'iou_threshold' in suggestions:
            new_iou = suggestions['iou_threshold']
            if 0.3 <= new_iou <= 0.6:
                self.optimal_iou_threshold = new_iou
        
        # Store optimization history
        self.optimization_history.append({
            'iteration': self.optimization_iterations,
            'parameters': {
                'yolo_threshold': self.optimal_yolo_threshold,
                'sam_confidence': self.optimal_sam_conf,
                'iou_threshold': self.optimal_iou_threshold
            },
            'quality': feedback.get('detection_quality', {})
        })

        # ------------------------------------------------------------
        # Auto-stop when sustained high accuracy is reached
        # ------------------------------------------------------------
        if self.auto_train_enabled and not self._train_complete:
            q = feedback.get("detection_quality", {}) or {}
            overall = q.get("overall_accuracy", None)
            coverage = q.get("coverage", None)
            label_acc = q.get("label_accuracy", None)
            if isinstance(overall, (int, float)):
                self._acc_history.append(float(overall))
                recent = list(self._acc_history)[-self.auto_train_required_consecutive:]
                if len(recent) >= self.auto_train_required_consecutive:
                    ok_acc = all(v >= self.auto_train_target_acc for v in recent)
                    ok_cov = (coverage is None) or (float(coverage) >= self.auto_train_min_coverage)
                    ok_lab = (label_acc is None) or (float(label_acc) >= self.auto_train_min_label_acc)
                    if ok_acc and ok_cov and ok_lab:
                        self._train_complete = True
                        self._train_complete_reason = (
                            f"Reached sustained accuracy: ≥{self.auto_train_target_acc:.0%} "
                            f"for {self.auto_train_required_consecutive} feedback iterations"
                        )
                        print(f"\n🏁 AUTO-TRAIN COMPLETE: {self._train_complete_reason}\n")
        
        # Keep only last 50 optimizations
        if len(self.optimization_history) > 50:
            self.optimization_history = self.optimization_history[-50:]
    
    def get_corrected_label(self, original_label):
        """Get corrected label if available from Gemini Vision feedback."""
        if original_label in self.label_corrections:
            correction = self.label_corrections[original_label]
            # Use correction if confidence is high (lower threshold for faster application)
            if correction['confidence'] > 0.7 and correction['applied_count'] >= 1:
                return correction['correct_label']
        return original_label
    
    def get_missing_objects(self):
        """Get list of objects Gemini Vision detected that we're missing."""
        return self.missing_objects

    def should_stop(self) -> bool:
        """Whether the online training loop reached its target and should stop the app."""
        return bool(self.auto_train_enabled and self._train_complete)

    def stop_reason(self) -> str:
        return self._train_complete_reason or ""
    
    def get_optimal_thresholds(self):
        """Get optimized detection thresholds."""
        return {
            'yolo_threshold': self.optimal_yolo_threshold,
            'sam_confidence': self.optimal_sam_conf,
            'iou_threshold': self.optimal_iou_threshold
        }

    def process_environmental_command(self, command, mask_labels, mask_centers):
        """Process dynamic environmental commands using LLM semantic understanding.

        Examples:
        - "Hey vibe, the lighting is extremely bright" → dim all lights, screens, windows
        - "It's too dark in here" → brighten the environment
        - "Too much visual noise" → blur distracting elements
        """
        if not self.available:
            return []

        try:
            # Get asset class mapping
            asset_map = getattr(self, 'asset_class_map', {})

            # Build context about available objects
            available_context = []
            for i, label in enumerate(mask_labels):
                asset_class = asset_map.get(label, "object")
                pos_info = ""
                if i < len(mask_centers) and mask_centers[i]:
                    cx, cy = mask_centers[i]
                    pos_info = f" at ({cx:.0f},{cy:.0f})"
                available_context.append(f"- {label} [{asset_class}]{pos_info}")

            context_str = "\n".join(available_context)

            prompt = f"""You are an ACCESSIBILITY AI analyzing an environmental sensory request.

USER REQUEST: "{command}"

AVAILABLE OBJECTS IN SCENE:
{context_str}

TASK: Determine which objects should be affected based on the environmental condition described.

EXAMPLES OF SEMANTIC UNDERSTANDING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. "lighting is extremely bright" / "lights hurt" / "too much glare"
   → Affect ALL: lighting sources (ceiling_light, lamp, window, etc.)
   → Affect ALL: screens (monitor, laptop_screen, tv_screen)
   → Action: brightness=0.2-0.4 (darken significantly)

2. "too dark" / "can't see"
   → Affect ALL: lighting sources
   → Action: brightness=1.5-2.0 (brighten)

3. "too much visual noise" / "overstimulating" / "distracting"
   → Affect: screens, moving objects, bright objects
   → Action: blur=true, blur_strength=25-35

4. "screen glare" / "monitor too bright"
   → Affect: screens only (asset_class=screen)
   → Action: brightness=0.3

5. "everything is too colorful" / "saturated"
   → Affect: ALL objects
   → Action: saturation=0.3-0.5

RETURN FORMAT - JSON array of affected objects:
[
  {{"target_label": "ceiling_light", "brightness": 0.3, "reason": "lighting source"}},
  {{"target_label": "window", "brightness": 0.3, "reason": "natural light source"}},
  {{"target_label": "laptop_screen", "brightness": 0.3, "reason": "artificial light source"}}
]

CRITICAL RULES:
1. Match objects by ASSET CLASS when environmental condition mentioned
2. Return SPECIFIC labels from the available objects list
3. Include ALL relevant objects (use asset_class to find them)
4. Choose appropriate effects (brightness, blur, saturation, contrast)
5. If request is specific to one object, only affect that object"""

            response = self.text_model.generate_content(prompt)
            text = response.text.strip()

            # Extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            if "[" in text and "]" in text:
                start = text.find("[")
                end = text.rfind("]") + 1
                text = text[start:end]

            results = json.loads(text)

            if isinstance(results, dict):
                results = [results]

            if results:
                affected = [r.get('target_label') for r in results if r.get('target_label')]
                print(f"  🧠 Semantic AI: Affecting {len(affected)} objects based on environment")
                for r in results[:3]:
                    reason = r.get('reason', 'matched')
                    print(f"     • {r.get('target_label')} - {reason}")

            return results

        except Exception as e:
            print(f"  ⚠️ Environmental processing error: {e}")
            return []
    
    def process_request_with_vision(self, user_request, frame, mask_labels, mask_centers):
        """Use Gemini VISION to SEE the frame and map user request to correct objects.
        
        ACCESSIBILITY MODES:
        1. BLUR MODE - Visual Noise Cancellation: Blur distracting elements
        2. COLOR MODE - Color Remapping: Change colors for reduced stimulation
        3. MOTION MODE - Motion Dampening: Reduce motion salience
        """
        if not self.available:
            return self._fallback_process(user_request, mask_labels)
        
        # Rate limiting check
        if not self._can_call_vision():
            print("⏳ Vision API rate limited, using text-only processing")
            return self._text_only_process(user_request, mask_labels)
        
        try:
            # Convert frame to PIL Image for Gemini Vision
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((512, 384))  # Smaller size for faster/cheaper API calls
            
            # Format labels with positions
            labels_with_pos = []
            for i, label in enumerate(mask_labels):
                if i < len(mask_centers) and mask_centers[i]:
                    cx, cy = mask_centers[i]
                    # Describe position
                    h_pos = "left" if cx < 213 else "right" if cx > 426 else "center"
                    v_pos = "top" if cy < 160 else "bottom" if cy > 320 else "middle"
                    labels_with_pos.append(f"  - '{label}' (at {h_pos}-{v_pos})")
                else:
                    labels_with_pos.append(f"  - '{label}'")
            
            labels_str = "\n".join(labels_with_pos)
            
            # Accessibility-focused prompt for sensory regulation
            prompt = f"""You are an ACCESSIBILITY ASSISTANT for sensory regulation. Analyze the scene and the user's request.

USER REQUEST: "{user_request}"

AVAILABLE OBJECTS (YOLO-detected, use EXACT names):
{labels_str}

⚠️ CRITICAL: "BLUR" ≠ "BLUE"! 
- "blur my face" → apply BLUR effect (blur=true), NOT color blue!
- "make it blue" → apply COLOR blue

VISUAL MODES TO APPLY:

1. BLUR MODE (Visual Noise Cancellation) - APPLY BLUR EFFECT:
   Triggers: "blur", "blurry", "hide", "block", "private", "obscure"
   Action: blur=true, blur_strength=15-35 (higher=more blur)
   Example: "blur my face" → {{"target_label": "face", "blur": true, "blur_strength": 25}}
   
2. DIM/BRIGHTNESS MODE:
   Triggers: "too bright", "lights hurt", "glare", "overstimulated", "harsh light", "sunlight"
   Action: brightness=0.2-0.5 (lower=darker), contrast=0.5-0.8
   
3. COLOR MODE (Color Remapping) - ONLY for explicit color words:
   Triggers: "make X blue", "turn Y green", "change to red"
   Action: color="blue"/"green"/"purple" etc.
   ⚠️ DO NOT use color mode when user says "blur"!

4. MOTION DAMPENING:
   Triggers: "movement", "motion", "too fast", "flickering"
   Action: motion_dampen=true, temporal_smooth=0.7-0.9

OBJECT MAPPING RULES:
- "my face" / "face" → find "face", "person", "man", "woman", "child"
- "screens"/"monitors" → find "monitor", "screen", "laptop", "TV"
- "lights" → find "ceiling light", "lamp", "window"
- "people"/"faces" → find "person", "face", "woman", "man"

RETURN FORMAT (JSON array):
[{{"target_label": "EXACT_LABEL", "blur": true, "blur_strength": 25}}]
[{{"target_label": "window", "brightness": 0.3}}]
[{{"target_label": "wall", "color": "blue"}}]

IMPORTANT: 
1. Only use labels from AVAILABLE OBJECTS list
2. "blur" means BLUR effect, not blue color!"""

            # Use VISION model with image - record the call
            self._record_vision_call()
            response = self.vision_model.generate_content([prompt, pil_image])
            text = response.text.strip()
            
            # Extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            results = json.loads(text)
            
            # Ensure it's a list
            if isinstance(results, dict):
                results = [results]
            
            if results:
                targets = [r.get('target_label') for r in results if r.get('target_label')]
                print(f"👁️ Gemini Vision: Sees {len(targets)} target(s): {targets}")
            
            return results
            
        except Exception as e:
            error_str = str(e)
            # Suppress 404 model errors (non-fatal, API compatibility issue)
            if "404" in error_str and "models/" in error_str:
                # Only log occasionally to avoid spam
                if not hasattr(self, '_last_vision_error_log'):
                    self._last_vision_error_log = 0
                if time.time() - self._last_vision_error_log > 60:  # Log once per minute
                    print(f"  ℹ️ Vision processing: Gemini API model compatibility (using fallback)")
                    self._last_vision_error_log = time.time()
            else:
                print(f"Vision processing error: {e}")
            # Fallback to text-only
            return self._text_only_process(user_request, mask_labels)
    
    def _text_only_process(self, user_request, mask_labels):
        """Text-only fallback when vision fails."""
        try:
            labels_str = "\n".join([f"  - {label}" for label in mask_labels])
            
            prompt = f"""Map this request to available labels.

REQUEST: "{user_request}"
LABELS:
{labels_str}

Return JSON array: [{{"target_label": "label", "color": "color", "confidence": 0.9}}]"""

            response = self.text_model.generate_content(prompt)
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            results = json.loads(text)
            if isinstance(results, dict):
                results = [results]
            return results
        except:
            return self._fallback_process(user_request, mask_labels)
    
    def process_request(self, user_request, available_objects, mask_labels):
        """Legacy text-only method - use process_request_with_vision instead."""
        return self._text_only_process(user_request, mask_labels)
    
    def _fallback_process(self, request, mask_labels):
        """Simple fallback when Gemini is unavailable. Supports multiple targets."""
        request_lower = request.lower()
        results = []
        
        # Color mapping
        color_words = {
            "red": "red", "blue": "blue", "green": "green",
            "yellow": "yellow", "purple": "purple", "orange": "orange",
            "pink": "pink", "cyan": "cyan", "white": "white", "black": "black",
            "dim": "dark_gray", "darken": "dark_gray", "dark": "dark_gray",
            "highlight": "yellow", "bright": "yellow",
            "hide": "black", "remove": "black"
        }
        
        # Common synonyms for structural elements
        synonyms = {
            "wall": ["wall", "background", "behind"],
            "ceiling": ["ceiling", "top", "above"],
            "floor": ["floor", "ground", "bottom"],
            "person": ["me", "myself", "person", "body", "face"],
        }
        
        # Split by "and" to handle multiple requests
        parts = request_lower.replace(",", " and ").split(" and ")
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Find color in this part
            detected_color = "blue"  # default
            for color_word, color_value in color_words.items():
                if color_word in part:
                    detected_color = color_value
                    break
            
            # Find target in this part
            target = None
            
            # Direct label match
            for label in mask_labels:
                if label.lower() in part:
                    target = label
                    break
            
            # Synonym match
            if not target:
                for label, words in synonyms.items():
                    if any(w in part for w in words):
                        # Find matching label in mask_labels
                        for ml in mask_labels:
                            if ml.lower().startswith(label):
                                target = ml
                                break
                        if target:
                            break
            
            if target:
                results.append({
                    "target_label": target,
                    "color": detected_color,
                    "confidence": 0.5
                })
        
        # If no results from splitting, try the whole request
        if not results:
            detected_color = "blue"
            for color_word, color_value in color_words.items():
                if color_word in request_lower:
                    detected_color = color_value
                    break
            
            target = None
            for label in mask_labels:
                if label.lower() in request_lower:
                        target = label
                        break
        
            if target:
                results.append({
            "target_label": target,
            "color": detected_color,
                    "confidence": 0.5
                })
        
        return results


class AudioProcessor:
    """Real-time audio processing with DeepFilterNet for voice isolation."""
    
    def __init__(self):
        self.available = AUDIO_AVAILABLE
        self.use_deepfilter = AUDIO_AVAILABLE and TORCH_AVAILABLE and DEEPFILTER_AVAILABLE
        self.is_active = False
        self.voice_isolation_enabled = False
        self.audio_stream = None
        self.model = None
        self.df_state = None
        self.fs = 48000 if self.use_deepfilter else 44100  # Use 48kHz for DeepFilterNet, 44.1kHz otherwise
        self.blocksize = int(self.fs * 0.1)  # 100ms blocks
        
        # Queues for async processing (only if using DeepFilterNet)
        if self.use_deepfilter:
            self.input_queue = queue.Queue(maxsize=3)
            self.output_queue = queue.Queue(maxsize=3)
            self.processing_lock = threading.Lock()
        
        if self.use_deepfilter:
            try:
                print("  🎵 Loading DeepFilterNet model...")
                self.model, self.df_state, _ = init_df()
                self.model.eval()
                print("  ✅ Audio processor ready (DeepFilterNet)")
                
                # Start processing worker
                self.worker_thread = threading.Thread(target=self._process_audio_worker, daemon=True)
                self.worker_thread.start()
            except Exception as e:
                print(f"  ⚠️ DeepFilterNet init failed: {e}")
                self.use_deepfilter = False
                print("  🎵 Using simple audio dampening (no DeepFilterNet)")
        elif self.available:
            print("  🎵 Audio processor ready (simple dampening mode)")
    
    def _process_audio_worker(self):
        """Worker thread that processes audio asynchronously (DeepFilterNet only)."""
        if not self.use_deepfilter:
            return
            
        while True:
            try:
                audio_data = self.input_queue.get(timeout=0.1)
                if audio_data is None:  # Shutdown signal
                    break
                
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
                
                # Process with DeepFilterNet
                try:
                    with torch.no_grad():
                        enhanced = enhance(self.model, self.df_state, audio_tensor)
                    
                    enhanced_np = enhanced.squeeze().cpu().numpy()
                    
                    try:
                        self.output_queue.put_nowait(enhanced_np)
                    except queue.Full:
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.put_nowait(enhanced_np)
                        except queue.Empty:
                            pass
                except Exception as e:
                    print(f"⚠️ Audio enhance error: {e}")
                
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Audio processing error: {e}")
                continue
    
    def _audio_callback(self, indata, outdata, frames, time, status):
        """Audio stream callback - processes or passes through audio."""
        if not self.voice_isolation_enabled:
            # Passthrough mode
            outdata[:, 0] = indata[:, 0]
            return
        
        # Voice isolation mode
        if self.use_deepfilter:
            # Use DeepFilterNet for advanced noise suppression
            audio_out = indata[:, 0].copy()
            
            # Queue input for processing
            try:
                self.input_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass
            
            # Try to get processed audio
            try:
                enhanced = self.output_queue.get_nowait()
                if len(enhanced) >= frames:
                    audio_out = enhanced[:frames]
                else:
                    audio_out = np.pad(enhanced, (0, frames - len(enhanced)), mode="constant")
            except queue.Empty:
                # Use passthrough if processing is slow
                pass
            
            outdata[:, 0] = audio_out
        else:
            # Simple dampening mode: reduce volume of surrounding audio
            # This is a basic low-pass filter effect to dampen background noise
            audio_in = indata[:, 0].copy()
            
            # Apply volume reduction (dampen by 70%)
            dampened = audio_in * 0.3  # Keep only 30% of original volume
            
            # Simple low-pass filter to reduce high-frequency noise
            # (very basic - just smooth the signal)
            if len(dampened) > 1:
                # Simple moving average for smoothing
                smoothed = np.convolve(dampened, np.ones(3)/3, mode='same')
                outdata[:, 0] = smoothed[:frames] if len(smoothed) >= frames else dampened[:frames]
            else:
                outdata[:, 0] = dampened[:frames]
    
    def start_audio_stream(self):
        """Start the audio processing stream."""
        if not self.available or self.is_active:
            return
        
        try:
            self.audio_stream = sd.Stream(
                samplerate=self.fs,
                blocksize=self.blocksize,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                latency="low"
            )
            self.audio_stream.start()
            self.is_active = True
            print("  🎵 Audio stream started")
        except Exception as e:
            print(f"  ⚠️ Failed to start audio stream: {e}")
    
    def stop_audio_stream(self):
        """Stop the audio processing stream."""
        if self.audio_stream and self.is_active:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.is_active = False
                print("  🎵 Audio stream stopped")
            except Exception as e:
                print(f"  ⚠️ Error stopping audio stream: {e}")
    
    def set_voice_isolation(self, enabled):
        """Enable or disable voice isolation (noise suppression)."""
        self.voice_isolation_enabled = enabled
        if enabled:
            print("  🎵 Voice isolation ENABLED (surrounding audio dampened)")
        else:
            print("  🎵 Voice isolation DISABLED (full audio passthrough)")
    
    def close(self):
        """Clean up resources."""
        self.stop_audio_stream()
        if self.use_deepfilter and hasattr(self, 'input_queue'):
            self.input_queue.put(None)  # Signal worker to stop


class VoiceListener:
    """Wake word voice listener - say "hey vibe" for video, "hey vibe audio" for audio."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Longer pause for natural speech
        
        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()  # Video commands
        self.audio_command_queue = queue.Queue()  # Audio commands (separate)
        self.listening = True
        self.is_recording = False  # Currently recording command
        self.is_processing = False  # Processing speech
        self.recording_mode = None  # "video" or "audio"
        self.last_command = ""
        self.status = "👂 Listening for 'hey vibe' or 'hey vibe audio'..."
        self._lock = threading.Lock()
        
        # Wake words - separate for video and audio
        self.video_wake_phrases = ["hey vibe"]
        self.audio_wake_phrases = ["hey vibe audio", "hey vibe audio mode", "audio mode"]
        self.stop_phrases = ["thanks", "thank you", "done", "stop"]
        
        # Calibrate
        with self.microphone as source:
            print("🎤 Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("✅ Voice listener ready:")
        print("   • Say 'hey vibe' for video commands")
        print("   • Say 'hey vibe audio' for audio commands")
        print("   • Say 'thanks' to process")
        
        # Start continuous listening thread
        self._listening_thread = threading.Thread(target=self._continuous_listen, daemon=True)
        self._listening_thread.start()
    
    def _continuous_listen(self):
        """Continuously listen for wake words: 'hey vibe' (video) or 'hey vibe audio'."""
        while self.listening:
            # Skip if already recording or processing
            if self.is_recording or self.is_processing:
                time.sleep(0.5)
                continue
                
            try:
                with self.microphone as source:
                    # Listen for wake word (short timeout)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    # Check for audio wake word first (longer phrase)
                    if any(phrase in text for phrase in self.audio_wake_phrases):
                        # Double-check we're not already recording
                        with self._lock:
                            if not self.is_recording and not self.is_processing:
                                self.is_recording = True
                                self.recording_mode = "audio"
                            else:
                                continue  # Already recording, skip
                        
                        self.status = "🎵 AUDIO MODE - say your audio command, then 'thanks'"
                        print("🎵 Audio wake word detected! Listening for audio command...")
                        
                        # Small delay to ensure microphone is released
                        time.sleep(0.3)
                        
                        # Start recording audio command
                        threading.Thread(target=self._record_command, daemon=True).start()
                    
                    # Check for video wake word
                    elif any(phrase in text for phrase in self.video_wake_phrases):
                        # Double-check we're not already recording
                        with self._lock:
                            if not self.is_recording and not self.is_processing:
                                self.is_recording = True
                                self.recording_mode = "video"
                            else:
                                continue  # Already recording, skip
                        
                        self.status = "🔴 RECORDING... say your command, then 'thanks'"
                        print("🎤 Video wake word detected! Listening for command...")
                        
                        # Small delay to ensure microphone is released
                        time.sleep(0.3)
                        
                        # Start recording video command
                        threading.Thread(target=self._record_command, daemon=True).start()
                    
                except sr.UnknownValueError:
                    pass  # Not a wake word, continue listening
                except sr.RequestError as e:
                    if self.listening:
                        time.sleep(0.5)  # Brief pause on error
                    
            except sr.WaitTimeoutError:
                continue  # No speech detected, keep listening
            except Exception as e:
                if self.listening:
                    print(f"⚠️ Listening error: {e}")
                    time.sleep(0.5)
    
    def _record_command(self):
        """Record command until 'thanks' is heard."""
        command_parts = []
        max_phrases = 10  # Limit to prevent infinite recording
        phrase_count = 0
        
        # Create a separate recognizer instance for command recording
        cmd_recognizer = sr.Recognizer()
        cmd_recognizer.energy_threshold = 300
        cmd_recognizer.pause_threshold = 0.8
        
        while self.is_recording and phrase_count < max_phrases:
            try:
                # Use a new microphone context to avoid conflicts
                with self.microphone as source:
                    # Adjust for ambient noise quickly
                    cmd_recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    # Listen for command phrase (longer timeout)
                    audio = cmd_recognizer.listen(source, timeout=5, phrase_time_limit=8)
                
                try:
                    text = cmd_recognizer.recognize_google(audio).lower()
                    
                    # Check for stop word
                    if any(phrase in text for phrase in self.stop_phrases):
                        # Remove stop word from last phrase
                        for phrase in self.stop_phrases:
                            text = text.replace(phrase, "").strip()
                        if text:
                            command_parts.append(text)
                        
                        # Stop recording
                        with self._lock:
                            self.is_recording = False
                        self.status = "🔄 Processing..."
                        break
                    else:
                        # Add to command
                        command_parts.append(text)
                        phrase_count += 1
                        self.status = f"🔴 Recording... ({phrase_count}/{max_phrases}) say 'thanks' when done"
                
                except sr.UnknownValueError:
                    # No speech detected, continue listening
                    continue
                except sr.RequestError as e:
                    print(f"⚠️ Speech API error: {e}")
                    with self._lock:
                        self.is_recording = False
                    break
                    
            except sr.WaitTimeoutError:
                # Timeout - assume command is done if we have something
                if command_parts:
                    with self._lock:
                        self.is_recording = False
                    break
                continue
            except OSError as e:
                # Microphone busy or unavailable
                print(f"⚠️ Microphone error: {e}")
                with self._lock:
                    self.is_recording = False
                break
            except Exception as e:
                print(f"⚠️ Recording error: {e}")
                import traceback
                traceback.print_exc()
                with self._lock:
                    self.is_recording = False
                break
        
        # Process collected command
        if command_parts:
            full_command = " ".join(command_parts).strip()
            if full_command and len(full_command) > 2:
                with self._lock:
                    self.is_processing = True
                
                self.last_command = full_command
                
                # Route to appropriate queue based on mode
                if self.recording_mode == "audio":
                    self.audio_command_queue.put(full_command)
                    self.status = f"🎵 Audio: {full_command[:40]}..."
                    print(f"🎵 Audio command: {full_command}")
                else:  # video mode
                self.command_queue.put(full_command)
                self.status = f"✅ Got: {full_command[:40]}..."
                    print(f"🗣️ Video command: {full_command}")
            else:
                self.status = "👂 Listening for 'hey vibe' or 'hey vibe audio'..."
        else:
            self.status = "👂 Listening for 'hey vibe' or 'hey vibe audio'..."
        
        # Reset state
        with self._lock:
            self.is_recording = False
            self.is_processing = False
            self.recording_mode = None
    
    def start_recording(self):
        """Legacy method - no longer used (wake word handles this)."""
        pass
    
    def stop_recording(self):
        """Legacy method - no longer used (wake word handles this)."""
        pass
    
    def is_busy(self):
        """Check if currently recording or processing."""
        with self._lock:
            return self.is_recording or self.is_processing
    
    def get_command(self):
        """Get next video command from queue (non-blocking)."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_audio_command(self):
        """Get next audio command from queue (non-blocking)."""
        try:
            return self.audio_command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        self.listening = False


class EnvironmentController:
    """Main controller combining SAM, Gemini, and Voice."""
    
    def __init__(self):
        print("\n" + "="*60)
        print("  🎙️ SAM + GEMINI VOICE CONTROLLER")
        print("="*60 + "\n")
        
        # Load models
        print("Loading models...")
        self.sam = FastSAM("FastSAM-s.pt")
        print("  ✓ FastSAM")
        
        # INDOOR OBJECT DETECTION - General Indoor Vocabulary
        # Comprehensive vocabulary for ANY indoor environment (home, office, classroom, etc.)
        self.indoor_campus_vocab = [
            # ===== STRUCTURAL ELEMENTS (BACKGROUND) =====
            "wall", "walls", "door", "doors", "window", "windows", "ceiling", "floor", 
            "corridor", "hallway", "room", "corner", "edge", "background", "surface",
            "interior", "space", "area", "section", "panel", "partition",
            
            # ===== FURNITURE =====
            "desk", "desks", "chair", "chairs", "table", "tables", "cabinet", "cabinets",
            "shelf", "shelves", "bookshelf", "bookshelves", "filing cabinet", "sofa", "couch",
            "bed", "bedroom furniture", "dresser", "wardrobe", "closet",
            
            # ===== LIGHTING =====
            "light", "lights", "light fixture", "light fixtures", "ceiling light", "ceiling lights",
            "overhead light", "overhead lights", "lamp", "lamps", "desk lamp", "floor lamp",
            "chandelier", "light switch", "light switches", "bulb", "bulbs",
            
            # ===== ELECTRONICS & SCREENS =====
            "laptop", "laptops", "computer", "computers", "monitor", "monitors", "screen", "screens",
            "keyboard", "keyboards", "mouse", "mice", "printer", "printers", "projector", "projectors",
            "tv", "television", "televisions", "speaker", "speakers", "camera", "cameras",
            "phone", "phones", "cell phone", "cell phones", "tablet", "tablets",
            
            # ===== DISPLAYS & BOARDS =====
            "whiteboard", "whiteboards", "blackboard", "blackboards", "projector screen", 
            "projector screens", "display", "displays",
            
            # ===== PERSONAL ITEMS =====
            "book", "books", "notebook", "notebooks", "paper", "papers", "pen", "pens", 
            "pencil", "pencils", "backpack", "backpacks", "bag", "bags", "handbag", "handbags",
            "binder", "binders", "folder", "folders", "textbook", "textbooks",
            "bottle", "bottles", "cup", "cups", "water bottle", "water bottles", 
            "coffee cup", "coffee cups", "mug", "mugs", "glass", "glasses",
            "headphones", "earbuds", "earphones",
            
            # ===== PEOPLE =====
            "person", "people", "student", "students", "teacher", "teachers", "professor", "professors",
            "man", "woman", "child", "children",
            
            # ===== ROOM ELEMENTS =====
            "door frame", "door frames", "window frame", "window frames", "door handle", "door handles",
            "outlet", "outlets", "electrical outlet", "electrical outlets", "power outlet", "power outlets",
            "vent", "vents", "air vent", "air vents", "heating vent", "heating vents",
            "air conditioning vent", "ac vent", "ventilation",
            
            # ===== APPLIANCES & FIXTURES =====
            "refrigerator", "fridge", "microwave", "oven", "stove", "dishwasher",
            "sink", "sinks", "faucet", "faucets", "toilet", "shower", "bathtub",
            "mirror", "mirrors", "towel", "towels",
            
            # ===== STORAGE & ORGANIZATION =====
            "locker", "lockers", "drawer", "drawers", "box", "boxes", "container", "containers",
            "trash can", "trash cans", "recycling bin", "recycling bins", "wastebasket", "wastebaskets",
            
            # ===== SAFETY & SIGNAGE =====
            "fire extinguisher", "fire extinguishers", "exit sign", "exit signs", 
            "emergency exit", "smoke detector", "smoke detectors",
            
            # ===== DECORATIVE & MISC =====
            "picture", "pictures", "painting", "paintings", "poster", "posters", "frame", "frames",
            "curtain", "curtains", "blinds", "shade", "shades", "rug", "rugs", "carpet",
            "pillow", "pillows", "blanket", "blankets"
        ]
        self._indoor_vocab_base = list(self.indoor_campus_vocab)
        self._dynamic_vocab = set()
        self._vocab_lock = threading.Lock()
        self._last_vocab_refresh = 0.0
        self._vocab_refresh_interval = float(os.environ.get("VOCAB_REFRESH_SEC", "2.0"))
        self._max_total_vocab = int(os.environ.get("MAX_TOTAL_VOCAB", "320"))
        
        # ==========================================
        # NO YOLO - Using only FastSAM + Gemini Vision
        # Gemini Vision provides open-vocabulary labeling for ALL segments
        # ==========================================
        self.use_yolo_world = False
        self.yolo = None  # No YOLO dependency
        print("  ✓ No YOLO (using Gemini Vision for labeling)")
        
        # MediaPipe for detailed body segmentation
        self.selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("  ✓ MediaPipe (selfie + face + hands + pose)")
        
        # Gemini agent
        self.gemini = GeminiAgent()
        
        # Voice listener
        self.voice = VoiceListener()
        
        # Audio processor (for voice isolation and audio modulation)
        self.audio_processor = AudioProcessor()
        if self.audio_processor.available:
            # Start audio stream (will be controlled by commands)
            self.audio_processor.start_audio_stream()
        
        # State
        self.masks = []  # List of (mask, label, center)
        self.active_effects = {}  # label -> color
        self.person_mask = None
        self.frame_count = 0
        self.last_scene_update = 0
        self.yolo_detections = []  # List of ((x1,y1,x2,y2), label, confidence)
        self.label_counts = {}  # Track duplicate labels for uniqueness
        self.current_frame = None  # Store current frame for Gemini Vision
        
        # ==========================================
        # LABEL PERSISTENCE (stop label flickering)
        # ==========================================
        self.persistent_labels = {}  # (region_key) -> (label, confidence, last_seen)
        self.label_lock_threshold = 0.3  # Lower threshold = lock labels faster
        self.label_persistence_time = 10.0  # Keep label for 10 seconds (was 3)
        self.label_change_threshold = 0.5  # Need 50% better confidence to change a label

        # Autopilot (optional): injects test commands to stress-test mapping until stop condition
        self.autopilot_enabled = os.environ.get("AUTO_AUTOPILOT", "1") == "1"
        self._autopilot_last = 0.0
        self._autopilot_interval = float(os.environ.get("AUTO_AUTOPILOT_SEC", "6.0"))
        
        # Performance optimizations - MUCH SLOWER updates for label stability
        self.last_mask_update = 0
        self.mask_update_interval = 1.0  # Update masks every 1 second (much slower)
        self.last_yolo_update = 0
        self.yolo_update_interval = 1.5  # Update YOLO every 1.5 seconds (much slower)
        self.frame_cache = None  # Cache frame for Gemini
        self.cached_contours = {}  # Cache contours for masks
        self.last_frame_shape = None
        
        # ==========================================
        # CLEAN VIEW MODE (V key toggle)
        # ==========================================
        self.clean_view_mode = False  # When True: hide labels/borders, only show active effects
        
        # ==========================================
        # MASK TRACKING (persistence during movement)
        # ==========================================
        self.tracked_masks = {}  # track_id -> {mask, label, center, velocity, last_seen, frames_missing}
        self.next_track_id = 0
        self.max_frames_missing = 15  # Keep mask visible for 15 frames after detection lost
        self.velocity_smoothing = 0.3  # How much to smooth velocity updates
        
        # Feedback loop system
        self.last_feedback_validation = 0
        self.feedback_validation_interval = 15.0  # Validate every 15 seconds (conservative for free tier)
        self.detection_history = []  # Track detection history for learning
        self.adaptive_thresholds = {
            "yolo_conf": 0.10,  # Lower YOLO confidence to detect more objects
            "sam_conf": 0.25,   # Lower SAM confidence
            "matching_iou": 0.08  # Lower matching threshold
        }
        
        # Sync adaptive thresholds with feedback loop
        self._sync_thresholds()
        
        # Color map (BGR)
        self.color_map = {
            "red": (0, 0, 255),
            "blue": (255, 50, 50),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "purple": (255, 0, 255),
            "orange": (0, 165, 255),
            "pink": (203, 192, 255),
            "cyan": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "dark_gray": (40, 40, 40),
        }
        
        print("\n✅ Ready!")
        print("   🎤 Say 'hey vibe' then your command, end with 'thanks'")
        print("   ⌨️  Q=quit  C=clear  S=screenshot  P=toggle autopilot\n")

    def _sanitize_class_name(self, name: str) -> str:
        name = (name or "").strip().lower()
        name = name.replace("-", " ").replace("_", " ")
        name = " ".join(name.split())
        return name

    def _maybe_refresh_yolo_world_vocab(self):
        """Dynamically expand YOLO-World classes based on Gemini missing_objects."""
        if not getattr(self, "use_yolo_world", False):
            return
        if not getattr(self.gemini, "available", False):
            return
        now = time.time()
        if now - self._last_vocab_refresh < self._vocab_refresh_interval:
            return
        self._last_vocab_refresh = now

        missing = self.gemini.get_missing_objects() or []
        if not missing:
            return

        added = []
        with self._vocab_lock:
            for m in missing[:12]:
                nm = self._sanitize_class_name(m.get("name", ""))
                should = m.get("should_detect", True)
                if not nm or not should:
                    continue
                # Avoid duplicates against base vocab
                if nm in self._dynamic_vocab:
                    continue
                if nm in self._indoor_vocab_base:
                    continue
                self._dynamic_vocab.add(nm)
                added.append(nm)

            if not added:
                return

            # Cap total vocab size to keep YOLO-World stable + fast
            dynamic_sorted = sorted(self._dynamic_vocab)
            max_dynamic = max(0, self._max_total_vocab - len(self._indoor_vocab_base))
            if len(dynamic_sorted) > max_dynamic:
                dynamic_sorted = dynamic_sorted[-max_dynamic:]
                self._dynamic_vocab = set(dynamic_sorted)

            new_vocab = self._indoor_vocab_base + dynamic_sorted
            self.indoor_campus_vocab = new_vocab

            try:
                self.yolo_world.set_classes(self.indoor_campus_vocab)
                print(f"  🧠 YOLO-World vocab expanded: +{len(added)} (total={len(self.indoor_campus_vocab)})")
                print(f"     Added: {', '.join(added[:8])}")
            except Exception as e:
                print(f"  ⚠️ YOLO-World set_classes failed: {e}")

    def _autopilot_step(self):
        """Inject a synthetic command to stress-test labeling/effects (optional)."""
        if not self.autopilot_enabled:
            return
        if self.voice.is_busy():
            return
        now = time.time()
        if now - self._autopilot_last < self._autopilot_interval:
            return
        if not self.masks:
            return

        # Prefer non-body-part / non-person targets for environment testing
        labels = [m[1] for m in self.masks]
        avoid_prefix = {"person", "face", "left_hand", "right_hand", "left_arm", "right_arm", "torso", "left_leg", "right_leg"}
        candidates = [l for l in labels if l.split("_")[0] not in avoid_prefix]
        if not candidates:
            candidates = labels

        # Pick a deterministic-ish target to reduce thrash
        target = candidates[int(now) % len(candidates)]
        actions = [
            f"make the {target} blue",
            f"make the {target} red",
            f"dim the {target}",
            f"reduce saturation of the {target}",
            f"lower contrast on the {target}",
        ]
        cmd = actions[int(now / self._autopilot_interval) % len(actions)]
        self._autopilot_last = now
        print(f"\n🤖 AUTOPILOT COMMAND: {cmd}")
        try:
            self._process_voice_command(cmd)
        except Exception as e:
            print(f"  ⚠️ Autopilot error: {e}")
    
    def process_frame(self, frame):
        """Process a single frame - OPTIMIZED."""
        h, w = frame.shape[:2]
        current_time = time.time()
        
        # Create display frame (copy for modifications)
        display = frame.copy()
        
        # Only convert to RGB when needed (not every frame)
        rgb_needed = False
        
        # Update person mask less frequently (every 2 frames for speed)
        if self.frame_count % 2 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_needed = True
            selfie_result = self.selfie.process(rgb)
            self.person_mask = selfie_result.segmentation_mask if selfie_result.segmentation_mask is not None else np.zeros((h, w), dtype=np.float32)
        
        self.frame_count += 1
        
        # Smart mask update: time-based instead of frame-based for consistent performance
        if current_time - self.last_mask_update > self.mask_update_interval:
            if not rgb_needed:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._update_masks(frame, h, w, rgb)
            self.last_mask_update = current_time
            # Clear contour cache when masks update
            self.cached_contours.clear()
        
        # Store current frame for Gemini Vision (only when needed, with throttling)
        if self.frame_count % 10 == 0:  # Update every 10 frames
            self.current_frame = frame.copy()
        
        # FEEDBACK LOOP: Continuous validation and self-correction
        if self.gemini.available and current_time - self.last_feedback_validation > self.feedback_validation_interval:
            mask_labels = [m[1] for m in self.masks]
            mask_centers = [m[2] for m in self.masks]
            
            # Prepare detection data for validation
            detected_objects = []
            for mask, label, center in self.masks:
                detected_objects.append({
                    "label": label,
                    "position": f"({center[0]}, {center[1]})" if center else "unknown",
                    "confidence": 0.7  # Default confidence
                })
            
            # Run feedback validation in background thread
            threading.Thread(
                target=self._run_feedback_loop,
                args=(frame.copy(), detected_objects, self.yolo_detections.copy()),
                daemon=True
            ).start()
            self.last_feedback_validation = current_time
        
        # Update scene analysis with Gemini Vision every 1 second (continuous analysis)
        if self.gemini.available and current_time - self.last_scene_update > 1.0:
            mask_labels = [m[1] for m in self.masks]
            mask_centers = [m[2] for m in self.masks]
            # Continuous scene analysis for better context
            threading.Thread(target=self.gemini.analyze_scene_continuous, 
                           args=(frame.copy(), mask_labels, mask_centers), daemon=True).start()
            self.last_scene_update = current_time
        
        # ==========================================
        # COMPREHENSIVE FEEDBACK LOOP (Self-Correction)
        # ==========================================
        if self.gemini.available:
            mask_labels = [m[1] for m in self.masks]
            mask_centers = [m[2] for m in self.masks]
            # Run feedback loop in background thread (every 500ms)
            threading.Thread(target=self.gemini.comprehensive_feedback_loop,
                           args=(frame.copy(), mask_labels, self.yolo_detections, mask_centers),
                           daemon=True).start()
            
            # Sync adaptive thresholds with feedback loop optimizations
            self._sync_thresholds()
        
        # Check for video commands
        command = self.voice.get_command()
        if command:
            self._process_voice_command(command)
        
        # Check for audio commands (separate pipeline)
        audio_command = self.voice.get_audio_command()
        if audio_command:
            self._process_audio_command(audio_command)
        
        # Optional autopilot to keep generating requests during the online training loop
        self._autopilot_step()
        
        # ==========================================
        # FASTSAM + GEMINI VISION LABELING (PERSISTENT)
        # Labels are cached by position and persist across frames
        # ==========================================
        
        # Initialize persistent label cache (by spatial position)
        if not hasattr(self, '_spatial_labels'):
            self._spatial_labels = {}  # "grid_x_grid_y_size" -> (label, confidence, timestamp)
            self._last_gemini_label_time = 0
        
        # Helper to get spatial key for a mask
        def get_spatial_key(center, mask_area):
            if not center or center[0] <= 0:
                return None
            cx, cy = center
            grid_x = int(cx / w * 16)  # 16x16 grid
            grid_y = int(cy / h * 16)
            size_bucket = 0 if mask_area < 0.02 else (1 if mask_area < 0.1 else 2)
            return f"{grid_x}_{grid_y}_{size_bucket}"
        
        # Request Gemini labels for unlabeled segments (every 3 seconds)
        unlabeled_masks = []
        for idx, (mask, old_label, center) in enumerate(self.masks):
            if mask is None:
                continue
            mask_area = np.sum(mask > 0.5) / (h * w)
            spatial_key = get_spatial_key(center, mask_area)
            if spatial_key and spatial_key not in self._spatial_labels:
                unlabeled_masks.append((idx, mask, center, spatial_key, mask_area))
        
        # Get Gemini labels for unlabeled segments
        if current_time - self._last_gemini_label_time > 3.0 and len(unlabeled_masks) > 0:
            gemini_labels = self.gemini.label_all_segments(frame, self.masks)
            if gemini_labels:
                for idx, label in gemini_labels.items():
                    if idx < len(self.masks):
                        mask, _, center = self.masks[idx]
                        if mask is not None:
                            mask_area = np.sum(mask > 0.5) / (h * w)
                            spatial_key = get_spatial_key(center, mask_area)
                            if spatial_key:
                                # Store with confidence and timestamp (persists for 30 seconds)
                                self._spatial_labels[spatial_key] = (label, 1.0, current_time)
                self._last_gemini_label_time = current_time
        
        # Clean up old labels (>30 seconds old)
        stale_keys = [k for k, v in self._spatial_labels.items() if current_time - v[2] > 30.0]
        for k in stale_keys:
            del self._spatial_labels[k]
        
        # Build label -> mask mapping with persistent labels
        all_labels_to_mask = {}
        updated_masks = []
        label_counts = {}  # For making unique labels
        
        for idx, (mask, old_label, center) in enumerate(self.masks):
            if mask is None:
                continue
            
            mask_area = np.sum(mask > 0.5) / (h * w)
            spatial_key = get_spatial_key(center, mask_area)
            
            # Get label from spatial cache or use old_label
            if spatial_key and spatial_key in self._spatial_labels:
                new_label = self._spatial_labels[spatial_key][0]
            elif old_label and not old_label.startswith("region_") and not old_label.startswith("segment_"):
                # Keep meaningful old labels (from MediaPipe: face, left_hand, etc.)
                new_label = old_label
            else:
                # Temporary label while waiting for Gemini
                if center and center[0] > 0:
                    cx, cy = center
                    h_pos = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
                    v_pos = "top" if cy < h/3 else "bottom" if cy > 2*h/3 else "middle"
                    new_label = f"~{v_pos}_{h_pos}"  # ~ prefix means pending
                else:
                    new_label = f"~segment_{idx}"
            
            # Make labels unique if duplicates
            base_label = new_label
            if base_label in label_counts:
                label_counts[base_label] += 1
                new_label = f"{base_label}_{label_counts[base_label]}"
            else:
                label_counts[base_label] = 1
            
            updated_masks.append((mask, new_label, center))
            all_labels_to_mask[new_label] = mask
        
        self.masks = updated_masks
        
        # ==========================================
        # MASK TRACKING (persistence during movement)
        # ==========================================
        self._update_tracked_masks(updated_masks, h, w)
        
        # Use tracked masks (includes interpolated positions for missing detections)
        tracked_masks_for_display = self._get_tracked_masks_for_display(h, w)
        
        # ==========================================
        # FILTER: Only include masks that have active effects (requested assets)
        # Only requested assets should have colored/blurred/shaded masks
        # Everything else remains unmasked until requested
        # ==========================================
        # Build all_labels_to_mask ONLY from masks that have active effects
        all_labels_to_mask = {}
        for mask, label, center in tracked_masks_for_display:
            # Only include masks that have been requested via voice commands
            matched_effect = self._match_effect(label)
            if matched_effect:
                all_labels_to_mask[label] = mask
        
        # ==========================================
        # DETECT AND MARK ADJACENT BOUNDARIES
        # Multi-color boundary tracing for adjacent objects
        # ==========================================
        adjacent_boundaries = self._detect_adjacent_boundaries(tracked_masks_for_display)

        # ==========================================
        # DRAW BORDERS AND LABELS (for ALL detected objects)
        # Show boundaries for all detected items so user can see what's available
        # But visual effects (shading/blur/color) are only applied when requested
        # ==========================================
        bright_colors = [
            (0, 255, 255),   # Cyan
            (255, 0, 255),   # Magenta
            (0, 255, 0),     # Bright Green
            (255, 255, 0),   # Yellow
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
            (0, 128, 255),   # Light Blue
            (255, 0, 128),   # Pink
            (255, 255, 255), # White
            (100, 255, 200), # Mint
        ]

        # Draw borders and labels for ALL masks (so user can see what's detected)
        # But visual effects will only be applied to requested items
        if not self.clean_view_mode:
            for idx, (mask, label, center) in enumerate(tracked_masks_for_display):
                if mask is None or mask.shape != (h, w):
                    continue

                mask_u8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Check if this mask has an active effect
                matched_effect = self._match_effect(label)
                has_effect = matched_effect is not None

                # Use different border colors: brighter for items with effects, dimmer for others
                if has_effect:
                    border_color = bright_colors[idx % len(bright_colors)]
                    border_width = 2
                else:
                    border_color = (100, 100, 100)  # Dim gray for unrequested items
                    border_width = 1

                cv2.drawContours(display, contours, -1, border_color, border_width)

            # ==========================================
            # DRAW MULTI-COLOR BOUNDARIES FOR ADJACENT OBJECTS
            # Different color for shared boundaries to make adjacent objects distinguishable
            # ==========================================
            boundary_color = (255, 255, 0)  # Yellow for adjacent boundaries
            for boundary_info in adjacent_boundaries:
                boundary_mask = boundary_info['boundary']
                # Draw the boundary with a distinct color
                boundary_contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, boundary_contours, -1, boundary_color, 2)
                
                # Draw label at center
                if center and center[0] > 0 and label:
                    cx, cy = int(center[0]), int(center[1])
                    
                    # Check if this is a confirmed label (from Gemini) or pending
                    is_pending = label.startswith("~")
                    display_label = label[1:] if is_pending else label  # Remove ~ prefix for display
                    
                    # Different styling for confirmed vs pending, and for items with/without effects
                    if is_pending:
                        # Pending labels: smaller, dimmer
                        (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                        cv2.rectangle(display, (cx-2, cy-th-2), (cx+tw+2, cy+2), (50, 50, 50), -1)
                        cv2.putText(display, display_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
                    else:
                        # Confirmed labels: bright for items with effects, dimmer for others
                        if has_effect:
                            (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(display, (cx-3, cy-th-4), (cx+tw+3, cy+4), (0, 0, 0), -1)
                            cv2.putText(display, display_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)
                        else:
                            # Dimmer styling for unrequested items
                            (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.rectangle(display, (cx-2, cy-th-3), (cx+tw+2, cy+3), (30, 30, 30), -1)
                            cv2.putText(display, display_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        # Apply effects ONLY to masks that have been explicitly requested via voice commands
        # all_labels_to_mask already contains only masks with active effects (filtered above)
        # Double-check to ensure no effects are applied unless explicitly requested
        for label, mask in all_labels_to_mask.items():
            # STRICT CHECK: Only apply effects if this label has an active effect
            matched_effect = self._match_effect(label)
            if not matched_effect:
                continue  # Skip - no effect requested for this mask
                # Check if it's a modulation or color
                if isinstance(matched_effect, str) and matched_effect.startswith("mod_"):
                    # Parse modulation parameters
                    try:
                        mod_json = matched_effect[4:]  # Remove "mod_" prefix
                        mod_params = json.loads(mod_json)
                        
                        # Create mask for this region
                        mask_bool = mask > 0.5
                        mask_3d = np.stack([mask, mask, mask], axis=2)
                        
                        # ==========================================
                        # MODE 1: BLUR (Visual Noise Cancellation)
                        # ==========================================
                        if mod_params.get('blur'):
                            blur_strength = int(mod_params.get('blur_strength', 25))
                            # Make sure blur_strength is odd
                            blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
                            
                            # Apply Gaussian blur to the entire frame
                            blurred = cv2.GaussianBlur(display, (blur_strength, blur_strength), 0)
                            
                            # Replace only the masked region with blurred version
                            display[mask_bool] = blurred[mask_bool]
                        
                        # ==========================================
                        # MODE 2: BRIGHTNESS/DIM (Reduce glare)
                        # ==========================================
                        if 'brightness' in mod_params:
                            brightness_factor = mod_params['brightness']
                            # Apply brightness only to masked region
                            display[mask_bool] = np.clip(
                                display[mask_bool].astype(np.float32) * brightness_factor,
                                0, 255
                            ).astype(np.uint8)
                        
                        # ==========================================
                        # MODE 3: CONTRAST REDUCTION
                        # ==========================================
                        if 'contrast' in mod_params:
                            contrast_factor = mod_params['contrast']
                            # Reduce contrast in masked region
                            region = display[mask_bool].astype(np.float32)
                            region = (region - 127.5) * contrast_factor + 127.5
                            display[mask_bool] = np.clip(region, 0, 255).astype(np.uint8)
                        
                        # ==========================================
                        # MODE 4: SATURATION REDUCTION
                        # ==========================================
                        if 'saturation' in mod_params:
                            sat_factor = mod_params['saturation']
                            # Convert masked region to HSV
                            hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV).astype(np.float32)
                            hsv[mask_bool, 1] *= sat_factor
                            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                            display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        
                        # ==========================================
                        # MODE 5: MOTION DAMPENING (temporal smooth)
                        # ==========================================
                        if mod_params.get('motion_dampen'):
                            # Store previous frame region for temporal smoothing
                            if not hasattr(self, 'prev_regions'):
                                self.prev_regions = {}
                            
                            region_key = f"motion_{label}"
                            smooth_factor = mod_params.get('temporal_smooth', 0.7)
                            
                            current_region = display[mask_bool].copy()
                            if region_key in self.prev_regions:
                                # Blend with previous frame
                                prev_region = self.prev_regions[region_key]
                                if prev_region.shape == current_region.shape:
                                    blended = (
                                        current_region.astype(np.float32) * (1 - smooth_factor) +
                                        prev_region.astype(np.float32) * smooth_factor
                                    ).astype(np.uint8)
                                    display[mask_bool] = blended
                            
                            # Store for next frame
                            self.prev_regions[region_key] = current_region
                        
                        # Yellow outline for active modulation effects
                        mask_u8 = (mask * 255).astype(np.uint8)
                        effect_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(display, effect_contours, -1, (0, 255, 255), 3)
                        
                    except Exception as e:
                        print(f"  ⚠️ Modulation error: {e}")
                        # NO FALLBACK - don't apply any effects if there's an error
                        # This ensures only explicitly requested effects are applied
                        pass
                else:
                    # ==========================================
                    # COLOR REMAPPING MODE
                    # ==========================================
                    color_name = matched_effect
                    bgr_color = self.color_map.get(color_name, (255, 50, 50))
                    
                    # Quality color overlay - blend smoothly
                    mask_bool = mask > 0.5
                    mask_3d = np.stack([mask, mask, mask], axis=2)
                    
                    # Create smooth color overlay with edge feathering
                    color_overlay = np.full_like(display, bgr_color, dtype=np.float32)
                    
                    # Blend: preserve some original texture while applying color
                    alpha = 0.6
                    display = np.where(
                        mask_3d > 0.5,
                        (display.astype(np.float32) * (1 - alpha) + color_overlay * alpha).astype(np.uint8),
                        display
                    )
                    
                    # Cyan outline for color effects
                    mask_u8 = (mask * 255).astype(np.uint8)
                    color_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, color_contours, -1, (255, 255, 0), 3)
        
        # ==========================================
        # YOLO LABELS ONLY (no SAM labels)
        # Labels are drawn with the YOLO bounding boxes above
        # ==========================================
        
        return display
    
    def _sync_thresholds(self):
        """Sync adaptive thresholds with feedback loop optimizations."""
        if self.gemini.available:
            optimal = self.gemini.get_optimal_thresholds()
            self.adaptive_thresholds["yolo_conf"] = optimal.get('yolo_threshold', 0.15)
            self.adaptive_thresholds["sam_conf"] = optimal.get('sam_confidence', 0.3)
            self.adaptive_thresholds["matching_iou"] = optimal.get('iou_threshold', 0.45)
    
    def _update_masks(self, frame, h, w, rgb=None):
        """Update segmentation masks and YOLO detections - OPTIMIZED."""
        self.masks = []
        self.label_counts = {}  # Reset label counts
        
        # Reuse RGB if provided, otherwise convert
        if rgb is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Only update YOLO if enough time has passed
        current_time = time.time()
        update_yolo = (current_time - self.last_yolo_update > self.yolo_update_interval)
        
        if update_yolo:
            self.yolo_detections = []
            self.last_yolo_update = current_time
        
        # ==========================================
        # BODY PARTS from MediaPipe (detailed segmentation)
        # ==========================================
        
        # Full person mask
        if self.person_mask is not None and np.any(self.person_mask > 0.5):
            pm = self.person_mask.copy()
            if pm.shape != (h, w):
                pm = cv2.resize(pm, (w, h))
            center = self._mask_center(pm)
            self.masks.append((pm, "person", center))
        
        # Face mask from FaceMesh
        face_result = self.face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            face_mask = np.zeros((h, w), dtype=np.float32)
            pts = np.array([[int(lm.x * w), int(lm.y * h)] 
                           for lm in face_result.multi_face_landmarks[0].landmark], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(face_mask, hull, 1.0)
            face_mask = cv2.GaussianBlur(face_mask, (5, 5), 0)
            center = self._mask_center(face_mask)
            self.masks.append((face_mask, "face", center))
        
        # Hands from MediaPipe Hands
        hands_result = self.hands.process(rgb)
        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for hand_lm, handedness in zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness):
                hand_mask = np.zeros((h, w), dtype=np.float32)
                pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_lm.landmark], dtype=np.int32)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(hand_mask, hull, 1.0)
                # Expand hand region
                kernel = np.ones((9, 9), np.uint8)
                hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
                hand_mask = cv2.GaussianBlur(hand_mask, (5, 5), 0)
                
                # Label based on handedness (mirrored in selfie view)
                label_side = handedness.classification[0].label
                if label_side == "Left":
                    hand_label = "right_hand"  # Mirrored
                else:
                    hand_label = "left_hand"
                
                center = self._mask_center(hand_mask)
                self.masks.append((hand_mask, hand_label, center))
        
        # Body parts from MediaPipe Pose
        pose_result = self.pose.process(rgb)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            
            # Define body part regions using landmark indices
            body_parts = {
                "left_arm": [11, 13, 15],  # shoulder, elbow, wrist
                "right_arm": [12, 14, 16],
                "torso": [11, 12, 24, 23],  # shoulders and hips
                "left_leg": [23, 25, 27, 31],  # hip, knee, ankle, foot
                "right_leg": [24, 26, 28, 32],
            }
            
            for part_name, indices in body_parts.items():
                pts = []
                valid = True
                for idx in indices:
                    if idx < len(lm) and lm[idx].visibility > 0.5:
                        pts.append([int(lm[idx].x * w), int(lm[idx].y * h)])
                    else:
                        valid = False
                        break
                
                if valid and len(pts) >= 3:
                    part_mask = np.zeros((h, w), dtype=np.float32)
                    pts_arr = np.array(pts, dtype=np.int32)
                    
                    # For limbs, create a thicker region
                    if "arm" in part_name or "leg" in part_name:
                        for i in range(len(pts) - 1):
                            cv2.line(part_mask, tuple(pts[i]), tuple(pts[i+1]), 1.0, 25)
                    else:
                        hull = cv2.convexHull(pts_arr)
                        cv2.fillConvexPoly(part_mask, hull, 1.0)
                    
                    # Smooth
                    kernel = np.ones((7, 7), np.uint8)
                    part_mask = cv2.dilate(part_mask, kernel, iterations=1)
                    part_mask = cv2.GaussianBlur(part_mask, (7, 7), 0)
                    
                    if np.any(part_mask > 0.3):
                        center = self._mask_center(part_mask)
                        self.masks.append((part_mask, part_name, center))
        
        # ==========================================
        # NO YOLO - Gemini Vision handles all labeling
        # ==========================================
        # YOLO disabled - using Gemini Vision for open-vocabulary labeling
        yolo_data = []
        used_yolo = set()
        self.yolo_detections = []  # Keep empty - no YOLO
        
        # SAM masks - OPTIMIZED: Adaptive resolution with feedback loop optimization
        used_pixels = np.zeros((h, w), dtype=bool)
        try:
            # Use optimized thresholds from feedback loop
            optimal_thresholds = self.gemini.get_optimal_thresholds()
            sam_conf = optimal_thresholds.get('sam_confidence', 0.3)
            
            # Use smaller resolution for faster processing, but maintain quality
            target_size = min(512, max(320, min(h, w)))  # Adaptive based on frame size
            sam_results = self.sam(frame, device="cpu", retina_masks=True,
                                  imgsz=target_size, conf=sam_conf, verbose=False)
            
            if sam_results and sam_results[0].masks is not None:
                if self.person_mask is not None:
                    used_pixels |= (self.person_mask > 0.5)
                
                for mask_data in sam_results[0].masks.data.cpu().numpy():
                    mask = cv2.resize(mask_data.astype(np.float32), (w, h))
                    mask_binary = mask > 0.5

                    # Remove overlap
                    clean_mask = mask_binary & ~used_pixels

                    if np.sum(clean_mask) < 500:
                        continue

                    # ==========================================
                    # ENHANCED MASK REFINEMENT
                    # Apply advanced edge refinement for tight, quality boundaries
                    # ==========================================
                    clean_mask_float = clean_mask.astype(np.float32)
                    refined_mask = self._refine_mask_edges(frame, clean_mask_float)

                    # Ensure refined mask is valid
                    if refined_mask is not None and np.sum(refined_mask > 0.5) >= 500:
                        clean_mask = refined_mask
                    else:
                        # Fallback to basic morphological refinement
                        mask_u8 = (clean_mask * 255).astype(np.uint8)
                        kernel = np.ones((3,3), np.uint8)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
                        clean_mask = mask_u8.astype(np.float32) / 255.0

                    used_pixels |= (clean_mask > 0.5)
                    
                    # Get label from YOLO or semantic analysis
                    label, yolo_idx = self._get_label(clean_mask, yolo_data, h, w, used_yolo, frame)
                    if yolo_idx is not None:
                        used_yolo.add(yolo_idx)
                    
                    # Validate suspicious YOLO labels - if area is too large, it's likely wrong
                    mask_area = np.sum(clean_mask > 0.3)
                    area_ratio = mask_area / (h * w)
                    if label in ["person", "backpack", "handbag"] and area_ratio > 0.12:
                        # Too large to be these objects - recalculate with semantic only
                        label, _ = self._get_label(clean_mask, [], h, w, set(), frame)
                    
                    # Apply feedback loop corrections from Gemini Vision
                    corrected_label = self.gemini.get_corrected_label(label)
                    if corrected_label != label:
                        print(f"  🔄 Applying Gemini Vision correction: '{label}' → '{corrected_label}'")
                        label = corrected_label
                    
                    # Make label unique if needed
                    label = self._unique_label(label)
                    center = self._mask_center(clean_mask)
                    
                    self.masks.append((clean_mask, label, center))
                    
        except Exception as e:
            print(f"SAM error: {e}")
    
        # ------------------------------------------------------------
        # Ensure YOLO detections become "assets" even if SAM didn't match them
        # This eliminates SAM/Yolo label conflicts and missing-asset issues.
        # ------------------------------------------------------------
        try:
            if yolo_data:
                for idx, (bbox, yolo_label, conf) in enumerate(yolo_data):
                    if idx in used_yolo:
                        continue
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, min(w - 1, int(x1)))
                    x2 = max(0, min(w, int(x2)))
                    y1 = max(0, min(h - 1, int(y1)))
                    y2 = max(0, min(h, int(y2)))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Create a conservative rectangular mask for the YOLO box (fallback asset)
                    bbox_mask = np.zeros((h, w), dtype=np.float32)
                    bbox_mask[y1:y2, x1:x2] = 1.0
                    # Don't overlap with already-used pixels (person + SAM masks)
                    bbox_mask = bbox_mask * (~used_pixels).astype(np.float32)

                    if np.sum(bbox_mask > 0.5) < 400:
                        continue

                    corrected = self.gemini.get_corrected_label(yolo_label)
                    label = corrected if corrected else yolo_label
                    label = self._unique_label(label)
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    self.masks.append((bbox_mask, label, center))
        except Exception as e:
            print(f"  ⚠️ YOLO-asset fallback error: {e}")
    
    def _unique_label(self, label):
        """Make label unique by adding number if duplicate."""
        if label not in self.label_counts:
            self.label_counts[label] = 1
            return label
        else:
            self.label_counts[label] += 1
            return f"{label}_{self.label_counts[label]}"
        
    def _get_region_key(self, cx, cy, area, h, w):
        """Generate a region key for label persistence. Groups nearby positions together."""
        # Quantize position to grid (32x32 grid)
        grid_x = int(cx / w * 32)
        grid_y = int(cy / h * 32)
        # Quantize area to buckets (small, medium, large)
        area_bucket = 0 if area < w*h*0.05 else (1 if area < w*h*0.2 else 2)
        return f"{grid_x}_{grid_y}_{area_bucket}"
    
    def _get_persistent_label(self, region_key, new_label, new_confidence):
        """Get or update persistent label for a region. AGGRESSIVELY prevents label flickering."""
        current_time = time.time()
        
        # Clean up old entries (only if REALLY old)
        stale_keys = [k for k, v in self.persistent_labels.items() 
                      if current_time - v[2] > self.label_persistence_time]
        for k in stale_keys:
            del self.persistent_labels[k]
        
        # Check if we already have a label for this region
        if region_key in self.persistent_labels:
            existing_label, existing_conf, last_seen = self.persistent_labels[region_key]
            
            # VERY STRICT: Only change if new confidence is MUCH higher (50%+ improvement)
            # AND the new label is different
            confidence_improvement = (new_confidence - existing_conf) / (existing_conf + 0.01)
            
            if new_label != existing_label and confidence_improvement > self.label_change_threshold:
                # Change is warranted - update to new label
                self.persistent_labels[region_key] = (new_label, new_confidence, current_time)
                return new_label
            else:
                # KEEP EXISTING LABEL - just refresh the timestamp
                self.persistent_labels[region_key] = (existing_label, existing_conf, current_time)
                return existing_label
        else:
            # New region - lock in the label immediately
            self.persistent_labels[region_key] = (new_label, new_confidence, current_time)
            return new_label
    
    def _get_label(self, mask, yolo_data, h, w, used_yolo, frame=None):
        """Get label for mask from YOLO or semantic analysis. Enhanced with better filtering."""
        mask_bbox = self._mask_bbox(mask)
        ys, xs = np.where(mask > 0.5)
        
        if len(ys) == 0:
            return "area", None
        
        # Mask properties
        cy = np.mean(ys)
        cx = np.mean(xs)
        area = len(ys)
        total_area = h * w
        area_ratio = area / total_area
        
        # Generate region key for label persistence
        region_key = self._get_region_key(cx, cy, area, h, w)
        
        # ============================================
        # METHOD 1: Match with YOLO detections (PRIORITY)
        # ============================================
        # STRICT MATCHING: Ensure SAM masks align EXACTLY with YOLO labels
        best_match = None
        best_score = 0
        best_idx = None
        best_iou = 0
        best_overlap = 0
        
        for idx, (yolo_bbox, label, conf) in enumerate(yolo_data):
            if idx in used_yolo:
                continue
            
            x1, y1, x2, y2 = yolo_bbox
            
            # Check IoU between mask bbox and YOLO bbox
            iou = self._calc_iou(mask_bbox, yolo_bbox)
            
            # Check if YOLO bbox center is inside mask
            yolo_cx = (x1 + x2) // 2
            yolo_cy = (y1 + y2) // 2
            center_in_mask = False
            if 0 <= yolo_cy < h and 0 <= yolo_cx < w:
                center_in_mask = mask[yolo_cy, yolo_cx] > 0.3
            
            # Check if mask center is inside YOLO bbox (with smaller margin for precision)
            margin = 15  # Reduced from 20 for tighter matching
            mask_center_in_yolo = (x1 - margin <= cx <= x2 + margin and 
                                   y1 - margin <= cy <= y2 + margin)
            
            # Calculate overlap percentage of mask pixels inside YOLO box
            mask_in_box = mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            mask_pixels = np.sum(mask > 0.3)
            overlap_ratio = np.sum(mask_in_box > 0.3) / (mask_pixels + 1e-6)
            
            # STRICT MATCHING CRITERIA:
            # 1. IoU must be significant (>0.2 for good match, >0.4 for excellent)
            # 2. Either center must be inside the other's region
            # 3. Overlap ratio must be substantial (>0.4 means 40% of mask is in YOLO box)
            
            # Calculate match score with stricter requirements
            score = 0
            if iou > 0.2:  # Require minimum IoU of 0.2
                score += iou * 2.0  # Increased weight
            if center_in_mask:
                score += 0.5  # Increased weight
            if mask_center_in_yolo:
                score += 0.5  # Increased weight
            if overlap_ratio > 0.4:  # Require at least 40% overlap
                score += overlap_ratio * 0.8  # Increased weight
            
            # STRICT MINIMUM SCORE: Require strong evidence of match
            min_score = 0.5  # Increased from 0.08 - much stricter
            if label in ["person", "backpack", "handbag", "suitcase"]:
                # These are often misclassified - require even higher confidence
                min_score = 0.7
                if area_ratio > 0.1:  # Large area is unlikely to be these objects
                    min_score = 0.9  # Very high threshold for large areas
            
            # Additional validation: Require both IoU and overlap to be good
            if iou < 0.15 and overlap_ratio < 0.3:
                continue  # Skip if both are too low
            
            if score > best_score and score > min_score:
                best_score = score
                best_match = label
                best_idx = idx
                best_iou = iou
                best_overlap = overlap_ratio
        
        if best_match:
            # Additional validation: if YOLO says "person" but mask is huge, likely wrong
            if best_match == "person" and area_ratio > 0.15:
                # Too large to be a person - use semantic labeling instead
                pass  # Fall through to semantic labeling
            else:
                # EXACT MATCH: Use persistent label to prevent flickering
                stable_label = self._get_persistent_label(region_key, best_match, best_score)
                # Log only every 200 iterations (much less spam)
                if getattr(self.gemini, "optimization_iterations", 0) % 200 == 0:
                    print(f"  ✅ SAM↔YOLO: '{stable_label}' (IoU={best_iou:.2f})")
                return stable_label, best_idx
        
        # ============================================
        # METHOD 2: Enhanced semantic labeling for structures
        # ============================================
        
        # Calculate aspect ratio and edge touching
        bbox_h = ys.max() - ys.min() if len(ys) > 0 else 0
        bbox_w = xs.max() - xs.min() if len(xs) > 0 else 0
        aspect = bbox_w / (bbox_h + 1) if bbox_h > 0 else 1
        
        touches_top = np.any(ys < h * 0.05)
        touches_bottom = np.any(ys > h * 0.95)
        touches_left = np.any(xs < w * 0.05)
        touches_right = np.any(xs > w * 0.05)
        touches_edges = touches_top or touches_bottom or touches_left or touches_right
        
        # Very large areas = structural elements (walls, floor, ceiling)
        if area_ratio > 0.12:
            if touches_top and cy < h * 0.4:
                return "ceiling", None
            elif touches_bottom and cy > h * 0.6:
                return "floor", None
            elif (touches_left or touches_right) and area_ratio > 0.08:
                # Vertical wall - check aspect ratio
                if aspect < 0.6:  # Tall and narrow = wall
                    return "wall", None
                else:
                    return "wall", None
            elif cy < h * 0.35:
                return "ceiling", None
            elif cy > h * 0.65:
                return "floor", None
            else:
                return "wall", None
        
        # DOORS: Medium size, vertical, near edges, tall
        if 0.05 < area_ratio < 0.15:
            if aspect < 0.7 and (touches_left or touches_right or touches_bottom):
                if 0.3 < cy/h < 0.8:  # Mid to lower height
                    return "door", None
        
        # WINDOWS: Rectangular, mid-height, often bright
        if 0.03 < area_ratio < 0.12 and 0.25 < cy/h < 0.75:
            if 1.2 < aspect < 3.5:  # Wider than tall
                # Check brightness if frame available (windows are usually bright)
                if frame is not None:
                    try:
                        mask_region = frame[int(ys.min()):int(ys.max()), int(xs.min()):int(xs.max())]
                        if len(mask_region) > 0:
                            gray = cv2.cvtColor(mask_region, cv2.COLOR_BGR2GRAY)
                            brightness = np.mean(gray)
                            # Windows are typically brighter than walls
                            if brightness > 80:  # Threshold for "bright"
                                return "window", None
                    except:
                        pass
                return "window", None  # Shape matches, assume window
        
        # Medium areas (3-15%) - furniture-sized, use descriptive names
        if area_ratio > 0.03:
            bbox_h = ys.max() - ys.min()
            bbox_w = xs.max() - xs.min()
            aspect = bbox_w / (bbox_h + 1)
            
            if cy > h * 0.55:
                # Lower area - furniture
                if aspect > 2:
                    return "table", None
                elif aspect < 0.5:
                    return "cabinet", None
                else:
                    return "furniture", None
            elif cy < h * 0.4:
                # Upper area
                if aspect > 1.5:
                    return "shelf", None
                else:
                    return "cabinet", None
        else:
                # Middle area
                if aspect > 2:
                    return "monitor", None
                else:
                    return "furniture", None
        
        # Small objects (0.5-3%) - items on surfaces
        if area_ratio > 0.005:
            bbox_h = ys.max() - ys.min()
            bbox_w = xs.max() - xs.min()
            aspect = bbox_w / (bbox_h + 1)
            
            # Try to guess based on shape
            if aspect > 1.5:
                return "item_wide", None  # Could be keyboard, book
            elif aspect < 0.7:
                return "item_tall", None  # Could be bottle, cup
            else:
                return "item", None  # Square-ish item
        
        # Very small objects
        return "small_item", None
    
    def _build_effect(self, color=None, brightness=None, saturation=None, contrast=None,
                     highlight_suppression=None, blur=None, blur_strength=None,
                     motion_dampen=None, temporal_smooth=None):
        """Build effect string from parameters. Returns color name or modulation dict."""
        # If only color specified, return color name
        if color and not any([brightness, saturation, contrast, blur, motion_dampen]):
            return color
        
        # Build modulation dictionary
        mod_dict = {}
        
        # Blur mode (Visual Noise Cancellation)
        if blur:
            mod_dict['blur'] = True
            mod_dict['blur_strength'] = blur_strength or 25
        
        # Brightness/Dim mode
        if brightness is not None:
            mod_dict['brightness'] = brightness
        
        # Saturation mode
        if saturation is not None:
            mod_dict['saturation'] = saturation
        
        # Contrast mode
        if contrast is not None:
            mod_dict['contrast'] = contrast
        
        # Highlight suppression
        if highlight_suppression is not None:
            mod_dict['highlight_suppression'] = highlight_suppression
        
        # Motion dampening mode
        if motion_dampen:
            mod_dict['motion_dampen'] = True
            mod_dict['temporal_smooth'] = temporal_smooth or 0.7
        
        # If we have any modulations, return as mod_ string
        if mod_dict:
            return f"mod_{json.dumps(mod_dict)}"
        
        # Default: return color if specified, otherwise None (no effect)
        # Don't default to blue - only use explicit color requests
        return color if color else None
    
    def _process_voice_command(self, command):
        """Process a voice command through Gemini VISION. Supports MULTIPLE targets and ENVIRONMENTAL commands."""
        print(f"\n{'='*50}")
        print(f"🎯 Processing: '{command}'")

        mask_labels = [m[1] for m in self.masks]
        mask_centers = [m[2] for m in self.masks]
        print(f"📋 Available labels: {mask_labels}")

        # Detect if this is an environmental command (not targeting specific objects)
        command_lower = command.lower()
        environmental_keywords = [
            "lighting", "lights", "bright", "dark", "glare", "harsh",
            "overstimulating", "overwhelming", "too much", "visual noise",
            "distracting", "colorful", "saturated", "contrast"
        ]

        is_environmental = any(keyword in command_lower for keyword in environmental_keywords)
        # Also check if no specific object is mentioned
        no_specific_object = not any(label.lower() in command_lower for label in mask_labels if not label.startswith("~"))

        if is_environmental and no_specific_object:
            print("🌍 Detected environmental command - using semantic understanding...")
            results = self.gemini.process_environmental_command(command, mask_labels, mask_centers)
        elif self.current_frame is not None:
            # Use VISION model - pass the current frame so Gemini can SEE what user is referring to
            print("👁️ Using Gemini Vision to analyze scene...")
            results = self.gemini.process_request_with_vision(
                command,
                self.current_frame,
                mask_labels,
                mask_centers
            )
        else:
            # Fallback to text-only
            results = self.gemini.process_request(
            command,
            self.gemini.scene_objects,
            mask_labels
        )
        
        # Handle both single result (dict) and multiple results (list)
        if isinstance(results, dict):
            results = [results]
        
        # ==========================================
        # POST-PROCESS: Fix BLUR vs BLUE confusion (AGGRESSIVE)
        # ==========================================
        command_lower = command.lower()
        user_wants_blur = "blur" in command_lower and "blue" not in command_lower
        
        if user_wants_blur:
            print("  🔧 Detected BLUR request - forcing blur mode on all results")
            for r in results:
                # Force blur mode regardless of what Gemini returned
                r["blur"] = True
                r["blur_strength"] = r.get("blur_strength") or 25
                r["color"] = None  # Remove any color
                # Set minimum confidence so effect gets applied
                if r.get("confidence", 0) < 0.5:
                    r["confidence"] = 0.5
        
        # ==========================================
        # POST-PROCESS: Extract explicit color from command
        # ==========================================
        explicit_colors = {
            "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
            "purple": "purple", "orange": "orange", "pink": "pink", "cyan": "cyan",
            "white": "white", "black": "black", "gray": "gray", "grey": "gray"
        }
        user_color = None
        for color_word, color_value in explicit_colors.items():
            # Check for exact color word (avoid "blur" matching "blue")
            if f" {color_word}" in f" {command_lower} " or command_lower.startswith(color_word):
                if color_word == "blue" and "blur" in command_lower:
                    continue  # Skip blue if blur is in command
                user_color = color_value
                break
        
        if user_color and not user_wants_blur:
            print(f"  🎨 Detected explicit color request: {user_color}")
            for r in results:
                r["color"] = user_color
                if r.get("confidence", 0) < 0.5:
                    r["confidence"] = 0.5
        
        # If Gemini failed or returned nothing, use direct text matching fallback
        if not results or len(results) == 0 or all(not r.get("target_label") for r in results):
            print("  🔄 Gemini Vision failed, using direct text matching...")
            results = self._direct_text_matching(command, mask_labels)
        
        applied_count = 0
        
        # Check if user wants "all" of something (e.g., "all ceiling lights")
        request_lower = command.lower()
        wants_all = "all" in request_lower or any(word in request_lower for word in ["every", "each"])
        
        for result in results:
            target = result.get("target_label")
            color = result.get("color")
            confidence = result.get("confidence", 0)
            
            # Handle modulation parameters (all modes)
            brightness = result.get("brightness")
            saturation = result.get("saturation")
            contrast = result.get("contrast")
            highlight_suppression = result.get("highlight_suppression")
            blur = result.get("blur")
            blur_strength = result.get("blur_strength")
            motion_dampen = result.get("motion_dampen")
            temporal_smooth = result.get("temporal_smooth")
            
            print(f"  🎯 Gemini picked: '{target}' (conf: {confidence:.0%})")
            if color:
                print(f"     🎨 Color: {color}")
            if blur:
                print(f"     🔵 BLUR MODE: strength={blur_strength or 25}")
            if brightness is not None:
                print(f"     💡 Brightness: {brightness}")
            if saturation is not None:
                print(f"     🎨 Saturation: {saturation}")
            if motion_dampen:
                print(f"     🎬 Motion Dampen: smooth={temporal_smooth or 0.7}")
            
            if target and confidence > 0.2:  # Lower threshold
                # If "all" requested, find ALL matching labels
                if wants_all:
                    matching_labels = []
                    target_base = target.lower().rstrip('0123456789').rstrip('_')
                    # Handle plurals
                    if target_base.endswith('s'):
                        target_singular = target_base[:-1]
                    else:
                        target_singular = target_base
                    
                    for label in mask_labels:
                        label_lower = label.lower()
                        label_base = label_lower.rstrip('0123456789').rstrip('_')
                        # Check if this label matches the target
                        if (target_base in label_lower or label_lower in target_base or
                            target_singular in label_lower or label_lower in target_singular):
                            matching_labels.append(label)
                    
                    if matching_labels:
                        print(f"  📋 Found {len(matching_labels)} matching labels: {matching_labels}")
                        for matched_label in matching_labels:
                            effect = self._build_effect(color, brightness, saturation, contrast, 
                                                       highlight_suppression, blur, blur_strength,
                                                       motion_dampen, temporal_smooth)
                            self.active_effects[matched_label] = effect
                            print(f"  ✅ Applied effect to '{matched_label}'")
                            applied_count += 1
                    else:
                        # Fall through to single match
                        matched_label = self._find_best_label_match(target, mask_labels)
                        if matched_label:
                            effect = self._build_effect(color, brightness, saturation, contrast,
                                                       highlight_suppression, blur, blur_strength,
                                                       motion_dampen, temporal_smooth)
                            self.active_effects[matched_label] = effect
                            print(f"  ✅ Applied effect to '{matched_label}'")
                            applied_count += 1
                else:
                    # Single match (existing logic)
                    matched_label = self._find_best_label_match(target, mask_labels)
                    if matched_label:
                        effect = self._build_effect(color, brightness, saturation, contrast,
                                                   highlight_suppression, blur, blur_strength,
                                                   motion_dampen, temporal_smooth)
                        self.active_effects[matched_label] = effect
                        print(f"  ✅ Applied effect to '{matched_label}'")
                        applied_count += 1
                    else:
                        # Store anyway for fuzzy matching during render
                        effect = self._build_effect(color, brightness, saturation, contrast,
                                                   highlight_suppression, blur, blur_strength,
                                                   motion_dampen, temporal_smooth)
                        self.active_effects[target] = effect
                        print(f"  ⚠️ No exact match, stored '{target}' -> effect")
                        applied_count += 1
        
        if applied_count == 0:
            print(f"❓ Couldn't determine target(s). Available: {mask_labels}")
            print(f"   Try: Press L to see available labels, then use exact names")
        else:
            print(f"🎨 Applied {applied_count} effect(s)")
    
    def _process_audio_command(self, command):
        """Process an audio command - completely separate from video pipeline."""
        print(f"\n{'='*50}")
        print(f"🎵 Processing AUDIO command: '{command}'")
        
        if not self.audio_processor.available:
            print("  ⚠️ Audio processor not available (missing dependencies)")
            return
        
        command_lower = command.lower()
        
        # Parse audio commands
        # Enable voice isolation (dampen surrounding audio)
        if any(phrase in command_lower for phrase in [
            "dampen", "damp", "reduce", "suppress", "isolate", "voice isolation",
            "noise suppression", "quiet", "mute background", "background noise"
        ]):
            self.audio_processor.set_voice_isolation(True)
            print("  ✅ Voice isolation ENABLED - surrounding audio dampened")
        
        # Disable voice isolation (full audio)
        elif any(phrase in command_lower for phrase in [
            "disable", "turn off", "stop", "full audio", "passthrough",
            "normal audio", "all audio"
        ]):
            self.audio_processor.set_voice_isolation(False)
            print("  ✅ Voice isolation DISABLED - full audio passthrough")
        
        # Toggle
        elif "toggle" in command_lower:
            new_state = not self.audio_processor.voice_isolation_enabled
            self.audio_processor.set_voice_isolation(new_state)
        
        else:
            print(f"  ❓ Unknown audio command. Try: 'dampen surrounding audio' or 'disable voice isolation'")
    
    def _direct_text_matching(self, command, available_labels):
        """Direct text matching fallback when Gemini fails - with ACCESSIBILITY MODES."""
        command_lower = command.lower()
        results = []
        
        # Pre-process command words for faster matching
        command_words = set(command_lower.split())
        
        # Check for "all" requests
        wants_all = "all" in command_lower or any(word in command_lower for word in ["every", "each"])
        
        # ==========================================
        # DETECT ACCESSIBILITY MODE FROM COMMAND
        # ==========================================
        # BLUR MODE triggers - Note: "blur" should NOT be confused with "blue"
        wants_blur = any(w in command_lower for w in ["blur", "blurry", "hide", "block", "private", "focus", "obscure"])
        blur_strength = 25
        if "strong" in command_lower or "heavy" in command_lower:
            blur_strength = 45
        elif "light" in command_lower or "slight" in command_lower:
            blur_strength = 15
        
        # DIM MODE triggers
        wants_dim = any(w in command_lower for w in ["dim", "darken", "bright", "glare", "harsh", "overstimulated"])
        brightness = 0.3 if wants_dim else None
        
        # MOTION MODE triggers
        wants_motion_dampen = any(w in command_lower for w in ["motion", "moving", "flickering", "shaking"])
        temporal_smooth = 0.8 if wants_motion_dampen else None
        
        # COLOR detection - IMPORTANT: "blur" should NOT match "blue"
        color = None
        # Check for exact color words, avoiding false matches like "blur" -> "blue"
        color_words = {
            "blue": [" blue", "blue "],  # Require space around "blue" to avoid "blur"
            "red": ["red"],
            "green": ["green"],
            "purple": ["purple"],
            "yellow": ["yellow"],
            "gray": ["gray", "grey"],
            "orange": ["orange"],
            "pink": ["pink"],
        }
        for c, patterns in color_words.items():
            # For "blue", check that it's not actually "blur"
            if c == "blue":
                # Only match "blue" if "blur" is NOT in command
                if "blur" not in command_lower and any(p in f" {command_lower} " for p in patterns):
                    color = c
                    break
            elif any(p in command_lower for p in patterns):
                color = c
                break
        
        # ==========================================
        # COMMON OBJECT PATTERNS
        # ==========================================
        # Screens/monitors (common distraction)
        if any(w in command_lower for w in ["screen", "monitor", "laptop", "computer", "tv", "display"]):
            screen_labels = [label for label in available_labels 
                           if any(w in label.lower() for w in ["screen", "monitor", "laptop", "computer", "tv", "display"])]
            for label in screen_labels:
                results.append({
                    "target_label": label,
                    "blur": wants_blur,
                    "blur_strength": blur_strength if wants_blur else None,
                    "brightness": brightness,
                    "confidence": 0.85
                })
        
        # People/faces (privacy mode) - includes "my face", "blur face", etc.
        face_triggers = ["people", "person", "face", "faces", "looking", "staring", "my face", "myself"]
        if any(w in command_lower for w in face_triggers):
            # Match person-related labels: face, person, man, woman, child, etc.
            people_labels = [label for label in available_labels 
                           if any(w in label.lower() for w in ["person", "face", "woman", "man", "people", "child", "boy", "girl"])]
            
            # If wants_blur is True (user said "blur"), apply blur
            # Otherwise, only blur if privacy keywords present
            should_blur = wants_blur or any(w in command_lower for w in ["hide", "private", "looking", "staring"])
            
            for label in people_labels:
                results.append({
                    "target_label": label,
                    "blur": should_blur,
                    "blur_strength": blur_strength if should_blur else None,
                    "color": color if not should_blur else None,  # Only apply color if NOT blurring
                    "confidence": 0.9
                })
        
        # Lights (brightness control)
        if any(w in command_lower for w in ["light", "lights", "lamp", "lamps", "bulb"]):
            light_labels = [label for label in available_labels 
                          if any(w in label.lower() for w in ["light", "lamp", "bulb", "overhead"])]
            for label in light_labels:
                results.append({
                    "target_label": label,
                    "brightness": 0.3,
                    "contrast": 0.6,
                    "confidence": 0.85
                })
        
        # Windows (sunlight control)
        if any(w in command_lower for w in ["window", "windows", "sunlight", "sun"]):
            window_labels = [label for label in available_labels if "window" in label.lower()]
            for label in window_labels:
                results.append({
                    "target_label": label,
                    "brightness": 0.25,
                    "contrast": 0.5,
                    "confidence": 0.85
                })
        
        # Ceiling
        if "ceiling" in command_lower:
            ceiling_labels = [label for label in available_labels if "ceiling" in label.lower()]
            for label in ceiling_labels:
                results.append({
                    "target_label": label,
                    "brightness": brightness,
                    "blur": wants_blur,
                    "blur_strength": blur_strength if wants_blur else None,
                    "confidence": 0.8
                })
        
        # Walls/background
        if any(w in command_lower for w in ["background", "wall", "walls"]):
            wall_labels = [label for label in available_labels 
                          if any(w in label.lower() for w in ["wall", "background", "room"])]
            for label in wall_labels:
                results.append({
                    "target_label": label,
                    "brightness": brightness,
                    "color": color,
                    "blur": wants_blur,
                    "blur_strength": blur_strength if wants_blur else None,
                    "confidence": 0.8
                })
        
        # ==========================================
        # GENERIC LABEL MATCHING (fallback)
        # ==========================================
        if not results:
            for label in available_labels:
                label_lower = label.lower()
                label_words = set(label_lower.replace('_', ' ').split())
                
                # Fast set intersection check
                if command_words & label_words or label_lower in command_lower or any(w in label_lower for w in command_words):
                    results.append({
                        "target_label": label,
                        "brightness": brightness,
                        "color": color,
                        "blur": wants_blur,
                        "blur_strength": blur_strength if wants_blur else None,
                        "motion_dampen": wants_motion_dampen,
                        "temporal_smooth": temporal_smooth,
                        "confidence": 0.7
                    })
                    if not wants_all:
                        break  # Only match first if not "all"
        
        return results
    
    def _run_feedback_loop(self, frame, detected_objects, yolo_detections):
        """Run the feedback loop to validate and correct detections."""
        # Note: The comprehensive_feedback_loop handles this automatically
        # This method is kept for compatibility but uses the comprehensive loop
        try:
            # Use comprehensive_feedback_loop instead (it handles everything)
            mask_labels = [obj.get("label", "unknown") for obj in detected_objects]
            mask_centers = []
            for obj in detected_objects:
                pos_str = obj.get("position", "unknown")
                # Try to parse position if it's in (x, y) format
                center = None
                if pos_str != "unknown" and "(" in pos_str:
                    try:
                        import re
                        match = re.search(r'\((\d+),\s*(\d+)\)', pos_str)
                        if match:
                            center = (float(match.group(1)), float(match.group(2)))
                    except:
                        pass
                mask_centers.append(center if center else (0, 0))
            
            # Call comprehensive feedback loop (handles all validation and correction internally)
            # This method applies corrections directly to gemini.label_corrections and updates thresholds
            self.gemini.comprehensive_feedback_loop(frame, mask_labels, yolo_detections, mask_centers)
            
            # Corrections are applied automatically by comprehensive_feedback_loop
            # The method updates gemini.label_corrections and optimal thresholds internally
        except Exception as e:
            print(f"  ⚠️ Feedback loop error: {e}")
    
    def _find_best_label_match(self, target, available_labels):
        """Find the best matching label from available labels - OPTIMIZED with caching."""
        if not target or not available_labels:
            return None
        
        target_lower = target.lower().strip()
        target_base = target_lower.rstrip('0123456789').rstrip('_')
        
        # Handle plurals: "lights" → "light", "windows" → "window"
        if target_lower.endswith('s') and len(target_lower) > 1:
            target_singular = target_lower[:-1]  # Remove 's'
        else:
            target_singular = target_lower
        
        # Quick exact match first (most common case)
        target_lower_set = {target_lower, target_base, target_singular}
        for label in available_labels:
            if label.lower() in target_lower_set:
                return label
        
        # 1. Direct exact match
        for label in available_labels:
            if label.lower() == target_lower:
                return label
        
        # 2. Singular/plural match ("lights" matches "light", "ceiling light")
        for label in available_labels:
            label_lower = label.lower()
            # Check if target (singular) matches label
            if target_singular in label_lower or label_lower in target_singular:
                return label
            # Check if label (singular) matches target
            label_singular = label_lower.rstrip('s') if label_lower.endswith('s') else label_lower
            if label_singular == target_singular or label_singular in target_singular:
                return label
        
        # 3. Base name match (face matches face, face_2, etc)
        for label in available_labels:
            label_base = label.lower().rstrip('0123456789').rstrip('_')
            if label_base == target_base or label_base == target_singular:
                return label
        
        # 4. Partial word match ("ceiling light" matches "ceiling light", "overhead light")
        target_words = set(target_singular.split())
        for label in available_labels:
            label_lower = label.lower()
            label_words = set(label_lower.replace('_', ' ').split())
            # If all target words are in label, it's a match
            if target_words and target_words.issubset(label_words):
                return label
            # If label words are in target, also match
            if label_words and label_words.issubset(set(target_singular.split())):
                return label
        
        # 5. Target is contained in label (face matches left_face)
        for label in available_labels:
            label_lower = label.lower()
            if target_base in label_lower or target_singular in label_lower:
                return label
        
        # 6. Label is contained in target (wall matches "the wall")
        for label in available_labels:
            label_base = label.lower().rstrip('0123456789').rstrip('_')
            label_singular = label_base.rstrip('s') if label_base.endswith('s') else label_base
            if label_base and (label_base in target_lower or label_singular in target_lower):
                return label
        
        return None
    
    def _match_effect(self, label):
        """STRICT match a label to active effects. Returns color or None."""
        label_lower = label.lower()
        
        # Direct exact match first
        if label in self.active_effects:
            return self.active_effects[label]
        
        # Check each active effect
        for effect_label, color in self.active_effects.items():
            effect_lower = effect_label.lower()
            
            # Exact match (case insensitive)
            if label_lower == effect_lower:
                return color
            
            # Strip trailing numbers for base match (wall matches wall_2)
            label_base = label_lower.rstrip('0123456789').rstrip('_')
            effect_base = effect_lower.rstrip('0123456789').rstrip('_')
            
            if label_base == effect_base and label_base:
                return color
            
            # STRICT synonym groups - only within same category
            synonym_groups = [
                {"wall", "background"},  # Structural - walls only
                {"floor", "ground"},     # Structural - floor only
                {"ceiling"},             # Structural - ceiling only
                {"person", "body"},      # Full body only
                {"face", "head"},        # Face only - NOT person
                {"left_hand"},           # Left hand only
                {"right_hand"},          # Right hand only
                {"hands"},               # Both hands
                {"left_arm"},
                {"right_arm"},
                {"torso", "chest"},
                {"left_leg"},
                {"right_leg"},
            ]
            
            for group in synonym_groups:
                effect_in_group = any(s == effect_base or s in effect_lower for s in group)
                label_in_group = any(s == label_base or s in label_lower for s in group)
                if effect_in_group and label_in_group:
                    return color
        
        return None
    
    def clear_effects(self):
        """Clear all active effects."""
        self.active_effects = {}
        print("🧹 Cleared all effects")
    
    def _refine_mask_edges(self, frame, mask, iterations=2):
        """Refine mask edges using bilateral filtering and morphological operations for tight, quality edges."""
        if mask is None or frame is None:
            return mask

        try:
            # Convert mask to uint8
            mask_u8 = (mask * 255).astype(np.uint8)

            # Apply bilateral filter to smooth while preserving edges
            # This helps create clean boundaries
            mask_filtered = cv2.bilateralFilter(mask_u8, 5, 50, 50)

            # Apply morphological opening to remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_opened = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel)

            # Apply morphological closing to fill small holes
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

            # Edge-aware refinement using the original frame
            # Create a fine mask using GrabCut-style refinement
            mask_refined = self._grabcut_refinement(frame, mask_closed)

            return mask_refined.astype(np.float32) / 255.0

        except Exception as e:
            # If refinement fails, return original mask
            return mask

    def _grabcut_refinement(self, frame, mask_u8):
        """Use GrabCut algorithm for precise mask refinement with tight edges."""
        try:
            h, w = mask_u8.shape

            # Create initial mask for GrabCut
            # GrabCut expects: 0=bg, 1=fg, 2=probable_bg, 3=probable_fg
            grabcut_mask = np.zeros(mask_u8.shape, dtype=np.uint8)

            # Core region (high confidence foreground)
            core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            core_mask = cv2.erode(mask_u8, core_kernel, iterations=2)
            grabcut_mask[core_mask > 128] = 1  # Definite foreground

            # Edge region (probable foreground)
            edge_mask = mask_u8 - core_mask
            grabcut_mask[edge_mask > 64] = 3  # Probable foreground

            # Background region
            bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated_mask = cv2.dilate(mask_u8, bg_kernel, iterations=2)
            grabcut_mask[dilated_mask == 0] = 0  # Definite background

            # Check if we have enough foreground pixels
            if np.sum(grabcut_mask == 1) < 100:
                return mask_u8  # Not enough data, return original

            # Apply GrabCut (just 1 iteration for speed)
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)

            try:
                cv2.grabCut(frame, grabcut_mask, None, bgd_model, fgd_model,
                           1, cv2.GC_INIT_WITH_MASK)

                # Extract foreground (1 and 3 are foreground)
                refined_mask = np.where((grabcut_mask == 1) | (grabcut_mask == 3), 255, 0).astype(np.uint8)

                return refined_mask

            except:
                # GrabCut failed, return original
                return mask_u8

        except Exception as e:
            return mask_u8

    def _detect_adjacent_boundaries(self, masks_with_labels):
        """Detect boundaries between adjacent objects for multi-color tracing."""
        if len(masks_with_labels) < 2:
            return []

        boundaries = []

        # Compare each pair of masks
        for i in range(len(masks_with_labels)):
            mask_i, label_i, _ = masks_with_labels[i]
            if mask_i is None:
                continue

            mask_i_u8 = (mask_i * 255).astype(np.uint8)

            for j in range(i + 1, len(masks_with_labels)):
                mask_j, label_j, _ = masks_with_labels[j]
                if mask_j is None:
                    continue

                mask_j_u8 = (mask_j * 255).astype(np.uint8)

                # Dilate both masks slightly to find where they would overlap
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                dilated_i = cv2.dilate(mask_i_u8, kernel, iterations=1)
                dilated_j = cv2.dilate(mask_j_u8, kernel, iterations=1)

                # Find intersection of dilated masks (adjacent boundary)
                intersection = cv2.bitwise_and(dilated_i, dilated_j)

                # If there's significant overlap, they're adjacent
                if np.sum(intersection > 128) > 50:  # At least 50 pixels of boundary
                    boundaries.append({
                        'mask_i': i,
                        'mask_j': j,
                        'label_i': label_i,
                        'label_j': label_j,
                        'boundary': intersection
                    })

        return boundaries

    def _mask_center(self, mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0)
        return (int(np.mean(xs)), int(np.mean(ys)))
    
    def _mask_bbox(self, mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    
    def _update_tracked_masks(self, new_masks, h, w):
        """Update mask tracking - persist masks during movement/brief detection gaps."""
        current_time = time.time()
        
        # Mark all tracks as not seen this frame
        for track_id in self.tracked_masks:
            self.tracked_masks[track_id]['seen_this_frame'] = False
        
        # Match new masks to existing tracks
        for mask, label, center in new_masks:
            if mask is None or center is None or center[0] <= 0:
                continue
            
            # Find best matching track by position and label
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track in self.tracked_masks.items():
                if track['seen_this_frame']:
                    continue  # Already matched
                
                # Calculate distance to track's predicted position
                track_center = track['center']
                if track['velocity']:
                    # Predict where the mask should be based on velocity
                    dt = current_time - track['last_seen']
                    pred_x = track_center[0] + track['velocity'][0] * dt
                    pred_y = track_center[1] + track['velocity'][1] * dt
                    track_center = (pred_x, pred_y)
                
                dist = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)
                
                # Bonus for matching label
                if track['label'] == label or (track['label'] and label and 
                    track['label'].replace("~", "") == label.replace("~", "")):
                    dist *= 0.5  # Prefer same label
                
                if dist < best_distance and dist < 100:  # Max 100px to match
                    best_distance = dist
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track = self.tracked_masks[best_track_id]
                
                # Calculate velocity
                dt = current_time - track['last_seen']
                if dt > 0 and dt < 1.0:  # Only update velocity if reasonable time gap
                    new_vx = (center[0] - track['center'][0]) / dt
                    new_vy = (center[1] - track['center'][1]) / dt
                    if track['velocity']:
                        # Smooth velocity
                        track['velocity'] = (
                            track['velocity'][0] * (1 - self.velocity_smoothing) + new_vx * self.velocity_smoothing,
                            track['velocity'][1] * (1 - self.velocity_smoothing) + new_vy * self.velocity_smoothing
                        )
                    else:
                        track['velocity'] = (new_vx, new_vy)
                
                track['mask'] = mask
                track['label'] = label
                track['center'] = center
                track['last_seen'] = current_time
                track['frames_missing'] = 0
                track['seen_this_frame'] = True
            else:
                # Create new track
                self.tracked_masks[self.next_track_id] = {
                    'mask': mask,
                    'label': label,
                    'center': center,
                    'velocity': None,
                    'last_seen': current_time,
                    'frames_missing': 0,
                    'seen_this_frame': True
                }
                self.next_track_id += 1
        
        # Update missing tracks (interpolate position) or remove stale ones
        tracks_to_remove = []
        for track_id, track in self.tracked_masks.items():
            if not track['seen_this_frame']:
                track['frames_missing'] += 1
                
                # Interpolate position using velocity
                if track['velocity'] and track['frames_missing'] < self.max_frames_missing:
                    dt = 1/30.0  # Assume 30fps
                    new_cx = track['center'][0] + track['velocity'][0] * dt
                    new_cy = track['center'][1] + track['velocity'][1] * dt
                    
                    # Keep within bounds
                    new_cx = max(0, min(w, new_cx))
                    new_cy = max(0, min(h, new_cy))
                    track['center'] = (new_cx, new_cy)
                    
                    # Shift mask if possible (simple translation)
                    if track['mask'] is not None:
                        shift_x = int(track['velocity'][0] * dt)
                        shift_y = int(track['velocity'][1] * dt)
                        if abs(shift_x) < 50 and abs(shift_y) < 50:
                            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                            track['mask'] = cv2.warpAffine(track['mask'].astype(np.float32), M, (w, h))
                
                # Remove if too many frames missing
                if track['frames_missing'] > self.max_frames_missing:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_masks[track_id]
    
    def _get_tracked_masks_for_display(self, h, w):
        """Get all tracked masks for display (including interpolated ones)."""
        result = []
        for track_id, track in self.tracked_masks.items():
            mask = track['mask']
            label = track['label']
            center = track['center']
            
            if mask is not None and mask.shape == (h, w):
                result.append((mask, label, center))
        
        return result
    
    def _calc_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter + 1e-6)
    
    def close(self):
        self.voice.stop()
        if self.audio_processor:
            self.audio_processor.close()
        self.selfie.close()
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()


class AsyncCamera:
    """Async camera capture - smooth FPS independent of processing."""
    
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
        
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        
        # Start capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """Continuously capture frames in background."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame
                    self.frame_count += 1
    
    def get_frame(self):
        """Get the latest frame (non-blocking)."""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def is_opened(self):
        return self.cap.isOpened()
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def main():
    controller = EnvironmentController()
    
    # Use async camera for smooth streaming
    camera = AsyncCamera(0, 640, 480, 30)
    
    if not camera.is_opened():
        print("❌ No camera!")
        return
    
    # Wait for first frame
    time.sleep(0.5)
    
    fps_times = []
    display_fps_times = []  # Track actual display FPS
    last_display_time = time.time()
    
    try:
        while True:
            # Get latest frame from async camera (always fresh, never blocks)
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Track display FPS (actual smooth frame rate)
            now = time.time()
            display_fps_times.append(now - last_display_time)
            last_display_time = now
            if len(display_fps_times) > 30:
                display_fps_times.pop(0)
            display_fps = 1.0 / (sum(display_fps_times) / len(display_fps_times)) if display_fps_times else 30
            
            # Process frame (may be slower, but camera keeps streaming)
            t0 = time.time()
            display = controller.process_frame(frame)
            elapsed = time.time() - t0
            
            # Processing FPS
            fps_times.append(elapsed)
            if len(fps_times) > 20:
                fps_times.pop(0)
            process_fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 30
            
            # UI overlay
            h, w = display.shape[:2]
            
            # Only show full UI if NOT in clean view mode
            if not controller.clean_view_mode:
                # Status bar
                cv2.rectangle(display, (0, 0), (w, 65), (30, 30, 30), -1)
                cv2.putText(display, f"Display: {display_fps:.0f} FPS | Process: {process_fps:.0f} | Objects: {len(controller.masks)}", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(display, controller.voice.status, 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if controller.voice.last_command:
                    cv2.putText(display, f"Last: {controller.voice.last_command[:40]}", 
                               (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Active effects panel
                y = 75
                cv2.putText(display, f"Labels: {len(controller.masks)} | Effects: {len(controller.active_effects)}", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                if controller.active_effects:
                    y += 18
                    cv2.putText(display, "ACTIVE EFFECTS:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    for label, color in list(controller.active_effects.items())[:6]:  # Show max 6
                        y += 14
                        cv2.putText(display, f"  {label} -> {color}", (10, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
                
                # Help text at bottom
                help_y = h - 25
                cv2.rectangle(display, (0, help_y - 5), (w, h), (30, 30, 30), -1)
                cv2.putText(display, "Say 'hey vibe' to start | V=clean view | C=clear | L=labels | S=save | Q=quit", 
                           (10, help_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            else:
                # Clean view: minimal UI
                cv2.putText(display, f"[CLEAN VIEW] V=toggle | {len(controller.active_effects)} effects", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            cv2.imshow("Voice Environment Controller", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):  # V - toggle clean view
                controller.clean_view_mode = not controller.clean_view_mode
                mode_str = "CLEAN (effects only)" if controller.clean_view_mode else "FULL (labels + borders)"
                print(f"  👁️ View mode: {mode_str}")
            elif key == ord('c'):
                controller.clear_effects()
            elif key == ord('p'):
                controller.autopilot_enabled = not controller.autopilot_enabled
                print(f"  🤖 Autopilot = {'ON' if controller.autopilot_enabled else 'OFF'}")
            elif key == ord('s'):
                fn = f"voice_env_{int(time.time())}.png"
                cv2.imwrite(fn, display)
                print(f"💾 Saved {fn}")
            elif key == ord('l'):  # L - list all labels
                labels = [m[1] for m in controller.masks]
                print(f"\n{'='*50}")
                print("📋 ALL AVAILABLE LABELS:")
                for i, label in enumerate(labels):
                    effect = controller.active_effects.get(label, None)
                    status = f" -> {effect}" if effect else ""
                    print(f"  {i+1}. {label}{status}")
                print(f"{'='*50}\n")
            # Voice recording now handled by wake words ("hey vibe" / "thanks")
            # No keyboard input needed

            # Auto-stop when the online training loop reaches target confidence
            if controller.gemini.should_stop():
                print(f"🏁 Stopping (auto-train complete): {controller.gemini.stop_reason()}")
                break
                
    finally:
        controller.close()
        camera.release()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()

