"""Gemini API integration for vision and text processing."""
import os
import time
import json
import numpy as np
import cv2
from PIL import Image
from collections import deque

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiAgent:
    """Handles Gemini text and vision processing with self-optimization."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.scene_objects = []
        self.last_scene_analysis = None
        self.scene_context = {}

        # Rate limiting (conservative for free tier)
        self.vision_call_count = 0
        self.last_vision_call = 0
        self.min_vision_interval = 6.0
        self.max_vision_calls_per_minute = 5
        self.vision_calls_this_minute = 0
        self.minute_start = time.time()

        # Feedback loop system
        self.detection_feedback = {}
        self.label_corrections = {}
        self.confidence_adjustments = {}
        self.optimization_history = []
        self.last_feedback_update = 0
        self.feedback_interval = 10.0
        self.detection_accuracy = {}
        self.missing_objects = []
        self.false_positives = []
        self.label_confidence_scores = {}

        # Online training loop
        self.auto_train_enabled = os.environ.get("AUTO_TRAIN", "1") == "1"
        self.auto_train_target_acc = float(os.environ.get("AUTO_TRAIN_TARGET_ACC", "0.90"))
        self.auto_train_required_consecutive = int(os.environ.get("AUTO_TRAIN_CONSEC", "30"))
        self.auto_train_min_coverage = float(os.environ.get("AUTO_TRAIN_MIN_COVERAGE", "0.70"))
        self.auto_train_min_label_acc = float(os.environ.get("AUTO_TRAIN_MIN_LABEL_ACC", "0.75"))
        self._acc_history = deque(maxlen=max(50, self.auto_train_required_consecutive))
        self._train_complete = False
        self._train_complete_reason = ""

        # Self-optimization parameters
        self.optimal_yolo_threshold = 0.15
        self.optimal_sam_conf = 0.3
        self.optimal_iou_threshold = 0.45
        self.optimization_iterations = 0

        if not self.api_key:
            print("⚠️  No GEMINI_API_KEY or GOOGLE_API_KEY found.")
            self.available = False
            return

        if not GEMINI_AVAILABLE:
            self.available = False
            return

        genai.configure(api_key=self.api_key)

        # Initialize models
        try:
            self.text_model = genai.GenerativeModel('gemini-pro')
            self.vision_model = genai.GenerativeModel('gemini-pro')
        except Exception as e1:
            try:
                self.text_model = genai.GenerativeModel('gemini-1.5-pro')
                self.vision_model = genai.GenerativeModel('gemini-1.5-pro')
            except Exception as e2:
                try:
                    self.text_model = genai.GenerativeModel('gemini-1.5-flash')
                    self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
                except Exception as e3:
                    print(f"⚠️ Could not initialize any Gemini model: {e1}, {e2}, {e3}")
                    raise

        self.available = True
        print("✅ Gemini initialized")

    def _can_call_vision(self):
        """Check if we're within rate limits for vision API calls."""
        current_time = time.time()
        if current_time - self.minute_start > 60:
            self.vision_calls_this_minute = 0
            self.minute_start = current_time
        if self.vision_calls_this_minute >= self.max_vision_calls_per_minute:
            return False
        if current_time - self.last_vision_call < self.min_vision_interval:
            return False
        return True

    def _record_vision_call(self):
        """Record a vision API call for rate limiting."""
        self.vision_call_count += 1
        self.vision_calls_this_minute += 1
        self.last_vision_call = time.time()

    def label_all_segments(self, frame, masks_with_centers):
        """Use Gemini Vision to label ALL SAM segments with enhanced open vocabulary detection."""
        if not self.available or not self._can_call_vision():
            return None

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((640, 480))

            positions = []
            for idx, (mask, existing_label, center) in enumerate(masks_with_centers):
                if center and center[0] > 0:
                    cx, cy = center
                    h = "left-side" if cx < 200 else "right-side" if cx > 400 else "center"
                    v = "upper" if cy < 150 else "lower" if cy > 300 else "middle"
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
LIGHTING: "ceiling_light", "table_lamp", "floor_lamp", "led_strip", "desk_light", "window", "skylight", "light_fixture"
SCREENS: "laptop_screen", "desktop_monitor", "tv_screen", "tablet", "phone_screen", "projector_screen", "smart_display"
PERSON/BODY: "face", "left_hand", "right_hand", "left_arm", "right_arm", "torso", "head", "person", "body"
FURNITURE: "desk", "chair", "table", "shelf", "cabinet", "bed", "couch", "bookshelf", "drawer", "wardrobe"
STRUCTURAL: "wall", "ceiling", "floor", "door", "window_frame", "column", "corner"
OBJECTS: "plant", "clock", "picture_frame", "poster", "book", "keyboard", "mouse", "speaker", "headphones", "cup", "bottle", "bag"

Return ONLY this JSON format with SPECIFIC labels:
[{{"region":1,"label":"ceiling_light","asset_class":"lighting"}},
 {{"region":2,"label":"laptop_screen","asset_class":"screen"}},
 {{"region":3,"label":"desk","asset_class":"furniture"}}]"""

            self._record_vision_call()
            response = self.vision_model.generate_content([prompt, pil_image])
            text = response.text.strip()

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            if "[" in text and "]" in text:
                start = text.find("[")
                end = text.rfind("]") + 1
                text = text[start:end]

            results = json.loads(text)
            labels = {}
            label_list = []
            asset_classes = {}

            for item in results:
                region = item.get("region", 0) - 1
                label = item.get("label", "unknown").lower().strip()
                asset_class = item.get("asset_class", "object").lower().strip()

                if 0 <= region < len(masks_with_centers) and label:
                    labels[region] = label
                    asset_classes[region] = asset_class
                    label_list.append(f"{label}[{asset_class}]")

            if not hasattr(self, 'asset_class_map'):
                self.asset_class_map = {}
            self.asset_class_map.update({labels[i]: asset_classes.get(i, "object") for i in labels.keys()})

            if label_list:
                print(f"  🏷️ Gemini Vision ID: {', '.join(label_list[:6])}{'...' if len(label_list) > 6 else ''}")
            return labels

        except json.JSONDecodeError:
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

            if existing_label and existing_label in ["face", "left_hand", "right_hand", "person",
                                                       "left_arm", "right_arm", "torso", "left_leg", "right_leg"]:
                labels[idx] = existing_label
                continue

            try:
                mask_bool = mask > 0.5
                if np.any(mask_bool):
                    region = frame[mask_bool]
                    brightness = np.mean(region)
                else:
                    brightness = 128
            except:
                brightness = 128

            if mask_area > 0.15:
                if cy < h * 0.35:
                    labels[idx] = "ceiling"
                elif cy > h * 0.7:
                    labels[idx] = "floor"
                else:
                    labels[idx] = "wall"
            elif brightness > 180 and cy < h * 0.5 and mask_area < 0.05:
                labels[idx] = "light"
            elif cy < h * 0.3 and mask_area > 0.03:
                labels[idx] = "ceiling"
            elif cy > h * 0.5 and 0.02 < mask_area < 0.15:
                labels[idx] = "furniture"
            elif brightness > 150 and 0.01 < mask_area < 0.08:
                labels[idx] = "screen"
            elif cx < w * 0.15 or cx > w * 0.85:
                if mask_area > 0.05:
                    labels[idx] = "wall"
                else:
                    labels[idx] = "object"
            elif mask_area > 0.05:
                labels[idx] = "surface"
            else:
                labels[idx] = "object"

        if labels:
            print(f"  🔄 Fallback labels: {', '.join(list(labels.values())[:8])}")
        return labels

    def analyze_scene_continuous(self, frame, mask_labels, mask_centers):
        """Continuous scene analysis - runs every second for context."""
        if not self.available:
            return

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((640, 480))

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
            self.scene_context = scene_context
            print(f"📊 Scene context: {scene_context.get('sensory_triggers', [])}")

        except Exception as e:
            error_str = str(e)
            if "404" in error_str and "models/" in error_str:
                if not hasattr(self, '_last_model_error_log'):
                    self._last_model_error_log = 0
                if time.time() - self._last_model_error_log > 60:
                    print(f"  ℹ️ Gemini Vision API model compatibility issue (non-fatal - using fallbacks)")
                    self._last_model_error_log = time.time()
            else:
                print(f"Continuous analysis error: {e}")

    def comprehensive_feedback_loop(self, frame, mask_labels, yolo_detections, mask_centers):
        """Comprehensive feedback loop: Gemini analyzes scene and self-corrects everything."""
        if not self.available:
            return

        current_time = time.time()
        if current_time - self.last_feedback_update < self.feedback_interval:
            return

        self.last_feedback_update = current_time

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((640, 480))

            detected_info = []
            for i, (label, center) in enumerate(zip(mask_labels, mask_centers)):
                if center:
                    cx, cy = center
                    detected_info.append({"label": label, "position": f"({cx}, {cy})", "index": i})

            yolo_info = []
            for (x1, y1, x2, y2), label, conf in yolo_detections:
                yolo_info.append({"label": label, "bbox": f"({x1},{y1})-({x2},{y2})", "confidence": conf})

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
   - Are our labels correct?
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
  "actual_objects": [{{"name": "wall", "position": "background", "confidence": 0.95}}],
  "label_corrections": [{{"current_label": "item_1", "correct_label": "desk", "confidence": 0.9}}],
  "missing_objects": [{{"name": "window", "position": "left side", "should_detect": true}}],
  "false_positives": [{{"label": "person", "reason": "This is actually a wall", "confidence": 0.85}}],
  "optimization_suggestions": {{"yolo_threshold": 0.15, "sam_confidence": 0.3, "iou_threshold": 0.45, "notes": "Lower YOLO threshold"}},
  "detection_quality": {{"overall_accuracy": 0.75, "label_accuracy": 0.8, "coverage": 0.7}}
}}"""

            response = self.vision_model.generate_content([prompt, pil_image])
            text = response.text.strip()

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            feedback = json.loads(text)

            # Process missing objects and map to generic masks
            try:
                missing = feedback.get("missing_objects", []) or []
                if missing and mask_centers and mask_labels:
                    h, w = frame.shape[:2]
                    generic_labels = {"item", "item_wide", "item_tall", "small_item", "furniture", "area"}

                    def _pos_score(pos_text, cx, cy):
                        t = (pos_text or "").lower()
                        score = 0.0
                        if "left" in t:
                            score += 1.0 if cx < w * 0.40 else 0.0
                        if "right" in t:
                            score += 1.0 if cx > w * 0.60 else 0.0
                        if "center" in t or "middle" in t:
                            score += 1.0 if (w * 0.40 <= cx <= w * 0.60) else 0.0
                        if "top" in t or "upper" in t:
                            score += 1.0 if cy < h * 0.40 else 0.0
                        if "bottom" in t or "lower" in t:
                            score += 1.0 if cy > h * 0.60 else 0.0
                        if "mid" in t or "middle" in t:
                            score += 1.0 if (h * 0.40 <= cy <= h * 0.60) else 0.0
                        if "background" in t:
                            score += 0.5
                        return score

                    candidates = []
                    for idx, (lab, cen) in enumerate(zip(mask_labels, mask_centers)):
                        if not cen:
                            continue
                        base = (lab or "").split("_")[0]
                        if base in generic_labels:
                            candidates.append((lab, cen, idx))

                    for m in missing[:8]:
                        name = (m.get("name") or "").strip().lower()
                        pos = m.get("position") or m.get("location") or ""
                        should = m.get("should_detect", True)
                        if not name or not should:
                            continue
                        best = None
                        best_s = -1.0
                        for lab, cen, idx in candidates:
                            cx, cy = float(cen[0]), float(cen[1])
                            s = _pos_score(pos, cx, cy)
                            if s > best_s:
                                best_s = s
                                best = (lab, idx)
                        if best and best_s >= 1.0:
                            feedback.setdefault("label_corrections", [])
                            feedback["label_corrections"].append({
                                "current_label": best[0],
                                "correct_label": name,
                                "confidence": 0.80
                            })
            except Exception as e:
                print(f"  ⚠️ Missing-object mapping error: {e}")

            self._apply_feedback_corrections(feedback, mask_labels, yolo_detections)
            self._update_optimization_parameters(feedback)
            self.optimization_iterations += 1

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
            if "404" in error_str and "models/" in error_str:
                if not hasattr(self, '_last_comprehensive_error_log'):
                    self._last_comprehensive_error_log = 0
                if time.time() - self._last_comprehensive_error_log > 60:
                    print(f"  ℹ️ Comprehensive feedback: Gemini API model compatibility (non-fatal)")
                    self._last_comprehensive_error_log = time.time()
            else:
                print(f"  ⚠️ Feedback loop error: {e}")

    def _apply_feedback_corrections(self, feedback, mask_labels, yolo_detections):
        """Apply label corrections and confidence adjustments from feedback."""
        corrections = feedback.get('label_corrections', [])
        correction_count = 0
        for correction in corrections:
            current = correction.get('current_label')
            correct = correction.get('correct_label')
            conf = correction.get('confidence', 0.8)

            if current and correct and conf > 0.7:
                if current not in self.label_corrections:
                    self.label_corrections[current] = {'correct_label': correct, 'confidence': conf, 'applied_count': 0}
                self.label_corrections[current]['applied_count'] += 1
                correction_count += 1
                print(f"  ✅ Gemini Vision corrected YOLO label: '{current}' → '{correct}' (conf: {conf:.0%})")

        missing = feedback.get('missing_objects', [])
        if missing:
            missing_names = [m.get('name') for m in missing if m.get('should_detect', True)]
            if missing_names:
                print(f"  📋 Gemini Vision found {len(missing_names)} missing objects: {', '.join(missing_names[:5])}")
                print(f"     → These should be detected but aren't in current YOLO/SAM detections")
            self.missing_objects = missing[:10]

        false_pos = feedback.get('false_positives', [])
        if false_pos:
            fp_labels = [fp.get('label') for fp in false_pos]
            print(f"  ⚠️ Gemini Vision found {len(false_pos)} false positives: {', '.join(fp_labels[:5])}")
            print(f"     → These are incorrectly detected by YOLO/SAM")
            self.false_positives = false_pos[:10]

        actual_objects = feedback.get('actual_objects', [])
        if actual_objects and len(actual_objects) > 0:
            seen_names = [obj.get('name', 'unknown') for obj in actual_objects[:8]]
            print(f"  👁️ Gemini Vision sees in scene: {', '.join(seen_names)}")

        if correction_count > 0:
            print(f"  🎯 Applied {correction_count} YOLO label correction(s) from Gemini Vision")

    def _update_optimization_parameters(self, feedback):
        """Update detection parameters based on feedback."""
        suggestions = feedback.get('optimization_suggestions', {})

        if 'yolo_threshold' in suggestions:
            new_threshold = suggestions['yolo_threshold']
            if 0.1 <= new_threshold <= 0.5:
                self.optimal_yolo_threshold = new_threshold

        if 'sam_confidence' in suggestions:
            new_conf = suggestions['sam_confidence']
            if 0.2 <= new_conf <= 0.5:
                self.optimal_sam_conf = new_conf

        if 'iou_threshold' in suggestions:
            new_iou = suggestions['iou_threshold']
            if 0.3 <= new_iou <= 0.6:
                self.optimal_iou_threshold = new_iou

        self.optimization_history.append({
            'iteration': self.optimization_iterations,
            'parameters': {
                'yolo_threshold': self.optimal_yolo_threshold,
                'sam_confidence': self.optimal_sam_conf,
                'iou_threshold': self.optimal_iou_threshold
            },
            'quality': feedback.get('detection_quality', {})
        })

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

        if len(self.optimization_history) > 50:
            self.optimization_history = self.optimization_history[-50:]

    def get_corrected_label(self, original_label):
        """Get corrected label if available from Gemini Vision feedback."""
        if original_label in self.label_corrections:
            correction = self.label_corrections[original_label]
            if correction['confidence'] > 0.7 and correction['applied_count'] >= 1:
                return correction['correct_label']
        return original_label

    def get_missing_objects(self):
        """Get list of objects Gemini Vision detected that we're missing."""
        return self.missing_objects

    def should_stop(self):
        """Whether the online training loop reached its target and should stop the app."""
        return bool(self.auto_train_enabled and self._train_complete)

    def stop_reason(self):
        return self._train_complete_reason or ""

    def get_optimal_thresholds(self):
        """Get optimized detection thresholds."""
        return {
            'yolo_threshold': self.optimal_yolo_threshold,
            'sam_confidence': self.optimal_sam_conf,
            'iou_threshold': self.optimal_iou_threshold
        }

    def process_environmental_command(self, command, mask_labels, mask_centers):
        """Process dynamic environmental commands using LLM semantic understanding."""
        if not self.available:
            return []

        try:
            asset_map = getattr(self, 'asset_class_map', {})
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
1. "lighting is extremely bright" / "lights hurt" / "too much glare"
   → Affect ALL: lighting sources (ceiling_light, lamp, window, etc.)
   → Affect ALL: screens (monitor, laptop_screen, tv_screen)
   → Action: brightness=0.2-0.4

2. "too dark" / "can't see"
   → Affect ALL: lighting sources
   → Action: brightness=1.5-2.0

3. "too much visual noise" / "overstimulating" / "distracting"
   → Affect: screens, moving objects, bright objects
   → Action: blur=true, blur_strength=25-35

RETURN FORMAT - JSON array of affected objects:
[
  {{"target_label": "ceiling_light", "brightness": 0.3, "reason": "lighting source"}},
  {{"target_label": "window", "brightness": 0.3, "reason": "natural light source"}},
  {{"target_label": "laptop_screen", "brightness": 0.3, "reason": "artificial light source"}}
]

CRITICAL RULES:
1. Match objects by ASSET CLASS when environmental condition mentioned
2. Return SPECIFIC labels from the available objects list
3. Include ALL relevant objects
4. Choose appropriate effects (brightness, blur, saturation, contrast)"""

            response = self.text_model.generate_content(prompt)
            text = response.text.strip()

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
        """Use Gemini VISION to SEE the frame and map user request to correct objects."""
        if not self.available:
            return self._fallback_process(user_request, mask_labels)

        if not self._can_call_vision():
            print("⏳ Vision API rate limited, using text-only processing")
            return self._text_only_process(user_request, mask_labels)

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((512, 384))

            labels_with_pos = []
            for i, label in enumerate(mask_labels):
                if i < len(mask_centers) and mask_centers[i]:
                    cx, cy = mask_centers[i]
                    h_pos = "left" if cx < 213 else "right" if cx > 426 else "center"
                    v_pos = "top" if cy < 160 else "bottom" if cy > 320 else "middle"
                    labels_with_pos.append(f"  - '{label}' (at {h_pos}-{v_pos})")
                else:
                    labels_with_pos.append(f"  - '{label}'")

            labels_str = "\n".join(labels_with_pos)

            prompt = f"""You are an ACCESSIBILITY ASSISTANT for sensory regulation. Analyze the scene and the user's request.

USER REQUEST: "{user_request}"

AVAILABLE OBJECTS (YOLO-detected, use EXACT names):
{labels_str}

⚠️ CRITICAL: "BLUR" ≠ "BLUE"!
- "blur my face" → apply BLUR effect (blur=true), NOT color blue!
- "make it blue" → apply COLOR blue

VISUAL MODES TO APPLY:

1. BLUR MODE (Visual Noise Cancellation):
   Triggers: "blur", "blurry", "hide", "block", "private", "obscure"
   Action: blur=true, blur_strength=15-35

2. DIM/BRIGHTNESS MODE:
   Triggers: "too bright", "lights hurt", "glare", "overstimulated"
   Action: brightness=0.2-0.5, contrast=0.5-0.8

3. COLOR MODE (Color Remapping):
   Triggers: "make X blue", "turn Y green", "change to red"
   Action: color="blue"/"green"/"purple" etc.
   ⚠️ DO NOT use color mode when user says "blur"!

4. MOTION DAMPENING:
   Triggers: "movement", "motion", "too fast", "flickering"
   Action: motion_dampen=true, temporal_smooth=0.7-0.9

RETURN FORMAT (JSON array):
[{{"target_label": "EXACT_LABEL", "blur": true, "blur_strength": 25}}]"""

            self._record_vision_call()
            response = self.vision_model.generate_content([prompt, pil_image])
            text = response.text.strip()

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            results = json.loads(text)
            if isinstance(results, dict):
                results = [results]

            if results:
                targets = [r.get('target_label') for r in results if r.get('target_label')]
                print(f"👁️ Gemini Vision: Sees {len(targets)} target(s): {targets}")

            return results

        except Exception as e:
            error_str = str(e)
            if "404" in error_str and "models/" in error_str:
                if not hasattr(self, '_last_vision_error_log'):
                    self._last_vision_error_log = 0
                if time.time() - self._last_vision_error_log > 60:
                    print(f"  ℹ️ Vision processing: Gemini API model compatibility (using fallback)")
                    self._last_vision_error_log = time.time()
            else:
                print(f"Vision processing error: {e}")
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

    def _fallback_process(self, request, mask_labels):
        """Simple fallback when Gemini is unavailable."""
        request_lower = request.lower()
        results = []

        color_words = {
            "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
            "purple": "purple", "orange": "orange", "pink": "pink", "cyan": "cyan",
            "white": "white", "black": "black", "dim": "dark_gray", "darken": "dark_gray",
            "dark": "dark_gray", "highlight": "yellow", "bright": "yellow", "hide": "black"
        }

        synonyms = {
            "wall": ["wall", "background", "behind"],
            "ceiling": ["ceiling", "top", "above"],
            "floor": ["floor", "ground", "bottom"],
            "person": ["me", "myself", "person", "body", "face"],
        }

        parts = request_lower.replace(",", " and ").split(" and ")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            detected_color = "blue"
            for color_word, color_value in color_words.items():
                if color_word in part:
                    detected_color = color_value
                    break

            target = None
            for label in mask_labels:
                if label.lower() in part:
                    target = label
                    break

            if not target:
                for label, words in synonyms.items():
                    if any(w in part for w in words):
                        for ml in mask_labels:
                            if ml.lower().startswith(label):
                                target = ml
                                break
                        if target:
                            break

            if target:
                results.append({"target_label": target, "color": detected_color, "confidence": 0.5})

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
                results.append({"target_label": target, "color": detected_color, "confidence": 0.5})

        return results
