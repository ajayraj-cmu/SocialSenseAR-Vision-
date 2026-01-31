"""Main environment controller combining SAM, Gemini, and Voice."""
import os
import time
import json
import threading
import cv2
import numpy as np
from ultralytics import FastSAM
import mediapipe as mp


class EnvironmentController:
    """Main controller combining SAM, Gemini, and Voice."""

    def __init__(self, gemini_agent, voice_listener, audio_processor):
        print("\n" + "="*60)
        print("  🎙️ SAM + GEMINI VOICE CONTROLLER")
        print("="*60 + "\n")

        # Store external dependencies
        self.gemini = gemini_agent
        self.voice = voice_listener
        self.audio_processor = audio_processor

        # Load models
        print("Loading models...")
        self.sam = FastSAM("models/FastSAM-s.pt")
        print("  ✓ FastSAM")

        # Indoor vocabulary for object detection
        self.indoor_campus_vocab = [
            "wall", "walls", "door", "doors", "window", "windows", "ceiling", "floor",
            "desk", "desks", "chair", "chairs", "table", "tables", "cabinet", "cabinets",
            "light", "lights", "lamp", "lamps", "ceiling light", "desk lamp",
            "laptop", "monitor", "screen", "keyboard", "mouse", "printer", "tv",
            "person", "people", "face", "book", "bottle", "cup", "bag", "phone"
        ]
        self._indoor_vocab_base = list(self.indoor_campus_vocab)
        self._dynamic_vocab = set()
        self._vocab_lock = threading.Lock()
        self._last_vocab_refresh = 0.0
        self._vocab_refresh_interval = float(os.environ.get("VOCAB_REFRESH_SEC", "2.0"))
        self._max_total_vocab = int(os.environ.get("MAX_TOTAL_VOCAB", "320"))

        # No YOLO - using Gemini Vision for labeling
        self.use_yolo_world = False
        self.yolo = None
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
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        print("  ✓ MediaPipe (selfie + face + hands + pose)")

        # State
        self.masks = []
        self.active_effects = {}
        self.person_mask = None
        self.frame_count = 0
        self.last_scene_update = 0
        self.yolo_detections = []
        self.label_counts = {}
        self.current_frame = None

        # Label persistence
        self.persistent_labels = {}
        self.label_lock_threshold = 0.3
        self.label_persistence_time = 10.0
        self.label_change_threshold = 0.5

        # Autopilot
        self.autopilot_enabled = os.environ.get("AUTO_AUTOPILOT", "1") == "1"
        self._autopilot_last = 0.0
        self._autopilot_interval = float(os.environ.get("AUTO_AUTOPILOT_SEC", "6.0"))

        # Performance optimizations
        self.last_mask_update = 0
        self.mask_update_interval = 1.0
        self.last_yolo_update = 0
        self.yolo_update_interval = 1.5
        self.frame_cache = None
        self.cached_contours = {}
        self.last_frame_shape = None

        # Clean view mode
        self.clean_view_mode = False

        # Mask tracking
        self.tracked_masks = {}
        self.next_track_id = 0
        self.max_frames_missing = 15
        self.velocity_smoothing = 0.3

        # Feedback loop system
        self.last_feedback_validation = 0
        self.feedback_validation_interval = 15.0
        self.detection_history = []
        self.adaptive_thresholds = {
            "yolo_conf": 0.10,
            "sam_conf": 0.25,
            "matching_iou": 0.08
        }

        self._sync_thresholds()

        # Color map (BGR)
        self.color_map = {
            "red": (0, 0, 255), "blue": (255, 50, 50), "green": (0, 255, 0),
            "yellow": (0, 255, 255), "purple": (255, 0, 255), "orange": (0, 165, 255),
            "pink": (203, 192, 255), "cyan": (255, 255, 0), "white": (255, 255, 255),
            "black": (0, 0, 0), "dark_gray": (40, 40, 40),
        }

        print("\n✅ Ready!")
        print("   🎤 Say 'hey vibe' then your command, end with 'thanks'")
        print("   ⌨️  Q=quit  C=clear  S=screenshot  P=toggle autopilot\n")

    def _sync_thresholds(self):
        """Sync adaptive thresholds with feedback loop optimizations."""
        if self.gemini.available:
            optimal = self.gemini.get_optimal_thresholds()
            self.adaptive_thresholds["yolo_conf"] = optimal.get('yolo_threshold', 0.15)
            self.adaptive_thresholds["sam_conf"] = optimal.get('sam_confidence', 0.3)
            self.adaptive_thresholds["matching_iou"] = optimal.get('iou_threshold', 0.45)

    def process_frame(self, frame):
        """Process a single frame - OPTIMIZED."""
        h, w = frame.shape[:2]
        current_time = time.time()
        display = frame.copy()

        # Update person mask less frequently
        if self.frame_count % 2 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selfie_result = self.selfie.process(rgb)
            self.person_mask = selfie_result.segmentation_mask if selfie_result.segmentation_mask is not None else np.zeros((h, w), dtype=np.float32)

        self.frame_count += 1

        # Smart mask update: time-based for consistent performance
        if current_time - self.last_mask_update > self.mask_update_interval:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._update_masks(frame, h, w, rgb)
            self.last_mask_update = current_time
            self.cached_contours.clear()

        # Store current frame for Gemini Vision
        if self.frame_count % 10 == 0:
            self.current_frame = frame.copy()

        # Feedback loop: Continuous validation and self-correction
        if self.gemini.available and current_time - self.last_feedback_validation > self.feedback_validation_interval:
            mask_labels = [m[1] for m in self.masks]
            mask_centers = [m[2] for m in self.masks]
            detected_objects = []
            for mask, label, center in self.masks:
                detected_objects.append({
                    "label": label,
                    "position": f"({center[0]}, {center[1]})" if center else "unknown",
                    "confidence": 0.7
                })
            threading.Thread(
                target=self._run_feedback_loop,
                args=(frame.copy(), detected_objects, self.yolo_detections.copy()),
                daemon=True
            ).start()
            self.last_feedback_validation = current_time

        # Update scene analysis with Gemini Vision
        if self.gemini.available and current_time - self.last_scene_update > 1.0:
            mask_labels = [m[1] for m in self.masks]
            mask_centers = [m[2] for m in self.masks]
            threading.Thread(target=self.gemini.analyze_scene_continuous,
                           args=(frame.copy(), mask_labels, mask_centers), daemon=True).start()
            self.last_scene_update = current_time

        # Comprehensive feedback loop
        if self.gemini.available:
            mask_labels = [m[1] for m in self.masks]
            mask_centers = [m[2] for m in self.masks]
            threading.Thread(target=self.gemini.comprehensive_feedback_loop,
                           args=(frame.copy(), mask_labels, self.yolo_detections, mask_centers),
                           daemon=True).start()
            self._sync_thresholds()

        # Check for video commands
        command = self.voice.get_command()
        if command:
            self._process_voice_command(command)

        # Check for audio commands
        audio_command = self.voice.get_audio_command()
        if audio_command:
            self._process_audio_command(audio_command)

        # Autopilot
        self._autopilot_step()

        # Initialize persistent label cache
        if not hasattr(self, '_spatial_labels'):
            self._spatial_labels = {}
            self._last_gemini_label_time = 0

        def get_spatial_key(center, mask_area):
            if not center or center[0] <= 0:
                return None
            cx, cy = center
            grid_x = int(cx / w * 16)
            grid_y = int(cy / h * 16)
            size_bucket = 0 if mask_area < 0.02 else (1 if mask_area < 0.1 else 2)
            return f"{grid_x}_{grid_y}_{size_bucket}"

        # Request Gemini labels for unlabeled segments
        unlabeled_masks = []
        for idx, (mask, old_label, center) in enumerate(self.masks):
            if mask is None:
                continue
            mask_area = np.sum(mask > 0.5) / (h * w)
            spatial_key = get_spatial_key(center, mask_area)
            if spatial_key and spatial_key not in self._spatial_labels:
                unlabeled_masks.append((idx, mask, center, spatial_key, mask_area))

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
                                self._spatial_labels[spatial_key] = (label, 1.0, current_time)
                self._last_gemini_label_time = current_time

        # Clean up old labels
        stale_keys = [k for k, v in self._spatial_labels.items() if current_time - v[2] > 30.0]
        for k in stale_keys:
            del self._spatial_labels[k]

        # Build label -> mask mapping with persistent labels
        all_labels_to_mask = {}
        updated_masks = []
        label_counts = {}

        for idx, (mask, old_label, center) in enumerate(self.masks):
            if mask is None:
                continue

            mask_area = np.sum(mask > 0.5) / (h * w)
            spatial_key = get_spatial_key(center, mask_area)

            if spatial_key and spatial_key in self._spatial_labels:
                new_label = self._spatial_labels[spatial_key][0]
            elif old_label and not old_label.startswith("region_") and not old_label.startswith("segment_"):
                new_label = old_label
            else:
                if center and center[0] > 0:
                    cx, cy = center
                    h_pos = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
                    v_pos = "top" if cy < h/3 else "bottom" if cy > 2*h/3 else "middle"
                    new_label = f"~{v_pos}_{h_pos}"
                else:
                    new_label = f"~segment_{idx}"

            base_label = new_label
            if base_label in label_counts:
                label_counts[base_label] += 1
                new_label = f"{base_label}_{label_counts[base_label]}"
            else:
                label_counts[base_label] = 1

            updated_masks.append((mask, new_label, center))
            all_labels_to_mask[new_label] = mask

        self.masks = updated_masks

        # Mask tracking
        self._update_tracked_masks(updated_masks, h, w)
        tracked_masks_for_display = self._get_tracked_masks_for_display(h, w)

        # Filter: only include masks with active effects
        all_labels_to_mask = {}
        for mask, label, center in tracked_masks_for_display:
            matched_effect = self._match_effect(label)
            if matched_effect:
                all_labels_to_mask[label] = mask

        # Detect adjacent boundaries
        adjacent_boundaries = self._detect_adjacent_boundaries(tracked_masks_for_display)

        # Draw borders and labels
        bright_colors = [
            (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0),
            (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128),
            (255, 255, 255), (100, 255, 200),
        ]

        if not self.clean_view_mode:
            for idx, (mask, label, center) in enumerate(tracked_masks_for_display):
                if mask is None or mask.shape != (h, w):
                    continue

                mask_u8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                matched_effect = self._match_effect(label)
                has_effect = matched_effect is not None

                if has_effect:
                    border_color = bright_colors[idx % len(bright_colors)]
                    border_width = 2
                else:
                    border_color = (100, 100, 100)
                    border_width = 1

                cv2.drawContours(display, contours, -1, border_color, border_width)

                if center and center[0] > 0 and label:
                    cx, cy = int(center[0]), int(center[1])
                    is_pending = label.startswith("~")
                    display_label = label[1:] if is_pending else label

                    if is_pending:
                        (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                        cv2.rectangle(display, (cx-2, cy-th-2), (cx+tw+2, cy+2), (50, 50, 50), -1)
                        cv2.putText(display, display_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
                    else:
                        if has_effect:
                            (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(display, (cx-3, cy-th-4), (cx+tw+3, cy+4), (0, 0, 0), -1)
                            cv2.putText(display, display_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)
                        else:
                            (tw, th), _ = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.rectangle(display, (cx-2, cy-th-3), (cx+tw+2, cy+3), (30, 30, 30), -1)
                            cv2.putText(display, display_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            # Draw adjacent boundaries
            boundary_color = (255, 255, 0)
            for boundary_info in adjacent_boundaries:
                boundary_mask = boundary_info['boundary']
                boundary_contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, boundary_contours, -1, boundary_color, 2)

        # Apply effects
        for label, mask in all_labels_to_mask.items():
            matched_effect = self._match_effect(label)
            if not matched_effect:
                continue

            if isinstance(matched_effect, str) and matched_effect.startswith("mod_"):
                try:
                    mod_json = matched_effect[4:]
                    mod_params = json.loads(mod_json)
                    mask_bool = mask > 0.5

                    # BLUR MODE
                    if mod_params.get('blur'):
                        blur_strength = int(mod_params.get('blur_strength', 25))
                        blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
                        blurred = cv2.GaussianBlur(display, (blur_strength, blur_strength), 0)
                        display[mask_bool] = blurred[mask_bool]

                    # BRIGHTNESS MODE
                    if 'brightness' in mod_params:
                        brightness_factor = mod_params['brightness']
                        display[mask_bool] = np.clip(
                            display[mask_bool].astype(np.float32) * brightness_factor, 0, 255
                        ).astype(np.uint8)

                    # CONTRAST MODE
                    if 'contrast' in mod_params:
                        contrast_factor = mod_params['contrast']
                        region = display[mask_bool].astype(np.float32)
                        region = (region - 127.5) * contrast_factor + 127.5
                        display[mask_bool] = np.clip(region, 0, 255).astype(np.uint8)

                    # SATURATION MODE
                    if 'saturation' in mod_params:
                        sat_factor = mod_params['saturation']
                        hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV).astype(np.float32)
                        hsv[mask_bool, 1] *= sat_factor
                        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                        display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    # MOTION DAMPENING MODE
                    if mod_params.get('motion_dampen'):
                        if not hasattr(self, 'prev_regions'):
                            self.prev_regions = {}
                        region_key = f"motion_{label}"
                        smooth_factor = mod_params.get('temporal_smooth', 0.7)
                        current_region = display[mask_bool].copy()
                        if region_key in self.prev_regions:
                            prev_region = self.prev_regions[region_key]
                            if prev_region.shape == current_region.shape:
                                blended = (
                                    current_region.astype(np.float32) * (1 - smooth_factor) +
                                    prev_region.astype(np.float32) * smooth_factor
                                ).astype(np.uint8)
                                display[mask_bool] = blended
                        self.prev_regions[region_key] = current_region

                    mask_u8 = (mask * 255).astype(np.uint8)
                    effect_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, effect_contours, -1, (0, 255, 255), 3)

                except Exception as e:
                    print(f"  ⚠️ Modulation error: {e}")
                    pass
            else:
                # COLOR REMAPPING MODE
                color_name = matched_effect
                bgr_color = self.color_map.get(color_name, (255, 50, 50))
                mask_bool = mask > 0.5
                mask_3d = np.stack([mask, mask, mask], axis=2)
                color_overlay = np.full_like(display, bgr_color, dtype=np.float32)
                alpha = 0.6
                display = np.where(
                    mask_3d > 0.5,
                    (display.astype(np.float32) * (1 - alpha) + color_overlay * alpha).astype(np.uint8),
                    display
                )
                mask_u8 = (mask * 255).astype(np.uint8)
                color_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, color_contours, -1, (255, 255, 0), 3)

        return display

    def _update_masks(self, frame, h, w, rgb=None):
        """Update segmentation masks - OPTIMIZED."""
        self.masks = []
        self.label_counts = {}

        if rgb is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_time = time.time()
        update_yolo = (current_time - self.last_yolo_update > self.yolo_update_interval)

        if update_yolo:
            self.yolo_detections = []
            self.last_yolo_update = current_time

        # Body parts from MediaPipe
        if self.person_mask is not None and np.any(self.person_mask > 0.5):
            pm = self.person_mask.copy()
            if pm.shape != (h, w):
                pm = cv2.resize(pm, (w, h))
            center = self._mask_center(pm)
            self.masks.append((pm, "person", center))

        # Face mask
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

        # Hands
        hands_result = self.hands.process(rgb)
        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for hand_lm, handedness in zip(hands_result.multi_hand_landmarks, hands_result.multi_handedness):
                hand_mask = np.zeros((h, w), dtype=np.float32)
                pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_lm.landmark], dtype=np.int32)
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(hand_mask, hull, 1.0)
                kernel = np.ones((9, 9), np.uint8)
                hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
                hand_mask = cv2.GaussianBlur(hand_mask, (5, 5), 0)

                label_side = handedness.classification[0].label
                hand_label = "right_hand" if label_side == "Left" else "left_hand"
                center = self._mask_center(hand_mask)
                self.masks.append((hand_mask, hand_label, center))

        # Body parts from Pose
        pose_result = self.pose.process(rgb)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            body_parts = {
                "left_arm": [11, 13, 15],
                "right_arm": [12, 14, 16],
                "torso": [11, 12, 24, 23],
                "left_leg": [23, 25, 27, 31],
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

                    if "arm" in part_name or "leg" in part_name:
                        for i in range(len(pts) - 1):
                            cv2.line(part_mask, tuple(pts[i]), tuple(pts[i+1]), 1.0, 25)
                    else:
                        hull = cv2.convexHull(pts_arr)
                        cv2.fillConvexPoly(part_mask, hull, 1.0)

                    kernel = np.ones((7, 7), np.uint8)
                    part_mask = cv2.dilate(part_mask, kernel, iterations=1)
                    part_mask = cv2.GaussianBlur(part_mask, (7, 7), 0)

                    if np.any(part_mask > 0.3):
                        center = self._mask_center(part_mask)
                        self.masks.append((part_mask, part_name, center))

        # SAM masks
        yolo_data = []
        used_yolo = set()
        self.yolo_detections = []

        used_pixels = np.zeros((h, w), dtype=bool)
        try:
            optimal_thresholds = self.gemini.get_optimal_thresholds()
            sam_conf = optimal_thresholds.get('sam_confidence', 0.3)
            target_size = min(512, max(320, min(h, w)))
            sam_results = self.sam(frame, device="cpu", retina_masks=True,
                                  imgsz=target_size, conf=sam_conf, verbose=False)

            if sam_results and sam_results[0].masks is not None:
                if self.person_mask is not None:
                    used_pixels |= (self.person_mask > 0.5)

                for mask_data in sam_results[0].masks.data.cpu().numpy():
                    mask = cv2.resize(mask_data.astype(np.float32), (w, h))
                    mask_binary = mask > 0.5
                    clean_mask = mask_binary & ~used_pixels

                    if np.sum(clean_mask) < 500:
                        continue

                    clean_mask_float = clean_mask.astype(np.float32)
                    refined_mask = self._refine_mask_edges(frame, clean_mask_float)

                    if refined_mask is not None and np.sum(refined_mask > 0.5) >= 500:
                        clean_mask = refined_mask
                    else:
                        mask_u8 = (clean_mask * 255).astype(np.uint8)
                        kernel = np.ones((3,3), np.uint8)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
                        clean_mask = mask_u8.astype(np.float32) / 255.0

                    used_pixels |= (clean_mask > 0.5)

                    label, yolo_idx = self._get_label(clean_mask, yolo_data, h, w, used_yolo, frame)
                    if yolo_idx is not None:
                        used_yolo.add(yolo_idx)

                    mask_area = np.sum(clean_mask > 0.3)
                    area_ratio = mask_area / (h * w)
                    if label in ["person", "backpack", "handbag"] and area_ratio > 0.12:
                        label, _ = self._get_label(clean_mask, [], h, w, set(), frame)

                    corrected_label = self.gemini.get_corrected_label(label)
                    if corrected_label != label:
                        print(f"  🔄 Applying Gemini Vision correction: '{label}' → '{corrected_label}'")
                        label = corrected_label

                    label = self._unique_label(label)
                    center = self._mask_center(clean_mask)
                    self.masks.append((clean_mask, label, center))

        except Exception as e:
            print(f"SAM error: {e}")

    def _unique_label(self, label):
        """Make label unique by adding number if duplicate."""
        if label not in self.label_counts:
            self.label_counts[label] = 1
            return label
        else:
            self.label_counts[label] += 1
            return f"{label}_{self.label_counts[label]}"

    def _get_label(self, mask, yolo_data, h, w, used_yolo, frame=None):
        """Get label for mask from YOLO or semantic analysis."""
        ys, xs = np.where(mask > 0.5)

        if len(ys) == 0:
            return "area", None

        cy = np.mean(ys)
        cx = np.mean(xs)
        area = len(ys)
        total_area = h * w
        area_ratio = area / total_area

        # Very large areas = structural elements
        if area_ratio > 0.12:
            touches_top = np.any(ys < h * 0.05)
            touches_bottom = np.any(ys > h * 0.95)
            if touches_top and cy < h * 0.4:
                return "ceiling", None
            elif touches_bottom and cy > h * 0.6:
                return "floor", None
            elif cy < h * 0.35:
                return "ceiling", None
            elif cy > h * 0.65:
                return "floor", None
            else:
                return "wall", None

        # Medium areas - furniture-sized
        if area_ratio > 0.03:
            bbox_h = ys.max() - ys.min()
            bbox_w = xs.max() - xs.min()
            aspect = bbox_w / (bbox_h + 1)

            if cy > h * 0.55:
                if aspect > 2:
                    return "table", None
                elif aspect < 0.5:
                    return "cabinet", None
                else:
                    return "furniture", None
            elif cy < h * 0.4:
                if aspect > 1.5:
                    return "shelf", None
                else:
                    return "cabinet", None
            else:
                if aspect > 2:
                    return "monitor", None
                else:
                    return "furniture", None

        # Small objects
        if area_ratio > 0.005:
            bbox_h = ys.max() - ys.min()
            bbox_w = xs.max() - xs.min()
            aspect = bbox_w / (bbox_h + 1)

            if aspect > 1.5:
                return "item_wide", None
            elif aspect < 0.7:
                return "item_tall", None
            else:
                return "item", None

        return "small_item", None

    def _refine_mask_edges(self, frame, mask, iterations=2):
        """Refine mask edges using bilateral filtering and morphological operations."""
        if mask is None or frame is None:
            return mask

        try:
            mask_u8 = (mask * 255).astype(np.uint8)
            mask_filtered = cv2.bilateralFilter(mask_u8, 5, 50, 50)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_opened = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel)
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
            return mask_closed.astype(np.float32) / 255.0
        except Exception:
            return mask

    def _detect_adjacent_boundaries(self, masks_with_labels):
        """Detect boundaries between adjacent objects for multi-color tracing."""
        if len(masks_with_labels) < 2:
            return []

        boundaries = []
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
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                dilated_i = cv2.dilate(mask_i_u8, kernel, iterations=1)
                dilated_j = cv2.dilate(mask_j_u8, kernel, iterations=1)
                intersection = cv2.bitwise_and(dilated_i, dilated_j)

                if np.sum(intersection > 128) > 50:
                    boundaries.append({
                        'mask_i': i, 'mask_j': j, 'label_i': label_i,
                        'label_j': label_j, 'boundary': intersection
                    })

        return boundaries

    def _mask_center(self, mask):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0)
        return (int(np.mean(xs)), int(np.mean(ys)))

    def _update_tracked_masks(self, new_masks, h, w):
        """Update mask tracking - persist masks during movement/brief detection gaps."""
        current_time = time.time()

        for track_id in self.tracked_masks:
            self.tracked_masks[track_id]['seen_this_frame'] = False

        for mask, label, center in new_masks:
            if mask is None or center is None or center[0] <= 0:
                continue

            best_track_id = None
            best_distance = float('inf')

            for track_id, track in self.tracked_masks.items():
                if track['seen_this_frame']:
                    continue

                track_center = track['center']
                if track['velocity']:
                    dt = current_time - track['last_seen']
                    pred_x = track_center[0] + track['velocity'][0] * dt
                    pred_y = track_center[1] + track['velocity'][1] * dt
                    track_center = (pred_x, pred_y)

                dist = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)

                if track['label'] == label or (track['label'] and label and
                    track['label'].replace("~", "") == label.replace("~", "")):
                    dist *= 0.5

                if dist < best_distance and dist < 100:
                    best_distance = dist
                    best_track_id = track_id

            if best_track_id is not None:
                track = self.tracked_masks[best_track_id]
                dt = current_time - track['last_seen']
                if dt > 0 and dt < 1.0:
                    new_vx = (center[0] - track['center'][0]) / dt
                    new_vy = (center[1] - track['center'][1]) / dt
                    if track['velocity']:
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
                self.tracked_masks[self.next_track_id] = {
                    'mask': mask, 'label': label, 'center': center, 'velocity': None,
                    'last_seen': current_time, 'frames_missing': 0, 'seen_this_frame': True
                }
                self.next_track_id += 1

        tracks_to_remove = []
        for track_id, track in self.tracked_masks.items():
            if not track['seen_this_frame']:
                track['frames_missing'] += 1
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

    def _autopilot_step(self):
        """Inject a synthetic command to stress-test labeling/effects."""
        if not self.autopilot_enabled or self.voice.is_busy():
            return
        now = time.time()
        if now - self._autopilot_last < self._autopilot_interval or not self.masks:
            return

        labels = [m[1] for m in self.masks]
        avoid_prefix = {"person", "face", "left_hand", "right_hand", "left_arm", "right_arm", "torso", "left_leg", "right_leg"}
        candidates = [l for l in labels if l.split("_")[0] not in avoid_prefix]
        if not candidates:
            candidates = labels

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

    def _process_voice_command(self, command):
        """Process a voice command through Gemini VISION."""
        print(f"\n{'='*50}")
        print(f"🎯 Processing: '{command}'")

        mask_labels = [m[1] for m in self.masks]
        mask_centers = [m[2] for m in self.masks]
        print(f"📋 Available labels: {mask_labels}")

        command_lower = command.lower()
        environmental_keywords = [
            "lighting", "lights", "bright", "dark", "glare", "harsh",
            "overstimulating", "overwhelming", "too much", "visual noise",
            "distracting", "colorful", "saturated", "contrast"
        ]

        is_environmental = any(keyword in command_lower for keyword in environmental_keywords)
        no_specific_object = not any(label.lower() in command_lower for label in mask_labels if not label.startswith("~"))

        if is_environmental and no_specific_object:
            print("🌍 Detected environmental command - using semantic understanding...")
            results = self.gemini.process_environmental_command(command, mask_labels, mask_centers)
        elif self.current_frame is not None:
            print("👁️ Using Gemini Vision to analyze scene...")
            results = self.gemini.process_request_with_vision(command, self.current_frame, mask_labels, mask_centers)
        else:
            results = []

        if isinstance(results, dict):
            results = [results]

        # Fix BLUR vs BLUE confusion
        user_wants_blur = "blur" in command_lower and "blue" not in command_lower
        if user_wants_blur:
            print("  🔧 Detected BLUR request - forcing blur mode on all results")
            for r in results:
                r["blur"] = True
                r["blur_strength"] = r.get("blur_strength") or 25
                r["color"] = None
                if r.get("confidence", 0) < 0.5:
                    r["confidence"] = 0.5

        # Extract explicit color from command
        explicit_colors = {
            "red": "red", "blue": "blue", "green": "green", "yellow": "yellow",
            "purple": "purple", "orange": "orange", "pink": "pink", "cyan": "cyan",
            "white": "white", "black": "black", "gray": "gray", "grey": "gray"
        }
        user_color = None
        for color_word, color_value in explicit_colors.items():
            if f" {color_word}" in f" {command_lower} " or command_lower.startswith(color_word):
                if color_word == "blue" and "blur" in command_lower:
                    continue
                user_color = color_value
                break

        if user_color and not user_wants_blur:
            print(f"  🎨 Detected explicit color request: {user_color}")
            for r in results:
                r["color"] = user_color
                if r.get("confidence", 0) < 0.5:
                    r["confidence"] = 0.5

        if not results or len(results) == 0 or all(not r.get("target_label") for r in results):
            print("  🔄 Gemini Vision failed, using direct text matching...")
            results = self._direct_text_matching(command, mask_labels)

        applied_count = 0
        request_lower = command.lower()
        wants_all = "all" in request_lower or any(word in request_lower for word in ["every", "each"])

        for result in results:
            target = result.get("target_label")
            color = result.get("color")
            confidence = result.get("confidence", 0)

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

            if target and confidence > 0.2:
                if wants_all:
                    matching_labels = []
                    target_base = target.lower().rstrip('0123456789').rstrip('_')
                    target_singular = target_base[:-1] if target_base.endswith('s') else target_base

                    for label in mask_labels:
                        label_lower = label.lower()
                        label_base = label_lower.rstrip('0123456789').rstrip('_')
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
                        matched_label = self._find_best_label_match(target, mask_labels)
                        if matched_label:
                            effect = self._build_effect(color, brightness, saturation, contrast,
                                                       highlight_suppression, blur, blur_strength,
                                                       motion_dampen, temporal_smooth)
                            self.active_effects[matched_label] = effect
                            print(f"  ✅ Applied effect to '{matched_label}'")
                            applied_count += 1
                else:
                    matched_label = self._find_best_label_match(target, mask_labels)
                    if matched_label:
                        effect = self._build_effect(color, brightness, saturation, contrast,
                                                   highlight_suppression, blur, blur_strength,
                                                   motion_dampen, temporal_smooth)
                        self.active_effects[matched_label] = effect
                        print(f"  ✅ Applied effect to '{matched_label}'")
                        applied_count += 1
                    else:
                        effect = self._build_effect(color, brightness, saturation, contrast,
                                                   highlight_suppression, blur, blur_strength,
                                                   motion_dampen, temporal_smooth)
                        self.active_effects[target] = effect
                        print(f"  ⚠️ No exact match, stored '{target}' -> effect")
                        applied_count += 1

        if applied_count == 0:
            print(f"❓ Couldn't determine target(s). Available: {mask_labels}")
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

        if any(phrase in command_lower for phrase in [
            "dampen", "damp", "reduce", "suppress", "isolate", "voice isolation",
            "noise suppression", "quiet", "mute background", "background noise"
        ]):
            self.audio_processor.set_voice_isolation(True)
            print("  ✅ Voice isolation ENABLED - surrounding audio dampened")
        elif any(phrase in command_lower for phrase in [
            "disable", "turn off", "stop", "full audio", "passthrough",
            "normal audio", "all audio"
        ]):
            self.audio_processor.set_voice_isolation(False)
            print("  ✅ Voice isolation DISABLED - full audio passthrough")
        elif "toggle" in command_lower:
            new_state = not self.audio_processor.voice_isolation_enabled
            self.audio_processor.set_voice_isolation(new_state)
        else:
            print(f"  ❓ Unknown audio command. Try: 'dampen surrounding audio' or 'disable voice isolation'")

    def _direct_text_matching(self, command, available_labels):
        """Direct text matching fallback when Gemini fails."""
        command_lower = command.lower()
        results = []
        command_words = set(command_lower.split())

        wants_blur = any(w in command_lower for w in ["blur", "blurry", "hide", "block", "private", "focus", "obscure"])
        blur_strength = 25
        if "strong" in command_lower or "heavy" in command_lower:
            blur_strength = 45
        elif "light" in command_lower or "slight" in command_lower:
            blur_strength = 15

        wants_dim = any(w in command_lower for w in ["dim", "darken", "bright", "glare", "harsh", "overstimulated"])
        brightness = 0.3 if wants_dim else None

        color = None
        color_words = {
            "blue": [" blue", "blue "],
            "red": ["red"], "green": ["green"], "purple": ["purple"],
            "yellow": ["yellow"], "gray": ["gray", "grey"],
            "orange": ["orange"], "pink": ["pink"],
        }
        for c, patterns in color_words.items():
            if c == "blue":
                if "blur" not in command_lower and any(p in f" {command_lower} " for p in patterns):
                    color = c
                    break
            elif any(p in command_lower for p in patterns):
                color = c
                break

        # Match common patterns
        if any(w in command_lower for w in ["screen", "monitor", "laptop", "computer", "tv", "display"]):
            screen_labels = [label for label in available_labels
                           if any(w in label.lower() for w in ["screen", "monitor", "laptop", "computer", "tv", "display"])]
            for label in screen_labels:
                results.append({
                    "target_label": label, "blur": wants_blur,
                    "blur_strength": blur_strength if wants_blur else None,
                    "brightness": brightness, "confidence": 0.85
                })

        face_triggers = ["people", "person", "face", "faces", "looking", "staring", "my face", "myself"]
        if any(w in command_lower for w in face_triggers):
            people_labels = [label for label in available_labels
                           if any(w in label.lower() for w in ["person", "face", "woman", "man", "people", "child", "boy", "girl"])]
            should_blur = wants_blur or any(w in command_lower for w in ["hide", "private", "looking", "staring"])

            for label in people_labels:
                results.append({
                    "target_label": label, "blur": should_blur,
                    "blur_strength": blur_strength if should_blur else None,
                    "color": color if not should_blur else None, "confidence": 0.9
                })

        # Generic label matching
        if not results:
            for label in available_labels:
                label_lower = label.lower()
                label_words = set(label_lower.replace('_', ' ').split())

                if command_words & label_words or label_lower in command_lower or any(w in label_lower for w in command_words):
                    results.append({
                        "target_label": label, "brightness": brightness, "color": color,
                        "blur": wants_blur, "blur_strength": blur_strength if wants_blur else None,
                        "confidence": 0.7
                    })
                    break

        return results

    def _run_feedback_loop(self, frame, detected_objects, yolo_detections):
        """Run the feedback loop to validate and correct detections."""
        try:
            mask_labels = [obj.get("label", "unknown") for obj in detected_objects]
            mask_centers = []
            for obj in detected_objects:
                pos_str = obj.get("position", "unknown")
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

            self.gemini.comprehensive_feedback_loop(frame, mask_labels, yolo_detections, mask_centers)
        except Exception as e:
            print(f"  ⚠️ Feedback loop error: {e}")

    def _find_best_label_match(self, target, available_labels):
        """Find the best matching label from available labels."""
        if not target or not available_labels:
            return None

        target_lower = target.lower().strip()
        target_base = target_lower.rstrip('0123456789').rstrip('_')
        target_singular = target_lower[:-1] if target_lower.endswith('s') and len(target_lower) > 1 else target_lower

        target_lower_set = {target_lower, target_base, target_singular}
        for label in available_labels:
            if label.lower() in target_lower_set:
                return label

        for label in available_labels:
            if label.lower() == target_lower:
                return label

        for label in available_labels:
            label_lower = label.lower()
            if target_singular in label_lower or label_lower in target_singular:
                return label

        for label in available_labels:
            label_base = label.lower().rstrip('0123456789').rstrip('_')
            if label_base == target_base or label_base == target_singular:
                return label

        return None

    def _match_effect(self, label):
        """STRICT match a label to active effects."""
        label_lower = label.lower()

        if label in self.active_effects:
            return self.active_effects[label]

        for effect_label, color in self.active_effects.items():
            effect_lower = effect_label.lower()
            if label_lower == effect_lower:
                return color

            label_base = label_lower.rstrip('0123456789').rstrip('_')
            effect_base = effect_lower.rstrip('0123456789').rstrip('_')

            if label_base == effect_base and label_base:
                return color

            synonym_groups = [
                {"wall", "background"}, {"floor", "ground"}, {"ceiling"},
                {"person", "body"}, {"face", "head"}, {"left_hand"}, {"right_hand"}
            ]

            for group in synonym_groups:
                effect_in_group = any(s == effect_base or s in effect_lower for s in group)
                label_in_group = any(s == label_base or s in label_lower for s in group)
                if effect_in_group and label_in_group:
                    return color

        return None

    def _build_effect(self, color=None, brightness=None, saturation=None, contrast=None,
                     highlight_suppression=None, blur=None, blur_strength=None,
                     motion_dampen=None, temporal_smooth=None):
        """Build effect string from parameters."""
        if color and not any([brightness, saturation, contrast, blur, motion_dampen]):
            return color

        mod_dict = {}
        if blur:
            mod_dict['blur'] = True
            mod_dict['blur_strength'] = blur_strength or 25
        if brightness is not None:
            mod_dict['brightness'] = brightness
        if saturation is not None:
            mod_dict['saturation'] = saturation
        if contrast is not None:
            mod_dict['contrast'] = contrast
        if highlight_suppression is not None:
            mod_dict['highlight_suppression'] = highlight_suppression
        if motion_dampen:
            mod_dict['motion_dampen'] = True
            mod_dict['temporal_smooth'] = temporal_smooth or 0.7

        if mod_dict:
            return f"mod_{json.dumps(mod_dict)}"

        return color if color else None

    def clear_effects(self):
        """Clear all active effects."""
        self.active_effects = {}
        print("🧹 Cleared all effects")

    def close(self):
        self.voice.stop()
        if self.audio_processor:
            self.audio_processor.close()
        self.selfie.close()
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()
