#!/usr/bin/env python3
"""
Advanced Sensory Modulation System
- Optimized YOLO/SAM pipeline for accurate object detection
- Continuous Gemini Vision scene analysis (every second)
- Natural language understanding for complex requests
- Full sensory modulation features (brightness, contrast, saturation, etc.)
- Object-local transformations with safety constraints
"""

import cv2
import numpy as np
import time
import os
import threading
import queue
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from io import BytesIO
from PIL import Image

# Models
from ultralytics import FastSAM, YOLO
import mediapipe as mp

# Speech recognition
import speech_recognition as sr

# Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Run: pip install google-generativeai")


@dataclass
class ObjectFeatureVector:
    """Feature vector for each segmented object."""
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    color_temp: float = 0.0  # Kelvin shift
    texture_detail: float = 1.0
    motion_scale: float = 1.0
    transition_time: float = 0.5
    edge_softness: float = 0.0
    highlight_suppression: float = 0.0


@dataclass
class SegmentedObject:
    """Represents a detected and segmented object."""
    id: int
    mask: np.ndarray
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    features: ObjectFeatureVector
    last_seen: float
    tracking_id: Optional[int] = None


class OptimizedDetector:
    """Optimized YOLO/SAM pipeline for accurate object detection."""
    
    def __init__(self):
        print("Loading optimized detection models...")
        
        # Use larger YOLO model for better accuracy
        self.yolo = YOLO("yolov8n.pt")
        # Could upgrade to yolov8m.pt or yolov8l.pt for better accuracy
        
        # FastSAM with optimized settings
        self.sam = FastSAM("FastSAM-s.pt")
        
        # MediaPipe for body parts
        self.selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)  # Better model
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Higher complexity for better accuracy
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        print("  ✓ All models loaded")
    
    def detect_objects(self, frame: np.ndarray) -> List[SegmentedObject]:
        """Detect and segment all objects with optimized pipeline."""
        h, w = frame.shape[:2]
        objects = []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Person segmentation (MediaPipe - most accurate)
        person_mask = self._get_person_mask(rgb, h, w)
        if np.any(person_mask > 0.5):
            bbox = self._mask_to_bbox(person_mask)
            center = self._mask_center(person_mask)
            objects.append(SegmentedObject(
                id=1,
                mask=person_mask,
                label="person",
                confidence=0.95,
                bbox=bbox,
                center=center,
                features=ObjectFeatureVector(),
                last_seen=time.time()
            ))
        
        # 2. Body parts (detailed segmentation)
        body_parts = self._get_body_parts(rgb, h, w, person_mask)
        objects.extend(body_parts)
        
        # 3. YOLO detections (optimized settings)
        yolo_detections = self._yolo_detect(frame, conf_threshold=0.2)  # Lower threshold
        
        # 4. SAM segmentation (optimized)
        sam_masks = self._sam_segment(frame, exclude_mask=person_mask)
        
        # 5. Match SAM masks to YOLO detections (improved matching)
        matched_objects = self._match_sam_to_yolo(sam_masks, yolo_detections, h, w, person_mask)
        objects.extend(matched_objects)
        
        return objects
    
    def _get_person_mask(self, rgb: np.ndarray, h: int, w: int) -> np.ndarray:
        """Get person mask from MediaPipe."""
        result = self.selfie.process(rgb)
        if result.segmentation_mask is not None:
            mask = result.segmentation_mask
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            return mask.astype(np.float32)
        return np.zeros((h, w), dtype=np.float32)
    
    def _get_body_parts(self, rgb: np.ndarray, h: int, w: int, person_mask: np.ndarray) -> List[SegmentedObject]:
        """Get detailed body part segmentation."""
        parts = []
        part_id = 10
        
        # Face
        face_result = self.face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            face_mask = np.zeros((h, w), dtype=np.float32)
            pts = np.array([[int(lm.x * w), int(lm.y * h)] 
                           for lm in face_result.multi_face_landmarks[0].landmark], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(face_mask, hull, 1.0)
            face_mask = cv2.GaussianBlur(face_mask, (5, 5), 0)
            bbox = self._mask_to_bbox(face_mask)
            center = self._mask_center(face_mask)
            parts.append(SegmentedObject(
                id=part_id, mask=face_mask, label="face", confidence=0.9,
                bbox=bbox, center=center, features=ObjectFeatureVector(),
                last_seen=time.time()
            ))
            part_id += 1
        
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
                label = "right_hand" if label_side == "Left" else "left_hand"
                bbox = self._mask_to_bbox(hand_mask)
                center = self._mask_center(hand_mask)
                parts.append(SegmentedObject(
                    id=part_id, mask=hand_mask, label=label, confidence=0.85,
                    bbox=bbox, center=center, features=ObjectFeatureVector(),
                    last_seen=time.time()
                ))
                part_id += 1
        
        # Pose-based body parts
        pose_result = self.pose.process(rgb)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks.landmark
            body_parts_map = {
                "left_arm": [11, 13, 15],
                "right_arm": [12, 14, 16],
                "torso": [11, 12, 24, 23],
                "left_leg": [23, 25, 27],
                "right_leg": [24, 26, 28],
            }
            
            for part_name, indices in body_parts_map.items():
                pts = []
                for idx in indices:
                    if idx < len(lm) and lm[idx].visibility > 0.5:
                        pts.append([int(lm[idx].x * w), int(lm[idx].y * h)])
                
                if len(pts) >= 3:
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
                        bbox = self._mask_to_bbox(part_mask)
                        center = self._mask_center(part_mask)
                        parts.append(SegmentedObject(
                            id=part_id, mask=part_mask, label=part_name, confidence=0.7,
                            bbox=bbox, center=center, features=ObjectFeatureVector(),
                            last_seen=time.time()
                        ))
                        part_id += 1
        
        return parts
    
    def _yolo_detect(self, frame: np.ndarray, conf_threshold: float = 0.2) -> List[Tuple]:
        """YOLO detection with optimized settings."""
        results = self.yolo(frame, verbose=False, conf=conf_threshold, iou=0.45)
        detections = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                label = self.yolo.names[int(box.cls[0])]
                conf = float(box.conf[0].cpu().numpy())
                detections.append(((x1, y1, x2, y2), label, conf))
        return detections
    
    def _sam_segment(self, frame: np.ndarray, exclude_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """SAM segmentation with optimized settings."""
        h, w = frame.shape[:2]
        masks = []
        
        try:
            results = self.sam(frame, device="cpu", retina_masks=True,
                             imgsz=512, conf=0.3, verbose=False)  # Higher resolution
            
            if results and results[0].masks is not None:
                for mask_data in results[0].masks.data.cpu().numpy():
                    mask = cv2.resize(mask_data.astype(np.float32), (w, h))
                    
                    # Exclude person area
                    if exclude_mask is not None:
                        mask = mask * (1 - exclude_mask * 0.8)
                    
                    # Filter small masks
                    if np.sum(mask > 0.3) < 500:
                        continue
                    
                    # Refine mask
                    mask_u8 = (mask * 255).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
                    mask = mask_u8.astype(np.float32) / 255.0
                    
                    masks.append(mask)
        except Exception as e:
            print(f"SAM error: {e}")
        
        return masks
    
    def _match_sam_to_yolo(self, sam_masks: List[np.ndarray], yolo_detections: List[Tuple],
                           h: int, w: int, person_mask: Optional[np.ndarray]) -> List[SegmentedObject]:
        """Improved matching of SAM masks to YOLO detections."""
        objects = []
        used_yolo = set()
        obj_id = 100
        
        for mask in sam_masks:
            mask_bbox = self._mask_to_bbox(mask)
            center = self._mask_center(mask)
            
            # Find best YOLO match
            best_match = None
            best_score = 0
            best_idx = None
            
            for idx, (yolo_bbox, label, conf) in enumerate(yolo_detections):
                if idx in used_yolo:
                    continue
                
                # Improved matching: IoU + center overlap + pixel overlap
                iou = self._calc_iou(mask_bbox, yolo_bbox)
                
                yolo_cx = (yolo_bbox[0] + yolo_bbox[2]) // 2
                yolo_cy = (yolo_bbox[1] + yolo_bbox[3]) // 2
                center_in_mask = False
                if 0 <= yolo_cy < h and 0 <= yolo_cx < w:
                    center_in_mask = mask[yolo_cy, yolo_cx] > 0.3
                
                mask_center_in_yolo = (yolo_bbox[0] <= center[0] <= yolo_bbox[2] and 
                                      yolo_bbox[1] <= center[1] <= yolo_bbox[3])
                
                # Pixel overlap ratio
                mask_in_box = mask[max(0, yolo_bbox[1]):min(h, yolo_bbox[3]), 
                                  max(0, yolo_bbox[0]):min(w, yolo_bbox[2])]
                overlap_ratio = np.sum(mask_in_box > 0.3) / (np.sum(mask > 0.3) + 1e-6)
                
                score = iou * 1.5 + (0.4 if center_in_mask else 0) + \
                       (0.3 if mask_center_in_yolo else 0) + overlap_ratio * 0.5
                score *= conf
                
                if score > best_score and score > 0.1:
                    best_score = score
                    best_match = (label, conf)
                    best_idx = idx
            
            if best_match:
                label, conf = best_match
                used_yolo.add(best_idx)
            else:
                # Fallback: semantic labeling
                label = self._semantic_label(mask, h, w)
                conf = 0.5
            
            objects.append(SegmentedObject(
                id=obj_id,
                mask=mask,
                label=label,
                confidence=conf,
                bbox=mask_bbox,
                center=center,
                features=ObjectFeatureVector(),
                last_seen=time.time()
            ))
            obj_id += 1
        
        return objects
    
    def _semantic_label(self, mask: np.ndarray, h: int, w: int) -> str:
        """Semantic labeling for unmatched masks."""
        ys, xs = np.where(mask > 0.5)
        if len(ys) == 0:
            return "unknown"
        
        cy = np.mean(ys)
        cx = np.mean(xs)
        area = len(ys)
        area_ratio = area / (h * w)
        
        # Structural elements
        touches_top = np.any(ys < h * 0.05)
        touches_bottom = np.any(ys > h * 0.95)
        touches_left = np.any(xs < w * 0.05)
        touches_right = np.any(xs > w * 0.05)
        
        if area_ratio > 0.15:
            if touches_top and cy < h * 0.4:
                return "ceiling"
            elif touches_bottom and cy > h * 0.6:
                return "floor"
            elif (touches_left or touches_right) and area_ratio > 0.1:
                return "wall"
            elif cy < h * 0.35:
                return "ceiling"
            elif cy > h * 0.65:
                return "floor"
            else:
                return "wall"
        
        # Windows (often bright, rectangular, mid-height)
        if 0.05 < area_ratio < 0.15 and 0.3 < cy/h < 0.7:
            bbox_h = ys.max() - ys.min()
            bbox_w = xs.max() - xs.min()
            aspect = bbox_w / (bbox_h + 1)
            if 1.2 < aspect < 3.0:
                return "window"
        
        # Furniture
        if area_ratio > 0.03:
            if cy > h * 0.55:
                return "furniture"
            elif cy < h * 0.4:
                return "shelf"
            else:
                return "surface"
        
        return "object"
    
    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box from mask."""
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    
    def _mask_center(self, mask: np.ndarray) -> Tuple[int, int]:
        """Get center point of mask."""
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return (0, 0)
        return (int(np.mean(xs)), int(np.mean(ys)))
    
    def _calc_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)
    
    def close(self):
        """Clean up resources."""
        self.selfie.close()
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()


# Continue in next part due to length...

