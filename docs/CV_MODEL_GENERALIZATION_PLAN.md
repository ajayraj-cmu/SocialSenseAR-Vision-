# CV Model Generalization Plan

**Goal:** Make SocialSenseAR generalizable for a curated subset of popular computer vision models, with active display support in Unity.

---

## 1. Target Models (Subset with GitHub)

| Model | GitHub | Stars | Output Type | Status |
|-------|--------|-------|-------------|--------|
| **YOLO** (Ultralytics) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 53k+ | detection, segmentation, pose | Add |
| **SAM3** | facebook/sam3 | — | segmentation (text-prompted) | ✅ Integrated |
| **MediaPipe** | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | 33k+ | face, hands, pose, holistic | Partial (legacy) |
| **Grounded SAM2** | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | 3.2k+ | detection + segmentation | ✅ Integrated |
| **RF-DETR** | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | 5.5k+ | detection + segmentation | Add |
| **Depth Anything** | [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything) | 7.9k+ | depth map | Add |
| **Sapiens** (Meta) | [facebookresearch/sapiens](https://github.com/facebookresearch/sapiens) | 5.2k+ | pose, body seg, depth | Add |
| **LiveKit Agents** | [livekit/livekit-agents](https://github.com/livekit/livekit-agents) | — | video/agent pipeline | Optional integration |

**Note on LiveKit:** LiveKit is a real-time video/agent platform, not a standalone CV model. It can host vision pipelines (e.g., with Gemini Live). Treat as optional for streaming/agent integration rather than a drop-in model.

---

## 2. Output-Type Taxonomy

Map each model into a small set of output types the pipeline can handle:

| Output Type | Description | Unity Rendering | Example Models |
|-------------|-------------|-----------------|----------------|
| `segmentation` | Mask per object | Blur/dim/pixelate on mask | SAM3, Grounded SAM2, RF-DETR, FastSAM |
| `detection` | Bbox + label, no mask | Bbox outline or rect mask | YOLO (detect), RF-DETR (detect-only) |
| `keypoints` | Landmarks (x,y) | Skeleton / points overlay | MediaPipe, YOLO-Pose, Sapiens |
| `depth` | Per-pixel depth | Depth colormap or depth bands | Depth Anything, MiDaS, Sapiens |
| `classification` | Scene-level label | Text HUD | (Future: CLIP, etc.) |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Server (Python)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  CV Model Registry                                                       │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ SAM3     │ YOLO     │ MediaPipe│ RF-DETR  │ Depth    │ Sapiens  │   │
│  │          │          │          │          │ Anything │          │   │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘   │
│       │          │          │          │          │          │          │
│       └──────────┴──────────┴────┬─────┴──────────┴──────────┘          │
│                                  ▼                                      │
│  Output Adapters (convert to canonical format)                           │
│  - Segmentation → SegmentData (existing)                                 │
│  - Detection    → SegmentData (bbox → rect mask)                         │
│  - Keypoints    → KeypointOverlay                                        │
│  - Depth        → DepthOverlay                                           │
│                                  │                                       │
│                                  ▼                                       │
│  Unified Pipeline Output                                                │
│  - segments[] (mask-based, for effects)                                  │
│  - keypoints[] (optional)                                                │
│  - depth_map (optional)                                                  │
│  - full_screen_filter (existing)                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                            WebSocket + Protobuf
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Unity Client (Quest 3)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Renderers:                                                              │
│  - MaskOverlayRenderer (existing: blur, dim, pixelate, highlight)        │
│  - BboxOverlayRenderer (new: detection boxes)                            │
│  - KeypointOverlayRenderer (new: pose/face/hand skeletons)               │
│  - DepthOverlayRenderer (new: depth colormap or bands)                   │
│  - TextHUD (existing: labels, voice agent)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Phases

### Phase 1: Abstractions & Adapters (No Protocol Change)

**Goal:** Define a common interface and adapters so new models produce `SegmentData` (or equivalent) without changing the wire format.

1. **Define `CVModelBase` interface**
   - `initialize()`, `process(frame_bgr) -> RawOutput`, `shutdown()`
   - `output_type: Literal["segmentation","detection","keypoints","depth"]`
   - `model_id: str` for config/logging

2. **Implement output adapters**
   - `SegmentationAdapter`: pass-through (current behavior)
   - `DetectionAdapter`: bbox → rectangular mask → `SegmentData`
   - `KeypointsAdapter`: landmarks → optional convex-hull mask or skip mask (keypoints only)
   - `DepthAdapter`: depth map → band masks (e.g., near/mid/far) → `SegmentData` or new `DepthPayload`

3. **Add YOLO segmenter**
   - Use `ultralytics` for detection or instance segmentation
   - Map to `SegmentData` via `DetectionAdapter` or `SegmentationAdapter`
   - Config: `--models yolo` or `pipeline_mode: yolo`

4. **Refactor MediaPipe**
   - Extract standalone MediaPipe module (face, hands, pose)
   - Output: keypoints; adapter: optional body-part masks or skeleton-only
   - Enable as `--models mediapipe` alongside SAM3 or standalone

**Deliverables:** YOLO and MediaPipe working via adapters, same protobuf, Unity unchanged (may show boxes as rect masks).

---

### Phase 2: Extensible Protocol

**Goal:** Extend protobuf to support keypoints and depth explicitly, so Unity can render them natively.

1. **Extend `socialsense.proto`**
   ```protobuf
   message KeypointOverlay {
     string model_id = 1;           // "mediapipe_pose", "yolo_pose"
     repeated KeypointGroup groups = 2;
   }
   message KeypointGroup {
     string label = 1;              // "body", "face", "left_hand"
     repeated Keypoint points = 2;
     repeated int32 skeleton_edges = 3;  // indices into points
   }
   message Keypoint {
     float x = 1; float y = 2;
     float visibility = 3;          // 0-1
     string name = 4;
   }

   message DepthOverlay {
     string model_id = 1;
     bytes depth_rgb = 2;           // RGB colormap (JPEG or raw)
     int32 width = 3; int32 height = 4;
     float near_m = 5; float far_m = 6;  // optional metric hints
   }

   message ServerMessage {
     // ... existing segments, conversation, etc.
     repeated KeypointOverlay keypoints = 20;
     optional DepthOverlay depth = 21;
   }
   ```

2. **Regenerate Python and C# from proto**
   - Update `server/proto/socialsense_pb2.py`
   - Update `unity-client/Assets/Scripts/Proto/SocialsenseMessages.cs`

3. **Server: populate keypoints/depth when models active**
   - Pipeline checks `config.active_models`
   - If MediaPipe/YOLO-Pose: fill `keypoints`
   - If Depth Anything: fill `depth` (e.g., JPEG-encoded colormap)

4. **Unity: add `KeypointOverlayRenderer` and `DepthOverlayRenderer`**
   - Keypoints: draw lines between skeleton edges, circles at points
   - Depth: render texture from `depth_rgb` (overlay or side panel)

**Deliverables:** Keypoints and depth displayed in Unity; MediaPipe pose/face/hands and Depth Anything fully supported.

---

### Phase 3: Model Registry & Multi-Model Pipeline

**Goal:** Run multiple models per frame, selectable via config.

1. **Model registry**
   ```python
   # server/vision/registry.py
   CV_MODELS = {
       "sam3": (SAM3Segmenter, "segmentation"),
       "grounded-sam2": (GroundedSAM2Segmenter, "segmentation"),
       "yolo": (YOLOSegmenter, "segmentation"),      # or detection
       "rf-detr": (RFDETRSegmenter, "segmentation"),
       "mediapipe": (MediaPipeDetector, "keypoints"),
       "depth-anything": (DepthAnythingEstimator, "depth"),
       "sapiens": (SapiensEstimator, "keypoints"),   # or segmentation
   }
   ```

2. **Config**
   ```yaml
   # or CLI
   active_models: ["sam3", "mediapipe"]   # primary + supplementary
   primary_model: "sam3"                  # for voice effects (blur/dim)
   ```

3. **Pipeline changes**
   - Orchestrator instantiates models from registry based on `active_models`
   - Primary model: produces `segments` for effects
   - Supplementary: produce `keypoints` or `depth` only
   - Merge outputs into single `ServerMessage`

4. **Add RF-DETR segmenter**
   - `pip install rfdetr`
   - Implement `RFDETRSegmenter` with `segment_frame() -> list[SegmentData]`
   - Register in `CV_MODELS`

5. **Add Depth Anything**
   - Implement `DepthAnythingEstimator`
   - Output: depth map → colormap JPEG
   - Populate `ServerMessage.depth`

6. **Add Sapiens (optional)**
   - Human-centric: pose, body seg, depth
   - Output: keypoints + optional masks
   - Register and wire to adapters

**Deliverables:** Multi-model selection, RF-DETR, Depth Anything, Sapiens integrated; config-driven activation.

---

### Phase 4: Unity Renderers & UX

**Goal:** Polished rendering and user controls for each output type.

1. **BboxOverlayRenderer**
   - Draw rectangles for detection-only outputs
   - Toggle: outline vs filled (for dim/blur on rect)

2. **KeypointOverlayRenderer**
   - MediaPipe: pose (33 pts), face (468), hands (21 each)
   - YOLO-Pose: COCO keypoints
   - Sapiens: body + face
   - Toggle per group (show/hide pose, face, hands)

3. **DepthOverlayRenderer**
   - Full overlay (semi-transparent colormap) or side panel
   - Optional depth bands as effect regions (e.g., dim far plane)

4. **Model selector in app**
   - In-device UI or web dashboard to switch `active_models`
   - Or keep config-file/CLI only for v1

**Deliverables:** All output types render cleanly; user can enable/disable per model.

---

## 5. File Structure (New/Modified)

```
server/
  vision/
    base.py              # CVModelBase, OutputAdapter
    adapters.py          # SegmentationAdapter, DetectionAdapter, KeypointsAdapter, DepthAdapter
    registry.py          # CV_MODELS, load_models()
    yolo_segmenter.py    # NEW
    rfdetr_segmenter.py  # NEW
    mediapipe_detector.py # REFACTOR (extract from legacy)
    depth_anything.py    # NEW
    sapiens_estimator.py # NEW (optional)
  pipeline/
    orchestrator.py      # MODIFY: multi-model, adapters, keypoints/depth
  proto/
    socialsense.proto    # EXTEND: KeypointOverlay, DepthOverlay
    socialsense_pb2.py   # REGENERATE

unity-client/
  Assets/Scripts/
    KeypointOverlayRenderer.cs   # NEW
    DepthOverlayRenderer.cs      # NEW
    BboxOverlayRenderer.cs       # NEW (or extend OverlayRenderer)
    OverlayRenderer.cs           # MODIFY: dispatch by output type
    Proto/SocialsenseMessages.cs # REGENERATE

config/
  cv_models.yaml         # NEW: model configs (paths, thresholds)
```

---

## 6. Model Integration Checklist (Per Model)

For each new model:

- [ ] Implement `XSegmenter` / `XEstimator` with `segment_frame(frame_bgr) -> RawOutput`
- [ ] Choose adapter: Segmentation / Detection / Keypoints / Depth
- [ ] Add to `registry.py` with `output_type`
- [ ] Add config entries in `config.py` (model path, thresholds)
- [ ] Add to `requirements.txt` or optional deps
- [ ] Document in README (install, run, flags)
- [ ] Add Unity renderer if new output type (keypoints, depth)
- [ ] Test with `test_client` (webcam) before Quest

---

## 7. Dependency Summary

| Model | Package | Notes |
|-------|---------|-------|
| YOLO | `ultralytics` | pip install ultralytics |
| SAM3 | `transformers`, `torch` | Already present |
| MediaPipe | `mediapipe` | Already present |
| Grounded SAM2 | custom | Already present |
| RF-DETR | `rfdetr` | pip install rfdetr |
| Depth Anything | `depth-anything` or clone repo | May need `timm`, `torch` |
| Sapiens | `sapiens` or clone | Meta HF, check license |

---

## 8. Suggested Order of Implementation

1. **Phase 1.1–1.2:** `CVModelBase` + adapters (DetectionAdapter for bbox→mask)
2. **Phase 1.3:** YOLO segmenter (easiest: ultralytics API)
3. **Phase 1.4:** MediaPipe refactor + keypoints adapter
4. **Phase 2:** Proto extension + keypoints/depth in Unity
5. **Phase 3.1–3.2:** Registry + multi-model pipeline
6. **Phase 3.4–3.6:** RF-DETR, Depth Anything, Sapiens
7. **Phase 4:** Polish Unity renderers and UX

---

## 9. Risk & Mitigations

| Risk | Mitigation |
|------|------------|
| Proto changes break Unity | Version protobuf, maintain backward compat (optional fields) |
| GPU memory with multi-model | Run models sequentially; optional “lite” variants |
| Latency with multiple models | Run supplementary models at lower FPS (e.g., pose at 15 Hz) |
| Depth map size | Compress as JPEG colormap; optional downscale |
| License compatibility | Check each model (SAM3 gated, Sapiens research, etc.) |

---

## 10. Success Criteria

- [ ] User can run `--models sam3,yolo,mediapipe` and see segmentation + boxes + pose
- [ ] User can run `--models depth-anything` and see depth overlay in Unity
- [ ] New model = new file + registry entry + adapter (no core pipeline rewrite)
- [ ] Unity renderers handle segmentation, detection, keypoints, depth without coupling to specific model
