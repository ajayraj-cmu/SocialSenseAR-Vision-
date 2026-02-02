# DEFERRED: Add Object Labels in Unity

## Status: On hold — fix mask stability and stereo alignment first

## Approach
Add floating 3D text labels (TextMeshPro) positioned at each segment's center on the overlay sphere. Labels are world-locked using the same capture rotation as the masks.

## Files to modify
- `QuestCameraKit/Unity-QuestVisionKit/Assets/Scripts/OverlayRenderer.cs` — all changes here

## Implementation

### 1. Cache intrinsics for inverse projection
In `Update()`, after computing `adjustedCx`, `adjustedCy`, focal length, etc., cache them into fields so `UpdateLabels()` can use them:
- `_cachedFocalLength`, `_cachedPrincipalPoint` (left-eye), `_cachedActualW/H`
- `_cachedCameraProjectionRot` (the `cameraProjectionRot` quaternion)

### 2. Label pool
- `List<GameObject> _labels` — pooled label objects
- Each label: `GameObject` with `TextMeshPro` (3D, not UGUI — no Canvas needed)
- Create on demand, reuse existing, hide unused

### 3. UpdateLabels() — called from UpdateOverlay()
For each segment with a non-empty label:
1. Convert `(centerX, centerY)` normalized [0,1] to pixel coords: `px = centerX * actualW`, `py = centerY * actualH`
2. Inverse pinhole projection to camera-local direction:
   ```
   dx = (px - adjustedCx) / focalX
   dy = (py - adjustedCy) / focalY
   localDir = normalize(dx, dy, 1.0)
   ```
3. Camera-local to world direction: `worldDir = (headRot * cameraTilt) * localDir`
   where `headRot = _captureRotation` (world-locked)
4. Position: `centerEyeAnchor.position + worldDir * (sphereRadius * 0.95)` (slightly in front of sphere)
5. Billboard: face the user's eye
6. Color: match BrightColors[segIndex]

### 4. Label styling
- Font size / character size tuned for readability at 1m
- Semi-transparent dark background (optional)
- Matching segment outline color

### 5. Cleanup
- `OnDestroy()`: destroy pooled label objects
- `ClearOverlay()`: hide all labels
