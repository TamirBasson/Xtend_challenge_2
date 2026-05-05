# Test Plan - Keyframe-Based Visual Servoing

## Test 1 - Keyframe Recording

Run:

```bash
python scripts/record_keyframes.py --config config/mission_config.yaml
```

Expected:

- Images are saved.
- JSON metadata is updated.
- Keyframes have sequence indexes.
- Keyframes preserve path order.

---

## Test 2 - Offline ORB Matching

Run:

```bash
python scripts/test_matching.py --current path/to/current.png --reference path/to/reference.png
```

Expected:

- Number of matches is printed.
- Number of RANSAC inliers is printed.
- Inlier ratio is printed.
- Optional debug image shows matched features.

Pass criteria:

```text
num_inliers >= minimum_inliers
inlier_ratio >= minimum_inlier_ratio
```

---

## Test 3 - Visual Error Direction

Use a reference image and a shifted current image.

Expected:

- If reference appears to the right of current, dx should be positive.
- If reference appears to the left of current, dx should be negative.
- If reference appears lower than current, dy should be positive.
- If reference appears higher than current, dy should be negative.

---

## Test 4 - Scale Error

If current image is farther away than the reference:

```text
scale_error > 0
```

This should generate positive forward motion.

If current image is closer than reference:

```text
scale_error < 0
```

This should generate stop or backward motion depending on configuration.

---

## Test 5 - Controller Output

Feed synthetic errors:

```python
dx_norm = 0.1
dy_norm = -0.05
scale_error = 0.2
```

Expected:

- Positive dx generates yaw or strafe in the configured positive direction.
- dy generates altitude correction.
- scale_error generates forward correction.
- Commands are clamped to max values.

---

## Test 6 - Keyframe Convergence

Expected:

- It does not advance after only one frame.
- It advances after required stable frames.
- It loads the next keyframe.

---

## Test 7 - Low Confidence Recovery

Expected:

- No strong forward movement.
- Recovery policy is selected.
- The event is logged.

---

## Test 8 - Target Detection Override

Expected:

- Keyframe servoing is ignored.
- Target centering command is generated.
- Drone stops when target area ratio exceeds threshold.

---

## Test 9 - Full Autonomous Demo

Run:

```bash
python scripts/run_navigation.py --config config/mission_config.yaml
```

Expected:

- System loads keyframes.
- Active keyframe is selected.
- Visual error is computed.
- Drone moves to reduce error.
- System advances through keyframes.
- Logs show the error decreasing over time.

---

## Test 10 - Live Visual Debug Overlay

Run:

```bash
python scripts/run_navigation.py --config config/mission_config.yaml --debug-video
```

Expected:

The live camera video should display all visual servoing information directly on the frame.

1. Keypoints Detection

The video should show:

ORB keypoints detected in the current camera frame
ORB keypoints from the active reference keyframe
Matched keypoints between current frame and reference keyframe
RANSAC inlier matches highlighted separately from rejected matches

Visual convention:

green points = detected keypoints
green lines = valid inlier matches
red lines = rejected/outlier matches

Active Keyframe Visualization

Overlay text:

Active keyframe: 003
Keyframe progress: 3 / N

Optional:

Small thumbnail of the active reference keyframe in a corner
Current frame remains the main view
3. Error Estimation Visualization

Overlay:

dx_norm: 0.12
dy_norm: -0.05
scale_error: 0.18
error_norm: 0.21

Also draw an error arrow from image center:

Arrow right = positive dx
Arrow left = negative dx
Arrow down = positive dy
Arrow up = negative dy
4. Movement Command Visualization

Overlay:

forward: 0.25
yaw: 0.10
strafe: 0.00
altitude: -0.05

Optional direction hint:

FORWARD
YAW RIGHT
ALTITUDE DOWN
5. Confidence and State Display

Overlay:

state: SERVO_TO_KEYFRAME
num_matches: 84
num_inliers: 42
inlier_ratio: 0.50
confidence: HIGH

Low confidence case:

state: RECOVERY
confidence: LOW
forward command blocked
6. Target Detection Override

When target is detected:

state: TARGET_TRACKING
target_found: true
target_confidence: 0.87
target_area_ratio: 0.14

Overlay should include:

Bounding box
Target center point
Image center point
Error arrow from image center to target

Behavior:

Keyframe servoing is ignored
Target centering command is used
Drone stops when target area ratio exceeds threshold