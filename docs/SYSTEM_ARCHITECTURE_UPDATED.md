# System Architecture - Keyframe-Based Visual Servoing with Existing Unity Drone Interface

## Overview

This project implements a lightweight autonomous navigation Proof-of-Concept for a stabilized drone in a Unity-based indoor environment.

The provided task already includes a working drone interface script that handles:

- TCP communication with the Unity drone build
- Sending `control_state` to Unity at 60Hz
- Receiving Unity telemetry
- NDI video stream capture
- Manual keyboard control
- Basic YOLO-based autopilot example

Therefore, this project should **not** reimplement low-level drone communication or keyboard control from scratch.

Instead, the autonomy stack should be implemented as an additional layer that receives camera frames, computes visual servoing commands, and updates the existing `control_state` through a clean adapter.

---

## Main Architecture

```text
Unity Drone Build
        ↓
Existing TCP / NDI Interface Script
        ↓
Camera Frame from NDI
        ↓
Autonomy Stack
        ↓
Keyframe-Based Visual Servoing
        ↓
Drone Control Adapter
        ↓
Existing control_state Dictionary
        ↓
send_to_unity() at 60Hz
        ↓
Unity Drone Motion
```

---

## Core Principle

The drone does not need full metric localization or SLAM.

Instead, the drone navigates by minimizing visual error between:

```text
current camera frame
```

and:

```text
active reference keyframe
```

Each saved keyframe is treated as a visual goal.

The system performs:

```text
ORB feature matching
        ↓
Visual error estimation
        ↓
Control command generation
        ↓
Update control_state
        ↓
Drone moves to reduce visual error
```

This is a closed-loop visual servoing approach.

---

## Important Existing Components

The provided interface script already includes the following important components.

### 1. Existing `control_state`

The script maintains a global control dictionary that represents the command state sent to Unity:

```python
control_state = {
    "trigger": 0.0,
    "reverse": 0.0,
    "joy_vertical": 0,
    "joy_horizontal": 0,
    "yaw": 0.0,
    "pitch": 0.0,
    "autopilot": False
}
```

Important fields for autonomy:

| Field | Meaning |
|---|---|
| `trigger` | Forward motion |
| `reverse` | Backward motion |
| `joy_horizontal` | Strafe left/right |
| `joy_vertical` | Altitude up/down |
| `yaw` | Rotate left/right |
| `pitch` | Camera tilt |
| `autopilot` | Disables manual keyboard ramping while autonomy is active |

---

### 2. Existing Unity Communication

The existing script already sends the current `control_state` to Unity inside `send_to_unity()` at approximately 60Hz.

Therefore, the autonomy stack only needs to update `control_state`.

It does not need to open a new socket or create a new Unity communication protocol.

---

### 3. Existing Frame Capture

The existing script already receives video frames from Unity using NDI.

The autonomy stack should reuse these frames:

```python
frame = np.copy(v.data)
bgr_frame = prepare_frame_for_yolo(frame)
```

The same frame can be used for:

- ORB feature matching
- Visual servoing
- YOLO target detection
- Debug visualization

---

### 4. Existing YOLO Autopilot Example

The provided script already contains a basic autopilot mode that:

1. Runs YOLO on the current frame.
2. Finds a detected target.
3. Computes image error from the center of the frame.
4. Updates `yaw`, `pitch`, and `trigger`.

This should be treated as a useful example of how autonomy should update `control_state`.

The new visual servoing system generalizes this idea from:

```text
YOLO target center error
```

to:

```text
ORB keypoint error relative to reference keyframe
```

---

## Updated Module Breakdown

## 1. Existing Unity Drone Interface

Status: already provided.

Responsibilities already handled:

- TCP server setup
- Unity connection
- Sending control packets
- Receiving telemetry
- Keyboard manual control
- NDI video streaming
- Basic YOLO inference
- Basic autopilot example

This module should remain mostly unchanged.

Only minimal integration points should be added.

---

## 2. Autonomy Stack

New module.

Responsible for high-level autonomous behavior.

Input:

```python
frame: np.ndarray
```

Output:

```python
command: dict
debug_info: dict
```

Expected interface:

```python
class AutonomyStack:
    def step(self, frame) -> tuple[dict, dict]:
        pass
```

The `step()` function should:

1. Check target detector.
2. If target is detected, generate target servoing command.
3. Otherwise, get active reference keyframe.
4. Match ORB features.
5. Estimate visual error.
6. Generate visual servoing command.
7. Return command and debug information.

---

## 3. Keyframe Recorder

Used during manual flight.

Responsibilities:

- Save current NDI frame as a reference keyframe.
- Store metadata in `data/keyframes/keyframes.json`.
- Preserve the order of the route.

Recommended keyframe metadata:

```json
{
  "id": 1,
  "name": "corridor_start",
  "image_path": "data/keyframes/kf_001.png",
  "sequence_index": 0,
  "motion_prior": "forward",
  "notes": "Used only for recovery/debug, not main control"
}
```

Important:

`motion_prior` is optional.  
It is not the main movement command.  
The actual motion is computed by visual servoing.

---

## 4. Keyframe Manager

Responsible for the ordered keyframe sequence.

Responsibilities:

- Load keyframe metadata.
- Load keyframe images.
- Track active keyframe index.
- Return the active reference keyframe.
- Advance to the next keyframe after convergence.
- Report mission completion after the final keyframe.

Expected interface:

```python
class KeyframeManager:
    def get_active_keyframe(self) -> dict:
        pass

    def advance(self) -> None:
        pass

    def is_complete(self) -> bool:
        pass
```

---

## 5. ORB Feature Matcher

Responsible for CPU-only feature matching.

Recommended implementation:

```python
orb = cv2.ORB_create(nfeatures=1000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
```

Matching process:

1. Convert current frame and reference keyframe to grayscale.
2. Extract ORB keypoints and descriptors.
3. Match descriptors using Hamming distance.
4. Filter matches by distance.
5. Optionally validate matches using Homography + RANSAC.
6. Return only reliable inlier matches.

Expected output:

```python
{
    "valid": bool,
    "current_points": np.ndarray,
    "reference_points": np.ndarray,
    "matches": list,
    "num_matches": int,
    "homography": np.ndarray | None,
    "inlier_mask": np.ndarray | None,
    "num_inliers": int,
    "inlier_ratio": float
}
```

---

## 6. Visual Error Estimator

Responsible for converting matched keypoints into visual errors.

Basic image-plane error:

```python
dx = mean(reference_x - current_x)
dy = mean(reference_y - current_y)
```

Scale error:

```python
spread_current = mean_distance(current_points, current_centroid)
spread_reference = mean_distance(reference_points, reference_centroid)

scale_error = spread_reference / spread_current - 1.0
```

Normalized errors:

```python
dx_norm = dx / image_width
dy_norm = dy / image_height
```

Overall error:

```python
error_norm = sqrt(dx_norm**2 + dy_norm**2 + scale_error**2)
```

Expected output:

```python
{
    "valid": bool,
    "dx": float,
    "dy": float,
    "dx_norm": float,
    "dy_norm": float,
    "scale_error": float,
    "error_norm": float,
    "num_inliers": int,
    "inlier_ratio": float,
    "reason": str
}
```

---

## 7. Visual Servo Controller

Responsible for converting visual errors into normalized drone commands.

Input:

```python
visual_error: dict
```

Output:

```python
command = {
    "forward": float,
    "strafe": float,
    "yaw": float,
    "altitude": float,
    "duration": float
}
```

Recommended control mapping:

```python
yaw = K_yaw * dx_norm
altitude = K_altitude * dy_norm
forward = K_forward * scale_error
```

The controller should clamp all command values:

```python
yaw = clamp(-1, yaw, 1)
altitude = clamp(-1, altitude, 1)
forward = clamp(-1, forward, 1)
```

Use short command durations:

```text
0.1 to 0.3 seconds
```

This keeps the control loop closed and responsive.

---

## 8. Drone Control Adapter

This is the bridge between the new autonomy stack and the existing script.

It should not send keyboard events.  
It should update the existing `control_state` directly.

Input command format:

```python
command = {
    "forward": 0.3,
    "strafe": 0.0,
    "yaw": -0.2,
    "altitude": 0.0,
    "duration": 0.2
}
```

Adapter behavior:

```python
def apply_servo_command(command):
    control_state["autopilot"] = True

    forward = command.get("forward", 0.0)
    yaw = command.get("yaw", 0.0)
    strafe = command.get("strafe", 0.0)
    altitude = command.get("altitude", 0.0)

    if forward >= 0:
        control_state["trigger"] = clamp(0, forward, 1)
        control_state["reverse"] = 0.0
    else:
        control_state["trigger"] = 0.0
        control_state["reverse"] = clamp(0, -forward, 1)

    control_state["yaw"] = clamp(-1, yaw, 1)
    control_state["joy_horizontal"] = clamp(-1, strafe, 1)
    control_state["joy_vertical"] = clamp(-1, altitude, 1)
```

Stop behavior:

```python
def stop_drone():
    control_state["trigger"] = 0.0
    control_state["reverse"] = 0.0
    control_state["yaw"] = 0.0
    control_state["pitch"] = 0.0
    control_state["joy_horizontal"] = 0
    control_state["joy_vertical"] = 0
```

---

## 9. Navigation FSM

Responsible for mission-level decision-making.

Recommended states:

```text
INIT
MANUAL_RECORDING
LOAD_KEYFRAMES
SERVO_TO_KEYFRAME
KEYFRAME_REACHED
NEXT_KEYFRAME
TARGET_DETECTED
ALIGN_TO_TARGET
APPROACH_TARGET
TARGET_REACHED
RECOVERY
MISSION_COMPLETE
```

Main logic:

```text
If target detected:
    target servoing override
Else:
    servo toward active keyframe

If visual error is valid:
    apply visual servoing command
Else:
    run recovery policy

If error_norm < threshold for N consecutive frames:
    advance to next keyframe
```

Use a convergence counter:

```python
if visual_error["error_norm"] < convergence_threshold:
    stable_count += 1
else:
    stable_count = 0

if stable_count >= required_stable_frames:
    keyframe_manager.advance()
```

---

## 10. Recovery Policy

Used when visual matching is unreliable.

Recovery triggers:

- Too few matches
- Too few RANSAC inliers
- Low inlier ratio
- Very high visual error
- Lost active keyframe

Recommended behavior:

```text
stop forward motion
yaw slowly left/right
try to reacquire matches
use motion_prior only as weak fallback
```

Important:

Recovery should avoid aggressive forward movement.

---

## 11. Target Detection Override

The existing script already includes YOLO inference.

The updated autonomy stack can reuse this or replace it with a fine-tuned target detector.

Priority:

```text
Target detected -> ignore keyframe servoing -> align to target -> approach -> stop
```

Target servoing:

```python
target_error_x = (target_center_x - image_center_x) / image_center_x
target_error_y = (target_center_y - image_center_y) / image_center_y

yaw = K_target_yaw * target_error_x
altitude = K_target_altitude * target_error_y
forward = K_target_forward
```

Stop condition:

```text
target centered
+
bbox_area_ratio above threshold
```

---

## 12. Logger and Debug Visualization

Log every step:

```text
timestamp
state
active_keyframe
num_matches
num_inliers
inlier_ratio
dx
dy
scale_error
error_norm
command_forward
command_yaw
command_altitude
target_found
target_confidence
```

Optional debug outputs:

- Draw ORB matches.
- Draw average error vector.
- Save current/reference side-by-side image.
- Save final screenshot.
- Save mission summary JSON.

---

## Integration Point in Existing Main Loop

Inside the existing NDI video loop, after the frame is copied and converted:

```python
frame = np.copy(v.data)
bgr_frame = prepare_frame_for_yolo(frame)
```

Add:

```python
if autonomy_enabled:
    command, debug_info = autonomy_stack.step(bgr_frame)
    apply_servo_command(command)
```

This integrates autonomy without changing the existing Unity communication thread.

---

## Final Runtime Flow

```text
Unity sends video through NDI
        ↓
Existing script receives frame
        ↓
AutonomyStack.step(frame)
        ↓
Target detector check
        ↓
If no target:
    active keyframe selected
    ORB matching performed
    visual error estimated
    visual servo command generated
        ↓
DroneControlAdapter updates control_state
        ↓
Existing send_to_unity() sends control_state to Unity
        ↓
Drone moves
        ↓
Repeat until keyframe convergence
        ↓
Advance to next keyframe
```

---

## Why This Architecture Fits the Task

This architecture is appropriate because:

- It reuses the provided Unity drone interface.
- It does not rebuild low-level control.
- It does not require ROS.
- It does not require SLAM.
- It works without GPU.
- It uses keyframes as visual goals.
- It performs closed-loop control rather than hardcoded movement.
- It is practical for a PoC in a known environment.
