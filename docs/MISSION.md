# Mission Definition - Keyframe-Based Visual Servoing Drone PoC

## Goal

Build a lightweight autonomous navigation Proof-of-Concept for a stabilized drone in a provided Unity-based indoor environment.

The Unity build, NDI camera stream, TCP communication, keyboard control, and low-level drone control interface are already provided by the existing script.

Therefore, this project should **not** rebuild the simulator, camera pipeline, or low-level drone interface.

Instead, the project should add an autonomy layer that uses:

- Existing Unity drone build
- Existing NDI camera frames
- Existing `control_state` command interface
- Manually recorded reference keyframes
- CPU-based ORB feature matching
- Feature-based visual error estimation
- Closed-loop visual servoing control
- Optional YOLO target detection override

The goal is not to build a general-purpose autonomous drone.

The goal is to demonstrate that the drone can move through the provided environment by visually converging toward a sequence of reference keyframes.

---

## Core Idea

The system navigates by minimizing the visual error between:

```text
current camera frame
```

and:

```text
active reference keyframe
```

Instead of:

```text
match keyframe -> execute fixed action
```

the system performs:

```text
match features -> estimate visual error -> generate correction command -> update control_state -> minimize error
```

This is a closed-loop visual servoing approach.

The existing Unity communication loop continues to send `control_state` to the drone at 60Hz.

---

## Provided Components

The provided script already includes:

- Unity build interface
- TCP communication with Unity
- NDI camera video stream
- Manual keyboard control
- Global `control_state` dictionary
- Command sending to Unity at 60Hz
- Basic YOLO inference support
- Basic YOLO-based autopilot example

These components should be reused.

The new autonomy stack should only consume the existing camera frames and update the existing `control_state`.

---

## Two Phases

## 1. Manual Keyframe Recording Phase

The operator manually flies the drone through the environment using the existing controls.

During this phase, the system saves representative reference keyframes from the existing NDI video stream.

A keyframe should be saved when the view changes significantly, for example:

- Entrance to corridor
- Middle of corridor
- End of corridor
- Before turn
- After turn
- Near doorway
- Search area
- Target region

Each keyframe represents a desired visual state, not a hardcoded movement command.

Example metadata:

```json
{
  "id": 1,
  "name": "corridor_start",
  "image_path": "data/keyframes/kf_001.png",
  "sequence_index": 0,
  "motion_prior": "forward",
  "notes": "motion_prior is used only for recovery/debug, not for main control"
}
```

### Important Note About `motion_prior`

`motion_prior` is optional.

It is **not** the command used to move the drone during normal operation.

The actual movement is computed from the visual servoing error.

`motion_prior` may only be used for:

- Recovery behavior when matching fails
- Debugging
- Logging
- Preventing obviously wrong fallback motion

Normal navigation should be based on:

```text
visual error -> control command
```

not:

```text
keyframe -> fixed action
```

---

## 2. Autonomous Visual Servoing Phase

During autonomous navigation:

1. Use the existing NDI stream to get the current frame.
2. Load the ordered keyframe sequence.
3. Select the active reference keyframe.
4. Convert the current frame and reference keyframe to grayscale for ORB feature extraction.
5. Match ORB features between the current frame and the reference keyframe.
6. Optionally validate the matches using Homography + RANSAC.
7. Estimate visual error:
   - `dx`
   - `dy`
   - `scale_error`
   - optional `rotation_error`
8. Convert the visual error into normalized drone commands:
   - `forward`
   - `strafe`
   - `yaw`
   - `altitude`
9. Use a `DroneControlAdapter` to update the existing `control_state`.
10. Repeat until the visual error is below a convergence threshold for several consecutive frames.
11. Advance to the next keyframe.
12. If the target object is detected, override keyframe servoing and approach the target.

---

## Visual Servoing Error

The visual error should be computed from matched ORB keypoints.

Basic image-plane displacement:

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

Normalized error:

```python
dx_norm = dx / image_width
dy_norm = dy / image_height
```

Overall error:

```python
error_norm = sqrt(dx_norm**2 + dy_norm**2 + scale_error**2)
```

The drone should move in a direction that reduces this error.

---

## Control Mapping

The visual servo controller should convert visual error into normalized control commands.

Example mapping:

```python
yaw = K_yaw * dx_norm
altitude = K_altitude * dy_norm
forward = K_forward * scale_error
```

The output command should use a normalized format:

```python
command = {
    "forward": float,
    "strafe": float,
    "yaw": float,
    "altitude": float,
    "duration": float
}
```

The command should then be passed to a `DroneControlAdapter`, which updates the existing `control_state`.

Example mapping to `control_state`:

```python
if command["forward"] >= 0:
    control_state["trigger"] = command["forward"]
    control_state["reverse"] = 0.0
else:
    control_state["trigger"] = 0.0
    control_state["reverse"] = abs(command["forward"])

control_state["yaw"] = command["yaw"]
control_state["joy_horizontal"] = command["strafe"]
control_state["joy_vertical"] = command["altitude"]
```

All values should be clamped to the valid range used by the Unity interface.

---

## Target Detection Override

The provided script already includes YOLO inference support.

If a target detector is enabled and the target object is detected, it should override keyframe visual servoing.

Priority:

```text
target detected -> align to target -> approach target -> stop
```

Target alignment can use image-center error:

```python
target_error_x = (target_center_x - image_center_x) / image_center_x
target_error_y = (target_center_y - image_center_y) / image_center_y
```

Example behavior:

- If target is left/right of image center, correct yaw or strafe.
- If target is above/below image center, correct altitude or pitch depending on available controls.
- If target is centered and large enough in the image, stop and report success.

---

## Success Criteria

## Minimum Success

The system should be able to:

- Save reference keyframes manually from the existing NDI camera stream.
- Load an ordered keyframe sequence.
- Select the active reference keyframe.
- Match the current frame to the active keyframe using ORB on CPU.
- Estimate `dx`, `dy`, and `scale_error`.
- Generate movement commands that reduce visual error.
- Update the existing `control_state` through a `DroneControlAdapter`.
- Advance to the next keyframe once convergence is reached.
- Log visual errors, commands, and keyframe transitions.

## Better Success

The system can be improved by adding:

- Homography + RANSAC validation to reject wrong matches.
- Confidence threshold and recovery behavior.
- P/PD-like control for smoother convergence.
- Target detector override.
- Stop behavior near the detected target.
- Final screenshot and mission summary.

---

## Recovery Behavior

If visual matching is unreliable, the drone should not keep moving forward.

Recovery should be triggered when:

- Too few ORB matches are found.
- Too few RANSAC inliers are found.
- Inlier ratio is too low.
- Visual error is too large.
- The active keyframe cannot be matched.

Recommended recovery behavior:

```text
stop forward motion
yaw slowly left/right
try to reacquire matches
use motion_prior only as weak fallback
```

The recovery policy should prioritize safety and stability.

---

## Logging and Debugging

The system should log each autonomy step.

Recommended log fields:

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
command_strafe
command_altitude
target_found
target_confidence
```

Optional debug outputs:

- Current frame
- Reference keyframe
- ORB feature matches visualization
- Error vector visualization
- Final screenshot
- Mission summary JSON

---

## Out of Scope

Do not implement:

- New Unity simulator
- New camera/video pipeline
- New low-level socket control protocol
- ROS
- Nav2
- Cartographer
- DROID-SLAM
- ORB-SLAM3
- Full metric localization
- Full 3D mapping
- Frontier exploration
- GPU-only models as required dependencies

---

## Final System Summary

The system reuses the provided Unity drone interface and NDI camera stream.

The new autonomy layer receives the current frame, compares it to the active reference keyframe using ORB feature matching, estimates visual error, and updates the existing `control_state` to reduce that error.

In short:

```text
NDI frame -> ORB matching -> visual error -> servo command -> control_state -> Unity drone motion
```

This provides a practical closed-loop autonomous navigation PoC without ROS, SLAM, or GPU dependency.

