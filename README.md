# Keyframe Visual Servoing for Autonomous Drone Navigation

## Quick Start

Run autonomous navigation with debug visualization:

```bash
python scripts/run_navigation.py --config config/mission_config.yaml --debug-video
```

---

## 1. Project Overview

This project implements an autonomous drone navigation pipeline based on **keyframe visual servoing** in a simulation environment.  
Its main goal is to guide a drone along a previously demonstrated route using camera feedback and image matching.

The system uses:

- A manually recorded sequence of reference keyframes
- Online visual matching between live camera frames and reference keyframes
- Control commands derived from image-space error

This approach keeps the navigation stack lightweight, interpretable, and practical for submission-level autonomous navigation tasks.

---

## 2. System Architecture

The navigation loop is organized as a visual-feedback control pipeline:

1. **Camera Input**  
   The drone captures live RGB frames from the onboard simulation camera.

2. **Feature Detection and Matching (ORB / BF Matching)**  
   Local features are extracted from both live and reference frames, then matched to establish correspondences.

3. **Error Estimation**  
   From matched geometry, the system estimates image-space alignment errors such as:
   - `dx` (horizontal error)
   - `dy` (vertical error)
   - `scale` (relative depth/size change)

4. **Control Command Generation (Visual Servoing)**  
   In the visual-servo loop, image-space error signals are converted into motion commands (yaw/translation/forward corrections) using configured gains and thresholds.

5. **Keyframe Progression**  
   Once alignment with the current keyframe is acceptable, the controller advances to the next keyframe until mission completion.

### YOLO Integration

The pipeline can include YOLO-based object/target detection to enhance scene understanding or add target-aware behavior during navigation.

---

## 3. Installation

### Requirements

- Python 3.8+ (recommended)
- OpenCV (`opencv-python`)
- NumPy
- PyYAML
- (Optional) Ultralytics/YOLO dependencies for object detection

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Usage

### 4.1 Keyframe Collection (Manual Mode)

Run:

```bash
python Sample_Drone_Interface.py
```

What happens in this phase:

- The user manually flies the drone through the desired route.
- The system captures and stores selected frames as keyframes.
- Keyframe metadata is written (e.g., keyframe index, timestamp, and related mission fields).

### 4.2 Autonomous Navigation

Run:

```bash
python scripts/run_navigation.py --config config/mission_config.yaml --debug-video
```

What happens in this phase:

- The recorded keyframe sequence is loaded from disk.
- Each live frame is matched against the active reference keyframe.
- Visual errors are converted into motion commands.
- The drone advances keyframe-by-keyframe until the route is completed.

Debug option:

- `--debug-video` enables real-time visual debugging overlays during navigation, helping you inspect feature matches, alignment error, and keyframe progression while the controller is running.

---

## 5. Controls

Control mapping can vary by simulator settings, but the workflow includes:

- **Basic flight controls**: takeoff, land, and manual translational movement
- **Manual route recording controls**: used during keyframe collection
- **Autopilot activation**: switch from manual mode to visual-servoing navigation
- **Camera view controls**: directional camera adjustments (arrow-key style, if enabled in your interface)

For concrete key bindings, refer to the interface logic in `Sample_Drone_Interface.py` and any project-specific control handlers in `scripts/`.

---

## 6. Debug Visualization

During operation, debug overlays can be enabled to inspect navigation quality in real time:

- Detected keypoints in live/reference frames
- Matched feature pairs
- Error vectors (`dx`, `dy`) and alignment indicators
- Suggested/active navigation direction
- Current keyframe index and progression status

These visual diagnostics are critical for tuning thresholds and validating controller behavior.

---

## 7. Logging

The system records navigation-relevant telemetry for analysis and tuning, including:

- `error_norm`
- `dx`, `dy`
- `inliers` (match quality / geometric consistency)
- issued control `commands`
- current `keyframe index`

Use these logs to diagnose unstable behavior, poor matching conditions, and controller gain issues.

---

## 8. Configuration

Mission and controller behavior are configured via:

- `config/mission_config.yaml`

Typical configurable groups include:

- Matching thresholds and acceptance criteria
- Feature extraction/matching parameters
- Control gains and command limits
- Keyframe handling behavior and progression logic

Centralizing these values in YAML allows reproducible experiments and fast iteration without code changes.

---

## 9. Project Structure

Typical repository layout:

```text
scripts/                 # Record/replay and runtime entry scripts
config/                  # Mission and controller configuration
data/keyframes/          # Captured keyframes and metadata
xlabs_nav/               # Navigation and visual-servoing modules
Sample_Drone_Interface.py# Drone simulation interface
```
---

## Notes

This project is designed as a robust, modular baseline for keyframe-driven autonomy in simulation.  
It is intentionally lightweight and well-suited for controlled mission replay, benchmarking, and further extension toward hybrid perception-navigation pipelines.
