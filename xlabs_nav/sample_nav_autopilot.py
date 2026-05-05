"""
Keyframe navigation error + simple P-control for Sample_Drone_Interface.

Parallel to the YOLO autopilot path: consumes live BGR frames, matches the active
keyframe with cached ORB descriptors (KeyframeManager), then maps
(dx_norm, dy_norm, scale_error) into a command dict that is applied via
apply_servo_command — the same adapter used by run_navigation.py.

This means both scripts share one sign source of truth. If a drone axis is
inverted, fix it in drone_control_adapter.apply_servo_command only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from xlabs_nav.drone_control_adapter import apply_servo_command
from xlabs_nav.keyframe_manager import KeyframeManager
from xlabs_nav.mission_config import MissionConfig, load_mission_config
from xlabs_nav.orb_matcher import extract_orb_features, match_orb_features
from xlabs_nav.visual_error import estimate_visual_error

# P-gains for the nav autopilot in Sample_Drone_Interface.
# Tune here without touching sign logic.
NAV_GAINS = {
    "K_yaw": 3.0,       # higher than PID default; P-only so needs more drive
    "K_strafe": 0.0,    # strafe off (yaw-only lateral correction)
    "K_altitude": 2.5,
    "K_scale": 0.45,    # maps scale_error → forward speed delta
    "trigger_cruise": 0.5,
}


def try_load_sample_nav_runtime(
    repo_root: Path,
    *,
    mission_config_path: Path | None = None,
) -> tuple[MissionConfig | None, KeyframeManager | None, str | None]:
    """
    Load mission YAML and keyframes + ORB cache.
    Returns (None, None, reason_string) on failure so the caller can degrade gracefully.
    """
    cfg_path = mission_config_path or (repo_root / "config" / "mission_config.yaml")
    try:
        mission = load_mission_config(cfg_path)
        km = KeyframeManager(mission)
        return mission, km, None
    except Exception as e:
        return None, None, str(e)


def compute_nav_p_command(
    frame_bgr: np.ndarray,
    mission: MissionConfig,
    keyframe_manager: KeyframeManager,
    control_state: dict[str, Any],
    *,
    gains: dict[str, float] | None = None,
    speed_scale: float = 0.4,
) -> tuple[dict[str, Any], bool]:
    """
    Match current frame to the active keyframe, compute visual error, apply P-gains,
    and write the result directly into control_state via apply_servo_command.

    Uses the same adapter as run_navigation.py — signs are consistent.

    Args:
        frame_bgr:        Current BGR frame.
        mission:          Loaded MissionConfig.
        keyframe_manager: Loaded KeyframeManager (holds ORB cache).
        control_state:    The shared drone control dict (written in-place).
        gains:            Optional dict to override NAV_GAINS entries.
        speed_scale:      Global multiplier applied to ALL axes (default 40 %).
                          Matches NAV_SPEED_SCALE in run_navigation.py.

    Returns:
        visual_error: dict from estimate_visual_error (valid, dx_norm, dy_norm, …)
        applied:      True if a valid command was written, False otherwise.
    """
    g = dict(NAV_GAINS)
    if gains:
        g.update({k: float(v) for k, v in gains.items() if k in g})

    cfg = mission.raw
    qc = cfg.get("quality", {})
    min_inliers = int(qc.get("minimum_inliers", 8))
    min_ratio = float(qc.get("minimum_inlier_ratio", 0.2))

    invalid_ve: dict[str, Any] = {
        "valid": False,
        "dx": 0.0, "dy": 0.0,
        "dx_norm": 0.0, "dy_norm": 0.0,
        "scale_error": 0.0, "error_norm": 0.0,
        "num_inliers": 0, "inlier_ratio": 0.0,
        "reason": "no_keyframe",
    }

    kf = keyframe_manager.get_active_keyframe()
    if kf is None:
        return invalid_ve, False

    kp_cur, des_cur, cur_hw = extract_orb_features(frame_bgr, cfg)
    match_result = match_orb_features(
        kp_cur, des_cur, cur_hw,
        kf["_orb_kp"], kf["_orb_des"], kf["_orb_hw"],
        cfg,
    )

    h, w = frame_bgr.shape[:2]
    visual_error = estimate_visual_error(match_result, w, h, min_inliers, min_ratio)

    if not visual_error.get("valid"):
        return visual_error, False

    dxn = float(visual_error["dx_norm"])
    dyn = float(visual_error["dy_norm"])
    scale_e = float(visual_error["scale_error"])

    # Error convention matches ServoPidController: e_yaw = -dx_norm, e_alt = -dy_norm.
    # speed_scale is applied uniformly to every axis so the caller (run_navigation.py
    # or Sample_Drone_Interface.py) controls overall speed in one place.
    cmd = {
        "yaw":      g["K_yaw"]      * (-dxn) * speed_scale,
        "strafe":   g["K_strafe"]   * (-dxn) * speed_scale,
        "altitude": g["K_altitude"] * (-dyn)  * speed_scale,
        "forward":  (g["trigger_cruise"] + g["K_scale"] * scale_e) * speed_scale,
        "duration": 0.0,
    }

    # Clamp forward; apply_servo_command clamps the rest.
    cmd["forward"] = float(np.clip(cmd["forward"], 0.0, 1.0))

    # Reverse mode: when the drone moves backward the camera frame of reference is
    # flipped left↔right in world space, so horizontal commands must be inverted.
    # Only yaw and strafe are affected — forward speed and altitude are unchanged.
    moving_backward = (
        float(control_state.get("reverse", 0.0)) > 0.0
        or bool(control_state.get("reverse_down", False))
    )
    if moving_backward:
        cmd["yaw"]    = -cmd["yaw"]
        cmd["strafe"] = -cmd["strafe"]

    # Write into control_state using the shared adapter (same sign logic as run_navigation.py).
    apply_servo_command(control_state, cmd)
    control_state["nav_autopilot"] = True

    return visual_error, True
