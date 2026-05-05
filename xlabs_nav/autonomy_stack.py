from __future__ import annotations

from typing import Any

import numpy as np

from xlabs_nav.drone_control_adapter import stop_drone
from xlabs_nav.keyframe_manager import KeyframeManager
from xlabs_nav.mission_config import MissionConfig
from xlabs_nav.orb_matcher import extract_orb_features, match_orb_features
from xlabs_nav.servo_controller import ServoPidController
from xlabs_nav.visual_error import estimate_visual_error

try:
    from xlabs_nav.yolo_target_tracker import YoloTargetTracker
except Exception:  # pragma: no cover
    # Tracker is optional; if ultralytics isn't installed AutonomyStack still
    # runs keyframe navigation. The error surfaces only if yolo.enabled=true.
    YoloTargetTracker = None  # type: ignore[assignment]


def _recovery_yaw_direction(frame_index: int, alternate_frames: int) -> float:
    phase = (frame_index // max(int(alternate_frames), 1)) % 2
    return 1.0 if phase == 0 else -1.0


def _selection_params(cfg: dict[str, Any]) -> dict[str, Any]:
    ks = cfg.get("keyframe_selection") or {}
    qc = cfg.get("quality") or {}
    return {
        "min_inlier_ratio": float(ks.get("min_inlier_ratio", qc.get("minimum_inlier_ratio", 0.2))),
        "min_matches": int(ks.get("min_matches", qc.get("minimum_matches", 12))),
        "min_inliers": int(ks.get("min_inliers", qc.get("minimum_inliers", 8))),
        "max_keyframes_to_score": int(ks.get("max_keyframes_to_score", 0)),
    }


def _scan_indices(active: int, total: int, max_score: int) -> list[int]:
    order = list(range(active, total)) + list(range(0, active))
    if max_score > 0 and len(order) > max_score:
        return order[:max_score]
    return order


def _select_best_keyframe_match(
    kp_cur: list[Any],
    des_cur: Any,
    cur_hw: tuple[int, int],
    keyframes: KeyframeManager,
    cfg: dict[str, Any],
    sel: dict[str, Any],
) -> tuple[int | None, dict[str, Any] | None, str]:
    """
    Returns (best_index, match_result_for_that_keyframe, reason_if_none).
    Tie: higher inlier_ratio, then more inliers, then earlier in forward-first scan order.
    """
    total = keyframes.total
    if total == 0:
        return None, None, "no_keyframes"
    active = keyframes.active_index
    indices = _scan_indices(active, total, sel["max_keyframes_to_score"])
    min_r = sel["min_inlier_ratio"]
    min_m = sel["min_matches"]
    min_i = sel["min_inliers"]

    best_i: int | None = None
    best_mr: dict[str, Any] | None = None
    best_ratio = -1.0
    best_inl = -1
    best_rank = 10**9

    for rank, i in enumerate(indices):
        kf = keyframes.get_keyframe_by_index(i)
        if kf is None:
            continue
        mr = match_orb_features(
            kp_cur,
            des_cur,
            cur_hw,
            kf["_orb_kp"],
            kf["_orb_des"],
            kf["_orb_hw"],
            cfg,
        )
        if not mr.get("valid"):
            continue
        if mr["num_matches"] < min_m or mr["num_inliers"] < min_i:
            continue
        r = float(mr["inlier_ratio"])
        if r < min_r:
            continue
        inl = int(mr["num_inliers"])
        better = False
        if r > best_ratio:
            better = True
        elif r == best_ratio and inl > best_inl:
            better = True
        elif r == best_ratio and inl == best_inl and rank < best_rank:
            better = True
        if better:
            best_ratio, best_inl, best_i, best_mr, best_rank = r, inl, i, mr, rank

    if best_i is None or best_mr is None:
        return None, None, "no_keyframe_above_selection_threshold"
    return best_i, best_mr, ""


class AutonomyStack:
    """ORB-only keyframe visual servoing + convergence advance + recovery."""

    def __init__(self, mission: MissionConfig):
        self._cfg = mission.raw
        self._keyframes = KeyframeManager(mission)
        self._stable_frames = 0
        self._frame_index = 0
        self._servo_pid = ServoPidController()
        self._error_norm_ema: float | None = None

        ycfg = self._cfg.get("yolo") or {}
        self._yolo: YoloTargetTracker | None = None
        if bool(ycfg.get("enabled", False)):
            if YoloTargetTracker is None:
                raise RuntimeError(
                    "yolo.enabled=true in mission config but ultralytics is not "
                    "installed. Run `pip install ultralytics` or set yolo.enabled=false."
                )
            self._yolo = YoloTargetTracker(ycfg, mission.repo_root)

    @property
    def keyframe_manager(self) -> KeyframeManager:
        return self._keyframes

    def force_advance_keyframe(self) -> bool:
        """
        Skip convergence checks and advance to the next keyframe in mission order.
        Returns False if the mission was already marked complete.
        """
        if self._keyframes.is_complete():
            return False
        self._keyframes.advance()
        self._servo_pid.reset()
        self._stable_frames = 0
        self._error_norm_ema = None
        return True

    def step(self, frame_bgr: np.ndarray, control_state: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        self._frame_index += 1
        qc = self._cfg["quality"]
        cc = self._cfg["convergence"]
        rc = self._cfg["recovery"]
        ctrl = self._cfg["controller"]
        sel = _selection_params(self._cfg)

        debug_info: dict[str, Any] = {
            "state": "SERVO_TO_KEYFRAME",
            "active_keyframe_id": "---",
            "keyframe_progress": (0, self._keyframes.total),
            "reference_bgr": None,
            "current_keypoints": [],
            "reference_keypoints": [],
            "match_cur_xy": np.zeros((0, 2), dtype=np.float32),
            "match_ref_xy": np.zeros((0, 2), dtype=np.float32),
            "match_is_inlier": np.zeros((0,), dtype=bool),
            "num_matches": 0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "confidence": "LOW",
            "dx": 0.0,
            "dy": 0.0,
            "dx_norm": 0.0,
            "dy_norm": 0.0,
            "scale_error": 0.0,
            "error_norm": 0.0,
            "forward": 0.0,
            "yaw": 0.0,
            "strafe": 0.0,
            "altitude": 0.0,
            "forward_blocked": False,
            "recovery_reason": "",
            "selection_best_ratio": 0.0,
            "selection_keyframe_index": -1,
            "keyframe_switched": False,
        }

        cmd_stop = {"forward": 0.0, "strafe": 0.0, "yaw": 0.0, "altitude": 0.0, "duration": ctrl["command_duration"]}

        if self._keyframes.is_complete():
            debug_info["state"] = "MISSION_COMPLETE"
            debug_info["keyframe_progress"] = (self._keyframes.total, self._keyframes.total)
            stop_drone(control_state)
            control_state["autopilot"] = False
            return cmd_stop, debug_info

        # --- YOLO INFERENCE ---
        # Run inference every frame so the debug overlay always reflects the
        # current detections, regardless of mission progress. YOLO is only
        # allowed to take over command issuing once the drone has reached the
        # final keyframe (active_index == total - 1). Until then, YOLO is
        # purely informational and keyframe navigation runs unmodified.
        if self._yolo is not None:
            yolo_info = self._yolo.step(frame_bgr)
            yolo_mode = str(yolo_info.get("mode", "LOST"))
            debug_info["yolo"] = yolo_info

            is_final_keyframe = (
                self._keyframes.active_index >= self._keyframes.total - 1
            )
            if (
                is_final_keyframe
                and yolo_mode in ("YOLO_TARGET_TRACKING", "TARGET_CENTERED")
            ):
                self._servo_pid.reset()
                self._stable_frames = 0
                self._error_norm_ema = None

                yolo_cmd = dict(yolo_info.get("command") or cmd_stop)
                yolo_cmd.setdefault("duration", ctrl["command_duration"])

                debug_info["state"] = yolo_mode
                debug_info["forward"] = float(yolo_cmd.get("forward", 0.0))
                debug_info["yaw"] = float(yolo_cmd.get("yaw", 0.0))
                debug_info["strafe"] = float(yolo_cmd.get("strafe", 0.0))
                debug_info["altitude"] = float(yolo_cmd.get("altitude", 0.0))
                debug_info["recovery_reason"] = ""
                debug_info["active_keyframe_id"] = "FINAL"
                debug_info["keyframe_progress"] = (
                    self._keyframes.total,
                    self._keyframes.total,
                )
                return yolo_cmd, debug_info

        kp_cur, des_cur, cur_hw = extract_orb_features(frame_bgr, self._cfg)
        best_i, match_result, sel_reason = _select_best_keyframe_match(
            kp_cur, des_cur, cur_hw, self._keyframes, self._cfg, sel
        )

        selection_failed = best_i is None or match_result is None

        # Prefer staying on the mission-order active keyframe unless another candidate
        # beats it by a clear inlier_ratio margin. Stops flip-flop relocalize → stable reset.
        margin = float(sel.get("prefer_active_inlier_ratio_margin", 0.0))
        if (
            not selection_failed
            and margin > 0.0
            and best_i is not None
            and match_result is not None
            and int(best_i) != int(self._keyframes.active_index)
        ):
            active_idx = int(self._keyframes.active_index)
            kf_a = self._keyframes.get_keyframe_by_index(active_idx)
            if kf_a is not None:
                mr_a = match_orb_features(
                    kp_cur,
                    des_cur,
                    cur_hw,
                    kf_a["_orb_kp"],
                    kf_a["_orb_des"],
                    kf_a["_orb_hw"],
                    self._cfg,
                )
                min_m = sel["min_matches"]
                min_i = sel["min_inliers"]
                min_r = sel["min_inlier_ratio"]
                ok_a = (
                    mr_a.get("valid")
                    and int(mr_a["num_matches"]) >= min_m
                    and int(mr_a["num_inliers"]) >= min_i
                    and float(mr_a["inlier_ratio"]) >= min_r
                )
                if ok_a:
                    r_best = float(match_result["inlier_ratio"])
                    r_active = float(mr_a["inlier_ratio"])
                    if r_active + margin >= r_best:
                        best_i = active_idx
                        match_result = mr_a
                        sel_reason = ""

        selection_failed = best_i is None or match_result is None
        if not selection_failed:
            debug_info["selection_best_ratio"] = float(match_result["inlier_ratio"])
            debug_info["selection_keyframe_index"] = int(best_i)
        else:
            debug_info["recovery_reason"] = sel_reason

        # Forbid backward relocalize — after advancing to keyframe N+1, never re-attach
        # to N just because the drone is still visually closer to it. Going backward in
        # mission order caused infinite advance/relocalize oscillation.
        if (
            not selection_failed
            and best_i is not None
            and int(best_i) > int(self._keyframes.active_index)
        ):
            self._keyframes.set_active_index(int(best_i))
            self._servo_pid.reset()
            self._stable_frames = 0
            self._error_norm_ema = None
            debug_info["keyframe_switched"] = True
        elif (
            not selection_failed
            and best_i is not None
            and int(best_i) < int(self._keyframes.active_index)
        ):
            # Selector wants to go back; refuse, but reuse the active keyframe's match
            # so visual_error / confidence reflect the active reference.
            active_idx = int(self._keyframes.active_index)
            kf_a = self._keyframes.get_keyframe_by_index(active_idx)
            if kf_a is not None:
                mr_a = match_orb_features(
                    kp_cur,
                    des_cur,
                    cur_hw,
                    kf_a["_orb_kp"],
                    kf_a["_orb_des"],
                    kf_a["_orb_hw"],
                    self._cfg,
                )
                match_result = mr_a
                best_i = active_idx
                debug_info["selection_keyframe_index"] = active_idx
                debug_info["selection_best_ratio"] = float(mr_a.get("inlier_ratio", 0.0))
                selection_failed = not bool(mr_a.get("valid", False))

        kf = self._keyframes.get_active_keyframe()
        if kf is None:
            debug_info["state"] = "MISSION_COMPLETE"
            stop_drone(control_state)
            return cmd_stop, debug_info

        if selection_failed:
            ref_kp = kf.get("_orb_kp") or []
            match_result = {
                "valid": False,
                "current_keypoints": kp_cur,
                "reference_keypoints": ref_kp,
                "current_points": np.zeros((0, 2), dtype=np.float32),
                "reference_points": np.zeros((0, 2), dtype=np.float32),
                "matches": [],
                "num_matches": 0,
                "homography": None,
                "inlier_mask": None,
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "reference_shape_hw": kf.get("_orb_hw", cur_hw),
                "current_shape_hw": cur_hw,
            }

        ref_bgr = kf["_bgr"]
        kid = kf.get("id", self._keyframes.active_index + 1)
        debug_info["active_keyframe_id"] = f"{int(kid):03d}"
        debug_info["keyframe_progress"] = (self._keyframes.active_index + 1, self._keyframes.total)
        debug_info["reference_bgr"] = ref_bgr

        debug_info["current_keypoints"] = match_result.get("current_keypoints") or []
        debug_info["reference_keypoints"] = match_result.get("reference_keypoints") or []

        min_matches = int(qc["minimum_matches"])
        min_inl = int(qc["minimum_inliers"])
        min_ratio = float(qc["minimum_inlier_ratio"])
        high_ratio = float(qc["high_confidence_inlier_ratio"])

        matches_weak = (
            selection_failed
            or not match_result["valid"]
            or match_result["num_matches"] < min_matches
            or match_result["num_inliers"] < min_inl
            or match_result["inlier_ratio"] < min_ratio
        )

        h_cur, w_cur = frame_bgr.shape[:2]
        visual_error = estimate_visual_error(
            match_result,
            w_cur,
            h_cur,
            min_inl,
            min_ratio,
        )

        debug_info["num_matches"] = match_result["num_matches"]
        debug_info["num_inliers"] = match_result["num_inliers"]
        debug_info["inlier_ratio"] = float(match_result["inlier_ratio"])
        debug_info["confidence"] = "HIGH" if match_result["inlier_ratio"] >= high_ratio else "LOW"

        mask = match_result.get("inlier_mask")
        cur_pts = match_result.get("current_points")
        ref_pts = match_result.get("reference_points")
        if mask is not None and cur_pts is not None and ref_pts is not None:
            debug_info["match_cur_xy"] = cur_pts.copy()
            debug_info["match_ref_xy"] = ref_pts.copy()
            debug_info["match_is_inlier"] = mask.copy()

        recovery_mode = matches_weak or not visual_error["valid"]
        yaw_sign = _recovery_yaw_direction(self._frame_index, int(rc["alternate_frames"]))
        recovery_yaw = float(rc["yaw_magnitude"]) * yaw_sign

        if recovery_mode:
            debug_info["state"] = "RECOVERY"
            debug_info["forward_blocked"] = True
            if not debug_info.get("recovery_reason"):
                debug_info["recovery_reason"] = visual_error.get("reason") or "matching_unstable"
            if matches_weak and not selection_failed:
                debug_info["recovery_reason"] = visual_error.get("reason") or "insufficient_inliers_or_matches"

        ve_for_cmd = visual_error
        cmd = self._servo_pid.compute(
            ve_for_cmd,
            ctrl,
            recovery_mode=recovery_mode,
            recovery_yaw=recovery_yaw,
        )

        # --- KEYFRAME ADVANCE ---
        # Rule (intentionally simple, matches user spec):
        #   advance when  raw_error_norm < error_norm_threshold
        #             AND confidence == "HIGH"  (inlier_ratio >= high_confidence_inlier_ratio)
        #   held for required_stable_frames consecutive frames.
        #
        # Recovery_mode is NOT a gate. If matching is poor, error_norm is high or
        # confidence is LOW, so err_ok will fail naturally without an extra check.
        # The optional EMA still updates so the overlay can display it, but it is NOT
        # part of the advance gate.
        raw_err = float(visual_error.get("error_norm", 1.0))
        ema_alpha = float(cc.get("error_ema_alpha", 1.0))
        if ema_alpha < 1.0:
            a = max(1e-6, min(1.0, ema_alpha))
            if self._error_norm_ema is None:
                self._error_norm_ema = raw_err
            else:
                self._error_norm_ema = a * raw_err + (1.0 - a) * self._error_norm_ema

        thr = float(cc["error_norm_threshold"])
        # Looser advance: only the error gate. Confidence is no longer required because
        # estimate_visual_error already enforces minimum_inliers / minimum_inlier_ratio
        # for visual_error.valid (and a poor match yields a high error_norm anyway).
        err_ok = bool(visual_error.get("valid", False)) and (raw_err < thr)

        if err_ok:
            self._stable_frames += 1
        else:
            if bool(cc.get("strict_stable_reset", True)):
                self._stable_frames = 0
            else:
                dec = max(1, int(cc.get("stable_decrement_on_bad", 1)))
                self._stable_frames = max(0, self._stable_frames - dec)

        # required_stable_frames=1 → advance immediately on the first good frame.
        req = max(1, int(cc.get("required_stable_frames", 1)))
        if self._stable_frames >= req:
            self._stable_frames = 0
            self._error_norm_ema = None
            prev_idx = int(self._keyframes.active_index)
            last_idx = prev_idx >= self._keyframes.total - 1
            self._keyframes.advance()
            self._servo_pid.reset()
            new_idx = int(self._keyframes.active_index)
            print(
                f"[autonomy] advance keyframe: {prev_idx + 1}/{self._keyframes.total} "
                f"-> {new_idx + 1}/{self._keyframes.total} (error_norm={raw_err:.3f})"
            )
            if last_idx or self._keyframes.is_complete():
                debug_info["state"] = "MISSION_COMPLETE"
                stop_drone(control_state)
                control_state["autopilot"] = False
                return cmd_stop, debug_info

        debug_info["dx"] = float(visual_error.get("dx", 0.0))
        debug_info["dy"] = float(visual_error.get("dy", 0.0))
        debug_info["dx_norm"] = float(visual_error.get("dx_norm", 0.0))
        debug_info["dy_norm"] = float(visual_error.get("dy_norm", 0.0))
        debug_info["scale_error"] = float(visual_error.get("scale_error", 0.0))
        debug_info["error_norm"] = float(visual_error.get("error_norm", 0.0))
        ema_a = float(cc.get("error_ema_alpha", 1.0))
        if self._error_norm_ema is not None and ema_a < 1.0:
            debug_info["error_norm_ema"] = float(self._error_norm_ema)
        else:
            debug_info["error_norm_ema"] = float(visual_error.get("error_norm", 0.0))
        debug_info["convergence_stable_frames"] = int(self._stable_frames)

        debug_info["forward"] = float(cmd["forward"])
        debug_info["yaw"] = float(cmd["yaw"])
        debug_info["strafe"] = float(cmd["strafe"])
        debug_info["altitude"] = float(cmd["altitude"])

        return cmd, debug_info
