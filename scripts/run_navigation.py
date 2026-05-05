"""
ORB-only keyframe visual servoing with optional live debug overlay.

Usage (from repository root):
  python scripts/run_navigation.py --config config/mission_config.yaml --debug-video

Arm and fly manually first; press N to start KBVS (N again pauses and returns to manual).
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import NDIlib as ndi
import cv2
import keyboard
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xlabs_nav.autonomy_stack import AutonomyStack
from xlabs_nav.debug_overlay import render_debug_overlay
from xlabs_nav.drone_control_adapter import apply_servo_command, stop_drone
from xlabs_nav.frame_utils import bgr_from_ndi
from xlabs_nav.mission_config import load_mission_config
from xlabs_nav.sample_nav_autopilot import compute_nav_p_command, try_load_sample_nav_runtime

# Scale all KBVS servo axes (forward, strafe, yaw, altitude) before sending to Unity.
# K_yaw=10 in config already saturates at max_yaw=1.0 before this scale; raising this
# lets the clamped yaw output actually reach Unity at a higher fraction of ±1.
NAV_SPEED_SCALE = 0.2

# Toggled with keyboard "n" (global hook) so it works with or without --debug-video.
_autonomy_enabled = False
# Press ] while KBVS is running to skip convergence and jump to the next keyframe.
_manual_advance_keyframe_pending = False

# --- duplicated minimal Unity bridge (matches Sample_Drone_Interface semantics) ---

control_state = {
    "btnAdown": False,
    "btnBdown": False,
    "btnCdown": False,
    "btnARMdown": False,
    "trigger": 0.0,
    "trigger_down": False,
    "reverse": 0.0,
    "reverse_down": False,
    "joy_vertical": 0,
    "joy_horizontal": 0,
    "yaw": 0.0,
    "pitch": 0.0,
    "thumb_down": False,
    "joy_click": False,
    "joy_up": False,
    "joy_down": False,
    "joy_left": False,
    "joy_right": False,
    "arrow_left": False,
    "arrow_right": False,
    "arrow_up": False,
    "arrow_down": False,
    "autopilot": False,
}

time_from_unity = 0.0
static_boxes = [
    {"x": -100, "y": -100, "width": 100, "height": 100, "id": "box1"},
    {"x": 100, "y": -100, "width": 100, "height": 100, "id": "box2"},
    {"x": 200, "y": -200, "width": 120, "height": 140, "id": "box3"},
]


def clamp(minimum: float, x: float, maximum: float) -> float:
    return max(minimum, min(x, maximum))


def on_key_event(event) -> None:
    global control_state, _autonomy_enabled, _manual_advance_keyframe_pending
    is_down = event.event_type == "down"
    key = event.name
    if key == "n" and is_down:
        _autonomy_enabled = not _autonomy_enabled
        if not _autonomy_enabled:
            stop_drone(control_state)
            control_state["autopilot"] = False
            print("KBVS: PAUSED — manual control (keyboard).")
        else:
            print("KBVS: RUNNING — visual servoing active.")
    if key == "]" and is_down:
        _manual_advance_keyframe_pending = True
        print("Manual keyframe advance requested (will apply on next loop if KBVS is on).")
    if key == "2":
        control_state["btnAdown"] = is_down
    if key == "b":
        control_state["btnBdown"] = is_down
    if key == "c":
        control_state["btnCdown"] = is_down
    if key == "1":
        control_state["btnARMdown"] = is_down
    if key == "w":
        control_state["trigger_down"] = is_down
    if key == "s":
        control_state["reverse_down"] = is_down
    if key == "e":
        control_state["joy_up"] = is_down
    if key == "f":
        control_state["joy_down"] = is_down
    if key == "a":
        control_state["joy_left"] = is_down
    if key == "d":
        control_state["joy_right"] = is_down
    if key == "k":
        control_state["joy_click"] = is_down
    if key == "left":
        control_state["arrow_left"] = is_down
    if key == "right":
        control_state["arrow_right"] = is_down
    if key == "up":
        control_state["arrow_up"] = is_down
    if key == "down":
        control_state["arrow_down"] = is_down
    if key == "p":
        control_state["thumb_down"] = is_down


def update_controls() -> None:
    while True:
        if control_state["autopilot"]:
            # Keep camera-view control available during autonomy without touching
            # navigation axes (forward/strafe/altitude/yaw from KBVS).
            if control_state["arrow_right"]:
                control_state["yaw"] = clamp(-1, control_state["yaw"] + 0.05, 1)
            elif control_state["arrow_left"]:
                control_state["yaw"] = clamp(-1, control_state["yaw"] - 0.05, 1)
            if control_state["joy_left"]:
                control_state["joy_horizontal"] = -1
            elif control_state["joy_right"]:
                control_state["joy_horizontal"] = 1
            else:
                control_state["joy_horizontal"] = 0
            if control_state["btnCdown"]:
                control_state["yaw"] = 0
            time.sleep(1 / 60)
            continue
        if control_state["trigger"] > 0 and not control_state["trigger_down"]:
            control_state["trigger"] = clamp(0, control_state["trigger"] - 0.1, 1)
        if control_state["trigger"] < 1 and control_state["trigger_down"]:
            control_state["trigger"] = clamp(0, control_state["trigger"] + 0.05, 1)
        if control_state["reverse"] > 0 and not control_state["reverse_down"]:
            control_state["reverse"] = clamp(0, control_state["reverse"] - 0.1, 1)
        if control_state["reverse"] < 1 and control_state["reverse_down"]:
            control_state["reverse"] = clamp(0, control_state["reverse"] + 0.05, 1)
        if control_state["joy_up"]:
            control_state["joy_vertical"] = -1
        elif control_state["joy_down"]:
            control_state["joy_vertical"] = 1
        else:
            control_state["joy_vertical"] = 0
        if control_state["joy_left"]:
            control_state["joy_horizontal"] = -1
        elif control_state["joy_right"]:
            control_state["joy_horizontal"] = 1
        else:
            control_state["joy_horizontal"] = 0
        if control_state["arrow_right"]:
            control_state["yaw"] = clamp(-1, control_state["yaw"] + 0.05, 1)
        elif control_state["arrow_left"]:
            control_state["yaw"] = clamp(-1, control_state["yaw"] - 0.05, 1)
        if control_state["arrow_up"]:
            control_state["pitch"] = clamp(-1, control_state["pitch"] - 0.05, 1)
        elif control_state["arrow_down"]:
            control_state["pitch"] = clamp(-1, control_state["pitch"] + 0.05, 1)
        if control_state["btnCdown"]:
            control_state["pitch"] = 0
            control_state["yaw"] = 0
        time.sleep(1 / 60)


def listen_to_unity(conn: socket.socket) -> None:
    global time_from_unity
    try:
        while True:
            raw_msglen = conn.recv(4)
            if not raw_msglen:
                break
            message_length = struct.unpack(">I", raw_msglen)[0]
            data = b""
            while len(data) < message_length:
                packet = conn.recv(message_length - len(data))
                if not packet:
                    break
                data += packet
            try:
                message = json.loads(data.decode("utf-8"))
                time_from_unity = message.get("time", 0)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"Connection Listener Error: {e}")


def send_to_unity(conn: socket.socket) -> None:
    try:
        while True:
            response = {
                "num_of_boxes": 3,
                "data": static_boxes,
                "time": time_from_unity,
                "btnAdown": control_state["btnAdown"],
                "btnBdown": control_state["btnBdown"],
                "btnCdown": control_state["btnCdown"],
                "btnARMdown": control_state["btnARMdown"],
                "trigger": control_state["trigger"],
                "triggerDown": control_state["trigger_down"],
                "reverse": control_state["reverse"],
                "reverseDown": control_state["reverse_down"],
                "joy_vertical": control_state["joy_vertical"],
                "joy_horizontal": control_state["joy_horizontal"],
                "yaw": control_state["yaw"],
                "pitch": control_state["pitch"],
                "thumbDown": control_state["thumb_down"],
                "joyClick": control_state["joy_click"],
            }
            response_bytes = json.dumps(response).encode("utf-8")
            conn.send(len(response_bytes).to_bytes(4, byteorder="big"))
            conn.send(response_bytes)
            time.sleep(1 / 60)
    except (ConnectionResetError, BrokenPipeError):
        print("Unity connection lost (Sender).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORB keyframe navigation + optional debug video")
    p.add_argument("--config", default="config/mission_config.yaml", help="Path to mission YAML")
    p.add_argument("--debug-video", action="store_true", help="Show OpenCV window with KBVS overlay")
    p.add_argument(
        "--simple-p",
        action="store_true",
        help=(
            "Use sample_nav_autopilot P-controller instead of AutonomyStack PID. "
            "Skips keyframe advancement and recovery logic — single active keyframe only."
        ),
    )
    return p.parse_args()


def main() -> int:
    global _manual_advance_keyframe_pending
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    mission = load_mission_config(cfg_path)
    net = mission.raw["network"]
    host = str(net["host"])
    port = int(net["port"])

    print("--- CONTROL SERVER (KBVS navigation) ---")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"Waiting for Unity on {host}:{port}...")
        conn, addr = server_socket.accept()
        print(f"Connected to Drone at {addr}")
    except Exception as e:
        print(f"CRITICAL: Could not start server: {e}")
        return 1

    threading.Thread(target=listen_to_unity, args=(conn,), daemon=True).start()
    threading.Thread(target=send_to_unity, args=(conn,), daemon=True).start()
    threading.Thread(target=update_controls, daemon=True).start()

    try:
        keyboard.hook(on_key_event)
        print("Keyboard hooks active (same bindings as Sample_Drone_Interface).")
    except Exception as e:
        print(f"WARNING: keyboard hook failed ({e}). Manual controls may not work.")

    stack = AutonomyStack(mission)
    overlay_cfg = {
        **mission.raw["overlay"],
        "max_draw_matches": mission.raw["matching"]["max_draw_matches"],
    }

    # --- Simple-P alternative ---
    simple_p_mission = None
    simple_p_km = None
    if args.simple_p:
        simple_p_mission, simple_p_km, sp_err = try_load_sample_nav_runtime(
            ROOT, mission_config_path=ROOT / args.config
        )
        if simple_p_km is None:
            print(f"ERROR: --simple-p could not load keyframes: {sp_err}")
            server_socket.close()
            ndi.destroy()
            return 1
        print(f"Simple-P controller active — {simple_p_km.total} keyframe(s) loaded.")
        print("Keyframe advancement is MANUAL in simple-P mode (no convergence check).")

    print("\n--- NDI VIDEO ---")
    if not ndi.initialize():
        print("ERROR: NDI initialize failed.")
        server_socket.close()
        return 1

    ndi_find = ndi.find_create_v2()
    if ndi_find is None:
        server_socket.close()
        return 1

    sources = []
    for _ in range(10):
        ndi.find_wait_for_sources(ndi_find, 500)
        sources = ndi.find_get_current_sources(ndi_find)
        if len(sources) > 0:
            break

    selected_source = sources[0] if sources else None
    if sources:
        for s in sources:
            print(f"NDI source: {s.ndi_name}")
            if "Unity" in s.ndi_name:
                selected_source = s
        print(f"Using: {selected_source.ndi_name}")
    else:
        print("WARNING: No NDI sources — video disabled.")

    ndi_recv = None
    if selected_source:
        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        ndi_recv = ndi.recv_create_v3(ndi_recv_create)
        ndi.recv_connect(ndi_recv, selected_source)
        ndi.find_destroy(ndi_find)
        ndi_find = None
    else:
        ndi.find_destroy(ndi_find)
        ndi_find = None
        print("ERROR: No NDI sources — cannot run KBVS without video.")
        ndi.destroy()
        server_socket.close()
        return 1

    window_name = "KBVS Debug (NDI)"
    scale_factor = 0.55

    print("\n=== KBVS NAVIGATION RUNNING ===")
    print("Arm and take off manually first. Press N when ready to start autonomous KBVS.")
    print("Press N again to pause KBVS and return to manual control.")
    print("Press ] (right bracket) while KBVS is ON to force-advance to the next keyframe.")
    print(f"Controller: {'Simple-P (sample_nav_autopilot)' if args.simple_p else 'AutonomyStack PID'}")
    print(f"KBVS command scale: {NAV_SPEED_SCALE:.0%} of configured max (forward/yaw/strafe/altitude).")
    if args.debug_video:
        print("Debug window: q / ESC quit, +/- zoom.")
    try:
        while True:
            if _manual_advance_keyframe_pending:
                _manual_advance_keyframe_pending = False
                if _autonomy_enabled:
                    if args.simple_p and simple_p_km is not None:
                        km = simple_p_km
                        if km.is_complete():
                            print("Manual advance ignored — mission already complete.")
                        else:
                            km.advance()
                            print(
                                f"Manual advance (simple-P) → keyframe "
                                f"{km.active_index + 1}/{km.total}"
                            )
                            if km.is_complete():
                                stop_drone(control_state)
                                print("Mission complete — stopping outputs.")
                    else:
                        km = stack.keyframe_manager
                        if km.is_complete():
                            print("Manual advance ignored — mission already complete.")
                        else:
                            stack.force_advance_keyframe()
                            print(
                                f"Manual advance → keyframe "
                                f"{km.active_index + 1}/{km.total}"
                            )
                            if km.is_complete():
                                stop_drone(control_state)
                                print("Mission complete — stopping outputs.")
                else:
                    print("Manual advance ignored — enable KBVS with N first.")

            if ndi_recv:
                t, v, a, _ = ndi.recv_capture_v2(ndi_recv, 1000)
                if t == ndi.FRAME_TYPE_VIDEO:
                    frame = np.copy(v.data)
                    bgr = bgr_from_ndi(frame)

                    if _autonomy_enabled and args.simple_p:
                        # --- Simple P-controller path ---
                        ve, applied = compute_nav_p_command(
                            bgr, simple_p_mission, simple_p_km, control_state,
                            speed_scale=NAV_SPEED_SCALE,
                        )
                        # Build a minimal dbg so the rest of the display code is reused.
                        dbg = {
                            "state": "SIMPLE_P" if applied else "SIMPLE_P_NO_MATCH",
                            "active_keyframe_id": (
                                simple_p_km.get_active_keyframe() or {}
                            ).get("id", "?"),
                            "keyframe_progress": (simple_p_km.active_index + 1, simple_p_km.total),
                            "reference_bgr": (simple_p_km.get_active_keyframe() or {}).get("_bgr"),
                            "current_keypoints": [],
                            "reference_keypoints": [],
                            "match_cur_xy": np.zeros((0, 2), dtype=np.float32),
                            "match_ref_xy": np.zeros((0, 2), dtype=np.float32),
                            "match_is_inlier": np.zeros((0,), dtype=bool),
                            "num_matches": int(ve.get("num_inliers", 0)),
                            "num_inliers": int(ve.get("num_inliers", 0)),
                            "inlier_ratio": float(ve.get("inlier_ratio", 0.0)),
                            "confidence": "HIGH" if ve.get("valid") else "LOW",
                            "dx": float(ve.get("dx", 0.0)),
                            "dy": float(ve.get("dy", 0.0)),
                            "dx_norm": float(ve.get("dx_norm", 0.0)),
                            "dy_norm": float(ve.get("dy_norm", 0.0)),
                            "scale_error": float(ve.get("scale_error", 0.0)),
                            "error_norm": float(ve.get("error_norm", 0.0)),
                            "forward": float(control_state.get("trigger", 0.0)),
                            "yaw": float(control_state.get("yaw", 0.0)),
                            "strafe": float(control_state.get("joy_horizontal", 0.0)),
                            "altitude": float(control_state.get("joy_vertical", 0.0)),
                            "forward_blocked": False,
                            "recovery_reason": ve.get("reason", "") if not applied else "",
                            "selection_keyframe_index": simple_p_km.active_index,
                            "selection_best_ratio": float(ve.get("inlier_ratio", 0.0)),
                            "keyframe_switched": False,
                        }

                    elif _autonomy_enabled:
                        # --- Full AutonomyStack PID path (unchanged) ---
                        cmd, dbg = stack.step(bgr, control_state)
                        if dbg.get("state") != "MISSION_COMPLETE":
                            cmd_scaled = dict(cmd)
                            for axis in ("forward", "strafe", "yaw", "altitude"):
                                cmd_scaled[axis] = float(cmd[axis]) * NAV_SPEED_SCALE

                            # Reverse mode: camera frame is flipped left↔right when
                            # moving backward, so invert horizontal commands only.
                            moving_backward = (
                                float(control_state.get("reverse", 0.0)) > 0.0
                                or bool(control_state.get("reverse_down", False))
                            )
                            if moving_backward:
                                cmd_scaled["yaw"]    = -cmd_scaled["yaw"]
                                cmd_scaled["strafe"] = -cmd_scaled["strafe"]

                            apply_servo_command(control_state, cmd_scaled)
                            dbg = {
                                **dbg,
                                "forward": cmd_scaled["forward"],
                                "strafe": cmd_scaled["strafe"],
                                "yaw": cmd_scaled["yaw"],
                                "altitude": cmd_scaled["altitude"],
                            }
                    else:
                        dbg = {
                            "state": "MANUAL",
                            "active_keyframe_id": "---",
                            "keyframe_progress": (0, stack.keyframe_manager.total),
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
                            "selection_keyframe_index": -1,
                            "selection_best_ratio": 0.0,
                            "keyframe_switched": False,
                        }

                    disp = bgr
                    if args.debug_video:
                        if _autonomy_enabled:
                            disp = render_debug_overlay(bgr, dbg, overlay_cfg)
                        else:
                            disp = bgr.copy()
                            y = 28
                            for line in (
                                "MANUAL — KBVS off",
                                "Press N to start autonomous navigation",
                            ):
                                cv2.putText(
                                    disp,
                                    line,
                                    (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                y += 28

                    if scale_factor != 1.0:
                        hh, ww = disp.shape[:2]
                        disp = cv2.resize(disp, (int(ww * scale_factor), int(hh * scale_factor)))

                    if args.debug_video:
                        try:
                            cv2.imshow(window_name, disp)
                        except cv2.error as e:
                            if "not implemented" in str(e):
                                print("OpenCV GUI unavailable (install opencv-python, not headless).")
                                args.debug_video = False
                            else:
                                raise

                    ndi.recv_free_video_v2(ndi_recv, v)
                elif t == ndi.FRAME_TYPE_AUDIO:
                    ndi.recv_free_audio_v2(ndi_recv, a)

            if args.debug_video:
                try:
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    break
                if key == ord("q") or key == 27:
                    break
                elif key == ord("+") or key == ord("="):
                    scale_factor = min(scale_factor + 0.1, 2.0)
                elif key == ord("-") or key == ord("_"):
                    scale_factor = max(scale_factor - 0.1, 0.2)
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        if ndi_recv:
            ndi.recv_destroy(ndi_recv)
        ndi.destroy()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        server_socket.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
