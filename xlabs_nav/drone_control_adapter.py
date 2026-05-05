from __future__ import annotations

from typing import Any, MutableMapping

from xlabs_nav.servo_controller import clamp


def apply_servo_command(control_state: MutableMapping[str, Any], command: dict[str, float]) -> None:
    """
    Map KBVS command dict to Unity control_state (same fields as Sample_Drone_Interface).

    Signs are positive-passthrough: cmd["yaw"] > 0 → drone turns right, matching the
    debug arrow direction. Confirmed on Xtend Unity build (yaw and strafe not negated).
    """
    control_state["autopilot"] = True

    forward = float(command.get("forward", 0.0))
    yaw = float(command.get("yaw", 0.0))
    strafe = float(command.get("strafe", 0.0))
    altitude = float(command.get("altitude", 0.0))

    if forward >= 0:
        control_state["trigger"] = clamp(0.0, forward, 1.0)
        control_state["reverse"] = 0.0
    else:
        control_state["trigger"] = 0.0
        control_state["reverse"] = clamp(0.0, -forward, 1.0)

    control_state["yaw"] = clamp(-1.0, yaw, 1.0)
    control_state["joy_horizontal"] = clamp(-1.0, strafe, 1.0)
    control_state["joy_vertical"] = clamp(-1.0, altitude, 1.0)


def stop_drone(control_state: MutableMapping[str, Any]) -> None:
    control_state["trigger"] = 0.0
    control_state["reverse"] = 0.0
    control_state["yaw"] = 0.0
    control_state["pitch"] = 0.0
    control_state["joy_horizontal"] = 0
    control_state["joy_vertical"] = 0
