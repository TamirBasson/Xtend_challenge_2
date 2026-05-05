from __future__ import annotations

import time
from typing import Any


def clamp(lo: float, x: float, hi: float) -> float:
    return max(lo, min(x, hi))


class ServoPidController:
    """
    PID on normalized visual errors (per axis), then clamp to Unity command limits.
    e_alt = -dy_norm (Unity joy_vertical positive = down).
    Horizontal: e_yaw = e_str = -dx_norm (aligned with debug arrow correction sense).
    apply_servo_command passes yaw/strafe through without negation (positive = right).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._i_yaw = 0.0
        self._i_alt = 0.0
        self._i_fwd = 0.0
        self._i_str = 0.0
        self._prev_e_yaw: float | None = None
        self._prev_e_alt: float | None = None
        self._prev_e_fwd: float | None = None
        self._prev_e_str: float | None = None
        self._t_prev: float | None = None

    def compute(
        self,
        visual_error: dict[str, Any],
        ctrl_cfg: dict[str, Any],
        *,
        recovery_mode: bool,
        recovery_yaw: float,
    ) -> dict[str, float]:
        duration = float(ctrl_cfg.get("command_duration", 0.15))

        if recovery_mode or not visual_error.get("valid"):
            self.reset()
            return {
                "forward": 0.0,
                "strafe": 0.0,
                "yaw": clamp(-1.0, recovery_yaw, 1.0),
                "altitude": 0.0,
                "duration": duration,
            }

        t = time.monotonic()
        dt = 1.0 / 60.0
        if self._t_prev is not None:
            dt = max(1e-4, min(t - self._t_prev, 0.25))
        self._t_prev = t

        dxn = float(visual_error.get("dx_norm", 0.0))
        dyn = float(visual_error.get("dy_norm", 0.0))
        scale_e = float(visual_error.get("scale_error", 0.0))
        e_yaw = -dxn
        e_alt = -dyn
        e_fwd = scale_e
        e_str = -dxn

        k_yaw = float(ctrl_cfg.get("K_yaw", 2.0))
        k_alt = float(ctrl_cfg.get("K_altitude", 2.0))
        k_fwd = float(ctrl_cfg.get("K_forward", 1.5))
        k_str = float(ctrl_cfg.get("K_strafe", 0.0))
        ki_yaw = float(ctrl_cfg.get("Ki_yaw", 0.0))
        ki_alt = float(ctrl_cfg.get("Ki_altitude", 0.0))
        ki_fwd = float(ctrl_cfg.get("Ki_forward", 0.0))
        ki_str = float(ctrl_cfg.get("Ki_strafe", 0.0))
        kd_yaw = float(ctrl_cfg.get("Kd_yaw", 0.0))
        kd_alt = float(ctrl_cfg.get("Kd_altitude", 0.0))
        kd_fwd = float(ctrl_cfg.get("Kd_forward", 0.0))
        kd_str = float(ctrl_cfg.get("Kd_strafe", 0.0))
        imax = float(ctrl_cfg.get("integral_max", 0.25))
        decay = float(ctrl_cfg.get("integral_decay", 0.0))

        self._i_yaw += e_yaw * dt
        self._i_alt += e_alt * dt
        self._i_fwd += e_fwd * dt
        self._i_str += e_str * dt
        self._i_yaw = clamp(-imax, self._i_yaw, imax)
        self._i_alt = clamp(-imax, self._i_alt, imax)
        self._i_fwd = clamp(-imax, self._i_fwd, imax)
        self._i_str = clamp(-imax, self._i_str, imax)
        if decay > 0.0:
            f = max(0.0, 1.0 - decay)
            self._i_yaw *= f
            self._i_alt *= f
            self._i_fwd *= f
            self._i_str *= f

        d_yaw = d_alt = d_fwd = d_str = 0.0
        if self._prev_e_yaw is not None:
            d_yaw = (e_yaw - self._prev_e_yaw) / dt
        if self._prev_e_alt is not None:
            d_alt = (e_alt - self._prev_e_alt) / dt
        if self._prev_e_fwd is not None:
            d_fwd = (e_fwd - self._prev_e_fwd) / dt
        if self._prev_e_str is not None:
            d_str = (e_str - self._prev_e_str) / dt
        self._prev_e_yaw = e_yaw
        self._prev_e_alt = e_alt
        self._prev_e_fwd = e_fwd
        self._prev_e_str = e_str

        yaw = k_yaw * e_yaw + ki_yaw * self._i_yaw + kd_yaw * d_yaw
        altitude = k_alt * e_alt + ki_alt * self._i_alt + kd_alt * d_alt
        forward = k_fwd * e_fwd + ki_fwd * self._i_fwd + kd_fwd * d_fwd
        strafe = k_str * e_str + ki_str * self._i_str + kd_str * d_str

        yaw = clamp(-float(ctrl_cfg.get("max_yaw", 1.0)), yaw, float(ctrl_cfg.get("max_yaw", 1.0)))
        altitude = clamp(
            -float(ctrl_cfg.get("max_altitude", 1.0)),
            altitude,
            float(ctrl_cfg.get("max_altitude", 1.0)),
        )
        forward = clamp(-float(ctrl_cfg.get("max_forward", 1.0)), forward, float(ctrl_cfg.get("max_forward", 1.0)))
        strafe = clamp(-float(ctrl_cfg.get("max_strafe", 1.0)), strafe, float(ctrl_cfg.get("max_strafe", 1.0)))

        return {
            "forward": forward,
            "strafe": strafe,
            "yaw": yaw,
            "altitude": altitude,
            "duration": duration,
        }
