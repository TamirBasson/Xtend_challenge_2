"""
YOLO-based target tracker.

Wraps an Ultralytics YOLO model and produces visual-servoing commands that
center the drone on the highest-confidence detection. Drives a small mode
machine consumed by AutonomyStack:

  TRACKING    — confident detection, error above tolerance → drive toward it.
  CENTERED    — bbox center within center_tolerance_px → zero commands.
  LOST        — no confident detection for >= lost_frames_tolerance frames.

Until the loss threshold is hit, the previous detection is reused (cached),
so brief occlusions don't immediately surrender control back to keyframe nav.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ultralytics is required for YoloTargetTracker. "
        "Install with `pip install ultralytics`."
    ) from exc


def _clamp(lo: float, x: float, hi: float) -> float:
    return max(lo, min(x, hi))


class YoloTargetTracker:
    """Detect the target with YOLO and emit servoing commands toward bbox center."""

    def __init__(self, yolo_cfg: dict[str, Any], repo_root: Path) -> None:
        self._cfg = dict(yolo_cfg or {})
        self._repo_root = Path(repo_root)

        model_path_str = str(self._cfg.get("model_path", "best.pt"))
        model_path = Path(model_path_str)
        if not model_path.is_absolute():
            model_path = (self._repo_root / model_path).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self._model_path = model_path
        device = self._cfg.get("device", "")
        device = str(device) if device not in (None, "") else None

        print(f"[yolo] loading model: {model_path}"
              f"{' (device=' + device + ')' if device else ''}")
        self._model = YOLO(str(model_path))
        self._device = device
        self._class_names: dict[int, str] = dict(self._model.names) if hasattr(self._model, "names") else {}

        self._conf_thr = float(self._cfg.get("confidence_threshold", 0.5))
        target_classes = self._cfg.get("target_classes")
        if target_classes is None:
            self._target_classes: set[str] | None = None
        else:
            cleaned = [str(c).strip() for c in target_classes if str(c).strip()]
            self._target_classes = set(cleaned) if cleaned else None

        self._imgsz = int(self._cfg.get("inference_imgsz", 640))
        self._infer_every_n = max(1, int(self._cfg.get("inference_every_n_frames", 1)))

        self._center_tol_px = float(self._cfg.get("center_tolerance_px", 40))
        self._lost_tol = max(1, int(self._cfg.get("lost_frames_tolerance", 10)))

        self._k_yaw = float(self._cfg.get("K_yaw", 1.5))
        self._kd_yaw = float(self._cfg.get("Kd_yaw", 0.0))
        self._k_alt = float(self._cfg.get("K_altitude", 1.0))
        self._kd_alt = float(self._cfg.get("Kd_altitude", 0.0))
        self._k_str = float(self._cfg.get("K_strafe", 0.0))
        self._kd_str = float(self._cfg.get("Kd_strafe", 0.0))

        self._forward_speed = float(self._cfg.get("forward_speed", 0.4))
        self._fwd_engage = float(self._cfg.get("forward_engage_norm_error", 0.15))

        self._max_yaw = float(self._cfg.get("max_yaw", 1.0))
        self._max_alt = float(self._cfg.get("max_altitude", 1.0))
        self._max_fwd = float(self._cfg.get("max_forward", 1.0))
        self._max_str = float(self._cfg.get("max_strafe", 1.0))
        self._cmd_duration = float(self._cfg.get("command_duration", 0.15))

        self._frame_idx = 0
        self._lost_count = 0
        self._prev_e_x: float | None = None
        self._prev_e_y: float | None = None
        self._last_detection: dict[str, Any] | None = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    def _zero_command(self) -> dict[str, float]:
        return {
            "forward": 0.0,
            "strafe": 0.0,
            "yaw": 0.0,
            "altitude": 0.0,
            "duration": self._cmd_duration,
        }

    def _empty_result(
        self,
        img_w: int,
        img_h: int,
        mode: str,
        reason: str = "",
    ) -> dict[str, Any]:
        return {
            "active": mode != "LOST",
            "mode": mode,
            "bbox": None,
            "class_id": -1,
            "class_name": "",
            "confidence": 0.0,
            "bbox_center": None,
            "image_center": (img_w / 2.0, img_h / 2.0),
            "error_x_px": 0.0,
            "error_y_px": 0.0,
            "error_x_norm": 0.0,
            "error_y_norm": 0.0,
            "error_norm": 0.0,
            "centered": False,
            "lost_count": self._lost_count,
            "stale": False,
            "reason": reason,
            "command": self._zero_command(),
        }

    def _select_best_detection(
        self,
        result: Any,
    ) -> tuple[tuple[float, float, float, float], int, float] | None:
        """Return (xyxy, cls_id, conf) for highest-conf detection passing filters."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            return None

        best_idx = -1
        best_conf = -1.0
        for i in range(xyxy.shape[0]):
            c = float(conf[i])
            if c < self._conf_thr:
                continue
            cid = int(cls[i])
            cname = self._class_names.get(cid, str(cid))
            if self._target_classes is not None and cname not in self._target_classes:
                continue
            if c > best_conf:
                best_conf = c
                best_idx = i

        if best_idx < 0:
            return None

        x1, y1, x2, y2 = (float(v) for v in xyxy[best_idx].tolist())
        cid = int(cls[best_idx])
        return (x1, y1, x2, y2), cid, best_conf

    def _compute_command(
        self,
        error_x_norm: float,
        error_y_norm: float,
        centered: bool,
    ) -> dict[str, float]:
        if centered:
            return self._zero_command()

        e_x = float(error_x_norm)
        e_y = float(error_y_norm)

        d_x = 0.0
        d_y = 0.0
        if self._prev_e_x is not None:
            d_x = e_x - self._prev_e_x
        if self._prev_e_y is not None:
            d_y = e_y - self._prev_e_y
        self._prev_e_x = e_x
        self._prev_e_y = e_y

        yaw = self._k_yaw * e_x + self._kd_yaw * d_x
        altitude = self._k_alt * e_y + self._kd_alt * d_y
        strafe = (self._k_str * e_x + self._kd_str * d_x) if self._k_str != 0.0 else 0.0

        err_norm = math.hypot(e_x, e_y)
        forward = self._forward_speed if err_norm < self._fwd_engage else 0.0

        yaw = _clamp(-self._max_yaw, yaw, self._max_yaw)
        altitude = _clamp(-self._max_alt, altitude, self._max_alt)
        strafe = _clamp(-self._max_str, strafe, self._max_str)
        forward = _clamp(-self._max_fwd, forward, self._max_fwd)

        return {
            "forward": forward,
            "strafe": strafe,
            "yaw": yaw,
            "altitude": altitude,
            "duration": self._cmd_duration,
        }

    def _build_result(
        self,
        img_w: int,
        img_h: int,
        bbox: tuple[float, float, float, float],
        class_id: int,
        confidence: float,
        stale: bool,
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = bbox
        bbox_cx = (x1 + x2) / 2.0
        bbox_cy = (y1 + y2) / 2.0
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0

        error_x_px = bbox_cx - img_cx
        error_y_px = bbox_cy - img_cy
        error_x_norm = error_x_px / max(img_w, 1)
        error_y_norm = error_y_px / max(img_h, 1)

        centered = (
            abs(error_x_px) < self._center_tol_px
            and abs(error_y_px) < self._center_tol_px
        )
        mode = "TARGET_CENTERED" if centered else "YOLO_TARGET_TRACKING"
        cmd = self._compute_command(error_x_norm, error_y_norm, centered)

        return {
            "active": True,
            "mode": mode,
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "class_id": int(class_id),
            "class_name": self._class_names.get(int(class_id), str(class_id)),
            "confidence": float(confidence),
            "bbox_center": (float(bbox_cx), float(bbox_cy)),
            "image_center": (float(img_cx), float(img_cy)),
            "error_x_px": float(error_x_px),
            "error_y_px": float(error_y_px),
            "error_x_norm": float(error_x_norm),
            "error_y_norm": float(error_y_norm),
            "error_norm": float(math.hypot(error_x_norm, error_y_norm)),
            "centered": bool(centered),
            "lost_count": int(self._lost_count),
            "stale": bool(stale),
            "reason": "ok",
            "command": cmd,
        }

    def step(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        self._frame_idx += 1
        h, w = frame_bgr.shape[:2]

        run_inference = (self._frame_idx % self._infer_every_n) == 1 or self._infer_every_n == 1
        detection: tuple[tuple[float, float, float, float], int, float] | None = None

        if run_inference:
            try:
                kwargs: dict[str, Any] = {
                    "imgsz": self._imgsz,
                    "conf": self._conf_thr,
                    "verbose": False,
                }
                if self._device is not None:
                    kwargs["device"] = self._device
                results = self._model.predict(frame_bgr, **kwargs)
            except Exception as e:
                self._lost_count += 1
                if self._last_detection is not None and self._lost_count < self._lost_tol:
                    bbox = self._last_detection["bbox"]
                    cid = self._last_detection["class_id"]
                    conf = self._last_detection["confidence"]
                    return self._build_result(w, h, bbox, cid, conf, stale=True)
                self._prev_e_x = None
                self._prev_e_y = None
                return self._empty_result(w, h, "LOST", reason=f"inference_error:{e}")

            if results:
                detection = self._select_best_detection(results[0])
        else:
            if self._last_detection is not None:
                bbox = self._last_detection["bbox"]
                cid = self._last_detection["class_id"]
                conf = self._last_detection["confidence"]
                return self._build_result(w, h, bbox, cid, conf, stale=True)

        if detection is None:
            self._lost_count += 1
            if self._last_detection is not None and self._lost_count < self._lost_tol:
                bbox = self._last_detection["bbox"]
                cid = self._last_detection["class_id"]
                conf = self._last_detection["confidence"]
                return self._build_result(w, h, bbox, cid, conf, stale=True)
            self._prev_e_x = None
            self._prev_e_y = None
            self._last_detection = None
            return self._empty_result(w, h, "LOST", reason="no_detection")

        bbox, cid, conf = detection
        self._lost_count = 0
        self._last_detection = {
            "bbox": bbox,
            "class_id": cid,
            "confidence": conf,
        }
        return self._build_result(w, h, bbox, cid, conf, stale=False)
