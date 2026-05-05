from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_BGR_GREEN = (0, 255, 0)
_BGR_RED = (0, 0, 255)
_BGR_CYAN = (255, 255, 0)
_BGR_WHITE = (255, 255, 255)
_BGR_YELLOW = (0, 255, 255)
_BGR_MAGENTA = (255, 0, 255)


def _outline_text(img: np.ndarray, text: str, org: tuple[int, int], fg: tuple[int, int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)


def _command_hints(fwd: float, yaw: float, strafe: float, alt: float) -> list[str]:
    lines: list[str] = []
    if abs(fwd) > 0.05:
        lines.append("FORWARD" if fwd > 0 else "REVERSE")
    if abs(yaw) > 0.05:
        lines.append("YAW RIGHT" if yaw > 0 else "YAW LEFT")
    if abs(strafe) > 0.05:
        lines.append("STRAFE RIGHT" if strafe > 0 else "STRAFE LEFT")
    if abs(alt) > 0.05:
        lines.append("ALTITUDE DOWN" if alt > 0 else "ALTITUDE UP")
    return lines


def _draw_yolo_overlay(out: np.ndarray, yolo_info: dict[str, Any]) -> None:
    """Draw YOLO bbox, class+conf label, and centers."""
    h, w = out.shape[:2]
    img_cx = int(w / 2)
    img_cy = int(h / 2)

    cv2.drawMarker(
        out,
        (img_cx, img_cy),
        _BGR_WHITE,
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    bbox = yolo_info.get("bbox")
    if not bbox:
        return

    centered = bool(yolo_info.get("centered", False))
    box_color = _BGR_CYAN if centered else _BGR_YELLOW
    stale = bool(yolo_info.get("stale", False))

    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

    cls_name = str(yolo_info.get("class_name", ""))
    conf = float(yolo_info.get("confidence", 0.0))
    label = f"{cls_name} {conf:.2f}"
    if stale:
        label += " [stale]"
    label_org = (x1, max(0, y1 - 6))
    _outline_text(out, label, label_org, box_color)

    bbox_center = yolo_info.get("bbox_center")
    if bbox_center:
        bcx = int(round(float(bbox_center[0])))
        bcy = int(round(float(bbox_center[1])))
        cv2.circle(out, (bcx, bcy), 5, _BGR_MAGENTA, -1, lineType=cv2.LINE_AA)


def render_debug_overlay(
    frame_bgr: np.ndarray,
    debug_info: dict[str, Any],
    overlay_cfg: dict[str, Any],
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    max_kp = int(overlay_cfg.get("max_keypoints_drawn", 600))
    max_lines = int(overlay_cfg.get("max_draw_matches", 200))
    thumb_w = int(overlay_cfg.get("thumbnail_width", 240))

    kps = debug_info.get("current_keypoints") or []
    if len(kps) > 0:
        step = max(1, len(kps) // max(1, max_kp))
        for i in range(0, len(kps), step):
            kp = kps[i]
            pt = (int(round(kp.pt[0])), int(round(kp.pt[1])))
            cv2.circle(out, pt, 2, _BGR_GREEN, -1, lineType=cv2.LINE_AA)

    ref_img = debug_info.get("reference_bgr")
    cur_m = debug_info.get("match_cur_xy")
    ref_m = debug_info.get("match_ref_xy")
    inl = debug_info.get("match_is_inlier")

    rh, rw = (ref_img.shape[:2] if ref_img is not None else (h, w))
    sx = w / max(float(rw), 1.0)
    sy = h / max(float(rh), 1.0)

    if cur_m is not None and ref_m is not None and inl is not None and len(cur_m) > 0:
        n = min(len(cur_m), max_lines)
        for i in range(n):
            p1 = (int(round(cur_m[i, 0])), int(round(cur_m[i, 1])))
            qx = int(round(float(ref_m[i, 0]) * sx))
            qy = int(round(float(ref_m[i, 1]) * sy))
            p2 = (qx, qy)
            col = _BGR_GREEN if bool(inl[i]) else _BGR_RED
            cv2.line(out, p1, p2, col, 1, cv2.LINE_AA)

    if ref_img is not None:
        tw = min(thumb_w, w // 3)
        ih, iw = ref_img.shape[:2]
        th = max(1, int(round(ih * (tw / max(iw, 1)))))
        thumb = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)
        rkps = debug_info.get("reference_keypoints") or []
        step_t = max(1, len(rkps) // 150)
        for i in range(0, len(rkps), step_t):
            kp = rkps[i]
            tx = int(round(kp.pt[0] * tw / max(iw, 1)))
            ty = int(round(kp.pt[1] * th / max(ih, 1)))
            cv2.circle(thumb, (tx, ty), 2, _BGR_GREEN, -1, lineType=cv2.LINE_AA)

        x0 = w - tw - 8
        y0 = h - th - 8
        if x0 >= 0 and y0 >= 0:
            out[y0 : y0 + th, x0 : x0 + tw] = thumb
            cv2.rectangle(out, (x0, y0), (x0 + tw, y0 + th), _BGR_WHITE, 1)

    cx = int(w / 2)
    cy = int(h / 2)
    dxn = float(debug_info.get("dx_norm", 0.0))
    dyn = float(debug_info.get("dy_norm", 0.0))
    err_px = float(np.hypot(dxn * w, dyn * h))
    if err_px > 2.0:
        # Horizontal: negate dx so arrow matches commanded correction (see ServoPidController e_yaw).
        ex = int(round(cx - dxn * w * 0.35))
        ey = int(round(cy + dyn * h * 0.35))
        cv2.arrowedLine(out, (cx, cy), (ex, ey), _BGR_CYAN, 2, tipLength=0.25)

    yolo_info = debug_info.get("yolo")
    if isinstance(yolo_info, dict):
        _draw_yolo_overlay(out, yolo_info)

    y_left = 22
    prog = debug_info.get("keyframe_progress") or (0, 1)
    kid = debug_info.get("active_keyframe_id", "---")
    lines_left = [
        f"state: {debug_info.get('state', '')}",
        f"Active keyframe: {kid}",
        f"Keyframe progress: {prog[0]} / {prog[1]}",
    ]
    sk = debug_info.get("selection_keyframe_index", -1)
    if sk >= 0:
        lines_left.append(f"Selection idx: {sk}  best_ratio: {float(debug_info.get('selection_best_ratio', 0.0)):.2f}")
        if debug_info.get("keyframe_switched"):
            lines_left.append("Keyframe re-localized")
    lines_left.extend(
        [
            f"num_matches: {debug_info.get('num_matches', 0)}",
            f"num_inliers: {debug_info.get('num_inliers', 0)}",
            f"inlier_ratio: {float(debug_info.get('inlier_ratio', 0.0)):.2f}",
            f"confidence: {debug_info.get('confidence', '')}",
        ]
    )
    if debug_info.get("forward_blocked"):
        lines_left.append("forward command blocked")
        lines_left.append(f"recovery: {debug_info.get('recovery_reason', '')}")

    state_text = debug_info.get("state", "")
    if state_text == "RECOVERY":
        state_color = _BGR_RED
    elif state_text == "TARGET_CENTERED":
        state_color = _BGR_CYAN
    elif state_text == "YOLO_TARGET_TRACKING":
        state_color = _BGR_YELLOW
    else:
        state_color = _BGR_GREEN

    for t in lines_left:
        _outline_text(out, t, (10, y_left), state_color)
        y_left += 22

    y_right = 22
    block_r = [
        f"dx_norm: {float(debug_info.get('dx_norm', 0.0)):.3f}",
        f"dy_norm: {float(debug_info.get('dy_norm', 0.0)):.3f}",
        f"scale_error: {float(debug_info.get('scale_error', 0.0)):.3f}",
        f"error_norm: {float(debug_info.get('error_norm', 0.0)):.3f}",
        f"err_ema (advance): {float(debug_info.get('error_norm_ema', 0.0)):.3f}",
        f"stable_frames: {int(debug_info.get('convergence_stable_frames', 0))}",
        "---",
        f"forward: {float(debug_info.get('forward', 0.0)):.3f}",
        f"yaw: {float(debug_info.get('yaw', 0.0)):.3f}",
        f"strafe: {float(debug_info.get('strafe', 0.0)):.3f}",
        f"altitude: {float(debug_info.get('altitude', 0.0)):.3f}",
    ]
    fwd = float(debug_info.get("forward", 0.0))
    yaw = float(debug_info.get("yaw", 0.0))
    strafe = float(debug_info.get("strafe", 0.0))
    alt = float(debug_info.get("altitude", 0.0))
    for hint in _command_hints(fwd, yaw, strafe, alt):
        block_r.append(hint)

    if isinstance(yolo_info, dict):
        block_r.append("---")
        block_r.append(f"YOLO mode: {yolo_info.get('mode', 'LOST')}")
        cls_name = str(yolo_info.get("class_name", ""))
        conf = float(yolo_info.get("confidence", 0.0))
        if cls_name:
            block_r.append(f"YOLO class: {cls_name}  conf: {conf:.2f}")
        bc = yolo_info.get("bbox_center")
        if bc:
            block_r.append(f"bbox_center: ({float(bc[0]):.0f}, {float(bc[1]):.0f})")
        ic = yolo_info.get("image_center")
        if ic:
            block_r.append(f"image_center: ({float(ic[0]):.0f}, {float(ic[1]):.0f})")
        block_r.append(
            f"error_x: {float(yolo_info.get('error_x_px', 0.0)):.1f} px  "
            f"({float(yolo_info.get('error_x_norm', 0.0)):+.3f})"
        )
        block_r.append(
            f"error_y: {float(yolo_info.get('error_y_px', 0.0)):.1f} px  "
            f"({float(yolo_info.get('error_y_norm', 0.0)):+.3f})"
        )
        block_r.append(f"lost_count: {int(yolo_info.get('lost_count', 0))}")
        if bool(yolo_info.get("stale", False)):
            block_r.append("(stale detection)")

    x_col = max(w - 340, 160)
    for t in block_r:
        if t == "---":
            y_right += 10
            continue
        _outline_text(out, t, (x_col, y_right), _BGR_WHITE)
        y_right += 22

    return out
