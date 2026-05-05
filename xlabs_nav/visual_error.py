from __future__ import annotations

import math
from typing import Any

import numpy as np


def _mean_dist_to_centroid(pts: np.ndarray) -> float:
    if pts.shape[0] == 0:
        return 0.0
    c = np.mean(pts, axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    return float(np.mean(d))


def estimate_visual_error(
    match_result: dict[str, Any],
    image_width: int,
    image_height: int,
    min_inliers: int,
    min_inlier_ratio: float,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """
    Image-plane error from inlier correspondences (reference minus current).
    dx = mean(ref_x - cur_x): positive means matched features lie farther right in the reference
    image than in the current image. Servo uses horizontal control error -dx_norm for Unity.
    scale_error = spread_reference / spread_current - 1
    """
    empty = {
        "valid": False,
        "dx": 0.0,
        "dy": 0.0,
        "dx_norm": 0.0,
        "dy_norm": 0.0,
        "scale_error": 0.0,
        "error_norm": 0.0,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "reason": "",
    }

    if not match_result.get("valid"):
        return {**empty, "reason": "match_failed"}

    mask = match_result.get("inlier_mask")
    cur = match_result.get("current_points")
    ref = match_result.get("reference_points")
    if mask is None or cur is None or ref is None:
        return {**empty, "reason": "missing_data"}

    cur_i = cur[mask]
    ref_i = ref[mask]
    num_inliers = int(cur_i.shape[0])
    ratio = float(match_result.get("inlier_ratio", 0.0))

    if num_inliers < min_inliers or ratio < min_inlier_ratio:
        return {
            **empty,
            "num_inliers": num_inliers,
            "inlier_ratio": ratio,
            "reason": "low_quality",
        }

    dx = float(np.mean(ref_i[:, 0] - cur_i[:, 0]))
    dy = float(np.mean(ref_i[:, 1] - cur_i[:, 1]))

    spread_c = _mean_dist_to_centroid(cur_i)
    spread_r = _mean_dist_to_centroid(ref_i)
    if spread_c < epsilon:
        return {
            **empty,
            "dx": dx,
            "dy": dy,
            "num_inliers": num_inliers,
            "inlier_ratio": ratio,
            "reason": "degenerate_spread",
        }

    scale_error = spread_r / spread_c - 1.0

    dx_norm = dx / max(image_width, 1)
    dy_norm = dy / max(image_height, 1)
    error_norm = math.sqrt(dx_norm**2 + dy_norm**2 + scale_error**2)

    return {
        "valid": True,
        "dx": dx,
        "dy": dy,
        "dx_norm": dx_norm,
        "dy_norm": dy_norm,
        "scale_error": float(scale_error),
        "error_norm": float(error_norm),
        "num_inliers": num_inliers,
        "inlier_ratio": ratio,
        "reason": "ok",
    }
