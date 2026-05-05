from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _orb_create(cfg: dict[str, Any]) -> cv2.ORB:
    ocfg = cfg["orb"]
    return cv2.ORB_create(
        nfeatures=int(ocfg["nfeatures"]),
        scaleFactor=float(ocfg["scale_factor"]),
        nlevels=int(ocfg["nlevels"]),
        edgeThreshold=int(ocfg["edge_threshold"]),
        firstLevel=int(ocfg["first_level"]),
        WTA_K=int(ocfg["WTA_K"]),
        scoreType=int(ocfg["score_type"]),
        patchSize=int(ocfg["patch_size"]),
        fastThreshold=int(ocfg["fast_threshold"]),
    )


def extract_orb_features(bgr: np.ndarray, cfg: dict[str, Any]) -> tuple[list[Any], np.ndarray | None, tuple[int, int]]:
    """ORB detect+compute once per image. Returns (keypoints, descriptors_or_None, (H,W))."""
    orb = _orb_create(cfg)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    h, w = bgr.shape[:2]
    return (kp or []), des, (h, w)


def _empty_match_out(
    kp_cur: list[Any],
    kp_ref: list[Any],
    cur_hw: tuple[int, int],
    ref_hw: tuple[int, int],
) -> dict[str, Any]:
    return {
        "valid": False,
        "current_keypoints": kp_cur,
        "reference_keypoints": kp_ref,
        "current_points": np.zeros((0, 2), dtype=np.float32),
        "reference_points": np.zeros((0, 2), dtype=np.float32),
        "matches": [],
        "num_matches": 0,
        "homography": None,
        "inlier_mask": None,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "reference_shape_hw": ref_hw,
        "current_shape_hw": cur_hw,
    }


def match_orb_features(
    kp_cur: list[Any],
    des_cur: np.ndarray | None,
    cur_hw: tuple[int, int],
    kp_ref: list[Any],
    des_ref: np.ndarray | None,
    ref_hw: tuple[int, int],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """BFMatcher + homography RANSAC from precomputed ORB features (current vs reference)."""
    mcfg = cfg["matching"]
    out = _empty_match_out(kp_cur, kp_ref, cur_hw, ref_hw)

    if des_cur is None or des_ref is None or len(kp_cur) == 0 or len(kp_ref) == 0:
        return out

    norm = cv2.NORM_HAMMING
    cross = bool(mcfg.get("bf_cross_check", True))
    bf = cv2.BFMatcher(norm, crossCheck=cross)
    matches = bf.match(des_cur, des_ref)
    max_dist = float(mcfg.get("max_match_distance", 80))
    matches = [m for m in matches if m.distance <= max_dist]
    matches = sorted(matches, key=lambda m: m.distance)

    if len(matches) == 0:
        return out

    pts_cur = np.float32([kp_cur[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts_cur,
        pts_ref,
        method=int(mcfg.get("ransac_method", cv2.RANSAC)),
        ransacReprojThreshold=float(mcfg.get("ransac_reproj_threshold", 5.0)),
    )

    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(matches), dtype=bool)
    num_inliers = int(np.sum(inlier_mask))
    inlier_ratio = num_inliers / max(len(matches), 1)

    out.update(
        valid=True,
        current_points=pts_cur.reshape(-1, 2),
        reference_points=pts_ref.reshape(-1, 2),
        matches=matches,
        num_matches=len(matches),
        homography=H,
        inlier_mask=inlier_mask,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
    )
    return out


def match_orb(
    current_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """ORB + BFMatcher + homography RANSAC. Returns points and inlier mask."""
    kp_cur, des_cur, cur_hw = extract_orb_features(current_bgr, cfg)
    kp_ref, des_ref, ref_hw = extract_orb_features(reference_bgr, cfg)
    return match_orb_features(kp_cur, des_cur, cur_hw, kp_ref, des_ref, ref_hw, cfg)
