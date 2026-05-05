"""
Record reference keyframes from the NDI stream (standalone; opens its own NDI receiver).

Usage (from repository root):
  python scripts/record_keyframes.py

Prefer saving from Sample_Drone_Interface.py (SPACE) while flying — only one NDI
receiver should run against Unity, or the feed can stall / feel unresponsive.

Controls:
  SPACE — save current frame as next keyframe (kf_001.png, kf_002.png, ...)
  q / ESC — quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import NDIlib as ndi
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xlabs_nav.frame_utils import bgr_from_ndi
from xlabs_nav.keyframe_io import append_keyframe_png, load_keyframe_manifest


def _draw_overlay(frame_bgr: np.ndarray, saved_count: int) -> np.ndarray:
    out = frame_bgr.copy()
    y = 28
    for line in (
        "SPACE: save keyframe",
        f"Saved keyframes: {saved_count}",
        "(Use Sample_Drone_Interface SPACE when flying; avoid two NDI apps.)",
    ):
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 26
    return out


def _ndi_setup() -> object | None:
    if not ndi.initialize():
        print("ERROR: NDI initialize failed.")
        return None

    ndi_find = ndi.find_create_v2()
    if ndi_find is None:
        ndi.destroy()
        return None

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
        print("WARNING: No NDI sources found.")

    if not selected_source:
        ndi.find_destroy(ndi_find)
        ndi.destroy()
        return None

    ndi_recv_create = ndi.RecvCreateV3()
    ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    ndi_recv = ndi.recv_create_v3(ndi_recv_create)
    ndi.recv_connect(ndi_recv, selected_source)
    ndi.find_destroy(ndi_find)
    return ndi_recv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record keyframes from NDI into data/keyframes/")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "keyframes" / "keyframes.json",
        help="Path to keyframes.json (default: data/keyframes/keyframes.json under repo root)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "keyframes",
        help="Directory for PNG files (default: data/keyframes under repo root)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    keyframes_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    keyframes_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries = load_keyframe_manifest(manifest_path)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: Invalid manifest {manifest_path}: {e}")
        return 1

    window_name = "Keyframe Recording (NDI)"
    print("--- KEYFRAME RECORDING (standalone NDI) ---")
    print("If Sample_Drone_Interface.py is running, use SPACE there instead — two NDI receivers often conflict.")
    print("SPACE: save keyframe | q / ESC: quit")

    ndi_recv = _ndi_setup()
    if ndi_recv is None:
        print("ERROR: No NDI receiver — cannot record.")
        return 1

    try:
        while True:
            t, v, a, _ = ndi.recv_capture_v2(ndi_recv, 1000)
            if t == ndi.FRAME_TYPE_VIDEO:
                frame = np.copy(v.data)
                bgr = bgr_from_ndi(frame)
                disp = _draw_overlay(bgr, len(entries))

                try:
                    cv2.imshow(window_name, disp)
                except cv2.error as e:
                    if "not implemented" in str(e).lower():
                        print("ERROR: OpenCV GUI unavailable (install opencv-python, not headless).")
                        ndi.recv_free_video_v2(ndi_recv, v)
                        return 1
                    raise

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    ndi.recv_free_video_v2(ndi_recv, v)
                    break
                if key == ord(" "):
                    entries, saved_path, err = append_keyframe_png(
                        bgr,
                        manifest_path=manifest_path,
                        keyframes_dir=keyframes_dir,
                        entries=entries,
                    )
                    if err:
                        print(f"ERROR: {err}")
                    elif saved_path is not None:
                        print(f"Saved {saved_path} ({len(entries)} keyframe(s) in manifest)")

                ndi.recv_free_video_v2(ndi_recv, v)
            elif t == ndi.FRAME_TYPE_AUDIO:
                ndi.recv_free_audio_v2(ndi_recv, a)

    except KeyboardInterrupt:
        pass
    finally:
        ndi.recv_destroy(ndi_recv)
        ndi.destroy()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
