import numpy as np
import cv2


def bgr_from_ndi(frame: np.ndarray) -> np.ndarray:
    """Convert NDI BGRX/BGRA frames to BGR for OpenCV processing."""
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
