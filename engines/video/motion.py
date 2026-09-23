# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import cv2
import numpy as np
from typing import List, Optional


def _mask_box(frame: np.ndarray, box: Optional[List[float]]) -> np.ndarray:
    if not box:
        return frame

    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x1 = max(0, int(x * w))
    y1 = max(0, int(y * h))
    x2 = min(w, int((x + bw) * w))
    y2 = min(h, int((y + bh) * h))

    masked = frame.copy()
    if x2 > x1 and y2 > y1:
        masked[y1:y2, x1:x2] = 0
    return masked


def prepare_motion_frame(frame_img: np.ndarray, facecam_box: Optional[List[float]] = None) -> np.ndarray:
    """Prepare a small grayscale frame for deterministic motion scoring."""
    masked = _mask_box(frame_img, facecam_box)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def compute_motion_score(previous_frame: Optional[np.ndarray], current_frame: np.ndarray) -> float:
    """Return normalized frame-difference motion in the range [0, 1]."""
    if previous_frame is None or current_frame is None:
        return 0.0

    diff = cv2.absdiff(previous_frame, current_frame)
    mean_delta = float(np.mean(diff)) / 255.0
    return max(0.0, min(1.0, mean_delta * 3.0))
