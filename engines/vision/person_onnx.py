# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""YOLOX person detection over onnxruntime.

The permissively licensed replacement for the Ultralytics backend. Recall uses
object detection for exactly one thing -- locating people, to validate facecam
plates -- so this module implements that and nothing else. See
docs/DETECTOR_RELICENSE_PLAN.md for why the swap was necessary (Ultralytics is
AGPL-3.0, which a closed-source distribution cannot satisfy).

YOLOX is Apache-2.0 (Megvii-BaseDetection/YOLOX), as are the released weights.

The exported graph is RAW: it emits [1, N, 85] where the box terms are grid
offsets, not pixels, and no NMS has been applied. Both steps are done here.
Layout per row is [cx, cy, w, h, objectness, 80 class scores].
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np

from core.bundle_paths import get_models_dir
from core.device import get_ort_providers

# yolox_s over yolox_tiny: measured on the facecam replay, s recovers 3 of the
# 4 plates tiny moved (3605cb26 0.295 -> 0.938, 767e8a81 0.814 -> 0.864,
# 521dc15b equivalent on face containment) at 52.6 ms/frame against tiny's 33.0
# -- still no slower than the YOLO26n it replaces (54.5). INPUT_SIZE differs
# between them (640 vs 416) and is read from the graph, so either can be set
# here without code changes.
MODEL_NAME = os.environ.get("RECALL_PERSON_MODEL", "yolox_s.onnx")

# Fallback only. The real value is READ FROM THE MODEL in _ensure_session():
# yolox_tiny is 416x416 (3549 rows) and yolox_s is 640x640 (8400 rows), and a
# hardcoded size next to a swappable model file is the exact failure this
# module is most exposed to -- the decode would still produce well-formed
# boxes, just wrong ones, with nothing in the logs.
#
# The letterbox below preserves aspect ratio, so a 16:9 gameplay frame is
# padded rather than squashed; squashing changes a person's aspect and costs
# recall on the tall, narrow boxes a facecam produces.
INPUT_SIZE = (416, 416)

# Feature-map strides the export concatenates, in order. 416/8, 416/16, 416/32
# squared and summed give the 3549 rows the graph returns.
STRIDES = (8, 16, 32)

# YOLOX pads with 114 (the training-time pad value). Any other value shifts the
# input distribution at the frame edges, which is exactly where corner-anchored
# facecams live.
PAD_VALUE = 114

PERSON_CLASS = 0  # COCO
NMS_IOU = 0.45

_session = None
_grids = None
_expanded_strides = None


def _resolve_model_path() -> str:
    if os.path.isabs(MODEL_NAME):
        return MODEL_NAME
    return os.path.join(get_models_dir(), "vision", MODEL_NAME)


def _ensure_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        path = _resolve_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Person-detection model not found at {path}. Fetch it with "
                "scripts/fetch_person_model.py."
            )
        _session = ort.InferenceSession(path, providers=get_ort_providers())
        _adopt_input_size(_session)
    return _session


def _adopt_input_size(session) -> None:
    """Take the letterbox resolution from the graph, not from a constant."""
    global INPUT_SIZE, _grids, _expanded_strides
    shape = session.get_inputs()[0].shape
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int):
        if (height, width) != INPUT_SIZE:
            INPUT_SIZE = (height, width)
            # Grids are derived from INPUT_SIZE, so they must be rebuilt or the
            # decode silently misaligns against the new row count.
            _grids = None
            _expanded_strides = None


def ensure_session():
    """Public warm-up: load the session now rather than on the first frame."""
    return _ensure_session()


def reset_session() -> None:
    """Drop the cached session (tests, and device changes)."""
    global _session
    _session = None


def _build_grids():
    """Grid offsets and per-anchor strides, in the graph's concatenation order.

    Cached because they depend only on INPUT_SIZE, and rebuilding them for every
    sampled frame would cost more than the inference at 1 fps over a full VOD.
    """
    global _grids, _expanded_strides
    if _grids is None:
        grids = []
        strides = []
        for stride in STRIDES:
            h = INPUT_SIZE[0] // stride
            w = INPUT_SIZE[1] // stride
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            strides.append(np.full((1, grid.shape[1], 1), stride))
        _grids = np.concatenate(grids, 1).astype(np.float32)
        _expanded_strides = np.concatenate(strides, 1).astype(np.float32)
    return _grids, _expanded_strides


def _letterbox(image: np.ndarray):
    """Resize preserving aspect into a PAD_VALUE canvas. Returns (chw, ratio).

    YOLOX anchors the image at the top-left of the canvas rather than centring
    it, so undoing the transform is a single divide by `ratio` with no offset.
    """
    padded = np.full((INPUT_SIZE[0], INPUT_SIZE[1], 3), PAD_VALUE, dtype=np.uint8)
    height, width = image.shape[:2]
    ratio = min(INPUT_SIZE[0] / height, INPUT_SIZE[1] / width)
    resized = cv2.resize(
        image,
        (int(width * ratio), int(height * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded[: resized.shape[0], : resized.shape[1]] = resized
    # BGR->CHW, float32, NO /255 scaling: the released YOLOX models take raw
    # 0-255 values. Dividing here silently halves recall.
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Greedy NMS. Returns indices into `boxes`, highest score first."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[best], x1[rest])
        yy1 = np.maximum(y1[best], y1[rest])
        xx2 = np.minimum(x2[best], x2[rest])
        yy2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def detect_persons(image: Optional[np.ndarray], conf: float) -> List[List[float]]:
    """Person boxes in `image` as pixel [x1, y1, x2, y2] in that image's frame.

    Signature-compatible with the Ultralytics-backed implementation it replaces.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return []

    # Session FIRST: it adopts INPUT_SIZE from the graph, and _letterbox reads
    # INPUT_SIZE. Letterboxing first would size the blob from a stale constant.
    session = _ensure_session()
    blob, ratio = _letterbox(image)
    raw = session.run(None, {session.get_inputs()[0].name: blob[None]})[0][0]

    grids, strides = _build_grids()
    centers = (raw[:, :2] + grids[0]) * strides[0]
    sizes = np.exp(raw[:, 2:4]) * strides[0]

    # objectness * class score, person column only.
    scores = raw[:, 4] * raw[:, 5 + PERSON_CLASS]
    hits = scores >= conf
    if not hits.any():
        return []

    centers, sizes, scores = centers[hits], sizes[hits], scores[hits]
    half = sizes / 2.0
    # Back to source-image pixels: undo the letterbox scale. No offset term --
    # _letterbox anchors at the top-left.
    boxes = np.concatenate([centers - half, centers + half], axis=1) / ratio

    keep = _nms(boxes, scores, NMS_IOU)
    return [boxes[i].tolist() for i in keep]
