# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Shared box geometry helpers.

Boxes are normalized ``[x, y, w, h]`` (top-left origin, width/height). This is
the one IoU used across the vision engines (facecam tracking, layout
clustering) and the pipeline's stable-facecam selection — previously copied
three times with identical math.
"""

from __future__ import annotations


def iou(box1, box2) -> float:
    """Intersection-over-union of two ``[x, y, w, h]`` boxes. 0.0 when disjoint."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area <= 0:
        return 0.0

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    return inter_area / float(box1_area + box2_area - inter_area)
