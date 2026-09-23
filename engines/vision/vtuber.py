# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Creator-confirmed VTuber cutout-mask preparation.

The top-center cutout is a destructive change to the creator's composition, so
image, face, and title heuristics are never allowed to enable it. A title may
suggest the option in the app, but the scan must carry an explicit creator
confirmation. Unknown always means normal facecam framing.

The alpha mask is prepared once from the stable source plate with GrabCut.  It
contains no rendered pixels: export keeps reading the live avatar pixels from
the VOD and uses only the saved alpha channel, so mouth/eye animation survives.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from core.geometry import iou


_VTUBER_TITLE = re.compile(
    r"(?:^|[^a-z0-9])(?:v[ -]?tuber|png[ -]?tuber|virtual\s+(?:streamer|youtuber))(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

OVERLAY_WIDTH = round(500.0 / 1080.0, 6)
OVERLAY_TOP = round(-28.0 / 1920.0, 6)
OVERLAY_LOWER_FADE = round(150.0 / 560.0, 6)
MATCH_IOU = 0.45
AVATAR_CLUSTER_IOU = 0.35


@dataclass
class ConfirmedAvatarLayout:
    """Persistent avatar geometry used only after creator confirmation."""

    box: List[float]
    timestamps: List[float]
    source_crop: List[float]
    support: int


def title_declares_vtuber(title: Any) -> bool:
    """Advisory UI hint only; this never authorizes VTuber composition."""
    return bool(_VTUBER_TITLE.search(str(title or "")))


def _safe_box(box: Sequence[float]) -> List[float]:
    x, y, w, h = [float(value) for value in box[:4]]
    x = max(0.0, min(0.99, x))
    y = max(0.0, min(0.99, y))
    w = max(0.01, min(1.0 - x, w))
    h = max(0.01, min(1.0 - y, h))
    return [x, y, w, h]


def source_crop_for_plate(
    plate: Sequence[float], source_width: int, source_height: int,
) -> List[float]:
    """Tight square crop for an edge-anchored 2D avatar plate.

    Facecam refinement deliberately keeps a little canvas around a plate.  The
    approved Skyrim proof trims roughly 10% from the loose leading edges while
    retaining the stream edge the avatar is anchored to.  Keeping this as a
    source crop (rather than changing the detected plate) preserves all normal
    facecam decisions and makes the mask and live-pixel crop identical.
    """
    x, y, w, h = _safe_box(plate)
    pw = w * max(1, int(source_width))
    ph = h * max(1, int(source_height))
    side = max(8.0, min(pw * 0.90, ph * 0.88))
    cx = (x + w / 2.0) * source_width
    cy = (y + h / 2.0) * source_height

    # Retain the nearest stream edges. VTuber sprites are normally attached to
    # a corner; this reproduces the reviewed bottom-right crop but also works
    # for left/top authored canvases.
    if cx >= source_width / 2.0:
        px = (x + w) * source_width - side
    else:
        px = x * source_width
    if cy >= source_height / 2.0:
        py = (y + h) * source_height - side
    else:
        py = y * source_height

    px = max(0.0, min(source_width - side, px))
    py = max(0.0, min(source_height - side, py))
    return [
        round(px / source_width, 6),
        round(py / source_height, 6),
        round(side / source_width, 6),
        round(side / source_height, 6),
    ]


def source_crop_for_avatar_plate(
    plate: Sequence[float], source_width: int, source_height: int,
) -> List[float]:
    """Square crop that preserves a tall, edge-anchored Live2D avatar.

    The ordinary facecam crop deliberately trims a loose camera plate. A
    full-height avatar is different: its detected width is normally the tight
    dimension and its feet leave the frame, so a bottom-anchored square based
    on that width retains the hair and shoulders without pulling in chat.
    """
    x, y, w, h = _safe_box(plate)
    pw = w * max(1, int(source_width))
    ph = h * max(1, int(source_height))
    side = max(8.0, min(pw * 0.99, ph * 0.96))
    cx = (x + w / 2.0) * source_width
    cy = (y + h / 2.0) * source_height
    px = (x + w) * source_width - side if cx >= source_width / 2.0 else x * source_width
    py = (y + h) * source_height - side if cy >= source_height / 2.0 else y * source_height
    px = max(0.0, min(source_width - side, px))
    py = max(0.0, min(source_height - side, py))
    return [
        round(px / source_width, 6),
        round(py / source_height, 6),
        round(side / source_width, 6),
        round(side / source_height, 6),
    ]


def detect_confirmed_avatar_layouts(
    unified_signals: Sequence[Any], frame_aspect: float = 16.0 / 9.0,
) -> List[ConfirmedAvatarLayout]:
    """Find a persistent avatar without letting a small stream pet win.

    This is intentionally not a general facecam detector and must only be used
    behind the explicit per-VOD VTuber confirmation. Live2D avatars are large,
    portrait-shaped, edge-anchored person tracks; decorative pets tend to be
    smaller panel contours. The separate lane preserves normal webcam behavior.
    """
    stamped = []
    vision_times = []
    for signal in unified_signals or []:
        vision = getattr(signal, "vision", None)
        if vision is None:
            continue
        timestamp = float(getattr(signal, "timestamp", 0.0) or 0.0)
        vision_times.append(timestamp)
        box = getattr(vision, "facecam_box", None)
        if box and len(box) >= 4:
            stamped.append((list(box[:4]), timestamp, getattr(vision, "facecam_source", None)))
    if not stamped:
        return []

    clusters: List[Dict[str, Any]] = []
    for box, timestamp, source in stamped:
        for cluster in clusters:
            if iou(box, cluster["anchor"]) >= AVATAR_CLUSTER_IOU:
                cluster["boxes"].append(box)
                cluster["times"].append(timestamp)
                cluster["sources"].append(source)
                break
        else:
            clusters.append({
                "anchor": box,
                "boxes": [box],
                "times": [timestamp],
                "sources": [source],
            })

    total = max(1, len(vision_times))
    timeline_span = max(1.0, max(vision_times) - min(vision_times)) if vision_times else 1.0
    min_support = max(12, int(math.ceil(total * 0.02)))
    candidates = []
    for cluster in clusters:
        support = len(cluster["boxes"])
        if support < min_support:
            continue
        median = [float(np.median([box[index] for box in cluster["boxes"]])) for index in range(4)]
        x, y, w, h = _safe_box(median)
        fallback_support = sum(source == "person_fallback" for source in cluster["sources"])
        fallback_share = fallback_support / support
        edge_anchored = x <= 0.06 or (x + w) >= 0.94
        pixel_aspect = (w / max(h, 1e-6)) * frame_aspect
        cluster_span = max(cluster["times"]) - min(cluster["times"])
        if not (
            fallback_share >= 0.70
            and edge_anchored
            and w * h >= 0.08
            and h >= 0.35
            and 0.55 <= pixel_aspect <= 1.20
            and cluster_span / timeline_span >= 0.35
        ):
            continue
        score = (w * h) * math.sqrt(support) * fallback_share
        candidates.append((score, median, list(cluster["times"]), support))

    if not candidates:
        return []
    _score, box, timestamps, support = max(candidates, key=lambda item: item[0])
    # Source dimensions are folded in later; this placeholder records that the
    # avatar-specific crop must be used rather than the ordinary camera trim.
    return [ConfirmedAvatarLayout(
        box=[round(value, 6) for value in box],
        timestamps=timestamps,
        source_crop=[],
        support=support,
    )]


def _frame_at(capture, timestamp: float):
    import cv2

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
    ok, frame = capture.read()
    return frame if ok else None


def _grabcut_alpha(frame, crop: Sequence[float]) -> Optional[np.ndarray]:
    import cv2

    height, width = frame.shape[:2]
    x, y, w, h = _safe_box(crop)
    x0, y0 = int(round(x * width)), int(round(y * height))
    x1 = int(round((x + w) * width))
    y1 = int(round((y + h) * height))
    roi = frame[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    if roi.size == 0 or min(roi.shape[:2]) < 32:
        return None

    mask = np.zeros(roi.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    inset = max(2, int(round(min(roi.shape[:2]) * 0.012)))
    rect = (inset, inset, roi.shape[1] - 2 * inset, roi.shape[0] - inset - 1)
    try:
        cv2.grabCut(
            roi, mask, rect, bg_model, fg_model, 7, cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0,
    ).astype(np.uint8)

    # Keep the largest coherent subject, then smooth only the boundary. Random
    # game texture selected by GrabCut is normally a small disconnected island.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    binary = np.where(labels == largest, 255, 0).astype(np.uint8)
    area = float(np.count_nonzero(binary)) / float(binary.size)
    ys, xs = np.where(binary > 0)
    if (
        area < 0.12 or area > 0.82 or not len(xs)
        or (xs.max() - xs.min()) < roi.shape[1] * 0.30
        or (ys.max() - ys.min()) < roi.shape[0] * 0.40
    ):
        return None
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return cv2.GaussianBlur(binary, (0, 0), 1.2)


def _layout_timestamps(layout: Any, duration: float) -> List[float]:
    values = sorted(float(value) for value in (getattr(layout, "timestamps", None) or []))
    if values:
        picks = [values[len(values) // 4], values[len(values) // 2], values[(len(values) * 3) // 4]]
        return list(dict.fromkeys(round(value, 3) for value in picks))
    duration = max(1.0, float(duration or 0.0))
    return [duration * 0.25, duration * 0.5, duration * 0.75]


def _prepare_mask(
    video_path: str,
    crop: Sequence[float],
    timestamps: Sequence[float],
    output_path: str,
    cancel_check=None,
) -> bool:
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return False
    frames, masks = [], []
    try:
        for timestamp in timestamps:
            if cancel_check:
                cancel_check()
            frame = _frame_at(capture, timestamp)
            if frame is None:
                continue
            alpha = _grabcut_alpha(frame, crop)
            if alpha is not None:
                frames.append(frame)
                masks.append(alpha)
    finally:
        capture.release()
    if not masks:
        return False

    # Majority consensus keeps a changing game background out of the static
    # plate while a stable PNG/Live2D silhouette survives. Slight dilation in
    # _grabcut_alpha leaves breathing room for mouth/eye motion.
    stack = np.stack([mask > 96 for mask in masks])
    alpha = (np.count_nonzero(stack, axis=0) >= math.ceil(len(masks) / 2)).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)

    frame = frames[len(frames) // 2]
    height, width = frame.shape[:2]
    x, y, w, h = _safe_box(crop)
    x0, y0 = int(round(x * width)), int(round(y * height))
    x1, y1 = int(round((x + w) * width)), int(round((y + h) * height))
    roi = frame[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    if roi.shape[:2] != alpha.shape[:2]:
        return False
    rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return bool(cv2.imwrite(output_path, rgba))


def prepare_vtuber_model(
    video_path: str,
    title: Any,
    layouts: Sequence[Any],
    duration: float,
    job_dir: str,
    confirmed: bool = False,
    cancel_check=None,
) -> Dict[str, Any]:
    """Build overlay specs only after an explicit per-session confirmation."""
    marker = title_declares_vtuber(title)
    model: Dict[str, Any] = {
        "version": 1,
        "status": "suggested" if marker else "off",
        "title_marker": marker,
        "evidence": [],
        "overlays": [],
    }
    if not confirmed:
        return model
    model["status"] = "confirmed_no_overlay"
    model["evidence"] = ["creator_confirmation"]
    if not layouts or not os.path.isfile(video_path):
        return model

    import cv2

    capture = cv2.VideoCapture(video_path)
    try:
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if source_width <= 0 or source_height <= 0:
        model["status"] = "probe_failed"
        return model

    mask_failed = 0
    for index, layout in enumerate(layouts):
        plate = list(getattr(layout, "box", None) or [])
        if len(plate) < 4:
            continue
        timestamps = _layout_timestamps(layout, duration)
        authored_crop = list(getattr(layout, "source_crop", None) or [])
        crop = authored_crop if len(authored_crop) >= 4 else (
            source_crop_for_avatar_plate(plate, source_width, source_height)
            if isinstance(layout, ConfirmedAvatarLayout)
            else source_crop_for_plate(plate, source_width, source_height)
        )
        mask_path = os.path.abspath(
            os.path.join(job_dir, f"vtuber_overlay_mask_{index}.png"),
        )
        if not _prepare_mask(
            video_path, crop, timestamps, mask_path, cancel_check=cancel_check,
        ):
            mask_failed += 1
            continue
        model["overlays"].append({
            "layout_index": index,
            "plate": [round(float(value), 6) for value in plate[:4]],
            "source_crop": crop,
            "mask_path": mask_path,
            "width": OVERLAY_WIDTH,
            "top": OVERLAY_TOP,
            "lower_fade": OVERLAY_LOWER_FADE,
            "verification": "creator_confirmation",
        })

    if model["overlays"]:
        model["status"] = "confirmed"
    elif mask_failed:
        model["status"] = "mask_failed"
    else:
        model["status"] = "no_avatar_plate"
    return model


def overlay_for_facecam(
    vtuber_model: Optional[Dict[str, Any]], facecam_box: Optional[Sequence[float]],
) -> Optional[Dict[str, Any]]:
    if (
        not isinstance(vtuber_model, dict)
        or vtuber_model.get("status") != "confirmed"
        or not facecam_box
    ):
        return None
    matches = [
        (iou(list(facecam_box), list(item.get("plate") or [])), item)
        for item in (vtuber_model.get("overlays") or [])
        if len(item.get("plate") or []) >= 4
    ]
    score, overlay = max(matches, default=(0.0, None), key=lambda pair: pair[0])
    if overlay is not None and score >= MATCH_IOU:
        return overlay
    overlays = list(vtuber_model.get("overlays") or [])
    # Once the creator has confirmed a single avatar track, an unrelated
    # camera-shaped widget (for example a stream pet) must not veto it.
    return overlays[0] if len(overlays) == 1 else None


def apply_overlay(layout: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Return a VTuber composition without mutating the normal layout."""
    resolved = dict(layout)
    resolved["type"] = "vtuber_overlay"
    resolved["facecam"] = list(overlay["source_crop"])
    source_crop = list(overlay["source_crop"])
    # A centered 9:16 gameplay crop can retain a few pixels of a large avatar
    # anchored on the left edge. Nudge gameplay away from the source avatar;
    # the keyed copy at the top remains the only visible instance.
    if source_crop[0] + source_crop[2] <= 0.5:
        resolved["focus_x"] = max(float(resolved.get("focus_x", 0.5)), 0.56)
    elif source_crop[0] >= 0.5:
        resolved["focus_x"] = min(float(resolved.get("focus_x", 0.5)), 0.48)
    resolved.pop("facecam_subject_top", None)
    resolved.pop("camera_cover", None)
    resolved["vtuber_overlay"] = {
        "mask_path": os.path.abspath(str(overlay["mask_path"])),
        "width": float(overlay.get("width", OVERLAY_WIDTH)),
        "top": float(overlay.get("top", OVERLAY_TOP)),
        "lower_fade": float(overlay.get("lower_fade", OVERLAY_LOWER_FADE)),
    }
    return resolved
