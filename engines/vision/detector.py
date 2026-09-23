# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import cv2
import numpy as np
from core.models.signal import VisionSignal
from core.device import get_torch_device
from core.geometry import iou as _calculate_iou
from engines.vision import person_onnx

# EMA weight on each new facecam detection when it keeps tracking the same plate
# (IoU >= 0.5). Lower = steadier box, less breathing from animated cam borders.
FACECAM_SMOOTHING = 0.4
# Expire a panel that is not refreshed. Without this, a face found during a
# full-camera intro can remain latched after an OBS scene switch.
FACECAM_MAX_MISSES = 3

# A fullscreen webcam scene is not a facecam *panel*: the streamer is the
# entire source frame.  Treating that scene as gameplay makes the ordinary
# 9:16 center crop slice away a creator who leans left/right.  This deliberately
# conservative geometry gate is used only when the normal facecam and authored
# gameplay detectors both returned None.  It recognizes the very large,
# bottom-anchored person box produced by a fullscreen camera, not an ordinary
# corner webcam or a standing game character.
FULLFRAME_PERSON_MIN_AREA = 0.30
FULLFRAME_PERSON_MIN_WIDTH = 0.55
FULLFRAME_PERSON_MIN_HEIGHT = 0.45
FULLFRAME_PERSON_MIN_BOTTOM = 0.88

# Module-level state for temporal tracking
_model = None
_DEVICE = get_torch_device()  # 'cuda' on an NVIDIA box, else 'cpu'
_facecam_history = []  # List of dicts: {'box': [x, y, w, h], 'frames_seen': int}
_stable_facecam = None  # No default fallback, starts as None
_stable_facecam_source = None
_facecam_misses = 0


def _ensure_model():
    """Warm the person-detection backend once, up front.

    Kept as a separate step from the first inference so the model-load cost
    lands during scan setup instead of being charged to the first sampled
    frame, where it would look like a stall.

    `_model` doubles as an override sentinel: unit tests that only exercise
    tracker control flow set it to a dummy so no weights are ever loaded.
    """
    global _model
    if _model is None:
        _model = person_onnx.ensure_session()
    return _model


# Person class index in the COCO label set every candidate backend is trained
# on. Named because `!= 0` at three call sites reads like a sentinel check.
_PERSON_CLASS = 0


def detect_persons(image, conf: float):
    """Person boxes in `image`, as pixel [x1, y1, x2, y2] in that image's frame.

    The ONLY place this module talks to the object detector. Recall uses the
    detector for exactly one thing -- finding people, to validate facecam
    plates -- so the whole dependency reduces to this signature.

    The backend is YOLOX over onnxruntime (Apache-2.0). It replaced Ultralytics
    YOLO, which is AGPL-3.0 and so could not ship in a closed-source build; see
    docs/DETECTOR_RELICENSE_PLAN.md. Measured on real VOD frames the two agree
    exactly on person count with mean IoU 0.888, and YOLOX-tiny is ~40% faster.

    `conf` stays a parameter rather than a constant because confidence
    thresholds are calibrated to a specific model's score distribution and do
    not transfer between detectors.
    """
    return person_onnx.detect_persons(image, conf)


def _is_fullframe_camera_person(box) -> bool:
    """Whether a normalized person box is strong fullscreen-camera evidence."""
    if not box or len(box) < 4:
        return False
    x, y, w, h = (float(value) for value in box[:4])
    return (
        x >= 0.0
        and y >= 0.0
        and w * h >= FULLFRAME_PERSON_MIN_AREA
        and w >= FULLFRAME_PERSON_MIN_WIDTH
        and h >= FULLFRAME_PERSON_MIN_HEIGHT
        and y + h >= FULLFRAME_PERSON_MIN_BOTTOM
    )


def detect_fullframe_camera_person(frame_img):
    """Return a large fullscreen-camera person box, or None.

    This is a one-frame composition probe, not temporal facecam tracking.  It
    intentionally shares the already-warm detector and does not mutate the
    facecam history.
    """
    if frame_img is None or frame_img.size == 0:
        return None
    h_img, w_img = frame_img.shape[:2]
    candidates = []
    for x1, y1, x2, y2 in detect_persons(frame_img, conf=0.35):
        box = [
            max(0.0, x1 / w_img),
            max(0.0, y1 / h_img),
            max(0.0, min(1.0, x2 / w_img) - max(0.0, x1 / w_img)),
            max(0.0, min(1.0, y2 / h_img) - max(0.0, y1 / h_img)),
        ]
        if _is_fullframe_camera_person(box):
            candidates.append(box)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[2] * item[3])

def reset_vision_state():
    """Reset per-video tracking while keeping the loaded model warm."""
    global _stable_facecam, _stable_facecam_source, _facecam_history, _facecam_misses
    _stable_facecam = None
    _stable_facecam_source = None
    _facecam_history = []
    _facecam_misses = 0

def has_stable_facecam() -> bool:
    return _stable_facecam is not None

def _pad_and_clamp_box(box, pad_ratio=0.04):
    """Expand a normalized [x, y, w, h] box slightly while keeping it in frame."""
    x, y, w, h = box
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    
    x1 = max(0.0, x - pad_x)
    y1 = max(0.0, y - pad_y)
    x2 = min(1.0, x + w + pad_x)
    y2 = min(1.0, y + h + pad_y)
    
    return [x1, y1, x2 - x1, y2 - y1]

# Content-aware plate cleanup. Detected cam boxes frequently swallow a window
# title bar (OBS "cam.exe"-style browser source), an alert/gutter strip, or
# letterbox padding — all near-uniform color bands the safety inset (a few %)
# is far too small to remove. These bands are flat across their whole span;
# the camera image is not. Scan inward from each edge and drop the contiguous
# flat band, bounded so a real face is never sliced.
FACECAM_TRIM_MAX_FRAC = 0.30   # never trim more than this off any one side
FACECAM_TRIM_FLAT_RANGE = 14.0  # robust intensity spread (0-255) below which a
                                # row/col counts as a solid chrome band


def _robust_line_range(line: np.ndarray) -> float:
    """Intensity spread of one row/column, ignoring a few outlier text pixels.

    Using an 8th–92nd percentile range instead of std keeps a solid bar with
    thin overlaid text (e.g. a title-bar label) from reading as content: only a
    genuinely varied band (the camera image) exceeds the flat threshold.
    """
    lo, hi = np.percentile(line, (8.0, 92.0))
    return float(hi - lo)


def _trim_uniform_borders(frame_img, box, frame_width, frame_height):
    """Shrink a facecam box inward past solid-color chrome/letterbox bands."""
    x, y, w, h = box
    x1 = max(0, int(round(x * frame_width)))
    y1 = max(0, int(round(y * frame_height)))
    x2 = min(frame_width, int(round((x + w) * frame_width)))
    y2 = min(frame_height, int(round((y + h) * frame_height)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return box
    crop = frame_img[y1:y2, x1:x2]
    if crop.size == 0:
        return box
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if _robust_line_range(gray.reshape(-1)) < FACECAM_TRIM_FLAT_RANGE:
        # The whole plate is flat (camera off, a loading/black frame): there is
        # no content boundary to hug, so trimming would only shrink the box for
        # no reason. Leave it untouched.
        return box
    rows, cols = gray.shape
    max_rows = int(rows * FACECAM_TRIM_MAX_FRAC)
    max_cols = int(cols * FACECAM_TRIM_MAX_FRAC)

    top = 0
    while top < max_rows and _robust_line_range(gray[top, :]) < FACECAM_TRIM_FLAT_RANGE:
        top += 1
    bottom = rows
    while bottom > rows - max_rows and _robust_line_range(gray[bottom - 1, :]) < FACECAM_TRIM_FLAT_RANGE:
        bottom -= 1
    left = 0
    while left < max_cols and _robust_line_range(gray[:, left]) < FACECAM_TRIM_FLAT_RANGE:
        left += 1
    right = cols
    while right > cols - max_cols and _robust_line_range(gray[:, right - 1]) < FACECAM_TRIM_FLAT_RANGE:
        right -= 1

    if top == 0 and left == 0 and bottom == rows and right == cols:
        return box  # nothing to trim
    if bottom - top < 8 or right - left < 8:
        return box  # would over-trim; keep the original box
    return [
        round((x1 + left) / frame_width, 4),
        round((y1 + top) / frame_height, 4),
        round((right - left) / frame_width, 4),
        round((bottom - top) / frame_height, 4),
    ]


def _expand_box_to_aspect(box, frame_width, frame_height, target_aspect):
    """Expand a normalized box to a pixel aspect ratio without leaving the frame."""
    x, y, w, h = box
    current_aspect = (w * frame_width) / max(1.0, h * frame_height)
    
    if current_aspect < target_aspect:
        new_w = h * frame_height * target_aspect / frame_width
        new_h = h
    else:
        new_w = w
        new_h = w * frame_width / target_aspect / frame_height
        
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = min(1.0, new_w)
    new_h = min(1.0, new_h)
    
    x1 = min(max(0.0, cx - new_w / 2.0), 1.0 - new_w)
    y1 = min(max(0.0, cy - new_h / 2.0), 1.0 - new_h)
    
    return [x1, y1, new_w, new_h]

def _candidate_score(box, tightness=0.0):
    x, y, w, h = box
    touches_edge = x <= 0.08 or y <= 0.08 or (x + w) >= 0.92 or (y + h) >= 0.92
    top_bias = 1.0 - min(1.0, y)
    edge_bonus = 0.35 if touches_edge else 0.0
    # Tightness (person_area / candidate_area, 0..1) replaces a raw size reward:
    # rewarding bigger area made contour-closing artifacts (a colored border
    # merged with several morphology iterations into progressively larger
    # blobs) win over the box that actually hugs the streamer, padding the
    # exported facecam with background the streamer never intended to feature.
    return edge_bonus + (top_bias * 0.25) + tightness

def _colored_border_score(frame_img, box, frame_width, frame_height):
    x, y, w, h = box
    x1 = max(0, int(x * frame_width))
    y1 = max(0, int(y * frame_height))
    x2 = min(frame_width, int((x + w) * frame_width))
    y2 = min(frame_height, int((y + h) * frame_height))
    crop = frame_img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([120, 55, 55]),
        np.array([179, 255, 255])
    )
    thickness = max(3, int(min(crop.shape[:2]) * 0.08))
    border = np.zeros(mask.shape, dtype=np.uint8)
    border[:thickness, :] = 1
    border[-thickness:, :] = 1
    border[:, :thickness] = 1
    border[:, -thickness:] = 1
    return float((mask[border == 1] > 0).mean())

def _panel_score(frame_img, box, frame_width, frame_height, person_box=None):
    # NOTE: `person_box` comes from detection run on THIS CANDIDATE'S crop, so
    # this ratio rewards a small sub-rectangle that a person happens to fill.
    # An IoU-based, scale-invariant variant was prototyped and measured on the
    # portrait-cam VOD that regressed under the YOLOX swap: it changed the plate
    # by 0.005 IoU (0.525 -> 0.520), i.e. not the cause. Kept as-is; the real
    # mechanism is panel-vs-fallback precedence, see
    # docs/DETECTOR_RELICENSE_PLAN.md.
    tightness = 0.0
    if person_box:
        candidate_area = box[2] * box[3]
        person_area = person_box[2] * person_box[3]
        if candidate_area > 0:
            tightness = max(0.0, min(1.0, person_area / candidate_area))
    return _candidate_score(box, tightness) + (1.5 * _colored_border_score(frame_img, box, frame_width, frame_height))


def _dedupe_boxes(boxes, iou_threshold=0.60):
    deduped = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(_calculate_iou(box, existing) < iou_threshold for existing in deduped):
            deduped.append(box)
    return deduped

def _find_panel_candidates(frame_img, frame_width, frame_height):
    """Find stable-looking overlay rectangles. YOLO validates these; it does not define the crop."""
    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        fx = x / frame_width
        fy = y / frame_height
        fw = w_box / frame_width
        fh = h_box / frame_height
        area = fw * fh
        aspect = w_box / max(1, h_box)
        touches_edge = fx <= 0.10 or fy <= 0.12 or (fx + fw) >= 0.90 or (fy + fh) >= 0.90

        if not touches_edge:
            continue
        if not (0.015 <= area <= 0.22):
            continue
        if not (0.12 <= fw <= 0.55 and 0.08 <= fh <= 0.45):
            continue
        if not (0.75 <= aspect <= 2.75):
            continue

        candidates.append(_pad_and_clamp_box([fx, fy, fw, fh], pad_ratio=0.015))

    # Many streamer facecams are defined by bright colored overlay frames rather
    # than a closed rectangular edge. Color-connected candidates catch those.
    hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(
        hsv,
        np.array([120, 55, 55]),
        np.array([179, 255, 255])
    )
    color_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    color_variants = [
        color_mask,
        cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, color_kernel, iterations=1),
        cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, color_kernel, iterations=2),
    ]

    for color_img in color_variants:
        color_contours, _ = cv2.findContours(color_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in color_contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            fx = x / frame_width
            fy = y / frame_height
            fw = w_box / frame_width
            fh = h_box / frame_height
            area = fw * fh
            aspect = w_box / max(1, h_box)
            touches_edge = fx <= 0.10 or fy <= 0.12 or (fx + fw) >= 0.90 or (fy + fh) >= 0.90

            if not touches_edge:
                continue
            if not (0.010 <= area <= 0.22):
                continue
            if not (0.10 <= fw <= 0.60 and 0.06 <= fh <= 0.35):
                continue
            if not (1.15 <= aspect <= 3.25):
                continue

            candidates.append(_pad_and_clamp_box([fx, fy, fw, fh], pad_ratio=0.01))

    return _dedupe_boxes(candidates)

def _panel_contains_person(frame_img, box, frame_width, frame_height):
    """Verify a panel candidate has a person in it, and return that person's
    box in frame-normalized coords (for tightness scoring), or None."""
    x, y, w, h = box
    x1 = max(0, int(x * frame_width))
    y1 = max(0, int(y * frame_height))
    x2 = min(frame_width, int((x + w) * frame_width))
    y2 = min(frame_height, int((y + h) * frame_height))
    crop = frame_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    best = None
    best_area = -1.0
    for px1, py1, px2, py2 in detect_persons(crop, conf=0.20):
        area = (px2 - px1) * (py2 - py1)
        if area > best_area:
            best_area = area
            best = [
                (x1 + px1) / frame_width,
                (y1 + py1) / frame_height,
                (px2 - px1) / frame_width,
                (py2 - py1) / frame_height,
            ]
    return best


def person_box_in_region(frame_img, box):
    """The largest person inside a normalized source region, or None.

    Returns frame-normalized [x, y, w, h] -- the same coordinate space as
    ``box`` -- so callers can ask WHERE the subject sits inside an established
    camera plate, not merely whether one is there. Like
    :func:`detect_person_in_region` this never discovers or moves the plate.
    """
    if frame_img is None or getattr(frame_img, "size", 0) == 0 or not box:
        return None
    _ensure_model()
    height, width = frame_img.shape[:2]
    return _panel_contains_person(frame_img, box, width, height)


def detect_person_in_region(frame_img, box) -> bool:
    """Whether a normalized source region currently contains a person.

    This validates an already-established camera layout; it does not discover
    or move the box and therefore cannot promote an arbitrary detection.
    """
    return person_box_in_region(frame_img, box) is not None

def _find_facecam_panel(frame_img, frame_width, frame_height):
    panel_candidates = _find_panel_candidates(frame_img, frame_width, frame_height)
    verified = []
    for box in panel_candidates:
        person_box = _panel_contains_person(frame_img, box, frame_width, frame_height)
        if person_box is not None:
            verified.append((box, person_box))
    if verified:
        verified.sort(
            key=lambda item: _panel_score(frame_img, item[0], frame_width, frame_height, item[1]),
            reverse=True
        )
        return verified[0][0]

    return None

def _find_facecam_from_person_detections(frame_img, frame_width, frame_height):
    """Find a small streamer-person detection that looks like an overlay, not gameplay."""
    candidates = []

    for x1, y1, x2, y2 in detect_persons(frame_img, conf=0.35):
        fx = max(0.0, x1 / frame_width)
        fy = max(0.0, y1 / frame_height)
        fw = min(1.0, x2 / frame_width) - fx
        fh = min(1.0, y2 / frame_height) - fy
        if fw <= 0 or fh <= 0:
            continue

        area = fw * fh
        touches_edge = fx <= 0.12 or fy <= 0.12 or (fx + fw) >= 0.88 or (fy + fh) >= 0.88

        # The in-game player model is usually large and near the play area center.
        # Facecam streamers are smaller overlays, typically hugging a corner or edge.
        if 0.005 <= area <= 0.16 and fw <= 0.45 and fh <= 0.55 and touches_edge:
            padded = _pad_and_clamp_box([fx, fy, fw, fh], pad_ratio=0.35)
            candidates.append((_candidate_score(padded), padded))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]

def run_vision(
    frame_img: np.ndarray,
    timestamp: float,
    detect_facecam: bool = True,
    stable_detections_required: int = 1,
) -> VisionSignal:
    global _model, _stable_facecam, _stable_facecam_source, _facecam_history, _facecam_misses
    
    _ensure_model()
        
    if frame_img is None or frame_img.size == 0:
        return VisionSignal(timestamp=timestamp, facecam_box=_stable_facecam, gameplay_box=[0.0, 0.0, 1.0, 1.0], facecam_source=_stable_facecam_source)

    if not detect_facecam:
        return VisionSignal(timestamp=timestamp, facecam_box=_stable_facecam, gameplay_box=[0.0, 0.0, 1.0, 1.0], facecam_source=_stable_facecam_source)
        
    h_img, w_img = frame_img.shape[:2]
    best_frame_box = _find_facecam_panel(frame_img, w_img, h_img)
    best_frame_source = "panel" if best_frame_box is not None else None
    if best_frame_box is None:
        # Connected overlay chrome can merge a cam panel into adjacent chat,
        # leaving no closed panel contour. The small edge-person fallback was
        # built for this case but previously never participated in run_vision.
        best_frame_box = _find_facecam_from_person_detections(frame_img, w_img, h_img)
        if best_frame_box is not None:
            best_frame_source = "person_fallback"

    if best_frame_box is not None:
        # Tighten onto the actual camera image before tracking/clustering, so a
        # swallowed title bar or gutter never reaches the export crop. Runs on
        # the small plate region only — negligible next to YOLO.
        best_frame_box = _trim_uniform_borders(frame_img, best_frame_box, w_img, h_img)

    # Update temporal tracking history if verified boxes are found
    verified_boxes = [best_frame_box] if best_frame_box else []
    if verified_boxes:
        _facecam_misses = 0
        IOU_THRESHOLD = 0.70
        new_history = []
        best_candidate = None
        max_frames_seen = -1
        
        for c_box in verified_boxes:
            matched = False
            for h_item in _facecam_history:
                if _calculate_iou(c_box, h_item['box']) > IOU_THRESHOLD:
                    updated_item = {
                        'box': c_box,
                        'frames_seen': h_item['frames_seen'] + 1
                    }
                    new_history.append(updated_item)
                    matched = True
                    
                    if updated_item['frames_seen'] > max_frames_seen:
                        max_frames_seen = updated_item['frames_seen']
                        best_candidate = c_box
                    break
                    
            if not matched:
                new_history.append({'box': c_box, 'frames_seen': 1})
                if max_frames_seen == -1:
                    best_candidate = c_box
                    max_frames_seen = 1
                    
        _facecam_history = new_history

        if best_candidate and max_frames_seen >= max(1, stable_detections_required):
            # Smooth the tracked box instead of snapping to each frame's raw
            # detection. Animated overlay borders (pulsing neon frames) make the
            # per-frame contour "breathe" a few percent every frame, which drags
            # the per-clip median crop around. An EMA damps that jitter while
            # still tracking real drift; a low-IoU jump (the streamer moved/
            # resized the cam, i.e. a scene change) snaps straight to the new box.
            if _stable_facecam is not None and _calculate_iou(best_candidate, _stable_facecam) >= 0.5:
                a = FACECAM_SMOOTHING
                _stable_facecam = [a * nc + (1.0 - a) * sc
                                   for nc, sc in zip(best_candidate, _stable_facecam)]
            else:
                _stable_facecam = best_candidate
            _stable_facecam_source = best_frame_source
    else:
        _facecam_misses += 1
        if _facecam_misses >= FACECAM_MAX_MISSES:
            _stable_facecam = None
            _stable_facecam_source = None
            _facecam_history = []
            
    return VisionSignal(
        timestamp=timestamp,
        facecam_box=_stable_facecam,
        gameplay_box=[0.0, 0.0, 1.0, 1.0],
        facecam_source=_stable_facecam_source,
    )
