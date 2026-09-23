# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Check camera layouts against spatially persistent face evidence.

Candidate rectangles can contain HUD art or interior camera fragments. This
module samples a face cloud across frames to identify persistent positions and
contradict unsupported layouts. One-frame detections are insufficient because
game art can also resemble faces.

Sparse face evidence causes abstention. This preserves layouts for scenes and
avatars a face detector cannot recognize. Rejection must not empty the layout
set, and window-local checks handle changes hidden by whole-recording evidence."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Minimum face box size as a fraction of frame height. A cam-plate face is
# large; this suppresses most of the cascade's taste for gameplay texture.
MIN_FACE_FRAC = 0.03

# Below this many detected faces the cloud is not a cloud and the anchor
# abstains entirely. A VTuber VOD lands here, which is the point.
MIN_FACE_SAMPLES = 25

# Share of sampled frames that must contain a face before absence of a face
# inside a layout counts as evidence AGAINST that layout. On a VOD where the
# detector only occasionally sees anyone, "no faces in this box" says nothing.
MIN_FACE_HIT_RATE = 0.25

# Reject only layouts holding a negligible share of an otherwise reliable face cloud.

FACELESS_SHARE_MAX = 0.02

# A layout is never rejected while it holds at least this many faces outright,
# regardless of share -- a genuine second cam that is only on screen briefly
# still holds real faces, and share alone would punish it for being brief.
KEEP_ABSOLUTE_FACES = 5


# Detector backend: Haar remains the default for the absence-based rejection rule.
# A more sensitive detector can also find faces in HUD art and weaken that rule.
# BlazeFace remains opt-in for experiments with positive face localization;
# changing detectors requires reevaluating the downstream rejection behavior.

FACE_MODEL_RELPATH = os.path.join("face", "blaze_face_short_range.tflite")
TILE_GRID = 3
TILE_OVERLAP = 0.2
FACE_MIN_CONFIDENCE = 0.4
# Two detections closer than this (normalized) are the same face seen in two
# overlapping tiles. Without this the tiling double-counts every face near a
# seam and every share it feeds is inflated.
TILE_DEDUPE_DIST = 0.04

_MP_DETECTOR = None
_MP_STATE = "unloaded"  # unloaded | ready | unavailable


def _mediapipe_detector():
    """The shared BlazeFace detector, or None when unavailable.

    Never raises: a missing package or model degrades to the Haar fallback so
    an install without the extra dependency still scans.
    """
    global _MP_DETECTOR, _MP_STATE
    if _MP_STATE != "unloaded":
        return _MP_DETECTOR
    try:
        from mediapipe.tasks.python import BaseOptions, vision

        from core.bundle_paths import get_models_dir

        model_path = os.path.join(get_models_dir(), FACE_MODEL_RELPATH)
        if not os.path.isfile(model_path):
            _MP_STATE = "unavailable"
            return None
        _MP_DETECTOR = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                min_detection_confidence=FACE_MIN_CONFIDENCE,
            )
        )
        _MP_STATE = "ready"
    except Exception:  # noqa: BLE001 - detector choice must never fail a scan
        _MP_DETECTOR = None
        _MP_STATE = "unavailable"
    return _MP_DETECTOR


def _detect_mediapipe(frame) -> List[tuple]:
    """(cx, cy, w, h, score) for every face, over an overlapping tile grid."""
    import cv2
    import mediapipe as mp

    detector = _mediapipe_detector()
    if detector is None:
        return []
    height, width = frame.shape[:2]
    span = TILE_GRID - (TILE_GRID - 1) * TILE_OVERLAP
    tile_w, tile_h = width / span, height / span
    found: List[tuple] = []
    for row in range(TILE_GRID):
        for col in range(TILE_GRID):
            x0 = int(col * tile_w * (1 - TILE_OVERLAP))
            y0 = int(row * tile_h * (1 - TILE_OVERLAP))
            x1, y1 = min(width, int(x0 + tile_w)), min(height, int(y0 + tile_h))
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            tile = frame[y0:y1, x0:x1]
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(tile, cv2.COLOR_BGR2RGB),
            )
            for det in (detector.detect(image).detections or []):
                bb = det.bounding_box
                score = float(det.categories[0].score) if det.categories else 0.0
                found.append((
                    (x0 + bb.origin_x + bb.width / 2.0) / width,
                    (y0 + bb.origin_y + bb.height / 2.0) / height,
                    bb.width / width,
                    bb.height / height,
                    score,
                ))
    # Overlapping tiles see seam-straddling faces twice; keep the best of each.
    kept: List[tuple] = []
    for face in sorted(found, key=lambda f: -f[4]):
        if any(abs(face[0] - k[0]) < TILE_DEDUPE_DIST
               and abs(face[1] - k[1]) < TILE_DEDUPE_DIST for k in kept):
            continue
        kept.append(face)
    return kept


def _detect_haar(frame) -> List[tuple]:
    """Fallback when MediaPipe is unavailable. Noisier and ~6x slower."""
    import cv2

    from engines.emotion.face_emotion import _get_face_detector

    height, width = frame.shape[:2]
    floor = int(height * MIN_FACE_FRAC)
    found = _get_face_detector().detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1, minNeighbors=5, minSize=(floor, floor),
    )
    return [
        ((x + w / 2.0) / width, (y + h / 2.0) / height,
         w / width, h / height, 1.0)
        for x, y, w, h in (tuple(float(v) for v in b) for b in found)
    ]


def _use_mediapipe() -> bool:
    """Opt-in only. See the backend note above for why this is not the default."""
    if os.environ.get("RECALL_FACE_DETECTOR", "").lower() != "mediapipe":
        return False
    return _mediapipe_detector() is not None


def detect_faces(frame) -> List[tuple]:
    """(cx, cy, w, h, score) per face, from the selected backend."""
    if _use_mediapipe():
        return _detect_mediapipe(frame)
    return _detect_haar(frame)


def detector_name() -> str:
    """Which backend detect_faces will use -- for provenance and cache keys."""
    return "mediapipe-blazeface-tiled" if _use_mediapipe() else "haar"


@dataclass
class FaceSample:
    """One detected face, normalized to the frame."""
    timestamp: float
    cx: float
    cy: float
    w: float
    h: float
    score: float = 1.0


@dataclass
class AnchorVerdict:
    """What the face cloud says about one job's layouts.

    ``rejected`` is the list of layout indices the cloud positively
    contradicts. ``abstained`` is True whenever the evidence was too thin to
    say anything, in which case ``rejected`` is always empty.
    """
    rejected: List[int]
    abstained: bool
    reason: str
    faces: int
    sampled: int
    shares: List[float]

    @property
    def hit_rate(self) -> float:
        return self.faces / self.sampled if self.sampled else 0.0


def sample_face_cloud(
    video_path: str,
    interval: float = 30.0,
    limit: int = 400,
    progress=None,
) -> Tuple[List[FaceSample], int]:
    """Haar-detect faces on FULL frames, sparsely sampled.

    Returns ``(faces, frames_sampled)``. Seeks once per sample rather than
    decoding the VOD, so the cost is roughly ``limit`` seeks regardless of
    runtime. CPU only -- no GPU, no model download.
    """
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"cannot open {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        step = max(1, int(round(fps * interval)))

        faces: List[FaceSample] = []
        sampled = 0
        position = 0
        while position < total and sampled < limit:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                break
            sampled += 1
            found = detect_faces(frame)
            if found:
                # Largest face per frame: the cam subject is nearer than any
                # background art, and one sample per frame keeps the cloud a
                # timeline rather than a per-frame pile.
                cx, cy, fw, fh, score = max(found, key=lambda f: f[2] * f[3])
                faces.append(FaceSample(
                    timestamp=position / fps, cx=cx, cy=cy,
                    w=fw, h=fh, score=score,
                ))
            if progress is not None and sampled % 50 == 0:
                progress(sampled, len(faces))
            position += step
        return faces, sampled
    finally:
        capture.release()


def _contains(box: Sequence[float], face: FaceSample) -> bool:
    x, y, w, h = (float(v) for v in box[:4])
    return x <= face.cx <= x + w and y <= face.cy <= y + h


def layout_face_shares(
    layouts: Sequence, faces: Sequence[FaceSample],
) -> List[float]:
    """Share of the face cloud whose centre falls inside each layout box."""
    if not faces:
        return [0.0] * len(layouts)
    return [
        sum(1 for face in faces if _contains(layout.box, face)) / len(faces)
        for layout in layouts
    ]


def evaluate(layouts: Sequence, faces: Sequence[FaceSample], sampled: int) -> AnchorVerdict:
    """Which layouts does the face cloud positively contradict?

    Abstains -- rejecting nothing -- whenever the evidence is thin. See the
    module docstring: this is the property every VTuber scan depends on.
    """
    shares = layout_face_shares(layouts, faces)
    if not layouts:
        return AnchorVerdict([], True, "no layouts", len(faces), sampled, shares)
    if len(faces) < MIN_FACE_SAMPLES:
        return AnchorVerdict(
            [], True, f"only {len(faces)} faces (< {MIN_FACE_SAMPLES})",
            len(faces), sampled, shares)
    hit_rate = len(faces) / sampled if sampled else 0.0
    if hit_rate < MIN_FACE_HIT_RATE:
        return AnchorVerdict(
            [], True, f"face hit-rate {hit_rate:.2f} (< {MIN_FACE_HIT_RATE})",
            len(faces), sampled, shares)

    counts = [int(round(share * len(faces))) for share in shares]
    rejected = [
        index for index, share in enumerate(shares)
        if share <= FACELESS_SHARE_MAX and counts[index] < KEEP_ABSOLUTE_FACES
    ]
    if len(rejected) >= len(layouts):
        # Every layout is faceless: the cloud disagrees with the whole model
        # rather than with one plate. That is a different failure (a
        # full-screen cam, or a detector firing only on game art) and this
        # rule is not entitled to resolve it by deleting everything.
        return AnchorVerdict(
            [], True, "every layout is faceless -- model-level disagreement",
            len(faces), sampled, shares)
    return AnchorVerdict(
        rejected, False,
        f"{len(rejected)} layout(s) contradicted by {len(faces)} faces",
        len(faces), sampled, shares)


# Cache the face cloud alongside signal caches to avoid repeating media seeks.

def cache_path(cache_dir: str, video_path: str, interval: float, limit: int) -> str:
    import hashlib

    # The detector is part of the key: a cloud sampled with Haar must not be
    # served to a caller now running BlazeFace, and vice versa.
    key = (f"{os.path.abspath(video_path)}|{interval}|{limit}"
           f"|{MIN_FACE_FRAC}|{detector_name()}")
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{digest}_faceanchor.cache.json.gz")


def load_cloud(path: str) -> Optional[Tuple[List[FaceSample], int]]:
    from core import signal_cache

    payload = signal_cache.load(path)
    if not isinstance(payload, dict):
        return None
    rows = payload.get("faces")
    if rows is None:
        return None
    return (
        [FaceSample(*[float(v) for v in row]) for row in rows],  # ts,cx,cy,w,h[,score]
        int(payload.get("sampled") or 0),
    )


def save_cloud(path: str, faces: Sequence[FaceSample], sampled: int) -> None:
    from core import signal_cache

    signal_cache.save(path, {
        "sampled": int(sampled),
        "faces": [
            [round(f.timestamp, 3), round(f.cx, 5), round(f.cy, 5),
             round(f.w, 5), round(f.h, 5), round(f.score, 4)]
            for f in faces
        ],
    })


def face_cloud_for(
    video_path: str,
    cache_dir: Optional[str] = None,
    interval: float = 30.0,
    limit: int = 400,
    progress=None,
) -> Tuple[List[FaceSample], int]:
    """Cached ``sample_face_cloud``."""
    path = cache_path(cache_dir, video_path, interval, limit) if cache_dir else None
    if path and os.path.isfile(path):
        cached = load_cloud(path)
        if cached is not None:
            return cached
    faces, sampled = sample_face_cloud(video_path, interval, limit, progress)
    if path:
        try:
            save_cloud(path, faces, sampled)
        except Exception:  # noqa: BLE001 - a cache write must never fail a scan
            pass
    return faces, sampled


# Positive anchoring uses spatial recurrence across frames. Game characters can
# look like human faces, so size or confidence in one frame cannot identify a
# camera. Persistent positions provide stronger evidence than a single detection.
# Two face centres within this distance belong to the same camera position.

CLUSTER_TOL = 0.05
# A position must recur in at least this many samples to be a camera at all.
CLUSTER_MIN_SAMPLES = 3


# Expected face position as a fraction of plate height. A face near the edge
# can indicate an interior fragment rather than a complete head-and-shoulders plate.

PLATE_FACE_CENTER = 0.53
PLATE_FACE_TOLERANCE = 0.22


def plate_face_position(box: Sequence[float], position: "CameraPosition") -> float:
    """Face centre as a fraction of plate height. <0 or >1 means outside."""
    y, h = float(box[1]), float(box[3])
    return (position.cy - y) / h if h else 0.0


def plate_frames_face_plausibly(
    box: Sequence[float], position: "CameraPosition",
) -> bool:
    """Does this plate frame the face the way a real cam plate does?"""
    return abs(
        plate_face_position(box, position) - PLATE_FACE_CENTER
    ) <= PLATE_FACE_TOLERANCE


@dataclass
class CameraPosition:
    """A place where a face keeps appearing."""
    cx: float
    cy: float
    h: float
    samples: int
    share: float


def camera_positions(faces: Sequence[FaceSample]) -> List[CameraPosition]:
    """Persistent face positions in the cloud, most-supported first.

    Greedy modes rather than true clustering: the question is only "where does
    a face keep showing up", and a face that appears in three sampled frames a
    minute apart is already saying something a single detection cannot.
    """
    points = [(f.cx, f.cy, f.h) for f in faces]
    used = [False] * len(points)
    found: List[CameraPosition] = []
    for index, point in enumerate(points):
        if used[index]:
            continue
        near = [
            k for k, other in enumerate(points)
            if not used[k]
            and abs(other[0] - point[0]) <= CLUSTER_TOL
            and abs(other[1] - point[1]) <= CLUSTER_TOL
        ]
        if len(near) < CLUSTER_MIN_SAMPLES:
            continue
        for k in near:
            used[k] = True
        found.append(CameraPosition(
            cx=float(np.median([points[k][0] for k in near])),
            cy=float(np.median([points[k][1] for k in near])),
            h=float(np.median([points[k][2] for k in near])),
            samples=len(near),
            share=len(near) / max(1, len(points)),
        ))
    return sorted(found, key=lambda p: -p.samples)


def layout_holding_camera(
    layouts: Sequence, faces: Sequence[FaceSample],
) -> Tuple[Optional[int], str]:
    """Which layout contains the dominant persistent face position?

    Returns ``(index, reason)``; index is None whenever the evidence is too
    thin or ambiguous to say, which every caller must treat as "change
    nothing". Positive evidence, so unlike the absence rule it does not depend
    on the detector failing to see things.
    """
    if not layouts:
        return None, "no layouts"
    if len(faces) < MIN_FACE_SAMPLES:
        return None, f"only {len(faces)} faces (< {MIN_FACE_SAMPLES})"
    positions = camera_positions(faces)
    if not positions:
        return None, "no persistent face position"
    best = positions[0]
    holders = [
        index for index, layout in enumerate(layouts)
        if _contains(layout.box, FaceSample(0.0, best.cx, best.cy, 0.0, best.h))
    ]
    if not holders:
        return None, (
            f"dominant face ({best.cx:.3f}, {best.cy:.3f}) "
            f"{best.share:.0%} sits in NO layout")
    if len(holders) > 1:
        # For overlapping plates, prefer the expected face position within the plate
        # over size alone; an interior fragment can otherwise crop the head.

        holders.sort(key=lambda i: abs(
            plate_face_position(layouts[i].box, best) - PLATE_FACE_CENTER))
    return holders[0], (
        f"dominant face ({best.cx:.3f}, {best.cy:.3f}) "
        f"{best.share:.0%} of cloud, in L{holders[0]} at "
        f"{plate_face_position(layouts[holders[0]].box, best):.2f} of plate height")


# Full-camera scenes use the existing full_camera composition with horizontal
# focus. Large faces call for the whole picture rather than a stacked camera band.
# The threshold is applied to face height in normalized frame coordinates.

FULLCAM_FACE_HEIGHT_MIN = 0.30
# Needs the same unanimity as the window test: one huge false detection must
# not convert a normal clip into a full-camera crop.
FULLCAM_MIN_FACES = 3
# Share of a window's samples that must show a large subject. Below 1.0 because
# clip windows span scene changes; well above 0.5 so a scene the clip merely
# touches cannot claim it.
FULLCAM_MAJORITY = 0.6


def fullframe_camera_scene(faces: Sequence[FaceSample]) -> Optional[float]:
    """Is this window a full-camera scene? Returns the subject's x, or None.

    The caller converts that to the renderer's cover-crop focus (facecam.py's
    ``_portrait_cover_focus``), because only it knows the source aspect.

    Returns None on thin or disagreeing evidence, so an undetected scene keeps
    whatever the layout model decided -- the same fail-open contract as the
    rest of this module.
    """
    if len(faces) < FULLCAM_MIN_FACES:
        return None
    big = [f for f in faces if f.h >= FULLCAM_FACE_HEIGHT_MIN]
    # Use a majority so a small-camera sample at a scene transition cannot
    # veto an otherwise consistent full-camera window.

    if len(big) / len(faces) < FULLCAM_MAJORITY:
        return None
    # Position comes from the large samples only; the transition frames
    # describe a different scene and must not drag the focus.
    xs = sorted(f.cx for f in big)
    return float(xs[len(xs) // 2])


# Window-local checks detect plates that are valid globally but exclude the
# face during a particular scene. Require a margin relative to plate size so a
# face straddling the border is not mistaken for a completely wrong rectangle.

WINDOW_OUTSIDE_MARGIN = 0.15

# Every sampled face in the window must be outside. One stray detection --
# Haar fires on game art -- must never unframe a clip on its own.
WINDOW_MIN_FACES = 2


def window_excludes_face(
    box: Optional[Sequence[float]], faces: Sequence[FaceSample],
) -> bool:
    """Does this plate demonstrably exclude the face in its own window?

    True only when there is real evidence AND all of it agrees: at least
    ``WINDOW_MIN_FACES`` detections in the window and every one of them clearly
    outside the plate. No plate, no faces, or any disagreement all return
    False, so the caller keeps whatever the layout model decided.

    Cropping to a rectangle that provably excludes the creator's face is worse
    than not cropping, which is what makes this worth acting on even though it
    cannot say what the RIGHT rectangle is.
    """
    if not box or len(faces) < WINDOW_MIN_FACES:
        return False
    x, y, w, h = (float(v) for v in box[:4])
    mx, my = w * WINDOW_OUTSIDE_MARGIN, h * WINDOW_OUTSIDE_MARGIN
    return all(
        not (x - mx <= face.cx <= x + w + mx and y - my <= face.cy <= y + h + my)
        for face in faces
    )


def faces_in_window(
    video_path: str, start: float, end: float, samples: int = 5,
) -> List[FaceSample]:
    """Face detections sampled evenly inside one clip window.

    Denser than the whole-VOD cloud on purpose: a 20s window contains at most
    one sample of a 1-per-30s cloud, which cannot support any conclusion.
    """
    import cv2

    span = max(0.0, float(end) - float(start))
    if span <= 0 or samples < 1:
        return []
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return []
    out: List[FaceSample] = []
    try:
        for index in range(samples):
            moment = float(start) + span * (index + 0.5) / samples
            capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            found = detect_faces(frame)
            if found:
                cx, cy, fw, fh, score = max(found, key=lambda f: f[2] * f[3])
                out.append(FaceSample(
                    timestamp=moment, cx=cx, cy=cy, w=fw, h=fh, score=score,
                ))
    finally:
        capture.release()
    return out


# Withdraw rejections that would remove the camera from too many affected
# windows. Face evidence alone should not turn an uncertain scene into no camera.

SAFE_CAM_LOSS_MAX = 0.25


def confirm_rejection(
    rejected: Sequence[int],
    before_boxes: Sequence[Optional[Sequence[float]]],
    after_boxes: Sequence[Optional[Sequence[float]]],
) -> Tuple[List[int], str]:
    """Withdraw a rejection that de-cams too many windows.

    ``before_boxes`` / ``after_boxes`` are the per-clip facecam boxes the
    engine produces with and without the rejected layouts -- the caller owns
    that computation because only it knows the clip windows. Returns the
    rejections that survive, plus a human-readable reason.

    This is the last fail-open guard. Rejecting a plate is only ever meant to
    move a clip onto the RIGHT camera; a rejection whose main effect is that
    clips lose their camera is doing something else, and the safe response to
    "something else" is to change nothing.
    """
    changed = [
        (before, after)
        for before, after in zip(before_boxes, after_boxes)
        if list(before or []) != list(after or [])
    ]
    if not changed:
        return [], "rejection changes no window"
    lost = sum(1 for _before, after in changed if not after)
    share = lost / len(changed)
    if share > SAFE_CAM_LOSS_MAX:
        return [], (
            f"withdrawn: {lost}/{len(changed)} changed windows would lose the "
            f"camera ({share:.0%} > {SAFE_CAM_LOSS_MAX:.0%}) -- model-level "
            f"disagreement, not a wrong plate")
    return list(rejected), (
        f"confirmed: {len(changed)} window(s) reframed, {lost} de-cammed")


def summarize(layouts: Sequence, verdict: AnchorVerdict) -> Dict:
    """Compact record for persistence / logging."""
    return {
        "faces": verdict.faces,
        "sampled": verdict.sampled,
        "hit_rate": round(verdict.hit_rate, 3),
        "abstained": verdict.abstained,
        "reason": verdict.reason,
        "rejected": list(verdict.rejected),
        "shares": [round(s, 4) for s in verdict.shares],
        "boxes": [[round(float(v), 4) for v in lay.box] for lay in layouts],
    }


def centre_distances(layouts: Sequence, faces: Sequence[FaceSample]) -> List[float]:
    """Median distance from the face cloud to each layout's centre.

    Diagnostic only -- reported alongside shares because it is what separates
    a plate that is CENTRED on its occupant from one that merely contains
    them. Not used by :func:`evaluate`, which stays on the single rule that
    was measured to separate totally.
    """
    out: List[float] = []
    for layout in layouts:
        bx, by, bw, bh = (float(v) for v in layout.box[:4])
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        if not faces:
            out.append(0.0)
            continue
        out.append(float(np.median(
            [float(np.hypot(f.cx - cx, f.cy - cy)) for f in faces])))
    return out
