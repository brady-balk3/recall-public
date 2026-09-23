# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Facecam Emotion Engine (plan §5.1).

Runs facial-emotion recognition over the facecam crop that the vision engine
(YOLO) already locates, producing a per-second arousal/valence/emotion track.
This is the highest-ROI new modality: the streamer's facial reaction is
game-agnostic and frame-accurate, and it adds the discrimination that voice
arousal alone lacks on a hype-heavy creator.

Model: HSEmotion ONNX (EfficientNet-B0, ``enet_b0_8_va_mtl``) via onnxruntime —
small (~10-30 MB), CPU real-time, no torch. Outputs 8 emotion probabilities plus
a valence and arousal head. Gracefully disables (face_present=False everywhere)
if the model can't load, so facecam-less or model-less runs still work.
"""

import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import cv2

from engines.video.frame_extractor import extract_frames, extract_frames_range

_MODEL = None
_DISABLED_REASON: Optional[str] = None
_MODEL_NAME = "enet_b0_8_va_mtl"  # 8 emotions + valence + arousal heads

# Which scorer produces (arousal, valence, emotion) from a face crop.
#
# "blendshape" (default) -- MediaPipe Face Landmarker, Apache-2.0 weights, the
#   scorer that can ship in a paid product. See engines/emotion/blendshape_scorer.
# "hsemotion" -- the original AffectNet-derived model. Kept selectable so the
#   two can be A/B'd on one machine, but it must NOT be packaged; the licence
#   chain does not cleanly reach commercial redistribution.
#
# Face DETECTION is unaffected either way: the Haar cascade below and the
# BlazeFace path in engines/vision/face_anchor.py are both already permissive.
_SCORER = os.getenv("RECALL_FACE_SCORER", "blendshape").strip().lower()

_FACE_DETECTOR = None

# HSEmotion va_mtl output layout: [8 emotion probs, valence, arousal].
_VALENCE_IDX = 8
_AROUSAL_IDX = 9


@dataclass
class FaceFrame:
    timestamp: float
    face_present: bool
    face_arousal: float       # per-VOD-normalized downstream; raw model arousal here
    face_valence: float
    face_emotion: str

    def to_dict(self):
        return asdict(self)


def _seed_hsemotion_cache() -> None:
    """Copy the bundled weights into HSEmotion's cache before it downloads them.

    ``hsemotion_onnx.get_model_path`` fetches ``enet_b0_8_va_mtl.onnx`` from
    GitHub into ``~/.hsemotion`` on first use. A packaged install must not
    depend on that: a creator who is offline, behind a filter, or hitting a
    moved URL would get a silently disabled face channel -- and ``face`` is one
    of the strongest ranker features, so the scan would quietly come out worse
    with nothing in the UI to say why. It also fails the model-stack acceptance
    bar ("packaged startup reliability without a first-run download").

    Seeding the cache rather than patching ``get_model_path`` keeps this working
    if the library changes how it resolves paths.
    """
    import os
    import shutil

    from core.bundle_paths import get_models_dir

    bundled = os.path.join(get_models_dir(), "emotion", f"{_MODEL_NAME}.onnx")
    if not os.path.isfile(bundled):
        return  # dev checkout without the asset; fall through to the download
    cache_dir = os.path.join(os.path.expanduser("~"), ".hsemotion")
    cached = os.path.join(cache_dir, f"{_MODEL_NAME}.onnx")
    if os.path.isfile(cached):
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # Copy to a temp name first so an interrupted copy can never leave a
        # truncated file that looks cached and then fails to load forever.
        staging = f"{cached}.partial"
        shutil.copyfile(bundled, staging)
        os.replace(staging, cached)
    except OSError as exc:  # noqa: BLE001 - never block a scan on this
        print(f"Could not seed bundled HSEmotion weights ({exc}); will try download.")


def _get_model():
    global _MODEL, _DISABLED_REASON
    if _MODEL is not None or _DISABLED_REASON is not None:
        return _MODEL
    try:
        _seed_hsemotion_cache()
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        _MODEL = HSEmotionRecognizer(model_name=_MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        _DISABLED_REASON = str(exc)
        print(f"Emotion engine disabled: HSEmotion unavailable ({_DISABLED_REASON})")
        return None
    return _MODEL


def scorer_cache_label() -> str:
    """Short identity of the active scorer, for cache keys.

    Cached FaceFrames are only valid for the scorer that produced them, so this
    has to change whenever the values would: the scorer itself, and the
    blendshape feature that derives arousal.
    """
    if _SCORER == "hsemotion":
        return "hse"
    from engines.emotion import blendshape_scorer

    return f"bs-{blendshape_scorer.FEATURE}"


def _scorer_ready() -> bool:
    """True when the selected scorer can score crops.

    False disables the face channel for the whole pass, exactly as a missing
    HSEmotion model always has: every frame reports face_present=False rather
    than a fabricated calm face.
    """
    if _SCORER == "hsemotion":
        return _get_model() is not None
    from engines.emotion import blendshape_scorer

    return blendshape_scorer._get_landmarker() is not None


def _score_crop(face_bgr: np.ndarray):
    """(arousal, valence, emotion) for a tight face crop, or None on failure."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    if _SCORER == "hsemotion":
        model = _get_model()
        if model is None:
            return None
        emotion, scores = model.predict_emotions(rgb, logits=False)
        scores = np.asarray(scores, dtype=np.float32).ravel()
        arousal = float(scores[_AROUSAL_IDX]) if scores.size > _AROUSAL_IDX else 0.0
        valence = float(scores[_VALENCE_IDX]) if scores.size > _VALENCE_IDX else 0.0
        return arousal, valence, str(emotion)

    from engines.emotion import blendshape_scorer

    return blendshape_scorer.score_face(rgb)


def reset_emotion_state():
    global _MODEL, _DISABLED_REASON
    _MODEL = None
    _DISABLED_REASON = None
    from engines.emotion import blendshape_scorer

    blendshape_scorer.reset_state()


def _get_face_detector():
    """OpenCV Haar frontal-face cascade (bundled with opencv-python, no extra dep)."""
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_DETECTOR = cv2.CascadeClassifier(path)
    return _FACE_DETECTOR


def _detect_face(panel: np.ndarray, last_bbox=None):
    """Find a tight face crop inside the facecam panel.

    YOLO gives the facecam *panel*; HSEmotion needs a tight face or it collapses
    to a single class. Detect with Haar, pad ~20%, and on a transient miss reuse
    the last known bbox (the face is roughly static within the panel).
    Returns (face_crop, bbox) or (None, last_bbox).
    """
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    detector = _get_face_detector()
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    bbox = None
    if len(faces):
        bbox = max(faces, key=lambda b: b[2] * b[3])
    elif last_bbox is not None:
        bbox = last_bbox

    if bbox is None:
        return None, last_bbox

    x, y, w, h = bbox
    pad = int(0.2 * h)
    ph, pw = panel.shape[:2]
    y0, y1 = max(0, y - pad), min(ph, y + h + pad)
    x0, x1 = max(0, x - pad), min(pw, x + w + pad)
    face = panel[y0:y1, x0:x1]
    if face.size == 0 or face.shape[0] < 24 or face.shape[1] < 24:
        return None, bbox
    return face, bbox


def _crop_facecam(frame_img: np.ndarray, box) -> Optional[np.ndarray]:
    """Crop a normalized [x, y, w, h] facecam box to pixels; None if too small."""
    if not box:
        return None
    h, w = frame_img.shape[:2]
    x, y, bw, bh = box
    x0 = max(0, int(x * w))
    y0 = max(0, int(y * h))
    x1 = min(w, int((x + bw) * w))
    y1 = min(h, int((y + bh) * h))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    return frame_img[y0:y1, x0:x1]


def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered median filter; robust to FER spikes on noisy gamer faces."""
    if window <= 1 or values.size == 0:
        return values
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def _iter_sampled_frames(video_path: str, fps_sample_rate: float, regions=None):
    """Yield (timestamp, frame): the whole VOD, or only the given ``(start,
    end)`` second ranges. Ranged decode seeks once per region and skips
    everything outside it (extract_frames_range), so a smart scan pays only
    for the scouted footage instead of a whole-VOD decode."""
    if regions is None:
        yield from extract_frames(video_path, fps_sample_rate=fps_sample_rate)
        return
    for start, end in sorted((float(s), float(e)) for s, e in regions):
        if end <= start:
            continue
        yield from extract_frames_range(
            video_path, start, end, fps_sample_rate=fps_sample_rate
        )


def analyze_faces(
    video_path: str,
    facecam_box=None,
    per_frame_boxes: Optional[dict] = None,
    fps_sample_rate: float = 1.0,
    smooth_window: int = 5,
    regions=None,
) -> List[FaceFrame]:
    """Produce a per-second FaceFrame track for the whole VOD.

    ``facecam_box``: a stable normalized box used for every frame (preferred --
    the facecam is static on screen and the per-frame YOLO box is jittery).
    ``per_frame_boxes``: optional {int(round(ts)): box} fallback when no stable
    box is given.
    ``regions``: optional list of ``(start, end)`` second ranges — analyze only
    those slices (fast-mode smart scan). Timestamps outside the regions simply
    get no FaceFrame; fusion already treats missing timestamps as face-absent.
    """
    ready = _scorer_ready()

    raw_frames: List[FaceFrame] = []
    last_bbox = None
    for timestamp, frame_img in _iter_sampled_frames(video_path, fps_sample_rate, regions):
        box = facecam_box
        if box is None and per_frame_boxes is not None:
            box = per_frame_boxes.get(int(round(timestamp)))

        panel = _crop_facecam(frame_img, box) if box else None
        face = None
        if panel is not None:
            face, last_bbox = _detect_face(panel, last_bbox)

        if not ready or face is None:
            raw_frames.append(FaceFrame(timestamp, False, 0.0, 0.0, "neutral"))
            continue

        try:
            scored = _score_crop(face)
        except Exception:  # noqa: BLE001 - a single bad crop must not kill the pass
            scored = None
        if scored is None:
            raw_frames.append(FaceFrame(timestamp, False, 0.0, 0.0, "neutral"))
            continue
        arousal, valence, emotion = scored
        raw_frames.append(FaceFrame(timestamp, True, arousal, valence, emotion))

    # Median-smooth arousal over present frames (gamer faces are noisy).
    present_idx = [i for i, f in enumerate(raw_frames) if f.face_present]
    if present_idx and smooth_window > 1:
        arr = np.array([raw_frames[i].face_arousal for i in present_idx], dtype=np.float32)
        sm = _median_smooth(arr, smooth_window)
        for j, i in enumerate(present_idx):
            raw_frames[i].face_arousal = float(sm[j])

    return raw_frames
