# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Facecam arousal from MediaPipe Face Landmarker blendshapes (Apache-2.0).

Why this exists
---------------
The shipped scorer is HSEmotion ``enet_b0_8_va_mtl``. Its *code* is Apache-2.0
but its weights derive from AffectNet, whose terms are research/non-commercial,
so the licence chain does not cleanly reach a paid product. See the "Known
licensing risks" section of THIRD_PARTY_NOTICES.md.

MediaPipe Face Landmarker is different in kind: it emits face *geometry*
(52 blendshape activations), not emotion labels, and Google ships the weights
Apache-2.0 with no dataset asterisk. The emotional interpretation moves out of
the model and into this file, where it is inspectable.

What it produces
----------------
The same three fields HSEmotion produced, because all three are consumed:

* ``arousal`` -- fusion's highest-weighted channel (0.40) and the input to
  eight absolute gates.
* ``valence`` -- written through fusion into ReactionFrame and never read by
  anything. Emitted as 0.0 deliberately; see ``score_face``.
* ``emotion`` -- only ever compared against {"fear", "surprise"} by the startle
  detector in engines/reaction/fusion.py, so only that distinction is modelled.

Measured, on 7,441 face crops over 16 VODs / 3 channels, scored by BOTH models
on identical crops (per-VOD Spearman against HSEmotion arousal, which is the
right statistic because fusion robust-z-normalises face arousal per VOD, so
only within-VOD ordering survives into a deck):

    sqrt(jawOpen) + 3*sqrt(mouthFunnel)   +0.409   <- FEATURE default "mix"
    mouthFunnel alone                     +0.388   <- best worst-case VOD
    jawOpen alone                         +0.362
    fitted 52-dim ridge, leave-one-       +0.356   <- fitting buys nothing
      CHANNEL-out                                     once folds are honest

That last line is the important one. The same ridge scores +0.447 under
leave-one-VOD-out; the gap is same-channel leakage, because the 16 VODs are
only 3 streamers. A fitted model was learning three faces, not arousal, so
this ships an unfitted feature.

~0.41 is NOT a drop-in match for HSEmotion -- roughly half the ordering is
shared. Which moments rank highly WILL change. That is expected and is not
something correlation can adjudicate: HSEmotion is one model's opinion, not
ground truth. Only a deck replay against real labels can say whether this is
better or worse editorially.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Optional, Tuple

import numpy as np

_LANDMARKER = None
_LANDMARKER_LOCK = threading.Lock()
_DISABLED_REASON: Optional[str] = None

# Blendshape indices are positional in the model's output and stable across
# runs, but resolve by NAME anyway -- a silent reordering would corrupt arousal
# without failing anything.
_NAME_INDEX: Optional[dict] = None

FEATURE = os.getenv("RECALL_FACE_BLENDSHAPE_FEATURE", "mix").strip().lower()

# Quantile knots mapping each raw feature onto HSEmotion's arousal scale,
# derived from the 7,441-crop paired corpus (scripts/collect_blendshape_arousal.py).
#
# This mapping is what lets the swap avoid touching eight scattered absolute
# thresholds (fusion 0.12/0.10, reaction_main 0.12, selection 0.40/0.15/0.22 and
# the spike deltas). Those constants are calibrated to HSEmotion's output
# distribution; a raw blendshape is on a completely different scale (mouthFunnel
# has median 0.0015 against HSEmotion's -0.03, and never goes negative). Mapping
# percentile-to-percentile keeps every gate passing the same FRACTION of scored
# frames it passed before, so the gates keep their tuned meaning.
#
# The mapping is monotone, so it cannot change within-VOD ordering -- it is
# purely a change of units, and fusion's own per-VOD robust-z is unaffected.
#
# Verified pass-rates after mapping (corpus-wide):
#   >=0.12  20.71% vs HSEmotion 20.72%     >=0.40  4.00% vs HSEmotion 3.88%
_KNOTS = {
    "mix": (
        (0.033975, 0.083279, 0.096449, 0.108271, 0.118174, 0.128929, 0.141983,
         0.154361, 0.168612, 0.187390, 0.209512, 0.232531, 0.261101, 0.298355,
         0.337612, 0.392481, 0.476873, 0.606745, 0.797926, 1.041780, 1.240623,
         1.371852, 1.482293, 2.033652),
        (-0.522126, -0.242686, -0.204546, -0.175998, -0.153128, -0.128446,
         -0.109484, -0.089344, -0.069576, -0.051486, -0.031847, -0.011303,
         0.008566, 0.034545, 0.059637, 0.089463, 0.124545, 0.170228, 0.225262,
         0.344677, 0.536398, 0.660700, 0.789899, 1.300320),
    ),
    "mouthfunnel": (
        (0.000016, 0.000162, 0.000234, 0.000302, 0.000372, 0.000459, 0.000556,
         0.000679, 0.000861, 0.001111, 0.001468, 0.001971, 0.002674, 0.003533,
         0.004710, 0.006406, 0.008804, 0.014195, 0.025296, 0.044408, 0.069448,
         0.091912, 0.111303, 0.418091),
        (-0.522126, -0.242686, -0.204546, -0.175998, -0.153128, -0.128446,
         -0.109484, -0.089344, -0.069576, -0.051486, -0.031847, -0.011303,
         0.008566, 0.034545, 0.059637, 0.089463, 0.124545, 0.170228, 0.225262,
         0.344677, 0.536398, 0.660700, 0.789899, 1.300320),
    ),
    "jawopen": (
        (0.000006, 0.000904, 0.001510, 0.002030, 0.002501, 0.003049, 0.003683,
         0.004372, 0.005087, 0.005892, 0.006870, 0.008127, 0.009717, 0.011896,
         0.014952, 0.020526, 0.031647, 0.055502, 0.114478, 0.234788, 0.358370,
         0.425089, 0.478070, 0.921086),
        (-0.522126, -0.242686, -0.204546, -0.175998, -0.153128, -0.128446,
         -0.109484, -0.089344, -0.069576, -0.051486, -0.031847, -0.011303,
         0.008566, 0.034545, 0.059637, 0.089463, 0.124545, 0.170228, 0.225262,
         0.344677, 0.536398, 0.660700, 0.789899, 1.300320),
    ),
}

# The startle path in fusion.py only ever asks whether the label is in
# {"fear", "surprise"}, so that split is all this reproduces.
#
# Calibrated against HSEmotion's own labels on the full 7,441-crop paired corpus
# (16 VODs, 3 channels; data/eval/blendshape_arousal_labeled.pkl). HSEmotion
# calls fear-or-surprise on 15.23% of scored frames; the threshold below is the
# matching quantile of this score, so the label fires at the same base rate it
# used to.
#
# Be honest about how well this reproduces the label:
#
#     brow + eye + jawOpen (this)     AUC 0.720
#     brow + eye only                 AUC 0.605
#
# KNOWN WEAK -- this is the least trustworthy part of the swap, and a fixed
# threshold does NOT transfer across creators. The global rate matches by
# construction (15.2%) while every individual channel is wrong, in both
# directions:
#
#     channel        HSEmotion says   this predicts   AUC
#     channel_a_             49.2%            8.8%        0.720
#     channel_c         18.5%           52.2%        0.740
#     channel_b            0.8%            0.1%        0.950
#
# The RANKING is decent everywhere (AUC 0.72-0.95); what does not transfer is
# the base RATE, which is creator-dependent by a factor of 60. So a single
# constant is wrong for everyone, and per-VOD rate-matching would be equally
# wrong because it would force every creator to the same rate.
#
# Why it is still safe to run: startle additionally requires a sustained
# multi-frame run AND a synchronized acoustic onset (engines/reaction/fusion.py),
# so an over-firing label mostly produces candidates that the acoustic gate
# then rejects. An UNDER-firing label (channel_a_ above) silently loses startles,
# which is the more damaging direction and is not visible in a deck.
#
# Dropping the label and gating on fusion's arousal jump alone was TRIED and
# REJECTED: it calibrated well (1.09x the old rate, and 862 vs 911 events across
# both scorers on one VOD) but the golden no-regression gate rejected it across
# 12 fixtures, with recall falling 0.417 -> 0.000 on one. Rate-matching was the
# wrong target -- the label carries real information about WHICH jumps are
# startles. See scripts/eval_startle_rule.py for the full record. Any retry
# needs a per-frame agreement target, not a rate target.
_SURPRISE_THRESHOLD = float(os.getenv("RECALL_FACE_SURPRISE_THRESHOLD", "0.1765"))

_SURPRISE_BROW = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
_SURPRISE_EYE = ("eyeWideLeft", "eyeWideRight")


def model_path() -> str:
    from core.bundle_paths import get_models_dir

    return os.path.join(get_models_dir(), "face", "face_landmarker.task")


def _get_landmarker():
    """Process-local FaceLandmarker, or None when it cannot be loaded.

    Perception runs faces in chunked workers (parallel_perception), so each
    worker builds its own; the lock guards the in-process singleton because a
    MediaPipe IMAGE-mode detector is not safe to call concurrently.
    """
    global _LANDMARKER, _DISABLED_REASON
    if _LANDMARKER is not None or _DISABLED_REASON is not None:
        return _LANDMARKER
    with _LANDMARKER_LOCK:
        if _LANDMARKER is not None or _DISABLED_REASON is not None:
            return _LANDMARKER
        try:
            from mediapipe.tasks.python import BaseOptions, vision

            path = model_path()
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            _LANDMARKER = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=path),
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                    num_faces=1,
                    running_mode=vision.RunningMode.IMAGE,
                )
            )
        except Exception as exc:  # noqa: BLE001
            _DISABLED_REASON = str(exc)
            print(f"Emotion engine disabled: blendshape scorer unavailable "
                  f"({_DISABLED_REASON})")
            return None
    return _LANDMARKER


def reset_state() -> None:
    global _LANDMARKER, _DISABLED_REASON, _NAME_INDEX
    _LANDMARKER = None
    _DISABLED_REASON = None
    _NAME_INDEX = None


def _feature_value(scores, index: dict) -> float:
    """Raw feature, before the mapping onto HSEmotion's scale.

    ``mix`` square-roots each shape before summing: both are near zero on most
    frames with a long thin tail, and sqrt compresses that tail so one wide-open
    jaw cannot swamp the funnel term.
    """
    if FEATURE == "mouthfunnel":
        return float(scores[index["mouthFunnel"]])
    if FEATURE == "jawopen":
        return float(scores[index["jawOpen"]])
    jaw = max(0.0, float(scores[index["jawOpen"]]))
    funnel = max(0.0, float(scores[index["mouthFunnel"]]))
    return math.sqrt(jaw) + 3.0 * math.sqrt(funnel)


def _to_hsemotion_scale(raw: float) -> float:
    src, dst = _KNOTS.get(FEATURE, _KNOTS["mix"])
    return float(np.interp(raw, src, dst))


def score_face(face_rgb: np.ndarray) -> Optional[Tuple[float, float, str]]:
    """(arousal, valence, emotion) for one tight face crop, or None.

    None means "no usable face in this crop" and the caller must treat the frame
    exactly as HSEmotion's failure path does -- face_present=False -- so a
    landmarker miss can never read as a calm face.

    Valence is 0.0 by design: it is written into ReactionFrame and never read by
    any gate, score, or UI, so inventing a smile-minus-frown proxy would add an
    unvalidated signal that nothing consumes.
    """
    global _NAME_INDEX

    landmarker = _get_landmarker()
    if landmarker is None:
        return None

    try:
        import mediapipe as mp

        with _LANDMARKER_LOCK:
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
            )
    except Exception:  # noqa: BLE001 - one bad crop must not kill the pass
        return None

    if not result.face_blendshapes:
        return None
    categories = result.face_blendshapes[0]

    if _NAME_INDEX is None:
        _NAME_INDEX = {c.category_name: i for i, c in enumerate(categories)}
    index = _NAME_INDEX

    required = ("jawOpen", "mouthFunnel") + _SURPRISE_BROW + _SURPRISE_EYE
    if any(name not in index for name in required):
        return None

    scores = [c.score for c in categories]
    arousal = _to_hsemotion_scale(_feature_value(scores, index))

    brow = sum(float(scores[index[n]]) for n in _SURPRISE_BROW) / len(_SURPRISE_BROW)
    eye = sum(float(scores[index[n]]) for n in _SURPRISE_EYE) / len(_SURPRISE_EYE)
    jaw = float(scores[index["jawOpen"]])
    # jawOpen carries most of the discrimination here (AUC 0.605 -> 0.697):
    # a startled gasp opens the mouth, and brow/eye alone do not separate it
    # from ordinary expressive talking.
    surprise = (brow + eye + jaw) / 3.0
    emotion = "surprise" if surprise >= _SURPRISE_THRESHOLD else "neutral"

    return arousal, 0.0, emotion
