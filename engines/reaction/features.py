# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Canonical clip feature vector for the learned ranker (plan §5.6).

ONE definition of the feature vector, shared by:
  * label capture (stored with each clip when it's produced),
  * training (scripts/train_ranker.py, build_base_ranker.py),
  * inference (ranker.py, used in selection).

Keep ``FEATURE_NAMES`` and ``feature_vector`` in lock-step and versioned: if you
change the layout, bump FEATURE_VERSION so stale stored vectors are ignored.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np

FEATURE_VERSION = 8  # Remove the visual-judge routing flag from model input.
# Feature-version compatibility is semantic, not merely key compatibility.
# v8 is a deletion-only migration from v7; future bumps must opt in explicitly.
LOSSLESS_FEATURE_MIGRATIONS = frozenset({(8, 7)})
# Every v6 feature is a summary statistic the SIGNAL SCORE already computes, so a
# model over them can only re-weight the signal score's own worldview -- measured
# 2026-07-26, quadrupling the training data lifted pairwise accuracy 0.6426 ->
# 0.7051 while deck precision sat still at 0.223 -> 0.224. v7 adds two scalars
# the signal score never computes, both read off the transcript:
#
#   31 features (v6)              pairwise 0.6702   tail_AUC 0.5721
#   + speech_rate_delta/exclaim   pairwise 0.7131   tail_AUC 0.6197
#
# i.e. two features bought two-thirds of what 4x the labels bought. Three other
# candidates were measured and REJECTED: words_per_sec scored BELOW baseline on
# both metrics, speech_after_end likewise (whether speech continues past the
# cut does not predict a keep), and ends_on_terminal was additive-neutral.
#
# NOT soft-paddable (see ensure_feature_layout): unlike the v6 content slots,
# where 0.0 honestly means "the VLM never looked", 0.0 here is a real
# measurement -- "flat delivery", "no exclamations". Padding historical rows
# with it would teach the model that 63% of clips were flatly delivered. Rows
# that cannot be reconstructed from trace + transcript are dropped instead.

# Ordinal strength of the confirmed game event (None -> 0).
GAME_TIER_ORDINAL = {
    None: 0.0,
    "generic_highlight": 0.2,
    "rank_progress": 0.4,
    "knock": 0.6,
    "elimination": 0.8,
    "terminal_win": 1.0,
}

MOMENT_TYPE_ORDINAL = {
    None: 0.0,
    "filler": 0.0,
    "story": 0.45,
    "wholesome": 0.65,
    "fail": 0.70,
    "rage": 0.75,
    "scare": 0.85,
    "funny": 0.95,
    "clutch": 1.0,
}

# v6 content-axis slots — soft-padded to 0.0 when upgrading v5 label rows.
V6_CONTENT_FEATURES = (
    "visual_win",
    "visual_routine_only",
    "visual_outcome_strength",
    "content_interest",
    "content_anchor",
)

FEATURE_NAMES = [
    "face",            # mean facecam arousal over the clip (0..~0.4)
    "voice",           # mean voice arousal (0..1)
    "speech",          # max speech hype (0..1)
    "game",            # game-evidence strength (0..1)
    "peak_value",      # R(t) peak height (per-VOD normalized)
    "mean_intensity",  # reaction_auc / duration (sustained intensity)
    "reaction_auc",    # integrated R(t) over the clip
    "duration",        # clip length in seconds
    "game_tier",       # ordinal of game_label
    "is_win",          # 1.0 if terminal_win else 0.0
    "peak_position",   # 0=start, 1=end; human clips prefer payoff after setup
    "payoff_position", # game anchor position, else peak position
    "lead_reaction",   # mean R(t) before payoff/peak
    "settle_reaction", # mean R(t) after payoff/peak
    "dead_air_ratio",  # fraction of clip with no reaction/game evidence
    "signal_agreement",# fraction of modalities contributing meaningful evidence
    # --- HUMAN_CLIPS feature v5: cold-open + semantic editorial read ---
    "hook_score",          # first-five-second signal/judge blend (0..1)
    "self_contained",      # understandable without outside stream context
    "payoff",              # whether something actually lands
    "moment_type_ordinal", # filler=0; clutch/funny are highest
    # --- VOD context (plan 22 §4.3): what "high" means in this VOD ---
    "vod_median",          # median of the fused R(t) across the whole VOD
    "vod_mad",             # MAD of R(t) — flat VOD vs constant-hype VOD
    "active_channel_count",# how many reaction channels this VOD had (1..5)
    "vod_duration_norm",   # VOD length in hours, capped at 12
    # --- Train-only (plan 22 §4.3): ALWAYS 0.0 at inference ---
    "presentation_rank",   # normalized review-order rank at label time
    # --- Content axis v6: grounded visual-judge evidence ---
    "visual_win",              # 1 if confirmed match_win (text-checked)
    "visual_routine_only",     # 1 if VLM flagged routine/filler-only
    "visual_outcome_strength", # match_win=1, elimination=0.6, death=0.5
    "content_interest",        # 0..1 selection content_interest_score
    "content_anchor",          # 1 if strong enough for arousal-gate bypass
    # --- Delivery v7: how the moment is SPOKEN, which no v6 feature sees ---
    "speech_rate_delta",       # words/sec in the 2nd half minus the 1st
    "exclaim_ratio",           # ? and ! per word (Whisper punctuation as heat)
]

# v7 delivery slots. Deliberately NOT in the soft-pad set: 0.0 is a real value
# here, so a missing row must be reconstructed or dropped, never padded.
V7_DELIVERY_FEATURES = (
    "speech_rate_delta",
    "exclaim_ratio",
)

_TERMINAL_HEAT = ("!", "?")


def delivery_features(window_words) -> Dict[str, float]:
    """The v7 delivery scalars for one candidate window.

    ``window_words`` is a sequence of ``(text, start)`` pairs already sliced to
    the window -- the shape produced by
    ``engines.semantic.judge._transcript_words``. Shared by scan-time
    computation and by the training-time backfill so both produce identical
    values from identical inputs.
    """
    words = [w for w in (window_words or []) if w]
    if len(words) < 2:
        return {"speech_rate_delta": 0.0, "exclaim_ratio": 0.0}
    starts = [float(w[1]) for w in words]
    lo, hi = starts[0], starts[-1]
    half = max(1e-3, (hi - lo) / 2.0)
    mid = lo + half
    first = sum(1 for s in starts if s < mid)
    second = len(starts) - first
    heat = sum(str(w[0]).count(ch) for w in words for ch in _TERMINAL_HEAT)
    return {
        "speech_rate_delta": (second - first) / half,
        "exclaim_ratio": heat / float(len(words)),
    }


def _as_get(clip):
    return clip.get if isinstance(clip, dict) else (lambda k, d=None: getattr(clip, k, d))


def _visual_view(clip) -> dict:
    """Normalize full + abbreviated (selection_trace) visual field names."""
    get = _as_get(clip)
    view = dict(clip) if isinstance(clip, dict) else {}
    # Abbreviated keys from candidates.v1.json selection_trace rows.
    if get("visual_semantic_verdict") is None and get("visual_verdict") is not None:
        view["visual_semantic_verdict"] = get("visual_verdict")
    if get("visual_semantic_outcome") is None and get("visual_outcome") is not None:
        view["visual_semantic_outcome"] = get("visual_outcome")
    if get("visual_semantic_onscreen_text") is None and get("visual_onscreen_text") is not None:
        view["visual_semantic_onscreen_text"] = get("visual_onscreen_text")
    if "visual_semantic_routine_only" not in view and get("visual_routine_only") is not None:
        view["visual_semantic_routine_only"] = get("visual_routine_only")
    return view


def content_axis_features(clip) -> Dict[str, float]:
    """v6 content-axis slots derived from visual-judge / selection fields."""
    # Local import avoids a hard cycle at module import time; selection does
    # not import features.
    from engines.reaction.selection import (
        DEFAULT_TUNING,
        _content_anchor,
        _visual_win_read,
        content_interest_score,
    )

    get = _as_get(clip)
    view = _visual_view(clip)
    verdict = view.get("visual_semantic_verdict")
    judged = verdict is not None and str(verdict).strip() != ""
    if not judged:
        return {name: 0.0 for name in V6_CONTENT_FEATURES}

    tuning = replace(DEFAULT_TUNING, content_axis=True)
    win = 1.0 if _visual_win_read(view) else 0.0
    routine = 1.0 if bool(view.get("visual_semantic_routine_only")) else 0.0
    outcome = str(view.get("visual_semantic_outcome") or "").lower()
    if outcome == "match_win" and not win:
        outcome = "none"
    outcome_strength = {
        "match_win": 1.0,
        "elimination": 0.6,
        "death_or_fail": 0.5,
    }.get(outcome, 0.0)

    # Prefer the score selection already wrote; otherwise recompute.
    interest_raw = get("content_interest", None)
    if interest_raw is None:
        interest = float(content_interest_score(view, tuning))
    else:
        interest = float(interest_raw or 0.0)

    anchor_raw = get("content_anchor", None)
    if anchor_raw is None:
        anchor = 1.0 if _content_anchor(view, tuning) else 0.0
    else:
        anchor = 1.0 if bool(anchor_raw) else 0.0

    return {
        "visual_win": win,
        "visual_routine_only": routine,
        "visual_outcome_strength": outcome_strength,
        "content_interest": max(0.0, min(1.0, interest)),
        "content_anchor": anchor,
    }


def feature_dict(clip) -> Dict[str, float]:
    """Build the named feature dict from a ReactionClip (or a dict with same keys)."""
    get = _as_get(clip)
    mb = get("modality_breakdown", {}) or {}
    start = float(get("start", 0.0) or get("start_time", 0.0) or 0.0)
    end = float(get("end", 0.0) or get("end_time", 0.0) or 0.0)
    duration = max(1.0, end - start)
    auc = float(get("reaction_auc", 0.0))
    label = get("game_label", None)
    peak_ts = float(get("peak_timestamp", start) or start)
    anchor_ts = get("game_anchor_time", None)
    anchor_ts = float(anchor_ts) if anchor_ts is not None else peak_ts
    values = {
        "face": float(mb.get("face", 0.0)),
        "voice": float(mb.get("voice", 0.0)),
        "speech": float(mb.get("speech", 0.0)),
        "game": float(get("game_evidence", 0.0) or mb.get("game", 0.0)),
        "peak_value": float(get("peak_value", 0.0) or 0.0),
        "mean_intensity": auc / duration,
        "reaction_auc": auc,
        "duration": duration,
        "game_tier": GAME_TIER_ORDINAL.get(label, 0.0),
        "is_win": 1.0 if label == "terminal_win" else 0.0,
        "peak_position": max(0.0, min(1.0, (peak_ts - start) / duration)),
        "payoff_position": max(0.0, min(1.0, (anchor_ts - start) / duration)),
        "lead_reaction": float(get("lead_reaction", 0.0) or 0.0),
        "settle_reaction": float(get("settle_reaction", 0.0) or 0.0),
        "dead_air_ratio": float(get("dead_air_ratio", 1.0)),
        "signal_agreement": float(get("signal_agreement", 0.0) or 0.0),
        "hook_score": float(get("hook_score", 0.0) or 0.0),
        "self_contained": float(get("semantic_self_contained", 0.0) or 0.0),
        "payoff": float(get("semantic_payoff", 0.0) or 0.0),
        "moment_type_ordinal": MOMENT_TYPE_ORDINAL.get(
            get("semantic_moment_type", None), 0.0,
        ),
        "vod_median": float(get("vod_median", 0.0) or 0.0),
        "vod_mad": float(get("vod_mad", 0.0) or 0.0),
        "active_channel_count": float(get("active_channel_count", 1.0) or 1.0),
        "vod_duration_norm": min(12.0, float(get("vod_duration", 0.0) or 0.0) / 3600.0),
        # Never populated from clip data: the trainer overwrites this column
        # from the label row; inference standardization sees a constant 0.
        "presentation_rank": 0.0,
        # Delivery v7. Computed upstream (reaction_main slices the transcript
        # once per candidate) and carried on the candidate, because the clip
        # dict alone does not hold the words. Absent -> 0.0, which is why
        # ensure_feature_layout refuses to accept a row that merely OMITS
        # them: a real 0.0 and a missing 0.0 must not be confused.
        "speech_rate_delta": float(get("speech_rate_delta", 0.0) or 0.0),
        "exclaim_ratio": float(get("exclaim_ratio", 0.0) or 0.0),
    }
    values.update(content_axis_features(clip))
    return values


def feature_vector(clip) -> np.ndarray:
    """Fixed-order feature vector matching FEATURE_NAMES."""
    fd = feature_dict(clip)
    return np.array([fd[name] for name in FEATURE_NAMES], dtype=np.float64)


def ensure_feature_layout(features: Optional[dict]) -> Optional[Dict[str, float]]:
    """Return a FEATURE_NAMES-complete dict, or None if too incomplete.

    Historical v5 Keep/Pass rows are soft-padded with 0.0 for the v6
    content-axis keys so the training corpus is not wiped by the bump.
    Rows missing any older required key are still dropped.

    The v7 delivery keys are deliberately NOT soft-paddable. For the v6
    content slots 0.0 is an honest absence ("the VLM never judged this"); for
    speech_rate_delta / exclaim_ratio it is a real measurement ("flat
    delivery", "no exclamations"), so padding would teach the model something
    false about every historical row. Callers that can rebuild them from a
    trace + transcript should do so before calling here (see
    scripts.train_base_ranker.backfill_delivery_features); anything left
    unreconstructed is dropped, loudly, by the caller.
    """
    if not isinstance(features, dict) or not features:
        return None
    missing = [name for name in FEATURE_NAMES if name not in features]
    if not missing:
        return {name: float(features[name]) for name in FEATURE_NAMES}
    if set(missing) <= set(V6_CONTENT_FEATURES):
        out = dict(features)
        for name in missing:
            out[name] = 0.0
        return {name: float(out[name]) for name in FEATURE_NAMES}
    return None


def stored_feature_version_is_compatible(version: int) -> bool:
    """Whether a persisted feature row has an explicit lossless migration."""
    value = int(version)
    return (
        value == FEATURE_VERSION
        or (FEATURE_VERSION, value) in LOSSLESS_FEATURE_MIGRATIONS
    )


def ensure_stored_feature_layout(
    features: Optional[dict],
    version: int,
) -> Optional[Dict[str, float]]:
    """Migrate a versioned persisted row without soft-padding its claims."""
    if not stored_feature_version_is_compatible(version):
        return None
    if not isinstance(features, dict) or any(
        name not in features for name in FEATURE_NAMES
    ):
        return None
    return {name: float(features[name]) for name in FEATURE_NAMES}


def merge_content_axis_into_features(
    features: dict,
    content_source: dict,
) -> Dict[str, float]:
    """Overlay v6 content-axis values onto an existing feature dict."""
    base = dict(features or {})
    base.update(content_axis_features(content_source))
    ensured = ensure_feature_layout(base)
    if ensured is None:
        # Still return padded content keys even if older keys are incomplete —
        # callers that need a full vector should use ensure_feature_layout.
        for name in V6_CONTENT_FEATURES:
            base.setdefault(name, 0.0)
        return base
    return ensured
