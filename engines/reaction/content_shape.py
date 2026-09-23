# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""VOD content-shape inference (not per-game profiles).

Infers one of a small set of shapes from scan-time evidence so selection can
admit story beats on narrative VODs without a catalog of game titles.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

SHAPES = ("narrative", "chaos_comedy", "competitive_mixed")
DEFAULT_SHAPE = "competitive_mixed"

_EVENT_LABELS = frozenset(
    {"elimination", "knock", "rank_progress", "terminal_win", "generic_highlight"}
)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _visual_moment_type(c: dict) -> str:
    raw = c.get("visual_semantic_moment_type")
    if raw is None:
        raw = c.get("visual_moment_type")
    if raw is None:
        raw = c.get("semantic_moment_type")
    return str(raw or "").strip().lower()


def summarize_candidates(candidates: Iterable[dict]) -> Dict[str, Any]:
    """Aggregate scan evidence used by ``infer_content_shape``."""
    rows = list(candidates or [])
    peaks: List[float] = []
    faces: List[float] = []
    voices: List[float] = []
    startles: List[float] = []
    moment_types: Dict[str, int] = {}
    verbal_complete = 0
    visual_judged = 0
    storyish = 0
    funny_fail = 0
    event_labeled = 0
    for c in rows:
        peaks.append(float(c.get("peak_value", 0.0) or 0.0))
        mb = c.get("modality_breakdown") or {}
        faces.append(float(mb.get("face", 0.0) or 0.0))
        voices.append(float(mb.get("voice", 0.0) or 0.0))
        startles.append(float(mb.get("startle", 0.0) or 0.0))
        if c.get("game_label") in _EVENT_LABELS:
            event_labeled += 1
        verdict = str(
            c.get("visual_semantic_verdict") or c.get("visual_verdict") or ""
        ).strip().lower()
        if not verdict:
            continue
        visual_judged += 1
        mt = _visual_moment_type(c)
        moment_types[mt] = moment_types.get(mt, 0) + 1
        if mt == "story":
            storyish += 1
        if mt in ("funny", "fail", "rage"):
            funny_fail += 1
        if (
            bool(c.get("visual_semantic_verbal_payoff"))
            and bool(c.get("visual_semantic_context_complete"))
            and not bool(c.get("visual_semantic_routine_only"))
        ):
            verbal_complete += 1

    n = max(1, len(rows))
    return {
        "candidate_count": len(rows),
        "mean_peak": _mean(peaks),
        "mean_face": _mean(faces),
        "mean_voice": _mean(voices),
        "mean_startle": _mean(startles),
        "high_startle_rate": sum(1 for s in startles if s >= 0.45) / n,
        "event_label_rate": event_labeled / n,
        "visual_judged": visual_judged,
        "story_rate": (storyish / visual_judged) if visual_judged else 0.0,
        "funny_fail_rate": (funny_fail / visual_judged) if visual_judged else 0.0,
        "verbal_complete_rate": (verbal_complete / visual_judged) if visual_judged else 0.0,
        "moment_types": moment_types,
    }


def infer_content_shape(
    candidates: Iterable[dict],
    *,
    game: Optional[str] = None,
) -> Dict[str, Any]:
    """Rule-based shape inference from scan evidence only.

    ``game`` is accepted for call-site compatibility but ignored — shape must
    not depend on a title catalog.
    """
    del game  # intentionally unused
    stats = summarize_candidates(candidates)
    scores = {
        "narrative": 0.0,
        "chaos_comedy": 0.0,
        "competitive_mixed": 0.0,
    }
    evidence: List[str] = []

    # Narrative: quiet peaks + story/verbal density, few competitive OCR events.
    if stats["mean_peak"] < 1.35:
        scores["narrative"] += 1.0
        evidence.append("low_mean_peak")
    if stats["story_rate"] >= 0.20:
        scores["narrative"] += 2.0
        evidence.append("story_moment_density")
    if stats["verbal_complete_rate"] >= 0.15:
        scores["narrative"] += 1.5
        evidence.append("verbal_complete_density")
    if stats["event_label_rate"] < 0.08 and stats["mean_peak"] < 1.6:
        scores["narrative"] += 0.75
        evidence.append("sparse_ocr_events")

    # Chaos / comedy: loud face/voice/startle, funny/fail moments.
    if stats["mean_face"] >= 0.12 or stats["high_startle_rate"] >= 0.05:
        scores["chaos_comedy"] += 1.5
        evidence.append("face_or_startle_density")
    if stats["mean_peak"] >= 1.8:
        scores["chaos_comedy"] += 1.25
        evidence.append("high_mean_peak")
    if stats["funny_fail_rate"] >= 0.20:
        scores["chaos_comedy"] += 2.0
        evidence.append("funny_fail_density")
    if stats["mean_voice"] >= 0.25 and stats["mean_face"] >= 0.08:
        scores["chaos_comedy"] += 0.75
        evidence.append("voice_face_reaction")

    # Competitive: OCR/event labels and mid arousal without comedy dominance.
    if stats["event_label_rate"] >= 0.10:
        scores["competitive_mixed"] += 2.0
        evidence.append("ocr_event_density")
    if 1.0 <= stats["mean_peak"] <= 2.2 and stats["funny_fail_rate"] < 0.25:
        scores["competitive_mixed"] += 0.75
        evidence.append("mid_arousal_band")

    # Stable default when nothing fires.
    if max(scores.values()) < 0.75:
        shape = DEFAULT_SHAPE
        confidence = 0.25
        evidence.append("default_low_evidence")
    else:
        shape = max(scores, key=scores.get)
        top = scores[shape]
        second = sorted(scores.values(), reverse=True)[1]
        confidence = round(min(1.0, 0.35 + (top - second) * 0.25 + top * 0.15), 3)

    return {
        "shape": shape,
        "confidence": confidence,
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "evidence": evidence,
        "stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()},
        "game_prior": None,
    }


def content_shape_policy_enabled(settings: Optional[dict]) -> bool:
    """Content-shape adaptation is a standard part of selection."""
    del settings  # retained for call-site compatibility
    return True
