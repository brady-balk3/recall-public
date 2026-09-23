# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from core.bundle_paths import get_data_dir
from core.models.adapters import (
    asset_from_video,
    clips_from_legacy,
    events_from_legacy,
    exports_from_paths,
    signals_from_legacy,
    stories_from_legacy,
)
from core.models.canonical import SCHEMA_VERSION, validate_traceability

ARTIFACT_VERSION = "1"
CANDIDATES_FILENAME = "candidates.v1.json"
REACTION_CURVE_FILENAME = "reaction_curve.v1.json"
CHAT_EVIDENCE_FILENAME = "chat_evidence.v1.json"

# Trace fields renamed 2026-07-26 when "heuristic" was split into its three
# meanings (docs/ACCEPTANCE_CRITERIA.md). The scalar these hold is a weighted
# sum of perception signals, not a model, so it is now signal_*.
#
# Every candidates.v1.json written before that commit carries the old spelling,
# and those traces are not regenerable: the transcript cache is keyed on the
# asset file, so a job whose VOD has been deleted can never be rescanned. That
# is the same failure that forced the v7 feature backfill. Readers of PERSISTED
# traces must therefore go through upgrade_trace_row; live in-run dicts never
# need it, because selection writes the new key before anything reads it.
LEGACY_TRACE_KEYS = {
    "heuristic_selection_score": "signal_selection_score",
    "heuristic_editorial_score": "signal_editorial_score",
    "heuristic_percentile": "signal_percentile",
}

# The word was also a persisted VALUE in three places, so grouping a mixed-age
# corpus by any of them would silently split one population into two buckets
# -- the sort of quiet miscount that makes an eval look like it moved.
#   ranker_policy            "heuristic" meant "no ranker ran at all"
#   selection_funnel.scoring "heuristic" meant the same at the funnel stage
#   reasons[]                "heuristic_challenger" is a candidate-pool lane
LEGACY_TRACE_VALUES = {"heuristic": "signals"}
LEGACY_REASONS = {"heuristic_challenger": "signals_challenger"}

# ``candidates.v1.json`` intentionally abbreviates the expensive visual-judge
# fields. Selection-only replays must restore the live names before invoking
# the selector; otherwise a judged candidate quietly becomes unjudged and the
# replay measures a different policy than the completed scan.
TRACE_VISUAL_TO_RUNTIME = {
    "visual_verdict": "visual_semantic_verdict",
    "visual_outcome": "visual_semantic_outcome",
    "visual_onscreen_text": "visual_semantic_onscreen_text",
    "visual_summary": "visual_semantic_summary",
    "visual_evidence": "visual_semantic_evidence",
    "visual_moment_type": "visual_semantic_moment_type",
    "visual_model_verdict": "visual_semantic_model_verdict",
    "visual_confidence": "visual_semantic_confidence",
    "visual_payoff": "visual_semantic_payoff",
    "visual_self_contained": "visual_semantic_self_contained",
    "visual_hook": "visual_semantic_hook",
    "visual_routine_only": "visual_semantic_routine_only",
    "visual_verbal_payoff": "visual_semantic_verbal_payoff",
    "visual_context_complete": "visual_semantic_context_complete",
    "visual_event": "visual_semantic_visual_event",
    "visual_streamer_reaction": "visual_semantic_streamer_reaction",
}

TRACE_TRANSIENT_SELECTION_FIELDS = {
    "disposition",
    "reasons",
    "selection_rank",
    "selection_score",
    "selection_disposition",
    "selection_rejection_reasons",
    "selection_funnel",
    "selection_suppressed_by",
    "moment_group_id",
    "moment_group_source",
    "content_anchor_capped",
    "chat_reaction_only",
    "hook_score",
    "ranker_score",
    "ranker_publishability_score",
    "personal_ranker_score",
    "personal_preference_delta",
    "personal_challenger_score",
    "personal_challenger_delta",
    "signal_selection_score",
    "signal_editorial_score",
    "ranker_percentile",
    "personal_ranker_percentile",
    "signal_percentile",
    "protected",
    "clears_bar",
}


def upgrade_trace_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fill new-name trace keys/values from their pre-rename spelling, in place.

    Only fills when the new key is genuinely absent -- a row that already
    carries it is left alone, so re-upgrading is a no-op and a fresh trace is
    never overwritten by a stale legacy field. Returns the same dict for
    convenient chaining.
    """
    if not isinstance(row, dict):
        return row
    for legacy, current in LEGACY_TRACE_KEYS.items():
        if current not in row and legacy in row:
            row[current] = row[legacy]

    policy = row.get("ranker_policy")
    if isinstance(policy, str) and policy in LEGACY_TRACE_VALUES:
        row["ranker_policy"] = LEGACY_TRACE_VALUES[policy]

    funnel = row.get("selection_funnel")
    if isinstance(funnel, dict):
        scoring = funnel.get("scoring")
        if isinstance(scoring, str) and scoring in LEGACY_TRACE_VALUES:
            funnel["scoring"] = LEGACY_TRACE_VALUES[scoring]

    reasons = row.get("reasons")
    if isinstance(reasons, list) and any(r in LEGACY_REASONS for r in reasons):
        row["reasons"] = [LEGACY_REASONS.get(r, r) for r in reasons]
    return row


def upgrade_trace_rows(
    rows: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """upgrade_trace_row over a candidate list, reporting whether any row was
    written before the rename. Callers print the flag ONCE rather than per row
    -- a silent fallback is exactly what docs/ACCEPTANCE_CRITERIA.md §3.8
    exists to prevent.
    """
    upgraded = [upgrade_trace_row(row) for row in rows]
    legacy_seen = any(
        isinstance(row, dict)
        and any(key in row for key in LEGACY_TRACE_KEYS)
        for row in rows
        if isinstance(row, dict)
    )
    return upgraded, legacy_seen


def hydrate_candidate_trace_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Restore one persisted candidate to the selector's live input shape.

    The returned row is a deep copy. Historical selection outputs are removed
    so the current selector recomputes them, while immutable perception,
    semantic, visual-judge, and feature-snapshot evidence remains frozen.
    """
    candidate = copy.deepcopy(row)
    upgrade_trace_row(candidate)
    if (
        candidate.get("selection_start") is not None
        and candidate.get("selection_end") is not None
    ):
        candidate["start"] = float(candidate["selection_start"])
        candidate["end"] = float(candidate["selection_end"])
    features = candidate.get("features") or {}

    for trace_name, runtime_name in TRACE_VISUAL_TO_RUNTIME.items():
        if candidate.get(runtime_name) is None and trace_name in candidate:
            candidate[runtime_name] = candidate.get(trace_name)

    for name in (
        "reaction_auc",
        "signal_agreement",
        "dead_air_ratio",
        "active_channel_count",
        "lead_reaction",
        "settle_reaction",
        "vod_median",
        "vod_mad",
        "speech_rate_delta",
        "exclaim_ratio",
    ):
        if candidate.get(name) is None and features.get(name) is not None:
            candidate[name] = float(features[name])
    if candidate.get("vod_duration") is None:
        duration_norm = features.get("vod_duration_norm")
        if duration_norm is not None:
            candidate["vod_duration"] = float(duration_norm) * 3600.0
    if candidate.get("modality_count") is None:
        breakdown = candidate.get("modality_breakdown") or {}
        candidate["modality_count"] = sum(
            1
            for name in ("voice", "face", "speech", "burst", "motion", "game_audio")
            if float(breakdown.get(name, 0.0) or 0.0) > 0.05
        )

    for name in TRACE_TRANSIENT_SELECTION_FIELDS:
        candidate.pop(name, None)
    return candidate


def match_candidate_features(
    job_id: str,
    start: float,
    end: float,
    *,
    min_cover: float = 0.5,
    data_dir: str | None = None,
) -> Dict[str, Any] | None:
    """Match a manual window to a persisted scan candidate, without replay.

    Coverage is measured against the shorter window so a creator's tighter or
    wider cut can still supervise the candidate that found the same moment.
    Missing/corrupt artifacts are an ordinary old-job condition and return
    ``None``; manual clip creation must never depend on this optional match.
    """
    path = os.path.join(
        data_dir or get_data_dir(), "jobs", str(job_id), CANDIDATES_FILENAME,
    )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            candidates = (json.load(handle) or {}).get("candidates") or []
    except (OSError, TypeError, ValueError):
        return None

    start = float(start)
    end = float(end)
    manual_span = max(0.001, end - start)
    best = None
    best_key = (-1.0, -1.0)
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("features"), dict):
            continue
        try:
            candidate_start = float(candidate["start"])
            candidate_end = float(candidate["end"])
        except (KeyError, TypeError, ValueError):
            continue
        candidate_span = max(0.001, candidate_end - candidate_start)
        overlap = max(0.0, min(end, candidate_end) - max(start, candidate_start))
        cover = overlap / min(manual_span, candidate_span)
        union = manual_span + candidate_span - overlap
        iou = overlap / max(0.001, union)
        if (cover, iou) > best_key:
            best_key = (cover, iou)
            best = candidate
    if best is None or best_key[0] < max(0.0, min(1.0, float(min_cover))):
        return None
    return {
        "features": dict(best["features"]),
        "cover_fraction": best_key[0],
        "iou": best_key[1],
        "candidate_start": float(best["start"]),
        "candidate_end": float(best["end"]),
        "candidate_disposition": best.get("disposition"),
        "candidate_reasons": list(best.get("reasons") or []),
    }


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return _safe(asdict(value))
    if hasattr(value, "to_dict"):
        return _safe(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hash_file(path: str) -> str | None:
    """Cheap content fingerprint for the manifest. Hashing a whole multi-GB VOD
    after every job cost minutes of pure disk I/O for a field nothing verifies
    byte-for-byte; a signature over size + mtime + the head and tail 1 MB is
    effectively as unique for provenance while reading only ~2 MB (plan 5.1)."""
    try:
        size = os.path.getsize(path)
        h = hashlib.sha256()
        h.update(f"{size}:{os.path.getmtime(path)}".encode("utf-8"))
        window = 1024 * 1024
        with open(path, "rb") as f:
            h.update(f.read(window))
            if size > window:
                f.seek(max(0, size - window))
                h.update(f.read(window))
        return h.hexdigest()
    except OSError:
        return None


def _hash_json(value: Any) -> str:
    raw = json.dumps(_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe(value), f, indent=2, sort_keys=True)


def write_candidates_artifact(
    job_id: str,
    selection_trace: Iterable,
    diagnostics: Dict[str, Any] | None = None,
) -> str:
    """Persist the per-candidate selection audit for a completed scan.

    One row per candidate: the reaction engine's selection_trace summary
    (window, scores, disposition, rejection reasons) plus the canonical
    feature dict and modality breakdown. This is the offline training/eval
    capture for the label loop (HUMAN_CLIPS Package 1), NOT a debug dump, so
    unlike write_job_artifacts it is NOT gated on the debugArtifacts setting:
    a packaged install that silently stops capturing candidates starves the
    learning loop of exactly the data it exists to collect. Size stays bounded
    — ~25 floats plus a few short strings per candidate.
    """
    artifact_dir = os.path.join(get_data_dir(), "jobs", job_id)
    os.makedirs(artifact_dir, exist_ok=True)
    payload = {
        "version": 1,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostics": diagnostics or {},
        "candidates": list(selection_trace or []),
    }
    path = os.path.join(artifact_dir, CANDIDATES_FILENAME)
    _write_json(path, payload)
    return path


def write_reaction_curve_artifact(
    job_id: str,
    timeline: Iterable[float],
    score: Iterable[float],
) -> str:
    """Persist the fused reaction curve that candidate generation thresholds.

    ``detect_arcs`` runs on this curve, and a candidate exists only where it
    crosses ``median + k * MAD`` of the curve's own distribution. Nothing
    below that line becomes a candidate, so it can never be labelled,
    rejected, or counted — which made the cost of the generation threshold
    unmeasurable from any artifact on disk (measured 2026-08-15: the curve was
    reconstructible only to 57% agreement from ``signals.v1.json``, which is
    not close enough to re-run the detector against).

    Ungated for the same reason as ``write_candidates_artifact``: this is the
    decision variable for the earliest and largest loss stage in the pipeline,
    and a packaged install that stops recording it cannot answer questions
    about candidate generation at all. Size is one float pair per frame, ~1 Hz
    — a few hundred KB on a four-hour VOD, against 5 MB of candidates.
    """
    artifact_dir = os.path.join(get_data_dir(), "jobs", job_id)
    os.makedirs(artifact_dir, exist_ok=True)
    timeline = [round(float(value), 3) for value in timeline]
    score = [round(float(value), 5) for value in score]
    if len(timeline) != len(score):
        raise ValueError(
            f"timeline/score length mismatch: {len(timeline)} vs {len(score)}"
        )
    payload = {
        "version": 1,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame_count": len(score),
        "timeline": timeline,
        "score": score,
    }
    path = os.path.join(artifact_dir, REACTION_CURVE_FILENAME)
    _write_json(path, payload)
    return path


def write_chat_evidence_artifact(job_id: str, evidence: Dict[str, Any]) -> str:
    """Persist bounded anonymous Twitch-chat evidence for Stream Memory.

    The summarizer owns the privacy contract; this writer only adds durable
    job provenance. Like candidate capture, this small derived artifact is not
    gated on debug mode because packaged scans need to remain searchable.
    """
    artifact_dir = os.path.join(get_data_dir(), "jobs", job_id)
    os.makedirs(artifact_dir, exist_ok=True)
    payload = {
        **dict(evidence or {}),
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(artifact_dir, CHAT_EVIDENCE_FILENAME)
    _write_json(path, payload)
    return path


def write_job_artifacts(
    *,
    job_id: str,
    source_path: str,
    settings: Dict[str, Any] | None,
    signals: Iterable,
    events: Iterable,
    stories: Iterable,
    clips: Iterable,
    captions: Iterable,
    exported_paths: Iterable[str],
    duration: float,
    stage_timings: Dict[str, Any] | None = None,
    ranker_mode: str | None = None,
    ranker_diagnostics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    artifact_dir = os.path.join(get_data_dir(), "jobs", job_id)
    os.makedirs(artifact_dir, exist_ok=True)

    asset_id = f"asset_{hashlib.sha1(os.path.abspath(source_path).encode('utf-8'), usedforsecurity=False).hexdigest()[:12]}"
    asset = asset_from_video(asset_id, source_path, duration)
    canonical_signals = signals_from_legacy(signals, asset_id)
    canonical_events = events_from_legacy(events, asset_id)
    canonical_stories = stories_from_legacy(stories, asset_id)
    canonical_clips = clips_from_legacy(clips, asset_id)
    canonical_exports = exports_from_paths(exported_paths, clips, asset_id)

    files = {
        "asset": "asset.v1.json",
        "signals": "signals.v1.json",
        "events": "events.v1.json",
        "stories": "stories.v1.json",
        "clips": "clips.v1.json",
        "captions": "captions.v1.json",
        "exports": "exports.v1.json",
    }
    payloads = {
        "asset": asset,
        "signals": canonical_signals,
        "events": canonical_events,
        "stories": canonical_stories,
        "clips": canonical_clips,
        "captions": list(captions or []),
        "exports": canonical_exports,
    }
    for key, filename in files.items():
        _write_json(os.path.join(artifact_dir, filename), payloads[key])

    all_canonical = [asset] + canonical_signals + canonical_events + canonical_stories + canonical_clips + canonical_exports
    manifest = {
        "version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "source": {
            "path": source_path,
            "sha256": _hash_file(source_path),
            "duration": float(duration or 0.0),
        },
        "schema_versions": {
            "canonical": SCHEMA_VERSION,
            "artifact": ARTIFACT_VERSION,
        },
        "config_hash": _hash_json(settings or {}),
        "model_versions": {
            "vision_cache": "vision-facecam-panel-v9-scene-layout",
            # Never claim personalization was active unless the runtime that
            # produced this artifact explicitly reports it.
            "ranker": ranker_mode or "not-recorded",
        },
        "clip_intelligence": {
            "ranker": ranker_diagnostics or {},
        },
        "stage_timings": stage_timings or {},
        "artifacts": {
            key: os.path.join(artifact_dir, filename)
            for key, filename in files.items()
        },
        "traceability_errors": validate_traceability(all_canonical),
    }
    _write_json(os.path.join(artifact_dir, "manifest.json"), manifest)
    return manifest
