# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Shared helpers for the Second-look / more-moments review tier.

Ceiling-cut candidates live in ``candidates.v1.json`` after selection. The scan
export pass renders the first batch alongside the primary deck; Theater Review
only reveals them. Stable ids keep scan-time rows and later reveal calls
idempotent.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional

from core.artifacts import hydrate_candidate_trace_row
from engines.reaction import selection as _selection

MORE_CANDIDATE_LABEL = "recall_more_candidate"
# How many ceiling-cut moments the Second-look tier offers.
#
# MEASURED 2026-07-25 against the creator's own keep/pass decisions (245
# reviewed Second-look cards): the tier keeps at 16.3% vs 35.8% for Primary,
# and its INTERNAL ORDER carries almost no signal — rank 1 keeps at 11%, rank
# 10 at 31%, rank 3 at 39%, rank 5 at 6%. Because the order is noise,
# truncating trades finds for cards close to linearly, so the only question is
# where the dilution stops being worth it. Combined keep rate by tier size:
#
#     15 (was)   30.9%      40/40 finds
#     10         32.3%      30/40
#      5         34.0%      17/40
#      3         35.2%      15/40
#      0         35.8%       0/40   (Primary alone)
#
# BUT that measured only the REVIEW axis. This tier is also the only mechanism
# that labels the bottom half of the ranker's ordering pool, and the pool is
# far smaller than it looks: measured 2026-07-26 over 8 scans, a VOD yields
# ~856 candidates, ~517 survive the evidence gates, and only **~56 survive the
# adaptive floor** -- that ~56 is what the ranker actually reorders. Primary
# labels 15 of it; Second Look labels the rest.
#
#     tier size   pool labelled   combined keep rate
#        15         ~29 / 56            30.9%
#         5         ~20 / 56            34.0%
#
# So the 3-point keep-rate gain costs ~9 off-policy labels per scan, and those
# labels are exactly what the learned ranker lacks: it is trained on the top
# half of the pool and asked to reorder all of it, which is why a sharper model
# (v8 embeddings) promoted badly from the unlabelled bottom half and cost 0.10
# recall. Held at 15 while the ranker is being trained -- see
# [[gie-ranker-distribution-mismatch]]. Drop back to 5 once the ranker ships
# and the tier is ordered by something better than noise.
MORE_CANDIDATE_LIMIT = 15
REVIEW_TIER_SECOND_LOOK = "second_look"

# Limit how much an overflow candidate overlaps an already selected clip. Group identity
# alone cannot detect duplicate windows assigned different group IDs.
SECOND_LOOK_PRIMARY_COVERAGE = 0.60

# Floor veto for the REVIEW lane. Cards below every one of these thresholds are
# the arousal floor -- no vocal reaction, no signal agreement, no hook -- and
# the creator rejects them at ~13:1.
#
# This is deliberately NOT a classifier. Plan 1b.2 retrained the Second-look
# rejector twice and it removed 1.7%, then 0.0%, of Passes at its zero-Keep
# operating point: scoring all ~440 cards drags the model through the ~80% that
# is genuinely ambiguous. A narrow threshold that stays silent outside the floor
# does better precisely because it refuses to have an opinion about the middle.
#
# MEASURED 2026-08-02, source-held-out over 443 reviewed cards / 36 VODs
# (threshold picked on 35 VODs, applied blind to the 36th):
#
#     budget   fires    keeps lost      passes cut   ratio
#      2%      69       5/92  (5.4%)    64  (18.2%)  13:1
#      3%     101       7/92  (7.6%)    94  (26.8%)  13:1
#      5%     112      10/92 (10.9%)   102  (29.1%)  10:1
#
# HONEST LIMIT: at a strict zero-Keep operating point this removes only 6.3% of
# Passes and still leaks 3 Keeps held-out, so it FAILS 1b.2's 20%-at-zero-loss
# gate exactly as the learned rejector did. There is no free filter here; 26.8%
# costs ~1 Keep in 13. The 3% budget is that trade, made deliberately.
#
# Vetoed cards are NOT discarded -- see ``partition_second_look``. The model is
# already confident about them (median held-out ranker score -0.86 vs -0.17 for
# survivors; only 24% sit near the decision boundary), so they are the least
# informative labels in the tier. Boundary-band sampling (plan 2.2,
# scripts/build_boundary_batch.py) is the lane that collects the informative
# ones, and it is gated out of evaluation on purpose.
SECOND_LOOK_FLOOR_VETO = {
    "hook_score": 0.2162,
    "voice": 0.1375,
    "signal_agreement": 0.4000,
}


def more_candidate_clip_id(
    job_id: str,
    start: float,
    end: float,
    peak: Optional[float] = None,
) -> str:
    """Stable id for a ceiling-cut candidate across scan persist and reveal."""
    start_f = float(start or 0.0)
    end_f = float(end or 0.0)
    peak_f = float(peak if peak is not None else start_f)
    seed = f"recall:{job_id}:{start_f:.3f}:{end_f:.3f}:{peak_f:.3f}"
    return f"more_{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:20]}"


def _grounded_consensus_overflow(row: Dict[str, Any]) -> bool:
    """Whether an overflow-cut trace row carries grounded dual-judge consensus.

    Uses the selector's own predicate over the hydrated row so this tier and
    the temporal-blackout rescue can never disagree about what "consensus"
    means. Trace rows persist the abbreviated ``visual_*`` field names;
    hydration restores the live ``visual_semantic_*`` names the predicate
    reads.
    """
    hydrated = hydrate_candidate_trace_row(row)
    return _selection._grounded_judge_consensus(
        hydrated, _selection.DEFAULT_TUNING
    )


def _sort_key(candidate: Dict[str, Any]):
    """Order the tier by the learned ranker when it scored this candidate.

    MEASURED 2026-07-26 against blind creator labels: inside the ceiling pool
    the ranker's picks keep at 37.5% vs 16.7% for a random draw from that same
    pool, while the signal order this used to follow carries almost no
    signal there (rank 1 keeps at 11%, rank 10 at 31%, rank 3 at 39%). The
    ranker cannot out-order the Primary deck -- leave-one-VOD-out put every
    ordering policy at or below signals -- but the tail is exactly where it
    wins, so the tail is exactly where it is used.

    Rows without a ranker score (no model installed, or traces frozen before
    one existed) keep the historical selection_rank order, so this degrades
    cleanly rather than shuffling old decks.
    """
    # Primary eligibility is always base-owned. Inside Second Look, a validated
    # personal adapter may safely order already-eligible tail candidates.
    raw_ranker = candidate.get("personal_ranker_score")
    if raw_ranker is None:
        raw_ranker = candidate.get("ranker_score")
    try:
        ranker_score = float(raw_ranker) if raw_ranker is not None else None
    except (TypeError, ValueError):
        ranker_score = None

    rank = candidate.get("selection_rank")
    try:
        numeric_rank = int(rank)
    except (TypeError, ValueError):
        numeric_rank = 1_000_000
    try:
        score = float(candidate.get("selection_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    start = float(candidate.get("start", 0.0) or 0.0)

    # Scored rows sort ahead of unscored ones, then by ranker score descending.
    if ranker_score is not None:
        return (0, -ranker_score, numeric_rank, start)
    return (1, float(numeric_rank), -score, start)


def ceiling_candidates_from_trace(
    trace: Optional[Iterable[Any]],
    *,
    job_id: str,
    limit: Optional[int] = MORE_CANDIDATE_LIMIT,
    include_shadow_audit: bool = False,
) -> List[Dict[str, Any]]:
    """Normalize and rank the Second-look pool from a selection trace.

    The pool is the ``clip_ceiling`` band plus any ``editorial_overflow_cutoff``
    row that carries grounded dual-judge consensus: both judges said post with
    grounded evidence, so the row's depth in the arousal ranking must not make
    it invisible to the creator. Plain overflow and ``predictable_pass`` rows
    stay excluded.

    ``limit=None`` returns every distinct creator-facing pool row (status
    totals). Shadow-personalization disagreements stay frozen in the source
    artifact but are not creator review cards. Offline evaluation tooling may
    opt into them explicitly with ``include_shadow_audit=True``; no product
    path should set that flag.
    """
    if not trace:
        return []
    rows = list(trace)
    if limit is None:
        slice_limit: Optional[int] = None
    else:
        slice_limit = max(1, int(limit))
    selected_groups = {
        str(raw.get("moment_group_id"))
        for raw in rows
        if (
            isinstance(raw, dict)
            and str(raw.get("disposition") or "").startswith("selected")
            and raw.get("moment_group_id")
        )
    }
    # Windows of the clips that DID make the deck, so a ceiling candidate
    # covering the same seconds can be dropped even when its group id differs.
    selected_windows: List[tuple] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        if not str(raw.get("disposition") or "").startswith("selected"):
            continue
        try:
            w_start = float(raw.get("start", 0.0) or 0.0)
            w_end = float(raw.get("end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if w_end > w_start:
            selected_windows.append((w_start, w_end))
    candidates: List[Dict[str, Any]] = []
    audit_candidates: List[Dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        disposition = raw.get("disposition")
        shadow_audit = bool(raw.get("shadow_primary_audit"))
        if shadow_audit and not include_shadow_audit:
            continue
        if shadow_audit:
            consensus_overflow = False
        elif disposition == "clip_ceiling":
            consensus_overflow = False
        elif disposition == "editorial_overflow_cutoff":
            if not _grounded_consensus_overflow(raw):
                continue
            consensus_overflow = True
        else:
            continue
        try:
            start = float(raw.get("start", 0.0) or 0.0)
            end = float(raw.get("end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        candidate = dict(raw)
        candidate["start"] = start
        candidate["end"] = end
        peak = candidate.get("peak_timestamp", start)
        try:
            peak_f = float(peak if peak is not None else start)
        except (TypeError, ValueError):
            peak_f = start
        candidate["peak_timestamp"] = peak_f
        candidate["id"] = more_candidate_clip_id(job_id, start, end, peak_f)
        if shadow_audit:
            # A challenger audit row is an intentionally counterfactual review
            # card. It bypasses the ordinary ceiling cap and selected-coverage
            # cleanup below: otherwise exactly the novel Primary entrant we
            # need a Keep/Pass label for could disappear before review.
            candidate["second_look_shadow_primary_audit"] = True
        if consensus_overflow:
            candidate["second_look_consensus_overflow"] = True
        (audit_candidates if shadow_audit else candidates).append(candidate)

    # Consensus-overflow rows compete for the SAME limit slots but sort after
    # the entire ceiling band: the tier's measured keep rates and its
    # no-backfill freeze are defined over that band, so widening admission
    # must not reorder or displace it. Within the overflow group the ordinary
    # ranker/rank order applies.
    candidates.sort(
        key=lambda c: (
            1 if c.get("second_look_consensus_overflow") else 0,
            *_sort_key(c),
        )
    )
    # Freeze the same overflow band the legacy deck would have shown. Hygiene
    # may remove fat from that band, but must not backfill it with deeper rows.
    if slice_limit is not None:
        candidates = candidates[:slice_limit]
    # Offline-only audit rows are a union, not a substitute for the measured
    # Second Look band. When explicitly requested they remain beyond its
    # ordinary cap so evaluation sees every policy disagreement. The product
    # default above never reaches this branch with audit rows.
    if include_shadow_audit:
        audit_candidates.sort(key=_sort_key)
        candidates.extend(audit_candidates)
    unique: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        group_id = candidate.get("moment_group_id")
        is_shadow_audit = bool(candidate.get("second_look_shadow_primary_audit"))
        if not is_shadow_audit and group_id and str(group_id) in selected_groups:
            continue
        if not is_shadow_audit and _covered_by_selected(candidate, selected_windows):
            continue
        # Ordinary Second Look retains its story-level collapse. Challenger
        # audit is different: one story may yield multiple *exact* selected
        # windows at different shadow weights, and each must be reviewable.
        # ``id`` is derived from the exact start/end/peak triple.
        identity = (
            f"audit:{candidate['id']}"
            if is_shadow_audit
            else str(candidate.get("moment_group_id") or candidate["id"])
        )
        existing = unique.get(identity)
        if is_shadow_audit and existing is not None:
            # The same exact trace instance can enter at more than one shadow
            # weight (or arrive twice through presentation paths). Retain one
            # review card, but preserve the complete counterfactual cohort.
            merged_weights = {
                float(weight)
                for row in (existing, candidate)
                for weight in row.get("shadow_primary_weights", [])
            }
            existing["shadow_primary_weights"] = sorted(merged_weights)
        elif is_shadow_audit or identity not in unique:
            unique[identity] = candidate
    ordered = list(unique.values())
    return ordered


def _covered_by_selected(
    candidate: Dict[str, Any],
    selected_windows: Iterable[tuple],
    threshold: float = SECOND_LOOK_PRIMARY_COVERAGE,
) -> bool:
    """True when a selected clip already contains most of this candidate.

    Coverage is measured against the CANDIDATE's own duration, not the overlap
    of the pair: a 15s card sitting inside a 50s selected clip is a duplicate of
    it, while a 50s card that merely contains a short selected clip is not.
    """
    start = float(candidate.get("start", 0.0) or 0.0)
    end = float(candidate.get("end", 0.0) or 0.0)
    duration = end - start
    if duration <= 0:
        return False
    for w_start, w_end in selected_windows:
        overlap = min(end, w_end) - max(start, w_start)
        if overlap > 0 and (overlap / duration) >= threshold:
            return True
    return False


def second_look_floor_reason(candidate: Dict[str, Any]) -> Optional[str]:
    """Name the floor threshold this candidate falls under, or None.

    Reads the candidate's ``features`` dict, falling back to top-level keys so
    both trace rows and hydrated candidates work. A feature that is missing
    never fires the veto -- an absent measurement is not evidence of a floor.
    """
    # A consensus-overflow row is in the tier BECAUSE both judges endorsed it
    # with grounded evidence despite weak arousal; the arousal floor is the
    # wrong instrument to re-litigate that admission.
    if (
        candidate.get("second_look_consensus_overflow")
        or candidate.get("second_look_shadow_primary_audit")
    ):
        return None
    features = candidate.get("features")
    source = features if isinstance(features, dict) else candidate
    for name, threshold in SECOND_LOOK_FLOOR_VETO.items():
        value = source.get(name)
        if isinstance(value, (int, float)) and float(value) <= threshold:
            return name
    return None


def partition_second_look(
    candidates: Iterable[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split ceiling candidates into (review, deferred).

    ``review`` is what Theater's More moments should show. ``deferred`` is the
    arousal floor: still real candidates with complete features, kept addressable
    so the training lane can use them, just not worth the creator's attention in
    a review pass. Each deferred row is tagged with ``second_look_floor_reason``
    so a later batch can say WHY it was held back.
    """
    review: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    for candidate in candidates:
        reason = second_look_floor_reason(candidate)
        if reason:
            tagged = dict(candidate)
            tagged["second_look_floor_reason"] = reason
            deferred.append(tagged)
        else:
            review.append(candidate)
    return review, deferred


def second_look_deck_score(index: int) -> float:
    """Descending in-tier score so Second-look cards sort under the review deck."""
    return max(0.01, 0.49 - int(index) * 0.01)
