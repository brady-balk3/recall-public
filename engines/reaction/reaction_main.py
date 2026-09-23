# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Reaction Engine entry point (plan §5.5).

Replaces the Story->Clip selection spine. Given the per-second signal tracks it:
  1. fuses a per-VOD-normalized R(t) curve,
  2. detects arcs/peaks,
  3. snaps boundaries to reaction onset + audio gaps,
  4. selects globally (ceiling, relative bar, NMS),
  5. emits ReactionClip candidates with full provenance for explainability and
     (Phase 4) learned-ranker training.
"""

from dataclasses import dataclass, field, asdict, replace as dataclass_replace
import collections
import copy
import hashlib
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np

from core.models.reaction import ReactionCurve  # noqa: E402
from engines.event.event_detector import classify_ocr_trigger  # noqa: E402
import engines.caption.sentence_index as sentence_index_mod
import engines.chat.chat_features as chat_features_mod
import engines.reaction.fusion as fusion
import engines.reaction.boundaries as boundaries_mod
import engines.reaction.editorial_hygiene as editorial_hygiene_mod
import engines.reaction.housekeeping as housekeeping_mod
import engines.reaction.pass_rejector as pass_rejector_mod
import engines.reaction.selection as selection_mod
import engines.reaction.features as features_mod
import engines.reaction.scene as scene_mod
import engines.reaction.segments as segments_mod
import engines.reaction.content_shape as content_shape_mod
import engines.vision.layout_map as layout_map_mod


# These are deliberately measurement lanes, not production policy.  A shadow
# challenger never changes Primary; we only surface the moments it *would* add
# so a creator can label the counterfactual before it can earn any authority.
SHADOW_AUDIT_WEIGHTS = (0.10, 0.20, 0.25)
SHADOW_AUDIT_PRIMARY_DISPOSITIONS = {
    "selected_primary", "selected_protected", "selected_temporal_rescue",
}
SELECTION_REPLAY_CONTRACT = "pre_presentation_v1"


def _selector_source_sha256() -> Optional[str]:
    """Fingerprint the exact selector implementation when a file is readable."""
    path = getattr(selection_mod, "__file__", None)
    if not path or not os.path.isfile(path):
        return None
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _materially_overlaps_base_primary(candidate: dict, selected: List[dict]) -> bool:
    """Whether a Base Primary decision already supervises this exact window.

    This is intentionally the same shared/min-span >= 0.5 rule the
    prospective evaluator uses to associate a candidate with a Keep/Pass.
    Moment groups alone are insufficient: adjacent windows in one group can
    be distinct counterfactual clips and need their own labels.
    """
    try:
        start = float(candidate.get("start", 0.0) or 0.0)
        end = float(candidate.get("end", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    candidate_span = max(0.001, end - start)
    if end <= start:
        return False
    for base in selected:
        try:
            base_start = float(base.get("start", 0.0) or 0.0)
            base_end = float(base.get("end", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            continue
        base_span = max(0.001, base_end - base_start)
        shared = max(0.0, min(end, base_end) - max(start, base_start))
        if shared / min(candidate_span, base_span) >= 0.5:
            return True
    return False


def _annotate_shadow_primary_audit(
    candidates: List[dict],
    selected: List[dict],
    score: np.ndarray,
    *,
    ranker_diagnostics: Optional[dict],
    max_clips: int,
    k: float,
    tuning,
    ranker_policy: str,
    ranker_blend_weight: float,
    ranker_publishability_floor: float,
    ranker_use_personalized_adaptive_floor: bool,
    second_look_rejector=None,
) -> None:
    """Mark counterfactual challenger-only Primary entrants for Second Look.

    Copies are intentional: ``select_clips`` annotates/reorders its inputs, and
    this audit lane must be unable to mutate the actual Base Primary decision.
    The marks are trace-only and are consumed by ``core.more_candidates``.
    """
    diagnostics = ranker_diagnostics if isinstance(ranker_diagnostics, dict) else {}
    metadata = diagnostics.get("challenger_metadata")
    challenger_sha = diagnostics.get("challenger_model_sha256")
    manifest_sha = diagnostics.get("challenger_manifest_sha256")
    # Resolver only emits this complete, mutually-bound set after loading a
    # compatible challenger whose manifest passed validation. Treat anything
    # partial as absent: a score without provenance is not review evidence.
    provenance_valid = bool(
        diagnostics.get("challenger_model_loaded")
        and isinstance(challenger_sha, str)
        and re.fullmatch(r"[a-f0-9]{64}", challenger_sha) is not None
        and isinstance(manifest_sha, str)
        and re.fullmatch(r"[a-f0-9]{64}", manifest_sha) is not None
        and isinstance(metadata, dict)
        and metadata.get("challenger_model_sha256") == challenger_sha
        and metadata.get("manifest_sha256") == manifest_sha
        and metadata.get("profile") == diagnostics.get("personalization_profile")
        and metadata.get("base_model_sha256") == diagnostics.get("base_model_sha256")
        and metadata.get("feature_version") is not None
        and metadata.get("model_label_count") is not None
        and isinstance(metadata.get("training_job_ids"), list)
        and isinstance(metadata.get("training_source_keys"), list)
    )
    if not provenance_valid or not candidates:
        return
    # Percentiles are deck-relative. If even one candidate lacks a finite
    # challenger delta, replaying only the scored subset would create a
    # different percentile universe from the real prospective lane. Fail inert
    # rather than pretending a partial shadow score is evidence.
    try:
        challenger_deltas_complete = all(
            candidate.get("personal_challenger_delta") is not None
            and np.isfinite(float(candidate["personal_challenger_delta"]))
            for candidate in candidates
        )
    except (TypeError, ValueError):
        challenger_deltas_complete = False
    if not challenger_deltas_complete:
        return

    # ``select_clips`` can refine boundaries. Carry a private source index on
    # each deep copy so the trace row marked for audit is the exact preselect
    # candidate instance, not whichever sibling later shares its group/window.
    audit_weights: Dict[int, List[float]] = {}
    for weight in SHADOW_AUDIT_WEIGHTS:
        shadow = copy.deepcopy(candidates)
        for source_index, candidate in enumerate(shadow):
            candidate["_shadow_audit_source_index"] = source_index
            delta = candidate.get("personal_challenger_delta")
            if delta is not None:
                candidate["personal_preference_delta"] = float(delta)
        shadow_selected = selection_mod.select_clips(
            shadow, score, max_clips=max_clips, k=k, tuning=tuning,
            ranker_policy=ranker_policy,
            ranker_blend_weight=ranker_blend_weight,
            personal_ranker_blend_weight=weight,
            ranker_publishability_floor=ranker_publishability_floor,
            ranker_use_personalized_adaptive_floor=(
                ranker_use_personalized_adaptive_floor
            ),
            second_look_rejector=second_look_rejector,
        )
        for candidate in shadow_selected:
            if candidate.get("selection_disposition") not in SHADOW_AUDIT_PRIMARY_DISPOSITIONS:
                continue
            if _materially_overlaps_base_primary(candidate, selected):
                continue
            source_index = candidate.get("_shadow_audit_source_index")
            if isinstance(source_index, int) and 0 <= source_index < len(candidates):
                audit_weights.setdefault(source_index, []).append(weight)

    for source_index, candidate in enumerate(candidates):
        weights = audit_weights.get(source_index)
        if weights:
            candidate["shadow_primary_audit"] = True
            candidate["shadow_primary_weights"] = weights


@dataclass
class ReactionClip:
    start: float
    end: float
    peak_timestamp: float
    reason: str
    modality_breakdown: Dict[str, float] = field(default_factory=dict)
    reaction_auc: float = 0.0
    score: float = 0.0
    hook_score: float = 0.0
    game_label: Optional[str] = None
    game_evidence: float = 0.0
    # Non-gameplay scene code (lobby / intermission) when this clip fired on
    # social hype with no gameplay behind it; None for real gameplay clips.
    scene_label: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)  # for learned-ranker label capture
    # Semantic judge output (Package C) when the judge ran on this candidate;
    # all None otherwise. The pipeline prefers semantic_title over the
    # transcript-quote fallback when present.
    semantic_title: Optional[str] = None
    semantic_hook_line: Optional[str] = None
    semantic_moment_type: Optional[str] = None
    semantic_verdict: Optional[str] = None
    creator_protected: bool = False
    recall_marker_ids: List[str] = field(default_factory=list)
    recall_marker_time: Optional[float] = None

    def to_dict(self):
        return asdict(self)


WIN_TARGET_LEN = 22.0  # target length of a merged win clip (lead-up + banner + celebration)
WIN_DEFAULT_LEAD = 14.0
WIN_MIN_LEAD = 10.0
WIN_MAX_LOOKBACK = 24.0
STRONG_LOBBY_TAIL_PAD = 2.0

EVENT_ANCHOR_LABELS = {"elimination", "knock"}
EVENT_LOOKBACK = 12.0
EVENT_LOOKAHEAD = 4.0

# Keep the causal judge gap separate from selection-time deduplication. Delayed aftermath
# can follow an elimination beyond the selection gap; the larger judge gap includes it
# while max_span bounds the group.
CAUSAL_JUDGE_MAX_GAP_SEC = 14.0

# Combat-encounter anchors (game-audio track): a sustained run of combat SFX /
# intense music marks the GAME entering a high-stakes state (boss fight, chase)
# even when a focused streamer goes quiet — the class of moment OCR-less games
# (horror/story titles) structurally missed. The resolution reaction (relief
# yell, death groan) trails the fight's audio, so the anchor searches a short
# window past the run's end.
ENCOUNTER_MIN_RUN_SEC = 12.0   # sustained intensity, not a one-off SFX hit
ENCOUNTER_GAP_SEC = 4.0        # bridge brief lulls inside one fight
ENCOUNTER_LOOKAHEAD = 8.0      # resolution reaction trails the last SFX
ENCOUNTER_MAX_INJECT = 6       # cap synthetic candidates on action-heavy VODs
ENCOUNTER_MIN_LEVEL = 0.10     # absolute floor on the per-VOD elevation bar

# Specific visible-chat reactions are direct human confirmation that a moment
# landed. Generic laughter alone is intentionally weak; several corroborating
# terms or a high-specificity phrase are required to reach selection's anchor.
VISIBLE_CROWD_TERMS = {
    "rip headphone": 1.00,
    "headphone wearers": 0.95,
    "i jumped": 1.00,
    "my ears": 0.95,
    "why you so loud": 0.90,
    "that sound": 0.80,
    "clip it": 1.00,
    "thats funny": 0.75,
    "so funny": 0.65,
    "screaming": 0.70,
    "no way": 0.35,
    "holy": 0.35,
    "wtf": 0.45,
    "lmao": 0.35,
    "lol": 0.20,
    # Win/achievement congratulations, HIGH-SPECIFICITY only. Observed
    # verbatim in a founder VOD's on-screen chat during VICTORY ROYALE
    # banners OCR itself never read ("YOU DID IT!"). Generic congratulations
    # ("gg", "lets go", "slay") were tried and REVERTED: tac-shooter chat
    # spams them at every round end and valorant_grind's golden precision
    # regressed 0.200 -> 0.133.
    "you did it": 0.75,
}


def _visible_crowd_reaction(texts: List[str]):
    normalized = " " + " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(text).lower()) for text in texts
    ) + " "
    found = [
        (term, weight) for term, weight in VISIBLE_CROWD_TERMS.items()
        if f" {term} " in normalized
    ]
    if not found:
        return 0.0, []
    score = min(1.0, sum(weight for _term, weight in found) / 2.2)
    return round(score, 4), [term for term, _weight in found]

# Viewer clip commands ("!clip", "clip it") are RETROACTIVE crowd anchors: the
# moment happened up to ~30s before the first command (reaction + typing lag;
# Twitch's own !clip captures the trailing 30s for the same reason). So the
# search window reaches back from the burst, never forward.
CLIP_CMD_LOOKBACK = 35.0
CLIP_CMD_MIN_DELAY = 2.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _thresholds_from_settings(settings: dict):
    sensitivity = float(settings.get("sensitivity", 0.7))
    aggressiveness = float(settings.get("aggressiveness", 0.6))
    # More sensitive / aggressive -> lower bar -> more candidates.
    k = _clamp(1.5 - sensitivity * 0.7 - aggressiveness * 0.2, 0.6, 1.6)
    onset_k = _clamp(k * 0.4, 0.2, 0.8)
    return k, onset_k


def _modality_breakdown(curve: ReactionCurve, lo: float, hi: float) -> Dict[str, float]:
    frames = [f for f in curve.frames if lo <= f.timestamp <= hi]
    if not frames:
        return {
            "face": 0.0, "face_max": 0.0, "voice": 0.0, "speech": 0.0, "burst": 0.0,
            "loud_onset": 0.0, "startle": 0.0,
            "chat": 0.0, "motion": 0.0,
            "game_audio": 0.0, "game": 0.0,
        }
    face_vals = [f.face_arousal for f in frames if f.face_present]
    return {
        "face": round(float(np.mean(face_vals)), 4) if face_vals else 0.0,
        # Peak face in-window; calibrated rescue prevents a Surprise spike from
        # being diluted by surrounding neutral frames.
        "face_max": round(float(max(face_vals)), 4) if face_vals else 0.0,
        "voice": round(float(np.mean([f.voice_arousal for f in frames])), 4),
        "speech": round(float(np.mean([f.speech_hype for f in frames])), 4),
        "burst": round(float(max((getattr(f, "burst", 0.0) for f in frames), default=0.0)), 4),
        "loud_onset": round(float(max((getattr(f, "loud_onset", 0.0) for f in frames), default=0.0)), 4),
        "startle": round(float(max((
            getattr(f, "startle", 0.0) for f in frames
        ), default=0.0)), 4),
        "chat": round(float(np.mean([getattr(f, "chat", 0.0) for f in frames])), 4),
        "motion": round(float(np.mean([getattr(f, "motion", 0.0) for f in frames])), 4),
        "game_audio": round(float(max((getattr(f, "game_intensity", 0.0) for f in frames), default=0.0)), 4),
        "game": round(float(max(f.game_evidence for f in frames)), 4),
    }


def _attach_face_spike_context(
    curve: ReactionCurve,
    candidates: List[dict],
    *,
    absolute_min: float = 0.40,
    percentile: float = 99.0,
    local_radius_sec: float = 30.0,
) -> None:
    """Attach cross-creator-safe face-spike evidence to every candidate.

    HSEmotion arousal is raw model output: 0.40 is rare on one creator but can
    sit below another creator's p95. Selection therefore needs both an absolute
    floor and this VOD's own tail, plus sustained support and a local-baseline
    delta. Values live in ``modality_breakdown`` so replay sees the exact live
    evidence.
    """
    present = sorted(
        (frame for frame in curve.frames if frame.face_present),
        key=lambda frame: float(frame.timestamp),
    )
    timestamps = np.asarray(
        [float(frame.timestamp) for frame in present],
        dtype=np.float64,
    )
    values = np.asarray(
        [float(frame.face_arousal) for frame in present],
        dtype=np.float64,
    )
    vod_p99 = (
        float(np.percentile(values, percentile))
        if values.size else 0.0
    )
    vod_median = float(np.median(values)) if values.size else 0.0
    threshold = max(float(absolute_min), vod_p99)

    for candidate in candidates:
        start = float(candidate.get("start", 0.0) or 0.0)
        end = float(candidate.get("end", start) or start)
        window_lo = int(np.searchsorted(timestamps, start, side="left"))
        window_hi = int(np.searchsorted(timestamps, end, side="right"))
        window_ts = timestamps[window_lo:window_hi]
        window_values = values[window_lo:window_hi]
        if window_values.size:
            peak_offset = int(np.argmax(window_values))
            peak = float(window_values[peak_offset])
            peak_timestamp = float(window_ts[peak_offset])
        else:
            peak = 0.0
            peak_timestamp = 0.0

        local_lo = int(np.searchsorted(
            timestamps, start - local_radius_sec, side="left",
        ))
        local_hi = int(np.searchsorted(
            timestamps, end + local_radius_sec, side="right",
        ))
        local_context = np.concatenate((
            values[local_lo:window_lo],
            values[window_hi:local_hi],
        ))
        local_median = (
            float(np.median(local_context))
            if local_context.size else vod_median
        )

        max_run = 0
        run = 0
        previous_ts = None
        hot_timestamps = window_ts[window_values >= threshold]
        for timestamp in hot_timestamps:
            timestamp = float(timestamp)
            if previous_ts is not None and timestamp - previous_ts <= 1.5:
                run += 1
            else:
                run = 1
            max_run = max(max_run, run)
            previous_ts = timestamp

        breakdown = dict(candidate.get("modality_breakdown") or {})
        breakdown.update({
            "face_max": round(peak, 4),
            "face_vod_p99": round(vod_p99, 4),
            "face_local_median": round(local_median, 4),
            "face_peak_delta": round(max(0.0, peak - local_median), 4),
            "face_spike_threshold": round(threshold, 4),
            "face_spike_frames": int(hot_timestamps.size),
            "face_spike_max_run": int(max_run),
            "face_spike_peak_timestamp": round(peak_timestamp, 3),
        })
        candidate["modality_breakdown"] = breakdown


def _candidate_context(curve, timeline, score, lo, hi, anchor_time=None, peak_time=None, active_channels=None):
    """Features that make ranking more like an editor than an event detector.

    ``active_channels`` is the set of reaction modalities this VOD actually has
    tracks for (voice is always present; face/speech/burst/chat depend on what
    ran). ``signal_agreement`` is normalized over THAT set, not a hardcoded
    "4 modalities" -- otherwise a game with no facecam, no chat, or (critically)
    no OCR HUD is structurally capped below any multi-signal bar it could ever
    clear, no matter how strong its reaction. Game evidence is deliberately NOT
    one of the agreement channels: it already gets a separate bypass in
    selection.py (``has_game_context``), so counting it here would double-dip
    for HUD games while permanently starving non-HUD games of a denominator
    slot they can never fill.
    """
    active_channels = active_channels or {"voice"}
    mask = (timeline >= lo) & (timeline <= hi)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return {
            "lead_reaction": 0.0,
            "settle_reaction": 0.0,
            "dead_air_ratio": 1.0,
            "signal_agreement": 0.0,
            "modality_count": 0,
            "hook_reaction_energy": 0.0,
            "hook_burst": 0.0,
            "hook_chat": 0.0,
        }

    pivot = float(anchor_time if anchor_time is not None else (peak_time if peak_time is not None else timeline[idxs[0]]))
    pre_mask = mask & (timeline < pivot)
    post_mask = mask & (timeline >= pivot)
    pre_vals = score[pre_mask]
    post_vals = score[post_mask]

    quiet = 0
    contributing = set()
    for idx in idxs:
        frame = curve.frames[idx]
        if score[idx] < 0.15 and frame.game_evidence <= 0:
            quiet += 1
        if "face" in active_channels and frame.face_present and frame.face_arousal >= 0.12:
            contributing.add("face")
        if "voice" in active_channels and frame.voice_arousal >= 0.25:
            contributing.add("voice")
        if "speech" in active_channels and frame.speech_hype >= 0.20:
            contributing.add("speech")
        # Laughter/scream burst -- the primary signal for reaction-only clips
        # (horror scares, funny moments) that have no OCR HUD to lean on.
        if "burst" in active_channels and frame.burst >= 0.12:
            contributing.add("burst")
        if "chat" in active_channels and frame.chat >= 2.5:
            contributing.add("chat")
        # Gameplay motion -- silent fights / multi-kill chaos without a yell.
        if "motion" in active_channels and getattr(frame, "motion", 0.0) >= 0.18:
            contributing.add("motion")

    denom = float(len(active_channels)) if active_channels else 1.0
    hook_mask = mask & (timeline <= lo + 5.0)
    hook_idxs = np.where(hook_mask)[0]
    hook_energy = float(np.mean(np.maximum(score[hook_mask], 0.0))) if hook_idxs.size else 0.0
    hook_burst = max((float(curve.frames[i].burst) for i in hook_idxs), default=0.0)
    hook_chat = max((float(curve.frames[i].chat) for i in hook_idxs), default=0.0)
    return {
        "lead_reaction": round(float(np.mean(pre_vals)) if pre_vals.size else 0.0, 4),
        "settle_reaction": round(float(np.mean(post_vals)) if post_vals.size else 0.0, 4),
        "dead_air_ratio": round(float(quiet / idxs.size), 4),
        "signal_agreement": round(float(len(contributing) / denom), 4),
        "modality_count": int(len(contributing)),
        "hook_reaction_energy": round(hook_energy, 4),
        "hook_burst": round(hook_burst, 4),
        "hook_chat": round(hook_chat, 4),
    }


def _build_reason(game_label: Optional[str], breakdown: Dict[str, float],
                  crowd_clip: int = 0) -> str:
    label_text = {
        "terminal_win": "Match win",
        "elimination": "Elimination",
        "knock": "Knockdown",
        "rank_progress": "Rank progress",
        "generic_highlight": "Highlight",
    }.get(game_label or "", "")
    # Crowd clip commands are the most honest label we have for a non-game
    # moment: viewers literally asked for it. Game labels still lead.
    if not label_text and crowd_clip >= 2:
        label_text = "Chat called for this clip"

    if breakdown.get("startle", 0.0) >= 0.60:
        if label_text:
            return f"{label_text} + sudden startle"
        return "Sudden startle reaction"

    # Modalities live on different raw scales (face arousal ~0..0.3, voice/speech
    # ~0..1), so normalize by a nominal "high" value before picking the dominant
    # one -- otherwise voice always wins the label.
    nominal = {"voice": 0.5, "face": 0.25, "speech": 0.45, "burst": 0.5}
    scaled = {k: breakdown.get(k, 0.0) / s for k, s in nominal.items()}
    dominant = max(scaled, key=scaled.get)
    dominant_text = {
        "face": "big facecam reaction",
        "voice": "strong voice reaction",
        "speech": "hype callout",
        "burst": "laughter/scream burst",
    }[dominant]

    # Only claim a reaction when that modality is actually elevated. Game
    # audio is the fallback explanation, never the headline: on shooter VODs
    # gunfire is elevated in almost every clip, so it may only name a clip
    # when no human modality earned the label (silent-fight encounter clips).
    if scaled[dominant] < 0.6:
        if breakdown.get("game_audio", 0.0) >= 0.25:
            if label_text:
                return f"{label_text} + intense game action"
            return "Intense game action"
        return label_text or "Reaction"
    if label_text:
        return f"{label_text} + {dominant_text}"
    return dominant_text.capitalize()


def _win_lead_start(curve, timeline, score, anchor_time, clip_max):
    """Choose a terminal-win start that favors the play before the banner."""
    default_start = max(0.0, anchor_time - min(WIN_DEFAULT_LEAD, clip_max - boundaries_mod.TAIL_MIN))
    lookback_start = max(0.0, anchor_time - min(WIN_MAX_LOOKBACK, clip_max - boundaries_mod.TAIL_MIN))
    lookback_end = max(0.0, anchor_time - 2.0)
    mask = (timeline >= lookback_start) & (timeline <= lookback_end)
    if not mask.any():
        return default_start

    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) * 1.4826
    activity_floor = max(0.45, median + 0.35 * mad)
    idxs = np.where(mask)[0]
    active = []
    for idx in idxs:
        frame = curve.frames[idx]
        if score[idx] >= activity_floor or frame.game_label in ("elimination", "knock"):
            active.append(idx)
    if not active:
        return default_start

    groups = []
    current = [active[0]]
    for idx in active[1:]:
        if timeline[idx] - timeline[current[-1]] <= 3.0:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    # Prefer the gameplay/reaction run nearest to the victory banner.
    group = min(groups, key=lambda g: anchor_time - timeline[g[-1]])
    lead_start = max(0.0, float(timeline[group[0]]) - 3.0)
    return min(lead_start, max(0.0, anchor_time - WIN_MIN_LEAD))


def _inject_protected_wins(curve, candidates, timeline, score, rms, clip_min, clip_max, active_channels=None, cuts=None):
    """Propose a candidate for every terminal_win run not already covered.

    The historical function name is retained for compatibility; selection no
    longer protects OCR wins from the human-reaction evidence gates.
    """
    win_frames = [f for f in curve.frames if f.game_label == "terminal_win"]
    if not win_frames:
        return

    # Cluster consecutive win frames (gap > 5s starts a new win).
    runs, current = [], [win_frames[0]]
    for f in win_frames[1:]:
        if f.timestamp - current[-1].timestamp <= 5.0:
            current.append(f)
        else:
            runs.append(current)
            current = [f]
    runs.append(current)

    for run in runs:
        anchor = max(run, key=lambda f: f.game_evidence)
        if any(c["start"] <= anchor.timestamp <= c["end"] for c in candidates):
            continue
        a_idx = int(np.argmin(np.abs(timeline - anchor.timestamp)))
        synthetic = boundaries_mod.Arc(
            start_idx=a_idx, end_idx=a_idx, peak_idx=a_idx,
            peak_value=float(score[a_idx]),
        )
        voice_arousal = np.array([f.voice_arousal for f in curve.frames], dtype=np.float64)
        bounds = boundaries_mod.snap_boundaries(
            synthetic, timeline, rms,
            anchor_time=anchor.timestamp, clip_min=clip_min, clip_max=clip_max,
            cuts=cuts, voice_arousal=voice_arousal,
        )
        lo, hi = bounds["start"], bounds["end"]
        in_span = (timeline >= lo) & (timeline <= hi)
        dens = getattr(curve, "event_density", None)
        dens_val = 1.0
        if dens is not None and in_span.any():
            dens_arr = np.asarray(dens, dtype=np.float64)
            if dens_arr.size == timeline.size:
                dens_val = float(np.max(dens_arr[in_span]))
        candidates.append({
            **bounds,
            "reaction_auc": round(float(np.sum(score[in_span])), 4),
            "modality_breakdown": _modality_breakdown(curve, lo, hi),
            "game_evidence": round(anchor.game_evidence, 4),
            "game_label": "terminal_win",
            "game_anchor_time": anchor.timestamp,
            "event_density": round(dens_val, 4),
            **_candidate_context(
                curve, timeline, score, lo, hi,
                anchor_time=anchor.timestamp, peak_time=bounds["peak_timestamp"],
                active_channels=active_channels,
            ),
        })


def _inject_game_event_clips(curve, candidates, timeline, score, rms, clip_min, clip_max, active_channels=None, cuts=None):
    """Add candidates for kill/knock OCR anchors even when no arc spans them.

    A common failure mode is reaction-first, payoff-second: the streamer reacts
    while fighting, the OCR elimination banner appears a few seconds later, and
    the reaction arc closes before the payoff. Pure arc selection then clips the
    setup but misses the kill. This path anchors a candidate on the gameplay
    event and reaches back to the nearby reaction peak.
    """
    event_frames = [f for f in curve.frames if f.game_label in EVENT_ANCHOR_LABELS]
    if not event_frames:
        return

    runs, current = [], [event_frames[0]]
    for f in event_frames[1:]:
        if f.game_label == current[-1].game_label and f.timestamp - current[-1].timestamp <= 5.0:
            current.append(f)
        else:
            runs.append(current)
            current = [f]
    runs.append(current)

    voice_arousal = np.array([f.voice_arousal for f in curve.frames], dtype=np.float64)
    for run in runs:
        anchor = max(run, key=lambda f: f.game_evidence)
        if any(c["start"] <= anchor.timestamp <= c["end"] for c in candidates):
            continue

        context = (timeline >= anchor.timestamp - EVENT_LOOKBACK) & (timeline <= anchor.timestamp + EVENT_LOOKAHEAD)
        idxs = np.where(context)[0]
        if idxs.size:
            peak_idx = int(idxs[int(np.argmax(score[idxs]))])
        else:
            peak_idx = int(np.argmin(np.abs(timeline - anchor.timestamp)))
        anchor_idx = int(np.argmin(np.abs(timeline - anchor.timestamp)))
        start_idx, end_idx = sorted((peak_idx, anchor_idx))
        synthetic = boundaries_mod.Arc(
            start_idx=start_idx,
            end_idx=end_idx,
            peak_idx=peak_idx,
            peak_value=float(score[peak_idx]),
        )
        bounds = boundaries_mod.snap_boundaries(
            synthetic, timeline, rms,
            anchor_time=anchor.timestamp,
            anchor_label=anchor.game_label,
            clip_min=clip_min,
            clip_max=clip_max,
            cuts=cuts,
            voice_arousal=voice_arousal,
        )
        lo, hi = bounds["start"], bounds["end"]
        in_span = (timeline >= lo) & (timeline <= hi)
        dens = getattr(curve, "event_density", None)
        dens_val = 1.0
        if dens is not None and in_span.any():
            dens_arr = np.asarray(dens, dtype=np.float64)
            if dens_arr.size == timeline.size:
                dens_val = float(np.max(dens_arr[in_span]))
        candidates.append({
            **bounds,
            "reaction_auc": round(float(np.sum(score[in_span])), 4),
            "modality_breakdown": _modality_breakdown(curve, lo, hi),
            "game_evidence": round(anchor.game_evidence, 4),
            "game_label": anchor.game_label,
            "game_anchor_time": anchor.timestamp,
            "event_anchored": True,
            "event_density": round(dens_val, 4),
            **_candidate_context(
                curve, timeline, score, lo, hi,
                anchor_time=anchor.timestamp, peak_time=bounds["peak_timestamp"],
                active_channels=active_channels,
            ),
        })


def _inject_encounter_clips(curve, candidates, timeline, score, rms,
                            clip_min, clip_max, active_channels=None,
                            cuts=None, voice_arousal=None):
    """Anchor candidates on sustained combat-audio encounters (boss fights).

    The game-audio fusion channel lifts R(t) during a fight, but a genuinely
    quiet streamer can keep the fused curve below the arc threshold for the
    entire encounter — no arc, no candidate, boss kill lost. This path detects
    the encounter from the game_intensity track itself (per-VOD elevation bar,
    gap-bridged runs, motion agreement so a loud static menu can't qualify)
    and injects a candidate anchored on the strongest reaction moment in
    [run start, run end + lookahead] — the kill/decision beat.

    Bonus channel posture, never a bypass: injected candidates face every
    ordinary selection gate. Because game audio is deliberately NOT an
    agreement channel, a window still needs real human signal (the relief
    yell at the kill) to survive the modality gate — a cutscene the streamer
    sleeps through dies there by design.
    """
    ga = np.array([getattr(f, "game_intensity", 0.0) for f in curve.frames],
                  dtype=np.float64)
    if ga.size == 0 or float(np.max(ga)) <= 0.0:
        return
    motion = np.array([getattr(f, "motion", 0.0) for f in curve.frames],
                      dtype=np.float64)

    # Per-VOD elevation bar: half of this VOD's near-max intensity, floored so
    # a VOD whose "loudest" game audio is still trivial produces no runs.
    theta_ga = max(ENCOUNTER_MIN_LEVEL, 0.5 * float(np.percentile(ga, 99.5)))
    idxs = np.where(ga >= theta_ga)[0]
    if idxs.size == 0:
        return

    runs, current = [], [int(idxs[0])]
    for idx in idxs[1:]:
        if timeline[idx] - timeline[current[-1]] <= ENCOUNTER_GAP_SEC:
            current.append(int(idx))
        else:
            runs.append(current)
            current = [int(idx)]
    runs.append(current)

    runs = [r for r in runs
            if timeline[r[-1]] - timeline[r[0]] >= ENCOUNTER_MIN_RUN_SEC]
    if not runs:
        return

    # Fights move. A static screen with loud music (menu, pause, credits)
    # must not register as an encounter.
    motion_median = float(np.median(motion))
    runs = [r for r in runs
            if float(np.mean(motion[r[0]:r[-1] + 1])) >= motion_median]
    if not runs:
        return

    # Strongest encounters first; cap so an action-heavy VOD can't flood the
    # candidate pool (its arcs already cover most of these anyway).
    runs.sort(key=lambda r: float(np.max(ga[r[0]:r[-1] + 1])), reverse=True)

    injected = 0
    for run in runs:
        if injected >= ENCOUNTER_MAX_INJECT:
            break
        lo_t = float(timeline[run[0]])
        hi_t = float(timeline[run[-1]])
        window = (timeline >= lo_t) & (timeline <= hi_t + ENCOUNTER_LOOKAHEAD)
        w_idxs = np.where(window)[0]
        if w_idxs.size == 0:
            continue
        peak_idx = int(w_idxs[int(np.argmax(score[w_idxs]))])
        peak_time = float(timeline[peak_idx])
        if any(c["start"] <= peak_time <= c["end"] for c in candidates):
            continue

        start_idx = min(run[0], peak_idx)
        end_idx = max(run[-1], peak_idx)
        synthetic = boundaries_mod.Arc(
            start_idx=start_idx, end_idx=end_idx, peak_idx=peak_idx,
            peak_value=float(score[peak_idx]),
        )
        bounds = boundaries_mod.snap_boundaries(
            synthetic, timeline, rms, clip_min=clip_min, clip_max=clip_max,
            cuts=cuts, voice_arousal=voice_arousal,
        )
        lo, hi = bounds["start"], bounds["end"]
        in_span = (timeline >= lo) & (timeline <= hi)
        dens = getattr(curve, "event_density", None)
        dens_val = 1.0
        if dens is not None and in_span.any():
            dens_arr = np.asarray(dens, dtype=np.float64)
            if dens_arr.size == timeline.size:
                dens_val = float(np.max(dens_arr[in_span]))
        candidates.append({
            **bounds,
            "reaction_auc": round(float(np.sum(score[in_span])), 4),
            "modality_breakdown": _modality_breakdown(curve, lo, hi),
            "game_evidence": 0.0,
            "game_label": None,
            "game_anchor_time": None,
            "event_density": round(dens_val, 4),
            "encounter_anchored": True,
            **_candidate_context(
                curve, timeline, score, lo, hi,
                peak_time=bounds["peak_timestamp"],
                active_channels=active_channels,
            ),
        })
        injected += 1


VISUAL_WIN_CLUSTER_GAP = 30.0


def _inject_visual_win_clips(curve, candidates, timeline, score, rms, win_times,
                             stride_sec, clip_min, clip_max,
                             active_channels=None, cuts=None, voice_arousal=None):
    """Anchor candidates on sweep-detected win banners OCR never read.

    The outcome sweep samples frames coarsely, so consecutive hits from one
    win sequence (banner -> XP screens) cluster together; the cluster's first
    hit minus half a stride approximates banner onset. Candidates carry
    ``visual_win_anchored`` so selection's confirmation step can strip the win
    evidence again if the full visual judge does not also read a match win —
    a single misread frame must not fabricate a terminal_win.

    Wins already covered by an OCR terminal_win candidate are skipped; the
    injected ones flow through the same _merge_win_clips path as OCR wins
    (win-lead lookback, banner merging), so downstream treatment is identical.
    """
    if not win_times:
        return
    win_times = sorted(float(t) for t in win_times)
    clusters, current = [], [win_times[0]]
    for t in win_times[1:]:
        if t - current[-1] <= VISUAL_WIN_CLUSTER_GAP:
            current.append(t)
        else:
            clusters.append(current)
            current = [t]
    clusters.append(current)

    for cluster in clusters:
        anchor = max(0.0, cluster[0] - stride_sec / 2.0)
        covered = any(
            c.get("game_label") == "terminal_win"
            and c["start"] - 5.0 <= anchor <= c["end"] + 5.0
            for c in candidates
        )
        if covered:
            continue
        a_idx = int(np.argmin(np.abs(timeline - anchor)))
        synthetic = boundaries_mod.Arc(
            start_idx=a_idx, end_idx=a_idx, peak_idx=a_idx,
            peak_value=float(score[a_idx]),
        )
        bounds = boundaries_mod.snap_boundaries(
            synthetic, timeline, rms,
            anchor_time=anchor, clip_min=clip_min, clip_max=clip_max,
            cuts=cuts, voice_arousal=voice_arousal,
        )
        lo, hi = bounds["start"], bounds["end"]
        in_span = (timeline >= lo) & (timeline <= hi)
        candidates.append({
            **bounds,
            "reaction_auc": round(float(np.sum(score[in_span])), 4),
            "modality_breakdown": _modality_breakdown(curve, lo, hi),
            "game_evidence": 1.0,
            "game_label": "terminal_win",
            "game_anchor_time": anchor,
            "event_density": 1.0,
            "visual_win_anchored": True,
            # The exact frame the sweep classified as a win — the banner is
            # certainly visible here, so the confirming judge samples it.
            "visual_win_frame_time": float(cluster[0]),
            **_candidate_context(
                curve, timeline, score, lo, hi,
                anchor_time=anchor, peak_time=bounds["peak_timestamp"],
                active_channels=active_channels,
            ),
        })


def _inject_clip_command_clips(curve, candidates, timeline, score, rms, onset_theta,
                               bursts, clip_min, clip_max, active_channels=None,
                               cuts=None, voice_arousal=None):
    """Anchor candidates on viewer clip-command bursts (crowd ground truth).

    A "!clip" burst is a direct viewer vote that a postable moment JUST
    happened. For each burst: find the reaction peak in the lookback window
    behind it. If an existing candidate already contains that peak, tag it
    with the crowd evidence (selection adds a bonus). Otherwise inject a
    synthetic candidate around the peak — onset-expanded exactly like
    detect_arcs, so the window carries the moment's real arc extent instead
    of a bare point that would floor-pad into a weak-moment flag. Injected
    candidates are viewer-recommended anchors downstream. The command earns a
    bounded ranking bonus and guaranteed judge-pool consideration, but final
    selection still applies editorial vetoes, quality flooring, and the budget.
    """
    if not bursts:
        return
    n = len(score)
    for burst in bursts:
        anchor = float(burst.time)
        mask = (timeline >= anchor - CLIP_CMD_LOOKBACK) & (timeline <= anchor - CLIP_CMD_MIN_DELAY)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        peak_idx = int(idxs[int(np.argmax(score[idxs]))])
        peak_time = float(timeline[peak_idx])

        covered = False
        for c in candidates:
            if c["start"] <= peak_time <= c["end"]:
                c["crowd_clip"] = int(c.get("crowd_clip", 0) or 0) + int(burst.count)
                covered = True
        if covered:
            continue

        s = peak_idx
        while s > 0 and score[s - 1] > onset_theta:
            s -= 1
        e = peak_idx
        while e < n - 1 and score[e + 1] > onset_theta:
            e += 1
        synthetic = boundaries_mod.Arc(
            start_idx=s, end_idx=e, peak_idx=peak_idx,
            peak_value=float(score[peak_idx]),
        )
        bounds = boundaries_mod.snap_boundaries(
            synthetic, timeline, rms, clip_min=clip_min, clip_max=clip_max,
            cuts=cuts, voice_arousal=voice_arousal,
        )
        lo, hi = bounds["start"], bounds["end"]
        in_span = (timeline >= lo) & (timeline <= hi)
        dens = getattr(curve, "event_density", None)
        dens_val = 1.0
        if dens is not None and in_span.any():
            dens_arr = np.asarray(dens, dtype=np.float64)
            if dens_arr.size == timeline.size:
                dens_val = float(np.max(dens_arr[in_span]))
        candidates.append({
            **bounds,
            "reaction_auc": round(float(np.sum(score[in_span])), 4),
            "modality_breakdown": _modality_breakdown(curve, lo, hi),
            "game_evidence": 0.0,
            "game_label": None,
            "game_anchor_time": None,
            "event_density": round(dens_val, 4),
            "crowd_clip": int(burst.count),
            "clip_command_time": anchor,
            **_candidate_context(
                curve, timeline, score, lo, hi,
                peak_time=bounds["peak_timestamp"],
                active_channels=active_channels,
            ),
        })


def _inject_creator_marker_clips(
    curve,
    candidates,
    timeline,
    score,
    rms,
    onset_theta,
    markers,
    clip_min,
    clip_max,
    active_channels=None,
    cuts=None,
    voice_arousal=None,
):
    """Turn explicit Remember events into creator-protected candidates.

    A marker is approximate: a viewer clicks after seeing the payoff and may
    be behind the broadcaster. Search its planned region for the strongest
    reaction, then use the normal content-aware boundary snap.
    """
    if not markers or not len(timeline):
        return
    n = len(score)
    for marker in markers:
        try:
            marker_time = float(marker["timestamp"])
            search_start = float(marker.get("search_start", marker_time - 90.0))
            search_end = float(marker.get("search_end", marker_time + 30.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (marker_time, search_start, search_end)):
            continue
        mask = (timeline >= max(0.0, search_start)) & (timeline <= search_end)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        peak_idx = int(idxs[int(np.argmax(score[idxs]))])
        peak_time = float(timeline[peak_idx])
        marker_id = str(marker.get("event_id") or "").strip()

        covered = [
            candidate for candidate in candidates
            if candidate["start"] <= peak_time <= candidate["end"]
        ]
        if covered:
            candidate = max(
                covered,
                key=lambda item: float(item.get("reaction_auc", 0.0) or 0.0),
            )
            candidate["creator_protected"] = True
            candidate["manual_anchor"] = True
            candidate["recall_marker_time"] = marker_time
            marker_ids = candidate.setdefault("recall_marker_ids", [])
            if marker_id and marker_id not in marker_ids:
                marker_ids.append(marker_id)
            continue

        start_idx = peak_idx
        while start_idx > 0 and score[start_idx - 1] > onset_theta:
            start_idx -= 1
        end_idx = peak_idx
        while end_idx < n - 1 and score[end_idx + 1] > onset_theta:
            end_idx += 1
        synthetic = boundaries_mod.Arc(
            start_idx=start_idx,
            end_idx=end_idx,
            peak_idx=peak_idx,
            peak_value=float(score[peak_idx]),
        )
        bounds = boundaries_mod.snap_boundaries(
            synthetic,
            timeline,
            rms,
            clip_min=clip_min,
            clip_max=clip_max,
            cuts=cuts,
            voice_arousal=voice_arousal,
        )
        lo, hi = bounds["start"], bounds["end"]
        in_span = (timeline >= lo) & (timeline <= hi)
        density = getattr(curve, "event_density", None)
        density_value = 1.0
        if density is not None and in_span.any():
            density_array = np.asarray(density, dtype=np.float64)
            if density_array.size == timeline.size:
                density_value = float(np.max(density_array[in_span]))
        candidates.append({
            **bounds,
            "reaction_auc": round(float(np.sum(score[in_span])), 4),
            "modality_breakdown": _modality_breakdown(curve, lo, hi),
            "game_evidence": 0.0,
            "game_label": None,
            "game_anchor_time": None,
            "event_density": round(density_value, 4),
            "creator_protected": True,
            "manual_anchor": True,
            "recall_marker_ids": [marker_id] if marker_id else [],
            "recall_marker_time": marker_time,
            **_candidate_context(
                curve,
                timeline,
                score,
                lo,
                hi,
                peak_time=bounds["peak_timestamp"],
                active_channels=active_channels,
            ),
        })


def _merge_win_clips(candidates, curve, score, timeline, clip_max, gap=25.0, active_channels=None):
    """Collapse terminal_win candidates within the same banner window into one clip.

    A win banner persists ~20-30s and its reaction sub-arcs can be sparse, so the
    merge gap must span the whole banner. Distinct wins are minutes apart, so a
    25s gap never merges across wins.
    """
    wins = [c for c in candidates if c.get("game_label") == "terminal_win"]
    if not wins:
        return candidates
    others = [c for c in candidates if c.get("game_label") != "terminal_win"]

    wins.sort(key=lambda c: c["start"])
    clusters, current = [], [wins[0]]
    for c in wins[1:]:
        if c["start"] <= current[-1]["end"] + gap:
            current.append(c)
        else:
            clusters.append(current)
            current = [c]
    clusters.append(current)

    merged = []
    for cl in clusters:
        anchor_times = [
            c.get("game_anchor_time", c.get("peak_timestamp"))
            for c in cl
            if c.get("game_anchor_time", c.get("peak_timestamp")) is not None
        ]
        anchor_time = min(anchor_times) if anchor_times else min(c["start"] for c in cl)
        # Anchor on the banner onset, then look backward for the fight/reaction
        # that caused it. This avoids clips that start when "Victory" appears.
        lo = _win_lead_start(curve, timeline, score, anchor_time, clip_max)
        best = max(cl, key=lambda c: c.get("peak_value", 0.0))
        hi = max(lo + WIN_TARGET_LEN, anchor_time + boundaries_mod.TAIL_MIN)
        # A sweep-witnessed banner frame must stay inside the merged window:
        # the confirming judge samples that exact frame, and the banner can
        # trail the estimated onset by several seconds (observed: two real
        # wins stripped because the window ended 4s before the banner).
        witnessed_times = [
            float(c["visual_win_frame_time"]) for c in cl
            if c.get("visual_win_frame_time") is not None
        ]
        if witnessed_times:
            # Banner + celebration: the reaction AFTER a win is routinely what
            # the creator actually clips ("jeez, I got a crown one"), so keep
            # ~10s of post-banner room, not just the witnessed frame itself.
            hi = max(hi, min(witnessed_times) + 10.0)
        hi = min(hi, lo + clip_max)
        in_span = (timeline >= lo) & (timeline <= hi)
        dens = getattr(curve, "event_density", None)
        dens_val = max((c.get("event_density", 1.0) for c in cl), default=1.0)
        if dens is not None and in_span.any():
            dens_arr = np.asarray(dens, dtype=np.float64)
            if dens_arr.size == timeline.size:
                dens_val = max(dens_val, float(np.max(dens_arr[in_span])))
        merged.append({
            "start": round(lo, 3),
            "end": round(hi, 3),
            "peak_timestamp": best["peak_timestamp"],
            "peak_value": best.get("peak_value", 0.0),
            "reaction_auc": round(float(np.sum(score[in_span])), 4),
            "modality_breakdown": _modality_breakdown(curve, lo, hi),
            "game_evidence": max(c.get("game_evidence", 0.0) for c in cl),
            "game_label": "terminal_win",
            "game_anchor_time": anchor_time,
            # A merged win resting SOLELY on sweep-detected banners still needs
            # the full visual judge's confirmation (see build_reaction_clips);
            # any OCR-evidenced member makes the win self-standing.
            "visual_win_anchored": all(c.get("visual_win_anchored") for c in cl),
            "visual_win_frame_time": min(
                (c["visual_win_frame_time"] for c in cl
                 if c.get("visual_win_frame_time") is not None),
                default=None,
            ),
            "event_density": round(float(dens_val), 4),
            **_candidate_context(
                curve, timeline, score, lo, hi,
                anchor_time=anchor_time, peak_time=best["peak_timestamp"],
                active_channels=active_channels,
            ),
        })
    return others + merged


def _semantic_candidate_indices(
    candidates: List[dict],
    budget: int,
    tuning,
    bar: Optional[float] = None,
    primary_target: Optional[int] = None,
) -> List[int]:
    """Choose a high-quality, time-balanced pool for the local editor.

    A global raw-peak top-N made the judge blind to most of a long VOD.  This
    pool keeps the strongest global candidates, reserves most of the budget
    for local standouts across the full timeline, and always includes rare
    explicit anchors.  Evidence gates deliberately do not participate: the
    judge is one of the evidence sources that may correct those coarse rules.
    """
    if not candidates or budget <= 0:
        return []

    def potential(c: dict) -> float:
        breakdown = c.get("modality_breakdown") or {}
        face_boost = 0.0
        if selection_mod._exceptional_face_spike(c, tuning):
            # Spend a judge slot on a calibrated face reaction, but keep the
            # lift bounded so it cannot consume a high-baseline creator's pool.
            face_boost = min(
                1.0,
                float(breakdown.get("face_peak_delta", 0.0) or 0.0),
            )
        return (
            float(c.get("peak_value", 0.0) or 0.0)
            + 0.70 * selection_mod.compute_hook_score(c, tuning)
            + 1.50 * float(breakdown.get("startle", 0.0) or 0.0)
            + 0.30 * float(c.get("game_evidence", 0.0) or 0.0)
            + 0.20 * min(3, int(c.get("crowd_clip", 0) or 0))
            + face_boost
        )

    def preliminary_score(c: dict) -> float:
        signal_score, _mean, _duration = selection_mod._score_components(
            c, tuning, tuning.game_bonus,
        )
        return (
            signal_score
            + tuning.hook_score_weight * selection_mod.compute_hook_score(c, tuning)
        )

    # The final deck-coverage loop already judges signal-path winners. Reserve
    # this initial batch for challengers that need semantic evidence to enter
    # the deck, rather than paying twice for the same obvious top candidates.
    deferred_primary = set()
    if bar is not None and primary_target is not None and primary_target > 0:
        projected = sorted(
            range(len(candidates)),
            key=lambda i: (
                candidates[i].get("game_label") == "terminal_win"
                or int(candidates[i].get("crowd_clip", 0) or 0) > 0,
                preliminary_score(candidates[i]),
            ),
            reverse=True,
        )
        projected_selected: List[int] = []
        for i in projected:
            c = candidates[i]
            if not selection_mod.passes_evidence_gates(c, bar, tuning):
                continue
            if any(
                selection_mod._overlap_fraction(c, candidates[j]) > tuning.nms_overlap
                for j in projected_selected
            ):
                continue
            if len(projected_selected) >= primary_target:
                continue
            projected_selected.append(i)
        deferred_primary = set(projected_selected)
        for i in deferred_primary:
            candidates[i]["judge_pool_reason"] = "deferred_primary"

    valid = [
        i for i, c in enumerate(candidates)
        if float(c.get("end", 0.0)) > float(c.get("start", 0.0))
        and i not in deferred_primary
    ]
    raw_order = sorted(valid, key=lambda i: potential(candidates[i]), reverse=True)
    # Boundary formation can produce several windows around the same payoff.
    # Judging all of them spends scarce semantic capacity on framing variants,
    # not additional moments. Keep the strongest representative per overlap
    # cluster; final sentence/boundary refinement still handles presentation.
    order: List[int] = []
    for i in raw_order:
        if any(
            selection_mod._overlap_fraction(candidates[i], candidates[j]) > 0.65
            for j in order
        ):
            candidates[i]["judge_pool_reason"] = "overlap_duplicate"
            continue
        order.append(i)
    chosen: List[int] = []
    chosen_set = set()

    def add(index: int, reason: str) -> None:
        if index in chosen_set or len(chosen) >= budget:
            return
        chosen.append(index)
        chosen_set.add(index)
        candidates[index]["judge_pool_reason"] = reason

    # Human commands, terminal wins, and disciplined startle compounds are too
    # valuable to lose to sampling.
    for i in order:
        if selection_mod._creator_protected(candidates[i]):
            add(i, "creator_marker")
    # Marker judging adds to the ordinary candidate allowance so marking a stream does not
    # displace its other moments. Cap the added allowance to bound judge cost; excess
    # markers compete normally.
    marker_slots = len(chosen)
    if marker_slots:
        budget += min(
            marker_slots, max(1, (budget * POOL_MARKER_ALLOWANCE_PCT) // 100),
        )
    for i in order:
        c = candidates[i]
        if (
            int(c.get("crowd_clip", 0) or 0) > 0
            or c.get("game_label") == "terminal_win"
            or selection_mod._strong_startle(c, tuning)
            or selection_mod._visible_crowd_anchor(c, tuning)
        ):
            add(i, "anchor")

    # Use two global lenses: the full signal score catches near-cutoff candidates;
    # raw potential catches candidates that a lobby/padding label unfairly
    # penalized. The remaining half seeks local standouts across the timeline.
    signal_target = min(budget, max(len(chosen), max(1, budget // 3)))
    score_order = sorted(
        order, key=lambda i: preliminary_score(candidates[i]), reverse=True,
    )
    for i in score_order:
        if len(chosen) >= signal_target:
            break
        add(i, "signals_challenger")

    global_target = min(budget, max(len(chosen), max(1, budget // 2)))
    for i in order:
        if len(chosen) >= global_target:
            break
        add(i, "raw_challenger")

    temporal_slots = budget - len(chosen)
    if temporal_slots > 0 and order:
        duration = max(float(candidates[i].get("end", 0.0)) for i in order)
        if duration > 0.0:
            bins: List[List[int]] = [[] for _ in range(temporal_slots)]
            for i in order:
                if i in chosen_set:
                    continue
                c = candidates[i]
                midpoint = 0.5 * (
                    float(c.get("start", 0.0)) + float(c.get("end", 0.0))
                )
                bin_index = min(
                    temporal_slots - 1,
                    max(0, int((midpoint / duration) * temporal_slots)),
                )
                bins[bin_index].append(i)
            for local_candidates in bins:
                if local_candidates:
                    add(local_candidates[0], "temporal")

    # Sparse timelines leave empty temporal bins.  Re-spend that capacity on
    # the next strongest candidates instead of silently shrinking the batch.
    for i in order:
        if len(chosen) >= budget:
            break
        add(i, "global_fill")
    return chosen


# Visual-judge pool anti-blackout: split the VOD into this many segments and
# guarantee each one at least this many judge slots. A backstop, not a quota —
# it runs after the arousal deck band so the shape-specific lanes still decide
# how most of the budget is spent; it only rescues segments left at zero.
# Extra judge slots a marked session may add on top of the ordinary budget,
# as a percentage of it. Keeps creator markers from displacing the challenger
# pool one-for-one while bounding the added judge cost of a heavily marked
# stream.
POOL_MARKER_ALLOWANCE_PCT = 10

POOL_TEMPORAL_SEGMENTS = 12
POOL_MIN_SLOTS_PER_SEGMENT = 2

# Share of the budget the AROUSAL RANK claims. Deliberately a minority: the
# top of the score order is where the score already knows the answer. Measured
# 2026-07-23 on a 3h Fortnite VOD — in the band around the admission floor the
# signal score does not separate creator keeps from rejects at all (keeps
# scored 5.81/5.92/5.84, rejects 5.87/5.91/5.94/8.68), so slots spent by rank
# buy little information.
POOL_DECK_BAND_PCT = 25
# ...and the share spent on the candidates the score is LEAST sure about —
# those nearest the adaptive admission floor, where a verdict actually decides
# something. Three creator-confirmed keeps on that VOD sat in this band and
# were never judged under top-N allocation.
POOL_UNCERTAINTY_PCT = 45
# The adaptive floor is best_signal * this fraction (mirrors select_clips);
# used only to locate the uncertain band, never to admit or reject.
POOL_FLOOR_FRACTION = 0.5

# The review deck is capped at 15 clips, but judge coverage must be materially
# wider than the final deck.  Eight decks restores the historical challenger
# ceiling (8 * 15 = 120) while retaining one shared evidence lane: visual and
# semantic judges evaluate the same candidates before final ranking.
FINALIST_POOL_DECKS = 8
FINALIST_POOL_MAX_CANDIDATES = 120


def _round_or_none(value, digits: int = 4):
    """Round a float for the trace, preserving None (never judged) as None.

    The distinction matters: 0.0 means the judge scored it zero, None means it
    was never looked at, and the content-axis gates treat those differently.
    """
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _speech_lane_score(c: dict) -> float:
    """Rank speech/story challengers without preferring late loud peaks."""
    mb = c.get("modality_breakdown") or {}
    voice = float(mb.get("voice", 0.0) or 0.0)
    speech = float(mb.get("speech", 0.0) or 0.0)
    dead = selection_mod.dead_air_ratio_of(c)
    peak = float(c.get("peak_value", 0.0) or 0.0)
    # Voice/speech up, dead-air down; mild peak penalty so end-of-VOD chaos
    # does not monopolize the speech lane on narrative shapes.
    return (
        voice
        + 1.25 * speech
        + max(0.0, 0.55 - dead)
        - 0.08 * max(0.0, peak - 1.2)
    )


def _visual_candidate_indices(
    candidates: List[dict],
    budget: int,
    tuning,
    *,
    content_shape: Optional[str] = None,
) -> List[int]:
    """Pool for the visual judge: the arousal deck PLUS low-arousal challengers.

    A pure arousal-top-K pool re-creates the shortlist trap (§7.1: quiet keeps
    sat at signal-score ranks 267-716 of 814). Three lanes:
      * the projected arousal deck + near-miss band (the clips whose content
        evidence will also refine ordering),
      * win/crowd discovery anchors,
      * a low-arousal challenger lane: speech-bearing, non-housekeeping,
        deduplicated windows spread across the timeline — where quiet keeps
        (unread win banners, calm funny beats) actually live.

    Under ``content_shape == "narrative"``, spend more of the budget on the
    speech lane with a lower peak floor and stronger whole-VOD temporal
    spread so story beats are not starved by late loud peaks.
    """
    if not candidates or budget <= 0:
        return []

    narrative_mix = str(content_shape or "").strip().lower() == "narrative"
    if narrative_mix:
        # ~35% arousal deck / ~65% speech-spread (of non-anchor slots).
        deck_band = max(1, (budget * POOL_DECK_BAND_PCT) // 100)
        peak_floor = 0.45
        # Loud peaks already consumed deck_band slots — keep the speech lane
        # for quieter story/speech windows across the VOD.
        peak_ceiling = 1.55
        dead_air_max = 0.70
    else:
        deck_band = max(1, (budget * POOL_DECK_BAND_PCT) // 100)
        peak_floor = 0.9
        peak_ceiling = None
        dead_air_max = 0.60

    def preliminary(c: dict) -> float:
        signal_score, _mean, _duration = selection_mod._score_components(
            c, tuning, tuning.game_bonus,
        )
        return signal_score + tuning.hook_score_weight * selection_mod.compute_hook_score(c, tuning)

    valid = [
        i for i, c in enumerate(candidates)
        if float(c.get("end", 0.0)) > float(c.get("start", 0.0))
    ]
    for candidate in candidates:
        candidate.pop("visual_pool_reason", None)
    order = sorted(valid, key=lambda i: preliminary(candidates[i]), reverse=True)

    chosen: List[int] = []
    chosen_set = set()

    def add(i: int, reason: str) -> bool:
        if i in chosen_set or len(chosen) >= budget:
            return False
        c = candidates[i]
        if any(
            selection_mod._overlap_fraction(c, candidates[j]) > 0.65
            for j in chosen
        ):
            return False
        chosen.append(i)
        chosen_set.add(i)
        c["visual_pool_reason"] = reason
        return True

    # Creator-marked windows are explicit memory requests. Reserve their visual
    # read before other anchors so a low-arousal marker cannot be crowded out.
    for i in order:
        if selection_mod._creator_protected(candidates[i]):
            add(i, "creator_marker")

    # Other anchors next (rare, high value), then the arousal deck + near-miss band.
    for i in order:
        c = candidates[i]
        if (
            int(c.get("crowd_clip", 0) or 0) > 0
            or c.get("game_label") == "terminal_win"
            or selection_mod._strong_startle(c, tuning)
        ):
            add(i, "anchor")

    def _segment_index(i: int, span: float, seg_count: int) -> int:
        c = candidates[i]
        midpoint = 0.5 * (float(c.get("start", 0.0)) + float(c.get("end", 0.0)))
        return min(
            seg_count - 1,
            max(0, int((midpoint / max(1.0, span)) * seg_count)),
        )

    def _fill_blackout_backstop(min_per_segment: int) -> None:
        """Guarantee no stretch of the VOD draws ZERO judge slots.

        Deliberately a backstop, not a quota: it runs AFTER the arousal deck
        band so the shape-specific lanes keep deciding how the bulk of the
        budget is spent, and it only touches segments nothing has reached.

        Why it exists: an unjudged candidate can never become a content
        anchor, so a region with no slots cannot ship. Measured 2026-07-23 —
        the first 30 min of a 3h VOD held 107/694 candidates (15%), drew 3/40
        pool slots, all three came back ``skip``, and the first hour of the
        deck was empty while scoring the same as the rest of the VOD.
        """
        if min_per_segment <= 0 or budget <= 0:
            return
        span = max(float(candidates[i].get("end", 0.0)) for i in valid)
        seg_count = max(1, min(POOL_TEMPORAL_SEGMENTS, budget))
        covered = collections.Counter(
            _segment_index(i, span, seg_count) for i in chosen
        )
        pending: Dict[int, List[int]] = {}
        for i in valid:
            if i in chosen_set or candidates[i].get("housekeeping"):
                continue
            pending.setdefault(_segment_index(i, span, seg_count), []).append(i)
        for seg in range(seg_count):
            need = min_per_segment - covered.get(seg, 0)
            if need <= 0:
                continue
            bucket = sorted(
                pending.get(seg, []),
                key=lambda idx: preliminary(candidates[idx]),
                reverse=True,
            )
            for i in bucket:
                if need <= 0 or len(chosen) >= budget:
                    break
                if add(i, "temporal_backstop"):
                    need -= 1

    for i in order[: deck_band + 10]:
        if len(chosen) >= deck_band:
            break
        add(i, "signal_deck")

    # Anti-blackout: after the arousal band has clustered, make sure no stretch
    # of the VOD is still at zero slots before the speech lane spends the rest.
    _fill_blackout_backstop(POOL_MIN_SLOTS_PER_SEGMENT)

    def _fill_uncertainty_lane(slots: int) -> None:
        """Judge what the score is least sure about, not what it ranks highest.

        Candidates far above the admission floor are getting in regardless;
        candidates far below are not. The verdict only changes an outcome near
        the floor — and that is exactly the band where the signal score was
        measured to carry no signal about creator taste. So order by DISTANCE
        from the floor, ascending.
        """
        if slots <= 0:
            return
        scores = [preliminary(candidates[i]) for i in valid]
        if not scores:
            return
        floor = max(scores) * POOL_FLOOR_FRACTION
        pending = [
            i for i in valid
            if i not in chosen_set and not candidates[i].get("housekeeping")
        ]
        pending.sort(key=lambda i: abs(preliminary(candidates[i]) - floor))
        taken = 0
        for i in pending:
            if taken >= slots or len(chosen) >= budget:
                break
            if add(i, "uncertainty"):
                taken += 1

    _fill_uncertainty_lane((budget * POOL_UNCERTAINTY_PCT) // 100)

    def _fill_speech_lane(slots: int) -> None:
        if slots <= 0:
            return
        challengers = []
        for i in valid:
            if i in chosen_set or candidates[i].get("housekeeping"):
                continue
            peak = float(candidates[i].get("peak_value", 0.0) or 0.0)
            dead = selection_mod.dead_air_ratio_of(candidates[i])
            if dead > dead_air_max or peak < peak_floor:
                continue
            if peak_ceiling is not None and peak > peak_ceiling:
                continue
            challengers.append(i)
        if not challengers:
            return
        duration = max(
            float(candidates[i].get("end", 0.0)) for i in valid
        )
        # Whole-VOD bins (not arousal-rank order) so early/mid story windows
        # compete evenly with late loud peaks.
        bin_count = max(1, min(slots, 12 if narrative_mix else slots))
        bins: List[List[int]] = [[] for _ in range(bin_count)]
        for i in challengers:
            c = candidates[i]
            midpoint = 0.5 * (float(c.get("start", 0.0)) + float(c.get("end", 0.0)))
            bin_index = min(
                bin_count - 1,
                max(0, int((midpoint / max(1.0, duration)) * bin_count)),
            )
            bins[bin_index].append(i)
        # Round-robin across bins; within a bin prefer speech/story quality.
        progressed = True
        while len(chosen) < budget and progressed:
            progressed = False
            for local in bins:
                if len(chosen) >= budget:
                    break
                local.sort(
                    key=lambda idx: _speech_lane_score(candidates[idx]),
                    reverse=True,
                )
                for i in list(local):
                    if i in chosen_set:
                        continue
                    if add(i, "speech"):
                        progressed = True
                        break

    _fill_speech_lane(budget - len(chosen))

    # Remainder: narrative keeps preferring speechy leftovers; others fall
    # back to the arousal order.
    if narrative_mix and len(chosen) < budget:
        leftovers = sorted(
            (
                i for i in valid
                if i not in chosen_set and not candidates[i].get("housekeeping")
            ),
            key=lambda i: _speech_lane_score(candidates[i]),
            reverse=True,
        )
        for i in leftovers:
            if len(chosen) >= budget:
                break
            add(i, "narrative_speech")
    for i in order:
        if len(chosen) >= budget:
            break
        add(i, "signal_remainder")
    return chosen


def _finalist_candidate_indices(
    candidates: List[dict],
    seed_indices: List[int],
    max_clips: int,
    tuning,
) -> List[int]:
    """Build one bounded pool that both judges evaluate before final ranking.

    The visual discovery pool supplies temporally diverse and low-arousal
    challengers. Direct event/reaction anchors enter first, then the strongest
    remaining time-balanced challengers fill an eight-deck finalist band. This avoids
    the old feedback loop where provisional winners alone received richer
    features and were then re-ranked against unjudged challengers.
    """
    if not candidates:
        return []
    target = max(
        12,
        min(
            FINALIST_POOL_MAX_CANDIDATES,
            max(1, int(max_clips)) * FINALIST_POOL_DECKS,
        ),
    )
    budget = min(
        len(candidates),
        FINALIST_POOL_MAX_CANDIDATES,
        max(target, len(seed_indices)),
    )

    def preliminary(index: int) -> float:
        candidate = candidates[index]
        signal_score, _mean, _duration = selection_mod._score_components(
            candidate, tuning, tuning.game_bonus,
        )
        return signal_score + tuning.hook_score_weight * selection_mod.compute_hook_score(
            candidate, tuning,
        )

    valid = [
        index for index, candidate in enumerate(candidates)
        if float(candidate.get("end", 0.0)) > float(candidate.get("start", 0.0))
        and not candidate.get("housekeeping")
    ]
    ranked = sorted(valid, key=preliminary, reverse=True)
    chosen: List[int] = []
    seen = set()

    def add(index: int) -> None:
        if index in seen or index not in valid or len(chosen) >= budget:
            return
        chosen.append(index)
        seen.add(index)

    # Direct evidence cannot be crowded out by generic high-arousal windows.
    for index in ranked:
        if selection_mod._creator_protected(candidates[index]):
            add(index)
    for index in ranked:
        candidate = candidates[index]
        if (
            candidate.get("game_label")
            in ("elimination", "knock", "rank_progress", "terminal_win")
            or selection_mod._calibrated_face_spike(candidate, tuning)
            or selection_mod._strong_startle(candidate, tuning)
            or int(candidate.get("crowd_clip", 0) or 0) > 0
        ):
            add(index)
    for index in seed_indices:
        add(index)
    # Preserve the old broad semantic lane's most important property: most of
    # the extra evidence budget is spent on local standouts across the entire
    # VOD, not only on globally loud moments.  The visual discovery seeds still
    # enter first because they include uncertainty and quiet-speech lanes.
    for index in _semantic_candidate_indices(
        candidates, budget, tuning,
    ):
        add(index)
    for index in ranked:
        add(index)
    return chosen


def _ensure_visual_judge_capacity(visual_judge, count: int) -> None:
    """Grant bounded capacity for a finalist/deck-coverage judge pass."""
    if visual_judge is None or count <= 0:
        return
    ensure = getattr(visual_judge, "ensure_capacity", None)
    if callable(ensure):
        ensure(count)
        return
    session = getattr(visual_judge, "session", None)
    session_ensure = getattr(session, "ensure_capacity", None) if session else None
    if callable(session_ensure):
        session_ensure(count)
    elif hasattr(visual_judge, "remaining"):
        visual_judge.remaining = max(
            int(getattr(visual_judge, "remaining", 0) or 0), count,
        )
    elif session is not None and hasattr(session, "remaining"):
        session.remaining = max(
            int(getattr(session, "remaining", 0) or 0), count,
        )


def _causal_judge_windows(
    candidates: List[dict],
    windows: List[dict],
    *,
    max_gap: float,
    max_span: float,
) -> List[dict]:
    """Share one complete judge cut across a connected gameplay-event beat.

    Candidate generation intentionally keeps short scoring arcs. A fight,
    elimination, spoken reaction, and delayed payoff can therefore become four
    adjacent candidates even though an editor experiences them as one moment.
    Keep the scoring arcs untouched, but let both judges inspect the connected
    union whenever it contains an elimination/knock anchor.
    """
    if len(candidates) != len(windows):
        raise ValueError("candidate/window length mismatch")
    if not candidates:
        return []

    # Collapse boundary variants before building connections. Otherwise four
    # duplicate crops can consume the whole local component while the actual
    # reaction/payoff immediately after them remains outside the judge window.
    nodes: List[dict] = []
    by_bounds: Dict[tuple, dict] = {}
    for index, (candidate, window) in enumerate(zip(candidates, windows)):
        if candidate.get("housekeeping"):
            continue
        start = float(window.get("start", candidate.get("start", 0.0)))
        end = float(window.get("end", candidate.get("end", 0.0)))
        if end <= start:
            continue
        key = (round(start, 3), round(end, 3))
        node = by_bounds.get(key)
        if node is None:
            node = {
                "start": start,
                "end": end,
                "indices": [],
                "event": False,
            }
            by_bounds[key] = node
            nodes.append(node)
        node["indices"].append(index)
        node["event"] = bool(
            node["event"]
            or candidate.get("game_label") in boundaries_mod.EVENT_TAIL_LABELS
        )

    def interval_gap(node: dict, start: float, end: float) -> float:
        if float(node["end"]) < start:
            return start - float(node["end"])
        if float(node["start"]) > end:
            return float(node["start"]) - end
        return 0.0

    proposals: Dict[int, tuple] = {}
    for seed in nodes:
        if not seed["event"]:
            continue
        chosen = [seed]
        chosen_ids = {id(seed)}
        group_start = float(seed["start"])
        group_end = float(seed["end"])
        while True:
            adjacent = [
                node for node in nodes
                if id(node) not in chosen_ids
                and interval_gap(node, group_start, group_end) <= max_gap
                and max(group_end, float(node["end"]))
                - min(group_start, float(node["start"])) <= max_span
            ]
            if not adjacent:
                break
            next_node = min(
                adjacent,
                key=lambda node: (
                    interval_gap(node, group_start, group_end),
                    min(group_start, float(node["start"])),
                    -float(node["end"]),
                ),
            )
            chosen.append(next_node)
            chosen_ids.add(id(next_node))
            group_start = min(group_start, float(next_node["start"]))
            group_end = max(group_end, float(next_node["end"]))

        if len(chosen) <= 1:
            continue
        member_indices = [
            index for node in chosen for index in node["indices"]
        ]
        proposal = (
            len(chosen),
            group_end - group_start,
            group_start,
            group_end,
            len(member_indices),
        )
        for index in member_indices:
            if proposal[:2] > proposals.get(index, (0, 0.0))[:2]:
                proposals[index] = proposal

    result = [dict(window) for window in windows]
    for index, proposal in proposals.items():
        _unique_count, _span, start, end, member_count = proposal
        result[index] = {"start": start, "end": end}
        candidates[index]["judge_grouped"] = True
        candidates[index]["judge_group_size"] = member_count
        candidates[index]["judge_group_reason"] = "event_causal_chain"
    return result


def _apply_verdict(c: dict, verdict) -> None:
    c["semantic_hook"] = verdict.hook_strength
    c["semantic_self_contained"] = verdict.self_contained
    c["semantic_payoff"] = verdict.payoff
    c["semantic_moment_type"] = verdict.moment_type
    c["semantic_verdict"] = verdict.verdict
    c["semantic_title"] = verdict.title
    c["semantic_hook_line"] = verdict.hook_line


def _editorial_window(
    candidate: dict,
    timeline: np.ndarray,
    rms: Optional[np.ndarray],
    voice_arousal: Optional[np.ndarray],
    sentence_starts: np.ndarray,
    sentence_ends: np.ndarray,
    *,
    clip_min: float,
    clip_max: float,
) -> dict:
    """Build the action/sentence-complete cut without changing score features."""
    prepared = dict(candidate)
    reopened = boundaries_mod.extend_to_action_onset(
        prepared, timeline, rms,
        anchor_time=prepared.get("game_anchor_time"),
        anchor_label=prepared.get("game_label"),
        clip_max=clip_max,
    )
    prepared["start"], prepared["end"] = reopened["start"], reopened["end"]
    if sentence_starts.size or sentence_ends.size:
        refined = boundaries_mod.refine_to_sentences(
            prepared, sentence_starts, sentence_ends,
            anchor_time=prepared.get("game_anchor_time"),
            anchor_label=prepared.get("game_label"),
            clip_min=clip_min, clip_max=clip_max,
        )
        prepared["start"], prepared["end"] = refined["start"], refined["end"]
    trimmed = boundaries_mod.trim_dead_air(
        prepared, timeline, rms, voice_arousal=voice_arousal,
        anchor_time=prepared.get("game_anchor_time"),
        anchor_label=prepared.get("game_label"),
        clip_min=clip_min,
    )
    return {"start": float(trimmed["start"]), "end": float(trimmed["end"])}


# A moment counts as heard when at least this share of it was transcribed.
UNHEARD_SPEECH_MAX_COVERAGE = 0.5


def _mark_unheard_speech(candidates: List[dict], transcript: Optional[dict]) -> None:
    """Flag candidates whose speech was never transcribed.

    Smart scan transcribes only a budget of likely regions, and a failed ASR
    pass leaves no transcript at all. The judges are then told "(no clear
    speech)" for moments nobody listened to, which a visual judge reads as a
    frames-only skip. Measured 2026-09-23 on a 5-minute test cut: 0.9 min of
    scout coverage, all 19 finalists visual-skipped, empty deck, while Best
    quality on the same file kept 4. A full transcript carries no
    ``covered_regions`` and so covers every candidate.
    """
    if transcript is None:
        regions = []
    elif "covered_regions" not in transcript:
        for c in candidates:
            c["speech_unheard"] = False
        return
    else:
        regions = [
            (float(start), float(end))
            for start, end in transcript.get("covered_regions") or []
        ]
    for c in candidates:
        start = float(c.get("start", 0.0) or 0.0)
        end = float(c.get("end", start) or start)
        span = max(0.0, end - start)
        heard = sum(max(0.0, min(end, r_end) - max(start, r_start)) for r_start, r_end in regions)
        c["speech_unheard"] = span > 0.0 and heard / span < UNHEARD_SPEECH_MAX_COVERAGE


def build_reaction_clips(
    unified_signals,
    prosody_frames,
    face_frames=None,
    asr_hype_frames=None,
    audio_event_frames=None,
    chat_frames=None,
    settings: Optional[dict] = None,
    ranker=None,
    transcript: Optional[dict] = None,
    semantic_judge=None,
    visual_judge=None,
    outcome_sweep=None,
    ranker_diagnostics: Optional[dict] = None,
):
    """Run the full reaction pipeline. Returns (ReactionCurve, List[ReactionClip]).

    ``ranker``: optional learned ranker (LogisticRanker). When provided, it scores
    each candidate (replacing the signal selection score, plan §5.6). The engine
    stays decoupled from the DB -- the caller resolves and passes the model in.

    ``transcript``: optional Whisper result dict (word timestamps). When present,
    boundary snapping prefers sentence starts/ends over merely quiet frames
    (Package B, "human cuts"). Absent -> boundaries behave exactly as before.

    ``semantic_judge``: optional callable ``(candidates: List[dict]) ->
    Dict[int, SemanticVerdict]`` (Package C); the caller resolves the model and
    binds the transcript/chat context (engines.semantic.judge.judge_candidates),
    keeping this engine model-free like ``ranker``. None -> selection behaves
    exactly as before.

    ``visual_judge``: optional callable with the same contract, backed by a
    VLM over keyframes (engines.semantic.visual_judge.VisualJudgeSession). It
    attaches grounded ``visual_semantic_*`` fields to the candidates it judges
    and ENABLES the content axis in selection (two-axis admission): a visible
    match win or a complete verbal beat earns that candidate past the arousal
    gates and lifts its rank. None -> content axis off, selection byte-
    identical to before.
    """
    settings = settings or {}
    # Auto-detected game id (engines.reaction.game_detect), never user-picked.
    # Selects which per-game OCR event vocabulary classify_ocr_trigger checks
    # (on top of the always-active "generic" bucket) and which non-gameplay
    # scene lexicon to use below.
    game_name = str(settings.get("game", "generic"))
    # Temporal segmentation (plan 22 §4.1): on variety streams the pipeline
    # passes a labeled game timeline; OCR triggers and scene classification
    # then use the game that was ACTUALLY on screen at each timestamp instead
    # of one global label. Absent/empty -> the global game, exactly as before.
    seg_index = segments_mod.SegmentIndex(settings.get("game_segments"), default=game_name)

    def _classify_ocr(text: str, timestamp: float = None):
        return classify_ocr_trigger(text, game=seg_index.game_at(timestamp))

    curve = fusion.build_curve(
        unified_signals,
        prosody_frames,
        _classify_ocr,
        face_frames=face_frames,
        asr_hype_frames=asr_hype_frames,
        audio_event_frames=audio_event_frames,
        chat_frames=chat_frames,
        weights=settings.get("weights"),
        gamma=settings.get("gamma", fusion.DEFAULT_GAMMA),
        smooth_window=int(settings.get("smooth_window", 5)),
    )
    if not curve.frames:
        return curve, []

    # Which reaction modalities this VOD actually has tracks for -- drives the
    # signal_agreement denominator in _candidate_context so a stream without a
    # facecam/chat/ASR (or a game with no OCR HUD at all) isn't structurally
    # capped below a multi-signal bar it could never satisfy. Voice is always
    # present (prosody runs on every VOD); game evidence is intentionally
    # excluded -- it's handled as a separate path in selection.py.
    active_channels = {"voice"}
    if face_frames:
        active_channels.add("face")
    if asr_hype_frames:
        active_channels.add("speech")
    if audio_event_frames:
        active_channels.add("burst")
    if chat_frames:
        active_channels.add("chat")
    if any(getattr(f, "motion", 0.0) > 0.02 for f in curve.frames):
        active_channels.add("motion")
    # game_audio is deliberately NOT an agreement channel, same rationale as
    # game evidence: it describes the GAME's state, not a human reaction, and
    # adding it to the denominator would retroactively tighten the agreement
    # gate for every clip on every VOD with a burst track. It contributes to
    # the fused curve (fusion.py) and anchors encounter candidates instead.

    timeline = np.array([f.timestamp for f in curve.frames], dtype=np.float64)
    score = np.array(curve.score, dtype=np.float64)
    density_track = getattr(curve, "event_density", None)
    if density_track is None or len(density_track) != len(curve.frames):
        density_track = [1.0] * len(curve.frames)
    density_arr = np.array(density_track, dtype=np.float64)

    # Aligned RMS track for silence snapping. Prosody frames are sampled at
    # exact 1s intervals (extract_prosody), so integer-second keys align them
    # to the timeline in O(n) — same keying build_curve uses for every track.
    rms = None
    if prosody_frames:
        rms_by_t = {int(round(f.timestamp)): float(f.rms) for f in prosody_frames}
        rms = np.array(
            [rms_by_t.get(int(round(t)), 0.0) for t in timeline],
            dtype=np.float64,
        )

    # Visual cut track (plan 22 §4.2): shot-change outliers from the motion
    # scores perception already computed. Boundary snapping prefers a quiet
    # frame AT a cut over an equally quiet frame mid-shot, so clips stop
    # opening halfway through a camera cut. None when vision didn't run.
    cuts = None
    motion_by_t = {
        int(round(s.timestamp)): float(s.vision.motion_score)
        for s in unified_signals
        if getattr(s, "vision", None) is not None
    }
    if motion_by_t:
        motion = np.array([motion_by_t.get(int(round(t)), 0.0) for t in timeline],
                          dtype=np.float64)
        cuts = boundaries_mod.detect_cuts(motion)

    # Sentence boundaries from word-level ASR timestamps (Package B), built
    # once per VOD and applied to the WINNERS after selection (see below).
    # Candidates must keep silence-snapped windows: selection's dead-air and
    # intensity gates were tuned against them, and pre-selection sentence
    # alignment changed the whole deck (junk chatter windows became
    # speech-dense; event clips diluted below the routine-kill gate). No
    # transcript -> empty arrays -> nothing moves.
    sentence_starts, sentence_ends = sentence_index_mod.build_sentence_index(transcript)

    k, onset_k = _thresholds_from_settings(settings)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) * 1.4826
    theta = median + k * mad
    onset_theta = median + onset_k * mad

    # Clip length is content-driven: the arc's own onset/settle extent defines
    # the window. clip_min/clip_max are wide sanity invariants (boundaries.py),
    # overridable in settings only for tests -- the pipeline never passes them.
    clip_min = float(settings.get("clip_min", boundaries_mod.HARD_MIN))
    clip_max = float(settings.get("clip_max", boundaries_mod.HARD_MAX))
    max_clips = int(settings.get("max_clips", 10))

    arcs = boundaries_mod.detect_arcs(score, theta, onset_theta)
    # Runaway arcs (sustained multi-minute hype) split at internal lulls into
    # separate candidates instead of being blind-trimmed around one peak.
    arcs = boundaries_mod.segment_arcs(arcs, timeline, score, max_len=clip_max)

    voice_arousal = np.array([f.voice_arousal for f in curve.frames], dtype=np.float64)
    candidates: List[dict] = []
    for arc in arcs:
        # Game anchor = strongest OCR evidence inside the arc span, if any.
        span = curve.frames[arc.start_idx:arc.end_idx + 1]
        anchor_time = None
        anchor_evidence = 0.0
        anchor_label = None
        for f in span:
            if f.game_evidence > anchor_evidence:
                anchor_evidence = f.game_evidence
                anchor_time = f.timestamp
                anchor_label = f.game_label

        bounds = boundaries_mod.snap_boundaries(
            arc, timeline, rms,
            anchor_time=anchor_time, anchor_label=anchor_label,
            clip_min=clip_min, clip_max=clip_max,
            cuts=cuts, voice_arousal=voice_arousal,
        )

        lo, hi = bounds["start"], bounds["end"]
        in_span = (timeline >= lo) & (timeline <= hi)
        reaction_auc = float(np.sum(score[in_span]))  # dt = 1s
        breakdown = _modality_breakdown(curve, lo, hi)
        density_val = float(np.max(density_arr[in_span])) if in_span.any() else 1.0

        candidates.append({
            **bounds,
            "reaction_auc": round(reaction_auc, 4),
            "modality_breakdown": breakdown,
            "game_evidence": round(anchor_evidence, 4),
            "game_label": anchor_label,
            "game_anchor_time": anchor_time,
            "event_density": round(density_val, 4),
            **_candidate_context(
                curve, timeline, score, lo, hi,
                anchor_time=anchor_time, peak_time=bounds["peak_timestamp"],
                active_channels=active_channels,
            ),
        })

    # Win anchors: OCR can arrive after the reaction arc. Propose a synthetic
    # candidate centered on any uncovered banner, then let selection require
    # corroborating human reaction before it reaches the review deck.
    _inject_protected_wins(curve, candidates, timeline, score, rms, clip_min, clip_max, active_channels=active_channels, cuts=cuts)

    # Gameplay event anchors: eliminations/knocks should include the payoff even
    # when the streamer's reaction peaked seconds before OCR saw the banner.
    _inject_game_event_clips(curve, candidates, timeline, score, rms, clip_min, clip_max, active_channels=active_channels, cuts=cuts)

    # Combat-encounter anchors (game-audio track): boss fights / chases in
    # OCR-less games where a focused streamer stays too quiet to form an arc.
    _inject_encounter_clips(
        curve, candidates, timeline, score, rms, clip_min, clip_max,
        active_channels=active_channels, cuts=cuts, voice_arousal=voice_arousal,
    )

    # Visual outcome sweep (content axis, perception half): win banners OCR
    # cannot read (stylized fonts defeat PP-OCR) become terminal_win anchors
    # through the same machinery as OCR wins. The full visual judge must
    # confirm each injected win below or its evidence is stripped again.
    # The sweep is only trustworthy with its confirmation step, so it requires
    # the full visual judge to be present as well.
    if outcome_sweep is not None and visual_judge is not None and timeline.size:
        try:
            sweep_hits = outcome_sweep(float(timeline[-1]))
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - optional channel; never block selection
            print(f"Outcome sweep skipped: {exc}")
            sweep_hits = []
        sweep_wins = [t for (t, outcome) in sweep_hits if outcome == "match_win"]
        if sweep_wins:
            print(f"Outcome sweep: {len(sweep_wins)} match-win frame(s) detected")
            stride = float(getattr(outcome_sweep, "stride_sec", 12.0))
            _inject_visual_win_clips(
                curve, candidates, timeline, score, rms, sweep_wins, stride,
                clip_min, clip_max, active_channels=active_channels, cuts=cuts,
                voice_arousal=voice_arousal,
            )

    # Crowd clip commands: viewers typing "!clip" / "clip it" are a direct,
    # retroactive hint that a moment just happened. Tag covered candidates
    # with the evidence; inject lookback-anchored candidates for uncovered
    # bursts. Commands are prioritized for judging, then compete normally.
    if chat_frames:
        clip_bursts = chat_features_mod.cluster_clip_bursts(
            (float(f.timestamp), int(getattr(f, "clip_intents", 0) or 0))
            for f in chat_frames
        )
        _inject_clip_command_clips(
            curve, candidates, timeline, score, rms, onset_theta, clip_bursts,
            clip_min, clip_max, active_channels=active_channels, cuts=cuts,
            voice_arousal=voice_arousal,
        )

    # Creator Remember markers are stronger than audience suggestions. They
    # still retain structural non-content vetoes in selection.py, but model
    # disagreement alone cannot erase an explicit creator request.
    _inject_creator_marker_clips(
        curve,
        candidates,
        timeline,
        score,
        rms,
        onset_theta,
        settings.get("creator_markers") or [],
        clip_min,
        clip_max,
        active_channels=active_channels,
        cuts=cuts,
        voice_arousal=voice_arousal,
    )

    # A win banner persists ~20-30s, spawning several terminal_win sub-arcs (some
    # near-empty). Collapse each banner window into ONE clip so a win = one clip.
    candidates = _merge_win_clips(candidates, curve, score, timeline, clip_max, active_channels=active_channels)

    # Calibrated peak evidence must exist before semantic/visual pool routing
    # and before any frozen trace is written.
    _attach_face_spike_context(
        curve,
        candidates,
        absolute_min=selection_mod.DEFAULT_TUNING.face_spike_absolute_min,
        percentile=selection_mod.DEFAULT_TUNING.face_spike_percentile,
    )

    # VOD context (plan 22 §4.3): per-VOD normalization makes peak/auc values
    # relative to THIS VOD's distribution — attach the distribution itself so
    # ranker training can compare labels across VODs.
    vod_context = {
        "vod_median": round(median, 4),
        "vod_mad": round(mad, 4),
        "active_channel_count": float(len(active_channels)),
        "vod_duration": float(timeline[-1]) if timeline.size else 0.0,
        "game_profile": game_name,
    }
    for c in candidates:
        c.update(vod_context)

    # Precision-first Second Look features. These describe whether audio and
    # motion build and settle inside the candidate; the validated rejector is
    # allowed to suppress overflow only, never the Primary 15.
    signal_timestamps = np.asarray(
        [float(getattr(signal, "timestamp", 0.0) or 0.0) for signal in unified_signals],
        dtype=np.float64,
    )
    if signal_timestamps.size:
        trajectory_channels = [
            np.asarray(
                [
                    float(
                        getattr(getattr(signal, "audio", None), "spike_score", 0.0)
                        or 0.0
                    )
                    for signal in unified_signals
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    float(
                        getattr(getattr(signal, "audio", None), "amplitude", 0.0)
                        or 0.0
                    )
                    for signal in unified_signals
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    float(
                        getattr(getattr(signal, "vision", None), "motion_score", 0.0)
                        or 0.0
                    )
                    for signal in unified_signals
                ],
                dtype=np.float64,
            ),
        ]
        for candidate in candidates:
            candidate["second_look_trajectory_features"] = (
                pass_rejector_mod.trajectory_features(
                    signal_timestamps,
                    trajectory_channels,
                    float(candidate["start"]),
                    float(candidate["end"]),
                ).tolist()
            )

    # Non-gameplay scene labeling (lobby / intermission), computed for EVERY
    # candidate before selection (not just the winners) so selection.select_clips
    # can deprioritize lobby/menu banter instead of only labeling it after the
    # fact. The reaction engine scores the streamer's reaction, and a hyped
    # pregame lobby routinely outscores focused gameplay on that alone -- real
    # VOD data showed lobby clips landing as the #1 and #2 highest-scored clips
    # in an entire scan. Spatially routed text (chat/alert overlays removed)
    # when annotated, so alert-widget text can't tilt the classification.
    ocr_by_t = [
        (float(s.timestamp), layout_map_mod.routed_text(s.ocr))
        for s in unified_signals
        if getattr(s, "ocr", None) and getattr(s.ocr, "text", None)
    ]

    def _scene_for(lo: float, hi: float, has_game: bool) -> Optional[str]:
        texts = [t for (ts, t) in ocr_by_t if lo - 1.0 <= ts <= hi + 1.0]
        # Scene lexicon for the game on screen during THIS clip's window.
        window_game = seg_index.game_at((lo + hi) / 2.0)
        return scene_mod.classify_scene(texts, game=window_game, has_game_evidence=has_game)

    # Spoken text per window, for housekeeping (intro/outro/logistics) rejection.
    # Sorted (start, text) segments so a candidate's window is a slice, not a
    # rescan; empty/absent transcript -> no text -> nothing is ever flagged.
    _transcript_segs = sorted(
        (
            (
                float(seg.get("start", 0.0)),
                float(seg.get("end", seg.get("start", 0.0))),
                (seg.get("text") or "").strip(),
            )
            for seg in (transcript or {}).get("segments", []) or []
        ),
        key=lambda s: s[0],
    )
    vod_duration_sec = float(vod_context.get("vod_duration", 0.0) or 0.0)

    def _spoken_text(lo: float, hi: float) -> str:
        # Any segment overlapping the window, not just those STARTING inside it:
        # a sign-off spoken across the snapped boundary must still be seen.
        return " ".join(
            text for (s0, s1, text) in _transcript_segs
            if text and s0 <= hi and s1 >= lo
        )

    # Word-level index for the v7 delivery features. Punctuation and casing are
    # kept: exclaim_ratio reads Whisper's own "?"/"!" as an excitement proxy, so
    # a lowercased/stripped word list would silently zero that feature.
    _transcript_words = sorted(
        (
            ((word.get("word") or "").strip(), float(word.get("start", 0.0)))
            for seg in (transcript or {}).get("segments", []) or []
            for word in (seg.get("words") or []) or []
            if (word.get("word") or "").strip()
        ),
        key=lambda w: w[1],
    )

    def _words_in_window(lo: float, hi: float):
        return [(text, start) for (text, start) in _transcript_words
                if lo <= start <= hi]

    non_content_regions = editorial_hygiene_mod.detect_non_content_regions(
        ocr_by_t,
        _transcript_segs,
        vod_duration_sec,
    )

    for c in candidates:
        ocr_context = [
            text for (timestamp, text) in ocr_by_t
            if c["start"] - 1.0 <= timestamp <= c["end"] + 1.0
        ]
        # VLM-free lobby evidence, measured before de-duplication so it
        # reflects how much of the window is actually waiting/menu state. Kept
        # as a ratio rather than a bool so a clip that merely *starts* on a
        # loading screen and then cuts to gameplay is not scored the same as
        # one that never leaves the lobby. This is the only routine-state
        # signal available when the visual judge did not reach a candidate.
        c["routine_non_gameplay_ratio"] = (
            round(
                sum(
                    1
                    for text in ocr_context
                    if selection_mod._ROUTINE_NON_GAMEPLAY_RE.search(str(text))
                )
                / len(ocr_context),
                4,
            )
            if ocr_context
            else 0.0
        )
        # De-duplicate repeated per-frame OCR while retaining first-seen order.
        c["ocr_context"] = list(dict.fromkeys(ocr_context))[:6]
        crowd_score, crowd_terms = _visible_crowd_reaction(c["ocr_context"])
        c["visible_crowd_reaction"] = crowd_score
        c["visible_crowd_terms"] = crowd_terms
        c["scene_label"] = _scene_for(
            c["start"], c["end"],
            c.get("game_evidence", 0.0) > 0.0 or c.get("game_label") is not None,
        )
        # Delivery features v7: HOW the moment is spoken (escalation +
        # punctuation heat). Word-level slice, so it needs the word list
        # rather than the segment text used for housekeeping below.
        c.update(features_mod.delivery_features(
            _words_in_window(c["start"], c["end"]),
        ))
        spoken_text = _spoken_text(c["start"], c["end"])
        is_housekeeping, housekeeping_kind = housekeeping_mod.detect(
            spoken_text,
            c["start"], c["end"], vod_duration_sec,
        )
        c["housekeeping"] = is_housekeeping
        c["housekeeping_kind"] = housekeeping_kind
        c["moment_token_hashes"] = editorial_hygiene_mod.moment_token_hashes(
            spoken_text,
        )
        non_content_region = editorial_hygiene_mod.classify_candidate_region(
            c["start"],
            c["end"],
            non_content_regions,
        )
        c["terminal_non_content"] = non_content_region is not None
        c["terminal_non_content_kind"] = (
            non_content_region.kind if non_content_region is not None else None
        )
        c["terminal_non_content_region"] = (
            {
                "start": round(non_content_region.start, 3),
                "end": round(non_content_region.end, 3),
            }
            if non_content_region is not None else None
        )

    # selection_mode: "review" (default, wider deck) vs "auto"/"post" (precision).
    # Resolved before the semantic judge so the judge prefilter applies the
    # same tuning deck select_clips will gate with.
    selection_mode = settings.get(
        "selection_mode",
        settings.get("selectionMode", "review"),
    )
    tuning = selection_mod.tuning_for_mode(str(selection_mode))
    # Content axis (two-axis admission) activates only when a visual judge
    # actually runs this scan — installs without the VLM keep today's
    # arousal-only behavior exactly.
    if visual_judge is not None:
        tuning = dataclass_replace(tuning, content_axis=True)
    _mark_unheard_speech(candidates, transcript)

    selection_mod._assign_face_spike_rescue_eligibility(candidates, tuning)

    # Visual judge (content axis): grounded VLM evidence over a bounded pool
    # mixing the projected arousal deck with low-arousal challengers. The
    # session attaches visual_semantic_* fields in place; selection turns them
    # into per-candidate gate bypasses and rank lift (content_axis tuning).
    # Preliminary content-shape (pre-visual) steers the pool mix toward
    # speech/story windows on narrative VODs; post-judge inference below may
    # refine the shape used for selection policy.
    pre_shape_info = content_shape_mod.infer_content_shape(
        candidates, game=str(settings.get("game", "generic") or "generic"),
    )
    # Score the compact reaction arc, but let the VLM inspect the same
    # action/sentence-complete cut a creator will receive. Previously the judge
    # saw one window and post-selection refinement could export a materially
    # different 20-50s clip.
    if visual_judge is not None:
        editorial_windows = [
            _editorial_window(
                candidate, timeline, rms, voice_arousal,
                sentence_starts, sentence_ends,
                clip_min=clip_min, clip_max=clip_max,
            )
            for candidate in candidates
        ]
        # The per-candidate editorial cut IS the shippable moment: action-onset
        # reopen gives it the lead-up, sentence refinement gives it clean speech
        # edges, dead-air trim tightens it. Keep it BEFORE causal grouping
        # widens the judged span, because the two windows answer different
        # questions:
        #
        #   judge_start/judge_end   -- what evidence the judges should SEE, so a
        #                              verdict is not made on a fragment that
        #                              excludes its own payoff.
        #   export_start/export_end -- what the creator should RECEIVE.
        #
        # Conflating them shipped the whole causal chain (up to clip_max, 90 s),
        # so a creator had to scrub a minute-plus clip to find the moment. That
        # also broke the property this judge-window block was added to
        # guarantee -- see the comment above: the judge is supposed to inspect
        # "the same action/sentence-complete cut a creator will receive".
        # Causal grouping made the judged window stop being that cut; keeping
        # both windows restores it.
        for candidate, editorial in zip(candidates, editorial_windows):
            candidate["export_start"] = round(editorial["start"], 3)
            candidate["export_end"] = round(editorial["end"], 3)
        editorial_windows = _causal_judge_windows(
            candidates,
            editorial_windows,
            max_gap=CAUSAL_JUDGE_MAX_GAP_SEC,
            max_span=float(clip_max),
        )
        for candidate, editorial in zip(candidates, editorial_windows):
            candidate["judge_start"] = round(editorial["start"], 3)
            candidate["judge_end"] = round(editorial["end"], 3)
            candidate["judge_window_changed"] = bool(
                abs(candidate["judge_start"] - float(candidate["start"])) > 0.001
                or abs(candidate["judge_end"] - float(candidate["end"])) > 0.001
            )
    visual_pool: List[int] = []
    if visual_judge is not None and candidates:
        visual_budget = max(1, int(settings.get("visual_judge_budget", 40)))
        visual_pool = _visual_candidate_indices(
            candidates, visual_budget, tuning,
            content_shape=pre_shape_info.get("shape"),
        )
        for i in visual_pool:
            candidates[i]["visual_pool"] = True
        if visual_pool:
            print(
                f"Visual pool: {len(visual_pool)}/{visual_budget} "
                f"(pre_shape={pre_shape_info.get('shape')})"
            )
            try:
                visual_judge([candidates[i] for i in visual_pool])
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001 - optional channel; never block selection
                print(f"Visual judge skipped: {exc}")

        # Sweep-win confirmation: a terminal_win resting solely on coarse
        # sweep frames keeps its evidence only when the full judge's
        # anchor-aware read also sees the win. Unconfirmed -> the candidate
        # faces the ordinary gates as a plain moment.
        stripped = 0
        for c in candidates:
            if (
                c.get("visual_win_anchored")
                and c.get("game_label") == "terminal_win"
                and not selection_mod._visual_win_read(c)
            ):
                c["game_label"] = None
                c["game_evidence"] = 0.0
                c["game_anchor_time"] = None
                stripped += 1
        if stripped:
            print(f"Outcome sweep: {stripped} unconfirmed win anchor(s) stripped")

    # One finalist pool, one evidence packet. Both optional judges see the
    # same bounded set before learned ranking and final selection. This closes
    # the provisional-winner feedback loop: a clip no longer gains semantic or
    # visual features merely because an earlier partial score happened to put
    # it in the deck while an otherwise comparable challenger stayed blank.
    finalist_indices = _finalist_candidate_indices(
        candidates, visual_pool, max_clips, tuning,
    )
    for index in finalist_indices:
        candidates[index]["finalist_pool"] = True

    missing_visual = [
        candidates[index] for index in finalist_indices
        if visual_judge is not None
        and not (
            candidates[index].get("visual_semantic_verdict")
            or candidates[index].get("visual_verdict")
        )
        and not candidates[index].get("_visual_judge_attempted")
    ]
    if missing_visual:
        _ensure_visual_judge_capacity(visual_judge, len(missing_visual))
        try:
            visual_judge(missing_visual)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - optional channel
            print(f"Visual judge (finalists) skipped: {exc}")
        finally:
            for candidate in missing_visual:
                candidate["_visual_judge_attempted"] = True

    missing_semantic = [
        candidates[index] for index in finalist_indices
        if semantic_judge is not None
        and candidates[index].get("semantic_verdict") is None
        and not candidates[index].get("_judge_attempted")
    ]
    if missing_semantic:
        try:
            verdicts = semantic_judge(missing_semantic) or {}
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - optional channel
            print(f"Semantic judge (finalists) skipped: {exc}")
            verdicts = {}
        for candidate in missing_semantic:
            candidate["_judge_attempted"] = True
        for local_index, verdict in verdicts.items():
            _apply_verdict(missing_semantic[local_index], verdict)

    if finalist_indices and (visual_judge is not None or semantic_judge is not None):
        visual_count = sum(
            1 for index in finalist_indices
            if candidates[index].get("visual_semantic_verdict")
            or candidates[index].get("visual_verdict")
        )
        semantic_count = sum(
            1 for index in finalist_indices
            if candidates[index].get("semantic_verdict") is not None
        )
        print(
            f"Finalist evidence: {len(finalist_indices)} candidates; "
            f"visual={visual_count}, semantic={semantic_count}"
        )

    # Content-shape inference (always logged). Policy applies by default
    # (``content_shape_selection``); set false to keep arousal-only admission.
    # Re-run after the visual judge so story/verbal density can refine the
    # pre-visual guess.
    content_shape_info = content_shape_mod.infer_content_shape(
        candidates, game=str(settings.get("game", "generic") or "generic"),
    )
    content_shape_info["pre_visual_shape"] = pre_shape_info.get("shape")
    content_shape_info["pre_visual_confidence"] = pre_shape_info.get("confidence")
    shape_policy_on = content_shape_mod.content_shape_policy_enabled(settings)
    if shape_policy_on:
        tuning = selection_mod.tuning_for_content_shape(
            tuning, content_shape_info.get("shape"), enabled=True,
        )
    print(
        "Content shape: "
        f"{content_shape_info.get('shape')} "
        f"(confidence={content_shape_info.get('confidence')}, "
        f"policy={'on' if shape_policy_on else 'off'})"
    )

    # The current learned feature vector needs both observed opening signals and
    # semantic fields before inference. ranker_service rejects stale schemas.
    for c in candidates:
        c["hook_score"] = selection_mod.compute_hook_score(c, tuning)

    # Optional separate admission model. The deck ranker is pairwise-trained:
    # it promises a monotonic ORDERING within a deck, not an absolute
    # publishability probability, and its Platt block is a retrofit onto that.
    # A pointwise gate model trained on keep/pass across all sources is the
    # right thing to threshold, so allow one to supply
    # ranker_publishability_score while the deck ranker keeps ordering.
    # Measured 2026-08-16: the gate model scores AUC 0.711 grouped-CV and
    # 0.749 on the off-policy slice -- the floor-rejected material a rescue
    # lane has to judge -- against 0.618 for the best single trace feature.
    gate_model = None
    gate_path = settings.get("ranker_calibrated_gate_model")
    if gate_path:
        try:
            from engines.reaction.ranker import LogisticRanker as _GateRanker
            gate_model = _GateRanker.load(gate_path)
            if not getattr(gate_model, "calibration", None):
                print(f"Gate model {gate_path} has no calibration block; ignoring")
                gate_model = None
            else:
                print(f"Calibrated admission scored by gate model: {gate_path}")
        except Exception as exc:  # noqa: BLE001 - never fail a scan on this
            print(f"Gate model load failed ({exc}); falling back to the ranker")
            gate_model = None

    # Learned ranker: score with the complete current feature vector.
    if ranker is not None:
        for c in candidates:
            vec = features_mod.feature_vector(c)
            c["ranker_score"] = float(ranker.predict(vec[None, :])[0])
            calibrated = (
                gate_model.predict_calibrated if gate_model is not None
                else getattr(ranker, "predict_calibrated", None)
            )
            if callable(calibrated):
                try:
                    c["ranker_publishability_score"] = float(
                        calibrated(vec[None, :])[0]
                    )
                except (ValueError, TypeError, AttributeError):
                    pass
            personal_predict = getattr(ranker, "predict_personal", None)
            if callable(personal_predict) and bool(getattr(ranker, "has_personal", False)):
                personal_score = float(personal_predict(vec[None, :])[0])
                c["personal_ranker_score"] = personal_score
                c["personal_preference_delta"] = personal_score - c["ranker_score"]
            # A newly trained per-install model is a shadow challenger until
            # it earns promotion on future reviewed decks. Persist its score
            # under a distinct namespace: selection.py intentionally knows
            # nothing about these fields, so even a non-zero personal blend
            # override cannot let an unvalidated challenger move the deck.
            challenger_predict = getattr(ranker, "predict_challenger", None)
            if callable(challenger_predict) and bool(
                getattr(ranker, "has_challenger", False)
            ):
                challenger_score = float(challenger_predict(vec[None, :])[0])
                c["personal_challenger_score"] = challenger_score
                c["personal_challenger_delta"] = (
                    challenger_score - c["ranker_score"]
                )

    # Default blend@0.35, chosen 2026-07-30 over both tail_only and replace.
    #
    # tail_only was the previous default: the ranker scored every candidate so
    # the Second-look tier could order by it, while Primary stayed
    # signal-ordered. It was the right call while the only measured alternative
    # was `replace`, which is genuinely worse -- replace shrinks decks
    # (clip_count 14.06 -> 13.38), spends its whole recall slack, and is
    # unstable: it PASSED the ship gate on 30 folds and FAILED on 34, and it
    # flipped sign between blinded cohorts (p@5 -0.067 then +0.200).
    #
    # Blending percentiles at 0.35 keeps the signal path's ordering and lets the
    # model tilt it. Four independent measurements agree, which no ranker
    # configuration had ever managed before:
    #   ship gate, on-policy, 34 folds : p@5 +0.053, recall -0.007, deck size intact
    #   policy sweep, on-policy, 35    : p@5 +0.053, recall -0.014
    #   blinded cohort 1, off-policy   : p@5 +0.067, AP +0.070
    #   blinded cohort 2, off-policy   : p@5 +0.200, AP +0.098
    #
    # Inert until a ranker model exists: the blend branch needs candidates
    # carrying ranker_score, so with no model on disk this resolves to the same
    # signal ordering as before.
    #
    # Scoped to the BASE model. Every measurement behind blend@0.35 is of the
    # LOVO base ranker. ``active_mode=personal`` only means a validated adapter
    # is attached to the stack: ranker_score remains base-owned, while the
    # separate personal delta stays at weight zero until its ordering policy has
    # its own measurement.
    _active_mode = str((ranker_diagnostics or {}).get("active_mode") or "")
    _default_policy = "blend" if _active_mode in ("base", "personal") else "tail_only"
    ranker_policy = str(
        settings.get("ranker_policy", _default_policy) or _default_policy
    )
    raw_blend_weight = settings.get("ranker_blend_weight")
    ranker_blend_weight = float(
        0.35 if raw_blend_weight is None else raw_blend_weight
    )
    # Personal influence remains shadow-only until its own ordering-policy gate
    # passes. Even when enabled later, selection.py hard-caps it at 0.25.
    personal_ranker_blend_weight = float(
        settings.get("personal_ranker_blend_weight", 0.0) or 0.0
    )
    raw_publishability_floor = settings.get("ranker_publishability_floor")
    ranker_publishability_floor = float(
        0.0 if raw_publishability_floor is None else raw_publishability_floor
    )
    # Calibrated admission: absolute P(keep) gate replaces the VOD-relative
    # arousal floors (see SelectionTuning.calibrated_admission). Requires a
    # ranker whose scores carry a calibration block AND a non-zero
    # publishability floor; otherwise it is inert and selection is
    # byte-identical to before.
    if bool(settings.get("ranker_calibrated_admission", False)):
        from dataclasses import replace as _dc_replace
        tuning = _dc_replace(tuning, calibrated_admission=True)
    # Union variant: incumbent gates unchanged + calibrated rescue of
    # floor-rejected candidates. Mutually exclusive with full replacement;
    # replacement wins if both are set.
    elif bool(settings.get("ranker_calibrated_union", False)):
        from dataclasses import replace as _dc_replace
        tuning = _dc_replace(tuning, calibrated_union=True)
    # False since 2026-07-30, to match the configuration blend@0.35 was measured
    # under. The tighter personalized floor (0.60 of the VOD's best signal
    # instead of 0.50) is premised on a learned ranker reliably narrowing the
    # pool before ordering it -- true under `replace`, not under `blend`, where
    # the deck is still substantially signal-ordered. Leaving it on would ship a
    # third configuration that none of the four measurements covered.
    ranker_use_personalized_adaptive_floor = bool(
        settings.get("ranker_use_personalized_adaptive_floor", False)
    )
    second_look_rejector = pass_rejector_mod.load_second_look_model(
        source_key=settings.get("second_look_source_key"),
    )

    def _snapshot_selection_inputs() -> None:
        """Freeze the exact candidate state entering the next selector pass."""
        # Boundaries move between passes, so coverage is re-read each time.
        _mark_unheard_speech(candidates, transcript)
        for candidate in candidates:
            candidate["_selection_start"] = float(candidate.get("start", 0.0))
            candidate["_selection_end"] = float(candidate.get("end", 0.0))
            candidate["_selection_feature_snapshot"] = features_mod.feature_dict(
                candidate
            )

    _snapshot_selection_inputs()
    selected = selection_mod.select_clips(
        candidates, score, max_clips=max_clips, k=k, tuning=tuning,
        ranker_policy=ranker_policy,
        ranker_blend_weight=ranker_blend_weight,
        personal_ranker_blend_weight=personal_ranker_blend_weight,
        ranker_publishability_floor=ranker_publishability_floor,
        ranker_use_personalized_adaptive_floor=ranker_use_personalized_adaptive_floor,
        second_look_rejector=second_look_rejector,
    )

    def _reselect_after_judge_update() -> List[dict]:
        # Freeze learned probabilities across coverage passes. New judge
        # fields may alter deterministic qualification/conflict rules, but a
        # provisional winner cannot entrench itself through a richer feature
        # vector than the challenger that replaces it.
        _snapshot_selection_inputs()
        return selection_mod.select_clips(
            candidates, score, max_clips=max_clips, k=k, tuning=tuning,
            ranker_policy=ranker_policy,
            ranker_blend_weight=ranker_blend_weight,
            personal_ranker_blend_weight=personal_ranker_blend_weight,
            ranker_publishability_floor=ranker_publishability_floor,
            ranker_use_personalized_adaptive_floor=ranker_use_personalized_adaptive_floor,
            second_look_rejector=second_look_rejector,
        )

    # Bounded shared coverage for the rare winner that entered from outside
    # the finalist pool. Each round sends the same union to both available
    # judges, then re-applies common qualify/dedupe/rank policy with learned
    # ordering frozen. There is no selected-only semantic lane and no visual
    # free pass.
    if visual_judge is not None or semantic_judge is not None:
        for _ in range(3):
            coverage = [
                candidate for candidate in selected
                if (
                    visual_judge is not None
                    and not (
                        candidate.get("visual_semantic_verdict")
                        or candidate.get("visual_verdict")
                    )
                    and not candidate.get("_visual_judge_attempted")
                )
                or (
                    semantic_judge is not None
                    and candidate.get("semantic_verdict") is None
                    and not candidate.get("_judge_attempted")
                )
            ]
            if not coverage:
                break
            for candidate in coverage:
                candidate["finalist_pool"] = True

            visual_targets = [
                candidate for candidate in coverage
                if visual_judge is not None
                and not (
                    candidate.get("visual_semantic_verdict")
                    or candidate.get("visual_verdict")
                )
                and not candidate.get("_visual_judge_attempted")
            ]
            if visual_targets:
                _ensure_visual_judge_capacity(visual_judge, len(visual_targets))
                try:
                    visual_judge(visual_targets)
                except InterruptedError:
                    raise
                except Exception as exc:  # noqa: BLE001 - optional channel
                    print(f"Visual judge (shared coverage) skipped: {exc}")
                finally:
                    for candidate in visual_targets:
                        candidate["_visual_judge_attempted"] = True

            semantic_targets = [
                candidate for candidate in coverage
                if semantic_judge is not None
                and candidate.get("semantic_verdict") is None
                and not candidate.get("_judge_attempted")
            ]
            if semantic_targets:
                try:
                    verdicts = semantic_judge(semantic_targets) or {}
                except InterruptedError:
                    raise
                except Exception as exc:  # noqa: BLE001 - optional channel
                    print(f"Semantic judge (shared coverage) skipped: {exc}")
                    verdicts = {}
                for candidate in semantic_targets:
                    candidate["_judge_attempted"] = True
                for local_index, verdict in verdicts.items():
                    _apply_verdict(semantic_targets[local_index], verdict)

            fully_evidenced = sum(
                1 for candidate in coverage
                if (
                    visual_judge is None
                    or candidate.get("visual_semantic_verdict")
                    or candidate.get("visual_verdict")
                )
                and (
                    semantic_judge is None
                    or candidate.get("semantic_verdict") is not None
                )
            )
            print(
                f"Shared judge coverage: {fully_evidenced}/{len(coverage)} "
                "selected candidates fully evidenced"
            )
            if fully_evidenced <= 0:
                break
            selected = _reselect_after_judge_update()

    # This is an audit-only counterfactual.  It runs after all Base selection
    # evidence is final and only annotates the trace rows that Second Look will
    # materialize; ``selected`` and the live candidate dictionaries' Primary
    # selection fields are never replaced from the copies above.
    _annotate_shadow_primary_audit(
        candidates, selected, score,
        ranker_diagnostics=ranker_diagnostics,
        max_clips=max_clips,
        k=k,
        tuning=tuning,
        ranker_policy=ranker_policy,
        ranker_blend_weight=ranker_blend_weight,
        ranker_publishability_floor=ranker_publishability_floor,
        ranker_use_personalized_adaptive_floor=(
            ranker_use_personalized_adaptive_floor
        ),
        second_look_rejector=second_look_rejector,
    )

    # Action-onset reopen (winners only, before sentence refinement): an
    # elim/knock window is built around the REACTION arc, but the streamer
    # reacts 15-25s after the fight that caused it, so the shipped clip opened
    # on the kill banner with the fight already over. Measured 2026-07-23 on a
    # creator-confirmed keep: combat audio ran from 4732, the window opened at
    # 4759, and the creator independently asked for ~4725; this lands at 4728.
    for c in selected:
        if visual_judge is not None and c.get("export_start") is not None:
            # Ship the editorial cut, NOT the grouped judge span. The judges
            # still saw the wider causal window -- that is what judge_start/
            # judge_end are for -- but the creator receives the moment with its
            # lead-up and payoff rather than the whole chain up to clip_max.
            c["start"] = float(c["export_start"])
            c["end"] = float(c["export_end"])
        else:
            reopened = boundaries_mod.extend_to_action_onset(
                c, timeline, rms,
                anchor_time=c.get("game_anchor_time"),
                anchor_label=c.get("game_label"),
                clip_max=clip_max,
            )
            c["start"], c["end"] = reopened["start"], reopened["end"]

    # Sentence refinement (Package B), winners only: open on a sentence start,
    # close on a sentence end. Presentation-only — selection already judged
    # the moment on its silence-snapped window, and features keep describing
    # that window. Invariant violations inside refine_to_sentences revert to
    # the selected bounds, so a refined clip can never lose its payoff.
    if sentence_starts.size or sentence_ends.size:
        for c in selected:
            if visual_judge is not None and c.get("export_start") is not None:
                continue
            refined = boundaries_mod.refine_to_sentences(
                c, sentence_starts, sentence_ends,
                anchor_time=c.get("game_anchor_time"),
                anchor_label=c.get("game_label"),
                clip_min=clip_min, clip_max=clip_max,
            )
            c["start"], c["end"] = refined["start"], refined["end"]

    # Dead-air trim (winners only, after sentence refinement). Tightens a clip
    # that still opens/closes on near-silence so it starts on the moment, not a
    # quiet gap. Presentation-only and self-reverting: an invariant break inside
    # trim_dead_air returns the input window, so a clip can never lose its
    # payoff, and a clip with no silent head/tail is returned unchanged.
    for c in selected:
        if visual_judge is not None and c.get("export_start") is not None:
            continue
        trimmed = boundaries_mod.trim_dead_air(
            c, timeline, rms, voice_arousal=voice_arousal,
            anchor_time=c.get("game_anchor_time"),
            anchor_label=c.get("game_label"),
            clip_min=clip_min,
        )
        c["start"], c["end"] = trimmed["start"], trimmed["end"]

    # Conversational completion (presentation-only): when the text and visual
    # judges disagree about whether a winner is self-contained, recover the
    # adjacent spoken setup and, for a missing landing, the next one or two
    # sentences. Selection and learned scores are already final above.
    for c in selected:
        policy = boundaries_mod.conversational_expansion_policy(c)
        expanded = boundaries_mod.expand_conversational_context(
            c,
            sentence_starts,
            sentence_ends,
            timeline,
            rms,
            expand_lead=bool(policy["lead"]),
            expand_tail=bool(policy["tail"]),
            clip_max=clip_max,
        )
        c["boundary_context_reason"] = policy["reason"]
        c["boundary_context_lead_added"] = expanded["lead_added"]
        c["boundary_context_tail_added"] = expanded["tail_added"]
        c["boundary_context_applied"] = bool(
            expanded["lead_added"] > 0.0 or expanded["tail_added"] > 0.0
        )
        c["start"], c["end"] = expanded["start"], expanded["end"]

    # A strong off-game reaction is admitted only with unusually dense human
    # and gameplay-change evidence. Give those rare social moments a short
    # breathing tail after sentence snapping; the founder-kept cat-on-screen
    # moment otherwise cut the response two seconds before it settled.
    for c in selected:
        if c.get("strong_lobby_reaction"):
            c["end"] = round(min(
                float(c["end"]) + STRONG_LOBBY_TAIL_PAD,
                float(c["start"]) + clip_max,
                float(timeline[-1]) if timeline.size else float(c["end"]),
            ), 3)

    # Persist a compact decision audit on the curve.  Candidate dictionaries
    # are otherwise private to this function, which previously made a miss
    # indistinguishable from "no candidate existed" once the call returned.
    curve.selection_trace = [
        {
            "start": round(float(c.get("start", 0.0)), 3),
            "end": round(float(c.get("end", 0.0)), 3),
            "selection_start": round(float(
                c.get("_selection_start", c.get("start", 0.0))
            ), 3),
            "selection_end": round(float(
                c.get("_selection_end", c.get("end", 0.0))
            ), 3),
            "judge_start": _round_or_none(c.get("judge_start"), 3),
            "judge_end": _round_or_none(c.get("judge_end"), 3),
            "judge_window_changed": bool(c.get("judge_window_changed", False)),
            "judge_grouped": bool(c.get("judge_grouped", False)),
            "judge_group_size": int(c.get("judge_group_size", 0) or 0),
            "judge_group_reason": c.get("judge_group_reason"),
            "boundary_context_applied": bool(c.get("boundary_context_applied")),
            "boundary_context_reason": c.get("boundary_context_reason"),
            "boundary_context_lead_added": _round_or_none(
                c.get("boundary_context_lead_added"), 3,
            ),
            "boundary_context_tail_added": _round_or_none(
                c.get("boundary_context_tail_added"), 3,
            ),
            "peak_timestamp": round(float(c.get("peak_timestamp", 0.0)), 3),
            "peak_value": round(float(c.get("peak_value", 0.0)), 4),
            "selection_score": round(float(c.get("selection_score", 0.0)), 4),
            "signal_selection_score": round(
                float(c.get("signal_selection_score", 0.0) or 0.0), 4,
            ),
            "signal_editorial_score": round(
                float(c.get("signal_editorial_score", 0.0) or 0.0), 4,
            ),
            "ranker_score": (
                float(c["ranker_score"])
                if c.get("ranker_score") is not None else None
            ),
            "ranker_publishability_score": _round_or_none(
                c.get("ranker_publishability_score"), 4,
            ),
            "personal_ranker_score": _round_or_none(
                c.get("personal_ranker_score"), 4,
            ),
            "personal_preference_delta": _round_or_none(
                c.get("personal_preference_delta"), 4,
            ),
            "personal_challenger_score": (
                float(c["personal_challenger_score"])
                if c.get("personal_challenger_score") is not None else None
            ),
            "personal_challenger_delta": (
                float(c["personal_challenger_delta"])
                if c.get("personal_challenger_delta") is not None else None
            ),
            "personal_ranker_percentile": _round_or_none(
                c.get("personal_ranker_percentile"), 4,
            ),
            "signal_percentile": (
                round(float(c["signal_percentile"]), 4)
                if c.get("signal_percentile") is not None else None
            ),
            "ranker_percentile": (
                round(float(c["ranker_percentile"]), 4)
                if c.get("ranker_percentile") is not None else None
            ),
            "ranker_policy": ranker_policy if ranker is not None else "signals",
            "ranker_publishability_floor": (
                round(ranker_publishability_floor, 4) if ranker is not None else None
            ),
            "selection_rank": c.get("selection_rank"),
            "disposition": c.get("selection_disposition", "unknown"),
            "reasons": list(c.get("selection_rejection_reasons") or []),
            "funnel": dict(c.get("selection_funnel") or {}),
            "scene_label": c.get("scene_label"),
            "game_label": c.get("game_label"),
            # Raw score inputs. _score_components reads these off the candidate
            # directly, so a trace without them cannot recompute its own
            # signal score — measured, every one of 651 candidates drifted
            # (game_evidence alone shifted eliminations by ~1.16).
            "game_evidence": _round_or_none(c.get("game_evidence")),
            "game_anchor_time": _round_or_none(c.get("game_anchor_time"), 3),
            "game_profile": c.get("game_profile"),
            "event_density": _round_or_none(c.get("event_density")),
            "crowd_clip": int(c.get("crowd_clip", 0) or 0),
            "creator_protected": bool(c.get("creator_protected")),
            "manual_anchor": bool(c.get("manual_anchor")),
            "recall_marker_ids": list(c.get("recall_marker_ids") or []),
            "recall_marker_time": _round_or_none(c.get("recall_marker_time"), 3),
            "active_channel_count": c.get("active_channel_count"),
            "signal_agreement": _round_or_none(c.get("signal_agreement")),
            "modality_count": int(c.get("modality_count", 0) or 0),
            # compute_hook_score reads these off the candidate too.
            "hook_burst": _round_or_none(c.get("hook_burst")),
            "hook_chat": _round_or_none(c.get("hook_chat")),
            "hook_reaction_energy": _round_or_none(c.get("hook_reaction_energy")),
            # Delivery v7. Persisted so a frozen trace can rebuild the feature
            # vector without the source VOD -- the transcript cache is keyed on
            # the asset file's path+size+mtime, so once a VOD is deleted these
            # are otherwise unrecoverable.
            "speech_rate_delta": _round_or_none(c.get("speech_rate_delta")),
            "exclaim_ratio": _round_or_none(c.get("exclaim_ratio")),
            "semantic_verdict": c.get("semantic_verdict"),
            # Remaining select_clips INPUTS. Enumerated from selection.py rather
            # than guessed: without these an offline replay of this trace
            # diverges (measured — a shipped 11-clip deck replayed as 15,
            # because semantic_moment_type is half of the consensus reject,
            # housekeeping is its own gate, and floor_padded is a score
            # penalty). The trace is only useful if it is a COMPLETE record of
            # what selection saw.
            "semantic_moment_type": c.get("semantic_moment_type"),
            "semantic_hook": _round_or_none(c.get("semantic_hook")),
            "semantic_payoff": _round_or_none(c.get("semantic_payoff")),
            "semantic_self_contained": _round_or_none(c.get("semantic_self_contained")),
            "housekeeping": bool(c.get("housekeeping")),
            "housekeeping_kind": c.get("housekeeping_kind"),
            "terminal_non_content": bool(c.get("terminal_non_content")),
            "terminal_non_content_kind": c.get("terminal_non_content_kind"),
            "terminal_non_content_region": dict(
                c.get("terminal_non_content_region") or {}
            ) or None,
            "moment_token_hashes": list(c.get("moment_token_hashes") or []),
            "moment_group_id": c.get("moment_group_id"),
            "moment_group_source": c.get("moment_group_source"),
            "second_look_keep_probability": _round_or_none(
                c.get("second_look_keep_probability")
            ),
            "second_look_reject_threshold": _round_or_none(
                c.get("second_look_reject_threshold")
            ),
            # Prospective challenger coverage: these rows were not Base
            # Primary, but would enter Primary under one or more shadow-only
            # personal weights.  They must remain distinct from ordinary
            # ceiling candidates so Second Look can retain them beyond its cap.
            "shadow_primary_audit": bool(c.get("shadow_primary_audit")),
            "shadow_primary_weights": [
                float(weight) for weight in c.get("shadow_primary_weights", [])
            ],
            "floor_padded": bool(c.get("floor_padded")),
            "judge_pool_reason": c.get("judge_pool_reason"),
            "visual_pool": bool(c.get("visual_pool")),
            "visual_pool_reason": c.get("visual_pool_reason"),
            "finalist_pool": bool(c.get("finalist_pool")),
            "chat_reaction_only": bool(c.get("chat_reaction_only")),
            "face_arousal_max": round(float(
                (c.get("modality_breakdown") or {}).get(
                    "face_max", c.get("face_arousal_max", 0.0),
                ) or 0.0
            ), 4),
            "calibrated_face_spike": bool(c.get("calibrated_face_spike")),
            "exceptional_face_spike": bool(c.get("exceptional_face_spike")),
            "face_spike_rescue_eligible": bool(c.get("face_spike_rescue_eligible")),
            "face_spike_floor_rescued": bool(c.get("face_spike_floor_rescued")),
            "calibrated_floor_rescued": bool(c.get("calibrated_floor_rescued")),
            "visual_post_grounded": bool(c.get("visual_post_grounded")),
            "visual_verdict": c.get("visual_semantic_verdict"),
            "visual_outcome": c.get("visual_semantic_outcome"),
            "visual_onscreen_text": c.get("visual_semantic_onscreen_text"),
            # Grounded VLM observations are already paid for by the bounded
            # visual-judge pass. Persist the two short descriptions so Stream
            # Memory can search them without decoding frames or rerunning a
            # model after the source VOD is deleted.
            "visual_summary": c.get("visual_semantic_summary"),
            "visual_evidence": c.get("visual_semantic_evidence"),
            "visual_win_anchored": bool(c.get("visual_win_anchored")),
            # Full decomposition, not just the verdict: every content-axis gate
            # (story bypass, chat-reaction demotion, skip veto, anchor cap,
            # shape inference) keys off THESE fields, so a trace carrying only
            # the verdict cannot reproduce its own run offline — measured
            # 2026-07-23, an offline replay of a 9-clip deck returned 15.
            # ~10 extra values per candidate keeps offline A/B possible without
            # re-running the VLM.
            "visual_moment_type": c.get("visual_semantic_moment_type"),
            "visual_model_verdict": c.get("visual_semantic_model_verdict"),
            "visual_confidence": _round_or_none(c.get("visual_semantic_confidence")),
            "visual_payoff": _round_or_none(c.get("visual_semantic_payoff")),
            "visual_self_contained": _round_or_none(c.get("visual_semantic_self_contained")),
            "visual_hook": _round_or_none(c.get("visual_semantic_hook")),
            "visual_routine_only": bool(c.get("visual_semantic_routine_only")),
            "visual_verbal_payoff": bool(c.get("visual_semantic_verbal_payoff")),
            "visual_context_complete": bool(c.get("visual_semantic_context_complete")),
            "visual_event": bool(c.get("visual_semantic_visual_event")),
            "visual_streamer_reaction": bool(c.get("visual_semantic_streamer_reaction")),
            "visual_false_skip_rejudged": bool(c.get("visual_false_skip_rejudged")),
            "content_anchor_capped": bool(c.get("content_anchor_capped")),
            "content_interest": round(float(c.get("content_interest", 0.0) or 0.0), 4),
            "content_anchor": bool(c.get("content_anchor")),
            "semantic_rescue": bool(c.get("semantic_rescue")),
            "strong_startle": bool(c.get("strong_startle")),
            "visible_crowd_reaction": round(
                float(c.get("visible_crowd_reaction", 0.0) or 0.0), 4,
            ),
            "visible_crowd_anchor": bool(c.get("visible_crowd_anchor")),
            # Persisted so the judge-free lobby veto is replayable offline;
            # ocr_context itself is transient and too large for the trace.
            "routine_non_gameplay_ratio": round(
                float(c.get("routine_non_gameplay_ratio", 0.0) or 0.0), 4,
            ),
            # Full learning payload (HUMAN_CLIPS Package 1): the exact feature
            # dict the ranker would score plus the per-channel contributions,
            # so every scan yields offline training/eval rows without
            # re-running perception. 25 floats per candidate keeps the trace
            # cheap enough to persist for every job.
            "features": dict(
                c.get("_selection_feature_snapshot")
                or features_mod.feature_dict(c)
            ),
            "modality_breakdown": {
                str(name): round(float(value), 4)
                for name, value in (c.get("modality_breakdown") or {}).items()
            },
        }
        for c in candidates
    ]
    curve.ranker_diagnostics = {
        **dict(ranker_diagnostics or {}),
        "model_loaded": bool(ranker is not None),
        "ranker_policy": ranker_policy if ranker is not None else "signals",
        "ranker_blend_weight": (
            round(ranker_blend_weight, 4) if ranker is not None else None
        ),
        "personal_ranker_blend_weight": (
            round(personal_ranker_blend_weight, 4) if ranker is not None else None
        ),
        "ranker_publishability_floor": (
            round(ranker_publishability_floor, 4) if ranker is not None else None
        ),
        "ranker_use_personalized_adaptive_floor": (
            ranker_use_personalized_adaptive_floor if ranker is not None else None
        ),
        "content_shape": content_shape_info,
        "content_shape_selection": bool(shape_policy_on),
        "selection_replay_contract": SELECTION_REPLAY_CONTRACT,
        "selector_source_sha256": _selector_source_sha256(),
        # Exact selector inputs for prospective personal-ranker replay.  The
        # candidate artifact is the immutable evidence source; reconstructing
        # these from today's defaults would compare a challenger against a
        # subtly different deck whenever production policy changes later.
        "selection_settings": {
            "sensitivity": float(settings.get("sensitivity", 0.7)),
            "aggressiveness": float(settings.get("aggressiveness", 0.6)),
            "selection_mode": str(selection_mode),
            "max_clips": int(max_clips),
            "game": str(settings.get("game", "generic") or "generic"),
            # Persist the exact frame-curve gate used by select_clips.  The
            # per-candidate vod_median/vod_mad features are intentionally
            # compact/rounded model inputs and cannot reproduce this threshold
            # safely for a prospective counterfactual near the quality bar.
            "quality_bar": float(selection_mod.quality_bar(score, k)),
            "ranker_policy": ranker_policy if ranker is not None else "signals",
            "active_ranker_mode": _active_mode or "signals",
            "ranker_blend_weight": (
                float(ranker_blend_weight) if ranker is not None else None
            ),
            "personal_ranker_blend_weight": (
                float(personal_ranker_blend_weight) if ranker is not None else None
            ),
            "ranker_publishability_floor": (
                float(ranker_publishability_floor) if ranker is not None else None
            ),
            "ranker_use_personalized_adaptive_floor": (
                ranker_use_personalized_adaptive_floor
                if ranker is not None else None
            ),
            "content_shape_selection": bool(shape_policy_on),
        },
        "visual_judge_handoff": {
            "pool_count": len(visual_pool),
            "prepared_window_count": sum(
                c.get("judge_start") is not None and c.get("judge_end") is not None
                for c in candidates
            ),
            "changed_window_count": sum(
                bool(c.get("judge_window_changed")) for c in candidates
            ),
            "pool_reasons": dict(sorted(collections.Counter(
                str(c.get("visual_pool_reason"))
                for c in candidates
                if c.get("visual_pool_reason")
            ).items())),
        },
        "editorial_hygiene": {
            "non_content_regions": [
                asdict(region) for region in non_content_regions
            ],
            "non_content_region_count": len(non_content_regions),
        },
    }
    disposition_counts: Dict[str, int] = {}
    for row in curve.selection_trace:
        disposition = str(row["disposition"])
        disposition_counts[disposition] = disposition_counts.get(disposition, 0) + 1
    print(
        "Selection decisions: "
        + ", ".join(
            f"{name}={count}" for name, count in sorted(disposition_counts.items())
        )
    )

    clips = [
        ReactionClip(
            start=c["start"],
            end=c["end"],
            peak_timestamp=c["peak_timestamp"],
            reason=_build_reason(c.get("game_label"), c["modality_breakdown"],
                                 crowd_clip=int(c.get("crowd_clip", 0) or 0)),
            modality_breakdown=c["modality_breakdown"],
            reaction_auc=c["reaction_auc"],
            score=round(c.get("selection_score", 0.0), 4),
            hook_score=round(c.get("hook_score", 0.0), 4),
            game_label=c.get("game_label"),
            game_evidence=c.get("game_evidence", 0.0),
            scene_label=c.get("scene_label"),
            features=features_mod.feature_dict(c),
            semantic_title=c.get("semantic_title"),
            semantic_hook_line=c.get("semantic_hook_line"),
            semantic_moment_type=c.get("semantic_moment_type"),
            semantic_verdict=c.get("semantic_verdict"),
            creator_protected=bool(c.get("creator_protected")),
            recall_marker_ids=list(c.get("recall_marker_ids") or []),
            recall_marker_time=c.get("recall_marker_time"),
        )
        for c in selected
    ]
    return curve, clips
