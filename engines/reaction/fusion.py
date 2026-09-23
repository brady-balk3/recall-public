# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Reaction fusion (plan §5.5, steps 1-2).

Builds the fused R(t) curve from per-second modality tracks:

  1. per-VOD robust-z normalization of each modality (relative to *this*
     creator's baseline),
  2. weighted fusion with a multiplicative game-evidence confirmation:

        base(t) = w_f*face_z*face_present + w_v*voice_z + w_s*speech_z
        R(t)    = relu(base) * (1 + gamma*game_evidence)

When a modality is absent (no facecam region, no ASR yet), its weight is
dropped and the remaining weights are renormalized per-frame so the curve is
not artificially depressed.
"""

from typing import Callable, Dict, List, Optional

import numpy as np

from core.models.reaction import ReactionFrame, ReactionCurve
from engines.vision.layout_map import routed_text

# Game-evidence tier weights (plan §5.4). OCR amplifies & labels, never sole-triggers.
GAME_TIER_WEIGHTS = {
    "terminal_win": 1.0,
    "elimination": 0.7,
    "knock": 0.5,
    "rank_progress": 0.3,
    "generic_highlight": 0.2,
}

# Default fusion weights (plan §5.5). Tuned in Phase 1, learned in Phase 4.
# "burst" (laughter/scream tagger) is additive on top of the original trio:
# weights renormalize per-frame over the *present* modalities, so when the
# burst track is absent the face/voice/speech proportions are exactly the
# validated 0.40/0.30/0.30.
# "chat" (Twitch chat velocity + emote burst) is additive like "burst": weights
# renormalize per-frame over present modalities, so a VOD without chat keeps the
# validated face/voice/speech proportions exactly. Chat is a strong crowd signal
# (spam/emote spikes track hype), hence a weight between voice/speech and burst.
# "motion" is gameplay frame-diff (facecam masked): lifts silent fights and
# multi-kill chaos even when the streamer stays quiet.
# "game_audio" is the game-side combat-SFX / intense-music track from the
# YamNet pass (audio_events.game_intensity): boss music, gunfire, roars. It is
# the OCR-vocabulary stand-in for games with no HUD banners — a horror boss
# fight registers here while the streamer plays silent and focused. Weighted
# below motion: it confirms the GAME is intense, not that a moment landed.
DEFAULT_WEIGHTS = {
    "face": 0.40,
    "voice": 0.30,
    "speech": 0.30,
    "burst": 0.22,
    "chat": 0.25,
    "motion": 0.20,
    "game_audio": 0.18,
}
DEFAULT_GAMMA = 0.8

# Raw game_intensity peak below this means the VOD has no meaningful game-audio
# track (or the events cache predates the field) -> channel stays inactive and
# the curve is byte-identical to the pre-game-audio behavior.
GAME_AUDIO_MIN_PEAK = 0.08
STARTLE_SYNC_SEC = 3
STARTLE_FACE_DELTA_SCALE = 0.25
STARTLE_MIN_REACTION_SEC = 4
STARTLE_MAX_REACTION_SEC = 15

# How far an OCR trigger's evidence diffuses around its timestamp (seconds).
GAME_DIFFUSE_SEC = 2.0
# Multi-kill density: nearby elim/knock hits boost game_evidence so rapid
# doubles/triples outrank isolated scrap kills.
EVENT_DENSITY_WINDOW_SEC = 15.0
EVENT_DENSITY_LABELS = frozenset({"elimination", "knock", "terminal_win"})
EVENT_DENSITY_BOOST = 0.40


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; edge-safe. window<=1 is a no-op."""
    if window <= 1 or values.size == 0:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    # 'same' length; divide by the actual count of in-bounds taps at the edges
    counts = np.convolve(np.ones_like(values, dtype=np.float64), kernel * window, mode="same")
    smoothed = np.convolve(values.astype(np.float64), kernel * window, mode="same") / np.maximum(counts, 1.0)
    return smoothed.astype(values.dtype)


def robust_z(values: np.ndarray) -> np.ndarray:
    """Per-VOD robust z-score: (x - median) / (1.4826 * MAD), clamped."""
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-6, mad * 1.4826)
    return np.clip((values - median) / scale, -4.0, 8.0)


def _nearest(track: List, timestamp: float, attr: str, tolerance: float = 0.75):
    """Nearest track sample value within ``tolerance`` seconds, else None."""
    best = None
    best_diff = tolerance
    for item in track:
        diff = abs(item.timestamp - timestamp)
        if diff <= best_diff:
            best_diff = diff
            best = item
    return getattr(best, attr) if best is not None else None


def build_game_evidence(unified_signals, classify_ocr: Callable, timeline: List[float]):
    """Diffuse OCR trigger tiers onto the timeline -> (evidence[], label[], density[]).

    ``density[i]`` is 1.0 for isolated events and grows with nearby elim/knock
    hits in a sliding window (multi-kill packages).
    """
    hits = []  # (timestamp, weight, label)
    for sig in unified_signals:
        ocr = getattr(sig, "ocr", None)
        if not ocr or not getattr(ocr, "text", None):
            continue
        # Spatially routed text (chat/alert overlay regions removed) when the
        # pipeline annotated it -- a chat message quoting "eliminated" must not
        # fire a game trigger.
        text = routed_text(ocr)
        if not text:
            continue
        try:
            # Segment-aware classification (plan 22 §4.1): the trigger
            # vocabulary follows the game on screen at this timestamp.
            trigger = classify_ocr(text, sig.timestamp)
        except TypeError:
            # Single-arg classifiers (tests, external callers) still work.
            trigger = classify_ocr(text)
        if not trigger:
            continue
        tier = trigger.get("tier", "generic_highlight")
        hits.append((sig.timestamp, GAME_TIER_WEIGHTS.get(tier, 0.2), tier))

    evidence = np.zeros(len(timeline), dtype=np.float32)
    labels: List[Optional[str]] = [None] * len(timeline)
    density = np.ones(len(timeline), dtype=np.float32)
    if not hits:
        return evidence, labels, density

    # Distinct action hits for density (ignore pure generic spam).
    action_hits = [ht for (ht, _hw, hlabel) in hits if hlabel in EVENT_DENSITY_LABELS]
    half_w = EVENT_DENSITY_WINDOW_SEC / 2.0

    # Each hit only touches timeline frames within GAME_DIFFUSE_SEC, so diffuse
    # per hit over its searchsorted window instead of scanning every hit for
    # every frame. Hit order is preserved: a later hit replaces an earlier one
    # only on a strictly greater value, exactly like the old best_w scan.
    tl = np.asarray(timeline, dtype=np.float64)
    for (ht, hw, hlabel) in hits:
        lo = int(np.searchsorted(tl, ht - GAME_DIFFUSE_SEC, side="left"))
        hi = int(np.searchsorted(tl, ht + GAME_DIFFUSE_SEC, side="right"))
        for i in range(lo, hi):
            val = hw * (1.0 - abs(tl[i] - ht) / GAME_DIFFUSE_SEC)
            if val > evidence[i]:
                evidence[i] = val
                labels[i] = hlabel

    if action_hits:
        action_ts = np.sort(np.asarray(action_hits, dtype=np.float64))
        nearby = (
            np.searchsorted(action_ts, tl + half_w, side="right")
            - np.searchsorted(action_ts, tl - half_w, side="left")
        )
        # 1 isolated kill -> 1.0; double/triple scales up.
        density = (1.0 + EVENT_DENSITY_BOOST * np.maximum(0, nearby - 1)).astype(np.float32)
        boosted = evidence > 0
        evidence[boosted] = np.minimum(1.0, evidence[boosted] * density[boosted])
    return evidence, labels, density


def build_curve(
    unified_signals,
    prosody_frames,
    classify_ocr: Callable,
    face_frames=None,
    asr_hype_frames=None,
    audio_event_frames=None,
    chat_frames=None,
    weights: Optional[Dict[str, float]] = None,
    gamma: float = DEFAULT_GAMMA,
    smooth_window: int = 5,
    return_components: bool = False,
):
    """Fuse all available modality tracks into a per-VOD-normalized ReactionCurve.

    ``return_components=True`` additionally returns the weight-independent
    per-second channel arrays (post-normalization, post-gate) plus the
    active-channel set. Purely observational — nothing downstream consumes
    it and the default path is byte-identical — but tuning/fitting tools
    (fit_fusion_weights.py) need ground-truth internals rather than
    reconstructions from frame fields, which silently lose gate state.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)

    timeline = sorted({round(float(s.timestamp), 3) for s in unified_signals})
    if not timeline:
        return ReactionCurve(fps=1.0, frames=[], score=[])

    n = len(timeline)
    voice_raw = np.zeros(n, dtype=np.float32)
    face_raw = np.zeros(n, dtype=np.float32)
    face_valence = np.zeros(n, dtype=np.float32)
    face_present = np.zeros(n, dtype=bool)
    face_emotion = ["neutral"] * n
    speech_raw = np.zeros(n, dtype=np.float32)
    burst_raw = np.zeros(n, dtype=np.float32)
    loud_onset_raw = np.zeros(n, dtype=np.float32)
    corroborated_loud_onset = np.zeros(n, dtype=np.float32)
    face_startle_evidence = np.zeros(n, dtype=np.float32)
    burst_labels: List[Optional[str]] = [None] * n
    chat_raw = np.zeros(n, dtype=np.float32)
    motion_raw = np.zeros(n, dtype=np.float32)
    game_audio_raw = np.zeros(n, dtype=np.float32)

    has_face_track = bool(face_frames)
    has_speech_track = bool(asr_hype_frames)
    has_burst_track = bool(audio_event_frames)
    has_chat_track = bool(chat_frames)

    # Motion comes from vision perception already on unified_signals.
    motion_by_t = {
        int(round(s.timestamp)): float(getattr(getattr(s, "vision", None), "motion_score", 0.0) or 0.0)
        for s in unified_signals
        if getattr(s, "vision", None) is not None
    }
    has_motion_track = bool(motion_by_t)

    # All tracks are sampled at 1 fps from t=0,1,2,..., so they align exactly on
    # the integer-second key -> O(1) lookup instead of O(n) nearest scans.
    prosody_by_t = {int(round(f.timestamp)): f for f in (prosody_frames or [])}
    face_by_t = {int(round(f.timestamp)): f for f in (face_frames or [])}
    speech_by_t = {int(round(f.timestamp)): f for f in (asr_hype_frames or [])}
    burst_by_t = {int(round(f.timestamp)): f for f in (audio_event_frames or [])}
    chat_by_t = {int(round(f.timestamp)): f for f in (chat_frames or [])}

    for i, t in enumerate(timeline):
        key = int(round(t))
        pf = prosody_by_t.get(key)
        voice_raw[i] = float(pf.voice_arousal) if pf is not None else 0.0

        if has_face_track:
            ff = face_by_t.get(key)
            if ff is not None and ff.face_present:
                face_present[i] = True
                face_raw[i] = float(ff.face_arousal)
                face_valence[i] = float(ff.face_valence)
                face_emotion[i] = ff.face_emotion

        if has_speech_track:
            sf = speech_by_t.get(key)
            speech_raw[i] = float(sf.speech_hype) if sf is not None else 0.0

        if has_burst_track:
            bf = burst_by_t.get(key)
            if bf is not None:
                tagger_burst = float(bf.burst)
                loud_onset = float(getattr(bf, "loud_onset", 0.0) or 0.0)
                loud_onset_raw[i] = loud_onset
                burst_raw[i] = max(tagger_burst, loud_onset)
                burst_labels[i] = (
                    "loud_onset"
                    if loud_onset > tagger_burst and loud_onset > 0.05
                    else bf.label
                )
                # Same tagger pass; frames cached before the field decode to 0.
                game_audio_raw[i] = float(getattr(bf, "game_intensity", 0.0) or 0.0)

        if has_chat_track:
            cf = chat_by_t.get(key)
            if cf is not None:
                chat_raw[i] = float(cf.chat)

        if has_motion_track:
            motion_raw[i] = float(motion_by_t.get(key, 0.0))

    game_evidence, game_labels, event_density = build_game_evidence(
        unified_signals, classify_ocr, timeline,
    )

    # Startle is a transition, not a static Fear label. Compute the unsmoothed
    # face jump before the normal reaction envelope blurs it, then synchronize
    # it with the acoustic onset after the human correlation gate below.
    if face_present.any():
        for idx in range(n):
            if (
                not face_present[idx]
                or str(face_emotion[idx]).strip().lower() not in {"fear", "surprise"}
            ):
                continue
            prior = face_raw[max(0, idx - STARTLE_SYNC_SEC):idx]
            prior_present = face_present[max(0, idx - STARTLE_SYNC_SEC):idx]
            if prior.size == 0 or not prior_present.any():
                continue
            baseline = float(np.median(prior[prior_present]))
            delta = max(0.0, float(face_raw[idx]) - baseline)
            face_startle_evidence[idx] = min(1.0, delta / STARTLE_FACE_DELTA_SCALE)

    # Temporal smoothing: a genuine reaction lasts several seconds, so collapse
    # per-second arousal noise into multi-second envelopes before normalizing.
    # Without this, R(t) is a wall of 1s spikes and no clean arcs form.
    voice_raw = _smooth(voice_raw, smooth_window)
    if face_present.any():
        face_raw = _smooth(face_raw, smooth_window)
    if has_chat_track:
        # Chat velocity is an envelope like voice arousal — smooth before norm.
        chat_raw = _smooth(chat_raw, smooth_window)
    if has_motion_track:
        motion_raw = _smooth(motion_raw, smooth_window)
    # A real encounter sustains for many seconds (boss music, repeated combat
    # SFX), so the game-audio track is an envelope like chat/motion. Presence
    # is decided on the raw pre-smooth peak: caches written before the field
    # decode to all-zeros and must leave the channel (and the curve) untouched.
    has_game_audio_track = bool(has_burst_track and float(np.max(game_audio_raw) if n else 0.0) > GAME_AUDIO_MIN_PEAK)
    if has_game_audio_track:
        game_audio_raw = _smooth(game_audio_raw, smooth_window)

    # Per-VOD normalization.
    voice_z = robust_z(voice_raw)
    face_z = robust_z(face_raw[face_present]) if face_present.any() else np.zeros(n)
    if face_present.any():
        # scatter the normalized values back to full-length, 0 where no face
        fz_full = np.zeros(n, dtype=np.float32)
        fz_full[face_present] = face_z
        face_z = fz_full
    speech_z = robust_z(speech_raw) if has_speech_track else np.zeros(n)
    # Burst is mostly zeros, so robust_z degenerates (MAD 0 turns any nonzero
    # into a max spike). And in mixed game audio the tagger's sigmoids are
    # depressed — a genuine laugh that tops all 521 classes can still score
    # only ~0.2 raw. So normalize per-VOD: noise floor, then scale by this
    # VOD's own near-max burst (min scale keeps a laugh-free VOD at ~zero).
    if has_burst_track:
        _floor = 0.05
        _peak = float(np.percentile(burst_raw, 99.9)) if n else 0.0
        _scale = max(0.15, _peak - _floor)
        burst_z = (np.clip((burst_raw - _floor) / _scale, 0.0, 1.0) * 4.0).astype(np.float32)
        # Bursts are 1-2s point events, not envelopes — mean-smoothing the raw
        # sparse track would dilute a genuine laugh below the noise floor. So
        # scale first (above, from raw values), then dilate ±1s and lightly
        # smooth so boundary snapping still finds an arc.
        if n > 2:
            dilated = burst_z.copy()
            dilated[1:] = np.maximum(dilated[1:], burst_z[:-1])
            dilated[:-1] = np.maximum(dilated[:-1], burst_z[1:])
            burst_z = _smooth(dilated, 3)

        # Correlation gate (plan 22 §4.4): the tagger hears the MIX, so an
        # in-game scream (horror stingers, NPC laughter) fires "burst" while
        # the streamer sits silent. A genuine human burst co-registers on a
        # human channel — laughter raises voice arousal; a facecam reaction
        # shows up as arousal. Require agreement within ±2s from voice or
        # face before burst contributes. Voice/prosody runs on every VOD, so
        # the gate always has a channel to check.
        agree = voice_z > 0.3
        if face_present.any():
            agree = agree | (face_present & (face_raw >= 0.12))
        widened = agree.copy()
        for shift in (1, 2):
            widened[shift:] |= agree[:-shift]
            widened[:-shift] |= agree[shift:]
        burst_z = np.where(widened, burst_z, 0.0)
        # Selection may reward a startle edge, but only after the same human
        # correlation gate that controls its contribution to R(t). Storing raw
        # game-only onsets here would let gunshots collect a ranking bonus.
        corroborated_loud_onset = np.where(
            widened, loud_onset_raw, 0.0,
        ).astype(np.float32)
    else:
        burst_z = np.zeros(n)

    startle_raw = np.zeros(n, dtype=np.float32)
    if face_startle_evidence.any() and corroborated_loud_onset.any():
        for idx in np.where(face_startle_evidence > 0.0)[0]:
            # A real visual startle persists for several frames after the
            # transition. Follow the contiguous Fear/Surprise run and look for
            # the acoustic edge inside it; one-frame FER noise cannot qualify.
            end = int(idx)
            limit = min(n, int(idx) + STARTLE_MAX_REACTION_SEC)
            while (
                end < limit
                and face_present[end]
                and str(face_emotion[end]).strip().lower() in {"fear", "surprise"}
            ):
                end += 1
            run = end - int(idx)
            if run < STARTLE_MIN_REACTION_SEC:
                continue
            lo = max(0, int(idx) - STARTLE_SYNC_SEC)
            hi = min(n, end + STARTLE_SYNC_SEC)
            onset = float(np.max(corroborated_loud_onset[lo:hi]))
            persistence = min(1.0, run / 6.0)
            startle_raw[idx] = (
                float(face_startle_evidence[idx]) * onset * persistence
            )

    motion_z = robust_z(motion_raw) if has_motion_track else np.zeros(n)

    # Game audio shares burst's normalization problem: quiet exploration keeps
    # the median/MAD near zero (robust_z degenerates), and the tagger's
    # sigmoids are depressed by the speech-heavy mix. Noise-floor it, scale by
    # this VOD's own near-max, and let a combat-free VOD stay at ~zero via the
    # minimum scale. No human-agreement gate — this track exists precisely for
    # the moments where the streamer goes quiet; false-fire risk is bounded by
    # the low fusion weight and selection's ordinary gates.
    if has_game_audio_track:
        _ga_floor = 0.04
        _ga_peak = float(np.percentile(game_audio_raw, 99.5)) if n else 0.0
        _ga_scale = max(0.15, _ga_peak - _ga_floor)
        game_audio_z = (np.clip((game_audio_raw - _ga_floor) / _ga_scale, 0.0, 1.0) * 4.0).astype(np.float32)
    else:
        game_audio_z = np.zeros(n)

    # Chat velocity: continuous rate signal, normalize per-VOD like voice.
    chat_z = robust_z(chat_raw) if has_chat_track else np.zeros(n)
    # Chat crowd-truth gate: pure chat spam with zero human reaction is often
    # bot/emote spam. Require voice or face agreement within ±2s (same idea as
    # burst) so chat still lifts real hype without inventing moments alone.
    if has_chat_track:
        human = voice_z > 0.25
        if face_present.any():
            human = human | (face_present & (face_raw >= 0.10))
        if has_motion_track:
            # Gameplay chaos co-occurring with chat is also real crowd signal.
            human = human | (motion_z > 0.8)
        widened = human.copy()
        for shift in (1, 2):
            widened[shift:] |= human[:-shift]
            widened[:-shift] |= human[shift:]
        # Soft gate: keep a floor so huge chat still contributes lightly.
        chat_z = np.where(widened, chat_z, chat_z * 0.25)

    # Which modalities exist at all in this VOD.
    global_active = {"voice"}
    if has_face_track and face_present.any():
        global_active.add("face")
    if has_speech_track:
        global_active.add("speech")
    if has_burst_track:
        global_active.add("burst")
    if has_chat_track:
        global_active.add("chat")
    if has_motion_track and float(np.max(motion_raw) if n else 0.0) > 0.02:
        global_active.add("motion")
    if has_game_audio_track:
        global_active.add("game_audio")

    # Per-frame the present-modality set only varies by face presence (voice is
    # always on; speech/burst/chat/motion/game_audio are global per-VOD), so the
    # weight renormalization has exactly two cases — blend both and select per frame.
    tracks = {"voice": voice_z, "face": face_z, "speech": speech_z,
              "burst": burst_z, "chat": chat_z, "motion": motion_z,
              "game_audio": game_audio_z}

    def _blend(mods):
        total = sum(weights.get(m, 0.0) for m in mods)
        if total <= 0:
            return np.zeros(n, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        for m in mods:
            out += (weights.get(m, 0.0) / total) * tracks[m]
        return out

    base_mods = ["voice"] + [m for m in ("speech", "burst", "chat", "motion", "game_audio")
                             if m in global_active]
    base = _blend(base_mods)
    if "face" in global_active:
        base = np.where(face_present, _blend(base_mods + ["face"]), base)
    base = np.maximum(0.0, base)  # only above-baseline excitement contributes
    score = (base * (1.0 + gamma * game_evidence.astype(np.float64))).astype(np.float32)

    frames = [
        ReactionFrame(
            timestamp=timeline[i],
            face_present=bool(face_present[i]),
            face_arousal=float(face_raw[i]),
            face_valence=float(face_valence[i]),
            face_emotion=face_emotion[i],
            voice_arousal=float(voice_raw[i]),
            speech_hype=float(speech_raw[i]),
            burst=float(burst_raw[i]),
            burst_label=burst_labels[i],
            loud_onset=float(corroborated_loud_onset[i]),
            startle=float(startle_raw[i]),
            chat=float(chat_raw[i]),
            motion=float(motion_raw[i]),
            game_intensity=float(game_audio_raw[i]),
            game_evidence=float(game_evidence[i]),
            game_label=game_labels[i],
        )
        for i in range(n)
    ]
    curve = ReactionCurve(fps=1.0, frames=frames, score=[float(x) for x in score])
    # Stash per-second density for candidate scoring (not part of ReactionFrame
    # schema so ranker feature vectors stay version-stable).
    curve.event_density = [float(x) for x in event_density]  # type: ignore[attr-defined]
    if return_components:
        return curve, {
            "voice_z": voice_z, "face_z": face_z, "speech_z": speech_z,
            "burst_z": burst_z, "chat_z": chat_z, "motion_z": motion_z,
            "game_audio_z": game_audio_z,
            "face_present": face_present,
            "game_evidence": game_evidence,
            "global_active": set(global_active),
        }
    return curve
