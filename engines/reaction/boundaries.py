# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Arc/peak detection + boundary snapping (plan §5.5, steps 3-4).

A contiguous run of R(t) above ``theta`` is one *arc* (a multi-kill streak
becomes a single elevated arc with sub-peaks -> one clip, fixing "missed the
next 3 kills"). Boundaries snap to reaction onset/settle and then to the
nearest audio-silence gap, fixing early/late cuts.

Clip length is an OUTPUT of the content, not a setting: the arc's own
onset-expanded extent (where the reaction built up and where it subsided)
defines the window. HARD_MIN/HARD_MAX below are engineering invariants, not
user knobs -- a healthy signal path should almost never touch them. Hitting
HARD_MIN means the candidate had no real arc around its peak (the window is
extended to a natural audio boundary and flagged ``floor_padded`` so selection
can treat the padding itself as weak-moment evidence). Arcs longer than
HARD_MAX are segmented at their internal lulls rather than trimmed blind.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Peak-relative floors only: every clip keeps at least this much room around
# its peak so a payoff is never the literal first/last frame.
LEAD_IN_MIN = 4.5
TAIL_MIN = 2.0

# Elim/knock banners need combat setup *before* OCR lag and a short celebration
# after. OCR "ELIMINATED" trails the kill by ~1-2s; LEAD_IN_MIN alone often
# opens mid-fight. EVENT_LEAD_IN reaches back for the aim/push that caused the
# banner. Asymmetric tails: once voice settles after the payoff, cut soon
# rather than holding EVENT_TAIL of quiet looting.
# terminal_win has dedicated lead/celebration handling downstream
# (reaction_main._merge_win_clips) and isn't affected by this.
EVENT_TAIL_LABELS = {"elimination", "knock"}
EVENT_LEAD_IN = 7.5
EVENT_TAIL = 5.0
EVENT_SETTLE_VOICE = 0.15
EVENT_SETTLE_PAD = 2.0
# Prefer postable short-form packages for event-anchored clips when the arc
# is longer than this; HARD_MAX still applies as the absolute ceiling.
EVENT_PREFERRED_MAX = 42.0

# Breathing room beyond the arc's own onset/settle extent (seconds).
LEAD_PAD = 2.5
TAIL_PAD = 3.0

SNAP_WINDOW_SEC = 2.0  # +/- window to search for an audio-silence gap

# Sentence-boundary snapping (HUMAN_CLIPS plan, Package B). Wider than
# SNAP_WINDOW_SEC because sentence boundaries are sparser than quiet frames:
# a start may sit a beat before the arc onset, and an end may need to run a
# few seconds past the settle point to finish the sentence rather than cut a
# word. Silence/cut snapping remains the fallback whenever no sentence
# boundary qualifies, so a missing transcript changes nothing.
# DO NOT widen this to "finish the sentence more often" -- measured and
# rejected 2026-07-26. Only 63.8% of winners end on a sentence boundary, and
# the misses are not a tuning fault: 35.9% of clips have NO sentence end inside
# the window, the next one a median 6.5s away. (Other suspects ruled out with
# the same instrumentation: trim_dead_air broke 0 of 183 good ends and 4 of 211
# good starts; the duration-revert branch fired twice in 287 calls.) So
# widening is the only lever -- and every widening trades iou for endings:
#
#   snap   ends on boundary   starts   median dur   mean_iou   golden gate
#    3.5        63.8%          73.5%      25.1s      0.4851    green
#    5.0        71.8%          75.6%      25.3s      0.4811    RED (3 fixtures)
#    7.0        79.8%          78.4%      28.0s      0.4646    —
#    9.0        85.7%          80.8%      29.7s      0.4637    —
#
# Recall and precision are identical at every setting (presentation-only), but
# even 5.0 regresses per-fixture mean_iou past tolerance -- 1.000 -> 0.927 on
# a reviewed recording. The reference windows are human cuts, and this is
# them saying creators cut tight rather than waiting for the sentence to land.
# A clip that ends mid-utterance is therefore not automatically wrong; fixing
# the ones that genuinely lose a punchline needs evidence a payoff is still
# coming (laughter/burst after the cut, a resolving phrase), not a wider window.
# Tried that next: extend_payoff_tail on winners, only when RMS/voice continued
# in (end, end+5s] (the median creator end-edit). Measured 2026-08-17 and
# rejected. Payoff_inclusion 0.801 -> 0.826 (6 fixtures up, 3 down) but mean_iou
# 0.666 -> 0.596, golden RED on 37 fixtures. Streamer speech is still going
# after almost every cut, so the energy gate is a de facto TAIL_PAD raise.
# Do not retry without a signal that is rare on already-correct ends.
SENTENCE_SNAP_SEC = 3.5
# When no sentence start sits within the symmetric window, a clip would open
# mid-phrase. Rather than accept that, look further FORWARD for the next clean
# start -- but only this far, and never past the payoff's lead-in (below). A
# boundary further away than this means a run-on utterance with no natural cut,
# where opening mid-phrase is genuinely unavoidable; snapping to the distant
# boundary there would instead swallow the whole setup.
SENTENCE_START_FORWARD_MAX = 7.0
# NB: no symmetric END extension. Measured on the golden set, forcing clips to
# run on to a sentence END regressed mean_iou on every truth-matching fixture:
# creators cut tight on the PAYOFF, not on grammatical completion, so a clean
# start earns quality but a "finished sentence" end fights the human boundary.
# Breathing room around the snapped word timestamps: open a beat before the
# sentence's first word, close a beat after its last -- a cut exactly on the
# phoneme boundary reads clipped.
SENTENCE_PREROLL = 0.25
SENTENCE_POSTPAD = 0.35

# Conversational context recovery is deliberately downstream of selection and
# much narrower than global padding. It activates only when the text judge says
# the delivered beat is not self-contained while the visual judge called the
# same window context-complete. That disagreement is the evidence that the
# visual read likely borrowed meaning from surrounding transcript or frames.
CONTEXT_SELF_CONTAINED_MAX = 0.80
CONTEXT_VISUAL_PAYOFF_MIN = 0.60
CONTEXT_SEMANTIC_PAYOFF_MAX = 0.50
CONTEXT_LEAD_MAX = 12.0
CONTEXT_TAIL_MAX = 12.0
CONTEXT_BRIDGE_GAP = 4.0
CONTEXT_SECOND_SENTENCE_GAP = 2.5
CONTEXT_TAIL_SENTENCE_MAX = 3

# Wide sanity invariants (see module docstring). Not user settings.
HARD_MIN = 8.0
HARD_MAX = 90.0


@dataclass
class Arc:
    start_idx: int
    end_idx: int
    peak_idx: int
    peak_value: float


def detect_arcs(score: np.ndarray, theta: float, onset_theta: float) -> List[Arc]:
    """Find contiguous supra-threshold regions, expanded down to the onset level."""
    n = score.size
    arcs: List[Arc] = []
    i = 0
    above = score > theta
    while i < n:
        if not above[i]:
            i += 1
            continue
        # core region above theta
        j = i
        while j < n and above[j]:
            j += 1
        core_start, core_end = i, j - 1
        peak_idx = core_start + int(np.argmax(score[core_start:core_end + 1]))

        # expand outward down to the onset threshold
        s = core_start
        while s > 0 and score[s - 1] > onset_theta:
            s -= 1
        e = core_end
        while e < n - 1 and score[e + 1] > onset_theta:
            e += 1

        arcs.append(Arc(start_idx=s, end_idx=e, peak_idx=peak_idx,
                        peak_value=float(score[peak_idx])))
        i = j
    return arcs


def _split_arc(arc: Arc, score: np.ndarray) -> List[Arc]:
    """Split one arc at its deepest internal lull into two sub-arcs."""
    lo, hi = arc.start_idx, arc.end_idx
    if hi - lo < 3:
        return [arc]
    # Deepest valley strictly inside the arc: the most natural cut point.
    interior = score[lo + 1:hi]
    valley = lo + 1 + int(np.argmin(interior))
    halves = []
    for s, e in ((lo, valley), (valley + 1, hi)):
        if e < s:
            continue
        peak = s + int(np.argmax(score[s:e + 1]))
        halves.append(Arc(start_idx=s, end_idx=e, peak_idx=peak,
                          peak_value=float(score[peak])))
    return halves if len(halves) == 2 else [arc]


def segment_arcs(
    arcs: List[Arc],
    timeline: np.ndarray,
    score: np.ndarray,
    max_len: float = HARD_MAX,
) -> List[Arc]:
    """Segment runaway arcs at internal lulls until each fits within max_len.

    An 8-minute sustained-hype run isn't one clip; each lull-bounded stretch
    becomes its own candidate and competes on its own merits in selection.
    """
    out: List[Arc] = []
    queue = list(arcs)
    # This loop is finite without a global iteration guard: every successful
    # split produces strictly smaller, non-overlapping index ranges, and an
    # unsplittable range is emitted immediately.  The old ``guard < 512`` was
    # applied to the entire VOD queue, so signal-rich long streams silently
    # dropped every arc after queue item 512.  In a 4-hour fixture that erased
    # the final third, including the strongest reaction in the whole VOD.
    while queue:
        arc = queue.pop(0)
        span = float(timeline[arc.end_idx] - timeline[arc.start_idx])
        if span <= max_len or arc.end_idx - arc.start_idx < 3:
            out.append(arc)
            continue
        halves = _split_arc(arc, score)
        if len(halves) == 1:
            out.append(arc)
        else:
            queue.extend(halves)
    out.sort(key=lambda a: a.start_idx)
    return out


# A candidate boundary within this fraction of the quietest frame's RMS is
# "quiet enough" — among those, prefer one adjacent to a visual cut.
CUT_TIE_RMS_FACTOR = 1.25


def detect_cuts(motion: np.ndarray, z_threshold: float = 2.5) -> np.ndarray:
    """Shot-change likelihood per frame from the 1 fps motion track (plan 22 §4.2).

    A hard camera cut / scene switch produces a frame-difference outlier far
    above gameplay's motion baseline. Robust-z over the motion track marks
    them — no new model, no extra frame reads. Returns a boolean cut mask.
    """
    if motion is None or motion.size == 0:
        return np.zeros(0, dtype=bool)
    median = float(np.median(motion))
    mad = float(np.median(np.abs(motion - median))) * 1.4826
    # A near-constant baseline (static menu screen, letterboxed replay) makes
    # MAD collapse to ~0 and would hide genuine cuts — floor the scale at a
    # few percent of the baseline so a real jump still registers, while a
    # perfectly flat track (all deviations 0) still yields no cuts.
    scale = max(mad, 0.05 * abs(median), 1e-6)
    return (motion - median) / scale > z_threshold


def _snap_to_silence(target_time: float, timeline: np.ndarray,
                     rms: Optional[np.ndarray],
                     cuts: Optional[np.ndarray] = None) -> float:
    """Snap a boundary to a quiet frame within +/- window, preferring one at a
    visual cut. Cut-adjacency only breaks near-ties between comparably quiet
    frames (plan 22 §4.2) — it never drags the boundary somewhere loud, and a
    clip must never open mid-camera-cut when a clean quiet cut sits nearby."""
    if rms is None:
        return target_time
    lo = target_time - SNAP_WINDOW_SEC
    hi = target_time + SNAP_WINDOW_SEC
    mask = (timeline >= lo) & (timeline <= hi)
    if not mask.any():
        return target_time
    idxs = np.where(mask)[0]
    quietest = idxs[int(np.argmin(rms[idxs]))]
    if cuts is None or cuts.size != timeline.size:
        return float(timeline[quietest])
    quiet_floor = float(rms[quietest]) * CUT_TIE_RMS_FACTOR + 1e-9
    candidates = [i for i in idxs if rms[i] <= quiet_floor and cuts[i]]
    if candidates:
        # Nearest cut among the comparably quiet frames.
        best = min(candidates, key=lambda i: abs(timeline[i] - target_time))
        return float(timeline[best])
    return float(timeline[quietest])


def _snap_to_sentence(target_time: float,
                      sentence_times: Optional[np.ndarray],
                      forward_only: bool = False) -> Optional[float]:
    """Nearest sentence boundary to the target within the snap window.

    Sentence times come from word-level ASR timestamps
    (engines/caption/sentence_index.py). No loudness test: a sentence START
    is by definition a speech onset -- the frame carrying the first word can
    never be as quiet as a mid-silence frame, so gating on RMS (tried first)
    rejected essentially every real boundary. The word timestamp itself is
    the evidence a human cut point exists here; the caller pads it
    (SENTENCE_PREROLL/SENTENCE_POSTPAD) so the cut breathes rather than
    clipping the first phoneme.
    ``forward_only`` searches [target, target + window] instead of +/- window:
    ends may run slightly long to finish a sentence but never pull backward
    into it. Returns None when no boundary is in the window; the caller falls
    back to the silence/cut snap, keeping the no-transcript path byte-identical.
    """
    if sentence_times is None:
        return None
    times = np.asarray(sentence_times, dtype=np.float64)
    if times.size == 0:
        return None
    lo = target_time if forward_only else target_time - SENTENCE_SNAP_SEC
    hi = target_time + SENTENCE_SNAP_SEC
    cands = times[(times >= lo) & (times <= hi)]
    if cands.size == 0:
        return None
    return float(cands[int(np.argmin(np.abs(cands - target_time)))])


def refine_to_sentences(
    bounds: dict,
    sentence_starts: Optional[np.ndarray],
    sentence_ends: Optional[np.ndarray],
    anchor_time: Optional[float] = None,
    anchor_label: Optional[str] = None,
    clip_min: float = HARD_MIN,
    clip_max: float = HARD_MAX,
) -> dict:
    """Presentation-only sentence refinement of an already-SELECTED window.

    Selection must keep judging candidates on the silence-snapped windows its
    gates were tuned against — snapping every candidate to sentences before
    selection made junk chatter look speech-dense (dead-air gate passes) while
    diluting event clips' mean intensity (routine-kill gate fires): the whole
    deck changed. A human editor picks the moment first and THEN cuts on the
    sentence, so this runs after select_clips, on winners only.

    Start moves to the nearest sentence start within +/-SENTENCE_SNAP_SEC
    (minus PREROLL breathing room); end to the nearest sentence end in a
    forward-biased window (plus POSTPAD) so a clip finishes its last sentence
    rather than cutting a word. Every invariant is re-applied afterwards —
    peak floors, anchor containment, min/max duration — and any violation
    reverts to the original window: a presentation tweak may never break the
    moment. Returns {"start", "end"}; features/auc deliberately keep the
    selected window's values (they describe the moment that won selection).
    """
    start = float(bounds["start"])
    end = float(bounds["end"])
    peak_time = float(bounds.get("peak_timestamp", start))
    orig = {"start": start, "end": end}

    s = _snap_to_sentence(start + SENTENCE_PREROLL, sentence_starts)
    if s is None and sentence_starts is not None:
        # Nothing within the symmetric window -> the clip would open mid-phrase.
        # Reach forward to the next sentence start, bounded so we never eat into
        # the payoff's lead-in and never chase a boundary so distant it swallows
        # the setup (a run-on utterance is then left mid-phrase, unavoidably).
        times = np.asarray(sentence_starts, dtype=np.float64)
        forward_limit = min(
            start + SENTENCE_START_FORWARD_MAX, peak_time - LEAD_IN_MIN,
        )
        forward = times[(times >= start) & (times <= forward_limit)]
        if forward.size:
            s = float(forward[0])
    if s is not None:
        start = s - SENTENCE_PREROLL
    # Forward-biased end: finishing the sentence beats trimming into it, but a
    # sentence that ended a beat before the current end is also a clean out.
    e = _snap_to_sentence(end - SENTENCE_POSTPAD, sentence_ends, forward_only=True)
    if e is None:
        e = _snap_to_sentence(end - 1.5, sentence_ends, forward_only=True)
    if e is not None:
        end = e + SENTENCE_POSTPAD

    # Re-apply the window invariants (same rules as snap_boundaries).
    start = min(start, peak_time - LEAD_IN_MIN)
    end = max(end, peak_time + TAIL_MIN)
    if anchor_time is not None:
        lead_floor = EVENT_LEAD_IN if anchor_label in EVENT_TAIL_LABELS else LEAD_IN_MIN
        start = min(start, float(anchor_time) - lead_floor)
        end = max(end, float(anchor_time) + TAIL_MIN)
    start = max(0.0, start)

    duration = end - start
    if duration < clip_min or duration > clip_max:
        return orig
    return {"start": round(start, 3), "end": round(end, 3)}


def _visual_moment_type(c: dict) -> str:
    raw = c.get("visual_semantic_moment_type")
    if raw is None:
        raw = c.get("visual_moment_type")
    if raw is None:
        raw = c.get("semantic_moment_type")
    return str(raw or "").strip().lower()


# 2026-08-21 blinded A/B (9 moved moments, all scare-end extensions):
# creator kept the ORIGINAL cut in 4/7 decisive picks vs 3 for widened
# (win rate 0.43 [0.06, 0.80]). DB recuts say creators widen scare ends,
# but machine-side widening is not preferred, so the trigger ships OFF.
# Re-enable only when a label round shows engine widening matches where
# humans actually cut. Tests patch this True to exercise the mechanism.
MOMENT_TYPE_EXPANSION_ENABLED = False


def conversational_expansion_policy(candidate: dict) -> dict:
    """Return evidence-backed lead/tail expansion flags for one winner.

    This is presentation policy, never an admission or ranking feature. A low
    text self-contained score plus an optimistic visual context read is a
    targeted signal that a cold viewer may be entering mid-story. Setup and
    payoff are decided independently: creator edits showed a real rage clip
    needed only its preceding question, while a visually postable joke needed
    only its delayed answer/tag. Keeping those paths separate avoids adding a
    generic lead merely because the text editor disliked the ending.

    Moment-type triggers (2026-08-21, probe_boundary_heterogeneity.py): recut
    rates among kept clips are elevated for rage (opens earlier, median -10 s)
    and scare (closes later, median +6.5 s) while fail is never recut
    (0/298), so a judge-authored rage/scare label turns the corresponding
    flag on by itself. Expansion stays sentence-bounded downstream -- the
    trigger widens the *permission*, not the cut.

    DEFAULT OFF (2026-08-21 blinded A/B, scripts/build_boundary_ab_review.py):
    9 moved moments, all scare-end extensions; creator picked the ORIGINAL cut
    in 4 of 7 decisive comparisons vs 3 for widened (win rate 0.43 [0.06,
    0.80] -- a coin flip leaning harmful). Creator recuts widen scare ends,
    but machine extension lands differently and is not preferred. Re-enable
    only with label-round evidence that engine-side widening matches where
    humans actually cut.
    """
    semantic_self = candidate.get("semantic_self_contained")
    visual_context = candidate.get(
        "visual_semantic_context_complete",
        candidate.get("visual_context_complete", False),
    )
    try:
        context_disagreement = (
            semantic_self is not None
            and float(semantic_self) < CONTEXT_SELF_CONTAINED_MAX
            and bool(visual_context)
        )
    except (TypeError, ValueError):
        context_disagreement = False
    semantic_verdict = str(candidate.get("semantic_verdict") or "").strip().lower()
    semantic_type = str(candidate.get("semantic_moment_type") or "").strip().lower()
    lead = bool(
        context_disagreement
        and semantic_verdict == "post"
        and semantic_type not in ("", "filler")
    )
    tail = False

    # Judge-labelled rant/scare triggers: independent of the disagreement
    # paths above so one signal's absence cannot veto the other's evidence.
    # Default OFF -- blinded A/B leaned harmful (3/7 vs control, see
    # MOMENT_TYPE_EXPANSION_ENABLED below for numbers).
    moment_type_trigger = None
    if not MOMENT_TYPE_EXPANSION_ENABLED:
        judged_types = set()
    else:
        judged_types = {
            semantic_type,
            _visual_moment_type(candidate),
        }
    if semantic_verdict in ("", "post", "maybe"):
        if "rage" in judged_types:
            moment_type_trigger = "lead"
        elif "scare" in judged_types:
            moment_type_trigger = "tail"
    if moment_type_trigger == "lead":
        lead = True
    elif moment_type_trigger == "tail":
        tail = True

    semantic_payoff = candidate.get("semantic_payoff")
    visual_payoff = candidate.get(
        "visual_semantic_payoff",
        candidate.get("visual_payoff"),
    )
    try:
        tail = tail or (
            bool(visual_context)
            and semantic_payoff is not None
            and visual_payoff is not None
            and float(semantic_payoff) < CONTEXT_SEMANTIC_PAYOFF_MAX
            and float(visual_payoff) >= CONTEXT_VISUAL_PAYOFF_MIN
        )
    except (TypeError, ValueError):
        pass

    if moment_type_trigger == "lead" and not context_disagreement:
        reason = "moment_type_rage_setup"
    elif moment_type_trigger == "tail" and not (
        bool(visual_context)
        and semantic_payoff is not None
        and visual_payoff is not None
        and float(semantic_payoff) < CONTEXT_SEMANTIC_PAYOFF_MAX
        and float(visual_payoff) >= CONTEXT_VISUAL_PAYOFF_MIN
    ):
        reason = "moment_type_scare_aftermath"
    else:
        reason = (
            "judge_setup_and_payoff_disagreement" if lead and tail
            else "judge_setup_disagreement" if lead
            else "judge_payoff_disagreement" if tail
            else None
        )
    return {
        "lead": lead,
        "tail": tail,
        "reason": reason,
    }


def expand_conversational_context(
    bounds: dict,
    sentence_starts: Optional[np.ndarray],
    sentence_ends: Optional[np.ndarray],
    timeline: np.ndarray,
    rms: Optional[np.ndarray],
    *,
    expand_lead: bool,
    expand_tail: bool,
    clip_max: float = HARD_MAX,
) -> dict:
    """Expand a selected cut to adjacent conversational units.

    The original window always remains intact. Lead recovery reaches the
    sentence containing the cut (or the immediately preceding sentence across
    a short pause), then snaps into its preceding quiet beat. Tail recovery may
    include two closely connected sentences so a joke's tag is not mistaken
    for disposable post-payoff chatter. Missing transcript is a strict no-op.
    """
    start = float(bounds["start"])
    end = float(bounds["end"])
    starts = np.asarray(sentence_starts if sentence_starts is not None else [], dtype=np.float64)
    ends = np.asarray(sentence_ends if sentence_ends is not None else [], dtype=np.float64)
    count = min(starts.size, ends.size)
    if count == 0 or (not expand_lead and not expand_tail):
        return {"start": start, "end": end, "lead_added": 0.0, "tail_added": 0.0}
    starts, ends = starts[:count], ends[:count]

    new_start = start
    new_end = end
    if expand_lead:
        prior = np.flatnonzero(starts < start - 0.05)
        if prior.size:
            idx = int(prior[-1])
            contains_cut = ends[idx] >= start - 0.05
            bridges_cut = 0.0 <= start - ends[idx] <= CONTEXT_BRIDGE_GAP
            target = float(starts[idx])
            if (contains_cut or bridges_cut) and start - target <= CONTEXT_LEAD_MAX:
                # Search the quiet beat before the first recovered word. The
                # min() guarantees silence snapping can never cut that word.
                snapped = _snap_to_silence(target - 1.0, timeline, rms)
                new_start = max(0.0, min(target - SENTENCE_PREROLL, snapped, start))

    if expand_tail:
        following = np.flatnonzero(ends > end + 0.05)
        if following.size:
            idx = int(following[0])
            bridges_cut = 0.0 <= starts[idx] - end <= CONTEXT_BRIDGE_GAP
            target_idx = idx
            if bridges_cut and ends[idx] - end <= CONTEXT_TAIL_MAX:
                # A joke tag can span a tiny question/answer bridge before the
                # actual landing ("Does it?" -> "No." -> "It smells like
                # paper."). Follow at most three tightly connected sentences;
                # the time ceiling remains the primary anti-run-on guard.
                while target_idx + 1 < count and target_idx - idx + 1 < CONTEXT_TAIL_SENTENCE_MAX:
                    next_idx = target_idx + 1
                    next_gap = float(starts[next_idx] - ends[target_idx])
                    if not (
                        0.0 <= next_gap <= CONTEXT_SECOND_SENTENCE_GAP
                        and ends[next_idx] - end <= CONTEXT_TAIL_MAX
                    ):
                        break
                    target_idx = next_idx
                target = float(ends[target_idx])
                snapped = _snap_to_silence(target + 1.0, timeline, rms)
                new_end = min(
                    float(timeline[-1]) if timeline.size else target + SENTENCE_POSTPAD,
                    max(target + SENTENCE_POSTPAD, snapped, end),
                )

    # Never trim the selected moment to make an expansion fit. Prefer setup
    # when both requested sides would exceed the wide engineering ceiling;
    # otherwise keep whichever complete adjacent unit fits intact.
    if new_end - new_start > clip_max:
        if end - new_start <= clip_max:
            new_end = end
        elif new_end - start <= clip_max:
            new_start = start
        else:
            new_start, new_end = start, end

    return {
        "start": round(new_start, 3),
        "end": round(new_end, 3),
        "lead_added": round(max(0.0, start - new_start), 3),
        "tail_added": round(max(0.0, new_end - end), 3),
    }


# Dead-air trim (presentation-only, winners after sentence refinement). A
# delivered window can still OPEN or CLOSE on a run of near-silence: sentence
# snapping only fires when a boundary sits in its window, and no-transcript
# clips fall back to the silence snap that deliberately lands in a quiet gap,
# so the clip opens on the gap rather than the action. A human editor opens ON
# the moment. Move the start forward past a leading silent run and the end back
# past a trailing one -- never across the peak lead/tail floors or a game
# anchor, and revert wholesale if the result breaks min duration. Only the head
# and tail runs are trimmed; mid-clip pauses (part of the moment's rhythm) are
# left untouched.
# Deliberately conservative. The golden truth windows are human cuts, and a
# human keeps a beat of quiet lead-in before the action -- an RMS test reads
# that beat as "dead air", so an aggressive trim fights the ground truth and
# regresses mean_iou on already-tight clips. These thresholds fire only on
# EGREGIOUS edge silence (multi-second) and keep a generous breathing pad, so a
# well-formed clip is left alone and only a clip that genuinely opens on a long
# quiet gap gets tightened.
DEAD_AIR_TRIM_MIN_RUN = 3.0      # only trim a silent head/tail longer than this
DEAD_AIR_QUIET_FACTOR = 1.3      # "quiet" = within this x the clip's own floor RMS
DEAD_AIR_VOICE_HOT = 0.22        # voice arousal at/above this is speech, never trimmed
DEAD_AIR_TRIM_PREROLL = 1.0      # breathing room kept before the first hot frame
DEAD_AIR_TRIM_POSTPAD = 1.2      # breathing room kept after the last hot frame


def trim_dead_air(
    bounds: dict,
    timeline: np.ndarray,
    rms: Optional[np.ndarray],
    voice_arousal: Optional[np.ndarray] = None,
    anchor_time: Optional[float] = None,
    anchor_label: Optional[str] = None,
    clip_min: float = HARD_MIN,
) -> dict:
    """Trim a leading/trailing run of near-silence off an already-SELECTED window.

    Presentation-only, run on winners after ``refine_to_sentences``. A frame is
    "hot" when its RMS clears the clip's own quiet floor OR it carries speech;
    the start moves forward to just before the first hot frame and the end back
    to just after the last, each only if the silent run it removes is longer
    than ``DEAD_AIR_TRIM_MIN_RUN``. Every window invariant is re-applied (peak
    lead/tail floors, anchor containment, min duration) and any violation
    reverts to the input window -- a packaging tweak may never eat the payoff.
    """
    if rms is None or timeline is None or timeline.size == 0:
        return bounds
    start = float(bounds["start"])
    end = float(bounds["end"])
    peak_time = float(bounds.get("peak_timestamp", start))
    orig = dict(bounds)

    mask = (timeline >= start) & (timeline <= end)
    idxs = np.where(mask)[0]
    if idxs.size < 3:
        return bounds

    quiet_floor = float(np.min(rms[idxs])) * DEAD_AIR_QUIET_FACTOR + 1e-9
    has_voice = (
        voice_arousal is not None and voice_arousal.size == timeline.size
    )

    def _hot(i: int) -> bool:
        if float(rms[i]) > quiet_floor:
            return True
        if has_voice and float(voice_arousal[i]) >= DEAD_AIR_VOICE_HOT:
            return True
        return False

    # Leading silent run: first hot frame from the front.
    first_hot = next((i for i in idxs if _hot(i)), None)
    if first_hot is not None:
        new_start = float(timeline[first_hot]) - DEAD_AIR_TRIM_PREROLL
        if new_start - start >= DEAD_AIR_TRIM_MIN_RUN:
            start = new_start

    # Trailing silent run: last hot frame from the back.
    last_hot = next((i for i in reversed(idxs) if _hot(i)), None)
    if last_hot is not None:
        new_end = float(timeline[last_hot]) + DEAD_AIR_TRIM_POSTPAD
        if end - new_end >= DEAD_AIR_TRIM_MIN_RUN:
            end = new_end

    # Re-apply the window invariants (never trim into the payoff or an anchor).
    start = min(start, peak_time - LEAD_IN_MIN)
    end = max(end, peak_time + TAIL_MIN)
    if anchor_time is not None:
        lead_floor = EVENT_LEAD_IN if anchor_label in EVENT_TAIL_LABELS else LEAD_IN_MIN
        start = min(start, float(anchor_time) - lead_floor)
        end = max(end, float(anchor_time) + TAIL_MIN)
    start = max(0.0, start)

    if end - start < clip_min:
        return orig
    result = dict(bounds)
    result["start"] = round(start, 3)
    result["end"] = round(end, 3)
    return result


def _event_settle_end(
    anchor_time: float,
    timeline: np.ndarray,
    voice_arousal: Optional[np.ndarray],
) -> Optional[float]:
    """First post-payoff quiet that looks like the moment is over.

    Requires two consecutive low-voice frames (or the last frame under the
    settle threshold) so a single breath doesn't chop the celebration.
    Returns a candidate end time, or None if voice stays hot through EVENT_TAIL.
    """
    if voice_arousal is None or voice_arousal.size == 0:
        return None
    anchor_idx = int(np.argmin(np.abs(timeline - anchor_time)))
    quiet_run = 0
    for idx in range(anchor_idx, len(voice_arousal)):
        dt = float(timeline[idx] - anchor_time)
        if dt > EVENT_TAIL:
            break
        if dt < TAIL_MIN:
            quiet_run = 0
            continue
        if float(voice_arousal[idx]) < EVENT_SETTLE_VOICE:
            quiet_run += 1
            if quiet_run >= 2:
                drop_time = float(timeline[idx])
                return drop_time + EVENT_SETTLE_PAD
        else:
            quiet_run = 0
    return None


# Action-onset lookback for event-anchored windows. The reaction arc measures
# when the STREAMER responded, which on a kill lags the fight by 15-25s, so a
# fixed lead off the banner opens the clip after the fight that earned it.
# Measured 2026-07-23 on a creator-confirmed keep: combat audio ran from 4732,
# the kill banner landed at 4759, the reaction peaked at 4771, and the window
# opened at 4759 — every spike that made the moment was outside it. The
# creator independently asked for a start of ~4725.
ACTION_LOOKBACK_MAX = 40.0   # never walk further back than this from the anchor
ACTION_LOOKBACK_GAP = 12.0   # quiet longer than this ends the run
ACTION_LOUD_FACTOR = 1.8     # "loud" = this x the local median rms
ACTION_LOOKBACK_PAD = 2.0    # breathing room before the first loud frame
# When a real combat run is found, the window may exceed EVENT_PREFERRED_MAX to
# hold it: that cap exists to stop a multi-minute hype envelope, not to cut the
# fight off the front of its own kill. Still far under HARD_MAX.
EVENT_ONSET_MAX = 58.0


def action_onset_start(
    anchor_time: float,
    timeline: np.ndarray,
    rms: Optional[np.ndarray],
) -> Optional[float]:
    """Walk backward from a game anchor through the contiguous run of loud
    frames that produced it, and return where that run began.

    Returns None when there is no usable rms track or no run to follow, so the
    caller keeps its existing fixed lead.
    """
    if rms is None or timeline is None or len(timeline) != len(rms):
        return None
    lo = anchor_time - ACTION_LOOKBACK_MAX
    mask = (timeline >= lo) & (timeline <= anchor_time)
    if not mask.any():
        return None
    idxs = np.where(mask)[0]
    window_rms = rms[idxs]
    median = float(np.median(window_rms))
    if median <= 0:
        return None
    loud = window_rms >= median * ACTION_LOUD_FACTOR
    if not loud.any():
        return None
    # Scan backward from the anchor; a quiet stretch longer than the gap
    # tolerance means the earlier activity belongs to a different moment.
    onset_idx = None
    last_loud_time = anchor_time
    for pos in range(len(idxs) - 1, -1, -1):
        t = float(timeline[idxs[pos]])
        if loud[pos]:
            if last_loud_time - t > ACTION_LOOKBACK_GAP:
                break
            onset_idx = idxs[pos]
            last_loud_time = t
    if onset_idx is None:
        return None
    return float(timeline[onset_idx]) - ACTION_LOOKBACK_PAD


def extend_to_action_onset(
    clip: dict,
    timeline: np.ndarray,
    rms: Optional[np.ndarray],
    anchor_time: Optional[float] = None,
    anchor_label: Optional[str] = None,
    clip_max: float = HARD_MAX,
) -> dict:
    """WINNERS ONLY: reopen an event clip on the fight that earned its banner.

    Presentation-only, exactly like ``refine_to_sentences``. Applying this at
    candidate time instead was measured to regress the golden set across the
    board (v1 precision 0.400 -> 0.339, v2 mean_iou 0.784 -> 0.701): a longer
    window lowers ``reaction_auc / duration``, which changes selection scores
    and therefore which clips ship. Selection must keep judging the moment on
    its silence-snapped window; only the shipped cut moves.

    Returns the input window unchanged whenever the extension is unavailable
    or would break an invariant, so a clip can never lose its payoff.
    """
    start = float(clip.get("start", 0.0))
    end = float(clip.get("end", 0.0))
    if anchor_label not in EVENT_TAIL_LABELS or anchor_time is None:
        return {"start": start, "end": end}
    onset = action_onset_start(float(anchor_time), timeline, rms)
    if onset is None or onset >= start:
        return {"start": start, "end": end}
    new_start = max(0.0, _snap_to_silence(onset, timeline, rms))
    if end - new_start > min(clip_max, EVENT_ONSET_MAX):
        new_start = max(0.0, end - min(clip_max, EVENT_ONSET_MAX))
    if new_start >= start:
        return {"start": start, "end": end}
    return {"start": round(new_start, 3), "end": round(end, 3)}


def snap_boundaries(
    arc: Arc,
    timeline: np.ndarray,
    rms: Optional[np.ndarray],
    anchor_time: Optional[float] = None,
    anchor_label: Optional[str] = None,
    clip_min: float = HARD_MIN,
    clip_max: float = HARD_MAX,
    cuts: Optional[np.ndarray] = None,
    voice_arousal: Optional[np.ndarray] = None,
) -> dict:
    """Convert an arc into a snapped clip window that trusts the arc's extent.

    The arc's onset-expanded start/end (where the reaction built and subsided)
    IS the window, plus small breathing pads, snapped to audio-silence gaps.
    ``clip_min``/``clip_max`` are sanity invariants: hitting the floor flags
    the candidate as padded; the ceiling trims lead first (distant build-up is
    more expendable than the payoff and its settle).

    Sentence alignment deliberately does NOT happen here: candidate windows
    feed selection's gates (dead-air, mean-intensity), which were tuned
    against silence-snapped windows. Winners get sentence-refined afterwards
    via ``refine_to_sentences`` (presentation only).

    Event-anchored elim/knock windows use a longer combat lead and an
    asymmetric tail that caps once voice settles after the banner.
    """
    peak_time = float(timeline[arc.peak_idx])
    raw_start = float(timeline[arc.start_idx])
    raw_end = float(timeline[arc.end_idx])
    raw_arc_span = raw_end - raw_start
    is_event = anchor_label in EVENT_TAIL_LABELS and anchor_time is not None
    lead_floor = EVENT_LEAD_IN if is_event else LEAD_IN_MIN

    start = _snap_to_silence(raw_start - LEAD_PAD, timeline, rms, cuts=cuts)
    end = _snap_to_silence(raw_end + TAIL_PAD, timeline, rms, cuts=cuts)

    # Peak-relative floors only -- the payoff is never the first/last frame.
    start = min(start, peak_time - LEAD_IN_MIN)
    end = max(end, peak_time + TAIL_MIN)

    # Guarantee a game anchor (kill-feed / win timestamp) stays inside the window.
    anchor_tail = EVENT_TAIL if is_event else TAIL_MIN
    settle_end = None
    if is_event:
        settle_end = _event_settle_end(float(anchor_time), timeline, voice_arousal)
        if settle_end is not None:
            # Floor: still keep a short celebration after the banner.
            anchor_tail = max(TAIL_MIN, min(EVENT_TAIL, settle_end - float(anchor_time)))
        start = min(start, float(anchor_time) - lead_floor)
        end = max(end, float(anchor_time) + anchor_tail)
        # Cap: if the reaction arc overstayed into quiet looting, cut at settle.
        if settle_end is not None and end > settle_end:
            snapped = _snap_to_silence(settle_end, timeline, rms, cuts=cuts)
            end = max(
                snapped,
                float(anchor_time) + TAIL_MIN,
                peak_time + TAIL_MIN,
            )
    elif anchor_time is not None:
        start = min(start, float(anchor_time) - LEAD_IN_MIN)
        end = max(end, float(anchor_time) + anchor_tail)

    start = max(0.0, start)

    # Event packages should stay postable: prefer a ~40s combat beat over a
    # full multi-minute hype envelope when the arc is oversized.
    effective_max = clip_max
    if is_event:
        effective_max = min(clip_max, EVENT_PREFERRED_MAX)

    floor_padded = False
    duration = end - start
    if duration < clip_min:
        # No real arc around this peak. Extend backward to a natural audio
        # boundary (quiet gap) so the clip opens with context, not mid-word;
        # only spill forward if the VOD head blocks the lookback.
        floor_padded = True
        deficit = clip_min - duration
        start = _snap_to_silence(start - deficit, timeline, rms, cuts=cuts)
        start = max(0.0, min(start, end - clip_min + TAIL_PAD))
        if end - start < clip_min:
            end = start + clip_min
    elif duration > effective_max:
        # Trim the lead first: the payoff + settle matter more than build-up
        # that started effective_max seconds earlier.
        start = max(start, end - effective_max)
        if start > peak_time - LEAD_IN_MIN:
            start = max(0.0, peak_time - LEAD_IN_MIN)
            end = start + effective_max
        if anchor_time is not None:
            start = min(start, max(0.0, float(anchor_time) - lead_floor))
            end = max(end, float(anchor_time) + anchor_tail)
            if end - start > effective_max:
                # Keep payoff near the back half: trim lead further.
                start = max(0.0, end - effective_max)

    return {
        "start": round(max(0.0, start), 3),
        "end": round(end, 3),
        "peak_timestamp": round(peak_time, 3),
        "peak_value": arc.peak_value,
        "raw_arc_span": round(raw_arc_span, 3),
        "floor_padded": floor_padded,
    }
