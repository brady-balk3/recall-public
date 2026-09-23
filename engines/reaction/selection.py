# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Global clip selection (plan §5.5, step 5).

Key behavior changes vs. the legacy ranking engine:
  * review count is a hard ceiling with an adaptive publishability floor, so
    weak VODs are allowed to yield fewer clips,
  * a **relative quality bar** (median(R) + k*MAD(R)) gates every candidate, so
    weak VODs yield few/zero clips by design,
  * adaptive spacing via **non-max suppression on overlap** (plus nearby-gap
    absorb) replaces the flat 60s MIN_GAP, so dense action survives as
    separate strong clips while same-fight multi-peaks collapse to one window.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class SelectionTuning:
    # Two candidates overlapping more than this fraction -> the weaker is suppressed.
    nms_overlap: float = 0.30
    # Same-fight / same-story peaks often produce adjacent short arcs with a
    # small gap (Fortnite elim→celebrate, narrative beat→reaction). Overlap-only
    # NMS lets those ship as 2–3 near-duplicate clips. When the gap is at most
    # this many seconds AND the union fits nearby_moment_max_span_sec, absorb
    # the weaker into the stronger window so one clip covers start→finish.
    nearby_moment_gap_sec: float = 12.0
    # Up to ~60s covers multi-peak fights (elim→wipe→react) without merging
    # distinct rotations; still under boundaries.HARD_MAX (90s).
    nearby_moment_max_span_sec: float = 60.0
    # R(t) is a robust-z reaction curve. On low-energy VODs the relative bar can
    # collapse to ~0, so keep ordinary clips from passing on tiny one-frame noise.
    ordinary_min_peak: float = 0.55
    # peak_value is the height of the loudest instant; reaction_auc is the area
    # under the reaction curve. They correlate at only +0.520, and it is the
    # AREA that predicts this creator's Keep (AUC 0.656, the best single
    # predictor in the trace) while the gate admitted on HEIGHT alone. A
    # 22-second laugh and a 2-second scream carry the same area; only the
    # scream used to clear ordinary_min_peak. This gives sustained reactions a
    # second way through, so the judge -- the only stage able to tell a funny
    # remark from a dull one -- actually sees them. Measured 2026-08-15 by
    # census over every candidate it admits across 8 VODs: 21/73 = 28.8% Keeps,
    # against 30.0% below the adaptive floor and a 5.6% uniform baseline, for
    # +11.8% judge calls.
    #
    # OFF (0.0) because the gates are serial and the NEXT one eats all of it:
    # every one of those 73 moments scores below the adaptive floor, so all 21
    # Keeps are discarded one stage later and nothing reaches the ranker. Net
    # delivered Keeps 0.0/VOD, for real judge cost. Verified two ways -- against
    # max(0.75, best_signal*0.50) and against a floor derived empirically from
    # each VOD's highest cut score -- both give 0 of 73 surviving.
    #
    # The path itself measures well and stays here ready to switch on. Turning
    # it on is only worth it once the adaptive floor admits low-signal
    # moments, and that has to be the change that leads.
    sustained_reaction_min_auc: float = 0.0
    ordinary_min_selection_score: float = 0.75
    # Ordinary no-game clips should feel like a moment, not just lobby chatter.
    # Require at least two contributing channels such as voice+face, voice+burst,
    # speech+face, or chat+voice -- scaled down when the VOD only has one track.
    ordinary_min_signal_agreement: float = 0.50
    ordinary_min_modalities: int = 2
    # Confirmed gameplay events already have OCR payoff evidence. They should
    # not be rejected just because optional channels (chat/burst/speech) were
    # quiet, but they still need at least one human signal in the window.
    event_min_signal_agreement: float = 0.40
    event_min_game_evidence: float = 0.50
    event_max_dead_air: float = 0.50
    # OCR confirms WHAT happened; the reaction confirms it MATTERED. A kill
    # banner with weak sustained reaction (streamer farming routine kills) is
    # never postable, on any selection path.
    event_min_mean_intensity: float = 0.55
    # A routine game event does not earn the visual-verdict floor bypass: the
    # VLM types a kill/win banner "post" because it is a recognizable event,
    # but the creator only wants the hectic ones. Startle OR a strong reaction
    # peak is what admits an event past the floor (see _event_reaction_landed
    # and _visual_floor_keep). Confirmed match wins use a gentler reaction bar.
    event_reaction_min_startle: float = 0.15
    event_reaction_min_peak: float = 2.5
    # A confirmed match win gets a gentler reaction bar than an ordinary kill
    # -- a quiet win is more postable than a quiet elim -- but still not a free
    # pass: two dead-calm VICTORY ROYALE clips were creator rejects.
    win_reaction_min_startle: float = 0.10
    win_reaction_min_peak: float = 1.2
    event_rich_min_candidates: int = 4
    # Event-rich precision policy is local in time. A handful of OCR events in
    # hour one must not suppress ordinary reactions three hours later.
    event_rich_window_sec: float = 30.0 * 60.0
    event_rich_dedupe_sec: float = 20.0
    ordinary_burst_exemption: float = 0.12
    event_rich_score_floor_fraction: float = 0.56
    event_rich_startle_exemption: float = 0.60
    # In an event-rich VOD, burst is useful evidence but cannot be the only
    # route into the deck. Great reactions are often voice/face/transcript led
    # (especially on long streams with sparse laughter tags). Semantic reads
    # are preferred; a stricter observed-signal fallback keeps the selector
    # useful on installs without the local judge.
    event_rich_signal_hook_min: float = 0.45
    event_rich_signal_agreement_min: float = 0.50
    event_rich_signal_max_dead_air: float = 0.45
    event_rich_post_payoff_min: float = 0.35
    event_rich_post_self_contained_min: float = 0.50
    event_rich_maybe_payoff_min: float = 0.60
    event_rich_maybe_self_contained_min: float = 0.70
    ordinary_max_dead_air: float = 0.65
    game_bonus: float = 0.5
    # Multi-kill / event-density bonus layered on top of game_evidence.
    event_density_weight: float = 0.55
    # Prefer payoff mid/late in the window (setup -> kill -> reaction).
    payoff_position_weight: float = 0.20
    # Editorial duration prior: with content-driven boundaries, windows of
    # different lengths compete over the same moment, and mean-intensity alone
    # always favors the tightest one. This counterweight expresses "slightly
    # longer with context beats shorter" without letting length buy rank.
    duration_prior_weight: float = 0.35
    # Floor-padded candidates had no real arc around their peak -- the padding
    # itself is weak-moment evidence.
    floor_padded_penalty: float = 0.40
    # Lobby/intermission banter drives huge voice/face reaction (streamers are
    # most animated chatting, not focused), which otherwise lets it outrank
    # real gameplay highlights on raw magnitude alone.
    lobby_score_penalty: float = 2.5
    lobby_ranker_penalty: float = 0.50
    lobby_max_duration: float = 15.0
    # Lobby banter is usually weak, but a founder-kept cat-on-screen moment
    # proved that a hard scene-duration ban also hides real off-game reactions.
    # Only a corroborated burst + sustained multi-modal response earns this
    # narrow exemption.
    lobby_reaction_min_peak: float = 1.25
    lobby_reaction_min_mean_intensity: float = 0.75
    lobby_reaction_min_signal_agreement: float = 0.65
    lobby_reaction_max_dead_air: float = 0.30
    lobby_reaction_min_burst: float = 0.15
    lobby_reaction_min_gameplay_change: float = 0.20
    # Legacy name kept for imports; lobby uses absolute penalty above.
    scene_score_multiplier: float = 0.4
    # Semantic judge blend (HUMAN_CLIPS Package C): weight on the LLM's
    # hook/payoff/self-contained read, added to selection_score only when the
    # judge actually ran on the candidate. Candidates without semantic fields
    # score exactly as before.
    semantic_score_weight: float = 0.8
    # First-five-second hook quality is a first-class rank signal. The score
    # blends the local judge (when available) with signal-only evidence, so it
    # remains useful on installs without an LLM model.
    hook_score_weight: float = 0.65
    # A startle is a corroborated loud onset synchronized with a sudden face
    # transition into Fear/Surprise. This compound is rare and high-value, so
    # it receives a strong editorial prior; static emotion and generic speech
    # onsets do not.
    startle_bonus_weight: float = 8.0
    # Visible stream-chat OCR is direct human evidence even when Twitch chat
    # download is unavailable. Only a corroborated phrase score qualifies;
    # generic isolated "lol" text stays below this floor.
    visible_crowd_bonus_weight: float = 1.5
    visible_crowd_anchor_min: float = 0.72
    # Review mode keeps single semantic concerns advisory, but a consensus
    # ``skip`` + ``filler`` verdict is a hard reject. Real SH2 evidence showed
    # those clips otherwise consuming nearly half the review deck. AUTO_TUNING
    # additionally rejects either signal and enforces a strict payoff floor.
    semantic_min_payoff: float = 0.0
    semantic_hard_reject: bool = False
    # Narrow judge consensus and generic visual filler are standard selector
    # invariants, not tuning switches.
    # A positive visual verdict outranks a weaker text-only rejection; a visual
    # skip remains a standard veto.
    # A narrow, replayable guard for the loading-screen failure mode. The
    # calibrated visual verdict may be promoted from skip -> maybe by a generic
    # transcript lexical fallback. When grounded non-gameplay state AND both
    # original judges still agree skip/filler, keep that consensus as a
    # structural veto. A directly model-authored lobby joke/story is unaffected.
    routine_non_gameplay_consensus_veto: bool = True
    # The consensus veto above needs a visual model verdict, so it is dead on
    # any VOD the judge did not reach. This companion keys only on signal-track
    # OCR, which every scan produces regardless of judge coverage. Ratio is the
    # share of in-window OCR samples showing explicit waiting/menu state; a
    # majority means the window never really left the lobby.
    routine_non_gameplay_ocr_veto: bool = True
    routine_non_gameplay_ocr_ratio: float = 0.60
    # Use visual verdicts for judged candidates near the quality floor, where arousal
    # alone may not distinguish publishable content. Unjudged candidates retain the
    # arousal floor. A maybe verdict also needs this fraction of the arousal floor.
    visual_floor_maybe_fraction: float = 0.5
    # Cold-face, voice-led moments with no gameplay outcome — typically the
    # streamer talking to chat. Demote hard so they cannot top the deck even
    # when the VLM labels them ``post`` (2/30 chatty deck, 2026-07-22).
    chat_reaction_score_penalty: float = 4.0
    # Peak face arousal, not the window mean. Mild ambient expression commonly
    # sits around 0.07-0.10 and is not evidence that a talk-only beat landed.
    chat_reaction_max_face: float = 0.22
    chat_reaction_min_voice: float = 0.15
    # Whether judge verdicts carry GATE authority (consensus skip+filler hard
    # reject, semantic rescue wiping gate rejections and qualifying editorial
    # overflow, and the verdict-keyed event-rich admission branches). False
    # demotes the judge's positive influence to rank-score contributions only:
    # verdicts still feed semantic_score_weight / the hook blend, but cannot
    # rescue. The standard negative policy still rejects a narrow skip+filler
    # consensus.
    # Field default True == the historical behavior; measured before any
    # production flip by scripts/judge_authority_matrix.py (judge_rank_only
    # config) — same measure-then-apply pattern as hard_total_cap.
    semantic_gate_authority: bool = True
    # Lead weight of the judge's hook read inside compute_hook_score (the
    # remainder stays on observed signals). 0.70 is the historical constant;
    # 0.0 removes semantic influence from the hook blend entirely (ablation
    # control lanes only — production keeps the lead).
    semantic_hook_lead: float = 0.70
    # A confident local-editor verdict is evidence, not merely a score nudge.
    # Review mode may rescue a candidate that a coarse signal rule
    # mislabeled (for example, a real punchline during a lobby screen).  The
    # verdict must agree on postability, payoff, and self-containment so an LLM
    # cannot wave generic filler through on a single optimistic field.
    semantic_rescue_min_hook: float = 0.45
    semantic_rescue_min_payoff: float = 0.60
    semantic_rescue_min_self_contained: float = 0.50
    # A visual ``skip`` is normally authoritative, but a fresh local VLM pass
    # can be unstable on fast reactions.  It becomes advisory only under a
    # five-way conflict: the text editor sees a complete payoff, the observed
    # signals are both broad and intense, and the frozen BASE ranker is also
    # strongly positive.  Founder replay found two reviewed Keeps and zero
    # reviewed Passes in this narrow slice; looser variants admitted a Pass.
    visual_skip_conflict_min_payoff: float = 0.70
    visual_skip_conflict_min_signal_agreement: float = 0.75
    visual_skip_conflict_min_peak: float = 2.50
    visual_skip_conflict_min_ranker_score: float = 0.70
    visual_skip_conflict_max_routine_ratio: float = 0.50
    # ``max_clips`` is the primary review-deck target, not permission to erase
    # additional moments the editor independently called postable.  Qualified
    # semantic/startle overflow is bounded; auto/post mode sets this to zero.
    editorial_overflow_fraction: float = 0.34
    editorial_overflow_min: int = 3
    # When true, ``max_clips`` becomes a HARD total ceiling: protected clips,
    # primary picks, and editorial overflow all count against the same budget
    # (protected clips sort first, so they naturally survive). Overflow keeps
    # its qualifying-evidence requirement but can no longer leak past the
    # budget. ``max_clips <= 0`` keeps its existing meaning either way.
    # Measured by scripts/ablation_matrix.py (hard_cap_* configs) before any
    # production flip — see the Package 3 decision rules there.
    hard_total_cap: bool = False
    # A hard cap must not create a VOD-relative blackout when a fully judged,
    # self-contained moment inside that span has stronger editorial consensus
    # than a weaker pick in an already dense cluster. A gap is abnormal relative
    # to the spacing expected from this deck's own first/last selected moments;
    # there is deliberately no universal minute cutoff. This is a replacement,
    # never deck padding: total review volume remains unchanged.
    temporal_blackout_expected_gap_multiplier: float = 2.5
    temporal_blackout_max_rescues: int = 2
    temporal_blackout_min_visual_payoff: float = 0.70
    temporal_blackout_min_semantic_payoff: float = 0.60
    temporal_blackout_min_self_contained: float = 0.70
    # ``max_clips`` is a ceiling, not a fill target. When more candidates clear
    # the absolute evidence gates than the requested deck can hold, retain only
    # candidates reasonably close to the VOD's strongest observed moment.
    # Human anchors remain exempt. This lets weak VODs yield fewer than the cap
    # instead of padding every review deck to exactly 15.
    adaptive_deck_score_floor_fraction: float = 0.50
    # When the ranker has scored the pool, the adaptive floor is redundant with
    # it: the floor is a hand-tuned proxy for "is this good enough", which is
    # exactly what the ranker computes, and the two disagree in the ranker's
    # favour. Measured 2026-08-16 over the 60-fixture founder set, ranker
    # loaded: dropping the floor moved precision 0.353 -> 0.364 AND recall
    # 0.481 -> 0.497 at unchanged p@5 0.457 -- the only config in a 32-config
    # matrix to improve both admission axes at once.
    #
    # It is deliberately NOT applied without a ranker. The same knockout with
    # optional models off measured slightly WORSE (precision 0.312 -> 0.308,
    # recall 0.452 -> 0.450): with nothing ranking the pool, the floor is doing
    # real work and earns its keep. This gate is only redundant once something
    # better is present.
    #
    # Keyed on ranker scores EXISTING, not on ranker_orders_deck: the live
    # fraction under production's blend policy is the signal one above, because
    # ranker_use_personalized_adaptive_floor is off, so keying this on the
    # ordering flag would leave it inert exactly where it was measured.
    ranker_scored_adaptive_deck_score_floor_fraction: float = 0.0
    # With a creator-trained ranker available, the signal-ranked candidate pool can
    # be narrowed more aggressively before personal ordering. Deterministic
    # first-run decks keep the more recall-friendly floor above.
    personalized_adaptive_deck_score_floor_fraction: float = 0.60
    # A semantic veto cannot be offset by a score bonus. Allow requested moments to
    # survive that judgement when configured, while keeping structural vetoes.
    crowd_clip_bonus: float = 0.5
    # A semantic veto cannot be offset by a score bonus. Allow requested moments to
    # survive that judgement when configured, while keeping structural vetoes.
    crowd_command_survives_semantic_reject: bool = True
    # Weight on undivided reaction_auc in the signal score. See the comment in
    # _score_components: the score otherwise uses average height and discards
    # area. 0.0 is the historical behaviour.
    reaction_area_weight: float = 0.0
    crowd_clip_pad_exempt_min: int = 2
    # An explicit viewer !clip marker protects its candidate from the evidence
    # gates. MEASURED 2026-07-25 across all 24 golden fixtures, dropping this
    # protection lost on every aggregate: recall 0.543 -> 0.518, precision
    # 0.310 -> 0.299, p@5 0.342 -> 0.325, mean_iou 0.485 -> 0.455 (5 fixtures
    # worse, 3 better, 16 unchanged), and it was the sole cause of all 13
    # golden regressions in 2be3a3e.
    #
    # The "command-heavy VOD monopolizes Primary" concern behind dropping it is
    # REAL but narrow: only 4 of 24 fixtures are command-heavy (13-14 of 15 deck
    # slots), while the fixtures this cost recall on carry just 1-4 command
    # clips. A binary switch therefore pays the full price on VODs that never
    # had the problem. The measured fix is a CAP on how much of Primary
    # commands may hold, not removing protection -- a first cap prototype
    # already beat both binary settings on recall (0.548) and mean_iou (0.493).
    # Set False to reproduce the no-protection lane when measuring that work.
    viewer_commands_protected: bool = True
    # An optional deck-share cap limits command-heavy protected pools. Keep it disabled
    # with a small command bonus: removing protection can discard requested moments rather
    # than merely rebalance them. Enabling this requires reviewing the recall-versus-
    # diversity tradeoff.
    crowd_command_deck_fraction: float = 0.0
    # Editorial hygiene runs after qualification but before the creator-facing
    # budget. Credits/outro regions are a structural veto, and the full
    # qualified pool is grouped by moment identity so neither Primary nor
    # Second Look can spend two cards on the same beat.
    editorial_hygiene_enabled: bool = True
    # Visual POST is advisory unless a real outcome, complete verbal beat,
    # crowd/startle, or calibrated face reaction grounds it. Rare sustained
    # face reactions may rescue an adaptive/event floor only when independent
    # context supports them; they never bypass evidence gates or receive
    # categorical sort priority over the learned/signal rank.
    face_spike_absolute_min: float = 0.40
    face_spike_percentile: float = 99.0
    face_spike_min_sustained_frames: int = 2
    face_spike_min_local_delta: float = 0.20
    # A calibrated spike receives one of these bounded rank corrections only
    # after it actually needed the adaptive/event floor rescue. Spikes that
    # already qualified get no ordering advantage merely for being spikes.
    face_spike_score_bonus: float = 1.0
    # Normalized ranker-path correction on the 0..1 learned/blend scale, not a
    # boolean priority tier; strong ordinary clips can and should stay above.
    face_spike_rank_bonus: float = 0.35
    face_spike_rescue_max_events: int = 3
    face_spike_event_merge_sec: float = 8.0
    moment_identity_max_span_sec: float = 120.0
    moment_anchor_cluster_sec: float = 8.0
    moment_token_max_gap_sec: float = 45.0
    moment_token_jaccard: float = 0.45
    moment_token_min_shared: int = 3
    # Second Look is a bounded uncertainty sample, not another full review
    # deck. Five is enough to expose close calls without making a weak VOD
    # feel like the engine manufactured fifteen more recommendations.
    second_look_band_size: int = 5
    # If strict relative floors leave a nearly empty deck, recover at most this
    # many candidates that already carry absolute direct evidence. This is not
    # quota filling: generic best-of-bad candidates remain rejected.
    low_yield_recovery_target: int = 3
    # --- Content axis (two-axis admission; Fable-5 clip-algo work) ----------
    # Every existing gate is an arousal threshold; a creator's quiet keeps
    # (wins whose banners OCR never read, calm funny beats) die at those gates
    # or at the deck cap while carrying strong CONTENT evidence a VLM can see.
    # When enabled, grounded visual-judge fields (visual_semantic_*) can lift
    # rank and — for REACTION-QUALIFIED VISUAL WINS only — bypass arousal gates
    # per candidate. Verbal beats used to share that bypass; measured 2026-07-22
    # creator decks (0/18, 2/30) showed verbal anchors force-entering
    # chat-thanks / story chatter that humans reject. Verbal still contributes
    # to content_interest rank among gate-qualified candidates.
    content_axis: bool = False
    # A content anchor needs grounded evidence, not a bare "post" verdict
    # (bare verdicts saturate, §7.1): a visible game outcome, or a complete
    # verbal beat a cold viewer can follow. Floors below keep it strict.
    content_min_confidence: float = 0.45
    content_min_payoff: float = 0.55
    content_min_self_contained: float = 0.50
    # Rank contribution: content_weight * content_interest_score (0..1) joins
    # selection_score so a content anchor is not budget-cut at the deck cap
    # (observed: three kept wins cleared every gate and died at clip_ceiling).
    # Sized against real decks: a quiet VOD's cap cutoff sits around 7.5-8.0
    # signal score while a quiet confirmed win scores ~2, so a full-interest
    # moment needs ~+6 to compete. Verbal beats use the lower weight below —
    # measured 2026-07-20 content-axis eval: boolean sort priority + full
    # weight on verbal anchors lifted mean recall/precision but slipped p@5.
    content_score_weight: float = 6.0
    content_verbal_score_weight: float = 3.0
    # Cap verbal content-anchor labels used for ranking and diagnostics to prevent verbal
    # moments from flooding the deck. These labels do not bypass arousal gates; validated
    # win anchors remain exempt.
    content_anchor_max: int = 3
    # Content-shape policy (engines.reaction.content_shape). When True AND the
    # inferred shape is narrative, complete story/verbal beats may bypass
    # arousal gates. Enabled by default via settings ``content_shape_selection``.
    # Chat-only moments stay blocked.
    allow_narrative_story_bypass: bool = False
    # Telemetry only — which shape produced this tuning (if any).
    content_shape: Optional[str] = None
    # --- Calibrated admission (absolute P(keep) gate) -----------------------
    # When True AND a calibrated publishability floor is active AND the
    # candidate carries ``ranker_publishability_score``, the VOD-relative
    # arousal bars are stood down in favor of the absolute calibrated gate:
    #   * peak_floor / score_floor in _gate_rejection_reasons,
    #   * the adaptive deck score floor,
    #   * the event-rich score floor.
    # Every structural veto (dead air, lobby, modality, hygiene, judge
    # consensus, routine OCR) still applies unchanged. Rationale: all three
    # relative bars key admission to proximity to the VOD's loudest moment;
    # measured 2026-08-15 they discard 28.8-30% Keep populations while the
    # pointwise-calibrated model separates at grouped-CV AUC 0.719 vs 0.618
    # for reaction_auc alone (data/eval/calibrated_gate.evaluation.json).
    # False preserves historical behavior exactly.
    calibrated_admission: bool = False
    # Union variant: the incumbent gates run UNCHANGED, and the calibrated
    # model additionally RESCUES a candidate the relative floors rejected when
    # its P(keep) clears the floor value. Strictly additive: no incumbent
    # admit is ever removed (the final publishability rejection is disabled in
    # this mode), so recall can only rise.
    #
    # MEASURED 2026-08-15, 57 golden fixtures, leave-that-VOD-out gate models
    # (data/eval/gate_replay_union_additive.evaluation.json), at tau 0.55:
    #   recall    0.441 -> 0.484 (+0.042), keeps delivered 252 -> 284 (+32)
    #   precision 0.320 -> 0.300 (-0.020), deck 14.2 -> 17.0
    #   p@5 0.421 unchanged, p@10 0.352 -> 0.350, must_not violations 4 -> 4
    #   23 fixtures better, 0 worse
    # The added clips carry a 20.5% marginal keep rate, against the founder's
    # 30% blinded probe of below-floor material -- the gap is truth censoring
    # (fixture truth only contains moments some run already surfaced), so the
    # measured precision dip is an upper bound on the real one.
    #
    # Full REPLACEMENT of the relative floors was measured first and failed at
    # every tau (recall 0.452 -> 0.309/0.240/0.167/0.143,
    # gate_replay.evaluation.json). Cause: the label corpus is ~23x enriched
    # for Keeps (41.9% vs 1.81% of real candidates), so a threshold picked on
    # it is not absolute in the wild -- per-VOD admit rate at tau=0.45 spans
    # 0.6%-20.7% of candidates and some decks emptied entirely.
    calibrated_union: bool = False
    # The active floor value, stamped by select_clips so per-candidate gate
    # logic can read it (the gates have no other channel to the parameter).
    # Not a tunable: whatever is passed as ranker_publishability_floor wins.
    calibrated_floor: float = 0.0
    # A floor rescue is worthless without a BUDGET. A rescued candidate is by
    # construction below the VOD-relative floor, so it sorts near the bottom
    # and dies at clip_ceiling one stage later -- the exact serial-filter
    # failure that made the peak_floor fix deliver 0 of 21 Keeps (see
    # sustained_reaction_min_auc).
    #
    # A score bonus cannot fix this: selection_score is unbounded and
    # peak-dominated (a scream pool sits near 16 while a quiet sustained laugh
    # sits near 3), so any constant large enough to matter on a loud VOD would
    # dominate a quiet one. Rescues therefore get RESERVED SLOTS instead --
    # at most this fraction of the deck, filled by calibrated P(keep)
    # descending. Bounded by construction, so keep rate cannot crater, and
    # scale-free, so it behaves the same on loud and quiet VODs.
    calibrated_rescue_deck_fraction: float = 0.25
    # Rescue slots must be ADDITIVE to the deck, not carved out of it.
    # Measured 2026-08-15 (union@0.55, 57 fixtures): carving them out of a
    # fixed 15 recovered 12 floor-killed truth windows but pushed 19 other
    # keeps into final_budget_cutoff -- net -7, exactly the zero-sum trap that
    # makes every isolated floor change fail. With this True the incumbent
    # picks keep their slots untouched and rescues extend the deck, so the
    # deck grows only when the model has evidence to fill it.
    calibrated_rescue_extends_deck: bool = True


def _duration_prior(duration: float) -> float:
    """0..1 preference curve over clip duration.

    Ramps 8s->14s (a bare blip earns nothing), full credit 14-45s (the
    watchable sweet spot), tapering to 0.4 by 90s (long clips must win on
    content, not length).
    """
    if duration <= 8.0:
        return 0.0
    if duration < 14.0:
        return (duration - 8.0) / 6.0
    if duration <= 45.0:
        return 1.0
    if duration >= 90.0:
        return 0.4
    return 1.0 - 0.6 * (duration - 45.0) / 45.0


def _payoff_position_bonus(c: dict) -> float:
    """Reward setup-before-payoff framing for event clips; mild nudge otherwise."""
    start = float(c.get("start", 0.0))
    end = float(c.get("end", start + 1.0))
    duration = max(1.0, end - start)
    anchor = c.get("game_anchor_time", c.get("peak_timestamp", start))
    try:
        pos = max(0.0, min(1.0, (float(anchor) - start) / duration))
    except (TypeError, ValueError):
        return 0.0
    label = c.get("game_label")
    if label in ("elimination", "knock", "terminal_win", "rank_progress"):
        # Ideal: payoff in the middle-late third after combat setup.
        if 0.35 <= pos <= 0.85:
            return 1.0
        if pos < 0.20:
            return -0.6  # banner as first frame -- missing the fight
        if pos > 0.92:
            return -0.3  # almost no celebration room
        return 0.2
    # Ordinary reactions: peak slightly after the open still reads better.
    if 0.30 <= pos <= 0.75:
        return 0.35
    return 0.0


# Review decks use the measured cap-15 interim policy. This limits what the
# creator must review; it does not erase lower-ranked candidates, which remain
# in the persisted selection trace with disposition ``clip_ceiling``.
#
# Positive judge authority is OFF in review mode: the local 4B semantic judge
# still runs (its titles/hook lines are shipped on every clip), but its verdicts
# do not rescue or score-nudge selection. The narrow skip+filler negative veto
# remains active because it independently agreed with creator passes. The
# judge_authority_matrix over the
# founder set showed every level of judge selection influence was neutral-to-
# negative vs. judge-off (full authority p@5 0.283, rank-only 0.250 vs. 0.300
# judge-off), and the titles-only lane reproduced the judge-off deck exactly.
# AUTO_TUNING keeps full judge authority — auto/post mode was designed judge-
# led and has not been separately measured here.
DEFAULT_TUNING = SelectionTuning(
    hard_total_cap=True,
    semantic_gate_authority=False,
    semantic_score_weight=0.0,
    semantic_hook_lead=0.0,
)

# Stricter deck for "auto / post these" mode: precision over recall.
AUTO_TUNING = SelectionTuning(
    ordinary_min_peak=0.70,
    ordinary_min_selection_score=0.95,
    ordinary_min_signal_agreement=0.55,
    ordinary_min_modalities=2,
    ordinary_max_dead_air=0.55,
    event_min_signal_agreement=0.40,
    event_max_dead_air=0.45,
    event_min_mean_intensity=0.60,
    event_rich_min_candidates=3,
    event_rich_score_floor_fraction=0.62,
    ordinary_burst_exemption=0.15,
    event_rich_signal_hook_min=0.55,
    event_rich_signal_agreement_min=0.55,
    event_rich_signal_max_dead_air=0.40,
    event_rich_post_payoff_min=0.45,
    event_rich_post_self_contained_min=0.60,
    event_rich_maybe_payoff_min=0.70,
    event_rich_maybe_self_contained_min=0.80,
    floor_padded_penalty=0.55,
    lobby_score_penalty=3.0,
    lobby_max_duration=12.0,
    payoff_position_weight=0.28,
    event_density_weight=0.70,
    semantic_score_weight=1.0,
    hook_score_weight=0.80,
    semantic_min_payoff=0.35,
    semantic_hard_reject=True,
    semantic_rescue_min_hook=0.60,
    semantic_rescue_min_payoff=0.75,
    semantic_rescue_min_self_contained=0.65,
    editorial_overflow_fraction=0.0,
    editorial_overflow_min=0,
    visible_crowd_anchor_min=0.85,
)


def tuning_for_mode(mode: Optional[str] = None) -> SelectionTuning:
    """Resolve selection preset from settings (review = default, auto = strict)."""
    m = (mode or "review").strip().lower()
    if m in ("auto", "post", "strict", "export"):
        return AUTO_TUNING
    return DEFAULT_TUNING


def tuning_for_content_shape(
    base: SelectionTuning,
    shape: Optional[str],
    *,
    enabled: bool,
) -> SelectionTuning:
    """Apply shape-specific overrides when the kill switch is on.

    When ``enabled`` is False, returns ``base`` unchanged so production stays
    on wins-only gate bypass regardless of inferred shape.
    """
    from dataclasses import replace

    if not enabled:
        return base
    name = str(shape or "").strip().lower()
    if name == "narrative":
        return replace(
            base,
            allow_narrative_story_bypass=True,
            content_anchor_max=max(int(base.content_anchor_max), 6),
            content_shape="narrative",
        )
    if name == "chaos_comedy":
        return replace(base, content_shape="chaos_comedy")
    if name == "competitive_mixed":
        return replace(base, content_shape="competitive_mixed")
    return replace(base, content_shape=name or None)


# Backward-compatible constants for existing tests/imports.
NMS_OVERLAP = DEFAULT_TUNING.nms_overlap
ORDINARY_MIN_PEAK = DEFAULT_TUNING.ordinary_min_peak
ORDINARY_MIN_SELECTION_SCORE = DEFAULT_TUNING.ordinary_min_selection_score
ORDINARY_MIN_SIGNAL_AGREEMENT = DEFAULT_TUNING.ordinary_min_signal_agreement
ORDINARY_MAX_DEAD_AIR = DEFAULT_TUNING.ordinary_max_dead_air


def quality_bar(score: np.ndarray, k: float) -> float:
    """Relative bar: median(R) + k*MAD(R) across the whole VOD."""
    if score.size == 0:
        return 0.0
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    return median + k * (mad * 1.4826)


def dead_air_ratio_of(c: dict) -> float:
    """Read ``dead_air_ratio`` null-safely without inverting a clean 0.0.

    The field is a 0..1 fraction where 0.0 means no dead air at all — the best
    possible value. The obvious ``value or 1.0`` guard is therefore wrong: it
    rewrites the cleanest candidates into the worst-case band, and 10.8% of
    persisted trace rows (5,159 of 47,939) sit at exactly 0.0. Only an explicit
    null falls back.
    """
    raw = c.get("dead_air_ratio", 1.0)
    return 1.0 if raw is None else float(raw)


def compute_hook_score(candidate: dict, tuning: Optional[SelectionTuning] = None) -> float:
    """Return a stable 0..1 first-five-second hook score.

    R(t) and raw chat intensity are unbounded, so both use saturating
    transforms; burst is already 0..1. When the semantic judge ran, its
    cold-viewer read leads (``tuning.semantic_hook_lead``, historically 0.70)
    while the observed signals keep the score grounded.
    """
    tuning = tuning or DEFAULT_TUNING
    energy = max(0.0, float(candidate.get("hook_reaction_energy", 0.0) or 0.0))
    burst = max(0.0, min(1.0, float(candidate.get("hook_burst", 0.0) or 0.0)))
    chat = max(0.0, float(candidate.get("hook_chat", 0.0) or 0.0))

    energy_unit = energy / (energy + 1.0) if energy else 0.0
    chat_unit = chat / (chat + 2.5) if chat else 0.0
    fallback = 0.68 * energy_unit + 0.20 * burst + 0.12 * chat_unit

    semantic = candidate.get("semantic_hook")
    lead = max(0.0, min(1.0, float(tuning.semantic_hook_lead)))
    if semantic is not None and lead > 0.0:
        semantic_unit = max(0.0, min(1.0, float(semantic)))
        return max(0.0, min(1.0, lead * semantic_unit + (1.0 - lead) * fallback))
    return max(0.0, min(1.0, fallback))


def _overlap_fraction(a: dict, b: dict) -> float:
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if inter <= 0:
        return 0.0
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return inter / shorter if shorter > 0 else 0.0


def _interval_gap_sec(a: dict, b: dict) -> float:
    """Seconds of separation between two windows (0 when they touch/overlap)."""
    if a["end"] < b["start"]:
        return float(b["start"] - a["end"])
    if b["end"] < a["start"]:
        return float(a["start"] - b["end"])
    return 0.0


def _absorb_nearby_moment(keeper: dict, other: dict) -> None:
    """Expand ``keeper`` to cover ``other`` and keep the stronger peak."""
    keeper["start"] = round(min(float(keeper["start"]), float(other["start"])), 3)
    keeper["end"] = round(max(float(keeper["end"]), float(other["end"])), 3)
    if float(other.get("peak_value", 0.0) or 0.0) > float(
        keeper.get("peak_value", 0.0) or 0.0
    ):
        keeper["peak_timestamp"] = other.get("peak_timestamp", keeper.get("peak_timestamp"))
        keeper["peak_value"] = other.get("peak_value", keeper.get("peak_value"))
    # Prefer a concrete gameplay label when absorbing a labeled sibling.
    if not keeper.get("game_label") and other.get("game_label"):
        keeper["game_label"] = other.get("game_label")
        keeper["game_evidence"] = max(
            float(keeper.get("game_evidence", 0.0) or 0.0),
            float(other.get("game_evidence", 0.0) or 0.0),
        )
    absorbed = list(keeper.get("nearby_absorbed") or [])
    absorbed.append({
        "start": float(other.get("start", 0.0)),
        "end": float(other.get("end", 0.0)),
        "peak_timestamp": float(other.get("peak_timestamp", 0.0) or 0.0),
    })
    keeper["nearby_absorbed"] = absorbed


def _nearby_moment_pair(
    a: dict,
    b: dict,
    *,
    overlap_thresh: float,
    gap_sec: float,
    max_span_sec: float,
) -> bool:
    """True when two windows are the same moment and safe to collapse."""
    lo = min(float(a["start"]), float(b["start"]))
    hi = max(float(a["end"]), float(b["end"]))
    if hi - lo > max_span_sec + 1e-9:
        return False
    if _overlap_fraction(a, b) > overlap_thresh:
        return True
    return _interval_gap_sec(a, b) <= gap_sec


def _moment_token_overlap(
    a: dict,
    b: dict,
    tuning: SelectionTuning,
) -> bool:
    left = set(a.get("moment_token_hashes") or [])
    right = set(b.get("moment_token_hashes") or [])
    shared = left & right
    if len(shared) < max(1, int(tuning.moment_token_min_shared)):
        return False
    union = left | right
    return bool(union) and len(shared) / len(union) >= tuning.moment_token_jaccard


def _token_moment_pair(a: dict, b: dict, tuning: SelectionTuning) -> bool:
    lo = min(float(a["start"]), float(b["start"]))
    hi = max(float(a["end"]), float(b["end"]))
    return (
        _interval_gap_sec(a, b) <= tuning.moment_token_max_gap_sec
        and hi - lo <= tuning.moment_identity_max_span_sec
        and _moment_token_overlap(a, b, tuning)
    )


def _same_moment_pair(
    a: dict,
    b: dict,
    tuning: SelectionTuning,
) -> bool:
    """Identity-aware duplicate test over the entire qualified pool."""
    lo = min(float(a["start"]), float(b["start"]))
    hi = max(float(a["end"]), float(b["end"]))
    if hi - lo > max(
        tuning.nearby_moment_max_span_sec,
        tuning.moment_identity_max_span_sec,
    ):
        return False
    if _nearby_moment_pair(
        a,
        b,
        overlap_thresh=tuning.nms_overlap,
        gap_sec=tuning.nearby_moment_gap_sec,
        max_span_sec=tuning.nearby_moment_max_span_sec,
    ):
        return True

    a_anchor = a.get("game_anchor_time")
    b_anchor = b.get("game_anchor_time")
    if a_anchor is not None and b_anchor is not None:
        same_label = (
            not a.get("game_label")
            or not b.get("game_label")
            or a.get("game_label") == b.get("game_label")
        )
        if same_label and abs(float(a_anchor) - float(b_anchor)) <= (
            tuning.moment_anchor_cluster_sec
        ):
            return True

    # Alternate windows around a spoken story often do not overlap and may
    # have different reaction peaks. The hashed token signature is the stable
    # identity cue; bound it in time so recurring catchphrases do not merge.
    return _token_moment_pair(a, b, tuning)


def _moment_group_id(candidate: dict) -> str:
    anchor = candidate.get("game_anchor_time")
    if anchor is None:
        anchor = candidate.get("peak_timestamp", candidate.get("start", 0.0))
    return f"moment_{int(round(float(anchor or 0.0) * 1000.0)):012d}"


def _assign_moment_groups(
    ranked: List[dict],
    tuning: SelectionTuning,
) -> List[dict]:
    """Assign connected moment identities before any deck ceiling is applied."""
    if not tuning.editorial_hygiene_enabled:
        for index, candidate in enumerate(ranked):
            candidate["moment_group_id"] = (
                f"{_moment_group_id(candidate)}_{index:04d}"
            )
            candidate["moment_group_source"] = "disabled"
        return ranked

    group_members: List[List[dict]] = []
    used_group_ids: set[str] = set()
    for candidate in ranked:
        group_index = next(
            (
                index
                for index, members in enumerate(group_members)
                if any(_same_moment_pair(candidate, member, tuning) for member in members)
            ),
            None,
        )
        if group_index is None:
            base_group_id = _moment_group_id(candidate)
            group_id = base_group_id
            suffix = 1
            while group_id in used_group_ids:
                group_id = f"{base_group_id}_{suffix:04d}"
                suffix += 1
            candidate["moment_group_id"] = group_id
            candidate["moment_group_source"] = "seed"
            used_group_ids.add(group_id)
            group_members.append([candidate])
            continue

        members = group_members[group_index]
        candidate["moment_group_id"] = members[0]["moment_group_id"]
        candidate["moment_group_source"] = (
            "tokens"
            if any(_token_moment_pair(candidate, member, tuning) for member in members)
            else "temporal"
        )
        members.append(candidate)
    return ranked


def _moment_representative_key(c: dict, tuning: SelectionTuning):
    """Prefer the window that contains the moment's strongest direct evidence.

    Candidate generation deliberately creates several crops around a peak. A
    later crop can receive a higher blended score while omitting the elim,
    face spike, or spoken payoff that made the moment worth clipping. Moment
    identity therefore chooses its representative on evidence first and uses
    rank only as a tie-breaker.
    """
    return (
        bool(_creator_protected(c)),
        bool(c.get("visual_win_anchored")),
        bool(_landed_game_event(c, tuning)),
        bool(_exceptional_face_spike(c, tuning)),
        bool(_strong_startle(c, tuning)),
        bool(_visible_crowd_anchor(c, tuning)),
        bool(_grounded_verbal_beat(c, tuning)),
        float(c.get("signal_selection_score", 0.0) or 0.0),
        float(c.get("selection_score", 0.0) or 0.0),
        -float(c.get("start", 0.0) or 0.0),
    )


def _carry_moment_crowd_evidence(
    representative: dict, members: List[dict], tuning: SelectionTuning,
) -> None:
    """Move a moment's viewer !clip evidence onto the crop that represents it.

    A viewer types !clip at the MOMENT, not at one crop of it. Candidate
    generation makes several windows around the same peak, the burst tags
    whichever one contains the reaction peak, and _prefer_moment_representatives
    then hands the deck slot to a different crop chosen on evidence. The
    ``protected`` flag used to stay behind on the demoted alternate, so the
    representative entered the deck unprotected -- and unprotected is exactly
    what _rebalance_temporal_blackouts looks for when it picks an eviction
    victim, while the tagged crop was already suppressed as a moment_duplicate.
    The moment the audience explicitly asked for then shipped in NEITHER crop.

    Candidate swaps must preserve protection across alternate crops of the same
    moment; otherwise deduplication and temporal balancing can remove both.

    This carries EVIDENCE, not score. selection_score is frozen well before this
    point, so a command still cannot outrank anything it could not outrank
    before -- the boost just stops being dropped when the representative changes.
    """
    crowd = max(
        (int(member.get("crowd_clip", 0) or 0) for member in members),
        default=0,
    )
    if crowd <= 0:
        return
    representative["crowd_clip"] = max(
        int(representative.get("crowd_clip", 0) or 0), crowd,
    )
    representative["crowd_recommended"] = True
    # Deliberately routed through ``protected`` rather than a fresh crowd_clip
    # exemption in _rebalance_temporal_blackouts: ``protected`` is the flag
    # crowd_command_deck_fraction demotes, so a command past that cap keeps
    # competing like an ordinary candidate instead of becoming unevictable.
    # Protection that a cap cannot reach is how !clip starts overriding
    # everything.
    if tuning.viewer_commands_protected:
        representative["protected"] = True
    # Only one crop of a moment can ship, so the alternates hand their claim
    # over instead of each holding a slot in the protected sort tier. Their
    # crowd_clip stays put for trace provenance.
    for member in members:
        if member is representative:
            continue
        if int(member.get("crowd_clip", 0) or 0) <= 0:
            continue
        member["crowd_evidence_carried_to"] = representative.get("moment_group_id")
        if not _creator_protected(member):
            member["protected"] = False


def _prefer_moment_representatives(
    ranked: List[dict], tuning: SelectionTuning,
) -> List[dict]:
    """Put each moment's best evidence-bearing crop in its first rank slot."""
    first_positions: dict[str, int] = {}
    members: dict[str, List[tuple[int, dict]]] = {}
    for index, candidate in enumerate(ranked):
        group_id = str(candidate.get("moment_group_id") or f"ungrouped_{index}")
        first_positions.setdefault(group_id, index)
        members.setdefault(group_id, []).append((index, candidate))

    reordered = list(ranked)
    for group_id, grouped in members.items():
        if len(grouped) < 2:
            continue
        first_index = first_positions[group_id]
        best_index, best = max(
            grouped,
            key=lambda item: _moment_representative_key(item[1], tuning),
        )
        # Before the swap decision: the representative carries the moment's
        # viewer evidence whether or not its position actually changes.
        _carry_moment_crowd_evidence(
            best, [member for _, member in grouped], tuning,
        )
        if best_index == first_index:
            continue
        previous = reordered[first_index]
        reordered[first_index] = best
        reordered[best_index] = previous
        best["moment_group_source"] = f"{best.get('moment_group_source', 'seed')}:representative"
        previous["moment_group_source"] = f"{previous.get('moment_group_source', 'temporal')}:alternate"
    return reordered


# The fractional agreement gate converts to a channel COUNT over a capped
# denominator. Uncapped, every extra track a VOD happens to have (chat, burst,
# motion...) silently raised the bar: 0.5 agreement means 2-of-4 on a lean
# capture but 3-of-6 on a fully-tracked Twitch VOD — so a genuine voice+face
# reaction in a horror VOD (chat quiet, no bursts, slow game = no motion) was
# structurally gated while the identical moment on a 4-channel VOD passed.
# Real-VOD case: SH2 boss-kill reaction, R(t) peak above two KEPT clips,
# rejected purely because the VOD had 6 active channels.
AGREEMENT_DENOM_CAP = 4


def _modality_gate(c: dict, min_agreement: float, tuning: SelectionTuning, is_event: bool) -> bool:
    """Ordinary clips need multi-modal agreement; events need a human channel."""
    active = int(round(float(c.get("active_channel_count", 1.0) or 1.0)))
    active = max(1, active)
    modality_count = int(c.get("modality_count", 0) or 0)
    if modality_count <= 0:
        # Fallback when older candidates lack the field: treat agreement as a
        # fraction of active channels (round half UP: banker's rounding would
        # turn agreement 0.5 on a single-channel VOD into zero modalities).
        agreement = float(c.get("signal_agreement", 0.0) or 0.0)
        modality_count = int(agreement * active + 0.5)
    effective = min(active, AGREEMENT_DENOM_CAP)
    required = int(np.ceil(min_agreement * effective - 1e-9))
    if is_event:
        # Confirmed OCR payoff: at least one human channel, more when the
        # tuning asks for it.
        return modality_count >= max(1, required)
    # Absolute modality floor, capped by how many tracks this VOD actually has
    # so a voice-only capture is not structurally impossible.
    need = min(tuning.ordinary_min_modalities, active)
    return modality_count >= max(need, required)


def _score_components(c: dict, tuning: SelectionTuning, game_bonus: float):
    """Signal selection score plus the derived values the gates read."""
    duration = max(1.0, c.get("end", 0.0) - c.get("start", 0.0))
    mean_intensity = c.get("reaction_auc", 0.0) / duration
    density = float(c.get("event_density", 1.0) or 1.0)
    signal_score = (
        1.5 * c.get("peak_value", 0.0)
        + mean_intensity
        # mean_intensity is reaction_auc/duration, i.e. average HEIGHT. That
        # divides out the area, and area is the best predictor of this
        # creator's Keep anywhere in the trace (reaction_auc, AUC 0.656). A
        # 14s laugh and a 2s chuckle of equal loudness score identically here
        # despite 7x the evidence. This term adds the undivided area back.
        # 0.0 keeps the historical behaviour exactly.
        + tuning.reaction_area_weight * float(c.get("reaction_auc", 0.0) or 0.0)
        + game_bonus * c.get("game_evidence", 0.0)
        + tuning.event_density_weight * max(0.0, density - 1.0)
    )
    # Applied uniformly regardless of game evidence: dead air is a quality
    # signal on its own (a clip that's mostly silence, whether or not OCR
    # fired), not something that should single out non-HUD-game clips.
    signal_score -= 0.35 * max(0.0, dead_air_ratio_of(c) - 0.35)
    signal_score += tuning.duration_prior_weight * _duration_prior(duration)
    signal_score += tuning.payoff_position_weight * _payoff_position_bonus(c)
    if c.get("floor_padded"):
        signal_score -= tuning.floor_padded_penalty
    if c.get("scene_label") in ("lobby", "intermission"):
        signal_score -= tuning.lobby_score_penalty
    crowd = int(c.get("crowd_clip", 0) or 0)
    if crowd > 0:
        signal_score += tuning.crowd_clip_bonus * (min(crowd, 3) / 3.0)
    startle = float(
        (c.get("modality_breakdown") or {}).get("startle", 0.0) or 0.0
    )
    signal_score += tuning.startle_bonus_weight * startle
    signal_score += tuning.visible_crowd_bonus_weight * float(
        c.get("visible_crowd_reaction", 0.0) or 0.0
    )
    return signal_score, mean_intensity, duration


def _semantic_rescue_allowed(c: dict, tuning: SelectionTuning) -> bool:
    """Return whether the local editor supplied strong positive consensus."""
    if not tuning.semantic_gate_authority:
        return False
    return (
        str(c.get("semantic_verdict") or "").strip().lower() == "post"
        and str(c.get("semantic_moment_type") or "").strip().lower() != "filler"
        and float(c.get("semantic_hook", 0.0) or 0.0) >= tuning.semantic_rescue_min_hook
        and float(c.get("semantic_payoff", 0.0) or 0.0) >= tuning.semantic_rescue_min_payoff
        and float(c.get("semantic_self_contained", 0.0) or 0.0)
        >= tuning.semantic_rescue_min_self_contained
    )


# Deterministic validation of a claimed on-screen win. The VLM transcribes
# banner text faithfully but misapplies the "#N is not a win" rule (observed:
# "YOU PLACED #92" and "YOUR TEAM PLACED #8" both confirmed as match_win), so
# the win claim is only accepted when the transcription itself reads like a
# victory and not like a placement.
import re as _re

_WIN_TEXT_RE = _re.compile(r"victor|champion|\bwinner\b|#\s*1\b|\b1st\b", _re.I)
_PLACE_VETO_RE = _re.compile(
    r"placed|defeat|eliminated|#\s*(?:[2-9]\b|\d{2,})|\b(?:[2-9]|\d{2,})(?:th|nd|rd)\b",
    _re.I,
)


def _visual_verdict(c: dict) -> str:
    """Normalized visual-judge verdict (full or trace-abbreviated field)."""
    raw = c.get("visual_semantic_verdict")
    if raw is None:
        raw = c.get("visual_verdict")
    return str(raw or "").strip().lower()


def _visual_outcome(c: dict) -> str:
    raw = c.get("visual_semantic_outcome")
    if raw is None:
        raw = c.get("visual_outcome")
    return str(raw or "").strip().lower()


def _strong_visual_skip_conflict(c: dict, tuning: SelectionTuning) -> bool:
    """Whether one unstable visual skip should fall back to normal gates.

    This does not positively rescue or score the candidate.  It only removes
    the categorical ``visual_skip`` veto; every ordinary evidence, quality,
    dedupe, and deck-budget rule still applies.  Requiring ``ranker_score``
    keeps the exception owned by the frozen base stack and unavailable to
    signal-only installs or personal preference alone.
    """
    semantic_verdict = str(c.get("semantic_verdict") or "").strip().lower()
    semantic_type = str(c.get("semantic_moment_type") or "").strip().lower()
    visual_outcome = _visual_outcome(c)
    if (
        semantic_verdict != "post"
        or semantic_type in ("", "filler")
        or visual_outcome not in ("", "none")
        or c.get("ranker_score") is None
    ):
        return False
    return (
        float(c.get("semantic_payoff", 0.0) or 0.0)
        >= tuning.visual_skip_conflict_min_payoff
        and float(c.get("signal_agreement", 0.0) or 0.0)
        >= tuning.visual_skip_conflict_min_signal_agreement
        and float(c.get("peak_value", 0.0) or 0.0)
        >= tuning.visual_skip_conflict_min_peak
        and float(c.get("ranker_score", 0.0) or 0.0)
        >= tuning.visual_skip_conflict_min_ranker_score
        and float(c.get("routine_non_gameplay_ratio", 0.0) or 0.0)
        < tuning.visual_skip_conflict_max_routine_ratio
    )


def _visual_moment_type(c: dict) -> str:
    raw = c.get("visual_semantic_moment_type")
    if raw is None:
        raw = c.get("visual_moment_type")
    return str(raw or "").strip().lower()


def _generic_visual_filler(c: dict) -> bool:
    """Conservative visual-editor contradiction for generic/non-HUD VODs."""
    profile = str(c.get("game_profile") or "").strip().lower()
    return (
        profile == "generic"
        and not c.get("game_label")
        and _visual_verdict(c) == "maybe"
        and _visual_moment_type(c) == "filler"
        # A grounded outcome is not generic filler even when the small visual
        # editor contradicts itself on the type.  A newly labelled GAME OVER
        # Keep exposed this hole in the original narrow veto.
        and _visual_outcome(c) in ("", "none")
    )


_ROUTINE_NON_GAMEPLAY_RE = _re.compile(
    r"\b(?:"
    r"loading|connecting|waiting\s+for\s+players|matchmaking|"
    r"finding\s+match|searching\s+for\s+match|main\s+menu"
    r")\b",
    _re.I,
)


def _visual_model_verdict(c: dict) -> str:
    raw = c.get("visual_semantic_model_verdict")
    if raw is None:
        raw = c.get("visual_model_verdict")
    return str(raw or "").strip().lower()


def _routine_non_gameplay_consensus(c: dict) -> bool:
    """True only for grounded waiting/menu state plus dual-judge rejection."""
    scene = str(c.get("scene_label") or "").strip().lower()
    onscreen = c.get("visual_semantic_onscreen_text")
    if onscreen is None:
        onscreen = c.get("visual_onscreen_text")
    non_gameplay = (
        scene in ("lobby", "intermission")
        or _visual_outcome(c) == "menu_or_shop"
        or bool(_ROUTINE_NON_GAMEPLAY_RE.search(str(onscreen or "")))
    )
    return (
        non_gameplay
        and _visual_model_verdict(c) == "skip"
        and str(c.get("semantic_verdict") or "").strip().lower() == "skip"
        and str(c.get("semantic_moment_type") or "").strip().lower() == "filler"
    )


def _visual_endorsement(c: dict) -> str:
    """The calibrated visual verdict, if the judge reached this candidate."""
    raw = c.get("visual_semantic_verdict")
    if raw is None:
        raw = c.get("visual_verdict")
    return str(raw or "").strip().lower()


def _routine_non_gameplay_ocr(c: dict, tuning: SelectionTuning) -> bool:
    """Deterministic lobby reject that never needs the visual judge.

    ``_routine_non_gameplay_consensus`` requires a visual model verdict, so on
    a VOD the judge never reached (measured 2026-07-27: one fixture had 15/15
    deck clips unjudged) it can never fire. This path keys only on the
    signal-track OCR that every scan produces, so coverage cannot disable it.

    Both judges keep their veto power: an explicit visual or text endorsement
    still wins, which is what preserves a lobby joke the model recognized as a
    real moment. What it removes is the unjudged waiting-screen filler.
    """
    ratio = float(c.get("routine_non_gameplay_ratio", 0.0) or 0.0)
    if ratio < tuning.routine_non_gameplay_ocr_ratio:
        return False
    if _visual_endorsement(c) in ("post", "maybe"):
        return False
    return str(c.get("semantic_verdict") or "").strip().lower() != "post"


def _verbal_beat_complete(c: dict, tuning: SelectionTuning) -> bool:
    """Complete verbal beat using the same floors as content_anchor."""
    conf = float(c.get("visual_semantic_confidence", 0.0) or 0.0)
    if conf < tuning.content_min_confidence:
        return False
    return (
        bool(c.get("visual_semantic_verbal_payoff"))
        and bool(c.get("visual_semantic_context_complete"))
        and not bool(c.get("visual_semantic_routine_only"))
        and float(c.get("visual_semantic_payoff", 0.0) or 0.0)
        >= tuning.content_min_payoff
        and float(c.get("visual_semantic_self_contained", 0.0) or 0.0)
        >= tuning.content_min_self_contained
    )


def _narrative_story_bypass_eligible(c: dict, tuning: SelectionTuning) -> bool:
    """Quiet story/verbal beats that may bypass arousal under narrative shape."""
    if not tuning.allow_narrative_story_bypass:
        return False
    if c.get("content_anchor_capped"):
        return False
    if _is_chat_reaction_only(c, tuning):
        return False
    if _visual_verdict(c) not in ("post", "maybe"):
        return False
    # Calibrated death/fail story cards (RIP banners, executions) are climaxes
    # even when the VLM typed them as fail rather than story.
    if _visual_outcome(c) == "death_or_fail":
        return True
    if _visual_moment_type(c) == "story" and _verbal_beat_complete(c, tuning):
        return True
    if _visual_moment_type(c) == "story":
        # Story-typed moments with context + non-routine still qualify when
        # editorial floats are present but slightly soft.
        if bool(c.get("visual_semantic_routine_only")):
            return False
        if not bool(c.get("visual_semantic_context_complete")):
            return False
        conf = float(c.get("visual_semantic_confidence", 0.0) or 0.0)
        payoff = float(c.get("visual_semantic_payoff", 0.0) or 0.0)
        # Soft floor is content_min_payoff - 0.10 (min 0.40). Round so
        # binary float noise on 0.55-0.10 cannot reject a true 0.45 payoff.
        soft_payoff = round(max(0.40, tuning.content_min_payoff - 0.10), 4)
        return (
            conf >= tuning.content_min_confidence
            and payoff + 1e-9 >= soft_payoff
        )
    return _verbal_beat_complete(c, tuning)


def _is_chat_reaction_only(c: dict, tuning: SelectionTuning) -> bool:
    """Streamer talking to chat: voice up, face cold, no gameplay outcome.

    Measured on the 2/30 creator deck — visual ``post`` + outcome ``none`` +
    negative face dominated the top ranks while gameplay was absent.
    """
    # Only demote moments the visual judge endorsed; unjudged / skip paths
    # are handled elsewhere and must not starve ordinary reaction scoring.
    if _visual_verdict(c) not in ("post", "maybe"):
        return False
    if _visual_win_read(c):
        return False
    if c.get("game_label") in (
        "elimination", "knock", "rank_progress", "terminal_win",
    ):
        return False
    outcome = _visual_outcome(c)
    if outcome in ("match_win", "elimination", "death_or_fail"):
        return False
    # Visible story action / death banners are not chat-thanks (Detroit
    # Connor-class climaxes often have a cold facecam).
    if bool(c.get("visual_semantic_visual_event")):
        return False
    onscreen = c.get("visual_semantic_onscreen_text")
    if onscreen is None:
        onscreen = c.get("visual_onscreen_text")
    if onscreen and _re.search(
        r"\b(?:rip\b|game\s*over|you\s+died|chapter\b|ending\b)",
        str(onscreen),
        _re.I,
    ):
        return False
    if _strong_startle(c, tuning) or _visible_crowd_anchor(c, tuning):
        return False
    mb = c.get("modality_breakdown") or {}
    face = _face_arousal_max(c)
    voice = float(mb.get("voice", 0.0) or 0.0)
    burst = float(mb.get("burst", 0.0) or 0.0)
    loud_onset = float(mb.get("loud_onset", 0.0) or 0.0)
    if voice < tuning.chat_reaction_min_voice:
        return False
    # A complete spoken beat plus a visible, moderate expression is grounded
    # content. Keep the 0.22 chat ceiling for ambient facecam, but do not turn
    # the new peak policy into a veto for real narrative beats around 0.15-0.21.
    if _grounded_verbal_beat(c, tuning) and face >= 0.15:
        return False
    if face >= tuning.chat_reaction_max_face:
        return False
    # Background motion and game audio are common under ordinary commentary;
    # they describe the stream layout, not whether the creator had a moment.
    # A real burst/onset still provides independent reaction evidence.
    if burst >= 0.35 or loud_onset >= 0.35:
        return False
    return True


def onscreen_text_confirms_win(text) -> bool:
    t = str(text or "")
    return bool(_WIN_TEXT_RE.search(t)) and not _PLACE_VETO_RE.search(t)


def _visual_win_read(c: dict) -> bool:
    if _visual_outcome(c) != "match_win":
        return False
    text = c.get("visual_semantic_onscreen_text")
    if text is None:
        text = c.get("visual_onscreen_text")
    return onscreen_text_confirms_win(text)


def content_interest_score(c: dict, tuning: SelectionTuning) -> float:
    """0..1 grounded content-interest from visual-judge fields (0 when off/absent).

    Two evidence routes, mirroring how a human editor reads a moment:
      * a VISIBLE game outcome (win banner, elimination, death screen) — pure
        perception the VLM reads from frames; deliberately NOT scaled by the
        4B's editorial payoff numbers, which are unreliable (a confirmed
        VICTORY ROYALE banner scored payoff 0.1 in testing);
      * a complete VERBAL beat (story/joke that resolves, followable cold) —
        here the editorial floors apply because the claim is editorial.
    """
    verdict = _visual_verdict(c)
    if not tuning.content_axis or not verdict or verdict == "skip":
        return 0.0
    # The outcome path is NOT gated on the model's self-reported confidence:
    # the enum read is grounded perception (and sweep-injected wins are
    # double-attested — sweep frame + confirming judge), while the 4B's
    # confidence float is unreliable (real confirmed wins reported 0.0).
    outcome = _visual_outcome(c)
    if outcome == "match_win" and not _visual_win_read(c):
        outcome = "none"
    outcome_evidence = {
        "match_win": 1.0,
        "elimination": 0.6,
        "death_or_fail": 0.5,
    }.get(outcome, 0.0)
    verbal_evidence = 0.0
    conf = float(c.get("visual_semantic_confidence", 0.0) or 0.0)
    if (
        conf >= tuning.content_min_confidence
        and bool(c.get("visual_semantic_verbal_payoff"))
        and bool(c.get("visual_semantic_context_complete"))
        and not bool(c.get("visual_semantic_routine_only"))
    ):
        payoff = float(c.get("visual_semantic_payoff", 0.0) or 0.0)
        self_contained = float(c.get("visual_semantic_self_contained", 0.0) or 0.0)
        if (
            payoff >= tuning.content_min_payoff
            and self_contained >= tuning.content_min_self_contained
        ):
            verbal_evidence = payoff * (0.60 + 0.40 * self_contained)
    # Cold-face chat talk must not earn verbal content lift.
    if verbal_evidence > 0.0 and _is_chat_reaction_only(c, tuning):
        verbal_evidence = 0.0
    return max(outcome_evidence, verbal_evidence)


def _content_gate_bypass(c: dict, tuning: SelectionTuning) -> bool:
    """Per-candidate arousal-gate bypass.

    Default: validated visual wins only. Under narrative content-shape policy
    (``allow_narrative_story_bypass``), complete story/verbal beats that are
    not chat-only may also bypass. Chat-thanks pollution (0/18, 2/30) stays
    blocked via ``_is_chat_reaction_only``.
    """
    if not tuning.content_axis or not _visual_verdict(c):
        return False
    # Two independent judges seeing the same causal window and agreeing on a
    # complete, high-payoff POST may correct coarse per-arc heuristics in any
    # content shape. This is intentionally much narrower than restoring broad
    # semantic authority: one judge, a MAYBE, or an ungrounded opinion cannot
    # bypass anything.
    if _grounded_judge_consensus(c, tuning):
        return True
    if _visual_win_read(c):
        # A confirmed win still needs the reaction to bypass gates (see
        # _event_reaction_landed): a dead-calm VICTORY ROYALE is a creator
        # pass. This is the SAME requirement _visual_floor_keep applies; both
        # bypass paths must agree or the calm win slips through the other one.
        return _event_reaction_landed(c, tuning)
    return _narrative_story_bypass_eligible(c, tuning)


def _labeled_story_floor_keep(c: dict, tuning: SelectionTuning) -> bool:
    """Calibrated maybe/story (narrative only) may keep adaptive/event floors.

    Does not lower global floor fractions — Fortnite/competitive decks stay on
    the ordinary publishability bars. Mirrors narrative gate-bypass eligibility
    so a climax that clears arousal gates is not discarded at the deck floor.
    """
    return _narrative_story_bypass_eligible(c, tuning)


def _visual_floor_keep(c: dict, tuning: SelectionTuning, floor: float) -> Optional[bool]:
    """Verdict-based floor decision, or None to fall back to the arousal floor.

    Returns None for candidates the visual judge never saw — those keep the
    existing arousal behavior exactly, which is what makes this safe to enable
    on installs where the VLM only covers part of the candidate pool.
    """
    verdict = _visual_verdict(c)
    if not verdict:
        return None  # never judged -> arousal floor decides, as today
    # A routine game event does not earn the verdict bypass. The VLM types a
    # kill/win banner "post" because it IS a recognizable event -- but the
    # creator only wants the hectic ones, and startle/peak are what separate
    # those (measured 2026-07-23: two calm terminal_win clips scoring <2.3
    # were the floor-bypassed passes on the sprite deck; blocking them cost 0
    # keeps and left channel_a untouched). Confirmed match wins use the gentler
    # reaction bar in _event_reaction_landed. A non-event "post" is unaffected.
    if not _event_reaction_landed(c, tuning):
        return float(c.get("signal_selection_score", 0.0) or 0.0) >= floor
    if verdict == "post":
        # POST is a model opinion, not creator approval. When the grounded
        # policy is enabled it may waive the floor only when another concrete
        # signal says a publishable beat actually happened. Otherwise the
        # candidate simply falls back to the same floor as an unjudged moment;
        # this is deliberately not a lobby/gameplay veto.
        if not _grounded_visual_post(c, tuning):
            return float(c.get("signal_selection_score", 0.0) or 0.0) >= floor
        return True
    if verdict == "maybe":
        # MAYBE only earns the softer floor when concrete evidence grounds the
        # read. A bare maybe/funny label is uncertainty, not a free pass.
        fraction = (
            max(0.0, tuning.visual_floor_maybe_fraction)
            if _grounded_visual_post(c, tuning)
            else 1.0
        )
        return float(c.get("signal_selection_score", 0.0) or 0.0) >= floor * fraction
    return False  # skip; normally already vetoed at the evidence gate


def _content_anchor(c: dict, tuning: SelectionTuning) -> bool:
    """Grounded content evidence for rank lift / telemetry / floor keep.

    Validated MATCH WINs and complete verbal beats both qualify as anchors for
    scoring. Under narrative content-shape policy, calibrated maybe/story
    climaxes (including death/fail story cards) also qualify so they keep
    adaptive/event-rich deck slots — without lowering floors globally.
    Gate bypass remains wins-only unless narrative story policy is on —
    see ``_content_gate_bypass``.
    """
    if not tuning.content_axis or not _visual_verdict(c):
        return False
    if c.get("content_anchor_capped"):
        return False
    if _visual_win_read(c):
        # Anchors earn a floor bypass, so a calm win must clear the reaction
        # bar here too -- otherwise it re-enters via content_anchor after
        # _visual_floor_keep correctly blocked it (reviewer-found, 2026-07-23).
        return _event_reaction_landed(c, tuning)
    if _is_chat_reaction_only(c, tuning):
        return False
    if _verbal_beat_complete(c, tuning):
        return True
    return _labeled_story_floor_keep(c, tuning)


def _strong_startle(c: dict, tuning: SelectionTuning) -> bool:
    return float(
        (c.get("modality_breakdown") or {}).get("startle", 0.0) or 0.0
    ) >= tuning.event_rich_startle_exemption


def _face_arousal_max(c: dict) -> float:
    """Peak in-window face arousal (falls back to mean face when max absent)."""
    mb = c.get("modality_breakdown") or {}
    if "face_max" in mb and mb.get("face_max") is not None:
        return float(mb.get("face_max") or 0.0)
    if c.get("face_arousal_max") is not None:
        return float(c.get("face_arousal_max") or 0.0)
    return float(mb.get("face", 0.0) or 0.0)


def _calibrated_face_spike(c: dict, tuning: SelectionTuning) -> bool:
    """Rare, sustained face reaction relative to this creator/VOD.

    The HSEmotion arousal head is raw model output and its baseline varies
    materially across creators. A global 0.40 cutoff made some decks entirely
    face-priority clips, so live generation persists the VOD p99, local delta,
    and sustained-frame count used here. Missing calibration fails closed.
    """
    mb = c.get("modality_breakdown") or {}
    vod_p99 = mb.get("face_vod_p99")
    sustained = mb.get("face_spike_max_run")
    local_delta = mb.get("face_peak_delta")
    if vod_p99 is None or sustained is None or local_delta is None:
        return False
    threshold = max(tuning.face_spike_absolute_min, float(vod_p99 or 0.0))
    return (
        _face_arousal_max(c) >= threshold
        and int(sustained or 0) >= tuning.face_spike_min_sustained_frames
        and float(local_delta or 0.0) >= tuning.face_spike_min_local_delta
    )


def _exceptional_face_spike(c: dict, tuning: SelectionTuning) -> bool:
    """Calibrated face spike within the bounded per-VOD rescue budget."""
    return bool(
        _calibrated_face_spike(c, tuning)
        and c.get("face_spike_rescue_eligible", True)
    )


def _assign_face_spike_rescue_eligibility(
    candidates: List[dict], tuning: SelectionTuning,
) -> None:
    """Bound rescue authority to the strongest distinct face events per VOD."""
    for candidate in candidates:
        candidate["face_spike_rescue_eligible"] = False

    qualified = [c for c in candidates if _calibrated_face_spike(c, tuning)]
    if not qualified:
        return
    qualified.sort(key=lambda c: float(
        (c.get("modality_breakdown") or {}).get(
            "face_spike_peak_timestamp",
            c.get("peak_timestamp", 0.0),
        ) or 0.0
    ))

    clusters: List[List[dict]] = []
    for candidate in qualified:
        timestamp = float((candidate.get("modality_breakdown") or {}).get(
            "face_spike_peak_timestamp",
            candidate.get("peak_timestamp", 0.0),
        ) or 0.0)
        if clusters:
            previous = float((clusters[-1][-1].get("modality_breakdown") or {}).get(
                "face_spike_peak_timestamp",
                clusters[-1][-1].get("peak_timestamp", 0.0),
            ) or 0.0)
        else:
            previous = None
        if previous is None or timestamp - previous > tuning.face_spike_event_merge_sec:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    def cluster_strength(cluster: List[dict]):
        return max((
            float((c.get("modality_breakdown") or {}).get("face_peak_delta", 0.0) or 0.0),
            int((c.get("modality_breakdown") or {}).get("face_spike_max_run", 0) or 0),
            _face_arousal_max(c),
        ) for c in cluster)

    clusters.sort(key=cluster_strength, reverse=True)
    for cluster in clusters[:max(0, int(tuning.face_spike_rescue_max_events))]:
        for candidate in cluster:
            candidate["face_spike_rescue_eligible"] = True


def _grounded_visual_post(c: dict, tuning: SelectionTuning) -> bool:
    """Whether a visual POST has evidence beyond the judge's opinion."""
    return bool(
        _landed_game_event(c, tuning)
        or _grounded_verbal_beat(c, tuning)
        or _strong_startle(c, tuning)
        or _visible_crowd_anchor(c, tuning)
        or _calibrated_face_spike(c, tuning)
    )


def _grounded_judge_consensus(c: dict, tuning: SelectionTuning) -> bool:
    """Require both judges and grounded evidence for a positive correction."""
    return bool(
        _visual_verdict(c) == "post"
        and _grounded_visual_post(c, tuning)
        and float(c.get("visual_semantic_payoff", 0.0) or 0.0)
        >= tuning.temporal_blackout_min_visual_payoff
        and float(c.get("visual_semantic_self_contained", 0.0) or 0.0)
        >= tuning.temporal_blackout_min_self_contained
        and bool(c.get("visual_semantic_context_complete"))
        and not bool(c.get("visual_semantic_routine_only"))
        and str(c.get("semantic_verdict") or "").strip().lower() == "post"
        and float(c.get("semantic_payoff", 0.0) or 0.0)
        >= tuning.temporal_blackout_min_semantic_payoff
        and float(c.get("semantic_self_contained", 0.0) or 0.0)
        >= tuning.temporal_blackout_min_self_contained
    )


def _rebalance_temporal_blackouts(
    selected: List[dict],
    eligible: List[dict],
    max_clips: int,
    tuning: SelectionTuning,
) -> None:
    """Replace dense weak picks with consensus moments inside long empty spans.

    This deliberately runs after ordinary ranking and the hard cap. It cannot
    increase deck size, cannot rescue an unjudged candidate, and cannot displace
    protected (creator-marked or viewer-!clip) or validated-win clips. Its sole
    job is preventing global score concentration from erasing a strongly
    confirmed beat elsewhere in a long VOD.
    """
    if (
        not tuning.hard_total_cap
        or max_clips <= 0
        or len(selected) < max_clips
        or len(selected) < 2
        or tuning.temporal_blackout_max_rescues <= 0
    ):
        return

    def midpoint(candidate: dict) -> float:
        return 0.5 * (
            float(candidate.get("start", 0.0))
            + float(candidate.get("end", 0.0))
        )

    def nearest_neighbor_distance(candidate: dict, deck: List[dict]) -> float:
        center = midpoint(candidate)
        distances = [
            abs(center - midpoint(other)) for other in deck if other is not candidate
        ]
        return min(distances, default=float("inf"))

    for _ in range(max(0, int(tuning.temporal_blackout_max_rescues))):
        ordered = sorted(selected, key=lambda candidate: float(candidate.get("start", 0.0)))
        selected_span = midpoint(ordered[-1]) - midpoint(ordered[0])
        expected_gap = selected_span / max(1, len(ordered) - 1)
        abnormal_gap = (
            expected_gap * tuning.temporal_blackout_expected_gap_multiplier
        )
        if expected_gap <= 0.0 or abnormal_gap <= 0.0:
            return
        gaps = sorted(
            (
                (
                    float(right.get("start", 0.0)) - float(left.get("end", 0.0)),
                    left,
                    right,
                )
                for left, right in zip(ordered, ordered[1:])
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        repair = None
        for gap_size, left, right in gaps:
            if gap_size < abnormal_gap:
                break
            gap_start = float(left.get("end", 0.0))
            gap_end = float(right.get("start", 0.0))
            # A rescue near either edge does not materially repair the empty
            # span. Scale the clearance to the gap rather than baking in a
            # duration that only makes sense for one VOD length.
            edge_clearance = gap_size * 0.20
            challengers = [
                candidate for candidate in eligible
                if candidate not in selected
                and gap_start + edge_clearance <= midpoint(candidate)
                <= gap_end - edge_clearance
                and _grounded_judge_consensus(candidate, tuning)
            ]
            if not challengers:
                continue
            challenger = max(
                challengers,
                key=lambda candidate: (
                    float(candidate.get("semantic_payoff", 0.0) or 0.0),
                    float(candidate.get("visual_semantic_payoff", 0.0) or 0.0),
                    min(midpoint(candidate) - gap_start, gap_end - midpoint(candidate)),
                    float(candidate.get("selection_score", 0.0) or 0.0),
                ),
            )
            repair = challenger
            break
        if repair is None:
            return

        # Carry viewer protection to the selected representative so changing the crop does
        # not drop an explicitly requested moment.
        removable = [
            candidate for candidate in selected
            if not candidate.get("protected")
            and not candidate.get("visual_win_anchored")
            and candidate.get("game_label") != "terminal_win"
            and nearest_neighbor_distance(candidate, selected) <= expected_gap
        ]
        if not removable:
            return
        victim = min(
            removable,
            key=lambda candidate: (
                _grounded_judge_consensus(candidate, tuning),
                float(candidate.get("selection_score", 0.0) or 0.0),
                nearest_neighbor_distance(candidate, selected),
            ),
        )
        selected.remove(victim)
        victim["selection_disposition"] = "temporal_blackout_rebalanced"
        victim["selection_rejection_reasons"] = ["temporal_blackout_rebalanced"]
        victim["selection_funnel"]["final"] = "temporal_blackout_rebalanced"

        repair["selection_disposition"] = "selected_temporal_rescue"
        repair["selection_rejection_reasons"] = []
        repair["selection_funnel"]["final"] = "selected_temporal_rescue"
        selected.append(repair)


def _grounded_verbal_beat(c: dict, tuning: SelectionTuning) -> bool:
    """A complete spoken story beat, including the calibrated soft boundary.

    The visual judge emits confidence/payoff in coarse steps. A story with
    explicit context and verbal payoff at 0.50 should not lose floor authority
    solely because the generic content payoff threshold is 0.55. This mirrors
    the already-shipped narrative story boundary while still excluding the
    context-free ``funny`` lobby reads that motivated this policy.
    """
    if _verbal_beat_complete(c, tuning):
        return True
    if _visual_moment_type(c) != "story":
        return False
    soft_payoff = round(max(0.40, tuning.content_min_payoff - 0.10), 4)
    return bool(
        float(c.get("visual_semantic_confidence", 0.0) or 0.0)
        >= tuning.content_min_confidence
        and bool(c.get("visual_semantic_verbal_payoff"))
        and bool(c.get("visual_semantic_context_complete"))
        and not bool(c.get("visual_semantic_routine_only"))
        and float(c.get("visual_semantic_payoff", 0.0) or 0.0) + 1e-9
        >= soft_payoff
        and float(c.get("visual_semantic_self_contained", 0.0) or 0.0)
        >= tuning.content_min_self_contained
    )


def _face_spike_rescue_anchor(c: dict, tuning: SelectionTuning) -> bool:
    """Independent context required before a face spike may rescue a floor.

    Face arousal alone can fire on ordinary expressions. The rescue therefore
    needs a detected gameplay event, startle/crowd evidence, or a complete
    spoken beat. A qualifying game event does not need the ordinary peak rule:
    the sustained face spike is itself the missing reaction evidence.
    """
    contextual_game_event = bool(
        c.get("game_label") in _REACTION_EVENT_LABELS
        and float(c.get("game_evidence", 0.0) or 0.0)
        >= tuning.event_min_game_evidence
        and dead_air_ratio_of(c) <= tuning.event_max_dead_air
    )
    return bool(
        contextual_game_event
        or _strong_startle(c, tuning)
        or _visible_crowd_anchor(c, tuning)
        or _grounded_verbal_beat(c, tuning)
    )


_REACTION_EVENT_LABELS = ("elimination", "knock", "rank_progress", "terminal_win")


def _event_reaction_landed(c: dict, tuning: SelectionTuning) -> bool:
    """Did the streamer actually react to this event, or just farm it?

    Separates a hectic/omg kill from a routine one. OCR game events and
    visually confirmed wins are subject to this check; an ordinary non-event
    returns True. Visual wins must be checked even when OCR missed the
    ``terminal_win`` label, or a calm VLM-confirmed win can still earn the
    content-anchor bypass. Measured 2026-07-23: startle separated creator keeps
    from routine events 3.9x (0.36 vs 0.09); voice/burst/loud showed no
    separation.
    """
    visual_win = _visual_win_read(c)
    if not visual_win and c.get("game_label") not in _REACTION_EVENT_LABELS:
        return True
    # NB: a confirmed match win is NOT auto-exempt. Measured 2026-07-23, two
    # clean VICTORY ROYALE clips (startle ~0) were creator PASSES -- "not every
    # match win, the hectic ones". The win still gets a lower reaction bar than
    # an ordinary kill (a quiet win is more postable than a quiet elim), but a
    # dead-calm win is a stat, not a clip.
    startle = float((c.get("modality_breakdown") or {}).get("startle", 0.0) or 0.0)
    peak = float(c.get("peak_value", 0.0) or 0.0)
    if visual_win:
        return (
            startle >= tuning.win_reaction_min_startle
            or peak >= tuning.win_reaction_min_peak
        )
    return (
        startle >= tuning.event_reaction_min_startle
        or peak >= tuning.event_reaction_min_peak
    )


def _visible_crowd_anchor(c: dict, tuning: SelectionTuning) -> bool:
    return float(c.get("visible_crowd_reaction", 0.0) or 0.0) >= (
        tuning.visible_crowd_anchor_min
    )


def _strong_lobby_reaction(c: dict, tuning: SelectionTuning) -> bool:
    if c.get("scene_label") not in ("lobby", "intermission"):
        return False
    duration = max(1.0, float(c.get("end", 0.0)) - float(c.get("start", 0.0)))
    mean_intensity = float(c.get("reaction_auc", 0.0) or 0.0) / duration
    breakdown = c.get("modality_breakdown") or {}
    burst = float(breakdown.get("burst", 0.0) or 0.0)
    gameplay_change = max(
        float(breakdown.get("motion", 0.0) or 0.0),
        float(breakdown.get("game_audio", 0.0) or 0.0),
    )
    return (
        float(c.get("peak_value", 0.0) or 0.0) >= tuning.lobby_reaction_min_peak
        and mean_intensity >= tuning.lobby_reaction_min_mean_intensity
        and float(c.get("signal_agreement", 0.0) or 0.0)
        >= tuning.lobby_reaction_min_signal_agreement
        and dead_air_ratio_of(c) <= tuning.lobby_reaction_max_dead_air
        and burst >= tuning.lobby_reaction_min_burst
        and gameplay_change >= tuning.lobby_reaction_min_gameplay_change
    )


def _trace_or_live(c: dict, live_name: str, trace_name: str, default=None):
    value = c.get(live_name)
    if value is None:
        value = c.get(trace_name, default)
    return value


def _landed_game_event(c: dict, tuning: SelectionTuning) -> bool:
    return (
        _visual_win_read(c)
        or c.get("game_label") in _REACTION_EVENT_LABELS
    ) and _event_reaction_landed(c, tuning)


def _terminal_region_has_real_beat(c: dict, tuning: SelectionTuning) -> bool:
    """Narrow exemptions for a meaningful beat that happens over an end card."""
    if c.get("strong_startle") or c.get("visible_crowd_anchor"):
        return True
    if _landed_game_event(c, tuning):
        return True
    payoff = float(
        _trace_or_live(c, "visual_semantic_payoff", "visual_payoff", 0.0) or 0.0
    )
    self_contained = float(
        _trace_or_live(
            c,
            "visual_semantic_self_contained",
            "visual_self_contained",
            0.0,
        )
        or 0.0
    )
    context_complete = bool(
        _trace_or_live(
            c,
            "visual_semantic_context_complete",
            "visual_context_complete",
            False,
        )
    )
    routine_only = bool(
        _trace_or_live(
            c,
            "visual_semantic_routine_only",
            "visual_routine_only",
            False,
        )
    )
    verbal_payoff = bool(
        _trace_or_live(
            c,
            "visual_semantic_verbal_payoff",
            "visual_verbal_payoff",
            False,
        )
    )
    streamer_reaction = bool(
        _trace_or_live(
            c,
            "visual_semantic_streamer_reaction",
            "visual_streamer_reaction",
            False,
        )
    )
    return (
        not routine_only
        and context_complete
        and payoff >= tuning.content_min_payoff
        and self_contained >= tuning.content_min_self_contained
        and (verbal_payoff or streamer_reaction)
    )


def _obviously_incomplete_beat(c: dict, tuning: SelectionTuning) -> bool:
    """Reject only judge-observed non-beats; unjudged installs stay unchanged."""
    if not tuning.editorial_hygiene_enabled:
        return False
    visual_seen = bool(_visual_verdict(c))
    semantic_seen = c.get("semantic_verdict") is not None
    if not visual_seen and not semantic_seen:
        return False
    if (
        c.get("strong_startle")
        or c.get("visible_crowd_anchor")
        or _landed_game_event(c, tuning)
        or _terminal_region_has_real_beat(c, tuning)
    ):
        return False
    routine_only = bool(
        _trace_or_live(
            c,
            "visual_semantic_routine_only",
            "visual_routine_only",
            False,
        )
    )
    context_complete = bool(
        _trace_or_live(
            c,
            "visual_semantic_context_complete",
            "visual_context_complete",
            False,
        )
    )
    has_visual_beat = any(
        (
            bool(
                _trace_or_live(
                    c,
                    "visual_semantic_verbal_payoff",
                    "visual_verbal_payoff",
                    False,
                )
            ),
            bool(
                _trace_or_live(
                    c,
                    "visual_semantic_visual_event",
                    "visual_event",
                    False,
                )
            ),
            bool(
                _trace_or_live(
                    c,
                    "visual_semantic_streamer_reaction",
                    "visual_streamer_reaction",
                    False,
                )
            ),
            _visual_outcome(c) not in ("", "none"),
        )
    )
    semantic_complete = (
        str(c.get("semantic_verdict") or "").lower() == "post"
        and str(c.get("semantic_moment_type") or "").lower() != "filler"
        and float(c.get("semantic_payoff", 0.0) or 0.0)
        >= tuning.semantic_rescue_min_payoff
        and float(c.get("semantic_self_contained", 0.0) or 0.0)
        >= tuning.semantic_rescue_min_self_contained
    )
    return (
        (routine_only and not semantic_complete)
        or (
            visual_seen
            and not context_complete
            and not has_visual_beat
            and not semantic_complete
        )
    )


def _semantic_complete_beat(c: dict, tuning: SelectionTuning) -> bool:
    """Text judge found a self-contained setup/payoff rather than filler."""
    if str(c.get("semantic_verdict") or "").strip().lower() != "post":
        return False
    if str(c.get("semantic_moment_type") or "").strip().lower() == "filler":
        return False
    self_contained = c.get("semantic_self_contained")
    if self_contained is None:
        self_contained = c.get("visual_semantic_self_contained", 0.0)
    return bool(
        float(c.get("semantic_payoff", 0.0) or 0.0)
        >= tuning.semantic_rescue_min_payoff
        and float(self_contained or 0.0)
        >= tuning.semantic_rescue_min_self_contained
    )


def _strong_observed_reaction(c: dict, tuning: SelectionTuning) -> bool:
    """Absolute signal fallback for real reactions that optional judges miss."""
    duration = max(1.0, float(c.get("end", 0.0)) - float(c.get("start", 0.0)))
    mean_intensity = float(c.get("reaction_auc", 0.0) or 0.0) / duration
    return bool(
        float(c.get("peak_value", 0.0) or 0.0) >= 1.25
        and mean_intensity >= 0.75
        and float(c.get("signal_agreement", 0.0) or 0.0) >= 0.65
        and dead_air_ratio_of(c) <= 0.45
    )


def _absolute_publishable_beat(c: dict, tuning: SelectionTuning) -> bool:
    """Evidence that this candidate contains a complete, publishable moment.

    This is deliberately independent of the VOD-relative quality floor. It
    prevents the best of a bad stream from becoming publishable merely because
    every neighboring candidate was weaker, while preserving direct gameplay,
    face, crowd, startle, and complete spoken-beat evidence.
    """
    return bool(
        _landed_game_event(c, tuning)
        or _grounded_visual_post(c, tuning)
        or _semantic_complete_beat(c, tuning)
        or _strong_lobby_reaction(c, tuning)
        or _strong_observed_reaction(c, tuning)
    )


def _sustained_reaction(c: dict, tuning: SelectionTuning) -> bool:
    """Does this moment carry enough reaction AREA to skip the peak floor?

    Admission only: this can add a keep path, never remove one, so every
    candidate that cleared ``ordinary_min_peak`` before still clears it. The
    moment still faces the judge, the adaptive floor, and the ranker -- this
    just stops discarding it before anything qualified has looked.
    """
    minimum = float(getattr(tuning, "sustained_reaction_min_auc", 0.0) or 0.0)
    if minimum <= 0.0:
        return False
    return float(c.get("reaction_auc", 0.0) or 0.0) >= minimum


def _gate_rejection_reasons(
    c: dict,
    bar: float,
    tuning: SelectionTuning,
    signal_score: float,
    mean_intensity: float,
    duration: float,
) -> List[str]:
    """Explain every evidence veto, unless editorial consensus rescues it."""
    has_strong_event_evidence = (
        c.get("game_label") in ("elimination", "knock", "rank_progress", "terminal_win")
        and c.get("game_evidence", 0.0) >= tuning.event_min_game_evidence
        and dead_air_ratio_of(c) <= tuning.event_max_dead_air
    )
    min_agreement = (
        tuning.event_min_signal_agreement
        if has_strong_event_evidence
        else tuning.ordinary_min_signal_agreement
    )
    reasons: List[str] = []
    housekeeping = bool(c.get("housekeeping"))
    if housekeeping:
        reasons.append("housekeeping")
    terminal_non_content = (
        tuning.editorial_hygiene_enabled
        and bool(c.get("terminal_non_content"))
        and not _terminal_region_has_real_beat(c, tuning)
    )
    if terminal_non_content:
        reasons.append("terminal_non_content")
    if (
        tuning.routine_non_gameplay_consensus_veto
        and _routine_non_gameplay_consensus(c)
    ):
        reasons.append("routine_non_gameplay")
    elif (
        tuning.routine_non_gameplay_ocr_veto
        and _routine_non_gameplay_ocr(c, tuning)
    ):
        reasons.append("routine_non_gameplay_ocr")
    # Calibrated admission stands the two arousal bars down: the absolute
    # P(keep) gate downstream decides instead. Only candidates the model
    # actually scored get the bypass, so partial ranker coverage degrades to
    # the historical gates rather than to no gate at all.
    _pub = c.get("ranker_publishability_score")
    calibrated_here = (
        bool(getattr(tuning, "calibrated_admission", False)) and _pub is not None
    ) or (
        # Union mode: the arousal bars stand down only for candidates the
        # model actually vouches for at the active floor.
        bool(getattr(tuning, "calibrated_union", False))
        and _pub is not None
        and float(_pub) >= float(getattr(tuning, "calibrated_floor", 0.0))
        and float(getattr(tuning, "calibrated_floor", 0.0)) > 0.0
    )
    if (
        not calibrated_here
        and c.get("peak_value", 0.0) < max(bar, tuning.ordinary_min_peak)
        and not _sustained_reaction(c, tuning)
    ):
        reasons.append("peak_floor")
    if (
        not calibrated_here
        and signal_score < tuning.ordinary_min_selection_score
    ):
        reasons.append("score_floor")
    if not _modality_gate(
        c, min_agreement, tuning, is_event=has_strong_event_evidence,
    ):
        reasons.append("modality_gate")
    if dead_air_ratio_of(c) > tuning.ordinary_max_dead_air:
        reasons.append("dead_air")
    if (
        c.get("scene_label") in ("lobby", "intermission")
        and duration > tuning.lobby_max_duration
        and not _strong_lobby_reaction(c, tuning)
    ):
        reasons.append("lobby_length")
    if (
        bool(c.get("floor_padded"))
        and not has_strong_event_evidence
        and int(c.get("crowd_clip", 0) or 0) < tuning.crowd_clip_pad_exempt_min
    ):
        reasons.append("floor_padded")
    if _obviously_incomplete_beat(c, tuning):
        reasons.append("incomplete_beat")
    if _generic_visual_filler(c):
        reasons.append("visual_filler")
    if (
        c.get("game_label") in ("elimination", "knock", "rank_progress", "terminal_win")
        and mean_intensity < tuning.event_min_mean_intensity
    ):
        reasons.append("routine_event")

    semantic_verdict = str(c.get("semantic_verdict") or "").strip().lower()
    semantic_type = str(c.get("semantic_moment_type") or "").strip().lower()
    if (
        _is_chat_reaction_only(c, tuning)
        and (
            semantic_verdict == "skip"
            or semantic_type == "filler"
            or not _grounded_visual_post(c, tuning)
        )
    ):
        reasons.append("non_event_chatter")

    # The absolute beat decision only has authority after both independent
    # judges saw the finalist. A single optional judge remains advisory on
    # partial-coverage and offline installations.
    judges_agree_on_scope = (
        c.get("semantic_verdict") is not None and bool(_visual_verdict(c))
    )
    if judges_agree_on_scope and not _absolute_publishable_beat(c, tuning):
        reasons.append("unpublishable_beat")

    semantic_ran = c.get("semantic_verdict") is not None
    semantic_override = (
        _visible_crowd_anchor(c, tuning)
        # A viewer !clip deliberately does NOT suppress the semantic reject
        # here, even under viewer protection. Suppressing it upstream would
        # neuter the consensus-filler veto that the protected branch of
        # _crowd_command_veto_reasons is supposed to retain — protection buys
        # deck priority, not immunity from both judges agreeing it is filler.
        # A visual opinion only outranks the transcript reject when concrete
        # event/reaction/payoff evidence grounds it. Bare post/maybe labels are
        # exactly the lobby/funny free-pass failure this conflict rule closes.
        or (
            _visual_verdict(c) in ("post", "maybe")
            and _grounded_visual_post(c, tuning)
        )
    )
    semantic_consensus_reject = (
        c.get("semantic_verdict") == "skip"
        and (
            c.get("semantic_moment_type") == "filler"
            or _visual_verdict(c) == "skip"
        )
    )
    is_semantic_reject = semantic_ran and not semantic_override and (
        semantic_consensus_reject
        or (
            tuning.semantic_gate_authority
            and
            tuning.semantic_hard_reject
            and (
                c.get("semantic_verdict") == "skip"
                or c.get("semantic_moment_type") == "filler"
                or float(c.get("semantic_payoff", 0.0) or 0.0)
                < tuning.semantic_min_payoff
            )
        )
    )
    if is_semantic_reject:
        reasons.append("semantic_reject")

    # Visual judge skip is usually "not a clip" evidence. Wins still bypass.
    # A very narrow base/text/signal conflict also makes the skip advisory:
    # fresh VLM scans gave opposite verdicts to the same founder-kept rage
    # moment, while the other independent lanes stayed strongly positive.
    # The candidate receives no rescue here; it simply faces the normal gates.
    # A skip from a judge that was told "(no clear speech)" about audio nobody
    # transcribed (Smart scan outside its scout regions, or a failed ASR pass)
    # speaks only for the frames, so it is not a categorical veto. The
    # candidate still faces every other gate, including a semantic reject.
    if (
        _visual_verdict(c) == "skip"
        and not c.get("speech_unheard")
        and not _content_gate_bypass(c, tuning)
        and not _strong_visual_skip_conflict(c, tuning)
    ):
        reasons.append("visual_skip")

    # The positive verdict may correct coarse signal/scene rules, but it
    # cannot contradict a semantic reject -- or a housekeeping sign-off, which is
    # structurally not a clip no matter how animated the goodbye was. Validated
    # visual wins earn the same per-candidate bypass: the arousal gates exist to
    # measure whether a moment LANDED, and a confirmed match win is direct
    # evidence it did. Verbal content beats do not get this bypass.
    # Visual skip is never wiped by semantic rescue.
    if "visual_skip" not in reasons and not is_semantic_reject and (
        _semantic_rescue_allowed(c, tuning)
        or _visible_crowd_anchor(c, tuning)
        or _content_gate_bypass(c, tuning)
    ):
        return [
            reason
            for reason in reasons
            if reason in (
                "housekeeping",
                "terminal_non_content",
                "incomplete_beat",
                "visual_filler",
                "non_event_chatter",
                "unpublishable_beat",
            )
        ]
    return reasons


_STRUCTURAL_VETO_REASONS = frozenset(
    (
        "housekeeping",
        "terminal_non_content",
        "routine_non_gameplay",
        "routine_non_gameplay_ocr",
        "incomplete_beat",
        "visual_filler",
        "non_event_chatter",
        "unpublishable_beat",
    )
)


def _creator_protected(c: dict) -> bool:
    """Only an explicit creator/manual marker receives absolute protection.

    Production scan candidates do not currently set these fields; manual clips
    are created after review. Keeping the distinction here prevents future
    creator-authored candidates from being conflated with Twitch audience
    suggestions again.
    """
    return bool(c.get("creator_protected") or c.get("manual_anchor"))


def _creator_veto_reasons(reasons: List[str]) -> List[str]:
    """Creator intent bypasses model disagreement, never structural non-content."""
    return [reason for reason in reasons if reason in _STRUCTURAL_VETO_REASONS]


def _crowd_command_veto_reasons(c: dict, reasons: List[str]) -> List[str]:
    """Keep only high-confidence vetoes for a viewer-recommended candidate.

    Viewer commands may propose quiet or oddly framed moments, so they retain a
    recall-friendly admission path through relative arousal gates. They cannot
    override structural non-content or text/vision consensus that the moment is
    filler. Once admitted, they receive only a bounded score bonus and compete
    normally at the adaptive floor and deck ceiling.
    """
    # A semantic veto cannot be offset by a score bonus. Allow requested moments to
    # survive that judgement when configured, while keeping structural vetoes.
    veto_reasons = set(_STRUCTURAL_VETO_REASONS)
    if DEFAULT_TUNING.crowd_command_survives_semantic_reject:
        # Structural non-content vetoes still apply; only the judge's filler
        # opinion stands down for an explicitly requested moment.
        pass
    else:
        veto_reasons.add("semantic_reject")
    kept = [reason for reason in reasons if reason in veto_reasons]
    semantic_verdict = str(c.get("semantic_verdict") or "").strip().lower()
    semantic_type = str(c.get("semantic_moment_type") or "").strip().lower()
    if (
        "visual_skip" in reasons
        and (semantic_verdict == "skip" or semantic_type == "filler")
    ):
        kept.append("visual_skip")
    return list(dict.fromkeys(kept))


def _passes_gates(
    c: dict,
    bar: float,
    tuning: SelectionTuning,
    signal_score: float,
    mean_intensity: float,
    duration: float,
) -> bool:
    """Hard evidence gates for non-protected candidates (the clears_bar test)."""
    return not _gate_rejection_reasons(
        c, bar, tuning, signal_score, mean_intensity, duration,
    )

def passes_evidence_gates(
    c: dict,
    bar: float,
    tuning: Optional[SelectionTuning] = None,
) -> bool:
    """Would this candidate clear the current evidence/editorial gates?"""
    tuning = tuning or DEFAULT_TUNING
    signal_score, mean_intensity, duration = _score_components(
        c, tuning, tuning.game_bonus,
    )
    reasons = _gate_rejection_reasons(
        c, bar, tuning, signal_score, mean_intensity, duration,
    )
    if _creator_protected(c):
        reasons = _creator_veto_reasons(reasons)
    elif (
        tuning.viewer_commands_protected
        and int(c.get("crowd_clip", 0) or 0) > 0
    ):
        # Protection is a RANKING privilege (deck-floor exemption + sort
        # priority), not a licence to ignore evidence. The high-confidence
        # vetoes still apply — structural non-content, and text/vision
        # consensus that the moment is filler. Wiping every reason here (the
        # original rollback lane) let a command wave housekeeping through,
        # contradicting the housekeeping invariant, and measured no
        # better on the golden set than keeping the vetoes.
        reasons = _crowd_command_veto_reasons(c, reasons)
    elif int(c.get("crowd_clip", 0) or 0) > 0:
        reasons = _crowd_command_veto_reasons(c, reasons)
    return not reasons


def _event_rich_ordinary_allowed(c: dict, tuning: SelectionTuning) -> bool:
    """Keep event-rich decks selective without making burst a global veto.

    OCR-backed events and protected human anchors are handled by the caller.
    For ordinary reactions, a real audio burst remains sufficient. When the
    semantic judge ran, its postability/payoff/self-containment read becomes
    the alternative evidence path. Without a usable verdict, require a strong
    opening plus corroborating channels and little dead air.

    This is deliberately an admission rule, not a quota: a long VOD may still
    produce few clips when none of these evidence paths clear.
    """
    if (
        _semantic_rescue_allowed(c, tuning)
        or _strong_startle(c, tuning)
        or _visible_crowd_anchor(c, tuning)
        or _strong_lobby_reaction(c, tuning)
        or _content_gate_bypass(c, tuning)
    ):
        return True
    burst = float((c.get("modality_breakdown") or {}).get("burst", 0.0) or 0.0)
    if burst >= tuning.ordinary_burst_exemption:
        return True

    # Without gate authority the verdict-keyed admission branches are skipped
    # entirely and every non-burst ordinary candidate takes the signal-only
    # fallback below — verdicts keep contributing to rank scores only.
    verdict = (
        str(c.get("semantic_verdict") or "").strip().lower()
        if tuning.semantic_gate_authority else ""
    )
    payoff = float(c.get("semantic_payoff", 0.0) or 0.0)
    self_contained = float(c.get("semantic_self_contained", 0.0) or 0.0)
    hook = float(c.get("hook_score", 0.0) or 0.0)
    if verdict == "post":
        return (
            payoff >= tuning.event_rich_post_payoff_min
            and self_contained >= tuning.event_rich_post_self_contained_min
        )
    if verdict == "maybe":
        return (
            hook >= tuning.event_rich_signal_hook_min
            and payoff >= tuning.event_rich_maybe_payoff_min
            and self_contained >= tuning.event_rich_maybe_self_contained_min
        )
    if verdict:
        return False

    return (
        hook >= tuning.event_rich_signal_hook_min
        and float(c.get("signal_agreement", 0.0) or 0.0) >= tuning.event_rich_signal_agreement_min
        and dead_air_ratio_of(c) <= tuning.event_rich_signal_max_dead_air
    )


def _distinct_event_times(candidates: List[dict], tuning: SelectionTuning) -> List[float]:
    """Deduplicated eligible OCR event anchors used for local deck policy.

    Sweep-detected visual wins are excluded: event-rich mode expresses "this
    VOD has a rich OCR event vocabulary, be precise", and injected banner
    reads say nothing about OCR coverage — counting them flipped a quiet
    variety VOD into event-rich precision and slashed its deck to 6 clips.
    """
    times = sorted(
        float(c.get("game_anchor_time", c.get("peak_timestamp", 0.0)) or 0.0)
        for c in candidates
        if c.get("game_label")
        in ("terminal_win", "elimination", "knock", "rank_progress")
        and not c.get("visual_win_anchored")
    )
    distinct: List[float] = []
    for timestamp in times:
        if not distinct or timestamp - distinct[-1] > tuning.event_rich_dedupe_sec:
            distinct.append(timestamp)
    return distinct


def _locally_event_rich(c: dict, event_times: List[float], tuning: SelectionTuning) -> bool:
    if len(event_times) < tuning.event_rich_min_candidates:
        return False
    center = float(c.get("peak_timestamp", 0.0) or 0.0)
    radius = max(0.0, tuning.event_rich_window_sec) / 2.0
    return sum(abs(timestamp - center) <= radius for timestamp in event_times) >= (
        tuning.event_rich_min_candidates
    )


def select_clips(
    candidates: List[dict],
    score: np.ndarray,
    max_clips: int,
    k: float,
    game_bonus: float = 0.5,
    tuning: Optional[SelectionTuning] = None,
    ranker_policy: str = "replace",
    ranker_blend_weight: float = 0.35,
    personal_ranker_blend_weight: float = 0.0,
    ranker_publishability_floor: float = 0.0,
    ranker_use_personalized_adaptive_floor: bool = True,
    second_look_rejector=None,
) -> List[dict]:
    """Rank, gate against the relative bar, and NMS-suppress overlaps."""
    if not candidates:
        return []

    tuning = tuning or DEFAULT_TUNING
    _assign_face_spike_rescue_eligibility(candidates, tuning)
    game_bonus = tuning.game_bonus if game_bonus == 0.5 else game_bonus
    bar = quality_bar(score, k)
    # ``tail_only`` is the measured default: the ranker SCORES every candidate
    # (so the Second-look tier can order by it) but does not touch
    # selection_score, so Primary stays signal-ordered. MEASURED 2026-07-26 --
    # leave-one-VOD-out over 16 deck-scoreable folds put every ordering policy
    # at or below signals (replace: p@5 -0.012, recall -0.064; best blend
    # MARGINAL), while blind creator labels put the ranker's picks in the
    # CEILING pool at 37.5% keep vs 16.7% for a random draw from that same
    # pool. It out-orders the tail and cannot out-order the head, so it is
    # given exactly the tail.
    ranker_policy = str(ranker_policy or "tail_only").strip().lower()
    if ranker_policy not in ("replace", "blend", "tail_only"):
        ranker_policy = "tail_only"
    # Under tail_only the ranker is present but not an ordering authority.
    ranker_orders_deck = ranker_policy != "tail_only"
    ranker_blend_weight = max(0.0, min(1.0, float(ranker_blend_weight)))
    # Personal taste is a bounded preference tilt, not a substitute for base
    # quality. The product can raise this from zero only after its own policy
    # gate passes, and even then one user can own at most 25% of ordering.
    personal_ranker_blend_weight = max(
        0.0, min(0.25, float(personal_ranker_blend_weight)),
    )
    ranker_publishability_floor = max(
        0.0, min(1.0, float(ranker_publishability_floor)),
    )
    # Calibrated admission stands the arousal gates down ONLY when the
    # absolute gate that replaces them is actually armed. A zero floor would
    # otherwise remove the old bars with nothing in their place.
    #
    # The floor has two sources: the ``ranker_publishability_floor`` parameter
    # (settings/production path) and ``tuning.calibrated_floor`` (the
    # SelectionTuning path, which is how scripts/ab_deck_review.py --set can
    # drive the lane). The parameter wins when it is set; the tuning value is
    # the fallback, so a tuning-only override is not silently erased by a
    # production floor of 0.0. The resolved value is stamped back onto the
    # tuning so per-candidate gate logic can read it.
    from dataclasses import replace as _dc_replace
    _calibrated_mode = getattr(tuning, "calibrated_admission", False) or getattr(
        tuning, "calibrated_union", False,
    )
    if _calibrated_mode:
        effective_floor = (
            ranker_publishability_floor
            if ranker_publishability_floor > 0.0
            else max(0.0, min(1.0, float(getattr(tuning, "calibrated_floor", 0.0))))
        )
        if effective_floor <= 0.0:
            tuning = _dc_replace(
                tuning,
                calibrated_admission=False,
                calibrated_union=False,
                calibrated_floor=0.0,
            )
        else:
            tuning = _dc_replace(tuning, calibrated_floor=effective_floor)
            ranker_publishability_floor = effective_floor

    # Verbal-anchor admission cap: rank prospective VERBAL anchors by evidence
    # strength and demote everything past content_anchor_max BEFORE the gate
    # pass below consults _content_anchor. Reaction-qualified win anchors are
    # exempt from this verbal cap (rare, deterministically validated). Demoted
    # candidates keep their content_interest rank contribution; they just face
    # the ordinary gates.
    if tuning.content_axis and tuning.content_anchor_max > 0:
        for c in candidates:
            c.pop("content_anchor_capped", None)
        prospective = [
            c for c in candidates
            if _content_anchor(c, tuning) and not _visual_win_read(c)
        ]
        prospective.sort(
            key=lambda c: (
                content_interest_score(c, tuning),
                float(c.get("peak_value", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for c in prospective[tuning.content_anchor_max:]:
            c["content_anchor_capped"] = True

    for c in candidates:
        c["selection_disposition"] = "unscored"
        c["selection_rejection_reasons"] = []
        c["selection_funnel"] = {
            "candidate": "generated",
            "scoring": (
                f"learned_{ranker_policy}" if "ranker_score" in c
                else "signals"
            ),
            "evidence": "pending",
            "publishability": (
                "pending" if "ranker_score" in c and ranker_publishability_floor > 0
                else "not_applied" if "ranker_score" in c
                else "not_available"
            ),
            "adaptive_quality": "not_reached",
            "event_quality": "not_reached",
            "final": "not_reached",
        }
        c.pop("selection_suppressed_by", None)
        c.pop("moment_group_id", None)
        c.pop("moment_group_source", None)
        c.pop("selection_rank", None)
        c.pop("ranker_percentile", None)
        c.pop("personal_ranker_percentile", None)
        c.pop("signal_percentile", None)
        signal_score, mean_intensity, duration = _score_components(
            c, tuning, game_bonus,
        )
        # Persisted-candidate replays may carry the score produced by the live
        # scan. Individual trace features are compact/rounded, so rebuilding
        # from them can reverse a near-tie at the deck cutoff. Production
        # candidates never carry this private replay-only key.
        if c.get("_persisted_signal_selection_score") is not None:
            signal_score = float(c["_persisted_signal_selection_score"])
        c["signal_selection_score"] = signal_score
        on_ranker_path = "ranker_score" in c and ranker_orders_deck
        if on_ranker_path:
            # Learned ranker ranks eligible candidates, but the evidence gate
            # stays signal-ordered. Ranker probabilities and signal scores are
            # different scales.
            ranker_score = c["ranker_score"]
            if c.get("scene_label") in ("lobby", "intermission"):
                ranker_score -= tuning.lobby_ranker_penalty
            c["selection_score"] = ranker_score
        else:
            # Signal score: peak prominence + mean intensity (area / duration), NOT raw
            # area, so a long mediocre clip can't outrank a short intense one.
            c["selection_score"] = signal_score
        # Deterministic demotions that are NOT model inputs stay on both paths:
        # neither chat_reaction_only nor scene_label appears in FEATURE_NAMES,
        # so the ranker cannot have learned them.
        if _is_chat_reaction_only(c, tuning):
            c["selection_score"] -= tuning.chat_reaction_score_penalty
            c["chat_reaction_only"] = True
        else:
            c["chat_reaction_only"] = False
        c["calibrated_face_spike"] = bool(_calibrated_face_spike(c, tuning))
        c["exceptional_face_spike"] = bool(_exceptional_face_spike(c, tuning))
        c["face_spike_floor_rescued"] = False
        c["visual_post_grounded"] = bool(_grounded_visual_post(c, tuning))
        c["hook_score"] = compute_hook_score(c, tuning)
        c["content_interest"] = content_interest_score(c, tuning)
        semantic_quality = 0.0
        if "semantic_hook" in c:
            payoff = float(c.get("semantic_payoff", 0.0) or 0.0)
            self_contained = float(c.get("semantic_self_contained", 0.0) or 0.0)
            # Self-containment improves a payoff; it cannot manufacture one.
            semantic_quality = payoff * (0.60 + 0.40 * self_contained)
        # The editorial bonuses are SIGNAL-PATH terms. On the ranker path
        # every one of them is already an input to the model -- hook_score,
        # payoff / self_contained / moment_type_ordinal, and the v6 content
        # features are all in FEATURE_NAMES -- so re-adding them here both
        # double-counts them and destroys scale: a ranker probability spans
        # 0..1 while the content term alone spans 0..content_score_weight
        # (6.0), which lets a single editorial bonus overwrite the model's
        # entire ordering. Editorial terms are therefore signal-path only.
        if not on_ranker_path:
            c["selection_score"] += tuning.hook_score_weight * c["hook_score"]
            # Content axis: grounded interest joins the RANK, not only
            # admission — measured on a founder VOD, three kept wins cleared
            # every gate and were budget-cut at the deck cap, so a bypass alone
            # cannot save them.
            if c["content_interest"] > 0.0:
                # Validated wins keep the full lift so they survive
                # clip_ceiling; verbal beats get a softer nudge so they do not
                # dominate p@5.
                content_weight = (
                    tuning.content_score_weight
                    if _visual_win_read(c)
                    else tuning.content_verbal_score_weight
                )
                c["selection_score"] += content_weight * c["content_interest"]
            c["selection_score"] += tuning.semantic_score_weight * semantic_quality
        c["signal_editorial_score"] = (
            signal_score
            + tuning.hook_score_weight * c["hook_score"]
            + tuning.semantic_score_weight * semantic_quality
        )
        if c.get("_persisted_signal_editorial_score") is not None:
            c["signal_editorial_score"] = float(
                c["_persisted_signal_editorial_score"]
            )
        # OCR proposes gameplay context; it is not a human editorial command.
        # Only an explicit viewer !clip marker bypasses evidence gates. A win
        # must still show that it mattered through reaction, agreement, and
        # dead-air checks.
        creator_protected = _creator_protected(c)
        legacy_viewer_protected = bool(
            tuning.viewer_commands_protected
            and int(c.get("crowd_clip", 0) or 0) > 0
        )
        c["protected"] = creator_protected or legacy_viewer_protected
        c["crowd_recommended"] = int(c.get("crowd_clip", 0) or 0) > 0
        c["semantic_rescue"] = _semantic_rescue_allowed(c, tuning)
        c["strong_startle"] = _strong_startle(c, tuning)
        c["visible_crowd_anchor"] = _visible_crowd_anchor(c, tuning)
        c["strong_lobby_reaction"] = _strong_lobby_reaction(c, tuning)
        c["content_anchor"] = _content_anchor(c, tuning)
        c.pop("second_look_keep_probability", None)
        c.pop("second_look_reject_threshold", None)
        rejection_reasons = _gate_rejection_reasons(
            c, bar, tuning, signal_score, mean_intensity, duration,
        )
        if legacy_viewer_protected:
            # High-confidence vetoes survive viewer protection — see the
            # matching branch in _passes_gates. The judge's filler opinion is
            # handled inside _crowd_command_veto_reasons via
            # crowd_command_survives_semantic_reject, which is unconditional
            # and blinded-A/B validated; a calibrated-mode-only duplicate of
            # that standdown lived here briefly and was removed rather than
            # run two mechanisms over the same veto.
            rejection_reasons = _crowd_command_veto_reasons(
                c, rejection_reasons,
            )
        elif creator_protected:
            rejection_reasons = _creator_veto_reasons(rejection_reasons)
        elif c["crowd_recommended"]:
            rejection_reasons = _crowd_command_veto_reasons(
                c, rejection_reasons,
            )
        c["selection_rejection_reasons"] = rejection_reasons
        c["clears_bar"] = not rejection_reasons
        c["selection_disposition"] = (
            "eligible" if c["clears_bar"] else "evidence_gate"
        )
        c["selection_funnel"]["evidence"] = (
            "legacy_viewer_protected"
            if legacy_viewer_protected and c["clears_bar"]
            else "creator_protected"
            if creator_protected and c["clears_bar"]
            else "passed"
            if c["clears_bar"]
            else "rejected"
        )
        if not c["clears_bar"]:
            c["selection_funnel"]["final"] = "evidence_gate"

    # Shadow-safe learned ordering: compare within-deck percentiles instead of
    # mixing an unbounded signal score with a 0..1 model probability.  The
    # production policy uses this measured blend for the universal base model.
    # A personal adapter can contribute only through its separately capped
    # percentile below; it never replaces the base score.
    ranked_candidates = [c for c in candidates if "ranker_score" in c]
    if ranker_policy == "blend" and ranked_candidates:
        def _percentiles(values):
            values = np.asarray(values, dtype=np.float64)
            if len(values) <= 1:
                return np.ones(len(values), dtype=np.float64)
            order = np.argsort(values, kind="stable")
            result = np.empty(len(values), dtype=np.float64)
            result[order] = np.arange(len(values), dtype=np.float64) / (len(values) - 1)
            return result

        signal_percentiles = _percentiles([
            float(c.get("signal_editorial_score", 0.0) or 0.0)
            for c in ranked_candidates
        ])
        ranker_percentiles = _percentiles([
            float(c.get("ranker_score", 0.0) or 0.0)
            - (
                tuning.lobby_ranker_penalty
                if c.get("scene_label") in ("lobby", "intermission") else 0.0
            )
            for c in ranked_candidates
        ])
        has_personal = any(
            c.get("personal_preference_delta") is not None
            for c in ranked_candidates
        )
        personal_weight = (
            personal_ranker_blend_weight if has_personal else 0.0
        )
        base_weight = min(ranker_blend_weight, 1.0 - personal_weight)
        personal_percentiles = _percentiles([
            float(c.get("personal_preference_delta", 0.0) or 0.0)
            for c in ranked_candidates
        ])
        signal_weight = max(0.0, 1.0 - base_weight - personal_weight)
        for c, signal_percentile, ranker_percentile, personal_percentile in zip(
            ranked_candidates, signal_percentiles, ranker_percentiles,
            personal_percentiles,
        ):
            c["signal_percentile"] = float(signal_percentile)
            c["ranker_percentile"] = float(ranker_percentile)
            c["personal_ranker_percentile"] = float(personal_percentile)
            c["selection_score"] = (
                signal_weight * float(signal_percentile)
                + base_weight * float(ranker_percentile)
                + personal_weight * float(personal_percentile)
            )

    eligible = [c for c in candidates if c["clears_bar"]]

    # Under calibrated admission the VOD-relative floors are stood down: they
    # measure distance from this VOD's loudest moment, which is exactly the
    # bar the absolute calibrated gate replaces. Deck size then comes from
    # how many candidates clear the absolute gate (bounded by max_clips as a
    # ceiling), not from proximity to the local maximum.
    calibrated_gate_active = (
        bool(getattr(tuning, "calibrated_admission", False))
        and ranker_publishability_floor > 0.0
        and any(c.get("ranker_publishability_score") is not None
                for c in eligible)
    )

    def _calibrated_floor_rescue(c: dict) -> bool:
        """Union mode: the model vouches for a floor-rejected candidate."""
        if not getattr(tuning, "calibrated_union", False):
            return False
        pub = c.get("ranker_publishability_score")
        return (
            pub is not None
            and float(pub) >= float(getattr(tuning, "calibrated_floor", 0.0))
        )

    # Apply an adaptive publishability floor only when the absolute gates left
    # more candidates than the requested deck can hold. This is intentionally
    # not a quota: candidates below the floor are rejected rather than used to
    # pad the deck back to max_clips.
    if calibrated_gate_active:
        for c in eligible:
            c["selection_funnel"]["adaptive_quality"] = "calibrated_gate"
    elif max_clips > 0 and len(eligible) > max_clips:
        # Exclude anchored wins and viewer-requested moments from the floor reference.
        # Their bonuses must not raise the admission bar for ordinary moments.
        floor_reference = [
            c for c in eligible
            if not c.get("visual_win_anchored")
            and not int(c.get("crowd_clip", 0) or 0)
        ] or eligible
        best_signal = max(
            (
                float(c.get("signal_selection_score", 0.0) or 0.0)
                for c in floor_reference
            ),
            default=0.0,
        )
        # The tighter "personalized" floor is only justified when the ranker is
        # actually ORDERING the deck -- its premise is that personal ranking
        # narrows the pool reliably. Under tail_only the deck is still
        # signals, so applying it would just cut recall for nothing.
        # (Measured 2026-07-26: replace_soft, which is `replace` ordering with
        # the signal-path 0.50 floor, recovers recall 0.459 -> 0.481, so a good
        # slice of `replace`'s recall loss is this floor rather than the
        # ordering.)
        ranker_scored_pool = any("ranker_score" in c for c in eligible)
        if (
            ranker_use_personalized_adaptive_floor
            and ranker_orders_deck
            and ranker_scored_pool
        ):
            adaptive_fraction = (
                tuning.personalized_adaptive_deck_score_floor_fraction
            )
        elif ranker_scored_pool:
            # Ranker present but not driving the personalized floor -- the
            # common production case. The floor duplicates the ranker's own
            # judgement here, so it stands down.
            adaptive_fraction = (
                tuning.ranker_scored_adaptive_deck_score_floor_fraction
            )
        else:
            adaptive_fraction = tuning.adaptive_deck_score_floor_fraction
        adaptive_floor = max(
            tuning.ordinary_min_selection_score,
            best_signal * adaptive_fraction,
        )
        above_adaptive_floor = []
        for c in eligible:
            # Verdict authority ADDS a keep path; it never removes one, so
            # every existing exemption (content anchors, narrative story
            # climaxes, crowd/startle) keeps working exactly as before.
            ordinary_keep = bool(_visual_floor_keep(c, tuning, adaptive_floor)) or (
                c.get("protected")
                or c.get("semantic_rescue")
                or c.get("strong_startle")
                or c.get("visible_crowd_anchor")
                or c.get("strong_lobby_reaction")
                or c.get("content_anchor")
                or _labeled_story_floor_keep(c, tuning)
                or float(c.get("signal_selection_score", 0.0) or 0.0)
                >= adaptive_floor
            )
            face_rescue = bool(
                not ordinary_keep
                and _exceptional_face_spike(c, tuning)
                and _face_spike_rescue_anchor(c, tuning)
            )
            if face_rescue:
                c["face_spike_floor_rescued"] = True
            calibrated_rescue = bool(
                not ordinary_keep
                and not face_rescue
                and _calibrated_floor_rescue(c)
            )
            if calibrated_rescue:
                c["calibrated_floor_rescued"] = True
            keep = ordinary_keep or face_rescue or calibrated_rescue
            if keep:
                c["selection_funnel"]["adaptive_quality"] = (
                    "calibrated_rescue" if calibrated_rescue else "passed"
                )
                above_adaptive_floor.append(c)
            else:
                c["clears_bar"] = False
                c["selection_disposition"] = "adaptive_quality_floor"
                c["selection_rejection_reasons"] = ["adaptive_quality_floor"]
                c["selection_funnel"]["adaptive_quality"] = "rejected"
                c["selection_funnel"]["final"] = "adaptive_quality_floor"
        eligible = above_adaptive_floor
    else:
        for c in eligible:
            c["selection_funnel"]["adaptive_quality"] = "not_needed"

    event_times = _distinct_event_times(eligible, tuning)
    # A detected game profile has a curated OCR vocabulary, so event-rich mode
    # is trustworthy VOD-wide (the validated Fortnite behavior). ``generic``
    # uses broad fallback words that can produce a few sparse hits across a
    # long variety/rage VOD; constrain those hits to their local half-hour.
    has_specific_game = any(
        str(c.get("game_profile", "generic") or "generic") != "generic"
        for c in eligible
    )
    globally_event_rich = (
        has_specific_game and len(event_times) >= tuning.event_rich_min_candidates
    )
    local_event_rich = {
        id(c): (
            globally_event_rich or _locally_event_rich(c, event_times, tuning)
        )
        for c in eligible
    }
    if calibrated_gate_active:
        for c in eligible:
            c["selection_funnel"]["event_quality"] = "calibrated_gate"
    elif any(local_event_rich.values()):
        best_signal = max((
            c.get("signal_selection_score", 0.0)
            for c in eligible if local_event_rich[id(c)]
        ), default=0.0)
        deck_floor = max(
            tuning.ordinary_min_selection_score,
            best_signal * tuning.event_rich_score_floor_fraction,
        )
        above_floor = []
        for c in eligible:
            ordinary_keep = bool(_visual_floor_keep(c, tuning, deck_floor)) or (
                not local_event_rich[id(c)]
                or c.get("protected")
                or c.get("semantic_rescue")
                or c.get("strong_startle")
                or c.get("visible_crowd_anchor")
                or c.get("strong_lobby_reaction")
                or c.get("content_anchor")
                or _labeled_story_floor_keep(c, tuning)
                or c.get("signal_selection_score", 0.0) >= deck_floor
            )
            face_rescue = bool(
                not ordinary_keep
                and _exceptional_face_spike(c, tuning)
                and _face_spike_rescue_anchor(c, tuning)
            )
            if face_rescue:
                c["face_spike_floor_rescued"] = True
            calibrated_rescue = bool(
                not ordinary_keep
                and not face_rescue
                and _calibrated_floor_rescue(c)
            )
            if calibrated_rescue:
                c["calibrated_floor_rescued"] = True
            keep = ordinary_keep or face_rescue or calibrated_rescue
            if keep:
                c["selection_funnel"]["event_quality"] = (
                    "calibrated_rescue" if calibrated_rescue else "passed"
                )
                above_floor.append(c)
            else:
                c["selection_disposition"] = "event_rich_floor"
                c["selection_rejection_reasons"] = ["event_rich_floor"]
                c["selection_funnel"]["event_quality"] = "rejected"
                c["selection_funnel"]["final"] = "event_rich_floor"
        eligible = above_floor
    else:
        for c in eligible:
            c["selection_funnel"]["event_quality"] = "not_needed"

    # A learned model previously only reordered clips; an all-junk VOD could
    # therefore still fill the entire cap with low predicted-Keep candidates.
    # Apply the optional absolute floor only after the deterministic evidence,
    # adaptive, and event-rich policies have formed their pool.  That makes the
    # shadow lane a final publishability filter instead of letting low model
    # scores reshape upstream event-density thresholds. Direct human !clip
    # markers remain protected. A zero floor preserves historical behavior.
    if ranker_publishability_floor > 0.0 and not getattr(
        tuning, "calibrated_union", False,
    ):
        # Union mode never REMOVES an incumbent admit: the floor value is
        # used only as the rescue bar above, so this rejection pass is
        # replacement-mode (and legacy shadow-lane) behavior.
        publishable = []
        for c in eligible:
            if c.get("protected"):
                c["selection_funnel"]["publishability"] = "protected"
                publishable.append(c)
                continue
            # Absolute eligibility belongs to the calibrated BASE model. The
            # personal adapter cannot make a universally weak moment eligible.
            publishability = c.get("ranker_publishability_score")
            if publishability is None:
                publishability = c.get("ranker_score", 0.0)
            if float(publishability or 0.0) >= ranker_publishability_floor:
                c["selection_funnel"]["publishability"] = "passed"
                publishable.append(c)
                continue
            c["clears_bar"] = False
            c["selection_disposition"] = "ranker_publishability_floor"
            c["selection_rejection_reasons"] = ["ranker_publishability_floor"]
            c["selection_funnel"]["publishability"] = "rejected"
            c["selection_funnel"]["final"] = "ranker_publishability_floor"
        eligible = publishable

    # Low-yield recovery is evidence-bounded. It restores only candidates that
    # cleared the absolute evidence gate and were later cut by a VOD-relative
    # adaptive/event floor. A weak VOD therefore can yield one or two credible
    # moments without padding itself back to the deck ceiling.
    recovery_target = max(0, int(tuning.low_yield_recovery_target))
    if len(eligible) < recovery_target:
        recoverable = [
            candidate for candidate in candidates
            if candidate not in eligible
            and candidate.get("selection_disposition")
            in ("adaptive_quality_floor", "event_rich_floor")
            and _absolute_publishable_beat(candidate, tuning)
        ]
        recoverable.sort(
            key=lambda candidate: (
                _moment_representative_key(candidate, tuning),
                float(candidate.get("selection_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for candidate in recoverable[: recovery_target - len(eligible)]:
            candidate["clears_bar"] = True
            candidate["selection_disposition"] = "eligible"
            candidate["selection_rejection_reasons"] = []
            candidate["selection_funnel"]["low_yield_recovery"] = "direct_evidence"
            candidate["selection_funnel"]["final"] = "not_reached"
            eligible.append(candidate)

    # Only an event that would have been lost specifically at an adaptive or
    # event-rich floor earns a rank correction. This makes the feature a
    # rescue for under-credited evidence, not a broad face-ranking policy.
    for c in eligible:
        if not c.get("face_spike_floor_rescued"):
            continue
        if ranker_orders_deck and "ranker_score" in c:
            c["selection_score"] += tuning.face_spike_rank_bonus
        else:
            c["selection_score"] += tuning.face_spike_score_bonus

    # Calibrated rescues claim RESERVED slots rather than a score bonus (see
    # calibrated_rescue_deck_fraction). Rescues past the reserved budget are
    # returned to the floor rejection they came from, so the model can widen
    # the deck by a bounded amount and never more.
    rescue_slots = 0
    if max_clips > 0 and tuning.calibrated_rescue_deck_fraction > 0.0:
        rescued = [c for c in eligible if c.get("calibrated_floor_rescued")]
        if rescued:
            rescue_slots = int(np.ceil(
                max_clips * tuning.calibrated_rescue_deck_fraction
            ))
            rescued.sort(
                key=lambda c: float(
                    c.get("ranker_publishability_score", 0.0) or 0.0
                ),
                reverse=True,
            )
            for c in rescued[rescue_slots:]:
                c.pop("calibrated_floor_rescued", None)
                c["clears_bar"] = False
                c["selection_disposition"] = "adaptive_quality_floor"
                c["selection_rejection_reasons"] = ["adaptive_quality_floor"]
                c["selection_funnel"]["adaptive_quality"] = (
                    "rescue_budget_exhausted"
                )
                c["selection_funnel"]["final"] = "adaptive_quality_floor"
            keep_ids = {id(c) for c in rescued[:rescue_slots]}
            eligible = [
                c for c in eligible
                if not c.get("calibrated_floor_rescued") or id(c) in keep_ids
            ]

    # Viewer-command deck cap: commands keep absolute protection only for the
    # top slice of the budget; the rest compete on score. This is the measured
    # middle ground between binary protection (4 of 24 fixtures were
    # command-monopolized at 13-14/15 slots) and no protection (sole cause of
    # all 13 golden regressions in 2be3a3e).
    if max_clips > 0 and tuning.crowd_command_deck_fraction > 0.0:
        command_budget = int(np.ceil(
            max_clips * tuning.crowd_command_deck_fraction
        ))
        viewer_commands = sorted(
            (
                c for c in eligible
                if c.get("protected") and not _creator_protected(c)
            ),
            key=lambda c: float(c.get("selection_score", 0.0) or 0.0),
            reverse=True,
        )
        for c in viewer_commands[command_budget:]:
            c["protected"] = False
            c["selection_funnel"]["command_cap"] = "demoted_to_ordinary"

    # Explicit human clip commands first, then validated WIN anchors, then
    # everyone by selection_score. Verbal content anchors still ENTER via the
    # arousal-gate bypass + softer score lift, but they no longer leapfrog
    # the top of the deck — that slipped p@5 on the 2026-07-20 founder eval.
    # Win anchors keep the boolean boost: without it, the OCR-missed Victory
    # Royale fixture collapsed from recall 0.57 back toward baseline.
    eligible.sort(
        key=lambda c: (
            c["protected"],
            bool(c.get("content_anchor") and _visual_win_read(c)),
            c["selection_score"],
        ),
        reverse=True,
    )
    # Place the reserved rescues INSIDE the budget window. They sort last by
    # score (that is what being under the floor means), so without this they
    # would be positioned past the ceiling and the reservation would be a
    # no-op. They take the tail of the window, never the head: a rescue joins
    # the deck, it does not outrank the VOD's genuinely strong moments.
    deck_budget = max_clips
    if rescue_slots > 0 and max_clips > 0:
        reserved = [c for c in eligible if c.get("calibrated_floor_rescued")]
        if reserved:
            others = [
                c for c in eligible if not c.get("calibrated_floor_rescued")
            ]
            if tuning.calibrated_rescue_extends_deck:
                # Additive: the incumbent's own picks keep every slot they
                # had, and the deck grows by however many rescues qualified.
                deck_budget = max_clips + len(reserved)
                eligible = others[:max_clips] + reserved + others[max_clips:]
            else:
                head_room = max(0, max_clips - len(reserved))
                eligible = others[:head_room] + reserved + others[head_room:]

    # Global identity grouping happens before the budget. This is intentionally
    # earlier than Primary/Second Look splitting: a duplicate cannot become
    # ``clip_ceiling`` merely because its stronger sibling already filled the
    # creator-facing deck.
    eligible = _assign_moment_groups(eligible, tuning)
    eligible = _prefer_moment_representatives(eligible, tuning)
    for rank, candidate in enumerate(eligible, start=1):
        candidate["selection_rank"] = rank

    selected: List[dict] = []
    ordinary_count = 0
    overflow_count = 0
    second_look_slots_seen = 0
    overflow_limit = 0
    if max_clips > 0 and tuning.editorial_overflow_fraction > 0.0:
        overflow_limit = max(
            tuning.editorial_overflow_min,
            int(np.ceil(max_clips * tuning.editorial_overflow_fraction)),
        )
    for cand in eligible:
        # Hard total ceiling (opt-in): once the deck holds max_clips, every
        # further candidate — protected or not — is budget-cut. Protected
        # clips sorted first above, so explicit human markers still win the
        # budget before ordinary picks; they just cannot exceed it.
        if (
            tuning.hard_total_cap
            and deck_budget > 0
            and len(selected) >= deck_budget
        ):
            # Duplicate-of-a-shipped-clip must be labelled as such even once the
            # deck is full. Otherwise it lands in clip_ceiling, which is the
            # pool the Second-look tier draws from, and the creator is shown the
            # same moment twice — measured 2026-07-23, 3 of 15 second-look
            # moments were repeats at 100%/71%/100% overlap. Those get passed as
            # "already seen", which silently teaches the ranker that a KEPT
            # moment is a reject. Deliberately does NOT absorb: expanding a clip
            # the creator has already been shown would change a shipped window.
            duplicate = any(
                _overlap_fraction(cand, s) > tuning.nms_overlap
                for s in selected
            )
            identity_duplicate = any(
                cand.get("moment_group_id")
                and cand.get("moment_group_id") == s.get("moment_group_id")
                for s in selected
            )
            reason = (
                "overlap_suppressed"
                if duplicate
                else "moment_duplicate"
                if identity_duplicate
                else "clip_ceiling"
            )
            if tuning.editorial_hygiene_enabled:
                inside_second_look_band = (
                    second_look_slots_seen
                    < max(0, int(tuning.second_look_band_size))
                )
                # NB: duplicates consume a band slot ON PURPOSE — the band is a
                # positional window over the ranked tail, not a quota of
                # Second-look cards, so a duplicate must not pull a deeper
                # (worse) candidate up into the tier. See
                # test_duplicate_consumes_second_look_slot_without_deep_refill.
                second_look_slots_seen += 1
                if reason == "clip_ceiling" and not inside_second_look_band:
                    reason = "editorial_overflow_cutoff"
                elif (
                    reason == "clip_ceiling"
                    and inside_second_look_band
                ):
                    if (
                        not cand.get("protected")
                        and second_look_rejector is not None
                    ):
                        probability = (
                            second_look_rejector.predict_keep_probability(cand)
                        )
                        if probability is not None:
                            cand["second_look_keep_probability"] = float(
                                probability
                            )
                            cand["second_look_reject_threshold"] = float(
                                second_look_rejector.threshold
                            )
                            if probability < second_look_rejector.threshold:
                                reason = "predictable_pass"
            cand["selection_disposition"] = reason
            cand["selection_rejection_reasons"] = [reason]
            cand["selection_funnel"]["final"] = reason
            continue
        # Same-moment absorb: overlap OR a short gap between adjacent arcs
        # (elim→celebrate, multi-peak fights). Expand the kept window so the
        # deck ships one start→finish clip instead of near-duplicates.
        nearby = next((
            s for s in selected
            if _nearby_moment_pair(
                cand, s,
                overlap_thresh=tuning.nms_overlap,
                gap_sec=tuning.nearby_moment_gap_sec,
                max_span_sec=tuning.nearby_moment_max_span_sec,
            )
        ), None)
        if nearby is not None:
            was_overlap = _overlap_fraction(cand, nearby) > tuning.nms_overlap
            _absorb_nearby_moment(nearby, cand)
            reason = (
                "overlap_suppressed" if was_overlap else "nearby_moment_merged"
            )
            cand["selection_disposition"] = reason
            cand["selection_rejection_reasons"] = [reason]
            cand["selection_funnel"]["final"] = reason
            cand["selection_suppressed_by"] = {
                "start": float(nearby.get("start", 0.0)),
                "end": float(nearby.get("end", 0.0)),
            }
            continue
        # Classic overlap suppress when the union would exceed the absorb span
        # cap (still drop the weaker duplicate; do not grow a mega-clip).
        overlapping = next((
            s for s in selected
            if _overlap_fraction(cand, s) > tuning.nms_overlap
        ), None)
        if overlapping is not None:
            cand["selection_disposition"] = "overlap_suppressed"
            cand["selection_rejection_reasons"] = ["overlap_suppressed"]
            cand["selection_funnel"]["final"] = "overlap_suppressed"
            cand["selection_suppressed_by"] = {
                "start": float(overlapping.get("start", 0.0)),
                "end": float(overlapping.get("end", 0.0)),
            }
            continue
        identity_match = next(
            (
                s for s in selected
                if cand.get("moment_group_id")
                and cand.get("moment_group_id") == s.get("moment_group_id")
                and cand.get("moment_group_source") == "tokens"
            ),
            None,
        )
        if identity_match is not None:
            cand["selection_disposition"] = "moment_duplicate"
            cand["selection_rejection_reasons"] = ["moment_duplicate"]
            cand["selection_funnel"]["final"] = "moment_duplicate"
            cand["selection_suppressed_by"] = {
                "start": float(identity_match.get("start", 0.0)),
                "end": float(identity_match.get("end", 0.0)),
                "moment_group_id": identity_match.get("moment_group_id"),
            }
            continue
        if not cand["protected"]:
            if (
                local_event_rich.get(id(cand), False)
                and cand.get("game_label") is None
                and not _event_rich_ordinary_allowed(cand, tuning)
            ):
                cand["selection_disposition"] = "event_rich_admission"
                cand["selection_rejection_reasons"] = ["event_rich_admission"]
                cand["selection_funnel"]["final"] = "event_rich_admission"
                continue
            at_primary_ceiling = (
                deck_budget >= 0 and ordinary_count >= deck_budget
            )
            if at_primary_ceiling:
                editorial_overflow = (
                    overflow_count < overflow_limit
                    and (
                        cand.get("semantic_rescue")
                        or cand.get("strong_startle")
                        or cand.get("visible_crowd_anchor")
                        or cand.get("content_anchor")
                    )
                )
                if not editorial_overflow:
                    cand["selection_disposition"] = "clip_ceiling"
                    cand["selection_rejection_reasons"] = ["clip_ceiling"]
                    cand["selection_funnel"]["final"] = "clip_ceiling"
                    continue
                overflow_count += 1
                cand["selection_disposition"] = "selected_editorial_overflow"
            else:
                ordinary_count += 1
                cand["selection_disposition"] = "selected_primary"
        else:
            cand["selection_disposition"] = "selected_protected"
        cand["selection_funnel"]["final"] = cand["selection_disposition"]
        selected.append(cand)

    _rebalance_temporal_blackouts(selected, eligible, max_clips, tuning)
    selected.sort(key=lambda c: c["start"])
    return selected
