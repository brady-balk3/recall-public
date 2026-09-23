# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Shared production defaults for reaction-engine calls.

The orchestrator and cached-track evaluation harness must agree on the deck
size and core engine knobs.  Keep this module dependency-light so the golden
harness can import the real policy without importing the full video pipeline.
"""

import math


DEFAULT_SENSITIVITY = 0.70
DEFAULT_AGGRESSIVENESS = 0.60
DEFAULT_SMOOTH_WINDOW = 5
DEFAULT_SELECTION_MODE = "review"
# Interim review policy while the learned ranker is still below its ship gate:
# keep the creator-facing deck tight, but retain every cut candidate in the
# persisted selection trace for diagnostics, replay learning, and recovery.
#
# Lowered 15 -> 10 on 2026-08-16. Measured over the 60-fixture founder set with
# the ranker active, this trades recall for precision deliberately:
#
#   cap 15   14.6 clips   precision 0.364   recall 0.497   resurface 0.213
#   cap 10    9.9 clips   precision 0.404   recall 0.396   resurface 0.143
#
# Per deck that is ~1.3 fewer keeps against ~3.4 fewer unjustified clips, and a
# third less re-surfacing of moments the creator already passed on. p@5 is
# unchanged at 0.457, so the top of the deck is identical either way -- this
# only decides what happens in the lower slots.
#
# Note this is the deck TOTAL, not a primary-only target: editorial overflow is
# budgeted as max(3, ceil(max_clips * editorial_overflow_fraction)), so the
# Second Look allowance moves 6 -> 4 slots with it. Second Look still runs;
# it just has less room behind a smaller primary deck.
REVIEW_DECK_HARD_CAP = 10


def auto_clip_count(duration: float, aggressiveness: float) -> int:
    """Production deck budget derived from VOD duration and density setting."""
    if duration <= 0:
        return 8

    minutes = duration / 60.0
    clips_per_10_minutes = 2.0 + float(aggressiveness) * 5.0
    target = int(math.ceil((minutes / 10.0) * clips_per_10_minutes))
    return max(6, min(30, target))


def production_engine_settings(duration: float, overrides=None) -> dict:
    """Return the effective settings used by a normal production scan.

    ``overrides`` uses reaction-engine normalized names for evaluation and
    internal tooling. Creator sensitivity/density settings no longer exist.
    """
    raw = dict(overrides or {})
    sensitivity = float(raw.get("sensitivity", DEFAULT_SENSITIVITY))
    aggressiveness = float(raw.get("aggressiveness", DEFAULT_AGGRESSIVENESS))
    for legacy_key in (
        "detectionSensitivity", "highlightAggressiveness", "clipLength",
    ):
        raw.pop(legacy_key, None)

    settings = {
        "sensitivity": max(0.0, min(1.0, sensitivity)),
        "aggressiveness": max(0.0, min(1.0, aggressiveness)),
        "smooth_window": DEFAULT_SMOOTH_WINDOW,
        "selection_mode": DEFAULT_SELECTION_MODE,
    }
    settings.update(raw)
    settings["sensitivity"] = max(0.0, min(1.0, sensitivity))
    settings["aggressiveness"] = max(0.0, min(1.0, aggressiveness))
    settings.setdefault("max_clips", auto_clip_count(duration, settings["aggressiveness"]))
    mode = str(settings.get("selection_mode", DEFAULT_SELECTION_MODE)).strip().lower()
    if mode not in {"auto", "post", "publish"}:
        budget = int(settings["max_clips"])
        if budget > 0:
            settings["max_clips"] = min(budget, REVIEW_DECK_HARD_CAP)
    return settings


def build_pipeline_reaction_settings(
    duration: float,
    pipeline_settings: dict,
    raw_settings: dict,
    game_segments,
    *,
    fast_mode: bool,
) -> dict:
    """Build the final selector settings used by a production scan.

    Keeping this adapter beside the production policy makes it difficult for
    the orchestrator to accidentally recreate only part of that policy. Game
    segments are accepted as serialized dictionaries so this module remains
    dependency-light for tests and evaluation harnesses.
    """
    selection_mode = pipeline_settings.get(
        "selection_mode",
        raw_settings.get("selectionMode", raw_settings.get("selection_mode", DEFAULT_SELECTION_MODE)),
    )
    overrides = {
        "sensitivity": pipeline_settings["sensitivity"],
        "aggressiveness": pipeline_settings["aggressiveness"],
        "smooth_window": DEFAULT_SMOOTH_WINDOW,
        "selection_mode": selection_mode,
    }
    if "max_clips" in raw_settings:
        overrides["max_clips"] = raw_settings["max_clips"]

    resolved = production_engine_settings(duration, overrides)
    resolved.update({
        "game": pipeline_settings.get("game", "generic"),
        "game_segments": list(game_segments),
        "content_shape_selection": True,
    })
    # Calibrated admission keys are read by build_reaction_clips, but this
    # function is a whitelist -- anything not named here never reaches the
    # engine. Omitting them made the lane silently inert: a scan could be
    # launched with ranker_calibrated_union set and produce a byte-identical
    # baseline deck, which is exactly what happened on 2026-08-16 (two 4-hour
    # scans, no experiment). Pass them through when present so an opt-in scan
    # actually opts in; absent, behaviour is unchanged.
    for key in (
        "ranker_calibrated_admission",
        "ranker_calibrated_union",
        "ranker_publishability_floor",
        "ranker_calibrated_gate_model",
    ):
        if key in raw_settings:
            resolved[key] = raw_settings[key]
    return resolved
