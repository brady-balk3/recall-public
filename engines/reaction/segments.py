# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Temporal game segmentation (plan 22 §4.1) — kill the one-game-per-VOD assumption.

Real streams are non-stationary: Just Chatting -> game A -> game B, each with
its own HUD vocabulary and lobby lexicon. Whole-VOD game detection gives a
variety stream ONE global label that's wrong for most of the runtime, which
misroutes OCR event triggers and scene classification for every clip outside
the dominant game.

This module reuses the existing fingerprint scorer over sliding windows of the
VOD's own (spatially routed) OCR text and merges the per-window winners into a
labeled timeline with hysteresis — a killcam or a chat message can't flip the
segment; a *sustained* fingerprint change can. Windows with no confident
winner become "generic" (Just Chatting, unknown titles, no-HUD games), which
downstream treats exactly like today's global fallback.

Cheap by construction: no new models, no frame reads — just re-scoring text
the pipeline already captured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from engines.reaction.game_detect import qualified_game, score_games
from engines.vision.layout_map import routed_text

WINDOW_SEC = 600.0   # fingerprints are sparse (banners/killfeed); a 10-min
HOP_SEC = 300.0      # window collects enough distinct hits to be confident
# A new game must win this many CONSECUTIVE windows to start a segment —
# one window is a killcam / stream raid / chat quote away from a false flip.
HYSTERESIS_WINDOWS = 2
# Bridge a short generic dip between two windows of the same game (banner-less
# midgame lull) instead of splitting the segment.
MAX_GENERIC_BRIDGE_WINDOWS = 1


@dataclass
class GameSegment:
    start: float
    end: float
    game: str = "generic"

    def to_dict(self) -> Dict:
        return {"start": self.start, "end": self.end, "game": self.game}


def _window_winner(texts: List[str]) -> str:
    hits = score_games(texts)
    winner, _confidence = qualified_game(hits)
    return winner


def detect_game_segments(unified_signals, window_sec: float = WINDOW_SEC,
                         hop_sec: float = HOP_SEC) -> List[GameSegment]:
    """Label the VOD timeline with per-window game detection + hysteresis."""
    stamped: List[tuple] = []
    duration = 0.0
    for sig in unified_signals:
        duration = max(duration, float(sig.timestamp))
        if getattr(sig, "ocr", None) and getattr(sig.ocr, "text", None):
            text = routed_text(sig.ocr)
            if text:
                stamped.append((float(sig.timestamp), text))
    if duration <= 0:
        return []

    # Per-hop winners.
    winners: List[str] = []
    hop_starts: List[float] = []
    t = 0.0
    while t < duration:
        lo, hi = t, t + window_sec
        winners.append(_window_winner([txt for (ts, txt) in stamped if lo <= ts < hi]))
        hop_starts.append(t)
        t += hop_sec

    if not winners:
        return [GameSegment(0.0, duration, "generic")]

    # Hysteresis: adopt a NEW game only after it wins HYSTERESIS_WINDOWS in a
    # row; otherwise stay on the current label. Generic never needs votes to
    # "win" — it's the absence of evidence — but short generic dips between
    # same-game windows are bridged below.
    smoothed: List[str] = []
    current = winners[0]
    streak_game: Optional[str] = None
    streak = 0
    for w in winners:
        if w == current:
            streak_game, streak = None, 0
        elif w == streak_game:
            streak += 1
            if streak >= HYSTERESIS_WINDOWS:
                current = w
                streak_game, streak = None, 0
        else:
            streak_game, streak = w, 1
            if HYSTERESIS_WINDOWS <= 1:
                current = w
                streak_game, streak = None, 0
        smoothed.append(current)

    # Bridge short generic dips inside one game's run.
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] == "generic":
            left = smoothed[i - 1]
            j = i
            while j < len(smoothed) and smoothed[j] == "generic":
                j += 1
            if (left != "generic" and j < len(smoothed) and smoothed[j] == left
                    and (j - i) <= MAX_GENERIC_BRIDGE_WINDOWS):
                for k in range(i, j):
                    smoothed[k] = left

    # Compress consecutive hops into segments.
    segments: List[GameSegment] = []
    seg_start = 0.0
    for i in range(1, len(smoothed) + 1):
        if i == len(smoothed) or smoothed[i] != smoothed[i - 1]:
            seg_end = duration if i == len(smoothed) else hop_starts[i]
            segments.append(GameSegment(seg_start, seg_end, smoothed[i - 1]))
            seg_start = seg_end
    return segments


class SegmentIndex:
    """O(1)-ish lookup of the game at a timestamp; falls back to a default."""

    def __init__(self, segments: Optional[List] = None, default: str = "generic"):
        self.default = default
        self.segments: List[GameSegment] = []
        for seg in segments or []:
            if isinstance(seg, GameSegment):
                self.segments.append(seg)
            elif isinstance(seg, dict):
                self.segments.append(GameSegment(
                    float(seg.get("start", 0.0)), float(seg.get("end", 0.0)),
                    str(seg.get("game", "generic")),
                ))

    def game_at(self, timestamp: Optional[float]) -> str:
        if timestamp is None:
            return self.default
        for seg in self.segments:
            if seg.start <= timestamp < seg.end:
                return seg.game if seg.game != "generic" else self.default
        return self.default

    @property
    def distinct_games(self) -> List[str]:
        seen: List[str] = []
        for seg in self.segments:
            if seg.game != "generic" and seg.game not in seen:
                seen.append(seg.game)
        return seen
