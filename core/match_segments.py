# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Recover individual matches from a scanned session's detected events.

A compilation of "that 29-kill game" is impossible while the only unit Recall
exposes is the whole VOD. The scan already detects every elimination and every
terminal screen and stores them as ``stream_memory_entries`` evidence rows, so
the match boundaries are recoverable without loading video, running a model, or
touching the scan pipeline: this module is a pure read over rows that already
exist.

Four independent signals mark a boundary, because no single one is reliable:

* a lobby screen between games;
* a terminal card -- ``GAME OVER``, ``YOU PLACED #12``, ``VICTORY ROYALE``;
* Fortnite's in-match elimination milestones ("20 Eliminations in a match"),
  which reset every game, so a counter that falls has crossed a boundary; and
* dead air longer than any realistic gap between kills in one match.

Kill counts are reported as a range. The detector logs discrete elimination
events, but rapid kills collapse into one ``Double Elimination saved`` event,
so the milestone counter is routinely the higher and more honest number.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# A match ends when the screen says so. Placement text lingers on screen for
# several seconds, so the same card is often detected repeatedly; segmentation
# de-duplicates on the matched text rather than closing a match per frame.
TERMINAL_RE = re.compile(
    r"GAME OVER|YOU\s*PLACED\s*#?\s*(\d+)|VICTORY\s*ROYALE|#\s*1\s+VICTORY",
    re.IGNORECASE,
)
PLACEMENT_RE = re.compile(r"YOU\s*PLACED\s*#?\s*(\d+)", re.IGNORECASE)
WIN_RE = re.compile(r"VICTORY\s*ROYALE|#\s*1\s+VICTORY", re.IGNORECASE)
# "20 Eliminations in a match", "5 Eliminations in a single game".
MILESTONE_RE = re.compile(r"(\d+)\s+Elimination[s]?\s+in\s+a", re.IGNORECASE)
# "Double Elimination saved", "Triple Elimination", "3 eliminations within a".
MULTI_RE = re.compile(
    r"\b(double|triple|quadruple|quad|multi)\s+elimination"
    r"|(\d+)\s+eliminations?\s+with(?:in)?\b",
    re.IGNORECASE,
)
LOBBY_RE = re.compile(r"\blobby\b", re.IGNORECASE)

MULTI_WORD_COUNT = {"double": 2, "triple": 3, "quad": 4, "quadruple": 4, "multi": 2}

# Longer than any believable pause between two kills in one match. Chosen from
# the observed gap distribution: within-match gaps cluster under 180s, while
# the pause across a lobby, queue and drop runs past 280s.
DEAD_AIR_SECONDS = 300.0
# A fragment this small with no eliminations is a lingering terminal card or a
# stray menu detection, not a match the creator played.
SLIVER_EVENT_COUNT = 2
# Longer than a battle-royale match runs. A segment past this has almost
# certainly swallowed a boundary Recall could not see, so it is still offered
# but never as a confident one.
LONG_MATCH_SECONDS = 1500.0


@dataclass(frozen=True)
class MatchMoment:
    """One detected event inside a match."""

    start: float
    end: float
    kind: str  # elimination | win | terminal | lobby | event
    title: str
    ocr_phrases: tuple[str, ...] = ()
    gameplay_events: tuple[str, ...] = ()
    reactions: tuple[str, ...] = ()
    audio_peak: float = 0.0
    motion_peak: float = 0.0
    milestone: Optional[int] = None
    multi_kill: int = 1

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class MatchSegment:
    """One game, bounded by the signals above."""

    index: int
    start: float
    end: float
    moments: tuple[MatchMoment, ...]
    outcome: str = "unknown"  # win | placed | eliminated | unknown
    placement: Optional[int] = None
    confidence: str = "low"  # high | medium | low
    boundary_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def eliminations(self) -> tuple[MatchMoment, ...]:
        return tuple(m for m in self.moments if m.kind == "elimination")

    @property
    def win_moment(self) -> Optional[MatchMoment]:
        for moment in self.moments:
            if moment.kind == "win":
                return moment
        return None

    @property
    def detected_kills(self) -> int:
        """Discrete elimination events, counting a collapsed multi-kill once each."""
        return sum(m.multi_kill for m in self.eliminations)

    @property
    def milestone_kills(self) -> int:
        """Highest in-game elimination milestone reached this match."""
        return max((m.milestone or 0 for m in self.moments), default=0)

    @property
    def kill_estimate(self) -> int:
        """Best single number for the match, favouring the in-game counter."""
        return max(self.detected_kills, self.milestone_kills)

    @property
    def has_win(self) -> bool:
        return self.outcome == "win"

    @property
    def kills_are_a_floor(self) -> bool:
        """True when the count comes from a milestone rather than counted events.

        The in-game milestones fire at 5, 10, 15, 20, 25 and so on, so the
        highest one reached is the last threshold crossed, not the final total:
        a match that ends on 29 kills reports 25. Saying "25" flat would be a
        claim Recall cannot make.
        """
        return self.milestone_kills > self.detected_kills

    def label(self) -> str:
        kills = self.kill_estimate
        count = f"{kills}+ kills" if self.kills_are_a_floor else f"{kills} kills"
        if self.outcome == "win":
            return f"{count} - Victory" if kills else "Victory"
        if self.outcome == "placed" and self.placement:
            return f"{count} - #{self.placement}" if kills else f"#{self.placement}"
        return count if kills else "Match"


def _float(value: Any, key: str) -> float:
    try:
        return float(value.get(key) or 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item or "").strip())


def _milestone(text: str) -> Optional[int]:
    found = [int(match) for match in MILESTONE_RE.findall(text)]
    return max(found) if found else None


def _multi_kill(text: str) -> int:
    best = 1
    for word, digits in MULTI_RE.findall(text):
        if word:
            best = max(best, MULTI_WORD_COUNT.get(word.lower(), 2))
        elif digits:
            # Guard against quest copy like "50 eliminations with an SMG",
            # which is a season goal rather than one collapsed multi-kill.
            count = int(digits)
            if 2 <= count <= 5:
                best = max(best, count)
    return best


def _moment(row: dict[str, Any]) -> MatchMoment:
    try:
        payload = json.loads(row.get("evidence_json") or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    text = str(row.get("text") or "")
    title = str(row.get("title") or "").strip()
    ocr = _strings(payload.get("ocr_phrases"))
    events = tuple(item.lower() for item in _strings(payload.get("gameplay_events")))
    reactions = _strings(payload.get("reactions"))
    raw_signals = payload.get("signal_summary")
    signals = raw_signals if isinstance(raw_signals, dict) else {}
    ocr_text = " ".join(ocr) or text

    if WIN_RE.search(ocr_text) or "terminal win" in events:
        kind = "win"
    elif title.lower() == "elimination" or "elimination" in events:
        kind = "elimination"
    elif title.lower() == "lobby" or (LOBBY_RE.search(text) and "elimination" not in events):
        kind = "lobby"
    elif TERMINAL_RE.search(ocr_text):
        kind = "terminal"
    else:
        kind = "event"

    return MatchMoment(
        start=float(row.get("start_time") or 0.0),
        end=float(row.get("end_time") or row.get("start_time") or 0.0),
        kind=kind,
        title=title or "Detected event",
        ocr_phrases=ocr,
        gameplay_events=events,
        reactions=reactions,
        audio_peak=_float(signals, "max_audio_spike"),
        motion_peak=_float(signals, "max_motion"),
        milestone=_milestone(text),
        multi_kill=_multi_kill(text) if kind == "elimination" else 1,
    )


def moment_from_evidence(row: dict[str, Any]) -> MatchMoment:
    """Score one detected event on its own, outside any match.

    A themed compilation ranks moments drawn from across the library, where
    there is no match to segment -- but the same evidence shape, and the same
    reading of it, still applies.
    """
    return _moment(row)


def load_moments(db, job_id: str) -> list[MatchMoment]:
    """Every detected event for one job, in stream order."""
    with db.get_connection() as conn:
        rows = conn.execute(
            """SELECT start_time, end_time, title, text, evidence_json
               FROM stream_memory_entries
               WHERE job_id = ? AND kind = 'evidence'
               ORDER BY start_time, id""",
            (job_id,),
        ).fetchall()
    return [_moment(dict(row)) for row in rows]


def _terminal_text(moment: MatchMoment) -> str:
    """The terminal card this moment shows, or '' when it shows none."""
    haystack = " ".join(moment.ocr_phrases)
    found = TERMINAL_RE.search(haystack)
    return found.group(0).upper().strip() if found else ""


def _finish(moments: list[MatchMoment], index: int, reasons: list[str]) -> MatchSegment:
    outcome, placement = "unknown", None
    for moment in moments:
        if moment.kind == "win":
            outcome, placement = "win", 1
            break
        haystack = " ".join(moment.ocr_phrases)
        placed = PLACEMENT_RE.search(haystack)
        if placed and outcome == "unknown":
            outcome, placement = "placed", int(placed.group(1))
        elif re.search(r"GAME OVER", haystack, re.IGNORECASE) and outcome == "unknown":
            outcome = "eliminated"

    # A match Recall can name the end of, and that carries a real kill count,
    # is one the creator will recognise. Everything else is offered quietly.
    elim_count = sum(1 for m in moments if m.kind == "elimination")
    if outcome in {"win", "placed"} and elim_count >= 3:
        confidence = "high"
    elif outcome != "unknown" or elim_count >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    start = min(m.start for m in moments)
    end = max(m.end for m in moments)
    if outcome == "win" and elim_count == 0:
        # A victory card with no kills under it is a shop or career screen the
        # detector read as a win, not a game that was played.
        confidence = "low"
    if end - start > LONG_MATCH_SECONDS and confidence == "high":
        confidence = "medium"

    return MatchSegment(
        index=index,
        start=start,
        end=end,
        moments=tuple(moments),
        outcome=outcome,
        placement=placement,
        confidence=confidence,
        boundary_reasons=tuple(reasons),
    )


def segment_matches(db, job_id: str) -> list[MatchSegment]:
    """Split one session's detected events into the matches that produced them."""
    return segment_moments(load_moments(db, job_id))


@dataclass
class _Group:
    """A match under construction, plus why it was cut where it was."""

    moments: list[MatchMoment] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # A group closed by its own terminal card is a finished game. Nothing that
    # happens afterwards belongs to it, so stray events never fold back in.
    closed_terminally: bool = False


def segment_moments(moments: list[MatchMoment]) -> list[MatchSegment]:
    """Pure segmentation, so the boundary rules stay testable without a database."""
    groups: list[_Group] = []
    current = _Group()
    highest_milestone = 0
    last_terminal = ""
    previous_end: Optional[float] = None

    def close(reason: str, *, terminally: bool = False) -> None:
        nonlocal current, highest_milestone
        if current.moments:
            if reason:
                current.reasons.append(reason)
            current.closed_terminally = terminally
            groups.append(current)
        current = _Group()
        highest_milestone = 0

    for moment in moments:
        gap = moment.start - previous_end if previous_end is not None else 0.0
        previous_end = max(previous_end or 0.0, moment.end)
        terminal = _terminal_text(moment)

        # The same terminal card again, with nothing since, is the previous
        # result still on screen -- not a new game to open.
        if not current.moments and terminal and terminal == last_terminal and groups:
            groups[-1].moments.append(moment)
            continue

        opens = ""
        if moment.kind == "lobby":
            opens = "lobby screen"
        elif moment.milestone is not None and moment.milestone < highest_milestone:
            # The in-game counter only ever climbs inside one match.
            opens = "elimination counter reset"
        elif gap > DEAD_AIR_SECONDS and current.moments:
            opens = f"{int(gap // 60)} minute gap"
        if opens:
            close("")
            current.reasons = [opens]

        current.moments.append(moment)
        if moment.milestone is not None:
            highest_milestone = max(highest_milestone, moment.milestone)

        if moment.kind == "win":
            last_terminal = terminal or "VICTORY ROYALE"
            close("victory", terminally=True)
        elif terminal and terminal != last_terminal:
            last_terminal = terminal
            close(f"terminal screen ({terminal.lower()})", terminally=True)
    close("")

    return _merge_slivers(groups)


def _merge_slivers(groups: list[_Group]) -> list[MatchSegment]:
    """Fold a stray detection into the match it trails, when one is open to it."""
    merged: list[_Group] = []
    for group in groups:
        carries_nothing = (
            len(group.moments) < SLIVER_EVENT_COUNT
            and not any(m.kind == "win" for m in group.moments)
            and not any(m.milestone for m in group.moments)
        )
        if carries_nothing and merged and not merged[-1].closed_terminally:
            merged[-1].moments.extend(group.moments)
            continue
        merged.append(group)

    return [
        _finish(group.moments, index, group.reasons)
        for index, group in enumerate(merged, start=1)
        if group.moments
    ]


def moment_strength(moment: MatchMoment) -> float:
    """How much one kill is worth in a compilation of the same match.

    Every kill in a match is the same event, so the ordinary selection scores
    do not separate them. What does separate them is how hard the creator and
    the room reacted, and whether the kill was a multi or a clutch.
    """
    score = 1.0
    score += 0.45 * min(moment.multi_kill - 1, 3)
    score += 0.30 * len([r for r in moment.reactions if "burst" in r.lower()])
    score += 0.20 * len([r for r in moment.reactions if "audio spike" in r.lower()])
    score += 0.35 * min(moment.audio_peak, 3.0)
    score += 0.25 * min(moment.motion_peak, 1.0)
    for tag, weight in (
        ("clutch", 0.7), ("story", 0.4), ("funny", 0.35),
        ("rage", 0.3), ("scare", 0.25), ("wholesome", 0.2), ("fail", -0.2),
    ):
        if tag in moment.gameplay_events:
            score += weight
    if moment.milestone:
        score += 0.15
    return round(score, 4)
