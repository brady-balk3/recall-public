# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional

from core.match_segments import (
    MatchMoment,
    MatchSegment,
    moment_from_evidence,
    moment_strength,
    segment_matches,
)
from core.stream_memory import StreamMemoryService


PROJECT_SCHEMA_VERSION = 1
GENERATION_VERSION = 1
MAX_PROJECT_CLIPS = 40
MAX_PER_SESSION = 8

# Montage pacing. A kill reads in a few seconds -- the shot, the hit marker,
# the reaction -- so the reel fits more of them by cutting tight rather than by
# running longer. These bound the per-kill length the target duration implies.
MIN_KILL_SECONDS = 6.0
MAX_KILL_SECONDS = 9.0
# Recall marks a kill when it reads the ELIMINATION banner, which appears after
# the shot -- so the cut leads its anchor. How far it must lead was measured
# against the on-screen kill counter, which increments the instant a kill
# registers: one marker landed exactly on its kill, one ran two seconds late,
# and one was at least six seconds late (the counter had already moved before
# the window opened). The detection is coarse, so the lead is a SHARE of the
# clip rather than a fixed offset: a longer clip buys a longer lead and covers
# more of that spread. Anything under about six seconds cannot.
KILL_LEAD_SHARE = 0.72
KILL_TAIL_SECONDS = 2.0
# The banner stays on screen and is re-read on a fixed cadence, so one kill can
# produce several markers. Verified against the on-screen kill counter: pairs
# ten and eleven seconds apart were one kill counted twice (the count went
# 22 -> 23 inside the FIRST window and stayed 23 through the second). A match
# offers far more markers than a reel has slots, so a duplicate -- a clip with
# no kill in it -- costs much more than a genuinely fast kill that gets merged.
MARKER_MERGE_SECONDS = 12.0
DEFAULT_REEL_SECONDS = 60.0
# The closing win keeps its whole payoff (victory, reaction, XP tally) unless
# that would swallow the reel; then it keeps the END of it, which is where the
# reaction lives.
WIN_SHARE_OF_REEL = 0.25
MIN_WIN_SECONDS = 10.0
# A kept clip arrives as a finished render, so unlike a detected event it
# cannot be tightened without re-encoding it. One that would eat a third of the
# reel is simply the wrong shape for a montage and is left out rather than
# allowed to blow the requested length.
MAX_CLIP_SHARE_OF_REEL = 0.35
MIN_CLIP_BUDGET_SECONDS = 12.0
_TITLE_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


class CompilationProjectError(RuntimeError):
    pass


class CompilationProjectNotFound(CompilationProjectError):
    pass


class CompilationProjectConflict(CompilationProjectError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split()).strip()[:limit]


def _parse_date(value: Optional[str], label: str) -> Optional[str]:
    cleaned = _clean(value, 10)
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned).isoformat()
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY-MM-DD.") from exc


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _shorten(value: str, limit: int) -> str:
    """Trim to a whole word, so a spoken line reads as a title rather than a cut-off."""
    if len(value) <= limit:
        return value
    trimmed = value[:limit].rsplit(" ", 1)[0].rstrip(",;:. ")
    return f"{trimmed or value[:limit]}…"


def _normalized_title(value: str) -> str:
    return " ".join(_TITLE_WORD_RE.findall(value.lower()))


def _overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left["job_id"] != right["job_id"]:
        return 0.0
    overlap = max(
        0.0,
        min(float(left["end_time"]), float(right["end_time"]))
        - max(float(left["start_time"]), float(right["start_time"])),
    )
    shorter = min(
        max(0.1, float(left["end_time"]) - float(left["start_time"])),
        max(0.1, float(right["end_time"]) - float(right["start_time"])),
    )
    return overlap / shorter


class CompilationProjectService:
    """Durable, local-first automatic compilation drafts.

    Generation is intentionally metadata-only. It reads creator-kept clips and
    Stream Memory evidence, writes a small editable manifest, and never loads a
    model, source recording, or renderer.
    """

    def __init__(self, db, memory: Optional[StreamMemoryService] = None):
        self.db = db
        self.memory = memory or StreamMemoryService(db)

    def _project_row(self, project_id: str) -> dict[str, Any]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM compilation_projects WHERE id = ?", (project_id,),
            ).fetchone()
        if row is None:
            raise CompilationProjectNotFound("Compilation project not found.")
        return dict(row)

    @staticmethod
    def _candidate_score(row: dict[str, Any], retrieval_rank: Optional[int]) -> float:
        selection = float(
            row.get("selection_score")
            if row.get("selection_score") is not None
            else row.get("deck_score")
            if row.get("deck_score") is not None
            else row.get("score") or 0.0
        )
        hook = float(row.get("hook_score") or row.get("score") or 0.0)
        reaction = float(row.get("reaction_auc") or 0.0)
        retrieval = 2.0 / (1.0 + float(retrieval_rank or 0)) if retrieval_rank else 0.0
        return selection + (0.25 * hook) + (0.10 * reaction) + retrieval

    def _candidates(
        self,
        *,
        query: str,
        game: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> list[dict[str, Any]]:
        retrieval: dict[str, dict[str, Any]] = {}
        if query:
            result = self.memory.search(
                query,
                kind="clip",
                decision="kept",
                game=game,
                date_from=date_from,
                date_to=date_to,
                limit=100,
                mode="hybrid",
            )
            retrieval = {
                str(item["clip_id"]): {
                    "rank": index,
                    "reason": item.get("match_reason") or "Memory match",
                }
                for index, item in enumerate(result["results"], start=1)
                if item.get("clip_id")
            }
            if not retrieval:
                return []

        clauses = ["COALESCE(c.kept, 0) = 1", "COALESCE(c.passed, 0) = 0"]
        params: list[Any] = []
        if retrieval:
            placeholders = ",".join("?" for _ in retrieval)
            clauses.append(f"c.id IN ({placeholders})")  # nosec B608 - placeholders only
            params.extend(retrieval)
        if date_from:
            clauses.append("COALESCE(j.source_date, substr(j.created_at, 1, 10)) >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("COALESCE(j.source_date, substr(j.created_at, 1, 10)) <= ?")
            params.append(date_to)
        if game:
            clauses.append("LOWER(COALESCE(m.game, '')) = LOWER(?)")
            params.append(game)

        sql = f"""
            SELECT c.id, c.job_id, c.title, c.start_time, c.end_time,
                   c.selection_score, c.deck_score, c.hook_score, c.score,
                   c.reaction_auc, c.export_path, c.preview_only,
                   j.session_name, j.source_date, j.created_at,
                   j.source_path, j.asset_path, COALESCE(m.game, '') AS game,
                   COALESCE(sa.source_key, NULLIF(j.source_path, ''), j.id) AS diversity_key
            FROM clips AS c
            JOIN jobs AS j ON j.id = c.job_id
            LEFT JOIN stream_memory_entries AS m
              ON m.clip_id = c.id AND m.kind = 'clip'
            LEFT JOIN source_asset_sessions AS sa ON sa.job_id = j.id
            WHERE {' AND '.join(clauses)}
            GROUP BY c.id
            ORDER BY COALESCE(j.source_date, substr(j.created_at, 1, 10)),
                     c.start_time, c.id
        """  # nosec B608 - clauses are fixed and values stay parameterized
        with self.db.get_connection() as conn:
            rows = [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]

        for row in rows:
            memory = retrieval.get(str(row["id"]), {})
            row["retrieval_rank"] = memory.get("rank")
            row["match_reason"] = memory.get("reason")
            row["project_score"] = self._candidate_score(row, memory.get("rank"))
            row["effective_date"] = row.get("source_date") or str(row.get("created_at") or "")[:10]
        return rows

    @staticmethod
    def _select(
        candidates: list[dict[str, Any]],
        *,
        max_clips: int,
        max_per_session: int,
        target_duration_seconds: Optional[float],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ranked = sorted(
            candidates,
            key=lambda row: (
                row.get("retrieval_rank") or 10_000,
                -float(row["project_score"]),
                str(row["id"]),
            ),
        )
        selected: list[dict[str, Any]] = []
        per_session: dict[str, int] = {}
        seen_titles: set[str] = set()
        duplicate_count = 0
        duration = 0.0
        for candidate in ranked:
            job_id = str(candidate["job_id"])
            diversity_key = str(candidate.get("diversity_key") or job_id)
            if per_session.get(diversity_key, 0) >= max_per_session:
                continue
            normalized_title = _normalized_title(str(candidate.get("title") or ""))
            is_duplicate = (
                bool(normalized_title and normalized_title in seen_titles)
                or any(_overlap_ratio(candidate, prior) >= 0.55 for prior in selected)
            )
            if is_duplicate:
                duplicate_count += 1
                continue
            clip_duration = max(
                0.1,
                float(candidate["end_time"] or 0.0) - float(candidate["start_time"] or 0.0),
            )
            if (
                target_duration_seconds
                and selected
                and duration + clip_duration > target_duration_seconds
            ):
                continue
            selected.append(candidate)
            duration += clip_duration
            per_session[diversity_key] = per_session.get(diversity_key, 0) + 1
            if normalized_title:
                seen_titles.add(normalized_title)
            if len(selected) >= max_clips:
                break

        selected.sort(
            key=lambda row: (
                str(row.get("effective_date") or ""),
                str(row.get("session_name") or "").lower(),
                float(row.get("start_time") or 0.0),
                str(row["id"]),
            )
        )
        return selected, {
            "candidates_considered": len(candidates),
            "duplicates_suppressed": duplicate_count,
            "sessions_represented": len({row.get("diversity_key") or row["job_id"] for row in selected}),
            "selected_duration_seconds": round(duration, 3),
            "selection_policy": "creator_kept_only",
        }

    def generate(
        self,
        *,
        title: str,
        query: str = "",
        game: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_clips: int = 12,
        max_per_session: int = 3,
        target_duration_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        title = _clean(title, 160)
        if not title:
            raise ValueError("Give this compilation a title.")
        query = _clean(query, 240)
        game = _clean(game, 120) or None
        date_from = _parse_date(date_from, "Start date")
        date_to = _parse_date(date_to, "End date")
        if date_from and date_to and date_from > date_to:
            raise ValueError("Start date must be on or before end date.")
        max_clips = max(2, min(MAX_PROJECT_CLIPS, int(max_clips)))
        max_per_session = max(1, min(MAX_PER_SESSION, int(max_per_session)))
        if target_duration_seconds is not None:
            target_duration_seconds = max(30.0, min(7200.0, float(target_duration_seconds)))

        candidates = self._candidates(
            query=query,
            game=game,
            date_from=date_from,
            date_to=date_to,
        )
        selected, summary = self._select(
            candidates,
            max_clips=max_clips,
            max_per_session=max_per_session,
            target_duration_seconds=target_duration_seconds,
        )
        if len(selected) < 2:
            raise CompilationProjectConflict(
                "Recall found fewer than two creator-kept clips for that compilation."
            )

        project_id = f"project_{uuid.uuid4().hex}"
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO compilation_projects(
                       id, title, template, query, game, date_from, date_to,
                       target_duration_seconds, max_clips, max_per_session,
                       status, generation_version, selection_summary,
                       created_at, updated_at
                   ) VALUES (?, ?, 'best_of_range', ?, ?, ?, ?, ?, ?, ?,
                             'draft', ?, ?, ?, ?)""",
                (
                    project_id, title, query, game, date_from, date_to,
                    target_duration_seconds, max_clips, max_per_session,
                    GENERATION_VERSION, json.dumps(summary, separators=(",", ":")),
                    now, now,
                ),
            )
            for position, row in enumerate(selected):
                retrieval_copy = (
                    f"Matched {row['match_reason'].lower()} for “{query}”. " if query else ""
                )
                reason = (
                    f"{retrieval_copy}Creator kept this moment; selected for strength "
                    "and cross-session variety."
                )
                conn.execute(
                    """INSERT INTO compilation_project_items(
                           id, project_id, clip_id, source_job_id,
                           source_group_snapshot, position,
                           included, selection_score, selection_reason,
                           title_snapshot, session_name_snapshot,
                           source_date_snapshot, start_time_snapshot,
                           end_time_snapshot, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"item_{uuid.uuid4().hex}", project_id, row["id"],
                        row["job_id"], row.get("diversity_key") or row["job_id"],
                        position, row["project_score"], reason,
                        row.get("title") or "Untitled clip",
                        row.get("session_name") or "Untitled session",
                        row.get("effective_date"), row.get("start_time") or 0.0,
                        row.get("end_time") or 0.0, now,
                    ),
                )
        return self.get(project_id)

    # -- where the kills actually are ----------------------------------------

    def _timeline(self, job_id: str) -> tuple[list[float], list[float], list[dict[str, Any]]]:
        """The job's reaction curve and its per-event game markers.

        A marker carries the exact second Recall read an ELIMINATION banner,
        which is a far better anchor than an evidence window: those run twelve
        to thirty seconds and say only that something happened inside them.
        """
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT timeline FROM reaction_timelines WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return [], [], []
        try:
            payload = json.loads(row["timeline"] or "{}")
        except (TypeError, ValueError):
            return [], [], []
        times = [float(value) for value in (payload.get("t") or [])]
        curve = [float(value) for value in (payload.get("r") or [])]
        markers = [
            marker for marker in (payload.get("game_markers") or [])
            if isinstance(marker, dict) and marker.get("t") is not None
        ]
        return times, curve, markers

    @staticmethod
    def _reaction_near(
        times: list[float], curve: list[float], at: float,
    ) -> float:
        """How hard the room reacted around one moment."""
        best = 0.0
        for index, when in enumerate(times):
            if at - 2.0 <= when <= at + 5.0 and index < len(curve):
                best = max(best, curve[index])
        return best

    @staticmethod
    def _peak_within(
        times: list[float], curve: list[float], start: float, end: float,
    ) -> tuple[float, float]:
        """When the reaction peaked inside a window, and how high.

        Measured on real windows the peak sits anywhere from half a second to
        fourteen seconds in, so the midpoint is not where the moment is.
        """
        best_at, best = (start + end) / 2.0, 0.0
        for index, when in enumerate(times):
            if start <= when <= end and index < len(curve) and curve[index] > best:
                best_at, best = when, curve[index]
        return best_at, round(best, 4)

    def _marker_kills(
        self, job_id: str, start: float, end: float,
    ) -> list[dict[str, Any]]:
        """One entry per distinct kill in a window, strongest reaction first."""
        times, curve, markers = self._timeline(job_id)
        stamps = sorted(
            float(marker["t"]) for marker in markers
            if str(marker.get("label") or "").lower() == "elimination"
            and start <= float(marker["t"]) <= end
        )
        merged: list[float] = []
        for stamp in stamps:
            # Keep the first reading of a banner, not each re-read of it.
            if merged and stamp - merged[-1] < MARKER_MERGE_SECONDS:
                continue
            merged.append(stamp)
        kills = []
        for stamp in merged:
            # The reaction can peak just before the banner is read; when it
            # does, it is the earlier and therefore safer anchor.
            peak_at, peak = self._peak_within(times, curve, stamp - 10.0, stamp + 3.0)
            kills.append({
                "at": min(stamp, peak_at) if peak > 0 else stamp,
                "reaction": self._reaction_near(times, curve, stamp),
            })
        return kills

    @staticmethod
    def _kill_span(at: float, seconds: float) -> tuple[float, float]:
        """A cut that leads its anchor, so it holds the shot and not the aftermath."""
        length = max(MIN_KILL_SECONDS, min(MAX_KILL_SECONDS, seconds))
        lead = max(length - KILL_TAIL_SECONDS, length * KILL_LEAD_SHARE)
        start = max(0.0, at - lead)
        return round(start, 3), round(start + length, 3)

    # -- one match inside one session ---------------------------------------

    def match_options(self, job_id: str) -> dict[str, Any]:
        """Every match Recall can see in one scanned session.

        Read-only: this walks detected events that the scan already wrote and
        never touches source video, a model, or the scan pipeline.
        """
        job_id = _clean(job_id, 100)
        with self.db.get_connection() as conn:
            job = conn.execute(
                """SELECT id, session_name, source_date, created_at, duration
                   FROM jobs WHERE id = ?""",
                (job_id,),
            ).fetchone()
        if job is None:
            raise CompilationProjectNotFound("That session no longer exists.")
        job = dict(job)

        segments = segment_matches(self.db, job_id)
        clips = self._match_clip_index(job_id)
        options = []
        for segment in segments:
            if segment.confidence == "low":
                # Offered nowhere: a lone menu detection is not a game, and
                # listing it as one would teach the creator to distrust this.
                continue
            moments = self._match_moments(segment, clips)
            options.append({
                "job_id": job_id,
                "match_index": segment.index,
                "label": segment.label(),
                "outcome": segment.outcome,
                "placement": segment.placement,
                "confidence": segment.confidence,
                "start_time": round(segment.start, 3),
                "end_time": round(segment.end, 3),
                "duration_seconds": round(segment.duration, 3),
                "kill_estimate": segment.kill_estimate,
                "detected_kills": segment.detected_kills,
                "milestone_kills": segment.milestone_kills,
                "has_win": segment.has_win,
                "moment_count": len(moments),
                "already_cut_count": sum(1 for m in moments if m["clip_id"]),
                "boundary_reasons": list(segment.boundary_reasons),
            })
        return {
            "job_id": job_id,
            "session_name": job.get("session_name") or "Untitled session",
            "source_date": job.get("source_date") or str(job.get("created_at") or "")[:10],
            "match_count": len(options),
            "matches": options,
        }

    def _spoken_titles(
        self, job_id: str, moments: list[MatchMoment]
    ) -> dict[float, str]:
        """What the creator said over each uncut moment.

        A storyboard of ten rows all reading "Elimination" is unreviewable.
        The transcript is already indexed, so the line spoken across a kill is
        a free and far more recognisable name for it.
        """
        if not moments:
            return {}
        window_start = min(m.start for m in moments)
        window_end = max(m.end for m in moments)
        with self.db.get_connection() as conn:
            rows = [
                dict(row) for row in conn.execute(
                    """SELECT start_time, end_time, text
                       FROM stream_memory_entries
                       WHERE job_id = ? AND kind = 'transcript'
                         AND end_time >= ? AND start_time <= ?
                       ORDER BY start_time""",
                    (job_id, window_start, window_end),
                ).fetchall()
            ]

        titles: dict[float, str] = {}
        for moment in moments:
            spoken = [
                _clean(row.get("text"), 200)
                for row in rows
                if row["end_time"] >= moment.start and row["start_time"] <= moment.end
            ]
            # The longest line in the window carries the most meaning; the
            # short ones are usually filler ("I", "yeah", "oh").
            best = max((line for line in spoken if len(line) > 3), key=len, default="")
            if best:
                titles[moment.start] = _shorten(best, 64)
        return titles

    def _spoken_at(self, job_id: str, start: float, end: float) -> str:
        """The line spoken across an arbitrary window, for naming a cut."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT text FROM stream_memory_entries
                   WHERE job_id = ? AND kind = 'transcript'
                     AND end_time >= ? AND start_time <= ?""",
                (job_id, start, end),
            ).fetchall()
        spoken = [_clean(row["text"], 200) for row in rows]
        best = max((line for line in spoken if len(line) > 3), key=len, default="")
        return _shorten(best, 64) if best else ""

    def _match_clip_index(self, job_id: str) -> list[dict[str, Any]]:
        """Clips already cut from this session, for reuse by detected moments."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT c.id, c.job_id, c.title, c.start_time, c.end_time,
                          c.selection_score, c.deck_score, c.hook_score, c.score,
                          c.reaction_auc, c.kept, c.passed,
                          j.session_name, j.source_date, j.created_at,
                          COALESCE(
                            (SELECT source_key FROM source_asset_sessions
                             WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1),
                            NULLIF(j.source_path, ''), j.id
                          ) AS diversity_key
                   FROM clips AS c
                   JOIN jobs AS j ON j.id = c.job_id
                   WHERE c.job_id = ?
                   ORDER BY c.start_time""",
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _overlapping_clip(
        moment: MatchMoment, clips: list[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """The clip already covering this moment, if the scan cut one.

        Reusing it keeps the creator's own title, thumbnail and review decision
        instead of proposing a second cut of a moment they already have.
        """
        best, best_overlap = None, 0.0
        for clip in clips:
            start = float(clip.get("start_time") or 0.0)
            end = float(clip.get("end_time") or 0.0)
            overlap = min(moment.end, end) - max(moment.start, start)
            if overlap <= 0:
                continue
            share = overlap / max(0.1, moment.end - moment.start)
            if share > best_overlap:
                best, best_overlap = clip, share
        return best if best_overlap >= 0.5 else None

    def _match_moments(
        self, segment: MatchSegment, clips: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Every compilable moment in one match, strongest first.

        One clip can span several kills. The strongest of them claims it and
        the rest are dropped: they are the same footage, and including them
        twice would repeat it in the reel.
        """
        moments: list[dict[str, Any]] = []
        win = segment.win_moment
        for moment in (*segment.eliminations, *([win] if win else [])):
            clip = self._overlapping_clip(moment, clips)
            strength = moment_strength(moment)
            if clip and bool(clip.get("kept")):
                # The creator already blessed this cut of the moment.
                strength += 1.0
            elif clip and bool(clip.get("passed")):
                strength -= 0.5
            moments.append({
                "moment": moment,
                "kind": "win" if moment.kind == "win" else "elimination",
                "clip_id": clip["id"] if clip else None,
                "clip": clip,
                "strength": round(strength, 4),
            })
        moments.sort(key=lambda entry: (-entry["strength"], entry["moment"].start))

        claimed: set[str] = set()
        unique: list[dict[str, Any]] = []
        for entry in moments:
            clip_id = entry["clip_id"]
            if clip_id and clip_id in claimed:
                continue
            if clip_id:
                claimed.add(clip_id)
            unique.append(entry)
        return unique

    @staticmethod
    def _tight_span(moment: MatchMoment, seconds: float) -> tuple[float, float]:
        """A montage-length cut centred on the kill.

        The detected window runs a dozen seconds and carries whatever the
        creator happened to be saying beforehand. A montage wants the kill, so
        the cut centres on the event and keeps only what the target allows.
        """
        centre = moment.start + (moment.duration / 2.0)
        half = max(MIN_KILL_SECONDS, seconds) / 2.0
        start = max(0.0, centre - half)
        return round(start, 3), round(start + (half * 2.0), 3)

    def _win_span(
        self, entry: dict[str, Any], target: float, *, job_id: str = "",
    ) -> tuple[float, float]:
        """The closing payoff: the victory screen and the reaction to it.

        Anchored on the terminal-win marker when the scan left one, because
        that is the second the victory was actually read. Trimming the tail of
        the creator's clip instead used to work only while the win owned half
        the reel; once it was cut back to a quarter, the tail no longer reached
        the victory at all and the reel ended on the aftermath.
        """
        moment = entry["moment"]
        clip = entry.get("clip")
        cap = max(MIN_WIN_SECONDS, target * WIN_SHARE_OF_REEL)

        if job_id:
            _, _, markers = self._timeline(job_id)
            terminal = sorted(
                float(marker["t"]) for marker in markers
                if str(marker.get("label") or "").lower() == "terminal_win"
                and moment.start - 30.0 <= float(marker["t"]) <= moment.end + 60.0
            )
            if terminal:
                # The victory screen is read as it appears, so the win needs
                # only a short run-up -- unlike a kill, whose banner lags it.
                start = max(0.0, terminal[0] - KILL_TAIL_SECONDS)
                return round(start, 3), round(start + cap, 3)

        start = float(moment.start)
        end = float(clip["end_time"]) if clip else float(moment.end)
        if end <= start:
            end = start + MIN_WIN_SECONDS
        if end - start > cap:
            start = end - cap
        return round(start, 3), round(end, 3)

    def generate_from_match(
        self,
        *,
        job_id: str,
        match_index: int,
        title: Optional[str] = None,
        max_clips: int = 12,
        target_duration_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        """Build a compilation from the strongest moments of a single match.

        This is the one generation path that deliberately does not spread
        across sessions. Every moment comes from the same game, and a victory
        is always kept and always placed last, because that is the shape a
        kill compilation is edited into.
        """
        job_id = _clean(job_id, 100)
        match_index = int(match_index)
        max_clips = max(2, min(MAX_PROJECT_CLIPS, int(max_clips)))
        if target_duration_seconds is not None:
            target_duration_seconds = max(30.0, min(7200.0, float(target_duration_seconds)))

        with self.db.get_connection() as conn:
            job = conn.execute(
                "SELECT id, session_name, source_date, created_at FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if job is None:
            raise CompilationProjectNotFound("That session no longer exists.")
        job = dict(job)

        segment = next(
            (s for s in segment_matches(self.db, job_id) if s.index == match_index), None
        )
        if segment is None:
            raise CompilationProjectNotFound("Recall no longer detects that match.")

        clips = self._match_clip_index(job_id)
        ranked = self._match_moments(segment, clips)
        win = next((entry for entry in ranked if entry["kind"] == "win"), None)

        target = float(target_duration_seconds or DEFAULT_REEL_SECONDS)
        win_span = self._win_span(win, target, job_id=job_id) if win else None
        win_seconds = (win_span[1] - win_span[0]) if win_span else 0.0
        kill_budget = max(0.0, target - win_seconds)

        # Kills come from the game markers when the scan left them: each is the
        # exact second a banner was read, where an evidence window is a
        # twelve-to-thirty second region that only says something happened
        # inside it. Centring a short cut in such a window landed on whatever
        # the creator was chatting about, not the kill.
        markers = self._marker_kills(job_id, segment.start, segment.end)
        if markers:
            candidates = [
                {
                    "at": marker["at"],
                    # The reaction curve at the marker is the honest measure of
                    # how good a kill was; every kill carries the same tags.
                    "strength": round(marker["reaction"], 4),
                    "moment": None,
                    "clip_id": None,
                    "clip": None,
                    "kind": "elimination",
                }
                for marker in markers
            ]
        else:
            candidates = [
                {
                    "at": entry["moment"].start + (entry["moment"].duration / 2.0),
                    "strength": entry["strength"],
                    "moment": entry["moment"],
                    "clip_id": entry["clip_id"],
                    "clip": entry["clip"],
                    "kind": "elimination",
                }
                for entry in ranked if entry["kind"] != "win"
            ]

        # How many kills the budget allows, then how long each may run so the
        # reel fills the requested length rather than stopping short of it.
        room = max(1, min(max_clips - (1 if win else 0), len(candidates)))
        count = max(1, min(room, int(kill_budget // MIN_KILL_SECONDS)))
        per_kill = min(MAX_KILL_SECONDS, max(MIN_KILL_SECONDS, kill_budget / count))

        ordered = sorted(candidates, key=lambda item: (-item["strength"], item["at"]))
        keep: list[dict[str, Any]] = []
        spent = 0.0
        for entry in ordered:
            span = self._kill_span(entry["at"], per_kill)
            length = span[1] - span[0]
            if keep and spent + length > kill_budget:
                continue
            # Two cuts that overlap show the same seconds twice.
            if any(span[0] < other["span"][1] and other["span"][0] < span[1] for other in keep):
                continue
            entry["span"] = span
            entry["captions"] = False
            keep.append(entry)
            spent += length
            if len(keep) >= room:
                break

        # Overlap and scarcity can leave the reel short; give what is left back
        # to the kills that made it in, up to their maximum length.
        leftover = kill_budget - spent
        if keep and leftover > 0.5:
            extra = min(MAX_KILL_SECONDS - per_kill, leftover / len(keep))
            if extra > 0.05:
                for entry in keep:
                    entry["span"] = self._kill_span(entry["at"], per_kill + extra)

        # Strong open, chronological middle, strongest close. Chronological
        # order alone opened on whatever happened first, which on a real match
        # was a quiet looting fight -- the weakest three seconds in the reel
        # sitting where a viewer decides whether to keep watching.
        keep.sort(key=lambda entry: entry["span"][0])
        if len(keep) > 2:
            opener = max(keep, key=lambda entry: entry["strength"])
            keep = [opener, *(entry for entry in keep if entry is not opener)]
        if win:
            win["span"] = win_span
            win["captions"] = True

        selected = [*keep, *([win] if win else [])]
        if len(selected) < 2:
            raise CompilationProjectConflict(
                "Recall detected fewer than two moments in that match."
            )

        session_name = job.get("session_name") or "Untitled session"
        effective_date = job.get("source_date") or str(job.get("created_at") or "")[:10]
        project_title = _clean(title, 160) or f"{session_name} - {segment.label()}"
        summary = {
            "candidates_considered": len(ranked),
            "duplicates_suppressed": 0,
            "sessions_represented": 1,
            "selected_duration_seconds": round(
                sum(entry["span"][1] - entry["span"][0] for entry in selected), 3
            ),
            "target_duration_seconds": round(target, 3),
            "selection_policy": "single_match_detected_moments",
            "match_kill_estimate": segment.kill_estimate,
            "match_outcome": segment.outcome,
            "match_confidence": segment.confidence,
            "uncut_moment_count": len(selected),
        }
        spoken = self._spoken_titles(
            job_id,
            [entry["moment"] for entry in selected if entry.get("moment") is not None],
        )

        project_id = f"project_{uuid.uuid4().hex}"
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO compilation_projects(
                       id, title, template, query, game, date_from, date_to,
                       target_duration_seconds, max_clips, max_per_session,
                       status, generation_version, selection_summary,
                       source_job_id, match_index, match_label,
                       match_start, match_end, created_at, updated_at
                   ) VALUES (?, ?, 'single_match', '', NULL, ?, ?, ?, ?, ?,
                             'draft', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id, project_title, effective_date, effective_date,
                    target_duration_seconds, max_clips, max_clips,
                    GENERATION_VERSION, json.dumps(summary, separators=(",", ":")),
                    job_id, segment.index, segment.label(),
                    round(segment.start, 3), round(segment.end, 3), now, now,
                ),
            )
            for position, entry in enumerate(selected):
                moment = entry.get("moment")
                clip = entry.get("clip")
                if entry["kind"] == "win":
                    reason = "The win this match ended on, with the reaction. Closes the reel."
                elif moment is not None and moment.multi_kill > 1:
                    reason = f"A {moment.multi_kill}-kill burst in this match."
                else:
                    reason = "Elimination, ranked on how hard the room reacted."
                moment_title = (
                    (clip.get("title") if clip else None)
                    or (spoken.get(moment.start) if moment is not None else None)
                    or self._spoken_at(job_id, *entry["span"])
                    or ("Victory" if entry["kind"] == "win" else "Elimination")
                )
                # The montage span, not the detected window and not the
                # creator's original cut: a kill is tightened to montage length
                # and the win is trimmed only if it would swallow the reel.
                item_start, item_end = entry["span"]
                conn.execute(
                    """INSERT INTO compilation_project_items(
                           id, project_id, clip_id, source_job_id,
                           source_group_snapshot, position,
                           included, selection_score, selection_reason,
                           title_snapshot, session_name_snapshot,
                           source_date_snapshot, start_time_snapshot,
                           end_time_snapshot, source_kind, moment_kind,
                           captions_enabled, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"item_{uuid.uuid4().hex}", project_id, None,
                        job_id, job_id, position, entry["strength"], reason,
                        _clean(moment_title, 200), session_name, effective_date,
                        round(item_start, 3), round(item_end, 3),
                        "event", entry["kind"], int(bool(entry["captions"])), now,
                    ),
                )
        return self.get(project_id)

    @staticmethod
    def _media_state(row: dict[str, Any]) -> tuple[bool, str]:
        if str(row.get("source_kind") or "clip") == "event" and not row.get("clip_id"):
            # A detected moment the creator has not cut yet. Its project memory
            # is complete; only the media does not exist.
            return False, "not_cut_yet"
        if not row.get("clip_id") or not row.get("current_clip_id"):
            return False, "clip_removed"
        export_path = str(row.get("export_path") or "")
        if export_path and os.path.isfile(export_path) and not row.get("preview_only"):
            return True, "rendered"
        source_candidates = [row.get("asset_path"), row.get("source_path")]
        if any(path and os.path.isfile(str(path)) for path in source_candidates):
            return True, "rebuildable"
        return False, "source_unavailable"

    def _items(self, project_id: str) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT i.*, c.id AS current_clip_id, c.export_path,
                          c.preview_only, c.thumb_path, j.source_path, j.asset_path
                   FROM compilation_project_items AS i
                   LEFT JOIN clips AS c ON c.id = i.clip_id
                   LEFT JOIN jobs AS j ON j.id = c.job_id
                   WHERE i.project_id = ?
                   ORDER BY i.position, i.created_at, i.id""",
                (project_id,),
            ).fetchall()
        items = []
        for source in rows:
            row = dict(source)
            available, media_state = self._media_state(row)
            items.append({
                "id": row["id"],
                "clip_id": row.get("clip_id"),
                "job_id": row.get("source_job_id"),
                "source_group": row.get("source_group_snapshot") or row.get("source_job_id"),
                "position": int(row.get("position") or 0),
                "included": bool(row.get("included")),
                "selection_score": row.get("selection_score"),
                "selection_reason": row.get("selection_reason") or "",
                "title": row.get("title_snapshot") or "Untitled clip",
                "session_name": row.get("session_name_snapshot") or "Untitled session",
                "source_date": row.get("source_date_snapshot"),
                "start_time": float(row.get("start_time_snapshot") or 0.0),
                "end_time": float(row.get("end_time_snapshot") or 0.0),
                "duration": max(
                    0.0,
                    float(row.get("end_time_snapshot") or 0.0)
                    - float(row.get("start_time_snapshot") or 0.0),
                ),
                "media_available": available,
                "media_state": media_state,
                "source_kind": str(row.get("source_kind") or "clip"),
                "moment_kind": row.get("moment_kind"),
                "muted": bool(row.get("muted")),
                "captions_enabled": (
                    None if row.get("captions_enabled") is None
                    else bool(row.get("captions_enabled"))
                ),
            })
        return items

    def _serialize(self, row: dict[str, Any], *, include_items: bool) -> dict[str, Any]:
        items = self._items(row["id"])
        included = [item for item in items if item["included"]]
        payload = {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "id": row["id"],
            "title": row["title"],
            "template": row["template"],
            "query": row.get("query") or "",
            "game": row.get("game"),
            "date_from": row.get("date_from"),
            "date_to": row.get("date_to"),
            "target_duration_seconds": row.get("target_duration_seconds"),
            "max_clips": int(row.get("max_clips") or 0),
            "max_per_session": int(row.get("max_per_session") or 0),
            "status": row.get("status") or "draft",
            "generation_version": int(row.get("generation_version") or 1),
            "selection_summary": _json_object(row.get("selection_summary")),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "approved_at": row.get("approved_at"),
            "source_job_id": row.get("source_job_id"),
            "match_index": row.get("match_index"),
            "match_label": row.get("match_label"),
            "match_start": row.get("match_start"),
            "match_end": row.get("match_end"),
            "reel_path": row.get("reel_path"),
            "reel_url": (
                f"/media/{os.path.basename(str(row.get('reel_path')))}"
                if row.get("reel_path") else None
            ),
            "reel_duration_seconds": row.get("reel_duration_seconds"),
            "reel_built_at": row.get("reel_built_at"),
            "reel_stale": bool(row.get("reel_stale")),
            "muted_count": sum(1 for item in included if item["muted"]),
            "uncut_count": sum(
                1 for item in included if item["media_state"] == "not_cut_yet"
            ),
            "item_count": len(items),
            "included_count": len(included),
            "available_count": sum(item["media_available"] for item in included),
            "session_count": len({
                item["source_group"] for item in included if item.get("source_group")
            }),
            "total_duration_seconds": round(sum(item["duration"] for item in included), 3),
        }
        if include_items:
            payload["items"] = items
        return payload

    def get(self, project_id: str) -> dict[str, Any]:
        return self._serialize(self._project_row(project_id), include_items=True)

    def list(self) -> list[dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = [
                dict(row) for row in conn.execute(
                    "SELECT * FROM compilation_projects ORDER BY updated_at DESC, id DESC"
                ).fetchall()
            ]
        return [self._serialize(row, include_items=False) for row in rows]

    def update(self, project_id: str, *, title: Optional[str] = None) -> dict[str, Any]:
        self._project_row(project_id)
        cleaned = _clean(title, 160) if title is not None else None
        if title is not None and not cleaned:
            raise ValueError("Compilation title cannot be empty.")
        if cleaned is not None:
            with self.db.get_connection() as conn:
                conn.execute(
                    """UPDATE compilation_projects
                       SET title = ?, status = 'draft', approved_at = NULL,
                           updated_at = ? WHERE id = ?""",
                    (cleaned, _utc_now(), project_id),
                )
        return self.get(project_id)

    def replace_items(self, project_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        self._project_row(project_id)
        with self.db.get_connection() as conn:
            existing = {
                row["id"] for row in conn.execute(
                    "SELECT id FROM compilation_project_items WHERE project_id = ?",
                    (project_id,),
                ).fetchall()
            }
            supplied = [str(item.get("id") or "") for item in items]
            if len(supplied) != len(set(supplied)) or set(supplied) != existing:
                raise ValueError("Send every project item exactly once when reordering.")
            now = _utc_now()
            for position, item in enumerate(items):
                conn.execute(
                    """UPDATE compilation_project_items
                       SET position = ?, included = ?, muted = ?
                       WHERE id = ? AND project_id = ?""",
                    (
                        position, int(bool(item.get("included", True))),
                        int(bool(item.get("muted", False))), item["id"], project_id,
                    ),
                )
            conn.execute(
                """UPDATE compilation_projects
                   SET status = 'draft', approved_at = NULL, updated_at = ?
                   WHERE id = ?""",
                (now, project_id),
            )
        self.mark_reel_stale(project_id)
        return self.get(project_id)

    def add_clip(
        self,
        project_id: str,
        clip_id: str,
        *,
        selection_reason: str = "Creator added this kept clip.",
    ) -> dict[str, Any]:
        """Append one creator-kept clip to a draft, idempotently."""
        project = self._project_row(project_id)
        cleaned_clip_id = _clean(clip_id, 100)
        if not cleaned_clip_id:
            raise ValueError("Clip id cannot be empty.")
        with self.db.get_connection() as conn:
            existing = conn.execute(
                """SELECT id FROM compilation_project_items
                   WHERE project_id = ? AND clip_id = ?""",
                (project_id, cleaned_clip_id),
            ).fetchone()
            if existing is not None:
                return self.get(project_id)
            item_count = int(conn.execute(
                "SELECT COUNT(*) FROM compilation_project_items WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0])
            if item_count >= int(project.get("max_clips") or MAX_PROJECT_CLIPS):
                raise CompilationProjectConflict(
                    "This compilation already has its maximum number of clips."
                )
            row = conn.execute(
                """SELECT c.id, c.job_id, c.title, c.start_time, c.end_time,
                          c.selection_score, c.deck_score, c.hook_score, c.score,
                          c.reaction_auc, c.kept, c.passed,
                          j.session_name, j.source_date, j.created_at, j.source_path,
                          COALESCE(
                            (SELECT source_key FROM source_asset_sessions
                             WHERE job_id = j.id ORDER BY created_at DESC LIMIT 1),
                            NULLIF(j.source_path, ''), j.id
                          ) AS diversity_key
                   FROM clips AS c
                   JOIN jobs AS j ON j.id = c.job_id
                   WHERE c.id = ?""",
                (cleaned_clip_id,),
            ).fetchone()
            if row is None:
                raise CompilationProjectConflict("That clip is no longer available.")
            candidate = dict(row)
            if not bool(candidate.get("kept")) or bool(candidate.get("passed")):
                raise CompilationProjectConflict(
                    "Keep this clip before adding it to a compilation."
                )
            position = int(conn.execute(
                """SELECT COALESCE(MAX(position), -1) + 1
                   FROM compilation_project_items WHERE project_id = ?""",
                (project_id,),
            ).fetchone()[0])
            now = _utc_now()
            effective_date = candidate.get("source_date") or str(
                candidate.get("created_at") or ""
            )[:10]
            conn.execute(
                """INSERT INTO compilation_project_items(
                       id, project_id, clip_id, source_job_id,
                       source_group_snapshot, position, included,
                       selection_score, selection_reason, title_snapshot,
                       session_name_snapshot, source_date_snapshot,
                       start_time_snapshot, end_time_snapshot, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"item_{uuid.uuid4().hex}", project_id, cleaned_clip_id,
                    candidate["job_id"], candidate.get("diversity_key") or candidate["job_id"],
                    position, self._candidate_score(candidate, None),
                    _clean(selection_reason, 500) or "Creator added this kept clip.",
                    candidate.get("title") or "Untitled clip",
                    candidate.get("session_name") or "Untitled session",
                    effective_date, candidate.get("start_time") or 0.0,
                    candidate.get("end_time") or 0.0, now,
                ),
            )
            conn.execute(
                """UPDATE compilation_projects
                   SET status = 'draft', approved_at = NULL, updated_at = ?
                   WHERE id = ?""",
                (now, project_id),
            )
        return self.get(project_id)

    # -- cutting detected moments into clips ---------------------------------

    def next_uncut_item(self, project_id: str) -> Optional[dict[str, Any]]:
        """The next included moment that still has no clip cut for it.

        Cutting is deliberately one moment per call. Each cut is a real render,
        so a single long-running batch would be neither cancellable nor honest
        about progress; the creator drives the loop and can stop after any one.
        """
        project = self.get(project_id)
        for item in project["items"]:
            if item["included"] and item["media_state"] == "not_cut_yet":
                return item
        return None

    def attach_clip(self, project_id: str, item_id: str, clip_id: str) -> dict[str, Any]:
        """Point a detected moment at the clip that was just cut for it."""
        self._project_row(project_id)
        cleaned = _clean(clip_id, 100)
        if not cleaned:
            raise ValueError("Clip id cannot be empty.")
        now = _utc_now()
        with self.db.get_connection() as conn:
            updated = conn.execute(
                """UPDATE compilation_project_items
                   SET clip_id = ?, source_kind = 'clip'
                   WHERE id = ? AND project_id = ? AND clip_id IS NULL""",
                (cleaned, item_id, project_id),
            ).rowcount
            if not updated:
                raise CompilationProjectConflict(
                    "That moment already points at a clip."
                )
            conn.execute(
                """UPDATE compilation_projects
                   SET status = 'draft', approved_at = NULL, updated_at = ?
                   WHERE id = ?""",
                (now, project_id),
            )
        return self.get(project_id)

    # -- one theme across many sessions --------------------------------------

    def _available_jobs(self) -> set[str]:
        """Sessions whose recording is still on disk, so a moment can be cut."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, source_path, asset_path FROM jobs WHERE status = 'completed'"
            ).fetchall()
        ready = set()
        for row in rows:
            for path in (row["source_path"], row["asset_path"]):
                if path and os.path.isfile(str(path)):
                    ready.add(str(row["id"]))
                    break
        return ready

    def _evidence_moments(
        self,
        *,
        query: str,
        game: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Detected events across the library, by theme or by window.

        This is the same material the single-match path uses; the only
        difference is that nothing here is bounded to one game.
        """
        rows: list[dict[str, Any]] = []
        ranks: dict[str, int] = {}
        if query:
            found = self.memory.search(
                query, kind="evidence", game=game,
                date_from=date_from, date_to=date_to,
                limit=limit, mode="hybrid",
            )
            ids = [str(item["id"]) for item in found["results"] if item.get("id")]
            ranks = {entry: index for index, entry in enumerate(ids, start=1)}
            if not ids:
                return []
            with self.db.get_connection() as conn:
                placeholders = ",".join("?" for _ in ids)
                rows = [dict(row) for row in conn.execute(
                    f"""SELECT e.*, j.session_name, j.source_date, j.created_at
                        FROM stream_memory_entries AS e
                        JOIN jobs AS j ON j.id = e.job_id
                        WHERE e.id IN ({placeholders})""",  # nosec B608 - placeholders only
                    ids,
                ).fetchall()]
        else:
            clauses = ["e.kind = 'evidence'"]
            params: list[Any] = []
            if date_from:
                clauses.append("COALESCE(j.source_date, substr(j.created_at, 1, 10)) >= ?")
                params.append(date_from)
            if date_to:
                clauses.append("COALESCE(j.source_date, substr(j.created_at, 1, 10)) <= ?")
                params.append(date_to)
            with self.db.get_connection() as conn:
                rows = [dict(row) for row in conn.execute(
                    f"""SELECT e.*, j.session_name, j.source_date, j.created_at
                        FROM stream_memory_entries AS e
                        JOIN jobs AS j ON j.id = e.job_id
                        WHERE {' AND '.join(clauses)}
                        LIMIT ?""",  # nosec B608 - clauses fixed, values parameterized
                    (*params, limit),
                ).fetchall()]

        moments = []
        curves: dict[str, tuple[list[float], list[float]]] = {}
        for row in rows:
            moment = moment_from_evidence(row)
            if moment.kind in {"lobby", "terminal"}:
                # Menus and placement cards are boundaries, never content.
                continue
            job = str(row["job_id"])
            if job not in curves:
                times, curve, _ = self._timeline(job)
                curves[job] = (times, curve)
            times, curve = curves[job]
            peak_at, peak = self._peak_within(times, curve, moment.start, moment.end)
            retrieval = ranks.get(str(row["id"]))
            strength = moment_strength(moment) + peak
            moments.append({
                "kind": "event",
                "moment": moment,
                "job_id": str(row["job_id"]),
                "session_name": row.get("session_name") or "Untitled session",
                "source_date": row.get("source_date") or str(row.get("created_at") or "")[:10],
                "clip_id": None,
                "start": moment.start,
                "end": moment.end,
                "peak_at": peak_at,
                "strength": round(strength, 4),
                "rank": retrieval,
                "is_win": moment.kind == "win",
                # The detector already names what it saw ("Scare", "Elimination"),
                # which beats a generic fallback when nothing was said over it.
                "detected_title": _clean(row.get("title"), 60),
                "title": "",
            })
        return moments

    def _kept_clip_moments(
        self,
        *,
        query: str,
        game: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> list[dict[str, Any]]:
        """Clips the creator kept, which carry their own rendered media.

        These need no source recording, so they stay usable long after the VOD
        they came from has been cleared off the disk.
        """
        candidates = self._candidates(
            query=query, game=game, date_from=date_from, date_to=date_to,
        )
        moments = []
        for row in candidates:
            export = str(row.get("export_path") or "")
            if not export or not os.path.isfile(export) or row.get("preview_only"):
                continue
            moments.append({
                "kind": "clip",
                "moment": None,
                "job_id": str(row["job_id"]),
                "session_name": row.get("session_name") or "Untitled session",
                "source_date": row.get("effective_date"),
                "clip_id": str(row["id"]),
                "start": float(row.get("start_time") or 0.0),
                "end": float(row.get("end_time") or 0.0),
                # Already reviewed and already rendered: the creator's own
                # verdict outranks a detector score.
                "strength": round(float(row.get("project_score") or 0.0) + 1.5, 4),
                "rank": row.get("retrieval_rank"),
                "is_win": False,
                "title": row.get("title") or "",
            })
        return moments

    def generate_from_theme(
        self,
        *,
        title: str,
        query: str = "",
        game: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        target_duration_seconds: Optional[float] = None,
        max_per_session: int = 3,
        ready_only: bool = True,
    ) -> dict[str, Any]:
        """One montage from one theme, drawn across every scanned session.

        The cross-session sibling of ``generate_from_match``: same material
        (detected events), same duration-driven packing, same output (a video).
        What changes is the unit -- a theme over the library rather than one
        game -- and therefore the ending, which has no win to land on unless
        the theme happens to surface one.
        """
        title = _clean(title, 160)
        if not title:
            raise ValueError("Give this compilation a title.")
        query = _clean(query, 240)
        game = _clean(game, 120) or None
        date_from = _parse_date(date_from, "Start date")
        date_to = _parse_date(date_to, "End date")
        if date_from and date_to and date_from > date_to:
            raise ValueError("Start date must be on or before end date.")
        target = float(target_duration_seconds or DEFAULT_REEL_SECONDS)
        target = max(30.0, min(7200.0, target))
        max_per_session = max(1, min(MAX_PER_SESSION, int(max_per_session)))

        ready_jobs = self._available_jobs()
        events = self._evidence_moments(
            query=query, game=game, date_from=date_from, date_to=date_to, limit=300,
        )
        clips = self._kept_clip_moments(
            query=query, game=game, date_from=date_from, date_to=date_to,
        )
        considered = len(events) + len(clips)
        # A detected event has no media of its own -- cutting one needs the
        # recording. A kept clip carries its render, so it stays usable.
        offline = [entry for entry in events if entry["job_id"] not in ready_jobs]
        longest_clip = max(MIN_CLIP_BUDGET_SECONDS, target * MAX_CLIP_SHARE_OF_REEL)
        oversized = [
            entry for entry in clips if (entry["end"] - entry["start"]) > longest_clip
        ]
        usable_clips = [entry for entry in clips if entry not in oversized]
        pool = usable_clips + [entry for entry in events if entry["job_id"] in ready_jobs]
        if not ready_only:
            pool = usable_clips + events

        # With a theme, relevance leads and strength breaks ties. Ranking on
        # strength alone let the same few strong clips win every query, so
        # "jump scares" and "funniest" returned nearly the same reel.
        pool.sort(key=lambda entry: (
            (entry.get("rank") or 10_000) if query else 0,
            -entry["strength"],
            str(entry["job_id"]),
            entry["start"],
        ))

        # The finale is whatever the material makes strongest. A match montage
        # ends on its win because the match actually ended there; a theme has
        # no such moment, and forcing one produced a Victory Royale closing a
        # compilation of funny moments.
        finale = pool[0] if pool else None
        rest = [entry for entry in pool if entry is not finale]

        finale_seconds = 0.0
        if finale:
            finale_span = (
                (finale["start"], finale["end"]) if finale["kind"] == "clip"
                else self._win_span(finale, target) if finale["is_win"]
                # A finale earns a little more room than a moment mid-reel.
                else self._kill_span(finale["peak_at"], MAX_KILL_SECONDS * 1.5)
            )
            finale["span"] = finale_span
            finale_seconds = finale_span[1] - finale_span[0]

        budget = max(0.0, target - finale_seconds)
        per_session: dict[str, int] = {}
        chosen: list[dict[str, Any]] = []
        spent = 0.0
        for entry in rest:
            if per_session.get(entry["job_id"], 0) >= max_per_session:
                continue
            span = (
                (entry["start"], entry["end"]) if entry["kind"] == "clip"
                else self._kill_span(entry["peak_at"], MAX_KILL_SECONDS)
            )
            length = max(MIN_KILL_SECONDS, span[1] - span[0])
            if chosen and spent + length > budget:
                continue
            entry["span"] = span
            chosen.append(entry)
            per_session[entry["job_id"]] = per_session.get(entry["job_id"], 0) + 1
            spent += length
            if spent >= budget:
                break

        chosen.sort(key=lambda entry: (str(entry["source_date"] or ""), entry["start"]))
        selected = [*chosen, *([finale] if finale else [])]
        if len(selected) < 2:
            raise CompilationProjectConflict(
                "Recall found fewer than two usable moments for that theme. "
                "Widen the dates, or restore a source so more can be cut."
            )

        spoken = self._spoken_titles_across(
            [(entry["job_id"], entry["moment"]) for entry in selected if entry["kind"] == "event"]
        )
        summary = {
            "candidates_considered": considered,
            "duplicates_suppressed": 0,
            "sessions_represented": len({entry["job_id"] for entry in selected}),
            "selected_duration_seconds": round(
                sum(entry["span"][1] - entry["span"][0] for entry in selected), 3
            ),
            "target_duration_seconds": round(target, 3),
            "selection_policy": "theme_across_sessions",
            "uncut_moment_count": sum(1 for entry in selected if not entry["clip_id"]),
            "offline_moment_count": len(offline),
            "oversized_clip_count": len(oversized),
            "finale_reason": "win" if (finale and finale["is_win"]) else "strongest",
        }

        project_id = f"project_{uuid.uuid4().hex}"
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO compilation_projects(
                       id, title, template, query, game, date_from, date_to,
                       target_duration_seconds, max_clips, max_per_session,
                       status, generation_version, selection_summary,
                       created_at, updated_at
                   ) VALUES (?, ?, 'theme_montage', ?, ?, ?, ?, ?, ?, ?,
                             'draft', ?, ?, ?, ?)""",
                (
                    project_id, title, query, game, date_from, date_to,
                    target, len(selected), max_per_session,
                    GENERATION_VERSION, json.dumps(summary, separators=(",", ":")),
                    now, now,
                ),
            )
            for position, entry in enumerate(selected):
                is_finale = entry is finale
                if is_finale and entry["is_win"]:
                    reason = "The win this theme turned up. Closes the reel."
                elif is_finale:
                    reason = "The strongest moment of the set. Closes the reel."
                elif entry["kind"] == "clip":
                    reason = "A clip you kept, already rendered."
                else:
                    reason = "Detected moment, ranked on how hard the room reacted."
                moment_title = (
                    entry["title"]
                    or spoken.get((entry["job_id"], entry["start"]))
                    or entry.get("detected_title")
                    or "Moment"
                )
                conn.execute(
                    """INSERT INTO compilation_project_items(
                           id, project_id, clip_id, source_job_id,
                           source_group_snapshot, position,
                           included, selection_score, selection_reason,
                           title_snapshot, session_name_snapshot,
                           source_date_snapshot, start_time_snapshot,
                           end_time_snapshot, source_kind, moment_kind,
                           captions_enabled, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"item_{uuid.uuid4().hex}", project_id, entry["clip_id"],
                        entry["job_id"], entry["job_id"], position, entry["strength"],
                        reason, _clean(moment_title, 200), entry["session_name"],
                        entry["source_date"], round(entry["span"][0], 3),
                        round(entry["span"][1], 3),
                        "clip" if entry["clip_id"] else "event",
                        "win" if entry["is_win"] else ("finale" if is_finale else "moment"),
                        int(bool(is_finale)), now,
                    ),
                )
        return self.get(project_id)

    def _spoken_titles_across(
        self, pairs: list[tuple[str, Optional[MatchMoment]]]
    ) -> dict[tuple[str, float], str]:
        """Transcript-derived names for uncut moments spread across sessions."""
        by_job: dict[str, list[MatchMoment]] = {}
        for job_id, moment in pairs:
            if moment is not None:
                by_job.setdefault(job_id, []).append(moment)
        titles: dict[tuple[str, float], str] = {}
        for job_id, moments in by_job.items():
            for start, text in self._spoken_titles(job_id, moments).items():
                titles[(job_id, start)] = text
        return titles

    # -- the rendered montage -----------------------------------------------

    def record_reel(
        self,
        project_id: str,
        *,
        path: str,
        duration_seconds: Optional[float] = None,
    ) -> dict[str, Any]:
        """Attach the montage that was just rendered for this project."""
        self._project_row(project_id)
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """UPDATE compilation_projects
                   SET reel_path = ?, reel_duration_seconds = ?, reel_built_at = ?,
                       reel_stale = 0, updated_at = ?
                   WHERE id = ?""",
                (str(path), duration_seconds, now, now, project_id),
            )
        return self.get(project_id)

    def mark_reel_stale(self, project_id: str) -> None:
        """The sequence changed, so the rendered montage no longer matches it.

        The file is kept: a creator who drops a moment and changes their mind
        should not have to wait for a re-render to see what they had.
        """
        with self.db.get_connection() as conn:
            conn.execute(
                """UPDATE compilation_projects
                   SET reel_stale = 1, updated_at = ?
                   WHERE id = ? AND reel_path IS NOT NULL""",
                (_utc_now(), project_id),
            )

    def reel_clip_ids(self, project_id: str) -> list[str]:
        """The included clips, in order, that a montage render should stitch."""
        return self.reel_plan(project_id)[0]

    def reel_plan(self, project_id: str) -> tuple[list[str], set[str]]:
        """The clips to stitch, in order, and which of them play silent."""
        project = self.get(project_id)
        playable = [
            item for item in project["items"]
            if item["included"] and item.get("clip_id")
        ]
        clip_ids = [str(item["clip_id"]) for item in playable]
        if len(clip_ids) < 2:
            raise CompilationProjectConflict(
                "A montage needs at least two moments that have been cut."
            )
        muted = {str(item["clip_id"]) for item in playable if item["muted"]}
        return clip_ids, muted

    def set_audio(self, project_id: str, *, muted: bool, keep_finale: bool = True) -> dict[str, Any]:
        """Silence the montage in one action, optionally sparing its last moment.

        The common shape for a posted montage is a track over silent gameplay
        with the creator's voice kept only where the reaction is the payoff, so
        that is one control rather than a per-row chore.
        """
        project = self.get(project_id)
        included = [item for item in project["items"] if item["included"]]
        spare = included[-1]["id"] if (included and keep_finale and muted) else None
        with self.db.get_connection() as conn:
            for item in project["items"]:
                conn.execute(
                    "UPDATE compilation_project_items SET muted = ? WHERE id = ? AND project_id = ?",
                    (int(bool(muted) and item["id"] != spare), item["id"], project_id),
                )
            conn.execute(
                """UPDATE compilation_projects
                   SET status = 'draft', approved_at = NULL, updated_at = ?
                   WHERE id = ?""",
                (_utc_now(), project_id),
            )
        self.mark_reel_stale(project_id)
        return self.get(project_id)

    def approve(self, project_id: str) -> dict[str, Any]:
        project = self.get(project_id)
        included = [item for item in project["items"] if item["included"]]
        if len(included) < 2:
            raise CompilationProjectConflict("Keep at least two clips in the compilation.")
        uncut = [item for item in included if item["media_state"] == "not_cut_yet"]
        if uncut:
            raise CompilationProjectConflict(
                f"Cut {len(uncut)} detected "
                f"{'moment' if len(uncut) == 1 else 'moments'} into clips, "
                "or drop them, before approving this compilation."
            )
        missing = [item for item in included if not item.get("clip_id")]
        if missing:
            raise CompilationProjectConflict(
                "Remove missing clips before approving this compilation."
            )
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """UPDATE compilation_projects
                   SET status = 'approved', approved_at = ?, updated_at = ?
                   WHERE id = ?""",
                (now, now, project_id),
            )
        return self.get(project_id)

    def approved_clip_ids(self, project_id: str) -> list[str]:
        project = self.get(project_id)
        if project["status"] != "approved":
            raise CompilationProjectConflict(
                "Approve the compilation draft before exporting it."
            )
        clip_ids = [
            str(item["clip_id"])
            for item in project["items"]
            if item["included"] and item.get("clip_id")
        ]
        if len(clip_ids) < 2:
            raise CompilationProjectConflict("This project no longer has two exportable clips.")
        return clip_ids

    def delete(self, project_id: str) -> dict[str, Any]:
        project = self._project_row(project_id)
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM compilation_projects WHERE id = ?", (project_id,))
        # The montage belongs to the project and nothing else points at it, so
        # deleting the draft and leaving a 130 MB file behind is a leak. The
        # clips it was stitched from are canonical and are never touched here.
        reel = str(project.get("reel_path") or "")
        if reel and os.path.isfile(reel):
            try:
                os.remove(reel)
            except OSError:
                # A file the creator has open is not worth failing a delete over.
                pass
        return {"status": "deleted", "project_id": project_id}
