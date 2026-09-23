# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Durable, platform-neutral live-session evidence for Recall.

The live layer deliberately records cheap human/platform events rather than
running the expensive VOD pipeline in real time.  After a recording exists,
``build_region_plan`` maps those events onto VOD time and hands bounded search
regions to the existing scan pipeline.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.database import DatabaseManager


SESSION_SCHEMA_VERSION = 1
REGION_PLAN_SCHEMA_VERSION = 1
DEFAULT_PRE_ROLL_SECONDS = 90.0
DEFAULT_POST_ROLL_SECONDS = 30.0
MAX_PLAN_EVENTS = 64
MAX_PLAN_SECONDS = 1800.0

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class RecallSessionError(ValueError):
    """Base error for invalid session operations."""


class RecallSessionNotFound(RecallSessionError):
    pass


class RecallSessionConflict(RecallSessionError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise RecallSessionError(f"{field} must be an ISO-8601 timestamp.") from exc
    else:
        raise RecallSessionError(f"{field} is required.")
    if parsed.tzinfo is None:
        raise RecallSessionError(f"{field} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _finite_nonnegative(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RecallSessionError(f"{field} must be a number.") from exc
    if not math.isfinite(result) or result < 0:
        raise RecallSessionError(f"{field} must be finite and non-negative.")
    return result


def _json_object(value: Optional[dict]) -> str:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise RecallSessionError("payload and metadata values must be objects.")
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _decode_json(value: Any) -> dict:
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


class RecallSessionService:
    """Own Recall Session state without depending on Twitch, OBS, or a UI."""

    def __init__(
        self,
        db: DatabaseManager,
        *,
        now: Callable[[], datetime] = _utc_now,
    ):
        self.db = db
        self._now = now

    @staticmethod
    def _slug(value: str, *, field: str) -> str:
        normalized = str(value or "").strip().lower()
        if not _SLUG_RE.fullmatch(normalized):
            raise RecallSessionError(
                f"{field} must start with a letter and contain only letters, numbers, '_' or '-'."
            )
        return normalized

    @staticmethod
    def _session_dict(row: Any) -> dict:
        result = dict(row)
        result["metadata"] = _decode_json(result.get("metadata"))
        return result

    @staticmethod
    def _event_dict(row: Any) -> dict:
        result = dict(row)
        result["payload"] = _decode_json(result.get("payload"))
        return result

    def get_active(self) -> Optional[dict]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM recall_sessions WHERE status = 'active' "
                "ORDER BY started_at_utc DESC LIMIT 1"
            ).fetchone()
        return self._session_dict(row) if row else None

    def list_sessions(self, *, limit: int = 50) -> list[dict]:
        bounded = max(1, min(int(limit), 200))
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM recall_sessions ORDER BY started_at_utc DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [self._session_dict(row) for row in rows]

    def get(self, session_id: str, *, include_events: bool = True) -> dict:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM recall_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                raise RecallSessionNotFound("Recall Session not found.")
            result = self._session_dict(row)
            if include_events:
                events = conn.execute(
                    "SELECT * FROM recall_session_events WHERE session_id = ? "
                    "ORDER BY occurred_at_utc, created_at, id",
                    (session_id,),
                ).fetchall()
                anchors = conn.execute(
                    "SELECT * FROM recall_session_clock_anchors WHERE session_id = ? "
                    "ORDER BY wall_time_utc, created_at, id",
                    (session_id,),
                ).fetchall()
                result["events"] = [self._event_dict(event) for event in events]
                result["clock_anchors"] = [dict(anchor) for anchor in anchors]
        return result

    def start(
        self,
        *,
        source_platform: str = "unknown",
        source_ref: Optional[str] = None,
        title: Optional[str] = None,
        started_at_utc: Any = None,
        initial_stream_time_seconds: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        platform = self._slug(source_platform, field="source_platform")
        started = _parse_utc(started_at_utc, field="started_at_utc") \
            if started_at_utc is not None else self._now().astimezone(timezone.utc)
        session_id = str(uuid.uuid4())
        clean_ref = str(source_ref).strip()[:2048] if source_ref else None
        clean_title = str(title).strip()[:200] if title else None
        with self.db.get_connection() as conn:
            active = conn.execute(
                "SELECT id FROM recall_sessions WHERE status = 'active' LIMIT 1"
            ).fetchone()
            if active:
                raise RecallSessionConflict(
                    "A Recall Session is already active. Stop it before starting another."
                )
            conn.execute(
                """INSERT INTO recall_sessions
                   (id, status, source_platform, source_ref, title,
                    started_at_utc, metadata, schema_version)
                   VALUES (?, 'active', ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    platform,
                    clean_ref,
                    clean_title,
                    _iso_utc(started),
                    _json_object(metadata),
                    SESSION_SCHEMA_VERSION,
                ),
            )
            if initial_stream_time_seconds is not None:
                stream_time = _finite_nonnegative(
                    initial_stream_time_seconds, field="initial_stream_time_seconds"
                )
                conn.execute(
                    """INSERT INTO recall_session_clock_anchors
                       (id, session_id, anchor_type, wall_time_utc,
                        stream_time_seconds, uncertainty_seconds, source, schema_version)
                       VALUES (?, ?, 'session_start', ?, ?, 1.0, 'explicit', ?)""",
                    (
                        str(uuid.uuid4()), session_id, _iso_utc(started), stream_time,
                        SESSION_SCHEMA_VERSION,
                    ),
                )
        return self.get(session_id)

    def stop(self, session_id: str, *, ended_at_utc: Any = None) -> dict:
        ended = _parse_utc(ended_at_utc, field="ended_at_utc") \
            if ended_at_utc is not None else self._now().astimezone(timezone.utc)
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT status, started_at_utc FROM recall_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                raise RecallSessionNotFound("Recall Session not found.")
            if row["status"] != "active":
                raise RecallSessionConflict("Recall Session is not active.")
            if ended < _parse_utc(row["started_at_utc"], field="started_at_utc"):
                raise RecallSessionError("ended_at_utc cannot be before the session start.")
            conn.execute(
                "UPDATE recall_sessions SET status = 'ended', ended_at_utc = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (_iso_utc(ended), session_id),
            )
        return self.get(session_id)

    def add_clock_anchor(
        self,
        session_id: str,
        *,
        wall_time_utc: Any,
        stream_time_seconds: float,
        anchor_type: str = "player_position",
        uncertainty_seconds: float = 1.0,
        source: str = "manual",
    ) -> dict:
        wall_time = _parse_utc(wall_time_utc, field="wall_time_utc")
        stream_time = _finite_nonnegative(stream_time_seconds, field="stream_time_seconds")
        uncertainty = _finite_nonnegative(uncertainty_seconds, field="uncertainty_seconds")
        kind = self._slug(anchor_type, field="anchor_type")
        clean_source = self._slug(source, field="source")
        anchor_id = str(uuid.uuid4())
        with self.db.get_connection() as conn:
            exists = conn.execute(
                "SELECT 1 FROM recall_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not exists:
                raise RecallSessionNotFound("Recall Session not found.")
            conn.execute(
                """INSERT INTO recall_session_clock_anchors
                   (id, session_id, anchor_type, wall_time_utc,
                    stream_time_seconds, uncertainty_seconds, source, schema_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    anchor_id, session_id, kind, _iso_utc(wall_time), stream_time,
                    uncertainty, clean_source, SESSION_SCHEMA_VERSION,
                ),
            )
        return {
            "id": anchor_id,
            "session_id": session_id,
            "anchor_type": kind,
            "wall_time_utc": _iso_utc(wall_time),
            "stream_time_seconds": stream_time,
            "uncertainty_seconds": uncertainty,
            "source": clean_source,
            "schema_version": SESSION_SCHEMA_VERSION,
        }

    def record_event(
        self,
        session_id: str,
        *,
        kind: str = "remember",
        source: str = "hotkey",
        occurred_at_utc: Any = None,
        stream_offset_seconds: Optional[float] = None,
        confidence: float = 1.0,
        payload: Optional[dict] = None,
    ) -> dict:
        occurred = _parse_utc(occurred_at_utc, field="occurred_at_utc") \
            if occurred_at_utc is not None else self._now().astimezone(timezone.utc)
        event_kind = self._slug(kind, field="kind")
        event_source = self._slug(source, field="source")
        try:
            clean_confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise RecallSessionError("confidence must be a number.") from exc
        if not math.isfinite(clean_confidence) or not 0 <= clean_confidence <= 1:
            raise RecallSessionError("confidence must be between 0 and 1.")
        stream_offset = None
        if stream_offset_seconds is not None:
            stream_offset = _finite_nonnegative(
                stream_offset_seconds, field="stream_offset_seconds"
            )
        event_id = str(uuid.uuid4())
        with self.db.get_connection() as conn:
            session = conn.execute(
                "SELECT status FROM recall_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not session:
                raise RecallSessionNotFound("Recall Session not found.")
            if session["status"] != "active":
                raise RecallSessionConflict("Events can only be added to an active Recall Session.")
            conn.execute(
                """INSERT INTO recall_session_events
                   (id, session_id, kind, source, occurred_at_utc,
                    stream_offset_seconds, confidence, payload, schema_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id, session_id, event_kind, event_source,
                    _iso_utc(occurred), stream_offset, clean_confidence,
                    _json_object(payload), SESSION_SCHEMA_VERSION,
                ),
            )
            conn.execute(
                "UPDATE recall_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
        return {
            "id": event_id,
            "session_id": session_id,
            "kind": event_kind,
            "source": event_source,
            "occurred_at_utc": _iso_utc(occurred),
            "stream_offset_seconds": stream_offset,
            "confidence": clean_confidence,
            "payload": payload or {},
            "schema_version": SESSION_SCHEMA_VERSION,
        }

    def delete_event(self, session_id: str, event_id: str) -> dict:
        """Remove one mark from a session.

        A stray press -- a misfire, or a moment that turned out to be nothing --
        costs a two-minute search window in the scan that follows, so it has to
        be removable. Allowed on an ended session as well as an active one:
        reviewing the marks before attaching the recording is exactly when a
        creator notices the bad one.
        """
        with self.db.get_connection() as conn:
            session = conn.execute(
                "SELECT id FROM recall_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not session:
                raise RecallSessionNotFound("Recall Session not found.")
            deleted = conn.execute(
                "DELETE FROM recall_session_events WHERE id = ? AND session_id = ?",
                (event_id, session_id),
            ).rowcount
            if not deleted:
                raise RecallSessionNotFound("Recall Session event not found.")
            conn.execute(
                "UPDATE recall_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
        return {"deleted": event_id, "session_id": session_id}

    def record_active_remember(self) -> dict:
        active = self.get_active()
        if not active:
            raise RecallSessionConflict("No Recall Session is active.")
        return self.record_event(active["id"], kind="remember", source="hotkey")

    @staticmethod
    def _merge_markers(markers: list[dict], duration: Optional[float]) -> list[dict]:
        regions: list[dict] = []
        for marker in sorted(markers, key=lambda item: item["search_start"]):
            start = max(0.0, float(marker["search_start"]))
            end = float(marker["search_end"])
            if duration is not None:
                end = min(end, duration)
            if end <= start:
                continue
            if regions and start <= regions[-1]["end"] + 12.0:
                regions[-1]["end"] = max(regions[-1]["end"], end)
                regions[-1]["event_ids"].append(marker["event_id"])
                regions[-1]["marker_times"].append(marker["timestamp"])
            else:
                regions.append({
                    "start": start,
                    "end": end,
                    "priority": "creator",
                    "event_ids": [marker["event_id"]],
                    "marker_times": [marker["timestamp"]],
                })
        return regions

    @staticmethod
    def _bounded_regions(regions: list[dict]) -> tuple[list[dict], int]:
        chosen: list[dict] = []
        used = 0.0
        omitted_events = 0
        for region in regions:
            length = float(region["end"]) - float(region["start"])
            if chosen and used + length > MAX_PLAN_SECONDS:
                omitted_events += len(region["event_ids"])
                continue
            chosen.append(region)
            used += length
        return chosen, omitted_events

    def build_region_plan(
        self,
        session_id: str,
        *,
        vod_started_at_utc: Any = None,
        vod_duration_seconds: Optional[float] = None,
        pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS,
        post_roll_seconds: float = DEFAULT_POST_ROLL_SECONDS,
    ) -> dict:
        session = self.get(session_id)
        duration = None
        if vod_duration_seconds is not None:
            duration = _finite_nonnegative(vod_duration_seconds, field="vod_duration_seconds")
        pre_roll = _finite_nonnegative(pre_roll_seconds, field="pre_roll_seconds")
        post_roll = _finite_nonnegative(post_roll_seconds, field="post_roll_seconds")

        anchors = list(session.get("clock_anchors") or [])
        if vod_started_at_utc is not None:
            vod_start = _parse_utc(vod_started_at_utc, field="vod_started_at_utc")
            # A confirmed VOD start supersedes the provisional "session began
            # at stream time X" anchor. Keep real player-position anchors:
            # they observe the viewer's actual playback clock and are more
            # precise than a wall-clock estimate.
            anchors = [
                anchor for anchor in anchors
                if anchor.get("anchor_type") != "session_start"
            ]
            anchors.append({
                "id": "job-attach-vod-start",
                "anchor_type": "vod_start",
                "wall_time_utc": _iso_utc(vod_start),
                "stream_time_seconds": 0.0,
                # Viewer-side live latency is not observable from a wall clock.
                # The wide backward search window handles it; keep the estimate
                # explicit so the UI never claims frame accuracy.
                "uncertainty_seconds": 30.0,
                "source": "job_attach",
            })

        parsed_anchors = []
        for anchor in anchors:
            parsed_anchors.append((
                anchor,
                _parse_utc(anchor["wall_time_utc"], field="wall_time_utc"),
            ))

        events = [
            event for event in session.get("events") or []
            if event.get("kind") == "remember"
        ]
        mapped: list[dict] = []
        unmapped_ids: list[str] = []
        limited_events = events[:MAX_PLAN_EVENTS]
        capacity_omitted = max(0, len(events) - len(limited_events))
        for event in limited_events:
            estimate = event.get("stream_offset_seconds")
            alignment = "player_position" if estimate is not None else None
            uncertainty = 1.0 if estimate is not None else None
            anchor_id = None
            if estimate is None and parsed_anchors:
                event_time = _parse_utc(event["occurred_at_utc"], field="occurred_at_utc")
                anchor, anchor_time = min(
                    parsed_anchors,
                    key=lambda item: abs((event_time - item[1]).total_seconds()),
                )
                estimate = float(anchor["stream_time_seconds"]) + (
                    event_time - anchor_time
                ).total_seconds()
                uncertainty = float(anchor.get("uncertainty_seconds") or 0.0)
                alignment = str(anchor.get("anchor_type") or "clock_anchor")
                anchor_id = anchor.get("id")
            if estimate is None or not math.isfinite(float(estimate)) or float(estimate) < 0:
                unmapped_ids.append(event["id"])
                continue
            timestamp = float(estimate)
            if duration is not None and timestamp > duration:
                unmapped_ids.append(event["id"])
                continue
            marker = {
                "event_id": event["id"],
                "timestamp": round(timestamp, 3),
                "search_start": round(max(0.0, timestamp - pre_roll), 3),
                "search_end": round(timestamp + post_roll, 3),
                "alignment": alignment,
                "anchor_id": anchor_id,
                "uncertainty_seconds": round(float(uncertainty or 0.0), 3),
                "confidence": float(event.get("confidence") or 0.0),
                "source": event.get("source"),
            }
            if duration is not None:
                marker["search_end"] = round(min(marker["search_end"], duration), 3)
            mapped.append(marker)

        regions = self._merge_markers(mapped, duration)
        regions, budget_omitted = self._bounded_regions(regions)
        selected_event_ids = {
            event_id for region in regions for event_id in region["event_ids"]
        }
        markers = [marker for marker in mapped if marker["event_id"] in selected_event_ids]
        omitted_count = capacity_omitted + budget_omitted
        return {
            "schema_version": REGION_PLAN_SCHEMA_VERSION,
            "session_id": session_id,
            "strategy": "creator_markers_then_deep_recall",
            "alignment_status": (
                "ready" if markers and not unmapped_ids and omitted_count == 0 else
                "partial" if markers else
                "unmapped"
            ),
            "vod_started_at_utc": (
                _iso_utc(_parse_utc(vod_started_at_utc, field="vod_started_at_utc"))
                if vod_started_at_utc is not None else None
            ),
            "vod_duration_seconds": duration,
            "pre_roll_seconds": pre_roll,
            "post_roll_seconds": post_roll,
            "markers": markers,
            "regions": regions,
            "mapped_event_count": len(markers),
            "unmapped_event_ids": unmapped_ids,
            "omitted_event_count": omitted_count,
            "total_region_seconds": round(sum(
                float(region["end"]) - float(region["start"]) for region in regions
            ), 3),
        }
