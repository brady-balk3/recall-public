# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from core.bundle_paths import get_data_dir
from core.artifacts import CANDIDATES_FILENAME
from core.clip_media import (
    REBUILDABLE,
    UNAVAILABLE,
    annotate_clip_rows,
    job_media_summary,
    summary_state,
)
from core.database import DatabaseManager
from core.clip_operations import serialized_clip_operation
from core.job_manager import JobManager
from core.recall_sessions import RecallSessionError, RecallSessionService
from core.more_candidates import (
    MORE_CANDIDATE_LABEL,
    MORE_CANDIDATE_LIMIT,
    ceiling_candidates_from_trace,
    more_candidate_clip_id,
    partition_second_look,
    second_look_deck_score,
)
from core.storage_lifecycle import (
    SourceLifecycleService,
    has_active_source_restores,
    source_in_use,
)

logger = logging.getLogger(__name__)

# How long the first audible clip after a silent run takes to reach full level.
# Short enough to keep the reaction's attack, long enough that it does not pop.
REEL_FADE_IN_SECONDS = 0.45

# In-flight clip media rebuilds, keyed by job id. Guards against a double click
# fanning a session's rebuild out into two competing FFmpeg queues.
_REBUILD_LOCK = threading.Lock()
_ACTIVE_REBUILDS: Dict[str, Dict[str, Any]] = {}


class _OperationProgressReporter:
    """Clamp operation progress and add guarded elapsed/ETA telemetry once."""

    def __init__(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]],
        *,
        kind: str,
        total_clips: int,
        snapshot: Callable[[], Dict[str, Any]],
    ):
        self.callback = callback
        self.kind = kind
        self.total_clips = total_clips
        self.snapshot = snapshot
        self.started_at = time.monotonic()
        self.last_progress = 0.0

    def report(self, **payload) -> None:
        if "progress" in payload:
            raw_progress = max(0.0, min(1.0, float(payload["progress"])))
            self.last_progress = max(self.last_progress, raw_progress)
            payload["progress"] = self.last_progress
            elapsed = max(0.0, time.monotonic() - self.started_at)
            eta = None
            if elapsed >= 1.0 and 0.02 <= self.last_progress < 1.0:
                eta = max(0.0, (elapsed / self.last_progress) - elapsed)
            payload.setdefault("elapsed_seconds", elapsed)
            payload.setdefault("eta_seconds", eta)
        if self.callback is None:
            return
        try:
            self.callback({
                "kind": self.kind,
                "total_clips": self.total_clips,
                **self.snapshot(),
                **payload,
            })
        except Exception:
            logger.debug("Operation progress callback failed", exc_info=True)


class LearningService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def maybe_train_background(self) -> None:
        try:
            from core import ranker_service
            ranker_service.sync_user_ranker(self.db)
        except Exception:
            logger.exception("Background ranker training failed")

    def status(self) -> Dict[str, Any]:
        from core import ranker_service
        return ranker_service.ranker_status(self.db)

    def train(self) -> Dict[str, Any]:
        from core import ranker_service
        return ranker_service.train_user_ranker(self.db)

    def reset(self) -> Dict[str, Any]:
        from core import ranker_service
        return ranker_service.reset_learning(self.db)


def _twitch_vod_started_at(source_path: str) -> Optional[str]:
    """The VOD's real UTC start, or None when it can't be established.

    A live Recall Session records wall-clock Remember presses, so mapping them
    onto VOD time needs the instant the VOD itself began. TwitchDownloaderCLI
    already returns that as ``created_at`` in the video block it embeds in every
    chat download, so a 1-second slice answers it in about a second.

    Returning None is a real answer: the planner then falls back to the
    session's own clock anchors rather than guessing a start time.
    """
    try:
        from engines.chat.twitch_chat import fetch_twitch_video_meta, is_twitch_vod
    except Exception:
        return None
    if not source_path or not is_twitch_vod(source_path):
        return None
    try:
        meta = fetch_twitch_video_meta(source_path, timeout=30.0)
    except Exception:
        logger.warning("Could not fetch Twitch VOD metadata for the session start time.")
        return None
    return (meta or {}).get("created_at") or None


class JobService:
    def __init__(
        self,
        db: DatabaseManager,
        job_manager: JobManager,
        recall_sessions: Optional[RecallSessionService] = None,
    ):
        self.db = db
        self.job_manager = job_manager
        self.lifecycle = SourceLifecycleService(db, job_manager)
        self.recall_sessions = recall_sessions

    def run(self, request: Any) -> Dict[str, str]:
        if has_active_source_restores():
            raise HTTPException(
                status_code=409,
                detail="Wait for source preparation to finish or cancel it before starting a scan.",
            )
        if request.source_type in ("url", "twitch"):
            preflight = self.lifecycle.preflight_download(request.source_path)
            if not preflight["allowed"]:
                raise HTTPException(
                    status_code=507,
                    detail={
                        "message": "There is not enough free disk space to safely download this Twitch source.",
                        "required_bytes": preflight["required_bytes"],
                        "free_bytes": preflight["free_bytes"],
                        "safe_sources": preflight["safe_sources"],
                    },
                )
        from core.source_date import source_date_for_input

        source_date = source_date_for_input(
            request.source_path,
            request.source_type,
            getattr(request, "source_date", None),
        )
        settings = dict(request.settings or {})
        recall_session_id = getattr(request, "recall_session_id", None)
        region_plan = None
        if recall_session_id:
            if self.recall_sessions is None:
                raise HTTPException(status_code=503, detail="Recall Sessions are unavailable.")
            # A live session's markers are wall-clock only. Without the VOD's
            # real start there is no honest mapping onto VOD time -- and the
            # session_start anchor written at "stream time 0" would silently
            # place every marker as though the stream began when the viewer
            # tuned in. Deriving the true start supersedes that anchor.
            vod_started_at = getattr(request, "vod_started_at_utc", None)
            if not vod_started_at and request.source_type in ("url", "twitch"):
                vod_started_at = _twitch_vod_started_at(request.source_path)
                if vod_started_at:
                    logger.info("Recall Session mapped to VOD start %s", vod_started_at)
            try:
                region_plan = self.recall_sessions.build_region_plan(
                    recall_session_id,
                    vod_started_at_utc=vod_started_at,
                )
            except RecallSessionError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if not region_plan.get("markers"):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "This Recall Session has no VOD-aligned Remember markers. "
                        "Provide the VOD start time or a player-position clock anchor."
                    ),
                )
            settings["recallRegionPlan"] = region_plan
        if recall_session_id:
            job_id = self.job_manager.create_job(
                request.source_path,
                request.source_type,
                request.session_name,
                source_date,
                recall_session_id=recall_session_id,
                region_plan=region_plan,
            )
        else:
            # Keep the stable no-session boundary exactly unchanged. Besides
            # compatibility, this makes the rollback property concrete: an
            # ordinary scan never has to know the live subsystem exists.
            job_id = self.job_manager.create_job(
                request.source_path,
                request.source_type,
                request.session_name,
                source_date,
            )
        # Selection/judge policy is engine-owned. The desktop only supplies the
        # remaining creator-facing processing and export preferences.
        self.job_manager.start_job(job_id, settings=settings)
        return {"job_id": job_id, "status": "started"}

    def download(self, request: Any) -> Dict[str, str]:
        if has_active_source_restores():
            raise HTTPException(
                status_code=409,
                detail="Wait for source preparation to finish or cancel it before downloading another VOD.",
            )
        job_id = self.job_manager.create_job(request.url, "url")
        self.job_manager.start_job(job_id, settings={"only_download": True})
        return {"job_id": job_id, "status": "download_started"}

    def cancel(self, job_id: str) -> Dict[str, str]:
        job = self.job_manager.get_job(job_id, include_events=False)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not self.job_manager.is_active(job_id):
            raise HTTPException(status_code=409, detail="This job is not currently running.")
        self.job_manager.cancel_job(job_id)
        return {"job_id": job_id, "status": "cancelling"}

    def diagnostics(self, job_id: str) -> Dict[str, str]:
        """Build a shareable report without source content or raw log strings."""
        from core.bundle_paths import is_frozen
        from core.diagnostics import support_report
        from core.process_governor import available_memory_gb

        job = self.job_manager.get_job(job_id, include_events=False)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        text = support_report(
            job,
            self.job_manager.get_job_events(job_id, limit=60),
            packaged=is_frozen(),
            cpu_count=os.cpu_count(),
            ram_gb=available_memory_gb(),
        )
        return {
            "job_id": job_id,  # Local API routing only; excluded from exported text/name.
            "filename": "recall-diagnostics.txt",
            "text": text,
        }

    def delete(self, job_id: str) -> Dict[str, str]:
        # A running pipeline thread would keep writing rows for a job that no
        # longer exists (and burn CPU for nothing) — refuse until cancelled.
        if self.job_manager.is_active(job_id):
            raise HTTPException(
                status_code=409,
                detail="This session is still processing. Cancel the scan before deleting it.",
            )
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT export_path, thumb_path FROM clips WHERE job_id = ?", (job_id,))
            clip_rows = cursor.fetchall()
            cursor.execute("SELECT poster_path FROM jobs WHERE id = ?", (job_id,))
            job_row = cursor.fetchone()
            poster_path = job_row["poster_path"] if job_row else None
            cursor.execute(
                "DELETE FROM exports WHERE clip_id IN (SELECT id FROM clips WHERE job_id = ?)",
                (job_id,),
            )
            # job_events holds an FK to jobs; with foreign_keys=ON the jobs
            # delete fails unless the event log goes first.
            cursor.execute("DELETE FROM job_events WHERE job_id = ?", (job_id,))
            cursor.execute("DELETE FROM reaction_timelines WHERE job_id = ?", (job_id,))
            cursor.execute("DELETE FROM clips WHERE job_id = ?", (job_id,))
            cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            cursor.execute(
                """UPDATE stream_memory_semantic_state
                   SET source_revision = source_revision + 1,
                       status = CASE WHEN indexed_entries > 0 THEN 'stale' ELSE 'not_built' END
                   WHERE id = 1"""
            )
            # Source ownership is independent of a session. The FK cascade only
            # unlinks this job; lifecycle reconciliation decides later whether
            # a Recall-downloaded source is safe to remove.

        for row in clip_rows:
            for path in (row["export_path"], row["thumb_path"]):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        logger.debug("Could not delete clip artifact %s", path, exc_info=True)
        if poster_path and os.path.exists(poster_path):
            try:
                os.remove(poster_path)
            except Exception:
                logger.debug("Could not delete session poster %s", poster_path, exc_info=True)
        artifact_dir = os.path.join(get_data_dir(), "jobs", job_id)
        if os.path.isdir(artifact_dir):
            shutil.rmtree(artifact_dir, ignore_errors=True)
        self.lifecycle.reconcile_sources(remove_expired=True)
        self.lifecycle.reconcile_recall_storage()
        return {"status": "deleted", "job_id": job_id}

    @staticmethod
    def db_data_root() -> str:
        return get_data_dir()

    def list(self) -> list:
        jobs = self.job_manager.list_jobs()
        # One batched pass for the whole gallery. Session cards need to know
        # whether their clip media survived before they mount a poster request,
        # and a per-card status call would be a hundred round trips.
        summaries = job_media_summary(self.db, [job["id"] for job in jobs])
        for job in jobs:
            summary = summaries.get(job["id"])
            if summary:
                job["media"] = {**summary, "state": summary_state(summary)}
        return jobs

    def media_status(self, job_id: str) -> Dict[str, Any]:
        """What survived on disk for this session, and whether it can come back."""
        summaries = job_media_summary(self.db, [job_id])
        summary = summaries.get(job_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, **summary, "state": summary_state(summary)}

    def get(self, job_id: str) -> Dict[str, Any]:
        job = self.job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def timeline(self, job_id: str) -> Dict[str, Any]:
        return {"job_id": job_id, "timeline": self.db.get_reaction_timeline(job_id)}

    def source_media(self, job_id: str):
        """The job's full source VOD as a seekable FileResponse (Cutting Room).

        asset_path is the pipeline-resolved local video (set for both local
        and downloaded sources once a scan has run); source_path is the
        fallback for local-file jobs whose asset_path was never written or
        was cleared by "clear downloaded VODs". Starlette's FileResponse
        implements single- and multi-range requests, so no custom Range
        handling is needed here.
        """
        import mimetypes

        from fastapi.responses import FileResponse

        with self.db.get_connection() as conn:
            row = conn.cursor().execute(
                "SELECT asset_path, source_path, source_type FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        candidates = [row["asset_path"]]
        if (row["source_type"] or "file") == "file":
            candidates.append(row["source_path"])
        path = next((p for p in candidates if p and os.path.isfile(p)), None)
        if not path:
            raise HTTPException(
                status_code=404,
                detail="The source recording for this session is no longer on disk.",
            )
        media_type = mimetypes.guess_type(path)[0] or "video/mp4"
        return FileResponse(path, media_type=media_type)

    def filmstrip_path(
        self,
        job_id: str,
        *,
        start: float = 0.0,
        end: float | None = None,
        frames: int = 24,
        width: int = 1600,
        height: int = 64,
    ) -> str:
        """Return a cached source-frame contact sheet for the VOD editor.

        The full-VOD navigator and precision rail deliberately share this
        endpoint so both surfaces show the same truthful source imagery at
        different temporal scales. Cache keys include source stat data and all
        render parameters; replacing a retained VOD cannot reuse stale frames.
        """
        import hashlib
        import math

        from core.thumbnails import generate_filmstrip

        with self.db.get_connection() as conn:
            row = conn.cursor().execute(
                "SELECT asset_path, source_path, source_type, duration "
                "FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")

        candidates = [row["asset_path"]]
        if (row["source_type"] or "file") == "file":
            candidates.append(row["source_path"])
        source = next((path for path in candidates if path and os.path.isfile(path)), None)
        if not source:
            raise HTTPException(
                status_code=404,
                detail="The source recording for this session is no longer on disk.",
            )

        duration = max(0.0, float(row["duration"] or 0.0))
        try:
            range_start = max(0.0, float(start))
            range_end = float(end) if end is not None else duration
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Invalid filmstrip range")
        if duration > 0:
            range_start = min(range_start, max(0.0, duration - 0.05))
            range_end = min(range_end, duration)
        if not math.isfinite(range_start) or not math.isfinite(range_end) or range_end <= range_start:
            raise HTTPException(status_code=422, detail="Filmstrip end must be after start")

        frame_count = max(4, min(48, int(frames)))
        render_width = max(320, min(2560, int(width)))
        render_height = max(28, min(160, int(height)))
        cell_width = max(48, min(240, int(math.ceil(render_width / frame_count))))
        stat = os.stat(source)
        cache_seed = "|".join((
            os.path.abspath(source),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            f"{range_start:.3f}",
            f"{range_end:.3f}",
            str(frame_count),
            str(cell_width),
            str(render_height),
        ))
        digest = hashlib.sha1(cache_seed.encode("utf-8")).hexdigest()[:24]  # nosec B324
        cache_dir = os.path.join(get_data_dir(), "cache", "filmstrips")
        out_path = os.path.join(cache_dir, f"filmstrip_{digest}.jpg")
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return out_path

        made = generate_filmstrip(
            source,
            out_path,
            start_seconds=range_start,
            end_seconds=range_end,
            frames=frame_count,
            cell_width=cell_width,
            cell_height=render_height,
        )
        if not made:
            raise HTTPException(status_code=404, detail="Source filmstrip unavailable")
        return made

    def poster_path(self, job_id: str) -> str:
        """Absolute path to a still for session gallery cards.

        Prefer a landscape frame from the source VOD at the strongest clip's
        reaction peak (else ~10% in). If the VOD is gone, fall back to a frame
        from the best exported clip MP4 so cards aren't blank.
        """
        from core.thumbnails import (
            clip_relative_peak,
            generate_thumbnail,
            thumb_filename,
        )

        exports_dir = os.path.join(get_data_dir(), "exports")
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT poster_path, asset_path, source_path, source_type, duration "
                "FROM jobs WHERE id = ?",
                (job_id,),
            )
            job = cursor.fetchone()
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            existing = job["poster_path"]
            if existing and os.path.exists(existing):
                return existing
            candidate = os.path.join(exports_dir, thumb_filename(f"job_{job_id}"))
            if os.path.exists(candidate):
                cursor.execute(
                    "UPDATE jobs SET poster_path = ? WHERE id = ?",
                    (candidate, job_id),
                )
                return candidate
            cursor.execute(
                "SELECT export_path, peak_timestamp, start_time, end_time, score, deck_score "
                "FROM clips WHERE job_id = ? AND COALESCE(story_label, '') != ? "
                "ORDER BY COALESCE(deck_score, score, 0) DESC LIMIT 1",
                (job_id, "recall_more_candidate"),
            )
            best = cursor.fetchone()

        source_candidates = [job["asset_path"]]
        if (job["source_type"] or "file") == "file":
            source_candidates.append(job["source_path"])
        source = next((p for p in source_candidates if p and os.path.isfile(p)), None)

        duration = float(job["duration"] or 0.0)
        seek = min(max(2.0, duration * 0.1 if duration > 0 else 20.0), 60.0)
        grab_from = source
        if source and best is not None:
            peak = best["peak_timestamp"]
            start = float(best["start_time"] or 0.0)
            end = float(best["end_time"] or 0.0)
            try:
                if peak is not None:
                    seek = max(0.0, float(peak))
                elif end > start:
                    seek = start + (end - start) / 2.0
            except (TypeError, ValueError):
                pass
            if duration > 0:
                seek = min(seek, max(0.0, duration - 0.5))
        elif best is not None and best["export_path"] and os.path.isfile(best["export_path"]):
            # Source VOD missing — still give the card a frame from a clip MP4.
            grab_from = best["export_path"]
            seek = clip_relative_peak(
                best["peak_timestamp"], best["start_time"], best["end_time"]
            )

        if not grab_from:
            raise HTTPException(
                status_code=404,
                detail="No source recording or clip export available for a poster.",
            )

        made = generate_thumbnail(
            grab_from,
            f"job_{job_id}",
            exports_dir,
            seek_seconds=seek,
            width=1280,
            jpeg_quality=3,
        )
        if not made:
            raise HTTPException(status_code=404, detail="Session poster unavailable")
        with self.db.get_connection() as conn:
            conn.cursor().execute(
                "UPDATE jobs SET poster_path = ? WHERE id = ?",
                (made, job_id),
            )
        return made


class ClipService:
    REEL_MIN_TIMEOUT_SECONDS = 120
    REEL_MAX_TIMEOUT_SECONDS = 3600
    REEL_TIMEOUT_OVERHEAD_SECONDS = 30
    REEL_TIMEOUT_PER_MEDIA_SECOND = 4.0

    MORE_CANDIDATE_LABEL = MORE_CANDIDATE_LABEL
    MORE_CANDIDATE_LIMIT = MORE_CANDIDATE_LIMIT

    @classmethod
    def _reel_timeout_seconds(cls, reel_duration: float) -> int:
        estimated = int(
            cls.REEL_TIMEOUT_OVERHEAD_SECONDS
            + max(0.0, float(reel_duration)) * cls.REEL_TIMEOUT_PER_MEDIA_SECOND
        )
        return min(
            cls.REEL_MAX_TIMEOUT_SECONDS,
            max(cls.REEL_MIN_TIMEOUT_SECONDS, estimated),
        )

    def __init__(
        self,
        db: DatabaseManager,
        learning_service: LearningService,
        job_manager: Optional[JobManager] = None,
    ):
        self.db = db
        self.learning_service = learning_service
        # Only used to stream rebuild progress into session activity. Optional
        # so the many tests that construct a bare ClipService keep working.
        self.job_manager = job_manager
        self.data_root = get_data_dir()
        self.exports_dir = os.path.join(self.data_root, "exports")
        self._framing_cache: Dict[str, tuple[Any, Dict[str, Any]]] = {}
        # One decoded probe frame per (job, clip midpoint). The VOD Editor
        # re-resolves framing on every in/out nudge and once per candidate in a
        # 15-clip batch; without this each of those seeks the source again.
        self._gameplay_probe_cache: Dict[tuple, Optional[List[float]]] = {}
        self._facecam_presence_probe_cache: Dict[tuple, bool] = {}
        self._fullframe_camera_probe_cache: Dict[tuple, Optional[float]] = {}
        # A review action must never fan out into many competing FFmpeg jobs.
        # The frontend requests proofs explicitly, and this lock is the final
        # guard against duplicate clicks or overlapping clients.
        self._preview_render_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Media rebuild
    #
    # Deliberately explicit. Opening a session whose exports were reclaimed
    # never starts work on its own: rebuilding is minutes of FFmpeg, and on a
    # Twitch session it may first need a multi-gigabyte download. The creator
    # asks for it, the same way they already ask to restore a source.
    # ------------------------------------------------------------------

    def rebuild_status(self, job_id: str) -> Dict[str, Any]:
        with _REBUILD_LOCK:
            active = dict(_ACTIVE_REBUILDS.get(job_id) or {})
        return active or {"status": "idle"}

    @serialized_clip_operation
    def _rebuild_missing_clip(self, clip_id: str) -> None:
        # The queue snapshot may predate an edit or another successful render.
        # Read again only after acquiring ownership, then retain it through commit.
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        if row is None or (row["export_path"] and os.path.isfile(row["export_path"])):
            return
        as_preview = bool(row["preview_only"] if row["preview_only"] is not None else 1)
        rendered_path, thumb_path = self._render_existing_clip(
            row, float(row["start_time"] or 0.0), float(row["end_time"] or 0.0),
            preview=as_preview,
        )
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE clips SET export_path = ?, thumb_path = ?, preview_only = ? WHERE id = ?",
                (rendered_path, thumb_path, int(as_preview), clip_id),
            )

    def rebuild_media(self, job_id: str) -> Dict[str, Any]:
        """Re-render this session's missing clip proxies from its source VOD.

        Each clip comes back at the fidelity it had. ``preview_only`` already
        records whether a clip was still a cheap review proxy or had been
        upgraded to a final-quality render by an export, so honouring it means a
        rebuilt session costs what the original cost -- no more, and no silent
        downgrade of a clip the creator had already exported.
        """
        summary = job_media_summary(self.db, [job_id]).get(job_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if summary[REBUILDABLE] == 0 and summary[UNAVAILABLE] == 0:
            return {"status": "nothing_to_rebuild", "job_id": job_id, **summary}
        if summary[REBUILDABLE] == 0 or summary["source_state"] != "local":
            # The source itself has to come back first. The Cutting Room and
            # Storage settings already own that flow, so point at it instead of
            # quietly starting a download from a clip-level action.
            return {
                "status": "source_required",
                "job_id": job_id,
                "restorable": summary["source_state"] == "restorable",
                **summary,
            }

        with _REBUILD_LOCK:
            if job_id in _ACTIVE_REBUILDS:
                return {"status": "already_running", "job_id": job_id, **summary}
            state = {"status": "rebuilding", "job_id": job_id, "done": 0,
                     "total": summary[REBUILDABLE], "failed": 0}
            _ACTIVE_REBUILDS[job_id] = state

        def emit(event_type: str, message: str, progress: Optional[float] = None) -> None:
            if self.job_manager is None:
                return
            try:
                self.job_manager.emit_event(
                    job_id, event_type, phase="Rebuild clips",
                    message=message, progress=progress, payload={"job_id": job_id},
                )
            except Exception:
                logger.debug("Rebuild event emit failed for job %s", job_id, exc_info=True)

        def worker() -> None:
            try:
                emit("clip_rebuild_started",
                     f"Rebuilding {state['total']} clip previews.", 0.0)
                with self.db.get_connection() as conn:
                    rows = conn.execute(
                        "SELECT * FROM clips WHERE job_id = ? ORDER BY "
                        "COALESCE(deck_score, score, 0) DESC",
                        (job_id,),
                    ).fetchall()
                pending = [
                    dict(row) for row in rows
                    if not (row["export_path"] and os.path.isfile(row["export_path"]))
                ]
                for clip_row in pending:
                    try:
                        self._rebuild_missing_clip(clip_row["id"])
                        state["done"] += 1
                    except Exception:
                        # One unrenderable clip must not abandon the other
                        # fourteen. Its row stays rebuildable for a later retry.
                        state["failed"] += 1
                        logger.debug(
                            "Clip rebuild failed for %s", clip_row["id"], exc_info=True
                        )
                    finished = state["done"] + state["failed"]
                    emit(
                        "clip_rebuild_progress",
                        f"Rebuilt {state['done']} of {len(pending)} clips.",
                        finished / max(1, len(pending)),
                    )
                state["status"] = "complete"
                emit("clip_rebuild_complete",
                     f"Rebuilt {state['done']} clips."
                     + (f" {state['failed']} could not be rebuilt." if state["failed"] else ""),
                     1.0)
            except Exception:
                state["status"] = "failed"
                logger.exception("Clip media rebuild failed for job %s", job_id)
                emit("clip_rebuild_failed", "Clip rebuild could not finish.")
            finally:
                with _REBUILD_LOCK:
                    _ACTIVE_REBUILDS.pop(job_id, None)

        threading.Thread(
            target=worker, name=f"recall-clip-rebuild-{job_id[:8]}", daemon=True
        ).start()
        return dict(state)

    def list_exports(self) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM exports ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_batch(self, clip_ids: List[str]) -> Dict[str, int]:
        if not clip_ids:
            return {"deleted": 0}
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(clip_ids))
            cursor.execute(
                f"SELECT id, export_path, features FROM clips WHERE id IN ({placeholders})",  # nosec B608
                clip_ids,
            )
            rows = cursor.fetchall()
            for row in rows:
                cursor.execute(
                    "INSERT INTO clip_labels (clip_id, features, label, event) VALUES (?, ?, ?, ?)",
                    (row["id"], row["features"], 0, "deleted"),
                )
            cursor.execute(f"DELETE FROM exports WHERE clip_id IN ({placeholders})", clip_ids)  # nosec B608
            cursor.execute(f"DELETE FROM clips WHERE id IN ({placeholders})", clip_ids)  # nosec B608
        for row in rows:
            path = row["export_path"]
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    logger.debug("Could not delete clip export %s", path, exc_info=True)
        return {"deleted": len(rows)}

    def _resolve_source_video(self, source_path: str) -> str:
        """Resolve a job source (URL or local path) to a playable local video
        file. Raises HTTP 500 on failure."""
        from pipeline.ingest.vod_resolver import resolve_input
        try:
            return resolve_input(source_path, output_dir=os.path.join(self.data_root, "assets"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to resolve source video: {str(e)}")

    def _local_source_video(self, job_row: Any) -> str:
        """Return an already-local source and never start an implicit restore."""
        asset_path = job_row["asset_path"]
        if asset_path and os.path.isfile(asset_path):
            return os.path.abspath(asset_path)
        source_type = (job_row["source_type"] or "file") if "source_type" in job_row.keys() else "file"
        source_path = job_row["source_path"]
        if source_type == "file":
            # Local recordings may be on removable/external media. Resolve the
            # creator-selected path through the existing local-file validator,
            # but never place it under Recall management.
            return self._resolve_source_video(source_path)
        raise HTTPException(
            status_code=409,
            detail=(
                "Restore the Twitch source before editing or final-quality rendering. "
                "Your session, review clips, and decisions are still available."
            ),
        )

    def _layout_from_meta(self, clip_id: str, export_path: Optional[str] = None) -> Dict[str, Any]:
        """The clip's saved export layout from its ``*_meta.json`` sidecar,
        defaulting to vertical_split. Also checks the meta keyed on the export
        filename stem (older clips wrote it that way)."""
        layout = {"type": "vertical_split"}
        candidates = [os.path.join(self.exports_dir, f"clip_{clip_id}_meta.json")]
        if export_path:
            stem = os.path.splitext(os.path.basename(export_path))[0]
            candidates.append(os.path.join(self.exports_dir, f"{stem}_meta.json"))
        for meta_path in candidates:
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        return json.load(f).get("layout", layout)
                except Exception:
                    logger.debug("Could not read clip layout metadata %s", meta_path, exc_info=True)
                    continue
        # New scans persist layout directly with the clip so re-editing remains
        # deterministic even if an export sidecar was moved or cleaned up.
        try:
            with self.db.get_connection() as conn:
                row = conn.cursor().execute(
                    "SELECT layout FROM clips WHERE id = ?", (clip_id,),
                ).fetchone()
            if row and row["layout"]:
                saved = json.loads(row["layout"])
                if isinstance(saved, dict):
                    return saved
        except Exception:
            logger.debug("Could not read persisted clip layout %s", clip_id, exc_info=True)
        return layout

    def _render_and_thumb(
        self,
        video_path: str,
        clip,
        *,
        peak_timestamp=None,
        ass_path=None,
        preview: bool = False,
        render_progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        **fades,
    ):
        """Render a clip to its final MP4 and grab a poster JPEG at the reaction
        peak. Returns (rendered_path, thumb_path). ``render_clip`` may raise; the
        poster is best-effort and comes back None on failure."""
        from core.thumbnails import clip_relative_peak, generate_thumbnail
        from engines.export.renderer import render_clip
        rendered_path = render_clip(
            video_path=video_path, clip=clip, output_dir=self.exports_dir,
            ass_path=ass_path,
            preview=preview,
            progress_callback=render_progress_callback,
            cancel_check=cancel_check,
            **fades,
        )
        thumb_path = None
        try:
            seek = clip_relative_peak(peak_timestamp, clip.start, clip.end)
            thumb_path = generate_thumbnail(rendered_path, clip.clip_id, self.exports_dir, seek_seconds=seek)
        except Exception:
            logger.debug("Could not generate thumbnail for rendered clip", exc_info=True)
            thumb_path = None
        return rendered_path, thumb_path

    def _render_existing_clip(
        self,
        clip_row,
        start: float,
        end: float,
        *,
        render_progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        **fades,
    ):
        """Resolve, reconstruct, and render an existing clip through one path."""
        from core.models.clip import GameClip

        with self.db.get_connection() as conn:
            job_row = conn.cursor().execute(
                "SELECT source_path, asset_path, source_type, duration FROM jobs WHERE id = ?",
                (clip_row["job_id"],),
            ).fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail="Job associated with clip not found")
        vod_duration = float(job_row["duration"] or 0.0)
        if vod_duration > 0 and end > vod_duration + 0.05:
            raise HTTPException(status_code=422, detail="Clip end time exceeds the source duration.")

        with source_in_use(self.db, clip_row["job_id"]):
            video_path = self._local_source_video(job_row)
            layout = self._layout_from_meta(clip_row["id"], clip_row["export_path"])
            ass_path = self._regenerate_ass_if_possible(video_path, clip_row["id"], start, end)
            clip = GameClip(
                clip_id=clip_row["id"],
                start=start,
                end=end,
                story_id=clip_row["story_label"] or "",
                score=clip_row["score"] or 0.8,
                layout=layout,
            )
            return self._render_and_thumb(
                video_path,
                clip,
                peak_timestamp=clip_row["peak_timestamp"],
                ass_path=ass_path,
                render_progress_callback=render_progress_callback,
                cancel_check=cancel_check,
                **fades,
            )

    @serialized_clip_operation
    def ensure_rendered(
        self,
        clip_id: str,
        progress_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Optional[str]:
        """Render a clip's final-quality MP4 if it's still just a review preview.

        Clips land in the library with a cheap low-res preview only (see
        pipeline/orchestrator/run_pipeline.py, ``export_clips(..., preview=True)``)
        -- the full-quality ffmpeg render now happens lazily, the first time a
        clip is actually exported (copy-to-folder, compile-reel) or edited,
        rather than for every candidate clip during the VOD scan. Returns the
        export_path, or None if rendering failed.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clips WHERE id = ?", (clip_id,))
            clip_row = cursor.fetchone()
            if not clip_row:
                return None
            already_final = (
                clip_row["export_path"]
                and os.path.exists(clip_row["export_path"])
                and not clip_row["preview_only"]
            )
            if already_final:
                return clip_row["export_path"]
        try:
            from engines.export.renderer import FFmpegCancelled

            rendered_path, new_thumb = self._render_existing_clip(
                clip_row,
                clip_row["start_time"],
                clip_row["end_time"],
                render_progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
        except FFmpegCancelled:
            raise
        except HTTPException:
            # Missing Twitch sources are a user-recoverable 409, not a silent
            # render miss. The caller can now show the explicit restore action.
            raise
        except Exception:
            logger.debug("Lazy full-quality render failed for clip %s", clip_id, exc_info=True)
            return None
        thumb_path = new_thumb or clip_row["thumb_path"]

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE clips SET export_path = ?, thumb_path = ?, preview_only = 0 WHERE id = ?",
                (rendered_path, thumb_path, clip_id),
            )
        return rendered_path

    def _persist_copied_exports(
        self,
        copied_records: List[tuple],
        platform: str,
    ) -> None:
        """Persist successful copies without reopening connections per clip."""
        if not copied_records:
            return
        self.db.record_labels_batch(
            [clip_id for clip_id, _ in copied_records],
            label=1,
            event="exported",
        )
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for clip_id, src in copied_records:
                cursor.execute(
                    "UPDATE clips SET exported_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (clip_id,),
                )
                existing = cursor.execute(
                    "SELECT id FROM exports WHERE clip_id = ?",
                    (clip_id,),
                ).fetchone()
                if existing:
                    cursor.execute(
                        "UPDATE exports SET platform = ?, file_path = ? WHERE clip_id = ?",
                        (platform, src, clip_id),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO exports (id, clip_id, platform, file_path) VALUES (?, ?, ?, ?)",
                        (f"export_{uuid.uuid4().hex[:12]}", clip_id, platform, src),
                    )

    def _source_date_for_export(self, row: Any) -> Optional[str]:
        """Resolve and persist a clip's recording date, including legacy jobs."""
        from core.source_date import local_recording_date, normalize_source_date

        stored = normalize_source_date(row["source_date"])
        if stored:
            return stored

        source_path = row["asset_path"] or row["source_path"]
        source_type = str(row["source_type"] or "file").lower()
        is_remote = str(row["source_path"] or "").lower().startswith(("http://", "https://"))
        resolved = None
        if source_path and source_type not in {"url", "twitch"} and not is_remote:
            resolved = local_recording_date(source_path)
        elif row["source_path"] and is_remote:
            # New scans persist this during the import probe. This branch is a
            # one-time best-effort backfill for Twitch jobs created before the
            # source_date column existed.
            try:
                from engines.chat.twitch_chat import fetch_twitch_video_meta

                metadata = fetch_twitch_video_meta(
                    row["source_path"],
                    timeout=15.0,
                    temp_dir=os.path.join(self.data_root, "temp"),
                ) or {}
                resolved = normalize_source_date(metadata.get("created_at"))
            except Exception:
                logger.debug(
                    "Could not recover Twitch source date for job %s",
                    row["job_id"],
                    exc_info=True,
                )

        if resolved and row["job_id"]:
            with self.db.get_connection() as conn:
                conn.cursor().execute(
                    "UPDATE jobs SET source_date = ? WHERE id = ?",
                    (resolved, row["job_id"]),
                )
        return resolved

    def copy_to_folder(
        self,
        clip_ids: List[str],
        dest_folder: str,
        preset: str = None,
        filename_template: str = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Copy rendered clips to a folder, applying an export preset (plan 9.5).

        Every clip is already a 1080x1920 vertical MP4 within every platform's
        length limit, so nothing is re-encoded or gated here. The preset just
        tags the export row's platform and feeds the ``{platform}`` filename
        token; ``filename_template`` renames files from a small token set.
        ``{index}`` is the clip's persistent per-VOD number, not its temporary
        position in this export request.
        """
        from core.export_presets import ensure_unique, render_filename
        from engines.export.renderer import FFmpegCancelled

        os.makedirs(dest_folder, exist_ok=True)
        copied, errors = 0, 0
        succeeded_clip_ids: List[str] = []
        failed: List[Dict[str, str]] = []
        total_clips = len(clip_ids)
        reporter = _OperationProgressReporter(
            progress_callback,
            kind="clips",
            total_clips=total_clips,
            snapshot=lambda: {
                "copied": copied,
                "errors": errors,
                "succeeded_clip_ids": list(succeeded_clip_ids),
                "failed": list(failed),
            },
        )
        report = reporter.report

        if not clip_ids:
            return {
                "copied": copied,
                "errors": errors,
                "succeeded_clip_ids": succeeded_clip_ids,
                "failed": failed,
                "cancelled": False,
            }

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(clip_ids))
            cursor.execute(
                f"""SELECT c.id, c.export_path, c.title, c.preview_only,
                            c.start_time, c.end_time, c.job_id,
                            j.source_date, j.source_type, j.source_path, j.asset_path
                     FROM clips AS c
                     LEFT JOIN jobs AS j ON j.id = c.job_id
                     WHERE c.id IN ({placeholders})""",  # nosec B608
                clip_ids,
            )
            by_id = {row["id"]: row for row in cursor.fetchall()}

            from core.clip_number import ensure_clip_numbers

            export_job_ids = list(dict.fromkeys(
                row["job_id"] for row in by_id.values() if row["job_id"]
            ))
            ensure_clip_numbers(conn, export_job_ids)
            number_rows = conn.execute(
                f"""SELECT id, job_id, clip_number
                     FROM clips
                     WHERE job_id IN ({','.join('?' * len(export_job_ids))})""",  # nosec B608
                export_job_ids,
            ).fetchall() if export_job_ids else []
            clip_numbers = {
                row["id"]: int(row["clip_number"])
                for row in number_rows
                if row["clip_number"] is not None
            }
            job_number_totals: Dict[str, int] = {}
            for number_row in number_rows:
                if number_row["clip_number"] is None:
                    continue
                number_job_id = str(number_row["job_id"])
                job_number_totals[number_job_id] = max(
                    job_number_totals.get(number_job_id, 0),
                    int(number_row["clip_number"]),
                )

        duration_by_id = {
            clip_id: max(
                0.1,
                float(row["end_time"] or 0.0) - float(row["start_time"] or 0.0),
            )
            for clip_id, row in by_id.items()
        }
        total_work = max(0.1, sum(duration_by_id.get(clip_id, 1.0) for clip_id in clip_ids))
        completed_work = 0.0
        cancelled = False

        platform = (preset or "manual").lower()
        taken: set = set()
        source_dates: Dict[str, Optional[str]] = {}
        copied_records: List[tuple] = []  # (clip_id, src) for clips actually copied
        # Preserve caller order for progress/copying. Filenames use the clip's
        # persistent per-VOD number, independent of this export selection.
        for position, clip_id in enumerate(clip_ids, start=1):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            row = by_id.get(clip_id)
            clip_duration = duration_by_id.get(clip_id, 1.0)
            src = row["export_path"] if row else None
            needs_render = bool(
                row and (not src or not os.path.exists(src) or row["preview_only"])
            )
            starting_progress = completed_work / total_work
            report(
                status="running",
                phase="rendering" if needs_render else "copying",
                current_clip_id=clip_id,
                current_index=position,
                completed_clips=position - 1,
                current_clip_progress=None,
                current_clip_duration=clip_duration,
                rendered_seconds=None,
                progress=starting_progress,
                message=(
                    f"Rendering clip {position} of {total_clips}"
                    if needs_render else f"Copying clip {position} of {total_clips}"
                ),
            )
            # Clips in the library only have a low-res review preview until
            # first exported -- render the final-quality file now.
            if needs_render:
                def on_render_progress(
                    render_state: Dict[str, float],
                    *,
                    _clip_id: str = clip_id,
                    _position: int = position,
                    _clip_duration: float = clip_duration,
                ) -> None:
                    current_fraction = max(
                        0.0,
                        min(1.0, float(render_state.get("render_progress") or 0.0)),
                    )
                    aggregate_progress = (
                        completed_work + (_clip_duration * current_fraction)
                    ) / total_work
                    report(
                        status="running",
                        phase="rendering",
                        current_clip_id=_clip_id,
                        current_index=_position,
                        completed_clips=_position - 1,
                        current_clip_progress=current_fraction,
                        current_clip_duration=float(
                            render_state.get("render_duration_seconds") or _clip_duration
                        ),
                        rendered_seconds=float(render_state.get("rendered_seconds") or 0.0),
                        progress=aggregate_progress,
                        message=f"Rendering clip {_position} of {total_clips}",
                    )

                try:
                    src = self.ensure_rendered(
                        clip_id,
                        progress_callback=on_render_progress,
                        cancel_check=cancel_check,
                    )
                except FFmpegCancelled:
                    cancelled = True
                    break
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            if not src or not os.path.exists(src):
                errors += 1
                failed.append({
                    "clip_id": clip_id,
                    "reason": (
                        "Clip record was not found."
                        if row is None
                        else "Recall could not render the final-quality file."
                    ),
                })
                completed_work += clip_duration
                completed_progress = completed_work / total_work
                report(
                    status="running",
                    phase="failed_clip",
                    current_clip_id=clip_id,
                    current_index=position,
                    completed_clips=position,
                    current_clip_progress=1.0,
                    current_clip_duration=clip_duration,
                    progress=completed_progress,
                    message=f"Clip {position} of {total_clips} could not be exported",
                )
                continue
            job_id = str(row["job_id"] or "")
            if job_id not in source_dates:
                source_dates[job_id] = self._source_date_for_export(row)
            stable_number = clip_numbers.get(clip_id, position)
            stem = render_filename(
                filename_template,
                title=row["title"] or "",
                index=stable_number,
                preset=preset,
                original_stem=os.path.splitext(os.path.basename(src))[0],
                source_date=source_dates[job_id],
                total=job_number_totals.get(job_id, stable_number),
            )
            dest_name = ensure_unique(dest_folder, stem, taken)
            try:
                shutil.copy2(src, os.path.join(dest_folder, dest_name))
            except Exception:
                errors += 1
                failed.append({
                    "clip_id": clip_id,
                    "reason": "Recall could not copy the final file to the selected folder.",
                })
                completed_work += clip_duration
                completed_progress = completed_work / total_work
                report(
                    status="running",
                    phase="failed_clip",
                    current_clip_id=clip_id,
                    current_index=position,
                    completed_clips=position,
                    current_clip_progress=1.0,
                    current_clip_duration=clip_duration,
                    progress=completed_progress,
                    message=f"Clip {position} of {total_clips} could not be copied",
                )
                continue
            copied += 1
            copied_records.append((clip_id, src))
            succeeded_clip_ids.append(clip_id)
            completed_work += clip_duration
            completed_progress = completed_work / total_work
            report(
                status="running",
                phase="copied",
                current_clip_id=clip_id,
                current_index=position,
                completed_clips=position,
                current_clip_progress=1.0,
                current_clip_duration=clip_duration,
                progress=completed_progress,
                message=f"Exported clip {position} of {total_clips}",
            )

        # Lazy renders may not have an exports row yet, so successful copies
        # need an upsert as well as the exported review label.
        self._persist_copied_exports(copied_records, platform)
        return {
            "copied": copied,
            "errors": errors,
            "succeeded_clip_ids": succeeded_clip_ids,
            "failed": failed,
            "cancelled": cancelled,
        }

    def _prepare_reel_sources(
        self,
        entries: List[tuple],
        durations: Dict[str, float],
        total_work: float,
        failed: List[Dict[str, str]],
        report: Callable[..., None],
        cancel_check: Optional[Callable[[], bool]],
    ) -> tuple[List[Dict[str, Any]], float]:
        """Resolve final-quality reel sources and report their weighted work."""
        from engines.export.renderer import FFmpegCancelled

        usable: List[Dict[str, Any]] = []
        completed_work = 0.0
        completed_clips = 0
        total_clips = len(entries)
        for position, (clip_id, row) in enumerate(entries, start=1):
            if cancel_check is not None and cancel_check():
                raise FFmpegCancelled("Reel preparation cancelled")
            if row is None:
                failed.append({
                    "clip_id": clip_id,
                    "reason": "Clip record was not found.",
                })
                completed_clips += 1
                progress = completed_work / total_work
                report(
                    status="running",
                    phase="failed_clip",
                    current_clip_id=clip_id,
                    current_index=position,
                    completed_clips=completed_clips,
                    current_clip_progress=1.0,
                    reel_progress=None,
                    progress=progress,
                    message=f"Clip {position} of {total_clips} is unavailable",
                )
                continue

            clip_duration = durations[clip_id]
            needs_render = bool(
                not row["export_path"]
                or not os.path.exists(row["export_path"])
                or row["preview_only"]
            )
            starting_progress = completed_work / total_work
            report(
                status="running",
                phase="rendering" if needs_render else "preparing_reel",
                current_clip_id=clip_id,
                current_index=position,
                completed_clips=completed_clips,
                current_clip_progress=None,
                current_clip_duration=clip_duration,
                reel_progress=None,
                progress=starting_progress,
                message=(
                    f"Rendering clip {position} of {total_clips}"
                    if needs_render
                    else f"Preparing clip {position} of {total_clips}"
                ),
            )
            if needs_render:
                def on_render_progress(
                    render_state: Dict[str, float],
                    *,
                    _clip_id: str = clip_id,
                    _position: int = position,
                    _clip_duration: float = clip_duration,
                ) -> None:
                    current_fraction = max(
                        0.0,
                        min(1.0, float(render_state.get("render_progress") or 0.0)),
                    )
                    aggregate_progress = (
                        completed_work + (_clip_duration * current_fraction)
                    ) / total_work
                    report(
                        status="running",
                        phase="rendering",
                        current_clip_id=_clip_id,
                        current_index=_position,
                        completed_clips=completed_clips,
                        current_clip_progress=current_fraction,
                        current_clip_duration=_clip_duration,
                        rendered_seconds=float(render_state.get("rendered_seconds") or 0.0),
                        reel_progress=None,
                        progress=aggregate_progress,
                        message=f"Rendering clip {_position} of {total_clips}",
                    )

                row["export_path"] = self.ensure_rendered(
                    clip_id,
                    progress_callback=on_render_progress,
                    cancel_check=cancel_check,
                )
                completed_work += clip_duration

            completed_clips += 1
            if row["export_path"] and os.path.exists(row["export_path"]):
                usable.append(row)
            else:
                failed.append({
                    "clip_id": clip_id,
                    "reason": "Recall could not render the final-quality file.",
                })
            progress = completed_work / total_work
            report(
                status="running",
                phase="preparing_reel",
                current_clip_id=clip_id,
                current_index=position,
                completed_clips=completed_clips,
                current_clip_progress=1.0,
                current_clip_duration=clip_duration,
                reel_progress=None,
                progress=progress,
                message=f"Prepared clip {position} of {total_clips}",
            )

        usable.sort(key=lambda row: row["start_time"] or 0.0)
        return usable, completed_work

    @staticmethod
    def _run_reel_encode_command(
        cmd: List[str],
        *,
        timeout_seconds: int,
        reel_duration: float,
        requested_count: int,
        failed: List[Dict[str, str]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        on_reel_progress: Callable[[Dict[str, float]], None],
        cancel_check: Optional[Callable[[], bool]],
    ) -> Optional[Dict[str, Any]]:
        """Run the reel command and normalize recoverable FFmpeg failures."""
        import subprocess
        from engines.export.renderer import _run_with_progress

        try:
            _run_with_progress(
                cmd,
                duration=reel_duration,
                background=False,
                progress_callback=on_reel_progress,
                timeout_seconds=timeout_seconds,
                cancel_check=cancel_check,
            )
        except subprocess.TimeoutExpired:
            return {
                "error": (
                    f"Reel render timed out after {timeout_seconds} seconds. "
                    "Try fewer or shorter clips, then retry."
                ),
                "errors": requested_count,
                "failed": failed,
                "succeeded_clip_ids": [],
            }
        except subprocess.CalledProcessError as exc:
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr or "")
            )
            return {
                "error": f"Reel render failed: {stderr.strip()[:300]}",
                "errors": requested_count,
                "failed": failed,
                "succeeded_clip_ids": [],
            }
        return None

    def _encode_reel_file(
        self,
        *,
        usable: List[Dict[str, Any]],
        durations: Dict[str, float],
        anticipated_reel_work: float,
        total_work: float,
        completed_work: float,
        requested_count: int,
        dest_folder: str,
        filename: Optional[str],
        failed: List[Dict[str, str]],
        report: Callable[..., None],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        cancel_check: Optional[Callable[[], bool]],
        mute_clip_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Encode prepared reel sources into one atomically promoted MP4."""
        import subprocess
        import tempfile
        from core.export_presets import sanitize_filename
        from core.ffmpeg_path import get_ffmpeg_path
        from engines.export.ffmpeg_builder import check_nvenc_available, cpu_h264_vcodec
        from engines.export.renderer import FFmpegCancelled, _run_with_progress

        os.makedirs(dest_folder, exist_ok=True)
        default_stem = f"recall_reel_{time.strftime('%Y%m%d_%H%M%S')}"
        stem = sanitize_filename(filename or default_stem, default_stem)
        out_path = os.path.join(dest_folder, f"{stem}.mp4")
        partial_path = os.path.join(
            dest_folder,
            f".{stem}.{uuid.uuid4().hex}.partial.mp4",
        )
        succeeded_clip_ids = [row["id"] for row in usable]
        reel_duration = sum(durations[row["id"]] for row in usable)
        completed_work += max(0.0, anticipated_reel_work - reel_duration)
        timeout_seconds = self._reel_timeout_seconds(reel_duration)

        # A muted clip is silenced into a throwaway copy rather than re-cut:
        # the video is stream-copied, so only its audio is touched, and the
        # canonical clip on disk keeps its sound for every other use.
        muted_ids = set(mute_clip_ids or ())
        silenced: Dict[str, str] = {}
        list_path = None
        treated_paths: List[str] = []
        try:
            if muted_ids:
                for index, row in enumerate(usable):
                    is_muted = row["id"] in muted_ids
                    # Sound arriving at full level straight out of silence lands as
                    # a pop against a track. The first audible clip after a muted
                    # one fades up instead -- which is exactly the moment the reel
                    # is built around, so it should not arrive as a glitch.
                    follows_silence = (
                        not is_muted
                        and index > 0
                        and usable[index - 1]["id"] in muted_ids
                    )
                    if not is_muted and not follows_silence:
                        continue
                    audio_filter = (
                        "volume=0" if is_muted
                        else f"afade=t=in:st=0:d={REEL_FADE_IN_SECONDS}"
                    )
                    treated = os.path.join(
                        dest_folder, f".{uuid.uuid4().hex}.audio.mp4",
                    )
                    treated_paths.append(treated)
                    if cancel_check is not None and cancel_check():
                        raise FFmpegCancelled("Reel compilation cancelled")
                    try:
                        _run_with_progress(
                            [
                                get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                                "-i", row["export_path"], "-map_metadata", "-1",
                                "-map_metadata:s", "-1", "-map_chapters", "-1", "-c:v", "copy",
                                "-af", audio_filter, "-c:a", "aac", "-b:a", "128k", treated,
                            ],
                            duration=durations[row["id"]], background=False,
                            progress_callback=lambda _state: None,
                            timeout_seconds=300, cancel_check=cancel_check,
                        )
                        if not os.path.isfile(treated) or os.path.getsize(treated) == 0:
                            raise OSError("Audio treatment produced no media")
                        if cancel_check is not None and cancel_check():
                            raise FFmpegCancelled("Reel compilation cancelled")
                        silenced[row["id"]] = treated
                    except (subprocess.SubprocessError, OSError):
                        return {
                            "error": "Recall could not apply the requested reel audio edits. No reel was published.",
                            "errors": requested_count,
                            "failed": failed,
                            "succeeded_clip_ids": [],
                        }

            list_fd, list_path = tempfile.mkstemp(suffix=".txt", text=True)
            with os.fdopen(list_fd, "w", encoding="utf-8") as handle:
                for row in usable:
                    source = silenced.get(row["id"], row["export_path"])
                    escaped = source.replace("\\", "/").replace("'", r"'\''")
                    handle.write(f"file '{escaped}'\n")

            if check_nvenc_available():
                vcodec = ["-c:v", "h264_nvenc", "-preset", "fast", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
            else:
                vcodec = cpu_h264_vcodec("reel")
            cmd = [
                get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", list_path,
                "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
            ] + vcodec + ["-c:a", "aac", "-b:a", "128k", partial_path]
            starting_progress = completed_work / total_work
            report(
                status="running",
                phase="compiling_reel",
                current_clip_id=None,
                current_index=requested_count,
                completed_clips=requested_count,
                current_clip_progress=None,
                current_clip_duration=None,
                reel_progress=None,
                reel_duration=reel_duration,
                rendered_seconds=None,
                progress=starting_progress,
                message=f"Compiling {len(usable)}-clip reel",
            )

            def on_reel_progress(render_state: Dict[str, float]) -> None:
                reel_fraction = max(
                    0.0,
                    min(1.0, float(render_state.get("render_progress") or 0.0)),
                )
                aggregate_progress = (
                    completed_work + (reel_duration * reel_fraction)
                ) / total_work
                report(
                    status="running",
                    phase="compiling_reel",
                    current_clip_id=None,
                    current_index=requested_count,
                    completed_clips=requested_count,
                    current_clip_progress=None,
                    reel_progress=reel_fraction,
                    reel_duration=reel_duration,
                    rendered_seconds=float(render_state.get("rendered_seconds") or 0.0),
                    progress=aggregate_progress,
                    message=f"Compiling {len(usable)}-clip reel",
                )

            error = self._run_reel_encode_command(
                cmd,
                timeout_seconds=timeout_seconds,
                reel_duration=reel_duration,
                requested_count=requested_count,
                failed=failed,
                progress_callback=progress_callback,
                on_reel_progress=on_reel_progress,
                cancel_check=cancel_check,
            )
            if error is not None:
                return error
            if not os.path.isfile(partial_path) or os.path.getsize(partial_path) == 0:
                return {
                    "error": "Reel render failed before the final file was created.",
                    "errors": requested_count,
                    "failed": failed,
                    "succeeded_clip_ids": [],
                }
            if cancel_check is not None and cancel_check():
                raise FFmpegCancelled("Reel compilation cancelled")
            os.replace(partial_path, out_path)
        finally:
            try:
                if list_path is not None:
                    os.remove(list_path)
            except OSError:
                logger.debug("Could not delete reel concat list %s", list_path, exc_info=True)
            try:
                if os.path.exists(partial_path):
                    os.remove(partial_path)
            except OSError:
                logger.debug("Could not delete partial reel %s", partial_path, exc_info=True)
            for treated in treated_paths:
                try:
                    if os.path.exists(treated):
                        os.remove(treated)
                except OSError:
                    logger.debug("Could not delete treated clip %s", treated, exc_info=True)

        self.update_state_batch(succeeded_clip_ids, exported=True)
        return {
            "path": out_path,
            "clips": len(usable),
            "copied": len(usable),
            "errors": len(failed),
            "succeeded_clip_ids": succeeded_clip_ids,
            "failed": failed,
        }


    def compile_reel(
        self,
        clip_ids: List[str],
        dest_folder: str,
        filename: str = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        mute_clip_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Concatenate rendered clips into one highlight reel (plan 22 §3.2).

        Clips play in VOD-timeline order (a reel tells the session's story
        start to finish, regardless of review order). Every rendered clip is
        already a 1080x1920 vertical MP4, but encoder settings can differ
        (NVENC vs x264 fallback mid-batch), so the concat re-encodes once for
        guaranteed clean joins — hard cuts, no transitions (crossfades are a
        follow-up once the xfade chain is worth its complexity).
        """
        if len(clip_ids) < 2:
            return {"error": "A reel needs at least 2 kept clips."}

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(clip_ids))
            cursor.execute(
                f"SELECT id, export_path, start_time, end_time, preview_only FROM clips WHERE id IN ({placeholders})",  # nosec B608
                clip_ids,
            )
            by_id = {row["id"]: dict(row) for row in cursor.fetchall()}

        entries = [(clip_id, dict(by_id[clip_id]) if clip_id in by_id else None) for clip_id in clip_ids]
        durations = {
            clip_id: max(
                0.1,
                float(row["end_time"] or 0.0) - float(row["start_time"] or 0.0),
            )
            for clip_id, row in entries
            if row is not None
        }
        anticipated_reel_work = sum(durations.values())
        render_work = sum(
            durations[clip_id]
            for clip_id, row in entries
            if row is not None and (
                not row["export_path"]
                or not os.path.exists(row["export_path"])
                or row["preview_only"]
            )
        )
        total_work = max(0.1, anticipated_reel_work + render_work)
        failed: List[Dict[str, str]] = []
        reporter = _OperationProgressReporter(
            progress_callback,
            kind="reel",
            total_clips=len(clip_ids),
            snapshot=lambda: {
                "errors": len(failed),
                "failed": list(failed),
            },
        )
        report = reporter.report
        usable, completed_work = self._prepare_reel_sources(
            entries,
            durations,
            total_work,
            failed,
            report,
            cancel_check,
        )
        if len(usable) < 2:
            return {
                "error": "Fewer than 2 of those clips have rendered files.",
                "errors": len(failed),
                "failed": failed,
                "succeeded_clip_ids": [],
            }

        return self._encode_reel_file(
            mute_clip_ids=mute_clip_ids,
            usable=usable,
            durations=durations,
            anticipated_reel_work=anticipated_reel_work,
            total_work=total_work,
            completed_work=completed_work,
            requested_count=len(clip_ids),
            dest_folder=dest_folder,
            filename=filename,
            failed=failed,
            report=report,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def update_state_batch(
        self, clip_ids: List[str], kept=None, passed=None, maybe=None,
        exported=None,
    ) -> Dict[str, int]:
        """Bulk review-state update — used to migrate pre-3.2 localStorage
        keeps into the DB in one call per session."""
        if not clip_ids:
            return {"updated": 0}
        verdict_updates = {
            "kept": kept,
            "passed": passed,
            "maybe": maybe,
        }
        active_verdicts = sum(value is True for value in (kept, passed, maybe))
        if active_verdicts > 1:
            raise HTTPException(status_code=422, detail="A clip can have only one review verdict")
        if kept is True:
            verdict_updates.update(passed=False, maybe=False)
        elif passed is True:
            verdict_updates.update(kept=False, maybe=False)
        elif maybe is True:
            verdict_updates.update(kept=False, passed=False)
        assignments = []
        values: List[Any] = []
        for column, value in verdict_updates.items():
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(1 if value else 0)
        if exported is not None:
            assignments.append("exported_at = " + ("CURRENT_TIMESTAMP" if exported else "NULL"))
        if not assignments:
            return {"updated": 0}
        placeholders = ",".join("?" * len(clip_ids))
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE clips SET {', '.join(assignments)} WHERE id IN ({placeholders})",  # nosec B608
                (*values, *clip_ids),
            )
            updated = cursor.rowcount
        return {"updated": updated}

    def label(self, clip_id: str, request: Any) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM clips WHERE id = ?", (clip_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Clip not found")
        if request.label is None:
            self.db.retract_label(clip_id, event=request.event)
            return {"clip_id": clip_id, "label": None, "event": request.event}
        label = float(request.label)
        if label not in (0.0, 0.5, 1.0):
            raise HTTPException(
                status_code=422,
                detail="Label must be 0 (Pass), 0.5 (Maybe), or 1 (Keep)",
            )
        self.db.record_label(clip_id, label=label, event=request.event)
        return {"clip_id": clip_id, "label": label, "event": request.event}

    def list_clips(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if job_id:
                cursor.execute(
                    "SELECT * FROM clips WHERE job_id = ? AND COALESCE(origin, '') <> 'compilation'",
                    (job_id,),
                )
            else:
                cursor.execute("SELECT * FROM clips WHERE COALESCE(origin, '') <> 'compilation'")
            rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            provenance = row.get("recall_provenance")
            if provenance:
                try:
                    row["recall_provenance"] = json.loads(provenance)
                except (TypeError, ValueError):
                    row["recall_provenance"] = None
        # export_path being set is not proof the file survived -- a creator can
        # reclaim the exports folder at any time. Tell the frontend what is
        # actually on disk so it stops requesting posters that 404.
        return annotate_clip_rows(self.db, rows)

    @staticmethod
    def _more_candidate_id(job_id: str, candidate: Dict[str, Any]) -> str:
        """Stable id for a ceiling-cut candidate across scan persist and reveal."""
        start = float(candidate.get("start", 0.0) or 0.0)
        end = float(candidate.get("end", 0.0) or 0.0)
        peak = float(candidate.get("peak_timestamp", start) or start)
        return more_candidate_clip_id(job_id, start, end, peak)

    def _more_candidate_rows(
        self, job_id: str, include_floor: bool = False,
    ) -> List[Dict[str, Any]]:
        """Ceiling candidates for this job.

        By default this is the REVIEW lane: the arousal floor is held back (see
        ``partition_second_look``). ``include_floor=True`` returns the deferred
        rows instead, for the training lane that still wants them.
        """
        with self.db.get_connection() as conn:
            job = conn.cursor().execute(
                "SELECT id FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        path = os.path.join(self.data_root, "jobs", job_id, CANDIDATES_FILENAME)
        if not os.path.isfile(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError, TypeError):
            logger.warning("Could not read candidate artifact for job %s", job_id, exc_info=True)
            return []

        # Full list for status totals; materialize/status slice to the batch.
        candidates = ceiling_candidates_from_trace(
            payload.get("candidates", []) if isinstance(payload, dict) else [],
            job_id=job_id,
            limit=None,
        )
        review, deferred = partition_second_look(candidates)
        return deferred if include_floor else review

    def deferred_more_candidates(self, job_id: str) -> List[Dict[str, Any]]:
        """Floor-vetoed ceiling candidates, kept addressable for training.

        These never reach Theater, but they are complete candidates with full
        features — scripts/build_boundary_batch.py and any later off-policy
        label pass can still draw on them.
        """
        return self._more_candidate_rows(job_id, include_floor=True)

    def more_candidate_status(self, job_id: str, limit: int = MORE_CANDIDATE_LIMIT) -> Dict[str, int]:
        candidates = self._more_candidate_rows(job_id)
        limit = max(1, min(int(limit or self.MORE_CANDIDATE_LIMIT), 30))
        batch = candidates[:limit]
        ids = [candidate["id"] for candidate in batch]
        loaded = 0
        if ids:
            placeholders = ",".join("?" * len(ids))
            with self.db.get_connection() as conn:
                loaded = conn.cursor().execute(
                    f"SELECT COUNT(*) FROM clips WHERE id IN ({placeholders})",  # nosec B608
                    ids,
                ).fetchone()[0]
        return {
            "available": len(batch),
            "available_total": len(candidates),
            "loaded": int(loaded),
            "held_back": len(self._more_candidate_rows(job_id, include_floor=True)),
        }

    def materialize_more_candidates(
        self,
        job_id: str,
        limit: int = MORE_CANDIDATE_LIMIT,
    ) -> Dict[str, Any]:
        """Reveal ceiling-cut moments into the review deck.

        New scans already persist and preview-render this tier during export.
        This path is idempotent for those rows and only backfills framing /
        cheap vertical proxies for older jobs that never got scan-time renders.
        """
        candidates = self._more_candidate_rows(job_id)
        limit = max(1, min(int(limit or self.MORE_CANDIDATE_LIMIT), 30))
        batch = candidates[:limit]
        if not batch:
            return {"available": 0, "available_total": len(candidates), "loaded": 0, "clips": []}

        # Ceiling-cut candidates that skipped the scan export path still need
        # scene-aware framing before a preview can be rendered.
        framing_job = self._manual_job_row(job_id)
        layouts = {
            candidate["id"]: self.resolve_manual_framing(
                job_id, candidate["start"], candidate["end"], requested_layout="auto",
                _job_row=framing_job,
            )
            for candidate in batch
        }

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            from core.clip_number import ensure_clip_numbers, next_clip_number

            ensure_clip_numbers(conn, [job_id])
            existing_numbers = {
                row["id"]: int(row["clip_number"])
                for row in conn.execute(
                    "SELECT id, clip_number FROM clips WHERE job_id = ?",
                    (job_id,),
                ).fetchall()
                if row["clip_number"] is not None
            }
            next_number = next_clip_number(conn, job_id)
            for index, candidate in enumerate(batch):
                features = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
                breakdown = (
                    candidate.get("modality_breakdown")
                    if isinstance(candidate.get("modality_breakdown"), dict)
                    else {}
                )
                strongest = sorted(
                    (
                        (str(name), float(value))
                        for name, value in breakdown.items()
                        if isinstance(value, (int, float)) and float(value) > 0
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
                signals = [name for name, _value in strongest]
                hook_score = max(0.0, min(1.0, float(features.get("hook_score", 0.0) or 0.0)))
                if hook_score <= 0:
                    hook_score = max(0.0, min(1.0, float(features.get("peak_value", 0.0) or 0.0)))
                tier_score = second_look_deck_score(index)
                clip_number = existing_numbers.get(candidate["id"])
                if clip_number is None:
                    clip_number = next_number
                    next_number += 1
                cursor.execute(
                    """INSERT OR IGNORE INTO clips
                       (id, job_id, clip_number, start_time, end_time, score, selection_score,
                        hook_score, deck_score, title,
                        description, signals, story_label, export_path, features,
                        reason, peak_timestamp, modality_breakdown, reaction_auc,
                        thumb_path, scene, tags, preview_only, hook_line, moment_type,
                        layout, semantic_verdict)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        candidate["id"], job_id, clip_number,
                        candidate["start"], candidate["end"],
                        hook_score,
                        float(candidate.get("selection_score", 0.0) or 0.0),
                        hook_score, tier_score, "Second-look moment",
                        "A lower-ranked moment Recall saved just outside the first review deck.",
                        json.dumps(signals), self.MORE_CANDIDATE_LABEL, None,
                        json.dumps(features) if features else None,
                        "This moment was just outside the first 15 and stayed available for a second look.",
                        candidate.get("peak_timestamp"),
                        json.dumps(breakdown) if breakdown else None,
                        features.get("reaction_auc"), None, candidate.get("scene_label"),
                        None, 1, None, None, json.dumps(layouts[candidate["id"]]),
                        candidate.get("semantic_verdict"),
                    ),
                )
                # Repair Second-look rows created by older builds. Stable IDs
                # make reveals idempotent, so INSERT OR IGNORE alone would
                # otherwise leave their missing framing in place forever.
                cursor.execute(
                    "UPDATE clips SET layout = ? WHERE id = ? AND layout IS NULL",
                    (json.dumps(layouts[candidate["id"]]), candidate["id"]),
                )

            ids = [candidate["id"] for candidate in batch]
            placeholders = ",".join("?" * len(ids))
            rows = cursor.execute(
                f"SELECT * FROM clips WHERE id IN ({placeholders})",  # nosec B608
                ids,
            ).fetchall()

        # Scan-time export already wrote preview MP4s for modern jobs. Only
        # render when a row is still missing its framed proxy (legacy scans).
        for row in rows:
            export_path = row["export_path"]
            if export_path and os.path.isfile(export_path):
                continue
            clip_id = row["id"]
            try:
                self.prepare_preview(clip_id)
            except Exception:
                logger.debug(
                    "Second-look preview render skipped for clip %s",
                    clip_id,
                    exc_info=True,
                )

        with self.db.get_connection() as conn:
            placeholders = ",".join("?" * len(ids))
            rows = conn.cursor().execute(
                f"SELECT * FROM clips WHERE id IN ({placeholders})",  # nosec B608
                ids,
            ).fetchall()

        by_id = {row["id"]: dict(row) for row in rows}
        ordered = annotate_clip_rows(
            self.db,
            [by_id[candidate["id"]] for candidate in batch if candidate["id"] in by_id],
        )
        return {
            "available": len(batch),
            "available_total": len(candidates),
            "loaded": len(ordered),
            "clips": ordered,
        }

    def preview(self, clip_id: str) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clips WHERE id = ?", (clip_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Clip not found")
        return annotate_clip_rows(self.db, [dict(row)])[0]

    def prepare_preview(self, clip_id: str, force: bool = False) -> Dict[str, Any]:
        """Render the framed vertical preview for a Second-look clip.

        Used when revealing more moments into the review deck and for any
        later explicit re-proof. Serialized so a batch reveal cannot stampede
        FFmpeg. ``force`` re-renders a clip that already has a proxy on disk --
        used to repair Second-look previews baked by a build that skipped the
        caption pass for that tier.
        """
        with self._preview_render_lock:
            return self._prepare_preview_locked(clip_id, force=force)

    @serialized_clip_operation
    def _prepare_preview_locked(self, clip_id: str, force: bool = False) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            clip_row = conn.cursor().execute(
                "SELECT * FROM clips WHERE id = ?", (clip_id,),
            ).fetchone()
        if not clip_row:
            raise HTTPException(status_code=404, detail="Clip not found")
        if not force and clip_row["export_path"] and os.path.isfile(clip_row["export_path"]):
            return self._frontend_clip_response(clip_id)
        if clip_row["story_label"] != self.MORE_CANDIDATE_LABEL:
            return self._frontend_clip_response(clip_id)

        layout = self._parse_layout(clip_row["layout"])
        if not layout:
            layout = self.resolve_manual_framing(
                clip_row["job_id"],
                float(clip_row["start_time"] or 0.0),
                float(clip_row["end_time"] or 0.0),
                requested_layout="auto",
            )
            with self.db.get_connection() as conn:
                conn.cursor().execute(
                    "UPDATE clips SET layout = ? WHERE id = ?",
                    (json.dumps(layout), clip_id),
                )
            with self.db.get_connection() as conn:
                clip_row = conn.cursor().execute(
                    "SELECT * FROM clips WHERE id = ?", (clip_id,),
                ).fetchone()

        try:
            rendered_path, thumb_path = self._render_existing_clip(
                clip_row,
                float(clip_row["start_time"] or 0.0),
                float(clip_row["end_time"] or 0.0),
                preview=True,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Preview rendering failed: {str(exc)}")

        with self.db.get_connection() as conn:
            conn.cursor().execute(
                "UPDATE clips SET export_path = ?, thumb_path = ?, preview_only = 1 WHERE id = ?",
                (rendered_path, thumb_path, clip_id),
            )
        return self._frontend_clip_response(clip_id)

    def update(self, clip_id: str, request: Any) -> Dict[str, Any]:
        """Persist creator edits to a clip's post metadata and review state
        (title, kept/passed flags, exported marker — plan 3.2). Additive fields
        are applied only when present so a partial PATCH never clobbers others."""
        fields: Dict[str, Any] = {}
        title = getattr(request, "title", None)
        if title is not None:
            fields["title"] = title.strip()[:200]
        kept = getattr(request, "kept", None)
        if kept is not None:
            fields["kept"] = 1 if kept else 0
        passed = getattr(request, "passed", None)
        if passed is not None:
            fields["passed"] = 1 if passed else 0
        maybe = getattr(request, "maybe", None)
        if maybe is not None:
            fields["maybe"] = 1 if maybe else 0
        active_verdicts = sum(value is True for value in (kept, passed, maybe))
        if active_verdicts > 1:
            raise HTTPException(status_code=422, detail="A clip can have only one review verdict")
        if kept is True:
            fields.update(passed=0, maybe=0)
        elif passed is True:
            fields.update(kept=0, maybe=0)
        elif maybe is True:
            fields.update(kept=0, passed=0)
        exported = getattr(request, "exported", None)
        if exported is not None:
            fields["exported_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if exported else None
        if not fields:
            return self._frontend_clip_response(clip_id)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM clips WHERE id = ?", (clip_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Clip not found")
            assignments = ", ".join(f"{col} = ?" for col in fields)
            cursor.execute(
                f"UPDATE clips SET {assignments} WHERE id = ?",  # nosec B608
                (*fields.values(), clip_id),
            )
        return self._frontend_clip_response(clip_id)

    def thumbnail_path(self, clip_id: str) -> str:
        """Absolute path to a clip's poster JPEG, generating it on demand for
        clips rendered before thumbnails existed (handbook/21 §6.1)."""
        from core.thumbnails import (
            generate_thumbnail,
            clip_relative_peak,
            thumb_filename,
        )
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT export_path, thumb_path, peak_timestamp, start_time, end_time "
                "FROM clips WHERE id = ?",
                (clip_id,),
            )
            row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Clip not found")

        existing = row["thumb_path"]
        if existing and os.path.exists(existing):
            return existing
        # Old session or a lost file: regenerate from the exported MP4.
        candidate = os.path.join(self.exports_dir, thumb_filename(clip_id))
        if os.path.exists(candidate):
            return candidate
        export_path = row["export_path"]
        # Database rows can outlive an install/data-folder move. The media route
        # already serves exports by basename from the active exports directory;
        # use the same recovery here so a playable clip never has a permanently
        # broken poster just because its stored absolute path is stale.
        if export_path and not os.path.isfile(export_path):
            relocated = os.path.join(self.exports_dir, os.path.basename(export_path))
            if os.path.isfile(relocated):
                export_path = relocated
        if not export_path or not os.path.isfile(export_path):
            raise HTTPException(status_code=404, detail="Thumbnail unavailable")
        seek = clip_relative_peak(row["peak_timestamp"], row["start_time"], row["end_time"])
        made = generate_thumbnail(export_path, clip_id, self.exports_dir, seek_seconds=seek)
        if not made:
            raise HTTPException(status_code=404, detail="Thumbnail unavailable")
        with self.db.get_connection() as conn:
            conn.cursor().execute(
                "UPDATE clips SET thumb_path = ? WHERE id = ?", (made, clip_id)
            )
        return made

    @serialized_clip_operation
    def edit(self, clip_id: str, request: Any) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clips WHERE id = ?", (clip_id,))
            clip_row = cursor.fetchone()
            if not clip_row:
                raise HTTPException(status_code=404, detail="Clip not found")
        try:
            rendered_path, thumb_path = self._render_existing_clip(
                clip_row, request.start_time, request.end_time,
                fade_in=request.fade_in, fade_out=request.fade_out,
                video_fade_in=request.video_fade_in, video_fade_out=request.video_fade_out,
                audio_fade_in=request.audio_fade_in, audio_fade_out=request.audio_fade_out,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE clips SET start_time = ?, end_time = ?, export_path = ?, thumb_path = ?, preview_only = 0 WHERE id = ?",
                (request.start_time, request.end_time, rendered_path, thumb_path, clip_id),
            )
            # Keep the exports table pointing at the live file (plan 4.1).
            cursor.execute(
                "UPDATE exports SET file_path = ? WHERE clip_id = ?",
                (rendered_path, clip_id),
            )
        # Boundary-edit supervision (HUMAN_CLIPS Package 1): a manual trim is
        # ground truth that the engine's window was wrong. Recorded in its own
        # table, never clip_labels — a re-cut is not a keep/reject preference
        # and must not enter ranker training. Only when the window actually
        # moved: edit() is also called for fade-only re-renders, and 1ms float
        # noise from the editor UI is not a decision.
        old_start = float(clip_row["start_time"] or 0.0)
        old_end = float(clip_row["end_time"] or 0.0)
        new_start, new_end = float(request.start_time), float(request.end_time)
        if abs(new_start - old_start) > 1e-3 or abs(new_end - old_end) > 1e-3:
            self.db.record_boundary_edit(
                clip_id, clip_row["job_id"], old_start, old_end, new_start, new_end,
            )
        # A re-render that lands on a new filename orphans the old MP4 —
        # remove it so edited clips stop leaking disk (plan 4.1).
        old_export = clip_row["export_path"]
        if old_export and os.path.abspath(old_export) != os.path.abspath(rendered_path) and os.path.exists(old_export):
            try:
                os.remove(old_export)
            except OSError:
                logger.debug("Could not delete superseded clip export %s", old_export, exc_info=True)
        return self._frontend_clip_response(clip_id)

    def _manual_job_row(self, job_id: str):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT source_path, asset_path, source_type, duration, framing "
                "FROM jobs WHERE id = ?",
                (job_id,),
            )
            job_row = cursor.fetchone()
            if not job_row:
                raise HTTPException(status_code=404, detail="Job not found")
        return job_row

    def manual_bounds(self, job_id: str, request: Any) -> tuple[float, float]:
        """Resolve either manual request shape to final source-VOD bounds."""
        job_row = self._manual_job_row(job_id)
        vod_duration = float(job_row["duration"] or 0.0)

        start_req = getattr(request, "start", None)
        end_req = getattr(request, "end", None)
        if start_req is not None and end_req is not None:
            # Precise bounds (Cutting Room): use them exactly — no
            # center/duration math — clamping only to the real VOD length.
            start = max(0.0, float(start_req))
            end = float(end_req)
            if vod_duration > 0:
                if start >= vod_duration:
                    raise HTTPException(status_code=422, detail="Clip start exceeds the source duration.")
                end = min(end, vod_duration)
            if end <= start:
                raise HTTPException(status_code=422, detail="Clip end must be after its start.")
            center = (start + end) / 2.0
        else:
            duration = float(getattr(request, "duration", 30.0) or 30.0)
            center = float(getattr(request, "timestamp", 0.0) or 0.0)
            if vod_duration > 0 and center > vod_duration:
                raise HTTPException(status_code=422, detail="Clip timestamp exceeds the source duration.")
            start = max(0.0, center - duration / 2.0)
            end = start + duration
            if vod_duration > 0 and end > vod_duration:
                end = vod_duration
                start = max(0.0, end - duration)
        return start, end

    @staticmethod
    def _parse_layout(raw: Any) -> Dict[str, Any] | None:
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (TypeError, ValueError):
            return None

    def _compiled_framing(self, job_id: str, raw: Any) -> Dict[str, Any] | None:
        """Parse and compile a job's large framing model once per API process."""
        if not raw:
            return None
        cached = self._framing_cache.get(job_id)
        if cached and cached[0] == raw:
            return cached[1]

        model = self._parse_layout(raw)
        if not model:
            return None

        from types import SimpleNamespace
        from engines.vision.facecam import FacecamLayout, FacecamObservation
        from engines.vision.gameplay_region import GameplayLayout, GameplayObservation

        compiled = {
            "model": model,
            "facecam_layouts": [
                FacecamLayout(
                    box=list(item.get("box") or []),
                    support=int(item.get("support", 0)),
                    fraction=float(item.get("fraction", 0.0)),
                    stability=float(item.get("stability", 0.0)),
                    # Sessions scanned before coverage was persisted fall back to
                    # ``fraction`` alone, exactly as they behave today.
                    coverage=float(item.get("coverage", 0.0)),
                    panel_support=int(item.get("panel_support", 0)),
                    fallback_support=int(item.get("fallback_support", 0)),
                    # Absent on sessions scanned before the cam crop was
                    # measured; those keep the constant fallback they rendered
                    # with, exactly as they behave today.
                    subject_top=(
                        float(item["subject_top"])
                        if item.get("subject_top") is not None else None
                    ),
                )
                for item in (model.get("facecam_layouts") or [])
                if len(item.get("box") or []) >= 4
            ],
            "facecam_signals": [
                SimpleNamespace(
                    timestamp=float(item[0]),
                    vision=SimpleNamespace(facecam_box=list(item[1])),
                )
                for item in (model.get("facecam_observations") or [])
                if isinstance(item, list) and len(item) >= 2 and len(item[1] or []) >= 4
            ],
            "facecam_timeline": [
                FacecamObservation(
                    timestamp=float(item.get("timestamp", 0.0)),
                    layout_index=(
                        int(item["layout_index"])
                        if item.get("layout_index") is not None else None
                    ),
                    state=str(item.get("state") or "unknown"),
                    source=item.get("source"),
                )
                for item in (model.get("facecam_timeline") or [])
                if isinstance(item, dict)
            ],
            "gameplay_layouts": [
                GameplayLayout(
                    box=list(item.get("box") or []),
                    timestamps=[float(value) for value in (item.get("timestamps") or [])],
                    support=int(item.get("support", 1)),
                    confidence=float(item.get("confidence", 0.0)),
                )
                for item in (model.get("gameplay_layouts") or [])
                if len(item.get("box") or []) >= 4
            ],
            "gameplay_observations": [
                GameplayObservation(
                    timestamp=float(item.get("timestamp", 0.0)),
                    box=list(item["box"]) if item.get("box") is not None else None,
                )
                for item in (model.get("gameplay_observations") or [])
            ],
        }
        # The desktop typically has one active session. Bound the cache anyway
        # so long-lived API processes do not retain every VOD ever opened.
        if len(self._framing_cache) >= 8:
            self._framing_cache.pop(next(iter(self._framing_cache)))
        self._framing_cache[job_id] = (raw, compiled)
        return compiled

    def _gameplay_probe(
        self,
        job_row: Any,
        game_layouts: List[Any],
        start: float,
        end: float,
    ) -> Optional[List[float]]:
        """The authored game window at this clip's midpoint, or None.

        Best-effort and layout-anchored: a one-frame rectangle only counts when
        it matches a layout the whole-VOD model already established, so a
        mis-detection can never invent a crop. Needs the local asset — a job
        whose source is still a URL keeps the scene-timeline answer rather than
        triggering a download from a framing request.
        """
        if not game_layouts:
            return None
        asset = job_row["asset_path"] or ""
        if not asset or not os.path.isfile(asset):
            return None
        midpoint = round((float(start) + float(end)) / 2.0, 1)
        key = (asset, midpoint)
        if key in self._gameplay_probe_cache:
            return self._gameplay_probe_cache[key]

        from engines.vision.gameplay_region import detect_gameplay_box_at, match_gameplay_layout
        try:
            box = match_gameplay_layout(
                game_layouts, detect_gameplay_box_at(asset, midpoint),
            )
        except Exception:  # noqa: BLE001 - the scene timeline remains the answer
            logger.debug("Gameplay probe failed at %.1fs", midpoint, exc_info=True)
            box = None
        if len(self._gameplay_probe_cache) >= 256:
            self._gameplay_probe_cache.pop(next(iter(self._gameplay_probe_cache)))
        self._gameplay_probe_cache[key] = box
        return box

    def _scan_model_manual_framing(
        self,
        job_row: Any,
        compiled: Dict[str, Any],
        start: float,
        end: float,
    ) -> tuple[Any, Any, Any, str]:
        """Resolve facecam/gameplay evidence from a persisted scan model."""
        from engines.vision.facecam import facecam_for_window, subject_top_for_box
        from engines.vision.gameplay_region import gameplay_for_window

        model = compiled["model"]
        layouts = compiled["facecam_layouts"]
        facecam_timeline = compiled["facecam_timeline"]
        game_layouts = compiled["gameplay_layouts"]
        game_observations = compiled["gameplay_observations"]
        observations = (
            facecam_timeline
            if int(model.get("version", 1) or 1) >= 2
            else None
        )
        facecam_box = facecam_for_window(
            layouts,
            compiled["facecam_signals"],
            start,
            end,
            observations=observations,
            layout_verifier=lambda box, timestamp: self._facecam_presence_probe(
                job_row, box, timestamp,
            ),
        )
        facecam_subject_top = subject_top_for_box(layouts, facecam_box)

        # Match scan-time order: probe this clip's midpoint first, then use the
        # sparse scene timeline. An explicit timeline miss means fullscreen;
        # only a model without any scene observations may inherit the fallback.
        gameplay_box = self._gameplay_probe(job_row, game_layouts, start, end)
        if gameplay_box is None:
            gameplay_box = gameplay_for_window(
                game_layouts,
                start,
                end,
                observations=game_observations or None,
            )
            if not game_observations:
                gameplay_box = gameplay_box or model.get("fallback_gameplay")
        resolved_auto = str(model.get("resolved_layout") or "vertical_split")
        return facecam_box, facecam_subject_top, gameplay_box, resolved_auto

    def _legacy_manual_framing(
        self,
        job_id: str,
        start: float,
        end: float,
    ) -> tuple[Any, Any, Any, str, str]:
        """Recover framing from saved clips for pre-model sessions."""
        with self.db.get_connection() as conn:
            rows = conn.cursor().execute(
                "SELECT start_time, end_time, layout FROM clips "
                "WHERE job_id = ? AND layout IS NOT NULL",
                (job_id,),
            ).fetchall()

        candidates = []
        midpoint = (start + end) / 2.0
        for row in rows:
            layout = self._parse_layout(row["layout"])
            if not layout:
                continue
            row_midpoint = (
                float(row["start_time"] or 0.0)
                + float(row["end_time"] or 0.0)
            ) / 2.0
            overlap = max(
                0.0,
                min(end, float(row["end_time"] or 0.0))
                - max(start, float(row["start_time"] or 0.0)),
            )
            candidates.append((layout, overlap, abs(row_midpoint - midpoint)))

        if not candidates:
            return None, None, None, "vertical_split", "fallback"

        nearest = min(candidates, key=lambda item: (-item[1], item[2]))[0]
        facecam_box = nearest.get("facecam")
        facecam_subject_top = nearest.get("facecam_subject_top")
        gameplay_box = nearest.get("gameplay")
        types = [
            item[0].get("type")
            for item in candidates
            if item[0].get("facecam")
            and item[0].get("type") in {"vertical_split", "gameplay_pip"}
        ]
        resolved_auto = "vertical_split"
        if types:
            counts = {layout_type: types.count(layout_type) for layout_type in set(types)}
            highest = max(counts.values())
            tied = {
                layout_type
                for layout_type, count in counts.items()
                if count == highest
            }
            nearest_type = nearest.get("type")
            resolved_auto = (
                nearest_type
                if nearest_type in tied
                else next(
                    layout_type
                    for layout_type in ("vertical_split", "gameplay_pip")
                    if layout_type in tied
                )
            )
        elif nearest.get("type") in {
            "vertical_split", "gameplay_pip", "full_gameplay",
        }:
            resolved_auto = nearest["type"]
        return (
            facecam_box,
            facecam_subject_top,
            gameplay_box,
            resolved_auto,
            "nearest_clip",
        )

    def _manual_fullframe_camera_focus(
        self,
        job_row: Any,
        start: float,
        end: float,
        gameplay_box: Any,
        *,
        requested: str,
        facecam_box: Any,
        layout_type: str,
    ) -> Optional[float]:
        """Probe an authored fullscreen camera only when auto framing needs it."""
        if requested != "auto" or facecam_box is not None or layout_type == "gameplay_pip":
            return None
        asset = job_row["asset_path"] or ""
        midpoint = round((float(start) + float(end)) / 2.0, 1)
        region_key = tuple(round(float(value), 4) for value in (gameplay_box or []))
        key = (asset, midpoint, region_key)
        if key in self._fullframe_camera_probe_cache:
            return self._fullframe_camera_probe_cache[key]
        if not asset or not os.path.isfile(asset):
            return None

        from engines.vision.facecam import fullframe_camera_focus_at

        camera_focus_x = fullframe_camera_focus_at(
            asset, midpoint, camera_region=gameplay_box,
        )
        if len(self._fullframe_camera_probe_cache) >= 256:
            self._fullframe_camera_probe_cache.pop(
                next(iter(self._fullframe_camera_probe_cache)))
        self._fullframe_camera_probe_cache[key] = camera_focus_x
        return camera_focus_x

    @staticmethod
    def _describe_manual_layout(
        resolved: Dict[str, Any],
        *,
        requested: str,
        evidence_source: str,
        focus_x: float,
    ) -> Dict[str, Any]:
        """Attach creator-facing provenance and composition language."""
        resolved["focus_x"] = max(0.0, min(1.0, float(focus_x)))
        resolved["requested"] = requested
        resolved["source"] = evidence_source
        if resolved.get("type") == "vtuber_overlay":
            composition = "VTuber overlay"
            reason = (
                "VTuber mode was confirmed for this VOD, so Recall uses the "
                "saved avatar cutout."
            )
        elif resolved.get("type") == "full_camera":
            composition = "Fullscreen camera"
            reason = (
                "Recall detected a fullscreen webcam and keeps the complete "
                "camera frame visible."
            )
        elif resolved.get("facecam"):
            composition = (
                "Gameplay PiP"
                if resolved["type"] == "gameplay_pip"
                else "Stacked facecam"
            )
            reason = (
                "Recall matched the facecam and gameplay regions detected for "
                "this part of the VOD."
            )
        else:
            composition = "Gameplay only"
            reason = (
                "No reliable facecam was detected in this range, so Recall "
                "keeps gameplay full-height."
            )
        resolved["label"] = composition
        resolved["reason"] = reason
        return resolved

    def resolve_manual_framing(
        self,
        job_id: str,
        start: float,
        end: float,
        *,
        requested_layout: str = "auto",
        focus_x: float = 0.5,
        _job_row: Any = None,
    ) -> Dict[str, Any]:
        """Resolve a VOD Editor selection through Recall's scan-time model.

        New scans persist the actual facecam and authored-gameplay timelines.
        Sessions made before that migration fall back to the closest automatic
        clip's saved geometry instead of the old generic stacked crop.
        """
        from engines.clip.layout import assign_layout

        job_row = _job_row or self._manual_job_row(job_id)
        compiled = self._compiled_framing(job_id, job_row["framing"])
        model = compiled["model"] if compiled else None
        if model and compiled:
            (
                facecam_box,
                facecam_subject_top,
                gameplay_box,
                resolved_auto,
            ) = self._scan_model_manual_framing(job_row, compiled, start, end)
            evidence_source = "scan_model"
        else:
            (
                facecam_box,
                facecam_subject_top,
                gameplay_box,
                resolved_auto,
                evidence_source,
            ) = self._legacy_manual_framing(job_id, start, end)

        requested = requested_layout if requested_layout in {
            "auto", "vertical_split", "gameplay_pip", "full_gameplay",
        } else "auto"
        layout_type = resolved_auto if requested == "auto" else requested
        if layout_type == "full_gameplay":
            facecam_box = None
            facecam_subject_top = None
        camera_focus_x = self._manual_fullframe_camera_focus(
            job_row,
            start,
            end,
            gameplay_box,
            requested=requested,
            facecam_box=facecam_box,
            layout_type=layout_type,
        )
        resolved = assign_layout(
            facecam_box,
            export_layout=layout_type,
            gameplay_box=gameplay_box,
            fullframe_camera=camera_focus_x is not None,
            camera_focus_x=camera_focus_x,
            facecam_subject_top=facecam_subject_top,
        )
        if requested == "auto" and resolved.get("facecam") and model:
            from engines.vision import vtuber as vtuber_mod

            overlay = vtuber_mod.overlay_for_facecam(
                model.get("vtuber"), facecam_box,
            )
            if overlay is not None:
                resolved = vtuber_mod.apply_overlay(resolved, overlay)
        return self._describe_manual_layout(
            resolved,
            requested=requested,
            evidence_source=evidence_source,
            focus_x=focus_x,
        )

    def _facecam_presence_probe(
        self,
        job_row: Any,
        box: List[float],
        timestamp: float,
    ) -> bool:
        """Cached positive verification for an unresolved persisted sample."""
        asset = job_row["asset_path"] or ""
        if not asset or not os.path.isfile(asset):
            return False
        box_key = tuple(round(float(value), 4) for value in box)
        key = (asset, round(float(timestamp), 1), box_key)
        if key in self._facecam_presence_probe_cache:
            return self._facecam_presence_probe_cache[key]
        from engines.vision.facecam import is_facecam_layout_at
        present = is_facecam_layout_at(asset, timestamp, box)
        if len(self._facecam_presence_probe_cache) >= 256:
            self._facecam_presence_probe_cache.pop(
                next(iter(self._facecam_presence_probe_cache)))
        self._facecam_presence_probe_cache[key] = present
        return present

    def _manual_context(self, job_id: str, request: Any):
        job_row = self._manual_job_row(job_id)
        start, end = self.manual_bounds(job_id, request)
        center = (start + end) / 2.0
        # Prefer the exact local asset that powered the scan and the Cutting
        # Room player. This avoids resolving (and potentially re-downloading)
        # a Twitch URL when its downloaded VOD is already present, and keeps
        # rendering aligned with the source the creator just scrubbed.
        video_path = self._local_source_video(job_row)
        raw_focus = getattr(request, "focus_x", 0.5)
        layout = self.resolve_manual_framing(
            job_id,
            start,
            end,
            requested_layout=getattr(request, "layout", "auto") or "auto",
            focus_x=float(0.5 if raw_focus is None else raw_focus),
        )
        return job_row, video_path, start, end, center, layout

    @staticmethod
    def _manual_fades(request: Any) -> Dict[str, float]:
        return {
            "video_fade_in": float(getattr(request, "video_fade_in", 0.0) or 0.0),
            "video_fade_out": float(getattr(request, "video_fade_out", 0.0) or 0.0),
            "audio_fade_in": float(getattr(request, "audio_fade_in", 0.0) or 0.0),
            "audio_fade_out": float(getattr(request, "audio_fade_out", 0.0) or 0.0),
        }

    def preview_manual(self, job_id: str, request: Any) -> str:
        """Render an exact low-resolution proof without creating a clip/label."""
        from core.models.clip import GameClip
        from engines.export.renderer import render_clip

        _, video_path, start, end, _, layout = self._manual_context(job_id, request)
        preview_id = f"manual_preview_{uuid.uuid4().hex[:12]}"
        clip = GameClip(
            clip_id=preview_id,
            start=start,
            end=end,
            story_id="manual_preview",
            score=0.75,
            layout=layout,
            reason="VOD Editor preview",
        )
        captions_enabled = getattr(request, "captions_enabled", None)
        # Same medium.en caption pass as final export so scrubbing matches the
        # deliverable (Distil-slice previews were inaccurate on words + timing).
        ass_path = self._regenerate_ass_if_possible(
            video_path, preview_id, start, end,
            enabled_override=captions_enabled,
            prefer_alignment=True,
        )
        preview_dir = os.path.join(self.data_root, "temp", "vod-editor-previews")
        os.makedirs(preview_dir, exist_ok=True)
        try:
            with source_in_use(self.db, job_id):
                return render_clip(
                    video_path=video_path,
                    clip=clip,
                    output_dir=preview_dir,
                    ass_path=ass_path,
                    preview=True,
                    **self._manual_fades(request),
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Preview rendering failed: {str(exc)}")

    def cleanup_manual_preview(self, path: str) -> None:
        stem = os.path.splitext(os.path.basename(path))[0]
        clip_id = stem[5:] if stem.startswith("clip_") else stem
        for candidate in (
            path,
            os.path.join(os.path.dirname(path), f"{stem}_meta.json"),
            os.path.join(self.data_root, "temp", f"caption_{clip_id}.ass"),
        ):
            try:
                if os.path.isfile(candidate):
                    os.remove(candidate)
            except OSError:
                logger.debug("Could not clean VOD Editor preview %s", candidate, exc_info=True)

    def create_manual(self, job_id: str, request: Any) -> Dict[str, Any]:
        from core.models.clip import GameClip

        _, video_path, start, end, center, layout = self._manual_context(job_id, request)
        memory_provenance = None
        memory_context = None
        memory_query = ""
        memory_entry_id = str(getattr(request, "memory_entry_id", "") or "").strip()
        if memory_entry_id:
            from core.stream_memory import StreamMemoryService

            try:
                memory_context = StreamMemoryService(self.db).action_context(memory_entry_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if memory_context["job_id"] != job_id:
                raise HTTPException(
                    status_code=422,
                    detail="That Memory moment belongs to a different session.",
                )
            evidence_start = float(memory_context["start_time"])
            evidence_end = float(memory_context["end_time"])
            evidence_anchor = (evidence_start + evidence_end) / 2.0
            if not start <= evidence_anchor <= end:
                raise HTTPException(
                    status_code=422,
                    detail="Keep the remembered timestamp inside the finished clip to preserve its provenance.",
                )
            memory_query = " ".join(
                str(getattr(request, "memory_query", "") or "").split()
            )[:240]
            memory_provenance = {
                "source": "stream_memory",
                "origin": "memory_result",
                "memory_entry_id": memory_context["entry_id"],
                "entry_fingerprint": memory_context["entry_fingerprint"],
                "entry_kind": memory_context["kind"],
                "evidence_kind": memory_context.get("evidence_kind"),
                "source_start": evidence_start,
                "source_end": evidence_end,
                "query": memory_query,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "schema_version": 1,
            }
        clip_id = f"manual_{uuid.uuid4().hex[:12]}"
        clip = GameClip(
            clip_id=clip_id,
            start=start,
            end=end,
            story_id="manual",
            score=0.75,
            layout=layout,
            reason="Manual clip",
        )
        # A compilation cut is a means to a montage, not a clip the creator
        # kept: it must not appear in Clip Library and must not train the
        # ranker as a positive example nobody chose.
        origin = str(getattr(request, "origin", "") or "").strip() or None
        is_library_clip = origin != "compilation"

        captions_enabled = getattr(request, "captions_enabled", None)
        ass_path = self._regenerate_ass_if_possible(
            video_path, clip_id, start, end,
            enabled_override=captions_enabled,
            # A montage cut takes its words from the scan's full-VOD pass
            # rather than re-decoding its own few seconds. The short-window
            # pass has no surrounding context and mis-hears: on a twelve-second
            # finale it turned "wait wait" into "weed". The VOD pass is
            # sometimes silent where speech was missed, but silent beats
            # confidently wrong on a clip about to be posted.
            prefer_alignment=is_library_clip,
        )
        try:
            # peak_timestamp=center makes the poster land at the clip midpoint.
            with source_in_use(self.db, job_id):
                rendered_path, thumb_path = self._render_and_thumb(
                    video_path,
                    clip,
                    peak_timestamp=center,
                    ass_path=ass_path,
                    **self._manual_fades(request),
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")

        from core.job_manager import format_vod_time
        requested_title = str(getattr(request, "title", "") or "").strip()
        title = requested_title[:200] or f"Manual clip at {format_vod_time(center)}"
        raw_tags = getattr(request, "tags", None) or []
        tags = [str(tag).strip()[:60] for tag in raw_tags if str(tag).strip()][:20]
        # Manual clips are the only direct truth for moments the selector did
        # not surface. New scans persist every candidate, so cheaply attach the
        # matching feature vector now; record_label below can then train on the
        # missed positive immediately. Old jobs/candidate-formation misses keep
        # the historical featureless fallback and remain valid manual clips.
        from core.artifacts import match_candidate_features
        candidate_match = match_candidate_features(job_id, start, end)
        manual_features = (
            json.dumps(candidate_match["features"]) if candidate_match else None
        )
        description = (
            "Created from a Stream Memory result in the VOD Editor."
            if memory_provenance else "Created manually in the VOD Editor."
        )
        reason = "Stream Memory handoff" if memory_provenance else "Manual clip"
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            from core.clip_number import next_clip_number

            clip_number = next_clip_number(conn, job_id)
            cursor.execute(
                """
                INSERT INTO clips (
                    id, job_id, clip_number, start_time, end_time, score, selection_score,
                    hook_score, deck_score, title, export_path,
                    description, signals, story_label, reason, peak_timestamp,
                    modality_breakdown, reaction_auc, tags, kept, preview_only, layout,
                    features, recall_provenance, origin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clip_id, job_id, clip_number, start, end, 0.75, None, None, 0.75,
                    title, rendered_path,
                    description,
                    json.dumps(["manual", "stream_memory"] if memory_provenance else ["manual"]),
                    "manual",
                    reason,
                    center,
                    json.dumps({"manual": 1.0}),
                    0.0,
                    json.dumps(tags) if tags else None,
                    1 if is_library_clip else 0,
                    0,
                    json.dumps(layout),
                    manual_features,
                    json.dumps(memory_provenance, separators=(",", ":")) if memory_provenance else None,
                    origin,
                ),
            )
            cursor.execute(
                "UPDATE clips SET thumb_path = ? WHERE id = ?",
                (thumb_path, clip_id),
            )
            cursor.execute(
                "INSERT INTO exports (id, clip_id, platform, file_path) VALUES (?, ?, ?, ?)",
                (f"export_{clip_id}", clip_id, "manual", rendered_path),
            )
        # Preserve this as an explicit missed-positive. When an artifact match
        # exists, record_label snapshots its feature vector; otherwise offline
        # replay tooling can still reconcile old jobs later. A compilation cut
        # is not a creator judgement, so it teaches the ranker nothing.
        if is_library_clip:
            self.db.record_label(clip_id, label=1, event="manual")
        if memory_context is not None:
            try:
                StreamMemoryService(self.db).record_action(
                    memory_query,
                    memory_entry_id,
                    "clip_created",
                    dedupe_key=f"clip_created:{clip_id}",
                    context={"created_clip_id": clip_id},
                )
            except Exception:
                logger.debug(
                    "Stream Memory clip-creation learning skipped for %s",
                    clip_id,
                    exc_info=True,
                )
        return self._frontend_clip_response(clip_id)

    def _caption_style(self, enabled_override: bool | None) -> Any:
        """Resolve the saved style with an optional per-render enable flag."""
        from engines.caption.caption_style import resolve_caption_style

        raw_style = self.db.get_setting("captionStyle", None)
        if enabled_override is not None:
            saved = raw_style if isinstance(raw_style, dict) else {}
            raw_style = {**saved, "enabled": enabled_override}
        return resolve_caption_style(raw_style)

    @staticmethod
    def _caption_audio_cache_key(video_path: str) -> Optional[str]:
        from core.cache_keys import audio_cache_key

        try:
            return audio_cache_key(video_path)
        except OSError:
            return None

    # -- caption text: read what is burned in, correct it -------------------

    def get_caption(self, clip_id: str, transcribe: bool = False) -> Dict[str, Any]:
        """The words currently burned into this clip, and where they came from.

        A stored correction is a cheap row read and always returned. Reading
        the MACHINE words is not cheap -- it re-runs the per-clip ASR -- so it
        happens only when ``transcribe`` asks for it, and otherwise the caller
        gets ``source: "none"``. Opening the clip editor must not cost a model
        load and twenty seconds for a caption nobody looked at.

        The fallback is the per-clip ASR and deliberately NOT a slice of the
        cached VOD transcript: that is a different decode of the same seconds
        and routinely disagrees with what is actually on screen (measured
        2026-09-01). Showing the creator text that is not the text in their
        video would make every edit a guess.
        """
        with self.db.get_connection() as conn:
            row = conn.cursor().execute(
                "SELECT * FROM clips WHERE id = ?", (clip_id,),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Clip not found")
        start = float(row["start_time"] or 0.0)
        end = float(row["end_time"] or 0.0)

        from engines.caption.caption_edits import words_to_text

        edit = self.db.get_caption_edit(clip_id)
        if edit:
            stale = (abs(float(edit["clip_start"]) - start) > self.CAPTION_EDIT_WINDOW_TOLERANCE
                     or abs(float(edit["clip_end"]) - end) > self.CAPTION_EDIT_WINDOW_TOLERANCE)
            if not stale:
                return {
                    "clip_id": clip_id, "source": "edit", "stale": False,
                    "text": words_to_text(edit["words"]),
                    "words": edit["words"],
                    "machine_text": edit.get("machine_text") or "",
                }
            # Held, not applied: the clip was re-cut after the correction, so
            # the stored timings describe a window that no longer exists.
            # Surfaced so the UI can say so rather than silently re-transcribing.
            return {
                **self._machine_caption(row, start, end, transcribe),
                "stale": True,
                "held_edit_text": words_to_text(edit["words"]),
            }

        return {**self._machine_caption(row, start, end, transcribe), "stale": False}

    def _machine_caption(self, row, start: float, end: float,
                         transcribe: bool) -> Dict[str, Any]:
        """The decode's words, or a cheap "not read yet" when not asked for."""
        from engines.caption.caption_edits import words_to_text

        if not transcribe:
            return {"clip_id": row["id"], "source": "none",
                    "text": "", "words": [], "machine_text": ""}
        words = self._machine_caption_words(row, start, end)
        return {"clip_id": row["id"], "source": "machine",
                "text": words_to_text(words), "words": words,
                "machine_text": words_to_text(words)}

    @serialized_clip_operation
    def save_caption(self, clip_id: str, text: str) -> Dict[str, Any]:
        """Store a corrected caption, re-timed onto the machine's word timings.

        The correction is persisted, not rendered here. The next render of this
        clip picks it up through _regenerate_ass_if_possible; the mp4 already on
        disk still carries the old words until then, which the response says so
        the UI can offer the re-render rather than implying one happened.
        """
        from engines.caption.caption_edits import retime_edited_words, words_to_text

        with self.db.get_connection() as conn:
            row = conn.cursor().execute(
                "SELECT * FROM clips WHERE id = ?", (clip_id,),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Clip not found")
        start = float(row["start_time"] or 0.0)
        end = float(row["end_time"] or 0.0)

        text = (text or "").strip()
        if not text:
            # An empty correction means "give me the machine words back", not
            # "burn in an empty caption".
            self.db.clear_caption_edit(clip_id)
            return {**self.get_caption(clip_id), "cleared": True, "needs_render": True}

        # Re-time against whatever is on screen now: a stored edit if there is
        # one (so a second correction refines the first rather than reverting
        # to the machine), otherwise the decode.
        existing = self.db.get_caption_edit(clip_id)
        fresh = (existing["words"] if existing
                 and abs(float(existing["clip_start"]) - start) <= self.CAPTION_EDIT_WINDOW_TOLERANCE
                 and abs(float(existing["clip_end"]) - end) <= self.CAPTION_EDIT_WINDOW_TOLERANCE
                 else self._machine_caption_words(row, start, end))
        # (unconditional: a correction cannot be timed without the decode)
        if not fresh:
            raise HTTPException(
                status_code=409,
                detail="No caption timings available for this clip yet; render it first.",
            )

        words = retime_edited_words(fresh, text)
        if not words:
            raise HTTPException(status_code=422, detail="Caption text could not be timed.")

        machine_text = (existing.get("machine_text") if existing else "") or words_to_text(fresh)
        self.db.record_caption_edit(
            clip_id, row["job_id"], start, end, words, machine_text=machine_text,
        )
        return {
            "clip_id": clip_id, "source": "edit", "stale": False,
            "text": words_to_text(words), "words": words,
            "machine_text": machine_text,
            "needs_render": True,
        }

    def _machine_caption_words(self, clip_row, start: float, end: float) -> List[Dict]:
        """The per-clip ASR's timed words for a clip -- the decode that renders."""
        from engines.caption.whisper_asr import transcribe_clip_window

        with self.db.get_connection() as conn:
            job_row = conn.cursor().execute(
                "SELECT source_path, asset_path, source_type, duration FROM jobs WHERE id = ?",
                (clip_row["job_id"],),
            ).fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail="Job associated with clip not found")

        with source_in_use(self.db, clip_row["job_id"]):
            video_path = self._local_source_video(job_row)
            temp_dir = os.path.join(self.data_root, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            transcript = transcribe_clip_window(
                video_path, start, end,
                os.path.join(temp_dir, f"caption_edit_{clip_row['id']}.wav"),
            )
        words: List[Dict] = []
        for segment in transcript.get("segments", []) or []:
            words.extend(segment.get("words") or [])
        return words

    CAPTION_EDIT_WINDOW_TOLERANCE = 0.05

    def _corrected_caption_ass(
        self,
        clip_id: str,
        start: float,
        end: float,
        ass_path: str,
        style: Any,
    ) -> Optional[str]:
        """Render the creator's stored caption words, if they still apply.

        Held back when the clip has been re-cut since the edit was made: the
        stored timings are clip-relative and describe the old window, so
        replaying them against a moved boundary would put right words on wrong
        frames -- worse than the machine text they replaced. The edit stays in
        the table for that case rather than being deleted, so re-cutting back
        restores it.
        """
        try:
            edit = self.db.get_caption_edit(clip_id)
        except Exception:  # noqa: BLE001 - a caption edit must never fail a render
            logger.debug("Caption edit lookup failed for clip %s", clip_id, exc_info=True)
            return None
        if not edit:
            return None

        tolerance = self.CAPTION_EDIT_WINDOW_TOLERANCE
        if (abs(float(edit["clip_start"]) - float(start)) > tolerance
                or abs(float(edit["clip_end"]) - float(end)) > tolerance):
            logger.info(
                "Caption edit for clip %s was made against %.3f-%.3f but the clip is "
                "now %.3f-%.3f; falling back to ASR for this render",
                clip_id, edit["clip_start"], edit["clip_end"], start, end,
            )
            return None

        from engines.caption.whisper_asr import generate_ass_for_clip

        # One synthetic segment: generate_tiktok_ass flattens segments into a
        # single word stream anyway, and chunks on real pauses in the timings.
        transcript = {"segments": [{"words": edit["words"]}], "language": "en"}
        try:
            written = generate_ass_for_clip(
                "", start, end, ass_path, style=style, transcript=transcript,
            )
        except Exception:  # noqa: BLE001 - fall through to the machine decode
            logger.debug("Caption edit render failed for clip %s", clip_id, exc_info=True)
            return None
        return written if written and os.path.exists(written) else None

    def _try_aligned_caption_ass(
        self,
        video_path: str,
        clip_id: str,
        start: float,
        end: float,
        ass_path: str,
        style: Any,
    ) -> Optional[str]:
        """Run the accurate short-window caption pass, returning only a real file."""
        from engines.caption.whisper_asr import generate_ass_for_clip

        try:
            proc_mode = self.db.get_setting("processingMode", None)
            cap_settings = {"processingMode": proc_mode} if proc_mode else None
            written = generate_ass_for_clip(
                video_path,
                start,
                end,
                ass_path,
                style=style,
                settings=cap_settings,
            )
            return written if written and os.path.exists(written) else None
        except Exception:
            logger.debug(
                "medium.en caption pass failed for clip %s; falling back to VOD slice",
                clip_id,
                exc_info=True,
            )
            return None

    @staticmethod
    def _caption_transcript_cache_labels() -> List[str]:
        """Transcript cache labels to look for, most current first."""
        from engines.caption.whisper_asr import asr_cache_label

        # One backend, one label. The whisper compatibility labels went with
        # the whisper backend on 2026-09-01: nothing on disk carried them, and
        # a build cannot ship without the Qwen weights that produce this one.
        return [asr_cache_label()]

    def _load_caption_transcript(
        self,
        audio_key: str,
    ) -> Optional[Dict[str, Any]]:
        from core import cache_keys, signal_cache

        for label in self._caption_transcript_cache_labels():
            cache_path = os.path.join(
                self.data_root,
                "cache",
                cache_keys.transcript_cache_name(
                    audio_key,
                    label,
                    signal_cache.CACHE_EXT,
                ),
            )
            transcript = signal_cache.load(cache_path)
            if transcript:
                return transcript
        return None

    @staticmethod
    def _write_cached_caption_ass(
        transcript: Dict[str, Any],
        start: float,
        end: float,
        ass_path: str,
        style: Any,
        clip_id: str,
    ) -> Optional[str]:
        from engines.caption.whisper_asr import generate_ass_from_transcript

        try:
            return generate_ass_from_transcript(
                transcript,
                start,
                end,
                ass_path,
                style=style,
            )
        except Exception:
            logger.debug(
                "Could not regenerate ASS captions for clip %s",
                clip_id,
                exc_info=True,
            )
            return None

    @staticmethod
    def _ass_has_dialogue(ass_path: str) -> bool:
        """True when the subtitle file actually carries a line.

        A window the scan heard nothing in still writes a valid, empty ASS.
        Treating that as success would leave the clip silently uncaptioned
        instead of falling through to a decode that might hear something.
        """
        try:
            with open(ass_path, "r", encoding="utf-8") as handle:
                return any(line.startswith("Dialogue:") for line in handle)
        except OSError:
            return False

    def _regenerate_ass_if_possible(
        self,
        video_path: str,
        clip_id: str,
        start: float,
        end: float,
        enabled_override: bool | None = None,
        prefer_alignment: bool = True,
    ) -> str | None:
        """Creator correction, else the scan's own words, else a short-window decode."""
        style = self._caption_style(enabled_override)
        if not style.enabled:
            return None

        temp_dir = os.path.join(self.data_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        ass_path = os.path.join(temp_dir, f"caption_{clip_id}.ass")
        audio_key = self._caption_audio_cache_key(video_path)

        # A correction outranks every decode below it. It has to: the per-clip
        # ASR is not stable across boundaries, so re-deciding from the audio
        # would quietly discard what the creator told us the words are.
        corrected = self._corrected_caption_ass(clip_id, start, end, ass_path, style)
        if corrected:
            return corrected

        # The scan already transcribed this recording end to end, with every
        # word timed against the whole VOD. Decoding the clip's own seconds
        # again is a SECOND transcription of audio Recall has already read, and
        # a worse one: the window carries no surrounding context, so it
        # mis-hears ("wait wait" -> "weed") and its alignment drifts across
        # silence (six seconds inside one 57s clip). The scan's words come
        # first now; the short-window pass is the fallback for the one case it
        # is genuinely needed -- a clip whose window the scan has no words for.
        if audio_key:
            transcript = self._load_caption_transcript(audio_key)
            if transcript:
                written = self._write_cached_caption_ass(
                    transcript,
                    start,
                    end,
                    ass_path,
                    style,
                    clip_id,
                )
                if written and self._ass_has_dialogue(written):
                    return written

        if not prefer_alignment:
            return None
        return self._try_aligned_caption_ass(
            video_path,
            clip_id,
            start,
            end,
            ass_path,
            style,
        )

    def _frontend_clip_response(self, clip_id: str) -> Dict[str, Any]:
        """Return the raw clip row (same shape as GET /clips rows) so the
        frontend runs edit/update responses through the *same* `toClip` mapper
        as the initial load — no second, drifting serializer to keep in sync."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clips WHERE id = ?", (clip_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Clip not found")
        return annotate_clip_rows(self.db, [dict(row)])[0]


class ExportOperationService:
    """Serialize user-initiated exports and expose truthful progress snapshots.

    Final-quality renders are deliberately serialized: two simultaneous FFmpeg
    exports fight for the same encoder and make both slower. Operation state is
    process-local and bounded because exported files and DB markers remain the
    durable authority across an app restart.
    """

    _TERMINAL = {"completed", "partial", "failed", "cancelled"}
    _MAX_OPERATIONS = 64

    def __init__(self, clip_service: ClipService):
        self.clip_service = clip_service
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()
        self._operations: Dict[str, Dict[str, Any]] = {}

    def start(
        self,
        clip_ids: List[str],
        dest_folder: str,
        preset: str = None,
        filename_template: str = None,
    ) -> Dict[str, Any]:
        return self._enqueue(
            kind="clips",
            clip_ids=clip_ids,
            target=self._run,
            target_args=(dest_folder, preset, filename_template),
            message="Waiting to prepare final files",
        )

    def start_reel(
        self,
        clip_ids: List[str],
        dest_folder: str,
        filename: str = None,
    ) -> Dict[str, Any]:
        return self._enqueue(
            kind="reel",
            clip_ids=clip_ids,
            target=self._run_reel,
            target_args=(dest_folder, filename),
            message="Waiting to prepare highlight reel",
        )

    def _enqueue(
        self,
        *,
        kind: str,
        clip_ids: List[str],
        target: Callable[..., None],
        target_args: tuple,
        message: str,
    ) -> Dict[str, Any]:
        prefix = "export" if kind == "clips" else kind
        operation_id = f"{prefix}_{uuid.uuid4().hex[:16]}"
        now = time.time()
        operation = {
            "id": operation_id,
            "kind": kind,
            "status": "queued",
            "phase": "queued",
            "total_clips": len(clip_ids),
            "completed_clips": 0,
            "current_clip_id": None,
            "current_index": 0,
            "current_clip_progress": None,
            "current_clip_duration": None,
            "rendered_seconds": None,
            "reel_progress": None,
            "reel_duration": None,
            "progress": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "copied": 0,
            "errors": 0,
            "succeeded_clip_ids": [],
            "failed": [],
            "path": None,
            "clips": None,
            "cancel_requested": False,
            "message": message,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._prune_locked()
            if len(self._operations) >= self._MAX_OPERATIONS:
                raise HTTPException(
                    status_code=429,
                    detail="Too many export operations are still waiting. Let the current export finish and try again.",
                )
            self._operations[operation_id] = operation
        threading.Thread(
            target=target,
            args=(operation_id, list(clip_ids), *target_args),
            name=f"recall-{operation_id}",
            daemon=True,
        ).start()
        return self.get(operation_id)

    def get(self, operation_id: str) -> Dict[str, Any]:
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                raise HTTPException(status_code=404, detail="Export operation not found")
            return dict(operation)

    def has_active_operations(self) -> bool:
        """True while an export or reel is queued, running, or cancelling."""
        with self._lock:
            return any(
                operation.get("status") not in self._TERMINAL
                for operation in self._operations.values()
            )

    def cancel(self, operation_id: str) -> Dict[str, Any]:
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                raise HTTPException(status_code=404, detail="Export operation not found")
            if operation.get("status") in self._TERMINAL:
                return dict(operation)
            operation["cancel_requested"] = True
            operation["phase"] = "cancelling"
            operation["message"] = (
                "Stopping highlight reel"
                if operation.get("kind") == "reel"
                else "Stopping final-file export"
            )
            operation["updated_at"] = time.time()
            return dict(operation)

    def _is_cancel_requested(self, operation_id: str) -> bool:
        with self._lock:
            operation = self._operations.get(operation_id)
            return bool(operation and operation.get("cancel_requested"))

    def _finish_cancelled(
        self,
        operation_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        result = result or {}
        copied = int(result.get("copied") or 0)
        operation = self.get(operation_id)
        if operation.get("kind") == "reel":
            message = "Highlight reel cancelled before the final file finished"
        else:
            message = (
                f"Export cancelled · {copied} clip{'s' if copied != 1 else ''} finished"
                if copied
                else "Export cancelled before any final files finished"
            )
        terminal = {
            **result,
            "status": "cancelled",
            "phase": "cancelled",
            "eta_seconds": 0.0,
            "cancel_requested": True,
            "message": message,
            "error": None,
            "finished_at": time.time(),
        }
        self._update(operation_id, **terminal)

    def _update(self, operation_id: str, **patch) -> None:
        with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None:
                return
            operation.update(patch)
            operation["updated_at"] = time.time()

    def _run(
        self,
        operation_id: str,
        clip_ids: List[str],
        dest_folder: str,
        preset: str,
        filename_template: str,
    ) -> None:
        with self._worker_lock:
            if self._is_cancel_requested(operation_id):
                self._finish_cancelled(operation_id)
                return
            self._update(
                operation_id,
                status="running",
                phase="starting",
                message="Preparing final files",
                started_at=time.time(),
            )
            try:
                result = self.clip_service.copy_to_folder(
                    clip_ids,
                    dest_folder,
                    preset=preset,
                    filename_template=filename_template,
                    progress_callback=lambda payload: self._update(operation_id, **payload),
                    cancel_check=lambda: self._is_cancel_requested(operation_id),
                )
            except HTTPException as exc:
                detail = exc.detail
                message = detail if isinstance(detail, str) else "Recall could not prepare the final files."
                self._update(
                    operation_id,
                    status="failed",
                    phase="failed",
                    progress=1.0,
                    eta_seconds=0.0,
                    completed_clips=len(clip_ids),
                    errors=len(clip_ids),
                    error=message,
                    message=message,
                    finished_at=time.time(),
                )
                return
            except Exception:
                logger.exception("Export operation %s failed", operation_id)
                self._update(
                    operation_id,
                    status="failed",
                    phase="failed",
                    progress=1.0,
                    eta_seconds=0.0,
                    completed_clips=len(clip_ids),
                    errors=len(clip_ids),
                    error="Recall could not prepare the final files.",
                    message="Export stopped before the final files were ready",
                    finished_at=time.time(),
                )
                return

            if result.get("cancelled"):
                self._finish_cancelled(operation_id, result)
                return

            copied = int(result.get("copied") or 0)
            errors = int(result.get("errors") or 0)
            status = "completed" if errors == 0 else ("partial" if copied else "failed")
            message = (
                f"Exported {copied} clip{'s' if copied != 1 else ''}"
                if status == "completed"
                else (
                    f"Exported {copied}; {errors} clip{'s' if errors != 1 else ''} need attention"
                    if status == "partial"
                    else "No clips were exported"
                )
            )
            self._update(
                operation_id,
                **result,
                status=status,
                phase=status,
                progress=1.0,
                current_clip_progress=1.0,
                eta_seconds=0.0,
                completed_clips=len(clip_ids),
                message=message,
                error=("No clips were exported. Check the destination and try again." if status == "failed" else None),
                finished_at=time.time(),
            )

    def _run_reel(
        self,
        operation_id: str,
        clip_ids: List[str],
        dest_folder: str,
        filename: str,
    ) -> None:
        with self._worker_lock:
            if self._is_cancel_requested(operation_id):
                self._finish_cancelled(operation_id)
                return
            self._update(
                operation_id,
                status="running",
                phase="starting",
                message="Preparing highlight reel",
                started_at=time.time(),
            )
            try:
                from engines.export.renderer import FFmpegCancelled

                result = self.clip_service.compile_reel(
                    clip_ids,
                    dest_folder,
                    filename=filename,
                    progress_callback=lambda payload: self._update(operation_id, **payload),
                    cancel_check=lambda: self._is_cancel_requested(operation_id),
                )
            except FFmpegCancelled:
                self._finish_cancelled(operation_id)
                return
            except HTTPException as exc:
                detail = exc.detail
                result = {
                    "error": detail if isinstance(detail, str) else "Recall could not prepare the highlight reel.",
                    "errors": len(clip_ids),
                    "failed": [],
                    "succeeded_clip_ids": [],
                }
            except Exception:
                logger.exception("Reel operation %s failed", operation_id)
                result = {
                    "error": "Recall could not prepare the highlight reel.",
                    "errors": len(clip_ids),
                    "failed": [],
                    "succeeded_clip_ids": [],
                }

            error = result.get("error")
            if error or not result.get("path"):
                message = str(error or "Reel stopped before the final file was ready")
                terminal = {
                    **result,
                    "status": "failed",
                    "phase": "failed",
                    "progress": 1.0,
                    "reel_progress": 1.0,
                    "eta_seconds": 0.0,
                    "completed_clips": len(clip_ids),
                    "errors": max(1, int(result.get("errors") or 0)),
                    "error": message,
                    "message": message,
                    "finished_at": time.time(),
                }
                self._update(operation_id, **terminal)
                return

            clips = int(result.get("clips") or 0)
            errors = int(result.get("errors") or 0)
            status = "partial" if errors else "completed"
            message = (
                f"Highlight reel ready · {clips} clip{'s' if clips != 1 else ''}"
                if not errors
                else f"Highlight reel ready · {clips} included, {errors} skipped"
            )
            terminal = {
                **result,
                "status": status,
                "phase": status,
                "progress": 1.0,
                "current_clip_progress": 1.0,
                "reel_progress": 1.0,
                "eta_seconds": 0.0,
                "completed_clips": len(clip_ids),
                "copied": clips,
                "message": message,
                "error": None,
                "finished_at": time.time(),
            }
            self._update(operation_id, **terminal)

    def _prune_locked(self) -> None:
        if len(self._operations) < self._MAX_OPERATIONS:
            return
        terminal = sorted(
            (
                operation for operation in self._operations.values()
                if operation.get("status") in self._TERMINAL
            ),
            key=lambda operation: float(operation.get("updated_at") or 0.0),
        )
        for operation in terminal[:max(1, len(self._operations) - self._MAX_OPERATIONS + 1)]:
            self._operations.pop(operation["id"], None)


class SettingsService:
    """Creator settings persisted in SQLite (plan 3.1) so scan preferences
    survive restarts and agree between the desktop app and a browser tab.
    Values are opaque JSON; the frontend owns the schema."""

    # Whitelist so a bad client can't grow the table without bound.
    KNOWN_KEYS = {
        "displayName",
        "processingMode",
        # How much of the machine a scan may use (process governor). The
        # frontend has always sent this in PERSISTED_SETTING_KEYS; it was
        # missing here, so update() silently dropped it and the creator's
        # scan-speed choice never survived a restart.
        "performanceProfile",
        "exportLayout",
        "captionStyle",
        "theme",
        "recallStorageCapGb",
        "sourceRetentionDays",
    }

    def __init__(self, db: DatabaseManager):
        self.db = db

    def get_all(self) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings")
            rows = cursor.fetchall()
        out: Dict[str, Any] = {}
        for row in rows:
            if row["key"] not in self.KNOWN_KEYS:
                continue
            try:
                out[row["key"]] = json.loads(row["value"])
            except (ValueError, TypeError):
                pass
        return out

    def update(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in (patch or {}).items():
            if key in self.KNOWN_KEYS:
                if key == "recallStorageCapGb":
                    try:
                        value = max(1.0, min(2000.0, float(value)))
                    except (TypeError, ValueError):
                        continue
                elif key == "sourceRetentionDays":
                    try:
                        value = max(1, min(365, int(round(float(value)))))
                    except (TypeError, ValueError):
                        continue
                self.db.set_setting(key, value)
        return self.get_all()


class SystemService:
    def __init__(self, db: DatabaseManager, job_manager: Optional[JobManager] = None):
        self.db = db
        self.job_manager = job_manager
        self.data_root = get_data_dir()
        self.lifecycle = SourceLifecycleService(db, job_manager, self.data_root)

    def _require_idle(self, operation: str) -> None:
        if self.job_manager is not None and self.job_manager.has_active_jobs():
            raise HTTPException(
                status_code=409,
                detail=f"Cannot {operation} while a scan is queued or running. Cancel it first.",
            )

    def _folder_size(self, path: str) -> int:
        total = 0
        if not os.path.exists(path):
            return total
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                fp = os.path.join(dirpath, filename)
                try:
                    if not os.path.islink(fp):
                        total += os.path.getsize(fp)
                except OSError:
                    logger.debug("Could not stat storage item %s", fp, exc_info=True)
        return total

    def _clear_folder(self, folder: str) -> int:
        path = os.path.join(self.data_root, folder)
        bytes_removed = self._folder_size(path)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        return bytes_removed

    def clear_cache(self) -> Dict[str, Any]:
        self._require_idle("clear the scan cache")
        try:
            result = self.lifecycle.clear_safe_cache()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {
            "status": "success",
            "message": "Scan cache cleared.",
            "bytes_removed": result["bytes_removed"],
        }

    def storage(self) -> Dict[str, Any]:
        """Truthful Recall-vs-source inventory; opening this view reconciles it."""
        return self.lifecycle.storage_report()

    def probe(self, source_path: str, source_type: str = "file") -> Dict[str, Any]:
        """Pre-scan probe for the import preview: real duration + a poster frame.

        Twitch probes use the bundled downloader for public metadata and create a
        short high-quality preview when the complete VOD is not already cached.
        Never raises; a failed probe just returns the fields that were available.
        """
        result: Dict[str, Any] = {
            "duration": None, "thumbnail": None, "title": None,
            "creator": None, "game": None, "vod_id": None, "source_date": None,
        }
        if not source_path:
            return result

        if source_type in ("url", "twitch") or source_path.startswith(("http://", "https://")):
            return self._probe_twitch(source_path, result)
        if source_type != "file" or not os.path.isfile(source_path):
            return result

        from core.source_date import local_recording_date

        result["source_date"] = local_recording_date(source_path)

        import re as _re
        import base64 as _base64
        import hashlib as _hashlib
        import shutil as _shutil
        import subprocess as _subprocess
        from core.ffmpeg_path import get_ffmpeg_path
        from core.thumbnails import generate_thumbnail

        # Resolve to an absolute ffmpeg so the probe doesn't depend on this
        # process inheriting a PATH that happens to include ffmpeg's dir.
        ffmpeg = get_ffmpeg_path()
        if ffmpeg == "ffmpeg":
            ffmpeg = _shutil.which("ffmpeg") or ffmpeg
        # Duration: ffmpeg prints "Duration: HH:MM:SS.ss" to stderr when probing.
        try:
            proc = _subprocess.run(
                [ffmpeg, "-i", source_path, "-hide_banner"],
                capture_output=True, text=True, timeout=30,
            )
            match = _re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
            if match:
                hours, minutes, seconds = match.groups()
                result["duration"] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except Exception:
            logger.debug("Source duration probe failed for %s", source_path, exc_info=True)

        # Poster ~10% in (clamped), reusing the clip thumbnail grabber, then
        # inlined as a data URL so no extra static route is needed.
        try:
            temp_dir = os.path.join(self.data_root, "temp")
            seek = min(max(2.0, (result["duration"] or 20.0) * 0.1), 60.0)
            probe_id = "probe_" + _hashlib.md5(
                source_path.encode("utf-8", "ignore"), usedforsecurity=False
            ).hexdigest()[:12]
            thumb = generate_thumbnail(
                source_path, probe_id, temp_dir, seek_seconds=seek,
                width=1280, jpeg_quality=2,
            )
            if thumb and os.path.exists(thumb):
                with open(thumb, "rb") as handle:
                    encoded = _base64.b64encode(handle.read()).decode("ascii")
                result["thumbnail"] = "data:image/jpeg;base64," + encoded
                try:
                    os.remove(thumb)
                except OSError:
                    logger.debug("Could not remove temporary probe thumbnail %s", thumb, exc_info=True)
        except Exception:
            logger.debug("Source thumbnail probe failed for %s", source_path, exc_info=True)

        return result

    def _probe_twitch(self, source_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort Twitch title/duration/poster probe without app secrets."""
        import base64 as _base64
        import re as _re
        import shutil as _shutil
        import subprocess as _subprocess
        import tempfile as _tempfile

        from core.ffmpeg_path import get_ffmpeg_path
        from core.thumbnails import generate_thumbnail
        from core.twitchdl_path import get_twitchdl_path
        from engines.chat.twitch_chat import fetch_twitch_video_meta

        match = _re.search(r"twitch\.tv/videos/(\d+)", source_path, _re.IGNORECASE)
        if not match:
            return result
        vod_id = match.group(1)
        result["vod_id"] = vod_id
        temp_root = os.path.join(self.data_root, "temp")
        os.makedirs(temp_root, exist_ok=True)

        try:
            meta = fetch_twitch_video_meta(source_path, timeout=45.0, temp_dir=temp_root) or {}
            result["title"] = meta.get("title")
            result["creator"] = meta.get("streamer")
            result["game"] = meta.get("game")
            result["duration"] = meta.get("length")
            from core.source_date import normalize_source_date

            result["source_date"] = normalize_source_date(meta.get("created_at"))
        except Exception:
            logger.debug("Twitch metadata probe failed for %s", source_path, exc_info=True)

        asset_path = None
        for extension in (".mp4", ".mkv", ".webm"):
            candidate = os.path.join(self.data_root, "assets", f"twitchvod_v{vod_id}{extension}")
            if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                asset_path = candidate
                break

        probe_dir = None
        try:
            if asset_path is None:
                exe = get_twitchdl_path()
                if not exe:
                    return result
                probe_dir = _tempfile.mkdtemp(prefix=f"recall_twitch_probe_{vod_id}_", dir=temp_root)
                asset_path = os.path.join(probe_dir, "preview.mp4")
                cmd = [
                    exe, "videodownload", "-u", vod_id, "-o", asset_path,
                    "-q", "best", "-b", "0", "-e", "3",
                    "--temp-path", os.path.join(probe_dir, "parts"),
                    "--collision", "Overwrite", "--banner", "false",
                ]
                ffmpeg = get_ffmpeg_path()
                if ffmpeg and not os.path.isabs(ffmpeg):
                    ffmpeg = _shutil.which(ffmpeg)
                if ffmpeg and os.path.isabs(ffmpeg) and os.path.exists(ffmpeg):
                    cmd += ["--ffmpeg-path", ffmpeg]
                completed = _subprocess.run(
                    cmd, capture_output=True, text=True, timeout=45,
                    creationflags=getattr(_subprocess, "CREATE_NO_WINDOW", 0),
                )
                if completed.returncode != 0 or not os.path.isfile(asset_path):
                    return result

            thumb = generate_thumbnail(
                asset_path, f"twitch_probe_{vod_id}", temp_root, seek_seconds=1.0,
                width=1280, jpeg_quality=2,
            )
            if thumb and os.path.exists(thumb):
                with open(thumb, "rb") as handle:
                    encoded = _base64.b64encode(handle.read()).decode("ascii")
                result["thumbnail"] = "data:image/jpeg;base64," + encoded
                try:
                    os.remove(thumb)
                except OSError:
                    pass
        except Exception:
            logger.debug("Twitch poster probe failed for %s", source_path, exc_info=True)
        finally:
            if probe_dir:
                _shutil.rmtree(probe_dir, ignore_errors=True)
        return result

    def prune_asset_retention(self) -> Dict[str, Any]:
        """Compatibility shim for old startup callers."""
        source_result = self.lifecycle.reconcile_sources(remove_expired=True)
        recall_result = self.lifecycle.reconcile_recall_storage()
        return {
            "status": "success",
            "bytes_removed": source_result["bytes_removed"]
            + recall_result["cache_bytes_removed"]
            + recall_result["temp_bytes_removed"],
        }

    def clear_assets(self) -> Dict[str, Any]:
        """Remove only sources that are already eligible under the aging policy."""
        self._require_idle("delete downloaded VODs")
        bytes_removed = 0
        for source in self.lifecycle.list_sources():
            if source["state"] == "safe_to_remove":
                bytes_removed += self.lifecycle.remove_source(
                    source["source_key"], require_eligible=True
                )["bytes_removed"]
        return {
            "status": "success",
            "message": "Eligible downloaded sources removed.",
            "bytes_removed": bytes_removed,
        }

    def source_status_for_job(self, job_id: str) -> Dict[str, Any]:
        try:
            return self.lifecycle.source_status_for_job(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found")

    def remove_source(self, source_key: str) -> Dict[str, Any]:
        try:
            return self.lifecycle.remove_source(source_key, require_eligible=True)
        except KeyError:
            raise HTTPException(status_code=404, detail="Downloaded source not found")
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    def pin_source(self, source_key: str, pinned: bool) -> Dict[str, Any]:
        try:
            return self.lifecycle.set_pinned(source_key, pinned)
        except KeyError:
            raise HTTPException(status_code=404, detail="Downloaded source not found")

    def restore_source(self, source_key: str) -> Dict[str, Any]:
        try:
            result = self.lifecycle.restore_source(source_key)
        except KeyError:
            raise HTTPException(status_code=404, detail="Downloaded source not found")
        if result["status"] == "blocked_low_disk":
            raise HTTPException(status_code=507, detail=result)
        if result["status"] == "blocked_active_scan":
            raise HTTPException(
                status_code=409,
                detail="Wait for the active scan to finish before restoring source media.",
            )
        return result

    def restore_source_for_job(self, job_id: str) -> Dict[str, Any]:
        status = self.source_status_for_job(job_id)
        if not status.get("source_key"):
            raise HTTPException(status_code=409, detail="Local creator recordings cannot be restored by Recall.")
        return self.restore_source(status["source_key"])

    def cancel_restore(self, source_key: str) -> Dict[str, Any]:
        return self.lifecycle.cancel_restore(source_key)

    def wipe(self) -> Dict[str, Any]:
        self._require_idle("wipe Recall data")
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM source_asset_sessions")
            cursor.execute("DELETE FROM source_assets")
            cursor.execute("DELETE FROM exports")
            cursor.execute("DELETE FROM clips")
            cursor.execute("DELETE FROM job_events")
            cursor.execute("DELETE FROM reaction_timelines")
            cursor.execute("DELETE FROM jobs")
            # Recall Session events/anchors cascade from their session. They
            # are creator data, not cache, so wipe them explicitly after jobs
            # have dropped their optional session reference.
            cursor.execute("DELETE FROM recall_sessions")
            # Learned personalization is derived from the clips being wiped, and
            # its rows key on clip ids that no longer exist. "Wipe all data"
            # must clear it too, or the tables dangle with orphaned references.
            cursor.execute("DELETE FROM clip_labels")
            cursor.execute("DELETE FROM ranker_metadata")
            cursor.execute("DELETE FROM stream_memory_shadow_runs")
            cursor.execute("DELETE FROM stream_memory_interactions")
            cursor.execute("DELETE FROM stream_memory_search_sessions")
            cursor.execute(
                """UPDATE stream_memory_semantic_state
                   SET source_revision = source_revision + 1,
                       indexed_revision = -1,
                       status = 'not_built', model_id = NULL, dimensions = 0,
                       indexed_entries = 0, index_bytes = 0, indexed_at = NULL,
                       last_error = NULL
                   WHERE id = 1"""
            )
        # Include "assets" (downloaded VODs): the old wipe left multi-GB VOD
        # files on disk while deleting every jobs row that referenced them,
        # so nothing could ever reclaim them. Drop the per-install ranker
        # model too so personalization really resets.
        for folder in ("temp", "exports", "cache", "jobs", "assets", "memory"):
            self._clear_folder(folder)
        # Resolve against THIS service's data root, not the import-time
        # constant: a caller holding a redirected root must not have the real
        # install's champion deleted out from under it.
        from core.ranker_service import user_model_path_for_root
        champion_path = user_model_path_for_root(self.data_root)
        if os.path.exists(champion_path):
            os.remove(champion_path)
        return {"status": "success", "message": "All data wiped."}
