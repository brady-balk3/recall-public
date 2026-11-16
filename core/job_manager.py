# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import asyncio
import threading
import uuid
import json
import math
import time
from typing import Dict, Any, Optional
from core.database import DatabaseManager
from core.bundle_paths import get_data_dir
from core.artifacts import write_candidates_artifact, write_job_artifacts

import os
import traceback


from core.timefmt import format_hms


def format_vod_time(seconds: float) -> str:
    """Hour-aware display timestamp. Thin alias over core.timefmt.format_hms,
    kept for the existing import sites (services.py, this module)."""
    return format_hms(seconds)


# Legacy review-order band retained for database ordering and older event
# consumers. Creator-facing UI must not present this value as confidence.
DECK_SCORE_TOP = 0.99
DECK_SCORE_FLOOR = 0.58
DECK_SCORE_FLAT = 0.85  # every clip scored identically -> no ordering info


def deck_percentile_scores(clips) -> dict:
    """Map clip ids to a legacy review-order scalar in the 0.58..0.99 band.

    The spacing preserves the deck's real selection-score gaps, min-max scaled
    into the historical storage band. It remains useful for stable ordering,
    but it is not a probability or a creator-facing score. Ties share a value;
    a one-clip deck receives the top ordering scalar.
    """
    clips = list(clips or [])
    if not clips:
        return {}
    if len(clips) == 1:
        return {clips[0].clip_id: DECK_SCORE_TOP}

    values = [float(getattr(clip, "score", 0.0) or 0.0) for clip in clips]
    low, high = min(values), max(values)
    if high - low <= 1e-9:
        return {clip.clip_id: DECK_SCORE_FLAT for clip in clips}
    span = DECK_SCORE_TOP - DECK_SCORE_FLOOR
    return {
        clip.clip_id: round(
            DECK_SCORE_FLOOR + span * (value - low) / (high - low), 4,
        )
        for clip, value in zip(clips, values)
    }

# User-friendly labels for the raw pipeline phase names. Kept backend-side so
# the SSE stream and the /jobs payload speak the same language as the UI.
STAGE_LABELS = {
    "Init": "Getting source ready",
    "Resolving Input": "Getting source ready",
    "Perception": "Scanning signals",
    "Reaction": "Finding moments",
    "Event": "Finding moments",
    "Story": "Finding moments",
    "Clip": "Finding moments",
    "Caption": "Preparing captions",
    "Export": "Rendering clips",
    "Complete": "Ready to review",
    "Cancelled": "Cancelled",
    "Cancelling": "Cancelling",
}

# Scan position, VOD length, stage budget and download share arrive as numbers
# (the pipeline's telemetry dict, resolve_input's download_callback). Progress
# messages are display text only: nothing here parses them.

STAGE_WINDOWS = {
    "Init": (0.0, 0.05),
    # Ends where the pipeline's first emit begins (vod_resolver's
    # RESOLVE_DONE_PROGRESS), so the download never overlaps Perception.
    "Resolving Input": (0.0, 0.05),
    "Perception": (0.08, 0.45),
    "Reaction": (0.45, 0.70),
    "Event": (0.45, 0.55),
    "Story": (0.55, 0.65),
    "Clip": (0.65, 0.75),
    "Caption": (0.75, 0.85),
    "Export": (0.85, 0.98),
    "Complete": (0.98, 1.0),
}

PIPELINE_PHASE_ORDER = (
    "Init", "Resolving Input", "Perception", "Event", "Story",
    "Reaction", "Clip", "Caption", "Export", "Complete",
)

PERSISTED_PROGRESS_INTERVAL_SECONDS = 1.0
PERSISTED_PROGRESS_DELTA = 0.01
PERSISTED_PROGRESS_EVENTS_PER_JOB = 600
PERSISTED_PROGRESS_TRIM_INTERVAL = 100
LIVE_STAGE_ETA_MAX_AGE_SECONDS = 30.0


# Events the live scan view consumes as they stream; kept out of the replayed
# recent_events window so the scan log stays readable.
LIVE_ONLY_EVENTS = ("candidate_verdict",)
# What the live strip needs to redraw after a reload: the newest curve and
# lanes, and every candidate event in order (a verdict only means something
# after its candidates_ready).
LATEST_STATE_EVENTS = ("timeline_ready", "scan_lanes")
CANDIDATE_EVENTS = ("candidates_ready", "candidate_verdict", "clip_captioned")


class JobManager:
    def __init__(self, db: DatabaseManager):
        self.db = db
        # In-memory tracking of active threads/cancellation flags could go here
        self._active_jobs = {}
        # Live telemetry per job_id (populated while a job runs). Fields not in
        # the DB, e.g. ETA samples and derived scan speed.
        self._telemetry: Dict[str, Dict[str, Any]] = {}
        # The scan thread, a request thread cancelling it, the preview-render
        # pool and every get_job all touch _telemetry. This lock covers its
        # compound updates and snapshots only: it is never held across a DB
        # write, a subprocess or event publishing, and no other lock is taken
        # while it is held (nor is it taken under _sub_lock), so the two can
        # never deadlock each other.
        self._telemetry_lock = threading.Lock()
        # HTTP SSE uses event-loop-owned asyncio queues so an idle stream never
        # occupies Starlette's shared worker threadpool. Pipeline work publishes
        # from ordinary worker threads via loop.call_soon_threadsafe().
        self._async_subscribers: Dict[str, Dict[int, tuple]] = {}
        self._next_async_subscriber_id = 0
        self._last_persisted_progress: Dict[str, tuple] = {}
        self._persisted_progress_since_trim: Dict[str, int] = {}
        self._sub_lock = threading.Lock()
        # Cached per-stage duration rates from past completed jobs (ETA model).
        self._stage_rate_cache: Dict[tuple, tuple] = {}
        # One pipeline runs at a time; extra jobs queue (FIFO, plan 6.3) instead
        # of thrashing the CPU workers and each other's caches.
        self._pipeline_sem = threading.Semaphore(1)
        self._sweep_interrupted_jobs()

    def _sweep_interrupted_jobs(self):
        """Jobs left 'running' by a crash or app close can never resume — mark
        them failed at startup so the DB (and every UI listing) tells the truth."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM jobs WHERE status IN ('running', 'pending', 'cancelling')"
                )
                interrupted_job_ids = [row["id"] for row in cursor.fetchall()]
                cursor.execute(
                    '''UPDATE jobs SET status = 'failed', current_stage = 'Failed',
                       message = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE status IN ('running', 'pending', 'cancelling')''',
                    ("Interrupted — the studio was closed mid-scan.",)
                )
                for job_id in interrupted_job_ids:
                    self._trim_persisted_progress(cursor, job_id)
        except Exception:
            traceback.print_exc()

    def is_active(self, job_id: str) -> bool:
        """True while this process has a live pipeline thread for the job."""
        return job_id in self._active_jobs

    def has_active_jobs(self, excluding: Optional[str] = None) -> bool:
        """True while any queued or running pipeline belongs to this process,
        other than ``excluding`` (a job whose scan has already completed)."""
        if excluding is None:
            return bool(self._active_jobs)
        # list() copies the keys in one step; another thread may add a job.
        return any(job_id != excluding for job_id in list(self._active_jobs))

    # --- SSE subscription plumbing -------------------------------------------
    def subscribe_async(self, job_id: str, max_events: int = 256):
        """Return an event-loop-owned, bounded subscriber for HTTP SSE.

        The pipeline publishes from background threads, so callers must create
        this subscriber from the loop that will consume it. A small integer id
        makes cleanup deterministic without relying on queue identity.
        """
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(max_events)))
        with self._sub_lock:
            self._next_async_subscriber_id += 1
            subscriber_id = self._next_async_subscriber_id
            self._async_subscribers.setdefault(job_id, {})[subscriber_id] = (loop, q)
        return subscriber_id, q

    def unsubscribe_async(self, job_id: str, subscriber_id: int):
        with self._sub_lock:
            subs = self._async_subscribers.get(job_id)
            if subs:
                subs.pop(subscriber_id, None)
                if not subs:
                    self._async_subscribers.pop(job_id, None)

    @staticmethod
    def _offer_async_event(q: "asyncio.Queue", event: dict):
        """Deliver without blocking, preserving semantic events when possible."""
        try:
            if q.full():
                buffered = []
                dropped_progress = False
                while True:
                    try:
                        queued = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if not dropped_progress and queued.get("event_type") == "stage_progress":
                        dropped_progress = True
                        continue
                    buffered.append(queued)

                incoming_is_progress = event.get("event_type") == "stage_progress"
                if not dropped_progress and incoming_is_progress:
                    for queued in buffered:
                        q.put_nowait(queued)
                    return
                if not dropped_progress and buffered:
                    buffered.pop(0)
                for queued in buffered:
                    q.put_nowait(queued)
            q.put_nowait(event)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            # Telemetry is best-effort. The next snapshot reconciles state.
            pass

    def _publish(self, job_id: str, event: dict):
        with self._sub_lock:
            async_subscribers = list(self._async_subscribers.get(job_id, {}).values())
        for loop, q in async_subscribers:
            try:
                loop.call_soon_threadsafe(self._offer_async_event, q, event)
            except RuntimeError:
                # The renderer disconnected while a worker was publishing.
                pass

    def emit_event(self, job_id: str, event_type: str, phase: str = None,
                   message: str = None, progress: float = None, payload: dict = None):
        """Persist a job event and fan it out to any live SSE subscribers."""
        payload_json = json.dumps(payload) if payload else None
        persist = self._should_persist_event(job_id, event_type, phase, progress)
        if persist:
            try:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        '''INSERT INTO job_events (job_id, event_type, phase, message, progress, payload)
                           VALUES (?, ?, ?, ?, ?, ?)''',
                        (job_id, event_type, phase, message, progress, payload_json)
                    )
                    if event_type == "stage_progress":
                        since_trim = self._persisted_progress_since_trim.get(job_id, 0) + 1
                        if since_trim >= PERSISTED_PROGRESS_TRIM_INTERVAL:
                            self._trim_persisted_progress(cursor, job_id)
                            since_trim = 0
                        self._persisted_progress_since_trim[job_id] = since_trim
                    if event_type in ("job_completed", "job_failed", "job_cancelled"):
                        self._trim_persisted_progress(cursor, job_id)
            except Exception:
                # Telemetry must never take down a running job.
                traceback.print_exc()
        if event_type in ("job_completed", "job_failed", "job_cancelled"):
            self._last_persisted_progress.pop(job_id, None)
            self._persisted_progress_since_trim.pop(job_id, None)
        event = {
            "job_id": job_id,
            "event_type": event_type,
            "phase": phase,
            "stage_label": STAGE_LABELS.get(phase, phase) if phase else None,
            "message": message,
            "progress": progress,
            "payload": payload,
            "created_at": time.time(),
        }
        self._publish(job_id, event)

    def _should_persist_event(self, job_id: str, event_type: str,
                              phase: Optional[str], progress: Optional[float]) -> bool:
        """Coalesce numeric progress heartbeats while preserving live delivery."""
        if event_type != "stage_progress" or progress is None:
            return True
        now = time.monotonic()
        try:
            current = float(progress)
        except (TypeError, ValueError):
            return True
        previous = self._last_persisted_progress.get(job_id)
        if previous:
            previous_at, previous_phase, previous_progress = previous
            if (
                phase == previous_phase
                and now - previous_at < PERSISTED_PROGRESS_INTERVAL_SECONDS
                and abs(current - previous_progress) < PERSISTED_PROGRESS_DELTA
            ):
                return False
        self._last_persisted_progress[job_id] = (now, phase, current)
        return True

    @staticmethod
    def _trim_persisted_progress(cursor, job_id: str):
        """Keep operational progress bounded; retain every meaningful event."""
        cursor.execute(
            "DELETE FROM job_events WHERE job_id = ? AND event_type = 'stage_progress' "
            "AND id NOT IN ("
            "SELECT id FROM job_events WHERE job_id = ? AND event_type = 'stage_progress' "
            "ORDER BY id DESC LIMIT ?)",
            (job_id, job_id, PERSISTED_PROGRESS_EVENTS_PER_JOB),
        )

    def _scan_state_events(self, job_id: str) -> list:
        """The live strip's state, oldest first, for a view opened mid-scan."""
        latest = []
        for kind in LATEST_STATE_EVENTS:
            latest.extend(self.get_job_events(job_id, limit=1, event_types=(kind,)))
        candidates = self.get_job_events(job_id, limit=800, event_types=CANDIDATE_EVENTS)
        return latest + list(reversed(candidates))

    def get_job_events(self, job_id: str, limit: int = 60,
                       event_types=None, exclude_types=None) -> list:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            params = [job_id]
            type_filter = ""
            if event_types:
                event_types = tuple(event_types)
                placeholders = ",".join("?" for _ in event_types)
                type_filter = f" AND event_type IN ({placeholders})"
                params.extend(event_types)
            if exclude_types:
                exclude_types = tuple(exclude_types)
                placeholders = ",".join("?" for _ in exclude_types)
                type_filter += f" AND event_type NOT IN ({placeholders})"
                params.extend(exclude_types)
            params.append(limit)
            cursor.execute(
                "SELECT job_id, event_type, phase, message, progress, payload, created_at "
                "FROM job_events WHERE job_id = ?" + type_filter
                + " ORDER BY id DESC LIMIT ?",
                tuple(params),
            )
            rows = []
            for row in cursor.fetchall():
                d = dict(row)
                d["stage_label"] = STAGE_LABELS.get(d.get("phase"), d.get("phase"))
                if d.get("payload"):
                    try:
                        d["payload"] = json.loads(d["payload"])
                    except Exception:
                        d["payload"] = None
                rows.append(d)
            return rows

    def create_job(
        self,
        source_path: str,
        source_type: str = "file",
        session_name: str = None,
        source_date: str = None,
        *,
        recall_session_id: str = None,
        region_plan: dict = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        with self.db.get_connection() as conn:
            conn.cursor().execute(
                '''INSERT INTO jobs (
                       id, status, session_name, source_type, source_path, source_date,
                       recall_session_id, region_plan
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    job_id, 'pending', session_name, source_type, source_path, source_date,
                    recall_session_id,
                    json.dumps(region_plan, separators=(",", ":")) if region_plan else None,
                )
            )
        return job_id

    def _row_ts(self, value) -> Optional[float]:
        """SQLite stores CURRENT_TIMESTAMP as UTC 'YYYY-MM-DD HH:MM:SS'."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            import calendar
            t = time.strptime(str(value).split(".")[0], "%Y-%m-%d %H:%M:%S")
            return calendar.timegm(t)
        except Exception:
            return None

    def get_job(self, job_id: str, include_events: bool = True) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Every column except framing, the 1-3 MB scan model that only the
            # manual-framing path reads (and selects for itself). This row
            # backs /jobs/{id} and every SSE snapshot, so SELECT * shipped
            # megabytes per poll. Read from the schema rather than listed by
            # hand so a column a later migration adds still reaches the job
            # payload, as it did under SELECT *.
            cursor.execute('PRAGMA table_info(jobs)')
            columns = ', '.join(
                '"' + str(info[1]).replace('"', '""') + '"'
                for info in cursor.fetchall() if info[1] != 'framing'
            )
            cursor.execute(f'SELECT {columns} FROM jobs WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            job = dict(row)

        # --- Derived / live telemetry, additive to the stored columns --------
        # A snapshot: the response is serialized after this returns, while the
        # scan thread may still be adding stages to the live timings dict.
        with self._telemetry_lock:
            tel = dict(self._telemetry.get(job_id, {}))
            if isinstance(tel.get("stage_timings"), dict) or tel.get("stage_detail"):
                tel["stage_timings"] = self._timings_with_detail(tel)
        job["stage_label"] = STAGE_LABELS.get(job.get("current_stage"), job.get("current_stage"))

        started = self._row_ts(job.get("started_at")) or tel.get("started_at")
        updated = self._row_ts(job.get("updated_at")) or tel.get("updated_at")
        status = job.get("status")
        now = time.time()
        if started:
            end = now if status in ("running", "pending", "cancelling") else (updated or now)
            job["elapsed_seconds"] = max(0.0, round(end - started, 1))
        else:
            job["elapsed_seconds"] = tel.get("elapsed_seconds")

        job["message"] = job.get("message") or tel.get("message")
        job["duration"] = job.get("duration") or tel.get("duration")
        job["stage_progress"] = tel.get("stage_progress", job.get("stage_progress"))
        job["scanned_seconds"] = tel.get("scanned_seconds", job.get("scanned_seconds"))
        job["scan_speed"] = tel.get("scan_speed")
        job["active_workers"] = tel.get("active_workers")
        job["total_workers"] = tel.get("total_workers")
        job["transcribed_seconds"] = tel.get("transcribed_seconds")
        job["transcription_speed"] = tel.get("transcription_speed")
        job["download_progress"] = tel.get("download_progress")
        job["clips_found"] = tel.get("clips_found", job.get("clips_found")) or 0
        # Prefer live telemetry; fall back to the DB-persisted JSON for jobs
        # with no in-memory telemetry (e.g. any completed job after a restart).
        # Capture the stored value BEFORE overwriting it — otherwise the
        # fallback re-reads the just-cleared field and always loses the DB copy.
        stored_timings = job.get("stage_timings")
        job["stage_timings"] = tel.get("stage_timings")
        if not job["stage_timings"] and stored_timings:
            try:
                job["stage_timings"] = json.loads(stored_timings)
            except Exception:
                job["stage_timings"] = {}
        stored_profile = job.get("scan_profile")
        job["scan_profile"] = tel.get("scan_profile")
        if not job["scan_profile"] and stored_profile:
            try:
                job["scan_profile"] = json.loads(stored_profile)
            except Exception:
                job["scan_profile"] = None
        stored_region_plan = job.get("region_plan")
        if stored_region_plan:
            try:
                job["region_plan"] = json.loads(stored_region_plan)
            except (TypeError, ValueError):
                job["region_plan"] = None

        eta_seconds, eta_confidence = self._compute_eta(job_id, job)
        job["eta_seconds"] = eta_seconds
        job["eta_confidence"] = eta_confidence
        job["eta_reliable"] = eta_confidence in ("medium", "high")
        job["scan_health"] = self._scan_health(job_id, tel)

        if include_events:
            # Per-candidate vision verdicts arrive by the dozen; they are for the
            # live strip and would push the scan's real story out of this window.
            job["recent_events"] = self.get_job_events(
                job_id, limit=40, exclude_types=LIVE_ONLY_EVENTS,
            )
            job["scan_state_events"] = self._scan_state_events(job_id)
        return job

    def _scan_health(self, job_id: str, tel: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Creator-visible scan health: semantic/visual judges + device.

        Live jobs read in-memory telemetry. Completed jobs (or after restart)
        fall back to the latest persisted judge events so Studio still knows
        whether a finished deck ran degraded.
        """
        tel = tel or self._telemetry.get(job_id, {}) or {}
        semantic = dict(tel.get("semantic_judge_health") or {})
        visual = dict(tel.get("visual_judge_health") or {})
        if not semantic:
            events = self.get_job_events(
                job_id,
                limit=1,
                event_types=("semantic_judge_health", "semantic_judge_degraded"),
            )
            if events:
                semantic = dict(events[0].get("payload") or {})
                if events[0].get("event_type") == "semantic_judge_degraded":
                    semantic.setdefault("degraded", True)
                    semantic.setdefault("status", "degraded")
        if not visual:
            events = self.get_job_events(
                job_id,
                limit=1,
                event_types=(
                    "visual_judge_health",
                    "visual_judge_degraded",
                    "visual_judge_skipped",
                ),
            )
            if events:
                visual = dict(events[0].get("payload") or {})
                event_type = events[0].get("event_type")
                if event_type == "visual_judge_degraded":
                    visual.setdefault("degraded", True)
                    visual.setdefault("status", "degraded")
                elif event_type == "visual_judge_skipped":
                    visual.setdefault("degraded", True)
                    visual.setdefault("status", "skipped")
        if semantic and "status" not in semantic:
            semantic["status"] = "degraded" if semantic.get("degraded") else "healthy"
        if visual and "status" not in visual:
            visual["status"] = "degraded" if visual.get("degraded") else "healthy"
        device = (
            semantic.get("device")
            or visual.get("device")
            or tel.get("scan_device")
        )
        if not device:
            try:
                # A label read on every job fetch in the idle API process:
                # device_hint never creates a CUDA context, where the full
                # GPU gate's probe would hold one (VRAM) until the app exits.
                from core.device import device_hint
                device = device_hint()
            except Exception:  # noqa: BLE001
                device = None
        semantic_bad = bool(semantic) and (
            semantic.get("degraded") or semantic.get("status") == "degraded"
        )
        visual_bad = bool(visual) and visual.get("status") in {"degraded", "skipped"}
        return {
            "semantic": semantic or None,
            "visual": visual or None,
            "device": device,
            "degraded": bool(semantic_bad or visual_bad),
        }

    def _historical_stage_rates(self, tel: dict = None) -> Dict[str, float]:
        """Median seconds-of-work per second-of-VOD for each pipeline stage,
        measured from this machine's recent completed jobs. Cached ~5 min."""
        now = time.time()
        tel = tel or {}
        profile = tel.get("scan_profile") or {}
        profile_key = (
            str(profile.get("mode") or "unknown"),
            bool(profile.get("facecam_tracking", True)),
        )
        cached = self._stage_rate_cache.get(profile_key)
        if cached and now - cached[0] < 300:
            tel["eta_history_matched"] = cached[2]
            return cached[1]
        samples: Dict[str, list] = {}
        candidates = []
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT stage_timings, duration, scan_profile FROM jobs
                       WHERE status = 'completed' AND stage_timings IS NOT NULL
                         AND duration > 0
                       ORDER BY created_at DESC LIMIT 8'''
                )
                for row in cursor.fetchall():
                    try:
                        timings = json.loads(row["stage_timings"]) or {}
                        vod = float(row["duration"])
                    except (ValueError, TypeError):
                        continue
                    try:
                        prior_profile = json.loads(row["scan_profile"]) if row["scan_profile"] else {}
                    except (ValueError, TypeError):
                        prior_profile = {}
                    prior_key = (
                        str(prior_profile.get("mode") or "unknown"),
                        bool(prior_profile.get("facecam_tracking", True)),
                    )
                    candidates.append((prior_key, timings, vod))
                matched = [item for item in candidates if item[0] == profile_key]
                selected = matched or candidates
                tel["eta_history_matched"] = bool(matched)
                for _, timings, vod in selected:
                    for phase, t in timings.items():
                        secs = (t or {}).get("duration_seconds")
                        if secs and vod > 0:
                            samples.setdefault(phase, []).append(float(secs) / vod)
        except Exception:
            traceback.print_exc()
        rates = {}
        for phase, vals in samples.items():
            vals.sort()
            rates[phase] = vals[len(vals) // 2]
        self._stage_rate_cache[profile_key] = (
            now, rates, bool(tel.get("eta_history_matched"))
        )
        return rates

    def scan_rates(self, mode: str = "quality", facecam_tracking: bool = True) -> Dict[str, Any]:
        """This machine's measured scan cost, for estimating a scan before it
        starts: wall seconds per second of VOD, from the median of each stage
        across recent completed scans. The Twitch download is kept apart
        because a local file skips it. Scans of the same mode are preferred;
        with none, any recent scan stands in (``matched`` says which)."""
        rows = []
        try:
            with self.db.get_connection() as conn:
                rows = conn.cursor().execute(
                    '''SELECT stage_timings, duration, scan_profile, source_type FROM jobs
                       WHERE status = 'completed' AND stage_timings IS NOT NULL
                         AND duration > 0
                       ORDER BY created_at DESC LIMIT 12'''
                ).fetchall()
        except Exception:
            traceback.print_exc()
        jobs = []
        for row in rows:
            try:
                timings = json.loads(row["stage_timings"]) or {}
                vod = float(row["duration"])
                profile = json.loads(row["scan_profile"]) if row["scan_profile"] else {}
            except (ValueError, TypeError):
                continue
            key = (str(profile.get("mode") or "unknown"), bool(profile.get("facecam_tracking", True)))
            rates = {
                phase: float((t or {}).get("duration_seconds") or 0.0) / vod
                for phase, t in timings.items()
            }
            jobs.append((key, row["source_type"], rates))
        matched = [job for job in jobs if job[0] == (str(mode), bool(facecam_tracking))]
        selected = matched or jobs
        if not selected:
            return {"per_vod_second": None, "download_per_vod_second": None, "samples": 0, "matched": False}

        def median(values):
            values = sorted(values)
            return values[len(values) // 2] if values else 0.0

        phases = {phase for _, _, rates in selected for phase in rates if phase != "Resolving Input"}
        scan = sum(median([rates.get(phase, 0.0) for _, _, rates in selected]) for phase in phases)
        # A zero download means the source was already on disk; it says nothing
        # about how long a fresh download takes.
        downloads = [rates.get("Resolving Input", 0.0) for _, source, rates in jobs if source == "twitch"]
        downloads = [value for value in downloads if value > 0]
        return {
            "per_vod_second": round(scan, 5),
            "download_per_vod_second": round(median(downloads), 5) if downloads else None,
            "samples": len(selected),
            "matched": bool(matched),
        }

    def _model_eta(self, tel: dict, phase: Optional[str]):
        """Per-stage ETA: measured progress inside the current stage, plus this
        machine's historical cost for the stages that haven't started yet.
        Avoids the classic failure of extrapolating the whole job from the
        slowest phase's rate. Returns None when there's no history to model from."""
        duration = tel.get("duration")
        if not duration or not phase or phase not in STAGE_WINDOWS:
            return None
        rates = self._historical_stage_rates(tel)
        if not rates:
            return None

        now = time.time()
        timings = tel.get("stage_timings") or {}
        started = (timings.get(phase) or {}).get("started_at")
        elapsed_in_stage = max(0.0, now - started) if started else 0.0
        stage_progress = tel.get("stage_progress") or 0.0

        # Expected total for the current stage: blend the measured pace (once
        # there's enough signal) with the historical expectation.
        expected_total = rates.get(phase, 0.0) * duration
        measured_total = elapsed_in_stage / stage_progress if stage_progress > 0.1 and elapsed_in_stage > 5 else None
        if measured_total and expected_total > 0:
            stage_total = 0.5 * measured_total + 0.5 * expected_total
        elif measured_total:
            stage_total = measured_total
        elif expected_total > 0:
            stage_total = expected_total
        else:
            return None
        remaining = max(0.0, stage_total - elapsed_in_stage)

        remaining += self._future_stage_eta(tel, phase, rates)

        confidence = "high" if stage_progress > 0.25 else "medium"
        return round(min(remaining, 6 * 3600)), confidence

    def _future_stage_eta(self, tel: dict, phase: Optional[str], rates=None) -> float:
        """Expected wall time for stages after ``phase`` on this machine.

        Stage windows overlap (Event and Reaction both begin around 45%), so
        ordering by hard-coded percentage drops real work. Use the pipeline's
        execution order and only add phases actually observed in local history.
        """
        duration = float(tel.get("duration") or 0.0)
        if duration <= 0 or phase not in PIPELINE_PHASE_ORDER:
            return 0.0
        rates = rates if rates is not None else self._historical_stage_rates(tel)
        timings = tel.get("stage_timings") or {}
        index = PIPELINE_PHASE_ORDER.index(phase)
        remaining = 0.0
        for future in PIPELINE_PHASE_ORDER[index + 1:]:
            if future == "Complete" or future in timings:
                continue
            rate = rates.get(future)
            if rate and rate > 0:
                remaining += float(rate) * duration
        return remaining

    def _compute_eta(self, job_id: str, job: dict):
        """Return (eta_seconds, confidence). Only trustworthy once there is a
        real, stable work rate; otherwise (None, 'unknown')."""
        status = job.get("status")
        if status not in ("running", "pending"):
            return None, "unknown"
        tel = self._telemetry.get(job_id)
        if not tel:
            return None, "unknown"
        phase = job.get("current_stage") or tel.get("phase")
        if phase == "Resolving Input" and tel.get("download_eta_seconds") is not None:
            return max(0.0, float(tel["download_eta_seconds"])), "high"

        # Parallel perception reports a straggler-aware live ETA. Add the
        # locally learned tail (reaction/caption/render) so "Time left" means
        # ready to review, not merely done scanning frames.
        live_stage_eta = tel.get("stage_eta_seconds")
        live_eta_updated_at = tel.get("eta_scope_updated_at")
        live_eta_phase = tel.get("eta_scope_phase")
        live_eta_fresh = (
            live_eta_updated_at is None
            or time.time() - float(live_eta_updated_at) <= LIVE_STAGE_ETA_MAX_AGE_SECONDS
        )
        live_eta_matches_phase = not live_eta_phase or live_eta_phase == phase
        if live_stage_eta is not None and live_eta_fresh and live_eta_matches_phase:
            future = self._future_stage_eta(tel, phase)
            confidence = tel.get("stage_eta_confidence") or "medium"
            if future > 0 and not tel.get("eta_history_matched", False) and confidence == "high":
                confidence = "medium"
            return round(min(float(live_stage_eta) + future, 12 * 3600)), confidence

        # Preferred: per-stage model from this machine's history (plan 1B.14).
        modeled = self._model_eta(tel, phase)
        if modeled is not None:
            return modeled

        # Fallback (first-ever scan): overall progress rate, as before.
        samples = tel.get("samples") or []
        progress = float(job.get("progress") or 0.0)
        if len(samples) < 3 or progress <= 0.04 or progress >= 0.99:
            return self._budget_eta(tel)
        # Rate over the sampled window (progress fraction per second).
        (t0, p0) = samples[0]
        (t1, p1) = samples[-1]
        dt = t1 - t0
        dp = p1 - p0
        if dt <= 0 or dp <= 0:
            # Overall progress is flat (e.g. a long refinement sub-stage) —
            # fall back to the stage's own self-reported time budget so the
            # UI isn't stuck saying "measuring time left" for many minutes.
            return self._budget_eta(tel)
        rate = dp / dt
        remaining = max(0.0, (1.0 - progress) / rate)
        # Cap absurd estimates from a momentarily tiny rate.
        remaining = min(remaining, 6 * 3600)
        elapsed = tel.get("elapsed_seconds") or (t1 - tel.get("started_at", t1))
        if len(samples) >= 8 and elapsed >= 20 and progress >= 0.1:
            confidence = "high"
        elif len(samples) >= 5 and elapsed >= 8:
            confidence = "medium"
        else:
            return self._budget_eta(tel)
        return round(remaining), confidence

    def _budget_eta(self, tel: dict):
        """Fallback ETA from a stage's self-reported time budget (e.g. facecam
        refinement), used while the overall progress signal is flat."""
        budget = tel.get("stage_budget_seconds")
        started = tel.get("stage_budget_started_at")
        if not budget or not started:
            return None, "unknown"
        remaining = max(0.0, budget - (time.time() - started))
        return round(remaining), "medium"

    def list_jobs(self) -> list:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # The framing model can be tens of megabytes per library. It is only
            # consumed by targeted manual-framing requests and must never ride the
            # session-list response that Studio refreshes during navigation.
            cursor.execute(
                # recall_session_id rides along deliberately: it is a short id,
                # and without it Recall Live cannot tell a creator whether an
                # earlier session was ever scanned.
                '''SELECT id, created_at, status, session_name, source_type,
                          source_path, duration, progress, current_stage, message,
                          started_at, updated_at, stage_progress, scanned_seconds,
                          clips_found, stage_timings, scan_profile, asset_path,
                          poster_path, source_date, recall_session_id
                   FROM jobs ORDER BY created_at DESC'''
            )
            return [dict(row) for row in cursor.fetchall()]

    def _record_progress(self, job_id: str, phase: str, progress: float,
                         message: Optional[str], telemetry: Optional[dict], now: float,
                         status: str = "running"):
        """Fold one progress report into the job's live telemetry.

        The caller holds _telemetry_lock. Returns the telemetry dict, the scan
        position for the jobs row, a VOD length learned by this report (None
        when there is none), which the caller stores outside the lock, and the
        progress to store and publish (see _held_progress).
        """
        tel = self._telemetry.setdefault(job_id, {
            "started_at": now, "samples": [], "clips_found": 0, "phase": None,
            "stage_timings": {},
        })
        progress = self._held_progress(tel, progress, status)

        telemetry = telemetry or {}
        incoming_eta_scope = telemetry.get("eta_scope")
        active_eta_scope = tel.get("eta_scope")
        eta_scope_changed = bool(
            active_eta_scope
            and incoming_eta_scope
            and incoming_eta_scope != active_eta_scope
        )
        eta_scope_ended = bool(
            active_eta_scope
            and incoming_eta_scope is None
            and telemetry.get("stage_eta_seconds") is None
        )
        if phase != tel.get("phase") or eta_scope_changed or eta_scope_ended:
            tel.pop("stage_budget_seconds", None)
            tel.pop("stage_budget_started_at", None)
            tel.pop("stage_eta_seconds", None)
            tel.pop("stage_eta_confidence", None)
            tel.pop("eta_scope", None)
            tel.pop("eta_scope_phase", None)
            tel.pop("eta_scope_updated_at", None)
            tel.pop("eta_scope_end_progress", None)

        for key in (
            "scanned_seconds", "scan_speed", "stage_eta_seconds",
            "stage_eta_confidence", "active_workers", "total_workers",
            "stalled_workers", "transcribed_seconds", "transcription_speed",
        ):
            if telemetry.get(key) is not None:
                tel[key] = telemetry[key]
        if incoming_eta_scope:
            tel["eta_scope"] = str(incoming_eta_scope)
            tel["eta_scope_phase"] = phase
            if telemetry.get("eta_scope_end_progress") is not None:
                tel["eta_scope_end_progress"] = float(telemetry["eta_scope_end_progress"])
        if telemetry.get("stage_eta_seconds") is not None:
            tel["eta_scope_updated_at"] = now

        # The pipeline passes scan position, VOD length and stage budget as
        # numbers; the message is never parsed for them.
        structured_scan = telemetry.get("scanned_seconds") is not None
        if structured_scan and "scan_speed" not in telemetry:
            # Sequential perception reports a position but no rate; give it the
            # elapsed average, which is what parsing its message used to give.
            # Parallel perception always sends the key, even while it is None.
            try:
                tel["scan_speed"] = round(
                    float(telemetry["scanned_seconds"]) / max(0.001, now - tel["started_at"]), 2
                )
            except Exception:
                pass
        learned_duration = None
        if telemetry.get("duration") is not None and "duration" not in tel:
            try:
                duration = float(telemetry["duration"])
            except (TypeError, ValueError):
                duration = 0.0
            if math.isfinite(duration) and duration > 0:
                tel["duration"] = duration
                learned_duration = duration
        # Sub-stage timings and Perception worker finish times, keyed by stage.
        # Held apart from stage_timings, which the ETA reads live: an entry
        # there means "this stage has started" to _future_stage_eta and to the
        # transition below, and Reaction's early steps report during
        # Perception. Folded in only where timings are stored or served.
        detail = telemetry.get("stage_detail")
        if isinstance(detail, dict):
            held = tel.setdefault("stage_detail", {})
            for stage, fields in detail.items():
                if isinstance(stage, str) and isinstance(fields, dict):
                    held.setdefault(stage, {}).update(fields)

        # Long sub-stages (Smart scan refinement) report a time budget; it
        # stands in as the ETA while their own progress is flat.
        if telemetry.get("stage_budget_seconds") is not None:
            try:
                tel["stage_budget_seconds"] = float(telemetry["stage_budget_seconds"])
                tel["stage_budget_started_at"] = now
            except (TypeError, ValueError):
                pass

        scanned_seconds = tel.get("scanned_seconds")
        tel["updated_at"] = now
        tel["elapsed_seconds"] = round(now - tel["started_at"], 1)
        tel["message"] = message
        tel["stage_progress"] = self._stage_progress(phase, progress)
        # Sample the progress curve for ETA (throttle to ~1s spacing).
        samples = tel["samples"]
        if not samples or (now - samples[-1][0]) >= 1.0:
            samples.append((now, float(progress)))
            del samples[:-30]  # keep a bounded recent window
        return tel, scanned_seconds, learned_duration, progress

    @staticmethod
    def _held_progress(tel: dict, progress, status: str):
        """A running job's overall progress never goes back.

        run_pipeline holds its own emits monotonic, but a job is more than one
        run_pipeline call: the download reports first, from resolve_input, and
        reached 0.075 while Perception began at 0.05, so the bar ran back on
        every Twitch scan. The resolver's numbers now end where the pipeline's
        begin; this is the backstop for the whole job, as _monotonic_emitter
        is inside the pipeline. Only a running report is held: Cancelling,
        Cancelled and Complete set their own number. _begin_scan gives each
        run a fresh telemetry entry, so a retry starts from zero again.
        Caller holds _telemetry_lock.
        """
        if status != "running":
            return progress
        try:
            value = float(progress)
        except (TypeError, ValueError):
            return progress
        if not math.isfinite(value):
            return progress
        highest = tel.get("progress_high_water")
        if highest is not None and value < highest:
            return highest
        tel["progress_high_water"] = value
        return progress

    def update_job_progress(self, job_id: str, phase: str, progress: float,
                            status: str = "running", message: str = None,
                            telemetry: dict = None):
        now = time.time()
        with self._telemetry_lock:
            tel, scanned_seconds, learned_duration, progress = self._record_progress(
                job_id, phase, progress, message, telemetry, now, status,
            )
            stage_progress = tel.get("stage_progress")
            stage_timings = json.dumps(self._timings_with_detail(tel))

        if learned_duration is not None:
            # Stored now, not when the pipeline returns, so the clock and ETA
            # survive a reload mid-scan. Written outside the lock.
            try:
                with self.db.get_connection() as conn:
                    conn.cursor().execute(
                        "UPDATE jobs SET duration = ? WHERE id = ?",
                        (learned_duration, job_id)
                    )
            except Exception:
                pass

        with self.db.get_connection() as conn:
            conn.cursor().execute(
                '''UPDATE jobs SET current_stage = ?, progress = ?, status = ?,
                   message = ?, updated_at = CURRENT_TIMESTAMP, scanned_seconds = ?,
                   stage_progress = ?, stage_timings = ? WHERE id = ?''',
                (
                    phase, progress, status, message, scanned_seconds,
                    stage_progress, stage_timings,
                    job_id,
                )
            )

        # Stage transition events. The transition is decided under the lock so
        # a cancel arriving mid-report cannot finish the same stage twice.
        with self._telemetry_lock:
            prev_phase = tel.get("phase")
            stage_changed = phase != prev_phase
            if stage_changed:
                if prev_phase:
                    self._finish_stage(job_id, prev_phase, now)
                tel["phase"] = phase
                tel.setdefault("stage_timings", {}).setdefault(phase, {"started_at": now})
        if stage_changed:
            if prev_phase:
                self.emit_event(job_id, "stage_completed", phase=prev_phase)
            self.emit_event(job_id, "stage_started", phase=phase,
                            message=message, progress=progress,
                            payload=self._event_payload(job_id, progress))
        else:
            self.emit_event(job_id, "stage_progress", phase=phase,
                            message=message, progress=progress,
                            payload=self._event_payload(job_id, progress))

    def _stage_progress(self, phase: str, progress: float) -> Optional[float]:
        window = STAGE_WINDOWS.get(phase)
        if not window:
            return None
        start, end = window
        if end <= start:
            return 1.0
        return round(max(0.0, min(1.0, (float(progress) - start) / (end - start))), 3)

    # Written by _finish_stage. Detail never replaces them: history and the
    # ETA read duration_seconds from every stored stage.
    _TIMING_KEYS = ("started_at", "ended_at", "duration_seconds")

    @classmethod
    def _timings_with_detail(cls, tel: dict) -> dict:
        """stage_timings with each stage's held detail folded in, as a copy.

        Caller holds _telemetry_lock. Every stage keeps its own timing keys;
        detail adds ``substages`` (Reaction, Perception) and ``workers``
        (Perception) beside them. A stage that has detail but has not started
        yet appears with the detail alone. That only happens mid-scan or in a
        scan stopped before it reached the stage, and stage history reads
        completed scans only, where every stage with detail has run.
        """
        timings = {
            phase: dict(timing) if isinstance(timing, dict) else timing
            for phase, timing in (tel.get("stage_timings") or {}).items()
        }
        for stage, fields in (tel.get("stage_detail") or {}).items():
            entry = timings.setdefault(stage, {})
            if isinstance(entry, dict):
                entry.update({
                    key: value for key, value in fields.items()
                    if key not in cls._TIMING_KEYS
                })
        return timings

    def _finish_stage(self, job_id: str, phase: str, now: float):
        # Caller holds _telemetry_lock.
        tel = self._telemetry.setdefault(job_id, {})
        timings = tel.setdefault("stage_timings", {})
        stage = timings.setdefault(phase, {})
        stage.setdefault("started_at", now)
        stage["ended_at"] = now
        stage["duration_seconds"] = round(max(0.0, now - stage["started_at"]), 1)

    def _stage_timings_snapshot(self, job_id: str) -> dict:
        with self._telemetry_lock:
            return self._timings_with_detail(self._telemetry.get(job_id) or {})

    def _event_payload(self, job_id: str, progress: float = None) -> dict:
        tel = self._telemetry.get(job_id, {})
        eta_seconds, eta_confidence = self._compute_eta(
            job_id, {"status": "running", "progress": progress}
        )
        payload = {
            "stage_progress": tel.get("stage_progress"),
            "scanned_seconds": tel.get("scanned_seconds"),
            "scan_speed": tel.get("scan_speed"),
            "transcribed_seconds": tel.get("transcribed_seconds"),
            "transcription_speed": tel.get("transcription_speed"),
            "clips_found": tel.get("clips_found", 0),
            "duration": tel.get("duration"),
            "eta_seconds": eta_seconds,
            "eta_confidence": eta_confidence,
            "download_progress": tel.get("download_progress"),
            "active_workers": tel.get("active_workers"),
            "total_workers": tel.get("total_workers"),
            "stalled_workers": tel.get("stalled_workers"),
            "eta_scope": tel.get("eta_scope"),
        }
        return {k: v for k, v in payload.items() if v is not None}

    def mark_job_failed(self, job_id: str, error_msg: str, error_traceback: str = None):
        # Preserve WHICH stage was running when the failure hit: the failure
        # UI diagnoses "stopped during Perception" very differently from
        # "stopped during Export", and overwriting current_stage with
        # "Failed" used to throw that context away.
        failed_stage = None
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT current_stage FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row and row["current_stage"] not in (None, "", "Failed"):
                failed_stage = row["current_stage"]
            message = f"[{failed_stage}] {error_msg}" if failed_stage else error_msg
            cursor.execute(
                '''UPDATE jobs SET status = ?, current_stage = ?, message = ?,
                   updated_at = CURRENT_TIMESTAMP WHERE id = ?''',
                ("failed", "Failed", message, job_id)
            )
        # The traceback rides on the job_failed event so the diagnostics
        # bundle can show WHERE it broke, not just the exception string.
        payload = {}
        if failed_stage:
            payload["failed_stage"] = failed_stage
        if error_traceback:
            payload["traceback"] = error_traceback[-8000:]
        self.emit_event(job_id, "job_failed", message=message,
                        payload=payload or None)

    def start_job(self, job_id: str, settings: dict = None):
        job = self.get_job(job_id, include_events=False)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # A batch should pay the ASR warm-up cost once. The final queued job
        # schedules release after a short idle window instead of keeping the
        # model resident for the lifetime of the desktop app.
        from engines.caption.asr import hold_asr_models
        hold_asr_models()

        # Register for cancellation before the worker starts, so a job can be
        # cancelled even while it's queued behind another scan.
        thread = threading.Thread(target=self._run_job_thread, args=(job_id, job['source_path'], settings), daemon=True)
        self._active_jobs[job_id] = {"thread": thread, "cancel_flag": False}
        with self.db.get_connection() as conn:
            conn.cursor().execute(
                "UPDATE jobs SET status = 'pending', current_stage = 'Init', "
                "message = 'Getting ready...', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (job_id,)
            )
        self.emit_event(job_id, "job_started", phase="Init", progress=0.0)
        thread.start()

    def _begin_scan(self, job_id: str):
        """Reset the scan clock at the moment the job actually starts running
        (after any queue wait), so elapsed time never counts time spent queued."""
        now = time.time()
        with self._telemetry_lock:
            self._telemetry[job_id] = {
                "started_at": now, "samples": [], "clips_found": 0, "phase": None,
                "stage_timings": {},
            }
        with self.db.get_connection() as conn:
            conn.cursor().execute(
                "UPDATE jobs SET started_at = CURRENT_TIMESTAMP, clips_found = 0 WHERE id = ?",
                (job_id,)
            )
        self.update_job_progress(job_id, "Init", 0.0, "running", message="Getting the source ready.")

    def _reaction_signals(self, clip) -> list:
        features = getattr(clip, "features", None) or {}
        breakdown = getattr(clip, "modality_breakdown", None) or {}
        reason = (getattr(clip, "reason", None) or "").lower()
        signals = []

        if features.get("is_win", 0.0) >= 1.0 or "match win" in reason:
            signals.append("match_win")
        elif features.get("game_tier", 0.0) >= 0.8 or "elimination" in reason:
            signals.append("elimination")
        elif features.get("game_tier", 0.0) >= 0.6 or "knock" in reason:
            signals.append("knock")
        elif features.get("game_tier", 0.0) >= 0.4 or "rank progress" in reason:
            signals.append("rank_progress")

        if breakdown.get("voice", 0.0) >= 0.35:
            signals.append("voice_reaction")
        if breakdown.get("face", 0.0) >= 0.12:
            signals.append("facecam_reaction")
        if breakdown.get("speech", 0.0) >= 0.20:
            signals.append("speech_hype")
        if breakdown.get("burst", 0.0) >= 0.30:
            signals.append("laughter_burst")
        if breakdown.get("chat", 0.0) >= 2.0:
            signals.append("chat_spike")
        if features.get("signal_agreement", 0.0) >= 0.75:
            signals.append("multi_signal")
        if features.get("dead_air_ratio", 1.0) <= 0.15:
            signals.append("low_dead_air")

        seen = set()
        return [s for s in signals if not (s in seen or seen.add(s))]

    def _hook_score(self, clip) -> float:
        """Calibrated 0..1 display score; avoids every good clip showing 99-100."""
        explicit = getattr(clip, "hook_score", None)
        if explicit is not None and float(explicit or 0.0) > 0.0:
            return max(0.0, min(1.0, float(explicit)))
        features = getattr(clip, "features", None) or {}
        if "hook_score" in features:
            return max(0.0, min(1.0, float(features.get("hook_score", 0.0) or 0.0)))
        if not features:
            return float(getattr(clip, "score", 0.0) or 0.0)

        peak = min(1.0, float(features.get("peak_value", 0.0)) / 3.0)
        intensity = min(1.0, float(features.get("mean_intensity", 0.0)) / 2.2)
        game = min(1.0, float(features.get("game_tier", 0.0)))
        agreement = min(1.0, float(features.get("signal_agreement", 0.0)) / 0.75)
        dead_air = 1.0 - min(1.0, float(features.get("dead_air_ratio", 1.0)))
        payoff = float(features.get("payoff_position", features.get("peak_position", 0.5)))
        payoff_shape = max(0.0, 1.0 - abs(payoff - 0.68) / 0.68)

        raw = (
            0.28 * peak +
            0.24 * intensity +
            0.18 * game +
            0.14 * agreement +
            0.10 * dead_air +
            0.06 * payoff_shape
        )
        if features.get("is_win", 0.0) >= 1.0:
            raw = max(raw, 0.82)

        return max(0.45, min(0.98, raw))

    def _persist_selected_clips(self, job_id, clips, captions, stories, exported_paths):
        """Persist the finished review deck and return its live UI events."""
        from core.more_candidates import (
            MORE_CANDIDATE_LABEL,
            REVIEW_TIER_SECOND_LOOK,
            second_look_deck_score,
        )

        caption_by_clip = {cap.clip_id: cap for cap in captions}
        story_by_id = {s.story_id: s for s in stories}
        primary_clips = [
            clip for clip in clips
            if getattr(clip, "review_tier", None) != REVIEW_TIER_SECOND_LOOK
        ]
        second_look_clips = [
            clip for clip in clips
            if getattr(clip, "review_tier", None) == REVIEW_TIER_SECOND_LOOK
        ]
        deck_scores = deck_percentile_scores(primary_clips)
        for index, clip in enumerate(second_look_clips):
            deck_scores[clip.clip_id] = second_look_deck_score(index)
        clip_found_events = []

        # Every thumbnail is an ffmpeg run. Make them all before the connection
        # opens: from the first INSERT below this connection holds SQLite's
        # write lock until it commits, and a label click or settings save
        # waited behind every remaining ffmpeg run.
        media = []
        for clip in clips:
            matched_export = next(
                (path for path in exported_paths if f"clip_{clip.clip_id}.mp4" in path),
                None,
            )
            thumb_path = None
            if matched_export:
                try:
                    from core.thumbnails import generate_thumbnail, clip_relative_peak
                    seek = clip_relative_peak(
                        getattr(clip, "peak_timestamp", None), clip.start, clip.end
                    )
                    thumb_path = generate_thumbnail(
                        matched_export,
                        clip.clip_id,
                        os.path.dirname(matched_export),
                        seek_seconds=seek,
                    )
                except Exception:
                    thumb_path = None
            media.append((matched_export, thumb_path))

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            from core.clip_number import next_clip_number

            job_row = cursor.execute(
                "SELECT recall_session_id FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            recall_session_id = (
                job_row["recall_session_id"] if job_row else None
            )

            first_clip_number = next_clip_number(conn, job_id)
            chronological = sorted(
                clips,
                key=lambda item: (
                    float(getattr(item, "start", 0.0) or 0.0),
                    float(getattr(item, "end", 0.0) or 0.0),
                    str(getattr(item, "clip_id", "")),
                ),
            )
            clip_numbers = {
                clip.clip_id: first_clip_number + offset
                for offset, clip in enumerate(chronological)
            }
            for clip, (matched_export, thumb_path) in zip(clips, media):
                clip_id = clip.clip_id
                is_second_look = getattr(clip, "review_tier", None) == REVIEW_TIER_SECOND_LOOK
                caption = caption_by_clip.get(clip_id)
                story = story_by_id.get(getattr(clip, "story_id", None))
                title = caption.title if caption else None
                description = caption.description if caption else None
                if not description and getattr(clip, "reason", None):
                    description = clip.reason
                if not title and story:
                    title = story.label
                if is_second_look:
                    title = title or "Second-look moment"
                    description = description or (
                        "A lower-ranked moment Recall saved just outside the first review deck."
                    )
                    story_label = MORE_CANDIDATE_LABEL
                else:
                    story_label = story.label if story else None

                signal_labels = []
                if story and not is_second_look:
                    for event in story.events:
                        if getattr(event, "event_type", None):
                            signal_labels.append(event.event_type)
                        signal_labels.extend(getattr(event, "signals", []) or [])
                signal_labels = list(dict.fromkeys(signal_labels))
                if not signal_labels:
                    signal_labels = self._reaction_signals(clip)

                features = getattr(clip, "features", None)
                features_json = json.dumps(features) if features else None
                hook_score = self._hook_score(clip)
                selection_score = float(getattr(clip, "score", 0.0) or 0.0)
                legacy_order_score = deck_scores.get(clip_id, hook_score)
                breakdown = getattr(clip, "modality_breakdown", None)
                breakdown_json = json.dumps(breakdown) if breakdown else None
                layout_json = json.dumps(getattr(clip, "layout", None) or {})
                marker_ids = list(getattr(clip, "recall_marker_ids", None) or [])
                recall_provenance = None
                creator_protected = bool(
                    getattr(clip, "creator_protected", False)
                )
                if recall_session_id or marker_ids or creator_protected:
                    origin = (
                        "creator_marker"
                        if marker_ids or creator_protected
                        else "session_scan"
                    )
                    recall_provenance = json.dumps({
                        "source": "recall_session",
                        "session_id": recall_session_id,
                        "origin": origin,
                        "creator_protected": creator_protected,
                        "marker_ids": marker_ids,
                        "marker_time": getattr(clip, "recall_marker_time", None),
                    }, separators=(",", ":"))

                post_tags = json.dumps(caption.tags) if caption and caption.tags else None
                cursor.execute(
                    '''INSERT INTO clips
                       (id, job_id, clip_number, start_time, end_time, score, selection_score, hook_score, deck_score, title, description, signals, story_label, export_path, features,
                        reason, peak_timestamp, modality_breakdown, reaction_auc, thumb_path, scene, tags, preview_only,
                        hook_line, moment_type, layout, semantic_verdict, recall_provenance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        clip_id, job_id, clip_numbers[clip_id], clip.start, clip.end, hook_score,
                        selection_score, hook_score, legacy_order_score,
                        title, description, json.dumps(signal_labels), story_label, matched_export,
                        features_json, getattr(clip, "reason", None),
                        getattr(clip, "peak_timestamp", None), breakdown_json,
                        getattr(clip, "reaction_auc", None), thumb_path,
                        getattr(clip, "scene_label", None), post_tags,
                        1 if matched_export else 0,
                        getattr(clip, "semantic_hook_line", None),
                        getattr(clip, "semantic_moment_type", None),
                        layout_json,
                        getattr(clip, "semantic_verdict", None),
                        recall_provenance,
                    ),
                )
                if matched_export and not is_second_look:
                    cursor.execute(
                        '''INSERT INTO exports (id, clip_id, platform, file_path) VALUES (?, ?, ?, ?)''',
                        (str(uuid.uuid4()), clip_id, "local", matched_export),
                    )

                # Second-look is persisted ready-to-reveal but must not inflate
                # the live found shelf or clips_found telemetry.
                if is_second_look:
                    continue

                telemetry = self._telemetry.setdefault(job_id, {})
                if not telemetry.get("candidates_emitted"):
                    telemetry["clips_found"] = telemetry.get("clips_found", 0) + 1
                clip_found_events.append({
                    "message": title or "New moment",
                    "payload": {
                        "clip_id": clip_id,
                        "title": title or "Untitled moment",
                        "timestamp": format_vod_time(clip.start or 0),
                        "duration": int((clip.end or 0) - (clip.start or 0)),
                        "score": int(round(legacy_order_score * 100)),
                        "replaces": clip_id,
                    },
                })
        return clip_found_events

    def _run_post_scan_janitor(self, job_id: str):
        """Apply the unified Recall/source lifecycle after a successful scan.

        Runs on the scan thread after the job is marked completed, while it is
        still in _active_jobs. The janitor defers whenever a scan is active, so
        it used to defer for this job too and never ran here at all; cleanup
        only happened at app start, in Storage or on session delete. The
        finished job is now excluded from that check. Any other job, queued
        behind this one or started meanwhile, still defers the cleanup: it may
        be about to read a source or a cache file. A failed or cancelled scan
        does not get here, so nothing a retry might want is touched for it.
        """
        from core.storage_lifecycle import SourceLifecycleService

        lifecycle = SourceLifecycleService(self.db, self, finished_job_id=job_id)
        lifecycle.reconcile_sources(remove_expired=True)
        lifecycle.reconcile_recall_storage()

    def _write_candidates(self, job_id: str, trace, diagnostics=None) -> None:
        """Write candidates.v1.json at most once per job.

        Same failure posture as write_job_artifacts: capture must never fail a
        creator's render. The once-per-job flag exists because the trace can
        arrive via both the timeline callback and the returned timeline dict
        (whichever fires first wins; they carry the same rows).
        """
        tel = self._telemetry.setdefault(job_id, {})
        if tel.get("candidates_written"):
            return
        try:
            runtime_diagnostics = dict(diagnostics or {})
            try:
                from core.ranker_service import ranker_status
                runtime_diagnostics["feedback"] = ranker_status(self.db)
            except Exception:  # noqa: BLE001 - diagnostics never block capture
                pass
            write_candidates_artifact(
                job_id, trace, diagnostics=runtime_diagnostics,
            )
            tel["candidates_written"] = True
            tel["ranker_diagnostics"] = runtime_diagnostics
        except Exception:
            traceback.print_exc()

    def _run_job_thread(self, job_id: str, source_path: str, settings: dict):
        # Only one pipeline runs at a time (FIFO queue, plan 6.3). Grab the slot
        # immediately if free, otherwise show the job as queued and wait.
        got_slot = self._pipeline_sem.acquire(blocking=False)
        try:
            if not got_slot:
                with self.db.get_connection() as conn:
                    conn.cursor().execute(
                        "UPDATE jobs SET status = 'pending', current_stage = 'Init', "
                        "message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        ("Waiting for an earlier scan to finish...", job_id)
                    )
                self.emit_event(job_id, "stage_progress", phase="Init",
                                message="Waiting for an earlier scan to finish...", progress=0.0)
                self._pipeline_sem.acquire()  # block until the running scan frees the slot
                got_slot = True

            # Cancelled while it sat in the queue — don't start.
            if self._active_jobs.get(job_id, {}).get("cancel_flag"):
                self.update_job_progress(job_id, "Cancelled", 0.0, "cancelled",
                                         message="Scan cancelled.")
                self.emit_event(job_id, "job_cancelled", phase="Cancelled")
                return

            self._begin_scan(job_id)
            self.update_job_progress(job_id, "Resolving Input", 0.0)

            # Keep server startup light. The pipeline imports torch, OCR,
            # Whisper, vision, and other heavy modules; load them only once a
            # user actually starts processing a video.
            from pipeline.ingest.vod_resolver import resolve_input
            from pipeline.orchestrator.run_pipeline import build_scan_profile, run_pipeline

            # resolve_input hands over the share downloaded as a number, just
            # before the progress line it belongs to, and that line carries it
            # to the job. A line without one (fetching info, finalizing) clears
            # the download ETA, as it did when the share was parsed from text.
            download_line = {}

            def download_share_cb(fraction, eta_seconds=None):
                download_line["fraction"] = fraction
                download_line["eta"] = eta_seconds

            def download_progress_cb(message, progress, eta_seconds=None):
                # Let a cancel during download abort promptly (plan 6.1): raising
                # here propagates out of resolve_input, which kills the
                # TwitchDownloaderCLI download subprocess.
                if self._active_jobs.get(job_id, {}).get("cancel_flag"):
                    raise InterruptedError("Job was cancelled")
                fraction = download_line.pop("fraction", None)
                eta = download_line.pop("eta", None)
                if eta is None:
                    eta = eta_seconds
                now = time.time()
                with self._telemetry_lock:
                    tel = self._telemetry.setdefault(job_id, {})
                    if fraction is not None:
                        ratio = min(1.0, max(0.0, float(fraction)))
                        tel["download_progress"] = ratio
                        started = tel.setdefault("download_started_at", now)
                        if eta is None and ratio >= 0.02:
                            eta = max(0.0, now - started) * (1.0 - ratio) / ratio
                    tel["download_eta_seconds"] = eta
                self.update_job_progress(job_id, "Resolving Input", progress, message=message)

            # 1. Resolve URL or Path
            video_path = resolve_input(
                source_path,
                output_dir=os.path.join(get_data_dir(), "assets"),
                progress_callback=download_progress_cb,
                cancel_check=lambda: bool(self._active_jobs.get(job_id, {}).get("cancel_flag")),
                download_callback=download_share_cb,
            )
            self._telemetry.setdefault(job_id, {}).pop("download_eta_seconds", None)
            self._telemetry.setdefault(job_id, {})["download_progress"] = 1.0
            # Remember which local media file backs this job so deletion can
            # reclaim downloaded VODs once nothing references them (plan 4.2).
            with self.db.get_connection() as conn:
                conn.cursor().execute(
                    "UPDATE jobs SET asset_path = ? WHERE id = ?",
                    (os.path.abspath(video_path), job_id)
                )
            from core.storage_lifecycle import SourceLifecycleService
            SourceLifecycleService(self.db, self).ensure_source_for_job(
                job_id, source_path, os.path.abspath(video_path)
            )
            scan_profile = build_scan_profile(settings or {})
            self._telemetry.setdefault(job_id, {})["scan_profile"] = scan_profile
            with self.db.get_connection() as conn:
                conn.cursor().execute(
                    "UPDATE jobs SET scan_profile = ? WHERE id = ?",
                    (json.dumps(scan_profile), job_id)
                )
            self.emit_event(
                job_id,
                "scan_profile",
                phase="Init",
                message=f"{scan_profile['name']} is ready.",
                # No progress number: this fires after the download reached
                # 0.05, and the live view takes any event's number as the
                # scan's, so a fixed 0.02 here dipped the bar on a real scan.
                payload={"scan_profile": scan_profile},
            )
            
            def progress_cb(phase, message, progress, telemetry=None):
                # We could check cancellation here
                if self._active_jobs.get(job_id, {}).get("cancel_flag"):
                    raise InterruptedError("Job was cancelled")
                self.update_job_progress(
                    job_id, phase, progress, message=message, telemetry=telemetry
                )

            def cancel_check():
                if self._active_jobs.get(job_id, {}).get("cancel_flag"):
                    raise InterruptedError("Job was cancelled")
                
            def clip_candidate_cb(clip):
                """Announce a clip the moment selection picks it (plan 1B.7) —
                captions + export still take minutes, and the found shelf
                shouldn't sit empty until then. The DB save later emits an
                enriched clip_found that replaces this placeholder."""
                tel = self._telemetry.setdefault(job_id, {})
                tel["clips_found"] = tel.get("clips_found", 0) + 1
                tel["candidates_emitted"] = True
                stamp = format_vod_time(clip.start or 0)
                self.emit_event(
                    job_id, "clip_found", phase="Clip",
                    message=f"Moment at {stamp}",
                    payload={
                        "clip_id": clip.clip_id,
                        "title": "New moment",
                        "timestamp": stamp,
                        "duration": int((clip.end or 0) - (clip.start or 0)),
                        "score": int(round(self._hook_score(clip) * 100)),
                        "pending": True,
                    },
                )

            def timeline_ready_cb(timeline):
                # The private selection-trace payload is training capture only
                # (HUMAN_CLIPS Package 1): strip it BEFORE the timeline is
                # persisted or emitted — reaction_timelines rows and the
                # frontend event stream must stay lean — and write it to the
                # job's artifact dir as candidates.v1.json instead.
                trace = (
                    timeline.pop("_selection_trace", None)
                    if isinstance(timeline, dict) else None
                )
                diagnostics = (
                    timeline.pop("_selection_diagnostics", None)
                    if isinstance(timeline, dict) else None
                )
                if trace:
                    self._write_candidates(job_id, trace, diagnostics)
                # A preliminary curve is a live preview only: it is never
                # persisted and never counts as the job's timeline.
                if not (isinstance(timeline, dict) and timeline.get("preliminary")):
                    self.db.set_reaction_timeline(job_id, timeline)
                    self._telemetry.setdefault(job_id, {})["timeline_emitted"] = True
                self.emit_event(
                    job_id, "timeline_ready", phase="Clip",
                    payload={"timeline": timeline},
                )

            def framing_ready_cb(framing):
                """Keep scene-aware crop evidence for future VOD Editor cuts."""
                with self.db.get_connection() as conn:
                    conn.cursor().execute(
                        "UPDATE jobs SET framing = ? WHERE id = ?",
                        (json.dumps(framing, separators=(",", ":")), job_id),
                    )

            def pipeline_event_cb(event_type, phase, message, payload=None):
                """Persist non-progress pipeline health signals for UI + diagnostics."""
                tel = self._telemetry.setdefault(job_id, {})
                if event_type == "source_metadata":
                    from core.source_date import normalize_source_date

                    source_date = normalize_source_date((payload or {}).get("source_date"))
                    if source_date:
                        with self.db.get_connection() as conn:
                            conn.cursor().execute(
                                "UPDATE jobs SET source_date = ? WHERE id = ?",
                                (source_date, job_id),
                            )
                    streamer = str((payload or {}).get("streamer") or "").strip()[:100]
                    if streamer:
                        with self.db.get_connection() as conn:
                            conn.cursor().execute(
                                "UPDATE jobs SET streamer = ? WHERE id = ?",
                                (streamer, job_id),
                            )
                if event_type in {"semantic_judge_health", "semantic_judge_degraded"}:
                    tel["semantic_judge_health"] = dict(payload or {})
                elif event_type in {
                    "visual_judge_health",
                    "visual_judge_degraded",
                    "visual_judge_skipped",
                }:
                    tel["visual_judge_health"] = dict(payload or {})
                self.emit_event(
                    job_id,
                    event_type,
                    phase=phase,
                    message=message,
                    payload=payload,
                )

            # 2. Run Pipeline. Pass the original source so the reaction engine
            # can pull Twitch chat when the source was a Twitch VOD (plan 9.1);
            # the resolved video_path is a local file and loses the URL.
            pipeline_settings = {
                **(settings or {}),
                "source_url": source_path,
                "job_id": job_id,
            }
            with self.db.get_connection() as conn:
                title_row = conn.cursor().execute(
                    "SELECT session_name, streamer FROM jobs WHERE id = ?", (job_id,),
                ).fetchone()
            if title_row and title_row["session_name"]:
                pipeline_settings["session_name"] = title_row["session_name"]
            # A re-scan already knows the channel; Twitch metadata names it
            # otherwise. Only the streamer-voice profile reads it.
            if title_row and title_row["streamer"]:
                pipeline_settings.setdefault("streamer", title_row["streamer"])
            # Settings > "Only caption my voice", default on. The desktop app
            # sends it with each scan; any other caller gets the saved choice.
            if "captionStreamerOnly" not in pipeline_settings:
                pipeline_settings["captionStreamerOnly"] = (
                    self.db.get_setting("captionStreamerOnly", True) is not False)
            (
                signals, stories, clips, captions, exported_paths, vod_duration,
                reaction_timeline, vod_transcript,
            ) = run_pipeline(
                video_path,
                settings=pipeline_settings,
                progress_callback=progress_cb,
                clip_callback=clip_candidate_cb,
                timeline_callback=timeline_ready_cb,
                cancel_check=cancel_check,
                framing_callback=framing_ready_cb,
                event_callback=pipeline_event_cb,
            )

            # Debug artifacts are for engine inspection/regression — on by
            # default in dev, off in the packaged app (plan 4.5) so end users
            # don't accumulate data/jobs/* they'll never read.
            from core.bundle_paths import is_frozen
            if (settings or {}).get("debugArtifacts", not is_frozen()):
                try:
                    from core.ranker_service import ranker_status
                    write_job_artifacts(
                        job_id=job_id,
                        source_path=video_path,
                        settings=settings or {},
                        signals=signals,
                        events=[],  # canonical artifact lane retained for schema compatibility
                        stories=stories,
                        clips=clips,
                        captions=captions,
                        exported_paths=exported_paths,
                        duration=vod_duration,
                        stage_timings=self._stage_timings_snapshot(job_id),
                        ranker_mode=ranker_status(self.db).get("active_mode"),
                        ranker_diagnostics=self._telemetry.get(job_id, {}).get(
                            "ranker_diagnostics", {}
                        ),
                    )
                except Exception:
                    # Artifacts are for inspection/regression; a creator's render
                    # should not fail just because debug serialization did.
                    traceback.print_exc()

            # Persist the source VOD length so the UI can place clips on a timeline.
            with self.db.get_connection() as conn:
                conn.cursor().execute(
                    "UPDATE jobs SET duration = ? WHERE id = ?",
                    (vod_duration, job_id)
                )
            # Fallback pop: timeline_ready_cb mutates the same dict object, so
            # this only fires when the callback path never ran (e.g. a caller
            # without a timeline_callback). The trace must never reach
            # set_reaction_timeline or the timeline_ready event either way.
            if isinstance(reaction_timeline, dict):
                trace = reaction_timeline.pop("_selection_trace", None)
                diagnostics = reaction_timeline.pop(
                    "_selection_diagnostics", None,
                )
                if trace:
                    self._write_candidates(job_id, trace, diagnostics)
            # VOD-level R(t) curve backing the review screen's reaction timeline.
            if reaction_timeline and not self._telemetry.get(job_id, {}).get("timeline_emitted"):
                self.db.set_reaction_timeline(job_id, reaction_timeline)
                # Surface the real curve to the live scan theater now — selection
                # is done, but captions + export still take minutes. (handbook/21 §6.3)
                self.emit_event(
                    job_id, "timeline_ready", phase="Clip",
                    payload={"timeline": reaction_timeline},
                )
            
            # Index the computed evidence so each clip can carry a real title,
            # description, and the signals that earned it a spot.
            clip_found_events = self._persist_selected_clips(
                job_id, clips, captions, stories, exported_paths
            )

            # Searchable Stream Memory is a derived, rebuildable index. It runs
            # only after the canonical clips/timeline exist, and an indexing
            # failure must never turn a successful scan into a failed scan.
            try:
                from core.stream_memory import StreamMemoryService

                memory_result = StreamMemoryService(self.db).index_job(
                    job_id,
                    transcript=vod_transcript,
                    transcript_coverage=(
                        "partial"
                        if pipeline_settings.get("processing_mode") == "fast"
                        else "full"
                    ),
                )
                self.emit_event(
                    job_id,
                    "memory_indexed",
                    phase="Memory",
                    message=(
                        "Added this session to Stream Memory "
                        f"({memory_result['total_entries']} searchable moments)."
                    ),
                    payload=memory_result,
                )
            except Exception as exc:
                traceback.print_exc()
                self.emit_event(
                    job_id,
                    "memory_index_failed",
                    phase="Memory",
                    message="Stream Memory will retry this session while Recall is idle.",
                    payload={"error": str(exc)[:500]},
                )

            for event in clip_found_events:
                self.emit_event(
                    job_id, "clip_found", phase="Clip",
                    message=event["message"],
                    payload=event["payload"],
                )

            from core.more_candidates import REVIEW_TIER_SECOND_LOOK
            primary_clip_count = sum(
                1 for clip in clips
                if getattr(clip, "review_tier", None) != REVIEW_TIER_SECOND_LOOK
            )
            clips_found = self._telemetry.get(job_id, {}).get("clips_found", primary_clip_count)
            with self.db.get_connection() as conn:
                conn.cursor().execute(
                    "UPDATE jobs SET clips_found = ? WHERE id = ?",
                    (clips_found, job_id)
                )

            self.update_job_progress(job_id, "Complete", 1.0, "completed",
                                     message="All done — ready to review.")
            # The completed timings are the newest local calibration sample.
            self._stage_rate_cache.clear()
            from core.storage_lifecycle import SourceLifecycleService
            SourceLifecycleService(self.db, self).mark_scan_complete(job_id)
            self.emit_event(job_id, "job_completed", phase="Complete", progress=1.0,
                            payload={"clips_found": clips_found})

            # Keep the scan cache bounded (plan 4.3). Eviction goes by tier,
            # then oldest first: over the target, this VOD's analysis WAV can
            # go too, which costs only a re-scan its extraction; nothing in
            # review reads it. A janitor error never fails a completed scan.
            try:
                self._run_post_scan_janitor(job_id)
            except Exception:
                traceback.print_exc()

        except InterruptedError:
            self.update_job_progress(job_id, "Cancelled", 0.0, "cancelled",
                                     message="Scan cancelled.")
            self.emit_event(job_id, "job_cancelled", phase="Cancelled")
        except Exception as e:
            traceback.print_exc()
            self.mark_job_failed(job_id, str(e), traceback.format_exc())
        finally:
            if got_slot:
                self._pipeline_sem.release()
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]
            self._release_telemetry(job_id)
            if got_slot:
                # Release when THIS job's pipeline slot frees, not when the
                # queue empties. A queued job enters _active_jobs as soon as
                # its thread starts and then blocks on the semaphore, so during
                # a batch _active_jobs was never empty and the release never
                # fired at all between VODs -- ASR stayed resident for the whole
                # queue. Cheap when that was 1.5 GB of medium.en; with the
                # ~12 GB Qwen stack it left the next VOD's perception pool
                # starved of VRAM. The semaphore guarantees no other pipeline
                # is running here, and release_asr_models() refuses while
                # any transcription is still active.
                try:
                    from engines.caption.asr import release_asr_models
                    release_asr_models()
                except Exception:
                    traceback.print_exc()

    def _release_telemetry(self, job_id: str):
        """Drop a terminal job's live telemetry; every scan used to leak one.

        get_job falls back to the stored columns for everything it reads from
        here that outlives the scan (timings, profile, position, duration,
        clips found, message), and _scan_health falls back to the persisted
        judge events. Stage timings are written once more first, because
        the failure path never writes them and a cancel stores them before
        finishing the stage it interrupted. The live-only values (speeds,
        worker counts, ETA samples, download progress) mean nothing once the
        scan has stopped and are dropped with the entry.
        """
        with self._telemetry_lock:
            tel = self._telemetry.get(job_id) or {}
            stage_timings = (
                json.dumps(self._timings_with_detail(tel))
                if tel.get("stage_timings") or tel.get("stage_detail") else None
            )
        if stage_timings is not None:
            try:
                with self.db.get_connection() as conn:
                    conn.cursor().execute(
                        "UPDATE jobs SET stage_timings = ? WHERE id = ?",
                        (stage_timings, job_id),
                    )
            except Exception:
                traceback.print_exc()
        with self._telemetry_lock:
            self._telemetry.pop(job_id, None)
        # emit_event drops these on the terminal event; this also covers a run
        # that ended without one.
        self._last_persisted_progress.pop(job_id, None)
        self._persisted_progress_since_trim.pop(job_id, None)

    def cancel_job(self, job_id: str):
        if job_id in self._active_jobs:
            self._active_jobs[job_id]["cancel_flag"] = True
            self.update_job_progress(job_id, "Cancelling", 0.0, "cancelling")
