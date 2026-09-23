# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from core.compilation_projects import CompilationProjectService
from core.storage_lifecycle import DOWNLOAD_RESERVE_BYTES, SourceLifecycleService


PREPARATION_SCHEMA_VERSION = 1
ACTIVE_STATUSES = ("queued", "running", "cancelling")
TERMINAL_ITEM_STATUSES = ("prepared", "failed", "cancelled", "interrupted")


class CompilationPreparationConflict(RuntimeError):
    pass


class CompilationPreparationStorageError(CompilationPreparationConflict):
    def __init__(self, message: str, plan: dict[str, Any]):
        super().__init__(message)
        self.plan = plan


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CompilationPreparationService:
    """Persist and run sequential source restoration for one compilation.

    Source restoration remains owned by ``SourceLifecycleService``. This layer
    contributes only project planning, explicit confirmation, durable progress,
    sequential scheduling, and cancellation. It never downloads or removes a
    file itself.
    """

    def __init__(
        self,
        db,
        projects: CompilationProjectService,
        job_manager: Any = None,
        lifecycle: Optional[SourceLifecycleService] = None,
        *,
        poll_seconds: float = 0.5,
    ):
        self.db = db
        self.projects = projects
        self.job_manager = job_manager
        self.lifecycle = lifecycle or SourceLifecycleService(db, job_manager)
        self.poll_seconds = max(0.05, float(poll_seconds))
        self._recover_interrupted()

    def _recover_interrupted(self) -> None:
        """Make a killed desktop run resumable without pretending it continued."""
        now = _utc_now()
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id FROM compilation_preparations "
                "WHERE status IN ('queued', 'running', 'cancelling')"
            ).fetchall()
            for row in rows:
                conn.execute(
                    """UPDATE compilation_preparation_items
                       SET status = 'interrupted',
                           error = 'Recall closed before this source finished.',
                           completed_at = ?, updated_at = ?
                       WHERE preparation_id = ? AND status IN ('pending', 'restoring')""",
                    (now, now, row["id"]),
                )
                conn.execute(
                    """UPDATE compilation_preparations
                       SET status = 'interrupted', current_source_key = NULL,
                           error = 'Preparation paused when Recall closed. Start again to continue.',
                           completed_at = ?, updated_at = ? WHERE id = ?""",
                    (now, now, row["id"]),
                )

    def _has_active_scan(self) -> bool:
        if self.job_manager is not None and self.job_manager.has_active_jobs():
            return True
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE status IN ('pending','running','cancelling') LIMIT 1"
            ).fetchone()
        return row is not None

    def _latest_row(self, project_id: str, *, active_only: bool = False):
        where = "AND status IN ('queued', 'running', 'cancelling')" if active_only else ""
        with self.db.get_connection() as conn:
            return conn.execute(
                f"""SELECT * FROM compilation_preparations
                    WHERE project_id = ? {where}
                    ORDER BY created_at DESC, id DESC LIMIT 1""",  # nosec B608
                (project_id,),
            ).fetchone()

    def _serialize_preparation(self, row: Any) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        record = dict(row)
        with self.db.get_connection() as conn:
            items = [
                dict(item)
                for item in conn.execute(
                    """SELECT source_key, label, position, status, estimated_bytes,
                              error, started_at, completed_at, updated_at
                       FROM compilation_preparation_items
                       WHERE preparation_id = ? ORDER BY position, id""",
                    (record["id"],),
                ).fetchall()
            ]
        completed = sum(item["status"] in TERMINAL_ITEM_STATUSES for item in items)
        return {
            "id": record["id"],
            "project_id": record["project_id"],
            "status": record["status"],
            "total_sources": int(record["total_sources"] or 0),
            "prepared_sources": int(record["prepared_sources"] or 0),
            "failed_sources": int(record["failed_sources"] or 0),
            "completed_sources": completed,
            "estimated_bytes": int(record["estimated_bytes"] or 0),
            "current_source_key": record["current_source_key"],
            "cancel_requested": bool(record["cancel_requested"]),
            "error": record["error"],
            "created_at": record["created_at"],
            "started_at": record["started_at"],
            "completed_at": record["completed_at"],
            "updated_at": record["updated_at"],
            "items": items,
        }

    def _missing_sources(self, project_id: str) -> list[dict[str, Any]]:
        project = self.projects.get(project_id)
        groups: dict[str, dict[str, Any]] = {}
        for item in project["items"]:
            if not item["included"] or item["media_available"]:
                continue
            job_id = item.get("job_id")
            if item["media_state"] == "clip_removed" or not job_id:
                key = f"removed:{item['id']}"
                groups[key] = {
                    "source_key": key,
                    "label": item["title"],
                    "kind": "clip_removed",
                    "state": "clip_removed",
                    "state_label": "Clip removed",
                    "restorable": False,
                    "clip_count": 1,
                    "session_names": [item["session_name"]],
                    "estimated_bytes": 0,
                    "last_error": None,
                }
                continue
            try:
                source = self.lifecycle.source_status_for_job(str(job_id))
            except KeyError:
                source = {
                    "source_key": None,
                    "state": "unavailable",
                    "state_label": "Source unavailable",
                    "restore_available": False,
                    "local_path": None,
                }
            source_key = source.get("source_key")
            group_key = str(source_key or f"local:{job_id}")
            existing = groups.get(group_key)
            if existing:
                existing["clip_count"] += 1
                if item["session_name"] not in existing["session_names"]:
                    existing["session_names"].append(item["session_name"])
                continue

            local_path = source.get("local_path")
            restorable = bool(source_key and source.get("restore_available") and not local_path)
            state = "restorable" if restorable else (
                "manual_required" if not source_key else "unavailable"
            )
            if source.get("state") == "in_use" and not local_path:
                state = "restoring"
                restorable = False
            label = (
                f"Twitch VOD {source.get('vod_id')}"
                if source_key and source.get("vod_id")
                else item["session_name"]
            )
            estimated = 0
            preflight = None
            if restorable:
                preflight = self.lifecycle.preflight_download(source.get("twitch_url") or source_key)
                estimated = int(preflight.get("required_bytes") or 0)
            groups[group_key] = {
                "source_key": group_key,
                "label": label,
                "kind": "twitch" if source_key else "local",
                "state": state,
                "state_label": (
                    "Ready to restore" if restorable
                    else "Restoring now" if state == "restoring"
                    else "Locate the original recording" if not source_key
                    else source.get("state_label") or "Unavailable"
                ),
                "restorable": restorable,
                "clip_count": 1,
                "session_names": [item["session_name"]],
                "estimated_bytes": estimated,
                "last_error": source.get("last_error"),
                "preflight": preflight,
            }
        return list(groups.values())

    def plan(self, project_id: str) -> dict[str, Any]:
        project = self.projects.get(project_id)
        sources = self._missing_sources(project_id)
        restorable = [source for source in sources if source["restorable"]]
        estimates = [int(source["estimated_bytes"]) for source in restorable]
        estimated_bytes = sum(estimates)
        preflights = [source.get("preflight") for source in restorable if source.get("preflight")]
        free_bytes = min(
            (int(item.get("free_bytes") or 0) for item in preflights),
            default=0,
        )
        reserve_bytes = DOWNLOAD_RESERVE_BYTES if restorable else 0
        disk_allowed = not restorable or free_bytes >= estimated_bytes + reserve_bytes
        safe_sources: dict[str, dict[str, Any]] = {}
        for preflight in preflights:
            for source in preflight.get("safe_sources") or []:
                safe_sources[str(source["source_key"])] = source
        active_scan = self._has_active_scan()
        active = self._serialize_preparation(self._latest_row(project_id, active_only=True))
        blockers = []
        if active_scan:
            blockers.append("Wait for the active scan to finish before preparing sources.")
        if active:
            blockers.append("This project already has a source preparation in progress.")
        if restorable and not disk_allowed:
            blockers.append("There is not enough free disk space to restore every selected source safely.")
        if not restorable and sources:
            blockers.append("The remaining sources need manual recovery or are no longer restorable.")
        return {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "project_id": project_id,
            "project_status": project["status"],
            "included_count": project["included_count"],
            "available_count": project["available_count"],
            "missing_clip_count": max(0, project["included_count"] - project["available_count"]),
            "required_source_count": len(sources),
            "restorable_source_count": len(restorable),
            "manual_source_count": sum(source["state"] == "manual_required" for source in sources),
            "unavailable_source_count": sum(
                source["state"] in {"unavailable", "clip_removed"} for source in sources
            ),
            "estimated_download_bytes": estimated_bytes,
            "free_bytes": free_bytes,
            "reserve_bytes": reserve_bytes,
            "disk_allowed": disk_allowed,
            "active_scan": active_scan,
            "can_start": bool(restorable and disk_allowed and not active_scan and not active),
            "blockers": blockers,
            "safe_sources": list(safe_sources.values()),
            "sources": [{key: value for key, value in source.items() if key != "preflight"} for source in sources],
            "preparation": active or self._serialize_preparation(self._latest_row(project_id)),
        }

    def start(self, project_id: str, source_keys: Optional[list[str]] = None) -> dict[str, Any]:
        plan = self.plan(project_id)
        if plan["preparation"] and plan["preparation"]["status"] in ACTIVE_STATUSES:
            raise CompilationPreparationConflict("This project is already preparing source media.")
        if plan["active_scan"]:
            raise CompilationPreparationConflict(
                "Wait for the active scan to finish before preparing source media."
            )
        available = {
            source["source_key"]: source for source in plan["sources"] if source["restorable"]
        }
        selected_keys = list(dict.fromkeys(source_keys or available.keys()))
        if not selected_keys:
            raise CompilationPreparationConflict("This project has no restorable Twitch sources.")
        if any(key not in available for key in selected_keys):
            raise ValueError("Prepare only restorable sources returned by the project plan.")
        selected = [available[key] for key in selected_keys]
        estimated = sum(int(source["estimated_bytes"]) for source in selected)
        if plan["free_bytes"] < estimated + plan["reserve_bytes"]:
            raise CompilationPreparationStorageError(
                "There is not enough free disk space to restore those sources safely.",
                plan,
            )

        preparation_id = f"preparation_{uuid.uuid4().hex}"
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO compilation_preparations(
                       id, project_id, status, total_sources, estimated_bytes,
                       created_at, updated_at
                   ) VALUES (?, ?, 'queued', ?, ?, ?, ?)""",
                (preparation_id, project_id, len(selected), estimated, now, now),
            )
            for position, source in enumerate(selected):
                conn.execute(
                    """INSERT INTO compilation_preparation_items(
                           id, preparation_id, source_key, label, position,
                           status, estimated_bytes, updated_at
                       ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
                    (
                        f"preparation_item_{uuid.uuid4().hex}", preparation_id,
                        source["source_key"], source["label"], position,
                        int(source["estimated_bytes"]), now,
                    ),
                )
        threading.Thread(
            target=self._run,
            args=(preparation_id,),
            name=f"recall-compilation-preparation-{preparation_id[-8:]}",
            daemon=True,
        ).start()
        return self.plan(project_id)

    def _cancel_requested(self, preparation_id: str) -> bool:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM compilation_preparations WHERE id = ?",
                (preparation_id,),
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def _finish_item(self, item_id: str, status: str, error: Optional[str] = None) -> None:
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """UPDATE compilation_preparation_items
                   SET status = ?, error = ?, completed_at = ?, updated_at = ?
                   WHERE id = ?""",
                (status, error[:500] if error else None, now, now, item_id),
            )

    def _refresh_counts(self, preparation_id: str) -> None:
        with self.db.get_connection() as conn:
            counts = conn.execute(
                """SELECT
                       SUM(CASE WHEN status = 'prepared' THEN 1 ELSE 0 END) AS prepared,
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
                   FROM compilation_preparation_items WHERE preparation_id = ?""",
                (preparation_id,),
            ).fetchone()
            conn.execute(
                """UPDATE compilation_preparations
                   SET prepared_sources = ?, failed_sources = ?, updated_at = ? WHERE id = ?""",
                (int(counts["prepared"] or 0), int(counts["failed"] or 0), _utc_now(), preparation_id),
            )

    def _run(self, preparation_id: str) -> None:
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """UPDATE compilation_preparations
                   SET status = 'running', started_at = ?, updated_at = ? WHERE id = ?""",
                (now, now, preparation_id),
            )
            items = [
                dict(row)
                for row in conn.execute(
                    """SELECT * FROM compilation_preparation_items
                       WHERE preparation_id = ? ORDER BY position, id""",
                    (preparation_id,),
                ).fetchall()
            ]
        fatal_error = None
        for item in items:
            if self._cancel_requested(preparation_id):
                self._finish_item(item["id"], "cancelled")
                continue
            if self._has_active_scan():
                fatal_error = "A scan started before source preparation could continue."
                self._finish_item(item["id"], "failed", fatal_error)
                continue
            with self.db.get_connection() as conn:
                conn.execute(
                    """UPDATE compilation_preparations
                       SET current_source_key = ?, updated_at = ? WHERE id = ?""",
                    (item["source_key"], _utc_now(), preparation_id),
                )
                conn.execute(
                    """UPDATE compilation_preparation_items
                       SET status = 'restoring', started_at = ?, updated_at = ? WHERE id = ?""",
                    (_utc_now(), _utc_now(), item["id"]),
                )
            try:
                result = self.lifecycle.restore_source(item["source_key"])
                if result["status"] == "blocked_low_disk":
                    self._finish_item(item["id"], "failed", "Not enough free disk space.")
                    continue
                if result["status"] == "blocked_active_scan":
                    self._finish_item(item["id"], "failed", "An active scan is using Recall's work lane.")
                    continue
                while True:
                    source = self.lifecycle.source_status(item["source_key"])
                    if source.get("local_path"):
                        self._finish_item(item["id"], "prepared")
                        break
                    if source.get("state") == "unavailable":
                        self._finish_item(
                            item["id"], "failed",
                            source.get("last_error") or "The Twitch source is no longer available.",
                        )
                        break
                    if self._cancel_requested(preparation_id):
                        self.lifecycle.cancel_restore(item["source_key"])
                        if source.get("state") != "in_use":
                            self._finish_item(item["id"], "cancelled")
                            break
                    time.sleep(self.poll_seconds)
            except Exception as exc:  # keep one failed VOD from losing the queue
                self._finish_item(item["id"], "failed", str(exc))
            finally:
                self._refresh_counts(preparation_id)

        cancelled = self._cancel_requested(preparation_id)
        with self.db.get_connection() as conn:
            counts = conn.execute(
                """SELECT
                       SUM(CASE WHEN status = 'prepared' THEN 1 ELSE 0 END) AS prepared,
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
                   FROM compilation_preparation_items WHERE preparation_id = ?""",
                (preparation_id,),
            ).fetchone()
            prepared = int(counts["prepared"] or 0)
            failed = int(counts["failed"] or 0)
            status = "cancelled" if cancelled else "partial" if failed else "completed"
            completed_at = _utc_now()
            conn.execute(
                """UPDATE compilation_preparations
                   SET status = ?, prepared_sources = ?, failed_sources = ?,
                       current_source_key = NULL, error = ?, completed_at = ?, updated_at = ?
                   WHERE id = ?""",
                (status, prepared, failed, fatal_error, completed_at, completed_at, preparation_id),
            )

    def cancel(self, project_id: str) -> dict[str, Any]:
        self.projects.get(project_id)
        row = self._latest_row(project_id, active_only=True)
        if row is None:
            raise CompilationPreparationConflict("This project is not preparing source media.")
        now = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """UPDATE compilation_preparations
                   SET cancel_requested = 1, status = 'cancelling', updated_at = ? WHERE id = ?""",
                (now, row["id"]),
            )
        if row["current_source_key"]:
            self.lifecycle.cancel_restore(row["current_source_key"])
        return self.plan(project_id)

