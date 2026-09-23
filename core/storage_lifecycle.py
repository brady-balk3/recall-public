# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Authoritative storage inventory and lifecycle policy for Recall.

Only disposable Recall-owned data is ever removed automatically. Rendered clips,
thumbnails, session records, and creator-owned local recordings are deliberately
outside every automatic deletion path.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from core.bundle_paths import get_data_dir
from core.source_identity import canonical_source_key


GIB = 1024 ** 3
DEFAULT_RECALL_CAP_GB = 25.0
DEFAULT_SOURCE_RETENTION_DAYS = 7
DOWNLOAD_RESERVE_BYTES = 2 * GIB
UNKNOWN_DOWNLOAD_ESTIMATE_BYTES = 4 * GIB

_SOURCE_USE_LOCK = threading.RLock()
_ACTIVE_SOURCE_KEYS: dict[str, int] = {}
_RESTORE_CANCEL: dict[str, threading.Event] = {}


def has_active_source_restores() -> bool:
    """Return whether Recall is actively restoring any managed Twitch source."""
    with _SOURCE_USE_LOCK:
        return bool(_RESTORE_CANCEL)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _inside(root: str, path: str) -> bool:
    """True only when path resolves beneath root (never root itself)."""
    if not path:
        return False
    try:
        root_abs = os.path.realpath(os.path.abspath(root))
        path_abs = os.path.realpath(os.path.abspath(path))
        return path_abs != root_abs and os.path.commonpath([root_abs, path_abs]) == root_abs
    except (OSError, ValueError):
        return False


def _file_size(path: Optional[str]) -> int:
    try:
        return os.path.getsize(path) if path and os.path.isfile(path) else 0
    except OSError:
        return 0


@contextlib.contextmanager
def source_in_use(db: Any, job_id: str):
    """Protect a source while an editor or render operation is using it."""
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT source_key FROM source_asset_sessions WHERE job_id = ?", (job_id,)
        ).fetchone()
    key = row["source_key"] if row else None
    if key:
        with _SOURCE_USE_LOCK:
            _ACTIVE_SOURCE_KEYS[key] = _ACTIVE_SOURCE_KEYS.get(key, 0) + 1
        try:
            SourceLifecycleService(db).touch_source(key)
            yield
        finally:
            with _SOURCE_USE_LOCK:
                remaining = _ACTIVE_SOURCE_KEYS.get(key, 1) - 1
                if remaining > 0:
                    _ACTIVE_SOURCE_KEYS[key] = remaining
                else:
                    _ACTIVE_SOURCE_KEYS.pop(key, None)
    else:
        yield


class SourceLifecycleService:
    def __init__(self, db: Any, job_manager: Any = None, data_root: Optional[str] = None):
        self.db = db
        self.job_manager = job_manager
        self.data_root = os.path.abspath(data_root or get_data_dir())
        self.assets_root = os.path.join(self.data_root, "assets")
        self.cache_root = os.path.join(self.data_root, "cache")
        self.temp_root = os.path.join(self.data_root, "temp")
        self.exports_root = os.path.join(self.data_root, "exports")
        self.jobs_root = os.path.join(self.data_root, "jobs")

    def _setting_float(self, key: str, default: float, *, minimum: float, maximum: float) -> float:
        try:
            value = float(self.db.get_setting(key, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    @property
    def cap_gb(self) -> float:
        return self._setting_float(
            "recallStorageCapGb", DEFAULT_RECALL_CAP_GB, minimum=1.0, maximum=2000.0
        )

    @property
    def retention_days(self) -> int:
        return int(
            round(
                self._setting_float(
                    "sourceRetentionDays",
                    DEFAULT_SOURCE_RETENTION_DAYS,
                    minimum=1.0,
                    maximum=365.0,
                )
            )
        )

    def _has_active_scan(self) -> bool:
        # The in-process job manager knows about work that has not reached the
        # jobs table yet, so it gets the first say before we fall back to the DB.
        if self.job_manager is not None and self.job_manager.has_active_jobs():
            return True
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE status IN ('pending','running','cancelling') LIMIT 1"
            ).fetchone()
        return row is not None

    @staticmethod
    def _folder_size(path: str) -> int:
        total = 0
        if not os.path.isdir(path):
            return 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                item = os.path.join(dirpath, filename)
                try:
                    if not os.path.islink(item):
                        total += os.path.getsize(item)
                except OSError:
                    continue
        return total

    @staticmethod
    def _entries_oldest_first(path: str) -> list[tuple[float, int, str]]:
        entries: list[tuple[float, int, str]] = []
        if not os.path.isdir(path):
            return entries
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                item = os.path.join(dirpath, filename)
                try:
                    if os.path.islink(item):
                        continue
                    stat = os.stat(item)
                except OSError:
                    continue
                entries.append((stat.st_mtime, stat.st_size, item))
        return sorted(entries)

    def _database_size(self) -> int:
        base = os.path.abspath(self.db.db_path)
        return sum(_file_size(base + suffix) for suffix in ("", "-wal", "-shm"))

    def _proxy_state(self, job_ids: Iterable[str]) -> bool:
        job_ids = list(job_ids)
        if not job_ids:
            return True
        placeholders = ",".join("?" * len(job_ids))
        with self.db.get_connection() as conn:
            rows = conn.execute(
                f"SELECT export_path FROM clips WHERE job_id IN ({placeholders})",  # nosec B608
                job_ids,
            ).fetchall()
        return all(row["export_path"] and os.path.isfile(row["export_path"]) for row in rows)

    def ensure_source_for_job(
        self,
        job_id: str,
        source_path: str,
        asset_path: Optional[str] = None,
    ) -> Optional[str]:
        source_key = canonical_source_key(source_path or "", asset_path or "")
        if not source_key.startswith("twitch:"):
            return None
        local_path = os.path.abspath(asset_path) if asset_path and os.path.isfile(asset_path) else None
        if local_path and not _inside(self.assets_root, local_path):
            raise ValueError("Recall-managed Twitch sources must stay inside data/assets.")
        now = _utcnow()
        size = _file_size(local_path)
        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO source_assets(
                    source_key, twitch_url, vod_id, local_path, file_size_bytes,
                    downloaded_at, last_used_at, lifecycle_state, restore_available
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(source_key) DO UPDATE SET
                    twitch_url = excluded.twitch_url,
                    vod_id = excluded.vod_id,
                    local_path = COALESCE(excluded.local_path, source_assets.local_path),
                    file_size_bytes = CASE WHEN excluded.local_path IS NOT NULL
                        THEN excluded.file_size_bytes ELSE source_assets.file_size_bytes END,
                    downloaded_at = CASE WHEN excluded.local_path IS NOT NULL
                        THEN COALESCE(source_assets.downloaded_at, excluded.downloaded_at)
                        ELSE source_assets.downloaded_at END,
                    last_used_at = excluded.last_used_at,
                    lifecycle_state = CASE WHEN excluded.local_path IS NOT NULL
                        THEN 'available' ELSE source_assets.lifecycle_state END,
                    restore_available = 1,
                    last_error = NULL,
                    updated_at = excluded.last_used_at
                """,
                (
                    source_key,
                    source_path,
                    source_key.split(":", 1)[1],
                    local_path,
                    size,
                    _iso(now) if local_path else None,
                    _iso(now),
                    "available" if local_path else "source_removed",
                ),
            )
            conn.execute("DELETE FROM source_asset_sessions WHERE job_id = ?", (job_id,))
            conn.execute(
                "INSERT INTO source_asset_sessions(source_key, job_id) VALUES (?, ?)",
                (source_key, job_id),
            )
        return source_key

    def mark_scan_complete(self, job_id: str) -> Optional[str]:
        with self.db.get_connection() as conn:
            link = conn.execute(
                "SELECT source_key FROM source_asset_sessions WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not link:
            return None
        source_key = link["source_key"]
        with self.db.get_connection() as conn:
            linked = [
                row["job_id"]
                for row in conn.execute(
                    "SELECT job_id FROM source_asset_sessions WHERE source_key = ?", (source_key,)
                ).fetchall()
            ]
        now = _utcnow()
        expires = now + timedelta(days=self.retention_days)
        proxies_ok = self._proxy_state(linked)
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE source_assets SET scan_completed = 1,
                    review_proxies_verified = ?, last_used_at = ?, retention_expires_at = ?,
                    lifecycle_state = CASE WHEN local_path IS NULL THEN 'source_removed' ELSE 'available' END,
                    updated_at = ? WHERE source_key = ?
                """,
                (int(proxies_ok), _iso(now), _iso(expires), _iso(now), source_key),
            )
        return source_key

    def touch_source(self, source_key: str) -> None:
        now = _utcnow()
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE source_assets SET last_used_at = ?, retention_expires_at = ?, "
                "updated_at = ? WHERE source_key = ?",
                (
                    _iso(now),
                    _iso(now + timedelta(days=self.retention_days)),
                    _iso(now),
                    source_key,
                ),
            )

    def _is_active_source(self, source_key: str) -> bool:
        with _SOURCE_USE_LOCK:
            if _ACTIVE_SOURCE_KEYS.get(source_key, 0) > 0:
                return True
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM source_asset_sessions sas
                JOIN jobs j ON j.id = sas.job_id
                WHERE sas.source_key = ? AND j.status IN ('pending','running','cancelling')
                LIMIT 1
                """,
                (source_key,),
            ).fetchone()
        return row is not None

    def _source_row(self, source_key: str):
        with self.db.get_connection() as conn:
            return conn.execute(
                "SELECT * FROM source_assets WHERE source_key = ?", (source_key,)
            ).fetchone()

    def _eligible(self, row: Any, now: Optional[datetime] = None) -> bool:
        if not row or not row["local_path"] or int(row["pinned"] or 0):
            return False
        if self._is_active_source(row["source_key"]):
            return False
        expires = _parse_timestamp(row["retention_expires_at"])
        return bool(
            int(row["scan_completed"] or 0)
            and int(row["review_proxies_verified"] or 0)
            and int(row["restore_available"] or 0)
            and expires
            and expires <= (now or _utcnow())
        )

    def _delete_managed_file(self, path: Optional[str], allowed_root: str) -> int:
        if not path or not _inside(allowed_root, path):
            return 0
        size = _file_size(path)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
        except FileNotFoundError:
            pass
        return size

    def remove_source(self, source_key: str, *, require_eligible: bool = True) -> Dict[str, Any]:
        row = self._source_row(source_key)
        if not row:
            raise KeyError(source_key)
        if self._is_active_source(source_key):
            raise RuntimeError("This source is in use. Wait for the active operation to finish.")
        if require_eligible and not self._eligible(row):
            raise RuntimeError("This source is still inside its retention window or is kept on this PC.")
        path = row["local_path"]
        if path and not _inside(self.assets_root, path):
            raise RuntimeError("Refusing to remove a source outside Recall's data/assets folder.")
        removed = self._delete_managed_file(path, self.assets_root)
        now = _iso(_utcnow())
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET asset_path = NULL WHERE id IN "
                "(SELECT job_id FROM source_asset_sessions WHERE source_key = ?)",
                (source_key,),
            )
            conn.execute(
                "UPDATE source_assets SET local_path = NULL, lifecycle_state = 'source_removed', "
                "updated_at = ?, last_error = NULL WHERE source_key = ?",
                (now, source_key),
            )
        return {"source_key": source_key, "bytes_removed": removed, "state": "source_removed"}

    def set_pinned(self, source_key: str, pinned: bool) -> Dict[str, Any]:
        if not self._source_row(source_key):
            raise KeyError(source_key)
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE source_assets SET pinned = ?, lifecycle_state = CASE "
                "WHEN ? THEN 'kept_on_pc' WHEN local_path IS NULL THEN 'source_removed' "
                "ELSE 'available' END, updated_at = ? WHERE source_key = ?",
                (int(pinned), int(pinned), _iso(_utcnow()), source_key),
            )
        return self.source_status(source_key)

    def reconcile_sources(self, *, remove_expired: bool = True) -> Dict[str, Any]:
        # Pick up legacy or newly-created jobs even if a scan was interrupted
        # before the normal tracking hook ran.
        with self.db.get_connection() as conn:
            jobs = conn.execute(
                "SELECT id, source_path, asset_path, status FROM jobs"
            ).fetchall()
        for job in jobs:
            key = canonical_source_key(job["source_path"] or "", job["asset_path"] or "")
            with self.db.get_connection() as conn:
                linked = conn.execute(
                    "SELECT 1 FROM source_asset_sessions WHERE job_id = ?", (job["id"],)
                ).fetchone()
            if key.startswith("twitch:") and not linked:
                self.ensure_source_for_job(job["id"], job["source_path"] or "", job["asset_path"])

        removed = 0
        now = _utcnow()
        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT * FROM source_assets ORDER BY created_at").fetchall()
        for row in rows:
            path = row["local_path"]
            if path and (not _inside(self.assets_root, path) or not os.path.isfile(path)):
                with self.db.get_connection() as conn:
                    conn.execute(
                        "UPDATE jobs SET asset_path = NULL WHERE id IN "
                        "(SELECT job_id FROM source_asset_sessions WHERE source_key = ?)",
                        (row["source_key"],),
                    )
                    conn.execute(
                        "UPDATE source_assets SET local_path = NULL, lifecycle_state = 'source_removed', "
                        "updated_at = ? WHERE source_key = ?",
                        (_iso(now), row["source_key"]),
                    )
                row = self._source_row(row["source_key"])
            elif path:
                with self.db.get_connection() as conn:
                    linked_ids = [
                        item["job_id"]
                        for item in conn.execute(
                            "SELECT job_id FROM source_asset_sessions WHERE source_key = ?",
                            (row["source_key"],),
                        ).fetchall()
                    ]
                last_used = _parse_timestamp(row["last_used_at"]) or now
                expires = _parse_timestamp(row["retention_expires_at"])
                if int(row["scan_completed"] or 0) and expires is None:
                    expires = last_used + timedelta(days=self.retention_days)
                with self.db.get_connection() as conn:
                    conn.execute(
                        "UPDATE source_assets SET file_size_bytes = ?, "
                        "review_proxies_verified = ?, retention_expires_at = COALESCE(?, retention_expires_at), "
                        "lifecycle_state = CASE WHEN pinned = 1 THEN 'kept_on_pc' ELSE 'available' END, "
                        "updated_at = ? WHERE source_key = ?",
                        (
                            _file_size(path),
                            int(self._proxy_state(linked_ids)),
                            _iso(expires) if expires else None,
                            _iso(now),
                            row["source_key"],
                        ),
                    )
                row = self._source_row(row["source_key"])
            if remove_expired and not self._has_active_scan() and self._eligible(row, now):
                removed += self.remove_source(row["source_key"], require_eligible=False)["bytes_removed"]
        return {"bytes_removed": removed}

    def _source_view(self, row: Any, now: Optional[datetime] = None) -> Dict[str, Any]:
        now = now or _utcnow()
        key = row["source_key"]
        active = self._is_active_source(key) or row["lifecycle_state"] == "restoring"
        local = row["local_path"] if row["local_path"] and os.path.isfile(row["local_path"]) else None
        expires = _parse_timestamp(row["retention_expires_at"])
        if active:
            state, label = "in_use", "In use"
        elif int(row["pinned"] or 0):
            state, label = "kept_on_pc", "Kept on this PC"
        elif not local:
            if int(row["restore_available"] or 0):
                state, label = "source_removed", "Source removed"
            else:
                state, label = "unavailable", "Unavailable"
        elif self._eligible(row, now):
            state, label = "safe_to_remove", "Safe to remove"
        else:
            remaining = max(0, (expires - now).total_seconds()) if expires else 0
            days = max(1, int((remaining + 86399) // 86400)) if remaining else self.retention_days
            state, label = "available", f"Available for {days} more day{'s' if days != 1 else ''}"
        with self.db.get_connection() as conn:
            sessions = [
                {"id": linked["id"], "name": linked["session_name"] or "Untitled session"}
                for linked in conn.execute(
                    """
                    SELECT j.id, j.session_name FROM jobs j
                    JOIN source_asset_sessions sas ON sas.job_id = j.id
                    WHERE sas.source_key = ? ORDER BY j.created_at DESC
                    """,
                    (key,),
                ).fetchall()
            ]
        return {
            "source_key": key,
            "vod_id": row["vod_id"],
            "twitch_url": row["twitch_url"],
            "local_path": local,
            "file_size_bytes": _file_size(local) or int(row["file_size_bytes"] or 0),
            "downloaded_at": row["downloaded_at"],
            "last_used_at": row["last_used_at"],
            "retention_expires_at": row["retention_expires_at"],
            "pinned": bool(row["pinned"]),
            "state": state,
            "state_label": label,
            "restore_available": bool(row["restore_available"]),
            "last_error": row["last_error"],
            "sessions": sessions,
        }

    def list_sources(self) -> list[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM source_assets ORDER BY COALESCE(last_used_at, created_at) DESC"
            ).fetchall()
        now = _utcnow()
        return [self._source_view(row, now) for row in rows]

    def source_status(self, source_key: str) -> Dict[str, Any]:
        row = self._source_row(source_key)
        if not row:
            raise KeyError(source_key)
        return self._source_view(row)

    def source_status_for_job(self, job_id: str) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT source_key FROM source_asset_sessions WHERE job_id = ?", (job_id,)
            ).fetchone()
            job = conn.execute(
                "SELECT source_type, source_path, asset_path FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if not job:
            raise KeyError(job_id)
        if not row:
            # Local creator recordings are never lifecycle-managed.
            path = job["asset_path"] or job["source_path"]
            return {
                "source_key": None,
                "state": "local" if path and os.path.isfile(path) else "unavailable",
                "state_label": "Local recording" if path and os.path.isfile(path) else "Unavailable",
                "restore_available": False,
                "local_path": path if path and os.path.isfile(path) else None,
                "pinned": True,
            }
        return self.source_status(row["source_key"])

    def preflight_download(self, source_path: str) -> Dict[str, Any]:
        source_key = canonical_source_key(source_path or "")
        if not source_key.startswith("twitch:"):
            return {"allowed": True, "required_bytes": 0, "free_bytes": 0, "safe_sources": []}
        row = self._source_row(source_key)
        if row and row["local_path"] and os.path.isfile(row["local_path"]):
            return {"allowed": True, "required_bytes": 0, "free_bytes": 0, "safe_sources": []}
        # Older installs may already have the canonical download on disk before
        # a source_assets row exists. The resolver will reuse it, so preflight
        # must not reserve another full VOD's worth of space.
        vod_id = source_key.split(":", 1)[1]
        for extension in (".mp4", ".mkv", ".webm"):
            candidate = os.path.join(self.assets_root, f"twitchvod_v{vod_id}{extension}")
            if os.path.isfile(candidate) and _file_size(candidate) > 0:
                return {"allowed": True, "required_bytes": 0, "free_bytes": 0, "safe_sources": []}
        os.makedirs(self.assets_root, exist_ok=True)
        free = shutil.disk_usage(self.assets_root).free
        estimate = int(row["file_size_bytes"] or 0) if row else 0
        estimate = max(estimate, UNKNOWN_DOWNLOAD_ESTIMATE_BYTES)
        safe = [source for source in self.list_sources() if source["state"] == "safe_to_remove"]
        return {
            "allowed": free >= estimate + DOWNLOAD_RESERVE_BYTES,
            "required_bytes": estimate,
            "free_bytes": free,
            "reserve_bytes": DOWNLOAD_RESERVE_BYTES,
            "safe_sources": safe,
        }

    def restore_source(self, source_key: str) -> Dict[str, Any]:
        row = self._source_row(source_key)
        if not row:
            raise KeyError(source_key)
        if row["local_path"] and os.path.isfile(row["local_path"]):
            self.touch_source(source_key)
            return {"status": "already_available", "source": self.source_status(source_key)}
        if self._has_active_scan():
            return {
                "status": "blocked_active_scan",
                "source": self.source_status(source_key),
            }
        with _SOURCE_USE_LOCK:
            if _ACTIVE_SOURCE_KEYS.get(source_key, 0) > 0:
                return {"status": "already_running", "source": self.source_status(source_key)}
        preflight = self.preflight_download(row["twitch_url"] or "")
        if not preflight["allowed"]:
            return {"status": "blocked_low_disk", "preflight": preflight, "source": self.source_status(source_key)}

        cancel = threading.Event()
        with _SOURCE_USE_LOCK:
            _ACTIVE_SOURCE_KEYS[source_key] = _ACTIVE_SOURCE_KEYS.get(source_key, 0) + 1
            _RESTORE_CANCEL[source_key] = cancel
        with self.db.get_connection() as conn:
            linked = conn.execute(
                "SELECT job_id FROM source_asset_sessions WHERE source_key = ? ORDER BY linked_at LIMIT 1",
                (source_key,),
            ).fetchone()
            conn.execute(
                "UPDATE source_assets SET lifecycle_state = 'restoring', last_error = NULL, updated_at = ? "
                "WHERE source_key = ?",
                (_iso(_utcnow()), source_key),
            )
        event_job_id = linked["job_id"] if linked else None

        def emit(event_type: str, message: str, progress: Optional[float] = None) -> None:
            if self.job_manager is not None and event_job_id:
                self.job_manager.emit_event(
                    event_job_id,
                    event_type,
                    phase="Restore source",
                    message=message,
                    progress=progress,
                    payload={"source_key": source_key},
                )

        def worker() -> None:
            try:
                from pipeline.ingest.vod_resolver import resolve_input

                emit("source_restore_started", "Restoring the Twitch source.", 0.0)

                def progress(message: str, value: float, _eta: Any = None) -> None:
                    if cancel.is_set():
                        raise InterruptedError("Source restoration cancelled")
                    emit("source_restore_progress", message, value)

                restored = resolve_input(
                    row["twitch_url"], output_dir=self.assets_root, progress_callback=progress,
                    cancel_check=cancel.is_set,
                )
                restored = os.path.abspath(restored)
                if not _inside(self.assets_root, restored) or not os.path.isfile(restored):
                    raise RuntimeError("Restoration did not produce a managed file in data/assets.")
                now = _utcnow()
                with self.db.get_connection() as conn:
                    conn.execute(
                        "UPDATE jobs SET asset_path = ? WHERE id IN "
                        "(SELECT job_id FROM source_asset_sessions WHERE source_key = ?)",
                        (restored, source_key),
                    )
                    conn.execute(
                        """
                        UPDATE source_assets SET local_path = ?, file_size_bytes = ?,
                            downloaded_at = ?, last_used_at = ?, retention_expires_at = ?,
                            lifecycle_state = 'available', restore_available = 1,
                            last_error = NULL, updated_at = ? WHERE source_key = ?
                        """,
                        (
                            restored,
                            _file_size(restored),
                            _iso(now),
                            _iso(now),
                            _iso(now + timedelta(days=self.retention_days)),
                            _iso(now),
                            source_key,
                        ),
                    )
                emit("source_restored", "Source restored and ready.", 1.0)
            except InterruptedError:
                with self.db.get_connection() as conn:
                    conn.execute(
                        "UPDATE source_assets SET lifecycle_state = 'source_removed', "
                        "last_error = NULL, updated_at = ? WHERE source_key = ?",
                        (_iso(_utcnow()), source_key),
                    )
                emit("source_restore_cancelled", "Source restoration cancelled.")
            except Exception as exc:
                with self.db.get_connection() as conn:
                    conn.execute(
                        "UPDATE source_assets SET lifecycle_state = 'unavailable', restore_available = 0, "
                        "last_error = ?, updated_at = ? WHERE source_key = ?",
                        (str(exc)[:500], _iso(_utcnow()), source_key),
                    )
                emit("source_restore_failed", "The Twitch source could not be restored.")
            finally:
                with _SOURCE_USE_LOCK:
                    remaining = _ACTIVE_SOURCE_KEYS.get(source_key, 1) - 1
                    if remaining > 0:
                        _ACTIVE_SOURCE_KEYS[source_key] = remaining
                    else:
                        _ACTIVE_SOURCE_KEYS.pop(source_key, None)
                    _RESTORE_CANCEL.pop(source_key, None)

        threading.Thread(
            target=worker, name=f"recall-source-restore-{row['vod_id']}", daemon=True
        ).start()
        return {"status": "restoring", "source": self.source_status(source_key)}

    def cancel_restore(self, source_key: str) -> Dict[str, Any]:
        with _SOURCE_USE_LOCK:
            event = _RESTORE_CANCEL.get(source_key)
        if not event:
            return {"status": "not_restoring"}
        event.set()
        return {"status": "cancelling"}

    def clear_safe_cache(self) -> Dict[str, Any]:
        if self._has_active_scan():
            raise RuntimeError("Wait for the active scan to finish before clearing scan data.")
        removed = 0
        for _, size, path in self._entries_oldest_first(self.cache_root):
            if not _inside(self.cache_root, path):
                continue
            try:
                os.remove(path)
                removed += size
            except OSError:
                continue
        os.makedirs(self.cache_root, exist_ok=True)
        return {"bytes_removed": removed}

    def _clean_abandoned_temp(self) -> int:
        if self._has_active_scan():
            return 0
        removed = self._folder_size(self.temp_root)
        if os.path.isdir(self.temp_root):
            shutil.rmtree(self.temp_root, ignore_errors=True)
        os.makedirs(self.temp_root, exist_ok=True)
        return removed

    def reconcile_recall_storage(self) -> Dict[str, Any]:
        cap_bytes = int(self.cap_gb * GIB)
        before = self._recall_categories()
        temp_removed = 0
        cache_removed = 0
        if not self._has_active_scan():
            temp_removed = self._clean_abandoned_temp()
            total = sum(before.values())
            if total > cap_bytes:
                for _, size, path in self._entries_oldest_first(self.cache_root):
                    if total <= cap_bytes:
                        break
                    if not _inside(self.cache_root, path):
                        continue
                    try:
                        os.remove(path)
                    except OSError:
                        continue
                    total -= size
                    cache_removed += size
        after = self._recall_categories()
        total_after = sum(after.values())
        overage = max(0, total_after - cap_bytes)
        return {
            "cap_bytes": cap_bytes,
            "categories": after,
            "total_bytes": total_after,
            "reclaimable_bytes": after["scan_data"] + self._folder_size(self.temp_root),
            "overage_bytes": overage,
            "cache_bytes_removed": cache_removed,
            "temp_bytes_removed": temp_removed,
            "cleanup_deferred": self._has_active_scan(),
            "message": (
                f"Recall is still {overage / GIB:.1f} GB over its target after safe cache cleanup. "
                "Clips and required session data were not removed."
                if overage
                else "Recall storage is within its target."
            ),
        }

    def _recall_categories(self) -> Dict[str, int]:
        return {
            "clips": self._folder_size(self.exports_root),
            "scan_data": self._folder_size(self.cache_root),
            # candidates.v1 and manifests remain because More Moments and
            # diagnostics consume them; they are required session data.
            "session_data": self._database_size() + self._folder_size(self.jobs_root),
        }

    def storage_report(self) -> Dict[str, Any]:
        source_cleanup = self.reconcile_sources(remove_expired=True)
        recall = self.reconcile_recall_storage()
        sources = self.list_sources()
        local_sources = [source for source in sources if source["local_path"]]
        safe = [source for source in sources if source["state"] == "safe_to_remove"]
        expiries = [
            _parse_timestamp(source["retention_expires_at"])
            for source in local_sources
            if not source["pinned"] and source["retention_expires_at"]
        ]
        expiries = [value for value in expiries if value is not None]
        return {
            "recall": recall,
            "downloaded_sources": {
                "count": len(local_sources),
                "total_bytes": sum(source["file_size_bytes"] for source in local_sources),
                "safe_to_remove_count": len(safe),
                "next_scheduled_removal": _iso(min(expiries)) if expiries else None,
                "retention_days": self.retention_days,
                "sources": sources,
                "bytes_removed": source_cleanup["bytes_removed"],
            },
        }
