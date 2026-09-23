# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from core.stream_memory import StreamMemoryService
from core.stream_memory_semantic import SemanticBuildCancelled


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class StreamMemoryContinuousIndexer:
    """Maintain derived Memory indexes without competing with foreground work.

    SQLite exact evidence remains the authority. This worker repairs completed
    jobs that missed their scan-time index transaction, then incrementally
    refreshes the rebuildable sentence-vector sidecar. Both paths yield when a
    scan or export appears, and persisted semantic state makes a killed update
    retryable on the next launch.
    """

    def __init__(
        self,
        service: StreamMemoryService,
        is_foreground_busy: Callable[[], bool],
        *,
        poll_seconds: float = 2.0,
    ):
        self.service = service
        self.is_foreground_busy = is_foreground_busy
        self.poll_seconds = max(0.05, float(poll_seconds))
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._exact_retry_at: Optional[datetime] = None
        self._last_result: dict[str, Any] = {"status": "idle"}

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._wake.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="recall-stream-memory-indexer",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))

    def wake(self) -> None:
        self._wake.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            result = dict(self._last_result)
        result["running"] = bool(thread is not None and thread.is_alive())
        return result

    def _cancel_requested(self) -> bool:
        return self._stop.is_set() or self.is_foreground_busy()

    def _set_result(self, **result: Any) -> dict[str, Any]:
        with self._lock:
            self._last_result = dict(result)
        return dict(result)

    def run_once(self) -> dict[str, Any]:
        """Run one bounded maintenance pass; public for deterministic tests."""
        if self._cancel_requested():
            return self._set_result(status="deferred", reason="foreground_busy")

        now = datetime.now(timezone.utc)
        exact_result = None
        if (
            (self._exact_retry_at is None or now >= self._exact_retry_at)
            and self.service.has_pending_exact_jobs()
        ):
            exact_result = self.service.reindex_completed(
                force=False,
                cancel_check=self._cancel_requested,
            )
            if exact_result.get("cancelled") or self._cancel_requested():
                return self._set_result(
                    status="preempted", phase="exact", exact=exact_result,
                )
            if int(exact_result.get("failed_jobs") or 0) > 0:
                self._exact_retry_at = now + timedelta(minutes=5)
            else:
                self._exact_retry_at = None

        semantic = self.service.semantic_state()
        source_revision = int(semantic.get("semantic_source_revision") or 0)
        indexed_revision = int(semantic.get("semantic_indexed_revision") or -1)
        source_entries = int(semantic.get("semantic_source_entries") or 0)
        semantic_status = str(semantic.get("semantic_status") or "not_built")
        if source_revision == indexed_revision and semantic_status == "ready":
            return self._set_result(status="idle", exact=exact_result)
        if source_entries < 2:
            return self._set_result(
                status="idle",
                reason="insufficient_content",
                exact=exact_result,
            )

        retry_at = _parse_utc(semantic.get("semantic_next_attempt_at"))
        if retry_at is not None and retry_at > now:
            return self._set_result(
                status="deferred",
                reason="retry_backoff",
                retry_at=retry_at.isoformat().replace("+00:00", "Z"),
                exact=exact_result,
            )
        try:
            result = self.service.build_semantic_index(
                cancel_check=self._cancel_requested,
            )
        except SemanticBuildCancelled:
            return self._set_result(status="preempted", phase="semantic", exact=exact_result)
        except (RuntimeError, ValueError) as exc:
            return self._set_result(
                status="deferred",
                reason="semantic_unavailable",
                error=str(exc)[:500],
                exact=exact_result,
            )
        except Exception as exc:  # derived maintenance must never stop the API
            return self._set_result(
                status="error",
                error=str(exc)[:500],
                exact=exact_result,
            )
        return self._set_result(status="updated", semantic=result, exact=exact_result)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as exc:  # keep later completed scans recoverable
                self._set_result(status="error", error=str(exc)[:500])
            self._wake.wait(self.poll_seconds)
            self._wake.clear()
