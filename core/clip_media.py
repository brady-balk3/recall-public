# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Truth about whether a clip's rendered media is still on disk.

Recall's storage policy (see ``core.storage_lifecycle``) never removes rendered
clips automatically -- only downloaded sources are lifecycle-managed, and those
already carry a ``source_removed`` state plus a working restore path. But a
creator can always reclaim that space themselves, and when they do, the clip
rows survive pointing at files that are gone.

Nothing that made a clip worth keeping lives in the MP4. Score, decision,
signals, layout, framing, timestamps, and the reaction peak are all in SQLite.
So a missing export is not a dead row, it is a *rebuild recipe*: the render
chain (restore source -> ``_local_source_video`` -> ``ensure_rendered`` ->
``generate_thumbnail``) can reconstruct the exact same clip from it.

This module only classifies. It never mutates and never deletes, and in
particular it never nulls ``export_path`` -- that column is half the recipe.
Callers that need the pixels back go through ClipService; callers that need to
render honest UI just read the state.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from core.source_identity import canonical_source_key


MORE_CANDIDATE_LABEL = "recall_more_candidate"

# Clip media states, in descending order of what the creator can do with them.
READY = "ready"
"""The rendered proxy is on disk. Everything works."""

SOURCE_WINDOW = "source_window"
"""No proxy yet by design (an unrevealed Second-look candidate), but the source
VOD is present so the clip can be played by seeking its window."""

REBUILDABLE = "rebuildable"
"""The proxy is gone and can be reconstructed -- either the source is still on
disk, or it is a Twitch VOD Recall can re-download."""

UNAVAILABLE = "unavailable"
"""The proxy is gone and the source cannot be recovered. Metadata and the
creator's decision survive; the pixels do not."""

# Job source availability, used to decide between REBUILDABLE and UNAVAILABLE.
SOURCE_LOCAL = "local"
SOURCE_RESTORABLE = "restorable"
SOURCE_GONE = "gone"


def _is_file(path: Any) -> bool:
    if not path:
        return False
    try:
        return os.path.isfile(str(path))
    except (OSError, ValueError):
        return False


def job_source_availability(db: Any, job_ids: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Map every job id to ``local`` / ``restorable`` / ``gone``.

    Batched on purpose. The sessions list asks about a hundred jobs at once, and
    ``SourceLifecycleService.source_status_for_job`` costs several queries per
    call -- fine for one Cutting Room banner, far too slow for a gallery.
    """
    with db.get_connection() as conn:
        if job_ids is not None:
            ids = list(job_ids)
            if not ids:
                return {}
            placeholders = ",".join("?" * len(ids))
            jobs = conn.execute(
                "SELECT id, source_type, source_path, asset_path "  # nosec B608
                f"FROM jobs WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        else:
            jobs = conn.execute(
                "SELECT id, source_type, source_path, asset_path FROM jobs"
            ).fetchall()
        restorable_keys = {
            row["source_key"]
            for row in conn.execute(
                "SELECT source_key FROM source_assets WHERE restore_available = 1"
            ).fetchall()
        }

    availability: Dict[str, str] = {}
    for job in jobs:
        if _is_file(job["asset_path"]):
            availability[job["id"]] = SOURCE_LOCAL
            continue
        # Local recordings live wherever the creator put them; source_path is
        # the real file. For Twitch jobs source_path is a URL, never a file.
        if (job["source_type"] or "file") == "file" and _is_file(job["source_path"]):
            availability[job["id"]] = SOURCE_LOCAL
            continue
        key = canonical_source_key(job["source_path"] or "", job["asset_path"] or "")
        availability[job["id"]] = (
            SOURCE_RESTORABLE if key in restorable_keys else SOURCE_GONE
        )
    return availability


def _get(row: Any, key: str) -> Any:
    """Read a column from either a sqlite3.Row or a plain dict."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def clip_media_state(row: Any, source_availability: str) -> str:
    """Classify one clip row against its job's source availability."""
    export_path = _get(row, "export_path")
    if _is_file(export_path):
        return READY

    never_rendered = not export_path and _get(row, "story_label") == MORE_CANDIDATE_LABEL
    if never_rendered and source_availability == SOURCE_LOCAL:
        # Legacy Second-look rows have always played by seeking the source.
        # That is not a degraded state, so don't dress it up as one.
        return SOURCE_WINDOW

    if source_availability in (SOURCE_LOCAL, SOURCE_RESTORABLE):
        return REBUILDABLE
    return UNAVAILABLE


def annotate_clip_rows(
    db: Any,
    rows: List[Dict[str, Any]],
    availability: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Add ``media_state`` to already-fetched clip dicts, in place.

    One stat per clip and one batched source lookup for the whole set, so the
    cost is the same whether the caller asked for a single clip or the library.
    """
    if not rows:
        return rows
    if availability is None:
        availability = job_source_availability(
            db, sorted({row.get("job_id") for row in rows if row.get("job_id")})
        )
    for row in rows:
        row["media_state"] = clip_media_state(
            row, availability.get(row.get("job_id"), SOURCE_GONE)
        )
    return rows


def job_media_summary(
    db: Any, job_ids: Optional[Sequence[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Per-session clip media counts for the sessions gallery and rebuild UI.

    Returns ``{job_id: {ready, rebuildable, unavailable, source_window, total,
    source_state}}``. Second-look candidates are counted like any other clip:
    the creator revealed them or they are still latent, but either way the
    session's media health is about the same files.
    """
    availability = job_source_availability(db, job_ids)
    with db.get_connection() as conn:
        if job_ids is not None:
            ids = list(job_ids)
            if not ids:
                return {}
            placeholders = ",".join("?" * len(ids))
            clips = conn.execute(
                "SELECT id, job_id, export_path, story_label "  # nosec B608
                f"FROM clips WHERE job_id IN ({placeholders})",
                ids,
            ).fetchall()
        else:
            clips = conn.execute(
                "SELECT id, job_id, export_path, story_label FROM clips"
            ).fetchall()

    summary: Dict[str, Dict[str, Any]] = {
        job_id: {
            READY: 0,
            SOURCE_WINDOW: 0,
            REBUILDABLE: 0,
            UNAVAILABLE: 0,
            "total": 0,
            "source_state": state,
        }
        for job_id, state in availability.items()
    }
    for clip in clips:
        bucket = summary.get(clip["job_id"])
        if bucket is None:
            continue
        bucket[clip_media_state(clip, bucket["source_state"])] += 1
        bucket["total"] += 1
    return summary


def summary_state(summary: Dict[str, Any]) -> str:
    """Collapse one session's counts into the single state a card should show."""
    if summary.get("total", 0) == 0:
        return READY
    if summary.get(REBUILDABLE, 0) == 0 and summary.get(UNAVAILABLE, 0) == 0:
        return READY
    if summary.get(READY, 0) or summary.get(SOURCE_WINDOW, 0):
        return "partial"
    return REBUILDABLE if summary.get(REBUILDABLE, 0) else UNAVAILABLE
