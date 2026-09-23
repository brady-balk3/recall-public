# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Persistent per-VOD clip numbering for creator-facing filenames."""

from __future__ import annotations

from typing import Iterable


def ensure_clip_numbers(conn, job_ids: Iterable[str]) -> None:
    """Append stable numbers to any legacy/null-numbered clips in each job.

    A job's first numbering pass is chronological. Once numbers exist, later
    manual or second-look clips append after the highest value so an exported
    clip's identity never changes.
    """
    for job_id in dict.fromkeys(str(value) for value in job_ids if value):
        row = conn.execute(
            "SELECT COALESCE(MAX(clip_number), 0) FROM clips WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        next_number = int(row[0] or 0) + 1
        missing = conn.execute(
            """SELECT id FROM clips
               WHERE job_id = ? AND clip_number IS NULL
               ORDER BY COALESCE(start_time, 0), COALESCE(end_time, 0), id""",
            (job_id,),
        ).fetchall()
        for clip_row in missing:
            conn.execute(
                "UPDATE clips SET clip_number = ? WHERE id = ?",
                (next_number, clip_row[0]),
            )
            next_number += 1


def next_clip_number(conn, job_id: str) -> int:
    """Return the next append-only number after repairing legacy rows."""
    ensure_clip_numbers(conn, [job_id])
    row = conn.execute(
        "SELECT COALESCE(MAX(clip_number), 0) + 1 FROM clips WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    return int(row[0] or 1)
