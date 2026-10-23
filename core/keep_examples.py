# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Pick a few of the creator's own past Keep/Pass decisions for the keep judge.

The keep judge calibrates on these. Measured 2026-10-13: six examples (three
kept, three passed) raised within-scan ordering from ~0.62 to 0.674; twenty
random examples made it worse (0.634), so the count stays small.

Read-only and best-effort: a fresh install with no reviews, a missing
transcript index, or any database error returns an empty list and the judge
runs without examples.
"""

import random
import sqlite3
from typing import List, Optional

MIN_SEC, MAX_SEC = 10.0, 45.0
MIN_WORDS = 8
RECENT_POOL = 300
TEXT_CHARS = 280


def select_keep_examples(
    db_path: str,
    *,
    n_keep: int = 3,
    n_pass: int = 3,
    exclude_job_id: Optional[str] = None,
    before: Optional[str] = None,
    streamer: Optional[str] = None,
    seed: int = 20261012,
) -> List[dict]:
    """``before`` (an ISO/SQLite timestamp) limits examples to decisions made
    earlier, so an offline replay of a past scan sees only what it could have.

    ``streamer`` prefers decisions on that channel's VODs: same-channel
    examples measured 0.674 against 0.640 for recent ones from any channel.
    When the channel has too few usable decisions (or is unknown), examples
    come from every channel instead."""
    if streamer:
        same = _select(db_path, n_keep, n_pass, exclude_job_id, before, streamer, seed)
        if same:
            return same
    return _select(db_path, n_keep, n_pass, exclude_job_id, before, None, seed)


def _select(db_path, n_keep, n_pass, exclude_job_id, before, streamer, seed) -> List[dict]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            """SELECT c.id, c.job_id, c.start_time, c.end_time, l.label
                 FROM clip_labels l
                 JOIN clips c ON c.id = l.clip_id
                 JOIN jobs j ON j.id = c.job_id
                WHERE l.event IN ('saved', 'passed')
                  AND (? IS NULL OR l.created_at < ?)
                  AND (? IS NULL OR LOWER(j.streamer) = LOWER(?))
                  AND l.id = (SELECT MAX(l2.id) FROM clip_labels l2
                               WHERE l2.clip_id = l.clip_id
                                 AND l2.event IN ('saved', 'passed'))
                ORDER BY l.id DESC
                LIMIT ?""",
            (before, before, streamer, streamer, RECENT_POOL),
        ).fetchall()
        pool = {1: [], 0: []}
        for clip_id, job_id, start, end, label in rows:
            if job_id == exclude_job_id or start is None or end is None:
                continue
            duration = float(end) - float(start)
            if not MIN_SEC <= duration <= MAX_SEC:
                continue
            text = " ".join(
                str(t) for (t,) in conn.execute(
                    """SELECT text FROM stream_memory_entries
                        WHERE job_id = ? AND kind = 'transcript'
                          AND end_time > ? AND start_time < ?
                        ORDER BY start_time""",
                    (job_id, float(start), float(end)),
                )
            ).strip()
            if len(text.split()) < MIN_WORDS:
                continue
            key = 1 if int(label or 0) >= 1 else 0
            pool[key].append({"label": key, "dur": duration, "text": text[:TEXT_CHARS], "id": clip_id})
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    if len(pool[1]) < n_keep or len(pool[0]) < n_pass:
        return []
    rng = random.Random(seed)
    picked = rng.sample(sorted(pool[1], key=lambda e: e["id"]), n_keep) + rng.sample(
        sorted(pool[0], key=lambda e: e["id"]), n_pass,
    )
    rng.shuffle(picked)
    return [{k: v for k, v in example.items() if k != "id"} for example in picked]
