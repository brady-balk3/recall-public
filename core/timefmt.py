# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Shared display-time formatting.

One hour-aware ``H:MM:SS`` / ``M:SS`` formatter for every user-facing timestamp
the backend emits. The old inline ``m:ss`` produced "82:07" on multi-hour VODs;
this was previously reimplemented as ``format_vod_time`` / ``_fmt_hms`` /
``_clip_timestamp`` in three modules.
"""

from __future__ import annotations


def format_hms(seconds: float) -> str:
    """Hour-aware display timestamp: "1:22:07" past an hour, "4:06" under."""
    total = max(0, int(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
