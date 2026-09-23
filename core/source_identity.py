# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Stable source identities shared by training, replay, and runtime."""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit, urlunsplit


_TWITCH_URL_RE = re.compile(r"(?:twitch\.tv/videos/|twitchvod_v)(\d+)", re.I)


def canonical_source_key(source_path: str, asset_path: str = "") -> str:
    """Return one identity for every rescan or local copy of the same VOD."""
    for raw in (source_path, asset_path):
        match = _TWITCH_URL_RE.search(str(raw or ""))
        if match:
            return f"twitch:{match.group(1)}"

    source = str(source_path or asset_path or "").strip()
    if not source:
        return "unknown:"
    parsed = urlsplit(source)
    if parsed.scheme and parsed.netloc:
        normalized = urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                "",
                "",
            )
        )
        return f"url:{normalized}"
    return "file:" + os.path.normcase(os.path.abspath(source))
