# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Per-platform export presets + filename templating (plan 9.5).

Recall renders every clip as a 1080x1920 vertical MP4 — already valid for
TikTok, YouTube Shorts and X — and highlight clips are far shorter than any
platform's length limit, so a preset here is deliberately *not* a length or
resolution gate (those would never bite). Instead a preset is a lightweight
platform tag: it drives the ``{platform}`` filename token, tags the export row,
and lets the UI offer a one-click jump to that platform's upload page.

Filename templating rewrites the copied file's name from a small token set:

    {title}     - the clip's title (sanitized and truncated)
    {index}     - stable zero-padded clip number within the source VOD
    {date}      - recording/VOD date, YYYY-MM-DD (or "undated")
    {platform}  - the selected preset id (or "clip" when none)
    {original}  - the original export filename stem
"""

from __future__ import annotations

import os
import re
from typing import Optional

from core.source_date import normalize_source_date

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TOKEN = re.compile(r"\{(title|index|date|platform|original)\}")
TITLE_MAX_LENGTH = 64


def sanitize_filename(name: str, fallback: str = "clip") -> str:
    """Reduce an arbitrary string to a safe, non-empty filename stem."""
    name = _UNSAFE.sub(" ", name or "")
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    name = name[:120].strip()
    return name or fallback


def truncate_title(
    title: str,
    fallback: str = "Untitled",
    max_length: int = TITLE_MAX_LENGTH,
) -> str:
    """Return a safe title, preferring a word boundary when it must be cut."""
    safe = sanitize_filename(title, fallback)
    if len(safe) <= max_length:
        return safe
    clipped = safe[:max_length].rstrip()
    boundary = clipped.rfind(" ")
    if boundary >= int(max_length * 0.6):
        clipped = clipped[:boundary].rstrip()
    return clipped or sanitize_filename(fallback, "Untitled")[:max_length]


def render_filename(
    template: Optional[str],
    *,
    title: str,
    index: int,
    preset: Optional[str],
    original_stem: str,
    source_date: Optional[str] = None,
    total: Optional[int] = None,
) -> str:
    """Render a creator-facing `.mp4` filename stem.

    The default is ``YYYY-MM-DD_clip-01_Truncated title``. A missing source date
    is labeled honestly as ``undated``; the processing day is never substituted.
    """
    fallback_title = sanitize_filename(original_stem, "Untitled")
    safe_title = truncate_title(title, fallback_title)
    width = max(2, len(str(max(index, total or index))))
    clip_number = f"{index:0{width}d}"
    tokens = {
        "title": safe_title,
        "index": clip_number,
        "date": normalize_source_date(source_date) or "undated",
        "platform": (preset or "clip").lower(),
        "original": original_stem or "clip",
    }
    if template and template.strip():
        rendered = _TOKEN.sub(lambda m: tokens[m.group(1)], template)
    else:
        rendered = f"{tokens['date']}_clip-{clip_number}_{safe_title}"
    return sanitize_filename(rendered, original_stem or f"clip-{index}")


def ensure_unique(dest_folder: str, stem: str, taken: set) -> str:
    """Return a `<stem>.mp4` filename unique within dest_folder and this batch."""
    candidate = f"{stem}.mp4"
    if candidate.lower() not in taken and not os.path.exists(os.path.join(dest_folder, candidate)):
        taken.add(candidate.lower())
        return candidate
    n = 2
    while True:
        candidate = f"{stem}-{n}.mp4"
        if candidate.lower() not in taken and not os.path.exists(os.path.join(dest_folder, candidate)):
            taken.add(candidate.lower())
            return candidate
        n += 1
