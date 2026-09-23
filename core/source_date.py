# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Resolve the recording date used in creator-facing export filenames.

The scan/job creation timestamp is intentionally not a fallback here: a VOD
may be processed days after it was recorded. Twitch supplies an authoritative
creation timestamp; local recordings prefer their embedded media timestamp and
fall back to the file's modified time.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import date, datetime
from typing import Optional

from core.ffmpeg_path import get_ffmpeg_path


def normalize_source_date(value: object) -> Optional[str]:
    """Return ``YYYY-MM-DD`` in local time for an ISO date/timestamp."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value.isoformat()
    else:
        raw = str(value).strip()
        if not raw:
            return None
        try:
            if len(raw) == 10:
                return date.fromisoformat(raw).isoformat()
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.date().isoformat()


def _ffprobe_path() -> Optional[str]:
    ffmpeg = get_ffmpeg_path()
    if ffmpeg and os.path.isabs(ffmpeg):
        extension = ".exe" if ffmpeg.lower().endswith(".exe") else ""
        sibling = os.path.join(os.path.dirname(ffmpeg), f"ffprobe{extension}")
        if os.path.isfile(sibling):
            return sibling
    return shutil.which("ffprobe")


def _embedded_creation_time(path: str) -> Optional[str]:
    ffprobe = _ffprobe_path()
    if ffprobe:
        try:
            proc = subprocess.run(
                [
                    ffprobe,
                    "-v", "error",
                    "-show_entries", "format_tags=creation_time:stream_tags=creation_time",
                    "-of", "json",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            payload = json.loads(proc.stdout or "{}") if proc.returncode == 0 else {}
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            payload = {}

        candidates = []
        format_tags = (payload.get("format") or {}).get("tags") or {}
        candidates.extend(format_tags.items())
        for stream in payload.get("streams") or []:
            candidates.extend(((stream or {}).get("tags") or {}).items())
        for key, value in candidates:
            if str(key).lower() == "creation_time":
                normalized = normalize_source_date(value)
                if normalized:
                    return normalized

    # The packaged app currently ships ffmpeg.exe without ffprobe.exe. FFmpeg's
    # ordinary input inspection prints the same global/stream creation_time to
    # stderr, so retain embedded-date support in that runtime as well.
    ffmpeg = get_ffmpeg_path()
    if ffmpeg and not os.path.isabs(ffmpeg):
        ffmpeg = shutil.which(ffmpeg)
    if not ffmpeg:
        return None
    try:
        proc = subprocess.run(
            [ffmpeg, "-i", path, "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (proc.stderr or "").splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key.strip().lower() == "creation_time":
            normalized = normalize_source_date(value.strip())
            if normalized:
                return normalized
    return None


def local_recording_date(path: str) -> Optional[str]:
    """Read a local recording's embedded date, then its local modified date."""
    if not path or not os.path.isfile(path):
        return None
    embedded = _embedded_creation_time(path)
    if embedded:
        return embedded
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def source_date_for_input(
    source_path: str,
    source_type: str = "file",
    provided: object = None,
) -> Optional[str]:
    """Resolve a supplied date or discover one for a local recording."""
    normalized = normalize_source_date(provided)
    if normalized:
        return normalized
    looks_remote = str(source_path or "").lower().startswith(("http://", "https://"))
    if source_type in {"url", "twitch"} or looks_remote:
        return None
    return local_recording_date(source_path)
