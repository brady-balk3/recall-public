# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Shared cache-key derivation (pipeline + API service must agree exactly).

The extracted WAV and the full-VOD transcript depend ONLY on the source file's
audio and (for transcripts) the ASR's cache label — never on vision/OCR/facecam
settings. Keying them on the content-only key means a sensitivity-slider
change doesn't re-extract or re-transcribe, and the API service can rebuild
the same key from the video path alone (it doesn't know the scan's settings).
"""

from __future__ import annotations

import hashlib
import json
import os
import re


def audio_cache_key(video_path: str) -> str:
    """Content-only key (path + size + mtime) for audio-derived artifacts.

    v2: multitrack recordings now route analysis to the mic-most stream
    (plan 22 §4.4). Track selection is deterministic per file, so the key can
    stay content-only, but v1 wavs were always the default mix — the version
    bump retires them (and their transcripts) instead of mixing semantics.
    """
    stat = os.stat(video_path)
    raw = f"audio-16k-mono-v2|{os.path.abspath(video_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


def transcript_cache_name(key: str, model_label: str, ext: str, fast: bool = False) -> str:
    """Filename for a cached transcript.

    ``fast`` transcripts cover only the scout regions picked by that scan's
    settings, so callers key them on the settings-dependent video hash;
    full transcripts use the content-only audio key.
    """
    # v2: scout transcripts record covered_regions (unheard != silent).
    prefix = "fast_speech_v2" if fast else "transcript"
    return f"{key}_{prefix}_{model_label}{ext}"


def face_cache_name(video_hash: str, scorer_label: str, ext: str,
                    region_hash: str = None) -> str:
    """Filename for cached facecam emotion frames.

    The scorer label is part of the key because the face scorer is selectable
    (RECALL_FACE_SCORER) and the two scorers produce different arousal on the
    same footage. Without it, switching scorers silently reuses the other one's
    frames: perception "completes" in under a second and the scan reports
    healthy numbers computed from the wrong model, which is undetectable
    downstream. Changing the blendshape feature changes the values too, so the
    label carries that as well.
    """
    stem = f"{video_hash}_faces_{scorer_label}"
    if region_hash:
        stem = f"{stem}_r{region_hash}"
    return f"{stem}{ext}"


def region_set_hash(regions) -> str:
    """Stable hash of a scouted region set (smart scan, HUMAN_CLIPS plan A §4).

    Region-scoped artifacts (facecam emotion, chat) cover only the ``(start,
    end)`` second ranges the scout picked, so their cache entries must key on
    the region set as well as the video: identical regions on a re-scan reuse
    the artifact, and any added/removed/moved region changes the key so stale
    data can never be served. Endpoints are rounded to 0.1s so float noise in
    an otherwise-identical region set can't change the key.
    """
    parts = "|".join(
        f"{float(start):.1f}-{float(end):.1f}"
        for start, end in sorted((float(s), float(e)) for s, e in (regions or []))
    )
    return hashlib.md5(f"regions-v1|{parts}".encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


# --------------------------------------------------------------- source index
#
# Cache file names start with an opaque key, so the storage janitor cannot tell
# from a name which VOD a file belongs to. The settings-independent audio key
# it can recompute from any source file still on disk (path + size + mtime);
# the settings-dependent video hashes and the visual judge's verdict keys it
# cannot, so each scan records them here, in one small index per source named
# after that source's audio key. The janitor keeps whatever an index of a
# source still on disk lists, and treats everything it cannot attribute as
# belonging to no source. An index is advisory: a missing or broken one only
# makes its files look orphaned, never keeps another source's files.

CACHE_INDEX_SUFFIX = "_cache_index_v1.json"

# Video hashes are md5 (32 hex), verdict keys sha256 (64 hex). Anything else
# in an index is ignored: the janitor builds paths to delete from these.
_INDEX_KEY = re.compile(r"[0-9a-f]{32}|[0-9a-f]{64}")


def cache_index_path(cache_dir: str, audio_key: str) -> str:
    return os.path.join(cache_dir, f"{audio_key}{CACHE_INDEX_SUFFIX}")


def read_cache_index(cache_dir: str, audio_key: str) -> dict:
    """``{"video_hashes": set, "visual_verdicts": set}``; empty on any problem."""
    index = {"video_hashes": set(), "visual_verdicts": set()}
    try:
        with open(cache_index_path(cache_dir, audio_key), "r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError):
        return index
    if not isinstance(stored, dict):
        return index
    for field in index:
        values = stored.get(field)
        if isinstance(values, list):
            index[field].update(
                value for value in values
                if isinstance(value, str) and _INDEX_KEY.fullmatch(value)
            )
    return index


def record_cache_index(cache_dir: str, audio_key: str, *, video_hashes=(),
                       visual_verdicts=()) -> None:
    """Add keys to a source's index (atomic rewrite). Raises OSError; callers
    treat a failure as best-effort, since it only costs the files' protection."""
    index = read_cache_index(cache_dir, audio_key)
    added = False
    for field, values in (("video_hashes", video_hashes), ("visual_verdicts", visual_verdicts)):
        for value in values:
            if isinstance(value, str) and _INDEX_KEY.fullmatch(value) and value not in index[field]:
                index[field].add(value)
                added = True
    if not added:
        return
    path = cache_index_path(cache_dir, audio_key)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump({field: sorted(values) for field, values in index.items()}, handle)
    os.replace(tmp_path, path)
