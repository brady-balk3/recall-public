# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Versioned, non-executable signal cache (plan 22 §1.2, replaces pickle).

Why not pickle: pickle deserialization executes arbitrary callables from the
stream (unsafe if a cache file is ever tampered with or shared), and a pickled
dataclass silently breaks when its module path or field set changes between
app versions — the load either explodes mid-scan or, worse, yields stale
objects with missing attributes.

Format: gzip-compressed JSON with an explicit envelope::

    {"format": "recall-signal-cache", "version": 1, "payload": ...}

Dataclasses are encoded as {"__dc__": "<ClassName>", ...fields} against a
fixed registry of known cacheable types. Anything wrong — unknown class,
version mismatch, truncated file, missing required field after a model
change — makes ``load`` return None and the pipeline rebuilds that artifact.
A cache is always rebuildable by design; refusing is the safe failure mode.

Numpy scalars/arrays are coerced to plain floats/ints/lists on encode, and
tuples become lists; every consumer reads these values numerically or by
index, so the round-trip is behavior-preserving.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import os
from typing import Any, Dict, Optional, Type

FORMAT_NAME = "recall-signal-cache"
FORMAT_VERSION = 1

# Preferred extension for new cache files (callers build their own filenames).
CACHE_EXT = ".cache.json.gz"

_TYPE_KEY = "__dc__"
_registry: Optional[Dict[str, Type]] = None


def _build_registry() -> Dict[str, Type]:
    """Every dataclass the pipeline caches. Lazy imports keep core importable
    even if an optional engine module is missing in a stripped build."""
    registry: Dict[str, Type] = {}

    def _add(*classes):
        for cls in classes:
            registry[cls.__name__] = cls

    from core.models.signal import AudioSignal, OCRSignal, VisionSignal, UnifiedSignal
    _add(AudioSignal, OCRSignal, VisionSignal, UnifiedSignal)

    for module_path, names in (
        ("engines.audio.prosody", ("ProsodyFrame",)),
        ("engines.audio.audio_events", ("AudioEventFrame",)),
        ("engines.audio.asr_hype", ("HypeFrame",)),
        ("engines.emotion.face_emotion", ("FaceFrame",)),
        ("engines.chat.chat_features", ("ChatMessage", "ChatFrame")),
    ):
        try:
            module = __import__(module_path, fromlist=list(names))
            _add(*(getattr(module, n) for n in names))
        except Exception:
            continue
    return registry


def _get_registry() -> Dict[str, Type]:
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


def _encode(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: Dict[str, Any] = {_TYPE_KEY: type(obj).__name__}
        for f in dataclasses.fields(obj):
            out[f.name] = _encode(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    # Numpy scalars/arrays without importing numpy (works when absent too).
    item = getattr(obj, "item", None)
    if item is not None and type(obj).__module__ == "numpy":
        try:
            return obj.item()
        except (ValueError, TypeError):
            return [_encode(v) for v in obj.tolist()]
    tolist = getattr(obj, "tolist", None)
    if tolist is not None and type(obj).__module__ == "numpy":
        return _encode(obj.tolist())
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        type_name = obj.get(_TYPE_KEY)
        if type_name is None:
            return {k: _decode(v) for k, v in obj.items()}
        cls = _get_registry().get(type_name)
        if cls is None:
            raise ValueError(f"unknown cached type: {type_name}")
        field_names = {f.name for f in dataclasses.fields(cls)}
        # Unknown fields (removed from the model since caching) are dropped;
        # a missing *required* field raises TypeError -> load returns None.
        kwargs = {
            k: _decode(v) for k, v in obj.items()
            if k != _TYPE_KEY and k in field_names
        }
        return cls(**kwargs)
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def save(path: str, payload: Any) -> None:
    """Atomically write ``payload`` (dataclasses/dicts/lists/primitives)."""
    envelope = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "payload": _encode(payload),
    }
    tmp_path = f"{path}.tmp"
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(envelope, f, separators=(",", ":"))
    os.replace(tmp_path, path)


def load(path: str) -> Optional[Any]:
    """Read a cache file; None on any mismatch/corruption (caller rebuilds)."""
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            envelope = json.load(f)
        if (
            not isinstance(envelope, dict)
            or envelope.get("format") != FORMAT_NAME
            or envelope.get("version") != FORMAT_VERSION
        ):
            return None
        return _decode(envelope["payload"])
    except Exception:
        return None
