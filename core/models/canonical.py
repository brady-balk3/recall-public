# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION = "1.0.0"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ProvenanceLink:
    object_id: str
    object_type: str
    relationship: str
    source_engine: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class CanonicalObject:
    id: str
    type: str
    asset_id: str
    timestamp: float
    confidence: float
    source_engine: str
    schema_version: str = SCHEMA_VERSION
    provenance: List[ProvenanceLink] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class Asset(CanonicalObject):
    source: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class Detection(CanonicalObject):
    label: str = ""
    bounds: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class Signal(CanonicalObject):
    signal_type: str = ""
    strength: float = 0.0


@dataclass(frozen=True)
class Event(CanonicalObject):
    event_type: str = ""
    timestamp_end: float = 0.0
    signal_ids: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass(frozen=True)
class Story(CanonicalObject):
    start_time: float = 0.0
    end_time: float = 0.0
    event_ids: List[str] = field(default_factory=list)
    importance: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class Clip(CanonicalObject):
    story_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    platform: str = "local"
    layout: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Export(CanonicalObject):
    clip_id: str = ""
    file_path: str = ""
    format: str = "mp4"
    platform: str = "local"
    status: str = "ready"


def validate_traceability(objects: List[CanonicalObject]) -> List[str]:
    """Return human-readable invariant violations for canonical objects."""
    by_id = {obj.id: obj for obj in objects}
    errors: List[str] = []

    for obj in objects:
        if not obj.id:
            errors.append(f"{obj.type} is missing id")
        if not obj.asset_id and obj.type != "asset":
            errors.append(f"{obj.type}:{obj.id} is missing asset_id")
        if obj.confidence < 0.0 or obj.confidence > 1.0:
            errors.append(f"{obj.type}:{obj.id} confidence is outside 0..1")
        for link in obj.provenance:
            if link.object_id and link.object_id not in by_id:
                errors.append(
                    f"{obj.type}:{obj.id} provenance references missing "
                    f"{link.object_type}:{link.object_id}"
                )

    return errors
