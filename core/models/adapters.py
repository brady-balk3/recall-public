# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

from typing import Iterable, List, Optional

from core.models.canonical import (
    Asset,
    Clip,
    Event,
    Export,
    ProvenanceLink,
    SCHEMA_VERSION,
    Signal,
    Story,
)


def asset_from_video(asset_id: str, source_path: str, duration: float) -> Asset:
    return Asset(
        id=asset_id,
        type="asset",
        asset_id=asset_id,
        timestamp=0.0,
        confidence=1.0,
        source_engine="ingest",
        schema_version=SCHEMA_VERSION,
        source=source_path,
        duration=float(duration or 0.0),
    )


def signals_from_legacy(signals: Iterable, asset_id: str) -> List[Signal]:
    out: List[Signal] = []
    for idx, sig in enumerate(signals or []):
        timestamp = float(getattr(sig, "timestamp", idx) or 0.0)
        metadata = {}
        strengths = []
        if getattr(sig, "audio", None):
            audio = sig.audio
            strengths.append(float(getattr(audio, "spike_score", 0.0) or 0.0))
            metadata["audio"] = audio.to_dict() if hasattr(audio, "to_dict") else getattr(audio, "__dict__", {})
        if getattr(sig, "ocr", None):
            ocr = sig.ocr
            strengths.append(float(getattr(ocr, "confidence", 0.0) or 0.0))
            metadata["ocr"] = getattr(ocr, "__dict__", {})
        if getattr(sig, "vision", None):
            vision = sig.vision
            strengths.append(float(getattr(vision, "motion_score", 0.0) or 0.0))
            metadata["vision"] = getattr(vision, "__dict__", {})
        strength = max(strengths) if strengths else 0.0
        out.append(Signal(
            id=f"sig_{idx:06d}",
            type="signal",
            asset_id=asset_id,
            timestamp=timestamp,
            confidence=max(0.0, min(1.0, strength)),
            source_engine="perception",
            signal_type="unified",
            strength=strength,
            metadata=metadata,
        ))
    return out


def events_from_legacy(events: Iterable, asset_id: str) -> List[Event]:
    out: List[Event] = []
    for idx, ev in enumerate(events or []):
        event_id = getattr(ev, "event_id", None) or f"evt_{idx:06d}"
        out.append(Event(
            id=event_id,
            type="event",
            asset_id=asset_id,
            timestamp=float(getattr(ev, "timestamp_start", 0.0) or 0.0),
            confidence=float(getattr(ev, "confidence", 0.0) or 0.0),
            source_engine="event-engine",
            event_type=getattr(ev, "event_type", "unknown"),
            timestamp_end=float(getattr(ev, "timestamp_end", 0.0) or 0.0),
            signal_ids=list(getattr(ev, "signals", []) or []),
            score=float(getattr(ev, "score", 0.0) or 0.0),
            metadata=getattr(ev, "metadata", {}) or {},
        ))
    return out


def stories_from_legacy(stories: Iterable, asset_id: str) -> List[Story]:
    out: List[Story] = []
    for idx, story in enumerate(stories or []):
        story_id = getattr(story, "story_id", None) or f"story_{idx:06d}"
        event_ids = [
            getattr(ev, "event_id", None) or getattr(ev, "id", None) or f"event_{i}"
            for i, ev in enumerate(getattr(story, "events", []) or [])
        ]
        out.append(Story(
            id=story_id,
            type="story",
            asset_id=asset_id,
            timestamp=float(getattr(story, "start", 0.0) or 0.0),
            confidence=float(getattr(story, "score", 0.0) or 0.0),
            source_engine="story-engine",
            provenance=[
                ProvenanceLink(object_id=eid, object_type="event", relationship="groups")
                for eid in event_ids
                if eid
            ],
            start_time=float(getattr(story, "start", 0.0) or 0.0),
            end_time=float(getattr(story, "end", 0.0) or 0.0),
            event_ids=event_ids,
            importance=float(getattr(story, "score", 0.0) or 0.0),
            label=getattr(story, "label", ""),
        ))
    return out


def clips_from_legacy(clips: Iterable, asset_id: str) -> List[Clip]:
    out: List[Clip] = []
    for idx, clip in enumerate(clips or []):
        clip_id = getattr(clip, "clip_id", None) or f"clip_{idx:06d}"
        story_id = getattr(clip, "story_id", "") or ""
        out.append(Clip(
            id=clip_id,
            type="clip",
            asset_id=asset_id,
            timestamp=float(getattr(clip, "start", 0.0) or 0.0),
            confidence=float(getattr(clip, "score", 0.0) or 0.0),
            source_engine="clip-engine",
            provenance=[
                ProvenanceLink(
                    object_id=story_id,
                    object_type="story",
                    relationship="renders",
                    metadata={
                        "reason": getattr(clip, "reason", None),
                        "peak_timestamp": getattr(clip, "peak_timestamp", None),
                    },
                )
            ] if story_id else [],
            story_id=story_id,
            start_time=float(getattr(clip, "start", 0.0) or 0.0),
            end_time=float(getattr(clip, "end", 0.0) or 0.0),
            layout=getattr(clip, "layout", {}) or {},
            metadata={
                "semantic_verdict": getattr(clip, "semantic_verdict", None),
                "semantic_moment_type": getattr(clip, "semantic_moment_type", None),
                "semantic_title": getattr(clip, "semantic_title", None),
                "features": getattr(clip, "features", {}) or {},
            },
        ))
    return out


def exports_from_paths(exported_paths: Iterable[str], clips: Iterable, asset_id: str) -> List[Export]:
    legacy_clips = list(clips or [])
    out: List[Export] = []
    for idx, path in enumerate(exported_paths or []):
        clip_ref: Optional[str] = None
        for clip in legacy_clips:
            cid = getattr(clip, "clip_id", "")
            if cid and f"clip_{cid}.mp4" in str(path):
                clip_ref = cid
                break
        clip_ref = clip_ref or (getattr(legacy_clips[idx], "clip_id", "") if idx < len(legacy_clips) else "")
        out.append(Export(
            id=f"export_{idx:06d}",
            type="export",
            asset_id=asset_id,
            timestamp=0.0,
            confidence=1.0,
            source_engine="export-engine",
            provenance=[
                ProvenanceLink(object_id=clip_ref, object_type="clip", relationship="exports")
            ] if clip_ref else [],
            clip_id=clip_ref,
            file_path=str(path),
        ))
    return out
