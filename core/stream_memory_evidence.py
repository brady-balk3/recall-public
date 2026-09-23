# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Build compact, rebuildable Stream Memory evidence from saved scan artifacts.

Evidence packs deliberately summarize artifacts that Recall already produced. They
never decode source media, run a model, or change selection/review truth.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


EVIDENCE_PACK_VERSION = 4
MAX_STANDALONE_EVIDENCE = 120
MAX_STANDALONE_CHAT_EVIDENCE = 40
MAX_OCR_PHRASES = 6
MAX_CHAT_PHRASES = 5
MAX_VISUAL_SCENES = 2
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
_EVENT_TERMS = (
    "victory", "victory royale", "winner", "won", "win", "elimination",
    "eliminated", "defeat", "defeated", "you died", "death", "dead",
    "boss", "round won", "round lost", "game over", "knocked", "clutch",
    "rank up", "rank progress", "match complete",
)
_STRUCTURED_GAME_LABELS = {
    "boss", "boss_fight", "clutch", "death", "defeat", "elimination",
    "knock", "rank_progress", "round_end", "terminal_win", "victory", "win",
}
_ROUTINE_OCR = {
    "back", "career", "chat", "compete", "continue", "emote", "esrb",
    "inventory", "locker", "menu", "options", "passes", "play", "quests",
    "settings", "shop", "spectate", "store",
}
_REACTION_LABELS = {
    "burst": "reaction burst",
    "chat": "chat burst",
    "face": "facecam reaction",
    "game": "gameplay reaction",
    "loud_onset": "audio spike",
    "motion": "high motion",
    "startle": "startle",
    "voice": "voice reaction",
}


@dataclass(frozen=True)
class EvidenceBuildResult:
    clip_packs: dict[str, dict]
    standalone_entries: list[dict]
    status: str
    source_bytes: int
    coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def pack_count(self) -> int:
        return len(self.clip_packs) + len(self.standalone_entries)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split()).strip()


def _slug_label(value: Any) -> str:
    return _clean(value).replace("_", " ").replace("-", " ").lower()


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _load_json(path: str, fallback: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, TypeError, ValueError):
        return fallback


def _iter_json_array(path: str) -> Iterable[dict]:
    """Yield a top-level JSON array without retaining a whole VOD artifact."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    with open(path, "r", encoding="utf-8") as handle:
        eof = False
        while not eof:
            chunk = handle.read(1024 * 1024)
            if chunk:
                buffer += chunk
            else:
                eof = True
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        return
                    started = True
                    position += 1
                    continue
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        return
                    break
                position = end
                if isinstance(value, dict):
                    yield value
                if position >= 4 * 1024 * 1024:
                    buffer = buffer[position:]
                    position = 0


def _hydrate_cached_visual_descriptions(
    data_root: str,
    source_path: Optional[str],
    candidates: list[dict],
) -> int:
    """Recover prior VLM observations only when the cache key proves identity.

    Old candidate traces retained the visual verdict and exact judge window but
    dropped the short summary strings. When the original source and exact local
    model still exist, VisualVerdictCache can deterministically reconstruct the
    key and recover the pristine JSON without decoding a frame or loading the
    model. A miss is ordinary and never falls back to fuzzy cache matching.
    """
    if not source_path or not os.path.isfile(source_path) or not candidates:
        return 0
    try:
        from engines.semantic.visual_judge import (
            DEFAULT_FRAME_COUNT,
            FALSE_SKIP_REJUDGE_FRAMES,
            VisualVerdictCache,
            resolve_models,
        )

        pair = resolve_models()
        if not pair:
            return 0
        cache = VisualVerdictCache(
            os.path.join(data_root, "cache", "visual_judge"),
        )
        model_path = pair[0]
    except Exception:
        return 0

    source_bytes = 0
    for candidate in candidates:
        if _clean(candidate.get("visual_summary")) or not candidate.get("visual_verdict"):
            continue
        attempts = []
        if candidate.get("visual_false_skip_rejudged"):
            attempts.append((
                max(DEFAULT_FRAME_COUNT, FALSE_SKIP_REJUDGE_FRAMES),
                "false-skip-rejudge-v1",
            ))
        attempts.append((DEFAULT_FRAME_COUNT, ""))
        for frame_count, pass_tag in attempts:
            try:
                key = cache.key(
                    source_path,
                    candidate,
                    model_path,
                    frame_count,
                    pass_tag=pass_tag,
                )
                raw = cache.load(key)
            except Exception:
                raw = None
            if not raw or not _clean(raw.get("summary")):
                continue
            candidate["visual_summary"] = raw.get("summary")
            candidate["visual_evidence"] = raw.get("visual_evidence")
            candidate["_visual_description_source"] = "visual_judge_cache"
            try:
                source_bytes += os.path.getsize(
                    os.path.join(cache.cache_dir, f"{key}.json"),
                )
            except OSError:
                pass
            break
    return source_bytes


def _window(candidate: dict) -> tuple[float, float, float]:
    start = _finite(candidate.get("selection_start"), _finite(candidate.get("start")))
    end = _finite(candidate.get("selection_end"), _finite(candidate.get("end"), start))
    end = max(start, end)
    peak = _finite(candidate.get("peak_timestamp"), (start + end) / 2.0)
    return start, end, peak


def _overlap(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _matching_candidate(candidates: list[dict], entry: dict) -> Optional[tuple[int, dict]]:
    start = _finite(entry.get("start_time"))
    end = max(start, _finite(entry.get("end_time"), start))
    midpoint = (start + end) / 2.0
    best: Optional[tuple[float, int, dict]] = None
    for index, candidate in enumerate(candidates):
        candidate_start, candidate_end, peak = _window(candidate)
        shared = _overlap(start, end, candidate_start, candidate_end)
        distance = abs(peak - midpoint)
        if shared <= 0.0 and distance > 4.0:
            continue
        coverage = shared / max(1.0, end - start)
        score = coverage * 4.0 + 1.0 / (1.0 + distance)
        if best is None or score > best[0]:
            best = (score, index, candidate)
    return (best[1], best[2]) if best else None


def _append_unique(values: list[str], value: Any) -> None:
    cleaned = _clean(value)
    if cleaned and cleaned.casefold() not in {item.casefold() for item in values}:
        values.append(cleaned)


def _bounded_visual_text(value: Any, limit: int) -> str:
    cleaned = _clean(value)
    if len(cleaned) < 8:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[: limit + 1]
    if not clipped[limit].isspace():
        clipped = clipped[:limit].rsplit(" ", 1)[0]
    else:
        clipped = clipped[:limit]
    return clipped.rstrip(" ,;:-")


def _candidate_visual_scene(candidate: dict) -> Optional[dict]:
    description = _bounded_visual_text(candidate.get("visual_summary"), 160)
    if not description or candidate.get("visual_routine_only"):
        return None
    support = _bounded_visual_text(candidate.get("visual_evidence"), 140)
    verdict = _slug_label(candidate.get("visual_verdict"))
    outcome = _slug_label(candidate.get("visual_outcome")).replace(" ", "_")
    confidence = _finite(candidate.get("visual_confidence"))
    grounded = bool(
        outcome not in {"", "none", "menu_or_shop"}
        or candidate.get("visual_event")
        or candidate.get("visual_streamer_reaction")
        or candidate.get("creator_protected")
        or candidate.get("recall_marker_ids")
        or (verdict in {"post", "maybe"} and confidence >= 0.55)
    )
    if not grounded:
        return None
    return {
        "description": description,
        "support": support or None,
        "outcome": outcome or None,
        "confidence": round(confidence, 4),
        "source": _clean(candidate.get("_visual_description_source")) or "candidate_trace",
    }


def _candidate_evidence(candidate: Optional[dict], candidate_index: Optional[int]) -> dict:
    labels: list[str] = []
    events: list[str] = []
    reactions: list[str] = []
    ocr: list[str] = []
    marker_ids: list[str] = []
    provenance: list[dict] = []
    visual_scenes: list[dict] = []
    layout: dict[str, bool] = {}
    if not candidate:
        return {
            "labels": labels, "events": events, "reactions": reactions,
            "ocr": ocr, "marker_ids": marker_ids, "provenance": provenance,
            "layout": layout, "visual_scenes": visual_scenes,
        }

    game_label = _slug_label(candidate.get("game_label"))
    scene_label = _slug_label(candidate.get("scene_label"))
    semantic_type = _slug_label(candidate.get("semantic_moment_type"))
    visual_type = _slug_label(candidate.get("visual_moment_type"))
    visual_text = _clean(candidate.get("visual_onscreen_text"))
    if game_label and game_label not in {"generic", "gameplay", "unknown"}:
        _append_unique(events, game_label)
    if scene_label and scene_label not in {"generic", "gameplay", "unknown"}:
        _append_unique(labels, scene_label)
    for moment_type in (semantic_type, visual_type):
        if moment_type and moment_type not in {"filler", "generic", "unknown"}:
            _append_unique(labels, moment_type)
    if visual_text:
        _append_unique(ocr, visual_text[:120])
    if candidate.get("visual_event"):
        _append_unique(events, "visual gameplay event")
    if candidate.get("visual_streamer_reaction"):
        _append_unique(reactions, "streamer reaction")
        _append_unique(reactions, "facecam reaction")
    if candidate.get("strong_startle"):
        _append_unique(reactions, "strong startle")
        _append_unique(labels, "scary moment")
    if candidate.get("crowd_clip"):
        _append_unique(reactions, "viewer clip request")
    visual_scene = _candidate_visual_scene(candidate)
    if visual_scene:
        visual_scenes.append(visual_scene)

    breakdown = candidate.get("modality_breakdown") or {}
    if isinstance(breakdown, dict):
        for key, label in _REACTION_LABELS.items():
            threshold = 0.55 if key in {"burst", "loud_onset", "startle"} else 0.65
            if _finite(breakdown.get(key)) >= threshold:
                _append_unique(reactions, label)

    marker_ids = [
        _clean(value) for value in (candidate.get("recall_marker_ids") or [])
        if _clean(value)
    ]
    if marker_ids or candidate.get("creator_protected"):
        _append_unique(labels, "creator marked live")
        _append_unique(labels, "Remember button")
    if candidate_index is not None:
        provenance.append({
            "artifact": "candidates.v1.json",
            "candidate_index": candidate_index,
            "disposition": _clean(candidate.get("disposition")) or None,
        })
    return {
        "labels": labels,
        "events": events,
        "reactions": reactions,
        "ocr": ocr,
        "marker_ids": marker_ids,
        "marker_time": candidate.get("recall_marker_time"),
        "provenance": provenance,
        "layout": layout,
        "visual_scenes": visual_scenes,
        "confidence": _finite(candidate.get("visual_confidence")),
    }


def _is_qualified_standalone(candidate: dict) -> bool:
    features = candidate.get("features") or {}
    game_label = _slug_label(candidate.get("game_label")).replace(" ", "_")
    moment_type = _slug_label(candidate.get("visual_moment_type"))
    confidence = _finite(candidate.get("visual_confidence"))
    return bool(
        candidate.get("creator_protected")
        or candidate.get("recall_marker_ids")
        or candidate.get("strong_startle")
        or _finite(features.get("is_win")) >= 1.0
        or _finite(features.get("visual_win")) >= 0.65
        or game_label in _STRUCTURED_GAME_LABELS
        or (
            candidate.get("visual_event")
            and confidence >= 0.65
            and moment_type not in {"", "filler", "generic", "unknown"}
        )
    )


def _coverage_summary(
    candidates: list[dict],
    chat_payload: dict,
    packs: Iterable[dict],
) -> dict[str, Any]:
    """Measure bounded evidence availability without interpreting missing truth."""
    marker_candidates = [
        row for row in candidates
        if row.get("creator_protected") or row.get("recall_marker_ids")
    ]
    strong_unselected = [
        row for row in candidates
        if not str(row.get("disposition") or "").startswith("selected")
        and not (row.get("creator_protected") or row.get("recall_marker_ids"))
        and _is_qualified_standalone(row)
    ]

    def described(rows: Iterable[dict]) -> int:
        return sum(_candidate_visual_scene(row) is not None for row in rows)

    chat_windows = [
        row for row in (chat_payload.get("windows") or []) if isinstance(row, dict)
    ]
    pack_rows = [row for row in packs if isinstance(row, dict)]
    return {
        "candidate_count": len(candidates),
        "creator_marker_candidates": len(marker_candidates),
        "creator_marker_visual_descriptions": described(marker_candidates),
        "strong_unselected_candidates": len(strong_unselected),
        "strong_unselected_visual_descriptions": described(strong_unselected),
        "admitted_visual_scenes": sum(
            len(row.get("visual_scenes") or []) for row in pack_rows
        ),
        "chat_artifact": bool(chat_payload),
        "chat_source_scope": _clean(chat_payload.get("source_scope")) or None,
        "chat_windows": len(chat_windows),
        "chat_phrase_windows": sum(
            bool(row.get("repeated_phrases")) for row in chat_windows
        ),
        "chat_clip_intents": sum(
            max(0, int(_finite(row.get("clip_intent_count")))) for row in chat_windows
        ),
        "admitted_chat_packs": sum(bool(row.get("chat_activity")) for row in pack_rows),
    }


def _entry_phrase(value: Any) -> str:
    cleaned = _clean(value).strip("-_:|/\\.,;!?[](){}")
    if len(cleaned) < 3 or len(cleaned) > 64 or not _WORD_RE.search(cleaned):
        return ""
    normalized = _slug_label(cleaned)
    if normalized in _ROUTINE_OCR:
        return ""
    return cleaned


def _event_phrase(value: str) -> bool:
    normalized = _slug_label(value)
    return any(term in normalized for term in _EVENT_TERMS)


def _signal_ocr_phrases(ocr: dict) -> Iterable[tuple[str, float]]:
    metadata = ocr.get("metadata") or {}
    gameplay_text = _slug_label(metadata.get("gameplay_text"))
    entries = metadata.get("entries") or []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            phrase = _entry_phrase(entry.get("text"))
            if not phrase:
                continue
            normalized = _slug_label(phrase)
            if gameplay_text and normalized not in gameplay_text and not _event_phrase(phrase):
                continue
            yield phrase, _finite(entry.get("confidence"), _finite(ocr.get("confidence")))
        return
    phrase = _entry_phrase(ocr.get("text"))
    if phrase:
        yield phrase, _finite(ocr.get("confidence"))


def _target_for_signal(targets: list[dict], timestamp: float) -> Iterable[dict]:
    for target in targets:
        if target["start_time"] - 3.0 <= timestamp <= target["end_time"] + 3.0:
            yield target


def _apply_signals(targets: list[dict], signals: Iterable[dict]) -> None:
    if not targets:
        return
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        timestamp = _finite(signal.get("timestamp"))
        metadata = signal.get("metadata") or {}
        for target in _target_for_signal(targets, timestamp):
            target["_samples"] += 1
            audio = metadata.get("audio") or {}
            if audio.get("is_spike"):
                target["_audio_spikes"] += 1
                target["_max_spike"] = max(
                    target["_max_spike"], _finite(audio.get("spike_score")),
                )
            vision = metadata.get("vision") or {}
            motion = _finite(vision.get("motion_score"))
            target["_motion_sum"] += motion
            target["_motion_max"] = max(target["_motion_max"], motion)
            if vision.get("facecam_box"):
                target["_facecam"] = True
            gameplay_box = vision.get("gameplay_box")
            if isinstance(gameplay_box, list) and len(gameplay_box) == 4:
                full_frame = all(
                    abs(_finite(value) - expected) <= 0.02
                    for value, expected in zip(gameplay_box, (0.0, 0.0, 1.0, 1.0))
                )
                target["_windowed_gameplay"] = target["_windowed_gameplay"] or not full_frame
            ocr = metadata.get("ocr") or {}
            for phrase, confidence in _signal_ocr_phrases(ocr):
                key = _slug_label(phrase)
                current = target["_ocr"].setdefault(key, {
                    "text": phrase, "count": 0, "confidence": 0.0,
                    "distance": float("inf"), "event": _event_phrase(phrase),
                })
                current["count"] += 1
                current["confidence"] = max(current["confidence"], confidence)
                midpoint = (target["start_time"] + target["end_time"]) / 2.0
                current["distance"] = min(current["distance"], abs(timestamp - midpoint))


def _apply_chat_evidence(targets: list[dict], chat_payload: dict) -> None:
    if not targets:
        return
    source_scope = _clean(chat_payload.get("source_scope")) or "unknown"
    for window in chat_payload.get("windows") or []:
        if not isinstance(window, dict):
            continue
        start = max(0.0, _finite(window.get("start")))
        end = max(start, _finite(window.get("end"), start))
        for target in targets:
            if _overlap(
                target["start_time"] - 3.0,
                target["end_time"] + 3.0,
                start,
                end,
            ) <= 0.0:
                continue
            target["_chat_messages"] += max(0, int(_finite(window.get("message_count"))))
            target["_chat_emotes"] += max(0, int(_finite(window.get("emote_count"))))
            target["_chat_clip_intents"] += max(
                0, int(_finite(window.get("clip_intent_count"))),
            )
            target["_chat_scope"] = source_scope
            for row in window.get("repeated_phrases") or []:
                if not isinstance(row, dict):
                    continue
                phrase = _clean(row.get("text"))
                count = max(0, int(_finite(row.get("count"))))
                if not phrase or count < 2:
                    continue
                key = phrase.casefold()
                current = target["_chat_phrases"].setdefault(
                    key, {"text": phrase, "count": 0},
                )
                current["count"] += count


def _finalize_pack(target: dict) -> dict:
    samples = max(1, int(target.pop("_samples")))
    ocr_rows = list(target.pop("_ocr").values())
    for row in ocr_rows:
        repeat_ratio = row["count"] / samples
        row["score"] = (
            row["confidence"]
            + (1.5 if row["event"] else 0.0)
            + min(0.45, len(row["text"]) / 100.0)
            + 0.25 / (1.0 + row["distance"])
            - (0.35 if repeat_ratio > 0.6 and not row["event"] else 0.0)
        )
    ocr_rows.sort(key=lambda row: (-row["score"], row["distance"], row["text"].casefold()))
    for row in ocr_rows[:MAX_OCR_PHRASES]:
        _append_unique(target["ocr"], row["text"])

    audio_spikes = int(target.pop("_audio_spikes"))
    max_spike = round(float(target.pop("_max_spike")), 4)
    motion_sum = float(target.pop("_motion_sum"))
    motion_max = round(float(target.pop("_motion_max")), 4)
    facecam = bool(target.pop("_facecam"))
    windowed_gameplay = bool(target.pop("_windowed_gameplay"))
    if audio_spikes:
        _append_unique(target["reactions"], "audio spike")

    chat_messages = int(target.pop("_chat_messages"))
    chat_emotes = int(target.pop("_chat_emotes"))
    chat_clip_intents = int(target.pop("_chat_clip_intents"))
    chat_scope = _clean(target.pop("_chat_scope")) or None
    chat_phrases = sorted(
        target.pop("_chat_phrases").values(),
        key=lambda row: (-row["count"], row["text"].casefold()),
    )[:MAX_CHAT_PHRASES]
    if chat_messages:
        _append_unique(target["reactions"], "chat burst")
        if chat_emotes >= 6:
            _append_unique(target["reactions"], "chat emote burst")
        if chat_clip_intents:
            _append_unique(target["reactions"], "viewer clip request")
        if not any(row.get("artifact") == "chat_evidence.v1.json" for row in target["provenance"]):
            target["provenance"].append({"artifact": "chat_evidence.v1.json"})

    searchable_parts = [
        *target["labels"], *target["events"], *target["reactions"], *target["ocr"],
        *(row["text"] for row in chat_phrases),
        *(
            value
            for scene in target["visual_scenes"]
            for value in (scene.get("description"), scene.get("support"))
            if value
        ),
    ]
    searchable_text = "; ".join(filter(None, searchable_parts))
    return {
        "version": EVIDENCE_PACK_VERSION,
        "anchor": target["anchor"],
        "search_text": searchable_text,
        "labels": target["labels"],
        "gameplay_events": target["events"],
        "reactions": target["reactions"],
        "ocr_phrases": target["ocr"],
        "visual_scenes": target["visual_scenes"][:MAX_VISUAL_SCENES],
        "layout": {
            "facecam_present": facecam,
            "windowed_gameplay": windowed_gameplay,
        },
        "signal_summary": {
            "samples": samples,
            "audio_spikes": audio_spikes,
            "max_audio_spike": max_spike,
            "mean_motion": round(motion_sum / samples, 4),
            "max_motion": motion_max,
        },
        "chat_activity": {
            "message_count": chat_messages,
            "emote_count": chat_emotes,
            "clip_intent_count": chat_clip_intents,
            "repeated_phrases": chat_phrases,
            "source_scope": chat_scope,
        } if chat_messages else None,
        "creator_marker": target.get("creator_marker"),
        "provenance": target["provenance"],
    }


def _new_target(
    *,
    anchor: str,
    start_time: float,
    end_time: float,
    candidate_data: dict,
    creator_marker: Optional[dict] = None,
) -> dict:
    return {
        "anchor": anchor,
        "start_time": start_time,
        "end_time": end_time,
        "labels": list(candidate_data.get("labels") or []),
        "events": list(candidate_data.get("events") or []),
        "reactions": list(candidate_data.get("reactions") or []),
        "ocr": list(candidate_data.get("ocr") or []),
        "visual_scenes": list(candidate_data.get("visual_scenes") or [])[:MAX_VISUAL_SCENES],
        "provenance": list(candidate_data.get("provenance") or []),
        "creator_marker": creator_marker,
        "_samples": 0,
        "_audio_spikes": 0,
        "_max_spike": 0.0,
        "_motion_sum": 0.0,
        "_motion_max": 0.0,
        "_facecam": False,
        "_windowed_gameplay": False,
        "_ocr": {},
        "_chat_messages": 0,
        "_chat_emotes": 0,
        "_chat_clip_intents": 0,
        "_chat_scope": None,
        "_chat_phrases": {},
    }


def build_evidence_packs(
    data_root: str,
    job_id: str,
    clip_entries: list[dict],
    *,
    region_plan: Optional[dict] = None,
    source_path: Optional[str] = None,
) -> EvidenceBuildResult:
    artifact_dir = os.path.join(data_root, "jobs", job_id)
    candidate_path = os.path.join(artifact_dir, "candidates.v1.json")
    signal_path = os.path.join(artifact_dir, "signals.v1.json")
    chat_path = os.path.join(artifact_dir, "chat_evidence.v1.json")
    candidate_payload = _load_json(candidate_path, {})
    candidates = (
        candidate_payload.get("candidates")
        if isinstance(candidate_payload, dict)
        else candidate_payload
    ) or []
    candidates = [row for row in candidates if isinstance(row, dict)]
    cached_visual_bytes = _hydrate_cached_visual_descriptions(
        data_root,
        source_path,
        candidates,
    )
    signal_exists = os.path.isfile(signal_path)
    candidate_exists = os.path.isfile(candidate_path)
    chat_exists = os.path.isfile(chat_path)
    chat_payload = _load_json(chat_path, {}) if chat_exists else {}
    source_bytes = sum(
        os.path.getsize(path)
        for path in (candidate_path, signal_path, chat_path)
        if os.path.isfile(path)
    ) + cached_visual_bytes
    if not candidate_exists and not signal_exists and not chat_exists and not region_plan:
        return EvidenceBuildResult(
            clip_packs={},
            standalone_entries=[],
            status="unavailable",
            source_bytes=0,
            coverage=_coverage_summary(candidates, chat_payload, []),
        )

    targets: list[dict] = []
    clip_targets: dict[str, dict] = {}
    matched_candidate_indexes: set[int] = set()
    for entry in clip_entries:
        match = _matching_candidate(candidates, entry)
        candidate_index, candidate = match if match else (None, None)
        if candidate_index is not None:
            matched_candidate_indexes.add(candidate_index)
        candidate_data = _candidate_evidence(candidate, candidate_index)
        marker_ids = list(candidate_data.get("marker_ids") or [])
        creator_marker = None
        if marker_ids or (candidate and candidate.get("creator_protected")):
            creator_marker = {
                "event_ids": marker_ids,
                "timestamp": candidate_data.get("marker_time"),
                "source": "recall_session",
            }
        target = _new_target(
            anchor="clip",
            start_time=_finite(entry.get("start_time")),
            end_time=max(
                _finite(entry.get("start_time")), _finite(entry.get("end_time")),
            ),
            candidate_data=candidate_data,
            creator_marker=creator_marker,
        )
        clip_targets[str(entry["clip_id"])] = target
        targets.append(target)

    standalone_targets: list[dict] = []
    for index, candidate in enumerate(candidates):
        if index in matched_candidate_indexes or not _is_qualified_standalone(candidate):
            continue
        start, end, _peak = _window(candidate)
        if any(_overlap(start, end, row["start_time"], row["end_time"]) > 0 for row in standalone_targets):
            continue
        candidate_data = _candidate_evidence(candidate, index)
        marker_ids = list(candidate_data.get("marker_ids") or [])
        creator_marker = None
        anchor = "detected_event"
        if marker_ids or candidate.get("creator_protected"):
            anchor = "creator_marker"
            creator_marker = {
                "event_ids": marker_ids,
                "timestamp": candidate_data.get("marker_time"),
                "source": "recall_session",
            }
        target = _new_target(
            anchor=anchor,
            start_time=start,
            end_time=end,
            candidate_data=candidate_data,
            creator_marker=creator_marker,
        )
        standalone_targets.append(target)
        if len(standalone_targets) >= MAX_STANDALONE_EVIDENCE:
            break

    # Chat-only bursts are searchable evidence, never review truth. Prefer the
    # strongest windows that do not already enrich a clip, detected event, or
    # creator marker, and keep the lane globally bounded per VOD.
    chat_windows = [
        row for row in (chat_payload.get("windows") or []) if isinstance(row, dict)
    ]
    chat_windows.sort(
        key=lambda row: (-_finite(row.get("strength")), _finite(row.get("start"))),
    )
    standalone_chat_count = 0
    for window in chat_windows:
        start = max(0.0, _finite(window.get("start")))
        end = max(start, _finite(window.get("end"), start))
        if any(
            _overlap(start, end, row["start_time"] - 3.0, row["end_time"] + 3.0) > 0.0
            for row in [*targets, *standalone_targets]
        ):
            continue
        if any(
            _overlap(
                start,
                end,
                max(0.0, _finite(marker.get("timestamp")) - 8.0),
                _finite(marker.get("timestamp")) + 8.0,
            ) > 0.0
            for marker in ((region_plan or {}).get("markers") or [])
            if isinstance(marker, dict)
        ):
            continue
        target = _new_target(
            anchor="chat_activity",
            start_time=start,
            end_time=end,
            candidate_data=_candidate_evidence(None, None),
        )
        standalone_targets.append(target)
        standalone_chat_count += 1
        if standalone_chat_count >= MAX_STANDALONE_CHAT_EVIDENCE:
            break

    existing_marker_ids = {
        event_id
        for target in [*targets, *standalone_targets]
        for event_id in ((target.get("creator_marker") or {}).get("event_ids") or [])
    }
    for marker_index, marker in enumerate((region_plan or {}).get("markers") or []):
        if not isinstance(marker, dict):
            continue
        event_id = _clean(marker.get("event_id"))
        if event_id and event_id in existing_marker_ids:
            continue
        timestamp = max(0.0, _finite(marker.get("timestamp")))
        candidate_data = _candidate_evidence(None, None)
        candidate_data["labels"] = ["creator marked live", "Remember button"]
        target = _new_target(
            anchor="creator_marker",
            start_time=max(0.0, timestamp - 8.0),
            end_time=timestamp + 8.0,
            candidate_data=candidate_data,
            creator_marker={
                "event_ids": [event_id] if event_id else [],
                "timestamp": timestamp,
                "source": _clean(marker.get("source")) or "recall_session",
                "ordinal": marker_index + 1,
                "alignment": marker.get("alignment"),
                "uncertainty_seconds": marker.get("uncertainty_seconds"),
            },
        )
        target["provenance"].append({
            "artifact": "jobs.region_plan",
            "marker_index": marker_index,
        })
        standalone_targets.append(target)

    all_targets = [*targets, *standalone_targets]
    if signal_exists and all_targets:
        _apply_signals(all_targets, _iter_json_array(signal_path))
    if chat_exists and all_targets:
        _apply_chat_evidence(all_targets, chat_payload)

    clip_packs = {
        clip_id: _finalize_pack(target) for clip_id, target in clip_targets.items()
    }
    standalone_entries = []
    for target in standalone_targets:
        pack = _finalize_pack(target)
        title = (
            "Remembered live"
            if pack["anchor"] == "creator_marker"
            else "Chat reaction"
            if pack["anchor"] == "chat_activity"
            else next(iter(pack["gameplay_events"] or pack["labels"]), "Detected moment").title()
        )
        standalone_entries.append({
            "start_time": target["start_time"],
            "end_time": target["end_time"],
            "title": title,
            "text": pack["search_text"],
            "evidence_kind": pack["anchor"],
            "evidence_json": pack,
        })
    all_packs = [*clip_packs.values(), *(row["evidence_json"] for row in standalone_entries)]
    return EvidenceBuildResult(
        clip_packs=clip_packs,
        standalone_entries=standalone_entries,
        status=(
            "indexed"
            if candidate_exists or signal_exists or chat_exists or region_plan
            else "unavailable"
        ),
        source_bytes=source_bytes,
        coverage=_coverage_summary(candidates, chat_payload, all_packs),
    )
