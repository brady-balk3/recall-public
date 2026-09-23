# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from core.bundle_paths import get_data_dir
from core.database import DatabaseManager
from core.stream_memory_semantic import (
    SEMANTIC_MODEL_ID as LSI_MODEL_ID,
    SemanticBuildCancelled,
    SemanticDocument,
    SemanticIndex,
    build_semantic_index as build_local_semantic_index,
    semantic_index_path,
)
from core.stream_memory_sentence import (
    SENTENCE_MODEL_ID,
    SentenceIndex,
    build_sentence_index,
    sentence_index_path,
    sentence_model_available,
    sentence_model_bytes,
)
from core.stream_memory_evidence import (
    EVIDENCE_PACK_VERSION,
    build_evidence_packs,
)


MEMORY_SCHEMA_VERSION = 13
MEMORY_LEARNING_VERSION = 1
MAX_QUERY_LENGTH = 240
MAX_RESULTS = 100
MAX_ALIAS_TERM_LENGTH = 80
MAX_EXPECTED_DESCRIPTION_LENGTH = 500
MAX_LEARNING_SEARCH_SESSIONS = 2000
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_QUOTED_RE = re.compile(r'"([^"]+)"')
_STOP_WORDS = {
    "a", "an", "and", "are", "at", "because", "every", "find", "for",
    "from", "i", "in", "is", "it", "me", "my", "of", "on", "show",
    "that", "the", "things", "this", "time", "to", "was", "when", "where", "with",
}
_SEMANTIC_BUILD_LOCK = threading.Lock()
_ACTION_TERM_FAMILY = (
    "snipe", "snipes", "sniped", "sniping", "sniper", "snipers",
    "headshot", "headshots", "noscope", "noscopes", "no scope",
    "long shot", "long-range shot", "precision shot", "one tap",
)
_CLIP_INTENT_TERMS = {
    "highlight", "highlights", "moment", "moments", "reaction", "reactions",
}
_GENERIC_MATCH_TERMS = _CLIP_INTENT_TERMS | {"aka", "made", "some", "somehow"}
_TWITCH_VIDEO_RE = re.compile(r"(?:twitch\.tv/videos/|videos/)(\d+)", re.IGNORECASE)
_QUERY_FILTER_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\b(?:clips?\s+)?(?:i\s+)?kept\b", "decision", "kept"),
    (r"\b(?:clips?\s+)?(?:i\s+)?passed\b", "decision", "passed"),
    (r"\bmaybe\s+clips?\b", "decision", "maybe"),
    (r"\b(?:unreviewed|not\s+reviewed)\b", "decision", "unreviewed"),
    (r"\b(?:unexported|(?:not|never)\s+exported)\b", "exported", "not_exported"),
    (r"\b(?:already\s+)?exported\b", "exported", "exported"),
    (r"\b(?:marked|remembered)(?:\s+(?:live|moments?|things?))?\b", "origin", "creator_marker"),
    (r"\bremember\s+button\b", "origin", "creator_marker"),
)
_QUERY_KIND_PATTERN = re.compile(r"\b(?:saved\s+)?clips?\b", re.IGNORECASE)
_QUERY_BEST_PATTERN = re.compile(r"\b(?:best|top|strongest)\b", re.IGNORECASE)
_QUERY_LAST_MONTH_PATTERN = re.compile(r"\blast\s+month\b", re.IGNORECASE)
_QUERY_THIS_MONTH_PATTERN = re.compile(r"\bthis\s+month\b", re.IGNORECASE)
_EXACT_TERM_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("snipe", "snipes", "sniped", "sniping", "sniper", "snipers"),
    ("headshot", "headshots"),
    ("noscope", "noscopes"),
    ("max", "maxed", "maxing", "maxxing", "maxxed"),
)
_CREATOR_CONCEPTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("aura maxxing", "aura maxing", "aura farming", "aura farm"),
     ("aura",)),
    (("funny", "hilarious", "joke", "laugh", "comedy"),
     ("funny", "laughter", "joke", "comedy", "laughter_burst")),
    (("clutch", "escape", "survive", "barely", "close call", "comeback"),
     ("clutch", "escape", "survived", "survival", "comeback", "close_call")),
    (("scare", "scary", "terrified", "horror", "monster", "jumpscare"),
     ("scare", "scared", "horror", "scream", "jumpscare", "monster")),
    (("rage", "angry", "mad", "frustrated", "frustration"),
     ("rage", "angry", "frustration", "shouting", "voice_spike")),
    (("story", "storytime", "anecdote", "remember when"),
     ("story", "storytelling", "anecdote", "conversation")),
    (("reaction", "react", "facecam", "expression"),
     ("reaction", "facecam_reaction", "voice_reaction", "expression")),
    (("friend", "friends", "collab", "together", "multiplayer"),
     ("friend", "friends", "collab", "multiplayer", "party")),
    (("wholesome", "heartwarming", "sweet", "kind"),
     ("wholesome", "heartwarming", "sweet", "kind")),
    (("fail", "failed", "mistake", "disaster", "died", "death"),
     ("fail", "failure", "mistake", "disaster", "death")),
    (("excited", "hype", "celebrate", "celebration", "victory"),
     ("excited", "hype", "celebration", "victory", "cheering")),
    (("fight", "battle", "combat", "boss"),
     ("fight", "battle", "combat", "boss", "gameplay")),
    (("crazy", "insane", "wild", "unbelievable", "unreal", "amazing", "epic"),
     ("crazy", "insane", "wild", "unbelievable", "unreal", "amazing", "epic",
      "hype", "reaction", "voice_reaction", "laughter_burst", "clutch")),
    (_ACTION_TERM_FAMILY,
     ("snipe", "sniper", "sniped", "sniping", "headshot", "noscope",
      "no scope", "long shot", "precision shot", "one tap")),
)

_COMPOUND_GLUE_WORDS = {
    "after", "before", "during", "followed", "happened", "then", "while",
}
_COMPOUND_CLAUSE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "creator_reaction",
        "label": "Strong creator reaction",
        "patterns": (
            r"\b(?:scream(?:ed|ing|s)?|yell(?:ed|ing|s)?|shout(?:ed|ing|s)?)\b",
            r"\b(?:strong|big|facecam|creator|streamer)\s+reaction\b",
            r"\b(?:startl(?:e|ed|ing)|jumpscare(?:d)?)\b",
        ),
        "fts_terms": (
            "strong startle", "streamer reaction", "voice reaction",
            "audio spike", "reaction burst",
        ),
    },
    {
        "id": "death",
        "label": "Death or defeat",
        "patterns": (
            r"\b(?:die|dies|died|dying|death|dead)\b",
            r"\b(?:defeat|defeated|game\s+over|round\s+lost)\b",
        ),
        "fts_terms": (
            "death", "dead", "defeat", "defeated", "you died", "game over",
            "round lost",
        ),
    },
    {
        "id": "elimination",
        "label": "Elimination",
        "patterns": (
            r"\b(?:eliminate|eliminated|elimination|eliminations)\b",
            r"\b(?:kill|kills|killed|killing|headshot|headshots|knock|knocked)\b",
        ),
        "fts_terms": (
            "elimination", "eliminated", "headshot", "knock", "knocked",
        ),
    },
    {
        "id": "victory",
        "label": "Win or victory",
        "patterns": (
            r"\b(?:victory(?:\s+royale)?|win|wins|won|winning|round\s+won)\b",
        ),
        "fts_terms": (
            "victory", "victory royale", "win", "winner", "won", "round won",
            "terminal win",
        ),
    },
    {
        "id": "boss",
        "label": "Boss encounter",
        "patterns": (
            r"\bboss(?:\s+(?:fight|battle|encounter))?\b",
        ),
        "fts_terms": ("boss", "boss fight", "boss battle", "boss encounter"),
    },
    {
        "id": "chat_excitement",
        "label": "Chat excitement",
        "patterns": (
            r"\bchat\s+(?:went|goes|was|got)\s+(?:crazy|wild)\b",
            r"\b(?:everyone|everybody)\s+went\s+(?:crazy|wild)\b",
            r"\b(?:chat|emote|viewer)\s+spam(?:med|ming)?\b",
            r"\bchat\s+burst\b",
            r"\bchat\b",
        ),
        "fts_terms": (
            "chat burst", "chat emote burst", "viewer clip request",
        ),
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_value(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return fallback
    return parsed


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split()).strip()


def _normalized_query(value: Any) -> str:
    return " ".join(_TOKEN_RE.findall(_clean_text(value).casefold()))[:MAX_QUERY_LENGTH]


def _entry_fingerprint(entry: Any) -> str:
    """Identify indexed evidence by content, never its rebuild-time row position."""
    kind = str(entry["kind"] or "")
    clip_id = str(entry["clip_id"] or "")
    start_ms = int(round(float(entry["start_time"] or 0.0) * 1000.0))
    end_ms = int(round(float(entry["end_time"] or 0.0) * 1000.0))
    title = _clean_text(entry["title"]).casefold()
    text = _clean_text(entry["text"]).casefold()
    payload = f"{kind}\n{clip_id}\n{start_ms}\n{end_ms}\n{title}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _revision_digest(rows: Iterable[Any]) -> str:
    payload = [dict(row) for row in rows]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _quoted_prefix(term: str) -> str:
    return f'"{term.replace(chr(34), chr(34) * 2)}"*'


def _term_family(token: str) -> Optional[tuple[str, ...]]:
    token = token.lower()
    return next((family for family in _EXACT_TERM_FAMILIES if token in family), None)


def _fts_token_expression(token: str) -> str:
    family = _term_family(token)
    if not family:
        return _quoted_prefix(token)
    # Phrase aliases live in the creator-concept lane. The exact token lane
    # expands bounded one-word inflections without turning FTS into a thesaurus.
    variants = sorted({value for value in family if " " not in value and "-" not in value})
    return f"({' OR '.join(_quoted_prefix(value) for value in variants)})"


def _fts_query(query: str, *, required_parts: Optional[list[str]] = None) -> str:
    raw = _clean_text(query)[:MAX_QUERY_LENGTH]
    phrases = [_clean_text(value) for value in _QUOTED_RE.findall(raw)]
    remainder = _QUOTED_RE.sub(" ", raw)
    tokens = [token.lower() for token in _TOKEN_RE.findall(remainder)]
    meaningful = [token for token in tokens if token not in _STOP_WORDS]
    parts = list(required_parts or [])
    parts.extend(f'"{phrase.replace(chr(34), chr(34) * 2)}"' for phrase in phrases if phrase)
    parts.extend(_fts_token_expression(token) for token in meaningful)
    if not parts:
        raise ValueError("Search for at least one word.")
    return " AND ".join(parts)


def _creator_concept_query(query: str) -> Optional[str]:
    raw = _clean_text(query).lower()
    tokens = set(_TOKEN_RE.findall(raw))
    groups: list[set[str]] = []
    for triggers, concepts in _CREATOR_CONCEPTS:
        matched = any(
            (" " in trigger and trigger in raw)
            or (" " not in trigger and any(
                token == trigger or (len(trigger) >= 4 and token.startswith(trigger))
                for token in tokens
            ))
            for trigger in triggers
        )
        if matched:
            group = set(concepts)
            if group not in groups:
                groups.append(group)
    if not groups:
        return None
    # Multiple creator intents must be represented by the same memory. This
    # prevents a query such as "crazy snipes" from degrading into rows that
    # only contain "crazy".
    return " AND ".join(
        f"({' OR '.join(_quoted_prefix(concept) for concept in sorted(group))})"
        for group in groups
    )


def _is_aura_maxxing_query(query: str) -> bool:
    raw = _clean_text(query).lower()
    return any(
        phrase in raw
        for phrase in ("aura maxxing", "aura maxing", "aura farming", "aura farm")
    )


def _prefers_clip_evidence(query: str) -> bool:
    raw = _clean_text(query).lower()
    tokens = set(_TOKEN_RE.findall(raw))
    return bool(tokens & _CLIP_INTENT_TERMS) or any(
        (" " in trigger and trigger in raw)
        or (" " not in trigger and any(
            token == trigger or (len(trigger) >= 4 and token.startswith(trigger))
            for token in tokens
        ))
        for trigger in _ACTION_TERM_FAMILY
    )


def _query_match_terms(query: str) -> set[str]:
    """Return bounded words that can explain which persisted field matched."""
    raw = _clean_text(query).lower()
    tokens = set(_TOKEN_RE.findall(raw))
    terms = {
        token for token in tokens
        if token not in _STOP_WORDS and token not in _GENERIC_MATCH_TERMS
    }
    for triggers, concepts in _CREATOR_CONCEPTS:
        matched = any(
            (" " in trigger and trigger in raw)
            or (" " not in trigger and any(
                token == trigger or (len(trigger) >= 4 and token.startswith(trigger))
                for token in tokens
            ))
            for trigger in triggers
        )
        if matched:
            terms.update(concepts)
    return {
        _clean_text(term).replace("_", " ").casefold()
        for term in terms
        if _clean_text(term)
    }


def _text_matches_terms(text: Any, terms: set[str]) -> bool:
    normalized = _clean_text(text).replace("_", " ").casefold()
    if not normalized or not terms:
        return False
    words = set(_TOKEN_RE.findall(normalized))
    for term in terms:
        if " " in term and term in normalized:
            return True
        family = _term_family(term)
        if family and any(value in words for value in family if " " not in value):
            return True
        if term in words:
            return True
    return False


def _source_video_identity(source_path: Any) -> Optional[str]:
    source = _clean_text(source_path)
    match = _TWITCH_VIDEO_RE.search(source)
    return f"twitch:{match.group(1)}" if match else None


def _month_bounds(reference: datetime, month_delta: int) -> tuple[str, str]:
    year = reference.year
    month = reference.month + month_delta
    while month < 1:
        year -= 1
        month += 12
    while month > 12:
        year += 1
        month -= 12
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        following = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        following = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    end = following - timedelta(days=1)
    return start.date().isoformat(), end.date().isoformat()


def _plan_search_query(
    query: str,
    *,
    kind: str,
    decision: str,
    exported: str,
    origin: str,
    date_from: Optional[str],
    date_to: Optional[str],
) -> Dict[str, Any]:
    remaining = _clean_text(query)
    interpreted: list[dict[str, str]] = []
    resolved = {
        "kind": kind,
        "decision": decision,
        "exported": exported,
        "origin": origin,
        "date_from": date_from,
        "date_to": date_to,
    }
    for pattern, field, value in _QUERY_FILTER_PATTERNS:
        if not re.search(pattern, remaining, flags=re.IGNORECASE):
            continue
        if resolved[field] == "all":
            resolved[field] = value
            interpreted.append({"field": field, "value": value})
        remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE)
    if _QUERY_KIND_PATTERN.search(remaining):
        if resolved["kind"] == "all":
            resolved["kind"] = "clip"
            interpreted.append({"field": "kind", "value": "clip"})
        remaining = _QUERY_KIND_PATTERN.sub(" ", remaining)

    best_first = bool(_QUERY_BEST_PATTERN.search(remaining))
    if best_first:
        remaining = _QUERY_BEST_PATTERN.sub(" ", remaining)
        interpreted.append({"field": "sort", "value": "best"})

    now = datetime.now().astimezone()
    relative_dates: Optional[tuple[str, str]] = None
    if _QUERY_LAST_MONTH_PATTERN.search(remaining):
        relative_dates = _month_bounds(now, -1)
        remaining = _QUERY_LAST_MONTH_PATTERN.sub(" ", remaining)
    elif _QUERY_THIS_MONTH_PATTERN.search(remaining):
        relative_dates = _month_bounds(now, 0)
        remaining = _QUERY_THIS_MONTH_PATTERN.sub(" ", remaining)
    if relative_dates:
        if not resolved["date_from"]:
            resolved["date_from"] = relative_dates[0]
        if not resolved["date_to"]:
            resolved["date_to"] = relative_dates[1]
        interpreted.append({
            "field": "date",
            "value": f"{resolved['date_from']}..{resolved['date_to']}",
        })
    return {
        "query": _clean_text(remaining),
        "interpreted_filters": interpreted,
        "best_first": best_first,
        **resolved,
    }


def _plan_compound_query(query: str, *, include_marker: bool = False) -> Dict[str, Any]:
    """Recognize only structured clauses Evidence Packs can prove together."""
    remaining = _clean_text(query)
    clauses: list[dict[str, Any]] = []
    for spec in _COMPOUND_CLAUSE_SPECS:
        matched = False
        for pattern in spec["patterns"]:
            if re.search(pattern, remaining, flags=re.IGNORECASE):
                matched = True
                remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE)
        if matched:
            clauses.append({
                "id": spec["id"],
                "label": spec["label"],
                "fts_terms": list(spec["fts_terms"]),
            })
    if include_marker:
        clauses.append({
            "id": "creator_marker",
            "label": "Marked live",
            "fts_terms": ["creator marked live", "Remember button"],
        })
    # One structured idea remains an ordinary hybrid search. Compound mode is
    # deliberately reserved for queries where partial-intent matches are risky.
    if len(clauses) < 2:
        return {"active": False, "clauses": [], "residual_query": ""}
    residual_tokens = [
        token for token in _TOKEN_RE.findall(remaining.casefold())
        if token not in _STOP_WORDS
        and token not in _GENERIC_MATCH_TERMS
        and token not in _COMPOUND_GLUE_WORDS
    ]
    return {
        "active": True,
        "clauses": clauses,
        "residual_query": " ".join(residual_tokens),
    }


def _compound_fts_expression(
    plan: Dict[str, Any],
    residual_expression: str,
) -> str:
    groups = [
        f"({' OR '.join(_quoted_prefix(term) for term in clause['fts_terms'])})"
        for clause in plan.get("clauses") or []
    ]
    if residual_expression:
        groups.append(residual_expression)
    return " AND ".join(groups)


def _matching_evidence_value(values: Iterable[Any], terms: Iterable[str]) -> Optional[str]:
    for value in values:
        original = _clean_text(value)
        normalized = original.replace("_", " ").replace("-", " ").casefold()
        for term in terms:
            pattern = rf"(?<!\w){re.escape(term.casefold())}(?!\w)"
            if re.search(pattern, normalized):
                return original
    return None


def _compound_clause_match(row: Dict[str, Any], clause_id: str) -> Optional[dict[str, str]]:
    evidence = _json_value(row.get("evidence_json"), {})
    reactions = list(evidence.get("reactions") or [])
    events = list(evidence.get("gameplay_events") or [])
    ocr = list(evidence.get("ocr_phrases") or [])
    visual_values = [
        value
        for scene in (evidence.get("visual_scenes") or [])
        if isinstance(scene, dict)
        for value in (
            scene.get("outcome"), scene.get("description"), scene.get("support"),
        )
        if value
    ]
    if clause_id == "creator_reaction":
        normalized = {
            _clean_text(value).replace("_", " ").casefold() for value in reactions
        }
        if "strong startle" in normalized:
            detail = "Strong startle"
        elif "audio spike" in normalized and normalized.intersection({
            "voice reaction", "streamer reaction", "facecam reaction", "reaction burst",
        }):
            partner = next(
                label for label in (
                    "voice reaction", "streamer reaction", "facecam reaction", "reaction burst",
                )
                if label in normalized
            )
            detail = f"{partner.title()} + audio spike"
        else:
            return None
        return {"clause_id": clause_id, "label": "Strong creator reaction", "detail": detail,
                "source": "reaction_signal"}
    if clause_id == "chat_excitement":
        chat = evidence.get("chat_activity") or {}
        messages = max(0, int(chat.get("message_count") or 0))
        emotes = max(0, int(chat.get("emote_count") or 0))
        if messages <= 0:
            return None
        detail = f"{messages} chat messages"
        if emotes:
            detail += f" + {emotes} emotes"
        return {"clause_id": clause_id, "label": "Chat excitement", "detail": detail,
                "source": "chat_activity"}
    if clause_id == "creator_marker":
        if not evidence.get("creator_marker") and row.get("evidence_kind") != "creator_marker":
            return None
        return {"clause_id": clause_id, "label": "Marked live", "detail": "Remember marker",
                "source": "creator_marker"}

    event_terms = {
        "death": ("death", "dead", "defeat", "defeated", "you died", "game over", "round lost"),
        "elimination": ("elimination", "eliminated", "headshot", "knock", "knocked"),
        "victory": ("victory", "victory royale", "win", "winner", "won", "round won", "terminal win"),
        "boss": ("boss", "boss fight", "boss battle", "boss encounter"),
    }
    labels = {
        "death": "Death or defeat",
        "elimination": "Elimination",
        "victory": "Win or victory",
        "boss": "Boss encounter",
    }
    for values, source in (
        (events, "gameplay_event"),
        (visual_values, "visual_description"),
        (ocr, "on_screen_text"),
    ):
        matched = _matching_evidence_value(values, event_terms[clause_id])
        if matched:
            return {"clause_id": clause_id, "label": labels[clause_id], "detail": matched,
                    "source": source}
    return None


def _compound_matches(row: Dict[str, Any], clauses: list[dict[str, Any]]) -> list[dict[str, str]]:
    matches = [
        _compound_clause_match(row, str(clause["id"]))
        for clause in clauses
    ]
    return [match for match in matches if match]


def _game_at(segments: Iterable[dict], timestamp: float) -> str:
    for segment in segments:
        try:
            start = float(segment.get("start", 0.0) or 0.0)
            end = float(segment.get("end", start) or start)
        except (TypeError, ValueError):
            continue
        if start <= timestamp <= end:
            return _clean_text(segment.get("game") or segment.get("label"))
    return ""


class StreamMemoryService:
    """Durable local search over rebuildable text evidence, never scan truth."""

    def __init__(
        self,
        db: DatabaseManager,
        data_root: Optional[str] = None,
        *,
        prefer_sentence_model: Optional[bool] = None,
    ):
        self.db = db
        self.data_root = data_root or get_data_dir()
        model_is_installed = sentence_model_available()
        self._sentence_model_available = (
            model_is_installed
            if prefer_sentence_model is None
            else bool(prefer_sentence_model and model_is_installed)
        )
        if self._sentence_model_available:
            self._semantic_model_id = SENTENCE_MODEL_ID
            self._semantic_path = sentence_index_path(self.data_root)
            self._semantic_index = SentenceIndex(self._semantic_path)
            self._semantic_builder = build_sentence_index
        else:
            self._semantic_model_id = LSI_MODEL_ID
            self._semantic_path = semantic_index_path(self.data_root)
            self._semantic_index = SemanticIndex(self._semantic_path)
            self._semantic_builder = build_local_semantic_index
        self._recover_interrupted_semantic_build()

    def _recover_interrupted_semantic_build(self) -> None:
        """Make a killed concept-index rebuild retryable on the next launch.

        Recall owns one backend process. Within that process the module-level
        lock distinguishes a live build from persisted state left behind by a
        terminated process. The previous sidecar remains safe and usable.
        """
        if _SEMANTIC_BUILD_LOCK.locked():
            return
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT status, indexed_entries FROM stream_memory_semantic_state WHERE id = 1"
            ).fetchone()
            if not row or row["status"] != "building":
                return
            has_previous_index = bool(
                int(row["indexed_entries"] or 0) > 0
                and os.path.isfile(self._semantic_path)
            )
            conn.execute(
                """UPDATE stream_memory_semantic_state
                   SET status = ?, last_error = ? WHERE id = 1""",
                (
                    "stale" if has_previous_index else "error",
                    "The previous concept-index update was interrupted. Try Update again.",
                ),
            )

    @staticmethod
    def _mark_semantic_stale(conn) -> None:
        conn.execute(
            """UPDATE stream_memory_semantic_state
               SET source_revision = source_revision + 1,
                   status = CASE
                     WHEN status = 'not_built' THEN 'not_built'
                     ELSE 'stale'
                   END,
                   pending_since = COALESCE(pending_since, CURRENT_TIMESTAMP),
                   retry_count = 0,
                   next_attempt_at = NULL
               WHERE id = 1"""
        )

    def _job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                """SELECT id, status, session_name, source_type, source_path,
                          asset_path, source_date, duration, created_at,
                          recall_session_id, region_plan
                   FROM jobs WHERE id = ?""",
                (job_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["region_plan"] = _json_value(result.get("region_plan"), {})
        return result

    def _timeline_segments(self, job_id: str) -> list[dict]:
        timeline = self.db.get_reaction_timeline(job_id) or {}
        segments = timeline.get("segments") or []
        return [segment for segment in segments if isinstance(segment, dict)]

    def list_aliases(self) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT id, canonical_term, alias, created_at, updated_at
                   FROM stream_memory_aliases
                   ORDER BY LOWER(canonical_term), LOWER(alias)"""
            ).fetchall()
        return {"aliases": [dict(row) for row in rows]}

    def add_alias(self, canonical_term: str, alias: str) -> Dict[str, Any]:
        canonical = _clean_text(canonical_term)
        alternate = _clean_text(alias)
        if not canonical or not alternate:
            raise ValueError("Both the remembered term and its alias are required.")
        if len(canonical) > MAX_ALIAS_TERM_LENGTH or len(alternate) > MAX_ALIAS_TERM_LENGTH:
            raise ValueError(f"Memory terms must be {MAX_ALIAS_TERM_LENGTH} characters or fewer.")
        if canonical.casefold() == alternate.casefold():
            raise ValueError("Use two different names for a Memory connection.")
        if not _TOKEN_RE.search(canonical) or not _TOKEN_RE.search(alternate):
            raise ValueError("Memory terms must contain a word or number.")
        now = _utc_now()
        with self.db.get_connection() as conn:
            existing = conn.execute(
                """SELECT id, canonical_term, alias, created_at, updated_at
                   FROM stream_memory_aliases WHERE alias = ? COLLATE NOCASE""",
                (alternate,),
            ).fetchone()
            if existing:
                if str(existing["canonical_term"]).casefold() != canonical.casefold():
                    raise ValueError(
                        f'“{alternate}” is already connected to “{existing["canonical_term"]}”.'
                    )
                return dict(existing)
            alias_id = f"alias_{uuid.uuid4().hex}"
            conn.execute(
                """INSERT INTO stream_memory_aliases
                   (id, canonical_term, alias, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (alias_id, canonical, alternate, now, now),
            )
            row = conn.execute(
                """SELECT id, canonical_term, alias, created_at, updated_at
                   FROM stream_memory_aliases WHERE id = ?""",
                (alias_id,),
            ).fetchone()
        return dict(row)

    def delete_alias(self, alias_id: str) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM stream_memory_aliases WHERE id = ?",
                (_clean_text(alias_id),),
            )
        if not cursor.rowcount:
            raise KeyError("Memory connection not found.")
        return {"deleted": True, "id": alias_id}

    def set_feedback(self, query: str, entry_id: str, verdict: str) -> Dict[str, Any]:
        query_text = _clean_text(query)[:MAX_QUERY_LENGTH]
        normalized = _normalized_query(query)
        if not normalized:
            raise ValueError("Search for at least one word before teaching Recall.")
        if verdict not in {"relevant", "not_relevant", "clear"}:
            raise ValueError("Feedback must be relevant, not_relevant, or clear.")
        with self.db.get_connection() as conn:
            entry = conn.execute(
                """SELECT id, job_id, clip_id, kind, entry_index, start_time,
                          end_time, title, text
                   FROM stream_memory_entries WHERE id = ?""",
                (_clean_text(entry_id),),
            ).fetchone()
            if not entry:
                raise KeyError("Memory result not found.")
            fingerprint = _entry_fingerprint(entry)
            identity_params = (
                entry["clip_id"], entry["clip_id"], entry["job_id"],
                entry["kind"], fingerprint,
            )
            if verdict == "clear":
                conn.execute(
                    """DELETE FROM stream_memory_feedback
                       WHERE normalized_query = ? AND (
                         (? IS NOT NULL AND clip_id = ?)
                         OR (job_id = ? AND kind = ? AND entry_fingerprint = ?)
                       )""",
                    (normalized, *identity_params),
                )
                case = conn.execute(
                    "SELECT id, status FROM stream_memory_evaluation_cases WHERE normalized_query = ?",
                    (normalized,),
                ).fetchone()
                if case:
                    conn.execute(
                        """DELETE FROM stream_memory_evaluation_targets
                           WHERE case_id = ? AND (
                             (? IS NOT NULL AND clip_id = ?)
                             OR (job_id = ? AND kind = ? AND entry_fingerprint = ?)
                           )""",
                        (case["id"], *identity_params),
                    )
                    remaining = conn.execute(
                        "SELECT 1 FROM stream_memory_evaluation_targets WHERE case_id = ? LIMIT 1",
                        (case["id"],),
                    ).fetchone()
                    if not remaining and case["status"] == "labeled":
                        conn.execute(
                            "DELETE FROM stream_memory_evaluation_cases WHERE id = ?",
                            (case["id"],),
                        )
                return {"query": normalized, "entry_id": entry_id, "verdict": None}
            now = _utc_now()
            conn.execute(
                """DELETE FROM stream_memory_feedback
                   WHERE normalized_query = ? AND (
                     (job_id = ? AND kind = ? AND entry_index = ?)
                     OR (? IS NOT NULL AND clip_id = ?)
                     OR (job_id = ? AND kind = ? AND entry_fingerprint = ?)
                   )""",
                (
                    normalized, entry["job_id"], entry["kind"], entry["entry_index"],
                    entry["clip_id"], entry["clip_id"], entry["job_id"],
                    entry["kind"], fingerprint,
                ),
            )
            conn.execute(
                """INSERT INTO stream_memory_feedback
                   (id, normalized_query, job_id, kind, entry_index, clip_id,
                    entry_fingerprint, verdict, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"feedback_{uuid.uuid4().hex}", normalized, entry["job_id"],
                    entry["kind"], entry["entry_index"], entry["clip_id"],
                    fingerprint, verdict, now, now,
                ),
            )
            case = conn.execute(
                "SELECT id, status FROM stream_memory_evaluation_cases WHERE normalized_query = ?",
                (normalized,),
            ).fetchone()
            if case:
                case_id = str(case["id"])
                next_status = "labeled" if verdict == "relevant" else str(case["status"])
                conn.execute(
                    """UPDATE stream_memory_evaluation_cases
                       SET query_text = ?, status = ?, updated_at = ? WHERE id = ?""",
                    (query_text, next_status, now, case_id),
                )
            else:
                case_id = f"memory_eval_{uuid.uuid4().hex}"
                conn.execute(
                    """INSERT INTO stream_memory_evaluation_cases
                       (id, query_text, normalized_query, status, created_at, updated_at)
                       VALUES (?, ?, ?, 'labeled', ?, ?)""",
                    (case_id, query_text, normalized, now, now),
                )
            conn.execute(
                """DELETE FROM stream_memory_evaluation_targets
                   WHERE case_id = ? AND (
                     (job_id = ? AND kind = ? AND entry_index = ?)
                     OR (? IS NOT NULL AND clip_id = ?)
                     OR (job_id = ? AND kind = ? AND entry_fingerprint = ?)
                   )""",
                (
                    case_id, entry["job_id"], entry["kind"], entry["entry_index"],
                    entry["clip_id"], entry["clip_id"], entry["job_id"],
                    entry["kind"], fingerprint,
                ),
            )
            conn.execute(
                """INSERT INTO stream_memory_evaluation_targets
                   (id, case_id, job_id, kind, entry_index, clip_id,
                    entry_fingerprint, verdict, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"memory_target_{uuid.uuid4().hex}", case_id, entry["job_id"],
                    entry["kind"], entry["entry_index"], entry["clip_id"],
                    fingerprint, verdict, now, now,
                ),
            )
        return {"query": normalized, "entry_id": entry_id, "verdict": verdict}

    def save_missed_query(self, query: str, expected_description: str) -> Dict[str, Any]:
        query_text = _clean_text(query)[:MAX_QUERY_LENGTH]
        normalized = _normalized_query(query_text)
        expected = _clean_text(expected_description)
        if not normalized:
            raise ValueError("Search for at least one word before saving a failed search.")
        if not expected:
            raise ValueError("Describe what Recall should have found.")
        if len(expected) > MAX_EXPECTED_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Expected memory descriptions must be {MAX_EXPECTED_DESCRIPTION_LENGTH} characters or fewer."
            )
        now = _utc_now()
        with self.db.get_connection() as conn:
            case = conn.execute(
                "SELECT id FROM stream_memory_evaluation_cases WHERE normalized_query = ?",
                (normalized,),
            ).fetchone()
            if case:
                relevant = conn.execute(
                    """SELECT 1 FROM stream_memory_evaluation_targets
                       WHERE case_id = ? AND verdict = 'relevant' LIMIT 1""",
                    (case["id"],),
                ).fetchone()
                if relevant:
                    raise ValueError("That search already has a relevant memory in the test set.")
                conn.execute(
                    """UPDATE stream_memory_evaluation_cases
                       SET query_text = ?, expected_description = ?, status = 'missed', updated_at = ?
                       WHERE id = ?""",
                    (query_text, expected, now, case["id"]),
                )
                case_id = str(case["id"])
            else:
                case_id = f"memory_eval_{uuid.uuid4().hex}"
                conn.execute(
                    """INSERT INTO stream_memory_evaluation_cases
                       (id, query_text, normalized_query, expected_description, status,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, 'missed', ?, ?)""",
                    (case_id, query_text, normalized, expected, now, now),
                )
        return self._evaluation_case(case_id)

    def delete_evaluation_case(self, case_id: str) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM stream_memory_evaluation_cases WHERE id = ?",
                (_clean_text(case_id),),
            )
        if not cursor.rowcount:
            raise KeyError("Memory search test case not found.")
        return {"deleted": True, "id": case_id}

    def resolve_evaluation_case(self, case_id: str, entry_id: str) -> Dict[str, Any]:
        case = self._evaluation_case(_clean_text(case_id))
        self.set_feedback(case["query"], entry_id, "relevant")
        return self._evaluation_case(case["id"])

    @staticmethod
    def _resolved_target_entry_id(conn, target: Any) -> Optional[str]:
        clip_id = target["clip_id"]
        if clip_id:
            rows = conn.execute(
                """SELECT id FROM stream_memory_entries
                   WHERE job_id = ? AND kind = 'clip' AND clip_id = ?""",
                (target["job_id"], clip_id),
            ).fetchall()
            return str(rows[0]["id"]) if len(rows) == 1 else None

        fingerprint = str(target["entry_fingerprint"] or "")
        if not fingerprint:
            return None
        rows = conn.execute(
            """SELECT id, kind, clip_id, start_time, end_time, title, text
               FROM stream_memory_entries WHERE job_id = ? AND kind = ?""",
            (target["job_id"], target["kind"]),
        ).fetchall()
        matches = [row for row in rows if _entry_fingerprint(row) == fingerprint]
        return str(matches[0]["id"]) if len(matches) == 1 else None

    def _evaluation_revisions(self) -> Dict[str, str]:
        with self.db.get_connection() as conn:
            truth_rows = conn.execute(
                """SELECT c.id AS case_id, c.normalized_query, c.expected_description,
                          c.status, t.job_id, t.kind, t.clip_id,
                          t.entry_fingerprint, t.verdict
                   FROM stream_memory_evaluation_cases AS c
                   LEFT JOIN stream_memory_evaluation_targets AS t ON t.case_id = c.id
                   ORDER BY c.id, t.id"""
            ).fetchall()
            alias_rows = conn.execute(
                """SELECT LOWER(canonical_term) AS canonical_term, LOWER(alias) AS alias
                   FROM stream_memory_aliases
                   ORDER BY LOWER(canonical_term), LOWER(alias)"""
            ).fetchall()
            semantic_row = conn.execute(
                """SELECT source_revision, indexed_revision, status, model_id,
                          dimensions, indexed_entries
                   FROM stream_memory_semantic_state WHERE id = 1"""
            ).fetchone()
        system_state = {
            "memory_schema_version": MEMORY_SCHEMA_VERSION,
            **(dict(semantic_row) if semantic_row else {}),
        }
        return {
            "truth_revision": _revision_digest(truth_rows),
            "alias_revision": _revision_digest(alias_rows),
            "system_revision": _revision_digest([system_state]),
        }

    def _evaluation_case(self, case_id: str) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                """SELECT id, query_text AS query, expected_description,
                          status, created_at, updated_at
                   FROM stream_memory_evaluation_cases WHERE id = ?""",
                (case_id,),
            ).fetchone()
            targets = conn.execute(
                """SELECT job_id, kind, clip_id, entry_fingerprint, verdict
                   FROM stream_memory_evaluation_targets WHERE case_id = ?""",
                (case_id,),
            ).fetchall()
        if not row:
            raise KeyError("Memory search test case not found.")
        item = dict(row)
        item["relevant_count"] = sum(target["verdict"] == "relevant" for target in targets)
        item["negative_count"] = sum(target["verdict"] == "not_relevant" for target in targets)
        with self.db.get_connection() as conn:
            resolved = [
                target for target in targets
                if self._resolved_target_entry_id(conn, target) is not None
            ]
        item["resolved_relevant_count"] = sum(
            target["verdict"] == "relevant" for target in resolved
        )
        item["stale_target_count"] = len(targets) - len(resolved)
        item["ready_to_score"] = item["resolved_relevant_count"] > 0
        return item

    def evaluation_summary(self) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            case_ids = [
                str(row["id"])
                for row in conn.execute(
                    """SELECT id FROM stream_memory_evaluation_cases
                       ORDER BY updated_at DESC, created_at DESC"""
                ).fetchall()
            ]
            latest_row = conn.execute(
                """SELECT id, system_id, case_count, metrics, created_at
                   FROM stream_memory_evaluation_runs
                   ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
        cases = [self._evaluation_case(case_id) for case_id in case_ids]
        scorable = sum(1 for case in cases if case["ready_to_score"])
        latest_run = None
        if latest_row:
            latest_run = dict(latest_row)
            latest_run["metrics"] = _json_value(latest_run.get("metrics"), {})
            revisions = self._evaluation_revisions()
            latest_run["stale"] = any(
                latest_run["metrics"].get(key) != value
                for key, value in revisions.items()
            )
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "target_cases": 20,
            "case_count": len(cases),
            "scorable_cases": scorable,
            "unresolved_cases": len(cases) - scorable,
            "cases": cases,
            "latest_run": latest_run,
        }

    def run_evaluation(self) -> Dict[str, Any]:
        summary = self.evaluation_summary()
        ready_cases = [case for case in summary["cases"] if case["ready_to_score"]]
        if not summary["cases"]:
            raise ValueError("Capture at least one search before running the baseline.")
        case_results: list[dict[str, Any]] = []
        reciprocal_rank = 0.0
        top_1 = top_3 = top_5 = 0
        negative_hits_top_5 = 0
        latencies: list[float] = []
        with self.db.get_connection() as conn:
            for case in ready_cases:
                targets = conn.execute(
                    """SELECT job_id, kind, clip_id, entry_fingerprint, verdict
                       FROM stream_memory_evaluation_targets WHERE case_id = ?""",
                    (case["id"],),
                ).fetchall()
                relevant_ids = {
                    entry_id for row in targets
                    if row["verdict"] == "relevant"
                    and (entry_id := self._resolved_target_entry_id(conn, row))
                }
                negative_ids = {
                    entry_id for row in targets
                    if row["verdict"] == "not_relevant"
                    and (entry_id := self._resolved_target_entry_id(conn, row))
                }
                started = time.perf_counter()
                response = self.search(
                    case["query"], mode="hybrid", limit=10,
                    apply_creator_feedback=False,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies.append(elapsed_ms)
                result_ids = [str(item["id"]) for item in response["results"]]
                rank = next(
                    (index for index, result_id in enumerate(result_ids, start=1)
                     if result_id in relevant_ids),
                    None,
                )
                if rank is not None:
                    reciprocal_rank += 1.0 / rank
                    top_1 += int(rank <= 1)
                    top_3 += int(rank <= 3)
                    top_5 += int(rank <= 5)
                negative_hits = sum(1 for result_id in result_ids[:5] if result_id in negative_ids)
                negative_hits_top_5 += negative_hits
                case_results.append({
                    "case_id": case["id"],
                    "query": case["query"],
                    "top_rank": rank,
                    "result_count": len(result_ids),
                    "negative_hits_top_5": negative_hits,
                    "latency_ms": round(elapsed_ms, 2),
                })
        for case in summary["cases"]:
            if case["ready_to_score"]:
                continue
            case_results.append({
                "case_id": case["id"],
                "query": case["query"],
                "top_rank": None,
                "result_count": 0,
                "negative_hits_top_5": 0,
                "latency_ms": None,
                "status": "unresolved_miss",
            })
        count = len(summary["cases"])
        measured_count = len(ready_cases)
        ordered_latencies = sorted(latencies)
        p95_index = math.ceil(0.95 * len(ordered_latencies)) - 1 if ordered_latencies else 0
        revisions = self._evaluation_revisions()
        metrics = {
            "queries_scored": count,
            "queries_with_targets": measured_count,
            "queries_unresolved": int(summary["unresolved_cases"]),
            "top_1_rate": round(top_1 / count, 4),
            "top_3_rate": round(top_3 / count, 4),
            "top_5_rate": round(top_5 / count, 4),
            "mean_reciprocal_rank_at_10": round(reciprocal_rank / count, 4),
            "negative_hits_top_5": negative_hits_top_5,
            "average_latency_ms": round(sum(latencies) / measured_count, 2) if latencies else 0.0,
            "p95_latency_ms": round(ordered_latencies[p95_index], 2) if latencies else 0.0,
            "cases": case_results,
            **revisions,
        }
        semantic = self.semantic_state()
        system_id = f"stream-memory-v{MEMORY_SCHEMA_VERSION}:{semantic.get('semantic_model_id') or 'keyword'}"
        run_id = f"memory_eval_run_{uuid.uuid4().hex}"
        created_at = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO stream_memory_evaluation_runs
                   (id, system_id, case_count, metrics, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, system_id, count, json.dumps(metrics, sort_keys=True), created_at),
            )
        return {
            "id": run_id,
            "system_id": system_id,
            "case_count": count,
            "metrics": metrics,
            "created_at": created_at,
            "stale": False,
        }

    def _feedback_for_query(self, query: str) -> dict[str, str]:
        normalized = _normalized_query(query)
        if not normalized:
            return {}
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT f.job_id, f.kind, f.clip_id, f.entry_fingerprint, f.verdict
                   FROM stream_memory_feedback AS f
                   WHERE f.normalized_query = ?""",
                (normalized,),
            ).fetchall()
            resolved = [
                (self._resolved_target_entry_id(conn, row), str(row["verdict"]))
                for row in rows
            ]
        return {entry_id: verdict for entry_id, verdict in resolved if entry_id}

    @staticmethod
    def _learning_stable_key(row: Dict[str, Any]) -> str:
        clip_id = _clean_text(row.get("clip_id"))
        if clip_id:
            return f"clip:{clip_id}"
        return (
            f"entry:{_clean_text(row.get('job_id'))}:"
            f"{_clean_text(row.get('kind'))}:{_entry_fingerprint(row)}"
        )

    @staticmethod
    def _learning_context(row: Dict[str, Any]) -> dict[str, Any]:
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            evidence = _json_value(row.get("evidence_json"), {})
        labels = [
            _clean_text(value)[:60]
            for value in (evidence.get("labels") or [])
            if _clean_text(value)
        ][:12]
        review_state = _clean_text(row.get("review_state"))
        if not review_state and row.get("kind") == "clip":
            review_state = (
                "kept" if row.get("kept") else
                "passed" if row.get("passed") else
                "maybe" if row.get("maybe") else "unreviewed"
            )
        origin = _clean_text(row.get("origin"))
        if not origin:
            origin = (
                "creator_marker" if row.get("evidence_kind") == "creator_marker" else
                "scan_evidence" if row.get("evidence_kind") else
                "clip" if row.get("kind") == "clip" else "transcript"
            )
        return {
            "kind": _clean_text(row.get("kind")),
            "match_source": _clean_text(row.get("match_source")),
            "origin": origin,
            "game": _clean_text(row.get("game"))[:120],
            "evidence_kind": _clean_text(row.get("evidence_kind")),
            "moment_type": _clean_text(row.get("moment_type"))[:80],
            "scene": _clean_text(row.get("scene"))[:80],
            "review_state": review_state,
            "exported": bool(row.get("exported") or row.get("exported_at")),
            "labels": labels,
        }

    def _record_search_session(
        self,
        query: str,
        filters: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> str:
        search_id = f"memory_search_{uuid.uuid4().hex}"
        result_snapshots = []
        for rank, row in enumerate(rows, start=1):
            result_snapshots.append({
                "entry_id": str(row["id"]),
                "stable_key": self._learning_stable_key(row),
                "job_id": str(row["job_id"]),
                "kind": str(row["kind"]),
                "entry_index": int(row.get("entry_index") or 0),
                "clip_id": row.get("clip_id"),
                "entry_fingerprint": _entry_fingerprint(row),
                "rank": rank,
                "context": self._learning_context(row),
            })
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO stream_memory_search_sessions
                   (id, query_text, normalized_query, filters_json, results_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    search_id,
                    _clean_text(query)[:MAX_QUERY_LENGTH],
                    _normalized_query(query),
                    json.dumps(filters, separators=(",", ":"), sort_keys=True),
                    json.dumps(result_snapshots, separators=(",", ":"), sort_keys=True),
                    _utc_now(),
                ),
            )
            conn.execute(
                """DELETE FROM stream_memory_search_sessions
                   WHERE id IN (
                     SELECT id FROM stream_memory_search_sessions
                     ORDER BY created_at DESC, id DESC
                     LIMIT -1 OFFSET ?
                   )""",
                (MAX_LEARNING_SEARCH_SESSIONS,),
            )
        return search_id

    def record_interactions(
        self,
        search_id: str,
        event_type: str,
        entries: list[dict[str, Any]],
    ) -> Dict[str, Any]:
        """Record visible/opened results from one server-issued search snapshot."""
        if event_type not in {"impression", "open"}:
            raise ValueError("Memory interactions must be impression or open events.")
        cleaned_search_id = _clean_text(search_id)
        if not cleaned_search_id:
            raise ValueError("Memory search id cannot be empty.")
        if not entries or len(entries) > MAX_RESULTS:
            raise ValueError(f"Record between 1 and {MAX_RESULTS} Memory results at a time.")
        with self.db.get_connection() as conn:
            session = conn.execute(
                """SELECT query_text, normalized_query, results_json
                   FROM stream_memory_search_sessions WHERE id = ?""",
                (cleaned_search_id,),
            ).fetchone()
            if not session:
                raise KeyError("That Memory search is no longer available.")
            snapshots = {
                str(item.get("entry_id")): item
                for item in _json_value(session["results_json"], [])
                if isinstance(item, dict) and item.get("entry_id")
            }
            recorded = 0
            for requested in entries:
                entry_id = _clean_text(requested.get("entry_id"))
                snapshot = snapshots.get(entry_id)
                if snapshot is None:
                    raise ValueError("That result was not part of this Memory search.")
                stable_key = str(snapshot["stable_key"])
                dedupe_key = f"{event_type}:{cleaned_search_id}:{stable_key}"
                cursor = conn.execute(
                    """INSERT OR IGNORE INTO stream_memory_interactions
                       (id, search_id, event_type, query_text, normalized_query,
                        job_id, kind, entry_index, clip_id, entry_fingerprint,
                        stable_key, rank, context_json, dedupe_key, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"memory_interaction_{uuid.uuid4().hex}", cleaned_search_id,
                        event_type, session["query_text"], session["normalized_query"],
                        snapshot["job_id"], snapshot["kind"], snapshot["entry_index"],
                        snapshot.get("clip_id"), snapshot["entry_fingerprint"],
                        stable_key, int(snapshot.get("rank") or 0),
                        json.dumps(snapshot.get("context") or {}, separators=(",", ":"), sort_keys=True),
                        dedupe_key, _utc_now(),
                    ),
                )
                recorded += int(cursor.rowcount > 0)
        return {"search_id": cleaned_search_id, "event_type": event_type, "recorded": recorded}

    def record_action(
        self,
        query: str,
        entry_id: str,
        event_type: str,
        *,
        dedupe_key: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist a verified Memory-origin creation or compilation action."""
        if event_type not in {"clip_created", "compilation_add"}:
            raise ValueError("Unknown Memory action event.")
        cleaned_dedupe_key = _clean_text(dedupe_key)[:240]
        if not cleaned_dedupe_key:
            raise ValueError("Memory action dedupe key cannot be empty.")
        cleaned_entry_id = _clean_text(entry_id)
        row = self._result_rows([cleaned_entry_id]).get(cleaned_entry_id)
        if row is None:
            raise KeyError("That Memory moment is no longer in the local index.")
        fingerprint = _entry_fingerprint(row)
        learning_context = self._learning_context(row)
        if context:
            learning_context["action"] = {
                _clean_text(key)[:60]: _clean_text(value)[:160]
                for key, value in context.items()
                if _clean_text(key) and _clean_text(value)
            }
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO stream_memory_interactions
                   (id, search_id, event_type, query_text, normalized_query,
                    job_id, kind, entry_index, clip_id, entry_fingerprint,
                    stable_key, rank, context_json, dedupe_key, created_at)
                   VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)""",
                (
                    f"memory_interaction_{uuid.uuid4().hex}", event_type,
                    _clean_text(query)[:MAX_QUERY_LENGTH], _normalized_query(query),
                    row["job_id"], row["kind"], int(row.get("entry_index") or 0),
                    row.get("clip_id"), fingerprint, self._learning_stable_key(row),
                    json.dumps(learning_context, separators=(",", ":"), sort_keys=True),
                    cleaned_dedupe_key, _utc_now(),
                ),
            )
        return {"event_type": event_type, "entry_id": cleaned_entry_id, "recorded": bool(cursor.rowcount)}

    @staticmethod
    def _learning_feature_keys(context: dict[str, Any]) -> list[str]:
        keys = []
        for field in ("kind", "match_source", "origin", "game", "evidence_kind", "moment_type", "scene"):
            value = _clean_text(context.get(field)).casefold()
            if value:
                keys.append(f"{field}:{value}")
        keys.extend(
            f"label:{_clean_text(label).casefold()}"
            for label in (context.get("labels") or [])
            if _clean_text(label)
        )
        return keys

    def _learning_revision(self) -> str:
        with self.db.get_connection() as conn:
            interaction_rows = conn.execute(
                """SELECT id, event_type, normalized_query, stable_key, rank, created_at
                   FROM stream_memory_interactions ORDER BY id"""
            ).fetchall()
            editorial_rows = conn.execute(
                """SELECT c.id, c.kept, c.passed, c.maybe, c.exported_at,
                          COALESCE(MAX(l.id), 0) AS latest_label_id,
                          COUNT(DISTINCT i.project_id) AS compilation_count
                   FROM clips AS c
                   LEFT JOIN clip_labels AS l ON l.clip_id = c.id
                   LEFT JOIN compilation_project_items AS i
                     ON i.clip_id = c.id AND i.included = 1
                   GROUP BY c.id ORDER BY c.id"""
            ).fetchall()
        return _revision_digest([*interaction_rows, *editorial_rows])

    def _learning_profile(self, *, exclude_query: str = "") -> dict[str, Any]:
        excluded = _normalized_query(exclude_query)
        with self.db.get_connection() as conn:
            interaction_rows = conn.execute(
                """SELECT event_type, normalized_query, stable_key, context_json
                   FROM stream_memory_interactions
                   WHERE ? = '' OR normalized_query != ?
                   ORDER BY created_at, id""",
                (excluded, excluded),
            ).fetchall()
            editorial_rows = conn.execute(
                """SELECT m.game, m.evidence_kind, m.evidence_json,
                          c.id AS clip_id, c.moment_type, c.scene,
                          c.kept, c.passed, c.maybe, c.exported_at,
                          COUNT(DISTINCT i.project_id) AS compilation_count
                   FROM stream_memory_entries AS m
                   JOIN clips AS c ON c.id = m.clip_id
                   LEFT JOIN compilation_project_items AS i
                     ON i.clip_id = c.id AND i.included = 1
                   WHERE m.kind = 'clip'
                   GROUP BY m.id"""
            ).fetchall()

        direct: dict[str, dict[str, float]] = {}
        feature_history: dict[str, dict[str, float]] = {}
        event_counts = {name: 0 for name in ("impression", "open", "clip_created", "compilation_add")}
        action_weight = {"open": 0.45, "clip_created": 1.0, "compilation_add": 0.8}
        for row in interaction_rows:
            event_type = str(row["event_type"])
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            context = _json_value(row["context_json"], {})
            bucket = direct.setdefault(str(row["stable_key"]), {"impressions": 0.0, "actions": 0.0})
            if event_type == "impression":
                bucket["impressions"] += 1.0
            else:
                bucket["actions"] += action_weight.get(event_type, 0.0)
            for key in self._learning_feature_keys(context):
                feature = feature_history.setdefault(key, {"impressions": 0.0, "actions": 0.0})
                if event_type == "impression":
                    feature["impressions"] += 1.0
                else:
                    feature["actions"] += action_weight.get(event_type, 0.0)

        editorial: dict[str, dict[str, float]] = {}
        reviewed_clips = 0
        exported_clips = 0
        compilation_clips = 0
        for row in editorial_rows:
            kept = bool(row["kept"])
            passed = bool(row["passed"])
            maybe = bool(row["maybe"])
            exported = bool(row["exported_at"])
            compilation_count = int(row["compilation_count"] or 0)
            if not any((kept, passed, maybe, exported, compilation_count)):
                continue
            reviewed_clips += int(any((kept, passed, maybe)))
            exported_clips += int(exported)
            compilation_clips += int(compilation_count > 0)
            signal = (
                (-1.0 if passed else 0.7 if kept else 0.15 if maybe else 0.0)
                + (0.2 if compilation_count else 0.0)
                + (0.3 if exported else 0.0)
            )
            signal = max(-1.0, min(1.0, signal))
            evidence = _json_value(row["evidence_json"], {})
            context = {
                "kind": "clip",
                "game": row["game"],
                "evidence_kind": row["evidence_kind"],
                "moment_type": row["moment_type"],
                "scene": row["scene"],
                "labels": evidence.get("labels") or [],
            }
            for key in self._learning_feature_keys(context):
                feature = editorial.setdefault(key, {"signal": 0.0, "count": 0.0})
                feature["signal"] += signal
                feature["count"] += 1.0

        return {
            "direct": direct,
            "feature_history": feature_history,
            "editorial": editorial,
            "event_counts": event_counts,
            "reviewed_clips": reviewed_clips,
            "exported_clips": exported_clips,
            "compilation_clips": compilation_clips,
        }

    def _shadow_learning_scores(
        self,
        rows: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        raw_rows = self._result_rows([str(row["id"]) for row in rows])
        scores: dict[str, dict[str, float]] = {}
        for row in rows:
            entry_id = str(row["id"])
            raw = raw_rows.get(entry_id, {})
            combined = {**raw, **row}
            context = self._learning_context(combined)
            stable_key = self._learning_stable_key(combined)
            direct = profile["direct"].get(stable_key, {"impressions": 0.0, "actions": 0.0})
            direct_score = min(
                1.0,
                float(direct["actions"]) / math.sqrt(float(direct["impressions"]) + 1.0),
            )
            history_values = []
            taste_values = []
            for key in self._learning_feature_keys(context):
                history = profile["feature_history"].get(key)
                if history:
                    history_values.append(
                        min(1.0, float(history["actions"]) / (float(history["impressions"]) + 5.0))
                    )
                taste = profile["editorial"].get(key)
                if taste:
                    taste_values.append(float(taste["signal"]) / (float(taste["count"]) + 4.0))
            history_score = sum(history_values) / len(history_values) if history_values else 0.0
            taste_score = sum(taste_values) / len(taste_values) if taste_values else 0.0
            scores[entry_id] = {
                "historical_relevance": round(0.7 * direct_score + 0.3 * history_score, 6),
                "editorial_taste": round(max(-1.0, min(1.0, taste_score)), 6),
            }
        return scores

    def learning_summary(self) -> Dict[str, Any]:
        profile = self._learning_profile()
        counts = profile["event_counts"]
        with self.db.get_connection() as conn:
            latest_row = conn.execute(
                """SELECT id, system_id, event_revision, case_count, metrics, created_at
                   FROM stream_memory_shadow_runs ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
        positive_actions = sum(counts.get(name, 0) for name in ("open", "clip_created", "compilation_add"))
        measurement_floor = (
            counts.get("impression", 0) >= 20
            and positive_actions >= 10
            and profile["reviewed_clips"] >= 20
        )
        latest_run = None
        if latest_row:
            latest_run = dict(latest_row)
            latest_run["metrics"] = _json_value(latest_run["metrics"], {})
            latest_run["stale"] = latest_run["event_revision"] != self._learning_revision()
        blockers = []
        if counts.get("impression", 0) < 20:
            blockers.append("Collect at least 20 result impressions.")
        if positive_actions < 10:
            blockers.append("Collect at least 10 opens or creation actions.")
        if profile["reviewed_clips"] < 20:
            blockers.append("Review at least 20 clips with Keep, Maybe, or Pass.")
        blockers.append("Shadow results require explicit promotion approval before live use.")
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "learning_version": MEMORY_LEARNING_VERSION,
            "status": "measurable" if measurement_floor else "collecting",
            "event_counts": counts,
            "positive_action_count": positive_actions,
            "reviewed_clip_count": profile["reviewed_clips"],
            "exported_clip_count": profile["exported_clips"],
            "compilation_clip_count": profile["compilation_clips"],
            "measurement_floor_met": measurement_floor,
            "live_weight": 0.0,
            "eligible_for_promotion": False,
            "promotion_blockers": blockers,
            "latest_run": latest_run,
        }

    def clear_learning_history(self) -> Dict[str, Any]:
        """Clear passive Memory behavior without deleting explicit creator truth."""
        with self.db.get_connection() as conn:
            interaction_count = int(conn.execute(
                "SELECT COUNT(*) FROM stream_memory_interactions"
            ).fetchone()[0])
            search_count = int(conn.execute(
                "SELECT COUNT(*) FROM stream_memory_search_sessions"
            ).fetchone()[0])
            run_count = int(conn.execute(
                "SELECT COUNT(*) FROM stream_memory_shadow_runs"
            ).fetchone()[0])
            conn.execute("DELETE FROM stream_memory_shadow_runs")
            conn.execute("DELETE FROM stream_memory_interactions")
            conn.execute("DELETE FROM stream_memory_search_sessions")
        return {
            "cleared": True,
            "searches": search_count,
            "interactions": interaction_count,
            "shadow_runs": run_count,
        }

    @staticmethod
    def _shadow_metric_summary(
        case_count: int,
        ranks: list[Optional[int]],
        negative_hits: int,
    ) -> dict[str, Any]:
        reciprocal = sum(1.0 / rank for rank in ranks if rank is not None and rank <= 10)
        return {
            "top_1_rate": round(sum(rank is not None and rank <= 1 for rank in ranks) / case_count, 4),
            "top_3_rate": round(sum(rank is not None and rank <= 3 for rank in ranks) / case_count, 4),
            "top_5_rate": round(sum(rank is not None and rank <= 5 for rank in ranks) / case_count, 4),
            "mean_reciprocal_rank_at_10": round(reciprocal / case_count, 4),
            "negative_hits_top_5": negative_hits,
        }

    def run_learning_shadow_evaluation(self) -> Dict[str, Any]:
        summary = self.evaluation_summary()
        if not summary["cases"]:
            raise ValueError("Capture at least one Memory search test before running learning shadow.")
        baseline_ranks: list[Optional[int]] = []
        shadow_ranks: list[Optional[int]] = []
        baseline_negative_hits = 0
        shadow_negative_hits = 0
        case_results = []
        with self.db.get_connection() as conn:
            for case in summary["cases"]:
                if not case["ready_to_score"]:
                    baseline_ranks.append(None)
                    shadow_ranks.append(None)
                    case_results.append({
                        "case_id": case["id"], "query": case["query"],
                        "baseline_rank": None, "shadow_rank": None,
                        "status": "unresolved_miss",
                    })
                    continue
                targets = conn.execute(
                    """SELECT job_id, kind, clip_id, entry_fingerprint, verdict
                       FROM stream_memory_evaluation_targets WHERE case_id = ?""",
                    (case["id"],),
                ).fetchall()
                relevant_ids = {
                    entry_id for target in targets
                    if target["verdict"] == "relevant"
                    and (entry_id := self._resolved_target_entry_id(conn, target))
                }
                negative_ids = {
                    entry_id for target in targets
                    if target["verdict"] == "not_relevant"
                    and (entry_id := self._resolved_target_entry_id(conn, target))
                }
                response = self.search(
                    case["query"], mode="hybrid", limit=40,
                    apply_creator_feedback=False, track_interactions=False,
                )
                rows = response["results"]
                baseline_ids = [str(row["id"]) for row in rows]
                profile = self._learning_profile(exclude_query=case["query"])
                learning_scores = self._shadow_learning_scores(rows, profile)
                shadow_rows = sorted(
                    enumerate(rows, start=1),
                    key=lambda item: (
                        item[0]
                        - 2.0 * learning_scores[str(item[1]["id"])]["historical_relevance"]
                        - learning_scores[str(item[1]["id"])]["editorial_taste"],
                        item[0],
                    ),
                )
                shadow_ids = [str(row["id"]) for _position, row in shadow_rows]
                baseline_rank = next(
                    (position for position, entry_id in enumerate(baseline_ids[:10], start=1)
                     if entry_id in relevant_ids), None,
                )
                shadow_rank = next(
                    (position for position, entry_id in enumerate(shadow_ids[:10], start=1)
                     if entry_id in relevant_ids), None,
                )
                baseline_hits = sum(entry_id in negative_ids for entry_id in baseline_ids[:5])
                shadow_hits = sum(entry_id in negative_ids for entry_id in shadow_ids[:5])
                baseline_ranks.append(baseline_rank)
                shadow_ranks.append(shadow_rank)
                baseline_negative_hits += baseline_hits
                shadow_negative_hits += shadow_hits
                case_results.append({
                    "case_id": case["id"], "query": case["query"],
                    "baseline_rank": baseline_rank, "shadow_rank": shadow_rank,
                    "baseline_negative_hits_top_5": baseline_hits,
                    "shadow_negative_hits_top_5": shadow_hits,
                })
        case_count = len(summary["cases"])
        baseline = self._shadow_metric_summary(case_count, baseline_ranks, baseline_negative_hits)
        shadow = self._shadow_metric_summary(case_count, shadow_ranks, shadow_negative_hits)
        deltas = {
            key: round(float(shadow[key]) - float(baseline[key]), 4)
            for key in ("top_1_rate", "top_3_rate", "top_5_rate", "mean_reciprocal_rank_at_10")
        }
        deltas["negative_hits_top_5"] = shadow_negative_hits - baseline_negative_hits
        learning = self.learning_summary()
        metrics = {
            "queries_scored": case_count,
            "queries_with_targets": int(summary["scorable_cases"]),
            "queries_unresolved": int(summary["unresolved_cases"]),
            "baseline": baseline,
            "shadow": shadow,
            "deltas": deltas,
            "cases": case_results,
            "measurement_floor_met": learning["measurement_floor_met"],
            "eligible_for_promotion": False,
            "live_weight": 0.0,
        }
        event_revision = self._learning_revision()
        semantic = self.semantic_state()
        system_id = (
            f"stream-memory-v{MEMORY_SCHEMA_VERSION}:learning-v{MEMORY_LEARNING_VERSION}:"
            f"{semantic.get('semantic_model_id') or 'keyword'}:shadow"
        )
        run_id = f"memory_shadow_run_{uuid.uuid4().hex}"
        created_at = _utc_now()
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT INTO stream_memory_shadow_runs
                   (id, system_id, event_revision, case_count, metrics, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, system_id, event_revision, case_count,
                 json.dumps(metrics, sort_keys=True), created_at),
            )
        return {
            "id": run_id, "system_id": system_id, "event_revision": event_revision,
            "case_count": case_count, "metrics": metrics, "created_at": created_at,
            "stale": False,
        }

    def _query_with_aliases(
        self,
        query: str,
        *,
        allow_empty: bool = False,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        raw = _clean_text(query)[:MAX_QUERY_LENGTH]
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT canonical_term, alias FROM stream_memory_aliases ORDER BY LENGTH(alias) DESC"
            ).fetchall()
        families: dict[str, set[str]] = {}
        labels: dict[str, str] = {}
        for row in rows:
            canonical = _clean_text(row["canonical_term"])
            key = canonical.casefold()
            labels.setdefault(key, canonical)
            families.setdefault(key, {canonical}).add(_clean_text(row["alias"]))

        remaining = raw
        required_parts: list[str] = []
        expansions: list[dict[str, Any]] = []
        semantic_terms: list[str] = []
        for key, variants in families.items():
            matched = None
            for variant in sorted(variants, key=len, reverse=True):
                pattern = rf"(?<!\w){re.escape(variant)}(?!\w)"
                if re.search(pattern, remaining, flags=re.IGNORECASE):
                    matched = variant
                    remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE)
                    break
            if not matched:
                continue
            ordered = sorted(variants, key=lambda value: (value.casefold() != key, value.casefold()))
            required_parts.append(
                f"({' OR '.join(_quoted_prefix(value) for value in ordered)})"
            )
            canonical = labels[key]
            semantic_terms.append(canonical)
            expansions.append({
                "canonical_term": canonical,
                "matched_term": matched,
                "terms": ordered,
            })
        if not _clean_text(remaining) and not required_parts and allow_empty:
            expression = ""
        else:
            try:
                expression = _fts_query(remaining, required_parts=required_parts)
            except ValueError:
                if not allow_empty:
                    raise
                expression = ""
        semantic_query = " ".join(filter(None, [raw, *semantic_terms]))
        return expression, semantic_query, expansions

    def _transcript_entries(
        self,
        job: Dict[str, Any],
        transcript: Optional[Dict[str, Any]],
    ) -> list[dict]:
        entries: list[dict] = []
        game_segments = self._timeline_segments(job["id"])
        for index, segment in enumerate((transcript or {}).get("segments") or []):
            if not isinstance(segment, dict):
                continue
            text = _clean_text(segment.get("text"))
            if not text:
                words = segment.get("words") or []
                text = _clean_text(" ".join(
                    str(word.get("word") or "")
                    for word in words
                    if isinstance(word, dict)
                ))
            if not text:
                continue
            try:
                start = max(0.0, float(segment.get("start", 0.0) or 0.0))
                end = max(start, float(segment.get("end", start) or start))
            except (TypeError, ValueError):
                continue
            entries.append({
                "id": str(uuid.uuid4()),
                "job_id": job["id"],
                "clip_id": None,
                "kind": "transcript",
                "entry_index": index,
                "start_time": start,
                "end_time": end,
                "title": "Conversation",
                "text": text,
                "session_name": _clean_text(job.get("session_name")) or "Untitled session",
                "game": _game_at(game_segments, start),
                "source_date": job.get("source_date"),
                "evidence_kind": None,
                "evidence_json": "{}",
            })
        return entries

    def _clip_entries(self, job: Dict[str, Any]) -> list[dict]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT id, start_time, end_time, title, description, reason,
                          signals, tags, hook_line, moment_type, scene
                   FROM clips WHERE job_id = ? ORDER BY start_time, id""",
                (job["id"],),
            ).fetchall()
        game_segments = self._timeline_segments(job["id"])
        entries = []
        for index, row in enumerate(rows):
            start = max(0.0, float(row["start_time"] or 0.0))
            end = max(start, float(row["end_time"] or start))
            signals = _json_value(row["signals"], [])
            tags = _json_value(row["tags"], [])
            searchable = _clean_text(" ".join([
                str(row["title"] or ""),
                str(row["description"] or ""),
                str(row["reason"] or ""),
                str(row["hook_line"] or ""),
                str(row["moment_type"] or ""),
                str(row["scene"] or ""),
                " ".join(str(value) for value in signals if value),
                " ".join(str(value) for value in tags if value),
            ]))
            entries.append({
                "id": str(uuid.uuid4()),
                "job_id": job["id"],
                "clip_id": row["id"],
                "kind": "clip",
                "entry_index": index,
                "start_time": start,
                "end_time": end,
                "title": _clean_text(row["title"]) or "Untitled moment",
                "text": searchable,
                "session_name": _clean_text(job.get("session_name")) or "Untitled session",
                "game": _game_at(game_segments, start),
                "source_date": job.get("source_date"),
                "evidence_kind": None,
                "evidence_json": "{}",
            })
        return entries

    @staticmethod
    def _apply_clip_evidence(entries: list[dict], packs: Dict[str, dict]) -> None:
        for entry in entries:
            pack = packs.get(str(entry.get("clip_id")))
            if not pack:
                continue
            search_text = _clean_text(pack.get("search_text"))
            if search_text:
                entry["text"] = _clean_text(f"{entry['text']} {search_text}")
            entry["evidence_kind"] = "clip_pack"
            entry["evidence_json"] = json.dumps(
                pack, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
            )

    def _standalone_evidence_entries(
        self,
        job: Dict[str, Any],
        rows: list[dict],
    ) -> list[dict]:
        game_segments = self._timeline_segments(job["id"])
        entries = []
        for index, row in enumerate(rows):
            start = max(0.0, float(row.get("start_time") or 0.0))
            end = max(start, float(row.get("end_time") or start))
            entries.append({
                "id": str(uuid.uuid4()),
                "job_id": job["id"],
                "clip_id": None,
                "kind": "evidence",
                "entry_index": index,
                "start_time": start,
                "end_time": end,
                "title": _clean_text(row.get("title")) or "Detected moment",
                "text": _clean_text(row.get("text")),
                "session_name": _clean_text(job.get("session_name")) or "Untitled session",
                "game": _game_at(game_segments, start),
                "source_date": job.get("source_date"),
                "evidence_kind": _clean_text(row.get("evidence_kind")) or "detected_event",
                "evidence_json": json.dumps(
                    row.get("evidence_json") or {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            })
        return entries

    def index_job(
        self,
        job_id: str,
        *,
        transcript: Optional[Dict[str, Any]] = None,
        transcript_coverage: str = "unknown",
    ) -> Dict[str, Any]:
        job = self._job(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")
        transcript_entries = self._transcript_entries(job, transcript)
        clip_entries = self._clip_entries(job)
        evidence_error = None
        try:
            evidence = build_evidence_packs(
                self.data_root,
                job_id,
                clip_entries,
                region_plan=job.get("region_plan") or {},
                source_path=next((
                    path for path in (job.get("asset_path"), job.get("source_path"))
                    if path and os.path.isfile(path)
                ), None),
            )
            self._apply_clip_evidence(clip_entries, evidence.clip_packs)
            evidence_entries = self._standalone_evidence_entries(
                job, evidence.standalone_entries,
            )
            evidence_status = evidence.status
            evidence_source_bytes = evidence.source_bytes
            evidence_pack_count = evidence.pack_count
            evidence_coverage = evidence.coverage
        except Exception as exc:  # evidence must never block transcript/clip indexing
            evidence_entries = []
            evidence_status = "error"
            evidence_source_bytes = 0
            evidence_pack_count = 0
            evidence_coverage = {}
            evidence_error = str(exc)[:500]
        entries = transcript_entries + clip_entries + evidence_entries
        columns = (
            "id", "job_id", "clip_id", "kind", "entry_index", "start_time",
            "end_time", "title", "text", "session_name", "game", "source_date",
            "evidence_kind", "evidence_json",
        )
        placeholders = ",".join("?" for _ in columns)
        with self.db.get_connection() as conn:
            conn.execute("DELETE FROM stream_memory_entries WHERE job_id = ?", (job_id,))
            if entries:
                conn.executemany(
                    f"INSERT INTO stream_memory_entries ({','.join(columns)}) VALUES ({placeholders})",  # nosec B608
                    [tuple(entry[column] for column in columns) for entry in entries],
                )
            conn.execute(
                """INSERT OR REPLACE INTO stream_memory_index_state
                   (job_id, transcript_status, transcript_coverage,
                     transcript_entries, clip_entries, evidence_status,
                     evidence_entries, evidence_version, evidence_source_bytes,
                     coverage_json, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    "indexed" if transcript_entries else "unavailable",
                    transcript_coverage if transcript_entries else "none",
                    len(transcript_entries),
                    len(clip_entries),
                    evidence_status,
                    evidence_pack_count,
                    EVIDENCE_PACK_VERSION,
                    evidence_source_bytes,
                    json.dumps(evidence_coverage, separators=(",", ":"), sort_keys=True),
                    _utc_now(),
                ),
            )
            self._mark_semantic_stale(conn)
        return {
            "job_id": job_id,
            "transcript_status": "indexed" if transcript_entries else "unavailable",
            "transcript_coverage": transcript_coverage if transcript_entries else "none",
            "transcript_entries": len(transcript_entries),
            "clip_entries": len(clip_entries),
            "evidence_entries": evidence_pack_count,
            "evidence_status": evidence_status,
            "evidence_source_bytes": evidence_source_bytes,
            "evidence_coverage": evidence_coverage,
            "evidence_error": evidence_error,
            "total_entries": len(entries),
        }

    def refresh_job_clips(self, job_id: str) -> Dict[str, Any]:
        """Rebuild only clip metadata while preserving transcript entries."""
        job = self._job(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")
        with self.db.get_connection() as conn:
            previous = conn.execute(
                """SELECT clip_id, evidence_json
                   FROM stream_memory_entries
                   WHERE job_id = ? AND kind = 'clip' AND evidence_kind = 'clip_pack'""",
                (job_id,),
            ).fetchall()
        preserved_packs = {
            str(row["clip_id"]): _json_value(row["evidence_json"], {})
            for row in previous if row["clip_id"]
        }
        entries = self._clip_entries(job)
        self._apply_clip_evidence(entries, preserved_packs)
        columns = (
            "id", "job_id", "clip_id", "kind", "entry_index", "start_time",
            "end_time", "title", "text", "session_name", "game", "source_date",
            "evidence_kind", "evidence_json",
        )
        placeholders = ",".join("?" for _ in columns)
        with self.db.get_connection() as conn:
            conn.execute(
                "DELETE FROM stream_memory_entries WHERE job_id = ? AND kind = 'clip'",
                (job_id,),
            )
            if entries:
                conn.executemany(
                    f"INSERT INTO stream_memory_entries ({','.join(columns)}) VALUES ({placeholders})",  # nosec B608
                    [tuple(entry[column] for column in columns) for entry in entries],
                )
            conn.execute(
                """INSERT INTO stream_memory_index_state
                   (job_id, transcript_status, transcript_coverage,
                    transcript_entries, clip_entries, indexed_at)
                   VALUES (?, 'unavailable', 'none', 0, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                     clip_entries = excluded.clip_entries,
                     indexed_at = excluded.indexed_at""",
                (job_id, len(entries), _utc_now()),
            )
            self._mark_semantic_stale(conn)
        return {"job_id": job_id, "clip_entries": len(entries)}

    def refresh_clip(self, clip_id: str) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT job_id FROM clips WHERE id = ?", (clip_id,),
            ).fetchone()
        if not row:
            raise KeyError(f"Clip not found: {clip_id}")
        return self.refresh_job_clips(row["job_id"])

    def refresh_job_evidence(self, job_id: str) -> Dict[str, Any]:
        """Upgrade evidence rows without risking historical transcript loss."""
        job = self._job(job_id)
        if not job:
            raise KeyError(f"Job not found: {job_id}")
        clip_entries = self._clip_entries(job)
        evidence = build_evidence_packs(
            self.data_root,
            job_id,
            clip_entries,
            region_plan=job.get("region_plan") or {},
            source_path=next((
                path for path in (job.get("asset_path"), job.get("source_path"))
                if path and os.path.isfile(path)
            ), None),
        )
        self._apply_clip_evidence(clip_entries, evidence.clip_packs)
        evidence_entries = self._standalone_evidence_entries(
            job, evidence.standalone_entries,
        )
        entries = clip_entries + evidence_entries
        columns = (
            "id", "job_id", "clip_id", "kind", "entry_index", "start_time",
            "end_time", "title", "text", "session_name", "game", "source_date",
            "evidence_kind", "evidence_json",
        )
        placeholders = ",".join("?" for _ in columns)
        with self.db.get_connection() as conn:
            conn.execute(
                "DELETE FROM stream_memory_entries WHERE job_id = ? AND kind IN ('clip', 'evidence')",
                (job_id,),
            )
            if entries:
                conn.executemany(
                    f"INSERT INTO stream_memory_entries ({','.join(columns)}) VALUES ({placeholders})",  # nosec B608
                    [tuple(entry[column] for column in columns) for entry in entries],
                )
            conn.execute(
                """INSERT INTO stream_memory_index_state
                   (job_id, transcript_status, transcript_coverage,
                     transcript_entries, clip_entries, evidence_status,
                     evidence_entries, evidence_version, evidence_source_bytes,
                     coverage_json, indexed_at)
                   VALUES (?, 'unavailable', 'none', 0, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(job_id) DO UPDATE SET
                     clip_entries = excluded.clip_entries,
                     evidence_status = excluded.evidence_status,
                      evidence_entries = excluded.evidence_entries,
                      evidence_version = excluded.evidence_version,
                      evidence_source_bytes = excluded.evidence_source_bytes,
                      coverage_json = excluded.coverage_json,
                      indexed_at = excluded.indexed_at""",
                (
                    job_id,
                    len(clip_entries),
                    evidence.status,
                    evidence.pack_count,
                    EVIDENCE_PACK_VERSION,
                    evidence.source_bytes,
                    json.dumps(evidence.coverage, separators=(",", ":"), sort_keys=True),
                    _utc_now(),
                ),
            )
            self._mark_semantic_stale(conn)
        return {
            "job_id": job_id,
            "clip_entries": len(clip_entries),
            "evidence_entries": evidence.pack_count,
            "evidence_status": evidence.status,
            "evidence_source_bytes": evidence.source_bytes,
            "evidence_coverage": evidence.coverage,
            "entries_written": len(entries),
        }

    @staticmethod
    def _cache_labels() -> list[str]:
        from engines.caption.whisper_asr import asr_cache_label

        # One backend, one label. The whisper compatibility labels went with
        # the whisper backend on 2026-09-01: nothing on disk carried them, and
        # a build cannot ship without the Qwen weights that produce this one.
        return [asr_cache_label()]

    def _cached_transcript(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from core import cache_keys, signal_cache

        candidates = [job.get("asset_path"), job.get("source_path")]
        source = next(
            (os.path.abspath(path) for path in candidates if path and os.path.isfile(path)),
            None,
        )
        if not source:
            return None
        try:
            audio_key = cache_keys.audio_cache_key(source)
        except OSError:
            return None
        for label in self._cache_labels():
            path = os.path.join(
                self.data_root,
                "cache",
                cache_keys.transcript_cache_name(
                    audio_key, label, signal_cache.CACHE_EXT,
                ),
            )
            transcript = signal_cache.load(path)
            if transcript:
                return transcript
        return None

    def reindex_completed(
        self,
        *,
        force: bool = False,
        cancel_check=None,
    ) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT j.id, s.job_id AS indexed_job_id,
                          COALESCE(s.evidence_version, 0) AS evidence_version
                   FROM jobs AS j
                   LEFT JOIN stream_memory_index_state AS s ON s.job_id = j.id
                   WHERE j.status = 'completed'
                   ORDER BY j.created_at"""
            ).fetchall()
        indexed = skipped = failed = transcript_jobs = evidence_jobs = 0
        cancelled = False
        entries = evidence_entries = 0
        errors = []
        for row in rows:
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            try:
                if row["indexed_job_id"] and not force:
                    if int(row["evidence_version"] or 0) >= EVIDENCE_PACK_VERSION:
                        skipped += 1
                        continue
                    result = self.refresh_job_evidence(row["id"])
                    indexed += 1
                    evidence_jobs += 1
                    evidence_entries += int(result["evidence_entries"])
                    entries += int(result["entries_written"])
                    continue
                job = self._job(row["id"])
                transcript = self._cached_transcript(job or {})
                result = self.index_job(
                    row["id"],
                    transcript=transcript,
                    transcript_coverage="full" if transcript else "none",
                )
                indexed += 1
                entries += result["total_entries"]
                transcript_jobs += int(result["transcript_entries"] > 0)
                evidence_jobs += int(result["evidence_status"] == "indexed")
                evidence_entries += int(result["evidence_entries"])
            except Exception as exc:  # noqa: BLE001 - one old job cannot block the library
                failed += 1
                errors.append({"job_id": row["id"], "error": str(exc)[:240]})
        return {
            "indexed_jobs": indexed,
            "skipped_jobs": skipped,
            "failed_jobs": failed,
            "transcript_jobs": transcript_jobs,
            "evidence_jobs": evidence_jobs,
            "evidence_entries": evidence_entries,
            "entries_written": entries,
            "cancelled": cancelled,
            "errors": errors[:20],
        }

    def has_pending_exact_jobs(self) -> bool:
        """Return whether a completed scan is missing the current exact index."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """SELECT 1
                   FROM jobs AS j
                   LEFT JOIN stream_memory_index_state AS s ON s.job_id = j.id
                   WHERE j.status = 'completed'
                     AND (s.job_id IS NULL OR COALESCE(s.evidence_version, 0) < ?)
                   LIMIT 1""",
                (EVIDENCE_PACK_VERSION,),
            ).fetchone()
        return row is not None

    def stats(self) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS indexed_jobs,
                          COALESCE(SUM(transcript_entries), 0) AS transcript_entries,
                          COALESCE(SUM(clip_entries), 0) AS clip_entries,
                          COALESCE(SUM(evidence_entries), 0) AS evidence_entries,
                          COALESCE(SUM(evidence_source_bytes), 0) AS evidence_source_bytes,
                          MAX(indexed_at) AS last_indexed_at
                   FROM stream_memory_index_state"""
            ).fetchone()
            completed = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'completed'"
            ).fetchone()[0]
            alias_count = conn.execute("SELECT COUNT(*) FROM stream_memory_aliases").fetchone()[0]
            feedback_count = conn.execute("SELECT COUNT(*) FROM stream_memory_feedback").fetchone()[0]
        semantic = self.semantic_state()
        coverage = self.evidence_coverage_summary()
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "completed_jobs": int(completed or 0),
            "indexed_jobs": int(row["indexed_jobs"] or 0),
            "transcript_entries": int(row["transcript_entries"] or 0),
            "clip_entries": int(row["clip_entries"] or 0),
            "evidence_entries": int(row["evidence_entries"] or 0),
            "evidence_source_bytes": int(row["evidence_source_bytes"] or 0),
            "evidence_version": EVIDENCE_PACK_VERSION,
            "last_indexed_at": row["last_indexed_at"],
            "alias_count": int(alias_count or 0),
            "feedback_count": int(feedback_count or 0),
            "evidence_coverage": coverage,
            **semantic,
        }

    def evidence_coverage_summary(self) -> Dict[str, Any]:
        """Aggregate compact coverage measurements; absence never implies no event."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT coverage_json
                   FROM stream_memory_index_state
                   WHERE evidence_version >= ? AND coverage_json != '{}'""",
                (EVIDENCE_PACK_VERSION,),
            ).fetchall()
        reports = [
            value for row in rows
            if isinstance((value := _json_value(row["coverage_json"], {})), dict)
        ]
        count_fields = (
            "candidate_count",
            "creator_marker_candidates",
            "creator_marker_visual_descriptions",
            "strong_unselected_candidates",
            "strong_unselected_visual_descriptions",
            "admitted_visual_scenes",
            "chat_windows",
            "chat_phrase_windows",
            "chat_clip_intents",
            "admitted_chat_packs",
        )
        totals = {
            field: sum(max(0, int(report.get(field) or 0)) for report in reports)
            for field in count_fields
        }

        def rate(numerator: str, denominator: str) -> Optional[float]:
            total = totals[denominator]
            return round(totals[numerator] / total, 4) if total else None

        return {
            "measured_jobs": len(reports),
            "chat_artifact_jobs": sum(bool(report.get("chat_artifact")) for report in reports),
            "chat_full_vod_jobs": sum(
                report.get("chat_source_scope") == "full_vod" for report in reports
            ),
            "chat_smart_region_jobs": sum(
                report.get("chat_source_scope") == "smart_regions" for report in reports
            ),
            **totals,
            "creator_marker_visual_coverage": rate(
                "creator_marker_visual_descriptions", "creator_marker_candidates",
            ),
            "strong_unselected_visual_coverage": rate(
                "strong_unselected_visual_descriptions", "strong_unselected_candidates",
            ),
            "truth_boundary": (
                "Coverage measures retained evidence only; missing evidence does not prove "
                "that no event occurred."
            ),
        }

    def semantic_state(self) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                """SELECT source_revision, indexed_revision, status, model_id,
                          dimensions, indexed_entries, index_bytes, indexed_at,
                          last_error, pending_since, last_attempt_at,
                          next_attempt_at, retry_count, last_build_kind,
                          last_encoded_entries, last_reused_entries,
                          (SELECT COUNT(*) FROM stream_memory_entries) AS source_entries
                   FROM stream_memory_semantic_state WHERE id = 1"""
            ).fetchone()
        state = dict(row) if row else {}
        file_available = os.path.isfile(self._semantic_path)
        status = str(state.get("status") or "not_built")
        if not file_available and status != "building":
            status = "error" if state.get("last_error") else "not_built"
        elif (
            file_available
            and status != "building"
            and int(state.get("indexed_revision") if state.get("indexed_revision") is not None else -1)
            != int(state.get("source_revision") if state.get("source_revision") is not None else 0)
        ):
            status = "stale"
        index_bytes = os.path.getsize(self._semantic_path) if file_available else 0
        return {
            "semantic_status": status,
            "semantic_available": file_available,
            "semantic_source_revision": int(state.get("source_revision") or 0),
            "semantic_indexed_revision": int(
                state.get("indexed_revision")
                if state.get("indexed_revision") is not None else -1
            ),
            "semantic_source_entries": int(state.get("source_entries") or 0),
            "semantic_model_id": state.get("model_id"),
            "semantic_dimensions": int(state.get("dimensions") or 0),
            "semantic_entries": int(state.get("indexed_entries") or 0),
            "semantic_index_bytes": int(index_bytes),
            "semantic_indexed_at": state.get("indexed_at"),
            "semantic_last_error": state.get("last_error"),
            "semantic_pending_since": state.get("pending_since"),
            "semantic_last_attempt_at": state.get("last_attempt_at"),
            "semantic_next_attempt_at": state.get("next_attempt_at"),
            "semantic_retry_count": int(state.get("retry_count") or 0),
            "semantic_last_build_kind": state.get("last_build_kind"),
            "semantic_last_encoded_entries": int(state.get("last_encoded_entries") or 0),
            "semantic_last_reused_entries": int(state.get("last_reused_entries") or 0),
            "semantic_model_bytes": sentence_model_bytes(),
        }

    def build_semantic_index(self, *, cancel_check=None) -> Dict[str, Any]:
        """Build a compact local concept index from existing Memory text.

        The sentence-model lane reuses unchanged vectors and atomically swaps
        the compact sidecar. An existing index remains readable if an update
        fails, is preempted by foreground work, or new sessions make it stale.
        """
        if not _SEMANTIC_BUILD_LOCK.acquire(blocking=False):
            raise RuntimeError("The Stream Memory concept index is already building.")
        try:
            with self.db.get_connection() as conn:
                state = conn.execute(
                    "SELECT source_revision FROM stream_memory_semantic_state WHERE id = 1"
                ).fetchone()
                source_revision = int(state["source_revision"] if state else 0)
                conn.execute(
                    """UPDATE stream_memory_semantic_state
                       SET status = 'building', last_error = NULL,
                           last_attempt_at = ?, next_attempt_at = NULL
                       WHERE id = 1""",
                    (_utc_now(),),
                )
                rows = conn.execute(
                    """SELECT id, job_id, kind, entry_index, start_time, end_time,
                              title, text, game
                       FROM stream_memory_entries
                       ORDER BY job_id, kind, entry_index"""
                ).fetchall()
            row_dicts = [dict(row) for row in rows]
            if cancel_check is not None and cancel_check():
                raise SemanticBuildCancelled("Stream Memory concept-index update was preempted.")
            transcript_groups: dict[str, list[dict]] = {}
            for row in row_dicts:
                if row["kind"] == "transcript":
                    transcript_groups.setdefault(row["job_id"], []).append(row)
            transcript_context: dict[str, str] = {}
            for group in transcript_groups.values():
                for index, row in enumerate(group):
                    if cancel_check is not None and index % 128 == 0 and cancel_check():
                        raise SemanticBuildCancelled(
                            "Stream Memory concept-index update was preempted."
                        )
                    # Individual ASR segments are often only a few words. A
                    # five-segment local window gives the latent index enough
                    # conversational context while every result still seeks to
                    # the exact center segment's timestamp.
                    window = group[max(0, index - 2):index + 3]
                    transcript_context[row["id"]] = " ".join(
                        value["text"] for value in window if value["text"]
                    )
            clip_context: dict[str, str] = {}
            for row_index, row in enumerate(row_dicts):
                if (
                    cancel_check is not None
                    and row_index % 64 == 0
                    and cancel_check()
                ):
                    raise SemanticBuildCancelled(
                        "Stream Memory concept-index update was preempted."
                    )
                if row["kind"] != "clip":
                    continue
                nearby = [
                    value["text"]
                    for value in transcript_groups.get(row["job_id"], [])
                    if float(value["start_time"] or 0.0) <= float(row["end_time"] or 0.0) + 8.0
                    and float(value["end_time"] or 0.0) >= float(row["start_time"] or 0.0) - 8.0
                    and value["text"]
                ]
                clip_context[row["id"]] = " ".join(nearby)
            documents = []
            for row in row_dicts:
                if row["kind"] == "transcript":
                    text = transcript_context.get(row["id"], row["text"])
                else:
                    text = " ".join(filter(None, [
                        row["title"], row["text"], row["game"],
                        clip_context.get(row["id"], ""),
                    ]))
                documents.append(SemanticDocument(entry_id=row["id"], text=text))
            if cancel_check is not None and cancel_check():
                raise SemanticBuildCancelled("Stream Memory concept-index update was preempted.")
            result = self._semantic_builder(
                documents,
                self._semantic_path,
                cancel_check=cancel_check,
            )
            legacy_paths = (
                os.path.join(self.data_root, "memory", "stream_memory_semantic.v1.npz"),
                os.path.join(self.data_root, "memory", "stream_memory_semantic.v2.npz"),
            )
            for legacy_path in legacy_paths:
                if legacy_path == self._semantic_path:
                    continue
                try:
                    os.remove(legacy_path)
                except OSError:
                    pass
            with self.db.get_connection() as conn:
                current = conn.execute(
                    "SELECT source_revision FROM stream_memory_semantic_state WHERE id = 1"
                ).fetchone()
                current_revision = int(current["source_revision"] if current else source_revision)
                status = "ready" if current_revision == source_revision else "stale"
                conn.execute(
                    """UPDATE stream_memory_semantic_state
                       SET indexed_revision = ?, status = ?, model_id = ?,
                           dimensions = ?, indexed_entries = ?, index_bytes = ?,
                           indexed_at = ?, last_error = NULL,
                           pending_since = CASE WHEN ? = 'ready' THEN NULL
                             ELSE COALESCE(pending_since, CURRENT_TIMESTAMP) END,
                           retry_count = 0, next_attempt_at = NULL,
                           last_build_kind = ?, last_encoded_entries = ?,
                           last_reused_entries = ?
                       WHERE id = 1""",
                    (
                        source_revision,
                        status,
                        self._semantic_model_id,
                        result.dimensions,
                        result.entries,
                        result.bytes_written,
                        _utc_now(),
                        status,
                        "incremental" if self._sentence_model_available else "compaction",
                        int(result.encoded_entries),
                        int(result.reused_entries),
                    ),
                )
            self._semantic_index.clear()
            return {
                "status": status,
                "entries_indexed": result.entries,
                "dimensions": result.dimensions,
                "index_bytes": result.bytes_written,
                "model_id": self._semantic_model_id,
                "build_kind": "incremental" if self._sentence_model_available else "compaction",
                "entries_encoded": int(result.encoded_entries),
                "entries_reused": int(result.reused_entries),
            }
        except SemanticBuildCancelled:
            with self.db.get_connection() as conn:
                conn.execute(
                    """UPDATE stream_memory_semantic_state
                       SET status = CASE WHEN indexed_entries > 0 THEN 'stale'
                                         ELSE 'not_built' END,
                           pending_since = COALESCE(pending_since, CURRENT_TIMESTAMP),
                           last_error = NULL, next_attempt_at = NULL
                       WHERE id = 1"""
                )
            raise
        except Exception as exc:
            retry_at = (
                datetime.now(timezone.utc) + timedelta(minutes=5)
            ).isoformat().replace("+00:00", "Z")
            with self.db.get_connection() as conn:
                conn.execute(
                    """UPDATE stream_memory_semantic_state
                       SET status = CASE WHEN indexed_entries > 0 THEN 'stale' ELSE 'error' END,
                           last_error = ?, retry_count = retry_count + 1,
                           next_attempt_at = ?,
                           pending_since = COALESCE(pending_since, CURRENT_TIMESTAMP)
                       WHERE id = 1""",
                    (str(exc)[:500], retry_at),
                )
            raise
        finally:
            _SEMANTIC_BUILD_LOCK.release()

    @staticmethod
    def _filter_clauses(
        *,
        kind: str,
        decision: str,
        exported: str,
        origin: str,
        game: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if kind in {"transcript", "clip", "evidence"}:
            clauses.append("m.kind = ?")
            params.append(kind)
        if decision in {"kept", "passed", "maybe", "unreviewed"}:
            clauses.append("m.clip_id IS NOT NULL")
            clauses.append({
                "kept": "COALESCE(c.kept, 0) = 1",
                "passed": "COALESCE(c.passed, 0) = 1",
                "maybe": "COALESCE(c.maybe, 0) = 1",
                "unreviewed": (
                    "COALESCE(c.kept, 0) = 0 AND COALESCE(c.passed, 0) = 0 "
                    "AND COALESCE(c.maybe, 0) = 0"
                ),
            }[decision])
        if exported in {"exported", "not_exported"}:
            clauses.append("m.clip_id IS NOT NULL")
            clauses.append(
                "c.exported_at IS NOT NULL"
                if exported == "exported"
                else "c.exported_at IS NULL"
            )
        if origin == "creator_marker":
            clauses.append(
                "(m.evidence_kind = 'creator_marker' "
                "OR c.recall_provenance LIKE '%\"origin\":\"creator_marker\"%')"
            )
        if game:
            clauses.append("LOWER(m.game) = LOWER(?)")
            params.append(_clean_text(game))
        if date_from:
            clauses.append("m.source_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("m.source_date <= ?")
            params.append(date_to)
        return clauses, params

    def _keyword_candidates(
        self,
        expression: str,
        filter_clauses: list[str],
        filter_params: list[Any],
        limit: int,
        *,
        prefer_clips: bool = False,
        best_first: bool = False,
        newest_first: bool = False,
    ) -> list[str]:
        if not expression:
            where = " AND ".join(filter_clauses) if filter_clauses else "1 = 1"
            order = (
                "COALESCE(c.selection_score, c.deck_score, c.score, 0) DESC, "
                if best_first else ""
            )
            sql = f"""
                SELECT m.id
                FROM stream_memory_entries AS m
                JOIN jobs AS j ON j.id = m.job_id
                LEFT JOIN clips AS c ON c.id = m.clip_id
                WHERE {where}
                ORDER BY {order}m.source_date DESC, j.created_at DESC, m.start_time
                LIMIT ?
            """  # nosec B608 - clauses are fixed templates, values stay parameterized
            with self.db.get_connection() as conn:
                return [
                    row["id"]
                    for row in conn.execute(sql, (*filter_params, limit)).fetchall()
                ]
        clauses = ["stream_memory_fts MATCH ?", *filter_clauses]
        params = [expression, *filter_params, limit]
        sql = f"""
            SELECT m.id
            FROM stream_memory_fts
            JOIN stream_memory_entries AS m ON m.rowid = stream_memory_fts.rowid
            JOIN jobs AS j ON j.id = m.job_id
            LEFT JOIN clips AS c ON c.id = m.clip_id
            WHERE {' AND '.join(clauses)}
            ORDER BY CASE WHEN ? AND m.kind = 'clip' THEN 0 ELSE 1 END,
                     CASE WHEN ? THEN COALESCE(c.selection_score, c.deck_score, c.score, 0) END DESC,
                     CASE WHEN ? THEN m.source_date END DESC,
                     CASE WHEN ? THEN j.created_at END DESC,
                     bm25(stream_memory_fts, 6.0, 1.0, 3.0, 2.0),
                     m.source_date DESC, j.created_at DESC, m.start_time
            LIMIT ?
        """
        params.insert(-1, int(prefer_clips))
        params.insert(-1, int(best_first))
        params.insert(-1, int(newest_first))
        params.insert(-1, int(newest_first))
        with self.db.get_connection() as conn:
            return [row["id"] for row in conn.execute(sql, tuple(params)).fetchall()]

    def _allowed_semantic_ids(
        self,
        filter_clauses: list[str],
        filter_params: list[Any],
    ) -> Optional[set[str]]:
        if not filter_clauses:
            return None
        sql = f"""
            SELECT m.id
            FROM stream_memory_entries AS m
            LEFT JOIN clips AS c ON c.id = m.clip_id
            WHERE {' AND '.join(filter_clauses)}
        """
        with self.db.get_connection() as conn:
            return {
                row["id"] for row in conn.execute(sql, tuple(filter_params)).fetchall()
            }

    def _clip_bridges(
        self,
        transcript_ids: list[str],
        clip_filter_clauses: list[str],
        clip_filter_params: list[Any],
    ) -> list[Dict[str, Any]]:
        """Map spoken evidence onto the clip that contains or closely follows it."""
        if not transcript_ids:
            return []
        placeholders = ",".join("?" for _ in transcript_ids)
        clauses = [
            "t.id IN (" + placeholders + ")",
            "t.kind = 'transcript'",
            "m.kind = 'clip'",
            "m.start_time <= t.end_time + 18.0",
            "m.end_time >= t.start_time - 18.0",
            *clip_filter_clauses,
        ]
        sql = f"""
            SELECT t.id AS transcript_id, t.text AS transcript_text,
                   m.id AS clip_entry_id,
                   CASE
                     WHEN t.end_time < m.start_time THEN m.start_time - t.end_time
                     WHEN t.start_time > m.end_time THEN t.start_time - m.end_time
                     ELSE 0.0
                   END AS distance_seconds
            FROM stream_memory_entries AS t
            JOIN stream_memory_entries AS m
              ON m.job_id = t.job_id AND m.kind = 'clip'
            JOIN clips AS c ON c.id = m.clip_id
            WHERE {' AND '.join(clauses)}
            ORDER BY distance_seconds, m.start_time
        """
        params = [*transcript_ids, *clip_filter_params]
        with self.db.get_connection() as conn:
            return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]

    @staticmethod
    def _row_matches_filters(
        row: Dict[str, Any],
        *,
        kind: str,
        decision: str,
        exported: str,
        origin: str,
    ) -> bool:
        if kind in {"transcript", "clip", "evidence"} and row.get("kind") != kind:
            return False
        if decision != "all":
            if row.get("kind") != "clip":
                return False
            if decision == "kept" and not bool(row.get("kept")):
                return False
            if decision == "passed" and not bool(row.get("passed")):
                return False
            if decision == "maybe" and not bool(row.get("maybe")):
                return False
            if decision == "unreviewed" and any(
                (row.get("kept"), row.get("passed"), row.get("maybe"))
            ):
                return False
        if exported == "exported" and not row.get("exported_at"):
            return False
        if exported == "not_exported" and (
            row.get("kind") != "clip" or row.get("exported_at")
        ):
            return False
        if origin == "creator_marker":
            provenance = _json_value(row.get("recall_provenance"), {})
            if (
                row.get("evidence_kind") != "creator_marker"
                and provenance.get("origin") != "creator_marker"
            ):
                return False
        return True

    def _result_rows(self, entry_ids: list[str]) -> dict[str, Dict[str, Any]]:
        if not entry_ids:
            return {}
        placeholders = ",".join("?" for _ in entry_ids)
        sql = f"""
            SELECT m.id, m.job_id, m.clip_id, m.kind, m.entry_index,
                   m.start_time, m.end_time,
                   m.title, m.text, m.session_name, m.game, m.source_date,
                   m.evidence_kind, m.evidence_json,
                   CASE WHEN m.kind = 'transcript' THEN (
                     SELECT GROUP_CONCAT(context.text, ' ')
                     FROM (
                       SELECT nearby.text
                       FROM stream_memory_entries AS nearby
                       WHERE nearby.job_id = m.job_id
                         AND nearby.kind = 'transcript'
                         AND nearby.entry_index BETWEEN m.entry_index - 2 AND m.entry_index + 2
                       ORDER BY nearby.entry_index
                     ) AS context
                   ) ELSE m.text END AS match_context,
                   j.source_type, j.source_path, j.asset_path, j.duration,
                   j.created_at AS job_created_at,
                   c.kept, c.passed, c.maybe, c.exported_at,
                   c.selection_score, c.recall_provenance, c.moment_type, c.scene,
                   s.transcript_coverage
            FROM stream_memory_entries AS m
            JOIN jobs AS j ON j.id = m.job_id
            LEFT JOIN clips AS c ON c.id = m.clip_id
            LEFT JOIN stream_memory_index_state AS s ON s.job_id = m.job_id
            WHERE m.id IN ({placeholders})
        """  # nosec B608 - placeholders are generated, values stay parameterized
        with self.db.get_connection() as conn:
            rows = [dict(row) for row in conn.execute(sql, tuple(entry_ids)).fetchall()]
        return {row["id"]: row for row in rows}

    def action_context(self, entry_id: str) -> Dict[str, Any]:
        """Resolve one indexed moment for a creator action without re-searching it.

        The row id is only a lookup handle. Durable clip provenance also records
        the content fingerprint and source interval so an index rebuild cannot
        silently change what the action referred to.
        """
        cleaned_id = _clean_text(entry_id)
        if not cleaned_id:
            raise ValueError("Memory entry id cannot be empty.")
        row = self._result_rows([cleaned_id]).get(cleaned_id)
        if row is None:
            raise KeyError("That Memory moment is no longer in the local index.")
        source_candidates = [row.get("asset_path"), row.get("source_path")]
        return {
            "entry_id": str(row["id"]),
            "entry_fingerprint": _entry_fingerprint(row),
            "job_id": str(row["job_id"]),
            "clip_id": row.get("clip_id"),
            "kind": str(row.get("kind") or ""),
            "evidence_kind": row.get("evidence_kind"),
            "start_time": float(row.get("start_time") or 0.0),
            "end_time": float(row.get("end_time") or row.get("start_time") or 0.0),
            "title": _clean_text(row.get("title")),
            "source_available": any(
                path and os.path.isfile(str(path)) for path in source_candidates
            ),
        }

    @staticmethod
    def _match_source(
        row: Dict[str, Any],
        query_terms: set[str],
        lanes: set[str],
        *,
        spoken_bridge: bool = False,
    ) -> str:
        """Explain the strongest persisted field supporting one result."""
        evidence = _json_value(row.get("evidence_json"), {})
        provenance = _json_value(row.get("recall_provenance"), {})
        if (
            row.get("evidence_kind") == "creator_marker"
            or provenance.get("origin") == "creator_marker"
        ):
            return "creator_marker"
        if spoken_bridge or row.get("kind") == "transcript":
            return "creator_speech"

        evidence_text = _clean_text(evidence.get("search_text"))
        core_text = _clean_text(row.get("text"))
        if evidence_text and core_text.endswith(evidence_text):
            core_text = core_text[:-len(evidence_text)].strip()
        if row.get("kind") == "clip" and _text_matches_terms(
            f"{row.get('title', '')} {core_text}", query_terms,
        ):
            return "clip_details"

        chat = evidence.get("chat_activity") or {}
        chat_text = " ".join(
            _clean_text(item.get("text"))
            for item in (chat.get("repeated_phrases") or [])
            if isinstance(item, dict)
        )
        chat_text = _clean_text(
            f"chat burst {chat_text} "
            f"{'viewer clip request' if chat.get('clip_intent_count') else ''} "
            f"{'chat emote burst' if chat.get('emote_count') else ''}"
        )
        if chat and _text_matches_terms(chat_text, query_terms):
            return "chat_activity"

        structured_text = " ".join(
            _clean_text(value)
            for key in ("labels", "gameplay_events", "reactions")
            for value in (evidence.get(key) or [])
        )
        if _text_matches_terms(structured_text, query_terms):
            return "gameplay_event"
        visual_text = " ".join(
            _clean_text(value)
            for scene in (evidence.get("visual_scenes") or [])
            if isinstance(scene, dict)
            for value in (scene.get("description"), scene.get("support"))
            if value
        )
        if _text_matches_terms(visual_text, query_terms):
            return "visual_description"
        ocr_text = " ".join(
            _clean_text(value) for value in (evidence.get("ocr_phrases") or [])
        )
        if _text_matches_terms(ocr_text, query_terms):
            return "on_screen_text"
        if "keyword" in lanes and row.get("kind") == "clip":
            return "clip_details"
        return "related_meaning"

    @staticmethod
    def _ranking_bonus(
        row: Dict[str, Any],
        match_source: str,
        *,
        prefer_clips: bool,
    ) -> float:
        """Use creator-owned provenance as a small, explainable reranking prior."""
        bonus = {
            "creator_marker": 0.0100,
            "creator_speech": 0.0080,
            "clip_details": 0.0060,
            "gameplay_event": 0.0050,
            "chat_activity": 0.0030,
            "visual_description": 0.0020,
            "on_screen_text": 0.0010,
            "related_meaning": 0.0000,
        }[match_source]
        if row.get("kind") == "clip":
            if row.get("kept"):
                bonus += 0.0060
            elif row.get("maybe"):
                bonus += 0.0015
            elif row.get("passed"):
                bonus -= 0.0030
            if prefer_clips:
                bonus += 0.0040
        if match_source == "on_screen_text":
            # OCR remains searchable, but a lone overlay/chat-shaped token must
            # not outrank creator speech or structured gameplay evidence.
            bonus -= 0.0025
        return bonus

    @staticmethod
    def _suppress_duplicate_source_moments(
        ranked_ids: list[str],
        row_map: dict[str, Dict[str, Any]],
    ) -> list[str]:
        """Collapse only cross-job copies of the same Twitch moment."""
        accepted: list[str] = []
        seen: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
        for entry_id in ranked_ids:
            row = row_map[entry_id]
            source_identity = _source_video_identity(row.get("source_path"))
            if not source_identity:
                accepted.append(entry_id)
                continue
            key = (source_identity, str(row.get("kind") or ""))
            start = float(row.get("start_time") or 0.0)
            end = float(row.get("end_time") or start)
            duplicate = any(
                prior_job != str(row.get("job_id"))
                and abs(prior_start - start) <= 5.0
                and abs(prior_end - end) <= 8.0
                for prior_job, prior_start, prior_end in seen.get(key, [])
            )
            if duplicate:
                continue
            seen.setdefault(key, []).append((str(row.get("job_id")), start, end))
            accepted.append(entry_id)
        return accepted

    def search(
        self,
        query: str,
        *,
        kind: str = "all",
        decision: str = "all",
        exported: str = "all",
        origin: str = "all",
        game: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 40,
        mode: str = "hybrid",
        apply_creator_feedback: bool = True,
        track_interactions: bool = False,
    ) -> Dict[str, Any]:
        limit = max(1, min(MAX_RESULTS, int(limit)))
        if mode not in {"hybrid", "keyword"}:
            raise ValueError("Search mode must be hybrid or keyword.")
        if kind not in {"all", "transcript", "clip", "evidence"}:
            raise ValueError("Search kind must be all, transcript, clip, or evidence.")
        if decision not in {"all", "kept", "passed", "maybe", "unreviewed"}:
            raise ValueError("Unknown review filter.")
        if exported not in {"all", "exported", "not_exported"}:
            raise ValueError("Unknown export filter.")
        if origin not in {"all", "creator_marker"}:
            raise ValueError("Unknown Memory origin filter.")
        plan = _plan_search_query(
            query,
            kind=kind,
            decision=decision,
            exported=exported,
            origin=origin,
            date_from=date_from,
            date_to=date_to,
        )
        compound_plan = _plan_compound_query(
            plan["query"],
            include_marker=plan["origin"] == "creator_marker",
        )
        kind = plan["kind"]
        decision = plan["decision"]
        exported = plan["exported"]
        origin = plan["origin"]
        date_from = plan["date_from"]
        date_to = plan["date_to"]
        allow_empty = bool(
            plan["interpreted_filters"]
            or kind != "all"
            or decision != "all"
            or exported != "all"
            or origin != "all"
            or game
            or date_from
            or date_to
        )
        expression, semantic_query, alias_expansions = self._query_with_aliases(
            plan["query"], allow_empty=allow_empty,
        )
        compound_expression = ""
        if compound_plan["active"]:
            residual_expression, _, _ = self._query_with_aliases(
                compound_plan["residual_query"],
                allow_empty=True,
            )
            compound_expression = _compound_fts_expression(
                compound_plan,
                residual_expression,
            )
        # Search all evidence first. Transcript matches can then surface the
        # actual clip that owns the spoken moment before result filters apply.
        evidence_filter_clauses, evidence_filter_params = self._filter_clauses(
            kind="all",
            decision="all",
            exported=exported,
            origin=origin,
            game=game,
            date_from=date_from,
            date_to=date_to,
        )
        transcript_filter_clauses, transcript_filter_params = self._filter_clauses(
            kind="transcript",
            decision="all",
            exported=exported,
            origin=origin,
            game=game,
            date_from=date_from,
            date_to=date_to,
        )
        clip_filter_clauses, clip_filter_params = self._filter_clauses(
            kind="clip",
            decision=decision,
            exported=exported,
            origin=origin,
            game=game,
            date_from=date_from,
            date_to=date_to,
        )
        candidate_limit = max(200, limit * 5)
        prefer_clips = _prefers_clip_evidence(plan["query"]) or kind == "clip"
        keyword_ids = self._keyword_candidates(
            expression, evidence_filter_clauses, evidence_filter_params, candidate_limit,
            prefer_clips=prefer_clips,
            best_first=plan["best_first"],
        )
        keyword_clip_ids = [] if kind in {"transcript", "evidence"} else self._keyword_candidates(
            expression, clip_filter_clauses, clip_filter_params, candidate_limit,
            prefer_clips=True,
            best_first=plan["best_first"],
        )
        compound_ids = [] if not compound_expression else self._keyword_candidates(
            compound_expression,
            evidence_filter_clauses,
            evidence_filter_params,
            candidate_limit,
            prefer_clips=prefer_clips,
            best_first=plan["best_first"],
        )
        compound_clip_ids = [] if (
            not compound_expression or kind in {"transcript", "evidence"}
        ) else self._keyword_candidates(
            compound_expression,
            clip_filter_clauses,
            clip_filter_params,
            candidate_limit,
            prefer_clips=True,
            best_first=plan["best_first"],
        )
        concept_ids: list[str] = []
        concept_clip_ids: list[str] = []
        concept_recent_ids: list[str] = []
        concept_recent_clip_ids: list[str] = []
        concept_expression = (
            _creator_concept_query(semantic_query)
            if mode == "hybrid" and semantic_query
            else None
        )
        if concept_expression:
            concept_ids = self._keyword_candidates(
                concept_expression, evidence_filter_clauses, evidence_filter_params, candidate_limit,
                prefer_clips=prefer_clips,
                best_first=plan["best_first"],
            )
            if _is_aura_maxxing_query(semantic_query):
                concept_recent_ids = self._keyword_candidates(
                    concept_expression, transcript_filter_clauses, transcript_filter_params,
                    candidate_limit, prefer_clips=prefer_clips, newest_first=True,
                    best_first=plan["best_first"],
                )
            if kind not in {"transcript", "evidence"}:
                concept_clip_ids = self._keyword_candidates(
                    concept_expression, clip_filter_clauses, clip_filter_params,
                    candidate_limit, prefer_clips=True, best_first=plan["best_first"],
                )
                if _is_aura_maxxing_query(semantic_query):
                    concept_recent_clip_ids = self._keyword_candidates(
                        concept_expression, clip_filter_clauses, clip_filter_params,
                        candidate_limit, prefer_clips=True, best_first=plan["best_first"],
                        newest_first=True,
                    )
        semantic_state = self.semantic_state()
        semantic_results: list[tuple[str, float]] = []
        semantic_clip_results: list[tuple[str, float]] = []
        latent_enabled = bool(
            mode == "hybrid" and semantic_query and semantic_state["semantic_available"]
        )
        if latent_enabled:
            try:
                semantic_results = self._semantic_index.search(
                    semantic_query,
                    allowed_ids=self._allowed_semantic_ids(
                        evidence_filter_clauses, evidence_filter_params,
                    ),
                    limit=candidate_limit,
                )
                if kind not in {"transcript", "evidence"}:
                    semantic_clip_results = self._semantic_index.search(
                        semantic_query,
                        allowed_ids=self._allowed_semantic_ids(
                            clip_filter_clauses, clip_filter_params,
                        ),
                        limit=candidate_limit,
                    )
            except Exception as exc:  # keyword search must remain dependable
                latent_enabled = False
                semantic_state["semantic_last_error"] = str(exc)[:500]

        fused: dict[str, float] = {}
        lanes: dict[str, set[str]] = {}
        semantic_scores: dict[str, float] = {}
        for rank, entry_id in enumerate(keyword_ids, start=1):
            fused[entry_id] = fused.get(entry_id, 0.0) + 1.40 / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("keyword")
        for rank, entry_id in enumerate(keyword_clip_ids, start=1):
            fused[entry_id] = fused.get(entry_id, 0.0) + 1.65 / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("keyword")
        for rank, entry_id in enumerate(compound_ids, start=1):
            fused[entry_id] = fused.get(entry_id, 0.0) + 1.85 / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("compound")
        for rank, entry_id in enumerate(compound_clip_ids, start=1):
            fused[entry_id] = fused.get(entry_id, 0.0) + 2.00 / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("compound")
        for rank, entry_id in enumerate(concept_ids, start=1):
            fused[entry_id] = fused.get(entry_id, 0.0) + 1.05 / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("semantic")
        for rank, entry_id in enumerate(concept_clip_ids, start=1):
            fused[entry_id] = fused.get(entry_id, 0.0) + 1.20 / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("semantic")
        for rank, entry_id in enumerate(concept_recent_ids, start=1):
            # Preserve one strong freshness sentinel for natural concepts so a
            # newly indexed matching session cannot sit beyond the first UI
            # batch merely because older, shorter phrases win BM25 length ties.
            recent_weight = 1.60 if rank == 1 else 0.60
            fused[entry_id] = fused.get(entry_id, 0.0) + recent_weight / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("semantic")
        for rank, entry_id in enumerate(concept_recent_clip_ids, start=1):
            recent_weight = 1.70 if rank == 1 else 0.70
            fused[entry_id] = fused.get(entry_id, 0.0) + recent_weight / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("semantic")
        accepted_semantic_results = []
        controlled_ids = (
            set(keyword_ids) | set(keyword_clip_ids)
            | set(compound_ids) | set(compound_clip_ids)
            | set(concept_ids) | set(concept_clip_ids)
            | set(concept_recent_ids) | set(concept_recent_clip_ids)
        )
        for entry_id, score in semantic_results:
            # Once a query maps to a deterministic creator concept, the latent
            # lane may strengthen those candidates but may not introduce rows
            # that fail the controlled concept coverage requirement.
            if concept_expression and entry_id not in controlled_ids:
                continue
            accepted_semantic_results.append((entry_id, score))
        for rank, (entry_id, score) in enumerate(accepted_semantic_results, start=1):
            fused[entry_id] = fused.get(entry_id, 0.0) + 0.55 / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("semantic")
            semantic_scores[entry_id] = score
        for rank, (entry_id, score) in enumerate(semantic_clip_results, start=1):
            if concept_expression and entry_id not in controlled_ids:
                continue
            fused[entry_id] = fused.get(entry_id, 0.0) + 0.70 / (60.0 + rank)
            lanes.setdefault(entry_id, set()).add("semantic")
            semantic_scores[entry_id] = max(semantic_scores.get(entry_id, score), score)

        clip_contexts: dict[str, tuple[float, str]] = {}
        meaningful_query_terms = [
            token for token in _TOKEN_RE.findall(_clean_text(plan["query"]).lower())
            if token not in _STOP_WORDS
        ]
        if kind not in {"transcript", "evidence"} and (
            concept_expression or len(meaningful_query_terms) >= 2
        ):
            # Only deterministic word/concept hits may assert that a phrase was
            # spoken during a clip. Latent similarity alone is too weak to make
            # that provenance claim.
            transcript_candidate_ids = (
                set(keyword_ids) | set(concept_ids) | set(concept_recent_ids)
            )
            transcript_ids = [
                entry_id for entry_id in fused
                if entry_id in transcript_candidate_ids
            ]
            for bridge in self._clip_bridges(
                transcript_ids,
                clip_filter_clauses,
                clip_filter_params,
            ):
                transcript_id = str(bridge["transcript_id"])
                clip_entry_id = str(bridge["clip_entry_id"])
                distance = float(bridge["distance_seconds"] or 0.0)
                bridge_score = fused.get(transcript_id, 0.0) * 1.18 + 0.0025 / (1.0 + distance)
                if bridge_score <= fused.get(clip_entry_id, 0.0):
                    continue
                fused[clip_entry_id] = bridge_score
                lanes.setdefault(clip_entry_id, set()).update(lanes.get(transcript_id, {"keyword"}))
                semantic_score = semantic_scores.get(transcript_id)
                if semantic_score is not None:
                    semantic_scores[clip_entry_id] = max(
                        semantic_scores.get(clip_entry_id, semantic_score), semantic_score,
                    )
                current = clip_contexts.get(clip_entry_id)
                if current is None or bridge_score > current[0]:
                    clip_contexts[clip_entry_id] = (
                        bridge_score,
                        _clean_text(bridge["transcript_text"]),
                    )
        base_ranked_ids = sorted(fused, key=fused.get, reverse=True)[:candidate_limit]
        row_map = self._result_rows(base_ranked_ids)
        eligible_ids = [
            entry_id for entry_id in base_ranked_ids
            if entry_id in row_map and self._row_matches_filters(
                row_map[entry_id],
                kind=kind,
                decision=decision,
                exported=exported,
                origin=origin,
            )
        ]
        compound_match_map: dict[str, list[dict[str, str]]] = {}
        if compound_plan["active"]:
            for entry_id in eligible_ids:
                matches = _compound_matches(
                    row_map[entry_id],
                    compound_plan["clauses"],
                )
                if len(matches) == len(compound_plan["clauses"]):
                    compound_match_map[entry_id] = matches
            # A recognized compound request fails closed: latent similarity or
            # one exciting signal may not satisfy only part of the remembered moment.
            eligible_ids = [
                entry_id for entry_id in eligible_ids
                if entry_id in compound_match_map
            ]
        query_terms = _query_match_terms(plan["query"])
        match_sources = {
            entry_id: self._match_source(
                row_map[entry_id],
                query_terms,
                lanes.get(entry_id, {"keyword"}),
                spoken_bridge=entry_id in clip_contexts,
            )
            for entry_id in eligible_ids
        }
        ranking_scores = {
            entry_id: fused[entry_id] + self._ranking_bonus(
                row_map[entry_id],
                match_sources[entry_id],
                prefer_clips=prefer_clips,
            ) + 0.0030 * len(compound_match_map.get(entry_id, []))
            for entry_id in eligible_ids
        }
        ranked_ids = sorted(
            eligible_ids,
            key=lambda entry_id: (
                ranking_scores[entry_id],
                _clean_text(row_map[entry_id].get("job_created_at")),
            ),
            reverse=True,
        )
        ranked_ids = self._suppress_duplicate_source_moments(ranked_ids, row_map)
        feedback = self._feedback_for_query(query) if apply_creator_feedback else {}
        ranked_ids = [
            entry_id for entry_id in ranked_ids
            if feedback.get(entry_id) != "not_relevant"
        ]
        ranked_ids.sort(key=lambda entry_id: feedback.get(entry_id) != "relevant")
        rows = [row_map[entry_id] for entry_id in ranked_ids[:limit]]
        for row in rows:
            source_candidates = [row.pop("asset_path", None), row.get("source_path")]
            row.pop("job_created_at", None)
            row["source_available"] = any(
                path and os.path.isfile(path) for path in source_candidates
            )
            row["kept"] = bool(row.get("kept"))
            row["passed"] = bool(row.get("passed"))
            row["maybe"] = bool(row.get("maybe"))
            row["exported"] = bool(row.get("exported_at"))
            row["review_state"] = (
                "kept" if row["kept"] else
                "passed" if row["passed"] else
                "maybe" if row["maybe"] else
                "unreviewed" if row.get("kind") == "clip" else None
            )
            evidence = _json_value(row.pop("evidence_json", None), {})
            row["evidence"] = evidence
            recall_provenance = _json_value(row.pop("recall_provenance", None), {})
            row["origin"] = (
                "creator_marker"
                if row.get("evidence_kind") == "creator_marker"
                or recall_provenance.get("origin") == "creator_marker"
                else "scan_evidence" if row.get("evidence_kind") else
                "clip" if row.get("kind") == "clip" else
                "transcript"
            )
            lane = lanes.get(row["id"], {"keyword"})
            row["match_type"] = "hybrid" if len(lane) > 1 else next(iter(lane))
            row["match_source"] = match_sources[row["id"]]
            row["match_reason"] = {
                "creator_marker": "Marked live",
                "creator_speech": "Creator speech",
                "clip_details": "Clip details",
                "gameplay_event": "Gameplay or reaction signal",
                "chat_activity": "Twitch chat activity",
                "visual_description": "Visual scene description",
                "on_screen_text": "On-screen text",
                "related_meaning": "Related meaning",
            }[row["match_source"]]
            if row["id"] in clip_contexts:
                row["match_context"] = clip_contexts[row["id"]][1]
                row["match_reason"] = "Spoken during this clip"
            matched_evidence = compound_match_map.get(row["id"], [])
            row["matched_evidence"] = matched_evidence
            row["compound_coverage"] = {
                "matched": len(matched_evidence),
                "required": len(compound_plan["clauses"]),
                "ratio": round(
                    len(matched_evidence) / max(1, len(compound_plan["clauses"])),
                    4,
                ),
            } if compound_plan["active"] else None
            if matched_evidence:
                row["match_reason"] = f"{len(matched_evidence)} clues matched together"
            row["semantic_score"] = round(semantic_scores[row["id"]], 4) if row["id"] in semantic_scores else None
            row["creator_feedback"] = feedback.get(row["id"])
        resolved_filters = {
            "kind": kind,
            "decision": decision,
            "exported": exported,
            "origin": origin,
            "game": _clean_text(game) or None,
            "date_from": date_from,
            "date_to": date_to,
        }
        search_id = (
            self._record_search_session(query, resolved_filters, rows)
            if track_interactions else None
        )
        for row in rows:
            row.pop("entry_index", None)
            row.pop("moment_type", None)
            row.pop("scene", None)
        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "search_id": search_id,
            "query": _clean_text(query),
            "effective_query": plan["query"],
            "count": len(rows),
            "mode_requested": mode,
            "mode_used": "hybrid" if mode == "hybrid" and (
                concept_ids or accepted_semantic_results
            ) else "keyword",
            "semantic_status": semantic_state["semantic_status"],
            "alias_expansions": alias_expansions,
            "interpreted_filters": plan["interpreted_filters"],
            "resolved_filters": resolved_filters,
            "compound_query": {
                "active": bool(compound_plan["active"]),
                "clauses": [
                    {"id": clause["id"], "label": clause["label"]}
                    for clause in compound_plan["clauses"]
                ],
                "residual_query": compound_plan["residual_query"],
            },
            "results": rows,
        }
