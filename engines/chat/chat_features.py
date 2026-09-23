# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Twitch-chat reaction channel (plan 9.1).

Turns a VOD's raw chat-replay messages into a per-second intensity track that
the reaction fusion consumes as another modality. Chat velocity (messages per
second) and emote/hype spam are among the strongest crowd-reaction signals on
Twitch and are far cheaper to compute than OCR or emotion, so this meaningfully
sharpens clip selection when a Twitch VOD's chat is available.

This module is dependency-free and deterministic: it takes already-fetched
messages (see twitch_chat.py for the optional downloader) and returns a
``list[ChatFrame]`` aligned to 1-second buckets. Absent chat -> empty list ->
the fusion drops the channel and renormalizes over the remaining modalities.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Sequence


CHAT_EVIDENCE_VERSION = 1
CHAT_EVIDENCE_WINDOW_SECONDS = 15.0
MAX_CHAT_EVIDENCE_WINDOWS = 160
MAX_REPEATED_PHRASES = 5


@dataclass
class ChatMessage:
    """One chat-replay message. ``emotes`` is a count of emotes in the message
    (0 when unknown); ``text`` is the message body used for a hype rule."""
    timestamp: float
    text: str = ""
    emotes: int = 0


@dataclass
class ChatFrame:
    """Per-second chat intensity sample, consumed by fusion.build_curve."""
    timestamp: float
    chat: float = 0.0        # raw intensity (rate + emote/hype weighting)
    messages: int = 0        # message count in this second (for evidence/UI)
    clip_intents: int = 0    # clip-command messages ("!clip", "clip it") this second


@dataclass
class ClipBurst:
    """A cluster of clip-command messages. ``time`` is the FIRST command's
    second — the anchor. Viewers type "!clip" seconds AFTER the moment (Twitch's
    own command captures the trailing ~30s), so consumers must look BACKWARD
    from ``time``, never forward."""
    time: float
    count: int


# Explicit clip-intent phrases only. Deliberately narrower than the "clip"
# hype token below: a viewer asking for a clip is a direct crowd vote that a
# postable moment just happened, and it anchors retroactive candidate
# injection (reaction_main), so precision matters more than recall here.
# Bare "clip" / "nice clip" / "clipping" stay hype-only.
_CLIP_INTENT_RE = re.compile(
    r"(?:^|\s)!clip(?!\w)"                      # bot command: !clip (not !clips)
    r"|\bclip\s*(?:it|that|this)\b"             # clip it / clipthat / clip this
    r"|\b(?:some(?:one|body)|mods?)\s+clip\b"   # someone clip / mods clip
    r"|\bclip\s+(?:pls|plz|please)\b",          # clip pls
    re.IGNORECASE,
)


def is_clip_intent(text: str) -> bool:
    """True when a chat message explicitly asks for the moment to be clipped."""
    return bool(_CLIP_INTENT_RE.search(text or ""))


def cluster_clip_bursts(intent_seconds, merge_gap: float = 12.0) -> List[ClipBurst]:
    """Cluster (timestamp, count) clip-intent seconds into ``ClipBurst``s.

    Commands within ``merge_gap`` seconds of the previous one belong to the
    same burst (viewers pile on for a few seconds after a moment). The burst
    anchor is the FIRST command — the closest one to the moment itself.
    """
    pts = sorted((float(t), int(n)) for t, n in intent_seconds if n > 0)
    bursts: List[ClipBurst] = []
    last_t: float = 0.0
    for t, n in pts:
        if bursts and (t - last_t) <= merge_gap:
            bursts[-1].count += n
        else:
            bursts.append(ClipBurst(time=t, count=n))
        last_t = t
    return bursts


# Hype tokens that spike during big moments. Case-insensitive, word-ish match.
_HYPE_TOKENS = (
    "pog", "poggers", "pogchamp", "letsgo", "lets go", "lfg", "ez", "clip",
    "clipit", "clip it", "insane", "omg", "holy", "no way", "noway", "gg",
    "w", "dub", "sheesh", "actual", "kekw", "lulw", "lmao", "lmfao",
)
_HYPE_RE = re.compile(r"|".join(re.escape(t) for t in _HYPE_TOKENS), re.IGNORECASE)
# Repeated caps/emote spam like "AAAA" or "!!!!" also reads as hype.
_SPAM_RE = re.compile(r"(.)\1{3,}")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,25}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d ().-]{7,}\d)(?!\d)")


def _message_weight(msg: ChatMessage) -> float:
    """Base 1 per message, plus emote and hype/spam bonuses. A single message
    can't dominate a whole second — the per-second sum is what matters."""
    weight = 1.0
    weight += 0.6 * float(max(0, msg.emotes))
    text = msg.text or ""
    if _HYPE_RE.search(text):
        weight += 0.8
    if _SPAM_RE.search(text):
        weight += 0.5
    return weight


def build_chat_track(messages: Sequence[ChatMessage], duration: float) -> List[ChatFrame]:
    """Bucket messages into 1-second ``ChatFrame``s over ``[0, duration]``.

    ``chat`` is the summed per-message weight in that second; the fusion layer
    smooths and per-VOD robust-z normalizes it, so absolute scale is irrelevant
    here — only the relative shape (where chat spikes) matters.
    """
    if not messages or duration <= 0:
        return []

    n = int(math.floor(duration)) + 1
    weights = [0.0] * n
    counts = [0] * n
    intents = [0] * n
    for msg in messages:
        t = msg.timestamp
        if t is None or t < 0 or t >= n:
            continue
        idx = int(t)
        weights[idx] += _message_weight(msg)
        counts[idx] += 1
        if is_clip_intent(msg.text):
            intents[idx] += 1

    if not any(counts):
        return []
    return [
        ChatFrame(timestamp=float(i), chat=weights[i], messages=counts[i],
                  clip_intents=intents[i])
        for i in range(n)
    ]


def _safe_repeated_phrase(text: str) -> str:
    """Return a bounded anonymous phrase, or empty when it should not persist.

    Chat evidence deliberately does not retain unique messages. This sanitizer
    also removes links, contact-shaped strings, and @mentions before the
    repeated-phrase threshold is applied. Explicit clip commands become a
    standardized label rather than storing the viewer's message verbatim.
    """
    value = " ".join(str(text or "").replace("\x00", " ").split()).strip()
    if not value:
        return ""
    if is_clip_intent(value):
        return "viewer clip request"
    value = _URL_RE.sub(" ", value)
    value = _EMAIL_RE.sub(" ", value)
    value = _MENTION_RE.sub(" ", value)
    value = _PHONE_RE.sub(" ", value)
    value = " ".join(value.split()).strip(" -_:|/\\.,;!?[](){}")
    if not value or len(value) > 64:
        return ""
    return value


def build_chat_evidence(
    messages: Sequence[ChatMessage],
    duration: float,
    *,
    source_scope: str = "full_vod",
) -> dict[str, Any]:
    """Summarize anonymous replay chat into compact, searchable burst windows.

    The returned payload is safe to persist: it contains no authors and never
    retains a message seen only once. Repeated short phrases, aggregate counts,
    emote volume, and standardized clip intent are enough to explain searches
    such as ``aura maxxing`` or ``when chat spammed W`` without creating a raw
    chat archive. Windows are globally bounded by strength for large VODs.
    """
    if not messages or duration <= 0:
        return {
            "version": CHAT_EVIDENCE_VERSION,
            "source_scope": source_scope,
            "window_seconds": CHAT_EVIDENCE_WINDOW_SECONDS,
            "windows": [],
            "privacy": {
                "authors_retained": False,
                "unique_messages_retained": False,
                "repeated_phrases_only": True,
            },
        }

    buckets: dict[int, dict[str, Any]] = {}
    for message in messages:
        try:
            timestamp = float(message.timestamp)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp > duration:
            continue
        bucket_index = int(timestamp // CHAT_EVIDENCE_WINDOW_SECONDS)
        bucket = buckets.setdefault(bucket_index, {
            "messages": 0,
            "emotes": 0,
            "clip_intents": 0,
            "intensity": 0.0,
            "phrases": Counter(),
            "display": {},
        })
        bucket["messages"] += 1
        bucket["emotes"] += max(0, int(message.emotes or 0))
        bucket["clip_intents"] += int(is_clip_intent(message.text))
        bucket["intensity"] += _message_weight(message)
        phrase = _safe_repeated_phrase(message.text)
        if phrase:
            key = phrase.casefold()
            bucket["phrases"][key] += 1
            bucket["display"].setdefault(key, phrase)

    windows: list[dict[str, Any]] = []
    for bucket_index, bucket in buckets.items():
        repeated = [
            {"text": bucket["display"][key], "count": count}
            for key, count in sorted(
                bucket["phrases"].items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= 2
        ][:MAX_REPEATED_PHRASES]
        message_count = int(bucket["messages"])
        emote_count = int(bucket["emotes"])
        clip_intent_count = int(bucket["clip_intents"])
        # Retain strong aggregate activity even when no phrase repeated, but
        # never retain its unique message bodies.
        if not repeated and clip_intent_count == 0 and message_count < 8 and emote_count < 6:
            continue
        start = bucket_index * CHAT_EVIDENCE_WINDOW_SECONDS
        end = min(float(duration), start + CHAT_EVIDENCE_WINDOW_SECONDS)
        repeat_votes = sum(row["count"] for row in repeated)
        strength = (
            float(bucket["intensity"])
            + repeat_votes * 1.5
            + clip_intent_count * 3.0
        )
        windows.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "message_count": message_count,
            "emote_count": emote_count,
            "clip_intent_count": clip_intent_count,
            "repeated_phrases": repeated,
            "strength": round(strength, 4),
        })

    windows.sort(key=lambda row: (-row["strength"], row["start"]))
    windows = windows[:MAX_CHAT_EVIDENCE_WINDOWS]
    windows.sort(key=lambda row: row["start"])
    return {
        "version": CHAT_EVIDENCE_VERSION,
        "source_scope": source_scope,
        "window_seconds": CHAT_EVIDENCE_WINDOW_SECONDS,
        "windows": windows,
        "privacy": {
            "authors_retained": False,
            "unique_messages_retained": False,
            "repeated_phrases_only": True,
        },
    }
