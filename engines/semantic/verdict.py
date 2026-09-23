# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Semantic verdict for one candidate clip (HUMAN_CLIPS plan, Package C).

One dataclass shared by the backend (which constrains generation to this
shape), the judge (which attaches it to candidates), and tests (which script
it). Keeping the parse/clamp logic here means every producer of a verdict —
real model or fake — goes through the same normalization, so downstream
selection code never sees an out-of-range score or an unknown label.
"""

from dataclasses import dataclass, asdict
from typing import Optional

# Closed vocabulary: selection treats "filler" as the reject class, everything
# else as a real moment kind. Keep in sync with the JSON schema in backend.py.
MOMENT_TYPES = (
    "funny", "clutch", "fail", "rage", "scare", "story", "wholesome", "filler",
)
VERDICTS = ("post", "maybe", "skip")

TITLE_MAX_WORDS = 8
HOOK_LINE_MAX_WORDS = 7


def _clamp01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _cap_words(text, max_words: int) -> str:
    words = str(text or "").split()
    return " ".join(words[:max_words]).strip()


@dataclass
class SemanticVerdict:
    hook_strength: float      # 0-1: do the first seconds of transcript grab?
    self_contained: float     # 0-1: comprehensible without stream context?
    payoff: float             # 0-1: does something actually land?
    moment_type: str          # one of MOMENT_TYPES
    title: str                # <= 8 words, streamer voice
    hook_line: str            # <= 7 words, for caption overlay
    verdict: str              # "post" | "maybe" | "skip"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_raw(cls, raw: dict) -> Optional["SemanticVerdict"]:
        """Build a normalized verdict from a model's parsed JSON, or None.

        Schema-constrained generation should always produce valid values, but
        the model file is user-swappable and grammar support varies by
        llama.cpp build — treat anything malformed as "no verdict" rather
        than letting a bad field poison selection.
        """
        if not isinstance(raw, dict):
            return None
        moment_type = str(raw.get("moment_type", "")).strip().lower()
        if moment_type not in MOMENT_TYPES:
            return None
        verdict = str(raw.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            return None
        return cls(
            hook_strength=_clamp01(raw.get("hook_strength")),
            self_contained=_clamp01(raw.get("self_contained")),
            payoff=_clamp01(raw.get("payoff")),
            moment_type=moment_type,
            title=_cap_words(raw.get("title"), TITLE_MAX_WORDS),
            hook_line=_cap_words(raw.get("hook_line"), HOOK_LINE_MAX_WORDS),
            verdict=verdict,
        )
