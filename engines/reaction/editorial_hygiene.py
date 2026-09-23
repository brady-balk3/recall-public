# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Taste-independent cleanup signals for the creator review deck.

This module deliberately does not rank clips.  It identifies two structural
classes that should be resolved before a deck budget is applied:

* sustained credits/end-card regions, and
* lightweight, privacy-preserving content fingerprints used to recognize the
  same story across alternate candidate windows.

The selector remains the authority for exemptions.  A region flag says "this
looks like non-content"; an explicit viewer marker or a clearly completed beat
may still keep the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, List, Optional, Sequence, Tuple


_STRONG_CREDIT_RE = re.compile(
    r"\b(?:"
    r"special\s+thanks|thank\s+you\s+for\s+playing|"
    r"a\s+game\s+by|created\s+by|developed\s+by|published\s+by|"
    r"all\s+rights\s+reserved|executive\s+producer|"
    r"directed\s+by"
    r")\b",
    re.IGNORECASE,
)
_CREDITS_LABEL_RE = re.compile(r"\bcredits?\b", re.IGNORECASE)
_ROLE_RE = re.compile(
    r"\b(?:producer|director|programmer|programming|developer|"
    r"game\s+design|lead\s+artist|sound\s+design|music\s+by|"
    r"voice\s+cast|localization|quality\s+assurance)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_TOKEN_STOP = frozenset(
    {
        "about", "after", "again", "ain't", "also", "and", "are", "because",
        "been", "before", "being", "but", "can", "could", "did", "does",
        "doing", "for", "from", "get", "got", "had", "has", "have", "here",
        "how", "into", "its", "just", "like", "look", "more", "not", "now",
        "okay", "one", "our", "out", "really", "right", "said", "see", "she",
        "that", "the", "their", "them", "then", "there", "they", "this",
        "through", "too", "was", "we", "were", "what", "when", "where",
        "which", "who", "why", "will", "with", "would", "yeah", "you", "your",
    }
)


@dataclass(frozen=True)
class NonContentRegion:
    start: float
    end: float
    kind: str
    evidence_count: int


def moment_token_hashes(*texts: str, limit: int = 24) -> List[str]:
    """Return stable hashed content tokens without persisting transcript text."""
    tokens = set()
    for text in texts:
        for token in _TOKEN_RE.findall(str(text or "").lower()):
            token = token.strip("'")
            if len(token) < 3 or token in _TOKEN_STOP or token.isdigit():
                continue
            tokens.add(token)
    ordered = sorted(
        hashlib.blake2s(token.encode("utf-8"), digest_size=4).hexdigest()
        for token in tokens
    )
    return ordered[: max(0, int(limit))]


def _merge_regions(regions: Sequence[NonContentRegion]) -> List[NonContentRegion]:
    if not regions:
        return []
    merged: List[NonContentRegion] = []
    for region in sorted(regions, key=lambda item: (item.start, item.end)):
        if not merged or region.start > merged[-1].end + 5.0:
            merged.append(region)
            continue
        previous = merged[-1]
        kinds = set(previous.kind.split("+")) | set(region.kind.split("+"))
        merged[-1] = NonContentRegion(
            start=min(previous.start, region.start),
            end=max(previous.end, region.end),
            kind="+".join(sorted(kinds)),
            evidence_count=previous.evidence_count + region.evidence_count,
        )
    return merged


def detect_non_content_regions(
    ocr_samples: Iterable[Tuple[float, str]],
    transcript_segments: Optional[Iterable[Tuple[float, float, str]]],
    vod_duration: float,
    *,
    credit_gap_sec: float = 60.0,
) -> List[NonContentRegion]:
    """Find sustained credits/end-card runs.

    A lone role word or menu label is not enough. Credits require repeated or
    compound evidence. Candidate-local sign-offs remain the responsibility of
    ``engines.reaction.housekeeping``; they are not propagated across later
    content.
    """
    duration = max(0.0, float(vod_duration or 0.0))
    hits: List[Tuple[float, int]] = []
    for raw_timestamp, raw_text in ocr_samples:
        text = str(raw_text or "")
        strong = len(_STRONG_CREDIT_RE.findall(text))
        roles = len(_ROLE_RE.findall(text))
        credits_label = 1 if _CREDITS_LABEL_RE.search(text) else 0
        score = strong * 3 + min(roles, 3) + credits_label
        if score >= 2:
            hits.append((max(0.0, float(raw_timestamp)), score))

    regions: List[NonContentRegion] = []
    if hits:
        run: List[Tuple[float, int]] = [hits[0]]
        runs: List[List[Tuple[float, int]]] = []
        for hit in hits[1:]:
            if hit[0] - run[-1][0] <= max(1.0, float(credit_gap_sec)):
                run.append(hit)
            else:
                runs.append(run)
                run = [hit]
        runs.append(run)
        for run in runs:
            evidence = sum(score for _, score in run)
            # Require persistence or compound evidence; a static menu option
            # that merely says "CREDITS" never enters ``hits`` above.
            if evidence < 6 and len(run) < 2:
                continue
            regions.append(
                NonContentRegion(
                    start=max(0.0, run[0][0] - 6.0),
                    end=min(duration or run[-1][0] + 18.0, run[-1][0] + 18.0),
                    kind="credits_or_endcard",
                    evidence_count=len(run),
                )
            )

    return _merge_regions(regions)


def region_overlap_fraction(
    start: float,
    end: float,
    region: NonContentRegion,
) -> float:
    duration = max(0.0, float(end) - float(start))
    if duration <= 0.0:
        return 0.0
    overlap = max(
        0.0,
        min(float(end), region.end) - max(float(start), region.start),
    )
    return overlap / duration


def classify_candidate_region(
    start: float,
    end: float,
    regions: Sequence[NonContentRegion],
    *,
    min_overlap: float = 0.50,
) -> Optional[NonContentRegion]:
    """Return the strongest region covering most of a candidate, if any."""
    matches = [
        (region_overlap_fraction(start, end, region), region)
        for region in regions
    ]
    matches = [item for item in matches if item[0] >= min_overlap]
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1].evidence_count))[1]
