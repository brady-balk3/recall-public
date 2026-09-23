# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Automatic game detection from a VOD's own OCR text.

The creator should never have to tell Recall what game they're playing --
Recall already runs OCR across the whole VOD for event triggers (kills/wins)
and lobby/menu detection, so identifying the game itself is free: score every
captured OCR string against a per-game fingerprint list (menu/HUD/killfeed
text distinctive to that title) and take the game with the most *distinct*
fingerprint hits.

Distinct-token counting (not raw hit count) matters: a persistent phrase OCR'd
every frame for 40 minutes must not identify a whole game by itself. The winner
also has to dominate the competing fingerprint evidence; otherwise unknown,
non-HUD, browser-heavy, and variety footage safely stays generic.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.bundle_paths import get_resource_dir
from engines.vision.layout_map import routed_text

# A game needs at least this many distinct fingerprint phrases hit somewhere
# in the VOD to be considered detected at all -- one lucky phrase match (OCR
# noise, a chat overlay quoting another game) shouldn't be enough.
MIN_DISTINCT_HITS = 2

# A winner with only a small share of all matched fingerprints is not a
# confident identification. Real failure cases were reported as 11-14%
# "confident" but still routed the whole scan: Undertale inherited GTA or
# Minecraft from repeated death text, while Silent Hill/browser footage
# inherited game titles held on screen. Generic is safer than a wrong curated
# OCR vocabulary.
MIN_WINNER_SHARE = 0.40

# Common OCR character confusions, collapsed to a canonical representative so a
# misread still matches its fingerprint (this VOD read ROYALE as "ROVALE" 203
# times, BATTLE ROYALE as "BATTLE ROVALE" 182 times -- none matched before).
# Applied identically to signatures AND OCR text, so it can only ADD matches,
# never change which game a clean read maps to. Kept conservative and guarded
# by validate_signatures.py (canonicalized tokens must stay distinct per game).
_OCR_CONFUSIONS = str.maketrans({
    "0": "O", "Q": "O",
    "1": "I", "L": "I", "|": "I", "!": "I",
    "5": "S",
    "8": "B",
    "2": "Z",
    "6": "G",
    "V": "Y",
})

_SIGNATURES_CACHE: Optional[Dict[str, List[str]]] = None


def _signatures_path() -> str:
    return os.path.join(get_resource_dir(), "configs", "game_signatures.json")


def _load_signatures() -> Dict[str, List[str]]:
    global _SIGNATURES_CACHE
    if _SIGNATURES_CACHE is not None:
        return _SIGNATURES_CACHE
    try:
        with open(_signatures_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    signatures = {
        game: [_normalize(tok) for tok in tokens]
        for game, tokens in data.items()
        if game != "_comment" and isinstance(tokens, list)
    }
    _SIGNATURES_CACHE = signatures
    return signatures


def _normalize(text: str) -> str:
    up = (text or "").upper().translate(_OCR_CONFUSIONS)
    up = re.sub(r"[^A-Z0-9 ]+", " ", up)
    return re.sub(r"\s+", " ", up).strip()


def _phrase_present(haystack: str, phrase: str) -> bool:
    pattern = r"\b" + r"\s+".join(re.escape(p) for p in phrase.split()) + r"\b"
    return re.search(pattern, haystack) is not None


@dataclass
class GameDetectionResult:
    game: str                              # detected game id, or "generic"
    confidence: float = 0.0                # winner's share of all distinct hits (0..1)
    hits: Dict[str, int] = field(default_factory=dict)  # distinct-token hit count per game


def score_games(texts: List[str]) -> Dict[str, int]:
    """Count distinct fingerprint phrases matched per game across all OCR text."""
    signatures = _load_signatures()
    if not signatures or not texts:
        return {}
    blob = " ".join(_normalize(t) for t in texts if t)
    if not blob:
        return {}
    scores: Dict[str, int] = {}
    for game, tokens in signatures.items():
        distinct = sum(1 for tok in tokens if tok and _phrase_present(blob, tok))
        if distinct:
            scores[game] = distinct
    return scores


def qualified_game(
    hits: Dict[str, int],
    *,
    min_distinct_hits: int = MIN_DISTINCT_HITS,
    min_winner_share: float = MIN_WINNER_SHARE,
) -> tuple[str, float]:
    """Return a game only when its distinct fingerprints win convincingly.

    Repetition is deliberately absent from this decision. A single phrase can
    persist because it is ordinary death text, a browser tile, a stream title,
    or a reference to another game. Two independent fingerprints are the
    minimum evidence, and a tied or heavily diluted winner remains generic.
    """
    if not hits:
        return "generic", 0.0
    winner_hits = max(int(value) for value in hits.values())
    leaders = [
        game for game, value in hits.items()
        if int(value) == winner_hits
    ]
    total = sum(max(0, int(value)) for value in hits.values())
    share = (winner_hits / total) if total else 0.0
    if (
        len(leaders) != 1
        or winner_hits < int(min_distinct_hits)
        or share < float(min_winner_share)
    ):
        return "generic", 0.0
    return leaders[0], round(share, 4)


def detect_game(unified_signals, min_distinct_hits: int = MIN_DISTINCT_HITS) -> GameDetectionResult:
    """Detect the game being played from OCR text collected across the VOD.

    Falls back to "generic" when no unique game clears ``min_distinct_hits``
    distinct fingerprints and the winner-share floor -- exactly like a game
    the fingerprint list doesn't cover yet, or a horror/story game with no
    recognizable HUD at all.
    """
    # Spatially routed text (chat/alert overlays removed) when annotated --
    # on-screen chat quoting another game must not misroute the whole scan.
    texts = [
        routed_text(s.ocr)
        for s in unified_signals
        if getattr(s, "ocr", None) and getattr(s.ocr, "text", None)
    ]
    texts = [t for t in texts if t]
    hits = score_games(texts)
    if not hits:
        return GameDetectionResult(game="generic", confidence=0.0, hits={})

    winner, confidence = qualified_game(
        hits,
        min_distinct_hits=min_distinct_hits,
    )
    return GameDetectionResult(game=winner, confidence=confidence, hits=hits)
