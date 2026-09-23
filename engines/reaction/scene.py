# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Non-gameplay scene detection (lobby / menu / intermission).

The reaction engine scores the *streamer's* reaction, so a hyped pregame lobby
(banter with chat, laughter) can outscore focused gameplay even though nothing
game-worthy happened. OCR only ever fires *positive* gameplay triggers
(kills/wins), so it can't tell a calm mid-match moment from the lobby.

This module reads the OCR text already captured across a clip's window and, when
it's dominated by menu/lobby/intermission text (and no gameplay trigger is
present), returns a scene code. The pipeline uses it to *label + filter* those
clips in review — never to silently delete them.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from core.bundle_paths import get_resource_dir

# Scene code -> creator-facing label shown in the review UI / clip title.
SCENE_LABELS = {
    "intermission": "Intermission",
    "lobby": "Lobby / Just chatting",
}

_LEXICON_CACHE: dict = {}


def _lexicon_path() -> str:
    return os.path.join(get_resource_dir(), "configs", "scene_lexicon.json")


def _load_lexicon() -> dict:
    path = _lexicon_path()
    cached = _LEXICON_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    _LEXICON_CACHE[path] = data
    return data


def _normalize(text: str) -> str:
    """Uppercase, fold common OCR confusions, strip to A-Z/0-9/space."""
    up = (text or "").upper().replace("|", "I").replace("!", "I").replace("0", "O")
    up = re.sub(r"[^A-Z0-9 ]+", " ", up)
    return re.sub(r"\s+", " ", up).strip()


def _phrase_present(haystack: str, phrase: str) -> bool:
    pattern = r"\b" + r"\s+".join(re.escape(p) for p in phrase.split()) + r"\b"
    return re.search(pattern, haystack) is not None


def _tokens_for(game: str) -> tuple:
    lex = _load_lexicon()
    game = (game or "generic").strip().lower()
    intermission = [_normalize(t) for t in lex.get("intermission", [])]
    lobby = [_normalize(t) for t in (lex.get("generic", []) + lex.get(game, []))]
    return intermission, lobby


def classify_scene(
    texts: List[str],
    game: str = "generic",
    has_game_evidence: bool = False,
) -> Optional[str]:
    """Classify a clip window's OCR text as a non-gameplay scene, or None.

    ``texts``: every OCR string captured inside the clip window.
    ``has_game_evidence``: True when a gameplay trigger (kill/win banner) fired
    in the window — real gameplay is never relabeled as a menu.

    Returns "intermission", "lobby", or None. Intermission (starting-soon / BRB)
    wins on a single hit; lobby needs a game-specific/menu hit. Both are ignored
    when gameplay evidence is present.
    """
    if has_game_evidence or not texts:
        return None

    intermission, lobby = _tokens_for(game)
    blob = " ".join(_normalize(t) for t in texts if t)
    if not blob:
        return None

    # A "starting soon" / "be right back" frame is unambiguous — one hit is enough.
    if any(_phrase_present(blob, tok) for tok in intermission):
        return "intermission"

    # Menu/lobby text: distinct hits so a lone overlay word can't trip it.
    distinct = {tok for tok in lobby if _phrase_present(blob, tok)}
    if len(distinct) >= 1:
        return "lobby"
    return None


def scene_display(scene: Optional[str]) -> Optional[str]:
    return SCENE_LABELS.get(scene) if scene else None
