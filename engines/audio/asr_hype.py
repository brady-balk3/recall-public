# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""ASR Hype Engine (plan §5.3).

Turns a VOD transcript into a per-second ``speech_hype`` track by matching a
configurable hype lexicon (``configs/hype_lexicon.json``) against the word
stream. A matched phrase deposits its weight at the phrase's start time and
diffuses +/-1.5s, so a "let's go!!" or "victory royale" callout lifts R(t)
even when the OCR banner and facecam are ambiguous.

The lexicon is editable JSON (per-game extensible) so hype words change without
code edits.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np

DIFFUSE_SEC = 1.5  # how far a matched word's hype spreads in time


@dataclass
class HypeFrame:
    timestamp: float
    speech_hype: float

    def to_dict(self):
        return asdict(self)


def load_lexicon(path: str, game: str = "default") -> Dict[str, float]:
    """Load the hype lexicon, merging the per-game profile over the default."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(data.get("default", {}))
    if game and game != "default":
        merged.update(data.get(game, {}))
    # drop comment / non-numeric keys
    return {k.lower(): float(v) for k, v in merged.items() if not k.startswith("_")}


def build_hype_track(transcript: dict, lexicon: Dict[str, float], duration: float) -> List[HypeFrame]:
    """Match lexicon phrases against the word stream -> diffused 1 fps hype track."""
    from engines.caption.vod_transcribe import iter_words

    words = list(iter_words(transcript))
    n = max(1, int(np.ceil(duration)) + 1)
    track = np.zeros(n, dtype=np.float32)

    # Pre-split lexicon phrases into token lists; longest first so multi-word
    # phrases ("victory royale") win over their single-word substrings.
    phrases = sorted(
        ((p.split(), w) for p, w in lexicon.items()),
        key=lambda pw: len(pw[0]), reverse=True,
    )

    word_texts = [w[0] for w in words]
    for i in range(len(words)):
        for tokens, weight in phrases:
            k = len(tokens)
            if i + k > len(words):
                continue
            if word_texts[i:i + k] == tokens:
                start = words[i][1]
                _deposit(track, start, weight)

    frames = [HypeFrame(timestamp=float(t), speech_hype=float(track[t])) for t in range(n)]
    return frames


def _deposit(track: np.ndarray, t: float, weight: float):
    """Add a triangular bump of `weight` at time t, decaying over +/-DIFFUSE_SEC."""
    lo = int(np.floor(t - DIFFUSE_SEC))
    hi = int(np.ceil(t + DIFFUSE_SEC))
    for s in range(max(0, lo), min(track.size, hi + 1)):
        decay = max(0.0, 1.0 - abs(s - t) / DIFFUSE_SEC)
        if decay > 0:
            track[s] = max(track[s], weight * decay)
