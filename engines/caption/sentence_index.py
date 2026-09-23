# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Sentence-boundary index from word-level ASR timestamps (HUMAN_CLIPS plan,
Package B).

A clip that opens mid-sentence reads machine-made regardless of content.
Whisper already gives us word-level timestamps; this module turns them into
two sorted arrays -- where sentences START and where they END -- so boundary
snapping (engines/reaction/boundaries.py) can prefer a natural spoken cut
over a merely quiet audio frame.

A sentence break is declared when either
  * the word's raw text ends in terminal punctuation (. ! ?) -- Whisper emits
    punctuation attached to the word, so ``"right."`` marks a sentence end, or
  * the gap to the next word exceeds GAP_BREAK_SEC -- streamers often trail
    off without punctuation, and a real pause is a natural cut whether or not
    the model punctuated it.

Deliberately dependency-free (numpy + stdlib): callers include the reaction
engine and the golden harness, neither of which should pull in ASR machinery
just to read timestamps out of an existing transcript dict.
"""

from typing import Optional, Tuple

import numpy as np

TERMINAL_PUNCT = (".", "!", "?")
# Closing quotes/brackets that can trail the terminal punctuation ('right."').
# Stripping a trailing apostrophe is harmless: "goin'" -> "goin" still has no
# terminal punctuation, so casual contractions never fake a sentence break.
_CLOSERS = "\"'”’)]"
# Inter-word silence long enough to count as a sentence break on its own.
GAP_BREAK_SEC = 0.8


def _is_terminal(text: str) -> bool:
    return text.rstrip(_CLOSERS).endswith(TERMINAL_PUNCT)


def build_sentence_index(transcript: Optional[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (sentence_starts, sentence_ends) arrays from a Whisper transcript.

    ``transcript`` is the Whisper result dict ({"segments": [{"words": [...]}]}
    with per-word {"word", "start", "end"}); None/empty yields two empty
    arrays, which downstream snapping treats as "no sentence data" -- the
    critical no-transcript regression guarantee lives on that path.

    Both arrays are float64 seconds, sorted ascending, and the same length:
    entry i is sentence i's first-word start and last-word end.
    """
    empty = (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    if not transcript:
        return empty

    # Flatten to (raw_text, start, end) and sort by start time: region-scoped
    # (fast-mode scout) transcripts are merged from separate Whisper calls, so
    # segment order is not guaranteed to be time order.
    words = []
    for segment in transcript.get("segments", []) or []:
        for word in segment.get("words", []) or []:
            text = (word.get("word") or "").strip()
            if not text:
                continue
            words.append((text, float(word.get("start", 0.0)), float(word.get("end", 0.0))))
    if not words:
        return empty
    words.sort(key=lambda w: w[1])

    starts = [words[0][1]]
    ends = []
    for i, (text, _w_start, w_end) in enumerate(words):
        is_last = i == len(words) - 1
        gap_break = (not is_last) and (words[i + 1][1] - w_end > GAP_BREAK_SEC)
        if is_last or gap_break or _is_terminal(text):
            ends.append(w_end)
            if not is_last:
                starts.append(words[i + 1][1])

    return (np.asarray(starts, dtype=np.float64),
            np.asarray(ends, dtype=np.float64))
