# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Re-time a creator's corrected caption text onto the machine's word timings.

Burned-in captions are word-by-word: generate_tiktok_ass highlights each word as
it is spoken, so every word needs a start and an end. A creator correcting a
caption gives us text and nothing else. This turns that text back into a timed
word list.

The rule is that CORRECTING A WORD MUST NOT MOVE THE WORDS AROUND IT. The
alignment head produced those timings from the audio and they are right; a
correction is almost always local ("charreded" -> "sharted", one word in a
sentence of nine), and re-timing the whole line to fix one word would trade a
spelling error for eight timing errors. So an unchanged word keeps its own
start and end, to the microsecond, and only the changed run is re-timed.

Word-count changes are the reason this is a diff and not a zip. MEASURED
2026-09-01 on clip c_6cc09070: the decode said "Buddhist characters," where the
creator said "Computah, show Kyruxi" -- two tokens standing in for three. A
positional zip would have shifted every later word by one and desynchronised the
rest of the clip. Here the two old tokens' combined span is redistributed across
the three new ones by character length, and the words after it never move.

Character length is a crude proxy for duration, but it is the right kind of
crude: it is monotonic in syllable count, it needs nothing from the audio, and
it only ever applies inside a span the creator has already told us is wrong.
The alternative -- re-running forced alignment on the edited text -- would be
more accurate and cost a model load per keystroke-save, for a span that is
typically two words long.

Insertions and deletions are folded into a neighbouring run rather than handled
separately, so every non-equal edit has both old words to take time from and new
words to give it to. That costs one adjacent word its exact timing and buys an
invariant worth more: the output always covers exactly the same span as the
input, in order, with no gap invented and none lost.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Sequence

# Below this, a re-timed span is treated as instantaneous and its words are
# spread evenly rather than by weight -- dividing a 0.0s span by character
# count is arithmetic that means nothing.
_MIN_SPAN = 1e-3


def words_to_text(words: Sequence[Dict]) -> str:
    """The caption as the creator reads it, from a timed word list."""
    return " ".join(str(word.get("word", "")).strip()
                    for word in words if str(word.get("word", "")).strip())


def _normalized_opcodes(old: Sequence[str], new: Sequence[str]):
    """Opcodes where every non-equal run has BOTH old and new words in it.

    difflib emits pure inserts (no old words, so no time to take) and pure
    deletes (no new words, so nothing to give time to). Each is merged into a
    neighbouring run, which makes every edit a replace and lets the caller
    treat them all identically. The merge is what costs one adjacent word its
    original timing; see the module docstring.
    """
    raw = SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
    ops: List[List] = []
    for tag, i1, i2, j1, j2 in raw:
        degenerate = tag != "equal" and (i1 == i2 or j1 == j2)
        if degenerate and ops:
            # Absorb into whatever came before, which then covers both sides.
            ops[-1][2], ops[-1][4] = i2, j2
            ops[-1][0] = "replace"
            continue
        ops.append([tag, i1, i2, j1, j2])

    if ops and ops[0][0] != "equal" and (ops[0][1] == ops[0][2] or ops[0][3] == ops[0][4]):
        # Nothing before it to absorb into: pull the run that follows instead.
        if len(ops) > 1:
            ops[0][2], ops[0][4] = ops[1][2], ops[1][4]
            ops[0][0] = "replace"
            del ops[1]
    return [tuple(op) for op in ops]


def retime_edited_words(words: Sequence[Dict], text: str) -> List[Dict]:
    """Map corrected ``text`` onto ``words``' timings.

    ``words`` is the machine's timed word list (whisper's shape: ``word`` with
    a leading space, ``start``, ``end``, clip-relative). ``text`` is what the
    creator says the words are. Returns a new timed list; the input is not
    mutated.

    Unchanged words keep their exact timings. A changed run is re-timed inside
    the span the words it replaces occupied, so the correction never disturbs
    the rest of the line.
    """
    old_words = [dict(word) for word in (words or [])
                 if str(word.get("word", "")).strip()]
    new_tokens = (text or "").split()

    if not new_tokens:
        return []
    if not old_words:
        # Nothing to inherit timings from. The caller (services) only reaches
        # this with a non-empty decode, but a caption is not worth a crash.
        return []

    old_tokens = [str(word["word"]).strip() for word in old_words]
    if old_tokens == new_tokens:
        return old_words

    out: List[Dict] = []
    for tag, i1, i2, j1, j2 in _normalized_opcodes(
        [token.lower() for token in old_tokens],
        [token.lower() for token in new_tokens],
    ):
        if tag == "equal":
            # Same words: keep the machine's timings exactly, but emit the
            # creator's spelling -- "equal" here is case-insensitive, so this
            # is where a capitalisation-only fix survives.
            for offset in range(j2 - j1):
                source = old_words[i1 + offset]
                out.append({**source, "word": " " + new_tokens[j1 + offset]})
            continue

        span_start = float(old_words[i1]["start"])
        span_end = float(old_words[i2 - 1]["end"])
        out.extend(_spread(new_tokens[j1:j2], span_start, span_end))

    return out


def _spread(tokens: Sequence[str], start: float, end: float) -> List[Dict]:
    """Lay ``tokens`` across ``start``..``end``, weighted by character length."""
    span = max(0.0, end - start)
    weights = [max(1, len(token)) for token in tokens]
    total = sum(weights)

    out: List[Dict] = []
    cursor = start
    for index, token in enumerate(tokens):
        if span <= _MIN_SPAN:
            share = span / len(tokens)
        else:
            share = span * (weights[index] / total)
        # The last token lands exactly on the span end rather than on an
        # accumulated float sum, so the run always covers what it replaced.
        word_end = end if index == len(tokens) - 1 else cursor + share
        out.append({"word": " " + token,
                    "start": round(cursor, 3),
                    "end": round(max(cursor, word_end), 3)})
        cursor = word_end
    return out
