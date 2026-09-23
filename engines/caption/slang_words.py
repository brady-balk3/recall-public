# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Deterministic corrections for ASR slang substitutions.

General ASR models can substitute common words for uncommon slang. Prompt
biasing alone does not guarantee consistent spellings, so this module applies
explicit token rules before forced alignment.

The rules are asymmetric: wetty, chart inflections, and charreded are replaced
unconditionally; shark is replaced only in an exclamatory frame because it is
also an ordinary noun. The chart rules are domain-specific and can alter a
literal use of that word. Word boundaries protect unrelated compounds, and
charred remains unchanged.

Every replacement preserves token count so the aligner and transcript retain
the same word indices."""

from __future__ import annotations

import re

# Whole-token substitutions that need no context to be safe.
_ALWAYS = {
    "wetty": "sweaty",
    "chart": "shart",
    "charts": "sharts",
    "charted": "sharted",
    "charting": "sharting",
    "charreded": "sharted",
}

# Words that can precede the target and turn it into an exclamation. Ordered
# alternation, longest first, so "ain't that a" wins over a bare article.
_FRAME = (
    r"(?:what|who|where|whatever)\s+the"
    r"|ain'?t\s+that\s+a"
    r"|you'?re\s+a"
    r"|holy"
    r"|oh\s+my"
)

# An intensifier may sit between the frame and the word ("whatever the freaking
# chart?"), and must not break the match.
_INTENSIFIER = r"(?:freaking|frickin'?|fricking|actual|damn|heck|hell)\s+"

# The prefix is captured whole and re-emitted verbatim: rebuilding it from its
# parts would normalise the creator's own spacing.
_FRAMED_SHARK = re.compile(
    rf"\b((?:{_FRAME})\s+(?:{_INTENSIFIER})?)(sharks?)\b",
    re.IGNORECASE,
)

_ALWAYS_TOKEN = re.compile(
    r"\b(" + "|".join(sorted(_ALWAYS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _match_case(source: str, replacement: str) -> str:
    """Carry ``source``'s capitalisation onto ``replacement``.

    Qwen capitalises sentence-initially and the caption renderer shows the text
    verbatim, so "Chart." must become "Shart." and not "shart.".
    """
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def correct_slang(text: str) -> str:
    """Rewrite this creator's slang back to what they actually said."""
    if not text:
        return text

    def _always(match: re.Match) -> str:
        word = match.group(1)
        return _match_case(word, _ALWAYS[word.lower()])

    text = _ALWAYS_TOKEN.sub(_always, text)

    def _framed(match: re.Match) -> str:
        prefix, word = match.group(1), match.group(2)
        plural = word.lower().endswith("s")
        return prefix + _match_case(word, "sharts" if plural else "shart")

    return _FRAMED_SHARK.sub(_framed, text)


# --------------------------------------------------- the -maxxing compounds

# Normalize productive maxxing compounds after forced alignment. Joining two
# tokens is safe for timing only after their individual boundaries exist: the
# compound spans from the stem's start to the suffix's end. Bare max is unchanged.
# Keep the single-x spelling during alignment and add the display spelling here;
# unusual doubled consonants can destabilize forced alignment.

_MAXING = re.compile(r"^(max{1,2}ing)([^\w\s]*)$", re.IGNORECASE)

# An ALREADY-closed compound, single or double x. The stem is required and must
# be a real word, so a bare "maxing" (no stem) never matches here.
_CLOSED = re.compile(r"^([A-Za-z][A-Za-z'-]{1,})(max{1,2}ing)([^\w\s]*)$",
                     re.IGNORECASE)

# Exclude function words from stems so ordinary grammatical sequences do not
# become invented compounds.
_NOT_A_STEM = frozenset("""
a an the and but or so we he she it they you i my your his her their our its
is was are were be been am not no just like too very of to in on at that this
""".split())

# Sentence-final punctuation on the stem means the two tokens are not one word.
_STEM = re.compile(r"^([A-Za-z][A-Za-z'-]*)$")


def join_maxxing(words):
    """Merge ``<stem> maxing`` word pairs into one ``<stem>maxxing`` token.

    ``words`` is the aligned word list -- dicts of ``word``/``start``/``end``,
    with whisper's leading space on each. Returns a new list; the input is not
    mutated.
    """
    if not words:
        return list(words or [])

    def _display(token: str) -> str:
        """The creator's spelling for an already-closed compound."""
        closed = _CLOSED.match(token)
        if not closed or closed.group(1).lower() in _NOT_A_STEM:
            return token
        return closed.group(1) + "maxxing" + closed.group(3)

    out = []
    for word in words:
        token = word["word"].strip()
        match = _MAXING.match(token)
        stem = _STEM.match(out[-1]["word"].strip()) if out else None
        if not match or stem is None or stem.group(1).lower() in _NOT_A_STEM:
            # Not a pair to merge, but it may still be a closed compound the
            # aligner saw with one x. Same timings, display spelling only --
            # and a word this leaves alone is copied verbatim, so nothing here
            # normalises whitespace the caption renderer joins on.
            display = _display(token)
            out.append(dict(word) if display == token
                       else {**word, "word": word["word"].replace(token, display, 1)})
            continue
        previous = out.pop()
        out.append({
            **previous,
            # The stem keeps its own capitalisation -- Qwen capitalises
            # sentence-initially and the caption renderer shows text verbatim,
            # so "Aura maxing." must become "Auramaxxing." and not lose the A.
            "word": " " + stem.group(1) + "maxxing" + match.group(2),
            "start": float(previous["start"]),
            "end": float(word["end"]),
        })
    return out
