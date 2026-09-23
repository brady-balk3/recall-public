# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Spelled-out numbers -> digits, for burned-in captions.

Qwen3-ASR writes numbers as words ("one hundred and thirty") no matter what the
prompt asks; the instruction "Write numbers as digits" was MEASURED 2026-08-18 to
have no effect. It is an ASR model, not an instruction follower, so the fix has
to be deterministic post-processing.

The target convention is whisper's own, because that is what the caption
renderer and the founder are used to seeing: small counts stay words ("five
times", "two games"), larger values become digits ("120", "130, 140"). A blanket
conversion would turn "that one game" into "that 1 game", which is worse than
the problem being solved.

Rules, therefore:

* convert when the parsed value is >= 20 ("twenty" -> 20, "one hundred and
  thirty" -> 130);
* convert multi-word values even below 20 when they carry a scale word
  ("a hundred" -> 100);
* leave bare single words below 20 alone ("one", "five", "twelve");
* never touch a word that is not part of a number phrase;
* read a hyphen as part of the number ("twenty-three" -> 23), which is how the
  ASR writes every compound it hears -- the spaced form "twenty three" is rare
  in its output, so a hyphen rule that only worked inside a scale phrase left
  "20-three kills" and "90-8000 XP" burned into captions;
* attach an ordinal only across a hyphen ("thirty-second" -> "32nd"), never
  across a space: "a thirty second clip" is a duration, not the 32nd of
  anything.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
# Ordinals stop a phrase dead -- nothing follows "fifty-fourth" inside the same
# number. Scale ordinals ("hundredth") are left out on purpose: they never show
# up in this speech and would need their own multiply/terminate rules.
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
}
# The values a group can sit at and still accept a unit on top: "twenty" takes
# "three", "twenty-three" does not take "four".
_OPEN_TENS = (0, 20, 30, 40, 50, 60, 70, 80, 90)
# "and" only continues a phrase after a scale word ("a hundred AND thirty"),
# never on its own -- otherwise "five and six" collapses into one number. A
# hyphen carries no such condition: between two number words it is always one
# compound number.
GLUE = {"and", "-"}

_TOKEN = re.compile(r"[A-Za-z']+|\d+|[^A-Za-z'\d\s]|\s+")


def _is_number_word(word: str) -> bool:
    low = word.lower()
    return low in UNITS or low in TENS or low in SCALES or low in ORDINALS


def _ordinal_suffix(value: int) -> str:
    if value % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _parse(words: List[str]) -> Optional[Tuple[int, int, bool]]:
    """Parse a number phrase at the start of ``words``.

    Returns ``(value, words_consumed, is_ordinal)``, or None when the run is
    not a number.
    """
    total = 0        # completed scale groups
    current = 0      # the group being built
    used = 0
    seen_scale = False
    saw_any = False
    ordinal = False

    index = 0
    while index < len(words):
        low = words[index].lower()
        # An ordinal joins the number in front of it only through a hyphen.
        hyphenated = index > 0 and words[index - 1] == "-"
        if low in UNITS:
            # "twenty five" is fine; "five six" is two numbers, not one.
            if current % 100 not in _OPEN_TENS and current:
                break
            current += UNITS[low]
        elif low in TENS:
            if current and current % 100 != 0:
                break
            current += TENS[low]
        elif low in SCALES:
            scale = SCALES[low]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            seen_scale = True
        elif low in ORDINALS and (hyphenated or not saw_any):
            step = ORDINALS[low]
            if step in TENS.values():
                if current and current % 100 != 0:
                    break
            elif current % 100 not in _OPEN_TENS and current:
                break
            current += step
            ordinal = True
            saw_any = True
            index += 1
            used = index
            break  # "fifty-fourth" ends the phrase; no number word may follow.
        elif low == "-":
            # A hyphen between two number words is always one compound number,
            # at any scale -- this is the form the ASR actually emits.
            if index + 1 >= len(words) or not _is_number_word(words[index + 1]):
                break
            index += 1
            continue
        elif low in GLUE:
            # "and" only bridges a gap inside a scaled phrase, and only if a
            # real number word follows.
            if not seen_scale or index + 1 >= len(words) \
                    or not _is_number_word(words[index + 1]):
                break
            index += 1
            continue
        else:
            break
        saw_any = True
        index += 1
        # Committed only after a real number word, never after glue: a phrase
        # that stops at the word past the glue ("one-two") would otherwise
        # swallow the hyphen and print "1two".
        used = index

    if not saw_any:
        return None
    return total + current, used, ordinal


def digitize(text: str, min_value: int = 20) -> str:
    """Rewrite spelled-out numbers as digits, whisper-style.

    ``min_value`` is the threshold below which a single bare word is left as
    prose. Phrases that use a scale word ("a hundred") always convert.
    """
    if not text:
        return text
    tokens = _TOKEN.findall(text)
    out: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.strip() or not token[0].isalpha():
            out.append(token)
            i += 1
            continue

        # Gather the candidate run: number words plus the whitespace/glue
        # between them, so "one hundred and thirty" is seen as one phrase.
        words: List[str] = []
        positions: List[int] = []
        j = i
        # "a hundred percent" should read "100 percent", not "a 100 percent";
        # the article IS the quantity here. Only when a scale word follows.
        article = False
        if token.lower() == "a":
            k = j + 1
            while k < len(tokens) and tokens[k].strip() == "":
                k += 1
            if k < len(tokens) and tokens[k].lower() in SCALES:
                article = True
                j = k
        while j < len(tokens):
            tok = tokens[j]
            if tok.strip() == "":
                j += 1
                continue
            if _is_number_word(tok) or (tok.lower() in GLUE and words):
                words.append(tok)
                positions.append(j)
                j += 1
                continue
            break

        parsed = _parse(words) if words else None
        if parsed is None:
            out.append(token)
            i += 1
            continue

        value, consumed, ordinal = parsed
        if consumed == 0:
            out.append(token)
            i += 1
            continue

        # "a hundred" style phrases carry a scale word; bare small numbers do not.
        has_scale = any(w.lower() in SCALES for w in words[:consumed])
        if value >= min_value or has_scale:
            out.append(str(value) + (_ordinal_suffix(value) if ordinal else ""))
            i = positions[consumed - 1] + 1
            _ = article
        else:
            out.append(token)
            i += 1
    return "".join(out)
