# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Stream-housekeeping detection (intros, sign-offs, logistics/promos).

A creator never clips their own goodbye, their "make sure to follow", or their
"can you guys hear me" mic check -- yet those windows sail through the reaction
gates because farewell/greeting chatter carries ordinary voice+face energy.
Real founder decks put "thank you for watching, I'll see you next time" and
"next stream will be Tuesday, join my discord" straight into the review deck.

This is a taste-INDEPENDENT quality defect: no personalization is needed to
know a sign-off is not postable. Detection is transcript-driven and
deliberately high-precision -- strong sign-off phrases reject anywhere, while
softer logistics/greeting language only counts as housekeeping when the window
also sits at the very start or end of the VOD (where intros/outros live).

Across the 24 golden fixtures, zero of 124 creator-kept ("post") moments
contain any of this language, so rejecting it cannot cost must-clip recall.
"""

from typing import Optional, Tuple

# Unambiguous end-of-stream sign-offs. A window containing any of these is a
# sign-off regardless of where it falls -- streamers do occasionally thank an
# audience mid-session, but "thanks for watching / see you next time" framing
# is never the payoff of a standalone clip.
SIGN_OFF = (
    "thanks for watching", "thank you for watching", "thanks for tuning in",
    "thank you guys for watching", "thank you all for watching",
    "see you next time", "see you guys next", "see you all next",
    "see you next stream", "see you tomorrow", "see you all tomorrow",
    "catch you later", "catch you guys later", "catch you all later",
    "end the stream", "ending the stream", "gonna end the stream",
    "end stream here", "call it a night", "call it there", "call it here",
    "gonna head out", "that's gonna be the stream", "that is the stream",
    "gonna wrap up", "gonna wrap it up", "wrap it up here",
    "good stream everyone", "good stream today", "appreciate you all",
    "appreciate you guys",
)

# Promo / housekeeping logistics. Common enough mid-stream (a streamer plugging
# their discord during gameplay) that these only count as housekeeping when the
# window is near a VOD boundary.
LOGISTICS = (
    "next stream", "join my discord", "join the discord", "in the discord",
    "follow me on", "hit that follow", "hit the follow", "make sure to follow",
    "make sure to subscribe", "check the description", "link in the description",
    "links in the description", "we're gonna raid", "we are gonna raid",
    "go raid", "gonna go raid",
)

# Intro / setup housekeeping. Near the START only.
GREETING = (
    "welcome back", "welcome in", "how's everyone doing", "how is everyone doing",
    "can you guys hear me", "can you hear me okay", "can everyone hear me",
    "testing the mic", "is the audio good", "is my audio", "we are live",
    "we're live now", "let me start the stream", "starting the stream",
    "let me get the stream", "just getting started",
)

# How close to the VOD's start/end a soft (logistics/greeting) signal must be to
# count. Absolute seconds -- intros/outros are a few minutes at most.
BOUNDARY_SEC = 180.0


def detect(
    text: str,
    start: float,
    end: float,
    vod_duration: float,
    boundary_sec: float = BOUNDARY_SEC,
) -> Tuple[bool, Optional[str]]:
    """Return (is_housekeeping, kind) for a candidate's transcript window.

    ``text`` is the joined transcript spoken across [start, end]; ``kind`` is
    one of ``"sign_off"`` / ``"logistics"`` / ``"greeting"`` or None. Empty
    text is never housekeeping (nothing was said -- let the dead-air gate rule).
    """
    t = (text or "").lower()
    if not t.strip():
        return False, None
    if any(phrase in t for phrase in SIGN_OFF):
        return True, "sign_off"
    near_start = start <= boundary_sec
    near_end = vod_duration > 0.0 and end >= vod_duration - boundary_sec
    if (near_start or near_end) and any(phrase in t for phrase in LOGISTICS):
        return True, "logistics"
    if near_start and any(phrase in t for phrase in GREETING):
        return True, "greeting"
    return False, None
