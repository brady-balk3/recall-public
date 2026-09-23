# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import re
import uuid

from core.models.clip import GameClip
from core.models.story import GameStory
from core.models.caption import ClipCaption
from engines.caption.formatter import format_tiktok, format_shorts


# Words too weak to carry a title on their own — used to trim quote edges and
# reject filler-only quotes.
_QUOTE_FILLER = {
    "uh", "um", "umm", "uhh", "hmm", "like", "so", "and", "the", "a", "an",
    "i", "you", "we", "it", "to", "of", "is", "was", "know", "yeah", "gonna",
    "just", "that", "this", "but", "or", "im", "its",
}


# Words a title must not END on. Stopping here reads as a sentence someone cut
# off mid-thought, which is what a fixed word-count truncation produces.
_DANGLING_TAIL = {
    "and", "but", "or", "so", "the", "a", "an", "to", "of", "that", "this",
    "it", "we", "i", "you", "he", "she", "they", "my", "your", "his", "her",
    "their", "our", "is", "was", "are", "were", "be", "been", "in", "on", "at",
    "for", "with", "from", "if", "when", "then", "than", "just", "like",
    "gonna", "wanna", "got", "get", "have", "has", "had", "do", "does", "did",
    "not", "no", "im", "its", "dont", "cant", "very", "really", "all", "some",
    "there", "here", "what", "who", "how", "because", "as", "up", "out",
}

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*")


def _trim_dangling(text):
    """Drop trailing connective words so the title ends on something solid."""
    words = text.split()
    while words and words[-1].strip(",.?!'\"").lower() in _DANGLING_TAIL:
        words.pop()
    return " ".join(words).strip(",;:-—– ")


def _clean_quote(tokens, max_words=8, max_chars=54):
    """Turn a run of transcript word tokens into a punchy title, or None.

    Prefers a natural sentence boundary over a fixed word count. A hard cut at
    N words leaves most titles hanging mid-thought ("I'm trying my best! We
    only have one"), so this looks for a real ending first and only falls back
    to truncation -- and even then refuses to stop on a connective word.
    """
    toks = [t.strip() for t in tokens if t and t.strip()]
    # Drop leading filler so titles start on a strong word.
    while toks and toks[0].strip(",.?!'\"").lower() in _QUOTE_FILLER:
        toks.pop(0)
    if not toks:
        return None
    window = re.sub(r"\s+", " ", " ".join(toks)).strip()

    # The window usually opens mid-sentence. If it starts with a stub that ends
    # almost immediately ("Tried. Oh my god, ..."), begin after that stub.
    lead = _SENTENCE_END.search(window)
    if lead and len(window[:lead.start()].split()) <= 2:
        remainder = window[lead.end():].strip()
        if remainder:
            window = remainder

    text = ""
    # 1) A sentence that actually ends inside the length budget is the best
    #    title we can get -- it reads complete on its own.
    for match in _SENTENCE_END.finditer(window):
        candidate = window[:match.end()].strip()
        if len(candidate) > max_chars:
            break
        if len(candidate.split()) >= 3:
            text = candidate
            break

    # 2) Otherwise fall back to truncation, then repair the tail.
    if not text:
        text = " ".join(window.split()[:max_words]).strip(",.-—– ")
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0]
        text = _trim_dangling(text)

    text = text.strip()
    if not text:
        return None
    non_filler = [t for t in text.split() if t.strip(",.?!'\"").lower() not in _QUOTE_FILLER]
    if len(non_filler) < 2 or len(text) < 6:
        return None
    return text[0].upper() + text[1:]


def _extract_quote(transcript, clip, pre=2.0, post=3.5):
    """Pull what the streamer said around the clip's reaction peak as a title."""
    if not transcript:
        return None
    peak = getattr(clip, "peak_timestamp", None)
    center = peak if peak is not None else (clip.start + clip.end) / 2.0
    lo = max(clip.start - 0.5, center - pre)
    hi = min(clip.end + 0.5, center + post)
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []) or []:
            ws = float(w.get("start", 0.0))
            if lo <= ws <= hi:
                words.append(str(w.get("word", "")))
    return _clean_quote(words)


# Short title prefix per non-gameplay scene.
_SCENE_TITLE = {"lobby": "Lobby", "intermission": "Intermission"}
_SCENE_FULL = {"lobby": "Lobby / Just chatting", "intermission": "Intermission"}


def _game_label(features: dict, reason: str):
    text = (reason or "").lower()
    if features.get("is_win", 0.0) >= 1.0 or "match win" in text:
        return "match_win"
    tier = float(features.get("game_tier", 0.0))
    if "elimination" in text or tier >= 0.8:
        return "elimination"
    if "knock" in text or tier >= 0.6:
        return "knock"
    if "rank progress" in text or tier >= 0.4:
        return "rank_progress"
    return None


def _dominant_reaction(breakdown: dict):
    nominal = {"voice": 0.45, "face": 0.18, "speech": 0.35}
    scaled = {k: float(breakdown.get(k, 0.0)) / v for k, v in nominal.items()}
    best = max(scaled, key=scaled.get)
    return best if scaled[best] >= 0.45 else None


def _reaction_text(kind):
    return {
        "voice": "voice reaction",
        "face": "facecam reaction",
        "speech": "hype callout",
        None: "selected moment",
    }[kind]


def _clean_reason(reason: str) -> str:
    reason = " ".join((reason or "").strip().split())
    if not reason:
        return ""
    return reason[:1].upper() + reason[1:]


def _quality_phrase(features: dict):
    agreement = float(features.get("signal_agreement", 0.0))
    dead_air = float(features.get("dead_air_ratio", 1.0))
    mean_intensity = float(features.get("mean_intensity", 0.0))
    if agreement >= 0.75 and dead_air <= 0.15:
        return "strong multi-signal evidence"
    if mean_intensity >= 1.5 and dead_air <= 0.20:
        return "sustained reaction"
    if dead_air <= 0.10:
        return "clean pacing"
    return "focused evidence"


def _build_title(clip: GameClip, story: GameStory):
    """Deterministic evidence label, used only when there is no quote to show.

    Carries no timecode: review surfaces render the clip's VOD position as its
    own field, so a "1:22:07 - " prefix here just spent the title's first third
    repeating it.
    """
    features = clip.features or {}
    breakdown = clip.modality_breakdown or {}
    reason = clip.reason or ""
    clean_reason = _clean_reason(reason)
    if clean_reason:
        return clean_reason

    game = _game_label(features, reason)
    reaction = _reaction_text(_dominant_reaction(breakdown))

    if game == "match_win":
        return "Match win evidence"
    if game == "elimination":
        return "Elimination evidence"
    if game == "knock":
        return "Knockdown evidence"
    if game == "rank_progress":
        return "Rank progress evidence"
    if reaction != "selected moment":
        return reaction[:1].upper() + reaction[1:]
    if story.label == "high_intensity_moment":
        return "High-intensity evidence"
    return "Selected clip"


def _build_description(clip: GameClip, quote: str = None):
    """POST-READY description (plan 22 §3.3) — reads like a caption a creator
    would publish, not like reviewer evidence-speak. Deterministic: quote of
    what was actually said, then the moment's reason in plain words."""
    reason = _clean_reason(clip.reason or "")
    if quote and reason:
        return f'"{quote}" — {reason[:1].lower() + reason[1:]}.'
    if quote:
        return f'"{quote}"'
    if reason:
        return f"{reason}."
    return "A moment worth clipping."


def _game_hashtag(game: str) -> str | None:
    """Detected game id -> a hashtag-safe tag ("league_of_legends" -> "leagueoflegends")."""
    if not game or game == "generic":
        return None
    return re.sub(r"[^a-z0-9]", "", game.lower()) or None


def _build_tags(clip: GameClip, game: str = "generic"):
    features = clip.features or {}
    breakdown = clip.modality_breakdown or {}
    label = _game_label(features, clip.reason or "")
    tags = []
    # The detected game is the highest-value discovery tag a clip can carry.
    game_tag = _game_hashtag(game)
    if game_tag:
        tags.append(game_tag)
    tags.extend(["gaming", "highlight"])
    if label == "match_win":
        tags.extend(["win", "clutch"])
    elif label == "elimination":
        tags.extend(["elim", "gameplay"])
    elif label == "knock":
        tags.extend(["knock", "gameplay"])
    elif label == "rank_progress":
        tags.extend(["ranked", "progress"])
    if float(breakdown.get("voice", 0.0)) >= 0.35 or float(breakdown.get("face", 0.0)) >= 0.12:
        tags.append("reaction")
    tags.extend(["shorts", "fyp"])

    deduped = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped[:6]


def generate_caption(clip: GameClip, story: GameStory, transcript: dict = None,
                     game: str = "generic") -> ClipCaption:
    """Generate social metadata for a clip.

    Title priority: (1) a non-gameplay scene label so lobby/intermission clips
    are never called gameplay reactions, (2) the semantic judge's title
    (HUMAN_CLIPS Package C) — a local LLM that actually read the transcript
    beats quoting it, (3) a quote of what the streamer actually said at the
    reaction peak (from the VOD transcript), and (4) the deterministic
    evidence label as a fallback.
    ``game`` is the auto-detected game id; it becomes the lead hashtag.
    """
    scene = getattr(clip, "scene_label", None)
    quote = _extract_quote(transcript, clip)
    semantic_title = (getattr(clip, "semantic_title", None) or "").strip()
    # A clip the judge voted "skip" survived selection on signal strength
    # alone; its title reflects the judge's dismissal ("Hello?"), which reads
    # worse than the quote fallback. Only trust titles the judge stood behind.
    if getattr(clip, "semantic_verdict", None) == "skip":
        semantic_title = ""

    if scene:
        prefix = _SCENE_TITLE.get(scene, "Lobby")
        title = f'{prefix}: "{quote}"' if quote else _SCENE_FULL.get(scene, "Lobby / Just chatting")
        desc = "Off-game moment (lobby / just chatting) — surfaced so you can keep or skip it."
        tags = ["justchatting", "stream", "clips", "fyp"]
    elif semantic_title:
        title = semantic_title
        desc = _build_description(clip, quote=quote)
        tags = _build_tags(clip, game=game)
    elif quote:
        title = quote
        desc = _build_description(clip, quote=quote)
        tags = _build_tags(clip, game=game)
    else:
        title = _build_title(clip, story)
        desc = _build_description(clip)
        tags = _build_tags(clip, game=game)

    return ClipCaption(
        caption_id=f"cap_{uuid.uuid4().hex[:8]}",
        clip_id=clip.clip_id,
        title=title,
        description=desc,
        tags=tags,
        tiktok_format=format_tiktok(title, tags),
        shorts_format=format_shorts(title, desc, tags),
    )
