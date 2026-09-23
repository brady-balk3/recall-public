# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import re
from typing import Dict, List, Optional
from core.models.signal import UnifiedSignal
from core.models.event import GameEvent

MOTION_THRESHOLD = 0.25

# Per-game OCR event trigger tiers. Tier names match fusion.GAME_TIER_WEIGHTS
# (terminal_win > elimination > knock > rank_progress > generic_highlight).
# "generic" is checked for EVERY game (including one the auto-detector didn't
# recognize) so an unrecognized or unlisted title still gets *some* OCR event
# capability from broadly common banner text, rather than none at all.
GAME_EVENT_TRIGGERS: Dict[str, Dict[str, tuple]] = {
    "generic": {
        # Deliberately no bare "VICTORY" here: cosmetic/shop text ("Victory
        # Dance" emote) makes it too loose as a global trigger. It's still a
        # valid win banner for specific games (e.g. Valorant) where it's
        # listed in that game's own dict below.
        "terminal_win": ("WINNER", "1ST PLACE", "CHAMPION", "MATCH WON"),
        "elimination": ("ELIMINATED", "KILLED", "DOWNED"),
        "generic_highlight": ("CLUTCH", "MVP", "ACE"),
    },
    "fortnite": {
        "terminal_win": (
            "VICTORY ROYALE",
            "#1 VICTORY ROYALE",
            "1 VICTORY ROYALE",
            "VICT0RY ROYALE",
            "VICTORY R0YALE",
        ),
        "elimination": ("ELIMINATED", "ELIMINATION", "HEADSHOT"),
        "knock": ("KNOCKED", "KNOCKDOWN"),
        "rank_progress": ("CROWN", "RANK", "PLACEMENT", "SURVIVOR"),
        "generic_highlight": ("CLUTCH", "ASSIST"),
    },
    "valorant": {
        "terminal_win": ("VICTORY", "DEFEAT"),
        "elimination": ("ATTACKERS WIN", "DEFENDERS WIN", "SPIKE HAS BEEN DEFUSED"),
        "knock": ("SPIKE HAS BEEN PLANTED",),
        "generic_highlight": ("ACE", "THRIFTY", "CLUTCH"),
    },
    "apex": {
        "terminal_win": ("CHAMPIONS OF THE ARENA", "CHAMPION"),
        "elimination": ("SQUAD ELIMINATED", "KNOCKED DOWN"),
        "knock": ("KNOCKDOWN SHIELD",),
        "rank_progress": ("RING DAMAGE", "RESPAWN BEACON"),
        "generic_highlight": ("LEGENDARY HUNT",),
    },
    "cod": {
        "terminal_win": ("VICTORY", "WINNER WINNER"),
        "elimination": ("ELIMINATED", "KILLED"),
        "knock": ("GULAG",),
        "rank_progress": ("CONTRACT COMPLETE", "LOADOUT DROP"),
        "generic_highlight": ("KILLSTREAK",),
    },
    "minecraft": {
        "terminal_win": ("ACHIEVEMENT GET",),
        "elimination": ("YOU DIED",),
        "generic_highlight": ("ENDER DRAGON",),
    },
    "gta": {
        "terminal_win": ("MISSION PASSED",),
        "elimination": ("WASTED", "BUSTED"),
    },
}

# Kept for backward compatibility with any external import of the old
# Fortnite-only constant name.
OCR_TRIGGER_TIERS = GAME_EVENT_TRIGGERS["fortnite"]


def _normalize_ocr_text(text: str) -> str:
    normalized = text.upper()
    normalized = normalized.replace("|", "I").replace("!", "I")
    normalized = re.sub(r"[^A-Z0-9# ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
    return re.search(pattern, text) is not None


# Tier priority order, matching fusion.GAME_TIER_WEIGHTS -- checked in this
# order so the strongest matching tier wins when text happens to match more
# than one (e.g. a banner mentioning both a rank and an elimination).
_TIER_ORDER = ("terminal_win", "elimination", "knock", "rank_progress", "generic_highlight")


def _resolve_trigger_dicts(game: str) -> List[Dict[str, tuple]]:
    """The generic bucket is always checked; a recognized game's dict is
    layered on top of it so an unrecognized/unlisted game still gets the
    broadly-common trigger phrases instead of no OCR event capability at all."""
    game = (game or "generic").strip().lower()
    dicts = [GAME_EVENT_TRIGGERS.get("generic", {})]
    if game != "generic" and game in GAME_EVENT_TRIGGERS:
        dicts.append(GAME_EVENT_TRIGGERS[game])
    return dicts


def classify_ocr_trigger(text: str, game: str = "generic") -> Optional[Dict[str, str]]:
    """Classify normalized OCR text into deterministic gameplay trigger tiers.

    ``game`` selects which per-game trigger vocabulary to check on top of the
    always-active "generic" bucket. This is normally the auto-detected game id
    from engines.reaction.game_detect -- never something the creator picks.
    """
    normalized = _normalize_ocr_text(text)
    if not normalized:
        return None

    compact = normalized.replace("0", "O")

    # Reject common non-event UI strings that previously masqueraded as
    # gameplay events in long Fortnite VODs. Kept universal (not game-gated):
    # a killfeed reporting who eliminated YOU, an assist notice, or a menu
    # stat screen reads as a false positive in any game phrased this way.
    compact_no_space = compact.replace(" ", "")
    if _contains_phrase(compact, "ELIMINATED BY") or "ELIMINATEDBY" in compact_no_space:
        return None
    if _contains_phrase(compact, "ELIMINATION ASSISTED"):
        return None
    if (
        (_contains_phrase(compact, "SELF REVIVE") or "SELFREVIVE" in compact_no_space)
        and (_contains_phrase(compact, "KNOCKED DOWN") or "KNOCKEDDOWN" in compact_no_space)
    ):
        return None

    # Fortnite's lobby/news tiles advertise cosmetics with the literal words
    # "Victory Royale". On a long VOD that persistent promo used to become a
    # terminal-win event every OCR sample, and terminal wins were then forced
    # into the review deck. A real result banner is visually isolated; lobby
    # captures carry navigation/promo context such as PLAY / SHOP / LOCKER or
    # the Battle Bus marketing copy observed in the failed founder scan.
    if "VICTORY" in compact and "ROYALE" in compact:
        fortnite_lobby_tokens = (
            "PLAY", "SHOP", "LOCKER", "SPRITES", "CAREER", "QUESTS",
            "BATTLE PASS", "DISCOVER",
        )
        lobby_token_count = sum(
            1 for token in fortnite_lobby_tokens
            if _contains_phrase(compact, token)
        )
        if _contains_phrase(compact, "NEW BATTLE BUS") or lobby_token_count >= 2:
            return None
        # Mode-objective HUD text: LTMs named after the banner print the win
        # words as a STANDING objective for the whole round (measured on a
        # real VOD: "BATTLE BUS VICTORY ROYALE 1:16 FIND THE BUS AND CLIMB
        # ABOARD TO ESCAPE AND WIN THE MATCH" fired terminal_win continuously,
        # anchoring "Match win" clips a full minute before the actual win).
        # Instructional context marks it: a real result banner never tells you
        # how to win. A single objective phrase vetoes the win read.
        objective_phrases = (
            "FIND THE BUS", "CLIMB ABOARD", "TO ESCAPE AND WIN",
            "AND WIN THE MATCH", "TO WIN THE MATCH", "ESCAPE THE ISLAND",
        )
        for phrase in objective_phrases:
            if (
                _contains_phrase(compact, phrase)
                or phrase.replace(" ", "") in compact_no_space
            ):
                return None

    for tier in _TIER_ORDER:
        for trigger_dict in _resolve_trigger_dicts(game):
            for trigger in trigger_dict.get(tier, ()):
                normalized_trigger = _normalize_ocr_text(trigger).replace("0", "O")
                if _contains_phrase(compact, normalized_trigger):
                    return {
                        "trigger": trigger,
                        "tier": tier,
                        "normalized_text": normalized,
                    }

    # OCR often drops the #1 prefix or confuses spacing on the Fortnite banner
    # badly enough that the exact-adjacency phrase match above misses it.
    # Left universal (not gated on game=="fortnite"): if detection missed and
    # fell back to generic but the banner really is Fortnite's, this is the
    # only thing that still catches it.
    if "VICTORY" in compact and "ROYALE" in compact:
        return {
            "trigger": "VICTORY ROYALE",
            "tier": "terminal_win",
            "normalized_text": normalized,
        }

    return None

def detect_base_events(window: List[UnifiedSignal]) -> List[GameEvent]:
    """Detect rule-based base events from a time window of signals."""
    if not window:
        return []
        
    start_ts = window[0].timestamp
    end_ts = window[-1].timestamp
    # Make sure end_ts > start_ts for proper ranges
    if end_ts == start_ts:
        end_ts += 1.0

    detected_events = []
    
    audio_signals = [s.audio for s in window if s.audio and s.audio.is_spike]
    if audio_signals:
        peak_audio = max(audio_signals, key=lambda sig: sig.spike_score)
        detected_events.append(GameEvent(
            timestamp_start=start_ts,
            timestamp_end=end_ts,
            event_type="audio_reaction_event",
            confidence=0.0,
            signals=["audio_reaction"],
            score=0.0,
            metadata={
                "peak_amplitude": peak_audio.amplitude,
                "baseline": peak_audio.baseline,
                "z_score": peak_audio.z_score,
                "spike_score": peak_audio.spike_score,
                "timestamps": [s.timestamp for s in audio_signals]
            }
        ))

    motion_signals = [
        s.vision for s in window
        if s.vision and s.vision.motion_score >= MOTION_THRESHOLD
    ]
    if motion_signals:
        peak_motion = max(motion_signals, key=lambda sig: sig.motion_score)
        detected_events.append(GameEvent(
            timestamp_start=start_ts,
            timestamp_end=end_ts,
            event_type="motion_action_event",
            confidence=0.0,
            signals=["motion_action"],
            score=0.0,
            metadata={
                "motion_score": peak_motion.motion_score,
                "timestamps": [s.timestamp for s in motion_signals]
            }
        ))

    trigger_hits = []
    for sig in window:
        if not sig.ocr or not sig.ocr.text:
            continue
        # This legacy event->story->clip path predates game auto-detection and
        # was only ever tuned against Fortnite content; pin it explicitly so
        # its narrow, exception-handler-only fallback behavior is unchanged.
        trigger = classify_ocr_trigger(sig.ocr.text, game="fortnite")
        if trigger:
            trigger_hits.append({
                **trigger,
                "timestamp": sig.timestamp,
                "confidence": sig.ocr.confidence,
                "raw_text": sig.ocr.metadata.get("raw_text", sig.ocr.text) if sig.ocr.metadata else sig.ocr.text,
            })

    if trigger_hits:
        best_tier = "generic_highlight"
        for tier in ("terminal_win", "elimination", "knock", "rank_progress", "generic_highlight"):
            if any(hit["tier"] == tier for hit in trigger_hits):
                best_tier = tier
                break

        detected_events.append(GameEvent(
            timestamp_start=start_ts,
            timestamp_end=end_ts,
            event_type="terminal_win_event" if best_tier == "terminal_win" else "ocr_trigger_event",
            confidence=0.0,
            signals=["ocr_text"],
            score=0.0,
            metadata={"triggers": trigger_hits, "ocr_tier": best_tier}
        ))
        
    return detected_events
