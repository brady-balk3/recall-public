# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Bounded, fully-local visual editorial judge for candidate clips.

The existing semantic judge reads transcript/OCR/signal evidence.  This module
adds the missing visual evidence without asking a VLM to watch an entire VOD:
six ordered keyframes are sampled around the candidate's reaction peak, then a
small local Qwen2.5-VL GGUF produces the same :class:`SemanticVerdict` contract
the reaction selector already understands.

Nothing downloads here.  ``resolve_models`` only enables the experiment when
both the language GGUF and multimodal projector are already present under
``models/vlm`` (or explicitly supplied through environment variables). Every
failure returns no verdict, so a weak/incompatible model cannot affect Recall's
production selector. This module is currently used only by the offline bakeoff.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
import re as _re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from core.bundle_paths import get_data_dir, get_models_dir
from core.device import get_torch_device
from engines.semantic import backend as text_backend
from engines.semantic import judge as text_judge
from engines.semantic.backend import (
    VERDICT_JSON_SCHEMA,
    MalformedCompletionError,
    family_rank,
    parse_completion,
)
from engines.semantic.verdict import SemanticVerdict


PROMPT_VERSION = "visual-editor-v8"
# Storage-schema stamp for VisualVerdictCache verdict entries, independent of
# prompt identity. Entries hold the PRISTINE model JSON; _calibrate_raw runs
# after every load, so cold scans and warm rescans calibrate exactly once from
# the same baseline. Any change to what an entry CONTAINS (as opposed to how
# the model was prompted) must bump this, not PROMPT_VERSION. Pre-"pristine"
# entries stored post-calibration JSON and must never be re-read: re-running
# calibration on its own output is not idempotent.
CACHE_SCHEMA_VERSION = "pristine-v1"
DEFAULT_FRAME_COUNT = 6
DEFAULT_MAX_CANDIDATES = 30
FRAME_MAX_SIDE = 512
# Facecam tiles (512 gameplay + 256px face crop from full-res) were tried on
# Qwen3.5-4B 2026-08-17, one labeled VOD: 9/15 -> 7/12, lost 2 keeps, 3 new
# cards all pass. Reverted. Bring them back for the omni judge so that model
# is not scored on a ~103x36 webcam inside the 512 frame.
JPEG_QUALITY = 78
VISUAL_N_CTX = 8192
# Fortnite HUD dumps used to truncate mid-string at 220 tokens
# ("Unterminated string … char 576") and drop ~19 reviews/scan.
VISUAL_MAX_TOKENS = 512
VISUAL_RETRY_MAX_TOKENS = 768

# Property ORDER is generation order: llama.cpp's schema-to-grammar keeps the
# listed order for required properties, so the model is forced to write its
# grounded observations (summary, banner text, outcome) BEFORE any editorial
# score and the verdict LAST. The v2 layout inherited the text-judge order and
# made the model commit to hook/payoff scores before describing the frames —
# observed to collapse a 4B model into all-zero "filler" verdicts.
VISUAL_VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        # --- Grounded reading first (perception, not taste). "onscreen_text"
        # transcribes any banner/feed text so the field is auditable; OCR
        # misses stylized banners (a founder VOD's VICTORY ROYALE banners were
        # never read) and the VLM recovers them trivially.
        "summary": {"type": "string", "maxLength": 120},
        "onscreen_text": {"type": "string", "maxLength": 80},
        "visible_outcome": {
            "type": "string",
            "enum": [
                "match_win", "elimination", "death_or_fail", "level_up",
                "menu_or_shop", "none",
            ],
        },
        "streamer_visible_reaction": {"type": "boolean"},
        "visual_event": {"type": "boolean"},
        "verbal_payoff": {"type": "boolean"},
        "context_complete": {"type": "boolean"},
        "routine_only": {"type": "boolean"},
        "visual_evidence": {"type": "string", "maxLength": 100},
        # --- Editorial read second, concluded from the observations above.
        "hook_strength": VERDICT_JSON_SCHEMA["properties"]["hook_strength"],
        "self_contained": VERDICT_JSON_SCHEMA["properties"]["self_contained"],
        "payoff": VERDICT_JSON_SCHEMA["properties"]["payoff"],
        "moment_type": VERDICT_JSON_SCHEMA["properties"]["moment_type"],
        "title": VERDICT_JSON_SCHEMA["properties"]["title"],
        "hook_line": VERDICT_JSON_SCHEMA["properties"]["hook_line"],
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "verdict": VERDICT_JSON_SCHEMA["properties"]["verdict"],
    },
    "required": [
        "summary", "onscreen_text", "visible_outcome",
        "streamer_visible_reaction", "visual_event", "verbal_payoff",
        "context_complete", "routine_only", "visual_evidence",
        "hook_strength", "self_contained", "payoff", "moment_type",
        "title", "hook_line", "confidence", "verdict",
    ],
}

_VISUAL_OUTCOME_VALUES = frozenset(
    VISUAL_VERDICT_JSON_SCHEMA["properties"]["visible_outcome"]["enum"]
)
_VISUAL_VERDICT_VALUES = frozenset(("post", "maybe", "skip"))


def _usable_verdict(raw) -> bool:
    """Does a free-form completion carry everything the schema guaranteed?

    The grammar used to make this structurally impossible to get wrong. Without
    it, verify explicitly: every required key present, the two enums in range,
    and the editorial floats numeric. A verdict that fails here is regenerated
    under the grammar, so selection never sees a half-filled read — and a model
    that silently drops ``visible_outcome`` can't quietly disable the win path.
    """
    if not isinstance(raw, dict):
        return False
    for key in VISUAL_VERDICT_JSON_SCHEMA["required"]:
        if key not in raw:
            return False
    if str(raw.get("verdict", "")).strip().lower() not in _VISUAL_VERDICT_VALUES:
        return False
    if str(raw.get("visible_outcome", "")).strip().lower() not in _VISUAL_OUTCOME_VALUES:
        return False
    for key in ("hook_strength", "self_contained", "payoff", "confidence"):
        try:
            float(raw.get(key))
        except (TypeError, ValueError):
            return False
    return True


def resolve_models() -> Optional[Tuple[str, str]]:
    """Return ``(model, mmproj)`` when a complete local VLM pair exists."""
    explicit_model = os.environ.get("RECALL_VLM_MODEL")
    explicit_mmproj = os.environ.get("RECALL_VLM_MMPROJ")
    if explicit_model or explicit_mmproj:
        if explicit_model and explicit_mmproj \
                and os.path.isfile(explicit_model) and os.path.isfile(explicit_mmproj):
            return os.path.abspath(explicit_model), os.path.abspath(explicit_mmproj)
        return None

    vlm_dir = os.path.join(get_models_dir(), "vlm")
    if not os.path.isdir(vlm_dir):
        return None
    pairs = []
    for dirpath, _dirs, names in os.walk(vlm_dir):
        files = sorted(
            os.path.join(dirpath, name)
            for name in names if name.lower().endswith(".gguf")
        )
        projectors = [
            path for path in files
            if os.path.basename(path).lower().startswith("mmproj-")
        ]
        models = [path for path in files if path not in projectors]
        if models and projectors:
            pairs.append((models[0], projectors[0]))
    if not pairs:
        return None

    def preference(pair):
        label = " ".join(pair).lower()
        # Prefer the newest known family when several experimental model pairs
        # are present; see backend.family_rank for why a plain sort is wrong.
        # Explicit environment paths still override this.
        return family_rank(label), label

    return sorted(pairs, key=preference)[0]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def judge_window(candidate: dict) -> Tuple[float, float]:
    """The editorial window the VLM must evaluate.

    Candidate scoring remains anchored to the compact reaction arc, but the
    visual judge should see the same sentence/action-complete cut a creator will
    receive. Invalid or absent prepared bounds fall back to the candidate span.
    """
    start = max(0.0, float(candidate.get("start", 0.0) or 0.0))
    end = max(start + 0.1, float(candidate.get("end", start + 0.1) or start + 0.1))
    try:
        prepared_start = max(
            0.0, float(candidate.get("judge_start", start) or start),
        )
        prepared_end = float(candidate.get("judge_end", end) or end)
    except (TypeError, ValueError):
        return start, end
    peak = float(candidate.get("peak_timestamp", start) or start)
    if prepared_end <= prepared_start or not prepared_start <= peak <= prepared_end:
        return start, end
    return prepared_start, prepared_end


def keyframe_timestamps(candidate: dict, frame_count: int = DEFAULT_FRAME_COUNT) -> List[float]:
    """Ordered setup/action/aftermath samples, concentrated around the peak."""
    start, end = judge_window(candidate)
    duration = end - start
    peak = float(candidate.get("peak_timestamp", start + duration * 0.55) or 0.0)
    if not start <= peak <= end:
        peak = start + duration * 0.55

    count = max(2, min(8, int(frame_count)))
    if count == 2:
        raw = [start + 0.5, peak]
    else:
        # Anchor the FIRST sample just inside the window: when a clip captures
        # the tail of a payoff (win banner, kill feed), the banner lives in the
        # opening seconds and a (i+1)/(n+1) spread steps right over it — a
        # founder win clip's VICTORY ROYALE banner at start+0.5s was missed by
        # every sample. The rest spread uniformly, with the two center samples
        # replaced by peak-adjacent frames for quick mid-window actions.
        raw = [start + min(0.5, duration * 0.05)]
        raw += [start + duration * (i + 1) / count for i in range(count - 1)]
        center = count // 2
        raw[max(1, center - 1)] = peak - min(1.0, duration * 0.08)
        raw[center] = peak
        # A game anchor (OCR or sweep-detected banner) is the payoff the judge
        # must SEE. Prefer the exact frame the sweep classified as a win (the
        # banner is certainly visible there); an onset estimate can land a few
        # seconds before the banner draws. Fall back to anchor + 1s.
        anchor_ts = None
        witnessed = candidate.get("visual_win_frame_time")
        if witnessed is not None:
            try:
                anchor_ts = float(witnessed)
            except (TypeError, ValueError):
                anchor_ts = None
        if anchor_ts is None:
            anchor = candidate.get("game_anchor_time")
            if anchor is not None:
                try:
                    anchor_ts = float(anchor) + 1.0
                except (TypeError, ValueError):
                    anchor_ts = None
        if anchor_ts is not None and start <= anchor_ts <= end:
            raw[min(len(raw) - 1, center + 1)] = anchor_ts

    safe_lo = start + min(0.15, duration * 0.02)
    safe_hi = end - min(0.15, duration * 0.02)
    values = sorted(_clamp(ts, safe_lo, safe_hi) for ts in raw)
    # Very short candidates can collapse samples onto the same timestamp.
    unique: List[float] = []
    for value in values:
        rounded = round(value, 3)
        if not unique or abs(rounded - unique[-1]) >= 0.05:
            unique.append(rounded)
    return unique


@dataclass(frozen=True)
class FrameSample:
    timestamp: float
    data_url: str


class VideoFrameExtractor:
    """One reusable OpenCV handle for all candidate seeks in a VOD."""

    def __init__(self, video_path: str, max_side: int = FRAME_MAX_SIDE):
        self.video_path = video_path
        self.max_side = max(128, int(max_side))
        self._capture = None

    def _open(self):
        if self._capture is not None:
            return self._capture
        import cv2

        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video for visual judge: {self.video_path}")
        self._capture = capture
        return capture

    def extract(self, timestamps: Sequence[float]) -> List[FrameSample]:
        import cv2

        capture = self._open()
        samples: List[FrameSample] = []
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            longest = max(height, width)
            if longest > self.max_side:
                scale = self.max_side / float(longest)
                frame = cv2.resize(
                    frame,
                    (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not ok:
                continue
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            samples.append(FrameSample(
                timestamp=float(timestamp),
                data_url=f"data:image/jpeg;base64,{payload}",
            ))
        return samples

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def _visual_prompt(
    candidate: dict,
    window_words: Sequence[str],
    chat_lines: Sequence[str],
    frame_samples: Sequence[FrameSample],
    game_context: Optional[str] = None,
    *,
    disagreement_rejudge: bool = False,
    strict_verbal_rejudge: bool = False,
) -> str:
    start = float(candidate.get("start", 0.0) or 0.0)
    end = float(candidate.get("end", start) or start)
    frame_order = ", ".join(
        f"frame {i + 1}=+{sample.timestamp - start:.1f}s"
        for i, sample in enumerate(frame_samples)
    )
    lines = [
        "You are a strict highlights editor reviewing one candidate from a live gaming stream.",
        "The attached frames are chronological samples from setup through aftermath.",
        "Judge BOTH the visible action and the spoken arc. The frames verify setting/action;",
        "the transcript carries jokes, stories, escalating dialogue, scares, and reactions.",
        "A compelling verbal bit can be postable even when the visuals are routine gameplay.",
        "Story climaxes count: character death/execution, betrayal, irreversible choices,",
        "identity reveals, farewell/sacrifice, or on-screen story banners resolving a beat.",
        "A postable clip needs a recognizable event, joke, story beat, skill play, failure,",
        "surprise, or emotional payoff. Reject routine gameplay, menus, travel, waiting,",
        "unresolved buildup, and generic excited talking with no visible or verbal payoff.",
        "Do not invent action between frames. If the evidence does not establish why a cold",
        "viewer would care, choose maybe or skip and lower confidence.",
        "Do not reject a complete spoken setup/payoff merely because sampled frames miss",
        "fast action. Do reject descriptive chatter that never becomes a standalone beat.",
        "Judge the complete moment, not image attractiveness.",
    ]
    if disagreement_rejudge:
        lines.extend([
            "",
            "DISAGREEMENT RE-JUDGE: a prior pass said skip, but transcript / onscreen text /",
            "prior description points at a resolving climax (death, execution, betrayal,",
            "irreversible choice, farewell). Re-observe denser frames carefully. Prefer",
            "maybe/post when the beat resolves; keep skip only for true filler with no",
            "resolving visual or spoken payoff. Do not invent character names or lore.",
        ])
    lines.extend([
        "",
        f"Window: {max(0.0, end - start):.1f}s; {frame_order}",
    ])
    if game_context:
        lines.append(f"Game/context label: {game_context}")
    lines.extend([
        f"Measured signals: {text_judge._signal_summary(candidate)}",
        "Transcript: " + (" ".join(window_words) if window_words else "(no clear speech)"),
    ])
    if chat_lines:
        lines.append("Chat: " + " | ".join(chat_lines))
    lines.extend([
        "",
        "Return JSON only. Observe FIRST, then judge:",
        "- summary: one factual sentence describing what actually happens.",
        "- onscreen_text: ONLY the main banner/announcement (e.g. 'VICTORY ROYALE #1',",
        "  'RIP …'). Do NOT dump chat, kill-feed spam, or HUD chrome. Empty if none.",
        "- visible_outcome: the game state outcome LITERALLY shown in any frame:",
        '  "match_win" (victory banner/win screen/#1 placement FOR THIS streamer),',
        '  "elimination" (kill/knock feed for the player), "death_or_fail"',
        "  (death/defeat screen, character shot/killed/executed, fail state),",
        '  "level_up" (rank/level/reward screen),',
        '  "menu_or_shop" (lobby, item shop, settings), or "none".',
        "  A victory screen while SPECTATING another player (spectator UI, someone",
        '  else\'s name on the banner, streamer already eliminated) is "none".',
        "  Read banners carefully; report what is written.",
        "- streamer_visible_reaction: true only when the facecam shows a clear",
        "  expression change (laugh, shock, celebration) rather than a neutral face.",
        "- visual_event: true only when the ordered frames establish meaningful action/change.",
        "  If visual_event is true, routine_only must be false.",
        "- verbal_payoff: true only when the transcript contains a complete joke, reveal,",
        "  scare, story beat, argument, decision, or escalating reaction that resolves.",
        "  Judge this from the TRANSCRIPT above, not the frames. Chat thank-you / sub",
        "  / follow talk is NOT a verbal payoff.",
        "- context_complete: true when a cold viewer can follow why the beat matters.",
        "- routine_only: true when this is only ordinary play/chat/travel/menu activity",
        "  AND the transcript carries no complete beat either. Never true when",
        "  visual_event is true or onscreen_text shows a resolving story/death banner.",
        "- visual_evidence: one short phrase for the visible setup/payoff (keep brief).",
        "- hook_strength, self_contained, payoff (0-1 each).",
        "- moment_type: funny, clutch, fail, rage, scare, story, wholesome, or filler.",
        "  Use story for narrative climaxes and fail for deaths/botches.",
        "- title (max 8 words) and hook_line (max 7 words).",
        "- confidence (0-1).",
        '- verdict: "post" only when clearly publishable, "maybe" when promising but',
        '  context/evidence is incomplete, or "skip" for filler.',
    ])
    if strict_verbal_rejudge:
        lines.extend([
            "",
            "COLD-VIEWER VERBAL PAYOFF RE-JUDGE:",
            "Topic and context are not payoff. A complete sentence, confident opinion,",
            "or understandable answer can still be ordinary conversation with no clip ending.",
            "Preference games (would-you-rather / this-or-that), isolated chat answers,",
            "descriptive commentary, advice, and anecdotes that merely stop are not story",
            "beats unless this window contains a turn: punchline, reveal, reversal,",
            "consequence, decision with stakes, or escalating reaction that lands.",
            "Judge only what resolves inside this window. Do not infer a later payoff.",
            "If removing the last third would not remove a meaningful landing, set",
            "verbal_payoff=false and payoff<=0.3. With no independent visual event or",
            "visible reaction, also use routine_only=true, moment_type=filler, verdict=skip.",
        ])
    return "\n".join(lines)


# Grounded story/death banners the VLM often transcribes then ignores.
# Keep this cross-title (RIP / game over / chapter cards) — not game lore.
_STORY_BANNER_RE = _re.compile(
    r"\b(?:"
    r"rip\b|game\s*over|you\s+died|mission\s+(?:failed|complete)|"
    r"chapter\b|ending\b|executed|assassinated"
    r")",
    _re.I,
)
# Resolving spoken beats — decisions, reveals, irreversible stakes.
# Lexical, but title-agnostic; chat-thanks are excluded below.
_VERBAL_RESOLVE_RE = _re.compile(
    r"\b(?:"
    r"it'?s\s+time|it'?s\s+happening|i\s+am\b|i'?m\s+not\b|"
    r"we(?:'re|\s+are)\s+never|never\s+going\s+back|"
    r"i\s+(?:won'?t|will|choose|decide)|"
    r"decide(?:d)?\b|betray|sacrifice|goodbye\b|farewell\b|"
    r"he(?:'s|\s+is)\s+dead|she(?:'s|\s+is)\s+dead|they(?:'re|\s+are)\s+dead|"
    r"shot\s+(?:him|her|me|them)|kill(?:ed)?\s+(?:him|her|me|them)"
    r")",
    _re.I,
)
_CHAT_THANKS_RE = _re.compile(
    r"\b(?:thanks?|thank\s+you|appreciate|subscri(?:be|ber)?s?|"
    r"follow(?:ers)?|donat(?:e|ion)|bits?\b|raid(?:ed)?|"
    r"gift(?:ed)?\s+sub)\b",
    _re.I,
)
# Deterministic game-state text that means the frames show waiting rather than
# gameplay.  Keep this deliberately narrow: it guards a generic transcript
# rescue, not the model's own ability to recognize a complete joke/story during
# a lobby. Exact-word matching avoids treating "reloading" as "loading".
_ROUTINE_NON_GAMEPLAY_RE = _re.compile(
    r"\b(?:"
    r"loading|connecting|waiting\s+for\s+players|matchmaking|"
    r"finding\s+match|searching\s+for\s+match|main\s+menu"
    r")\b",
    _re.I,
)
# Title-agnostic climax language in the model's own text fields / OCR.
# Deliberately avoids bare BR verbs (eliminated, kill feed) that would
# false-trigger Fortnite ordinary fights into a re-judge loop.
_CLIMAX_TEXT_RE = _re.compile(
    r"\b(?:"
    r"executed|assassinated|betray(?:al|ed|s)?|sacrifice(?:d|s)?|"
    r"goodbye|farewell|irreversible|"
    r"(?:final|last)\s+(?:choice|decision|moment|scene)|"
    r"(?:is|are|was|were)\s+dead|"
    r"shot\s+(?:and\s+)?(?:killed|dead)|"
    r"kill(?:s|ed)?\s+(?:him|her|them)\b|"
    r"game\s*over|you\s+died|\brip\b|"
    r"chapter\s+(?:end|ending)|ending\s+scene|"
    r"mission\s+failed"
    r")",
    _re.I,
)
# Extra denser-frame passes when text/content says climax but the first
# visual read still hard-skips after calibration (cold facecam / sparse
# samples). Separate from the pool budget so Fortnite decks are not starved.
FALSE_SKIP_REJUDGE_BUDGET = 5
FALSE_SKIP_REJUDGE_FRAMES = 8
VERBAL_PAYOFF_REJUDGE_BUDGET = 5
VERBAL_REJUDGE_MIN_CHAT = 0.5
VERBAL_TEXT_CONSENSUS_MIN_CHAT = 10.0


def _onscreen_story_banner(text: str) -> bool:
    blob = str(text or "").strip()
    if len(blob) < 3:
        return False
    return bool(_STORY_BANNER_RE.search(blob))


def _transcript_verbal_payoff(window_words: Optional[Sequence[str]]) -> bool:
    """Detect a resolving spoken beat the VLM often marks verbal_payoff=false.

    Uses the same transcript the prompt already supplies. Chat thank-you /
    sub / follow talk is explicitly not a payoff.
    """
    if not window_words:
        return False
    text = " ".join(str(w) for w in window_words if str(w).strip())
    if len(text.split()) < 6:
        return False
    if _CHAT_THANKS_RE.search(text) and not _VERBAL_RESOLVE_RE.search(text):
        return False
    if _CHAT_THANKS_RE.search(text):
        # Mixed windows: require resolve language to outweigh chat-thanks.
        thanks_hits = len(_CHAT_THANKS_RE.findall(text))
        resolve_hits = len(_VERBAL_RESOLVE_RE.findall(text))
        if thanks_hits >= resolve_hits:
            return False
    return bool(_VERBAL_RESOLVE_RE.search(text))


def _routine_non_gameplay_state(raw: dict) -> bool:
    """Whether grounded visual text says this is a waiting/menu state."""
    outcome = str(raw.get("visible_outcome", "") or "").strip().lower()
    if outcome == "menu_or_shop":
        return True
    onscreen = str(raw.get("onscreen_text", "") or "")
    return bool(_ROUTINE_NON_GAMEPLAY_RE.search(onscreen))


def _climax_text_blob(raw: dict) -> str:
    return " ".join(
        str(raw.get(key, "") or "")
        for key in ("summary", "visual_evidence", "title", "onscreen_text", "hook_line")
    )


def _content_climax_signal(
    raw: dict,
    window_words: Optional[Sequence[str]] = None,
) -> bool:
    """Text/content evidence that a resolving climax is present.

    Used to detect false visual skips (model or cache said skip while the
    transcript, banner, or the model's own description points at a climax).
    Title-agnostic — no character/game lore.
    """
    if _onscreen_story_banner(str(raw.get("onscreen_text", "") or "")):
        return True
    outcome = str(raw.get("visible_outcome", "") or "").strip().lower()
    if outcome == "death_or_fail":
        return True
    if _routine_non_gameplay_state(raw):
        # A loading/menu screen plus a broad lexical hit ("I am", "I'm not")
        # is not enough to spend a false-skip rejudge. Preserve the exception
        # only when the visual model itself authored a complete spoken beat.
        return (
            bool(raw.get("verbal_payoff", False))
            and bool(raw.get("context_complete", False))
        )
    if _transcript_verbal_payoff(window_words):
        return True
    if _CLIMAX_TEXT_RE.search(_climax_text_blob(raw)):
        return True
    moment = str(raw.get("moment_type", "") or "").strip().lower()
    if moment in ("story", "fail") and not bool(raw.get("routine_only", False)):
        # Model typed a climax class then contradicted itself with skip.
        return True
    return False


def _false_skip_disagreement(
    raw: dict,
    window_words: Optional[Sequence[str]] = None,
) -> bool:
    """True when calibrated verdict is still skip but content says climax."""
    if str(raw.get("verdict", "")).lower() != "skip":
        return False
    return _content_climax_signal(raw, window_words)


def _optimistic_verbal_story_disagreement(candidate: dict, raw: dict) -> bool:
    """Whether a generic story claim rests entirely on an optimistic transcript read."""
    profile = str(candidate.get("game_profile") or "").strip().lower()
    semantic_verdict = str(candidate.get("semantic_verdict") or "").strip().lower()
    outcome = str(raw.get("visible_outcome") or "").strip().lower()
    return (
        profile == "generic"
        and not candidate.get("game_label")
        and semantic_verdict in ("skip", "maybe")
        and str(raw.get("verdict") or "").strip().lower() in ("post", "maybe")
        and str(raw.get("moment_type") or "").strip().lower() == "story"
        and bool(raw.get("verbal_payoff"))
        and bool(raw.get("context_complete"))
        and not bool(raw.get("visual_event"))
        and not bool(raw.get("streamer_visible_reaction"))
        and outcome in ("", "none")
    )


def _strict_verbal_unsupported(raw: dict) -> bool:
    """Whether the cold-viewer pass found no standalone verbal landing."""
    verdict = str(raw.get("verdict") or "").strip().lower()
    moment = str(raw.get("moment_type") or "").strip().lower()
    outcome = str(raw.get("visible_outcome") or "").strip().lower()
    grounded_visual = (
        bool(raw.get("visual_event"))
        or bool(raw.get("streamer_visible_reaction"))
        or outcome not in ("", "none")
    )
    return (
        verdict == "skip"
        or moment == "filler"
        or (
            not grounded_visual
            and (
                not bool(raw.get("verbal_payoff"))
                or not bool(raw.get("context_complete"))
            )
        )
    )


def _promote_skip_to_maybe(value: dict, *, outcome: str = "") -> dict:
    """Change verdict authority without inventing model confidence.

    The old rescue also synthesized 0.55 payoff, self-containment, and
    confidence. Those fabricated values then satisfied downstream gates as if
    the VLM had observed them. Preserve every decomposition score exactly as the
    judge authored it; a promoted ``maybe`` still has to earn selection through
    independent evidence.
    """
    value["verdict"] = "maybe"
    if value.get("moment_type") == "filler":
        value["moment_type"] = (
            "fail" if outcome == "death_or_fail" else "story"
        )
    return value


def _grounded_skip_promotion(raw: dict) -> bool:
    """Whether non-lexical evidence may overturn a model-authored skip."""
    outcome = str(raw.get("visible_outcome", "") or "").strip().lower()
    onscreen = str(raw.get("onscreen_text", "") or "")
    model_verbal = bool(raw.get("verbal_payoff", False)) and not bool(
        raw.get("verbal_payoff_from_transcript", False)
    )
    return bool(
        _onscreen_story_banner(onscreen)
        or outcome == "death_or_fail"
        or (
            bool(raw.get("visual_event", False))
            and bool(raw.get("verbal_payoff", False))
        )
        or (model_verbal and bool(raw.get("context_complete", False)))
    )


def _calibrate_raw(
    raw: dict,
    window_words: Optional[Sequence[str]] = None,
) -> dict:
    """Apply only logically explicit rubric constraints to the model output.

    This is not a hidden quality rule: the booleans are model-authored
    decomposition fields (plus the transcript/onscreen text the model was
    asked to read). They prevent contradictory JSON such as
    ``routine_only=true`` + visual_event/story banner, and keep a clearly
    complete verbal beat from becoming a destructive visual veto.
    """
    value = dict(raw)
    # Input is always the pristine model JSON: cache entries store the
    # pre-calibration output (CACHE_SCHEMA_VERSION) and calibration runs once
    # per load, so the derived markers written below never round-trip back in
    # as evidence. model_verdict still prefers an existing value so an
    # accidental re-application cannot clobber the model's own verdict.
    prior_model = value.get("model_verdict")
    verdict = str(prior_model or value.get("verdict", "")).lower()
    if prior_model is None:
        value["model_verdict"] = verdict

    visual_event = bool(value.get("visual_event", False))
    verbal_payoff = bool(value.get("verbal_payoff", False))
    context_complete = bool(value.get("context_complete", False))
    routine_only = bool(value.get("routine_only", False))
    outcome = str(value.get("visible_outcome", "") or "").strip().lower()
    onscreen = str(value.get("onscreen_text", "") or "")
    story_banner = _onscreen_story_banner(onscreen)
    routine_non_gameplay = _routine_non_gameplay_state(value)
    # The lexical fallback is intentionally broad for sparse narrative games.
    # Do not let it overturn a visual-model skip on an explicit loading/menu
    # state; the model's own verbal_payoff remains authoritative for a genuine
    # lobby joke or story.
    transcript_payoff = (
        _transcript_verbal_payoff(window_words)
        if not routine_non_gameplay or verbal_payoff
        else False
    )

    if transcript_payoff and not verbal_payoff:
        verbal_payoff = True
        value["verbal_payoff"] = True
        value["verbal_payoff_from_transcript"] = True
    if story_banner or visual_event or verbal_payoff:
        if routine_only:
            routine_only = False
            value["routine_only"] = False
    if story_banner and outcome in ("", "none"):
        # RIP / death / story title cards are fail/story outcomes the enum
        # already supports; the 4B often leaves them as none.
        if _re.search(r"\b(?:rip|game\s*over|you\s+died|executed)\b", onscreen, _re.I):
            value["visible_outcome"] = "death_or_fail"
            outcome = "death_or_fail"
    if story_banner:
        if not context_complete:
            value["context_complete_from_calibration"] = True
        context_complete = True
        value["context_complete"] = True
    elif verbal_payoff and (visual_event or transcript_payoff):
        if not context_complete:
            value["context_complete_from_calibration"] = True
        context_complete = True
        value["context_complete"] = True

    if routine_only and not visual_event and not verbal_payoff and not story_banner:
        value["verdict"] = "skip"
        value["moment_type"] = "filler"
        try:
            value["payoff"] = min(0.3, float(value.get("payoff", 0.0)))
        except (TypeError, ValueError):
            value["payoff"] = 0.0
    elif verdict == "skip" and _grounded_skip_promotion(value):
        # Sparse frames / cold facecam often yield skip on real climaxes.
        # Promote to maybe from grounded evidence; never jump straight to post.
        _promote_skip_to_maybe(value, outcome=outcome)
    return value


class VisualLlamaBackend:
    """Lazy llama.cpp vision backend; only one local model is resident.

    Model-agnostic by design: the chat handler is chosen from what the GGUF
    itself carries, not from a family hardcoded here, so a newer VLM dropped
    into ``models/vlm`` runs under its own prompt format.
    """

    def __init__(self, model_path: str, mmproj_path: str):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self._llm = None
        self._chat_handler_obj = None
        self.handler_kind = ""

    def _chat_handler(self):
        """Prefer llama.cpp's generic mtmd handler, which renders the model's
        OWN ``tokenizer.chat_template`` from GGUF metadata and dispatches the
        vision tower through mtmd. The legacy Qwen25VLChatHandler hardcodes a
        Qwen2.5-VL prompt layout, so it silently mis-formats every other
        family — including the Qwen3-VL lane it was previously loading.

        Fall back to the legacy handler only for a GGUF with no embedded chat
        template (older community converts), where mtmd cannot build a prompt.
        That has to be decided BEFORE the model loads: MTMDChatHandler only
        validates the projector path in its constructor and does not read
        tokenizer.chat_template until the first completion, so a constructor
        guard would let a template-less model fail once per candidate instead.
        """
        from llama_cpp.llama_chat_format import MTMDChatHandler, Qwen25VLChatHandler

        if self._has_chat_template():
            self.handler_kind = "mtmd"
            return MTMDChatHandler(clip_model_path=self.mmproj_path, verbose=False)
        print(
            "Visual judge model carries no chat template; "
            "falling back to the legacy Qwen2.5-VL prompt format."
        )
        self.handler_kind = "qwen25vl-legacy"
        return Qwen25VLChatHandler(clip_model_path=self.mmproj_path, verbose=False)

    def _has_chat_template(self) -> bool:
        """Whether the GGUF embeds its own chat template.

        ``vocab_only`` reads metadata without allocating the weights, so this
        costs a fraction of a second and no VRAM. An unreadable probe prefers
        mtmd: it is the general path, and its failure is loud rather than a
        silently mis-formatted prompt.
        """
        from llama_cpp import Llama

        probe = None
        try:
            probe = Llama(model_path=self.model_path, vocab_only=True, verbose=False)
            template = probe.metadata.get("tokenizer.chat_template")
            return isinstance(template, str) and bool(template.strip())
        except Exception as exc:  # noqa: BLE001 - probe is advisory only
            print(f"Visual judge chat-template probe failed ({exc}); assuming mtmd.")
            return True
        finally:
            if probe is not None:
                try:
                    probe.close()
                except Exception:  # noqa: BLE001
                    pass

    def _load(self):
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama

        handler = self._chat_handler()
        # Keep our own reference: close() has to free the handler's vision
        # tower, and Llama neither owns nor releases it (see close()).
        self._chat_handler_obj = handler
        logical = os.cpu_count() or 4
        threads = max(2, min(8, logical // 2))
        self._llm = Llama(
            model_path=self.model_path,
            chat_handler=handler,
            n_ctx=VISUAL_N_CTX,
            n_batch=512,
            n_gpu_layers=-1 if get_torch_device() == "cuda" else 0,
            n_threads=threads,
            flash_attn=get_torch_device() == "cuda",
            verbose=False,
        )
        self._llm.set_cache(None)
        return self._llm

    def generate_sweep(self, prompt: str, frames: Sequence[FrameSample], count: int) -> dict:
        """One constrained completion classifying several independent frames."""
        content = [{"type": "text", "text": prompt}]
        for i, frame in enumerate(frames):
            content.extend([
                {"type": "text", "text": f"\nFrame {i + 1}:"},
                {"type": "image_url", "image_url": {"url": frame.data_url}},
            ])
        result = self._load().create_chat_completion(
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_object",
                "schema": _sweep_schema(count),
            },
            temperature=0.1,
            max_tokens=SWEEP_MAX_TOKENS,
        )
        return parse_completion(result["choices"][0]["message"]["content"])

    def generate(self, prompt: str, frames: Sequence[FrameSample]) -> dict:
        content = [{"type": "text", "text": prompt}]
        for i, frame in enumerate(frames):
            content.extend([
                {"type": "text", "text": f"\nChronological frame {i + 1}:"},
                {"type": "image_url", "image_url": {"url": frame.data_url}},
            ])
        # Grammar-free FIRST. Measured on this VOD set: schema-constrained
        # sampling decodes at 38.5 tok/s vs 108.4 tok/s unconstrained (the
        # GBNF stack is checked against Qwen3's ~151k vocab every token), so
        # the grammar — not image encode (9 ms/frame) or prefill (0.45s) — is
        # most of the 3.8s/candidate visual-judge cost. The prompt already
        # enumerates the fields in schema order, so an unconstrained
        # completion keeps the observations-before-verdict layout the 4B model
        # needs; _usable_verdict verifies that per call rather than trusting
        # it. Anything malformed, incomplete, or out-of-enum falls back to the
        # constrained path below, so the guarantee is unchanged.
        try:
            raw = self._complete_verdict(content, VISUAL_MAX_TOKENS, constrained=False)
            if _usable_verdict(raw):
                return raw
            print("Visual judge free-form verdict incomplete; retrying constrained.")
        except MalformedCompletionError:
            pass
        try:
            return self._complete_verdict(content, VISUAL_MAX_TOKENS, constrained=True)
        except MalformedCompletionError:
            # Truncation mid-string on dense HUD frames (Fortnite). One wider
            # retry; still never blocks the scan if both attempts fail.
            print(
                "Visual judge completion malformed; regenerating once "
                f"with max_tokens={VISUAL_RETRY_MAX_TOKENS}."
            )
            return self._complete_verdict(
                content, VISUAL_RETRY_MAX_TOKENS, constrained=True,
            )

    def _complete_verdict(
        self, content: list, max_tokens: int, *, constrained: bool = True,
    ) -> dict:
        kwargs = {}
        if constrained:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": VISUAL_VERDICT_JSON_SCHEMA,
            }
        result = self._load().create_chat_completion(
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=max_tokens,
            **kwargs,
        )
        raw = parse_completion(result["choices"][0]["message"]["content"])
        if not isinstance(raw, dict):
            raise MalformedCompletionError(
                "visual judge completion was not a JSON object",
                raw_text=str(raw),
            )
        return raw

    def close(self) -> None:
        """Release BOTH GPU allocations this backend makes.

        ``Llama.close()`` frees the language weights, and that is all it frees.
        llama-cpp's chat handlers (MTMDChatHandler and the legacy
        Llava15/Qwen25VL pair) allocate the mmproj vision tower separately, on
        the FIRST image completion rather than at construction, and register
        its free() on a private ``_exit_stack``. Neither class defines close()
        or __del__, so nothing ever unwinds that stack: the tower outlives the
        Llama instance, the scan, and every later scan in the same process.

        MEASURED 2026-08-14 (RTX 4070 Ti SUPER, qwen3.5-4b + mmproj-F16):
        weights +3408 MiB (all returned by Llama.close()), first image
        completion +888 MiB, of which close() returned 10 MiB -- 878 MiB
        retained per backend, i.e. per scan. Six scans left ~5.3 GB of the
        card allocated to a process sitting idle, which is what shrank later
        jobs' GPU worker pools (7 -> 4) and what the user felt as frame drops
        in games with Recall merely open.

        Order matters: the mtmd context is initialized FROM the llama model
        (``mtmd_init_from_file(..., llama_model.model, ...)``), so it is
        released first, before the model it was built against.
        """
        handler, self._chat_handler_obj = self._chat_handler_obj, None
        llm, self._llm = self._llm, None
        try:
            if handler is not None:
                self._free_chat_handler(handler)
        finally:
            if llm is not None:
                llm.close()

    @staticmethod
    def _free_chat_handler(handler) -> None:
        """Unwind a llama-cpp chat handler's GPU allocations, best-effort.

        Prefers a real ``close()`` so this keeps working if llama-cpp grows
        one; falls back to the private ``_exit_stack`` it actually ships
        today. Cleanup must never propagate -- a scan that produced clips is
        finished, and failing to free VRAM cannot be allowed to fail the job.
        """
        closer = getattr(handler, "close", None)
        if callable(closer):
            try:
                closer()
                return
            except Exception:  # noqa: BLE001 - fall through to the exit stack
                pass
        stack = getattr(handler, "_exit_stack", None)
        if stack is None:
            print(
                "Visual judge: chat handler exposes no close() or _exit_stack; "
                "its vision tower may stay resident until the process exits."
            )
            return
        try:
            stack.close()
        except Exception as exc:  # noqa: BLE001 - never fail a finished scan
            print(f"Visual judge: could not free the vision tower ({exc}).")


class TextRoleBackend:
    """Serve the TEXT judge from the vision lane's already-resident instance.

    Both judges run in the same scan (engines/reaction/reaction_main.py calls
    the text judge, then the visual judge, then both again for deck coverage),
    so loading a second GGUF meant ~2.5 GB of weights resident purely to read
    transcripts. A natively-multimodal model answers text-only prompts on the
    same instance — verified: text-only constrained completion, bare-string
    content, and a vision call afterwards all succeed with no state carried
    between modes.

    Exposes exactly the ``generate``/``generate_batch`` surface
    judge.judge_candidates() duck-types, so the text judge is unaware it is
    sharing. Deliberately does NOT own the instance: ``close`` is a no-op
    because VisualJudgeSession owns the lifecycle, and closing the shared
    model here would pull it out from under the visual judge mid-scan.
    """

    def __init__(self, visual_backend: "VisualLlamaBackend"):
        self.visual_backend = visual_backend

    @property
    def model_path(self) -> str:
        return self.visual_backend.model_path

    def _complete(self, prompt: str, schema: dict, max_tokens: int) -> dict:
        llm = self.visual_backend._load()
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object", "schema": schema},
            temperature=text_backend.TEMPERATURE,
            max_tokens=max_tokens,
        )
        return parse_completion(result["choices"][0]["message"]["content"])

    def generate(self, prompt: str) -> dict:
        return self._complete(
            prompt, VERDICT_JSON_SCHEMA, text_backend.MAX_TOKENS,
        )

    def generate_batch(self, prompt: str, count: int) -> list:
        raw = self._complete(
            prompt,
            text_backend._batch_verdict_schema(count),
            72 * max(1, count),
        )
        return text_backend.map_triage_verdicts(raw)

    def close(self) -> None:
        """No-op: VisualJudgeSession owns the shared instance."""


class VisualVerdictCache:
    """Small, non-executable JSON cache so VOD comparisons are resumable."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(get_data_dir(), "cache", "visual_judge")

    def key(
        self,
        video_path: str,
        candidate: dict,
        model_path: str,
        frame_count: int,
        *,
        pass_tag: str = "",
    ) -> str:
        def stamp(path: str) -> str:
            try:
                stat = os.stat(path)
                return f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}"
            except OSError:
                return os.path.abspath(path)

        raw = "|".join([
            PROMPT_VERSION,
            CACHE_SCHEMA_VERSION,
            stamp(video_path),
            stamp(model_path),
            str(frame_count),
            str(pass_tag or ""),
            f"{judge_window(candidate)[0]:.3f}",
            f"{judge_window(candidate)[1]:.3f}",
            f"{float(candidate.get('peak_timestamp', 0.0) or 0.0):.3f}",
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def load(self, key: str) -> Optional[dict]:
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def save(self, key: str, value: dict) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"{key}.json")
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def _attach_visual_fields(candidate: dict, verdict: SemanticVerdict, raw: dict) -> None:
    candidate["visual_semantic_hook"] = verdict.hook_strength
    candidate["visual_semantic_self_contained"] = verdict.self_contained
    candidate["visual_semantic_payoff"] = verdict.payoff
    candidate["visual_semantic_moment_type"] = verdict.moment_type
    candidate["visual_semantic_verdict"] = verdict.verdict
    candidate["visual_semantic_title"] = verdict.title
    candidate["visual_semantic_hook_line"] = verdict.hook_line
    candidate["visual_semantic_summary"] = str(raw.get("summary", ""))[:240]
    candidate["visual_semantic_evidence"] = str(raw.get("visual_evidence", ""))[:240]
    candidate["visual_semantic_outcome"] = str(raw.get("visible_outcome", "") or "")
    candidate["visual_semantic_onscreen_text"] = str(raw.get("onscreen_text", "") or "")[:120]
    candidate["visual_semantic_streamer_reaction"] = bool(
        raw.get("streamer_visible_reaction", False),
    )
    candidate["visual_semantic_visual_event"] = bool(raw.get("visual_event", False))
    candidate["visual_semantic_verbal_payoff"] = bool(raw.get("verbal_payoff", False))
    candidate["visual_semantic_context_complete"] = bool(
        raw.get("context_complete", False),
    )
    candidate["visual_semantic_routine_only"] = bool(raw.get("routine_only", False))
    candidate["visual_semantic_model_verdict"] = str(raw.get("model_verdict", "") or "")
    candidate["visual_false_skip_rejudged"] = bool(raw.get("false_skip_rejudged", False))
    candidate["visual_verbal_payoff_rejudged"] = bool(
        raw.get("verbal_payoff_rejudged", False),
    )
    candidate["visual_verbal_payoff_rejected"] = bool(
        raw.get("verbal_payoff_rejudge_rejected", False),
    )
    try:
        candidate["visual_semantic_confidence"] = _clamp(float(raw.get("confidence", 0.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        candidate["visual_semantic_confidence"] = 0.0


OUTCOME_SWEEP_VERSION = "outcome-sweep-v1"
SWEEP_BATCH = 8
SWEEP_STRIDE_SEC = 12.0
SWEEP_MAX_TOKENS = 120

_SWEEP_OUTCOMES = (
    "match_win", "elimination", "death_or_fail", "level_up",
    "menu_or_shop", "none",
)


def _sweep_schema(count: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "outcomes": {
                "type": "array",
                "items": {"type": "string", "enum": list(_SWEEP_OUTCOMES)},
                "minItems": count,
                "maxItems": count,
            },
        },
        "required": ["outcomes"],
    }


_SWEEP_PROMPT = (
    "These frames are INDEPENDENT samples from different moments of one gaming "
    "stream. For each frame, classify the game-state outcome LITERALLY shown:\n"
    '- "match_win": a victory banner / #1 placement / win screen for the player '
    "(e.g. 'VICTORY ROYALE'). A placement like #12 is NOT a win.\n"
    '- "elimination": a kill/knock announcement banner for the player.\n'
    '- "death_or_fail": a death, defeat, or game-over screen.\n'
    '- "level_up": an XP / rank / reward / match-summary screen.\n'
    '- "menu_or_shop": lobby, item shop, settings, loading screen.\n'
    '- "none": ordinary gameplay or anything else.\n'
    "Read banner text carefully. Return JSON only: "
    '{"outcomes": [one label per frame, in order]}.'
)


class OutcomeSweep:
    """Coarse whole-VOD visible-outcome scan (the perception half of the
    content axis). OCR misses stylized banners entirely (a founder VOD's four
    VICTORY ROYALE banners were never read); this sweeps single frames at a
    stride below the banner-sequence duration and batches several frames per
    completion so the cost stays bounded. Hits feed the reaction engine's
    existing terminal_win injection path; the full visual judge later confirms
    each injected candidate, so a stray misread cannot fabricate a win."""

    def __init__(
        self,
        video_path: str,
        *,
        backend=None,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        frame_extractor=None,
        cache: Optional[VisualVerdictCache] = None,
        stride_sec: float = SWEEP_STRIDE_SEC,
        batch: int = SWEEP_BATCH,
        cancel_check=None,
    ):
        self.video_path = video_path
        self.model_path = model_path or "test-backend"
        self.backend = backend or VisualLlamaBackend(self.model_path, str(mmproj_path))
        self.frame_extractor = frame_extractor or VideoFrameExtractor(video_path)
        self.cache = cache or VisualVerdictCache()
        self.stride_sec = max(4.0, float(stride_sec))
        self.batch = max(1, min(12, int(batch)))
        self.cancel_check = cancel_check
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.elapsed_sec = 0.0

    def _key(self, timestamps: Sequence[float]) -> str:
        try:
            stat = os.stat(self.video_path)
            stamp = f"{os.path.abspath(self.video_path)}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            stamp = os.path.abspath(self.video_path)
        raw = "|".join([
            OUTCOME_SWEEP_VERSION, stamp, os.path.abspath(str(self.model_path)),
            ",".join(f"{ts:.1f}" for ts in timestamps),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _classify_batch(self, timestamps: Sequence[float]) -> List[str]:
        key = self._key(timestamps)
        cached = self.cache.load(key)
        if cached is not None and isinstance(cached.get("outcomes"), list):
            self.cache_hits += 1
            return [str(o) for o in cached["outcomes"]]
        frames = self.frame_extractor.extract(timestamps)
        if not frames:
            return ["none"] * len(timestamps)
        raw = self.backend.generate_sweep(_SWEEP_PROMPT, frames, len(frames))
        outcomes = raw.get("outcomes") if isinstance(raw, dict) else None
        if not isinstance(outcomes, list):
            raise ValueError("outcome sweep returned no outcomes array")
        labels = ["none"] * len(timestamps)
        # Map decoded frames back onto requested slots (some frames may fail
        # to decode near EOF; extract() preserves order).
        decoded = {round(sample.timestamp, 3): i for i, sample in enumerate(frames)}
        for slot, ts in enumerate(timestamps):
            idx = decoded.get(round(float(ts), 3))
            if idx is not None and idx < len(outcomes):
                label = str(outcomes[idx]).strip().lower()
                labels[slot] = label if label in _SWEEP_OUTCOMES else "none"
        self.cache.save(key, {"outcomes": labels})
        return labels

    def _classify_all(self, timestamps: List[float]) -> List[Tuple[float, str]]:
        hits: List[Tuple[float, str]] = []
        for i in range(0, len(timestamps), self.batch):
            if self.cancel_check:
                self.cancel_check()
            chunk = timestamps[i:i + self.batch]
            self.calls += 1
            try:
                labels = self._classify_batch(chunk)
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001 - sweep is optional evidence
                self.failures += 1
                print(f"Outcome sweep batch skipped ({chunk[0]:.0f}s): {exc}")
                continue
            hits.extend(
                (ts, label) for ts, label in zip(chunk, labels) if label != "none"
            )
        return hits

    def __call__(self, duration: float) -> List[Tuple[float, str]]:
        """Return [(timestamp, outcome)] for every non-"none" sample.

        Two-stage scan: a coarse pass at 2x stride, then a refine pass at
        stride/2 in ±2x-stride neighborhoods around every coarse non-"none"
        hit. A win sequence (banner -> XP screens -> lobby) spans ~20-40s of
        distinctive frames, so the coarse pass reliably lands at least one
        sample in it while roughly halving total frames on quiet VODs; the
        refine pass then recovers precise onset frames for anchoring.
        """
        duration = max(0.0, float(duration))
        if duration <= 0.0:
            return []
        started = time.perf_counter()

        coarse_step = self.stride_sec * 2.0
        coarse: List[float] = []
        t = coarse_step
        while t < duration:
            coarse.append(round(t, 3))
            t += coarse_step
        if not coarse:
            coarse = [round(duration / 2.0, 3)]
        coarse_hits = self._classify_all(coarse)

        # Refine around win-shaped outcomes only: the banner itself reads
        # match_win for a mere 12-18s, but the XP/summary screens that follow
        # read level_up — together they make the win sequence visible at the
        # coarse stride. Menus/lobbies (the ~200 most frequent coarse hits on
        # a variety VOD) never need refinement; refining everything cost as
        # much as the single-stage sweep saved.
        fine_step = max(4.0, self.stride_sec / 2.0)
        wanted: List[float] = []
        seen = {round(ts, 3) for ts in coarse}
        for hit_ts, label in coarse_hits:
            if label not in ("match_win", "level_up"):
                continue
            lo = max(fine_step, hit_ts - coarse_step)
            hi = min(duration, hit_ts + coarse_step)
            t = lo
            while t <= hi:
                key = round(t, 3)
                if key not in seen:
                    seen.add(key)
                    wanted.append(key)
                t += fine_step
        refined_hits = self._classify_all(sorted(wanted)) if wanted else []

        self.elapsed_sec = time.perf_counter() - started
        return sorted(coarse_hits + refined_hits)

    def health(self) -> dict:
        return {
            "mode": "outcome_sweep",
            "stride_sec": self.stride_sec,
            "batch": self.batch,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "elapsed_sec": round(self.elapsed_sec, 1),
        }

    def close(self) -> None:
        close = getattr(self.frame_extractor, "close", None)
        if callable(close):
            close()


class VisualJudgeSession:
    """Callable visual judge with a per-scan budget, cache, and health record."""

    def __init__(
        self,
        video_path: str,
        transcript: Optional[dict],
        *,
        chat_frames=None,
        game_context: Optional[str] = None,
        backend=None,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        frame_extractor=None,
        cache: Optional[VisualVerdictCache] = None,
        frame_count: int = DEFAULT_FRAME_COUNT,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        cancel_check=None,
        false_skip_rejudge_budget: int = FALSE_SKIP_REJUDGE_BUDGET,
        verbal_payoff_rejudge_budget: int = VERBAL_PAYOFF_REJUDGE_BUDGET,
    ):
        self.video_path = video_path
        self.transcript = transcript
        self.chat_frames = chat_frames
        self.game_context = game_context
        self.frame_count = max(2, min(8, int(frame_count)))
        self.max_candidates = max(1, int(max_candidates))
        self.remaining = self.max_candidates
        self.rejudge_remaining = max(0, int(false_skip_rejudge_budget))
        self.verbal_rejudge_remaining = max(0, int(verbal_payoff_rejudge_budget))
        self.cancel_check = cancel_check
        self.model_path = model_path or "test-backend"
        self.mmproj_path = mmproj_path
        self.backend = backend or VisualLlamaBackend(self.model_path, str(mmproj_path))
        self.frame_extractor = frame_extractor or VideoFrameExtractor(video_path)
        self.cache = cache or VisualVerdictCache()
        self.records: List[dict] = []
        self.cache_hits = 0
        self.failures = 0
        self.false_skip_rejudges = 0
        self.verbal_payoff_rejudges = 0
        self.verbal_payoff_rejects = 0
        self.unavailable = False
        self._words = text_judge._transcript_words(transcript)

    def _generate_raw(
        self,
        candidate: dict,
        words: Sequence[str],
        chat: Sequence[str],
        *,
        frame_count: int,
        pass_tag: str = "",
        disagreement_rejudge: bool = False,
        strict_verbal_rejudge: bool = False,
    ) -> Tuple[dict, List[float], bool]:
        """Load or generate one calibrated visual JSON for a candidate."""
        cache_key = self.cache.key(
            self.video_path, candidate, self.model_path, frame_count,
            pass_tag=pass_tag,
        )
        raw = self.cache.load(cache_key)
        cached = raw is not None
        timestamps = keyframe_timestamps(candidate, frame_count)
        if raw is None:
            frames = self.frame_extractor.extract(timestamps)
            if len(frames) < 2:
                raise ValueError(
                    f"Only {len(frames)} visual frame(s) decoded for candidate",
                )
            prompt = _visual_prompt(
                candidate, words, chat, frames,
                game_context=self.game_context,
                disagreement_rejudge=disagreement_rejudge,
                strict_verbal_rejudge=strict_verbal_rejudge,
            )
            raw = self.backend.generate(prompt, frames)
            if not isinstance(raw, dict):
                raise ValueError("Visual judge returned no JSON object")
            # Drop any stale markers a model might echo from the prompt.
            raw.pop("false_skip_rejudged", None)
            # Cache the PRISTINE model JSON (CACHE_SCHEMA_VERSION contract);
            # calibration runs below on both paths, so it always applies
            # exactly once to model-authored fields and a warm rescan cannot
            # diverge from the cold scan.
            self.cache.save(cache_key, raw)
        else:
            self.cache_hits += 1
        raw = _calibrate_raw(raw, window_words=words)
        return raw, timestamps, cached

    def _maybe_verbal_payoff_rejudge(
        self,
        candidate: dict,
        raw: dict,
        words: Sequence[str],
        chat: Sequence[str],
    ) -> Tuple[dict, List[float], bool]:
        """Second opinion for generic story claims with no grounded visual beat.

        A globally stricter prompt can reject worthwhile clips. This bounded pass
        is only authoritative when the original candidate was chat-spike driven
        and either the strict editor finds no landing or the text editor also
        called the clip filler. Its scope is intentionally narrower than the
        initial visual judgment to limit false rejection.
        """
        if self.verbal_rejudge_remaining <= 0:
            return raw, [], False
        if not _optimistic_verbal_story_disagreement(candidate, raw):
            return raw, [], False
        self.verbal_rejudge_remaining -= 1
        self.verbal_payoff_rejudges += 1
        pass_tag = "strict-verbal-payoff-rejudge-v1"
        raw2, timestamps, cached = self._generate_raw(
            candidate,
            words,
            chat,
            frame_count=self.frame_count,
            pass_tag=pass_tag,
            strict_verbal_rejudge=True,
        )
        chat_signal = float(candidate.get("hook_chat", 0.0) or 0.0)
        semantic_verdict = str(candidate.get("semantic_verdict") or "").strip().lower()
        semantic_moment = str(candidate.get("semantic_moment_type") or "").strip().lower()
        strict_reject = (
            _strict_verbal_unsupported(raw2)
            and chat_signal >= VERBAL_REJUDGE_MIN_CHAT
        )
        text_consensus_reject = (
            semantic_verdict == "skip"
            and semantic_moment == "filler"
            and chat_signal >= VERBAL_TEXT_CONSENSUS_MIN_CHAT
        )
        if not (strict_reject or text_consensus_reject):
            raw["verbal_payoff_rejudged"] = True
            raw["verbal_payoff_rejudge_rejected"] = False
            raw["verbal_payoff_rejudge_verdict"] = str(raw2.get("verdict") or "")
            return raw, timestamps, cached

        self.verbal_payoff_rejects += 1
        raw2["verbal_payoff_rejudged"] = True
        raw2["verbal_payoff_rejudge_rejected"] = True
        raw2["verbal_payoff_prior_verdict"] = str(raw.get("verdict") or "")
        raw2["verdict"] = "skip"
        raw2["moment_type"] = "filler"
        raw2["verbal_payoff"] = False
        raw2["context_complete"] = False
        raw2["routine_only"] = True
        try:
            raw2["payoff"] = min(0.3, float(raw2.get("payoff", 0.0) or 0.0))
        except (TypeError, ValueError):
            raw2["payoff"] = 0.0
        return raw2, timestamps, cached

    def _maybe_false_skip_rejudge(
        self,
        candidate: dict,
        raw: dict,
        words: Sequence[str],
        chat: Sequence[str],
    ) -> Tuple[dict, List[float], bool]:
        """Second denser pass when text/content says climax but verdict is skip.

        Calibration already rescues clear banner/ASR cases. This path covers
        cold-facecam / sparse-sample false skips where the model's own text
        (or transcript) still describes a climax after the first pass.
        """
        if self.rejudge_remaining <= 0:
            return raw, [], False
        if not _false_skip_disagreement(raw, words):
            return raw, [], False
        self.rejudge_remaining -= 1
        self.false_skip_rejudges += 1
        rejudge_frames = max(self.frame_count, FALSE_SKIP_REJUDGE_FRAMES)
        pass_tag = "false-skip-rejudge-v1"
        raw2, timestamps, cached = self._generate_raw(
            candidate, words, chat,
            frame_count=rejudge_frames,
            pass_tag=pass_tag,
            disagreement_rejudge=True,
        )
        raw2["false_skip_rejudged"] = True
        # Persist the disagreement trail from the first pass for audits.
        raw2["false_skip_prior_verdict"] = str(
            raw.get("model_verdict") or raw.get("verdict") or "",
        )
        if (
            str(raw2.get("verdict", "")).lower() == "skip"
            and _content_climax_signal(raw2, words)
            and _grounded_skip_promotion(raw2)
        ):
            # Second pass still describes a climax — floor to maybe so
            # selection's visual_skip veto cannot discard it.
            outcome = str(raw2.get("visible_outcome", "") or "").strip().lower()
            _promote_skip_to_maybe(raw2, outcome=outcome)
            raw2["false_skip_promoted"] = True
        # No write-back: the pass_tag entry already holds the pristine
        # second-pass JSON, and this flag/floor application is a pure function
        # of that JSON plus the transcript window, so a warm rescan re-derives
        # the same post-rejudge floor instead of reading it off disk.
        return raw2, timestamps, cached

    def __call__(self, candidates: List[dict]) -> Dict[int, SemanticVerdict]:
        verdicts: Dict[int, SemanticVerdict] = {}
        for idx, candidate in enumerate(candidates):
            if self.remaining <= 0:
                break
            if self.cancel_check:
                self.cancel_check()
            self.remaining -= 1
            started = time.perf_counter()
            judge_start, judge_end = judge_window(candidate)
            lo = judge_start - text_judge.WINDOW_PAD_SEC
            hi = judge_end + text_judge.WINDOW_PAD_SEC
            words = text_judge._words_in_window(self._words, lo, hi)
            chat = text_judge._chat_lines(self.chat_frames, lo, hi)
            try:
                raw, timestamps, cached = self._generate_raw(
                    candidate, words, chat, frame_count=self.frame_count,
                )
                rejudge_timestamps: List[float] = []
                false_skip_rejudged = False
                verbal_payoff_rejudged = False
                if _optimistic_verbal_story_disagreement(candidate, raw):
                    raw, verbal_timestamps, verbal_cached = self._maybe_verbal_payoff_rejudge(
                        candidate, raw, words, chat,
                    )
                    if raw.get("verbal_payoff_rejudged"):
                        verbal_payoff_rejudged = True
                        cached = cached and verbal_cached
                        if verbal_timestamps:
                            timestamps = verbal_timestamps
                if (
                    not raw.get("verbal_payoff_rejudge_rejected")
                    and _false_skip_disagreement(raw, words)
                ):
                    raw, rejudge_timestamps, rejudge_cached = self._maybe_false_skip_rejudge(
                        candidate, raw, words, chat,
                    )
                    if raw.get("false_skip_rejudged"):
                        false_skip_rejudged = True
                        cached = cached and rejudge_cached
                        if rejudge_timestamps:
                            timestamps = rejudge_timestamps

                verdict = SemanticVerdict.from_raw(raw)
                if verdict is None:
                    raise ValueError("Visual judge returned an invalid editorial verdict")
                _attach_visual_fields(candidate, verdict, raw)
                verdicts[idx] = verdict
                self.records.append({
                    "candidate_index": idx,
                    "start": round(float(candidate.get("start", 0.0)), 3),
                    "end": round(float(candidate.get("end", 0.0)), 3),
                    "judge_start": round(judge_start, 3),
                    "judge_end": round(judge_end, 3),
                    "peak_timestamp": round(float(candidate.get("peak_timestamp", 0.0) or 0.0), 3),
                    "frame_timestamps": timestamps,
                    "cached": cached,
                    "false_skip_rejudged": false_skip_rejudged,
                    "verbal_payoff_rejudged": verbal_payoff_rejudged,
                    "verbal_payoff_rejected": bool(
                        raw.get("verbal_payoff_rejudge_rejected", False),
                    ),
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "raw": raw,
                })
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001 - optional model never blocks a scan
                self.failures += 1
                if not verdicts and len(self.records) == 0:
                    # The orchestrator may use this to activate the text fallback
                    # after closing the failed visual runtime.
                    self.unavailable = True
                self.records.append({
                    "candidate_index": idx,
                    "start": round(float(candidate.get("start", 0.0)), 3),
                    "end": round(float(candidate.get("end", 0.0)), 3),
                    "judge_start": round(judge_start, 3),
                    "judge_end": round(judge_end, 3),
                    "cached": False,
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "error": str(exc)[:500],
                })
                print(f"Visual semantic judge skipped candidate {idx}: {exc}")
        return verdicts

    def ensure_capacity(self, n: int) -> None:
        """Guarantee room to judge ``n`` more candidates (deck-coverage pass).

        The pre-selection pool can exhaust ``max_candidates`` on moments that
        later die at the evidence gate, leaving selected winners unjudged.
        Deck coverage must still be able to read those winners.
        """
        need = max(0, int(n))
        if self.remaining < need:
            self.remaining = need

    def health(self) -> dict:
        judged = len({
            (row.get("start"), row.get("end"))
            for row in self.records
            if "raw" in row
        })
        unique_attempts = len({
            (row.get("start"), row.get("end"))
            for row in self.records
        })
        failure_rows = [
            {
                "candidate_index": int(row.get("candidate_index", -1)),
                "start": row.get("start"),
                "end": row.get("end"),
                "error": str(row.get("error") or "")[:200],
            }
            for row in self.records
            if row.get("error")
        ]
        # A selected candidate can be revisited by deck coverage. Report one
        # affected moment to creators while retaining raw attempt counts for
        # diagnostics, otherwise a single cached bad verdict looks systemic.
        unique_failures = list({
            (row.get("start"), row.get("end"), row.get("error")): row
            for row in failure_rows
        }.values())
        recent_failures = unique_failures[-5:]
        return {
            "mode": "visual",
            "candidate_budget": self.max_candidates,
            "candidates_attempted": len(self.records),
            "unique_candidates_attempted": unique_attempts,
            "judged_candidates": judged,
            "cache_hits": self.cache_hits,
            "failures": len(unique_failures),
            "failure_attempts": self.failures,
            "false_skip_rejudges": self.false_skip_rejudges,
            "verbal_payoff_rejudges": self.verbal_payoff_rejudges,
            "verbal_payoff_rejects": self.verbal_payoff_rejects,
            "remaining_budget": self.remaining,
            "rejudge_remaining": self.rejudge_remaining,
            "verbal_rejudge_remaining": self.verbal_rejudge_remaining,
            "degraded": bool(unique_failures),
            "model": os.path.basename(self.model_path),
            # Which prompt format the weights actually ran under; a lane that
            # quietly fell back to the legacy format is a quality regression
            # that would otherwise be invisible in the scan log.
            "chat_handler": getattr(self.backend, "handler_kind", ""),
            "frame_count": self.frame_count,
            "recent_failures": recent_failures,
        }

    def close(self) -> None:
        close = getattr(self.frame_extractor, "close", None)
        if callable(close):
            close()
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()
