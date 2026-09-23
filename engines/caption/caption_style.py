# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Caption styling (plan 9.4).

Turns the creator's caption-style settings (a small JSON object owned by the
frontend) into concrete ASS directives used by the .ass generators in
``whisper_asr``. Fonts, sizes and positions are whitelisted so a bad client
can't inject arbitrary lines into the ASS ``[V4+ Styles]`` header, and colors
are validated hex → ASS ``&HAABBGGRR``.

The object shape (all optional; sensible defaults fill the gaps):

    {
      "enabled": true,
      "font": "Arial Black",
      "size": "medium",          # small | medium | large
      "textColor": "#FFFFFF",
      "highlightColor": "#FFFF00",
      "position": "bottom"        # bottom | middle | top
    }
"""

from __future__ import annotations

import copy
import re
from typing import Optional


# Fonts we're willing to name in the ASS header. Keyed case-insensitively on the
# frontend value; the value is the exact font name written into the file.
FONT_WHITELIST = {
    "arial black": "Arial Black",
    "impact": "Impact",
    "arial": "Arial",
    "verdana": "Verdana",
    "georgia": "Georgia",
    "trebuchet ms": "Trebuchet MS",
    "segoe ui black": "Segoe UI Black",
}
DEFAULT_FONT = "Arial Black"

# Named style presets (plan 22 §3.4): one click instead of four knobs. Each is
# just a raw settings dict composed of whitelisted values — a preset can never
# express anything the sanitizer wouldn't allow, and explicit fields sent
# alongside a preset override it (pick "neon-pop" then nudge the size).
STYLE_PRESETS = {
    "classic":        {"font": "Arial Black", "size": "medium", "textColor": "#FFFFFF", "highlightColor": "#FFFF00", "position": "bottom"},
    "bold-center":    {"font": "Impact", "size": "large", "textColor": "#FFFFFF", "highlightColor": "#E9BE6B", "position": "middle"},
    "minimal-lower":  {"font": "Arial", "size": "small", "textColor": "#FFFFFF", "highlightColor": "#F08A5C", "position": "bottom"},
    "neon-pop":       {"font": "Impact", "size": "large", "textColor": "#FFFFFF", "highlightColor": "#39FF14", "position": "bottom"},
    "sunset":         {"font": "Verdana", "size": "medium", "textColor": "#FFF3E9", "highlightColor": "#FF6B4A", "position": "bottom"},
    "ice":            {"font": "Segoe UI Black", "size": "medium", "textColor": "#EAF6FF", "highlightColor": "#4AC8FF", "position": "bottom"},
    "headline":       {"font": "Trebuchet MS", "size": "medium", "textColor": "#FFFFFF", "highlightColor": "#FFD24A", "position": "top"},
    "studio-heat":    {"font": "Arial Black", "size": "medium", "textColor": "#F6F0E9", "highlightColor": "#F08A5C", "position": "bottom"},
}

# Font sizes at the fixed 1080x1920 caption canvas.
SIZE_MAP = {"small": 56, "medium": 72, "large": 92}
DEFAULT_SIZE = "medium"

# ASS numpad alignment + vertical margin per position.
POSITION_MAP = {
    "bottom": (2, 250),
    "middle": (5, 0),
    "top": (8, 130),
}
DEFAULT_POSITION = "bottom"

DEFAULT_TEXT_COLOR = "&H00FFFFFF"       # white
DEFAULT_HIGHLIGHT_COLOR = "&H0000FFFF"  # yellow

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
    "MarginR, MarginV, Encoding"
)
_EVENTS_FORMAT = (
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text"
)


def _hex_to_ass(value: Optional[str], fallback: str) -> str:
    """Convert ``#RRGGBB`` (or ``#RGB``) into an opaque ASS ``&H00BBGGRR``."""
    if not isinstance(value, str):
        return fallback
    match = _HEX_RE.match(value.strip())
    if not match:
        return fallback
    h = match.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}".upper()


class CaptionStyle:
    """Resolved, sanitized caption style ready to render into an .ass file."""

    def __init__(self, raw: Optional[dict] = None):
        raw = raw if isinstance(raw, dict) else {}
        # A named preset seeds the fields; explicit fields override it.
        preset = STYLE_PRESETS.get(str(raw.get("preset", "")).strip().lower())
        if preset:
            raw = {**preset, **{k: v for k, v in raw.items() if k != "preset" and v is not None}}
        self.enabled = bool(raw.get("enabled", True))
        self.font = FONT_WHITELIST.get(
            str(raw.get("font", "")).strip().lower(), DEFAULT_FONT
        )
        self.size = SIZE_MAP.get(str(raw.get("size", "")).strip().lower(), SIZE_MAP[DEFAULT_SIZE])
        self.text_color = _hex_to_ass(raw.get("textColor"), DEFAULT_TEXT_COLOR)
        self.highlight_color = _hex_to_ass(raw.get("highlightColor"), DEFAULT_HIGHLIGHT_COLOR)
        self.alignment, self.margin_v = POSITION_MAP.get(
            str(raw.get("position", "")).strip().lower(), POSITION_MAP[DEFAULT_POSITION]
        )

    def facecam_safe(self, has_top_facecam: bool) -> "CaptionStyle":
        """Relocate a top-aligned caption below a top facecam band.

        Both export layouts place the facecam at the top of the frame
        (vertical_split's top band; gameplay_pip's top-center panel), so a
        ``top`` caption would burn text over the streamer's face. When the clip
        carries a top facecam, a top caption drops to the vertical center —
        clear of the cam band above and the platform's bottom UI below. Every
        other position is already clear and is returned unchanged, so the
        default bottom look never moves.
        """
        if not has_top_facecam or self.alignment != POSITION_MAP["top"][0]:
            return self
        safe = copy.copy(self)
        safe.alignment, safe.margin_v = POSITION_MAP["middle"]
        return safe

    @property
    def outline(self) -> int:
        # Scale the outline with the font size so big text keeps its punch and
        # small text doesn't drown in border.
        return max(2, round(self.size / 14))

    def style_line(self) -> str:
        return (
            f"Style: Default,{self.font},{self.size},{self.text_color},"
            f"&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,"
            f"{self.outline},0,{self.alignment},10,10,{self.margin_v},1"
        )

    def header(self) -> str:
        return (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n\n"
            "[V4+ Styles]\n"
            f"{_STYLE_FORMAT}\n"
            f"{self.style_line()}\n\n"
            "[Events]\n"
            f"{_EVENTS_FORMAT}\n"
        )


def resolve_caption_style(raw) -> CaptionStyle:
    """Coerce raw settings (dict, CaptionStyle, or None) into a CaptionStyle."""
    if isinstance(raw, CaptionStyle):
        return raw
    return CaptionStyle(raw)
