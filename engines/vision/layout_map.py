# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Stream layout inference from OCR box statistics (layout-agnostic pass 1).

Not every streamer runs fullscreen gameplay with a corner facecam: chat panels,
alert widgets, and windowed game captures are common, and their text lands in
the same full-frame OCR blob as the game HUD. That contaminates everything
downstream -- a chat message quoting "eliminated" fires a game trigger, alert
text tilts lobby classification, chat mentioning another title pollutes game
auto-detection.

This module infers, once per VOD, which screen regions are *chat-like overlays*
(persistent text zones whose content constantly churns) purely from the OCR
entries the pipeline already captured -- no extra model, no extra frame reads.
The pipeline then annotates each OCR signal with ``gameplay_text`` (the joined
text of entries OUTSIDE those regions) and game detection / event triggers /
scene classification prefer it over the raw blob.

Signature of a chat/alert overlay vs. game HUD:
  * chat: text present in most sampled frames AND the content is different
    almost every time it's read (messages scroll),
  * HUD (health/kill count): persistent but mostly *stable* text,
  * banners/killfeed (real triggers): high churn but LOW presence -- they only
    exist for seconds around an event, so the presence gate keeps them routed
    to gameplay.
Thresholds are deliberately conservative: misclassifying HUD as chat would
silently drop real triggers, so a region must be persistently occupied AND
high-churn across many samples before anything is filtered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Grid resolution for occupancy statistics. 16x16 keeps a 1080p cell around
# 120x67 px -- fine enough to isolate a chat column from adjacent gameplay.
GRID = 16

# Hysteresis thresholds: STRONG cells seed a chat region; WEAK cells (text
# drifting a row between samples lands in adjacent cells at lower presence)
# only join a region already seeded by a strong neighbor. A cell must hold
# text in at least this fraction of OCR-bearing frames (transient banners /
# killfeed stay well below even the weak gate) ...
STRONG_PRESENCE_RATIO = 0.45
WEAK_PRESENCE_RATIO = 0.20
# ... and its text must differ between consecutive occupied reads at least
# this often (scrolling chat ~1.0; stable HUD labels ~0.0). Strict for both
# tiers -- churn is the HUD-vs-chat discriminator.
MIN_CHURN_RATIO = 0.50
# Lexical chat-likeness guard (plan 22 §4.1): a speedrun timer or "now
# playing" ticker is persistent AND churns every read, but each read is a
# single token ("1:23:45"); chat lines are multi-word. Require an average of
# at least this many words per occupied read before a cell can be chat.
MIN_AVG_WORDS = 2.0
# Minimum occupied samples before a cell can be judged at all.
MIN_OCCUPIED_FRAMES = 8
# Below this many OCR-bearing frames the VOD gives too little evidence to
# infer any layout -- return None and leave routing untouched.
MIN_FRAMES_FOR_LAYOUT = 20


@dataclass
class LayoutMap:
    """Inferred overlay regions, normalized to the observed frame extent."""
    frame_extent: Tuple[float, float]              # (width, height) in OCR pixel space
    chat_regions: List[List[float]] = field(default_factory=list)  # [x, y, w, h] normalized
    cells_flagged: int = 0
    frames_sampled: int = 0

    def in_chat_region(self, cx_norm: float, cy_norm: float) -> bool:
        for x, y, w, h in self.chat_regions:
            if x <= cx_norm <= x + w and y <= cy_norm <= y + h:
                return True
        return False


def _normalize_text(text: str) -> str:
    """Match engines.ocr.ocr text normalization so routed text is drop-in."""
    normalized = (text or "").upper()
    normalized = normalized.replace("|", "I").replace("!", "I")
    normalized = re.sub(r"[^A-Z0-9# ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _box_rect(box) -> Optional[Tuple[float, float, float, float]]:
    """Bounding rect of an OCR entry box: flat [x1,y1,x2,y2] or a 4-point polygon."""
    if box is None:
        return None
    try:
        if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            x1, y1, x2, y2 = (float(v) for v in box)
        else:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 < x1 or y2 < y1:
            return None
        return (x1, y1, x2, y2)
    except (TypeError, ValueError, IndexError):
        return None


def _box_center(box) -> Optional[Tuple[float, float]]:
    rect = _box_rect(box)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _ocr_entries(signal) -> List[dict]:
    ocr = getattr(signal, "ocr", None)
    metadata = getattr(ocr, "metadata", None) if ocr else None
    if not isinstance(metadata, dict):
        return []
    entries = metadata.get("entries")
    return entries if isinstance(entries, list) else []


def build_layout_map(unified_signals) -> Optional[LayoutMap]:
    """Infer chat-like overlay regions from per-entry OCR statistics.

    Returns None when the VOD has too few OCR-bearing frames (or no per-entry
    boxes at all) to say anything -- callers must treat None as "no routing".
    """
    # Pass 1: collect (timestamp-ordered) per-frame entry rects + texts and
    # the frame extent implied by the boxes themselves.
    frames: List[List[Tuple[Tuple[float, float, float, float], str]]] = []
    max_x, max_y = 0.0, 0.0
    for sig in unified_signals:
        entries = _ocr_entries(sig)
        if not entries:
            continue
        frame_items = []
        for entry in entries:
            rect = _box_rect(entry.get("box"))
            text = entry.get("text")
            if rect is None or not text:
                continue
            frame_items.append((rect, str(text)))
            max_x = max(max_x, rect[2])
            max_y = max(max_y, rect[3])
        if frame_items:
            frames.append(frame_items)

    if len(frames) < MIN_FRAMES_FOR_LAYOUT or max_x <= 0 or max_y <= 0:
        return None

    # Extent estimate: text never quite reaches the frame edge, so pad a
    # little. Only consistency matters -- routing happens in the same space.
    extent_w = max_x * 1.05
    extent_h = max_y * 1.05

    # Pass 2: per-cell presence + churn. A box occupies every cell it spans
    # (not just its center) so scrolling chat lines that drift a row between
    # samples still register as one persistently occupied column.
    occupied: Dict[Tuple[int, int], int] = {}
    churn: Dict[Tuple[int, int], int] = {}
    words: Dict[Tuple[int, int], int] = {}
    prev_texts: Dict[Tuple[int, int], frozenset] = {}
    for frame_items in frames:
        cell_texts: Dict[Tuple[int, int], set] = {}
        for (x1, y1, x2, y2), text in frame_items:
            col_lo = max(0, min(GRID - 1, int(x1 / extent_w * GRID)))
            col_hi = max(0, min(GRID - 1, int(x2 / extent_w * GRID)))
            row_lo = max(0, min(GRID - 1, int(y1 / extent_h * GRID)))
            row_hi = max(0, min(GRID - 1, int(y2 / extent_h * GRID)))
            normalized = _normalize_text(text)
            for row in range(row_lo, row_hi + 1):
                for col in range(col_lo, col_hi + 1):
                    cell_texts.setdefault((row, col), set()).add(normalized)
        for cell, texts in cell_texts.items():
            occupied[cell] = occupied.get(cell, 0) + 1
            words[cell] = words.get(cell, 0) + sum(len(t.split()) for t in texts)
            current = frozenset(texts)
            if cell in prev_texts and prev_texts[cell] != current:
                churn[cell] = churn.get(cell, 0) + 1
            prev_texts[cell] = current

    n_frames = len(frames)
    strong, weak = set(), set()
    for cell, count in occupied.items():
        if count < MIN_OCCUPIED_FRAMES:
            continue
        presence = count / n_frames
        if presence < WEAK_PRESENCE_RATIO:
            continue
        # Churn ratio over *revisits* (occupied frames after the first).
        revisits = count - 1
        if revisits <= 0:
            continue
        if churn.get(cell, 0) / revisits < MIN_CHURN_RATIO:
            continue
        # Chat is multi-word text; a churning single-token cell is a timer /
        # score ticker whose text must stay routed to gameplay (plan 22 §4.1).
        if words.get(cell, 0) / count < MIN_AVG_WORDS:
            continue
        if presence >= STRONG_PRESENCE_RATIO:
            strong.add(cell)
        else:
            weak.add(cell)

    if not strong:
        return LayoutMap(frame_extent=(extent_w, extent_h), chat_regions=[],
                         cells_flagged=0, frames_sampled=n_frames)

    # Pass 3: cluster (4-connectivity) over strong+weak cells; keep only
    # components containing at least one strong seed.
    regions: List[List[float]] = []
    flagged = 0
    remaining = strong | weak
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.discard((nr, nc))
                    component.add((nr, nc))
                    stack.append((nr, nc))
        if not (component & strong):
            continue
        flagged += len(component)
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        regions.append([
            min(cols) / GRID,
            min(rows) / GRID,
            (max(cols) - min(cols) + 1) / GRID,
            (max(rows) - min(rows) + 1) / GRID,
        ])

    return LayoutMap(
        frame_extent=(extent_w, extent_h),
        chat_regions=regions,
        cells_flagged=flagged,
        frames_sampled=n_frames,
    )


def annotate_gameplay_text(unified_signals, layout: Optional[LayoutMap]) -> int:
    """Set ``metadata["gameplay_text"]`` on every OCR signal with entry boxes.

    ``gameplay_text`` is the normalized join of entries OUTSIDE the inferred
    chat regions; the excluded text is kept in ``metadata["overlay_text"]`` for
    diagnostics (and a future on-screen chat-velocity channel). Signals without
    per-entry boxes are left untouched, so consumers falling back to ``.text``
    keep working. Returns how many signals had at least one entry filtered.
    """
    if layout is None or not layout.chat_regions:
        return 0
    extent_w, extent_h = layout.frame_extent
    filtered_count = 0
    for sig in unified_signals:
        entries = _ocr_entries(sig)
        if not entries:
            continue
        keep, dropped = [], []
        for entry in entries:
            center = _box_center(entry.get("box"))
            text = entry.get("text")
            if not text:
                continue
            if center is not None and layout.in_chat_region(
                center[0] / extent_w, center[1] / extent_h
            ):
                dropped.append(str(text))
            else:
                keep.append(str(text))
        sig.ocr.metadata["gameplay_text"] = _normalize_text(" ".join(keep))
        if dropped:
            sig.ocr.metadata["overlay_text"] = _normalize_text(" ".join(dropped))
            filtered_count += 1
    return filtered_count


def routed_text(ocr_signal) -> Optional[str]:
    """The text a gameplay-semantics consumer should read for this signal.

    Prefers the spatially routed ``gameplay_text`` when the pipeline annotated
    it (which may legitimately be "" when every entry was overlay text), else
    falls back to the full-frame text.
    """
    if ocr_signal is None:
        return None
    metadata = getattr(ocr_signal, "metadata", None)
    if isinstance(metadata, dict) and "gameplay_text" in metadata:
        return metadata["gameplay_text"]
    return getattr(ocr_signal, "text", None)
