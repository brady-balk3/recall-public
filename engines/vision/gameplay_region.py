# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Gameplay region detection from temporal motion statistics (layout pass 2).

Windowed stream layouts (game capture at ~75% of frame, chat panel down one
side, cam + alert widgets along an edge) break the export assumption that
gameplay fills the frame: a center 9:16 crop slices through window chrome and
chat. This module finds the actual gameplay rectangle once per VOD so export
crops frame the game itself.

The discriminator is *median* per-cell motion across sampled frame pairs:
  * the game viewport renders continuously (camera sway, HUD animation) ->
    high median motion,
  * chat columns only change when a message lands -> near-zero median,
  * window chrome / borders / static art -> zero.
The facecam panel also moves constantly, so its detected box is excluded
before taking the largest connected high-motion region.

Returns None whenever the evidence is weak or the region is effectively the
whole frame -- callers treat None as "fullscreen gameplay, keep the existing
center-crop behavior". Long VODs sample about once every 90 seconds (capped at
180 seeks) so brief authored scenes are represented without turning this into
a second full video pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

GRID = 24                    # motion-statistics grid (24x24 cells)
CELL_PX = 10                 # analysis frame is resized to GRID*CELL_PX square
DEFAULT_SAMPLES = 20         # frame pairs sampled uniformly across the VOD
PAIR_GAP_FRAMES = 4          # frames grabbed between the two reads of a pair
MIN_MEDIAN_PEAK = 2.0        # below this 95th-pct median motion, video is too static to judge
ACTIVE_FRACTION = 0.30       # cell is active if its median >= this * the 95th-pct peak
MIN_REGION_AREA = 0.25       # gameplay smaller than this fraction of frame -> distrust
FULLSCREEN_COVERAGE = 0.88   # region covering this much of both axes == fullscreen -> None
# Windowed game captures mirror a 16:9 game; accept modest chrome/deviation
# (confirmed real layouts measure 1.78-1.79) but reject one-axis shrinkage
# (the measured hallucination scored 1.556 — keep the floor above it).
# A later Fortnite miss scored ~1.609 with dual-axis quiet edges
# ([0.1667, 0.125, 0.7917, 0.875]); raise the floor and add the dual-axis
# quiet-edge guard below so that family cannot pass either gate alone.
WINDOWED_ASPECT_MIN = 1.68
WINDOWED_ASPECT_MAX = 2.00
# Quiet-edge hallucination: a large motion blob that pulled in thin margins
# on 2+ sides with no substantial chrome strip (chat/widget column).
NEAR_FULLSCREEN_MIN_W = 0.75
NEAR_FULLSCREEN_MIN_H = 0.75
THIN_MARGIN_MIN = 0.02
THIN_MARGIN_MAX = 0.18
THICK_CHROME_MIN = 0.18

# Authored stream canvases commonly place the actual content inside a bordered
# window while chat, facecam, and alert widgets occupy the remaining canvas.
# Short-term motion cannot recover the edges of a dark/static game, so a second
# detector looks for the long, persistent rectangle that defines the content
# window.  It is intentionally style/color agnostic: Canny + Hough sees a
# lavender Windows-95 frame, a white OBS stroke, and a plain dark divider alike.
STRUCTURAL_MAX_WIDTH = 960
STRUCTURAL_MIN_AREA = 0.35
STRUCTURAL_MAX_AREA = 0.90
STRUCTURAL_CLUSTER_IOU = 0.72
STRUCTURAL_MIN_SUPPORT = 2
# A real authored content window should be visually denser than the surrounding
# canvas. Fullscreen gameplay often contains long HUD/divider lines that form a
# convincing rectangle, but the "outside" of that rectangle is still equally
# detailed gameplay. In uncertain cases, returning None is the safe choice:
# preserving the full stream is far less destructive than hard-cropping to a
# false viewport.
STRUCTURAL_MIN_EDGE_CONTRAST = 1.15
STRUCTURAL_EDGE_GUARD_FRAC = 0.012
# A dark game can have far fewer edges than a detailed stream overlay. In that
# case edge contrast is not a valid veto, but a large, complete 16:9 window is.
# Small low-contrast rectangles remain rejected (the common fullscreen HUD
# false-positive case).
STRUCTURAL_DARK_WINDOW_MIN_AREA = 0.50
STRUCTURAL_MIN_ASPECT_SCORE = 0.72  # admits 4:3 through ultrawide landscape
STRUCTURAL_MIN_MOTION_CONTRAST = 1.25
STRUCTURAL_SAMPLE_INTERVAL = 90.0
STRUCTURAL_MAX_SAMPLES = 180


@dataclass
class GameplayLayout:
    """One recurring authored content-window geometry and when it appeared."""

    box: List[float]
    timestamps: List[float] = field(default_factory=list)
    support: int = 1
    confidence: float = 0.0


@dataclass
class GameplayObservation:
    """The resolved authored layout (or fullscreen/unknown) at one sample."""

    timestamp: float
    box: Optional[List[float]] = None


@dataclass
class GameplayLayoutModel:
    """Persistent layout geometries plus the sampled scene timeline."""

    layouts: List[GameplayLayout] = field(default_factory=list)
    observations: List[GameplayObservation] = field(default_factory=list)
    sample_interval: float = 0.0


def is_plausible_windowed_gameplay(box: Optional[List[float]]) -> bool:
    """True when a normalized [x,y,w,h] box looks like a real authored window.

    False means callers should treat the frame as fullscreen (no gameplay crop).
    Rejects near-fullscreen quiet-edge shrinks that cleared older aspect-only
    gates — notably Fortnite Twitch VODs emitting ~[0.17, 0.13, 0.79, 0.88].
    """
    if not box or len(box) < 4:
        return False
    x, y, w, h = (float(part) for part in box[:4])
    if w <= 0.0 or h <= 0.0:
        return False
    if w * h < MIN_REGION_AREA:
        return False
    if w >= FULLSCREEN_COVERAGE and h >= FULLSCREEN_COVERAGE:
        return False
    pixel_aspect = (w * 16.0) / max(1e-6, h * 9.0)
    if not (WINDOWED_ASPECT_MIN <= pixel_aspect <= WINDOWED_ASPECT_MAX):
        return False
    if w >= NEAR_FULLSCREEN_MIN_W and h >= NEAR_FULLSCREEN_MIN_H:
        margins = [x, y, 1.0 - (x + w), 1.0 - (y + h)]
        thin = sum(1 for margin in margins if THIN_MARGIN_MIN <= margin < THIN_MARGIN_MAX)
        thick = sum(1 for margin in margins if margin >= THICK_CHROME_MIN)
        # Authored layouts usually have one thick chrome strip (chat/widgets).
        # Quiet-edge noise is thin on multiple sides with no thick strip.
        if thin >= 2 and thick == 0:
            return False
    return True


def _box_iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def _cluster_lines(lines, tolerance: float):
    """Cluster axis-aligned Hough segments by their x/y position."""
    clusters = []
    for position, span_a, span_b in sorted(lines):
        match = next(
            (cluster for cluster in clusters if abs(position - cluster["position"]) <= tolerance),
            None,
        )
        if match is None:
            clusters.append({
                "position": float(position),
                "positions": [float(position)],
                "spans": [(float(min(span_a, span_b)), float(max(span_a, span_b)))],
            })
            continue
        match["positions"].append(float(position))
        match["position"] = float(np.median(match["positions"]))
        match["spans"].append((float(min(span_a, span_b)), float(max(span_a, span_b))))
    return clusters


def _line_coverage(cluster, span_start: float, span_end: float) -> float:
    length = max(1.0, span_end - span_start)
    return max(
        (
            max(0.0, min(span_end, b) - max(span_start, a)) / length
            for a, b in cluster["spans"]
        ),
        default=0.0,
    )


def _edge_contrast(edges: np.ndarray, left: float, top: float,
                   right: float, bottom: float) -> float:
    """Inside/outside edge-density ratio, excluding the candidate border.

    The Hough lines themselves would otherwise make any rectangle look more
    content-like. A small guard removes those strokes from both samples.
    """
    height, width = edges.shape[:2]
    guard_x = max(2, int(round(STRUCTURAL_EDGE_GUARD_FRAC * width)))
    guard_y = max(2, int(round(STRUCTURAL_EDGE_GUARD_FRAC * height)))
    x0 = max(0, min(width - 1, int(round(left)) + guard_x))
    x1 = max(x0 + 1, min(width, int(round(right)) - guard_x))
    y0 = max(0, min(height - 1, int(round(top)) + guard_y))
    y1 = max(y0 + 1, min(height, int(round(bottom)) - guard_y))
    inside = edges[y0:y1, x0:x1]
    if inside.size == 0:
        return 0.0

    outside_mask = np.ones((height, width), dtype=bool)
    ox0 = max(0, int(round(left)) - guard_x)
    ox1 = min(width, int(round(right)) + guard_x)
    oy0 = max(0, int(round(top)) - guard_y)
    oy1 = min(height, int(round(bottom)) + guard_y)
    outside_mask[oy0:oy1, ox0:ox1] = False
    outside = edges[outside_mask]
    if outside.size == 0:
        return float("inf")

    inside_density = float(np.count_nonzero(inside)) / float(inside.size)
    outside_density = float(np.count_nonzero(outside)) / float(outside.size)
    if inside_density <= 0.0:
        return 0.0
    return inside_density / max(outside_density, 1e-6)


def detect_structural_content_box(frame_img) -> Optional[List[float]]:
    """Find the main bordered content window in one stream frame.

    Returns a normalized ``[x, y, w, h]`` box.  Full-frame/no-border content
    returns ``None`` so ordinary fullscreen games keep the normal center crop.
    """
    if frame_img is None or getattr(frame_img, "size", 0) == 0:
        return None
    source_h, source_w = frame_img.shape[:2]
    scale = min(1.0, STRUCTURAL_MAX_WIDTH / max(1.0, float(source_w)))
    width = max(2, int(round(source_w * scale)))
    height = max(2, int(round(source_h * scale)))
    frame = cv2.resize(frame_img, (width, height)) if scale < 1.0 else frame_img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(45, width // 14),
        minLineLength=max(40, int(0.18 * width)),
        maxLineGap=max(8, int(0.025 * width)),
    )
    if lines is None:
        return None

    horizontal = []
    vertical = []
    horizontal_slop = max(3, int(round(0.008 * height)))
    vertical_slop = max(3, int(round(0.008 * width)))
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y2 - y1) <= horizontal_slop and abs(x2 - x1) >= 0.18 * width:
            horizontal.append(((y1 + y2) / 2.0, x1, x2))
        if abs(x2 - x1) <= vertical_slop and abs(y2 - y1) >= 0.18 * height:
            vertical.append(((x1 + x2) / 2.0, y1, y2))
    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    h_clusters = _cluster_lines(horizontal, max(4.0, 0.012 * height))
    v_clusters = _cluster_lines(vertical, max(4.0, 0.012 * width))
    candidates = []
    for left in v_clusters:
        for right in v_clusters:
            if right["position"] <= left["position"]:
                continue
            box_w = (right["position"] - left["position"]) / width
            if not 0.38 <= box_w <= 1.0:
                continue
            for top in h_clusters:
                for bottom in h_clusters:
                    if bottom["position"] <= top["position"]:
                        continue
                    box_h = (bottom["position"] - top["position"]) / height
                    if not 0.35 <= box_h <= 0.95:
                        continue
                    area = box_w * box_h
                    if not STRUCTURAL_MIN_AREA <= area <= STRUCTURAL_MAX_AREA:
                        continue

                    top_cov = _line_coverage(top, left["position"], right["position"])
                    bottom_cov = _line_coverage(bottom, left["position"], right["position"])
                    left_cov = _line_coverage(left, top["position"], bottom["position"])
                    right_cov = _line_coverage(right, top["position"], bottom["position"])
                    if top_cov < 0.45 or bottom_cov < 0.40 or max(left_cov, right_cov) < 0.45:
                        continue

                    edge_contrast = _edge_contrast(
                        edges,
                        left["position"], top["position"],
                        right["position"], bottom["position"],
                    )
                    # In a dark title such as Silent Hill, the actual game
                    # window is intentionally sparse while chat/widgets around
                    # it are edge-heavy. Large authored windows are allowed to
                    # win on geometry; small low-contrast HUD rectangles are
                    # still rejected.
                    if (
                        edge_contrast < STRUCTURAL_MIN_EDGE_CONTRAST
                        and area < STRUCTURAL_DARK_WINDOW_MIN_AREA
                    ):
                        continue

                    pixel_aspect = (
                        (right["position"] - left["position"])
                        / max(1.0, bottom["position"] - top["position"])
                    )
                    aspect_score = max(0.0, 1.0 - abs(pixel_aspect - 16.0 / 9.0) / (16.0 / 9.0))
                    if aspect_score < STRUCTURAL_MIN_ASPECT_SCORE:
                        continue
                    completeness = (top_cov + bottom_cov + left_cov + right_cov) / 4.0
                    edge_count = sum((
                        left["position"] < 0.02 * width,
                        right["position"] > 0.98 * width,
                        top["position"] < 0.02 * height,
                        bottom["position"] > 0.98 * height,
                    ))
                    confidence = (
                        1.4 * completeness
                        + 1.5 * aspect_score
                        + 0.35 * area
                        + 0.04 * min(edge_contrast, 3.0)
                        - 0.08 * edge_count
                    )
                    candidates.append((confidence, [
                        left["position"] / width,
                        top["position"] / height,
                        box_w,
                        box_h,
                    ]))

    if not candidates:
        return None
    _, best = max(candidates, key=lambda item: item[0])
    return [round(float(value), 4) for value in best]


def detect_gameplay_box_at(video_path: str, timestamp: float) -> Optional[List[float]]:
    """Detect the authored main-content window at one clip timestamp."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = cap.read()
        return detect_structural_content_box(frame) if ok else None
    except Exception:  # noqa: BLE001 - export layout detection is best-effort
        return None
    finally:
        cap.release()


def _motion_contrast(frame_a: np.ndarray, frame_b: np.ndarray, box: List[float]) -> float:
    """Mean short-term motion inside a candidate versus the surrounding canvas."""
    if frame_a is None or frame_b is None:
        return 0.0
    height, width = frame_a.shape[:2]
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b).astype(np.float32)
    x, y, w, h = (float(value) for value in box[:4])
    x0 = max(0, min(width - 1, int(round(x * width))))
    x1 = max(x0 + 1, min(width, int(round((x + w) * width))))
    y0 = max(0, min(height - 1, int(round(y * height))))
    y1 = max(y0 + 1, min(height, int(round((y + h) * height))))
    inside = diff[y0:y1, x0:x1]
    outside_mask = np.ones((height, width), dtype=bool)
    outside_mask[y0:y1, x0:x1] = False
    outside = diff[outside_mask]
    if inside.size == 0 or outside.size == 0:
        return 0.0
    return float(np.mean(inside)) / max(0.05, float(np.mean(outside)))


def _sample_count(duration: float, samples: Optional[int]) -> int:
    if samples is not None:
        return max(6, int(samples))
    return min(
        STRUCTURAL_MAX_SAMPLES,
        max(24, int(np.ceil(float(duration) / STRUCTURAL_SAMPLE_INTERVAL))),
    )


def detect_gameplay_layout_model(
    video_path: str,
    duration: float,
    samples: Optional[int] = None,
) -> GameplayLayoutModel:
    """Build persistent authored windows and an explicit scene timeline.

    A candidate must recur and contain more short-term motion than the canvas
    around it. Samples that do not resolve to a persistent authored window are
    retained as ``box=None`` so a fullscreen/transition scene cannot inherit a
    crop merely because that crop appeared earlier in the VOD.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or duration <= 1.0:
        return GameplayLayoutModel()
    count = _sample_count(duration, samples)
    edge_margin = min(5.0, 0.03 * float(duration))
    sample_times = np.linspace(edge_margin, max(edge_margin, duration - edge_margin), count)
    raw_observations: List[GameplayObservation] = []
    try:
        for timestamp in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame_a = cap.read()
            if not ok or frame_a is None:
                continue
            for _ in range(PAIR_GAP_FRAMES):
                cap.grab()
            ok_b, frame_b = cap.read()
            box = detect_structural_content_box(frame_a)
            if (
                not ok_b
                or frame_b is None
                or box is None
                or _motion_contrast(frame_a, frame_b, box) < STRUCTURAL_MIN_MOTION_CONTRAST
            ):
                box = None
            raw_observations.append(GameplayObservation(float(timestamp), box))
    except Exception:  # noqa: BLE001 - layout modeling is best-effort
        return GameplayLayoutModel()
    finally:
        cap.release()

    layouts: List[GameplayLayout] = []
    for observation in raw_observations:
        box = observation.box
        if box is None:
            continue
        match = next(
            (layout for layout in layouts if _box_iou(layout.box, box) >= STRUCTURAL_CLUSTER_IOU),
            None,
        )
        if match is None:
            layout = GameplayLayout(
                box=list(box), timestamps=[observation.timestamp], support=1, confidence=1.0,
            )
            setattr(layout, "_members", [box])
            layouts.append(layout)
            continue
        match.timestamps.append(observation.timestamp)
        match.support += 1
        members = getattr(match, "_members", [match.box])
        members.append(box)
        setattr(match, "_members", members)
        match.box = [
            round(float(np.median([member[i] for member in members])), 4)
            for i in range(4)
        ]

    total = max(1, len(raw_observations))
    # Scene layouts are allowed to be short relative to a multi-hour VOD. Two
    # motion-validated observations are enough; the singleton gate still drops
    # one-off transitions and decorative rectangles.
    min_support = STRUCTURAL_MIN_SUPPORT
    observed_interval = (
        float(np.median(np.diff([o.timestamp for o in raw_observations])))
        if len(raw_observations) > 1 else 0.0
    )
    qualified = []
    for layout in layouts:
        layout.confidence = round(layout.support / total, 4)
        if hasattr(layout, "_members"):
            delattr(layout, "_members")
        recurring = any(
            (later - earlier) <= max(1.0, observed_interval * 1.6)
            for earlier, later in zip(layout.timestamps, layout.timestamps[1:])
        )
        if layout.support >= min_support and recurring:
            qualified.append(layout)
    qualified.sort(key=lambda layout: layout.support, reverse=True)

    # Replace noisy per-frame geometry with the persistent representative and
    # keep non-matches as explicit fullscreen/unknown observations.
    observations = []
    for observation in raw_observations:
        match = None
        if observation.box is not None:
            match = max(
                qualified,
                key=lambda layout: _box_iou(layout.box, observation.box),
                default=None,
            )
            if match is not None and _box_iou(match.box, observation.box) < STRUCTURAL_CLUSTER_IOU:
                match = None
        observations.append(GameplayObservation(
            timestamp=observation.timestamp,
            box=list(match.box) if match is not None else None,
        ))

    # Fill a single missed observation only when the same authored layout is
    # present on both sides. This repairs a dark/static sample without bridging
    # an actual scene transition.
    for index in range(1, len(observations) - 1):
        if observations[index].box is not None:
            continue
        previous = observations[index - 1].box
        following = observations[index + 1].box
        if (
            previous is not None
            and following is not None
            and _box_iou(previous, following) >= STRUCTURAL_CLUSTER_IOU
        ):
            observations[index].box = list(previous)

    interval = (
        float(np.median(np.diff([o.timestamp for o in observations])))
        if len(observations) > 1 else 0.0
    )
    return GameplayLayoutModel(qualified, observations, interval)


def detect_gameplay_layouts(
    video_path: str,
    duration: float,
    samples: Optional[int] = None,
) -> List[GameplayLayout]:
    """Compatibility wrapper returning persistent authored geometries."""
    return detect_gameplay_layout_model(video_path, duration, samples).layouts


def gameplay_for_window(
    layouts: List[GameplayLayout],
    start: float,
    end: float,
    observations: Optional[List[GameplayObservation]] = None,
) -> Optional[List[float]]:
    """Choose the scene geometry nearest a clip's midpoint.

    When timeline observations are provided, ``None`` is meaningful: the
    nearest sampled scene was fullscreen/unknown and must not inherit a crop.
    """
    if not layouts:
        return None
    midpoint = (float(start) + float(end)) / 2.0
    if observations:
        nearest = min(observations, key=lambda observation: abs(observation.timestamp - midpoint))
        return list(nearest.box) if nearest.box is not None else None
    best = min(
        layouts,
        key=lambda layout: (
            min((abs(timestamp - midpoint) for timestamp in layout.timestamps), default=float("inf")),
            -layout.support,
        ),
    )
    return list(best.box)


def match_gameplay_layout(
    layouts: List[GameplayLayout],
    observed_box: Optional[List[float]],
) -> Optional[List[float]]:
    """Return a persistent representative only when a local box matches it."""
    if not layouts or observed_box is None:
        return None
    best = max(layouts, key=lambda layout: _box_iou(layout.box, observed_box))
    if _box_iou(best.box, observed_box) < STRUCTURAL_CLUSTER_IOU:
        return None
    return list(best.box)


def _largest_component(active: np.ndarray):
    """Largest 4-connected True component of a boolean grid, as a cell set."""
    best: set = set()
    seen = np.zeros_like(active, dtype=bool)
    rows, cols = active.shape
    for r in range(rows):
        for c in range(cols):
            if not active[r, c] or seen[r, c]:
                continue
            component = {(r, c)}
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and active[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        component.add((nr, nc))
                        stack.append((nr, nc))
            if len(component) > len(best):
                best = component
    return best


def detect_gameplay_box(
    video_path: str,
    duration: Optional[float] = None,
    facecam_box: Optional[List[float]] = None,
    samples: int = DEFAULT_SAMPLES,
) -> Optional[List[float]]:
    """Detect the gameplay viewport as a normalized [x, y, w, h] box, or None.

    None means "treat as fullscreen": detection failed, the video is too
    static to judge, or gameplay genuinely fills the frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        if duration is None or duration <= 0:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            duration = (n_frames / fps) if fps > 0 else 0.0
        if duration <= 10.0:
            return None

        size = GRID * CELL_PX
        motion_grids = []
        for t in np.linspace(0.06 * duration, 0.94 * duration, max(4, samples)):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame_a = cap.read()
            if not ok or frame_a is None:
                continue
            for _ in range(PAIR_GAP_FRAMES):
                cap.grab()
            ok, frame_b = cap.read()
            if not ok or frame_b is None:
                continue
            gray_a = cv2.cvtColor(cv2.resize(frame_a, (size, size)), cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(cv2.resize(frame_b, (size, size)), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_a, gray_b).astype(np.float32)
            cells = diff.reshape(GRID, CELL_PX, GRID, CELL_PX).mean(axis=(1, 3))
            motion_grids.append(cells)
    except Exception:  # noqa: BLE001 - detection is best-effort, never blocks a scan
        return None
    finally:
        cap.release()

    if len(motion_grids) < 4:
        return None

    median_grid = np.median(np.stack(motion_grids), axis=0)
    peak = float(np.percentile(median_grid, 95))
    if peak < MIN_MEDIAN_PEAK:
        return None

    active = median_grid >= max(1.0, ACTIVE_FRACTION * peak)

    # The facecam panel moves constantly too; mask it out so a cam window
    # adjacent to the game viewport can't fuse into the gameplay region.
    if facecam_box:
        fx, fy, fw, fh = (float(v) for v in facecam_box[:4])
        r_lo = max(0, min(GRID - 1, int(fy * GRID)))
        r_hi = max(0, min(GRID - 1, int((fy + fh) * GRID)))
        c_lo = max(0, min(GRID - 1, int(fx * GRID)))
        c_hi = max(0, min(GRID - 1, int((fx + fw) * GRID)))
        active[r_lo:r_hi + 1, c_lo:c_hi + 1] = False

    component = _largest_component(active)
    if not component:
        return None

    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    x = min(cols) / GRID
    y = min(rows) / GRID
    w = (max(cols) - min(cols) + 1) / GRID
    h = (max(rows) - min(rows) + 1) / GRID

    if w * h < MIN_REGION_AREA:
        return None
    if w >= FULLSCREEN_COVERAGE and h >= FULLSCREEN_COVERAGE:
        return None  # gameplay fills the frame; existing center-crop behavior is right

    candidate = [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]
    # A real windowed game capture is ~16:9 like the source it mirrors — every
    # authored layout the model has confirmed on real VODs measures ~1.78
    # pixel aspect. One-axis shrink and dual-axis quiet-edge shrinks (measured
    # Fortnite: [0.1667, 0.125, 0.7917, 0.875]) are hallucinations; cropping
    # to them zooms every export. Distrust -> fullscreen.
    if not is_plausible_windowed_gameplay(candidate):
        return None
    return candidate
