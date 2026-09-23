# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Whole-VOD facecam layout model.

The per-frame detector (engines/vision/detector.py) emits ONE smoothed facecam
box that SNAPS on a low-IoU jump. Two failure modes follow: (1) a transient false
positive -- a game character, a HUD portrait -- can hijack the box for a stretch,
and (2) a streamer who uses two cam layouts collapses to one.

This module decides the facecam location(s) ONCE from the whole VOD. It clusters
every frame's box by position and keeps only clusters that are:
  * PERSISTENT   -- present across a real fraction of the VOD (a game character
                    shows up briefly and scattered; a real cam is on for a large
                    share of the runtime),
  * SPATIALLY STABLE -- the cluster's boxes are tight around their median,
  * PLAUSIBLE    -- a corner/edge panel, not the whole frame (engines.clip.layout).

Each surviving cluster is a real layout. Each clip is then framed with the layout
its OWN window matches; if a clip's per-frame tracking wandered onto a character
(its window boxes match no real layout), it falls back to the dominant persistent
cam rather than cropping the character. A clip with no cam at all -> None (the
export renders full-frame gameplay).
"""

from dataclasses import dataclass, field, replace as dataclass_replace
from statistics import median
from typing import Callable, List, Optional

import numpy as np

from core.geometry import iou as _iou
from engines.clip.layout import _is_plausible_facecam

# Clustering + qualification thresholds. A real cam forms a big, tight cluster;
# a character/HUD blip forms a small, scattered, short-lived one.
CLUSTER_IOU = 0.4          # boxes overlapping this much are the same layout
MIN_SUPPORT_FRACTION = 0.12  # a layout must appear in >=12% of vision frames
MIN_SUPPORT_FRAMES = 8       # ...and at least this many frames (small-VOD guard)
STABILITY_MIN = 0.5          # mean IoU of the cluster's boxes to their median

# Secondary admission handles cameras that alternate positions and therefore
# split support across the recording. Require plausible shape, stability, and
# recurrence; stability alone also admits static overlays and thumbnail grids.
# This adds an admission path without weakening the primary gate.

ALT_SAMPLE_SHARE = 0.02      # scale the evidence floor on short scans only...
ALT_SUPPORT_CAP_FRAMES = 60  # ...but never make VOD length the deciding factor
ALT_STABILITY_MIN = 0.75     # a secondary layout must be unusually stable
ALT_MIN_SPAN_SHARE = 0.30    # and recur across a meaningful part of the stream
ALT_MIN_OCCUPIED_BINS = 3    # not merely two distant one-frame false positives
# Require coverage of timeline bins as well as first-to-last span; isolated
# bursts of a recurring game character can otherwise resemble a persistent camera.

ALT_MIN_COVERAGE = 0.08
CAM_ASPECT_MIN = 1.05        # w/h in PIXELS, so callers pass frame dimensions
CAM_ASPECT_MAX = 1.95        # 4:3 = 1.33, 16:9 = 1.78; chat/grids ran 2.0+

# Portrait cameras need a lower aspect floor than landscape cameras. Admit
# them only with primary support and stability, since weak portrait candidates
# can be standing game characters. The stricter alternate and tenure paths keep
# their existing aspect floor; very wide overlay strips remain excluded.

CAM_ASPECT_PORTRAIT_MIN = 0.55  # below this is a sliver, not a camera

# Tenure admission handles a camera present for a sustained exclusive stretch
# that is too short to satisfy whole-recording coverage. A genuine camera move
# vacates the established plate; a false rival usually overlaps it in time.
# Require a long run to exclude transient detections. Stability is intentionally
# not required here because frame-edge plates can fragment into interior contours.

TENURE_MIN_SECONDS = 180.0   # an unbroken stretch this long is a real scene
TENURE_MAX_GAP = 60.0        # detector dropout that does not end a stretch
TENURE_MAX_RIVAL = 0.10      # admitted cams may be ~absent during that stretch
TENURE_MIN_SELF = 0.60       # ...and this candidate must actually own it

# Pool occupancy over fragments contained within the candidate plate. A
# frame-edge camera can break into several co-located clusters when its contour
# is open. Spatial containment prevents unrelated rivals donating occupancy.
# Pooling changes only self-occupancy; run detection, support, box geometry and
# rival checks remain separate. This matches NESTED_LAYOUT_CONTAINMENT below.

TENURE_POOL_CONTAINMENT = 0.75

# Require tenure candidates to touch both a horizontal and vertical frame edge.
# Exclusivity alone also admits people inside full-camera scenes, which are
# handled separately by is_fullframe_camera_at. Allow modest detector undershoot.

TENURE_EDGE_TOLERANCE = 0.06
# For a clip window that matches no real layout (tracker wandered) or has no
# detections, fall back to the dominant cam only if it's on for at least this
# much of the VOD -- otherwise the cam may genuinely be off, so render full-frame.
DOMINANT_FALLBACK_FRACTION = 0.5
# Timeline coverage also supports the dominant fallback. Detection fraction
# can fall when the tracker wanders even though the camera remains visible.

DOMINANT_FALLBACK_COVERAGE = 0.6
# Short timeline bins avoid over-crediting cameras that toggle on and off.

COVERAGE_BIN_SEC = 10.0      # timeline bin size for the coverage measure
WINDOW_MATCH_IOU = 0.4       # a window box "belongs to" a layout at/above this IoU
# Old persisted timelines predate detector provenance. A raw peak box much
# larger than the layout it matched is characteristic of the 35%-padded person
# fallback and may use the same verified-window tiebreak as an explicitly tagged
# fallback sample. The verifier is still mandatory, so size alone changes none.
FALLBACK_OVERSIZE_RATIO = 1.6
FALLBACK_WINDOW_VOTE_RATIO = 2
# A direct-panel sample needs stronger window opposition than a person fallback
# before it is overridden. Require a decisive vote and a person verification at
# the peak, preserving real scene transitions where the proposed camera is absent.

PANEL_WINDOW_VOTE_RATIO = 4
# A camera can move for only a few seconds -- too briefly to become a safe
# whole-VOD layout -- while the panel detector remains locally consistent. In
# that narrow unresolved-window case, recover the established camera plate's
# size at the local vertical position. Direct panel provenance, a majority of
# the window's detections, edge placement, spatial stability, and a source-frame
# person check are all mandatory; missing any one fails closed to no camera.
WINDOW_LOCAL_PANEL_MIN_FRAMES = 3
WINDOW_LOCAL_PANEL_MIN_SHARE = 0.50
WINDOW_LOCAL_PANEL_STABILITY_MIN = 0.55
WINDOW_LOCAL_PANEL_EDGE_TOLERANCE = 0.12
WINDOW_LOCAL_MAX_KNOWN_SHARE = 0.20
# Collapse strongly nested layouts: open camera contours can yield separate
# clusters around interior chairs, faces or door frames. Containment identifies
# fragments while allowing genuinely distinct camera positions to survive.

NESTED_LAYOUT_CONTAINMENT = 0.75
# A decorated camera overlay can form a second, larger rectangle around the
# actual webcam (for example animated sprites hanging below the camera plate).
# Keep the inner geometry only when it is overwhelmingly the better detector
# result: strong panel provenance, substantially more observations/coverage,
# and a containing region dominated by the loose person fallback.  This stays
# well clear of the contourless-camera case, where the outer fallback plate has
# more support than the interior chair/face fragments and must remain the crop.
NESTED_INNER_PANEL_RATIO = 0.80
# Require a strong fallback majority before discarding an outer plate.
# A partly noisy real camera can still mix fallback and direct-panel evidence.

NESTED_OUTER_FALLBACK_RATIO = 0.70
NESTED_INNER_SUPPORT_RATIO = 1.50
NESTED_INNER_COVERAGE_MARGIN = 0.15
NESTED_REFINED_ASPECT_MIN_RATIO = 0.90


@dataclass
class FacecamLayout:
    box: List[float]     # representative [x, y, w, h] (cluster median)
    support: int         # frames in this cluster
    fraction: float      # support / total vision frames
    stability: float     # mean IoU of member boxes to the median (0..1)
    # Share of the VOD's TIMELINE (coarse bins) containing at least one member
    # frame. This separates the two situations ``fraction`` conflates: a cam
    # that is genuinely off half the runtime scores low on both, while a cam
    # that is always on with a noisy tracker scores low on fraction but ~1.0
    # here despite intermittent tracker mistakes.
    coverage: float = 0.0
    # Raw observation times assigned to this layout. Kept internal to framing
    # analysis so plate refinement samples only scenes where this camera exists.
    timestamps: List[float] = field(default_factory=list)
    panel_support: int = 0
    fallback_support: int = 0
    # The detector found a stronger real panel inside a fallback-derived
    # decorated rectangle. Edge refinement may still see the decorations as a
    # bottom border, so keep the panel cluster's aspect as a lower bound.
    preserve_panel_aspect: bool = False
    # Where the subject's head starts inside this plate, as a fraction of the
    # plate's own height (0 = flush with the top border). Measured by
    # :func:`measure_subject_tops`; None when unmeasured, which every renderer
    # must treat as "fall back to the old constant". See that function for why
    # camera aspect bounds apply.
    subject_top: Optional[float] = None


@dataclass
class FacecamObservation:
    """One sampled scene decision.

    ``layout_index=None`` used to collapse camera absence and tracker failure
    into one value.  That made a single game-character false positive at the
    editorial peak erase a facecam established by thousands of surrounding
    samples.  ``state`` keeps those cases distinct while remaining compatible
    with persisted v2 timelines whose state is unknown.
    """
    timestamp: float
    layout_index: Optional[int]
    state: str = "unknown"
    source: Optional[str] = None


@dataclass
class FacecamLayoutModel:
    layouts: List[FacecamLayout] = field(default_factory=list)
    observations: List[FacecamObservation] = field(default_factory=list)
    interval: float = 0.0


def _camera_shaped(box: List[float], frame_aspect: float) -> bool:
    """Is this box's PIXEL aspect ratio camera-like (roughly 4:3 to 16:9)?

    Boxes are normalized, so the frame's own aspect has to be folded back in:
    a 0.20 x 0.235 box on a 16:9 frame is 1.51 wide-to-tall, not 0.85.
    """
    w, h = float(box[2]), float(box[3])
    if w <= 0.0 or h <= 0.0:
        return False
    aspect = (w / h) * frame_aspect
    return CAM_ASPECT_MIN <= aspect <= CAM_ASPECT_MAX


def _facecam_pixel_aspect(box: List[float], frame_aspect: float) -> float:
    w, h = float(box[2]), float(box[3])
    return (w / max(h, 1e-6)) * frame_aspect


def _plate_shape_ok(box: List[float], frame_aspect: float) -> bool:
    """Is this box a plausible camera PLATE shape (landscape or portrait)?

    Used where the question is "could this rectangle be a camera at all", as
    opposed to `_camera_shaped`, which asks the stricter "is this the landscape
    shape a rescue path is allowed to admit on weak evidence".
    """
    return (
        CAM_ASPECT_PORTRAIT_MIN
        <= _facecam_pixel_aspect(box, frame_aspect)
        <= CAM_ASPECT_MAX
    )


def _valid_layout_shape(
    layout: "FacecamLayout", frame_aspect: float, min_fraction: float,
) -> bool:
    """Final shape invariant applied to every surviving layout.

    Landscape passes as before. Wide is always malformed -- that is where every
    measured false positive lives. Portrait is a real camera shape, but only
    with primary-path evidence behind it; see CAM_ASPECT_PORTRAIT_MIN.
    """
    if _camera_shaped(layout.box, frame_aspect):
        return True
    if not _plate_shape_ok(layout.box, frame_aspect):
        return False
    return layout.fraction >= min_fraction and layout.stability >= STABILITY_MIN


def detect_facecam_layouts(
    unified_signals,
    min_fraction: float = MIN_SUPPORT_FRACTION,
    min_frames: int = MIN_SUPPORT_FRAMES,
    frame_aspect: float = 16.0 / 9.0,
) -> List[FacecamLayout]:
    """Cluster whole-VOD facecam detections into real, persistent layouts."""
    vis = [s for s in unified_signals if getattr(s, "vision", None) is not None]
    total = len(vis)
    boxes = [s.vision.facecam_box for s in vis if s.vision.facecam_box]
    if not boxes or total == 0:
        return []

    # Carry timestamps alongside boxes so a cluster's temporal coverage can be
    # measured, not just its share of frames.
    stamped = [
        (
            s.vision.facecam_box,
            float(getattr(s, "timestamp", 0.0) or 0.0),
            getattr(s.vision, "facecam_source", None),
        )
        for s in vis if s.vision.facecam_box
    ]
    # Coverage denominator spans the WHOLE sampled timeline (every vision frame),
    # NOT just frames where a box was detected -- otherwise a cam present only in
    # the first 40% spans just that 40% and reports coverage 1.0, defeating the
    # point. Falls back to box timestamps only when
    # no vision timestamp is available.
    vis_times = [
        float(getattr(s, "timestamp", 0.0) or 0.0)
        for s in vis if getattr(s, "timestamp", None) is not None
    ]
    if not vis_times:
        vis_times = [t for _b, t, _source in stamped]
    span_lo, span_hi = (min(vis_times), max(vis_times)) if vis_times else (0.0, 0.0)
    bin_count = max(1, int((span_hi - span_lo) // COVERAGE_BIN_SEC) + 1)

    clusters: List[dict] = []
    for b, t, source in stamped:
        for c in clusters:
            if _iou(b, c["anchor"]) >= CLUSTER_IOU:
                c["boxes"].append(b)
                c["times"].append(t)
                c["sources"].append(source)
                break
        else:
            clusters.append({
                "anchor": b, "boxes": [b], "times": [t], "sources": [source],
            })

    largest_support = max((len(c["boxes"]) for c in clusters), default=0)

    alt_support = max(
        min_frames,
        min(ALT_SUPPORT_CAP_FRAMES, int(np.ceil(total * ALT_SAMPLE_SHARE))),
    )

    layouts: List[FacecamLayout] = []
    occupied_bins: List[set] = []
    # Clusters that clear shape/size but miss both whole-VOD gates. They are
    # re-examined once the admitted set is known, because the third path scores
    # a candidate against the cams already accepted rather than against the VOD.
    deferred: List[dict] = []
    # Every cluster's shape, including ones rejected below. Tenure occupancy is
    # pooled over these: the fragments of a contourless plate are frequently the
    # small, unstable, implausible-on-their-own clusters that never reach
    # ``deferred`` at all, and those frames are still the camera being on screen.
    cluster_shapes: List[tuple] = []
    for c in clusters:
        member = c["boxes"]
        n = len(member)
        median = [float(np.median([b[i] for b in member])) for i in range(4)]
        cluster_shapes.append((median, c["times"]))
        # A near-full-frame / off-corner cluster is not a cam panel (a game
        # character or the whole scene) -- reject before the support gate so it
        # can't win on volume alone.
        if not _is_plausible_facecam(median):
            continue
        fraction = n / total
        if n < min_frames:
            continue
        stability = float(np.mean([_iou(b, median) for b in member]))
        camera_shaped = _camera_shaped(median, frame_aspect)
        panel_support = sum(source == "panel" for source in c["sources"])
        fallback_support = sum(source == "person_fallback" for source in c["sources"])
        source_known = panel_support + fallback_support == n
        # A loose person detector can repeatedly lock onto a corner HUD or game
        # character. It may still establish the dominant camera on contourless
        # overlays, but a secondary fallback-only cluster has to prove a real,
        # exclusive camera tenure via the third path below.
        fallback_only_alternate = (
            source_known and fallback_support == n and n < largest_support
        )
        occupied = {
            int((t - span_lo) // COVERAGE_BIN_SEC) for t in c["times"]
        }
        coverage = len(occupied) / bin_count
        record = {
            "median": median, "n": n, "fraction": fraction,
            "stability": stability, "coverage": coverage,
            "occupied": occupied, "times": c["times"],
            "camera_shaped": camera_shaped,
            "panel_support": panel_support,
            "fallback_support": fallback_support,
        }
        if fallback_only_alternate:
            deferred.append(record)
            continue
        if fraction >= min_fraction:
            if stability < STABILITY_MIN:
                deferred.append(record)
                continue
        else:
            # Secondary path: a camera-shaped, unusually stable cluster that
            # RECURS across the stream, and only misses the share gate because
            # this VOD has several cams. Never widens what the primary path
            # already admits.
            times = c["times"]
            span = (max(times) - min(times)) if times else 0.0
            vod_span = max(1e-6, span_hi - span_lo)
            # The old secondary gate was another whole-VOD fraction (10%). On
            # the latest 4h job a real layout had 1,434 stable observations and
            # lost because 1,434 / 14,352 rounds to 9.9916%. Scale the evidence
            # requirement for small fixtures, then cap it at one minute of 1Hz
            # observations so VOD duration cannot erase a well-established cam.
            if not (
                n >= alt_support
                and stability >= ALT_STABILITY_MIN
                and (span / vod_span) >= ALT_MIN_SPAN_SHARE
                and len(occupied) >= ALT_MIN_OCCUPIED_BINS
                and coverage >= ALT_MIN_COVERAGE
                and camera_shaped
            ):
                deferred.append(record)
                continue
        layouts.append(FacecamLayout(
            box=[round(x, 4) for x in median],
            support=n,
            fraction=round(fraction, 4),
            stability=round(stability, 4),
            coverage=round(coverage, 4),
            timestamps=[float(t) for t in c["times"]],
            panel_support=panel_support,
            fallback_support=fallback_support,
        ))
        occupied_bins.append(occupied)

    # Third path: a cluster that held an unbroken stretch of the stream with no
    # admitted camera anywhere in it. See TENURE_MIN_SECONDS.
    admitted_times = sorted(t for record in layouts for t in record.timestamps)
    for record in deferred:
        if (
            not record["camera_shaped"]
            or record["n"] < alt_support
            or not _corner_anchored(record["median"])
        ):
            continue
        # Pool the frames of every cluster whose box sits inside this plate --
        # the interior fragments (chair, face, door frame) the detector produces
        # when it cannot close a contour around a flush-edge camera. Spatially
        # scoped, so nothing outside this plate can donate occupancy.
        pooled_times = [
            t
            for shape, shape_times in cluster_shapes
            if _containment(shape, record["median"]) >= TENURE_POOL_CONTAINMENT
            for t in shape_times
        ]
        tenure = _exclusive_tenure(
            sorted(record["times"]),
            admitted_times,
            vis_times,
            occupancy_times=sorted(pooled_times) or None,
        )
        if tenure is None:
            continue
        layouts.append(FacecamLayout(
            box=[round(x, 4) for x in record["median"]],
            support=record["n"],
            fraction=round(record["fraction"], 4),
            stability=round(record["stability"], 4),
            coverage=round(record["coverage"], 4),
            timestamps=[float(t) for t in record["times"]],
            panel_support=record["panel_support"],
            fallback_support=record["fallback_support"],
        ))
        occupied_bins.append(record["occupied"])

    # Shape is a layout invariant, not merely an alternate-layout rescue
    # condition. Keep malformed primary clusters through nested-layout folding
    # so an inner person/chair fragment can still contribute support to its
    # valid outer camera plate; then drop any malformed survivor. A valid inner
    # camera is never absorbed by a malformed outer region (see collapse guard).
    layouts = _collapse_nested_layouts(
        layouts,
        occupied_bins,
        total,
        bin_count,
        frame_aspect=frame_aspect,
    )
    layouts = [
        layout for layout in layouts
        if _valid_layout_shape(layout, frame_aspect, min_fraction)
    ]
    layouts.sort(key=lambda L: L.support, reverse=True)
    return layouts


def _corner_anchored(box: List[float]) -> bool:
    """Does this box touch one vertical AND one horizontal frame edge?

    See TENURE_EDGE_TOLERANCE: this is what keeps a person detected inside a
    full-frame camera shot from being admitted as a corner cam panel.
    """
    x, y, w, h = (float(v) for v in box[:4])
    vertical = min(x, 1.0 - (x + w)) <= TENURE_EDGE_TOLERANCE
    horizontal = min(y, 1.0 - (y + h)) <= TENURE_EDGE_TOLERANCE
    return vertical and horizontal


def _edge_anchored(box: List[float]) -> bool:
    """Whether a refined panel still hugs at least one authored frame edge."""
    x, y, w, h = (float(v) for v in box[:4])
    return min(x, y, 1.0 - (x + w), 1.0 - (y + h)) <= TENURE_EDGE_TOLERANCE


# Half-width of the neighbourhood the local tenancy measure reads. Wide enough
# that a clip-length stretch of detector dropout cannot empty it, short enough
# that a cam move earlier or later in the VOD is not what gets measured. The
# separation it produces is reported at the call site in ``facecam_for_window``.
LOCAL_TENANCY_SEC = 90.0
# Reuses TENURE_MAX_RIVAL's meaning: an admitted cam may be ~absent from a
# stretch it does not own. A rival that coexists with it does not qualify.
LOCAL_TENANCY_MAX_RIVAL = TENURE_MAX_RIVAL


def _dominant_vacated_locally(
    observations: Optional[List["FacecamObservation"]],
    layout_index: int,
    target: float,
) -> bool:
    """Did the dominant layout stop being detected around this frame?

    True means the old plate is empty here -- the signature of a real camera
    move, and the only case that may skip the rival check. A HUD panel that
    merely shares the frame with the camera scores far from this.
    """
    if not observations:
        return False
    mine = 0
    dominant = 0
    for item in observations:
        if abs(item.timestamp - target) > LOCAL_TENANCY_SEC:
            continue
        if item.layout_index == layout_index:
            mine += 1
        elif item.layout_index == 0:
            dominant += 1
    if mine == 0:
        return False
    return dominant / (mine + dominant) <= LOCAL_TENANCY_MAX_RIVAL


def _exclusive_tenure(
    times: List[float],
    admitted_times: List[float],
    vis_times: List[float],
    occupancy_times: Optional[List[float]] = None,
) -> Optional[tuple]:
    """The candidate's longest unbroken stretch, if it held it alone.

    Returns ``(start, end)`` when the stretch is long enough AND no already
    admitted camera was meaningfully present in it, else None. See the
    TENURE_* constants for why tenancy is the signal that separates a camera
    the streamer moved from a game character the tracker latched onto.

    ``occupancy_times`` supplies the frames counted for the TENURE_MIN_SELF
    occupancy test, pooled over co-located fragments of the same plate (see
    TENURE_POOL_CONTAINMENT). It defaults to ``times``, so callers that do not
    pool behave exactly as before. The stretch itself is always found from
    ``times``: pooling answers "was this camera on screen", not "how long did
    this cluster run".
    """
    if not times:
        return None

    # Runs are broken by a gap larger than a detector dropout, not by every
    # missed frame: the detector drops the plate constantly on a flush-edge cam,
    # which is the exact case this path exists to recover.
    best = run_start = previous = times[0]
    best_end = times[0]
    for timestamp in times[1:]:
        if timestamp - previous > TENURE_MAX_GAP:
            if previous - run_start > best_end - best:
                best, best_end = run_start, previous
            run_start = timestamp
        previous = timestamp
    if previous - run_start > best_end - best:
        best, best_end = run_start, previous
    if best_end - best < TENURE_MIN_SECONDS:
        return None

    # Denominator is sampled frames in the window, so a stretch the scan barely
    # sampled cannot manufacture a high share from a handful of observations.
    sampled = sum(1 for t in vis_times if best <= t <= best_end)
    if sampled <= 0:
        return None
    rival = sum(1 for t in admitted_times if best <= t <= best_end) / sampled
    mine_times = times if occupancy_times is None else occupancy_times
    mine = sum(1 for t in mine_times if best <= t <= best_end) / sampled
    if rival > TENURE_MAX_RIVAL or mine < TENURE_MIN_SELF:
        return None
    return (best, best_end)


def _observation_layout_index(
    box: Optional[List[float]], layouts: List[FacecamLayout],
) -> Optional[int]:
    if not box or not layouts:
        return None
    best_index = None
    best_score = WINDOW_MATCH_IOU
    for index, layout in enumerate(layouts):
        # Raw detections sometimes describe the person or chair INSIDE a plate.
        # Nested clusters are folded into the plate, so containment is valid
        # membership evidence even where raw-box IoU is deliberately small.
        score = max(_iou(box, layout.box), _containment(box, layout.box))
        if score >= best_score:
            best_score = score
            best_index = index
    return best_index


def detect_facecam_layout_model(
    unified_signals,
    min_fraction: float = MIN_SUPPORT_FRACTION,
    min_frames: int = MIN_SUPPORT_FRAMES,
    frame_aspect: float = 16.0 / 9.0,
) -> FacecamLayoutModel:
    """Build persistent facecam geometries plus an explicit scene timeline."""
    vis = [s for s in unified_signals if getattr(s, "vision", None) is not None]
    layouts = detect_facecam_layouts(
        unified_signals,
        min_fraction=min_fraction,
        min_frames=min_frames,
        frame_aspect=frame_aspect,
    )
    observations = []
    for signal in vis:
        raw_box = getattr(signal.vision, "facecam_box", None)
        layout_index = _observation_layout_index(raw_box, layouts)
        # A missing raw box is not proof that the camera was switched off; it
        # is another detector miss until source-frame verification says what is
        # actually inside the established plate. ``absent`` is reserved for a
        # future positive scene classifier, never inferred from no detection.
        state = "matched" if layout_index is not None else "unresolved"
        observations.append(FacecamObservation(
            timestamp=float(getattr(signal, "timestamp", 0.0) or 0.0),
            layout_index=layout_index,
            state=state,
            source=getattr(signal.vision, "facecam_source", None),
        ))

    # Repair only a tiny detector dropout bracketed by the SAME layout. Longer
    # gaps remain explicit None; that is the boundary that keeps full-camera
    # scenes from inheriting a corner crop used elsewhere in the VOD.
    index = 0
    while index < len(observations):
        if observations[index].layout_index is not None:
            index += 1
            continue
        gap_start = index
        while index < len(observations) and observations[index].layout_index is None:
            index += 1
        gap_end = index
        previous = observations[gap_start - 1].layout_index if gap_start > 0 else None
        following = observations[gap_end].layout_index if gap_end < len(observations) else None
        if gap_end - gap_start <= 2 and previous is not None and previous == following:
            for fill in range(gap_start, gap_end):
                observations[fill].layout_index = previous
                observations[fill].state = "matched"

    interval = (
        float(np.median(np.diff([item.timestamp for item in observations])))
        if len(observations) > 1 else 0.0
    )
    return FacecamLayoutModel(layouts, observations, interval)


def _containment(inner: List[float], outer: List[float]) -> float:
    """Share of ``inner``'s area that lies inside ``outer`` (0..1)."""
    ix = max(0.0, min(inner[0] + inner[2], outer[0] + outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[1] + inner[3], outer[1] + outer[3]) - max(inner[1], outer[1]))
    area = inner[2] * inner[3]
    return (ix * iy) / area if area > 0 else 0.0


def _collapse_nested_layouts(
    layouts: List[FacecamLayout],
    occupied_bins: List[set],
    total: int,
    bin_count: int,
    *,
    frame_aspect: float = 16.0 / 9.0,
) -> List[FacecamLayout]:
    """Fold every layout that sits inside a larger one into that larger one.

    A cam panel does not contain another cam panel: the inner box is the detector
    latching onto structure INSIDE the camera image. The outer box survives with
    the inner one's support and timeline coverage added, because both were the
    same physical cam all along -- and those counts feed the fallback gates that
    decide whether a clip gets a cam at all.
    """
    areas = [L.box[2] * L.box[3] for L in layouts]
    camera_shaped = [_camera_shaped(layout.box, frame_aspect) for layout in layouts]
    absorbed = {}

    def _prefer_inner_camera(inner: FacecamLayout, outer: FacecamLayout) -> bool:
        inner_known = inner.panel_support + inner.fallback_support
        outer_known = outer.panel_support + outer.fallback_support
        if inner_known < inner.support or outer_known < outer.support:
            return False
        inner_panel_ratio = inner.panel_support / max(1, inner_known)
        outer_fallback_ratio = outer.fallback_support / max(1, outer_known)
        return (
            inner_panel_ratio >= NESTED_INNER_PANEL_RATIO
            and outer_fallback_ratio >= NESTED_OUTER_FALLBACK_RATIO
            and inner.support >= outer.support * NESTED_INNER_SUPPORT_RATIO
            and inner.coverage >= outer.coverage + NESTED_INNER_COVERAGE_MARGIN
        )

    # Usually the largest containing rectangle is the camera plate. A decorated
    # overlay is the measured exception: the true panel is nested inside a
    # larger fallback-derived camera+decoration box. Reverse that one absorption
    # edge so the outer observations still count, but the panel geometry wins.
    preferred_inner = {}
    for j, outer in enumerate(layouts):
        candidates = [
            i for i, inner in enumerate(layouts)
            if i != j
            and areas[i] < areas[j]
            and camera_shaped[i]
            and camera_shaped[j]
            and _containment(inner.box, outer.box) >= NESTED_LAYOUT_CONTAINMENT
            and _prefer_inner_camera(inner, outer)
        ]
        if candidates:
            preferred_inner[j] = max(
                candidates,
                key=lambda i: (
                    layouts[i].panel_support,
                    layouts[i].support,
                    layouts[i].coverage,
                ),
            )
            absorbed[j] = preferred_inner[j]

    for i, inner in enumerate(layouts):
        best_j, best_area = None, 0.0
        for j, outer in enumerate(layouts):
            if i == j or areas[j] <= areas[i]:
                continue
            # A wide malformed overlay must not absorb a valid camera. Portrait cameras
            # remain valid containers: landscape fragments inside them can be interior
            # face detections rather than separate camera plates.

            if camera_shaped[i] and not _plate_shape_ok(outer.box, frame_aspect):
                continue
            if _containment(inner.box, outer.box) < NESTED_LAYOUT_CONTAINMENT:
                continue
            if preferred_inner.get(j) == i:
                continue
            if areas[j] > best_area:
                best_j, best_area = j, areas[j]
        if best_j is not None:
            absorbed[i] = best_j

    # Follow chains (an inner box inside an inner box) to the outermost survivor.
    def _root(i: int) -> int:
        seen = set()
        while i in absorbed and i not in seen:
            seen.add(i)
            i = absorbed[i]
        return i

    if not absorbed:
        return layouts

    decorated_panel_roots = set(preferred_inner.values())
    merged: List[FacecamLayout] = []
    for j, layout in enumerate(layouts):
        if j in absorbed:
            continue
        support = layout.support
        panel_support = layout.panel_support
        fallback_support = layout.fallback_support
        bins = set(occupied_bins[j])
        timestamps = list(layout.timestamps)
        for i in absorbed:
            if _root(i) == j:
                support += layouts[i].support
                panel_support += layouts[i].panel_support
                fallback_support += layouts[i].fallback_support
                bins |= occupied_bins[i]
                timestamps.extend(layouts[i].timestamps)
        merged.append(dataclass_replace(
            layout,
            support=support,
            fraction=round(support / total, 4) if total else layout.fraction,
            coverage=round(len(bins) / bin_count, 4) if bin_count else layout.coverage,
            timestamps=sorted(timestamps),
            panel_support=panel_support,
            fallback_support=fallback_support,
            preserve_panel_aspect=(
                layout.preserve_panel_aspect or j in decorated_panel_roots
            ),
        ))
    return merged


def _preserve_decorated_panel_aspect(
    layout: FacecamLayout,
    candidate: List[float],
    frame_aspect: float,
) -> List[float]:
    """Stop a sprite tray from becoming the refined camera's bottom edge."""
    if not layout.preserve_panel_aspect:
        return candidate
    original = [float(value) for value in layout.box[:4]]
    original_aspect = _facecam_pixel_aspect(original, frame_aspect)
    candidate_aspect = _facecam_pixel_aspect(candidate, frame_aspect)
    if candidate_aspect >= original_aspect * NESTED_REFINED_ASPECT_MIN_RATIO:
        return candidate
    x, y, w, h = (float(value) for value in candidate[:4])
    capped_h = (w * frame_aspect) / max(original_aspect, 1e-6)
    if capped_h >= h:
        return candidate
    capped = [x, y, w, capped_h]
    return capped if _is_plausible_facecam(capped) else candidate


# Refine plate borders using persistent temporal-mean gradients. Detector
# boxes can overshoot camera geometry, while motion alone cannot separate a
# camera floating over gameplay. Fixed straight borders survive temporal averaging.

REFINE_SAMPLES = 11          # frames sampled across each layout's own timeline
REFINE_MARGIN = 0.14         # per-side search margin (fraction of box dims)
REFINE_MIN_KEEP = 0.45       # refined box keeps at least this area fraction
REFINE_MAX_GROW = 1.30       # ...and at most this (edges just outside the box)
REFINE_EDGE_PROMINENCE = 2.2  # a snap line must beat the profile median by this
# A rectangle's chosen side has to clear the same bar, so that a claimed side
# admitted as a fallback cannot pass for a border it does not have.
PLATE_SIDE_MIN_PROMINENCE = REFINE_EDGE_PROMINENCE
_REFINE_ANALYSIS_W = 320     # analysis crop width (px)

# Relocate before edge snapping when the true plate lies outside the snap
# margin. Temporal variation can identify live feed within a static authored
# panel. If motion spans the search region, decline relocation and let the
# edge snap handle a camera floating over moving gameplay.

RELOCATE_MARGIN = 0.60        # per-side search margin (fraction of box dims)
RELOCATE_MIN_MOTION = 5.0     # mean |temporal std| below this => region is static
RELOCATE_MIN_IOU = 0.50       # never relocate onto a different panel entirely
RELOCATE_MIN_KEEP = 0.40      # relocated box area vs original, lower bound...
RELOCATE_MAX_GROW = 2.00      # ...and upper (this stage exists to fix undershoot)
RELOCATE_MAX_AREA_FRAC = 0.85  # motion spanning ~all of the search area => decline
RELOCATE_MIN_FILL = 0.50      # a plate is a solid block of motion, not a streak

# Score candidate rectangles using border support and a flat camera-aspect
# prior. An adjacent overlay border can be stronger than the camera border.
# Containment of the moving subject prevents rectangles that crop into the feed.

PLATE_ASPECT_MIN = CAM_ASPECT_MIN  # cropped/portrait-ish plates remain neutral
PLATE_ASPECT_MAX = 16.0 / 9.0
# Keep the aspect prior flat inside the normal camera range so it does not
# prefer an interior rectangle merely for being closer to widescreen.

PLATE_ASPECT_TOL = 0.30
# ...and hard-floored, so the prior can only ever break a near-tie between border
# lines. Unfloored it reaches a 4.4x penalty at aspect 1.14, enough to drag a box
# onto a weak spurious line purely to reach 16:9 -- which is what it did to the
# 80x70 synthetic plate in RefineLayoutBoxesTests. Floored at 0.55 the prior
# spans at most 1.8x, so a rectangle whose borders are genuinely better supported
# always wins and a non-16:9 cam keeps its real shape.
PLATE_ASPECT_PRIOR_FLOOR = 0.55
PLATE_MAX_LINES = 7           # strongest border lines considered per axis
# A box side this close to the frame boundary is pinned -- see _locate_plate_by_edges.
PLATE_EDGE_PIN_PX = 12
PLATE_LINE_MIN_SEP = 8        # px; keeps a 2px border from counting twice
# The winning rectangle must beat the claimed box's own border support by this
# much before anything is moved. Without it, noise reshapes already-good boxes.
PLATE_MIN_GAIN = 1.25
# Filter weak candidate sides before averaging border support. A minimum-only
# score is brittle when one legitimate camera edge has low contrast.

PLATE_SIDE_MIN_FRAC = 0.5
# Run refinement on every layout; detection fraction does not reliably identify
# boxes needing repair. Pin frame-flush sides before searching so interior detail
# cannot pull the crop inward. PLATE_MIN_GAIN decides whether geometry changes.
# For paired decorative borders, tighten to the innermost line containing the feed.

PLATE_SEED_TOL = 2            # px slack when testing "still contains the feed"
PLATE_RING_FRAC = 0.06        # how far inward a border ring's inner edge can sit
# Clamp to the feed only when it fills most of the tightened plate. A still,
# dark room can leave motion concentrated on the subject and is not a crop boundary.

PLATE_SEED_FILL = 0.75
# A candidate side has to clear the same bar that makes a line a line
# (REFINE_EDGE_PROMINENCE). Only the claimed sides -- admitted unconditionally
# below so that leaving the plate alone stays an option -- can fall under it.
# Scoring averages the four sides and then multiplies by the aspect prior, so a
# side with no edge beneath it still wins whenever it makes the rectangle more
# camera-shaped: on a real VOD the plate's true right edge (prominence 10.3)
# lost to the claim's edge sitting 70px out in the gameplay, which is why the
# export pulled a black bar in beside the camera. A pinned side is exempt,
# because a screen-clipped boundary has no line to measure.
#
# ...and the same clipping hides a flush plate's side from the search entirely
# when the claim starts INSIDE the camera: there is no line at the frame
# boundary to land on, so the side stays stranded on whatever decor sits at the
# claim (measured on a VOD whose cam was flush left -- the export cropped into
# the room and pulled gameplay in on the other side). What does reach past the
# claim is the plate's own top and bottom borders: they run across the missing
# strip and stop where the camera stops.
PLATE_FLUSH_CONTINUE_FRAC = 0.45  # outside-strip border support vs inside
PLATE_FLUSH_MAX_FRAC = 0.60       # strip size ceiling, vs the plate's own span
PLATE_FLUSH_MIN_PX = 8            # too short a span to average meaningfully


def _persistent_edge_profiles(
    stack: np.ndarray,
    row_span: Optional[tuple] = None,
    col_span: Optional[tuple] = None,
):
    """(col_profile, row_profile) of temporal-mean gradient magnitude.

    ``stack`` is (frames, rows, cols) float32 luma. A stable vertical border
    peaks in the column profile; a stable horizontal border in the row profile.

    A rectangle's vertical border exists only along that rectangle's own HEIGHT,
    so the column profile is averaged over ``row_span`` alone (and the row
    profile over ``col_span``). Averaging the whole search window instead
    dilutes a real border with whatever else the window happens to contain, and
    the window is deliberately much larger than the plate: it spans
    +/-RELOCATE_MARGIN on every side.

    Restricting the averaging span prevents adjacent gameplay from diluting
    camera borders. Additional temporal samples cannot fix spatial dilution.

    Spans are given in the stack's own coordinates and never shift the returned
    profiles' index space, so callers keep addressing lines by window pixel.
    """
    rows = slice(None) if row_span is None else slice(*row_span)
    cols = slice(None) if col_span is None else slice(*col_span)
    dx = np.abs(np.diff(stack[:, rows, :], axis=2)).mean(axis=0)  # (rows, cols-1)
    dy = np.abs(np.diff(stack[:, :, cols], axis=1)).mean(axis=0)  # (rows-1, cols)
    return dx.mean(axis=0), dy.mean(axis=1)


def _snap_side(profile: np.ndarray, lo: int, hi: int, fallback: int) -> int:
    """Index of the strongest prominent persistent edge in [lo, hi), else fallback."""
    lo = max(0, lo)
    hi = min(len(profile), hi)
    if hi - lo < 2:
        return fallback
    window = profile[lo:hi]
    baseline = float(np.median(profile)) + 1e-6
    peak = int(np.argmax(window))
    if float(window[peak]) < REFINE_EDGE_PROMINENCE * baseline:
        return fallback
    return lo + peak


def _relocate_to_live_feed(cv2, frames, box, width, height):
    """Move ``box`` onto the region that actually moves, or None to decline.

    Searches a neighborhood far wider than the edge snap's, because this stage
    exists to fix boxes that are wrong by more than the snap can reach.
    """
    x, y, w, h = (float(v) for v in box[:4])
    mx, my = w * RELOCATE_MARGIN, h * RELOCATE_MARGIN
    x0 = max(0, int((x - mx) * width))
    y0 = max(0, int((y - my) * height))
    x1 = min(width, int((x + w + mx) * width))
    y1 = min(height, int((y + h + my) * height))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None

    stack = np.stack([
        cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        for f in frames
    ])
    motion = stack.std(axis=0)
    if float(motion.mean()) < RELOCATE_MIN_MOTION:
        return None  # nothing in here moves: camera off, or a still frame region

    # Otsu, not a percentile: the plate's share of the search area is unknown, so
    # a fixed percentile either fragments the plate or swallows the chrome.
    scaled = np.clip(motion / max(1e-6, float(motion.max())) * 255.0, 0, 255)
    level, mask = cv2.threshold(
        scaled.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if level / 255.0 * float(motion.max()) < RELOCATE_MIN_MOTION:
        return None  # the split landed in noise, not on a real static/live edge

    # Close across compression blocks and momentarily-still parts of the subject
    # (hair, a held pose) so the feed reads as one region.
    kernel_px = max(9, int(round(min(x1 - x0, y1 - y0) * 0.06)) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_px, kernel_px)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Choose the moving blob with the strongest overlap inside this plate, not
    # the largest blob nearby. Gameplay can dominate area but cannot constrain a
    # valid camera rectangle. A merged camera-and-gameplay blob remains ambiguous.

    def _inside_plate(contour) -> float:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        ox = max(0.0, min((x0 + cx + cw) / width, x + w) - max((x0 + cx) / width, x))
        oy = max(0.0, min((y0 + cy + ch) / height, y + h) - max((y0 + cy) / height, y))
        return ox * oy

    blob = max(contours, key=_inside_plate)
    if _inside_plate(blob) <= 0.0:
        return None  # nothing that moves is inside the plate: no feed to seed with
    bx, by, bw, bh = cv2.boundingRect(blob)
    if bw < 16 or bh < 16:
        return None
    search_area = float((x1 - x0) * (y1 - y0))
    if (bw * bh) / search_area > RELOCATE_MAX_AREA_FRAC:
        return None  # motion fills the search area -> cam floats over gameplay
    # Motion can merge camera, adjacent gameplay, and scrolling chat into one
    # large seed. Bounding it by size alone can damage framing; separating those
    # regions requires evidence beyond motion.

    if float(cv2.contourArea(blob)) / float(bw * bh) < RELOCATE_MIN_FILL:
        return None  # a sweeping streak / scattered HUD, not a solid plate

    return [
        (x0 + bx) / width,
        (y0 + by) / height,
        bw / width,
        bh / height,
    ]


def _prominent_lines(profile: np.ndarray, limit: int, min_sep: int):
    """Strongest persistent border lines in a profile, as (index, prominence)."""
    baseline = float(np.median(profile)) + 1e-6
    picked = []
    for i in np.argsort(profile)[::-1]:
        i = int(i)
        if float(profile[i]) < REFINE_EDGE_PROMINENCE * baseline:
            break
        if all(abs(i - p) >= min_sep for p, _ in picked):
            picked.append((i, float(profile[i]) / baseline))
        if len(picked) >= limit:
            break
    if not picked:
        return picked
    strongest = max(p for _i, p in picked)
    return [(i, p) for i, p in picked if p >= PLATE_SIDE_MIN_FRAC * strongest]


def _crossing_strength(stack: np.ndarray, index: int, axis: int) -> np.ndarray:
    """Gradient ACROSS one border line, sampled along that line's length.

    ``axis=0`` reads ``index`` as a row (a horizontal border, one value per
    column); ``axis=1`` reads it as a column (a vertical border, per row).
    Unlike :func:`_persistent_edge_profiles` nothing is averaged along the
    line, because where the line stops is exactly the question.
    """
    if axis == 0:
        index = max(1, min(stack.shape[1] - 1, int(index)))
        return np.abs(stack[:, index, :] - stack[:, index - 1, :]).mean(axis=0)
    index = max(1, min(stack.shape[2] - 1, int(index)))
    return np.abs(stack[:, :, index] - stack[:, :, index - 1]).mean(axis=0)


def _line_contrast(stack: np.ndarray, index: int, axis: int) -> np.ndarray:
    """How far one border line stands out from the texture beside it.

    Raw crossing strength cannot answer "is there a line here": a busy region
    reads high everywhere. The neighbours a few pixels either side give the
    local floor, and what is left is the line itself.

    A border is a stroke, not a single pixel, and its steepest crossing sits
    at the stroke's own edge -- reading the exact index instead lands *inside*
    a 2px border, where there is nothing to cross.
    """
    limit = (stack.shape[1] if axis == 0 else stack.shape[2]) - 1

    def _at(offsets):
        usable = [o for o in offsets if 1 <= index + o <= limit]
        if not usable:
            return None
        return np.stack(
            [_crossing_strength(stack, index + o, axis) for o in usable])

    near = _at((-2, -1, 0, 1, 2))
    floor = _at((-8, -7, -6, -5, 5, 6, 7, 8))
    if near is None or floor is None:
        return np.zeros(stack.shape[2] if axis == 0 else stack.shape[1])
    return np.maximum(near.max(axis=0) - np.median(floor, axis=0), 0.0)


def _border_continues(
    stack: np.ndarray,
    first: int,
    second: int,
    axis: int,
    inside: tuple,
    outside: tuple,
) -> bool:
    """Do the plate's two perpendicular borders run across ``outside`` too?

    ``first`` / ``second`` are the perpendicular border lines and ``axis`` is
    their own axis (0 = they are rows). ``inside`` / ``outside`` are spans
    measured ALONG those lines: the plate's current extent, and the strip
    between it and the frame boundary.
    """
    if (outside[1] - outside[0] < PLATE_FLUSH_MIN_PX
            or inside[1] - inside[0] < PLATE_FLUSH_MIN_PX):
        return False
    # Median along each span, not mean: the question is whether the line RUNS
    # across the strip, and a mean lets a few high columns (the subject against
    # gameplay, a HUD element) stand in for a line that is not there.
    within = beyond = 0.0
    for line in (first, second):
        contrast = _line_contrast(stack, line, axis)
        within += float(np.median(contrast[inside[0]:inside[1]]))
        beyond += float(np.median(contrast[outside[0]:outside[1]]))
    if within <= 0.0:
        return False  # no line to follow: nothing says where the plate ends
    return beyond >= PLATE_FLUSH_CONTINUE_FRAC * within


def _extend_flush_sides(
    stack: np.ndarray,
    lines: tuple,
    reaches: dict,
    found: tuple,
    pinned: dict,
) -> tuple:
    """Push a side out to the frame boundary when the plate is clipped by it.

    ``lines`` and the result are (left, top, right, bottom) in window pixels.
    ``reaches`` says, per side, whether the search window is flush with the
    frame boundary there -- a side cannot be extended past what was sampled.
    ``found`` is the (columns, rows) the line search actually discovered: a
    real line between the side and the boundary means the strip belongs to
    something else (an abutting chat panel shares the cam's top and bottom
    borders), so the plate ends at the side after all.
    """
    left, top, right, bottom = (int(v) for v in lines)
    columns, rows = found
    height, width = stack.shape[1], stack.shape[2]

    def _clear(candidates, lo: int, hi: int) -> bool:
        return not any(lo < int(i) < hi for i, _p in candidates)

    if (reaches["left"] and not pinned["left"] and left > PLATE_EDGE_PIN_PX
            and left <= (right - left) * PLATE_FLUSH_MAX_FRAC
            and _clear(columns, 0, left)
            and _border_continues(
                stack, top, bottom, 0, (left, right), (0, left))):
        left = 0
    if (reaches["right"] and not pinned["right"]
            and width - right > PLATE_EDGE_PIN_PX
            and width - right <= (right - left) * PLATE_FLUSH_MAX_FRAC
            and _clear(columns, right, width)
            and _border_continues(
                stack, top, bottom, 0, (left, right), (right, width))):
        right = width
    if (reaches["top"] and not pinned["top"] and top > PLATE_EDGE_PIN_PX
            and top <= (bottom - top) * PLATE_FLUSH_MAX_FRAC
            and _clear(rows, 0, top)
            and _border_continues(
                stack, left, right, 1, (top, bottom), (0, top))):
        top = 0
    if (reaches["bottom"] and not pinned["bottom"]
            and height - bottom > PLATE_EDGE_PIN_PX
            and height - bottom <= (bottom - top) * PLATE_FLUSH_MAX_FRAC
            and _clear(rows, bottom, height)
            and _border_continues(
                stack, left, right, 1, (top, bottom), (bottom, height))):
        bottom = height
    return left, top, right, bottom


def _aspect_prior(w_px: float, h_px: float) -> float:
    """Flat across 4:3..16:9, decaying only outside normal camera shapes."""
    if w_px <= 0 or h_px <= 0:
        return 0.0
    aspect = w_px / h_px
    if PLATE_ASPECT_MIN <= aspect <= PLATE_ASPECT_MAX:
        return 1.0
    nearest = PLATE_ASPECT_MIN if aspect < PLATE_ASPECT_MIN else PLATE_ASPECT_MAX
    decay = float(np.exp(-abs(np.log(aspect / nearest)) / PLATE_ASPECT_TOL))
    return max(PLATE_ASPECT_PRIOR_FLOOR, decay)


def _locate_plate_by_edges(cv2, frames, box, width, height, seed=None):
    """Search a wide neighborhood for the rectangle that best explains the plate.

    Returns a normalized box or None to decline. ``seed`` (normalized) is the
    moving region, which the result must contain.
    """
    x, y, w, h = (float(v) for v in box[:4])
    mx, my = w * RELOCATE_MARGIN, h * RELOCATE_MARGIN
    x0 = max(0, int((x - mx) * width))
    y0 = max(0, int((y - my) * height))
    x1 = min(width, int((x + w + mx) * width))
    y1 = min(height, int((y + h + my) * height))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None

    # Native resolution: at the old 320px analysis width a 1920-wide frame maps
    # ~6px per sample, which merges the 13/24 border pair into one line.
    stack = np.stack([
        cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        for f in frames
    ])
    # Scope each profile to the claimed plate's own span; see
    # _persistent_edge_profiles. Clamped into the window and widened to a
    # minimum so a degenerate claim cannot collapse the average to nothing.
    def _span(lo: int, hi: int, limit: int) -> tuple:
        lo = max(0, min(limit - 2, int(lo)))
        hi = max(lo + 2, min(limit, int(hi)))
        return (lo, hi)

    col_profile, row_profile = _persistent_edge_profiles(
        stack,
        row_span=_span(int(y * height) - y0, int((y + h) * height) - y0, y1 - y0),
        col_span=_span(int(x * width) - x0, int((x + w) * width) - x0, x1 - x0),
    )

    xs = _prominent_lines(col_profile, PLATE_MAX_LINES, PLATE_LINE_MIN_SEP)
    ys = _prominent_lines(row_profile, PLATE_MAX_LINES, PLATE_LINE_MIN_SEP)
    # Keep what the search actually FOUND: the claimed sides added below are a
    # fallback, not evidence, and _extend_flush_sides has to tell them apart.
    xs_found, ys_found = list(xs), list(ys)
    # Include the claimed sides so leaving the plate unchanged remains an option.

    def _prominence_at(profile, index):
        index = max(0, min(len(profile) - 1, int(index)))
        return float(profile[index]) / (float(np.median(profile)) + 1e-6)

    claimed = (
        int(round(x * width)) - x0, int(round(y * height)) - y0,
        int(round((x + w) * width)) - x0, int(round((y + h) * height)) - y0,
    )

    def _with_claimed(lines, value, profile):
        value = max(0, min(len(profile) - 1, value))
        if all(abs(value - p) >= PLATE_LINE_MIN_SEP for p, _ in lines):
            return lines + [(value, _prominence_at(profile, value))]
        return lines
    xs = _with_claimed(_with_claimed(xs, claimed[0], col_profile), claimed[2], col_profile)
    ys = _with_claimed(_with_claimed(ys, claimed[1], row_profile), claimed[3], row_profile)
    if len(xs) < 2 or len(ys) < 2:
        return None

    # Pin frame-flush sides: their boundary is clipped by the screen and has no
    # external border line to find. Visible gradient peaks lie inside the camera.

    def _pin(lines, value, profile):
        return [(value, _prominence_at(profile, value))]

    xs_left = _pin(xs, claimed[0], col_profile) if x * width <= PLATE_EDGE_PIN_PX else xs
    xs_right = (
        _pin(xs, claimed[2], col_profile)
        if (x + w) * width >= width - PLATE_EDGE_PIN_PX else xs
    )
    ys_top = _pin(ys, claimed[1], row_profile) if y * height <= PLATE_EDGE_PIN_PX else ys
    ys_bottom = (
        _pin(ys, claimed[3], row_profile)
        if (y + h) * height >= height - PLATE_EDGE_PIN_PX else ys
    )
    pinned = {
        "left": xs_left is not xs, "right": xs_right is not xs,
        "top": ys_top is not ys, "bottom": ys_bottom is not ys,
    }
    # A side can only be pushed out to a boundary the window actually sampled.
    reaches = {
        "left": x0 == 0, "right": x1 == width,
        "top": y0 == 0, "bottom": y1 == height,
    }

    # Drop sides with no line under them before scoring; see the note beside
    # PLATE_FLUSH_CONTINUE_FRAC for what admitting one costs.

    def _supported(lines, is_pinned: bool):
        if is_pinned:
            return lines
        return [(i, p) for i, p in lines if p >= PLATE_SIDE_MIN_PROMINENCE]

    xs_left = _supported(xs_left, pinned["left"])
    xs_right = _supported(xs_right, pinned["right"])
    ys_top = _supported(ys_top, pinned["top"])
    ys_bottom = _supported(ys_bottom, pinned["bottom"])

    # Compute the baseline from the claimed sides directly. Reading only candidate
    # lines can produce a zero baseline when nearby-line suppression removes a side
    # and would silently disable the required gain over existing geometry.

    baseline_score = (
        (
            _prominence_at(col_profile, claimed[0])
            + _prominence_at(col_profile, claimed[2])
            + _prominence_at(row_profile, claimed[1])
            + _prominence_at(row_profile, claimed[3])
        ) / 4.0
    ) * _aspect_prior(
        max(1.0, claimed[2] - claimed[0]), max(1.0, claimed[3] - claimed[1]))

    seed_px = None
    if seed is not None:
        seed_px = (
            seed[0] * width - x0, seed[1] * height - y0,
            (seed[0] + seed[2]) * width - x0, (seed[1] + seed[3]) * height - y0,
        )

    best, best_score, best_lines = None, 0.0, None
    for li, lp in xs_left:
        for ri, rp in xs_right:
            if ri - li < 24:
                continue
            for ti, tp in ys_top:
                for bi, bp in ys_bottom:
                    if bi - ti < 24:
                        continue
                    # The subject must sit inside the plate; a rectangle that
                    # crops into the person is the exact regression this guards.
                    if seed_px is not None and not (
                        li <= seed_px[0] + 2 and ti <= seed_px[1] + 2
                        and ri >= seed_px[2] - 2 and bi >= seed_px[3] - 2
                    ):
                        continue
                    cand = [
                        (x0 + li) / width, (y0 + ti) / height,
                        (ri - li) / width, (bi - ti) / height,
                    ]
                    if not _is_plausible_facecam(cand):
                        continue
                    area_ratio = (cand[2] * cand[3]) / max(1e-6, w * h)
                    if not (RELOCATE_MIN_KEEP <= area_ratio <= RELOCATE_MAX_GROW):
                        continue
                    if _iou(cand, [x, y, w, h]) < RELOCATE_MIN_IOU:
                        continue
                    support = (lp + rp + tp + bp) / 4.0
                    score = support * _aspect_prior(ri - li, bi - ti)
                    if score > best_score:
                        best, best_score = cand, score
                        best_lines = (li, ti, ri, bi)

    if best is None:
        return None
    if best_score < PLATE_MIN_GAIN * baseline_score:
        return None  # the claimed box already explains the plate well enough

    def _finish(lines):
        """Normalized box for ``lines``, flush sides restored, or None."""
        li, ti, ri, bi = _extend_flush_sides(
            stack, lines, reaches, (xs_found, ys_found), pinned)
        box = [
            (x0 + li) / width, (y0 + ti) / height,
            (ri - li) / width, (bi - ti) / height,
        ]
        return box if _is_plausible_facecam(box) else None

    if seed_px is None:
        return _finish(best_lines) or best

    # Strip the window's border ring: walk each side inward to the innermost
    # detected line that still contains the live feed.
    li, ti, ri, bi = best_lines
    # Measure feed fill against the plate before tightening. Tightening inflates
    # the ratio and can wrongly trigger a clamp onto the subject.

    seed_area = max(0.0, seed_px[2] - seed_px[0]) * max(0.0, seed_px[3] - seed_px[1])
    seed_spans_plate = seed_area >= PLATE_SEED_FILL * max(1.0, (ri - li) * (bi - ti))

    # Ring step, seed-independent: a window border is a thin ring, so a second
    # strong line just inside a chosen side is that ring's inner edge. Bounded by
    # PLATE_RING_FRAC, because stepping to any line further in lands on unrelated
    # structure -- walking the sides toward the feed instead put the right edge on
    # chat.exe's border at x376, back inside the camera.
    tol = PLATE_SEED_TOL
    ring_x = max(4, int(round((ri - li) * PLATE_RING_FRAC)))
    ring_y = max(4, int(round((bi - ti) * PLATE_RING_FRAC)))
    # A pinned side has no ring: it is clipped by the screen edge, so stepping
    # "inward off the window border" would just crop into the camera.
    if not pinned["left"]:
        li = min([i for i, _p in xs if li < i <= li + ring_x
                  and i <= seed_px[0] + tol], default=li)
    if not pinned["right"]:
        ri = max([i for i, _p in xs if ri - ring_x <= i < ri
                  and i >= seed_px[2] - tol], default=ri)
    if not pinned["top"]:
        ti = min([i for i, _p in ys if ti < i <= ti + ring_y
                  and i <= seed_px[1] + tol], default=ti)
    if not pinned["bottom"]:
        bi = max([i for i, _p in ys if bi - ring_y <= i < bi
                  and i >= seed_px[3] - tol], default=bi)

    if seed_spans_plate:
        # ...per side, so a screen-clipped edge keeps its pinned coordinate: the
        # moving region stops at the last pixel that moves, which is inside the
        # frame boundary, never on it.
        if not pinned["left"]:
            li = int(round(seed_px[0]))
        if not pinned["top"]:
            ti = int(round(seed_px[1]))
        if not pinned["right"]:
            ri = int(round(seed_px[2]))
        if not pinned["bottom"]:
            bi = int(round(seed_px[3]))

    tightened = [
        (x0 + li) / width, (y0 + ti) / height,
        (ri - li) / width, (bi - ti) / height,
    ]
    area_ratio = (tightened[2] * tightened[3]) / max(1e-6, w * h)
    if (
        ri - li < 24 or bi - ti < 24
        or not _is_plausible_facecam(tightened)
        or not (RELOCATE_MIN_KEEP <= area_ratio <= RELOCATE_MAX_GROW)
        or _iou(tightened, [x, y, w, h]) < RELOCATE_MIN_IOU
    ):
        return _finish(best_lines) or best
    # Flush sides come last, after the gates above have judged the tightened
    # geometry: growing back to a screen boundary is evidence-gated in its own
    # right, and measuring it against the claim would re-reject it.
    return _finish((li, ti, ri, bi)) or tightened


def _layout_sample_times(
    layout: FacecamLayout, duration: float, samples: int,
) -> List[float]:
    """Frame times to inspect this layout at: its own scenes, spread evenly."""
    times = sorted({
        max(0.0, min(duration, float(value)))
        for value in layout.timestamps
    })
    if len(times) < 5:
        # Backward compatibility for layouts reconstructed from older persisted
        # framing and for callers/tests that provide geometry only. New scans
        # always carry cluster-member timestamps.
        return list(np.linspace(0.08 * duration, 0.92 * duration, max(5, samples)))
    count = min(len(times), max(5, samples))
    indexes = np.linspace(0, len(times) - 1, count).round().astype(int)
    return [times[int(index)] for index in indexes]


# Measure the subject top to guide cover-crops inside camera plates. A fixed
# vertical bias cannot accommodate both high and low subject placement. Use the
# person detector rather than a face-only detector to preserve the head outline.
# Collect enough samples for intermittent detections; missing evidence retains
# the existing renderer fallback.

SUBJECT_TOP_SAMPLES = 24
SUBJECT_TOP_MIN_HITS = 4      # below this the median is noise, not a measurement
SUBJECT_TOP_MAX = 0.60        # a "subject" starting below this is not the plate's


def measure_subject_tops(
    video_path: str,
    layouts: List[FacecamLayout],
    duration: float,
    samples: int = SUBJECT_TOP_SAMPLES,
) -> List[FacecamLayout]:
    """Attach ``subject_top`` to each portrait layout. Best-effort.

    Run AFTER :func:`refine_layout_boxes`: the measurement is a fraction of the
    plate's height, so it is only meaningful against the plate's final box.
    Any failure (unreadable video, too few person detections) leaves
    ``subject_top`` at None and the renderer on its existing constant.
    """
    if not layouts or duration <= 0:
        return layouts
    try:
        import cv2
        from engines.vision.detector import person_box_in_region
    except Exception:  # noqa: BLE001 - optional at import time in some test envs
        return layouts

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return layouts
    try:
        frame_w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        frame_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        frame_aspect = (frame_w / frame_h) if frame_w > 0 and frame_h > 0 else 16.0 / 9.0
        measured: List[FacecamLayout] = []
        for layout in layouts:
            box = [float(value) for value in (layout.box or [])[:4]]
            # Measure landscape and portrait plates because both render paths use subject
            # placement. Keep the upper aspect bound: an extremely wide region is not a
            # camera plate with meaningful subject-top geometry.

            if len(box) < 4 or _facecam_pixel_aspect(box, frame_aspect) > CAM_ASPECT_MAX:
                measured.append(layout)
                continue
            tops = []
            for timestamp in _layout_sample_times(layout, duration, samples):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                person = person_box_in_region(frame, box)
                if person is None:
                    continue
                # Frame-normalized -> fraction of the plate's own height. A
                # partial detection (an arm, the subject half out of frame)
                # starts far down the plate; those are the samples the median
                # is here to outvote, and the ceiling drops the worst of them.
                top = (float(person[1]) - box[1]) / max(1e-6, box[3])
                if 0.0 <= top <= SUBJECT_TOP_MAX:
                    tops.append(top)
            if len(tops) >= SUBJECT_TOP_MIN_HITS:
                measured.append(dataclass_replace(
                    layout, subject_top=round(float(median(tops)), 4)))
            else:
                measured.append(layout)
        return measured
    except Exception:  # noqa: BLE001 - framing measurement must never stop a scan
        return layouts
    finally:
        cap.release()


def subject_top_for_box(
    layouts: List[FacecamLayout], box: Optional[List[float]],
) -> Optional[float]:
    """``subject_top`` of the layout ``box`` came from, if it was measured.

    Clip framing carries boxes, not layouts, so the measurement has to be
    matched back by geometry -- on WINDOW_MATCH_IOU, the same threshold
    facecam_for_window already uses to decide a window box belongs to a layout.
    """
    if not box or len(box) < 4:
        return None
    best, best_iou = None, WINDOW_MATCH_IOU
    for layout in layouts or []:
        if layout.subject_top is None or len(layout.box or []) < 4:
            continue
        overlap = _iou(list(box[:4]), list(layout.box[:4]))
        if overlap >= best_iou:
            best, best_iou = layout.subject_top, overlap
    return best


def refine_layout_boxes(
    video_path: str,
    layouts: List[FacecamLayout],
    duration: float,
    samples: int = REFINE_SAMPLES,
) -> List[FacecamLayout]:
    """Snap each layout's box to the actual live camera feed inside it.

    Best-effort: any failure (unreadable video, too-static region, implausible
    trim) keeps the original box for that layout. Never grows a box beyond the
    original plus the search margin.
    """
    if not layouts or duration <= 0:
        return layouts
    try:
        import cv2
    except Exception:  # noqa: BLE001 - optional at import time in some test envs
        return layouts

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return layouts
    try:
        def _read_frames(times: List[float]):
            frames = []
            for timestamp in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
            return frames

        refined: List[FacecamLayout] = []
        for layout in layouts:
            frames = _read_frames(
                _layout_sample_times(layout, duration, samples))
            if len(frames) < 5:
                refined.append(layout)
                continue
            height, width = frames[0].shape[:2]
            original = [float(v) for v in layout.box[:4]]
            # Relocate grossly misplaced boxes before edge snapping. Motion supplies a
            # lower bound on the live feed, not the whole plate; still room areas must survive.

            seed = _relocate_to_live_feed(cv2, frames, original, width, height)
            moved = _locate_plate_by_edges(
                cv2, frames, original, width, height, seed=seed)
            if moved is not None:
                # Do not resnap geometry already placed by this stage at lower analysis
                # resolution. Adjacent decorative edges can merge after downscaling.

                moved = _preserve_decorated_panel_aspect(
                    layout, moved, width / max(1.0, float(height)))
                refined.append(dataclass_replace(
                    layout, box=[round(v, 4) for v in moved]))
                continue

            # Stage 2: snap each side to its persistent border line.
            x, y, w, h = (float(v) for v in layout.box[:4])
            mx, my = w * REFINE_MARGIN, h * REFINE_MARGIN
            x0 = max(0, int((x - mx) * width))
            y0 = max(0, int((y - my) * height))
            x1 = min(width, int((x + w + mx) * width))
            y1 = min(height, int((y + h + my) * height))
            if x1 - x0 < 16 or y1 - y0 < 16:
                refined.append(layout)
                continue
            scale = _REFINE_ANALYSIS_W / float(x1 - x0)
            size = (_REFINE_ANALYSIS_W, max(8, int(round((y1 - y0) * scale))))
            stack = np.stack([
                cv2.resize(
                    cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), size,
                    interpolation=cv2.INTER_AREA,
                ).astype(np.float32)
                for f in frames
            ])
            col_profile, row_profile = _persistent_edge_profiles(stack)

            # Each original side, in analysis coordinates, searches its own
            # Ã‚Â± margin neighborhood for a persistent border line.
            def to_col(nx: float) -> int:
                return int(round((nx * width - x0) * scale))

            def to_row(ny: float) -> int:
                return int(round((ny * height - y0) * scale))

            m_c = max(2, int(round(mx * width * scale)))
            m_r = max(2, int(round(my * height * scale)))
            left = _snap_side(col_profile, to_col(x) - m_c, to_col(x) + m_c, to_col(x))
            right = _snap_side(col_profile, to_col(x + w) - m_c, to_col(x + w) + m_c, to_col(x + w))
            top = _snap_side(row_profile, to_row(y) - m_r, to_row(y) + m_r, to_row(y))
            bottom = _snap_side(row_profile, to_row(y + h) - m_r, to_row(y + h) + m_r, to_row(y + h))
            if right - left < 8 or bottom - top < 8:
                refined.append(layout)
                continue
            new_box = [
                (x0 + left / scale) / width,
                (y0 + top / scale) / height,
                ((right - left) / scale) / width,
                ((bottom - top) / scale) / height,
            ]
            area_ratio = (new_box[2] * new_box[3]) / max(1e-6, w * h)
            if (
                area_ratio < REFINE_MIN_KEEP
                or area_ratio > REFINE_MAX_GROW
                or not _is_plausible_facecam(new_box)
            ):
                refined.append(layout)
                continue
            refined.append(dataclass_replace(
                layout, box=[round(v, 4) for v in new_box],
            ))
        return refined
    except Exception:  # noqa: BLE001 - refinement must never block a scan
        return layouts
    finally:
        cap.release()


def _dominant_fallback(layouts: List[FacecamLayout]) -> Optional[List[float]]:
    """The dominant cam box, but only when it's on for enough of the VOD to be
    confidently present even where this clip's detections are missing/wandering."""
    if not layouts:
        return None
    dominant = layouts[0]
    reliable = (
        dominant.fraction >= DOMINANT_FALLBACK_FRACTION
        or dominant.coverage >= DOMINANT_FALLBACK_COVERAGE
    )
    return dominant.box if reliable else None


def _window_local_panel_candidates(
    layouts: List[FacecamLayout],
    unified_signals,
    start: float,
    end: float,
    target: float,
) -> List[List[float]]:
    """Recover a short-lived edge camera using local direct-panel evidence.

    The detector often sees only the person inside a borderless panel, so its
    raw box is not safe crop geometry. Reuse the established plate dimensions,
    place them on the raw cluster's horizontal edge, and bottom-align them to
    the local detections. The caller must still verify the proposed plate in the
    source frame before it can be returned.
    """
    dominant = _dominant_fallback(layouts)
    if dominant is None:
        return []
    dx, _dy, dw, dh = (float(value) for value in dominant[:4])
    dominant_left = dx <= TENURE_EDGE_TOLERANCE
    dominant_right = 1.0 - (dx + dw) <= TENURE_EDGE_TOLERANCE
    if not (dominant_left or dominant_right):
        return []

    window = [
        signal for signal in unified_signals
        if start <= float(getattr(signal, "timestamp", 0.0) or 0.0) <= end
        and getattr(signal, "vision", None) is not None
        and getattr(signal.vision, "facecam_box", None)
    ]
    panels = [
        (
            float(getattr(signal, "timestamp", 0.0) or 0.0),
            [float(value) for value in signal.vision.facecam_box[:4]],
        )
        for signal in window
        if getattr(signal.vision, "facecam_source", None) == "panel"
    ]
    if (
        len(panels) < WINDOW_LOCAL_PANEL_MIN_FRAMES
        or len(panels) / max(1, len(window)) < WINDOW_LOCAL_PANEL_MIN_SHARE
    ):
        return []

    # A short window can still contain two different panel-like regions (for
    # example browser chrome followed by the actual camera). Cluster them and
    # let source verification decide; never average them into a third crop.
    clusters: List[List[tuple[float, List[float]]]] = []
    for timestamp, box in panels:
        best = None
        best_score = CLUSTER_IOU
        for index, cluster in enumerate(clusters):
            representative = [
                float(median(values))
                for values in zip(*(item[1] for item in cluster))
            ]
            score = max(_iou(box, representative), _containment(box, representative))
            if score >= best_score:
                best, best_score = index, score
        if best is None:
            clusters.append([(timestamp, box)])
        else:
            clusters[best].append((timestamp, box))

    proposals = []
    for cluster in clusters:
        if len(cluster) < WINDOW_LOCAL_PANEL_MIN_FRAMES:
            continue
        boxes = [item[1] for item in cluster]
        raw = [float(median(values)) for values in zip(*boxes)]
        raw_edge = min(raw[0], 1.0 - (raw[0] + raw[2]))
        if raw_edge > WINDOW_LOCAL_PANEL_EDGE_TOLERANCE:
            continue
        stability = float(np.mean([_iou(box, raw) for box in boxes]))
        if stability < WINDOW_LOCAL_PANEL_STABILITY_MIN:
            continue
        raw_center = raw[0] + raw[2] / 2.0
        x = 0.0 if raw_center <= 0.5 else 1.0 - dw
        bottom = float(median([box[1] + box[3] for box in boxes]))
        y = min(max(0.0, bottom - dh), 1.0 - dh)
        candidate = [round(x, 4), round(y, 4), round(dw, 4), round(dh, 4)]
        if _is_plausible_facecam(candidate):
            distance = min(abs(item[0] - target) for item in cluster)
            proposals.append((distance, -len(cluster), candidate))
    proposals.sort(key=lambda item: (item[0], item[1]))
    unique = []
    for _distance, _support, candidate in proposals:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def facecam_for_window(
    layouts: List[FacecamLayout],
    unified_signals,
    start: float,
    end: float,
    observations: Optional[List[FacecamObservation]] = None,
    at: Optional[float] = None,
    layout_verifier: Optional[Callable[[List[float], float], bool]] = None,
) -> Optional[List[float]]:
    """Pick the facecam box for one clip window from the VOD layout model.

    New framing models pass an explicit scene timeline: unmatched/fullscreen
    samples are None and can never borrow a crop from a different scene. The raw
    signal-voting path remains for persisted v1 jobs and older callers.
    """
    if not layouts:
        return None

    if observations is not None:
        window = [
            item
            for item in observations
            if start <= item.timestamp <= end
        ]
        if not window:
            return None
        target = float(at) if at is not None and start <= float(at) <= end else (start + end) / 2.0
        nearest = min(window, key=lambda item: abs(item.timestamp - target))
        layout_index = nearest.layout_index
        if layout_index is not None and 0 <= layout_index < len(layouts):
            # Verify the dominant plate before accepting a rival alternate. Stable game
            # portraits and HUD panels can look like cameras. A true camera move vacates
            # the old position, whereas a rival coexists with it. Local tenancy narrows
            # the edge exemption; non-edge alternates still require verification.

            dominant = _dominant_fallback(layouts)
            if (
                layout_index != 0
                and dominant is not None
                and layout_verifier is not None
                and not (
                    _edge_anchored(layouts[layout_index].box)
                    and _dominant_vacated_locally(
                        observations, layout_index, target,
                    )
                )
            ):
                try:
                    if layout_verifier(dominant, target):
                        return dominant
                except Exception:  # noqa: BLE001 - keep exact scene evidence
                    pass

            # The loose person fallback occasionally produces one enormous box
            # at the editorial peak and overlaps the wrong layout. If the local
            # timeline overwhelmingly supports another known layout, verify
            # that surrounding winner in the source frame before overriding the
            # exact sample. A genuine scene transition fails that verification
            # because the old camera is no longer present.
            fallback_peak = getattr(nearest, "source", None) == "person_fallback"
            if getattr(nearest, "source", None) is None:
                # Compatibility for v1/v2 framing saved before ``source``.
                raw_window = [
                    signal
                    for signal in unified_signals
                    if start <= signal.timestamp <= end
                    and getattr(signal, "vision", None) is not None
                    and getattr(signal.vision, "facecam_box", None)
                ]
                if raw_window:
                    raw_nearest = min(
                        raw_window, key=lambda signal: abs(signal.timestamp - target))
                    raw_box = raw_nearest.vision.facecam_box
                    layout_box = layouts[layout_index].box
                    raw_area = float(raw_box[2]) * float(raw_box[3])
                    layout_area = float(layout_box[2]) * float(layout_box[3])
                    fallback_peak = raw_area >= layout_area * FALLBACK_OVERSIZE_RATIO
            votes = [0] * len(layouts)
            for item in window:
                if item.layout_index is not None and 0 <= item.layout_index < len(layouts):
                    votes[item.layout_index] += 1
            majority_index = max(range(len(layouts)), key=lambda i: votes[i])
            vote_ratio = (
                FALLBACK_WINDOW_VOTE_RATIO if fallback_peak
                else PANEL_WINDOW_VOTE_RATIO
            )
            if (
                majority_index != layout_index
                and votes[majority_index] >= max(
                    3, votes[layout_index] * vote_ratio)
                and layout_verifier is not None
            ):
                candidate = layouts[majority_index].box
                try:
                    if layout_verifier(candidate, target):
                        return candidate
                except Exception:  # noqa: BLE001 - keep exact scene evidence
                    pass
            return layouts[layout_index].box

        # ``absent`` is positive no-camera evidence. ``unknown`` is a timeline
        # persisted before states existed, so preserve its historical fail-safe
        # behavior. Only explicit detector uncertainty may recover a known layout.
        if getattr(nearest, "state", "unknown") != "unresolved":
            return None

        votes = [0] * len(layouts)
        for item in window:
            if item.layout_index is not None and 0 <= item.layout_index < len(layouts):
                votes[item.layout_index] += 1
        known_share = sum(votes) / max(1, len(window))
        if (
            layout_verifier is not None
            and known_share <= WINDOW_LOCAL_MAX_KNOWN_SHARE
        ):
            local_candidates = _window_local_panel_candidates(
                layouts, unified_signals, start, end, target,
            )
            # Local person verification cannot distinguish a moved camera from a torso or
            # game portrait. If the established dominant plate remains occupied, prefer it.

            if local_candidates:
                dominant = _dominant_fallback(layouts)
                if dominant is not None:
                    try:
                        if layout_verifier(dominant, target):
                            return dominant
                    except Exception:  # noqa: BLE001 - keep trying local recovery
                        pass
            for local in local_candidates:
                try:
                    if layout_verifier(local, target):
                        return local
                except Exception:  # noqa: BLE001 - keep trying safe fallbacks
                    continue
        if sum(votes):
            winner = max(range(len(layouts)), key=lambda i: votes[i])
            # Apply the same rival check to unresolved peaks. Window sample counts can
            # favor HUD detections; only a vacated dominant plate supports a camera move.

            dominant = _dominant_fallback(layouts)
            if (
                winner != 0
                and dominant is not None
                and layout_verifier is not None
                and not (
                    _edge_anchored(layouts[winner].box)
                    and _dominant_vacated_locally(observations, winner, target)
                )
            ):
                try:
                    if layout_verifier(dominant, target):
                        return dominant
                except Exception:  # noqa: BLE001 - keep exact scene evidence
                    pass
            candidate = layouts[winner].box
        else:
            candidate = _dominant_fallback(layouts)
        if candidate is None or layout_verifier is None:
            return None
        try:
            return candidate if layout_verifier(candidate, target) else None
        except Exception:  # noqa: BLE001 - verification must fail closed
            return None

    window = [
        s.vision.facecam_box
        for s in unified_signals
        if start <= s.timestamp <= end
        and getattr(s, "vision", None) is not None
        and s.vision.facecam_box
    ]
    if not window:
        # No small cam was detected anywhere in this clip. That is meaningful
        # for scene changes (full-camera intros, BRB screens, gameplay with the
        # cam intentionally hidden): do not borrow a PiP from another scene.
        return None

    votes = [0] * len(layouts)
    for b in window:
        best_i, best_iou = None, WINDOW_MATCH_IOU
        for i, layout in enumerate(layouts):
            v = _iou(b, layout.box)
            if v >= best_iou:
                best_iou, best_i = v, i
        if best_i is not None:
            votes[best_i] += 1

    if sum(votes) == 0:
        # Every box in the window is off every real layout -> the per-frame
        # tracker wandered (a character/HUD). Never crop that; use the real cam.
        return _dominant_fallback(layouts)

    best_i = max(range(len(layouts)), key=lambda i: votes[i])
    return layouts[best_i].box


def is_facecam_layout_at(
    video_path: str,
    timestamp: Optional[float],
    box: Optional[List[float]],
) -> bool:
    """Confirm that a person is still present inside an established cam box.

    Used only when the ordinary tracker wandered at a clip's editorial peak.
    Two of three bracketed frames must agree, so one game-frame coincidence
    cannot resurrect a camera that was genuinely switched off.
    """
    if not video_path or timestamp is None or not box:
        return False
    try:
        import cv2
        from engines.vision.detector import detect_person_in_region

        capture = cv2.VideoCapture(video_path)
        hits = 0
        try:
            center = max(0.0, float(timestamp))
            for sample_time in (max(0.0, center - 1.0), center, center + 1.0):
                capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
                ok, frame = capture.read()
                if ok and detect_person_in_region(frame, box):
                    hits += 1
        finally:
            capture.release()
        return hits >= 2
    except Exception:  # noqa: BLE001 - composition probing must never stop a scan
        return False


def _portrait_cover_focus(subject_center_x: float, source_aspect: float) -> float:
    """Map a subject center to the renderer's horizontal cover-crop focus.

    ``focus=0`` anchors the portrait crop left and ``focus=1`` anchors it
    right. Keeping this conversion beside the camera detector lets the render
    module remain deterministic and avoids treating normalized subject X as
    though it were already a crop offset.
    """
    target_aspect = 9.0 / 16.0
    visible_width = min(1.0, target_aspect / max(1e-6, float(source_aspect)))
    if visible_width >= 1.0:
        return 0.5
    left = max(0.0, min(1.0 - visible_width, float(subject_center_x) - visible_width / 2.0))
    return left / (1.0 - visible_width)


def fullframe_camera_focus_at(
    video_path: str,
    timestamp: Optional[float],
    camera_region: Optional[List[float]] = None,
) -> Optional[float]:
    """Return a stable portrait-crop focus for a fullscreen webcam scene.

    The candidate peak is the right frame to inspect: it is what selection and
    the thumbnail present as the moment.  The gate in detector.py is purposely
    strict, so a miss keeps historical full-gameplay behavior. Confirmed
    detections are converted into the renderer's cover-crop coordinate and the
    median is persisted, producing a steady creator-first crop instead of
    per-frame tracking jitter.
    """
    if not video_path or timestamp is None:
        return None
    try:
        import cv2
        from engines.vision.detector import detect_fullframe_camera_person
        from engines.emotion.face_emotion import _detect_face

        capture = cv2.VideoCapture(video_path)
        focuses = []
        try:
            center = max(0.0, float(timestamp))
            # A creator can be bent partly outside the frame at the exact
            # reaction peak.  Inspect a tiny bracket rather than turning one
            # transient detector miss into a gameplay crop for the whole clip.
            offsets = (-4.0, -2.0, 0.0, 2.0, 4.0) if camera_region else (-2.0, 0.0, 2.0)
            for offset in offsets:
                sample_time = max(0.0, center + offset)
                capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    continue
                if camera_region:
                    height, width = frame.shape[:2]
                    x, y, w, h = (float(value) for value in camera_region[:4])
                    x1 = max(0, min(width - 1, int(round(x * width))))
                    y1 = max(0, min(height - 1, int(round(y * height))))
                    x2 = max(x1 + 1, min(width, int(round((x + w) * width))))
                    y2 = max(y1 + 1, min(height, int(round((y + h) * height))))
                    frame = frame[y1:y2, x1:x2]
                subject = detect_fullframe_camera_person(frame)
                if subject is not None:
                    frame_h, frame_w = frame.shape[:2]
                    # A large-person box often includes the chair and blankets;
                    # on asymmetric setups that can sit far from the actual
                    # face. Prefer the lightweight face detector for framing,
                    # while keeping the stricter person hit as the scene gate.
                    _, face_bbox = _detect_face(frame)
                    if face_bbox is not None:
                        face_x, _, face_w, _ = (float(value) for value in face_bbox[:4])
                        subject_center = (face_x + face_w / 2.0) / max(1, frame_w)
                    else:
                        subject_center = float(subject[0]) + float(subject[2]) / 2.0
                    focuses.append(_portrait_cover_focus(subject_center, frame_w / max(1, frame_h)))
        finally:
            capture.release()
        # Overriding an established gameplay viewport needs two independent
        # confirmations across a wider bracket; the exact reaction pose may be
        # the one frame where a leaning creator's person box fragments.
        required = 2 if camera_region else 1
        return float(median(focuses)) if len(focuses) >= required else None
    except Exception:  # noqa: BLE001 - composition probing must never stop a scan
        return None


def is_fullframe_camera_at(
    video_path: str,
    timestamp: Optional[float],
    camera_region: Optional[List[float]] = None,
) -> bool:
    """Compatibility predicate for callers that only need scene identity."""
    return fullframe_camera_focus_at(video_path, timestamp, camera_region) is not None
