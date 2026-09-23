# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


# A real facecam panel occupies a modest corner of the frame. A detected box
# that covers most of the frame is a misdetection (the in-game character, a
# full-screen face, a UI panel). Rendering it as the vertical-split top band
# would crop a large slice of GAMEPLAY and stack it over the gameplay, so the
# clip looks like the video overlaid on itself. Reject it and fall back to a
# clean full-frame crop instead.
FACECAM_MAX_W = 0.55
FACECAM_MAX_H = 0.60
FACECAM_MAX_AREA = 0.33

# Height alone was never the invariant that check protects -- WIDTH and AREA
# are. A landscape plate is never tall, so a single 0.60 ceiling was
# indistinguishable from the real rule until portrait cameras appeared: a
# vertical webcam and a PNGtuber avatar plate are legitimately taller than they
# are wide. Measured across the 40-VOD library, the tallest landscape plate
# reaches h=0.42, while the Skyrim PNGtuber's avatar plate is h=0.63 at w=0.32
# (area 0.205) -- comfortably inside the area/width bounds, and rejected on
# height alone.
#
# So a box may exceed FACECAM_MAX_H only while it stays NARROW, which is
# precisely the shape that cannot be the failure this gate exists to stop: a
# tall AND wide box still covers most of the frame and is still rejected by
# FACECAM_MAX_W / FACECAM_MAX_AREA. The narrow bound sits above the two real
# portrait cams (w=0.27, 0.32) and far below FACECAM_MAX_W.
FACECAM_PORTRAIT_MAX_H = 0.75
FACECAM_PORTRAIT_MAX_W = 0.36
# ...and it must still be a PLATE, not a vertical sliver. This is
# CAM_ASPECT_PORTRAIT_MIN (0.55, in pixels) carried into the normalized units
# this frame-agnostic check works in: 0.55 / (16/9) = 0.31. The three real
# portrait cams in the library sit at 0.48-0.51.
FACECAM_PORTRAIT_MIN_WH = 0.31

# Auto composition is resolved once per VOD so clips do not visually jitter
# between stacked and PiP because one local motion sample crossed a threshold.
# PiP is useful for action-forward games whose important play stays near the
# center; stacked preserves substantially more horizontal context for slower,
# exploratory, puzzle, and narrative games.
AUTO_PIP_MOTION_P75 = 0.18
AUTO_PIP_GAME_P75 = 0.25
AUTO_PIP_EVENT_FRACTION = 0.20
# Camera prominence: a persistent cam layout covering at least this fraction
# of the VOD at at least this plate area marks a reaction-led creator ->
# ``auto`` prefers the cam-forward stacked split. Skipped for action FPS/BR
# titles (Fortnite, Valorant, …) where a normal corner webcam is expected and
# PiP is the correct default — measured regression: mastery-monday Fortnite
# with a 4.6% corner cam at 40% uptime was forced to stacked.
CAM_PROMINENT_FRACTION = 0.30
CAM_PROMINENT_AREA = 0.045
# Huge webcam plates still force stacked even on action games (cam IS the show).
CAM_DOMINANT_AREA = 0.12
ACTION_GAME_LABELS = {"elimination", "knock", "terminal_win", "rank_progress"}
# Keys match configs/game_signatures.json / detect_game().
ACTION_PIP_GAMES = {
    "fortnite", "valorant", "apex", "cod", "overwatch", "cs2",
    "r6", "pubg", "halo", "destiny", "rocket_league",
}


def _percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    blend = index - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def resolve_export_layout(
    requested: str,
    clips: Iterable[Any],
    facecam_layouts: Iterable[Any] = None,
    game: Optional[str] = None,
) -> str:
    """Resolve ``auto`` to one stable composition for the whole VOD.

    ``facecam_layouts`` (optional, engines.vision.facecam layout objects) lets
    ``auto`` see camera prominence: a creator whose cam plate is large and on
    screen for much of the VOD is a reaction-led channel — the cam IS the
    content, and gameplay-PiP (which shrinks the cam to a floating panel and
    shows ~32% of the horizontal game FOV) is the wrong composition no matter
    how busy the game looks.

    ``game`` (optional detect_game id) changes the prominence rule for action
    FPS/BR titles: a normal corner webcam must not veto PiP on Fortnite.
    Only a dominant cam plate still forces stacked there.
    """
    if requested in ("vertical_split", "gameplay_pip"):
        return requested

    game_key = str(game or "generic").strip().lower().replace(" ", "_")
    action_title = game_key in ACTION_PIP_GAMES
    area_floor = CAM_DOMINANT_AREA if action_title else CAM_PROMINENT_AREA

    for layout in facecam_layouts or []:
        box = list(getattr(layout, "box", None) or [])
        fraction = float(getattr(layout, "fraction", 0.0) or 0.0)
        if len(box) >= 4 and fraction >= CAM_PROMINENT_FRACTION:
            if float(box[2]) * float(box[3]) >= area_floor:
                return "vertical_split"

    clips = list(clips or [])
    if not clips:
        return "vertical_split"

    motion_values = []
    game_values = []
    action_events = 0
    for clip in clips:
        breakdown = getattr(clip, "modality_breakdown", None) or {}
        motion_values.append(float(breakdown.get("motion", 0.0) or 0.0))
        game_values.append(max(
            float(breakdown.get("game", 0.0) or 0.0),
            float(getattr(clip, "game_evidence", 0.0) or 0.0),
        ))
        if getattr(clip, "game_label", None) in ACTION_GAME_LABELS:
            action_events += 1

    event_fraction = action_events / max(1, len(clips))
    # A single medium-strength motion signal is not enough to throw away the
    # horizontal context that stacked framing preserves. In practice, slower
    # simulation and narrative games can cross the motion threshold through
    # camera pans, animated UI, or a few busy moments even though PiP is the
    # wrong composition for the VOD. Require two independent action signals so
    # ``auto`` stays conservative and predictable; creators can still force PiP
    # explicitly in Export settings.
    action_votes = sum((
        _percentile(motion_values, 0.75) >= AUTO_PIP_MOTION_P75,
        _percentile(game_values, 0.75) >= AUTO_PIP_GAME_P75,
        event_fraction >= AUTO_PIP_EVENT_FRACTION,
    ))
    # Known action FPS/BR titles default to PiP (gameplay is the content; a
    # corner webcam is expected). Other titles still need two action signals.
    if action_title:
        action_forward = True
    else:
        action_forward = action_votes >= 2
    return "gameplay_pip" if action_forward else "vertical_split"


def _is_plausible_facecam(box: List[float]) -> bool:
    if not box or len(box) < 4:
        return False
    w, h = float(box[2]), float(box[3])
    if w <= 0.0 or h <= 0.0:
        return False
    if w > FACECAM_MAX_W or (w * h) > FACECAM_MAX_AREA:
        return False
    if h <= FACECAM_MAX_H:
        return True
    # Taller than a landscape plate ever is: allowed only while narrow enough
    # that it cannot be the frame-filling misdetection this gate guards, and
    # still wide enough relative to its height to be a plate at all.
    return (
        w <= FACECAM_PORTRAIT_MAX_W
        and h <= FACECAM_PORTRAIT_MAX_H
        and (w / h) >= FACECAM_PORTRAIT_MIN_WH
    )


def assign_layout(
    facecam_box: List[float] = None,
    export_layout: str = "auto",
    gameplay_box: List[float] = None,
    fullframe_camera: bool = False,
    camera_focus_x: Optional[float] = None,
    facecam_subject_top: Optional[float] = None,
) -> Dict[str, Any]:
    """Assign layout metadata for the clip.

    ``gameplay_box`` is the detected game viewport (normalized [x, y, w, h])
    for windowed stream layouts; None means gameplay fills the frame. A facecam
    box that isn't a plausible corner panel is dropped (full-frame gameplay)
    rather than rendered as a self-overlay.

    ``facecam_subject_top`` is where the subject's head starts inside the
    facecam plate (measure_subject_tops). Portrait plates are cropped hard to
    reach the render's landscape window, so this is what keeps the face in it;
    it travels with the clip because the renderer sees clips, not layouts.

    The gameplay layer always cover-fills its target rectangle at render time —
    it is never letterboxed — so there is no per-clip "contain" flag to carry.
    """
    # A recurring authored viewport can switch from gameplay to a camera feed.
    # ``gameplay_box`` then supplies the camera source bounds; it must not veto
    # positive camera-content evidence.
    layout_type = "full_camera" if fullframe_camera else "full_gameplay"
    if facecam_box and _is_plausible_facecam(facecam_box):
        layout_type = "gameplay_pip" if export_layout == "gameplay_pip" else "vertical_split"
    else:
        facecam_box = None
    layout = {
        "type": layout_type,
        "facecam": facecam_box,
        "gameplay": gameplay_box,
    }
    if layout_type == "full_camera" and camera_focus_x is not None:
        layout["focus_x"] = max(0.0, min(1.0, float(camera_focus_x)))
    if facecam_box and facecam_subject_top is not None:
        layout["facecam_subject_top"] = max(
            0.0, min(1.0, float(facecam_subject_top)))
    return layout
