# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Deterministic 9:16 composition — the pure half of clip framing.

Perception (facecam / gameplay detection) is uncertain and lives upstream in the
vision engines. By the time control reaches this module the geometry is DECIDED.
There is no detection here and no detection-conditioned branching in the render
path: given a validated :class:`LayoutPlan`, exactly one filter graph is emitted.

Two rules keep this reliable, so a small change can't quietly break a neighbour:

  1. Gameplay reaches the canvas through ONE primitive, :func:`_cover_fill`.
     Fullscreen, windowed, PiP-background, stacked-background, standalone — all
     of them call it. Change how gameplay fills once, it changes everywhere.
     It NEVER letterboxes: it scales to cover and crops, always filling.

  2. All the "is this box sane?" decisions happen once, in :func:`resolve_plan`.
     A non-plausible facecam is dropped to a clean full-frame crop rather than
     rendered as a self-overlay; a fullscreen "window" is dropped to None. The
     graph builder trusts the plan completely.

Stacked facecam cover-fills a fixed-height top band (face-safe top bias when the
plate is taller than the band). That keeps the cam:game split identical across
clips; only the crop inside the band varies with the detected plate.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from engines.clip.layout import _is_plausible_facecam

# Composition kinds. These match the persisted layout["type"] vocabulary.
KIND_FULL = "full_gameplay"
KIND_CAMERA = "full_camera"
KIND_STACK = "vertical_split"
KIND_PIP = "gameplay_pip"
KIND_VTUBER = "vtuber_overlay"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _fmt_filter_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _safe_facecam_box(facecam_metadata: List[float]):
    fx, fy, fw, fh = [float(part) for part in facecam_metadata[:4]]
    fx = _clamp(fx, 0.0, 0.98)
    fy = _clamp(fy, 0.0, 0.98)
    fw = _clamp(fw, 0.02, 1.0 - fx)
    fh = _clamp(fh, 0.02, 1.0 - fy)
    return fx, fy, fw, fh


def _safe_gameplay_box(gameplay_metadata):
    """Sanitize a detected gameplay box; None when absent or ~fullscreen."""
    if not gameplay_metadata:
        return None
    gx, gy, gw, gh = [float(part) for part in gameplay_metadata[:4]]
    gx = _clamp(gx, 0.0, 0.9)
    gy = _clamp(gy, 0.0, 0.9)
    gw = _clamp(gw, 0.1, 1.0 - gx)
    gh = _clamp(gh, 0.1, 1.0 - gy)
    if gw >= 0.94 and gh >= 0.94:
        return None
    # Defense in depth: reject quiet-edge hallucinations already persisted on
    # clips (same gate as engines.vision.gameplay_region).
    try:
        from engines.vision.gameplay_region import is_plausible_windowed_gameplay
        if not is_plausible_windowed_gameplay([gx, gy, gw, gh]):
            return None
    except Exception:  # noqa: BLE001 - never block export on a validation import
        pass
    return gx, gy, gw, gh


# Structural window detection lands on the border stroke, so the raw box can
# carry a sliver of window chrome (title-bar underside, frame line). Shave a
# small inset before cropping; the vertical zoom crops the sides much harder
# than this anyway, so the only visible effect is a clean top/bottom edge.
GAMEPLAY_INSET_FRAC_TOP = 0.02
GAMEPLAY_INSET_FRAC_BOTTOM = 0.015
GAMEPLAY_INSET_FRAC_SIDES = 0.01
GAMEPLAY_FACECAM_GUTTER = 0.01
GAMEPLAY_MIN_REMAINDER_FRAC = 0.55
GAMEPLAY_MIN_FACE_OVERLAP_FRAC = 0.15


def _gameplay_box_without_facecam(gameplay_metadata, facecam_metadata=None):
    """Keep a duplicated source facecam out of a windowed gameplay crop.

    Some stream layouts overlay the cam on top of the game. If a structural
    gameplay box is later portrait-cropped around its center, that embedded cam
    can remain in the lower layer while Recall also renders the dedicated cam
    above it. Shift the usable game box away from an edge-anchored cam whenever
    enough of the viewport remains; otherwise leave it untouched rather than
    destroying the game framing.
    """
    box = _safe_gameplay_box(gameplay_metadata)
    if box is None or not facecam_metadata:
        return box
    gx, gy, gw, gh = box
    fx, fy, fw, fh = _safe_facecam_box(facecam_metadata)
    overlap_w = max(0.0, min(gx + gw, fx + fw) - max(gx, fx))
    overlap_h = max(0.0, min(gy + gh, fy + fh) - max(gy, fy))
    if overlap_w <= 0.0 or overlap_h <= 0.0:
        return box
    if overlap_h / max(1e-6, min(gh, fh)) < GAMEPLAY_MIN_FACE_OVERLAP_FRAC:
        return box

    min_width = max(0.10, gw * GAMEPLAY_MIN_REMAINDER_FRAC)
    game_right = gx + gw
    face_center = fx + fw / 2.0
    game_center = gx + gw / 2.0
    if face_center <= game_center:
        clean_left = max(gx, fx + fw + GAMEPLAY_FACECAM_GUTTER)
        clean_width = game_right - clean_left
        if clean_width >= min_width:
            return clean_left, gy, clean_width, gh
    else:
        clean_right = min(game_right, fx - GAMEPLAY_FACECAM_GUTTER)
        clean_width = clean_right - gx
        if clean_width >= min_width:
            return gx, gy, clean_width, gh
    return box


def _gameplay_crop_prefix(gameplay_metadata, facecam_metadata=None) -> str:
    """Crop-to-gameplay filter step (with trailing comma), or ""."""
    box = _gameplay_box_without_facecam(gameplay_metadata, facecam_metadata)
    if box is None:
        return ""
    gx, gy, gw, gh = box
    gx += gw * GAMEPLAY_INSET_FRAC_SIDES
    gy += gh * GAMEPLAY_INSET_FRAC_TOP
    gw -= gw * 2 * GAMEPLAY_INSET_FRAC_SIDES
    gh -= gh * (GAMEPLAY_INSET_FRAC_TOP + GAMEPLAY_INSET_FRAC_BOTTOM)
    return (
        f"crop=iw*{_fmt_filter_number(gw)}:ih*{_fmt_filter_number(gh)}:"
        f"iw*{_fmt_filter_number(gx)}:ih*{_fmt_filter_number(gy)},"
    )


# Handbook 13 §6.2: crop to the detected facecam plate with a safety inset
# that scales with the plate size (trims border remnants, decorative overlay
# frames, and anything just outside the box -- e.g. a cat tail poking past
# the plate edge -- rather than pasting that bleed onto the gameplay), and no
# loud external stroke -- the streamer's native facecam frame plus a subtle
# 1px inner edge is the default treatment.
# Detected plates tend to sit slightly loose on the trailing edges (right/bottom)
# relative to left/top, so the inset is biased rather than trimmed evenly on
# all four sides -- otherwise the leading edges get over-trimmed before the
# trailing-edge bleed is fully gone.
FACECAM_INSET_FRAC_LEFT = 0.04
FACECAM_INSET_FRAC_TOP = 0.05
FACECAM_INSET_FRAC_RIGHT = 0.04
FACECAM_INSET_FRAC_BOTTOM = 0.06
# Stacked cover-fill retains more of the crop's trailing edge than PiP. Real
# plates can be detected slightly below their visible frame, which otherwise
# turns those retained source rows into a horizontal band above gameplay.
FACECAM_STACK_INSET_FRAC_BOTTOM = 0.10
FACECAM_INNER_EDGE = "drawbox=x=0:y=0:w=iw:h=ih:color=black@0.35:t=1"
# A stacked camera is flush with gameplay, so a bottom edge reads as a dark
# separator baked into the video. Keep the subtle treatment on the exposed
# top/side edges only; floating PiP cameras still use the full inner edge.
FACECAM_STACK_INNER_EDGE = (
    "drawbox=x=0:y=0:w=iw:h=1:color=black@0.35:t=fill,"
    "drawbox=x=0:y=0:w=1:h=ih:color=black@0.35:t=fill,"
    "drawbox=x=iw-1:y=0:w=1:h=ih:color=black@0.35:t=fill"
)

# Facecam framing (stacked): fixed ~30% top band so every stacked card shares
# the same cam:game split. Plate aspect only affects cover-fill cropping inside
# the band (face-safe top bias), not how tall the band is. PiP keeps its own
# wide-window crop path below.
_SOURCE_ASPECT = 16.0 / 9.0  # streams are 16:9; used to map a normalized box to output px
FACECAM_BAND_H = 576  # exactly 30% of 1920
# Sized against creator feedback: at 760x560 the floating cam read as "a
# reduced little square" next to the full-width band the split layout gives;
# the cam is the content for reaction-led creators, so let it take ~3/4 of
# the canvas width when its aspect allows.
PIP_CAM_MAX_W = 810
PIP_CAM_MAX_H = 640
# PiP cams render as a WIDE window: a taller-than-16:9 plate is cropped to
# 16:9 with a face-safe top bias before scaling. This is the look creators
# read as "an actual webcam" (validated against real decks); it is only safe
# now that refine_layout_boxes snaps the plate to its true border — cropping
# the previously-bloated boxes was what made faces look zoomed, which is why
# an earlier pass removed the crop entirely.
PIP_CAM_TARGET_ASPECT = 16.0 / 9.0
# Fraction of the trimmed height taken from the TOP of the plate. Webcam
# plates carry their dead space as headroom (and any residual top-edge
# detection bleed sits there too — measured on a real VOD where a HUD strip
# above the cam out-competed the plate border in the edge snap), so the trim
# is top-heavy: it removes headroom/bleed and raises the eye line toward
# standard framing. Verified frame-by-frame on a real creator cam at
# 0.35/0.55/0.65/0.72 — lower values left visible gameplay bleed. Stacked
# cover-fill reuses the same bias when the plate is taller than the band.
PIP_CAM_FACE_BIAS = 0.72
PIP_TOP_MARGIN = 24

# Portrait plates need subject-relative framing because a fixed downward bias can remove
# the face. Derive the bias from measured subject position; keep the fallback for
# unmeasured layouts.
PIP_CAM_PORTRAIT_ASPECT = 1.05
PIP_CAM_PORTRAIT_FACE_BIAS = 0.35

# Leave a small amount of rendered headroom above the measured subject. Subject-relative
# framing accommodates both high webcam faces and low avatar heads.
CAM_SUBJECT_HEADROOM = 0.04


def _subject_top_in_crop(subject_top: float, bottom_inset: float) -> float:
    """Re-express a subject top measured on the PLATE in inset-crop units.

    ``subject_top`` is a fraction of the detected plate's height, but what
    reaches the canvas is the plate minus its safety insets, so the two
    coordinate spaces differ by exactly that trim.
    """
    kept = max(1e-6, 1.0 - FACECAM_INSET_FRAC_TOP - bottom_inset)
    return (float(subject_top) - FACECAM_INSET_FRAC_TOP) / kept


def _face_bias(
    aspect: float,
    target_aspect: float,
    subject_top: Optional[float] = None,
    bottom_inset: float = FACECAM_INSET_FRAC_BOTTOM,
) -> float:
    """Vertical crop bias for a plate of this pixel aspect.

    ``subject_top`` (a measured fraction of the plate's height) turns the bias
    into geometry rather than a guess: cover-filling a ``target_aspect`` window
    keeps ``aspect / target_aspect`` of the plate's height, and the bias is
    simply where that window has to sit for the subject's head to land
    CAM_SUBJECT_HEADROOM below its top edge.
    """
    # Use measured subject position for both portrait and landscape plates. A fixed
    # landscape bias can clip the chin when the subject sits low; retain existing
    # constants only when measurement is unavailable.
    fallback = (PIP_CAM_FACE_BIAS if aspect >= PIP_CAM_PORTRAIT_ASPECT
                else PIP_CAM_PORTRAIT_FACE_BIAS)
    if subject_top is None:
        return fallback
    kept = aspect / max(1e-6, target_aspect)
    if kept >= 1.0:
        return fallback  # nothing is trimmed; bias is inert
    top = _subject_top_in_crop(subject_top, bottom_inset)
    return _clamp((top - CAM_SUBJECT_HEADROOM * kept) / (1.0 - kept), 0.0, 1.0)


def _facecam_plate_aspect(fw: float, fh: float) -> float:
    """Pixel aspect (w/h) of the detected plate, assuming a 16:9 source frame."""
    return max(0.2, (fw / max(fh, 1e-6)) * _SOURCE_ASPECT)


def _facecam_band_height(fw: float = 0.0, fh: float = 0.0) -> int:
    """Fixed top-band height (even px). Plate size does not change the split."""
    return FACECAM_BAND_H


def _facecam_cropped_dims(
    fw: float,
    fh: float,
    bottom_inset: float = FACECAM_INSET_FRAC_BOTTOM,
):
    """Normalized (w, h) of the plate AFTER the safety inset — what renders."""
    cw = max(fw - fw * (FACECAM_INSET_FRAC_LEFT + FACECAM_INSET_FRAC_RIGHT), fw * 0.5)
    ch = max(fh - fh * (FACECAM_INSET_FRAC_TOP + bottom_inset), fh * 0.5)
    return cw, ch


def _pip_facecam_size(fw: float, fh: float, preview: bool):
    """Wide-window PiP footprint for the camera plate.

    Tall plates render at PIP_CAM_TARGET_ASPECT via the face-safe crop in
    :func:`_pip_wide_crop`; already-wide plates keep their own aspect. Bounded
    by both width and height so no camera takes over the portrait canvas.
    """
    cw, ch = _facecam_cropped_dims(fw, fh)
    aspect = max(_facecam_plate_aspect(cw, ch), PIP_CAM_TARGET_ASPECT)
    scale = 0.5 if preview else 1.0
    max_w = int(PIP_CAM_MAX_W * scale)
    max_h = int(PIP_CAM_MAX_H * scale)
    width = max_w
    height = int(round(width / aspect))
    if height > max_h:
        height = max_h
        width = int(round(height * aspect))
    width = max(2, width - width % 2)
    height = max(2, height - height % 2)
    return width, height


def _pip_wide_crop(
    fw: float, fh: float, subject_top: Optional[float] = None,
) -> str:
    """Filter stage cropping a tall plate to the wide PiP aspect ("" if wide).

    Operates on the already-plate-cropped stream, keeping full width and
    trimming height with the face-safe top bias.
    """
    cw, ch = _facecam_cropped_dims(fw, fh)
    aspect = _facecam_plate_aspect(cw, ch)
    if aspect >= PIP_CAM_TARGET_ASPECT - 1e-3:
        return ""
    height_frac = aspect / PIP_CAM_TARGET_ASPECT
    bias = _face_bias(aspect, PIP_CAM_TARGET_ASPECT, subject_top)
    return (
        f"crop=iw:ih*{_fmt_filter_number(height_frac)}:0:"
        f"(ih-oh)*{_fmt_filter_number(bias)},"
    )


def _facecam_plate_crop(
    fx: float,
    fy: float,
    fw: float,
    fh: float,
    bottom_inset: float = FACECAM_INSET_FRAC_BOTTOM,
) -> str:
    cw, ch = _facecam_cropped_dims(fw, fh, bottom_inset)
    cx = fx + fw * FACECAM_INSET_FRAC_LEFT
    cy = fy + fh * FACECAM_INSET_FRAC_TOP
    return (
        f"crop=iw*{_fmt_filter_number(cw)}:ih*{_fmt_filter_number(ch)}:"
        f"iw*{_fmt_filter_number(cx)}:ih*{_fmt_filter_number(cy)}"
    )


@dataclass(frozen=True)
class LayoutPlan:
    """A fully validated composition spec. No perception, no ambiguity left.

    ``facecam`` / ``gameplay`` are sanitized normalized boxes (or None).
    ``gameplay_crop`` is the ffmpeg crop-to-source prefix ("" for fullscreen).
    ``facecam_subject_top`` is the measured head position inside the facecam
    plate (None = unmeasured; the portrait crop falls back to its constant).
    """

    kind: str
    facecam: Optional[Tuple[float, float, float, float]]
    gameplay: Optional[Tuple[float, float, float, float]]
    gameplay_crop: str
    focus_x: float
    facecam_subject_top: Optional[float] = None
    # A full_camera scene whose subject position was MEASURED, not guessed.
    # See KIND_CAMERA in build_video_graph: the fit-over-blur treatment exists
    # because a 9:16 cover crop can slice a creator whose position is unknown.
    # When the face anchor supplies that position the crop is safe, and it
    # keeps the subject full-height instead of shrinking them into a letterbox.
    # Defaults False so every clip without an anchor renders exactly as before.
    camera_cover: bool = False

    @property
    def windowed(self) -> bool:
        return self.gameplay is not None


def resolve_plan(
    facecam_metadata: Optional[List[float]],
    gameplay_metadata: Optional[List[float]],
    layout_type: str,
    focus_x: float,
    facecam_subject_top: Optional[float] = None,
    camera_cover: bool = False,
) -> LayoutPlan:
    """Turn raw (possibly bad) detection metadata into ONE validated plan.

    This is the only place framing decisions and sanity gates live, so the graph
    builder never has to defend against a bad box:

      * a facecam that isn't a plausible corner panel is dropped -> full-frame
        gameplay, never a self-overlay;
      * ``full_gameplay`` drops any facecam;
      * a fullscreen / absent gameplay box becomes None (center-fill the frame).
    """
    focus_x = _clamp(focus_x, 0.0, 1.0)
    gameplay = _safe_gameplay_box(gameplay_metadata)
    gameplay_crop = _gameplay_crop_prefix(gameplay_metadata, facecam_metadata)

    use_facecam = (
        bool(facecam_metadata)
        and layout_type != KIND_FULL
        and _is_plausible_facecam(facecam_metadata)
    )
    if not use_facecam:
        kind = KIND_CAMERA if layout_type == KIND_CAMERA else KIND_FULL
        return LayoutPlan(kind, None, gameplay, gameplay_crop, focus_x,
                          camera_cover=bool(camera_cover) and kind == KIND_CAMERA)

    if layout_type == KIND_VTUBER:
        kind = KIND_VTUBER
    else:
        kind = KIND_PIP if layout_type == KIND_PIP else KIND_STACK
    facecam = _safe_facecam_box(facecam_metadata)
    subject_top = (
        _clamp(facecam_subject_top, 0.0, 1.0)
        if facecam_subject_top is not None else None
    )
    return LayoutPlan(
        kind, facecam, gameplay, gameplay_crop, focus_x, subject_top)


def _cover_fill(width: int, height: int, focus_x: float, scale_flags: str, source_crop: str) -> str:
    """THE single way gameplay reaches the canvas.

    Optionally crop to a source rect, then scale UP to cover the target and crop
    to it — filling the target completely. Never scales-to-fit + pads, so there
    are never black letterbox gutters. A non-center ``focus_x`` slides the final
    crop horizontally to keep the action in frame.
    """
    chain = f"{source_crop}scale={width}:{height}:force_original_aspect_ratio=increase{scale_flags},"
    if abs(focus_x - 0.5) < 1e-6:
        return chain + f"crop={width}:{height}"
    focus = _fmt_filter_number(focus_x)
    return chain + f"crop={width}:{height}:(iw-ow)*{focus}:(ih-oh)/2"


def build_video_graph(
    plan: LayoutPlan,
    out_w: int,
    out_h: int,
    scale_flags: str,
    preview: bool,
    vtuber_overlay: Optional[dict] = None,
) -> List[str]:
    """Emit the video filter chains for ``plan``, ending in ``[outv_raw]``.

    Pure and deterministic: identical plan -> identical graph. The only three
    shapes are full-frame, stacked (cam band over gameplay), and PiP (gameplay
    with a floating cam). Gameplay always flows through :func:`_cover_fill`.
    """
    def fill(width: int, height: int) -> str:
        return _cover_fill(width, height, plan.focus_x, scale_flags, plan.gameplay_crop)

    if plan.kind == KIND_VTUBER and plan.facecam is not None and vtuber_overlay:
        fx, fy, fw, fh = plan.facecam
        width_frac = _clamp(vtuber_overlay.get("width", 500.0 / 1080.0), 0.1, 0.9)
        top_frac = _clamp(vtuber_overlay.get("top", -28.0 / 1920.0), -0.25, 0.5)
        lower_fade = _clamp(vtuber_overlay.get("lower_fade", 150.0 / 560.0), 0.0, 0.5)
        avatar_w = max(2, int(round(out_w * width_frac)))
        avatar_w -= avatar_w % 2
        avatar_h = avatar_w
        overlay_y = int(round(out_h * top_frac))
        crop = (
            f"crop=iw*{_fmt_filter_number(fw)}:ih*{_fmt_filter_number(fh)}:"
            f"iw*{_fmt_filter_number(fx)}:ih*{_fmt_filter_number(fy)}"
        )
        fade_expr = (
            "geq=lum='lum(X,Y)*clip((H-Y)/(H*"
            f"{_fmt_filter_number(lower_fade)})\\,0\\,1)'"
            if lower_fade > 0.0 else "null"
        )
        return [
            "[0:v]setpts=PTS-STARTPTS,split[bg_src][avatar_src]",
            f"[bg_src]{fill(out_w, out_h)}[bg_scaled]",
            f"[avatar_src]{crop},format=rgba,scale={avatar_w}:{avatar_h}{scale_flags}[avatar_pixels]",
            f"[1:v]alphaextract,scale={avatar_w}:{avatar_h}{scale_flags},{fade_expr}[avatar_mask]",
            "[avatar_pixels][avatar_mask]alphamerge=shortest=1[avatar]",
            f"[bg_scaled][avatar]overlay=x=(W-w)/2:y={overlay_y}:shortest=1:format=auto,setsar=1[outv_raw]",
        ]

    if plan.kind == KIND_PIP and plan.facecam is not None:
        fx, fy, fw, fh = plan.facecam
        cam_crop = _facecam_plate_crop(fx, fy, fw, fh)
        pip_cam_w, pip_cam_h = _pip_facecam_size(fw, fh, preview)
        wide_crop = _pip_wide_crop(fw, fh, plan.facecam_subject_top)
        return [
            "[0:v]setpts=PTS-STARTPTS,split[bg_src][cam_src]",
            f"[bg_src]{fill(out_w, out_h)}[bg_scaled]",
            # Wide-window cam: tall plates take the face-safe 16:9 crop, then
            # scale into the restrained footprint (aspects match by design, so
            # nothing is squashed).
            f"[cam_src]{cam_crop},{wide_crop}scale={pip_cam_w}:{pip_cam_h}{scale_flags},"
            f"{FACECAM_INNER_EDGE}[cam_scaled]",
            f"[bg_scaled][cam_scaled]overlay=x=(W-w)/2:y={PIP_TOP_MARGIN}[outv_raw]",
        ]

    if plan.kind == KIND_STACK and plan.facecam is not None:
        fx, fy, fw, fh = plan.facecam
        cam_crop = _facecam_plate_crop(
            fx, fy, fw, fh, FACECAM_STACK_INSET_FRAC_BOTTOM,
        )
        band_h = _facecam_band_height(fw, fh) * out_h // 1920
        band_h -= band_h % 2
        game_h = out_h - band_h
        # The band is what this plate cover-fills, so it -- not PiP's 16:9 --
        # is the target aspect the subject has to be placed inside.
        stack_bias = _face_bias(
            _facecam_plate_aspect(
                *_facecam_cropped_dims(fw, fh, FACECAM_STACK_INSET_FRAC_BOTTOM)),
            out_w / max(1, band_h),
            plan.facecam_subject_top,
            FACECAM_STACK_INSET_FRAC_BOTTOM,
        )
        return [
            "[0:v]setpts=PTS-STARTPTS,split[bg_src][cam_src]",
            # Facecam (top band): cover-fill the band (no side letterbox). Tall
            # plates trim headroom with the same face-safe bias PiP uses.
            f"[cam_src]{cam_crop},scale={out_w}:{band_h}:force_original_aspect_ratio=increase{scale_flags},"
            f"crop={out_w}:{band_h}:(iw-ow)/2:(ih-oh)*{_fmt_filter_number(stack_bias)},"
            f"{FACECAM_STACK_INNER_EDGE}[cam_scaled]",
            f"[bg_src]{fill(out_w, game_h)}[bg_scaled]",
            "[cam_scaled][bg_scaled]vstack=inputs=2[outv_raw]",
        ]

    if plan.kind == KIND_CAMERA and plan.camera_cover:
        # The subject's position was MEASURED (face anchor), so the objection
        # below does not apply: the crop can be placed on them rather than
        # guessed. Cover-filling keeps the creator full-height, where the
        # fit-over-blur treatment shrinks them into the middle third.
        return [
            f"[0:v]setpts=PTS-STARTPTS,"
            f"{_cover_fill(out_w, out_h, plan.focus_x, scale_flags, plan.gameplay_crop)}"
            f"[outv_raw]"
        ]

    if plan.kind == KIND_CAMERA:
        # Fullscreen webcams are landscape camera originals, not gameplay.  A
        # 9:16 cover crop can lose the creator entirely when they lean toward an
        # edge.  Keep the complete source visible over a softly blurred cover
        # of itself: no black bars, no guessed face crop, and no sliced subject.
        blur_sigma = 16 if preview else 28
        camera_crop = plan.gameplay_crop
        return [
            "[0:v]setpts=PTS-STARTPTS,split[bg_src][camera_src]",
            f"[bg_src]{fill(out_w, out_h)},gblur=sigma={blur_sigma}[bg_scaled]",
            f"[camera_src]{camera_crop}scale={out_w}:{out_h}:force_original_aspect_ratio=decrease{scale_flags}[camera_scaled]",
            "[bg_scaled][camera_scaled]overlay=x=(W-w)/2:y=(H-h)/2[outv_raw]",
        ]

    # KIND_FULL — straight portrait cover-crop of the (optionally windowed) game.
    return [f"[0:v]setpts=PTS-STARTPTS,{fill(out_w, out_h)}[outv_raw]"]
