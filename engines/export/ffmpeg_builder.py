# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import os
import subprocess
from typing import Dict, List, Optional

from core.ffmpeg_path import get_ffmpeg_path
from engines.export.composition import (
    FACECAM_INNER_EDGE,
    _clamp,
    build_video_graph,
    resolve_plan,
)

# FACECAM_INNER_EDGE is re-exported for callers/tests that import it here; the
# geometry itself now lives in engines.export.composition (the deterministic
# render module). This file owns only encoder selection, fades, captions, audio,
# and command assembly — nothing that depends on detection geometry.
__all__ = [
    "FACECAM_INNER_EDGE",
    "build_ffmpeg_command",
    "check_nvenc_available",
    "cpu_h264_vcodec",
    "get_cpu_h264_encoder",
]

_nvenc_available = None
_cpu_h264_encoder = None

# Software H.264 encoders, best quality-per-bit first.
#
# libx264 is GPL, which is the whole reason this is a probe and not a constant:
# an LGPL ffmpeg build (one without --enable-gpl, so the binary can ship in a
# closed-source product) has no libx264 at all and offers libopenh264 instead,
# and Windows builds also expose the MediaFoundation encoder. Probing lets the
# bundled binary be swapped without touching any export code.
#
# Only libx264 understands -crf/-preset; the others are bitrate-targeted, so
# each entry carries its own quality ladder rather than a shared one.
# Order is MEASURED, not assumed. On a 1080x1920 crop of real gameplay
# (hardest of three sampled segments), against a lossless reference:
#
#   libx264  -preset slow -crf 18            1.42 MB   SSIM 0.99797
#   h264_mf  -b:v 12M                        3.89 MB   SSIM 0.99773
#   libopenh264 -b:v 12M (rc bitrate,high)   1.95 MB   SSIM 0.99531
#   libopenh264 -b:v 20M (rc bitrate,high)   2.22 MB   SSIM 0.99548
#
# libx264 wins outright and stays first, but it is GPL and absent from the LGPL
# build. Between the other two, h264_mf effectively matches x264's quality and
# openh264 does not: openh264 PLATEAUS around 0.9955 no matter how many bits it
# is given (12M -> 20M moves SSIM by 0.0002), so it cannot be tuned out of the
# gap. h264_mf costs ~2.7x the bitrate for that quality, which is the right
# trade for short social clips that the platform re-encodes anyway.
#
# h264_mf is Windows-only; Recall ships Windows-only, and openh264 remains as
# the last resort for any build that lacks both.
_CPU_H264_PREFERENCE = ("libx264", "h264_mf", "libopenh264")

# Quality tiers, keyed by encoder. "final" is the deliverable, "reel" the
# stitched compilation, "preview" the throwaway review proxy made during a scan.
# The bitrate ladders target 1080x1920 9:16 gameplay and are set to land near
# the CRF rungs they replace; they are inherently coarser than CRF, which is
# one more reason to keep libx264 first when the build has it.
_CPU_H264_ARGS = {
    "libx264": {
        "final": ["-preset", "slow", "-crf", "18"],
        "reel": ["-preset", "veryfast", "-crf", "23"],
        "preview": ["-preset", "ultrafast", "-crf", "30"],
    },
    "h264_mf": {
        "final": ["-b:v", "12M"],
        "reel": ["-b:v", "10M"],
        "preview": ["-b:v", "3M"],
    },
    # openh264 defaults to quality-mode rate control, which undershot the target
    # by 10x on easy content (0.86 Mb/s against -b:v 8M) and produced visibly
    # starved output. `-rc_mode bitrate` makes the target mean something.
    # `-profile high -coder cabac` lifts it off Constrained Baseline, which is
    # the default and has neither B-frames nor CABAC.
    "libopenh264": {
        "final": ["-b:v", "12M", "-rc_mode", "bitrate", "-profile", "high", "-coder", "cabac"],
        "reel": ["-b:v", "10M", "-rc_mode", "bitrate", "-profile", "high", "-coder", "cabac"],
        "preview": ["-b:v", "3M", "-rc_mode", "bitrate", "-profile", "high", "-coder", "cabac"],
    },
}


def _ffmpeg_encoders() -> str:
    """`ffmpeg -encoders` stdout, or "" if the binary can't be interrogated."""
    try:
        res = subprocess.run(
            [get_ffmpeg_path(), "-encoders"], capture_output=True, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return res.stdout or ""
    except Exception:
        return ""


def check_nvenc_available() -> bool:
    global _nvenc_available
    if _nvenc_available is not None:
        return _nvenc_available
    _nvenc_available = "h264_nvenc" in _ffmpeg_encoders()
    return _nvenc_available


def get_cpu_h264_encoder() -> str:
    """Name of the software H.264 encoder this ffmpeg build actually carries.

    Falls back to libx264 when the probe fails, which keeps behaviour identical
    to the hardcoded encoder this replaced: a build we can't interrogate is
    assumed to be the GPL one that has always been bundled.
    """
    global _cpu_h264_encoder
    if _cpu_h264_encoder is not None:
        return _cpu_h264_encoder
    listing = _ffmpeg_encoders()
    for name in _CPU_H264_PREFERENCE:
        if name in listing:
            _cpu_h264_encoder = name
            return _cpu_h264_encoder
    _cpu_h264_encoder = "libx264"
    return _cpu_h264_encoder


def cpu_h264_vcodec(tier: str = "final") -> List[str]:
    """`-c:v ...` args for a software H.264 encode at the given quality tier."""
    encoder = get_cpu_h264_encoder()
    ladder = _CPU_H264_ARGS.get(encoder, _CPU_H264_ARGS["libx264"])
    return ["-c:v", encoder] + ladder[tier]


def build_ffmpeg_command(input_path: str, output_path: str, start: float, end: float, facecam_metadata: List[float] = None, ass_path: str = None, fade_in: float = 0.0, fade_out: float = 0.0, layout_type: str = "vertical_split", gameplay_metadata: List[float] = None, preview: bool = False, video_fade_in: float = None, video_fade_out: float = None, audio_fade_in: float = None, audio_fade_out: float = None, gameplay_focus_x: float = 0.5, facecam_subject_top: float = None, camera_cover: bool = False, vtuber_overlay: Optional[Dict] = None) -> List[str]:
    """
    Build the ffmpeg command to render a vertical 9:16 video.

    The framing geometry is resolved once into a validated LayoutPlan and turned
    into a filter graph by engines.export.composition (the deterministic render
    module). This function only wraps that graph with fades, captions, audio, the
    encoder, and command assembly.

    - vertical_split with a facecam: stacked facecam band over gameplay.
    - gameplay_pip: full-height gameplay with a floating top facecam.
    - full_camera: a subject-focused portrait crop of the camera scene.
    - camera_cover (full_camera only): the subject's position was MEASURED by
      the face anchor, so the 9:16 crop can be placed on them and keep them
      full-height instead of fitting the whole frame over a blur. Default
      False renders exactly as before.
    - facecam_subject_top (portrait plates): where the subject's head sits in
      the plate, so the cam crop keeps the face instead of guessing.
    - no/implausible facecam: a straight 9:16 cover-crop of the gameplay.
    - gameplay_metadata (windowed capture): the gameplay chain crops to that
      viewport first, so framing tracks the game window, not chat/chrome.
    - preview=True: a cheap low-res review proxy (never the deliverable).
    """
    duration = max(0.1, float(end) - float(start))
    fade_in = _clamp(fade_in, 0.0, duration / 2.0)
    fade_out = _clamp(fade_out, 0.0, duration / 2.0)
    video_fade_in = _clamp(fade_in if video_fade_in is None else video_fade_in, 0.0, duration / 2.0)
    video_fade_out = _clamp(fade_out if video_fade_out is None else video_fade_out, 0.0, duration / 2.0)
    audio_fade_in = _clamp(fade_in if audio_fade_in is None else audio_fade_in, 0.0, duration / 2.0)
    audio_fade_out = _clamp(fade_out if audio_fade_out is None else audio_fade_out, 0.0, duration / 2.0)

    out_w, out_h = (540, 960) if preview else (1080, 1920)
    # Every vertical clip upscales a portrait slice of the landscape source, so
    # the resampler quality is visible on the final render. Lanczos is much
    # sharper than ffmpeg's default (bilinear) for that enlargement at a
    # fraction of a second's cost. Preview proxies stay on the cheap default —
    # they exist for the in-app review pass, never as the deliverable.
    scale_flags = "" if preview else ":flags=lanczos"

    plan = resolve_plan(
        facecam_metadata, gameplay_metadata, layout_type, gameplay_focus_x,
        facecam_subject_top, camera_cover,
    )
    overlay_mask = None
    if layout_type == "vtuber_overlay" and vtuber_overlay:
        overlay_mask = os.path.abspath(str(vtuber_overlay.get("mask_path") or ""))
        if not os.path.isfile(overlay_mask):
            raise ValueError(f"VTuber overlay mask does not exist: {overlay_mask}")
    filter_chains = build_video_graph(
        plan, out_w, out_h, scale_flags, preview,
        vtuber_overlay=vtuber_overlay if overlay_mask else None,
    )

    v_filters = []
    if video_fade_in > 0.0:
        v_filters.append(f"fade=t=in:st=0:d={video_fade_in}")
    if video_fade_out > 0.0:
        v_filters.append(f"fade=t=out:st={duration - video_fade_out}:d={video_fade_out}")

    if v_filters:
        fade_str = ",".join(v_filters)
        filter_chains.append(f"[outv_raw]{fade_str}[outv_faded]")
        last_v_label = "[outv_faded]"
    else:
        last_v_label = "[outv_raw]"

    if ass_path:
        # Convert path to forward slashes and escape the drive colon for FFmpeg filterchain
        safe_path = ass_path.replace('\\', '/').replace(':', '\\\\:')

        # ass filter reads the .ass file directly including all custom styles and colors
        filter_chains.append(f"{last_v_label}ass={safe_path}[outv]")
    else:
        filter_chains.append(f"{last_v_label}null[outv]")

    a_filters = ["asetpts=PTS-STARTPTS"]
    if audio_fade_in > 0.0:
        a_filters.append(f"afade=t=in:st=0:d={audio_fade_in}")
    if audio_fade_out > 0.0:
        a_filters.append(f"afade=t=out:st={duration - audio_fade_out}:d={audio_fade_out}")

    a_filter_str = ",".join(a_filters)
    filter_chains.append(f"[0:a]{a_filter_str}[outa]")
    filter_complex = ";".join(filter_chains)

    # Dynamically select encoder with proper VBR configuration for NVENC.
    # CQ/CRF 23 read visibly soft/blocky on fast-motion gameplay; 18 is a much
    # safer floor for this content and the encode-time cost is negligible next
    # to the rest of the pipeline. Preview renders (Clip Library review only,
    # never the deliverable) trade quality for speed since every candidate
    # clip gets one during the VOD scan.
    if preview:
        if check_nvenc_available():
            vcodec = ["-c:v", "h264_nvenc", "-preset", "fast", "-rc", "vbr", "-cq", "30", "-b:v", "0"]
        else:
            vcodec = cpu_h264_vcodec("preview")
    elif check_nvenc_available():
        # Final export: the deliverable, and only seconds of footage per clip,
        # so use NVENC's highest-quality preset (p7 + hq tuning) rather than the
        # streaming-tuned "fast" preset, which read soft/blocky on fast-motion
        # gameplay. p7 costs a few extra seconds per clip on the GPU and never
        # touches the scan (which only makes the cheap preview proxies above).
        # CQ 18: measured SSIM 0.996 vs CQ 16 (imperceptible) while ~15%
        # smaller — the preset + lanczos carry the visible quality, not the CQ.
        # High profile over NVENC's default Main: measured slightly BETTER
        # (SSIM 0.99677 vs 0.99664) and 3% smaller on a 9:16 cover-crop of
        # fast-motion gameplay. It is free, so it is unconditional.
        #
        # maxrate caps an otherwise unbounded VBR stream. Uncapped, CQ 18 spent
        # 27-35 Mbps re-encoding a crop of a ~6 Mbps Twitch VOD, i.e. most of
        # those bits preserved the source's own compression artifacts. The cap
        # costs 0.0013 SSIM against a lossless reference (0.99530 vs 0.99664)
        # and roughly halves the file, but that difference does not survive
        # delivery: pushed through a TikTok-like 4 Mbps/30fps pass, capped
        # scored 0.87997 against uncapped's 0.87977 -- a cleaner input
        # re-encodes marginally better. Sized for the platforms these clips are
        # actually posted to, not for archival.
        vcodec = [
            "-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq",
            "-rc", "vbr", "-cq", "18", "-b:v", "0",
            "-profile:v", "high", "-maxrate", "16M", "-bufsize", "32M",
        ]
    else:
        # CPU fallback: on libx264, slow over medium for better quality-per-bit
        # at the same CRF — a single exported clip is short and user-initiated,
        # so the extra encode time is fine. An LGPL build without libx264 gets
        # its own bitrate ladder. NVENC path is unaffected either way.
        vcodec = cpu_h264_vcodec("final")

    cmd = [
        get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration),
        "-i", input_path,
    ]
    if overlay_mask:
        # Give the static alpha plate the same finite timeline as the source.
        # An unbounded `-loop 1` input can otherwise extend the MP4 container
        # duration even though the visible video and audio have ended.
        cmd += [
            "-loop", "1", "-framerate", "60", "-t", str(duration),
            "-i", overlay_mask,
        ]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        # Twitch (and some OBS) VODs carry a full-VOD-length bin_data/text
        # "SubtitleHandler" track that ffmpeg otherwise copies in via default
        # chapter/metadata mapping. It survives -map/-dn/-sn (it rides in as a
        # chapter, not a mapped stream), lands in the clip as a data stream
        # whose duration ~= the whole VOD, and makes Windows Explorer report the
        # clip's Length as hours. Dropping chapters + global metadata strips it
        # so the container duration matches the actual video/audio.
        "-map_chapters", "-1",
        "-map_metadata", "-1",
        "-map_metadata:s", "-1",
    ] + vcodec + [
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
    ]
    if overlay_mask:
        # Belt-and-suspenders output clamp: both image and source inputs are
        # finite, and the muxer is also told the intended clip duration.
        cmd += ["-t", str(duration)]
    cmd += [
        output_path
    ]

    return cmd
