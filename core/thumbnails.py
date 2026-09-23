# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Clip poster (thumbnail) generation.

Grid cards and filmstrips in the Theater+Grid UI need a still image per clip.
We grab a single JPEG from the exported MP4 at the clip's *reaction peak* — the
money frame — not the first frame, which is usually a fade/transition moment.

Used both at clip-save time (job_manager, batch) and lazily on demand
(ClipService.thumbnail, for clips rendered before thumbnails existed).
"""
from __future__ import annotations

import os
import subprocess
import threading
from typing import Optional

from core.ffmpeg_path import get_ffmpeg_path


def thumb_filename(clip_id: str) -> str:
    return f"thumb_{clip_id}.jpg"


def clip_relative_peak(peak_timestamp, start_time, end_time) -> float:
    """Where the reaction peaks *inside* the clip, in seconds, clamped away from
    the very edges so the poster isn't a fade frame. Falls back to the clip
    midpoint (or 1.0s) when no peak is known."""
    try:
        start = float(start_time or 0.0)
        end = float(end_time or 0.0)
        peak = float(peak_timestamp) if peak_timestamp is not None else None
    except (TypeError, ValueError):
        return 1.0
    duration = max(0.0, end - start)
    lo = min(0.4, duration / 4) if duration else 0.0
    hi = max(lo, duration - 0.4) if duration else 1.0
    if peak is None or duration <= 0:
        return max(lo, min(hi, duration / 2)) if duration else 1.0
    return max(lo, min(hi, peak - start))


def generate_thumbnail(
    export_path: str,
    clip_id: str,
    exports_dir: str,
    seek_seconds: float = 1.0,
    width: int = 480,
    jpeg_quality: int = 4,
) -> Optional[str]:
    """Grab a single poster JPEG from an exported clip. Returns the written
    path, or None on any failure (never raises — a missing poster degrades to
    the video-first-frame fallback in the UI)."""
    if not export_path or not os.path.exists(export_path):
        return None
    os.makedirs(exports_dir, exist_ok=True)
    out_path = os.path.join(exports_dir, thumb_filename(clip_id))
    ffmpeg = get_ffmpeg_path()
    width = max(160, min(3840, int(width)))
    jpeg_quality = max(2, min(31, int(jpeg_quality)))

    def _grab(seek: float) -> bool:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, seek):.2f}",
            "-i", export_path,
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", str(jpeg_quality),
            out_path,
        ]
        try:
            subprocess.run(
                cmd, check=True, timeout=30,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            return False
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0

    # Seeking past the end of a short/corrupt clip yields no frame; fall back to
    # the first frame so every clip still gets a poster.
    if _grab(float(seek_seconds)):
        return out_path
    if seek_seconds > 0 and _grab(0.0):
        return out_path
    return None


def generate_filmstrip(
    video_path: str,
    out_path: str,
    *,
    start_seconds: float,
    end_seconds: float,
    frames: int = 24,
    cell_width: int = 96,
    cell_height: int = 54,
    jpeg_quality: int = 82,
) -> Optional[str]:
    """Build a seekable source-VOD contact sheet for editor filmstrips.

    OpenCV performs independent timestamp seeks instead of decoding the entire
    range sequentially. That distinction matters for a three-hour VOD: a sparse
    24-frame overview should take roughly the cost of 24 poster grabs, not a
    three-hour decode. Failed individual seeks remain visibly blank rather than
    being replaced with fabricated imagery.
    """
    if not video_path or not os.path.isfile(video_path):
        return None
    try:
        start = max(0.0, float(start_seconds))
        end = float(end_seconds)
        frame_count = max(4, min(48, int(frames)))
        width = max(48, min(240, int(cell_width)))
        height = max(28, min(160, int(cell_height)))
        quality = max(55, min(95, int(jpeg_quality)))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None

    # Imported lazily so thumbnail-only API startup stays light and so the
    # packaged runtime can resolve its already-bundled OpenCV/native binaries.
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        return None

    span = end - start
    cells = []
    decoded = 0
    target_ratio = width / height
    try:
        for index in range(frame_count):
            timestamp = start + span * ((index + 0.5) / frame_count)
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                cells.append(np.zeros((height, width, 3), dtype=np.uint8))
                continue

            decoded += 1
            source_height, source_width = frame.shape[:2]
            source_ratio = source_width / max(1, source_height)
            if source_ratio > target_ratio:
                crop_width = max(1, int(round(source_height * target_ratio)))
                left = max(0, (source_width - crop_width) // 2)
                frame = frame[:, left:left + crop_width]
            elif source_ratio < target_ratio:
                crop_height = max(1, int(round(source_width / target_ratio)))
                top = max(0, (source_height - crop_height) // 2)
                frame = frame[top:top + crop_height, :]
            cells.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()

    if not decoded:
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet = np.concatenate(cells, axis=1)
    temp_path = f"{out_path}.tmp-{os.getpid()}-{threading.get_ident()}.jpg"
    try:
        if not cv2.imwrite(temp_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            return None
        os.replace(temp_path, out_path)
    except OSError:
        return None
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
    return out_path
