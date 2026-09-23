# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import cv2
import numpy as np
from typing import Iterator, Tuple, Dict

# Consecutive failed reads tolerated before extract_frames_range gives up on a
# slice. A transient decode hiccup recovers in a frame or two; ~300 (~10s at
# 30fps) distinguishes that from genuine end-of-stream / unrecoverable corruption.
_MAX_DECODE_FAILURES = 300

def extract_metadata(video_path: str) -> Dict[str, float]:
    """Extract metadata (fps, duration, resolution) from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    cap.release()
    return {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height
    }

def extract_frames(video_path: str, fps_sample_rate: float = 1.0) -> Iterator[Tuple[float, np.ndarray]]:
    """Yields (timestamp, frame_image) by sampling the video at the given fps rate."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback

    frame_interval = int(round(fps / fps_sample_rate))
    if frame_interval < 1:
        frame_interval = 1

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    current_frame = 0

    while current_frame < frame_count:
        # Seek to the target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = current_frame / fps

        # Yield the image array in memory instead of writing to disk
        yield (timestamp, frame)

        # Jump ahead
        current_frame += frame_interval

    cap.release()


def extract_frames_range(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps_sample_rate: float = 1.0,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (timestamp, frame) for a time slice, sampling at fps_sample_rate.

    Seeks ONCE to the slice start, then advances sequentially with grab() (cheap)
    and only decodes the frames we keep with retrieve(). Much faster than the
    per-frame POS_FRAMES seeking in extract_frames -- this is what the parallel
    perception workers use so each decodes only its own range.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_interval = max(1, int(round(fps / fps_sample_rate)))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = int(round(end_sec * fps)) if end_sec else int(total)
    if total:
        end_frame = min(end_frame, int(total))

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame
    consecutive_failures = 0
    while idx < end_frame:
        ret, frame = cap.read()  # decode the frame we keep
        if not ret:
            # A single failed read used to end the slice — but on stitched
            # Twitch VODs under concurrent load, a *transient* decode hiccup
            # (h264 "no frame!") makes read() fail for a frame or two and then
            # recover. Treating that as end-of-stream silently truncated a
            # worker's slice (the golden-set dropped-slice bug). Skip the bad
            # frame and continue; only give up after a long run of failures,
            # which means genuine EOF or an unrecoverable corrupt tail.
            consecutive_failures += 1
            if consecutive_failures >= _MAX_DECODE_FAILURES:
                break
            # Resync idx to the decoder's real position so later timestamps
            # don't drift; nudge forward manually if the position is unknown.
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            idx = int(pos) if pos and pos > idx else idx + 1
            continue
        consecutive_failures = 0
        yield (idx / fps, frame)
        # Skip the next (frame_interval - 1) frames cheaply without decoding.
        for _ in range(frame_interval - 1):
            if not cap.grab():
                break  # let the outer read() re-detect EOF vs a transient miss
            idx += 1
        idx += 1

    cap.release()
