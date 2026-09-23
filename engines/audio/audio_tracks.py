# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Multitrack audio: find the mic-most track (plan 22 §4.4).

OBS multitrack recordings commonly carry [full mix, mic-only, game/desktop]
as separate streams. Every reaction channel (prosody, laughter tagger, ASR)
works dramatically better on the mic-only track — game audio stops firing the
burst tagger and music stops depressing speech signals. Detection is a cheap
local probe: decode a 60s sample from each stream, run the prosody analyzer
we already ship, and pick the stream with the highest voiced-speech ratio —
but only when it clearly beats the default mix (never guess on a tie).

Everything is best-effort: any failure returns None and the pipeline uses the
default (first) audio stream exactly as before.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Optional

from core.ffmpeg_path import get_ffmpeg_path

# The candidate stream must beat the default mix's voiced ratio by this much
# to be trusted — mixes contain the mic too, so a near-tie means "no evidence
# the extra track is the mic" and we stay on the default.
MIN_VOICED_ADVANTAGE = 0.08
SAMPLE_SECONDS = 60


def count_audio_streams(video_path: str) -> int:
    """Number of audio streams, parsed from ffmpeg's stream listing."""
    try:
        result = subprocess.run(
            [get_ffmpeg_path(), "-hide_banner", "-i", video_path],
            capture_output=True, text=True, timeout=30,
        )
        # ffmpeg exits non-zero without an output file; the stream map is
        # still printed on stderr.
        return len(re.findall(r"Stream #\d+:\d+.*?: Audio", result.stderr or ""))
    except Exception:
        return 1


def _voiced_ratio(video_path: str, stream_index: int, offset: float) -> float:
    """Fraction of sampled seconds that read as energetic voiced speech."""
    from engines.audio.prosody import extract_prosody

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [
            get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(max(0.0, offset)), "-t", str(SAMPLE_SECONDS),
            "-i", video_path,
            "-map", f"0:a:{stream_index}",
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            wav_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, timeout=120)
        frames = extract_prosody(wav_path)
        if not frames:
            return 0.0
        voiced = sum(1 for f in frames if f.voiced and f.rms > 0.005)
        return voiced / len(frames)
    except Exception:
        return 0.0
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def select_mic_track(video_path: str, duration: float = 0.0) -> Optional[int]:
    """Index of the mic-most audio stream, or None to use the default.

    None is the common single-track case AND the honest answer whenever the
    probe can't clearly identify a better stream.
    """
    n_streams = count_audio_streams(video_path)
    if n_streams <= 1:
        return None

    # Sample from ~30% into the VOD (past intro screens, inside real content).
    offset = max(0.0, min(duration * 0.3, max(0.0, duration - SAMPLE_SECONDS - 5)))
    ratios = [
        _voiced_ratio(video_path, index, offset)
        for index in range(min(n_streams, 4))  # more than 4 tracks is exotic; cap the probe cost
    ]
    default_ratio = ratios[0]
    best_index = max(range(len(ratios)), key=lambda i: ratios[i])
    if best_index != 0 and ratios[best_index] >= default_ratio + MIN_VOICED_ADVANTAGE:
        return best_index
    return None
