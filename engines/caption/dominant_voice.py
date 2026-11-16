# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Whose voice is a caption phrase? Scores against the VOD's dominant voice.

The map and the VOD gate here feed engines.caption.streamer_voice, which the
scan and every caption path use (Settings > "Only caption my voice"). The
phrase rule below (filter_clip_transcript) is superseded by streamer_voice's
per-word detector and is not wired in (see scripts/research_dominant_voice.py).

The VOD's speech regions are cut into 1.5 s windows every 0.75 s and embedded
with speaker_id.embed. The streamer is taken to be the voice that dominates the
VOD: the centroid starts as the mean of every window and is re-centred five
times on the closest 30%. No transcript, no labels, no fixed regions -- the
filter removed in October built its profile from six fixed 3-minute regions and
scored the streamer's own windows at 0.05-0.3.

MEASURED 2026-11-12 against Fortnite HUD voice icons on five squad VODs of one
streamer (in-game voice): window AUC 0.992-0.994; streamer windows median
0.54-0.59, teammates 0.05-0.08. Dropping a phrase whose BEST window is under
0.15 removed 78% of 3,593 teammate phrases and lost 1.8% of 1,711 streamer
phrases, none of the 70 inside clip windows. Four of the five used VAD regions
as phrases; the one with a cached transcript (the unit this module filters)
gave 67% removed, 1.2% lost. On three solo VODs of another streamer at most
0.2% of full windows scored under 0.20, so the VOD gate keeps every word there.
Unmeasured: reaction streams, VTubers, other games, Discord-only parties.
"""

import os
import struct
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from engines.caption import speaker_id

SAMPLE_RATE = speaker_id.SAMPLE_RATE

WINDOW_SEC = 1.5
HOP_SEC = 0.75
# A region shorter than this is embedded as a window this long centred on it:
# speaker_id.embed refuses less. 41% of the regions in a squad VOD are under
# 1 s, so dropping them would leave most interjections unscored (and kept).
MIN_WINDOW_SEC = speaker_id.MIN_SEGMENT_SEC

CENTROID_ITERATIONS = 5
CENTROID_TOP_FRACTION = 0.30

# A phrase is dropped only when no window over it reaches this. Max rather than
# mean: a phrase with the streamer talking over a teammate keeps its words.
# MEASURED on the five squad VODs (pooled phrases), teammates removed /
# streamer lost / both-icons-lit dropped:
#     max  < 0.10   57% / 1.1% / 6%       mean < 0.15   83% / 1.9% / 17%
#     max  < 0.15   78% / 1.8% / 11%      mean < 0.20   93% / 4.2% / 21%
#     max  < 0.20   91% / 4.0% / 15%      max  < 0.25   95% / 6.5% / 19%
# 0.15, because a missing caption is worse than a guest's line: 0.20 doubles
# the streamer's loss (and loses 2 of 70 phrases inside clips) for 13 points.
# Re-scoring each phrase at caption time on a dense 0.25 s grid was MEASURED
# 2026-11-13 on the same phrases and did not win: at equal teammate removal it
# lost as much of the streamer (80%: 2.3% dense vs 1.9% coarse; 70%: 1.5% vs
# 1.4%) for ~2 points fewer both-icons phrases dropped, and needs the clip's
# audio embedded at caption time. Its 0.15 is not this 0.15: it removes 12
# points fewer teammates.
DROP_BELOW = 0.15

# The VOD gate. Filter only when a second population of voices exists (FAR) AND
# the dominant cluster is substantial (NEAR). Six squad VODs: 35-52% of full
# windows under 0.20, 38-54% at or over 0.40. Three solo VODs: 0.0-0.2% under
# 0.20, 98-99% at or over 0.40. Anything else -- a stream with no clear
# dominant voice -- keeps every word. The gate cannot tell WHICH voice
# dominates: a reaction stream where the video out-talks the streamer would
# pass it with the wrong centroid (unmeasured).
FAR_SIMILARITY = 0.20
NEAR_SIMILARITY = 0.40
MIN_FAR_FRACTION = 0.10
MIN_NEAR_FRACTION = 0.25

CACHE_VERSION = 1


def speech_windows(regions: Sequence[Tuple[float, float]],
                   duration: float) -> List[Tuple[float, float]]:
    """1.5 s windows every 0.75 s inside each speech region, in time order.

    A region's last window is pulled flush with its end when the grid stops
    more than 0.25 s short, so a phrase's final words are covered. Regions of
    1.0-1.5 s are one window; shorter ones get a 1.0 s window centred on them.
    """
    windows: List[Tuple[float, float]] = []
    for start, end in regions:
        start, end = float(start), float(end)
        span = end - start
        if span >= WINDOW_SEC:
            first = len(windows)
            at = start
            while at + WINDOW_SEC <= end + 1e-6:
                windows.append((at, at + WINDOW_SEC))
                at += HOP_SEC
            if windows[-1][1] < end - 0.25 and len(windows) > first:
                windows.append((end - WINDOW_SEC, end))
        elif span >= MIN_WINDOW_SEC:
            windows.append((start, end))
        elif span > 0:
            centre = (start + end) / 2
            at = min(max(0.0, centre - MIN_WINDOW_SEC / 2), max(0.0, duration - MIN_WINDOW_SEC))
            windows.append((at, at + MIN_WINDOW_SEC))
    windows.sort()
    return windows


def dominant_centroid(embeddings: np.ndarray,
                      iterations: int = CENTROID_ITERATIONS,
                      top_fraction: float = CENTROID_TOP_FRACTION) -> Optional[np.ndarray]:
    """Unit vector of the voice that dominates ``embeddings`` (rows unit-norm)."""
    if embeddings is None or len(embeddings) == 0:
        return None
    centre = embeddings.mean(axis=0)
    norm = float(np.linalg.norm(centre))
    if norm == 0:
        return None
    centre = centre / norm
    keep = max(1, int(len(embeddings) * top_fraction))
    for _ in range(iterations):
        closest = np.argsort(-(embeddings @ centre), kind="stable")[:keep]
        centre = embeddings[closest].mean(axis=0)
        norm = float(np.linalg.norm(centre))
        if norm == 0:
            return None
        centre = centre / norm
    return centre


@dataclass
class VoiceMap:
    """Per-window similarity to the dominant voice, in VOD time."""

    starts: np.ndarray
    ends: np.ndarray
    sims: np.ndarray
    centroid: np.ndarray

    def fractions(self) -> Tuple[float, float]:
        """(far, near): shares of full-length windows under FAR / at or over NEAR.

        Only full 1.5 s windows count: padded 1 s windows over short blips score
        lower for every voice (median 0.46 vs 0.59 for the streamer) and would
        make a solo VOD look like it had a second speaker.
        """
        full = (self.ends - self.starts) >= WINDOW_SEC - 1e-3
        sims = self.sims[full]
        if sims.size == 0:
            return 0.0, 0.0
        return (float((sims < FAR_SIMILARITY).mean()),
                float((sims >= NEAR_SIMILARITY).mean()))

    def active(self) -> bool:
        """Whether this VOD has a second voice population worth filtering."""
        far, near = self.fractions()
        return far >= MIN_FAR_FRACTION and near >= MIN_NEAR_FRACTION

    def span_score(self, start: float, end: float) -> Optional[float]:
        """Best similarity of any window overlapping [start, end], or None."""
        if end <= start or self.starts.size == 0:
            return None
        # Windows are sorted by start and none is longer than WINDOW_SEC.
        lo = int(np.searchsorted(self.starts, start - WINDOW_SEC, side="left"))
        hi = int(np.searchsorted(self.starts, end, side="left"))
        if hi <= lo:
            return None
        overlap = (np.minimum(self.ends[lo:hi], end) - np.maximum(self.starts[lo:hi], start)) > 0
        if not overlap.any():
            return None
        return float(self.sims[lo:hi][overlap].max())

    def keeps(self, start: float, end: float) -> bool:
        """False only when this VOD is filtered and no window over the span
        reaches DROP_BELOW. Unscored speech (no window over it) is kept."""
        if not self.active():
            return True
        score = self.span_score(start, end)
        return score is None or score >= DROP_BELOW

    def cache_entry(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "centroid": [round(float(v), 6) for v in self.centroid],
            "windows": [[round(float(a), 3), round(float(b), 3)]
                        for a, b in zip(self.starts, self.ends)],
            "sims": [round(float(v), 4) for v in self.sims],
        }


def voice_map_from_cache(entry) -> Optional[VoiceMap]:
    """The VoiceMap in a cached entry, or None when it is stale or malformed."""
    if not isinstance(entry, dict) or entry.get("version") != CACHE_VERSION:
        return None
    if entry.get("window_sec") != WINDOW_SEC or entry.get("hop_sec") != HOP_SEC:
        return None
    try:
        windows = np.asarray(entry["windows"], dtype=np.float64).reshape(-1, 2)
        sims = np.asarray(entry["sims"], dtype=np.float32)
        centroid = np.asarray(entry["centroid"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    if len(windows) != len(sims) or centroid.shape != (256,):
        return None
    if not (np.all(np.isfinite(windows)) and np.all(np.isfinite(sims))):
        return None
    if np.any(np.diff(windows[:, 0]) < 0):
        return None
    return VoiceMap(windows[:, 0], windows[:, 1], sims, centroid)


def _pcm16_data(path: str) -> Optional[np.ndarray]:
    """The 16 kHz mono int16 samples of a WAV, memory-mapped.

    speaker_id.wav_reader reads the whole file into float32, ~0.9 GB for a
    four-hour VOD; this keeps the cost to one window at a time.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"RIFF":
                return None
            handle.read(4)
            if handle.read(4) != b"WAVE":
                return None
            fmt_ok = False
            while True:
                header = handle.read(8)
                if len(header) < 8:
                    return None
                chunk, size = header[:4], struct.unpack("<I", header[4:])[0]
                if chunk == b"fmt ":
                    body = handle.read(size + (size & 1))
                    tag, channels, rate = struct.unpack("<HHI", body[:8])
                    bits = struct.unpack("<H", body[14:16])[0]
                    fmt_ok = tag == 1 and channels == 1 and rate == SAMPLE_RATE and bits == 16
                elif chunk == b"data":
                    if not fmt_ok:
                        return None
                    offset = handle.tell()
                    count = min(size, os.path.getsize(path) - offset) // 2
                    return np.memmap(path, dtype=np.int16, mode="r", offset=offset, shape=(count,))
                else:
                    handle.seek(size + (size & 1), os.SEEK_CUR)
    except (OSError, struct.error, ValueError):
        return None


def embed_windows(pcm: np.ndarray, regions: Sequence[Tuple[float, float]],
                  embed: Callable[[np.ndarray], Optional[np.ndarray]] = None,
                  cancel_check=None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(starts, ends, unit embeddings) of every speech window of ``pcm``.

    ``pcm`` is the int16 sample array of the analysis WAV (_pcm16_data);
    windows that cannot be embedded are left out. None when none could be.
    """
    embed = embed or speaker_id.embed
    duration = len(pcm) / SAMPLE_RATE
    starts, ends, vectors = [], [], []
    for index, (start, end) in enumerate(speech_windows(regions, duration)):
        # Every 50 windows is ~2.7 s of one core: a cancelled scan's early
        # thread stops well inside _EarlyWork.JOIN_SECONDS.
        if cancel_check and index % 50 == 0:
            cancel_check()
        lo, hi = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        vector = embed(np.asarray(pcm[lo:hi], dtype=np.float32))
        if vector is None:
            continue
        starts.append(start)
        ends.append(end)
        vectors.append(vector)
    if not vectors:
        return None
    return np.asarray(starts), np.asarray(ends), np.asarray(vectors, dtype=np.float32)


def build_voice_map(wav_path: str, regions: Sequence[Tuple[float, float]],
                    embed: Callable[[np.ndarray], Optional[np.ndarray]] = None,
                    cancel_check=None) -> Optional[VoiceMap]:
    """Embed every speech window of the analysis WAV and score it.

    ``embed`` defaults to speaker_id.embed. None when the WAV is unreadable,
    the model is missing, or nothing could be embedded: callers then keep
    every word. MEASURED 2026-11-12: ~54 ms of one CPU core per window, of
    which the Kaldi filterbank is ~1 ms; 1,400-2,600 windows per VOD-hour, so
    75-140 s per VOD-hour on one thread. Batching the ONNX session gained
    nothing (53.6 vs 53.9 ms/window at batch 1 vs 16). The filterbank is
    numpy now (engines.caption.kaldi_fbank, embedding cosine 1.000000 against
    the torch one it replaced, whose default thread pool burned 394 ms of CPU
    per window for the same 50 ms wall).
    """
    pcm = _pcm16_data(wav_path)
    if pcm is None:
        return None
    embedded = embed_windows(pcm, regions, embed, cancel_check)
    if embedded is None:
        return None
    starts, ends, matrix = embedded
    centroid = dominant_centroid(matrix)
    if centroid is None:
        return None
    return VoiceMap(starts, ends, (matrix @ centroid).astype(np.float32),
                    centroid.astype(np.float32))


def filter_clip_transcript(transcript: dict, clip_start: float,
                           voice_map: Optional[VoiceMap]) -> Tuple[dict, list]:
    """Drop the per-clip decode's phrases that are not the streamer's.

    ``transcript`` is clip-relative (asr.transcribe_clip_window). Each segment
    is a phrase: its words' span, shifted by ``clip_start``, is scored against
    the VOD-time windows. The per-clip decode re-transcribes the audio, so its
    words never line up one-to-one with the VOD pass; only times are shared.
    Returns (filtered copy, [(vod_start, vod_end, text, score)] dropped). With
    no map, or a VOD the gate leaves alone, the transcript comes back whole.
    """
    if voice_map is None or not voice_map.active():
        return transcript, []
    kept_segments, dropped = [], []
    for segment in transcript.get("segments", []) or []:
        words = segment.get("words") or []
        if not words:
            kept_segments.append(segment)
            continue
        start = clip_start + float(words[0].get("start", 0.0))
        end = clip_start + float(words[-1].get("end", 0.0))
        if voice_map.keeps(start, end):
            kept_segments.append(segment)
        else:
            dropped.append((start, end, segment.get("text", ""), voice_map.span_score(start, end)))
    filtered = dict(transcript)
    filtered["segments"] = kept_segments
    return filtered, dropped
