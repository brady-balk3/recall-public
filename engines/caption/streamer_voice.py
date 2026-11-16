# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Captions in the streamer's voice only.

A scan builds, once per VOD and from the VOD's own audio, a detector that says
every 20 ms whether the STREAMER is speaking (streamer_activity, trained with
no labels on weak labels from the dominant-voice map, dominant_voice), and
caches its probabilities by the settings-independent audio key. Every path
that shows or burns machine caption words then drops a word when no frame over
its span reaches KEEP_AT -- on words in VOD time, so the per-clip decode
(clip start + word time) and the scan's own transcript are filtered alike.

MEASURED 2026-11-15 with THIS module end to end (scripts/research_streamer_
activity.py `streamer-voice`: maps built from the 16 kHz analysis audio of
four Fortnite squad VODs and one solo VOD, then the product's saved real
per-clip decodes of 69 clips filtered): of the HUD-labelled words, 244/275
teammate words removed (88.7%), 2/227 streamer words lost (0.9%), 10/200
words spoken while both talk dropped (5.0%); 0/134 words removed on the solo
VOD, whose gate stayed off. The research detector (44.1 kHz stereo) measured
87.6% / 0.4% / 5.5% on the same words; the two agree on 97.2% of them.
Gating the audio before transcription was worse (it changes and loses the
streamer's words) and is not used.

No smoothing: on the same words, dropping a kept word stranded between two
dropped ones lost one more streamer word and removed no teammate word, and
keeping a dropped word between two kept ones brought one teammate word back
and saved no streamer word. One-word caption chunks went from 10% of the
decode's chunks to 12% filtered.

Each VOD passes a gate first (dominant_voice.VoiceMap.active): a solo stream
has no second voice population and is never filtered. When a streamer's
voice profile (voice_profile) exists and this VOD's dominant voice is NOT
that streamer, the per-VOD bootstrap has latched onto someone else; the VOD
is then filtered with the profile's detectors, or not at all.

Nothing here raises into a scan or a render: a missing model, an unreadable
WAV or a malformed cache means no map, and no map means every word is kept.
Pure numpy and onnxruntime (via speaker_id); no torch.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from engines.caption import dominant_voice as dv
from engines.caption import streamer_activity as sa

CACHE_VERSION = 1
SAMPLE_RATE = dv.SAMPLE_RATE

# A word is kept when its best 20 ms frame reaches this. MEASURED on the
# labelled decodes above: 0.5 removes 87.6% of teammate words for 0.4% of the
# streamer's; the threshold matching 90% removal (0.83) loses 4.8%.
KEEP_AT = 0.5

# A lone short burst: one or two kept words, at most LONE_MAX_SEC long, none
# reaching LONE_SURE_AT, with no other kept word within LONE_QUIET_SEC. That
# is what a teammate's callout looks like when the detector half-believes it
# ("Knocked one." at 0.87/0.95 in a clip the streamer never spoke in, found by
# the creator 2026-11-16). MEASURED on the same labelled decodes: teammate
# words removed 87.6% -> 90.2%, streamer words lost 0.4% -> 0.9% (2 of 227).
# The streamer's own sure exclamations ("Oh my God!" at 1.00) are untouched.
LONE_MAX_WORDS = 2
LONE_MAX_SEC = 1.0
LONE_SURE_AT = 0.97
LONE_QUIET_SEC = 3.0
# Kept words closer than this, with no dropped word between, are one burst.
LONE_RUN_GAP_SEC = 1.0

# Weak labels (streamer_activity.weak_labels): speech whose every covering
# dominant-voice window scores >= 0.45 is the streamer, < 0.10 someone else.
# The research script's WEAK_HI / WEAK_LO, never tuned on HUD labels.
STREAMER_AT = 0.45
OTHER_BELOW = 0.10

# Log-mel is computed this many seconds of the memory-mapped WAV at a time: a
# four-hour VOD is 0.9 GB as float32. The FFT of a block is complex128 per
# 40 ms frame, ~16x the block's own samples: MEASURED 2026-11-15 (tracemalloc,
# 20 minutes of audio) a 600 s block peaked at 614 MB, a 60 s one at 66 MB,
# for the same CPU time (0.48 vs 0.38 s).
LOGMEL_BLOCK_SEC = 60.0

# Probabilities are stored as 0-255 at the detector's 20 ms rate: 180 KB per
# VOD-hour before compression. A word on the 0.5 line moves by at most 1/510.
_LEVELS = 255

# Profile agreement (voice_profile). MEASURED 2026-11-15 on the window
# embeddings of nine VODs: centroids of six squad VODs of one streamer agree at
# cosine 0.93-0.97, against 0.24-0.35 for three VODs of another streamer and
# 0.16-0.21 for a teammate's mean voice. Thinning the streamer's own windows to
# 4-30% of a labelled VOD (a squad that out-talks her) flipped the per-VOD
# centroid to a teammate in 9 of 20 trials -- cosine to the profile 0.13-0.29,
# window AUC streamer-vs-teammate 0.10-0.38, i.e. the filter would have removed
# HER -- while the profile centroid kept AUC 0.96-0.98 on the same windows.
PROFILE_AGREE = 0.80
# When the profile takes over, "is the streamer here at all": the share of full
# windows at or over NEAR against the profile centroid. 0.04-0.26 in the
# thinned trials above, ~0.0 for another streamer's VOD (cosine 0.24-0.35).
PROFILE_MIN_NEAR = 0.02


def cache_name(audio_key: str, ext: str) -> str:
    """``{audio_key}_streamer_voice_v1{ext}``: keyed like the WAV and the
    speech regions, so the storage janitor attributes it to its source and
    keeps it in the results tier."""
    return f"{audio_key}_streamer_voice_v{CACHE_VERSION}{ext}"


# ---------------------------------------------------------------- array codec


def pack_array(array: np.ndarray) -> Dict[str, Any]:
    """A numpy array as JSON-safe data (dtype, shape, base64 bytes)."""
    array = np.ascontiguousarray(array)
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "b64": base64.b64encode(array.tobytes()).decode("ascii")}


_ALLOWED_DTYPES = {"|u1", "<f4", "<f2"}


def unpack_array(entry, dtype: Optional[str] = None, ndim: Optional[int] = None) -> Optional[np.ndarray]:
    """The array in ``entry``, or None when it is malformed or not ``dtype``."""
    try:
        if not isinstance(entry, dict) or entry.get("dtype") not in _ALLOWED_DTYPES:
            return None
        if dtype is not None and entry["dtype"] != dtype:
            return None
        shape = tuple(int(v) for v in entry["shape"])
        if ndim is not None and len(shape) != ndim:
            return None
        raw = base64.b64decode(entry["b64"].encode("ascii"), validate=True)
        array = np.frombuffer(raw, dtype=np.dtype(entry["dtype"]))
        if array.size != int(np.prod(shape, dtype=np.int64)):
            return None
        array = array.reshape(shape).copy()
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        return None
    return array


def pack_mlp(mlp: sa.MLP) -> Dict[str, Any]:
    return {name: pack_array(np.asarray(getattr(mlp, name), np.float32))
            for name in ("w1", "b1", "w2", "b2")}


def unpack_mlp(entry) -> Optional[sa.MLP]:
    if not isinstance(entry, dict):
        return None
    w1 = unpack_array(entry.get("w1"), "<f4", 2)
    b1 = unpack_array(entry.get("b1"), "<f4", 1)
    w2 = unpack_array(entry.get("w2"), "<f4", 2)
    b2 = unpack_array(entry.get("b2"), "<f4", 1)
    if any(v is None for v in (w1, b1, w2, b2)):
        return None
    if not (w1.shape[1] == b1.shape[0] == w2.shape[0] and w2.shape[1] == 1 and b2.shape == (1,)):
        return None
    if w1.shape[0] != sa.N_MELS * len(sa.CONTEXT_SPANS):
        return None
    return sa.MLP(w1, b1, w2, b2)


# ---------------------------------------------------------------- the map


@dataclass
class StreamerVoice:
    """One VOD's streamer-voice map: what the scan caches, what captions read.

    ``probs`` is p(streamer speaking) per 20 ms frame as 0-255, None when the
    VOD is not filtered; ``gate`` records why, with the numbers behind it.
    ``mlp`` is the detector trained on this VOD (None when none was), kept for
    the streamer's voice profile.
    """

    gate: Dict[str, Any]
    centroid: np.ndarray
    probs: Optional[np.ndarray] = None
    mlp: Optional[sa.MLP] = None
    frame_sec: float = sa.FRAME_SEC
    first_centre: float = sa.FRAME_SEC

    @property
    def active(self) -> bool:
        return bool(self.gate.get("active")) and self.probs is not None and self.probs.size > 0

    def word_score(self, start: float, end: float) -> Optional[float]:
        """Best probability over the frames spanning [start, end] (VOD seconds):
        from the frame centred at or before ``start`` to the one at or after
        ``end``, the research script's span_frames. None when the span lies
        wholly outside the map."""
        if self.probs is None or not np.isfinite(start) or not np.isfinite(end):
            return None
        end = max(end, start)
        # In streamer_activity's 44.1 kHz sample units, exactly as the
        # research scored words: the 16 kHz grid has the same 20 ms frames
        # centred at the same times, and the same float expression keeps a
        # word that starts on a frame centre on the same side of it.
        # (from_cache refuses any other grid.)
        hop, half, rate = sa.HOP, sa.WIN / 2, sa.SAMPLE_RATE
        lo = int(np.floor((start * rate - half) / hop))
        hi = int(np.ceil((end * rate - half) / hop)) + 1
        lo, hi = max(0, lo), min(len(self.probs), max(hi, lo + 1))
        if hi <= lo:
            return None
        return float(self.probs[lo:hi].max()) / _LEVELS

    def keeps(self, start: float, end: float) -> bool:
        """False only when this VOD is filtered and the word's best frame is
        under KEEP_AT. A word the map does not cover is kept."""
        if not self.active:
            return True
        score = self.word_score(start, end)
        return score is None or score >= KEEP_AT

    def summary(self) -> Dict[str, Any]:
        """The gate and its numbers, for the scan's stage detail and logs."""
        out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.gate.items()}
        out["frames"] = 0 if self.probs is None else int(self.probs.size)
        return out

    def to_cache(self) -> Dict[str, Any]:
        return {
            "version": CACHE_VERSION,
            "frame_sec": self.frame_sec,
            "first_centre": self.first_centre,
            "keep_at": KEEP_AT,
            "gate": dict(self.gate),
            "centroid": pack_array(np.asarray(self.centroid, np.float32)),
            "probs": None if self.probs is None else pack_array(np.asarray(self.probs, np.uint8)),
            "mlp": None if self.mlp is None else pack_mlp(self.mlp),
        }

    @classmethod
    def from_cache(cls, entry) -> Optional["StreamerVoice"]:
        """The map in a cached entry, or None when it is stale or malformed
        (the caller then treats the VOD as unfiltered, or rebuilds)."""
        if not isinstance(entry, dict) or entry.get("version") != CACHE_VERSION:
            return None
        if entry.get("frame_sec") != sa.FRAME_SEC or entry.get("first_centre") != sa.FRAME_SEC:
            return None
        if entry.get("keep_at") != KEEP_AT:
            return None
        gate = entry.get("gate")
        if not isinstance(gate, dict) or not isinstance(gate.get("active"), bool):
            return None
        centroid = unpack_array(entry.get("centroid"), "<f4", 1)
        if centroid is None or centroid.shape != (256,):
            return None
        probs = None
        if entry.get("probs") is not None:
            probs = unpack_array(entry["probs"], "|u1", 1)
            if probs is None:
                return None
        if gate["active"] and probs is None:
            return None
        mlp = unpack_mlp(entry["mlp"]) if entry.get("mlp") is not None else None
        return cls(gate=gate, centroid=centroid, probs=probs, mlp=mlp)


# ---------------------------------------------------------------- caption filter


def keep_mask(words: Sequence[dict], voice: Optional[StreamerVoice],
              offset: float = 0.0) -> List[bool]:
    """Per word, in order: did the streamer say it. A word is kept when its
    best frame reaches KEEP_AT, unless it sits in a lone short burst (the
    LONE_* rule above). A word with unreadable times, or one the map does not
    cover, is kept and counts as the streamer speaking. ``words`` are read in
    time order, as every transcript here stores them."""
    words = list(words or [])
    if voice is None or not voice.active:
        return [True] * len(words)
    spans: List[Optional[Tuple[float, float]]] = []
    scores: List[float] = []
    for word in words:
        try:
            start = offset + float(word.get("start", 0.0))
            end = offset + float(word.get("end", 0.0))
        except (TypeError, ValueError, AttributeError):
            spans.append(None)
            scores.append(1.0)
            continue
        score = voice.word_score(start, end)
        spans.append((start, max(end, start)))
        scores.append(1.0 if score is None else score)
    mask = [span is None or score >= KEEP_AT for span, score in zip(spans, scores)]

    runs: List[List[int]] = []
    current: List[int] = []
    for i, kept in enumerate(mask):
        if not kept or spans[i] is None:
            if current:
                runs.append(current)
            current = []
            continue
        if current and spans[i][0] - spans[current[-1]][1] >= LONE_RUN_GAP_SEC:
            runs.append(current)
            current = []
        current.append(i)
    if current:
        runs.append(current)

    kept_before = list(mask)  # the rule reads the plain per-word answer
    for run in runs:
        first, last = spans[run[0]], spans[run[-1]]
        if len(run) > LONE_MAX_WORDS or last[1] - first[0] > LONE_MAX_SEC:
            continue
        if max(scores[i] for i in run) >= LONE_SURE_AT:
            continue
        members = set(run)
        company = any(
            kept_before[j] and j not in members and (
                spans[j] is None
                or (spans[j][1] > first[0] - LONE_QUIET_SEC and spans[j][0] < last[1] + LONE_QUIET_SEC))
            for j in range(max(0, run[0] - 64), min(len(words), run[-1] + 65))
        )
        if not company:
            for i in run:
                mask[i] = False
    return mask


def filter_words(words: Sequence[dict], voice: Optional[StreamerVoice],
                 offset: float = 0.0) -> List[dict]:
    """The words the streamer said. ``offset`` maps word times to VOD time:
    the clip start for a clip-relative decode, 0 for the scan's transcript.
    With no map, or a VOD the gate leaves alone, every word comes back."""
    words = list(words or [])
    return [word for word, kept in zip(words, keep_mask(words, voice, offset)) if kept]


def filter_transcript(transcript: Optional[dict], voice: Optional[StreamerVoice],
                      offset: float = 0.0) -> Optional[dict]:
    """A copy of ``transcript`` holding only the streamer's words.

    Per WORD, not per phrase: the phrase rule (dominant_voice.filter_clip_
    transcript) removed 81.1% of teammate words for 1.8% of the streamer's on
    the same decodes. A segment left with no words is dropped; one that lost
    some has its text rebuilt from what is left, so nothing downstream can
    read a dropped word back out of it. The input is not modified.
    """
    if transcript is None or voice is None or not voice.active:
        return transcript
    source = transcript.get("segments", []) or []
    # One pass over every word, so a burst is judged against its neighbours in
    # the next segment too, not only its own.
    flat = [word for segment in source for word in (segment.get("words") or [])]
    mask = iter(keep_mask(flat, voice, offset))
    segments = []
    dropped = 0
    for segment in source:
        words = segment.get("words") or []
        if not words:
            segments.append(segment)
            continue
        kept = [word for word in words if next(mask)]
        dropped += len(words) - len(kept)
        if not kept:
            continue
        if len(kept) == len(words):
            segments.append(segment)
            continue
        trimmed = dict(segment)
        trimmed["words"] = kept
        trimmed["text"] = " ".join(str(w.get("word", "")).strip() for w in kept).strip()
        trimmed["start"] = float(kept[0].get("start", segment.get("start", 0.0)))
        trimmed["end"] = float(kept[-1].get("end", segment.get("end", 0.0)))
        segments.append(trimmed)
    filtered = dict(transcript)
    filtered["segments"] = segments
    filtered["streamer_voice_dropped"] = int(transcript.get("streamer_voice_dropped", 0)) + dropped
    return filtered


def window_transcript(transcript: dict, start: float, end: float,
                      voice: Optional[StreamerVoice]) -> Tuple[dict, int]:
    """(one-segment VOD-timed transcript, words heard) for [start, end] of the
    scan's transcript: the words asr.slice_transcript_words would take, minus
    those the streamer did not say. Scored at their own VOD times, not at the
    clip-relative times slicing clamps to the window."""
    heard = []
    for segment in (transcript or {}).get("segments", []) or []:
        for word in segment.get("words") or []:
            try:
                ws, we = float(word.get("start", 0.0)), float(word.get("end", 0.0))
            except (TypeError, ValueError):
                continue
            if we < start or ws > end:
                continue
            heard.append(word)
    return {"segments": [{"words": filter_words(heard, voice, 0.0)}]}, len(heard)


# ---------------------------------------------------------------- building


def wav_logmel(pcm: np.ndarray, cancel_check: Optional[Callable[[], None]] = None,
               block_sec: float = LOGMEL_BLOCK_SEC) -> np.ndarray:
    """The detector's 48-band log-mel of 16 kHz int16 ``pcm``, block by block.

    float16, as the research script stored and trained on it (34.6 MB per
    VOD-hour as float32, half that here). Equal to one pass over all samples:
    each block is cut on the frame grid.
    """
    lm = sa.LogMel(SAMPLE_RATE)
    n = lm.frame_count(len(pcm))
    out = np.empty((n, sa.N_MELS), np.float16)
    per = max(1, int(round(block_sec / sa.FRAME_SEC)))
    for f0 in range(0, n, per):
        if cancel_check:
            cancel_check()
        f1 = min(n, f0 + per)
        block = np.asarray(pcm[f0 * lm.hop:(f1 - 1) * lm.hop + lm.win], dtype=np.float32) / 32768.0
        out[f0:f1] = lm(block)
    return out


def _probability_bytes(p: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(p, np.float64) * _LEVELS), 0, _LEVELS).astype(np.uint8)


def _train_on_this_vod(logmel, regions, starts, ends, sims, seed=0):
    """(probabilities, mlp, weak-label counts) of the VOD's own detector, or
    (None, None, counts) when the weak labels hold only one class."""
    n = len(logmel)
    speech = sa.frames_in_spans(n, [(float(a), float(b)) for a, b in regions])
    min_sim, max_sim = sa.window_extremes(n, starts, ends, sims)
    labels = sa.weak_labels(speech, min_sim, max_sim, STREAMER_AT, OTHER_BELOW)
    counts = {"weak_streamer": int((labels == 1).sum()),
              "weak_other_speech": int(((labels == 0) & speech).sum()),
              "weak_game": int(((labels == 0) & ~speech).sum())}
    if counts["weak_streamer"] == 0 or (labels == 0).sum() == 0:
        return None, None, counts
    detector = sa.train_detector(logmel, labels, seed=seed)
    return detector.raw_probability(logmel), detector.mlp, counts


def _profile_probability(logmel, mlps: Sequence[sa.MLP]) -> np.ndarray:
    """Mean probability of the profile's detectors over this VOD, each under
    THIS VOD's normaliser (how the research's pooled model transferred)."""
    norm = sa.fit_normalizer(logmel, False)
    total = None
    for mlp in mlps:
        p = sa.Detector(norm, mlp, False).raw_probability(logmel)
        total = p.astype(np.float64) if total is None else total + p
    return (total / len(mlps)).astype(np.float32)


def build_streamer_voice(wav_path: str, regions: Sequence[Tuple[float, float]], *,
                         embed=None, cancel_check: Optional[Callable[[], None]] = None,
                         profile=None, seed: int = 0) -> Optional[StreamerVoice]:
    """Build one VOD's map from its 16 kHz analysis WAV and VAD speech regions.

    ``profile`` (voice_profile.VoiceProfile or None) is the streamer's stored
    voice. None when the WAV is unreadable or nothing could be embedded;
    otherwise a StreamerVoice whose gate says whether and why it filters.

    MEASURED 2026-11-15 on five VODs (2.1-4.3 h), one core: 54-106 CPU-s per
    VOD-hour in all, nearly all of it the embeddings (~54 ms per 1.5 s window,
    1,400-2,300 windows per VOD-hour); log-mel, training and inference of a
    4.3-hour VOD 6 s + 9 s wall (16.8 s CPU with numpy's default BLAS pool).
    Memory: the WAV is memory-mapped (its pages are file-backed and count
    toward the working set, not the commit); 352 MB peak of allocations for
    the 4.3-hour VOD, the float16 log-mel (17 MB per VOD-hour) and the
    training rows (~180 MB, fixed) included.
    """
    started = time.perf_counter()
    pcm = dv._pcm16_data(wav_path)
    if pcm is None:
        return None
    try:
        embedded = dv.embed_windows(pcm, regions, embed, cancel_check)
        if embedded is None:
            return None
        starts, ends, matrix = embedded
        centroid = dv.dominant_centroid(matrix)
        if centroid is None:
            return None
        sims = (matrix @ centroid).astype(np.float32)
        vmap = dv.VoiceMap(starts, ends, sims, centroid.astype(np.float32))
        far, near = vmap.fractions()
        full = int(((ends - starts) >= dv.WINDOW_SEC - 1e-3).sum())
        gate: Dict[str, Any] = {
            "active": False, "reason": "", "source": None,
            "far": far, "near": near, "windows": int(len(starts)), "full_windows": full,
            "profile_cosine": None,
        }
        embed_seconds = time.perf_counter() - started
        probs, mlp = None, None
        profile_centroid = getattr(profile, "centroid", None)
        cosine = None if profile_centroid is None else float(centroid @ profile_centroid)
        gate["profile_cosine"] = cosine
        if cosine is not None and cosine < PROFILE_AGREE:
            # This VOD's dominant voice is not the streamer's (see
            # PROFILE_AGREE): its own weak labels would teach the detector
            # the wrong voice, so they are never used.
            pmap = dv.VoiceMap(starts, ends, (matrix @ profile_centroid).astype(np.float32),
                               profile_centroid)
            pfar, pnear = pmap.fractions()
            gate.update(profile_far=pfar, profile_near=pnear)
            mlps = list(getattr(profile, "mlps", []) or [])
            if not mlps:
                gate["reason"] = "not_the_streamer_no_profile_model"
            elif pfar < dv.MIN_FAR_FRACTION:
                gate["reason"] = "not_the_streamer_one_voice"
            elif pnear < PROFILE_MIN_NEAR:
                gate["reason"] = "streamer_not_heard"
            else:
                logmel = wav_logmel(pcm, cancel_check)
                if cancel_check:
                    cancel_check()
                probs = _profile_probability(logmel, mlps)
                gate.update(active=True, reason="profile", source="profile",
                            profile_models=len(mlps))
        elif vmap.active():
            logmel = wav_logmel(pcm, cancel_check)
            if cancel_check:
                cancel_check()
            probs, mlp, counts = _train_on_this_vod(logmel, regions, starts, ends, sims, seed)
            gate.update(counts)
            if probs is None:
                gate["reason"] = "no_weak_labels"
            else:
                gate.update(active=True, reason="second_voice", source="vod")
        else:
            gate["reason"] = "one_voice" if far < dv.MIN_FAR_FRACTION else "no_dominant_voice"
        gate["embed_seconds"] = round(embed_seconds, 2)
        gate["build_seconds"] = round(time.perf_counter() - started, 2)
        return StreamerVoice(gate=gate, centroid=centroid.astype(np.float32),
                             probs=None if probs is None else _probability_bytes(probs), mlp=mlp)
    finally:
        del pcm  # the memmap holds the WAV open; Windows will not delete an open file
