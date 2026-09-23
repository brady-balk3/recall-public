# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Attribute captions to the primary speaker using local voice embeddings.

Stream audio can include narration, party chat and the creator's own speech.
Acoustic bandwidth alone does not reliably distinguish them: loudness and
encoding affect the spectrum. Voice embeddings provide a speaker similarity
signal, with a profile sampled across the whole recording.

WeSpeaker ResNet34-LM accepts 80-bin Kaldi filterbanks and produces 256-dimensional
embeddings. Missing models, profiles or audio leave the transcript unchanged.
Preserving the creator's speech takes precedence over aggressive filtering.
"""

import os
from typing import List, Optional

import numpy as np

from core.bundle_paths import get_speech_models_dir, is_frozen

SAMPLE_RATE = 16000
EMBED_MODEL_FILENAME = "voxceleb_resnet34_LM.onnx"
EMBED_MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"

# Shorter segments contain too little speech for a reliable embedding.
MIN_SEGMENT_SEC = 1.0

# Sample multiple regions: an individual clip may contain only a guest or
# narration and must not define the primary speaker's identity on its own.
PROFILE_REGIONS = 6
PROFILE_REGION_SEC = 180.0
PROFILE_MIN_SEGMENTS = 12
CLUSTER_DISTANCE = 0.55

# Normal speech and shouting can form separate clusters for the same speaker.
# Merge similar clusters before taking the centroid, subject to recurrence.
PROFILE_MERGE_MIN = 0.45

# A guest may resemble the primary speaker within one region. Require merged
# clusters to recur across the recording; similarity remains required too.
# The dominant cluster remains the anchor regardless of this merge rule.
PROFILE_MIN_REGION_SHARE = 0.5

# Prefer conservative rejection: dropping the creator's reaction is worse than
# retaining a guest line. These heuristics still need broader quality validation.
STREAMER_MIN_SIMILARITY = 0.35

# Transcript segments can cross speaker turns. Overlapping windows allow each
# word to use a nearby score rather than assigning one speaker to the segment.
GRID_WINDOW_SEC = 1.5
GRID_HOP_SEC = 0.5
# Smooth isolated noisy scores without erasing sustained speaker changes.
GRID_SMOOTH = 3

# Restore very short rejected runs surrounded by accepted speech. Count words
# because aligner timestamps can stretch a single word across a long pause.
MIN_TURN_WORDS = 2

_SESSION = None
_SESSION_TRIED = False


def _model_path() -> Optional[str]:
    """Bundled model first; fall back to the HF cache in dev.

    A packaged build must ship the weights -- see the bundle_paths note on
    faster-whisper. Downloading 26MB mid-scan on a user's machine (or failing to,
    offline) is not an acceptable runtime behaviour.
    """
    bundled = os.path.join(get_speech_models_dir(), "speaker", EMBED_MODEL_FILENAME)
    if os.path.isfile(bundled):
        return bundled
    managed = bool(os.environ.get("RECALL_SPEECH_SNAPSHOT", "")) and not os.environ.get("RECALL_MODELS_DIR", "").strip()
    if is_frozen() or managed:
        return None
    try:
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from core.model_catalog import SPEAKER_REVISION, SPEAKER_FILES
        from core.model_downloads import matches
        cached = hf_hub_download(EMBED_MODEL_REPO, EMBED_MODEL_FILENAME,
                                 revision=SPEAKER_REVISION, token=False)
        return cached if matches(Path(cached), SPEAKER_FILES[EMBED_MODEL_FILENAME]) else None
    except Exception:  # noqa: BLE001 - attribution is optional, never fatal
        return None


def _session():
    global _SESSION, _SESSION_TRIED
    if _SESSION is None and not _SESSION_TRIED:
        _SESSION_TRIED = True
        try:
            import onnxruntime as ort
            path = _model_path()
            if path:
                _SESSION = ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            print(f"Speaker attribution unavailable: {exc}")
    return _SESSION


def embed(samples: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalised 256-d speaker embedding for one mono 16k window."""
    session = _session()
    if session is None or samples.size < int(SAMPLE_RATE * MIN_SEGMENT_SEC):
        return None
    try:
        import torch
        from engines.caption.kaldi_fbank import fbank
        feats = fbank(
            torch.from_numpy(samples).unsqueeze(0),
            num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
            sample_frequency=SAMPLE_RATE, energy_floor=0.0,
            window_type="hamming", htk_compat=True, use_energy=False,
        )
        feats = feats - feats.mean(dim=0, keepdim=True)  # CMN, as WeSpeaker does
        out = session.run(None, {"feats": feats.unsqueeze(0).numpy()})[0][0]
    except Exception:  # noqa: BLE001 - one bad window must not stop captioning
        return None
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else None


def _segment_windows(transcript: dict):
    for segment in transcript.get("segments", []) or []:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        if end - start >= MIN_SEGMENT_SEC:
            yield segment, start, end


def wav_reader(path: str):
    """read_window over one 16k mono wav, indexed by time WITHIN that file."""
    import wave
    try:
        with wave.open(path, "rb") as handle:
            if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
                return lambda _s, _e: None
            pcm = np.frombuffer(
                handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
    except Exception:  # noqa: BLE001
        return lambda _s, _e: None

    def read(start, end):
        window = pcm[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        return window if window.size else None
    return read


def profile_for_vod(video_path: str, transcript: dict, temp_dir: str,
                    duration: float = 0.0, cancel_check=None) -> Optional[np.ndarray]:
    """Build the streamer profile from a few regions spread across the VOD.

    Extracts PROFILE_REGIONS short wavs (a handful of ffmpeg seeks, not one per
    segment), reads the already-computed VOD transcript over them, and cleans up.
    Returns None on any failure -- attribution is an enhancement, never a gate.
    """
    from engines.caption.whisper_asr import extract_clip_audio

    if _session() is None or not video_path or not temp_dir:
        return None
    if duration <= 0:
        duration = max((float(s.get("end", 0.0))
                        for s in transcript.get("segments", []) or []), default=0.0)
    if duration <= 0:
        return None

    span = duration / max(1, PROFILE_REGIONS)
    readers, written = [], []
    try:
        for index in range(PROFILE_REGIONS):
            if cancel_check:
                cancel_check()
            lo = index * span
            hi = min(duration, lo + PROFILE_REGION_SEC)
            if hi - lo < MIN_SEGMENT_SEC:
                continue
            path = os.path.join(temp_dir, f"spkprofile_{index}.wav")
            try:
                extract_clip_audio(video_path, lo, hi, path)
            except Exception:  # noqa: BLE001 - a missing region just shrinks the sample
                continue
            written.append(path)
            readers.append((lo, hi, wav_reader(path)))

        def read_window(start, end):
            for lo, hi, reader in readers:
                if lo <= start and end <= hi:
                    return reader(start - lo, end - lo)
            return None

        return build_profile(read_window, transcript, duration)
    finally:
        for path in written:
            try:
                os.remove(path)
            except OSError:
                pass


def similarity(profile, vector) -> float:
    """Closeness to the streamer's NEAREST voice (see build_profile)."""
    return float(np.max(np.asarray(profile) @ vector))


def _average_linkage_labels(matrix: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Small dependency-free fallback for cosine average-linkage clustering.

    The packaged engine does not ship scikit-learn. Treating that import
    failure as one giant cluster blends guests into the streamer profile, so
    reproduce the clustering rule for the modest profile sample instead.
    """
    count = len(matrix)
    pairwise = 1.0 - np.clip(matrix @ matrix.T, -1.0, 1.0)
    clusters = {index: [index] for index in range(count)}
    sizes = {index: 1 for index in range(count)}
    distances = {
        (left, right): float(pairwise[left, right])
        for left in range(count - 1)
        for right in range(left + 1, count)
    }

    while len(clusters) > 1:
        (left, right), best_distance = min(distances.items(), key=lambda item: item[1])
        if best_distance > distance_threshold:
            break

        left_size = sizes[left]
        right_size = sizes[right]
        other_labels = [label for label in clusters if label not in (left, right)]
        updated = {}
        for other in other_labels:
            left_key = (min(left, other), max(left, other))
            right_key = (min(right, other), max(right, other))
            updated[(min(left, other), max(left, other))] = (
                left_size * distances[left_key] + right_size * distances[right_key]
            ) / (left_size + right_size)

        clusters[left].extend(clusters.pop(right))
        sizes[left] = left_size + right_size
        sizes.pop(right)
        distances = {
            pair: distance
            for pair, distance in distances.items()
            if right not in pair and left not in pair
        }
        distances.update(updated)

    labels = np.empty(count, dtype=int)
    for label, members in enumerate(clusters.values()):
        labels[members] = label
    return labels


def build_profile(read_window, transcript: dict, duration: float) -> Optional[np.ndarray]:
    """Streamer voice centroids, or None when the VOD gives too little to go on.

    ``read_window(start, end)`` returns mono 16k float samples for a VOD-time
    span, so this stays independent of how the caller gets audio.
    """
    if _session() is None or duration <= 0:
        return None
    regions = []
    span = duration / max(1, PROFILE_REGIONS)
    for index in range(PROFILE_REGIONS):
        lo = index * span
        regions.append((lo, min(duration, lo + PROFILE_REGION_SEC)))

    embeddings, weights, region_of = [], [], []
    for segment, start, end in _segment_windows(transcript):
        index = next((i for i, (lo, hi) in enumerate(regions)
                      if lo <= start and end <= hi), None)
        if index is None:
            continue
        samples = read_window(start, end)
        if samples is None:
            continue
        vector = embed(samples)
        if vector is not None:
            embeddings.append(vector)
            weights.append(end - start)
            region_of.append(index)
    if len(embeddings) < PROFILE_MIN_SEGMENTS:
        return None

    matrix = np.stack(embeddings)
    weights = np.asarray(weights)
    region_of = np.asarray(region_of)
    min_regions = max(1, int(round(len(regions) * PROFILE_MIN_REGION_SHARE)))
    try:
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=CLUSTER_DISTANCE,
            metric="cosine", linkage="average").fit_predict(matrix)
    except Exception:  # noqa: BLE001 - packaged builds omit scikit-learn
        labels = _average_linkage_labels(matrix, CLUSTER_DISTANCE)

    def centroid(mask):
        vector = matrix[mask].mean(axis=0)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else None

    speech = {label: float(weights[labels == label].sum()) for label in set(labels)}
    dominant = max(speech, key=speech.get)
    anchor = centroid(labels == dominant)
    if anchor is None:
        return None

    # One centroid PER streamer voice, not one blended centroid over all of
    # them. Averaging his shouting voice into his speaking voice pulls the
    # result away from both: measured on channel_a 8/9 it left only 0.012 between
    # the quietest shouted line and the loudest narrator line. Scoring against
    # the nearest voice keeps each one sharp and widens that margin.
    profile = [anchor]
    for label in speech:
        if label == dominant:
            continue
        mask = labels == label
        if len(set(region_of[mask])) < min_regions:
            continue  # owns a stretch, not the stream -- see PROFILE_MIN_REGION_SHARE
        member = centroid(mask)
        if member is not None and float(anchor @ member) >= PROFILE_MERGE_MIN:
            profile.append(member)
    return np.stack(profile)


def segment_similarity(read_window, transcript: dict, profile) -> List[float]:
    """Per-segment cosine similarity to the profile; NaN where unmeasurable."""
    scores = []
    for segment, start, end in _segment_windows(transcript):
        samples = read_window(start, end)
        vector = embed(samples) if samples is not None else None
        scores.append(similarity(profile, vector) if vector is not None else float("nan"))
    return scores


def _grid(read_window, duration: float, profile):
    """(centre, smoothed similarity) across the clip; empty when unusable."""
    centres, scores = [], []
    centre = GRID_HOP_SEC / 2.0
    while centre < duration:
        lo = max(0.0, centre - GRID_WINDOW_SEC / 2.0)
        hi = min(duration, lo + GRID_WINDOW_SEC)
        samples = read_window(lo, hi) if hi - lo >= MIN_SEGMENT_SEC else None
        vector = embed(samples) if samples is not None else None
        centres.append(centre)
        scores.append(similarity(profile, vector) if vector is not None else float("nan"))
        centre += GRID_HOP_SEC
    if not centres:
        return [], []

    smoothed = []
    half = max(0, GRID_SMOOTH // 2)
    for index in range(len(scores)):
        window = [s for s in scores[max(0, index - half):index + half + 1]
                  if not np.isnan(s)]
        smoothed.append(float(np.median(window)) if window else float("nan"))
    return centres, smoothed


def filter_words_to_streamer(transcript: dict, read_window, profile, duration: float,
                             threshold: float = STREAMER_MIN_SIMILARITY) -> dict:
    """Drop individual WORDS spoken by someone other than the streamer.

    Each word reads the smoothed grid score nearest its midpoint. Words with no
    usable score are kept, as is a segment that keeps any word; a segment left
    with nothing is dropped. Timestamps are clip-relative, matching the decode.
    """
    if profile is None or _session() is None or duration <= 0:
        return transcript
    centres, scores = _grid(read_window, duration, profile)
    if not centres:
        return transcript
    centres = np.asarray(centres)

    def keeps(word) -> bool:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        score = scores[int(np.argmin(np.abs(centres - (start + end) / 2.0)))]
        return np.isnan(score) or score >= threshold

    def restore_short_holes(flags):
        """Re-keep a cut run under MIN_TURN_WORDS that has speech on both sides."""
        index = 0
        while index < len(flags):
            if flags[index]:
                index += 1
                continue
            run_end = index
            while run_end < len(flags) and not flags[run_end]:
                run_end += 1
            bracketed = index > 0 and run_end < len(flags)
            if bracketed and (run_end - index) < MIN_TURN_WORDS:
                for fill in range(index, run_end):
                    flags[fill] = True
            index = run_end
        return flags

    # Hole-filling runs over the clip's whole word sequence, not per segment: a
    # wobble lands on a segment boundary as often as inside one, and a word at
    # the end of a segment has its bracketing neighbour in the next.
    all_words = [word for segment in transcript.get("segments", []) or []
                 for word in (segment.get("words") or [])]
    verdict = dict(zip(
        (id(word) for word in all_words),
        restore_short_holes([keeps(word) for word in all_words]),
    ))

    segments = []
    for segment in transcript.get("segments", []) or []:
        words = segment.get("words") or []
        if not words:
            segments.append(segment)   # nothing to attribute -- keep it
            continue
        kept = [word for word in words if verdict[id(word)]]
        if not kept:
            continue
        text = "".join(str(word.get("word", "")) for word in kept)
        segments.append({**segment, "words": kept, "text": text,
                         "start": float(kept[0].get("start", segment.get("start", 0.0))),
                         "end": float(kept[-1].get("end", segment.get("end", 0.0)))})
    return {**transcript, "segments": segments}


def filter_to_streamer(transcript: dict, read_window, profile,
                       threshold: float = STREAMER_MIN_SIMILARITY) -> dict:
    """Drop segments confidently spoken by someone other than the streamer.

    Fails open in every uncertain case: no profile, no model, an unreadable
    window or a segment too short to embed all keep their words.
    """
    if profile is None or _session() is None:
        return transcript
    kept = []
    for segment in transcript.get("segments", []) or []:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        vector = None
        if end - start >= MIN_SEGMENT_SEC:
            samples = read_window(start, end)
            vector = embed(samples) if samples is not None else None
        if vector is not None and similarity(profile, vector) < threshold:
            continue
        kept.append(segment)
    return {**transcript, "segments": kept}
