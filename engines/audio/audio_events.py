# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Audio-event burst detection: laughter & screams as a reaction channel.

Whisper hears words and prosody hears pitch, but laughter — the strongest
highlight signal in gaming content — is non-speech. This module runs a
YAMNet-class AudioSet tagger (ONNX, via the onnxruntime we already ship for
OCR/emotion) over the VOD's 16 kHz mono wav and emits a per-second ``burst``
track: max sigmoid score over the laughter/scream class groups, augmented by
a model-independent sudden-loudness onset for startles the class model misses.

The same pass also emits ``game_intensity``: pooled combat/action SFX and
intense-music scores. This is GAME-side evidence (boss music, gunfire,
explosions, monster roars), not a human reaction — it exists so games with no
OCR HUD vocabulary (story/horror titles) still register that the game itself
entered a high-stakes state while the streamer plays quiet and focused. One
model, one pass over the wav: both tracks come from the same class scores.

Model files (not committed; see models/README-audio-events.md):
  models/yamnet.onnx           — waveform-in AudioSet tagger (521 classes)
  models/yamnet_class_map.csv  — AudioSet class index -> display_name

If the model files are absent the channel reports unavailable and the reaction
engine simply fuses without it — same graceful degradation as face/ASR.
"""

import csv
import os
import wave
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from core.bundle_paths import get_models_dir
from core.device import get_ort_providers

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15600          # YAMNet frame: 0.975 s @ 16 kHz
HOP_SECONDS = 1.0               # score one window per integer second (aligns with other tracks)
BATCH_WINDOWS = 256             # scored per ONNX call when the model accepts batches

# Class tags miss some startles because the game sting and streamer scream are
# mixed together. A relative loudness onset is model-independent evidence of
# the abrupt acoustic edge that defines a jumpscare. It is still human-gated
# in fusion.py, so a loud game sound with no voice/face reaction cannot create
# a highlight by itself.
LOUD_ONSET_LOOKBACK_SEC = 8
LOUD_ONSET_MIN_RMS = 0.003
LOUD_ONSET_MIN_DB_RISE = 6.0
LOUD_ONSET_FULL_DB_RISE = 18.0

# AudioSet display names pooled into each burst group. Sigmoid outputs, 0..1.
LAUGH_CLASSES = {
    "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh",
    "Chuckle, chortle",
}
SCREAM_CLASSES = {
    "Screaming", "Shout", "Yell", "Whoop", "Battle cry",
}

# Game-side action/combat SFX. Human vocal classes (Bellow, Screaming, ...)
# are deliberately absent — streamer reactions belong to the burst track; this
# group must fire on what the GAME is doing. AudioSet display names.
COMBAT_CLASSES = {
    "Explosion", "Gunshot, gunfire", "Machine gun", "Fusillade",
    "Artillery fire", "Boom", "Smash, crash", "Shatter", "Bang", "Thunder",
    "Roar", "Roaring cats (lions, tigers)", "Growling",
}
# Boss/chase scoring cues. Mood subclasses only — bare "Music" fires on every
# lobby playlist and menu loop, which is exactly what this track must ignore.
INTENSE_MUSIC_CLASSES = {"Exciting music", "Angry music", "Scary music"}
# Music mood is weaker evidence than a gunshot/explosion actually landing.
INTENSE_MUSIC_WEIGHT = 0.6


@dataclass
class AudioEventFrame:
    timestamp: float
    burst: float                  # max pooled laughter/scream score, 0..1
    label: Optional[str] = None   # "laughter" | "scream" when burst > 0
    # Pooled combat-SFX / intense-music score, 0..1. Default 0.0 keeps caches
    # written before this field decodable (signal_cache fills from defaults).
    game_intensity: float = 0.0
    # Relative 1-second loudness onset, 0..1. Kept separate from the semantic
    # tagger score so downstream can tell why the burst channel fired.
    loud_onset: float = 0.0

    def to_dict(self):
        return asdict(self)


_session = None
_class_names: Optional[List[str]] = None


def _model_paths():
    models = get_models_dir()
    return (
        os.path.join(models, "yamnet.onnx"),
        os.path.join(models, "yamnet_class_map.csv"),
    )


def audio_events_available() -> bool:
    onnx_path, map_path = _model_paths()
    return os.path.exists(onnx_path) and os.path.exists(map_path)


def _load():
    global _session, _class_names
    if _session is not None:
        return _session, _class_names
    import onnxruntime as ort
    onnx_path, map_path = _model_paths()
    _session = ort.InferenceSession(onnx_path, providers=get_ort_providers())
    with open(map_path, newline="", encoding="utf-8") as f:
        _class_names = [row["display_name"] for row in csv.DictReader(f)]
    return _session, _class_names


def _read_wav_mono16k(wav_path: str) -> np.ndarray:
    """Load the pipeline's cached wav (already 16 kHz mono PCM16) as float32 -1..1."""
    with wave.open(wav_path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
            raise ValueError(
                f"expected {SAMPLE_RATE} Hz mono (pipeline extract_audio output), "
                f"got {w.getframerate()} Hz x{w.getnchannels()}"
            )
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _score_windows(windows: np.ndarray) -> np.ndarray:
    """Run the tagger on a [batch, WINDOW_SAMPLES] array -> [batch, n_classes] scores.

    Handles the two common YAMNet ONNX export shapes: rank-1 dynamic waveform
    (scored one window at a time) and rank-2 [batch|1, 15600].
    """
    session, _ = _load()
    inp = session.get_inputs()[0]
    rank = len(inp.shape)
    out_name = session.get_outputs()[0].name

    scores = []
    if rank == 1:
        for row in windows:
            out = session.run([out_name], {inp.name: row.astype(np.float32)})[0]
            scores.append(np.asarray(out, dtype=np.float32).reshape(-1))
    else:
        batch_ok = not isinstance(inp.shape[0], int) or inp.shape[0] > 1
        if batch_ok:
            out = session.run([out_name], {inp.name: windows.astype(np.float32)})[0]
            arr = np.asarray(out, dtype=np.float32)
            # Some exports emit [batch, frames, classes]; pool frames by max.
            if arr.ndim == 3:
                arr = arr.max(axis=1)
            return arr
        for row in windows:
            out = session.run([out_name], {inp.name: row[None, :].astype(np.float32)})[0]
            scores.append(np.asarray(out, dtype=np.float32).reshape(-1))
    return np.stack(scores)


def _loud_onset_scores(windows: np.ndarray) -> np.ndarray:
    """Return per-window sudden-loudness scores relative to recent context."""
    if windows.size == 0:
        return np.zeros(len(windows), dtype=np.float32)
    # Avoid ``windows ** 2``: a multi-hour VOD already holds a large window
    # matrix, and materializing another full-size array can add >1 GB. einsum
    # reduces directly to one value per second.
    energy = np.einsum("ij,ij->i", windows, windows, optimize=True)
    rms = np.sqrt(energy / max(1, windows.shape[1]))
    onset = np.zeros(len(windows), dtype=np.float32)
    span = max(1, int(LOUD_ONSET_LOOKBACK_SEC / HOP_SECONDS))
    db_span = max(1e-6, LOUD_ONSET_FULL_DB_RISE - LOUD_ONSET_MIN_DB_RISE)
    for idx in range(1, len(windows)):
        current = float(rms[idx])
        if current < LOUD_ONSET_MIN_RMS:
            continue
        baseline = float(np.median(rms[max(0, idx - span):idx]))
        rise_db = 20.0 * np.log10(max(current, 1e-8) / max(baseline, 1e-8))
        onset[idx] = float(np.clip(
            (rise_db - LOUD_ONSET_MIN_DB_RISE) / db_span, 0.0, 1.0,
        ))
    return onset


def analyze_audio_events(wav_path: str, duration: Optional[float] = None) -> Optional[List[AudioEventFrame]]:
    """Per-second laughter/scream burst track for the VOD.

    Returns None when the model files aren't installed (channel disabled).
    """
    if not audio_events_available():
        return None

    _, class_names = _load()
    laugh_idx = [i for i, n in enumerate(class_names) if n in LAUGH_CLASSES]
    scream_idx = [i for i, n in enumerate(class_names) if n in SCREAM_CLASSES]
    combat_idx = [i for i, n in enumerate(class_names) if n in COMBAT_CLASSES]
    music_idx = [i for i, n in enumerate(class_names) if n in INTENSE_MUSIC_CLASSES]
    if not laugh_idx and not scream_idx:
        return None

    audio = _read_wav_mono16k(wav_path)
    total_seconds = int(duration if duration is not None else len(audio) / SAMPLE_RATE)
    if total_seconds <= 0:
        return []

    # One 0.975 s window per integer second; zero-pad the tail.
    hop = int(SAMPLE_RATE * HOP_SECONDS)
    windows = np.zeros((total_seconds, WINDOW_SAMPLES), dtype=np.float32)
    for sec in range(total_seconds):
        start = sec * hop
        chunk = audio[start:start + WINDOW_SAMPLES]
        windows[sec, :len(chunk)] = chunk
    loud_onsets = _loud_onset_scores(windows)

    frames: List[AudioEventFrame] = []
    for batch_start in range(0, total_seconds, BATCH_WINDOWS):
        batch = windows[batch_start:batch_start + BATCH_WINDOWS]
        scores = _score_windows(batch)
        laugh = scores[:, laugh_idx].max(axis=1) if laugh_idx else np.zeros(len(batch))
        scream = scores[:, scream_idx].max(axis=1) if scream_idx else np.zeros(len(batch))
        combat = scores[:, combat_idx].max(axis=1) if combat_idx else np.zeros(len(batch))
        music = scores[:, music_idx].max(axis=1) if music_idx else np.zeros(len(batch))
        for j in range(len(batch)):
            burst = float(max(laugh[j], scream[j]))
            label = None
            if burst > 0.05:
                label = "laughter" if laugh[j] >= scream[j] else "scream"
            frames.append(AudioEventFrame(
                timestamp=float(batch_start + j),
                burst=burst,
                label=label,
                game_intensity=float(max(combat[j], INTENSE_MUSIC_WEIGHT * music[j])),
                loud_onset=float(loud_onsets[batch_start + j]),
            ))
    return frames
