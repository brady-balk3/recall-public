# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Voice Prosody Engine (plan §5.2).

Turns the already-extracted 16 kHz mono ``audio.wav`` into a per-second
``voice_arousal`` track that separates *genuine hype* (rising pitch, energy
bursts, fast speech) from steady-state loudness (music, mic gain) — the thing
the existing RMS spike detector in ``voice_spike.py`` cannot do.

Packaging note (plan §13.2): F0 is estimated with a numpy/scipy autocorrelation
pitch tracker rather than ``librosa.pyin`` so we do NOT pull in librosa/numba,
keeping the frozen backend small. Everything here is numpy + scipy, which are
already in the tree.
"""

import wave
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

# Human voiced-speech fundamental frequency range. Lags outside this band are
# ignored by the autocorrelation pitch search.
F0_MIN_HZ = 75.0
F0_MAX_HZ = 400.0

# Autocorrelation peak must clear this fraction of the zero-lag energy for the
# window to count as voiced; otherwise F0 is reported as 0 (silence / noise).
VOICING_THRESHOLD = 0.30


@dataclass
class ProsodyFrame:
    timestamp: float
    rms: float            # short-term energy (RMS) of the window
    f0: float             # estimated fundamental frequency in Hz (0 if unvoiced)
    voiced: bool          # whether the window was classified as voiced speech
    zcr: float            # zero-crossing rate (speech-rate / fricative proxy)
    energy_var: float     # variance of sub-window RMS (burstiness)
    voice_arousal: float  # 0..1 fused prosodic excitement (per-VOD normalized)

    def to_dict(self):
        return asdict(self)


def _read_wav_mono(wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, sample_rate


def _estimate_f0(window: np.ndarray, sample_rate: int) -> float:
    """Autocorrelation-based fundamental frequency estimate (Hz), 0 if unvoiced."""
    if window.size < 2:
        return 0.0

    w = window - float(np.mean(window))
    energy = float(np.dot(w, w))
    if energy <= 1e-9:
        return 0.0

    # Full autocorrelation via FFT, keep non-negative lags.
    n = w.size
    fft_size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spec = np.fft.rfft(w, fft_size)
    acf = np.fft.irfft(spec * np.conj(spec), fft_size)[:n]

    min_lag = max(1, int(sample_rate / F0_MAX_HZ))
    max_lag = min(n - 1, int(sample_rate / F0_MIN_HZ))
    if max_lag <= min_lag:
        return 0.0

    search = acf[min_lag:max_lag + 1]
    best_idx = int(np.argmax(search))
    best_lag = min_lag + best_idx
    # Normalize the peak against the zero-lag (total energy) for a voicing decision.
    if acf[0] <= 0 or (search[best_idx] / acf[0]) < VOICING_THRESHOLD:
        return 0.0
    return float(sample_rate) / float(best_lag)


def _robust_unit(values: np.ndarray) -> np.ndarray:
    """Map raw values to ~0..1 via positive robust-z (median/MAD), squashed.

    Negative deviations (below baseline) collapse toward 0; strong positive
    deviations approach 1. This is an internal convenience track; the Reaction
    Engine re-normalizes per-VOD when fusing, so exact scaling is not critical.
    """
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1e-6, mad * 1.4826)
    z = (values - median) / scale
    z = np.clip(z, 0.0, 6.0)          # only excitement above baseline matters
    return 1.0 - np.exp(-z / 2.0)     # smooth squash into 0..1


def extract_prosody(wav_path: str, window_duration_sec: float = 1.0) -> List[ProsodyFrame]:
    """Compute a per-window prosody track from a mono wav file."""
    samples, sample_rate = _read_wav_mono(wav_path)
    if samples.size == 0:
        return []

    win = max(1, int(sample_rate * window_duration_sec))
    sub = max(1, win // 8)  # sub-windows for energy-variance / burstiness

    timestamps: List[float] = []
    rms_vals: List[float] = []
    f0_vals: List[float] = []
    zcr_vals: List[float] = []
    evar_vals: List[float] = []

    for i in range(0, samples.size, win):
        window = samples[i:i + win]
        if window.size == 0:
            break

        rms = float(np.sqrt(np.mean(window ** 2)))
        f0 = _estimate_f0(window, sample_rate)
        # Zero-crossing rate normalized to crossings-per-sample.
        zcr = float(np.mean(np.abs(np.diff(np.sign(window))) > 0)) if window.size > 1 else 0.0
        sub_rms = [
            float(np.sqrt(np.mean(window[j:j + sub] ** 2)))
            for j in range(0, window.size, sub)
            if window[j:j + sub].size
        ]
        energy_var = float(np.var(sub_rms)) if sub_rms else 0.0

        timestamps.append(i / sample_rate)
        rms_vals.append(rms)
        f0_vals.append(f0)
        zcr_vals.append(zcr)
        evar_vals.append(energy_var)

    rms_arr = np.array(rms_vals, dtype=np.float32)
    f0_arr = np.array(f0_vals, dtype=np.float32)
    zcr_arr = np.array(zcr_vals, dtype=np.float32)
    evar_arr = np.array(evar_vals, dtype=np.float32)

    # F0 *rise rate*: positive frame-to-frame jumps in pitch (excitement onset).
    f0_rise = np.zeros_like(f0_arr)
    if f0_arr.size > 1:
        diff = np.diff(f0_arr)
        # only count rises where both frames are voiced
        voiced_pair = (f0_arr[:-1] > 0) & (f0_arr[1:] > 0)
        f0_rise[1:] = np.where(voiced_pair, np.maximum(0.0, diff), 0.0)

    energy_u = _robust_unit(rms_arr)
    f0_rise_u = _robust_unit(f0_rise)
    rate_u = _robust_unit(zcr_arr)
    burst_u = _robust_unit(evar_arr)

    # Blend: energy and pitch-rise dominate; speech rate and burstiness assist.
    arousal = 0.45 * energy_u + 0.30 * f0_rise_u + 0.15 * rate_u + 0.10 * burst_u
    arousal = np.clip(arousal, 0.0, 1.0)

    frames: List[ProsodyFrame] = []
    for idx, ts in enumerate(timestamps):
        frames.append(ProsodyFrame(
            timestamp=ts,
            rms=float(rms_arr[idx]),
            f0=float(f0_arr[idx]),
            voiced=bool(f0_arr[idx] > 0),
            zcr=float(zcr_arr[idx]),
            energy_var=float(evar_arr[idx]),
            voice_arousal=float(arousal[idx]),
        ))
    return frames
