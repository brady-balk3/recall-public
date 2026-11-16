# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Is the streamer speaking right now? A per-VOD frame detector we train ourselves.

Wired in through engines.caption.streamer_voice (the mel-only detector at
16 kHz, no synthetic overlap); see scripts/research_streamer_activity.py for
every measurement.

Every 20 ms the detector gives the probability that the STREAMER is speaking,
true during overlap too. It is trained per VOD, with no labels, from the VOD's
own audio:

* weak labels from the dominant-voice map (engines.caption.dominant_voice):
  speech covered only by windows that score >= 0.45 against the VOD's
  dominant voice is "streamer", speech covered only by windows under 0.10 is
  "someone else", and audio 0.3 s or more from any speech is "game";
* optional synthetic overlap (``extra`` in train_detector): confident-streamer
  audio mixed with confident-other audio, labelled "streamer";
* features: a 48-band log-mel of the mid (L+R) signal, normalised per VOD,
  summarised over four time spans (see context_features);
* model: a one-hidden-layer MLP (64 units, 12.5k weights), trained and run in
  numpy (no torch).

MEASURED 2026-11-13 against Fortnite HUD voice icons on five squad VODs of one
streamer (scripts/research_streamer_activity.py): frame AUC streamer-only vs
teammate-only 0.989 (dominant-voice windows 0.969), both-talking vs
teammate-only 0.963 (0.929). On the one labelled VOD with word timings, at 80%
of teammate words removed it loses 0.72% of the streamer's words where the
phrase rule loses 1.90%. Synthetic overlap and the stereo/pitch cues gave no
measurable gain; 16 kHz mono input does as well as 44.1 kHz stereo. Cost: log-mel
~4 CPU-s, training ~1.3 CPU-s, inference ~0.6 CPU-s per VOD-hour.

Everything here is pure numpy; decoding the audio is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 44100
HOP = 882            # 20 ms
WIN = 1764           # 40 ms
NFFT = 2048
FRAME_SEC = HOP / SAMPLE_RATE
PITCH_NFFT = 4096    # >= 2 * WIN, so the autocorrelation is not circular
PITCH_FMIN, PITCH_FMAX = 60.0, 400.0
PITCH_BAND_HZ = 4000.0
YIN_THRESHOLD = 0.2
N_MELS = 48
MEL_FMIN, MEL_FMAX = 60.0, 20000.0
CLIP_LEVEL = 0.989

# Band edges (Hz) for the stereo and bandwidth cues; the speech reference band
# for the energy ratios is 300-3400 Hz.
CUE_BANDS = ((100, 1000), (1000, 4000), (4000, 6000), (6000, 8000),
             (8000, 11000), (11000, 14000), (14000, 20000))
SPEECH_BAND = (300, 3400)


def frame_time(index) -> np.ndarray:
    """Centre time (s) of frame ``index`` relative to the first sample."""
    return (np.asarray(index, dtype=np.float64) * HOP + WIN / 2) / SAMPLE_RATE


def frame_count(n_samples: int) -> int:
    return max(0, (int(n_samples) - WIN) // HOP + 1)


def cue_names() -> List[str]:
    names = ["level_db", "side_mid_db", "ild_db", "corr"]
    for lo, hi in CUE_BANDS:
        tag = f"{lo // 1000}-{hi // 1000}k" if lo >= 1000 else f"{lo}-{hi}"
        names += [f"side_mid_db_{tag}", f"ild_db_{tag}", f"coh_{tag}", f"ratio_db_{tag}"]
    names += ["rolloff95_hz", "rolloff99_hz", "bw50_hz", "centroid_hz", "flatness",
              "f0_hz", "voicing", "clip_frac"]
    return names


def _mel_filters(n_fft: int, sr: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bank = np.zeros((n_mels, len(freqs)), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rise = (freqs - lo) / max(mid - lo, 1e-9)
        fall = (hi - freqs) / max(hi - mid, 1e-9)
        bank[i] = np.clip(np.minimum(rise, fall), 0.0, None)
        if not bank[i].any():  # narrower than one bin: take the nearest bin
            bank[i, int(np.argmin(np.abs(freqs - mid)))] = 1.0
    return bank.astype(np.float32)


class LogMel:
    """The detector's log-mel at any sample rate, on the same 20 ms / 40 ms grid.

    At 44.1 kHz it equals FeatureExtractor's log-mel; at 16 kHz (the scan's
    analysis WAV) the bands stop below 8 kHz, so a model must be trained and
    run at one rate.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = int(sample_rate)
        self.hop = int(round(FRAME_SEC * self.sample_rate))
        self.win = 2 * self.hop
        self.nfft = 1 << (self.win - 1).bit_length()
        self.window = np.hanning(self.win + 2)[1:-1].astype(np.float32)
        fmax = min(MEL_FMAX, 0.475 * self.sample_rate)
        self.mel = _mel_filters(self.nfft, self.sample_rate, N_MELS, MEL_FMIN, fmax)

    def frame_count(self, n_samples: int) -> int:
        return max(0, (int(n_samples) - self.win) // self.hop + 1)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """[frames, N_MELS] log10 mel power of the mid channel ([samples] or [samples, 2])."""
        audio = np.asarray(audio, dtype=np.float32)
        mono = audio.mean(axis=1) if audio.ndim == 2 else audio
        n = self.frame_count(len(mono))
        if n == 0:
            return np.zeros((0, N_MELS), np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(mono, self.win)[::self.hop][:n]
        spec = np.fft.rfft(frames * self.window, self.nfft)
        power = spec.real ** 2 + spec.imag ** 2
        return np.log10(power @ self.mel.T + 1e-10).astype(np.float32)


class FeatureExtractor:
    """Per-frame log-mel and physical cues from 44.1 kHz stereo float audio."""

    def __init__(self):
        self._logmel = LogMel(SAMPLE_RATE)
        self.window = np.hanning(WIN + 2)[1:-1].astype(np.float32)
        self.freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
        self.mel = _mel_filters(NFFT, SAMPLE_RATE, N_MELS, MEL_FMIN, MEL_FMAX)
        self.bands = [(self.freqs >= lo) & (self.freqs < hi) for lo, hi in CUE_BANDS]
        self.speech = (self.freqs >= SPEECH_BAND[0]) & (self.freqs < SPEECH_BAND[1])
        pitch_freqs = np.fft.rfftfreq(PITCH_NFFT, 1.0 / SAMPLE_RATE)
        self.pitch_band = pitch_freqs <= PITCH_BAND_HZ
        wacf = np.fft.irfft(np.abs(np.fft.rfft(self.window, PITCH_NFFT)) ** 2, PITCH_NFFT)
        self.lag_lo = int(SAMPLE_RATE / PITCH_FMAX)
        self.lag_hi = int(SAMPLE_RATE / PITCH_FMIN)
        self.wacf = (wacf[:self.lag_hi + 2] / wacf[0]).astype(np.float64)
        self.flat_band = (self.freqs >= 300) & (self.freqs < 4000)

    def logmel(self, audio: np.ndarray) -> np.ndarray:
        """Only the log-mel of the mid channel ([samples] mono or [samples, 2]):
        all the shipped (mel-only) detector reads, at a fraction of the cost."""
        return self._logmel(audio)

    def __call__(self, stereo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(logmel [n, N_MELS], cues [n, len(cue_names())]) for ``stereo`` [samples, 2]."""
        stereo = np.asarray(stereo, dtype=np.float32)
        if stereo.ndim == 1:
            stereo = np.stack([stereo, stereo], axis=1)
        n = frame_count(len(stereo))
        n_cues = len(cue_names())
        if n == 0:
            return np.zeros((0, N_MELS), np.float32), np.zeros((0, n_cues), np.float32)
        view = np.lib.stride_tricks.sliding_window_view
        left = view(stereo[:, 0], WIN)[::HOP][:n]
        right = view(stereo[:, 1], WIN)[::HOP][:n]
        clip = ((np.abs(left) >= CLIP_LEVEL) | (np.abs(right) >= CLIP_LEVEL)).mean(axis=1)
        L = np.fft.rfft(left * self.window, NFFT)
        R = np.fft.rfft(right * self.window, NFFT)
        pl, pr = (L.real ** 2 + L.imag ** 2), (R.real ** 2 + R.imag ** 2)
        cross = L * np.conj(R)
        pm = np.abs(L + R) ** 2 / 4.0
        ps = np.abs(L - R) ** 2 / 4.0
        eps = 1e-12
        cues = np.zeros((n, n_cues), dtype=np.float64)

        def db(x):
            return 10.0 * np.log10(x + eps)

        tm, ts = pm.sum(1), ps.sum(1)
        tl, tr = pl.sum(1), pr.sum(1)
        cues[:, 0] = db(tm / NFFT)
        cues[:, 1] = db(ts) - db(tm)
        cues[:, 2] = db(tl) - db(tr)
        cues[:, 3] = cross.real.sum(1) / np.sqrt(tl * tr + eps)
        speech_energy = pm[:, self.speech].sum(1)
        col = 4
        for band in self.bands:
            bm, bs = pm[:, band].sum(1), ps[:, band].sum(1)
            bl, br = pl[:, band].sum(1), pr[:, band].sum(1)
            bx = cross[:, band].sum(1)
            cues[:, col] = db(bs) - db(bm)
            cues[:, col + 1] = db(bl) - db(br)
            cues[:, col + 2] = (np.abs(bx) ** 2) / (bl * br + eps)
            cues[:, col + 3] = db(bm) - db(speech_energy)
            col += 4
        cum = np.cumsum(pm, axis=1)
        total = cum[:, -1:] + eps
        cues[:, col] = self.freqs[np.minimum((cum < 0.95 * total).sum(1), len(self.freqs) - 1)]
        cues[:, col + 1] = self.freqs[np.minimum((cum < 0.99 * total).sum(1), len(self.freqs) - 1)]
        # Highest frequency whose 16-bin (~345 Hz) smoothed level is within
        # 50 dB of the frame's smoothed peak: a codec's hard low-pass shows up
        # as a ceiling here.
        smooth = db(_box_mean(pm.T, -8, 7).T)
        above = smooth >= (smooth.max(1, keepdims=True) - 50.0)
        last = len(self.freqs) - 1 - np.argmax(above[:, ::-1], axis=1)
        cues[:, col + 2] = self.freqs[last]
        cues[:, col + 3] = (pm * self.freqs).sum(1) / (tm + eps)
        fb = pm[:, self.flat_band] + eps
        cues[:, col + 4] = np.exp(np.log(fb).mean(1)) / fb.mean(1)
        f0, voicing = self._pitch((left + right) * 0.5)
        cues[:, col + 5] = f0
        cues[:, col + 6] = voicing
        cues[:, col + 7] = clip
        logmel = np.log10(pm @ self.mel.T + 1e-10).astype(np.float32)
        return logmel, cues.astype(np.float32)

    def _pitch(self, mid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """YIN on the band-limited autocorrelation: (F0 Hz or 0, voicing 0-1)."""
        spec = np.fft.rfft(mid * self.window, PITCH_NFFT)
        power = spec.real ** 2 + spec.imag ** 2
        power[:, ~self.pitch_band] = 0.0
        acf = np.fft.irfft(power, PITCH_NFFT)[:, :self.lag_hi + 2]
        energy = acf[:, :1]
        ok = energy[:, 0] > 1e-9
        r = np.where(ok[:, None], acf / np.maximum(energy, 1e-12), 0.0) / self.wacf
        d = np.maximum(1.0 - r, 0.0)
        d[:, 0] = 0.0
        lags = np.arange(d.shape[1])
        csum = np.cumsum(d, axis=1)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * lags[1:] / np.maximum(csum[:, 1:], 1e-12)
        sub = cmnd[:, self.lag_lo:self.lag_hi + 1]
        below = sub < YIN_THRESHOLD
        # First dip under the threshold (then its local minimum), else the global minimum.
        first = np.where(below.any(1), np.argmax(below, axis=1), np.argmin(sub, axis=1))
        idx = first.copy()
        rows = np.arange(len(sub))
        for _ in range(sub.shape[1]):
            nxt = np.minimum(idx + 1, sub.shape[1] - 1)
            move = below.any(1) & (sub[rows, nxt] < sub[rows, idx]) & (nxt != idx)
            if not move.any():
                break
            idx = np.where(move, nxt, idx)
        value = sub[rows, idx]
        lo = sub[rows, np.maximum(idx - 1, 0)]
        hi = sub[rows, np.minimum(idx + 1, sub.shape[1] - 1)]
        denom = lo - 2 * value + hi
        shift = np.where(np.abs(denom) > 1e-12, 0.5 * (lo - hi) / np.where(np.abs(denom) > 1e-12, denom, 1.0), 0.0)
        lag = idx + self.lag_lo + np.clip(shift, -1, 1)
        voicing = np.clip(1.0 - value, 0.0, 1.0) * ok
        f0 = np.where(ok, SAMPLE_RATE / lag, 0.0)
        return f0, voicing


# ---------------------------------------------------------------- detector


# Physical cues the "mel+cues" variant appends to the log-mel: mid/side and
# coherence per band (voices are centred mono, the game is wide stereo) and a
# voicing-weighted log pitch. The log-mel already carries every band ratio.
DETECTOR_CUES = ("side_mid_db_100-1000", "side_mid_db_1-4k", "side_mid_db_4-6k", "side_mid_db_6-8k",
                 "coh_100-1000", "coh_1-4k", "coh_4-6k", "coh_6-8k", "voicing")


def detector_inputs(logmel: np.ndarray, cues: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-frame detector input: the log-mel, plus DETECTOR_CUES and a
    voicing-weighted log2 pitch when ``cues`` is given."""
    logmel = np.asarray(logmel, dtype=np.float32)
    if cues is None:
        return logmel
    names = cue_names()
    cues = np.asarray(cues, dtype=np.float32)
    picked = [np.maximum(cues[:, names.index(n)], -80.0) for n in DETECTOR_CUES]
    f0 = cues[:, names.index("f0_hz")]
    voicing = cues[:, names.index("voicing")]
    picked.append(np.log2(np.maximum(f0, PITCH_FMIN)) * voicing)
    return np.concatenate([logmel, np.stack(picked, axis=1)], axis=1)


CONTEXT_SPANS = ((-2, 2), (-12, -3), (3, 12), (-37, 37))  # frames: 0.1 s, left/right 0.2 s, 1.5 s


def _box_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of x[t+lo .. t+hi] (inclusive) per row, edges clamped."""
    n = len(x)
    pad = np.concatenate([np.repeat(x[:1], max(0, -lo), 0), x, np.repeat(x[-1:], max(0, hi), 0)])
    csum = np.concatenate([np.zeros((1, x.shape[1]), np.float64), np.cumsum(pad, axis=0, dtype=np.float64)])
    off = max(0, -lo)
    start = np.arange(n) + lo + off
    stop = np.arange(n) + hi + off + 1
    return ((csum[stop] - csum[start]) / (hi - lo + 1)).astype(np.float32)


CONTEXT_MARGIN = max(max(abs(lo), abs(hi)) for lo, hi in CONTEXT_SPANS)
CONTEXT_CHUNK = 50000  # frames per block (1000 s): bounds memory on a whole VOD


def context_features(frames: np.ndarray, rows: Optional[np.ndarray] = None,
                     transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                     chunk: int = CONTEXT_CHUNK) -> np.ndarray:
    """Stack box means over CONTEXT_SPANS: the frame's neighbourhood, what came
    before, what comes after, and the surrounding 1.5 s (the speaker's context).

    ``rows`` (sorted frame indices) limits the output to those frames;
    ``transform`` maps a block of raw frames to model inputs (e.g. stack cues
    and normalise). Both work block by block, so a four-hour VOD never holds
    its full context matrix. Results equal a single pass over all frames.
    """
    n = len(frames)
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64)
    transform = transform or (lambda block: np.asarray(block, dtype=np.float32))
    width = None
    out = None
    for start in range(0, max(n, 1), chunk):
        stop = min(n, start + chunk)
        lo_row, hi_row = np.searchsorted(rows, start), np.searchsorted(rows, stop)
        if hi_row <= lo_row and out is not None:
            continue
        a, b = max(0, start - CONTEXT_MARGIN), min(n, stop + CONTEXT_MARGIN)
        block = transform(frames[a:b]) if b > a else transform(frames[:0])
        if out is None:
            width = block.shape[1] * len(CONTEXT_SPANS)
            out = np.empty((len(rows), width), np.float32)
        if hi_row <= lo_row:
            continue
        ctx = np.concatenate([_box_mean(block, lo, hi) for lo, hi in CONTEXT_SPANS], axis=1)
        out[lo_row:hi_row] = ctx[rows[lo_row:hi_row] - a]
    return out


@dataclass
class Normalizer:
    """Per-VOD mean/variance normalisation (statistics of the VOD's own frames)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, frames: np.ndarray, max_rows: int = 200000, seed: int = 0) -> "Normalizer":
        frames = np.asarray(frames, dtype=np.float32)
        if len(frames) > max_rows:
            frames = frames[np.random.default_rng(seed).choice(len(frames), max_rows, replace=False)]
        return cls(frames.mean(0), frames.std(0) + 1e-3)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        return ((np.asarray(frames, dtype=np.float32) - self.mean) / self.std).astype(np.float32)


@dataclass
class MLP:
    """One hidden layer (ReLU), sigmoid output; numpy forward and Adam training."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    history: List[float] = field(default_factory=list)

    @classmethod
    def init(cls, n_in: int, n_hidden: int = 64, seed: int = 0) -> "MLP":
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden)).astype(np.float32),
                   np.zeros(n_hidden, np.float32),
                   rng.normal(0, np.sqrt(1.0 / n_hidden), (n_hidden, 1)).astype(np.float32),
                   np.zeros(1, np.float32))

    def logits(self, x: np.ndarray, batch: int = 65536) -> np.ndarray:
        out = np.empty(len(x), np.float32)
        for i in range(0, len(x), batch):
            h = np.maximum(x[i:i + batch] @ self.w1 + self.b1, 0.0)
            out[i:i + batch] = (h @ self.w2 + self.b2)[:, 0]
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.logits(x), -30, 30)))

    def fit(self, x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None,
            epochs: int = 8, batch: int = 512, lr: float = 2e-3, l2: float = 1e-4,
            seed: int = 0) -> "MLP":
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        weight = np.ones(len(y), np.float32) if weight is None else np.asarray(weight, np.float32)
        rng = np.random.default_rng(seed)
        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        for _ in range(epochs):
            order = rng.permutation(len(x))
            total = 0.0
            for i in range(0, len(x), batch):
                idx = order[i:i + batch]
                xb, yb, wb = x[idx], y[idx], weight[idx]
                pre = xb @ self.w1 + self.b1
                h = np.maximum(pre, 0.0)
                z = (h @ self.w2 + self.b2)[:, 0]
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                wsum = wb.sum() + 1e-9
                total += float((wb * -(yb * np.log(p + 1e-7) + (1 - yb) * np.log(1 - p + 1e-7))).sum())
                dz = ((p - yb) * wb / wsum)[:, None].astype(np.float32)
                g_w2 = h.T @ dz + l2 * self.w2
                g_b2 = dz.sum(0)
                dh = (dz @ self.w2.T) * (pre > 0)
                g_w1 = xb.T @ dh + l2 * self.w1
                g_b1 = dh.sum(0)
                step += 1
                for k, (param, grad) in enumerate(zip(params, (g_w1, g_b1, g_w2, g_b2))):
                    m[k] = 0.9 * m[k] + 0.1 * grad
                    v[k] = 0.999 * v[k] + 0.001 * grad * grad
                    mh = m[k] / (1 - 0.9 ** step)
                    vh = v[k] / (1 - 0.999 ** step)
                    param -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
            self.history.append(total / max(float(weight.sum()), 1e-9))
        return self


def smooth_probability(p: np.ndarray, frames: int = 5) -> np.ndarray:
    """Centred moving average of the logit over ``frames`` frames (odd)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    if frames <= 1 or len(p) == 0:
        return p.astype(np.float32)
    half = frames // 2
    logit = np.log(p / (1 - p))[:, None]
    smooth = _box_mean(logit, -half, half)[:, 0]
    return (1.0 / (1.0 + np.exp(-smooth))).astype(np.float32)


# ---------------------------------------------------------------- weak labels


def frames_in_spans(n_frames: int, spans: Sequence[Tuple[float, float]], t0: float = 0.0) -> np.ndarray:
    """Boolean per frame: frame centre inside any span (seconds, VOD time)."""
    mask = np.zeros(n_frames, bool)
    if n_frames == 0:
        return mask
    times = frame_time(np.arange(n_frames)) + t0
    for a, b in spans:
        lo, hi = np.searchsorted(times, a, side="left"), np.searchsorted(times, b, side="right")
        mask[lo:hi] = True
    return mask


def window_extremes(n_frames: int, starts: np.ndarray, ends: np.ndarray, sims: np.ndarray,
                    t0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Per frame, (min, max) similarity of the dominant-voice windows covering
    it; NaN where none does."""
    lo_sim = np.full(n_frames, np.inf, np.float32)
    hi_sim = np.full(n_frames, -np.inf, np.float32)
    times = frame_time(np.arange(n_frames)) + t0
    for a, b, s in zip(starts, ends, sims):
        i0, i1 = np.searchsorted(times, a, side="left"), np.searchsorted(times, b, side="right")
        if i1 > i0:
            lo_sim[i0:i1] = np.minimum(lo_sim[i0:i1], s)
            hi_sim[i0:i1] = np.maximum(hi_sim[i0:i1], s)
    lo_sim[~np.isfinite(lo_sim)] = np.nan
    hi_sim[~np.isfinite(hi_sim)] = np.nan
    return lo_sim, hi_sim


def weak_labels(speech: np.ndarray, min_sim: np.ndarray, max_sim: np.ndarray,
                streamer_at: float = 0.45, other_below: float = 0.10,
                game_margin: int = 15) -> np.ndarray:
    """1 streamer, 0 not-streamer, -1 unused, per frame.

    Streamer: speech frame whose every covering window scores >= streamer_at.
    Other: speech frame whose every covering window scores < other_below.
    Game: a frame at least ``game_margin`` frames from any speech.
    """
    labels = np.full(len(speech), -1, np.int8)
    labels[speech & (min_sim >= streamer_at)] = 1
    labels[speech & (max_sim < other_below)] = 0
    near = _box_mean(speech.astype(np.float32)[:, None], -game_margin, game_margin)[:, 0] > 0
    labels[~near] = 0
    return labels


def runs(mask: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """[start, stop) index runs where ``mask`` is True, at least ``min_len`` long."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= min_len]


def mix(target: np.ndarray, other: np.ndarray, ratio_db: float) -> np.ndarray:
    """``target`` plus ``other`` scaled so the other's RMS sits ``ratio_db`` from
    the target's (negative: quieter). Both stereo float arrays of equal length."""
    rt = float(np.sqrt(np.mean(np.square(target)) + 1e-12))
    ro = float(np.sqrt(np.mean(np.square(other)) + 1e-12))
    gain = rt / ro * (10 ** (ratio_db / 20.0))
    out = target + gain * other
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    return out / peak * 0.99 if peak > 0.99 else out


# ---------------------------------------------------------------- detector bundle


def raw_frames(logmel: np.ndarray, cues: Optional[np.ndarray], uses_cues: bool) -> np.ndarray:
    """What a Detector reads per frame: the log-mel, with the raw cue columns
    appended when the model uses them (detector_inputs picks from those)."""
    if not uses_cues:
        return np.asarray(logmel)
    if cues is None:
        raise ValueError("this detector needs the cue columns")
    return np.concatenate([np.asarray(logmel), np.asarray(cues, dtype=np.asarray(logmel).dtype)], axis=1)


def _inputs(raw_block: np.ndarray, uses_cues: bool) -> np.ndarray:
    raw_block = np.asarray(raw_block, dtype=np.float32)
    return detector_inputs(raw_block[:, :N_MELS], raw_block[:, N_MELS:] if uses_cues else None)


def fit_normalizer(raw: np.ndarray, uses_cues: bool, max_rows: int = 200000, seed: int = 0) -> Normalizer:
    """Normaliser over a random sample of the VOD's own frames."""
    rows = np.arange(len(raw))
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(len(rows), max_rows, replace=False))
    return Normalizer.fit(_inputs(raw[rows], uses_cues), max_rows=max_rows, seed=seed)


def model_inputs(raw: np.ndarray, norm: Normalizer, uses_cues: bool,
                 rows: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalised context features for ``rows`` (all frames when None)."""
    return context_features(raw, rows=rows, transform=lambda block: norm(_inputs(block, uses_cues)))


@dataclass
class Detector:
    """Normaliser + MLP over context features of the per-frame inputs
    (detector_inputs: the log-mel, optionally with cues)."""

    norm: Normalizer
    mlp: MLP
    uses_cues: bool = False

    def raw_probability(self, raw: np.ndarray, smooth_frames: int = 5) -> np.ndarray:
        """Per-frame probability for ``raw`` = raw_frames(...), block by block."""
        out = np.empty(len(raw), np.float32)
        for start in range(0, len(raw), CONTEXT_CHUNK):
            rows = np.arange(start, min(len(raw), start + CONTEXT_CHUNK))
            out[rows] = self.mlp.predict(model_inputs(raw, self.norm, self.uses_cues, rows))
        return smooth_probability(out, smooth_frames)

    def probability(self, logmel: np.ndarray, cues: Optional[np.ndarray] = None,
                    smooth_frames: int = 5) -> np.ndarray:
        return self.raw_probability(raw_frames(logmel, cues, self.uses_cues), smooth_frames)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.norm.mean, "std": self.norm.std, "w1": self.mlp.w1, "b1": self.mlp.b1,
                "w2": self.mlp.w2, "b2": self.mlp.b2, "uses_cues": np.array(self.uses_cues)}

    @classmethod
    def from_dict(cls, d) -> "Detector":
        uses = bool(np.asarray(d["uses_cues"])) if "uses_cues" in d else False
        return cls(Normalizer(np.asarray(d["mean"]), np.asarray(d["std"])),
                   MLP(np.asarray(d["w1"]), np.asarray(d["b1"]), np.asarray(d["w2"]), np.asarray(d["b2"])), uses)


def balanced_fit(xs: Sequence[np.ndarray], ys: Sequence[np.ndarray], n_hidden: int = 64,
                 epochs: int = 8, seed: int = 0) -> MLP:
    """MLP on the concatenated rows, the two classes weighted to equal mass."""
    x, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weight = np.where(y > 0.5, 0.5 / pos, 0.5 / neg).astype(np.float32) * len(y)
    return MLP.init(x.shape[1], n_hidden, seed).fit(x, y, weight, epochs=epochs, seed=seed)


def train_detector(raw: np.ndarray, labels: np.ndarray,
                   extra: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                   norm: Optional[Normalizer] = None, uses_cues: bool = False,
                   n_hidden: int = 64, max_per_class: int = 60000, epochs: int = 8,
                   seed: int = 0) -> Detector:
    """Fit on the VOD's own weakly labelled frames (labels 1/0, -1 ignored).

    ``raw`` is raw_frames(...) for the whole VOD. ``extra`` is (model inputs
    already normalised with ``norm`` and stacked, labels): the synthetic-overlap
    frames, which have no place in the VOD's frame sequence. Each source is
    subsampled to ``max_per_class`` rows per class; the classes are weighted
    to equal mass.
    """
    norm = norm or fit_normalizer(raw, uses_cues, seed=seed)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cls_value in (1, 0):
        idx = np.where(labels == cls_value)[0]
        if len(idx) > max_per_class:
            idx = np.sort(rng.choice(idx, max_per_class, replace=False))
        if len(idx):
            xs.append(model_inputs(raw, norm, uses_cues, idx))
            ys.append(np.full(len(idx), cls_value, np.float32))
    if extra is not None and len(extra[0]):
        ex, ey = extra
        if len(ex) > max_per_class:
            keep = rng.choice(len(ex), max_per_class, replace=False)
            ex, ey = ex[keep], ey[keep]
        xs.append(np.asarray(ex, np.float32))
        ys.append(np.asarray(ey, np.float32))
    return Detector(norm, balanced_fit(xs, ys, n_hidden, epochs, seed), uses_cues)
