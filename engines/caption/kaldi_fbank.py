# SPDX-License-Identifier: BSD-2-Clause
# Derived from PyTorch Audio; see LICENSE.torchaudio.
# Source: https://raw.githubusercontent.com/pytorch/audio/34c52a67e8941bbd8e6adaca0eb0b9eabec11d78/src/torchaudio/compliance/kaldi.py
# Modification: a numpy re-implementation of `fbank` for the one option set
# engines/caption/speaker_id.embed uses; the torch version that preceded it
# retained fbank and its dependency functions with the math unchanged. Each
# step below names the torchaudio function it reproduces. The torch version is
# kept as tests/unit/kaldi_fbank_torch_reference.py, and test_kaldi_fbank.py
# holds this one to it.
"""Kaldi-compatible log-mel filterbanks in numpy (torchaudio.compliance.kaldi).

Only what speaker_id.embed asks for is implemented, i.e. torchaudio's
``fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
dither=0.0, sample_frequency=16000, energy_floor=0.0, window_type="hamming",
htk_compat=True, use_energy=False)`` with every other argument at its default:
preemphasis 0.97, DC offset removed per frame, snip_edges, the 400-sample
frame zero-padded to a 512-point FFT, power spectrum, low_freq 20 Hz,
high_freq at Nyquist, no VTLN, natural log floored at float32 epsilon, no
mean subtraction. ``energy_floor``, ``raw_energy`` and ``htk_compat`` only
shape the energy column, which ``use_energy=False`` drops, so they have no
counterpart here.

Why numpy: this was the last torch code under engines/, and the shipped
engine is dropping torch (docs/TORCH_REMOVAL_SPIKE_2026-11.md). MEASURED
2026-11-15 on 100 real 1.5 s speech windows, through speaker_id.embed with its
one-thread ONNX session: 400 ms CPU per window with the torch filterbank
(torch's 8 pool threads keep spinning while the model runs) against 49 ms with
this one, both ~50 ms wall. The filterbank alone is 5.5 ms CPU here, 6.2 ms in
torch.

Computed in float64 and returned as float32. torch ran the same steps in
float32, mel triangles included, so the two differ by torch's own rounding,
not by method. MEASURED on 398 speech windows from two VODs: the largest
log-mel difference to torch was 1.53e-3, exactly torch's own float32-vs-
float64 difference on the same windows, and the speaker embeddings matched at
cosine >= 0.9999998 (test_kaldi_fbank.py holds the bound).
"""

import numpy as np

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
FRAME_LENGTH_MS = 25.0
FRAME_SHIFT_MS = 10.0
PREEMPHASIS = 0.97
LOW_FREQ = 20.0

# _get_waveform_and_window_properties: int() truncation, then
# _next_power_of_2 for round_to_power_of_two.
WINDOW_SIZE = int(SAMPLE_RATE * FRAME_LENGTH_MS * 0.001)  # 400
WINDOW_SHIFT = int(SAMPLE_RATE * FRAME_SHIFT_MS * 0.001)  # 160
PADDED_WINDOW_SIZE = 1 << (WINDOW_SIZE - 1).bit_length()  # 512

# _get_epsilon: torch.finfo(torch.float).eps, whatever the computing dtype.
EPSILON = float(np.finfo(np.float32).eps)


def _mel_scale(freq):
    """mel_scale / mel_scale_scalar."""
    return 1127.0 * np.log(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_banks() -> np.ndarray:
    """get_mel_banks with vtln_warp_factor == 1.0, plus fbank's zero column.

    Shape (NUM_MEL_BINS, PADDED_WINDOW_SIZE // 2 + 1): triangles over the FFT
    bins below Nyquist, and a zero weight on the Nyquist bin itself.
    """
    nyquist = 0.5 * SAMPLE_RATE
    high_freq = nyquist  # high_freq=0.0 means "offset from Nyquist"
    fft_bin_width = SAMPLE_RATE / PADDED_WINDOW_SIZE
    mel_low = _mel_scale(LOW_FREQ)
    mel_delta = (_mel_scale(high_freq) - mel_low) / (NUM_MEL_BINS + 1)
    bins = np.arange(NUM_MEL_BINS, dtype=np.float64)[:, None]
    left = mel_low + bins * mel_delta
    center = mel_low + (bins + 1.0) * mel_delta
    right = mel_low + (bins + 2.0) * mel_delta
    mel = _mel_scale(fft_bin_width * np.arange(PADDED_WINDOW_SIZE // 2))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    weights = np.maximum(0.0, np.minimum(up, down))
    return np.pad(weights, ((0, 0), (0, 1)))


# _feature_window_function(HAMMING): torch.hamming_window(periodic=False,
# alpha=0.54, beta=0.46) is 0.54 - 0.46 cos(2 pi n / (N - 1)), np.hamming.
_WINDOW = np.hamming(WINDOW_SIZE)
_BANKS_T = np.ascontiguousarray(_mel_banks().T)


def fbank(samples: np.ndarray) -> np.ndarray:
    """(frames, 80) float32 log-mel filterbanks of one mono 16 kHz signal.

    ``samples`` keep their scale (speaker_id passes int16-range floats).
    Raises ValueError for fewer than 400 samples, where torchaudio asserted.
    """
    waveform = np.asarray(samples, dtype=np.float64)
    if waveform.ndim != 1:
        raise ValueError("fbank expects a 1-D mono signal")
    if waveform.size < WINDOW_SIZE:
        raise ValueError(
            f"need at least {WINDOW_SIZE} samples for one frame, got {waveform.size}")

    # _get_strided, snip_edges=True: only frames that fit entirely.
    count = 1 + (waveform.size - WINDOW_SIZE) // WINDOW_SHIFT
    frames = np.lib.stride_tricks.sliding_window_view(waveform, WINDOW_SIZE)
    frames = frames[::WINDOW_SHIFT][:count]

    # _get_window, in its order: DC offset, preemphasis against the previous
    # sample (the first sample against itself: torch pads with "replicate"),
    # window, zero-pad to the FFT size.
    frames = frames - frames.mean(axis=1, keepdims=True)
    previous = np.concatenate((frames[:, :1], frames[:, :-1]), axis=1)
    frames = (frames - PREEMPHASIS * previous) * _WINDOW

    # fbank: power spectrum through the mel triangles, log floored at epsilon.
    spectrum = np.abs(np.fft.rfft(frames, n=PADDED_WINDOW_SIZE, axis=1)) ** 2.0
    energies = spectrum @ _BANKS_T
    return np.log(np.maximum(energies, EPSILON)).astype(np.float32)
