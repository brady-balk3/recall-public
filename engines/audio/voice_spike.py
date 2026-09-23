# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from core.bundle_paths import get_data_dir
import os
import subprocess
import wave
import numpy as np
from typing import List
from core.models.signal import AudioSignal
from core.ffmpeg_path import get_ffmpeg_path

def extract_audio(video_path: str, output_path: str | None = None,
                  audio_track: int = None) -> str:
    """Extract audio from video using FFmpeg.

    ``audio_track`` selects a specific audio stream (0-based) on multitrack
    recordings — plan 22 §4.4 routes analysis to the mic-only track when one
    is detected. None keeps ffmpeg's default stream selection.
    """
    if output_path is None:
        output_path = os.path.join(get_data_dir(), 'cache', 'audio.wav')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    # Extract mono audio at 16kHz
    cmd = [get_ffmpeg_path(), "-i", video_path]
    if audio_track is not None:
        cmd += ["-map", f"0:a:{int(audio_track)}"]
    cmd += [
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path, "-y"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return output_path

def _clamp(value: float, min_value: float = 0.0, max_value: float = 5.0) -> float:
    return max(min_value, min(max_value, value))

def _robust_deviation(values: np.ndarray, baseline: float) -> float:
    if len(values) == 0:
        return 0.005
    mad = float(np.median(np.abs(values - baseline)))
    return max(0.005, mad * 1.4826)

def analyze_audio_spikes(
    wav_path: str,
    window_duration_sec: float = 1.0,
    spike_threshold: float = 1.0,
    baseline_window_sec: float = 60.0
) -> List[AudioSignal]:
    """Analyze audio with an adaptive baseline and robust spike score."""
    signals = []
    
    with wave.open(wav_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        
    # Convert binary data to numpy array
    samples = np.frombuffer(audio_data, dtype=np.int16)
    
    # Normalize samples to [-1.0, 1.0]
    samples = samples.astype(np.float32) / 32768.0
    
    samples_per_window = max(1, int(sample_rate * window_duration_sec))
    amplitudes = []
    timestamps = []

    for i in range(0, len(samples), samples_per_window):
        window = samples[i:i+samples_per_window]
        if len(window) == 0:
            break

        amplitudes.append(float(np.sqrt(np.mean(window**2))))
        timestamps.append(i / sample_rate)

    if not amplitudes:
        return signals

    amplitude_array = np.array(amplitudes, dtype=np.float32)
    baseline_window_count = max(3, int(round(baseline_window_sec / window_duration_sec)))

    for idx, rms_amplitude in enumerate(amplitudes):
        left = max(0, idx - baseline_window_count + 1)
        baseline_values = amplitude_array[left:idx + 1]
        baseline = float(np.median(baseline_values))
        deviation = _robust_deviation(baseline_values, baseline)
        z_score = (rms_amplitude - baseline) / deviation
        spike_score = _clamp(z_score)
        is_spike = spike_score >= spike_threshold

        signals.append(AudioSignal(
            timestamp=timestamps[idx],
            amplitude=rms_amplitude,
            is_spike=is_spike,
            baseline=baseline,
            z_score=float(z_score),
            spike_score=spike_score
        ))
        
    return signals
