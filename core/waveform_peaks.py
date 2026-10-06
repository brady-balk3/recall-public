# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Waveform peaks for a whole source VOD, for the editor's timeline.

The editor streams hours-long VODs by byte range and never holds their audio,
so it can't draw a waveform itself (decoding a 4 h track in the renderer is
~5 GB of samples). Instead ffmpeg decodes the track here once, as a stream,
and we keep only the loudest sample of each short bucket: 200 buckets a
second, one byte each, about 2.8 MB for four hours.

File format (little-endian), so the editor needs no custom HTTP headers:

    b"RWF2" | u32 sample_rate | u32 bucket_size | u32 total_samples
            | f32 loud_level | u8 peaks...

Each peak is the loudest absolute sample of the louder channel in its bucket,
matching the editor's own summary, stored on a decibel scale: byte b means
(b * 80 / 255 - 80) dBFS, and 0 means silence (-80 dB or quieter). A linear
byte would leave three steps for everything below -40 dB, where much of a
stream's mix sits. ``loud_level`` is the VOD's own loud level (the 99.5th
percentile of its non-silent peaks, linear 0..1), so the editor can draw a
quiet stream relative to itself instead of as a flat line.
"""
from __future__ import annotations

import os
import struct
import subprocess
import threading
from typing import Optional

from core.ffmpeg_path import get_ffmpeg_path

MAGIC = b"RWF2"
FLOOR_DB = 80.0
LOUD_PERCENTILE = 99.5
SAMPLE_RATE = 16_000
BUCKET_SIZE = 80  # 200 buckets per second at 16 kHz
CHANNELS = 2
# Buckets read from ffmpeg per chunk: 10 minutes of audio, ~38 MB of samples.
_CHUNK_BUCKETS = 200 * 600

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(os.path.abspath(path), threading.Lock())


def build_waveform_peaks(source_path: str, out_path: str) -> Optional[str]:
    """Write the peaks file for ``source_path`` to ``out_path`` (once).

    Returns the path, or None when the source has no decodable audio. Safe to
    call concurrently: the second caller waits for the first and reuses it.
    """
    if not source_path or not os.path.isfile(source_path):
        return None
    with _lock_for(out_path):
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 20:
            return out_path
        return _build(source_path, out_path)


def _build(source_path: str, out_path: str) -> Optional[str]:
    import numpy as np

    command = [
        get_ffmpeg_path(), "-v", "error", "-nostdin",
        "-i", source_path,
        "-map", "0:a:0", "-vn",
        "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "-",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(  # nosec B603 - fixed argv, no shell
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=creationflags,
    )
    frame_bytes = CHANNELS * 2
    bucket_bytes = BUCKET_SIZE * frame_bytes
    chunk_bytes = _CHUNK_BUCKETS * bucket_bytes
    peaks: list[np.ndarray] = []
    total_samples = 0
    leftover = b""

    def consume(data: bytes) -> None:
        nonlocal total_samples
        samples = np.frombuffer(data, dtype="<i2").reshape(-1, CHANNELS)
        total_samples += samples.shape[0]
        peaks.append(_bucket_peaks(samples, np))

    try:
        assert process.stdout is not None
        while True:
            data = process.stdout.read(chunk_bytes)
            if not data:
                break
            # Pipe reads can come back short: only whole buckets are consumed,
            # so a bucket never straddles two reads and shifts the rest.
            data = leftover + data
            usable = len(data) - (len(data) % bucket_bytes)
            leftover = data[usable:]
            if usable:
                consume(data[:usable])
        tail = len(leftover) - (len(leftover) % frame_bytes)
        if tail:
            consume(leftover[:tail])
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
    if process.returncode != 0 or total_samples == 0:
        return None

    linear = np.concatenate(peaks)
    audible = linear[linear > 10 ** (-FLOOR_DB / 20)]
    loud_level = float(np.percentile(audible, LOUD_PERCENTILE)) if audible.size else 0.0
    body = _encode_db(linear, np).tobytes()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    temp_path = f"{out_path}.tmp-{os.getpid()}-{threading.get_ident()}"
    with open(temp_path, "wb") as handle:
        handle.write(MAGIC + struct.pack("<IIIf", SAMPLE_RATE, BUCKET_SIZE, total_samples, loud_level) + body)
    os.replace(temp_path, out_path)
    return out_path


def _bucket_peaks(samples, np):
    """Loudest absolute sample per bucket (last bucket may be short), linear 0..1."""
    loud = np.abs(samples.astype(np.int32)).max(axis=1)
    count = -(-loud.shape[0] // BUCKET_SIZE)
    padded = np.zeros(count * BUCKET_SIZE, dtype=np.int32)
    padded[: loud.shape[0]] = loud
    return (padded.reshape(count, BUCKET_SIZE).max(axis=1) / 32768.0).astype(np.float32)


def _encode_db(linear, np):
    """Linear peaks as bytes on the -80..0 dBFS scale; 0 is silence."""
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.maximum(linear, 1e-12))
    encoded = np.rint((db + FLOOR_DB) * (255 / FLOOR_DB))
    return np.clip(encoded, 0, 255).astype(np.uint8)


def decode_db(byte: int) -> float:
    """A peak byte back to linear amplitude (for tests and tools)."""
    return 0.0 if byte <= 0 else 10 ** ((byte * FLOOR_DB / 255 - FLOOR_DB) / 20)
