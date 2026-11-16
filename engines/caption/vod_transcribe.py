# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""VOD-level ASR (plan §5.3).

Transcribe the whole VOD once, before clip selection, so the transcript can both
(a) feed the ASR hype track that informs WHICH clips to pick, and (b) feed
titles / semantic context. Burned-in captions re-run the same ASR per clip
for preview=export quality (see caption_generator / generate_ass_for_clip).

Uses Qwen3-ASR with the forced aligner, the only backend, through
engines.caption.asr (see qwen_asr).
"""

import os
import tempfile

from engines.caption.asr import get_asr_model
from engines.caption.asr import transcribe_with_cancel
from engines.caption.asr import extract_clip_audio


def transcribe_vod(wav_path: str, language: str = "en", model_size: str | None = None,
                   cancel_check=None, progress_callback=None, settings: dict | None = None,
                   words_callback=None, regions=None, step_timer=None) -> dict:
    """Full-VOD transcription with word timestamps. Returns the whisper-shaped result dict.

    ``words_callback`` receives each chunk's timed words as they land,
    ``regions`` are speech regions the caller already found in ``wav_path``
    (the VAD pass is then skipped), and ``step_timer`` times the speech
    detection on its own (see QwenASRAdapter.transcribe). Each is only passed
    through when given.
    """
    _ = settings
    model = get_asr_model(model_size)
    extra = {
        key: value for key, value in (
            ("words_callback", words_callback),
            ("regions", regions),
            ("step_timer", step_timer),
        ) if value is not None
    }
    return transcribe_with_cancel(
        model,
        wav_path,
        cancel_check=cancel_check,
        word_timestamps=True,
        language=language,
        condition_on_previous_text=False,  # avoid drift over a long VOD
        progress_callback=progress_callback,
        **extra,
    )


def detect_speech_regions(wav_path: str, cancel_check=None):
    """The speech regions transcribe_vod would find in ``wav_path``, for a
    caller that runs the VAD early and hands them over as ``regions``.

    Same duration probe and same function as QwenASRAdapter.transcribe, so the
    regions are the ones it would have computed itself. Qwen3-ASR is the only
    backend (asr.get_asr_model), so its VAD is the one to match.
    """
    from engines.caption import qwen_asr

    return qwen_asr.speech_regions(
        wav_path, qwen_asr.audio_duration(wav_path), cancel_check=cancel_check,
    )


# Version of the cached speech regions, part of their file name
# (``{audio_key}_speech_regions_v1``). Bump it whenever speech_regions can
# return different regions for the same WAV: the Silero weights or onnx-asr's
# VAD defaults, the seam merge, or the block length. VAD_BLOCK_SEC is also
# stored in each entry and checked on load, because Silero restarts at every
# block and it is the one tunable here that moves regions.
SPEECH_REGIONS_CACHE_VERSION = 1


def speech_regions_cache_entry(regions) -> dict:
    """What run_pipeline caches for ``regions`` (detect_speech_regions output)."""
    from engines.caption import qwen_asr

    return {
        "vad_block_sec": qwen_asr.VAD_BLOCK_SEC,
        "regions": [[float(start), float(end)] for start, end in regions],
    }


def speech_regions_from_cache(entry):
    """The regions in a cached entry, or None when it is missing, from other
    VAD settings, or malformed (the caller then runs the VAD again)."""
    import math

    from engines.caption import qwen_asr

    if not isinstance(entry, dict) or entry.get("vad_block_sec") != qwen_asr.VAD_BLOCK_SEC:
        return None
    rows = entry.get("regions")
    if not isinstance(rows, list):
        return None
    regions = []
    for row in rows:
        if not (isinstance(row, list) and len(row) == 2
                and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) for value in row)
                and 0.0 <= row[0] <= row[1]):
            return None
        regions.append((float(row[0]), float(row[1])))
    return regions


def transcribe_vod_regions(wav_path: str, regions, language: str = "en",
                           model_size: str | None = None, cancel_check=None,
                           progress_callback=None, settings: dict | None = None,
                           step_timer=None) -> dict:
    """Transcribe selected VOD regions and offset word timestamps to VOD time.

    This is the fast-mode "speech scout": it avoids a full 2-3 hour ASR pass
    while still giving selection a chance to notice contextual hype in the
    regions that the cheap first pass already thinks are plausible.

    ``step_timer`` is passed to every region's transcription, so the speech
    detection inside each one accumulates under one name.
    """
    _ = settings
    model = get_asr_model(model_size)
    # ``covered_regions`` tells readers which stretches were actually heard, so
    # silence outside them is "not transcribed", not "nobody spoke". A full
    # VOD transcript carries no such key and counts as covering everything.
    merged = {"segments": [], "covered_regions": []}
    segment_id = 0
    total_duration = sum(max(0.0, float(end) - float(start)) for start, end in regions)
    completed_duration = 0.0

    for idx, (start, end) in enumerate(regions):
        if cancel_check:
            cancel_check()
        start = max(0.0, float(start))
        end = max(start, float(end))
        if end - start < 1.0:
            continue

        temp_audio = None
        try:
            fd, temp_audio = tempfile.mkstemp(prefix=f"recall_region_{idx}_", suffix=".wav")
            os.close(fd)
            extract_clip_audio(wav_path, start, end, temp_audio)
            result = transcribe_with_cancel(
                model,
                temp_audio,
                cancel_check=cancel_check,
                word_timestamps=True,
                language=language,
                condition_on_previous_text=False,
                progress_callback=(
                    (lambda processed, _total, completed=completed_duration:
                        progress_callback(completed + processed, total_duration))
                    if progress_callback else None
                ),
                **({"step_timer": step_timer} if step_timer else {}),
            )
        finally:
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except OSError:
                    pass

        for segment in result.get("segments", []):
            shifted = dict(segment)
            shifted["id"] = segment_id
            segment_id += 1
            shifted["start"] = float(segment.get("start", 0.0)) + start
            shifted["end"] = float(segment.get("end", 0.0)) + start
            shifted_words = []
            for word in segment.get("words", []) or []:
                shifted_word = dict(word)
                shifted_word["start"] = float(word.get("start", 0.0)) + start
                shifted_word["end"] = float(word.get("end", 0.0)) + start
                shifted_words.append(shifted_word)
            shifted["words"] = shifted_words
            merged["segments"].append(shifted)
        merged["covered_regions"].append([start, end])

        completed_duration += end - start
        if progress_callback:
            progress_callback(completed_duration, total_duration)

    return merged


def iter_words(transcript: dict):
    """Yield (text_lower, start, end) for every word in the transcript."""
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            text = (word.get("word") or "").strip().lower()
            if not text:
                continue
            # strip surrounding punctuation but keep internal apostrophes
            cleaned = text.strip(".,!?;:\"'()[]")
            if cleaned:
                yield cleaned, float(word.get("start", 0.0)), float(word.get("end", 0.0))
