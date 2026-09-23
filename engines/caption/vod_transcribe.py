# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""VOD-level ASR (plan §5.3).

Transcribe the whole VOD once, before clip selection, so the transcript can both
(a) feed the ASR hype track that informs WHICH clips to pick, and (b) feed
titles / semantic context. Burned-in captions re-run medium.en per clip for
preview=export quality (see caption_generator / generate_ass_for_clip).

Uses faster-whisper medium.en by default (engines.caption.whisper_asr).
"""

import os
import tempfile

from engines.caption.whisper_asr import get_whisper_model
from engines.caption.whisper_asr import transcribe_with_cancel
from engines.caption.whisper_asr import extract_clip_audio


def transcribe_vod(wav_path: str, language: str = "en", model_size: str | None = None,
                   cancel_check=None, progress_callback=None, settings: dict | None = None) -> dict:
    """Full-VOD transcription with word timestamps. Returns the Whisper result dict."""
    _ = settings
    model = get_whisper_model(model_size)
    return transcribe_with_cancel(
        model,
        wav_path,
        cancel_check=cancel_check,
        word_timestamps=True,
        language=language,
        condition_on_previous_text=False,  # avoid drift over a long VOD
        progress_callback=progress_callback,
    )


def transcribe_vod_regions(wav_path: str, regions, language: str = "en",
                           model_size: str | None = None, cancel_check=None,
                           progress_callback=None, settings: dict | None = None) -> dict:
    """Transcribe selected VOD regions and offset word timestamps to VOD time.

    This is the fast-mode "speech scout": it avoids a full 2-3 hour ASR pass
    while still giving selection a chance to notice contextual hype in the
    regions that the cheap first pass already thinks are plausible.
    """
    _ = settings
    model = get_whisper_model(model_size)
    merged = {"segments": []}
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
