# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Compatibility alias for ``engines.caption.asr``, under its old names.

The module was renamed in 2026-11 because it no longer has anything to do
with Whisper (Qwen3-ASR is the only backend). Product code and tests import
``engines.caption.asr``; this stays for one release so research scripts and
the public tree's port do not all have to change at once. Delete it in the
release after the first one that ships ``engines.caption.asr``, once nothing
imports it (``git grep whisper_asr``).

Only names are re-exported. The loaded model, its lock, the release timer and
the transcription counter live in ``engines.caption.asr`` alone, so patching
an attribute HERE changes nothing the product calls: patch
``engines.caption.asr`` instead. Private names are deliberately not
re-exported, so code still reaching for them fails loudly instead of reading
a stale copy of a rebound global.
"""

from engines.caption.asr import (  # noqa: F401 - re-exports
    CAPTION_ALIGNMENT_SIZE,
    CAPTION_ALIGNMENT_SIZE_QUALITY,
    CAPTION_AUDIO_LAG_SEC,
    CHUNK_GAP_SEC,
    COLLAPSE_DUR_SEC,
    DEFAULT_ASR_MODEL_SIZE,
    DEFAULT_MODEL_IDLE_SECONDS,
    LINGER_PAD_SEC,
    MAX_REFLOW_WORD_SEC,
    MIN_REFLOW_WORD_SEC,
    QWEN_CACHE_LABEL,
    STREAM_ASR_PROMPT,
    asr_cache_label,
    asr_display_name,
    clip_transcript_to_vod_time,
    extract_clip_audio,
    format_time_ass,
    generate_ass_for_clip,
    generate_ass_from_transcript,
    generate_tiktok_ass,
    get_asr_model,
    has_loaded_asr_models,
    hold_asr_models,
    release_asr_models,
    resolve_asr_model_size,
    resolve_caption_model_size,
    schedule_asr_model_release,
    slice_transcript_words,
    transcribe_clip_window,
    transcribe_with_cancel,
)

# The old names.
DEFAULT_WHISPER_SIZE = DEFAULT_ASR_MODEL_SIZE
QUALITY_WHISPER_SIZE = DEFAULT_ASR_MODEL_SIZE
resolve_whisper_size = resolve_asr_model_size
get_whisper_model = get_asr_model
has_loaded_whisper_models = has_loaded_asr_models
hold_whisper_models = hold_asr_models
schedule_whisper_model_release = schedule_asr_model_release
release_whisper_models = release_asr_models
