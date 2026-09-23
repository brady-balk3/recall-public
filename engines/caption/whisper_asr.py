# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
import os
import subprocess
import gc
import sys
import threading

from core.ffmpeg_path import get_ffmpeg_path

_MODELS = {}
_MODEL_LOCK = threading.RLock()
_MODEL_RELEASE_TIMER = None
_ACTIVE_TRANSCRIPTIONS = 0
_RELEASE_WHEN_UNUSED = False
DEFAULT_MODEL_IDLE_SECONDS = 180.0

# Full English Whisper for VOD transcription AND burned-in captions.
# Distil was faster but too weak on word accuracy + timing for review previews;
# medium.en has proper alignment heads so the same family can serve hype,
# titles, and on-screen captions. RECALL_WHISPER_MODEL still overrides.
DEFAULT_WHISPER_SIZE = "medium.en"
QUALITY_WHISPER_SIZE = "medium.en"

def resolve_whisper_size(settings: dict | None = None) -> str:
    override = os.environ.get("RECALL_WHISPER_MODEL")
    if override:
        return override
    # processingMode no longer shrinks the ASR model â€” preview/export quality
    # needs medium.en either way. Mode still gates facecam/OCR cadence elsewhere.
    _ = settings
    return DEFAULT_WHISPER_SIZE


# Per-clip caption burn uses the same medium.en family (alignment heads).
# RECALL_CAPTION_MODEL overrides.
CAPTION_ALIGNMENT_SIZE = "medium.en"
CAPTION_ALIGNMENT_SIZE_QUALITY = "medium.en"


def resolve_caption_model_size(settings: dict | None = None) -> str:
    override = os.environ.get("RECALL_CAPTION_MODEL")
    if override:
        return override
    _ = settings
    return CAPTION_ALIGNMENT_SIZE


# Per-clip decoder context biases ambiguous vocabulary toward streaming speech.
# Keep this separate from the VOD pass: with condition_on_previous_text=False,
# initial context does not condition every window. Vocabulary prompting can also
# introduce words into quiet audio, so avoid treating it as an instruction.
# Deterministic corrections in slang_words complement this decoding hint.
STREAM_ASR_PROMPT = ("Yo chat, look at this. Chat, what do you mean? Chat, we're so cooked. "
                     "Holy shart, these guys are so sweaty. "
                     # The -maxxing compounds. Qwen hears the construction well
                     # but splits it in two, and falls to "axing" / "Orimax"
                     # when it does not; exemplars give it the closed form to
                     # reach for. SINGLE x on purpose -- this string is what the
                     # ASR emits and therefore what the forced aligner has to
                     # time, and the doubled x collapses it (see slang_words).
                     # join_maxxing adds the second x after alignment.
                     "He's auramaxing, I'm fishmaxing, they're bushmaxing.")


# Cache label for the ASR in use. The two backends before this one did not
# merely decode differently, they produced different WORDS ("slay" vs "sleigh")
# and differently-derived timings, so the label namespaced the transcript cache
# across the swap. Qwen3-ASR is now the only backend and the label is a
# constant; it stays a label so a future swap has the same protection.
QWEN_CACHE_LABEL = "qwen3-asr-1_7b"


def asr_cache_label(model_size: str | None = None) -> str:
    """The transcript cache namespace. ``model_size`` is accepted and ignored:
    Qwen3-ASR is one model, and callers still thread a size through for the
    audio/settings keys around it."""
    _ = model_size
    return QWEN_CACHE_LABEL


def asr_display_name(model_size: str | None = None) -> str:
    """What to show the user for the ASR doing the work."""
    _ = model_size
    return "Qwen3-ASR"


def release_asr_models() -> int:
    """Free ASR weights now, without waiting for the idle timer.

    Called once the VOD transcript exists, because what runs next is the VLM
    judge and the two do not fit on a 16 GB card together. ``hold_whisper_models``
    deliberately suppresses the idle release for the duration of a scan queue,
    which is right for a 1.5 GB whisper and wrong for a ~12 GB Qwen stack, so
    this bypasses the timer rather than reschedules it.
    """
    return release_whisper_models()


def transcribe_with_cancel(model, audio_path, cancel_check=None, **kwargs):
    """Run Whisper with best-effort cancellation at supported boundaries."""
    global _ACTIVE_TRANSCRIPTIONS
    with _MODEL_LOCK:
        _ACTIVE_TRANSCRIPTIONS += 1
    try:
        if cancel_check:
            cancel_check()
        # Every backend behind this call is an adapter written to the
        # cancel_check/progress_callback contract. The capability sniff that
        # used to sit here existed only to strip kwargs the raw openai-whisper
        # model could not accept; it went with that backend.
        result = model.transcribe(audio_path, cancel_check=cancel_check, **kwargs)
        if cancel_check:
            cancel_check()
        return result
    finally:
        with _MODEL_LOCK:
            _ACTIVE_TRANSCRIPTIONS = max(0, _ACTIVE_TRANSCRIPTIONS - 1)
            should_schedule = _RELEASE_WHEN_UNUSED and _ACTIVE_TRANSCRIPTIONS == 0
        if should_schedule:
            schedule_whisper_model_release()


def _load_qwen_asr():
    """Qwen3-ASR + forced aligner, the product ASR since the 2026-08 swap.

    Returned behind openai-whisper's transcribe() contract, so every consumer
    â€” vod_transcribe, asr_hype, the sentence index, the judge, captions â€” reads
    one result-dict shape regardless of what produced it.
    """
    from engines.caption.qwen_asr import QwenASRAdapter
    return QwenASRAdapter()


def get_whisper_model(model_size: str | None = None):
    """The resident ASR model, loading it on first use.

    Qwen3-ASR is the only backend. There is deliberately NO fallback: the
    packaged build cannot be produced without these weights (recall-engine.spec
    raises if they are absent), so a load failure here means a broken install,
    not a machine that would be better served by a lesser model. The previous
    faster-whisper rung was removed 2026-09-01 because silently finishing a scan
    with worse captions is harder to diagnose than failing on the spot -- it
    looks like the ASR regressed rather than like the install did.

    ``model_size`` is accepted and ignored; see asr_cache_label.
    """
    global _MODEL_RELEASE_TIMER
    key = f"qwen:{model_size or resolve_whisper_size()}"
    with _MODEL_LOCK:
        if _MODEL_RELEASE_TIMER is not None:
            _MODEL_RELEASE_TIMER.cancel()
            _MODEL_RELEASE_TIMER = None
        if key not in _MODELS:
            _MODELS[key] = _load_qwen_asr()
        return _MODELS[key]


def has_loaded_whisper_models() -> bool:
    with _MODEL_LOCK:
        return bool(_MODELS)


def hold_whisper_models() -> None:
    """Keep ASR warm while a scan queue is active."""
    global _MODEL_RELEASE_TIMER, _RELEASE_WHEN_UNUSED
    with _MODEL_LOCK:
        _RELEASE_WHEN_UNUSED = False
        if _MODEL_RELEASE_TIMER is not None:
            _MODEL_RELEASE_TIMER.cancel()
            _MODEL_RELEASE_TIMER = None


def schedule_whisper_model_release(delay_seconds: float | None = None) -> bool:
    """Release ASR after a quiet period; repeated work restarts the timer."""
    global _MODEL_RELEASE_TIMER, _RELEASE_WHEN_UNUSED
    with _MODEL_LOCK:
        _RELEASE_WHEN_UNUSED = True
        if _MODEL_RELEASE_TIMER is not None:
            _MODEL_RELEASE_TIMER.cancel()
            _MODEL_RELEASE_TIMER = None
        if not _MODELS or _ACTIVE_TRANSCRIPTIONS:
            return False
        if delay_seconds is None:
            delay_seconds = float(os.environ.get(
                "RECALL_MODEL_IDLE_SECONDS", str(DEFAULT_MODEL_IDLE_SECONDS),
            ))
        delay_seconds = max(0.0, float(delay_seconds))
        if delay_seconds == 0:
            release_whisper_models()
            return True
        timer = threading.Timer(delay_seconds, release_whisper_models)
        timer.daemon = True
        _MODEL_RELEASE_TIMER = timer
        timer.start()
        return True


def release_whisper_models() -> int:
    """Release cached ASR weights once the scan queue has gone idle.

    ``medium.en`` is the model and faster-whisper is its preferred runtime; the
    OpenAI implementation is only a fallback. Clearing this one-entry cache does
    not remove that redundancy path, it only avoids keeping the successful model
    resident forever after work has finished.
    """
    global _MODEL_RELEASE_TIMER
    with _MODEL_LOCK:
        if _ACTIVE_TRANSCRIPTIONS:
            return 0
        if _MODEL_RELEASE_TIMER is not None:
            _MODEL_RELEASE_TIMER.cancel()
            _MODEL_RELEASE_TIMER = None
        models = list(_MODELS.values())
        released = len(models)
        _MODELS.clear()
    for model in models:
        close = getattr(model, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    models.clear()
    gc.collect()
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    try:
        if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
            cuda.empty_cache()
    except Exception:
        pass
    return released

def extract_clip_audio(source_audio: str, start: float, end: float, temp_audio_out: str):
    duration = end - start
    cmd = [
        get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start),
        "-t", str(duration),
        "-i", source_audio,
        "-ac", "1",
        "-ar", "16000",
        temp_audio_out
    ]
    subprocess.run(cmd, check=True)

# Caption pacing: a chunk never spans a pause longer than this, and no word's
# display line outlives the word by more than the reading pad.
CHUNK_GAP_SEC = 0.9
LINGER_PAD_SEC = 0.45
# Whisper word starts often lead the audible onset slightly. Shift captions
# later so words do not flash and vanish before they are heard.
CAPTION_AUDIO_LAG_SEC = 0.12

# A word whose ASR duration is under this carries no real timing â€” Whisper
# collapsed it onto a neighbouring instant. When a run of such words appears
# (a whole spoken phrase dumped on one timestamp because the segment got a
# wrong, too-short end), we re-time the run at a natural reading pace instead
# of letting it flash for a single frame and vanish while it is still spoken.
COLLAPSE_DUR_SEC = 0.08
MIN_REFLOW_WORD_SEC = 0.22
MAX_REFLOW_WORD_SEC = 0.5


def _reflow_collapsed_words(words: list, limit: float | None = None) -> list:
    """Re-time runs of zero/near-zero-duration words at a natural reading pace.

    ``limit`` is the length of the media these words came from. A run at the very
    end has no following word to aim at, so it used to spread forward unbounded
    and land past the end of the clip -- which silently undid the aligner's own
    clamp (qwen_aligner.align: "a word cannot be spoken after the audio it was
    transcribed from ends") and pushed the caption somewhere ffmpeg drops it.
    A final collapsed run must fit within the media duration to remain visible.

    Whisper (any size) sometimes collapses a whole phrase onto one timestamp:
    the segment gets a wrong, too-short end and every word after the first lands
    at that instant (observed on speech over loud game SFX). Burned in, the
    phrase flashes for a frame and disappears seconds before it stops being
    spoken. Detect maximal runs of collapsed words and spread them forward â€”
    evenly toward the next word that DOES have real timing, or, for a run at the
    very end with nothing after it, at a fixed per-word pace. Well-timed words
    are left untouched, so normal captions are unaffected.
    """
    out = [dict(w) for w in words]
    n = len(out)
    i = 0
    while i < n:
        if float(out[i]["end"]) - float(out[i]["start"]) > COLLAPSE_DUR_SEC:
            i += 1
            continue
        # Maximal run [i, j) of collapsed (no-real-duration) words.
        j = i
        while j < n and float(out[j]["end"]) - float(out[j]["start"]) <= COLLAPSE_DUR_SEC:
            j += 1
        run_start = float(out[i]["start"])
        boundary = float(out[j]["start"]) if j < n else None
        count = j - i
        if boundary is not None and boundary > run_start:
            per = max(MIN_REFLOW_WORD_SEC, min(MAX_REFLOW_WORD_SEC, (boundary - run_start) / count))
        else:
            per = MAX_REFLOW_WORD_SEC
        # A trailing run aims at the end of the media instead of at a next word.
        # The pace floor is deliberately NOT applied here: fitting inside the
        # clip matters more than a comfortable reading pace for words that would
        # otherwise not be shown at all.
        stop = boundary
        if boundary is None and limit is not None and limit > run_start:
            stop = limit
            per = min(per, (limit - run_start) / count)
        t = run_start
        for k in range(i, j):
            start = t
            end = t + per
            if stop is not None:
                start = min(start, stop)
                end = min(end, stop)
            out[k]["start"] = start
            out[k]["end"] = end
            t += per
        i = j
    return out


def format_time_ass(seconds: float) -> str:
    """Format seconds to ASS time format H:MM:SS.cs"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_tiktok_ass(result: dict, output_ass: str, style=None,
                        duration: float | None = None):
    """Generates an ASS file with TikTok-style word-by-word highlights.

    ``style`` is a caption-style object/dict (plan 9.4) controlling font, size,
    colors and position; omitted â†’ the classic white-text / yellow-highlight
    look. A disabled style writes nothing and returns None so the caller skips
    caption burn-in.

    ``duration`` is the length of the clip being captioned. Events are clamped
    to it: the audio lag and the reading pad both push a line later, so a word
    spoken near the end can otherwise be scheduled past the last frame, where
    ffmpeg drops it without a word of complaint.
    """
    from engines.caption.caption_style import resolve_caption_style

    style = resolve_caption_style(style)
    if not style.enabled:
        return None

    header = style.header()
    text_color = style.text_color
    highlight_color = style.highlight_color

    events = []

    # Flatten every segment's words into one time-ordered stream, then repair
    # collapsed timing before chunking. Segment boundaries add nothing here (the
    # chunker splits on real speech pauses), and flattening lets the reflow see
    # the next well-timed word even when a bad segment end sits between them.
    words = []
    for segment in result.get('segments', []):
        words.extend(segment.get('words') or [])
    limit = float(duration) if duration else None
    words = _reflow_collapsed_words(words, limit)

    # Chunk words by count AND time: a chunk must never span a speech pause,
    # and the transcript-slicing path feeds the whole clip as one synthetic
    # segment, so count-only chunking froze caption text on screen across every
    # silence in the clip ("words linger rather than showing up when the person
    # is saying them").
    chunk_size = 4
    chunks = []
    current = []
    for word in words:
        if current and (
            len(current) >= chunk_size
            or float(word['start']) - float(current[-1]['end']) > CHUNK_GAP_SEC
        ):
            chunks.append(current)
            current = []
        current.append(word)
    if current:
        chunks.append(current)

    for chunk_idx, chunk in enumerate(chunks):
        if not chunk:
            continue

        # Hard stop: a chunk's linger must never overlap the next chunk's
        # first line. Without this, LINGER_PAD_SEC (~0.9s) keeps the old
        # 4-word phrase on screen while the new chunk starts â€” libass
        # stacks both Layer-0 lines, which reads as repeating/flashing
        # captions that also look like wrong words.
        next_chunk_start = None
        if chunk_idx + 1 < len(chunks) and chunks[chunk_idx + 1]:
            next_chunk_start = float(chunks[chunk_idx + 1][0]["start"])

        # Progressive karaoke: only words spoken so far are visible.
        # Showing the whole 4-word chunk up front made upcoming words appear
        # seconds before they were said (and vanish before the audio).
        for i, target_word in enumerate(chunk):
            start = max(0.0, float(target_word["start"]) + CAPTION_AUDIO_LAG_SEC)
            word_end = max(start, float(target_word["end"]) + CAPTION_AUDIO_LAG_SEC)
            start_time = format_time_ass(start)

            if i + 1 < len(chunk):
                # Hand off cleanly at the next word â€” no linger into it.
                next_start = max(
                    0.0, float(chunk[i + 1]["start"]) + CAPTION_AUDIO_LAG_SEC,
                )
                capped_end = next_start
            else:
                # Last word: short reading pad, clipped before the next chunk.
                capped_end = word_end + LINGER_PAD_SEC
                if next_chunk_start is not None:
                    next_start = max(0.0, next_chunk_start + CAPTION_AUDIO_LAG_SEC)
                    capped_end = min(capped_end, next_start)

            # Prefer ~120ms on screen, but never past the next event.
            hard_stop = None
            if i + 1 < len(chunk):
                hard_stop = max(
                    0.0, float(chunk[i + 1]["start"]) + CAPTION_AUDIO_LAG_SEC,
                )
            elif next_chunk_start is not None:
                hard_stop = max(0.0, next_chunk_start + CAPTION_AUDIO_LAG_SEC)
            capped_end = max(capped_end, start + 0.12)
            if hard_stop is not None:
                capped_end = min(capped_end, hard_stop)
            if capped_end <= start:
                capped_end = start + 0.05
            # Last, so nothing added above can reach past the final frame. A
            # word that begins after the clip ends has nothing to show at all.
            if limit is not None:
                if start >= limit:
                    continue
                capped_end = min(capped_end, limit)
            end_time = format_time_ass(capped_end)

            # Only words up to the current one â€” no future-word spoilers.
            text_parts = []
            for w in chunk[: i + 1]:
                clean_word = w["word"].strip()
                if w is target_word:
                    text_parts.append(
                        f"{{\\c{highlight_color}&}}{clean_word}{{\\c{text_color}&}}"
                    )
                else:
                    text_parts.append(clean_word)

            full_text = " ".join(text_parts)
            events.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{full_text}")

    with open(output_ass, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write("\n".join(events))
    return output_ass

def generate_ass_from_transcript(transcript: dict, start: float, end: float, output_ass: str, style=None) -> str:
    """Build a clip .ass by SLICING a precomputed VOD transcript (no re-transcription).

    Reuses the single VOD-level Whisper pass (handbook/20 Â§1.1): pick words inside
    [start, end] and rebase their timestamps to clip-relative (the exported clip
    starts at 0), then render with the existing TikTok-style formatter.
    """
    clip_words = []
    for segment in transcript.get("segments", []):
        # Whisper emits no-speech segments whose "words" is None; iterating
        # that raised and the caller's catch-all shipped the clip with NO
        # captions at all ("sometimes it's not getting the words").
        for w in segment.get("words") or []:
            ws, we = float(w.get("start", 0.0)), float(w.get("end", 0.0))
            if we < start or ws > end:
                continue
            clip_words.append({
                "word": w.get("word", ""),
                "start": max(0.0, ws - start),
                "end": max(0.0, we - start),
            })
    # One synthetic segment; generate_tiktok_ass chunks words into groups of 4.
    # Returns None (no file written) when the caption style is disabled.
    return generate_tiktok_ass({"segments": [{"words": clip_words}]}, output_ass,
                               style=style, duration=max(0.0, float(end) - float(start)))


def transcribe_clip_window(source_audio: str, start: float, end: float,
                           temp_audio: str, cancel_check=None, settings=None,
                           speaker_profile=None) -> dict:
    """Per-clip ASR for one window. Returns a CLIP-RELATIVE transcript.

    Split out of generate_ass_for_clip so the burned-in captions and the title
    quote are produced from the SAME decode. They used to disagree: captions
    came from this pass and the title from the VOD-level pass, so a clip could
    show one wording on screen and another on its card.

    ``speaker_profile`` (engines.caption.speaker_id) drops segments spoken by
    someone other than the streamer -- a friend in party chat, or the narration
    of a video being reacted to. Filtering happens here, while the clip's audio
    still exists, so both surfaces inherit it.
    """
    try:
        if cancel_check:
            cancel_check()
        extract_clip_audio(source_audio, start, end, temp_audio)
        # Captions re-transcribe each clip with medium.en so review previews
        # match export quality (word accuracy + alignment heads). The VOD-level
        # medium.en pass still feeds hype / semantic; slicing it is only a
        # fallback when per-clip ASR fails.
        #
        # VAD is deliberately OFF: faster-whisper's Silero VAD over-trims these
        # short clips and can drop real speech. Missing
        # captions are worse than a slightly-early one, and the alignment model
        # alone already carries the timing win.
        model = get_whisper_model(resolve_caption_model_size(settings))
        # Match VOD ASR: pin English and disable cross-segment conditioning so
        # short clip audio can't drift into hallucinated / wrong words. The
        # domain prompt is what conditioning cannot supply here -- and unlike on
        # the VOD pass it actually reaches this audio (see STREAM_ASR_PROMPT).
        result = transcribe_with_cancel(
            model,
            temp_audio,
            cancel_check=cancel_check,
            word_timestamps=True,
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=STREAM_ASR_PROMPT,
        )
        if speaker_profile is not None:
            from engines.caption import speaker_id
            result = speaker_id.filter_words_to_streamer(
                result, speaker_id.wav_reader(temp_audio), speaker_profile,
                duration=max(0.0, float(end) - float(start)))
        return result
    finally:
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass


def clip_transcript_to_vod_time(transcript: dict, start: float) -> dict:
    """Rebase a clip-relative transcript onto VOD time.

    generate_ass_from_transcript slices the other way (VOD -> clip). The title
    quote is picked with VOD-time bounds, so a per-clip decode has to come back
    the same distance before _extract_quote can read it.
    """
    segments = []
    for segment in transcript.get("segments", []) or []:
        shifted = dict(segment)
        shifted["start"] = float(segment.get("start", 0.0)) + start
        shifted["end"] = float(segment.get("end", 0.0)) + start
        shifted["words"] = [
            dict(word,
                 start=float(word.get("start", 0.0)) + start,
                 end=float(word.get("end", 0.0)) + start)
            for word in (segment.get("words") or [])
        ]
        segments.append(shifted)
    return {"segments": segments}


def generate_ass_for_clip(source_audio: str, start: float, end: float, output_ass: str,
                          cancel_check=None, style=None, settings=None,
                          transcript: dict = None) -> str:
    """Render one clip's .ass, transcribing it first unless ``transcript`` is given.

    ``transcript`` is a clip-relative result from transcribe_clip_window, passed
    in by callers that also want the same decode for the title.
    """
    if transcript is None:
        transcript = transcribe_clip_window(
            source_audio, start, end, output_ass.replace(".ass", ".wav"),
            cancel_check=cancel_check, settings=settings,
        )
    return generate_tiktok_ass(transcript, output_ass, style=style,
                               duration=max(0.0, float(end) - float(start)))
