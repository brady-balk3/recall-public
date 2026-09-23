# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Qwen3-ASR as Recall's transcription backend, replacing faster-whisper.

`medium.en` is a 2023 model and it mishears this workload in ways a 2026 model
does not -- "Vidy screamers always have V-Bucks" for "Why do streamers always
have V Bucks?", "it's super sleigh" for "slay", "do it up" for "duo it up", and
no punctuation or casing at all. Measured across five VODs and four games, the
Skyrim VOD disagreed 0/10 and the rest 3-4/10, with Qwen right nearly every time.

Whisper supplied three things, and all three are replaced here:

* **words** -- Qwen3-ASR-1.7B through the llama.cpp already installed. The
  bundled mtmd carries `clip_graph_qwen3a`, so no second interpreter and no
  `transformers` (the upstream `qwen-asr` package pins `transformers==4.57.6`
  and pulls in accelerate, librosa, sox, gradio and flask -- none of which
  belongs in a shipped desktop app);
* **word timings** -- `qwen_aligner`, the ONNX forced aligner, measured at
  0.040s median against whisper on 1563 words across four games. Qwen3-ASR
  emits plain text with no timestamp tokens, so this is not optional;
* **chunk edges** -- Silero VAD via `onnx-asr`. Qwen3-ASR is not a long-form
  model: audio is cut into chunks, and the cuts are placed in silence the VAD
  found, so a chunk edge can never split a word. That is why there is no
  overlap window and no seam de-duplication anywhere in this file.

The output is the same dict every consumer already reads --
``{"segments": [{id, start, end, text, words: [{word, start, end}]}]}`` -- so
`vod_transcribe`, `asr_hype`, `sentence_index`, `judge`, `caption_generator` and
`generate_ass_for_clip` are untouched. `whisper_asr.get_whisper_model()` hands
them this adapter; it is the only backend, the faster-whisper rung having been
removed on 2026-09-01.

Silence is skipped rather than transcribed: a chunk spans from its first speech
region to its last, so a quiet stretch costs nothing.
"""

from __future__ import annotations

import collections
import os
import re
import subprocess
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.bundle_paths import (
    get_qwen_aligner_dir,
    get_qwen_asr_dir,
    get_silero_vad_dir,
)
from core.device import get_ort_providers, get_torch_device
from core.ffmpeg_path import get_ffmpeg_path
from engines.caption.qwen_aligner import QwenForcedAligner, split_words

SAMPLE_RATE = 16000

# Chunk length. The floor is a quality question, not a performance one: Qwen
# reads a phrase better with the sentence around it, and an 18s window is the
# probe's artifact rather than a target. Two hard ceilings sit far above the cap
# -- n_ctx (13 audio positions per second) and the aligner's 5000 frame classes
# at 80 ms, i.e. 400s per alignment call -- so neither is what binds. VRAM is.
CHUNK_TARGET_SEC = 60.0
# 90 rather than 120. MEASURED 2026-08-19, aligner VRAM above a ~4.2 GB floor
# (fp32 weights + CUDA context) against chunk length:
#     30s 4271 MiB (59x)   60s 4640 MiB (136x)
#     90s 5193 MiB (126x)  120s 7340 MiB (121x)
# Attention is quadratic in sequence length, and a 120s chunk is ~3000 tokens,
# so the last 30 seconds cost 2.1 GB and bought no speed at all. That 2.1 GB is
# what the VLM judge needs on a 16 GB card.
CHUNK_MAX_SEC = 90.0

# Silence wide enough to place a cut inside with room to spare on both sides.
MIN_CUT_SILENCE_SEC = 0.4

# A gap this long always ends a chunk, however little speech came before it.
# Without this rule a short region followed by a long quiet stretch never
# reaches the length that would justify a cut, so the silence gets carried into
# the chunk -- and a VOD with a five-minute AFK becomes five minutes of GPU
# spent transcribing nothing. Short pauses inside a sentence stay carried,
# because Qwen reads a phrase better with its own pauses intact.
MAX_CARRY_SILENCE_SEC = 2.0

# VAD runs over the file in blocks so a four-hour VOD never becomes a 14 GB
# float array. 10 minutes is 38 MB.
VAD_BLOCK_SEC = 600.0

# A pause this long ends a transcript segment, matching the caption chunker's
# own pacing constant (whisper_asr.CHUNK_GAP_SEC).
SEGMENT_GAP_SEC = 0.9
SEGMENT_MAX_WORDS = 30

_SENTENCE_END = re.compile(r"[.!?]['\"]?$")
_MODEL_LOCK = threading.RLock()


# ---------------------------------------------------------------- audio input


def audio_duration(path: str) -> float:
    """Seconds of audio, via ffmpeg's own decoder rather than a wav header.

    Callers hand this module both the cached 16 kHz analysis WAV and temp files
    cut per clip; probing rather than assuming keeps one code path for both.
    """
    result = subprocess.run(
        [get_ffmpeg_path(), "-hide_banner", "-i", path, "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    matches = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", result.stderr or "")
    if not matches:
        return 0.0
    hours, minutes, seconds = matches[-1]
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_slice(path: str, start: float, span: float) -> np.ndarray:
    """Mono float32 at 16 kHz for ``span`` seconds from ``start``."""
    if span <= 0:
        return np.zeros(0, dtype=np.float32)
    result = subprocess.run(
        [get_ffmpeg_path(), "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{span:.3f}", "-i", path,
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    raw = np.frombuffer(result.stdout, dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


# ------------------------------------------------------------------ VAD pass


def speech_regions(path: str, duration: float,
                   cancel_check=None) -> List[Tuple[float, float]]:
    """Speech spans in file time, from Silero.

    This is what whisper's segment list used to provide. Blocks are processed
    independently and then merged at the seams, so a phrase straddling a block
    boundary comes back as one region rather than two.
    """
    import onnx_asr

    # Preload torch's CUDA/cuDNN before onnxruntime builds a session, or the
    # CUDA provider fails with "cublasLt64_12.dll is missing" and silently
    # drops to CPU (core/device.py owns this).
    get_ort_providers()
    # onnx-asr ships the loader, not the Silero weights. Frozen builds pass the
    # required bundled directory so a first scan never reaches Hugging Face;
    # dev keeps None and uses onnx-asr's normal local cache resolution.
    vad = onnx_asr.load_vad("silero", path=get_silero_vad_dir())

    regions: List[Tuple[float, float]] = []
    offset = 0.0
    while offset < duration:
        if cancel_check:
            cancel_check()
        span = min(VAD_BLOCK_SEC, duration - offset)
        block = read_slice(path, offset, span)
        if block.size:
            found = next(vad.segment_batch(
                block[None, :].astype(np.float32),
                np.asarray([block.size], dtype=np.int64), SAMPLE_RATE))
            for begin, end in found:
                regions.append((offset + begin / SAMPLE_RATE,
                                offset + end / SAMPLE_RATE))
        offset += span

    merged: List[Tuple[float, float]] = []
    for begin, end in sorted(regions):
        if merged and begin - merged[-1][1] < 0.1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return merged


def plan_chunks(regions: Sequence[Tuple[float, float]],
                target: float = CHUNK_TARGET_SEC,
                cap: float = CHUNK_MAX_SEC) -> List[Tuple[float, float]]:
    """Group speech regions into chunks that begin and end in silence.

    A chunk runs from the start of its first region to the end of its last, so
    silence between regions is carried (Qwen wants the pauses inside a sentence)
    but silence around the chunk is not (nobody needs a transcript of nothing).
    A run of speech with no gap wide enough to cut in is split at ``cap``, which
    is the only case where a word can land on an edge.
    """
    chunks: List[Tuple[float, float]] = []
    if not regions:
        return chunks

    begin = regions[0][0]
    last = regions[0][1]
    for index in range(1, len(regions) + 1):
        at_end = index == len(regions)
        span = last - begin
        gap = (regions[index][0] - last) if not at_end else 0.0

        if at_end:
            chunks.append((begin, last))
            break
        if span >= cap or gap >= MAX_CARRY_SILENCE_SEC:
            chunks.append((begin, last))
            begin, last = regions[index]
            continue
        if span >= target and gap >= MIN_CUT_SILENCE_SEC:
            chunks.append((begin, last))
            begin, last = regions[index]
            continue
        last = regions[index][1]
    return _enforce_cap(chunks, cap)


def _enforce_cap(chunks: Sequence[Tuple[float, float]],
                 cap: float) -> List[Tuple[float, float]]:
    """Hard-split anything still over ``cap`` into equal parts.

    The silence-seeking loop above cannot always honour the cap: a single
    unbroken speech region longer than it has no gap to cut in, so somebody
    talking for three minutes without a 0.4s pause would otherwise produce a
    chunk that overruns n_ctx and the aligner's 400s ceiling both.

    Equal parts rather than cap-sized parts plus a remainder: splitting 149s
    into 75+74 puts one cut in the middle of a word, where 120+29 puts one cut
    in the middle of a word AND leaves a 29s fragment with no context around it.
    """
    out: List[Tuple[float, float]] = []
    for begin, end in chunks:
        span = end - begin
        if span <= cap:
            out.append((begin, end))
            continue
        parts = int(span // cap) + 1
        width = span / parts
        for index in range(parts):
            out.append((begin + width * index,
                        begin + width * (index + 1) if index < parts - 1 else end))
    return out


# ------------------------------------------------------------- context biasing

_PROPER_NOUN = re.compile(r"\b([A-Z][a-zA-Z-]{2,})\b")

# Words that reach the top of a frequency list purely by starting a sentence.
# Qwen capitalises after punctuation, so interrogatives, greetings and
# interjections would otherwise crowd out the names this is trying to find.
_STOPWORDS = frozenset("""
The This That There Then They Their These Those And But For You Your It Its Is
Was Were We He She His Her Him Not No Yes Oh Okay Ok So If When What Why How Who
All Just Like Well Yeah Hey Let Can Do Don Get Got Go Going Gonna Come Now One
Two Three Because Right Actually Really Maybe Alright Some Every Never Nice
Thank Thanks Please Sorry Wait Look See Know Think Did Are Where Should Someone
Somebody Sometimes Ooh Hold Hello Here Also Have Stop Guys God Good Wow Yay Yep
Nothing Anyways Cause Very Does Ask Welcome Damn Honestly Whoa Mmm Isn Sit Bro
Dude Been Give Make Take Want Need Said Say Even Something Anything Everything
Nobody Everybody Only
""".split())


class ContextBuilder:
    """Per-scan biasing text, derived from the scan itself.

    Qwen3-ASR biases recognition on arbitrary background text, and this is what
    fixes proper nouns: without it the model heard "Elf" for the squadmate
    "Alpha". The probe mined those names from the VOD's own whisper transcript,
    which is not available any more, so this bootstraps from the chunks already
    transcribed in THIS run -- the first chunk gets the caller's prompt alone,
    and every chunk after it also gets the names the model has been producing.

    Self-correcting in aggregate for the same reason the whisper version was: a
    real name recurs across a stream, a one-off mishear does not.
    """

    def __init__(self, prompt: Optional[str] = None, limit: int = 28,
                 minimum: int = 3):
        self.prompt = (prompt or "").strip()
        self.limit = limit
        self.minimum = minimum
        self._counts: collections.Counter = collections.Counter()

    def observe(self, text: str) -> None:
        for token in _PROPER_NOUN.findall(text or ""):
            if token not in _STOPWORDS:
                self._counts[token] += 1

    def context(self) -> str:
        names = [word for word, count in self._counts.most_common(self.limit)
                 if count >= self.minimum]
        parts = [part for part in (self.prompt, " ".join(names)) if part]
        return ". ".join(parts).strip()


# ------------------------------------------------------------------ the model


def _audio_chat_handler():
    """MTMDChatHandler that accepts an AUDIO-only projector.

    Two things in the stock handler assume vision:

    * ``_init_mtmd_context`` asserts ``mtmd_support_vision`` and raises "Vision
      is not supported by this model". The bitmap path itself is
      modality-agnostic -- ``mtmd_helper_bitmap_init_from_buf`` sniffs the
      buffer and builds an audio chunk from WAV bytes -- so the assertion is
      the only obstacle, and it is replaced with the matching audio check.
    * the chat template is rendered with content as a LIST of parts, which
      Qwen3-ASR's template cannot index ("can only concatenate str (not list)
      to str"). Flattening to a string with the media marker inline fixes it.
    """
    from llama_cpp._utils import suppress_stdout_stderr
    from llama_cpp.llama_chat_format import MTMDChatHandler

    class AudioMTMDChatHandler(MTMDChatHandler):
        def _init_mtmd_context(self, llama_model):
            self.verbose = llama_model.verbose
            if self.mtmd_ctx is not None:
                return
            with suppress_stdout_stderr(disable=self.verbose):
                params = self._mtmd_cpp.mtmd_context_params_default()
                params.use_gpu = self.use_gpu
                params.print_timings = self.verbose
                params.n_threads = llama_model.n_threads
                params.flash_attn_type = llama_model.context_params.flash_attn_type
                self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
                    self.clip_model_path.encode(), llama_model.model, params)
                if self.mtmd_ctx is None:
                    raise ValueError(
                        f"Failed to load mtmd context from: {self.clip_model_path}")
                if not self._mtmd_cpp.mtmd_support_audio(self.mtmd_ctx):
                    raise ValueError("Audio is not supported by this projector")

                def mtmd_free():
                    with suppress_stdout_stderr(disable=self.verbose):
                        if self.mtmd_ctx is not None:
                            self._mtmd_cpp.mtmd_free(self.mtmd_ctx)
                            self.mtmd_ctx = None

                self._exit_stack.callback(mtmd_free)

        @classmethod
        def _convert_message_for_template(cls, message, media_marker):
            data = dict(message)
            content = data.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        parts.append(str(part))
                    elif part.get("type") == "image_url":
                        parts.append(media_marker)
                    else:
                        parts.append(str(part.get("text", "")))
                data["content"] = "".join(parts)
            return data

    return AudioMTMDChatHandler


# Qwen3-ASR covers 52 languages and occasionally answers in the wrong one. On
# unclear filler it falls back to Chinese: MEASURED across five VODs it emitted
# 10 such tokens (0.004-0.021% of words), almost all the filler "嗯。",
# plus "我妈哭了。" and "刷。" -- and one of them landed inside an exported
# clip, where a burned-in caption would render it as a glyph nobody can read.
#
# Recall's speech lane is English-first and says so (run_pipeline emits a notice
# when a VOD's detected language is not English), so a token written in a script
# the workload never uses is an artifact, not content. Dropped rather than
# transliterated: there is nothing to recover, and leaving it would put it in
# the transcript the judge and the title quote read as well as on screen.
#
# Deliberately narrow. Only CJK ideographs, kana, hangul and their fullwidth
# punctuation match -- accented Latin ("cafe", "naive"), emoji and symbols are
# untouched, because those DO occur in this workload and are not a failure mode.
_CJK = re.compile(
    "["
    "　-〿"      # CJK punctuation, incl. the trailing full stop
    "぀-ヿ"      # hiragana + katakana
    "㐀-䶿"      # CJK extension A
    "一-鿿"      # CJK unified ideographs
    "가-힯"      # hangul syllables
    "＀-￯"      # fullwidth forms
    "]"
)


# Chinese is what this workload actually produced, but the model speaks 52
# languages and the failure is "answered in the wrong one", not "answered in
# Chinese". Enumerating CJK ranges would leave Cyrillic, Greek, Arabic, Thai and
# the rest to be discovered one incident at a time, so the rule is the general
# one: a LETTER from outside the Latin block means the token is not English.
#
# 0x24f is the end of Latin Extended-B, which keeps every accent this workload
# legitimately uses -- cafe, naive, Bjorn, Timea all sit below it. Emoji and
# symbols are category So/Sk, never L, so they are untouched.
_LATIN_MAX = 0x24F


def _is_foreign(token: str) -> bool:
    import unicodedata
    for char in token:
        if _CJK.match(char):
            return True          # CJK punctuation carries no letter of its own
        if unicodedata.category(char).startswith("L") and ord(char) > _LATIN_MAX:
            return True
    return False


def drop_foreign_script(text: str) -> str:
    """Remove whitespace tokens written in a non-Latin script.

    Whole tokens go, not just the offending characters: a token that is part
    Chinese is not a salvageable English word, and stripping mid-token would
    leave a fragment for the aligner to time.
    """
    if not text:
        return text
    return " ".join(token for token in text.split() if not _is_foreign(token))


def _wav_bytes(audio: np.ndarray) -> bytes:
    """16-bit PCM WAV in memory -- mtmd sniffs the bytes to build an audio chunk."""
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(
            np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


def _model_files() -> Tuple[str, str]:
    directory = get_qwen_asr_dir()
    backbone = os.path.join(directory, "Qwen3-ASR-1.7B-Q8_0.gguf")
    projector = os.path.join(directory, "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
    for path in (backbone, projector):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Qwen3-ASR weights missing: {path}. "
                "Run scripts/fetch_qwen_asr.py. There is no fallback backend: "
                "a packaged build cannot be produced without these weights, so "
                "reaching this in one means the install is damaged.")
    return backbone, projector


def _context_size(seconds: float) -> int:
    """n_ctx for the longest chunk: 13 audio positions a second, plus the reply."""
    budget = int(seconds * 13.0 * 1.4) + 1024
    size = 4096
    while size < budget:
        size *= 2
    return size


class QwenASRAdapter:
    """Qwen3-ASR + forced aligner behind openai-whisper's transcribe() contract.

    Written to openai-whisper's result-dict shape because every consumer already
    read it -- `vod_transcribe`, `asr_hype`, the sentence index, the judge and
    captions are all unchanged by which model produced the words. The contract
    outlived the backend it was named for.
    """

    def __init__(self, chunk_target: float = CHUNK_TARGET_SEC,
                 chunk_max: float = CHUNK_MAX_SEC):
        from llama_cpp import Llama

        backbone, projector = _model_files()
        self.chunk_target = chunk_target
        self.chunk_max = chunk_max

        self._handler = _audio_chat_handler()(
            clip_model_path=projector, verbose=False)
        on_gpu = get_torch_device() == "cuda"
        self._llm = Llama(
            model_path=backbone, chat_handler=self._handler,
            n_ctx=_context_size(chunk_max),
            n_gpu_layers=-1 if on_gpu else 0,
            n_threads=max(2, (os.cpu_count() or 4) // 2), verbose=False,
        )
        self._llm.set_cache(None)
        self._aligner = QwenForcedAligner(get_qwen_aligner_dir(), use_gpu=on_gpu)
        print(f"ASR backend: Qwen3-ASR-1.7B (llama.cpp, "
              f"{'cuda' if on_gpu else 'cpu'}) + forced aligner "
              f"({self._aligner.providers[0]})")

        if self.chunk_max > self._aligner.max_seconds:
            raise ValueError(
                f"chunk cap {self.chunk_max:.0f}s exceeds the aligner's "
                f"{self._aligner.max_seconds:.0f}s ceiling")

    # -- one chunk ---------------------------------------------------------

    def _words_for_chunk(self, audio: np.ndarray, context: str) -> Tuple[str, List[Dict]]:
        """Transcribe one chunk and time its words. Returns (text, words)."""
        text = self._transcribe_chunk(audio, context)
        if not text.strip():
            return "", []

        # The aligner strips punctuation to time words, but the transcript's
        # text must keep it -- the judge and the title quote read prose, not
        # tokens. Both sequences are built from the same split, so raw token i
        # and aligned word i are the same word.
        raw = [token for token in text.split() if split_words(token)]
        aligned = self._aligner.align(audio, text)
        words: List[Dict] = []
        for index, timing in enumerate(aligned):
            if index >= len(raw):
                break
            words.append({
                # Leading space matches whisper's convention, which the caption
                # renderer and _extract_quote both assume when joining.
                "word": " " + raw[index],
                "start": float(timing["start"]),
                "end": float(timing["end"]),
            })
        return text, words

    def _transcribe_chunk(self, audio: np.ndarray, context: str) -> str:
        import base64

        url = ("data:audio/wav;base64,"
               + base64.b64encode(_wav_bytes(audio)).decode("ascii"))
        messages = []
        if context:
            messages.append({"role": "system",
                             "content": "Background for this recording: " + context})
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Transcribe the audio verbatim."},
        ]})
        seconds = audio.size / SAMPLE_RATE
        result = self._llm.create_chat_completion(
            messages=messages, temperature=0.0,
            # Roughly twice this workload's speech rate, so a long chunk is
            # never cut short by the token budget.
            max_tokens=max(512, int(seconds * 4 * 1.4)),
        )
        text = str(result["choices"][0]["message"]["content"] or "").strip()
        # Qwen3-ASR emits a structured header, e.g. `language English<asr_text>`.
        if "<asr_text>" in text:
            text = text.split("<asr_text>", 1)[1]
        text = text.replace("</asr_text>", "").strip()
        from engines.caption.number_words import digitize
        from engines.caption.slang_words import correct_slang
        # Before alignment, so the word list and the aligner see the same
        # tokens -- stripping afterwards would shift every index against the
        # timings the head produced. correct_slang is one-token-in-one-token-out
        # for the same reason, so it can sit anywhere in this chain.
        return drop_foreign_script(correct_slang(digitize(text)))

    # -- segments ----------------------------------------------------------

    @staticmethod
    def _segments(words: Iterable[Dict]) -> List[Dict]:
        """Group timed words into transcript segments.

        Breaks on a pause, on sentence-final punctuation, or on length. Qwen
        punctuates, which whisper did not, so these boundaries are better than
        the ones the sentence index used to get -- but boundary snapping reads
        word timings either way, so nothing downstream depends on the choice.
        """
        segments: List[Dict] = []
        current: List[Dict] = []

        def flush():
            if not current:
                return
            segments.append({
                "id": len(segments),
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
                "text": "".join(word["word"] for word in current),
                "words": list(current),
            })
            current.clear()

        previous_end = None
        for word in words:
            if current and previous_end is not None:
                gap = float(word["start"]) - previous_end
                ended = _SENTENCE_END.search(current[-1]["word"].strip())
                if (gap >= SEGMENT_GAP_SEC
                        or (ended and len(current) >= 3)
                        or len(current) >= SEGMENT_MAX_WORDS):
                    flush()
            current.append(word)
            previous_end = float(word["end"])
        flush()
        return segments

    # -- the contract ------------------------------------------------------

    def transcribe(self, audio_path, word_timestamps=True, language=None,
                   condition_on_previous_text=True, cancel_check=None,
                   progress_callback=None, vad_filter=False,
                   initial_prompt=None, **_ignored) -> Dict:
        """Transcribe a whole file. Returns the openai-whisper result dict.

        ``initial_prompt`` becomes the biasing context rather than a decoder
        prefix -- the same intent (tell the model what this recording is about),
        expressed the way Qwen3-ASR accepts it. Unlike whisper's, it is useful
        on long audio as well as short.

        ``word_timestamps``, ``condition_on_previous_text`` and ``vad_filter``
        are accepted and ignored: words are always timed, chunks never condition
        on each other, and the VAD is not optional here.
        """
        _ = (word_timestamps, condition_on_previous_text, vad_filter)
        if cancel_check:
            cancel_check()

        duration = audio_duration(audio_path)
        regions = speech_regions(audio_path, duration, cancel_check=cancel_check)
        chunks = plan_chunks(regions, self.chunk_target, self.chunk_max)

        builder = ContextBuilder(initial_prompt)
        words: List[Dict] = []
        for begin, end in chunks:
            if cancel_check:
                cancel_check()
            audio = read_slice(audio_path, begin, end - begin)
            if not audio.size:
                continue
            try:
                text, timed = self._words_for_chunk(audio, builder.context())
            except Exception as exc:  # noqa: BLE001 - one chunk must not lose the VOD
                print(f"Qwen ASR chunk {begin:.0f}-{end:.0f}s failed: "
                      f"{type(exc).__name__}: {exc}")
                continue
            builder.observe(text)
            for word in timed:
                word["start"] += begin
                word["end"] += begin
            words.extend(timed)
            if progress_callback:
                progress_callback(min(end, duration), duration)

        if progress_callback and duration:
            progress_callback(duration, duration)
        # After alignment, not before: join_maxxing turns two tokens into one
        # and may only do that once the timings exist (see slang_words). The
        # merged word spans both, and _segments rebuilds its text from this
        # list, so the transcript and the word timings stay consistent.
        from engines.caption.slang_words import join_maxxing
        return {"segments": self._segments(join_maxxing(words)),
                "language": language or "en"}

    def close(self) -> None:
        with _MODEL_LOCK:
            aligner, self._aligner = getattr(self, "_aligner", None), None
            handler, self._handler = getattr(self, "_handler", None), None
            llm, self._llm = getattr(self, "_llm", None), None
        if aligner is not None:
            aligner.close()
        try:
            stack = getattr(handler, "_exit_stack", None)
            if stack is not None:
                stack.close()
        finally:
            if llm is not None:
                llm.close()
