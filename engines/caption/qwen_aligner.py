# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Word timings from Qwen3-ForcedAligner, through ONNX, with no new dependency.

This is the piece that lets whisper leave. Qwen3-ASR reads the words better but
emits plain text; every timing consumer in Recall (progressive karaoke captions,
the sentence index boundary snapping cuts on, the hype channel, the judge's
window queries) needs a start and an end per word. The aligner supplies them
from audio + the transcript, in one non-autoregressive forward pass.

It is driven directly rather than through the upstream `qwen-asr` package, which
pins `transformers==4.57.6` and pulls in accelerate, librosa, sox, gradio and
flask -- none of which belongs in a shipped desktop app. What the package
actually does to the tensors is small, and all four pieces are already here:

* mel features -- the preprocessor is literally `WhisperFeatureExtractor`
  (128 bins, n_fft 400, hop 160, 16 kHz), so `openai-whisper`'s own
  `log_mel_spectrogram` produces them bit for bit, including the log clamp and
  the ``(x + 4) / 4`` scale;
* tokenizer -- `tokenizers` loads the export's `tokenizer.json`;
* inference -- `onnxruntime-gpu`, already shipped for `hsemotion-onnx`;
* the algorithm below, reimplemented from the Apache-2.0 reference
  (`qwen_asr.inference.qwen3_forced_aligner`).

The contract, which is not guessable and was read off that reference:

    "<|audio_start|>" + "<|audio_pad|>" * n_audio + "<|audio_end|>"
    + "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"

Two `<timestamp>` slots per word, start then end. The head emits, at every text
position, a distribution over 5000 frame indices; ``argmax`` at the timestamp
slots is the frame, and frame x 80 ms is the time. 5000 frames is a hard
**400-second ceiling** on one call -- chunk below it.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Frame time and vocabulary come from the export's own config rather than being
# hardcoded, so a re-export with different settings fails loudly instead of
# silently producing times that are wrong by a constant factor.
_DEFAULT_FRAME_MS = 80.0

AUDIO_START = "<|audio_start|>"
AUDIO_PAD = "<|audio_pad|>"
AUDIO_END = "<|audio_end|>"
TIMESTAMP = "<timestamp>"


def audio_token_count(mel_frames: int) -> int:
    """How many audio positions ``mel_frames`` of mel collapse to.

    Two stride-2 convolutions and then the encoder's own pooling, which the
    reference expresses per whole second (13 positions per 100 mel frames, i.e.
    per second at hop 160) plus the ragged remainder. Reimplemented exactly:
    getting this off by one shifts every word in the chunk.
    """
    leave = mel_frames % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel_frames // 100) * 13


def _is_kept(char: str) -> bool:
    """Letters, digits and the apostrophe survive; punctuation does not."""
    if char == "'":
        return True
    import unicodedata
    category = unicodedata.category(char)
    return category.startswith("L") or category.startswith("N")


def split_words(text: str) -> List[str]:
    """Whitespace split, then strip everything that is not a letter or digit.

    Matches the reference's space-language tokenizer. Punctuation has no
    duration, and leaving it attached would ask the head to time a token that
    was never spoken.
    """
    words: List[str] = []
    for chunk in (text or "").split():
        cleaned = "".join(c for c in chunk if _is_kept(c))
        if cleaned:
            words.append(cleaned)
    return words


def spread_ties(frames: Sequence[int]) -> List[int]:
    """Spread runs of identical frames evenly across the span they occupy.

    MEASURED 2026-08-18: on a window of "Wait a minute" repeated thirteen times,
    the head put eight consecutive words on frame 78 (6.24s). Repeated phrasing
    is where forced alignment has the least to work with -- every repetition
    looks the same, so nothing in the text says which one is which -- and the
    monotonic repair above cannot help, because a tie IS non-decreasing.

    Eight words sharing one start is a caption defect: a progressive renderer
    pops them on screen together. So this exists -- but it is OFF by default,
    because the argument for it did not survive measurement. On the same VOD's
    six windows it is marginally WORSE against whisper (within 80 ms 0.589 ->
    0.576, p90 1.04 -> 1.10s), which says whisper's own times in a repeated
    passage are no more trustworthy than the aligner's, so "spread them out"
    does not move them toward truth.

    Revisit at Phase 3, where captions are actually rendered and the question
    becomes watchable instead of arithmetic. Do not turn it on before then on
    the strength of the rendering argument alone.
    """
    data = [int(f) for f in frames]
    count = len(data)
    i = 0
    while i < count:
        j = i
        while j + 1 < count and data[j + 1] == data[i]:
            j += 1
        run = j - i + 1
        if run > 2:
            # Stop at the next distinct frame, or extend by the run's own width
            # when the tie runs to the end and there is nothing to stop at.
            ceiling = data[j + 1] if j + 1 < count else data[i] + run
            step = (ceiling - data[i]) / run
            if step > 0:
                for k in range(1, run):
                    data[i + k] = int(round(data[i] + step * k))
        i = j + 1
    return data


def repair_monotonic(frames: Sequence[float]) -> List[int]:
    """Force the frame sequence to be non-decreasing, interpolating the rest.

    The head predicts each slot independently, so nothing stops it emitting a
    time that runs backwards. The reference's fix, reimplemented: keep the
    longest non-decreasing subsequence as trustworthy, and rebuild every run
    outside it from its neighbours -- short runs snap to whichever side is
    nearer, longer runs are spread evenly across the gap.

    Doing this is not cosmetic. A caption renderer given a word that starts
    before the previous one ended will either flicker or drop it.
    """
    data = [float(f) for f in frames]
    count = len(data)
    if count == 0:
        return []

    # Longest non-decreasing subsequence, O(n^2) -- n is words-per-chunk x 2,
    # a few hundred, so the simple version is not worth improving.
    length = [1] * count
    parent = [-1] * count
    for i in range(1, count):
        for j in range(i):
            if data[j] <= data[i] and length[j] + 1 > length[i]:
                length[i] = length[j] + 1
                parent[i] = j
    best = max(range(count), key=lambda i: length[i])
    trusted = [False] * count
    index = best
    while index != -1:
        trusted[index] = True
        index = parent[index]

    result = list(data)
    i = 0
    while i < count:
        if trusted[i]:
            i += 1
            continue
        j = i
        while j < count and not trusted[j]:
            j += 1

        left = next((result[k] for k in range(i - 1, -1, -1) if trusted[k]), None)
        right = next((result[k] for k in range(j, count) if trusted[k]), None)

        if left is None and right is None:
            i = j
            continue
        if left is None:
            for k in range(i, j):
                result[k] = right
        elif right is None:
            for k in range(i, j):
                result[k] = left
        elif j - i <= 2:
            # One or two stray slots: snap each to the nearer trusted anchor
            # rather than inventing a slope from two points.
            for k in range(i, j):
                result[k] = left if (k - (i - 1)) <= (j - k) else right
        else:
            step = (right - left) / (j - i + 1)
            for k in range(i, j):
                result[k] = left + step * (k - i + 1)
        i = j
    return [int(round(value)) for value in result]


class QwenForcedAligner:
    """Qwen3-ForcedAligner-0.6B over onnxruntime.

    ``model_dir`` is what ``scripts/fetch_qwen_aligner.py`` writes.
    """

    def __init__(self, model_dir: str, use_gpu: bool = True,
                 variant: Optional[str] = None, spread: bool = False):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = model_dir
        self.spread = spread
        path = self._resolve_model(model_dir, variant)
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        self.frame_ms = float(config.get("timestamp_segment_time")
                              or _DEFAULT_FRAME_MS)

        self.tokenizer = Tokenizer.from_file(
            os.path.join(model_dir, "tokenizer.json"))
        self._special = {
            name: self._token_id(name)
            for name in (AUDIO_START, AUDIO_PAD, AUDIO_END, TIMESTAMP)
        }
        # The config's own id is the authority; the tokenizer file agreeing with
        # it is what makes the timestamp mask correct.
        declared = config.get("timestamp_token_id")
        if declared is not None and int(declared) != self._special[TIMESTAMP]:
            raise ValueError(
                f"timestamp token id mismatch: config {declared}, "
                f"tokenizer {self._special[TIMESTAMP]}")

        # core.device is the authority on providers, not this file. Naive
        # provider selection picks CUDA and then fails at session create with
        # "cublasLt64_12.dll is missing" -- onnxruntime-gpu needs CUDA/cuDNN on
        # the loader path, and the repo already solves that by preloading
        # torch's bundled cu126 DLLs. MEASURED: without it, CUDA EP does not
        # load here at all.
        if use_gpu:
            from core.device import get_ort_providers
            providers = list(get_ort_providers())
        else:
            providers = ["CPUExecutionProvider"]

        # No arena tuning here on purpose. kSameAsRequested was MEASURED
        # 2026-08-19 against the kNextPowerOfTwo default and made no difference
        # (7531 vs 7375 MiB on a 120s chunk), because the footprint is real
        # activation memory rather than allocator rounding. Chunk LENGTH is the
        # lever that works -- see CHUNK_MAX_SEC in qwen_asr.
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(path, options, providers=providers)
        self.providers = self.session.get_providers()
        self.max_frames = self._class_count()
        # The export is not uniform about integer width -- input_ids and
        # attention_mask are int64 while feature_attention_mask is int32 -- and
        # onnxruntime rejects a mismatch outright rather than casting. Read the
        # widths off the graph so a re-export cannot silently break this.
        self._dtypes = {
            entry.name: self._numpy_dtype(entry.type)
            for entry in self.session.get_inputs()
        }

    @staticmethod
    def _resolve_model(model_dir: str, variant: Optional[str]) -> str:
        names = [variant] if variant else ["model.onnx", "model_q4.onnx"]
        for name in names:
            path = os.path.join(model_dir, "onnx", name)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"no aligner graph under {os.path.join(model_dir, 'onnx')}; "
            "run scripts/fetch_qwen_aligner.py")

    def _token_id(self, token: str) -> int:
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"tokenizer has no {token!r}")
        return int(value)

    @staticmethod
    def _numpy_dtype(onnx_type: str):
        """`tensor(int32)` -> ``np.int32``, defaulting to float32."""
        name = (onnx_type or "").strip()
        if name.startswith("tensor(") and name.endswith(")"):
            name = name[len("tensor("):-1]
        return {"int32": np.int32, "int64": np.int64,
                "float": np.float32, "float32": np.float32,
                "float16": np.float16}.get(name, np.float32)

    def _class_count(self) -> int:
        """Frames the head can address -- the ceiling on one call's audio."""
        shape = self.session.get_outputs()[0].shape
        last = shape[-1] if shape else None
        return int(last) if isinstance(last, int) else 5000

    @property
    def max_seconds(self) -> float:
        return self.max_frames * self.frame_ms / 1000.0

    def _mel(self, audio: np.ndarray) -> np.ndarray:
        from whisper.audio import log_mel_spectrogram

        mel = log_mel_spectrogram(audio.astype(np.float32), n_mels=128)
        return np.ascontiguousarray(mel.numpy(), dtype=np.float32)

    def _build_ids(self, words: Sequence[str], n_audio: int) -> np.ndarray:
        ids: List[int] = [self._special[AUDIO_START]]
        ids.extend([self._special[AUDIO_PAD]] * n_audio)
        ids.append(self._special[AUDIO_END])
        stamp = self._special[TIMESTAMP]
        for word in words:
            # Each word is encoded between two special tokens, so it tokenizes
            # with no leading space -- the same shape the reference produces by
            # concatenating the string and letting the tokenizer split it.
            ids.extend(self.tokenizer.encode(word, add_special_tokens=False).ids)
            ids.extend([stamp, stamp])
        return np.asarray([ids], dtype=np.int64)

    def align(self, audio: np.ndarray, text: str) -> List[Dict]:
        """Return ``[{word, start, end}]`` for ``text`` spoken in ``audio``.

        ``audio`` is mono float32 at 16 kHz in [-1, 1]; times are seconds from
        the start of that array, so a caller working on a chunk adds the chunk
        offset itself.
        """
        words = split_words(text)
        if not words or audio.size == 0:
            return []
        duration = audio.size / 16000.0
        if duration > self.max_seconds:
            raise ValueError(
                f"{duration:.0f}s exceeds the aligner's {self.max_seconds:.0f}s "
                "ceiling; chunk the audio first")

        mel = self._mel(audio)
        frames = mel.shape[-1]
        ids = self._build_ids(words, audio_token_count(frames))
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "input_features": mel[None, :, :],
            "feature_attention_mask": np.ones((1, frames)),
        }
        outputs = self.session.run(["logits"], {
            name: np.ascontiguousarray(value, dtype=self._dtypes.get(
                name, value.dtype))
            for name, value in feed.items()
        })
        predicted = outputs[0][0].argmax(axis=-1)
        stamps = predicted[ids[0] == self._special[TIMESTAMP]]
        if stamps.size < 2 * len(words):
            raise RuntimeError(
                f"aligner returned {stamps.size} slots for {len(words)} words")

        fixed = repair_monotonic(stamps[:2 * len(words)])
        if self.spread:
            fixed = spread_ties(fixed)
        scale = self.frame_ms / 1000.0
        # The head can name any of its 5000 frames, including ones past the end
        # of THIS audio. Unclamped, a word then lands beyond the chunk it came
        # from -- MEASURED 2026-08-19 on a 4.96h VOD: 2 of 22752 words jumped
        # ~46s forward, past the following chunk's start, so the transcript ran
        # backwards at two seams. Clamping is the honest fix: a word cannot be
        # spoken after the audio it was transcribed from ends.
        out: List[Dict] = []
        for index, word in enumerate(words):
            start = min(fixed[index * 2] * scale, duration)
            end = min(fixed[index * 2 + 1] * scale, duration)
            out.append({"word": word,
                        "start": round(start, 3),
                        "end": round(max(end, start), 3)})
        return out

    def close(self) -> None:
        self.session = None


def read_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Mono float32 in [-1, 1] from 16-bit PCM WAV bytes.

    `soundfile` is not a dependency and `ffmpeg` already hands this repo
    16 kHz mono PCM, so the stdlib reader is enough.
    """
    import io
    import wave

    with wave.open(io.BytesIO(data), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate
