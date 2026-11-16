# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Local voice embeddings: mono 16 kHz audio in, a speaker vector out.

WeSpeaker ResNet34-LM takes 80-bin Kaldi filterbanks (engines.caption.kaldi_fbank)
and returns a 256-dimensional embedding. A missing model or an unusable window
returns None, never an error, so any caller can treat it as optional.

This module used to also filter caption words down to the streamer's voice. That
filter is gone. It built a fixed profile from six 3-minute regions per VOD and
scored each word on a 1.5 s grid, and on 2026-10-06 it cut 40-86% of the words in
most clips of two Fortnite duo VODs, and the words it cut were the streamer's own
speech: streamer windows scored 0.05-0.3 against its 0.35 bar. It had been off
since then. The embedding itself was not the problem: a 2026-11-05 prototype that
centres on the VOD's dominant voice across all its speech windows separated the
streamer from squad teammates at a window AUC of 0.97 with this same model.
embed() stays for that approach.
"""

import os
from typing import Optional

import numpy as np

from core.bundle_paths import get_speech_models_dir, is_frozen

SAMPLE_RATE = 16000
EMBED_MODEL_FILENAME = "voxceleb_resnet34_LM.onnx"
EMBED_MODEL_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"

# Shorter segments contain too little speech for a reliable embedding.
MIN_SEGMENT_SEC = 1.0
# One intra-op thread, like the VAD (qwen_asr.VAD_THREADS), so a whole-VOD
# embedding pass can run beside Perception. MEASURED 2026-11-12 on 600 windows:
# 1 thread 53.6 ms wall / 53.2 ms CPU per window, 2 threads 29.4 / 58.4, 4
# threads 19.8 / 78.4 -- extra threads buy wall time with CPU the scan needs.
EMBED_THREADS = 1

_SESSION = None
_SESSION_TRIED = False


def _model_path() -> Optional[str]:
    """Bundled model first; fall back to the HF cache in dev.

    A packaged build must ship the weights -- see the bundle_paths note on
    faster-whisper. Downloading 26MB mid-scan on a user's machine (or failing to,
    offline) is not an acceptable runtime behaviour.
    """
    bundled = os.path.join(get_speech_models_dir(), "speaker", EMBED_MODEL_FILENAME)
    if os.path.isfile(bundled):
        return bundled
    managed = bool(os.environ.get("RECALL_SPEECH_SNAPSHOT", "")) and not os.environ.get("RECALL_MODELS_DIR", "").strip()
    if is_frozen() or managed:
        return None
    try:
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from core.model_catalog import SPEAKER_REVISION, SPEAKER_FILES
        from core.model_downloads import matches
        cached = hf_hub_download(EMBED_MODEL_REPO, EMBED_MODEL_FILENAME,
                                 revision=SPEAKER_REVISION, token=False)
        return cached if matches(Path(cached), SPEAKER_FILES[EMBED_MODEL_FILENAME]) else None
    except Exception:  # noqa: BLE001 - embeddings are optional, never fatal
        return None


def _session():
    global _SESSION, _SESSION_TRIED
    if _SESSION is None and not _SESSION_TRIED:
        _SESSION_TRIED = True
        try:
            import onnxruntime as ort
            path = _model_path()
            if path:
                options = ort.SessionOptions()
                options.intra_op_num_threads = EMBED_THREADS
                options.inter_op_num_threads = 1
                _SESSION = ort.InferenceSession(
                    path, sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            print(f"Speaker embeddings unavailable: {exc}")
    return _SESSION


def embed(samples: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalised 256-d speaker embedding for one mono 16k window.

    ``samples`` are int16-range floats, as wav_reader returns them, not [-1, 1]:
    Kaldi filterbanks are computed on that scale.
    """
    session = _session()
    if session is None or samples.size < int(SAMPLE_RATE * MIN_SEGMENT_SEC):
        return None
    try:
        # 80 bins, 25/10 ms frames, Hamming window, no dither (see kaldi_fbank).
        from engines.caption.kaldi_fbank import fbank
        feats = fbank(samples)
        feats = feats - feats.mean(axis=0, keepdims=True)  # CMN, as WeSpeaker does
        out = session.run(None, {"feats": feats[None]})[0][0]
    except Exception:  # noqa: BLE001 - one bad window must not stop the caller
        return None
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else None


def wav_reader(path: str):
    """read_window over one 16k mono wav, indexed by time WITHIN that file.

    Returns samples on the int16 scale embed() expects, so a caller embedding
    windows of the scan's analysis WAV does not have to know that convention.
    """
    import wave
    try:
        with wave.open(path, "rb") as handle:
            if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
                return lambda _s, _e: None
            pcm = np.frombuffer(
                handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
    except Exception:  # noqa: BLE001
        return lambda _s, _e: None

    def read(start, end):
        window = pcm[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        return window if window.size else None
    return read
