# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Resolve on-disk resource paths in both dev and PyInstaller-bundled modes.

In dev, model weights and other resource files live in the obvious project-
relative spot. Once PyInstaller freezes the backend, `__file__`-relative path
math breaks: bundled modules don't keep the original engines/<x>/file.py
layout, so anything computed from `__file__` resolves to nonsense. PyInstaller
sets `sys.frozen = True` and exposes the bundle's extraction dir as
`sys._MEIPASS` — use that instead whenever it's present.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _configured_root(name: str) -> str | None:
    configured = os.environ.get(name, "").strip()
    if not configured:
        return None
    if not os.path.isabs(configured) or (os.name == "nt" and not os.path.splitdrive(configured)[0]):
        raise ValueError(f"{name} must be an absolute path")
    return os.path.normpath(configured)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_resource_dir() -> str:
    """Root directory to resolve bundled resources (models/, etc.) from."""
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return PROJECT_ROOT


def get_models_dir() -> str:
    """Opt-in external model root, otherwise the existing dev/bundle location.

    Setup and inference must receive the same absolute override. This selects
    one complete model root; it does not merge partial downloads with bundled
    weights or create directories during lookup.
    """
    configured = _configured_root("RECALL_MODELS_DIR")
    if configured:
        return configured
    return os.path.join(get_resource_dir(), "models")


def is_installed_build() -> bool:
    """True for the setup-installed app, whose installer leaves a marker file.

    The engine runs from ``resources/recall-engine/`` and the marker sits in
    ``resources/``. Portable and dev builds have no marker.
    """
    if not is_frozen():
        return False
    resources = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.isfile(os.path.join(resources, "recall-installation.json"))


def get_downloaded_models_dir() -> str:
    """Root for the large models setup downloads (ASR, aligner, text and visual judge).

    An installed build keeps them per user in ``%LOCALAPPDATA%/Recall/models``, outside
    the install directory, so an upgrade (which replaces that directory) keeps
    them. Small models stay bundled under ``get_models_dir()``. Portable, dev and
    explicit ``RECALL_MODELS_DIR`` roots hold everything in one place.
    """
    configured = _configured_root("RECALL_MODELS_DIR")
    if configured:
        return configured
    local = os.environ.get("LOCALAPPDATA", "").strip()
    if local and os.path.isabs(local) and is_installed_build():
        return os.path.join(os.path.normpath(local), "Recall", "models")
    return get_models_dir()


def get_engines_root() -> str:
    """Root of the native engines package tree."""
    return os.path.join(get_resource_dir(), "engines")


def get_engine_dir(name: str) -> str:
    return os.path.join(get_engines_root(), name)


def get_qwen_asr_dir() -> str:
    """Directory holding the Qwen3-ASR GGUF pair (backbone + mmproj).

    ``models/asr/qwen3-asr-1.7b`` is the shipping location -- nested under
    `asr/` beside the parakeet checkpoint an earlier experiment left there,
    rather than beside it, so the two never get confused for each other. The
    experimental directory is checked second so a dev box that pulled the
    weights during the migration keeps working without moving 2.4 GB around.
    """
    snapshot = os.environ.get("RECALL_SPEECH_SNAPSHOT", "")
    if snapshot and not os.environ.get("RECALL_MODELS_DIR", "").strip():
        return os.path.join(snapshot, "asr", "qwen3-asr-1.7b")
    return _first_holding(
        "Qwen3-ASR-1.7B-Q8_0.gguf",
        os.path.join("asr", "qwen3-asr-1.7b"),
        os.path.join("experimental", "qwen3-asr-1.7b"),
    )


def get_qwen_aligner_dir() -> str:
    """Directory holding the Qwen3-ForcedAligner ONNX export.

    Same two-location rule as ``get_qwen_asr_dir``. The aligner is what supplies
    word timings now that whisper no longer runs, so its absence is a hard
    failure rather than a degraded transcript.
    """
    snapshot = os.environ.get("RECALL_SPEECH_SNAPSHOT", "")
    if snapshot and not os.environ.get("RECALL_MODELS_DIR", "").strip():
        return os.path.join(snapshot, "asr", "qwen3-aligner-0.6b")
    return _first_holding(
        os.path.join("onnx", "model.onnx"),
        os.path.join("asr", "qwen3-aligner-0.6b"),
        os.path.join("experimental", "qwen3-aligner-0.6b"),
    )


def get_silero_vad_dir() -> str | None:
    """Local Silero VAD directory, or ``None`` in dev when not provisioned.

    ``onnx-asr`` distributes the loader but resolves ``silero_vad.onnx`` from
    Hugging Face when no path is supplied. A portable build must stay offline,
    so frozen runs always point at the build-required local asset. Returning
    the path even after a damaged install is intentional: a missing file then
    fails into Recall's Whisper fallback instead of attempting a download.
    """
    directory = os.path.join(get_speech_models_dir(), "asr", "silero-vad")
    managed = bool(os.environ.get("RECALL_SPEECH_SNAPSHOT", "")) and not os.environ.get("RECALL_MODELS_DIR", "").strip()
    if not is_frozen() and not managed and not os.path.isfile(os.path.join(directory, "silero_vad.onnx")):
        return None
    return directory


def get_speech_models_dir() -> str:
    """Root for the selected speech group, including VAD and speaker assets."""
    snapshot = os.environ.get("RECALL_SPEECH_SNAPSHOT", "")
    if snapshot and not os.environ.get("RECALL_MODELS_DIR", "").strip():
        return snapshot
    return get_models_dir()


def _first_holding(marker: str, *candidates: str) -> str:
    """First candidate under the downloaded-models root that contains ``marker``.

    Existence of the DIRECTORY is not enough: ``models/asr`` already exists for
    an unrelated checkpoint, so a bare isdir() check silently resolves to an
    empty location and the failure surfaces much later as a missing-weights
    error pointing at the wrong path. Falling back to the first candidate keeps
    that error naming where the file is supposed to be.
    """
    models = get_downloaded_models_dir()
    for name in candidates:
        if os.path.exists(os.path.join(models, name, marker)):
            return os.path.join(models, name)
    return os.path.join(models, candidates[0])


def get_data_dir() -> str:
    """Writable runtime data root (db, exports, cache, per-install ranker model).

    Explicit RECALL_STORAGE_ROOT: <configured root>/data.
    Dev without an override: <project_root>/data, matching existing behavior.
    Frozen: a `data/` folder next to the packaged exe — this app ships as a
    portable build (not an installer into Program Files), so keeping
    everything self-contained next to the executable is the right convention
    rather than scattering into %APPDATA%.
    """
    configured = _configured_root("RECALL_STORAGE_ROOT")
    if configured:
        return os.path.join(configured, "data")
    if is_frozen():
        base = os.path.dirname(sys.executable)
    else:
        base = PROJECT_ROOT
    return os.path.join(base, "data")


def get_model_setup_dir() -> str:
    """Managed speech versions live separately from disposable runtime data."""
    configured = _configured_root("RECALL_STORAGE_ROOT")
    if configured:
        return os.path.join(configured, "model-setup")
    return os.path.join(os.path.dirname(get_data_dir()), "model-setup")
