# PyInstaller spec for the Recall backend, bundled so the packaged
# Electron app doesn't require a system Python install.
#
# Build with:
#   cd <recall-source>
#   python -m PyInstaller apps/api/recall-engine.spec --noconfirm
#
# Output lands in apps/api/dist/recall-engine/ (onedir build — chosen over
# onefile because onefile re-extracts a multi-GB payload to a temp dir on
# every launch, which would make startup painfully slow for a multi-GB
# torch+ASR+VLM bundle).

import os
import json
import hashlib
import shutil
import sys
import tempfile
from PyInstaller.utils.hooks import collect_all, collect_submodules

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(SPEC)), "..", ".."))
# collect_submodules("core") below imports the package for real (using this
# spec script's own live interpreter), which needs the repo root on sys.path.
sys.path.insert(0, PROJECT_ROOT)

# apps/api/stubs comes FIRST so the aistudio_sdk stub wins over the real
# package in site-packages. Upstream declares no license -- no LICENSE file,
# UNKNOWN for License and Home-page -- so it cannot lawfully be redistributed,
# but paddlex imports two of its symbols on the PaddleOCR construction path
# (not merely the download path), so excluding it outright breaks OCR. The stub
# satisfies the import and raises if the Baidu download is ever reached.
# Verified: PaddleOCR constructs against the stub and loads the bundled
# PP-OCRv6 ONNX models. See apps/api/stubs/aistudio_sdk/.
pathex = [os.path.join(PROJECT_ROOT, "apps", "api", "stubs"), PROJECT_ROOT]

from core.model_vault import seal_file  # noqa: E402 - needs PROJECT_ROOT on sys.path

from publish.model_payload import ModelPayloadManifest, validate_model_destinations
from publish.verify_model_bundle import expected_model_payload, verify_model_bundle
from publish.verify_release_privacy import verify_release_privacy
from publish.verify_native_inputs import (
    configure_native_search, pin_microsoft_runtime_inputs, verify_native_inputs,
)
from publish.dependency_notices import notice_payload, verify_notice_payload
from publish.export_public_models import validate_public_model

configure_native_search()
_microsoft_runtime_root = os.path.join(
    PROJECT_ROOT, "vendor", "microsoft-runtime", "14.51.36247.0", "x64",
)
pin_microsoft_runtime_inputs([], _microsoft_runtime_root)

_model_manifest_path = os.environ.get("RECALL_MODEL_MANIFEST", "").strip()
if not _model_manifest_path:
    raise RuntimeError("Set RECALL_MODEL_MANIFEST to a reviewed model file-and-hash manifest before packaging.")
_model_payload = ModelPayloadManifest(os.path.join(PROJECT_ROOT, "models"), _model_manifest_path)


# Recall's own trained artifacts are sealed on the way into the bundle; the
# public third-party weights (Qwen, YOLOX, PP-OCR, YAMNet,
# MediaPipe) are not, because they are freely downloadable and sealing 7 GB of
# them would cost build time for nothing. See core/model_vault.py for the
# honest scope of what sealing does and does not protect.
#
# Sealed copies go to a temp dir rather than next to the source: writing
# `models/*.sealed` into the repo would leave build droppings in a tree that
# training scripts also write to, and the plain .json must stay the source of
# truth for dev and eval.
_SEAL_DIR = tempfile.mkdtemp(prefix="recall-sealed-")
_sealed_sources = {}


def _seal(source_path):
    """Seal `source_path` into the build temp dir and return the sealed path."""
    _model_payload.verify_source(source_path, asset=True)
    validate_public_model(source_path)
    target = os.path.join(_SEAL_DIR, os.path.basename(source_path) + ".sealed")
    seal_file(source_path, target)
    _sealed_sources[os.path.abspath(target)] = os.path.abspath(source_path)
    return target

# Heavy/ML packages whose native binaries, data files (model configs, tokenizer
# assets, bundled ONNX weights), and dynamic import graphs PyInstaller's static
# analysis won't fully catch on its own.
COLLECT_PACKAGES = [
    "torch",
    "torchvision",
    # Not the ASR model -- that left with the fallback backend. qwen_aligner
    # imports whisper.audio.log_mel_spectrogram, which loads the mel filter
    # bank from whisper/assets/, and PyInstaller will not find that data file
    # from a function-level import.
    "whisper",
    "onnxruntime",
    "paddleocr",
    "paddlex",
    "cv2",
    "scipy",
    "uvicorn",
    "fastapi",
    "llama_cpp",
    # Qwen3-ASR uses onnx-asr's bundled Silero graph for VAD and the Rust
    # tokenizers extension for the forced aligner. Both are imported lazily,
    # so PyInstaller cannot discover their native/data payloads reliably from
    # the entry point alone.
    "onnx_asr",
    "tokenizers",
]

datas = []
_dependency_notice_manifest = os.environ.get("RECALL_DEPENDENCY_NOTICE_MANIFEST", "").strip()
if not _dependency_notice_manifest:
    raise RuntimeError("Set RECALL_DEPENDENCY_NOTICE_MANIFEST to the reviewed dependency notice manifest.")
_dependency_notice_datas, _expected_dependency_notices = notice_payload(_dependency_notice_manifest)
datas.extend(_dependency_notice_datas)
datas.append((os.path.join(PROJECT_ROOT, "engines", "caption", "LICENSE.torchaudio"), "notices"))
binaries = []
hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
]

# core/ is imported normally in most places, but a handful of engine modules
# (reaction-engine, story-engine's scoring.py) are loaded via
# spec_from_file_location from raw bundled data (see the engines/ Tree below)
# rather than a traceable `import` statement — PyInstaller's static analyzer
# never parses those files at all, so anything they import under core.* needs
# to be listed explicitly here instead of being auto-discovered.
#
# core/ and core/models/ are both implicit namespace packages (no
# __init__.py), and collect_submodules doesn't recurse into the nested
# models/ namespace package reliably — so core.models.* is listed by hand
# rather than trusted to auto-discovery.
hiddenimports += collect_submodules("core", filter=lambda name: name != "core.fixture_autoexport")
hiddenimports += collect_submodules("engines")
hiddenimports += [
    "core.models.caption",
    "core.models.adapters",
    "core.models.canonical",
    "core.models.clip",
    "core.models.event",
    "core.models.reaction",
    "core.models.signal",
    "core.models.story",
    "core.artifacts",
    "core.migrations",
    "core.migrations.runner",
]

def _keep_runtime_submodule(name):
    parts = set(name.split("."))
    blocked = {
        "benchmarks",
        "conftest",
        "docs",
        "examples",
        "test",
        "testing",
        "tests",
        "tutorial",
        "tutorials",
    }
    return not (parts & blocked)


DATA_EXCLUDES = [
    "**/benchmarks/**",
    "**/docs/**",
    "**/examples/**",
    "**/test/**",
    "**/testing/**",
    "**/tests/**",
    "**/tutorial/**",
    "**/tutorials/**",
    "**/*.py",
    "**/*.pyc",
    "**/*.pyo",
]


for pkg in COLLECT_PACKAGES:
    d, b, h = collect_all(
        pkg,
        include_py_files=False,
        filter_submodules=_keep_runtime_submodule,
        exclude_datas=DATA_EXCLUDES,
    )
    datas += d
    binaries += b
    hiddenimports += h

# --- Bundled resources resolved at runtime via core/bundle_paths.py ---

# Person-detection weights (YOLOX, Apache-2.0). Hard-fail rather than ship an
# engine whose facecam detection cannot run at all.
#
# There is deliberately NO fallback to the old .pt weights: those are
# Ultralytics YOLO under AGPL-3.0 and must never reach a distributed build.
# Discovered, not hardcoded: engines/vision/person_onnx.py resolves the model
# by name (RECALL_PERSON_MODEL, default yolox_s.onnx) and reads its input size
# from the graph, so packaging the wrong file would fail only at runtime.
from engines.vision.person_onnx import MODEL_NAME as _PERSON_MODEL_NAME

_person_model = os.path.join(PROJECT_ROOT, "models", "vision", _PERSON_MODEL_NAME)
if not os.path.exists(_person_model):
    raise FileNotFoundError(
        f"Expected person-detection weights at {_person_model}. Fetch them with "
        "scripts/fetch_person_model.py before packaging."
    )
datas.append((str(_model_payload.verify_source(_person_model, asset=True)), os.path.join("models", "vision")))
# A base ranker is optional: Recall intentionally stays on its deterministic
# heuristic until a real-data v5 base has earned shipping. Never make a package
# build depend on the removed synthetic/stale model.
_ranker_base = os.path.join(PROJECT_ROOT, "models", "public-release", "ranker_base.json")
if os.path.exists(_ranker_base) or os.path.exists(os.path.join(PROJECT_ROOT, "models", "ranker_base.json")):
    datas.append((_seal(_ranker_base), "models"))

_second_look_rejector = os.path.join(
    PROJECT_ROOT, "models", "public-release", "second_look_rejector.json"
)
if os.path.exists(_second_look_rejector) or os.path.exists(os.path.join(PROJECT_ROOT, "models", "second_look_rejector.json")):
    datas.append((_seal(_second_look_rejector), "models"))
datas.append((os.path.join(PROJECT_ROOT, "configs", "hype_lexicon.json"), "configs"))

# Face-emotion weights: MediaPipe Face Landmarker (Apache-2.0), which replaced
# HSEmotion's enet_b0_8_va_mtl. HSEmotion's package is Apache-2.0 but its
# weights derive from AffectNet, whose terms are research/non-commercial, so
# the licence chain never cleanly reached a paid product. The landmarker emits
# face geometry (52 blendshapes) rather than emotion labels, and Google ships
# the weights permissively with no dataset asterisk.
#
# The hard-fail is inherited deliberately, only repointed: a build whose face
# channel silently disables itself is the failure this exists to prevent --
# `face` carries the highest fusion weight and its absence is invisible in the
# UI. See engines/emotion/blendshape_scorer.py.
_face_landmarker = os.path.join(
    PROJECT_ROOT, "models", "face", "face_landmarker.task"
)
if not os.path.exists(_face_landmarker):
    raise FileNotFoundError(
        f"Expected the MediaPipe Face Landmarker at {_face_landmarker}. Fetch "
        "float16/1/face_landmarker.task from "
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/ "
        "before packaging."
    )
datas.append((str(_model_payload.verify_source(_face_landmarker, asset=True)), os.path.join("models", "face")))

# Stream Memory sentence embeddings. This is a creator-facing search feature,
# not part of scan inference, but omitting it silently drops Meaning + words
# back to the compact LSI fallback. Package the checksum-pinned Apache-2.0
# model, tokenizer, and license together so installed builds behave like dev.
_memory_model_dir = os.path.join(
    PROJECT_ROOT, "models", "memory", "all-minilm-l6-v2"
)
for _memory_model_name in (
    "model_quint8_avx2.onnx",
    "tokenizer.json",
    "LICENSE",
):
    _memory_model_path = os.path.join(_memory_model_dir, _memory_model_name)
    if not os.path.isfile(_memory_model_path):
        raise FileNotFoundError(
            f"Expected Stream Memory model asset at {_memory_model_path}. "
            "Run scripts/fetch_memory_embedding_model.py before packaging."
        )
    datas.append((
        str(_model_payload.verify_source(
            _memory_model_path,
            asset=_memory_model_name != "LICENSE",
            notice=_memory_model_name == "LICENSE",
        )),
        os.path.join("models", "memory", "all-minilm-l6-v2"),
    ))

# OCR weights are stored locally so the packaged backend does not download
# PaddleX official models on first launch.
_ocr_models = os.path.join(PROJECT_ROOT, "models", "ocr")
_required_ocr_dirs = (
    os.path.join(_ocr_models, "PP-OCRv6_small_det_onnx"),
    os.path.join(_ocr_models, "PP-OCRv6_small_rec_onnx"),
)
_missing_ocr_dirs = [path for path in _required_ocr_dirs if not os.path.exists(path)]
if _missing_ocr_dirs:
    raise FileNotFoundError(
        f"Expected PP-OCRv6 ONNX models under {_ocr_models}; missing {_missing_ocr_dirs}. "
        "Populate models/ocr/PP-OCRv6_small_det_onnx and "
        "models/ocr/PP-OCRv6_small_rec_onnx before packaging."
    )
for _ocr_directory in ("PP-OCRv6_small_det_onnx", "PP-OCRv6_small_rec_onnx"):
    datas.extend(_model_payload.directory(
        f"ocr/{_ocr_directory}", ["inference.onnx", "inference.json", "inference.yml"],
    ))

# Semantic judge model. It is enabled by default in the engine, so a packaged
# build must contain both llama-cpp-python (collected above) and the GGUF rather
# than silently degrading a first-party feature on every user machine.
_llm_models = os.path.join(PROJECT_ROOT, "models", "llm")
_gguf_models = []
if os.path.isdir(_llm_models):
    _gguf_models = [name for name in os.listdir(_llm_models) if name.lower().endswith(".gguf")]
if not _gguf_models:
    raise FileNotFoundError("Expected at least one semantic judge GGUF under models/llm before packaging.")
from engines.semantic.backend import family_rank

_selected_llm = min(_gguf_models, key=lambda name: (family_rank(name), name))
datas.extend(_model_payload.directory("llm", [_selected_llm]))

# Visual judge model. Default ON whenever a complete VLM pair exists under
# models/vlm (see run_pipeline: measured recall +0.05 / precision +0.03 / p@5
# +0.02 against arousal-only), and it degrades *silently* when the pair is
# absent — so omitting it here ships an engine measurably worse than the one
# every quality number in this repo was produced on. Only ONE lane is packaged
# — the one resolve_models() would actually choose — because any other complete
# pair under models/vlm is ~2.8 GB of payload that can never be selected.
#
# The lane is DISCOVERED, not hardcoded: with a directory name baked in here, a
# model upgrade keeps packaging the superseded lane while dev runs the new one,
# and the mismatch is invisible until someone diffs a packaged scan. Ranking
# reuses the engine's own family_rank so the two cannot drift.

# --- Commercial-licensing guard -------------------------------------------
#
# Not every model on this dev box may be redistributed in a paid product, and
# the ones that may not are indistinguishable from the ones that may at the
# filesystem level. Packaging is the last point where the distinction is still
# enforceable, so it is enforced here rather than trusted to a tidy models/ dir.
#
#   qwen2.5-vl-3b   Qwen RESEARCH license (non-commercial). Note this is a
#                   size-specific trap: Qwen2.5-VL 7B/72B ARE Apache-2.0, only
#                   the 3B is research-only, so "it's a Qwen model" is not a
#                   safe inference.
#   minicpm         MiniCPM Model License; commercial use requires registration
#                   with the publisher. The 4.5 evaluation is closed and its
#                   weights stay on disk only as an experiment record.
#
# Matching is on the path, lowercased. Keep markers narrow enough that they
# cannot swallow a permissively licensed sibling.
_NONCOMMERCIAL_MARKERS = (
    "qwen2.5-vl-3b",
    "qwen25-vl-3b",
    "minicpm",
    # Ultralytics YOLO: AGPL-3.0, replaced by the YOLOX/onnxruntime backend in
    # engines/vision/person_onnx.py. Listed so a reinstalled dependency or a
    # leftover weights file cannot quietly return -- AGPL would oblige us to
    # publish Recall's own source. See docs/DETECTOR_RELICENSE_PLAN.md.
    "ultralytics",
    "yolo11n.pt",
    "yolo26n.pt",
    # HSEmotion: the package is Apache-2.0 but its weights derive from
    # AffectNet (research/non-commercial terms). Replaced by the MediaPipe
    # Face Landmarker blendshape scorer. The scorer is still selectable via
    # RECALL_FACE_SCORER=hsemotion for A/B measurement on a dev box, so these
    # markers are what stop a developer's local install from being packaged.
    "hsemotion",
    "enet_b0_8_va_mtl",
)


def _noncommercial_marker(path):
    """The marker `path` trips, or None if it is clear to redistribute."""
    text = str(path).replace("\\", "/").lower()
    for marker in _NONCOMMERCIAL_MARKERS:
        if marker in text:
            return marker
    return None


_vlm_root = os.path.join(PROJECT_ROOT, "models", "vlm")
_vlm_lanes = []
for _lane_dir, _subdirs, _names in os.walk(_vlm_root):
    _lane_ggufs = sorted(
        os.path.join(_lane_dir, name)
        for name in _names if name.lower().endswith(".gguf")
    )
    _lane_proj = [
        path for path in _lane_ggufs
        if os.path.basename(path).lower().startswith("mmproj-")
    ]
    _lane_weights = [path for path in _lane_ggufs if path not in _lane_proj]
    if _lane_proj and _lane_weights:
        # Filter here rather than after ranking: a non-commercial lane that is
        # merely outranked today would silently become the packaged lane the
        # moment a newer lane is removed or renamed.
        _lane_marker = _noncommercial_marker(f"{_lane_weights[0]} {_lane_proj[0]}")
        if _lane_marker:
            print(
                f"recall-engine.spec: skipping VLM lane {_lane_dir} "
                f"(non-commercial: {_lane_marker})"
            )
            continue
        _vlm_lanes.append((_lane_dir, _lane_weights[0], _lane_proj[0]))
if not _vlm_lanes:
    raise FileNotFoundError(
        f"Expected a complete VLM pair (model + mmproj-* GGUF) under {_vlm_root}. "
        "The visual judge degrades silently when it is missing, so a package "
        "build must not proceed without it."
    )
# Same key as visual_judge.resolve_models(): rank on the pair's file labels.
_vlm_choice = min(
    _vlm_lanes,
    key=lambda lane: (
        family_rank(f"{lane[1]} {lane[2]}".lower()),
        f"{lane[1]} {lane[2]}".lower(),
    ),
)
_vlm_lane = _vlm_choice[0]
_vlm_rel = os.path.relpath(_vlm_lane, _vlm_root)
_vlm_manifest_dir = "vlm" if _vlm_rel == os.curdir else "vlm/" + _vlm_rel.replace("\\", "/")
datas.extend(_model_payload.directory(
    _vlm_manifest_dir, [os.path.basename(_vlm_choice[1]), os.path.basename(_vlm_choice[2])],
))

# YAMNet audio-event tagger. engines/audio/audio_events.py resolves both files
# through get_models_dir(); audio_events_available() just returns False when
# they are absent, so a missing bundle disables audio events with no error.
for _yamnet_name in ("yamnet.onnx", "yamnet_class_map.csv"):
    _yamnet_path = os.path.join(PROJECT_ROOT, "models", _yamnet_name)
    if not os.path.exists(_yamnet_path):
        raise FileNotFoundError(
            f"Expected {_yamnet_path} for bundled audio-event tagging."
        )
    datas.append((str(_model_payload.verify_source(_yamnet_path, asset=True)), "models"))

# ASR weights, pre-seeded so there is no first-run download.
#
# Qwen3-ASR + Qwen3-ForcedAligner are the shipping default. The fetch scripts
# originally staged them under models/experimental; models/asr is the canonical
# shipping location. Accept either source, but always install into the canonical
# bundled paths consumed by core.bundle_paths.
def _first_model_source(marker, *relative_dirs):
    for relative_dir in relative_dirs:
        candidate = os.path.join(PROJECT_ROOT, "models", relative_dir)
        if os.path.exists(os.path.join(candidate, marker)):
            return candidate
    expected = os.path.join(PROJECT_ROOT, "models", relative_dirs[0], marker)
    raise FileNotFoundError(f"Expected packaged model asset at {expected}")


_qwen_asr_dir = _first_model_source(
    "Qwen3-ASR-1.7B-Q8_0.gguf",
    os.path.join("asr", "qwen3-asr-1.7b"),
    os.path.join("experimental", "qwen3-asr-1.7b"),
)
for _qwen_asr_name in (
    "Qwen3-ASR-1.7B-Q8_0.gguf",
    "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
):
    _qwen_asr_path = os.path.join(_qwen_asr_dir, _qwen_asr_name)
    if not os.path.isfile(_qwen_asr_path):
        raise FileNotFoundError(f"Expected Qwen3-ASR asset at {_qwen_asr_path}")
    datas.append((
        str(_model_payload.verify_source(_qwen_asr_path, asset=True)),
        os.path.join("models", "asr", "qwen3-asr-1.7b"),
    ))

_qwen_aligner_dir = _first_model_source(
    os.path.join("onnx", "model.onnx"),
    os.path.join("asr", "qwen3-aligner-0.6b"),
    os.path.join("experimental", "qwen3-aligner-0.6b"),
)
for _qwen_aligner_name in (
    "config.json",
    "tokenizer.json",
    os.path.join("onnx", "model.onnx"),
    os.path.join("onnx", "model.onnx_data"),
):
    _qwen_aligner_path = os.path.join(_qwen_aligner_dir, _qwen_aligner_name)
    if not os.path.isfile(_qwen_aligner_path):
        raise FileNotFoundError(
            f"Expected Qwen3 forced-aligner asset at {_qwen_aligner_path}"
        )
    datas.append((
        str(_model_payload.verify_source(_qwen_aligner_path, asset=True)),
        os.path.join(
            "models", "asr", "qwen3-aligner-0.6b",
            os.path.dirname(_qwen_aligner_name),
        ),
    ))

# Provision these canonical files with scripts/fetch_audio_support_models.py.
# The release cannot depend on whichever HF snapshot sorts first on this PC.
from core.model_catalog import VAD_FILES, SPEAKER_FILES

datas.extend(_model_payload.directory(
    "asr/silero-vad", list(VAD_FILES), pinned=VAD_FILES,
))
datas.extend(_model_payload.directory(
    "speaker", list(SPEAKER_FILES), pinned=SPEAKER_FILES,
))

# Direct-file and sealed model selections also bring their explicitly reviewed
# notices. Deduplicate entries already returned by directory-based selections.
_existing_model_datas = set(datas)
datas.extend(entry for entry in _model_payload.notice_datas() if entry not in _existing_model_datas)
validate_model_destinations(datas)

# Installed setup fetches the pinned large models from Hugging Face. Keep the
# fully reviewed source selection above as the build input contract, then omit
# only catalogued destinations. Small models and all attribution files remain
# bundled; portable builds retain the complete offline payload.
if os.environ.get("RECALL_INSTALLER_DOWNLOAD_MODELS", "").strip() == "1":
    from core.installer_models import installed_download_paths, download_groups

    _download_paths = set(installed_download_paths())
    _download_specs = {
        relative: (size, digest)
        for _, _, files in download_groups()
        for relative, size, digest in files.values()
    }
    _download_matches = set()
    _installed_datas = []
    for _source, _destination in datas:
        _target = (
            str(_destination).replace("\\", "/").rstrip("/")
            + "/" + os.path.basename(_source)
        )
        _model_target = _target[len("models/"):] if _target.startswith("models/") else None
        if _model_target in _download_paths:
            _source_relative = os.path.relpath(_source, _model_payload.root).replace("\\", "/")
            _size, _digest = _download_specs[_model_target]
            if (_model_payload.files.get(_source_relative) != _digest
                    or os.path.getsize(_source) != _size):
                raise RuntimeError("Installer download differs from reviewed model input: " + _model_target)
            _download_matches.add(_model_target)
        else:
            _installed_datas.append((_source, _destination))
    if not _download_paths or _download_matches != _download_paths:
        raise RuntimeError(
            "Installed model download catalog does not match the selected model payload: "
            + ", ".join(sorted(_download_paths - _download_matches))
        )
    datas = _installed_datas
    print(f"recall-engine.spec: setup will download {len(_download_matches)} pinned model files")

_expected_model_payload = expected_model_payload(datas, _model_payload, _sealed_sources)


# Never bundle the runtime database from data/. It contains creator source paths,
# clips, labels, and settings from the build machine. A fresh install creates an
# empty database from the versioned migrations on first launch.

# FFmpeg is a reviewed external input, not an arbitrary PATH dependency.
_ffmpeg_path = os.path.join(PROJECT_ROOT, "vendor", "ffmpeg", "ffmpeg.exe")
_ffmpeg_sha256 = "aa6cdeec90cbc60dce4652d3b15213b49f104b2b9e64d41cc0d5268eb8b0f30f"
if not os.path.isfile(_ffmpeg_path):
    raise FileNotFoundError(
        f"Expected the reviewed FFmpeg executable at {_ffmpeg_path}."
    )
_ffmpeg_digest = hashlib.sha256()
with open(_ffmpeg_path, "rb") as _stream:
    for _chunk in iter(lambda: _stream.read(1024 * 1024), b""):
        _ffmpeg_digest.update(_chunk)
if _ffmpeg_digest.hexdigest() != _ffmpeg_sha256:
    raise RuntimeError("FFmpeg differs from the reviewed binary; do not package it")
datas.append((_ffmpeg_path, "ffmpeg"))

# Twitch VOD and chat ingestion need the CLI on a clean Windows install. Use the
# three files from the official 1.56.4 Windows x64 release together: its
# COPYRIGHT.txt requires both notices to accompany redistributed binaries.
# Release archive: lay295/TwitchDownloader, tag 1.56.4,
# TwitchDownloaderCLI-1.56.4-Windows-x64.zip (SHA-256
# 2d0545fd22d0860aaeafb14afa771270b4351a8535a915fc6c6488d6aec31f42).
_twitchdl_dir = os.path.join(PROJECT_ROOT, "tools", "twitchdownloader")
_twitchdl_files = {
    "TwitchDownloaderCLI.exe": "c2f08e759ef110ed420bbb8419fd7b1cbcae59c80e964f270d7ff6ea1e05472c",
    "COPYRIGHT.txt": "0bbe7b0458bed417f70981a598fd257a5fee32fe3e70ccdf178774b9aa0dbf3e",
    "THIRD-PARTY-LICENSES.txt": "826f542cf7f4ee8e06f75876a6066e2ec370b28803b764990b47b9e50cd742ad",
}
for _name, _expected_hash in _twitchdl_files.items():
    _source = os.path.join(_twitchdl_dir, _name)
    if not os.path.isfile(_source):
        raise FileNotFoundError(f"Expected pinned TwitchDownloader file at {_source}")
    _digest = hashlib.sha256()
    with open(_source, "rb") as _stream:
        for _chunk in iter(lambda: _stream.read(1024 * 1024), b""):
            _digest.update(_chunk)
    if _digest.hexdigest() != _expected_hash:
        raise RuntimeError(f"TwitchDownloader file differs from reviewed release: {_name}")
    datas.append((_source, "twitchdownloader"))

# LGPL compliance paperwork, shipped in the same directory as the binary.
#
# The bundled ffmpeg is an LGPLv3 build (BtbN, --enable-version3 and NO
# --enable-gpl), so conveying it obliges us to pass along the license and a
# written offer for the corresponding source. Both the LGPL and GPL texts are
# required: the LGPL is drafted as additional permissions on top of the GPL.
# "Next to the object code" is the standard the license sets, so these travel
# with ffmpeg.exe rather than living only in the repo. Hard-fail: shipping the
# binary WITHOUT them is the non-compliant case, and it is invisible unless the
# build checks.
_ffmpeg_licensing = os.path.join(PROJECT_ROOT, "vendor", "ffmpeg", "licensing")
for _notice in (
    "LICENSE.LGPLv3.txt",
    "LICENSE.GPLv3.txt",
    "BUILD_IDENTITY.txt",
    "README.md",
):
    _notice_path = os.path.join(_ffmpeg_licensing, _notice)
    if not os.path.exists(_notice_path):
        raise FileNotFoundError(
            f"Expected ffmpeg licensing file at {_notice_path}. The bundled "
            "ffmpeg requires its licensing and source notices; "
            "see vendor/ffmpeg/licensing/README.md."
        )
    datas.append((_notice_path, os.path.join("ffmpeg", "licensing")))

# Backstop: nothing non-commercial may reach the bundle by ANY route.
#
# The VLM lane filter above covers the one path that picks a model dynamically,
# but datas/binaries are also appended by ~15 other blocks and by collect_all()
# over whole packages, and a stray models/experimental copy landing in one of
# those would ship silently. This sweep is the single place that cannot be
# bypassed by adding another append above it. It fails the BUILD rather than
# warning, because a warning in a 10-minute package run is a warning nobody
# reads.
_blocked = []
for _entry in list(datas) + list(binaries):
    _source = _entry[0] if isinstance(_entry, (tuple, list)) else _entry
    _marker = _noncommercial_marker(_source)
    if _marker:
        _blocked.append((_source, _marker))
if _blocked:
    _detail = "\n".join(f"  {src}  (matched: {mark})" for src, mark in _blocked)
    raise SystemExit(
        "recall-engine.spec: refusing to package non-commercially-licensed "
        f"model weights:\n{_detail}\n"
        "These may not be redistributed in a paid product. Remove them from "
        "the packaged set, or update _NONCOMMERCIAL_MARKERS if the license "
        "changed."
    )

# NOTE: there is deliberately no `block_cipher` here. PyInstaller removed
# bytecode encryption in 6.0 and now tells you to delete the argument
# (PyInstaller/building/build_main.py). Passing `cipher=None` was silently
# inert — worse than useless, because it read like protection the bundle did
# not have. Obfuscation of Recall's own artifacts lives in core/model_vault.py.

a = Analysis(
    [os.path.join(PROJECT_ROOT, "apps", "api", "main.py")],
    pathex=pathex,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # matplotlib is still NOT excluded, but the reason is now only historical:
    # it existed because ultralytics' __init__ resolved through a chain that
    # imported matplotlib.pyplot at load time. Ultralytics is gone (replaced by
    # engines/vision/person_onnx.py), and no first-party module imports
    # matplotlib, so this is very likely dead weight worth several MB.
    #
    # Left in place deliberately: paddleocr/paddlex may still pull it in
    # lazily, and a wrong exclusion produces an ImportError only at runtime in
    # the PACKAGED app, which no unit test can catch. Remove it only alongside
    # a real package build plus a scan smoke test.
    excludes=[
        # MediaPipe's optional microphone recorder catches ImportError here.
        # Recall reads recording files through FFmpeg and does not use that API.
        # Excluding it also avoids shipping unused PortAudio/ASIO binaries for
        # three architectures. Real BlazeFace inference was checked without it.
        "sounddevice",
        "_sounddevice",
        "_sounddevice_data",
        # Legacy optional scorer is installed on some development machines.
        # Keep its Python modules out as well as blocking its model weights.
        "hsemotion_onnx",
        # Developer evaluation/training scripts are not application payload.
        # Optional ranker promotion fails closed when that tooling is absent.
        "scripts",
        "core.fixture_autoexport",
        "tkinter",
        "pytest",
        "_pytest",
        "IPython",
        "jupyter",
        "notebook",
        "sphinx",
        "torch.utils.benchmark",
        "torch.distributed.elastic",
        "torch.distributed.fsdp",
        # NOT torch.testing or torch.distributed.rpc, despite the names
        # sounding safe to trim. torch/autograd/gradcheck.py imports
        # torch.testing unconditionally, and torch/_jit_internal.py imports
        # torch.distributed.rpc unconditionally -- both are on the core
        # `import torch` chain (autograd -> nn.modules -> torch.__init__),
        # not opt-in features. Every `import torch` needs them, on every
        # machine, whether or not their nominal purpose (test helpers, RPC)
        # is ever used. Excluding either doesn't trim an unused feature -- it
        # breaks `import torch` entirely, which get_torch_device()'s
        # try/except then silently swallowed into "cpu" (see core/device.py),
        # reporting CPU-only on GPU-equipped machines. The three excludes
        # that remain above were checked: each is only referenced from
        # peripheral torch.profiler / torch.distributed.checkpoint /
        # torch.distributed.launcher modules that are not part of the core
        # init chain.
    ],
    noarchive=False,
    # -O equivalent: strips `assert` statements from the compiled bytecode, so
    # no runtime invariant may depend on one (they are debug aids here, not
    # control flow).
    #
    # NOT optimize=2 (-OO). That additionally strips docstrings, which was the
    # original intent (opening the PYZ handed over `core.ranker_service`'s
    # plan-section reference and design summary verbatim) -- but PyInstaller's
    # `optimize` only changes what gets compiled into the bytecode; it does
    # not make the frozen interpreter report sys.flags.optimize == 2 at
    # runtime. numpy's `_core/overrides.py` checks that live flag before
    # calling `add_docstring`, sees optimize < 2, and tries to attach a
    # docstring anyway -- which is now None, so numpy (and therefore cv2, and
    # therefore the whole engine) fails on import with "argument docstring of
    # add_docstring should be a str" before ever binding to a port. This is a
    # known PyInstaller/numpy incompatibility, not fixable by cleaning the
    # build cache. Level 1 has no such runtime-flag dependency in numpy.
    optimize=1,
)

# PATH may contain unrelated application runtimes. Do not silently redistribute
# their DLLs merely because PyInstaller found them before the intended runtime.
a.binaries = pin_microsoft_runtime_inputs(a.binaries, _microsoft_runtime_root)
# These helpers have no static consumers in the frozen native graph. Recall
# does not invoke CUDA profiling or multi-GPU solver APIs. A synthetic scan
# with three GPU perception workers, caption export and two rerenders passed
# with both cuSOLVERMg and alternate NVRTC omitted. Keep the exact-named files
# out of release output, even if a future torch hook discovers them.
_omitted_cuda_files = {
    "nvperf_host.dll", "cusolvermg64_11.dll", "nvrtc64_120_0.alt.dll",
}
a.binaries = [
    entry for entry in a.binaries
    if os.path.basename(entry[0]).casefold() not in _omitted_cuda_files
]
verify_native_inputs(a.binaries, [
    sys.prefix, sys.base_prefix, os.path.join(PROJECT_ROOT, "vendor"),
    _twitchdl_dir,  # Exact CLI bytes were pinned before Analysis.
    os.environ.get("SystemRoot", r"C:\Windows"),
])

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="recall-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Electron launches the engine with windowsHide=True. Keep the console
    # bootloader so Python and native libraries receive valid standard handles;
    # the window remains hidden in the product, while direct smoke runs retain
    # actionable startup diagnostics.
    # console=False: the engine is a loopback HTTP server that nobody reads
    # stdout from -- the bundled path spawns it with stdio "ignore". Built as a
    # console app it gets its own black window on Windows, which is what users
    # were left staring at beside the real UI. Diagnostics now go to the
    # Electron log file instead (see diag() in apps/desktop/electron/main.ts).
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="recall-engine",
)

# Catch a future hook that adds an omitted file through data collection or a
# different binary destination, rather than silently distributing it again.
for _folder, _dirs, _files in os.walk(coll.name):
    if any(name.casefold() in _omitted_cuda_files for name in _files):
        raise RuntimeError("Unreviewed CUDA helper entered the release payload: " + _folder)

_twitchdl_output = os.path.join(coll.name, coll.contents_directory or "", "twitchdownloader")
for _name, _expected_hash in _twitchdl_files.items():
    _output = os.path.join(_twitchdl_output, _name)
    if not os.path.isfile(_output):
        raise RuntimeError(f"Packaged TwitchDownloader file is missing: {_name}")
    _digest = hashlib.sha256()
    with open(_output, "rb") as _stream:
        for _chunk in iter(lambda: _stream.read(1024 * 1024), b""):
            _digest.update(_chunk)
    if _digest.hexdigest() != _expected_hash:
        raise RuntimeError(f"Packaged TwitchDownloader file changed: {_name}")
print("recall-engine.spec: TwitchDownloader CLI and notices verified")

# Verify copied bytes, including any additional files introduced by analysis
# hooks. A successful PyInstaller exit must mean the model payload also passed.
_model_verification = verify_model_bundle(
    os.path.join(coll.name, coll.contents_directory or "", "models"),
    _expected_model_payload,
)
with open(os.path.join(coll.name, "model-payload-verification.json"), "w", encoding="utf-8") as _report:
    json.dump(_model_verification, _report, indent=2)
print("recall-engine.spec: final model payload verified against reviewed inputs")
verify_release_privacy(coll.name)
_dependency_notice_verification = verify_notice_payload(
    os.path.join(coll.name, coll.contents_directory or "", "notices", "python-dependencies"),
    _expected_dependency_notices,
)
with open(os.path.join(coll.name, "dependency-notice-verification.json"), "w", encoding="utf-8") as _report:
    json.dump(_dependency_notice_verification, _report, indent=2)
