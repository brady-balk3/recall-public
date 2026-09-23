# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
r"""One front door for the model weights a clone does not come with.

GitHub is not a weight host, so large models are fetched rather than
committed. They were fetchable before this script existed -- four separate
scripts, of which the README named two -- which meant a fresh clone following
the README ended up with no ASR at all and no error to say so, because a missing
ASR disables transcripts silently rather than raising.

    venv\Scripts\python.exe scripts/fetch_models.py --check   # what is missing
    venv\Scripts\python.exe scripts/fetch_models.py           # fetch the rest

ASR and aligner downloads use immutable revisions and verified file hashes.
Model distribution and licensing review remain separate release requirements.

TIERS
  runtime   Everything the engine needs to scan a VOD end to end. Without any
            one of these a lane turns itself off quietly, which is the failure
            mode this script exists to make loud.
  packaging Additionally required by apps/api/recall-engine.spec. Judge models
            use the installer's pinned Hugging Face revisions and checksums;
            --all downloads them automatically, along with the bundled OCR,
            face, and audio-event assets. Fetching does not establish release readiness.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from core.bundle_paths import get_models_dir
from core.model_catalog import ASR_FILES, ALIGNER_FILES, VAD_FILES, SPEAKER_FILES
from core.model_downloads import matches
from core.installer_models import download_groups, install_models
from pathlib import Path
from scripts.fetch_person_model import MODELS as PERSON_MODELS, sha256
from scripts.fetch_memory_embedding_model import FILES as MEMORY_FILES, LICENSE_SHA256
from scripts.fetch_small_release_assets import OCR_MODELS, YAMNET, FACE_EXPECTED

MODELS = get_models_dir()


class Model:
    """One weight group: how to tell it is here, and how to get it."""

    def __init__(self, name, tier, present_if, script=None, manual=None, note="", artifacts=None, download_group=None, checksums=None, script_args=()):
        self.name = name
        self.tier = tier
        self.present_if = present_if      # paths under models/, all must exist
        self.script = script              # scripts/<script> fetches it
        self.manual = manual              # instructions when there is no script
        self.note = note
        self.artifacts = artifacts or {}
        self.download_group = download_group
        self.checksums = checksums or {}
        self.script_args = script_args

    def present(self) -> bool:
        return not self.missing()

    def missing(self) -> list[str]:
        missing = []
        for relative in self.present_if:
            path = os.path.join(MODELS, relative)
            if (
                not os.path.exists(path)
                or not os.path.isfile(path)
                or (os.path.isfile(path) and os.path.getsize(path) == 0)
            ):
                missing.append(relative)
        for relative, expected in self.artifacts.items():
            if relative not in missing and not matches(Path(MODELS) / relative, expected):
                missing.append(relative)
        for relative, expected in self.checksums.items():
            if relative not in missing and sha256(os.path.join(MODELS, relative)) != expected:
                missing.append(relative)
        return missing


def installer_judge(name, prefix, note):
    """Use exactly the same filenames, revisions and hashes as installed setup."""
    group = next(group for group in download_groups()
                 if all(spec[0].startswith(prefix + "/") for spec in group[2].values()))
    artifacts = {relative: (size, digest) for relative, size, digest in group[2].values()}
    return Model(name, "packaging", list(artifacts), note=note,
                 artifacts=artifacts, download_group=group)


MODEL_SET = [
    Model("Person detection (YOLOX-s, Apache-2.0)", "runtime",
          ["vision/yolox_s.onnx"], script="fetch_person_model.py",
          artifacts={"vision/yolox_s.onnx": (PERSON_MODELS["yolox_s.onnx"]["size"],
                                             PERSON_MODELS["yolox_s.onnx"]["sha256"])},
          note="facecam validation"),
    Model("Speech recognition (Qwen3-ASR-1.7B, Apache-2.0)", "runtime",
          ["asr/qwen3-asr-1.7b/Qwen3-ASR-1.7B-Q8_0.gguf",
           "asr/qwen3-asr-1.7b/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"],
          script="fetch_qwen_asr.py",
          artifacts={"asr/qwen3-asr-1.7b/" + name: spec for name, spec in ASR_FILES.items()},
          note="transcripts, clip titles, burned-in captions"),
    Model("Word alignment (Qwen3-ForcedAligner-0.6B, Apache-2.0)", "runtime",
          ["asr/qwen3-aligner-0.6b/" + name for name in (
              "onnx/model.onnx", "onnx/model.onnx_data", "config.json",
              "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
              "added_tokens.json", "special_tokens_map.json", "export_metadata.json",
          )],
          script="fetch_qwen_aligner.py",
          artifacts={"asr/qwen3-aligner-0.6b/" + name: spec for name, spec in ALIGNER_FILES.items()
                     if name != "onnx/model_q4.onnx"},
          note="caption word timings; Qwen3-ASR emits no timestamps of its own"),
    Model("Speech activity detection (Silero VAD)", "runtime",
          ["asr/silero-vad/" + name for name in VAD_FILES],
          script="fetch_audio_support_models.py",
          artifacts={"asr/silero-vad/" + name: spec for name, spec in VAD_FILES.items()},
          note="speech chunk boundaries"),
    Model("Speaker attribution (WeSpeaker)", "runtime",
          ["speaker/" + name for name in SPEAKER_FILES],
          script="fetch_audio_support_models.py",
          artifacts={"speaker/" + name: spec for name, spec in SPEAKER_FILES.items()},
          note="speaker attribution for transcript segments"),
    Model("Stream Memory embeddings (all-MiniLM-L6-v2, Apache-2.0)", "runtime",
          ["memory/all-minilm-l6-v2/model_quint8_avx2.onnx",
           "memory/all-minilm-l6-v2/tokenizer.json", "memory/all-minilm-l6-v2/LICENSE"],
          script="fetch_memory_embedding_model.py",
          checksums={"memory/all-minilm-l6-v2/" + name: digest
                     for name, digest in [*MEMORY_FILES.values(), ("LICENSE", LICENSE_SHA256)]},
          note="local semantic search over past streams"),
    Model("Audio events (YAMNet, Apache-2.0)", "optional",
          ["yamnet.onnx", "yamnet_class_map.csv"],
          script="fetch_small_release_assets.py", script_args=("--group", "yamnet"),
          artifacts={name: (size, digest) for name, _repo, _revision, size, digest in YAMNET},
          note="laughter/scream channel; the reaction engine fuses without it"),
    Model("Face Landmarker (MediaPipe, Apache-2.0)", "packaging",
          ["face/face_landmarker.task"],
          script="fetch_small_release_assets.py", script_args=("--group", "face"),
          artifacts={"face/face_landmarker.task": FACE_EXPECTED},
          note="face geometry used by the packaged engine"),
    Model("Text in video (PP-OCRv6 small, Apache-2.0)", "packaging",
          [f"ocr/{model}/{name}" for model, (_revision, files) in OCR_MODELS.items()
           for name, _size, _digest, _upstream in files],
          script="fetch_small_release_assets.py", script_args=("--group", "ocr"),
          artifacts={f"ocr/{model}/{name}": (size, digest)
                     for model, (_revision, files) in OCR_MODELS.items()
                     for name, size, digest, _upstream in files},
          note="bundled OCR detection and recognition"),
    installer_judge("Semantic judge (Qwen3-4B-Instruct-2507 GGUF, Apache-2.0)", "llm",
                    "runtime-optional: the judge disables itself when absent"),
    installer_judge("Visual judge (Qwen3.5-4B GGUF + mmproj, Apache-2.0)", "vlm",
                    "runtime-optional: visual review needs both the model and projection"),
]

TIER_ORDER = {"runtime": 0, "optional": 1, "packaging": 2}


def report(include_packaging: bool = False) -> int:
    missing_required = 0
    for tier in ("runtime", "optional", "packaging"):
        entries = [m for m in MODEL_SET if m.tier == tier]
        if not entries:
            continue
        print(f"\n[{tier}]")
        for model in entries:
            if model.present():
                print(f"  ok      {model.name}")
                continue
            print(f"  MISSING {model.name}")
            print(f"          {model.note}")
            if model.download_group:
                print("          fetch: python scripts/fetch_models.py --all")
            elif model.script:
                print(f"          fetch: python scripts/{model.script} {' '.join(model.script_args)}".rstrip())
            else:
                print(f"          {model.manual}")
            if tier == "runtime" or (include_packaging and tier in ("optional", "packaging")):
                missing_required += 1
    print()
    return missing_required


def fetch(models, dry_run: bool) -> int:
    failures = 0
    for model in models:
        if model.present():
            print(f"-- {model.name}: already present")
            continue
        if model.download_group:
            repo, revision, _files = model.download_group
            print(f"-- {model.name}\n   Hugging Face: {repo} at {revision}")
            if not dry_run:
                try:
                    install_models(Path(MODELS), groups=(model.download_group,))
                    if not model.present():
                        raise RuntimeError("Downloaded judge files did not verify")
                except Exception as exc:
                    print(f"   FAILED: {model.name}: {exc}")
                    failures += 1
            continue
        if not model.script:
            print(f"-- {model.name}: no fetcher; {model.manual}")
            failures += 1
            continue
        cmd = [sys.executable, os.path.join(ROOT, "scripts", model.script), *model.script_args]
        print(f"-- {model.name}\n   {' '.join(cmd[1:])}")
        if dry_run:
            continue
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            print(f"   FAILED: {model.name}")
            failures += 1
        elif not model.present():
            print(f"   INCOMPLETE: {model.name}")
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="report what is present or missing and exit")
    parser.add_argument("--all", action="store_true",
                        help="also fetch what only packaging needs")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the fetch commands without running them")
    args = parser.parse_args()

    if args.check:
        missing = report(include_packaging=args.all)
        if missing:
            print(f"{missing} required model group(s) missing -- "
                  "install them before using their features.")
        return 1 if missing else 0

    wanted = [m for m in MODEL_SET if m.tier == "runtime" or args.all]
    failures = fetch(wanted, args.dry_run)
    if not args.dry_run:
        print("\n-- final state --")
        failures += report(include_packaging=args.all)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
