# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Fetch the Qwen3-ASR-1.7B GGUF pair -- the words half of the ASR stack.

Runs on the llama.cpp already installed: the bundled mtmd carries
`clip_graph_qwen3a` and reports `inp_audio=True` for the shipped projector, so
there is no second interpreter and no `transformers` to install.

Timings come from the forced aligner, which is a separate download --
`scripts/fetch_qwen_aligner.py`. Both are needed; Qwen3-ASR emits plain text
with no timestamp tokens at all.

    venv\\Scripts\\python.exe scripts\\fetch_qwen_asr.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.bundle_paths import get_models_dir
from core.model_catalog import ASR_REPO as REPO, ASR_REVISION as REVISION, ASR_FILES
from core.model_downloads import install_files

FILES = list(ASR_FILES)
TARGET = os.path.join(get_models_dir(), "asr", "qwen3-asr-1.7b")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=TARGET)
    args = parser.parse_args(argv)

    print(f"{REPO}@{REVISION} -> {args.target}")
    install_files(REPO, REVISION, ASR_FILES, args.target)
    print("All ASR files verified.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
