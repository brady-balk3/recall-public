# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Fetch the Qwen3-ForcedAligner ONNX export -- word timings without whisper.

Why ONNX and not the GGUF everything else in this repo runs on: the aligner's
output comes from a word-level classification head, and `42ailab`'s GGUF build
ships that head as a separate `aligner-head.bin` outside the GGUF, for their own
`42model` engine. The installed llama.cpp has no code for it -- `mtmd.dll`
carries `clip_graph_qwen3a`, but neither it nor `llama.dll` contains any aligner
or classification-head symbol, so loading it there would generate text and no
times, silently.

The ONNX export bakes the head into the graph:

    inputs   input_ids [batch, text]   input_features [batch, 128, frames]
             attention_mask            feature_attention_mask
    output   logits [batch, text, 5000]

and runs on the `onnxruntime-gpu` this app already ships for `hsemotion-onnx`.
No new dependency -- see `docs/ASR_REPLACEMENT_PLAN.md`.

Default is fp32 (3.7 GB), which the export validates against the reference
implementation at max abs diff 4.1e-05. ``--q4`` is 1.0 GB but its own smoke
test reports max abs diff 7.8 on the logits and the upstream README frames it as
browser-friendly rather than accurate; measure before trusting it.

    venv\\Scripts\\python.exe scripts\\fetch_qwen_aligner.py
    venv\\Scripts\\python.exe scripts\\fetch_qwen_aligner.py --q4
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.bundle_paths import get_models_dir
from core.model_catalog import ALIGNER_REPO as REPO, ALIGNER_REVISION as REVISION, ALIGNER_FILES
from core.model_downloads import install_files

SUPPORT = [name for name in ALIGNER_FILES if not name.startswith("onnx/")]
FP32 = ["onnx/model.onnx", "onnx/model.onnx_data"]
Q4 = ["onnx/model_q4.onnx"]
TARGET = os.path.join(get_models_dir(), "asr", "qwen3-aligner-0.6b")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q4", action="store_true",
                        help="1.0 GB quantized variant instead of 3.7 GB fp32")
    parser.add_argument("--target", default=TARGET)
    args = parser.parse_args(argv)

    files = {name: ALIGNER_FILES[name] for name in SUPPORT + (Q4 if args.q4 else FP32)}
    print(f"{REPO}@{REVISION} -> {args.target}")
    install_files(REPO, REVISION, files, args.target)
    print("All aligner files verified.")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
