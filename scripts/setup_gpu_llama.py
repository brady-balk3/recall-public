# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Install the CUDA build of llama-cpp-python and verify GPU offload.

The PyPI llama-cpp-python wheel is CPU-only. On this stack that silently ran
the semantic text judge AND the visual judge ~10x slower than a GPU build
(measured 2026-07-19: ~48s -> ~5s per visual verdict on an RTX 4070 Ti
SUPER). This installs the prebuilt CUDA wheel from the official cu124 index
(same version as requirements.txt, so nothing else moves).

ggml-cuda.dll imports cudart64_12 and cublas64_12 (-> cublasLt64_12), and
llama_cpp loads it with the legacy search order (PATH). Nothing is copied for
that: core.device.prepare_llama_cuda() puts the bin folders of NVIDIA's
nvidia-cuda-runtime-cu12 and nvidia-cublas-cu12 wheels (scripts/requirements.txt)
on PATH, and every engine import of llama_cpp calls it first. The packaged
engine gets the same from its runtime hook (apps/api/pyi_rth_recall_cuda.py).
This script's checks call it too, so they see what the engine sees.

Idempotent and verify-first: run it again after any llama-cpp-python upgrade
or venv rebuild. Exits non-zero when GPU offload is still unavailable so
CI/packaging can gate on it.

Usage:
    venv\\Scripts\\python.exe scripts\\setup_gpu_llama.py            # install + verify
    venv\\Scripts\\python.exe scripts\\setup_gpu_llama.py --verify   # check only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

LLAMA_VERSION = "0.3.34"  # keep in lock-step with scripts/requirements.txt
CUDA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Run in a fresh interpreter after a reinstall, so the DLL search is evaluated
# from scratch. Same order as the engine: register the CUDA folders, then import.
PROBE = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from core.device import prepare_llama_cuda; prepare_llama_cuda(); "
    "import llama_cpp; "
    "sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)"
)


def gpu_offload_supported() -> bool:
    try:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from core.device import prepare_llama_cuda
        prepare_llama_cuda()
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001 - any failure means "not working"
        print(f"llama_cpp not loadable: {exc}")
        return False


def install_cuda_wheel() -> bool:
    cmd = [
        sys.executable, "-m", "pip", "install",
        f"llama-cpp-python=={LLAMA_VERSION}",
        "--force-reinstall", "--no-deps", "--index-url", CUDA_INDEX,
    ]
    print("+", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="check GPU offload without installing anything")
    args = parser.parse_args(argv)

    if gpu_offload_supported():
        print("OK: llama-cpp-python already supports GPU offload.")
        return 0
    if args.verify:
        print("FAIL: llama-cpp-python has no GPU offload (CPU wheel, or the "
              "nvidia-*-cu12 wheels are missing). Run without --verify to fix.")
        return 1

    if not install_cuda_wheel():
        print("FAIL: CUDA wheel install failed; the CPU wheel remains usable.")
        return 1
    probe = subprocess.run([sys.executable, "-c", PROBE, REPO_ROOT])
    if probe.returncode == 0:
        print("OK: GPU offload verified.")
        return 0
    print("FAIL: GPU offload still unavailable after install; install the pinned "
          "nvidia-*-cu12 wheels (scripts/requirements.txt). The app will run the "
          "judges on CPU (slow but functional).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
