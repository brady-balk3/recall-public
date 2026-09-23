# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Install the CUDA build of llama-cpp-python and make its DLLs loadable.

The PyPI llama-cpp-python wheel is CPU-only. On this stack that silently ran
the semantic text judge AND the visual judge ~10x slower than a GPU build
(measured 2026-07-19: ~48s -> ~5s per visual verdict on an RTX 4070 Ti
SUPER). Two things are needed on Windows, both automated here:

  1. the prebuilt CUDA wheel from the official cu124 index (same version as
     requirements.txt, so nothing else moves);
  2. CUDA runtime DLLs next to llama.dll — the wheel expects cudart/cublas on
     PATH, and the most reliable local source is torch's own bundled copies
     (torch+cu126 ships cudart64_12/cublas64_12/cublasLt64_12).

Idempotent and verify-first: run it again after any llama-cpp-python upgrade
or venv rebuild (a reinstall wipes the colocated DLLs). Exits non-zero when
GPU offload is still unavailable so CI/packaging can gate on it.

Usage:
    venv\\Scripts\\python.exe scripts\\setup_gpu_llama.py            # install + verify
    venv\\Scripts\\python.exe scripts\\setup_gpu_llama.py --verify   # check only
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

LLAMA_VERSION = "0.3.34"  # keep in lock-step with scripts/requirements.txt
CUDA_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
CUDA_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")


def gpu_offload_supported() -> bool:
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001 - any failure means "not working"
        print(f"llama_cpp not loadable: {exc}")
        return False


def torch_lib_dir() -> str | None:
    try:
        import torch
        if not torch.cuda.is_available():
            print("torch has no CUDA — install the +cu126 torch build first "
                  "(see scripts/requirements.txt GPU note).")
            return None
        return os.path.join(os.path.dirname(torch.__file__), "lib")
    except Exception as exc:  # noqa: BLE001
        print(f"torch not importable: {exc}")
        return None


def _llama_package_dir() -> str | None:
    # Do NOT import llama_cpp here: right after the CUDA wheel install the
    # native lib cannot load precisely BECAUSE the DLLs are still missing.
    import importlib.util
    spec = importlib.util.find_spec("llama_cpp")
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    return locations[0] if locations else None


def colocate_dlls() -> bool:
    lib = torch_lib_dir()
    if lib is None:
        return False
    package_dir = _llama_package_dir()
    if package_dir is None:
        print("llama_cpp package not found")
        return False
    target = os.path.join(package_dir, "lib")
    if not os.path.isdir(target):
        matches = glob.glob(
            os.path.join(package_dir, "**", "llama.dll"), recursive=True,
        )
        if not matches:
            print("could not locate llama.dll inside the llama_cpp package")
            return False
        target = os.path.dirname(matches[0])
    copied = []
    for name in CUDA_DLLS:
        src = os.path.join(lib, name)
        dst = os.path.join(target, name)
        if not os.path.isfile(src):
            print(f"missing {name} in {lib}")
            return False
        if not os.path.isfile(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copy2(src, dst)
            copied.append(name)
    print(f"CUDA DLLs colocated into {target}"
          + (f" (copied: {', '.join(copied)})" if copied else " (already present)"))
    return True


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
        print("FAIL: llama-cpp-python has no GPU offload "
              "(CPU wheel, or CUDA DLLs missing). Run without --verify to fix.")
        return 1

    if not install_cuda_wheel():
        print("FAIL: CUDA wheel install failed; the CPU wheel remains usable.")
        return 1
    # A fresh interpreter is needed after reinstall; do the DLL step in-process
    # (llama_cpp import may fail until DLLs are in place, which colocate handles
    # by locating the package directory without importing the native lib).
    if not colocate_dlls():
        return 1
    # Verify in a clean interpreter so DLL search paths are evaluated fresh.
    probe = subprocess.run([
        sys.executable, "-c",
        "import llama_cpp; import sys; "
        "sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)",
    ])
    if probe.returncode == 0:
        print("OK: GPU offload verified.")
        return 0
    print("FAIL: GPU offload still unavailable after install; "
          "the app will run the judges on CPU (slow but functional).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
