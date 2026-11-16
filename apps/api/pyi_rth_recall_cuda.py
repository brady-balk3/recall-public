# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""PyInstaller runtime hook: put the bundle's nvidia-cuda folder on PATH.

Runs before apps/api/main.py and before any engine module, in the API process
and in every frozen worker (recall-engine.spec, runtime_hooks).

Why it has to run this early: llama_cpp loads llama.dll with winmode=0, so
ggml-cuda.dll resolves its static imports cudart64_12.dll and cublas64_12.dll
through the legacy DLL search order, PATH included, and ignores
os.add_dll_directory. The engines import llama_cpp before they ask core.device
anything (QwenASRAdapter, the semantic and visual judges), so a registration
made later is too late, and llama.dll would fail to load at all. The bundle
used to dodge that with its own 608 MB copy of the three DLLs in llama_cpp/lib;
with the folder on PATH from the start, llama.cpp and onnxruntime share the one
copy in nvidia-cuda. Measured 2026-11-15 in a dev process without those copies:
llama_supports_gpu_offload() was True with the PATH entry, and `import
llama_cpp` failed with "Failed to load shared library" without it.

Registering loads no DLL and needs no GPU, so CPU-only processes pay nothing.
"""


def _recall_cuda_search_path():
    from core.device import prepare_llama_cuda

    prepare_llama_cuda()


_recall_cuda_search_path()
del _recall_cuda_search_path
