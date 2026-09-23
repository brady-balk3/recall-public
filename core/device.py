# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Central accelerator selection (handbook/20 §3.2).

Detects the best available compute device once and exposes it to the engines, so
GPU usage is opportunistic with a clean CPU fallback — the product must still run
on any machine (handbook/19 §7).

- torch (YOLO, Whisper): CUDA if available, else CPU.
- onnxruntime (PP-OCRv6, HSEmotion, Parakeet): prefer CUDA > DirectML > CPU
  providers IF the corresponding onnxruntime build is installed AND the CUDA EP
  can actually initialize; otherwise CPU.

onnxruntime-gpu needs CUDA/cuDNN DLLs on the loader path. We use torch's bundled
CUDA 12 + cuDNN 9 (cu126) via ``onnxruntime.preload_dlls()`` so ORT and torch
share one runtime. Pin ``onnxruntime-gpu`` to a CUDA-12 build (≤1.26.x) — 1.27+
defaults to CUDA 13 and will not load against torch's cublasLt64_12.dll.
"""

_torch_device = None
_ort_providers = None
_ort_cuda_libs_ready = False
_ort_cuda_ep_ok = None  # None=untested, True/False after probe


def reset_device_cache() -> None:
    """Forget cached probes so an explicit idle-time rescan can run again."""
    global _torch_device, _ort_providers, _ort_cuda_libs_ready, _ort_cuda_ep_ok
    _torch_device = None
    _ort_providers = None
    _ort_cuda_libs_ready = False
    _ort_cuda_ep_ok = None


def _ensure_ort_cuda_libs() -> None:
    """Load CUDA/cuDNN DLLs for ORT (torch import + ort.preload_dlls)."""
    global _ort_cuda_libs_ready
    if _ort_cuda_libs_ready:
        return
    _ort_cuda_libs_ready = True
    try:
        import torch  # noqa: F401 - loads bundled CUDA/cuDNN into the process
    except Exception:  # noqa: BLE001
        pass
    try:
        import onnxruntime as ort
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            # Share torch's CUDA 12 / cuDNN 9 DLLs with ORT (CUDA EP docs).
            preload(cuda=True, cudnn=True, msvc=True)
    except Exception:  # noqa: BLE001 - CPU fallback still works
        pass


def _probe_ort_cuda_ep() -> bool:
    """True when CUDAExecutionProvider can create a real session (not just list)."""
    global _ort_cuda_ep_ok
    if _ort_cuda_ep_ok is not None:
        return _ort_cuda_ep_ok
    _ensure_ort_cuda_libs()
    try:
        import numpy as np
        import onnxruntime as ort  # noqa: F401

        from onnxruntime import OrtValue

        # Tiny float32 allocation on CUDA — cheaper than shipping a probe .onnx.
        # If CUDA EP can't load its DLLs this raises the same provider_bridge
        # error the engines would hit at session create.
        _ = OrtValue.ortvalue_from_numpy(np.zeros((1,), dtype=np.float32), "cuda", 0)
        _ort_cuda_ep_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"ORT CUDA EP unavailable ({exc}); using CPUExecutionProvider")
        _ort_cuda_ep_ok = False
    return _ort_cuda_ep_ok


def get_torch_device() -> str:
    """Return 'cuda' if a usable CUDA GPU is present, else 'cpu'. Cached."""
    global _torch_device
    if _torch_device is None:
        try:
            import torch
            _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:  # noqa: BLE001
            # A bare `except: pass` here once hid a packaging bug (2026-08-06):
            # the frozen build excluded torch.testing / torch.distributed.rpc,
            # both load-bearing on torch's own `import torch` chain, so every
            # GPU machine silently fell back to "cpu" with zero signal. Log
            # the real reason so a future bad exclude/DLL issue is visible
            # instead of only ever showing up as "why is this GPU box on CPU".
            print(f"get_torch_device() failed, falling back to cpu: {exc!r}")
            _torch_device = "cpu"
    return _torch_device


def get_ort_providers():
    """Ordered onnxruntime providers: CUDA > DirectML > CPU, filtered to usable."""
    global _ort_providers
    if _ort_providers is None:
        try:
            import onnxruntime as ort
            available = set(ort.get_available_providers())
        except Exception:  # noqa: BLE001
            available = set()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in providers:
            _ensure_ort_cuda_libs()
            if not _probe_ort_cuda_ep():
                providers = [p for p in providers if p != "CUDAExecutionProvider"]
                if not providers:
                    providers = ["CPUExecutionProvider"]
        _ort_providers = providers
    elif "CUDAExecutionProvider" in _ort_providers:
        _ensure_ort_cuda_libs()
    return _ort_providers


def device_summary() -> str:
    return (
        f"torch={get_torch_device()} onnxruntime={get_ort_providers()} "
        f"llama_gpu={llama_gpu_offload()}"
    )


def llama_gpu_offload():
    """Whether llama-cpp-python can offload to GPU; None when not installed.

    Surfaced in every scan's device log because a CPU-only wheel silently ran
    the semantic and visual judges ~10x slower for weeks (2026-07-19). Fix:
    scripts/setup_gpu_llama.py.
    """
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:  # noqa: BLE001 - optional dependency; absence is a state
        return None
