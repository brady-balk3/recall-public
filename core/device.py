# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Central accelerator selection (handbook/20 §3.2).

Detects the best available compute device once and exposes it to the engines, so
GPU usage is opportunistic with a clean CPU fallback — the product must still run
on any machine (handbook/19 §7).

- GPU work (llama.cpp offload, OCR, person detection, the aligner): CUDA when
  an NVIDIA GPU is visible to this process AND onnxruntime's CUDA EP can
  actually initialize on it (``get_torch_device``); otherwise CPU.
- onnxruntime (PP-OCRv6, HSEmotion, Parakeet): prefer CUDA > DirectML > CPU
  providers IF the corresponding onnxruntime build is installed AND the CUDA EP
  can actually initialize; otherwise CPU.

Nothing here imports torch. It used to, for two things: ``torch.cuda.
is_available()`` as the GPU gate, and ``import torch`` purely so torch's
bundled CUDA/cuDNN DLLs were in the process before ``onnxruntime.
preload_dlls()``. Measured 2026-11-11: that import cost every process that
asked 1.3 s and ~477 MB RSS (docs/TORCH_REMOVAL_SPIKE_2026-11.md). The gate is
now NVML (nvml.dll ships with the NVIDIA driver) plus the existing ORT probe,
and the DLLs are loaded by path from NVIDIA's own nvidia-*-cu12 runtime wheels
(dev) or the bundle's ``nvidia-cuda`` folder (packaged engine). Those are
byte-identical to the copies torch/lib carried (sha256, 2026-11-15), so the
GPU code runs on the same binaries as before.

onnxruntime-gpu needs CUDA 12 + cuDNN 9 DLLs on the loader path. Pin
``onnxruntime-gpu`` to a CUDA-12 build (≤1.26.x) — 1.27+ defaults to CUDA 13
and will not load against the cublasLt64_12.dll we ship.
"""

import importlib.util
import os
import re

_torch_device = None
_ort_providers = None
_ort_cuda_libs_ready = False
_ort_cuda_ep_ok = None  # None=untested, True/False after probe
_nvml_error_reported = False
# os.add_dll_directory handles, kept so the registrations stay inspectable
# (and could be undone with close()); nothing here ever closes them.
_dll_directory_handles = []
_registered_dll_dirs = set()  # normcase'd folders already given to _register_dll_dirs
_cuda_dll_provider = None  # (name, [dirs]) the DLLs were loaded from; diagnostics


def reset_device_cache() -> None:
    """Forget cached probes so an explicit idle-time rescan can run again."""
    global _torch_device, _ort_providers, _ort_cuda_libs_ready, _ort_cuda_ep_ok
    _torch_device = None
    _ort_providers = None
    _ort_cuda_libs_ready = False
    _ort_cuda_ep_ok = None


# --- GPU presence: NVML, no CUDA context ------------------------------------
#
# NVML answers "which NVIDIA GPUs exist, what are they called, how much VRAM is
# free" without initialising CUDA: no context, no VRAM, no cuInit. nvidia-smi
# is a front end to the same library. It does NOT honour CUDA_VISIBLE_DEVICES,
# so _visible_ordinals applies CUDA's rules for that variable itself.

_NVML_SUCCESS = 0
_NVML_NAME_LEN = 96  # NVML_DEVICE_NAME_V2_BUFFER_SIZE
_NVML_UUID_LEN = 96  # >= NVML_DEVICE_UUID_V2_BUFFER_SIZE (96)


def _load_nvml():
    """The NVML library, or None when no NVIDIA driver is installed."""
    import ctypes

    if os.name == "nt":
        # Current drivers install nvml.dll into System32; drivers before R460
        # kept it beside nvidia-smi only.
        program_files = os.environ.get("ProgramW6432") or os.environ.get(
            "ProgramFiles", r"C:\Program Files")
        candidates = [
            "nvml.dll",
            os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvml.dll"),
        ]
    else:
        candidates = ["libnvidia-ml.so.1"]
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


def _query_nvml():
    """(gpus, error) for every GPU NVML reports, in NVML (= nvidia-smi) order.

    Each GPU is {"index", "name", "uuid", "total_bytes", "free_bytes"}.
    ``error`` is None on success, "no_driver" when NVML cannot be loaded (no
    NVIDIA driver: the normal state of an AMD/Intel machine), or a message when
    the driver is there but NVML failed. Any failure yields no GPUs.
    """
    import ctypes

    nvml = _load_nvml()
    if nvml is None:
        return [], "no_driver"

    class _Memory(ctypes.Structure):  # nvmlMemory_t (v1)
        _fields_ = [("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    try:
        rc = nvml.nvmlInit_v2()
        if rc != _NVML_SUCCESS:
            return [], f"nvmlInit_v2 returned {rc}"
        try:
            count = ctypes.c_uint(0)
            rc = nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
            if rc != _NVML_SUCCESS:
                return [], f"nvmlDeviceGetCount_v2 returned {rc}"
            gpus = []
            for index in range(count.value):
                handle = ctypes.c_void_p()
                rc = nvml.nvmlDeviceGetHandleByIndex_v2(
                    ctypes.c_uint(index), ctypes.byref(handle))
                if rc != _NVML_SUCCESS:
                    return [], f"nvmlDeviceGetHandleByIndex_v2({index}) returned {rc}"
                name = ctypes.create_string_buffer(_NVML_NAME_LEN)
                uuid = ctypes.create_string_buffer(_NVML_UUID_LEN)
                memory = _Memory()
                if (nvml.nvmlDeviceGetName(handle, name, ctypes.c_uint(_NVML_NAME_LEN))
                        != _NVML_SUCCESS
                        or nvml.nvmlDeviceGetUUID(handle, uuid, ctypes.c_uint(_NVML_UUID_LEN))
                        != _NVML_SUCCESS
                        or nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
                        != _NVML_SUCCESS):
                    return [], f"NVML device query failed for GPU {index}"
                gpus.append({
                    "index": index,
                    "name": name.value.decode("utf-8", "replace").strip(),
                    "uuid": uuid.value.decode("ascii", "replace").strip(),
                    "total_bytes": int(memory.total),
                    "free_bytes": int(memory.free),
                })
            return gpus, None
        finally:
            nvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 - a missing export on an old driver
        return [], f"NVML query failed: {exc!r}"


def nvml_gpus():
    """Every NVIDIA GPU NVML reports (ignores CUDA_VISIBLE_DEVICES); [] if none."""
    return _query_nvml()[0]


def _leading_int(text: str) -> int:
    """strtoul-style: the integer ``text`` starts with, or -1 when there is none.

    CUDA parses each CUDA_VISIBLE_DEVICES entry this way, so "1gpu" means 1.
    """
    match = re.match(r"[+-]?\d+", text)
    return int(match.group(0)) if match else -1


def _visible_ordinals(env_value, uuids):
    """NVML ordinals that CUDA_VISIBLE_DEVICES leaves visible to this process.

    Follows CUDA's own rules, as torch.cuda._parse_visible_devices and
    _device_count_nvml implement them: unset means every GPU; the list is read
    left to right and stops at the first invalid entry (so "-1", "" and
    "none" hide everything); an index past the last GPU also stops it; a
    repeated entry empties the whole list. "GPU-<uuid prefix>" entries match a
    unique UUID prefix. MIG ids cannot be mapped to a GPU without the MIG API,
    so any MIG list is taken to mean "the first GPU".

    The ordinals are NVML's (PCI bus) order. CUDA's default order is fastest
    first, so on a machine with two different NVIDIA GPUs an explicit index
    list may name a different card than CUDA would; the GPU/CPU answer is the
    same, only a name or free-VRAM figure could differ.
    """
    count = len(uuids)
    if env_value is None:
        return list(range(count))
    entries = env_value.split(",")
    if env_value.startswith("MIG-"):
        return [0] if count else []
    visible = []
    if env_value.startswith("GPU-"):
        for entry in entries:
            if not entry.startswith("GPU-"):
                break
            matches = [i for i, uuid in enumerate(uuids) if uuid.startswith(entry)]
            if len(matches) != 1:  # unknown or ambiguous prefix
                break
            if matches[0] in visible:
                return []
            visible.append(matches[0])
        return visible
    for entry in entries:
        ordinal = _leading_int(entry.strip())
        if ordinal in visible:
            return []
        if ordinal < 0 or ordinal >= count:
            break
        visible.append(ordinal)
    return visible


def visible_gpus():
    """NVIDIA GPUs this process may use, honouring CUDA_VISIBLE_DEVICES.

    Never creates a CUDA context or allocates VRAM. Perception workers that
    must stay on the CPU set CUDA_VISIBLE_DEVICES=-1 and get [] here without
    NVML even being loaded. A driver whose NVML fails is reported once and
    treated as no GPU: failing closed to CPU with a visible reason is the rule
    (see get_torch_device).
    """
    global _nvml_error_reported
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_value is not None and not env_value.startswith(("GPU-", "MIG-")) \
            and _leading_int(env_value.split(",")[0].strip()) < 0:
        # Hidden outright ("-1", "", "none"): no need to ask the driver.
        return []
    gpus, error = _query_nvml()
    if error and error != "no_driver" and not _nvml_error_reported:
        _nvml_error_reported = True
        print(f"GPU probe: NVIDIA driver present but NVML failed ({error}); using cpu")
    ordinals = _visible_ordinals(env_value, [gpu["uuid"] for gpu in gpus])
    return [gpus[i] for i in ordinals]


# --- CUDA 12 / cuDNN 9 DLLs for onnxruntime -----------------------------------
#
# onnxruntime_providers_cuda.dll imports cublasLt64_12, cublas64_12, cufft64_11,
# cudart64_12 and cudnn64_9. Without them on the loader path the CUDA EP fails
# at session create ("cublasLt64_12.dll is missing", measured; see
# engines/caption/qwen_aligner.py). They used to arrive as a side effect of
# `import torch`, which loads every DLL in torch/lib. Each provider below names
# directories holding the set; the first one that has it wins.

_REQUIRED_CUDA_DLLS = ("cublasLt64_12.dll", "cudart64_12.dll", "cudnn64_9.dll")
# Every NVIDIA runtime DLL the engine's GPU code needs, and so exactly the set
# recall-engine.spec collects into the bundle's nvidia-cuda folder (it imports
# this tuple). Derived 2026-11-15 from PE import tables (pefile) plus the names
# the libraries load at run time:
#   onnxruntime_providers_cuda.dll -> cublasLt64_12, cublas64_12, cufft64_11,
#     cudart64_12, cudnn64_9;
#   ggml-cuda.dll (llama.cpp) -> cudart64_12, cublas64_12 (-> cublasLt64_12);
#   cudnn64_9 / cudnn_graph64_9 load the other six cuDNN sub-libraries by name;
#     the runtime-compiled engines and cufft64_11 load NVRTC, which loads its
#     builtins.
# Not needed, though torch/lib had them: cuSPARSE, cuSOLVER(Mg), cuRAND, CUPTI,
# nvJitLink, nvToolsExt, cuFFTW (no importer among the above), and zlibwapi:
# cuDNN 9 still names it, but NVIDIA's own cuDNN wheel does not ship it and a
# CUDA session over a conv model never loaded it (checked in the process's
# module list, 2026-11-15).
CUDA_RUNTIME_DLLS = (
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cufft64_11.dll",
    "cudnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "nvrtc64_120_0.dll",
    "nvrtc-builtins64_126.dll",
)
# The packaged engine's copy of CUDA_RUNTIME_DLLS: one folder under _internal,
# so ORT's preload_dlls(directory=...) finds every file in it and llama.cpp
# finds cudart/cuBLAS there through PATH (apps/api/pyi_rth_recall_cuda.py).
FROZEN_CUDA_DIRNAME = "nvidia-cuda"
# Pins the DLL source to one provider by name ("nvidia-wheels", "torch-lib",
# "frozen-bundle"), with no fallback to the others, so a test or a packaged
# smoke run can assert "these DLLs and no others". Unset or "auto": first
# complete provider in CUDA_DLL_PROVIDERS order.
CUDA_DLL_PROVIDER_ENV = "RECALL_CUDA_DLL_PROVIDER"
# DLLs torch's import used to load that ORT's preload_dlls() does not list:
# cuDNN's cnn sub-library and NVRTC, which the runtime-compiled cuDNN engines
# load on demand. Loading them here keeps the process's DLL set what it was.
# The ".alt" NVRTC variant is omitted from the bundle (recall-engine.spec) and
# is not matched.
_EXTRA_CUDA_DLL = re.compile(
    r"^(cudnn(_\w+)?64_9|nvrtc64_\d+_\d+|nvrtc-builtins64_\d+)\.dll$", re.IGNORECASE)


def _ort_site_packages():
    """The directory holding the onnxruntime package, found without importing it."""
    try:
        spec = importlib.util.find_spec("onnxruntime")
    except (ImportError, ValueError):
        return None
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    return os.path.dirname(locations[0]) if locations else None


def _nvidia_wheel_dirs(site_packages):
    """bin folders of NVIDIA's own nvidia-*-cu12 wheels, when all are installed.

    They are what ORT's preload_dlls() looks for next to onnxruntime
    (nvidia/<lib>/bin), pinned in scripts/requirements.txt. Installed, each
    wheel puts its DLLs in its own bin folder (cublas: cublas64_12,
    cublasLt64_12, nvblas64_12; cudnn: the eight cudnn*64_9; cuda_nvrtc: nvrtc
    and its builtins), so a process gets one directory per wheel.
    """
    if not site_packages:
        return []
    root = os.path.join(site_packages, "nvidia")
    required = {"cublas": "cublasLt64_12.dll", "cuda_runtime": "cudart64_12.dll",
                "cudnn": "cudnn64_9.dll"}
    dirs = []
    for package, marker in required.items():
        folder = os.path.join(root, package, "bin")
        if not os.path.isfile(os.path.join(folder, marker)):
            return []
        dirs.append(folder)
    for package in ("cufft", "cuda_nvrtc"):
        folder = os.path.join(root, package, "bin")
        if os.path.isdir(folder):
            dirs.append(folder)
    return dirs


def _torch_lib_dirs():
    """torch's lib folder (cu126 CUDA + cuDNN), located WITHOUT importing torch."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return []
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    return [os.path.join(locations[0], "lib")] if locations else []


def _frozen_cuda_dirs():
    """The packaged engine's nvidia-cuda folder under _MEIPASS (one folder).

    recall-engine.spec collects CUDA_RUNTIME_DLLS there from the nvidia wheels
    and does not bundle torch, so in a frozen build this is the only provider
    that can match: the wheel and torch lookups find no nvidia/ or torch/
    package under _MEIPASS.
    """
    from core.bundle_paths import get_resource_dir, is_frozen

    if not is_frozen():
        return []
    return [os.path.join(get_resource_dir(), FROZEN_CUDA_DIRNAME)]


CUDA_DLL_PROVIDERS = (
    ("nvidia-wheels", lambda: _nvidia_wheel_dirs(_ort_site_packages())),
    ("torch-lib", _torch_lib_dirs),
    ("frozen-bundle", _frozen_cuda_dirs),
)


def _forced_provider_name():
    """The provider CUDA_DLL_PROVIDER_ENV pins, or None for the default order."""
    value = os.environ.get(CUDA_DLL_PROVIDER_ENV, "").strip()
    return None if value.lower() in ("", "auto") else value


def _present_dlls(dirs):
    return {entry.casefold() for folder in dirs if os.path.isdir(folder)
            for entry in os.listdir(folder)}


def _locate(locate):
    """A provider's existing dirs, or None when its lookup itself failed."""
    try:
        return [d for d in locate() if os.path.isdir(d)]
    except Exception:  # noqa: BLE001 - a broken candidate is skipped
        return None


def find_cuda_dll_provider(providers=CUDA_DLL_PROVIDERS):
    """(name, dirs) of the first provider whose dirs hold the required DLLs.

    With CUDA_DLL_PROVIDER_ENV set, only the provider of that name is tried.
    """
    forced = _forced_provider_name()
    for name, locate in providers:
        if forced is not None and name != forced:
            continue
        dirs = _locate(locate)
        if not dirs:
            continue
        present = _present_dlls(dirs)
        if all(dll.casefold() in present for dll in _REQUIRED_CUDA_DLLS):
            return name, dirs
    return None


def _cuda_dll_report(provider, providers=None) -> str:
    """One log line naming where the CUDA DLLs come from, or why nothing matched.

    The packaged engine has no other trace of this: a bundle missing a DLL
    would otherwise surface only as "ORT CUDA EP unavailable" and a CPU scan.
    """
    forced = _forced_provider_name()
    pinned = f" ({CUDA_DLL_PROVIDER_ENV}={forced})" if forced else ""
    providers = CUDA_DLL_PROVIDERS if providers is None else providers
    if provider is not None:
        name, dirs = provider
        missing = [dll for dll in CUDA_RUNTIME_DLLS
                   if dll.casefold() not in _present_dlls(dirs)]
        line = f"CUDA DLLs: {name}{pinned} from {os.pathsep.join(dirs)}"
        if missing:
            line += f"; MISSING {', '.join(missing)} (GPU work that needs them will fail)"
        return line
    tried = []
    for name, locate in providers:
        if forced is not None and name != forced:
            continue
        dirs = _locate(locate)
        if dirs is None:
            tried.append(f"{name}: lookup failed")
        elif not dirs:
            tried.append(f"{name}: no folder")
        else:
            present = _present_dlls(dirs)
            missing = [dll for dll in _REQUIRED_CUDA_DLLS if dll.casefold() not in present]
            tried.append(f"{name}: {os.pathsep.join(dirs)} lacks {', '.join(missing)}")
    if forced is not None and not tried:
        tried.append(f"no provider is named {forced!r}")
    return (f"CUDA DLLs: no complete set found{pinned} [{'; '.join(tried)}]; "
            "onnxruntime falls back to the default DLL search")


def _register_dll_dirs(dirs) -> None:
    """Make ``dirs`` findable by every later DLL load in this process.

    Two mechanisms because the two GPU consumers load differently:
    os.add_dll_directory serves loads that use the LOAD_LIBRARY_SEARCH_*
    flags (ORT's provider), and PATH serves llama_cpp, which loads llama.dll
    with winmode=0, so ggml-cuda.dll resolves cudart64_12/cublas64_12 through
    the legacy search order (PATH, or a module already loaded by that name).
    `import torch` did both for torch/lib; this keeps that environment, and
    child processes (perception workers) inherit the PATH entry. Idempotent.
    """
    if os.name != "nt":
        return
    path = os.environ.get("PATH", "")
    on_path = {os.path.normcase(os.path.normpath(e)) for e in path.split(os.pathsep) if e}
    prepend = []
    for folder in dirs:
        key = os.path.normcase(os.path.normpath(folder))
        if key in _registered_dll_dirs:
            continue
        _registered_dll_dirs.add(key)
        _dll_directory_handles.append(os.add_dll_directory(folder))
        if key not in on_path:
            prepend.append(folder)
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + ([path] if path else []))


def prepare_llama_cuda() -> None:
    """Put the CUDA DLL folders on the search path before llama_cpp loads.

    Loads no DLL and needs no GPU, so it is safe in CPU-only processes. Call
    it before the first `import llama_cpp`: ggml-cuda.dll imports cudart64_12
    and cublas64_12 statically, so llama.dll cannot load without them even on
    the CPU path. Every engine import of llama_cpp calls this first; nothing is
    copied beside llama.dll any more. Copies an older setup_gpu_llama.py left in
    llama_cpp/lib, if still there, are found first.
    """
    try:
        provider = find_cuda_dll_provider()
        if provider is not None:
            _register_dll_dirs(provider[1])
    except Exception as exc:  # noqa: BLE001 - llama's own import error is clearer
        print(f"prepare_llama_cuda() failed: {exc!r}")


def _load_extra_cuda_dlls(dirs) -> None:
    import ctypes

    for folder in dirs:
        for entry in sorted(os.listdir(folder)):
            if _EXTRA_CUDA_DLL.match(entry):
                try:
                    ctypes.CDLL(os.path.join(folder, entry))
                except OSError:
                    pass  # same as before: ORT's own error names what is missing


def _ensure_ort_cuda_libs() -> None:
    """Load the CUDA 12 / cuDNN 9 DLLs for ORT's CUDA EP, without torch."""
    global _ort_cuda_libs_ready, _cuda_dll_provider
    if _ort_cuda_libs_ready:
        return
    _ort_cuda_libs_ready = True
    try:
        provider = find_cuda_dll_provider()
        _cuda_dll_provider = provider
        print(_cuda_dll_report(provider))
        if provider:
            # What `import torch` did first: put the folder on the process's
            # DLL search path for libraries loaded later by name.
            _register_dll_dirs(provider[1])
        import onnxruntime as ort
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            if provider is None and _forced_provider_name() is not None:
                # Pinned to a provider that is incomplete: no preload at all.
                # ORT's own search would take torch/lib whenever a +cu12 torch
                # is installed, which is exactly what the pin is there to rule
                # out; the probe then fails and says why.
                pass
            elif provider is None:
                # ORT's own search: torch/lib if a CUDA-12 torch is installed,
                # nvidia wheels next to it, then PATH.
                preload(cuda=True, cudnn=True, msvc=True)
            elif provider[0] == "nvidia-wheels":
                # "" = nvidia/<lib>/bin under ORT's site-packages, which is
                # exactly where _nvidia_wheel_dirs looked.
                preload(cuda=True, cudnn=True, msvc=True, directory="")
            else:
                preload(cuda=True, cudnn=True, msvc=True, directory=provider[1][0])
        if provider and os.name == "nt":
            _load_extra_cuda_dlls(provider[1])
    except Exception as exc:  # noqa: BLE001 - CPU fallback still works
        print(f"CUDA DLL preload failed: {exc!r}")


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
    """Return 'cuda' if this process can run CUDA work, else 'cpu'. Cached.

    The GPU gate for the whole engine: llama.cpp offload, OCR, the aligner,
    the judges and scan estimates all ask it. The name is kept for one release
    so call sites stay put; it no longer imports torch.

    'cuda' needs both: an NVIDIA GPU visible to this process (NVML plus
    CUDA_VISIBLE_DEVICES, no CUDA context) and onnxruntime's CUDA EP passing
    _probe_ort_cuda_ep. The probe is only reached when a GPU is visible, so a
    CPU-only worker or a machine without an NVIDIA driver never loads a CUDA
    DLL for this answer. torch.cuda.is_available() answered the same question
    with torch's copy of the same CUDA 12 runtime; the probe also needs cuDNN
    and cuFFT to load, which every GPU consumer here needs anyway.
    """
    global _torch_device
    if _torch_device is None:
        try:
            if not visible_gpus():
                _torch_device = "cpu"
            else:
                _torch_device = "cuda" if _probe_ort_cuda_ep() else "cpu"
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


def device_hint() -> str:
    """'cuda'/'cpu' for DISPLAY only; never loads CUDA or creates a context.

    The cached get_torch_device() answer when this process already made one,
    otherwise whether NVML sees a visible NVIDIA GPU. For labels such as an
    old job's device, read in the idle API process, where running the CUDA
    probe would hold a CUDA context (VRAM) for the rest of the session.
    """
    if _torch_device is not None:
        return _torch_device
    try:
        return "cuda" if visible_gpus() else "cpu"
    except Exception:  # noqa: BLE001 - a label must never raise
        return "cpu"


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
        if "CUDAExecutionProvider" in providers and not visible_gpus():
            # No visible NVIDIA GPU (CPU-only perception workers set
            # CUDA_VISIBLE_DEVICES=-1): the probe below would only fail, after
            # loading ~1.9 GB of CUDA/cuDNN DLL images for nothing.
            providers = [p for p in providers if p != "CUDAExecutionProvider"]
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
    prepare_llama_cuda()
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:  # noqa: BLE001 - optional dependency; absence is a state
        return None
