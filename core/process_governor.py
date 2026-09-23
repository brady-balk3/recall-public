# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Process governor (resource plan, package 1): keep heavy scans polite.

A VOD scan pegs every core for an hour or more. At normal priority the
perception workers compete head-to-head with the user's browser/game/OS and
the whole machine feels unusable — even though the scan doesn't need any
cycle *right now*, it just needs them eventually. Windows priority classes
solve exactly this: BELOW_NORMAL (+ EcoQoS on hybrid CPUs) background work
still consumes every idle cycle at full speed but yields instantly to
foreground apps. This is the HandBrake / backup-tool / search-indexer model.

Deliberately NOT used: PROCESS_MODE_BACKGROUND_BEGIN. It would also lower
I/O priority (nice) but aggressively trims the working set — workers hold
1-3 GB of OCR/emotion models, and trimming them causes page-fault churn
that's worse than the I/O contention it fixes.

Every call here is best-effort: failing to set priority must never break a
scan, so errors are swallowed and a bool reports what happened.
"""

import os
import subprocess
import time


LEGACY_ENV_FALLBACKS = {
    "RECALL_WORKER_MEM_GB": "GIE_WORKER_MEM_GB",
    "RECALL_MEM_RESERVE_GB": "GIE_MEM_RESERVE_GB",
    "RECALL_WORKER_VRAM_GB": "GIE_WORKER_VRAM_GB",
    "RECALL_VRAM_RESERVE_GB": "GIE_VRAM_RESERVE_GB",
    "RECALL_WORKER_STALL_WARN_SEC": "GIE_WORKER_STALL_WARN_SEC",
    "RECALL_WORKER_STALL_FAIL_SEC": "GIE_WORKER_STALL_FAIL_SEC",
}


def get_recall_env(name: str, default=None):
    """Read a RECALL_* setting, falling back to its legacy GIE_* alias.

    Presence, rather than truthiness, decides precedence so an explicitly set
    primary value behaves exactly like a directly read environment variable.
    """
    if name in os.environ:
        return os.environ[name]
    legacy_name = LEGACY_ENV_FALLBACKS.get(name)
    if legacy_name and legacy_name in os.environ:
        return os.environ[legacy_name]
    return default


# Memory per perception worker, MEASURED 2026-07-12 (Windows, 1080p frame
# through vision+OCR after full model load): ~1.1 GB working set / ~1.4 GB
# private commit. 1.5 adds margin for long-run growth. RESERVE_GB is RAM kept
# free for the OS + foreground apps. Env-overridable.
PER_WORKER_GB = float(get_recall_env("RECALL_WORKER_MEM_GB", "1.5"))
RESERVE_GB = float(get_recall_env("RECALL_MEM_RESERVE_GB", "4.0"))

# VRAM per GPU-OCR perception worker, MEASURED 2026-07-25 (RTX 4070 Ti SUPER,
# 6 workers: 2164 -> 8222 MiB used) = ~1.0 GB each, which is mostly CUDA
# context + the onnxruntime arena rather than the models (PP-OCRv6 small and
# yolo26n are tiny). 1.2 adds margin. VRAM_RESERVE_GB keeps the desktop
# compositor and the user's own apps off the edge; the local VLM needs ~3 GB
# but runs in a LATER stage, never concurrently with perception.
PER_WORKER_VRAM_GB = float(get_recall_env("RECALL_WORKER_VRAM_GB", "1.2"))
VRAM_RESERVE_GB = float(get_recall_env("RECALL_VRAM_RESERVE_GB", "2.0"))

# Below this many GPU workers, the CPU pool is simply faster. MEASURED
# 2026-07-25 at production settings: a GPU worker sustains ~2.2-3.0x realtime
# (346 ms/frame under 6-way contention) vs ~0.97x for a CPU worker (799 ms),
# so 2 GPU workers (~5-6x) roughly ties the 6-worker CPU pool (5.80x) and 3+
# clearly beat it. Fewer than 2 and we would trade a working pool for a slower
# one, so a VRAM-starved machine keeps the full CPU pool instead.
MIN_GPU_WORKERS = 2

# Performance profiles (resource plan pkg 2). Every profile keeps
# BELOW_NORMAL priority — background work at full speed on an idle machine
# costs nothing — so profiles only trade worker count and RAM headroom:
#   background  streaming/gaming while Recall scans; few workers, big reserve
#   balanced    default; the measured sweet spot for "PC stays usable"
#   max         user walked away; saturate cores, minimal reserve
PROFILES = {
    "background": {"min_workers": 1, "max_workers": 4, "cpu_headroom": 0.75, "reserve_gb": 6.0},
    "balanced": {"min_workers": 2, "max_workers": 12, "cpu_headroom": 2, "reserve_gb": RESERVE_GB},
    "max": {"min_workers": 2, "max_workers": 16, "cpu_headroom": 1, "reserve_gb": 2.0},
}


def resolve_profile(name) -> dict:
    """Profile settings for a UI name; unknown/None falls back to balanced."""
    return PROFILES.get(str(name or "").strip().lower(), PROFILES["balanced"])


def physical_cpu_count() -> int:
    """Best-effort PHYSICAL core count (SMT/hyperthread siblings collapsed).

    Worker count MUST derive from this, not os.cpu_count() (logical threads).
    Each perception worker is pinned to one thread of CPU-bound OCR — exactly
    the SIMD-heavy workload that gets ~nothing from an SMT sibling (shared
    execution units + L1/L2). Spawning a worker per *logical* thread therefore
    oversubscribes every physical core: past the physical count you pay full
    memory (each worker holds ~1.5 GB of models) and scheduling cost for near-
    zero added throughput, and the cache/working-set contention thrashes. On an
    8-core / 16-thread box the knee is a hard 8.

    Windows: GetLogicalProcessorInformation, counting RelationProcessorCore
    entries — accurate on hybrid P/E CPUs where logical != 2 * physical, for
    the single processor group (<=64 logical CPUs) every target machine has.
    Linux: distinct (physical id, core id) pairs in /proc/cpuinfo. Any failure
    falls back to os.cpu_count() unchanged — never worse than today, and always
    a sane non-zero number.
    """
    logical = os.cpu_count() or 4

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            RelationProcessorCore = 0

            class _Union(ctypes.Union):
                _fields_ = [("Flags", ctypes.c_byte),
                            ("Reserved", ctypes.c_ulonglong * 2)]

            class _SLPI(ctypes.Structure):
                _fields_ = [("ProcessorMask", ctypes.c_void_p),   # ULONG_PTR
                            ("Relationship", ctypes.c_int),        # enum
                            ("u", _Union)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetLogicalProcessorInformation.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetLogicalProcessorInformation.restype = wintypes.BOOL

            length = wintypes.DWORD(0)
            # First call sizes the buffer (fails with ERROR_INSUFFICIENT_BUFFER).
            kernel32.GetLogicalProcessorInformation(None, ctypes.byref(length))
            count = length.value // ctypes.sizeof(_SLPI)
            if count > 0:
                buf = (_SLPI * count)()
                if kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(length)):
                    cores = sum(1 for e in buf
                                if e.Relationship == RelationProcessorCore)
                    if cores > 0:
                        return cores
        except Exception:
            pass
        return logical

    try:
        pairs = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except Exception:
        pass
    return logical


def profile_workers(profile_name=None, cpu_count=None, available_gb=None) -> int:
    """Worker count for a machine + profile: CPU-derived, then RAM-capped.

    ``cpu_count`` is a PHYSICAL core count (see physical_cpu_count) — never
    logical threads, or every profile oversubscribes SMT siblings and thrashes.
    ``cpu_headroom`` is the number of physical cores left free when >= 1
    (OS / GPU feeder / the main process running Whisper + the sole SSE/DB
    writer), or the *fraction* of cores to leave free when < 1 (the background
    profile scales its politeness with core count instead of pinning a
    constant).
    """
    profile = resolve_profile(profile_name)
    cpu = cpu_count or physical_cpu_count() or 4
    headroom = profile["cpu_headroom"]
    reserved_cpus = headroom if headroom >= 1 else cpu * headroom
    workers = int(cpu - reserved_cpus)
    workers = max(profile["min_workers"], min(profile["max_workers"], workers))
    return memory_capped_workers(
        workers, available_gb=available_gb, reserve_gb=profile["reserve_gb"]
    )


def enter_background_mode() -> bool:
    """Drop the CURRENT process to background CPU scheduling.

    Windows: BELOW_NORMAL priority class + EcoQoS power throttling (the
    scheduler prefers E-cores on hybrid CPUs and deprioritizes vs foreground).
    POSIX: os.nice(10). Returns True if at least the priority class stuck.
    """
    if os.name != "nt":
        try:
            os.nice(10)
            return True
        except Exception:
            return False

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declare 64-bit HANDLE types: the default c_int truncates the -1
        # pseudo-handle and SetPriorityClass fails with ERROR_INVALID_HANDLE.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        ok = bool(kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS))

        # EcoQoS (Win10 1709+). Older Windows lacks SetProcessInformation's
        # power-throttling class — failing is fine, priority class already stuck.
        try:
            class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
                _fields_ = [
                    ("Version", wintypes.ULONG),
                    ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG),
                ]

            PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
            ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS

            kernel32.SetProcessInformation.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ]
            kernel32.SetProcessInformation.restype = wintypes.BOOL

            state = PROCESS_POWER_THROTTLING_STATE(
                PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
                PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            )
            kernel32.SetProcessInformation(
                handle, ProcessPowerThrottling,
                ctypes.byref(state), ctypes.sizeof(state),
            )
        except Exception:
            pass

        return ok
    except Exception:
        return False


def subprocess_creationflags(background: bool = True) -> int:
    """creationflags for Popen/run so a child (ffmpeg) starts at low priority."""
    if background and os.name == "nt":
        return subprocess.BELOW_NORMAL_PRIORITY_CLASS
    return 0


def available_memory_gb():
    """Best-effort available physical RAM in GB, or None if unknowable."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys / (1024 ** 3)
        except Exception:
            pass
        return None

    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)  # kB -> GB
    except Exception:
        pass
    return None


# A whole scan's GPU worker pool is decided from ONE free-VRAM reading, so a
# transient dip decides the next 20+ minutes of throughput. Dips are normal at
# exactly the moment we sample: the previous job's worker processes are still
# unwinding their CUDA contexts, and the user's own apps allocate whenever they
# feel like it. Sample a few times and keep the MAX -- the high-water mark is
# the honest picture of what the card will actually have free for the scan,
# where any single sample can only ever be an underestimate of it. Three
# samples over ~0.8s is free against a multi-hour scan.
VRAM_SAMPLES = max(1, int(float(get_recall_env("RECALL_VRAM_SAMPLES", "3"))))
VRAM_SAMPLE_INTERVAL_SEC = max(
    0.0, float(get_recall_env("RECALL_VRAM_SAMPLE_INTERVAL_SEC", "0.4"))
)


def stable_available_vram_gb(samples: int | None = None,
                             interval_sec: float | None = None):
    """FREE VRAM in GB from several samples (the max), or None if unknowable.

    Returns None only when EVERY sample failed -- a probe that works
    intermittently should still plan the scan rather than fall back to CPU.
    """
    samples = VRAM_SAMPLES if samples is None else max(1, int(samples))
    if interval_sec is None:
        interval_sec = VRAM_SAMPLE_INTERVAL_SEC
    best = None
    for i in range(samples):
        if i and interval_sec > 0:
            time.sleep(interval_sec)
        reading = available_vram_gb()
        if reading is not None and (best is None or reading > best):
            best = reading
    return best


def available_vram_gb():
    """Best-effort FREE VRAM in GB from ONE sample, or None if unknowable.

    Callers sizing a worker pool want ``stable_available_vram_gb`` instead;
    this is the single-shot probe underneath it.

    Deliberately prefers ``nvidia-smi`` over ``torch.cuda.mem_get_info``: the
    torch call initializes a CUDA context in the caller (~300 MB) purely to ask
    how much VRAM is free, which on a tight card eats the very headroom the
    answer is used to allocate. torch is only the fallback for machines where
    nvidia-smi isn't on PATH but CUDA works.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            # Multi-GPU: the workers all land on device 0, so read the first row.
            first = result.stdout.strip().splitlines()[0]
            return float(first.strip()) / 1024.0
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, _total = torch.cuda.mem_get_info()
            return free_bytes / (1024 ** 3)
    except Exception:
        pass
    return None


def gpu_perception_workers(cpu_workers: int, available_vram=None,
                           per_worker_gb: float = PER_WORKER_VRAM_GB,
                           reserve_gb: float = VRAM_RESERVE_GB) -> int:
    """How many perception workers may run with GPU OCR/vision; 0 means CPU.

    Returns 0 whenever GPU perception should not be used at all — no NVIDIA
    GPU, unknowable VRAM, or so little free VRAM that the surviving worker
    count would be slower than the full CPU pool (see MIN_GPU_WORKERS). The
    caller keeps its CPU worker count in that case, so this can never *reduce*
    throughput; it only ever swaps a slower pool for a faster one.
    """
    cpu_workers = max(1, int(cpu_workers))
    if available_vram is None:
        available_vram = stable_available_vram_gb()
    if available_vram is None:
        return 0
    fit = int((available_vram - reserve_gb) / max(0.1, per_worker_gb))
    workers = min(cpu_workers, fit)
    if workers < MIN_GPU_WORKERS:
        return 0
    return workers


def memory_capped_workers(workers: int, available_gb=None,
                          per_worker_gb: float = PER_WORKER_GB,
                          reserve_gb: float = RESERVE_GB) -> int:
    """Cap a CPU-derived worker count by available RAM.

    Priority classes fix CPU contention but nothing fixes running out of
    RAM: each worker privately loads its own OCR/emotion/torch stack, and
    overshooting available memory pages OTHER apps out (the machine stays
    slow even after the scan). Unknown availability leaves the count as-is.
    """
    workers = max(1, int(workers))
    if available_gb is None:
        available_gb = available_memory_gb()
    if available_gb is None:
        return workers
    fit = int((available_gb - reserve_gb) / max(0.1, per_worker_gb))
    return max(1, min(workers, fit))
