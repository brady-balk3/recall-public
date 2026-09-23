# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Structured, fail-safe hardware routing report for first-run setup.

The scan pipeline remains the authority for live resource decisions. This
module reports the same device/provider state in a creator-facing shape and
never turns a failed accelerator probe into an application startup failure.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from core.device import (
    get_ort_providers,
    get_torch_device,
    llama_gpu_offload,
    reset_device_cache,
)
from core.process_governor import (
    available_memory_gb,
    available_vram_gb,
    gpu_perception_workers,
    physical_cpu_count,
    profile_workers,
)


CAPABILITY_SCHEMA_VERSION = 1
_CAPABILITIES_CACHE: Optional[Dict[str, Any]] = None


def _round_gb(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(max(0.0, float(value)), 1)


def _gpu_vendor(name: str, pnp_device_id: str = "") -> str:
    haystack = f"{name} {pnp_device_id}".lower()
    if "nvidia" in haystack or "ven_10de" in haystack:
        return "nvidia"
    if "amd" in haystack or "radeon" in haystack or "ven_1002" in haystack:
        return "amd"
    if "intel" in haystack or "ven_8086" in haystack:
        return "intel"
    return "unknown"


def _windows_gpu_adapters() -> List[Dict[str, str]]:
    """Return display adapters without assuming CUDA is available."""
    if os.name != "nt":
        return []
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if not executable:
        return []
    script = (
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "@(Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,PNPDeviceID) | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        payload = json.loads(result.stdout)
        rows = payload if isinstance(payload, list) else [payload]
    except Exception:
        return []

    adapters: List[Dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "").strip()[:200]
        if not name:
            continue
        pnp_device_id = str(row.get("PNPDeviceID") or "").strip()[:300]
        adapters.append(
            {
                "name": name,
                "vendor": _gpu_vendor(name, pnp_device_id),
                "driver_version": str(row.get("DriverVersion") or "").strip()[:100],
            }
        )
    return adapters


def _torch_cuda_name() -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0) or "").strip() or None
    except Exception:
        pass
    return None


def _detect_gpu_adapters(torch_device: str) -> List[Dict[str, str]]:
    adapters = _windows_gpu_adapters()
    if torch_device != "cuda":
        return adapters

    cuda_name = _torch_cuda_name()
    if not cuda_name:
        return adapters
    if any(adapter["vendor"] == "nvidia" and adapter["name"] == cuda_name for adapter in adapters):
        return adapters
    return [
        {"name": cuda_name, "vendor": "nvidia", "driver_version": ""},
        *adapters,
    ]


def _primary_adapter(
    adapters: List[Dict[str, str]], torch_device: str
) -> Optional[Dict[str, str]]:
    if not adapters:
        return None
    preferred = ("nvidia", "amd", "intel", "unknown")
    if torch_device == "cuda":
        preferred = ("nvidia", "amd", "intel", "unknown")
    for vendor in preferred:
        match = next((adapter for adapter in adapters if adapter["vendor"] == vendor), None)
        if match:
            return match
    return adapters[0]


def _warning(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


def get_hardware_capabilities(refresh: bool = False) -> Dict[str, Any]:
    """Return the safe route and the probes that justify it.

    ``refresh`` is intended for the Settings button and must only be allowed by
    the API while the job manager is idle.
    """
    global _CAPABILITIES_CACHE
    if _CAPABILITIES_CACHE is not None and not refresh:
        return deepcopy(_CAPABILITIES_CACHE)
    if refresh:
        reset_device_cache()
        _CAPABILITIES_CACHE = None

    try:
        torch_device = get_torch_device()
    except Exception:
        torch_device = "cpu"
    try:
        ort_providers = list(get_ort_providers())
    except Exception:
        ort_providers = ["CPUExecutionProvider"]
    try:
        llama_gpu = llama_gpu_offload()
    except Exception:
        llama_gpu = None

    logical_cores = max(1, os.cpu_count() or 1)
    try:
        physical_cores = max(1, physical_cpu_count())
    except Exception:
        physical_cores = logical_cores
    try:
        available_ram = available_memory_gb()
    except Exception:
        available_ram = None
    try:
        cpu_workers = profile_workers(
            "balanced",
            cpu_count=physical_cores,
            available_gb=available_ram,
        )
    except Exception:
        cpu_workers = max(1, min(8, physical_cores - 1))

    adapters = _detect_gpu_adapters(torch_device)
    primary_gpu = _primary_adapter(adapters, torch_device)
    cuda_ready = torch_device == "cuda"
    ort_cuda_ready = "CUDAExecutionProvider" in ort_providers

    free_vram = None
    gpu_workers = 0
    if cuda_ready:
        try:
            free_vram = available_vram_gb()
            gpu_workers = gpu_perception_workers(
                cpu_workers,
                available_vram=free_vram,
            )
        except Exception:
            free_vram = None
            gpu_workers = 0

    route = "nvidia_cuda" if cuda_ready else "cpu"
    reason = "accelerated"
    warnings: List[Dict[str, str]] = []
    vendor = primary_gpu["vendor"] if primary_gpu else None

    if not cuda_ready:
        if vendor == "nvidia":
            reason = "nvidia_cuda_unavailable"
            warnings.append(
                _warning(
                    "nvidia_cuda_unavailable",
                    "An NVIDIA GPU was found, but Recall could not initialize CUDA. CPU mode is active.",
                )
            )
        elif vendor in {"amd", "intel"}:
            reason = "unsupported_gpu"
            warnings.append(
                _warning(
                    "unsupported_gpu",
                    f"{vendor.upper()} GPU acceleration is not supported in this alpha. CPU mode is active.",
                )
            )
        else:
            reason = "no_supported_gpu"
            warnings.append(
                _warning(
                    "no_supported_gpu",
                    "No supported NVIDIA accelerator was found. CPU mode is active.",
                )
            )
    elif not ort_cuda_ready:
        warnings.append(
            _warning(
                "onnx_cuda_unavailable",
                "CUDA is available, but OCR acceleration could not initialize and will use CPU fallback.",
            )
        )
    if cuda_ready and gpu_workers == 0:
        warnings.append(
            _warning(
                "gpu_perception_cpu_fallback",
                "GPU perception workers could not be safely allocated, so that stage will use CPU fallback.",
            )
        )
    if cpu_workers <= 1:
        warnings.append(
            _warning(
                "limited_resources",
                "Recall will use one analysis worker to avoid exhausting available memory.",
            )
        )

    report: Dict[str, Any] = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "reason": reason,
        "cpu": {
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
        },
        "memory": {
            "available_gb": _round_gb(available_ram),
        },
        "gpu": deepcopy(primary_gpu),
        "adapters": deepcopy(adapters),
        "acceleration": {
            "cuda_ready": cuda_ready,
            "torch_device": torch_device,
            "onnx_providers": ort_providers,
            "onnx_cuda_ready": ort_cuda_ready,
            "llama_gpu": llama_gpu,
            "free_vram_gb": _round_gb(free_vram),
            "gpu_perception_ready": gpu_workers > 0,
        },
        "recommended": {
            "performance_profile": "balanced",
            "worker_mode": "gpu" if gpu_workers > 0 else "cpu",
            "cpu_workers": cpu_workers,
            "gpu_workers": gpu_workers,
        },
        "warnings": warnings,
    }
    _CAPABILITIES_CACHE = deepcopy(report)
    return report
