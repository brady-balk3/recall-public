# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Diagnostic: does this engine run its GPU code, and on which CUDA DLLs?

Not part of a scan and never started by the app. The packaged engine runs it
when launched as

    recall-engine.exe --recall-gpu-smoke <report.json>

(dispatched by apps/api/pyi_rth_recall_gpu_smoke.py before main.py, so no
database, job manager or server is built), and a dev venv can run it as

    venv\\Scripts\\python.exe apps\\api\\engine_gpu_smoke.py <report.json>

It loads no ASR/VLM/judge weights. In order, the order a scan can hit them:

1. ``import llama_cpp`` FIRST, before anything has loaded a CUDA DLL, and ask
   llama_supports_gpu_offload(): the case the bundle's dropped llama_cpp/lib
   copies depend on (pyi_rth_recall_cuda.py);
2. core.device.device_summary(), the line every scan logs;
3. one onnxruntime session over the bundled PP-OCRv6 detector (9.9 MB, conv
   layers, so cuDNN runs) on the providers get_ort_providers() picked;
4. OCR start-up through engines.ocr.ocr.warm_ocr() (paddlex + modelscope, with
   the torch guard) and one recognition on a rendered text image;
5. the process's loaded CUDA-family DLLs (EnumProcessModules) and any torch
   module in sys.modules.

The report is JSON because the windowed engine has no stdout. Exit code 0 when
every check holds: no torch module loaded, every loaded CUDA-family DLL from
one folder (the bundle's nvidia-cuda, or the provider core.device chose in
dev), and, if core.device answered "cuda", the CUDA EP active in the session.
"""

import json
import os
import sys
import time
import traceback

# cu*, cudnn*, nvrtc*, plus torch's own and the helpers torch/lib carried.
_CUDA_FAMILY = ("cu", "nvrtc", "nvjitlink", "nvtoolsext", "nvblas", "zlibwapi",
                "torch", "c10", "caffe2")


def _loaded_modules():
    """Full paths of every module in this process (psapi; psutil not required)."""
    import ctypes
    from ctypes import wintypes

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    process = kernel32.GetCurrentProcess()
    capacity = 1024
    while True:
        handles = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(process, handles, ctypes.sizeof(handles),
                                        ctypes.byref(needed)):
            raise ctypes.WinError(ctypes.get_last_error())
        count = needed.value // ctypes.sizeof(wintypes.HMODULE)
        if count <= capacity:
            break
        capacity = count
    name = ctypes.create_unicode_buffer(32768)
    paths = []
    for index in range(count):
        if psapi.GetModuleFileNameExW(process, handles[index], name, 32768):
            paths.append(name.value)
    return paths


def _cuda_modules():
    return sorted(path for path in _loaded_modules()
                  if os.path.basename(path).lower().startswith(_CUDA_FAMILY))


def _step(report, name, function):
    started = time.perf_counter()
    try:
        report[name] = {"ok": True, "result": function()}
    except Exception as exc:  # noqa: BLE001 - the report is the point
        report[name] = {"ok": False, "error": repr(exc),
                        "traceback": traceback.format_exc(limit=6)}
    report[name]["seconds"] = round(time.perf_counter() - started, 2)


def run():
    from core import device
    from core.bundle_paths import get_models_dir, is_frozen

    report = {
        "frozen": is_frozen(),
        "executable": sys.executable,
        "meipass": getattr(sys, "_MEIPASS", None),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        device.CUDA_DLL_PROVIDER_ENV: os.environ.get(device.CUDA_DLL_PROVIDER_ENV),
        "path_head": os.environ.get("PATH", "").split(os.pathsep)[:3],
    }

    def llama_first():
        if not report["frozen"]:
            # A dev venv has no runtime hook: the engine's own call puts the
            # DLL folders on the path. The frozen engine must manage without.
            device.prepare_llama_cuda()
        import llama_cpp

        return {"llama_supports_gpu_offload": bool(llama_cpp.llama_supports_gpu_offload()),
                "cuda_modules_after_llama_import": _cuda_modules()}

    _step(report, "llama_first", llama_first)
    _step(report, "device_summary", device.device_summary)
    report["cuda_dll_provider"] = device._cuda_dll_provider

    def ort_session():
        import numpy as np
        import onnxruntime as ort

        model = os.path.join(get_models_dir(), "ocr", "PP-OCRv6_small_det_onnx",
                             "inference.onnx")
        providers = device.get_ort_providers()
        session = ort.InferenceSession(model, providers=providers)
        feed = {session.get_inputs()[0].name:
                np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)}
        outputs = session.run(None, feed)
        return {"requested": providers, "active": session.get_providers(),
                "output_shapes": [list(o.shape) for o in outputs]}

    _step(report, "ort_session", ort_session)

    def ocr():
        import cv2
        import numpy as np
        from engines.ocr import ocr as ocr_engine

        available = ocr_engine.warm_ocr()
        image = np.full((160, 900, 3), 255, dtype=np.uint8)
        cv2.putText(image, "VICTORY ROYALE 42", (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 0), 4, cv2.LINE_AA)
        signal = ocr_engine.run_ocr(image, 0.0)
        return {"warm_ocr": available, "device": ocr_engine._ocr_device(), "text": signal.text,
                "confidence": round(float(signal.confidence), 3),
                "metadata": signal.metadata}

    _step(report, "ocr", ocr)

    def face_scorer():
        # The scorer a scan uses (RECALL_FACE_SCORER). It disables itself
        # quietly when it cannot load, and a scan then has no face channel.
        from engines.emotion import face_emotion

        ready = face_emotion._scorer_ready()
        reason = face_emotion._DISABLED_REASON
        if face_emotion._SCORER != "hsemotion":
            from engines.emotion import blendshape_scorer
            reason = blendshape_scorer._DISABLED_REASON
        return {"scorer": face_emotion.scorer_cache_label(), "ready": bool(ready), "disabled_reason": reason}

    _step(report, "face_scorer", face_scorer)

    modules = _cuda_modules()
    report["cuda_modules"] = modules
    report["torch_modules"] = sorted(
        m for m in sys.modules if m.split(".")[0] in ("torch", "torchvision", "torchaudio")
        and sys.modules[m] is not None)

    provider = device._cuda_dll_provider or device.find_cuda_dll_provider()
    allowed = [os.path.normcase(os.path.normpath(d)) + os.sep for d in (provider or (None, []))[1]]
    stray = [m for m in modules
             if not any(os.path.normcase(m).startswith(prefix) for prefix in allowed)]
    gpu = device.get_torch_device() == "cuda"
    session = report["ort_session"].get("result") or {}
    checks = {
        "no_torch_module": not report["torch_modules"],
        "cuda_dlls_only_from_provider": not stray,
        "frozen_provider_is_bundle_folder": (not report["frozen"]) or (
            provider is not None and provider[0] == "frozen-bundle"),
        "cuda_ep_active_when_gpu": (not gpu) or (
            "CUDAExecutionProvider" in session.get("active", [])),
        "every_step_ran": all(report[name]["ok"]
                              for name in ("llama_first", "device_summary", "ort_session", "ocr",
                                           "face_scorer")),
        "ocr_read_text": "VICTORY" in ((report["ocr"].get("result") or {}).get("text") or ""),
        "face_scorer_ready": bool((report["face_scorer"].get("result") or {}).get("ready")),
    }
    report["stray_cuda_modules"] = stray
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--recall-gpu-smoke" in argv:
        argv = argv[argv.index("--recall-gpu-smoke") + 1:]
    out_path = argv[0] if argv else None
    try:
        report = run()
    except Exception:  # noqa: BLE001
        report = {"passed": False, "error": traceback.format_exc()}
    text = json.dumps(report, indent=2, default=str)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    if sys.stdout is not None:
        print(text)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    raise SystemExit(main())
