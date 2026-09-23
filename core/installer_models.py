# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Install the pinned large models from Hugging Face before launching Recall."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from core.model_catalog import ASR_REPO, ASR_REVISION, ASR_FILES
from core.model_catalog import ALIGNER_REPO, ALIGNER_REVISION, ALIGNER_FILES
from core.model_downloads import install_files, matches


def download_groups():
    # remote filename -> (installed relative filename, byte count, SHA256)
    def group(repo, revision, files, directory):
        return repo, revision, {
            name: (f"{directory}/{name}", size, digest)
            for name, (size, digest) in files.items()
        }
    return (
        group(ASR_REPO, ASR_REVISION, ASR_FILES, "asr/qwen3-asr-1.7b"),
        group(ALIGNER_REPO, ALIGNER_REVISION,
              {name: ALIGNER_FILES[name] for name in (
                  "config.json", "tokenizer.json", "onnx/model.onnx", "onnx/model.onnx_data")},
              "asr/qwen3-aligner-0.6b"),
        ("bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
         "ae44f08e1392f39c0e474af10c3ff8355c8b6688", {
             "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf": (
                 "llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497280736,
                 "2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e")}),
        group("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", {
            "Qwen3.5-4B-Q4_K_M.gguf": (2740937888, "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"),
            "mmproj-F16.gguf": (672423616, "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864"),
        }, "vlm/qwen3.5-4b"),
    )


def installed_download_paths() -> set[str]:
    return {spec[0] for _, _, files in download_groups() for spec in files.values()}


def install_models(root: Path, *, progress=None, groups=None) -> None:
    from filelock import FileLock

    root = root.absolute()
    if root.resolve() != root:
        raise ValueError("Model installation directory must not be redirected")
    root.mkdir(parents=True, exist_ok=True)
    # The installer owns this operation; the API is never started in this process.
    with FileLock(str(root.parent / "model-install.lock"), timeout=0):
        for repo, revision, files in (download_groups() if groups is None else groups):
            for remote, (relative, size, digest) in files.items():
                destination = root / relative
                if destination.resolve() != destination or not destination.is_relative_to(root):
                    raise ValueError("Redirected model destination")
                if matches(destination, (size, digest)):
                    if progress:
                        progress({"phase": "ready", "file": relative})
                    continue
                # The upstream text GGUF has a different filename. Keep download
                # publication atomic, then rename verified bytes to Recall's name.
                remote_parts = Path(remote).parts
                target = destination.parent
                if len(remote_parts) > 1:
                    target = destination.parents[len(remote_parts) - 1]
                install_files(repo, revision, {remote: (size, digest)}, str(target),
                              progress=progress)
                downloaded = target / remote
                if downloaded != destination:
                    os.replace(downloaded, destination)
                if not matches(destination, (size, digest)):
                    raise ValueError("Installed model verification failed")


def _installer_streams():
    # A windowless PyInstaller executable sets Python streams to None, even
    # when NSIS supplies output pipes. Reconnect those pipes for setup details.
    for name, handle_id in (("stdout", -11), ("stderr", -12)):
        if getattr(sys, name) is not None:
            continue
        try:
            import ctypes
            import msvcrt
            get_handle = ctypes.windll.kernel32.GetStdHandle
            get_handle.restype = ctypes.c_void_p
            handle = get_handle(handle_id)
            if not handle or handle == ctypes.c_void_p(-1).value:
                raise OSError("No installer output pipe")
            fd = msvcrt.open_osfhandle(handle, os.O_WRONLY)
            stream = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
        except (OSError, ImportError, AttributeError):
            stream = open(os.devnull, "w", encoding="utf-8")
        setattr(sys, name, stream)


def main() -> int:
    from core.bundle_paths import get_resource_dir

    _installer_streams()
    previous = None
    def report(event):
        nonlocal previous
        state = (event["phase"], event["file"])
        if state != previous:
            print(f"{state[0]}: {state[1]}", flush=True)
            previous = state
    try:
        install_models(Path(get_resource_dir()) / "models", progress=report)
        print("Recall models verified and ready.", flush=True)
        return 0
    except Exception as exc:
        print(f"Model setup did not finish: {exc}", file=sys.stderr, flush=True)
        return 1
