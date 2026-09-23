# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Application-callable pinned downloads; no subprocess or model loading.

Hugging Face owns resumable transport and cache locking. Only verified bytes
are copied into the runtime directory. Publication is atomic per file, not
across a model group; callers must finish setup before starting inference.
"""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import shutil
from typing import Callable


class ModelDownloadCancelled(Exception):
    """Setup was cancelled before publication of the current file."""


def _check_cancel(cancel_check):
    if cancel_check is not None and cancel_check():
        raise ModelDownloadCancelled("Model setup cancelled")


def _existing_parent(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def _check_space(target: Path, cache: Path, required: int) -> None:
    # Budget a cache copy plus an installed copy. Existing cache entries are not
    # discounted: this deliberately covers forced repair and leaves headroom.
    target_volume = _existing_parent(target)
    cache_volume = _existing_parent(cache)
    reserve = 128 * 1024 * 1024
    if target_volume.stat().st_dev == cache_volume.stat().st_dev:
        checks = [(target_volume, 2 * required + reserve)]
    else:
        checks = [(target_volume, required + reserve), (cache_volume, required + reserve)]
    for volume, needed in checks:
        if shutil.disk_usage(volume).free < needed:
            raise OSError(errno.ENOSPC, "Not enough free space for model download and installation")


def matches(path: Path, expected: tuple[int, str], *, cancel_check=None) -> bool:
    _check_cancel(cancel_check)
    size, digest = expected
    if not path.is_file() or path.stat().st_size != size:
        return False
    actual = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            _check_cancel(cancel_check)
            actual.update(block)
    return actual.hexdigest() == digest


def install_files(
    repo: str, revision: str, files: dict[str, tuple[int, str]], target: str, *,
    cancel_check: Callable[[], bool] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> None:
    """Install a trusted built-in catalog selection, repairing corrupt files.

    The upstream cache is separate from runtime files. It may retain an extra
    copy of downloaded weights; cache size/cleanup belongs to setup management.
    """
    _check_cancel(cancel_check)

    def report(phase, name, completed=0, total=0):
        if progress is not None:
            progress({"phase": phase, "file": name, "bytes_completed": completed,
                      "bytes_total": total, "cancellation_deferred": phase == "downloading"})

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An immutable model revision is required")
    root = Path(target).resolve()
    planned = []
    for name, expected in files.items():
        parts = name.split("/")
        if (
            not name or PurePosixPath(name).is_absolute() or "\\" in name
            or any(part in ("", ".", "..") or ":" in part for part in parts)
            or type(expected[0]) is not int or expected[0] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", expected[1])
        ):
            raise ValueError("Invalid model artifact")
        destination = root.joinpath(*parts)
        if not destination.resolve().is_relative_to(root):
            raise ValueError("Model destination escapes its root")
        planned.append((name, expected, destination))
    needed = []
    for name, expected, destination in planned:
        report("checking", name, total=expected[0])
        if matches(destination, expected, cancel_check=cancel_check):
            report("ready", name, expected[0], expected[0])
        else:
            needed.append((name, expected, destination))
    if not needed:
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_root = Path(HF_HUB_CACHE).resolve()
    _check_cancel(cancel_check)
    _check_space(root, cache_root, sum(expected[0] for _, expected, _ in needed))
    for name, expected, destination in needed:
        # HF's cache owns resumable transport. Verify while copying so a large
        # model does not need a separate full disk pass before publication.
        for force in (False, True):
            _check_cancel(cancel_check)
            _check_space(root, cache_root, expected[0])
            report("downloading", name, total=expected[0])
            # HF's blocking call cannot be interrupted here. A cancel request
            # remains pending until it returns, before any runtime publication.
            cached = Path(hf_hub_download(
                repo_id=repo, revision=revision, filename=name, force_download=force,
                token=False,
            ))
            _check_cancel(cancel_check)
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(prefix=".recall-model-", dir=destination.parent)
            try:
                digest = hashlib.sha256()
                size = 0
                with os.fdopen(handle, "wb") as output, cached.open("rb") as source:
                    report("installing", name, total=expected[0])
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        _check_cancel(cancel_check)
                        size += len(block)
                        if size > expected[0]:
                            break
                        digest.update(block)
                        output.write(block)
                        report("installing", name, size, expected[0])
                if size == expected[0] and digest.hexdigest() == expected[1]:
                    _check_cancel(cancel_check)
                    os.replace(temporary, destination)
                    report("ready", name, size, expected[0])
                    break
                if force:
                    raise ValueError(f"Model checksum mismatch: {name}")
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
