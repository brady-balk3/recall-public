# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Publish verified speech versions and select them only at backend startup."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from core.model_downloads import ModelDownloadCancelled, matches

SNAPSHOT_ENV = "RECALL_SPEECH_SNAPSHOT"


def artifact_paths(groups):
    return {f"{directory}/{name}": spec
            for _, _, _, files, directory in groups for name, spec in files.items()}


def _safe_file(root: Path, relative: str) -> Path:
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise ValueError("Missing or redirected speech artifact")
    return path


def publish_speech(root: Path, stage: Path, plan_id: str, groups, cancel_check, progress) -> None:
    """Caller owns the setup filesystem lock; existing versions are never edited."""
    expected = artifact_paths(groups)
    actual = set()
    def unreadable(error):
        raise error
    for directory, subdirs, files in os.walk(stage, onerror=unreadable):
        for name in subdirs:
            path = Path(directory) / name
            if path.resolve() != path:
                raise ValueError("Redirected speech directory")
        for name in files:
            path = Path(directory) / name
            if re.fullmatch(r"\.recall-model-[a-z0-9_]{8}", name):
                # Python mkstemp names left by an interrupted verified copy.
                # Only unlink a regular file in the owned stage, never a link.
                if path.resolve() != path or not path.is_file():
                    raise ValueError("Redirected setup temporary")
                path.unlink()
                continue
            actual.add(path.relative_to(stage).as_posix())
    if actual != set(expected):
        raise ValueError("Speech staging inventory changed")
    for relative, spec in expected.items():
        progress({"phase": "checking", "file": relative, "bytes_completed": 0,
                  "bytes_total": spec[0], "cancellation_deferred": False})
        if not matches(_safe_file(stage, relative), spec, cancel_check=cancel_check):
            raise ValueError("Speech artifact verification failed")
    if cancel_check():
        raise ModelDownloadCancelled()
    installed = root / "installed"
    if installed.resolve() != installed:
        raise ValueError("Redirected speech version directory")
    installed.mkdir(parents=True, exist_ok=True)
    name = f"{plan_id}-{uuid.uuid4().hex}"
    progress({"phase": "publishing", "file": None, "bytes_completed": 0,
              "bytes_total": 0, "cancellation_deferred": True})
    if cancel_check():
        raise ModelDownloadCancelled()
    # Rename on the same filesystem. Never repair/overwrite an activated version.
    os.rename(stage, installed / name)
    handle, temporary = tempfile.mkstemp(prefix=".speech-selection-", dir=root)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"version": 1, "plan_id": plan_id, "directory": name}, stream)
        os.replace(temporary, root / "speech-selection.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    # If selection publication fails, the old selection survives; the new
    # immutable directory is an orphan that can be reclaimed by future cleanup.


def initialize_speech_models(root: Path, plan_id: str, groups) -> str:
    """Call once in the API parent, never in spawned inference workers.

    Workers inherit the parent's selected path. Persisting a new selection does
    not change that environment. Startup checks sizes/paths; full hashes were
    checked before publication, avoiding a multi-GB hash pass on every launch.
    """
    os.environ.pop(SNAPSHOT_ENV, None)
    if os.environ.get("RECALL_MODELS_DIR", "").strip():
        return "configured_root"
    selection = root / "speech-selection.json"
    try:
        if not selection.exists():
            return "bundled"
        if selection.stat().st_size > 4096:
            return "invalid_selection"
        value = json.loads(selection.read_text(encoding="utf-8"))
        name = value["directory"]
        if value["version"] != 1 or value["plan_id"] != plan_id or not isinstance(name, str):
            return "invalid_selection"
        if not re.fullmatch(re.escape(plan_id) + r"-[0-9a-f]{32}", name):
            return "invalid_selection"
        version = root / "installed" / name
        if version.resolve() != version:
            return "invalid_selection"
        for relative, spec in artifact_paths(groups).items():
            if _safe_file(version, relative).stat().st_size != spec[0]:
                return "invalid_selection"
        os.environ[SNAPSHOT_ENV] = str(version)
        return "managed"
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return "invalid_selection"
