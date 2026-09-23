# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Single-owner speech-model preparation, isolated from active inference files."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

from filelock import FileLock, Timeout

from core.bundle_paths import get_models_dir
from core.model_catalog import ASR_REPO, ASR_REVISION, ASR_FILES, ALIGNER_REPO, ALIGNER_REVISION, ALIGNER_FILES
from core.model_catalog import VAD_REPO, VAD_REVISION, VAD_FILES, SPEAKER_REPO, SPEAKER_REVISION, SPEAKER_FILES
from core.model_downloads import ModelDownloadCancelled, install_files
from core.model_activation import publish_speech


def speech_groups() -> tuple:
    return (
        ("asr", ASR_REPO, ASR_REVISION, dict(ASR_FILES), "asr/qwen3-asr-1.7b"),
        ("aligner", ALIGNER_REPO, ALIGNER_REVISION,
         {name: spec for name, spec in ALIGNER_FILES.items() if name != "onnx/model_q4.onnx"},
         "asr/qwen3-aligner-0.6b"),
        ("vad", VAD_REPO, VAD_REVISION, dict(VAD_FILES), "asr/silero-vad"),
        ("speaker", SPEAKER_REPO, SPEAKER_REVISION, dict(SPEAKER_FILES), "speaker"),
    )


def speech_plan(groups: tuple | None = None) -> dict:
    """A no-network, no-disk-scan quote, not an installed-model readiness check."""
    groups = speech_groups() if groups is None else groups
    # Bind consent and staging identity to the entire selected artifact recipe.
    fingerprint = hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()
    selected_bytes = sum(size for _, _, _, files, _ in groups for size, _ in files.values())
    return {
        "plan_id": fingerprint,
        "component": "speech",
        "activation_supported": True,
        "estimated_download_bytes": selected_bytes,
        "installed_bytes": selected_bytes,
        "estimated_additional_disk_bytes": 2 * selected_bytes + 2 * 128 * 1024 * 1024,
        "disk_estimate_includes_cache": True,
        "estimate_notes": ["cache_may_reduce_transfer", "retries_may_increase_transfer",
                           "transport_cache_overhead_not_included"],
        "groups": [
            {"id": group, "label": {"asr": "Speech recognition", "aligner": "Caption timing",
                                    "vad": "Speech detection", "speaker": "Speaker attribution"}[group],
             "source": repo, "revision": revision, "file_count": len(files),
             "artifact_bytes": sum(size for size, _ in files.values())}
            for group, repo, revision, files, _ in groups
        ],
    }


class ModelSetupConflict(Exception):
    pass


class ModelSetupController:
    """Prepare speech assets and select verified versions for the next restart.

    State is process-local. After restart, starting again verifies/reuses staged
    files. A filesystem lock excludes other processes writing the same stage.
    """

    def __init__(self, staging_root: str, runtime_source: str = "uninitialized"):
        self._groups = speech_groups()
        self._plan = speech_plan(self._groups)
        fingerprint = self._plan["plan_id"][:16]
        self._root = Path(staging_root).resolve()
        self._target = self._root / f"speech-{fingerprint}"
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._state = {"status": "idle", "operation_id": None, "component": "speech",
                       "plan_id": self._plan["plan_id"],
                       "runtime_source": runtime_source, "restart_required": False,
                       "activation_required": True, "progress": None, "error_code": None}

    def status(self) -> dict:
        with self._lock:
            result = dict(self._state)
            if result["progress"] is not None:
                result["progress"] = dict(result["progress"])
            return result

    def plan(self) -> dict:
        # Do not return mutable nested structures owned by this controller.
        result = json.loads(json.dumps(self._plan))
        result["activation_supported"] = not bool(os.environ.get("RECALL_MODELS_DIR", "").strip())
        return result

    def start(self, plan_id: str) -> dict:
        with self._lock:
            if plan_id != self._plan["plan_id"]:
                raise ModelSetupConflict("The download plan changed; review it before starting setup")
            if self._state["status"] in ("running", "activating", "cancelling"):
                return self.status()
            self._validate_staging()
            self._cancel = threading.Event()
            self._state.update(status="running", operation_id=uuid.uuid4().hex,
                               progress=None, error_code=None, activation_required=True)
            worker = threading.Thread(target=self._run, name="recall-model-setup", daemon=True)
            try:
                worker.start()
            except Exception:
                self._state.update(status="failed", error_code="worker_start_failed")
                raise
            return self.status()

    def cancel(self, operation_id: str) -> dict:
        with self._lock:
            if operation_id != self._state["operation_id"]:
                raise ModelSetupConflict("That setup operation is no longer current")
            if self._state["status"] in ("running", "activating", "cancelling"):
                self._cancel.set()
                self._state["status"] = "cancelling"
            return self.status()

    def activate(self, operation_id: str) -> dict:
        with self._lock:
            if os.environ.get("RECALL_MODELS_DIR", "").strip():
                raise ModelSetupConflict("An explicit model directory is configured; managed activation is disabled")
            if operation_id != self._state["operation_id"]:
                raise ModelSetupConflict("That setup operation is no longer current")
            if self._state["status"] in ("activating", "restart_required"):
                return self.status()
            if self._state["status"] != "completed":
                raise ModelSetupConflict("Finish downloading before activating speech models")
            self._cancel = threading.Event()
            self._state.update(status="activating", error_code=None)
            worker = threading.Thread(target=self._activate, name="recall-model-activation", daemon=True)
            try:
                worker.start()
            except Exception:
                self._state.update(status="completed", error_code="worker_start_failed")
                raise
            return self.status()

    def _activate(self) -> None:
        outcome, error_code = "restart_required", None
        try:
            with FileLock(str(self._root / ".setup.lock"), timeout=0):
                self._validate_staging()
                publish_speech(self._root, self._target, self._plan["plan_id"], self._groups,
                               self._cancel.is_set, lambda event: self._progress("speech", event))
        except ModelDownloadCancelled:
            outcome = "cancelled"
        except Timeout:
            outcome, error_code = "failed", "setup_busy"
        except OSError as error:
            outcome = "failed"
            error_code = "disk_full" if error.errno == errno.ENOSPC else "activation_failed"
        except Exception:
            outcome, error_code = "failed", "activation_failed"
        with self._lock:
            # Publication commits after its final cancellation check. A later
            # cancel request must not misreport a committed selection as cancelled.
            self._state.update(status=outcome, error_code=error_code)
            if outcome == "restart_required":
                self._state.update(activation_required=False, restart_required=True)

    def _progress(self, group: str, event: dict) -> None:
        with self._lock:
            self._state["progress"] = {**event, "group": group}

    def _validate_staging(self, destination: Path | None = None) -> None:
        active = Path(get_models_dir()).resolve()
        for path in (self._root, self._target, destination or self._target):
            resolved = path.resolve()
            if resolved != path or not resolved.is_relative_to(self._root):
                raise ModelSetupConflict("Setup staging must not redirect through links or junctions")
            if resolved.is_relative_to(active) or active.is_relative_to(resolved):
                raise ModelSetupConflict("Setup staging must be separate from active models")

    def _run(self) -> None:
        outcome, error_code = "completed", None
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(self._root / ".setup.lock"), timeout=0):
                self._validate_staging()
                for group, repo, revision, files, directory in self._groups:
                    destination = self._target / directory
                    self._validate_staging(destination)
                    install_files(repo, revision, files, str(destination),
                                  cancel_check=self._cancel.is_set,
                                  progress=lambda event, group=group: self._progress(group, event))
        except ModelDownloadCancelled:
            outcome = "cancelled"
        except Timeout:
            outcome, error_code = "failed", "setup_busy"
        except OSError as error:
            outcome = "failed"
            error_code = "disk_full" if error.errno == errno.ENOSPC else "storage_or_download_failed"
        except Exception:
            # Do not expose cache paths, URLs, tokens, or raw library errors.
            outcome, error_code = "failed", "model_setup_failed"
        with self._lock:
            if outcome == "completed" and self._cancel.is_set():
                outcome = "cancelled"
            self._state.update(status=outcome, error_code=error_code)
