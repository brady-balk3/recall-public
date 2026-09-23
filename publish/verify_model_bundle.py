# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Verify the final frozen models tree against the selected reviewed inputs."""

import hashlib
import os
from pathlib import Path

from core.model_vault import unseal_bytes
from publish.build_public_tree import _hash, _relative_path
from publish.model_payload import ModelPayloadManifest, validate_model_destinations


def expected_model_payload(
    datas: list[tuple[str, str]],
    manifest: ModelPayloadManifest,
    sealed_sources: dict[str, str],
) -> dict[str, tuple[str, bool]]:
    """Freeze installed names and reviewed hashes before PyInstaller analysis.

    Sealed files are checked against their reviewed plaintext at the output,
    so changing a source between approval and sealing cannot approve new bytes.
    """
    validate_model_destinations(datas)
    expected = {}
    for source_name, destination in datas:
        destination = str(destination).replace("\\", "/").rstrip("/")
        if destination.casefold() != "models" and not destination.casefold().startswith("models/"):
            continue
        source = Path(source_name).absolute()
        original = Path(sealed_sources.get(str(source), str(source))).absolute()
        try:
            original_name = original.relative_to(manifest.root).as_posix()
        except ValueError as exc:
            raise ValueError("Final model input has no reviewed source") from exc
        if original_name not in manifest.files:
            raise ValueError("Final model input has no reviewed hash")
        target = _relative_path(f"{destination}/{source.name}").split("/", 1)[1]
        expected[target] = (manifest.files[original_name], str(source) in sealed_sources)
    if not expected:
        raise ValueError("Expected a nonempty reviewed model payload")
    return expected


def verify_model_bundle(
    models_root: str | Path, expected: dict[str, tuple[str, bool]],
) -> dict:
    """Reject missing, additional, redirected or changed final model files.

    This covers models/ only. It does not certify other bundled dependencies,
    source privacy, or licensing. Run before archiving a quiescent build tree.
    """
    root = Path(models_root).absolute()
    if not expected or root.resolve() != root or not root.is_dir():
        raise ValueError("Expected an unredirected model output directory")
    expected_dirs = {""}
    for name in expected:
        _relative_path(name)
        expected_dirs.update(parent.as_posix() for parent in Path(name).parents if parent != Path("."))
    actual = {}

    def walk_error(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        parent = Path(directory)
        for name in dirs:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if path.resolve() != path or relative not in expected_dirs:
                raise ValueError("Unexpected or redirected model output directory")
        for name in files:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if path.resolve() != path or not path.is_file() or relative not in expected:
                raise ValueError("Unexpected or redirected model output file")
            actual[relative] = path
    if set(actual) != set(expected):
        raise ValueError("Final model output is missing reviewed files")
    inventory = []
    for name, path in sorted(actual.items()):
        digest, sealed = expected[name]
        if sealed:
            output_digest = hashlib.sha256(unseal_bytes(path.read_bytes())).hexdigest()
        else:
            output_digest = _hash(path)
        if output_digest != digest:
            raise ValueError(f"Final model output differs from reviewed bytes: {name}")
        inventory.append({
            "path": name, "bytes": path.stat().st_size,
            "sha256": _hash(path) if sealed else output_digest,
            "reviewed_sha256": digest, "sealed": sealed,
        })
    return {"version": 1, "scope": "models", "files": inventory}
