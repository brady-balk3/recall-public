# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Copy explicitly reviewed dependency notices and verify the installed bytes.

The manifest records a relative source_root and files with path/sha256 entries.
This verifies file integrity, not completeness of licensing obligations.
"""
import json
from pathlib import Path
import re

from publish.build_public_tree import _hash, _relative_path


def notice_payload(manifest_path):
    manifest_path = Path(manifest_path).absolute()
    if manifest_path.resolve() != manifest_path:
        raise ValueError("Notice manifest must not be redirected")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_root = _relative_path(document.get("source_root"))
    root = manifest_path.parent / relative_root
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Notice source root is missing or redirected")
    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a nonempty dependency notice manifest")
    expected, datas, seen = {}, [], set()
    for entry in entries:
        name = _relative_path(entry.get("path"))
        digest = entry.get("sha256")
        if name.casefold() in seen or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Duplicate notice path or invalid hash")
        seen.add(name.casefold())
        source = root / name
        if source.resolve() != source or not source.is_file() or _hash(source) != digest:
            raise ValueError(f"Dependency notice changed or missing: {name}")
        expected[name] = digest
        datas.append((str(source), str(Path("notices/python-dependencies") / Path(name).parent)))
    return datas, expected


def verify_notice_payload(root, expected):
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Dependency notice output is missing or redirected")
    actual = set()
    for path in root.rglob("*"):
        if path.resolve() != path:
            raise ValueError("Redirected dependency notice output")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name not in expected or _hash(path) != expected[name]:
                raise ValueError(f"Unexpected or changed dependency notice: {name}")
            actual.add(name)
    if actual != set(expected):
        raise ValueError("Dependency notice output is incomplete")
    return {"scope": "notices/python-dependencies", "verified_files": len(actual)}
