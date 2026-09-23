# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Stage only explicitly reviewed public files; never copy working history.

Supply --manifest with JSON {"version": 1, "files": [{"path": "LICENSE",
"origin": "repository", "sha256": "<reviewed file hash>"}]}.
An origin of "override" selects the same relative path under publish/overrides.
Each final path occurs once. Hashes attest the exact reviewed bytes, not privacy
by themselves. Manifest approval and binary-payload review remain release gates.
Existing destinations are refused, never recursively removed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import shutil
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OVERRIDES = os.path.join(ROOT, "publish", "overrides")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WINDOWS_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.IGNORECASE)


def _inside(root: str, path: str) -> bool:
    root, path = os.path.realpath(root), os.path.realpath(path)
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def _hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Manifest paths must be relative POSIX paths")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(
        part in ("", ".", "..") or any(char in '<>:"|?*' for char in part)
        or part.endswith((" ", ".")) or _WINDOWS_DEVICE.match(part)
        or any(ord(char) < 32 for char in part)
        for part in parts
    ):
        raise ValueError("Unsafe manifest path")
    if any(part.lower() == ".git" for part in parts):
        raise ValueError("Git history must never enter the public tree")
    return value


def _reviewed_files(manifest_path: str):
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Unsupported public-file manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("A nonempty reviewed file manifest is required")
    planned, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid manifest entry")
        rel = _relative_path(entry.get("path"))
        origin = entry.get("origin")
        expected = entry.get("sha256")
        if origin not in ("repository", "override"):
            raise ValueError("Every file needs a repository or override origin")
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ValueError("Every file needs a lowercase SHA-256 of reviewed bytes")
        if rel.casefold() in seen:
            raise ValueError("Duplicate destination in public-file manifest")
        seen.add(rel.casefold())
        base = ROOT if origin == "repository" else OVERRIDES
        src = os.path.join(base, *rel.split("/"))
        if not _inside(base, src) or not os.path.isfile(src):
            raise ValueError(f"Missing or unsafe approved input: {rel}")
        resolved_parts = os.path.relpath(os.path.realpath(src), os.path.realpath(base)).replace("\\", "/").split("/")
        if any(part.casefold() == ".git" for part in resolved_parts):
            raise ValueError("Git history must never enter the public tree")
        if _hash(src) != expected:
            raise ValueError(f"Approved input changed; review required: {rel}")
        planned.append((rel, origin, src, expected))
    # Reject file/directory collisions before creating the destination.
    for rel in seen:
        parts = rel.split("/")
        if any("/".join(parts[:i]) in seen for i in range(1, len(parts))):
            raise ValueError("File/directory collision in public-file manifest")
    return planned


def build(out: str, manifest_path: str) -> tuple[list[str], list[str]]:
    out = os.path.abspath(out)
    if _inside(ROOT, out) or _inside(out, ROOT):
        raise ValueError("Public output must be separate from the working repository")
    planned = _reviewed_files(manifest_path)
    # All inputs are checked before creation. No deletion or overwrite path.
    os.makedirs(out, exist_ok=False)
    kept, overlaid = [], []
    for rel, origin, src, expected in planned:
        dst = os.path.join(out, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        if _hash(dst) != expected:
            os.unlink(dst)
            raise ValueError("Input changed during staging; discard this incomplete output")
        kept.append(rel)
        if origin == "override":
            overlaid.append(rel)
    return kept, overlaid


def verify(out: str, manifest_path: str) -> int:
    """Check the staged source without importing it or running development tests."""
    expected = {rel: digest for rel, _origin, _src, digest in _reviewed_files(manifest_path)}
    expected_dirs = {
        str(parent) for rel in expected for parent in PurePosixPath(rel).parents
        if str(parent) != "."
    }
    actual = set()
    if os.path.islink(out) or os.path.isjunction(out):
        raise ValueError("Public output must not be a redirected directory")
    for root, dirs, files in os.walk(out, followlinks=False):
        for name in dirs + files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, out).replace("\\", "/")
            if os.path.islink(path) or os.path.isjunction(path):
                raise ValueError(f"Redirected staged entry: {rel}")
            if name in dirs:
                if rel not in expected_dirs:
                    raise ValueError(f"Unexpected staged directory: {rel}")
                continue
            if not os.path.isfile(path) or rel not in expected:
                raise ValueError(f"Unexpected staged file: {rel}")
            if _hash(path) != expected[rel]:
                raise ValueError(f"Staged file changed: {rel}")
            if rel.endswith((".py", ".spec")):
                with open(path, "rb") as stream:
                    compile(stream.read(), rel, "exec")
            # TypeScript configs allow JSON comments/trailing commas; tsc
            # validates those during the desktop build instead.
            elif rel.endswith(".json") and not PurePosixPath(rel).name.startswith("tsconfig"):
                with open(path, encoding="utf-8-sig") as stream:
                    json.load(stream)
            actual.add(rel)
    if actual != set(expected):
        raise ValueError("Staged source is missing reviewed files")
    print(f"Verified {len(actual)} exact source files and Python/JSON syntax; no application code executed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new directory, outside the working repository")
    parser.add_argument("--manifest", required=True, help="reviewed paths, origins and SHA-256 hashes")
    parser.add_argument("--verify", action="store_true", help="check exact staged files, hashes and Python/JSON syntax")
    args = parser.parse_args()
    kept, overlaid = build(args.out, args.manifest)
    print(f"Staged {len(kept)} approved files, including {len(overlaid)} explicit overrides.")
    for rel in kept:
        print(f"{_hash(os.path.join(args.out, *rel.split('/')))}  {rel}")
    print("Stage a fresh repository. This manifest does not certify privacy or binary packaging.")
    return verify(args.out, args.manifest) if args.verify else 0


if __name__ == "__main__":
    raise SystemExit(main())
