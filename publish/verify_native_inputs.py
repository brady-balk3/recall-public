# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Reject native build inputs discovered in unrelated applications or PATH caches.

This checks origin boundaries, not redistribution rights. Package and vendor
libraries inside the accepted roots still need their own license review.
"""
from pathlib import Path
import hashlib
import os
import sys


# Official VC redistributable 14.51.36247.0, x64. These four root-level
# dependencies previously came from System32. Keep wheel-local DLLs intact.
MICROSOFT_RUNTIME_HASHES = {
    "msvcp140_1.dll": "206c931bf90fdad8816de3b5e2ef80b2bcaa9406c89ecc05fe6fddffe251e982",
    "msvcp140_atomic_wait.dll": "3d0cbfaa1bf3eecf5a3f4491d2960ee803cb994f30292c6adc4a07c498f60e2b",
    "vcomp140.dll": "95d4ce4a6802d1e18b5e0e1722cc30ea72ca7e033f83828f05c0b7b993fe7cbf",
    "msvcp140.dll": "7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca",
}


def pin_microsoft_runtime_inputs(binaries, runtime_root):
    """Bind root-level VC dependencies to reviewed vendor bytes, or fail.

    Call before Analysis with an empty list to validate prerequisites early,
    then on its completed binary table. Nested wheel dependencies are unchanged.
    """
    root = Path(runtime_root).absolute()
    if root.resolve() != root:
        raise ValueError("Microsoft runtime directory must not be redirected")
    sources = {}
    for name, digest in MICROSOFT_RUNTIME_HASHES.items():
        path = root / name
        if (path.resolve() != path or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError(f"Missing or changed pinned Microsoft runtime: {path}")
        sources[name] = str(path)
    return [
        (destination, sources.get(destination.casefold(), source), kind)
        if kind == "BINARY" else (destination, source, kind)
        for destination, source, kind in binaries
    ]


def configure_native_search():
    """Set the DLL search PATH inside the build interpreter, before analysis.

    Shell-level PATH overrides may not survive a managed process launcher.
    PyInstaller still discovers native dependencies inside imported wheels.
    """
    windows = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    paths = [Path(sys.executable).parent, Path(sys.base_prefix),
             Path(sys.base_prefix) / "DLLs", windows / "System32", windows]
    os.environ["PATH"] = os.pathsep.join(str(p) for p in paths)
    return paths


def verify_native_inputs(binaries, allowed_roots):
    roots = [Path(root).resolve() for root in allowed_roots]
    if not roots:
        raise ValueError("Native input verification requires explicit source roots")
    rejected = []
    for destination, source, _kind in binaries:
        path = Path(source).resolve()
        if not path.is_file() or not any(path.is_relative_to(root) for root in roots):
            rejected.append(f"{destination}: {source}")
    if rejected:
        raise ValueError(
            "Native dependencies came from outside the build environment, vendor directory, "
            "or Windows. Use a controlled build PATH and reviewed runtime inputs.\n"
            + "\n".join(rejected)
        )
    return {"verified_native_inputs": len(binaries)}
