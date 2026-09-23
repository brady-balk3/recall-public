# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Build a Zip64 portable archive without risking the last successful archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import tempfile
import zipfile
import sys

# Keep the direct script entry point usable outside the repository cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from publish.verify_release_privacy import verify_release_privacy


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Portable payload must not contain symbolic links or junctions")
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError("Portable payload contains a nonregular file")
    return info


def _inventory(source_path: Path):
    files = []
    def unreadable(error):
        raise error

    for root, directories, names in os.walk(source_path, onerror=unreadable):
        for name in directories:
            _regular(Path(root) / name)
        for name in names:
            path = Path(root) / name
            info = _regular(path)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Portable payload changed during packaging")
            files.append((path, info.st_size, info.st_mtime_ns))
    return sorted(files)


def build(source: str, destination: str) -> int:
    source_path = Path(source).absolute()
    _regular(source_path)
    source_path = source_path.resolve()
    destination_path = Path(destination).absolute()
    if not source_path.is_dir():
        raise ValueError("Portable source must be a directory")
    verify_release_privacy(source_path)
    if destination_path.resolve().is_relative_to(source_path):
        raise ValueError("Archive must be outside the payload")
    files = _inventory(source_path)
    if not files:
        raise ValueError("Refusing an empty portable payload")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".recall-package-", suffix=".zip", dir=destination_path.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path, size, modified in sorted(files):
                before = _regular(path)
                if (before.st_size, before.st_mtime_ns) != (size, modified):
                    raise ValueError("Portable payload changed during packaging")
                archive.write(path, path.relative_to(source_path).as_posix())
                after = _regular(path)
                if (after.st_size, after.st_mtime_ns) != (size, modified):
                    raise ValueError("Portable payload changed during packaging")
        if _inventory(source_path) != files:
            raise ValueError("Portable payload changed during packaging")
        verify_release_privacy(source_path)
        # Close the complete ZIP before atomically replacing the previous build.
        os.replace(temporary, destination_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(f"Published portable archive ({build(args.source, args.out)} files).")


if __name__ == "__main__":
    main()
