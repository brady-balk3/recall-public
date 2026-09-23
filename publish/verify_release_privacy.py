# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Reject known Recall runtime state at release layout boundaries.

This is a read-only backstop for reused build folders, not a content scanner or
a substitute for reviewed source/model inputs. Third-party package data deeper
inside _internal is outside this check; its notices/assets have separate gates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import stat


def _redirected(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def verify_release_privacy(root: str | Path) -> None:
    root = Path(root).absolute()
    if _redirected(root) or not root.is_dir():
        raise ValueError("Release privacy check requires a regular payload directory")
    # Current engine and Electron layouts, including the older in-place layout.
    boundaries = [Path("."), Path("_internal"), Path("resources"),
                  Path("resources/recall-engine"), Path("resources/recall-engine/_internal")]
    private_dirs = {"data", "model-setup", "logs", ".git"}
    for relative in boundaries:
        directory = root / relative
        # Examine each existing ancestor before traversing a nested boundary.
        for part in reversed((relative, *relative.parents)):
            candidate = root / part
            if candidate.is_symlink() or (candidate.exists() and _redirected(candidate)):
                raise ValueError(f"Redirected release boundary: {part.as_posix()}")
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError(f"Invalid release directory: {relative.as_posix()}")
        for entry in directory.iterdir():
            name = entry.name.casefold()
            if (name in private_dirs or name == ".env" or name.startswith(".env.")
                    or name.startswith("ranker_user")
                    or name.endswith((".log", ".db", ".db-wal", ".db-shm", ".db-journal",
                                      ".sqlite", ".sqlite-wal", ".sqlite-shm", ".sqlite-journal",
                                      ".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".sqlite3-journal"))):
                raise ValueError(
                    f"Private runtime state in release: {entry.relative_to(root).as_posix()}. "
                    "Build into a fresh directory; existing data has not been changed."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    verify_release_privacy(parser.parse_args().root)
    print("Known Recall runtime-data locations are absent from the release layout.")
