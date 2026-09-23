# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Fetch pinned VAD and speaker assets into the canonical runtime model root."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bundle_paths import get_models_dir
from core.model_catalog import VAD_REPO, VAD_REVISION, VAD_FILES, SPEAKER_REPO, SPEAKER_REVISION, SPEAKER_FILES
from core.model_downloads import install_files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("vad", "speaker", "all"), default="all")
    args = parser.parse_args(argv)
    root = Path(get_models_dir())
    groups = (
        ("vad", VAD_REPO, VAD_REVISION, VAD_FILES, root / "asr" / "silero-vad"),
        ("speaker", SPEAKER_REPO, SPEAKER_REVISION, SPEAKER_FILES, root / "speaker"),
    )
    for group, repo, revision, files, target in groups:
        if args.group not in ("all", group):
            continue
        install_files(repo, revision, files, str(target))
        print(f"Verified {group} assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
