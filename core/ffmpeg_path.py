# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Resolve the ffmpeg binary to invoke (handbook/20, exe bundling pass).

In dev, ffmpeg is whatever's on PATH (the engines have always assumed this).
In the packaged app, there is no guarantee the end user has ffmpeg installed at
all, so PyInstaller bundles a copy next to the backend exe and Electron points
us at it via the RECALL_FFMPEG_PATH env var. Resolution order:

1. RECALL_FFMPEG_PATH env var (set by electron/main.ts when packaged).
2. A copy bundled alongside this PyInstaller build (sys._MEIPASS / exe dir).
3. Bare "ffmpeg", relying on PATH (dev mode).
"""

import os
import sys

_resolved: str | None = None


def get_ffmpeg_path() -> str:
    global _resolved
    if _resolved is not None:
        return _resolved

    env_path = os.environ.get("RECALL_FFMPEG_PATH")
    if env_path and os.path.exists(env_path):
        _resolved = env_path
        return _resolved

    # PyInstaller onedir builds expose the bundle dir as sys._MEIPASS; a onefile
    # build also extracts there at runtime. Either way, look for ffmpeg.exe
    # bundled as a sibling resource.
    bundle_dir = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    candidate = os.path.join(bundle_dir, "ffmpeg", "ffmpeg.exe")
    if os.path.exists(candidate):
        _resolved = candidate
        return _resolved

    _resolved = "ffmpeg"
    return _resolved
