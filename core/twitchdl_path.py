# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Resolve the TwitchDownloaderCLI binary used to fetch VOD chat replay.

Chat replay is fetched by shelling out to TwitchDownloaderCLI's ``chatdownload``
command (it uses Twitch's current API with proper auth and exits cleanly),
replacing the abandoned ``chat_downloader`` PyPI package, whose Twitch handler
is broken against the current API and hangs in an infinite retry loop.

Resolution order mirrors core/ffmpeg_path.py:
  1. RECALL_TWITCHDL_PATH env var (set by electron/main.ts when packaged).
  2. A copy bundled alongside a PyInstaller build (sys._MEIPASS / exe dir).
  3. A dev copy under <project_root>/tools/twitchdownloader/.
  4. Bare "TwitchDownloaderCLI", relying on PATH.

Returns None only when nothing resolves AND the name isn't on PATH, so callers
can cleanly drop the (optional) chat channel.
"""

import os
import shutil
import sys

from core.bundle_paths import PROJECT_ROOT, is_frozen

_EXE = "TwitchDownloaderCLI.exe" if os.name == "nt" else "TwitchDownloaderCLI"
_resolved: str | None = None
_checked = False


def get_twitchdl_path() -> str | None:
    """Absolute path to TwitchDownloaderCLI, or the bare name if only on PATH,
    or None when it can't be found at all."""
    global _resolved, _checked
    if _checked:
        return _resolved
    _checked = True

    env_path = os.environ.get("RECALL_TWITCHDL_PATH")
    if env_path and os.path.exists(env_path):
        _resolved = env_path
        return _resolved

    bundle_dir = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    candidates = [os.path.join(bundle_dir, "twitchdownloader", _EXE)]
    if not is_frozen():
        candidates.append(os.path.join(PROJECT_ROOT, "tools", "twitchdownloader", _EXE))
    for candidate in candidates:
        if os.path.exists(candidate):
            _resolved = candidate
            return _resolved

    # Last resort: on PATH under its bare name.
    on_path = shutil.which("TwitchDownloaderCLI") or shutil.which(_EXE)
    _resolved = on_path
    return _resolved
