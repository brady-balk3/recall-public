# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""PyInstaller runtime hook: `recall-engine.exe --recall-gpu-smoke <report.json>`.

Runs apps/api/engine_gpu_smoke.py and exits before main.py, so the check builds
no database, job manager or server and never touches the data folder. The app
never passes this flag. It is a runtime hook rather than a branch in main.py
so the diagnostic stays out of the API's startup code; it runs after
pyi_rth_recall_cuda.py (recall-engine.spec lists it second), so the check sees
the same DLL search path a real engine process gets.
"""


def _recall_gpu_smoke():
    import sys

    if "--recall-gpu-smoke" not in sys.argv:
        return
    from apps.api.engine_gpu_smoke import main

    raise SystemExit(main(sys.argv[1:]))


_recall_gpu_smoke()
del _recall_gpu_smoke
