# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
r"""Fetch the YOLOX person-detection weights the vision engine needs.

    venv\Scripts\python.exe scripts/fetch_person_model.py

The weights are not committed (20 MB binary) and are not a pip package, so a
fresh clone has to fetch them once before scanning or packaging. YOLOX and its
released weights are Apache-2.0, which is the entire point: they replaced
Ultralytics YOLO, whose AGPL-3.0 terms a closed-source build cannot satisfy.

The download is checksum-pinned. An unpinned model file is a silent correctness
risk -- a different export of "yolox_tiny" can carry a different input size or
output layout, and engines/vision/person_onnx.py would decode it into plausible
but wrong boxes rather than failing.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from core.bundle_paths import get_models_dir

TARGET_DIR = os.path.join(get_models_dir(), "vision")

# yolox_s is the shipping default; yolox_tiny is kept fetchable because it is
# ~40% faster and is the right pick if throughput ever outranks framing
# accuracy. Both are decoded by the same code -- input size comes from the graph.
MODELS = {
    "yolox_s.onnx": {
        "url": (
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
            "0.1.1rc0/yolox_s.onnx"
        ),
        "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        "size": 35858002,
        "license": "Apache-2.0 (Megvii-BaseDetection/YOLOX)",
    },
    "yolox_tiny.onnx": {
        "url": (
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
            "0.1.1rc0/yolox_tiny.onnx"
        ),
        "sha256": "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        "size": 20219662,
        "license": "Apache-2.0 (Megvii-BaseDetection/YOLOX)",
    },
}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, spec: dict, force: bool) -> bool:
    dest = os.path.join(TARGET_DIR, name)
    if os.path.exists(dest) and not force:
        actual = sha256(dest)
        if actual == spec["sha256"]:
            print(f"  {name}: already present and verified")
            return True
        print(f"  {name}: present but CHECKSUM MISMATCH -- refetching")
        print(f"      expected {spec['sha256']}")
        print(f"      actual   {actual}")

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"  {name}: downloading {spec['size'] / 1e6:.1f} MB")
    print(f"      from {spec['url']}")
    print(f"      license: {spec['license']}")
    temp = dest + ".part"
    try:
        with urllib.request.urlopen(spec["url"]) as response, open(temp, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as error:  # noqa: BLE001 - report cleanly, do not traceback
        print(f"      FAILED: {error}", file=sys.stderr)
        if os.path.exists(temp):
            os.remove(temp)
        return False

    actual = sha256(temp)
    if actual != spec["sha256"]:
        os.remove(temp)
        print("      FAILED: checksum mismatch after download", file=sys.stderr)
        print(f"        expected {spec['sha256']}", file=sys.stderr)
        print(f"        actual   {actual}", file=sys.stderr)
        return False

    os.replace(temp, dest)
    print(f"      OK -> {dest}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refetch even if present.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing files and exit non-zero if any is missing or wrong.",
    )
    args = parser.parse_args()

    print(f"Person-detection weights -> {TARGET_DIR}")
    ok = True
    for name, spec in MODELS.items():
        dest = os.path.join(TARGET_DIR, name)
        if args.verify_only:
            if not os.path.exists(dest):
                print(f"  {name}: MISSING")
                ok = False
            elif sha256(dest) != spec["sha256"]:
                print(f"  {name}: CHECKSUM MISMATCH")
                ok = False
            else:
                print(f"  {name}: OK")
            continue
        ok &= fetch(name, spec, args.force)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
