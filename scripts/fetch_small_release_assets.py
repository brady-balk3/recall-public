# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Fetch the checksum-pinned small model assets required by packaged Recall.

The OCR files come from PaddlePaddle's official Hugging Face repositories.
YAMNet's ONNX export and its AudioSet class map come from separate public
conversions. The Face Landmarker task is served by Google. Keep exact artifact
hashes here so a fresh checkout cannot silently package different models.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bundle_paths import get_models_dir
from core.model_downloads import matches


# (local path, upstream file, bytes, local SHA-256, upstream SHA-256 if the
# checked-in local file uses Windows CRLF instead of the publisher's LF).
OCR_MODELS = {
    "PP-OCRv6_small_det_onnx": (
        "28fe5895c24fd108c19eb3e8479f4ab385fbfc62",
        (
            ("inference.onnx", 9880512, "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e", None),
            ("inference.json", 229307, "89240f689a4a77aad75ef55a8df0a15c8e1d4980a327d17e58f24bbadde5aeab", None),
            ("inference.yml", 937, "e75f7f36ade9b9e0545c18aac4eaff2546c61d0809662735b3efa88e86eb0e58", "193f435274bf9f0b5f71a929bbfbcf148282df7e633b34e7c373e8f44741b516"),
        ),
    ),
    "PP-OCRv6_small_rec_onnx": (
        "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
        (
            ("inference.onnx", 21159378, "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634", None),
            ("inference.json", 208004, "f0bf53c853937a917affdd74467472167727f8ab0f0f7bded01c4a16c27e46e6", None),
            ("inference.yml", 169330, "26a2802dcc5b4553f0961ab0288bc491ce86069abbd69c19d61779940c9c7ab3", "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1"),
        ),
    ),
}

YAMNET = (
    ("yamnet.onnx", "zeropointnine/yamnet-onnx", "ac2ca3bd45d12ec1f19f1144205ea529b4e9dedf", 16093603, "1510041dce24a2e9e84ec546807ac408ae496da6d1ed41bc3ccba649623f8e19"),
    ("yamnet_class_map.csv", "audiomagic/yamnet-onnx", "f25b741c2f0bdc6d7e6db24b5fddda23347dbafd", 14096, "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"),
)

FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_EXPECTED = (3758596, "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff")


def _publish(source, destination: Path, expected: tuple[int, str], upstream_sha256: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".recall-small-model-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            if upstream_sha256:
                # Existing release manifests pin CRLF YAML. Validate the exact
                # official LF file before reproducing those local bytes.
                raw = source.read()
                if hashlib.sha256(raw).hexdigest() != upstream_sha256:
                    raise ValueError(f"Upstream checksum mismatch: {destination.name}")
                output.write(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
            else:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
        if not matches(Path(temporary), expected):
            raise ValueError(f"Downloaded model size/checksum mismatch: {destination}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fetch_hf(repo: str, revision: str, name: str, destination: Path,
              expected: tuple[int, str], upstream_sha256: str | None = None) -> None:
    if matches(destination, expected):
        print(f"ready {destination}")
        return
    from huggingface_hub import hf_hub_download

    for force in (False, True):
        cached = Path(hf_hub_download(
            repo_id=repo, revision=revision, filename=name,
            force_download=force, token=False,
        ))
        try:
            with cached.open("rb") as source:
                _publish(source, destination, expected, upstream_sha256)
            print(f"verified {destination}")
            return
        except ValueError:
            if force:
                raise


def _fetch_face(destination: Path) -> None:
    if matches(destination, FACE_EXPECTED):
        print(f"ready {destination}")
        return
    with urllib.request.urlopen(FACE_URL, timeout=60) as source:
        _publish(source, destination, FACE_EXPECTED)
    print(f"verified {destination}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "ocr", "yamnet", "face"), default="all")
    parser.add_argument("--check", action="store_true", help="Check local bytes without downloading")
    args = parser.parse_args(argv)
    root = Path(get_models_dir())
    selected = []
    if args.group in ("all", "ocr"):
        for model, (revision, files) in OCR_MODELS.items():
            for name, size, digest, upstream in files:
                selected.append(("hf", f"PaddlePaddle/{model}", revision, name,
                                 root / "ocr" / model / name, (size, digest), upstream))
    if args.group in ("all", "yamnet"):
        for name, repo, revision, size, digest in YAMNET:
            selected.append(("hf", repo, revision, name, root / name, (size, digest), None))
    if args.group in ("all", "face"):
        selected.append(("face", None, None, None, root / "face" / "face_landmarker.task",
                         FACE_EXPECTED, None))

    missing = False
    for kind, repo, revision, name, destination, expected, upstream in selected:
        if args.check:
            verified = matches(destination, expected)
            print(f"{'ready' if verified else 'missing or invalid'} {destination}")
            missing |= not verified
        elif kind == "face":
            _fetch_face(destination)
        else:
            _fetch_hf(repo, revision, name, destination, expected, upstream)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
