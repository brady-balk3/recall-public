# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.stream_memory_sentence import sentence_model_dir


REVISION = "5641a7880f40ebf4035d05e60c5f9b7a9c272c84"
BASE_URL = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
    f"{REVISION}/"
)
FILES = {
    "onnx/model_quint8_avx2.onnx": (
        "model_quint8_avx2.onnx",
        "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
    ),
    "tokenizer.json": (
        "tokenizer.json",
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    ),
}
LICENSE_URL = (
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
    "826711e54e001c83835913827a843d8dd0a1def9/LICENSE"
)
LICENSE_SHA256 = "1e66d43b04a3f2428303ad3316d1fbb996991541192892b22d21f0065a093b2b"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    destination = sentence_model_dir()
    os.makedirs(destination, exist_ok=True)
    downloads = [(BASE_URL + remote, local, digest)
                 for remote, (local, digest) in FILES.items()]
    downloads.append((LICENSE_URL, "LICENSE", LICENSE_SHA256))
    for url, local_name, expected_hash in downloads:
        target = os.path.join(destination, local_name)
        if os.path.isfile(target) and (not expected_hash or _sha256(target) == expected_hash):
            print(f"ready {local_name}")
            continue
        handle, temp_path = tempfile.mkstemp(prefix="recall-memory-model-", dir=destination)
        os.close(handle)
        try:
            print(f"downloading {local_name}")
            urllib.request.urlretrieve(url, temp_path)
            actual_hash = _sha256(temp_path)
            if expected_hash and actual_hash != expected_hash:
                raise RuntimeError(
                    f"SHA-256 mismatch for {local_name}: {actual_hash}"
                )
            os.replace(temp_path, target)
            print(f"installed {local_name} ({os.path.getsize(target)} bytes, {actual_hash})")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
