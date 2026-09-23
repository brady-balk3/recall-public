# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Keep Recall's *own* trained artifacts out of plain sight in a packaged build.

WHAT THIS IS NOT
================
This is obfuscation, not security. The key ships inside the same binary that
does the decrypting, so anyone willing to read the bytecode can recover it —
that is unavoidable for any artifact that must be readable on a user's machine
without a network round-trip. Do not build a licensing or entitlement story on
top of it.

WHAT IT ACTUALLY BUYS
=====================
Before: `models/ranker_base.json` shipped as pretty-printed JSON. Unzip the
release and you are reading the 32 feature names that *are* the selection
design, plus the trained weights, plus `"architecture":
"shallow_tanh_logit_ensemble"`. No tooling, no expertise, no effort.

After: the payload is a high-entropy blob. `strings`, a text editor, and a
casual browse through the archive all come up empty. Recovering it means
locating the loader in stripped bytecode and reimplementing the derivation.
That moves the cost from "zero" to "a motivated afternoon", which is the
entire realistic goal.

Public third-party weights (Qwen, YOLO, PP-OCR, faster-whisper, YAMNet,
hsemotion) are deliberately NOT sealed: they are freely downloadable, sealing
them would buy nothing, and it would add ~7 GB of pointless work to every
build.

DEV VS PACKAGED
===============
The repo keeps the plain `.json` as the source of truth, so training scripts,
eval runs, and hand-inspection are all untouched. `seal_file()` runs at package
time (see apps/api/recall-engine.spec) and emits a `.sealed` sibling; only the
sealed form is bundled. `load_protected_json()` prefers plain JSON when it is
present and falls back to the sealed blob, so the same loader serves both
modes and dev never depends on the sealing path being correct.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import struct
import zlib
from typing import Any

SEALED_SUFFIX = ".sealed"

_MAGIC = b"RCLV1\x00"
_HEADER = struct.Struct("<6sH32s")  # magic, salt length, HMAC digest

# Assembled at import rather than written as one literal so the secret is not a
# single contiguous run of bytes in the frozen bytecode. A disassembler still
# recovers it; `strings` on the 75 MB exe does not. That asymmetry is the point
# — see the module docstring on what this does and does not claim.
_KEY_PARTS = (
    b"recall.vault",
    bytes((0x5B, 0xA1, 0x0C, 0xE7, 0x39, 0x8D, 0x42, 0xF6)),
    b"ranker+rejector",
    bytes((0xC4, 0x17, 0x9E, 0x2B, 0x60, 0xDA, 0x85, 0x31)),
)


def _root_key() -> bytes:
    digest = hashlib.sha256()
    for part in _KEY_PARTS:
        digest.update(part)
    return digest.digest()


def _keystream(key: bytes, salt: bytes, length: int) -> bytes:
    """SHA-256 in counter mode.

    Deliberately stdlib-only. pycryptodome is present in the bundle today, but
    reaching for it here would add a hidden-import edge to a PyInstaller spec
    that has a documented history of breaking in ways that only surface after
    an 8 GB build.
    """
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(key + salt + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:length])


def _derive(salt: bytes) -> tuple[bytes, bytes]:
    """(cipher key, MAC key) for one artifact. Distinct so neither reveals the other."""
    base = _root_key()
    return (
        hashlib.sha256(base + b"\x01" + salt).digest(),
        hashlib.sha256(base + b"\x02" + salt).digest(),
    )


def seal_bytes(plaintext: bytes, *, salt: bytes | None = None) -> bytes:
    """Compress, encrypt, and authenticate. Compression first: it collapses the
    repeated key names in a 4 MB JSON model and leaves no structure for the
    keystream to expose."""
    salt = salt if salt is not None else os.urandom(16)
    cipher_key, mac_key = _derive(salt)
    compressed = zlib.compress(plaintext, 9)
    body = bytes(
        a ^ b for a, b in zip(compressed, _keystream(cipher_key, salt, len(compressed)))
    )
    tag = hmac.new(mac_key, salt + body, hashlib.sha256).digest()
    return _HEADER.pack(_MAGIC, len(salt), tag) + salt + body


def unseal_bytes(blob: bytes) -> bytes:
    """Inverse of `seal_bytes`. Raises ValueError on anything unexpected.

    Fails closed on purpose: a truncated or tampered artifact must not decode
    into a *partially* valid model that silently reranks someone's clips.
    """
    if len(blob) < _HEADER.size:
        raise ValueError("Sealed artifact is truncated.")
    magic, salt_length, tag = _HEADER.unpack(blob[: _HEADER.size])
    if magic != _MAGIC:
        raise ValueError("Not a sealed Recall artifact.")
    salt = blob[_HEADER.size : _HEADER.size + salt_length]
    body = blob[_HEADER.size + salt_length :]
    if len(salt) != salt_length:
        raise ValueError("Sealed artifact is truncated.")
    cipher_key, mac_key = _derive(salt)
    expected = hmac.new(mac_key, salt + body, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, tag):
        raise ValueError("Sealed artifact failed its integrity check.")
    compressed = bytes(
        a ^ b for a, b in zip(body, _keystream(cipher_key, salt, len(body)))
    )
    return zlib.decompress(compressed)


def seal_file(source_path: str, target_path: str | None = None) -> str:
    """Write the sealed form of `source_path`. Returns the path written."""
    target_path = target_path or (source_path + SEALED_SUFFIX)
    with open(source_path, "rb") as handle:
        payload = handle.read()
    with open(target_path, "wb") as handle:
        handle.write(seal_bytes(payload))
    return target_path


def sealed_path_for(path: str) -> str:
    return path + SEALED_SUFFIX


def protected_json_exists(path: str) -> bool:
    """True when either form is available, so callers keep their existing
    `os.path.isfile` fail-closed behaviour without knowing about sealing."""
    return os.path.isfile(path) or os.path.isfile(sealed_path_for(path))


def load_protected_json(path: str) -> Any:
    """Load `path` as JSON, transparently preferring plain over sealed.

    Plain-first keeps dev, training, and eval on exactly the bytes a human can
    open, and means a broken sealing step can never masquerade as a working
    build: in the packaged tree the plain file does not exist at all.
    """
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    sealed = sealed_path_for(path)
    if os.path.isfile(sealed):
        with open(sealed, "rb") as handle:
            return json.loads(unseal_bytes(handle.read()).decode("utf-8"))
    raise FileNotFoundError(path)
