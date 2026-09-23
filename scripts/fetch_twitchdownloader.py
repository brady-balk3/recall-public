# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Fetch the pinned Windows x64 TwitchDownloader CLI used for VOD ingestion.

The official release archive and all three installed files are checksum-pinned.
The copyright and third-party notices must travel with the executable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
TARGET = TOOLS / "twitchdownloader"
URL = (
    "https://github.com/lay295/TwitchDownloader/releases/download/1.56.4/"
    "TwitchDownloaderCLI-1.56.4-Windows-x64.zip"
)
ARCHIVE_BYTES = 45795843
ARCHIVE_SHA256 = "2d0545fd22d0860aaeafb14afa771270b4351a8535a915fc6c6488d6aec31f42"
FILES = {
    "TwitchDownloaderCLI.exe": (68441306, "c2f08e759ef110ed420bbb8419fd7b1cbcae59c80e964f270d7ff6ea1e05472c"),
    "COPYRIGHT.txt": (2199, "0bbe7b0458bed417f70981a598fd257a5fee32fe3e70ccdf178774b9aa0dbf3e"),
    "THIRD-PARTY-LICENSES.txt": (156975, "826f542cf7f4ee8e06f75876a6066e2ec370b28803b764990b47b9e50cd742ad"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unredirected(path: Path) -> bool:
    return path.resolve() == path.absolute()


def _ready(directory: Path) -> bool:
    if not _unredirected(directory) or not directory.is_dir():
        return False
    if {path.name for path in directory.iterdir()} != set(FILES):
        return False
    return all(
        (directory / name).is_file()
        and _unredirected(directory / name)
        and (directory / name).stat().st_size == size
        and _sha256(directory / name) == digest
        for name, (size, digest) in FILES.items()
    )


def fetch(archive: Path | None = None) -> str:
    if _ready(TARGET):
        return "already verified"
    if TARGET.exists():
        raise ValueError("Existing TwitchDownloader directory differs from the pinned release; move it aside before fetching")
    if not _unredirected(TOOLS) or not _unredirected(TARGET):
        raise ValueError("TwitchDownloader destination redirects outside the repository")
    TOOLS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".twitchdownloader-", dir=TOOLS) as temporary:
        stage = Path(temporary)
        package = stage / "release.zip"
        if archive is None:
            with urllib.request.urlopen(URL, timeout=60) as response, package.open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > ARCHIVE_BYTES:
                        raise ValueError("TwitchDownloader archive exceeded the pinned size")
                    output.write(chunk)
        else:
            archive = archive.absolute()
            if not archive.is_file() or not _unredirected(archive):
                raise ValueError("Supplied TwitchDownloader archive is missing or redirected")
            with archive.open("rb") as source, package.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
        if package.stat().st_size != ARCHIVE_BYTES or _sha256(package) != ARCHIVE_SHA256:
            raise ValueError("TwitchDownloader archive size or checksum mismatch")

        payload = stage / "payload"
        payload.mkdir()
        with zipfile.ZipFile(package) as bundle:
            if set(bundle.namelist()) != set(FILES):
                raise ValueError("TwitchDownloader archive file list changed")
            for name, (size, digest) in FILES.items():
                member = bundle.getinfo(name)
                if member.is_dir() or member.file_size != size:
                    raise ValueError(f"TwitchDownloader file size changed: {name}")
                output_path = payload / name
                with bundle.open(member) as source, output_path.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                if output_path.stat().st_size != size or _sha256(output_path) != digest:
                    raise ValueError(f"TwitchDownloader file checksum changed: {name}")
        if not _ready(payload):
            raise ValueError("TwitchDownloader staged files did not verify")
        if TARGET.exists():
            if _ready(TARGET):
                return "already verified"
            raise ValueError("TwitchDownloader destination changed while fetching")
        os.replace(payload, TARGET)
    if not _ready(TARGET):
        raise ValueError("Published TwitchDownloader files did not verify")
    return "fetched and verified"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a saved copy of the official release archive")
    parser.add_argument("--verify-only", action="store_true", help="Check installed files without downloading")
    args = parser.parse_args()
    if args.verify_only:
        if not _ready(TARGET):
            raise ValueError("Pinned TwitchDownloader CLI and notices are missing or changed")
        print("TwitchDownloader CLI and notices verified")
    else:
        print("TwitchDownloader:", fetch(args.archive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
