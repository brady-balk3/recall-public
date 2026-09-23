# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Stage pinned VC runtime DLLs without running Microsoft's installer.

Usage: python -m publish.prepare_microsoft_runtime
Requires 7-Zip (or the desktop's installed build dependencies). This prepares
build inputs; it does not grant redistribution rights or install system files.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import urllib.request

from publish.verify_native_inputs import MICROSOFT_RUNTIME_HASHES, pin_microsoft_runtime_inputs

VERSION = "14.51.36247.0"
ARCHIVE_SHA256 = "843068991daaa1f73ad9f6239bce4d0f6a07a51f18c37ea2a867e9beca71295c"
ARCHIVE_BYTES = 18731856
URL = (
    "https://download.visualstudio.microsoft.com/download/pr/"
    "ebdab8e5-1d7b-4d9f-a11b-cbb1720c3b12/"
    "843068991DAAA1F73AD9F6239BCE4D0F6A07A51F18C37EA2A867E9BECA71295C/VC_redist.x64.exe"
)
REPO = Path(__file__).resolve().parents[1]


def prepare(output, archive=None, seven_zip=None):
    output = Path(output).absolute()
    if output.resolve() != output:
        raise ValueError("Runtime destination must not be redirected")
    if output.exists():
        pin_microsoft_runtime_inputs([], output)
        return "already verified"
    candidates = [seven_zip] if seven_zip else [
        REPO / "apps/desktop/node_modules/7zip-bin/win/x64/7za.exe",
        REPO / "apps/desktop/node_modules/electron-winstaller/vendor/7z.exe",
        shutil.which("7z"),
    ]
    extractor = next((str(Path(p).absolute()) for p in candidates if p and Path(p).is_file()), None)
    if not extractor:
        raise ValueError("Install desktop build dependencies or supply --seven-zip PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="recall-vc-runtime-", dir=output.parent) as temporary:
        work = Path(temporary)
        if not work.resolve().is_relative_to(output.parent.resolve()):
            raise ValueError("Temporary runtime staging escaped its parent")
        if archive:
            source = Path(archive)
            if source.stat().st_size != ARCHIVE_BYTES:
                raise ValueError("Microsoft redistributable size mismatch")
            data = source.read_bytes()
        else:
            with urllib.request.urlopen(URL, timeout=60) as response:
                data = response.read(ARCHIVE_BYTES + 1)
        if len(data) != ARCHIVE_BYTES or hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Microsoft redistributable checksum mismatch")
        # Offsets and payload name belong to the exact authenticated package
        # above. Never interpret another EXE or execute this one.
        offset = 630000
        length = struct.unpack_from("<I", data, offset + 8)[0]
        if data[offset:offset + 4] != b"MSCF" or offset + length > len(data):
            raise ValueError("Pinned runtime cabinet layout changed")
        cabinet = work / "payload.cab"
        cabinet.write_bytes(data[offset:offset + length])
        payload = work / "payload"
        raw = work / "raw"
        subprocess.run([extractor, "e", str(cabinet), "a4", "-o" + str(payload), "-y"],
                       check=True, capture_output=True)
        members = [name + "_amd64" for name in MICROSOFT_RUNTIME_HASHES]
        subprocess.run([extractor, "e", str(payload / "a4"), *members, "-o" + str(raw), "-y"],
                       check=True, capture_output=True)
        staged = work / "verified"
        staged.mkdir()
        for name in MICROSOFT_RUNTIME_HASHES:
            shutil.copyfile(raw / (name + "_amd64"), staged / name)
        pin_microsoft_runtime_inputs([], staged)
        # Another builder may have completed the same prerequisite meanwhile.
        if output.exists():
            pin_microsoft_runtime_inputs([], output)
        else:
            staged.rename(output)
    return "prepared and verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded official EXE")
    parser.add_argument("--seven-zip", type=Path)
    parser.add_argument("--output", type=Path,
                        default=REPO / "vendor/microsoft-runtime" / VERSION / "x64")
    args = parser.parse_args()
    print("Microsoft runtime:", prepare(args.output, args.archive, args.seven_zip))


if __name__ == "__main__":
    main()
