# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Reviewed per-file model payload selection for PyInstaller builds.

Manifest: {"version": 1, "files": [{"path": "ocr/.../inference.onnx",
"sha256": "<reviewed SHA-256>", "role": "asset", "notices": ["ocr/LICENSE"]},
{"path": "ocr/LICENSE", "sha256": "<reviewed SHA-256>", "role": "notice"}]}.
Paths are relative to models/. Every asset must explicitly list its notices;
an empty list records that the reviewer selected no accompanying notices.
Approval must include required notices; hashes alone do not certify privacy
or redistribution rights. Never populate an approved manifest automatically.
"""

import json
from pathlib import Path
import re

from publish.build_public_tree import _hash, _relative_path


def validate_model_destinations(datas: list[tuple[str, str]]) -> None:
    """Reject ambiguous model destinations before PyInstaller chooses a winner."""
    selected = {}
    for source_name, destination in datas:
        destination = str(destination).replace("\\", "/").rstrip("/")
        if destination.casefold() != "models" and not destination.casefold().startswith("models/"):
            continue
        source = Path(source_name).absolute()
        if not source.is_file():
            raise ValueError("Model payload entries must select individual files")
        target = _relative_path(f"{destination}/{source.name}").casefold()
        if target in selected and selected[target] != source:
            raise ValueError(f"Conflicting model payload destination: {target}")
        selected[target] = source


class ModelPayloadManifest:
    def __init__(self, models_root: str, manifest_path: str):
        self.root = Path(models_root).resolve()
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise ValueError("Unsupported model payload manifest")
        entries = manifest.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("A reviewed nonempty model payload manifest is required")
        self.files = {}
        self.notices = set()
        self.asset_notices = {}
        self.selected_notices = set()
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid model payload entry")
            name = _relative_path(entry.get("path"))
            digest = entry.get("sha256")
            role = entry.get("role", "asset")
            if role not in ("asset", "notice"):
                raise ValueError("Model payload role must be asset or notice")
            if name.casefold() in seen:
                raise ValueError("Duplicate model payload destination")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Model payload entries require a SHA-256")
            seen.add(name.casefold())
            self.files[name] = digest
            if role == "notice":
                if "notices" in entry:
                    raise ValueError("Notice entries cannot reference other notices")
                self.notices.add(name)
            else:
                references = entry.get("notices")
                if not isinstance(references, list):
                    raise ValueError("Every model asset requires an explicit notices list")
                references = [_relative_path(reference) for reference in references]
                if len({reference.casefold() for reference in references}) != len(references):
                    raise ValueError("Duplicate model notice reference")
                self.asset_notices[name] = references
        for references in self.asset_notices.values():
            if not set(references).issubset(self.notices):
                raise ValueError("Model notice references must name reviewed notice entries")

    def verify_source(self, source: str | Path, *, asset: bool = False, notice: bool = False) -> Path:
        """Require review for a single source, including files sealed later."""
        if asset and notice:
            raise ValueError("A model payload cannot require both roles")
        source = Path(source).absolute()
        try:
            name = source.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("Model payload source must be inside models/") from exc
        if (name not in self.files or (asset and name in self.notices)
                or (notice and name not in self.notices)):
            raise ValueError(f"Model payload needs manifest approval: {name}")
        self._verify_file(source, name)
        references = self.asset_notices.get(name, [])
        for reference in references:
            self._verify_file(self.root.joinpath(*reference.split("/")), reference)
        self.selected_notices.update(references)
        if name in self.notices:
            self.selected_notices.add(name)
        return source

    def _verify_file(self, source: Path, name: str) -> None:
        if source.resolve() != source or not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"Missing, empty or redirected model payload: {name}")
        if _hash(source) != self.files[name]:
            raise ValueError(f"Model payload changed; review required: {name}")

    def notice_datas(self) -> list[tuple[str, str]]:
        """Collect notices for selected assets, including directly bundled files."""
        result = []
        for name in sorted(self.selected_notices):
            source = self.root.joinpath(*name.split("/"))
            self._verify_file(source, name)
            result.append((str(source), str(Path("models", *name.split("/")[:-1]))))
        return result

    def directory(self, relative: str, required: list[str], *, pinned: dict | None = None) -> list[tuple[str, str]]:
        """Return explicit PyInstaller datas, including approved notices only."""
        relative = _relative_path(relative)
        required_paths = {f"{relative}/{_relative_path(name)}" for name in required}
        if not required_paths.issubset(self.files):
            raise ValueError(f"Selected runtime files need manifest approval: {relative}")
        if required_paths & self.notices:
            raise ValueError("Runtime model files must use the asset role")
        for name, (_, digest) in (pinned or {}).items():
            if self.files.get(f"{relative}/{name}") != digest:
                raise ValueError(f"Reviewed payload does not match the pinned runtime catalog: {relative}")
        chosen = required_paths | {
            notice for name in required_paths for notice in self.asset_notices[name]
        }
        result = []
        for name in sorted(chosen):
            source = self.verify_source(self.root.joinpath(*name.split("/")))
            result.append((str(source), str(Path("models", *name.split("/")[:-1]))))
        return result
