# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brady Balk
"""Add payload verification to the reviewed electron-builder NSIS include."""

import argparse
import hashlib
from pathlib import Path


UPSTREAM_SHA256 = "0e319437dd01dcbf911f3f48f664fde0cefbaef704f1cdb1739f63d563f5d4a0"
SECTION_SHA256 = "74c4dbb4163cdd90ca4850b75d304e916235c882d6af7b5cdad63bffbc977072"
ANCHOR = "      fun_extract:\n"
CHECK = r'''
        # Recall: verify every payload before extraction, including downloads
        # and explicit --package-file paths. This installer supports x64 only.
        !ifndef APP_64_HASH
          !error "Recall requires an x64 package hash"
        !endif
        !ifdef APP_32_HASH
          !error "Review Recall integrity handling before adding architectures"
        !endif
        !ifdef APP_ARM64_HASH
          !error "Review Recall integrity handling before adding architectures"
        !endif
        ClearErrors
        ${StdUtils.HashFile} $3 "SHA2-512" "$packageFile"
        IfErrors recall_package_invalid
        StrCmp $3 "${APP_64_HASH}" recall_package_verified
        recall_package_invalid:
          MessageBox MB_OK|MB_ICONSTOP "The application download is incomplete or has changed. Please download the installer and package again." /SD IDOK
          SetErrorLevel 2
          Quit
        recall_package_verified:
'''


def prepare(desktop: Path, output: Path) -> None:
    upstream = desktop / "node_modules/app-builder-lib/templates/nsis/include/installer.nsh"
    content = upstream.read_bytes()
    if hashlib.sha256(content).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("NSIS template changed; review payload verification before building")
    text = content.decode("utf-8").replace("\r\n", "\n")
    if text.count(ANCHOR) != 1:
        raise ValueError("Expected exactly one pre-extraction insertion point")
    section_path = desktop / "node_modules/app-builder-lib/templates/nsis/installSection.nsh"
    section_bytes = section_path.read_bytes()
    if hashlib.sha256(section_bytes).hexdigest() != SECTION_SHA256:
        raise ValueError("NSIS install section changed; review update ordering before building")
    section = section_bytes.decode("utf-8").replace("\r\n", "\n")
    remove_anchor = "!insertmacro uninstallOldVersion SHELL_CONTEXT\n"
    if section.count(remove_anchor) != 1:
        raise ValueError("Expected exactly one initial uninstall operation")
    preparation_start = text.index("      Var /GLOBAL packageFile\n")
    preparation_end = text.index(ANCHOR) + len(ANCHOR)
    preparation = text[preparation_start:preparation_end].replace("fun_extract", "recall_prepare_extract")
    preparation += CHECK.replace("recall_package_", "recall_prepare_")
    # Keep the checked package outside the installation that will be removed.
    # This also preserves an explicitly supplied archive for subsequent use.
    preparation += r'''
      StrCmp $packageFile "$PLUGINSDIR\package.7z" recall_prepare_staged
      ClearErrors
      CopyFiles /SILENT "$packageFile" "$PLUGINSDIR\package.7z"
      IfErrors recall_prepare_copy_failed
      StrCpy $packageFile "$PLUGINSDIR\package.7z"
      ClearErrors
      ${StdUtils.HashFile} $3 "SHA2-512" "$packageFile"
      IfErrors recall_prepare_copy_failed
      StrCmp $3 "${APP_64_HASH}" recall_prepare_staged recall_prepare_copy_failed
      recall_prepare_copy_failed:
        MessageBox MB_OK|MB_ICONSTOP "The application package could not be prepared. Your existing installation has not been replaced. Check available disk space and try again." /SD IDOK
        SetErrorLevel 2
        Quit
      recall_prepare_staged:
      !define RECALL_PREPARE_BEFORE_UNINSTALL 1
'''
    prepare_macro = "\n!macro recallPrepareApplicationPackage\n" + preparation + "\n!macroend\n"
    replacement = '''      !ifndef RECALL_PREPARE_BEFORE_UNINSTALL
        !error "Recall must verify the application package before removing an existing installation"
      !endif
''' + CHECK
    text = text[:preparation_start] + replacement + text[preparation_end:]
    text = text.replace("!macro installApplicationFiles\n", prepare_macro + "\n!macro installApplicationFiles\n", 1)
    section = section.replace(remove_anchor, '''!ifdef APP_PACKAGE_URL
  !ifndef APP_BUILD_DIR
    SetOutPath "$PLUGINSDIR"
    !insertmacro recallPrepareApplicationPackage
  !endif
!endif

''' + remove_anchor, 1)
    license_bytes = (desktop / "node_modules/electron-builder/LICENSE").read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    patched = "; Derived from electron-builder 26.15.3; see LICENSE.electron-builder.\n"
    patched += "!define RECALL_PAYLOAD_INTEGRITY 1\n"
    patched += text
    patched += '\n!cd "${BUILD_RESOURCES_DIR}/../node_modules/app-builder-lib/templates/nsis"\n'
    (output / "installer.nsh").write_text(patched, encoding="utf-8")
    (output / "installSection.nsh").write_text(section, encoding="utf-8")
    (output / "LICENSE.electron-builder").write_bytes(license_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.desktop, args.out)
