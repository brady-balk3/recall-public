; SPDX-License-Identifier: AGPL-3.0-or-later
; Copyright (C) 2026 Brady Balk
!if "$%RECALL_NSIS_INTEGRITY_DIR%" == ""
  !error "Prepare the reviewed Recall NSIS integrity include before building"
!endif
; NSIS searches its working directory before added include directories.
; Preserve upstream template lookup, then select the reviewed installer override.
!addincludedir "${BUILD_RESOURCES_DIR}/../node_modules/app-builder-lib/templates/nsis"
!macro customInit
  !cd "$%RECALL_NSIS_INTEGRITY_DIR%"
  !ifndef BUILD_UNINSTALLER
    ; Validate an explicitly supplied package before the update path can remove
    ; the installed application. Keep the extraction-time check as well.
    ${StdUtils.GetParameter} $0 "package-file" ""
    StrCmp $0 "" recall_local_package_checked
    IfFileExists "$0" 0 recall_local_package_invalid
    ClearErrors
    ${StdUtils.HashFile} $3 "SHA2-512" "$0"
    IfErrors recall_local_package_invalid
    StrCmp $3 "${APP_64_HASH}" recall_local_package_checked
    recall_local_package_invalid:
      MessageBox MB_OK|MB_ICONSTOP "The supplied application package is missing or has changed. Your existing installation has not been replaced. Please download the package again." /SD IDOK
      SetErrorLevel 2
      Quit
    recall_local_package_checked:
  !endif
!macroend
!macro customInstall
  !ifndef RECALL_PAYLOAD_INTEGRITY
    !error "Recall payload integrity override was not selected"
  !endif
  !ifndef RECALL_CHUNK_TRANSPORT
    !error "Build through build-installed.cjs to prepare GitHub release chunks"
  !endif
  recall_install_models_retry:
    DetailPrint "Downloading and verifying Recall models. This may take several minutes."
    nsExec::ExecToLog '"$INSTDIR\resources\recall-engine\recall-engine.exe" --install-models'
    Pop $0
    StrCmp $0 "0" recall_install_models_done
    IfSilent recall_install_models_cancel
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "Recall could not finish downloading its models. Check your connection and available disk space, then retry. Verified files will be reused." IDRETRY recall_install_models_retry
    recall_install_models_cancel:
      SetErrorLevel 2
      Quit
    recall_install_models_done:
!macroend
