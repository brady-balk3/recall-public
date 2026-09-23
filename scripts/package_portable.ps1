<#
.SYNOPSIS
Build the Recall portable distribution: engine bundle -> Electron app dir -> zip.

.DESCRIPTION
This script retains the portable ZIP path. Installed builds have a separate
scripts/package_installed.ps1 entry point using nsis-web and an externally
hosted application payload. The large engine/model payload is never embedded
into a single-file NSIS installer by this portable pipeline.

.PARAMETER SkipEngine
Reuse the existing apps/api/dist/recall-engine instead of re-running PyInstaller.
The engine build is the slow half; skip it when only the frontend changed.

.PARAMETER SkipElectron
Reuse the existing release/win-unpacked and only rebuild the zip. Useful when
the payload is already correct and only the archive step needs redoing.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\package_portable.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipEngine,
    [switch]$SkipElectron
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DesktopDir = Join-Path $RepoRoot 'apps\desktop'
$BuildStamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$OutputRelative = if ($SkipElectron) { 'release' } else { "release/portable-$BuildStamp" }
$UnpackedDir = Join-Path (Join-Path $DesktopDir $OutputRelative) 'win-unpacked'
$EngineDir = if ($SkipEngine -or $SkipElectron) {
    Join-Path $RepoRoot 'apps\api\dist\recall-engine'
} else {
    Join-Path $RepoRoot "apps/api/dist-portable/$BuildStamp/recall-engine"
}

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

# --- 1. Engine (PyInstaller) ------------------------------------------------
if ($SkipElectron) {
    Write-Step 'Archive-only reuse: skipping engine and Electron builds'
} elseif ($SkipEngine) {
    Write-Step "Skipping engine build (-SkipEngine)"
    if (-not (Test-Path (Join-Path $EngineDir 'recall-engine.exe'))) {
        throw "No existing engine bundle to reuse. Re-run without -SkipEngine."
    }
} else {
    Write-Step "Building engine bundle (PyInstaller)"
    Push-Location $RepoRoot
    try {
        & (Join-Path $RepoRoot 'venv\Scripts\python.exe') -m publish.prepare_microsoft_runtime
        if ($LASTEXITCODE -ne 0) { throw 'Pinned Microsoft runtime preparation failed.' }
        & (Join-Path $RepoRoot 'venv\Scripts\python.exe') -m PyInstaller `
            'apps/api/recall-engine.spec' --noconfirm `
            --distpath "apps/api/dist-portable/$BuildStamp" --workpath "apps/api/build-portable/$BuildStamp"
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
}

# Refuse a reused engine containing state from a previous app launch.
if (-not $SkipElectron) {
    & (Join-Path $RepoRoot 'venv\Scripts\python.exe') `
    (Join-Path $RepoRoot 'publish\verify_release_privacy.py') `
        --root $EngineDir
    if ($LASTEXITCODE -ne 0) { throw 'Engine release privacy check failed.' }
}

# --- 2. Electron app directory ---------------------------------------------
# A launched portable folder can contain scan data. Build into a fresh directory
# instead of deleting previous outputs to make electron-builder's rename work.
if ($SkipElectron) {
    Write-Step "Skipping Electron build (-SkipElectron)"
} else {
    # `--win dir` only: see the makensis note above.
    Write-Step "Building Electron app directory"
    $savedPortableOutput = $env:RECALL_PORTABLE_OUTPUT
    $savedPortableEngine = $env:RECALL_PORTABLE_ENGINE_DIR
    $env:RECALL_PORTABLE_OUTPUT = $OutputRelative
    $env:RECALL_PORTABLE_ENGINE_DIR = $EngineDir
    Push-Location $DesktopDir
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Desktop compilation failed with exit code $LASTEXITCODE" }
        & (Join-Path $DesktopDir 'node_modules\.bin\electron-builder.cmd') `
            --config electron-builder.portable.cjs --win dir --x64 --publish never
        if ($LASTEXITCODE -ne 0) { throw "electron-builder failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
        $env:RECALL_PORTABLE_OUTPUT = $savedPortableOutput
        $env:RECALL_PORTABLE_ENGINE_DIR = $savedPortableEngine
    }
}

if (-not (Test-Path $UnpackedDir)) {
    throw "Expected $UnpackedDir after electron-builder --win dir."
}

# --- 3. Zip -----------------------------------------------------------------
$version = (Get-Content (Join-Path $DesktopDir 'package.json') -Raw | ConvertFrom-Json).version
$zipPath = Join-Path (Join-Path $DesktopDir $OutputRelative) "Recall-Windows-x64-$version.zip"

$payloadBytes = (Get-ChildItem $UnpackedDir -Recurse -File | Measure-Object Length -Sum).Sum
Write-Step ("Compressing {0:N2} GB -> {1}" -f ($payloadBytes / 1GB), (Split-Path $zipPath -Leaf))

# Build beside the destination and publish only after the ZIP closes successfully.
# The helper uses Zip64 and POSIX entry names, and preserves the previous release
# on compression errors, changed inputs, or a locked destination.
& (Join-Path $RepoRoot 'venv\Scripts\python.exe') `
    (Join-Path $RepoRoot 'scripts\build_portable_archive.py') `
    --source $UnpackedDir --out $zipPath
if ($LASTEXITCODE -ne 0) { throw "Portable archive failed with exit code $LASTEXITCODE" }

$zipBytes = (Get-Item $zipPath).Length
Write-Host ""
Write-Host ("Done: {0}" -f $zipPath) -ForegroundColor Green
Write-Host ("  payload {0:N2} GB -> zip {1:N2} GB" -f ($payloadBytes / 1GB), ($zipBytes / 1GB))
Write-Host ""
Write-Host "Users need roughly twice the zip size free to extract." -ForegroundColor Yellow
Write-Host "Recall.exe is at the root of the zip; data/ is created next to it on first run."
