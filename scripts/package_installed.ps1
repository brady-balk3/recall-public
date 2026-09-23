<# Build a web installer plus its external payload; never upload automatically. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$PackageUrl,
    [string]$ModelManifest,
    [string]$DependencyNoticeManifest,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DesktopDir = Join-Path $RepoRoot 'apps\desktop'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$outputRelative = "release-installed/$stamp"
$savedUrl = $env:RECALL_INSTALLER_PACKAGE_URL
$savedManifest = $env:RECALL_MODEL_MANIFEST
$savedNotices = $env:RECALL_DEPENDENCY_NOTICE_MANIFEST
$savedOutput = $env:RECALL_INSTALLER_OUTPUT
$savedEngine = $env:RECALL_INSTALLER_ENGINE_DIR
$savedIntegrity = $env:RECALL_NSIS_INTEGRITY_DIR
$savedDownloadModels = $env:RECALL_INSTALLER_DOWNLOAD_MODELS
try {
    $env:RECALL_INSTALLER_PACKAGE_URL = $PackageUrl
    $env:RECALL_INSTALLER_OUTPUT = $outputRelative
    $env:RECALL_INSTALLER_ENGINE_DIR = "../api/dist-installed/$stamp/recall-engine"
    $env:RECALL_INSTALLER_DOWNLOAD_MODELS = '1'
    Push-Location $DesktopDir
    try {
        & node -e "const config = require('./electron-builder.installed.cjs'); require('app-builder-lib/out/util/config/config').validateConfiguration(config, {isEnabled:false}).catch(error => {console.error(error.message); process.exitCode = 1;});"
        if ($LASTEXITCODE -ne 0) { throw 'Invalid installer configuration.' }
    } finally { Pop-Location }
    if ($ValidateOnly) {
        Write-Output 'Installer URL/configuration is valid. No engine build, downloads or publication performed.'
        return
    }
    if (-not $ModelManifest) {
        $ModelManifest = Join-Path $RepoRoot 'publish\model-payload-release.json'
    }
    if (-not (Test-Path -LiteralPath $ModelManifest -PathType Leaf)) {
        throw 'Reviewed model payload manifest is missing.'
    }
    $env:RECALL_MODEL_MANIFEST = (Resolve-Path -LiteralPath $ModelManifest).Path
    if (-not $DependencyNoticeManifest) {
        $DependencyNoticeManifest = Join-Path $RepoRoot 'publish\dependency-notice-release.json'
    }
    if (-not (Test-Path -LiteralPath $DependencyNoticeManifest -PathType Leaf)) {
        throw 'Reviewed dependency notice manifest is missing.'
    }
    $env:RECALL_DEPENDENCY_NOTICE_MANIFEST = (Resolve-Path -LiteralPath $DependencyNoticeManifest).Path
    $env:RECALL_NSIS_INTEGRITY_DIR = Join-Path $DesktopDir "build-generated/$stamp/nsis-integrity"
    & (Join-Path $RepoRoot 'venv\Scripts\python.exe') `
        (Join-Path $RepoRoot 'publish\prepare_installer_integrity.py') `
        --desktop $DesktopDir --out $env:RECALL_NSIS_INTEGRITY_DIR
    if ($LASTEXITCODE -ne 0) { throw 'Installer payload integrity preparation failed.' }
    Push-Location $RepoRoot
    try {
        & (Join-Path $RepoRoot 'venv\Scripts\python.exe') -m publish.prepare_microsoft_runtime
        if ($LASTEXITCODE -ne 0) { throw 'Pinned Microsoft runtime preparation failed.' }
        & (Join-Path $RepoRoot 'venv\Scripts\python.exe') (Join-Path $RepoRoot 'scripts\fetch_twitchdownloader.py')
        if ($LASTEXITCODE -ne 0) { throw 'Pinned TwitchDownloader CLI preparation failed.' }
        & (Join-Path $RepoRoot 'venv\Scripts\python.exe') (Join-Path $RepoRoot 'scripts\fetch_small_release_assets.py')
        if ($LASTEXITCODE -ne 0) { throw 'Pinned small model preparation failed.' }
        & (Join-Path $RepoRoot 'venv\Scripts\python.exe') -m PyInstaller `
            'apps/api/recall-engine.spec' --noconfirm `
            --distpath "apps/api/dist-installed/$stamp" --workpath "apps/api/build-installed/$stamp"
        if ($LASTEXITCODE -ne 0) { throw 'Installed engine build failed.' }
    } finally { Pop-Location }
    Push-Location $DesktopDir
    try {
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'Desktop compilation failed.' }
        & node (Join-Path $DesktopDir 'build\build-installed.cjs')
        if ($LASTEXITCODE -ne 0) { throw 'Web installer build failed.' }
    } finally { Pop-Location }
    $output = Join-Path $DesktopDir $outputRelative
    & (Join-Path $RepoRoot 'venv\Scripts\python.exe') `
        (Join-Path $RepoRoot 'publish\verify_release_privacy.py') `
        --root (Join-Path $output 'win-unpacked')
    if ($LASTEXITCODE -ne 0) { throw 'Installed payload privacy check failed; do not distribute these artifacts.' }
    $assetsDir = Join-Path $output 'nsis-web'
    $assets = @(Get-ChildItem -LiteralPath $assetsDir -File | Where-Object { $_.Extension -eq '.exe' -or $_.Name -match '\.7z\.part[0-9]{3}$' })
    if (-not ($assets | Where-Object Extension -eq '.exe') -or -not ($assets | Where-Object Name -Match '\.7z\.part[0-9]{3}$')) {
        throw 'Expected both an installer and an external application payload.'
    }
    if ($assets | Where-Object Length -GE 2147483648) {
        throw 'A release asset exceeds the GitHub per-file limit.'
    }
    $inventory = @($assets | ForEach-Object {
        [ordered]@{ file = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    [ordered]@{ payloadUrl = $PackageUrl; assets = $inventory } | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $assetsDir 'upload-plan.json') -Encoding UTF8
    Write-Output "Built installer assets in $assetsDir"
    Write-Output 'Nothing uploaded. Upload the installer and .7z.partNNN assets to the configured GitHub release. The original .7z is local build output only.'
} finally {
    $env:RECALL_INSTALLER_PACKAGE_URL = $savedUrl
    $env:RECALL_MODEL_MANIFEST = $savedManifest
    $env:RECALL_DEPENDENCY_NOTICE_MANIFEST = $savedNotices
    $env:RECALL_INSTALLER_OUTPUT = $savedOutput
    $env:RECALL_INSTALLER_ENGINE_DIR = $savedEngine
    $env:RECALL_NSIS_INTEGRITY_DIR = $savedIntegrity
    $env:RECALL_INSTALLER_DOWNLOAD_MODELS = $savedDownloadModels
}
