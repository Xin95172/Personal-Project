param(
    [ValidateSet('Debug','Release')][string]$Configuration = 'Debug',
    [string]$Python = 'python',
    [switch]$SkipBuild,
    [switch]$SharedTimeline,
    [switch]$FrameProbe
)
$ErrorActionPreference = 'Stop'
if ($FrameProbe -and (-not $SharedTimeline -or $Configuration -ne 'Debug')) { throw 'FrameProbe requires Debug SharedTimeline.' }
$kit = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10'
$version = '10.0.28000.0'
if (-not $SkipBuild) {
    $prepareArgs = @((Join-Path $PSScriptRoot 'prepare.py'))
    if ($SharedTimeline) { $prepareArgs += '--shared-timeline' }
    if ($FrameProbe) { $prepareArgs += '--frame-probe' }
    & $Python @prepareArgs
    if ($LASTEXITCODE) { throw 'prepare failed' }
    & (Join-Path $PSScriptRoot 'build.ps1') -Configuration $Configuration
}
$source = Join-Path $PSScriptRoot "build-source\audio\sysvad\TabletAudioSample\x64\$Configuration"
$mode = if ($SharedTimeline) { 'B' } else { 'A' }
$generatedMode = Get-Content (Join-Path $PSScriptRoot 'build-source\.virmixer-mode.json') | ConvertFrom-Json
if ($generatedMode.mode -ne $mode) { throw 'Requested package mode differs from generated source.' }
if ([bool]$generatedMode.frameProbe -ne [bool]$FrameProbe) { throw 'Requested frame probe differs from generated source.' }
# A fresh directory prevents stale sample binaries/catalogs entering a package.
$package = Join-Path $PSScriptRoot ('out\packages\' + $Configuration + '-' + $mode + '-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $package | Out-Null
foreach ($file in 'VirMixerAudio.sys','VirMixerAudio.inf') {
    Copy-Item -LiteralPath (Join-Path $source $file) -Destination $package
}
& "$kit\Tools\$version\x64\infverif.exe" /v /u "$package\VirMixerAudio.inf"
if ($LASTEXITCODE) { throw 'InfVerif failed' }
& "$kit\bin\$version\x86\Inf2Cat.exe" "/driver:$package" /os:10_CO_X64,10_NI_X64,10_GE_X64 /verbose
if ($LASTEXITCODE) { throw 'Inf2Cat failed' }
if (-not (Test-Path "$package\VirMixerAudio.cat")) { throw 'Catalog missing' }
$hashes = @{}
Get-ChildItem -LiteralPath $package -File | ForEach-Object { $hashes[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
$sourceManifestHash = (Get-FileHash (Join-Path $PSScriptRoot 'build-source\.virmixer-generated.json') -Algorithm SHA256).Hash
@{ configuration=$Configuration; mode=$mode; sharedTimeline=[bool]$SharedTimeline; frameProbe=[bool]$FrameProbe; sourceManifestSha256=$sourceManifestHash; kit=$version; files=$hashes; signed=$false } | ConvertTo-Json -Depth 4 | Set-Content "$package\manifest.json"
& $Python (Join-Path $PSScriptRoot 'tests\verify_package.py') $package
if ($LASTEXITCODE) { throw 'Offline package consistency verification failed' }
Set-Content (Join-Path $PSScriptRoot "out\latest-package-$Configuration.txt") $package
Set-Content (Join-Path $PSScriptRoot "out\latest-package-$Configuration-$mode.txt") $package
Write-Host "Validated unsigned package: $package"
