param(
    [ValidateSet('Debug','Release')][string]$Configuration = 'Debug'
)
$ErrorActionPreference = 'Stop'
$driverRoot = $PSScriptRoot
$vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswherePath)) {
    throw 'Missing Visual Studio C++ build tools. Install a WDK-compatible Visual Studio C++ version and matching Windows SDK/WDK first. No driver was built or installed.'
}
$vsInstall = & $vswherePath -latest -products '*' -requires Microsoft.Component.MSBuild -property installationPath
$buildTool = if ($vsInstall) { Join-Path $vsInstall 'MSBuild\Current\Bin\amd64\MSBuild.exe' }
if (-not $buildTool -or -not (Test-Path -LiteralPath $buildTool)) { throw 'x64 MSBuild not found.' }
$kitsRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10'
$wdkHeader = Get-ChildItem -LiteralPath (Join-Path $kitsRoot 'Include') -Filter ntddk.h -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $wdkHeader) { throw 'Windows Driver Kit headers were not found. Install matching SDK/WDK and its Visual Studio extension.' }
$sourceRoot = Join-Path $driverRoot 'build-source\audio\sysvad'
if (-not (Test-Path -LiteralPath $sourceRoot)) { throw 'Run python driver/prepare.py first.' }
$buildArgs = @('/m', '/p:Platform=x64', '/p:PreferredToolArchitecture=x64', "/p:Configuration=$Configuration", '/p:SignMode=Off', "/p:IntDir=x64\$Configuration\")
foreach ($project in @('EndpointsCommon\EndpointsCommon.vcxproj', 'TabletAudioSample\TabletAudioSample.vcxproj')) {
    & $buildTool (Join-Path $sourceRoot $project) @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "WDK build failed: $project (exit $LASTEXITCODE). No driver was installed." }
}
$binary = Join-Path $sourceRoot "TabletAudioSample\x64\$Configuration\VirMixerAudio.sys"
if (-not (Test-Path -LiteralPath $binary)) { throw 'Build returned success but expected .sys was not produced.' }
Write-Host "Built unsigned development driver: $binary"
Write-Host 'Next: WDK INF validation, catalog creation, signing and isolated target-machine tests. No system settings were changed.'
