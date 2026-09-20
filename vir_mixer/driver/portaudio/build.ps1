param(
    [ValidateSet('baseline','fixed')][string]$Variant='fixed',
    [ValidateSet('Debug','Release')][string]$Configuration='Release',
    [string]$Python='python'
)
$ErrorActionPreference='Stop'
$driverRoot=Split-Path $PSScriptRoot
& $Python (Join-Path $PSScriptRoot 'prepare.py') --variant $Variant
if ($LASTEXITCODE) { throw 'PortAudio generation failed' }
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products '*' -property installationPath
$cmake = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
if (-not (Test-Path $cmake)) { throw 'Visual Studio CMake component missing' }
$source=Join-Path $driverRoot "out\portaudio-$Variant-src"
$build=Join-Path $driverRoot "out\portaudio-$Variant-build"
& $cmake -S $source -B $build -G 'Visual Studio 18 2026' -A x64 '-DCMAKE_POLICY_VERSION_MINIMUM=3.5' -DPA_USE_ASIO=OFF -DPA_BUILD_STATIC=OFF -DPA_BUILD_SHARED=ON -DPA_BUILD_TESTS=OFF -DPA_BUILD_EXAMPLES=OFF
if ($LASTEXITCODE) { throw 'CMake configuration failed' }
& $cmake --build $build --config $Configuration
if ($LASTEXITCODE) { throw 'PortAudio build failed' }
$dll = @(Get-ChildItem (Join-Path $build $Configuration) -Filter 'portaudio*.dll')
if ($dll.Count -ne 1) { throw 'Expected one PortAudio DLL' }
@{variant=$Variant; configuration=$Configuration; revision='147dd722548358763a8b649b3e4b41dfffbcfbb6'; dll=$dll[0].FullName; sha256=(Get-FileHash $dll[0].FullName).Hash} | ConvertTo-Json | Set-Content (Join-Path $build "manifest-$Configuration.json")
Write-Host "Built project-local DLL: $($dll[0].FullName). No installed Python DLL or driver changed."

