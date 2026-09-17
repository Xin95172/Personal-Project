param([string]$Zig)
$ErrorActionPreference = 'Stop'
$testSource = Join-Path $PSScriptRoot 'tests\ring_test.cpp'
$testOutput = Join-Path $PSScriptRoot 'out\ring_test.exe'
New-Item -ItemType Directory -Force (Split-Path $testOutput) | Out-Null
if ($Zig) {
    & $Zig c++ -std=c++17 -Wall -Wextra -Werror $testSource -o $testOutput
} else {
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'Run from an x64 Native Tools PowerShell, or supply -Zig with zig.exe path.' }
    $testObject = Join-Path $PSScriptRoot 'out\ring_test.obj'
    & cl.exe /nologo /std:c++17 /EHsc /W4 /WX $testSource "/Fe:$testOutput" "/Fo:$testObject"
}
if ($LASTEXITCODE -ne 0) { throw 'Native ring test compilation failed.' }
& $testOutput
if ($LASTEXITCODE -ne 0) { throw 'Native ring tests failed.' }
