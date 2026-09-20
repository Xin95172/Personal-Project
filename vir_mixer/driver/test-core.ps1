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

$traceSource = Join-Path $PSScriptRoot 'tests\cable_trace_test.cpp'
$traceOutput = Join-Path $PSScriptRoot 'out\cable_trace_test.exe'
$traceStubs = Join-Path $PSScriptRoot 'tests\kernel_stubs'
if ($Zig) {
    & $Zig c++ -std=c++17 -Wall -Wextra -Werror "-I$traceStubs" $traceSource -o $traceOutput
} else {
    $traceObject = Join-Path $PSScriptRoot 'out\cable_trace_test.obj'
    & cl.exe /nologo /std:c++17 /EHsc /W4 /WX /D_CRT_SECURE_NO_WARNINGS "/I$traceStubs" $traceSource "/Fe:$traceOutput" "/Fo:$traceObject"
}
if ($LASTEXITCODE -ne 0) { throw 'Diagnostic wrapper test compilation failed.' }
& $traceOutput (Join-Path $PSScriptRoot 'out\stream-trace-synthetic.log')
if ($LASTEXITCODE -ne 0) { throw 'Diagnostic wrapper tests failed.' }

$timelineSource = Join-Path $PSScriptRoot 'tests\cable_timeline_test.cpp'
$timelineOutput = Join-Path $PSScriptRoot 'out\cable_timeline_test.exe'
if ($Zig) {
    & $Zig c++ -std=c++17 -Wall -Wextra -Werror $timelineSource -o $timelineOutput
} else {
    & cl.exe /nologo /std:c++17 /EHsc /W4 /WX $timelineSource "/Fe:$timelineOutput" "/Fo:$timelineOutput.obj"
}
if ($LASTEXITCODE) { throw 'Timeline compilation failed.' }
& $timelineOutput
if ($LASTEXITCODE) { throw 'Timeline tests failed.' }

$kernelTimelineSource = Join-Path $PSScriptRoot 'tests\cable_timeline_kernel_test.cpp'
$kernelTimelineOutput = Join-Path $PSScriptRoot 'out\cable_timeline_kernel_test.exe'
if ($Zig) {
    & $Zig c++ -std=c++17 -Wall -Wextra -Werror "-I$traceStubs" $kernelTimelineSource -o $kernelTimelineOutput
} else {
    & cl.exe /nologo /std:c++17 /EHsc /W4 /WX /D_CRT_SECURE_NO_WARNINGS "/I$traceStubs" $kernelTimelineSource "/Fe:$kernelTimelineOutput" "/Fo:$kernelTimelineOutput.obj"
}
if ($LASTEXITCODE) { throw 'B wrapper compilation failed.' }
& $kernelTimelineOutput
if ($LASTEXITCODE) { throw 'B wrapper tests failed.' }

$probeSource = Join-Path $PSScriptRoot 'tests\frame_probe_test.cpp'
$probeOutput = Join-Path $PSScriptRoot 'out\frame_probe_test.exe'
if ($Zig) {
    & $Zig c++ -std=c++17 -Wall -Wextra -Werror "-I$traceStubs" $probeSource -o $probeOutput
} else {
    & cl.exe /nologo /std:c++17 /EHsc /W4 /WX /D_CRT_SECURE_NO_WARNINGS "/I$traceStubs" $probeSource "/Fe:$probeOutput" "/Fo:$probeOutput.obj"
}
if ($LASTEXITCODE) { throw 'Frame probe compilation failed.' }
& $probeOutput
if ($LASTEXITCODE) { throw 'Frame probe tests failed.' }
