$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# This project already used the xin conda environment on this computer.
$voicePython = Join-Path $env:USERPROFILE 'anaconda3/envs/xin/python.exe'
if (-not (Test-Path -LiteralPath $voicePython)) {
    $voicePython = 'python'
}
& $voicePython -X utf8 app.py
