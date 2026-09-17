$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$mixerPython = Join-Path $env:USERPROFILE 'anaconda3/envs/xin/python.exe'
if (-not (Test-Path -LiteralPath $mixerPython)) { $mixerPython = 'python' }
& $mixerPython -X utf8 .\run_mixer.py
