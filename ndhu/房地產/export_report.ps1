$ErrorActionPreference = "Stop"

$reportDir = Join-Path $PSScriptRoot "reports"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

python -m jupyter nbconvert `
    --to html `
    --execute (Join-Path $PSScriptRoot "main.ipynb") `
    --output-dir $reportDir `
    --output "hedonic_regression_report" `
    --no-input `
    --ExecutePreprocessor.timeout=120

Write-Host "Report exported to:" (Join-Path $reportDir "hedonic_regression_report.html")
