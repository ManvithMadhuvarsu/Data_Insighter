$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$stderrLog = Join-Path $repo 'output\flask_stderr.log'
$stdoutLog = Join-Path $repo 'output\flask_stdout.log'

Write-Host "Tailing Flask logs..."
Write-Host "stderr: $stderrLog"
Write-Host "stdout: $stdoutLog"
Write-Host ''
Write-Host 'Press Ctrl+C to stop tailing.'
Write-Host ''

if (-not (Test-Path $stderrLog)) {
    New-Item -ItemType File -Path $stderrLog -Force | Out-Null
}

if (-not (Test-Path $stdoutLog)) {
    New-Item -ItemType File -Path $stdoutLog -Force | Out-Null
}

Get-Content -Path $stderrLog, $stdoutLog -Wait -Tail 40
