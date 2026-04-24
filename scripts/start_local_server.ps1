$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\mscma\AppData\Local\Programs\Python\Python310\python.exe'
$outputDir = Join-Path $repo 'output'
$stdoutLog = Join-Path $outputDir 'flask_stdout.log'
$stderrLog = Join-Path $outputDir 'flask_stderr.log'

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -like 'python*.exe' -and
        $_.CommandLine -like '*app.py*'
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

Start-Sleep -Seconds 2

if (Test-Path $stdoutLog) {
    Remove-Item -LiteralPath $stdoutLog -Force -ErrorAction SilentlyContinue
}

if (Test-Path $stderrLog) {
    Remove-Item -LiteralPath $stderrLog -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType File -Path $stdoutLog -Force | Out-Null
New-Item -ItemType File -Path $stderrLog -Force | Out-Null

$process = Start-Process `
    -FilePath $python `
    -ArgumentList '-u', 'app.py' `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Host "Started Flask server with PID $($process.Id)"
Write-Host "Open http://127.0.0.1:5000"
Write-Host "Readable request/error log: $stderrLog"
