param(
    [string]$Url = 'http://127.0.0.1:5000',
    [string]$BrowserPath
)

$ErrorActionPreference = 'Stop'

function Get-PreferredBrowserPath {
    $candidates = @(
        'C:\Program Files\Google\Chrome\Application\chrome.exe',
        'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
        'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw 'No supported local browser was found. Install Chrome or Edge, or pass -BrowserPath explicitly.'
}

if (-not $BrowserPath) {
    $BrowserPath = Get-PreferredBrowserPath
}

if (-not (Test-Path $BrowserPath)) {
    throw "Browser executable not found: $BrowserPath"
}

$profileRoot = Join-Path ([System.IO.Path]::GetTempPath()) 'data_insighter_local_browser'
New-Item -ItemType Directory -Force -Path $profileRoot | Out-Null

$arguments = @(
    '--new-window',
    "--user-data-dir=$profileRoot",
    '--no-first-run',
    '--allow-insecure-localhost',
    '--disable-extensions',
    '--disable-component-extensions-with-background-pages',
    '--disable-background-networking',
    '--disable-default-apps',
    '--disable-sync',
    '--disable-popup-blocking',
    '--disable-features=Translate,OptimizationHints,AutofillServerCommunication,MediaRouter',
    $Url
)

$process = Start-Process -FilePath $BrowserPath -ArgumentList $arguments -PassThru

Write-Host "Opened local browser using:"
Write-Host "  $BrowserPath"
Write-Host "URL:"
Write-Host "  $Url"
Write-Host "Profile:"
Write-Host "  $profileRoot"
Write-Host "PID:"
Write-Host "  $($process.Id)"
