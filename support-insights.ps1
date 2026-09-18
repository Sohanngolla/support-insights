# support-insights.ps1
# Windows PowerShell equivalent of the ./support-insights bash script.
# Same logic: create/activate venv if missing, install deps if needed,
# build the database from the CSV if missing, then start the server.
#
# Usage: .\support-insights.ps1
# (If PowerShell blocks script execution, run once as Administrator:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# )

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found -- creating one (first run only)..."
    python -m venv .venv
}

$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

$depsMarker = ".venv\.deps_installed"
$needsInstall = $true
if (Test-Path $depsMarker) {
    $markerTime = (Get-Item $depsMarker).LastWriteTime
    $reqTime = (Get-Item "requirements.txt").LastWriteTime
    if ($markerTime -gt $reqTime) {
        $needsInstall = $false
    }
}

if ($needsInstall) {
    Write-Host "Installing dependencies..."
    & $venvPip install --quiet -r requirements.txt
    New-Item -Path $depsMarker -ItemType File -Force | Out-Null
}

if (-not (Test-Path "data\support.db")) {
    Write-Host "No database found -- running ingestion first..."
    & $venvPython -m app.ingest
}

$apiHost = if ($env:API_HOST) { $env:API_HOST } else { "127.0.0.1" }
$apiPort = if ($env:API_PORT) { $env:API_PORT } else { "8000" }

Write-Host "Starting Support Insights on http://${apiHost}:${apiPort}"
Write-Host "  UI:   http://${apiHost}:${apiPort}/"
Write-Host "  Docs: http://${apiHost}:${apiPort}/docs"

& $venvPython -m uvicorn app.main:app --host $apiHost --port $apiPort
