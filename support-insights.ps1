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

# Windows sometimes has a "python" that isn't real Python at all -- it's a
# Microsoft Store shortcut stub that prints an install prompt instead of
# actually running anything, and does this SILENTLY when called from a
# script (no error, just does nothing useful). The `py` launcher that ships
# with the official python.org installer is not affected by this and is
# the more reliable choice when both exist.
function Get-PythonCommand {
    foreach ($candidate in @("py", "python", "python3")) {
        try {
            $version = & $candidate --version 2>&1
            if ($LASTEXITCODE -eq 0 -and $version -match "Python \d") {
                return $candidate
            }
        } catch {
            continue
        }
    }
    return $null
}

$pythonCmd = Get-PythonCommand
if (-not $pythonCmd) {
    Write-Host ""
    Write-Host "ERROR: No working Python installation found." -ForegroundColor Red
    Write-Host "Install Python from https://python.org/downloads (NOT the Microsoft Store version)."
    Write-Host "During install, check the box: 'Add python.exe to PATH'."
    Write-Host "Then close and reopen PowerShell and run this script again."
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "No .venv found -- creating one (first run only)..."
    & $pythonCmd -m venv .venv
}

$venvPython = ".\.venv\Scripts\python.exe"
$venvPip = ".\.venv\Scripts\pip.exe"

# Verify the venv actually got created with real files in it -- if the
# Python that ran above was the fake Store stub, `-m venv` exits 0 but
# creates nothing usable, and every step after this would fail with a
# confusing "pip.exe not recognized" error instead of a clear one.
if (-not (Test-Path $venvPython)) {
    Write-Host ""
    Write-Host "ERROR: .venv exists but has no working Python inside it." -ForegroundColor Red
    Write-Host "This usually means the 'python' command that created it was the Windows"
    Write-Host "Store stub, not real Python. Fix:"
    Write-Host "  1. Delete the broken venv:  Remove-Item -Recurse -Force .venv"
    Write-Host "  2. Install real Python from https://python.org/downloads"
    Write-Host "     (check 'Add python.exe to PATH' during install)"
    Write-Host "  3. Run this script again."
    exit 1
}

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
