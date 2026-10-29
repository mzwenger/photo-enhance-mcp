# Windows: instala uv si falta y ejecuta el instalador (install.py).
# Uso (PowerShell):  powershell -ExecutionPolicy Bypass -File setup.ps1 [--no-models]
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Instalando uv (gestor de Python de Astral)..."
    powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv run --no-project --python 3.12 install.py @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
