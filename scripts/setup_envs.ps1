<#
.SYNOPSIS
    Create one isolated .venv per feature-extraction package (+ the TPOT benchmark)
    using uv.

.DESCRIPTION
    Each package and the benchmark get their own virtual environment under .venvs/.
    This isolation avoids dependency conflicts (e.g. kats requires Python 3.7 with
    old numpy/pandas, while the others use Python 3.12 with current versions).

    Prerequisite: uv  (https://docs.astral.sh/uv/). Install with:
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    Kats note: kats requires Python 3.7. uv will try to fetch a managed CPython 3.7.
    If that fails, install Python 3.7 on your system and run:
        uv venv .venvs/kats --python <path-to-python3.7>
        uv pip install -r requirements/kats.txt --python .venvs/kats

.PARAMETER Names
    Subset of environments to (re)create. Default: all.

.EXAMPLE
    ./scripts/setup_envs.ps1
    ./scripts/setup_envs.ps1 -Names tsfresh,benchmark
#>
param(
    [string[]]$Names = @("tsfresh", "tsfel", "pycatch22", "seglearn", "tsfeatures", "kats", "benchmark")
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/..").Path
$venvRoot = Join-Path $root ".venvs"
New-Item -ItemType Directory -Force -Path $venvRoot | Out-Null

# Resolve the uv executable: check PATH first, then common install locations
# (uv may be installed under a conda/mamba env's Scripts dir, ~/.local/bin,
# ~/.cargo/bin, or the standalone installer's default locations).
function Resolve-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:USERPROFILE\.cargo\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe",
        "$env:APPDATA\uv\uv.exe"
    )
    # Also scan every *conda*/*mamba* Scripts dir on the system drive.
    $candidates += @(
        Get-ChildItem -Path "$env:LOCALAPPDATA","$env:USERPROFILE","$env:ProgramData" -Filter "Scripts" -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '(conda|mamba|miniforge|anaconda|miniconda)' } |
            ForEach-Object { Join-Path $_.FullName "uv.exe" }
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    $msg = "uv not found on PATH or in common install locations. " +
           "Install it: powershell -ExecutionPolicy ByPass -c " +
           "'irm https://astral.sh/uv/install.ps1 | iex', " +
           "or add its directory to PATH."
    Write-Error $msg
    exit 1
}

$uv = Resolve-Uv
Write-Host "Using uv: $uv" -ForegroundColor DarkGray

$pythonVersions = @{
    "tsfresh"    = "3.12"
    "tsfel"      = "3.12"
    "pycatch22"  = "3.10"
    "seglearn"   = "3.12"
    "tsfeatures" = "3.12"
    "benchmark"  = "3.12"
    "kats"       = "3.7"
}

foreach ($name in $Names) {
    if (-not $pythonVersions.ContainsKey($name)) {
        Write-Error "Unknown environment name: '$name'. Valid: $($pythonVersions.Keys -join ', ')"
        exit 1
    }
    $py = $pythonVersions[$name]
    $venv = Join-Path $venvRoot $name
    $req = Join-Path $root "requirements/$name.txt"
    Write-Host ""
    Write-Host "==> [$name] creating venv (Python $py) at $venv" -ForegroundColor Cyan
    & $uv venv "$venv" --python "$py"
    Write-Host "==> [$name] installing dependencies from $req" -ForegroundColor Cyan
    & $uv pip install -r "$req" --python "$venv"
}

Write-Host ""
Write-Host "Done. Environments created under $venvRoot" -ForegroundColor Green
Write-Host "Activate a venv with:  .venvs\<name>\Scripts\Activate.ps1"
Write-Host "Or run a command in a venv with:  uv pip run --python .venvs\<name> python <script>"
