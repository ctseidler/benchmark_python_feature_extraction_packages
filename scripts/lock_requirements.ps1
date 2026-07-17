<#
.SYNOPSIS
    Freeze per-environment requirements into fully pinned lock files using
    `uv pip compile` (no interpreter/venv required to resolve).

.DESCRIPTION
    Produces requirements/<name>.lock.txt with all transitive dependencies pinned
    to exact versions, resolved for the Python version each environment uses.
    Commit the lock files for full reproducibility.

.EXAMPLE
    ./scripts/lock_requirements.ps1
    ./scripts/lock_requirements.ps1 -Names tsfresh,benchmark
#>
param(
    [string[]]$Names = @("tsfresh", "tsfel", "pycatch22", "seglearn", "tsfeatures", "kats", "benchmark")
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/..").Path

# Resolve the uv executable: check PATH first, then common install locations.
function Resolve-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:USERPROFILE\.cargo\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe",
        "$env:APPDATA\uv\uv.exe"
    )
    $candidates += @(
        Get-ChildItem -Path "$env:LOCALAPPDATA","$env:USERPROFILE","$env:ProgramData" -Filter "Scripts" -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '(conda|mamba|miniforge|anaconda|miniconda)' } |
            ForEach-Object { Join-Path $_.FullName "uv.exe" }
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    Write-Error "uv not found. Install: powershell -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex'"
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
        Write-Error "Unknown environment name: '$name'."
        exit 1
    }
    $py = $pythonVersions[$name]
    $req = Join-Path $root "requirements/$name.txt"
    $out = Join-Path $root "requirements/$name.lock.txt"
    Write-Host "==> [$name] compiling lock for Python $py -> $out" -ForegroundColor Cyan
    & $uv pip compile --python-version "$py" "$req" -o "$out"
}

Write-Host ""
Write-Host "Done. Lock files written to requirements/*.lock.txt" -ForegroundColor Green
