<#
.SYNOPSIS
    Run the performance benchmark (TPOT and/or fixed-classifier baselines) across
    all datasets and seeds (full reproduction).

.DESCRIPTION
    Loops over the 5 published random seeds (run_1..run_5) and all 4 datasets,
    invoking performance_bechmark.py (TPOT) and/or
    scripts/fixed_classifier_evaluation.py (RF/XGBoost/LogReg/SVM) once per
    (seed, dataset) using the `benchmark` virtual environment. Each run writes to
    results/11_Performance_Benchmarks/run_<i>__random_seed_<seed>/<Dataset_Display_Name>/.

    Prerequisite: create the benchmark env first:
        ./scripts/setup_envs.ps1 -Names benchmark

    NOTE: each TPOT run can take up to --max-time-mins (default 120 = 2h). A full
    TPOT sweep is 4 datasets x 5 seeds x 2h = up to 40h. The fixed-classifier
    baselines are fast (seconds per package). Reduce --max-time-mins for quick
    sanity checks.

.PARAMETER Datasets
    Subset of datasets to run. Default: all. Valid keys: bosch_cnc,
    cnc_mill_tool_wear, condition_monitoring_of_hydraulic_systems, turning_dataset.

.PARAMETER Runs
    Run indices (1..5) to run. Default: all.

.PARAMETER MaxTimeMins
    TPOT total search budget per run in minutes. Default: 120 (2h).

.PARAMETER Baselines
    Also run the fixed-classifier baselines (RF, XGBoost, LogReg, SVM) for each
    (seed, dataset) in addition to TPOT.

.PARAMETER SkipTpot
    Skip the TPOT runs (use with -Baselines to run only the fixed classifiers).

.EXAMPLE
    ./scripts/run_all_performance.ps1
    ./scripts/run_all_performance.ps1 -Datasets bosch_cnc -Runs 1 -MaxTimeMins 5
    ./scripts/run_all_performance.ps1 -Baselines -SkipTpot      # baselines only
#>
param(
    [string[]]$Datasets = @("bosch_cnc", "cnc_mill_tool_wear", "condition_monitoring_of_hydraulic_systems", "turning_dataset"),
    [int[]]$Runs = @(1, 2, 3, 4, 5),
    [int]$MaxTimeMins = 120,
    [switch]$Baselines,
    [switch]$SkipTpot
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot/..").Path
$py = Join-Path $root ".venvs/benchmark/Scripts/python.exe"

if (-not (Test-Path $py)) {
    Write-Error "Benchmark venv not found at $py. Run: ./scripts/setup_envs.ps1 -Names benchmark"
    exit 1
}

# Seed per run index (matches the published run_1..run_5 layout).
$seeds = @{ 1 = 27; 2 = 44; 3 = 821; 4 = 1492; 5 = 7429 }

$tpotScript = Join-Path $root "performance_bechmark.py"
$baselineScript = Join-Path $root "scripts/fixed_classifier_evaluation.py"

foreach ($run in $Runs) {
    $seed = $seeds[$run]
    foreach ($ds in $Datasets) {
        Write-Host ""
        Write-Host "==> run $run (seed $seed) | dataset $ds" -ForegroundColor Cyan

        if (-not $SkipTpot) {
            Write-Host "  [TPOT]" -ForegroundColor Yellow
            & $py $tpotScript --dataset $ds --seed $seed --run $run --max-time-mins $MaxTimeMins
            if ($LASTEXITCODE -ne 0) {
                Write-Error "TPOT run failed: dataset=$ds run=$run seed=$seed (exit $LASTEXITCODE)"
                exit $LASTEXITCODE
            }
        }

        if ($Baselines) {
            Write-Host "  [fixed classifiers]" -ForegroundColor Yellow
            & $py $baselineScript --dataset $ds --seed $seed --run $run --n-jobs 20
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Baseline run failed: dataset=$ds run=$run seed=$seed (exit $LASTEXITCODE)"
                exit $LASTEXITCODE
            }
        }
    }
}

Write-Host ""
Write-Host "Done. Results under results/11_Performance_Benchmarks/run_*" -ForegroundColor Green
