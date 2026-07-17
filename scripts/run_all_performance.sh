#!/usr/bin/env bash
# Run the performance benchmark (TPOT and/or fixed-classifier baselines) across
# all datasets and seeds (full reproduction).
#
# Loops over the 5 published random seeds (run_1..run_5) and all 4 datasets,
# invoking performance_bechmark.py (TPOT) and/or
# scripts/fixed_classifier_evaluation.py (RF/XGBoost/LogReg/SVM) once per
# (seed, dataset) using the `benchmark` virtual environment.
#
# Prerequisite: create the benchmark env first:
#     ./scripts/setup_envs.sh -Names benchmark
#
# NOTE: each TPOT run can take up to --max-time-mins (default 120 = 2h). A full
# TPOT sweep is 4 datasets x 5 seeds x 2h = up to 40h. The fixed-classifier
# baselines are fast (seconds per package).
#
# Usage:
#   ./scripts/run_all_performance.sh
#   ./scripts/run_all_performance.sh --datasets bosch_cnc --runs 1 --max-time-mins 5
#   ./scripts/run_all_performance.sh --baselines --skip-tpot   # baselines only
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="$root/.venvs/benchmark/bin/python"

if [ ! -x "$py" ]; then
    echo "ERROR: benchmark venv not found at $py" >&2
    echo "Run: ./scripts/setup_envs.sh -Names benchmark" >&2
    exit 1
fi

# Defaults
datasets=(bosch_cnc cnc_mill_tool_wear condition_monitoring_of_hydraulic_systems turning_dataset)
runs=(1 2 3 4 5)
max_time_mins=120
baselines=false
skip_tpot=false

# Simple arg parsing
while [ $# -gt 0 ]; do
    case "$1" in
        --datasets) shift; IFS=',' read -ra datasets <<< "$1";;
        --runs) shift; IFS=',' read -ra runs <<< "$1";;
        --max-time-mins) shift; max_time_mins="$1";;
        --baselines) baselines=true;;
        --skip-tpot) skip_tpot=true;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
    shift
done

# Seed per run index (matches the published run_1..run_5 layout).
declare -A seeds=( [1]=27 [2]=44 [3]=821 [4]=1492 [5]=7429 )

tpot_script="$root/performance_bechmark.py"
baseline_script="$root/scripts/fixed_classifier_evaluation.py"

for run in "${runs[@]}"; do
    seed="${seeds[$run]}"
    for ds in "${datasets[@]}"; do
        echo ""
        echo "==> run $run (seed $seed) | dataset $ds"

        if [ "$skip_tpot" = false ]; then
            echo "  [TPOT]"
            "$py" "$tpot_script" --dataset "$ds" --seed "$seed" --run "$run" --max-time-mins "$max_time_mins"
        fi

        if [ "$baselines" = true ]; then
            echo "  [fixed classifiers]"
            "$py" "$baseline_script" --dataset "$ds" --seed "$seed" --run "$run" --n-jobs 20
        fi
    done
done

echo ""
echo "Done. Results under results/11_Performance_Benchmarks/run_*"
