# Reproduction Guide

This guide reproduces the full benchmark: feature extraction across all packages,
the TPOT performance evaluation (5 seeds × 4 datasets), and the statistical
comparison / ranking of the feature-extraction packages.

> **Compute warning:** a full reproduction is expensive. Feature extraction for
> `tsfresh` (comprehensive) on the larger datasets can take several hours, and the
> TPOT performance sweep uses up to a 2 h budget per (dataset, seed) — i.e. up to
> ~40 h for the full 4 × 5 grid. Use `--max-time-mins` to run quick sanity checks.

---

## 1. Prerequisites

### 1.1 Install uv

This project uses [uv](https://docs.astral.sh/uv/) to manage **one isolated virtual
environment per feature-extraction package** (plus one for the TPOT benchmark). This
is required because the packages have conflicting dependencies — most notably
**Kats, which requires Python 3.7**, while all other packages run on **Python 3.12**
(`pycatch22` uses **Python 3.10**, the only version with prebuilt wheels on Windows).

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 Create the virtual environments

From the repository root:

```powershell
./scripts/setup_envs.ps1                      # Windows: create all .venvs
./scripts/setup_envs.ps1 -Names tsfresh       # ...or only selected ones
```
```bash
./scripts/setup_envs.sh                       # Linux / macOS: create all .venvs
./scripts/setup_envs.sh -Names tsfresh,benchmark
```

This creates `.venvs/<package>/` for each package. See the README for the full
environment / Python-version table.

### 1.3 (Optional) Freeze lock files for bit-for-bit reproducibility

```powershell
./scripts/lock_requirements.ps1
```
```bash
./scripts/lock_requirements.sh
```

This runs `uv pip compile` to produce `requirements/<name>.lock.txt` files.

---

## 2. Download the datasets

Raw data is **not** included in this repository (it is excluded via `.gitignore`)
and must be downloaded into `data/<dataset>/`. Each `data/<dataset>/` subfolder
contains the dataset's own license and, where available, a README / description.

| Dataset key | Source |
|---|---|
| `condition_monitoring_of_hydraulic_systems` | UCI ML Repository — [Condition monitoring of hydraulic systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) (ZeMA gGmbH). See `data/condition_monitoring_of_hydraulic_systems/description.txt`. |
| `bosch_cnc` | Bosch CNC Machining dataset. See `data/bosch_cnc/README.md` and the companion paper (DOI [10.1016/j.procir.2022.04.022](https://doi.org/10.1016/j.procir.2022.04.022)). |
| `cnc_mill_tool_wear` | CNC Milling Tool Wear dataset. See `data/cnc_mill_tool_wear/` for the included files. |
| `turning_dataset` | Turning chatter dataset. See `data/turning_dataset/` for the included files. |

After download, the folder layout should match the structure shown in the main README.

---

## 3. Feature extraction

Feature extraction is configured via `feature_extraction_config.toml` and run
**one package at a time** in that package's virtual environment.

### 3.1 Configure a run

Edit `feature_extraction_config.toml`:

```toml
[benchmark_settings]
dataset = "CNC_Mill_Tool_Wear"        # dataset display name
data_dir = "data/cnc_mill_tool_wear/" # input data
export_dir = "results/cnc_mill_tool_wear/"  # where run logs / intermediate output go
packages = ["tsfresh"]                # ONE package per run

# ...per-package settings (see the file for all options)...
```

### 3.2 Run extraction for a package

Run the script in the matching package venv:

```powershell
# Windows
.\.venvs\tsfresh\Scripts\python.exe feature_extraction_benchmark.py
```
```bash
# Linux / macOS
./.venvs/tsfresh/bin/python feature_extraction_benchmark.py
```

Repeat for each package (`tsfresh`, `tsfel`, `pycatch22`, `seglearn`, `tsfeatures`,
`kats`), each time setting `packages = ["<pkg>"]` in the config and
running with the corresponding venv. (The config already runs one package at a time,
and `src/base.py` lazy-imports both extractors and data loaders, so only the active
package's dependencies are required.)

Extracted per-package feature CSVs are aggregated under
`results/03_Extracted_Features_Aggregated_Files/` (one CSV per dataset) and the
per-package folders used by the performance step live under
`results/02_Extracted_Features/<Dataset_Display_Name>/<package>/`.

Use `scripts/count_extracted_features.py` to summarise the number of features
extracted per package.

---

## 4. Performance evaluation (TPOT)

The performance benchmark trains a TPOT AutoML classifier on the extracted features
for each (dataset, seed). It is run with the **`benchmark`** venv.

### 4.1 Single run

```powershell
.\.venvs\benchmark\Scripts\python.exe performance_bechmark.py `
    --dataset condition_monitoring_of_hydraulic_systems `
    --seed 7429 --run 5 --max-time-mins 120
```
```bash
./.venvs/benchmark/bin/python performance_bechmark.py \
    --dataset condition_monitoring_of_hydraulic_systems \
    --seed 7429 --run 5 --max-time-mins 120
```

Output is written to
`results/11_Performance_Benchmarks/run_<run>__random_seed_<seed>/<Dataset_Display_Name>/`:

- `tpot_scores.csv` — scores (accuracy, balanced accuracy, F1) per package (appended).
- `<package>_tpot_pipeline.txt` — the best pipeline found by TPOT.
- `tpot_checkpoints_<package>/` — TPOT periodic checkpoints (gitignored).

### 4.2 Full sweep (5 seeds × 4 datasets)

The 5 published seeds are `run_1=27`, `run_2=44`, `run_3=821`, `run_4=1492`,
`run_5=7429`.

```powershell
./scripts/run_all_performance.ps1                                  # full sweep
./scripts/run_all_performance.ps1 -Datasets bosch_cnc -Runs 1 -MaxTimeMins 5  # quick test
```
```bash
./scripts/run_all_performance.sh
./scripts/run_all_performance.sh --datasets bosch_cnc --runs 1 --max-time-mins 5
```

### 4.3 CLI reference (`performance_bechmark.py`)

| Flag | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Dataset key: `bosch_cnc`, `cnc_mill_tool_wear`, `condition_monitoring_of_hydraulic_systems`, `turning_dataset`. |
| `--seed` | `7429` | Random seed. |
| `--run` | *(none)* | Run index → names the output dir `run_<run>__random_seed_<seed>`. If omitted, dir is `seed_<seed>`. |
| `--data-dir` | `results/02_Extracted_Features/<display_name>` | Input dir with per-package feature CSV subfolders + `targets.csv`. |
| `--export-dir` | `results/11_Performance_Benchmarks/run_<run>__random_seed_<seed>/<display_name>` | Output dir. |
| `--max-time-mins` | `120` | TPOT total search budget (minutes). |
| `--max-eval-time-mins` | `4` | TPOT per-pipeline eval time (minutes). |
| `--population-size` | `20` | TPOT population size. |
| `--cv` | `3` | TPOT inner CV folds. |
| `--n-jobs` | `20` | Parallel jobs. |
| `--early-stop` | `10` | TPOT early-stop generations. |

### 4.4 Fixed-classifier baselines (`scripts/fixed_classifier_evaluation.py`)

The TPOT AutoML benchmark confounds feature quality with the AutoML search
budget. To isolate feature quality, the same extracted features
are also evaluated with **fixed classifiers using no hyperparameter search** —
identical, sensible defaults for every package and dataset — so differences in
score reflect the features, not search luck. This is the budget-fair comparison
that isolates feature quality from search luck.

Four classifiers, each wrapped in the **same leakage-safe Pipeline** as TPOT
(impute → variance filter → [scaler] → classifier, fit on train only):

| Classifier | Scaled? | Key hyperparameters |
|---|:-:|---|
| `random_forest` | no | `n_estimators=200` |
| `xgboost` | no | `n_estimators=200, max_depth=6, learning_rate=0.1` |
| `logreg` | yes | `max_iter=1000` |
| `svm` | yes | `C=1.0, kernel=rbf` |

#### Single run

```powershell
.\.venvs\benchmark\Scripts\python.exe scripts\fixed_classifier_evaluation.py `
    --dataset turning_dataset --seed 7429 --run 5
```
```bash
./.venvs/benchmark/bin/python scripts/fixed_classifier_evaluation.py \
    --dataset turning_dataset --seed 7429 --run 5
```

Output: `fixed_classifier_scores.csv` (appended) with a `classifier` column:

```
dataset;run;seed;package;classifier;accuracy;balanced_accuracy;f1_score
```

To select a subset of classifiers, add `--classifiers random_forest xgboost`.

#### Merging with TPOT scores

Concatenate `tpot_scores.csv` (tag `classifier=tpot`) with
`fixed_classifier_scores.csv` into one master long-format table for
`scripts/compare_algorithms.py` (columns: `dataset`, `run`, `package`,
`classifier`, `balanced_accuracy`).

### 4.5 Group/time-aware split sensitivity (Bosch, `scripts/group_aware_evaluation.py`)

The default evaluation uses a single random stratified 80/20 split, which lets
the model see every machine / operation / timeframe in both train and test.
This overestimates generalisation for an industrial dataset
where the real question is transfer to **unseen** machines, operations or later
time periods. This script runs a sensitivity analysis on the Bosch CNC dataset
only, using group/time-aware splits, and reports the balanced-accuracy drop
relative to the random split:

| Split strategy | Folds | Description |
|---|:-:|---|
| `random` | 1 | stratified 80/20 (the default random-split baseline) |
| `leave_machine_out` | 3 | LeaveOneGroupOut over `machine` (train 2 → test 1) |
| `leave_operation_out` | 15 | LeaveOneGroupOut over `operation` (train 14 → test 1) |
| `temporal` | 1 | train on earlier years {2019, 2020} → test on {2021} |

Each split is scored with the **same fixed classifiers** as §4.4 under the
**same leakage-safe preprocessing** (fit on train only). TPOT is optionally
supported via `--include-tpot` but is off by default (the AutoML search budget
is a confound for a sensitivity analysis and is compute-heavy across the
3 + 15 group folds).

#### Single run

```powershell
.\.venvs\benchmark\Scripts\python.exe scripts\group_aware_evaluation.py `
    --dataset bosch_cnc --seed 7429 --run 5
```
```bash
./.venvs/benchmark/bin/python scripts/group_aware_evaluation.py \
    --dataset bosch_cnc --seed 7429 --run 5
```

Output: `group_aware_scores.csv` (appended) with `split_strategy`, `fold`,
`n_train`, `n_test` columns:

```
dataset;run;seed;package;classifier;split_strategy;fold;n_train;n_test;accuracy;balanced_accuracy;f1_score
```

To run a subset, add e.g. `--split-strategies random temporal` and/or
`--classifiers random_forest xgboost`.

#### Why Bosch only

Group/time-aware splits require per-sample group metadata that the other three
datasets do not expose in their targets:

- **CNC Mill Tool Wear** — `clamp_pressure;tool_condition`; no machine/run/year
  identifier (the UCI "experiment" number is not carried into the targets).
- **Hydraulic Systems** — condition labels only; no machine/group identifier.
- **Turning Dataset** — `measurement_idx;label`; windows are shuffled with no
  preserved run/file grouping.

Bosch is the only dataset whose targets carry `machine` and `operation` and
whose filenames encode the acquisition `year` (parsed from
`M01_Aug_2019_OP00_000.h5`). The infeasibility for the other datasets is
discussed as a limitation in the manuscript.

---

## 5. Statistical comparison and ranking

After the performance sweep, merge the per-run TPOT and fixed-classifier score
CSVs into one long-format master table, then run the nonparametric comparison.

### 5.1 Build the master table

```bash
./.venvs/benchmark/bin/python scripts/prepare_master_table.py
```

This concatenates every `tpot_scores.csv` and `fixed_classifier_scores.csv`
under `results/11_Performance_Benchmarks/run_*__random_seed_*/` into
`results/12_Rankings/performance_master_table.csv` (columns: `dataset`, `run`,
`seed`, `algorithm`, `classifier`, `accuracy`, `balanced_accuracy`, `f1_score`).

### 5.2 Nonparametric comparison and ranking

```bash
# Friedman (Iman-Davenport) + Holm post-hoc, per classifier
./.venvs/benchmark/bin/python scripts/compare_algorithms.py \
    results/12_Rankings/performance_master_table.csv \
    --direction higher --alpha 0.05
```

`compare_algorithms.py` analyses each classifier in turn (use `--classifier tpot`
to select one). It also supports a mixed-effects model (`--mode mixed`) and can
write a summary table (`--summary-only --summary-out summary.csv`).

Ranking outputs are written to `results/12_Rankings/`.

---

## 6. Repository layout (results)

| Path | Contents | Tracked? |
|---|---|---|
| `results/01_Feature_Extraction_Runs/` | Per-run extraction logs / intermediate output | No (gitignored) |
| `results/02_Extracted_Features/` | Per-package feature CSVs (large) | No (gitignored) |
| `results/03_Extracted_Features_Aggregated_Files/` | Aggregated feature CSVs per dataset | Yes |
| `results/11_Performance_Benchmarks/run_*__random_seed_*/` | TPOT scores + best pipelines | Yes (checkpoints excluded) |
| `results/12_Rankings/` | Ranking tables + AutoML results | Yes |
