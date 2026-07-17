# Benchmarking Feature Extraction Packages

## Description
Performance benchmarking of several Python-packages for automatic feature extraction from time-series.

> **Reproducing the results:** For the full, end-to-end reproduction walkthrough
> (feature extraction → TPOT + fixed-classifier evaluation → group-aware splits →
> statistical ranking), see **[REPRODUCTION.md](REPRODUCTION.md)**.

## Datasets
The benchmark is performed on the following datasets:
| Dataset | Sampling Rate | # Samples | # Time-Series per Sample | # Targets |
|--|:-:|:-:|:-:|:-:|
| Bosch CNC | 2 kHz | 1700 | 3 | 15 |
| CNC Mill Tool Wear | 10 Hz | 18 | 45 | 2 |
| Condition Monitoring of Hydraulic Systems | variable (1 - 100 Hz) | 2205 | 17 | 5 |
| Turning Dataset | 10 kHz | depends on window_size (we use win_size=1s -> 808) | 1 | 2 |

There is a separate Dataloader available for every dataset for consistent access.

> **Note on dataset scope:** The published benchmark covers the four datasets
> above. Loaders for two additional audio datasets — **IDMT-ISA-Compressed-Air**
> (`src/loading/idmt_isa_compressed_air_dataset.py`) and **MIMII**
> (`src/loading/mimii.py`) — are included in `src/loading/` for completeness, but
> they are **not** part of the performance benchmark. Their extra audio
> dependencies live in `requirements/audio_loaders.txt`.

## Feature Extraction Packages
The benchmark is performed for the following feature extraction packages:
| Package | Publication Year | # Features per Sensor | Link |
|--|:--:|:--:|--|
| kats | 2022 | 68 | [GitHub](https://github.com/facebookresearch/Kats) |
| pycatch22 | 2019 | 22 / 24 | [GitHub](https://github.com/DynamicsAndNeuralSystems/pycatch22) |
| seglearn | 2018 | 28 | [GitHub](https://github.com/dmbee/seglearn) |
| tsfeatures | 2019 | 43 | [GitHub](https://github.com/Nixtla/tsfeatures) |
| TSFEL | 2020 | ~ 65 | [GitHub](https://github.com/fraunhoferportugal/tsfel) |
| tsfresh | 2018 | 794 | [GitHub](https://github.com/blue-yonder/tsfresh) |

There is a separate FeatureExtractor available for every package for consistent access.
Please visit the linked repositories of the packages for more information about them.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage **one isolated virtual
environment per feature-extraction package** (plus one for the TPOT performance
benchmark). This isolation is necessary because the packages have conflicting
dependencies — most notably **Kats, which requires Python 3.7**, while all other
packages run on **Python 3.12**.

### Prerequisite: install uv

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create all virtual environments

From the repository root, run one of:

```powershell
./scripts/setup_envs.ps1                      # Windows: create all .venvs
./scripts/setup_envs.ps1 -Names tsfresh       # ...or only selected ones
```
```bash
./scripts/setup_envs.sh                       # Linux/macOS: create all .venvs
./scripts/setup_envs.sh tsfresh benchmark     # ...or only selected ones
```

This creates the following environments under `.venvs/`:

| Environment   | Python | Contents                                     |
|---------------|:------:|----------------------------------------------|
| `tsfresh`     | 3.12   | tsfresh + shared loaders/benchmark deps      |
| `tsfel`       | 3.12   | TSFEL + shared deps                          |
| `pycatch22`   | 3.10   | pycatch22 + shared deps (wheels only for 3.10) |
| `seglearn`    | 3.12   | seglearn + shared deps                       |
| `tsfeatures`  | 3.12   | tsfeatures + shared deps                     |
| `kats`        | 3.7    | Kats (requires Python 3.7)                   |
| `benchmark`   | 3.12   | TPOT + XGBoost + scikit-posthocs + aeon      |

Each environment's dependencies are declared in `requirements/<name>.txt`
(shared deps live in `requirements/base.txt`).

### Reproducible lock files (optional, recommended)

To freeze every transitive dependency to an exact version (resolved for each
environment's Python version), run:

```powershell
./scripts/lock_requirements.ps1
```
```bash
./scripts/lock_requirements.sh
```

This writes `requirements/<name>.lock.txt`. To install from a lock file instead:

```bash
uv pip install -r requirements/<name>.lock.txt --python .venvs/<name>
```

### Kats on Python 3.7

`uv venv --python 3.7` will attempt to fetch a managed CPython 3.7. If your
platform does not provide one, install Python 3.7 yourself and point uv at it:

```bash
uv venv .venvs/kats --python /path/to/python3.7
uv pip install -r requirements/kats.txt --python .venvs/kats
```

## Usage

Feature extraction is run **one package at a time**, each in its own virtual
environment (only the active package needs to be installed — `src/base.py`
imports packages lazily).

1) Create the virtual environments (see Installation).
2) Activate the environment for the package you want to run, e.g.:
   ```powershell
   .venvs\tsfresh\Scripts\Activate.ps1      # Windows PowerShell
   ```
   ```bash
   source .venvs/tsfresh/bin/activate        # Linux/macOS
   ```
   ...or run a one-off command in a venv without activating:
   ```bash
   uv pip run --python .venvs/tsfresh python feature_extraction_benchmark.py
   ```
3) Set the active package in `feature_extraction_config.toml`
   (`packages = ["tsfresh"]`) and configure the dataset / export directory.
4) Run `python feature_extraction_benchmark.py`. Datasets are downloaded
   automatically if necessary.
5) Extracted features and measurement metadata are exported to `export_dir`
   as specified in the configuration file.

### Performance evaluation

The downstream performance benchmark runs in the **`benchmark`** environment
(`tpot`, `xgboost`, `scikit-posthocs`, `aeon`). It comprises three tracks, all
sharing one leakage-safe preprocessing pipeline (`src/evaluation/preprocessing.py`):

1. **TPOT AutoML** (`performance_bechmark.py`) — automated pipeline search per
   (dataset, seed); the headline benchmark.
2. **Fixed-classifier baselines** (`scripts/fixed_classifier_evaluation.py`) —
   the same features scored with fixed, search-free classifiers (Random Forest,
   XGBoost, Logistic Regression, SVM) to isolate feature quality from the AutoML
   search budget.
3. **Group/time-aware split sensitivity** (`scripts/group_aware_evaluation.py`,
   Bosch only) — leave-machine-out / leave-operation-out / temporal splits to
   test generalisation to unseen machines, operations and later time periods.

After the sweep, merge the per-run scores and run the statistical ranking:

```bash
source .venvs/benchmark/bin/activate
python performance_bechmark.py --dataset bosch_cnc --seed 7429 --run 5 --max-time-mins 120
python scripts/fixed_classifier_evaluation.py --dataset bosch_cnc --seed 7429 --run 5
python scripts/group_aware_evaluation.py --dataset bosch_cnc --seed 7429 --run 5
python scripts/prepare_master_table.py
python scripts/compare_algorithms.py results/12_Rankings/performance_master_table.csv --direction higher
```

See **[REPRODUCTION.md](REPRODUCTION.md)** for the full CLI reference, the
5-seed sweep helpers (`scripts/run_all_performance.{ps1,sh}`) and the ranking
outputs.

## Troubleshooting
- **Kats install**: see [facebookresearch/Kats#308](https://github.com/facebookresearch/Kats/issues/308).
- **uv cannot fetch Python 3.7**: install Python 3.7 on your system and pass its
  path to `uv venv --python` (see the Kats section above).
- **pycatch22 build fails on Python 3.12**: pycatch22 only publishes wheels for
  CPython 3.10, so the `pycatch22` environment is created with Python 3.10 (handled
  automatically by the setup scripts). Do not switch it to 3.12 unless you have
  MSVC build tools installed.

## Project Structure

### Overview
```
├── archive/                # Earlier iterations of analysis scripts (not part of the final pipeline)
├── data/                   # Datasets (not tracked; download separately — see REPRODUCTION.md)
├── notebooks/              # Exploration / plotting notebooks
├── requirements/           # Per-environment pip requirements (base.txt + <package>.txt)
├── results/                # Benchmark outputs (partly tracked — see REPRODUCTION.md)
├── scripts/                # Evaluation, ranking and environment-setup scripts
├── src/
│   ├── evaluation/         # Classifiers, leakage-safe preprocessing, shared CLI config
│   ├── extraction/         # Feature extractor classes (one per package)
│   ├── loading/            # Dataloader classes (one per dataset)
│   └── base.py             # Shared benchmark infrastructure (lazy imports)
├── tests/                  # pytest suite (loading, extraction, evaluation)
├── .gitignore
├── README.md
├── REPRODUCTION.md         # Full reproduction guide
├── pyproject.toml
├── feature_extraction_benchmark.py   # Feature-extraction entry point
├── feature_extraction_config.toml    # Feature-extraction configuration
├── performance_bechmark.py           # TPOT performance-benchmark entry point
├── environment.yml / environment_kats.yml   # Legacy conda envs (superseded by requirements/ + uv)
└── LICENSE
```

### Key files & directories
- *feature_extraction_benchmark.py* / *feature_extraction_config.toml*: entry
  point and configuration for the feature-extraction experiments.
- *performance_bechmark.py*: entry point for the TPOT performance benchmark.
- *scripts/*: evaluation scripts (fixed-classifier baselines, group-aware
  splits, master-table merge, statistical comparison) plus environment setup
  helpers (`setup_envs`, `lock_requirements`, `run_all_performance`).
- *src/extraction*: feature-extractor classes for the packages listed above.
- *src/loading*: dataloader classes for the datasets listed above.
- *src/evaluation*: shared classifiers, leakage-safe preprocessing and CLI
  config used by all performance-evaluation scripts.
- *requirements/*: per-environment dependencies (installed via `uv` into
  `.venvs/<name>/`). `environment.yml` / `environment_kats.yml` are legacy
  conda specs, superseded by `requirements/` + uv.

## Tests
Unittests are available to test the data loading. Run `pytest` from the console to run the tests.

Note: As some datasets are very large (>> 10 GB), these tests take some time. Tests with expected
runtime > 10 min are disabled by default.

## Authors and acknowledgment

- Christian T. Seidler
- Daniel Müller
- Simon Wolf
- Hartmut Eigenbrod
- Marco F. Huber
