"""Shared configuration and CLI helpers for the performance-evaluation scripts.

The dataset registry, default seeds and path-resolution logic are identical for the
TPOT benchmark (``performance_bechmark.py``), the fixed-classifier baselines
(``scripts/fixed_classifier_evaluation.py``) and the group/time-aware split
evaluation. Centralising them here avoids drift between the scripts and
keeps the per-package virtual environments lean (no script imports a heavy
dependency such as TPOT just to read the dataset registry).
"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATASETS",
    "DEFAULT_SEEDS",
    "add_common_args",
    "resolve_paths",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

# Dataset registry: key -> {display_name (results dir), target_column (in targets.csv)}.
DATASETS = {
    "bosch_cnc": {"display_name": "Bosch_CNC", "target_column": "label"},
    "cnc_mill_tool_wear": {"display_name": "CNC_Mill_Tool_Wear", "target_column": "tool_condition"},
    "condition_monitoring_of_hydraulic_systems": {
        "display_name": "Condition_Monitoring_of_Hydraulic_Systems",
        "target_column": "cooler_condition",
    },
    "turning_dataset": {"display_name": "Turning_Dataset", "target_column": "label"},
}

# Seeds used across the published runs (run_1..run_5).
DEFAULT_SEEDS = [27, 44, 821, 1492, 7429]


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the dataset/seed/path arguments shared by all evaluation scripts.

    Each script extends the returned parser with its own scenario-specific
    arguments (e.g. TPOT budget, or the set of fixed classifiers).
    """
    parser.add_argument("--dataset", required=True, choices=list(DATASETS), help="Dataset key.")
    parser.add_argument("--seed", type=int, default=7429, help="Random seed.")
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run index for the run_<n>__random_seed_<seed> output dir. "
        "If omitted, the output dir is named seed_<seed>.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Input dir with per-package feature CSV subfolders + targets.csv. "
        "Defaults to results/02_Extracted_Features/<display_name>.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Output dir. Defaults to "
        "results/11_Performance_Benchmarks/run_<run>__random_seed_<seed>/<display_name>.",
    )
    return parser


def resolve_paths(
    dataset: str,
    seed: int,
    run: int | None = None,
    data_dir: Path | None = None,
    export_dir: Path | None = None,
) -> tuple[Path, Path, str]:
    """Resolve the data/export directories and target column for a dataset.

    Returns
    -------
    data_dir : pathlib.Path
        Directory with per-package feature subfolders + ``targets.csv``.
    export_dir : pathlib.Path
        Output directory (created if missing).
    target_column : str
        Label column name in ``targets.csv``.
    """
    meta = DATASETS[dataset]
    display_name = meta["display_name"]

    if data_dir is None:
        data_dir = REPO_ROOT / "results" / "02_Extracted_Features" / display_name
    else:
        data_dir = Path(data_dir)

    if export_dir is None:
        if run is not None:
            run_dir = (
                REPO_ROOT
                / "results"
                / "11_Performance_Benchmarks"
                / f"run_{run}__random_seed_{seed}"
            )
        else:
            run_dir = REPO_ROOT / "results" / "11_Performance_Benchmarks" / f"seed_{seed}"
        export_dir = run_dir / display_name
    else:
        export_dir = Path(export_dir)

    export_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, export_dir, meta["target_column"]
