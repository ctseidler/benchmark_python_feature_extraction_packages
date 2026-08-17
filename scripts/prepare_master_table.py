#!/usr/bin/env python3
"""
Merge all per-run TPOT + fixed-classifier score CSVs into one master table.

Reads every ``tpot_scores.csv`` and ``fixed_classifier_scores.csv`` under
``results/11_Performance_Benchmarks/run_<i>__random_seed_<seed>/<Dataset>/`` and
concatenates them into a single long-format CSV that the ranking / critical-
difference scripts (``compare_algorithms.py``, ``critical_difference.py``) and
the new ``rank_performance.py`` consume.

The merge normalises:

* column names to a common schema
  (``dataset, run, seed, algorithm, classifier, accuracy, balanced_accuracy,
  f1_score``);
* the package *folder* names (which encode the extraction config AND the
  parallelism, e.g. ``tsfresh__params_efficient__jobs_60``) into clean
  *algorithm* display names that match the paper (e.g. ``tsfresh_efficient``).
  The ``__jobs_<n>`` / ``__threads_<n>`` suffix is a parallelism setting, not a
  feature configuration, so folders that differ only in that suffix map to the
  SAME algorithm (their features are identical);
* the ``classifier`` column: ``"tpot"`` for TPOT rows, the baseline name
  (``random_forest`` / ``xgboost`` / ``logreg`` / ``svm``) for baseline rows, so
  the TPOT-vs-fixed comparison can be done with one grouping.

Output
------
``results/12_Rankings/performance_master_table.csv`` (semicolon-separated,
decimal=``","``) — the single input for all downstream ranking analyses.

Replaces the stale ``AutoML_results.csv`` / ``AutoML_results_2.csv`` which held
the pre-leakage-fix results.

Created on: 09-07-2026
by Christian Seidler <christian.seidler@ipa.fraunhofer.de>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Package-folder -> algorithm display-name mapping.
# Folders that differ ONLY in the __jobs_<n> / __threads_<n> parallelism suffix
# map to the same algorithm (identical features). Names follow the paper's
# convention (lowercase, package_paramstyle).
# --------------------------------------------------------------------------- #
PACKAGE_DISPLAY_MAP: dict[str, str] = {
    # tsfresh
    "tsfresh__params_comprehensive__jobs_0": "tsfresh_comprehensive",
    "tsfresh__params_comprehensive__jobs_60": "tsfresh_comprehensive",
    "tsfresh__params_efficient__jobs_0": "tsfresh_efficient",
    "tsfresh__params_efficient__jobs_60": "tsfresh_efficient",
    # tsfel (domain is the feature config; jobs is parallelism)
    "tsfel__domain_none__jobs_1": "tsfel_none",
    "tsfel__domain_none__jobs_60": "tsfel_none",
    "tsfel__domain_spectral__jobs_1": "tsfel_spectral",
    "tsfel__domain_statistical__jobs_1": "tsfel_statistical",
    "tsfel__domain_temporal__jobs_1": "tsfel_temporal",
    # seglearn
    "seglearn__features_all": "seglearn_all",
    "seglearn__features_default": "seglearn_default",
    # pycatch22
    "pycatch22__catch24_false": "pycatch22",
    # tsfeatures (threads is parallelism)
    "tsfeatures__threads_1": "tsfeatures",
    "tsfeatures__threads_60": "tsfeatures",
    # kats
    "kats": "kats",
}

# Canonical column order of the master table.
MASTER_COLUMNS = [
    "dataset",
    "run",
    "seed",
    "algorithm",
    "classifier",
    "accuracy",
    "balanced_accuracy",
    "f1_score",
]


def _map_package(folder_name: str) -> str:
    """Map a package folder name to its algorithm display name.

    Falls back to the folder name itself (stripped of a trailing ``__jobs_*`` /
    ``__threads_*`` parallelism suffix) if the folder is not in the explicit
    map, so new package configs are not silently dropped.
    """
    if folder_name in PACKAGE_DISPLAY_MAP:
        return PACKAGE_DISPLAY_MAP[folder_name]
    # Heuristic fallback: strip a trailing __jobs_<n> or __threads_<n>.
    import re

    cleaned = re.sub(r"__jobs_\d+$", "", folder_name)
    cleaned = re.sub(r"__threads_\d+$", "", cleaned)
    return cleaned


def _load_one(csv_path: Path, classifier_label: str) -> pd.DataFrame:
    """Load one score CSV and normalise its columns.

    Parameters
    ----------
    csv_path : Path
        Path to a ``tpot_scores.csv`` (no ``classifier`` column) or a
        ``fixed_classifier_scores.csv`` (has a ``classifier`` column).
    classifier_label : str
        ``"tpot"`` for TPOT CSVs; ignored for baseline CSVs (the per-row
        ``classifier`` column is kept instead).
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    if "classifier" not in df.columns:
        df["classifier"] = classifier_label
    df = df.rename(columns={"package": "algorithm"})
    df["algorithm"] = df["algorithm"].map(_map_package)
    # Keep only the master columns that exist; missing ones become NaN.
    for col in MASTER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[MASTER_COLUMNS]


def build_master_table(results_root: Path) -> pd.DataFrame:
    """Concatenate every run/dataset score CSV into one master DataFrame."""
    bench_dir = results_root / "11_Performance_Benchmarks"
    tpot_files = sorted(bench_dir.glob("run_*__*/*/tpot_scores.csv"))
    base_files = sorted(bench_dir.glob("run_*__*/*/fixed_classifier_scores.csv"))

    if not tpot_files and not base_files:
        sys.exit(
            f"No tpot_scores.csv / fixed_classifier_scores.csv found under {bench_dir}. "
            "Run the performance benchmark first."
        )

    frames = [_load_one(f, "tpot") for f in tpot_files]
    frames += [_load_one(f, "") for f in base_files]
    master = pd.concat(frames, ignore_index=True)

    # Some algorithms were extracted with multiple parallelism settings
    # (e.g. tsfel__domain_none__jobs_1 and __jobs_60, or tsfeatures__threads_1
    # and __threads_60). These produce IDENTICAL features (the n_jobs/threads
    # setting only affects extraction speed, not the feature values), so they
    # are duplicate evaluations of the same algorithm. Averaging their scores
    # collapses each (dataset, run, algorithm, classifier) group to a single
    # row, which the Friedman / ranking tests require (one score per algorithm
    # per block). ``seed`` is kept constant across the variants by taking the
    # first (they share the same run -> same seed).
    score_cols = ["accuracy", "balanced_accuracy", "f1_score"]
    group_cols = ["dataset", "run", "algorithm", "classifier"]
    n_before = len(master)
    master = master.groupby(group_cols, as_index=False)[score_cols + ["seed"]].agg(
        {**{c: "mean" for c in score_cols}, "seed": "first"}
    )
    n_after = len(master)
    if n_before != n_after:
        print(
            f"  deduplicated parallelism variants: {n_before} -> {n_after} rows "
            f"({n_before - n_after} duplicates averaged)"
        )

    # Stable, readable ordering.
    master = master.sort_values(["dataset", "run", "algorithm", "classifier"]).reset_index(
        drop=True
    )
    return master


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Merge per-run TPOT + fixed-classifier score CSVs into one master table."
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
        help="Root of the results/ tree (default: <repo>/results).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-root>/12_Rankings/performance_master_table.csv).",
    )
    return p.parse_args()


def main() -> None:
    """Run the script."""
    args = parse_args()
    out_path = args.out or (args.results_root / "12_Rankings" / "performance_master_table.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    master = build_master_table(args.results_root)
    master.to_csv(out_path, index=False, sep=";", decimal=",")

    n_runs = master["run"].nunique()
    n_ds = master["dataset"].nunique()
    n_algo = master["algorithm"].nunique()
    n_clf = master["classifier"].nunique()
    print(f"Wrote {len(master)} rows to {out_path}")
    print(f"  runs={n_runs}  datasets={n_ds}  algorithms={n_algo}  classifiers={n_clf}")
    print("\nRows per (dataset, classifier):")
    print(master.groupby(["dataset", "classifier"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
