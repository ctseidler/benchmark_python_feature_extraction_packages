#!/usr/bin/env python3
"""Group/time-aware split sensitivity analysis for the Bosch CNC dataset.

The default performance evaluation uses a single random stratified 80/20 split.
This is too weak for an industrial dataset where
generalisation across machines / operations / time is the real question: a
random split lets the model see samples from every machine, every operation and
every timeframe in both train and test, which inflates scores relative to a
deployed system that must generalise to *unseen* machines, operations or later
time periods.

This script runs a **sensitivity analysis** on the Bosch CNC dataset only, using
group/time-aware splits, and reports the balanced-accuracy drop relative to the
random stratified split:

    - ``random``               : stratified 80/20 split (the default random-split baseline).
    - ``leave_machine_out``    : LeaveOneGroupOut over ``machine`` (3 folds;
                                 train on 2 machines, test on the held-out one).
    - ``leave_operation_out``  : LeaveOneGroupOut over ``operation`` (15 folds;
                                 train on 14 operations, test on the held-out one).
    - ``temporal``             : train on the earlier acquisition years, test on
                                 the latest year (forward-in-time evaluation;
                                 train {2019, 2020} -> test {2021}).

Each split is scored with the SAME fixed classifiers as the fixed-classifier
baselines (:mod:`src.evaluation.classifiers`) under the SAME leakage-safe
preprocessing protocol (:mod:`src.evaluation.preprocessing`): the column-selection
mask and the impute/variance/scaler pipeline are fit on the TRAIN split only and
applied to the test split. TPOT is optionally supported via ``--include-tpot`` but
is off by default because the AutoML search budget is a confound for a
sensitivity analysis and is compute-heavy across the 3+15 group folds.

Why Bosch only
--------------
Group/time-aware splits require per-sample group metadata that the other three
datasets do not expose in their targets:

    - CNC Mill Tool Wear   : targets are ``clamp_pressure;tool_condition`` — no
                             machine/run/year identifier (the UCI "experiment"
                             number is not carried into the feature targets).
    - Hydraulic Systems    : targets are condition labels
                             (``cooler_condition;valve_condition;...``) with no
                             machine/group identifier.
    - Turning Dataset      : targets are ``measurement_idx;label`` — windows are
                             shuffled with no preserved run/file grouping.

Bosch is the only dataset whose targets carry ``machine`` and ``operation`` and
whose filenames encode the acquisition ``year`` (parsed from e.g.
``M01_Aug_2019_OP00_000.h5``). The infeasibility for the other datasets is
discussed as a limitation in the manuscript.

Output
------
A long-format CSV (``group_aware_scores.csv``) with one row per
(dataset, package, classifier, split_strategy, fold):

    dataset;run;seed;package;classifier;split_strategy;fold;
    n_train;n_test;accuracy;balanced_accuracy;f1_score

For ``leave_machine_out`` / ``leave_operation_out`` the per-fold scores can be
aggregated (mean +/- std) downstream; the random split gives a single reference
row to quantify the group/time drop.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from glob import glob
from pathlib import Path

# Make the repo root importable when running the script from anywhere (e.g.
# `python scripts/group_aware_evaluation.py`), so `from src...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.evaluation.classifiers import CLASSIFIERS, build_full_pipeline
from src.evaluation.config import REPO_ROOT, add_common_args, resolve_paths
from src.evaluation.preprocessing import (
    build_preprocessing_pipeline,
    compute_nan_column_mask,
    load_features_and_targets,
)
from src.loading.bosch_file_utils import enumerate_files

# Splits offered by this script.
SPLIT_STRATEGIES = ("random", "leave_machine_out", "leave_operation_out", "temporal")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Group/time-aware split sensitivity analysis (Bosch CNC)."
    )
    add_common_args(parser)
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=list(CLASSIFIERS),
        choices=list(CLASSIFIERS),
        help="Subset of fixed classifiers to run. Default: all.",
    )
    parser.add_argument("--n-jobs", type=int, default=20, help="Parallel jobs (RF/XGB).")
    parser.add_argument(
        "--split-strategies",
        nargs="+",
        default=list(SPLIT_STRATEGIES),
        choices=list(SPLIT_STRATEGIES),
        help="Subset of split strategies to run. Default: all.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "bosch_cnc" / "data",
        help="Root of the raw Bosch dataset (for filename->year parsing). "
        "Default: data/bosch_cnc/data.",
    )
    parser.add_argument(
        "--include-tpot",
        action="store_true",
        help="Also run the TPOT AutoML benchmark per split (compute-heavy; off by "
        "default — the fixed classifiers are the budget-fair comparison).",
    )
    parser.add_argument(
        "--max-time-mins", type=int, default=120, help="TPOT total search budget (minutes)."
    )
    parser.add_argument(
        "--max-eval-time-mins", type=int, default=4, help="TPOT per-pipeline eval time (minutes)."
    )
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--early-stop", type=int, default=10)
    return parser.parse_args()


def attach_year(targets: pd.DataFrame, raw_data_dir: Path) -> pd.DataFrame:
    """Attach a ``year`` column to the Bosch targets by re-enumerating raw data.

    The existing ``targets.csv`` (written by the feature-extraction benchmark)
    carries ``machine;operation;label`` but not the acquisition year. The year
    is encoded in the raw ``.h5`` filenames (e.g. ``M01_Aug_2019_OP00_000.h5``),
    so it is recovered by re-enumerating the raw dataset with the *same* logic
    the loader uses (see :func:`src.loading.bosch_file_utils.enumerate_files`).

    Safety check
    ------------
    The re-enumerated ``machine/operation/label`` sequence is compared
    row-for-row against the existing targets. They must match exactly — this
    guarantees the parsed ``year`` is aligned to the correct feature row. If
    they differ (e.g. the extracted features are from a different dataset
    version) a :class:`ValueError` is raised rather than silently misaligning.

    Parameters
    ----------
    targets : pandas.DataFrame
        Existing Bosch targets (columns include ``machine``, ``operation``,
        ``label``), as read from ``targets.csv``.
    raw_data_dir : pathlib.Path
        Root of the raw Bosch dataset.

    Returns
    -------
    pandas.DataFrame
        ``targets`` with an added integer ``year`` column, reset to a clean
        RangeIndex aligned to the enumeration order.
    """
    file_df = enumerate_files(raw_data_dir)
    regen = file_df[["machine", "operation", "label"]].reset_index(drop=True)
    existing = targets[["machine", "operation", "label"]].reset_index(drop=True)
    if not regen.equals(existing):
        raise ValueError(
            "Re-enumerated Bosch files do not match the existing targets.csv "
            "row-for-row (machine/operation/label). The extracted features may "
            "be from a different dataset version or enumeration order; cannot "
            "safely attach the year column for group/time-aware splits."
        )
    targets = targets.reset_index(drop=True).copy()
    targets["year"] = file_df["year"].reset_index(drop=True).to_numpy()
    return targets


def iter_splits(groups: pd.DataFrame, y: pd.Series, random_seed: int, strategies):
    """Yield ``(strategy, fold_id, train_idx, test_idx)`` for the requested splits.

    Parameters
    ----------
    groups : pandas.DataFrame
        Per-sample group columns (``machine``, ``operation``, ``year``), aligned
        to ``y`` with a clean RangeIndex.
    y : pandas.Series
        Encoded labels, aligned to ``groups``.
    random_seed : int
        Seed for the random stratified split.
    strategies : iterable of str
        Subset of :data:`SPLIT_STRATEGIES` to produce.

    Yields
    ------
    strategy : str
    fold_id : str
        Identifier for the fold (held-out group name, or temporal split
        description, or ``""`` for the random split).
    train_idx, test_idx : numpy.ndarray
        Positional indices into ``groups`` / ``y``.
    """
    strategies = set(strategies)
    idx = np.arange(len(groups))
    y_arr = y.to_numpy()

    if "random" in strategies:
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, stratify=y_arr, random_state=random_seed
        )
        yield "random", "", train_idx, test_idx

    if "leave_machine_out" in strategies:
        logo = LeaveOneGroupOut()
        machine_groups = groups["machine"].to_numpy()
        if groups["machine"].nunique() < 2:
            warnings.warn(
                "leave_machine_out skipped: fewer than 2 unique machines in the "
                f"aligned subset ({sorted(groups['machine'].unique())}).",
                stacklevel=2,
            )
        else:
            for train_idx, test_idx in logo.split(idx, y_arr, machine_groups):
                yield (
                    "leave_machine_out",
                    str(groups["machine"].iloc[test_idx[0]]),
                    train_idx,
                    test_idx,
                )

    if "leave_operation_out" in strategies:
        logo = LeaveOneGroupOut()
        operation_groups = groups["operation"].to_numpy()
        if groups["operation"].nunique() < 2:
            warnings.warn(
                "leave_operation_out skipped: fewer than 2 unique operations in "
                f"the aligned subset ({sorted(groups['operation'].unique())}).",
                stacklevel=2,
            )
        else:
            for train_idx, test_idx in logo.split(idx, y_arr, operation_groups):
                yield (
                    "leave_operation_out",
                    str(groups["operation"].iloc[test_idx[0]]),
                    train_idx,
                    test_idx,
                )

    if "temporal" in strategies:
        years = sorted(groups["year"].unique())
        if len(years) < 2:
            raise ValueError(
                f"Temporal split needs at least two distinct years; found only {years}."
            )
        max_year = years[-1]
        year_arr = groups["year"].to_numpy()
        train_idx = idx[year_arr < max_year]
        test_idx = idx[year_arr == max_year]
        fold_id = f"train_{'-'.join(str(yr) for yr in years[:-1])}__test_{max_year}"
        yield "temporal", fold_id, train_idx, test_idx


def evaluate_split(X, y, train_idx, test_idx, clf_name, seed, n_jobs, tpot_args=None):
    """Fit one classifier on a leakage-safe train split and score the test split.

    The column-selection mask and the impute/variance(/scaler) pipeline are fit
    on the TRAIN split only and applied to the test split, identical to the
    leakage-safe protocol used by the other evaluations. For the fixed classifiers the preprocessing and
    classifier are a single :class:`~sklearn.pipeline.Pipeline`; for TPOT the
    leakage-safe preprocessing (``scale=False``) is applied first and TPOT then
    searches on the transformed train split (mirroring ``performance_bechmark.py``).
    """
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Column selection (all-NaN / >=50%-NaN) from TRAIN only.
    keep_mask = compute_nan_column_mask(X_train, max_nan_fraction=0.5)
    X_train = X_train.loc[:, keep_mask]
    X_test = X_test.loc[:, keep_mask]

    if clf_name == "tpot":
        from tpot import TPOTClassifier

        pre = build_preprocessing_pipeline(scale=False)
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)
        tpot = TPOTClassifier(
            search_space="linear",
            max_time_mins=tpot_args["max_time_mins"],
            max_eval_time_mins=tpot_args["max_eval_time_mins"],
            population_size=tpot_args["population_size"],
            cv=tpot_args["cv"],
            scorers="balanced_accuracy",
            verbose=0,
            random_state=seed,
            n_jobs=n_jobs,
            early_stop=tpot_args["early_stop"],
        )
        tpot.fit(X_train_t, y_train)
        y_pred = tpot.predict(X_test_t)
    else:
        pipeline = build_full_pipeline(clf_name, seed, n_jobs)
        with warnings.catch_warnings():
            # Silence convergence warnings (e.g. LogReg on high-dim features).
            warnings.simplefilter("ignore", category=UserWarning)
            pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

    return (
        accuracy_score(y_test, y_pred),
        balanced_accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average="weighted"),
        len(train_idx),
        len(test_idx),
    )


def main() -> None:
    """Run the group/time-aware split sensitivity analysis."""
    args = parse_args()

    if args.dataset != "bosch_cnc":
        sys.exit(
            "group_aware_evaluation.py is Bosch-specific: it needs the "
            "machine/operation group columns and the filename-encoded year. "
            f"Dataset '{args.dataset}' has no usable group metadata "
            "(see the module docstring / manuscript limitation discussion)."
        )

    data_dir, export_dir, target_column = resolve_paths(
        args.dataset, args.seed, args.run, args.data_dir, args.export_dir
    )
    random_seed = args.seed

    tpot_args = {
        "max_time_mins": args.max_time_mins,
        "max_eval_time_mins": args.max_eval_time_mins,
        "population_size": args.population_size,
        "cv": args.cv,
        "early_stop": args.early_stop,
    }

    # The classifier list: fixed baselines + (optionally) tpot.
    clf_names = list(args.classifiers)
    if args.include_tpot:
        clf_names = clf_names + ["tpot"]

    score_export_path = export_dir / "group_aware_scores.csv"

    # Per-package feature subfolders.
    subdirs = [d for d in glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]

    # Read + year-augment the targets once (shared across packages).
    targets = pd.read_csv(data_dir / "targets.csv", sep=";", decimal=",")
    targets = attach_year(targets, args.raw_data_dir)

    n_machine = targets["machine"].nunique()
    n_operation = targets["operation"].nunique()
    years = sorted(targets["year"].unique())
    print(
        f"Bosch groups: machine={n_machine} (LOGO folds), "
        f"operation={n_operation} (LOGO folds), years={years} (temporal split)."
    )

    for subdir in tqdm(subdirs):
        package_name = subdir.split(os.sep)[-1]
        print(f"\nProcessing package: {package_name}")

        X, y, _ = load_features_and_targets(subdir, target_column)

        # Align the year-augmented targets to the feature rows POSITIONALLY
        # (load_features_and_targets already reset X to a clean RangeIndex and
        # the year-augmented targets share that enumeration order), then reset
        # to a clean shared RangeIndex so positional split indices are
        # unambiguous. Positional alignment is required because some packages
        # write a non-unique feature index (e.g. seglearn).
        n_X = len(X)
        tgt = targets.iloc[:n_X].reset_index(drop=True)
        groups = tgt[["machine", "operation", "year"]].reset_index(drop=True)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Encode labels (label set is a known property -> fit on full y).
        encoder = LabelEncoder()
        y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)

        for strategy, fold_id, train_idx, test_idx in iter_splits(
            groups, y_encoded, random_seed, args.split_strategies
        ):
            for clf_name in clf_names:
                acc, bal_acc, f1, n_train, n_test = evaluate_split(
                    X,
                    y_encoded,
                    train_idx,
                    test_idx,
                    clf_name,
                    random_seed,
                    args.n_jobs,
                    tpot_args=tpot_args,
                )
                score_results = {
                    "dataset": args.dataset,
                    "run": args.run if args.run is not None else "",
                    "seed": random_seed,
                    "package": package_name,
                    "classifier": clf_name,
                    "split_strategy": strategy,
                    "fold": fold_id,
                    "n_train": n_train,
                    "n_test": n_test,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "f1_score": f1,
                }
                score_df = pd.DataFrame([score_results])
                if not score_export_path.exists():
                    score_df.to_csv(score_export_path, index=False, sep=";", decimal=",")
                else:
                    score_df.to_csv(
                        score_export_path,
                        index=False,
                        sep=";",
                        decimal=",",
                        mode="a",
                        header=False,
                    )
                print(
                    f"  {strategy:<22} fold={fold_id:<28} {clf_name:<14} "
                    f"bal_acc={bal_acc:.4f} (n_train={n_train}, n_test={n_test})"
                )


if __name__ == "__main__":
    main()
