#!/usr/bin/env python3
"""
Fixed-classifier baselines for the feature-extraction benchmark.

The TPOT AutoML benchmark (``performance_bechmark.py``) confounds feature quality
with the AutoML search budget: a package whose features suit TPOT's search space
can win by search luck rather than feature quality. This script removes that
confound by evaluating the SAME extracted features with a set of FIXED
classifiers that use no hyperparameter search — identical, sensible defaults for
every package and dataset — so differences in score reflect feature quality, not
search budget.

Preprocessing is leakage-safe and identical to the TPOT benchmark: the data is
split FIRST on the raw features, then imputation / variance filtering / scaling
are fit on the TRAIN split only via an sklearn Pipeline and applied to the test
split (see ``src/evaluation/preprocessing.py``).

Classifiers (each wrapped in the shared leakage-safe Pipeline):
    - RandomForest (tree-based, no scaling needed)
    - XGBoost      (tree-based, no scaling needed)
    - LogisticRegression (scaled)
    - SVM (RBF, scaled)

Output: a long-format CSV (``fixed_classifier_scores.csv``) with a ``classifier``
column, so it can be concatenated with the TPOT scores (classifier="tpot") into a
single master table for ``scripts/compare_algorithms.py``.
"""

import argparse
import os
import sys
import warnings
from glob import glob
from pathlib import Path

# Make the repo root importable when running the script from anywhere (e.g.
# `python scripts/fixed_classifier_evaluation.py`), so `from src...` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.evaluation.classifiers import CLASSIFIERS, build_full_pipeline
from src.evaluation.config import add_common_args, resolve_paths
from src.evaluation.preprocessing import compute_nan_column_mask, load_features_and_targets


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run fixed-classifier baselines on extracted feature sets."
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
    return parser.parse_args()


def main() -> None:
    """Run the script."""
    args = parse_args()
    data_dir, export_dir, target_column = resolve_paths(
        args.dataset, args.seed, args.run, args.data_dir, args.export_dir
    )
    random_seed = args.seed

    score_export_path = export_dir / "fixed_classifier_scores.csv"

    # Get a list of all subdirectories in the data directory (one per package).
    subdirs = [d for d in glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]

    for subdir in tqdm(subdirs):
        package_name = subdir.split(os.sep)[-1]
        print(f"\nProcessing package: {package_name}")

        X, y, _ = load_features_and_targets(subdir, target_column)

        # Encode labels (label set is a known property of the problem, not a
        # fitted statistic -> fitting on full y is not leakage).
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # --- Leakage-safe protocol: split FIRST on raw features ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_seed
        )

        # Column selection (all-NaN / >=50%-NaN) from TRAIN only.
        keep_mask = compute_nan_column_mask(X_train, max_nan_fraction=0.5)
        X_train = X_train.loc[:, keep_mask]
        X_test = X_test.loc[:, keep_mask]

        for clf_name in args.classifiers:
            print(f"  -> classifier: {clf_name}")
            pipeline = build_full_pipeline(clf_name, random_seed, args.n_jobs)

            with warnings.catch_warnings():
                # Silence convergence warnings (e.g. LogReg on high-dim features);
                # they do not affect the reported test score.
                warnings.simplefilter("ignore", category=UserWarning)
                pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            score_results = {
                "dataset": args.dataset,
                "run": args.run if args.run is not None else "",
                "seed": random_seed,
                "package": package_name,
                "classifier": clf_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
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
                f"     balanced_accuracy={score_results['balanced_accuracy']:.4f} "
                f"accuracy={score_results['accuracy']:.4f} "
                f"f1={score_results['f1_score']:.4f}"
            )


if __name__ == "__main__":
    main()
