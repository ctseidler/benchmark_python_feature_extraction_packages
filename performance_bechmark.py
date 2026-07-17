"""
Script to run performance benchmarks on extracted feature set using TPOT.

Preprocessing follows a leakage-safe protocol: the data is
split FIRST on the raw features, then all imputation / variance filtering is fit on
the TRAIN split only (via an sklearn Pipeline) and applied to the test split, so no
test-set statistic influences any fitted step. See ``src/evaluation/preprocessing.py``.

"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="stopit")

from glob import glob

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier
from tqdm import tqdm

from src.evaluation.config import add_common_args, resolve_paths
from src.evaluation.preprocessing import (
    build_preprocessing_pipeline,
    compute_nan_column_mask,
    load_features_and_targets,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a TPOT performance benchmark on extracted feature sets."
    )
    add_common_args(parser)
    parser.add_argument(
        "--max-time-mins", type=int, default=120, help="TPOT total search budget (minutes)."
    )
    parser.add_argument(
        "--max-eval-time-mins", type=int, default=4, help="TPOT per-pipeline eval time (minutes)."
    )
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--early-stop", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Run the script."""
    args = parse_args()
    data_dir, export_dir, target_column = resolve_paths(
        args.dataset, args.seed, args.run, args.data_dir, args.export_dir
    )
    random_seed = args.seed

    # Get a list of all subdirectories in the data directory
    subdirs = [d for d in glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]

    for subdir in tqdm(subdirs):
        package_name = subdir.split(os.sep)[-1]
        print(f"\nProcessing package: {package_name}")

        # Load features + targets with the shared helper, which aligns by
        # POSITION (not by CSV index) — critical because some packages write a
        # non-unique index (e.g. seglearn repeats 0 for every recording), for
        # which index-based alignment would collapse all labels to one class.
        X, y, _ = load_features_and_targets(subdir, target_column)

        # Encode labels. The label SET is a known property of the problem
        # (not a fitted statistic), so fitting on the full y so every class
        # is represented is not leakage; each split is transformed with it.
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # --- Leakage-safe protocol ---
        # Split FIRST on the RAW features, before any fitted preprocessing,
        # so no test-set statistic influences imputation / variance filtering.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=random_seed
        )

        # Column selection (all-NaN / >=50%-NaN) computed on the TRAIN split
        # only, then the SAME mask is applied to both splits -> the decision
        # which columns to drop comes solely from train (no test statistic).
        keep_mask = compute_nan_column_mask(X_train, max_nan_fraction=0.5)
        X_train = X_train.loc[:, keep_mask]
        X_test = X_test.loc[:, keep_mask]

        # Impute (median) + variance filter fit on TRAIN only, then transform
        # both. No scaler here: TPOT's linear search space adds its own
        # scalers/selectors internally, trained on its train-CV folds only.
        preprocessing = build_preprocessing_pipeline(scale=False)
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.transform(X_test)

        # Initialize and train the TPOT classifier
        tpot = TPOTClassifier(
            search_space="linear",
            max_time_mins=args.max_time_mins,
            max_eval_time_mins=args.max_eval_time_mins,
            population_size=args.population_size,
            cv=args.cv,
            scorers="balanced_accuracy",
            verbose=1,
            random_state=random_seed,
            n_jobs=args.n_jobs,
            early_stop=args.early_stop,
            periodic_checkpoint_folder=export_dir / f"tpot_checkpoints_{package_name}",
        )
        tpot.fit(X_train, y_train)

        # Export the best pipeline
        export_path = export_dir / f"{package_name}_tpot_pipeline.txt"
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(str(tpot.fitted_pipeline_))

        # Export score results to a CSV file
        score_results = {
            "dataset": args.dataset,
            "run": args.run if args.run is not None else "",
            "seed": random_seed,
            "package": package_name,
            "accuracy": accuracy_score(y_test, tpot.predict(X_test)),
            "balanced_accuracy": balanced_accuracy_score(y_test, tpot.predict(X_test)),
            "f1_score": f1_score(y_test, tpot.predict(X_test), average="weighted"),
        }
        score_df = pd.DataFrame([score_results])
        score_export_path = export_dir / "tpot_scores.csv"
        if not score_export_path.exists():
            score_df.to_csv(score_export_path, index=False, sep=";", decimal=",")
        else:
            score_df.to_csv(
                score_export_path, index=False, sep=";", decimal=",", mode="a", header=False
            )

        print("Best pipeline found:", tpot.fitted_pipeline_)
        print("Test accuracy:", accuracy_score(y_test, tpot.predict(X_test)))
        print("Test balanced accuracy:", balanced_accuracy_score(y_test, tpot.predict(X_test)))
        print("Test F1 score:", f1_score(y_test, tpot.predict(X_test), average="weighted"))


if __name__ == "__main__":
    main()
