#!/usr/bin/env python3
"""
Count the number of extracted features for a given dataset, per package.

Scans a directory of per-package feature CSVs (as produced under
``results/02_Extracted_Features/<Dataset_Display_Name>/<package>/``) and prints,
for each package subdirectory, the total number of feature columns, the number of
CSV files and the (average) number of samples.

The feature CSVs are expected to be semicolon-separated with a comma decimal
separator and a first column used as the index (matching the export format of
``feature_extraction_benchmark.py``).

Example
-------
    python scripts/count_extracted_features.py \\
        --dir results/02_Extracted_Features/CNC_Mill_Tool_Wear
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Count the extracted features per package for a dataset."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help=(
            "Directory containing one subdirectory per package, each holding the "
            "package's extracted-feature CSVs (e.g. "
            "results/02_Extracted_Features/CNC_Mill_Tool_Wear)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the script."""
    args = parse_args()
    dir_to_search: Path = args.dir

    if not dir_to_search.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_to_search}")

    # Get a list of all subdirectories in the specified directory
    subdirs = [d for d in dir_to_search.iterdir() if d.is_dir()]

    for package_dir in subdirs:
        # Get a list of all csv files in the package directory
        csv_files = list(package_dir.glob("*.csv"))
        if not csv_files:
            print(f"{package_dir.name}: no CSV files found")
            continue

        # Open each csv file and count the number of features
        num_features = 0
        num_samples = 0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0, decimal=",", sep=";", header=0)
            num_features += df.shape[1]  # Count the number of columns (features)
            num_samples += df.shape[0]
        num_samples //= len(csv_files)
        # Print the package name and the number of features
        result = (
            f"{package_dir.name}: {num_features} features, "
            f"{len(csv_files)} files, {num_samples} samples"
        )
        print(result)


if __name__ == "__main__":
    main()
