"""Leakage-safe preprocessing and data-loading helpers for performance evaluation.

Background
----------
The original performance benchmark applied several preprocessing steps
(drop all-NaN columns, drop >50%-NaN columns, forward/backward fill, drop
zero-variance columns) to the FULL feature matrix **before** the train/test
split. Every one of those steps uses statistics computed across the whole
dataset, including the held-out test rows, which is a form of data leakage.
Forward/backward fill in particular can impute test
values from train rows (and vice-versa).

This module implements the leakage-safe replacement:

1. Split FIRST on the raw features.
2. Compute any column-selection masks on the TRAIN split only, then apply the
   same mask to the test split (the *decision* which columns to keep comes from
   train; no test statistic is used).
3. Fit all imputation / variance filtering / scaling on the TRAIN split only
   via an :class:`sklearn.pipeline.Pipeline`, then ``transform`` both splits.

Everything here is reusable so that the TPOT benchmark, the fixed-classifier
baselines and the group/time-aware splits share one
identical, leakage-safe preprocessing protocol.
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "load_features_and_targets",
    "compute_nan_column_mask",
    "build_preprocessing_pipeline",
    "InfToNaN",
]


class InfToNaN(BaseEstimator, TransformerMixin):
    """Replace ``+inf`` / ``-inf`` with ``NaN`` (stateless, leakage-safe).

    Several feature-extraction packages (notably kats) emit ``inf`` for
    ill-conditioned statistics (e.g. division by zero). ``SimpleImputer`` only
    handles ``NaN``, so without this step sklearn rejects the matrix with
    ``ValueError: Input X contains infinity``. Placed as the FIRST pipeline
    step, before any statistic is computed, so the downstream imputer /
    variance filter / scaler never see an ``inf``.
    """

    def fit(self, X, y=None):  # noqa: D401 - stateless
        return self

    def transform(self, X):
        # Accept DataFrame or ndarray; return same type.
        if isinstance(X, pd.DataFrame):
            return X.replace([np.inf, -np.inf], np.nan)
        return np.where(np.isinf(X), np.nan, X)


def load_features_and_targets(
    data_dir: str | Path,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load the per-package feature CSVs and the targets for one package folder.

    Expects ``data_dir`` to contain one subdirectory per package, each holding
    one or more ``;``-separated feature CSVs (``decimal=","``), plus a
    ``targets.csv`` file at the top level.

    Parameters
    ----------
    data_dir : str | Path
        Directory with per-package feature subfolders and a ``targets.csv``.
    target_column : str
        Column name in ``targets.csv`` to use as the label.

    Returns
    -------
    X : pandas.DataFrame
        Concatenated feature matrix (samples x features).
    y : pandas.Series
        Target column aligned to ``X``'s index.
    package_name : str
        Name of the package subfolder.
    """
    data_dir = Path(data_dir)
    package_name = data_dir.name

    csv_files = sorted(glob(os.path.join(data_dir, "*.csv")))
    dfs = [pd.read_csv(f, sep=";", decimal=",", index_col=0) for f in csv_files]
    X = pd.concat(dfs, axis=1)

    targets = pd.read_csv(data_dir.parent / "targets.csv", sep=";", decimal=",")

    # Positional alignment: the feature CSVs and targets.csv are both written in
    # the loader's recording-enumeration order, so row i of X corresponds to row
    # i of targets. Align by POSITION rather than by the CSV index, because some
    # packages write a non-unique index (e.g. seglearn repeats a per-segment
    # index of 0 for every recording), for which index-based alignment
    # (``targets.loc[X.index]``) would collapse to a single repeated row.
    n_X, n_t = len(X), len(targets)
    if n_t >= n_X:
        targets = targets.iloc[:n_X].reset_index(drop=True)
        X = X.reset_index(drop=True)
    else:
        X = X.iloc[:n_t].reset_index(drop=True)
        targets = targets.reset_index(drop=True)

    y = targets[target_column]
    return X, y, package_name


def compute_nan_column_mask(
    X_train: pd.DataFrame,
    max_nan_fraction: float = 0.5,
) -> np.ndarray:
    """Boolean mask of columns to KEEP, computed on the TRAIN split only.

    Drops columns that are all-NaN or whose NaN fraction in ``X_train`` is
    ``>= max_nan_fraction``. The same mask must be applied to the test split so
    that the *decision* which columns to drop comes solely from train (no test
    statistic is used) — this is leakage-safe.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    max_nan_fraction : float, default 0.5
        Columns with a train NaN fraction at or above this value are dropped.

    Returns
    -------
    numpy.ndarray
        Boolean array (len = n_columns) where ``True`` means "keep".
    """
    # Treat inf as missing too: kats/tsfresh can emit inf for ill-conditioned
    # statistics; such columns should be dropped under the same fraction rule.
    nan_fraction = X_train.replace([np.inf, -np.inf], np.nan).isnull().mean(axis=0)
    keep = (nan_fraction < max_nan_fraction).to_numpy()
    return keep


def build_preprocessing_pipeline(scale: bool = False) -> Pipeline:
    """Build a leakage-safe preprocessing pipeline to fit on TRAIN only.

    Steps
    -----
    1. ``SimpleImputer(strategy="median")`` — replaces the old
       ``ffill().bfill()`` (which leaked across the train/test boundary) with a
       median imputed from the TRAIN split only.
    2. ``VarianceThreshold()`` — drops zero-variance / constant columns using
       the TRAIN variance only (replaces the old ``X.var() > 0`` on the full
       set).
    3. (optional) ``StandardScaler()`` — mean/variance from TRAIN only; required
       for distance/gradient classifiers (LogReg, SVM). Tree-based models and
       TPOT's linear search space do not need it, so ``scale`` defaults to
       ``False``.

    The returned pipeline is fit on the training split and then applied (via
    ``transform``) to both splits, so no test statistic influences any fitted
    step.

    Parameters
    ----------
    scale : bool, default False
        If ``True``, append a ``StandardScaler`` (for LR/SVM baselines).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted preprocessing pipeline.
    """
    steps = [
        ("inf_to_nan", InfToNaN()),
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
