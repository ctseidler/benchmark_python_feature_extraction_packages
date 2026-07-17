"""Fixed-classifier registry + leakage-safe full-pipeline builder.

Shared by the fixed-classifier baselines (``scripts/fixed_classifier_evaluation.py``)
and the group/time-aware split evaluation (``scripts/group_aware_evaluation.py``)
so both use identical fixed classifiers and identical leakage-safe
preprocessing (see :mod:`src.evaluation.preprocessing`). Centralising the
registry here avoids drift between the two scripts and keeps the classifier
definitions out of the ``scripts/`` folder (which is not an importable package).

The fixed classifiers remove the AutoML search-budget confound that the
TPOT benchmark introduces: every package and
dataset is scored with the *same*, sensible, search-free defaults, so
differences in score reflect feature quality rather than search luck.

Classifiers (each wrapped in the shared leakage-safe Pipeline):
    - RandomForest   (tree-based, no scaling needed)
    - XGBoost        (tree-based, no scaling needed)
    - LogisticRegression (scaled)
    - SVM (RBF)      (scaled)
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.evaluation.preprocessing import build_preprocessing_pipeline

__all__ = ["CLASSIFIERS", "build_full_pipeline"]


def _xgboost_factory(seed, n_jobs):
    """Build the XGBoost classifier, importing lazily.

    XGBoost is imported inside the factory so that this module can be imported
    (and the registry inspected) in environments where ``xgboost`` is not
    installed; the import only fails if the ``xgboost`` classifier is actually
    used.
    """
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=n_jobs,
        eval_metric="logloss",
        verbosity=0,
    )


# Classifier registry: name -> {factory (callable(seed, n_jobs) -> estimator),
# scale (whether the leakage-safe pipeline should add a StandardScaler)}.
#
# NOTE: sklearn>=1.8 removed ``n_jobs`` from LogisticRegression (it had no
# effect there), so the LogReg/SVM factories accept but ignore ``n_jobs``.
CLASSIFIERS = {
    "random_forest": {
        "factory": lambda seed, n_jobs: RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=n_jobs
        ),
        "scale": False,
    },
    "xgboost": {"factory": _xgboost_factory, "scale": False},
    "logreg": {
        "factory": lambda seed, n_jobs: LogisticRegression(
            max_iter=1000,
            random_state=seed,
        ),
        "scale": True,
    },
    "svm": {
        "factory": lambda seed, n_jobs: SVC(C=1.0, kernel="rbf", random_state=seed),
        "scale": True,
    },
}


def build_full_pipeline(clf_name, seed, n_jobs):
    """Build the full leakage-safe pipeline: preprocessing + fixed classifier.

    The preprocessing steps (impute -> variance filter -> optional scaler) come
    from :func:`src.evaluation.preprocessing.build_preprocessing_pipeline`; the
    classifier is appended as the final step. The returned pipeline is fit on
    the TRAIN split only and applied to the test split, which is leakage-safe.
    """
    spec = CLASSIFIERS[clf_name]
    pre = build_preprocessing_pipeline(scale=spec["scale"])
    clf = spec["factory"](seed, n_jobs)
    return Pipeline(steps=pre.steps + [("classifier", clf)])
