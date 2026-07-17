"""
Test suite for the group/time-aware split logic.

Covers the pure split-generation logic (``iter_splits``) with synthetic groups
and the year-attachment helper (``attach_year``) with a monkeypatched
``enumerate_files``. The split logic is dataset-agnostic, so no raw Bosch data is
required to test it; the real-data end-to-end run is validated separately via the
``scripts/group_aware_evaluation.py`` smoke run.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the repo root and the scripts/ folder importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import group_aware_evaluation as gae  # noqa: E402


def _make_groups(n_per_machine=10):
    """Build a synthetic aligned (groups, y) pair.

    3 machines x n_per_machine rows = 30 rows; 5 operations and 3 years cycled
    across the rows; balanced 2-class labels.
    """
    machines = ["M01", "M02", "M03"]
    operations = [f"OP0{i}" for i in range(5)]
    years = [2019, 2020, 2021]
    n = len(machines) * n_per_machine
    machine = []
    for m in machines:
        machine.extend([m] * n_per_machine)
    operation = [operations[i % len(operations)] for i in range(n)]
    year = [years[i % len(years)] for i in range(n)]
    groups = pd.DataFrame({"machine": machine, "operation": operation, "year": year})
    y = pd.Series([0, 1] * (n // 2))
    return groups, y


def test_iter_splits_random_single_stratified_split():
    """The random strategy yields exactly one stratified 80/20 split."""
    groups, y = _make_groups()
    splits = list(gae.iter_splits(groups, y, random_seed=27, strategies=["random"]))
    assert len(splits) == 1
    strategy, fold_id, train_idx, test_idx = splits[0]
    assert strategy == "random"
    assert fold_id == ""
    assert len(train_idx) + len(test_idx) == len(groups)
    assert len(test_idx) == round(0.2 * len(groups))  # 6 of 30
    assert len(train_idx) == 24


def test_iter_splits_leave_machine_out_three_folds():
    """LeaveOneGroupOut over machine yields 3 folds, one per held-out machine."""
    groups, y = _make_groups()
    splits = list(gae.iter_splits(groups, y, random_seed=27, strategies=["leave_machine_out"]))
    assert len(splits) == 3
    held_out = set()
    for strategy, fold_id, train_idx, test_idx in splits:
        assert strategy == "leave_machine_out"
        test_machines = set(groups["machine"].iloc[test_idx])
        assert len(test_machines) == 1  # one machine per fold
        held_out.add(fold_id)
        # Train must exclude the held-out machine entirely.
        train_machines = set(groups["machine"].iloc[train_idx])
        assert test_machines.isdisjoint(train_machines)
        assert len(test_idx) == 10  # n_per_machine
    assert held_out == {"M01", "M02", "M03"}


def test_iter_splits_leave_operation_out_five_folds():
    """LeaveOneGroupOut over operation yields 5 folds, one per held-out operation."""
    groups, y = _make_groups()
    splits = list(gae.iter_splits(groups, y, random_seed=27, strategies=["leave_operation_out"]))
    assert len(splits) == 5
    held_out = set()
    for strategy, fold_id, train_idx, test_idx in splits:
        assert strategy == "leave_operation_out"
        test_ops = set(groups["operation"].iloc[test_idx])
        assert len(test_ops) == 1
        held_out.add(fold_id)
        train_ops = set(groups["operation"].iloc[train_idx])
        assert test_ops.isdisjoint(train_ops)
    assert held_out == {f"OP0{i}" for i in range(5)}


def test_iter_splits_temporal_train_earlier_test_latest():
    """Temporal split trains on earlier years and tests on the latest year."""
    groups, y = _make_groups()
    splits = list(gae.iter_splits(groups, y, random_seed=27, strategies=["temporal"]))
    assert len(splits) == 1
    strategy, fold_id, train_idx, test_idx = splits[0]
    assert strategy == "temporal"
    assert fold_id == "train_2019-2020__test_2021"
    # Test year must be strictly the latest; train strictly earlier.
    assert set(groups["year"].iloc[test_idx]) == {2021}
    assert set(groups["year"].iloc[train_idx]) == {2019, 2020}
    assert len(train_idx) + len(test_idx) == len(groups)


def test_iter_splits_temporal_single_year_raises():
    """A temporal split with only one distinct year is not defined -> ValueError."""
    groups, y = _make_groups()
    groups["year"] = 2021  # collapse to a single year
    with pytest.raises(ValueError, match="at least two distinct years"):
        list(gae.iter_splits(groups, y, random_seed=27, strategies=["temporal"]))


def test_iter_splits_subset_strategies():
    """Requesting a subset of strategies yields only those splits."""
    groups, y = _make_groups()
    splits = list(gae.iter_splits(groups, y, random_seed=27, strategies=["random", "temporal"]))
    strategies = {s[0] for s in splits}
    assert strategies == {"random", "temporal"}
    assert len(splits) == 2  # 1 random + 1 temporal


def test_attach_year_matches_when_enumeration_aligns(monkeypatch):
    """attach_year adds the parsed year when the re-enumeration matches targets."""
    targets = pd.DataFrame(
        {"machine": ["M01", "M02"], "operation": ["OP00", "OP01"], "label": ["good", "bad"]}
    )
    file_df = pd.DataFrame(
        {
            "machine": ["M01", "M02"],
            "operation": ["OP00", "OP01"],
            "label": ["good", "bad"],
            "filename": ["M01_Aug_2019_OP00_000.h5", "M02_Sep_2020_OP01_001.h5"],
            "path": ["p1", "p2"],
            "year": [2019, 2020],
        }
    )
    monkeypatch.setattr(gae, "enumerate_files", lambda d: file_df)
    out = gae.attach_year(targets, Path("dummy"))
    assert list(out.columns) == ["machine", "operation", "label", "year"]
    assert list(out["year"]) == [2019, 2020]
    assert out.index.tolist() == [0, 1]  # clean RangeIndex


def test_attach_year_raises_on_mismatch(monkeypatch):
    """attach_year raises if the re-enumerated metadata does not match targets."""
    targets = pd.DataFrame(
        {"machine": ["M01", "M02"], "operation": ["OP00", "OP01"], "label": ["good", "bad"]}
    )
    # Divergent machine/operation -> must not silently misalign the year.
    file_df = pd.DataFrame(
        {
            "machine": ["M01", "M03"],
            "operation": ["OP00", "OP01"],
            "label": ["good", "bad"],
            "filename": ["f1", "f2"],
            "path": ["p1", "p2"],
            "year": [2019, 2020],
        }
    )
    monkeypatch.setattr(gae, "enumerate_files", lambda d: file_df)
    with pytest.raises(ValueError, match="do not match"):
        gae.attach_year(targets, Path("dummy"))
