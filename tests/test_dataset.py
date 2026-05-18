"""Smoke tests for dataset loading + split_by_airfoil correctness."""
from __future__ import annotations

import pandas as pd
import pytest

from aerosurrogate.dataset import DATASET_PATH, load_dataset, split_by_airfoil
from aerosurrogate.models import FEATURES, TARGETS

requires_dataset = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason=f"dataset not present at {DATASET_PATH}",
)


@requires_dataset
def test_dataset_has_required_columns():
    df = load_dataset()
    required = {"airfoil", *FEATURES, *TARGETS}
    assert required.issubset(df.columns), f"missing columns: {required - set(df.columns)}"


@requires_dataset
def test_dataset_nonempty_and_has_variety():
    df = load_dataset()
    assert len(df) > 100
    assert df["airfoil"].nunique() >= 5, "need multiple airfoils for split_by_airfoil to be meaningful"


@requires_dataset
def test_split_by_airfoil_holds_out_entire_shapes():
    """Critical invariant: no airfoil name should appear in both train AND test."""
    df = load_dataset()
    train, test, _ = split_by_airfoil(df, test_frac=0.2, seed=42)
    overlap = set(train["airfoil"]) & set(test["airfoil"])
    assert not overlap, f"airfoils leaked into both sets: {overlap}"


@requires_dataset
def test_split_seed_is_deterministic():
    df = load_dataset()
    _, test1, names1 = split_by_airfoil(df, test_frac=0.2, seed=7)
    _, test2, names2 = split_by_airfoil(df, test_frac=0.2, seed=7)
    assert names1 == names2
    pd.testing.assert_frame_equal(
        test1.reset_index(drop=True), test2.reset_index(drop=True)
    )
