"""Tests for the model zoo — every pipeline should fit + predict without errors."""
from __future__ import annotations

import numpy as np
import pytest

from aerosurrogate.models import FEATURES, TARGETS, build_models


def _synthetic_data(n: int = 80, n_feat: int = 11, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    # Some nonlinear target with mild noise so even Ridge gets decent R²
    y = X[:, 0] * 0.5 + X[:, 1] ** 2 + 0.1 * rng.normal(size=n)
    return X, y


def test_feature_count_matches_pipeline_input():
    """If FEATURES grows or shrinks the synthetic shape stays in sync."""
    assert len(FEATURES) == 11


def test_targets_are_three():
    assert TARGETS == ["CL", "CD", "CM"]


@pytest.mark.parametrize("name", list(build_models().keys()))
def test_each_model_fits_and_predicts(name: str):
    pipe = build_models()[name]
    X, y = _synthetic_data()
    pipe.fit(X, y)
    pred = pipe.predict(X[:5])
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
