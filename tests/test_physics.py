"""Property tests grounded in classical aerodynamics.

Two flavors:
1. Closed-form identity checks on the thin-airfoil-theory baseline.
2. Empirical sanity checks on the NeuralFoil-generated dataset — symmetric
   airfoils should have ~zero lift at zero α, the linear-regime lift slope
   should be near 2π/rad, etc. These both validate the dataset *and* signal
   aerospace literacy.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aerosurrogate.dataset import DATASET_PATH, load_dataset
from aerosurrogate.physics import (
    thin_airfoil_cl,
    thin_airfoil_cl_slope_per_deg,
    thin_airfoil_zero_lift_alpha,
)

# ---- closed-form identity checks ----------------------------------------

def test_symmetric_airfoil_zero_lift_at_alpha_zero():
    """Thin airfoil theory: a symmetric (zero-camber) airfoil makes no lift at α=0."""
    assert thin_airfoil_cl(0.0, max_camber=0.0) == pytest.approx(0.0)


def test_lift_slope_is_two_pi_per_radian():
    """dC_L/dα = 2π / rad ≈ 0.1097 / deg — one of the most-tested identities in aero."""
    assert thin_airfoil_cl_slope_per_deg() == pytest.approx(2 * math.pi / 180.0)


def test_cambered_airfoil_has_negative_zero_lift_alpha():
    """Positive max camber shifts the zero-lift α negative (lift at α=0)."""
    assert thin_airfoil_zero_lift_alpha(max_camber=0.02) < 0
    assert thin_airfoil_zero_lift_alpha(max_camber=0.0) == pytest.approx(0.0)


def test_thin_airfoil_cl_is_linear_in_alpha():
    """Symbolic check: CL(α=4°) ≈ 2 · CL(α=2°) for a symmetric airfoil."""
    cl4 = float(thin_airfoil_cl(4.0))
    cl2 = float(thin_airfoil_cl(2.0))
    assert cl4 == pytest.approx(2.0 * cl2, rel=1e-12)


def test_thin_airfoil_cl_vectorizes():
    """Accepts array-like input and returns matching-shape array."""
    alphas = np.linspace(-4, 4, 9)
    out = thin_airfoil_cl(alphas, max_camber=0.0)
    assert out.shape == alphas.shape
    # Symmetric airfoil: should be exactly odd-symmetric around α=0
    np.testing.assert_allclose(out, -out[::-1], atol=1e-12)


# ---- empirical sanity checks on NeuralFoil-generated dataset ------------

requires_dataset = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason=f"dataset not present at {DATASET_PATH}; run scripts/build_dataset.py",
)


@requires_dataset
def test_symmetric_naca_airfoils_have_near_zero_cl_at_low_alpha():
    """Empirical: symmetric airfoils at small |α| should produce small CL.

    The dataset samples α at a fixed grid (none exactly at 0°), so we check
    the nearest-to-zero band: at |α| ≤ 1.5° a symmetric airfoil should have
    |CL| < 0.2 — anything larger would mean the data pipeline got camber
    detection wrong, or NeuralFoil is reporting wildly off values.
    """
    df = load_dataset()
    sym = df[(df["max_camber"] < 1e-4) & (df["alpha_deg"].abs() <= 1.5)]
    if sym.empty:
        pytest.skip("no symmetric airfoil rows at |α| ≤ 1.5° in cached dataset")
    mean_abs_cl = float(sym["CL"].abs().mean())
    assert mean_abs_cl < 0.2, (
        f"mean |CL| at |α|≤1.5° for symmetric airfoils = {mean_abs_cl:.4f}"
    )


@requires_dataset
def test_linear_regime_lift_slope_near_thin_airfoil_value():
    """Empirical dC_L/dα in the linear regime should be ~0.09–0.12 per degree.

    Thin airfoil theory predicts exactly 2π/rad ≈ 0.110. Real viscous,
    finite-thickness airfoils sit slightly below that — typically 0.09–0.11.
    A median fit across all airfoils outside that range would indicate a
    data-pipeline problem (wrong units, bad solver settings, etc.).
    """
    df = load_dataset()
    linear = df[df["alpha_deg"].abs() < 4]
    slopes: list[float] = []
    for _, grp in linear.groupby("airfoil"):
        if len(grp) < 4:
            continue
        slope, _ = np.polyfit(grp["alpha_deg"], grp["CL"], 1)
        slopes.append(float(slope))
    assert slopes, "no airfoils had >=4 samples in the linear regime"
    median_slope = float(np.median(slopes))
    assert 0.08 < median_slope < 0.13, (
        f"median dC_L/dα = {median_slope:.4f} /deg, expected 0.08–0.13"
    )


@requires_dataset
def test_drag_is_strictly_positive():
    """Sanity: every drag coefficient should be > 0. Negative CD = numerical bug."""
    df = load_dataset()
    assert (df["CD"] > 0).all(), (
        f"{(df['CD'] <= 0).sum()} rows have CD <= 0; check NeuralFoil settings"
    )


@requires_dataset
def test_drag_bucket_for_symmetric_airfoils():
    """Symmetric airfoils: CD should be minimized near α=0 (drag bucket)."""
    df = load_dataset()
    sym = df[df["max_camber"] < 1e-4]
    if sym.empty:
        pytest.skip("no symmetric airfoils in cached dataset")
    # Bin by |α| and check mean CD increases with |α|
    sym = sym.assign(abs_alpha=sym["alpha_deg"].abs())
    near = sym[sym["abs_alpha"] < 2.0]["CD"].mean()
    far = sym[sym["abs_alpha"] > 8.0]["CD"].mean()
    assert pd.notna(near) and pd.notna(far), "missing α bins"
    assert near < far, (
        f"CD near α=0 ({near:.4f}) should be less than CD at |α|>8° ({far:.4f})"
    )
