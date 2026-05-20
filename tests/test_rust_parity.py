"""Python ↔ Rust parity tests for the lifting-line solver.

The Rust port (`aerosurrogate_rs`, built from `rust/aerosurrogate-py` via
PyO3 + maturin) must reproduce the Python reference solver's outputs to
within `1e-10` on every wing configuration in the suite. Floating-point
order-of-operations differences between numpy and nalgebra produce ≤1e-14
disagreement on the same algorithm; this tolerance is generous against
that baseline.

Skipped automatically if `aerosurrogate_rs` is not installed (covers
contributors who haven't run `maturin develop` yet).

Mirror of the Python reference tests in `tests/test_lifting_line.py`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerosurrogate.lifting_line import (
    FlatPlatePostStall as PyFlatPlate,
)
from aerosurrogate.lifting_line import (
    ThinAirfoilSection as PyThinAirfoil,
)
from aerosurrogate.lifting_line import (
    Wing as PyWing,
)
from aerosurrogate.lifting_line import (
    alpha_sweep as py_alpha_sweep,
)
from aerosurrogate.lifting_line import (
    glauert_fourier_llt as py_glauert,
)
from aerosurrogate.lifting_line import (
    solve_lifting_line as py_solve,
)

rs = pytest.importorskip(  # noqa: E402 — must come after package import for ruff's import sorting
    "aerosurrogate_rs",
    reason="install the Rust wheel with: cd rust/aerosurrogate-py && maturin develop --release",
)

TOL = 1e-10


def _check_scalar(name: str, py_val: float, rs_val: float, tol: float = TOL) -> None:
    diff = abs(py_val - rs_val)
    assert diff < tol, f"{name}: |Δ| = {diff:.3e} (py={py_val:.12e}, rs={rs_val:.12e})"


# -----------------------------------------------------------------------
# Wing geometry parity
# -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "factory_args",
    [
        ("rectangular", dict(span=10.0, chord=1.0, n_sections=80)),
        ("elliptic", dict(span=10.0, root_chord=1.5915, n_sections=80)),
        ("tapered", dict(span=10.0, root_chord=1.2, taper_ratio=0.5, n_sections=80)),
    ],
)
def test_wing_geometry_matches_python(factory_args: tuple[str, dict]) -> None:
    name, kwargs = factory_args
    py_wing = getattr(PyWing, name)(**kwargs)
    rs_wing = getattr(rs.Wing, name)(**kwargs)
    _check_scalar("span", py_wing.span, rs_wing.span, tol=1e-15)
    _check_scalar("area", py_wing.area, rs_wing.area, tol=1e-15)
    _check_scalar("AR", py_wing.aspect_ratio, rs_wing.aspect_ratio, tol=1e-12)
    np.testing.assert_allclose(py_wing.chord_cp, rs_wing.chord_cp, atol=1e-14)
    np.testing.assert_allclose(py_wing.y_cp, rs_wing.y_cp, atol=1e-14)
    np.testing.assert_allclose(py_wing.y_edges, rs_wing.y_edges, atol=1e-14)


# -----------------------------------------------------------------------
# Solver scalar parity across the canonical wings
# -----------------------------------------------------------------------

def _ellip_ar(ar: float, span: float = 10.0):
    c_root = 4.0 * span / (math.pi * ar)
    return (
        PyWing.elliptic(span=span, root_chord=c_root, n_sections=80),
        rs.Wing.elliptic(span=span, root_chord=c_root, n_sections=80),
    )


@pytest.mark.parametrize("aspect_ratio", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("alpha_deg", [-2.0, 0.0, 2.0, 5.0, 8.0, 12.0])
def test_parity_elliptic_thinairfoil(aspect_ratio: float, alpha_deg: float) -> None:
    py_wing, rs_wing = _ellip_ar(aspect_ratio)
    py_res = py_solve(py_wing, alpha_deg, PyThinAirfoil(), tol=1e-12)
    rs_res = rs.solve_lifting_line(rs_wing, alpha_deg, rs.ThinAirfoilSection(), tol=1e-12)
    _check_scalar("CL", py_res.CL, rs_res.CL)
    _check_scalar("CDi", py_res.CDi, rs_res.CDi)
    _check_scalar("CD_profile", py_res.CD_profile, rs_res.CD_profile)
    _check_scalar("CD", py_res.CD, rs_res.CD)
    # Span efficiency is undefined when CDi → 0 (the α = 0 case for a
    # symmetric thin-airfoil section). Both sides correctly emit NaN; skip
    # the comparison there because `|NaN − NaN| = NaN`.
    if math.isfinite(py_res.span_efficiency) and math.isfinite(rs_res.span_efficiency):
        _check_scalar("e", py_res.span_efficiency, rs_res.span_efficiency, tol=1e-9)
    else:
        assert math.isnan(py_res.span_efficiency) and math.isnan(rs_res.span_efficiency)


@pytest.mark.parametrize("alpha_deg", [0.0, 3.0, 6.0])
def test_parity_rectangular_thinairfoil(alpha_deg: float) -> None:
    py_wing = PyWing.rectangular(span=10.0, chord=1.0, n_sections=80)
    rs_wing = rs.Wing.rectangular(span=10.0, chord=1.0, n_sections=80)
    py_res = py_solve(py_wing, alpha_deg, PyThinAirfoil(), tol=1e-12)
    rs_res = rs.solve_lifting_line(rs_wing, alpha_deg, rs.ThinAirfoilSection(), tol=1e-12)
    _check_scalar("CL", py_res.CL, rs_res.CL)
    _check_scalar("CDi", py_res.CDi, rs_res.CDi)


def test_parity_tapered_with_washout() -> None:
    """Washout case — exercise the analytical twist function on both sides."""
    py_wing = PyWing.tapered(
        span=10.0, root_chord=1.2, taper_ratio=0.5,
        twist_root_deg=0.0, twist_tip_deg=-3.0, n_sections=80,
    )
    rs_wing = rs.Wing.tapered(
        span=10.0, root_chord=1.2, taper_ratio=0.5,
        twist_root_deg=0.0, twist_tip_deg=-3.0, n_sections=80,
    )
    py_res = py_solve(py_wing, 5.0, PyThinAirfoil(), tol=1e-12)
    rs_res = rs.solve_lifting_line(rs_wing, 5.0, rs.ThinAirfoilSection(), tol=1e-12)
    _check_scalar("CL", py_res.CL, rs_res.CL)
    _check_scalar("CDi", py_res.CDi, rs_res.CDi)
    np.testing.assert_allclose(py_res.Gamma, rs_res.Gamma, atol=1e-10)


# -----------------------------------------------------------------------
# Post-stall sweep parity
# -----------------------------------------------------------------------

def test_parity_flat_plate_alpha_sweep() -> None:
    py_wing = PyWing.rectangular(span=10.0, chord=1.0, n_sections=60)
    rs_wing = rs.Wing.rectangular(span=10.0, chord=1.0, n_sections=60)
    alphas = np.arange(0.0, 51.0, 3.0)
    py_out = py_alpha_sweep(py_wing, alphas, PyFlatPlate(), warm_start=True)
    rs_out = rs.alpha_sweep(rs_wing, alphas, rs.FlatPlatePostStall(), warm_start=True)
    np.testing.assert_allclose(py_out["CL"], rs_out["CL"], atol=TOL)
    np.testing.assert_allclose(py_out["CDi"], rs_out["CDi"], atol=TOL)
    assert list(py_out["converged"]) == list(rs_out["converged"])


# -----------------------------------------------------------------------
# Glauert parity
# -----------------------------------------------------------------------

def test_parity_glauert_elliptic() -> None:
    py_wing = PyWing.elliptic(span=10.0, root_chord=4 * 10 / (math.pi * 8), n_sections=80)
    rs_wing = rs.Wing.elliptic(span=10.0, root_chord=4 * 10 / (math.pi * 8), n_sections=80)
    py_g = py_glauert(py_wing, 5.0, n_modes=20)
    rs_g = rs.glauert_fourier_llt(rs_wing, 5.0, n_modes=20)
    _check_scalar("CL", py_g.CL, rs_g.CL, tol=1e-12)
    _check_scalar("CDi", py_g.CDi, rs_g.CDi, tol=1e-12)
    _check_scalar("e", py_g.span_efficiency, rs_g.span_efficiency, tol=1e-12)


def test_parity_glauert_rectangular() -> None:
    py_wing = PyWing.rectangular(span=10.0, chord=10.0 / 8.0, n_sections=120)
    rs_wing = rs.Wing.rectangular(span=10.0, chord=10.0 / 8.0, n_sections=120)
    py_g = py_glauert(py_wing, 5.0, n_modes=40)
    rs_g = rs.glauert_fourier_llt(rs_wing, 5.0, n_modes=40)
    _check_scalar("CL", py_g.CL, rs_g.CL, tol=1e-12)
    _check_scalar("CDi", py_g.CDi, rs_g.CDi, tol=1e-12)
