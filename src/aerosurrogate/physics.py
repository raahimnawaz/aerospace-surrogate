"""Closed-form aerodynamic baselines (pre-CFD, pre-ML).

These are the textbook approximations that aerospace engineers have used for a
century. They're terrible at high angle of attack — the whole point of
comparing the ML models against them is to show *where* ML actually buys us
something and where the classical formula is already enough.

Reference: Anderson, *Fundamentals of Aerodynamics*, ch. 4 (thin airfoil theory).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def thin_airfoil_zero_lift_alpha(max_camber: ArrayLike) -> NDArray[np.float64]:
    """Zero-lift angle of attack (degrees) for a thin cambered airfoil.

    For a parabolic camber line ``z(x) = 4·m·x·(1−x)`` the Glauert integral
    collapses to

        α_{L=0} = −2·m          [radians]

    where ``m = max_camber / chord``. Real NACA 4-digit camber lines are
    piecewise (linear/parabolic), so this is an approximation — but it
    captures the linear-regime shift to within a few tenths of a degree for
    moderate camber.

    Accepts scalar or array `max_camber` and returns the same shape as a
    numpy array (always degrees).
    """
    return -np.degrees(2.0 * np.asarray(max_camber, dtype=np.float64))


def thin_airfoil_cl(
    alpha_deg: ArrayLike, max_camber: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Lift coefficient via thin airfoil theory.

        C_L = 2π · (α − α_{L=0})       [α in radians]

    Valid only in the linear regime (|α| roughly < 6°). Ignores viscosity,
    thickness, Reynolds number, and stall entirely. The whole point of this
    baseline is to make explicit what those omissions cost in accuracy.

    `alpha_deg` and `max_camber` broadcast together — pass arrays of equal
    length to compute per-row predictions across a dataset.
    """
    alpha = np.asarray(alpha_deg, dtype=np.float64)
    a_zero = thin_airfoil_zero_lift_alpha(max_camber)
    return 2.0 * np.pi * np.radians(alpha - a_zero)


def thin_airfoil_cl_slope_per_deg() -> float:
    """Lift curve slope ``dC_L/dα = 2π``/rad ≈ 0.1097 / deg.

    Independent of camber, Reynolds, thickness — this is the universal
    small-α prediction of thin airfoil theory and one of the most-tested
    identities in aerodynamics.
    """
    return 2.0 * np.pi / 180.0
