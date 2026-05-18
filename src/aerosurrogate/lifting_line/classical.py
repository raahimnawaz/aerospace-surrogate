"""Classical Glauert Fourier-series lifting-line theory.

This is the textbook formulation of Prandtl-Glauert LLT: assume a linear
sectional polar ``Cl = a₀·(α − α_{L=0})``, expand the circulation as a
Fourier sine series in the transformed coordinate ``θ = arccos(−2y/b)``,
and collocate at ``M`` points to get an ``M × M`` linear system for the
Fourier coefficients ``{A_n}``.

The whole machinery sits in roughly 60 lines because the linearity assumption
on the sectional polar collapses the nonlinear LLT system into a single
linear solve. It exists in this package for one reason: as an **independent
cross-validation** of the Newton-iteration solver in :mod:`solver`. The
two formulations are mathematically distinct — Fourier-series collocation
vs. horseshoe-vortex discretization with Newton iteration — but for a
linear sectional polar they must agree to within discretization error.
That agreement is enforced as a unit test in
``tests/test_lifting_line.py::test_newton_matches_glauert_*``.

References
----------
* Glauert, H., *The Elements of Aerofoil and Airscrew Theory*, Cambridge
  Univ. Press, 1926/1948, §11 ("The Monoplane Aerofoil").
* Anderson, J. D., *Fundamentals of Aerodynamics*, 6th ed., McGraw-Hill,
  2017, §5.3 ("Prandtl's Classical Lifting-Line Theory").
* Bertin & Cummings, *Aerodynamics for Engineers*, 6th ed., Pearson, 2014,
  §7.4 ("Lifting-Line Theory for Unswept Wings").
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import Wing


@dataclass
class GlauertResult:
    """Output of :func:`glauert_fourier_llt`.

    Attributes
    ----------
    A_n
        Fourier coefficients of the circulation series, ``Γ(θ) = 2bV Σ A_n sin(nθ)``,
        for ``n = 1, 3, 5, …, 2M−1`` (odd-only for symmetric loading).
    n_indices
        The odd-mode indices ``n``, shape ``(M,)``.
    CL
        Total wing lift coefficient: ``CL = π · AR · A_1``.
    CDi
        Induced drag coefficient: ``CDi = π · AR · Σ n · A_n²``.
    span_efficiency
        Span efficiency ``e = A_1² / Σ n·A_n² = 1 / (1 + Σ_{n≥3} n·(A_n/A_1)²)``.
        Equals 1 for an elliptic planform (all higher modes vanish).
    """
    A_n: NDArray[np.float64]
    n_indices: NDArray[np.int64]
    CL: float
    CDi: float
    span_efficiency: float


def glauert_fourier_llt(
    wing: Wing,
    alpha_deg: float,
    lift_slope_per_rad: float = 2.0 * math.pi,
    alpha_L0_deg: float = 0.0,
    n_modes: int = 30,
) -> GlauertResult:
    """Solve classical Prandtl-Glauert LLT by Fourier-series collocation.

    For a planar, unswept wing with linear sectional lift slope ``a₀`` and
    zero-lift angle ``α_{L=0}`` (both constant along span), expand the
    bound circulation as a sine series in ``θ = arccos(−2y/b)``::

        Γ(θ) = 2 · b · V_∞ · Σ_{n odd} A_n · sin(n·θ)

    Substituting into the fundamental LLT equation and collocating at ``M``
    points uniformly distributed in ``θ`` gives the linear system::

        Σ_n A_n · [ 4b/(a₀ · c(θ_i)) · sin(n·θ_i) + n · sin(n·θ_i)/sin(θ_i) ]
            = α_geom(θ_i) − α_{L=0}                      (rad)

    Only odd ``n`` (1, 3, 5, …) appear in the symmetric-loading case, which
    covers any wing without an asymmetric twist or chord distribution. This
    function assumes symmetric loading.

    Parameters
    ----------
    wing
        :class:`Wing` instance. The chord and twist distributions are
        sampled at the Glauert collocation points by linear interpolation
        from the wing's stored control-point values.
    alpha_deg
        Wing root angle of attack, degrees.
    lift_slope_per_rad
        Linear sectional lift-curve slope ``a₀ = dCl/dα``, in 1/rad.
        Default ``2π`` (thin-airfoil theory). Real viscous airfoils sit
        around ``5.7 − 6.1`` /rad.
    alpha_L0_deg
        Zero-lift angle of attack, degrees (negative for positive camber).
    n_modes
        Number of Fourier modes retained. ``30`` is overkill for the wings
        in this package (errors saturate around N=10-15) but is cheap.

    Returns
    -------
    :class:`GlauertResult`
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1; got {n_modes}")

    # Odd mode indices: n = 1, 3, 5, …, 2M-1
    n_indices = np.arange(1, 2 * n_modes, 2, dtype=np.int64)
    M = len(n_indices)

    # Collocation in θ ∈ (0, π/2) only. For symmetric (odd-mode-only) loading
    # the points θ and π − θ produce identical equations — collocating across
    # the full (0, π) range would make the system rank-deficient and force
    # ``np.linalg.solve`` to make a numerical choice between the duplicates,
    # introducing spurious nonzero coefficients in the higher modes. The
    # right-half-only placement gives M unique equations for M unknowns.
    theta = (np.arange(M) + 0.5) * (np.pi / 2.0) / M
    y = -(wing.span / 2.0) * np.cos(theta)

    # Chord and twist at each collocation point — evaluated analytically if
    # the wing was built by a factory method, otherwise linearly interpolated
    # from the control-point grid.
    chord_at = wing.chord_at(y)
    twist_at = wing.twist_at(y)

    # Build the M × M linear system A · x = b, where x_j = A_{n_j}.
    # Row i (collocation θ_i):
    #     Σ_j A_{n_j} · [ 4b/(a₀ · c_i) · sin(n_j · θ_i)  +  n_j · sin(n_j · θ_i) / sin(θ_i) ]
    #     = α_geom(θ_i) − α_{L=0}
    n_grid = n_indices[None, :]                                 # (1, M)
    theta_col = theta[:, None]                                  # (M, 1)
    chord_col = chord_at[:, None]                               # (M, 1)
    sin_nth = np.sin(n_grid * theta_col)                        # (M, M)
    A_matrix = (
        4.0 * wing.span / (lift_slope_per_rad * chord_col) * sin_nth
        + n_grid * sin_nth / np.sin(theta_col)
    )

    rhs = np.deg2rad(alpha_deg + twist_at - alpha_L0_deg)       # (M,)
    A_n = np.linalg.solve(A_matrix, rhs).astype(np.float64)     # (M,)

    AR = wing.aspect_ratio
    CL = math.pi * AR * float(A_n[0])
    CDi = math.pi * AR * float(np.sum(n_indices * A_n ** 2))
    if abs(A_n[0]) > 1e-12:
        e = 1.0 / (1.0 + float(np.sum(n_indices[1:] * (A_n[1:] / A_n[0]) ** 2)))
    else:
        e = float("nan")

    return GlauertResult(
        A_n=A_n,
        n_indices=n_indices,
        CL=CL,
        CDi=CDi,
        span_efficiency=e,
    )
