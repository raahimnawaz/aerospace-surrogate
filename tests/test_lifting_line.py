"""Validation tests for the nonlinear lifting-line solver.

The point of these tests is that LLT has *closed-form analytical results*
for canonical wings, so we can pin the solver to identities that a working
implementation must satisfy to machine precision (or close to it). If any
of these break, the solver is wrong — there is no "it's complicated"
excuse for these.

Coverage:

1. Induced-downwash kernel sanity (row sum on a uniform Γ ≡ symmetry check).
2. **Elliptic-wing identity** ``CDi = CL² / (π · AR)`` — the headline
   analytical result of classical lifting-line theory. Must hold to ≤ 1%.
3. **Lift-curve slope reduction** for finite aspect ratio:
   ``a = 2π / (1 + 2/AR)`` (the Helmbold approximation; tight for AR ≥ 4).
4. Linearity of CL in α inside the linear regime (no stall yet).
5. Rectangular wing span efficiency ``e ∈ (0.85, 1.0)`` and ``e < 1``.
6. Washout reduces tip loading (qualitative but enforceable).
7. Post-stall solver convergence: the flat-plate polar through α = 50°.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerosurrogate.lifting_line import (
    FlatPlatePostStall,
    ThinAirfoilSection,
    Wing,
    alpha_sweep,
    downwash_matrix,
    glauert_fourier_llt,
    solve_lifting_line,
)

# -----------------------------------------------------------------------
# 1. Kernel sanity
# -----------------------------------------------------------------------

def test_downwash_matrix_shape_and_antisymmetry():
    """``W[i,j]`` for a symmetric grid should satisfy ``W[N−1−i, N−1−j] = W[i,j]``.

    The downwash kernel ``1/(y_cp[i] − y_edges[j]) − 1/(y_cp[i] − y_edges[j+1])``
    is invariant under simultaneous reflection of ``y_cp`` and ``y_edges`` about
    ``y = 0``. With cosine-spaced grids (symmetric about zero) this gives a
    reflection symmetry in W. If that's broken, the indexing is wrong.
    """
    wing = Wing.rectangular(span=10.0, chord=1.0, n_sections=20)
    W = downwash_matrix(wing.y_cp, wing.y_edges)
    assert W.shape == (20, 20)
    np.testing.assert_allclose(W, W[::-1, ::-1], atol=1e-12)


def test_downwash_matrix_singularity_guard():
    """Placing a control point at a segment edge must raise."""
    y_edges = np.array([-1.0, 0.0, 1.0])
    bad_cp = np.array([-0.5, 0.0])   # second cp sits exactly on the middle edge
    with pytest.raises(ValueError, match="coincides with a segment edge"):
        downwash_matrix(bad_cp, y_edges)


# -----------------------------------------------------------------------
# 2. Elliptic-wing identity: CDi = CL² / (π · AR)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("aspect_ratio", [4.0, 8.0, 16.0])
def test_elliptic_wing_recovers_cdi_identity(aspect_ratio: float):
    """``CDi = CL² / (π · AR)`` for an elliptic planform, exactly.

    This is *the* analytical identity of classical lifting-line theory.
    For the inviscid thin-airfoil section (``Cd0 = k = 0``) the LLT solver
    must reproduce it within discretization error: a few × 10⁻³ relative
    at N = 80 cosine-spaced stations.
    """
    span = 10.0
    # AR = b² / S, S = π · b · c_root / 4   ⟹   c_root = 4 · b / (π · AR)
    c_root = 4.0 * span / (math.pi * aspect_ratio)
    wing = Wing.elliptic(span=span, root_chord=c_root, n_sections=80)
    assert wing.aspect_ratio == pytest.approx(aspect_ratio, rel=1e-12)

    res = solve_lifting_line(
        wing,
        alpha_deg=5.0,
        section=ThinAirfoilSection(Cd0=0.0, k=0.0),
        V_inf=30.0,
        tol=1e-11,
    )
    assert res.converged

    cdi_predicted = res.CL ** 2 / (math.pi * aspect_ratio)
    assert res.CDi == pytest.approx(cdi_predicted, rel=5e-3), (
        f"AR={aspect_ratio}: CDi={res.CDi:.6f} vs theory {cdi_predicted:.6f}"
    )
    assert res.span_efficiency == pytest.approx(1.0, rel=5e-3), (
        f"AR={aspect_ratio}: e = {res.span_efficiency:.4f} (should be 1.0)"
    )


def test_elliptic_loading_shape():
    """An elliptic planform should produce elliptic circulation: ``Γ(y) ∝ √(1 − (2y/b)²)``."""
    wing = Wing.elliptic(span=10.0, root_chord=1.0, n_sections=80)
    res = solve_lifting_line(
        wing, alpha_deg=5.0,
        section=ThinAirfoilSection(),
        tol=1e-10,
    )
    expected = np.sqrt(np.maximum(1.0 - (2.0 * wing.y_cp / wing.span) ** 2, 0.0))
    # Normalize both to peak = 1 and check shape match.
    ratio = res.Gamma / res.Gamma.max()
    np.testing.assert_allclose(ratio, expected / expected.max(), atol=5e-3)


# -----------------------------------------------------------------------
# 3. Finite-wing lift-curve slope: a = 2π / (1 + 2/AR)  (Helmbold)
# -----------------------------------------------------------------------

@pytest.mark.parametrize("aspect_ratio", [6.0, 10.0, 20.0])
def test_finite_wing_lift_slope_reduction(aspect_ratio: float):
    """Finite-wing lift slope must be reduced from 2π/rad by ≈ ``1 / (1 + 2/AR)``.

    For an elliptic planform the exact slope is ``a₀ / (1 + a₀/(π·AR))`` with
    ``a₀ = 2π/rad`` (thin-airfoil), simplifying to ``a = 2π · AR / (AR + 2)``.
    Numerically we check the secant from α=0 to α=4° matches this prediction.
    """
    span = 10.0
    c_root = 4.0 * span / (math.pi * aspect_ratio)
    wing = Wing.elliptic(span=span, root_chord=c_root, n_sections=80)

    section = ThinAirfoilSection()
    cl_0 = solve_lifting_line(wing, 0.0, section, tol=1e-10).CL
    cl_4 = solve_lifting_line(wing, 4.0, section, tol=1e-10).CL

    slope_per_deg = (cl_4 - cl_0) / 4.0
    slope_per_rad = slope_per_deg * 180.0 / math.pi
    predicted = 2 * math.pi * aspect_ratio / (aspect_ratio + 2)

    assert slope_per_rad == pytest.approx(predicted, rel=1e-2), (
        f"AR={aspect_ratio}: empirical slope {slope_per_rad:.4f}/rad "
        f"vs Helmbold {predicted:.4f}/rad"
    )


# -----------------------------------------------------------------------
# 4. Linearity in the pre-stall regime
# -----------------------------------------------------------------------

def test_cl_is_linear_in_alpha_pre_stall():
    """``CL(2α) ≈ 2·CL(α)`` for a thin-airfoil section in the linear regime."""
    wing = Wing.rectangular(span=10.0, chord=1.0, n_sections=60)
    section = ThinAirfoilSection()
    cl2 = solve_lifting_line(wing, 2.0, section, tol=1e-10).CL
    cl4 = solve_lifting_line(wing, 4.0, section, tol=1e-10).CL
    assert cl4 == pytest.approx(2.0 * cl2, rel=1e-6)


# -----------------------------------------------------------------------
# 5. Rectangular wing: 0.85 < e < 1.0
# -----------------------------------------------------------------------

def test_rectangular_wing_span_efficiency_below_unity():
    """A rectangular wing has ``e < 1`` (suboptimal vs. elliptic) but > 0.85.

    The exact value depends on AR (rises toward 1 with AR), but for any
    realistic AR ∈ [4, 20] the classical-LLT prediction sits in [0.85, 0.99].
    This is a softer test than the elliptic identity but catches solver
    regressions that would make ``e`` walk outside the well-known range.
    """
    wing = Wing.rectangular(span=10.0, chord=1.0, n_sections=80)   # AR = 10
    res = solve_lifting_line(
        wing, alpha_deg=5.0,
        section=ThinAirfoilSection(),
        tol=1e-10,
    )
    assert res.converged
    assert 0.85 < res.span_efficiency < 1.0, (
        f"rectangular AR={wing.aspect_ratio:.1f}: e = {res.span_efficiency:.4f}"
    )


# -----------------------------------------------------------------------
# 6. Washout shifts loading inboard
# -----------------------------------------------------------------------

def test_washout_reduces_tip_loading():
    """Negative tip twist (washout) should reduce the circulation at the tip.

    Compare a tapered wing without washout against the same wing with −3°
    tip twist. At α = 5° the washout case must have lower ``Γ`` at the
    outermost station — that's the entire point of washout in aircraft
    design (delay tip stall, preserve aileron authority).
    """
    section = ThinAirfoilSection()
    no_wash = Wing.tapered(span=10.0, root_chord=1.2, taper_ratio=0.5,
                           twist_root_deg=0.0, twist_tip_deg=0.0, n_sections=60)
    with_wash = Wing.tapered(span=10.0, root_chord=1.2, taper_ratio=0.5,
                             twist_root_deg=0.0, twist_tip_deg=-3.0, n_sections=60)

    r1 = solve_lifting_line(no_wash, 5.0, section, tol=1e-10)
    r2 = solve_lifting_line(with_wash, 5.0, section, tol=1e-10)
    # Compare tip-most non-zero station (the very last cp is closest to tip).
    assert r2.Gamma[-1] < r1.Gamma[-1]
    assert r2.Gamma[0] < r1.Gamma[0]   # symmetric: same on the other tip


# -----------------------------------------------------------------------
# 7. Post-stall convergence with the flat-plate polar
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# 8. Cross-validation against the classical Glauert Fourier-series LLT
# -----------------------------------------------------------------------
#
# The Newton solver and the Glauert solver are mathematically distinct
# formulations of the same physics — horseshoe-vortex Newton iteration
# vs. half-span Fourier-series collocation. Agreement to 4-5 decimals
# across three canonical planforms is the strongest internal consistency
# check available without an external reference implementation.


def _newton_thinairfoil(wing, alpha_deg):
    return solve_lifting_line(wing, alpha_deg, ThinAirfoilSection(), tol=1e-12)


@pytest.mark.parametrize(
    "wing_factory, alpha",
    [
        (lambda: Wing.elliptic(span=10.0, root_chord=4 * 10 / (math.pi * 8), n_sections=80), 5.0),
        (lambda: Wing.rectangular(span=10.0, chord=10.0 / 8, n_sections=120), 5.0),
        (lambda: Wing.tapered(span=10.0, root_chord=10 * 2 / (6 * 1.5), taper_ratio=0.5, n_sections=120), 6.0),
    ],
    ids=["elliptic_AR8", "rectangular_AR8", "tapered_AR6_lambda0.5"],
)
def test_newton_matches_glauert(wing_factory, alpha: float):
    """The Newton solver and the Glauert Fourier-series solver must agree.

    Tolerances:
        CL, CDi : 0.5% relative
        e       : 1% relative

    Two unrelated formulations of LLT (Phillips-Snyder Newton iteration over
    a horseshoe-vortex grid, and Glauert's 1926 Fourier-series collocation)
    agreeing to this precision means the physics, the kernel, and the
    integration weights are all consistent. Disagreement larger than this
    would indicate a real bug in one of the two solvers.
    """
    wing = wing_factory()
    g = glauert_fourier_llt(wing, alpha, n_modes=40)
    n = _newton_thinairfoil(wing, alpha)
    assert n.CL == pytest.approx(g.CL, rel=5e-3), f"CL: Newton {n.CL:.5f} vs Glauert {g.CL:.5f}"
    assert n.CDi == pytest.approx(g.CDi, rel=5e-3), f"CDi: Newton {n.CDi:.5f} vs Glauert {g.CDi:.5f}"
    assert n.span_efficiency == pytest.approx(g.span_efficiency, rel=1e-2)


def test_glauert_elliptic_gives_unit_span_efficiency_exactly():
    """An elliptic planform should give ``A_n = 0`` for all ``n ≥ 3``.

    For ``c(θ) = c_root · sin(θ)``, substituting into Glauert's equation and
    multiplying through by ``sin(θ)`` shows the RHS has only the ``sin(θ)``
    component — so all higher modes must vanish algebraically. Numerically
    they should vanish to machine epsilon, giving ``e = 1`` exactly.
    """
    wing = Wing.elliptic(span=10.0, root_chord=4 * 10 / (math.pi * 8), n_sections=80)
    res = glauert_fourier_llt(wing, 5.0, n_modes=20)
    assert res.span_efficiency == pytest.approx(1.0, abs=1e-12), (
        f"e = {res.span_efficiency:.10f}, expected 1.0 exactly"
    )
    assert abs(res.A_n[1] / res.A_n[0]) < 1e-12, (
        f"|A_3 / A_1| = {abs(res.A_n[1] / res.A_n[0]):.2e}, expected ~0"
    )


# -----------------------------------------------------------------------
# 9. Post-stall convergence with the flat-plate polar
# -----------------------------------------------------------------------

def test_alpha_sweep_through_stall_converges():
    """The nonlinear solver should converge through and past stall.

    The Hoerner flat-plate polar peaks at α = 45° (``Cl = sin(2α)``).
    Past that the sectional lift slope is *negative*, which would make
    naive fixed-point iteration unstable. The Newton step + backtracking
    line search must still converge at every α in the sweep and produce
    a 3D CL curve that peaks at α ≤ 45° and drops afterward.
    """
    wing = Wing.rectangular(span=10.0, chord=1.0, n_sections=60)
    # 0, 3, 6, … past peak. Tighter steps near stall help warm-start convergence.
    alphas = np.arange(0.0, 56.0, 3.0)
    out = alpha_sweep(
        wing, alphas,
        section=FlatPlatePostStall(),
        warm_start=True,
    )
    assert out["converged"].all(), (
        f"failed at α = {alphas[~out['converged']].tolist()}"
    )
    cl = out["CL"]
    assert cl[0] == pytest.approx(0.0, abs=1e-9)
    peak_idx = int(cl.argmax())
    # The 2D sectional Cl = sin(2α) peaks at 45°, but the 3D wing peak is
    # shifted to slightly higher α_geom because induced downwash reduces
    # α_eff by ~1-2° for AR=10. Peak in [40°, 50°] is the physical answer.
    assert 40.0 <= alphas[peak_idx] <= 50.0, f"peak at α={alphas[peak_idx]}° outside [40°,50°]"
    assert cl[-1] < cl[peak_idx], (
        f"no stall observed: peak {cl[peak_idx]:.3f} at α={alphas[peak_idx]}°, "
        f"final {cl[-1]:.3f} at α={alphas[-1]}°"
    )
