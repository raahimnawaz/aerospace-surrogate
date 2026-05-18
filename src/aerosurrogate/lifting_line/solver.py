"""Nonlinear lifting-line solver for a planar, unswept finite-span wing.

Combines a 2D sectional polar (any :class:`SectionalAero`) with the
classical induced-downwash kernel from :mod:`biot_savart` to give a
viscous, post-stall-capable wing solver. The form follows the *modern*
(i.e. iterated nonlinear) lifting-line theory of Phillips & Snyder,
*Journal of Aircraft* 37 (4), 2000, specialized here to the planar /
unswept / no-dihedral case so the bound vortex induces no in-plane
velocity and the kernel reduces to the 1-D Cauchy form.

Why nonlinear LLT? Classical (linear) LLT assumes ``Cl = a₀ · (α − α_{L=0})``
with constant lift-curve slope ``a₀``. That breaks once any section enters
stall: ``Cl`` is no longer linear in ``α``. Nonlinear LLT lets the
sectional polar be arbitrary — including stalled — and finds the
self-consistent circulation distribution by Newton iteration. With a
viscous sectional polar (NeuralFoil, XFOIL, wind-tunnel data) it produces
3D wing predictions that include profile drag, induced drag, *and* stall
behavior at the cost of one N×N linear solve per iteration.

Governing system
----------------

At every spanwise station ``i`` the bound circulation ``Γ_i`` must be
consistent with the 2D Kutta-Joukowski relation::

    F_i(Γ) ≡ Γ_i − ½ · V_∞ · c_i · Cl_section(α_eff_i, Re_i)   =   0    (*)

where ``α_eff_i = α_∞ + θ_i − α_induced_i`` and ``α_induced_i = w_i / V_∞``
with ``w = W · Γ`` the downwash from :func:`biot_savart.downwash_matrix`.

Newton iteration on the nonlinear system (*) uses the Jacobian::

    J_ij = ∂F_i/∂Γ_j = δ_ij + ½ · c_i · a_i · W[i,j]

where ``a_i = dCl/dα`` evaluated at the current ``α_eff_i`` (in 1/rad).
We obtain ``a_i`` by a one-sided finite difference on the sectional polar
so this works with *any* :class:`SectionalAero` — even a black-box
NeuralFoil. Per iteration we solve one N×N linear system; convergence is
quadratic when the polar is smooth and stays robust through stall with a
single line-search safeguard.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .biot_savart import downwash_matrix
from .geometry import Wing
from .sections import SectionalAero


@dataclass
class LiftingLineResult:
    """Output of :func:`solve_lifting_line`.

    All spanwise arrays are sampled at the wing's control points.

    Attributes
    ----------
    Gamma
        Bound circulation ``Γ(y)``, [m²/s], shape ``(N,)``.
    alpha_eff_deg
        Effective angle of attack at each section after subtracting
        induced downwash, degrees.
    alpha_induced_deg
        Induced (downwash) angle at each section, degrees.
    cl_local
        Sectional lift coefficient at the converged ``α_eff``.
    cd_local
        Sectional profile drag coefficient at the converged ``α_eff``.
    CL
        Total wing lift coefficient: ``CL = (2 / (S V_∞)) · ∫ Γ dy``.
    CDi
        Induced drag coefficient: ``CDi = (2 / (S V_∞²)) · ∫ Γ w_i dy``.
    CD_profile
        Profile (viscous-section) drag coefficient: ``(1/S) · ∫ Cd_2D · c · dy``.
    CD
        Total drag coefficient ``CDi + CD_profile``.
    span_efficiency
        Oswald-style span efficiency ``e = CL² / (π · AR · CDi)``. Equals 1
        for an elliptically loaded wing (the validation identity).
    converged
        ``True`` iff the Newton residual fell below the tolerance.
    iterations
        Number of Newton steps performed.
    residual
        Final residual ``‖F(Γ)‖∞``.
    """
    Gamma: NDArray[np.float64]
    alpha_eff_deg: NDArray[np.float64]
    alpha_induced_deg: NDArray[np.float64]
    cl_local: NDArray[np.float64]
    cd_local: NDArray[np.float64]
    CL: float
    CDi: float
    CD_profile: float
    CD: float
    span_efficiency: float
    converged: bool
    iterations: int
    residual: float


def _spanwise_integral(values: NDArray[np.float64], y_edges: NDArray[np.float64]) -> float:
    """Midpoint-rule integral of a control-point-sampled field across the span.

    Each control point carries its entire segment width::

        ∫ f(y) dy ≈ Σ_i f_i · (y_edges[i+1] − y_edges[i])

    For cosine-spaced grids this matches the horseshoe-vortex discretization
    the solver implicitly assumes.
    """
    widths = np.diff(y_edges)
    return float(np.sum(values * widths))


def _eval_residual(
    Gamma: NDArray[np.float64],
    alpha_geom_deg: NDArray[np.float64],
    chord: NDArray[np.float64],
    W: NDArray[np.float64],
    section: SectionalAero,
    V_inf: float,
    Re_arr: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate ``F(Γ) = Γ − ½·V·c·Cl(α_eff)`` and the intermediate fields."""
    w_i = W @ Gamma
    alpha_induced_deg = np.degrees(w_i / V_inf)
    alpha_eff_deg = alpha_geom_deg - alpha_induced_deg
    cl_local = np.asarray(section.cl(alpha_eff_deg, Re_arr), dtype=np.float64)
    F = Gamma - 0.5 * V_inf * chord * cl_local
    return F, w_i, alpha_eff_deg, cl_local


def solve_lifting_line(
    wing: Wing,
    alpha_deg: float,
    section: SectionalAero,
    V_inf: float = 30.0,
    Re_ref: float = 1_000_000.0,
    max_iter: int = 100,
    tol: float = 1e-10,
    Gamma_init: NDArray[np.float64] | None = None,
    fd_step_deg: float = 1e-3,
) -> LiftingLineResult:
    """Solve the nonlinear LLT system for a planar wing at a single ``α``.

    Solves ``F(Γ) = Γ − ½·V·c·Cl_section(α_eff(Γ)) = 0`` by Newton iteration
    with a finite-difference Jacobian. A simple backtracking line search keeps
    the iteration stable across stall, where the local lift-curve slope can
    flatten or invert.

    Parameters
    ----------
    wing
        :class:`Wing` instance (cosine-spaced spanwise discretization).
    alpha_deg
        Wing root angle of attack, degrees. The wing's geometric twist is
        added per-section to get each section's angle to freestream.
    section
        2D sectional polar implementing :class:`SectionalAero`.
    V_inf
        Freestream speed, m/s. The dimensional scale matters only for ``Γ``;
        ``CL``, ``CDi``, and ``e`` are independent of it.
    Re_ref
        Reference Reynolds number passed to the sectional polar. The
        analytical sections ignore it; the NeuralFoil section uses it.
    max_iter
        Maximum Newton iterations.
    tol
        Convergence threshold on ``‖F(Γ)‖∞`` relative to ``max(‖Γ‖∞, 1)``.
    Gamma_init
        Optional initial guess, shape ``(N,)``. If ``None``, starts from the
        2D estimate ``Γ(y) = ½ V c(y) Cl_section(α_geom, Re_ref)``.
    fd_step_deg
        Finite-difference step for the local lift-slope ``dCl/dα``, in
        degrees. ``1e-3`` is small enough to be accurate on smooth polars
        and large enough to dodge floating-point noise on noisy ones.

    Returns
    -------
    :class:`LiftingLineResult`
    """
    n = wing.n_sections
    W = downwash_matrix(wing.y_cp, wing.y_edges)
    chord = wing.chord_cp
    twist = wing.twist_deg_cp
    alpha_geom_section = alpha_deg + twist                    # (N,)
    Re_arr = np.full(n, Re_ref)
    eye = np.eye(n)

    # Initial guess: 2D sectional estimate (no induced effects).
    if Gamma_init is None:
        cl0 = np.asarray(section.cl(alpha_geom_section, Re_arr), dtype=np.float64)
        Gamma = 0.5 * V_inf * chord * cl0
    else:
        Gamma = np.array(Gamma_init, dtype=np.float64, copy=True)
        if Gamma.shape != (n,):
            raise ValueError(f"Gamma_init must have shape ({n},); got {Gamma.shape}")

    converged = False
    iters = 0
    fd_step_rad = np.deg2rad(fd_step_deg)

    for iters in range(1, max_iter + 1):  # noqa: B007  (iters is consumed below)
        F, w_i, alpha_eff, cl_local = _eval_residual(
            Gamma, alpha_geom_section, chord, W, section, V_inf, Re_arr
        )
        residual = float(np.max(np.abs(F)) / max(float(np.max(np.abs(Gamma))), 1.0))
        if residual < tol:
            converged = True
            break

        # Local lift-slope a_i = dCl/dα|_α_eff_i  (in 1/rad), one-sided FD.
        cl_plus = np.asarray(section.cl(alpha_eff + fd_step_deg, Re_arr), dtype=np.float64)
        a_local = (cl_plus - cl_local) / fd_step_rad           # (N,)

        # Jacobian J = I + diag(½·c·a) · W
        J = eye + (0.5 * chord * a_local)[:, None] * W

        try:
            dGamma = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            # Stagnant: nudge with a damped gradient step and continue.
            dGamma = 0.1 * F

        # Backtracking line search: shrink step until residual decreases.
        # Robust across stall where the linear model overshoots.
        step = 1.0
        F_norm = float(np.linalg.norm(F))
        for _bt in range(8):
            Gamma_trial = Gamma - step * dGamma
            F_trial, *_ = _eval_residual(
                Gamma_trial, alpha_geom_section, chord, W, section, V_inf, Re_arr
            )
            if np.all(np.isfinite(F_trial)) and float(np.linalg.norm(F_trial)) < F_norm:
                Gamma = Gamma_trial
                break
            step *= 0.5
        else:
            # No step decreased the residual; accept the smallest step anyway
            # so iteration keeps moving (typical in deep stall with shallow slope).
            Gamma = Gamma - step * dGamma

    # Final post-processing using the converged Γ.
    _, w_i, alpha_eff, cl_local = _eval_residual(
        Gamma, alpha_geom_section, chord, W, section, V_inf, Re_arr
    )
    cd_local = np.asarray(section.cd(alpha_eff, Re_arr), dtype=np.float64)

    S = wing.area
    CL = (2.0 / (S * V_inf)) * _spanwise_integral(Gamma, wing.y_edges)
    CDi = (2.0 / (S * V_inf * V_inf)) * _spanwise_integral(Gamma * w_i, wing.y_edges)
    CD_profile = (1.0 / S) * _spanwise_integral(cd_local * chord, wing.y_edges)
    CD = CDi + CD_profile

    AR = wing.aspect_ratio
    if CDi > 1e-12:
        e = (CL * CL) / (np.pi * AR * CDi)
    else:
        e = float("nan")

    F_final, *_ = _eval_residual(
        Gamma, alpha_geom_section, chord, W, section, V_inf, Re_arr
    )
    final_residual = float(np.max(np.abs(F_final)) / max(float(np.max(np.abs(Gamma))), 1.0))

    return LiftingLineResult(
        Gamma=Gamma,
        alpha_eff_deg=alpha_eff,
        alpha_induced_deg=alpha_geom_section - alpha_eff,
        cl_local=cl_local,
        cd_local=cd_local,
        CL=float(CL),
        CDi=float(CDi),
        CD_profile=float(CD_profile),
        CD=float(CD),
        span_efficiency=float(e),
        converged=converged,
        iterations=iters,
        residual=final_residual,
    )


def alpha_sweep(
    wing: Wing,
    alpha_deg_grid: NDArray[np.float64],
    section: SectionalAero,
    V_inf: float = 30.0,
    Re_ref: float = 1_000_000.0,
    warm_start: bool = True,
    **solver_kwargs,
) -> dict[str, NDArray[np.float64]]:
    """Run :func:`solve_lifting_line` across a grid of angles of attack.

    With ``warm_start=True`` each solve is initialized from the previous
    α's converged ``Γ``, which dramatically reduces iteration counts when
    sweeping through stall.

    Returns a dict of ``(M,)``-shaped arrays: ``alpha_deg``, ``CL``, ``CDi``,
    ``CD_profile``, ``CD``, ``span_efficiency``, ``converged`` (bool),
    ``iterations``.
    """
    alpha_grid = np.atleast_1d(np.asarray(alpha_deg_grid, dtype=np.float64))
    CLs = np.empty_like(alpha_grid)
    CDis = np.empty_like(alpha_grid)
    CD_profs = np.empty_like(alpha_grid)
    CDs = np.empty_like(alpha_grid)
    es = np.empty_like(alpha_grid)
    converged = np.empty(alpha_grid.size, dtype=bool)
    iters = np.empty(alpha_grid.size, dtype=np.int64)

    Gamma_prev: NDArray[np.float64] | None = None
    for k, a in enumerate(alpha_grid):
        res = solve_lifting_line(
            wing,
            float(a),
            section,
            V_inf=V_inf,
            Re_ref=Re_ref,
            Gamma_init=Gamma_prev if warm_start else None,
            **solver_kwargs,
        )
        CLs[k] = res.CL
        CDis[k] = res.CDi
        CD_profs[k] = res.CD_profile
        CDs[k] = res.CD
        es[k] = res.span_efficiency
        converged[k] = res.converged
        iters[k] = res.iterations
        if warm_start:
            Gamma_prev = res.Gamma

    return {
        "alpha_deg": alpha_grid,
        "CL": CLs,
        "CDi": CDis,
        "CD_profile": CD_profs,
        "CD": CDs,
        "span_efficiency": es,
        "converged": converged,
        "iterations": iters,
    }
