"""Sectional 2D aerodynamic polars for the lifting-line solver.

The lifting-line solver closes its nonlinear system by querying, at every
spanwise station and every iteration, a 2D sectional polar ``Cl(α, Re)`` and
``Cd(α, Re)``. The solver does not care where the polar comes from — only
that it implements the :class:`SectionalAero` protocol.

This module ships three implementations spanning the project's data sources:

* :class:`ThinAirfoilSection` — closed-form ``Cl = 2π(α − α_{L=0})`` from
  ``aerosurrogate.physics``, with an optional parabolic profile-drag model
  ``Cd = Cd0 + k·(Cl − Cl_min)²``. Inviscid in spirit. Used to validate
  the solver against the elliptic-wing analytical identity
  ``CDi = CL² / (π · AR)`` (which holds exactly only when the profile-drag
  contribution is zeroed out).
* :class:`FlatPlatePostStall` — Hoerner ``Cl = sin(2α)`` / ``Cd = 2 sin²(α)``.
  This is *not* a thin-airfoil model in the linear regime, but it has a
  smooth analytical stall at ``α = 45°``, which lets the nonlinear LLT
  solver be exercised through and past stall without any external dependency.
* :class:`NeuralFoilSection` — wraps Peter Sharpe's NeuralFoil (a neural-net
  surrogate for XFOIL) so the LLT solver inherits NeuralFoil's viscous
  panel-method fidelity. This is the production polar; using it here lifts
  the project from "NeuralFoil is a 2D ceiling" to "NeuralFoil sets the
  sectional polar inside a 3D wing solver."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..physics import thin_airfoil_cl, thin_airfoil_zero_lift_alpha


@runtime_checkable
class SectionalAero(Protocol):
    """Anything that returns 2D sectional ``Cl`` and ``Cd`` at given ``α``, ``Re``.

    Implementations must be vectorized: passing an array of ``α`` (in degrees)
    and a matching-shape array of ``Re`` must return an array of the same shape.
    """

    def cl(self, alpha_deg: ArrayLike, Re: ArrayLike) -> NDArray[np.float64]:
        ...

    def cd(self, alpha_deg: ArrayLike, Re: ArrayLike) -> NDArray[np.float64]:
        ...


@dataclass
class ThinAirfoilSection:
    """Closed-form thin-airfoil polar ``Cl = 2π(α − α_{L=0})``.

    Lift comes directly from ``aerosurrogate.physics.thin_airfoil_cl`` — the
    same closed-form baseline this repo benchmarks ML models against. Drag is
    modeled as a parabolic polar in ``Cl``::

        Cd = Cd0 + k · (Cl − Cl_min)²

    Setting ``Cd0 = k = 0`` gives a truly inviscid section, which the LLT
    solver uses to recover the analytical identity ``CDi = CL² / (π · AR)``
    on an elliptic wing.

    This class is the natural "physics-only" baseline for the LLT solver —
    parallel to how thin-airfoil theory is the physics-only baseline for the
    2D regression problem in ``physics.py``.

    Stall is not modeled. ``Cl`` grows linearly with ``α`` forever.
    """

    max_camber: float = 0.0
    Cd0: float = 0.0
    k: float = 0.0
    Cl_min_drag: float = 0.0

    @property
    def alpha_L0_deg(self) -> float:
        """Zero-lift angle of attack, degrees."""
        return float(thin_airfoil_zero_lift_alpha(self.max_camber))

    def cl(self, alpha_deg: ArrayLike, Re: ArrayLike | None = None) -> NDArray[np.float64]:  # noqa: ARG002
        return thin_airfoil_cl(alpha_deg, self.max_camber)

    def cd(self, alpha_deg: ArrayLike, Re: ArrayLike | None = None) -> NDArray[np.float64]:
        cl_val = self.cl(alpha_deg, Re)
        return self.Cd0 + self.k * (cl_val - self.Cl_min_drag) ** 2


@dataclass
class FlatPlatePostStall:
    """Hoerner flat-plate polar valid across the full ``[−90°, +90°]`` range::

        Cl(α) = sin(2α)               = 2 sin α cos α
        Cd(α) = Cd0 + 2 sin²(α)

    Reduces to ``Cl ≈ 2α`` (not ``2π α``) at small ``α`` — strictly worse
    than thin-airfoil theory in the linear regime. The purpose of this
    section is *post-stall behavior*: ``Cl`` peaks at ``α = 45°`` and
    decreases beyond, giving the nonlinear LLT solver a sectional polar
    that genuinely stalls so we can demonstrate post-stall wing behavior
    (drop in ``CL``, redistribution of loading toward the unstalled portions).

    Reference: Hoerner, *Fluid-Dynamic Lift*, ch. 4.
    """

    Cd0: float = 0.0

    def cl(self, alpha_deg: ArrayLike, Re: ArrayLike | None = None) -> NDArray[np.float64]:  # noqa: ARG002
        a = np.deg2rad(np.asarray(alpha_deg, dtype=np.float64))
        return np.sin(2.0 * a)

    def cd(self, alpha_deg: ArrayLike, Re: ArrayLike | None = None) -> NDArray[np.float64]:  # noqa: ARG002
        a = np.deg2rad(np.asarray(alpha_deg, dtype=np.float64))
        return self.Cd0 + 2.0 * np.sin(a) ** 2


class NeuralFoilSection:
    """Wrap a NeuralFoil airfoil (XFOIL-grade viscous polar) as a SectionalAero.

    NeuralFoil [#nf]_ is a neural-net surrogate for XFOIL: it returns viscous,
    boundary-layer-resolved 2D sectional aerodynamics in microseconds. Using
    it as the polar inside this lifting-line solver is the whole point of
    the LLT extension — it couples a data-driven 2D viscous model with a
    classical 3D inviscid wing theory, giving a viscous-3D total drag buildup
    cheaper than any panel method.

    Both ``neuralfoil`` and ``aerosandbox`` are *optional* dependencies; the
    imports are lazy so the rest of this package works without them. Install
    with ``pip install -e ".[build]"``.

    Parameters
    ----------
    airfoil_name
        Airfoil identifier resolvable by ``aerosandbox.Airfoil``. Examples:
        ``"naca2412"``, ``"naca0012"``, ``"clarky"``, ``"e387"``.
    model_size
        NeuralFoil model size — one of ``"xxsmall"``, ``"xsmall"``, ``"small"``,
        ``"medium"``, ``"large"``, ``"xlarge"``, ``"xxlarge"``, ``"xxxlarge"``.
        Larger = more accurate, marginally slower. ``"medium"`` is a good default.
    n_crit
        Critical amplification factor (transition criterion). ``9`` is
        standard for clean wind tunnels; ``5`` for noisy / dirty conditions.

    .. [#nf] Sharpe, P. D., "NeuralFoil." MIT-licensed.
       https://github.com/peterdsharpe/NeuralFoil
    """

    def __init__(
        self,
        airfoil_name: str = "naca0012",
        model_size: str = "medium",
        n_crit: float = 9.0,
    ):
        self.airfoil_name = airfoil_name
        self.model_size = model_size
        self.n_crit = float(n_crit)
        # Build the Airfoil once at construction; reuse for every query.
        try:
            import aerosandbox as asb  # noqa: F401
            import neuralfoil  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "NeuralFoilSection requires the optional 'neuralfoil' and "
                "'aerosandbox' packages. Install with: "
                "pip install -e \".[build]\""
            ) from e
        self._airfoil = asb.Airfoil(airfoil_name)

    def _query(self, alpha_deg: ArrayLike, Re: ArrayLike) -> dict:
        import neuralfoil
        a = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
        r = np.broadcast_to(np.asarray(Re, dtype=np.float64), a.shape).copy()
        return neuralfoil.get_aero_from_airfoil(
            airfoil=self._airfoil,
            alpha=a,
            Re=r,
            n_crit=self.n_crit,
            model_size=self.model_size,
        )

    def cl(self, alpha_deg: ArrayLike, Re: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self._query(alpha_deg, Re)["CL"], dtype=np.float64)

    def cd(self, alpha_deg: ArrayLike, Re: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self._query(alpha_deg, Re)["CD"], dtype=np.float64)
