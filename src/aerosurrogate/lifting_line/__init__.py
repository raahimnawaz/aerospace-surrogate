"""3D finite-wing analysis by nonlinear lifting-line theory.

This subpackage extends the 2D-airfoil scope of ``aerosurrogate`` to a
finite, planar, unswept wing. It pairs *any* 2D sectional polar — the
repo's thin-airfoil theory baseline, a trained sklearn surrogate, or
NeuralFoil itself — with classical Prandtl-style induced-downwash
analysis to compute total wing lift, profile drag, and induced drag.

The point is to remove the ceiling that a pure 2D surrogate puts on
the project. A NeuralFoil-only pipeline can never report induced drag —
which is 30-50% of total drag at cruise for a high-aspect-ratio wing.
Wrapping NeuralFoil inside a lifting-line solver gives a *3D* answer
without giving up the 2D viscous fidelity NeuralFoil provides.

Public API
----------

* :class:`Wing` — planar, unswept wing with cosine-spaced spanwise grid.
  Factory methods ``Wing.rectangular``, ``Wing.elliptic``, ``Wing.tapered``.
* :class:`SectionalAero` — protocol for any 2D polar ``Cl(α, Re)``, ``Cd(α, Re)``.
* :class:`ThinAirfoilSection` — closed-form thin-airfoil polar.
* :class:`FlatPlatePostStall` — Hoerner flat-plate polar with analytic stall.
* :class:`NeuralFoilSection` — NeuralFoil airfoil polar (optional dep).
* :func:`solve_lifting_line` — single-α nonlinear LLT solve.
* :func:`alpha_sweep` — warm-started α-sweep returning ``CL``, ``CDi``, ``CD``, ``e``.
* :class:`LiftingLineResult` — output dataclass.
* :func:`downwash_matrix` — N×N induced-downwash influence matrix.

The headline validation identity — ``CDi = CL² / (π · AR)`` on an
elliptic wing — is enforced as a unit test in ``tests/test_lifting_line.py``.
"""
from __future__ import annotations

from .biot_savart import downwash_matrix
from .geometry import Wing
from .sections import (
    FlatPlatePostStall,
    NeuralFoilSection,
    SectionalAero,
    ThinAirfoilSection,
)
from .solver import LiftingLineResult, alpha_sweep, solve_lifting_line

__all__ = [
    "FlatPlatePostStall",
    "LiftingLineResult",
    "NeuralFoilSection",
    "SectionalAero",
    "ThinAirfoilSection",
    "Wing",
    "alpha_sweep",
    "downwash_matrix",
    "solve_lifting_line",
]
