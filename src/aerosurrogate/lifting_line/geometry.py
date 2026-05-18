"""Wing geometry for planar, unswept lifting-line analysis.

A ``Wing`` discretizes a finite-span planar wing into ``N`` spanwise stations
with cosine spacing — denser at the tips where the loading varies fastest.
Each station has a chord ``c(y)`` and a geometric twist ``θ(y)``; per-station
sectional aerodynamics are supplied separately via the ``sections`` module.

Cosine spacing places the i-th control point at::

    y_cp[i] = −(b/2) · cos((i + 0.5) · π / N)        for i = 0 … N−1

with segment edges between adjacent control points (and tips clamped to
``±b/2``). This is the standard LLT grid: it concentrates resolution where
``dΓ/dy`` is largest and avoids placing a control point at the tip
singularity.

Factory methods cover the three canonical planforms used to validate the
solver:

* :py:meth:`Wing.rectangular` — constant chord. Classical LLT predicts
  span efficiency ``e ≈ 0.88 – 0.96`` depending on aspect ratio.
* :py:meth:`Wing.elliptic` — chord ``c(y) = c_root · √(1 − (2y/b)²)``.
  Yields the analytical identity ``CDi = CL² / (π · AR)`` exactly, i.e.
  ``e = 1``. This is *the* test case for any LLT implementation.
* :py:meth:`Wing.tapered` — linear taper from root to tip, optionally with
  linear washout (twist varying linearly with span).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Wing:
    """A planar, unswept finite-span wing for lifting-line analysis.

    All quantities are pre-sampled at cosine-spaced control points; the
    object is frozen so the discretization can't drift after construction.
    Prefer the factory classmethods (``rectangular``, ``elliptic``, ``tapered``)
    over direct construction.

    Attributes
    ----------
    span
        Total wingspan ``b`` (tip to tip), in meters.
    area
        Reference planform area ``S = ∫ c(y) dy``, in square meters. Stored
        rather than recomputed so the analytical exact area for each factory
        planform is preserved (avoiding small trapezoidal-rule error).
    chord_cp
        Chord at each control point, shape ``(N,)``.
    twist_deg_cp
        Geometric twist at each control point, in degrees (positive = leading
        edge up), shape ``(N,)``.
    y_cp
        Control-point spanwise locations, shape ``(N,)``.
    y_edges
        Segment edges (control points lie at segment midpoints), shape ``(N+1,)``.
    """
    span: float
    area: float
    chord_cp: NDArray[np.float64]
    twist_deg_cp: NDArray[np.float64]
    y_cp: NDArray[np.float64]
    y_edges: NDArray[np.float64]
    name: str = field(default="custom")

    @property
    def n_sections(self) -> int:
        return int(self.chord_cp.size)

    @property
    def aspect_ratio(self) -> float:
        """``AR = b² / S``."""
        return float(self.span ** 2 / self.area)

    @property
    def mean_chord(self) -> float:
        """``c̄ = S / b``."""
        return float(self.area / self.span)

    # ---- factory constructors -------------------------------------------

    @classmethod
    def _make_grid(cls, span: float, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cosine-spaced control points and matching segment edges.

        Control points are at the midpoints of ``N`` cosine-spaced intervals
        in the half-angle ``θ ∈ [0, π]`` with ``y = −(b/2) cos θ``. Segment
        edges in θ-space are equispaced, giving cosine-clustered edges in
        y-space. Tips are clamped to exactly ``±b/2``.
        """
        if n < 4:
            raise ValueError(f"n_sections must be >= 4 for a meaningful LLT grid; got {n}")
        # control points: midpoints of N equispaced θ-intervals
        theta_cp = np.pi * (np.arange(n) + 0.5) / n
        y_cp = -(span / 2.0) * np.cos(theta_cp)
        # edges: N+1 equispaced θ values
        theta_edges = np.pi * np.arange(n + 1) / n
        y_edges = -(span / 2.0) * np.cos(theta_edges)
        # snap tips exactly (cos may give -1.0 or +1.0 with FP error)
        y_edges[0] = -span / 2.0
        y_edges[-1] = +span / 2.0
        return y_cp, y_edges

    @classmethod
    def rectangular(
        cls,
        span: float,
        chord: float,
        twist_deg: float = 0.0,
        n_sections: int = 60,
    ) -> "Wing":
        """Constant-chord, constant-twist rectangular wing.

        Reference area is exact: ``S = b · c``.
        """
        y_cp, y_edges = cls._make_grid(span, n_sections)
        return cls(
            span=span,
            area=span * chord,
            chord_cp=np.full(n_sections, chord, dtype=np.float64),
            twist_deg_cp=np.full(n_sections, twist_deg, dtype=np.float64),
            y_cp=y_cp,
            y_edges=y_edges,
            name="rectangular",
        )

    @classmethod
    def elliptic(
        cls,
        span: float,
        root_chord: float,
        twist_deg: float = 0.0,
        n_sections: int = 60,
    ) -> "Wing":
        """Elliptic planform with ``c(y) = c_root · √(1 − (2y/b)²)``.

        Reference area is exact: ``S = π · b · c_root / 4``.
        Classical LLT predicts span efficiency ``e = 1`` for this planform,
        which is the cornerstone analytical check of the solver.
        """
        y_cp, y_edges = cls._make_grid(span, n_sections)
        chord = root_chord * np.sqrt(np.maximum(1.0 - (2.0 * y_cp / span) ** 2, 0.0))
        return cls(
            span=span,
            area=np.pi * span * root_chord / 4.0,
            chord_cp=chord,
            twist_deg_cp=np.full(n_sections, twist_deg, dtype=np.float64),
            y_cp=y_cp,
            y_edges=y_edges,
            name="elliptic",
        )

    @classmethod
    def tapered(
        cls,
        span: float,
        root_chord: float,
        taper_ratio: float,
        twist_root_deg: float = 0.0,
        twist_tip_deg: float = 0.0,
        n_sections: int = 60,
    ) -> "Wing":
        """Linearly tapered wing with optional linear washout.

        ``taper_ratio = c_tip / c_root``. Twist varies linearly from root
        (at ``y=0``) to tip (at ``|y|=b/2``); negative ``twist_tip_deg``
        relative to root gives washout (tip stalls last).

        Reference area is exact: ``S = b · c_root · (1 + λ) / 2``.
        """
        if taper_ratio <= 0:
            raise ValueError(f"taper_ratio must be > 0; got {taper_ratio}")
        y_cp, y_edges = cls._make_grid(span, n_sections)
        eta = np.abs(2.0 * y_cp / span)   # 0 at root, 1 at tip
        chord = root_chord * (1.0 - (1.0 - taper_ratio) * eta)
        twist = twist_root_deg + (twist_tip_deg - twist_root_deg) * eta
        return cls(
            span=span,
            area=span * root_chord * (1.0 + taper_ratio) / 2.0,
            chord_cp=chord,
            twist_deg_cp=twist,
            y_cp=y_cp,
            y_edges=y_edges,
            name=f"tapered(λ={taper_ratio:g})",
        )
