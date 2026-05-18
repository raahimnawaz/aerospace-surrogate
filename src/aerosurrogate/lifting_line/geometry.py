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

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ChordTwistFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


# Named field constructors for the analytical chord/twist functions of each
# factory planform. Using closures with explicit types (rather than inline
# lambdas) keeps mypy happy and produces clearer tracebacks.
def _const_field(value: float) -> ChordTwistFn:
    """Spatially-constant field ``f(y) = value`` for every y."""
    def _f(y: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.full_like(y, value, dtype=np.float64)
    return _f


def _elliptic_chord_field(root_chord: float, span: float) -> ChordTwistFn:
    """Elliptic chord ``c(y) = c_root · √(1 − (2y/b)²)``."""
    def _f(y: NDArray[np.float64]) -> NDArray[np.float64]:
        return root_chord * np.sqrt(np.maximum(1.0 - (2.0 * y / span) ** 2, 0.0))
    return _f


def _tapered_chord_field(root_chord: float, taper: float, span: float) -> ChordTwistFn:
    """Linearly tapered chord ``c(y) = c_root · (1 − (1 − λ) · |2y/b|)``."""
    def _f(y: NDArray[np.float64]) -> NDArray[np.float64]:
        return root_chord * (1.0 - (1.0 - taper) * np.abs(2.0 * y / span))
    return _f


def _linear_twist_field(twist_root: float, twist_tip: float, span: float) -> ChordTwistFn:
    """Linear washout twist ``θ(y) = θ_root + (θ_tip − θ_root) · |2y/b|``."""
    def _f(y: NDArray[np.float64]) -> NDArray[np.float64]:
        return twist_root + (twist_tip - twist_root) * np.abs(2.0 * y / span)
    return _f


@dataclass(frozen=True)
class Wing:
    """A planar, unswept finite-span wing for lifting-line analysis.

    All quantities are pre-sampled at cosine-spaced control points; the
    object is frozen so the discretization can't drift after construction.
    Factory classmethods (``rectangular``, ``elliptic``, ``tapered``) also
    record the analytical chord and twist functions, which solvers that
    discretize differently (e.g. :func:`classical.glauert_fourier_llt`)
    can use to evaluate the planform exactly at their own collocation
    points instead of interpolating from this object's grid.

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
    # Optional analytical evaluators. When present, alternative discretizations
    # can sample the planform at their own grid without interpolating from cp.
    _chord_fn: ChordTwistFn | None = field(default=None, repr=False, compare=False)
    _twist_fn: ChordTwistFn | None = field(default=None, repr=False, compare=False)

    def chord_at(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Chord at arbitrary spanwise locations ``y``.

        Uses the analytical chord function when available (factory-built
        wings); falls back to linear interpolation from ``chord_cp`` for
        custom wings.
        """
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if self._chord_fn is not None:
            return self._chord_fn(y_arr)
        return np.interp(y_arr, self.y_cp, self.chord_cp)

    def twist_at(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Geometric twist (degrees) at arbitrary spanwise locations ``y``."""
        y_arr = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if self._twist_fn is not None:
            return self._twist_fn(y_arr)
        return np.interp(y_arr, self.y_cp, self.twist_deg_cp)

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
    ) -> Wing:
        """Constant-chord, constant-twist rectangular wing.

        Reference area is exact: ``S = b · c``.
        """
        y_cp, y_edges = cls._make_grid(span, n_sections)
        c_val = float(chord)
        t_val = float(twist_deg)
        return cls(
            span=span,
            area=span * c_val,
            chord_cp=np.full(n_sections, c_val, dtype=np.float64),
            twist_deg_cp=np.full(n_sections, t_val, dtype=np.float64),
            y_cp=y_cp,
            y_edges=y_edges,
            name="rectangular",
            _chord_fn=_const_field(c_val),
            _twist_fn=_const_field(t_val),
        )

    @classmethod
    def elliptic(
        cls,
        span: float,
        root_chord: float,
        twist_deg: float = 0.0,
        n_sections: int = 60,
    ) -> Wing:
        """Elliptic planform with ``c(y) = c_root · √(1 − (2y/b)²)``.

        Reference area is exact: ``S = π · b · c_root / 4``.
        Classical LLT predicts span efficiency ``e = 1`` for this planform,
        which is the cornerstone analytical check of the solver.
        """
        y_cp, y_edges = cls._make_grid(span, n_sections)
        b = float(span)
        cr = float(root_chord)
        t_val = float(twist_deg)
        chord = cr * np.sqrt(np.maximum(1.0 - (2.0 * y_cp / b) ** 2, 0.0))
        return cls(
            span=b,
            area=np.pi * b * cr / 4.0,
            chord_cp=chord,
            twist_deg_cp=np.full(n_sections, t_val, dtype=np.float64),
            y_cp=y_cp,
            y_edges=y_edges,
            name="elliptic",
            _chord_fn=_elliptic_chord_field(cr, b),
            _twist_fn=_const_field(t_val),
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
    ) -> Wing:
        """Linearly tapered wing with optional linear washout.

        ``taper_ratio = c_tip / c_root``. Twist varies linearly from root
        (at ``y=0``) to tip (at ``|y|=b/2``); negative ``twist_tip_deg``
        relative to root gives washout (tip stalls last).

        Reference area is exact: ``S = b · c_root · (1 + λ) / 2``.
        """
        if taper_ratio <= 0:
            raise ValueError(f"taper_ratio must be > 0; got {taper_ratio}")
        y_cp, y_edges = cls._make_grid(span, n_sections)
        b = float(span)
        cr = float(root_chord)
        lam = float(taper_ratio)
        tr = float(twist_root_deg)
        tt = float(twist_tip_deg)
        eta = np.abs(2.0 * y_cp / b)      # 0 at root, 1 at tip
        chord = cr * (1.0 - (1.0 - lam) * eta)
        twist = tr + (tt - tr) * eta
        return cls(
            span=b,
            area=b * cr * (1.0 + lam) / 2.0,
            chord_cp=chord,
            twist_deg_cp=twist,
            y_cp=y_cp,
            y_edges=y_edges,
            name=f"tapered(λ={lam:g})",
            _chord_fn=_tapered_chord_field(cr, lam, b),
            _twist_fn=_linear_twist_field(tr, tt, b),
        )
