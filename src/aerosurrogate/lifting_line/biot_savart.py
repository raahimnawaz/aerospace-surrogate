"""Downwash induction by a horseshoe-vortex sheet on a planar lifting line.

For an unswept, planar wing aligned with the y-axis (no sweep, no dihedral),
the bound vortex of each horseshoe lies *on* the lifting line itself. A
straight vortex induces no velocity on its own axis, so at any control
point on the lifting line, the only contribution to the induced downwash
comes from the two semi-infinite trailing legs of each horseshoe.

For a horseshoe vortex with bound circulation ``Γ_j`` and spanwise endpoints
``y_edge[j]`` (left) and ``y_edge[j+1]`` (right), the induced z-velocity at
a point ``y_cp[i]`` on the lifting line — derived directly from Biot-Savart
applied to two semi-infinite vortex filaments — is::

    w_z(y_cp[i]) = −(Γ_j / 4π) · [1/(y_cp[i] − y_edge[j])
                                   − 1/(y_cp[i] − y_edge[j+1])]

Adopting the LLT sign convention where positive downwash means flow going
downward (which reduces the effective angle of attack at the section),
``w_i = −w_z``, giving the form used throughout this package::

    w_i(y_cp[i]) = (Γ_j / 4π) · [1/(y_cp[i] − y_edge[j])
                                   − 1/(y_cp[i] − y_edge[j+1])]

Summing across all horseshoes and writing as a matrix-vector product::

    w_i = W · Γ        where   W[i,j] = (1/4π) · [1/(y_cp[i] − y_edge[j])
                                                   − 1/(y_cp[i] − y_edge[j+1])]

The solver then needs only one O(N²) matrix-vector multiply per iteration.

Reference: Anderson, *Fundamentals of Aerodynamics*, §5.3 (Prandtl's Classical
Lifting-Line Theory); Phillips & Snyder, *J. Aircraft* 37 (4), 2000, for the
nonlinear iterative form built on top of this kernel.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def downwash_matrix(y_cp: NDArray[np.float64], y_edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the N×N downwash influence matrix W with ``w_i = W · Γ``.

    Parameters
    ----------
    y_cp
        Spanwise control-point locations, shape ``(N,)``. Each must lie
        strictly inside the corresponding segment ``[y_edges[i], y_edges[i+1]]``.
    y_edges
        Spanwise segment edges, shape ``(N+1,)``, strictly increasing.
        Horseshoe ``j`` extends from ``y_edges[j]`` to ``y_edges[j+1]``.

    Returns
    -------
    W : ndarray, shape ``(N, N)``
        Influence matrix. ``W[i, j]`` is the downwash induced at control
        point ``i`` per unit circulation of segment ``j``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or any control point coincides with a
        segment edge (the kernel has a 1/0 singularity there).
    """
    y_cp = np.asarray(y_cp, dtype=np.float64)
    y_edges = np.asarray(y_edges, dtype=np.float64)
    n = y_cp.size
    if y_edges.size != n + 1:
        raise ValueError(
            f"y_edges must have length N+1={n + 1}, got {y_edges.size}"
        )
    if np.any(np.diff(y_edges) <= 0):
        raise ValueError("y_edges must be strictly increasing")

    # broadcast: rows index control points, cols index segments
    dy_left = y_cp[:, None] - y_edges[None, :-1]   # (N, N)
    dy_right = y_cp[:, None] - y_edges[None, 1:]   # (N, N)

    # Control point at segment edge would hit a 1/0 — check before dividing.
    if np.any(dy_left == 0.0) or np.any(dy_right == 0.0):
        raise ValueError(
            "A control point coincides with a segment edge; "
            "use midpoint placement (e.g., cosine spacing offset by half-cell)."
        )

    return (1.0 / (4.0 * np.pi)) * (1.0 / dy_left - 1.0 / dy_right)
