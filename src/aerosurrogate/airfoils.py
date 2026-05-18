"""Airfoil sources: UIUC database curated list + parametric NACA 4-digit sweep.

`aerosandbox` is imported lazily so the rest of the package is usable from a
cached CSV without the heavy XFOIL/CFD stack installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aerosandbox as asb


# Curated UIUC airfoils spanning sailplane / general-aviation / low-Re ranges.
UIUC_AIRFOILS: list[str] = [
    "naca0008", "naca0012", "naca0015", "naca0018",
    "naca1408", "naca2410", "naca2412", "naca2415", "naca4412", "naca4415",
    "naca23012", "naca63012a", "naca64a010",
    "s1210", "s1223", "s2027", "s4083", "s8036", "sd7003", "sd7037", "sd7062",
    "e387", "e423", "e472",
    "clarky", "clarkysm", "goe398", "goe796",
    "rg15", "ag35", "ag36", "ag38", "ag40d", "ag45c",
    "fx63137", "fx72ls160", "mh32", "mh60",
    "ah79100c", "ah93w215", "ls417",
]


def naca4_name(m: int, p: int, t: int) -> str:
    """NACA 4-digit name from (max camber %, camber position tenths, thickness %)."""
    return f"naca{m}{p}{t:02d}"


def parametric_naca_airfoils() -> list[tuple[str, float, float]]:
    """Parametric sweep across NACA 4-digit (camber, position, thickness)."""
    out: list[tuple[str, float, float]] = []
    for m in (0, 2, 4, 6):                       # max camber %
        for p in (0, 2, 4, 6):                   # camber position (tenths)
            if m == 0 and p != 0:                # symmetric airfoils have p == 0
                continue
            for t in (8, 10, 12, 15, 18, 21):    # max thickness %
                name = naca4_name(m, p, t)
                cam_pos = p / 10.0 if m > 0 else 0.0
                out.append((name, cam_pos, t / 100.0))
    return out


def load_airfoils() -> list[tuple[str, asb.Airfoil, float]]:
    """Load all airfoil shapes as `(name, Airfoil, camber_pos_hint)` tuples.

    Requires `aerosandbox` — only needed when rebuilding the dataset, not for
    loading the cached CSV.
    """
    import aerosandbox as asb

    items: list[tuple[str, asb.Airfoil, float]] = []
    seen: set[str] = set()

    for name, cam_pos, _t in parametric_naca_airfoils():
        if name in seen:
            continue
        try:
            af = asb.Airfoil(name)
            if af.coordinates.shape[0] < 20:
                continue
            items.append((name, af, cam_pos))
            seen.add(name)
        except Exception:
            pass

    for name in UIUC_AIRFOILS:
        if name in seen:
            continue
        try:
            af = asb.Airfoil(name)
            if af.coordinates.shape[0] < 20:
                continue
            items.append((name, af, 0.0))   # camber position unknown for non-NACA
            seen.add(name)
        except Exception:
            pass
    return items
