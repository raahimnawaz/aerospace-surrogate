"""Geometric feature extraction from airfoil coordinates."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aerosandbox as asb


def airfoil_features(af: asb.Airfoil, cam_pos_hint: float) -> dict[str, float]:
    """Scalar shape features used as model inputs.

    Camber and thickness are normalized by chord. Local-thickness samples at
    25/50/75% chord capture the airfoil profile beyond just max thickness.
    """
    return {
        "max_camber":    float(af.max_camber()),
        "camber_pos":    float(cam_pos_hint),
        "max_thickness": float(af.max_thickness()),
        "le_radius":     float(af.LE_radius()),
        "te_angle_deg":  float(af.TE_angle()),
        "thickness_25":  float(af.local_thickness(x_over_c=0.25)),
        "thickness_50":  float(af.local_thickness(x_over_c=0.50)),
        "thickness_75":  float(af.local_thickness(x_over_c=0.75)),
    }
