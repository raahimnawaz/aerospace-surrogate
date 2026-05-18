"""sklearn model zoo for the airfoil-coefficient regression problem."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES: list[str] = [
    "max_camber", "camber_pos", "max_thickness", "le_radius", "te_angle_deg",
    "thickness_25", "thickness_50", "thickness_75",
    "alpha_deg", "log10_Re", "n_crit",
]
TARGETS: list[str] = ["CL", "CD", "CM"]


def build_models() -> dict[str, Pipeline]:
    """Four-tier model zoo, monotonically increasing in flexibility.

    The point of the zoo is to show how much each step of capacity buys us
    on this dataset — a Ridge baseline that gets to R² ≈ 0.85 is a strong
    statement that the problem is mostly linear in the chosen features.
    """
    return {
        "Linear (Ridge)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Poly-2 Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                n_jobs=-1, random_state=0,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("model", GradientBoostingRegressor(
                n_estimators=600, max_depth=4, learning_rate=0.05, random_state=0,
            )),
        ]),
    }
