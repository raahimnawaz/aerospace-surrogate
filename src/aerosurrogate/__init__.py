"""aerosurrogate — ML surrogates for 2D airfoil aerodynamic coefficients,
plus a nonlinear lifting-line solver that lifts them to 3D finite wings.

Public API:
    FEATURES, TARGETS     — feature and target column names used by every model
    load_dataset()        — load the cached NeuralFoil-labeled dataset
    split_by_airfoil()    — train/test split that holds out entire airfoil shapes
    build_models()        — dict of sklearn pipelines (Ridge / Poly-Ridge / RF / GBM)
    evaluate(df, target)  — per-model CV + held-out test metrics
    regime_eval(...)      — per-regime (linear / pre-stall / stall) breakdown
    thin_airfoil_cl(...)  — closed-form aerodynamic baseline

    lifting_line             — subpackage: 3D finite-wing analysis (LLT)
    Wing, SectionalAero      — wing geometry + 2D-polar protocol
    solve_lifting_line(...)  — single-α nonlinear lifting-line solve
    alpha_sweep(...)         — warm-started α-sweep returning CL/CDi/CD/e
"""
from . import lifting_line
from .dataset import DATASET_PATH, load_dataset, split_by_airfoil
from .eval import (
    evaluate,
    evaluate_cl_with_physics,
    feature_importance,
    regime_eval,
    regime_eval_cl_with_physics,
)
from .lifting_line import (
    FlatPlatePostStall,
    GlauertResult,
    LiftingLineResult,
    NeuralFoilSection,
    SectionalAero,
    ThinAirfoilSection,
    Wing,
    alpha_sweep,
    glauert_fourier_llt,
    solve_lifting_line,
)
from .models import FEATURES, TARGETS, build_models
from .physics import (
    thin_airfoil_cl,
    thin_airfoil_cl_slope_per_deg,
    thin_airfoil_zero_lift_alpha,
)

__all__ = [
    "DATASET_PATH",
    "FEATURES",
    "FlatPlatePostStall",
    "GlauertResult",
    "LiftingLineResult",
    "NeuralFoilSection",
    "SectionalAero",
    "TARGETS",
    "ThinAirfoilSection",
    "Wing",
    "alpha_sweep",
    "build_models",
    "evaluate",
    "evaluate_cl_with_physics",
    "feature_importance",
    "glauert_fourier_llt",
    "lifting_line",
    "load_dataset",
    "regime_eval",
    "regime_eval_cl_with_physics",
    "solve_lifting_line",
    "split_by_airfoil",
    "thin_airfoil_cl",
    "thin_airfoil_cl_slope_per_deg",
    "thin_airfoil_zero_lift_alpha",
]
