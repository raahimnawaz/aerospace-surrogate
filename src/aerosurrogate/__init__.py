"""aerosurrogate — ML surrogates for 2D airfoil aerodynamic coefficients.

Public API:
    FEATURES, TARGETS     — feature and target column names used by every model
    load_dataset()        — load the cached NeuralFoil-labeled dataset
    split_by_airfoil()    — train/test split that holds out entire airfoil shapes
    build_models()        — dict of sklearn pipelines (Ridge / Poly-Ridge / RF / GBM)
    evaluate(df, target)  — per-model CV + held-out test metrics
    regime_eval(...)      — per-regime (linear / pre-stall / stall) breakdown
    thin_airfoil_cl(...)  — closed-form aerodynamic baseline
"""
from .dataset import DATASET_PATH, load_dataset, split_by_airfoil
from .eval import evaluate, feature_importance, regime_eval
from .models import FEATURES, TARGETS, build_models
from .physics import (
    thin_airfoil_cl,
    thin_airfoil_cl_slope_per_deg,
    thin_airfoil_zero_lift_alpha,
)

__all__ = [
    "DATASET_PATH",
    "FEATURES",
    "TARGETS",
    "build_models",
    "evaluate",
    "feature_importance",
    "load_dataset",
    "regime_eval",
    "split_by_airfoil",
    "thin_airfoil_cl",
    "thin_airfoil_cl_slope_per_deg",
    "thin_airfoil_zero_lift_alpha",
]
