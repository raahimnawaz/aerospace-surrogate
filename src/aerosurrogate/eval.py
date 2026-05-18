"""Cross-validated evaluation + per-regime breakdown.

The interesting question for an ML surrogate of an aerodynamic solver is not
"what's the global R²" but "in which flight regime does the model break?".
`regime_eval` splits by angle of attack and computes per-regime metrics —
that's where the ML-vs-physics story lives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from .dataset import split_by_airfoil
from .models import FEATURES, build_models
from .physics import thin_airfoil_cl

# Aerodynamic regimes, defined by angle of attack. Boundaries are
# conventional rules of thumb for moderate-Reynolds 2D airfoils.
REGIMES: dict[str, tuple[float, float]] = {
    "linear":     (-4.0, 4.0),     # Cl proportional to alpha, viscosity barely matters
    "pre_stall":  (4.0, 10.0),     # Cl still increasing but nonlinear
    "stall":      (10.0, 90.0),    # separation; Cl drops, CD spikes
    "negative":   (-90.0, -4.0),   # negative-lift / inverted regimes
}


def evaluate(df: pd.DataFrame, target: str, test_frac: float = 0.2) -> pd.DataFrame:
    """Per-model 5-fold CV on train + held-out airfoil test set."""
    train_df, test_df, _ = split_by_airfoil(df, test_frac=test_frac)
    X_tr, y_tr = train_df[FEATURES].to_numpy(), train_df[target].to_numpy()
    X_te, y_te = test_df[FEATURES].to_numpy(),  test_df[target].to_numpy()
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    rows = []
    for name, pipe in build_models().items():
        cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "model": name,
            "cv_R2_mean": cv_r2.mean(),
            "cv_R2_std":  cv_r2.std(),
            "test_R2":    r2_score(y_te, pred),
            "test_MAE":   mean_absolute_error(y_te, pred),
            "test_RMSE":  np.sqrt(mean_squared_error(y_te, pred)),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("test_R2", ascending=False)
        .reset_index(drop=True)
    )


def regime_eval(df: pd.DataFrame, target: str, model_name: str = "GradientBoosting",
                test_frac: float = 0.2) -> pd.DataFrame:
    """Per-regime R² / MAE for one model on the held-out airfoil set.

    The story this surfaces: the same model can be near-perfect in the linear
    regime and useless in the stall regime. A single global R² hides that.
    """
    train_df, test_df, _ = split_by_airfoil(df, test_frac=test_frac)
    pipe = build_models()[model_name]
    pipe.fit(train_df[FEATURES].to_numpy(), train_df[target].to_numpy())

    rows = []
    for regime, (lo, hi) in REGIMES.items():
        mask = (test_df["alpha_deg"] >= lo) & (test_df["alpha_deg"] < hi)
        sub = test_df[mask]
        if len(sub) < 5:
            continue
        X = sub[FEATURES].to_numpy()
        y = sub[target].to_numpy()
        pred = pipe.predict(X)
        rows.append({
            "regime":    regime,
            "alpha_lo":  lo,
            "alpha_hi":  hi,
            "n_samples": len(sub),
            "R2":        r2_score(y, pred),
            "MAE":       mean_absolute_error(y, pred),
            "RMSE":      np.sqrt(mean_squared_error(y, pred)),
        })
    return pd.DataFrame(rows)


def _physics_cl_predictions(test_df: pd.DataFrame) -> np.ndarray:
    """Thin-airfoil-theory C_L prediction for every row in `test_df`.

    Uses each row's own `max_camber` so the baseline gets the correct
    zero-lift α per airfoil. This is the strongest version of TAT — it
    knows the geometry, just not the viscous/nonlinear physics.
    """
    return thin_airfoil_cl(
        test_df["alpha_deg"].to_numpy(),
        test_df["max_camber"].to_numpy(),
    )


def evaluate_cl_with_physics(df: pd.DataFrame, test_frac: float = 0.2) -> pd.DataFrame:
    """Same as `evaluate(df, 'CL')` but with Thin Airfoil Theory as a row.

    TAT has no training step, so cv_R2 columns are NaN. The reason to include
    it is to show the ML models *against the textbook closed-form baseline*:
    if RandomForest barely beats TAT, that's a strong "this problem is mostly
    linear" signal.
    """
    ml = evaluate(df, "CL", test_frac=test_frac)

    _, test_df, _ = split_by_airfoil(df, test_frac=test_frac)
    y = test_df["CL"].to_numpy()
    pred = _physics_cl_predictions(test_df)
    tat = pd.DataFrame([{
        "model":      "Thin Airfoil Theory",
        "cv_R2_mean": np.nan,
        "cv_R2_std":  np.nan,
        "test_R2":    r2_score(y, pred),
        "test_MAE":   mean_absolute_error(y, pred),
        "test_RMSE":  np.sqrt(mean_squared_error(y, pred)),
    }])
    return (
        pd.concat([tat, ml], ignore_index=True)
        .sort_values("test_R2", ascending=False)
        .reset_index(drop=True)
    )


def regime_eval_cl_with_physics(
    df: pd.DataFrame, model_name: str = "GradientBoosting", test_frac: float = 0.2,
) -> pd.DataFrame:
    """Per-regime CL accuracy for ML vs Thin Airfoil Theory.

    Surfaces the story the project exists to tell: TAT is competitive with
    ML in the linear regime, ML pulls ahead in pre-stall, and both struggle
    in the stall regime.
    """
    train_df, test_df, _ = split_by_airfoil(df, test_frac=test_frac)
    pipe = build_models()[model_name]
    pipe.fit(train_df[FEATURES].to_numpy(), train_df["CL"].to_numpy())

    rows = []
    for regime, (lo, hi) in REGIMES.items():
        mask = (test_df["alpha_deg"] >= lo) & (test_df["alpha_deg"] < hi)
        sub = test_df[mask]
        if len(sub) < 5:
            continue
        y = sub["CL"].to_numpy()
        ml_pred = pipe.predict(sub[FEATURES].to_numpy())
        tat_pred = _physics_cl_predictions(sub)
        rows.append({
            "regime":     regime,
            "alpha_band": f"[{lo:+.0f}, {hi:+.0f})",
            "n":          len(sub),
            "TAT_R2":     r2_score(y, tat_pred),
            "TAT_MAE":    mean_absolute_error(y, tat_pred),
            f"{model_name}_R2":  r2_score(y, ml_pred),
            f"{model_name}_MAE": mean_absolute_error(y, ml_pred),
        })
    return pd.DataFrame(rows)


def feature_importance(df: pd.DataFrame, target: str) -> pd.Series:
    """MDI feature importance from a Random Forest, sorted descending."""
    rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    rf.fit(df[FEATURES], df[target])
    return pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
