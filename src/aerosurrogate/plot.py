"""Figures used in the README + paper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .dataset import split_by_airfoil
from .eval import REGIMES, _physics_cl_predictions, feature_importance
from .models import FEATURES, TARGETS, build_models


def plot_predicted_vs_actual(df: pd.DataFrame, out_path: Path,
                             model_name: str = "GradientBoosting") -> None:
    """Three-panel scatter: predicted vs actual for CL, CD, CM on held-out airfoils."""
    import matplotlib.pyplot as plt

    train_df, test_df, _ = split_by_airfoil(df, test_frac=0.2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, target in zip(axes, TARGETS, strict=False):
        pipe = build_models()[model_name]
        pipe.fit(train_df[FEATURES].to_numpy(), train_df[target].to_numpy())
        y = test_df[target].to_numpy()
        pred = pipe.predict(test_df[FEATURES].to_numpy())
        ax.scatter(y, pred, s=6, alpha=0.4)
        lo, hi = float(y.min()), float(y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"true {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.set_title(f"{target}   R² = {r2_score(y, pred):.3f}")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name} on held-out airfoils (NeuralFoil ground truth)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    plt.close(fig)


def plot_residuals_vs_alpha(df: pd.DataFrame, out_path: Path,
                            model_name: str = "GradientBoosting") -> None:
    """Residual scatter vs angle of attack with regime bands shaded.

    This is the figure that tells the story: ML residuals are tight in the
    linear regime, fan out in the stall regime. Aerospace audience reads it
    instantly.
    """
    import matplotlib.pyplot as plt

    train_df, test_df, _ = split_by_airfoil(df, test_frac=0.2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, target in zip(axes, TARGETS, strict=False):
        pipe = build_models()[model_name]
        pipe.fit(train_df[FEATURES].to_numpy(), train_df[target].to_numpy())
        y = test_df[target].to_numpy()
        pred = pipe.predict(test_df[FEATURES].to_numpy())
        resid = pred - y
        ax.scatter(test_df["alpha_deg"], resid, s=6, alpha=0.4)
        ax.axhline(0, color="k", lw=1)
        # Shade the stall regime
        lo, hi = REGIMES["stall"]
        ax.axvspan(lo, min(hi, float(test_df["alpha_deg"].max())),
                   color="red", alpha=0.08, label="stall regime")
        ax.set_xlabel("angle of attack [deg]")
        ax.set_ylabel(f"{target} residual (pred − true)")
        ax.set_title(f"{target}: residuals vs α")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{model_name}: where does the model fail?")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    plt.close(fig)


def plot_ml_vs_physics_cl(df: pd.DataFrame, out_path: Path,
                          model_name: str = "GradientBoosting") -> None:
    """CL prediction error vs angle of attack: ML model vs thin airfoil theory.

    The headline figure of the project. TAT is the closed-form baseline (zero
    learned params); the ML model has 11 features and ~600 boosting trees.
    Plotting absolute residuals side-by-side across the α range shows
    exactly where each method wins and where they fail.
    """
    import matplotlib.pyplot as plt

    train_df, test_df, _ = split_by_airfoil(df, test_frac=0.2)
    pipe = build_models()[model_name]
    pipe.fit(train_df[FEATURES].to_numpy(), train_df["CL"].to_numpy())

    y = test_df["CL"].to_numpy()
    ml_pred = pipe.predict(test_df[FEATURES].to_numpy())
    tat_pred = _physics_cl_predictions(test_df)
    alpha = test_df["alpha_deg"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Left: residuals scatter, both methods overlaid
    ax = axes[0]
    ax.scatter(alpha, np.abs(tat_pred - y), s=8, alpha=0.5, color="C1",
               label="Thin Airfoil Theory")
    ax.scatter(alpha, np.abs(ml_pred - y), s=8, alpha=0.5, color="C0",
               label=model_name)
    lo, hi = REGIMES["stall"]
    ax.axvspan(lo, min(hi, float(alpha.max())), color="red", alpha=0.08,
               label="stall regime")
    ax.set_xlabel("angle of attack [deg]")
    ax.set_ylabel("|CL error|  (pred − true|)")
    ax.set_title("CL absolute error vs α")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # Right: median + IQR by α bin, easier to read trend
    ax = axes[1]
    bins = np.arange(-6, 16, 2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for label, pred, color in [
        ("Thin Airfoil Theory", tat_pred, "C1"),
        (model_name,            ml_pred,  "C0"),
    ]:
        med_list: list[float] = []
        q25_list: list[float] = []
        q75_list: list[float] = []
        for i in range(len(bins) - 1):
            mask = (alpha >= bins[i]) & (alpha < bins[i + 1])
            if mask.sum() < 5:
                med_list.append(np.nan)
                q25_list.append(np.nan)
                q75_list.append(np.nan)
                continue
            err = np.abs(pred[mask] - y[mask])
            med_list.append(float(np.median(err)))
            q25_list.append(float(np.percentile(err, 25)))
            q75_list.append(float(np.percentile(err, 75)))
        med_arr = np.array(med_list)
        q25_arr = np.array(q25_list)
        q75_arr = np.array(q75_list)
        ax.plot(centers, med_arr, color=color, lw=2, label=label, marker="o")
        ax.fill_between(centers, q25_arr, q75_arr, color=color, alpha=0.2)
    ax.axvspan(lo, min(hi, float(alpha.max())), color="red", alpha=0.08)
    ax.set_xlabel("angle of attack [deg]")
    ax.set_title("Median |CL error| ± IQR by α bin")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("CL: ML surrogate vs classical thin airfoil theory")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of feature importance for each of CL, CD, CM."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, target in zip(axes, TARGETS, strict=False):
        imp = feature_importance(df, target)
        # plot bottom-up so most important is on top
        ax.barh(np.arange(len(imp))[::-1], imp.values, color="steelblue")
        ax.set_yticks(np.arange(len(imp))[::-1])
        ax.set_yticklabels(imp.index)
        ax.set_xlabel("importance (MDI)")
        ax.set_title(target)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Random Forest feature importance per target")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")
    plt.close(fig)
