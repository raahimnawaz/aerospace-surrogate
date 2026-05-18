"""Regenerate every figure committed to figures/ from the cached dataset.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

from aerosurrogate.dataset import load_dataset
from aerosurrogate.plot import (
    plot_feature_importance,
    plot_predicted_vs_actual,
    plot_residuals_vs_alpha,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    df = load_dataset()
    plot_predicted_vs_actual(df, FIG_DIR / "predicted_vs_actual.png")
    plot_residuals_vs_alpha(df, FIG_DIR / "residuals_vs_alpha.png")
    plot_feature_importance(df, FIG_DIR / "feature_importance.png")


if __name__ == "__main__":
    main()
