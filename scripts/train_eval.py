"""Train every model in the zoo on the cached dataset and print headline tables.

Usage:
    python scripts/train_eval.py
"""
from __future__ import annotations

from aerosurrogate.dataset import load_dataset
from aerosurrogate.eval import (
    evaluate,
    evaluate_cl_with_physics,
    feature_importance,
    regime_eval,
    regime_eval_cl_with_physics,
)
from aerosurrogate.models import FEATURES


def main() -> None:
    df = load_dataset()
    print(
        f"dataset: {len(df)} samples, "
        f"{df['airfoil'].nunique()} airfoils, "
        f"{len(FEATURES)} features\n"
    )

    # CL gets the special physics-baseline treatment.
    print("=== CL  (with Thin Airfoil Theory baseline) ===")
    print(evaluate_cl_with_physics(df).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("  per-regime CL: ML vs Thin Airfoil Theory")
    rg = regime_eval_cl_with_physics(df, model_name="GradientBoosting")
    if not rg.empty:
        print("  " + rg.to_string(
            index=False, float_format=lambda v: f"{v:.4f}").replace("\n", "\n  "))
    print()
    print("  feature importance (RandomForest, target=CL):")
    print("  " + feature_importance(df, "CL").to_string(
        float_format=lambda v: f"{v:.3f}").replace("\n", "\n  "))
    print()

    # CD and CM: ML only (TAT doesn't predict drag or moment).
    for target in ("CD", "CM"):
        print(f"=== {target} ===")
        print(evaluate(df, target).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        print()
        print(f"  per-regime breakdown (GradientBoosting, target={target}):")
        rg = regime_eval(df, target, model_name="GradientBoosting")
        if not rg.empty:
            print("  " + rg.to_string(
                index=False, float_format=lambda v: f"{v:.4f}").replace("\n", "\n  "))
        print()
        print(f"  feature importance (RandomForest, target={target}):")
        print("  " + feature_importance(df, target).to_string(
            float_format=lambda v: f"{v:.3f}").replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
