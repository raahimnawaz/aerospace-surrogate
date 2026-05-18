"""Train every model in the zoo on the cached dataset and print headline tables.

Usage:
    python scripts/train_eval.py
"""
from __future__ import annotations

from aerosurrogate.dataset import load_dataset
from aerosurrogate.eval import evaluate, feature_importance, regime_eval
from aerosurrogate.models import FEATURES, TARGETS


def main() -> None:
    df = load_dataset()
    print(
        f"dataset: {len(df)} samples, "
        f"{df['airfoil'].nunique()} airfoils, "
        f"{len(FEATURES)} features\n"
    )

    for target in TARGETS:
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
