"""Rebuild the NeuralFoil-labeled dataset from scratch.

Requires `aerosandbox` + `neuralfoil` (install via `pip install .[build]`).
The result is written to data/aero_dataset.csv and committed to the repo so
training/eval workflows don't need either heavy dep.

Usage:
    python scripts/build_dataset.py
"""
from __future__ import annotations

import warnings

from aerosurrogate.dataset import build_dataset


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    df = build_dataset(model_size="large", cache=True)
    print(
        f"\ndataset: {len(df)} samples, "
        f"{df['airfoil'].nunique()} airfoils, "
        f"{len(df.columns)} columns"
    )


if __name__ == "__main__":
    main()
