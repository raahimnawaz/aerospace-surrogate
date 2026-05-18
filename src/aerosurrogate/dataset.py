"""Dataset construction (via NeuralFoil) and held-out-shape splitting."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "aero_dataset.csv"


def load_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Read the cached NeuralFoil-labeled dataset (one row per airfoil × α × Re × n_crit)."""
    return pd.read_csv(path)


def build_dataset(
    n_alpha: int = 16,
    n_re: int = 5,
    n_crit_values: tuple[float, ...] = (5.0, 9.0),
    model_size: str = "large",
    cache: bool = True,
    cache_path: Path = DATASET_PATH,
) -> pd.DataFrame:
    """For each airfoil × (α, Re, n_crit), query NeuralFoil for CL / CD / CM.

    Requires `aerosandbox` + `neuralfoil`. If `cache=True` and the CSV exists,
    we just reload it — the heavy XFOIL-surrogate path is opt-in.
    """
    if cache and cache_path.exists():
        print(f"loading cached dataset from {cache_path}")
        return load_dataset(cache_path)

    import neuralfoil as nf  # noqa: F401 — imported lazily

    from .airfoils import load_airfoils
    from .features import airfoil_features

    airfoils = load_airfoils()
    print(f"loaded {len(airfoils)} airfoils")

    alphas = np.linspace(-6.0, 14.0, n_alpha)
    res = np.geomspace(1e5, 1e7, n_re)

    rows: list[dict] = []
    t0 = time.time()
    for i, (name, af, cam_pos) in enumerate(airfoils):
        geom = airfoil_features(af, cam_pos)
        for re in res:
            for nc in n_crit_values:
                try:
                    out = nf.get_aero_from_airfoil(
                        airfoil=af, alpha=alphas, Re=float(re),
                        n_crit=float(nc), model_size=model_size,
                    )
                except Exception:
                    continue
                cl = np.asarray(out["CL"]).ravel()
                cd = np.asarray(out["CD"]).ravel()
                cm = np.asarray(out["CM"]).ravel()
                conf = np.asarray(
                    out.get("analysis_confidence", np.ones_like(cl))
                ).ravel()
                for k, a in enumerate(alphas):
                    if not np.isfinite(cl[k]) or not np.isfinite(cd[k]):
                        continue
                    if conf[k] < 0.90:    # drop low-confidence NeuralFoil predictions
                        continue
                    rows.append({
                        "airfoil": name,
                        **geom,
                        "alpha_deg": float(a),
                        "log10_Re":  float(np.log10(re)),
                        "n_crit":    float(nc),
                        "CL": float(cl[k]),
                        "CD": float(cd[k]),
                        "CM": float(cm[k]),
                    })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(airfoils)} airfoils  ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    print(
        f"built {len(df)} samples from {df['airfoil'].nunique()} airfoils "
        f"in {time.time() - t0:.1f}s"
    )
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"cached -> {cache_path}")
    return df


def split_by_airfoil(
    df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hold out *entire airfoils* so test set is generalization to unseen shapes.

    A naive row-level split would put α=5° of NACA0012 in train and α=6° of the
    same airfoil in test — trivial to "predict" by interpolation. Splitting on
    `airfoil` name forces the model to generalize across shapes.
    """
    rng = np.random.default_rng(seed)
    names = np.array(df["airfoil"].unique(), dtype=object)
    rng.shuffle(names)
    n_test = max(1, int(len(names) * test_frac))
    test_names = set(names[:n_test])
    is_test = df["airfoil"].isin(test_names)
    return df[~is_test].copy(), df[is_test].copy(), sorted(test_names)
