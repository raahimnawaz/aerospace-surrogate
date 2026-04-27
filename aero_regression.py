"""
Aerodynamic coefficient regression — real data version.

Predicts lift (Cl), drag (Cd) and moment (Cm) coefficients of 2D airfoils
from geometry + flow parameters.

Data pipeline (everything is real):
  * Airfoil shapes come from the UIUC Airfoil Database (via aerosandbox),
    plus a parametric sweep of NACA 4-digit airfoils.
  * Aerodynamic labels come from NeuralFoil (Mach et al., MIT) — a
    neural-network surrogate for XFOIL trained on millions of viscous
    panel-method solutions. Accuracy is within a few drag counts of XFOIL
    over the validated envelope.

Features (per sample)
    max_camber        max camber / chord
    camber_pos        x-location of max camber [0,1]   (NACA only; 0 otherwise)
    max_thickness     max thickness / chord
    le_radius         leading-edge radius / chord
    te_angle_deg      trailing-edge wedge angle [deg]
    thickness_25      local thickness at x/c = 0.25
    thickness_50      local thickness at x/c = 0.50
    thickness_75      local thickness at x/c = 0.75
    alpha_deg         angle of attack [deg]
    log10_Re          log10(Reynolds number)
    n_crit            transition amplification factor (turbulence proxy)

Targets
    CL, CD, CM        lift, drag, pitching-moment coefficients

Models compared: Ridge, Polynomial-Ridge, RandomForest, GradientBoosting.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import aerosandbox as asb
import neuralfoil as nf

warnings.filterwarnings("ignore", category=RuntimeWarning)

RNG = np.random.default_rng(7)
CACHE = Path(__file__).parent / "aero_dataset.csv"


# ---------- 1. Airfoil sources --------------------------------------------

# Curated UIUC airfoils spanning sailplane / general-aviation / low-Re ranges.
UIUC_AIRFOILS: list[str] = [
    "naca0008", "naca0012", "naca0015", "naca0018",
    "naca1408", "naca2410", "naca2412", "naca2415", "naca4412", "naca4415",
    "naca23012", "naca63012a", "naca64a010",
    "s1210", "s1223", "s2027", "s4083", "s8036", "sd7003", "sd7037", "sd7062",
    "e387", "e423", "e472",
    "clarky", "clarkysm", "goe398", "goe796",
    "rg15", "ag35", "ag36", "ag38", "ag40d", "ag45c",
    "fx63137", "fx72ls160", "mh32", "mh60",
    "ah79100c", "ah93w215", "ls417",
]


def naca4_name(m: int, p: int, t: int) -> str:
    """Build a NACA 4-digit name from camber/camber-pos/thickness integers."""
    return f"naca{m}{p}{t:02d}"


def parametric_naca_airfoils() -> list[tuple[str, float, float]]:
    """Parametric sweep of NACA 4-digit airfoils. Returns (name, camber_pos, _) tuples."""
    out = []
    for m in (0, 2, 4, 6):                 # max camber %
        for p in (0, 2, 4, 6):             # camber position (tenths)
            if m == 0 and p != 0:          # symmetric airfoils have p==0
                continue
            for t in (8, 10, 12, 15, 18, 21):  # max thickness %
                name = naca4_name(m, p, t)
                cam_pos = p / 10.0 if m > 0 else 0.0
                out.append((name, cam_pos, t / 100.0))
    return out


def load_airfoils() -> list[tuple[str, asb.Airfoil, float]]:
    """Return list of (name, Airfoil, camber_pos_hint) — one entry per shape."""
    items: list[tuple[str, asb.Airfoil, float]] = []
    seen: set[str] = set()

    for name, cam_pos, _t in parametric_naca_airfoils():
        if name in seen:
            continue
        try:
            af = asb.Airfoil(name)
            if af.coordinates.shape[0] < 20:
                continue
            items.append((name, af, cam_pos))
            seen.add(name)
        except Exception:
            pass

    for name in UIUC_AIRFOILS:
        if name in seen:
            continue
        try:
            af = asb.Airfoil(name)
            if af.coordinates.shape[0] < 20:
                continue
            items.append((name, af, 0.0))   # cam_pos unknown for non-NACA
            seen.add(name)
        except Exception:
            pass
    return items


# ---------- 2. Geometric feature extraction --------------------------------

def airfoil_features(af: asb.Airfoil, cam_pos_hint: float) -> dict[str, float]:
    """Extract scalar geometry features from coordinates."""
    return {
        "max_camber":    float(af.max_camber()),
        "camber_pos":    float(cam_pos_hint),
        "max_thickness": float(af.max_thickness()),
        "le_radius":     float(af.LE_radius()),
        "te_angle_deg":  float(af.TE_angle()),
        "thickness_25":  float(af.local_thickness(x_over_c=0.25)),
        "thickness_50":  float(af.local_thickness(x_over_c=0.50)),
        "thickness_75":  float(af.local_thickness(x_over_c=0.75)),
    }


# ---------- 3. Build labelled dataset via NeuralFoil -----------------------

def build_dataset(
    n_alpha: int = 16,
    n_re: int = 5,
    n_crit_values: tuple[float, ...] = (5.0, 9.0),
    model_size: str = "large",
    cache: bool = True,
) -> pd.DataFrame:
    """For each airfoil x (alpha, Re, n_crit), query NeuralFoil for CL/CD/CM."""
    if cache and CACHE.exists():
        print(f"loading cached dataset from {CACHE}")
        return pd.read_csv(CACHE)

    airfoils = load_airfoils()
    print(f"loaded {len(airfoils)} airfoils")

    alphas = np.linspace(-6.0, 14.0, n_alpha)
    res = np.geomspace(1e5, 1e7, n_re)

    rows = []
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
                conf = np.asarray(out.get("analysis_confidence", np.ones_like(cl))).ravel()
                for k, a in enumerate(alphas):
                    if not np.isfinite(cl[k]) or not np.isfinite(cd[k]):
                        continue
                    if conf[k] < 0.90:        # drop low-confidence NF predictions
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
            print(f"  {i+1}/{len(airfoils)} airfoils  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    print(f"built {len(df)} samples from {df['airfoil'].nunique()} airfoils "
          f"in {time.time()-t0:.1f}s")
    if cache:
        df.to_csv(CACHE, index=False)
        print(f"cached -> {CACHE}")
    return df


# ---------- 4. Model zoo ---------------------------------------------------

FEATURES = [
    "max_camber", "camber_pos", "max_thickness", "le_radius", "te_angle_deg",
    "thickness_25", "thickness_50", "thickness_75",
    "alpha_deg", "log10_Re", "n_crit",
]
TARGETS = ["CL", "CD", "CM"]


def build_models() -> dict[str, Pipeline]:
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


# ---------- 5. Evaluation --------------------------------------------------

def split_by_airfoil(df: pd.DataFrame, test_frac: float = 0.2):
    """Hold out entire airfoils so generalization is to *unseen shapes*."""
    rng = np.random.default_rng(0)
    names = df["airfoil"].unique()
    rng.shuffle(names)
    n_test = max(1, int(len(names) * test_frac))
    test_names = set(names[:n_test])
    is_test = df["airfoil"].isin(test_names)
    return df[~is_test].copy(), df[is_test].copy(), sorted(test_names)


def evaluate(df: pd.DataFrame, target: str) -> pd.DataFrame:
    train_df, test_df, _ = split_by_airfoil(df, test_frac=0.2)
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
    return pd.DataFrame(rows).sort_values("test_R2", ascending=False).reset_index(drop=True)


def feature_importance(df: pd.DataFrame, target: str) -> pd.Series:
    rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    rf.fit(df[FEATURES], df[target])
    return pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)


# ---------- 6. Plot --------------------------------------------------------

def plot_predictions(df: pd.DataFrame, out_path: str = "aero_predictions.png") -> None:
    import matplotlib.pyplot as plt

    train_df, test_df, _ = split_by_airfoil(df, test_frac=0.2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, target in zip(axes, TARGETS):
        X_tr, y_tr = train_df[FEATURES].to_numpy(), train_df[target].to_numpy()
        X_te, y_te = test_df[FEATURES].to_numpy(),  test_df[target].to_numpy()
        gbr = build_models()["GradientBoosting"].fit(X_tr, y_tr)
        pred = gbr.predict(X_te)
        ax.scatter(y_te, pred, s=6, alpha=0.4)
        lo, hi = float(y_te.min()), float(y_te.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"true {target}")
        ax.set_ylabel(f"predicted {target}")
        ax.set_title(f"{target}   R² = {r2_score(y_te, pred):.3f}")
        ax.grid(alpha=0.3)
    fig.suptitle("Gradient Boosting on held-out airfoils (NeuralFoil ground truth)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


# ---------- 7. Main --------------------------------------------------------

def main() -> None:
    df = build_dataset(model_size="large", cache=True)
    print(f"\ndataset: {len(df)} samples, "
          f"{df['airfoil'].nunique()} airfoils, "
          f"{len(FEATURES)} features\n")

    for target in TARGETS:
        print(f"=== {target} ===")
        print(evaluate(df, target).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        print("\nfeature importance (RandomForest):")
        print(feature_importance(df, target).to_string(
            float_format=lambda v: f"{v:.3f}"))
        print()

    plot_predictions(df)


if __name__ == "__main__":
    main()
