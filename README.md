# aerosurrogate

[![CI](https://github.com/raahimnawaz/aerospace_project/actions/workflows/ci.yml/badge.svg)](https://github.com/raahimnawaz/aerospace_project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A reproducible benchmark for replacing expensive panel-method aerodynamic solvers with small sklearn models, judged honestly against the 100-year-old closed-form baseline that's already in every aerospace textbook.

**Headline result.** On lift coefficient prediction, gradient boosting matches thin airfoil theory in the linear regime (R² 0.952 vs 0.950), pulls ahead through pre-stall (R² 0.89 vs 0.80), and stays useful into stall where thin airfoil theory collapses to **R² = −1.77**.

![CL: ML surrogate vs classical thin airfoil theory](figures/ml_vs_physics_cl.png)

---

## What this is

An aerospace engineer designing a wing needs three numbers for every airfoil cross-section: lift coefficient $C_L$, drag coefficient $C_D$, and pitching-moment coefficient $C_M$. Computing them accurately means running XFOIL (panel + boundary-layer solver) or higher-fidelity CFD — seconds to hours per query. In a design loop iterating across thousands of airfoils × dozens of flight conditions, that cost is the bottleneck.

**The question this benchmark answers:** how close can a cheap regression get to a panel-method solver, and *where exactly does the cheap model break* compared to the classical closed-form theory?

## Headline numbers

Dataset: **15,334 samples** from **107 airfoils** (parametric NACA 4-digit sweep + curated UIUC database), labeled by NeuralFoil. Test set is **held-out entire airfoils** so accuracy measures generalization to unseen geometries, not interpolation between known points.

### CL — lift coefficient (with closed-form baseline)

| Model | CV R² | Test R² | Test MAE |
|---|---:|---:|---:|
| **Poly-2 Ridge** | 0.989 | **0.986** | 0.055 |
| GradientBoosting | 0.998 | 0.981 | 0.046 |
| RandomForest | 0.998 | 0.978 | 0.049 |
| Linear (Ridge) | 0.968 | 0.960 | 0.092 |
| Thin Airfoil Theory | — | 0.910 | 0.117 |

### CL by flight regime (the differentiator)

| Regime | α band | TAT R² | TAT MAE | GBM R² | GBM MAE |
|---|---|---:|---:|---:|---:|
| Linear | [−4°, +4°) | **0.950** | 0.058 | 0.952 | **0.046** |
| Pre-stall | [+4°, +10°) | 0.800 | 0.080 | **0.890** | **0.042** |
| Stall | [+10°, +14°] | **−1.766** | 0.282 | **0.769** | **0.052** |
| Negative | [−6°, −4°) | 0.888 | 0.066 | **0.927** | **0.042** |

**Reading the table.** In the linear regime the textbook formula is essentially as good as a 600-tree gradient boosting machine — both hit R² ≈ 0.95. The ML model only earns its keep at higher angles of attack: through pre-stall it lifts R² from 0.80 → 0.89, and in stall (where TAT's potential-flow assumption fails completely and its R² goes *negative*) ML still maintains R² = 0.77. This is the textbook regime split — and the project makes the boundary explicit instead of hiding it under a single global R².

### CD and CM (ML only — TAT doesn't predict drag or moment)

| Target | Best model | Test R² | Test MAE |
|---|---|---:|---:|
| CD | GradientBoosting | 0.888 | 0.0016 |
| CM | GradientBoosting | 0.870 | 0.013 |

Feature-importance highlights: CD is dominated by `α` (47%) and `log₁₀(Re)` (39%) — viscous regime matters. CM is dominated by `max_camber` (70%) — physically correct, since camber sets the moment.

---

## The math

**Thin airfoil theory** (Glauert, 1926) linearizes the potential-flow boundary value problem around a thin cambered plate. For a parabolic camber line $z(x) = 4m\,x(1-x)$ the integral collapses to

$$
C_L = 2\pi\,(\alpha - \alpha_{L=0}), \qquad
\alpha_{L=0} \approx -2 \cdot \frac{m}{c} \quad [\text{rad}]
$$

It has **zero learned parameters** — just $\pi$, $\alpha$, and camber. Comparing ML models against it tells you two things: where ML actually buys something (capacity to model viscosity, separation, thickness effects), and where it doesn't (small-α linear regime, where classical theory is already within a few percent).

The lift-curve slope $\partial C_L / \partial \alpha = 2\pi$ /rad ≈ 0.110 /deg is one of the most-tested identities in aerodynamics. The empirical lift slope across all airfoils in our held-out set, fit linearly in $|\alpha| < 4°$, has **median 0.105 /deg** — consistent with theory to 5%.

---

## Method

### Data pipeline

| Stage | Detail |
|---|---|
| Shapes | Parametric NACA 4-digit (camber 0/2/4/6%, position, thickness 8/10/12/15/18/21%) + curated UIUC database (Selig, Eppler, Wortmann, AeroLab) — **107 airfoils total** |
| Conditions | α ∈ [−6°, +14°] × 16 points; Re ∈ [10⁵, 10⁷] × 5 points geomspaced; n_crit ∈ {5, 9} |
| Labels | [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) (MIT) — neural-net surrogate for XFOIL trained on millions of viscous panel-method solutions; rows with `analysis_confidence < 0.90` dropped |
| Featurization | 8 geometric features (max camber, camber position, max thickness, leading-edge radius, trailing-edge wedge angle, local thickness at 25/50/75% chord) + 3 flow features (α, log₁₀(Re), n_crit) |

### Splitting

`split_by_airfoil()` holds out **entire airfoil shapes** for the test set. A naïve row-level split would put α=5° of NACA 0012 in train and α=6° of the same airfoil in test — the model can "predict" by interpolating its own training trajectory. Splitting on shape forces real generalization across the wing-design loop's actual use case.

### Models

Four pipelines, monotonically increasing in capacity, plus the closed-form physics baseline.

| Model | Capacity |
|---|---|
| Thin Airfoil Theory | closed form, 0 learned params |
| Ridge (linear) | linear in 11 features |
| Poly-2 + Ridge | linear in (11 + ${11 \choose 2}$) interaction terms |
| Random Forest | 400 trees, default depth |
| Gradient Boosting | 600 trees, depth 4, lr 0.05 |

### Property tests (`tests/test_physics.py`)

Nine aerodynamics-grounded tests that double as a dataset sanity check:

- $C_L(\alpha=0, m=0) = 0$ — symmetric airfoils make no lift at zero α.
- $\partial C_L / \partial \alpha = 2\pi$ /rad — closed-form identity.
- Odd symmetry: $C_L(-\alpha) = -C_L(+\alpha)$ for symmetric airfoils.
- Empirical lift slope (NeuralFoil data) falls in [0.08, 0.13] /deg — actual 0.105.
- Symmetric airfoils have $|C_L| < 0.2$ at $|\alpha| \le 1.5°$ — actual mean 0.07.
- $C_D > 0$ for every row — no negative drag.
- Drag bucket: symmetric airfoils have minimum $C_D$ near $\alpha = 0$ — verified.

---

## Figures

### Predicted vs actual (held-out airfoils)
![Predicted vs actual](figures/predicted_vs_actual.png)

### Where does the model fail? (residuals vs α with stall band shaded)
![Residuals vs α](figures/residuals_vs_alpha.png)

### Feature importance (Random Forest MDI)
![Feature importance](figures/feature_importance.png)

---

## Repo layout

```
src/aerosurrogate/
├── airfoils.py     UIUC + parametric NACA loaders (lazy aerosandbox import)
├── features.py     geometric feature extraction
├── dataset.py      build_dataset, load_dataset, split_by_airfoil
├── models.py       four sklearn pipelines
├── physics.py      thin airfoil theory baselines + Glauert α_{L=0}
├── eval.py         per-model + per-regime evaluation (with TAT comparison for CL)
└── plot.py         four figure generators

scripts/
├── build_dataset.py   rebuild data/aero_dataset.csv from scratch (needs neuralfoil)
├── train_eval.py      train every model, print headline + per-regime tables
└── make_figures.py    regenerate figures/

tests/
├── test_physics.py    9 closed-form + empirical aerodynamics property tests
├── test_dataset.py    split-by-airfoil invariants + determinism
└── test_models.py     every pipeline fits + predicts on synthetic data
```

## Quickstart

```bash
git clone https://github.com/raahimnawaz/aerospace_project
cd aerospace_project
pip install -e ".[dev]"

pytest -q                          # 19 tests, ~3s on cached data
python scripts/train_eval.py       # reproduce every table in this README
python scripts/make_figures.py     # regenerate figures/
```

Rebuilding the dataset from scratch (slow, needs `aerosandbox` + `neuralfoil`):

```bash
pip install -e ".[build]"
python scripts/build_dataset.py
```

---

## What this benchmark does not claim

- **Does not beat NeuralFoil.** Our labels *come from* NeuralFoil, so the surrogate's ceiling is NeuralFoil's accuracy. The benchmark is "how close can a 600-tree GBM get to NeuralFoil at much lower inference cost," not "we have a better airfoil solver."
- **Does not extrapolate.** Test set is held-out shapes within the same NACA + UIUC distribution. Generalization to wholly novel geometries (e.g. transonic airfoils, supercritical sections) is not measured here.
- **Does not handle 3D effects.** This is 2D-airfoil regression. Wingtip vortices, sweep, dihedral, and induced drag all sit outside the model.

These are the right limitations to be explicit about. The contribution is the **regime-aware benchmark methodology** + the explicit ML-vs-classical-theory comparison, not a new surrogate model.
