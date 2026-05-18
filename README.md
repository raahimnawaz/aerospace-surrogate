# aerosurrogate

[![CI](https://github.com/raahimnawaz/aerospace_project/actions/workflows/ci.yml/badge.svg)](https://github.com/raahimnawaz/aerospace_project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

ML surrogates for 2D airfoil aerodynamic coefficients (CL / CD / CM), benchmarked against thin airfoil theory.

> **Status:** in progress — package + tests + physics baselines landed, regime analysis + README polish + headline figures still to come.

## What this is

A reproducible benchmark for replacing expensive panel-method solvers (XFOIL, NeuralFoil) with small sklearn models on the airfoil-coefficient regression task. The question this is built to answer: **how accurate can a cheap regression be at predicting lift / drag / moment, and where does it break down vs the classical thin-airfoil-theory baseline?**

## Data

- **Shapes:** 100+ airfoils — parametric NACA 4-digit sweep + curated UIUC database (sailplane / general-aviation / low-Re profiles).
- **Labels:** [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) (MIT) — a neural-network surrogate for XFOIL, accurate to within a few drag counts over the validated envelope.
- **Features (11):** geometric (max camber, camber position, max thickness, leading-edge radius, trailing-edge wedge angle, thickness at 25/50/75% chord) + flow (α, log₁₀(Re), n_crit).
- **Splitting:** held out entire airfoil shapes — not random rows — so the test set measures generalization to *unseen geometries*.

## Models

| Model | Notes |
|---|---|
| Thin Airfoil Theory | $C_L = 2\pi (\alpha - \alpha_{L=0})$, $\alpha_{L=0} \approx -2 \cdot m/c$ — closed-form baseline, no fitting |
| Linear (Ridge) | scaled features, α=1.0 |
| Polynomial-2 + Ridge | degree-2 interactions |
| Random Forest | 400 trees, default depth |
| Gradient Boosting | 600 trees, depth 4, lr 0.05 |

## Layout

```
src/aerosurrogate/
├── airfoils.py     UIUC + parametric NACA loaders (lazy aerosandbox import)
├── features.py     geometric feature extraction
├── dataset.py      build_dataset, load_dataset, split_by_airfoil
├── models.py       sklearn pipelines
├── physics.py      thin airfoil theory baselines
├── eval.py         per-model + per-regime evaluation
└── plot.py         figures
```

## Quickstart

```bash
git clone https://github.com/raahimnawaz/aerospace_project
cd aerospace_project
pip install -e ".[dev]"

pytest -q                        # 11 tests incl. physics property tests
python scripts/train_eval.py     # train + evaluate every model on the cached dataset
python scripts/make_figures.py   # regenerate figures/
```

To rebuild the dataset from scratch (slow, requires NeuralFoil):

```bash
pip install -e ".[build]"
python scripts/build_dataset.py
```
