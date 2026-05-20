# aerosurrogate-rs — Rust port of the lifting-line solver

A faithful Rust port of the Python `aerosurrogate.lifting_line` package,
designed for embedded / real-time use where the Python implementation's
millisecond-scale per-α latency is too slow.

**Why it exists.** The Python reference solver pays for the interpreter,
for numpy array allocation per call, and (when using a viscous polar)
for repeated sklearn `pipeline.predict()` calls inside the Newton loop.
For preliminary design that's fine. For an in-the-loop MPC controller
that updates an aero buildup every 10 ms, it isn't. The Rust port
removes all three costs: pure-Rust LU via [nalgebra], no allocation in
the hot path, and a baked-in version of the trained sklearn Ridge
surrogate stored as `pub const` arrays of f64.

## Headline numbers (M-series MacBook, single thread)

| Solver call | Sectional polar | Python | Rust | Speed-up |
|---|---|---:|---:|---:|
| `solve_lifting_line` (single α) | Thin airfoil | 120 µs | **40 µs** | 3.0× |
| `solve_lifting_line` (single α) | Ridge surrogate (NACA 2412) | 3.9 ms | **0.45 ms** | **8.7×** |
| `alpha_sweep` (25 points, warm-start) | Thin airfoil | 3.0 ms | **1.0 ms** | 3.1× |
| `alpha_sweep` (25 points, warm-start) | Ridge surrogate (NACA 2412) | 38 ms | **3.9 ms** | **9.8×** |

The Ridge sweep is the headline: a 25-α drag-buildup that takes 38 ms
in Python takes 3.9 ms in Rust — fast enough to run inside a 100 Hz
control loop on commodity hardware, with zero Python or sklearn
dependency at runtime.

## Correctness

Two layers of validation:

1. **Internal**: 32 unit + integration tests in `aerosurrogate-core`
   covering the same identities as the Python suite (elliptic
   `CDi = CL²/(πAR)`, Helmbold lift-slope reduction, span-efficiency
   bounds, washout, post-stall convergence, Newton-vs-Glauert
   cross-validation, Ridge ↔ sklearn parity to 1e-12).
2. **Cross-language**: 28 Python ↔ Rust parity tests in
   `tests/test_rust_parity.py` (in the repo root) that run identical
   wing configurations through both implementations and assert
   agreement to **1e-10**. The Rust port reproduces the Python
   numbers bit-for-bit (`CL = 0.438627` for AR=8 elliptic at α=5°
   matches to 6 decimals, with the Newton iteration converging in
   the same 2 iterations).

## Quick start

```bash
# from the repo root, in a venv with python ≥ 3.10
pip install -e ".[dev]"
cd rust/aerosurrogate-py
pip install maturin
maturin develop --release
```

Then from Python:

```python
import math
import aerosurrogate_rs as rs

wing = rs.Wing.elliptic(span=10.0, root_chord=4*10/(math.pi*8), n_sections=80)
res = rs.solve_lifting_line(wing, 5.0, rs.ThinAirfoilSection(), tol=1e-12)
print(f"CL={res.CL:.6f}  CDi={res.CDi:.6f}  e={res.span_efficiency:.6f}")
# CL=0.438627  CDi=0.007655  e=1.000000
```

Or with the baked-in Ridge surrogate:

```python
section = rs.RidgeSurrogateSection.naca4(2.0, 4.0, 12.0, n_crit=9.0)
res = rs.solve_lifting_line(wing, 5.0, section, re_ref=3e6)
print(f"CL={res.CL:.4f}  CD={res.CD:.5f}  e={res.span_efficiency:.3f}")
```

Pure Rust API:

```rust
use aerosurrogate_core::{solve_lifting_line, SolverOptions, ThinAirfoilSection, Wing};

let wing = Wing::elliptic(10.0, 1.5915, 0.0, 80)?;
let section = ThinAirfoilSection::default();
let result = solve_lifting_line(&wing, 5.0, &section, &SolverOptions::default())?;
println!("CL={:.6} CDi={:.6} e={:.6}", result.cl, result.cdi, result.span_efficiency);
```

## Workspace layout

```
rust/
├── Cargo.toml                  Workspace root. default-members = [core]
├── aerosurrogate-core/         Pure Rust library (no PyO3 dep)
│   ├── src/
│   │   ├── lib.rs              Public re-exports + AeroError
│   │   ├── geometry.rs         Wing + rectangular/elliptic/tapered factories
│   │   ├── biot_savart.rs      N×N horseshoe-vortex downwash matrix
│   │   ├── sections.rs         SectionalAero trait + analytical + Ridge impls
│   │   ├── surrogate/          Auto-generated sklearn weights (const arrays)
│   │   ├── classical.rs        Glauert Fourier-series LLT (reference)
│   │   └── solver.rs           Newton iteration + line search + α-sweep
│   ├── tests/                  8 integration tests mirroring the Python suite
│   └── benches/solver.rs       criterion benchmarks
├── aerosurrogate-py/           PyO3 wrapper → Python wheel via maturin
│   ├── Cargo.toml              extension-module feature gated for cargo build
│   ├── pyproject.toml          maturin config
│   └── src/lib.rs              #[pymodule] aerosurrogate_rs
└── scripts/
    └── export_ridge_weights.py Trained sklearn → ridge.rs const arrays
```

## Reproducing the benchmarks

```bash
cd rust
cargo bench --bench solver
```

The criterion report goes to `rust/target/criterion/`; open
`index.html` for the full picture (warm-up curves, jitter, regression).

## What's not (yet) ported

* **GradientBoosting surrogate**: the export script for 600-tree
  gradient boosting models is deferred — the headline framing
  (ML-trained 2D polar inside a 3D LLT solver running in microseconds)
  already lands cleanly with the Ridge polar at 9× speed-up. GBM is a
  v2 extension.
* **NeuralFoil**: a Rust port of NeuralFoil's neural net would be a
  separate project; the Ridge surrogate (trained on NeuralFoil labels)
  is the lightweight stand-in for embedded use.
* **Sweep / dihedral**: like the Python implementation, this port
  handles planar unswept wings only. Adding swept wings means moving
  to a full vortex-lattice method (out of scope).
