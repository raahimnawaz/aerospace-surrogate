//! Per-α latency benchmarks for the Rust LLT solver.
//!
//! Compares the analytical sectional polars (no allocation, pure scalar
//! arithmetic) against the baked-in Ridge surrogate inside the same
//! Newton-iteration solver. The headline number for the project's README
//! comes from these benches.
//!
//! Run with:
//!     cargo bench --bench solver
//!
//! Numbers go into rust/README.md and the main README's latency table.

use std::f64::consts::PI;

use aerosurrogate_core::{
    alpha_sweep, solve_lifting_line, RidgeSurrogateSection, SolverOptions, ThinAirfoilSection,
    Wing,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_thinairfoil_single_alpha(c: &mut Criterion) {
    let wing = Wing::elliptic(10.0, 4.0 * 10.0 / (PI * 8.0), 0.0, 80).unwrap();
    let section = ThinAirfoilSection::default();
    let opts = SolverOptions::default();
    c.bench_function("solve_lifting_line/elliptic_AR8/thinairfoil/N=80", |b| {
        b.iter(|| solve_lifting_line(&wing, black_box(5.0), &section, &opts).unwrap())
    });
}

fn bench_ridge_single_alpha(c: &mut Criterion) {
    let wing = Wing::rectangular(8.0, 1.0, 0.0, 80).unwrap();
    let section = RidgeSurrogateSection::naca4(2.0, 4.0, 12.0, 9.0); // NACA 2412
    let opts = SolverOptions::default();
    c.bench_function("solve_lifting_line/rectangular_AR8/ridge_naca2412/N=80", |b| {
        b.iter(|| {
            solve_lifting_line(&wing, black_box(5.0), &section, &opts).unwrap()
        })
    });
}

fn bench_thinairfoil_alpha_sweep(c: &mut Criterion) {
    let wing = Wing::elliptic(10.0, 4.0 * 10.0 / (PI * 8.0), 0.0, 80).unwrap();
    let section = ThinAirfoilSection::default();
    let alphas: Vec<f64> = (0..25).map(|i| -2.0 + (i as f64) * 0.5).collect();
    let opts = SolverOptions::default();
    c.bench_function("alpha_sweep/25_points/elliptic_AR8/thinairfoil", |b| {
        b.iter(|| alpha_sweep(&wing, black_box(&alphas), &section, &opts, true).unwrap())
    });
}

fn bench_ridge_alpha_sweep(c: &mut Criterion) {
    let wing = Wing::rectangular(8.0, 1.0, 0.0, 80).unwrap();
    let section = RidgeSurrogateSection::naca4(2.0, 4.0, 12.0, 9.0);
    let alphas: Vec<f64> = (0..25).map(|i| -2.0 + (i as f64) * 0.5).collect();
    let opts = SolverOptions::default();
    c.bench_function("alpha_sweep/25_points/rectangular_AR8/ridge_naca2412", |b| {
        b.iter(|| alpha_sweep(&wing, black_box(&alphas), &section, &opts, true).unwrap())
    });
}

criterion_group!(
    benches,
    bench_thinairfoil_single_alpha,
    bench_ridge_single_alpha,
    bench_thinairfoil_alpha_sweep,
    bench_ridge_alpha_sweep,
);
criterion_main!(benches);
