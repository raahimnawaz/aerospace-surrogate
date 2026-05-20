//! Cross-validation: the Newton solver and the classical Glauert
//! Fourier-series solver must agree on every canonical planform.
//!
//! These are two mathematically distinct formulations of the same physics:
//! horseshoe-vortex Newton iteration vs. half-span Fourier-series
//! collocation. Agreement to 0.5% relative is the strongest internal
//! consistency check available without an external reference implementation.
//!
//! Mirror of
//! `tests/test_lifting_line.py::test_newton_matches_glauert`.

use std::f64::consts::PI;

use aerosurrogate_core::{
    glauert_fourier_llt, solve_lifting_line, SolverOptions, ThinAirfoilSection, Wing,
};
use approx::assert_relative_eq;

fn check_case(wing: Wing, alpha: f64) {
    let g = glauert_fourier_llt(&wing, alpha, 2.0 * PI, 0.0, 40).unwrap();
    let n = solve_lifting_line(
        &wing,
        alpha,
        &ThinAirfoilSection::default(),
        &SolverOptions {
            tol: 1e-12,
            ..Default::default()
        },
    )
    .unwrap();
    assert_relative_eq!(n.cl, g.cl, max_relative = 5e-3);
    assert_relative_eq!(n.cdi, g.cdi, max_relative = 5e-3);
    assert_relative_eq!(n.span_efficiency, g.span_efficiency, max_relative = 1e-2);
}

#[test]
fn newton_matches_glauert_elliptic_ar8() {
    let wing = Wing::elliptic(10.0, 4.0 * 10.0 / (PI * 8.0), 0.0, 80).unwrap();
    check_case(wing, 5.0);
}

#[test]
fn newton_matches_glauert_rectangular_ar8() {
    let wing = Wing::rectangular(10.0, 10.0 / 8.0, 0.0, 120).unwrap();
    check_case(wing, 5.0);
}

#[test]
fn newton_matches_glauert_tapered_ar6() {
    // S = b · c_root · (1 + λ)/2; we want S = b²/AR ⟹ c_root = 2b/(AR(1+λ))
    let span = 10.0;
    let ar = 6.0;
    let lam = 0.5;
    let c_root = 2.0 * span / (ar * (1.0 + lam));
    let wing = Wing::tapered(span, c_root, lam, 0.0, 0.0, 120).unwrap();
    check_case(wing, 6.0);
}
