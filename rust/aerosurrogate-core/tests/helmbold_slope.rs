//! Finite-wing lift-curve slope must be reduced from 2π/rad to
//! `a = 2π · AR / (AR + 2)` (Helmbold approximation; exact for elliptic
//! wings with thin-airfoil sections).
//!
//! Mirror of `tests/test_lifting_line.py::test_finite_wing_lift_slope_reduction`.

use std::f64::consts::PI;

use aerosurrogate_core::{solve_lifting_line, SolverOptions, ThinAirfoilSection, Wing};
use approx::assert_relative_eq;

fn slope_for_ar(ar: f64) -> f64 {
    let span = 10.0;
    let c_root = 4.0 * span / (PI * ar);
    let wing = Wing::elliptic(span, c_root, 0.0, 80).unwrap();
    let section = ThinAirfoilSection::default();
    let opts = SolverOptions {
        tol: 1e-10,
        ..Default::default()
    };
    let cl_0 = solve_lifting_line(&wing, 0.0, &section, &opts).unwrap().cl;
    let cl_4 = solve_lifting_line(&wing, 4.0, &section, &opts).unwrap().cl;
    (cl_4 - cl_0) / 4.0 * 180.0 / PI
}

fn predicted(ar: f64) -> f64 {
    2.0 * PI * ar / (ar + 2.0)
}

#[test]
fn helmbold_ar6() {
    assert_relative_eq!(slope_for_ar(6.0), predicted(6.0), max_relative = 1e-2);
}

#[test]
fn helmbold_ar10() {
    assert_relative_eq!(slope_for_ar(10.0), predicted(10.0), max_relative = 1e-2);
}

#[test]
fn helmbold_ar20() {
    assert_relative_eq!(slope_for_ar(20.0), predicted(20.0), max_relative = 1e-2);
}
