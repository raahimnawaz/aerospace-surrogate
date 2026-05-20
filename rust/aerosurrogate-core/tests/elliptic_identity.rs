//! `CDi = CL² / (π · AR)` for an elliptic planform — *the* analytical
//! identity of classical lifting-line theory. Mirror of the Python test
//! `tests/test_lifting_line.py::test_elliptic_wing_recovers_cdi_identity`.

use std::f64::consts::PI;

use aerosurrogate_core::{solve_lifting_line, SolverOptions, ThinAirfoilSection, Wing};
use approx::assert_relative_eq;

fn run_case(aspect_ratio: f64) {
    let span = 10.0;
    let c_root = 4.0 * span / (PI * aspect_ratio);
    let wing = Wing::elliptic(span, c_root, 0.0, 80).unwrap();
    assert_relative_eq!(wing.aspect_ratio(), aspect_ratio, max_relative = 1e-12);

    let section = ThinAirfoilSection::default();
    let opts = SolverOptions {
        tol: 1e-11,
        ..Default::default()
    };
    let res = solve_lifting_line(&wing, 5.0, &section, &opts).unwrap();
    assert!(res.converged, "AR={aspect_ratio}: did not converge");

    let cdi_predicted = res.cl * res.cl / (PI * aspect_ratio);
    assert_relative_eq!(res.cdi, cdi_predicted, max_relative = 5e-3);
    assert_relative_eq!(res.span_efficiency, 1.0, max_relative = 5e-3);
}

#[test]
fn elliptic_identity_ar4() {
    run_case(4.0);
}

#[test]
fn elliptic_identity_ar8() {
    run_case(8.0);
}

#[test]
fn elliptic_identity_ar16() {
    run_case(16.0);
}
