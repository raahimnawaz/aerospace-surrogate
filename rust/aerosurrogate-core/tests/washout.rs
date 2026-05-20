//! Negative tip twist (washout) must reduce the circulation at the tip,
//! the entire reason washout is used in aircraft design. Mirror of
//! `tests/test_lifting_line.py::test_washout_reduces_tip_loading`.

use aerosurrogate_core::{solve_lifting_line, SolverOptions, ThinAirfoilSection, Wing};

#[test]
fn washout_reduces_tip_loading() {
    let opts = SolverOptions {
        tol: 1e-10,
        ..Default::default()
    };
    let section = ThinAirfoilSection::default();

    let no_wash = Wing::tapered(10.0, 1.2, 0.5, 0.0, 0.0, 60).unwrap();
    let with_wash = Wing::tapered(10.0, 1.2, 0.5, 0.0, -3.0, 60).unwrap();

    let r1 = solve_lifting_line(&no_wash, 5.0, &section, &opts).unwrap();
    let r2 = solve_lifting_line(&with_wash, 5.0, &section, &opts).unwrap();

    let n = r1.gamma.len();
    // Outermost stations on either tip; cosine spacing means index 0 is
    // closest to one tip, n-1 is closest to the other.
    assert!(
        r2.gamma[0] < r1.gamma[0],
        "washout did not reduce Γ at left tip: {} >= {}",
        r2.gamma[0],
        r1.gamma[0]
    );
    assert!(
        r2.gamma[n - 1] < r1.gamma[n - 1],
        "washout did not reduce Γ at right tip: {} >= {}",
        r2.gamma[n - 1],
        r1.gamma[n - 1]
    );
}
