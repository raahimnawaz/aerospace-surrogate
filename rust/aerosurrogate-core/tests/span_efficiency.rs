//! A rectangular wing has `e < 1` (suboptimal vs. elliptic) but > 0.85
//! for any realistic AR. Mirror of
//! `tests/test_lifting_line.py::test_rectangular_wing_span_efficiency_below_unity`.

use aerosurrogate_core::{solve_lifting_line, SolverOptions, ThinAirfoilSection, Wing};

#[test]
fn rectangular_span_efficiency_in_expected_range() {
    let wing = Wing::rectangular(10.0, 1.0, 0.0, 80).unwrap(); // AR = 10
    let res = solve_lifting_line(
        &wing,
        5.0,
        &ThinAirfoilSection::default(),
        &SolverOptions {
            tol: 1e-10,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(res.converged);
    let e = res.span_efficiency;
    assert!(0.85 < e && e < 1.0, "AR=10 rectangular: e = {e:.4}");
}

#[test]
fn rectangular_span_efficiency_decreases_with_ar() {
    let opts = SolverOptions {
        tol: 1e-10,
        ..Default::default()
    };
    let section = ThinAirfoilSection::default();
    let mut e_prev = 1.0;
    for ar in [4.0, 8.0, 12.0, 16.0, 20.0] {
        let wing = Wing::rectangular(10.0, 10.0 / ar, 0.0, 80).unwrap();
        let res = solve_lifting_line(&wing, 5.0, &section, &opts).unwrap();
        assert!(res.converged);
        assert!(
            res.span_efficiency < e_prev,
            "e should decrease with AR; at AR={ar} got {} >= previous {e_prev}",
            res.span_efficiency
        );
        e_prev = res.span_efficiency;
    }
}
