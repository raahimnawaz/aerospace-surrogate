//! An elliptic planform should produce elliptic circulation
//! `Γ(y) ∝ √(1 − (2y/b)²)`. Mirror of
//! `tests/test_lifting_line.py::test_elliptic_loading_shape`.

use aerosurrogate_core::{solve_lifting_line, SolverOptions, ThinAirfoilSection, Wing};

#[test]
fn elliptic_loading_matches_expected_shape() {
    let wing = Wing::elliptic(10.0, 1.0, 0.0, 80).unwrap();
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

    let peak = res.gamma.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let expected_peak: f64 = wing
        .y_cp
        .iter()
        .map(|&y| {
            let s = 1.0 - (2.0 * y / wing.span).powi(2);
            s.max(0.0).sqrt()
        })
        .fold(0.0_f64, f64::max);

    for i in 0..wing.n_sections() {
        let y = wing.y_cp[i];
        let s = (1.0 - (2.0 * y / wing.span).powi(2)).max(0.0).sqrt();
        let ratio_actual = res.gamma[i] / peak;
        let ratio_expected = s / expected_peak;
        assert!(
            (ratio_actual - ratio_expected).abs() < 5e-3,
            "section {i} (y={y:.3}): ratio {ratio_actual:.6} vs expected {ratio_expected:.6}"
        );
    }
}
