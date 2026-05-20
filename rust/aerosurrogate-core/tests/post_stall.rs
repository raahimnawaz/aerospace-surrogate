//! Post-stall convergence on a flat-plate polar (`Cl = sin(2α)`).
//!
//! The polar peaks at α = 45°; past that the section's lift slope is
//! negative, which would destabilize naive fixed-point iteration. The
//! Newton + backtracking line-search must still converge at every α and
//! produce a 3D CL curve that peaks at α ≤ 50° and drops afterward.
//!
//! Mirror of `tests/test_lifting_line.py::test_alpha_sweep_through_stall_converges`.

use aerosurrogate_core::{alpha_sweep, FlatPlatePostStall, SolverOptions, Wing};

#[test]
fn flat_plate_sweep_through_stall() {
    let wing = Wing::rectangular(10.0, 1.0, 0.0, 60).unwrap();
    let alphas: Vec<f64> = (0..=18).map(|i| i as f64 * 3.0).collect(); // 0, 3, …, 54
    let out = alpha_sweep(
        &wing,
        &alphas,
        &FlatPlatePostStall::default(),
        &SolverOptions::default(),
        true,
    )
    .unwrap();

    for (i, &converged) in out.converged.iter().enumerate() {
        assert!(
            converged,
            "did not converge at α = {} deg (iters={})",
            alphas[i], out.iterations[i]
        );
    }

    // CL(0) ≈ 0, peak in [40°, 50°], CL(54°) < CL_peak (post-stall).
    assert!(out.cl[0].abs() < 1e-9);
    let peak_idx = out
        .cl
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let peak_alpha = alphas[peak_idx];
    assert!(
        (40.0..=50.0).contains(&peak_alpha),
        "peak at α={peak_alpha}° outside [40°, 50°]"
    );
    let cl_peak = out.cl[peak_idx];
    let cl_last = out.cl[alphas.len() - 1];
    assert!(
        cl_last < cl_peak,
        "no stall observed: peak {cl_peak:.3} at α={peak_alpha}°, final {cl_last:.3} at α={}°",
        alphas[alphas.len() - 1]
    );
}
