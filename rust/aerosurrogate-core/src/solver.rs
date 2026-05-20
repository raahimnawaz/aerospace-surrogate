//! Nonlinear lifting-line solver for a planar, unswept finite-span wing.
//!
//! Mirror of Python `aerosurrogate.lifting_line.solver`.
//!
//! Solves the Phillips-Snyder nonlinear LLT system
//!
//! ```text
//! F_i(Γ) = Γ_i − ½ · V_∞ · c_i · Cl_section(α_eff_i(Γ), Re_i) = 0
//! ```
//!
//! by Newton iteration with a finite-difference Jacobian and a backtracking
//! line search. The Jacobian is
//!
//! ```text
//! J_ij = δ_ij + ½ · c_i · a_i · W[i,j]
//! ```
//!
//! where `a_i = dCl/dα` is the local lift-curve slope (1/rad) obtained by a
//! one-sided FD on the sectional polar, and `W` is the horseshoe-vortex
//! downwash kernel from [`crate::biot_savart`].

use nalgebra::{DMatrix, DVector};

use crate::{biot_savart::downwash_matrix, AeroError, Result, SectionalAero, Wing};

/// Output of [`solve_lifting_line`]. All spanwise arrays are sampled at the
/// wing's control points.
#[derive(Debug, Clone)]
pub struct LiftingLineResult {
    /// Bound circulation Γ(y) (m²/s).
    pub gamma: DVector<f64>,
    /// Effective angle of attack (degrees) at each section.
    pub alpha_eff_deg: DVector<f64>,
    /// Induced (downwash) angle (degrees) at each section.
    pub alpha_induced_deg: DVector<f64>,
    /// Sectional Cl at the converged α_eff.
    pub cl_local: DVector<f64>,
    /// Sectional Cd at the converged α_eff.
    pub cd_local: DVector<f64>,
    /// Total wing CL.
    pub cl: f64,
    /// Induced drag coefficient.
    pub cdi: f64,
    /// Profile (viscous-section) drag coefficient.
    pub cd_profile: f64,
    /// Total drag coefficient `CDi + CD_profile`.
    pub cd: f64,
    /// Span efficiency `e = CL² / (π · AR · CDi)`.
    pub span_efficiency: f64,
    /// `true` if the Newton residual fell below `tol`.
    pub converged: bool,
    /// Newton steps consumed.
    pub iterations: usize,
    /// Final relative residual `‖F(Γ)‖∞ / max(‖Γ‖∞, 1)`.
    pub residual: f64,
}

/// Solver tunables. [`Default`] matches the Python `solve_lifting_line` defaults.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// Maximum Newton iterations.
    pub max_iter: usize,
    /// Convergence threshold on `‖F‖∞ / max(‖Γ‖∞, 1)`.
    pub tol: f64,
    /// FD step (degrees) for the local lift-slope `dCl/dα`.
    pub fd_step_deg: f64,
    /// Freestream speed (m/s). Cancels out of CL/CDi/e.
    pub v_inf: f64,
    /// Reference Reynolds number passed to the sectional polar.
    pub re_ref: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-10,
            fd_step_deg: 1e-3,
            v_inf: 30.0,
            re_ref: 1_000_000.0,
        }
    }
}

/// Trapezoidal-like spanwise integral matching the Python `_spanwise_integral`.
///
/// Each control point carries its entire segment width:
/// `∫ f(y) dy ≈ Σ_i f_i · (y_edges[i+1] − y_edges[i])`.
fn spanwise_integral(values: &DVector<f64>, y_edges: &DVector<f64>) -> f64 {
    let n = values.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += values[i] * (y_edges[i + 1] - y_edges[i]);
    }
    sum
}

/// Evaluate `F(Γ) = Γ − ½·V·c·Cl(α_eff)` and the intermediate fields.
///
/// Returns `(F, w_i, alpha_eff_deg, cl_local)`.
fn eval_residual(
    gamma: &DVector<f64>,
    alpha_geom_deg: &DVector<f64>,
    chord: &DVector<f64>,
    w: &DMatrix<f64>,
    section: &dyn SectionalAero,
    v_inf: f64,
    re_arr: &DVector<f64>,
) -> (DVector<f64>, DVector<f64>, DVector<f64>, DVector<f64>) {
    let w_i = w * gamma;
    let alpha_induced_deg = w_i.map(|w_val| (w_val / v_inf).to_degrees());
    let alpha_eff_deg = alpha_geom_deg - &alpha_induced_deg;
    let cl_local = section.cl(&alpha_eff_deg, re_arr);
    let mut f = DVector::<f64>::zeros(gamma.len());
    for i in 0..gamma.len() {
        f[i] = gamma[i] - 0.5 * v_inf * chord[i] * cl_local[i];
    }
    (f, w_i, alpha_eff_deg, cl_local)
}

/// Solve the nonlinear LLT system for a planar wing at a single α.
///
/// Mirror of Python `solve_lifting_line`. Returns a fully-populated
/// [`LiftingLineResult`] including the converged Γ distribution, sectional
/// fields at α_eff, and the integrated CL/CDi/CD/e.
///
/// # Errors
///
/// Returns [`AeroError::Init`] if `gamma_init` is the wrong shape, or
/// [`AeroError::Grid`] propagated from [`crate::biot_savart::downwash_matrix`].
pub fn solve_lifting_line(
    wing: &Wing,
    alpha_deg: f64,
    section: &dyn SectionalAero,
    opts: &SolverOptions,
) -> Result<LiftingLineResult> {
    solve_lifting_line_with_init(wing, alpha_deg, section, opts, None)
}

/// Same as [`solve_lifting_line`] but allows passing an initial circulation
/// distribution (used for warm-starting in [`alpha_sweep`]).
pub fn solve_lifting_line_with_init(
    wing: &Wing,
    alpha_deg: f64,
    section: &dyn SectionalAero,
    opts: &SolverOptions,
    gamma_init: Option<&DVector<f64>>,
) -> Result<LiftingLineResult> {
    let n = wing.n_sections();
    let w_matrix = downwash_matrix(&wing.y_cp, &wing.y_edges)?;
    let chord = &wing.chord_cp;
    let twist = &wing.twist_deg_cp;
    let alpha_geom_deg = twist.map(|t| alpha_deg + t);
    let re_arr = DVector::from_element(n, opts.re_ref);

    // Initial guess: 2D sectional estimate (no induced effects yet).
    let mut gamma = match gamma_init {
        Some(g) => {
            if g.len() != n {
                return Err(AeroError::Init(format!(
                    "gamma_init must have length {n}; got {}",
                    g.len()
                )));
            }
            g.clone()
        }
        None => {
            let cl0 = section.cl(&alpha_geom_deg, &re_arr);
            let mut g0 = DVector::<f64>::zeros(n);
            for i in 0..n {
                g0[i] = 0.5 * opts.v_inf * chord[i] * cl0[i];
            }
            g0
        }
    };

    let fd_step_rad = opts.fd_step_deg.to_radians();
    let mut converged = false;
    let mut iters = 0usize;

    // Buffers reused across iterations.
    let mut j_mat = DMatrix::<f64>::zeros(n, n);

    for it in 1..=opts.max_iter {
        iters = it;
        let (f, _w_i, alpha_eff, cl_local) = eval_residual(
            &gamma,
            &alpha_geom_deg,
            chord,
            &w_matrix,
            section,
            opts.v_inf,
            &re_arr,
        );
        let gamma_inf = gamma.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let f_inf = f.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let residual = f_inf / gamma_inf.max(1.0);
        if residual < opts.tol {
            converged = true;
            break;
        }

        // Local lift-slope a_i = dCl/dα|_α_eff_i (1/rad), one-sided FD.
        let alpha_eff_plus = alpha_eff.map(|a| a + opts.fd_step_deg);
        let cl_plus = section.cl(&alpha_eff_plus, &re_arr);
        let mut a_local = DVector::<f64>::zeros(n);
        for i in 0..n {
            a_local[i] = (cl_plus[i] - cl_local[i]) / fd_step_rad;
        }

        // J = I + diag(½ · c · a) · W
        // Build by scaling rows of W; faster than constructing diag mat.
        j_mat.fill(0.0);
        for i in 0..n {
            let scale = 0.5 * chord[i] * a_local[i];
            for j in 0..n {
                j_mat[(i, j)] = scale * w_matrix[(i, j)];
            }
            j_mat[(i, i)] += 1.0;
        }

        // Solve J · dGamma = F.
        let lu = j_mat.clone().lu();
        let d_gamma = match lu.solve(&f) {
            Some(dg) => dg,
            None => {
                // Singular Jacobian — damped gradient nudge and continue.
                f.clone() * 0.1
            }
        };

        // Backtracking line search.
        let f_norm = f.norm();
        let mut step = 1.0_f64;
        let mut accepted = false;
        for _ in 0..8 {
            let gamma_trial = &gamma - &d_gamma * step;
            let (f_trial, _, _, _) = eval_residual(
                &gamma_trial,
                &alpha_geom_deg,
                chord,
                &w_matrix,
                section,
                opts.v_inf,
                &re_arr,
            );
            let trial_finite = f_trial.iter().all(|x| x.is_finite());
            if trial_finite && f_trial.norm() < f_norm {
                gamma = gamma_trial;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            // No step reduced the residual; accept the smallest step anyway
            // so iteration keeps moving (typical in deep stall with shallow slope).
            gamma -= &d_gamma * step;
        }
    }

    // Final post-processing using the converged Γ.
    let (f_final, w_i, alpha_eff, cl_local) = eval_residual(
        &gamma,
        &alpha_geom_deg,
        chord,
        &w_matrix,
        section,
        opts.v_inf,
        &re_arr,
    );
    let cd_local = section.cd(&alpha_eff, &re_arr);

    let s_ref = wing.area;
    let v = opts.v_inf;
    let cl = (2.0 / (s_ref * v)) * spanwise_integral(&gamma, &wing.y_edges);
    let gamma_times_wi = DVector::from_iterator(n, (0..n).map(|i| gamma[i] * w_i[i]));
    let cdi = (2.0 / (s_ref * v * v)) * spanwise_integral(&gamma_times_wi, &wing.y_edges);
    let cd_chord = DVector::from_iterator(n, (0..n).map(|i| cd_local[i] * chord[i]));
    let cd_profile = (1.0 / s_ref) * spanwise_integral(&cd_chord, &wing.y_edges);
    let cd_total = cdi + cd_profile;

    let ar = wing.aspect_ratio();
    let e = if cdi > 1e-12 {
        (cl * cl) / (std::f64::consts::PI * ar * cdi)
    } else {
        f64::NAN
    };

    let gamma_inf = gamma.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let f_inf = f_final.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let final_residual = f_inf / gamma_inf.max(1.0);

    let alpha_induced_deg = &alpha_geom_deg - &alpha_eff;

    Ok(LiftingLineResult {
        gamma,
        alpha_eff_deg: alpha_eff,
        alpha_induced_deg,
        cl_local,
        cd_local,
        cl,
        cdi,
        cd_profile,
        cd: cd_total,
        span_efficiency: e,
        converged,
        iterations: iters,
        residual: final_residual,
    })
}

/// Output of [`alpha_sweep`].
#[derive(Debug, Clone)]
pub struct AlphaSweepResult {
    /// The α grid that was swept (degrees).
    pub alpha_deg: DVector<f64>,
    /// CL at each α.
    pub cl: DVector<f64>,
    /// Induced drag at each α.
    pub cdi: DVector<f64>,
    /// Profile drag at each α.
    pub cd_profile: DVector<f64>,
    /// Total drag at each α.
    pub cd: DVector<f64>,
    /// Span efficiency at each α.
    pub span_efficiency: DVector<f64>,
    /// Whether each solve converged.
    pub converged: Vec<bool>,
    /// Newton iterations consumed at each α.
    pub iterations: Vec<usize>,
}

/// Run [`solve_lifting_line`] across a grid of angles of attack.
///
/// With `warm_start = true` each solve is initialized from the previous α's
/// converged Γ, which dramatically reduces iteration counts when sweeping
/// through stall.
pub fn alpha_sweep(
    wing: &Wing,
    alpha_grid: &[f64],
    section: &dyn SectionalAero,
    opts: &SolverOptions,
    warm_start: bool,
) -> Result<AlphaSweepResult> {
    let m = alpha_grid.len();
    let mut cl_arr = DVector::<f64>::zeros(m);
    let mut cdi_arr = DVector::<f64>::zeros(m);
    let mut cdp_arr = DVector::<f64>::zeros(m);
    let mut cd_arr = DVector::<f64>::zeros(m);
    let mut e_arr = DVector::<f64>::zeros(m);
    let mut conv = Vec::with_capacity(m);
    let mut iter_counts = Vec::with_capacity(m);
    let mut gamma_prev: Option<DVector<f64>> = None;

    for (k, &a) in alpha_grid.iter().enumerate() {
        let init = if warm_start { gamma_prev.as_ref() } else { None };
        let res = solve_lifting_line_with_init(wing, a, section, opts, init)?;
        cl_arr[k] = res.cl;
        cdi_arr[k] = res.cdi;
        cdp_arr[k] = res.cd_profile;
        cd_arr[k] = res.cd;
        e_arr[k] = res.span_efficiency;
        conv.push(res.converged);
        iter_counts.push(res.iterations);
        if warm_start {
            gamma_prev = Some(res.gamma);
        }
    }

    Ok(AlphaSweepResult {
        alpha_deg: DVector::from_column_slice(alpha_grid),
        cl: cl_arr,
        cdi: cdi_arr,
        cd_profile: cdp_arr,
        cd: cd_arr,
        span_efficiency: e_arr,
        converged: conv,
        iterations: iter_counts,
    })
}
