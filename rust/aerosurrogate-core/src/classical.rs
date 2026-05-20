//! Classical Glauert Fourier-series lifting-line theory.
//!
//! Mirror of Python `aerosurrogate.lifting_line.classical`.
//!
//! For a planar, unswept wing with linear sectional lift slope `a₀` and
//! zero-lift angle `α_{L=0}` (both constant along span), expand the bound
//! circulation as a sine series in `θ = arccos(−2y/b)`:
//!
//! ```text
//! Γ(θ) = 2 · b · V_∞ · Σ_{n odd} A_n · sin(n·θ)
//! ```
//!
//! Substituting into the fundamental LLT equation and collocating at `M`
//! points in `(0, π/2)` (symmetric loading uses odd modes only) gives the
//! linear system
//!
//! ```text
//! Σ_n A_n · [ 4b/(a₀ · c(θ_i)) · sin(n·θ_i) + n · sin(n·θ_i)/sin(θ_i) ]
//!     = α_geom(θ_i) − α_{L=0}                      (rad)
//! ```
//!
//! solved by a single LU decomposition. This is mathematically distinct
//! from the Newton-iteration solver in [`crate::solver`]; for a linear
//! sectional polar the two must agree to within discretization error,
//! enforced by the `glauert_vs_newton` integration test.

use nalgebra::{DMatrix, DVector};

use crate::{AeroError, Result, Wing};

/// Output of [`glauert_fourier_llt`].
#[derive(Debug, Clone)]
pub struct GlauertResult {
    /// Odd-index Fourier coefficients `A_n` for `n = 1, 3, 5, …, 2M−1`.
    pub a_n: DVector<f64>,
    /// Total wing lift coefficient `CL = π · AR · A_1`.
    pub cl: f64,
    /// Induced drag coefficient `CDi = π · AR · Σ n · A_n²`.
    pub cdi: f64,
    /// Span efficiency `e = A_1² / Σ n · A_n²`. Equals 1 for elliptic loading.
    pub span_efficiency: f64,
}

/// Solve classical Prandtl-Glauert LLT by Fourier-series collocation.
///
/// `lift_slope_per_rad` defaults to `2π` (thin-airfoil theory). Real
/// viscous airfoils sit around `5.7 − 6.1` /rad.
///
/// `n_modes` is the number of odd Fourier modes; the collocation system
/// is `M × M` with `M = n_modes`. 30 modes is overkill for the wings in
/// this crate (errors saturate around N=10-15) but cheap.
pub fn glauert_fourier_llt(
    wing: &Wing,
    alpha_deg: f64,
    lift_slope_per_rad: f64,
    alpha_l0_deg: f64,
    n_modes: usize,
) -> Result<GlauertResult> {
    if n_modes < 1 {
        return Err(AeroError::Geometry(format!(
            "n_modes must be >= 1; got {n_modes}"
        )));
    }
    let m = n_modes;
    // Odd mode indices: n = 1, 3, 5, …, 2m-1
    let n_indices: Vec<usize> = (0..m).map(|k| 2 * k + 1).collect();

    // Collocation in θ ∈ (0, π/2) only. For symmetric (odd-mode-only) loading
    // the points θ and π − θ produce identical equations — collocating across
    // the full (0, π) range would make the system rank-deficient.
    let theta = DVector::from_iterator(
        m,
        (0..m).map(|i| (i as f64 + 0.5) * (std::f64::consts::PI / 2.0) / m as f64),
    );
    let y = theta.map(|t| -(wing.span / 2.0) * t.cos());

    // Chord and twist at each collocation point.
    let chord_at = wing.chord_at(&y);
    let twist_at = wing.twist_at(&y);

    // Build the M × M system: row i of A is
    //   A[i, j] = 4b/(a₀ c_i) · sin(n_j θ_i)  +  n_j sin(n_j θ_i) / sin(θ_i)
    let mut a_matrix = DMatrix::<f64>::zeros(m, m);
    for i in 0..m {
        let sin_theta_i = theta[i].sin();
        let c_i = chord_at[i];
        for (j, &n) in n_indices.iter().enumerate() {
            let sin_n_th = (n as f64 * theta[i]).sin();
            a_matrix[(i, j)] = 4.0 * wing.span / (lift_slope_per_rad * c_i) * sin_n_th
                + (n as f64) * sin_n_th / sin_theta_i;
        }
    }

    // RHS: (α + twist − α_{L=0}) in radians at each collocation point.
    let rhs = DVector::from_iterator(
        m,
        (0..m).map(|i| (alpha_deg + twist_at[i] - alpha_l0_deg).to_radians()),
    );

    let lu = a_matrix.lu();
    let a_n = lu.solve(&rhs).ok_or_else(|| {
        AeroError::Geometry("Glauert system was singular".into())
    })?;

    let ar = wing.aspect_ratio();
    let cl = std::f64::consts::PI * ar * a_n[0];
    let cdi = std::f64::consts::PI
        * ar
        * n_indices
            .iter()
            .zip(a_n.iter())
            .map(|(&n, &a)| n as f64 * a * a)
            .sum::<f64>();

    let span_eff = if a_n[0].abs() > 1e-12 {
        let denom: f64 = n_indices
            .iter()
            .skip(1)
            .zip(a_n.iter().skip(1))
            .map(|(&n, &a)| n as f64 * (a / a_n[0]).powi(2))
            .sum();
        1.0 / (1.0 + denom)
    } else {
        f64::NAN
    };

    Ok(GlauertResult {
        a_n,
        cl,
        cdi,
        span_efficiency: span_eff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    /// An elliptic planform should give `A_n = 0` for all `n ≥ 3`.
    ///
    /// For `c(θ) = c_root · sin(θ)`, substituting into Glauert's equation
    /// and multiplying through by `sin(θ)` shows the RHS has only the
    /// `sin(θ)` component — so all higher modes must vanish algebraically.
    /// Numerically they should vanish to machine epsilon, giving `e = 1`
    /// exactly.
    #[test]
    fn elliptic_gives_unit_span_efficiency_exactly() {
        let wing = Wing::elliptic(10.0, 4.0 * 10.0 / (PI * 8.0), 0.0, 80).unwrap();
        let res = glauert_fourier_llt(&wing, 5.0, 2.0 * PI, 0.0, 20).unwrap();
        assert_relative_eq!(res.span_efficiency, 1.0, epsilon = 1e-12);
        let ratio = (res.a_n[1] / res.a_n[0]).abs();
        assert!(ratio < 1e-12, "|A_3/A_1| = {ratio:.3e}, expected ~0");
    }
}
