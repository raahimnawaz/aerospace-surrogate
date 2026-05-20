//! Horseshoe-vortex downwash induction on the lifting line.
//!
//! Mirror of Python `aerosurrogate.lifting_line.biot_savart`.
//!
//! For an unswept, planar wing aligned with the y-axis, the bound vortex of
//! each horseshoe lies on the lifting line. A straight vortex induces no
//! velocity on its own axis, so at any control point on the lifting line
//! the only contribution to the induced downwash comes from the two
//! semi-infinite trailing legs of each horseshoe.
//!
//! For a horseshoe with bound circulation `Γ_j` and spanwise endpoints
//! `y_edge[j]` (left) and `y_edge[j+1]` (right), the induced downwash at a
//! control point `y_cp[i]` is, from Biot-Savart on two semi-infinite vortex
//! filaments and the LLT sign convention (positive downwash means flow going
//! downward, reducing the section's effective angle of attack):
//!
//! ```text
//! w_i(y_cp[i]) = (Γ_j / 4π) · [1/(y_cp[i] − y_edge[j])
//!                              − 1/(y_cp[i] − y_edge[j+1])]
//! ```
//!
//! Summing across all horseshoes gives `w = W · Γ` with the matrix W
//! computed by [`downwash_matrix`].

use nalgebra::{DMatrix, DVector};

use crate::{AeroError, Result};

/// Build the `N × N` downwash influence matrix `W` such that `w_i = W · Γ`.
///
/// # Arguments
///
/// * `y_cp` — control-point locations, shape `(N,)`.
/// * `y_edges` — segment edges, shape `(N+1,)`, strictly increasing.
///   Horseshoe `j` spans from `y_edges[j]` to `y_edges[j+1]`.
///
/// # Errors
///
/// Returns [`AeroError::Grid`] if shapes are inconsistent, edges are not
/// strictly increasing, or any control point coincides with a segment edge
/// (which would hit the 1/0 singularity of an infinitely thin vortex).
pub fn downwash_matrix(
    y_cp: &DVector<f64>,
    y_edges: &DVector<f64>,
) -> Result<DMatrix<f64>> {
    let n = y_cp.len();
    if y_edges.len() != n + 1 {
        return Err(AeroError::Grid(format!(
            "y_edges must have length N+1={n_plus_one}, got {got}",
            n_plus_one = n + 1,
            got = y_edges.len()
        )));
    }
    for k in 0..n {
        if y_edges[k + 1] <= y_edges[k] {
            return Err(AeroError::Grid(
                "y_edges must be strictly increasing".into(),
            ));
        }
    }
    let inv_4pi = 1.0 / (4.0 * std::f64::consts::PI);
    let mut w = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let yi = y_cp[i];
        for j in 0..n {
            let dy_left = yi - y_edges[j];
            let dy_right = yi - y_edges[j + 1];
            if dy_left == 0.0 || dy_right == 0.0 {
                return Err(AeroError::Grid(
                    "A control point coincides with a segment edge; use \
                     midpoint placement (e.g. cosine spacing offset by half-cell)."
                        .into(),
                ));
            }
            w[(i, j)] = inv_4pi * (1.0 / dy_left - 1.0 / dy_right);
        }
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Wing;
    use approx::assert_relative_eq;

    /// `W[i,j]` for a symmetric grid should satisfy `W[N−1−i, N−1−j] = W[i,j]`.
    ///
    /// The downwash kernel is invariant under simultaneous reflection of
    /// `y_cp` and `y_edges` about `y = 0`. With cosine-spaced grids (symmetric
    /// about zero) this gives a reflection symmetry in W. If that's broken,
    /// the indexing is wrong.
    #[test]
    fn downwash_matrix_has_reflection_symmetry() {
        let wing = Wing::rectangular(10.0, 1.0, 0.0, 20).unwrap();
        let w = downwash_matrix(&wing.y_cp, &wing.y_edges).unwrap();
        for i in 0..20 {
            for j in 0..20 {
                assert_relative_eq!(w[(i, j)], w[(19 - i, 19 - j)], epsilon = 1e-12);
            }
        }
    }

    /// Placing a control point at a segment edge must raise.
    #[test]
    fn downwash_matrix_rejects_cp_on_edge() {
        let y_edges = DVector::from_vec(vec![-1.0, 0.0, 1.0]);
        let bad_cp = DVector::from_vec(vec![-0.5, 0.0]);
        let result = downwash_matrix(&bad_cp, &y_edges);
        match result {
            Err(AeroError::Grid(msg)) => assert!(msg.contains("coincides")),
            other => panic!("expected Grid error, got {other:?}"),
        }
    }

    /// `W · Γ_elliptic` should give the *constant* `Γ_0 / (2b)` downwash
    /// distribution that classical LLT predicts for an elliptic wing.
    /// This is the same identity that the elliptic-wing CDi test relies on.
    #[test]
    fn downwash_for_elliptic_loading_is_constant() {
        use std::f64::consts::PI;
        let span = 10.0;
        let ar = 8.0;
        let c_root = 4.0 * span / (PI * ar);
        let wing = Wing::elliptic(span, c_root, 0.0, 80).unwrap();
        let w = downwash_matrix(&wing.y_cp, &wing.y_edges).unwrap();
        // Elliptic Γ(y) = Γ_0 · sqrt(1 - (2y/b)^2)
        let gamma_0 = 30.0_f64;
        let gamma = wing.y_cp.map(|y| {
            let s = 1.0 - (2.0 * y / span).powi(2);
            gamma_0 * s.max(0.0).sqrt()
        });
        let w_i = &w * &gamma;
        let expected = gamma_0 / (2.0 * span);
        for w_val in w_i.iter() {
            assert_relative_eq!(*w_val, expected, max_relative = 1e-2);
        }
    }
}
