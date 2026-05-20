//! Wing geometry for planar, unswept lifting-line analysis.
//!
//! A [`Wing`] discretizes a finite-span planar wing into `N` spanwise stations
//! with cosine spacing — denser at the tips where the loading varies fastest.
//! Each station has a chord `c(y)` and a geometric twist `θ(y)`; per-station
//! sectional aerodynamics are supplied separately via the [`crate::sections`]
//! module.
//!
//! Mirror of Python `aerosurrogate.lifting_line.geometry`.
//!
//! Cosine spacing places the i-th control point at:
//!
//! ```text
//! y_cp[i] = −(b/2) · cos((i + 0.5) · π / N)        for i = 0 … N−1
//! ```
//!
//! with segment edges between adjacent control points (and tips clamped to
//! ±b/2). This is the standard LLT grid: it concentrates resolution where
//! `dΓ/dy` is largest and avoids placing a control point at the tip
//! singularity.

use std::sync::Arc;

use nalgebra::DVector;

use crate::{AeroError, Result};

/// Field function: evaluates a scalar field (chord or twist) at arbitrary
/// spanwise locations.
pub type FieldFn = Arc<dyn Fn(&DVector<f64>) -> DVector<f64> + Send + Sync>;

/// A planar, unswept finite-span wing.
///
/// All quantities are pre-sampled at cosine-spaced control points. Factory
/// constructors ([`Wing::rectangular`], [`Wing::elliptic`], [`Wing::tapered`])
/// also record the analytical chord and twist functions, which solvers that
/// discretize differently (e.g. [`crate::classical::glauert_fourier_llt`])
/// can use to evaluate the planform exactly at their own collocation points
/// instead of interpolating from this object's grid.
///
/// `Arc` is used for the analytical field functions so the struct is `Clone`
/// (callers wanting to feed the same wing to multiple sweeps don't have to
/// rebuild the closures).
pub struct Wing {
    /// Total wingspan `b` (m), tip to tip.
    pub span: f64,
    /// Reference planform area `S = ∫ c(y) dy` (m²). Stored rather than
    /// recomputed so the analytical exact area for each factory planform
    /// is preserved (avoiding small trapezoidal-rule error).
    pub area: f64,
    /// Chord at each control point.
    pub chord_cp: DVector<f64>,
    /// Geometric twist (degrees) at each control point.
    pub twist_deg_cp: DVector<f64>,
    /// Control-point spanwise locations, shape `(N,)`.
    pub y_cp: DVector<f64>,
    /// Segment edges, shape `(N+1,)`. Control points lie at segment midpoints.
    pub y_edges: DVector<f64>,
    /// Human-readable planform name (e.g. `"elliptic"`).
    pub name: &'static str,
    chord_fn: FieldFn,
    twist_fn: FieldFn,
}

impl Clone for Wing {
    fn clone(&self) -> Self {
        Self {
            span: self.span,
            area: self.area,
            chord_cp: self.chord_cp.clone(),
            twist_deg_cp: self.twist_deg_cp.clone(),
            y_cp: self.y_cp.clone(),
            y_edges: self.y_edges.clone(),
            name: self.name,
            chord_fn: Arc::clone(&self.chord_fn),
            twist_fn: Arc::clone(&self.twist_fn),
        }
    }
}

impl std::fmt::Debug for Wing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wing")
            .field("name", &self.name)
            .field("span", &self.span)
            .field("area", &self.area)
            .field("aspect_ratio", &self.aspect_ratio())
            .field("n_sections", &self.n_sections())
            .finish_non_exhaustive()
    }
}

impl Wing {
    /// Number of spanwise control points.
    pub fn n_sections(&self) -> usize {
        self.chord_cp.len()
    }

    /// Aspect ratio `AR = b² / S`.
    pub fn aspect_ratio(&self) -> f64 {
        self.span * self.span / self.area
    }

    /// Mean aerodynamic chord `c̄ = S / b`.
    pub fn mean_chord(&self) -> f64 {
        self.area / self.span
    }

    /// Chord at arbitrary spanwise locations `y`.
    ///
    /// Uses the analytical chord function recorded by the factory
    /// constructor.
    pub fn chord_at(&self, y: &DVector<f64>) -> DVector<f64> {
        (self.chord_fn)(y)
    }

    /// Geometric twist (degrees) at arbitrary spanwise locations `y`.
    pub fn twist_at(&self, y: &DVector<f64>) -> DVector<f64> {
        (self.twist_fn)(y)
    }

    /// Cosine-spaced control points and matching segment edges.
    ///
    /// Mirrors `Wing._make_grid` in the Python reference: control points
    /// at the midpoints of N equispaced θ-intervals, edges at the boundaries.
    fn make_grid(span: f64, n: usize) -> Result<(DVector<f64>, DVector<f64>)> {
        if n < 4 {
            return Err(AeroError::Geometry(format!(
                "n_sections must be >= 4 for a meaningful LLT grid; got {n}"
            )));
        }
        if !(span > 0.0) {
            return Err(AeroError::Geometry(format!(
                "span must be > 0; got {span}"
            )));
        }
        let half = span / 2.0;
        // Control points: midpoints of N equispaced θ-intervals.
        let y_cp = DVector::from_iterator(
            n,
            (0..n).map(|i| {
                let theta = std::f64::consts::PI * (i as f64 + 0.5) / n as f64;
                -half * theta.cos()
            }),
        );
        // Edges: N+1 equispaced θ values. Tips clamped exactly to ±b/2.
        let mut y_edges = DVector::from_iterator(
            n + 1,
            (0..=n).map(|i| {
                let theta = std::f64::consts::PI * (i as f64) / n as f64;
                -half * theta.cos()
            }),
        );
        y_edges[0] = -half;
        y_edges[n] = half;
        Ok((y_cp, y_edges))
    }

    /// Constant-chord, constant-twist rectangular wing.
    ///
    /// Reference area is exact: `S = b · c`.
    pub fn rectangular(
        span: f64,
        chord: f64,
        twist_deg: f64,
        n_sections: usize,
    ) -> Result<Self> {
        if !(chord > 0.0) {
            return Err(AeroError::Geometry(format!(
                "chord must be > 0; got {chord}"
            )));
        }
        let (y_cp, y_edges) = Self::make_grid(span, n_sections)?;
        let chord_cp = DVector::from_element(n_sections, chord);
        let twist_deg_cp = DVector::from_element(n_sections, twist_deg);
        let chord_fn: FieldFn = Arc::new(move |y: &DVector<f64>| {
            DVector::from_element(y.len(), chord)
        });
        let twist_fn: FieldFn = Arc::new(move |y: &DVector<f64>| {
            DVector::from_element(y.len(), twist_deg)
        });
        Ok(Self {
            span,
            area: span * chord,
            chord_cp,
            twist_deg_cp,
            y_cp,
            y_edges,
            name: "rectangular",
            chord_fn,
            twist_fn,
        })
    }

    /// Elliptic planform with `c(y) = c_root · √(1 − (2y/b)²)`.
    ///
    /// Reference area is exact: `S = π · b · c_root / 4`.
    /// Classical LLT predicts span efficiency `e = 1` for this planform,
    /// which is the cornerstone analytical check of the solver.
    pub fn elliptic(
        span: f64,
        root_chord: f64,
        twist_deg: f64,
        n_sections: usize,
    ) -> Result<Self> {
        if !(root_chord > 0.0) {
            return Err(AeroError::Geometry(format!(
                "root_chord must be > 0; got {root_chord}"
            )));
        }
        let (y_cp, y_edges) = Self::make_grid(span, n_sections)?;
        let chord_cp = y_cp.map(|y| {
            let s = 1.0 - (2.0 * y / span).powi(2);
            root_chord * s.max(0.0).sqrt()
        });
        let twist_deg_cp = DVector::from_element(n_sections, twist_deg);
        let b = span;
        let cr = root_chord;
        let chord_fn: FieldFn = Arc::new(move |y: &DVector<f64>| {
            y.map(|yi| {
                let s = 1.0 - (2.0 * yi / b).powi(2);
                cr * s.max(0.0).sqrt()
            })
        });
        let twist_fn: FieldFn = Arc::new(move |y: &DVector<f64>| {
            DVector::from_element(y.len(), twist_deg)
        });
        Ok(Self {
            span,
            area: std::f64::consts::PI * span * root_chord / 4.0,
            chord_cp,
            twist_deg_cp,
            y_cp,
            y_edges,
            name: "elliptic",
            chord_fn,
            twist_fn,
        })
    }

    /// Linearly tapered wing with optional linear washout.
    ///
    /// `taper_ratio = c_tip / c_root`. Twist varies linearly from root
    /// (at `y=0`) to tip (at `|y|=b/2`); negative `twist_tip_deg`
    /// relative to root gives washout (tip stalls last).
    ///
    /// Reference area is exact: `S = b · c_root · (1 + λ) / 2`.
    pub fn tapered(
        span: f64,
        root_chord: f64,
        taper_ratio: f64,
        twist_root_deg: f64,
        twist_tip_deg: f64,
        n_sections: usize,
    ) -> Result<Self> {
        if !(root_chord > 0.0) {
            return Err(AeroError::Geometry(format!(
                "root_chord must be > 0; got {root_chord}"
            )));
        }
        if !(taper_ratio > 0.0) {
            return Err(AeroError::Geometry(format!(
                "taper_ratio must be > 0; got {taper_ratio}"
            )));
        }
        let (y_cp, y_edges) = Self::make_grid(span, n_sections)?;
        let one_minus_lam = 1.0 - taper_ratio;
        let b = span;
        let cr = root_chord;
        let lam = taper_ratio;
        let tr = twist_root_deg;
        let tt = twist_tip_deg;
        let chord_cp = y_cp.map(|y| {
            let eta = (2.0 * y / b).abs();
            cr * (1.0 - one_minus_lam * eta)
        });
        let twist_deg_cp = y_cp.map(|y| {
            let eta = (2.0 * y / b).abs();
            tr + (tt - tr) * eta
        });
        let chord_fn: FieldFn = Arc::new(move |y: &DVector<f64>| {
            y.map(|yi| {
                let eta = (2.0 * yi / b).abs();
                cr * (1.0 - (1.0 - lam) * eta)
            })
        });
        let twist_fn: FieldFn = Arc::new(move |y: &DVector<f64>| {
            y.map(|yi| {
                let eta = (2.0 * yi / b).abs();
                tr + (tt - tr) * eta
            })
        });
        Ok(Self {
            span,
            area: span * root_chord * (1.0 + taper_ratio) / 2.0,
            chord_cp,
            twist_deg_cp,
            y_cp,
            y_edges,
            name: "tapered",
            chord_fn,
            twist_fn,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn rectangular_area_is_exact() {
        let w = Wing::rectangular(10.0, 1.0, 0.0, 60).unwrap();
        assert_relative_eq!(w.area, 10.0, max_relative = 1e-12);
        assert_relative_eq!(w.aspect_ratio(), 10.0, max_relative = 1e-12);
        assert_eq!(w.n_sections(), 60);
    }

    #[test]
    fn elliptic_area_is_exact() {
        let w = Wing::elliptic(10.0, 1.59, 0.0, 80).unwrap();
        assert_relative_eq!(w.area, PI * 10.0 * 1.59 / 4.0, max_relative = 1e-12);
    }

    #[test]
    fn tapered_area_is_exact() {
        let w = Wing::tapered(8.0, 1.2, 0.5, 0.0, 0.0, 50).unwrap();
        assert_relative_eq!(w.area, 8.0 * 1.2 * 1.5 / 2.0, max_relative = 1e-12);
    }

    #[test]
    fn rectangular_chord_at_returns_constant() {
        let w = Wing::rectangular(10.0, 1.0, 0.0, 30).unwrap();
        let y = DVector::from_vec(vec![-4.0, -1.0, 0.0, 2.5]);
        let c = w.chord_at(&y);
        for ci in c.iter() {
            assert_relative_eq!(*ci, 1.0);
        }
    }

    #[test]
    fn elliptic_chord_at_zero_root_equals_root_chord() {
        let w = Wing::elliptic(10.0, 2.5, 0.0, 30).unwrap();
        let y = DVector::from_vec(vec![0.0, 5.0, -5.0]);
        let c = w.chord_at(&y);
        assert_relative_eq!(c[0], 2.5, max_relative = 1e-12);
        // At ±b/2 the elliptic chord vanishes.
        assert!(c[1].abs() < 1e-12);
        assert!(c[2].abs() < 1e-12);
    }

    #[test]
    fn tapered_washout_twist_at_tip_matches_tip_twist() {
        let w = Wing::tapered(8.0, 1.0, 0.5, 0.0, -3.0, 40).unwrap();
        let y = DVector::from_vec(vec![0.0, 4.0, -4.0]);
        let t = w.twist_at(&y);
        assert_relative_eq!(t[0], 0.0, max_relative = 1e-12);
        assert_relative_eq!(t[1], -3.0, max_relative = 1e-12);
        assert_relative_eq!(t[2], -3.0, max_relative = 1e-12);
    }

    #[test]
    fn y_edges_clamped_at_tips() {
        let w = Wing::rectangular(8.0, 1.0, 0.0, 40).unwrap();
        assert_relative_eq!(w.y_edges[0], -4.0, max_relative = 1e-12);
        assert_relative_eq!(w.y_edges[w.y_edges.len() - 1], 4.0, max_relative = 1e-12);
    }

    #[test]
    fn rejects_invalid_geometry() {
        assert!(Wing::rectangular(10.0, 1.0, 0.0, 2).is_err());
        assert!(Wing::rectangular(-1.0, 1.0, 0.0, 30).is_err());
        assert!(Wing::rectangular(10.0, -1.0, 0.0, 30).is_err());
        assert!(Wing::tapered(10.0, 1.0, 0.0, 0.0, 0.0, 30).is_err());
    }
}
