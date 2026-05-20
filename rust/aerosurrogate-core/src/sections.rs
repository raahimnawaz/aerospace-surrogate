//! Sectional 2D aerodynamic polars for the lifting-line solver.
//!
//! Mirror of Python `aerosurrogate.lifting_line.sections`.
//!
//! The solver closes its nonlinear system by querying, at every spanwise
//! station and every Newton iteration, a 2D sectional polar `Cl(α, Re)` and
//! `Cd(α, Re)`. The solver does not care where the polar comes from — only
//! that it implements the [`SectionalAero`] trait.
//!
//! Implementations in this module:
//!
//! * [`ThinAirfoilSection`] — closed-form `Cl = 2π · (α − α_{L=0})` with an
//!   optional parabolic profile-drag model `Cd = Cd0 + k·(Cl − Cl_min)²`.
//!   Used for the elliptic-wing analytical validation.
//! * [`FlatPlatePostStall`] — Hoerner flat-plate polar `Cl = sin(2α)`,
//!   `Cd = 2sin²(α) + Cd0`. Has an analytical stall at α = 45°, used to
//!   exercise the nonlinear solver post-stall.

use nalgebra::DVector;

/// Anything that returns 2D sectional `Cl` and `Cd` at given α (degrees) and Re.
///
/// Implementations must be vectorized: passing arrays of α and Re must return
/// matching-shape arrays of Cl/Cd. They must also be `Sync` so the solver can
/// hold them behind a `&dyn SectionalAero` reference across thread boundaries
/// (used in benches and the warm-started sweep).
pub trait SectionalAero: Sync {
    /// Sectional lift coefficient at each (α, Re) pair.
    fn cl(&self, alpha_deg: &DVector<f64>, re: &DVector<f64>) -> DVector<f64>;

    /// Sectional drag coefficient at each (α, Re) pair.
    fn cd(&self, alpha_deg: &DVector<f64>, re: &DVector<f64>) -> DVector<f64>;
}

// -------------------------------------------------------------------------
// Thin-airfoil polar
// -------------------------------------------------------------------------

/// Closed-form thin-airfoil polar: `Cl = 2π(α − α_{L=0})`, parabolic drag.
///
/// Lift is exact thin-airfoil theory. Drag is a parabolic polar in `Cl`:
///
/// ```text
/// Cd = Cd0 + k · (Cl − Cl_min)²
/// ```
///
/// Setting `cd0 = k = 0` gives a truly inviscid section, which the LLT
/// solver uses to recover the analytical identity `CDi = CL² / (π · AR)`
/// on an elliptic wing.
///
/// Stall is not modeled. `Cl` grows linearly with α forever.
#[derive(Debug, Default, Clone, Copy)]
pub struct ThinAirfoilSection {
    /// Maximum camber as a fraction of chord. Sets the zero-lift angle.
    pub max_camber: f64,
    /// Zero-lift profile drag coefficient.
    pub cd0: f64,
    /// Induced-of-section / pressure-drag coefficient on the parabolic polar.
    pub k: f64,
    /// Cl at minimum drag (drag bucket center).
    pub cl_min_drag: f64,
}

impl ThinAirfoilSection {
    /// Zero-lift angle of attack (degrees) for the chosen camber.
    ///
    /// For a parabolic camber line `z(x) = 4·m·x·(1−x)` the Glauert integral
    /// collapses to `α_{L=0} = −2·m` in radians.
    pub fn alpha_l0_deg(&self) -> f64 {
        (-2.0 * self.max_camber).to_degrees()
    }
}

impl SectionalAero for ThinAirfoilSection {
    fn cl(&self, alpha_deg: &DVector<f64>, _re: &DVector<f64>) -> DVector<f64> {
        let a_l0 = self.alpha_l0_deg();
        alpha_deg.map(|a| 2.0 * std::f64::consts::PI * (a - a_l0).to_radians())
    }

    fn cd(&self, alpha_deg: &DVector<f64>, re: &DVector<f64>) -> DVector<f64> {
        let cl = self.cl(alpha_deg, re);
        let cd0 = self.cd0;
        let k = self.k;
        let clm = self.cl_min_drag;
        cl.map(|c| cd0 + k * (c - clm).powi(2))
    }
}

// -------------------------------------------------------------------------
// Hoerner flat plate
// -------------------------------------------------------------------------

/// Hoerner flat-plate polar valid across `[−90°, +90°]`:
///
/// ```text
/// Cl(α) = sin(2α)         (≡ 2 sin α cos α)
/// Cd(α) = cd0 + 2 sin²(α)
/// ```
///
/// Reduces to `Cl ≈ 2α` (not `2π α`) at small α — strictly worse than
/// thin-airfoil theory in the linear regime. The purpose of this section
/// is post-stall behavior: `Cl` peaks at α = 45° and decreases beyond,
/// giving the nonlinear LLT solver a polar that genuinely stalls so we
/// can demonstrate post-stall wing behavior.
///
/// Reference: Hoerner, *Fluid-Dynamic Lift*, ch. 4.
#[derive(Debug, Default, Clone, Copy)]
pub struct FlatPlatePostStall {
    /// Zero-lift profile drag added on top of `2 sin²(α)`.
    pub cd0: f64,
}

impl SectionalAero for FlatPlatePostStall {
    fn cl(&self, alpha_deg: &DVector<f64>, _re: &DVector<f64>) -> DVector<f64> {
        alpha_deg.map(|a| (2.0 * a.to_radians()).sin())
    }

    fn cd(&self, alpha_deg: &DVector<f64>, _re: &DVector<f64>) -> DVector<f64> {
        let cd0 = self.cd0;
        alpha_deg.map(|a| cd0 + 2.0 * a.to_radians().sin().powi(2))
    }
}

// -------------------------------------------------------------------------
// Ridge ML surrogate adapter
// -------------------------------------------------------------------------

/// Use the baked-in Poly-2 Ridge surrogate as a `SectionalAero`.
///
/// The surrogate consumes 11 features: 8 geometric (max camber, camber
/// position, max thickness, leading-edge radius, trailing-edge wedge angle,
/// local thickness at 25/50/75% chord) + 3 flow (α, log₁₀ Re, n_crit).
/// Geometric features are fixed at construction so the [`SectionalAero`]
/// interface only varies α (and Re) at runtime.
///
/// This is the headline ML surrogate from the Python `aerosurrogate.models`
/// zoo, baked into the Rust binary as `pub const` arrays via
/// `rust/scripts/export_ridge_weights.py`. No Python interpreter, no sklearn,
/// no allocation in the hot path; one query is ~200 multiply-adds.
#[derive(Debug, Clone, Copy)]
pub struct RidgeSurrogateSection {
    /// Fixed geometric features baked at construction (length 8).
    pub geometry: [f64; 8],
    /// `n_crit` (transition criterion) — varies with conditions but not α.
    pub n_crit: f64,
}

impl RidgeSurrogateSection {
    /// Construct from geometric features in the order expected by the
    /// surrogate: `[max_camber, camber_pos, max_thickness, le_radius,
    /// te_angle_deg, thickness_25, thickness_50, thickness_75]`.
    pub fn from_geometry(geometry: [f64; 8], n_crit: f64) -> Self {
        Self { geometry, n_crit }
    }

    /// Convenience: build the geometry vector for a NACA 4-digit symmetric
    /// airfoil from the 4-digit code. Useful for tests and the demo CLI.
    ///
    /// The geometric features must match how the dataset was built; this
    /// helper applies the same approximation as `aerosurrogate.airfoils`
    /// uses for parametric NACA shapes.
    pub fn naca4(camber_pct: f64, camber_pos_tenth: f64, thickness_pct: f64, n_crit: f64) -> Self {
        let max_camber = camber_pct / 100.0;
        let camber_pos = camber_pos_tenth / 10.0;
        let t = thickness_pct / 100.0;
        // NACA 4-digit thickness distribution: y_t(x) = 5t [0.2969√x - 0.1260x
        // - 0.3516x² + 0.2843x³ - 0.1015x⁴].
        let nact = |x: f64| {
            5.0 * t
                * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x * x
                    + 0.2843 * x.powi(3)
                    - 0.1015 * x.powi(4))
        };
        // LE radius and TE wedge angle approximations consistent with
        // `aerosurrogate.features`.
        let le_radius = 1.1019 * t * t;
        let te_angle_deg = (0.3516_f64).atan().to_degrees(); // ~19.5°, the NACA TE angle.
        let geometry = [
            max_camber,
            camber_pos,
            t,
            le_radius,
            te_angle_deg,
            nact(0.25),
            nact(0.50),
            nact(0.75),
        ];
        Self { geometry, n_crit }
    }

    fn pack_features(&self, alpha_deg: f64, re: f64) -> [f64; 11] {
        let mut f = [0.0_f64; 11];
        f[..8].copy_from_slice(&self.geometry);
        f[8] = alpha_deg;
        f[9] = re.log10();
        f[10] = self.n_crit;
        f
    }
}

impl SectionalAero for RidgeSurrogateSection {
    fn cl(&self, alpha_deg: &DVector<f64>, re: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(
            alpha_deg.len(),
            alpha_deg
                .iter()
                .zip(re.iter())
                .map(|(&a, &r)| crate::surrogate::ridge::predict(&self.pack_features(a, r)).cl),
        )
    }

    fn cd(&self, alpha_deg: &DVector<f64>, re: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(
            alpha_deg.len(),
            alpha_deg
                .iter()
                .zip(re.iter())
                .map(|(&a, &r)| crate::surrogate::ridge::predict(&self.pack_features(a, r)).cd),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn thin_airfoil_zero_lift_at_alpha_zero_no_camber() {
        let s = ThinAirfoilSection::default();
        let a = DVector::from_vec(vec![0.0]);
        let r = DVector::from_vec(vec![1e6]);
        assert_relative_eq!(s.cl(&a, &r)[0], 0.0, max_relative = 1e-15);
    }

    #[test]
    fn thin_airfoil_slope_is_two_pi_per_radian() {
        // dCl/dα = 2π/rad. Sample two angles, take secant.
        let s = ThinAirfoilSection::default();
        let r = DVector::from_vec(vec![1e6; 2]);
        let a = DVector::from_vec(vec![0.0, 1.0]); // 0 and 1 degree
        let cl = s.cl(&a, &r);
        let slope_per_deg = cl[1] - cl[0];
        let slope_per_rad = slope_per_deg / 1.0_f64.to_radians();
        assert_relative_eq!(slope_per_rad, 2.0 * PI, max_relative = 1e-12);
    }

    #[test]
    fn thin_airfoil_cambered_has_negative_alpha_l0() {
        let s = ThinAirfoilSection {
            max_camber: 0.02,
            ..Default::default()
        };
        assert!(s.alpha_l0_deg() < 0.0);
    }

    #[test]
    fn flat_plate_peaks_at_45_degrees() {
        let s = FlatPlatePostStall::default();
        let r = DVector::from_vec(vec![1e6; 3]);
        let a = DVector::from_vec(vec![44.0, 45.0, 46.0]);
        let cl = s.cl(&a, &r);
        assert!(cl[1] > cl[0]);
        assert!(cl[1] > cl[2]);
        assert_relative_eq!(cl[1], 1.0, max_relative = 1e-12);
    }

    #[test]
    fn flat_plate_drag_is_positive_everywhere() {
        let s = FlatPlatePostStall { cd0: 0.0 };
        let a: DVector<f64> = DVector::from_iterator(
            181,
            (-90..=90).map(|i| i as f64),
        );
        let r = DVector::from_element(181, 1e6);
        let cd = s.cd(&a, &r);
        for cd_val in cd.iter() {
            assert!(*cd_val >= 0.0);
        }
    }
}
