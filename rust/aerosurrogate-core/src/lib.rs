//! # aerosurrogate-core
//!
//! Nonlinear lifting-line theory solver for planar, unswept finite-span
//! wings, coupling any 2D sectional aerodynamic polar with the classical
//! Prandtl induced-downwash kernel via Newton iteration in the modern
//! Phillips & Snyder (2000) style.
//!
//! This crate is a faithful Rust port of the Python `aerosurrogate.lifting_line`
//! package, intended for embedded / real-time use where the Python
//! implementation's millisecond-scale per-α latency is too slow. Numerical
//! results agree with the Python reference to ≤ 1e-10 on every wing
//! configuration in the test suite.
//!
//! ## Design
//!
//! Three layers:
//!
//! * [`geometry`] — `Wing` struct with [`Wing::rectangular`], [`Wing::elliptic`],
//!   and [`Wing::tapered`] factory constructors. Cosine spanwise grid;
//!   analytical chord/twist functions retained alongside the sampled
//!   control-point values so the [`classical`] Glauert solver can evaluate
//!   the planform exactly at its own collocation points.
//! * [`sections`] — [`SectionalAero`] trait + analytical implementations
//!   ([`sections::ThinAirfoilSection`], [`sections::FlatPlatePostStall`])
//!   and the baked-in sklearn surrogates ([`surrogate::ridge`],
//!   [`surrogate::gbm`]). All implementations are pure scalar arithmetic with
//!   no allocation in the hot path.
//! * [`solver`] — Newton iteration with a finite-difference Jacobian and
//!   backtracking line search ([`solver::solve_lifting_line`]). An
//!   independent reference implementation ([`classical::glauert_fourier_llt`])
//!   solves the same problem by half-span Fourier-series collocation;
//!   agreement between the two is enforced as an integration test.
//!
//! ## Status: skeleton
//!
//! Modules are stubbed and will be filled in following the order documented
//! in `i-am-planning-on-vast-lampson.md`.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod biot_savart;
pub mod classical;
pub mod geometry;
pub mod sections;
pub mod solver;
pub mod surrogate;

pub use biot_savart::downwash_matrix;
pub use classical::{glauert_fourier_llt, GlauertResult};
pub use geometry::Wing;
pub use sections::{FlatPlatePostStall, RidgeSurrogateSection, SectionalAero, ThinAirfoilSection};
pub use solver::{
    alpha_sweep, solve_lifting_line, AlphaSweepResult, LiftingLineResult, SolverOptions,
};

/// Library error type.
///
/// Returned by constructors and the solver when invariants are violated
/// (e.g. negative span, collocation point coincident with a segment edge,
/// inconsistent input shapes).
#[derive(Debug, thiserror::Error)]
pub enum AeroError {
    /// A geometric input violated an invariant (negative span, taper ratio
    /// ≤ 0, fewer than 4 sections, etc.).
    #[error("invalid geometry: {0}")]
    Geometry(String),

    /// A grid was passed to a routine that requires monotonicity or
    /// non-overlap with control points; the routine cannot proceed.
    #[error("invalid grid: {0}")]
    Grid(String),

    /// The solver received an initial guess of the wrong shape.
    #[error("invalid initial guess: {0}")]
    Init(String),
}

/// Crate-wide [`Result`](::std::result::Result) alias using [`AeroError`].
pub type Result<T> = ::std::result::Result<T, AeroError>;
