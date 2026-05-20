//! PyO3 bindings for `aerosurrogate-core`.
//!
//! Exposes the LLT solver to Python under the module name `aerosurrogate_rs`.
//! The Python tests/scripts can then do `from aerosurrogate_rs import Wing,
//! solve_lifting_line, ...` and get the same numerical answers as the
//! reference Python solver in `src/aerosurrogate.lifting_line.*`.
//!
//! Wrapping conventions:
//!
//! * Each Rust struct gets a `#[pyclass]` wrapper that owns the underlying
//!   core type. Field access goes through `#[getter]`s, returning numpy
//!   arrays where the core has `DVector<f64>`.
//! * Sectional polar wrappers are thin shells over the core types; the
//!   Python solver entry point dispatches on the section's runtime type.

use aerosurrogate_core as core;
use nalgebra::DVector;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn dvec_to_py(py: Python<'_>, v: DVector<f64>) -> Py<PyArray1<f64>> {
    v.as_slice().to_vec().into_pyarray_bound(py).unbind()
}

fn map_err(e: core::AeroError) -> PyErr {
    match e {
        core::AeroError::Geometry(msg) => PyValueError::new_err(msg),
        core::AeroError::Grid(msg) => PyValueError::new_err(msg),
        core::AeroError::Init(msg) => PyRuntimeError::new_err(msg),
    }
}

// ------------------------------------------------------------------------
// Wing
// ------------------------------------------------------------------------

/// Planar, unswept finite-span wing. Mirror of Python
/// `aerosurrogate.lifting_line.Wing`.
#[pyclass(name = "Wing", module = "aerosurrogate_rs")]
#[derive(Clone)]
struct PyWing {
    inner: core::Wing,
}

#[pymethods]
impl PyWing {
    /// Rectangular wing: constant chord, constant twist.
    #[staticmethod]
    #[pyo3(signature = (span, chord, twist_deg=0.0, n_sections=60))]
    fn rectangular(span: f64, chord: f64, twist_deg: f64, n_sections: usize) -> PyResult<Self> {
        let w = core::Wing::rectangular(span, chord, twist_deg, n_sections).map_err(map_err)?;
        Ok(Self { inner: w })
    }

    /// Elliptic planform: `c(y) = c_root · √(1 − (2y/b)²)`.
    #[staticmethod]
    #[pyo3(signature = (span, root_chord, twist_deg=0.0, n_sections=60))]
    fn elliptic(
        span: f64,
        root_chord: f64,
        twist_deg: f64,
        n_sections: usize,
    ) -> PyResult<Self> {
        let w = core::Wing::elliptic(span, root_chord, twist_deg, n_sections).map_err(map_err)?;
        Ok(Self { inner: w })
    }

    /// Linearly tapered wing with optional washout.
    #[staticmethod]
    #[pyo3(signature = (span, root_chord, taper_ratio, twist_root_deg=0.0, twist_tip_deg=0.0, n_sections=60))]
    fn tapered(
        span: f64,
        root_chord: f64,
        taper_ratio: f64,
        twist_root_deg: f64,
        twist_tip_deg: f64,
        n_sections: usize,
    ) -> PyResult<Self> {
        let w = core::Wing::tapered(
            span,
            root_chord,
            taper_ratio,
            twist_root_deg,
            twist_tip_deg,
            n_sections,
        )
        .map_err(map_err)?;
        Ok(Self { inner: w })
    }

    #[getter]
    fn span(&self) -> f64 {
        self.inner.span
    }
    #[getter]
    fn area(&self) -> f64 {
        self.inner.area
    }
    #[getter]
    fn aspect_ratio(&self) -> f64 {
        self.inner.aspect_ratio()
    }
    #[getter]
    fn n_sections(&self) -> usize {
        self.inner.n_sections()
    }
    #[getter]
    fn chord_cp(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.chord_cp.clone())
    }
    #[getter]
    fn twist_deg_cp(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.twist_deg_cp.clone())
    }
    #[getter]
    fn y_cp(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.y_cp.clone())
    }
    #[getter]
    fn y_edges(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.y_edges.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "Wing({}, span={}, area={}, AR={:.3}, N={})",
            self.inner.name,
            self.inner.span,
            self.inner.area,
            self.inner.aspect_ratio(),
            self.inner.n_sections()
        )
    }
}

// ------------------------------------------------------------------------
// Sectional polars
// ------------------------------------------------------------------------

/// Internal enum tracking which concrete `SectionalAero` to dispatch to.
/// Keeps the Python-side API a single `section` argument while letting
/// the Rust solver hold `&dyn SectionalAero` to the right impl.
#[derive(Clone)]
enum SectionKind {
    ThinAirfoil(core::ThinAirfoilSection),
    FlatPlate(core::FlatPlatePostStall),
    Ridge(core::RidgeSurrogateSection),
}

impl SectionKind {
    fn as_dyn(&self) -> &dyn core::SectionalAero {
        match self {
            SectionKind::ThinAirfoil(s) => s,
            SectionKind::FlatPlate(s) => s,
            SectionKind::Ridge(s) => s,
        }
    }
}

/// Thin-airfoil sectional polar. Mirror of Python `ThinAirfoilSection`.
#[pyclass(name = "ThinAirfoilSection", module = "aerosurrogate_rs")]
#[derive(Clone)]
struct PyThinAirfoil {
    kind: SectionKind,
}

#[pymethods]
impl PyThinAirfoil {
    #[new]
    #[pyo3(signature = (max_camber=0.0, cd0=0.0, k=0.0, cl_min_drag=0.0))]
    fn new(max_camber: f64, cd0: f64, k: f64, cl_min_drag: f64) -> Self {
        Self {
            kind: SectionKind::ThinAirfoil(core::ThinAirfoilSection {
                max_camber,
                cd0,
                k,
                cl_min_drag,
            }),
        }
    }
}

/// Hoerner flat-plate sectional polar valid over [-90°, +90°].
#[pyclass(name = "FlatPlatePostStall", module = "aerosurrogate_rs")]
#[derive(Clone)]
struct PyFlatPlate {
    kind: SectionKind,
}

#[pymethods]
impl PyFlatPlate {
    #[new]
    #[pyo3(signature = (cd0=0.0))]
    fn new(cd0: f64) -> Self {
        Self {
            kind: SectionKind::FlatPlate(core::FlatPlatePostStall { cd0 }),
        }
    }
}

/// Baked-in Poly-2 Ridge surrogate as a sectional polar.
///
/// Geometric features are baked at construction; the LLT solver only varies
/// α and Re at runtime. Construct via `from_geometry(geometry, n_crit)` or
/// the NACA 4-digit convenience constructor `naca4(camber_pct, camber_pos_tenth,
/// thickness_pct, n_crit)`.
#[pyclass(name = "RidgeSurrogateSection", module = "aerosurrogate_rs")]
#[derive(Clone)]
struct PyRidge {
    kind: SectionKind,
}

#[pymethods]
impl PyRidge {
    #[staticmethod]
    #[pyo3(signature = (geometry, n_crit=9.0))]
    fn from_geometry(geometry: PyReadonlyArray1<f64>, n_crit: f64) -> PyResult<Self> {
        let s = geometry.as_slice()?;
        if s.len() != 8 {
            return Err(PyValueError::new_err(format!(
                "geometry must have length 8, got {}",
                s.len()
            )));
        }
        let mut g = [0.0_f64; 8];
        g.copy_from_slice(s);
        Ok(Self {
            kind: SectionKind::Ridge(core::RidgeSurrogateSection::from_geometry(g, n_crit)),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (camber_pct, camber_pos_tenth, thickness_pct, n_crit=9.0))]
    fn naca4(
        camber_pct: f64,
        camber_pos_tenth: f64,
        thickness_pct: f64,
        n_crit: f64,
    ) -> Self {
        Self {
            kind: SectionKind::Ridge(core::RidgeSurrogateSection::naca4(
                camber_pct,
                camber_pos_tenth,
                thickness_pct,
                n_crit,
            )),
        }
    }
}

trait HasSection {
    fn section(&self) -> &SectionKind;
}
impl HasSection for PyThinAirfoil {
    fn section(&self) -> &SectionKind {
        &self.kind
    }
}
impl HasSection for PyFlatPlate {
    fn section(&self) -> &SectionKind {
        &self.kind
    }
}
impl HasSection for PyRidge {
    fn section(&self) -> &SectionKind {
        &self.kind
    }
}

fn extract_section(obj: &Bound<'_, PyAny>) -> PyResult<SectionKind> {
    if let Ok(s) = obj.extract::<PyThinAirfoil>() {
        Ok(s.kind)
    } else if let Ok(s) = obj.extract::<PyFlatPlate>() {
        Ok(s.kind)
    } else if let Ok(s) = obj.extract::<PyRidge>() {
        Ok(s.kind)
    } else {
        Err(PyValueError::new_err(
            "section must be a ThinAirfoilSection / FlatPlatePostStall / RidgeSurrogateSection",
        ))
    }
}

// ------------------------------------------------------------------------
// LiftingLineResult
// ------------------------------------------------------------------------

/// Mirror of Python `LiftingLineResult`.
#[pyclass(name = "LiftingLineResult", module = "aerosurrogate_rs")]
struct PyLLResult {
    inner: core::LiftingLineResult,
}

#[pymethods]
impl PyLLResult {
    #[getter]
    fn CL(&self) -> f64 {
        self.inner.cl
    }
    #[getter]
    fn CDi(&self) -> f64 {
        self.inner.cdi
    }
    #[getter]
    fn CD_profile(&self) -> f64 {
        self.inner.cd_profile
    }
    #[getter]
    fn CD(&self) -> f64 {
        self.inner.cd
    }
    #[getter]
    fn span_efficiency(&self) -> f64 {
        self.inner.span_efficiency
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }
    #[getter]
    fn residual(&self) -> f64 {
        self.inner.residual
    }
    #[getter]
    fn Gamma(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.gamma.clone())
    }
    #[getter]
    fn alpha_eff_deg(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.alpha_eff_deg.clone())
    }
    #[getter]
    fn alpha_induced_deg(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.alpha_induced_deg.clone())
    }
    #[getter]
    fn cl_local(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.cl_local.clone())
    }
    #[getter]
    fn cd_local(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.cd_local.clone())
    }
}

// ------------------------------------------------------------------------
// Solver entry points
// ------------------------------------------------------------------------

fn build_opts(v_inf: f64, re_ref: f64, tol: f64, max_iter: usize, fd_step_deg: f64) -> core::SolverOptions {
    core::SolverOptions {
        max_iter,
        tol,
        fd_step_deg,
        v_inf,
        re_ref,
    }
}

/// Solve the nonlinear LLT system at a single α.
#[pyfunction]
#[pyo3(signature = (wing, alpha_deg, section, *, v_inf=30.0, re_ref=1_000_000.0, tol=1e-10, max_iter=100, fd_step_deg=1e-3))]
#[allow(clippy::too_many_arguments)]
fn solve_lifting_line(
    wing: PyWing,
    alpha_deg: f64,
    section: &Bound<'_, PyAny>,
    v_inf: f64,
    re_ref: f64,
    tol: f64,
    max_iter: usize,
    fd_step_deg: f64,
) -> PyResult<PyLLResult> {
    let kind = extract_section(section)?;
    let opts = build_opts(v_inf, re_ref, tol, max_iter, fd_step_deg);
    let res = core::solve_lifting_line(&wing.inner, alpha_deg, kind.as_dyn(), &opts)
        .map_err(map_err)?;
    Ok(PyLLResult { inner: res })
}

/// Warm-started α-sweep. Returns a dict of numpy arrays matching the Python
/// `alpha_sweep` API.
#[pyfunction]
#[pyo3(signature = (wing, alpha_deg, section, *, v_inf=30.0, re_ref=1_000_000.0, tol=1e-10, max_iter=100, fd_step_deg=1e-3, warm_start=true))]
#[allow(clippy::too_many_arguments)]
fn alpha_sweep<'py>(
    py: Python<'py>,
    wing: PyWing,
    alpha_deg: PyReadonlyArray1<'py, f64>,
    section: &Bound<'py, PyAny>,
    v_inf: f64,
    re_ref: f64,
    tol: f64,
    max_iter: usize,
    fd_step_deg: f64,
    warm_start: bool,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let kind = extract_section(section)?;
    let opts = build_opts(v_inf, re_ref, tol, max_iter, fd_step_deg);
    let alphas: Vec<f64> = alpha_deg.as_slice()?.to_vec();
    let out = core::alpha_sweep(&wing.inner, &alphas, kind.as_dyn(), &opts, warm_start)
        .map_err(map_err)?;

    let d = pyo3::types::PyDict::new_bound(py);
    d.set_item("alpha_deg", dvec_to_py(py, out.alpha_deg))?;
    d.set_item("CL", dvec_to_py(py, out.cl))?;
    d.set_item("CDi", dvec_to_py(py, out.cdi))?;
    d.set_item("CD_profile", dvec_to_py(py, out.cd_profile))?;
    d.set_item("CD", dvec_to_py(py, out.cd))?;
    d.set_item("span_efficiency", dvec_to_py(py, out.span_efficiency))?;
    d.set_item("converged", out.converged)?;
    d.set_item("iterations", out.iterations)?;
    Ok(d)
}

// ------------------------------------------------------------------------
// Glauert
// ------------------------------------------------------------------------

/// Mirror of Python `GlauertResult`.
#[pyclass(name = "GlauertResult", module = "aerosurrogate_rs")]
struct PyGlauertResult {
    inner: core::GlauertResult,
}

#[pymethods]
impl PyGlauertResult {
    #[getter]
    fn CL(&self) -> f64 {
        self.inner.cl
    }
    #[getter]
    fn CDi(&self) -> f64 {
        self.inner.cdi
    }
    #[getter]
    fn span_efficiency(&self) -> f64 {
        self.inner.span_efficiency
    }
    #[getter]
    fn A_n(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        dvec_to_py(py, self.inner.a_n.clone())
    }
}

/// Classical Glauert Fourier-series LLT.
#[pyfunction]
#[pyo3(signature = (wing, alpha_deg, *, lift_slope_per_rad=2.0 * std::f64::consts::PI, alpha_l0_deg=0.0, n_modes=30))]
fn glauert_fourier_llt(
    wing: PyWing,
    alpha_deg: f64,
    lift_slope_per_rad: f64,
    alpha_l0_deg: f64,
    n_modes: usize,
) -> PyResult<PyGlauertResult> {
    let res = core::glauert_fourier_llt(&wing.inner, alpha_deg, lift_slope_per_rad, alpha_l0_deg, n_modes)
        .map_err(map_err)?;
    Ok(PyGlauertResult { inner: res })
}

// ------------------------------------------------------------------------
// Module init
// ------------------------------------------------------------------------

/// Crate version.
#[pyfunction]
fn _version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn aerosurrogate_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_version, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lifting_line, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(glauert_fourier_llt, m)?)?;
    m.add_class::<PyWing>()?;
    m.add_class::<PyThinAirfoil>()?;
    m.add_class::<PyFlatPlate>()?;
    m.add_class::<PyRidge>()?;
    m.add_class::<PyLLResult>()?;
    m.add_class::<PyGlauertResult>()?;
    Ok(())
}
