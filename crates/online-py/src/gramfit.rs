//! `polars_online.gram`'s fits in Rust (docs/PLAN.md tasks 226 and 227):
//! the arrays come in through the buffer protocol, are read once into Rust
//! memory, and the fits run on the bank's pool with the GIL released.

use online_polars::gramfit::{
    FitColumns, GramPath, SubsetFit, SubsetOut, cd_path_of, fit_subsets, lars_paths,
};
use online_polars::online_core::gramfit::{Comoments, GramArrays, LarsLimits, LarsStop, OwnedGram};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

/// One Gram as the Python layer hands it over: `k` and `weight_sum`, then
/// `means`, `comoments`, `cross_moments`, `means_by_target`,
/// `cross_centred`, `target_weights`, `target_means`, `target_vars` and
/// `target_n_kish`, each a C-contiguous float64 buffer (a numpy array) --
/// but `comoments`, which may be float32 and whole or packed (task 229).
pub(crate) type GramIn<'py> = (
    usize,
    f64,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

/// A path as the Python layer reads it: the knots' penalties, their
/// coefficients (`n × k`, the bytes of native-endian float64s), the active
/// columns at each, and why it stopped.
type PathOut<'py> = (
    Vec<f64>,
    Bound<'py, PyByteArray>,
    Vec<Vec<usize>>,
    &'static str,
);

/// A ridge fit as the Python layer reads it: `coef`, `se` and `t` (each the
/// bytes of `k` native-endian float64s), then `resid_var`, `sigma2`, `r2`
/// and `n`.
type RidgeOut<'py> = (
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    f64,
    f64,
    f64,
    f64,
);

/// `v` as the bytes of native-endian float64s, which numpy reads with
/// `frombuffer` in one copy, where a list costs a Python float apiece.
pub(crate) fn float_bytes<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyByteArray> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    PyByteArray::new(py, &bytes)
}

/// `v` as the bytes of native-endian float32s, or float64s: a co-moment
/// matrix `ModelBank.gram` hands over in the dtype asked for. Each float32
/// is its float64 rounded to nearest (`as`).
pub(crate) fn matrix_bytes<'py>(
    py: Python<'py>,
    v: &[f64],
    float32: bool,
) -> Bound<'py, PyByteArray> {
    if !float32 {
        return float_bytes(py, v);
    }
    let bytes: Vec<u8> = v.iter().flat_map(|&x| (x as f32).to_ne_bytes()).collect();
    PyByteArray::new(py, &bytes)
}

fn floats(py: Python<'_>, obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    PyBuffer::<f64>::get(obj)
        .and_then(|b| b.to_vec(py))
        .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// A co-moment matrix as handed over: float64 or float32, whole or packed.
enum Matrix {
    F64(Vec<f64>),
    F32(Vec<f32>),
}

impl Matrix {
    fn read(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(b) = PyBuffer::<f64>::get(obj) {
            return Ok(Self::F64(b.to_vec(py)?));
        }
        PyBuffer::<f32>::get(obj)
            .and_then(|b| b.to_vec(py))
            .map(Self::F32)
            .map_err(|e| PyValueError::new_err(format!("comoments: float64 or float32, {e}")))
    }

    fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }
}

/// A Gram read into Rust memory: its co-moments as handed over, beside the
/// rest in float64.
pub(crate) struct InGram {
    base: OwnedGram,
    comoments: Matrix,
    packed: bool,
}

impl InGram {
    fn arrays(&self) -> GramArrays<'_> {
        let comoments = match (&self.comoments, self.packed) {
            (Matrix::F64(v), false) => Comoments::Full(v),
            (Matrix::F64(v), true) => Comoments::Packed(v),
            (Matrix::F32(v), false) => Comoments::Full32(v),
            (Matrix::F32(v), true) => Comoments::Packed32(v),
        };
        GramArrays {
            comoments,
            ..self.base.arrays()
        }
    }
}

/// A Gram's arrays read into Rust memory, their lengths checked. The
/// co-moments may be float64 or float32, whole (`k²` numbers) or packed
/// (`k(k+1)/2`).
fn read(py: Python<'_>, g: &GramIn<'_>) -> PyResult<InGram> {
    let k = g.0;
    let base = OwnedGram {
        k,
        weight_sum: g.1,
        means: floats(py, &g.2, "means")?,
        comoments: Vec::new(),
        cross_moments: floats(py, &g.4, "cross_moments")?,
        means_by_target: floats(py, &g.5, "means_by_target")?,
        cross_centred: floats(py, &g.6, "cross_centred")?,
        target_weights: floats(py, &g.7, "target_weights")?,
        target_means: floats(py, &g.8, "target_means")?,
        target_vars: floats(py, &g.9, "target_vars")?,
        target_n_kish: floats(py, &g.10, "target_n_kish")?,
    };
    let comoments = Matrix::read(py, &g.3)?;
    let m = base.target_weights.len();
    let packed = comoments.len() != k * k && comoments.len() == k * (k + 1) / 2;
    let shapes = [
        ("means", base.means.len(), k),
        (
            "comoments",
            comoments.len(),
            if packed { k * (k + 1) / 2 } else { k * k },
        ),
        ("cross_moments", base.cross_moments.len(), m * k),
        ("means_by_target", base.means_by_target.len(), m * k),
        ("cross_centred", base.cross_centred.len(), m * k),
        ("target_means", base.target_means.len(), m),
        ("target_vars", base.target_vars.len(), m),
        ("target_n_kish", base.target_n_kish.len(), m),
    ];
    for (what, got, want) in shapes {
        if got != want {
            return Err(PyValueError::new_err(format!(
                "{what} has {got} entries; a Gram of {k} columns and {m} targets has {want}"
            )));
        }
    }
    Ok(InGram {
        base,
        comoments,
        packed,
    })
}

fn read_all(py: Python<'_>, grams: &[GramIn<'_>]) -> PyResult<Vec<InGram>> {
    grams.iter().map(|g| read(py, g)).collect()
}

fn stop_name(stop: LarsStop) -> &'static str {
    match stop {
        LarsStop::MaxSteps => "max_steps",
        LarsStop::MaxActive => "max_active",
        LarsStop::End => "end",
    }
}

fn path_out(py: Python<'_>, p: GramPath) -> PathOut<'_> {
    let coefs = float_bytes(py, &p.coefs);
    (p.penalties, coefs, p.active, stop_name(p.stop))
}

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Each listed target's lasso path on each Gram, by least angle regression
/// (`online_core::gramfit::lars_lasso`), on the bank's pool: one list per
/// Gram, one path per target in `targets[g]`.
#[pyfunction]
#[pyo3(signature = (grams, targets, slots, icept, weights, max_steps, max_active))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gram_lars_paths<'py>(
    py: Python<'py>,
    grams: Vec<GramIn<'_>>,
    targets: Vec<Vec<usize>>,
    slots: Vec<usize>,
    icept: Option<usize>,
    weights: Vec<f64>,
    max_steps: Option<usize>,
    max_active: Option<usize>,
) -> PyResult<Vec<Vec<PathOut<'py>>>> {
    let owned = read_all(py, &grams)?;
    if targets.len() != owned.len() {
        return Err(PyValueError::new_err(format!(
            "{} target lists for {} Grams",
            targets.len(),
            owned.len()
        )));
    }
    check_axes(&owned, &targets, &slots, icept)?;
    let cols = FitColumns { slots, icept };
    let limits = LarsLimits {
        max_steps,
        max_active,
    };
    let paths = py
        .detach(|| {
            let views: Vec<GramArrays<'_>> = owned.iter().map(InGram::arrays).collect();
            lars_paths(&views, &targets, &cols, &weights, limits)
        })
        .map_err(value_err)?;
    Ok(paths
        .into_iter()
        .map(|per| per.into_iter().map(|p| path_out(py, p)).collect())
        .collect())
}

/// Target `target`'s elastic-net path on one Gram by the `lasso` model's
/// coordinate descent (`online_core::gramfit::cd_path`): `penalties × k`,
/// flat.
#[pyfunction]
#[pyo3(signature = (gram, target, slots, icept, penalties, l1_ratio, weights, max_iter, tol))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gram_cd_path(
    py: Python<'_>,
    gram: GramIn<'_>,
    target: usize,
    slots: Vec<usize>,
    icept: Option<usize>,
    penalties: Vec<f64>,
    l1_ratio: f64,
    weights: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<f64>> {
    let owned = read(py, &gram)?;
    check_axes(std::slice::from_ref(&owned), &[vec![target]], &slots, icept)?;
    if weights.len() != slots.len() {
        return Err(PyValueError::new_err(format!(
            "{} penalty weights for {} features",
            weights.len(),
            slots.len()
        )));
    }
    let cols = FitColumns { slots, icept };
    Ok(py.detach(|| {
        cd_path_of(
            &owned.arrays(),
            target,
            &cols,
            &penalties,
            l1_ratio,
            &weights,
            max_iter,
            tol,
        )
    }))
}

/// The subsets' merged Grams, checked and fitted on the pool.
fn subsets_fit(
    py: Python<'_>,
    grams: &[GramIn<'_>],
    subsets: &[Vec<usize>],
    targets: Vec<usize>,
    cols: FitColumns,
    fit: SubsetFit,
) -> PyResult<Vec<Vec<SubsetOut>>> {
    let owned = read_all(py, grams)?;
    let all: Vec<Vec<usize>> = vec![targets.clone(); owned.len()];
    check_axes(&owned, &all, &cols.slots, cols.icept)?;
    if let Some(&i) = subsets.iter().flatten().find(|&&i| i >= owned.len()) {
        return Err(PyValueError::new_err(format!(
            "subset names Gram {i} of {}",
            owned.len()
        )));
    }
    if subsets.iter().any(Vec::is_empty) {
        return Err(PyValueError::new_err("a subset names no Gram"));
    }
    py.detach(|| {
        let views: Vec<GramArrays<'_>> = owned.iter().map(InGram::arrays).collect();
        fit_subsets(&views, subsets, &targets, &cols, &fit)
    })
    .map_err(value_err)
}

/// Each subset of the Grams merged and each listed target's ridge fit on
/// it, with its statistics (`online_core::gramfit::ridge_fits`): one list
/// per subset, one fit per target.
#[pyfunction]
#[pyo3(signature = (grams, subsets, targets, slots, icept, ridge, standardize))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gram_ridge_subsets<'py>(
    py: Python<'py>,
    grams: Vec<GramIn<'_>>,
    subsets: Vec<Vec<usize>>,
    targets: Vec<usize>,
    slots: Vec<usize>,
    icept: Option<usize>,
    ridge: f64,
    standardize: bool,
) -> PyResult<Vec<Vec<RidgeOut<'py>>>> {
    let fit = SubsetFit::Ridge { ridge, standardize };
    let out = subsets_fit(
        py,
        &grams,
        &subsets,
        targets,
        FitColumns { slots, icept },
        fit,
    )?;
    Ok(out
        .into_iter()
        .map(|per| {
            per.into_iter()
                .filter_map(|o| match o {
                    SubsetOut::Ridge(f) => Some((
                        float_bytes(py, &f.coef),
                        float_bytes(py, &f.se),
                        float_bytes(py, &f.t),
                        f.resid_var,
                        f.sigma2,
                        f.r2,
                        f.n,
                    )),
                    SubsetOut::Path(_) => None,
                })
                .collect()
        })
        .collect())
}

/// Each subset of the Grams merged and each listed target's lasso path on
/// it, by least angle regression: one list per subset, one path per target.
#[pyfunction]
#[pyo3(signature = (grams, subsets, targets, slots, icept, weights, max_steps, max_active))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gram_path_subsets<'py>(
    py: Python<'py>,
    grams: Vec<GramIn<'_>>,
    subsets: Vec<Vec<usize>>,
    targets: Vec<usize>,
    slots: Vec<usize>,
    icept: Option<usize>,
    weights: Vec<f64>,
    max_steps: Option<usize>,
    max_active: Option<usize>,
) -> PyResult<Vec<Vec<PathOut<'py>>>> {
    if !weights.is_empty() && weights.len() != slots.len() {
        return Err(PyValueError::new_err(format!(
            "{} penalty weights for {} features",
            weights.len(),
            slots.len()
        )));
    }
    let fit = SubsetFit::Path {
        weights,
        limits: LarsLimits {
            max_steps,
            max_active,
        },
    };
    let out = subsets_fit(
        py,
        &grams,
        &subsets,
        targets,
        FitColumns { slots, icept },
        fit,
    )?;
    Ok(out
        .into_iter()
        .map(|per| {
            per.into_iter()
                .filter_map(|o| match o {
                    SubsetOut::Path(p) => Some(path_out(py, p)),
                    SubsetOut::Ridge(_) => None,
                })
                .collect()
        })
        .collect())
}

/// Every slot, the intercept and every target in range of every Gram: the
/// Python layer resolves names to positions, so this guards the indexing.
pub(crate) fn check_axes(
    grams: &[InGram],
    targets: &[Vec<usize>],
    slots: &[usize],
    icept: Option<usize>,
) -> PyResult<()> {
    for (g, ts) in grams.iter().map(|g| &g.base).zip(targets) {
        let m = g.target_weights.len();
        if let Some(&c) = slots.iter().chain(icept.as_ref()).find(|&&c| c >= g.k) {
            return Err(PyValueError::new_err(format!(
                "column {c} out of range for a Gram of {} columns",
                g.k
            )));
        }
        if let Some(&t) = ts.iter().find(|&&t| t >= m) {
            return Err(PyValueError::new_err(format!(
                "target {t} out of range for a Gram with {m} targets"
            )));
        }
    }
    let axes = |g: &InGram| (g.base.k, g.base.target_weights.len());
    let first = grams.first().map(axes);
    if grams.iter().any(|g| Some(axes(g)) != first) {
        return Err(PyValueError::new_err(
            "every Gram must have the same columns and targets",
        ));
    }
    Ok(())
}
