//! `polars_online.gram`'s fits in Rust (docs/PLAN.md task 226): the
//! arrays come in through the buffer protocol, are read once into Rust
//! memory, and the fits run on the bank's pool with the GIL released.

use online_polars::gramfit::{FitColumns, GramPath, cd_path_of, lars_paths};
use online_polars::online_core::gramfit::{GramArrays, LarsLimits, LarsStop};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

/// One Gram as the Python layer hands it over: `k`, then `means`,
/// `comoments`, `cross_moments`, `means_by_target` and `cross_centred`, each
/// a C-contiguous float64 buffer (a numpy array).
pub(crate) type GramIn<'py> = (
    usize,
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

/// `v` as the bytes of native-endian float64s, which numpy reads with
/// `frombuffer` in one copy, where a list costs a Python float apiece.
pub(crate) fn float_bytes<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyByteArray> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    PyByteArray::new(py, &bytes)
}

/// A Gram's arrays, read into Rust memory.
pub(crate) struct OwnedGram {
    k: usize,
    means: Vec<f64>,
    comoments: Vec<f64>,
    cross_moments: Vec<f64>,
    means_by_target: Vec<f64>,
    cross_centred: Vec<f64>,
}

fn floats(py: Python<'_>, obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    PyBuffer::<f64>::get(obj)
        .and_then(|b| b.to_vec(py))
        .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

impl OwnedGram {
    pub(crate) fn read(py: Python<'_>, g: &GramIn<'_>) -> PyResult<Self> {
        let k = g.0;
        let out = Self {
            k,
            means: floats(py, &g.1, "means")?,
            comoments: floats(py, &g.2, "comoments")?,
            cross_moments: floats(py, &g.3, "cross_moments")?,
            means_by_target: floats(py, &g.4, "means_by_target")?,
            cross_centred: floats(py, &g.5, "cross_centred")?,
        };
        let m = out.cross_moments.len().checked_div(k).unwrap_or(0);
        let shapes = [
            ("means", out.means.len(), k),
            ("comoments", out.comoments.len(), k * k),
            ("cross_moments", out.cross_moments.len(), m * k),
            ("means_by_target", out.means_by_target.len(), m * k),
            ("cross_centred", out.cross_centred.len(), m * k),
        ];
        for (what, got, want) in shapes {
            if got != want {
                return Err(PyValueError::new_err(format!(
                    "{what} has {got} entries; a Gram of {k} columns and {m} targets has {want}"
                )));
            }
        }
        Ok(out)
    }

    pub(crate) fn arrays(&self) -> GramArrays<'_> {
        GramArrays {
            k: self.k,
            means: &self.means,
            comoments: &self.comoments,
            cross_moments: &self.cross_moments,
            means_by_target: &self.means_by_target,
            cross_centred: &self.cross_centred,
        }
    }
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
    let owned = grams
        .iter()
        .map(|g| OwnedGram::read(py, g))
        .collect::<PyResult<Vec<_>>>()?;
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
            let views: Vec<GramArrays<'_>> = owned.iter().map(OwnedGram::arrays).collect();
            lars_paths(&views, &targets, &cols, &weights, limits)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
    let owned = OwnedGram::read(py, &gram)?;
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

/// Every slot, the intercept and every target in range of every Gram: the
/// Python layer resolves names to positions, so this guards the indexing.
pub(crate) fn check_axes(
    grams: &[OwnedGram],
    targets: &[Vec<usize>],
    slots: &[usize],
    icept: Option<usize>,
) -> PyResult<()> {
    for (g, ts) in grams.iter().zip(targets) {
        let m = g.arrays().targets();
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
    Ok(())
}
