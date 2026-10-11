//! `po.gbdt`'s native engine (docs/PLAN.md task 242): XGBoost's and
//! LightGBM's libraries, loaded by the paths their Python packages loaded
//! them from, build the binned training objects from frames pulled through a
//! Python iterator, and hand back the handle, which
//! `python/polars_online/gbdt.py` wraps in the library's own class.
//!
//! A frame crosses on the Arrow C Data Interface, as `ModelBank`'s do. The
//! build runs with the GIL released; the reader thread takes it for each
//! frame it pulls, and never holds it while it waits for the library.

use std::collections::HashMap;
use std::sync::Arc;

use std::path::PathBuf;

use online_polars::gbdt::{
    Frame, Frames, GbdtError, IpcFrames, Kind, Layout, LgbLib, LgbOptions, Sampler, XgbLib,
    XgbOptions, conversion_pool, lightgbm_dataset, xgboost_matrix,
};
use polars::prelude::DataFrame;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator};
use pyo3_polars::PyDataFrame;

/// A build's error as a Python exception: the source's own exception as it
/// was raised, and a refusal or a library's error as `ValueError`.
fn to_py(e: GbdtError) -> PyErr {
    match e {
        GbdtError::Source(inner) => match inner.downcast::<PyErr>() {
            Ok(py) => *py,
            Err(other) => PyValueError::new_err(other.to_string()),
        },
        GbdtError::Msg(m) => PyValueError::new_err(m),
    }
}

/// Frames from a Python callable that returns a fresh iterator of
/// `pl.DataFrame`s for each pass (`lambda: plan.collect_batches(...)`).
struct PyFrames {
    factory: Py<PyAny>,
}

impl Frames for PyFrames {
    fn pass(&mut self, out: &mut dyn FnMut(Frame) -> bool) -> Result<(), GbdtError> {
        let src = |e: PyErr| GbdtError::Source(Box::new(e));
        let it: Py<PyIterator> = Python::attach(|py| -> PyResult<Py<PyIterator>> {
            Ok(self.factory.bind(py).call0()?.try_iter()?.unbind())
        })
        .map_err(src)?;
        let result = loop {
            // The GIL for the pull alone: handing the frame on may block
            // until the library wants it, and the library's log callback
            // needs the GIL meanwhile.
            let next = Python::attach(|py| -> PyResult<Option<DataFrame>> {
                let mut it = it.bind(py).clone();
                match it.next() {
                    None => Ok(None),
                    Some(frame) => Ok(Some(frame?.extract::<PyDataFrame>()?.0)),
                }
            });
            match next {
                Ok(Some(df)) => {
                    if !out(Frame::Df(df)) {
                        break Ok(());
                    }
                }
                Ok(None) => break Ok(()),
                Err(e) => break Err(src(e)),
            }
        };
        Python::attach(|_| drop(it));
        result
    }
}

/// Arrow IPC files, and each feature's Enum categories (`None` for a
/// feature that is not one).
type IpcSource = (Vec<String>, Vec<Option<Vec<String>>>);

/// The frames of a build: Arrow IPC files read here when `ipc` names them
/// (with each feature's Enum categories), otherwise the Python iterator
/// `batches` returns.
fn frames(
    batches: Py<PyAny>,
    ipc: Option<IpcSource>,
    layout: &Layout,
) -> PyResult<Box<dyn Frames>> {
    Ok(match ipc {
        Some((files, categories)) => Box::new(
            IpcFrames::new(
                files.into_iter().map(PathBuf::from).collect(),
                layout.clone(),
                categories,
            )
            .map_err(to_py)?,
        ),
        None => Box::new(PyFrames { factory: batches }),
    })
}

fn layout(
    names: Vec<String>,
    kinds: Vec<String>,
    levels: Vec<u32>,
    label: String,
    weight: Option<String>,
) -> PyResult<Layout> {
    if kinds.len() != names.len() || levels.len() != names.len() {
        return Err(PyValueError::new_err(format!(
            "{} names, {} kinds and {} level counts",
            names.len(),
            kinds.len(),
            levels.len()
        )));
    }
    let kinds = kinds
        .iter()
        .zip(&levels)
        .map(|(k, &n)| match k.as_str() {
            "float" => Ok(Kind::Float),
            "int" => Ok(Kind::Int),
            "bool" => Ok(Kind::Bool),
            "category" => Ok(Kind::Category { levels: n }),
            other => Err(PyValueError::new_err(format!("feature kind {other:?}"))),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Layout {
        names,
        kinds,
        label,
        weight,
    })
}

/// XGBoost's library, loaded from `path` (`xgboost.core._LIB.path`).
#[pyclass(name = "GbdtXgboost", module = "polars_online._polars_online", frozen)]
pub(crate) struct PyXgb {
    lib: Arc<XgbLib>,
}

#[pymethods]
impl PyXgb {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        XgbLib::load(path)
            .map(|lib| Self { lib: Arc::new(lib) })
            .map_err(to_py)
    }

    /// The library's version, `(major, minor, patch)`.
    #[getter]
    fn version(&self) -> (i32, i32, i32) {
        self.lib.version()
    }

    /// A `QuantileDMatrix` handle (which the caller owns) and the build's
    /// counts and timings.
    #[pyo3(signature = (batches, names, kinds, levels, label, weight, max_bin, threads,
                        sample_rows, seed, ref_handle, read_ahead, ipc=None))]
    #[allow(clippy::too_many_arguments)]
    fn matrix(
        &self,
        py: Python<'_>,
        batches: Py<PyAny>,
        names: Vec<String>,
        kinds: Vec<String>,
        levels: Vec<u32>,
        label: String,
        weight: Option<String>,
        max_bin: usize,
        threads: usize,
        sample_rows: Option<usize>,
        seed: u64,
        ref_handle: Option<usize>,
        read_ahead: usize,
        ipc: Option<IpcSource>,
    ) -> PyResult<(usize, HashMap<&'static str, f64>)> {
        let layout = layout(names, kinds, levels, label, weight)?;
        let frames = frames(batches, ipc, &layout)?;
        let opts = XgbOptions {
            max_bin,
            threads,
            sample_rows,
            seed,
            read_ahead,
        };
        let lib = Arc::clone(&self.lib);
        let built = py
            .detach(move || xgboost_matrix(&lib, frames, &layout, ref_handle, &opts))
            .map_err(to_py)?;
        let stats = HashMap::from([
            ("rows", built.rows as f64),
            ("passes", built.passes as f64),
            ("batches", built.batches as f64),
            ("sample_s", built.sample_s),
            ("build_s", built.build_s),
            ("waited_s", built.waited_s),
        ]);
        Ok((built.handle, stats))
    }
}

/// LightGBM's library, loaded from `path`, `version` being
/// `lightgbm.__version__`.
#[pyclass(name = "GbdtLightgbm", module = "polars_online._polars_online", frozen)]
pub(crate) struct PyLgb {
    lib: Arc<LgbLib>,
}

#[pymethods]
impl PyLgb {
    #[new]
    fn new(path: &str, version: &str) -> PyResult<Self> {
        LgbLib::load(path, version)
            .map(|lib| Self { lib: Arc::new(lib) })
            .map_err(to_py)
    }

    /// Whether this LightGBM takes a whole Arrow stream (4.7.0 on).
    #[getter]
    fn has_arrow(&self) -> bool {
        self.lib.has_arrow()
    }

    /// A `Dataset` handle (which the caller owns) and the build's counts and
    /// timings.
    #[pyo3(signature = (batches, names, kinds, levels, label, weight, params, threads, rows,
                        in_memory, reference, read_ahead, ipc=None))]
    #[allow(clippy::too_many_arguments)]
    fn dataset(
        &self,
        py: Python<'_>,
        batches: Py<PyAny>,
        names: Vec<String>,
        kinds: Vec<String>,
        levels: Vec<u32>,
        label: String,
        weight: Option<String>,
        params: String,
        threads: usize,
        rows: u64,
        in_memory: bool,
        reference: Option<usize>,
        read_ahead: usize,
        ipc: Option<IpcSource>,
    ) -> PyResult<(usize, HashMap<&'static str, f64>)> {
        let layout = layout(names, kinds, levels, label, weight)?;
        let frames = frames(batches, ipc, &layout)?;
        let opts = LgbOptions {
            params,
            threads,
            rows,
            in_memory,
            read_ahead,
        };
        let lib = Arc::clone(&self.lib);
        let built = py
            .detach(move || lightgbm_dataset(&lib, frames, &layout, reference, &opts))
            .map_err(to_py)?;
        let stats = HashMap::from([
            ("rows", built.rows as f64),
            ("passes", built.passes as f64),
            ("in_memory", f64::from(u8::from(built.in_memory))),
            ("sample_s", built.sample_s),
            ("build_s", built.build_s),
            ("waited_s", built.waited_s),
        ]);
        Ok((built.handle, stats))
    }
}

/// The sample XGBoost's cuts come from, for the Python engine: the same rows
/// and the same `ref` matrix the native engine builds, so both engines grow
/// the same trees. It needs neither library.
#[pyclass(name = "GbdtSampler", module = "polars_online._polars_online")]
pub(crate) struct PySampler {
    inner: Sampler,
    layout: Layout,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl PySampler {
    #[new]
    #[pyo3(signature = (names, kinds, levels, label, weight, sample_rows, seed, threads))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        names: Vec<String>,
        kinds: Vec<String>,
        levels: Vec<u32>,
        label: String,
        weight: Option<String>,
        sample_rows: usize,
        seed: u64,
        threads: usize,
    ) -> PyResult<Self> {
        let layout = layout(names, kinds, levels, label, weight)?;
        Ok(Self {
            inner: Sampler::new(
                layout.kinds.clone(),
                sample_rows,
                seed,
                layout.weight.is_some(),
            ),
            layout,
            pool: conversion_pool(threads).map_err(to_py)?,
        })
    }

    /// Offer the next frame of the stream.
    fn push(&mut self, py: Python<'_>, df: PyDataFrame) -> PyResult<()> {
        let (inner, layout, pool) = (&mut self.inner, &self.layout, &self.pool);
        py.detach(|| inner.push_frame(&df.0, layout, pool))
            .map_err(to_py)
    }

    /// Rows offered so far.
    #[getter]
    fn rows_seen(&self) -> u64 {
        self.inner.rows_seen()
    }

    /// The `ref` matrix: little-endian float32 bytes, row-major, and its
    /// rows and columns.
    fn ref_matrix<'py>(
        &self,
        py: Python<'py>,
        max_bin: usize,
    ) -> (Bound<'py, PyBytes>, usize, usize) {
        let (m, rows) = self.inner.ref_matrix(max_bin, &self.pool);
        let bytes: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
        (PyBytes::new(py, &bytes), rows, self.layout.names.len())
    }
}

/// Bytes of memory available to the process now: the system's available
/// memory, or a cgroup's limit less its use (polars' own reading,
/// `polars_utils::sys::MEMINFO`). `lightgbm_dataset(in_memory="auto")`
/// sizes the data against it.
#[pyfunction]
fn gbdt_available_memory() -> u64 {
    polars_utils::sys::MEMINFO.free()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gbdt_available_memory, m)?)?;
    m.add_class::<PyXgb>()?;
    m.add_class::<PyLgb>()?;
    m.add_class::<PySampler>()?;
    Ok(())
}
