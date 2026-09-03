//! Python bindings: the `ModelBank` class, the runner entry point, and the
//! `online` expression namespace plugin (docs/PLAN.md §6 -- in-memory only;
//! the Python side warns on every use). Specs cross the boundary as JSON
//! (Python dicts are serialized by the thin wrapper in `python/polars_online/`).

use online_polars::{Bank, GroupKey, Spec};
use polars::prelude::PolarsError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};

/// Route this extension's allocations through the allocator py-polars is
/// using, imported from its `polars.polars._allocator` capsule (falling back
/// to the system allocator when it is absent).
///
/// **This is not what makes the two copies of Polars safe** — that is the
/// Arrow C Data Interface. Every `Series` crossing the boundary travels as a
/// `SeriesExport`: a `#[repr(C)]` struct of `ArrowSchema`/`ArrowArray`
/// pointers carrying a `release` callback *into the binary that produced it*.
/// Each side therefore frees its own memory with its own allocator, and no
/// Rust `DataFrame`, no `Drop` impl and no raw buffer ownership ever crosses.
/// Verified against py-polars 1.28.1 through 1.44.1.
///
/// What this does buy: one allocator arena in the process instead of two.
/// Polars uses jemalloc on Linux and mimalloc on Windows, so without this our
/// allocations came from a second, independent heap that could neither reuse
/// nor return pages to the first.
#[global_allocator]
static ALLOC: pyo3_polars::PolarsAllocator = pyo3_polars::PolarsAllocator::new();

/// The expression namespace plugin (docs/PLAN.md section 6). Polars hands it
/// the whole column in either engine, so `python/polars_online/_expr.py`
/// warns on every use and points at `lf.online.fit_predict` for a stream.
mod expr;

/// Parse a spec (or a list of them) from the JSON the Python builders emit.
///
/// The error names the field, not a JSON offset: `targets="y"` reads
/// `invalid spec: targets: invalid type: string "y", expected a sequence`
/// rather than `... at line 1 column 42` (docs/IMPROVEMENTS.md U2).
pub(crate) fn from_json<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, String> {
    let mut de = serde_json::Deserializer::from_str(json);
    serde_path_to_error::deserialize(&mut de).map_err(|e| {
        let path = e.path().to_string();
        // serde_json appends " at line L column C" to every message; the
        // path replaces it.
        let inner = e.into_inner().to_string();
        let msg = inner
            .rsplit_once(" at line ")
            .map_or(inner.as_str(), |(m, _)| m)
            .to_string();
        let at = if path == "." {
            String::new()
        } else {
            format!("{path}: ")
        };
        format!("invalid spec: {at}{msg}")
    })
}

fn parse_specs(specs_json: &str) -> PyResult<Vec<Spec>> {
    from_json(specs_json).map_err(PyValueError::new_err)
}

/// `(group, instance, k, n_eff, means, comoments, cross_moments, target_weights)`
/// — the flat shape `ModelBank.gram` reshapes into numpy arrays.
type GramRow = (
    Option<String>,
    String,
    usize,
    f64,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<f64>,
);

/// Chunk-fed model bank: feed ordered chunks, get the input chunk back with one
/// struct column appended per spec. Memory is O(state), not O(data).
// `module` matters for pickle: `__reduce__` hands back `ModelBank.load_bytes`,
// and pickle serializes that by qualified name -- which fails while the class
// claims to live in `builtins` (pyo3's default).
#[pyclass(name = "ModelBank", module = "polars_online._polars_online")]
struct PyModelBank {
    inner: Bank,
}

/// `e` as the `OSError` Python raises for its kind -- pyo3 chooses the
/// subclass (`FileNotFoundError`, `PermissionError`, ...) -- carrying `msg`,
/// since an `io::Error`'s own message has no path in it.
fn os_err(kind: std::io::ErrorKind, msg: String) -> PyErr {
    PyErr::from(std::io::Error::new(kind, msg))
}

/// A run's error as Python sees it: a file that could not be read or written
/// (the runner's `IO` errors, kind intact) is an `OSError`; everything else --
/// a config the runner refused, a column a spec names that the frames lack,
/// a bank error mid-stream -- is a `ValueError` with the message.
fn run_err(e: &PolarsError) -> PyErr {
    match e {
        PolarsError::IO { error, .. } => os_err(error.kind(), e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// The error for a bank reached from a second thread while `fit_predict` is
/// running on the first (the GIL is released for the run), or for a
/// `fit_predict` reached while `predict` calls are still returning. Every
/// method takes its borrow this way, so the refusal is this message and not
/// pyo3's "Already mutably borrowed".
fn busy(what: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "ModelBank.{what}: the bank is in use on another thread; a bank is one \
         ordered stream and cannot learn from two places at once (concurrent \
         `predict` calls are fine). Wait for the call to return, or give each \
         thread its own bank."
    ))
}

#[pymethods]
impl PyModelBank {
    #[new]
    fn new(specs_json: &str) -> PyResult<Self> {
        let specs = parse_specs(specs_json)?;
        let inner = Bank::new(specs).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Run all specs over one chunk; returns the output struct columns only.
    ///
    /// The GIL is released for the run, so a second thread can reach this
    /// method while the first is inside it. The borrow refuses it -- a bank is
    /// one ordered stream and cannot be fed from two places at once -- and the
    /// refusal says so, rather than pyo3's "Already borrowed".
    fn fit_predict(slf: &Bound<'_, Self>, df: PyDataFrame) -> PyResult<Vec<PySeries>> {
        let mut this = slf.try_borrow_mut().map_err(|_| busy("fit_predict"))?;
        let bank = &mut this.inner;
        let df = df.into();
        let cols = slf
            .py()
            .detach(|| bank.fit_predict(&df))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(cols
            .into_iter()
            .map(|c| PySeries(c.take_materialized_series()))
            .collect())
    }

    /// Score one chunk against the bank as it stands (`Bank::predict`): the
    /// output columns `fit_predict` would produce, and no learning. A shared
    /// borrow, so scoring threads never refuse each other; only a
    /// `fit_predict` in flight does.
    fn predict(slf: &Bound<'_, Self>, df: PyDataFrame) -> PyResult<Vec<PySeries>> {
        let this = slf.try_borrow().map_err(|_| busy("predict"))?;
        let bank = &this.inner;
        let df = df.into();
        let cols = slf
            .py()
            .detach(|| bank.predict(&df))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(cols
            .into_iter()
            .map(|c| PySeries(c.take_materialized_series()))
            .collect())
    }

    /// `Bank::save`: the filesystem's error becomes the `OSError` of its
    /// kind, with the path.
    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        let this = slf.try_borrow().map_err(|_| busy("save"))?;
        this.inner
            .save(std::path::Path::new(path))
            .map_err(|e| os_err(e.kind(), format!("{path}: {e}")))
    }

    fn save_bytes(slf: &Bound<'_, Self>) -> PyResult<Vec<u8>> {
        let this = slf.try_borrow().map_err(|_| busy("save_bytes"))?;
        this.inner.save_bytes().map_err(PyValueError::new_err)
    }

    /// `Bank::load_bytes`; a refusal is a `ValueError` with its reason. The
    /// file itself is read on the Python side (`ModelBank.load`), so that a
    /// missing one is the `FileNotFoundError` `open` raises.
    #[staticmethod]
    #[pyo3(signature = (bytes, specs_json=None))]
    fn load_bytes(bytes: &[u8], specs_json: Option<&str>) -> PyResult<Self> {
        let specs = specs_json.map(parse_specs).transpose()?;
        let inner = Bank::load_bytes(bytes, specs.as_deref()).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Pickle and `copy.deepcopy` support, routed through the same versioned
    /// msgpack as `save_bytes`/`load_bytes` -- the state file already carries
    /// the specs, so one blob reconstructs the whole bank. Production users
    /// reach for pickle without asking (multiprocessing, joblib, caching), and
    /// "cannot pickle" is a worse answer than reusing the serialization that
    /// is already tested for exact resume.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (Vec<u8>,))> {
        let this = slf.try_borrow().map_err(|_| busy("pickle"))?;
        let bytes = this.inner.save_bytes().map_err(PyValueError::new_err)?;
        let loader = slf.py().get_type::<PyModelBank>().getattr("load_bytes")?;
        Ok((loader, (bytes,)))
    }

    /// Output struct field names per spec (in order), for schema inspection.
    fn output_fields(slf: &Bound<'_, Self>) -> PyResult<Vec<Vec<String>>> {
        let this = slf.try_borrow().map_err(|_| busy("output_fields"))?;
        Ok(this
            .inner
            .specs()
            .iter()
            .map(online_polars::output_fields)
            .collect())
    }

    /// Per spec, `{group_key_or_None: count}` of jittered/failed solves.
    #[allow(clippy::type_complexity)]
    fn solve_failures(slf: &Bound<'_, Self>) -> PyResult<Vec<Vec<(Option<String>, u64)>>> {
        let this = slf.try_borrow().map_err(|_| busy("solve_failures"))?;
        Ok(this
            .inner
            .solve_failures()
            .into_iter()
            .map(|per_spec| per_spec.into_iter().map(|(k, n)| (k.0, n)).collect())
            .collect())
    }

    /// The EW accumulators behind a spec's fit (ENHANCEMENTS E30), as flat
    /// tuples the Python layer reshapes into numpy arrays:
    /// `(group, instance, k, n_eff, means, comoments, cross_moments,
    /// target_weights)`.
    #[pyo3(signature = (spec, group=None))]
    fn gram(slf: &Bound<'_, Self>, spec: usize, group: Option<&str>) -> PyResult<Vec<GramRow>> {
        let this = slf.try_borrow().map_err(|_| busy("gram"))?;
        Ok(this
            .inner
            .gram(spec, group)
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|g| {
                (
                    g.group.0,
                    g.instance,
                    g.k,
                    g.n_eff,
                    g.means,
                    g.comoments,
                    g.cross_moments,
                    g.target_weights,
                )
            })
            .collect())
    }

    fn spec_names(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
        let this = slf.try_borrow().map_err(|_| busy("spec_names"))?;
        Ok(this.inner.specs().iter().map(|s| s.name.clone()).collect())
    }

    /// The specs as JSON, so a loaded bank can show them as dicts again.
    fn specs_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        let this = slf.try_borrow().map_err(|_| busy("specs_json"))?;
        serde_json::to_string(this.inner.specs()).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Per spec: `(group, rows_processed, last_clock)` for every group held.
    #[allow(clippy::type_complexity)]
    fn groups(slf: &Bound<'_, Self>) -> PyResult<Vec<Vec<(Option<String>, u64, Option<f64>)>>> {
        let this = slf.try_borrow().map_err(|_| busy("groups"))?;
        Ok(this
            .inner
            .groups()
            .into_iter()
            .map(|v| v.into_iter().map(|(k, n, c)| (k.0, n, c)).collect())
            .collect())
    }

    #[pyo3(signature = (keys, spec=None))]
    fn drop_groups(
        slf: &Bound<'_, Self>,
        keys: Vec<Option<String>>,
        spec: Option<usize>,
    ) -> PyResult<usize> {
        let mut this = slf.try_borrow_mut().map_err(|_| busy("drop_groups"))?;
        let keys: Vec<GroupKey> = keys.into_iter().map(GroupKey).collect();
        this.inner
            .drop_groups(&keys, spec)
            .map_err(PyValueError::new_err)
    }

    fn rows_seen(slf: &Bound<'_, Self>) -> PyResult<u64> {
        let this = slf.try_borrow().map_err(|_| busy("rows_seen"))?;
        Ok(this.inner.rows_seen())
    }
}

/// The name of the format `path`'s extension says it is, or a `ValueError`
/// naming the extensions the runner knows. One extension table, in Rust.
#[pyfunction]
fn format_of_path(path: &str) -> PyResult<&'static str> {
    online_polars::Format::from_path(std::path::Path::new(path))
        .map(|f| f.name())
        .map_err(PyValueError::new_err)
}

/// The runner's format names, in the order the docs list them.
#[pyfunction]
fn formats() -> Vec<&'static str> {
    online_polars::Format::ALL
        .iter()
        .map(|f| f.name())
        .collect()
}

/// The runner's `chunk_rows` when a config does not say.
#[pyfunction]
fn default_chunk_rows() -> usize {
    online_polars::DEFAULT_CHUNK_ROWS
}

/// A Python exception raised inside the run -- by the frames iterator or the
/// progress callback -- kept whole, so the caller gets its `KeyboardInterrupt`
/// or `ZeroDivisionError` back rather than a `ValueError` with its text.
/// The run itself only sees a `PolarsError` and stops.
#[derive(Clone, Default)]
struct PyFailure(std::sync::Arc<std::sync::Mutex<Option<PyErr>>>);

impl PyFailure {
    fn set(&self, e: PyErr) -> polars::prelude::PolarsError {
        let mut slot = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let text = e.to_string();
        slot.get_or_insert(e);
        polars::prelude::PolarsError::ComputeError(text.into())
    }

    fn take(&self) -> Option<PyErr> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

/// Frames from a Python iterator, pulled on the runner's reader thread. The
/// GIL is held for the `__next__` call and the frame's export, and released
/// while the runner works; py-polars' own batch iterators release it
/// themselves while they wait for the engine, so a plan with Python UDFs
/// in it makes progress too.
struct PyFrames {
    iter: Py<pyo3::types::PyIterator>,
    failure: PyFailure,
}

impl Iterator for PyFrames {
    type Item = polars::prelude::PolarsResult<polars::prelude::DataFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        Python::attach(|py| {
            let item = self.iter.bind(py).clone().next()?;
            Some(
                item.and_then(|obj| obj.extract::<PyDataFrame>())
                    .map(|df| df.0)
                    .map_err(|e| self.failure.set(e)),
            )
        })
    }
}

/// Stream frames through a bank and write the output file the config names
/// (ENHANCEMENTS E8, E32). Config comes in as JSON; `frames` is an iterator
/// of `polars.DataFrame`s in stream order -- `polars_online.run` makes it
/// with py-polars' `collect_batches`, so the reading is py-polars' -- and
/// `schema` an empty frame with their schema, for a stream with no frames.
/// `progress`, if given, is called with `(rows, chunks)` after each chunk;
/// raising in it ends the run without publishing the output. Returns
/// `(rows, chunks)`.
///
/// The GIL is released for the run: the iterator and the callback take it
/// back for their calls only.
#[pyfunction]
#[pyo3(signature = (config_json, frames, schema, progress=None))]
fn run_config_frames(
    py: Python<'_>,
    config_json: &str,
    frames: &Bound<'_, PyAny>,
    schema: PyDataFrame,
    progress: Option<Py<PyAny>>,
) -> PyResult<(usize, usize)> {
    let cfg: online_polars::RunConfig = from_json(config_json)
        .map_err(|e| PyValueError::new_err(e.replacen("invalid spec", "invalid run config", 1)))?;
    let failure = PyFailure::default();
    let frames = PyFrames {
        iter: frames.try_iter()?.unbind(),
        failure: failure.clone(),
    };
    let input = online_polars::Input::Batches {
        frames: Box::new(frames),
        schema: schema.0.schema().as_ref().clone(),
    };
    let report = |s: online_polars::RunStats| match &progress {
        None => Ok(()),
        Some(cb) => Python::attach(|py| {
            cb.call1(py, (s.rows, s.chunks))
                .map(|_| ())
                .map_err(|e| failure.set(e))
        }),
    };
    let result = py.detach(|| online_polars::run_config_on(&cfg, input, report));
    match result {
        Ok(stats) => Ok((stats.rows, stats.chunks)),
        Err(e) => Err(failure.take().unwrap_or_else(|| run_err(&e))),
    }
}

/// Validate a single spec (raises ValueError with the reason).
#[pyfunction]
fn validate_spec(spec_json: &str) -> PyResult<()> {
    let spec: Spec = from_json(spec_json).map_err(PyValueError::new_err)?;
    spec.validate().map_err(PyValueError::new_err)?;
    online_polars::build_models(&spec)
        .map(|_| ())
        .map_err(PyValueError::new_err)
}

/// The output index as JSON: one object per field with the machine values its
/// name encodes (kind, target, halflife/lam, ridge, feature_set, lambda,
/// quantile, columns). JSON keeps the FFI trivial; the Python side turns it
/// into a DataFrame.
#[pyfunction]
fn spec_output_index(spec_json: &str) -> PyResult<String> {
    let spec: Spec = from_json(spec_json).map_err(PyValueError::new_err)?;
    spec.validate().map_err(PyValueError::new_err)?;
    serde_json::to_string(&online_polars::output_index(&spec))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Output field names for a spec, without building a bank.
#[pyfunction]
fn spec_output_fields(spec_json: &str) -> PyResult<Vec<String>> {
    let spec: Spec = from_json(spec_json).map_err(PyValueError::new_err)?;
    spec.validate().map_err(PyValueError::new_err)?;
    Ok(online_polars::output_fields(&spec))
}

/// Version of the compiled extension, checked against the Python package version.
#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// State-file schema version (see `online_core::SCHEMA_VERSION`).
#[pyfunction]
fn schema_version() -> u32 {
    online_core::SCHEMA_VERSION
}

/// Every model this build can construct, as spec `type` names. What the
/// Python builders and the per-model test sweeps are checked against
/// (docs/EXTENDING.md).
#[pyfunction]
fn model_kinds() -> Vec<&'static str> {
    online_polars::ModelKind::KINDS.to_vec()
}

#[pymodule]
fn _polars_online(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModelBank>()?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(model_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(validate_spec, m)?)?;
    m.add_function(wrap_pyfunction!(run_config_frames, m)?)?;
    m.add_function(wrap_pyfunction!(format_of_path, m)?)?;
    m.add_function(wrap_pyfunction!(formats, m)?)?;
    m.add_function(wrap_pyfunction!(default_chunk_rows, m)?)?;
    m.add_function(wrap_pyfunction!(spec_output_fields, m)?)?;
    m.add_function(wrap_pyfunction!(spec_output_index, m)?)?;
    Ok(())
}
