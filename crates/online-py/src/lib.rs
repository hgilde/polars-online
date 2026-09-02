//! Python bindings: the `online` expression namespace plugin and the `ModelBank`
//! class (docs/PLAN.md §6). Specs cross the boundary as JSON (Python dicts are
//! serialized by the thin wrapper in `python/polars_online/`).

use online_polars::{Bank, GroupKey, Spec};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
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

/// The error for a bank reached from a second thread while `fit_predict` is
/// running on the first (the GIL is released for the run).
fn busy(what: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "ModelBank.{what}: the bank is running fit_predict on another thread; a bank \
         is one ordered stream and cannot be used concurrently. Wait for the call to \
         return, or give each thread its own bank."
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

    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        let this = slf.try_borrow().map_err(|_| busy("save"))?;
        this.inner
            .save(std::path::Path::new(path))
            .map_err(PyIOError::new_err)
    }

    fn save_bytes(slf: &Bound<'_, Self>) -> PyResult<Vec<u8>> {
        let this = slf.try_borrow().map_err(|_| busy("save_bytes"))?;
        this.inner.save_bytes().map_err(PyValueError::new_err)
    }

    #[staticmethod]
    #[pyo3(signature = (path, specs_json=None))]
    fn load(path: &str, specs_json: Option<&str>) -> PyResult<Self> {
        let specs = specs_json.map(parse_specs).transpose()?;
        let inner =
            Bank::load(std::path::Path::new(path), specs.as_deref()).map_err(PyIOError::new_err)?;
        Ok(Self { inner })
    }

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
    fn output_fields(&self) -> Vec<Vec<String>> {
        self.inner
            .specs()
            .iter()
            .map(online_polars::output_fields)
            .collect()
    }

    /// Per spec, `{group_key_or_None: count}` of jittered/failed solves.
    fn solve_failures(&self) -> Vec<Vec<(Option<String>, u64)>> {
        self.inner
            .solve_failures()
            .into_iter()
            .map(|per_spec| per_spec.into_iter().map(|(k, n)| (k.0, n)).collect())
            .collect()
    }

    /// The EW accumulators behind a spec's fit (ENHANCEMENTS E30), as flat
    /// tuples the Python layer reshapes into numpy arrays:
    /// `(group, instance, k, n_eff, means, comoments, cross_moments,
    /// target_weights)`.
    #[pyo3(signature = (spec, group=None))]
    fn gram(&self, spec: usize, group: Option<&str>) -> PyResult<Vec<GramRow>> {
        Ok(self
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

    fn spec_names(&self) -> Vec<String> {
        self.inner.specs().iter().map(|s| s.name.clone()).collect()
    }

    /// The specs as JSON, so a loaded bank can show them as dicts again.
    fn specs_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.specs()).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Per spec: `(group, rows_processed, last_clock)` for every group held.
    #[allow(clippy::type_complexity)]
    fn groups(&self) -> Vec<Vec<(Option<String>, u64, Option<f64>)>> {
        self.inner
            .groups()
            .into_iter()
            .map(|v| v.into_iter().map(|(k, n, c)| (k.0, n, c)).collect())
            .collect()
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

    fn rows_seen(&self) -> u64 {
        self.inner.rows_seen()
    }
}

/// Stream a parquet file through a bank, parquet in -> parquet out
/// (ENHANCEMENTS E8). Config comes in as JSON; returns `(rows, chunks)`.
///
/// The GIL is released for the whole run, so this does not block other Python
/// threads while a large file streams.
#[pyfunction]
fn run_config(py: Python<'_>, config_json: &str) -> PyResult<(usize, usize)> {
    let cfg: online_polars::RunConfig = from_json(config_json)
        .map_err(|e| PyValueError::new_err(e.replacen("invalid spec", "invalid run config", 1)))?;
    let stats = py
        .detach(|| online_polars::run_config(&cfg, |_| {}))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((stats.rows, stats.chunks))
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

#[pymodule]
fn _polars_online(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModelBank>()?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(validate_spec, m)?)?;
    m.add_function(wrap_pyfunction!(run_config, m)?)?;
    m.add_function(wrap_pyfunction!(spec_output_fields, m)?)?;
    m.add_function(wrap_pyfunction!(spec_output_index, m)?)?;
    Ok(())
}
