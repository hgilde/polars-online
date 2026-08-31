//! Python bindings: the `online` expression namespace plugin and the `ModelBank`
//! class (docs/PLAN.md §6). Specs cross the boundary as JSON (Python dicts are
//! serialized by the thin wrapper in `python/polars_online/`).

use online_polars::{Bank, Spec};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};

mod expr;

fn parse_specs(specs_json: &str) -> PyResult<Vec<Spec>> {
    serde_json::from_str(specs_json)
        .map_err(|e| PyValueError::new_err(format!("invalid spec: {e}")))
}

/// Chunk-fed model bank: feed ordered chunks, get the input chunk back with one
/// struct column appended per spec. Memory is O(state), not O(data).
// `module` matters for pickle: `__reduce__` hands back `ModelBank.load_bytes`,
// and pickle serializes that by qualified name -- which fails while the class
// claims to live in `builtins` (pyo3's default).
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

#[pyclass(name = "ModelBank", module = "polars_online._polars_online")]
struct PyModelBank {
    inner: Bank,
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
    fn fit_predict(&mut self, py: Python<'_>, df: PyDataFrame) -> PyResult<Vec<PySeries>> {
        let df = df.into();
        let cols = py
            .detach(|| self.inner.fit_predict(&df))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(cols
            .into_iter()
            .map(|c| PySeries(c.take_materialized_series()))
            .collect())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(std::path::Path::new(path))
            .map_err(PyIOError::new_err)
    }

    fn save_bytes(&self) -> PyResult<Vec<u8>> {
        self.inner.save_bytes().map_err(PyValueError::new_err)
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
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Vec<u8>,))> {
        let bytes = self.inner.save_bytes().map_err(PyValueError::new_err)?;
        let loader = py.get_type::<PyModelBank>().getattr("load_bytes")?;
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
}

/// Stream a parquet file through a bank, parquet in -> parquet out
/// (ENHANCEMENTS E8). Config comes in as JSON; returns `(rows, chunks)`.
///
/// The GIL is released for the whole run, so this does not block other Python
/// threads while a large file streams.
#[pyfunction]
fn run_config(py: Python<'_>, config_json: &str) -> PyResult<(usize, usize)> {
    let cfg: online_polars::RunConfig = serde_json::from_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("invalid run config: {e}")))?;
    let stats = py
        .detach(|| online_polars::run_config(&cfg, |_| {}))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((stats.rows, stats.chunks))
}

/// Validate a single spec (raises ValueError with the reason).
#[pyfunction]
fn validate_spec(spec_json: &str) -> PyResult<()> {
    let spec: Spec = serde_json::from_str(spec_json)
        .map_err(|e| PyValueError::new_err(format!("invalid spec: {e}")))?;
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
    let spec: Spec = serde_json::from_str(spec_json)
        .map_err(|e| PyValueError::new_err(format!("invalid spec: {e}")))?;
    spec.validate().map_err(PyValueError::new_err)?;
    serde_json::to_string(&online_polars::output_index(&spec))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Output field names for a spec, without building a bank.
#[pyfunction]
fn spec_output_fields(spec_json: &str) -> PyResult<Vec<String>> {
    let spec: Spec = serde_json::from_str(spec_json)
        .map_err(|e| PyValueError::new_err(format!("invalid spec: {e}")))?;
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
