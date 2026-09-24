//! Python bindings: the `ModelBank` class. Specs cross the boundary as JSON
//! (Python dicts are serialized by the thin wrapper in
//! `python/polars_online/`), and frames cross on the Arrow C Data Interface.

use online_polars::{Bank, GroupKey, Spec, StructArray, chunk_from_frame, export_struct_to_c};
use polars::prelude::PolarsError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
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

/// `(group, instance, k, n_eff, n_kish, means, comoments, cross_moments,
/// target_weights, target_means, target_vars, target_n_kish)` — the flat
/// shape `ModelBank.gram` reshapes into numpy arrays. The trailing four are
/// `None` for a state written before task 38 (docs/ENHANCEMENTS.md E45).
type GramRow = (
    Option<String>,
    String,
    usize,
    f64,
    Option<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<Option<f64>>>,
);

/// One [`GramRow`] with its lag block beside it (docs/ENHANCEMENTS.md E56):
/// `(lags, L*k*k cross-moments)`, or `None` for a spec without lags; and the
/// Gram's targets with their column means (docs/PLAN.md task 81) and their
/// cross-moments centred at those means (review 2026-09-12, N4): indices
/// into the spec's targets, and one `k`-long list of each per target.
/// Beside the row rather than in it because pyo3 converts tuples up to
/// twelve elements and the row is already twelve.
type GramRowWithLags = (
    GramRow,
    Option<(Vec<usize>, Vec<f64>)>,
    (Vec<usize>, Vec<Vec<f64>>, Vec<Vec<f64>>),
);

/// `(group, instance, n_eff, coef)` — one decay instance's flat `coef` list,
/// `None` before its first solve; `ModelBank.coef` lays it out with
/// `coef_index`.
type CoefRow = (Option<String>, String, f64, Option<Vec<f64>>);

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

/// A polars error as Python sees it: a file that could not be read or written
/// (`IO` errors, kind intact) is an `OSError`; everything else -- a column a
/// spec names that the frames lack, a bank error mid-stream -- is a
/// `ValueError` with the message.
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

/// One spec's output struct, handed to Python over the Arrow PyCapsule
/// interface instead of as a pyo3-polars `PySeries`.
///
/// A consumer -- `pl.Series`, pyarrow, duckdb -- calls `__arrow_c_array__`
/// and takes ownership of the two C structs it returns. Exporting consumes
/// the array, so a second call raises rather than hand out buffers that have
/// already been given away.
#[pyclass(name = "ArrowStruct", module = "polars_online._polars_online")]
struct PyArrowStruct {
    name: String,
    array: Option<StructArray>,
}

#[pymethods]
impl PyArrowStruct {
    /// The spec this struct is the output of. The capsule carries the field
    /// name, but `pl.Series(obj)` does not read it, so the caller names the
    /// series itself.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The Arrow PyCapsule interface: a `(schema, array)` pair of capsules
    /// named `"arrow_schema"` and `"arrow_array"`, as the specification
    /// requires.
    ///
    /// `requested_schema` is accepted and ignored. The bank emits one layout,
    /// and the specification lets a producer return its own when it cannot
    /// honour a request.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__(
        slf: &Bound<'_, Self>,
        requested_schema: Option<Bound<'_, PyAny>>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let _ = requested_schema;
        let py = slf.py();
        // The exclusive borrow, refused with a message of its own rather than
        // pyo3's "Already borrowed", as every `ModelBank` method takes its.
        let mut this = slf.try_borrow_mut().map_err(|_| {
            PyRuntimeError::new_err(
                "ArrowStruct.__arrow_c_array__: this struct is being exported on another \
                 thread; a struct exports once, so wait for that call to return",
            )
        })?;
        let st = this.array.take().ok_or_else(|| {
            PyValueError::new_err(
                "this ArrowStruct has already been exported: the Arrow PyCapsule \
                 interface hands its buffers to the consumer, so it can be read once",
            )
        })?;
        let (schema, array) = export_struct_to_c(&this.name, st);
        let schema = PyCapsule::new_with_value(py, schema, c"arrow_schema")?;
        let array = PyCapsule::new_with_value(py, array, c"arrow_array")?;
        Ok((schema.into_any().unbind(), array.into_any().unbind()))
    }
}

/// Each spec's struct array as an `ArrowStruct`, named after its spec.
fn wrap_arrow(specs: &[Spec], arrays: Vec<StructArray>) -> Vec<PyArrowStruct> {
    specs
        .iter()
        .zip(arrays)
        .map(|(s, st)| PyArrowStruct {
            name: s.name.clone(),
            array: Some(st),
        })
        .collect()
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

    /// `fit_predict` with the output handed back as Arrow: one `ArrowStruct`
    /// per spec, each exposing `__arrow_c_array__`.
    ///
    /// The values are `fit_predict`'s exactly; what differs is the way out.
    /// `PySeries` reaches py-polars' private `_export`/`_import`, which is why
    /// this package carries a polars floor and why that interface promises no
    /// stability. The PyCapsule interface is public and standardised, and any
    /// Arrow consumer can read it (docs/PLAN.md task 86).
    ///
    /// The frame still arrives as a `PyDataFrame`; moving the input over is
    /// the other half of the work.
    fn fit_predict_arrow(slf: &Bound<'_, Self>, df: PyDataFrame) -> PyResult<Vec<PyArrowStruct>> {
        let mut this = slf
            .try_borrow_mut()
            .map_err(|_| busy("fit_predict_arrow"))?;
        let bank = &mut this.inner;
        let df = df.into();
        let arrays = slf
            .py()
            .detach(|| {
                let (chunk, fresh) = chunk_from_frame(&df, bank.specs(), bank.clock_origins())?;
                let out = bank.fit_predict_arrow(&chunk)?;
                bank.commit_clock_origins(fresh);
                Ok::<_, polars::error::PolarsError>(out)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(wrap_arrow(bank.specs(), arrays))
    }

    /// `predict` with the output handed back as Arrow, as
    /// `fit_predict_arrow` is to `fit_predict`.
    fn predict_arrow(slf: &Bound<'_, Self>, df: PyDataFrame) -> PyResult<Vec<PyArrowStruct>> {
        let this = slf.try_borrow().map_err(|_| busy("predict_arrow"))?;
        let bank = &this.inner;
        let df = df.into();
        let arrays = slf
            .py()
            .detach(|| {
                let (chunk, _) = chunk_from_frame(&df, bank.specs(), bank.clock_origins())?;
                bank.predict_arrow(&chunk)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(wrap_arrow(bank.specs(), arrays))
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

    /// The state as JSON, for reading. `ValueError` when the state holds a
    /// value JSON cannot carry, rather than a quietly lossy export.
    #[pyo3(signature = (pretty = true))]
    fn save_json_string(slf: &Bound<'_, Self>, pretty: bool) -> PyResult<String> {
        let this = slf.try_borrow().map_err(|_| busy("save_json_string"))?;
        this.inner
            .save_json_string(pretty)
            .map_err(PyValueError::new_err)
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

    /// The readiness notices raised since the last call
    /// (`Bank::take_notices`), for the Python layer to warn with.
    fn take_notices(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
        let mut this = slf.try_borrow_mut().map_err(|_| busy("take_notices"))?;
        Ok(this.inner.take_notices())
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

    /// The EW accumulators behind a spec's fit (ENHANCEMENTS E30, E45), as
    /// flat tuples the Python layer reshapes into numpy arrays: see
    /// [`GramRow`].
    #[pyo3(signature = (spec, group=None))]
    fn gram(
        slf: &Bound<'_, Self>,
        spec: usize,
        group: Option<&str>,
    ) -> PyResult<Vec<GramRowWithLags>> {
        let this = slf.try_borrow().map_err(|_| busy("gram"))?;
        Ok(this
            .inner
            .gram(spec, group)
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|g| {
                (
                    (
                        g.group.0,
                        g.instance,
                        g.k,
                        g.n_eff,
                        g.n_kish,
                        g.means,
                        g.comoments,
                        g.cross_moments,
                        g.target_weights,
                        g.target_means,
                        g.target_vars,
                        g.target_n_kish,
                    ),
                    g.lags.zip(g.lag_comoments),
                    (g.targets, g.means_by_target, g.cross_centred),
                )
            })
            .collect())
    }

    /// The coefficients behind a spec's fit, per (group, instance): the flat
    /// `coef` list the output reports, as of the last row learned from.
    #[pyo3(signature = (spec, group=None))]
    fn coef(slf: &Bound<'_, Self>, spec: usize, group: Option<&str>) -> PyResult<Vec<CoefRow>> {
        let this = slf.try_borrow().map_err(|_| busy("coef"))?;
        Ok(this
            .inner
            .coef(spec, group)
            .map_err(PyValueError::new_err)?
            .into_iter()
            .map(|c| (c.group.0, c.instance, c.n_eff, c.coef))
            .collect())
    }

    /// The output struct on the last row each stream learned from, per
    /// group (`Bank::last_row`): the sorted group keys and a struct series
    /// with one row per key.
    #[pyo3(signature = (spec, group=None))]
    fn last_row(
        slf: &Bound<'_, Self>,
        spec: usize,
        group: Option<&str>,
    ) -> PyResult<(Vec<Option<String>>, PySeries)> {
        let this = slf.try_borrow().map_err(|_| busy("last_row"))?;
        let (keys, col) = this
            .inner
            .last_row(spec, group)
            .map_err(PyValueError::new_err)?;
        Ok((
            keys.into_iter().map(|k| k.0).collect(),
            PySeries(col.take_materialized_series()),
        ))
    }

    /// What each group of a spec has been fed (`Bank::summary`), one row per
    /// group.
    #[pyo3(signature = (spec, group=None))]
    fn summary(slf: &Bound<'_, Self>, spec: usize, group: Option<&str>) -> PyResult<PyDataFrame> {
        let this = slf.try_borrow().map_err(|_| busy("summary"))?;
        this.inner
            .summary(spec, group)
            .map(PyDataFrame)
            .map_err(PyValueError::new_err)
    }

    /// Per-column statistics of what each group of a spec has been fed
    /// (`Bank::describe`), one row per (group, column).
    #[pyo3(signature = (spec, group=None))]
    fn describe(slf: &Bound<'_, Self>, spec: usize, group: Option<&str>) -> PyResult<PyDataFrame> {
        let this = slf.try_borrow().map_err(|_| busy("describe"))?;
        this.inner
            .describe(spec, group)
            .map(PyDataFrame)
            .map_err(PyValueError::new_err)
    }

    /// The groups that have closed and not been read (`Bank::closed_groups`),
    /// oldest first, as one long frame. `drop` removes them from the queue.
    #[pyo3(signature = (spec=None, drop=true))]
    fn closed_groups(
        slf: &Bound<'_, Self>,
        spec: Option<usize>,
        drop: bool,
    ) -> PyResult<PyDataFrame> {
        let mut this = slf.try_borrow_mut().map_err(|_| busy("closed_groups"))?;
        Ok(PyDataFrame(
            this.inner
                .closed_groups(spec, drop)
                .map_err(PyValueError::new_err)?,
        ))
    }

    /// The pairs of a `marginal` spec (`Bank::marginal`), one row per
    /// (group, instance, feature, target).
    #[pyo3(signature = (spec, group=None))]
    fn marginal(slf: &Bound<'_, Self>, spec: usize, group: Option<&str>) -> PyResult<PyDataFrame> {
        let this = slf.try_borrow().map_err(|_| busy("marginal"))?;
        this.inner
            .marginal(spec, group)
            .map(PyDataFrame)
            .map_err(PyValueError::new_err)
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
/// Refresh-time sampling (`online_polars::RefreshTime`, E58): fed frames of
/// the long input in stream order, it returns the grid points each chunk
/// completed. State lives across calls, so any chunking gives one grid.
#[pyclass(name = "RefreshTime", module = "polars_online._polars_online")]
struct PyRefreshTime {
    inner: online_polars::RefreshTime,
    series: String,
    time: String,
    value: String,
    by: Option<String>,
    keep: Vec<String>,
}

#[pymethods]
impl PyRefreshTime {
    #[new]
    #[pyo3(signature = (names, series, time, value, by=None, pairs=false, keep=None))]
    fn new(
        names: Vec<String>,
        series: String,
        time: String,
        value: String,
        by: Option<String>,
        pairs: bool,
        keep: Option<Vec<String>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: online_polars::RefreshTime::new(names, pairs).map_err(PyValueError::new_err)?,
            series,
            time,
            value,
            by,
            keep: keep.unwrap_or_default(),
        })
    }

    /// The grid points completed by this chunk, in order.
    fn feed(slf: &Bound<'_, Self>, df: PyDataFrame) -> PyResult<PyDataFrame> {
        let mut this = slf.try_borrow_mut().map_err(|_| busy("feed"))?;
        // Destructured so the sampler and the column names are separate
        // borrows: `feed` needs `&mut` on the one and `&` on the others.
        let PyRefreshTime {
            inner,
            series,
            time,
            value,
            by,
            keep,
        } = &mut *this;
        let cols = online_polars::RefreshCols {
            series,
            time,
            value,
            by: by.as_deref(),
            keep,
        };
        Ok(PyDataFrame(
            inner.feed(&df.0, &cols).map_err(|e| run_err(&e))?,
        ))
    }
}

#[pyfunction]
fn format_of_path(path: &str) -> PyResult<&'static str> {
    online_polars::Format::from_path(std::path::Path::new(path))
        .map(|f| f.name())
        .map_err(PyValueError::new_err)
}

/// The bank's `chunk_rows` when a caller does not say.
#[pyfunction]
fn default_chunk_rows() -> usize {
    online_polars::DEFAULT_CHUNK_ROWS
}

/// Fill, validate and build a single spec, as the bank does
/// (`Spec::check`; raises ValueError with the reason).
#[pyfunction]
fn validate_spec(spec_json: &str) -> PyResult<()> {
    let mut spec: Spec = from_json(spec_json).map_err(PyValueError::new_err)?;
    spec.check().map_err(PyValueError::new_err)
}

/// The output index as JSON: one object per field with the machine values its
/// name encodes (kind, target, halflife/lam, ridge, feature_set, lambda,
/// quantile, columns). JSON keeps the FFI trivial; the Python side turns it
/// into a DataFrame.
#[pyfunction]
fn spec_output_index(spec_json: &str) -> PyResult<String> {
    let mut spec: Spec = from_json(spec_json).map_err(PyValueError::new_err)?;
    spec.check().map_err(PyValueError::new_err)?;
    serde_json::to_string(&online_polars::output_index(&spec))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Every coefficient of a spec's output as JSON: the `coef` field and
/// position it sits at, the column name `unnest` gives it, and the machine
/// values (target, halflife/lam, ridge, feature_set, lambda, term).
#[pyfunction]
fn spec_coef_fields(spec_json: &str) -> PyResult<String> {
    let mut spec: Spec = from_json(spec_json).map_err(PyValueError::new_err)?;
    spec.check().map_err(PyValueError::new_err)?;
    serde_json::to_string(&online_polars::coef_fields(&spec))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Output field names for a spec, without building a bank.
#[pyfunction]
fn spec_output_fields(spec_json: &str) -> PyResult<Vec<String>> {
    let mut spec: Spec = from_json(spec_json).map_err(PyValueError::new_err)?;
    spec.check().map_err(PyValueError::new_err)?;
    Ok(online_polars::output_fields(&spec))
}

/// Version of the compiled extension, checked against the Python package version.
#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// A duration's length in nanoseconds, from polars' duration text
/// (`"10m"`, `"1h30m"`); ValueError says what is wrong with the text.
#[pyfunction]
fn parse_duration(text: &str) -> PyResult<i64> {
    online_polars::parse_duration(text).map_err(PyValueError::new_err)
}

/// The text a duration of `ns` nanoseconds is written as: `"10m"`.
#[pyfunction]
fn format_duration(ns: i64) -> String {
    online_polars::format_duration(ns)
}

/// The parameters measured in clock units, which take a duration under a
/// temporal clock, keyed `"*"` for the shared ones and by model `type` for
/// the rest; and the rates per clock unit, which a temporal clock refuses
/// (docs/PLAN.md task 88).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn spec_clock_fields() -> (
    Vec<(&'static str, Vec<&'static str>)>,
    Vec<(&'static str, Vec<&'static str>)>,
) {
    let table = |t: &[(&'static str, &'static [&'static str])]| {
        t.iter()
            .map(|(owner, fields)| (*owner, fields.to_vec()))
            .collect()
    };
    (
        table(online_polars::CLOCK_FIELDS),
        table(online_polars::CLOCK_RATES),
    )
}

/// State-file schema version (see `online_core::SCHEMA_VERSION`).
#[pyfunction]
fn schema_version() -> u32 {
    online_core::SCHEMA_VERSION
}

/// Size of the bank's thread pool.
///
/// The pool is built at the first bank call from ``POLARS_ONLINE_MAX_THREADS``
/// (unset: one thread per core) and never resized; this builds it if nothing
/// has yet. Polars' own pool, ``POLARS_MAX_THREADS``, is separate --
/// ``pl.thread_pool_size()`` reports that one.
///
/// Raises ``ValueError`` when the variable is set to anything but a
/// non-negative integer.
#[pyfunction]
fn thread_pool_size() -> PyResult<usize> {
    online_polars::thread_pool_size().map_err(|e| PyValueError::new_err(e.to_string()))
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
    m.add_class::<PyArrowStruct>()?;
    m.add_class::<PyRefreshTime>()?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(thread_pool_size, m)?)?;
    m.add_function(wrap_pyfunction!(model_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(validate_spec, m)?)?;
    m.add_function(wrap_pyfunction!(parse_duration, m)?)?;
    m.add_function(wrap_pyfunction!(format_duration, m)?)?;
    m.add_function(wrap_pyfunction!(spec_clock_fields, m)?)?;
    m.add_function(wrap_pyfunction!(format_of_path, m)?)?;
    m.add_function(wrap_pyfunction!(default_chunk_rows, m)?)?;
    m.add_function(wrap_pyfunction!(spec_output_fields, m)?)?;
    m.add_function(wrap_pyfunction!(spec_output_index, m)?)?;
    m.add_function(wrap_pyfunction!(spec_coef_fields, m)?)?;
    Ok(())
}
