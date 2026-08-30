//! Python bindings: the `online` expression namespace plugin and the `ModelBank` class.
//!
//! See `docs/PLAN.md` §6. Scaffold only: expressions land in task 8, `ModelBank` in task 7.
//! For now the module exists so `maturin develop` produces an importable, loadable plugin.

use pyo3::prelude::*;

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
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    Ok(())
}
