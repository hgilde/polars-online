//! Polars-side plumbing: column extraction, per-group state, the chunk-fed model
//! bank and versioned msgpack save/load (docs/PLAN.md §5).
//!
//! ```
//! use online_polars::{Bank, Spec};
//! use polars::prelude::*;
//!
//! // A spec is what the Python builders and the CLI's TOML produce; JSON here.
//! // No `clock` column means a row-count clock, so `halflife` is in rows.
//! let spec: Spec = serde_json::from_str(r#"{
//!     "name": "ridge",
//!     "model": {"type": "ew_ridge", "ridge": 1e-6},
//!     "targets": ["y"],
//!     "features": ["x"],
//!     "halflife": 20.0,
//!     "min_periods": 3.0
//! }"#)?;
//! let x: Vec<f64> = (0..40).map(|i| (i % 5) as f64).collect();
//! let y: Vec<f64> = x.iter().map(|x| 1.0 + 2.0 * x).collect();
//! let df = df!("x" => x, "y" => y)?;
//!
//! // One struct column per spec, with a field per target and quantity
//! // (`pred_y`, `resid_y`, ...). `pred` is out of sample -- computed before
//! // the row's own target is learned -- and null until `min_periods`.
//! let mut bank = Bank::new(vec![spec])?;
//! let out = bank.fit_predict(&df.slice(0, 20))?;
//! let pred = out[0].as_materialized_series().struct_()?.field_by_name("pred_y")?;
//! assert_eq!(pred.get(0)?, AnyValue::Null);
//! assert!((pred.f64()?.get(19).unwrap() - 9.0).abs() < 1e-4, "x = 4, so y = 9");
//!
//! // State is a versioned msgpack blob: a loaded bank continues where the
//! // saved one stopped, with identical output.
//! let bytes = bank.save_bytes()?;
//! let mut resumed = Bank::load_bytes(&bytes, Some(bank.specs()))?;
//! let rest = df.slice(20, 20);
//! let (a, b) = (bank.fit_predict(&rest)?, resumed.fit_predict(&rest)?);
//! assert!(a[0].as_materialized_series().equals_missing(b[0].as_materialized_series()));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod atomic;
mod bank;
mod pool;
mod runner;
mod spec;
mod stream;

pub use bank::{
    Bank, Coef, CoefField, FieldMeta, Gram, GroupKey, PAR_MIN_ROWS, coef_fields, output_fields,
    output_index,
};
pub use online_core;
pub use pool::{THREADS_VAR, pool, thread_pool_size};
pub use runner::{
    DEFAULT_CHUNK_ROWS, Format, Input, Output, RunConfig, RunOptions, RunStats, run, run_config,
    run_config_on,
};
pub use spec::{Compare, FloatOrList, ModelKind, Num, SessionGapSpec, Spec};
pub use stream::{AnyModel, ChunkOut, LastRow, Stream, StreamState, build_models, combo_labels};
