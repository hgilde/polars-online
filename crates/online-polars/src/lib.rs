//! Polars-side plumbing: column extraction, per-group state, the chunk-fed model
//! bank and versioned msgpack save/load (docs/PLAN.md §5).
//!
//! ```
//! use online_polars::{Bank, Spec};
//! use polars::prelude::*;
//!
//! // A spec is what the Python builders and the CLI's TOML produce; JSON here.
//! // No `clock` column means a row-count clock, so `half_life` is in rows.
//! let spec: Spec = serde_json::from_str(r#"{
//!     "name": "ridge",
//!     "model": {"type": "ewridge", "ridge": 1e-6},
//!     "targets": ["y"],
//!     "features": ["x"],
//!     "half_life": 20.0,
//!     "min_weight": 3.0
//! }"#)?;
//! let x: Vec<f64> = (0..40).map(|i| (i % 5) as f64).collect();
//! let y: Vec<f64> = x.iter().map(|x| 1.0 + 2.0 * x).collect();
//! let df = df!("x" => x, "y" => y)?;
//!
//! // One struct column per spec, with a field per target and quantity
//! // (`pred_y`, `resid_y`, ...). `pred` is out of sample -- computed before
//! // the row's own target is learned -- and null until `min_weight`.
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

pub mod arrow;
mod atomic;
mod bank;
mod column;
mod defaults;
mod formula;
mod pool;
mod refresh;
mod resid_window;
mod resolvers;
mod rows;
mod runner;
mod span;
mod spec;
mod spec_diff;
mod stream;
mod summary;
mod targets;
mod windows;
mod windows_frame;

pub use arrow::{
    ArrowChunk, ArrowCol, ClockArray, ClockCol, chunk_from_frame, chunk_from_frame_at,
    export_struct_to_c,
};
// The Arrow types an `ArrowChunk` is built from and a `fit_predict_arrow`
// result is read as, plus the C Data Interface export a binding layer hands
// to a consumer. Re-exported so that feeding the bank as Arrow needs no
// polars crate of the caller's own (docs/PLAN.md task 86).
pub use polars_arrow::array::{Float64Array, Int64Array, StructArray, UInt64Array, Utf8ViewArray};
pub use polars_arrow::ffi::{ArrowArray, ArrowSchema, export_array_to_c, export_field_to_c};
// The string type `ArrowChunk` names its columns with. `ArrowChunk::new` takes
// anything that converts to it, `&str` included, so a caller need not hold
// one -- but may, and this is where it comes from.
pub use polars::prelude::PlSmallStr;

pub use bank::{
    Bank, Coef, CoefField, FieldMeta, Gram, GroupKey, Learned, MIN_BANK_SCHEMA_VERSION,
    PAR_MIN_ROWS, coef_fields, output_fields, output_index,
};
pub use defaults::resolved_defaults;
pub use formula::{Formula, Literal, Node, OpNode};
pub use online_core;
pub use pool::{THREADS_VAR, pool, thread_pool_size};
pub use refresh::{RefreshCols, RefreshTime};
pub use rows::FeatureRows;
pub use runner::{
    DEFAULT_CHUNK_SIZE, Format, Input, Output, RENAMED_RUN_KEYS, RunConfig, RunOptions, RunStats,
    name_renamed_run_key, run, run_config, run_config_on, run_config_on_reported,
    run_config_reported,
};
pub use span::{Span, SpanList, format_duration, parse_duration, seconds_of};
pub use spec::{
    CLOCK_FIELDS, CLOCK_RATES, ClockScale, Compare, DEPRECATED, DURATION_OR_UNIT_FREE_FIELDS,
    FloatOrList, ModelKind, Num, RENAMED, RENAMED_VALUES, RidgeScale, SessionGapSpec, ShardSpec,
    Spec, deprecation_notice, forward_deprecated, forward_deprecated_with, name_renamed,
};
pub use stream::{
    AnyModel, ChunkOut, LastRow, Stream, StreamState, build_models, combo_labels, marginal_shards,
};
pub use summary::{ColumnStats, DataSummary, Role};
pub use targets::{TargetDef, Targets};
pub use windows::{
    Closed, Direction, Emitted, KernelDef, OpDef, OpKind, Partial, Refusal, RowIn, Stat, Windows,
};
pub use windows_frame::{Like, WindowsConfig, WindowsRun};
