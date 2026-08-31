//! Polars-side plumbing: column extraction, per-group state, the chunk-fed model
//! bank and versioned msgpack save/load (docs/PLAN.md §5).

mod bank;
mod runner;
mod spec;
mod stream;

pub use bank::{Bank, FieldMeta, GroupKey, output_fields, output_index};
pub use online_core;
pub use runner::{RunConfig, RunStats, run_config};
pub use spec::{FloatOrList, ModelKind, Num, SessionGapSpec, Spec};
pub use stream::{AnyModel, ChunkOut, Stream, StreamState, build_models, combo_labels};
