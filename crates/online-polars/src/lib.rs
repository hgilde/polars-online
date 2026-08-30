//! Polars-side plumbing: column extraction, per-group state, the chunk-fed model
//! bank and versioned msgpack save/load (docs/PLAN.md §5).

mod bank;
mod spec;
mod stream;

pub use bank::{Bank, output_fields};
pub use online_core;
pub use spec::{FloatOrList, ModelKind, Num, SessionGapSpec, Spec};
pub use stream::{AnyModel, RowOut, Stream, StreamState, build_models, combo_labels};
