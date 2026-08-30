//! Pure-Rust online (streaming) regression models.
//!
//! This crate knows nothing about Polars, Python, or clocks-as-columns: it consumes
//! one row at a time (`&[f64]` features, `&[Option<f64>]` targets, a clock delta and a
//! weight) and produces a [`Step`]. All plumbing lives in `online-polars` / `online-py`.
//!
//! See `docs/PLAN.md` §2 and §4. Scaffold only: the trait and models land in task 3+.

/// Version of the serialized model-state layout.
///
/// Bump on any state layout change and keep a loader for the previous version
/// (`docs/PLAN.md`, hard rule 5).
pub const SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    /// The state schema version must be a real, monotonically bumped version (rule 5).
    #[test]
    fn schema_version_is_set() {
        const { assert!(SCHEMA_VERSION >= 1) };
    }
}
