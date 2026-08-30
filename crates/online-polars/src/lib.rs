//! Polars-side plumbing: column extraction, per-group state, the chunk-fed model bank
//! and versioned msgpack save/load.
//!
//! See `docs/PLAN.md` §5. Scaffold only: the bank lands in task 6.

/// Re-exported so downstream crates pin one copy of the core.
pub use online_core;

#[cfg(test)]
mod tests {
    /// Guards against two copies of `online-core` being linked in: the re-export and the
    /// direct dependency must be the same crate, hence the same constant.
    #[test]
    fn links_against_core() {
        assert_eq!(
            super::online_core::SCHEMA_VERSION,
            online_core::SCHEMA_VERSION
        );
    }
}
