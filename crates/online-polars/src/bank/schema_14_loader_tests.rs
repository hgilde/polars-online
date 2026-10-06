//! A schema-14 stream's loader rebuilds the held-rows clock from the rows
//! still held. A file of its own, keeping bank.rs under
//! `tests/test_repo_hygiene.py`'s 250 KB cap for a source file.
use crate::stream::Stream;
use crate::{Bank, Spec};
use polars::prelude::*;

/// Schema 15 keeps each instance's held-rows clock; a schema-14 file has
/// none, and its loader rebuilds it as 14 did at every chunk boundary:
/// the sum over the rows still held (`StreamState::pending_clock`).
#[test]
fn a_schema_14_stream_rebuilds_the_held_rows_clock_from_the_held_rows() {
    let spec: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "rls"}, "targets": ["y"],
                "features": ["x0"], "clock": "t", "half_life": 0.2,
                "gap_cap": 0.4, "embargo": 0.3}"#,
    )
    .unwrap();
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    let df = df!(
        "t" => [0.1, 0.2, 0.4, 0.5, 0.8],
        "x0" => [0.0; 5],
        "y" => [0.0; 5]
    )
    .unwrap();
    bank.fit_predict(&df).unwrap();
    let saved = bank.states[0].values().next().unwrap().save();
    assert!(
        !saved.pending.is_empty(),
        "the stream should end with rows held"
    );
    let held: f64 = saved.pending.iter().map(|p| p.d_clock).sum();

    let mut old = saved.clone();
    old.pending_clock.clear();
    let rebuilt = Stream::restore(&spec, &old).unwrap().save().pending_clock;
    assert!(
        !rebuilt.is_empty() && rebuilt.iter().all(|&c| c == held),
        "{rebuilt:?}"
    );

    let kept = Stream::restore(&spec, &saved).unwrap().save().pending_clock;
    assert_eq!(kept, saved.pending_clock);
}
