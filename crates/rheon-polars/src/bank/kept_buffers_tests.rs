//! A run's output buffers kept from an earlier run are not filled with NaN
//! again where the run writes them in full (`ChunkOut::renew_run`): what a
//! row it does not process leaves in them is never read. A chunk fed to a
//! bank holding buffers a full chunk of finite values left must give what
//! the same chunk gives a bank holding none, to the bit, rows skipped for a
//! missing feature included.
use crate::stream::ChunkOut;
use crate::{Bank, Spec};
use polars::prelude::*;

/// Spec text for the cases: grouped or not, wide or narrow, one target or
/// several, a statistic model with no target, a model that withholds rows,
/// a spec whose rows are held back (every row of which the buffers are
/// filled for, as before).
const SPECS: &[&str] = &[
    r#"{"name": "a", "model": {"type": "ew_cov"}, "features": ["x0", "x1", "x2", "x3"],
        "clock": "t", "gap_cap": 100.0, "half_life": 30.0}"#,
    r#"{"name": "b", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1", "x2"],
        "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "min_weight": 8.0, "emit_sigma": true, "emit_zscore": true}"#,
    r#"{"name": "c", "model": {"type": "ewridge"}, "targets": ["y", "z"], "features": ["x0", "x1"],
        "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "group": "g", "emit_sigma": true, "conformal": 0.9}"#,
    r#"{"name": "d", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"],
        "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "embargo": 3.0, "emit_sigma": true}"#,
    r#"{"name": "e", "model": {"type": "seqtest"}, "targets": ["y"], "features": [],
        "clock": "t", "gap_cap": 100.0, "group": "g"}"#,
    r#"{"name": "f", "model": {"type": "ew_cov", "stats": ["mean", "cov", "corr"]},
        "features": ["x0", "x1", "x2"], "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "group": "g"}"#,
];

/// `n` rows from row `from`; `holes` makes every `holes`-th row's `x0`
/// missing (the bank skips it), 0 for none.
fn frame(from: usize, n: usize, holes: usize) -> DataFrame {
    let at = |i: usize| from + i;
    let x = |i: usize, c: usize| ((at(i) * (7 + 2 * c)) % 23) as f64 - 11.0 + 0.25 * c as f64;
    df!(
        "t" => (0..n).map(|i| at(i) as f64 * 1.5).collect::<Vec<_>>(),
        "g" => (0..n).map(|i| (at(i) * 5 % 3) as i64).collect::<Vec<_>>(),
        "x0" => (0..n).map(|i| {
            if holes > 0 && at(i) % holes == 1 { f64::NAN } else { x(i, 0) }
        }).collect::<Vec<_>>(),
        "x1" => (0..n).map(|i| x(i, 1)).collect::<Vec<_>>(),
        "x2" => (0..n).map(|i| x(i, 2)).collect::<Vec<_>>(),
        "x3" => (0..n).map(|i| x(i, 3)).collect::<Vec<_>>(),
        "y" => (0..n).map(|i| 0.5 * x(i, 0) - x(i, 1) + ((at(i) * 13) % 7) as f64).collect::<Vec<_>>(),
        "z" => (0..n).map(|i| x(i, 2) + ((at(i) * 3) % 5) as f64).collect::<Vec<_>>()
    )
    .unwrap()
}

fn bank_of(text: &str) -> Bank {
    let spec: Spec = serde_json::from_str(text).unwrap();
    let mut bank = Bank::new(vec![spec]).unwrap();
    bank.set_threads(Some(1)).unwrap();
    bank
}

#[test]
fn buffers_kept_from_a_run_of_finite_rows_change_nothing() {
    for text in SPECS {
        // The same chunks, to a bank that keeps its runs' buffers and to
        // one that keeps none: the second chunk of each skips rows.
        let (mut kept, mut fresh) = (bank_of(text), bank_of(text));
        let mut reused = false;
        for (from, n, holes) in [
            (0, 60, 0),
            (60, 60, 3),
            (120, 17, 2),
            (137, 60, 0),
            (197, 40, 4),
        ] {
            let df = frame(from, n, holes);
            reused |= !kept.spares.outs.is_empty();
            fresh.spares = super::Spares::default();
            let a = kept.fit_predict(&df).unwrap();
            let b = fresh.fit_predict(&df).unwrap();
            assert_eq!(a.len(), b.len());
            for (ca, cb) in a.iter().zip(&b) {
                assert!(
                    ca.as_materialized_series()
                        .equals_missing(cb.as_materialized_series()),
                    "{text}: rows {from}.. differ with buffers kept"
                );
            }
        }
        assert!(
            reused,
            "{text}: no run's buffers were kept, so nothing was tested"
        );
        assert_eq!(
            kept.save_bytes().unwrap(),
            fresh.save_bytes().unwrap(),
            "{text}: the state moved"
        );
    }
}

/// The buffers a run does not fill again are the ones it writes in full:
/// kept for a spec with nothing held back, filled for any other.
#[test]
fn only_a_spec_with_nothing_held_back_keeps_its_buffers() {
    let dense: Vec<bool> = SPECS
        .iter()
        .map(|text| ChunkOut::keeps_buffers(&serde_json::from_str(text).unwrap()))
        .collect();
    assert_eq!(dense, [true, true, true, false, true, true]);
}
