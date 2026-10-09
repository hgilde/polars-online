//! An `audit` spec through the bank (docs/PLAN.md task 223 (b)): a null and
//! a NaN counted apart through Arrow's validity, every row read whatever its
//! columns hold, a restart that keeps the counts, and the same frames from 1,
//! 7 and 600 chunks (hard rule 3). The statistics' own oracles are the core's
//! (`audit/tests.rs`) and scipy's and statsmodels' (`tests/test_audit.py`).

use online_polars::{AuditTable, Bank, Spec};
use polars::prelude::*;

fn spec(extra: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "a", "model": {{"type": "audit", "pairs": true}}, "targets": ["x"],
            "features": ["x", "z"], "clock": "t", "gap_cap": 5.0{extra}}}"#
    ))
    .unwrap()
}

/// 1,200 rows: `x` null on every 10th and NaN on every 25th (one row both
/// ways counts as null, the validity winning), `z` infinite on every 40th,
/// a duplicate stamp every 50th row and a gap every 300th.
fn frame() -> DataFrame {
    let n = 1200;
    let x: Vec<Option<f64>> = (0..n)
        .map(|i| {
            if i % 10 == 3 {
                None
            } else if i % 25 == 7 {
                Some(f64::NAN)
            } else {
                Some(((i * 7919) % 1000) as f64 / 100.0)
            }
        })
        .collect();
    let z: Vec<f64> = (0..n)
        .map(|i| {
            if i % 40 == 9 {
                f64::INFINITY
            } else {
                ((i * 104_729) % 977) as f64
            }
        })
        .collect();
    let mut t = 0.0;
    let ts: Vec<f64> = (0..n)
        .map(|i| {
            t += match i {
                0 => 0.0,
                _ if i % 300 == 0 => 20.0,
                _ if i % 50 == 0 => 0.0,
                _ => 1.0,
            };
            t
        })
        .collect();
    let s: Vec<i64> = (0..n).map(|i| i as i64 / 400).collect();
    df!("t" => ts, "x" => x, "z" => z, "s" => s).unwrap()
}

fn feed(bank: &mut Bank, df: &DataFrame, chunks: usize) {
    let size = df.height().div_ceil(chunks);
    let mut at = 0;
    while at < df.height() {
        bank.fit_predict(&df.slice(at as i64, size)).unwrap();
        at += size;
    }
}

fn read(bank: &Bank, table: AuditTable) -> DataFrame {
    bank.audit(0, None, table, false).unwrap()
}

fn count(f: &DataFrame, col: &str, row: usize) -> u64 {
    f.column(col).unwrap().u64().unwrap().get(row).unwrap()
}

#[test]
fn a_null_and_a_nan_are_counted_apart_and_every_row_is_read() {
    let df = frame();
    let mut bank = Bank::new(vec![spec("")]).unwrap();
    feed(&mut bank, &df, 1);
    let cols = read(&bank, AuditTable::Columns);
    let n = df.height() as u64;
    let nulls = (0..n).filter(|i| i % 10 == 3).count() as u64;
    let nans = (0..n).filter(|i| i % 10 != 3 && i % 25 == 7).count() as u64;
    let infs = (0..n).filter(|i| i % 40 == 9).count() as u64;
    assert_eq!(
        (
            count(&cols, "rows", 0),
            count(&cols, "null", 0),
            count(&cols, "nan", 0)
        ),
        (n, nulls, nans)
    );
    assert_eq!(count(&cols, "count", 0), n - nulls - nans);
    assert_eq!(
        (count(&cols, "pos_inf", 1), count(&cols, "null", 1)),
        (infs, 0)
    );
    // Every row was read: the summary processed rows a feature's null
    // would have skipped for any other model.
    let summary = bank.summary(0, None).unwrap();
    assert_eq!(count(&summary, "rows_processed", 0), n);
    let pairs = read(&bank, AuditTable::Pairs);
    let both = (0..n)
        .filter(|i| i % 10 != 3 && i % 25 != 7 && i % 40 != 9)
        .count() as u64;
    assert_eq!(count(&pairs, "count", 0), both);
    let clock = read(&bank, AuditTable::Clock);
    let gaps = (1..n).filter(|i| i % 300 == 0).count() as u64;
    let dups = (1..n).filter(|i| i % 300 != 0 && i % 50 == 0).count() as u64;
    assert_eq!(count(&clock, "steps", 0), n - 1);
    assert_eq!(
        (count(&clock, "gaps", 0), count(&clock, "duplicates", 0)),
        (gaps, dups)
    );
    // The largest step is the gap as it elapsed, 20, not the 5 `gap_cap`
    // ages a model by (review 6, C-5).
    let max_step = clock.column("max_step").unwrap().f64().unwrap().get(0);
    assert_eq!(max_step, Some(20.0));
}

/// A temporal clock's steps come back as `Duration`s in the column's unit,
/// the step as it elapsed (review 6, C-5): five minutes a row, and a gap of
/// two hours past a `gap_cap` of one.
#[test]
fn a_temporal_clocks_steps_are_durations_as_they_elapsed() {
    let minute = 60_000_000_000i64;
    let ts: Vec<i64> = (0..10)
        .map(|i| i * 5 * minute + if i >= 6 { 120 * minute } else { 0 })
        .collect();
    for unit in [
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        let t = Series::new("t".into(), ts.clone())
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap()
            .cast(&DataType::Datetime(unit, None))
            .unwrap();
        let df = DataFrame::new(
            10,
            vec![
                t.into(),
                Column::new("x".into(), (0..10).map(f64::from).collect::<Vec<_>>()),
                Column::new("z".into(), vec![1.0; 10]),
            ],
        )
        .unwrap();
        let spec: Spec = serde_json::from_str(
            r#"{"name": "a", "model": {"type": "audit"}, "targets": ["x"],
                "features": ["x", "z"], "clock": "t", "gap_cap": "1h"}"#,
        )
        .unwrap();
        let mut bank = Bank::new(vec![spec]).unwrap();
        bank.fit_predict(&df).unwrap();
        let clock = read(&bank, AuditTable::Clock);
        let per_unit = match unit {
            TimeUnit::Milliseconds => 1_000_000,
            TimeUnit::Microseconds => 1_000,
            TimeUnit::Nanoseconds => 1,
        };
        for (col, want) in [("max_step", 125 * minute), ("step_mean", 5 * minute)] {
            let c = clock.column(col).unwrap();
            assert_eq!(c.dtype(), &DataType::Duration(unit), "{col}");
            let got = c.cast(&DataType::Int64).unwrap().i64().unwrap().get(0);
            assert_eq!(got, Some(want / per_unit), "{col} in {unit:?}");
        }
        assert_eq!(count(&clock, "gaps", 0), 1);
    }
}

#[test]
fn the_frames_are_the_same_from_one_chunk_or_many() {
    let df = frame();
    let mut one = Bank::new(vec![spec("")]).unwrap();
    feed(&mut one, &df, 1);
    for chunks in [7, 600] {
        let mut many = Bank::new(vec![spec("")]).unwrap();
        feed(&mut many, &df, chunks);
        for table in [AuditTable::Columns, AuditTable::Pairs, AuditTable::Clock] {
            assert!(
                read(&many, table).equals_missing(&read(&one, table)),
                "{chunks} chunks, {table:?}"
            );
        }
    }
}

/// A restart -- a session that resets -- starts the run and the clock over
/// and keeps every count: the audit reads 1,200 rows, not the last session's.
#[test]
fn a_restart_keeps_the_counts() {
    let df = frame();
    let mut bank = Bank::new(vec![spec(r#", "session": "s", "session_gap": "reset""#)]).unwrap();
    feed(&mut bank, &df, 3);
    let cols = read(&bank, AuditTable::Columns);
    assert_eq!(count(&cols, "rows", 0), df.height() as u64);
    let summary = bank.summary(0, None).unwrap();
    assert_eq!(count(&summary, "resets", 0), 2);
    // Two restarts: two steps fewer than rows less one.
    let clock = read(&bank, AuditTable::Clock);
    assert_eq!(count(&clock, "steps", 0), df.height() as u64 - 3);
}

#[test]
fn a_spec_that_is_not_an_audit_and_pairs_not_kept_are_refused() {
    let ridge: Spec = serde_json::from_str(
        r#"{"name": "r", "model": {"type": "ewridge"}, "targets": ["x"], "features": ["z"],
            "half_life": 10.0}"#,
    )
    .unwrap();
    let no_pairs: Spec = serde_json::from_str(
        r#"{"name": "a", "model": {"type": "audit"}, "targets": ["x"], "features": ["x", "z"]}"#,
    )
    .unwrap();
    let bank = Bank::new(vec![ridge, no_pairs]).unwrap();
    let e = bank.audit(0, None, AuditTable::Columns, false).unwrap_err();
    assert!(e.contains("not \"audit\""), "{e}");
    let e = bank.audit(1, None, AuditTable::Pairs, false).unwrap_err();
    assert!(e.contains("pairs=True"), "{e}");
    assert!(
        bank.audit(1, None, AuditTable::Clock, false)
            .unwrap()
            .height()
            == 0
    );
    assert!(AuditTable::parse("rows").is_err());
}
