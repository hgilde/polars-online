//! The window budget's pre-pass (docs/PLAN.md task 115 (d); the user,
//! 2026-09-29: "115.1 follow your suggestion"). A chunk that would take a
//! window's ring past a refusing budget is refused before any row is
//! learned, and the pre-pass's verdict is the ring's own: a bank run
//! without it refuses the same chunk with the same words, having learned
//! part of it.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// Two groups on clocks of their own, gaps of one to six units, a null
/// feature and a null target now and then, a session that changes every
/// `session_rows` rows of the frame, and weights in `[0.5, 1.5)`.
fn frame(n: usize, session_rows: usize) -> DataFrame {
    let mut s = 99u64;
    let mut lcg = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut g, mut t, mut sess) = (Vec::new(), Vec::new(), Vec::new());
    let (mut x0, mut x1, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut clocks = [0.0f64, 0.0];
    for i in 0..n {
        let k = i % 2;
        g.push(format!("g{k}"));
        clocks[k] += 1.0 + lcg().abs() * 5.0;
        t.push(clocks[k]);
        sess.push(format!("s{}", i / session_rows));
        let (a, b) = (lcg(), lcg());
        x0.push(if i % 17 == 5 { None } else { Some(a) });
        x1.push(Some(b));
        y.push(if i % 23 == 7 {
            None
        } else {
            Some(2.0 * a - b + 0.1 * lcg())
        });
        w.push(1.0 + 0.5 * lcg());
    }
    df!("g" => g, "t" => t, "s" => sess, "x0" => x0, "x1" => x1, "y" => y, "w" => w).unwrap()
}

fn spec(model: &str, extra: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{
            "name": "m",
            "model": {model},
            "features": ["x0", "x1"],
            "clock": "t",
            "gap_cap": 30.0,
            "half_life": 40.0,
            "weight": "w",
            "group": "g",
            "min_weight": 0.0
            {extra}
        }}"#
    ))
    .unwrap()
}

/// Feed `df` in pieces of `size` rows to a bank with the pre-pass and to
/// one without, holding them to the same output until one refuses; then
/// the other must refuse the same piece with the same words, and the bank
/// with the pre-pass must be as it was before that piece and go on. The
/// piece refused, if any.
fn prepass_against_the_ring(spec: &Spec, df: &DataFrame, size: usize) -> Option<usize> {
    let mut with = Bank::new(vec![spec.clone()]).unwrap();
    let mut without = Bank::new(vec![spec.clone()]).unwrap();
    without.set_window_prepass(false);
    let mut start = 0;
    let mut k = 0;
    while start < df.height() {
        let piece = df.slice(start as i64, size);
        let before = with.save_bytes().unwrap();
        match (with.fit_predict(&piece), without.fit_predict(&piece)) {
            (Ok(a), Ok(b)) => {
                for (ca, cb) in a.iter().zip(&b) {
                    assert!(
                        ca.as_materialized_series()
                            .equals_missing(cb.as_materialized_series()),
                        "piece {k}: the pre-pass changed an output"
                    );
                }
            }
            (Err(a), Err(b)) => {
                let (a, b) = (a.to_string(), b.to_string());
                assert_eq!(a, b, "piece {k}: the two refusals differ");
                assert!(a.contains("window_budget"), "{a}");
                assert_eq!(
                    with.save_bytes().unwrap(),
                    before,
                    "piece {k}: the refused piece taught the bank something"
                );
                assert!(
                    without.save_bytes().is_err(),
                    "the bank without the pre-pass learned part of the piece and stops"
                );
                // The bank goes on: it scores the piece it refused.
                with.predict(&piece).unwrap();
                return Some(k);
            }
            (a, b) => panic!(
                "piece {k}: with the pre-pass {:?}, without it {:?}",
                a.map(|_| ()),
                b.map(|_| ())
            ),
        }
        start += size;
        k += 1;
    }
    None
}

/// The replay reaches every path a ring is offered on: each windowed kind,
/// the residual ring beside the fit's, a snapshot cadence, and rows a label
/// delay holds back into a later piece. Where sessions reset the models, or
/// close the group and start it afresh, every eight rows of a group, no
/// ring ever fills: a replay that missed the reset or the close would
/// refuse a chunk the bank takes, which the comparison would catch.
#[test]
#[ignore = "extended: a second or more (1.4 s)"]
fn the_prepass_refuses_what_the_ring_would_and_leaves_the_bank_as_it_was() {
    let df = frame(600, 80);
    let short = frame(600, 16);
    let ridge = r#"{"type": "ewridge", "window_size": 200.0, "window_every": 1,
                    "window_budget": {"refuse": 0.004}}"#;
    // (label, spec, frame, whether the ring reaches the budget)
    let cases: Vec<(&str, Spec, &DataFrame, bool)> = vec![
        ("ewridge", spec(ridge, r#", "targets": ["y"]"#), &df, true),
        (
            "ewridge with sigma's ring",
            spec(ridge, r#", "targets": ["y"], "emit_sigma": true"#),
            &df,
            true,
        ),
        // A snapshot cadence on the clock, on the rows, and both
        // (docs/PLAN.md task 162): the shadow carries the spacing.
        (
            "ewridge every 3 clock units",
            spec(
                r#"{"type": "ewridge", "window_size": 600.0, "window_every": 3,
                    "window_budget": {"refuse": 0.004}}"#,
                r#", "targets": ["y"]"#,
            ),
            &df,
            true,
        ),
        (
            "ewridge every 3 rows",
            spec(
                r#"{"type": "ewridge", "window_size": 600.0, "max_rows_between_snapshots": 3,
                    "window_budget": {"refuse": 0.004}}"#,
                r#", "targets": ["y"]"#,
            ),
            &df,
            true,
        ),
        (
            "ewridge every 8 clock units or 2 rows",
            spec(
                r#"{"type": "ewridge", "window_size": 600.0, "window_every": 8,
                    "max_rows_between_snapshots": 2, "window_budget": {"refuse": 0.004}}"#,
                r#", "targets": ["y"]"#,
            ),
            &df,
            true,
        ),
        (
            "ewridge resetting every eight rows",
            spec(
                ridge,
                r#", "targets": ["y"], "session": "s", "session_gap": "reset""#,
            ),
            &short,
            false,
        ),
        (
            "ewridge closing every eight rows",
            spec(
                ridge,
                r#", "targets": ["y"], "session": "s", "group_close": "session""#,
            ),
            &short,
            false,
        ),
        (
            "ewridge with a label delay",
            spec(ridge, r#", "targets": ["y"], "embargo": 8.0"#),
            &df,
            true,
        ),
        (
            "ew_cov",
            spec(
                r#"{"type": "ew_cov", "stats": ["mean", "var", "cov"], "window_size": 200.0,
                    "window_budget": {"refuse": 0.003}}"#,
                r#", "targets": []"#,
            ),
            &df,
            true,
        ),
        (
            "marginal",
            spec(
                r#"{"type": "marginal", "window_size": 200.0,
                    "window_budget": {"refuse": 0.004}}"#,
                r#", "targets": ["y"]"#,
            ),
            &df,
            true,
        ),
    ];
    for (label, s, frame, reaches) in &cases {
        for size in [1, 7, 40, 600] {
            let at = prepass_against_the_ring(s, frame, size);
            eprintln!("{label}, pieces of {size}: refused at {at:?}");
            assert_eq!(at.is_some(), *reaches, "{label}, pieces of {size}");
        }
    }
}

/// Under `drift_action = "reset"` the replay cannot foresee the resets, so
/// it abstains, and the ring's own refusal stops the run as before: both
/// banks refuse the same piece, and both stop.
#[test]
fn under_drift_resets_the_ring_still_stops_the_run() {
    let df = frame(400, 80);
    let s = spec(
        r#"{"type": "ewridge", "window_size": 200.0, "window_every": 1,
            "window_budget": {"refuse": 0.004}}"#,
        r#", "targets": ["y"], "emit_drift": true, "drift_threshold": 20.0,
            "drift_action": "reset""#,
    );
    let mut bank = Bank::new(vec![s]).unwrap();
    let err = bank.fit_predict(&df).unwrap_err().to_string();
    assert!(err.contains("window_budget"), "{err}");
    let again = bank.save_bytes().unwrap_err();
    assert!(again.contains("cannot go on"), "{again}");
    // The JSON export refuses it too, in one place for every caller (review
    // round 4, PA12): it handed out the state `save_bytes` refuses, and the
    // Python wrapper encoded the whole state as msgpack first to refuse it.
    for pretty in [false, true] {
        let json = bank.save_json_string(pretty).unwrap_err();
        assert!(json.contains("cannot go on"), "{json}");
        assert!(json.contains("window_budget"), "{json}");
    }
}

/// The replay decides the window's edge on the rows' stamps, as the ring
/// does (docs/PLAN.md task 175), under either edge (task 196). Rows 1 ms
/// apart under a window of a second and a snapshot of 40 bytes every row.
/// Under `closed = "both"`, a refusing budget between 1001 and 1002
/// snapshots: the ring keeps the snapshot exactly a second old, holds 1001
/// from row 1000 on and crosses the budget at row 1001's snapshot. A replay
/// on the summed clock dropped that snapshot until row 1007 and would cross
/// only at row 1008, so it let the first piece of 1004 rows through, and
/// the ring refused it half learned. Under `"right"`, the default, that
/// snapshot has left at row 1000, so the ring holds 1001 only between a
/// row's snapshot and its trim: a budget between 1000 and 1001 snapshots is
/// crossed at row 1000's, and the replay refuses the same first piece.
#[test]
fn the_prepass_decides_the_windows_edge_on_the_stamps() {
    // 40,060 bytes, between 1001 snapshots of 40 and 1002; and 40,020,
    // between 1000 and 1001.
    for (closed, mib) in [
        ("both", "0.038204193115234375"),
        ("right", "0.038166046142578125"),
    ] {
        prepass_decides_the_windows_edge_on_the_stamps(closed, mib);
    }
}

fn prepass_decides_the_windows_edge_on_the_stamps(closed: &str, mib: &str) {
    let n = 1_100usize;
    let t = Series::new(
        "t".into(),
        (0..n as i64)
            .map(|i| 1_704_067_200_000_000_000 + i * 1_000_000)
            .collect::<Vec<i64>>(),
    )
    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
    .unwrap();
    let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 101) as f64 / 101.0).collect();
    let df = df!("t" => t, "x0" => x.clone(), "x1" => x).unwrap();
    let s: Spec = serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ew_cov", "stats": ["mean"], "window_size": "1s",
            "closed": "{closed}", "window_budget": {{"refuse": {mib}}}}},
            "features": ["x0"], "clock": "t", "gap_cap": "1d", "half_life": "1h",
            "min_weight": 0.0}}"#
    ))
    .unwrap();
    assert_eq!(
        prepass_against_the_ring(&s, &df, 1_004),
        Some(0),
        "{closed}"
    );
}
