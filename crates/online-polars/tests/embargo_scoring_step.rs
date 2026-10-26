//! Under an `embargo`, the step a held row is scored over (docs/PLAN.md
//! task 278, D11).
//!
//! A row held under an embargo is scored from a model that has learned only
//! the rows whose delay has passed. Its score therefore reads the clock from
//! the last of those rows to the row itself: the held rows' steps and its
//! own. It was read over the row's own step alone, as if the row sat one
//! step after the last learned row: `holt` forecast one step ahead where the
//! row was several, `kalman` added one step's drift to the row's variance.
//!
//! Every oracle here is the definition, with nothing of the embargo in it: a
//! bank fed only the rows matured when the row is scored, asked to `predict`
//! the row. The clock steps are whole numbers, so the held rows' summed
//! steps and the oracle's one subtraction are the same doubles, and the
//! relation holds to the bit.

use online_polars::{Bank, Spec};
use polars::prelude::*;

const DELAY: f64 = 15.0;
const ROWS: usize = 160;

/// A seeded generator in `[0, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Whole clock steps of 1 to 5 from 0, so a row waits for 3 to 15 others; a
/// feature `x`, and a target `y` on a trend in the clock with noise.
fn frame() -> (DataFrame, Vec<f64>) {
    let mut r = lcg(11);
    let mut t = Vec::with_capacity(ROWS);
    let mut now = 0.0;
    for i in 0..ROWS {
        if i > 0 {
            now += (1.0 + (r() * 5.0).floor()).min(5.0);
        }
        t.push(now);
    }
    let x: Vec<f64> = (0..ROWS).map(|_| r() * 2.0 - 1.0).collect();
    let y: Vec<f64> = t
        .iter()
        .zip(&x)
        .map(|(t, x)| 0.5 * t + 2.0 * x + r() - 0.5)
        .collect();
    let df = df!("t" => t.clone(), "x" => x, "y" => y).unwrap();
    (df, t)
}

fn spec_of(model: &str, features: &str, rest: &str) -> Spec {
    let text = format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": {features},
            "clock": "t", "min_weight": 0.0, "half_life": 20.0{rest}}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn holt(rest: &str) -> Spec {
    spec_of(r#"{"type": "holt"}"#, "[]", rest)
}

fn kalman(rest: &str) -> Spec {
    spec_of(
        r#"{"type": "kalman", "q": [0.01, 0.01], "obs_var": 0.25}"#,
        r#"["x"]"#,
        &format!(r#", "emit_error_inflation": true{rest}"#),
    )
}

fn delayed(s: &Spec) -> Spec {
    let mut text = serde_json::to_value(s).unwrap();
    text["embargo"] = serde_json::json!(DELAY);
    serde_json::from_value(text).unwrap()
}

fn out_frame(df: &DataFrame, cols: Vec<Column>) -> DataFrame {
    DataFrame::new(df.height(), cols).unwrap()
}

fn fields(out: &DataFrame) -> Vec<Series> {
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series()
}

/// Written after the row's update, or counting the held rows' clock: not
/// what a bank's `predict` of the row reads.
fn compared(name: &str) -> bool {
    !(name.contains("coef") || name == "settled_frac")
}

/// The fields of `a`'s row `i` and `b`'s row `j` that differ, to the bit; a
/// NaN equals a NaN, a null a null.
fn differing(a: &DataFrame, i: usize, b: &DataFrame, j: usize) -> Vec<String> {
    let (fa, fb) = (fields(a), fields(b));
    let mut bad = Vec::new();
    for (sa, sb) in fa.iter().zip(&fb) {
        assert_eq!(sa.name(), sb.name());
        if !compared(sa.name()) {
            continue;
        }
        let (va, vb) = (sa.slice(i as i64, 1), sb.slice(j as i64, 1));
        let same = match (va.f64(), vb.f64()) {
            (Ok(x), Ok(y)) => {
                x.get(0).map(f64::to_bits) == y.get(0).map(f64::to_bits)
                    || x.get(0).is_some_and(f64::is_nan) && y.get(0).is_some_and(f64::is_nan)
            }
            _ => va.equals_missing(&vb),
        };
        if !same {
            bad.push(sa.name().to_string());
        }
    }
    bad
}

/// How many rows have matured when row `i` is scored: those at least
/// `DELAY` behind it on the clock.
fn matured(t: &[f64], i: usize) -> usize {
    t.iter().take_while(|&&v| v <= t[i] - DELAY).count()
}

/// Each row scored by a bank fed only the rows matured by then, walking
/// forward; `None` where none has. With `own_step`, at the clock one of the
/// row's own steps after the last matured row instead: what the defect read.
fn as_matured(s: &Spec, df: &DataFrame, t: &[f64], own_step: bool) -> Vec<Option<DataFrame>> {
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    let mut fed = 0;
    (0..df.height())
        .map(|i| {
            let k = matured(t, i);
            if k > fed {
                bank.fit_predict(&df.slice(fed as i64, k - fed)).unwrap();
                fed = k;
            }
            if k == 0 {
                return None;
            }
            let mut row = df.slice(i as i64, 1);
            if own_step {
                let at = t[k - 1] + (t[i] - t[i - 1]);
                row.with_column(Column::new("t".into(), [at])).unwrap();
            }
            Some(out_frame(&row, bank.predict(&row).unwrap()))
        })
        .collect()
}

/// The fields on which the delayed run parts from `oracle`, over the rows
/// the oracle scored, and the number of those rows.
fn parted(held: &DataFrame, oracle: &[Option<DataFrame>]) -> (Vec<String>, usize) {
    let mut bad: Vec<String> = Vec::new();
    let mut n = 0;
    for (i, o) in oracle.iter().enumerate() {
        if let Some(o) = o {
            n += 1;
            for c in differing(held, i, o, 0) {
                if !bad.contains(&c) {
                    bad.push(c);
                }
            }
        }
    }
    (bad, n)
}

fn fit_predict(s: &Spec, df: &DataFrame) -> DataFrame {
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    out_frame(df, bank.fit_predict(df).unwrap())
}

/// The relation, and its activation: scored over each row's own step, the
/// fields `moves` name part from the delayed run, and only those.
fn check(s: &Spec, moves: &[&str]) {
    let (df, t) = frame();
    let held = fit_predict(&delayed(s), &df);
    let (bad, n) = parted(&held, &as_matured(s, &df, &t, false));
    assert!(n > ROWS * 4 / 5, "{n}");
    assert!(bad.is_empty(), "parts from the matured bank on {bad:?}");
    let (mut moved, _) = parted(&held, &as_matured(s, &df, &t, true));
    moved.sort();
    let mut want: Vec<String> = moves.iter().map(|m| m.to_string()).collect();
    want.sort();
    assert_eq!(moved, want);
}

#[test]
fn a_held_holt_row_forecasts_from_the_last_learned_row() {
    check(&holt(r#", "gap_cap": 1e9"#), &["pred_y", "resid_y"]);
}

#[test]
fn a_held_kalman_row_reads_the_drift_since_the_last_learned_row() {
    // The noise gate on, at a limit that withholds rows: the gate reads the
    // step too, and a withheld row is a null the oracle must match.
    let s = kalman(r#", "gap_cap": 1e9, "max_error_inflation": 4.5"#);
    let (df, _) = frame();
    let held = fit_predict(&delayed(&s), &df);
    let withheld = fields(&held)
        .into_iter()
        .find(|f| f.name() == "pred_y")
        .unwrap()
        .null_count();
    assert!(withheld > 0 && withheld < ROWS, "{withheld}");
    check(
        &s,
        &["error_inflation_y", "pred_y", "resid_y", "withheld_reason"],
    );
}

/// A gap past `gap_cap` is a gap the cap holds whether one row spans it or
/// several held rows do: the held rows' capped steps summed and capped are
/// the cap of the raw sum.
#[test]
fn the_step_since_the_last_learned_row_is_capped_as_one_gap() {
    check(&holt(r#", "gap_cap": 3.0"#), &["pred_y", "resid_y"]);
    check(&kalman(r#", "gap_cap": 3.0"#), &["error_inflation_y"]);
}

/// `predict` against a bank holding rows scores as `fit_predict` would: from
/// the last learned row, the held rows' clock counted.
#[test]
fn predict_against_a_bank_holding_rows_is_the_matured_banks() {
    let (df, t) = frame();
    for s in [holt(r#", "gap_cap": 1e9"#), kalman(r#", "gap_cap": 3.0"#)] {
        for m in [40, 97, 130] {
            let mut bank = Bank::new(vec![delayed(&s)]).unwrap();
            bank.fit_predict(&df.slice(0, m)).unwrap();
            let k = matured(&t, m - 1);
            assert!(m - k >= 3, "{m}: {k}");
            let mut plain = Bank::new(vec![s.clone()]).unwrap();
            plain.fit_predict(&df.slice(0, k)).unwrap();
            for i in m..m + 10 {
                let row = df.slice(i as i64, 1);
                let a = out_frame(&row, bank.predict(&row).unwrap());
                let b = out_frame(&row, plain.predict(&row).unwrap());
                let bad = differing(&a, 0, &b, 0);
                assert!(bad.is_empty(), "row {i} after {m}: {bad:?}");
            }
        }
    }
}

/// The step reads the held rows' clock, which the stream keeps across a
/// chunk boundary and a save: chunked and resumed, every field to the bit.
#[test]
fn the_held_clock_crosses_chunks_and_a_save() {
    let (df, _) = frame();
    for s in [holt(r#", "gap_cap": 1e9"#), kalman(r#", "gap_cap": 3.0"#)] {
        let s = delayed(&s);
        let whole = fit_predict(&s, &df);
        for size in [1usize, 7, 50] {
            let mut bank = Bank::new(vec![s.clone()]).unwrap();
            let mut start = 0;
            while start < ROWS {
                let piece = df.slice(start as i64, size);
                let got = out_frame(&piece, bank.fit_predict(&piece).unwrap());
                for r in 0..piece.height() {
                    let bad = differing(&got, r, &whole, start + r);
                    assert!(bad.is_empty(), "size {size}, row {}: {bad:?}", start + r);
                }
                start += size;
                // Saved and loaded at every boundary of the coarsest cut.
                if size == 50 {
                    let bytes = bank.save_bytes().unwrap();
                    bank = Bank::load_bytes(&bytes, None).unwrap();
                }
            }
        }
    }
}
