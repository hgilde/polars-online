//! `drift_action = "reset"` starts the model over (docs/PLAN.md task 278,
//! batch 3): from the row after the one that fired, a stream equals a fresh
//! bank fed those rows, every field to the bit.
//!
//! A fresh bank's first row has a clock step of 0. The row after a reset
//! read the ordinary step, so the rebuilt model decayed by it: `rls`'s prior
//! `delta I` (predictions up to 0.017 from a fresh bank's), `kalman`'s
//! covariance under a fixed process noise, and every kind's `settled_frac`;
//! and `learned_clock` showed the clock of a row the rebuilt model never
//! learned from. The Python sweep (`tests/test_drift_reset.py`) covers the
//! ten linear kinds and every diagnostic; this holds the cases that failed,
//! where `cargo mutants` sees them, through one chunk, a chunk boundary on
//! the row that fired, and a half-life grid's coupled restart.

use online_polars::{Bank, Spec};
use polars::prelude::*;

const N: usize = 1200;

/// A seeded generator in `[-1, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Two features on an irregular clock; at the middle row both slopes flip
/// and the level moves by 6 against noise of about 0.1.
fn frame() -> DataFrame {
    let mut r = lcg(7);
    let (mut t, mut x0, mut x1, mut y) = (vec![], vec![], vec![], vec![]);
    let mut clock = 0.0;
    for i in 0..N {
        clock += 1.0 + 0.5 * r();
        let (a, b) = (r(), r());
        let sign = if i < N / 2 { 1.0 } else { -1.0 };
        let level = if i < N / 2 { 0.3 } else { 6.3 };
        t.push(clock);
        x0.push(a);
        x1.push(b);
        y.push(level + sign * (2.0 * a - 1.5 * b) + 0.1 * r());
    }
    df!("t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
}

fn spec(model: &str, half_life: &str) -> Spec {
    let text = format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": ["x0", "x1"],
            "clock": "t", "gap_cap": 10.0, "half_life": {half_life}, "min_weight": 10.0,
            "coef_every": 0, "emit_drift": true, "drift_threshold": 20.0,
            "drift_action": "reset", "emit_sigma": true, "emit_clocks": true}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// `df` through `bank`, cut at each row of `cuts`, as the output columns.
fn feed(bank: &mut Bank, df: &DataFrame, cuts: &[usize]) -> DataFrame {
    let mut edges = vec![0];
    edges.extend_from_slice(cuts);
    edges.push(df.height());
    let mut out: Option<DataFrame> = None;
    for w in edges.windows(2) {
        let piece = df.slice(w[0] as i64, w[1] - w[0]);
        let got = DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap()).unwrap();
        match out.as_mut() {
            None => out = Some(got),
            Some(o) => {
                o.vstack_mut(&got).unwrap();
            }
        }
    }
    out.unwrap()
}

fn fields(out: &DataFrame) -> Vec<Series> {
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series()
}

/// The first row any instance's detector fired on: the evidence the reset
/// path ran.
fn first_fire(out: &DataFrame) -> usize {
    let drift: Vec<Series> = fields(out)
        .into_iter()
        .filter(|s| s.name().starts_with("drift_"))
        .collect();
    let t = (0..out.height())
        .find(|&i| drift.iter().any(|s| s.bool().unwrap().get(i) == Some(true)))
        .expect("the detector never fired: the test would compare two identical runs");
    assert!((N / 2..N - 100).contains(&t), "a fire at {t}");
    t
}

/// Every field of `a` from row `from` on, against `b`, value for value: a
/// float by its bits, anything else by its `AnyValue`.
fn assert_tail_same(a: &DataFrame, from: usize, b: &DataFrame, what: &str) {
    let (fa, fb) = (fields(a), fields(b));
    assert_eq!(fa.len(), fb.len());
    for (sa, sb) in fa.iter().zip(&fb) {
        let sa = sa.slice(from as i64, sb.len());
        assert_eq!(sa.len(), sb.len(), "{what}: {}", sb.name());
        if let (Ok(xa), Ok(xb)) = (sa.f64(), sb.f64()) {
            for (i, (p, q)) in xa.iter().zip(xb.iter()).enumerate() {
                assert_eq!(
                    p.map(f64::to_bits),
                    q.map(f64::to_bits),
                    "{what}: {} at row {}: {p:?} vs {q:?}",
                    sb.name(),
                    from + i
                );
            }
        } else {
            for i in 0..sb.len() {
                assert_eq!(
                    sa.get(i).unwrap(),
                    sb.get(i).unwrap(),
                    "{what}: {} at row {}",
                    sb.name(),
                    from + i
                );
            }
        }
    }
}

#[test]
fn a_reset_is_a_fresh_bank_from_the_next_row() {
    let df = frame();
    for (model, half_life) in [
        (r#"{"type": "rls"}"#, "200.0"),
        (r#"{"type": "rls", "delta": 100.0}"#, "200.0"),
        (
            r#"{"type": "kalman", "q": [0.01, 0.01, 0.01], "obs_var": 0.01}"#,
            "200.0",
        ),
        (r#"{"type": "ewridge"}"#, "200.0"),
        // A grid: every instance restarts after all have read the row.
        (r#"{"type": "rls"}"#, "[100.0, 200.0]"),
    ] {
        let s = spec(model, half_life);
        let what = format!("{model} at {half_life}");
        let one = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[]);
        let t = first_fire(&one);
        let fresh = feed(
            &mut Bank::new(vec![s.clone()]).unwrap(),
            &df.slice(t as i64 + 1, N - t - 1),
            &[],
        );
        assert_tail_same(&one, t + 1, &fresh, &what);
        // A chunk that ends on the row that fired: the restart's mark
        // crosses the boundary with the rebuilt model.
        let cut = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[t + 1]);
        assert_tail_same(&cut, 0, &one, &format!("{what}, cut after {t}"));
    }
}
