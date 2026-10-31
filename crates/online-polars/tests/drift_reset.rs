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
//!
//! Under an `embargo` the detection fires as a held row is released, and
//! the first held row behind it read its step from the row that fired; and
//! a skipped row right after the reset left the next row at a step of 0
//! where a fresh bank steps it from the skipped row (review round 10, S7-1
//! and S7-3). Those cases are here too, with `learned_clock` on a row that
//! releases the row that fired and the one after it (S7-2).

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
    spec_with(model, half_life, "")
}

/// [`spec`] with more keys, each written `, "key": value`.
fn spec_with(model: &str, half_life: &str, more: &str) -> Spec {
    let text = format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": ["x0", "x1"],
            "clock": "t", "gap_cap": 10.0, "half_life": {half_life}, "min_weight": 10.0,
            "coef_every": 0, "emit_drift": true, "drift_threshold": 20.0,
            "drift_action": "reset", "emit_sigma": true, "emit_clocks": true{more}}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// The clock column.
fn clock(df: &DataFrame) -> Vec<f64> {
    df.column("t")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

/// The rows an `embargo` of `e` releases at row `t`: those whose wait ran
/// out at `t` and not before.
fn released_at(df: &DataFrame, t: usize, e: f64) -> Vec<usize> {
    let c = clock(df);
    (0..t)
        .filter(|&j| c[t] - c[j] >= e && (t == 0 || c[t - 1] - c[j] < e))
        .collect()
}

/// `df` with feature `x0` null on row `at`: a row the stream skips.
fn skip_row(df: &DataFrame, at: usize) -> DataFrame {
    let x0: Vec<Option<f64>> = df
        .column("x0")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, v)| if i == at { None } else { v })
        .collect();
    let mut out = df.clone();
    out.with_column(Series::new("x0".into(), x0).into())
        .unwrap();
    out
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
    assert_fields_same(a, from, b, what, |_| true);
}

/// [`assert_tail_same`] over the fields `keep` names.
fn assert_fields_same(
    a: &DataFrame,
    from: usize,
    b: &DataFrame,
    what: &str,
    keep: impl Fn(&str) -> bool,
) {
    let (fa, fb) = (fields(a), fields(b));
    assert_eq!(fa.len(), fb.len());
    for (sa, sb) in fa.iter().zip(&fb) {
        if !keep(sb.name()) {
            continue;
        }
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

/// The fields that hold what the model has learned, and the clocks: under an
/// `embargo` the rows held when the detector fired were scored by the model
/// before the reset, and their records feed the rebuilt instance's residual
/// diagnostics as they are released (the score a row was shown is the one
/// its residual reads), where a fresh bank scored them itself -- so `sigma`
/// and the detector differ, by design. `settled_frac` adds the held rows'
/// clock, a running sum whose last bit keeps the rows before the reset.
fn learned(name: &str) -> bool {
    [
        "pred",
        "coef",
        "weight_sum",
        "withheld_reason",
        "scored_clock",
        "learned_clock",
    ]
    .iter()
    .any(|f| name.starts_with(f))
}

/// The first output row a detector fired on under an `embargo` of `e`, and
/// the row it fired on: of the rows released there, the last unless the
/// row shows that one learned by the rebuilt model, then the one before.
fn embargo_fire(out: &DataFrame, df: &DataFrame, e: f64) -> (usize, usize) {
    let t = first_fire(out);
    let released = released_at(df, t, e);
    let last = *released
        .last()
        .expect("a row released where the detection fired");
    match learned_clock(out, t) {
        AnyValue::Null => (t, last),
        shown => {
            assert_eq!(shown, AnyValue::Float64(clock(df)[last]));
            assert_eq!(released.len(), 2, "rows released at {t}: {released:?}");
            (t, released[0])
        }
    }
}

fn learned_clock(out: &DataFrame, row: usize) -> AnyValue<'static> {
    fields(out)
        .into_iter()
        .find(|f| f.name() == "learned_clock")
        .expect("emit_clocks")
        .get(row)
        .unwrap()
        .into_static()
}

/// Under an `embargo` the detection fires as a held row is released, and
/// the rows held behind it teach the rebuilt model: the first of them must
/// read the step a fresh bank's first row reads, 0, not its step from the
/// row that fired, and the rows scored before it is released read the held
/// clock without that step. Unmarked, `rls` at `delta = 100` decayed its
/// prior by it and predicted up to 1.1e-2 away from a fresh bank fed the
/// rows after the one that fired (review round 10, S7-1). From the row
/// that fired on, the model's own fields equal the fresh bank's to the bit,
/// through one chunk and a chunk that ends right after the detection.
///
/// On a clock of whole steps the held rows' clock sums exactly, and
/// `settled_frac` is held to the bit too: the rows scored between the
/// detection and the first held row's release read the held clock without
/// that row's step from the row that fired, as the fresh bank's do.
#[test]
fn under_an_embargo_a_reset_is_a_fresh_bank_from_the_next_row() {
    let mut whole = frame();
    let steps: Vec<f64> = (0..N).map(|i| i as f64).collect();
    whole
        .with_column(Series::new("t".into(), steps).into())
        .unwrap();
    for (df, exact) in [(frame(), false), (whole, true)] {
        under_an_embargo(&df, exact);
    }
}

fn under_an_embargo(df: &DataFrame, exact: bool) {
    let df = df.clone();
    let keep = |name: &str| learned(name) || (exact && name.starts_with("settled_frac"));
    for (model, half_life) in [
        (r#"{"type": "rls", "delta": 100.0}"#, "200.0"),
        (
            r#"{"type": "kalman", "q": [0.01, 0.01, 0.01], "obs_var": 0.01}"#,
            "200.0",
        ),
        (r#"{"type": "rls"}"#, "[100.0, 200.0]"),
    ] {
        let s = spec_with(model, half_life, r#", "embargo": 3.0"#);
        let what = format!("{model} at {half_life} under an embargo, whole steps {exact}");
        let one = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[]);
        let (t, r1) = embargo_fire(&one, &df, 3.0);
        let fresh = feed(
            &mut Bank::new(vec![s.clone()]).unwrap(),
            &df.slice(r1 as i64 + 1, N - r1 - 1),
            &[],
        );
        // The fresh bank's rows from `t` on: its first rows are still held.
        let skip = t - r1 - 1;
        assert!(skip > 0, "a row held behind the one that fired");
        let fresh_tail = fresh.slice(skip as i64, fresh.height() - skip);
        assert_fields_same(&one, t, &fresh_tail, &what, keep);
        let cut = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[t + 1]);
        assert_tail_same(&cut, 0, &one, &format!("{what}, cut after {t}"));
        // Scored against the bank as it stands after the detection, with
        // rows still held: the rows after it read what the fresh bank fed
        // the rows after the one that fired reads.
        let mut a = Bank::new(vec![s.clone()]).unwrap();
        feed(&mut a, &df.slice(0, t + 1), &[]);
        let mut b = Bank::new(vec![s.clone()]).unwrap();
        feed(&mut b, &df.slice(r1 as i64 + 1, t - r1), &[]);
        let next = df.slice(t as i64 + 1, 5);
        let scored = |bank: &Bank| DataFrame::new(5, bank.predict(&next).unwrap()).unwrap();
        assert_fields_same(
            &scored(&a),
            0,
            &scored(&b),
            &format!("{what}, scored"),
            keep,
        );
    }
}

/// A detection whose release empties the buffer: a jump of 5 on a clock of
/// whole steps releases every held row with the one that fired, and the
/// row that released them is the first the rebuilt model will learn. It is
/// scored as a fresh bank's first row, over no step and no held clock, and
/// every model field from it on, `settled_frac` included, is the fresh
/// bank's to the bit, and so is a row scored against the bank after it.
#[test]
fn a_reset_that_empties_the_buffer_is_a_fresh_bank_from_the_next_row() {
    let s = spec_with(
        r#"{"type": "rls", "delta": 100.0}"#,
        "200.0",
        r#", "embargo": 3.0"#,
    );
    let mut df = frame();
    let steps: Vec<f64> = (0..N).map(|i| i as f64).collect();
    df.with_column(Series::new("t".into(), steps.clone()).into())
        .unwrap();
    let probe = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[]);
    let (_, r1) = embargo_fire(&probe, &df, 3.0);
    let mut c = steps;
    c.iter_mut().skip(r1 + 1).for_each(|v| *v += 5.0);
    let mut jumped = df.clone();
    jumped
        .with_column(Series::new("t".into(), c).into())
        .unwrap();
    let at = r1 + 1;
    assert_eq!(released_at(&jumped, at, 3.0).last(), Some(&r1));
    let one = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &jumped, &[]);
    assert_eq!(embargo_fire(&one, &jumped, 3.0), (at, r1));
    let fresh = feed(
        &mut Bank::new(vec![s.clone()]).unwrap(),
        &jumped.slice(at as i64, N - at),
        &[],
    );
    let keep = |name: &str| learned(name) || name.starts_with("settled_frac");
    assert_fields_same(&one, at, &fresh, "an emptied buffer", keep);
    // Scored right after: the one row held is the first the fresh bank saw.
    let mut a = Bank::new(vec![s.clone()]).unwrap();
    feed(&mut a, &jumped.slice(0, at + 1), &[]);
    let mut b = Bank::new(vec![s.clone()]).unwrap();
    feed(&mut b, &jumped.slice(at as i64, 1), &[]);
    let next = jumped.slice(at as i64 + 1, 3);
    let scored = |bank: &Bank| DataFrame::new(3, bank.predict(&next).unwrap()).unwrap();
    assert_fields_same(
        &scored(&a),
        0,
        &scored(&b),
        "scored on an emptied buffer",
        keep,
    );
}

/// A skipped row right after the one that fired: a fresh bank fed the rows
/// after it reads no step on its skipped first row and steps its first
/// accepted row from it, so the restarted stream must too, where it read a
/// step of 0 (review round 10, S7-3). Without an embargo every field from
/// the skipped row on equals the fresh bank's, through one chunk, a chunk
/// that ends on the skipped row, and a save and a load there; under one,
/// the model's own fields do, from the row that fired.
#[test]
fn a_skipped_row_after_a_reset_starts_the_fresh_clock() {
    let df = frame();
    let model = r#"{"type": "rls", "delta": 100.0}"#;
    let s = spec(model, "200.0");
    let t = first_fire(&feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[]));
    let gapped = skip_row(&df, t + 1);
    let one = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &gapped, &[]);
    assert_eq!(first_fire(&one), t, "the skipped row is after the fire");
    let fresh = feed(
        &mut Bank::new(vec![s.clone()]).unwrap(),
        &gapped.slice(t as i64 + 1, N - t - 1),
        &[],
    );
    assert_tail_same(&one, t + 1, &fresh, "a skipped row after the reset");
    // The row after the skipped one steps from it: a nonzero step, as the
    // fresh bank's clock reads it, which the old mark read as 0.
    let c = clock(&df);
    assert!(c[t + 2] - c[t + 1] > 0.4, "the step under test is not 0");
    // A chunk ends on the skipped row, and the bank is saved and loaded
    // there: the fresh clock crosses both.
    let mut a = Bank::new(vec![s.clone()]).unwrap();
    let head = feed(&mut a, &gapped.slice(0, t + 2), &[]);
    let bytes = a.save_bytes().unwrap();
    let mut b = Bank::load_bytes(&bytes, Some(std::slice::from_ref(&s))).unwrap();
    let tail = feed(&mut b, &gapped.slice(t as i64 + 2, N - t - 2), &[]);
    let mut both = head;
    both.vstack_mut(&tail).unwrap();
    assert_tail_same(&both, 0, &one, "saved on the skipped row");

    // Under an embargo the row skipped is the one after the released row
    // that fired; the first held row after it carries its fresh step.
    let s = spec_with(model, "200.0", r#", "embargo": 3.0"#);
    let probe = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[]);
    let (t, r1) = embargo_fire(&probe, &df, 3.0);
    let gapped = skip_row(&df, r1 + 1);
    let one = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &gapped, &[]);
    assert_eq!(embargo_fire(&one, &gapped, 3.0), (t, r1));
    let fresh = feed(
        &mut Bank::new(vec![s.clone()]).unwrap(),
        &gapped.slice(r1 as i64 + 1, N - r1 - 1),
        &[],
    );
    let skip = t - r1 - 1;
    let fresh_tail = fresh.slice(skip as i64, fresh.height() - skip);
    assert_fields_same(&one, t, &fresh_tail, "under an embargo", learned);
    for cuts in [vec![r1 + 2], vec![t + 1]] {
        let cut = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &gapped, &cuts);
        assert_tail_same(&cut, 0, &one, &format!("under an embargo, cut at {cuts:?}"));
    }
}

/// A clock jump releases the row that fired and the row after it at one
/// output row: the rebuilt model learned the second, so the row shows its
/// clock in `learned_clock`, as the fresh bank fed the rows after the one
/// that fired shows it; it was null, the flag of the output row read for
/// every row released there (review round 10, S7-2).
#[test]
fn a_row_released_after_the_reset_shows_in_learned_clock() {
    let df = frame();
    let s = spec_with(r#"{"type": "rls"}"#, "200.0", r#", "embargo": 3.0"#);
    let probe = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[]);
    let (_, r1) = embargo_fire(&probe, &df, 3.0);
    // Every clock from row r1 + 2 on moves 5 later: that row releases r1
    // and r1 + 1 together.
    let mut c = clock(&df);
    c.iter_mut().skip(r1 + 2).for_each(|v| *v += 5.0);
    let mut jumped = df.clone();
    jumped
        .with_column(Series::new("t".into(), c.clone()).into())
        .unwrap();
    let at = r1 + 2;
    let released = released_at(&jumped, at, 3.0);
    assert!(
        released.ends_with(&[r1, r1 + 1]),
        "released at {at}: {released:?}"
    );
    let one = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &jumped, &[]);
    assert_eq!(
        first_fire(&one),
        at,
        "the detection is flagged where r1 is released"
    );
    assert_eq!(learned_clock(&one, at), AnyValue::Float64(c[r1 + 1]));
    let fresh = feed(
        &mut Bank::new(vec![s.clone()]).unwrap(),
        &jumped.slice(r1 as i64 + 1, N - r1 - 1),
        &[],
    );
    assert_fields_same(
        &one,
        at,
        &fresh.slice(1, fresh.height() - 1),
        "the jump",
        learned,
    );
}

/// A clock reset on a skipped row restarts the clock's own fold there, so
/// the next accepted row steps from it, as a fresh stream's skipped first
/// row has it step: the restart leaves nothing for its mark to take out.
/// The mark took the step out anyway, and that row read a step of 0
/// (review round 10, S7-3). Every field from the skipped row on equals a
/// fresh bank's fed the rows from it, with and without an embargo.
#[test]
fn a_clock_reset_on_a_skipped_row_steps_the_next_row_from_it() {
    let at = 400;
    let mut c = clock(&frame());
    c.iter_mut().skip(at).for_each(|v| *v -= 50.0);
    let mut df = skip_row(&frame(), at);
    df.with_column(Series::new("t".into(), c.clone()).into())
        .unwrap();
    assert!(c[at + 1] - c[at] > 0.4, "the step under test is not 0");
    // Without the drift reset, whose fresh clock would hide it.
    for more in ["", r#", "embargo": 3.0"#] {
        let text = format!(
            r#"{{"name": "m", "model": {{"type": "rls", "delta": 100.0}}, "targets": ["y"],
                "features": ["x0", "x1"], "clock": "t", "gap_cap": 10.0, "half_life": 200.0,
                "coef_every": 0, "emit_clocks": true, "restart_after_step_back": 0.0{more}}}"#
        );
        let s: Spec = serde_json::from_str(&text).unwrap();
        let one = feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, &[]);
        let fresh = feed(
            &mut Bank::new(vec![s.clone()]).unwrap(),
            &df.slice(at as i64, N - at),
            &[],
        );
        assert_tail_same(
            &one,
            at,
            &fresh,
            &format!("a clock reset on a skipped row{more}"),
        );
    }
}
