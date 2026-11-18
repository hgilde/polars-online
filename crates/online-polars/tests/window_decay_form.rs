//! A row's held mass takes the discount's own `exp2` from a hundredth of a
//! half-life up (docs/PLAN.md task 461), and `exp_m1` below it. The switch
//! itself is pinned where it is made, to the bit, by the unit test of
//! `windows/memo_tests.rs`. Here a stream whose gaps lie just below, just
//! above and on both sides of that edge is run through the operators and held
//! to an independent recursion written with `exp_m1` masses at every gap (the
//! form before): the two agree to a bound far inside the movement the change
//! is allowed, and the outputs are the same bits however the stream is
//! chunked and whichever thread resolves a group.
use online_polars::{Bank, PAR_MIN_ROWS, Spec, WindowsConfig, WindowsRun};
use polars::prelude::*;
use serde_json::json;

/// Gaps are whole units of `2^-24`, so a gap and a clock stamp are exact
/// doubles and the stream's gaps are the test's own, not a rounding's.
const UNIT: f64 = 1.0 / 16_777_216.0;
/// The edge, in half-lives, as the code states it.
const EDGE: f64 = 1e-2;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The last gap, in units, that is below the edge at half-life `h`.
fn last_unit_below(h: f64) -> u64 {
    let mut u = (EDGE * h / UNIT) as u64;
    while (u as f64 * UNIT) / h >= EDGE {
        u -= 1;
    }
    while ((u + 1) as f64 * UNIT) / h < EDGE {
        u += 1;
    }
    u
}

#[derive(Clone, Copy, Debug)]
enum Side {
    Below,
    Above,
    Both,
}

/// A stream of `n` rows whose gaps are the `spread` nearest units on `side`
/// of the edge, in no pattern a memo could hold (a repeated gap is answered
/// from it, and the edge is not what is then tested).
fn stream(n: usize, h: f64, side: Side, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let below = last_unit_below(h);
    let spread = 40.0;
    let mut r = Lcg(seed);
    let (mut t, mut ts, mut x) = (1.0e3, vec![], vec![]);
    for i in 0..n {
        let off = (r.next() * spread) as u64;
        let units = match side {
            Side::Below => below - off,
            Side::Above => below + 1 + off,
            Side::Both => {
                if i % 2 == 0 {
                    below - off
                } else {
                    below + 1 + off
                }
            }
        };
        let gap = units as f64 * UNIT;
        let x_of = gap / h;
        match side {
            Side::Below => assert!(x_of < EDGE),
            Side::Above => assert!(x_of >= EDGE),
            Side::Both => {}
        }
        t += gap;
        ts.push(t);
        x.push(10.0 * r.next() - 3.0);
    }
    (ts, x)
}

fn frame(ts: &[f64], x: &[f64]) -> DataFrame {
    df!("t" => ts, "x" => x).unwrap()
}

fn config(h: f64) -> WindowsConfig {
    serde_json::from_value(json!({
        "formulas": [
            {"name": "mean", "tree": ["ewm_mean", ["col", "x"], {"half_life": h}]},
            {"name": "std", "tree": ["ewm_std", ["col", "x"], {"half_life": h, "bias": true}]},
        ],
        "clock": "t",
        "gap_cap": 1.0e9,
    }))
    .unwrap()
}

/// The mean and the standard deviation columns, the stream fed in chunks of
/// `chunk` rows.
fn run(ts: &[f64], x: &[f64], h: f64, chunk: usize) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let f = frame(ts, x);
    let mut w = WindowsRun::new(config(h), f.schema()).unwrap();
    let mut out: Option<DataFrame> = None;
    let mut at = 0;
    while at < f.height() {
        let len = chunk.min(f.height() - at);
        let o = w.feed(&f.slice(at as i64, len), None).unwrap();
        match &mut out {
            None => out = Some(o),
            Some(a) => {
                a.vstack_mut(&o).unwrap();
            }
        }
        at += len;
    }
    let tail = w.finish().unwrap();
    let mut out = out.unwrap();
    out.vstack_mut(&tail).unwrap();
    let col = |n: &str| out.column(n).unwrap().f64().unwrap().iter().collect();
    (col("mean"), col("std"))
}

/// The time-weighted mean by its definition: a row's value held over the gap
/// before it, weighed by that gap's decayed time `c (1 - 2^-x)`, taken by
/// `exp_m1` at every gap (the first row's held over all the time before it,
/// `c`); the sums decayed by `2^-x` a row.
fn reference(ts: &[f64], x: &[f64], h: f64) -> Vec<f64> {
    let c = h / std::f64::consts::LN_2;
    let (mut w, mut s) = (0.0, 0.0);
    let mut mean = vec![];
    for i in 0..ts.len() {
        let (lambda, m) = if i == 0 {
            (0.0, c)
        } else {
            let d = ts[i] - ts[i - 1];
            (
                (-(d / h)).exp2(),
                -c * (-(d * std::f64::consts::LN_2 / h)).exp_m1(),
            )
        };
        w = w * lambda + m;
        s = s * lambda + x[i] * m;
        mean.push(s / w);
    }
    mean
}

/// At the edge on both sides and between, in several half-lives: the
/// operators against the old form's recursion, and every chunking one set of
/// bits.
#[test]
fn a_stream_either_side_of_the_edge_is_the_old_form_to_rounding_and_chunk_invariant() {
    // Below the edge the held mass is the old form's, so a mean is the
    // reference's to the bit; above it the mass comes from the discount's
    // `exp2` and a mean moves by rounding, the mass's `2^-52 / (1 - d)` at
    // the edge being 1.6e-14. Held at 3e-14: the largest of these streams
    // moves 3.4e-15, and a form wrong by a digit would show.
    const BOUND: f64 = 3e-14;
    let n = 4_000;
    for h in [1.0, 4.0, 0.25] {
        for side in [Side::Below, Side::Above, Side::Both] {
            let (ts, x) = stream(n, h, side, 7);
            let want = reference(&ts, &x, h);
            let (mean, std) = run(&ts, &x, h, n);
            let worst = (0..n)
                .map(|i| (mean[i].unwrap() - want[i]).abs() / want[i].abs().max(1.0))
                .fold(0.0f64, f64::max);
            assert!(worst <= BOUND, "h {h}, {side:?}: mean moved {worst:e}");
            match side {
                Side::Below => assert_eq!(worst, 0.0, "h {h}: below the edge nothing moves"),
                Side::Above => {
                    assert!(worst > 0.0, "h {h}: above the edge the form is the new one")
                }
                Side::Both => {}
            }
            // One chunk and many chunks are the same bits.
            for chunk in [1, 37, 1_000] {
                let (m2, s2) = run(&ts, &x, h, chunk);
                let bits =
                    |v: &[Option<f64>]| v.iter().map(|o| o.map(f64::to_bits)).collect::<Vec<_>>();
                assert_eq!(bits(&m2), bits(&mean), "h {h}, {side:?}, chunk {chunk}");
                assert_eq!(bits(&s2), bits(&std), "h {h}, {side:?}, chunk {chunk}");
            }
        }
    }
}

/// `n` rows of `groups` groups interleaved on one clock whose mean gap, per
/// group, is a twentieth of the formulas' half-life of 5: a hundredth of a
/// half-life, so a group's gaps fall on both sides of the edge.
fn at_the_edge(n: usize, groups: usize, seed: u64) -> DataFrame {
    let mut r = Lcg(seed);
    let (mut t, mut g, mut x, mut mid) = (vec![], vec![], vec![], vec![]);
    let (mut clock, mut level) = (0.0, 100.0);
    for i in 0..n {
        clock += 0.05 / groups as f64 * 2.0 * r.next();
        t.push(clock);
        g.push(format!("g{}", (r.next() * groups as f64) as usize));
        x.push(if i % 13 == 4 {
            None
        } else {
            Some(r.next() - 0.5)
        });
        level += r.next() - 0.5;
        mid.push(level);
    }
    df!("t" => t, "g" => g, "x" => x, "mid" => mid).unwrap()
}

fn spec() -> Spec {
    // The target reads the future by a window and the past by a decay with a
    // half-life of 5, so a backward `ewm_mean` is resolved natively, a group
    // at a time on the pool's threads.
    let tree = r#"["-", ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}], ["ewm_mean", ["col", "mid"], {"half_life": 5.0}]]"#;
    serde_json::from_str(&format!(
        r#"{{
            "name": "m",
            "model": {{"type": "ewridge", "ridge": 1e-6, "max_rows_between_solves": 1}},
            "targets": [{{"name": "fwd", "formula": {tree}}}],
            "features": ["x"],
            "clock": "t", "gap_cap": 100.0, "half_life": 50.0, "group": "g",
            "min_weight": 3.0, "emit_clocks": true, "embargo": 12.5,
            "fit_intercept": true
        }}"#
    ))
    .unwrap()
}

fn feed(df: &DataFrame, chunks: usize) -> (DataFrame, Vec<u8>) {
    let mut bank = Bank::new(vec![spec()]).unwrap();
    let n = df.height();
    let step = n.div_ceil(chunks);
    let mut acc: Option<DataFrame> = None;
    let mut i = 0;
    while i < n {
        let len = step.min(n - i);
        let cols = bank.fit_predict_from(&df.slice(i as i64, len), 0).unwrap();
        let st = cols[0].struct_().unwrap().clone().unnest();
        match &mut acc {
            None => acc = Some(st),
            Some(a) => {
                a.vstack_mut(&st).unwrap();
            }
        }
        i += len;
    }
    (acc.unwrap(), bank.save_bytes().unwrap())
}

fn column(df: &DataFrame, name: &str) -> Vec<Option<f64>> {
    df.column(name).unwrap().f64().unwrap().iter().collect()
}

/// A target with a backward decay over gaps on both sides of the edge,
/// resolved on the pool's threads (a chunk of several groups from
/// `PAR_MIN_ROWS` rows up, or of many groups) and on the calling thread: the
/// predictions, the rows learned and the state's bytes are the same on either
/// side, at every chunking.
#[test]
fn a_backward_decay_at_the_edge_resolves_to_the_same_bits_on_many_threads() {
    let n = 2 * PAR_MIN_ROWS + 1_500;
    for groups in [3, 12] {
        let df = at_the_edge(n, groups, 11 + groups as u64);
        // Gaps on both sides of the edge, group by group: the edge is crossed.
        let (mut last, mut below, mut above) = (std::collections::HashMap::new(), 0, 0);
        let (ts, gs) = (
            df.column("t").unwrap().f64().unwrap(),
            df.column("g").unwrap().str().unwrap(),
        );
        for (t, g) in ts.iter().zip(gs.iter()) {
            if let Some(p) = last.insert(g.unwrap().to_string(), t.unwrap()) {
                if (t.unwrap() - p) / 5.0 >= EDGE {
                    above += 1;
                } else {
                    below += 1;
                }
            }
        }
        assert!(
            below > n / 10 && above > n / 10,
            "{groups} groups: {below} below, {above} above"
        );
        let (want, want_bytes) = feed(&df, 1);
        let predicted = column(&want, "pred_fwd")
            .iter()
            .filter(|p| p.is_some())
            .count();
        assert!(
            predicted > n / 2,
            "{groups} groups: {predicted} rows predicted"
        );
        for chunks in [5, 2] {
            let (got, bytes) = feed(&df, chunks);
            assert_eq!(
                column(&got, "pred_fwd"),
                column(&want, "pred_fwd"),
                "{groups} groups, {chunks} chunks"
            );
            assert_eq!(
                column(&got, "learned_clock"),
                column(&want, "learned_clock"),
                "{groups} groups, {chunks} chunks"
            );
            assert!(
                bytes == want_bytes,
                "{groups} groups, {chunks} chunks: state"
            );
        }
    }
}
