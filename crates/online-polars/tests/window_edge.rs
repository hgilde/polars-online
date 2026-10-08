//! A model window's edge, decided on the decayed clock held exactly
//! (docs/PLAN.md task 175; review CB1).
//!
//! A model window (`window_size` on `ewridge`, `lasso`, `ew_cov`, `ew_class`
//! and `marginal`) keyed its snapshots by a clock the model summed itself
//! from the doubles it was stepped with. A thousand steps of 1 ms sum to
//! 1.0000000000000007 s, so a row exactly one window old was dropped on rows
//! 1001 to 1007 of a 1 ms stream, and kept from row 1008 on. The edge is now
//! decided from each row's stamp: the decayed clock held exactly, integer
//! nanoseconds on a temporal clock and the raw value beside the time the caps
//! removed on a number clock, as the window operators decide theirs.
//!
//! Every oracle here is the definition, computed in the test from the raw
//! integer nanoseconds or the raw numbers, never from `ClockState`: at each
//! row the window holds the learned rows whose decayed age at the last
//! learned row is less than `window_size` under `closed = "right"`, the
//! default, or at most it under `"both"` (docs/PLAN.md task 196, N17). A
//! row's age adds up each step after `gap_cap` and `session_gap` (README, *A
//! hard window*). Every test runs both edges.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// The two edges a model window takes, as a spec writes them.
const EDGES: [&str; 2] = ["right", "both"];

/// Whether a row `age` old is inside a window `w` under `closed`: an age
/// less than `w` under `"right"`, at most `w` under `"both"`.
fn inside_by<T: PartialOrd>(closed: &str, age: T, w: T) -> bool {
    if closed == "right" { age < w } else { age <= w }
}

/// Whether the newest snapshot, `age` old, has left a window `w` under
/// `closed`: the complement of [`inside_by`].
fn left_by<T: PartialOrd>(closed: &str, age: T, w: T) -> bool {
    !inside_by(closed, age, w)
}

/// The five windowed models, each over the one feature `x`: the model's
/// JSON with `{window}` where its window settings go, and its targets.
const MODELS: [(&str, &str, &str); 5] = [
    (
        "ewridge",
        r#"{"type": "ewridge", "ridge": 1e-6, {window}}"#,
        r#""targets": ["y"],"#,
    ),
    (
        "lasso",
        r#"{"type": "lasso", "lasso_path": [0.1, 0.01], {window}}"#,
        r#""targets": ["y"],"#,
    ),
    (
        "ew_cov",
        r#"{"type": "ew_cov", "stats": ["mean"], {window}}"#,
        "",
    ),
    (
        "ew_class",
        r#"{"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0, {window}}"#,
        r#""targets": ["label"],"#,
    ),
    (
        "marginal",
        r#"{"type": "marginal", {window}}"#,
        r#""targets": ["y"],"#,
    ),
];

/// A spec of model `kind` with the window settings `window` (JSON members of
/// the model object) and the spec-level settings `rest`, on clock `t`, no
/// decay unless `rest` names one, and no weight floor.
fn spec(kind: &str, window: &str, rest: &str) -> Spec {
    let (_, model, targets) = MODELS
        .iter()
        .find(|(k, _, _)| *k == kind)
        .expect("a windowed model");
    let model = model.replace("{window}", window);
    let text = format!(
        r#"{{"name": "m", "model": {model}, {targets} "features": ["x"], "clock": "t",
            "min_weight": 0.0, {rest}}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

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

/// Rows at the clock values `t` (a column of any dtype), a feature `x`
/// (null where `skip` says), a target `y`, a label `label` and a session
/// `s`.
fn frame(t: Series, skip: &[bool], sessions: &[&str], seed: u64) -> DataFrame {
    let n = t.len();
    let mut r = lcg(seed);
    let x: Vec<Option<f64>> = (0..n)
        .map(|i| {
            let v = r();
            (!skip[i]).then_some(v)
        })
        .collect();
    let y: Vec<f64> = (0..n).map(|_| 0.5 + r()).collect();
    let label: Vec<&str> = (0..n)
        .map(|i| if (i * 7) % 3 == 0 { "a" } else { "b" })
        .collect();
    df!("t" => t, "x" => x, "y" => y, "label" => label, "s" => sessions.to_vec()).unwrap()
}

/// `df` fed to `spec`'s bank in chunks of `size` rows (all at once for 0).
fn run(spec: &Spec, df: &DataFrame, size: usize) -> DataFrame {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    let size = if size == 0 { df.height() } else { size };
    let mut out: Option<DataFrame> = None;
    let mut start = 0;
    while start < df.height() {
        let piece = df.slice(start as i64, size);
        let got = DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap()).unwrap();
        match out.as_mut() {
            None => out = Some(got),
            Some(o) => {
                o.vstack_mut(&got).unwrap();
            }
        }
        start += size;
    }
    out.unwrap()
}

/// One field of the spec's struct, as numbers (`None` where null).
fn field(out: &DataFrame, name: &str) -> Vec<Option<f64>> {
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect()
}

/// Per row, how many learned rows the window holds when the row is scored:
/// the rows `j` since the last restart, learned before the row, whose
/// decayed clock is at most `window` behind the last learned row's. `None`
/// for a row that is not learned (`decayed[i]` is `None`) and for one with
/// nothing learned before it since the restart.
fn counts_inside<T: Copy>(
    decayed: &[Option<T>],
    restarts: &[bool],
    inside: impl Fn(T, T) -> bool,
) -> Vec<Option<usize>> {
    let mut out = Vec::with_capacity(decayed.len());
    let mut learned: Vec<T> = Vec::new();
    for (i, d) in decayed.iter().enumerate() {
        if restarts[i] {
            learned.clear();
        }
        out.push(match (d, learned.last()) {
            (Some(_), Some(&last)) => Some(learned.iter().filter(|&&j| inside(last, j)).count()),
            _ => None,
        });
        if let Some(d) = d {
            learned.push(*d);
        }
    }
    out
}

/// The emitted `weight_sum` against the oracle's counts, no decay and unit
/// weights making the window's weight its count of rows: the rows where
/// they differ, with both.
fn mismatches(got: &[Option<f64>], want: &[Option<usize>]) -> Vec<(usize, Option<f64>, usize)> {
    got.iter()
        .zip(want)
        .enumerate()
        .filter_map(|(i, (g, w))| match w {
            Some(w) if *g != Some(*w as f64) => Some((i, *g, *w)),
            _ => None,
        })
        .collect()
}

/// [`mismatches`], recorded under `what` for one assertion over every case.
fn check(failures: &mut Vec<String>, what: &str, got: &[Option<f64>], want: &[Option<usize>]) {
    let bad = mismatches(got, want);
    if !bad.is_empty() {
        failures.push(format!(
            "{what}: {} rows hold another count than the window's, first {:?}",
            bad.len(),
            &bad[..bad.len().min(6)]
        ));
    }
}

fn assert_no_failures(failures: &[String]) {
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Nanoseconds since the Unix epoch of 2024-01-01T00:00:00.
const T0_NS: i64 = 1_704_067_200_000_000_000;

/// `n` rows 1 ms apart from 2024-01-01, as a `Datetime` column in `unit`,
/// and the same instants in integer nanoseconds.
fn millisecond_rows(n: usize, unit: TimeUnit) -> (Series, Vec<i64>) {
    let ns: Vec<i64> = (0..n as i64).map(|i| T0_NS + i * 1_000_000).collect();
    let per = match unit {
        TimeUnit::Milliseconds => 1_000_000,
        TimeUnit::Microseconds => 1_000,
        TimeUnit::Nanoseconds => 1,
    };
    let raw: Vec<i64> = ns.iter().map(|v| v / per).collect();
    let t = Series::new("t".into(), raw)
        .cast(&DataType::Datetime(unit, None))
        .unwrap();
    (t, ns)
}

/// The reviewer's case (`$S2/core-b/edge_accumulated_clock.py`), for every
/// windowed model, every `Datetime` unit and both edges: rows 1 ms apart and
/// a window of `"1s"`, with no decay, so `weight_sum` counts the rows
/// inside. Under `closed = "both"` each row's window holds exactly the rows
/// whose age at the last learned row is at most a second, a row exactly a
/// second old included: 1001 rows from row 1001 on. Under `"right"`, the
/// default, that row has left: 1000, as Polars' `rolling_sum_by("1s")`
/// counts them. The summed clock dropped that row on rows 1001 to 1007.
#[test]
fn a_row_exactly_one_window_old_is_decided_by_the_edge_for_every_model_and_unit() {
    let n = 1_400;
    let mut failures = Vec::new();
    for (closed, unit) in EDGES.into_iter().flat_map(|c| {
        [
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ]
        .map(|u| (c, u))
    }) {
        let (t, ns) = millisecond_rows(n, unit);
        let df = frame(t, &vec![false; n], &vec!["s"; n], 7);
        let decayed: Vec<Option<i64>> = ns.iter().map(|&v| Some(v - ns[0])).collect();
        let want = counts_inside(&decayed, &vec![false; n], |last, j| {
            inside_by(closed, last - j, 1_000_000_000)
        });
        let edge = if closed == "right" { 1_000 } else { 1_001 };
        assert_eq!(want[1_001], Some(edge), "the oracle's own edge");
        for (kind, _, _) in MODELS {
            let s = spec(
                kind,
                &format!(r#""window_size": "1s", "closed": "{closed}""#),
                r#""gap_cap": "1d", "half_life": "inf""#,
            );
            let got = field(&run(&s, &df, 0), "weight_sum");
            check(
                &mut failures,
                &format!("{kind} on {unit:?}, {closed}"),
                &got,
                &want,
            );
        }
    }
    assert_no_failures(&failures);
}

/// The default edge is `"right"`: a spec that names none counts as
/// `closed = "right"` does, row for row.
#[test]
fn the_default_edge_is_right() {
    let n = 1_100;
    let (t, _) = millisecond_rows(n, TimeUnit::Nanoseconds);
    let df = frame(t, &vec![false; n], &vec!["s"; n], 7);
    let rest = r#""gap_cap": "1d", "half_life": "inf""#;
    for (kind, _, _) in MODELS {
        let plain = run(&spec(kind, r#""window_size": "1s""#, rest), &df, 0);
        let right = run(
            &spec(kind, r#""window_size": "1s", "closed": "right""#, rest),
            &df,
            0,
        );
        assert_same_floats(&plain, &right, kind);
        assert_eq!(field(&plain, "weight_sum")[1_001], Some(1_000.0), "{kind}");
    }
}

/// The reviewer's script itself, with decay: an `ew_cov`'s windowed mean
/// and `weight_sum` at a half-life of an hour, against the weighted mean of
/// the rows inside -- less than a second old, or at most a second under
/// `closed = "both"` -- each weighted `0.5^(age / 1h)` with its age taken
/// from the integer nanoseconds, to 1e-9.
#[test]
fn a_decayed_windowed_mean_is_the_mean_of_the_rows_inside() {
    for closed in EDGES {
        decayed_windowed_mean_is_the_mean_of_the_rows_inside(closed);
    }
}

fn decayed_windowed_mean_is_the_mean_of_the_rows_inside(closed: &str) {
    let n = 2_600;
    let (t, ns) = millisecond_rows(n, TimeUnit::Nanoseconds);
    let df = frame(t, &vec![false; n], &vec!["s"; n], 0);
    let x: Vec<f64> = df
        .column("x")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let s = spec(
        "ew_cov",
        &format!(r#""window_size": "1s", "closed": "{closed}""#),
        r#""gap_cap": "1d", "half_life": "1h""#,
    );
    let out = run(&s, &df, 0);
    let (mean, wsum) = (field(&out, "mean_x"), field(&out, "weight_sum"));
    let mut bad = Vec::new();
    for i in 1..n {
        let now = ns[i - 1];
        let (mut sw, mut swx) = (0.0, 0.0);
        for j in (0..i).filter(|&j| inside_by(closed, now - ns[j], 1_000_000_000)) {
            let age = (now - ns[j]) as f64 / 1e9;
            let w = 0.5f64.powf(age / 3600.0);
            sw += w;
            swx += w * x[j];
        }
        // `ew_cov` reports no mean on its first rows, whatever the window.
        let (Some(m), Some(w)) = (mean[i], wsum[i]) else {
            assert!(i < 3, "row {i} reports no mean");
            continue;
        };
        if (m - swx / sw).abs() > 1e-9 || (w - sw).abs() > 1e-9 * sw {
            bad.push((i, m, swx / sw, w, sw));
        }
    }
    assert!(
        bad.is_empty(),
        "{closed}: {} rows differ from the rows inside, first {:?}",
        bad.len(),
        &bad[..bad.len().min(4)]
    );
}

/// `n` rows on a number clock at `i / 10`, which no sum of steps of a tenth
/// reproduces: `0.8 − 0.5` is `0.30000000000000004`.
fn tenths(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / 10.0).collect()
}

/// A number clock whose steps do not add exactly, a tenth apart under a
/// window of `0.3`: a row's window holds the rows whose raw value is less
/// than `0.3` below the last learned row's (at most `0.3` under `closed =
/// "both"`) by one subtraction, as the window operators decide, and as
/// Polars' `rolling_*_by` does. The summed clock kept the row at `0.5` in
/// the window of the row at `0.8`, on 186 of 400 rows.
#[test]
fn a_number_clocks_edge_is_one_subtraction_of_the_raw_values() {
    let n = 400;
    let t = tenths(n);
    let df = frame(
        Series::new("t".into(), t.clone()),
        &vec![false; n],
        &vec!["s"; n],
        3,
    );
    let decayed: Vec<Option<f64>> = t.iter().map(|&v| Some(v)).collect();
    let mut failures = Vec::new();
    for closed in EDGES {
        let want = counts_inside(&decayed, &vec![false; n], |last, j| {
            inside_by(closed, last - j, 0.3)
        });
        assert_eq!(want[9], Some(3), "0.8 − 0.5 is outside");
        for (kind, _, _) in MODELS {
            let s = spec(
                kind,
                &format!(r#""window_size": 0.3, "closed": "{closed}""#),
                r#""gap_cap": 1.0, "half_life": "inf""#,
            );
            let got = field(&run(&s, &df, 0), "weight_sum");
            check(&mut failures, &format!("{kind}, {closed}"), &got, &want);
        }
    }
    assert_no_failures(&failures);
}

/// One stream through every clock event the decayed clock has, on a
/// `Datetime` clock, and the decayed clock of each learned row from the
/// definition in integer nanoseconds: rows 100 ms apart, which no sum of
/// doubles adds exactly; a gap of an hour, capped at 10 s; a session change,
/// whose step is `session_gap`'s 2.5 s; a run of skipped rows (a null
/// feature), each step capped and their total folded into the next learned
/// row under the cap; and a step back of two hours, past
/// `restart_after_step_back`, where the model starts over.
struct Events {
    df: DataFrame,
    /// Per row, its decayed clock in nanoseconds when it is learned.
    decayed: Vec<Option<i128>>,
    /// The rows the model starts over at.
    restarts: Vec<bool>,
}

const CAP_NS: i128 = 10_000_000_000;
const SESSION_GAP_NS: i128 = 2_500_000_000;
const RESTART_NS: i128 = 60_000_000_000;

fn events() -> Events {
    // The raw instants, in ns from T0: each row's step from the last.
    let mut steps: Vec<i64> = Vec::new();
    let mut sessions: Vec<&str> = Vec::new();
    let mut skip: Vec<bool> = Vec::new();
    let mut push = |step: i64, session: &'static str, skipped: bool| {
        steps.push(step);
        sessions.push(session);
        skip.push(skipped);
    };
    let ms = 1_000_000i64;
    push(0, "s1", false);
    for i in 1..700 {
        // Now and then a row is skipped, its 100 ms folded into the next.
        push(100 * ms, "s1", i % 37 == 5);
    }
    // An hour's gap: 10 s to the models.
    push(3_600_000 * ms, "s1", false);
    for _ in 0..450 {
        push(100 * ms, "s1", false);
    }
    // Four skipped rows 4 s apart: no step is past the cap, but the 16 s
    // they fold into the next learned row is, so it gets the cap.
    for _ in 0..4 {
        push(4_000 * ms, "s1", true);
    }
    for _ in 0..400 {
        push(100 * ms, "s1", false);
    }
    // A session change 300 ms on: 2.5 s to the models.
    push(300 * ms, "s2", false);
    for _ in 0..450 {
        push(100 * ms, "s2", false);
    }
    // Two hours back: past `restart_after_step_back`, a restart.
    push(-7_200_000 * ms, "s2", false);
    for _ in 0..450 {
        push(100 * ms, "s2", false);
    }
    let n = steps.len();
    let mut ns = Vec::with_capacity(n);
    let mut at = T0_NS;
    for &s in &steps {
        at += s;
        ns.push(at);
    }

    // The decayed clock, from the definition.
    let mut decayed = vec![None; n];
    let mut restarts = vec![false; n];
    let (mut clock, mut pending) = (0i128, 0i128);
    for i in 0..n {
        let step = if i == 0 {
            0
        } else {
            let raw = i128::from(ns[i]) - i128::from(ns[i - 1]);
            if sessions[i] != sessions[i - 1] {
                SESSION_GAP_NS.clamp(0, CAP_NS)
            } else if raw < 0 {
                assert!(-raw > RESTART_NS, "the stream steps back only to restart");
                restarts[i] = true;
                clock = 0;
                pending = 0;
                0
            } else {
                raw.min(CAP_NS)
            }
        };
        if skip[i] {
            pending += step;
        } else {
            clock += (pending + step).min(CAP_NS);
            pending = 0;
            decayed[i] = Some(clock);
        }
    }
    let t = Series::new("t".into(), ns)
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    Events {
        df: frame(t, &skip, &sessions, 11),
        decayed,
        restarts,
    }
}

/// The settings `events()` is run under.
const EVENT_CLOCK: &str = r#""gap_cap": "10s", "half_life": "inf", "session": "s",
    "session_gap": "2500ms", "restart_after_step_back": "1m""#;

/// Through every clock event: a window of 30 s holds what the decayed clock
/// says, for every windowed model -- across the capped gap a row's age adds
/// up the cap, across the session change `session_gap`, across the skipped
/// rows their folded and capped total, and after the restart only the rows
/// since it.
#[test]
fn across_every_clock_event_the_window_holds_what_the_decayed_clock_says() {
    for closed in EDGES {
        across_every_clock_event(closed);
    }
}

fn across_every_clock_event(closed: &str) {
    let e = events();
    let want = counts_inside(&e.decayed, &e.restarts, |last, j| {
        inside_by(closed, last - j, 30_000_000_000)
    });
    // The oracle has edges to decide: rows exactly 30 s old, a stretch
    // straddling the capped gap, and the restart.
    let edges = e
        .decayed
        .iter()
        .flatten()
        .filter(|&&d| e.decayed.iter().flatten().any(|&k| k - d == 30_000_000_000))
        .count();
    assert!(
        edges > 1_000,
        "{edges} rows sit exactly a window before another"
    );
    // And every event is in the stream: the steps between learned rows
    // include the cap twice (the hour's gap, the folded run), the session's
    // gap once, and the stream restarts once.
    let learned: Vec<i128> = e.decayed.iter().flatten().copied().collect();
    let steps: Vec<i128> = learned.windows(2).map(|w| w[1] - w[0]).collect();
    assert_eq!(steps.iter().filter(|&&s| s == CAP_NS).count(), 2);
    assert_eq!(steps.iter().filter(|&&s| s == SESSION_GAP_NS).count(), 1);
    assert_eq!(e.restarts.iter().filter(|&&r| r).count(), 1);
    assert!(e.decayed.iter().any(Option::is_none), "skipped rows");
    let mut failures = Vec::new();
    let window = format!(r#""window_size": "30s", "closed": "{closed}""#);
    for (kind, _, _) in MODELS {
        let s = spec(kind, &window, EVENT_CLOCK);
        let got = field(&run(&s, &e.df, 0), "weight_sum");
        check(&mut failures, &format!("{kind}, {closed}"), &got, &want);
    }
    assert_no_failures(&failures);
}

/// Every float field of the spec's struct, row by row, bit for bit.
fn assert_same_floats(a: &DataFrame, b: &DataFrame, what: &str) {
    let (sa, sb) = (
        a.column("m").unwrap().struct_().unwrap().fields_as_series(),
        b.column("m").unwrap().struct_().unwrap().fields_as_series(),
    );
    for (fa, fb) in sa.iter().zip(&sb) {
        let (Ok(xa), Ok(xb)) = (fa.f64(), fb.f64()) else {
            continue;
        };
        for (i, (p, q)) in xa.iter().zip(xb.iter()).enumerate() {
            assert_eq!(
                p.map(f64::to_bits),
                q.map(f64::to_bits),
                "{what}: {} at row {i}: {p:?} vs {q:?}",
                fa.name()
            );
        }
    }
}

/// Chunk invariance across every clock event (hard rule 3): chunks of 1, 7
/// and 37 rows give the whole run's bits, and each holds the decayed
/// clock's window.
#[test]
#[ignore = "extended: a second or more (2.5 s); bank.rs::chunk_invariance keeps hard rule 3"]
fn chunking_moves_no_edge_across_every_clock_event() {
    let e = events();
    let mut failures = Vec::new();
    for closed in EDGES {
        let want = counts_inside(&e.decayed, &e.restarts, |last, j| {
            inside_by(closed, last - j, 30_000_000_000)
        });
        let window = format!(r#""window_size": "30s", "closed": "{closed}""#);
        for kind in ["ewridge", "ew_cov", "marginal"] {
            let s = spec(kind, &window, EVENT_CLOCK);
            let whole = run(&s, &e.df, 0);
            let what = format!("{kind}, {closed}");
            check(&mut failures, &what, &field(&whole, "weight_sum"), &want);
            for size in [1, 7, 37] {
                let chunked = run(&s, &e.df, size);
                assert_same_floats(&whole, &chunked, &format!("{what} in chunks of {size}"));
            }
        }
    }
    assert_no_failures(&failures);
}

/// The rows the window's snapshots are taken at, from the rule written out:
/// the first row; then each row that `due(stamp, newest snapshot's stamp)`
/// says is `window_every` past the newest snapshot, or whose newest snapshot
/// has left the window.
fn snapshot_rows<T: Copy>(stamps: &[T], due: impl Fn(T, T) -> bool) -> Vec<usize> {
    let mut rows: Vec<usize> = Vec::new();
    for (i, &c) in stamps.iter().enumerate() {
        if rows.last().is_none_or(|&last| due(c, stamps[last])) {
            rows.push(i);
        }
    }
    rows
}

/// Under a clock spacing the window at each row holds the rows from its
/// boundary on: the oldest snapshot no more than `window` behind the last
/// learned row. The count of rows from that snapshot's row to the last
/// learned one.
fn counts_from_boundary<T: Copy>(
    stamps: &[T],
    snaps: &[usize],
    within: impl Fn(T, T) -> bool,
) -> Vec<Option<usize>> {
    (0..stamps.len())
        .map(|i| {
            (i > 0).then(|| {
                let b = snaps
                    .iter()
                    .copied()
                    .find(|&j| j < i && within(stamps[i - 1], stamps[j]))
                    .expect("the newest snapshot is inside the window");
                i - b
            })
        })
        .collect()
}

/// One clock of [`window_every_spaces_the_snapshots_exactly_and_a_save_between_them_resumes`]:
/// the window settings, the frame, each row's count and the snapshot rows.
type SpacingCase = (&'static str, DataFrame, Vec<Option<usize>>, Vec<usize>);

/// `window_every` spaces the snapshots on the stamps, exactly, on two clocks
/// whose steps do not add exactly: tenths of a number clock under a spacing
/// of `0.3` and a window of `1.0`, and 100 ms steps of a `Datetime` clock
/// under `"300ms"` and `"1s"`. Each row's window is the one the rule written
/// out gives (the summed clock trimmed the snapshot exactly one window old
/// on rows 14 and 17 of the number clock, and on 169 rows of the `Datetime`
/// one); and a bank saved at any row between two snapshots, and loaded, goes
/// on as the one that never stopped.
#[test]
fn window_every_spaces_the_snapshots_exactly_and_a_save_between_them_resumes() {
    for closed in EDGES {
        window_every_spaces_the_snapshots_exactly(closed);
    }
}

fn window_every_spaces_the_snapshots_exactly(closed: &str) {
    let n = 300;
    let num = tenths(n);
    let ns: Vec<i64> = (0..n as i64).map(|i| T0_NS + i * 100_000_000).collect();
    let cases: Vec<SpacingCase> = vec![
        {
            let snaps = snapshot_rows(&num, |c, last| {
                c - last >= 0.3 || left_by(closed, c - last, 1.0)
            });
            (
                if closed == "right" {
                    r#""window_size": 1.0, "window_every": 0.3, "closed": "right""#
                } else {
                    r#""window_size": 1.0, "window_every": 0.3, "closed": "both""#
                },
                frame(
                    Series::new("t".into(), num.clone()),
                    &vec![false; n],
                    &vec!["s"; n],
                    5,
                ),
                counts_from_boundary(&num, &snaps, |last, j| inside_by(closed, last - j, 1.0)),
                snaps,
            )
        },
        {
            let snaps = snapshot_rows(&ns, |c, last| {
                c - last >= 300_000_000 || left_by(closed, c - last, 1_000_000_000)
            });
            (
                if closed == "right" {
                    r#""window_size": "1s", "window_every": "300ms", "closed": "right""#
                } else {
                    r#""window_size": "1s", "window_every": "300ms", "closed": "both""#
                },
                frame(
                    Series::new("t".into(), ns.clone())
                        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                        .unwrap(),
                    &vec![false; n],
                    &vec!["s"; n],
                    5,
                ),
                counts_from_boundary(&ns, &snaps, |last, j| {
                    inside_by(closed, last - j, 1_000_000_000)
                }),
                snaps,
            )
        },
    ];
    let mut failures = Vec::new();
    for (i, (window, df, want, snaps)) in cases.iter().enumerate() {
        let rest = if i == 0 {
            r#""gap_cap": 1.0, "half_life": "inf""#
        } else {
            r#""gap_cap": "1s", "half_life": "inf""#
        };
        for (kind, _, _) in MODELS {
            let s = spec(kind, window, rest);
            let whole = run(&s, df, 0);
            check(
                &mut failures,
                &format!("{kind} under {window}"),
                &field(&whole, "weight_sum"),
                want,
            );
            // A save at each row between two snapshots of the first few.
            for cut in (snaps[2] + 1..snaps[5]).filter(|c| !snaps.contains(c)) {
                let mut bank = Bank::new(vec![s.clone()]).unwrap();
                let head = df.slice(0, cut);
                let a = DataFrame::new(cut, bank.fit_predict(&head).unwrap()).unwrap();
                let bytes = bank.save_bytes().unwrap();
                let mut back = Bank::load_bytes(&bytes, None).unwrap();
                let tail = df.slice(cut as i64, n - cut);
                let b = DataFrame::new(n - cut, back.fit_predict(&tail).unwrap()).unwrap();
                let mut resumed = a;
                resumed.vstack_mut(&b).unwrap();
                assert_same_floats(&whole, &resumed, &format!("{kind} saved at row {cut}"));
            }
        }
    }
    assert_no_failures(&failures);
}

/// The residual window's boundary is the model's (review 2026-09-12, S1):
/// an `ewridge`'s `sigma`, with no decay, is the root mean square of the
/// residuals of the rows inside the model's window -- on a 1 ms clock the
/// rows less than a second old, or at most a second under `closed =
/// "both"`, the row exactly a second old included -- and `weight_sum`
/// counts the same rows.
#[test]
fn the_residual_windows_boundary_is_the_models() {
    for closed in EDGES {
        residual_windows_boundary_is_the_models(closed);
    }
}

fn residual_windows_boundary_is_the_models(closed: &str) {
    let n = 1_300;
    let (t, ns) = millisecond_rows(n, TimeUnit::Nanoseconds);
    let df = frame(t, &vec![false; n], &vec!["s"; n], 9);
    let s = spec(
        "ewridge",
        &format!(r#""window_size": "1s", "closed": "{closed}""#),
        r#""gap_cap": "1d", "half_life": "inf", "emit_sigma": true"#,
    );
    let out = run(&s, &df, 0);
    let (resid, sigma, wsum) = (
        field(&out, "resid_y"),
        field(&out, "sigma_y"),
        field(&out, "weight_sum"),
    );
    let mut bad = Vec::new();
    for i in 1..n {
        let now = ns[i - 1];
        let inside: Vec<usize> = (0..i)
            .filter(|&j| inside_by(closed, now - ns[j], 1_000_000_000))
            .collect();
        if wsum[i] != Some(inside.len() as f64) {
            bad.push((i, "weight_sum", wsum[i], inside.len() as f64));
        }
        let sq: Vec<f64> = inside
            .iter()
            .filter_map(|&j| resid[j].filter(|r| r.is_finite()))
            .map(|r| r * r)
            .collect();
        if sq.is_empty() {
            continue;
        }
        let want = (sq.iter().sum::<f64>() / sq.len() as f64).sqrt();
        match sigma[i] {
            Some(got) if (got - want).abs() <= 1e-9 * want => {}
            got => bad.push((i, "sigma", got, want)),
        }
    }
    assert!(
        bad.is_empty(),
        "{closed}: {} rows, first {:?}",
        bad.len(),
        &bad[..bad.len().min(6)]
    );
}

/// Under `embargo` each row is learned a delay after it arrives, and its
/// replay keys the row's snapshot by the stamp it arrived with. The window
/// at each row holds the learned rows less than a second behind the newest
/// learned one -- which `learned_clock` names -- or at most a second, the
/// row exactly a second behind included, under `closed = "both"`, for every
/// windowed model.
#[test]
fn an_embargoed_row_is_learned_at_its_own_stamp() {
    let n = 1_300;
    let (t, ns) = millisecond_rows(n, TimeUnit::Nanoseconds);
    let df = frame(t, &vec![false; n], &vec!["s"; n], 13);
    let mut failures = Vec::new();
    for (closed, (kind, _, _)) in EDGES
        .into_iter()
        .flat_map(|c| MODELS.into_iter().map(move |m| (c, m)))
    {
        let s = spec(
            kind,
            &format!(r#""window_size": "1s", "closed": "{closed}""#),
            r#""gap_cap": "1d", "half_life": "inf", "embargo": "5ms", "emit_clocks": true"#,
        );
        let out = run(&s, &df, 0);
        let learned: Vec<Option<i64>> = out
            .column("m")
            .unwrap()
            .struct_()
            .unwrap()
            .field_by_name("learned_clock")
            .unwrap()
            .cast(&DataType::Int64)
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .collect();
        // The rows learned by each row's turn are those up to the newest
        // learned one, in order; the window holds the last second of them.
        let want: Vec<Option<usize>> = learned
            .iter()
            .map(|l| {
                l.map(|l| {
                    let last = ns.binary_search(&l).expect("a row's clock");
                    (0..=last)
                        .filter(|&j| inside_by(closed, ns[last] - ns[j], 1_000_000_000))
                        .count()
                })
            })
            .collect();
        let edge = if closed == "right" { 1_000 } else { 1_001 };
        assert!(
            want.iter().filter(|w| **w == Some(edge)).count() > 200,
            "{kind}, {closed}: the window reaches a row exactly a second back"
        );
        check(
            &mut failures,
            &format!("{kind}, {closed}"),
            &field(&out, "weight_sum"),
            &want,
        );
    }
    assert_no_failures(&failures);
}
