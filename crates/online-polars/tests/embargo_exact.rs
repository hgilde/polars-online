//! An `embargo` decided on the elapsed clock held exactly (docs/PLAN.md
//! task 176), as a model window's edge is decided on the decayed clock held
//! exactly (task 175).
//!
//! The release counted each held row's embargo down in doubles,
//! `remaining -= elapsed`, and the rounded steps drifted: on rows 1 ms apart
//! under `embargo = "2s"` every row was learned 2.001 s after it arrived, a
//! row late, where `"5ms"`, `"300ms"` and `"1s"` happened to land. A row is
//! now released by comparing two places on the elapsed clock: integer
//! nanoseconds on a temporal clock, the raw values by one subtraction on a
//! number clock.
//!
//! Every oracle here is the definition, computed in the test from the raw
//! integer nanoseconds or the raw numbers, never from `ClockState`: row `u`
//! is learned before an accepted row `s` is scored when no restart lies
//! between them and the time that passed from `u` to `s` is at least the
//! embargo. The time that passed is every step of the clock column, skipped
//! rows' included, uncapped, and `session_gap` where a session restarts the
//! clock (task 153). With no decay and unit weights `weight_sum` counts the
//! rows learned, and `learned_clock` names the newest of them (task 152).

use online_polars::{Bank, Like, Spec, WindowsConfig, WindowsRun};
use polars::prelude::*;

/// Nanoseconds since the Unix epoch of 2024-01-01T00:00:00.
const T0_NS: i64 = 1_704_067_200_000_000_000;
const MS: i64 = 1_000_000;

/// An `ewridge` on the feature `x` and the target `y` (or the targets
/// `targets`, JSON), on clock `t`, with no decay, no weight floor and the
/// clock fields, and the spec-level settings `rest`.
fn spec_with(targets: &str, rest: &str) -> Spec {
    let text = format!(
        r#"{{"name": "m", "model": {{"type": "ew_ridge", "ridge": 1e-6}}, "targets": {targets},
            "features": ["x"], "clock": "t", "min_weight": 0.0, "half_life": "inf",
            "emit_clocks": true, {rest}}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn spec(rest: &str) -> Spec {
    spec_with(r#"["y"]"#, rest)
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
/// (null where `skip` says), a target `y`, a drifting level `mid` and a
/// session `s`.
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
    let mut level = 100.0;
    let mid: Vec<f64> = (0..n)
        .map(|_| {
            level += 0.1 * r();
            level
        })
        .collect();
    df!("t" => t, "x" => x, "y" => y, "mid" => mid, "s" => sessions.to_vec()).unwrap()
}

/// `n` rows 1 ms apart from 2024-01-01, as a `Datetime` column in `unit`,
/// and the same instants in integer nanoseconds.
fn millisecond_rows(n: usize, unit: TimeUnit) -> (Series, Vec<i64>) {
    let ns: Vec<i64> = (0..n as i64).map(|i| T0_NS + i * MS).collect();
    let per = match unit {
        TimeUnit::Milliseconds => MS,
        TimeUnit::Microseconds => 1_000,
        TimeUnit::Nanoseconds => 1,
    };
    let raw: Vec<i64> = ns.iter().map(|v| v / per).collect();
    let t = Series::new("t".into(), raw)
        .cast(&DataType::Datetime(unit, None))
        .unwrap();
    (t, ns)
}

/// `df` through a fresh bank of `spec`, in chunks of `size` rows (all at
/// once for 0), as the bank's output columns.
fn run(spec: &Spec, df: &DataFrame, size: usize) -> DataFrame {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    feed(&mut bank, df, size)
}

fn feed(bank: &mut Bank, df: &DataFrame, size: usize) -> DataFrame {
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

fn field(out: &DataFrame, name: &str) -> Series {
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap()
}

fn weight_sum(out: &DataFrame) -> Vec<Option<f64>> {
    field(out, "weight_sum").f64().unwrap().iter().collect()
}

/// `learned_clock` on a temporal clock, as integer nanoseconds whatever the
/// column's unit.
fn learned_ns(out: &DataFrame) -> Vec<Option<i64>> {
    field(out, "learned_clock")
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap()
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .iter()
        .collect()
}

fn learned_f64(out: &DataFrame) -> Vec<Option<f64>> {
    field(out, "learned_clock").f64().unwrap().iter().collect()
}

/// Per accepted row `s`, the rows learned before it is scored, from the
/// definition: the accepted rows `u < s` since the last restart for which
/// `waited(s, u)` holds -- how many, and the newest. `None` for a skipped
/// row, whose output is null.
fn released(
    accepted: &[bool],
    restarts: &[bool],
    waited: impl Fn(usize, usize) -> bool,
) -> Vec<Option<(usize, Option<usize>)>> {
    let mut start = 0;
    (0..accepted.len())
        .map(|s| {
            if restarts[s] {
                start = s;
            }
            accepted[s].then(|| {
                let rows: Vec<usize> = (start..s)
                    .filter(|&u| accepted[u] && waited(s, u))
                    .collect();
                (rows.len(), rows.last().copied())
            })
        })
        .collect()
}

/// The rows where `weight_sum` is not the count `want` gives, or the clock
/// `clock_of` reads off `learned` is not the newest row's: each with both.
fn mismatches<C: PartialEq + Copy + std::fmt::Debug>(
    want: &[Option<(usize, Option<usize>)>],
    weights: &[Option<f64>],
    learned: &[Option<C>],
    clock_of: impl Fn(usize) -> C,
) -> Vec<String> {
    let mut bad = Vec::new();
    for (s, w) in want.iter().enumerate() {
        let Some((count, newest)) = *w else { continue };
        let clock = newest.map(&clock_of);
        if weights[s] != Some(count as f64) || learned[s] != clock {
            bad.push(format!(
                "row {s}: weight_sum {:?} learned_clock {:?}, want {count} and {clock:?}",
                weights[s], learned[s]
            ));
        }
    }
    bad
}

fn assert_none(what: &str, bad: &[String]) {
    assert!(
        bad.is_empty(),
        "{what}: {} rows learned another set of rows than the embargo's, first:\n{}",
        bad.len(),
        bad[..bad.len().min(6)].join("\n")
    );
}

/// The reviewer's probe (`$S3/fixwork/cb1/embargo_probe.py`) through the
/// bank, in every `Datetime` unit: rows 1 ms apart under an embargo of
/// `"2s"` learn each row exactly 2,000 rows later, at 2.000 s, a row exactly
/// one embargo back included; and under `"5ms"`, `"300ms"` and `"1s"`, which
/// the countdown in doubles happened to land, likewise. The oracle compares
/// the rows' integer nanoseconds with the embargo's.
#[test]
fn a_row_is_learned_exactly_one_embargo_later_in_every_unit() {
    let n = 2_600;
    let mut failures = Vec::new();
    for unit in [
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        let (t, ns) = millisecond_rows(n, unit);
        let df = frame(t, &vec![false; n], &vec!["s"; n], 7);
        for (embargo, embargo_ns) in [
            ("2s", 2_000 * MS),
            ("1s", 1_000 * MS),
            ("300ms", 300 * MS),
            ("5ms", 5 * MS),
        ] {
            let want = released(&vec![true; n], &vec![false; n], |s, u| {
                ns[s] - ns[u] >= embargo_ns
            });
            let rows_back = (embargo_ns / MS) as usize;
            assert_eq!(
                want[rows_back + 10],
                Some((11, Some(10))),
                "the oracle learns row 10 at row {}",
                rows_back + 10
            );
            let s = spec(&format!(r#""gap_cap": "1d", "embargo": "{embargo}""#));
            let out = run(&s, &df, 0);
            let bad = mismatches(&want, &weight_sum(&out), &learned_ns(&out), |u| ns[u]);
            if !bad.is_empty() {
                failures.push(format!(
                    "{embargo} on {unit:?}: {} rows, first {}",
                    bad.len(),
                    bad[0]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A number clock whose steps do not add exactly, a tenth apart: a row is
/// learned at the first row whose raw value is at least the embargo past
/// its own by one subtraction, as a window's edge is decided (task 175).
/// Under `0.3`, `0.7 − 0.4` is `0.29999999999999993`, so the row at `0.4`
/// waits for `0.8`, and `0.3 − 0.0` is `0.3`, so the row at `0.0` goes at
/// `0.3`; the countdown in doubles happened to agree on every row there.
/// It did not under `0.5` and `1.0`, where it learned the rows at `0.0` and
/// `0.1` (`0.0` and `0.2`) a row late, nor under `2.0`, where it learned the
/// row at `0.3` at `2.3`, a row early: `2.3 − 0.3` is `1.9999999999999998`.
#[test]
fn a_number_clocks_release_is_one_subtraction_of_the_raw_values() {
    let n = 400;
    let t: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let df = frame(
        Series::new("t".into(), t.clone()),
        &vec![false; n],
        &vec!["s"; n],
        3,
    );
    let mut failures = Vec::new();
    for embargo in [0.3, 0.5, 1.0, 2.0] {
        let want = released(&vec![true; n], &vec![false; n], |s, u| {
            t[s] - t[u] >= embargo
        });
        if embargo == 0.3 {
            assert_eq!(want[3], Some((1, Some(0))), "0.3 − 0.0 is the embargo");
            assert_eq!(want[7], Some((4, Some(3))), "0.7 − 0.4 falls short");
            assert_eq!(want[8], Some((6, Some(5))), "0.8 − 0.5 is past it");
        }
        if embargo == 2.0 {
            assert_eq!(want[23], Some((3, Some(2))), "2.3 − 0.3 falls short");
            assert_eq!(want[24], Some((5, Some(4))));
        }
        let s = spec(&format!(r#""gap_cap": 1.0, "embargo": {embargo}"#));
        let out = run(&s, &df, 0);
        let bad = mismatches(&want, &weight_sum(&out), &learned_f64(&out), |u| t[u]);
        if !bad.is_empty() {
            failures.push(format!(
                "tenths under {embargo}: {} rows: {}",
                bad.len(),
                bad.join("; ")
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The embargo on a row-count clock: the `has_clock = false` branch, whose
/// places are the rows themselves. With no clock column every row is one
/// step, a skipped row's included, so row `u` is learned before row `s` is
/// scored when `s - u` is at least the embargo, from the definition, whole
/// and at chunks of 1, 7 and 37 rows; and `scored_clock` is the row's own
/// number. Every embargo test ran a `Datetime` or a number clock (review
/// 2026-10-06, PB6).
#[test]
fn on_a_row_count_clock_the_embargo_counts_rows_skipped_ones_included() {
    let n = 300;
    let skip: Vec<bool> = (0..n)
        .map(|i| i % 11 == 4 || (100..106).contains(&i))
        .collect();
    let accepted: Vec<bool> = skip.iter().map(|s| !s).collect();
    // `t` is in the frame and named by no spec here: the clock is the rows.
    let df = frame(
        Series::new("t".into(), vec![0.0; n]),
        &skip,
        &vec!["s"; n],
        29,
    );
    let mut failures = Vec::new();
    for embargo in [1usize, 3, 10, 40] {
        let want = released(&accepted, &vec![false; n], |s, u| s - u >= embargo);
        // Row 106, the first after six skipped rows, is learned exactly an
        // embargo later and not a row sooner; counted in accepted rows alone
        // it would go six rows later.
        let newest = |s: usize| want[s].and_then(|(_, newest)| newest);
        assert_eq!(newest(106 + embargo), Some(106), "embargo {embargo}");
        assert!(newest(105 + embargo) < Some(106), "embargo {embargo}");
        let text = format!(
            r#"{{"name": "m", "model": {{"type": "ew_ridge", "ridge": 1e-6}}, "targets": ["y"],
                "features": ["x"], "min_weight": 0.0, "half_life": "inf",
                "emit_clocks": true, "embargo": {embargo}}}"#
        );
        let s: Spec = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
        for size in [0, 1, 7, 37] {
            let out = run(&s, &df, size);
            let learned: Vec<Option<i64>> =
                field(&out, "learned_clock").i64().unwrap().iter().collect();
            let mut bad = mismatches(&want, &weight_sum(&out), &learned, |u| u as i64);
            let scored: Vec<Option<i64>> =
                field(&out, "scored_clock").i64().unwrap().iter().collect();
            for s in (0..n).filter(|&s| accepted[s] && scored[s] != Some(s as i64)) {
                bad.push(format!("row {s}: scored_clock {:?}", scored[s]));
            }
            if !bad.is_empty() {
                failures.push(format!(
                    "{embargo} rows, chunks of {size}: {} rows, first {}",
                    bad.len(),
                    bad[0]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A temporal embargo is compared with the time that passed in integer
/// nanoseconds at any length, the embargo's own duration against the rows'
/// own steps. Past about 97 days a double of seconds no longer tells a
/// nanosecond apart: `"100d2ns"` and a step of 100 days and 1 ns are both
/// 8640000.000000002 s, so the countdown learned a row 1 ns before its
/// embargo had passed. In nanoseconds the row waits for the row exactly
/// the embargo later.
#[test]
fn a_long_embargo_is_compared_in_integer_nanoseconds() {
    let day = 86_400_000 * MS;
    let embargo = 100 * day + 2;
    let ns: Vec<i64> = [0, 100 * day + 1, 100 * day + 2, 100 * day + 3]
        .iter()
        .map(|v| T0_NS + v)
        .collect();
    assert_eq!(
        online_core::seconds_of_ns(i128::from(embargo)),
        online_core::seconds_of_ns(i128::from(embargo - 1)),
        "the two are one double of seconds"
    );
    let n = ns.len();
    let t = Series::new("t".into(), ns.clone())
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    let df = frame(t, &vec![false; n], &vec!["s"; n], 17);
    let want = released(&vec![true; n], &vec![false; n], |s, u| {
        ns[s] - ns[u] >= embargo
    });
    assert_eq!(want[1], Some((0, None)), "1 ns short");
    assert_eq!(want[2], Some((1, Some(0))), "exactly the embargo");
    let s = spec(r#""gap_cap": "1d", "embargo": "100d2ns""#);
    let out = run(&s, &df, 0);
    assert_none(
        "an embargo of 100 days and 2 ns",
        &mismatches(&want, &weight_sum(&out), &learned_ns(&out), |u| ns[u]),
    );
}

/// One stream through every clock event the elapsed clock has, on a
/// `Datetime` clock, with each row's place on the elapsed clock from the
/// definition in integer nanoseconds: rows 100 ms apart, which no sum of
/// doubles adds exactly, under an embargo of `"1s"`; a gap of 700 ms past a
/// `gap_cap` of 300 ms, on an accepted row and on a skipped one, which
/// counts in full; rows skipped now and then, whose steps count; a session
/// change with the clock running on, which counts its step, not
/// `session_gap`; a session change with the clock an hour back, which
/// counts `session_gap`'s 350 ms; and a step back of two hours within the
/// session, past `restart_after_step_back`, which drops every held row.
struct Events {
    df: DataFrame,
    ns: Vec<i64>,
    /// Per row, its place on the elapsed clock since the last restart.
    place: Vec<i64>,
    accepted: Vec<bool>,
    restarts: Vec<bool>,
}

const CAP_NS: i64 = 300 * MS;
const SESSION_GAP_NS: i64 = 350 * MS;
const EMBARGO_NS: i64 = 1_000 * MS;

/// The settings `events()` is run under.
const EVENT_CLOCK: &str = r#""gap_cap": "300ms", "session": "s", "session_gap": "350ms",
    "restart_after_step_back": "1m", "embargo": "1s""#;

fn events() -> Events {
    let mut steps: Vec<i64> = Vec::new();
    let mut sessions: Vec<&str> = Vec::new();
    let mut skip: Vec<bool> = Vec::new();
    let mut push = |step: i64, session: &'static str, skipped: bool| {
        steps.push(step);
        sessions.push(session);
        skip.push(skipped);
    };
    push(0, "s1", false);
    for i in 1..300 {
        push(100 * MS, "s1", i % 23 == 7);
    }
    // A gap past the cap and short of the embargo, on an accepted row.
    push(700 * MS, "s1", false);
    for _ in 0..200 {
        push(100 * MS, "s1", false);
    }
    // The same gap on a skipped row.
    push(700 * MS, "s1", true);
    for _ in 0..200 {
        push(100 * MS, "s1", false);
    }
    // A session change with the clock running on.
    push(100 * MS, "s2", false);
    for _ in 0..200 {
        push(100 * MS, "s2", false);
    }
    // A session change with the clock an hour back.
    push(-3_600_000 * MS, "s3", false);
    for _ in 0..200 {
        push(100 * MS, "s3", false);
    }
    // Two hours back within the session: a restart.
    push(-7_200_000 * MS, "s3", false);
    for _ in 0..200 {
        push(100 * MS, "s3", false);
    }
    let n = steps.len();
    let mut ns = Vec::with_capacity(n);
    let mut at = T0_NS;
    for &s in &steps {
        at += s;
        ns.push(at);
    }

    // The elapsed clock, from the definition.
    let mut place = vec![0i64; n];
    let mut restarts = vec![false; n];
    for i in 1..n {
        let raw = ns[i] - ns[i - 1];
        let step = if raw >= 0 {
            raw
        } else if sessions[i] != sessions[i - 1] {
            SESSION_GAP_NS
        } else {
            restarts[i] = true;
            0
        };
        place[i] = if restarts[i] { 0 } else { place[i - 1] + step };
    }
    let t = Series::new("t".into(), ns.clone())
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    Events {
        df: frame(t, &skip, &sessions, 11),
        ns,
        place,
        accepted: skip.iter().map(|s| !s).collect(),
        restarts,
    }
}

/// Through every clock event, a row is learned once the time that passed
/// since it is at least the embargo, exactly: across the capped gap the
/// whole 700 ms count, not the cap; across the session change that runs on,
/// the step; across the one that restarts the clock, `session_gap`; a skipped
/// row's step counts; and the restart drops the rows held before it, which
/// are never learned.
#[test]
fn across_every_clock_event_a_row_waits_the_time_that_passed_exactly() {
    let e = events();
    let want = released(&e.accepted, &e.restarts, |s, u| {
        e.place[s] - e.place[u] >= EMBARGO_NS
    });
    // The oracle has edges to decide, and every event is in the stream: rows
    // exactly one embargo before another, the capped gap releasing the held
    // rows its whole 700 ms reaches (seven, where the cap's 300 ms reaches
    // three) and holding the rest, and the restart.
    let exact = (0..e.ns.len())
        .filter(|&s| (0..s).any(|u| e.place[s] - e.place[u] == EMBARGO_NS))
        .count();
    assert!(
        exact > 1_000,
        "{exact} rows sit exactly an embargo after another"
    );
    let gap = 300;
    let (Some((at_gap, _)), Some((before, _))) = (want[gap], want[gap - 1]) else {
        panic!("the gap's row is accepted");
    };
    assert_eq!(at_gap, before + 7, "the gap releases seven rows at once");
    let capped = (0..gap)
        .filter(|&u| e.accepted[u] && e.place[gap - 1] + CAP_NS - e.place[u] >= EMBARGO_NS)
        .count();
    assert_eq!(
        capped,
        before + 3,
        "where the cap in its place releases three"
    );
    assert!(
        (0..gap).any(|u| e.accepted[u] && e.place[gap] - e.place[u] < EMBARGO_NS),
        "and holds the rest"
    );
    assert_eq!(e.restarts.iter().filter(|&&r| r).count(), 1);
    let s = spec(EVENT_CLOCK);
    let out = run(&s, &e.df, 0);
    assert_none(
        "every clock event",
        &mismatches(&want, &weight_sum(&out), &learned_ns(&out), |u| e.ns[u]),
    );
}

/// Every float field of the spec's struct and `learned_clock`, row by row,
/// bit for bit.
fn assert_same(a: &DataFrame, b: &DataFrame, what: &str) {
    let (sa, sb) = (
        a.column("m").unwrap().struct_().unwrap().fields_as_series(),
        b.column("m").unwrap().struct_().unwrap().fields_as_series(),
    );
    for (fa, fb) in sa.iter().zip(&sb) {
        if let (Ok(xa), Ok(xb)) = (fa.f64(), fb.f64()) {
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
    assert_eq!(learned_ns(a), learned_ns(b), "{what}: learned_clock");
}

/// Chunk invariance across every clock event (hard rule 3): chunks of 1, 7
/// and 37 rows give the whole run's bits, and the whole run holds the
/// definition's releases; and a bank saved at rows where it holds rows
/// under the embargo -- before and after the capped gaps, the session
/// changes and the restart -- and loaded goes on as the one that never
/// stopped.
#[test]
fn chunking_and_a_save_while_rows_are_held_move_no_release() {
    let e = events();
    let s = spec(EVENT_CLOCK);
    let whole = run(&s, &e.df, 0);
    let want = released(&e.accepted, &e.restarts, |s, u| {
        e.place[s] - e.place[u] >= EMBARGO_NS
    });
    assert_none(
        "the whole run",
        &mismatches(&want, &weight_sum(&whole), &learned_ns(&whole), |u| e.ns[u]),
    );
    for size in [1, 7, 37] {
        assert_same(&whole, &run(&s, &e.df, size), &format!("chunks of {size}"));
    }
    let n = e.df.height();
    for cut in [
        5, 150, 299, 300, 301, 503, 704, 905, 906, 1_107, 1_108, 1_200,
    ] {
        let mut bank = Bank::new(vec![s.clone()]).unwrap();
        let head = feed(&mut bank, &e.df.slice(0, cut), 0);
        // Rows are held at the cut: the rows learned trail the rows read.
        let held = weight_sum(&head)
            .last()
            .copied()
            .flatten()
            .is_none_or(|w| w + 1.0 < e.accepted[..cut].iter().filter(|&&a| a).count() as f64);
        assert!(held, "rows are held at row {cut}");
        let bytes = bank.save_bytes().unwrap();
        let mut back = Bank::load_bytes(&bytes, None).unwrap();
        let tail = feed(&mut back, &e.df.slice(cut as i64, n - cut), 0);
        let mut resumed = head;
        resumed.vstack_mut(&tail).unwrap();
        assert_same(&whole, &resumed, &format!("saved at row {cut}"));
    }
}

/// The formula a target is learned as: the next window's decayed mean of
/// `mid` less the row's own, over a forward window of `window`.
fn tree(window: &str) -> String {
    format!(
        r#"["-", ["rewm_mean", ["col", "mid"], {{"half_life": "1s", "window_size": "{window}"}}], ["col", "mid"]]"#
    )
}

/// The column form of a formula target: the formula written by the window
/// operators `like=` the spec, then fed back as a plain target.
fn column_form(df: &DataFrame, tree: &str, like: &Spec) -> DataFrame {
    let config: WindowsConfig = serde_json::from_str(&format!(
        r#"{{"formulas": [{{"name": "fwd", "tree": {tree}}}], "clock": "t", "gap_cap": "1d"}}"#
    ))
    .unwrap();
    let config = WindowsConfig {
        like: Some(Like {
            spec: like.name.clone(),
            accept: like.features.clone(),
        }),
        ..config
    };
    let mut run = WindowsRun::new(config, df.schema()).unwrap();
    let mut out = run.feed(df, None).unwrap();
    out.vstack_mut(&run.finish().unwrap()).unwrap();
    out
}

/// A formula target under an embargo equal to its window (task 104), on
/// rows 1 ms apart under `"2s"`, on both paths. The column form knows the
/// target at its row, so the embargo alone releases it: exactly 2,000 rows
/// later. The native path also waits for the window to close, and under
/// `closed="right"` a row exactly one window later is inside it, so the
/// window closes a row after: 2,001 rows later. With a window of `"1s"`
/// under the same embargo the native path's window closes first and the
/// embargo releases the row, exactly 2,000 rows later, as the column form
/// does.
#[test]
fn a_formula_target_under_an_embargo_equal_to_its_window_on_both_paths() {
    let n = 2_600;
    let (t, ns) = millisecond_rows(n, TimeUnit::Nanoseconds);
    let df = frame(t, &vec![false; n], &vec!["s"; n], 5);
    let all = vec![true; n];
    let none = vec![false; n];
    let embargo = 2_000 * MS;
    let rest = r#""gap_cap": "1d", "embargo": "2s""#;
    let mut failures = Vec::new();
    for (window, window_ns) in [("2s", 2_000 * MS), ("1s", 1_000 * MS)] {
        let tree = tree(window);
        let native = spec_with(&format!(r#"[{{"name": "fwd", "formula": {tree}}}]"#), rest);
        let plain = spec_with(r#"["fwd"]"#, rest);
        // The column form: the embargo alone.
        let want = released(&all, &none, |s, u| ns[s] - ns[u] >= embargo);
        let out = run(&plain, &column_form(&df, &tree, &plain), 0);
        let bad = mismatches(&want, &weight_sum(&out), &learned_ns(&out), |u| ns[u]);
        if !bad.is_empty() {
            failures.push(format!(
                "column form, window {window}: {} rows, first {}",
                bad.len(),
                bad[0]
            ));
        }
        // The native path: the window closed -- a row past its far edge --
        // and the embargo passed.
        let want = released(&all, &none, |s, u| {
            ns[s] - ns[u] > window_ns && ns[s] - ns[u] >= embargo
        });
        let rows_back = if window == "2s" { 2_001 } else { 2_000 };
        assert_eq!(want[rows_back], Some((1, Some(0))), "window {window}");
        let out = run(&native, &df, 0);
        let bad = mismatches(&want, &weight_sum(&out), &learned_ns(&out), |u| ns[u]);
        if !bad.is_empty() {
            failures.push(format!(
                "native, window {window}: {} rows, first {}",
                bad.len(),
                bad[0]
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// After a drift reset under an embargo, `settled_frac` still counts the
/// clock of the rows held at the reset (review round 4, PB1). A drift reset
/// keeps the held rows -- each teaches the rebuilt model as it is released
/// -- and the reset zeroed the clock they covered, so every release took
/// back a delta the reset had never added, and the field left their clock
/// out for the rest of the stream: `min_settled_frac = 0.95` opened 59 rows
/// late. The oracle is the definition, `1 - 2^(-T/h)` with `T` the clock
/// the decay has covered: on a row-count clock `T = r - 1` before row `r`
/// (the first row decays by nothing), and after a reset at row `R` the
/// rebuilt model has covered none of it while the `E - 1` rows held there
/// have covered `E - 1`, so `T = j + E - 1` at row `R + j`. A grid of
/// half-lives resets every instance together; one instance resets itself.
#[test]
fn after_a_drift_reset_settled_frac_keeps_the_held_rows_clock() {
    let (n, change, embargo) = (1_200usize, 400usize, 60usize);
    let mut r = lcg(17);
    let x: Vec<f64> = (0..n).map(|_| r()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let level = if i < change {
                2.0 * x[i]
            } else {
                -2.0 * x[i] + 8.0
            };
            level + 0.05 * r()
        })
        .collect();
    let df = df!("x" => x, "y" => y).unwrap();
    for (half_lives, suffixes) in [
        ("20.0", vec![("", 20.0)]),
        ("[20.0, 40.0]", vec![("@h20", 20.0), ("@h40", 40.0)]),
    ] {
        let spec: Spec = serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {{"type": "ew_ridge", "ridge": 1e-6, "standardize": false,
                 "max_rows_between_solves": 1}}, "targets": ["y"], "features": ["x"],
                 "half_life": {half_lives}, "min_weight": 3.0, "embargo": {embargo},
                 "emit_drift": true, "drift_delta": 0.5, "drift_threshold": 5.0,
                 "drift_action": "reset"}}"#
        ))
        .unwrap();
        for size in [0, 7, 37] {
            let out = run(&spec, &df, size);
            let what = format!("half_life {half_lives}, chunks of {size}");
            // The rows a drift flag is on, in any instance: each is a reset
            // of every instance, at the row that released the row that
            // tripped it.
            let flagged: Vec<usize> = (0..n)
                .filter(|&i| {
                    suffixes.iter().any(|(sfx, _)| {
                        field(&out, &format!("drift_y{sfx}")).bool().unwrap().get(i) == Some(true)
                    })
                })
                .collect();
            // Evidence the reset happened, and where: the first row after
            // the change is released `embargo` rows later.
            let reset = *flagged
                .first()
                .unwrap_or_else(|| panic!("{what}: no drift reset"));
            assert!(reset >= change + embargo, "{what}: a reset at {reset}");
            let end = flagged.iter().copied().find(|&i| i > reset).unwrap_or(n);
            assert!(end - reset > 200, "{what}: the next reset at {end}");
            for (sfx, h) in &suffixes {
                let got = field(&out, &format!("settled_frac{sfx}"));
                let got = got.f64().unwrap();
                let at = |row: usize, t: f64| {
                    let want = 1.0 - (-(t / h)).exp2();
                    let v = got.get(row).unwrap();
                    assert!(
                        (v - want).abs() <= 1e-12,
                        "{what}, instance {sfx:?}: row {row} reads {v}, the definition {want} \
                         (T = {t}; reset at {reset})"
                    );
                };
                for row in 1..reset {
                    at(row, (row - 1) as f64);
                }
                for row in reset..end {
                    at(row, (row - reset + embargo - 1) as f64);
                }
            }
        }
    }
}
