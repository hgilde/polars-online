//! `solve_every`, `pca_every` and `prune_every` decided on the decayed clock
//! held exactly (docs/PLAN.md task 180), as `window_every` (task 175) and
//! `coef_every` (task 178) are.
//!
//! Each of the three summed its clock since the last event in doubles, and
//! the rounded steps drifted: two thousand steps of 1 ms sum to
//! 1.9999999999998905 s, so a cadence of `"2s"` fired a row late, and every
//! event after it a row later again. Each now keeps the stamp of its last
//! event (a solve of `ewridge`, `lasso`, `huber` or `quantile`, a refresh of
//! `ew_cov`'s components, a checkpoint of `micro`) and compares the row's
//! stamp with it: integer nanoseconds on a temporal clock, one subtraction
//! of the raw values on a number clock.
//!
//! Every oracle here is the definition, computed in the test from the raw
//! integer nanoseconds or the raw numbers, never from `ClockState`: after a
//! model's first event, each next one is at the first row whose clock is at
//! least the cadence past the last event's. A regression's first solve and
//! `ew_cov`'s first refresh are forced by `min_weight`, so they are read
//! from the output; `micro` has none, and its clock runs from the first row.
//!
//! The events are read from the output: a solve where `coef`, written on
//! every row, changes; a refresh where the loadings in force change between
//! a row and the next (a row is scored with the ones before it); a
//! checkpoint where the live summaries (`n_micro`) fall between a row and
//! the next -- each row opens a summary of its own (`eps` is tiny), none is
//! evicted, and a checkpoint prunes the faded ones.

use online_polars::{Bank, Spec};
use polars::prelude::*;

/// What an event looks like in a model's output.
#[derive(Clone, Copy, PartialEq)]
enum Event {
    /// A solve: `coef` changes.
    Coef,
    /// A refresh of the components: the loadings in force change.
    Loadings,
    /// A checkpoint: the live summaries fall.
    Pruned,
}

/// The five cadences, `huber` and `quantile` for `robust`'s: a name, the
/// model's JSON with `{every}` where its cadence goes, its targets, its
/// features, the spec-level settings beyond the clock's, and its event.
const CASES: [(&str, &str, &str, &str, &str, Event); 6] = [
    (
        "ewridge",
        r#"{"type": "ewridge", "ridge": 1e-6, "solve_every": {every}}"#,
        r#""targets": ["y"],"#,
        r#"["x"]"#,
        r#""coef_every": 0,"#,
        Event::Coef,
    ),
    (
        "lasso",
        r#"{"type": "lasso", "lasso_path": [0.0], "solve_every": {every}}"#,
        r#""targets": ["y"],"#,
        r#"["x"]"#,
        r#""coef_every": 0,"#,
        Event::Coef,
    ),
    (
        "huber",
        r#"{"type": "huber", "solve_every": {every}}"#,
        r#""targets": ["y"],"#,
        r#"["x"]"#,
        r#""coef_every": 0,"#,
        Event::Coef,
    ),
    (
        "quantile",
        r#"{"type": "quantile", "quantile": 0.5, "solve_every": {every}}"#,
        r#""targets": ["y"],"#,
        r#"["x"]"#,
        r#""coef_every": 0,"#,
        Event::Coef,
    ),
    (
        "ew_cov",
        r#"{"type": "ew_cov", "stats": ["mean"], "pca": 1, "pca_every": {every}}"#,
        "",
        r#"["x", "x2"]"#,
        "",
        Event::Loadings,
    ),
    (
        "micro",
        r#"{"type": "micro", "eps": 1e-6, "max_clusters": 6000, "prune_every": {every}}"#,
        "",
        r#"["x", "x2"]"#,
        r#""coef_every": 0,"#,
        Event::Pruned,
    ),
];

/// The clock parameters of a temporal spec and of a number one: no decay
/// but `micro`'s, whose summaries must fade between checkpoints.
fn clock_fields(name: &str, temporal: bool) -> &'static str {
    match (name, temporal) {
        ("micro", true) => r#""gap_cap": "1h", "half_life": "50ms""#,
        ("micro", false) => r#""gap_cap": 100.0, "half_life": 0.05"#,
        (_, true) => r#""gap_cap": "1h", "half_life": "inf""#,
        (_, false) => r#""gap_cap": 100.0, "half_life": "inf""#,
    }
}

/// Case `name`'s spec on clock `t` under the cadence `every` (JSON: a
/// duration's string or a number), with no weight floor.
fn spec(name: &str, every: &str, temporal: bool) -> (Spec, Event) {
    let (_, model, targets, features, rest, event) =
        CASES.iter().find(|c| c.0 == name).copied().expect("a case");
    let model = model.replace("{every}", every);
    let clock = clock_fields(name, temporal);
    let text = format!(
        r#"{{"name": "m", "model": {model}, {targets} "features": {features}, "clock": "t",
            "min_weight": 0.0, {rest} {clock}}}"#
    );
    let spec = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
    (spec, event)
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

/// Rows at the clock values `t` (a column of any dtype), two features and a
/// target, every value distinct.
fn frame(t: Series, seed: u64) -> DataFrame {
    let n = t.len();
    let mut r = lcg(seed);
    let x: Vec<f64> = (0..n).map(|_| r()).collect();
    let x2: Vec<f64> = x.iter().map(|v| 0.5 * v + r()).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + r()).collect();
    df!("t" => t, "x" => x, "x2" => x2, "y" => y).unwrap()
}

/// Nanoseconds since the Unix epoch of 2024-01-01T00:00:00.
const T0_NS: i64 = 1_704_067_200_000_000_000;
const MS: i64 = 1_000_000;

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

/// `df` through `bank`, in chunks of `size` rows (all at once for 0), as
/// the bank's output columns.
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

fn run(spec: &Spec, df: &DataFrame, size: usize) -> DataFrame {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    feed(&mut bank, df, size)
}

fn field(out: &DataFrame, name: &str) -> Series {
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap()
}

/// The rows an event happened at, read from the output as the module docs
/// say.
fn events(out: &DataFrame, event: Event) -> Vec<usize> {
    match event {
        Event::Coef => {
            let coef: Vec<Option<Vec<u64>>> = field(out, "coef")
                .list()
                .unwrap()
                .amortized_iter()
                .map(|s| {
                    s.map(|s| {
                        s.as_ref()
                            .f64()
                            .unwrap()
                            .iter()
                            .map(|v| v.map_or(u64::MAX, f64::to_bits))
                            .collect()
                    })
                })
                .collect();
            (0..coef.len())
                .filter(|&i| coef[i].is_some() && (i == 0 || coef[i] != coef[i - 1]))
                .collect()
        }
        Event::Loadings => {
            let cols: Vec<Vec<Option<u64>>> = ["pc0_loading_x", "pc0_loading_x2"]
                .iter()
                .map(|f| {
                    field(out, f)
                        .f64()
                        .unwrap()
                        .iter()
                        .map(|v| v.map(f64::to_bits))
                        .collect()
                })
                .collect();
            let at = |i: usize| cols.iter().map(|c| c[i]).collect::<Vec<_>>();
            (0..out.height() - 1)
                .filter(|&i| at(i) != at(i + 1))
                .collect()
        }
        Event::Pruned => {
            let n: Vec<Option<f64>> = field(out, "n_micro")
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .iter()
                .collect();
            (0..out.height() - 1)
                .filter(|&i| matches!((n[i], n[i + 1]), (Some(a), Some(b)) if b < a))
                .collect()
        }
    }
}

/// The definition: from `first` (the model's first event; `None` for a
/// clock that runs from the first row), each next event is the first row
/// whose clock is at least the cadence past the last event's, `reached(row,
/// last)` deciding it from the raw values. An event read from the next
/// row's output cannot be seen at the last row.
fn oracle(
    n: usize,
    event: Event,
    first: Option<usize>,
    reached: impl Fn(usize, usize) -> bool,
) -> Vec<usize> {
    let (mut rows, mut last) = match first {
        Some(f) => (vec![f], f),
        None => (vec![], 0),
    };
    let seen = if event == Event::Coef { n } else { n - 1 };
    for i in last + 1..seen {
        if reached(i, last) {
            rows.push(i);
            last = i;
        }
    }
    rows
}

/// The model's first event, which the cadence counts from: `micro`'s clock
/// runs from the first row instead.
fn first_event(got: &[usize], event: Event) -> Option<usize> {
    match event {
        Event::Pruned => None,
        _ => got.first().copied(),
    }
}

/// Task 180: on 1 ms `Datetime` rows, in each unit, every cadence of `"2s"`
/// fires every 2,000th row after its first event, as the raw nanoseconds
/// say. Summed in doubles each came a row late: 2001 rows apart, the first
/// checkpoint at 2001.
#[test]
fn every_cadence_fires_every_two_thousand_rows_of_a_millisecond_clock() {
    let n = 6_100;
    let mut failures = Vec::new();
    for unit in [
        TimeUnit::Milliseconds,
        TimeUnit::Microseconds,
        TimeUnit::Nanoseconds,
    ] {
        let (t, ns) = millisecond_rows(n, unit);
        let df = frame(t, 11);
        for (name, ..) in CASES {
            let (spec, event) = spec(name, r#""2s""#, true);
            let got = events(&run(&spec, &df, 0), event);
            let want = oracle(n, event, first_event(&got, event), |i, last| {
                i128::from(ns[i]) - i128::from(ns[last]) >= 2_000_000_000
            });
            if got != want {
                failures.push(format!(
                    "{name} {unit:?}: fired {got:?}, the clock says {want:?}"
                ));
            }
            // The case is the claim: three events after the first, 2,000
            // rows apart.
            let gaps: Vec<usize> = want.windows(2).map(|w| w[1] - w[0]).collect();
            assert!(
                want.len() >= 3 && gaps.iter().all(|&g| g == 2_000),
                "{name}: {want:?}"
            );
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A thousand rows of a number clock at tenths one to three apart: the
/// raw values `k / 10`, `k` growing by 1, 2 or 3.
fn irregular_tenths() -> Vec<f64> {
    let mut r = lcg(41);
    let mut k = 0.0_f64;
    (0..1_000)
        .map(|_| {
            k += 1.0 + (1.5 * (r() + 1.0)).floor();
            k / 10.0
        })
        .collect()
}

/// The rows the old rule fired at, for a case to show it parts from the
/// definition: the steps the stream hands on a number clock, the raw
/// values' differences, summed in doubles from the last event.
fn by_summed_steps(t: &[f64], event: Event, first: Option<usize>, every: f64) -> Vec<usize> {
    let (mut rows, mut clock) = (first.into_iter().collect::<Vec<_>>(), 0.0_f64);
    let seen = if event == Event::Coef {
        t.len()
    } else {
        t.len() - 1
    };
    for i in first.unwrap_or(0) + 1..seen {
        clock += t[i] - t[i - 1];
        if clock >= every {
            rows.push(i);
            clock = 0.0;
        }
    }
    rows
}

/// A number clock of tenths: an event once one subtraction of the raw
/// values reaches the cadence. At `i / 10` under 0.3 that lands 3 rows
/// after the last on some events and 4 on others (`0.3 - 0.0` reaches it,
/// `1.2 - 0.9` does not); the raw values' differences summed from the last
/// event land on the same rows there, so this case held before task 180
/// too. Tenths one to three apart under 1.3 are where those sums part from
/// the subtraction, and the old rule fired elsewhere.
#[test]
fn a_number_clock_of_tenths_fires_by_one_subtraction_of_its_raw_values() {
    let regular: Vec<f64> = (0..400).map(|i| i as f64 / 10.0).collect();
    let irregular = irregular_tenths();
    let mut failures = Vec::new();
    for (t, every, json) in [(&regular, 0.3, "0.3"), (&irregular, 1.3, "1.3")] {
        let n = t.len();
        let df = frame(Series::new("t".into(), t.clone()), 12);
        for (name, ..) in CASES {
            let (spec, event) = spec(name, json, false);
            let got = events(&run(&spec, &df, 0), event);
            let first = first_event(&got, event);
            let want = oracle(n, event, first, |i, last| t[i] - t[last] >= every);
            if got != want {
                failures.push(format!(
                    "{name} under {every}: fired {got:?},\n  the clock says {want:?}"
                ));
            }
            let parted = by_summed_steps(t, event, first, every) != want;
            assert_eq!(parted, every == 1.3, "{name} under {every}: the case");
            if every == 0.3 {
                let gaps: std::collections::BTreeSet<usize> =
                    want.windows(2).map(|w| w[1] - w[0]).collect();
                assert_eq!(gaps, [3, 4].into(), "{name}: the case needs both");
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Hard rule 3: the events, and every output with them, are the same fed
/// in chunks of 1, 7 and 37 rows as in one.
#[test]
fn the_events_do_not_depend_on_the_chunking() {
    let n = 4_100;
    let (t, _) = millisecond_rows(n, TimeUnit::Microseconds);
    let df = frame(t, 13);
    for (name, ..) in CASES {
        let (spec, event) = spec(name, r#""2s""#, true);
        let whole = run(&spec, &df, 0);
        assert_eq!(
            events(&whole, event).len(),
            3 - usize::from(event == Event::Pruned)
        );
        for size in [1, 7, 37] {
            let parts = run(&spec, &df, size);
            assert!(parts.equals_missing(&whole), "{name}: chunks of {size}");
        }
    }
}

/// A save between two events resumes where the unbroken run goes on: the
/// stamp of the last event is state. On 1 ms rows, and on the irregular
/// tenths under 1.3 at every save point among their first forty rows and
/// some later: there the clock since the last event, summed in doubles, is
/// not the raw values' difference, and at rows 6 and 7 (a regression's,
/// `micro`'s) and 8 to 10 (`ew_cov`'s) a resume that rebuilt the last
/// event's stamp from that sum would fire a row off. (On a temporal clock
/// the sum gives the nanoseconds back exactly, so only a number clock can
/// show a stamp that was not saved.)
#[test]
fn a_save_between_two_events_resumes_where_the_unbroken_run_does() {
    let (t, _) = millisecond_rows(4_600, TimeUnit::Nanoseconds);
    let ms = frame(t, 14);
    let numbers = frame(Series::new("t".into(), irregular_tenths()), 16);
    let early_and_late: Vec<usize> = (2..=40).chain((41..1_000).step_by(97)).collect();
    let mut failures = Vec::new();
    for (name, ..) in CASES {
        for (df, every, temporal, cuts) in [
            (&ms, r#""2s""#, true, vec![1_000, 2_001, 3_999]),
            (&numbers, "1.3", false, early_and_late.clone()),
        ] {
            let (spec, event) = spec(name, every, temporal);
            let whole = run(&spec, df, 0);
            assert!(events(&whole, event).len() >= 2, "{name}: too few events");
            for cut in cuts {
                let mut first = Bank::new(vec![spec.clone()]).unwrap();
                feed(&mut first, &df.head(Some(cut)), 0);
                let saved = first.save_bytes().unwrap();
                let mut resumed =
                    Bank::load_bytes(&saved, Some(std::slice::from_ref(&spec))).unwrap();
                let tail = feed(&mut resumed, &df.slice(cut as i64, df.height() - cut), 0);
                if !tail.equals_missing(&whole.slice(cut as i64, df.height() - cut)) {
                    failures.push(format!("{name} {every}: resumed at {cut}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
