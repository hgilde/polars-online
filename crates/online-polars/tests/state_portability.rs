//! Cross-platform state test (docs/PLAN.md §9 class 7; hard rule 5).
//!
//! A state written on one OS must load on another and go on. The release
//! does the real hand-off: `release.yml`'s write-state leg runs this test on
//! macOS with `ONLINE_WRITE_STATE`, uploads the file, and the read-state legs
//! run it on Windows and Linux with `ONLINE_FOREIGN_STATE`. That is the
//! interface, and it is unchanged. One file carries three states, framed
//! (`pack`):
//!
//! - a bank of every kind `tests/test_model_registry.py`'s `MINIMAL` builds,
//!   on a row-count clock, beside a spec on a `Datetime` clock with
//!   duration parameters and a weight column, a windowed one, one under an
//!   `embargo` with rows held across the save, a `group_close` one, and a
//!   formula target with a window open across the save;
//! - a `refresh_time` state, part-way through an interval;
//! - a `with_windows` state, with rows held for a window that looks ahead.
//!
//! The read leg loads each, feeds the rest of its stream, and compares the
//! output with the run that never left this machine, to the golden
//! pipeline's tolerance: the numbers are computed on the writer's OS, whose
//! libm may differ in a last bit (`tests/test_golden_pipeline.py`). Bytes are
//! compared where the format promises them: a state re-saved after a load is
//! the bytes it was loaded from, on any OS, and two runs of one input save
//! the same bytes. It held one `ewridge` spec on a number clock (review
//! 2026-10-06, TA2 and PC12).
//!
//! Locally, and whenever the variables are absent, every check runs on a
//! file this process wrote: the same code path, with only the writer
//! differing.

use std::path::PathBuf;

use online_polars::{Bank, RefreshCols, RefreshTime, Spec, WindowsConfig, WindowsRun};
use polars::prelude::*;

/// Each kind as `MINIMAL` builds it (generated from the Python builders and
/// compacted), then the settings the hand-off needs beside them: `(name,
/// model, the rest of the spec)`. `test_model_registry.py` holds this list to
/// the registry.
const SPECS: &[(&str, &str, &str)] = &[
    (
        "bocpd",
        r#"{"type": "bocpd", "hazard": 250.0, "emission": "diag"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"]"#,
    ),
    (
        "corrchange",
        r#"{"type": "corrchange", "kind": "monitor", "span_rows": 20, "alpha": 0.05, "alpha_adjust": "bonferroni", "scalar": false, "norm": "l1", "reset_on_flag": false}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"]"#,
    ),
    (
        "deco",
        r#"{"type": "deco", "dynamics": "ew"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 50.0"#,
    ),
    (
        "ew_class",
        r#"{"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0}"#,
        r#""targets": ["lab"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "ew_cov",
        r#"{"type": "ew_cov"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 50.0"#,
    ),
    (
        "ewridge",
        r#"{"type": "ewridge", "standardize": false, "ridge_scale": "mean", "target_gaps": "own_rows"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "ftrl",
        r#"{"type": "ftrl", "strict_binary": false, "loss": "logistic"}"#,
        r#""targets": ["yb"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "hmm",
        r#"{"type": "hmm", "k": 2, "covariance": "full", "precision_prior": 0.1, "learn": true}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 50.0"#,
    ),
    (
        "holt",
        r#"{"type": "holt"}"#,
        r#""targets": ["y"], "features": [], "half_life": 50.0"#,
    ),
    (
        "huber",
        r#"{"type": "huber", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "kalman",
        r#"{"type": "kalman", "coef_half_life": 50.0, "share_p": false, "standardize": true}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "kmeans",
        r#"{"type": "kmeans", "k": 2}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 50.0"#,
    ),
    (
        "lasso",
        r#"{"type": "lasso", "lasso_path": [0.1, 0.0], "target_gaps": "own_rows"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "marginal",
        r#"{"type": "marginal"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "micro",
        r#"{"type": "micro", "eps": 0.3}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "half_life": 50.0"#,
    ),
    (
        "pa",
        r#"{"type": "pa", "mode": "pa1"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "quantile",
        r#"{"type": "quantile", "quantile": 0.5, "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "rcov",
        r#"{"type": "rcov", "kind": "kernel", "kernel": "parzen", "jitter": 2, "psd": true, "block_rows": 100}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "group": "g", "group_close": "monotone""#,
    ),
    (
        "rls",
        r#"{"type": "rls"}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "seqtest",
        r#"{"type": "seqtest"}"#,
        r#""targets": ["y"], "features": []"#,
    ),
    (
        "sgd",
        r#"{"type": "sgd", "loss": "squared", "learning_rate": 0.01, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    // `sgd`'s per-loss state (review round 5, D3): the residual scale under
    // the Huber loss, the target's spread under the epsilon-insensitive one,
    // and the logistic link.
    (
        "sgd_huber",
        r#"{"type": "sgd", "loss": "huber", "huber_delta": 1.5, "learning_rate": 0.01, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "sgd_eps",
        r#"{"type": "sgd", "loss": "epsilon_insensitive", "eps": 0.5, "learning_rate": 0.01, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["y"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "sgd_logistic",
        r#"{"type": "sgd", "loss": "logistic", "learning_rate": 0.01, "schedule": "constant", "standardize": false}"#,
        r#""targets": ["yb"], "features": ["x0"], "half_life": 50.0"#,
    ),
    (
        "on_a_datetime",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "clock": "ts", "half_life": "10m", "gap_cap": "1h", "weight": "w""#,
    ),
    (
        "windowed",
        r#"{"type": "ew_cov", "window_size": 30.0}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0"#,
    ),
    (
        "embargoed",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 40.0, "gap_cap": 50.0, "embargo": 3.0"#,
    ),
    (
        "closing",
        r#"{"type": "ew_cov"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "group": "g", "group_close": "monotone""#,
    ),
    (
        "formula",
        r#"{"type": "ewridge"}"#,
        r#""targets": [{"name": "fwd", "formula": ["-", ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}], ["col", "mid"]]}], "features": ["x0"], "clock": "t", "half_life": 40.0, "gap_cap": 50.0, "embargo": 12.0"#,
    ),
    // The integer clock (task 200; review round 5, B1): an `Int64` clock
    // under a fractional `gap_cap` and `embargo`, saved right after a
    // skipped row; a `UInt32` clock starting over at each session, under a
    // `session_gap`; and a group closed on session, whose PCA continuity
    // is kept per group.
    (
        "on_an_int64",
        r#"{"type": "ewridge"}"#,
        r#""targets": ["y"], "features": ["x0", "xi"], "clock": "ti", "half_life": 160.0, "gap_cap": 6.5, "embargo": 12.5, "weight": "w""#,
    ),
    (
        "on_a_uint32",
        r#"{"type": "ew_cov"}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "clock": "tu", "half_life": 80.0, "gap_cap": 6.5, "session": "s", "session_gap": 3.5"#,
    ),
    (
        "sessioned",
        r#"{"type": "ew_cov", "pca": 1}"#,
        r#""targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "group": "g", "session": "s", "group_close": "session""#,
    ),
];

/// The specs that close groups, whose estimates `closed_groups` reports.
const CLOSING: [&str; 3] = ["rcov", "closing", "sessioned"];

/// The rows of the stream, and the row it is saved at.
const N: usize = 400;
const SPLIT: usize = 200;

/// How far two platforms' numbers may part, relative: the golden pipeline's.
const TOL: f64 = 1e-12;

fn specs() -> Vec<Spec> {
    SPECS
        .iter()
        .map(|(name, model, rest)| {
            let text = format!(r#"{{"name": "{name}", "model": {model}, {rest}}}"#);
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
        })
        .collect()
}

/// A seeded generator in `[-1, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Nanoseconds since the Unix epoch of 2024-01-01T00:00:00.
const T0_NS: i64 = 1_704_067_200_000_000_000;

/// The bank's stream: features `x0`, `x1` (null now and then), a target
/// `y`, a 0/1 target `yb`, a label `lab`, weights `w` with zeros among
/// them, a drifting level `mid`, a number clock `t` in quarter steps with
/// one gap past every cap, the same instants as a `Datetime` in
/// nanoseconds `ts`, and a group key `g` that only grows, blocks of 50.
/// For the integer clock (task 200): the same clock in whole units, `ti =
/// 4t` as `Int64`; a session `s` of 40 rows, and `tu` as `UInt32`, `ti`
/// since the session began, so it steps back at each session change; and
/// `xi`, `x1`'s value null on the last row of each session, so the row
/// before the save is skipped. None of them draws on the generator.
fn frame() -> DataFrame {
    let mut r = lcg(7);
    let (mut t, mut ts, mut g, mut x0, mut x1) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut y, mut yb, mut lab, mut w, mut mid) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut ti, mut tu, mut s, mut xi) = (vec![], vec![], vec![], vec![]);
    let (mut clock, mut level) = (0.0f64, 100.0f64);
    let mut session_start = 0i64;
    for i in 0..N {
        if i > 0 {
            // Quarter steps, exact in nanoseconds; one gap past every cap.
            clock += if i == 260 {
                7200.0
            } else {
                0.25 * (1.0 + (4.0 * (r() + 1.0)).floor())
            };
        }
        t.push(clock);
        ts.push(T0_NS + (clock * 1e9) as i64);
        g.push((i / 50) as i64);
        let (a, b) = (r(), r());
        x0.push((i % 17 != 5).then_some(a));
        x1.push((i % 23 != 9).then_some(b));
        let target = 2.0 * a - b + 0.1 * r();
        y.push(target);
        yb.push(if target > 0.0 { 1.0 } else { 0.0 });
        lab.push(if target > 0.0 { "a" } else { "b" });
        w.push(if i % 13 == 6 { 0.0 } else { 1.0 + 0.5 * r() });
        level += 0.1 * r();
        mid.push(level);
        let whole = (clock * 4.0) as i64;
        if i % 40 == 0 {
            session_start = whole;
        }
        ti.push(whole);
        tu.push((whole - session_start) as u32);
        s.push((i / 40) as i64);
        xi.push((i % 40 != 39).then_some(b));
    }
    assert!(
        xi[SPLIT - 1].is_none(),
        "the row before the save is skipped"
    );
    let ts = Series::new("ts".into(), ts)
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    df!("t" => t, "ts" => ts, "g" => g, "x0" => x0, "x1" => x1, "y" => y, "yb" => yb,
        "lab" => lab, "w" => w, "mid" => mid, "ti" => ti, "tu" => tu, "s" => s, "xi" => xi)
    .unwrap()
}

/// The `refresh_time` stream: three series ticking at their own pace on a
/// number clock, in the long form, `(series, t, v)`.
fn ticks() -> DataFrame {
    let mut r = lcg(11);
    let (mut series, mut t, mut v) = (vec![], vec![], vec![]);
    let mut clock = 0.0;
    for _ in 0..300 {
        clock += 0.5 * (1.0 + (2.0 * (r() + 1.0)).floor());
        let pick = r();
        series.push(if pick < -0.2 {
            "a"
        } else if pick < 0.5 {
            "b"
        } else {
            "c"
        });
        t.push(clock);
        v.push(100.0 + r());
    }
    df!("series" => series, "t" => t, "v" => v).unwrap()
}

fn refresh_cols() -> RefreshCols<'static> {
    RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: None,
        keep: &[],
    }
}

fn refresh_names() -> Vec<String> {
    ["a", "b", "c"].map(str::to_string).to_vec()
}

/// A `with_windows` run over the bank's stream: a backward mean and a mean
/// that looks ahead, so rows are held across the save.
fn windows_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "level", "tree": ["ewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}]},
               {"name": "ahead", "tree": ["-", ["rewm_mean", ["col", "mid"], {"half_life": 5.0, "window_size": 10.0}], ["col", "mid"]]}
             ],
             "clock": "t", "gap_cap": 50.0}"#,
    )
    .unwrap()
}

fn window_input(df: &DataFrame) -> DataFrame {
    df.select(["t", "mid"]).unwrap()
}

/// Every state the hand-off carries, as it stands after the first part of
/// its stream.
struct Saved {
    bank: Bank,
    refresh: RefreshTime,
    windows: WindowsRun,
}

fn first_part() -> Saved {
    let df = frame();
    let mut bank = Bank::new(specs()).unwrap();
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    let mut refresh = RefreshTime::new(refresh_names(), false).unwrap();
    let ticks = ticks();
    refresh
        .feed(&ticks.slice(0, ticks.height() / 2), &refresh_cols())
        .unwrap();
    let input = window_input(&df);
    let mut windows = WindowsRun::new(windows_config(), input.schema()).unwrap();
    windows.feed(&input.slice(0, SPLIT), None).unwrap();
    assert!(
        windows.held() > 0,
        "the window looking ahead holds rows across the save"
    );
    Saved {
        bank,
        refresh,
        windows,
    }
}

const MAGIC: &[u8] = b"polars-online state hand-off\n";

/// The parts, each as its name's length, its name, its length and its bytes.
fn pack(parts: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    for (name, bytes) in parts {
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

fn unpack(file: &[u8]) -> Vec<(String, Vec<u8>)> {
    assert!(file.starts_with(MAGIC), "not a hand-off file");
    fn take<'a>(file: &'a [u8], at: &mut usize, len: usize) -> &'a [u8] {
        let piece = &file[*at..*at + len];
        *at += len;
        piece
    }
    let mut at = MAGIC.len();
    let mut parts = Vec::new();
    while at < file.len() {
        let name_len = u32::from_le_bytes(take(file, &mut at, 4).try_into().unwrap()) as usize;
        let name = String::from_utf8(take(file, &mut at, name_len).to_vec()).unwrap();
        let len = u64::from_le_bytes(take(file, &mut at, 8).try_into().unwrap()) as usize;
        parts.push((name, take(file, &mut at, len).to_vec()));
    }
    parts
}

fn save_all(saved: &Saved) -> Vec<u8> {
    pack(&[
        ("bank", saved.bank.save_bytes().unwrap()),
        ("refresh_time", saved.refresh.save_bytes().unwrap()),
        ("with_windows", saved.windows.save_bytes().unwrap()),
    ])
}

/// What each state gives over the rest of its stream: the bank's output and
/// its closing specs' `closed_groups`, the grid's points, and the windows'
/// rows with what `finish` releases.
struct Rest {
    bank: DataFrame,
    closed: Vec<DataFrame>,
    refresh: DataFrame,
    windows: DataFrame,
}

fn the_rest(mut saved: Saved) -> Rest {
    let df = frame();
    let second = df.slice(SPLIT as i64, N - SPLIT);
    let bank = DataFrame::new(second.height(), saved.bank.fit_predict(&second).unwrap()).unwrap();
    let closed = CLOSING
        .iter()
        .map(|name| {
            let at = SPECS.iter().position(|(n, _, _)| n == name).unwrap();
            saved.bank.closed_groups(Some(at), false).unwrap()
        })
        .collect();
    let ticks = ticks();
    let half = ticks.height() / 2;
    let refresh = saved
        .refresh
        .feed(
            &ticks.slice(half as i64, ticks.height() - half),
            &refresh_cols(),
        )
        .unwrap();
    let input = window_input(&df).slice(SPLIT as i64, N - SPLIT);
    let mut windows = saved.windows.feed(&input, None).unwrap();
    windows
        .vstack_mut(&saved.windows.finish().unwrap())
        .unwrap();
    Rest {
        bank,
        closed,
        refresh,
        windows,
    }
}

/// `file`'s states, loaded as a read leg loads them.
fn load_all(file: &[u8]) -> (Saved, Vec<(String, Vec<u8>)>) {
    let parts = unpack(file);
    let names: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["bank", "refresh_time", "with_windows"]);
    let df = frame();
    let saved = Saved {
        bank: Bank::load_bytes(&parts[0].1, Some(&specs()))
            .unwrap_or_else(|e| panic!("the bank: {e}")),
        refresh: RefreshTime::load_bytes(&parts[1].1, refresh_names(), false, false)
            .unwrap_or_else(|e| panic!("refresh_time: {e}")),
        windows: WindowsRun::load_bytes(&parts[2].1, windows_config(), window_input(&df).schema())
            .unwrap_or_else(|e| panic!("with_windows: {e}")),
    };
    (saved, parts)
}

/// Every column of `got` against `want`: the nulls in the same rows, the
/// numbers within `tol` of each other relative to `1 + |want|`, and every
/// other value equal.
fn same(what: &str, got: &DataFrame, want: &DataFrame, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    for (g, w) in got.columns().iter().zip(want.columns()) {
        same_series(
            &format!("{what}.{}", w.name()),
            g.as_materialized_series(),
            w.as_materialized_series(),
            tol,
        );
    }
}

fn same_series(what: &str, got: &Series, want: &Series, tol: f64) {
    assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
    assert_eq!(got.len(), want.len(), "{what}: length");
    match want.dtype() {
        DataType::Struct(_) => {
            let (g, w) = (got.struct_().unwrap(), want.struct_().unwrap());
            for (gf, wf) in g.fields_as_series().iter().zip(w.fields_as_series()) {
                same_series(&format!("{what}.{}", wf.name()), gf, &wf, tol);
            }
        }
        DataType::List(inner) if **inner == DataType::Float64 => {
            let (g, w) = (got.list().unwrap(), want.list().unwrap());
            for i in 0..want.len() {
                match (g.get_as_series(i), w.get_as_series(i)) {
                    (Some(a), Some(b)) => same_series(&format!("{what}[{i}]"), &a, &b, tol),
                    (None, None) => {}
                    (a, b) => panic!("{what}, row {i}: {a:?} against {b:?}"),
                }
            }
        }
        DataType::Float64 => {
            let (g, w) = (got.f64().unwrap(), want.f64().unwrap());
            for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                let ok = match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) if a.is_nan() || b.is_nan() => a.is_nan() && b.is_nan(),
                    (Some(a), Some(b)) => (a - b).abs() <= tol * (1.0 + b.abs()),
                    _ => false,
                };
                assert!(ok, "{what}, row {i}: {a:?} against {b:?}");
            }
        }
        _ => assert!(got.equals_missing(want), "{what}: {got:?} against {want:?}"),
    }
}

fn same_rest(what: &str, got: &Rest, want: &Rest, tol: f64) {
    same(&format!("{what}: the bank"), &got.bank, &want.bank, tol);
    for ((name, g), w) in CLOSING.iter().zip(&got.closed).zip(&want.closed) {
        assert!(w.height() > 0, "{name}: a group closes in the second part");
        same(&format!("{what}: {name}'s closed groups"), g, w, tol);
    }
    assert!(want.refresh.height() > 10, "the grid has points to compare");
    same(
        &format!("{what}: refresh_time"),
        &got.refresh,
        &want.refresh,
        tol,
    );
    same(
        &format!("{what}: with_windows"),
        &got.windows,
        &want.windows,
        tol,
    );
}

/// A state written now loads now and goes on as the run that never stopped,
/// to the bit: the code path the cross-OS hand-off runs, with only the
/// writer differing. And a loaded state saves the bytes it was loaded from.
#[test]
fn state_round_trips_through_bytes() {
    let file = save_all(&first_part());
    let (loaded, parts) = load_all(&file);
    for ((name, bytes), again) in parts.iter().zip(unpack(&save_all(&loaded))) {
        assert!(
            *bytes == again.1,
            "{name}: a loaded state re-saves other bytes"
        );
    }
    same_rest(
        "in process",
        &the_rest(loaded),
        &the_rest(first_part()),
        0.0,
    );
}

/// Every state saves the same bytes for the same input, which is what makes
/// the artifact hand-off between OSes meaningful.
#[test]
fn state_bytes_are_deterministic() {
    let (a, b) = (save_all(&first_part()), save_all(&first_part()));
    for ((name, x), (_, y)) in unpack(&a).iter().zip(unpack(&b)) {
        assert!(*x == y, "{name}: two runs of one input saved other bytes");
    }
}

/// When CI hands over a file written on the other OS
/// (`ONLINE_FOREIGN_STATE`), load each state and go on: the output is the
/// run that never left this machine, to the golden pipeline's tolerance,
/// and each state re-saves the bytes it came in.
#[test]
fn loads_a_state_written_on_another_os() {
    let Some(path) = std::env::var_os("ONLINE_FOREIGN_STATE") else {
        eprintln!("ONLINE_FOREIGN_STATE not set; skipping the cross-OS hand-off");
        return;
    };
    let path = PathBuf::from(path);
    let file = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let (loaded, parts) = load_all(&file);
    for ((name, bytes), again) in parts.iter().zip(unpack(&save_all(&loaded))) {
        assert!(
            *bytes == again.1,
            "{name}: a state written on another OS re-saves other bytes here"
        );
    }
    same_rest(
        "a state written on another OS",
        &the_rest(loaded),
        &the_rest(first_part()),
        TOL,
    );
}

/// Writes the file CI hands to the other OS. Run with
/// `ONLINE_WRITE_STATE=<path> cargo test -p online-polars --test state_portability`.
#[test]
fn writes_the_handoff_state_when_asked() {
    let Some(path) = std::env::var_os("ONLINE_WRITE_STATE") else {
        return;
    };
    std::fs::write(PathBuf::from(path), save_all(&first_part())).unwrap();
}
