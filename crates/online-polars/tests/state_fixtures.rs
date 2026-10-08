//! Frozen state fixtures for the files a bank and its helpers write: a bank
//! file (several specs, a group closed and drained, rows held under an
//! `embargo`, a temporal clock, a window), a `with_windows` state holding
//! rows for a window that looks ahead, and a `refresh_time` state part-way
//! through an interval -- each embedded as its bytes beside the input that
//! follows it and the frames that input gave (docs/PLAN.md task 198; review
//! round 4, D1). The models' own states are `online-core`'s
//! `tests/state_fixtures.rs`, on the same three checks:
//!
//! 1. **it loads**;
//! 2. **it goes on to the bit**: the next input gives the frames it gave
//!    when the fixture was written -- to the bit on the platform that wrote
//!    it, to the golden pipeline's tolerance elsewhere (libm's last bit
//!    differs by platform); a null is a null, and every other value equal;
//! 3. **it saves its bytes again**.
//!
//! The inputs and outputs are frozen as Arrow IPC (a format polars reads
//! back however it writes), in hex, as the states are: no binary file
//! (CLAUDE.md hard rule 1). Regenerate with
//!
//! ```text
//! PRINT_STATE_FIXTURES=1 cargo test -p online-polars --test state_fixtures
//! ```
//!
//! before 1.0, whenever the bank's schema, the windows state's version or
//! the `refresh_time` state's moves, or a number does; the bank's minimum
//! schema moves with the schema until 1.0 and stays at 1.0's after, as
//! `online-core`'s does. The windows state and the formula tree are
//! labelled unstable (docs/PLAN.md task 198, D5): a change to either
//! regenerates its fixture rather than keeping it with a loader.

#[path = "state_fixtures/index.rs"]
mod frozen;

use std::fmt::Write as _;
use std::io::Cursor;
use std::path::PathBuf;

use online_polars::online_core::SCHEMA_VERSION;
use online_polars::{
    Bank, MIN_BANK_SCHEMA_VERSION, RefreshCols, RefreshTime, Spec, WindowsConfig, WindowsRun,
};
use polars::prelude::*;

/// One frozen fixture, as `tests/state_fixtures/<name>.rs` holds it.
pub struct Fixture {
    pub name: &'static str,
    /// The schema of the build that wrote it (`online_core::SCHEMA_VERSION`);
    /// a helper's state carries a version of its own in its bytes, which
    /// its loader checks.
    pub schema: u32,
    pub writer: &'static str,
    pub state: &'static str,
    /// The input the continuation is fed, as Arrow IPC in hex.
    pub input: &'static str,
    /// What it gave, each frame named, as Arrow IPC in hex.
    pub output: &'static [(&'static str, &'static str)],
}

const REGENERATE: &str = "PRINT_STATE_FIXTURES=1 cargo test -p online-polars --test state_fixtures";

/// The schemas before the current one whose bank fixtures are kept: none
/// before 1.0.
const PREVIOUS: &[u32] = &[];

/// The golden pipeline's tolerance, off the writer's platform.
const TOL: f64 = 1e-12;

fn regenerating() -> bool {
    std::env::var("PRINT_STATE_FIXTURES").is_ok_and(|v| v == "1")
}

fn writer() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    s.chunks(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}

fn ipc(df: &DataFrame) -> Vec<u8> {
    let mut out = Vec::new();
    IpcWriter::new(&mut out)
        .finish(&mut df.clone())
        .expect("a frame writes as IPC");
    out
}

fn from_ipc(hex_text: &str) -> DataFrame {
    IpcReader::new(Cursor::new(unhex(hex_text)))
        .finish()
        .expect("a frozen frame reads back")
}

// --- the streams ------------------------------------------------------------

/// The bank's specs: a temporal clock with duration parameters and a weight,
/// rows held under an `embargo`, groups closed as the key grows, a window,
/// and the integer clock's forms (task 200; review round 5, B1): an `Int64`
/// clock under a fractional `gap_cap` and `embargo` (rows held, each with
/// an integer stamp), saved right after a skipped row, so the removed and
/// pending ticks hold time and are written; a `UInt32`
/// clock that starts over at each session, under a `session_gap`, for the
/// width and the elapsed clock's removed ticks; and a group closed on
/// session, whose PCA continuity is kept per group (`pca_prev_by_group`).
fn specs() -> Vec<Spec> {
    [
        r#"{"name": "dated", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "ts", "half_life": "10m", "gap_cap": "1h", "weight": "w"}"#,
        r#"{"name": "embargoed", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"], "clock": "t", "half_life": 40.0, "gap_cap": 50.0, "embargo": 3.0}"#,
        r#"{"name": "closing", "model": {"type": "ew_cov"}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "group": "g", "group_close": "monotone"}"#,
        r#"{"name": "windowed", "model": {"type": "ew_cov", "window_size": 12.0}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0}"#,
        r#"{"name": "int_clock", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "xi"], "clock": "ti", "half_life": 160.0, "gap_cap": 6.5, "embargo": 12.5, "weight": "w"}"#,
        r#"{"name": "uint_clock", "model": {"type": "ew_cov"}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "tu", "half_life": 80.0, "gap_cap": 6.5, "session": "s", "session_gap": 3.5}"#,
        r#"{"name": "sessioned", "model": {"type": "ew_cov", "pca": 1}, "targets": ["x0"], "features": ["x0", "x1"], "clock": "t", "half_life": 20.0, "gap_cap": 50.0, "group": "g", "session": "s", "group_close": "session"}"#,
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap_or_else(|e| panic!("{text}: {e}")))
    .collect()
}

/// The spec whose closed groups the continuation drains.
const CLOSING: usize = 2;

/// Rows of the bank's stream, and the row it is saved at.
const N: usize = 120;
const SPLIT: usize = 80;

/// A seeded generator in `[-1, 1)`, integer arithmetic alone.
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

/// Features `x0`, `x1` (null now and then), a target `y`, weights `w` with
/// zeros among them, a drifting level `mid`, a number clock `t` in quarter
/// steps, the same instants as a `Datetime` `ts`, and a group key `g` that
/// only grows, blocks of 25. Beside them, for the integer clock (task 200):
/// the same clock in whole units, `ti = 4t` as `Int64`; a session `s` of 20
/// rows, and `tu` as `UInt32`, `ti` since the session began, so it steps
/// back at each session change; `xi`, `x1`'s value null on the last row of
/// each session, so the row before the save ([`SPLIT`]) is skipped; and a
/// running count `cum`, an integer input to `increment`. None of them
/// draws on the generator, so the other columns are what they were.
fn frame() -> DataFrame {
    let mut r = lcg(198);
    let (mut t, mut ts, mut g, mut x0, mut x1) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut y, mut w, mut mid) = (vec![], vec![], vec![]);
    let (mut ti, mut tu, mut s, mut xi, mut cum) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut clock, mut level) = (0.0f64, 100.0f64);
    let (mut session_start, mut count) = (0i64, 0i64);
    for i in 0..N {
        if i > 0 {
            clock += 0.25 * (1.0 + (4.0 * (r() + 1.0)).floor());
        }
        t.push(clock);
        ts.push(T0_NS + (clock * 1e9) as i64);
        g.push((i / 25) as i64);
        let (a, b) = (r(), r());
        x0.push((i % 17 != 5).then_some(a));
        x1.push((i % 23 != 9).then_some(b));
        y.push(2.0 * a - b + 0.1 * r());
        w.push(if i % 13 == 6 { 0.0 } else { 1.0 + 0.5 * r() });
        level += 0.1 * r();
        mid.push(level);
        let whole = (clock * 4.0) as i64;
        if i % 20 == 0 {
            session_start = whole;
        }
        ti.push(whole);
        tu.push((whole - session_start) as u32);
        s.push((i / 20) as i64);
        xi.push((i % 20 != 19).then_some(b));
        count += 1 + (i % 3) as i64;
        cum.push(count);
    }
    assert!(
        xi[SPLIT - 1].is_none(),
        "the row before the save is skipped, so the pending ticks hold its step"
    );
    let ts = Series::new("ts".into(), ts)
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    df!("t" => t, "ts" => ts, "g" => g, "x0" => x0, "x1" => x1, "y" => y, "w" => w, "mid" => mid,
        "ti" => ti, "tu" => tu, "s" => s, "xi" => xi, "cum" => cum)
    .unwrap()
}

/// Three series ticking at their own pace on a number clock, long form.
fn ticks() -> DataFrame {
    let mut r = lcg(11);
    let (mut series, mut t, mut v) = (vec![], vec![], vec![]);
    let mut clock = 0.0;
    for _ in 0..120 {
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

/// The same ticks on an integer clock: `t` doubled, as `Int64` (task 200).
fn ticks_int() -> DataFrame {
    let mut df = ticks();
    let t = (df.column("t").unwrap().as_materialized_series() * 2.0)
        .cast(&DataType::Int64)
        .unwrap();
    df.with_column(t.into()).unwrap();
    df
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

/// A backward mean and a mean that looks ahead, so rows are held across the
/// save.
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

/// The same two means on the `Int64` clock, under a fractional `gap_cap`,
/// and an `increment` of an integer input (task 200; review round 5, B1):
/// the rows' `off` in the integer form, and the increment's previous value
/// kept as an integer.
fn windows_int_config() -> WindowsConfig {
    serde_json::from_str(
        r#"{"formulas": [
               {"name": "level", "tree": ["ewm_mean", ["col", "mid"], {"half_life": 20.0, "window_size": 40.0}]},
               {"name": "ahead", "tree": ["-", ["rewm_mean", ["col", "mid"], {"half_life": 20.0, "window_size": 40.0}], ["col", "mid"]]},
               {"name": "step", "tree": ["increment", ["col", "cum"]]}
             ],
             "clock": "ti", "gap_cap": 200.5}"#,
    )
    .unwrap()
}

fn window_int_input(df: &DataFrame) -> DataFrame {
    df.select(["ti", "mid", "cum"]).unwrap()
}

// --- the three kinds --------------------------------------------------------

/// A state loaded and continued: the bytes it saved before it went on, and
/// the frames the input gave, named; or why it did not load.
type Ran = Result<(Vec<u8>, Vec<(String, DataFrame)>), String>;

/// What a kind's state is: how it is made, loaded, saved and continued.
struct Kind {
    name: &'static str,
    /// The state after the first part, and the input of the continuation.
    first: fn() -> (Vec<u8>, DataFrame),
    /// The state loaded from bytes, continued on the input: the frames it
    /// gave, named, and the bytes the loaded state saved before it went on.
    run: fn(&[u8], &DataFrame) -> Ran,
}

fn bank_first() -> (Vec<u8>, DataFrame) {
    let df = frame();
    let mut bank = Bank::new(specs()).unwrap();
    bank.fit_predict(&df.slice(0, SPLIT)).unwrap();
    // A group closed before the save is drained, and the next is open
    // across it.
    let drained = bank.closed_groups(Some(CLOSING), true).unwrap();
    assert!(drained.height() > 0, "a group closed before the save");
    (
        bank.save_bytes().unwrap(),
        df.slice(SPLIT as i64, N - SPLIT),
    )
}

fn bank_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut bank = Bank::load_bytes(bytes, Some(&specs()))?;
    let again = bank.save_bytes()?;
    let cols = bank.fit_predict(input).map_err(|e| e.to_string())?;
    let out = DataFrame::new(input.height(), cols).map_err(|e| e.to_string())?;
    let closed = bank.closed_groups(Some(CLOSING), false)?;
    Ok((
        again,
        vec![
            ("fit_predict".into(), out),
            ("closed_groups".into(), closed),
        ],
    ))
}

fn windows_first_of(config: WindowsConfig, input: DataFrame) -> (Vec<u8>, DataFrame) {
    let mut run = WindowsRun::new(config, input.schema()).unwrap();
    run.feed(&input.slice(0, SPLIT), None).unwrap();
    assert!(run.held() > 0, "rows held across the save");
    (
        run.save_bytes().unwrap(),
        input.slice(SPLIT as i64, N - SPLIT),
    )
}

fn windows_run_of(config: WindowsConfig, bytes: &[u8], input: &DataFrame) -> Ran {
    let mut run = WindowsRun::load_bytes(bytes, config, input.schema())?;
    let again = run.save_bytes()?;
    let fed = run.feed(input, None).map_err(|e| e.to_string())?;
    let finished = run.finish().map_err(|e| e.to_string())?;
    Ok((
        again,
        vec![("feed".into(), fed), ("finish".into(), finished)],
    ))
}

fn windows_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_config(), window_input(&frame()))
}

fn windows_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_config(), bytes, input)
}

fn windows_int_first() -> (Vec<u8>, DataFrame) {
    windows_first_of(windows_int_config(), window_int_input(&frame()))
}

fn windows_int_run(bytes: &[u8], input: &DataFrame) -> Ran {
    windows_run_of(windows_int_config(), bytes, input)
}

fn refresh_first_of(ticks: DataFrame) -> (Vec<u8>, DataFrame) {
    let half = ticks.height() / 2;
    let mut r = RefreshTime::new(refresh_names(), false).unwrap();
    r.feed(&ticks.slice(0, half), &refresh_cols()).unwrap();
    (
        r.save_bytes().unwrap(),
        ticks.slice(half as i64, ticks.height() - half),
    )
}

fn refresh_run(bytes: &[u8], input: &DataFrame) -> Ran {
    let mut r = RefreshTime::load_bytes(bytes, refresh_names(), false, false)?;
    let again = r.save_bytes()?;
    let fed = r.feed(input, &refresh_cols()).map_err(|e| e.to_string())?;
    Ok((again, vec![("feed".into(), fed)]))
}

fn refresh_first() -> (Vec<u8>, DataFrame) {
    refresh_first_of(ticks())
}

fn refresh_int_first() -> (Vec<u8>, DataFrame) {
    refresh_first_of(ticks_int())
}

fn kinds() -> Vec<Kind> {
    vec![
        Kind {
            name: "bank",
            first: bank_first,
            run: bank_run,
        },
        Kind {
            name: "with_windows",
            first: windows_first,
            run: windows_run,
        },
        Kind {
            name: "with_windows_int",
            first: windows_int_first,
            run: windows_int_run,
        },
        Kind {
            name: "refresh_time",
            first: refresh_first,
            run: refresh_run,
        },
        Kind {
            name: "refresh_time_int",
            first: refresh_int_first,
            run: refresh_run,
        },
    ]
}

fn kind_of(f: &Fixture) -> Kind {
    kinds()
        .into_iter()
        .find(|k| k.name == f.name)
        .unwrap_or_else(|| panic!("{}: no such kind; regenerate with `{REGENERATE}`", f.name))
}

// --- comparison -------------------------------------------------------------

/// Every column of `got` against `want`: the nulls in the same rows, the
/// numbers within `tol` of each other relative to `1 + |want|` (the same
/// bits at 0; a NaN on both sides agrees), and every other value equal.
fn same(what: &str, got: &DataFrame, want: &DataFrame, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    for (g, w) in got.columns().iter().zip(want.columns()) {
        assert_eq!(g.name(), w.name(), "{what}: column names");
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
                    (Some(a), Some(b)) if tol == 0.0 => a.to_bits() == b.to_bits(),
                    (Some(a), Some(b)) => (a - b).abs() <= tol * (1.0 + b.abs()),
                    _ => false,
                };
                assert!(ok, "{what}, row {i}: got {a:?}, frozen {b:?}");
            }
        }
        _ => assert!(got.equals_missing(want), "{what}: {got:?} against {want:?}"),
    }
}

// --- the checks -------------------------------------------------------------

#[test]
fn every_fixture_loads_goes_on_to_the_bit_and_saves_its_bytes_again() {
    if regenerating() {
        return;
    }
    assert_eq!(
        frozen::ALL.iter().map(|f| f.name).collect::<Vec<_>>(),
        kinds().iter().map(|k| k.name).collect::<Vec<_>>(),
        "the frozen fixtures are not the kinds: regenerate with `{REGENERATE}`"
    );
    for f in frozen::ALL {
        let kind = kind_of(f);
        let bytes = unhex(f.state);
        let input = from_ipc(f.input);
        let (again, frames) = (kind.run)(&bytes, &input).unwrap_or_else(|e| {
            panic!(
                "{}: the frozen state does not load: {e}\n(a layout or version change ships \
                 its fixtures: before 1.0, regenerate with `{REGENERATE}`)",
                f.name
            )
        });
        assert!(
            again == bytes,
            "{}: the loaded state saves other bytes",
            f.name
        );
        let tol = if f.writer == writer() { 0.0 } else { TOL };
        assert_eq!(frames.len(), f.output.len(), "{}", f.name);
        for ((name, got), (frozen_name, want)) in frames.iter().zip(f.output) {
            assert_eq!(name, frozen_name, "{}", f.name);
            let want = from_ipc(want);
            // Every frame has rows to compare: a group closes in the
            // continuation, the held rows are released, the grid has points.
            assert!(want.height() > 0, "{}: {name} is empty", f.name);
            same(&format!("{}: {name}", f.name), got, &want, tol);
        }
    }
}

/// Every schema a bank file loads from has its fixtures: the bank's minimum
/// cannot sit below the oldest set kept, nor the schema move without one.
#[test]
fn the_bank_fixtures_cover_the_schemas_a_bank_loads() {
    if regenerating() {
        return;
    }
    let bank = frozen::ALL
        .iter()
        .find(|f| f.name == "bank")
        .expect("a bank fixture");
    assert_eq!(
        bank.schema, SCHEMA_VERSION,
        "SCHEMA_VERSION is {SCHEMA_VERSION} and the frozen bank file is of {}: before 1.0, \
         regenerate with `{REGENERATE}` and raise MIN_BANK_SCHEMA_VERSION with it",
        bank.schema
    );
    let covered: Vec<u32> = PREVIOUS.iter().copied().chain([bank.schema]).collect();
    let claimed: Vec<u32> = (MIN_BANK_SCHEMA_VERSION..=SCHEMA_VERSION).collect();
    assert_eq!(
        claimed, covered,
        "a bank loads schemas {MIN_BANK_SCHEMA_VERSION}..={SCHEMA_VERSION} and fixtures cover \
         {covered:?}: each needs a frozen file, held to its loader"
    );
}

/// Every leaf of a msgpack value: its dotted path and its value.
fn leaves(v: &rmpv::Value, path: &str, out: &mut Vec<(String, rmpv::Value)>) {
    match v {
        rmpv::Value::Map(m) => {
            for (k, v) in m {
                let key = k.as_str().map_or_else(|| k.to_string(), str::to_string);
                leaves(v, &format!("{path}.{key}"), out);
            }
        }
        rmpv::Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                leaves(v, &format!("{path}.{i}"), out);
            }
        }
        _ => out.push((path.to_string(), v.clone())),
    }
}

/// The integer clock's forms (docs/PLAN.md task 200, schema 42 and windows
/// state 8) are each written by a fixture, so a change to their tags or
/// layouts cannot pass the harness unseen (review round 5, B1): the bank's
/// `ClockValue::I64`, `Stamp::Int`, the three `Ticks` fields -- skipped
/// when zero, so written only by a state saved while they hold time -- a
/// fraction in one of them, `ClockDtype::Int` at two widths, and the
/// per-group PCA continuity; the windows state's `OffForm::Int`, its
/// group's stamp as a `ClockValue::I64` and an increment's integer previous
/// value; the refresh state's `Instant::Int`.
#[test]
fn every_form_of_the_integer_clock_is_written_by_a_fixture() {
    if regenerating() {
        return;
    }
    type Holds = fn(&str, &rmpv::Value) -> bool;
    const FORMS: &[(&str, &str, Holds)] = &[
        ("bank", "ClockValue::I64", |p, _| {
            p.ends_with(".prev_clock.I64")
        }),
        ("bank", "Stamp::Int", |p, _| p.contains(".stamp.Int.")),
        ("bank", "Ticks: removed_int", |p, _| {
            p.ends_with(".removed_int.whole")
        }),
        ("bank", "Ticks: elapsed_removed_int", |p, _| {
            p.ends_with(".elapsed_removed_int.whole")
        }),
        ("bank", "Ticks: pending_int", |p, _| {
            p.ends_with(".pending_int.whole")
        }),
        ("bank", "a Ticks fraction", |p, v| {
            p.ends_with("_int.frac") && v.as_f64().is_some_and(|f| f != 0.0)
        }),
        ("bank", "ClockDtype::Int(I64)", |p, v| {
            p.contains(".clock_dtypes.") && p.ends_with(".Int") && v.as_str() == Some("I64")
        }),
        ("bank", "ClockDtype::Int(U32)", |p, v| {
            p.contains(".clock_dtypes.") && p.ends_with(".Int") && v.as_str() == Some("U32")
        }),
        ("bank", "pca_prev_by_group", |p, _| {
            p.starts_with(".pca_prev_by_group.")
        }),
        ("with_windows_int", "OffForm::Int", |p, v| {
            p.ends_with(".form") && v.as_str() == Some("Int")
        }),
        ("with_windows_int", "ClockValue::I64", |p, _| {
            p.ends_with(".prev_clock.I64")
        }),
        (
            "with_windows_int",
            "the group's stamp as ClockValue::I64",
            |p, _| p.ends_with(".stamp.I64"),
        ),
        // An `i128` rides as sixteen bytes, so the leaf is a binary, not an
        // integer: present is what is checked.
        ("with_windows_int", "an increment's prev_int", |p, v| {
            p.contains(".prev_int.") && !v.is_nil()
        }),
        ("refresh_time_int", "Instant::Int", |p, _| {
            p.contains(".last_time.") && p.ends_with(".Int")
        }),
    ];
    for (kind, form, holds) in FORMS {
        let f = frozen::ALL
            .iter()
            .find(|f| f.name == *kind)
            .unwrap_or_else(|| panic!("{kind}: no fixture; add the kind and regenerate"));
        let v = rmpv::decode::read_value(&mut unhex(f.state).as_slice()).unwrap();
        let mut all = Vec::new();
        leaves(&v, "", &mut all);
        assert!(
            all.iter().any(|(p, v)| holds(p, v)),
            "{kind}: no leaf of the state holds {form}, so the fixture freezes nothing of it"
        );
    }
}

// --- regeneration -----------------------------------------------------------

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/state_fixtures")
}

/// `text` as a Rust string literal of hex, wrapped.
fn hex_literal(out: &mut String, indent: &str, text: &str) {
    writeln!(out, "\"\\").unwrap();
    let chunks: Vec<&[u8]> = text.as_bytes().chunks(96).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let tail = if i + 1 == chunks.len() { "\"" } else { "\\" };
        writeln!(out, "{indent}{}{tail}", std::str::from_utf8(chunk).unwrap()).unwrap();
    }
}

fn file_of(kind: &Kind) -> String {
    let (bytes, input) = (kind.first)();
    let (again, frames) =
        (kind.run)(&bytes, &input).unwrap_or_else(|e| panic!("{}: {e}", kind.name));
    assert!(again == bytes, "{}: re-saves other bytes", kind.name);
    // Two runs of the first part write the same bytes, which is what makes
    // a frozen file comparable at all.
    assert!(
        (kind.first)().0 == bytes,
        "{}: two runs saved other bytes",
        kind.name
    );
    let mut out = format!(
        "// @generated by `{REGENERATE}`: do not edit by hand.\n\
         pub const FIXTURE: crate::Fixture = crate::Fixture {{\n    name: {:?},\n    schema: {SCHEMA_VERSION},\n    writer: {:?},\n    state: ",
        kind.name,
        writer()
    );
    hex_literal(&mut out, "        ", &hex(&bytes));
    out.push_str("    ,\n    input: ");
    hex_literal(&mut out, "        ", &hex(&ipc(&input)));
    out.push_str("    ,\n    output: &[\n");
    for (name, df) in &frames {
        write!(out, "        (\n            {name:?},\n            ").unwrap();
        hex_literal(&mut out, "            ", &hex(&ipc(df)));
        out.push_str("        ),\n");
    }
    out.push_str("    ],\n};\n");
    out
}

/// Rewrites every fixture, and the index, when `PRINT_STATE_FIXTURES=1`.
#[test]
fn regenerate_when_asked() {
    if !regenerating() {
        return;
    }
    let dir = dir();
    std::fs::create_dir_all(&dir).unwrap();
    let mut index = format!(
        "// @generated by `{REGENERATE}`: do not edit by hand.\n\
         // The frozen fixtures of a bank file and its helpers' states.\n\n"
    );
    let kinds = kinds();
    for kind in &kinds {
        let text = file_of(kind);
        std::fs::write(dir.join(format!("{}.rs", kind.name)), &text).unwrap();
        writeln!(
            index,
            "pub mod {} {{\n    include!(\"{}.rs\");\n}}",
            kind.name, kind.name
        )
        .unwrap();
        println!("wrote {}.rs ({} bytes)", kind.name, text.len());
    }
    writeln!(
        index,
        "\n/// Every fixture.\npub const ALL: &[&crate::Fixture] = &["
    )
    .unwrap();
    for kind in &kinds {
        writeln!(index, "    &{}::FIXTURE,", kind.name).unwrap();
    }
    writeln!(index, "];").unwrap();
    std::fs::write(dir.join("index.rs"), index).unwrap();
}
