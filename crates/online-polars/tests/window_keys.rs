//! Group keys and session columns of every dtype give the windows one answer
//! (docs/PLAN.md task 431).
//!
//! The window core reads a chunk's group keys from the column's own values
//! (an integer, a string, a category) and its session column's the same way,
//! where it once cast each to text and hashed the text every chunk. A group is
//! named by its key's text and made at the first row that reaches it, so the
//! same stream keyed by integers, by their text, by a categorical or an enum of
//! it, or by a float that prints distinctly, must give the same outputs to the
//! bit, whatever its chunking, with a save and a load in the middle, with nulls
//! among the keys, and with 1, 7, 100 and 2,000 keys.
use online_polars::{WindowsConfig, WindowsRun};
use polars::prelude::*;
use serde_json::{Value, json};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Int64,
    Int32,
    UInt8,
    UInt64,
    Str,
    Cat,
    Enum,
    Float,
}

const KINDS: [Kind; 8] = [
    Kind::Int64,
    Kind::Int32,
    Kind::UInt8,
    Kind::UInt64,
    Kind::Str,
    Kind::Cat,
    Kind::Enum,
    Kind::Float,
];

struct Data {
    ts: Vec<i64>,
    mid: Vec<f64>,
    /// Per row: the key's number, `None` for a null key.
    key: Vec<Option<i64>>,
    /// Per row: the session's number, `None` for a null session.
    sess: Vec<Option<i64>>,
}

fn data(n: usize, keys: usize, null_keys: bool, run: usize) -> Data {
    let mut r = Rng(0x9E3779B97F4A7C15);
    let (mut t, mut m) = (0.0f64, 100.0f64);
    let mut d = Data {
        ts: vec![],
        mid: vec![],
        key: vec![],
        sess: vec![],
    };
    let mut cur = 0i64;
    for i in 0..n {
        t += -0.5 * r.next().ln();
        m += (r.next() - 0.5) * 0.02;
        d.ts.push((t * 1e9) as i64);
        d.mid.push(m);
        if i % run == 0 {
            cur = (r.next() * keys as f64) as i64;
        }
        d.key.push(if null_keys && i % 41 == 40 {
            None
        } else {
            Some(cur)
        });
        d.sess.push(if i % 97 == 96 {
            None
        } else {
            Some((i / 700) as i64 - 3)
        });
    }
    d
}

/// A column of numbers as the dtype of a kind.
fn column(name: &str, v: &[Option<i64>], kind: Kind, domain: usize) -> Column {
    let ints = Column::new(name.into(), v);
    match kind {
        Kind::Int64 => ints,
        Kind::Int32 => ints.cast(&DataType::Int32).unwrap(),
        Kind::UInt8 => ints
            .cast(&DataType::Int64)
            .unwrap()
            .cast(&DataType::UInt8)
            .unwrap(),
        Kind::UInt64 => ints
            .cast(&DataType::Int64)
            .unwrap()
            .cast(&DataType::UInt64)
            .unwrap(),
        Kind::Str => ints.cast(&DataType::String).unwrap(),
        Kind::Cat => ints
            .cast(&DataType::String)
            .unwrap()
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap(),
        Kind::Enum => {
            let names: Vec<String> = (-3..domain as i64).map(|k| k.to_string()).collect();
            let frozen = FrozenCategories::new(names.iter().map(String::as_str)).unwrap();
            ints.cast(&DataType::String)
                .unwrap()
                .cast(&DataType::from_frozen_categories(frozen))
                .unwrap()
        }
        Kind::Float => ints.cast(&DataType::Float64).unwrap(),
    }
}

fn frame(
    d: &Data,
    from: usize,
    to: usize,
    group: Option<Kind>,
    sess: Option<Kind>,
    domain: usize,
) -> DataFrame {
    let mut cols = vec![
        Column::new("ts".into(), &d.ts[from..to])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap(),
        Column::new("mid".into(), &d.mid[from..to]),
    ];
    if let Some(k) = group {
        cols.push(column("g", &d.key[from..to], k, domain));
    }
    if let Some(k) = sess {
        cols.push(column("s", &d.sess[from..to], k, domain));
    }
    DataFrame::new_infer_height(cols).unwrap()
}

/// What the operators of a stream look at: the rows so far, a window of them,
/// or the rows ahead in a window (`rewm_*`, task 441: a forward kernel is
/// resolved by a later row, and a variance has none).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Running,
    Windowed,
    Forward,
}

fn config(mode: Mode, group: bool, session: bool) -> WindowsConfig {
    config_with(mode, group, session, "5m")
}

/// [`config`] with a `gap_cap`: a key whose rows are further apart than the cap
/// starts its windows over at each of them.
fn config_with(mode: Mode, group: bool, session: bool, cap: &str) -> WindowsConfig {
    let op = |name: &str| {
        let mut o = json!({"half_life": "5s"});
        if mode != Mode::Running {
            o["window_size"] = json!("30s");
        }
        json!({"name": name, "tree": [name, ["col", "mid"], o]})
    };
    let formulas = if mode == Mode::Forward {
        vec![op("rewm_mean"), op("rewm_sum"), op("rewm_rate")]
    } else {
        vec![op("ewm_mean"), op("ewm_std"), op("ewm_sum")]
    };
    let mut c = json!({
        "formulas": formulas,
        "clock": "ts",
        "gap_cap": cap,
    });
    if group {
        c["group"] = json!("g");
    }
    if session {
        c["session"] = json!("s");
    }
    serde_json::from_value::<Value>(c)
        .map(|v| serde_json::from_value(v).unwrap())
        .unwrap()
}

/// Every float output's bits and null mask, and the state at the end.
fn digest(
    d: &Data,
    cfg: &WindowsConfig,
    chunks: &[usize],
    reload: bool,
    group: Option<Kind>,
    sess: Option<Kind>,
    domain: usize,
) -> (Vec<u64>, Vec<u8>) {
    let n = d.ts.len();
    let schema = frame(d, 0, 1, group, sess, domain).schema().clone();
    let mut run = WindowsRun::new(cfg.clone(), &schema).unwrap();
    // Per output column, its rows in order, whatever the chunking.
    let mut cols: Vec<Vec<u64>> = vec![];
    let mut take = |f: &DataFrame| {
        let floats: Vec<&Float64Chunked> = f
            .columns()
            .iter()
            .filter(|c| c.name().contains("ewm_"))
            .filter_map(|c| c.f64().ok())
            .collect();
        cols.resize(floats.len(), vec![]);
        for (col, ca) in cols.iter_mut().zip(floats) {
            col.extend(ca.iter().map(|v| v.map_or(0x7ff8_dead_beef, f64::to_bits)));
        }
    };
    let mut at = 0;
    for (k, &len) in chunks.iter().cycle().enumerate() {
        let to = (at + len).min(n);
        if at >= to {
            break;
        }
        take(
            &run.feed(&frame(d, at, to, group, sess, domain), None)
                .unwrap(),
        );
        at = to;
        if reload && k == 1 {
            let bytes = run.save_bytes().unwrap();
            run = WindowsRun::load_bytes(&bytes, cfg.clone(), &schema).unwrap();
            assert_eq!(
                run.save_bytes().unwrap(),
                bytes,
                "a loaded run re-saves its bytes"
            );
        }
    }
    take(&run.finish().unwrap());
    (cols.concat(), run.save_bytes().unwrap())
}

fn check(keys: usize, null_keys: bool, run: usize, mode: Mode) {
    check_with(5000, keys, null_keys, run, mode, "5m", &KINDS);
}

fn check_with(
    n: usize,
    keys: usize,
    null_keys: bool,
    run: usize,
    mode: Mode,
    cap: &str,
    kinds: &[Kind],
) {
    let d = data(n, keys, null_keys, run);
    let cfg = config_with(mode, true, false, cap);
    let (want, _) = digest(&d, &cfg, &[n], false, Some(Kind::Int64), None, keys);
    for &kind in kinds {
        if kind == Kind::UInt8 && keys > 200 {
            continue;
        }
        for chunks in [vec![n], vec![1000], vec![333, 1, 700, 2000], vec![7]] {
            for reload in [false, true] {
                if chunks == [7] && keys > 100 {
                    continue;
                }
                let (got, _) = digest(&d, &cfg, &chunks, reload, Some(kind), None, keys);
                if got != want {
                    let at = got.iter().zip(&want).position(|(a, b)| a != b);
                    eprintln!(
                        "first difference at {at:?} of {} / {}",
                        got.len(),
                        want.len()
                    );
                }
                assert_eq!(
                    got, want,
                    "{kind:?}, {keys} keys, nulls {null_keys}, runs of {run}, {mode:?}, cap {cap}, chunks {chunks:?}, reload {reload}"
                );
            }
        }
    }
}

#[test]
fn every_dtype_of_key_gives_the_windows_one_answer() {
    // Interleaved keys with nulls among them, keys in runs, and a window.
    check(7, true, 1, Mode::Running);
    check(100, false, 13, Mode::Windowed);
    check(7, true, 1, Mode::Forward);
    check(100, false, 13, Mode::Forward);
}

#[test]
#[ignore = "extended: the whole grid of keys in a test build"]
fn every_dtype_of_key_gives_the_windows_one_answer_across_the_grid() {
    for keys in [1, 7, 100] {
        for null_keys in [false, true] {
            for mode in [Mode::Running, Mode::Windowed, Mode::Forward] {
                check(keys, null_keys, 1, mode);
                check(keys, null_keys, 13, mode);
            }
        }
    }
}

#[test]
#[ignore = "extended: many keys in a test build"]
fn every_dtype_of_key_gives_the_windows_one_answer_at_two_thousand_keys() {
    check(2000, false, 1, Mode::Running);
    check(2000, true, 5, Mode::Windowed);
    check(2000, true, 5, Mode::Forward);
}

#[test]
fn every_dtype_of_key_gives_the_windows_one_answer_where_rows_restart_their_group() {
    // A row every half second at 100 keys is 50 s after its key's last: past a
    // cap of 30 s about half of them restart their key's windows. At one key
    // a cap of 1 s is passed by one row in seven.
    for mode in [Mode::Running, Mode::Windowed, Mode::Forward] {
        let kinds = [Kind::Int64, Kind::Str, Kind::Cat];
        check_with(5000, 100, false, 1, mode, "30s", &kinds);
        check_with(5000, 1, false, 1, mode, "1s", &kinds);
        check_with(5000, 7, true, 1, mode, "2s", &kinds);
    }
}

#[test]
#[ignore = "extended: ten thousand keys in a test build"]
fn every_dtype_of_key_gives_the_windows_one_answer_where_every_row_restarts_its_group() {
    // 10,000 keys at 5 minutes: each key's rows are 5,000 s apart.
    for mode in [Mode::Running, Mode::Windowed, Mode::Forward] {
        check_with(
            40_000,
            10_000,
            false,
            1,
            mode,
            "5m",
            &[Kind::Int64, Kind::Str],
        );
    }
}

#[test]
fn every_dtype_of_session_column_gives_the_windows_one_answer() {
    let n = 5000;
    for keys in [0usize, 1, 7, 100] {
        let d = data(n, keys.max(1), false, 1);
        for mode in [Mode::Running, Mode::Windowed, Mode::Forward] {
            let cfg = config(mode, keys > 0, true);
            let group = (keys > 0).then_some(Kind::Int64);
            let (want, _) = digest(&d, &cfg, &[n], false, group, Some(Kind::Int64), 10);
            for kind in [
                Kind::Int32,
                Kind::Str,
                Kind::Cat,
                Kind::Enum,
                Kind::Float,
                Kind::Int64,
            ] {
                for chunks in [vec![n], vec![1000], vec![333, 1, 700, 2000]] {
                    for reload in [false, true] {
                        let (got, _) = digest(&d, &cfg, &chunks, reload, group, Some(kind), 10);
                        if matches!(kind, Kind::Float) {
                            // A float session is its own text ("1.0"), another
                            // value of the hash: the answer is the same only
                            // where the sessions' identity is.
                            assert_eq!(got, want, "float session, {keys} keys, {chunks:?}");
                            continue;
                        }
                        assert_eq!(
                            got, want,
                            "{kind:?} session, {keys} keys, {mode:?}, {chunks:?}, reload {reload}"
                        );
                    }
                }
            }
        }
    }
}
