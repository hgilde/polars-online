//! A bit-identity digest of the window operators (docs/PLAN.md tasks 403, 412,
//! 430): `WindowsRun`'s outputs and saved state over a matrix of cases, so the
//! next change to `windows.rs` that is meant to leave every output as it was
//! is checked by the suite, not by a harness kept beside it.
//!
//! The matrix is the four clocks (datetime, float, integer, none) with and
//! without a group, with no window and two windows, under the four values of
//! `closed`; every statistic (mean, standard deviation, variance, sum, rate,
//! quantile) looking back and, where a window is set, the mean, sum and
//! quantile looking ahead; a second input with its own nulls; `min_samples`
//! and `partial`; sessions, the restart policy and a refused step back. Each
//! case runs as four chunkings, two of them saved and loaded in the middle,
//! and the loaded run's own save must equal the bytes it was loaded from.
//!
//! Two hashes are pinned. [`MASKS`] hashes only which output rows are null
//! (and every output's length) and must hold on every platform: whether a row
//! has a value is decided from integers and sums, not from the last bit of an
//! exponential. [`BITS`] hashes every output float's bits and every state
//! byte, and holds where the exponential is Apple's, on arm64 macOS: the
//! system `exp2` and `expm1` differ in a last bit between libms, so no other
//! platform's bits are pinned (they are still run, and the loaded run's
//! re-save is still compared with its bytes).
//!
//! A change that moves outputs on purpose prints the new values: run
//! `WINDOW_DIGEST=print cargo test -p rheon-polars --test window_digest --
//! --nocapture` before and after, diff the `CASE` lines to name the cases
//! that moved and why, and replace the constants with the `TOTAL` line's.
use polars::prelude::*;
use rheon_polars::{WindowsConfig, WindowsRun};
use serde_json::{Value, json};

/// The digest of every output's null mask (all platforms).
const MASKS: u64 = 0x7d9052681ddf16a6;
/// The digest of every output float's bits and every state byte, arm64 macOS.
const BITS: u64 = 0x162b1609e20133e2;
/// Rows a case runs over.
const ROWS: usize = 2000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Dt,
    F,
    I,
    None,
}

struct Data {
    ts_ns: Vec<i64>,
    ts_f: Vec<f64>,
    ts_i: Vec<i64>,
    mid: Vec<Option<f64>>,
    m2: Vec<Option<f64>>,
    g: Vec<i64>,
    s: Vec<i64>,
}

/// A stream's rows: exponential gaps, or with `grid` gaps of exactly a tenth
/// of a second (and ties), so that a row is often exactly one window old, a
/// distance whose seconds are not whole doubles.
fn data(n: usize, seed: u64, steps_back: bool, grid: bool) -> Data {
    let mut r = Rng(seed);
    let (mut t, mut m) = (0.0f64, 100.0f64);
    let mut ticks = 0i64;
    let mut d = Data {
        ts_ns: vec![],
        ts_f: vec![],
        ts_i: vec![],
        mid: vec![],
        m2: vec![],
        g: vec![],
        s: vec![],
    };
    let mut sess = 0i64;
    for i in 0..n {
        let u = r.next();
        // exponential gaps, ties (20%), rare large gaps past the cap, rare steps back
        let gap = if u < 0.2 {
            0.0
        } else if u > 0.995 {
            400.0
        } else {
            if grid { 0.1 } else { -0.5 * r.next().ln() }
        };
        let back = steps_back && i > 0 && i % 997 == 0;
        if back {
            t -= 3.0;
            ticks -= 30;
        } else {
            t += gap;
            // The grid counts tenths of a second as integers.
            ticks += if gap == 0.0 {
                0
            } else if gap > 1.0 {
                4000
            } else {
                1
            };
        }
        m += (r.next() - 0.5) * 0.02;
        if grid {
            d.ts_ns
                .push(ticks * 100_000_000 + 1_700_000_000_000_000_000);
            d.ts_f.push(ticks as f64 / 10.0);
            d.ts_i.push(ticks * 100 + 5_000_000);
        } else {
            d.ts_ns.push((t * 1e9) as i64 + 1_700_000_000_000_000_000);
            d.ts_f.push(((t * 1e9) as i64) as f64 / 1e9);
            d.ts_i.push((t * 1000.0) as i64 + 5_000_000);
        }
        d.mid.push(if r.next() < 0.1 { None } else { Some(m) });
        d.m2.push(if r.next() < 0.3 {
            None
        } else {
            Some(m * 0.5 + r.next())
        });
        d.g.push((r.next() * 7.0) as i64);
        if r.next() < 0.003 {
            sess += 1;
        }
        d.s.push(sess);
    }
    d
}

fn frame(d: &Data, kind: Kind, grouped: bool, session: bool, from: usize, to: usize) -> DataFrame {
    let mut cols: Vec<Column> = vec![];
    match kind {
        Kind::Dt => cols.push(
            Column::new("ts".into(), &d.ts_ns[from..to])
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap(),
        ),
        Kind::F => cols.push(Column::new("ts".into(), &d.ts_f[from..to])),
        Kind::I => cols.push(Column::new("ts".into(), &d.ts_i[from..to])),
        Kind::None => {}
    }
    cols.push(Column::new("mid".into(), &d.mid[from..to]));
    cols.push(Column::new("m2".into(), &d.m2[from..to]));
    if grouped {
        cols.push(Column::new("g".into(), &d.g[from..to]));
    }
    if session {
        cols.push(Column::new("sess".into(), &d.s[from..to]));
    }
    DataFrame::new_infer_height(cols).unwrap()
}

/// Two FNV-1a accumulators: every bit of the output and the state, and which
/// output rows are null.
struct H {
    bits: u64,
    mask: u64,
}
impl H {
    fn new() -> H {
        H {
            bits: 0xcbf29ce484222325,
            mask: 0xcbf29ce484222325,
        }
    }
    fn u(&mut self, v: u64) {
        self.bits = (self.bits ^ v).wrapping_mul(0x100000001b3);
    }
    fn m(&mut self, v: u64) {
        self.mask = (self.mask ^ v).wrapping_mul(0x100000001b3);
    }
    fn frame(&mut self, f: &DataFrame) {
        for c in f.columns() {
            self.u(c.len() as u64);
            self.m(c.len() as u64);
            if let Ok(ca) = c.f64() {
                for v in ca.iter() {
                    self.u(v.map_or(0x7ff8_dead_beef, f64::to_bits));
                    self.m(u64::from(v.is_some()));
                }
            } else if let Ok(ca) = c.i64() {
                for v in ca.iter() {
                    self.u(v.map_or(1, |x| x as u64));
                    self.m(u64::from(v.is_some()));
                }
            }
        }
    }
    fn bytes(&mut self, b: &[u8]) {
        for x in b {
            self.u(u64::from(*x));
        }
    }
    /// A refusal's text, the same on every platform.
    fn text(&mut self, t: &str) {
        for x in t.bytes() {
            self.u(u64::from(x));
            self.m(u64::from(x));
        }
    }
}

#[derive(Clone)]
struct Case {
    grid: bool,
    name: String,
    config: Value,
    kind: Kind,
    grouped: bool,
    session: bool,
    steps_back: bool,
}

fn op2(name: &str, hl: &Value, ws: Option<&Value>, extra: Value) -> Value {
    let mut t = op(name, hl, ws, extra);
    t[1] = json!(["col", "m2"]);
    t
}

fn op(name: &str, hl: &Value, ws: Option<&Value>, extra: Value) -> Value {
    let mut o = json!({"half_life": hl});
    // A sum's default is 0 (an empty window sums to 0.0, the empty-sum tests');
    // here it asks for one value, as it did, so the hashes pin nothing else
    // moving.
    if name.ends_with("sum") {
        o["min_samples"] = json!(1);
    }
    if let Some(w) = ws {
        o["window_size"] = w.clone();
    }
    if let Value::Object(e) = extra {
        for (k, v) in e {
            o[k] = v;
        }
    }
    json!([name, ["col", "mid"], o])
}

#[test]
#[ignore = "extended: half a minute in a test build"]
fn the_window_operators_keep_every_output_bit_and_every_null() {
    let print = std::env::var("WINDOW_DIGEST").is_ok_and(|v| v == "print");
    let n = ROWS;
    let mut cases: Vec<Case> = vec![];
    for kind in [Kind::Dt, Kind::F, Kind::I, Kind::None] {
        for grouped in [false, true] {
            for ws in [None, Some(0usize), Some(1)] {
                // window size: none, or a window (long / short)
                let (hl, w): (Value, Option<Value>) = match (kind, ws) {
                    (Kind::Dt, None) => (json!("5s"), None),
                    (Kind::Dt, Some(0)) => (json!("5s"), Some(json!("60s"))),
                    (Kind::Dt, Some(_)) => (json!("2s"), Some(json!("7s"))),
                    (Kind::F, None) => (json!(5.0), None),
                    (Kind::F, Some(0)) => (json!(5.0), Some(json!(60.0))),
                    (Kind::F, Some(_)) => (json!(2.0), Some(json!(7.0))),
                    (Kind::I, None) => (json!(5000.0), None),
                    (Kind::I, Some(0)) => (json!(5000.0), Some(json!(60000.0))),
                    (Kind::I, Some(_)) => (json!(2000.0), Some(json!(7000.0))),
                    (Kind::None, None) => (json!(10.0), None),
                    (Kind::None, Some(0)) => (json!(10.0), Some(json!(100.0))),
                    (Kind::None, Some(_)) => (json!(4.0), Some(json!(15.0))),
                };
                for closed in ["right", "left", "both", "none"] {
                    if w.is_none() && closed != "right" {
                        continue;
                    }
                    let mut formulas = vec![];
                    let ex = |q: bool| {
                        let mut o = json!({"closed": closed});
                        if q {
                            o["quantile"] = json!(0.5);
                        }
                        o
                    };
                    formulas.push(
                        json!({"name": "mean", "tree": op("ewm_mean", &hl, w.as_ref(), ex(false))}),
                    );
                    formulas.push(
                        json!({"name": "std", "tree": op("ewm_std", &hl, w.as_ref(), ex(false))}),
                    );
                    formulas.push(json!({"name": "var", "tree": op("ewm_var", &hl, w.as_ref(), json!({"closed": closed, "bias": true}))}));
                    formulas.push(
                        json!({"name": "sum", "tree": op("ewm_sum", &hl, w.as_ref(), ex(false))}),
                    );
                    formulas.push(
                        json!({"name": "rate", "tree": op("ewm_rate", &hl, w.as_ref(), ex(false))}),
                    );
                    formulas.push(
                        json!({"name": "q", "tree": op("ewm_quantile", &hl, w.as_ref(), ex(true))}),
                    );
                    if let Some(w) = &w {
                        formulas.push(json!({"name": "ahead_mean", "tree": op("rewm_mean", &hl, Some(w), json!({"closed": closed}))}));
                        formulas.push(json!({"name": "ahead_sum", "tree": op("rewm_sum", &hl, Some(w), json!({"closed": closed, "partial": "keep"}))}));
                        formulas.push(json!({"name": "ahead_q", "tree": op("rewm_quantile", &hl, Some(w), json!({"closed": closed, "quantile": 0.25}))}));
                    }
                    formulas.push(json!({"name": "mean2", "tree": op2("ewm_mean", &hl, w.as_ref(), ex(false))}));
                    formulas.push(
                        json!({"name": "std2", "tree": op2("ewm_std", &hl, w.as_ref(), ex(false))}),
                    );
                    formulas.push(json!({"name": "var2", "tree": op2("ewm_var", &hl, w.as_ref(), json!({"closed": closed}))}));
                    // partial / min_samples variants on the mean
                    formulas.push(json!({"name": "mean_ms", "tree": op("ewm_mean", &hl, w.as_ref(), json!({"closed": closed, "min_samples": 3, "partial": "null"}))}));
                    let mut config = json!({"formulas": formulas});
                    if kind != Kind::None {
                        config["clock"] = json!("ts");
                        config["gap_cap"] = match kind {
                            Kind::Dt => json!("30s"),
                            Kind::F => json!(30.0),
                            _ => json!(30000.0),
                        };
                    }
                    if grouped {
                        config["group"] = json!("g");
                    }
                    // a window with groups and no clock has no forward kernel (needs a clock column)
                    let cases_ok = !(kind == Kind::None && grouped && w.is_some());
                    if cases_ok {
                        cases.push(Case {
                            name: format!("{kind:?} g={grouped} ws={ws:?} {closed}"),
                            config: config.clone(),
                            kind,
                            grouped,
                            session: false,
                            steps_back: false,
                            grid: false,
                        });
                    }
                    // the sessions and the step-back policy, once per clock/window
                    if closed == "right" && kind != Kind::None && cases_ok {
                        let mut c2 = config.clone();
                        c2["session"] = json!("sess");
                        c2["session_gap"] = json!("reset");
                        cases.push(Case {
                            name: format!("{kind:?} g={grouped} ws={ws:?} session"),
                            config: c2,
                            kind,
                            grouped,
                            session: true,
                            steps_back: false,
                            grid: false,
                        });
                        let mut c3 = config.clone();
                        c3["restart_after_step_back"] = match kind {
                            Kind::Dt => json!("1s"),
                            Kind::F => json!(1.0),
                            _ => json!(1000.0),
                        };
                        cases.push(Case {
                            name: format!("{kind:?} g={grouped} ws={ws:?} restart"),
                            config: c3,
                            kind,
                            grouped,
                            session: false,
                            steps_back: true,
                            grid: false,
                        });
                        // a refused step back
                        cases.push(Case {
                            name: format!("{kind:?} g={grouped} ws={ws:?} refuse"),
                            config: config.clone(),
                            kind,
                            grouped,
                            session: false,
                            steps_back: true,
                            grid: false,
                        });
                    }
                }
            }
        }
    }
    // Each case again on a regular grid of stamps.
    let grid_cases: Vec<Case> = cases
        .iter()
        .map(|c| Case {
            name: format!("{} grid", c.name),
            grid: true,
            ..c.clone()
        })
        .collect();
    cases.extend(grid_cases);
    let streams = [
        data(n, 0x9E3779B97F4A7C15, false, false),
        data(n, 0xD1B54A32D192ED03, true, false),
        data(n, 0x9E3779B97F4A7C15, false, true),
        data(n, 0xD1B54A32D192ED03, true, true),
    ];
    let mut total = H::new();
    for c in &cases {
        let data = &streams[usize::from(c.grid) * 2 + usize::from(c.steps_back)];
        let config: WindowsConfig = match serde_json::from_value(c.config.clone()) {
            Ok(x) => x,
            Err(e) => {
                panic!("CASE {} CONFIG-ERR {e}", c.name);
            }
        };
        let mut line = String::new();
        for (ci, chunks) in [
            vec![n],
            vec![500, 500, 250, 250, 250, 250],
            vec![7, 1, 13, 64, 1500, 415],
            vec![1000, 1000],
        ]
        .iter()
        .enumerate()
        {
            let mut h = H::new();
            let schema = frame(data, c.kind, c.grouped, c.session, 0, 1)
                .schema()
                .clone();
            let mut run = match WindowsRun::new(config.clone(), &schema) {
                Ok(r) => r,
                Err(e) => {
                    h.text(&format!("{e:?}"));
                    line += " new-err";
                    total.u(h.bits);
                    total.m(h.mask);
                    continue;
                }
            };
            let (mut at, mut err) = (0usize, false);
            for (k, &len) in chunks.iter().enumerate() {
                let to = (at + len).min(n);
                if at >= to {
                    break;
                }
                let f = frame(data, c.kind, c.grouped, c.session, at, to);
                match run.feed(&f, None) {
                    Ok(o) => h.frame(&o),
                    Err(e) => {
                        h.text(&format!("{e}"));
                        err = true;
                        break;
                    }
                }
                at = to;
                // save/load mid-stream after the second chunk in the last two chunkings
                if ci >= 2 && k == 1 {
                    let bytes = run.save_bytes().expect("save");
                    h.bytes(&bytes);
                    run = WindowsRun::load_bytes(&bytes, config.clone(), &schema).expect("load");
                    // the loaded run's own re-save must agree with the bytes
                    assert_eq!(
                        run.save_bytes().expect("resave"),
                        bytes,
                        "{} resave",
                        c.name
                    );
                }
            }
            if !err {
                match run.finish() {
                    Ok(o) => h.frame(&o),
                    Err(e) => h.text(&format!("{e}")),
                }
                h.bytes(&run.save_bytes().expect("save end"));
            }
            line += &format!(" {:016x}/{:016x}", h.mask, h.bits);
            total.u(h.bits);
            total.m(h.mask);
        }
        if print {
            println!("CASE {} {line}", c.name);
        }
    }
    println!(
        "TOTAL {} cases masks {:016x} bits {:016x}",
        cases.len(),
        total.mask,
        total.bits
    );
    assert_eq!(total.mask, MASKS, "which output rows are null moved");
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        assert_eq!(total.bits, BITS, "an output bit or a state byte moved");
    }
}
