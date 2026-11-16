//! Time `WindowsRun::feed` over a stream: one operator or several, with or
//! without a window, a group or a clock, at a chunk size (docs/PLAN.md tasks
//! 403, 412). Run it with `cargo run --release -p online-bench --example
//! winbench`; it prints a digest of the outputs beside the time, so two builds
//! can be compared for speed and for identity. Alternate the two builds'
//! binaries when timing a change, so that load moves both alike.
//! WB_KIND=dt|f|none WB_WS=none|60s|60.0 WB_OPS=1|4|8 WB_GROUPS=0|N WB_ROWS=n WB_REPS=n WB_CHUNK=n
//! WB_OPSET=mean|mixed|quant|msrq WB_PLAIN=0 (the plain-run loop off)
//! Task 441: WB_REVERSE=1 makes the operators forward (`rewm_*`, which need a WB_WS); WB_OP=mean|sum|rate|quantile|std|var
//! names the one operator of WB_OPS=1; `msrq` is mean, sum, rate, quantile (what a forward kernel can run).
//! The features task 429 prices, each off by default: WB_CAP=on|off (a gap_cap is required with a clock; `fire` also
//! puts a 400 s gap every thousand rows, past the 5 minute cap), WB_CLOSED=right|left|both|none, WB_NULLS=fraction of null
//! inputs, WB_TIES=fraction of repeated stamps, WB_SESSION=rows per session (0 none, a
//! session column with no session_gap), WB_RESTART=1 (restart_after_step_back set, no
//! step back in the data). Task 431: WB_KEY=int|str|cat (the group column's type, default int),
//! WB_SESSKEY=int|str|cat (the session column's), WB_RUN=n (the group stays the same for n rows at a time),
//! WB_CAPSEC=seconds (the gap_cap, default 300: with 10,000 keys at 0.5 s a row each key is silent
//! for 5,000 s on average, so every row of it is past the cap; a longer cap keeps the groups alive).
use std::time::Instant;

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

fn env(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

/// A key column of the given kind from integers: as they are, as their text, or categorical.
fn key_column(name: &str, v: &[i64], kind: &str) -> Column {
    let ints = Column::new(name.into(), v);
    match kind {
        "int" => ints,
        "str" => ints.cast(&DataType::String).unwrap(),
        "cat" => ints
            .cast(&DataType::String)
            .unwrap()
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap(),
        k => panic!("WB_KEY {k}"),
    }
}

fn main() {
    let kind = env("WB_KIND", "dt");
    let ws = env("WB_WS", "none");
    let ops: usize = env("WB_OPS", "1").parse().unwrap();
    let groups: usize = env("WB_GROUPS", "0").parse().unwrap();
    let rows: usize = env("WB_ROWS", "4000000").parse().unwrap();
    let reps: usize = env("WB_REPS", "3").parse().unwrap();
    let chunk: usize = env("WB_CHUNK", "65536").parse().unwrap();
    let opset = env("WB_OPSET", "mean");
    let gap = env("WB_GAP", "exp05");
    let cap_fire = env("WB_CAP", "on") == "fire";
    let closed = env("WB_CLOSED", "right");
    let nulls: f64 = env("WB_NULLS", "0").parse().unwrap();
    let ties: f64 = env("WB_TIES", "0").parse().unwrap();
    let session_rows: usize = env("WB_SESSION", "0").parse().unwrap();
    let restart = env("WB_RESTART", "0") == "1";
    let key_kind = env("WB_KEY", "int");
    let sess_kind = env("WB_SESSKEY", "int");
    let run_len: usize = env("WB_RUN", "1").parse().unwrap();
    let cap_sec: f64 = env("WB_CAPSEC", "300").parse().unwrap();
    // WB_PLAIN=0 switches the core's plain-run loop off (task 429), to time what it saves.
    let plain = env("WB_PLAIN", "1");
    let temporal = kind == "dt";
    // WB_HL: the half-life's text (d2: a nanosecond clock with a long half-life).
    let hl: Value = if temporal { json!(env("WB_HL", "5s")) } else { json!(5.0) };
    let wsv: Option<Value> = match ws.as_str() {
        "none" => None,
        s if temporal => Some(json!(s)),
        s => Some(json!(s.parse::<f64>().unwrap())),
    };
    let names = [
        "ewm_mean", "ewm_std", "ewm_sum", "ewm_var", "ewm_mean", "ewm_sum", "ewm_std", "ewm_var",
    ];
    let reverse = env("WB_REVERSE", "0") == "1";
    let one_op = env("WB_OP", "mean");
    let msrq = ["mean", "sum", "rate", "quantile"];
    let formulas: Vec<Value> = (0..ops)
        .map(|i| {
            let op: String = if opset == "quant" {
                "ewm_quantile".into()
            } else if opset == "mixed" {
                names[i % 8].into()
            } else if opset == "msrq" {
                format!("ewm_{}", msrq[i % 4])
            } else {
                format!("ewm_{one_op}")
            };
            let op = if reverse {
                op.replacen("ewm_", "rewm_", 1)
            } else {
                op
            };
            let op = op.as_str();
            let mut o = json!({"half_life": hl});
            if closed != "right" {
                o["closed"] = json!(closed);
            }
            if let Some(w) = &wsv {
                o["window_size"] = w.clone();
            }
            if op.ends_with("ewm_quantile") {
                o["quantile"] = json!(0.5);
            }
            json!({"name": format!("f{i}"), "tree": [op, ["col", "mid"], o]})
        })
        .collect();
    let mut config = json!({"formulas": formulas});
    if kind != "none" {
        config["clock"] = json!("ts");
        config["gap_cap"] = if temporal {
            json!(format!("{}s", cap_sec as u64))
        } else {
            json!(cap_sec)
        };
        if restart {
            config["restart_after_step_back"] = if temporal { json!("1s") } else { json!(1.0) };
        }
    }
    if session_rows > 0 {
        config["session"] = json!("s");
    }
    if groups > 0 {
        config["group"] = json!("g");
    }
    let config: WindowsConfig = serde_json::from_value(config).expect("config");

    let mut r = Rng(0x9E3779B97F4A7C15);
    let mut t = 0.0f64;
    let mut m = 100.0f64;
    let mut ts_ns = Vec::with_capacity(rows);
    let mut ts_f = Vec::with_capacity(rows);
    let mut mid: Vec<Option<f64>> = Vec::with_capacity(rows);
    let mut g = Vec::with_capacity(rows);
    let mut sess = Vec::with_capacity(rows);
    let mut r2 = Rng(0xD1B54A32D192ED03);
    let mut cur_g = 0i64;
    for i in 0..rows {
        let u = r.next();
        let step = match gap.as_str() {
            "reg" => 1.0,
            // WB_GAP=nsN (exponential, mean N ns) or regN (every gap N ns): a nanosecond clock
            g if g.starts_with("ns") => -g[2..].parse::<f64>().unwrap() * 1e-9 * u.ln(),
            g if g.starts_with("reg") => g[3..].parse::<f64>().unwrap() * 1e-9,
            "exp1" => -u.ln(),
            _ => -0.5 * u.ln(),
        };
        t += if ties > 0.0 && r2.next() < ties {
            0.0
        } else if cap_fire && i % 1000 == 999 {
            400.0
        } else {
            step
        };
        sess.push(i.checked_div(session_rows).map_or(0, |s| s as i64));
        m += (r.next() - 0.5) * 0.02;
        ts_ns.push((t * 1e9) as i64);
        ts_f.push(((t * 1e9) as i64) as f64 / 1e9);
        mid.push(if nulls > 0.0 && r2.next() < nulls {
            None
        } else {
            Some(m)
        });
        if i % run_len.max(1) == 0 {
            cur_g = if groups > 0 {
                (r.next() * groups as f64) as i64
            } else {
                0
            };
        }
        g.push(cur_g);
    }
    let mut frames = Vec::new();
    let mut at = 0;
    while at < rows {
        let to = (at + chunk).min(rows);
        let mut cols: Vec<Column> = Vec::new();
        if kind == "dt" {
            cols.push(
                Column::new("ts".into(), &ts_ns[at..to])
                    .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                    .unwrap(),
            );
        } else if kind == "f" {
            cols.push(Column::new("ts".into(), &ts_f[at..to]));
        }
        cols.push(Column::new("mid".into(), &mid[at..to]));
        if groups > 0 {
            cols.push(key_column("g", &g[at..to], &key_kind));
        }
        if session_rows > 0 {
            cols.push(key_column("s", &sess[at..to], &sess_kind));
        }
        frames.push(DataFrame::new_infer_height(cols).unwrap());
        at = to;
    }
    let mut best = f64::MAX;
    let mut digest = 0u64;
    let mut plain_rows = 0u64;
    let mut scratch_kb = 0usize;
    for rep in 0..reps {
        let mut run = WindowsRun::new(config.clone(), frames[0].schema()).expect("run");
        if plain == "0" {
            run.set_plain_run(false);
        }
        let mut h = 0xcbf29ce484222325u64;
        let t0 = Instant::now();
        let mut outs = Vec::new();
        for f in &frames {
            let o = run.feed(f, None).expect("feed");
            if rep == 0 {
                outs.push(o);
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        if rep == 0 {
            for o in &outs {
                for c in o.columns() {
                    if let Ok(ca) = c.f64() {
                        for v in ca.iter() {
                            h = (h ^ v.map_or(0x7ff8_dead_beef, f64::to_bits))
                                .wrapping_mul(0x100000001b3);
                        }
                    }
                }
            }
            digest = h;
            plain_rows = run.plain_rows();
            scratch_kb = run.plain_scratch_bytes() / 1024;
            // the saved state's bytes too
            let bytes = run.save_bytes().expect("save");
            let mut sh = 0xcbf29ce484222325u64;
            for b in &bytes {
                sh = (sh ^ *b as u64).wrapping_mul(0x100000001b3);
            }
            digest ^= sh.rotate_left(17);
        }
    }
    println!(
        "RESULT kind={kind} ws={ws} ops={ops} opset={opset} groups={groups} rows={rows} best={best:.4}s ns_row={:.1} digest={digest:016x} plain_rows={plain_rows} scratch_kb={scratch_kb}",
        best * 1e9 / rows as f64
    );
}
