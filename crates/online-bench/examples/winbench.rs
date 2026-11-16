//! Time `WindowsRun::feed` over a stream: one operator or several, with or
//! without a window, a group or a clock, at a chunk size (docs/PLAN.md tasks
//! 403, 412). Run it with `cargo run --release -p online-bench --example
//! winbench`; it prints a digest of the outputs beside the time, so two builds
//! can be compared for speed and for identity. Alternate the two builds'
//! binaries when timing a change, so that load moves both alike.
//! WB_KIND=dt|f|none WB_WS=none|60s|60.0 WB_OPS=1|4|8 WB_GROUPS=0|N WB_ROWS=n WB_REPS=n WB_CHUNK=n
//! WB_OPSET=mean|mixed|quant
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
    let temporal = kind == "dt";
    let hl: Value = if temporal { json!("5s") } else { json!(5.0) };
    let wsv: Option<Value> = match ws.as_str() {
        "none" => None,
        s if temporal => Some(json!(s)),
        s => Some(json!(s.parse::<f64>().unwrap())),
    };
    let names = [
        "ewm_mean", "ewm_std", "ewm_sum", "ewm_var", "ewm_mean", "ewm_sum", "ewm_std", "ewm_var",
    ];
    let formulas: Vec<Value> = (0..ops)
        .map(|i| {
            let op = if opset == "quant" {
                "ewm_quantile"
            } else if opset == "mixed" {
                names[i % 8]
            } else {
                "ewm_mean"
            };
            let mut o = json!({"half_life": hl});
            if let Some(w) = &wsv {
                o["window_size"] = w.clone();
            }
            if op == "ewm_quantile" {
                o["quantile"] = json!(0.5);
            }
            json!({"name": format!("f{i}"), "tree": [op, ["col", "mid"], o]})
        })
        .collect();
    let mut config = json!({"formulas": formulas});
    if kind != "none" {
        config["clock"] = json!("ts");
        config["gap_cap"] = if temporal { json!("5m") } else { json!(300.0) };
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
    let mut mid = Vec::with_capacity(rows);
    let mut g = Vec::with_capacity(rows);
    for _ in 0..rows {
        let u = r.next();
        t += match gap.as_str() {
            "reg" => 1.0,
            "exp1" => -u.ln(),
            _ => -0.5 * u.ln(),
        };
        m += (r.next() - 0.5) * 0.02;
        ts_ns.push((t * 1e9) as i64);
        ts_f.push(((t * 1e9) as i64) as f64 / 1e9);
        mid.push(m);
        g.push(if groups > 0 {
            (r.next() * groups as f64) as i64
        } else {
            0
        });
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
            cols.push(Column::new("g".into(), &g[at..to]));
        }
        frames.push(DataFrame::new_infer_height(cols).unwrap());
        at = to;
    }
    let mut best = f64::MAX;
    let mut digest = 0u64;
    for rep in 0..reps {
        let mut run = WindowsRun::new(config.clone(), frames[0].schema()).expect("run");
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
        "RESULT kind={kind} ws={ws} ops={ops} opset={opset} groups={groups} rows={rows} best={best:.4}s ns_row={:.1} digest={digest:016x}",
        best * 1e9 / rows as f64
    );
}
