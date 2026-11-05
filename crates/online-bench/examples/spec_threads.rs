//! What the bank's pool gains across the specs on one stream: the wall time
//! of `Bank::fit_predict` with 1, 8 and 64 specs on pools of 1, 4 and 8
//! threads (docs/PERFORMANCE.md, "How cost grows"). It gates nothing: a
//! wall time moves with the machine's load, which it prints first.
//!
//!     cargo run --profile dev-wheel -p online-bench --example spec_threads
//!
//! Each bank is warmed on 4,096 rows, then timed over 65,536 rows fed as 16
//! calls of 4,096, the least of three runs.

use std::time::Instant;

use online_polars::{Bank, Spec};
use polars::prelude::*;
use serde_json::{Value, json};

fn spec(name: String, model: Value, features: usize, target: bool) -> Spec {
    let mut v = json!({
        "name": name,
        "model": model,
        "features": (0..features).map(|j| format!("x{j}")).collect::<Vec<_>>(),
        "clock": "t",
        "gap_cap": 100.0,
        "half_life": 1000.0,
    });
    if v["model"]["type"] == "bocpd" {
        // A run-length posterior forgets by its hazard, not a half-life.
        v.as_object_mut().expect("an object").remove("half_life");
    }
    if target {
        v["targets"] = json!(["y0"]);
        v["min_weight"] = json!(features as f64);
    }
    let mut s: Spec = serde_json::from_value(v).expect("a spec");
    s.check().expect("a valid spec");
    s
}

fn specs(n: usize, mixed: bool) -> Vec<Spec> {
    (0..n)
        .map(|i| {
            let name = format!("m{i}");
            if !mixed {
                return spec(name, json!({"type": "ewridge"}), 8, true);
            }
            match i % 5 {
                0 => spec(name, json!({"type": "ewridge"}), 8, true),
                1 => spec(name, json!({"type": "huber"}), 8, true),
                2 => spec(name, json!({"type": "ew_cov"}), 8, false),
                3 => spec(name, json!({"type": "kmeans", "n_clusters": 4}), 8, false),
                _ => spec(
                    name,
                    json!({"type": "bocpd", "prune_below": 1e-6, "max_run": 50}),
                    8,
                    false,
                ),
            }
        })
        .collect()
}

fn frame(from: usize, n: usize) -> DataFrame {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ from as u64;
    let mut draw = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut cols: Vec<Vec<f64>> = (0..8).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let mut fit = 0.0;
        for (j, c) in cols.iter_mut().enumerate() {
            let x = draw();
            fit += x / (j + 1) as f64;
            c.push(x);
        }
        y.push(fit + 0.3 * draw());
    }
    let mut out: Vec<Column> = cols
        .into_iter()
        .enumerate()
        .map(|(j, c)| Column::new(format!("x{j}").into(), c))
        .collect();
    out.push(Column::new("y0".into(), y));
    out.push(Column::new(
        "t".into(),
        (from..from + n).map(|i| i as f64).collect::<Vec<_>>(),
    ));
    DataFrame::new_infer_height(out).expect("one length")
}

fn main() {
    let load = std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();
    println!("load: {load}");
    online_polars::pool().expect("the pool");
    let (warm, calls, call) = (4_096, 16, 4_096);
    let frames: Vec<DataFrame> = (0..calls).map(|c| frame(warm + c * call, call)).collect();
    println!("| specs | kind | 1 thread (ms) | 4 threads | 8 threads | speed-up at 4 | at 8 |");
    println!("|---|---|---|---|---|---|---|");
    for (n, mixed) in [(1, false), (8, false), (64, false), (5, true), (40, true)] {
        let mut ms = Vec::new();
        for threads in [1, 4, 8] {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let mut bank = Bank::new(specs(n, mixed)).expect("a bank");
                bank.set_threads(Some(threads)).expect("threads");
                bank.fit_predict(&frame(0, warm)).expect("warm");
                let at = Instant::now();
                for f in &frames {
                    std::hint::black_box(bank.fit_predict(f).expect("a call"));
                }
                best = best.min(at.elapsed().as_secs_f64() * 1e3);
            }
            ms.push(best);
        }
        println!(
            "| {n} | {} | {:.1} | {:.1} | {:.1} | {:.2}x | {:.2}x |",
            if mixed { "mix of five" } else { "ewridge" },
            ms[0],
            ms[1],
            ms[2],
            ms[0] / ms[1],
            ms[0] / ms[2]
        );
    }
}
