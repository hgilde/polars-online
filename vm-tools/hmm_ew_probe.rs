//! Digest and timing of hmm (k=20 features, K=2 states) and ew_class (k=20,
//! full covariance, 3 classes) through Bank::fit_predict, on a fixed seeded
//! input. Usage: hmm_ew_probe digest | time [rows] [reps]
//! Copied into crates/online-polars/examples/ of a measurement tree (untracked).
use online_polars::{Bank, Spec};
use polars::prelude::*;
use std::time::Instant;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const K: usize = 20;

fn frame(n: usize, classes: usize) -> DataFrame {
    let mut s = 12345u64;
    let mut cols: Vec<Vec<f64>> = vec![Vec::with_capacity(n); K];
    let mut y = Vec::with_capacity(n);
    let mut state = 0usize;
    for _ in 0..n {
        if (lcg(&mut s) + 1.0) / 2.0 < 0.02 {
            state = ((lcg(&mut s) + 1.0) / 2.0 * classes as f64) as usize % classes;
        }
        y.push(state as f64);
        for (j, c) in cols.iter_mut().enumerate() {
            c.push(lcg(&mut s) + 1.5 * state as f64 * (1.0 + (j % 3) as f64 * 0.2));
        }
    }
    let mut out: Vec<Column> = Vec::new();
    for (j, c) in cols.into_iter().enumerate() {
        out.push(Column::new(format!("x{j}").into(), c));
    }
    out.push(Column::new("y".into(), y));
    DataFrame::new(n, out).unwrap()
}

fn spec(model: &str) -> Spec {
    let feats: Vec<String> = (0..K).map(|j| format!("\"x{j}\"")).collect();
    let m = if model == "hmm" {
        r#"{"type": "hmm", "n_components": 2, "precision_prior": 0.1}"#
    } else {
        r#"{"type": "ew_class", "classes": ["a", "b", "c"], "precision_prior": 1.0, "covariance": "full"}"#
    };
    serde_json::from_str(&format!(
        r#"{{"name": "s", "model": {m}, "targets": ["y"], "features": [{}], "half_life": 200.0, "min_weight": 30.0}}"#,
        feats.join(",")
    ))
    .unwrap()
}

fn h(acc: &mut u64, v: u64) {
    *acc = (*acc ^ v).wrapping_mul(0x100000001b3);
}

fn hash_series(name: &str, s: &Series, out: &mut Vec<(String, u64)>) {
    if let Ok(st) = s.struct_() {
        for f in st.fields_as_series() {
            hash_series(&format!("{name}.{}", f.name()), &f, out);
        }
        return;
    }
    let mut acc = 0xcbf29ce484222325u64;
    h(&mut acc, s.len() as u64);
    h(&mut acc, s.null_count() as u64);
    let c = s.cast(&DataType::Float64).unwrap_or_else(|_| s.clone());
    if let Ok(ca) = c.f64() {
        for v in ca.iter() {
            h(&mut acc, v.map_or(0xdead_beef_u64, |x| x.to_bits()));
        }
    } else {
        for i in 0..s.len() {
            for b in format!("{:?}", s.get(i).unwrap()).bytes() {
                h(&mut acc, b as u64);
            }
        }
    }
    out.push((name.to_string(), acc));
}

fn run(model: &str, df: &DataFrame) -> Vec<Column> {
    let mut bank = Bank::new(vec![spec(model)]).unwrap();
    bank.fit_predict(df).unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("digest") => {
            for model in ["hmm", "ew_class"] {
                let df = frame(20_000, if model == "hmm" { 2 } else { 3 });
                let cols = run(model, &df);
                let mut v = Vec::new();
                for c in &cols {
                    hash_series(c.name().as_str(), c.as_materialized_series(), &mut v);
                }
                for (n, x) in v {
                    println!("{model} {n} {x:016x}");
                }
            }
        }
        Some("time") => {
            let rows: usize = a.get(2).map_or(20_000, |x| x.parse().unwrap());
            let reps: usize = a.get(3).map_or(3, |x| x.parse().unwrap());
            for model in ["hmm", "ew_class"] {
                let df = frame(rows, if model == "hmm" { 2 } else { 3 });
                let mut best = f64::MAX;
                for _ in 0..reps {
                    let t = Instant::now();
                    let c = run(model, &df);
                    let e = t.elapsed().as_secs_f64() * 1e9 / rows as f64;
                    std::hint::black_box(&c);
                    best = best.min(e);
                }
                println!("time {model} {best:.1}");
            }
        }
        _ => panic!("digest | time"),
    }
}
