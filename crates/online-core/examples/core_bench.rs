//! The pure-core throughput floor: `EwRidge::step` alone, no Polars, no
//! extraction, no assembly. The gap between this and the bank's rows/s is the
//! integration overhead docs/PERFORMANCE.md is about.
//!
//!     cargo run --release -p online-core --example core_bench

use online_core::*;
use std::hint::black_box;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(k: usize, m: usize, solve_every_rows: u32) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: m,
        add_intercept: true,
        decay: Decay::Halflife(500.0),
        ridge: vec![1e-6],
        feature_sets: vec![],
        standardize: false,
        ridge_decay: false,
        coef0: None,
        session_shrink: None,
        long_halflife: None,
        min_periods: (k + 1) as f64,
        solve_every: f64::MAX,
        max_rows_between_solves: solve_every_rows,
    }
}

fn bench(name: &str, k: usize, m: usize, solve_rows: u32) {
    const N: usize = 500_000;
    let mut s = 42u64;
    // Pre-generate rows so the RNG is not in the measured loop.
    let xs: Vec<Vec<f64>> = (0..N)
        .map(|_| (0..k).map(|_| lcg(&mut s)).collect())
        .collect();
    let ys: Vec<Vec<Option<f64>>> = xs
        .iter()
        .map(|x| {
            (0..m)
                .map(|j| Some(x[0] - 0.5 * x[k - 1] + j as f64))
                .collect()
        })
        .collect();

    let mut model = EwRidge::new(cfg(k, m, solve_rows)).unwrap();
    let t = Instant::now();
    let mut acc = 0.0;
    for i in 0..N {
        let step = model.step(&xs[i], &ys[i], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        acc += step.n_eff;
        black_box(&step.pred);
    }
    let dt = t.elapsed().as_secs_f64();
    black_box(acc);
    println!("{name:38} {:>12.0} rows/s", N as f64 / dt);
}

fn main() {
    // Warmup pass so the first measured line is not paying page faults.
    bench("warmup (ignore)", 5, 1, 10_000);
    for &(k, m, solve, label) in &[
        (5usize, 1usize, 10_000u32, "ewridge k=5  m=1  solve~never"),
        (20, 1, 10_000, "ewridge k=20 m=1  solve~never"),
        (50, 1, 10_000, "ewridge k=50 m=1  solve~never"),
        (20, 10, 10_000, "ewridge k=20 m=10 solve~never"),
        (20, 1, 1, "ewridge k=20 m=1  solve EVERY row"),
        (20, 1, 25, "ewridge k=20 m=1  solve every 25"),
    ] {
        bench(label, k, m, solve);
    }
}
