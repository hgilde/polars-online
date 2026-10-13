//! `ewridge`'s cost per feature-row against its target count (docs/PLAN.md
//! task 257): one accumulator of `k` features, `T` targets, a 256-row block
//! (`gram_block_rows`), never solved -- the Gram-export workload, whose
//! Grams are merged and solved offline. `EwRidge::step` is timed whole
//! (prediction, the target moments, the cross-moments and the Gram), in ns
//! of wall time per feature per row, on one thread and on the pool's.
//!
//!     cargo run --release -p online-core --example gram_targets_bench -- [k [rows]]
//!
//! `GAPS=1` makes each target null on about one row in ten, its own rows,
//! under `target_gaps = "pairwise"`; `OWN=1` runs `own_rows` instead.
//! `ONLY=T` times one target count, `THREADS=n` one thread count, `ROUNDS=r`
//! repeats each case (the line shows the fastest and slowest).

use online_core::*;
use std::hint::black_box;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(k: usize, t: usize, gaps: TargetGaps) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: t,
        fit_intercept: true,
        decay: Decay::Lam(1.0),
        ridge: vec![1e-3],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: 1e300,
        solve_every: f64::MAX,
        max_rows_between_solves: u32::MAX,
        solve_share: None,
        gram_block_rows: 256,
        target_gaps: gaps,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

struct Data {
    x: Vec<f64>,
    y: Vec<Option<f64>>,
}

fn data(k: usize, t: usize, n: usize, gaps: bool) -> Data {
    let mut s = 7u64;
    let mut x = Vec::with_capacity(n * k);
    let mut y = Vec::with_capacity(n * t);
    for _ in 0..n {
        let row = x.len();
        x.extend((0..k).map(|_| 10.0 + lcg(&mut s)));
        let fit: f64 = x[row..].iter().step_by(7).sum();
        for j in 0..t {
            let v = fit + j as f64 + 0.3 * lcg(&mut s);
            let null = gaps && lcg(&mut s) > 0.8;
            y.push((!null).then_some(v));
        }
    }
    Data { x, y }
}

/// ns per feature per row over the timed rows, after a warm-up of two
/// blocks.
fn run(k: usize, t: usize, d: &Data, gaps: TargetGaps, threads: usize) -> f64 {
    let n = d.x.len() / k;
    let warm = 512;
    let mut model = EwRidge::new(cfg(k, t, gaps)).unwrap();
    model.set_gram_threads(threads);
    let feed = |model: &mut EwRidge, r: usize| {
        let step = model.step(
            &d.x[r * k..(r + 1) * k],
            &d.y[r * t..(r + 1) * t],
            if r == 0 { 0.0 } else { 1.0 },
            1.0,
        );
        black_box(step.n_eff);
    };
    for r in 0..warm {
        feed(&mut model, r);
    }
    let clock = Instant::now();
    for r in warm..n {
        feed(&mut model, r);
    }
    let dt = clock.elapsed().as_secs_f64();
    black_box(model.cross_moments());
    dt * 1e9 / ((n - warm) * k) as f64
}

fn env(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("a width and a row count, as integers"))
        .collect();
    let k = args.first().copied().unwrap_or(200);
    let n = args.get(1).copied().unwrap_or(512 + 256 * 64);
    let gappy = env("GAPS") == Some(1);
    let gaps = if env("OWN") == Some(1) {
        TargetGaps::OwnRows
    } else {
        TargetGaps::Pairwise
    };
    let rounds = env("ROUNDS").unwrap_or(3);
    let pool_n = env("POOL").unwrap_or(14);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_n)
        .build()
        .expect("a pool");
    let targets: Vec<usize> = env("ONLY").map_or(vec![1, 2, 5, 10], |t| vec![t]);
    let counts: Vec<usize> = env("THREADS").map_or(vec![1, pool_n], |c| vec![c]);
    println!("k = {k}, {n} rows, {gaps:?}, gaps in the targets: {gappy}; ns per feature-row");
    for &t in &targets {
        let d = data(k, t, n, gappy);
        for &threads in &counts {
            let times: Vec<f64> = (0..rounds)
                .map(|_| pool.install(|| run(k, t, &d, gaps, threads)))
                .collect();
            let lo = times.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = times.iter().copied().fold(0.0, f64::max);
            println!("T = {t:>2}  threads {threads:>2}  {lo:.2}-{hi:.2}");
        }
    }
}
