//! `Sgd::step` alone at a wide `k`, no Polars: the per-feature cost of the
//! inner loop, which is what docs/PERFORMANCE.md §19 measured at 13–14 ns
//! per feature per row through the bank against about 5 ns for sklearn's
//! batched Cython loop.
//!
//!     cargo run --release -p online-core --example sgd_bench

use online_core::*;
use std::hint::black_box;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(k: usize, scale: bool, schedule: LearningRate, l2: f64) -> SgdCfg {
    SgdCfg {
        n_features: k,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        loss: SgdLoss::Squared,
        learning_rate: 0.2 / k as f64,
        schedule,
        l2,
        min_periods: 50.0,
        scale_features: scale,
        clip_gradient: 1e3,
        constraint: None,
    }
}

fn bench(name: &str, k: usize, n: usize, cfg: SgdCfg) {
    let mut s = 42u64;
    let xs: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..k).map(|_| lcg(&mut s)).collect())
        .collect();
    let ys: Vec<Vec<Option<f64>>> = xs
        .iter()
        .map(|x| vec![Some(x[0] - 0.5 * x[k - 1])])
        .collect();
    let mut model = Sgd::new(cfg).unwrap();
    let t = Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        let step = model.step(&xs[i], &ys[i], 1.0, 1.0);
        acc += step.n_eff;
        black_box(&step.pred);
    }
    let dt = t.elapsed().as_secs_f64();
    black_box(acc);
    black_box(model.coefficients());
    println!(
        "{name:44} k={k:>6} {:>9.1} us/row {:>6.2} ns/feature",
        dt / n as f64 * 1e6,
        dt / n as f64 / k as f64 * 1e9
    );
}

fn main() {
    bench(
        "warmup (ignore)",
        100,
        2_000,
        cfg(100, false, LearningRate::Constant, 0.0),
    );
    for &(k, n) in &[(1_000usize, 20_000usize), (10_000, 2_000)] {
        bench(
            "constant, unscaled",
            k,
            n,
            cfg(k, false, LearningRate::Constant, 0.0),
        );
        bench(
            "constant, unscaled, l2=1e-4",
            k,
            n,
            cfg(k, false, LearningRate::Constant, 1e-4),
        );
        bench(
            "constant, scale_features",
            k,
            n,
            cfg(k, true, LearningRate::Constant, 0.0),
        );
        bench(
            "inv_scaling, unscaled",
            k,
            n,
            cfg(k, false, LearningRate::InvScaling { power: 0.25 }, 0.0),
        );
        bench(
            "adagrad, unscaled",
            k,
            n,
            cfg(k, false, LearningRate::AdaGrad, 0.0),
        );
    }
}
