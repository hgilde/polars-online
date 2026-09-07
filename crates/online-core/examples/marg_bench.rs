//! `marginal` with none of its optional parts, over a fixed stream.
//!
//! This exists to answer one question: does a stream that asks for no lags,
//! no bins and no window pay for the fact that they exist? Run it against a
//! build from before the feature under suspicion, in a separate worktree and
//! `CARGO_TARGET_DIR`, best of the runs each prints (docs/PERFORMANCE.md
//! §17). It found two real costs that way -- a call inside `learn`'s loop
//! and an `O(p)` scan outside its guard -- neither of which any test could
//! have seen.
use online_core::*;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn main() {
    const ROWS: usize = 1_000_000;
    const K: usize = 8;
    let mut seed = 12345u64;
    let rows: Vec<(Vec<f64>, f64)> = (0..ROWS)
        .map(|_| {
            let x: Vec<f64> = (0..K).map(|_| lcg(&mut seed)).collect();
            let y = x[0] + 0.3 * lcg(&mut seed);
            (x, y)
        })
        .collect();
    let mut best = f64::INFINITY;
    for _ in 0..25 {
        let mut m = Marginal::new(MarginalCfg {
            n_features: K,
            n_targets: 1,
            decay: Decay::Halflife(500.0),
            min_periods: vec![3.0],
            lags: Vec::new(),
            serial_rule: None,
            bins: None,
            window: None,
            window_every: None,
        })
        .unwrap();
        let t0 = Instant::now();
        for (i, (x, y)) in rows.iter().enumerate() {
            OnlineModel::step(&mut m, x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let el = t0.elapsed().as_secs_f64();
        best = best.min(el);
        std::hint::black_box(m.pair(0, 0));
    }
    println!("{:.1} rows/s ({:.1} ms)", ROWS as f64 / best, best * 1e3);
}
