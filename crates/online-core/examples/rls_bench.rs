//! `Rls::step` alone, no Polars: the arithmetic the C5 square-root rewrite
//! changed. Same file runs against the covariance form (50c1a38^) and the QR
//! form, because `RlsCfg` is unchanged between them.
use online_core::*;
use std::hint::black_box;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn main() {
    let rows = 200_000usize;
    for k in [5usize, 20, 50] {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let mut m = Rls::new(RlsCfg {
                n_features: k,
                n_targets: 1,
                add_intercept: true,
                decay: Decay::Halflife(500.0),
                ridge: 1.0,
                coef0: None,
                min_periods: 25.0,
            })
            .unwrap();
            let mut s = 12345u64;
            let x: Vec<f64> = (0..rows * k).map(|_| lcg(&mut s)).collect();
            let y: Vec<f64> = (0..rows).map(|_| lcg(&mut s)).collect();
            let t0 = Instant::now();
            for i in 0..rows {
                black_box(m.step(&x[i * k..(i + 1) * k], &[Some(y[i])], 1.0, 1.0));
            }
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!("rls k={k}: {:.0} rows/s", rows as f64 / best);
    }
}
