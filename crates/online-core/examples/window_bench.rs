//! What a `window` costs a row between solves (review 2026-09-12, P1, P3):
//! `EwRidge` and an accumulate-only `EwCov`, with and without a window, at a
//! `k` where an O(k²) copy of the accumulator is the whole cost of a row.
//!
//!     cargo run --release -p online-core --example window_bench

use online_core::*;
use std::hint::black_box;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn ridge(k: usize, window: Option<f64>) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(500.0),
        ridge: vec![1e-6],
        feature_sets: vec![],
        standardize: false,
        ridge_decay: false,
        coef_prior: None,
        session_shrink: None,
        long_halflife: None,
        min_periods: (k + 1) as f64,
        solve_every: f64::MAX,
        max_rows_between_solves: 50,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window,
        window_every: window.map(|_| 25),
    }
}

/// An `ew_cov` that emits no statistics: E43's accumulate-only use, whose
/// value is its state, and `n_eff` the one number a row reads.
fn cov(k: usize, window: Option<f64>) -> EwCovCfg {
    EwCovCfg {
        n_features: k,
        decay: Decay::Halflife(500.0),
        stats: vec![],
        min_periods: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0,
        lags: vec![],
        window,
        window_every: window.map(|_| 25),
    }
}

/// Microseconds per row over `rows` rows of `k` features, after `rows / 4`
/// rows of warm-up.
fn time(mut step: impl FnMut(&[f64], f64), k: usize, rows: usize) -> f64 {
    let mut s = 7u64;
    let data: Vec<Vec<f64>> = (0..rows + rows / 4)
        .map(|_| (0..k).map(|_| lcg(&mut s)).collect())
        .collect();
    let (warm, timed) = data.split_at(rows / 4);
    for x in warm {
        step(x, x[0]);
    }
    let t = Instant::now();
    for x in timed {
        step(x, x[0]);
    }
    t.elapsed().as_secs_f64() * 1e6 / rows as f64
}

fn main() {
    let (k, rows) = (200, 4000);
    for window in [None, Some(1000.0)] {
        let mut m = EwRidge::new(ridge(k, window)).unwrap();
        let us = time(
            |x, y| {
                black_box(m.step(x, &[Some(y)], 1.0, 1.0));
            },
            k,
            rows,
        );
        println!("ewridge k={k} window={window:?}: {us:.1} us/row");
    }
    for window in [None, Some(1000.0)] {
        let mut m = EwCovModel::new(cov(k, window)).unwrap();
        let us = time(
            |x, _| {
                black_box(m.step(x, &[], 1.0, 1.0));
            },
            k,
            rows,
        );
        println!("ew_cov (no statistics) k={k} window={window:?}: {us:.1} us/row");
    }
}
