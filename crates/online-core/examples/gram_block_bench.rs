//! `gram_block_rows` (docs/ENHANCEMENTS.md E51, docs/PLAN.md task 71): the
//! `EwRidge::step` throughput with the Gram update per row against blocked,
//! single-threaded, at the widths the option is for.
//!
//!     cargo run --release -p online-core --example gram_block_bench
//!
//! Two lines per width and block: `solve never` is the update alone (the
//! first solve is taken in the warm-up and none comes due in the timed
//! rows), which is what the block changes; `solve /512` puts a solve every
//! 512 rows back in, which is the same `O(k^3)` on both sides and dilutes
//! the ratio -- that is the number a real cadence sees. The product is
//! `faer`'s, sequential, so these are one core's numbers; the bank's own
//! pool parallelises across groups, not inside a step.
//!
//! Widths and row counts may be given as pairs on the command line, for a
//! width the table does not have:
//!
//!     cargo run --release -p online-core --example gram_block_bench -- 64 65536

use online_core::*;
use std::hint::black_box;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn cfg(k: usize, block: usize, solve_rows: u32) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(5000.0),
        ridge: vec![1e-3],
        feature_sets: vec![],
        standardize: false,
        ridge_decay: false,
        coef_prior: None,
        session_shrink: None,
        long_halflife: None,
        min_periods: (k + 1) as f64,
        solve_every: f64::MAX,
        max_rows_between_solves: solve_rows,
        gram_block_rows: block,
        window: None,
        window_every: None,
    }
}

/// Rows per second over `n` timed rows, after `k + 2` warm-up rows that
/// take the first solve.
fn rows_per_s(k: usize, block: usize, solve_rows: u32, n: usize) -> f64 {
    let mut s = 7u64;
    let xs: Vec<Vec<f64>> = (0..n + k + 2)
        .map(|_| (0..k).map(|_| lcg(&mut s)).collect())
        .collect();
    let mut model = EwRidge::new(cfg(k, block, solve_rows)).unwrap();
    let mut acc = 0.0;
    let mut feed = |model: &mut EwRidge, x: &Vec<f64>, first: bool| {
        let y = Some(x[0] - 0.5 * x[k - 1] + 0.25);
        let step = model.step(x, &[y], if first { 0.0 } else { 1.0 }, 1.0);
        acc += step.n_eff;
        black_box(&step.pred);
    };
    for (i, x) in xs[..k + 2].iter().enumerate() {
        feed(&mut model, x, i == 0);
    }
    let t = Instant::now();
    for x in &xs[k + 2..] {
        feed(&mut model, x, false);
    }
    let dt = t.elapsed().as_secs_f64();
    black_box(acc);
    n as f64 / dt
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("width and row count, as integers"))
        .collect();
    let widths: Vec<(usize, usize)> = if args.is_empty() {
        vec![(256, 32_768), (1000, 4096), (2000, 2048)]
    } else {
        assert!(args.len() % 2 == 0, "pairs of width and row count");
        args.chunks(2).map(|p| (p[0], p[1])).collect()
    };
    println!(
        "{:>6} {:>6} {:>12} {:>12} {:>7}",
        "k", "block", "solve", "rows/s", "x"
    );
    for &(k, n) in &widths {
        for &(solve_rows, label) in &[(u32::MAX, "never"), (512u32, "/512")] {
            let base = rows_per_s(k, 0, solve_rows, n);
            println!("{k:>6} {:>6} {label:>12} {base:>12.0} {:>7.2}", 0, 1.0);
            for block in [64usize, 256] {
                let r = rows_per_s(k, block, solve_rows, n);
                println!("{k:>6} {block:>6} {label:>12} {r:>12.0} {:>7.2}", r / base);
            }
        }
    }
}
