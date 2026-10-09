//! `gram_threads` (docs/PLAN.md task 225): one wide `EwCov`'s cost per Gram
//! entry per row, `t / (k² · rows)`, against the thread count, with the
//! moments compared to the bit against one thread's.
//!
//!     cargo run --release -p online-core --example gram_threads_bench -- [k...]
//!
//! Two paths: `block 256` is `ewridge`'s `gram_block_rows = 256`, the work in
//! [`EwCov::flush`]'s merge; `block 0` is the per-row rank-1 update that
//! `ew_cov` and an unblocked `ewridge` run. Everything runs inside one
//! rayon pool of `POOL` threads, as a bank's groups do, and `groups G` runs
//! `G` accumulators at once on that pool, each asking for `t` threads: the
//! many-groups case, where the threads come out of the same pool (set
//! `MANY_GROUPS=1`; `DIGEST=1` prints the moments' bits instead of times).

use online_core::EwCov;
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn rows(k: usize, n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n * k).map(|_| 10.0 + lcg(&mut s)).collect()
}

/// Feed `n` rows (after one warm-up block) and return seconds per row and
/// the accumulator.
fn run(k: usize, block: usize, threads: usize, xs: &[f64], warm: usize) -> (f64, EwCov) {
    let mut c = EwCov::new(k);
    c.set_block_rows(block);
    c.set_threads(threads);
    let lam = 0.999;
    let n = xs.len() / k;
    for r in 0..warm {
        c.update(&xs[r * k..(r + 1) * k], lam, 1.0);
    }
    c.flush();
    let t = Instant::now();
    for r in warm..n {
        c.update(&xs[r * k..(r + 1) * k], lam, 1.0);
    }
    c.flush();
    (t.elapsed().as_secs_f64() / (n - warm) as f64, c)
}

fn same_bits(a: &EwCov, b: &EwCov) -> bool {
    let bits = |c: &EwCov| -> Vec<u64> {
        c.comoments()
            .iter()
            .chain(c.means())
            .map(|v| v.to_bits())
            .collect()
    };
    bits(a) == bits(b) && a.n_eff().to_bits() == b.n_eff().to_bits()
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("widths, as integers"))
        .collect();
    let ks = if args.is_empty() {
        vec![1000, 2000, 4000]
    } else {
        args
    };
    let pool_n = std::env::var("POOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(14usize);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_n)
        .build()
        .expect("a pool");
    let counts = [1usize, 2, 4, 8, 14];
    println!("pool of {pool_n} threads; ps per Gram entry per row");
    for &k in &ks {
        let er = (k * k) as f64;
        for block in [256usize, 0] {
            // About a second of single-thread work per measurement.
            let n = if block > 0 {
                (4.0e10 / er).clamp(1024.0, 65536.0) as usize / 256 * 256 + 256
            } else {
                (1.5e9 / er).clamp(64.0, 65536.0) as usize + 16
            };
            let warm = if block > 0 { 256 } else { 16 };
            let xs = rows(k, n, 7 + k as u64);
            let (t1, one) = pool.install(|| run(k, block, 1, &xs, warm));
            if std::env::var_os("DIGEST").is_some() {
                // The moments' bits, for a build-to-build comparison.
                let h = one
                    .comoments()
                    .iter()
                    .chain(one.means())
                    .fold(0xcbf29ce484222325u64, |h, v| {
                        (h ^ v.to_bits()).wrapping_mul(0x100000001b3)
                    });
                println!("k={k} block {block} digest {h:016x}");
                continue;
            }
            print!("k={k:>5} block {block:>3}: 1 thr {:6.2}", t1 / er * 1e12);
            for &t in &counts[1..] {
                let (tt, many) = pool.install(|| run(k, block, t, &xs, warm));
                let same = if same_bits(&one, &many) {
                    ""
                } else {
                    " BITS DIFFER"
                };
                print!(
                    " | {t:>2} thr {:6.2} ({:4.1}x){same}",
                    tt / er * 1e12,
                    t1 / tt
                );
            }
            println!();
        }
    }
    // The many-groups case: `G` accumulators stepped at once on the pool,
    // each asking for `t` threads.
    if std::env::var_os("MANY_GROUPS").is_none() {
        return;
    }
    let k = 1000;
    let er = (k * k) as f64;
    let n = 4096 + 256;
    let xs = rows(k, n, 99);
    for g in [1usize, 4, 14] {
        for t in [1usize, 8] {
            let start = Instant::now();
            let outs: Vec<EwCov> = pool.install(|| {
                use rayon::prelude::*;
                (0..g)
                    .into_par_iter()
                    .map(|_| run(k, 256, t, &xs, 256).1)
                    .collect()
            });
            let wall = start.elapsed().as_secs_f64();
            let entry_rows = er * n as f64 * g as f64;
            println!(
                "groups {g:>2} x k={k}, block 256, gram_threads {t}: {:6.2} ps wall per entry-row (all groups)",
                wall / entry_rows * 1e12
            );
            assert!(outs.windows(2).all(|w| same_bits(&w[0], &w[1])));
        }
        // The other design: a pool of 8 of the product's own, which every
        // group's accumulator installs into, beside the bank's 14.
        let own = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("a pool");
        let start = Instant::now();
        pool.install(|| {
            use rayon::prelude::*;
            (0..g)
                .into_par_iter()
                .for_each(|_| drop(own.install(|| run(k, 256, 8, &xs, 256))))
        });
        let wall = start.elapsed().as_secs_f64();
        println!(
            "groups {g:>2} x k={k}, block 256, gram_threads 8 on a pool of its own: {:6.2} ps wall per entry-row (all groups)",
            wall / (er * n as f64 * g as f64) * 1e12
        );
    }
}
