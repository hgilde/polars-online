//! Every matrix product path the library runs, as the bits of its result:
//! one line per case, for a build-to-build comparison (docs/PLAN.md task
//! 258, where faer stopped linking `private-gemm-x86` on x86-64).
//!
//!     cargo run --release -p online-core --example product_digest > a.txt
//!
//! The cases: a wide `EwCov`'s moments, blocked (`gram_block_rows = 256`,
//! the merge's products) and per row, at one and four `gram_threads`; a
//! factorization and solve (`SpdFactor`: faer's Cholesky steps, the
//! triangular solves, the trailing products); an eigendecomposition
//! (`Pca`); and faer's own product `XᵀX`. The widths straddle faer's
//! thresholds: 16³ entries (its small-matrix kernel), 64 (its Cholesky's
//! recursion), 128 and 256 (the cut's tile and leaf).

use faer::prelude::*;
use online_core::{EwCov, Pca, SpdFactor};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn digest<'a>(vs: impl IntoIterator<Item = &'a f64>) -> u64 {
    vs.into_iter().fold(0xcbf29ce484222325u64, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(0x100000001b3)
    })
}

fn spd(k: usize) -> Vec<f64> {
    let mut s = 3u64;
    let n = k + 7;
    let x: Vec<f64> = (0..n * k).map(|_| 0.5 + lcg(&mut s)).collect();
    let mut g = vec![0.0; k * k];
    // A plain sum, so the matrix is the same on every build.
    for r in 0..n {
        let row = &x[r * k..(r + 1) * k];
        for i in 0..k {
            for j in 0..k {
                g[i * k + j] += row[i] * row[j];
            }
        }
    }
    for i in 0..k {
        g[i * k + i] += 1e-3 * n as f64;
    }
    g
}

fn main() {
    let pools: Vec<(usize, rayon::ThreadPool)> = [1usize, 4]
        .iter()
        .map(|&n| {
            let p = rayon::ThreadPoolBuilder::new().num_threads(n).build();
            (n, p.expect("a pool"))
        })
        .collect();
    for k in [17usize, 65, 130, 300, 700] {
        let n = 2048;
        let mut s = 7 + k as u64;
        let xs: Vec<f64> = (0..n * k).map(|_| 10.0 + lcg(&mut s)).collect();
        for block in [256usize, 0] {
            for (t, pool) in &pools {
                let c = pool.install(|| {
                    let mut c = EwCov::new(k);
                    c.set_block_rows(block);
                    c.set_threads(*t);
                    for r in 0..n {
                        c.update(&xs[r * k..(r + 1) * k], 0.999, 1.0);
                    }
                    c.flush();
                    c
                });
                let h = digest(c.comoments().iter().chain(c.means()));
                println!("ewcov k={k} block={block} threads={t} {h:016x}");
            }
        }
    }
    for k in [50usize, 65, 129, 300, 700] {
        let a = spd(k);
        let mut s = 9u64;
        let b: Vec<f64> = (0..k * 3).map(|_| lcg(&mut s)).collect();
        for (t, pool) in &pools {
            let x = pool.install(|| SpdFactor::of(&a, k).expect("SPD").solve(&b, k, 3));
            println!("solve k={k} pool={t} {:016x}", digest(&x));
        }
    }
    for k in [100usize, 300] {
        let a = spd(k);
        for (t, pool) in &pools {
            let p = pool.install(|| Pca::of(&a, k, 2, None).expect("an eigendecomposition"));
            let h = digest(p.eig.iter().chain(&p.loadings));
            println!("eigen k={k} pool={t} {h:016x}");
        }
    }
    for (n, k) in [
        (1usize, 40usize),
        (7, 40),
        (300, 17),
        (300, 130),
        (1000, 300),
    ] {
        let mut s = 11u64;
        let x: Vec<f64> = (0..n * k).map(|_| lcg(&mut s)).collect();
        let xm = MatRef::from_row_major_slice(&x, n, k);
        let g = pools[0].1.install(|| xm.transpose() * xm);
        let v: Vec<f64> = (0..k * k).map(|ij| g[(ij / k, ij % k)]).collect();
        println!("faer xtx n={n} k={k} {:016x}", digest(&v));
    }
}
