//! Task 231: a solve's cost and bits in pools of 1, 8 and 14 threads,
//! faer's own calls at its global parallelism (what `solve.rs` called
//! before) against the cut by shape it calls now.
//!
//!     cargo run --release -p online-core --example solve_threads_bench -- [k...]
//!
//! Per width: the factorization plus a one-column solve, and an
//! eigendecomposition, in milliseconds (the fastest of five), and whether
//! each result's bits equal the pool of 1's, and the old pool of 1's.

use faer::Conj;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::prelude::*;
use online_core::{Pca, SpdFactor};
use std::time::Instant;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn spd(k: usize) -> Vec<f64> {
    let mut s = 3u64;
    let n = k + 7;
    let x: Vec<f64> = (0..n * k).map(|_| 0.5 + lcg(&mut s)).collect();
    let xm = MatRef::from_row_major_slice(&x, n, k);
    let g = xm.transpose() * xm;
    (0..k * k)
        .map(|ij| g[(ij / k, ij % k)] / n as f64 + if ij / k == ij % k { 1e-3 } else { 0.0 })
        .collect()
}

/// What `solve.rs` did before task 231: faer at the global parallelism.
fn old_solve(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let par = faer::get_global_parallelism();
    let mut l = Mat::from_fn(k, k, |i, j| if i < j { 0.0 } else { a[i * k + j] });
    let mut mem = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
        k,
        par,
        Default::default(),
    ));
    llt::factor::cholesky_in_place(
        l.as_mut(),
        Default::default(),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .unwrap();
    let mut x = Mat::from_fn(k, 1, |i, _| b[i]);
    let mut mem = MemBuffer::new(llt::solve::solve_in_place_scratch::<f64>(k, 1, par));
    llt::solve::solve_in_place_with_conj(
        l.as_ref(),
        Conj::No,
        x.as_mut(),
        par,
        MemStack::new(&mut mem),
    );
    (0..k).map(|i| x[(i, 0)]).collect()
}

fn new_solve(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    SpdFactor::of(a, k).unwrap().solve(b, k, 1)
}

fn old_eigen(a: &[f64], k: usize) -> Vec<f64> {
    let m = Mat::from_fn(k, k, |i, j| a[i * k + j]);
    let e = m.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let mut out: Vec<f64> = (0..k).map(|i| e.S()[i]).collect();
    out.extend((0..k).map(|i| e.U()[(i, k - 1)]));
    out
}

fn new_eigen(a: &[f64], k: usize) -> Vec<f64> {
    let p = Pca::of(a, k, 1, None).unwrap();
    let mut out = p.eig.clone();
    out.extend(p.loadings.iter().copied());
    out
}

fn best_ms<F: FnMut() -> Vec<f64>>(mut f: F) -> (f64, Vec<f64>) {
    let mut best = f64::INFINITY;
    let mut out = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        out = f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, out)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn main() {
    let ks: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let ks = if ks.is_empty() {
        vec![512, 1000, 2000, 4000]
    } else {
        ks
    };
    let pools: Vec<(usize, rayon::ThreadPool)> = [1usize, 8, 14]
        .iter()
        .map(|&n| {
            (
                n,
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .unwrap(),
            )
        })
        .collect();
    for &k in &ks {
        let a = spd(k);
        let mut s = 9u64;
        let b: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
        let (mut old1, mut new1) = (Vec::new(), Vec::new());
        for (n, p) in &pools {
            let (to, o) = p.install(|| best_ms(|| old_solve(&a, &b, k)));
            let (tn, x) = p.install(|| best_ms(|| new_solve(&a, &b, k)));
            if *n == 1 {
                old1 = bits(&o);
                new1 = bits(&x);
            }
            let worst = o
                .iter()
                .zip(&x)
                .map(|(p, q)| ((p - q) / p.abs().max(1e-300)).abs())
                .fold(0.0f64, f64::max);
            println!(
                "solve k={k:>5} pool {n:>2}: old {to:8.2} ms, new {tn:8.2} ms | old = old pool 1: {} | new = new pool 1: {} | new = old pool 1: {} | worst rel new vs old here {worst:.1e}",
                bits(&o) == old1,
                bits(&x) == new1,
                bits(&x) == old1
            );
        }
        if k > 2000 {
            continue;
        }
        let (mut old1, mut new1) = (Vec::new(), Vec::new());
        for (n, p) in &pools {
            let (to, o) = p.install(|| best_ms(|| old_eigen(&a, k)));
            let (tn, x) = p.install(|| best_ms(|| new_eigen(&a, k)));
            // The same quantities: the eigenvalues, then the top vector, its
            // sign as `Pca` sets it.
            let mut o_top = vec![o[k - 1]];
            let sign = if x[1..].iter().zip(&o[k..]).map(|(p, q)| p * q).sum::<f64>() < 0.0 {
                -1.0
            } else {
                1.0
            };
            o_top.extend(o[k..].iter().map(|v| sign * v));
            if *n == 1 {
                old1 = bits(&o_top);
                new1 = bits(&x);
            }
            println!(
                "eigen k={k:>5} pool {n:>2}: old {to:8.2} ms, new {tn:8.2} ms | old = old pool 1: {} | new = new pool 1: {} | new = old pool 1: {}",
                bits(&o_top) == old1,
                bits(&x) == new1,
                bits(&x) == old1
            );
        }
    }
}
