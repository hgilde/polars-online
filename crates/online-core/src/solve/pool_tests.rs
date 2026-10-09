//! A solve's bits do not follow the size of the pool it runs in
//! (docs/PLAN.md task 231). faer's own factorizations run at its global
//! parallelism, `Par::rayon(0)`, which is the current rayon pool's size, and
//! from `k = 512` their last bits followed it: a wide `ewridge` gave other
//! coefficients under another `POLARS_ONLINE_MAX_THREADS`, or on a machine
//! with another core count. Each case here runs the same inputs in pools of
//! 1, 3, 8 and 14 threads and compares the results to the bit.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A positive definite `k × k`, row-major: the Gram of `k + 7` random rows
/// on an offset, with a small ridge, as `ewridge` solves.
pub(super) fn spd(k: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let n = k + 7;
    let x: Vec<f64> = (0..n * k).map(|_| 0.5 + lcg(&mut s)).collect();
    let xm = faer::MatRef::from_row_major_slice(&x, n, k);
    let g = xm.transpose() * xm;
    let mut a = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            a[i * k + j] = g[(i, j)] / n as f64 + if i == j { 1e-3 } else { 0.0 };
        }
    }
    for i in 0..k {
        for j in 0..i {
            let v = a[i * k + j];
            a[j * k + i] = v;
        }
    }
    a
}

pub(super) fn pools() -> Vec<rayon::ThreadPool> {
    [1usize, 3, 8, 14]
        .iter()
        .map(|&n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("a test pool")
        })
        .collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The factor, `solve_spd`, a factor's solve and its quadratic forms, with
/// one right-hand side and with two, at the widths where faer went
/// parallel. Each is compared on its own, so a failure names it.
#[test]
#[ignore = "extended: three factorizations up to k = 1,000 in four pools, in a debug build"]
fn a_solve_is_the_same_in_every_pool() {
    let pools = pools();
    let mut moved = Vec::new();
    for k in [512usize, 730, 1000] {
        let a = spd(k, 3 + k as u64);
        let mut s = 7u64;
        let b: Vec<f64> = (0..2 * k).map(|_| lcg(&mut s)).collect();
        let b1 = &b[..k];
        let run = |p: &rayon::ThreadPool| -> Vec<(&str, Vec<u64>)> {
            p.install(|| {
                let f = SpdFactor::of(&a, k).expect("positive definite");
                let l: Vec<f64> = (0..k * k).map(|ij| f.l[(ij / k, ij % k)]).collect();
                vec![
                    ("the factor", bits(&l)),
                    ("ln det", vec![f.log_det().to_bits()]),
                    (
                        "solve_spd, one rhs",
                        bits(&solve_spd(&a, b1, k, 1).unwrap().0),
                    ),
                    ("solve_spd, two", bits(&solve_spd(&a, &b, k, 2).unwrap().0)),
                    ("solve, one rhs", bits(&f.solve(b1, k, 1))),
                    ("solve, two", bits(&f.solve(&b, k, 2))),
                    ("quad_forms, one rhs", bits(&f.quad_forms(b1, k, 1))),
                    ("quad_forms, two", bits(&f.quad_forms(&b, k, 2))),
                ]
            })
        };
        let one = run(&pools[0]);

        for (p, n) in pools.iter().zip([1, 3, 8, 14]).skip(1) {
            for ((what, got), (_, want)) in run(p).iter().zip(&one) {
                if got != want {
                    moved.push(format!("k = {k}, pool of {n}: {what}"));
                }
            }
        }
    }
    assert!(moved.is_empty(), "moved a bit: {moved:#?}");
}

/// The inverse's diagonal, `k` right-hand sides at once.
#[test]
#[ignore = "extended: a 512-wide inverse in four pools, in a debug build"]
fn an_inverse_diagonal_is_the_same_in_every_pool() {
    let pools = pools();
    let k = 512;
    let a = spd(k, 11);
    let f = SpdFactor::of(&a, k).expect("positive definite");
    let one = bits(&pools[0].install(|| f.inverse_diagonal(k)));
    for p in &pools[1..] {
        assert_eq!(bits(&p.install(|| f.inverse_diagonal(k))), one);
    }
}

/// faer's self-adjoint eigensolver, which `ew_cov`'s principal components
/// read (`Pca::of`) and `rcov`'s repair of a covariance that is not
/// positive semidefinite, at the widths where it could go parallel.
#[test]
#[ignore = "extended: eigendecompositions in four pools, in a debug build"]
fn principal_components_are_the_same_in_every_pool() {
    let pools = pools();
    let mut moved = Vec::new();
    for k in [300usize, 512, 730] {
        let a = spd(k, 5 + k as u64);
        let run = |p: &rayon::ThreadPool| {
            p.install(|| {
                let pca = crate::Pca::of(&a, k, 3, None).expect("a decomposition");
                let mut out = bits(&pca.eig);
                out.extend(bits(&pca.loadings));
                out
            })
        };
        let one = run(&pools[0]);
        for (p, n) in pools.iter().zip([1, 3, 8, 14]).skip(1) {
            if run(p) != one {
                moved.push(format!("k = {k}, pool of {n}"));
            }
        }
    }
    assert!(moved.is_empty(), "moved a bit: {moved:#?}");
}

/// The cut factor is faer's own sequential factor to the bit, at the cut
/// that ships, where faer blocks (past 64) and where the trailing product
/// is more than one piece (past 256): on aarch64 the pieces are the calls
/// faer makes. Not on x86-64, where a product's pieces need not be the
/// single kernel call's (`crate::pieces`); the pool tests hold there too.
#[cfg(not(target_arch = "x86_64"))]
#[test]
fn the_cut_factor_is_faers_own_to_the_bit() {
    for k in [1usize, 64, 65, 100, 129, 300] {
        assert_eq!(factor_bits(k, 4, (LEAF, TILE)), seq_bits(k), "k = {k}");
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[test]
#[ignore = "extended: two factorizations at k = 512 and 1,000 in a debug build"]
fn the_cut_factor_is_faers_own_to_the_bit_wide() {
    for k in [512usize, 1000] {
        assert_eq!(factor_bits(k, 4, (LEAF, TILE)), seq_bits(k), "k = {k}");
    }
}

/// A fine cut makes a narrow matrix many pieces and many bands, so a cheap
/// case shows the thread count deciding nothing (a debug build).
#[test]
fn a_finely_cut_factor_is_the_same_at_every_thread_count() {
    let pools = pools();
    for k in [130usize, 200] {
        let one = factor_bits(k, 1, (16, 8));
        for (p, n) in pools.iter().zip([1, 3, 8, 14]) {
            assert_eq!(
                p.install(|| factor_bits(k, n, (16, 8))),
                one,
                "k = {k}, {n} threads"
            );
        }
    }
}

fn lower(k: usize) -> Mat<f64> {
    let a = spd(k, 3 + k as u64);
    Mat::from_fn(k, k, |i, j| if i < j { 0.0 } else { a[i * k + j] })
}

fn lower_bits(l: &Mat<f64>) -> Vec<u64> {
    let k = l.nrows();
    (0..k)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .map(|(i, j)| l[(i, j)].to_bits())
        .collect()
}

fn factor_bits(k: usize, threads: usize, cut: (usize, usize)) -> Vec<u64> {
    let mut l = lower(k);
    cholesky_in_place(l.as_mut(), threads, cut).expect("positive definite");
    lower_bits(&l)
}

fn seq_bits(k: usize) -> Vec<u64> {
    let mut l = lower(k);
    cholesky_seq(l.as_mut()).expect("positive definite");
    lower_bits(&l)
}
