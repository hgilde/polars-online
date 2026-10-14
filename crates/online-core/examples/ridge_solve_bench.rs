//! Task 256 (b): `gramfit::ridge_fits` against 1.0.1's, at k features and
//! T targets, in one process.
//!
//!     cargo run --release -p online-core --example ridge_solve_bench -- [k...]
//!
//! 1.0.1 solved every target at once with faer's blocked
//! `llt::solve::solve_in_place_with_conj` and summed each target's `b' C b`
//! and `m' C⁻¹ m` alone, one chain of additions each; a copy of that code
//! is below (`old_fits`, the intercept-and-no-standardize case this runs).
//! Today's solves the targets together by substitution in each target's own
//! arithmetic (`trisolve.rs`) and sums the two forms for every target at
//! once (`colstats.rs`). Both share the rest: the design, its two Cholesky
//! factorizations and the triangular inverse, timed alone as `shared`, so
//! `rest = call − shared` is the solve and the statistics.
//!
//! Each line: milliseconds a call, the fastest of 7 rounds to 1,500
//! columns, 3 to 5,000 and 1 past, for 1.0.1's and today's, their ratio, and the same for the rest. Before timing, it
//! asserts what the new path is for: target T−1's fit is the same bits
//! fitted alone, among the last 10 and among all 50; and that the two
//! paths agree to rounding.

use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::prelude::*;
use online_core::gramfit::{Design, GramArrays, OwnedGram, Response, RidgeFit, ridge_fits};
use std::time::Instant;

const TARGETS: usize = 50;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A Gram of `k` features behind an intercept (column 0) and 50 targets:
/// co-moments `U Uᵀ / 4 + I` with `U` of rank 4 (near-collinear, positive
/// definite), the rest drawn at random.
fn gram(k: usize) -> OwnedGram {
    let mut s = 7u64;
    let kz = k + 1;
    let u: Vec<f64> = (0..4 * kz).map(|_| lcg(&mut s)).collect();
    let mut comoments = vec![0.0; kz * kz];
    for i in 1..kz {
        for j in 1..=i {
            let v = (0..4).map(|r| u[r * kz + i] * u[r * kz + j]).sum::<f64>() / 4.0
                + if i == j { 1.0 } else { 0.0 };
            comoments[i * kz + j] = v;
            comoments[j * kz + i] = v;
        }
    }
    let per_target = |s: &mut u64, scale: f64| -> Vec<f64> {
        (0..TARGETS * kz).map(|_| scale * lcg(s)).collect()
    };
    OwnedGram {
        k: kz,
        weight_sum: 2000.0,
        means: (0..kz)
            .map(|i| if i == 0 { 1.0 } else { lcg(&mut s) })
            .collect(),
        comoments,
        cross_moments: per_target(&mut s, 1.0),
        means_by_target: per_target(&mut s, 1.0),
        cross_centred: per_target(&mut s, 0.5),
        target_weights: vec![2000.0; TARGETS],
        target_means: (0..TARGETS).map(|_| lcg(&mut s)).collect(),
        target_vars: vec![1e3; TARGETS],
        target_n_kish: vec![2000.0 + 4.0 * k as f64; TARGETS],
        within: None,
    }
}

fn cholesky(a: &[f64], n: usize) -> Option<Mat<f64>> {
    let mut l = Mat::from_fn(n, n, |i, j| if i >= j { a[i * n + j] } else { 0.0 });
    let mut mem = MemBuffer::new(llt::factor::cholesky_in_place_scratch::<f64>(
        n,
        Par::Seq,
        Default::default(),
    ));
    llt::factor::cholesky_in_place(
        l.as_mut(),
        Default::default(),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .ok()?;
    Some(l)
}

fn inverse_factor(c: &[f64], n: usize) -> Option<Mat<f64>> {
    let l = cholesky(c, n)?;
    let mut inv = Mat::<f64>::zeros(n, n);
    faer::linalg::triangular_inverse::invert_lower_triangular(inv.as_mut(), l.as_ref(), Par::Seq);
    Some(inv)
}

fn inverse_diagonal(inv: &Mat<f64>, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i..n).map(|j| inv[(j, i)] * inv[(j, i)]).sum())
        .collect()
}

/// What both paths share: the design, `A + ρ I`'s factor, `A`'s inverse
/// factor and its diagonal.
fn shared(g: &GramArrays<'_>, slots: &[usize], ridge: f64) -> f64 {
    let n = slots.len();
    let design = Design::of(g, slots, Some(0));
    let mut a = design.a.clone();
    for p in 0..n {
        a[p * n + p] += ridge;
    }
    let l = cholesky(&a, n).expect("positive definite");
    let inv = inverse_factor(&design.a, n).expect("positive definite");
    let d = inverse_diagonal(&inv, n);
    l[(n - 1, n - 1)] + d[n - 1]
}

/// 1.0.1's `ridge_fits`, with an intercept and without `standardize`.
fn old_fits(g: &GramArrays<'_>, targets: &[usize], slots: &[usize], ridge: f64) -> Vec<RidgeFit> {
    let k = g.k;
    let n = slots.len();
    let design = Design::of(g, slots, Some(0));
    let mut a = design.a.clone();
    for p in 0..n {
        a[p * n + p] += ridge;
    }
    let l = cholesky(&a, n).expect("positive definite");
    let resps: Vec<Response> = targets
        .iter()
        .map(|&t| Response::of(g, t, slots, Some(0)))
        .collect();
    let mut x = Mat::from_fn(n, resps.len(), |r, c| resps[c].rhs[r]);
    let mut mem = MemBuffer::new(llt::solve::solve_in_place_scratch::<f64>(
        n,
        resps.len(),
        Par::Seq,
    ));
    llt::solve::solve_in_place_with_conj(
        l.as_ref(),
        faer::Conj::No,
        x.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
    );
    let c = &design.a;
    let mut inv: Option<(Mat<f64>, Vec<f64>)> = None;
    targets
        .iter()
        .zip(&resps)
        .enumerate()
        .map(|(col, (&t, resp))| {
            let mut coef = vec![0.0; k];
            let mut fit = 0.0;
            let b: Vec<f64> = (0..n).map(|p| x[(p, col)]).collect();
            for (p, &i) in slots.iter().enumerate() {
                coef[i] = b[p];
                fit += b[p] * resp.means[p];
            }
            coef[0] = resp.ybar - fit;
            let var_y = g.target_vars[t];
            let nkish = g.target_n_kish[t];
            let bc: f64 = b.iter().zip(&resp.rhs).map(|(u, v)| u * v).sum();
            let cb: f64 = (0..n)
                .map(|p| b[p] * (0..n).map(|q| c[p * n + q] * b[q]).sum::<f64>())
                .sum();
            let resid_var = (var_y - 2.0 * bc + cb).max(0.0);
            let sigma2 = resid_var * nkish / (nkish - (n + 1) as f64);
            let (l_inv, diag) = inv.get_or_insert_with(|| {
                let l_inv = inverse_factor(c, n).expect("positive definite");
                let diag = inverse_diagonal(&l_inv, n);
                (l_inv, diag)
            });
            let mut se = vec![f64::NAN; k];
            let mut tstat = vec![f64::NAN; k];
            for (p, &i) in slots.iter().enumerate() {
                let v = (diag[p].max(0.0) * sigma2 / nkish).sqrt();
                se[i] = v;
                tstat[i] = coef[i] / v;
            }
            let m = &resp.means;
            let q: f64 = (0..n)
                .map(|j| (0..=j).map(|i| l_inv[(j, i)] * m[i]).sum::<f64>().powi(2))
                .sum();
            se[0] = ((1.0 + q) * sigma2 / nkish).sqrt();
            tstat[0] = coef[0] / se[0];
            RidgeFit {
                coef,
                se,
                t: tstat,
                resid_var,
                sigma2,
                r2: 1.0 - resid_var / var_y,
                n: nkish,
            }
        })
        .collect()
}

fn bits(f: &RidgeFit) -> Vec<u64> {
    f.coef
        .iter()
        .chain(&f.se)
        .chain(&f.t)
        .chain([&f.resid_var, &f.sigma2, &f.r2])
        .map(|v| v.to_bits())
        .collect()
}

/// The largest difference between two fits, relative to each vector's
/// largest entry.
fn apart(a: &RidgeFit, b: &RidgeFit) -> f64 {
    let rel = |x: &[f64], y: &[f64]| {
        let top = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        x.iter()
            .zip(y)
            .fold(0.0f64, |m, (u, v)| m.max((u - v).abs() / top))
    };
    rel(&a.coef, &b.coef).max(rel(&a.se, &b.se))
}

/// Milliseconds `f` takes.
fn time(f: impl FnOnce()) -> f64 {
    let t0 = Instant::now();
    f();
    t0.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let ks: Vec<usize> = {
        let args: Vec<usize> = std::env::args()
            .skip(1)
            .map(|a| a.parse().unwrap())
            .collect();
        if args.is_empty() {
            vec![500, 1500, 3000, 5000]
        } else {
            args
        }
    };
    let ridge = 1.0;
    println!(
        "{:>6} {:>3} {:>11} {:>11} {:>6} {:>10} {:>10} {:>10} {:>6}",
        "k", "T", "1.0.1 ms", "now ms", "ratio", "shared ms", "rest 1.0.1", "rest now", "ratio"
    );
    for k in ks {
        let reps = if k <= 1500 {
            3
        } else if k <= 5000 {
            2
        } else {
            1
        };
        let g = gram(k);
        let g = g.arrays();
        let slots: Vec<usize> = (1..=k).collect();
        let sets: [Vec<usize>; 3] = [
            vec![TARGETS - 1],
            (TARGETS - 10..TARGETS).collect(),
            (0..TARGETS).collect(),
        ];
        // What the new path is for, before it is timed: the last target's
        // bits alone, among 10 and among 50.
        let fits: Vec<Vec<RidgeFit>> = sets
            .iter()
            .map(|ts| ridge_fits(&g, ts, &slots, Some(0), ridge, false))
            .collect();
        let last = bits(&fits[0][0]);
        assert_eq!(
            bits(fits[1].last().unwrap()),
            last,
            "k={k}: T=10 moved a neighbour's bits"
        );
        assert_eq!(
            bits(fits[2].last().unwrap()),
            last,
            "k={k}: T=50 moved a neighbour's bits"
        );
        for (a, b) in fits[1].iter().zip(&fits[2][TARGETS - 10..]) {
            assert_eq!(bits(a), bits(b), "k={k}: T=10 and T=50 disagree");
        }
        let old = old_fits(&g, &sets[2], &slots, ridge);
        let gap = old
            .iter()
            .zip(&fits[2])
            .fold(0.0f64, |m, (a, b)| m.max(apart(a, b)));
        assert!(gap < 1e-9, "k={k}: 1.0.1's fits and today's {gap:e} apart");
        let old_last = bits(&old_fits(&g, &sets[0], &slots, ridge)[0]);
        let old_moves = old_last != bits(old.last().unwrap());
        println!(
            "# k={k}: a target's bits alone = among 10 = among 50 (asserted); today's within {gap:.1e} of 1.0.1's; 1.0.1's moved with its neighbours: {old_moves}"
        );
        // Each round times the shared part, 1.0.1's call and today's in
        // turn, so a slow moment on a busy machine lands on all three; the
        // fastest of each is kept, `shared` over every target count.
        let mut sh = f64::INFINITY;
        let mut rows = Vec::new();
        for ts in &sets {
            let (mut o, mut nw) = (f64::INFINITY, f64::INFINITY);
            for _ in 0..reps {
                sh = sh.min(time(|| {
                    std::hint::black_box(shared(&g, &slots, ridge));
                }));
                o = o.min(time(|| {
                    std::hint::black_box(old_fits(&g, ts, &slots, ridge));
                }));
                nw = nw.min(time(|| {
                    std::hint::black_box(ridge_fits(&g, ts, &slots, Some(0), ridge, false));
                }));
            }
            rows.push((ts.len(), o, nw));
        }
        for (t, o, nw) in rows {
            // The rest is a difference of two timings: below the noise of
            // the shared part it says nothing, and is left out.
            let rest = |v: f64| {
                if v > sh {
                    format!("{:.2}", v - sh)
                } else {
                    "-".into()
                }
            };
            let ratio = if o > sh && nw > sh {
                format!("{:.2}", (nw - sh) / (o - sh))
            } else {
                "-".into()
            };
            println!(
                "{k:>6} {t:>3} {o:>11.2} {nw:>11.2} {:>6.2} {sh:>10.2} {:>10} {:>10} {ratio:>6}",
                nw / o,
                rest(o),
                rest(nw),
            );
        }
    }
}
