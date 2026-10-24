//! [`super::trisolve`] (docs/PLAN.md task 256): each column's bits its own,
//! whatever columns are solved beside it; the arithmetic the textbook
//! recurrences', term by term; and the solve against faer's partial-pivot
//! LU (`crate::oracle`), which shares no arithmetic with it, within the
//! forward-error bound Cholesky's backward error gives.

use super::trisolve::{backward, forward, solve_llt};
use super::*;
use crate::SplitMix64;
use crate::oracle::{self, U, gamma};
use faer::Mat;
use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;

fn normal(rng: &mut SplitMix64) -> f64 {
    let u = 1.0 - rng.uniform();
    let v = rng.uniform();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// A symmetric positive definite `A = GᵀG + ρ I`, row-major `n × n`, with
/// `G` of `rows` rows, its columns a shared factor plus noise of scale
/// `noise` (near-collinear when small, and rank-deficient but for `ρ` when
/// `rows < n`), `ρ` `rho` times `GᵀG`'s mean diagonal, and an upper bound
/// on `A`'s 2-norm condition number: `λ_max ≤ trace(GᵀG) = ‖G‖_F²` and
/// `λ_min ≥ ρ`.
fn spd(seed: u64, n: usize, rows: usize, noise: f64, rho: f64) -> (Vec<f64>, f64) {
    let mut rng = SplitMix64::new(seed);
    let mut g = vec![0.0; rows * n];
    for r in 0..rows {
        let shared = normal(&mut rng);
        for c in 0..n {
            g[r * n + c] = shared * (1.0 + 0.1 * c as f64 / n as f64) + noise * normal(&mut rng);
        }
    }
    let frob: f64 = g.iter().map(|v| v * v).sum();
    let rho = rho * frob / n as f64;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v: f64 = (0..rows).map(|r| g[r * n + i] * g[r * n + j]).sum();
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] += rho;
    }
    (a, (frob + rho) / rho)
}

/// `A`'s lower Cholesky factor by faer, as `ridge_fits` factors it.
fn factor(a: &[f64], n: usize) -> Mat<f64> {
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
    .expect("positive definite");
    l
}

/// A row-major `n × t` right-hand side of standard normals.
fn rhs(seed: u64, n: usize, t: usize) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    (0..n * t).map(|_| normal(&mut rng)).collect()
}

/// Columns `cols` of a row-major `n × t`, row-major `n × cols.len()`.
fn columns(x: &[f64], t: usize, cols: &[usize]) -> Vec<f64> {
    x.chunks_exact(t)
        .flat_map(|row| cols.iter().map(move |&c| row[c]))
        .collect()
}

fn column(x: &[f64], t: usize, c: usize) -> Vec<f64> {
    columns(x, t, &[c])
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn solved(l: &Mat<f64>, b: &[f64], t: usize) -> Vec<f64> {
    let mut x = b.to_vec();
    solve_llt(l, &mut x, t);
    x
}

/// Every order the tests visit: the edges of the panels (16 rows) and of
/// the tiles' rows (16, 8, 4 and 1), and past them.
const ORDERS: [usize; 18] = [
    1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 20, 21, 31, 33, 64, 65,
];

#[test]
fn a_column_solved_beside_others_is_the_column_solved_alone() {
    let mut rng = SplitMix64::new(256);
    for (case, &n) in ORDERS.iter().enumerate() {
        let (a, _) = spd(case as u64, n, n + 3, 0.3, 0.5);
        let l = factor(&a, n);
        // The tiles are 8, 4 and 1 columns wide on aarch64, 4 and 1 elsewhere.
        for t in [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 17, 31, 64] {
            let b = rhs(1000 + case as u64, n, t);
            let all = solved(&l, &b, t);
            for c in 0..t {
                let alone = solved(&l, &column(&b, t, c), 1);
                assert_eq!(bits(&alone), bits(&column(&all, t, c)), "n={n} t={t} c={c}");
            }
            // Random subsets in random orders, each column's bits those it
            // has among all of them.
            for _ in 0..4 {
                let mut cols: Vec<usize> = (0..t).filter(|_| rng.uniform() < 0.6).collect();
                if cols.is_empty() {
                    cols.push(t - 1);
                }
                for i in (1..cols.len()).rev() {
                    let j = (rng.uniform() * (i + 1) as f64) as usize;
                    cols.swap(i, j.min(i));
                }
                let some = solved(&l, &columns(&b, t, &cols), cols.len());
                assert_eq!(
                    bits(&some),
                    bits(&columns(&all, t, &cols)),
                    "n={n} t={t} {cols:?}"
                );
            }
        }
    }
}

/// The recurrences of the module's docs, one column, one term at a time.
fn by_recurrence(l: &Mat<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for j in 0..i {
            s -= l[(i, j)] * y[j];
        }
        y[i] = s / l[(i, i)];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1..n).rev() {
            s -= l[(j, i)] * y[j];
        }
        y[i] = s / l[(i, i)];
    }
    y
}

#[test]
fn each_column_is_the_textbook_recurrence_to_the_bit() {
    for (case, &n) in ORDERS.iter().enumerate() {
        let (a, _) = spd(50 + case as u64, n, n / 2 + 1, 0.05, 1e-3);
        let l = factor(&a, n);
        let t = 11;
        let b = rhs(60 + case as u64, n, t);
        let x = solved(&l, &b, t);
        for c in 0..t {
            let want = by_recurrence(&l, &column(&b, t, c));
            assert_eq!(bits(&column(&x, t, c)), bits(&want), "n={n} c={c}");
        }
    }
}

/// `max_i |b_i − (T y)_i| / (|T| |y|)_i` over the rows of a triangular
/// `T y = b`, the residual summed exactly ([`oracle::dot`]): the
/// componentwise backward error of a substitution, which Higham's Theorem
/// 8.5 holds to `γ_n` whatever `T`'s condition.
fn backward_error(n: usize, entry: impl Fn(usize, usize) -> f64, y: &[f64], b: &[f64]) -> f64 {
    (0..n)
        .map(|i| {
            let terms: Vec<usize> = (0..n).filter(|&j| entry(i, j) != 0.0).collect();
            let r = oracle::dot(
                std::iter::once((b[i], 1.0)).chain(terms.iter().map(|&j| (-entry(i, j), y[j]))),
            );
            let scale: f64 = terms.iter().map(|&j| (entry(i, j) * y[j]).abs()).sum();
            r.abs() / scale
        })
        .fold(0.0, f64::max)
}

#[test]
fn each_substitution_is_backward_stable_row_by_row() {
    for (case, &n) in ORDERS.iter().chain(&[130]).enumerate() {
        for (noise, rho) in [(1.0, 1.0), (1e-4, 1e-9)] {
            let (a, _) = spd(70 + case as u64, n, n / 2 + 2, noise, rho);
            let l = factor(&a, n);
            let t = 5;
            let b = rhs(80 + case as u64, n, t);
            let mut y = b.clone();
            forward(&l, &mut y, t);
            let mut x = y.clone();
            backward(&l, &mut x, t);
            let lower = |i: usize, j: usize| if j <= i { l[(i, j)] } else { 0.0 };
            let upper = |i: usize, j: usize| if j >= i { l[(j, i)] } else { 0.0 };
            for c in 0..t {
                let (bc, yc, xc) = (column(&b, t, c), column(&y, t, c), column(&x, t, c));
                // `(1 + u)`: the exact residual is rounded once.
                let bound = gamma(n) * (1.0 + U);
                let ef = backward_error(n, lower, &yc, &bc);
                let eb = backward_error(n, upper, &xc, &yc);
                assert!(ef <= bound, "forward n={n} c={c}: {ef:e} > {bound:e}");
                assert!(eb <= bound, "back n={n} c={c}: {eb:e} > {bound:e}");
            }
        }
    }
}

/// Twice Cholesky's forward bound: `(A + ΔA) x̂ = b` with `|ΔA| ≤
/// γ_{3n+1} |L||Lᵀ|` (Higham, Theorem 10.4) and `‖|L||Lᵀ|‖₂ ≤ n‖A‖₂`, so
/// the relative error is at most `κ n γ_{3n+1}` to first order, and the
/// LU's own as much again.
fn forward_bound(n: usize, kappa: f64) -> f64 {
    2.0 * kappa * n as f64 * gamma(3 * n + 1)
}

/// `ρ`'s scale ([`spd`]) that makes the bound bite at 0.05: the most
/// ill-conditioned system it still says something about.
fn ill(n: usize) -> f64 {
    n as f64 * forward_bound(n, 1.0) / 0.05
}

/// The rows of [`spd`]'s `G` for [`against_lu`]'s systems of order `n`:
/// fewer than `n` (but one at `n = 1`), so `GᵀG` is singular, `ρ` alone
/// sets the least eigenvalue and the condition number is its bound. At `n
/// ≤ 6`, `n/2 + 3` rows made `GᵀG` full rank, and the ill leg's `κ` was
/// 2.1e9 at `n = 4`, 2,000 times below the bound it was meant to reach
/// (review round 8, A-3; docs/PLAN.md task 269).
fn lu_rows(n: usize) -> usize {
    (n / 2 + 3).min(n.saturating_sub(1)).max(1)
}

/// `‖x̂ − x‖₂ / ‖x‖₂` for each column against faer's partial-pivot LU on
/// `A` itself, held to [`forward_bound`]; the worst ratio of the two.
fn against_lu(seed: u64, n: usize, t: usize, noise: f64, rho: f64) -> f64 {
    let (a, kappa) = spd(seed, n, lu_rows(n), noise, rho);
    let l = factor(&a, n);
    let b = rhs(seed + 1, n, t);
    let x = solved(&l, &b, t);
    let bound = forward_bound(n, kappa);
    assert!(bound < 0.5, "a vacuous bound: n={n} κ ≤ {kappa:e}");
    let lu = oracle::solve_columns(&a, &b, t);
    let mut worst: f64 = 0.0;
    for c in 0..t {
        let want = column(&lu, t, c);
        let got = column(&x, t, c);
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).powi(2))
            .sum::<f64>()
            .sqrt();
        let size: f64 = want.iter().map(|w| w * w).sum::<f64>().sqrt();
        let rel = err / size;
        assert!(
            rel <= bound,
            "n={n} t={t} c={c}: {rel:e} > {bound:e} (κ ≤ {kappa:e})"
        );
        worst = worst.max(rel / bound);
    }
    worst
}

#[test]
fn the_solve_is_the_lu_solve_within_its_forward_error() {
    let mut worst = [0.0f64; 2];
    for (case, &n) in ORDERS.iter().chain(&[130, 200]).enumerate() {
        for t in [1, 2, 3, 8, 13, 64] {
            let seed = 900 + 100 * case as u64 + t as u64;
            // Well conditioned, then near-collinear columns under a ridge
            // near 0, `κ` as large as the bound allows (4e12 at n = 4, 2e9
            // at 200: [`the_ill_systems_are_as_ill_as_their_bound`]).
            worst[0] = worst[0].max(against_lu(seed, n, t, 1.0, 1.0));
            worst[1] = worst[1].max(against_lu(seed, n, t, 1e-4, ill(n)));
        }
    }
    eprintln!(
        "worst error / bound: well {:e}, ill {:e}",
        worst[0], worst[1]
    );
}

/// The ill leg of [`the_solve_is_the_lu_solve_within_its_forward_error`]
/// is as ill as the bound it is held to says, at every order it visits: its
/// condition number, from faer's self-adjoint eigensolver, at least half
/// [`spd`]'s bound on it, which the forward bound is computed from
/// (review round 8, A-3; docs/PLAN.md task 269). With `n/2 + 3` rows, the
/// orders up to 6 sat 50 to 2,000 times below it.
#[test]
fn the_ill_systems_are_as_ill_as_their_bound() {
    for (case, &n) in ORDERS.iter().chain(&[130, 200]).enumerate() {
        if n < 2 {
            // A system of one is as well conditioned as a number.
            continue;
        }
        let seed = 900 + 100 * case as u64 + 8;
        let (a, bound) = spd(seed, n, lu_rows(n), 1e-4, ill(n));
        let (evals, _) = oracle::sym_eigen(&a);
        let top = evals.iter().copied().fold(f64::MIN, f64::max);
        let least = evals.iter().copied().fold(f64::MAX, f64::min);
        let kappa = top / least;
        assert!(
            // The eigensolver's least eigenvalue is `ρ` to `ε·λ_max`, a
            // few thousandths of it at the smallest orders.
            kappa <= 1.05 * bound && kappa >= 0.5 * bound,
            "n={n}: κ {kappa:e} against its bound {bound:e}"
        );
    }
}

#[test]
#[ignore = "extended: orders 511 and 1,500, faer's LU at each (90 s, debug build)"]
fn the_solve_is_the_lu_solve_within_its_forward_error_up_to_1500() {
    for (case, n) in [511, 1500].into_iter().enumerate() {
        for t in [1, 64] {
            let seed = 7000 + 100 * case as u64 + t as u64;
            let well = against_lu(seed, n, t, 1.0, 1.0);
            let ill = against_lu(seed, n, t, 1e-4, ill(n));
            eprintln!("n={n} t={t} worst error / bound: well {well:e}, ill {ill:e}");
        }
    }
}

#[test]
fn no_columns_and_no_rows_are_nothing_to_solve() {
    let (a, _) = spd(1, 3, 4, 0.5, 1.0);
    let l = factor(&a, 3);
    let mut x: Vec<f64> = Vec::new();
    solve_llt(&l, &mut x, 0);
    assert!(x.is_empty());
    let l0 = Mat::<f64>::zeros(0, 0);
    solve_llt(&l0, &mut x, 4);
    assert!(x.is_empty());
}

/// A Gram of `targets` targets on `k` features and the intercept (column
/// 0), every row at weight 1, two-pass; the targets missing on rows of
/// their own, so each has its own means and Kish size.
fn gram(seed: u64, rows: usize, k: usize, targets: usize) -> OwnedGram {
    let mut rng = SplitMix64::new(seed);
    let mut z: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<Vec<Option<f64>>> = Vec::new();
    for i in 0..rows {
        let base = normal(&mut rng);
        let mut row = vec![1.0];
        row.extend((0..k).map(|j| 0.7 * base + normal(&mut rng) + j as f64 * 0.1));
        let ys = (0..targets)
            .map(|t| {
                let v = row[1 + t % k] - 0.5 * row[1 + (3 * t + 1) % k] + normal(&mut rng);
                (i % (t + 2) != 1).then_some(v)
            })
            .collect();
        z.push(row);
        y.push(ys);
    }
    let kz = k + 1;
    let all: Vec<usize> = (0..rows).collect();
    let mean = |idx: &[usize], f: &dyn Fn(usize) -> f64| -> f64 {
        idx.iter().map(|&i| f(i)).sum::<f64>() / idx.len() as f64
    };
    let means: Vec<f64> = (0..kz).map(|a| mean(&all, &|i| z[i][a])).collect();
    let mut comoments = vec![0.0; kz * kz];
    for a in 0..kz {
        for b in 0..kz {
            comoments[a * kz + b] = mean(&all, &|i| (z[i][a] - means[a]) * (z[i][b] - means[b]));
        }
    }
    let mut g = OwnedGram {
        k: kz,
        weight_sum: rows as f64,
        means,
        comoments,
        cross_moments: Vec::new(),
        means_by_target: Vec::new(),
        cross_centred: Vec::new(),
        target_weights: Vec::new(),
        target_means: Vec::new(),
        target_vars: Vec::new(),
        target_n_kish: Vec::new(),
        within: None,
    };
    // Each target's own rows, by the target's index into every row.
    #[allow(clippy::needless_range_loop)]
    for t in 0..targets {
        let own: Vec<usize> = all.iter().copied().filter(|&i| y[i][t].is_some()).collect();
        let yv = |i: usize| y[i][t].unwrap();
        let ybar = mean(&own, &yv);
        let m: Vec<f64> = (0..kz).map(|a| mean(&own, &|i| z[i][a])).collect();
        for (a, &ma) in m.iter().enumerate() {
            g.cross_moments.push(mean(&own, &|i| z[i][a] * yv(i)));
            g.cross_centred
                .push(mean(&own, &|i| (z[i][a] - ma) * (yv(i) - ybar)));
        }
        g.means_by_target.extend(m);
        g.target_weights.push(own.len() as f64);
        g.target_means.push(ybar);
        g.target_vars.push(mean(&own, &|i| (yv(i) - ybar).powi(2)));
        g.target_n_kish.push(own.len() as f64);
    }
    g
}

fn fit_bits(f: &RidgeFit) -> Vec<u64> {
    let mut v = bits(&f.coef);
    v.extend(bits(&f.se));
    v.extend(bits(&f.t));
    v.extend(bits(&[f.resid_var, f.sigma2, f.r2, f.n]));
    v
}

#[test]
fn a_targets_ridge_fit_is_its_own_whichever_targets_share_the_call() {
    let mut rng = SplitMix64::new(4);
    // 9 slots and 17 on the intercept: a tail panel and a tail tile.
    for (k, targets) in [(9, 13), (17, 21)] {
        let g = gram(k as u64, 300, k, targets);
        let slots: Vec<usize> = (1..=k).collect();
        for (icept, ridge, standardize) in [
            (Some(0), 0.0, false),
            (Some(0), 0.4, true),
            (None, 1e-6, false),
        ] {
            let fit = |ts: &[usize]| ridge_fits(&g.arrays(), ts, &slots, icept, ridge, standardize);
            let alone: Vec<Vec<u64>> = (0..targets).map(|t| fit_bits(&fit(&[t])[0])).collect();
            let every: Vec<usize> = (0..targets).collect();
            let mut picks = vec![every.clone(), every.iter().rev().copied().collect()];
            for _ in 0..12 {
                let mut ts: Vec<usize> = every
                    .iter()
                    .copied()
                    .filter(|_| rng.uniform() < 0.5)
                    .collect();
                // The same target twice is two columns, each its own.
                ts.push((rng.uniform() * targets as f64) as usize % targets);
                for i in (1..ts.len()).rev() {
                    let j = ((rng.uniform() * (i + 1) as f64) as usize).min(i);
                    ts.swap(i, j);
                }
                picks.push(ts);
            }
            for ts in &picks {
                for (f, &t) in fit(ts).iter().zip(ts) {
                    assert_eq!(
                        fit_bits(f),
                        alone[t],
                        "k={k} {icept:?} {ridge} target {t} of {ts:?}"
                    );
                }
            }
            assert!(
                alone
                    .iter()
                    .all(|b| b[1..=k].iter().any(|&c| f64::from_bits(c) != 0.0))
            );
        }
    }
}
