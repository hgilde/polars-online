//! The Gram fits against the definition of the lasso and against faer's LU
//! (`crate::oracle`), which shares no arithmetic with the LARS factor.

use super::*;
use crate::SplitMix64;
use crate::oracle;

/// A Gram in correlation form from `n` rows of `k` columns: column `j` is
/// `z_j + mix · z_0` for standard normal `z`, and `y = Σ_j beta_j x_j +
/// noise`. Returns `(R, d)`, standardized so `R` has a unit diagonal.
pub(super) fn design(
    seed: u64,
    n: usize,
    k: usize,
    mix: f64,
    beta: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut normal = || {
        // Box-Muller from two uniforms in (0, 1].
        let u = 1.0 - rng.uniform();
        let v = rng.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let mut x = vec![0.0; n * k];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let base = normal();
        for j in 0..k {
            x[i * k + j] = normal() + mix * base;
        }
        y[i] = (0..k)
            .map(|j| beta.get(j).copied().unwrap_or(0.0) * x[i * k + j])
            .sum::<f64>()
            + normal();
    }
    standardized(&x, &y, n, k)
}

/// `(R, d)` of rows `x` (`n × k`) and `y`: each column centred and scaled
/// to unit population variance, `y` centred.
pub(super) fn standardized(x: &[f64], y: &[f64], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = x.to_vec();
    for j in 0..k {
        let m = (0..n).map(|i| x[i * k + j]).sum::<f64>() / n as f64;
        let v = (0..n).map(|i| (x[i * k + j] - m).powi(2)).sum::<f64>() / n as f64;
        for i in 0..n {
            x[i * k + j] = (x[i * k + j] - m) / v.sqrt();
        }
    }
    let ym = y.iter().sum::<f64>() / n as f64;
    let mut r = vec![0.0; k * k];
    let mut d = vec![0.0; k];
    for a in 0..k {
        for b in 0..k {
            r[a * k + b] = (0..n).map(|i| x[i * k + a] * x[i * k + b]).sum::<f64>() / n as f64;
        }
        d[a] = (0..n).map(|i| x[i * k + a] * (y[i] - ym)).sum::<f64>() / n as f64;
    }
    (r, d)
}

/// The lasso's optimality conditions at `lambda`, by definition: with
/// `c = d − R b`, `c_j = λ sign(b_j)` where `b_j ≠ 0` and `|c_j| ≤ λ`
/// elsewhere.
pub(super) fn assert_kkt(r: &[f64], d: &[f64], b: &[f64], lambda: f64, tol: f64) {
    let k = d.len();
    for j in 0..k {
        let c = d[j] - (0..k).map(|i| r[j * k + i] * b[i]).sum::<f64>();
        if b[j] != 0.0 {
            assert!(
                (c - lambda * b[j].signum()).abs() <= tol,
                "column {j}: c = {c}, λ = {lambda}, b = {}",
                b[j]
            );
        } else {
            assert!(
                c.abs() <= lambda + tol,
                "column {j}: |c| = {} > λ = {lambda}",
                c.abs()
            );
        }
    }
}

/// `R`'s block on rows `a` and columns `b`, row-major.
fn block(r: &[f64], k: usize, a: &[usize], b: &[usize]) -> Vec<f64> {
    a.iter()
        .flat_map(|&i| b.iter().map(move |&j| r[i * k + j]))
        .collect()
}

/// A bound, per knot and column, on `|d_j − R_j·b − λ s_j|` for the
/// active columns of the path, the residual of the active-set system
/// `R_AA b_A = d_A − λ s_A` the knot's coefficients solve in exact
/// arithmetic (docs/PLAN.md task 274). LARS reaches them by steps, `b ←
/// b + γδ` with `δ` solved from the active set's Cholesky factor, so the
/// residual is what the steps leave, counted from the path's own
/// magnitudes as a running error bound (Higham §3.3) in the module docs'
/// terms:
///
/// - the first knot's columns sit within `TIE·λ_max` of `λ_max`, or, joined
///   by a step within the tie of 0, `|1 − s a|` times that;
/// - a step adds `γ·E|δ|` (`R_AA δ̂ = s + e`, `|e| ≤ E|δ̂|`, the factor
///   solve's backward error, Higham Theorem 10.4), `u|R|(|b| + |γδ|)`
///   (the update's rounding) and `u λ` (the penalty's subtraction);
/// - a column leaving is set to 0 from `|b_l| ≤ TIE·λ_max·|δ_l|` (its step
///   was within the tie of `γ`), which moves every other residual by
///   `|R_il|` times that; alone at its knot its step is `γ`, and it is
///   short of 0 by its step's rounding, `4u|b_l|`;
/// - a column entering does so with `|c_e − s λ|` at most `|1 − s a_e|`
///   times its step's distance from `γ` (the tie, 0 for a column alone at
///   its knot, and `4u` of the step's own rounding), plus the rounding of
///   the `c_e` and `a_e` it was read from;
/// - at a knot where several columns change, the columns staying are
///   solved afresh: their residual is that one solve's, `E|b|` and the
///   right-hand side's rounding, and the entering ones move by the
///   staying ones' change, `|R_eS||R_SS⁻¹|` times their old residuals.
pub(super) fn residual_bounds(r: &[f64], d: &[f64], path: &LarsPath) -> Vec<Vec<f64>> {
    use oracle::{U, abs_times, gamma};
    let k = d.len();
    let tie = lars::TIE * path.penalties[0];
    let mut res = vec![0.0f64; k];
    let first = &path.active[0];
    if first.len() == 1 {
        res[first[0]] = U * d[first[0]].abs();
    } else if !first.is_empty() {
        // Tied at `λ_max`, or joined within the tie of a step of 0, which
        // leaves a column `|1 − s a|` times that short: at most `1 +
        // |R_jA||δ|`, `δ = R_AA⁻¹s` over the knot's columns.
        let raa = block(r, k, first, first);
        let sa: Vec<f64> = first.iter().map(|&j| d[j].signum()).collect();
        let delta = oracle::solve(&raa, &sa);
        for &j in first {
            let a: f64 = first
                .iter()
                .zip(&delta)
                .filter(|(i, _)| **i != j)
                .map(|(&i, di)| (r[j * k + i] * di).abs())
                .sum();
            res[j] = tie * (1.0 + a) + U * d[j].abs();
        }
    }
    let mut out = vec![res.clone()];
    for n in 1..path.len() {
        let (prev, cur) = (&path.active[n - 1], &path.active[n]);
        let (lam0, lam) = (path.penalties[n - 1], path.penalties[n]);
        let step = lam0 - lam;
        let bp = &path.coefs[(n - 1) * k..n * k];
        let bc = &path.coefs[n * k..(n + 1) * k];
        let delta: Vec<f64> = prev.iter().map(|&i| (bc[i] - bp[i]) / step).collect();
        let ad: Vec<f64> = delta.iter().map(|v| v.abs()).collect();
        let rpp = block(r, k, prev, prev);
        let ed = abs_times(&oracle::cholesky_backward(&rpp), &ad);
        // The update's rounding as it reaches column `j`.
        let moved = |j: usize| -> f64 {
            prev.iter()
                .zip(&delta)
                .map(|(&i, di)| r[j * k + i].abs() * (bc[i].abs() + (step * di).abs()))
                .sum::<f64>()
                * U
        };
        let mut next = vec![0.0f64; k];
        for (p, &i) in prev.iter().enumerate() {
            next[i] = res[i] + step * ed[p] + moved(i) + U * lam;
        }
        let entering: Vec<usize> = cur.iter().copied().filter(|j| !prev.contains(j)).collect();
        let gone = prev.iter().filter(|i| !cur.contains(i)).count();
        // One column changing at the knot is the one whose step `γ` is: its
        // distance from `γ` is the step's own rounding, not the tie.
        let alone = entering.len() + gone == 1;
        let leaving: Vec<(usize, f64)> = prev
            .iter()
            .zip(&delta)
            .filter(|(i, _)| !cur.contains(i))
            .map(|(&i, &di)| {
                let short = if alone {
                    4.0 * U * bp[i].abs()
                } else {
                    tie * di.abs()
                };
                (i, short)
            })
            .collect();
        for &i in cur {
            next[i] += leaving
                .iter()
                .map(|&(l, bl)| r[i * k + l].abs() * bl)
                .sum::<f64>();
        }
        for &e in &entering {
            let c: f64 = d[e] - prev.iter().map(|&i| r[e * k + i] * bp[i]).sum::<f64>();
            let a: f64 = prev
                .iter()
                .zip(&delta)
                .map(|(&i, di)| r[e * k + i] * di)
                .sum();
            let s = c.signum();
            let ec = gamma(prev.len() + 1)
                * prev
                    .iter()
                    .map(|&i| (r[e * k + i] * bp[i]).abs())
                    .sum::<f64>()
                + U * d[e].abs();
            let ea = gamma(prev.len())
                * prev
                    .iter()
                    .zip(&ad)
                    .map(|(&i, di)| r[e * k + i].abs() * di)
                    .sum::<f64>();
            let off = if alone { 0.0 } else { tie };
            next[e] =
                (1.0 - s * a).abs() * (off + 4.0 * U * step) + ec + step * ea + moved(e) + U * lam;
        }
        if entering.len() + leaving.len() > 1 {
            // The columns staying, solved afresh at this penalty.
            let staying: Vec<usize> = cur
                .iter()
                .copied()
                .filter(|j| !entering.contains(j))
                .collect();
            if !staying.is_empty() {
                let rss = block(r, k, &staying, &staying);
                let bs: Vec<f64> = staying.iter().map(|&i| bc[i].abs()).collect();
                let fresh = abs_times(&oracle::cholesky_backward(&rss), &bs);
                let old: Vec<f64> = staying.iter().map(|&i| next[i]).collect();
                let change = abs_times(&oracle::inverse(&rss), &old);
                for (p, &i) in staying.iter().enumerate() {
                    next[i] = fresh[p] + U * (d[i].abs() + lam);
                }
                for &e in &entering {
                    next[e] += staying
                        .iter()
                        .zip(&change)
                        .map(|(&i, ch)| r[e * k + i].abs() * ch)
                        .sum::<f64>();
                }
            }
        }
        res = next;
        out.push(res.clone());
    }
    out
}

/// The most knot `n`'s active coefficients may be off the oracle's solve of
/// `R_AA b_A = rhs`: `b − b* = −R_AA⁻¹ r_A` for the residual `r` that
/// [`residual_bounds`] bounds, so `|R_AA⁻¹|·r_A`, beside the oracle's own
/// error ([`oracle::solve_error`]).
pub(super) fn knot_error(r: &[f64], act: &[usize], res: &[f64], rhs: &[f64]) -> Vec<f64> {
    let k = res.len();
    let raa = block(r, k, act, act);
    let want = oracle::solve(&raa, rhs);
    let ra: Vec<f64> = act.iter().map(|&i| res[i]).collect();
    let lib = oracle::abs_times(&oracle::inverse(&raa), &ra);
    let own = oracle::solve_error(&raa, rhs, &want);
    lib.iter().zip(&own).map(|(a, b)| a + b).collect()
}

#[test]
fn every_knot_meets_the_lasso_conditions_and_the_active_set_solve() {
    let beta = [1.0, -0.8, 0.6, 0.0, 0.3, 0.0, -0.2, 0.0];
    let (r, d) = design(7, 400, 8, 0.6, &beta);
    let live = vec![true; 8];
    let path = lars_lasso(&r, &d, &live, LarsLimits::default());
    assert_eq!(path.stop, LarsStop::End);
    assert_eq!(*path.penalties.last().unwrap(), 0.0);
    assert!(path.len() >= 9, "eight columns enter: {}", path.len());
    assert!(!path.is_empty());
    let none = LarsPath {
        penalties: Vec::new(),
        coefs: Vec::new(),
        active: Vec::new(),
        stop: LarsStop::End,
    };
    assert!(none.is_empty(), "a path built with no knot");
    assert_eq!(none.len(), 0);
    for w in path.penalties.windows(2) {
        assert!(w[1] < w[0], "λ falls along the path: {w:?}");
    }
    let res = residual_bounds(&r, &d, &path);
    for (n, &lambda) in path.penalties.iter().enumerate() {
        let b = &path.coefs[n * 8..(n + 1) * 8];
        assert_kkt(&r, &d, b, lambda, 1e-12);
        // The active coefficients solve R_AA b_A = d_A − λ s_A (faer's LU).
        let act = &path.active[n];
        if act.is_empty() {
            continue;
        }
        let m = act.len();
        let mut raa = Vec::with_capacity(m * m);
        for &i in act {
            for &j in act {
                raa.push(r[i * 8 + j]);
            }
        }
        let rhs: Vec<f64> = act
            .iter()
            .map(|&i| {
                let c = d[i] - (0..8).map(|j| r[i * 8 + j] * b[j]).sum::<f64>();
                d[i] - lambda * c.signum()
            })
            .collect();
        let want = oracle::solve(&raa, &rhs);
        // Task 274: the residual the path's steps leave, through `R_AA⁻¹`
        // ([`knot_error`]); it was `1e-12` relative to `1 + |b|`.
        // Measured on aarch64 (macOS): at most 0.24 of the bound, at the first step from the
        // first knot (a few roundings, nothing accumulated yet); 0.010 at the path's end.
        let bound = knot_error(&r, act, &res[n], &rhs);
        for p in 0..m {
            oracle::assert_within(
                b[act[p]],
                want[p],
                bound[p],
                format_args!("knot {n}, column {}", act[p]),
            );
        }
    }
    // The end of the path is the least-squares fit (faer's LU), held as
    // the last knot is. It was `1e-12`.
    let n = path.len() - 1;
    let last = &path.coefs[n * 8..];
    let ls = oracle::solve(&r, &d);
    let all: Vec<usize> = (0..8).collect();
    let mut every = path.active[n].clone();
    every.sort_unstable();
    assert_eq!(every, all);
    let bound = knot_error(&r, &all, &res[n], &d);
    for j in 0..8 {
        oracle::assert_within(
            last[j],
            ls[j],
            bound[j],
            format_args!("the end, column {j}"),
        );
    }
}

/// A path with a column leaving: correlated columns with signs that
/// disagree, the first of seeds 0..200 whose path has one.
pub(super) fn path_with_a_drop() -> (Vec<f64>, Vec<f64>, LarsPath) {
    let beta = [1.0, 1.0, -0.9, 0.0, 0.5, -0.4];
    for seed in 0..200 {
        let (r, d) = design(seed, 60, 6, 1.5, &beta);
        let path = lars_lasso(&r, &d, &[true; 6], LarsLimits::default());
        if path.active.windows(2).any(|w| w[1].len() < w[0].len()) {
            return (r, d, path);
        }
    }
    panic!("no seed in 0..200 makes a column leave");
}

#[test]
fn a_column_leaves_where_its_coefficient_crosses_zero() {
    let (r, d, path) = path_with_a_drop();
    let k = 6;
    let n = path
        .active
        .windows(2)
        .position(|w| w[1].len() < w[0].len())
        .unwrap()
        + 1;
    let gone: Vec<usize> = path.active[n - 1]
        .iter()
        .copied()
        .filter(|j| !path.active[n].contains(j))
        .collect();
    assert_eq!(gone.len(), 1);
    assert_eq!(path.coefs[n * k + gone[0]], 0.0, "it leaves at exactly 0");
    for (m, support) in supports(&path).into_iter().enumerate() {
        assert_eq!(support, knot_support(&path, k, m), "knot {m}");
    }
    // It comes back: the path runs on to the least-squares fit on every
    // column (faer's LU), where a path that barred it for good would end
    // on the others alone.
    assert_eq!(path.stop, LarsStop::End);
    assert_eq!(path.active.last().unwrap().len(), k);
    let ls = oracle::solve(&r, &d);
    let last = &path.coefs[(path.len() - 1) * k..];
    // Task 274: the residual the path's steps leave, through `R⁻¹`
    // ([`knot_error`]); it was `1e-10`.
    // Measured on aarch64 (macOS): at most 0.024 of the bound.
    let all: Vec<usize> = (0..k).collect();
    let res = residual_bounds(&r, &d, &path);
    let bound = knot_error(&r, &all, res.last().unwrap(), &d);
    for j in 0..k {
        oracle::assert_within(
            last[j],
            ls[j],
            bound[j],
            format_args!("the end, column {j}"),
        );
    }
    // Without the drop the coefficient would change sign: the conditions
    // hold at every knot, the one after the drop included.
    for (m, &lambda) in path.penalties.iter().enumerate() {
        assert_kkt(&r, &d, &path.coefs[m * k..(m + 1) * k], lambda, 1e-12);
    }
}

/// Each knot's support as the active sets say it: the set before the knot,
/// less the column leaving at it, and the last knot's own set.
pub(super) fn supports(path: &LarsPath) -> Vec<Vec<usize>> {
    (0..path.len())
        .map(|n| {
            let mut s: Vec<usize> = if n == 0 {
                Vec::new()
            } else if n + 1 == path.len() && *path.penalties.last().unwrap() == 0.0 {
                path.active[n].clone()
            } else {
                path.active[n - 1]
                    .iter()
                    .copied()
                    .filter(|j| path.active[n].contains(j))
                    .collect()
            };
            s.sort_unstable();
            s
        })
        .collect()
}

/// Knot `n`'s coefficients' support.
pub(super) fn knot_support(path: &LarsPath, k: usize, n: usize) -> Vec<usize> {
    (0..k).filter(|&j| path.coefs[n * k + j] != 0.0).collect()
}

/// [`design`] with every column a mix of all of them, `x = z (I + M)` for
/// a standard normal `M`: correlations of both signs, which is what makes
/// columns leave a lasso path and come back.
pub(super) fn mixed_design(seed: u64, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed ^ 0x9e37_79b9);
    let mut normal = || {
        let u = 1.0 - rng.uniform();
        let v = rng.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let m: Vec<f64> = (0..k * k)
        .map(|ij| normal() + if ij / k == ij % k { 1.0 } else { 0.0 })
        .collect();
    let beta: Vec<f64> = (0..k).map(|_| normal()).collect();
    let mut x = vec![0.0; n * k];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let z: Vec<f64> = (0..k).map(|_| normal()).collect();
        for j in 0..k {
            x[i * k + j] = (0..k).map(|q| z[q] * m[q * k + j]).sum();
        }
        y[i] = (0..k).map(|j| beta[j] * x[i * k + j]).sum::<f64>() + 2.0 * normal();
    }
    standardized(&x, &y, n, k)
}

#[test]
fn columns_leave_and_come_back_and_every_knot_is_the_lasso() {
    let (mut leaves, mut returns) = (0, 0);
    for seed in 0..30 {
        let (r, d) = mixed_design(seed, 200, 10);
        let path = lars_lasso(&r, &d, &[true; 10], LarsLimits::default());
        for (n, &lambda) in path.penalties.iter().enumerate() {
            assert_kkt(&r, &d, &path.coefs[n * 10..(n + 1) * 10], lambda, 1e-11);
        }
        for w in path.active.windows(2) {
            if w[1].len() < w[0].len() {
                leaves += 1;
            }
        }
        // A column back the step after it left.
        for w in path.active.windows(3) {
            let gone: Vec<&usize> = w[0].iter().filter(|j| !w[1].contains(j)).collect();
            if gone.len() == 1 && w[2].contains(gone[0]) {
                returns += 1;
            }
        }
        let ls = oracle::solve(&r, &d);
        let last = &path.coefs[(path.len() - 1) * 10..];
        // Task 274: the residual the path's steps leave, through `R⁻¹`
        // ([`knot_error`]); it was `1e-9`.
        // Measured on aarch64 (macOS): at most 0.013 of the bound.
        let all: Vec<usize> = (0..10).collect();
        assert_eq!(path.active.last().unwrap().len(), 10);
        let res = residual_bounds(&r, &d, &path);
        let bound = knot_error(&r, &all, res.last().unwrap(), &d);
        for j in 0..10 {
            oracle::assert_within(
                last[j],
                ls[j],
                bound[j],
                format_args!("seed {seed}, the end, column {j}"),
            );
        }
    }
    assert!(
        leaves > 0 && returns > 0,
        "{leaves} leaves, {returns} returns"
    );
}

#[test]
fn the_path_stops_after_max_steps_or_at_max_active() {
    let (r, d) = design(11, 300, 10, 0.4, &[1.0, -1.0, 0.5, 0.5, 0.2]);
    let live = vec![true; 10];
    let full = lars_lasso(&r, &d, &live, LarsLimits::default());
    let steps = lars_lasso(
        &r,
        &d,
        &live,
        LarsLimits {
            max_steps: Some(3),
            max_active: None,
        },
    );
    assert_eq!(steps.stop, LarsStop::MaxSteps);
    assert_eq!(steps.len(), 4, "the first knot and three steps");
    assert_eq!(steps.penalties[..], full.penalties[..4]);
    assert_eq!(steps.coefs[..], full.coefs[..40]);
    let act = lars_lasso(
        &r,
        &d,
        &live,
        LarsLimits {
            max_steps: None,
            max_active: Some(4),
        },
    );
    assert_eq!(act.stop, LarsStop::MaxActive);
    assert_eq!(act.active.last().unwrap().len(), 4);
    assert!(act.active[..act.len() - 1].iter().all(|a| a.len() < 4));
    assert_eq!(act.penalties[..], full.penalties[..act.len()]);
    // One active from the start.
    let one = lars_lasso(
        &r,
        &d,
        &live,
        LarsLimits {
            max_steps: None,
            max_active: Some(1),
        },
    );
    assert_eq!(one.len(), 1, "the first column is active from λ_max down");
    assert_eq!(one.stop, LarsStop::MaxActive);
    assert_eq!(one.active[0].len(), 1);
    assert!(full.coefs[..10].iter().all(|&v| v == 0.0));
    for (n, support) in supports(&full).into_iter().enumerate() {
        assert_eq!(support, knot_support(&full, 10, n), "knot {n}");
    }
}

#[test]
fn a_dead_column_never_enters_and_a_collinear_one_is_set_aside() {
    let (mut r, mut d) = design(5, 200, 5, 0.3, &[1.0, 0.8, 0.0, 0.0, 0.5]);
    // Column 1 dead: excluded whatever its correlation.
    let mut live = vec![true; 5];
    live[1] = false;
    let path = lars_lasso(&r, &d, &live, LarsLimits::default());
    assert!(path.active.iter().all(|a| !a.contains(&1)));
    assert!((0..path.len()).all(|n| path.coefs[n * 5 + 1] == 0.0));
    // Column 4 a copy of column 0: one of the two is set aside.
    for j in 0..5 {
        r[4 * 5 + j] = r[j];
        r[j * 5 + 4] = r[j * 5];
    }
    r[4 * 5 + 4] = 1.0;
    d[4] = d[0];
    let path = lars_lasso(&r, &d, &[true; 5], LarsLimits::default());
    let last = path.active.last().unwrap();
    assert!(
        !(last.contains(&0) && last.contains(&4)),
        "both copies active: {last:?}"
    );
    assert_eq!(
        last.len(),
        4,
        "the other four columns active at the end: {last:?}"
    );
    assert_eq!(path.stop, LarsStop::End);
    assert!(path.coefs.iter().all(|v| v.is_finite()));
}

#[test]
fn a_target_uncorrelated_with_every_column_is_one_knot_at_zero() {
    let (r, _) = design(2, 50, 3, 0.0, &[]);
    let path = lars_lasso(&r, &[0.0; 3], &[true; 3], LarsLimits::default());
    assert_eq!(path.penalties, vec![0.0]);
    assert_eq!(path.coefs, vec![0.0; 3]);
    let none = lars_lasso(&r, &[1.0, 2.0, 3.0], &[false; 3], LarsLimits::default());
    assert_eq!(none.penalties, vec![0.0]);
}

#[test]
fn coordinate_descent_reaches_the_lars_knots() {
    let beta = [1.0, -0.8, 0.6, 0.0, 0.3, 0.0, -0.2, 0.0];
    let (r, d) = design(7, 400, 8, 0.6, &beta);
    let path = lars_lasso(&r, &d, &[true; 8], LarsLimits::default());
    let grid: Vec<f64> = path.penalties.clone();
    let cd = cd_path(
        &r,
        &d,
        &[true; 8],
        &grid,
        1.0,
        &[1.0; 8],
        10_000,
        1e-13,
        f64::NAN,
    );
    assert!(cd.unconverged.iter().all(|&u| !u));
    let cd = cd.coefs;
    for (a, b) in cd.iter().zip(&path.coefs) {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
}

#[test]
fn coordinate_descent_meets_the_elastic_net_conditions_with_weights() {
    // Definition: c_j = d_j − (R b)_j − l₂_j b_j equals l₁_j sign(b_j) where
    // b_j ≠ 0 and is at most l₁_j in size elsewhere.
    let (r, d) = design(13, 300, 6, 0.5, &[1.0, 0.5, -0.5, 0.0, 0.2, 0.0]);
    let w = [1.0, 0.0, 2.0, 1.0, 0.5, 1.0];
    let (lam, ratio) = (0.05, 0.7);
    let mut live = vec![true; 6];
    live[5] = false;
    let b = cd_path(
        &r,
        &d,
        &live,
        &[0.2, lam],
        ratio,
        &w,
        10_000,
        1e-14,
        f64::NAN,
    )
    .coefs;
    let b = &b[6..];
    assert_eq!(b[5], 0.0, "a dead column stays at 0");
    for j in 0..5 {
        let (l1, l2) = (lam * ratio * w[j], lam * (1.0 - ratio) * w[j]);
        let c = d[j] - (0..6).map(|i| r[j * 6 + i] * b[i]).sum::<f64>() - l2 * b[j];
        if b[j] != 0.0 {
            assert!((c - l1 * b[j].signum()).abs() < 1e-10, "{j}: {c}");
        } else {
            assert!(c.abs() <= l1 + 1e-10, "{j}: {c}");
        }
    }
    assert_ne!(b[1], 0.0, "an unpenalized column is never shrunk to 0");
}

#[test]
fn coordinate_descent_stops_at_max_iter() {
    let (r, d) = design(3, 100, 4, 0.9, &[1.0, 1.0, 1.0, 1.0]);
    let one = cd_path(&r, &d, &[true; 4], &[0.0], 1.0, &[1.0; 4], 1, 1e-7, 1.0);
    let many = cd_path(
        &r,
        &d,
        &[true; 4],
        &[0.0],
        1.0,
        &[1.0; 4],
        10_000,
        1e-7,
        1.0,
    );
    assert_ne!(one.coefs, many.coefs, "one sweep is not the converged fit");
    // The cap is said per penalty (task 278, D6: it was silent).
    assert_eq!(one.unconverged, vec![true]);
    assert_eq!(many.unconverged, vec![false]);
}

#[test]
fn weighted_knots_meet_the_weighted_conditions() {
    let (r, d) = design(17, 300, 6, 0.5, &[1.0, -0.6, 0.4, 0.0, 0.3, 0.0]);
    let live = [true, true, true, true, false, true];
    let w = [1.0, 3.0, 0.5, 1.0, 0.0, 2.0];
    let path = lars_lasso_weighted(
        &DenseRows { r: &r, k: 6 },
        &d,
        &live,
        &w,
        LarsLimits::default(),
    )
    .unwrap();
    for (n, &lambda) in path.penalties.iter().enumerate() {
        let b = &path.coefs[n * 6..(n + 1) * 6];
        assert_eq!(b[4], 0.0, "a dead column stays at 0 whatever its weight");
        for j in (0..6).filter(|&j| live[j]) {
            let c = d[j] - (0..6).map(|i| r[j * 6 + i] * b[i]).sum::<f64>();
            if b[j] != 0.0 {
                assert!((c - lambda * w[j] * b[j].signum()).abs() < 1e-12, "{n} {j}");
            } else {
                assert!(c.abs() <= lambda * w[j] + 1e-12, "{n} {j}");
            }
        }
    }
    let ones = lars_lasso_weighted(
        &DenseRows { r: &r, k: 6 },
        &d,
        &live,
        &[1.0; 6],
        LarsLimits::default(),
    )
    .unwrap();
    assert_eq!(ones, lars_lasso(&r, &d, &live, LarsLimits::default()));
    let err = lars_lasso_weighted(
        &DenseRows { r: &r, k: 6 },
        &d,
        &[true; 6],
        &[1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        LarsLimits::default(),
    )
    .unwrap_err();
    assert!(err.contains("column 1"), "{err}");
}

/// With no weight at 1 the weights still apply: the knots meet the
/// weighted conditions, and the path is not the plain one.
#[test]
fn weights_none_of_which_is_one_meet_the_weighted_conditions() {
    let (r, d) = design(19, 300, 5, 0.5, &[1.0, -0.6, 0.4, 0.0, 0.3]);
    let live = [true; 5];
    let w = [2.0, 0.5, 3.0, 1.5, 0.25];
    let path = lars_lasso_weighted(
        &DenseRows { r: &r, k: 5 },
        &d,
        &live,
        &w,
        LarsLimits::default(),
    )
    .unwrap();
    assert!(path.len() > 2);
    for (n, &lambda) in path.penalties.iter().enumerate() {
        let b = &path.coefs[n * 5..(n + 1) * 5];
        for j in 0..5 {
            let c = d[j] - (0..5).map(|i| r[j * 5 + i] * b[i]).sum::<f64>();
            if b[j] != 0.0 {
                assert!((c - lambda * w[j] * b[j].signum()).abs() < 1e-12, "{n} {j}");
            } else {
                assert!(c.abs() <= lambda * w[j] + 1e-12, "{n} {j}");
            }
        }
    }
    assert_ne!(path, lars_lasso(&r, &d, &live, LarsLimits::default()));
}

/// The lasso is the same problem on `(c R, c d)` at `c λ`: `f(b)` is `c`
/// times the original, so the path's knots fall at `c λ` with the same
/// coefficients and active sets. `c` a power of 4 scales every operation
/// exactly, the factor's square roots too, so the two agree to the bit. A
/// diagonal of `c` in place of 1 is where a collinearity floor not taken as
/// a share of the diagonal would set every column aside, or none.
#[test]
fn the_path_scales_with_the_gram() {
    let (r, d) = design(23, 200, 6, 0.7, &[1.0, -0.5, 0.0, 0.4, 0.0, 0.2]);
    let live = [true; 6];
    let base = lars_lasso(&r, &d, &live, LarsLimits::default());
    assert_eq!(base.active.last().unwrap().len(), 6);
    for c in [2f64.powi(-40), 2f64.powi(40)] {
        let rc: Vec<f64> = r.iter().map(|v| v * c).collect();
        let dc: Vec<f64> = d.iter().map(|v| v * c).collect();
        let got = lars_lasso(&rc, &dc, &live, LarsLimits::default());
        let want: Vec<u64> = base.penalties.iter().map(|v| (v * c).to_bits()).collect();
        let pen: Vec<u64> = got.penalties.iter().map(|v| v.to_bits()).collect();
        assert_eq!(pen, want, "c = {c:e}");
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got.coefs), bits(&base.coefs), "c = {c:e}");
        assert_eq!(got.active, base.active, "c = {c:e}");
    }
}

/// A path read a row at a time from a Gram ([`GramRows`]) is the path on
/// the correlation matrix formed whole, to the bit, with the intercept and
/// through the origin, on columns of unequal spread.
#[test]
fn a_path_read_from_a_gram_is_the_path_on_its_correlation_matrix() {
    let (r, d) = mixed_design(4, 120, 5);
    // Column 0 the intercept; columns 1..=5 scaled by `s`, means off 0.
    let k = 6;
    let s = [0.5, 3.0, 1.0, 7.0, 0.2];
    let mut como = vec![0.0; k * k];
    let mut cc = vec![0.0; k];
    for i in 0..5 {
        for j in 0..5 {
            como[(i + 1) * k + j + 1] = r[i * 5 + j] * s[i] * s[j];
        }
        cc[i + 1] = d[i] * s[i];
    }
    let means = [1.0, 0.3, -2.0, 1.5, 0.0, 4.0];
    let cross: Vec<f64> = (0..k).map(|i| 0.7 * means[i] + cc[i]).collect();
    let g = GramArrays {
        k,
        means: &means,
        comoments: Comoments::Full(&como),
        cross_moments: &cross,
        means_by_target: &means,
        cross_centred: &cc,
        weight_sum: 120.0,
        target_weights: &[120.0],
        target_means: &[0.7],
        target_vars: &[2.0],
        target_n_kish: &[120.0],
        within: None,
    };
    let slots = [1, 2, 3, 4, 5];
    for icept in [Some(0), None] {
        let corr = Correlation::of(&Design::of(&g, &slots, icept));
        let resp = Response::of(&g, 0, &slots, icept);
        let dd = corr.scaling.d(&resp);
        let rows = GramRows::new(g, &slots, icept);
        let path = lars_lasso_rows(&rows, &dd, &corr.scaling.live, LarsLimits::default());
        assert_eq!(
            path,
            lars_lasso(&corr.r, &dd, &corr.scaling.live, LarsLimits::default())
        );
        assert!(path.len() > 5, "{icept:?}: {} knots", path.len());
    }
}

#[test]
fn a_system_centres_at_the_targets_means_and_recovers_the_intercept() {
    // Two targets over three columns, the intercept at 0.
    let k = 3;
    let means = [1.0, 2.0, -1.0];
    let como = [0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 1.0, 9.0];
    let cross = [5.0, 11.0, -4.0, 2.0, 4.5, -2.5];
    let mbt = [1.0, 2.0, -1.0, 1.0, 2.5, -1.5];
    let cc = [0.0, 1.0, 2.0, 0.0, -0.5, 0.25];
    let g = GramArrays {
        k,
        means: &means,
        comoments: Comoments::Full(&como),
        cross_moments: &cross,
        means_by_target: &mbt,
        cross_centred: &cc,
        weight_sum: 10.0,
        target_weights: &[10.0, 10.0],
        target_means: &[5.0, 2.0],
        target_vars: &[1.0, 1.0],
        target_n_kish: &[10.0, 10.0],
        within: None,
    };
    assert_eq!(g.targets(), 2);
    let design = Design::of(&g, &[1, 2], Some(0));
    assert_eq!(design.a, vec![4.0, 1.0, 1.0, 9.0]);
    let resp = Response::of(&g, 1, &[1, 2], Some(0));
    assert_eq!(resp.rhs, vec![-0.5, 0.25]);
    assert_eq!(resp.means, vec![2.5, -1.5]);
    assert_eq!(resp.ybar, 2.0);
    let corr = Correlation::of(&design);
    assert_eq!(corr.scaling.scale, vec![2.0, 3.0]);
    assert_eq!(corr.r, vec![1.0, 1.0 / 6.0, 1.0 / 6.0, 1.0]);
    assert_eq!(corr.scaling.d(&resp), vec![-0.25, 0.25 / 3.0]);
    let coef = resp.coef(&corr.scaling, &[2.0, 3.0], k, &[1, 2], Some(0));
    // Row by row from the Gram, the same numbers.
    for icept in [Some(0), None] {
        let whole = Correlation::of(&Design::of(&g, &[1, 2], icept));
        let rows = GramRows::new(g, &[1, 2], icept);
        assert_eq!(rows.scaling, whole.scaling);
        let mut row = [0.0; 2];
        for p in 0..2 {
            rows.row(p, &mut row);
            assert_eq!(row[..], whole.r[p * 2..(p + 1) * 2]);
        }
    }
    assert_eq!(coef, vec![2.0 - (1.0 * 2.5 + 1.0 * -1.5), 1.0, 1.0]);
    // Through the origin: the raw moments.
    assert_eq!(
        Design::of(&g, &[1, 2], None).a,
        vec![4.0 + 4.0, 1.0 - 2.0, 1.0 - 2.0, 9.0 + 1.0]
    );
    let resp = Response::of(&g, 0, &[1, 2], None);
    assert_eq!(resp.rhs, vec![11.0, -4.0]);
    assert_eq!((resp.means.clone(), resp.ybar), (vec![2.0, -1.0], 0.0));
    // A constant column is dead, its coefficient 0.
    let como0 = [0.0; 9];
    let g0 = GramArrays {
        comoments: Comoments::Full(&como0),
        ..g
    };
    let corr = Correlation::of(&Design::of(&g0, &[1, 2], Some(0)));
    assert_eq!(corr.scaling.live, vec![false, false]);
    assert_eq!(
        corr.scaling.scale,
        vec![1.0, 1.0],
        "a dead column's scale is 1"
    );
    assert_eq!(corr.r, vec![1.0, 0.0, 0.0, 1.0]);
}

/// A dead column's row and column of `R` are 0 but for its 1 on the
/// diagonal whatever its co-moments with the others: a variance rounded to
/// just below 0 beside a covariance that is not 0 (as a Gram streamed in
/// float32, or a column nearly constant, can hold) reads as no correlation.
#[test]
fn a_dead_columns_correlations_are_zero_whatever_its_comoments() {
    let design = Design {
        a: vec![4.0, 1e-9, 0.5, 1e-9, -1e-18, 2e-9, 0.5, 2e-9, 9.0],
        n: 3,
    };
    let corr = Correlation::of(&design);
    assert_eq!(corr.scaling.live, vec![true, false, true]);
    assert_eq!(corr.scaling.scale, vec![2.0, 1.0, 3.0]);
    let third = 0.5 / 6.0;
    assert_eq!(
        corr.r,
        vec![1.0, 0.0, third, 0.0, 1.0, 0.0, third, 0.0, 1.0]
    );
}

/// The co-moments' size and form, in each of the four forms
/// `ModelBank.gram()` hands them over in.
#[test]
fn the_comoments_say_their_size_and_form() {
    let (f64s, f32s) = ([1.0, 2.0, 2.0, 5.0], [1.0f32, 2.0, 5.0]);
    let forms = [
        (Comoments::Full(&f64s), 4, false),
        (Comoments::Full32(&[1.0f32, 2.0, 2.0, 5.0]), 4, false),
        (Comoments::Packed(&[1.0, 2.0, 5.0]), 3, true),
        (Comoments::Packed32(&f32s), 3, true),
    ];
    for (c, len, packed) in forms {
        assert_eq!(
            (c.len(), c.is_empty(), c.is_packed()),
            (len, false, packed),
            "{c:?}"
        );
        assert_eq!(c.get(2, 1, 0), 2.0, "{c:?}");
    }
    for c in [
        Comoments::Full(&[]),
        Comoments::Full32(&[]),
        Comoments::Packed(&[]),
        Comoments::Packed32(&[]),
    ] {
        assert!(c.is_empty(), "{c:?}");
        assert_eq!(c.len(), 0, "{c:?}");
    }
}

/// A ridge makes a system with a constant column solvable, and the
/// standard errors read the centred co-moments of the columns with
/// variance: the constant's own is NaN, the other's least squares' on it
/// alone (review round 8, A-5; docs/PLAN.md task 269), where a factor of
/// the whole, which is not positive definite, made every one NaN. And
/// where the columns with variance are not positive definite themselves --
/// two copies -- the errors are NaN, as numpy's clip keeps a NaN, never
/// the 0 `f64::max(NaN, 0)` gives, which reads as an exact coefficient
/// (review 6, E-1).
#[test]
fn a_standard_error_of_a_system_that_is_not_positive_definite_is_nan() {
    let k = 3;
    let como = [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    let g = GramArrays {
        k,
        means: &[1.0, 2.0, 7.0],
        comoments: Comoments::Full(&como),
        cross_moments: &[5.0, 11.0, 35.0],
        means_by_target: &[1.0, 2.0, 7.0],
        cross_centred: &[0.0, 1.0, 0.0],
        weight_sum: 50.0,
        target_weights: &[50.0],
        target_means: &[5.0],
        target_vars: &[1.0],
        target_n_kish: &[50.0],
        within: None,
    };
    for standardize in [false, true] {
        let fit = &ridge_fits(&g, &[0], &[1, 2], Some(0), 0.1, standardize)[0];
        assert!(fit.coef.iter().all(|c| c.is_finite()), "{:?}", fit.coef);
        assert!(fit.sigma2.is_finite(), "{}", fit.sigma2);
        // Slot 2 is the constant: no variance, no error.
        assert!(
            fit.se[2].is_nan() && fit.t[2].is_nan(),
            "{standardize}: {:?}",
            fit.se
        );
        // Slot 1 on its own: `C⁻¹ = 1/4`, and the intercept's `1 + m²/4`.
        let se1 = (0.25 * fit.sigma2 / 50.0).sqrt();
        assert!(
            (fit.se[1] - se1).abs() <= 1e-15 * se1,
            "{standardize}: {:?}",
            fit.se
        );
        assert_eq!(fit.t[1], fit.coef[1] / fit.se[1]);
        let se0 = ((1.0 + 2.0 * 2.0 / 4.0) * fit.sigma2 / 50.0).sqrt();
        assert!(
            (fit.se[0] - se0).abs() <= 1e-15 * se0,
            "{standardize}: {:?}",
            fit.se
        );
    }
    // Two copies of a column with variance: their co-moments are singular.
    let copies = [0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 0.0, 4.0, 4.0];
    let g = GramArrays {
        comoments: Comoments::Full(&copies),
        cross_centred: &[0.0, 1.0, 1.0],
        means: &[1.0, 2.0, 2.0],
        means_by_target: &[1.0, 2.0, 2.0],
        ..g
    };
    let fit = &ridge_fits(&g, &[0], &[1, 2], Some(0), 0.1, false)[0];
    assert!(fit.coef.iter().all(|c| c.is_finite()), "{:?}", fit.coef);
    assert!(fit.sigma2.is_finite(), "{}", fit.sigma2);
    for p in [0, 1, 2] {
        assert!(fit.se[p].is_nan() && fit.t[p].is_nan(), "{p}: {:?}", fit.se);
    }
}

/// A column set aside as collinear takes no step and makes no knot: each
/// knot but the last is a column entering or leaving (review 6, E-5: the
/// path moved to where the column would have entered first, and recorded a
/// knot there with the active set unchanged; past `n` rows every column
/// left did, 987 of 1,126 knots at 100 rows and 2,000 columns). A column
/// exactly the sum of two others is collinear within rounding, and more
/// columns than rows leaves every column past the rank collinear.
#[test]
fn a_column_set_aside_makes_no_knot() {
    let mut rng = SplitMix64::new(77);
    let mut normal = || {
        let u = 1.0 - rng.uniform();
        let v = rng.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let mut cases = 0;
    let mut set_aside = 0;
    for (n, k, sum) in [(30, 60, false), (200, 9, true), (20, 40, true)] {
        let width = k + usize::from(sum);
        let mut x = vec![0.0; n * width];
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..k {
                x[i * width + j] = normal();
            }
            if sum {
                x[i * width + k] = x[i * width] + x[i * width + 1];
            }
            y[i] = x[i * width] - 0.5 * x[i * width + 2] + normal();
        }
        let (r, d) = standardized(&x, &y, n, width);
        let path = lars_lasso(&r, &d, &vec![true; width], LarsLimits::default());
        let ends = path.len() - 1;
        for knot in 1..path.len() {
            if knot == ends && path.penalties[knot] == 0.0 {
                continue;
            }
            assert_ne!(
                path.active[knot],
                path.active[knot - 1],
                "{n} rows, {width} columns: knot {knot} of {} at {} moves nothing in or out",
                path.len(),
                path.penalties[knot]
            );
        }
        // Columns were set aside: more than the path's last active set
        // were live, and no more than the rank entered.
        let last = path.active.last().unwrap().len();
        assert!(last <= n.min(width), "{last}");
        set_aside += width - last;
        cases += 1;
        for knot in 0..path.len() {
            assert_kkt(
                &r,
                &d,
                &path.coefs[knot * width..(knot + 1) * width],
                path.penalties[knot],
                1e-8,
            );
        }
    }
    assert_eq!(cases, 3);
    assert!(set_aside > 0);
}

/// A sweep that moves no coefficient by `tol · s` or more ends the descent
/// (the module doc), `s` the target's spread: the path at `tol` is the
/// sweeps run one at a time -- at `tol = 0`, where none ends early but on a
/// sweep that moves nothing -- up to the first whose largest move is under
/// `tol · s`. A sweep that moves one by exactly `tol · s` does not end it.
/// `s` is a power of two above every `|d_i|`, so `tol · s` is exact.
#[test]
fn coordinate_descent_stops_at_the_first_sweep_that_moves_less_than_tol() {
    let (r, d) = design(3, 100, 4, 0.9, &[1.0, -1.0, 0.5, 1.0]);
    let live = [true; 4];
    let spread = 8.0;
    assert!(d.iter().all(|v| v.abs() < spread));
    let run = |max_iter: usize, tol: f64| {
        let path = cd_path(
            &r,
            &d,
            &live,
            &[0.01],
            1.0,
            &[1.0; 4],
            max_iter,
            tol / spread,
            spread,
        );
        path.coefs
    };
    // Sweep m's coefficients and its largest move, by definition.
    let mut after = vec![vec![0.0; 4]];
    let mut moves = vec![f64::INFINITY];
    for m in 1..=60 {
        let b = run(m, 0.0);
        let moved = b
            .iter()
            .zip(&after[m - 1])
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        after.push(b);
        moves.push(moved);
    }
    // A sweep whose move is below every earlier one, and one later that
    // moves less still.
    let m0 = (5..40)
        .find(|&m| moves[..m].iter().all(|&v| v > moves[m]))
        .expect("the moves fall");
    let tol = moves[m0];
    let stop = (m0 + 1..=60)
        .find(|&m| moves[m] < tol)
        .expect("a sweep moves less");
    assert_ne!(after[stop], after[60], "the stop is not the converged fit");
    assert_eq!(run(10_000, tol), after[stop], "tol = sweep {m0}'s move");
    // Just above that move, the sweep that made it is the last.
    assert_eq!(run(10_000, tol * (1.0 + 1e-12)), after[m0]);
}

/// The rule is dimensionless (task 278, D3): the target scaled by `2^p`,
/// its spread and the penalties with it, is the same descent scaled, to
/// the bit, stopping on the same sweep -- with the spread given, and with
/// it unknown (NaN), where the floor `max_i |d_i|` is the scale. Before,
/// `tol` was compared with the move itself: at `2^-30` the first sweeps'
/// moves were already under it.
#[test]
fn coordinate_descent_is_the_same_at_any_scale_of_the_target() {
    let (r, d) = design(5, 200, 5, 0.95, &[1.0, -0.7, 0.5, 0.0, 0.3]);
    let live = [true; 5];
    let grid = [0.1, 0.01, 0.0];
    let at = |c: f64, spread: f64, sweeps: usize| {
        let dc: Vec<f64> = d.iter().map(|v| v * c).collect();
        let lams: Vec<f64> = grid.iter().map(|v| v * c).collect();
        cd_path(
            &r,
            &dc,
            &live,
            &lams,
            1.0,
            &[1.0; 5],
            sweeps,
            1e-10,
            spread * c,
        )
    };
    for spread in [1.7, f64::NAN] {
        for sweeps in [20, 100_000] {
            let base = at(1.0, spread, sweeps);
            assert!(base.coefs.iter().any(|&v| v != 0.0));
            if sweeps == 20 {
                assert!(base.unconverged.iter().any(|&u| u), "the cap's path ran");
            } else {
                assert!(base.unconverged.iter().all(|&u| !u), "the rule's path ran");
            }
            for p in [-30, -20, 20, 30] {
                let c = 2f64.powi(p);
                let scaled = at(c, spread, sweeps);
                let want: Vec<f64> = base.coefs.iter().map(|v| v * c).collect();
                assert_eq!(
                    scaled.coefs, want,
                    "spread {spread}, {sweeps} sweeps, 2^{p}"
                );
                assert_eq!(scaled.unconverged, base.unconverged, "2^{p}");
            }
        }
    }
}

/// `sweep_tol` is `tol` times the larger of the target's spread and the
/// largest `|d_i|`, the spread unknown (NaN) leaving the floor; `settled`
/// ends a descent on a move under the limit, or on no move at all, which a
/// limit of 0 still allows.
#[test]
fn the_sweep_limit_is_tol_times_the_targets_spread() {
    let d = [1.0, -5.0, 0.0];
    assert_eq!(sweep_tol(2.0, 8.0, &d), 16.0);
    assert_eq!(sweep_tol(2.0, 3.0, &d), 10.0, "the floor binds");
    assert_eq!(
        sweep_tol(2.0, f64::NAN, &d),
        10.0,
        "an unknown spread is the floor"
    );
    assert_eq!(sweep_tol(2.0, f64::NAN, &[0.0, 0.0]), 0.0);
    assert!(cd::settled(0.5, 1.0));
    assert!(
        !cd::settled(1.0, 1.0),
        "a move of exactly the limit goes on"
    );
    assert!(!cd::settled(1.5, 1.0));
    assert!(cd::settled(0.0, 0.0), "no move ends it at a limit of 0");
    assert!(!cd::settled(1e-300, 0.0));
}

/// A target with no spread, every `d_i` 0: the limit is 0, and the descent
/// from 0 moves nothing in its first sweep, which ends it. Counted
/// unconverged it would be a cap hit on every constant target.
#[test]
fn a_target_with_no_spread_converges_in_a_sweep() {
    let (r, _) = design(2, 50, 3, 0.3, &[]);
    for spread in [0.0, f64::NAN] {
        let path = cd_path(
            &r,
            &[0.0; 3],
            &[true; 3],
            &[0.1, 0.0],
            1.0,
            &[1.0; 3],
            1,
            1e-10,
            spread,
        );
        assert_eq!(path.coefs, vec![0.0; 6]);
        assert_eq!(path.unconverged, vec![false, false]);
    }
}

/// The spread the rule reads: the standard deviation of a centred fit, the
/// root mean square of one through the origin, and 0 -- the floor's cue --
/// for a second moment below 0 or NaN.
#[test]
fn the_targets_spread_is_centred_with_an_intercept_and_raw_without() {
    assert_eq!(target_spread(16.0, 3.0, true), 4.0);
    assert_eq!(target_spread(16.0, 3.0, false), 5.0);
    assert_eq!(target_spread(-1e-20, 0.0, true), 0.0);
    assert_eq!(target_spread(f64::NAN, 1.0, true), 0.0);
    assert_eq!(target_spread(f64::NAN, 1.0, false), 0.0);
}
