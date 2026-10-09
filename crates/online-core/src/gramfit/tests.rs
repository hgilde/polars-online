//! The Gram fits against the definition of the lasso and against faer's LU
//! (`crate::oracle`), which shares no arithmetic with the LARS factor.

use super::*;
use crate::SplitMix64;
use crate::oracle;

/// A Gram in correlation form from `n` rows of `k` columns: column `j` is
/// `z_j + mix · z_0` for standard normal `z`, and `y = Σ_j beta_j x_j +
/// noise`. Returns `(R, d)`, standardized so `R` has a unit diagonal.
fn design(seed: u64, n: usize, k: usize, mix: f64, beta: &[f64]) -> (Vec<f64>, Vec<f64>) {
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
fn standardized(x: &[f64], y: &[f64], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
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
fn assert_kkt(r: &[f64], d: &[f64], b: &[f64], lambda: f64, tol: f64) {
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

#[test]
fn every_knot_meets_the_lasso_conditions_and_the_active_set_solve() {
    let beta = [1.0, -0.8, 0.6, 0.0, 0.3, 0.0, -0.2, 0.0];
    let (r, d) = design(7, 400, 8, 0.6, &beta);
    let live = vec![true; 8];
    let path = lars_lasso(&r, &d, &live, LarsLimits::default());
    assert_eq!(path.stop, LarsStop::End);
    assert_eq!(*path.penalties.last().unwrap(), 0.0);
    assert!(path.len() >= 9, "eight columns enter: {}", path.len());
    for w in path.penalties.windows(2) {
        assert!(w[1] < w[0], "λ falls along the path: {w:?}");
    }
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
        for p in 0..m {
            assert!(
                (b[act[p]] - want[p]).abs() <= 1e-12 * (1.0 + want[p].abs()),
                "knot {n}, column {}: {} vs {}",
                act[p],
                b[act[p]],
                want[p]
            );
        }
    }
    // The end of the path is the least-squares fit (faer's LU).
    let last = &path.coefs[(path.len() - 1) * 8..];
    let ls = oracle::solve(&r, &d);
    for j in 0..8 {
        assert!(
            (last[j] - ls[j]).abs() < 1e-12,
            "{j}: {} vs {}",
            last[j],
            ls[j]
        );
    }
}

/// A path with a column leaving: correlated columns with signs that
/// disagree, the first of seeds 0..200 whose path has one.
fn path_with_a_drop() -> (Vec<f64>, Vec<f64>, LarsPath) {
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
    for j in 0..k {
        assert!(
            (last[j] - ls[j]).abs() < 1e-10,
            "{j}: {} vs {}",
            last[j],
            ls[j]
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
fn supports(path: &LarsPath) -> Vec<Vec<usize>> {
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
fn knot_support(path: &LarsPath, k: usize, n: usize) -> Vec<usize> {
    (0..k).filter(|&j| path.coefs[n * k + j] != 0.0).collect()
}

/// [`design`] with every column a mix of all of them, `x = z (I + M)` for
/// a standard normal `M`: correlations of both signs, which is what makes
/// columns leave a lasso path and come back.
fn mixed_design(seed: u64, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
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
        for j in 0..10 {
            assert!(
                (last[j] - ls[j]).abs() < 1e-9,
                "seed {seed}, {j}: {} vs {}",
                last[j],
                ls[j]
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
    let cd = cd_path(&r, &d, &[true; 8], &grid, 1.0, &[1.0; 8], 10_000, 1e-13);
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
    let b = cd_path(&r, &d, &live, &[0.2, lam], ratio, &w, 10_000, 1e-14);
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
    let one = cd_path(&r, &d, &[true; 4], &[0.0], 1.0, &[1.0; 4], 1, 1e-7);
    let many = cd_path(&r, &d, &[true; 4], &[0.0], 1.0, &[1.0; 4], 10_000, 1e-7);
    assert_ne!(one, many, "one sweep is not the converged fit");
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
        comoments: &como,
        cross_moments: &cross,
        means_by_target: &mbt,
        cross_centred: &cc,
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
        comoments: &como0,
        ..g
    };
    let corr = Correlation::of(&Design::of(&g0, &[1, 2], Some(0)));
    assert_eq!(corr.scaling.live, vec![false, false]);
    assert_eq!(corr.r, vec![1.0, 0.0, 0.0, 1.0]);
}
