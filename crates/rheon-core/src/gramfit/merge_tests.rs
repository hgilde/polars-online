//! [`merge`] and [`ridge_fits`] against their definitions on the rows:
//! the pooled rows' moments computed two-pass, the residuals' variance,
//! and faer's LU (`crate::oracle`) for every solve and inverse.

use super::*;
use crate::SplitMix64;
use crate::oracle;

/// Rows: `z` with the intercept's 1 in front of `k` features, and two
/// targets, the second missing on a share of the rows.
pub(super) struct Rows {
    pub(super) z: Vec<Vec<f64>>,
    pub(super) y: Vec<[Option<f64>; 2]>,
}

pub(super) fn rows(seed: u64, n: usize, k: usize, level: f64) -> Rows {
    let mut rng = SplitMix64::new(seed);
    let mut normal = || {
        let u = 1.0 - rng.uniform();
        let v = rng.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let base = normal();
        let mut row = vec![1.0];
        row.extend((0..k).map(|j| level + normal() + 0.5 * base * (j % 2) as f64));
        let y0 = row[1..]
            .iter()
            .enumerate()
            .map(|(j, x)| x * (j as f64 - 1.0))
            .sum::<f64>()
            + normal();
        let y1 = (i % 3 != 0).then(|| row[1] - 2.0 * row[k] + 0.3 * normal());
        z.push(row);
        y.push([Some(y0), y1]);
    }
    Rows { z, y }
}

/// The Gram of `rows`, every row at weight 1, two-pass: centre, then
/// multiply. A target absent from every row has the moments the bank
/// reports for it: 0, and no Kish size.
pub(super) fn gram_of(r: &Rows) -> OwnedGram {
    let k = r.z[0].len();
    let n = r.z.len() as f64;
    let mean = |idx: &[usize], f: &dyn Fn(usize) -> f64| -> f64 {
        idx.iter().map(|&i| f(i)).sum::<f64>() / idx.len() as f64
    };
    let all: Vec<usize> = (0..r.z.len()).collect();
    let means: Vec<f64> = (0..k).map(|a| mean(&all, &|i| r.z[i][a])).collect();
    let mut comoments = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            comoments[a * k + b] = mean(&all, &|i| (r.z[i][a] - means[a]) * (r.z[i][b] - means[b]));
        }
    }
    let mut g = OwnedGram {
        k,
        weight_sum: n,
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
    for t in 0..2 {
        let own: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&i| r.y[i][t].is_some())
            .collect();
        if own.is_empty() {
            g.cross_moments.extend(vec![0.0; k]);
            g.cross_centred.extend(vec![0.0; k]);
            g.means_by_target.extend(vec![0.0; k]);
            g.target_weights.push(0.0);
            g.target_means.push(0.0);
            g.target_vars.push(0.0);
            g.target_n_kish.push(f64::NAN);
            continue;
        }
        let y = |i: usize| r.y[i][t].unwrap();
        let ybar = mean(&own, &y);
        let m: Vec<f64> = (0..k).map(|a| mean(&own, &|i| r.z[i][a])).collect();
        for (a, &ma) in m.iter().enumerate() {
            g.cross_moments.push(mean(&own, &|i| r.z[i][a] * y(i)));
            g.cross_centred
                .push(mean(&own, &|i| (r.z[i][a] - ma) * (y(i) - ybar)));
        }
        g.means_by_target.extend(m);
        g.target_weights.push(own.len() as f64);
        g.target_means.push(ybar);
        g.target_vars.push(mean(&own, &|i| (y(i) - ybar).powi(2)));
        g.target_n_kish.push(own.len() as f64);
    }
    g
}

pub(super) fn joined(parts: &[&Rows]) -> Rows {
    Rows {
        z: parts.iter().flat_map(|p| p.z.clone()).collect(),
        y: parts.iter().flat_map(|p| p.y.clone()).collect(),
    }
}

pub(super) fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

// --- Rounding bounds for the fits against the oracle (docs/PLAN.md task
// 274). `ridge_fits` factorizes by Cholesky, solves by substitution
// ([`super::trisolve`]), and reads `diag(A⁻¹)` and `m'A⁻¹m` from `L⁻¹`;
// the oracle solves and inverts by faer's LU. Each bound is the model's
// a-priori rounding plus the oracle's own, a posteriori.

/// `|x̂ − x|` for the slopes `b = x̃/s` of `ridge_fits`, which solves
/// `sys·x̃ = rhs_s` by Cholesky (Higham, Theorem 10.4, through
/// [`oracle::forward_error`]), beside the oracle's LU solve of the same
/// system (its residual, [`oracle::solve_error`]), and the division by `s`
/// on each side.
pub(super) fn slope_error(sys: &[f64], rhs_s: &[f64], s: &[f64]) -> Vec<f64> {
    let n = rhs_s.len();
    let x = oracle::solve(sys, rhs_s);
    let fe = oracle::forward_error(sys, &x, &oracle::cholesky_backward(sys), &vec![0.0; n]);
    let own = oracle::solve_error(sys, rhs_s, &x);
    (0..n)
        .map(|p| (fe[p] + own[p]) / s[p] + 2.0 * oracle::U * (x[p] / s[p]).abs())
        .collect()
}

/// faer's lower Cholesky factor `L̂` of `c` and `X̂ = L̂⁻¹`, for the bounds
/// below only: no expected value is read from them.
fn factor_and_inverse(c: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = c.len().isqrt();
    let m = faer::Mat::from_fn(n, n, |i, j| c[i * n + j]);
    let llt = m.llt(faer::Side::Lower).expect("positive definite");
    let l = llt.L();
    let mut inv = faer::Mat::<f64>::zeros(n, n);
    faer::linalg::triangular_inverse::invert_lower_triangular(inv.as_mut(), l, faer::Par::Seq);
    (
        (0..n * n).map(|ij| l[(ij / n, ij % n)]).collect(),
        (0..n * n).map(|ij| inv[(ij / n, ij % n)]).collect(),
    )
}

/// `F ≥ |X̂ − L̂⁻¹|`, the triangular inverse's forward error: `γ_n|X||L||X|`
/// for either residual bound of a substitution-based inverse (Higham §8.3,
/// Methods 1 and 2), doubled for faer's blocked recursion.
fn inverse_factor_error(l: &[f64], x: &[f64]) -> Vec<f64> {
    let n = l.len().isqrt();
    let g = oracle::gamma(2 * n);
    let lx: Vec<f64> = (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n)
                .map(|m| l[i * n + m].abs() * x[m * n + j].abs())
                .sum()
        })
        .collect();
    (0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            g * (0..n)
                .map(|m| x[i * n + m].abs() * lx[m * n + j])
                .sum::<f64>()
        })
        .collect()
}

/// The most `ridge_fits`' `diag(C⁻¹)` may be off, as it forms it: `L̂L̂ᵀ =
/// C + ΔC`, `|ΔC| ≤ γ_{n+1}√(cᵢᵢcⱼⱼ)` ([`oracle::cholesky_backward`]'s
/// step), moving `(C⁻¹)ᵢᵢ` by `(|C⁻¹|ΔC|C⁻¹|)ᵢᵢ`; `X̂ = L̂⁻¹` off by `F`
/// ([`inverse_factor_error`]), moving `Σⱼ X̂ⱼᵢ²` by `2Σⱼ|Xⱼᵢ|Fⱼᵢ`; and the
/// sum's own `γ_n`.
pub(super) fn inverse_diagonal_error(c: &[f64]) -> Vec<f64> {
    let n = c.len().isqrt();
    let inv = oracle::inverse(c);
    let d: Vec<f64> = (0..n).map(|i| c[i * n + i].sqrt()).collect();
    let (l, x) = factor_and_inverse(c);
    let f = inverse_factor_error(&l, &x);
    let g = oracle::gamma(n + 1) * (1.0 + oracle::gamma(n + 1));
    (0..n)
        .map(|i| {
            let cd: f64 = (0..n).map(|m| inv[i * n + m].abs() * d[m]).sum();
            let tri: f64 = (0..n)
                .map(|j| 2.0 * x[j * n + i].abs() * f[j * n + i])
                .sum();
            g * cd * cd + tri + oracle::gamma(n) * inv[i * n + i].abs()
        })
        .collect()
}

/// The most `ridge_fits`' `m'C⁻¹m`, read as `‖L̂⁻¹m‖²`, may be off: the
/// factor's backward error through `x = C⁻¹m`, `γ_{n+1}(Σ|xᵢ|√cᵢᵢ)²`; `F`
/// through `y = L̂⁻¹m`, `2|y|ᵀF|m|`; and the products' sums, `γ_n` of
/// `2|y|ᵀ|X̂||m| + ‖y‖²`.
pub(super) fn factor_quad_error(c: &[f64], m: &[f64]) -> f64 {
    let n = m.len();
    let xs = oracle::solve(c, m);
    let d: Vec<f64> = (0..n).map(|i| c[i * n + i].sqrt()).collect();
    let (l, x) = factor_and_inverse(c);
    let f = inverse_factor_error(&l, &x);
    let am: Vec<f64> = m.iter().map(|v| v.abs()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| x[i * n + j] * m[j]).sum())
        .collect();
    let fm = oracle::abs_times(&f, &am);
    let xm = oracle::abs_times(&x, &am);
    let g = oracle::gamma(n + 1) * (1.0 + oracle::gamma(n + 1));
    let cd: f64 = xs.iter().zip(&d).map(|(a, b)| a.abs() * b).sum();
    let yy: f64 = y.iter().map(|v| v * v).sum();
    g * cd * cd
        + (0..n)
            .map(|i| 2.0 * y[i].abs() * (fm[i] + oracle::gamma(n) * xm[i]))
            .sum::<f64>()
        + oracle::gamma(n) * yy
}

/// The most `(LU⁻¹)ᵢᵢ` and `m'(LU⁻¹)m` of the oracle may be off: its own
/// inverse's residual ([`oracle::inverse_error`]), and the quadratic
/// form's ([`oracle::quad_form_error`]).
pub(super) fn oracle_diagonal_error(c: &[f64]) -> Vec<f64> {
    let n = c.len().isqrt();
    let e = oracle::inverse_error(c);
    (0..n).map(|i| e[i * n + i]).collect()
}

/// `√(d·σ²/n)` on both sides from `d` and `d + Δd`, `σ²` shared: half `Δd`'s
/// share of `d`, and the product, the quotient and the root on each side.
pub(super) fn se_error(se: f64, d: f64, dd: f64) -> f64 {
    se * (0.5 * dd / d.abs() + 6.0 * oracle::U)
}

/// Bounds on the rounding [`gram_of`] leaves in each field it sums from
/// `rows` over the rows `own`, two-pass at weight 1: a mean `Σf/n` within
/// `γ_{N+1}Σ|f|/n` (Higham (3.5)); a centred moment `Σ(f − f̄)(g − ḡ)/n`
/// within `γ_{N+4}Σ|f − f̄||g − ḡ|/n` (two subtractions, a product, the sum
/// and the quotient) and, since the deviations from a computed mean `f̄ +
/// δ` sum to `−nδ`, `|δ_f δ_g|` more.
pub(super) struct GramErrors {
    /// The `k` column means, and the target's.
    pub(super) means: Vec<f64>,
    pub(super) ybar: f64,
    /// The `k × k` co-moments, row-major, and the target's centred
    /// cross-moments.
    pub(super) comoments: Vec<f64>,
    pub(super) cross: Vec<f64>,
}

pub(super) fn gram_errors(r: &Rows, t: usize, own: &[usize]) -> GramErrors {
    let k = r.z[0].len();
    let n = own.len() as f64;
    let g1 = oracle::gamma(own.len() + 1);
    let g4 = oracle::gamma(own.len() + 4);
    let y = |i: usize| r.y[i][t].unwrap();
    let mean = |f: &dyn Fn(usize) -> f64| own.iter().map(|&i| f(i)).sum::<f64>() / n;
    let mz: Vec<f64> = (0..k).map(|a| mean(&|i| r.z[i][a])).collect();
    let my = mean(&y);
    let means: Vec<f64> = (0..k).map(|a| g1 * mean(&|i| r.z[i][a].abs())).collect();
    let ybar = g1 * mean(&|i| y(i).abs());
    let comoments = (0..k * k)
        .map(|ab| {
            let (a, b) = (ab / k, ab % k);
            g4 * mean(&|i| ((r.z[i][a] - mz[a]) * (r.z[i][b] - mz[b])).abs()) + means[a] * means[b]
        })
        .collect();
    let cross = (0..k)
        .map(|a| g4 * mean(&|i| ((r.z[i][a] - mz[a]) * (y(i) - my)).abs()) + means[a] * ybar)
        .collect();
    GramErrors {
        means,
        ybar,
        comoments,
        cross,
    }
}

#[test]
fn a_merge_is_the_gram_of_the_pooled_rows() {
    for level in [0.0, 1e4] {
        let parts: Vec<Rows> = (0..3)
            .map(|s| rows(s, 50 + 30 * s as usize, 3, level))
            .collect();
        let grams: Vec<OwnedGram> = parts.iter().map(gram_of).collect();
        let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
        let got = merge(&views, &[0, 1, 2, 3], Some(0));
        let want = gram_of(&joined(&parts.iter().collect::<Vec<_>>()));
        let tol = 1e-12;
        assert_eq!(got.weight_sum, want.weight_sum);
        close(&got.means, &want.means, tol, "means");
        close(&got.comoments, &want.comoments, tol, "comoments");
        close(
            &got.cross_moments,
            &want.cross_moments,
            tol,
            "cross_moments",
        );
        close(
            &got.means_by_target,
            &want.means_by_target,
            tol,
            "means_by_target",
        );
        close(
            &got.cross_centred,
            &want.cross_centred,
            tol,
            "cross_centred",
        );
        assert_eq!(got.target_weights, want.target_weights);
        close(&got.target_means, &want.target_means, tol, "target_means");
        close(&got.target_vars, &want.target_vars, tol, "target_vars");
        close(
            &got.target_n_kish,
            &want.target_n_kish,
            tol,
            "target_n_kish",
        );
        assert_ne!(
            got.target_weights[0], got.target_weights[1],
            "a target with gaps"
        );
    }
}

#[test]
fn a_merge_on_some_columns_is_that_block_of_the_whole_merge() {
    let parts: Vec<Rows> = (0..3).map(|s| rows(10 + s, 40, 4, 2.0)).collect();
    let grams: Vec<OwnedGram> = parts.iter().map(gram_of).collect();
    let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
    let whole = merge(&views, &[0, 1, 2, 3, 4], Some(0));
    let cols = [0, 3, 1];
    let some = merge(&views, &cols, Some(0));
    for (r, &i) in cols.iter().enumerate() {
        assert_eq!(some.means[r], whole.means[i]);
        for (c, &j) in cols.iter().enumerate() {
            assert_eq!(some.comoments[r * 3 + c], whole.comoments[i * 5 + j]);
        }
        for t in 0..2 {
            assert_eq!(
                some.cross_centred[t * 3 + r],
                whole.cross_centred[t * 5 + i]
            );
            assert_eq!(
                some.cross_moments[t * 3 + r],
                whole.cross_moments[t * 5 + i]
            );
        }
    }
    assert_eq!(some.target_vars, whole.target_vars);
    // One part is its own union, as it is.
    let one = merge(&views[..1], &[0, 1, 2, 3, 4], Some(0));
    assert_eq!(one, grams[0]);
}

#[test]
fn a_part_with_no_weight_changes_nothing() {
    let part = rows(20, 60, 3, 1.0);
    let g = gram_of(&part);
    let mut empty = g.clone();
    empty.weight_sum = 0.0;
    empty.target_weights = vec![0.0, 0.0];
    empty.target_n_kish = vec![f64::NAN, f64::NAN];
    let got = merge(&[g.arrays(), empty.arrays()], &[0, 1, 2, 3], Some(0));
    close(&got.comoments, &g.comoments, 1e-15, "comoments");
    close(&got.cross_centred, &g.cross_centred, 1e-15, "cross_centred");
    close(&got.target_n_kish, &g.target_n_kish, 1e-15, "target_n_kish");
    let none = merge(&[empty.arrays(), empty.arrays()], &[0, 1, 2, 3], Some(0));
    assert!(none.target_n_kish.iter().all(|v| v.is_nan()));
    assert!(none.comoments.iter().all(|v| v.is_finite()));
}

#[test]
fn a_ridge_fit_solves_its_system_and_its_statistics_are_the_residuals() {
    let r = rows(30, 400, 3, 3.0);
    let g = gram_of(&r);
    let slots = [1, 2, 3];
    for (ridge, standardize) in [(0.0, false), (0.5, false), (0.5, true)] {
        let fits = ridge_fits(&g.arrays(), &[0, 1], &slots, Some(0), ridge, standardize);
        for (t, fit) in fits.iter().enumerate() {
            // The system, by faer's LU.
            let a: Vec<f64> = slots
                .iter()
                .flat_map(|&i| slots.iter().map(move |&j| (i, j)))
                .map(|(i, j)| g.comoments[i * 4 + j])
                .collect();
            let rhs: Vec<f64> = slots.iter().map(|&i| g.cross_centred[t * 4 + i]).collect();
            let s: Vec<f64> = (0..3)
                .map(|p| {
                    if standardize {
                        a[p * 3 + p].sqrt()
                    } else {
                        1.0
                    }
                })
                .collect();
            let mut sys = vec![0.0; 9];
            for p in 0..3 {
                for q in 0..3 {
                    sys[p * 3 + q] =
                        a[p * 3 + q] / (s[p] * s[q]) + if p == q { ridge } else { 0.0 };
                }
            }
            let rhs_s: Vec<f64> = (0..3).map(|p| rhs[p] / s[p]).collect();
            let x = oracle::solve(&sys, &rhs_s);
            let b: Vec<f64> = (0..3).map(|p| x[p] / s[p]).collect();
            // Task 274: the same system on both sides, Cholesky against LU
            // ([`slope_error`]); it was `1e-12` relative to `1 + |b|`.
            // Measured on aarch64 (macOS): at most 0.17 (slopes) and 0.071 (intercept) of the
            // bound.
            let db = slope_error(&sys, &rhs_s, &s);
            for p in 0..3 {
                oracle::assert_within(fit.coef[p + 1], b[p], db[p], format_args!("slope {p}"));
            }
            let m = &g.means_by_target[t * 4..(t + 1) * 4];
            let icept = g.target_means[t] - (0..3).map(|p| b[p] * m[p + 1]).sum::<f64>();
            // `ȳ − m·b`, the same sum on both sides of slopes `Δb` apart:
            // `|m|·Δb`, and each side's sum and difference.
            let bm: Vec<(f64, f64)> = (0..3).map(|p| (b[p], m[p + 1])).collect();
            let di = (0..3).map(|p| m[p + 1].abs() * db[p]).sum::<f64>()
                + 2.0 * (oracle::dot_error(bm.iter().copied()) + oracle::U * icept.abs());
            oracle::assert_within(fit.coef[0], icept, di, "intercept");
            let own: Vec<usize> = (0..r.z.len()).filter(|&i| r.y[i][t].is_some()).collect();
            let rv = if t == 0 {
                // A target on every row of the Gram: the variance of its
                // residuals, two-pass.
                let res: Vec<f64> = own
                    .iter()
                    .map(|&i| {
                        let y = r.y[i][t].unwrap();
                        y - fit
                            .coef
                            .iter()
                            .zip(&r.z[i])
                            .map(|(c, z)| c * z)
                            .sum::<f64>()
                    })
                    .collect();
                let rm = res.iter().sum::<f64>() / res.len() as f64;
                res.iter().map(|v| (v - rm).powi(2)).sum::<f64>() / res.len() as f64
            } else {
                // One with gaps reads its own rows' cross-moments beside
                // the Gram's co-moments, as `coef_stats` does.
                let bab: f64 = (0..3)
                    .map(|p| b[p] * (0..3).map(|q| a[p * 3 + q] * b[q]).sum::<f64>())
                    .sum();
                let brhs: f64 = (0..3).map(|p| b[p] * rhs[p]).sum();
                g.target_vars[t] - 2.0 * brhs + bab
            };
            // Task 274. On every row the expected value is the residuals' own
            // variance, from the model's coefficients: no oracle, the
            // tolerance kept. With gaps it is `S_yy − 2b'r + b'Ab` from the
            // oracle's `b` on the model's moments: slopes `Δb` apart move it
            // by `2|Ab − r|ᵀΔb` (and `|Δb|ᵀ|A||Δb|`), and each side's two dot
            // products, its double sum and its two differences round.
            // Measured on aarch64 (macOS): at most 0.046, 0.045 and 0.043 of the bound for
            // `resid_var`, `sigma2` and `r2`.
            let drv = if t == 0 {
                1e-10 * (1.0 + rv.abs())
            } else {
                let ab: Vec<f64> = (0..3)
                    .map(|p| (0..3).map(|q| a[p * 3 + q] * b[q]).sum::<f64>() - rhs[p])
                    .collect();
                let abs_a: Vec<f64> = a.iter().map(|v| v.abs()).collect();
                let bab = oracle::abs_form(&b, &abs_a, &b);
                let brhs: f64 = (0..3).map(|p| (b[p] * rhs[p]).abs()).sum();
                (0..3).map(|p| 2.0 * ab[p].abs() * db[p]).sum::<f64>()
                    + oracle::abs_form(&db, &abs_a, &db)
                    + 2.0
                        * (oracle::gamma(3) * 2.0 * brhs
                            + oracle::gamma(6) * bab
                            + oracle::gamma(2) * (g.target_vars[t] + 2.0 * brhs + bab))
            };
            oracle::assert_within(fit.resid_var, rv, drv, "resid_var");
            let n = own.len() as f64;
            assert_eq!(fit.n, n);
            let sigma2 = rv * n / (n - 4.0);
            let r2 = 1.0 - rv / g.target_vars[t];
            oracle::assert_within(
                fit.sigma2,
                sigma2,
                drv * n / (n - 4.0) + 6.0 * oracle::U * sigma2.abs(),
                "sigma2",
            );
            oracle::assert_within(
                fit.r2,
                r2,
                drv / g.target_vars[t] + 4.0 * oracle::U * (1.0 + r2.abs()),
                "r2",
            );
            // se from the inverse of the slots' co-moments, by faer's LU,
            // beside `diag(A⁻¹)` from `L⁻¹` (task 274: [`se_error`] of
            // [`inverse_diagonal_error`] and the oracle's own); `σ²` the
            // model's on both sides. It was `1e-10` relative to `1 + |se|`.
            // Measured on aarch64 (macOS): at most 0.14 (`se`) and 0.16 (`t`)
            // of the bound.
            let inv = oracle::inverse(&a);
            let (dd, od) = (inverse_diagonal_error(&a), oracle_diagonal_error(&a));
            for p in 0..3 {
                let se = (inv[p * 3 + p] * fit.sigma2 / n).sqrt();
                let dse = se_error(se, inv[p * 3 + p], dd[p] + od[p]);
                oracle::assert_within(fit.se[p + 1], se, dse, format_args!("se {p}"));
                let tv = fit.coef[p + 1] / se;
                let dt = tv.abs() * (dse / se + 2.0 * oracle::U);
                oracle::assert_within(fit.t[p + 1], tv, dt, format_args!("t {p}"));
            }
            // The intercept's, least squares' `(X'X)⁻¹` at it. On every row
            // of the Gram, by faer's LU of the raw moments of `[1, z]`
            // themselves; with gaps, `1 + m_t' C⁻¹ m_t` by faer's LU,
            // `m_t` the target's own column means.
            // Task 274: the model reads `1 + m'A⁻¹m` from `L⁻¹`
            // ([`factor_quad_error`]). With gaps the oracle reads the same
            // form on the same moments (its own error a posteriori); on
            // every row it reads `(R⁻¹)₀₀` of the raw moments of `[1, z]`,
            // summed afresh from the rows: there each side's sums carry
            // their own rounding ([`gram_errors`]), moving the model's form
            // by `2|x|ᵀΔm + |x|ᵀΔA|x|` (`x = A⁻¹m`) and the oracle's by
            // `(|R⁻¹|ΔR|R⁻¹|)₀₀`, `ΔR ≤ γ_{N+1}` of the raw sums of `|zᵢzⱼ|`.
            // It was `1e-10` relative to `1 + |se₀|`.
            // Measured on aarch64 (macOS): at most 0.070 (`se₀`) and 0.10 (`t₀`) of the bound.
            let mm = &m[1..];
            let (q, dq) = if t == 0 {
                let raw: Vec<f64> = (0..4)
                    .flat_map(|i| (0..4).map(move |j| (i, j)))
                    .map(|(i, j)| own.iter().map(|&o| r.z[o][i] * r.z[o][j]).sum::<f64>() / n)
                    .collect();
                let ri = oracle::inverse(&raw);
                let g1 = oracle::gamma(own.len() + 1);
                let dr: Vec<f64> = (0..16)
                    .map(|ij| {
                        g1 * own
                            .iter()
                            .map(|&o| (r.z[o][ij / 4] * r.z[o][ij % 4]).abs())
                            .sum::<f64>()
                            / n
                    })
                    .collect();
                let col0: Vec<f64> = (0..4).map(|i| ri[i * 4].abs()).collect();
                let test = oracle::abs_form(&col0, &dr, &col0) + oracle::inverse_error(&raw)[0];
                let ge = gram_errors(&r, t, &own);
                let xs = oracle::solve(&a, mm);
                let dc: Vec<f64> = slots
                    .iter()
                    .flat_map(|&i| slots.iter().map(move |&j| (i, j)))
                    .map(|(i, j)| ge.comoments[i * 4 + j])
                    .collect();
                let moved = 2.0 * (0..3).map(|p| xs[p].abs() * ge.means[p + 1]).sum::<f64>()
                    + oracle::abs_form(&xs, &dc, &xs);
                (ri[0], test + moved + factor_quad_error(&a, mm))
            } else {
                let q = 1.0 + oracle::quad_form(&a, mm);
                (
                    q,
                    factor_quad_error(&a, mm) + oracle::quad_form_error(&a, mm),
                )
            };
            let dq = dq + 2.0 * oracle::U * q;
            let se0 = (q * fit.sigma2 / n).sqrt();
            let dse0 = se_error(se0, q, dq);
            oracle::assert_within(fit.se[0], se0, dse0, "the intercept's se");
            let t0 = fit.coef[0] / se0;
            oracle::assert_within(
                fit.t[0],
                t0,
                t0.abs() * (dse0 / se0 + 2.0 * oracle::U),
                "the intercept's t",
            );
        }
    }
}

#[test]
fn a_system_that_does_not_factorize_gives_nan() {
    // A constant column under no ridge: its pivot is exactly 0.
    let mut k = rows(31, 100, 3, 0.0);
    for row in &mut k.z {
        row[2] = 4.0;
    }
    let g = gram_of(&k);
    let fits = ridge_fits(&g.arrays(), &[0], &[1, 2, 3], Some(0), 0.0, false);
    assert!(
        fits[0].coef.iter().all(|v| v.is_nan()),
        "{:?}",
        fits[0].coef
    );
    assert!(fits[0].se[1..].iter().all(|v| v.is_nan()));
    // Copies under a ridge: solvable, the copies split evenly.
    let mut r = rows(31, 100, 3, 0.0);
    for row in &mut r.z {
        row[3] = row[1];
    }
    let g = gram_of(&r);
    let fits = ridge_fits(&g.arrays(), &[0], &[1, 2, 3], Some(0), 0.1, false);
    assert!(fits[0].coef.iter().all(|v| v.is_finite()));
    assert!((fits[0].coef[1] - fits[0].coef[3]).abs() < 1e-12);
    // A constant column standardized is dropped at 0.
    let mut c = rows(32, 100, 3, 0.0);
    for row in &mut c.z {
        row[2] = 4.0;
    }
    let g = gram_of(&c);
    let fits = ridge_fits(&g.arrays(), &[0], &[1, 2, 3], Some(0), 0.1, true);
    assert_eq!(fits[0].coef[2], 0.0);
    assert!(fits[0].coef[1].is_finite() && fits[0].coef[3].is_finite());
}

/// A column of no variance among the slots -- a constant, which
/// `standardize` drops at 0, or through a Gram pooled within parts, where
/// every mean is 0 -- leaves the others their standard errors: least
/// squares' on the columns kept, `C_kept⁻¹` by faer's LU, at the fit's own
/// `sigma2`, and the intercept's `1 + m' C_kept⁻¹ m`. Its own is NaN. A
/// factor of the whole system, whose row for it is zero, made every error
/// NaN (review round 8, A-5; docs/PLAN.md task 269).
#[test]
fn a_column_of_no_variance_leaves_the_others_their_standard_errors() {
    let mut r = rows(34, 300, 3, 1.0);
    for row in &mut r.z {
        row[2] = 4.0;
    }
    let g = gram_of(&r);
    assert_eq!(
        g.comoments[2 * 4 + 2],
        0.0,
        "the case: a column of no variance"
    );
    let kept = [1usize, 3];
    let sub = |c: &[f64]| -> Vec<f64> {
        kept.iter()
            .flat_map(|&i| kept.iter().map(move |&j| c[i * 4 + j]))
            .collect()
    };
    let check = |fit: &RidgeFit, a_kept: &[f64], m: Option<&[f64]>, what: &str| {
        assert!(fit.sigma2.is_finite() && fit.n > 0.0, "{what}");
        // Task 274: `diag(C⁻¹)` from `L⁻¹` against the LU's inverse, and
        // `1 + m'C⁻¹m` from `L⁻¹` against its LU form, on the same matrix,
        // `σ²` the model's on both sides ([`se_error`]); they were `1e-12`
        // relative to `1 + |se|`.
        // Measured on aarch64 (macOS): at most 0.27 of the bound, on a `t` of 12, 3 ulps off
        // against 11: a few roundings each side, nothing to accumulate.
        let inv = oracle::inverse(a_kept);
        let (dd, od) = (
            inverse_diagonal_error(a_kept),
            oracle_diagonal_error(a_kept),
        );
        for (p, &i) in kept.iter().enumerate() {
            let se = (inv[p * 2 + p] * fit.sigma2 / fit.n).sqrt();
            let dse = se_error(se, inv[p * 2 + p], dd[p] + od[p]);
            oracle::assert_within(fit.se[i], se, dse, format_args!("{what}: se {i}"));
            let tv = fit.coef[i] / se;
            let dt = tv.abs() * (dse / se + 2.0 * oracle::U);
            oracle::assert_within(fit.t[i], tv, dt, format_args!("{what}: t {i}"));
        }
        assert!(
            fit.se[2].is_nan() && fit.t[2].is_nan(),
            "{what}: {:?}",
            fit.se
        );
        match m {
            Some(m) => {
                let mk: Vec<f64> = kept.iter().map(|&i| m[i]).collect();
                let q = 1.0 + oracle::quad_form(a_kept, &mk);
                let dq = factor_quad_error(a_kept, &mk)
                    + oracle::quad_form_error(a_kept, &mk)
                    + 2.0 * oracle::U * q;
                let se0 = (q * fit.sigma2 / fit.n).sqrt();
                oracle::assert_within(
                    fit.se[0],
                    se0,
                    se_error(se0, q, dq),
                    format_args!("{what}: se0"),
                );
            }
            None => assert!(fit.se[0].is_nan(), "{what}"),
        }
    };
    // Standardized, with an intercept: the constant dropped at 0.
    for ridge in [0.0, 0.3] {
        let fits = ridge_fits(&g.arrays(), &[0, 1], &[1, 2, 3], Some(0), ridge, true);
        for (t, fit) in fits.iter().enumerate() {
            assert_eq!(fit.coef[2], 0.0);
            let m = &g.means_by_target[t * 4..(t + 1) * 4];
            let what = format!("standardized, ridge {ridge}, target {t}");
            check(fit, &sub(&g.comoments), Some(m), &what);
        }
    }
    // Pooled within parts, through the origin: means of 0, so the raw
    // moments are the centred ones, and the constant has none.
    let mut w = g.clone();
    w.within = Some(vec![1, 1]);
    w.means.iter_mut().for_each(|m| *m = 0.0);
    w.means_by_target.iter_mut().for_each(|m| *m = 0.0);
    w.target_means.iter_mut().for_each(|m| *m = 0.0);
    w.cross_moments = w.cross_centred.clone();
    for standardize in [false, true] {
        let fits = ridge_fits(&w.arrays(), &[0, 1], &[1, 2, 3], None, 0.3, standardize);
        for (t, fit) in fits.iter().enumerate() {
            let what = format!("within, through the origin, standardize {standardize}, target {t}");
            check(fit, &sub(&w.comoments), None, &what);
        }
    }
}

/// A column `standardize` drops is no degree of freedom: the fit's
/// `sigma2` is least squares' on the columns kept, by faer's LU of the raw
/// normal equations of `[1, z1, z3]` and the residuals themselves, over
/// `n - 3`. Counting the dropped column divided by `n - 4` (review round
/// 9, B-3; docs/PLAN.md task 270). Pooled within parts, the kept columns
/// and the parts' intercepts.
#[test]
fn a_dropped_column_is_no_degree_of_freedom() {
    let mut r = rows(35, 60, 3, 1.0);
    for row in &mut r.z {
        row[2] = 1.5;
    }
    let g = gram_of(&r);
    let n = r.z.len() as f64;
    let fits = ridge_fits(&g.arrays(), &[0], &[1, 2, 3], Some(0), 0.0, true);
    let fit = &fits[0];
    assert_eq!(fit.coef[2], 0.0, "the case: the constant dropped");
    let cols = [0usize, 1, 3];
    let raw: Vec<f64> = cols
        .iter()
        .flat_map(|&i| cols.iter().map(move |&j| (i, j)))
        .map(|(i, j)| r.z.iter().map(|z| z[i] * z[j]).sum::<f64>())
        .collect();
    let rhs: Vec<f64> = cols
        .iter()
        .map(|&i| {
            r.z.iter()
                .zip(&r.y)
                .map(|(z, y)| z[i] * y[0].unwrap())
                .sum()
        })
        .collect();
    let b = oracle::solve(&raw, &rhs);
    let coef = [b[0], b[1], b[2]];
    // Task 274. The oracle solves the raw normal equations of `[1, z₁, z₃]`,
    // summed afresh from the rows: within `γ_{N+1}` of the sums of `|zᵢzⱼ|`
    // and `|zᵢy|`, through its forward error ([`oracle::forward_error`]),
    // and its own LU error. The model solves the kept columns' centred
    // system by Cholesky, standardized, from the Gram's sums
    // ([`gram_errors`]): the slopes move by `|C⁻¹|(ΔC|b| + Δc)` and by the
    // solve's backward error, and the intercept `ȳ − m·b` by `Δȳ + |b|ᵀΔm +
    // |m|ᵀΔb` and its sum. It was `1e-12` relative to `1 + |b|`.
    // Measured on aarch64 (macOS): at most 0.018 of the bound.
    let all: Vec<usize> = (0..r.z.len()).collect();
    let g1 = oracle::gamma(r.z.len() + 1);
    let dr: Vec<f64> = cols
        .iter()
        .flat_map(|&i| cols.iter().map(move |&j| (i, j)))
        .map(|(i, j)| g1 * r.z.iter().map(|z| (z[i] * z[j]).abs()).sum::<f64>())
        .collect();
    let drhs: Vec<f64> = cols
        .iter()
        .map(|&i| {
            g1 * r
                .z
                .iter()
                .zip(&r.y)
                .map(|(z, y)| (z[i] * y[0].unwrap()).abs())
                .sum::<f64>()
        })
        .collect();
    let test = oracle::forward_error(&raw, &b, &dr, &drhs);
    let own = oracle::solve_error(&raw, &rhs, &b);
    let ge = gram_errors(&r, 0, &all);
    let kept = [1usize, 3];
    let (gc, gec) = (&g.comoments, &ge.comoments);
    let c: Vec<f64> = kept
        .iter()
        .flat_map(|&i| kept.iter().map(move |&j| gc[i * 4 + j]))
        .collect();
    let dc: Vec<f64> = kept
        .iter()
        .flat_map(|&i| kept.iter().map(move |&j| gec[i * 4 + j]))
        .collect();
    let cr: Vec<f64> = kept.iter().map(|&i| g.cross_centred[i]).collect();
    let dcr: Vec<f64> = kept.iter().map(|&i| ge.cross[i]).collect();
    let sk: Vec<f64> = (0..2).map(|p| c[p * 2 + p].sqrt()).collect();
    let sys: Vec<f64> = (0..4).map(|pq| c[pq] / (sk[pq / 2] * sk[pq % 2])).collect();
    let xs = oracle::solve(&sys, &(0..2).map(|p| cr[p] / sk[p]).collect::<Vec<_>>());
    let bk: Vec<f64> = (0..2).map(|p| xs[p] / sk[p]).collect();
    let chol = oracle::forward_error(&sys, &xs, &oracle::cholesky_backward(&sys), &[0.0; 2]);
    let moved = oracle::forward_error(&c, &bk, &dc, &dcr);
    let dbk: Vec<f64> = (0..2)
        .map(|p| chol[p] / sk[p] + moved[p] + 4.0 * oracle::U * bk[p].abs())
        .collect();
    let mk: Vec<f64> = kept.iter().map(|&i| g.means_by_target[i]).collect();
    let dicept = ge.ybar
        + (0..2)
            .map(|p| bk[p].abs() * ge.means[kept[p]] + mk[p].abs() * dbk[p])
            .sum::<f64>()
        + oracle::dot_error(bk.iter().copied().zip(mk.iter().copied()))
        + oracle::U * coef[0].abs();
    let dcoef = [dicept, dbk[0], dbk[1]];
    for (p, &i) in cols.iter().enumerate() {
        oracle::assert_within(
            fit.coef[i],
            coef[p],
            dcoef[p] + test[p] + own[p],
            format_args!("least squares on the kept columns, column {i}"),
        );
    }
    let rss: f64 =
        r.z.iter()
            .zip(&r.y)
            .map(|(z, y)| {
                let e = y[0].unwrap() - cols.iter().zip(&coef).map(|(&i, c)| c * z[i]).sum::<f64>();
                e * e
            })
            .sum();
    close(
        &[fit.sigma2],
        &[rss / (n - 3.0)],
        1e-12,
        "sigma2 over n - 3",
    );
    // Task 274: `n·(R⁻¹)ₚₚ` of the raw sums (the oracle's LU, its sums'
    // rounding through `|R⁻¹|ΔR|R⁻¹|` and its own residual) against the
    // model's `diag(C⁻¹)` from `L⁻¹` and its Gram's rounding through
    // `|C⁻¹|ΔC|C⁻¹|` -- for the intercept, `1 + m'C⁻¹m` and `2|x|ᵀΔm +
    // |x|ᵀΔC|x|`, `x = C⁻¹m` -- under the two `σ²`, whose difference, held
    // above, enters as it is: `σ²`'s is a sum of residuals against a
    // difference of moments, no linear algebra, and this check is the
    // inverse's. It was `1e-10` relative to `1 + |se|`.
    // Measured on aarch64 (macOS): at most 0.019 of the bound.
    let inv = oracle::inverse(&raw);
    let ie = oracle::inverse_error(&raw);
    let ci = oracle::inverse(&c);
    let dd = inverse_diagonal_error(&c);
    let xm = oracle::solve(&c, &mk);
    let q = 1.0 + mk.iter().zip(&xm).map(|(a, b)| a * b).sum::<f64>();
    let abs_ci: Vec<f64> = ci.iter().map(|v| v.abs()).collect();
    for (p, &i) in cols.iter().enumerate() {
        let se = (inv[p * 3 + p] * rss / (n - 3.0)).sqrt();
        let col: Vec<f64> = (0..3).map(|m| inv[m * 3 + p].abs()).collect();
        let d_test = oracle::abs_form(&col, &dr, &col) + ie[p * 3 + p];
        let d_model = if p == 0 {
            factor_quad_error(&c, &mk)
                + 2.0 * (0..2).map(|m| xm[m].abs() * ge.means[kept[m]]).sum::<f64>()
                + oracle::abs_form(&xm, &dc, &xm)
                + 2.0 * oracle::U * q
        } else {
            let cc: Vec<f64> = (0..2).map(|m| abs_ci[m * 2 + p - 1]).collect();
            dd[p - 1] + oracle::abs_form(&cc, &dc, &cc)
        };
        // Both in units of `(R⁻¹)ₚₚ = d/n`.
        let s2 = rss / (n - 3.0);
        let rel = (d_test + d_model / n) / inv[p * 3 + p] + (fit.sigma2 - s2).abs() / s2;
        let dse = se * (0.5 * rel + 6.0 * oracle::U);
        oracle::assert_within(fit.se[i], se, dse, format_args!("se, column {i}"));
    }
    // Pooled within parts, through the origin: the constant has no
    // variance and is no degree of freedom either.
    let mut w = g.clone();
    w.within = Some(vec![2, 2]);
    w.means.iter_mut().for_each(|m| *m = 0.0);
    w.means_by_target.iter_mut().for_each(|m| *m = 0.0);
    w.target_means.iter_mut().for_each(|m| *m = 0.0);
    w.cross_moments = w.cross_centred.clone();
    for standardize in [false, true] {
        let fits = ridge_fits(&w.arrays(), &[0, 1], &[1, 2, 3], None, 0.3, standardize);
        for (t, fit) in fits.iter().enumerate() {
            let nk = w.target_n_kish[t];
            close(
                &[fit.sigma2],
                &[fit.resid_var * nk / (nk - 2.0 - 2.0)],
                1e-12,
                &format!("within, standardize {standardize}, target {t}"),
            );
        }
    }
}

#[test]
fn without_an_intercept_the_system_is_the_raw_moments() {
    // A level in the target that no column carries, so the fit through
    // the origin leaves a residual mean.
    let mut r = rows(33, 300, 2, 2.0);
    for y in &mut r.y {
        y[0] = y[0].map(|v| v + 3.0);
    }
    let g = gram_of(&r);
    let fits = ridge_fits(&g.arrays(), &[0], &[1, 2], None, 0.0, false);
    // E[z zᵀ] b = E[z y] over the features alone, by faer's LU.
    let raw: Vec<f64> = [1, 2]
        .iter()
        .flat_map(|&i| [1, 2].map(|j| g.comoments[i * 3 + j] + g.means[i] * g.means[j]))
        .collect();
    let rhs = [g.cross_moments[1], g.cross_moments[2]];
    let b = oracle::solve(&raw, &rhs);
    // Task 274: the same system on both sides, Cholesky against LU
    // ([`slope_error`]); it was `1e-12` relative to `1 + |b|`.
    // Measured on aarch64 (macOS): at most 0.085 of the bound.
    let db = slope_error(&raw, &rhs, &[1.0, 1.0]);
    for p in 0..2 {
        oracle::assert_within(fits[0].coef[p + 1], b[p], db[p], format_args!("slope {p}"));
    }
    assert_eq!(fits[0].coef[0], 0.0);
    // And its statistics are least squares' through the origin, read from
    // the residuals themselves: their mean square, not their variance, an
    // uncentred R², and `(Z'Z)⁻¹` of the raw moments, by faer's LU.
    let fit = &fits[0];
    let res: Vec<f64> = (0..r.z.len())
        .map(|i| r.y[i][0].unwrap() - b[0] * r.z[i][1] - b[1] * r.z[i][2])
        .collect();
    let n = res.len() as f64;
    let ms = res.iter().map(|e| e * e).sum::<f64>() / n;
    let rm = res.iter().sum::<f64>() / n;
    assert!(
        rm.abs() > 0.1,
        "a residual mean to tell the two apart: {rm}"
    );
    close(&[fit.resid_var], &[ms], 1e-10, "resid_var");
    close(&[fit.sigma2], &[ms * n / (n - 2.0)], 1e-10, "sigma2");
    let yy = (0..r.z.len())
        .map(|i| r.y[i][0].unwrap().powi(2))
        .sum::<f64>()
        / n;
    close(&[fit.r2], &[1.0 - ms / yy], 1e-10, "r2, uncentred");
    // Task 274: `diag(A⁻¹)` from `L⁻¹` against the LU's inverse of the same
    // raw moments, `σ²` the model's on both sides ([`se_error`]); it was
    // `1e-10` relative to `1 + |se|`. The three statistics above compare
    // the residuals' sums with the moments' differences, and the oracle's
    // slopes enter them only to second order (least squares, no ridge):
    // no linear algebra, their tolerance kept.
    // Measured on aarch64 (macOS): equal to the bit; the result moved by 100u of itself fails
    // it (3.8 and 4.1 times the bound).
    let inv = oracle::inverse(&raw);
    let (dd, od) = (inverse_diagonal_error(&raw), oracle_diagonal_error(&raw));
    for p in 0..2 {
        let se = (inv[p * 2 + p] * fit.sigma2 / n).sqrt();
        let dse = se_error(se, inv[p * 2 + p], dd[p] + od[p]);
        oracle::assert_within(fit.se[p + 1], se, dse, format_args!("se {p}"));
        let tv = fit.coef[p + 1] / se;
        let dt = tv.abs() * (dse / se + 2.0 * oracle::U);
        oracle::assert_within(fit.t[p + 1], tv, dt, format_args!("t {p}"));
    }
    assert!(fit.se[0].is_nan(), "column 0 is no slot");
}

/// `m` packed: the upper triangle with the diagonal, row by row.
fn packed(m: &[f64], k: usize) -> Vec<f64> {
    (0..k)
        .flat_map(|i| (i..k).map(move |j| m[i * k + j]))
        .collect()
}

#[test]
fn the_packed_index_is_the_upper_triangle_row_by_row_either_way_round() {
    let k = 5;
    let mut n = 0;
    for i in 0..k {
        for j in i..k {
            assert_eq!(Comoments::packed_index(k, i, j), n);
            assert_eq!(Comoments::packed_index(k, j, i), n);
            n += 1;
        }
    }
    assert_eq!(n, k * (k + 1) / 2);
}

#[test]
fn a_packed_or_float32_part_merges_and_fits_as_its_whole_float64_matrix() {
    let parts: Vec<Rows> = (0..3).map(|s| rows(40 + s, 60, 3, 5.0)).collect();
    let grams: Vec<OwnedGram> = parts.iter().map(gram_of).collect();
    let k = 4;
    for g in &grams {
        for i in 0..k {
            for j in 0..k {
                assert_eq!(g.comoments[i * k + j], g.comoments[j * k + i], "symmetric");
            }
        }
    }
    let full: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
    let packs: Vec<Vec<f64>> = grams.iter().map(|g| packed(&g.comoments, k)).collect();
    let as_packed: Vec<GramArrays<'_>> = full
        .iter()
        .zip(&packs)
        .map(|(g, p)| GramArrays {
            comoments: Comoments::Packed(p),
            ..*g
        })
        .collect();
    let cols = [0, 1, 2, 3];
    let want = merge(&full, &cols, Some(0));
    assert_eq!(
        merge(&as_packed, &cols, Some(0)),
        want,
        "packed: the same, to the bit"
    );
    // float32: the float64 matrices rounded to float32, then the same.
    let f32s: Vec<Vec<f32>> = grams
        .iter()
        .map(|g| g.comoments.iter().map(|&v| v as f32).collect())
        .collect();
    let widened: Vec<Vec<f64>> = f32s
        .iter()
        .map(|v| v.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let as_f32: Vec<GramArrays<'_>> = full
        .iter()
        .zip(&f32s)
        .map(|(g, v)| GramArrays {
            comoments: Comoments::Full32(v),
            ..*g
        })
        .collect();
    let as_wide: Vec<GramArrays<'_>> = full
        .iter()
        .zip(&widened)
        .map(|(g, v)| GramArrays {
            comoments: Comoments::Full(v),
            ..*g
        })
        .collect();
    let got = merge(&as_f32, &cols, Some(0));
    assert_eq!(got, merge(&as_wide, &cols, Some(0)));
    close(
        &got.comoments,
        &want.comoments,
        1e-6,
        "float32 within its rounding",
    );
    let packs32: Vec<f32> = packs[0].iter().map(|&v| v as f32).collect();
    let p32 = GramArrays {
        comoments: Comoments::Packed32(&packs32),
        ..full[0]
    };
    assert_eq!(
        merge(&[p32], &cols, Some(0)).comoments,
        merge(&as_wide[..1], &cols, Some(0)).comoments
    );
    let fits = ridge_fits(&as_packed[0], &[0, 1], &[1, 2, 3], Some(0), 0.1, true);
    let whole = ridge_fits(&full[0], &[0, 1], &[1, 2, 3], Some(0), 0.1, true);
    for (a, b) in fits.iter().zip(&whole) {
        assert_eq!(
            (&a.coef, &a.se[1..], a.resid_var),
            (&b.coef, &b.se[1..], b.resid_var)
        );
    }
}

/// `g` with its columns in the order `perm`: column `p` of the result is
/// column `perm[p]` of `g`.
fn permuted(g: &OwnedGram, perm: &[usize]) -> OwnedGram {
    let k = g.k;
    let per_target = |v: &[f64]| -> Vec<f64> {
        (0..v.len() / k)
            .flat_map(|t| perm.iter().map(move |&i| v[t * k + i]))
            .collect()
    };
    OwnedGram {
        means: perm.iter().map(|&i| g.means[i]).collect(),
        comoments: perm
            .iter()
            .flat_map(|&i| perm.iter().map(move |&j| g.comoments[i * k + j]))
            .collect(),
        cross_moments: per_target(&g.cross_moments),
        means_by_target: per_target(&g.means_by_target),
        cross_centred: per_target(&g.cross_centred),
        ..g.clone()
    }
}

/// The intercept need not be the first column: with it last, a merge and a
/// ridge fit are the same numbers, to the bit, column for column. Every
/// other test puts it first, where `t·k + c` and `t·k − c` read the same
/// entry for the first target.
#[test]
fn the_intercept_may_be_any_column() {
    let parts: Vec<Rows> = (0..2).map(|s| rows(50 + s, 70, 3, 2.0)).collect();
    let grams: Vec<OwnedGram> = parts.iter().map(gram_of).collect();
    let k = 4;
    // Old column i at position (i + 3) % 4: the intercept last.
    let perm = [1, 2, 3, 0];
    let pos = |i: usize| (i + k - 1) % k;
    let moved: Vec<OwnedGram> = grams.iter().map(|g| permuted(g, &perm)).collect();
    let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
    let moved_views: Vec<GramArrays<'_>> = moved.iter().map(OwnedGram::arrays).collect();
    let want = merge(&views, &[0, 1, 2, 3], Some(0));
    let cols: Vec<usize> = (0..k).map(pos).collect();
    assert_eq!(merge(&moved_views, &cols, Some(pos(0))), want);
    for standardize in [false, true] {
        let a = ridge_fits(&views[0], &[0, 1], &[1, 2, 3], Some(0), 0.2, standardize);
        let b = ridge_fits(
            &moved_views[0],
            &[0, 1],
            &[0, 1, 2],
            Some(pos(0)),
            0.2,
            standardize,
        );
        for (x, y) in a.iter().zip(&b) {
            let back = |v: &[f64]| (0..k).map(|i| v[pos(i)].to_bits()).collect::<Vec<_>>();
            let bits = |v: &[f64]| v.iter().map(|u| u.to_bits()).collect::<Vec<_>>();
            assert_eq!(back(&y.coef), bits(&x.coef));
            assert_eq!(back(&y.se), bits(&x.se));
            assert_eq!(
                (y.resid_var, y.sigma2, y.r2, y.n),
                (x.resid_var, x.sigma2, x.r2, x.n)
            );
        }
    }
}

/// A target's Kish size where the parts give none: a part with no weight
/// and a Kish size of 0, rather than NaN, adds nothing to the sum of
/// squared weights; parts that carry weight but no Kish size pool to none
/// (NaN), never to an infinite size; and parts with no weight at all pool
/// to finite moments, `α` and `β` both 0 (the module doc).
#[test]
fn a_part_without_a_kish_size_adds_none() {
    let g = gram_of(&rows(21, 60, 3, 1.0));
    let mut empty = g.clone();
    empty.weight_sum = 0.0;
    empty.target_weights = vec![0.0, 0.0];
    empty.target_n_kish = vec![0.0, 0.0];
    let got = merge(&[g.arrays(), empty.arrays()], &[0, 1, 2, 3], Some(0));
    assert_eq!(got.target_n_kish, g.target_n_kish);
    let none = merge(&[empty.arrays(), empty.arrays()], &[0, 1, 2, 3], Some(0));
    for (what, v) in [
        ("target_vars", &none.target_vars),
        ("target_means", &none.target_means),
        ("cross_moments", &none.cross_moments),
        ("cross_centred", &none.cross_centred),
        ("means_by_target", &none.means_by_target),
    ] {
        assert!(v.iter().all(|x| x.is_finite()), "{what}: {v:?}");
    }
    let mut unknown = g.clone();
    unknown.target_n_kish = vec![f64::NAN, f64::NAN];
    let pooled = merge(
        &[unknown.arrays(), unknown.arrays()],
        &[0, 1, 2, 3],
        Some(0),
    );
    assert!(pooled.target_weights.iter().all(|&w| w > 0.0));
    assert!(
        pooled.target_n_kish.iter().all(|v| v.is_nan()),
        "{:?}",
        pooled.target_n_kish
    );
}

/// One slot beside the intercept, its co-moment 1 and its covariance with
/// the target 1, so the ridge-0 slope is 1: a Gram built by hand for the
/// statistics' edges.
fn one_slot(var_y: f64, n_kish: f64) -> OwnedGram {
    OwnedGram {
        k: 2,
        weight_sum: n_kish,
        means: vec![1.0, 0.0],
        comoments: vec![0.0, 0.0, 0.0, 1.0],
        cross_moments: vec![3.0, 1.0],
        means_by_target: vec![1.0, 0.0],
        cross_centred: vec![0.0, 1.0],
        target_weights: vec![n_kish],
        target_means: vec![3.0],
        target_vars: vec![var_y],
        target_n_kish: vec![n_kish],
        within: None,
    }
}

/// The statistics at their edges, as `coef_stats` computes them: a
/// residual variance below 0 (a target with gaps reads its own rows'
/// covariances beside the Gram's co-moments, which need not agree) is 0,
/// so `r2` is 1, the standard error 0 and its t NaN, not infinite; at a
/// Kish size no larger than the parameters there is no `sigma2`, not an
/// infinite one; and a `sigma2` that overflows gives no standard errors.
#[test]
fn the_statistics_at_their_edges() {
    let fit = |g: &OwnedGram| ridge_fits(&g.arrays(), &[0], &[1], Some(0), 0.0, false).remove(0);
    // 0.5 − 2·1 + 1 = −0.5.
    let low = fit(&one_slot(0.5, 10.0));
    assert_eq!(low.coef, vec![3.0, 1.0]);
    assert_eq!((low.resid_var, low.sigma2, low.r2), (0.0, 0.0, 1.0));
    assert_eq!(low.se[1], 0.0);
    assert!(low.t[1].is_nan(), "{:?}", low.t);
    // Two rows of Kish's count for a slope and an intercept: 2 − 2 = 0.
    let tight = fit(&one_slot(2.0, 2.0));
    assert_eq!(tight.resid_var, 1.0);
    assert!(tight.sigma2.is_nan(), "{}", tight.sigma2);
    assert!(tight.se[1].is_nan() && tight.t[1].is_nan());
    // f64::MAX · 10 / 8 overflows.
    let huge = fit(&one_slot(f64::MAX, 10.0));
    assert_eq!(huge.sigma2, f64::INFINITY);
    assert!(
        huge.se[1].is_nan() && huge.t[1].is_nan(),
        "{:?} {:?}",
        huge.se,
        huge.t
    );
}

/// A constant target has no `r2` about its mean, whatever its residual
/// variance. Through the origin the statistics are raw, as least squares
/// without a constant reads them: a target constant at 5 on a column of
/// mean 2 and variance 1 has `E[x²] = 5`, `E[x y] = 10`, `b = 2`, and
/// residuals `5 − 2x` of mean square `25 − 2·2·10 + 2·5·2 = 5`, an
/// uncentred `r2` of `1 − 5/25` -- statsmodels' 0.8 for that fit. With an
/// intercept the residuals are 0 and there is no variance to explain.
#[test]
fn a_constant_target_has_no_r2() {
    let g = OwnedGram {
        k: 1,
        weight_sum: 10.0,
        means: vec![2.0],
        comoments: vec![1.0],
        cross_moments: vec![10.0],
        means_by_target: vec![2.0],
        cross_centred: vec![0.0],
        target_weights: vec![10.0],
        target_means: vec![5.0],
        target_vars: vec![0.0],
        target_n_kish: vec![10.0],
        within: None,
    };
    let fit = ridge_fits(&g.arrays(), &[0], &[0], None, 0.0, false).remove(0);
    assert_eq!((fit.coef[0], fit.resid_var), (2.0, 5.0));
    assert!((fit.r2 - 0.8).abs() < 1e-15, "{}", fit.r2);
    // The same rows with the intercept's column: `b = 0`, no residual.
    let g = OwnedGram {
        k: 2,
        means: vec![1.0, 2.0],
        comoments: vec![0.0, 0.0, 0.0, 1.0],
        cross_moments: vec![5.0, 10.0],
        means_by_target: vec![1.0, 2.0],
        cross_centred: vec![0.0, 0.0],
        ..g
    };
    let fit = ridge_fits(&g.arrays(), &[0], &[1], Some(0), 0.0, false).remove(0);
    assert_eq!((fit.coef[0], fit.coef[1], fit.resid_var), (5.0, 0.0, 0.0));
    assert!(fit.r2.is_nan(), "{}", fit.r2);
    // An exact fit's errors are 0, and a t over an error of 0 is NaN, the
    // intercept's as the slope's, never an infinite t.
    assert_eq!((fit.se[0], fit.se[1]), (0.0, 0.0));
    assert!(fit.t[0].is_nan() && fit.t[1].is_nan(), "{:?}", fit.t);
}
