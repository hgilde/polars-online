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
            let x = oracle::solve(&sys, &(0..3).map(|p| rhs[p] / s[p]).collect::<Vec<_>>());
            let b: Vec<f64> = (0..3).map(|p| x[p] / s[p]).collect();
            close(&fit.coef[1..], &b, 1e-12, "slopes");
            let m = &g.means_by_target[t * 4..(t + 1) * 4];
            let icept = g.target_means[t] - (0..3).map(|p| b[p] * m[p + 1]).sum::<f64>();
            close(&fit.coef[..1], &[icept], 1e-12, "intercept");
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
            close(&[fit.resid_var], &[rv], 1e-10, "resid_var");
            let n = own.len() as f64;
            assert_eq!(fit.n, n);
            close(&[fit.sigma2], &[rv * n / (n - 4.0)], 1e-10, "sigma2");
            close(&[fit.r2], &[1.0 - rv / g.target_vars[t]], 1e-10, "r2");
            // se from the inverse of the slots' co-moments, by faer's LU.
            let inv = oracle::inverse(&a);
            for p in 0..3 {
                let se = (inv[p * 3 + p] * fit.sigma2 / n).sqrt();
                close(&[fit.se[p + 1]], &[se], 1e-10, "se");
                close(&[fit.t[p + 1]], &[fit.coef[p + 1] / se], 1e-10, "t");
            }
            // The intercept's, least squares' `(X'X)⁻¹` at it. On every row
            // of the Gram, by faer's LU of the raw moments of `[1, z]`
            // themselves; with gaps, `1 + m_t' C⁻¹ m_t` by faer's LU,
            // `m_t` the target's own column means.
            let q = if t == 0 {
                let raw: Vec<f64> = (0..4)
                    .flat_map(|i| (0..4).map(move |j| (i, j)))
                    .map(|(i, j)| own.iter().map(|&o| r.z[o][i] * r.z[o][j]).sum::<f64>() / n)
                    .collect();
                oracle::inverse(&raw)[0]
            } else {
                1.0 + oracle::quad_form(&a, &m[1..])
            };
            let se0 = (q * fit.sigma2 / n).sqrt();
            close(&[fit.se[0]], &[se0], 1e-10, "the intercept's se");
            close(&[fit.t[0]], &[fit.coef[0] / se0], 1e-10, "the intercept's t");
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
    let b = oracle::solve(&raw, &[g.cross_moments[1], g.cross_moments[2]]);
    close(&fits[0].coef[1..], &b, 1e-12, "slopes");
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
    assert!(rm.abs() > 0.1, "a residual mean to tell the two apart: {rm}");
    close(&[fit.resid_var], &[ms], 1e-10, "resid_var");
    close(&[fit.sigma2], &[ms * n / (n - 2.0)], 1e-10, "sigma2");
    let yy = (0..r.z.len()).map(|i| r.y[i][0].unwrap().powi(2)).sum::<f64>() / n;
    close(&[fit.r2], &[1.0 - ms / yy], 1e-10, "r2, uncentred");
    let inv = oracle::inverse(&raw);
    for p in 0..2 {
        let se = (inv[p * 2 + p] * fit.sigma2 / n).sqrt();
        close(&[fit.se[p + 1]], &[se], 1e-10, "se");
        close(&[fit.t[p + 1]], &[fit.coef[p + 1] / se], 1e-10, "t");
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
}
