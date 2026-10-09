//! [`merge`] and [`ridge_fits`] against their definitions on the rows:
//! the pooled rows' moments computed two-pass, the residuals' variance,
//! and faer's LU (`crate::oracle`) for every solve and inverse.

use super::*;
use crate::SplitMix64;
use crate::oracle;

/// Rows: `z` with the intercept's 1 in front of `k` features, and two
/// targets, the second missing on a share of the rows.
struct Rows {
    z: Vec<Vec<f64>>,
    y: Vec<[Option<f64>; 2]>,
}

fn rows(seed: u64, n: usize, k: usize, level: f64) -> Rows {
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
/// multiply.
fn gram_of(r: &Rows) -> OwnedGram {
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
    };
    for t in 0..2 {
        let own: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&i| r.y[i][t].is_some())
            .collect();
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

fn joined(parts: &[&Rows]) -> Rows {
    Rows {
        z: parts.iter().flat_map(|p| p.z.clone()).collect(),
        y: parts.iter().flat_map(|p| p.y.clone()).collect(),
    }
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
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
            assert!(
                fit.se[0].is_nan() && fit.t[0].is_nan(),
                "none for the intercept"
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

#[test]
fn without_an_intercept_the_system_is_the_raw_moments() {
    let r = rows(33, 300, 2, 2.0);
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
