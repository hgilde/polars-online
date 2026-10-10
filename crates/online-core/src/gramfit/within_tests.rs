//! [`merge_within`] and the fits on what it returns (docs/PLAN.md task
//! 240), against their definitions on the rows: each part's rows demeaned
//! by its own means, then the Gram of all of them, two-pass; and every
//! solve and inverse by faer's LU (`crate::oracle`).

use super::merge_tests::{Rows, close, gram_of, rows};
use super::*;
use crate::oracle;

/// Three parts at levels far apart, so the gaps between their means are
/// what a union would add; the third has no rows of the second target.
fn parts(level: f64) -> Vec<Rows> {
    let mut out: Vec<Rows> = (0..3)
        .map(|s| rows(60 + s, 40 + 25 * s as usize, 3, level * (1.0 + s as f64)))
        .collect();
    for y in &mut out[2].y {
        y[1] = None;
    }
    out
}

fn mean(v: impl Iterator<Item = f64> + Clone) -> f64 {
    v.clone().sum::<f64>() / v.count() as f64
}

/// The Gram of the rows of `parts`, each part's demeaned by its own means:
/// for the co-moments, the features about their means over the part's
/// rows; for a target, the features about their means over its rows in
/// the part and the target about its own mean there. The intercept stays
/// the constant 1. Every row at weight 1, two-pass. Per target, the parts
/// it has rows in.
fn demeaned(parts: &[Rows]) -> OwnedGram {
    let k = parts[0].z[0].len();
    let mut dz: Vec<Vec<f64>> = Vec::new();
    for p in parts {
        let m: Vec<f64> = (0..k).map(|a| mean(p.z.iter().map(|r| r[a]))).collect();
        dz.extend(p.z.iter().map(|r| (0..k).map(|a| r[a] - m[a]).collect()));
    }
    let w = dz.len() as f64;
    let mut means = vec![0.0; k];
    means[0] = 1.0;
    let mut g = OwnedGram {
        k,
        weight_sum: w,
        means: means.clone(),
        comoments: (0..k * k)
            .map(|e| dz.iter().map(|r| r[e / k] * r[e % k]).sum::<f64>() / w)
            .collect(),
        cross_moments: Vec::new(),
        means_by_target: Vec::new(),
        cross_centred: Vec::new(),
        target_weights: Vec::new(),
        target_means: Vec::new(),
        target_vars: Vec::new(),
        target_n_kish: Vec::new(),
        within: Some(Vec::new()),
    };
    for t in 0..2 {
        let mut pairs: Vec<(Vec<f64>, f64)> = Vec::new();
        let mut count = 0;
        for p in parts {
            let own: Vec<usize> = (0..p.z.len()).filter(|&i| p.y[i][t].is_some()).collect();
            if own.is_empty() {
                continue;
            }
            count += 1;
            let ybar = mean(own.iter().map(|&i| p.y[i][t].unwrap()));
            let m: Vec<f64> = (0..k)
                .map(|a| mean(own.iter().map(|&i| p.z[i][a])))
                .collect();
            pairs.extend(own.iter().map(|&i| {
                let z = (0..k).map(|a| p.z[i][a] - m[a]).collect();
                (z, p.y[i][t].unwrap() - ybar)
            }));
        }
        let tw = pairs.len() as f64;
        let cc: Vec<f64> = (0..k)
            .map(|a| pairs.iter().map(|(z, y)| z[a] * y).sum::<f64>() / tw)
            .collect();
        g.cross_moments.extend(&cc);
        g.cross_centred.extend(&cc);
        g.means_by_target.extend(&means);
        g.target_weights.push(tw);
        g.target_means.push(0.0);
        g.target_vars
            .push(pairs.iter().map(|(_, y)| y * y).sum::<f64>() / tw);
        g.target_n_kish.push(tw);
        g.within.as_mut().unwrap().push(count);
    }
    g
}

const ALL: [usize; 4] = [0, 1, 2, 3];

#[test]
fn a_within_merge_is_the_gram_of_the_rows_demeaned_per_part() {
    for level in [0.0, 1e4] {
        let parts = parts(level);
        let grams: Vec<OwnedGram> = parts.iter().map(gram_of).collect();
        let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
        let got = merge_within(&views, &ALL, Some(0));
        let want = demeaned(&parts);
        let tol = 1e-12;
        assert_eq!(got.weight_sum, want.weight_sum);
        assert_eq!(got.means, want.means, "0, and the intercept's 1");
        close(&got.comoments, &want.comoments, tol, "comoments");
        close(
            &got.cross_moments,
            &want.cross_moments,
            tol,
            "cross_moments",
        );
        close(
            &got.cross_centred,
            &want.cross_centred,
            tol,
            "cross_centred",
        );
        assert_eq!(got.means_by_target, want.means_by_target);
        assert_eq!(got.target_weights, want.target_weights);
        assert_eq!(got.target_means, want.target_means);
        close(&got.target_vars, &want.target_vars, tol, "target_vars");
        close(
            &got.target_n_kish,
            &want.target_n_kish,
            tol,
            "target_n_kish",
        );
        assert_eq!(
            got.within,
            Some(vec![3, 2]),
            "the third has no second target"
        );
        // The union is another Gram: the gaps between the parts' means.
        let union = merge(&views, &ALL, Some(0));
        if level > 0.0 {
            assert!((union.comoments[5] - got.comoments[5]).abs() > level);
        }
        assert_eq!(union.within, None);
    }
}

/// The parts' means made equal, column by column and target by target:
/// every gap the union adds is 0, so the two forms are the same numbers,
/// to the bit.
#[test]
fn parts_with_equal_means_pool_within_as_they_pool_together() {
    let mut grams: Vec<OwnedGram> = parts(3.0).iter().map(gram_of).collect();
    let first = grams[0].clone();
    for g in &mut grams[1..] {
        g.means.clone_from(&first.means);
        g.means_by_target.clone_from(&first.means_by_target);
        g.target_means.clone_from(&first.target_means);
        for t in 0..2 {
            g.cross_moments[t * 4] = first.cross_moments[t * 4];
        }
        // The third part has no rows of the second target: its weight 0
        // takes it out of both forms.
    }
    let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
    let within = merge_within(&views, &ALL, Some(0));
    let union = merge(&views, &ALL, Some(0));
    assert_eq!(within.weight_sum, union.weight_sum);
    assert_eq!(within.comoments, union.comoments);
    assert_eq!(within.cross_centred, union.cross_centred);
    assert_eq!(within.target_weights, union.target_weights);
    assert_eq!(within.target_vars, union.target_vars);
    assert_eq!(within.target_n_kish, union.target_n_kish);
}

#[test]
fn one_part_within_is_its_own_moments_about_its_means() {
    for p in parts(2.0).iter().map(gram_of) {
        let got = merge_within(&[p.arrays()], &ALL, Some(0));
        assert_eq!(got.comoments, p.comoments);
        assert_eq!(got.cross_centred, p.cross_centred);
        assert_eq!(
            got.cross_moments, p.cross_centred,
            "centred: its means are 0"
        );
        assert_eq!(got.target_vars, p.target_vars);
        assert_eq!(
            (got.weight_sum, &got.target_weights),
            (p.weight_sum, &p.target_weights)
        );
        assert_eq!(got.means, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(got.target_means, vec![0.0, 0.0]);
        let present: Vec<usize> = p
            .target_weights
            .iter()
            .map(|&w| usize::from(w > 0.0))
            .collect();
        assert_eq!(got.within, Some(present));
    }
}

/// A part already pooled within is folded in as its moments: pooling it
/// again within, beside more parts, is the one merge of them all, to the
/// bit, and it brings its parts to the count.
#[test]
fn a_part_pooled_within_pools_again_as_its_parts() {
    let grams: Vec<OwnedGram> = parts(5.0).iter().map(gram_of).collect();
    let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
    let first_two = merge_within(&views[..2], &ALL, Some(0));
    let nested = merge_within(&[first_two.arrays(), views[2]], &ALL, Some(0));
    assert_eq!(nested, merge_within(&views, &ALL, Some(0)));
    assert_eq!(nested.within, Some(vec![3, 2]));
    // A union of one part is that part, its count kept.
    assert_eq!(merge(&[first_two.arrays()], &ALL, Some(0)), first_two);
}

/// The within residuals of target `t` on `parts` at slopes `b`, two-pass:
/// each row less its part's means over the target's rows, so each part's
/// own intercept is fitted and the residuals average 0 in every part.
fn within_residuals(parts: &[Rows], t: usize, b: &[f64]) -> Vec<f64> {
    let mut res = Vec::new();
    for p in parts {
        let own: Vec<usize> = (0..p.z.len()).filter(|&i| p.y[i][t].is_some()).collect();
        if own.is_empty() {
            continue;
        }
        let ybar = mean(own.iter().map(|&i| p.y[i][t].unwrap()));
        let m: Vec<f64> = (1..4)
            .map(|a| mean(own.iter().map(|&i| p.z[i][a])))
            .collect();
        res.extend(own.iter().map(|&i| {
            let fitted: f64 = (0..3).map(|q| b[q] * (p.z[i][q + 1] - m[q])).sum();
            p.y[i][t].unwrap() - ybar - fitted
        }));
    }
    res
}

/// On a Gram pooled within, a fit has the within estimator's slopes, no
/// intercept, and a residual variance and standard errors whose degrees of
/// freedom lose one per part the target has rows in.
#[test]
fn a_fit_within_has_no_intercept_and_a_degree_less_per_part() {
    let slots = [1, 2, 3];
    for absent in [false, true] {
        // The first target on every row; or absent from the third part,
        // which then spends no degree of freedom on it.
        let mut parts = parts(1e3);
        if absent {
            for y in &mut parts[2].y {
                y[0] = None;
            }
        }
        let grams: Vec<OwnedGram> = parts.iter().map(gram_of).collect();
        let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
        let g = merge_within(&views, &ALL, Some(0));
        let fits = ridge_fits(&g.arrays(), &[0, 1], &slots, Some(0), 0.0, false);
        let como = &g.comoments;
        let a: Vec<f64> = slots
            .iter()
            .flat_map(|&i| slots.iter().map(move |&j| como[i * 4 + j]))
            .collect();
        let inv = oracle::inverse(&a);
        for (t, fit) in fits.iter().enumerate() {
            assert!(fit.coef[0].is_nan(), "no intercept: {:?}", fit.coef);
            let rhs: Vec<f64> = slots.iter().map(|&i| g.cross_centred[t * 4 + i]).collect();
            let b = oracle::solve(&a, &rhs);
            close(&fit.coef[1..], &b, 1e-10, "slopes");
            let res = within_residuals(&parts, t, &b);
            let n = res.len() as f64;
            assert_eq!(fit.n, n);
            let rv = if t == 0 && !absent {
                // On every row of the Gram: the variance of the residuals.
                res.iter().map(|e| e * e).sum::<f64>() / n
            } else {
                // Not: its own rows' moments beside the Gram's co-moments,
                // as `coef_stats` reads them.
                let bab: f64 = (0..3)
                    .map(|p| b[p] * (0..3).map(|q| a[p * 3 + q] * b[q]).sum::<f64>())
                    .sum();
                let brhs: f64 = (0..3).map(|p| b[p] * rhs[p]).sum();
                (g.target_vars[t] - 2.0 * brhs + bab).max(0.0)
            };
            close(&[fit.resid_var], &[rv], 1e-9, "resid_var");
            let groups = [if absent { 2.0 } else { 3.0 }, 2.0][t];
            assert_eq!(g.within.as_ref().unwrap()[t] as f64, groups);
            let sigma2 = rv * n / (n - 3.0 - groups);
            close(
                &[fit.sigma2],
                &[sigma2],
                1e-9,
                "sigma2: a degree less per part",
            );
            for p in 0..3 {
                let se = (inv[p * 3 + p] * sigma2 / n).sqrt();
                close(&[fit.se[p + 1]], &[se], 1e-9, "se");
            }
            assert!(fit.se[0].is_nan() && fit.t[0].is_nan());
        }
        assert!(fits[0].sigma2 > 0.0, "a residual to count degrees by");
        // Without the intercept the demeaned rows' raw moments are their
        // centred ones: the same fit, and nothing to recover.
        let origin = ridge_fits(&g.arrays(), &[0, 1], &slots, None, 0.0, false);
        for (o, f) in origin.iter().zip(&fits) {
            assert_eq!(o.coef[1..], f.coef[1..]);
            assert_eq!(o.coef[0], 0.0, "column 0 is no slot");
            assert_eq!((o.resid_var, o.sigma2, o.n), (f.resid_var, f.sigma2, f.n));
            assert_eq!(o.se[1..], f.se[1..]);
        }
    }
}

/// A Gram pooled within has no intercept to recover, so a path's or a
/// ridge's is NaN: its `ȳ` is the mean of the demeaned target, 0, which
/// would read as a fitted intercept of `−m · b`.
#[test]
fn a_gram_pooled_within_recovers_no_intercept() {
    let grams: Vec<OwnedGram> = parts(0.0).iter().map(gram_of).collect();
    let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
    let g = merge_within(&views, &ALL, Some(0));
    let resp = Response::of(&g.arrays(), 0, &[1, 2, 3], Some(0));
    assert!(resp.ybar.is_nan());
    let corr = Correlation::of(&Design::of(&g.arrays(), &[1, 2, 3], Some(0)));
    let coef = resp.coef(&corr.scaling, &[0.5, 0.0, 0.0], 4, &[1, 2, 3], Some(0));
    assert!(coef[0].is_nan());
    assert!(coef[1..].iter().all(|c| c.is_finite()));
    // The same arrays, not marked: the intercept is read, as before.
    let plain = GramArrays {
        within: None,
        ..g.arrays()
    };
    assert_eq!(Response::of(&plain, 0, &[1, 2, 3], Some(0)).ybar, 0.0);
}

/// `parts` with every target's value negated on the rows of each part whose
/// sign is −1.
fn negated(parts: &[Rows], signs: &[f64]) -> Vec<Rows> {
    parts
        .iter()
        .zip(signs)
        .map(|(p, &s)| Rows {
            z: p.z.clone(),
            y: p.y
                .iter()
                .map(|y| y.map(|v| v.map(|x| if s < 0.0 { -x } else { x })))
                .collect(),
        })
        .collect()
}

/// A sign per part on the target is what negating the part's target rows
/// gives, to the bit: the centred cross-moments negate, and the
/// co-moments, the target's weight and its variance do not move. Every
/// sign +1 is no signs.
#[test]
fn a_sign_per_part_is_the_part_with_its_target_negated() {
    let parts = parts(2e3);
    let grams: Vec<OwnedGram> = parts.iter().map(gram_of).collect();
    let views: Vec<GramArrays<'_>> = grams.iter().map(OwnedGram::arrays).collect();
    let g = merge_within(&views, &ALL, Some(0));
    for signs in [[1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]] {
        let flipped: Vec<OwnedGram> = negated(&parts, &signs).iter().map(gram_of).collect();
        let flipped_views: Vec<GramArrays<'_>> = flipped.iter().map(OwnedGram::arrays).collect();
        let want = merge_within(&flipped_views, &ALL, Some(0));
        let cc = within_cross(&views, &ALL, &signs);
        assert_eq!(cc, want.cross_centred, "{signs:?}");
        assert_eq!(
            (&g.comoments, &g.target_vars, &g.target_weights),
            (&want.comoments, &want.target_vars, &want.target_weights)
        );
        // And so are the fits on it.
        let signed = GramArrays {
            cross_moments: &cc,
            cross_centred: &cc,
            ..g.arrays()
        };
        let a = ridge_fits(&signed, &[0, 1], &[1, 2, 3], Some(0), 0.1, true);
        let b = ridge_fits(&want.arrays(), &[0, 1], &[1, 2, 3], Some(0), 0.1, true);
        assert_eq!(a.len(), 2);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.coef[1..], y.coef[1..]);
            assert_eq!((x.resid_var, x.sigma2), (y.resid_var, y.sigma2));
        }
    }
    let all_plus = within_cross(&views, &ALL, &[1.0, 1.0, 1.0]);
    assert_eq!(all_plus, within_cross(&views, &ALL, &[]));
    assert_eq!(all_plus, g.cross_centred);
    // Negating every part negates the pooled cross-moments, exactly.
    let all_minus = within_cross(&views, &ALL, &[-1.0, -1.0, -1.0]);
    let neg: Vec<f64> = g.cross_centred.iter().map(|v| -v).collect();
    assert_eq!(all_minus, neg);
}
