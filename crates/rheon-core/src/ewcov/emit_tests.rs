//! The statistics' emit: the correlations, the partial correlations and the
//! lagged correlations take their quotient whatever the denominator is and
//! choose after, so the compiler divides two at a time. Each is held to the
//! form that divides only where the denominator is positive, to the bit, on
//! the values a stream reaches: signed zeros, subnormals, overflow, NaN and
//! infinity, a column with no spread.

use super::*;
use crate::model::OnlineModel;
use crate::oracle::CanonBits;
use std::hint::black_box;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A number of the kinds the emit meets: mostly ordinary, often a value
/// that decides a branch or a clamp.
fn awkward(state: &mut u64) -> f64 {
    let u = lcg(state);
    let pick = lcg(state).abs();
    if pick < 0.05 {
        0.0
    } else if pick < 0.1 {
        -0.0
    } else if pick < 0.13 {
        f64::NAN
    } else if pick < 0.16 {
        f64::INFINITY
    } else if pick < 0.19 {
        f64::NEG_INFINITY
    } else if pick < 0.23 {
        1e-310 * u
    } else if pick < 0.27 {
        1e160 * u
    } else if pick < 0.3 {
        1.0
    } else if pick < 0.33 {
        -1.0
    } else {
        u
    }
}

/// The correlation as every build computed it before the quotient was
/// taken ahead of the choice.
fn correlation_by_branch(cij: f64, si: f64, sj: f64) -> f64 {
    let d = si * sj;
    if d > 0.0 {
        (cij / d).clamp(-1.0, 1.0)
    } else {
        f64::NAN
    }
}

#[test]
fn the_correlation_is_the_branching_form_to_the_bit() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -3.25,
        1e-310,
        -1e-310,
        1e-160,
        1e154,
        -1e154,
        1e300,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
    ];
    let (mut nan, mut up, mut down, mut inside, mut negative_zero) = (0, 0, 0, 0, 0);
    for &cij in &values {
        for &si in &values {
            for &sj in &values {
                let (cij, si, sj) = (black_box(cij), black_box(si), black_box(sj));
                let want = correlation_by_branch(cij, si, sj);
                let got = correlation(cij, si, sj);
                assert_eq!(
                    got.canon_bits(),
                    want.canon_bits(),
                    "correlation({cij:e}, {si:e}, {sj:e})"
                );
                nan += usize::from(got.is_nan());
                up += usize::from(got == 1.0);
                down += usize::from(got == -1.0);
                inside += usize::from(got.abs() < 1.0 && got != 0.0);
                negative_zero += usize::from(got == 0.0 && got.is_sign_negative());
            }
        }
    }
    // Every kind of answer was reached.
    assert!(nan > 100 && up > 10 && down > 10 && inside > 10 && negative_zero > 0);
}

#[test]
fn the_correlations_of_a_triangle_are_the_branching_form_to_the_bit() {
    // The packed triangle's pairs through the method the emit calls: every
    // pair `(i, j)` once, at its place, for the widths that leave over a
    // different count of rows from the fours.
    let mut state = 11;
    for k in (1..=26).chain([33, 50]) {
        for round in 0..6 {
            let mut tri = LowerTriangle::zeros(k);
            let mut full = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..=i {
                    let v = awkward(&mut state);
                    full[i * k + j] = v;
                    full[j * k + i] = v;
                }
            }
            tri.copy_from(&full);
            let std: Vec<f64> = (0..k).map(|_| awkward(&mut state)).collect();
            let mut got = vec![f64::MAX; k * (k - 1) / 2];
            tri.upper_rows_map(k, &mut got, &std, correlation);
            let mut at = 0;
            for i in 0..k {
                for j in (i + 1)..k {
                    let want = correlation_by_branch(tri.get(k, j, i), std[i], std[j]);
                    assert_eq!(
                        got[at].to_bits(),
                        want.to_bits(),
                        "k = {k}, round {round}, pair ({i}, {j})"
                    );
                    at += 1;
                }
            }
            assert_eq!(at, got.len());
        }
    }
}

#[test]
fn the_partial_correlations_are_the_pair_at_a_time_form_to_the_bit() {
    let mut state = 23;
    let (mut nan, mut clamped, mut inside) = (0, 0, 0);
    for k in (1..=26).chain([33, 50]) {
        for _ in 0..6 {
            let p: Vec<f64> = (0..k * k).map(|_| awkward(&mut state)).collect();
            // Sentinels where the row is longer than the scratch it reads.
            let mut got = vec![f64::MAX; k * (k - 1) / 2];
            let mut scratch = vec![7.0; 3];
            partial_corrs_into(&p, k, &mut got, &mut scratch);
            assert_eq!(scratch.len(), k, "the scratch holds the diagonal's roots");
            let mut at = 0;
            for i in 0..k {
                for j in (i + 1)..k {
                    let want = partial_corr(&p, k, i, j);
                    assert_eq!(
                        got[at].to_bits(),
                        want.to_bits(),
                        "k = {k}, pair ({i}, {j})"
                    );
                    nan += usize::from(want.is_nan());
                    clamped += usize::from(want.abs() == 1.0);
                    inside += usize::from(want.abs() < 1.0 && want != 0.0);
                    at += 1;
                }
            }
            assert_eq!(at, got.len());
        }
    }
    assert!(nan > 1000 && clamped > 100 && inside > 100);
}

#[test]
fn the_lagged_correlations_are_the_element_at_a_time_form_to_the_bit() {
    let mut state = 37;
    let (mut nan, mut finite) = (0, 0);
    for k in 1..=13 {
        for n_lags in 1..=4 {
            for _ in 0..4 {
                let c: Vec<f64> = (0..n_lags * k * k).map(|_| awkward(&mut state)).collect();
                let std: Vec<f64> = (0..k).map(|_| awkward(&mut state)).collect();
                let mut got = vec![f64::MAX; n_lags * k * k];
                lag_corrs_into(&c, &std, k, &mut got);
                let mut at = 0;
                for li in 0..n_lags {
                    for a in 0..k {
                        for b in 0..k {
                            let d = std[a] * std[b];
                            let want = if d > 0.0 {
                                c[li * k * k + a * k + b] / d
                            } else {
                                f64::NAN
                            };
                            assert_eq!(
                                got[at].canon_bits(),
                                want.canon_bits(),
                                "k = {k}, lag {li}, ({a}, {b})"
                            );
                            nan += usize::from(want.is_nan());
                            finite += usize::from(want.is_finite());
                            at += 1;
                        }
                    }
                }
                assert_eq!(at, got.len());
            }
        }
    }
    assert!(nan > 500 && finite > 500);
}

#[test]
fn the_slots_of_a_row_are_the_stored_moments_read_the_long_way_to_the_bit() {
    // The whole path, a model's `predict` over a stream with a column of no
    // spread, a column of zeros, a large one and a tiny one: each slot of the
    // correlations, the partial correlations and the lagged correlations is
    // the stored accumulator read pair by pair, as every earlier build did.
    for k in [2usize, 3, 4, 5, 6, 7, 8, 9, 13, 20] {
        let lags = vec![1usize, 3];
        let stats = vec![
            EwCovStat::Mean,
            EwCovStat::Corr,
            EwCovStat::PartialCorr,
            EwCovStat::LagCorr,
            EwCovStat::Std,
        ];
        let mut m = EwCovModel::new(EwCovCfg {
            bias: false,
            n_features: k,
            decay: crate::Decay::Halflife(30.0),
            stats: stats.clone(),
            min_weight: 2.0,
            precision_prior: Some(1.0),
            mahal_quantiles: Vec::new(),
            pca: 0,
            pca_every: 0.0,
            max_rows_between_pca: u32::MAX,
            lags: lags.clone(),
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let mut state = 5 + k as u64;
        let mut x = vec![0.0; k];
        for _ in 0..120 {
            for (j, v) in x.iter_mut().enumerate() {
                *v = match j % 5 {
                    1 => 3.0,
                    2 => 0.0,
                    3 => 1e100 * lcg(&mut state),
                    4 => 1e-170 * lcg(&mut state),
                    _ => lcg(&mut state),
                };
            }
            m.step(&x, &[], 1.0, 1.0);
        }
        let got = m.predict(&x, 1.0).pred;
        let cov = m.cov();
        let lag = m.lag().expect("lags are kept");
        let std: Vec<f64> = (0..k).map(|i| cov.var(i).sqrt()).collect();
        let pairs = k * (k - 1) / 2;
        let precision = cov.precision();
        let mut want = Vec::new();
        want.extend((0..k).map(|i| cov.mean(i)));
        for i in 0..k {
            for j in (i + 1)..k {
                want.push(correlation_by_branch(cov.cov(j, i), std[i], std[j]));
            }
        }
        for i in 0..k {
            for j in (i + 1)..k {
                want.push(match &precision {
                    Some(p) => partial_corr(p, k, i, j),
                    None => f64::NAN,
                });
            }
        }
        for li in 0..lags.len() {
            for a in 0..k {
                for b in 0..k {
                    let d = std[a] * std[b];
                    want.push(if d > 0.0 {
                        lag.get(li, a, b) / d
                    } else {
                        f64::NAN
                    });
                }
            }
        }
        let unbiased = cov.unbiased_factor();
        want.extend((0..k).map(|i| (unbiased * cov.var(i)).sqrt()));
        assert_eq!(got.len(), want.len(), "k = {k}");
        assert_eq!(
            got.len(),
            k + 2 * pairs + lags.len() * k * k + k,
            "the slots are the stats' widths"
        );
        for (at, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.canon_bits(), w.canon_bits(), "k = {k}, slot {at}");
        }
        // The stream reached a NaN and a number in the correlations.
        let corr = &got[k..k + pairs];
        assert!(corr.iter().any(|c| c.is_nan()), "k = {k}");
        // Two ordinary columns, the first and the sixth, make a correlation.
        if k >= 6 {
            assert!(corr.iter().any(|c| c.is_finite()), "k = {k}");
        }
    }
}
