//! `audit`'s tests. Each statistic is held to its definition, written out
//! over the whole stream at once (two passes, sorted values): the
//! accumulators are one-pass, the oracle is not. scipy's moments and
//! statsmodels' Dickey-Fuller statistic hold the same numbers from Python
//! (`tests/test_second_opinion.py`).

use super::*;
use crate::{OnlineModel, SplitMix64, StateError};

fn cfg(k: usize) -> AuditCfg {
    AuditCfg {
        n_columns: k,
        pairs: false,
        distinct_cap: DISTINCT_CAP,
        gap_cap: None,
        has_clock: true,
    }
}

/// A standard normal draw, Box-Muller: libm, so nothing here is pinned to
/// the bit.
fn normal(r: &mut SplitMix64) -> f64 {
    let (u, v) = (r.uniform().max(1e-300), r.uniform());
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

fn feed(m: &mut Audit, rows: &[Vec<f64>]) {
    for (i, x) in rows.iter().enumerate() {
        m.step(x, &[], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300)
}

/// Each kind of value that is not usable is counted apart, and nothing of
/// it reaches the moments.
#[test]
fn every_kind_of_unusable_value_is_counted_apart() {
    let mut m = Audit::new(cfg(1)).unwrap();
    let vals = [
        1.0,
        NULL,
        f64::NAN,
        -f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        2e100,
        -1e101,
        f64::MAX,
        3.0,
        NULL,
        1e100,
    ];
    for v in vals {
        m.step(&[v], &[], 1.0, 1.0);
    }
    let r = m.column(0);
    assert_eq!(
        (r.null, r.nan, r.pos_inf, r.neg_inf, r.beyond, r.count),
        (2, 2, 1, 1, 3, 3)
    );
    assert_eq!(r.rows, vals.len() as u64);
    assert_eq!(
        (r.min, r.max),
        (1.0, 1e100),
        "1e100 is the bound, and usable"
    );
    assert!(is_null(NULL) && NULL.is_nan() && !is_null(f64::NAN));
}

/// The moments are their two-pass definitions, and the kurtosis and skew
/// scipy's biased estimators.
#[test]
fn the_moments_are_their_definitions() {
    let mut r = SplitMix64::new(7);
    let xs: Vec<f64> = (0..5000)
        .map(|_| 1e4 + normal(&mut r) * normal(&mut r).abs())
        .collect();
    let mut m = Audit::new(cfg(1)).unwrap();
    feed(&mut m, &xs.iter().map(|&v| vec![v]).collect::<Vec<_>>());
    let c = m.column(0);
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let mk = |p: i32| xs.iter().map(|x| (x - mean).powi(p)).sum::<f64>() / n;
    let (m2, m3, m4) = (mk(2), mk(3), mk(4));
    assert!(close(c.mean, mean, 1e-14), "{} {mean}", c.mean);
    assert!(close(c.std, (m2 * n / (n - 1.0)).sqrt(), 1e-10));
    assert!(
        close(c.skew, m3 / m2.powf(1.5), 1e-8),
        "{} {}",
        c.skew,
        m3 / m2.powf(1.5)
    );
    assert!(close(c.kurtosis, m4 / (m2 * m2) - 3.0, 1e-8));
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!((c.min, c.max), (lo, hi));
}

/// Every value within the input bound keeps the state finite, so it saves
/// and loads: the fourth power of a deviation of `1e100` is past a double,
/// and the moments are kept over a power of two for it. Kurtosis and skew
/// do not depend on the scale, so the oracle reads them off the values over
/// `1e90`; the mean and spread are the definitions' at full size.
#[test]
fn values_at_the_input_bound_keep_a_finite_state() {
    let mut r = SplitMix64::new(17);
    let mut xs: Vec<f64> = (0..400).map(|_| normal(&mut r)).collect();
    xs.extend([1e100, -1e100, 3e99, -7e98]);
    xs.extend((0..400).map(|_| 1e99 * normal(&mut r)));
    let mut m = Audit::new(AuditCfg {
        pairs: true,
        ..cfg(2)
    })
    .unwrap();
    for &v in &xs {
        m.step(&[v, -v], &[], 1.0, 1.0);
    }
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let back = Audit::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert_eq!(back, m);
    let c = m.column(0);
    let ys: Vec<f64> = xs.iter().map(|x| x / 1e90).collect();
    let n = ys.len() as f64;
    let mean = ys.iter().sum::<f64>() / n;
    let mk = |p: i32| ys.iter().map(|y| (y - mean).powi(p)).sum::<f64>() / n;
    assert!(
        close(c.kurtosis, mk(4) / (mk(2) * mk(2)) - 3.0, 1e-9),
        "{}",
        c.kurtosis
    );
    assert!(close(c.skew, mk(3) / mk(2).powf(1.5), 1e-9));
    assert!(close(c.mean, mean * 1e90, 1e-9));
    assert!(close(c.std, (mk(2) * n / (n - 1.0)).sqrt() * 1e90, 1e-9));
    assert!(c.autocorr.is_finite() && c.unit_root_t.is_finite());
    assert!(close(m.pair(0, 1).unwrap().corr, -1.0, 1e-12));
    let (mut a, mut b) = (m.clone(), m.clone());
    a.merge(&b).unwrap();
    assert!(close(a.column(0).kurtosis, c.kurtosis, 1e-9));
    b.merge(
        &Audit::new(AuditCfg {
            pairs: true,
            ..cfg(2)
        })
        .unwrap(),
    )
    .unwrap();
    assert_eq!(b.column(0).mean, c.mean);
}

/// Below `2^65` nothing is scaled: the moments are the unscaled arithmetic's
/// to the bit, and so are they past it, wherever that arithmetic stays
/// finite, a power of two dividing exactly.
#[test]
fn a_scale_moves_no_bit_where_the_arithmetic_is_finite() {
    let mut r = SplitMix64::new(19);
    let xs: Vec<f64> = (0..500).map(|_| 1e15 * normal(&mut r)).collect();
    let mut plain = Moments::default();
    let mut scaled = Moments::default();
    scaled.rescale(pow2(40));
    for &x in &xs {
        plain.add(x);
        scaled.add(x);
    }
    assert_eq!(plain.scale, 1.0, "1e15 is below 2^65");
    assert_eq!(plain.mean().to_bits(), scaled.mean().to_bits());
    assert_eq!(plain.std().to_bits(), scaled.std().to_bits());
    assert_eq!(plain.kurtosis().to_bits(), scaled.kurtosis().to_bits());
    assert_eq!(scale_for(2f64.powi(65)), 2.0);
    assert_eq!(scale_for(2f64.powi(65) * 0.999), 1.0);
}

/// A constant column has no spread: the standard deviation is 0 to the bit,
/// and what divides by it is undefined.
#[test]
fn a_constant_column_reads_no_spread() {
    let mut m = Audit::new(cfg(1)).unwrap();
    for _ in 0..1000 {
        m.step(&[0.1], &[], 1.0, 1.0);
    }
    let c = m.column(0);
    assert_eq!(c.std, 0.0);
    assert!(c.kurtosis.is_nan() && c.skew.is_nan() && c.autocorr.is_nan());
    assert_eq!((c.distinct, c.top_value, c.top_count), (Some(1), 0.1, 1000));
    assert_eq!((c.median, c.mad), (0.1, 0.0));
    assert!(c.robust_z.is_nan());
    assert_eq!((c.longest_run, c.equal_prev, c.adjacent), (1000, 999, 999));
}

/// Runs and the rows equal to the row before: a value that is not usable
/// ends a run, and `0` equals `-0`.
#[test]
fn runs_end_at_a_change_or_a_value_that_is_not_usable() {
    let mut m = Audit::new(cfg(1)).unwrap();
    for v in [1.0, 1.0, 1.0, 2.0, NULL, 2.0, 2.0, 0.0, -0.0, 0.0, 0.0, 5.0] {
        m.step(&[v], &[], 1.0, 1.0);
    }
    let c = m.column(0);
    assert_eq!(c.longest_run, 4, "0, -0, 0, 0");
    // 1=1, 1=1, 2=2, 0=-0, -0=0, 0=0; the null breaks 2 | 2.
    assert_eq!(c.equal_prev, 6);
    // Twelve rows, the first and the null and the row after it have no
    // usable row before.
    assert_eq!(c.adjacent, 9);
    assert_eq!(c.distinct, Some(4), "1, 2, 0 and 5: -0 is 0");
}

/// The distinct count and every count are exact up to the cap; past it each
/// count is within the decrements of its truth, and the decrements within
/// `n / (cap + 1)` (Misra & Gries' bound).
#[test]
fn the_counts_are_exact_to_the_cap_and_bounded_past_it() {
    let cap = 16;
    let mut m = Audit::new(AuditCfg {
        distinct_cap: cap,
        ..cfg(1)
    })
    .unwrap();
    let mut r = SplitMix64::new(3);
    let mut truth = std::collections::BTreeMap::<u64, u64>::new();
    let n = 20_000u64;
    for i in 0..n {
        // 5% on -999, the rest over 400 values.
        let v = if r.uniform() < 0.05 {
            -999.0
        } else {
            (r.next_u64() % 400) as f64
        };
        *truth.entry(key(v)).or_default() += 1;
        m.step(&[v], &[], 1.0, 1.0);
        if i == 10 {
            let c = m.column(0);
            assert!(c.distinct.is_some_and(|d| d <= 11) && c.count_error == 0);
        }
    }
    let c = m.column(0);
    assert_eq!(c.distinct, None, "past the cap");
    let d = c.count_error;
    assert!(d > 0 && d <= n / (cap as u64 + 1), "{d}");
    assert_eq!(c.top_value, -999.0);
    let real = truth[&key(-999.0)];
    assert!(c.top_count <= real && real - c.top_count <= d);
    for &(k, est) in &m.columns[0].counts.counts {
        let t = truth[&k];
        assert!(est <= t && t - est <= d, "{k}: {est} of {t}, error {d}");
    }
    let second = truth
        .iter()
        .filter(|(k, _)| **k != key(-999.0))
        .map(|e| *e.1)
        .max();
    assert!(c.second_count <= second.unwrap());
}

/// The median and MAD are exact while the counts are, numpy's even-count
/// median among them.
#[test]
fn the_median_and_mad_are_exact_from_exact_counts() {
    let mut m = Audit::new(cfg(1)).unwrap();
    let xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    for v in xs {
        m.step(&[v], &[], 1.0, 1.0);
    }
    let c = m.column(0);
    // Sorted: 1 1 2 3 4 5 6 9 -> median 3.5; |x - 3.5| sorted: 0.5 0.5 1.5
    // 1.5 2.5 2.5 2.5 5.5 -> MAD 2.0.
    assert_eq!((c.median, c.mad), (3.5, 2.0));
    assert!(close(c.robust_z, 5.5 / (MAD_SCALE * 2.0), 1e-15));
}

/// Past the cap the digest reads the median and the MAD to within a rank
/// resolution of about one percent: the values at ranks 1% either side
/// bracket each, at a level where a relative-error sketch could not.
#[test]
fn the_digest_reads_the_median_and_mad_to_a_rank() {
    for (seed, level) in [(1u64, 0.0), (2, 1e4), (3, -1e8)] {
        let mut r = SplitMix64::new(seed);
        let xs: Vec<f64> = (0..50_000).map(|_| level + normal(&mut r)).collect();
        let mut m = Audit::new(cfg(1)).unwrap();
        for &v in &xs {
            m.step(&[v], &[], 1.0, 1.0);
        }
        let c = m.column(0);
        let mut s = xs.clone();
        s.sort_by(f64::total_cmp);
        let q = |p: f64| s[((s.len() - 1) as f64 * p) as usize];
        assert!(
            q(0.49) <= c.median && c.median <= q(0.51),
            "level {level}: median {}",
            c.median
        );
        let mut dev: Vec<f64> = xs.iter().map(|x| (x - q(0.5)).abs()).collect();
        dev.sort_by(f64::total_cmp);
        let dq = |p: f64| dev[((dev.len() - 1) as f64 * p) as usize];
        assert!(
            dq(0.48) <= c.mad && c.mad <= dq(0.52),
            "level {level}: MAD {} not in [{}, {}]",
            c.mad,
            dq(0.48),
            dq(0.52)
        );
        assert!(m.columns[0].digest.centroids.len() <= COMPRESSION as usize + 1);
    }
}

/// The lag-1 autocorrelation is the correlation of `x[..n-1]` with
/// `x[1..]`, and the Dickey-Fuller statistic the t of their regression's
/// slope against 1, both written out.
#[test]
fn the_persistence_is_the_regression_of_a_row_on_the_row_before() {
    let mut r = SplitMix64::new(11);
    let mut x = 0.0;
    let xs: Vec<f64> = (0..3000)
        .map(|_| {
            x = 0.9 * x + normal(&mut r);
            x
        })
        .collect();
    let mut m = Audit::new(cfg(1)).unwrap();
    for &v in &xs {
        m.step(&[v], &[], 1.0, 1.0);
    }
    let c = m.column(0);
    let (a, b) = (&xs[..xs.len() - 1], &xs[1..]);
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let saa: f64 = a.iter().map(|v| (v - ma).powi(2)).sum();
    let sbb: f64 = b.iter().map(|v| (v - mb).powi(2)).sum();
    let sab: f64 = a.iter().zip(b).map(|(u, v)| (u - ma) * (v - mb)).sum();
    assert!(close(c.autocorr, sab / (saa * sbb).sqrt(), 1e-12));
    let rho = sab / saa;
    let resid: f64 = a
        .iter()
        .zip(b)
        .map(|(u, v)| (v - mb - rho * (u - ma)).powi(2))
        .sum();
    let tau = (rho - 1.0) / (resid / (n - 2.0) / saa).sqrt();
    assert!(close(c.unit_root_t, tau, 1e-9), "{} {tau}", c.unit_root_t);
    assert!(c.unit_root_t < -5.0, "an AR(0.9) rejects a unit root");
}

/// Pairs are over the rows where both columns are usable.
#[test]
fn pairs_are_over_the_rows_both_columns_hold() {
    let mut m = Audit::new(AuditCfg {
        pairs: true,
        ..cfg(3)
    })
    .unwrap();
    let rows = [
        [1.0, 1.0, 5.0],
        [2.0, 2.0, NULL],
        [3.0, 3.0, 1.0],
        [4.0, f64::NAN, 2.0],
        [5.0, 5.0, 0.0],
    ];
    for x in rows {
        m.step(&x, &[], 1.0, 1.0);
    }
    let p01 = m.pair(0, 1).unwrap();
    assert_eq!((p01.count, p01.equal), (4, 4));
    assert!(close(p01.corr, 1.0, 1e-15));
    let p02 = m.pair(0, 2).unwrap();
    assert_eq!(p02.count, 4);
    // x = 1 3 4 5, z = 5 1 2 0.
    let (x, z) = ([1.0, 3.0, 4.0, 5.0], [5.0, 1.0, 2.0, 0.0]);
    let (mx, mz) = (13.0 / 4.0, 2.0);
    let sxz: f64 = x.iter().zip(&z).map(|(a, b)| (a - mx) * (b - mz)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let szz: f64 = z.iter().map(|b| (b - mz) * (b - mz)).sum();
    assert!(close(p02.corr, sxz / (sxx * szz).sqrt(), 1e-14));
    assert_eq!(m.pair(1, 2).unwrap().count, 3);
    assert!(m.pair(1, 0).is_none() && m.pair(0, 3).is_none());
    assert!(
        Audit::new(cfg(3)).unwrap().pair(0, 1).is_none(),
        "pairs off"
    );
}

/// The clock: the first row and a restart's have no step; a step of 0 is a
/// duplicate, one at `gap_cap` a gap, the rest regular.
#[test]
fn the_clock_counts_duplicates_gaps_and_regular_steps() {
    let mut m = Audit::new(AuditCfg {
        gap_cap: Some(10.0),
        ..cfg(1)
    })
    .unwrap();
    for d in [0.0, 1.0, 0.0, 2.0, 10.0, 3.0] {
        m.step(&[1.0], &[], d, 1.0);
    }
    m.restart();
    m.step(&[1.0], &[], 0.0, 1.0);
    m.step(&[1.0], &[], 2.0, 1.0);
    let c = m.clock().unwrap();
    assert_eq!(
        (c.steps, c.duplicates, c.gaps, c.regular),
        (6, 1, Some(1), 4)
    );
    // Regular steps 1, 2, 3, 2: mean 2, population sd sqrt(0.5).
    assert!(close(c.step_mean, 2.0, 1e-15) && close(c.step_std, 0.5f64.sqrt(), 1e-15));
    assert!(close(c.step_cv, 0.5f64.sqrt() / 2.0, 1e-15));
    assert_eq!(c.max_step, 10.0);
    assert_eq!(m.column(0).longest_run, 6, "a restart ends the run");
    let none = Audit::new(AuditCfg {
        has_clock: false,
        ..cfg(1)
    })
    .unwrap();
    assert!(none.clock().is_none());
    let mut uncapped = Audit::new(cfg(1)).unwrap();
    uncapped.step(&[1.0], &[], 0.0, 1.0);
    uncapped.step(&[1.0], &[], 1e9, 1.0);
    assert_eq!(uncapped.clock().unwrap().gaps, None);
}

/// Hard rule 9: a row of weight 0, the first included, adds nothing to
/// `n_eff` and divides nothing by zero; its values are counted, since a
/// value is a value whatever its row weighs.
#[test]
fn a_zero_weight_row_is_counted_and_adds_no_weight() {
    let mut m = Audit::new(AuditCfg {
        pairs: true,
        ..cfg(2)
    })
    .unwrap();
    let s = m.step(&[1.0, 2.0], &[], 0.0, 0.0);
    assert_eq!(s.n_eff, 0.0);
    m.step(&[3.0, 2.0], &[], 1.0, 0.0);
    assert_eq!(m.n_eff(), 0.0);
    m.step(&[5.0, 1.0], &[], 1.0, 2.5);
    assert_eq!(m.predict(&[0.0, 0.0], 1.0).n_eff, 2.5);
    let c = m.column(0);
    assert_eq!((c.count, c.mean), (3, 3.0));
    assert!(m.pair(0, 1).unwrap().corr.is_finite());
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let back: State = rmp_serde::from_slice(&bytes).unwrap();
    assert_eq!(Audit::restore(&back).unwrap(), m);
}

/// Merging the audits of two streams gives the audit of both: the counts
/// exactly, the moments to rounding, the counters exactly under the cap.
#[test]
fn a_merge_is_the_audit_of_both_streams() {
    let mut r = SplitMix64::new(5);
    let make = |r: &mut SplitMix64, n: usize, shift: f64| -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let a = if i % 37 == 0 { NULL } else { shift + normal(r) };
                let b = (r.next_u64() % 9) as f64;
                vec![
                    a,
                    b,
                    if i % 50 == 0 {
                        f64::INFINITY
                    } else {
                        a * 2.0 + b
                    },
                ]
            })
            .collect()
    };
    let (s1, s2) = (make(&mut r, 3000, 0.0), make(&mut r, 2000, 5.0));
    let c = AuditCfg {
        pairs: true,
        gap_cap: Some(5.0),
        ..cfg(3)
    };
    let (mut a, mut b, mut both) = (
        Audit::new(c.clone()).unwrap(),
        Audit::new(c.clone()).unwrap(),
        Audit::new(c).unwrap(),
    );
    feed(&mut a, &s1);
    feed(&mut b, &s2);
    feed(&mut both, &s1);
    both.restart();
    feed(&mut both, &s2);
    a.merge(&b).unwrap();
    for j in 0..3 {
        let (x, y) = (a.column(j), both.column(j));
        assert_eq!(
            (
                x.rows,
                x.null,
                x.pos_inf,
                x.count,
                x.min,
                x.max,
                x.longest_run
            ),
            (
                y.rows,
                y.null,
                y.pos_inf,
                y.count,
                y.min,
                y.max,
                y.longest_run
            )
        );
        assert_eq!((x.equal_prev, x.adjacent), (y.equal_prev, y.adjacent));
        for (u, v) in [
            (x.mean, y.mean),
            (x.std, y.std),
            (x.skew, y.skew),
            (x.kurtosis, y.kurtosis),
            (x.autocorr, y.autocorr),
        ] {
            assert!(close(u, v, 1e-10), "column {j}: {u} vs {v}");
        }
    }
    // Column 1 has nine values: exact counts merge exactly.
    assert_eq!(a.columns[1].counts, both.columns[1].counts);
    assert_eq!(a.column(1).median, both.column(1).median);
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let (p, q) = (a.pair(i, j).unwrap(), b.pair(i, j).unwrap());
        let w = both.pair(i, j).unwrap();
        assert_eq!(p.count, w.count);
        assert_eq!(p.equal, w.equal);
        assert!(close(p.corr, w.corr, 1e-10) && q.count < w.count);
    }
    let (x, y) = (a.clock().unwrap(), both.clock().unwrap());
    assert_eq!(
        (x.steps, x.duplicates, x.gaps),
        (y.steps, y.duplicates, y.gaps)
    );
    assert_eq!(a.n_eff(), both.n_eff());
    let other = Audit::new(cfg(2)).unwrap();
    assert!(a.merge(&other).is_err());
}

/// Past the cap the merge keeps Misra-Gries' bound: every count within the
/// summed decrements of its truth.
#[test]
fn a_merge_past_the_cap_keeps_the_bound() {
    let c = AuditCfg {
        distinct_cap: 8,
        ..cfg(1)
    };
    let mut r = SplitMix64::new(9);
    let mut truth = std::collections::BTreeMap::<u64, u64>::new();
    let mut parts = Vec::new();
    for _ in 0..3 {
        let mut m = Audit::new(c.clone()).unwrap();
        for _ in 0..4000 {
            let v = if r.uniform() < 0.2 {
                7.0
            } else {
                (r.next_u64() % 100) as f64
            };
            *truth.entry(key(v)).or_default() += 1;
            m.step(&[v], &[], 1.0, 1.0);
        }
        parts.push(m);
    }
    let mut all = parts[0].clone();
    all.merge(&parts[1]).unwrap();
    all.merge(&parts[2]).unwrap();
    let d = all.column(0).count_error;
    assert!(d <= 12_000 / 9, "{d}");
    for &(k, est) in &all.columns[0].counts.counts {
        let t = truth[&k];
        assert!(est <= t && t - est <= d, "{est} of {t}, error {d}");
    }
    assert_eq!(all.column(0).top_value, 7.0);
}

/// `restore` refuses a state whose vectors are not the cfg's, and one whose
/// cfg `new` refuses.
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let m = Audit::new(AuditCfg {
        pairs: true,
        ..cfg(3)
    })
    .unwrap();
    let edits: [fn(&mut Audit); 6] = [
        |a| {
            a.columns.pop();
        },
        |a| {
            a.pairs.pop();
        },
        |a| a.columns[0].counts.counts = vec![(5, 1), (3, 1)],
        |a| a.columns[1].digest.buffer = vec![0.0; DIGEST_BUFFER],
        |a| a.columns[2].prev = Some(f64::INFINITY),
        |a| a.n_eff = f64::NAN,
    ];
    for edit in edits {
        let mut bad = m.clone();
        edit(&mut bad);
        match Audit::restore(&State::new(ModelState::Audit(Box::new(bad)))) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
    let mut zero = m.clone();
    zero.cfg.distinct_cap = 0;
    match Audit::restore(&State::new(ModelState::Audit(Box::new(zero)))) {
        Err(StateError::Invalid(e)) => assert!(e.contains("configuration"), "{e}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn new_refuses_what_it_cannot_run() {
    for bad in [
        AuditCfg {
            n_columns: 0,
            ..cfg(1)
        },
        AuditCfg {
            distinct_cap: 0,
            ..cfg(1)
        },
        AuditCfg {
            distinct_cap: MAX_DISTINCT_CAP + 1,
            ..cfg(1)
        },
        AuditCfg {
            gap_cap: Some(0.0),
            ..cfg(1)
        },
        AuditCfg {
            gap_cap: Some(f64::INFINITY),
            ..cfg(1)
        },
    ] {
        assert!(Audit::new(bad).is_err());
    }
}

/// A saved audit goes on as the one that saved it, from any row: the
/// digest's buffer, the counters and a run in progress travel with it.
#[test]
fn a_restored_audit_goes_on_where_it_stopped() {
    let mut r = SplitMix64::new(13);
    let rows: Vec<Vec<f64>> = (0..2000)
        .map(|i| {
            vec![
                normal(&mut r),
                (i / 30) as f64,
                if i % 7 == 0 { NULL } else { 1.0 },
            ]
        })
        .collect();
    let c = AuditCfg {
        pairs: true,
        distinct_cap: 32,
        ..cfg(3)
    };
    let mut whole = Audit::new(c.clone()).unwrap();
    feed(&mut whole, &rows);
    for cut in [1, 255, 256, 257, 1000] {
        let mut a = Audit::new(c.clone()).unwrap();
        feed(&mut a, &rows[..cut]);
        let bytes = rmp_serde::to_vec(&a.state()).unwrap();
        let mut b = Audit::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for x in &rows[cut..] {
            b.step(x, &[], 1.0, 1.0);
        }
        assert_eq!(b, whole, "cut at {cut}");
    }
}
