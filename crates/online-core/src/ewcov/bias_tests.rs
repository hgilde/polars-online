//! `EwCovCfg::bias` (docs/PLAN.md task 244): the unbiased moments against
//! their definition, written from the rows, and a single row of weight read
//! as undefined.

use super::*;
use crate::OnlineModel;

/// A row: its features, its clock and its weight.
type Row = (Vec<f64>, f64, f64);

/// The rows the definition keeps at `now`: positive weight, and inside
/// `window` (age below it, `closed = "right"`), each with its weight
/// `w · 0.5^(age / half_life)`.
fn kept(rows: &[Row], now: f64, half_life: f64, window: Option<f64>) -> Vec<(&[f64], f64)> {
    rows.iter()
        .filter(|(_, t, w)| *w > 0.0 && window.is_none_or(|win| now - *t < win))
        .map(|(x, t, w)| (x.as_slice(), w * 0.5_f64.powf((now - t) / half_life)))
        .collect()
}

/// The definition at `now`, two-pass: `(V₁, V₂, mean, S)`, `S` the weighted
/// mean of the deviations' outer products, row-major.
fn definition(
    rows: &[Row],
    now: f64,
    half_life: f64,
    window: Option<f64>,
) -> (f64, f64, Vec<f64>, Vec<f64>) {
    let k = rows[0].0.len();
    let kept = kept(rows, now, half_life, window);
    let v1: f64 = kept.iter().map(|(_, u)| u).sum();
    let v2: f64 = kept.iter().map(|(_, u)| u * u).sum();
    let mean: Vec<f64> = (0..k)
        .map(|a| kept.iter().map(|(x, u)| u * x[a]).sum::<f64>() / v1)
        .collect();
    let mut s = vec![0.0; k * k];
    for (x, u) in &kept {
        for a in 0..k {
            for b in 0..k {
                s[a * k + b] += u * (x[a] - mean[a]) * (x[b] - mean[b]) / v1;
            }
        }
    }
    (v1, v2, mean, s)
}

/// The largest eigenvalue of a symmetric positive semi-definite `3 x 3`
/// matrix by power iteration from `(1, 1, 1)`, which shares nothing with the
/// eigensolver `Pca::of` runs (`crate::oracle`'s is the same one).
fn top_eigenvalue(s: &[f64]) -> f64 {
    let mut v = [1.0, 1.0, 1.0];
    let mut lambda = 0.0;
    for _ in 0..2000 {
        let next: Vec<f64> = (0..3)
            .map(|a| (0..3).map(|b| s[a * 3 + b] * v[b]).sum())
            .collect();
        let norm = next.iter().map(|y| y * y).sum::<f64>().sqrt();
        lambda = norm / v.iter().map(|y| y * y).sum::<f64>().sqrt();
        for a in 0..3 {
            v[a] = next[a] / norm;
        }
    }
    lambda
}

fn cfg(half_life: f64, window: Option<f64>, bias: bool) -> EwCovCfg {
    EwCovCfg {
        n_features: 3,
        decay: crate::Decay::Halflife(half_life),
        stats: vec![
            EwCovStat::Var,
            EwCovStat::Std,
            EwCovStat::Cov,
            EwCovStat::Corr,
            EwCovStat::PartialCorr,
            EwCovStat::Mahal,
        ],
        min_weight: 0.0,
        bias,
        precision_prior: Some(0.05),
        mahal_quantiles: Vec::new(),
        pca: 2,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// The slots of `cfg`'s statistics: three variances, three standard
/// deviations, three covariances, three correlations, three partial
/// correlations, the distance, then per component its variance, share,
/// three loadings and score.
const VAR: usize = 0;
const STD: usize = 3;
const COV: usize = 6;
const CORR: usize = 9;
const MAHAL: usize = 15;
const PC: usize = 16;

/// A uniform draw in `[0, 1)` from a seed.
fn uniform(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

/// An irregular stream: steps of 0.5 to 3 clock units, weights over two
/// decades, a fifth of them 0, the first two rows among those, a feature at
/// a level of 100.
fn stream(n: usize, seed: u64) -> Vec<Row> {
    let mut s = seed;
    let mut t = 0.0;
    (0..n)
        .map(|i| {
            if i > 0 {
                t += 0.5 + 2.5 * uniform(&mut s);
            }
            let w = if i < 2 || uniform(&mut s) < 0.2 {
                0.0
            } else {
                10f64.powf(2.0 * uniform(&mut s) - 1.0)
            };
            let a = 3.0 * uniform(&mut s) - 1.0;
            let x = vec![
                a + 100.0,
                0.5 * a + uniform(&mut s),
                uniform(&mut s) - 0.3 * a,
            ];
            (x, t, w)
        })
        .collect()
}

/// The relative errors of a read against the definition scaled by `c`: the
/// variances, the standard deviations, the covariances (of the root of the
/// two variances), the distance in the metric `c·(S + ridge·I)` (its square
/// by the test oracle's LU) and the first component's variance, `c` times
/// the top eigenvalue of `S` by power iteration.
fn errors(pred: &[f64], c: f64, x: &[f64], mean: &[f64], s: &[f64], ridge: f64) -> [f64; 5] {
    let mut e = [0.0f64; 5];
    for a in 0..3 {
        let var = c * s[a * 3 + a];
        e[0] = e[0].max((pred[VAR + a] - var).abs() / var);
        e[1] = e[1].max((pred[STD + a] - var.sqrt()).abs() / var.sqrt());
    }
    for (slot, (a, b)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
        let scale = c * (s[a * 3 + a] * s[b * 3 + b]).sqrt();
        e[2] = e[2].max((pred[COV + slot] - c * s[a * 3 + b]).abs() / scale);
    }
    let metric: Vec<f64> = (0..9)
        .map(|ab| c * (s[ab] + if ab / 3 == ab % 3 { ridge } else { 0.0 }))
        .collect();
    let delta: Vec<f64> = (0..3).map(|a| x[a] - mean[a]).collect();
    let want = crate::oracle::quad_form(&metric, &delta).sqrt();
    e[3] = (pred[MAHAL] - want).abs() / want;
    let top = c * top_eigenvalue(s);
    e[4] = (pred[PC] - top).abs() / top;
    e
}

/// Every unbiased moment is the definition's, `c = V₁² / (V₁² − V₂)` times
/// the weighted mean of the deviations' products, with and without a
/// window, over irregular steps and rows of weight 0 (hard rule 9, the first
/// two rows among them); `mahal` is in the metric of `c·(S + s·prior·I)`
/// ([`errors`]). Without a window, to 1e-12. Under one the moments are a
/// subtraction whose rounding is the history's, not the few rows a short
/// window holds (`crate::window`'s module docs), so the biased twin is held
/// to its own definition beside it and the factor may add no more than a
/// few times the twin's error. Every correlation, the components' shares
/// within a rounding, their loadings and scores and the weight are the
/// twin's, to the bit.
#[test]
fn the_unbiased_moments_are_the_definition() {
    let h = 4.0;
    let rows = stream(220, 2026);
    for window in [None, Some(9.0)] {
        let mut unbiased = EwCovModel::new(cfg(h, window, false)).unwrap();
        let mut biased = EwCovModel::new(cfg(h, window, true)).unwrap();
        let (mut worst, mut twin_worst, mut checked) = ([0.0f64; 5], [0.0f64; 5], 0);
        for i in 0..rows.len() {
            let (x, t, w) = &rows[i];
            let d = if i == 0 { 0.0 } else { t - rows[i - 1].1 };
            let ridge = 0.05 * unbiased.view().precision_scale();
            let got = unbiased.step(x, &[], d, *w);
            let twin = biased.step(x, &[], d, *w);
            assert_eq!(got.n_eff.to_bits(), twin.n_eff.to_bits(), "row {i}");
            for slot in CORR..MAHAL {
                assert_eq!(
                    got.pred[slot].to_bits(),
                    twin.pred[slot].to_bits(),
                    "row {i}"
                );
            }
            for j in 0..2 {
                let base = PC + j * 6;
                for slot in (base + 2)..(base + 6) {
                    assert_eq!(
                        got.pred[slot].to_bits(),
                        twin.pred[slot].to_bits(),
                        "row {i}"
                    );
                }
                let (a, b) = (got.pred[base + 1], twin.pred[base + 1]);
                assert!(
                    (a - b).abs() <= 4.0 * f64::EPSILON * b.abs() || (a.is_nan() && b.is_nan()),
                    "row {i} pc{j} share {a} against {b}"
                );
            }
            if i == 0 {
                continue;
            }
            let now = rows[i - 1].1;
            let held = kept(&rows[..i], now, h, window).len();
            if held < 2 {
                for slot in [VAR, STD, COV, MAHAL, PC] {
                    assert!(got.pred[slot].is_nan(), "row {i} slot {slot}");
                }
                continue;
            }
            // Fewer than four rows leave three columns' matrix singular, and
            // `mahal` then reads a window's rounding against the ridge alone.
            if window.is_some() && held < 4 {
                continue;
            }
            let (v1, v2, mean, s) = definition(&rows[..i], now, h, window);
            let c = v1 * v1 / (v1 * v1 - v2);
            let (e, te) = (
                errors(&got.pred, c, x, &mean, &s, ridge),
                errors(&twin.pred, 1.0, x, &mean, &s, ridge),
            );
            for q in 0..5 {
                worst[q] = worst[q].max(e[q]);
                twin_worst[q] = twin_worst[q].max(te[q]);
            }
            checked += 1;
        }
        assert!(checked > 120, "{checked}");
        println!("window {window:?}: {worst:?}, the twin {twin_worst:?}, over {checked} reads");
        for q in 0..5 {
            let bound = match window {
                None => 1e-12,
                Some(_) => (4.0 * twin_worst[q]).max(1e-12),
            };
            assert!(
                worst[q] < bound,
                "{window:?} error {q}: {worst:?} {twin_worst:?}"
            );
        }
    }
}

/// `W² / (W² − Q)` is `n / (n − 1)` for `n` unit rows and no decay, to the
/// bit, and undefined before a row, on the first, and without `Q`.
#[test]
fn the_factor_is_bessels_for_unit_rows_and_undefined_for_one() {
    let mut c = EwCov::new(1);
    assert!(c.unbiased_factor().is_nan());
    for n in 1..60u32 {
        c.update(&[f64::from(n).sin()], 1.0, 1.0);
        let n = f64::from(n);
        if n == 1.0 {
            assert!(c.unbiased_factor().is_nan());
        } else {
            assert_eq!(
                c.unbiased_factor().to_bits(),
                (n / (n - 1.0)).to_bits(),
                "{n}"
            );
        }
    }
    c.set_moments(&[0.0], &[1.0], 3.0, None);
    assert!(c.unbiased_factor().is_nan());
    // A sum of squares of 0 beside a weight is one no digit was kept of
    // (`crate::truncated`'s floor), not a sum of none: no factor, where
    // `W² / (W² − 0)` would read the weighted mean as unbiased.
    c.set_moments(&[0.0], &[1.0], 3.0, Some(0.0));
    assert!(c.unbiased_factor().is_nan());
}

/// A window of light rows after heavy ones: where the subtraction keeps
/// digits of the window's squared weights, the eleven equal rows inside it
/// read `c = 11 / 10`, the biased twin's variance times Bessel's factor;
/// where it keeps none (rows of `1e-7` after rows of 1), the factor is
/// undefined and the variance null, as Kish's size is `None` there
/// ([`EwCov::n_kish`]), where the twin still reads the window's weighted
/// mean.
#[test]
fn a_window_whose_squared_weights_kept_no_digit_reads_undefined() {
    for (light, digits) in [(1e-3, true), (1e-7, false)] {
        let model = |bias: bool| {
            let mut c = cfg(f64::INFINITY, Some(10.0), bias);
            c.stats = vec![EwCovStat::Var];
            c.precision_prior = None;
            c.pca = 0;
            let mut m = EwCovModel::new(c).unwrap();
            m.set_window_closed(crate::WindowClosed::Both);
            m
        };
        let (mut unbiased, mut biased) = (model(false), model(true));
        let mut s = 5u64;
        let (mut got, mut twin) = (Vec::new(), Vec::new());
        for i in 0..1013 {
            let x = [uniform(&mut s), uniform(&mut s), uniform(&mut s)];
            let w = if i < 1000 { 1.0 } else { light };
            let d = if i == 0 { 0.0 } else { 1.0 };
            got = unbiased.step(&x, &[], d, w).pred;
            twin = biased.step(&x, &[], d, w).pred;
        }
        assert!(twin[0].is_finite() && twin[0] > 0.0, "{light}: {twin:?}");
        if digits {
            let want = 11.0 / 10.0 * twin[0];
            assert!(
                (got[0] - want).abs() <= 1e-3 * want,
                "{light}: {got:?} {want}"
            );
        } else {
            assert!(got[0].is_nan(), "{light}: {got:?}");
        }
    }
}

/// One row of weight has no spread to correct (hard rule 9): read before
/// any, on the first, and aged by rows of weight 0 after it, every second
/// moment is NaN however the rounding of `W²` against `Q` falls, where the
/// biased twin reads 0; the correlations are NaN in both. Rows of weight 0
/// before and after it, and a second row of weight, leave the moments the
/// definition's: nothing undefined reached the state. The factor alone
/// would have read some of these rows as a variance of 0, `W² − Q` landing
/// a hair above zero: the count of such reads is printed evidence that the
/// rows-of-weight rule is the one deciding.
#[test]
fn one_row_of_weight_reads_undefined_however_the_rounding_falls() {
    let mut above_zero = 0;
    for (h, heavy) in [
        (3.0, 5.0),
        (0.7, 2.5),
        (11.0, 7.25),
        (1.3, 3.0),
        (29.0, 2.0),
    ] {
        for window in [None, Some(40.0)] {
            let mut unbiased = EwCovModel::new(cfg(h, window, false)).unwrap();
            let mut biased = EwCovModel::new(cfg(h, window, true)).unwrap();
            let mut s = 99u64;
            let mut rows: Vec<Row> = Vec::new();
            let mut t = 0.0;
            for i in 0..30 {
                let d = if i == 0 {
                    0.0
                } else {
                    0.25 + 1.5 * uniform(&mut s)
                };
                t += d;
                let w = match i {
                    3 => heavy,
                    i if i >= 20 => 1.0,
                    _ => 0.0,
                };
                let x = vec![
                    uniform(&mut s),
                    uniform(&mut s) + 4.0,
                    uniform(&mut s) - 1.0,
                ];
                let one = unbiased.cov.rows_learned() == 1;
                if one && unbiased.cov.unbiased_factor().is_finite() {
                    above_zero += 1;
                }
                let got = unbiased.step(&x, &[], d, w);
                let twin = biased.step(&x, &[], d, w);
                if (4..=20).contains(&i) {
                    assert!(one, "row {i}");
                    for slot in [VAR, STD, COV, MAHAL] {
                        assert!(got.pred[slot].is_nan(), "h {h} row {i} slot {slot}");
                    }
                    assert_eq!(twin.pred[VAR], 0.0, "h {h} row {i}");
                    assert!(got.pred[CORR].is_nan() && twin.pred[CORR].is_nan());
                }
                if i >= 22 {
                    let now = rows[i - 1].1;
                    let (v1, v2, _, sm) = definition(&rows, now, h, window);
                    let var = v1 * v1 / (v1 * v1 - v2) * sm[0];
                    assert!(
                        (got.pred[VAR] - var).abs() <= 1e-12 * var,
                        "h {h} row {i}: {} against {var}",
                        got.pred[VAR]
                    );
                }
                rows.push((x, t, w));
            }
        }
    }
    println!("reads of one row of weight whose W² − Q rounded above zero: {above_zero}");
    assert!(above_zero > 0, "no read exercised the rule");
}

/// A component refreshed while one row of weight is read has a variance of
/// NaN, and the next refresh, with two, a number; the loadings are the
/// biased twin's throughout.
#[test]
fn a_component_refreshed_on_one_row_of_weight_recovers_at_the_next() {
    let mut unbiased = EwCovModel::new(cfg(5.0, None, false)).unwrap();
    let mut biased = EwCovModel::new(cfg(5.0, None, true)).unwrap();
    let mut s = 7u64;
    for (i, w) in [0.0, 2.0, 0.0, 1.0, 1.0].into_iter().enumerate() {
        let x = [uniform(&mut s), uniform(&mut s), uniform(&mut s)];
        let d = if i == 0 { 0.0 } else { 1.0 };
        unbiased.step(&x, &[], d, w);
        biased.step(&x, &[], d, w);
        let (u, b) = (unbiased.pca().unwrap(), biased.pca().unwrap());
        assert_eq!(u.loadings, b.loadings, "row {i}");
        // Refreshed after the row: one row of weight after rows 1 and 2.
        if (1..=2).contains(&i) {
            assert!(u.eig[0].is_nan() && b.eig[0] == 0.0, "row {i}: {u:?}");
        } else if i >= 3 {
            let c = unbiased.cov.unbiased_factor();
            assert_eq!(u.eig[0].to_bits(), (c * b.eig[0]).to_bits(), "row {i}");
        }
    }
}

/// Under a window, the rows of weight counted are the window's: rows two
/// clock units apart under a window of 1.5 leave one row inside at every
/// read, undefined, though the accumulator has learned them all.
#[test]
fn a_window_holding_one_row_of_weight_reads_undefined() {
    let mut unbiased = EwCovModel::new(cfg(4.0, Some(1.5), false)).unwrap();
    let mut biased = EwCovModel::new(cfg(4.0, Some(1.5), true)).unwrap();
    let mut s = 21u64;
    for i in 0..40 {
        let x = [uniform(&mut s), uniform(&mut s), uniform(&mut s)];
        let d = if i == 0 { 0.0 } else { 2.0 };
        let got = unbiased.step(&x, &[], d, 1.0);
        let twin = biased.step(&x, &[], d, 1.0);
        if i >= 2 {
            assert!((got.n_eff - 1.0).abs() < 1e-12, "row {i}: {}", got.n_eff);
            assert!(
                got.pred[VAR].is_nan() && got.pred[MAHAL].is_nan(),
                "row {i}"
            );
            assert_eq!(twin.pred[VAR], 0.0, "row {i}");
        }
    }
    assert!(unbiased.cov.rows_learned() == 40);
}
