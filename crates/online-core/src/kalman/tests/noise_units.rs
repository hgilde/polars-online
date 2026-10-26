//! `P` in units of the noise (`noise_units`, docs/PLAN.md task 278): the
//! gain reads no noise estimate, and the intercept's prior is diffuse. Each
//! test shows the form it holds activating against the form before it, `P`
//! absolute with the estimate in the gain, on the same rows. A file of its
//! own because `kalman.rs` stands near the source-size cap
//! (`tests/test_repo_hygiene.py`).

use super::*;

/// Two features correlated at 0.5, a known truth and noise 0.5; the first
/// `burst` rows have `+50` on a fifth of them where `x0 > 0`, the burst of
/// large residuals task 278 (a) found capturing the filter (D-K).
fn rows(n: usize, burst: usize, seed: u64) -> Vec<([f64; 2], f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let x0 = lcg(&mut s);
            let x1 = 0.5 * x0 + 0.75f64.sqrt() * lcg(&mut s);
            let mut y = 1.0 + 2.0 * x0 - x1 + 0.5 * lcg(&mut s);
            if i < burst && x0 > 0.0 && lcg(&mut s) > 0.2 {
                y += 50.0;
            }
            let w = 0.5 + (lcg(&mut s) + 1.0);
            ([x0, x1], y, w)
        })
        .collect()
}

fn unstandardized(units: bool) -> KalmanCfg {
    KalmanCfg {
        noise_units: units,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        half_life: vec![],
        q: Some(vec![0.0; 3]),
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    }
}

/// With no process noise, `P` in units of the noise is Bayesian regression
/// whatever the noise: the posterior mean under the prior `N(0, p0)` on each
/// slope, the intercept's diffuse, solves `(Σ w z zᵀ + diag(0, 1/p0, ...)) b
/// = Σ w z y`, which faer's LU solves here (`crate::oracle`). A burst of
/// residuals 100 noise deviations wide among the first 500 rows moves it no
/// differently from any rows. With the estimate in the gain the burst
/// inflated the noise, which weighs each row by its inverse, so the first
/// rows held the fit: slopes off by more than 1 after 2,000 rows (D-K).
/// Measured 2026-10-25: within 2.4e-15 of the solve (relative), against a
/// bound of 1e-10, a condition of about 10 times `n u`; the form before,
/// slopes 1.85 off.
#[test]
fn with_no_process_noise_a_burst_leaves_the_fit_the_batch_posterior() {
    let data = rows(2000, 500, 31);
    let burst = data
        .iter()
        .take(500)
        .filter(|(x, y, _)| (y - (1.0 + 2.0 * x[0] - x[1])).abs() > 20.0)
        .count();
    assert!(burst > 50, "{burst} rows of the burst");
    let fit = |units: bool| {
        let mut m = Kalman::new(unstandardized(units)).unwrap();
        for (i, (x, y, w)) in data.iter().enumerate() {
            m.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, *w);
        }
        m.coefficients()[0].clone()
    };
    let mut a = vec![0.0; 9];
    let mut rhs = vec![0.0; 3];
    for (x, y, w) in &data {
        let z = [1.0, x[0], x[1]];
        for i in 0..3 {
            rhs[i] += w * z[i] * y;
            for j in 0..3 {
                a[i * 3 + j] += w * z[i] * z[j];
            }
        }
    }
    a[4] += 1.0;
    a[8] += 1.0;
    let want = crate::oracle::solve(&a, &rhs);
    let got = fit(true);
    let worst = (0..3)
        .map(|i| (got[i] - want[i]).abs() / want[i].abs().max(1.0))
        .fold(0.0f64, f64::max);
    eprintln!("in units of the noise: {got:?} against {want:?}, {worst:.1e} apart");
    assert!(worst <= 1e-10, "{got:?} against {want:?}");
    let before = fit(false);
    let gap = (1..3)
        .map(|i| (before[i] - want[i]).abs())
        .fold(0.0f64, f64::max);
    eprintln!("with the estimate in the gain: {before:?}, slopes {gap:.2} off");
    assert!(gap > 1.0, "the burst captured the form before: {gap}");
}

/// The first row with a diffuse intercept is the limit of an ordinary
/// update as the intercept's prior variance grows without bound: `b_0` takes
/// the row's innovation, and `P` the posterior of `b_0 = y − xᵀ b_{1..}`,
/// `P_00 = p0 |x|² + 1/w`, `P_0i = −p0 x_i`, the slopes' block at its prior.
/// The ordinary update from `P_00 = 1e12` reaches it to about 1e-12 of its
/// size.
#[test]
fn the_first_row_takes_the_diffuse_intercepts_exact_step() {
    let mut m = Kalman::new(unstandardized(true)).unwrap();
    assert_eq!(m.p[0][0], 0.0, "the intercept waits, diffuse");
    let (x, y, w) = ([0.7, -1.3], 4.25, 2.0);
    m.step(&x, &[Some(y)], 0.0, w);
    let s = 0.7f64 * 0.7 + 1.3 * 1.3 + 1.0 / w;
    assert_eq!(m.beta[0], vec![y, 0.0, 0.0]);
    let want = [s, -0.7, 1.3, -0.7, 1.0, 0.0, 1.3, 0.0, 1.0];
    for (g, e) in m.p[0].iter().zip(want) {
        assert!(
            (g - e).abs() <= 4.0 * f64::EPSILON * e.abs().max(1.0),
            "{:?}",
            m.p[0]
        );
    }
    // The ordinary update from a large finite prior on the intercept.
    let big = 1e12;
    let p0 = [big, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let z = [1.0, x[0], x[1]];
    let pz: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| p0[i * 3 + j] * z[j]).sum())
        .collect();
    let zpz: f64 = (0..3).map(|i| z[i] * pz[i]).sum();
    let sd = zpz + 1.0 / w;
    for i in 0..3 {
        assert!((m.beta[0][i] - pz[i] / sd * y).abs() <= 1e-9, "b_{i}");
        for j in 0..3 {
            let lim = p0[i * 3 + j] - pz[i] * pz[j] / sd;
            assert!((m.p[0][i * 3 + j] - lim).abs() <= 1e-3, "P_{i}{j}: {lim}");
        }
    }
}

fn drifting(units: bool, share: bool, m: usize) -> KalmanCfg {
    KalmanCfg {
        noise_units: units,
        n_features: 2,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(300.0),
        half_life: vec![200.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: share,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    }
}

/// `P`'s recursion reads `z`, the weight and the process noise, which a
/// half-life derives in units of the noise: two targets of noise 0.5 and
/// 50, observed on the same rows, keep the same `P` to the bit, and the
/// gain is one gain. With the estimate in the gain each target's `P` moved
/// with its own noise.
#[test]
fn the_covariance_reads_no_noise() {
    let fit = |units: bool| {
        let mut k = Kalman::new(drifting(units, false, 2)).unwrap();
        for (i, (x, y, w)) in rows(3000, 0, 7).into_iter().enumerate() {
            let loud = y + 99.5 * lcg(&mut (i as u64 + 1));
            k.step(
                &x,
                &[Some(y), Some(loud)],
                if i == 0 { 0.0 } else { 1.0 },
                w,
            );
        }
        k
    };
    let k = fit(true);
    assert!(k.sig2[1] > 100.0 * k.sig2[0], "{:?}", k.sig2);
    let bits = |p: &[f64]| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&k.p[0]), bits(&k.p[1]));
    let before = fit(false);
    assert_ne!(bits(&before.p[0]), bits(&before.p[1]), "the form before");
}

/// So with every target on every row, one shared `P` is each target's own,
/// and `share_p` predicts as the per-target filters do, to the bit; with
/// the estimate in the gain the shared `P` read the mean noise, and the
/// two parted.
#[test]
fn share_p_is_the_per_target_filter_when_every_target_is_on_every_row() {
    let run = |units: bool, share: bool| {
        let mut k = Kalman::new(drifting(units, share, 2)).unwrap();
        let mut out = Vec::new();
        for (i, (x, y, w)) in rows(2000, 0, 11).into_iter().enumerate() {
            let other = 3.0 - y + 2.0 * lcg(&mut (i as u64 + 5));
            out.push(
                k.step(
                    &x,
                    &[Some(y), Some(other)],
                    if i == 0 { 0.0 } else { 1.0 },
                    w,
                )
                .pred,
            );
        }
        out
    };
    let bits = |v: &[Vec<f64>]| v.iter().flatten().map(|p| p.to_bits()).collect::<Vec<_>>();
    let (own, shared) = (run(true, false), run(true, true));
    assert!(own.iter().skip(1).all(|p| p.iter().all(|v| v.is_finite())));
    assert_eq!(bits(&own), bits(&shared));
    assert_ne!(
        bits(&run(false, false)),
        bits(&run(false, true)),
        "the form before"
    );
}

/// A level `L` in the target reaches only the intercept: from the first
/// row, which the diffuse intercept takes whole, every prediction moves by
/// `L` to the rounding of a number of `L`'s size, a few `u (|L| + max|y|)`
/// (measured 2026-10-25: at most 16.7 of them over the levels below; bound
/// 64).
/// With the estimate in the gain the first innovations, the level itself,
/// sized the noise and the prior, and the level moved the fit for thousands
/// of rows (K1: 0.03 of the target's spread at rows 3,000-6,000 for 1e4).
#[test]
fn a_level_moves_every_prediction_by_the_level_from_the_first_row() {
    let data = rows(6000, 0, 3);
    let run = |units: bool, level: f64| {
        let mut k = Kalman::new(drifting(units, false, 1)).unwrap();
        data.iter()
            .enumerate()
            .map(|(i, (x, y, w))| {
                k.step(x, &[Some(y + level)], if i == 0 { 0.0 } else { 1.0 }, *w)
                    .pred[0]
            })
            .collect::<Vec<f64>>()
    };
    let base = run(true, 0.0);
    let ymax = data.iter().map(|(_, y, _)| y.abs()).fold(0.0f64, f64::max);
    assert!(base[0].is_nan() && base[1..].iter().all(|p| p.is_finite()));
    for level in [1e4, -1e6, 1e8] {
        let got = run(true, level);
        let unit = f64::EPSILON / 2.0 * (level.abs() + ymax);
        let worst = got[1..]
            .iter()
            .zip(&base[1..])
            .map(|(g, b)| (g - level - b).abs() / unit)
            .fold(0.0f64, f64::max);
        eprintln!("level {level:e}: worst {worst:.1} units of u (|L| + max|y|)");
        assert!(worst <= 64.0, "level {level}: {worst}");
    }
    let before = run(false, 1e4);
    let base_before = run(false, 0.0);
    let moved = before
        .iter()
        .zip(&base_before)
        .skip(3000)
        .map(|(g, b)| (g - 1e4 - b).abs())
        .fold(0.0f64, f64::max);
    assert!(moved > 1e-3, "the form before moved by the level: {moved}");
}

/// A target first seen after its shared covariance took its first row is
/// centred as the diffuse intercept is: its intercept takes its first
/// innovation, so its first prediction is about its level, not 0. With
/// none observed yet it is not predicted.
#[test]
fn a_target_first_seen_late_under_share_p_is_centred_on_its_first_row() {
    let mut k = Kalman::new(drifting(true, true, 2)).unwrap();
    let mut first = None;
    for (i, (x, y, w)) in rows(400, 0, 5).into_iter().enumerate() {
        let late = (i >= 100).then_some(1e6 + y);
        let pred = k
            .step(&x, &[Some(y), late], if i == 0 { 0.0 } else { 1.0 }, w)
            .pred;
        if i <= 100 {
            assert!(pred[1].is_nan(), "row {i}: {}", pred[1]);
        } else if first.is_none() {
            first = Some(pred[1]);
        }
    }
    let first = first.unwrap();
    assert!((first - 1e6).abs() < 10.0, "{first}");
}

/// A zero-weight row, or a null target, before the first row that observes
/// it leaves the intercept diffuse: it is the first row that observes the
/// target that takes the step, and every row before it is predicted as
/// nothing.
#[test]
fn rows_that_observe_nothing_leave_the_intercept_diffuse() {
    let mut k = Kalman::new(drifting(true, false, 1)).unwrap();
    let data = rows(10, 0, 9);
    for (i, (x, y, _)) in data.iter().enumerate().take(4) {
        let (y, w) = if i % 2 == 0 {
            (Some(*y), 0.0)
        } else {
            (None, 1.0)
        };
        let pred = k.step(x, &[y], if i == 0 { 0.0 } else { 1.0 }, w).pred;
        assert!(pred[0].is_nan());
        assert!(k.open(0), "row {i}");
    }
    let (x, y, w) = data[4];
    k.step(&x, &[Some(y)], 1.0, w);
    assert!(!k.open(0));
    assert_eq!(k.beta[0][0], y, "the intercept took the innovation whole");
    let (x, _, _) = data[5];
    assert_eq!(k.predict(&x, 1.0).pred[0], y);
}

/// Every readout scales `P` in units of the noise by the target's own
/// residual variance `σ²`: with no process noise, unstandardized, `P` is the
/// posterior's `(Σ w z zᵀ + diag(0, 1/p0, ...))⁻¹`, which faer inverts
/// here, so `pred_var(x) = σ² (zᵀ P z + 1)`, `se_coef² = σ² diag(P)`, the
/// row's `error_inflation = sqrt(1 + zᵀ P z)`, which needs no `σ²`, and the
/// summary's `sqrt(1 + Σ_i P_ii E[z_i²])`. Before a residual variance there
/// is no noise to scale by: `pred_var` and the variances are NaN, the
/// inflation infinite.
#[test]
fn the_readouts_are_p_times_the_targets_own_noise() {
    let data = rows(400, 0, 13);
    let mut m = Kalman::new(unstandardized(true)).unwrap();
    let x0 = [0.3, -0.2];
    let mut infl = Vec::new();
    m.row_error_inflation_into(&x0, 1.0, &mut infl);
    assert!(infl[0].is_infinite());
    assert!(m.pred_var(&x0)[0].is_nan());
    for (i, (x, y, w)) in data.iter().enumerate() {
        m.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, *w);
        if i == 0 {
            // Sized, and no residual variance yet.
            let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance() else {
                panic!()
            };
            assert!(v[0].iter().all(|v| v.is_nan()), "{v:?}");
        }
    }
    let mut a = vec![0.0; 9];
    for (x, _, w) in &data {
        let z = [1.0, x[0], x[1]];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] += w * z[i] * z[j];
            }
        }
    }
    a[4] += 1.0;
    a[8] += 1.0;
    let p = crate::oracle::inverse(&a);
    let s2 = m.sig2[0];
    assert!(s2 > 0.0);
    let z = [1.0, x0[0], x0[1]];
    let quad: f64 = (0..3)
        .map(|i| (0..3).map(|j| z[i] * p[i * 3 + j] * z[j]).sum::<f64>())
        .sum();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs();
    let got = m.pred_var(&x0)[0];
    assert!(
        close(got, s2 * (quad + 1.0)),
        "{got} against {}",
        s2 * (quad + 1.0)
    );
    let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance() else {
        panic!()
    };
    for i in 0..3 {
        assert!(
            close(v[0][i], s2 * p[i * 4]),
            "{i}: {} against {}",
            v[0][i],
            s2 * p[i * 4]
        );
    }
    m.row_error_inflation_into(&x0, 1.0, &mut infl);
    assert!(close(infl[0], (1.0 + quad).sqrt()), "{infl:?}");
    // `E[z_i²]` over the rows, as the summary reads it.
    let ez2 = m.design_second_moments();
    let trace: f64 = (0..3).map(|i| p[i * 4] * ez2[i]).sum();
    let mut summary = Vec::new();
    m.error_inflation_into(&mut summary);
    assert!(close(summary[0], (1.0 + trace).sqrt()), "{summary:?}");
}

/// An explicit `q` is in the coefficients' own units, so in units of the
/// noise `P` takes `q / σ²`: after a gap, `P` as it stands carries `q D² /
/// σ²` on each slot beside what it held, and a filter that has no noise
/// yet takes none.
#[test]
fn an_explicit_q_is_measured_against_the_noise() {
    let q = vec![0.0, 0.02, 0.03];
    let mut cfg = unstandardized(true);
    cfg.q = Some(q.clone());
    let mut m = Kalman::new(cfg).unwrap();
    let mut held = vec![0.0; 3];
    m.q_held_into(0.0, &mut held);
    assert_eq!(held, vec![0.0; 3], "no noise, no process noise");
    for (i, (x, y, w)) in rows(300, 0, 17).iter().enumerate() {
        m.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, *w);
    }
    let s2 = m.sig2[0];
    m.q_held_into(s2, &mut held);
    for i in 0..3 {
        assert_eq!(held[i], q[i] / s2);
    }
    // A row that observes nothing, a gap of 3: `P` as it stands carries it.
    m.step(&[0.1, 0.2], &[None], 3.0, 1.0);
    let now = m.covariance_now(0);
    for i in 1..3 {
        let want = m.p[0][i * 4] + q[i] / s2 * 9.0;
        assert!((now[i * 4] - want).abs() <= 1e-12 * want, "{i}");
    }
}

/// Unstandardized every slot takes the process noise, but a diffuse
/// intercept takes none before its first row: a first observed row after a
/// gap of rows that observe nothing, which the noise is charged for, still
/// takes the diffuse step whole.
#[test]
fn a_gap_before_the_first_observed_row_leaves_the_intercept_diffuse() {
    let mut cfg = unstandardized(true);
    cfg.q = None;
    cfg.half_life = vec![20.0];
    let mut m = Kalman::new(cfg).unwrap();
    for _ in 0..3 {
        m.step(&[0.5, 0.5], &[Some(9.0)], 2.0, 0.0);
    }
    assert!(m.open(0));
    let (x, y, w) = ([0.7, -1.3], 1e6 + 4.25, 2.0);
    m.step(&x, &[Some(y)], 2.0, w);
    assert_eq!(m.beta[0][0], y, "the level, whole");
    let pz: Vec<f64> = (0..3).map(|i| m.p[0][i * 3]).collect();
    // `P_00 = z' P z + 1 / w` over the slopes' prior and the gap's noise.
    let slopes = [m.p[0][4], m.p[0][8]];
    let want = slopes[0] * 0.49 + slopes[1] * 1.69 + 1.0 / w;
    assert!(
        (pz[0] - want).abs() <= 1e-12 * want,
        "{} against {want}",
        pz[0]
    );
    assert!(
        slopes[0] > 1.0,
        "the slopes took the gap's noise: {slopes:?}"
    );
}

/// Per target, a target first seen after another has moved the moments has
/// its slopes' priors carried into an anchor away from them, whose cross
/// terms with the intercept are kept off its diffuse row, so the row leaves
/// the intercept's posterior variance at least `1 / w`, where those terms
/// would leave a prior sure of 0; and its first prediction is at its level,
/// 1e6.
#[test]
fn a_target_first_seen_late_takes_its_level_whole_beside_another() {
    let mut k = Kalman::new(drifting(true, false, 2)).unwrap();
    let mut s = 23u64;
    let mut first = None;
    for i in 0..600 {
        // A feature whose mean climbs, so the anchor trails it.
        let x = [i as f64 / 50.0 + 0.3 * lcg(&mut s), lcg(&mut s)];
        let y = 1.0 + 0.5 * x[0] - x[1] + 0.1 * lcg(&mut s);
        let late = (i >= 400).then_some(1e6 + y);
        let pred = k
            .step(&x, &[Some(y), late], if i == 0 { 0.0 } else { 1.0 }, 1.0)
            .pred;
        if i == 400 {
            // The diffuse posterior: `P_00 = z' P z + 1 / w`, at least 1.
            let (p00, c) = (k.p[1][0], k.map_c.clone());
            eprintln!("P_00 after its first row {p00}, the anchor's shift {c:?}");
            assert!(p00 >= 1.0, "{p00}");
        }
        if i > 400 && first.is_none() {
            first = Some(pred[1]);
        }
    }
    let first = first.unwrap();
    assert!((first - 1e6).abs() < 10.0, "{first}");
}

/// A variance of exactly 0 beside covariances that are not 0 is no
/// covariance: `|P_ij|² ≤ P_ii P_jj` holds of every positive semi-definite
/// matrix, so a row of a 0 variance is 0 throughout. A slot's diagonal
/// rounded to 0 by an update at the input bound left such a `P` (task 278
/// (b32): a reverting filter on features and weights of 1e100), which the
/// diagonal's own check passed, and which read as unsized before the
/// transition and sized after it, its coupling writing the intercept's
/// variance from those covariances: `predict` withheld what `step`
/// predicted, -1.5e82. It is sized again at the end of the row, as a
/// negative variance is, and the next row's `predict` is its `step`, in
/// both forms. Damaged here by hand.
#[test]
fn a_zero_variance_beside_covariances_is_sized_again() {
    for units in [false, true] {
        let mut c = drifting(units, false, 1);
        c.revert_half_life = vec![f64::INFINITY, 25.0, 6.0];
        let mut m = Kalman::new(c).unwrap();
        let data = rows(200, 0, 29);
        for (i, (x, y, w)) in data.iter().enumerate().take(150) {
            m.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, *w);
        }
        let k = 3;
        // The slopes' variances 0, their covariances kept.
        for i in 1..k {
            m.p[0][i * k + i] = 0.0;
        }
        assert!(m.p[0][k + 2] != 0.0, "a covariance is kept");
        // A row that observes nothing ends with the check too: the damage
        // was done by hand between rows, where a row's own update would have
        // been caught at its end.
        m.step(&data[150].0, &[None], 1.0, 1.0);
        let p = &m.p[0];
        for i in 0..k {
            for j in 0..k {
                assert!(
                    p[i * k + j] * p[i * k + j] <= p[i * k + i] * p[j * k + j] * (1.0 + 1e-12),
                    "units {units}: P[{i}][{j}] against its variances: {p:?}"
                );
            }
        }
        for (x, y, w) in data.iter().skip(150) {
            let served = m.predict(x, 1.0).pred;
            let got = m.step(x, &[Some(*y)], 1.0, *w).pred;
            assert_eq!(served[0].to_bits(), got[0].to_bits(), "units {units}");
        }
    }
}
