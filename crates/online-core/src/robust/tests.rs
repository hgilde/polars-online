//! `robust.rs`'s unit tests, in a file of their own so that neither file
//! passes the 250 KB a source file is held to.

use super::*;

/// A state whose vectors are not the cfg's is refused, where it loaded
/// and panicked on the first `step` (review 2026-09-18, B3).
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    use crate::{ModelState, OnlineModel, StateError};
    let m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
    let mut s = m.state();
    let ModelState::Robust(inner) = &mut s.model else {
        unreachable!()
    };
    inner.cross[0].pop();
    match Robust::restore(&s) {
        Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
        other => panic!("{other:?}"),
    }
}
use crate::{EwRidge, EwRidgeCfg};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// The stream the model contract's proptest failed on once its values
/// reached the input bound (review 2026-09-26, G2): a target of `1e100`
/// once, then features of `1e100` at weights from `1e-100` up. The fit
/// extrapolated to `-1.5e199` at `1e100`, that residual set the scale to
/// `5.8e148`, and the next row outside the band nudged the slope to
/// `1e248` -- one term of the score against a Gram holding no curvature
/// for a row that far out -- so the prediction at `1e100` read `-inf`.
/// Bounded by the row's leverage, the nudge brings the row to its band's
/// edge and every prediction is a number.
#[test]
fn a_quantile_fit_stays_finite_through_the_bound() {
    use crate::OnlineModel;
    let mut c = cfg(2, 2, RobustLoss::Quantile { tau: 0.5 });
    c.decay = Decay::Halflife(20.0);
    c.ridge = 1e-6;
    c.min_weight = 3.0;
    let mut m = Robust::new(c).unwrap();
    let rows: Vec<([f64; 2], Option<f64>, f64)> = vec![
        ([0.0, 0.0], None, 1.0),
        ([0.0, 0.0], Some(0.0), 1.0),
        ([0.0, 0.0], Some(0.0), 1.0),
        ([0.0, 0.0], Some(0.0), 1.0),
        ([0.0, 0.0], Some(0.0), 1.1785268524789025),
        ([0.0, 0.0], Some(0.0), 1.0),
        ([0.0, 0.0], Some(1e100), 2.0699847121199655),
        ([0.0, 0.0], Some(0.0), 1.4511913482607992),
        ([0.0, 0.0], Some(0.0), 1.0),
        ([1.4589775263789706, 0.0], Some(0.0), 1.0),
        ([1e100, 0.0], Some(0.0), 1e-100),
        ([0.0, 0.0], Some(0.0), 3.8874731502776667),
        ([1e100, 0.0], Some(0.0), 1.0),
        ([1e100, 0.0], None, 1.0),
    ];
    for (i, (x, y0, w)) in rows.iter().enumerate() {
        let out = m.step(x, &[*y0, None], if i == 0 { 0.0 } else { 1.0 }, *w);
        assert!(
            out.pred[0].is_nan() || out.pred[0].is_finite(),
            "row {i}: {:?}",
            out.pred
        );
        // Row 0 carries no target: no fit yet (review round 4, CC1).
        let beta = &m.beta.as_ref().unwrap()[0];
        let none_yet = i == 0 && beta.iter().all(|b| b.is_nan());
        assert!(
            none_yet || beta.iter().all(|b| b.is_finite()),
            "row {i}: {beta:?}"
        );
    }
    // The last nudge brought the row toward its band rather than past
    // it: the fit at `x = 1e100` is within the residual it started from.
    let at = m.predict(&[1e100, 0.0], 1.0).pred[0];
    assert!(at.is_finite() && at.abs() < 1e199, "{at:e}");
}

fn cfg(k: usize, m: usize, loss: RobustLoss) -> RobustCfg {
    RobustCfg {
        n_features: k,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        loss,
        ridge: 1e-8,
        standardize: false,
        min_weight: (k + 1) as f64,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 1e-3,
        huber_scale: HuberScale::Rms,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows: 0,
        min_weight_holds_learning: false,
        mad_anchor: false,
    }
}

#[test]
fn cfg_validation_rejects_each_bad_field() {
    let huber = RobustLoss::Huber { delta: 1.5 };
    let bad = |loss: RobustLoss, f: &dyn Fn(&mut RobustCfg), want: &str| {
        let mut c = cfg(2, 1, loss);
        f(&mut c);
        match c.validate() {
            Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
            Ok(()) => panic!("expected rejection mentioning {want:?}"),
        }
    };
    bad(huber, &|c| c.n_features = 0, "must be >= 1");
    bad(huber, &|c| c.n_targets = 0, "must be >= 1");

    // delta is the crossover from squared to absolute loss.
    for d in [0.0, -1.0, f64::NAN] {
        bad(
            huber,
            &|c| c.loss = RobustLoss::Huber { delta: d },
            "huber_delta",
        );
    }
    cfg(2, 1, RobustLoss::Huber { delta: 1e-9 })
        .validate()
        .unwrap();

    // The quantile is an open interval: 0 and 1 are not quantiles a
    // weighted least-squares reformulation can represent.
    for t in [0.0, 1.0, -0.1, 1.1, f64::NAN] {
        bad(
            huber,
            &|c| c.loss = RobustLoss::Quantile { tau: t },
            "quantile must be in",
        );
    }
    for t in [1e-6, 0.5, 1.0 - 1e-6] {
        cfg(2, 1, RobustLoss::Quantile { tau: t })
            .validate()
            .unwrap();
    }

    // ridge may be zero; quantile_eps may not (it divides). Each named
    // with its value (review 2026-10-06, CC7).
    bad(
        huber,
        &|c| c.ridge = -1e-9,
        "ridge must be finite and >= 0, got -0.000000001",
    );
    let mut ok = cfg(2, 1, huber);
    ok.ridge = 0.0;
    ok.validate().unwrap();
    bad(
        huber,
        &|c| c.quantile_eps = 0.0,
        "quantile_eps must be finite and > 0, got 0",
    );
    bad(
        huber,
        &|c| c.quantile_eps = -1.0,
        "quantile_eps must be finite and > 0, got -1",
    );
}

#[test]
fn n_eff_counts_observations_not_irls_weights() {
    // The defect T-A5 found: the IRLS weights a quantile fit then used
    // reached `2 / quantile_eps`, so counting them made `n_eff` -- and
    // therefore `min_weight` -- meaningless. It must be the plain
    // weighted observation count, identical to every other model's, and it
    // still is now that the fit weighs the rows in its band (N9).
    for loss in [
        RobustLoss::Huber { delta: 1.5 },
        RobustLoss::Quantile { tau: 0.5 },
        RobustLoss::Quantile { tau: 0.9 },
    ] {
        let mut c = cfg(1, 1, loss);
        c.decay = Decay::Halflife(20.0);
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut want = 0.0;
        for i in 0..30 {
            let d = if i == 0 { 0.0 } else { 1.0 };
            let step = m.step(&[i as f64], &[Some(3.0 * i as f64)], d, 1.0);
            assert!(
                (step.n_eff - want).abs() < 1e-12,
                "{loss:?} row {i}: {} vs {want}",
                step.n_eff
            );
            want = want * 0.5f64.powf(d / 20.0) + 1.0;
        }
        assert!(
            want < 31.0,
            "an observation count cannot exceed the row count"
        );
    }
}

#[test]
fn a_solve_failure_is_counted_and_the_previous_fit_is_kept() {
    // Two perfectly collinear features with no ridge: the normal equations
    // are singular. The model must keep its last good coefficients rather
    // than emit NaN, and say so in `solve_failures`.
    let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
    c.ridge = 0.0;
    c.fit_intercept = true;
    c.min_weight = 2.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 101u64;
    for i in 0..40 {
        let a = lcg(&mut s);
        // x1 == x0, and the intercept is constant: rank deficient by two.
        m.step(
            &[a, a],
            &[Some(2.0 * a + 1.0)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let beta = m.coefficients().unwrap()[0].clone();
    assert!(
        beta.iter().all(|v| v.is_finite()),
        "never NaN, even singular: {beta:?}"
    );
    // Either the jitter rescued it (counted) or the solve failed (counted);
    // silently succeeding on a singular system is the outcome to rule out.
    assert!(m.solve_failures > 0, "a singular solve must be recorded");
}

/// `a_solve_failure_is_counted_and_the_previous_fit_is_kept` on the
/// standardized path, as the review wrote it (2026-09-12, S13). The two
/// collinear features give a singular correlation matrix, which the
/// jitter ladder rescues and counts; a solve that fails outright is the
/// next test's.
#[test]
fn a_standardized_solve_failure_is_counted_and_the_previous_fit_is_kept() {
    let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
    c.ridge = 0.0;
    c.standardize = true;
    c.min_weight = 2.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 101u64;
    for i in 0..40 {
        let a = lcg(&mut s);
        m.step(
            &[a, a],
            &[Some(2.0 * a + 1.0)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let beta = m.coefficients().unwrap()[0].clone();
    assert!(
        beta.iter().all(|v| v.is_finite()),
        "never NaN, even singular: {beta:?}"
    );
    assert!(m.solve_failures > 0, "a singular solve must be recorded");
}

/// A standardized solve that fails at every jitter keeps the previous
/// fit, as the plain one does, and is counted as the plain one is: it
/// returned through `?` before the count (review 2026-09-12, S13). No
/// stream of rows gives a correlation matrix that every jitter fails on,
/// so the accumulator is handed one: a correlation of 2.
#[test]
fn a_standardized_solve_that_fails_outright_is_counted() {
    let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
    c.standardize = true;
    let mut m = Robust::new(c).unwrap();
    let mut s = 103u64;
    for i in 0..40 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0] - x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let beta = m.coefficients().unwrap()[0].clone();
    let before = m.solve_failures;
    let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
    m.cov[0].set_moments(
        &[1.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
        w,
        q,
    );
    m.solve();
    assert_eq!(
        m.solve_failures,
        before + 1,
        "every jitter failed: one failure"
    );
    assert_eq!(
        m.coefficients().unwrap()[0],
        beta,
        "and the previous fit is kept"
    );
}

/// `null_targets_decay_the_residual_variance_weight_without_adding_to_it`
/// (ewridge.rs) with the target present at weight zero: a row that
/// teaches nothing leaves `σ²` where it was and ages its weight, as a
/// null row does. The zero weight skipped the ageing, so `σ²` -- the
/// scale of every Huber cut -- forgot less across such a row than across
/// a null (review 2026-09-12, S13; S9 was the same in `kalman`).
#[test]
fn a_zero_weight_row_ages_the_residual_variance_as_a_null_does() {
    let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1.5 });
    c.decay = Decay::Halflife(10.0);
    c.min_weight = 2.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 53u64;
    for i in 0..60 {
        let x = [lcg(&mut s)];
        let y = 2.0 * x[0] + 0.1 + 0.05 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let (sig, w) = (m.sig2[0], m.wsig[0]);
    assert!(sig > 0.0 && w > 0.0);
    let lam = 0.5f64.powf(3.0 / 10.0);
    m.step(&[0.5], &[Some(-500.0)], 3.0, 0.0);
    assert_eq!(m.sig2[0], sig, "weight 0 must not move sigma2");
    assert!(
        (m.wsig[0] - w * lam).abs() < 1e-12,
        "but its weight ages: {} vs {}",
        m.wsig[0],
        w * lam
    );
}

/// A row of weight 0 learns nothing, the residual variance included:
/// across every such row `σ²` keeps its bits, not `(a σ²) / a` --
/// `kalman`'s twin (task 158) for both losses. A quantile row outside
/// the band is a nudge, which skipped no zero weight before the `σ²`
/// update, and moved it by an ulp on some rows (review 2026-10-05,
/// CC5); the Huber row of weight 0 is a fit of weight 0, which did.
#[test]
fn a_zero_weight_row_keeps_the_residual_variance_to_the_bit() {
    for loss in [
        RobustLoss::Quantile { tau: 0.5 },
        RobustLoss::Quantile { tau: 0.9 },
        RobustLoss::Huber { delta: 1.345 },
    ] {
        let mut c = cfg(1, 1, loss);
        c.decay = Decay::Halflife(9.0);
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 107u64;
        let mut checked = 0;
        for i in 0..3000 {
            let x = [lcg(&mut s)];
            let y = 1.0 + x[0] + 0.3 * lcg(&mut s);
            // Far outside the band, where a quantile row is a nudge.
            let zero = i % 3 == 2;
            let (y, w) = if zero { (y + 50.0, 0.0) } else { (y, 1.0) };
            let before = m.sigma2()[0];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 0.7 }, w);
            if zero && i > 60 {
                assert!(before > 0.0, "{loss:?}, row {i}");
                assert_eq!(
                    m.sigma2()[0].to_bits(),
                    before.to_bits(),
                    "{loss:?}, row {i}"
                );
                checked += 1;
            }
        }
        assert!(checked > 900, "{loss:?}: {checked}");
    }
}

/// `σ²` is the EW mean of the squared out-of-sample errors: every row
/// ages its weight, and a row with a target, a weight and a prediction
/// adds `w·r²` (the row weight, not the robust one). Held on a stream
/// with zero-weight rows (S13) and a clock gap that takes `n_eff` under
/// `min_weight`. The fit's own prediction is the one it learns from,
/// whether or not the gate withholds it from the output (task 278,
/// D10): the rows after the gap are withheld and still learned. Under
/// the gate that held learning back (a state written before schema 64)
/// those rows had no prediction and aged nothing either (N6).
#[test]
fn the_residual_variance_ages_on_every_row() {
    for holds in [false, true] {
        let hl = 10.0;
        let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1.5 });
        c.decay = Decay::Halflife(hl);
        c.min_weight = 3.0;
        c.min_weight_holds_learning = holds;
        let mut m = Robust::new(c).unwrap();
        let (mut want, mut wsig, mut unpredicted, mut withheld) = (0.0f64, 0.0f64, 0, 0);
        let mut s = 61u64;
        for i in 0..160 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.1 + 0.2 * lcg(&mut s);
            let d = match i {
                0 => 0.0,
                80 => 200.0,
                _ => 1.0,
            };
            let (y, w) = match i % 9 {
                4 => (None, 1.0),
                7 => (Some(y), 0.0),
                _ => (Some(y), 1.0),
            };
            // The fit's own prediction, from the last solve.
            let own = m.slot_predictions(&x)[0];
            let p = m.step(&x, &[y], d, w).pred[0];
            let learned_from = if holds { p } else { own };
            wsig *= 0.5f64.powf(d / hl);
            match y {
                Some(y) if w > 0.0 && learned_from.is_finite() => {
                    let r = y - learned_from;
                    let ws_new = wsig + w;
                    want = (wsig * want + w * r * r) / ws_new;
                    wsig = ws_new;
                    if p.is_nan() {
                        withheld += 1;
                    }
                }
                Some(_) if w > 0.0 && wsig > 0.0 => unpredicted += 1,
                _ => {}
            }
            let got = m.sigma2()[0];
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "holds {holds}, row {i}: sigma2 {got}, the EW mean of the squared errors {want}"
            );
        }
        if holds {
            assert!(
                unpredicted >= 2,
                "the gap must leave rows with no prediction"
            );
        } else {
            assert_eq!(unpredicted, 0, "every row after the first solve is learned");
            assert!(withheld >= 2, "the gap must leave rows the gate withholds");
        }
    }
}

#[test]
fn huber_resists_outliers_that_break_least_squares() {
    let mut hub = Robust::new(cfg(1, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
    let mut ols = EwRidge::new(EwRidgeCfg {
        n_features: 1,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        ridge: vec![1e-8],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: 2.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: crate::TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    let mut s = 77u64;
    for i in 0..600 {
        let x = [lcg(&mut s)];
        // clean relationship y = 2x, with 3% enormous outliers
        let outlier = i % 33 == 7;
        let y = if outlier {
            500.0 * lcg(&mut s)
        } else {
            2.0 * x[0]
        };
        let d = if i == 0 { 0.0 } else { 1.0 };
        hub.step(&x, &[Some(y)], d, 1.0);
        ols.step(&x, &[Some(y)], d, 1.0);
    }
    let h = hub.coefficients().unwrap()[0][1];
    let o = ols.coefficients().unwrap()[0][1];
    assert!(
        (h - 2.0).abs() < (o - 2.0).abs(),
        "huber {h} should beat ols {o} (truth 2.0)"
    );
    assert!((h - 2.0).abs() < 0.5, "huber slope {h}");
}

#[test]
fn huber_matches_least_squares_without_outliers() {
    // With a huge delta nothing is downweighted, so it must reduce to OLS.
    let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1e9 })).unwrap();
    let mut s = 78u64;
    for i in 0..400 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 1.5 * x[0] - 0.5 * x[1] + 0.25;
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let b = &m.coefficients().unwrap()[0];
    assert!((b[0] - 0.25).abs() < 1e-6);
    assert!((b[1] - 1.5).abs() < 1e-6);
    assert!((b[2] + 0.5).abs() < 1e-6);
}

#[test]
fn quantile_tracks_the_requested_quantile() {
    // y = 1 + noise with an asymmetric spread; tau = 0.9 must sit clearly
    // above tau = 0.5, which must sit above tau = 0.1.
    let mut lo = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.1 })).unwrap();
    let mut mid = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.5 })).unwrap();
    let mut hi = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.9 })).unwrap();
    let mut s = 79u64;
    for i in 0..4000 {
        let x = [lcg(&mut s)];
        let y = 1.0 + 2.0 * lcg(&mut s); // uniform(-1,3) around the level
        let d = if i == 0 { 0.0 } else { 1.0 };
        lo.step(&x, &[Some(y)], d, 1.0);
        mid.step(&x, &[Some(y)], d, 1.0);
        hi.step(&x, &[Some(y)], d, 1.0);
    }
    let (a, b, c) = (
        lo.coefficients().unwrap()[0][0],
        mid.coefficients().unwrap()[0][0],
        hi.coefficients().unwrap()[0][0],
    );
    assert!(a < b && b < c, "quantile levels out of order: {a} {b} {c}");
}

/// The weight the per-target gate reads is the rows the target was
/// present on, decayed -- hard rule 8's -- whatever the loss does with
/// them. From N9 to the second review's F1 the quantile fit reported its
/// band's weight, which a half-life caps at the band's share of the
/// effective sample, so a `min_weight` above that share closed the gate
/// for good.
#[test]
fn the_target_weight_is_the_rows_present_whatever_the_band_holds() {
    let hl = 30.0;
    let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.9 });
    c.decay = Decay::Halflife(hl);
    c.min_weight = 0.0;
    let mut m = Robust::new(c).unwrap();
    let (mut want, mut s, mut out) = (0.0f64, 93u64, Vec::new());
    for i in 0..400 {
        let x = [lcg(&mut s)];
        let (y, w) = match i % 9 {
            4 => (None, 1.0),
            7 => (Some(x[0] + lcg(&mut s)), 0.0),
            _ => (Some(x[0] + lcg(&mut s)), 1.0),
        };
        let d = if i == 0 { 0.0 } else { 1.0 };
        m.target_n_eff_into(&mut out);
        assert!(
            (out[0] - want).abs() <= 1e-12 * want.max(1.0),
            "row {i}: {} reported, {want} present",
            out[0]
        );
        m.step(&x, &[y], d, w);
        want *= 0.5f64.powf(d / hl);
        if y.is_some() && w > 0.0 {
            want += w;
        }
    }
    assert!(want > 30.0, "the stream must have settled: {want}");
}

/// The bounded-extremes contract's case (batch 7): a row at the input
/// bound sets the Gram and the cross-moment at its own scale, every later
/// row is outside the band, and the nudges, `2h * psi * z` over the
/// band's weight, cannot move the fit until that weight has decayed to
/// nothing -- where each is a step that outgrows the band. The warm-up
/// rebuilt such a fit while it read the band's weight; since F3 moved it
/// to the rows present, a band holding under one row per coefficient
/// takes least-squares rows until it holds rows again.
#[test]
fn a_band_under_a_row_per_coefficient_takes_least_squares_rows() {
    let hl = 20.0;
    let lam = 0.5f64.powf(1.0 / hl);
    let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
    c.decay = Decay::Halflife(hl);
    c.min_weight = 0.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 11u64;
    for i in 0..300 {
        let x = [lcg(&mut s)];
        let y = Some(x[0] + 0.1 * lcg(&mut s));
        m.step(&x, &[y], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let mut out = Vec::new();
    m.target_n_eff_into(&mut out);
    assert!(
        out[0] > 6.0 && m.wj[0] > 2.0,
        "settled: {} present, {} in the band",
        out[0],
        m.wj[0]
    );

    // Past the warm-up, with the band holding rows: a row far outside
    // it is a nudge, and the band's weight only decays.
    let x = [0.3];
    let wj = m.wj[0];
    m.step(&x, &[Some(1e6)], 1.0, 1.0);
    assert!(
        (m.wj[0] - lam * wj).abs() <= 1e-12 * wj,
        "a nudge weighs nothing: {} from {wj}",
        m.wj[0]
    );

    // The band starved to under a row per coefficient (`k = 2`): the same
    // row is a least-squares row, weighed at its weight and aimed at `y`.
    // `wj` is set by hand, so the accumulator's own weight is left where
    // it was; the target mean's share is the one `wj` gives.
    m.wj[0] = 1.5;
    let y0 = m.ybar[0];
    m.step(&x, &[Some(1e6)], 1.0, 1.0);
    let aged = lam * 1.5;
    assert!(
        (m.wj[0] - (aged + 1.0)).abs() <= 1e-12,
        "the row's weight enters: {} for {}",
        m.wj[0],
        aged + 1.0
    );
    let want = y0 + 1.0 / (aged + 1.0) * (1e6 - y0);
    assert!(
        (m.ybar[0] - want).abs() <= 1e-9 * want.abs(),
        "the target mean took the row at its target: {} for {want}",
        m.ybar[0]
    );
}

/// A level costs the fit nothing (the review of 2026-09-18, S2), as
/// `ewridge`'s test of the same name says of it: Huber at a `delta` that
/// makes it least squares, on the same stream at the origin and shifted
/// by `1e8` -- features and target alike, a price regressed on prices --
/// gives the same slopes and predictions that differ by the shift, plain
/// and standardized. The cross-moments were raw, and both solves lost
/// `L²·ε`, which at `1e8` was the whole fit. The tolerances are the
/// data's own resolution at `1e8`, `ulp ≈ 1.5e-8`.
#[test]
fn a_level_costs_the_fit_nothing() {
    for standardize in [false, true] {
        let run = |level: f64| {
            let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
            c.standardize = standardize;
            c.ridge = 1e-6;
            c.decay = Decay::Halflife(200.0);
            c.min_weight = 10.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 29u64;
            let mut preds = Vec::new();
            for i in 0..600 {
                let u = [lcg(&mut s), lcg(&mut s)];
                let x = [level + u[0], level + u[1]];
                let y = level + 2.0 * u[0] - u[1] + 0.1 * lcg(&mut s);
                let d = if i == 0 { 0.0 } else { 1.0 };
                preds.push(m.step(&x, &[Some(y)], d, 1.0).pred[0] - level);
            }
            (preds, m.coefficients().unwrap()[0].clone())
        };
        let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
        for i in 1..3 {
            assert!(
                (b0[i] - b8[i]).abs() < 1e-6,
                "standardize {standardize}: slope {i}: {} vs {}",
                b0[i],
                b8[i]
            );
        }
        let mut worst = 0.0f64;
        for (t, (a, b)) in p0.iter().zip(&p8).enumerate() {
            assert_eq!(a.is_finite(), b.is_finite(), "row {t}: {a} vs {b}");
            if a.is_finite() {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(
            worst < 1e-5,
            "standardize {standardize}: the predictions part by {worst}"
        );
    }
}

/// The same for the quantile loss, whose nudge enters the same centred
/// cross-moment: `ȳ += nudge/wj`, `c += nudge·(z − m)/wj` (S2).
#[test]
fn a_level_costs_the_quantile_fit_nothing() {
    let run = |level: f64| {
        let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.75 });
        c.decay = Decay::Halflife(500.0);
        c.min_weight = 20.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 37u64;
        let mut preds = Vec::new();
        for i in 0..1500 {
            let u = lcg(&mut s);
            let y = level + 2.0 * u + lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            preds.push(m.step(&[level + u], &[Some(y)], d, 1.0).pred[0] - level);
        }
        (preds, m.coefficients().unwrap()[0].clone())
    };
    let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
    assert!(
        (b0[1] - 2.0).abs() < 0.2,
        "the fixture is not what it claims: {b0:?}"
    );
    assert!(
        (b0[1] - b8[1]).abs() < 1e-6,
        "slope: {} vs {}",
        b0[1],
        b8[1]
    );
    let mut worst = 0.0f64;
    for (a, b) in p0.iter().zip(&p8) {
        assert_eq!(a.is_finite(), b.is_finite());
        if a.is_finite() {
            worst = worst.max((a - b).abs());
        }
    }
    assert!(worst < 1e-5, "the predictions part by {worst}");
}

/// With an intercept the cross-moment's slot 0 is exactly 0: `z_0 = 1`
/// and `m_0 = 1` once a row has entered, so nothing level-sized ever sits
/// in the right-hand side (S2).
#[test]
fn the_intercept_slot_of_the_cross_moment_is_exactly_zero() {
    let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
    let mut s = 41u64;
    for i in 0..50 {
        let x = [1e8 + lcg(&mut s), 1e8 + lcg(&mut s)];
        let y = 1e8 + x[0] - x[1];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        assert_eq!(m.cross[0][0], 0.0, "row {i}");
    }
}

/// N9: after the warm-up a row outside the band moves the fit by its
/// nudge alone -- the Gram's weight does not see it at all.
#[test]
fn a_quantile_row_outside_the_band_nudges_and_weighs_nothing() {
    let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.9 });
    c.min_weight = 0.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 90u64;
    for i in 0..200 {
        let x = [lcg(&mut s)];
        let y = 1.0 + x[0] + 0.3 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let gram_before = m.cov[0].n_eff();
    let mut present = Vec::new();
    m.target_n_eff_into(&mut present);
    let (p_before, b_before) = (present[0], m.coefficients().unwrap()[0].clone());
    // Far above the fit, where the band is `quantile_eps * sigma` wide.
    m.step(&[0.5], &[Some(500.0)], 1.0, 1.0);
    m.target_n_eff_into(&mut present);
    let b_after = m.coefficients().unwrap()[0].clone();
    assert!(
        (m.cov[0].n_eff() - gram_before).abs() < 1e-12,
        "a row outside the band weighs nothing in the Gram: {gram_before} -> {}",
        m.cov[0].n_eff()
    );
    assert!(
        (present[0] - p_before - 1.0).abs() < 1e-12,
        "and is a row the target was present on: {p_before} -> {}",
        present[0]
    );
    assert!(
        b_after[0] > b_before[0],
        "tau = 0.9 follows a row above it: {b_before:?} -> {b_after:?}"
    );
    assert!(
        b_after[0] - b_before[0] < 1.0,
        "by a nudge, not by the row itself: {b_before:?} -> {b_after:?}"
    );
}

/// Task 147: the same rows at a hundred times the weight make the same
/// quantile fit. The warm-up and the band's floor read the target's
/// weight against row counts, so rows at weight 100 left the warm-up on
/// their first row, and the fit, leaning on a one-row Gram, reached 1e51.
#[test]
fn a_quantile_fit_is_the_same_at_a_hundred_times_the_weight() {
    let fit = |scale: f64| {
        let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.8 });
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 97u64;
        let mut preds = Vec::new();
        for i in 0..600 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
            let w = scale * (0.5 + 0.5 * (lcg(&mut s) + 1.0));
            preds.push(
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w)
                    .pred[0],
            );
        }
        preds
    };
    let (one, many) = (fit(1.0), fit(100.0));
    for (i, (a, b)) in one.iter().zip(&many).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + a.abs()),
            "row {i}: {a} against {b}"
        );
    }
}

/// N9: under the warm-up a quantile fit is ordinary least squares, bit for
/// bit -- a Newton step needs a Hessian, and a band around a fit built from
/// a handful of rows is not one.
#[test]
fn a_quantile_fit_warms_up_as_least_squares() {
    let mut qc = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
    qc.min_weight = 0.0;
    let mut lc = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
    lc.min_weight = 0.0;
    let (mut q, mut l) = (Robust::new(qc).unwrap(), Robust::new(lc).unwrap());
    let mut s = 91u64;
    // `WARM_ROWS` per coefficient is nine rows here, so the tenth is the
    // first the quantile fit takes a step on.
    for i in 0..12 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        let qs = q.step(&x, &[Some(y)], d, 1.0);
        let ls = l.step(&x, &[Some(y)], d, 1.0);
        if (1..9).contains(&i) {
            // To the bit, NaN on the rows under a row per coefficient,
            // which both models' floor withholds.
            let bits = |p: &[f64]| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&qs.pred),
                bits(&ls.pred),
                "row {i} is inside the warm-up"
            );
            assert_eq!(qs.pred[0].is_finite(), i >= 3, "row {i}: the floor");
        }
    }
    assert_ne!(
        q.coefficients().unwrap()[0],
        l.coefficients().unwrap()[0],
        "and it parts from least squares once the band is in force"
    );
}

#[test]
fn reweighting_uses_the_prior_residual_only() {
    // An enormous single observation must not be able to fully absorb
    // itself: the weight comes from the prediction made BEFORE the update.
    let mut m = Robust::new(cfg(1, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
    let mut s = 80u64;
    for i in 0..200 {
        let x = [lcg(&mut s)];
        m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let before = m.coefficients().unwrap()[0].clone();
    m.step(&[1.0], &[Some(1e6)], 1.0, 1.0);
    let after = m.coefficients().unwrap()[0].clone();
    // it moves, but nowhere near 1e6
    assert!(
        (after[1] - before[1]).abs() < 100.0,
        "{:?} -> {:?}",
        before,
        after
    );
}

#[test]
fn state_roundtrip() {
    let mut m1 = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
    let mut s = 81u64;
    let rows: Vec<([f64; 2], f64)> = (0..120)
        .map(|_| {
            let x = [lcg(&mut s), lcg(&mut s)];
            (x, x[0] - 0.5 * x[1])
        })
        .collect();
    for (i, (x, y)) in rows[..60].iter().enumerate() {
        m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
    let mut m2 = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for (x, y) in &rows[60..] {
        assert_eq!(
            m1.step(x, &[Some(*y)], 1.0, 1.0).pred,
            m2.step(x, &[Some(*y)], 1.0, 1.0).pred
        );
    }
}

/// The band systems are state (task 170; [`BandSystems`]): a model and
/// its copy without them are not equal, and a model restored from its
/// state -- msgpack named and compact, and JSON -- equals the one saved,
/// its band systems included. Until schema 33 they were not state, a
/// restore built them again at the first nudge, and any two models that
/// differed only in them compared equal (task 166).
#[test]
fn the_band_systems_are_state() {
    use crate::OnlineModel;
    let mut m = Robust::new(cfg(3, 1, RobustLoss::Quantile { tau: 0.5 })).unwrap();
    let mut s = 7u64;
    for i in 0..50 {
        let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let y = x[0] - x[2] + 0.3 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert!(
        m.systems.0.iter().any(Option::is_some),
        "the case needs a system"
    );
    let mut bare = m.clone();
    bare.systems = BandSystems::default();
    assert_ne!(bare, m, "the band systems are compared");
    let state = m.state();
    let named: crate::State =
        rmp_serde::from_slice(&rmp_serde::to_vec_named(&state).unwrap()).unwrap();
    let compact: crate::State = rmp_serde::from_slice(&rmp_serde::to_vec(&state).unwrap()).unwrap();
    let json: crate::State = serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    for (how, back) in [("named", named), ("compact", compact), ("json", json)] {
        assert_eq!(Robust::restore(&back).unwrap(), m, "{how}");
    }
}

#[test]
fn new_surfaces_the_validation_error() {
    let e = Robust::new(cfg(1, 1, RobustLoss::Quantile { tau: 0.0 })).unwrap_err();
    assert!(e.contains("quantile must be in"), "{e}");
}

/// A robust fit has no window, so it keeps no runs (docs/PLAN.md task
/// 128).
#[test]
fn a_robust_fit_keeps_no_runs() {
    let m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
    assert!(m.cov.iter().all(|c| !c.keeps_runs()));
}

/// A state from before the runs' flag keeps no runs once restored: this
/// model has no window to read them (review 2026-09-26, C4).
#[test]
fn restore_keeps_no_runs() {
    let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.35 })).unwrap();
    let mut s = 3u64;
    for i in 0..20 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let d = if i == 0 { 0.0 } else { 1.0 };
        crate::OnlineModel::step(&mut m, &x, &[Some(x[0])], d, 1.0);
    }
    let mut v = serde_json::to_value(crate::OnlineModel::state(&m)).unwrap();
    crate::window::json_edit(&mut v, "runs", &mut |x| {
        *x = serde_json::json!({"x": [], "start": []});
    });
    let m = <Robust as crate::OnlineModel>::restore(&serde_json::from_value(v).unwrap()).unwrap();
    assert!(m.cov.iter().all(|c| !c.keeps_runs()));
}

// --- task 158: the mutation survivors --------------------------------

/// `solve_share` is refused unless it is finite and positive, and
/// accepted at any such value.
#[test]
fn a_bad_solve_share_is_refused() {
    let huber = RobustLoss::Huber { delta: 1.5 };
    for f in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut c = cfg(2, 1, huber);
        c.solve_share = Some(f);
        match c.validate() {
            Err(e) => assert!(e.contains("solve_share must be finite and > 0"), "{f}: {e}"),
            Ok(()) => panic!("solve_share {f} accepted"),
        }
    }
    for f in [1e-9, 0.02, 1.0, 5.0] {
        let mut c = cfg(2, 1, huber);
        c.solve_share = Some(f);
        c.validate().unwrap();
    }
}

/// The warm-up is `WARM_ROWS` rows per coefficient, and no more: with
/// three coefficients the row with nine before it is the first Newton
/// step, so the tenth row's prediction is the first to part from least
/// squares.
#[test]
fn the_warm_up_ends_at_three_rows_per_coefficient() {
    let mut qc = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
    qc.min_weight = 0.0;
    let mut lc = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
    lc.min_weight = 0.0;
    let (mut q, mut l) = (Robust::new(qc).unwrap(), Robust::new(lc).unwrap());
    let mut s = 91u64;
    for i in 0..=10 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        let (qs, ls) = (
            q.step(&x, &[Some(y)], d, 1.0),
            l.step(&x, &[Some(y)], d, 1.0),
        );
        if i == 10 {
            assert_ne!(qs.pred, ls.pred, "the ninth row was a Newton step");
        } else if i > 0 {
            let bits = |p: &[f64]| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&qs.pred),
                bits(&ls.pred),
                "row {i} follows least-squares rows only"
            );
        }
    }
}

/// The band's weight is read in rows of the target's mean weight
/// (task 147): at a weight of 4 a row, a band holding 6 holds 1.5 rows,
/// under one row per coefficient (`k = 2`), and the next row is a
/// least-squares row; a band holding 8 holds exactly 2, one row per
/// coefficient, and a row far outside it is a nudge. With no decay and
/// equal weights the count is exactly a quarter of the weight.
#[test]
fn the_band_is_counted_in_rows_of_the_mean_weight() {
    let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
    c.min_weight = 0.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 13u64;
    for i in 0..300 {
        let x = [lcg(&mut s)];
        let y = x[0] + 0.1 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 4.0);
    }
    assert_eq!((m.nobs[0], m.wobs[0]), (300.0, 1200.0));
    for (band, least_squares) in [(6.0, true), (8.0, false)] {
        let mut t = m.clone();
        t.wj[0] = band;
        t.step(&[0.3], &[Some(1e6)], 1.0, 4.0);
        let want = if least_squares { band + 4.0 } else { band };
        assert_eq!(
            t.wj[0],
            want,
            "a band of {band} ({} rows): least squares {least_squares}",
            band / 4.0
        );
    }
}

/// The band is open, `|r| < h`: a residual of exactly `h` is outside it.
/// Rows of zeros keep the fit at 0 and every residual exactly 0, so the
/// target has no scale, and a row is least squares whatever its residual
/// (task 177). With an `s²` of 1 put in, `h` is `quantile_eps` itself
/// once the floor `(k/n)^(2/5)` is under it; a row at `h` then nudges
/// (the band's weight stays) and one just under it is a fit row (the
/// band's weight takes it). The test read the band at the literal scale
/// of 1 the model took before a residual, which is gone.
#[test]
fn the_band_is_open_at_its_edge() {
    let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
    c.min_weight = 0.0;
    c.quantile_eps = 0.5;
    let mut m = Robust::new(c).unwrap();
    for i in 0..20 {
        m.step(&[0.0], &[Some(0.0)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert_eq!(m.sig2[0], 0.0, "every residual so far is exactly 0");
    assert_eq!(m.predict(&[0.0], 1.0).pred[0], 0.0);
    assert!(
        (2.0f64 / 20.0).powf(0.4) < 0.5,
        "the floor is under quantile_eps"
    );
    let wj = m.wj[0];
    let mut unscaled = m.clone();
    unscaled.step(&[0.0], &[Some(0.5)], 1.0, 1.0);
    assert_eq!(unscaled.wj[0], wj + 1.0, "with no scale, least squares");
    m.sig2[0] = 1.0;
    let mut edge = m.clone();
    edge.step(&[0.0], &[Some(0.5)], 1.0, 1.0);
    assert_eq!(edge.wj[0], wj, "a residual of h is outside the band");
    let mut inside = m.clone();
    inside.step(&[0.0], &[Some(0.5 - 1e-12)], 1.0, 1.0);
    assert_eq!(inside.wj[0], wj + 1.0, "one just under h is inside");
}

/// Through the origin and standardized, a feature that has only ever been
/// 0 has no scale and is dropped with a coefficient of 0: the fit of the
/// other is the fit without it, and no solve fails.
#[test]
fn a_feature_without_scale_is_dropped_through_the_origin() {
    let run = |dead: bool| {
        let mut c = cfg(
            if dead { 2 } else { 1 },
            1,
            RobustLoss::Huber { delta: 1e9 },
        );
        c.fit_intercept = false;
        c.standardize = true;
        let mut m = Robust::new(c).unwrap();
        let mut s = 17u64;
        for i in 0..50 {
            let a = lcg(&mut s);
            let x = if dead { vec![a, 0.0] } else { vec![a] };
            let y = 1.5 * a + 0.1 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        (m.coefficients().unwrap()[0].clone(), m.solve_failures)
    };
    let ((with, failed), (without, _)) = (run(true), run(false));
    assert_eq!(failed, 0, "no solve failed");
    assert_eq!(with[1], 0.0, "the dead feature's coefficient");
    assert!(
        (with[0] - without[0]).abs() <= 1e-12 * without[0].abs(),
        "{} against {}",
        with[0],
        without[0]
    );
}

/// `a_solve_failure_is_counted_and_the_previous_fit_is_kept` through the
/// origin: two equal features and no ridge make the raw Gram singular,
/// the jitter ladder rescues it, and the rescue is counted.
#[test]
fn a_solve_failure_through_the_origin_is_counted() {
    let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
    c.ridge = 0.0;
    c.fit_intercept = false;
    c.min_weight = 2.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 101u64;
    for i in 0..40 {
        let a = lcg(&mut s);
        m.step(
            &[a, a],
            &[Some(2.0 * a)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let beta = &m.coefficients().unwrap()[0];
    assert!(beta.iter().all(|v| v.is_finite()), "{beta:?}");
    assert!(m.solve_failures > 0, "a singular solve must be recorded");
}

/// The rows a model solves on: where its coefficients appear or move.
/// Every row is fresh, so a solve always moves them.
fn solve_rows(m: &mut Robust, n: usize) -> Vec<usize> {
    let mut s = 7u64;
    let mut out = Vec::new();
    for i in 0..n {
        let before = m.coefficients().map(<[Vec<f64>]>::to_vec);
        let x = [lcg(&mut s)];
        let y = x[0] + lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        if m.coefficients().map(<[Vec<f64>]>::to_vec) != before {
            out.push(i);
        }
    }
    out
}

/// The clock cadence: the first solve on the row the target holds a row
/// per coefficient (row 1, two coefficients), whatever `min_weight` says
/// (task 278, D10), then one each `solve_every` of clock (5), with `max_rows_between_solves` too
/// far off to decide. Under the gate that held learning back the first
/// fell on the row the weight reached `min_weight`, here the same row.
#[test]
fn the_clock_cadence_solves_every_solve_every() {
    let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
    c.solve_every = 5.0;
    c.max_rows_between_solves = 1000;
    let mut held = Robust::new(RobustCfg {
        min_weight_holds_learning: true,
        ..c.clone()
    })
    .unwrap();
    let mut m = Robust::new(c).unwrap();
    assert_eq!(solve_rows(&mut m, 30), [1, 6, 11, 16, 21, 26]);
    assert_eq!(solve_rows(&mut held, 30), [1, 6, 11, 16, 21, 26]);
}

/// The row cadence: no clock (`solve_every` infinite), a solve every
/// `max_rows_between_solves` rows (4) after the first, on row 1, the
/// model's floor (and the weight's `min_weight`, under the gate that held
/// learning back).
#[test]
fn the_row_cadence_solves_every_max_rows_between_solves() {
    let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
    c.solve_every = f64::INFINITY;
    c.max_rows_between_solves = 4;
    let mut held = Robust::new(RobustCfg {
        min_weight_holds_learning: true,
        ..c.clone()
    })
    .unwrap();
    let mut m = Robust::new(c).unwrap();
    assert_eq!(solve_rows(&mut m, 20), [1, 5, 9, 13, 17]);
    assert_eq!(solve_rows(&mut held, 20), [1, 5, 9, 13, 17]);
}

/// Task 180: `solve_every` reads the stamps its caller hands, for both
/// losses: every 2,000th 1 ms row under 2 s after the first solve, on
/// row 1, where the target holds a row per coefficient, whatever
/// `min_weight` (0 here) says (task 278, D10), where the clock summed
/// since the last solve, read when none is handed, is a row late each
/// time, as it always was.
#[test]
fn solve_every_measures_the_stamps_it_is_handed() {
    let losses = [
        RobustLoss::Huber { delta: 1.0 },
        RobustLoss::Quantile { tau: 0.5 },
    ];
    for loss in losses {
        for (stamped, want) in [(true, [1, 2001, 4001]), (false, [1, 2002, 4003])] {
            let mut c = cfg(1, 1, loss);
            c.min_weight = 0.0;
            c.solve_every = 2.0;
            c.max_rows_between_solves = u32::MAX;
            let mut m = Robust::new(c).unwrap();
            let got = crate::since::events_on_millisecond_rows(4_100, stamped, |i, stamp, d| {
                if let Some(s) = stamp {
                    m.stamp_next(s);
                }
                let x = (i % 7) as f64;
                m.step(&[x], &[Some(1.0 + 2.0 * x)], d, 1.0);
                m.rows_since_solve == 0
            });
            assert_eq!(got, want, "{loss:?}, stamped: {stamped}");
        }
    }
}

/// Task 278, D13: on the row clock the model's start is a row before
/// its first row, so before the first solve at `min_weight` (here never
/// met) `solve_every = 7` solves on the rows `max_rows_between_solves =
/// 7` does (0-based 6, 13, ...), for both losses. Stamped as a clock
/// column's rows, whose start is the first row, the clock's come a row
/// later. Under the gate that held learning back (a state written before
/// schema 64), whose first solve waits for `min_weight`; since, the first
/// solve falls at the model's floor and the cadence counts from it (task
/// 278, D10).
#[test]
fn on_the_row_clock_solve_every_is_its_row_cap() {
    let losses = [
        RobustLoss::Huber { delta: 1.0 },
        RobustLoss::Quantile { tau: 0.5 },
    ];
    for loss in losses {
        let solves = |every: f64, cap: u32, on_rows: bool| {
            let mut c = cfg(1, 1, loss);
            c.min_weight = 1e9;
            c.min_weight_holds_learning = true;
            c.solve_every = every;
            c.max_rows_between_solves = cap;
            let mut m = Robust::new(c).unwrap();
            crate::since::events_on_the_row_clock(30, |i, stamp, d| {
                if on_rows {
                    m.stamp_next_on_rows(stamp);
                } else {
                    m.stamp_next(stamp);
                }
                let x = (i % 7) as f64;
                m.step(&[x], &[Some(1.0 + 2.0 * x)], d, 1.0);
                m.rows_since_solve == 0
            })
        };
        let rows = [6, 13, 20, 27];
        assert_eq!(solves(7.0, u32::MAX, true), rows, "{loss:?}");
        assert_eq!(solves(1e18, 7, true), rows, "{loss:?}");
        assert_eq!(solves(7.0, u32::MAX, false), [7, 14, 21, 28], "{loss:?}");
    }
}

/// The share cadence, set as the spec sets it (`set_solve_share`, after
/// building): a solve once the weight learned since the last one reaches
/// `solve_share` of the weight the fit holds (docs/PLAN.md task 115
/// (b)), the clock left out. The rows are the rule's, written out.
#[test]
fn the_share_cadence_solves_at_its_share_of_the_weight() {
    let mut c = cfg(1, 1, RobustLoss::Huber { delta: 1e9 });
    c.solve_every = f64::INFINITY;
    c.max_rows_between_solves = 1000;
    let mut m = Robust::new(c).unwrap();
    assert_eq!(m.solve_share(), None);
    m.set_solve_share(Some(0.25));
    assert_eq!(m.solve_share(), Some(0.25));
    let (mut held, mut since, mut fit, mut want) = (0.0, 0.0, false, Vec::new());
    for i in 0..60 {
        held += 1.0;
        since += 1.0;
        if since >= 0.25 * held || (!fit && held >= 2.0) {
            want.push(i);
            (since, fit) = (0.0, true);
        }
    }
    assert!(want.len() > 8 && want[want.len() - 1] - want[want.len() - 2] > 5);
    assert_eq!(solve_rows(&mut m, 60), want);
    m.set_solve_share(None);
    assert_eq!(m.solve_share(), None);
}

/// A target that has never been present has no fit, so no prediction,
/// beside one that has.
#[test]
fn a_target_never_present_has_no_prediction() {
    let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.5 });
    c.min_weight = 2.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 3u64;
    for i in 0..20 {
        let x = [lcg(&mut s)];
        m.step(&x, &[Some(x[0]), None], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let p = m.predict(&[0.5], 1.0).pred;
    assert!(p[0].is_finite() && p[1].is_nan(), "{p:?}");
}

/// A target with no weight at a solve has no fit: its row of
/// coefficients is NaN, so once its rows arrive it predicts nothing
/// until a solve has seen them. The solve left it the zeros `beta`
/// starts at and spent the first-solve trigger, so the target was
/// predicted as exactly 0.0 until the next scheduled solve, 56 rows of
/// 80 under `solve_every = 1000` (review round 4, CC1). And a late
/// target has a first solve of its own, as a fresh model has: on the
/// row it reaches the model's floor, a row per coefficient (task 278,
/// D10), or under the gate that held learning back the row its own
/// weight first reached its `min_weight`: here row 11 either way, so it
/// is predicted from row 12 on.
/// It waited for the next solve on the cadence, the row cap 25 rows
/// after the first solve at row 1, and was null until row 27
/// (docs/PLAN.md task 195, S9b).
#[test]
fn a_target_that_joins_after_the_first_solve_is_not_predicted_from_zeros() {
    for (loss, holds) in [
        (RobustLoss::Huber { delta: 1.5 }, true),
        (RobustLoss::Quantile { tau: 0.5 }, true),
        (RobustLoss::Huber { delta: 1.5 }, false),
        (RobustLoss::Quantile { tau: 0.5 }, false),
    ] {
        let mut c = cfg(1, 2, loss);
        c.min_weight = 2.0;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 25;
        c.min_weight_holds_learning = holds;
        let mut m = Robust::new(c).unwrap();
        // Its first solve: at the end of row 11, its second, under the
        // gate that held learning back its weight's `min_weight`, and
        // since the model's floor, a row per coefficient.
        let first = 11;
        let mut s = 11u64;
        for i in 0..40 {
            let x = [lcg(&mut s)];
            let b = (i >= 10).then(|| 1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s));
            let p = m.step(
                &x,
                &[Some(0.5 + x[0]), b],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
            let what = format!("{loss:?}, holds {holds}, row {i}: {:?}", p.pred);
            if i <= first {
                assert!(p.pred[1].is_nan(), "{what}");
            } else {
                // Its own first solve saw its rows.
                assert!((p.pred[1] - (1.0 - 2.0 * x[0])).abs() < 0.1, "{what}");
            }
            if (2..=11).contains(&i) {
                let beta = m.coefficients().expect("solved at row 1");
                assert_eq!(
                    beta[1].iter().all(|v| v.is_nan()),
                    i < first,
                    "{loss:?}, holds {holds}, row {i}: {beta:?}"
                );
                assert!(beta[0].iter().all(|v| v.is_finite()), "{what}");
            }
        }
    }
}

/// Under the gate that held learning back (a state written before
/// schema 64), the threshold a late target's own first solve waits for
/// is its own, as a bank holds one per target: at 5 the target joining
/// at row 10 is first solved at the end of its fifth row, row 14, where
/// at the model's 2 it was row 11; one solve, and none on the rows
/// between (docs/PLAN.md task 195, S9b). Since, its own threshold
/// withholds its output alone: it is first solved on the row it reaches
/// the model's floor, a row per coefficient, row 11, and every
/// coefficient is the same at 5 as at 2 (task 278, D10).
/// A threshold list of the wrong length, or one below 0, is refused, and
/// so is a state holding one.
#[test]
fn a_late_targets_first_solve_waits_for_its_own_min_weight() {
    let run = |holds: bool, own: f64| {
        let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.345 });
        c.min_weight = 2.0;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 1000;
        c.min_weight_holds_learning = holds;
        let mut m = Robust::new(c).unwrap();
        assert!(m.set_target_min_weight(vec![2.0]).is_err());
        assert!(m.set_target_min_weight(vec![2.0, -1.0]).is_err());
        m.set_target_min_weight(vec![2.0, own]).unwrap();
        let mut s = 7u64;
        let mut first = None;
        let mut coefs = Vec::new();
        for i in 0..20 {
            let x = [lcg(&mut s)];
            let b = (i >= 10).then(|| 1.0 - 2.0 * x[0]);
            m.step(&x, &[Some(x[0]), b], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let fit = m
                .coefficients()
                .is_some_and(|b| b[1].iter().all(|v| v.is_finite()));
            if fit && first.is_none() {
                first = Some(i);
            }
            coefs.push(
                m.coefficients()
                    .map(|b| b.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>()),
            );
        }
        (m, first, coefs)
    };
    let (_, first, _) = run(true, 5.0);
    assert_eq!(first, Some(14), "the gate held it to its own 5");
    let (_, first, _) = run(true, 2.0);
    assert_eq!(first, Some(11));
    let (m, first, at5) = run(false, 5.0);
    assert_eq!(first, Some(11), "solved at the model's floor");
    let (_, _, at2) = run(false, 2.0);
    assert_eq!(at5, at2, "its own threshold moves no coefficient");
    let mut st = m.state();
    let crate::ModelState::Robust(inner) = &mut st.model else {
        unreachable!()
    };
    inner.target_min_weight = vec![1.0];
    assert!(matches!(Robust::restore(&st), Err(StateError::Invalid(_))));
}

/// A target whose weight a gap takes to exactly 0 keeps the fit its
/// moments hold, as at one half-life short of the underflow: a fit does
/// not move on a gap (hard rule 8), and past the underflow the decay
/// forgets as just short of it (docs/PLAN.md task 115 (c)). The solve
/// on the row after the gap gave it zeros, predicted as 0.0 once rows
/// came (review round 4, CC1).
#[test]
fn a_target_whose_weight_underflows_keeps_its_fit() {
    for loss in [
        RobustLoss::Huber { delta: 1.5 },
        RobustLoss::Quantile { tau: 0.5 },
    ] {
        let fit = |gap: f64| {
            let mut c = cfg(1, 1, loss);
            c.decay = Decay::Halflife(10.0);
            c.min_weight = 0.0;
            let mut m = Robust::new(c).unwrap();
            let mut s = 29u64;
            for i in 0..60 {
                let x = [lcg(&mut s)];
                let y = 1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(1.0 - 2.0 * x[0])], gap, 0.0);
            (m.wj[0], m.coefficients().unwrap()[0].clone())
        };
        let (w_forgot, forgot) = fit(10_750.0);
        let (w_aged, aged) = fit(10_740.0);
        assert!(w_forgot == 0.0 && w_aged > 0.0, "{loss:?}: the case");
        for (a, b) in forgot.iter().zip(&aged) {
            assert!(
                (a - b).abs() <= 1e-9 * (1.0 + b.abs()),
                "{loss:?}: {forgot:?} against {aged:?}"
            );
        }
        assert!((aged[1] + 2.0).abs() < 0.05, "{loss:?}: {aged:?}");
    }
}

/// A first solve that fails at every jitter leaves no fit: NaN, not the
/// zeros `beta` starts at, which predicted 0.0 as though learned (review
/// round 4, CC1; `ewridge`'s CA5). No stream of rows gives a
/// correlation matrix every jitter fails on, so the accumulator is
/// handed one, as `a_standardized_solve_that_fails_outright_is_counted`
/// hands it.
#[test]
fn a_first_solve_that_fails_leaves_no_fit() {
    let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
    c.standardize = true;
    // The gate that held the first solve back with learning (a state
    // written before schema 64), so the rows fill the sums unsolved.
    c.min_weight = 1e9;
    c.min_weight_holds_learning = true;
    c.solve_every = f64::INFINITY;
    c.max_rows_between_solves = u32::MAX;
    let mut m = Robust::new(c).unwrap();
    let mut s = 103u64;
    for i in 0..40 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0] - x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    assert!(
        m.coefficients().is_none(),
        "min_weight held the first solve"
    );
    let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
    m.cov[0].set_moments(
        &[1.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
        w,
        q,
    );
    m.solve();
    assert_eq!(m.solve_failures, 1, "every jitter failed");
    let beta = m.coefficients().expect("a solve ran");
    assert!(beta[0].iter().all(|v| v.is_nan()), "{beta:?}");
}

/// A model deserialized on its own, outside `restore`, has no row buffer
/// (it is not state); its first step makes one, and it goes on as the
/// model it was saved from.
#[test]
fn a_model_deserialized_without_restore_steps_on() {
    let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.5 })).unwrap();
    let mut s = 5u64;
    let mut row = || {
        let x = [lcg(&mut s), lcg(&mut s)];
        (x, x[0] - x[1])
    };
    for i in 0..10 {
        let (x, y) = row();
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let bytes = rmp_serde::to_vec_named(&m).unwrap();
    let mut back: Robust = rmp_serde::from_slice(&bytes).unwrap();
    assert!(back.zbuf.is_empty(), "the buffer is not saved");
    for _ in 0..10 {
        let (x, y) = row();
        assert_eq!(
            m.step(&x, &[Some(y)], 1.0, 1.0).pred,
            back.step(&x, &[Some(y)], 1.0, 1.0).pred
        );
    }
}

/// Each of the shape checks on load refuses on its own: an accumulator
/// of the wrong width, a cross-moment row too many, a per-target vector
/// short, a coefficient row short, a coefficient row too many.
#[test]
fn each_shape_check_refuses_on_its_own() {
    let mut m = Robust::new(cfg(2, 1, RobustLoss::Huber { delta: 1.0 })).unwrap();
    let mut s = 9u64;
    for i in 0..10 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert!(m.beta.is_some());
    type Spoil = (&'static str, fn(&mut Robust));
    let cases: [Spoil; 5] = [
        ("an accumulator two wide", |m| {
            m.cov[0] = EwCov::new(2).without_runs();
        }),
        ("a cross-moment row too many", |m| {
            m.cross.push(vec![0.0; 3])
        }),
        ("no residual variance", |m| m.sig2.clear()),
        ("a short coefficient row", |m| {
            m.beta.as_mut().unwrap()[0].truncate(2);
        }),
        ("a coefficient row too many", |m| {
            m.beta.as_mut().unwrap().push(vec![0.0; 3]);
        }),
    ];
    for (what, spoil) in cases {
        let mut bad = m.clone();
        spoil(&mut bad);
        match Robust::restore(&bad.state()) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
            other => panic!("{what}: {other:?}"),
        }
    }
    assert!(Robust::restore(&m.state()).is_ok());
}

/// A zero-weight row is not one of the rows a quantile fit counts toward
/// its warm-up and band floor (hard rule 9): with no decay, a stream with
/// zero-weight rows among its first rows is the stream without them.
#[test]
fn zero_weight_rows_do_not_count_toward_the_warm_up() {
    let run = |ghosts: bool| {
        let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.7 });
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 91u64;
        let mut preds = Vec::new();
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 0.5 + x[0] - 0.25 * x[1] + 0.3 * lcg(&mut s);
            if ghosts && i < 8 {
                m.step(&[-x[1], x[0]], &[Some(-y)], 0.0, 0.0);
            }
            let d = if i == 0 { 0.0 } else { 1.0 };
            preds.push(m.step(&x, &[Some(y)], d, 1.0).pred[0]);
        }
        (preds, m.nobs[0])
    };
    let ((with, n_with), (without, n_without)) = (run(true), run(false));
    assert_eq!(n_with, n_without, "the zero-weight rows were counted");
    for (i, (a, b)) in with.iter().zip(&without).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
            "row {i}: {a} against {b}"
        );
    }
}

/// What `row_update` makes of target `j`'s next row, `y` scored at
/// `pred` at weight `w` after a clock step of `d`, under `scale`: its
/// other inputs read from the model's state before the row, as `step`
/// forms them.
fn update_of(
    m: &Robust,
    j: usize,
    (pred, y, d, w): (f64, f64, f64, f64),
    scale: Option<f64>,
) -> RowUpdate {
    let lam = m.cfg.decay.factor(d);
    let (present, rows, aged) = (lam * m.wobs[j], lam * m.nobs[j], lam * m.wj[j]);
    let aged_rows = if present > 0.0 {
        aged * (rows / present)
    } else {
        0.0
    };
    m.row_update(y, pred, scale, w, rows, aged_rows)
}

/// Whether target 0's next row, `y` scored at `pred` at weight `w` after
/// a clock step of `d`, is a nudge: `row_update`'s own reading of the
/// model's state before the row, as `step` forms its inputs.
fn is_nudge(m: &Robust, pred: f64, y: f64, d: f64, w: f64) -> bool {
    matches!(
        update_of(m, 0, (pred, y, d, w), m.residual_scale(0)),
        RowUpdate::Nudge { .. }
    )
}

/// A nudge brings its row at most to its target, never past it (the
/// module docs): the solve after a nudged row moves that row's
/// prediction by no more than its residual. The step was bounded by the
/// band Gram's diagonal leverage, `Σ d²/v`, where the solve moves the
/// row by its full leverage, `1 + uᵀA⁻¹u` over the solve's own system:
/// against features correlated at 0.999 a row at (+1, −1) reads about 2
/// on the diagonal and about 2,000 in full, and was thrown 6 to 13 times
/// its residual, 30 to 127 times at 0.9999 (review 2026-10-05, TC1b;
/// this test on the old code: 5.9 to 7.5, and 30 to 36). Every
/// nudged row is held to it -- with an intercept, standardized or not,
/// and through the origin -- and on uncorrelated features, where the
/// two leverages agree. Every hundredth row past the warm-up runs
/// against the correlation, at (+1, −1), 0.4 above the true line.
#[test]
fn a_nudge_never_moves_its_row_past_its_residual() {
    for rho in [0.0, 0.999, 0.9999] {
        for (fit_intercept, standardize) in [(true, false), (true, true), (false, false)] {
            let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
            c.decay = Decay::Halflife(500.0);
            c.min_weight = 0.0;
            c.ridge = 1e-6;
            c.quantile_eps = 0.2;
            c.fit_intercept = fit_intercept;
            c.standardize = standardize;
            let mut m = Robust::new(c).unwrap();
            let mut s = 5u64;
            let (mut checked, mut glitches, mut worst) = (0, 0, (0.0f64, 0usize));
            for i in 0..3000usize {
                let a = 3f64.sqrt() * lcg(&mut s);
                let b = rho * a + (1.0 - rho * rho).sqrt() * 3f64.sqrt() * lcg(&mut s);
                let glitch = i >= 1000 && i % 100 == 0;
                let x = if glitch { [1.0, -1.0] } else { [a, b] };
                let noise = if glitch { 0.4 } else { 0.5 * lcg(&mut s) };
                // Through the origin the line through the origin.
                let level = if fit_intercept { 1.0 } else { 0.0 };
                let y = level + 0.8 * x[0] - 0.4 * x[1] + noise;
                let d = if i == 0 { 0.0 } else { 1.0 };
                let before = m.predict(&x, d).pred[0];
                let nudge = before.is_finite() && is_nudge(&m, before, y, d, 1.0);
                m.step(&x, &[Some(y)], d, 1.0);
                if nudge {
                    let after = m.predict(&x, 1.0).pred[0];
                    let moved = (after - before).abs() / (y - before).abs();
                    // NaN, a movement not a number, counts as the worst.
                    if moved.is_nan() || moved > worst.0 {
                        worst = (moved, i);
                    }
                    checked += 1;
                    glitches += usize::from(glitch);
                }
            }
            let case = format!("rho {rho}, intercept {fit_intercept}, standardize {standardize}");
            assert!(checked > 300, "{case}: {checked} nudges");
            assert!(glitches > 0, "{case}: no glitch row was nudged");
            assert!(
                worst.0 <= 1.0 + 1e-9,
                "{case}: row {} moved {:.3} times its residual",
                worst.1,
                worst.0
            );
        }
    }
}

/// On uncorrelated features the full leverage is the diagonal one: a
/// band Gram set to a diagonal matrix reads a row's leverage as `Σ
/// d²/(v + ridge)` unstandardized and `Σ (d²/v)/(1 + ridge)` standardized,
/// to rounding, and so as the old reading, `Σ d²/v`, to the ridge's
/// share of the smallest variance.
#[test]
fn the_full_leverage_is_the_diagonal_one_on_uncorrelated_features() {
    for standardize in [false, true] {
        let mut c = cfg(3, 1, RobustLoss::Quantile { tau: 0.5 });
        c.ridge = 1e-6;
        c.standardize = standardize;
        let mut m = Robust::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..50 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[2])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (mean, var) = ([1.0, 0.2, -0.3, 0.5], [0.0, 0.7, 1.9, 0.04]);
        let mut cen = vec![0.0; 16];
        for i in 0..4 {
            cen[i * 4 + i] = var[i];
        }
        let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
        m.cov[0].set_moments(&mean, &cen, w, q);
        m.systems.0[0] = None;
        m.zbuf = RowBuf(vec![1.0, 1.1, -2.0, 0.9]);
        let ridge = 1e-6;
        let term = |i: usize| (m.zbuf[i] - mean[i]).powi(2);
        let with_ridge: f64 = 1.0
            + (1..4)
                .map(|i| {
                    if standardize {
                        term(i) / var[i] / (1.0 + ridge)
                    } else {
                        term(i) / (var[i] + ridge)
                    }
                })
                .sum::<f64>();
        let old: f64 = 1.0 + (1..4).map(|i| term(i) / var[i]).sum::<f64>();
        let full = m.nudge_movement(0);
        assert!(
            (full - with_ridge).abs() <= 1e-12 * with_ridge,
            "standardize {standardize}: {full} against {with_ridge}"
        );
        assert!(
            (full - old).abs() <= ridge / 0.04 * old,
            "standardize {standardize}: {full} against the old {old}"
        );
    }
}

/// A nudge is bounded by the row's own residual over its leverage (G2).
/// At the value a feature has always taken the leverage is 0, so a heavy
/// row outside the band moves the fit exactly to its target and no
/// further; off that value the leverage is unbounded (the Gram holds no
/// curvature for it) and the fit does not move. At a weight of `1e4` the
/// step before the bound is far past the residual, so the bound decides.
#[test]
fn a_nudge_on_a_feature_without_spread_is_bounded_by_the_residual() {
    let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
    c.min_weight = 0.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 23u64;
    for i in 0..40 {
        let y = 0.1 * lcg(&mut s);
        m.step(&[1.0], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert_eq!(m.cov[0].cov(1, 1), 0.0, "the feature has no spread");
    let p = m.predict(&[1.0], 1.0).pred[0];
    let mut on = m.clone();
    on.step(&[1.0], &[Some(p + 0.5)], 1.0, 1e4);
    assert_eq!(on.wj[0], m.wj[0], "the row was a nudge");
    let q = on.predict(&[1.0], 1.0).pred[0];
    assert!(
        (q - (p + 0.5)).abs() <= 1e-12,
        "brought to its target: {p} -> {q}, target {}",
        p + 0.5
    );
    let mut off = m.clone();
    let p2 = off.predict(&[2.0], 1.0).pred[0];
    off.step(&[2.0], &[Some(p2 + 0.5)], 1.0, 1e4);
    assert_eq!(off.wj[0], m.wj[0], "the row was a nudge");
    assert_eq!(off.coefficients(), m.coefficients(), "and moved nothing");
}

/// A leverage whose quadratic form is not a number is unbounded, and
/// the nudge's step 0, as for a feature with no spread. Features
/// correlated at 0.99 and a row at `(1e160, 5e159)`: `A⁻¹u` has entries
/// of opposite signs, so the form's two terms overflow to `+∞` and `−∞`,
/// and their sum is NaN. `f64::max(NaN, 0.0)` read it as 0: with an
/// intercept a leverage of 1, the least there is, bounding the step by
/// the whole residual, and through the origin a leverage of 0, bounding
/// it by nothing -- the overshoot the bound is there to stop (review
/// 2026-09-26, G2). Read as NaN, `|r| / NaN` bounds nothing either
/// (task 181). With an intercept and through the origin.
#[test]
fn a_leverage_that_is_not_a_number_is_unbounded() {
    for fit_intercept in [true, false] {
        let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
        c.fit_intercept = fit_intercept;
        let mut m = Robust::new(c).unwrap();
        let mut s = 29u64;
        for i in 0..50 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        // The band Gram set to unit variances correlated at 0.99, about
        // means of 0: centred with an intercept, raw through the origin.
        let off = usize::from(fit_intercept);
        let k = 2 + off;
        let mut mean = vec![0.0; k];
        let mut moments = vec![0.0; k * k];
        if fit_intercept {
            mean[0] = 1.0;
        }
        for i in 0..2 {
            for j in 0..2 {
                moments[(i + off) * k + j + off] = if i == j { 1.0 } else { 0.99 };
            }
        }
        let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
        m.cov[0].set_moments(&mean, &moments, w, q);
        m.systems.0[0] = None;
        let row = [1e160, 0.5e160];
        m.zbuf = RowBuf(if fit_intercept {
            vec![1.0, row[0], row[1]]
        } else {
            row.to_vec()
        });
        let q = m.nudge_movement(0);
        assert_eq!(q, f64::INFINITY, "intercept {fit_intercept}: {q}");
        // The same system reads a row at the data's scale.
        m.zbuf[off] = 1.0;
        m.zbuf[off + 1] = 0.5;
        let q = m.nudge_movement(0);
        assert!(q.is_finite() && q > 1.0, "intercept {fit_intercept}: {q}");
    }
}

/// A row with a feature that is not a finite number has no finite
/// prediction, so it reached the loss as a least-squares row and was
/// learned: its NaN went into the Gram and the cross-moment, every solve
/// after it failed -- the 40 of 40 rows that followed one -- and the fit
/// froze where it stood. Task 182 refused it in that arm, and counted it
/// in `n_eff` and the target's present weight. By the rule every model
/// keeps (`OnlineModel`, task 183) it is a row of weight 0 and counted
/// nowhere: it reports nothing, the Gram ages, the cross-moment and the
/// target's mean stay, and the state is the same row's at weight 0, its
/// features finite, byte for byte. Huber and the quantile loss alike; a
/// NaN, an infinity and a feature past the input bound alike.
#[test]
fn a_row_that_is_not_a_number_is_not_learned() {
    let bytes = |m: &Robust| rmp_serde::to_vec(&m.state()).unwrap();
    for loss in [
        RobustLoss::Huber { delta: 1.0 },
        RobustLoss::Quantile { tau: 0.5 },
    ] {
        for bad in [
            [f64::NAN, 0.5],
            [0.5, f64::NEG_INFINITY],
            [2.0 * crate::INPUT_BOUND, 0.5],
        ] {
            let case = format!("{loss:?}, {bad:?}");
            let mut c = cfg(2, 1, loss);
            c.decay = Decay::Halflife(50.0);
            c.ridge = 1e-6;
            c.min_weight = 3.0;
            let lam = c.decay.factor(1.0);
            let mut m = Robust::new(c).unwrap();
            let mut s = 9u64;
            let mut at_the_row = None;
            for i in 0..80 {
                let x = [lcg(&mut s), lcg(&mut s)];
                let y = 1.0 + x[0] + 2.0 * x[1] + 0.1 * lcg(&mut s);
                if i == 40 {
                    let before = m.clone();
                    let mut no_weight = m.clone();
                    no_weight.step(&x, &[Some(1.0)], 1.0, 0.0);
                    let out = m.step(&bad, &[Some(1.0)], 1.0, 1.0);
                    assert!(out.pred[0].is_nan(), "{case}: {out:?}");
                    assert_eq!(m.cross, before.cross, "{case}");
                    assert_eq!(m.ybar, before.ybar, "{case}");
                    assert_eq!(m.cov[0].means(), before.cov[0].means(), "{case}");
                    assert_eq!(m.cov[0].comoments(), before.cov[0].comoments(), "{case}");
                    assert_eq!(m.wj[0], lam * before.wj[0], "{case}");
                    assert_eq!(m.w_raw, lam * before.w_raw, "{case}");
                    assert_eq!(m.wobs[0], lam * before.wobs[0], "{case}");
                    assert_eq!(m.nobs[0], lam * before.nobs[0], "{case}");
                    assert_eq!(m.sig2, before.sig2, "{case}");
                    assert_eq!(bytes(&m), bytes(&no_weight), "{case}");
                    at_the_row = m.coefficients().map(<[_]>::to_vec);
                    continue;
                }
                let out = m.step(&x, &[Some(y)], 1.0, 1.0);
                if i > 40 {
                    assert!(out.pred[0].is_finite(), "{case}, row {i}: {out:?}");
                }
            }
            assert_eq!(m.solve_failures, 0, "{case}");
            let beta = m.coefficients().unwrap()[0].clone();
            assert_ne!(
                Some(vec![beta.clone()]),
                at_the_row,
                "{case}: the fit moved on"
            );
            for (b, want) in beta.iter().zip([1.0, 1.0, 2.0]) {
                assert!((b - want).abs() < 0.1, "{case}: {beta:?}");
            }
        }
    }
}

// --- task 170: the band factor moved where the move is exact ---------

/// The leverage a nudge would read for `probe` against target 0's band
/// system, from the system `m` keeps and from one built afresh from its
/// Gram: `(kept, fresh)`.
fn movement_kept_and_fresh(m: &Robust, probe: &[f64]) -> (f64, f64) {
    let z: Vec<f64> = if m.cfg.fit_intercept {
        std::iter::once(1.0).chain(probe.iter().copied()).collect()
    } else {
        probe.to_vec()
    };
    let (mut kept, mut fresh) = (m.clone(), m.clone());
    fresh.systems = BandSystems::default();
    kept.zbuf.copy_from_slice(&z);
    fresh.zbuf.copy_from_slice(&z);
    (kept.nudge_movement(0), fresh.nudge_movement(0))
}

/// Under `ridge = 0` a row inside the band moves the band system's
/// matrix by a scale and a rank-one step -- `A' = a·A + a·b·ũ ũᵀ` from the
/// centred Gram, `a·A + b·x̃ x̃ᵀ` from the raw one through the origin --
/// and under `standardize` by a rescaling after it, so the kept factor is
/// moved with it in `O(k²)` instead of dropped and rebuilt in `O(k³)` at
/// the next nudge (task 170, TC1b's cost). Held, after every such row a
/// solve did not follow, to a system built afresh from the Gram: the same
/// columns and scales to the bit, and the leverage a nudge reads -- on the
/// row itself, on a row against the features' correlation and on a third
/// -- within `1e-12` of the fresh system's (measured: 9.5e-15). With an
/// intercept and through the origin, standardized and not, under a
/// half-life and without one.
#[test]
#[ignore = "extended: a second or more (2.7 s)"]
fn without_a_ridge_a_band_row_moves_the_kept_factor() {
    for (fit_intercept, standardize) in [(true, false), (true, true), (false, false), (false, true)]
    {
        for hl in [200.0, f64::INFINITY] {
            let case =
                format!("intercept {fit_intercept}, standardize {standardize}, half-life {hl}");
            let mut c = cfg(3, 1, RobustLoss::Quantile { tau: 0.5 });
            c.decay = Decay::Halflife(hl);
            c.ridge = 0.0;
            c.min_weight = 0.0;
            c.quantile_eps = 0.2;
            c.fit_intercept = fit_intercept;
            c.standardize = standardize;
            c.solve_every = f64::INFINITY;
            c.max_rows_between_solves = 50;
            let mut m = Robust::new(c).unwrap();
            let mut s = 17u64;
            let (mut moved, mut worst) = (0usize, 0.0f64);
            for i in 0..3000usize {
                let a = lcg(&mut s);
                let x = [a, 0.9 * a + 0.4 * lcg(&mut s), lcg(&mut s)];
                let level = if fit_intercept { 1.0 } else { 0.0 };
                let y = level + x[0] - 0.5 * x[1] + 0.25 * x[2] + 0.5 * lcg(&mut s);
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
                let Some(sys) = m.systems.0.first().and_then(Option::as_ref) else {
                    continue;
                };
                // A factor no band row moved is fresh from a solve or a
                // nudge, the to-the-bit case of `solve.rs`.
                if sys.factor.as_ref().is_none_or(|f| f.moves() == 0) {
                    continue;
                }
                moved += 1;
                let (_, keep, scales) = m.band_system(0, 0);
                assert_eq!(keep, sys.keep, "{case}, row {i}");
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&scales), bits(&sys.s), "{case}, row {i}");
                for probe in [x, [1.0, -1.0, 0.5], [0.3, 0.2, -0.8]] {
                    let (kept, fresh) = movement_kept_and_fresh(&m, &probe);
                    assert!(fresh.is_finite() && fresh > 0.0, "{case}, row {i}: {fresh}");
                    worst = worst.max((kept - fresh).abs() / fresh);
                }
            }
            assert!(moved > 300, "{case}: {moved} rows read a moved factor");
            assert!(worst <= 1e-12, "{case}: the leverage parts by {worst:e}");
        }
    }
}

/// With a ridge on the band system, a band row moves `A` by a diagonal
/// shift as well, `(1 − a)·ridge·I`, which no `O(k²)` move makes, so the
/// kept factor is dropped and the next nudge factorizes afresh, as before
/// task 170: the default configuration (`ridge = 1e-6`) keeps TC1b's
/// numbers to the bit. A Huber fit drops it too: no nudge reads it.
#[test]
fn with_a_ridge_or_under_huber_a_band_row_drops_the_kept_factor() {
    for (loss, ridge) in [
        (RobustLoss::Quantile { tau: 0.5 }, 1e-6),
        (RobustLoss::Huber { delta: 1.345 }, 0.0),
    ] {
        let mut c = cfg(3, 1, loss);
        c.decay = Decay::Halflife(200.0);
        c.ridge = ridge;
        c.min_weight = 0.0;
        c.quantile_eps = 0.2;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 50;
        let mut m = Robust::new(c).unwrap();
        let mut s = 17u64;
        let mut fits = 0;
        for i in 0..3000usize {
            let a = lcg(&mut s);
            let x = [a, 0.9 * a + 0.4 * lcg(&mut s), lcg(&mut s)];
            let y = 1.0 + x[0] - 0.5 * x[1] + 0.25 * x[2] + 0.5 * lcg(&mut s);
            let wj = m.wj[0];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let fit = m.wj[0] > wj * 0.5f64.powf(1.0 / 200.0) * (1.0 + 1e-12);
            if fit && m.rows_since_solve > 0 {
                fits += 1;
                assert!(m.systems.0[0].is_none(), "{loss:?}, row {i}");
            }
        }
        assert!(fits > 300, "{loss:?}: {fits} fit rows");
    }
}

/// A band row that gives a standardized fit a column it did not keep --
/// a feature that had held one value takes another -- changes the
/// system's shape, which no move makes: the kept factor is dropped, and
/// the next nudge builds the system over the new columns (task 170).
#[test]
fn a_band_row_that_changes_the_kept_columns_drops_the_factor() {
    let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
    c.ridge = 0.0;
    c.standardize = true;
    c.min_weight = 0.0;
    c.quantile_eps = 0.2;
    c.solve_every = f64::INFINITY;
    c.max_rows_between_solves = 1000;
    let mut m = Robust::new(c).unwrap();
    let mut s = 31u64;
    for i in 0..300 {
        let a = lcg(&mut s);
        let y = 0.5 + a + 0.3 * lcg(&mut s);
        m.step(&[a, 0.0], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    // A nudge, so that a system is kept.
    let p = m.predict(&[0.1, 0.0], 1.0).pred[0];
    m.step(&[0.1, 0.0], &[Some(p + 50.0)], 1.0, 1.0);
    let sys = m.systems.0[0].as_ref().expect("the nudge kept a system");
    assert_eq!(sys.keep, [0], "the feature without spread is not kept");
    // A row at its prediction is inside the band, and moves feature 1.
    let p = m.predict(&[0.1, 0.5], 1.0).pred[0];
    let wj = m.wj[0];
    m.step(&[0.1, 0.5], &[Some(p)], 1.0, 1.0);
    assert!(m.wj[0] > wj, "the row was inside the band");
    assert!(m.systems.0[0].is_none(), "the system's columns changed");
    let p = m.predict(&[0.1, 0.0], 1.0).pred[0];
    m.step(&[0.1, 0.0], &[Some(p + 50.0)], 1.0, 1.0);
    assert_eq!(m.systems.0[0].as_ref().unwrap().keep, [0, 1]);
}

// --- task 170: the band systems are state ---------------------------

/// TC1b's stream: two features correlated at 0.999 and, from row 1000,
/// a row against the correlation every hundred rows, at (+1, −1), on
/// the true line, whose leverage is near 2,000 and whose nudge's bound
/// binds (`a_nudge_never_moves_its_row_past_its_residual`).
fn tc1b_rows(n: usize) -> Vec<([f64; 2], f64)> {
    let mut s = 5u64;
    (0..n)
        .map(|i| {
            let a = 3f64.sqrt() * lcg(&mut s);
            let b = 0.999 * a + (1.0 - 0.999f64 * 0.999).sqrt() * 3f64.sqrt() * lcg(&mut s);
            let glitch = i >= 1000 && i % 100 == 0;
            let x = if glitch { [1.0, -1.0] } else { [a, b] };
            let noise = if glitch { 0.4 } else { 0.5 * lcg(&mut s) };
            (x, 1.0 + 0.8 * x[0] - 0.4 * x[1] + noise)
        })
        .collect()
}

/// A quantile fit on `tc1b_rows`' configuration, at `ridge`.
fn tc1b_cfg(ridge: f64, standardize: bool, share: bool) -> RobustCfg {
    let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
    c.decay = Decay::Halflife(500.0);
    c.min_weight = 0.0;
    c.ridge = ridge;
    c.quantile_eps = 0.2;
    c.standardize = standardize;
    c.solve_every = 10.0;
    c.max_rows_between_solves = u32::MAX;
    c.solve_share = share.then_some(crate::DEFAULT_SOLVE_SHARE);
    c
}

/// Target 0's kept factor's moves, if a factor is kept.
fn band_moves(m: &Robust) -> Option<u32> {
    m.systems
        .0
        .first()?
        .as_ref()?
        .factor
        .as_ref()
        .map(SpdFactor::moves)
}

/// A quantile fit resumes to the bit at every save point (task 170):
/// saved -- named msgpack, as a bank writes it -- and restored at the 49
/// rows of the probe and at the three rows before each row against the
/// correlation, it goes on as the fit that never stopped, its band factor
/// read back as it was saved, the moves its band rows made since the
/// last solve included. Before the band systems were state, a restore
/// built a fresh factor where the saved fit read a moved one, and at
/// `ridge = 0` the probe's save point before row 1100 parted from the
/// uninterrupted fit by up to 7.5e-14 (1 of 49). Under both cadences,
/// standardized and not, and at the default ridge, whose kept factor
/// was already a fresh one. Each comparison runs the 300 rows after its
/// save point: a parting persists, and a solve comes within ten.
#[test]
#[ignore = "extended: a save at every row (1.9 s)"]
fn a_quantile_fit_resumes_to_the_bit_at_every_save_point() {
    use crate::OnlineModel;
    let rows = tc1b_rows(3000);
    let mut cuts: Vec<usize> = (1100..2900).step_by(37).collect();
    assert_eq!(cuts.len(), 49);
    cuts.extend((1100..2900).step_by(100).flat_map(|g| [g - 2, g - 1, g]));
    let step = |m: &mut Robust, i: usize| {
        let (x, y) = rows[i];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
            .pred[0]
            .to_bits()
    };
    for ridge in [0.0, 1e-6] {
        for standardize in [false, true] {
            for share in [false, true] {
                let case = format!("ridge {ridge}, standardize {standardize}, share {share}");
                let c = tc1b_cfg(ridge, standardize, share);
                let mut straight = Robust::new(c.clone()).unwrap();
                let mut saved = Vec::new();
                let mut preds = Vec::new();
                for i in 0..rows.len() {
                    if cuts.contains(&i) {
                        saved.push((i, rmp_serde::to_vec_named(&straight.state()).unwrap()));
                    }
                    preds.push(step(&mut straight, i));
                }
                let (mut off, mut moved) = (Vec::new(), 0);
                for (cut, bytes) in saved {
                    let mut back = Robust::restore(&rmp_serde::from_slice(&bytes).unwrap())
                        .unwrap_or_else(|e| panic!("{case}, cut {cut}: {e}"));
                    moved += usize::from(band_moves(&back).is_some_and(|k| k > 0));
                    if (cut..(cut + 300).min(rows.len())).any(|i| step(&mut back, i) != preds[i]) {
                        off.push(cut);
                    }
                }
                assert!(
                    off.is_empty(),
                    "{case}: {} save points off the bit: {off:?}",
                    off.len()
                );
                if ridge == 0.0 {
                    assert!(
                        moved > 10,
                        "{case}: {moved} save points held a moved factor"
                    );
                }
            }
        }
    }
}

/// A state holding a moved band factor goes through JSON and back as
/// itself (task 170): the restored fit equals the saved one, its factor
/// with the moves it had made, and goes on to the bit.
#[test]
fn a_moved_band_factor_survives_a_json_round_trip() {
    use crate::OnlineModel;
    let rows = tc1b_rows(1400);
    for standardize in [false, true] {
        let mut m = Robust::new(tc1b_cfg(0.0, standardize, true)).unwrap();
        let mut i = 0;
        while i < 1000 || band_moves(&m).is_none_or(|k| k == 0) {
            let (x, y) = rows[i];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            i += 1;
        }
        let json = serde_json::to_string(&m.state()).unwrap();
        let mut back = Robust::restore(&serde_json::from_str(&json).unwrap()).unwrap();
        assert_eq!(back, m, "standardize {standardize}");
        assert_eq!(
            band_moves(&back),
            band_moves(&m),
            "standardize {standardize}"
        );
        for (x, y) in &rows[i..] {
            assert_eq!(
                back.step(x, &[Some(*y)], 1.0, 1.0).pred[0].to_bits(),
                m.step(x, &[Some(*y)], 1.0, 1.0).pred[0].to_bits(),
                "standardize {standardize}"
            );
        }
    }
}

/// Each way a band system can be damaged in a state is refused on its
/// own (task 170), where the model would read a leverage from a system
/// that is not its Gram's: a slot too many, a column the Gram does not
/// keep, a scale an ulp off, no factor where columns are kept, one of
/// the wrong order, and one where none is kept. A factor that is not one
/// -- a pivot not positive, an entry not finite or missing, a jitter that
/// does not fit its rung, more moves than the cap, any on a jittered
/// factor -- is refused where the state is read; beside them a jittered
/// factor that made no move is read.
#[test]
fn each_damaged_band_system_is_refused_on_its_own() {
    use crate::OnlineModel;
    use serde_json::json;
    let fitted = |constant: bool| {
        let mut c = cfg(3, 1, RobustLoss::Quantile { tau: 0.5 });
        c.standardize = true;
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let x = if constant { [1.0, 2.0, 3.0] } else { x };
            let y = x[0] - x[2] + 0.3 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert!(m.systems.0[0].is_some(), "the case needs a system");
        m.state()
    };
    let base = fitted(false);
    let none_kept = fitted(true);
    type Spoil = (&'static str, bool, fn(&mut BandSystems), &'static str);
    let cases: [Spoil; 6] = [
        (
            "a slot too many",
            false,
            |b| b.0.push(None),
            "band systems for",
        ),
        (
            "a column the Gram does not keep",
            false,
            |b| {
                b.0[0].as_mut().unwrap().keep.pop();
            },
            "not of its Gram's",
        ),
        (
            "a scale an ulp off",
            false,
            |b| {
                let s = &mut b.0[0].as_mut().unwrap().s;
                s[1] = f64::from_bits(s[1].to_bits() + 1);
            },
            "not of its Gram's",
        ),
        (
            "no factor where columns are kept",
            false,
            |b| {
                b.0[0].as_mut().unwrap().factor = None;
            },
            "band factor is not of its 3",
        ),
        (
            "a factor of the wrong order",
            false,
            |b| {
                b.0[0].as_mut().unwrap().factor = SpdFactor::of(&[1.0], 1);
            },
            "band factor is not of its 3",
        ),
        (
            "a factor where no column is kept",
            true,
            |b| {
                b.0[0].as_mut().unwrap().factor = SpdFactor::of(&[1.0], 1);
            },
            "band factor is not of its 0",
        ),
    ];
    for (what, constant, spoil, want) in cases {
        let mut s = if constant {
            none_kept.clone()
        } else {
            base.clone()
        };
        let ModelState::Robust(inner) = &mut s.model else {
            unreachable!()
        };
        spoil(&mut inner.systems);
        match Robust::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains(want), "{what}: {e}"),
            other => panic!("{what}: {other:?}"),
        }
    }
    assert!(Robust::restore(&base).is_ok() && Robust::restore(&none_kept).is_ok());

    let v = serde_json::to_value(&base).unwrap();
    type JsonEdit = (&'static str, fn(&mut serde_json::Value));
    type Edit = (&'static str, &'static [JsonEdit]);
    let read = |edits: &[JsonEdit]| {
        let mut v = v.clone();
        for (key, edit) in edits {
            crate::window::json_edit(&mut v, key, &mut |x| edit(x));
        }
        serde_json::from_value::<crate::State>(v)
    };
    let refused: [Edit; 7] = [
        (
            "a pivot that is not positive",
            &[("lower", |x| x[0] = json!(-1.0))],
        ),
        (
            "an entry that is not finite",
            &[("lower", |x| x[1] = json!("nan"))],
        ),
        (
            "an entry missing",
            &[("lower", |x| {
                x.as_array_mut().unwrap().pop();
            })],
        ),
        (
            "a shift on the first rung",
            &[("jitter", |x| *x = json!(0.5))],
        ),
        (
            "a shift that is not finite",
            &[
                ("attempts", |x| *x = json!(1)),
                ("jitter", |x| *x = json!("inf")),
            ],
        ),
        ("more moves than the cap", &[("moves", |x| *x = json!(65))]),
        (
            "a move on a jittered factor",
            &[
                ("attempts", |x| *x = json!(1)),
                ("jitter", |x| *x = json!(1e-12)),
                ("moves", |x| *x = json!(1)),
            ],
        ),
    ];
    for (what, edits) in refused {
        assert!(read(edits).is_err(), "{what}");
    }
    assert!(
        read(&[
            ("attempts", |x| *x = json!(5)),
            ("jitter", |x| *x = json!(1e-3))
        ])
        .is_err()
    );
    let jittered = read(&[
        ("attempts", |x| *x = json!(1)),
        ("jitter", |x| *x = json!(1e-12)),
    ]);
    assert!(
        Robust::restore(&jittered.unwrap()).is_ok(),
        "a jittered factor is read"
    );
}

/// Until a target has a residual scale, Huber down-weights no row: it is
/// least squares to the bit, through the row that gives the scale its
/// first residual (task 177). The target is held at exactly zero for ten
/// rows, so every residual is exactly zero and no scale exists, and then
/// jumps by 1 000, which the old cut of `delta` times the literal 1
/// weighed at 0.0015. From that row the scale exists, and the next jump
/// it judges is cut: the fits part.
#[test]
fn before_a_scale_exists_huber_is_least_squares() {
    let run = |delta: f64| {
        let mut c = cfg(1, 1, RobustLoss::Huber { delta });
        c.min_weight = 2.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 131u64;
        let mut out = Vec::new();
        for i in 0..30 {
            let x = [lcg(&mut s)];
            let y = if i < 10 { 0.0 } else { 1000.0 + x[0] };
            let scaled = m.residual_scale(0).is_some();
            let p = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            out.push((p.pred[0], scaled));
        }
        out
    };
    let (hub, ls) = (run(1.5), run(f64::INFINITY));
    for (i, (_, scaled)) in hub.iter().enumerate() {
        assert_eq!(*scaled, i > 10, "row {i}: a scale from row 11 on");
    }
    for (i, ((h, _), (l, _))) in hub.iter().zip(&ls).enumerate().take(12) {
        assert!(
            h.to_bits() == l.to_bits() || (h.is_nan() && l.is_nan()),
            "row {i}: huber {h} against least squares {l}"
        );
    }
    assert!(hub[10].0 == 0.0 && hub[11].0 > 50.0, "{:?}", &hub[9..12]);
    assert!(
        (hub[12].0 - ls[12].0).abs() > 1.0,
        "row 12: huber {} against least squares {}: the scale cut row 11",
        hub[12].0,
        ls[12].0
    );
}

/// Until a target has a residual scale, a quantile row past the warm-up
/// is least squares, to the bit, through the row that gives the scale
/// its first residual (task 177): a band is a width in units of the
/// scale, and there is none to draw. The target is held at exactly zero
/// for twenty rows, well past the warm-up, so every residual is exactly
/// zero; the band the literal scale of 1 drew took them as band rows
/// aimed `2h(tau - 1/2)` above the target, and the fit left zero. Then
/// the target jumps by 1 000. From that row the scale exists, and the
/// band shapes the rows after it: the fits part.
#[test]
fn before_a_scale_exists_a_quantile_row_is_least_squares() {
    let run = |loss: RobustLoss| {
        let mut c = cfg(1, 1, loss);
        c.min_weight = 2.0;
        c.quantile_eps = 0.2;
        let mut m = Robust::new(c).unwrap();
        let mut s = 137u64;
        let mut out = Vec::new();
        for i in 0..40 {
            let x = [lcg(&mut s)];
            let y = if i < 20 {
                0.0
            } else {
                1000.0 + x[0] + lcg(&mut s)
            };
            let scaled = m.residual_scale(0).is_some();
            let p = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            out.push((p.pred[0], scaled));
        }
        out
    };
    let q = run(RobustLoss::Quantile { tau: 0.9 });
    let ls = run(RobustLoss::Huber {
        delta: f64::INFINITY,
    });
    for (i, ((a, _), (b, _))) in q.iter().zip(&ls).enumerate().take(22) {
        assert!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "row {i}: quantile {a} against least squares {b}"
        );
    }
    for (i, (_, scaled)) in q.iter().enumerate() {
        assert_eq!(*scaled, i > 20, "row {i}: a scale from row 21 on");
    }
    assert!(q[20].0 == 0.0 && q[21].0 > 20.0, "{:?}", &q[19..22]);
    assert!(
        (q[22].0 - ls[22].0).abs() > 1.0,
        "row 22: quantile {} against least squares {}: the band shaped row 21",
        q[22].0,
        ls[22].0
    );
}

/// A row of [`units_rows`]: features, targets, clock step, weight.
type UnitsRow = ([f64; 2], [Option<f64>; 3], f64, f64);

/// The stream [`scaling_the_targets_scales_every_prediction_to_the_bit`]
/// runs: three targets on two features. The first is held at exactly
/// zero for twelve rows, so its residuals are exactly zero until it
/// moves; the second joins on row 3 and misses one row in five; the
/// third carries an outlier one row in seven, for the cut to act on once
/// a scale exists. A row of weight 0 and a gap of five clock units fall
/// before any residual, a gap of 300 takes the weight under
/// `min_weight`, and one of 40 000 (1 333 half-lives) ages every weight
/// to exactly 0.
fn units_rows() -> Vec<UnitsRow> {
    let mut s = 113u64;
    (0..260)
        .map(|i| {
            let x = [lcg(&mut s), 1.0 + lcg(&mut s)];
            let y0 = 0.5 + x[0] - 2.0 * x[1] + 0.2 * lcg(&mut s);
            let y1 = -x[0] + 0.7 * x[1] + 0.3 * lcg(&mut s);
            let kick = if i % 7 == 3 { 25.0 * lcg(&mut s) } else { 0.0 };
            let y2 = 1.0 + 2.0 * x[0] + 0.1 * lcg(&mut s) + kick;
            let ys = [
                match i {
                    _ if i < 12 => Some(0.0),
                    _ if i % 11 == 7 => None,
                    _ => Some(y0),
                },
                (i >= 3 && i % 5 != 2).then_some(y1),
                Some(y2),
            ];
            let d = match i {
                0 => 0.0,
                4 => 5.0,
                150 => 300.0,
                200 => 40_000.0,
                _ => 1.0,
            };
            let w = if i == 2 { 0.0 } else { 1.5 + lcg(&mut s) };
            (x, ys, d, w)
        })
        .collect()
}

/// Scaling every target by `c` scales every prediction by `c`, `σ²` by
/// `c²` and every coefficient by `c`: the fit has no unit of its own.
/// Until a target had a residual scale -- no residual yet, or every one
/// so far exactly zero -- its Huber cut was `delta` times the literal 1,
/// in the target's units, so a target in millions had its first
/// predicted rows down-weighted as outliers and one in millionths none
/// (task 177; `kalman`'s twin is task 172's). Now such a row is
/// down-weighted by nothing. Powers of two keep every operation exact,
/// so the comparison is to the bit, over the warm-up and after
/// ([`units_rows`] says what the stream holds), with an intercept and
/// without, standardized and not.
#[test]
fn scaling_the_targets_scales_every_prediction_to_the_bit() {
    let rows = units_rows();
    // Predictions, the model after the stream, the rows judged before
    // their target had a scale (a prediction, a weight and a residual
    // other than 0, and `σ²` still 0), and the rows the scale cut.
    let run = |base: &RobustCfg, c: f64| {
        let mut m = Robust::new(base.clone()).unwrap();
        let (mut preds, mut unscaled, mut cut) = (Vec::new(), 0, 0);
        for (x, ys, d, w) in &rows {
            let ys: Vec<Option<f64>> = ys.iter().map(|y| y.map(|v| v * c)).collect();
            // The fit's own prediction, the one it learns from, whether
            // or not `min_weight` withholds it (task 278, D10).
            let p = m.slot_predictions(x);
            for (j, y) in ys.iter().enumerate() {
                let Some(y) = *y else { continue };
                let r = y - p[j];
                if !r.is_finite() || *w <= 0.0 || r == 0.0 {
                    continue;
                }
                let RobustLoss::Huber { delta } = m.cfg.loss else {
                    unreachable!()
                };
                if m.sig2[j] == 0.0 {
                    unscaled += 1;
                } else if r.abs() > delta * m.sig2[j].sqrt() {
                    cut += 1;
                }
            }
            preds.push(m.step(x, &ys, *d, *w).pred);
        }
        (preds, m, unscaled, cut)
    };
    let mut cases = Vec::new();
    for intercept in [true, false] {
        for standardize in [false, true] {
            let mut c = cfg(2, 3, RobustLoss::Huber { delta: 1.5 });
            c.decay = Decay::Halflife(30.0);
            c.min_weight = 3.0;
            c.fit_intercept = intercept;
            c.standardize = standardize;
            cases.push((
                format!("intercept {intercept}, standardize {standardize}"),
                c,
            ));
        }
    }
    // The first predictions later in the stream, at another cut.
    let mut c = cfg(2, 3, RobustLoss::Huber { delta: 1.345 });
    c.decay = Decay::Halflife(30.0);
    c.min_weight = 20.0;
    cases.push(("min_weight 20, delta 1.345".into(), c));

    for (name, base) in &cases {
        let (want, m1, unscaled, cut) = run(base, 1.0);
        // Every target is judged once before it has a scale: the first
        // when it leaves zero, the others on their first prediction.
        assert!(unscaled >= 3, "{name}: {unscaled} rows before a scale");
        assert!(cut >= 20, "{name}: the scale cut {cut} rows");
        let predicted = want
            .iter()
            .filter(|p| p.iter().all(|v| v.is_finite()))
            .count();
        assert!(predicted > 200, "{name}: {predicted} rows predicted");
        for c in [2f64.powi(20), 2f64.powi(-20)] {
            let (got, mc, _, _) = run(base, c);
            assert_scaled(name, c, (&want, &m1), (&got, &mc));
        }
    }
}

/// The run on targets scaled by `c` (`got` and its model) is the run on
/// the targets as they are (`want`) scaled by `c`, to the bit: every
/// prediction by `c`, a NaN where `want` has one, and after the stream
/// `σ²` by `c²` and every coefficient by `c`.
fn assert_scaled(
    name: &str,
    c: f64,
    (want, m1): (&[Vec<f64>], &Robust),
    (got, mc): (&[Vec<f64>], &Robust),
) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        for (j, (g, w)) in g.iter().zip(w).enumerate() {
            let scaled = w * c;
            assert!(
                g.to_bits() == scaled.to_bits() || (g.is_nan() && w.is_nan()),
                "{name}, c = {c:e}, row {i}, target {j}: {g} against {scaled} \
                 ({:.2e} apart, relative)",
                ((g - scaled) / scaled).abs()
            );
        }
    }
    let (bc, b1) = (mc.coefficients().unwrap(), m1.coefficients().unwrap());
    for j in 0..m1.cfg.n_targets {
        assert_eq!(
            mc.sig2[j].to_bits(),
            (m1.sig2[j] * c * c).to_bits(),
            "{name}, c = {c:e}, target {j}"
        );
        for (a, b) in bc[j].iter().zip(&b1[j]) {
            assert_eq!(a.to_bits(), (b * c).to_bits(), "{name}, c = {c:e}");
        }
    }
}

/// [`scaling_the_targets_scales_every_prediction_to_the_bit`] for the
/// quantile loss. Its band is `quantile_eps`, or the floor, times `s`,
/// and `s` was the literal 1 until the target had a residual scale, so a
/// band drawn before one was in the target's own units. Such a band is
/// drawn past the warm-up only: on a target's first prediction when
/// `min_weight` holds it back past three rows per coefficient, and while
/// every residual so far is exactly zero (the first target, held at zero
/// here past its warm-up). Now such a row is a least-squares row, as in
/// the warm-up (task 177). At `tau = 0.5` an in-band row is a
/// least-squares row anyway, so only a nudge parted the units there.
#[test]
fn scaling_the_targets_scales_every_quantile_prediction_to_the_bit() {
    let mut rows = units_rows();
    for row in rows.iter_mut().take(24) {
        row.1[0] = Some(0.0);
    }
    // Predictions, the model after the stream, the rows before a scale
    // that the literal 1 would have decided otherwise (a band row, or a
    // nudge, in place of a least-squares row), and the rows after one
    // that the band shaped (a band row or a nudge).
    let run = |base: &RobustCfg, c: f64| {
        let mut m = Robust::new(base.clone()).unwrap();
        let (mut preds, mut decided, mut banded) = (Vec::new(), 0, 0);
        for (x, ys, d, w) in &rows {
            let ys: Vec<Option<f64>> = ys.iter().map(|y| y.map(|v| v * c)).collect();
            // The fit's own prediction, the one it learns from, whether
            // or not `min_weight` withholds it (task 278, D10).
            let p = m.slot_predictions(x);
            for (j, y) in ys.iter().enumerate() {
                let Some(y) = *y else { continue };
                if !p[j].is_finite() || *w <= 0.0 {
                    continue;
                }
                let row = (p[j], y, *d, *w);
                let scale = m.residual_scale(j);
                let update = update_of(&m, j, row, scale);
                if scale.is_none() {
                    decided += usize::from(update_of(&m, j, row, Some(1.0)) != update);
                } else {
                    banded += usize::from(update != RowUpdate::Fit { w: *w, target: y });
                }
            }
            preds.push(m.step(x, &ys, *d, *w).pred);
        }
        (preds, m, decided, banded)
    };
    let quantile = |tau: f64, min_weight: f64, intercept: bool, standardize: bool| {
        let mut c = cfg(2, 3, RobustLoss::Quantile { tau });
        c.decay = Decay::Halflife(30.0);
        c.quantile_eps = 0.2;
        c.min_weight = min_weight;
        c.fit_intercept = intercept;
        c.standardize = standardize;
        c
    };
    let mut ridge_free = quantile(0.25, 20.0, false, false);
    ridge_free.ridge = 0.0;
    let cases = [
        ("tau 0.9, min_weight 3", quantile(0.9, 3.0, true, false)),
        (
            "tau 0.5, min_weight 20, standardized",
            quantile(0.5, 20.0, true, true),
        ),
        ("tau 0.25, through the origin, ridge 0", ridge_free),
        (
            "tau 0.75, through the origin, standardized",
            quantile(0.75, 3.0, false, true),
        ),
    ];
    for (name, base) in &cases {
        let (want, m1, _, banded) = run(base, 1.0);
        assert!(banded >= 100, "{name}: the band shaped {banded} rows");
        let predicted = want
            .iter()
            .filter(|p| p.iter().all(|v| v.is_finite()))
            .count();
        assert!(predicted > 200, "{name}: {predicted} rows predicted");
        let mut decided = 0;
        for c in [2f64.powi(20), 2f64.powi(-20)] {
            let (got, mc, d, _) = run(base, c);
            assert_scaled(name, c, (&want, &m1), (&got, &mc));
            decided += d;
        }
        // At `2^20` a residual before a scale is far outside the band
        // the literal drew, so a row past the warm-up was a nudge.
        assert!(decided >= 1, "{name}: the literal decided {decided} rows");
    }
}

// ---- data shares (docs/PLAN.md task 116, B) ----

/// With `huber_delta` past every residual no row is down-weighted, so
/// the band Gram is the EW Gram of the rows: `support_coef_j` is
/// `1 − λ ((Σ + λI)⁻¹)_jj`, `Σ` the rows' centred covariance (in
/// correlation form under `standardize`), here summed from the rows
/// with no decay and inverted by `crate::oracle`'s LU.
#[test]
fn the_data_shares_are_the_shrinkage_diagonal() {
    for standardize in [false, true] {
        let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1e9 });
        c.ridge = 0.4;
        c.standardize = standardize;
        let mut m = Robust::new(c).unwrap();
        let mut s = 7u64;
        let mut rows = Vec::new();
        for i in 0..200 {
            let a = 1.5 * lcg(&mut s);
            let x = [a, 0.6 * lcg(&mut s) + 0.5 * a];
            let y = 1.0 + x[0] - x[1] + 0.3 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            rows.push(x);
        }
        let n = rows.len() as f64;
        let mean = [0, 1].map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / n);
        let mut a = [0.0; 4];
        for r in &rows {
            for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                a[i * 2 + j] += (r[i] - mean[i]) * (r[j] - mean[j]) / n;
            }
        }
        if standardize {
            let sd = [a[0].sqrt(), a[3].sqrt()];
            for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                a[i * 2 + j] /= sd[i] * sd[j];
            }
        }
        a[0] += 0.4;
        a[3] += 0.4;
        let inv = crate::oracle::inverse(&a);
        let got = m.support_coef().unwrap();
        assert!(got[0][0].is_nan(), "the intercept is not a share");
        // Task 274: a share moves by `0.4·Δ(A⁻¹)ⱼⱼ`, and `(A⁻¹)ⱼⱼ` by
        // `cⱼᵀ·ΔA·cⱼ` (`cⱼ` its column) for every error `ΔA` of the system
        // on either side: the model's co-moments, carried along its
        // weighted Welford run ([`crate::ewcov::rounding::Rounding`]); the
        // test's, two-pass, each term three roundings and each addition
        // `u` of its partial sum, the means' errors second order; under
        // `standardize` each side's correlation `C_ij/√(C_ii C_jj)`, its
        // inputs' errors through `ΔC_ij/(sᵢsⱼ) + ½|R_ij|(ΔC_ii/C_ii +
        // ΔC_jj/C_jj)` and three roundings; the ridge's addition; the
        // model's `diag(A⁻¹)`, a Cholesky solve per column (its backward
        // error); and the oracle's inverse, by its residual. It was
        // `1e-9`.
        // Measured on aarch64 (macOS): at most 0.0071 (plain) and 0.0020 (standardized) of
        // the bound: two hundred rows with no decay, the moments' worst-case running bound
        // linear in their count.
        use crate::oracle::{self, U, abs_form};
        let mut tr = crate::ewcov::rounding::Rounding::new(2, 0.0);
        for r in &rows {
            tr.update(r, 1.0, 1.0);
        }
        let mut et = [0.0f64; 4];
        let mut acc = [0.0f64; 4];
        for r in &rows {
            for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let t = (r[i] - mean[i]) * (r[j] - mean[j]) / n;
                acc[i * 2 + j] += t;
                et[i * 2 + j] += 3.0 * U * t.abs() + U * acc[i * 2 + j].abs();
            }
        }
        let cmat = acc;
        let to_system = |e: &[f64]| -> Vec<f64> {
            if !standardize {
                return e.to_vec();
            }
            let sd = [cmat[0].sqrt(), cmat[3].sqrt()];
            (0..4)
                .map(|ij| {
                    let (i, j) = (ij / 2, ij % 2);
                    let r = cmat[ij] / (sd[i] * sd[j]);
                    e[ij] / (sd[i] * sd[j])
                        + 0.5 * r.abs() * (e[i * 3] / cmat[i * 3] + e[j * 3] / cmat[j * 3])
                        + 3.0 * U * r.abs()
                })
                .collect()
        };
        let (dl, dt) = (to_system(&tr.ec), to_system(&et));
        let e = oracle::cholesky_backward(&a);
        let own = oracle::inverse_error(&a);
        for j in 0..2 {
            let want = 1.0 - 0.4 * inv[j * 2 + j];
            let cj = [inv[j], inv[2 + j]];
            let da: Vec<f64> = (0..4)
                .map(|ij| {
                    dl[ij]
                        + dt[ij]
                        + e[ij]
                        + if ij % 3 == 0 {
                            2.0 * U * a[ij].abs()
                        } else {
                            0.0
                        }
                })
                .collect();
            let dinv = abs_form(&cj, &da, &cj) + own[j * 2 + j];
            let bound = 0.4 * dinv + 2.0 * U * (want.abs() + 0.4 * inv[j * 2 + j].abs());
            oracle::assert_within(
                got[0][j + 1],
                want,
                bound,
                format_args!("standardize {standardize}, share {j}"),
            );
        }
    }
}

/// A duplicated pair splits its coefficient evenly in the band Gram
/// either loss keeps, through the origin as well; a target no solve
/// has fit has no shares; and the shares survive a state round trip.
#[test]
fn a_duplicated_pair_reads_half_under_either_loss() {
    for loss in [
        RobustLoss::Huber { delta: 1.345 },
        RobustLoss::Quantile { tau: 0.5 },
    ] {
        for fit_intercept in [true, false] {
            let mut c = cfg(3, 2, loss);
            c.fit_intercept = fit_intercept;
            let mut m = Robust::new(c).unwrap();
            assert!(m.support_coef().is_none(), "before the first solve");
            let mut s = 11u64;
            for i in 0..400 {
                let a = lcg(&mut s);
                let x = [a, lcg(&mut s), a];
                let y = 1.0 + 2.0 * a - x[1] + 0.3 * lcg(&mut s);
                m.step(&x, &[Some(y), None], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            let got = m.support_coef().unwrap();
            let off = usize::from(fit_intercept);
            let shares = &got[0][off..];
            assert!(
                (shares[0] - 0.5).abs() < 1e-3
                    && (shares[1] - 1.0).abs() < 1e-3
                    && (shares[2] - 0.5).abs() < 1e-3,
                "{loss:?}, intercept {fit_intercept}: {:?}",
                got[0]
            );
            assert!(got[1].iter().all(|v| v.is_nan()), "no row, no fit");
            let back = Robust::restore(&m.state()).unwrap();
            assert_eq!(back, m);
            let bytes = rmp_serde::to_vec(&m.state()).unwrap();
            let again: crate::State = rmp_serde::from_slice(&bytes).unwrap();
            // Compared by their bits: the intercept's share is NaN.
            assert_eq!(Robust::restore(&again).unwrap(), m);
        }
    }
}

/// The shares wait for a read (`PendingShares`): read now, read after
/// the stream settles them, or saved, they are the same bits, and a
/// settled model holds no factor for them.
#[test]
fn the_shares_are_the_same_whenever_they_are_taken() {
    for loss in [
        RobustLoss::Huber { delta: 1.345 },
        RobustLoss::Quantile { tau: 0.3 },
    ] {
        let mut m = Robust::new(cfg(3, 1, loss)).unwrap();
        let mut s = 23u64;
        for i in 0..150 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y = 1.0 + x[0] - x[2] + 0.3 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert_eq!(m.pending_readiness(), 1, "{loss:?}: the last solve waits");
        let unread = m.clone();
        let read = m.support_coef().unwrap();
        m.settle_readiness();
        assert_eq!(m.pending_readiness(), 0);
        let settled = m.support_coef().unwrap();
        let bits = |v: &[Vec<f64>]| v.concat().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&read), bits(&settled), "{loss:?}");
        assert_eq!(unread, m, "equal on the values a read gives");
        let saved = rmp_serde::to_vec(&unread.state()).unwrap();
        assert_eq!(saved, rmp_serde::to_vec(&m.state()).unwrap());
        // The shares are a ridge's: below 1, above 0.
        assert!(
            settled[0][1..].iter().all(|v| *v > 0.9 && *v < 1.0),
            "{settled:?}"
        );
    }
}

/// A stream of `rows` rows whose first and third features are one
/// column twice, the second a column of its own; target 1 null
/// throughout when `second` is false.
fn duplicated(m: &mut Robust, rows: usize, second: bool) {
    let mut s = 11u64;
    for i in 0..rows {
        let a = lcg(&mut s);
        let x = [a, lcg(&mut s), a];
        let y = 1.0 + 2.0 * a - x[1] + 0.3 * lcg(&mut s);
        let y1 = second.then(|| 0.5 - a + 0.3 * lcg(&mut s));
        let ys: Vec<Option<f64>> = [Some(y), y1][..m.cfg.n_targets].to_vec();
        m.step(&x, &ys, if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
}

/// With no ridge, a feature the rows never move leaves a row and a
/// column of 0 in the band Gram, whose factor then takes the jitter
/// ladder's shift `δ`: the shares are the shifted system's,
/// `1 − (ridge + δ) (A⁻¹)_jj` with `A` carrying `δ`
/// (`Robust::pending_shares`), so the still feature reads 0, the data
/// carrying nothing of it, and the others all but 1, under either loss
/// (docs/PLAN.md task 220). Read with the shift taken off, the still
/// feature's share was 2, clamped to 1.
#[test]
fn a_feature_the_rows_never_move_reads_no_share_through_the_jitter() {
    for loss in [
        RobustLoss::Huber { delta: 1.345 },
        RobustLoss::Quantile { tau: 0.5 },
    ] {
        let mut c = cfg(3, 1, loss);
        c.ridge = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 17u64;
        for i in 0..200 {
            let x = [lcg(&mut s), 0.25, lcg(&mut s)];
            let y = 1.0 + 2.0 * x[0] - x[2] + 0.3 * lcg(&mut s);
            let before = m.solve_failures;
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if m.solve_failures == before || i < 10 {
                continue;
            }
            let got = m.support_coef().unwrap();
            let shares = &got[0][1..];
            assert!(
                shares[1].abs() < 1e-9
                    && [shares[0], shares[2]]
                        .iter()
                        .all(|v| *v > 0.99 && *v <= 1.0),
                "{loss:?}, row {i}: {:?}",
                got[0]
            );
        }
        assert!(m.solve_failures > 100, "{loss:?}: {}", m.solve_failures);
    }
}

/// A target whose solve fails keeps its fit and the shares of that fit
/// (`Robust::solve`): stored, as the stream leaves them at the end of a
/// run, and then a solve that fails at every jitter, the accumulator
/// handed a correlation of 2 as `a_standardized_solve_that_fails_outright_is_counted`
/// hands it. The shares keep their bits (docs/PLAN.md task 220).
#[test]
fn a_solve_that_fails_keeps_the_shares_of_the_fit_it_keeps() {
    let mut c = cfg(2, 1, RobustLoss::Huber { delta: 1.5 });
    c.standardize = true;
    let mut m = Robust::new(c).unwrap();
    let mut s = 103u64;
    for i in 0..40 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0] - x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    m.settle_readiness();
    let (beta, shares) = (
        m.coefficients().unwrap().to_vec(),
        m.support_coef().unwrap(),
    );
    assert!(
        shares[0][1..].iter().all(|v| *v > 0.0 && *v < 1.0),
        "{shares:?}"
    );
    let before = m.solve_failures;
    let (w, q) = (m.cov[0].n_eff(), m.cov[0].q_sum());
    m.cov[0].set_moments(
        &[1.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
        w,
        q,
    );
    m.solve();
    assert_eq!(m.solve_failures, before + 1, "every jitter failed");
    assert_eq!(m.coefficients().unwrap(), beta, "the fit is kept");
    let bits = |v: &[Vec<f64>]| v.concat().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&m.support_coef().unwrap()), bits(&shares));
}

/// A target no solve has fit has no shares, NaN, whatever a file held
/// for it: one edited to hold numbers there loads, its shape being
/// right, and the next solve, which skips the target, clears them
/// (`Shares::unfit`; docs/PLAN.md task 220). The fitted target's are
/// the solve's.
#[test]
fn a_target_no_solve_has_fit_holds_no_shares_whatever_a_file_held() {
    let mut m = Robust::new(cfg(3, 2, RobustLoss::Huber { delta: 1.345 })).unwrap();
    duplicated(&mut m, 60, false);
    m.settle_readiness();
    let mut s = m.state();
    let ModelState::Robust(inner) = &mut s.model else {
        unreachable!()
    };
    assert!(inner.support.stored[1].iter().all(|v| v.is_nan()));
    inner.support.stored[1] = vec![0.5; 4];
    let mut back = Robust::restore(&s).unwrap();
    back.step(&[0.1, 0.2, 0.1], &[Some(1.0), None], 1.0, 1.0);
    let got = back.support_coef().unwrap();
    assert!(got[1].iter().all(|v| v.is_nan()), "{got:?}");
    assert!(got[0][1..].iter().all(|v| v.is_finite()), "{got:?}");
}

/// Two models are equal only where their shares are, by their bits:
/// one share a rounding step apart makes them two (`Shares`'s
/// `PartialEq`; docs/PLAN.md task 220).
#[test]
fn models_whose_shares_differ_are_not_equal() {
    let mut m = Robust::new(cfg(3, 1, RobustLoss::Huber { delta: 1.345 })).unwrap();
    duplicated(&mut m, 60, false);
    m.settle_readiness();
    assert_eq!(m, m.clone());
    let mut other = m.clone();
    other.support.stored[0][2] = other.support.stored[0][2].next_up();
    assert_ne!(m, other);
}

/// The data shares a state carries are a solve's shape, each break
/// refused alone (docs/PLAN.md task 220): after a solve, a target's
/// shares a value short, and one target too many with each row of the
/// right width; before the first solve, any.
#[test]
fn a_state_whose_shares_have_the_wrong_shape_is_refused() {
    let fresh = Robust::new(cfg(3, 2, RobustLoss::Huber { delta: 1.345 })).unwrap();
    let mut m = fresh.clone();
    duplicated(&mut m, 60, true);
    m.settle_readiness();
    type Spoil = (&'static str, fn(&mut Shares));
    let cases: [Spoil; 2] = [
        ("a target's shares a value short", |s| {
            s.stored[1].pop();
        }),
        ("a target too many", |s| s.stored.push(vec![0.5; 4])),
    ];
    for (what, spoil) in cases {
        for base in [&fresh, &m] {
            let mut s = base.state();
            let ModelState::Robust(inner) = &mut s.model else {
                unreachable!()
            };
            if inner.support.stored.is_empty() {
                inner.support.stored = vec![vec![0.5; 4]; 2];
            }
            spoil(&mut inner.support);
            match Robust::restore(&s) {
                Err(StateError::Invalid(e)) => {
                    assert!(e.contains("data shares"), "{what}: {e}");
                }
                other => panic!("{what}: {other:?}"),
            }
        }
    }
    assert!(Robust::restore(&fresh.state()).is_ok() && Robust::restore(&m.state()).is_ok());
}

/// Task 252: through the origin and standardized, features that are
/// always 0 have no raw second moment, so the band system keeps no
/// column: no factorization, every coefficient 0, nothing counted as a
/// failure -- in the solve and in the nudges of the rows outside the
/// band alike, whose leverage is then 0.
#[test]
fn features_with_no_second_moment_keep_no_column_through_the_origin() {
    for loss in [
        RobustLoss::Huber { delta: 1.0 },
        RobustLoss::Quantile { tau: 0.5 },
    ] {
        let mut c = cfg(2, 1, loss);
        c.fit_intercept = false;
        c.standardize = true;
        c.min_weight = 0.0;
        let mut m = Robust::new(c).unwrap();
        let mut s = 21u64;
        for i in 0..40 {
            let y = 1.0 + 3.0 * lcg(&mut s);
            let out = m.step(&[0.0, 0.0], &[Some(y)], f64::from(u8::from(i > 0)), 1.0);
            // From the model's floor, a row per coefficient, on.
            if i > 1 {
                assert_eq!(out.pred[0], 0.0, "{loss:?}, row {i}");
            }
        }
        assert_eq!(m.coefficients().unwrap()[0], vec![0.0, 0.0], "{loss:?}");
        assert_eq!(m.solve_failures, 0, "{loss:?}");
    }
}

/// Task 252: with an intercept and standardized, a feature that never
/// moves has no usable variance and is dropped, so a quantile nudge
/// from a row outside the band reads a band system of no column: its
/// leverage is the intercept's alone, and it is taken -- the target's
/// mean moves.
#[test]
fn a_nudge_against_a_band_system_of_no_column_reads_the_intercepts_leverage() {
    let mut c = cfg(1, 1, RobustLoss::Quantile { tau: 0.5 });
    c.standardize = true;
    c.min_weight = 0.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 22u64;
    for i in 0..30 {
        let y = 2.0 + 0.1 * lcg(&mut s);
        m.step(&[1.0], &[Some(y)], f64::from(u8::from(i > 0)), 1.0);
    }
    m.systems.0[0] = None;
    let ybar = m.ybar[0];
    m.step(&[1.0], &[Some(50.0)], 1.0, 1.0);
    assert!(
        m.ybar[0] > ybar,
        "the nudge was taken: {} against {ybar}",
        m.ybar[0]
    );
    if let Some(sys) = m.systems.0[0].as_ref() {
        assert!(sys.keep.is_empty() && sys.factor.is_none());
    }
}

/// Task 252: a quantile nudge from a row that deviates on a feature
/// with no spread among the rows in the band has no curvature to lean
/// on: its movement is unbounded and its step 0, so neither the
/// target's mean nor its cross-moments move.
#[test]
fn a_nudge_along_a_feature_with_no_spread_takes_no_step() {
    let mut c = cfg(2, 1, RobustLoss::Quantile { tau: 0.5 });
    c.min_weight = 0.0;
    let mut m = Robust::new(c).unwrap();
    let mut s = 9u64;
    for i in 0..40 {
        let x1 = lcg(&mut s);
        let y = 2.0 * x1 + 0.01 * lcg(&mut s);
        m.step(&[1.0, x1], &[Some(y)], f64::from(u8::from(i > 0)), 1.0);
    }
    let (ybar, lo, cross) = (m.ybar[0], m.ybar_lo[0], m.cross[0].clone());
    let out = m.step(&[2.0, 0.5], &[Some(100.0)], 1.0, 1.0);
    assert!(out.pred[0].is_finite(), "the row was scored");
    assert_eq!(
        (m.ybar[0].to_bits(), m.ybar_lo[0].to_bits()),
        (ybar.to_bits(), lo.to_bits())
    );
    assert_eq!(m.cross[0], cross);
}

/// A late target's own first solve reads whether a solve has fit *it*:
/// under feature sets each target has one slot per set, target-major,
/// and the one read is its own weighing set's. With two targets and two
/// sets the late target's slot is the fourth; the first target's
/// weighing slot, the second, was fit at row 1, and read in its place
/// the late target would wait for the row cap, 25 rows after it, null
/// meanwhile (docs/PLAN.md task 272 (m)).
#[test]
fn a_late_targets_first_solve_reads_its_own_slot_under_feature_sets() {
    // The late target's: at the end of its second row, row 11, its
    // weight's `min_weight` under the gate that held learning back (a
    // state written before schema 64), and since at its third, row 12,
    // the model's floor, which reads its own row count alone (task 278,
    // D10).
    // The first target's first solve: row 1 under that gate, its
    // `min_weight` of 2, and since row 2, a row per coefficient of three.
    for (holds, global, first) in [(true, 1, 11), (false, 2, 12)] {
        let mut c = cfg(2, 2, RobustLoss::Huber { delta: 1.5 });
        c.min_weight = 2.0;
        c.solve_every = f64::INFINITY;
        c.max_rows_between_solves = 25;
        c.feature_sets = vec![("a".into(), vec![0]), ("all".into(), vec![0, 1])];
        c.min_weight_holds_learning = holds;
        let mut m = Robust::new(c).unwrap();
        assert_eq!((m.cfg.n_sets(), m.cfg.weighing_set()), (2, 1));
        let mut s = 4u64;
        for i in 0..20 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let late = (i >= 10).then(|| 1.0 - 2.0 * x[0] + x[1] + 0.01 * lcg(&mut s));
            let y0 = 0.5 + x[0] - x[1] + 0.01 * lcg(&mut s);
            m.step(&x, &[Some(y0), late], f64::from(u8::from(i > 0)), 1.0);
            if i >= global {
                let beta = m.coefficients().expect("solved");
                assert!(beta[..2].iter().flatten().all(|v| v.is_finite()), "row {i}");
                for slot in [2, 3] {
                    assert_eq!(
                        beta[slot].iter().all(|v| v.is_finite()),
                        i >= first,
                        "holds {holds}, row {i}, slot {slot}: {beta:?}"
                    );
                }
            }
        }
    }
}

/// A target that the first solve fit -- present on the rows before it,
/// below its own threshold -- has no first solve of its own when its
/// weight later reaches that threshold: a solve has fit it, and the
/// next falls on the cadence, here the row cap 25 rows on, so the
/// coefficients hold until then (docs/PLAN.md task 195, S9b, and 272
/// (m)).
#[test]
fn a_target_the_first_solve_fit_waits_for_the_cadence_at_its_own_threshold() {
    let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.5 });
    c.min_weight = 2.0;
    c.solve_every = f64::INFINITY;
    c.max_rows_between_solves = 25;
    let mut m = Robust::new(c).unwrap();
    m.set_target_min_weight(vec![2.0, 5.0]).unwrap();
    let mut s = 6u64;
    let mut first = None;
    for i in 0..24 {
        let x = [lcg(&mut s)];
        let y = [
            Some(0.5 + x[0] + 0.01 * lcg(&mut s)),
            Some(1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s)),
        ];
        m.step(&x, &y, f64::from(u8::from(i > 0)), 1.0);
        if i == 1 {
            let beta = m.coefficients().expect("the first solve, at row 1");
            assert!(beta.iter().flatten().all(|v| v.is_finite()), "both fit");
            assert!(
                m.wobs[1] < 5.0,
                "the second target is below its own threshold"
            );
            first = Some(beta.to_vec());
        }
        if i > 1 {
            assert_eq!(m.coefficients(), first.as_deref(), "row {i}");
        }
    }
    assert!(
        m.wobs[1] >= 5.0,
        "the second target's weight passed its threshold"
    );
}

/// A target whose weight its decay has taken to exactly 0 has forgotten
/// its rows, as every model forgets what a decay that underflows takes
/// (`model_contract.rs`): it predicts nothing, though the fit the last
/// solve made for it is kept for when its rows return, while a target
/// whose history the same gap took starts over and goes on once it has
/// a row per coefficient again (docs/PLAN.md task 115 (c), 272 (m) and
/// 278, D10).
#[test]
fn a_target_whose_weight_has_decayed_to_nothing_predicts_nothing() {
    let mut c = cfg(1, 2, RobustLoss::Huber { delta: 1.5 });
    c.decay = Decay::Halflife(1.0);
    c.min_weight = 1.0;
    c.solve_every = f64::INFINITY;
    c.max_rows_between_solves = 1000;
    let mut m = Robust::new(c).unwrap();
    let mut s = 9u64;
    for i in 0..30 {
        let x = [lcg(&mut s)];
        let y = [
            Some(0.5 + x[0]),
            Some(1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s)),
        ];
        m.step(&x, &y, f64::from(u8::from(i > 0)), 1.0);
    }
    // A gap of 2,000 halflives, on a row the second target is absent
    // from: its weight decays to 0, the first's starts over.
    m.step(&[0.3], &[Some(0.8), None], 2000.0, 1.0);
    assert_eq!(m.wj[1], 0.0);
    assert!(m.wj[0] > 0.0 && m.w_raw >= 1.0);
    let beta = m.coefficients().unwrap();
    assert!(beta[1].iter().all(|v| v.is_finite()), "the kept fit");
    // The first target forgot its rows too, and starts over as a fresh
    // one does: its floor, a row per coefficient, counts again, so its
    // next row is withheld and solves it (task 278, D10).
    let p = m.predict(&[0.3], 0.0);
    assert!(p.pred.iter().all(|v| v.is_nan()), "{:?}", p.pred);
    m.step(&[0.6], &[Some(1.1), None], 1.0, 1.0);
    let p = m.predict(&[0.3], 0.0);
    assert!(p.pred[0].is_finite(), "{:?}", p.pred);
    assert!(p.pred[1].is_nan(), "{:?}", p.pred);
}
