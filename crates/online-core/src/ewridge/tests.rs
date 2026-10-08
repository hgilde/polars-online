//! `ewridge`'s tests, kept in a file of their own: with the tests review
//! 2026-10-05 (task 160) added, `ewridge.rs` passed the repository's 250 KB
//! cap for a source file (`tests/test_repo_hygiene.py`), as `marginal.rs`
//! did at task 158.

use super::*;
use crate::oracle;

fn cfg(k: usize, m: usize) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: k,
        n_targets: m,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        ridge: vec![1e-8],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: (k + 1) as f64,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// With `ridge_scale` the prior sits on the sum scale, so it starts the
/// fit and then fades: the usual warm start.
#[test]
fn coef0_with_ridge_decay_warms_the_start_then_fades() {
    let mut c = cfg(2, 1);
    c.ridge = vec![10.0];
    c.ridge_scale = true;
    c.standardize = false;
    c.coef_prior = Some(vec![vec![0.0, 5.0, -5.0]]);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();

    // With no target seen yet the fit is essentially the prior. (Not
    // exactly: the feature row itself has already entered S, which pulls a
    // little even with no target.)
    let mut s = 71u64;
    let x0 = [lcg(&mut s), lcg(&mut s)];
    m.step(&x0, &[None], 0.0, 1.0);
    let early = m.coefficients().unwrap()[0].clone();
    assert!(
        (early[1] - 5.0).abs() < 0.5 && (early[2] + 5.0).abs() < 0.5,
        "cold start should sit at the prior: {early:?}"
    );

    // With enough contradicting evidence it moves to the truth.
    for i in 0..20000 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 1.5 * x[0] - 0.5 * x[1];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let late = m.coefficients().unwrap()[0].clone();
    assert!(
        (late[1] - 1.5).abs() < 0.05,
        "prior did not wash out: {late:?}"
    );
    assert!((late[2] + 0.5).abs() < 0.05);
}

/// Without `ridge_scale` the pull is permanent, because `S` is a weighted
/// mean and never outgrows a fixed `ridge`. Worth pinning: it is the
/// opposite of the usual "the prior washes out" intuition.
#[test]
fn coef0_without_ridge_decay_pulls_forever() {
    let mut c = cfg(1, 1);
    c.ridge = vec![10.0];
    c.coef_prior = Some(vec![vec![0.0, 5.0]]);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 72u64;
    for i in 0..50000 {
        let x = [lcg(&mut s)];
        m.step(&x, &[Some(1.5 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let b = m.coefficients().unwrap()[0][1];
    assert!(
        b > 3.0,
        "a fixed ridge should keep pulling toward the prior forever, got {b}"
    );
}

#[test]
fn coef0_shrinks_toward_the_prior_not_zero() {
    // Same ridge, same data, different priors => the fits differ, and each
    // sits between the data's answer and its own prior.
    let run = |prior: Option<Vec<Vec<f64>>>| {
        let mut c = cfg(1, 1);
        c.ridge = vec![50.0];
        c.coef_prior = prior;
        c.min_weight = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 73u64;
        for i in 0..200 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients().unwrap()[0][1]
    };
    let toward_zero = run(None);
    let toward_ten = run(Some(vec![vec![0.0, 10.0]]));
    assert!(
        toward_zero < 2.0,
        "no prior should shrink toward 0: {toward_zero}"
    );
    assert!(
        toward_ten > 2.0,
        "a prior of 10 should pull up: {toward_ten}"
    );
}

#[test]
fn coef0_works_with_standardization() {
    // The prior is stated in original units; on badly scaled features the
    // standardized path must still honour it.
    let mut c = cfg(1, 1);
    c.ridge = vec![1e6];
    c.standardize = true;
    c.coef_prior = Some(vec![vec![0.0, 0.02]]);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 79u64;
    for i in 0..500 {
        let x = [100.0 * lcg(&mut s)];
        let y = 0.05 * x[0];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    // An overwhelming ridge pins the fit at the prior, in original units.
    let b = m.coefficients().unwrap()[0][1];
    assert!(
        (b - 0.02).abs() < 5e-3,
        "expected ~0.02 in original units, got {b}"
    );
}

/// A session break should be able to revert partway toward the long run,
/// rather than only choosing between "carry on" and "start over".
#[test]
fn session_shrink_reverts_toward_the_long_run() {
    // Long run: slope 1. Today: slope -1 for a while. After a session
    // break with shrink f, the fit should sit between the two.
    let build = |f: Option<f64>| {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(50.0);
        c.session_shrink = f;
        c.long_half_life = f.map(|_| 100_000.0);
        c.min_weight = 0.0;
        EwRidge::new(c).unwrap()
    };
    let run = |m: &mut EwRidge| {
        let mut s = 91u64;
        // a long history at slope +1
        for i in 0..4000 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        // then a shorter stretch at slope -1
        for _ in 0..300 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(-x[0])], 1.0, 1.0);
        }
        m.coefficients().unwrap()[0][1]
    };

    let mut plain = build(None);
    let before_plain = run(&mut plain);
    let mut shrunk = build(Some(0.9));
    let before_shrunk = run(&mut shrunk);
    // Both have been dragged to roughly -1 by the recent regime.
    assert!(before_plain < -0.5 && before_shrunk < -0.5);

    // The session boundary reverts the shrinking one toward +1.
    shrunk.blend_toward_long_run();
    plain.blend_toward_long_run(); // no twin configured: a no-op
    let after_shrunk = shrunk.coefficients_after_blend();
    let after_plain = plain.coefficients_after_blend();

    assert!(
        (after_plain - before_plain).abs() < 1e-9,
        "no twin configured should mean no change: {before_plain} -> {after_plain}"
    );
    assert!(
        after_shrunk > before_shrunk + 0.5,
        "shrink should pull back toward the long run: {before_shrunk} -> {after_shrunk}"
    );
}

/// Task 145: `session_shrink = f` fits on `1 − f` of today's rows and `f`
/// of the long run's, at today's weight. Worked from the rows alone: each
/// accumulator weighs a row by `λ^age`; the two kernels, each normalised,
/// mix `(1 − f, f)`; and the ridge solve on those weighted moments is
/// `faer`'s, sharing nothing with the blend or the model's Cholesky. The
/// weight is today's across the blend, bit for bit.
#[test]
fn a_blend_fits_a_share_of_the_long_run() {
    let (h_fast, h_slow, n, ridge) = (40.0, 400.0, 600usize, 1e-3);
    for f in [0.25, 0.5, 1.0] {
        let mut c = cfg(2, 1);
        c.decay = Decay::Halflife(h_fast);
        c.ridge = vec![ridge];
        c.session_shrink = Some(f);
        c.long_half_life = Some(h_slow);
        c.min_weight = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 11u64;
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let x = [lcg(&mut s), lcg(&mut s) + 3.0];
            let slope = if i < n / 2 { 1.0 } else { -1.0 };
            let y = 0.5 + slope * x[0] - x[1] + 0.1 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            rows.push((x, y));
        }
        let weight = m.n_eff();
        m.blend_toward_long_run();
        assert_eq!(
            m.n_eff().to_bits(),
            weight.to_bits(),
            "f = {f}: the weight moved"
        );

        let kernel = |h: f64| -> Vec<f64> {
            (0..n)
                .map(|i| 0.5f64.powf((n - 1 - i) as f64 / h))
                .collect()
        };
        let (kf, ks) = (kernel(h_fast), kernel(h_slow));
        let (sf, ss): (f64, f64) = (kf.iter().sum(), ks.iter().sum());
        let a: Vec<f64> = (0..n)
            .map(|i| (1.0 - f) * kf[i] / sf + f * ks[i] / ss)
            .collect();
        let mean = |g: &dyn Fn(usize) -> f64| (0..n).map(|i| a[i] * g(i)).sum::<f64>();
        let mx = [mean(&|i| rows[i].0[0]), mean(&|i| rows[i].0[1])];
        let my = mean(&|i| rows[i].1);
        let cov = |p: usize, q: usize| mean(&|i| (rows[i].0[p] - mx[p]) * (rows[i].0[q] - mx[q]));
        let cxy = |p: usize| mean(&|i| (rows[i].0[p] - mx[p]) * (rows[i].1 - my));
        let b = oracle::solve(
            &[cov(0, 0) + ridge, cov(0, 1), cov(1, 0), cov(1, 1) + ridge],
            &[cxy(0), cxy(1)],
        );
        let want = [my - mx[0] * b[0] - mx[1] * b[1], b[0], b[1]];
        let got = &m.coefficients().unwrap()[0];
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
                "f = {f}: {got:?} vs {want:?}"
            );
        }
    }
}

/// E45: the target moments must survive a blend as a mixture, not go
/// stale, and `f = 0` (or a twin identical to the model) must leave them
/// exactly where they were -- the invariant the means, co-moments and
/// weights already hold.
#[test]
fn a_blend_mixes_the_target_moments() {
    let build = |f: f64| {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(50.0);
        c.session_shrink = Some(f);
        c.long_half_life = Some(50.0); // the same half_life: an identical twin
        c.min_weight = 0.0;
        EwRidge::new(c).unwrap()
    };
    let run = |m: &mut EwRidge| {
        let mut s = 7u64;
        for i in 0..500 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(3.0 + 2.0 * x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
    };
    let mut twin = build(0.5);
    run(&mut twin);
    let (m0, v0, q0) = {
        let tm = twin.target_moments().unwrap();
        (tm.means()[0], tm.vars()[0], tm.q()[0])
    };
    twin.blend_toward_long_run();
    let tm = twin.target_moments().unwrap();
    assert!(
        (tm.means()[0] - m0).abs() < 1e-12,
        "{} vs {m0}",
        tm.means()[0]
    );
    assert!((tm.vars()[0] - v0).abs() < 1e-9, "{} vs {v0}", tm.vars()[0]);
    assert!((tm.q()[0] - q0).abs() < 1e-9, "{} vs {q0}", tm.q()[0]);

    // A genuinely slower twin moves them toward the long run. The level
    // sits at 0 for a long stretch and jumps to 10 for a short one, so
    // the fast window's mean is ~10 and the twin's is much lower.
    let mut slow = {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(20.0);
        c.session_shrink = Some(0.9);
        c.long_half_life = Some(5000.0);
        c.min_weight = 0.0;
        EwRidge::new(c).unwrap()
    };
    let mut s = 11u64;
    for i in 0..2000 {
        let x = [lcg(&mut s)];
        let level = if i < 1900 { 0.0 } else { 10.0 };
        slow.step(
            &x,
            &[Some(level + x[0])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let before = slow.target_moments().unwrap().means()[0];
    assert!(
        before > 9.0,
        "the fast window should be at the new level: {before}"
    );
    slow.blend_toward_long_run();
    let tm = slow.target_moments().unwrap();
    assert!(
        tm.means()[0] < before - 5.0,
        "the blend should pull the mean toward the long run: {before} -> {}",
        tm.means()[0]
    );
    // And the mixture is still a valid set of moments.
    assert!(tm.vars()[0] > 0.0);
    assert!(tm.q()[0] > 0.0 && tm.n_kish(slow.target_weights())[0].unwrap() > 1.0);
}

#[test]
fn session_shrink_config_is_validated() {
    let mut c = cfg(1, 1);
    c.session_shrink = Some(0.5);
    assert!(EwRidge::new(c).is_err(), "shrink without long_half_life");
    let mut c = cfg(1, 1);
    c.long_half_life = Some(1000.0);
    assert!(EwRidge::new(c).is_err(), "long_half_life without shrink");
    let mut c = cfg(1, 1);
    c.session_shrink = Some(1.5);
    c.long_half_life = Some(1000.0);
    assert!(EwRidge::new(c).is_err(), "shrink out of range");
}

#[test]
fn coef0_shape_is_validated() {
    let mut c = cfg(2, 1);
    c.coef_prior = Some(vec![vec![0.0, 1.0]]); // too short
    assert!(EwRidge::new(c).is_err());
    let mut c = cfg(2, 1);
    c.coef_prior = Some(vec![vec![0.0, 1.0, f64::NAN]]);
    assert!(EwRidge::new(c).is_err());
    // coef_prior *is* allowed with ridge_scale -- that combination is the
    // fading warm start -- so only the shape rules above are enforced.
    let mut c = cfg(2, 1);
    c.ridge_scale = true;
    c.standardize = false;
    c.coef_prior = Some(vec![vec![0.0, 1.0, 2.0]]);
    assert!(EwRidge::new(c).is_ok());
}

/// Deterministic pseudo-random stream (no external rng dependency).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

#[test]
fn recovers_static_beta() {
    let beta = [0.5, -1.0, 2.0];
    let mut m = EwRidge::new(cfg(3, 1)).unwrap();
    let mut s = 42u64;
    let mut last = Step {
        pred: vec![],
        n_eff: 0.0,
        extra: None,
    };
    for i in 0..500 {
        let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let y: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>() + 3.0;
        let d = if i == 0 { 0.0 } else { 1.0 };
        last = m.step(&x, &[Some(y)], d, 1.0);
    }
    let c = &m.coefficients().unwrap()[0];
    assert!((c[0] - 3.0).abs() < 1e-6, "intercept {}", c[0]);
    for i in 0..3 {
        assert!((c[i + 1] - beta[i]).abs() < 1e-6, "beta[{i}] {}", c[i + 1]);
    }
    assert!(last.pred[0].is_finite());
}

#[test]
fn pred_is_out_of_sample_and_warmup_nan() {
    let mut m = EwRidge::new(cfg(2, 1)).unwrap();
    let mut s = 7u64;
    for i in 0..3 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let st = m.step(
            &x,
            &[Some(lcg(&mut s))],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
        assert!(st.pred[0].is_nan(), "warmup row {i} must be NaN");
    }
    let st = m.step(&[0.1, 0.2], &[Some(0.3)], 1.0, 1.0);
    assert!(st.pred[0].is_finite());
}

#[test]
fn solve_schedule_staleness() {
    // With a large solve_every, coefficients stay fixed between solves.
    let mut c = cfg(1, 1);
    c.solve_every = 1e9;
    c.max_rows_between_solves = 10;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 9u64;
    let mut snapshots = vec![];
    for i in 0..40 {
        let x = [lcg(&mut s)];
        let y = 2.0 * x[0] + 0.01 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        snapshots.push(m.coefficients().map(|b| b[0].clone()));
    }
    // Between solve rows the coefficients are bit-identical.
    let mut changes = 0;
    for w in snapshots.windows(2) {
        if w[0] != w[1] {
            changes += 1;
        }
    }
    assert!(
        changes <= 5,
        "expected sparse solves, got {changes} changes"
    );
}

#[test]
fn multi_target_and_grids() {
    let mut c = cfg(3, 2);
    c.ridge = vec![1e-8, 10.0];
    c.feature_sets = vec![("a".into(), vec![0, 1]), ("b".into(), vec![2])];
    let m = EwRidge::new(c.clone()).unwrap();
    assert_eq!(m.n_outputs(), 2 * 4);

    let mut m = EwRidge::new(c).unwrap();
    let mut s = 3u64;
    let mut last = None;
    for i in 0..200 {
        let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let y0 = x[0] - x[1];
        let y1 = 3.0 * x[2];
        last = Some(m.step(
            &x,
            &[Some(y0), Some(y1)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        ));
    }
    let st = last.unwrap();
    assert_eq!(st.pred.len(), 8);
    assert!(st.pred.iter().all(|p| p.is_finite()));
    // combo "b" (only x2) predicts y1 well, y0 badly; heavy ridge shrinks.
    let b = m.coefficients().unwrap();
    // target 1 (y1), combo index 2 = fs "b", small ridge: coef on x2 ~ 3
    assert!((b[4 + 2][3] - 3.0).abs() < 0.05, "{:?}", b[6]);
    // heavy-ridge combo shrinks toward zero
    assert!(b[4 + 3][3].abs() < b[4 + 2][3].abs());
    // feature set "b" never touches x0/x1
    assert_eq!(b[4 + 2][1], 0.0);
    assert_eq!(b[4 + 2][2], 0.0);
}

#[test]
fn standardize_matches_plain_when_ridge_tiny() {
    // On well-conditioned data the centered/standardized solve and the raw
    // solve are algebraically identical (ridge ~ 0), so they must agree
    // tightly. (On badly scaled data they diverge because the raw normal
    // equations are ill-conditioned -- which is why standardize exists.)
    // Ridge is dropped to 1e-12 and the feature scales are put ~1e3 apart,
    // so the centering and scaling matrices are far from the identity and
    // an operation applied in the wrong direction cannot hide inside the
    // tolerance. The invariance is exact only at ridge = 0, because the
    // penalty lands on the raw scale in one path and the standardized
    // scale in the other.
    let mut ca = cfg(2, 1);
    ca.ridge = vec![1e-12];
    let mut cb = ca.clone();
    cb.standardize = true;
    let mut ma = EwRidge::new(ca).unwrap();
    let mut mb = EwRidge::new(cb).unwrap();
    let mut s = 11u64;
    for i in 0..300 {
        let x = [400.0 + 3.0 * lcg(&mut s), 0.002 * lcg(&mut s)];
        let y = 7.0 + 0.5 * x[0] - 300.0 * x[1] + 0.001 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        ma.step(&x, &[Some(y)], d, 1.0);
        mb.step(&x, &[Some(y)], d, 1.0);
    }
    let a = &ma.coefficients().unwrap()[0];
    let b = &mb.coefficients().unwrap()[0];
    for i in 0..3 {
        assert!(
            (a[i] - b[i]).abs() < 1e-6 * (1.0 + a[i].abs()),
            "coef {i}: plain {} vs standardized {}",
            a[i],
            b[i]
        );
    }
    // And the standardized path recovered the generating relationship, so
    // the agreement is not two paths failing the same way. The intercept is
    // reconstructed from the centered fit, which is the step most easily
    // lost.
    assert!((b[0] - 7.0).abs() < 0.1, "intercept {}", b[0]);
    assert!((b[1] - 0.5).abs() < 1e-3, "slope 0 {}", b[1]);
    assert!((b[2] + 300.0).abs() < 5.0, "slope 1 {}", b[2]);
}

/// A model with a slow twin, fed `n` rows of a deterministic stream.
fn blended_pair(shrink: f64) -> EwRidge {
    blended_pair_at(shrink, 0.0)
}

/// [`blended_pair`] with every feature, and so the target, at `level`.
fn blended_pair_at(shrink: f64, level: f64) -> EwRidge {
    let mut c = cfg(2, 1);
    c.session_shrink = Some(shrink);
    c.long_half_life = Some(400.0);
    c.decay = Decay::Halflife(20.0);
    c.min_weight = 3.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 23u64;
    for i in 0..200 {
        let x = [level + lcg(&mut s), level + 0.5 + lcg(&mut s)];
        // A relationship that flips halfway, so fast and slow genuinely
        // disagree by the time the session boundary arrives.
        let sign = if i < 100 { 1.0 } else { -1.0 };
        let y = sign * (2.0 * x[0] - x[1]);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    m
}

#[test]
fn coef0_solves_the_ridge_problem_it_claims_to() {
    // With `coef_prior = c`, the penalty shrinks toward `c` rather than zero:
    //     beta = (C + rI)^-1 (d + r c)
    // on the centered accumulators, with the intercept recovered after.
    // The existing coef_prior tests check the *direction* of the pull; this
    // pins the closed form, computed by hand from the model's own state.
    let (r, c0) = (0.7, vec![vec![0.0, 3.0, -2.0]]);
    let mut cfg_ = cfg(2, 1);
    cfg_.ridge = vec![r];
    cfg_.coef_prior = Some(c0.clone());
    cfg_.min_weight = 3.0;
    let mut m = EwRidge::new(cfg_).unwrap();
    let mut s = 127u64;
    for i in 0..200 {
        let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
        let y = 1.0 + 2.0 * x[0] - x[1] + 0.05 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }

    // The 2x2 penalized normal equations on the centered moments.
    let (c00, c01, c11) = (
        m.gram(0).cov(1, 1),
        m.gram(0).cov(1, 2),
        m.gram(0).cov(2, 2),
    );
    // d_i = E[x_i y] - E[x_i] E[y], from the tracked cross-moment means.
    let raw = m.cross_moments();
    let d0 = raw[0][1] - m.gram(0).mean(1) * raw[0][0];
    let d1 = raw[0][2] - m.gram(0).mean(2) * raw[0][0];
    let (a00, a11) = (c00 + r, c11 + r);
    let (rhs0, rhs1) = (d0 + r * c0[0][1], d1 + r * c0[0][2]);
    let det = a00 * a11 - c01 * c01;
    let want = [
        (rhs0 * a11 - c01 * rhs1) / det,
        (a00 * rhs1 - c01 * rhs0) / det,
    ];

    let got = &m.coefficients().unwrap()[0];
    for i in 0..2 {
        assert!(
            (got[i + 1] - want[i]).abs() < 1e-9 * (1.0 + want[i].abs()),
            "slope {i}: {} vs {}",
            got[i + 1],
            want[i]
        );
    }
    // The intercept is reconstructed, not fitted: mean(y) - b'mean(x).
    let want0 = raw[0][0] - got[1] * m.gram(0).mean(1) - got[2] * m.gram(0).mean(2);
    assert!((got[0] - want0).abs() < 1e-9, "{} vs {want0}", got[0]);

    // An overwhelming penalty must land on coef_prior exactly.
    let mut cfg_ = cfg(2, 1);
    cfg_.ridge = vec![1e12];
    cfg_.coef_prior = Some(c0.clone());
    cfg_.min_weight = 3.0;
    let mut m = EwRidge::new(cfg_).unwrap();
    let mut s = 131u64;
    for i in 0..100 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let got = &m.coefficients().unwrap()[0];
    assert!(
        (got[1] - 3.0).abs() < 1e-3,
        "slope 0 -> coef_prior: {}",
        got[1]
    );
    assert!(
        (got[2] + 2.0).abs() < 1e-3,
        "slope 1 -> coef_prior: {}",
        got[2]
    );
}

#[test]
fn a_singular_solve_is_counted_and_the_previous_fit_is_kept() {
    // `run_solve` records both outcomes: a solve rescued by jitter and a
    // total failure. Two perfectly collinear features with no ridge give
    // a rank-deficient system; the model must never emit NaN and must say
    // that something went wrong.
    let mut c = cfg(2, 1);
    c.ridge = vec![0.0];
    c.min_weight = 2.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 137u64;
    for i in 0..40 {
        let a = lcg(&mut s);
        m.step(
            &[a, a],
            &[Some(2.0 * a + 1.0)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    assert!(m.solve_failures > 0, "a singular solve must be recorded");
    let beta = &m.coefficients().unwrap()[0];
    assert!(beta.iter().all(|v| v.is_finite()), "never NaN: {beta:?}");

    // A well-conditioned stream records nothing.
    let mut c = cfg(2, 1);
    c.min_weight = 2.0;
    let mut ok = EwRidge::new(c).unwrap();
    let mut s = 139u64;
    for i in 0..40 {
        let x = [lcg(&mut s), lcg(&mut s)];
        ok.step(
            &x,
            &[Some(x[0] - x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    assert_eq!(ok.solve_failures, 0, "a healthy fit must record no failure");
}

#[test]
fn the_slow_twin_is_the_same_model_at_the_long_halflife() {
    // The twin's accumulators are updated by a second copy of the update
    // block inside `step`, which nothing else reaches. The oracle is the
    // obvious one: a standalone model configured at `long_half_life` and
    // fed the same rows must end up with identical statistics.
    let mut c = cfg(2, 1);
    c.session_shrink = Some(0.4);
    c.long_half_life = Some(300.0);
    c.decay = Decay::Halflife(15.0);
    c.min_weight = 3.0;

    let mut twin_cfg = cfg(2, 1);
    twin_cfg.decay = Decay::Halflife(300.0);
    twin_cfg.min_weight = 3.0;

    let mut m = EwRidge::new(c).unwrap();
    let mut reference = EwRidge::new(twin_cfg).unwrap();
    let mut s = 43u64;
    for i in 0..150 {
        let x = [lcg(&mut s), 2.0 + lcg(&mut s)];
        // Every fourth row has a null target and an irregular gap, so the
        // twin's null branch (`*wj *= slow_lam`) is exercised too.
        let y = if i % 4 == 3 {
            None
        } else {
            Some(1.5 * x[0] - 0.5 * x[1])
        };
        let d = if i == 0 { 0.0 } else { 1.0 + (i % 3) as f64 };
        m.step(&x, &[y], d, 1.0);
        reference.step(&x, &[y], d, 1.0);
    }

    let slow = m.slow.as_ref().unwrap();
    let k = m.cfg.k_total();
    assert!((slow.grams.grams[0].n_eff() - reference.gram(0).n_eff()).abs() < 1e-9);
    for i in 0..k {
        assert!(
            (slow.grams.grams[0].mean(i) - reference.gram(0).mean(i)).abs() < 1e-9,
            "mean {i}"
        );
        for j in 0..k {
            assert!((slow.grams.grams[0].cov(i, j) - reference.gram(0).cov(i, j)).abs() < 1e-9);
        }
    }
    assert!((slow.wj[0] - reference.acc.wj[0]).abs() < 1e-9);
    let (got, want) = (slow.cross.raw(0), reference.acc.cross.raw(0));
    for i in 0..k {
        assert!((got[i] - want[i]).abs() < 1e-9, "r[{i}]");
    }
    // And it is genuinely slower than the fast side, or the test would
    // pass with the twin wired to the wrong decay.
    assert!(
        slow.grams.grams[0].n_eff() > 3.0 * m.gram(0).n_eff(),
        "{} vs {}",
        slow.grams.grams[0].n_eff(),
        m.gram(0).n_eff()
    );
}

#[test]
fn residual_variance_is_the_ew_mean_of_squared_out_of_sample_errors() {
    // `sigma2` is only surfaced through the Polars layer, so nothing in
    // this crate pinned its recursion. It is an EW mean on the model's own
    // clock, over the *predicted* residual -- and rows before the first
    // prediction contribute nothing.
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(25.0);
    c.min_weight = 3.0;
    let mut m = EwRidge::new(c).unwrap();

    let (mut want, mut wsig) = (0.0, 0.0);
    let mut s = 47u64;
    for i in 0..120 {
        let x = [lcg(&mut s)];
        let y = 2.0 * x[0] + 0.3 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        let lam = 0.5f64.powf(d / 25.0);
        let p = m.step(&x, &[Some(y)], d, 1.0).pred[0];
        if p.is_finite() {
            let resid = y - p;
            let ws_new = lam * wsig + 1.0;
            want = (lam * wsig * want + resid * resid) / ws_new;
            wsig = ws_new;
        }
        assert!(
            (m.sigma2()[0] - want).abs() < 1e-12,
            "row {i}: {} vs {want}",
            m.sigma2()[0]
        );
    }
    // The weight saturates at 1/(1 - lam) ~ 36.6 for this half-life.
    assert!(
        wsig > 30.0,
        "the recursion should have run, not been skipped"
    );
    assert!(
        want > 0.0 && want < 1.0,
        "plausible residual variance: {want}"
    );
}

#[test]
fn null_targets_decay_the_residual_variance_weight_without_adding_to_it() {
    // The `None` arm decays `wj` and `wsig` but must not fold a residual
    // in -- otherwise a gap in the target inflates or freezes sigma.
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(10.0);
    c.min_weight = 2.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 53u64;
    for i in 0..60 {
        let x = [lcg(&mut s)];
        m.step(
            &x,
            &[Some(2.0 * x[0] + 0.1)],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let (sig, w, wj) = (m.sigma2()[0], m.wsig[0], m.acc.wj[0]);
    assert!(sig > 0.0 && w > 0.0);

    let lam = 0.5f64.powf(3.0 / 10.0);
    m.step(&[0.5], &[None], 3.0, 1.0);
    assert_eq!(m.sigma2()[0], sig, "a null target must not move sigma2");
    assert!((m.wsig[0] - w * lam).abs() < 1e-12, "but its weight decays");
    assert!((m.acc.wj[0] - wj * lam).abs() < 1e-12);
}

/// `σ²` is the EW mean of the squared out-of-sample errors: every row
/// ages its weight by the row's `lam`, and a row with a target, a
/// weight and a prediction adds `w·r²`. A row with a target and no
/// prediction -- here the rows after a clock gap has taken `n_eff` under
/// `min_weight` -- rightly added nothing, and aged nothing either, so
/// `σ²` forgot less across them than the clock says (N6, found beside
/// review 2026-09-12 S13).
#[test]
fn the_residual_variance_ages_on_every_row() {
    let hl = 10.0;
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(hl);
    c.min_weight = 3.0;
    let mut m = EwRidge::new(c).unwrap();
    let (mut want, mut wsig, mut unpredicted) = (0.0f64, 0.0f64, 0);
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
        let p = m.step(&x, &[y], d, w).pred[0];
        wsig *= 0.5f64.powf(d / hl);
        match y {
            Some(y) if w > 0.0 && p.is_finite() => {
                let r = y - p;
                let ws_new = wsig + w;
                want = (wsig * want + w * r * r) / ws_new;
                wsig = ws_new;
            }
            Some(_) if w > 0.0 && wsig > 0.0 => unpredicted += 1,
            _ => {}
        }
        let got = m.sigma2()[0];
        assert!(
            (got - want).abs() <= 1e-12 * want,
            "row {i}: sigma2 {got}, the EW mean of the squared errors {want}"
        );
    }
    assert!(
        unpredicted >= 2,
        "the gap must leave rows with no prediction"
    );
}

/// Under `ridge_scale` the penalty is a pseudo-observation of the
/// history, `prior_scale · ridge · I` on the sum scale, and it decays
/// with the history. A blend keeps today's weight `W`, so it keeps
/// today's prior with it: the penalty stays the same share of the data
/// (task 145). The blend rebuilt the Gram from `EwCov::new`, which put
/// the prior back at full strength on every session boundary (review
/// 2026-09-12, C6). At `f = 1` the fit is the twin's moments at today's
/// weight and prior, worked from the rows: `(W S_H + ps · ridge I) β =
/// W r_H`, `S_H` and `r_H` the rows' raw moments under the twin's kernel,
/// solved by `faer`.
#[test]
fn a_blend_keeps_the_decaying_prior_with_the_weight() {
    let (hl, long, ridge, n) = (20.0, 400.0, 5.0, 250usize);
    let build = |f: f64| {
        let mut c = cfg(2, 1);
        c.ridge = vec![ridge];
        c.ridge_scale = true;
        c.decay = Decay::Halflife(hl);
        c.long_half_life = Some(long);
        c.session_shrink = Some(f);
        c.min_weight = 0.0;
        EwRidge::new(c).unwrap()
    };
    let (mut part, mut full) = (build(0.3), build(1.0));
    let mut s = 31u64;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
        let y = 1.0 + 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        part.step(&x, &[Some(y)], d, 1.0);
        full.step(&x, &[Some(y)], d, 1.0);
        rows.push(([1.0, x[0], x[1]], y));
    }
    let fast = part.gram(0).prior_scale();
    let slow = part.slow.as_ref().unwrap().grams.grams[0].prior_scale();
    assert!(fast < 1e-3 && slow > 0.5, "fast {fast}, slow {slow}");
    part.blend_toward_long_run();
    assert_eq!(
        part.gram(0).prior_scale().to_bits(),
        fast.to_bits(),
        "prior_scale {} after the blend, today's {fast}",
        part.gram(0).prior_scale()
    );

    let (w, ps) = (full.gram(0).n_eff(), full.gram(0).prior_scale());
    full.blend_toward_long_run();
    assert_eq!(full.gram(0).prior_scale().to_bits(), ps.to_bits());
    let ks: Vec<f64> = (0..n)
        .map(|i| 0.5f64.powf((n - 1 - i) as f64 / long))
        .collect();
    let total: f64 = ks.iter().sum();
    let raw = |g: &dyn Fn(usize) -> f64| (0..n).map(|i| ks[i] * g(i)).sum::<f64>() / total;
    let a: Vec<Vec<f64>> = (0..3)
        .map(|p| {
            (0..3)
                .map(|q| {
                    w * raw(&|i| rows[i].0[p] * rows[i].0[q])
                        + if p == q { ps * ridge } else { 0.0 }
                })
                .collect()
        })
        .collect();
    let b: Vec<f64> = (0..3)
        .map(|p| w * raw(&|i| rows[i].0[p] * rows[i].1))
        .collect();
    let want = oracle::solve(&a.concat(), &b);
    let got = full.coefficients().unwrap()[0].clone();
    for i in 0..3 {
        assert!(
            (got[i] - want[i]).abs() <= 1e-9 * (1.0 + want[i].abs()),
            "coef {i}: {} after a full blend, {} from the rows",
            got[i],
            want[i]
        );
    }
}

/// The window's boundary across a clock gap longer than the window,
/// through the model: the row after the gap is the only one inside, so
/// the window's weight is that row's, at every cadence -- every row, five
/// rows, five clock units, and thirty, longer than the window. With a
/// cadence of 5 rows the boundary stayed at the last snapshot before the
/// gap, and the rows since it stayed in (found testing review 2026-09-12,
/// S6).
#[test]
fn a_window_holds_nothing_from_before_a_gap_longer_than_it() {
    for (every, rows) in [
        (None, None),
        (None, Some(5)),
        (Some(5.0), None),
        (Some(30.0), None),
    ] {
        let mut c = cfg(1, 1);
        c.window = Some(20.0);
        c.window_every = every;
        c.max_rows_between_snapshots = rows;
        c.min_weight = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 67u64;
        for i in 0..101 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        for (inside, d) in [(1.0, 100.0), (2.0, 1.0)] {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(x[0])], d, 1.0);
            assert!(
                (m.n_eff() - inside).abs() < 1e-9,
                "{every:?}, {rows:?}: a weight of {} in the window, {inside} rows inside it",
                m.n_eff()
            );
        }
    }
}

/// `predict` and `n_eff` read the window's weights without the O(k²)
/// view (review 2026-09-12, P1), and must read the view's numbers, to
/// the bit, on every row: before the window has aged anything out, while
/// it does, across a gap that empties it, and per target with gaps.
#[test]
fn the_window_weights_are_the_views_to_the_bit() {
    let mut c = cfg(2, 2);
    c.decay = Decay::Halflife(15.0);
    c.window = Some(12.0);
    c.max_rows_between_snapshots = Some(3);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 71u64;
    for i in 0..120 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y0 = (i % 5 != 2).then_some(x[0] - x[1]);
        let y1 = (i % 3 != 1).then_some(x[0] + 0.5);
        let d = match i {
            0 => 0.0,
            60 => 40.0,
            _ => 1.0,
        };
        m.step(&x, &[y0, y1], d, if i % 11 == 4 { 0.0 } else { 1.0 });
        let live = (m.acc.cross.w, m.acc.wj.clone());
        let cheap = m.window_weights().unwrap_or_else(|| live.clone());
        let full = m
            .view()
            .map_or_else(|| live.clone(), |v| (v.acc.cross.w, v.acc.wj.clone()));
        assert_eq!(cheap.0.to_bits(), full.0.to_bits(), "row {i}");
        assert_eq!(cheap.1, full.1, "row {i}");
    }
}

#[test]
fn the_solve_schedule_controls_when_coefficients_move() {
    // Almost every test here solves on every row, which masks the three
    // clauses of the `due` condition. Each is checked on its own.
    let coefs = |c: EwRidgeCfg, ds: &[f64]| {
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 59u64;
        let mut out = Vec::new();
        for (i, &d) in ds.iter().enumerate() {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(3.0 * x[0])], if i == 0 { 0.0 } else { d }, 1.0);
            out.push(m.coefficients().map(|b| b[0][1]));
        }
        out
    };
    let ones = vec![1.0; 24];

    // Row-counted. `solve_every = 0` means "every row" and short-circuits
    // the rest of the condition, so the row cap is only visible with a
    // clock schedule that will not fire.
    let mut c = cfg(1, 1);
    c.min_weight = 2.0;
    c.solve_every = 1e9;
    c.max_rows_between_solves = 5;
    let by_rows = coefs(c, &ones);
    let changes: Vec<usize> = by_rows
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] != w[1])
        .map(|(i, _)| i + 1)
        .collect();
    // First solve as soon as min_weight is met, then strictly every 5
    // accepted rows: nothing in between, and it does not stop.
    assert_eq!(changes, vec![1, 6, 11, 16, 21], "24 rows, cap of 5");

    // Clock-counted: the same stream on a clock that advances 2 per row
    // must solve half as often when `solve_every` is 4.
    let mut c = cfg(1, 1);
    c.min_weight = 2.0;
    c.max_rows_between_solves = u32::MAX;
    c.solve_every = 4.0;
    let slow = coefs(c, &[2.0; 24]);
    let n_slow = slow.windows(2).filter(|w| w[0] != w[1]).count();

    let mut c = cfg(1, 1);
    c.min_weight = 2.0;
    c.max_rows_between_solves = u32::MAX;
    c.solve_every = 0.0; // 0 means "every row"
    let every = coefs(c, &ones);
    let n_every = every.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(
        n_slow < n_every,
        "solve_every should throttle: {n_slow} vs {n_every}"
    );

    // The first solve is not throttled: it happens as soon as min_weight
    // is met, however long the schedule says to wait.
    let mut c = cfg(1, 1);
    c.min_weight = 3.0;
    c.max_rows_between_solves = u32::MAX;
    c.solve_every = 1e9;
    let first = coefs(c, &ones);
    assert!(
        first.iter().take(6).any(|b| b.is_some()),
        "the first solve must not wait for the schedule"
    );
}

#[test]
fn cfg_validation_rejects_each_bad_field() {
    // One case per rejection in `EwRidgeCfg::validate`, each matched on the
    // message so a mutation that reports the wrong reason is caught too --
    // and each accompanied by the nearest *valid* config, so a validator
    // that rejects everything cannot pass either.
    let bad = |f: &dyn Fn(&mut EwRidgeCfg), want: &str| {
        let mut c = cfg(2, 1);
        f(&mut c);
        match c.validate() {
            Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
            Ok(()) => panic!("expected rejection mentioning {want:?}"),
        }
    };
    let good = |f: &dyn Fn(&mut EwRidgeCfg)| {
        let mut c = cfg(2, 1);
        f(&mut c);
        c.validate().expect("should be accepted");
    };

    bad(&|c| c.n_features = 0, "must be >= 1");
    bad(&|c| c.n_targets = 0, "must be >= 1");
    bad(&|c| c.ridge = vec![], "at least one value");

    // ridge_scale alone is fine; it is the combination that is refused.
    good(&|c| c.ridge_scale = true);
    bad(
        &|c| {
            c.ridge_scale = true;
            c.standardize = true;
        },
        "incompatible",
    );
    bad(
        &|c| {
            c.ridge_scale = true;
            c.ridge = vec![1e-6, 1.0];
        },
        "incompatible",
    );

    bad(&|c| c.session_shrink = Some(-0.1), "in [0, 1]");
    bad(&|c| c.session_shrink = Some(1.5), "in [0, 1]");
    bad(&|c| c.session_shrink = Some(0.5), "needs long_half_life");
    bad(&|c| c.long_half_life = Some(100.0), "no effect without");
    bad(
        &|c| {
            c.session_shrink = Some(0.5);
            c.long_half_life = Some(0.0);
        },
        "must be > 0",
    );
    bad(
        &|c| {
            c.session_shrink = Some(0.5);
            c.long_half_life = Some(f64::NAN);
        },
        "must be > 0",
    );
    good(&|c| {
        c.session_shrink = Some(0.0);
        c.long_half_life = Some(100.0);
    });
    good(&|c| {
        c.session_shrink = Some(1.0);
        c.long_half_life = Some(100.0);
    });

    // coef_prior is one vector per target, each of length k_total (2 + intercept).
    bad(
        &|c| c.coef_prior = Some(vec![vec![0.0; 3], vec![0.0; 3]]),
        "1 vector of",
    );
    bad(&|c| c.coef_prior = Some(vec![vec![0.0; 2]]), "length 3");
    bad(
        &|c| c.coef_prior = Some(vec![vec![0.0, 0.0, f64::NAN]]),
        "finite",
    );
    bad(
        &|c| c.coef_prior = Some(vec![vec![0.0, 0.0, f64::INFINITY]]),
        "finite",
    );
    good(&|c| c.coef_prior = Some(vec![vec![1.0, 2.0, 3.0]]));

    // Empty is its own message: it named "out-of-range indices", which
    // an empty set has not got (review 2026-09-12, S7).
    bad(&|c| c.feature_sets = vec![("a".into(), vec![])], "is empty");
    bad(
        &|c| c.feature_sets = vec![("a".into(), vec![2])],
        "out-of-range",
    );
    good(&|c| c.feature_sets = vec![("a".into(), vec![0]), ("b".into(), vec![0, 1])]);

    cfg(2, 1).validate().expect("the baseline config is valid");
}

#[test]
fn blend_is_the_data_share_mixture() {
    // Every arithmetic step of `blend_toward_long_run` is checked against
    // the same quantities recomputed by hand from the pre-blend state, so
    // a factor applied to the wrong side, a missing re-centering, or a
    // swapped index all show up. At the origin and at 1e8: the oracle is
    // the centred mixture, which a level costs nothing, where it was the
    // raw one re-centred, which at 1e8 has no digits left to compare
    // (review 2026-09-12, C16; docs/PLAN.md task 112). The shares are
    // `(1 − f, f)` of the data and the weight is today's (task 145).
    for level in [0.0, 1e8] {
        blend_is_the_data_share_mixture_at(level);
    }
}

fn blend_is_the_data_share_mixture_at(level: f64) {
    // The rounding the means' difference carries at the level: the one
    // input the mixture takes from level-sized numbers.
    let tol = 1e-10 + 64.0 * f64::EPSILON * level;
    let f = 0.3;
    let mut m = blended_pair_at(f, level);
    let before = m.clone();
    let slow = before.slow.as_ref().unwrap();
    let k = m.cfg.k_total();

    let (wf, ws) = (before.gram(0).n_eff(), slow.grams.grams[0].n_eff());
    assert!(wf > 0.0 && ws > wf, "the slow twin should hold more weight");
    let (af, as_) = (1.0 - f, f);

    m.blend_toward_long_run();

    assert_eq!(
        m.gram(0).n_eff().to_bits(),
        wf.to_bits(),
        "the weight is today's"
    );
    let (fast, twin) = (before.gram(0), &slow.grams.grams[0]);
    for i in 0..k {
        let want = af * fast.mean(i) + as_ * twin.mean(i);
        assert!(
            (m.gram(0).mean(i) - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "level {level}, mean {i}: {} vs {want}",
            m.gram(0).mean(i)
        );
    }
    for i in 0..k {
        for j in 0..k {
            // The centred mixture: each side's co-moment about its own
            // mean, and the spread between the two means about the mixed
            // one, `a·C_f + b·C_s + a·b·(m_f − m_s)(m_f − m_s)ᵀ`.
            let (di, dj) = (fast.mean(i) - twin.mean(i), fast.mean(j) - twin.mean(j));
            let want = af * fast.cov(i, j) + as_ * twin.cov(i, j) + af * as_ * di * dj;
            assert!(
                (m.gram(0).cov(i, j) - want).abs() <= tol,
                "level {level}, cov {i},{j}: {} vs {want}",
                m.gram(0).cov(i, j)
            );
        }
    }
    // The raw cross-moments mix linearly, whatever split the centred ones
    // were mixed in; at a level they are level-sized, so held relatively.
    for (j, got) in m.cross_moments().iter().enumerate() {
        assert_eq!(m.acc.wj[j].to_bits(), before.acc.wj[j].to_bits());
        let (fast_r, slow_r) = (before.acc.cross.raw(j), slow.cross.raw(j));
        for i in 0..k {
            let want = af * fast_r[i] + as_ * slow_r[i];
            assert!(
                (got[i] - want).abs() <= 1e-12 * (1.0 + want.abs()),
                "level {level}, r[{j}][{i}]"
            );
        }
    }
}

#[test]
fn blend_endpoints_are_identity_and_full_replacement() {
    // f = 0 must not touch the state; f = 1 must land exactly on the
    // twin's moments, at today's weight (task 145).
    let mut zero = blended_pair(0.0);
    let before = zero.clone();
    zero.blend_toward_long_run();
    assert_eq!(zero, before, "session_shrink = 0 must be a no-op");

    let mut one = blended_pair(1.0);
    let before = one.clone();
    let slow = one.slow.clone().unwrap();
    one.blend_toward_long_run();
    let k = one.cfg.k_total();
    assert_eq!(
        one.gram(0).n_eff().to_bits(),
        before.gram(0).n_eff().to_bits()
    );
    for i in 0..k {
        assert!(
            (one.gram(0).mean(i) - slow.grams.grams[0].mean(i)).abs() < 1e-12,
            "mean {i}"
        );
        for j in 0..k {
            assert!(
                (one.gram(0).cov(i, j) - slow.grams.grams[0].cov(i, j)).abs() < 1e-12,
                "cov {i},{j}"
            );
        }
    }
    for (j, got) in one.cross_moments().iter().enumerate() {
        assert_eq!(one.acc.wj[j].to_bits(), before.acc.wj[j].to_bits());
        let want = slow.cross.raw(j);
        for i in 0..k {
            assert!((got[i] - want[i]).abs() < 1e-12, "r[{j}][{i}]");
        }
    }
}

/// A feature held over the window has no spread there (task 94), so an
/// unstandardized solve with no ridge meets an exactly singular Gram
/// where it met one singular up to rounding: the solver's jitter takes
/// it, the feature's slope is exactly 0 (its right-hand side is zeroed
/// too), and every prediction stays finite (review 2026-09-25).
#[test]
fn a_held_feature_under_a_window_leaves_an_unregularized_solve_finite() {
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(20.0);
    c.window = Some(30.0);
    c.standardize = false;
    c.ridge = vec![0.0];
    c.min_weight = 5.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 3u64;
    for i in 0..200 {
        let x0 = lcg(&mut s);
        let x1 = if i < 100 { lcg(&mut s) } else { 0.37 };
        let y = 1.0 + 2.0 * x0 - x1;
        let out = m.step(&[x0, x1], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        assert!(
            out.pred[0].is_finite() || i < 10,
            "row {i}: {}",
            out.pred[0]
        );
        if i >= 150 {
            let slope = m.coefficients().map_or(f64::NAN, |c| c[0][2]);
            assert!(
                slope.abs() < 1e-9,
                "row {i}: slope {slope} on a feature without spread"
            );
        }
    }
}

/// A row of weight 0 takes no step in any mean, to the bit: the Gram's,
/// the cross-moments' feature and target means, and the target moments'
/// (`crate::comp::add` says why a zero step would move a pair). The rows
/// 0.7 and 5.292162135665459 at unit weight and no decay leave each mean
/// with a low part of a whole rounding step, where adding zero would
/// round the double up.
#[test]
fn a_row_of_no_weight_leaves_every_mean_as_it_was() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(f64::INFINITY);
    c.min_weight = 1.0;
    let mut m = EwRidge::new(c).unwrap();
    for v in [0.7, 5.292162135665459] {
        m.step(&[v], &[Some(v)], 1.0, 1.0);
    }
    let cross = &m.acc.cross;
    assert_eq!(cross.m[1], 2.996081067832729, "the fixture");
    assert_eq!(
        cross.m_lo[1], 4.440892098500626e-16,
        "a whole step below the double"
    );
    let before = (
        cross.m.clone(),
        cross.my.clone(),
        m.acc.tm.means().to_vec(),
        m.acc
            .grams
            .grams
            .iter()
            .map(|g| g.means().to_vec())
            .collect::<Vec<_>>(),
    );
    m.step(&[1.0], &[Some(1.0)], 1.0, 0.0);
    let cross = &m.acc.cross;
    let after = (
        cross.m.clone(),
        cross.my.clone(),
        m.acc.tm.means().to_vec(),
        m.acc
            .grams
            .grams
            .iter()
            .map(|g| g.means().to_vec())
            .collect::<Vec<_>>(),
    );
    assert_eq!(after, before);
}

#[test]
fn blend_before_any_data_is_a_no_op() {
    // The other half of the doc comment's promise: a no-op when the twin
    // is not configured, *or before it has seen anything*. With both sides
    // at zero weight the mixture's denominator is zero, and the guard has
    // to catch that rather than divide.
    let mut c = cfg(2, 1);
    c.session_shrink = Some(0.5);
    c.long_half_life = Some(200.0);
    let mut m = EwRidge::new(c).unwrap();
    assert!(m.slow.is_some());
    let before = m.clone();
    m.blend_toward_long_run();
    assert_eq!(m, before, "nothing seen yet: nothing to blend");
    for i in 0..m.cfg.k_total() {
        assert!(m.gram(0).mean(i).is_finite(), "and no NaN got in");
    }

    // Still safe on a session boundary that arrives with a session's worth
    // of zero-weight rows behind it.
    for _ in 0..5 {
        m.step(&[1.0, 2.0], &[Some(3.0)], 1.0, 0.0);
    }
    let before = m.clone();
    m.blend_toward_long_run();
    assert_eq!(m, before, "zero-weight rows carry no weight to blend");
}

#[test]
fn blend_without_a_twin_is_a_no_op() {
    // No `session_shrink` at all: there is no twin to blend with, and the
    // method must return before touching anything.
    let mut c = cfg(2, 1);
    c.min_weight = 3.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 29u64;
    for i in 0..50 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert!(m.slow.is_none());
    let before = m.clone();
    m.blend_toward_long_run();
    assert_eq!(m, before);
}

#[test]
fn blend_moves_the_fit_toward_the_long_run_relationship() {
    // The point of the feature, not just its arithmetic: the last session
    // ran at the opposite sign to the long run, and blending must pull the
    // fit back -- monotonically in the shrink parameter.
    let slope = |f: f64| {
        let mut m = blended_pair(f);
        m.blend_toward_long_run();
        m.coefficients_after_blend()
    };
    let (none, half, full) = (slope(0.0), slope(0.5), slope(1.0));
    assert!(none < 0.0, "the last session was negative: {none}");
    assert!(
        full > none,
        "the long run should pull it up: {full} vs {none}"
    );
    assert!(
        none < half && half < full,
        "reversion should be monotone: {none} < {half} < {full}"
    );
}

#[test]
fn standardize_without_intercept_matches_plain_when_ridge_tiny() {
    // `solve_standardized` has a second, quite different branch for
    // `fit_intercept = false`: it scales by the *raw* second-moment
    // diagonals rather than centering first. The invariance is the same --
    // a diagonal rescale of the normal equations cannot move the solution
    // when the penalty is negligible -- so a scaling applied in the wrong
    // direction, or to only one of A, b and the unscaled result, shows up
    // here. Features are deliberately on very different scales (~4 and
    // ~0.003) so the scaling matrix is far from the identity.
    let mut ca = cfg(2, 1);
    ca.fit_intercept = false;
    ca.min_weight = 2.0;
    // The invariance is exact only at ridge = 0: a penalty applies on the
    // raw scale in one path and the standardized scale in the other, and
    // the two feature scales here differ by ~1e3, so 1e-8 is enough to
    // separate them at 1e-6.
    ca.ridge = vec![1e-12];
    let mut cb = ca.clone();
    cb.standardize = true;
    let mut ma = EwRidge::new(ca).unwrap();
    let mut mb = EwRidge::new(cb).unwrap();
    let mut s = 17u64;
    for i in 0..300 {
        let x = [4.0 + lcg(&mut s), 0.003 * lcg(&mut s)];
        let y = 0.25 * x[0] - 40.0 * x[1] + 0.001 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        ma.step(&x, &[Some(y)], d, 1.0);
        mb.step(&x, &[Some(y)], d, 1.0);
    }
    let a = &ma.coefficients().unwrap()[0];
    let b = &mb.coefficients().unwrap()[0];
    assert_eq!(a.len(), 2, "no intercept slot when fit_intercept is false");
    for i in 0..2 {
        assert!(
            (a[i] - b[i]).abs() < 1e-6 * (1.0 + a[i].abs()),
            "coef {i}: plain {} vs standardized {}",
            a[i],
            b[i]
        );
    }
    // And it actually recovered the generating coefficients, so the
    // agreement is not two paths being wrong the same way.
    assert!((b[0] - 0.25).abs() < 1e-3, "{}", b[0]);
    assert!((b[1] + 40.0).abs() < 1.0, "{}", b[1]);
}

#[test]
fn standardize_without_intercept_drops_a_zero_column() {
    // The `s[i] > 0.0` guard in the no-intercept branch: a feature that is
    // identically zero has zero raw second moment, so it cannot be scaled
    // and must come out with a zero coefficient rather than a NaN.
    let mut c = cfg(2, 1);
    c.fit_intercept = false;
    c.standardize = true;
    c.min_weight = 2.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 19u64;
    for i in 0..100 {
        let x = [1.0 + lcg(&mut s), 0.0];
        let y = 2.0 * x[0];
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let b = &m.coefficients().unwrap()[0];
    assert_eq!(b[1], 0.0, "the all-zero feature must be dropped, not NaN");
    assert!((b[0] - 2.0).abs() < 1e-6, "{}", b[0]);
}

#[test]
fn zero_variance_feature_dropped_in_standardized_solve() {
    let mut c = cfg(2, 1);
    c.standardize = true;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 13u64;
    for i in 0..100 {
        let x = [lcg(&mut s), 5.0]; // constant second feature
        let y = 2.0 * x[0];
        let st = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        if i > 10 {
            assert!(st.pred[0].is_finite(), "row {i}");
        }
    }
    let b = &m.coefficients().unwrap()[0];
    assert_eq!(b[2], 0.0, "full coef {b:?}"); // dropped, not blown up
    assert!((b[1] - 2.0).abs() < 1e-6); // 1e-8 ridge itself shifts this by ~2e-8
}

/// A windowed ridge fit, solved directly from the normal equations over
/// exactly the rows inside the window. Written from the definition:
/// `(Z'WZ + ridge·I) beta = Z'Wy` with `W = diag(lam^age)` over the rows
/// inside the window -- age less than it under `closed = "right"`, the
/// default, at most it under `"both"` ([`inside`]; docs/PLAN.md task 196)
/// -- and nothing else.
#[allow(clippy::too_many_arguments)]
fn direct_window_fit(
    xs: &[[f64; 1]],
    ys: &[f64],
    t: &[f64],
    half_life: f64,
    window: f64,
    ridge: f64,
    upto: usize,
    closed: crate::WindowClosed,
) -> [f64; 2] {
    let now = t[upto - 1];
    // z = [1, x]: a 2x2 normal system, solved in closed form.
    let (mut s11, mut s1x, mut sxx, mut s1y, mut sxy, mut wsum) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..upto {
        if !inside(now - t[i], window, closed) {
            continue;
        }
        let w = 0.5_f64.powf((now - t[i]) / half_life);
        let x = xs[i][0];
        wsum += w;
        s11 += w;
        s1x += w * x;
        sxx += w * x * x;
        s1y += w * ys[i];
        sxy += w * x * ys[i];
    }
    // The model accumulates *means*, so the ridge sits on the mean scale,
    // and on the slope alone: the intercept is not penalized.
    let (a11, a12, a22) = (s11 / wsum, s1x / wsum, sxx / wsum + ridge);
    let (b1, b2) = (s1y / wsum, sxy / wsum);
    let det = a11 * a22 - a12 * a12;
    [(b1 * a22 - b2 * a12) / det, (b2 * a11 - b1 * a12) / det]
}

/// Whether a row `age` clock units old is inside a `window` with edge
/// `closed`, by the definition (docs/PLAN.md task 196, N17).
fn inside(age: f64, window: f64, closed: crate::WindowClosed) -> bool {
    match closed {
        crate::WindowClosed::Right => age < window,
        crate::WindowClosed::Both => age <= window,
    }
}

/// PLAN §13.4 (1), for the Gram: a windowed fit is the fit of the rows in
/// the window, and nothing older reaches the coefficients, under either
/// edge.
#[test]
fn a_windowed_fit_is_the_fit_of_the_rows_inside_the_window() {
    for closed in [crate::WindowClosed::Right, crate::WindowClosed::Both] {
        windowed_fit_is_the_fit_of_the_rows_inside(closed);
    }
}

fn windowed_fit_is_the_fit_of_the_rows_inside(closed: crate::WindowClosed) {
    let (half_life, window, ridge) = (30.0, 80.0, 1e-8);
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(half_life);
    c.ridge = vec![ridge];
    c.min_weight = 0.0;
    c.solve_every = 0.0;
    c.max_rows_between_solves = 1; // solve every row, so beta is never stale
    c.window = Some(window);
    let mut m = EwRidge::new(c).unwrap();
    m.set_window_closed(closed);

    let mut s = 11u64;
    let (mut xs, mut ys, mut t, mut clock) = (vec![], vec![], vec![], 0.0);
    let mut on_the_boundary = 0;
    for i in 0..150 {
        // `lcg` is in [-1, 1), so take its magnitude: a clock must not go
        // backwards, and one that does makes the window meaningless.
        // Quarter units, exact in a double, so some rows land exactly
        // one window old (the boundary is inclusive).
        let d = if i == 0 {
            0.0
        } else {
            0.25 * (2.0 + (lcg(&mut s).abs() * 8.0).floor())
        };
        clock += d;
        let x = [lcg(&mut s) * 4.0 - 2.0];
        // A relationship that changes halfway, so a window that forgets
        // the old one reports something the full history could not.
        let y = if i < 75 {
            3.0 * x[0] + 1.0
        } else {
            -2.0 * x[0] + 5.0
        } + 0.05 * lcg(&mut s);
        m.step(&x, &[Some(y)], d, 1.0);
        xs.push(x);
        ys.push(y);
        t.push(clock);
        if i >= 8 {
            // Membership first: if the truncated weight matches the sum
            // over the rows the definition keeps, the two agree on *which*
            // rows are in, and any difference left is arithmetic.
            let now = t[i];
            let wsum: f64 = (0..=i)
                .filter(|&j| inside(now - t[j], window, closed))
                .map(|j| 0.5_f64.powf((now - t[j]) / half_life))
                .sum();
            on_the_boundary += (0..=i).filter(|&j| now - t[j] == window).count();
            assert!(
                (m.n_eff() - wsum).abs() < 1e-9 * wsum,
                "row {i}: n_eff {} vs {wsum} -- the window holds different rows",
                m.n_eff()
            );
            let want = direct_window_fit(&xs, &ys, &t, half_life, window, ridge, i + 1, closed);
            let got = m.coefficients().unwrap();
            for (slot, wanted) in want.iter().enumerate() {
                // To rounding: the window is a subtraction, but at 2.7
                // half-lives it discards a sixth of the weight and loses
                // next to nothing (1.2e-14 measured). The "eight
                // significant figures" this once allowed was the oracle's
                // own ridge on the intercept, which the model leaves free
                // (review 2026-09-12, D6).
                assert!(
                    (got[0][slot] - wanted).abs() < 1e-12 * wanted.abs().max(1.0),
                    "row {i} slot {slot}: {} vs {wanted}",
                    got[0][slot]
                );
            }
        }
    }
    // The boundary was exercised, not only the rows either side of it.
    assert!(
        on_the_boundary >= 5,
        "{on_the_boundary} fits had a row exactly one window old"
    );
}

/// PLAN task 94, for the ridge: a feature that held one value over every
/// row inside the window has no evidence there, so its slope is exactly
/// zero -- `(0 + ridge)·b = 0` -- where the window's subtraction left a
/// variance and a cross-moment that a small ridge divided one by the
/// other. At a level of 1e6 that was a slope on the order of the
/// remainder over the ridge. The feature moved until row 150 and holds
/// after it, so from row 161 on the window of 10 holds none of its
/// moves.
#[test]
fn a_feature_held_over_the_window_gets_no_slope() {
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(20.0);
    c.ridge = vec![1e-8];
    c.min_weight = 0.0;
    c.window = Some(10.0);
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 5u64;
    let held = 1e6 + 0.37;
    for i in 0..220 {
        let x0 = lcg(&mut s);
        let x1 = if i < 150 { 1e6 + lcg(&mut s) } else { held };
        let y = 2.0 * x0 + 0.5 * (x1 - 1e6) + 0.1 * lcg(&mut s);
        m.step(&[x0, x1], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        if i >= 161 {
            let coef = &m.coefficients().unwrap()[0];
            assert_eq!(coef[2], 0.0, "row {i}: the held feature's slope");
            assert!(
                (coef[1] - 2.0).abs() < 0.2,
                "row {i}: the moving one's, {}",
                coef[1]
            );
        }
    }
}

/// PLAN §13.4 (2): the guarantee, on the fit rather than a moment. A
/// relationship that held before the window cannot bend the coefficients
/// inside it.
#[test]
fn a_fit_cannot_be_moved_by_rows_older_than_its_window() {
    let run = |old_slope: f64| {
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(25.0);
        c.ridge = vec![1e-8];
        c.min_weight = 0.0;
        c.max_rows_between_solves = 1;
        c.window = Some(60.0);
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 3u64;
        for i in 0..200 {
            let x = [lcg(&mut s) * 2.0 - 1.0];
            // Everything before row 130 follows `old_slope`; the last 70
            // rows are the same in both runs and all lie inside a
            // 60-unit window at the end.
            let y = if i < 130 {
                old_slope * x[0]
            } else {
                4.0 * x[0]
            };
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.coefficients().unwrap()[0].clone()
    };
    let a = run(1.0);
    let b = run(-500.0);
    for slot in 0..2 {
        assert!(
            (a[slot] - b[slot]).abs() < 1e-6,
            "slot {slot}: a slope of -500 outside the window moved the fit: {} vs {}",
            a[slot],
            b[slot]
        );
    }
    assert!(
        (a[1] - 4.0).abs() < 0.01,
        "the in-window slope is 4: {}",
        a[1]
    );
}

/// PLAN §13.4 (5): a window no stream reaches changes nothing.
#[test]
fn a_ridge_window_no_stream_reaches_is_the_untruncated_fit() {
    let mk = |window: Option<f64>| {
        let mut c = cfg(2, 1);
        c.decay = Decay::Halflife(40.0);
        c.min_weight = 0.0;
        c.max_rows_between_solves = 1;
        c.window = window;
        EwRidge::new(c).unwrap()
    };
    let (mut plain, mut windowed) = (mk(None), mk(Some(1e9)));
    let mut s = 7u64;
    for i in 0..80 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = x[0] - 2.0 * x[1] + 0.1 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        let a = plain.step(&x, &[Some(y)], d, 1.0);
        let b = windowed.step(&x, &[Some(y)], d, 1.0);
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits(), "row {i}");
        assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "row {i} n_eff");
    }
}

#[test]
fn state_roundtrip_continues_identically() {
    let mut m1 = EwRidge::new(cfg(2, 1)).unwrap();
    let mut s = 5u64;
    let mut rows = vec![];
    for _ in 0..60 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = x[0] + 0.1 * lcg(&mut s);
        rows.push((x, y));
    }
    for (i, (x, y)) in rows[..30].iter().enumerate() {
        m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
    let restored: State = rmp_serde::from_slice(&bytes).unwrap();
    let mut m2 = EwRidge::restore(&restored).unwrap();
    for (x, y) in &rows[30..] {
        let a = m1.step(x, &[Some(*y)], 1.0, 1.0);
        let b = m2.step(x, &[Some(*y)], 1.0, 1.0);
        assert_eq!(a.pred, b.pred);
        assert_eq!(a.n_eff, b.n_eff);
    }
}

#[test]
fn null_target_is_predict_only() {
    let mut m = EwRidge::new(cfg(1, 2)).unwrap();
    let mut s = 17u64;
    for i in 0..50 {
        let x = [lcg(&mut s)];
        m.step(
            &x,
            &[Some(2.0 * x[0]), Some(-x[0])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let (c1, my1) = (m.acc.cross.c[1].clone(), m.acc.cross.my[1]);
    let mean_z1 = m.acc.cross.mj[1].clone();
    let m_all = m.acc.cross.m.clone();
    let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
    assert!(st.pred[1].is_finite()); // pred still emitted
    // Target 1's own moments do not move: no data added (mean form), its
    // own mean of `z` included, to the bit; the all-row mean did move.
    assert_eq!(m.acc.cross.c[1], c1);
    assert_eq!(m.acc.cross.my[1], my1);
    assert_eq!(m.acc.cross.mj[1], mean_z1);
    assert_ne!(m.acc.cross.m, m_all);
}

/// A level costs the fit nothing (review 2026-09-12, N1). The same stream
/// at the origin and shifted by `1e8` -- features and target alike, a
/// price regressed on prices -- must give the same slopes, and
/// predictions that differ by the shift. The cross-moments were raw, and
/// the solves formed `E[z·y] − m·ȳ` or read the raw normal equations,
/// both of which lose `L²·ε`: at `1e8` the prediction was off by about
/// ten. Blocked too, where the Gram's own mean moves only at a flush and
/// the cross-moments keep theirs. The tolerances are the data's own
/// resolution at `1e8`, `ulp ≈ 1.5e-8`.
#[test]
fn a_level_costs_the_fit_nothing() {
    for (standardize, block) in [(false, 0), (true, 0), (false, 8), (true, 8)] {
        let run = |level: f64| {
            let mut c = cfg(2, 1);
            c.standardize = standardize;
            c.ridge = vec![1e-6];
            c.decay = Decay::Halflife(200.0);
            c.min_weight = 10.0;
            if block > 0 {
                c.gram_block_rows = block;
                c.solve_every = 5.0;
                c.max_rows_between_solves = 10;
            }
            let mut m = EwRidge::new(c).unwrap();
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
        let what = format!("standardize {standardize}, block {block}");
        let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
        for i in 1..3 {
            assert!(
                (b0[i] - b8[i]).abs() < 1e-6,
                "{what}: slope {i}: {} vs {}",
                b0[i],
                b8[i]
            );
        }
        let mut worst = 0.0f64;
        for (t, (a, b)) in p0.iter().zip(&p8).enumerate() {
            assert_eq!(a.is_finite(), b.is_finite(), "{what}: row {t}: {a} vs {b}");
            if a.is_finite() {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-5, "{what}: the predictions part by {worst}");
    }
}

/// A target present on every row has the all-row mean as its own, to the
/// bit, blocked or not -- the same steps over the same rows -- and a
/// target with gaps a mean of its own rows (N1; own means since review
/// 2026-09-26, G3).
#[test]
fn a_target_present_on_every_row_has_the_all_row_mean() {
    for block in [0, 8] {
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(50.0);
        if block > 0 {
            c.gram_block_rows = block;
            c.solve_every = 5.0;
            c.max_rows_between_solves = 10;
        }
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 31u64;
        for i in 0..100 {
            let x = [1e8 + lcg(&mut s), 3.0 + lcg(&mut s)];
            // Target 1 is null every fifth row; target 0 never is.
            let y1 = (i % 5 != 2).then_some(x[1]);
            let d = if i == 0 { 0.0 } else { 1.0 };
            m.step(&x, &[Some(x[0] + x[1]), y1], d, 1.0);
        }
        assert_eq!(m.acc.cross.mj[0], m.acc.cross.m, "block {block}");
        assert!(
            m.acc.cross.mj[1][2] != m.acc.cross.m[2],
            "block {block}: a target with gaps"
        );
        // Unblocked, the cross-moments' copy of the Gram's mean is the
        // Gram's own, to the bit.
        if block == 0 {
            assert_eq!(m.acc.cross.m.as_slice(), m.gram(0).means());
            assert_eq!(m.acc.cross.mj[0].as_slice(), m.gram(0).means());
        }
    }
}

// ---- gram_block_rows (docs/ENHANCEMENTS.md E51, docs/PLAN.md task 71) ----

/// A half-life, a solve cadence in both clock and rows, and a small ridge:
/// the setting the option is for. `block` is the only difference between
/// the two sides of every comparison below.
fn blocked_cfg(k: usize, block: usize) -> EwRidgeCfg {
    let mut c = cfg(k, 1);
    c.decay = Decay::Halflife(200.0);
    c.ridge = vec![1e-3];
    c.solve_every = 10.0;
    c.max_rows_between_solves = 50;
    c.gram_block_rows = block;
    c
}

/// `(x, y, d_clock, weight)`: a missing target every 17th row, a zero
/// weight every 23rd, a clock gap of 5 half-lives every 41st and one
/// infinite gap (`lam == 0`, the history discarded) -- everything a held
/// row has to carry into the merge.
fn ridge_rows(n: usize, k: usize, seed: u64) -> Vec<(Vec<f64>, Option<f64>, f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y: f64 = x
                .iter()
                .enumerate()
                .map(|(j, v)| (j as f64 + 1.0) * v)
                .sum::<f64>()
                + 0.5
                + 0.1 * lcg(&mut s);
            let y = if i % 17 == 5 { None } else { Some(y) };
            let d = match i {
                0 => 0.0,
                301 => f64::INFINITY,
                _ if i % 41 == 0 => 1000.0,
                _ => 1.0,
            };
            let w = if i % 23 == 7 {
                0.0
            } else {
                0.5 + lcg(&mut s).abs()
            };
            (x, y, d, w)
        })
        .collect()
}

fn assert_pred_close(a: &[f64], b: &[f64], what: &str) {
    for (i, (p, q)) in a.iter().zip(b).enumerate() {
        let same = (p.is_nan() && q.is_nan()) || (p - q).abs() <= 1e-9 * (1.0 + p.abs());
        assert!(same, "{what}: pred[{i}] {p} vs {q}");
    }
}

/// The merge is the per-row recursion to rounding: the same predictions
/// at every row, the same `n_eff` to the bit (the scalars never left the
/// per-row path), and the same coefficients at the end. Checked with
/// blocks smaller than, equal to and larger than the solve cadence, and
/// with the path actually taken: a block was pending at some point.
#[test]
fn a_blocked_fit_is_the_per_row_fit_to_rounding() {
    for k in [1usize, 3, 12] {
        for block in [1usize, 7, 50, 64] {
            let mut plain = EwRidge::new(blocked_cfg(k, 0)).unwrap();
            let mut blocked = EwRidge::new(blocked_cfg(k, block)).unwrap();
            let mut held = false;
            for (i, (x, y, d, w)) in ridge_rows(600, k, 9 + k as u64).iter().enumerate() {
                let a = plain.step(x, &[*y], *d, *w);
                let b = blocked.step(x, &[*y], *d, *w);
                assert_pred_close(&a.pred, &b.pred, &format!("k={k} block={block} row {i}"));
                assert_eq!(a.n_eff, b.n_eff, "k={k} block={block} row {i}: n_eff");
                held |= blocked.gram(0).has_pending();
            }
            // A block of one fills on the row that opens it, so nothing
            // is ever seen pending; every larger block is.
            assert_eq!(
                held,
                block > 1,
                "k={k} block={block}: rows held between steps"
            );
            let (ca, cb) = (
                plain.coefficients().unwrap(),
                blocked.coefficients().unwrap(),
            );
            for (i, (p, q)) in ca[0].iter().zip(&cb[0]).enumerate() {
                assert!(
                    (p - q).abs() <= 1e-9 * (1.0 + p.abs()),
                    "k={k} block={block}: coef[{i}] {p} vs {q}"
                );
            }
        }
    }
}

/// A solve reads the matrix, so it merges the block first: after every
/// solve nothing is pending, whichever of the two schedules fired it. A
/// zero-weight row never solves, nor counts toward the row cap (task 214),
/// so the count after it is the solve's 0 with the row held.
#[test]
fn a_solve_merges_the_block_first() {
    let mut m = EwRidge::new(blocked_cfg(3, 64)).unwrap();
    let mut solves = 0;
    for (x, y, d, w) in ridge_rows(400, 3, 4) {
        m.step(&x, &[y], d, w);
        if w > 0.0 && m.rows_since_solve == 0 {
            solves += 1;
            assert!(!m.gram(0).has_pending(), "a solve left rows pending");
        }
    }
    assert!(solves > 10, "the schedule fired {solves} times");
}

/// A blend reads both matrices in full, so it merges both blocks first,
/// and the blended accumulator keeps the block size: the option is a
/// property of the model, not of one accumulator's lifetime.
#[test]
fn a_blend_merges_the_held_blocks_first_and_keeps_the_block_size() {
    let shrink = |block: usize| {
        let mut c = blocked_cfg(2, block);
        c.session_shrink = Some(0.5);
        c.long_half_life = Some(1e4);
        EwRidge::new(c).unwrap()
    };
    let (mut plain, mut blocked) = (shrink(0), shrink(16));
    for (x, y, d, w) in ridge_rows(37, 2, 5) {
        plain.step(&x, &[y], d, w);
        blocked.step(&x, &[y], d, w);
    }
    assert!(blocked.gram(0).has_pending(), "row 37 should sit mid-block");
    assert!(blocked.slow.as_ref().unwrap().grams.grams[0].has_pending());
    plain.blend_toward_long_run();
    blocked.blend_toward_long_run();
    assert!(!blocked.gram(0).has_pending());
    assert!(!blocked.slow.as_ref().unwrap().grams.grams[0].has_pending());
    assert_eq!(blocked.gram(0).block_rows(), 16);
    assert_eq!(
        blocked.slow.as_ref().unwrap().grams.grams[0].block_rows(),
        16
    );
    let (a, b) = (
        plain.coefficients_after_blend(),
        blocked.coefficients_after_blend(),
    );
    assert!((a - b).abs() <= 1e-9 * (1.0 + a.abs()), "{a} vs {b}");
    // And the blended accumulator goes on holding rows.
    for (x, y, d, w) in ridge_rows(5, 2, 6) {
        blocked.step(&x, &[y], d, w);
    }
    assert!(blocked.gram(0).has_pending());
}

/// The held rows are in the state, so a save mid-block resumes the same
/// merge on the same rows: bit for bit, both encodings, and the restored
/// model keeps holding rows rather than flushing on load.
#[test]
fn a_state_saved_mid_block_resumes_the_blocked_fit_bit_for_bit() {
    let rows = ridge_rows(120, 3, 8);
    for named in [false, true] {
        let mut one = EwRidge::new(blocked_cfg(3, 32)).unwrap();
        for (x, y, d, w) in &rows[..23] {
            one.step(x, &[*y], *d, *w);
        }
        assert!(one.gram(0).has_pending(), "row 23 should sit mid-block");
        let st = one.state();
        let bytes = if named {
            rmp_serde::to_vec_named(&st).unwrap()
        } else {
            rmp_serde::to_vec(&st).unwrap()
        };
        let restored: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut two = EwRidge::restore(&restored).unwrap();
        assert!(
            two.gram(0).has_pending(),
            "the held rows did not survive the save"
        );
        assert_eq!(two.gram(0).block_rows(), 32);
        for (x, y, d, w) in &rows[23..] {
            let a = one.step(x, &[*y], *d, *w);
            let b = two.step(x, &[*y], *d, *w);
            let bits = |p: &[f64]| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.pred), bits(&b.pred), "named={named}");
            assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "named={named}");
        }
        assert_eq!(one.coefficients(), two.coefficients());
    }
}

/// The settings under which a block cannot pay, or cannot be read,
/// are refused with the reason; `0` is the untouched path whatever the
/// schedule.
#[test]
fn gram_block_rows_is_refused_where_it_cannot_pay() {
    let err = |c: EwRidgeCfg| EwRidge::new(c).err().unwrap_or_default();

    let mut c = blocked_cfg(3, 64);
    c.window = Some(50.0);
    assert!(err(c).contains("gram_block_rows and window do not combine"));

    let mut c = blocked_cfg(3, 64);
    c.solve_every = 0.0;
    let e = err(c);
    assert!(
        e.contains("needs a solve cadence") && e.contains("solve_every = 0"),
        "{e}"
    );

    let mut c = blocked_cfg(3, 64);
    c.max_rows_between_solves = 1;
    let e = err(c);
    assert!(e.contains("max_rows_between_solves = 1"), "{e}");

    // 2^30 rows of 4 floats is 32 GiB; the twin doubles it.
    let mut c = blocked_cfg(3, 1 << 30);
    c.session_shrink = Some(0.5);
    c.long_half_life = Some(1e4);
    let e = err(c);
    assert!(
        e.contains("over the 256 MiB budget") && e.contains("64.0 GiB") && e.contains("twin"),
        "{e}"
    );
    // Just inside the budget builds.
    let mut c = blocked_cfg(3, (256 << 20) / (4 * 8));
    assert!(EwRidge::new(c.clone()).is_ok());
    c.gram_block_rows += 1;
    assert!(err(c).contains("over the 256 MiB budget"));

    let mut c = blocked_cfg(3, 0);
    c.solve_every = 0.0;
    c.max_rows_between_solves = 1;
    assert!(EwRidge::new(c).is_ok(), "0 is off, whatever the schedule");
}

// ---- target_gaps (docs/PLAN.md task 81) ----

/// Features, three targets, clock delta, weight.
type GappyRow = ([f64; 2], [Option<f64>; 3], f64, f64);

/// Three targets with three patterns of missing rows -- present on every
/// row, on a schedule, and where a feature is low -- beside a model of
/// each target alone, fed the same rows. Zero-weight rows are in the
/// stream too, some with targets missing.
fn gappy_rows(n: usize) -> Vec<GappyRow> {
    let mut s = 61u64;
    (0..n)
        .map(|i| {
            let x = [lcg(&mut s), 2.0 + lcg(&mut s)];
            let base = 1.5 * x[0] - 0.5 * x[1] + 0.05 * lcg(&mut s);
            let y = [
                Some(3.0 + base),
                (i % 3 != 1).then_some(-1.0 + base),
                (x[0] < 0.3).then_some(7.0 + 0.5 * base),
            ];
            let d = if i == 0 { 0.0 } else { 1.0 };
            let w = if i % 11 == 5 {
                0.0
            } else {
                0.5 + (i % 3) as f64
            };
            (x, y, d, w)
        })
        .collect()
}

/// Under `own_rows` each target of a bank is fitted on exactly its rows:
/// its Gram is, to the bit, the Gram a model of that target alone keeps
/// -- whichever targets it shared one with and wherever it split from
/// them, blocked or not -- and so are its cross-moments, its weight and
/// its fit. Every row counts toward `n_eff` whatever the targets, and the
/// split Grams survive a save in both encodings.
#[test]
fn under_own_rows_each_target_is_the_model_of_that_target_alone() {
    for block in [0usize, 8] {
        let mut c = cfg(2, 3);
        c.decay = Decay::Halflife(30.0);
        if block > 0 {
            c.gram_block_rows = block;
            c.solve_every = 3.0;
            c.max_rows_between_solves = 5;
        }
        let mut bank = EwRidge::new(c.clone()).unwrap();
        let one = EwRidgeCfg {
            n_targets: 1,
            ..c.clone()
        };
        let mut alone: Vec<EwRidge> = (0..3).map(|_| EwRidge::new(one.clone()).unwrap()).collect();
        let rows = gappy_rows(300);
        for (x, y, d, w) in &rows[..200] {
            bank.step(x, y, *d, *w);
            for (j, a) in alone.iter_mut().enumerate() {
                a.step(x, &[y[j]], *d, *w);
            }
        }
        // Three patterns of missing rows, three Grams.
        assert_eq!(bank.acc.grams.grams.len(), 3, "block {block}");
        for (j, a) in alone.iter().enumerate() {
            let g = &bank.acc.grams.grams[bank.acc.grams.of[j]];
            assert_eq!(g, a.gram(0), "block {block}: target {j}'s Gram");
            assert_eq!(
                g.n_eff(),
                bank.acc.wj[j],
                "block {block}: target {j}'s weight"
            );
            assert_eq!(
                bank.acc.cross.c[j], a.acc.cross.c[0],
                "block {block}: c[{j}]"
            );
            assert_eq!(bank.acc.cross.my[j], a.acc.cross.my[0], "block {block}");
            assert_eq!(bank.n_eff(), a.n_eff(), "block {block}: every row counts");
            let (b, want) = (
                &bank.coefficients().unwrap()[j],
                &a.coefficients().unwrap()[0],
            );
            for (p, q) in b.iter().zip(want) {
                assert!(
                    (p - q).abs() <= 1e-12 * (1.0 + q.abs()),
                    "block {block}: target {j}: {b:?} vs {want:?}"
                );
            }
        }
        for named in [false, true] {
            let st = bank.state();
            let bytes = if named {
                rmp_serde::to_vec_named(&st).unwrap()
            } else {
                rmp_serde::to_vec(&st).unwrap()
            };
            let mut back = EwRidge::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            let mut live = bank.clone();
            for (x, y, d, w) in &rows[200..] {
                let (p, q) = (live.step(x, y, *d, *w), back.step(x, y, *d, *w));
                let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&p.pred), bits(&q.pred), "block {block} named={named}");
            }
            assert_eq!(live, back, "block {block} named={named}");
        }
    }
}

/// N3 (docs/PLAN.md task 81): a slope no longer moves with the target's
/// level. The target is present only where `x0 < 0.2`, so its rows' mean
/// of `x0` is not the all-row mean, and the old right-hand side added
/// `(m_j − m)·ȳ_j / Var(x)` to the slope. Under either reading the
/// slopes at a level of `1e3` are the slopes at 0, and the intercept
/// takes the level.
#[test]
fn a_target_level_moves_no_slope_under_either_reading() {
    for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
        for standardize in [false, true] {
            let run = |level: f64| {
                let mut c = cfg(2, 1);
                c.decay = Decay::Halflife(60.0);
                c.standardize = standardize;
                c.target_gaps = gaps;
                c.ridge = vec![1e-6];
                let mut m = EwRidge::new(c).unwrap();
                let mut s = 67u64;
                for i in 0..400 {
                    let x = [lcg(&mut s), lcg(&mut s)];
                    let y = level + 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s);
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    m.step(&x, &[(x[0] < 0.2).then_some(y)], d, 1.0);
                }
                m.coefficients().unwrap()[0].clone()
            };
            let (b0, b1) = (run(0.0), run(1e3));
            let what = format!("{gaps:?}, standardize {standardize}");
            for i in 1..3 {
                assert!(
                    (b0[i] - b1[i]).abs() < 1e-9,
                    "{what}: slope {i}: {} vs {}",
                    b0[i],
                    b1[i]
                );
            }
            assert!(
                (b1[0] - b0[0] - 1e3).abs() < 1e-8,
                "{what}: the intercept takes the level"
            );
        }
    }
}

/// A blend under `own_rows` mixes each Gram with the twin's Gram of the
/// same targets. The twin sees the same rows and targets, so it splits
/// where the fast side does, and each target's blended accumulators are,
/// to the bit, those of a model of that target alone blended at the same
/// boundary.
#[test]
fn a_blend_under_own_rows_is_each_targets_own_blend() {
    let mut c = cfg(2, 3);
    c.decay = Decay::Halflife(15.0);
    c.session_shrink = Some(0.4);
    c.long_half_life = Some(200.0);
    c.min_weight = 3.0;
    let mut bank = EwRidge::new(c.clone()).unwrap();
    let one = EwRidgeCfg {
        n_targets: 1,
        ..c.clone()
    };
    let mut alone: Vec<EwRidge> = (0..3).map(|_| EwRidge::new(one.clone()).unwrap()).collect();
    for (x, y, d, w) in gappy_rows(200) {
        bank.step(&x, &y, d, w);
        for (j, a) in alone.iter_mut().enumerate() {
            a.step(&x, &[y[j]], d, w);
        }
    }
    let twin = bank.slow.as_ref().unwrap();
    assert_eq!(
        twin.grams.of, bank.acc.grams.of,
        "the twin splits where the fast side does"
    );
    bank.blend_toward_long_run();
    for a in &mut alone {
        a.blend_toward_long_run();
    }
    for (j, a) in alone.iter().enumerate() {
        assert_eq!(
            bank.acc.grams.grams[bank.acc.grams.of[j]],
            *a.gram(0),
            "target {j}"
        );
        assert_eq!(bank.acc.wj[j], a.acc.wj[0], "target {j}");
        assert_eq!(bank.acc.cross.c[j], a.acc.cross.c[0], "target {j}");
        let (b, want) = (
            &bank.coefficients().unwrap()[j],
            &a.coefficients().unwrap()[0],
        );
        for (p, q) in b.iter().zip(want) {
            assert!(
                (p - q).abs() <= 1e-12 * (1.0 + q.abs()),
                "target {j}: {b:?} vs {want:?}"
            );
        }
    }
}

/// `pairwise` keeps one Gram over every row whatever the gaps, and its
/// slopes are pairwise-complete moments: the Gram's centred co-moments
/// against each target's own centred cross-moments.
#[test]
fn pairwise_keeps_one_gram_over_every_row() {
    let mut c = cfg(2, 3);
    c.decay = Decay::Halflife(30.0);
    c.target_gaps = TargetGaps::Pairwise;
    let mut m = EwRidge::new(c).unwrap();
    for (x, y, d, w) in gappy_rows(200) {
        m.step(&x, &y, d, w);
    }
    assert_eq!(m.acc.grams.grams.len(), 1);
    assert_eq!(m.gram(0).n_eff(), m.n_eff(), "the Gram is over every row");
    assert!(
        m.acc.wj[2] < m.acc.wj[0],
        "a target with gaps has less weight"
    );
}

/// A row with no weight learns nothing, so it parts no targets: one
/// that is null on a zero-weight row keeps its Gram, which is still, to
/// the bit, the Gram of a model of that target alone. A zero-weight row
/// that also takes all of the weight -- `lam = 0`, an infinite clock gap
/// -- parts none either: taking the row and skipping it are both the
/// decay alone (task 159, R1; it once split them there).
#[test]
fn a_zero_weight_row_splits_no_gram() {
    let mut c = cfg(2, 2);
    c.decay = Decay::Halflife(20.0);
    let mut bank = EwRidge::new(c.clone()).unwrap();
    let one = EwRidgeCfg { n_targets: 1, ..c };
    let mut alone = [
        EwRidge::new(one.clone()).unwrap(),
        EwRidge::new(one).unwrap(),
    ];
    let mut s = 71u64;
    for i in 0..120 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 1.0 + x[0] - 2.0 * x[1];
        // Target 1 is null on exactly the zero-weight rows.
        let w = if i % 10 == 9 { 0.0 } else { 1.0 };
        let ys = [Some(y), (w > 0.0).then_some(2.0 * y)];
        let d = if i == 0 { 0.0 } else { 1.0 };
        bank.step(&x, &ys, d, w);
        for (j, a) in alone.iter_mut().enumerate() {
            a.step(&x, &[ys[j]], d, w);
        }
    }
    assert_eq!(
        bank.acc.grams.grams.len(),
        1,
        "a zero-weight row split them"
    );
    for (j, a) in alone.iter().enumerate() {
        assert_eq!(bank.gram(0), a.gram(0), "target {j}");
        assert_eq!(bank.acc.wj[j], a.acc.wj[0], "target {j}");
    }
    bank.step(&[0.1, 0.2], &[Some(1.0), None], f64::INFINITY, 0.0);
    assert_eq!(bank.acc.grams.grams.len(), 1, "an infinite gap split them");
    let of = &bank.acc.grams.of;
    assert_eq!(bank.acc.grams.grams[of[1]].n_eff(), 0.0);
    assert_eq!(bank.acc.grams.grams[of[0]].n_eff(), bank.acc.wj[0]);
}

/// Under `own_rows` a target not seen yet is alone in a Gram with no
/// weight. With `ridge = 0` there is no system there, so it has no fit,
/// NaN (zeros until review round 4, CC1), and no failure is counted; a
/// solve of the empty Gram was a jittered one, counted, on every row until
/// the target came.
#[test]
fn a_target_not_seen_yet_costs_no_solve_failure() {
    let mut c = cfg(2, 2);
    c.ridge = vec![0.0];
    c.min_weight = 3.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 73u64;
    let mut row = |m: &mut EwRidge, i: usize| {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y = 1.0 + x[0] - 2.0 * x[1] + 0.01 * lcg(&mut s);
        m.step(&x, &[Some(y), None], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    };
    // The first rows are fewer than the terms: target 0's own solves are
    // singular there, and counted, as least squares on them would be.
    for i in 0..10 {
        row(&mut m, i);
    }
    let warm = m.solve_failures;
    for i in 10..60 {
        row(&mut m, i);
    }
    assert_eq!(m.acc.grams.grams.len(), 2);
    assert_eq!(m.solve_failures, warm, "the empty Gram was solved");
    assert!(m.coefficients().unwrap()[1].iter().all(|b| b.is_nan()));
    assert!(m.predict(&[0.1, 0.2], 1.0).pred[1].is_nan());
}

/// Runs are a window's (docs/PLAN.md task 128): the Grams keep them
/// under a window alone.
#[test]
fn only_a_windowed_ridge_keeps_runs() {
    for window in [None, Some(40.0)] {
        let mut c = cfg(2, 1);
        c.window = window;
        let m = EwRidge::new(c).unwrap();
        assert_eq!(
            m.gram(0).keeps_runs(),
            window.is_some(),
            "window {window:?}"
        );
    }
}

/// The window's snapshot counts every vector it holds in its footprint,
/// the cross-moments' low parts included (docs/PLAN.md task 130).
/// Every target present on every row, so each snapshot holds one Gram.
#[test]
fn the_window_footprint_counts_every_vector() {
    let snap = |k: usize, t: usize| {
        let mut c = cfg(k, t);
        c.window = Some(12.0);
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..40 {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y: Vec<Option<f64>> = (0..t).map(|j| Some(x[j % k] + lcg(&mut s))).collect();
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m.win.as_ref().unwrap().snaps.boundary().unwrap().1.clone()
    };
    crate::window::assert_footprint_counts_every_vector(&snap(2, 1), &snap(5, 3), "ewridge");
}

/// The same, over a snapshot holding two Grams: the second target absent
/// on every third row from the fifth splits the Gram under `own_rows`
/// (the others, always present, share one), and both sizes hold two, so
/// a footprint summing the first Gram alone would show (review
/// 2026-09-26, C missing 7).
#[test]
fn the_window_footprint_counts_every_gram() {
    let snap = |k: usize, t: usize| {
        let mut c = cfg(k, t);
        c.window = Some(12.0);
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 7u64;
        for i in 0..40 {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y: Vec<Option<f64>> = (0..t)
                .map(|j| (j != 1 || i < 5 || i % 3 != 1).then(|| x[j % k] + lcg(&mut s)))
                .collect();
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert_eq!(m.acc.grams.grams.len(), 2, "the Gram split");
        m.win.as_ref().unwrap().snaps.boundary().unwrap().1.clone()
    };
    crate::window::assert_footprint_counts_every_vector(
        &snap(2, 2),
        &snap(5, 3),
        "ewridge, two Grams",
    );
}

/// A state from before the runs' flag restored under a spec without a
/// window keeps no runs, as a model built under it keeps none; a state
/// whose runs are off under a windowed spec is refused; and a windowed
/// spec whose state has no ring is refused too (review 2026-09-26, C4
/// and C5).
#[test]
fn restore_holds_the_runs_and_the_ring_to_the_window() {
    let build = |window: Option<f64>| {
        let mut c = cfg(2, 1);
        c.window = window;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 3u64;
        for i in 0..30 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        serde_json::to_value(m.state()).unwrap()
    };
    let restore = |v: serde_json::Value| EwRidge::restore(&serde_json::from_value(v).unwrap());
    let mut v = build(None);
    crate::window::json_edit(&mut v, "runs", &mut |x| {
        *x = serde_json::json!({"x": [], "start": []});
    });
    assert!(
        restore(v)
            .unwrap()
            .acc
            .grams
            .grams
            .iter()
            .all(|g| !g.keeps_runs())
    );
    let mut v = build(Some(12.0));
    crate::window::json_edit(&mut v, "runs", &mut |x| {
        x["off"] = serde_json::json!(true);
    });
    assert!(matches!(restore(v), Err(StateError::Invalid(_))));
    let mut v = build(Some(12.0));
    crate::window::json_edit(&mut v, "win", &mut |x| *x = serde_json::Value::Null);
    assert!(
        matches!(restore(v), Err(StateError::Invalid(_))),
        "a windowed spec without its ring"
    );
}

/// A state whose vectors are not the cfg's is refused by `restore`, where
/// it loaded and panicked on the first `step` (review 2026-09-18, B3;
/// docs/PLAN.md task 111: every other model had this test).
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let m = EwRidge::new(cfg(2, 1)).unwrap();
    for shorten in [0usize, 1] {
        let mut s = m.state();
        let ModelState::EwRidge(inner) = &mut s.model else {
            unreachable!()
        };
        if shorten == 0 {
            inner.wsig.pop();
        } else {
            inner.sig2.pop();
        }
        match EwRidge::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{e}"),
            other => panic!("{other:?}"),
        }
    }
}

/// The fit and what the last solve left for the readiness statistics
/// are checked against the cfg as the accumulators are, as `lasso` and
/// `rls` check theirs: a slot per target and combo, each `k_total` long;
/// `edf`, `support` and `system_of` per slot, each `system_of` a system
/// or none; and the slow twin exactly when `session_shrink` asks for it.
/// Each loaded: an empty `beta` and `predict` panicked, a short slot
/// predicted a constant, an empty `edf` withheld every row, and a state
/// without its twin blended nothing (review 2026-10-05, CA1). Before the
/// first solve every one of them is empty, which loads.
#[test]
fn a_state_whose_fit_or_readiness_is_the_wrong_shape_is_refused() {
    let mut c = cfg(2, 2);
    c.ridge = vec![0.1, 1.0];
    let (m, _) = fitted(c.clone(), 30, 23);
    assert!(m.beta.is_some() && m.ready.edf.len() == 4);
    let restored = |f: &dyn Fn(&mut EwRidge)| {
        let mut s = m.state();
        let ModelState::EwRidge(inner) = &mut s.model else {
            unreachable!()
        };
        f(inner);
        EwRidge::restore(&s)
    };
    assert!(restored(&|_| {}).is_ok(), "the state as saved");
    type Damage<'a> = (&'a str, &'a dyn Fn(&mut EwRidge));
    let damage: [Damage; 12] = [
        ("no slot", &|r: &mut EwRidge| r.beta = Some(Fit(Vec::new()))),
        ("a slot short", &|r: &mut EwRidge| {
            r.beta.as_mut().unwrap().pop();
        }),
        ("a coefficient short", &|r: &mut EwRidge| {
            r.beta.as_mut().unwrap()[1].pop();
        }),
        ("no edf", &|r: &mut EwRidge| r.ready.edf.clear()),
        ("a support slot short", &|r: &mut EwRidge| {
            r.ready.support[2].pop();
        }),
        ("no support", &|r: &mut EwRidge| r.ready.support.clear()),
        ("no system_of", &|r: &mut EwRidge| r.ready.system_of.clear()),
        ("a system_of past the systems", &|r: &mut EwRidge| {
            r.ready.system_of[0] = 7;
        }),
        ("a system_of one past the systems", &|r: &mut EwRidge| {
            r.ready.system_of[0] = r.acc.grams.grams.len() * r.cfg.n_combos();
        }),
        ("readiness without a fit", &|r: &mut EwRidge| r.beta = None),
        // Each part of the readiness on its own: no fit needs all three
        // empty, not one.
        ("readiness without a fit or an edf", &|r: &mut EwRidge| {
            r.beta = None;
            r.ready.edf.clear();
        }),
        (
            "readiness without a fit or a system_of",
            &|r: &mut EwRidge| {
                r.beta = None;
                r.ready.system_of.clear();
            },
        ),
    ];
    for (what, f) in damage {
        match restored(f) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
            other => panic!("{what}: {other:?}"),
        }
    }
    // Before the first solve: no fit and no readiness, which loads.
    let fresh = EwRidge::new(c.clone()).unwrap();
    assert!(fresh.beta.is_none() && fresh.ready.edf.is_empty());
    assert_eq!(EwRidge::restore(&fresh.state()).unwrap(), fresh);
    // The twin, exactly when `session_shrink` asks for one.
    let mut shrink = c;
    shrink.session_shrink = Some(0.5);
    shrink.long_half_life = Some(200.0);
    let (twin, _) = fitted(shrink, 30, 29);
    let mut s = twin.state();
    let ModelState::EwRidge(inner) = &mut s.model else {
        unreachable!()
    };
    assert!(inner.slow.is_some());
    inner.slow = None;
    match EwRidge::restore(&s) {
        Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "no twin: {e}"),
        other => panic!("no twin: {other:?}"),
    }
    let mut s = m.state();
    let ModelState::EwRidge(inner) = &mut s.model else {
        unreachable!()
    };
    inner.slow = twin.slow.clone();
    match EwRidge::restore(&s) {
        Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "a stray twin: {e}"),
        other => panic!("a stray twin: {other:?}"),
    }
}

/// A window's snapshots are held to the cfg's shape, as the live
/// accumulators are: a snapshot a target short -- in its residual weights
/// or spreads, its target weights, its cross-moments or its target
/// moments -- or a Gram snapshot of the wrong width loaded, and the next
/// row's `view()` read `old.wsig[1]` or `old.wj[1]` of a one-entry vector
/// and panicked (review 2026-10-06, CA2). Each edit reaches only the ring,
/// under `win`, so the live vectors `restore` checked already stay intact.
#[test]
fn a_window_snapshot_of_the_wrong_shape_is_refused() {
    let mut c = cfg(2, 2);
    c.window = Some(12.0);
    c.min_weight = 0.0;
    let (m, rows) = fitted(c, 30, 23);
    assert!(
        m.view().is_some(),
        "rows have aged out: the boundary is read"
    );
    let state = serde_json::to_value(m.state()).unwrap();
    assert!(EwRidge::restore(&serde_json::from_value(state.clone()).unwrap()).is_ok());
    for key in [
        "wsig", "sig2", "wj", "my", "mj", "c", "m", "of", "mean", "var",
    ] {
        let mut v = state.clone();
        crate::window::json_edit(&mut v["model"]["EwRidge"]["win"], key, &mut |x| {
            x.as_array_mut().unwrap().pop();
        });
        match EwRidge::restore(&serde_json::from_value(v).unwrap()) {
            Err(StateError::Invalid(e)) => assert!(e.contains("snapshots"), "{key}: {e}"),
            Ok(mut back) => {
                let (x, y, w) = &rows[0];
                back.step(x, y, 1.0, *w);
                panic!("{key}: a snapshot of the wrong shape loaded and was read");
            }
            Err(e) => panic!("{key}: {e}"),
        }
    }
}

/// The live accumulators' widths too: a Gram whose means or co-moments are
/// short of its `k`, or target moments whose variances or Kish sums are a
/// target short, loaded -- `restore` held each Gram to its `k` alone and
/// the target moments to their means -- and the next row's update indexed
/// past them (found beside CA2, review 2026-10-06).
#[test]
fn a_live_gram_or_target_moment_of_the_wrong_width_is_refused() {
    let (m, rows) = fitted(cfg(2, 2), 30, 23);
    let state = serde_json::to_value(m.state()).unwrap();
    let places = [
        ("a Gram's means", "/acc/grams/grams/0/m"),
        ("a Gram's co-moments", "/acc/grams/grams/0/c"),
        ("the target variances", "/acc/tm/var"),
        ("the target Kish sums", "/acc/tm/q"),
    ];
    for (what, at) in places {
        let mut v = state.clone();
        v["model"]["EwRidge"]
            .pointer_mut(at)
            .and_then(serde_json::Value::as_array_mut)
            .unwrap_or_else(|| panic!("{what}: not in the state"))
            .pop();
        match EwRidge::restore(&serde_json::from_value(v).unwrap()) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
            Ok(mut back) => {
                let (x, y, w) = &rows[0];
                back.step(x, y, 1.0, *w);
                panic!("{what}: a short accumulator loaded and was read");
            }
            Err(e) => panic!("{what}: {e}"),
        }
    }
}

/// The systems kept for the per-row leverage are held to the cfg's shape:
/// a column index past the features, or a scale or a mean short of the
/// columns, loaded, and `row_error_inflation_into`, which runs on every row
/// the field is asked for, read `x[6]` or failed the factor's length
/// assertion (review 2026-10-06, CA4).
#[test]
fn a_kept_system_of_the_wrong_shape_is_refused() {
    let mut c = cfg(3, 1);
    c.min_weight = 0.0;
    let (mut m, _) = fitted(c, 40, 5);
    m.set_keep_factor(true);
    m.solve();
    let x = [0.3, 1.1, 2.4];
    let mut out = Vec::new();
    m.row_error_inflation_into(&x, 1.0, &mut out);
    assert!(out[0].is_finite(), "the fixture reads a leverage: {out:?}");
    assert!(EwRidge::restore(&m.state()).is_ok(), "the state as saved");
    type Damage<'a> = (&'a str, &'a dyn Fn(&mut System));
    let damage: [Damage; 3] = [
        ("a column index past the features", &|s| s.z[0] = 7),
        ("a scale short", &|s| {
            s.s.pop();
        }),
        ("a mean short", &|s| {
            s.mean.pop();
        }),
    ];
    for (what, f) in damage {
        let mut st = m.state();
        let ModelState::EwRidge(inner) = &mut st.model else {
            unreachable!()
        };
        f(inner
            .ready
            .systems
            .iter_mut()
            .flatten()
            .next()
            .expect("a kept system"));
        match EwRidge::restore(&st) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{what}: {e}"),
            Ok(back) => {
                back.row_error_inflation_into(&x, 1.0, &mut out);
                panic!("{what}: loaded and was read: {out:?}");
            }
            Err(e) => panic!("{what}: {e}"),
        }
    }
}

/// A state's configuration is held to what `new` holds a fresh one to: a
/// feature set naming a feature past the features changed none of the
/// widths the stream compares, loaded, and the next solve gathered
/// `cov.raw(6, 6)` of a 3x3 Gram; a half-life of 0 loaded too (review
/// 2026-10-06, CA6, CF4).
#[test]
fn a_state_whose_cfg_new_refuses_is_refused() {
    let mut c = cfg(2, 1);
    c.feature_sets = vec![("a".into(), vec![0, 1])];
    c.min_weight = 0.0;
    let (m, rows) = fitted(c, 20, 3);
    type Damage<'a> = (&'a str, &'a dyn Fn(&mut EwRidgeCfg));
    let damage: [Damage; 2] = [
        ("has out-of-range indices", &|c| c.feature_sets[0].1[1] = 5),
        ("half_life must be > 0", &|c| c.decay = Decay::Halflife(0.0)),
    ];
    for (says, f) in damage {
        let mut st = m.state();
        let ModelState::EwRidge(inner) = &mut st.model else {
            unreachable!()
        };
        f(&mut inner.cfg);
        assert!(inner.cfg.validate().is_err(), "{says}: `new` refuses it");
        match EwRidge::restore(&st) {
            Err(StateError::Invalid(e)) => {
                assert!(e.contains("configuration") && e.contains(says), "{e}");
            }
            Ok(mut back) => {
                for (x, y, w) in &rows {
                    back.step(x, y, 1.0, *w);
                }
                panic!("{says}: loaded and ran: {:?}", back.coefficients());
            }
            Err(e) => panic!("{says}: {e}"),
        }
    }
}

/// A weighted row of the oracle: features, two targets, the weight.
type OracleRow = (Vec<f64>, [f64; 2], f64);
/// A weighted row of [`fitted`]: features, the targets, the weight.
type FitRow = (Vec<f64>, Vec<Option<f64>>, f64);

/// Every solve against its closed form, from weighted sums written out
/// here (mutation baseline, docs/PLAN.md task 113): two targets and a
/// ridge grid of two (one under `ridge_scale`, which refuses a grid), so
/// each system's slot `j * nc + c` is its own; a
/// `coef_prior` that is not zero and a ridge that is not small, so the
/// penalty's centre moves every coefficient; under both target-gap
/// rules, where the targets read one Gram or one each. No decay, so the
/// weighted means are plain weighted averages:
///
/// - through the origin, raw: `(E[zz'] + λI) β = E[zy] + λ c0`;
/// - through the origin, standardized: the penalty per slot is `λ E[z_i²]`;
/// - `ridge_scale`: `(W E[zz'] + λI) β = W E[zy] + λ c0`, `W` the weight;
/// - with an intercept: the slopes from the weighted covariance, with
///   `λ` (plain) or `λ Var(x_i)` (standardized), and `β_0 = ȳ − m·β`.
#[test]
fn every_solve_is_its_closed_form_across_targets_and_ridges() {
    let (k, m) = (3usize, 2usize);
    let ridges = [0.7, 4.0];
    let prior = [vec![0.4, -1.0, 2.0, 0.5], vec![-0.3, 0.8, 0.1, -2.0]];
    let mut s = 20260927u64;
    let rows: Vec<OracleRow> = (0..60)
        .map(|_| {
            let x: Vec<f64> = (0..k)
                .map(|i| 2.0 * lcg(&mut s) + [0.5, -1.0, 3.0][i])
                .collect();
            let y0 = 1.0 + 2.0 * x[0] - x[1] + 0.5 * x[2] + 0.3 * lcg(&mut s);
            let y1 = -2.0 + 0.5 * x[0] + x[1] - 1.5 * x[2] + 0.3 * lcg(&mut s);
            (x, [y0, y1], 0.5 + (lcg(&mut s) + 1.0))
        })
        .collect();
    let wsum: f64 = rows.iter().map(|r| r.2).sum();
    let mean = |f: &dyn Fn(&OracleRow) -> f64| -> f64 {
        rows.iter().map(|r| r.2 * f(r)).sum::<f64>() / wsum
    };
    for (intercept, standardize, ridge_scale) in [
        (false, false, false),
        (false, true, false),
        (true, false, true),
        (true, false, false),
        (true, true, false),
    ] {
        for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
            let mut c = cfg(k, m);
            c.fit_intercept = intercept;
            c.standardize = standardize;
            c.ridge_scale = ridge_scale;
            // `ridge_scale` takes one ridge, never a grid.
            let grid: Vec<f64> = if ridge_scale {
                vec![ridges[1]]
            } else {
                ridges.to_vec()
            };
            c.ridge = grid.clone();
            c.target_gaps = gaps;
            c.min_weight = 0.0;
            let off = usize::from(intercept);
            c.coef_prior = Some(prior.iter().map(|p| p[1 - off..].to_vec()).collect());
            let mut model = EwRidge::new(c).unwrap();
            for (i, (x, y, w)) in rows.iter().enumerate() {
                model.step(
                    x,
                    &[Some(y[0]), Some(y[1])],
                    if i == 0 { 0.0 } else { 1.0 },
                    *w,
                );
            }
            let beta = model.coefficients().unwrap().to_vec();
            let x_new = [0.3, -0.7, 2.2];
            let pred = model.predict(&x_new, 1.0).pred;
            let kz = k + off;
            let z = |r: &OracleRow, i: usize| -> f64 {
                if intercept {
                    if i == 0 { 1.0 } else { r.0[i - 1] }
                } else {
                    r.0[i]
                }
            };
            for j in 0..m {
                let c0 = &prior[j][1 - off..];
                for (ci, &lam) in grid.iter().enumerate() {
                    let want: Vec<f64> = if intercept && !ridge_scale {
                        let mx: Vec<f64> = (0..k).map(|i| mean(&|r| r.0[i])).collect();
                        let my = mean(&|r| r.1[j]);
                        let cov =
                            |a: usize, b: usize| mean(&|r| (r.0[a] - mx[a]) * (r.0[b] - mx[b]));
                        let pen = |i: usize| if standardize { lam * cov(i, i) } else { lam };
                        let a: Vec<Vec<f64>> = (0..k)
                            .map(|a| {
                                (0..k)
                                    .map(|b| cov(a, b) + if a == b { pen(a) } else { 0.0 })
                                    .collect()
                            })
                            .collect();
                        let rhs: Vec<f64> = (0..k)
                            .map(|a| {
                                mean(&|r| (r.0[a] - mx[a]) * (r.1[j] - my)) + pen(a) * c0[a + 1]
                            })
                            .collect();
                        let slopes = oracle::solve(&a.concat(), &rhs);
                        let b0 = my - (0..k).map(|i| mx[i] * slopes[i]).sum::<f64>();
                        std::iter::once(b0).chain(slopes).collect()
                    } else {
                        let scale = if ridge_scale { wsum } else { 1.0 };
                        let pen = |i: usize| {
                            if standardize {
                                lam * mean(&|r| z(r, i) * z(r, i))
                            } else {
                                lam
                            }
                        };
                        let a: Vec<Vec<f64>> = (0..kz)
                            .map(|a| {
                                (0..kz)
                                    .map(|b| {
                                        scale * mean(&|r| z(r, a) * z(r, b))
                                            + if a == b { pen(a) } else { 0.0 }
                                    })
                                    .collect()
                            })
                            .collect();
                        let rhs: Vec<f64> = (0..kz)
                            .map(|a| scale * mean(&|r| z(r, a) * r.1[j]) + pen(a) * c0[a])
                            .collect();
                        oracle::solve(&a.concat(), &rhs)
                    };
                    let got = &beta[j * grid.len() + ci];
                    let zx: Vec<f64> = if intercept {
                        std::iter::once(1.0).chain(x_new).collect()
                    } else {
                        x_new.to_vec()
                    };
                    let p_want: f64 = zx.iter().zip(&want).map(|(a, b)| a * b).sum();
                    let p_got = pred[j * grid.len() + ci];
                    assert!(
                        (p_got - p_want).abs() <= 1e-9 * (1.0 + p_want.abs()),
                        "predict, target {j}, ridge {lam}: {p_got} against {p_want}"
                    );
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() <= 1e-9 * (1.0 + w.abs()),
                            "{intercept} {standardize} {ridge_scale} {gaps:?}: target {j}, \
                             ridge {lam}, slot {i}: {g} against the closed form {w}"
                        );
                    }
                }
            }
        }
    }
}

// --- the mutation baseline's survivors (docs/PLAN.md task 113) --------

/// A small fitted model: `k` features at levels, weights that move, no
/// decay, a solve every row.
fn fitted(c: EwRidgeCfg, n: usize, seed: u64) -> (EwRidge, Vec<FitRow>) {
    let (k, m) = (c.n_features, c.n_targets);
    let mut model = EwRidge::new(c).unwrap();
    let mut s = seed;
    let rows: Vec<_> = (0..n)
        .map(|i| {
            let x: Vec<f64> = (0..k).map(|j| 2.0 * lcg(&mut s) + j as f64).collect();
            let y = (0..m)
                .map(|t| {
                    let v = 1.0
                        + x.iter()
                            .enumerate()
                            .map(|(j, xj)| (j + t + 1) as f64 * 0.3 * xj)
                            .sum::<f64>()
                        + 0.2 * lcg(&mut s);
                    // The second target is absent on every third row.
                    (t == 0 || i % 3 != 1).then_some(v)
                })
                .collect::<Vec<_>>();
            (x, y, 0.5 + (lcg(&mut s) + 1.0))
        })
        .collect();
    for (i, (x, y, w)) in rows.iter().enumerate() {
        model.step(x, y, if i == 0 { 0.0 } else { 1.0 }, *w);
    }
    (model, rows)
}

#[test]
fn readiness_is_compared_bit_for_bit_and_field_by_field() {
    assert!(same_bits(&[f64::NAN], &[f64::NAN]), "a NaN is itself");
    assert!(!same_bits(&[0.0], &[-0.0]), "the bits, not the value");
    assert!(!same_bits(&[1.0], &[2.0]));
    assert!(!same_bits(&[1.0], &[1.0, 2.0]), "the lengths");
    let mut c = cfg(2, 1);
    c.min_weight = 0.0;
    let (mut m, _) = fitted(c, 30, 3);
    m.set_keep_factor(true);
    m.solve();
    // Stored, so the changes below reach the values a read gives rather
    // than the stale ones a pending solve stands over.
    m.settle_readiness();
    let base = m.ready.clone();
    assert_eq!(base, base.clone());
    assert!(!base.systems.is_empty() && !base.support.is_empty());
    let changes: [&dyn Fn(&mut Readiness); 5] = [
        &|r| r.edf[0] += 1.0,
        &|r| r.support.push(vec![]),
        // Slot 0 is the intercept, whose share is NaN by definition.
        &|r| r.support[0][1] += 1.0,
        &|r| r.systems[0] = None,
        &|r| r.system_of[0] += 1,
    ];
    for (i, change) in changes.iter().enumerate() {
        let mut r = base.clone();
        change(&mut r);
        assert_ne!(r, base, "change {i} went unseen");
    }
}

/// Slots whose solve's shares nothing has read yet (docs/PLAN.md task
/// 140).
fn unread(m: &EwRidge) -> usize {
    m.ready
        .pending
        .iter()
        .flatten()
        .filter(|p| p.shares.get().is_none())
        .count()
}

/// Slot-major floats as bits: NaN is itself, and `-0` is not `0`.
fn bits(v: Option<Vec<Vec<f64>>>) -> Option<Vec<u64>> {
    v.map(|v| v.into_iter().flatten().map(f64::to_bits).collect())
}

/// One configuration per solve path -- centred plain and standardized,
/// raw and scaled through the origin, `ridge_scale`, a window, a grid
/// over feature sets -- each solving every third row, so a solve waits
/// unread across rows, beside one solving every row. `fitted`'s second
/// target is absent on every third row, so it keeps a Gram of its own.
fn lazy_cfgs() -> Vec<(&'static str, EwRidgeCfg)> {
    let base = || {
        let mut c = cfg(3, 2);
        c.decay = Decay::Halflife(25.0);
        c.min_weight = 0.0;
        c.solve_every = 3.0;
        c.max_rows_between_solves = 100;
        c
    };
    let mut out = Vec::new();
    let mut c = base();
    c.ridge = vec![0.3];
    out.push(("centred plain", c));
    let mut c = base();
    c.standardize = true;
    c.ridge = vec![1e-8, 0.5];
    c.feature_sets = vec![("a".into(), vec![0, 1]), ("b".into(), vec![1, 2])];
    out.push(("centred standardized, a grid over feature sets", c));
    let mut c = base();
    c.fit_intercept = false;
    c.ridge = vec![0.2];
    out.push(("raw through the origin", c));
    let mut c = base();
    c.fit_intercept = false;
    c.standardize = true;
    out.push(("scaled through the origin", c));
    let mut c = base();
    c.ridge_scale = true;
    c.ridge = vec![3.0];
    out.push(("ridge_scale", c));
    let mut c = base();
    c.standardize = true;
    c.window = Some(12.0);
    out.push(("a window", c));
    let mut c = base();
    c.solve_every = 0.0;
    c.max_rows_between_solves = 1;
    out.push(("a solve every row", c));
    out
}

/// A solve's shares wait until something reads them (docs/PLAN.md task
/// 140), and they are read from the solve's own factor, so the
/// accumulators moving on cannot reach them: taken at the solve, rows
/// later, only at the end, stored at every row or written by a save,
/// they are the same bits, and so is the state.
#[test]
fn a_late_read_gives_the_bits_a_read_at_the_solve_gives() {
    for (name, c) in lazy_cfgs() {
        let (_, rows) = fitted(c.clone(), 60, 11);
        let new = || EwRidge::new(c.clone()).unwrap();
        // Read at every row, at every fifth, never, and stored at every
        // row.
        let (mut eager, mut late, mut never, mut stored) = (new(), new(), new(), new());
        let (mut infl, mut want_infl) = (Vec::new(), Vec::new());
        let mut read_late = 0;
        for (i, (x, y, w)) in rows.iter().enumerate() {
            let d = if i == 0 { 0.0 } else { 1.0 };
            for m in [&mut eager, &mut late, &mut never, &mut stored] {
                m.step(x, y, d, *w);
            }
            let want = bits(eager.support_coef());
            eager.error_inflation_into(&mut want_infl);
            stored.settle_readiness();
            assert!(
                stored.ready.pending.iter().all(Option::is_none),
                "{name}: settled, nothing waits"
            );
            assert_eq!(bits(stored.support_coef()), want, "{name}, row {i}");
            if i % 5 == 4 {
                // Unread, and from a solve rows back: the accumulators
                // have moved since.
                if late.rows_since_solve > 0 {
                    read_late += unread(&late);
                }
                assert_eq!(bits(late.support_coef()), want, "{name}, row {i}");
                late.error_inflation_into(&mut infl);
                assert!(same_bits(&infl, &want_infl), "{name}, row {i}");
            }
        }
        if c.max_rows_between_solves > 1 {
            assert!(read_late > 0, "{name}: no read came rows after its solve");
        }
        let waiting = unread(&never);
        assert!(waiting > 0, "{name}: nothing was left for the save to take");
        let bytes = |m: &EwRidge| rmp_serde::to_vec_named(&m.state()).unwrap();
        let want = bytes(&eager);
        assert_eq!(
            bytes(&never),
            want,
            "{name}: a save of {waiting} unread slots"
        );
        assert_eq!(bytes(&late), want, "{name}");
        assert_eq!(bytes(&stored), want, "{name}");
        assert!(
            never == eager,
            "{name}: compared on the values a read gives"
        );
        let back = EwRidge::restore(&rmp_serde::from_slice(&want).unwrap()).unwrap();
        assert_eq!(
            bits(back.support_coef()),
            bits(eager.support_coef()),
            "{name}"
        );
    }
}

/// The noise gate reads only which side of `max_error_inflation` a slot
/// is on, so where a solve is unread and the `edf` bound puts the ratio
/// below the limit, the bound stands in (docs/PLAN.md task 140). The
/// decision is the exact ratio's at every row and limit; the ratio is
/// exact wherever it reaches the limit, and so is the largest, which
/// the unreachable-gate warning names; the bound is never below the
/// exact ratio.
#[test]
fn the_gate_bound_decides_as_the_exact_ratio_would() {
    let (mut stood_in, mut read_at_limit) = (0, 0);
    for (name, c) in lazy_cfgs() {
        let (_, rows) = fitted(c.clone(), 60, 13);
        for limit in [1.05, std::f64::consts::SQRT_2, 3.0] {
            let mut m = EwRidge::new(c.clone()).unwrap();
            let (mut gate, mut exact) = (Vec::new(), Vec::new());
            for (i, (x, y, w)) in rows.iter().enumerate() {
                // The gate first: the exact read below takes the shares.
                let has = m.error_inflation_gate_into(&[], 1.0, &mut gate, limit);
                stood_in += unread(&m);
                assert_eq!(has, m.error_inflation_into(&mut exact));
                let at = format!("{name}, row {i}, limit {limit}: {gate:?} against {exact:?}");
                assert_eq!(gate.len(), exact.len(), "{at}");
                for (&g, &e) in gate.iter().zip(&exact) {
                    assert_eq!(g >= limit, e >= limit, "{at}");
                    if e >= limit {
                        assert_eq!(g.to_bits(), e.to_bits(), "{at}");
                        read_at_limit += usize::from(e.is_finite());
                    } else {
                        assert!(e <= g, "{at}");
                    }
                }
                let worst = |v: &[f64]| v.iter().cloned().fold(0.0, f64::max);
                if exact.iter().any(|&e| e >= limit) {
                    assert_eq!(worst(&gate).to_bits(), worst(&exact).to_bits(), "{at}");
                }
                m.step(x, y, if i == 0 { 0.0 } else { 1.0 }, *w);
            }
        }
    }
    assert!(
        stood_in > 0 && read_at_limit > 0,
        "the bound stood in for {stood_in} slots, the exact ratio was read at the limit \
         for {read_at_limit}: each branch must run"
    );
}

/// A share can be NaN: a factor faer accepts can still overflow its
/// inverse, and with no ridge `1 − 0·∞` is NaN. The exact ratio is then
/// infinite and the gate withholds, so the bound may stand in only where
/// the inverse is certain to come out as numbers
/// ([`SpdFactor::inverse_is_finite`]). Features near `1e-155`, raw and
/// through the origin, put the Gram near `1e-309`.
#[test]
fn a_share_that_is_not_a_number_withholds_whichever_way_the_gate_reads() {
    let mut c = cfg(3, 1);
    c.fit_intercept = false;
    c.ridge = vec![0.0];
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let limit = std::f64::consts::SQRT_2;
    let (mut gate, mut exact) = (Vec::new(), Vec::new());
    let (mut s, mut not_a_number) = (7, 0);
    for i in 0..40 {
        let x: Vec<f64> = (0..3).map(|j| 3e-155 * (lcg(&mut s) + j as f64)).collect();
        let y = x.iter().sum::<f64>() + 1e-156 * lcg(&mut s);
        m.error_inflation_gate_into(&[], 1.0, &mut gate, limit);
        m.error_inflation_into(&mut exact);
        assert_eq!(
            gate[0] >= limit,
            exact[0] >= limit,
            "row {i}: {gate:?} against {exact:?}"
        );
        if m.beta.is_some() && m.ready.edf_at(0).is_nan() {
            not_a_number += 1;
        }
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert!(
        not_a_number > 0,
        "no share came out NaN, so the case never ran"
    );
}

#[test]
fn the_kept_factors_follow_the_setting_and_survive_a_load() {
    let mut c = cfg(3, 1);
    c.min_weight = 0.0;
    let (mut m, _) = fitted(c, 40, 5);
    m.set_keep_factor(true);
    m.solve();
    let x = [0.3, 1.1, 2.4];
    let mut on = Vec::new();
    m.row_error_inflation_into(&x, 1.0, &mut on);
    assert!(on[0].is_finite() && on[0] > 1.0, "{on:?}");
    // A loaded model rebuilds the factors from the systems it carries:
    // through the bytes a file holds, which carry no factor (an
    // in-memory `State` clones them, and would hide a missing rebuild).
    let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
    let back = EwRidge::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    let mut loaded = Vec::new();
    back.row_error_inflation_into(&x, 1.0, &mut loaded);
    assert_eq!(loaded, on);
    // Off, nothing is kept and the answer is infinite.
    m.set_keep_factor(false);
    let mut off = Vec::new();
    m.row_error_inflation_into(&x, 1.0, &mut off);
    assert_eq!(off, vec![f64::INFINITY]);
}

/// A row whose feature is not a number has no leverage to read: its row
/// inflation is NaN in every slot, centred and standardized, raw and
/// through the origin. The quadratic form's clamp, `f64::max(NaN, 0.0)`,
/// read it as a form of 0, so the row reported the least inflation a row
/// can have, `sqrt(1 + 1/n)` with an intercept (task 181). The bank never
/// hands the model a NaN feature; the Rust API does.
#[test]
fn a_row_that_is_not_a_number_has_no_row_inflation() {
    for (fit_intercept, standardize, ridge_scale) in [
        (true, false, false),
        (true, true, false),
        (true, false, true),
        (false, false, false),
        (false, true, false),
    ] {
        let case = format!(
            "intercept {fit_intercept}, standardize {standardize}, ridge_scale {ridge_scale}"
        );
        let mut c = cfg(3, 2);
        c.min_weight = 0.0;
        c.fit_intercept = fit_intercept;
        c.standardize = standardize;
        c.ridge_scale = ridge_scale;
        // A grid of two ridges, but one under `ridge_scale`, which takes no grid.
        c.ridge = if ridge_scale {
            vec![3.0]
        } else {
            vec![1e-3, 0.5]
        };
        let slots = 2 * c.ridge.len();
        let (mut m, _) = fitted(c, 40, 13);
        m.set_keep_factor(true);
        m.solve();
        let mut out = Vec::new();
        m.row_error_inflation_into(&[0.3, 1.1, 2.4], 1.0, &mut out);
        assert_eq!(out.len(), slots, "{case}");
        assert!(
            out.iter().all(|v| v.is_finite() && *v >= 1.0),
            "{case}: {out:?}"
        );
        for x in [[f64::NAN, 1.1, 2.4], [0.3, 1.1, f64::NAN]] {
            m.row_error_inflation_into(&x, 1.0, &mut out);
            assert!(out.iter().all(|v| v.is_nan()), "{case}, {x:?}: {out:?}");
        }
    }
}

/// `sqrt(1 + h)`, `h` the row's leverage over Kish's sample size, from
/// the definition: centred and standardized, `h = (1 + v'(R + λI)⁻¹v)/n`
/// with `v` the row centred and scaled and `R` the correlation; under
/// `ridge_scale`, `h = W z'(W E[zz'] + λI)⁻¹z / n`.
#[test]
fn the_row_error_inflation_is_the_leverage_of_its_definition() {
    for ridge_scale in [false, true] {
        let mut c = cfg(3, 1);
        c.min_weight = 0.0;
        c.standardize = !ridge_scale;
        c.ridge_scale = ridge_scale;
        c.ridge = vec![if ridge_scale { 3.0 } else { 0.5 }];
        let lam = c.ridge[0];
        let mut m = EwRidge::new(c.clone()).unwrap();
        m.set_keep_factor(true);
        let (_, rows) = fitted(c, 50, 9);
        for (i, (x, y, w)) in rows.iter().enumerate() {
            m.step(x, y, if i == 0 { 0.0 } else { 1.0 }, *w);
        }
        m.solve();
        let x = [0.3, 1.1, 2.4];
        let mut got = Vec::new();
        m.row_error_inflation_into(&x, 1.0, &mut got);
        let ws: f64 = rows.iter().map(|r| r.2).sum();
        let wq: f64 = rows.iter().map(|r| r.2 * r.2).sum();
        let n = ws * ws / wq;
        let mean = |f: &dyn Fn(&[f64]) -> f64| rows.iter().map(|r| r.2 * f(&r.0)).sum::<f64>() / ws;
        let want = if ridge_scale {
            let z = |r: &[f64], i: usize| if i == 0 { 1.0 } else { r[i - 1] };
            let a: Vec<Vec<f64>> = (0..4)
                .map(|i| {
                    (0..4)
                        .map(|j| ws * mean(&|r| z(r, i) * z(r, j)) + if i == j { lam } else { 0.0 })
                        .collect()
                })
                .collect();
            let v: Vec<f64> = (0..4).map(|i| z(&x, i)).collect();
            let q = oracle::quad_form(&a.concat(), &v) * ws;
            (1.0 + q / n).sqrt()
        } else {
            let mx: Vec<f64> = (0..3).map(|i| mean(&|r| r[i])).collect();
            let cov = |a: usize, b: usize| mean(&|r| (r[a] - mx[a]) * (r[b] - mx[b]));
            let sd: Vec<f64> = (0..3).map(|i| cov(i, i).sqrt()).collect();
            let a: Vec<Vec<f64>> = (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| cov(i, j) / (sd[i] * sd[j]) + if i == j { lam } else { 0.0 })
                        .collect()
                })
                .collect();
            let v: Vec<f64> = (0..3).map(|i| (x[i] - mx[i]) / sd[i]).collect();
            let q = oracle::quad_form(&a.concat(), &v);
            (1.0 + (1.0 + q) / n).sqrt()
        };
        assert!(
            (got[0] - want).abs() <= 1e-10 * want,
            "ridge_scale {ridge_scale}: {} against {want}",
            got[0]
        );
    }
}

#[test]
fn the_error_inflation_has_a_slot_per_fit_and_waits_for_min_periods() {
    // Two targets and three ridges: `m * nc` = 6 where `m + nc` = 5.
    let mut c = cfg(2, 2);
    c.ridge = vec![1e-6, 0.1, 1.0];
    c.min_weight = 6.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 13u64;
    let mut out = Vec::new();
    for i in 0..8 {
        m.error_inflation_into(&mut out);
        assert_eq!(out.len(), 6);
        // Solved from the second row on, but withheld below the floor,
        // whose check is `<`: at exactly six rows of weight 1 it is met.
        if i < 6 {
            assert!(out.iter().all(|v| *v == f64::INFINITY), "row {i}: {out:?}");
        } else {
            assert!(
                out.iter().all(|v| v.is_finite() && *v > 1.0),
                "row {i}: {out:?}"
            );
            assert!(m.beta.is_some());
        }
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0]), Some(x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    // A state whose readiness did not come with it answers infinity; the
    // statistic is never indexed past what it holds.
    m.ready.edf.clear();
    m.error_inflation_into(&mut out);
    assert!(out.iter().all(|v| *v == f64::INFINITY), "{out:?}");
}

#[test]
fn each_target_reports_its_own_weight() {
    let mut c = cfg(2, 2);
    c.min_weight = 0.0;
    let (m, rows) = fitted(c, 30, 17);
    let mut out = vec![-1.0];
    assert!(m.target_n_eff_into(&mut out));
    let want: Vec<f64> = (0..2)
        .map(|t| rows.iter().filter(|r| r.1[t].is_some()).map(|r| r.2).sum())
        .collect();
    assert_eq!(out.len(), 2);
    for (g, w) in out.iter().zip(&want) {
        assert!((g - w).abs() <= 1e-12 * w, "{out:?} against {want:?}");
    }
    assert!(want[1] < want[0], "the second target missed rows");
}

#[test]
fn a_window_budget_reports_its_overrun_and_clears() {
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(30.0);
    c.window = Some(40.0);
    c.max_rows_between_snapshots = Some(2);
    c.min_weight = 0.0;
    let (mut m, _) = fitted(c, 100, 21);
    assert_eq!(m.window_over_budget(), None, "no budget, no overrun");
    m.set_window_budget(Some(crate::WindowBudget::Refuse(1e-6)));
    match m.window_over_budget() {
        Some((bytes, every)) => {
            assert!(bytes > 1, "{bytes}");
            assert_eq!(every, crate::Cadence::of(None, Some(2)));
        }
        None => panic!("a ring of snapshots is past a budget of one byte"),
    }
    m.set_window_budget(None);
    assert_eq!(m.window_over_budget(), None);
}

#[test]
fn the_gram_parts_are_the_fits_and_carry_the_target_moments() {
    let mut c = cfg(2, 2);
    c.min_weight = 0.0;
    let (m, rows) = fitted(c, 30, 23);
    let (parts, moments) = m.gram_parts();
    assert_eq!(parts.len(), 2, "one Gram per target under own_rows");
    assert_eq!(parts[0].targets, vec![0]);
    let w0: f64 = rows.iter().map(|r| r.2).sum();
    assert!((parts[0].cov.n_eff() - w0).abs() <= 1e-12 * w0);
    assert!(
        moments.is_some(),
        "no window, so the history's moments are given"
    );
}

/// The doc: re-solve "when anything was mixed and there is a fit to
/// replace". Before the first fit there is none, so a blend solves
/// nothing.
#[test]
fn a_blend_before_the_first_fit_solves_nothing() {
    let mut c = cfg(2, 1);
    c.long_half_life = Some(500.0);
    c.session_shrink = Some(0.5);
    // The first fit waits for `min_weight` whatever the schedule.
    c.min_weight = 100.0;
    c.solve_every = 1e9;
    c.max_rows_between_solves = 100_000;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 29u64;
    for i in 0..5 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    assert!(m.coefficients().is_none(), "no solve scheduled yet");
    m.blend_toward_long_run();
    assert!(
        m.coefficients().is_none(),
        "the blend solved with no fit to replace"
    );
}

/// Kish's sample size and the residual spread under a window are those
/// of the rows inside it, from the definition: the rows less than one
/// window old (at most one, under `closed = "both"`), weighted
/// `2^(-age / half_life)`.
#[test]
fn the_window_kish_and_spread_are_the_rows_inside_it() {
    for closed in [crate::WindowClosed::Right, crate::WindowClosed::Both] {
        window_kish_and_spread_are_the_rows_inside(closed);
    }
}

fn window_kish_and_spread_are_the_rows_inside(closed: crate::WindowClosed) {
    let (half_life, window) = (30.0, 80.0);
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(half_life);
    c.window = Some(window);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    m.set_window_closed(closed);
    let mut s = 31u64;
    let (mut t, mut resid, mut clock) = (vec![], vec![], 0.0);
    for i in 0..150 {
        let d = if i == 0 {
            0.0
        } else {
            0.25 * (2.0 + (lcg(&mut s).abs() * 8.0).floor())
        };
        clock += d;
        let x = [lcg(&mut s) * 4.0 - 2.0];
        let y = 3.0 * x[0] + 1.0 + 0.5 * lcg(&mut s);
        let pred = m.step(&x, &[Some(y)], d, 1.0).pred[0];
        t.push(clock);
        resid.push(pred.is_finite().then_some(y - pred));
        if i < 20 {
            continue;
        }
        let kept: Vec<(f64, Option<f64>)> = (0..=i)
            .filter(|&r| inside(clock - t[r], window, closed))
            .map(|r| (0.5_f64.powf((clock - t[r]) / half_life), resid[r]))
            .collect();
        let (ws, wq) = kept
            .iter()
            .fold((0.0, 0.0), |(a, b), (w, _)| (a + w, b + w * w));
        let kish = m.gram_kish()[0].expect("a window with rows in it");
        assert!(
            (kish - ws * ws / wq).abs() <= 1e-8 * kish,
            "row {i}: {kish} against {}",
            ws * ws / wq
        );
        let (rs, rw) = kept
            .iter()
            .filter_map(|(w, r)| r.map(|r| (w * r * r, *w)))
            .fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
        let sig2 = m.sigma2()[0];
        assert!(
            (sig2 - rs / rw).abs() <= 1e-8 * sig2,
            "row {i}: {sig2} against {}",
            rs / rw
        );
    }
}

/// The target moments under a window are those of the rows inside it,
/// each target over the rows it was present on (docs/PLAN.md task 136):
/// its weight, mean, variance about the mean and `Q = Σw²`, weighted
/// `2^(-age / half_life)`. They were `None` under a window, since the
/// snapshots held none; the second target is missing on every third row.
/// Under either edge.
#[test]
fn the_windowed_target_moments_are_the_rows_inside_it() {
    for closed in [crate::WindowClosed::Right, crate::WindowClosed::Both] {
        windowed_target_moments_are_the_rows_inside(closed);
    }
}

fn windowed_target_moments_are_the_rows_inside(closed: crate::WindowClosed) {
    let (half_life, window) = (30.0, 80.0);
    let mut c = cfg(1, 2);
    c.decay = Decay::Halflife(half_life);
    c.window = Some(window);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    m.set_window_closed(closed);
    let mut s = 37u64;
    let (mut t, mut ys, mut clock, mut checked) = (vec![], vec![], 0.0, 0);
    for i in 0..160 {
        let d = if i == 0 {
            0.0
        } else {
            0.25 * (2.0 + (lcg(&mut s).abs() * 8.0).floor())
        };
        clock += d;
        let x = [lcg(&mut s) * 4.0 - 2.0];
        let y0 = 3.0 * x[0] + 1.0 + 0.5 * lcg(&mut s);
        let y1 = (i % 3 != 0).then(|| 50.0 - x[0] + lcg(&mut s));
        m.step(&x, &[Some(y0), y1], d, 1.0);
        t.push(clock);
        ys.push([Some(y0), y1]);
        if i < 30 {
            continue;
        }
        let (parts, tm) = m.gram_parts();
        let tm = tm.expect("the snapshots carry the target moments");
        for j in [0usize, 1] {
            let kept: Vec<(f64, f64)> = (0..=i)
                .filter(|&r| inside(clock - t[r], window, closed))
                .filter_map(|r| ys[r][j].map(|y| (0.5f64.powf((clock - t[r]) / half_life), y)))
                .collect();
            let w: f64 = kept.iter().map(|k| k.0).sum();
            let q: f64 = kept.iter().map(|k| k.0 * k.0).sum();
            let mean = kept.iter().map(|k| k.0 * k.1).sum::<f64>() / w;
            let var = kept.iter().map(|k| k.0 * (k.1 - mean).powi(2)).sum::<f64>() / w;
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-8 * b.abs().max(1.0);
            assert!(close(tm.means()[j], mean), "row {i}, target {j}: mean");
            assert!(
                close(tm.vars()[j], var),
                "row {i}, target {j}: {} against {var}",
                tm.vars()[j]
            );
            assert!(close(tm.q()[j], q), "row {i}, target {j}: Q");
            let part = parts.iter().find(|p| p.targets.contains(&j)).unwrap();
            let at = part.targets.iter().position(|&x| x == j).unwrap();
            assert!(close(part.target_weights[at], w), "row {i}, target {j}: W");
        }
        checked += 1;
    }
    assert!(checked > 100);
}

/// What the target moments add to a window's snapshot (docs/PLAN.md task
/// 136, the user's condition that it not add much): exactly `4·T`
/// doubles, the mean, variance and `Q` of each target and what its mean
/// leaves out, beside the Gram's `k_total²` and the cross-moments. The
/// fourth since review 2026-10-05, CB2: the truncation reads the two
/// means' difference as the pairs'. The shares are printed for PLAN.
#[test]
fn the_target_moments_add_four_doubles_a_target_to_a_snapshot() {
    for (k, targets) in [
        (1usize, 1usize),
        (5, 1),
        (10, 1),
        (10, 5),
        (50, 1),
        (50, 10),
    ] {
        let mut c = cfg(k, targets);
        c.window = Some(1e9);
        c.min_weight = 0.0;
        c.target_gaps = crate::TargetGaps::Pairwise;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 5u64;
        for i in 0..3 {
            let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
            let y: Vec<Option<f64>> = (0..targets).map(|_| Some(lcg(&mut s))).collect();
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let snap = m.acc.snapshot(1.0);
        let with = crate::Footprint::footprint(&snap);
        let mut bare = snap.clone();
        bare.tm = None;
        let without = crate::Footprint::footprint(&bare);
        assert_eq!(with - without, 4 * targets * 8, "k = {k}, T = {targets}");
        println!(
            "k = {k}, T = {targets}: {without} -> {with} bytes, +{:.1} %",
            100.0 * (with - without) as f64 / without as f64
        );
    }
}

/// Each target's spread is its first fit's squared residuals over the
/// rows it learned, weighted -- not another target's, not another
/// ridge's. A row of weight 0 at the first prediction, while the spread
/// holds no weight at all, leaves it alone (hard rule 9: no 0/0).
#[test]
fn each_targets_spread_is_its_first_fits_residuals() {
    let mut c = cfg(2, 2);
    c.ridge = vec![1e-6, 2.0];
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 37u64;
    let mut acc = [(0.0, 0.0); 2];
    for i in 0..60 {
        let x = [lcg(&mut s), lcg(&mut s) + 1.0];
        let y = [
            Some(1.0 + x[0] - x[1] + 0.3 * lcg(&mut s)),
            // The second target starts late, so while the first is
            // predicted it is not: a row it is present on must not read
            // the first's prediction.
            (i >= 8 && i % 4 != 2).then(|| 2.0 * x[1] + 0.3 * lcg(&mut s)),
        ];
        let w = if i == 3 {
            0.0
        } else if i < 3 {
            1.0
        } else {
            0.5 + (lcg(&mut s) + 1.0)
        };
        let pred = m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, w).pred;
        if i == 3 {
            assert!(pred[0].is_finite(), "the first prediction is this row's");
        }
        for j in 0..2 {
            if let Some(yj) = y[j] {
                let p = pred[j * 2];
                if w > 0.0 && p.is_finite() {
                    acc[j].0 += w * (yj - p) * (yj - p);
                    acc[j].1 += w;
                }
            }
        }
    }
    let sig2 = m.sigma2();
    for j in 0..2 {
        let want = acc[j].0 / acc[j].1;
        assert!(
            (sig2[j] - want).abs() <= 1e-10 * want,
            "target {j}: {} against {want}",
            sig2[j]
        );
    }
}

/// A state naming schema 17 is refused by its version, where its own means
/// were read as offsets (docs/PLAN.md task 198: the floor is the schema
/// shipped); and a twin of another width is refused.
#[test]
fn a_schema_17_state_is_refused_and_so_is_a_mismatched_twin() {
    let mut c = cfg(2, 2);
    c.target_gaps = TargetGaps::Pairwise;
    c.min_weight = 0.0;
    let (m, _) = fitted(c, 60, 41);
    let mut s17 = m.state();
    s17.schema_version = 17;
    assert!(matches!(
        EwRidge::restore(&s17),
        Err(StateError::SchemaVersion { found: 17, .. })
    ));
    let twin = |k: usize| {
        let mut c = cfg(k, 1);
        c.long_half_life = Some(200.0);
        c.session_shrink = Some(0.5);
        c.min_weight = 0.0;
        fitted(c, 20, 47).0
    };
    let (mut two, three) = (twin(2), twin(3));
    two.slow = three.slow.clone();
    assert!(
        EwRidge::restore(&two.state()).is_err(),
        "a twin of three features beside two"
    );
}

#[test]
fn a_window_that_is_not_a_positive_length_is_refused_by_name() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut c = cfg(1, 1);
        c.window = Some(bad);
        let err = EwRidge::new(c).expect_err("refused");
        assert!(
            err.contains("window_size must be finite and > 0"),
            "{bad}: {err}"
        );
    }
}

/// Standardized, a feature that never moves has no scale and gets no
/// slope, and when no feature moves the fit is the target's mean: the
/// intercept alone. Its system keeps no column, so a row's leverage is the
/// mean's own `1 / n_kish` and the row reads `sqrt(1 + 1 / n_kish)`, with
/// `n_kish = (Σ w)² / Σ w²` over the rows' weights (the doc; docs/PLAN.md
/// task 218). It read infinite, from the factor a live solve keeps none of
/// for a system with no column, which this test pinned as "no system kept"
/// though the system was kept; a load factored it, and read the mean's.
#[test]
fn with_no_feature_moving_the_fit_is_the_mean_and_reads_its_own_leverage() {
    let mut c = cfg(2, 1);
    c.standardize = true;
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    m.set_keep_factor(true);
    let mut s = 53u64;
    let (mut ys, mut ws, mut wq) = (0.0, 0.0, 0.0);
    for i in 0..30 {
        let y = 5.0 + lcg(&mut s);
        let w = 0.5 + (lcg(&mut s) + 1.0);
        m.step(&[3.0, -2.0], &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, w);
        ys += w * y;
        ws += w;
        wq += w * w;
    }
    let beta = &m.coefficients().unwrap()[0];
    assert!((beta[0] - ys / ws).abs() <= 1e-12 * (ys / ws), "{beta:?}");
    assert_eq!(&beta[1..], &[0.0, 0.0]);
    let mut out = Vec::new();
    m.row_error_inflation_into(&[3.0, -2.0], 1.0, &mut out);
    let want = (1.0 + wq / (ws * ws)).sqrt();
    assert!((out[0] - want).abs() <= 1e-12, "{out:?} against {want}");
}

/// Under `pairwise` the Gram is over every row, and a target's
/// intercept is `ȳ_j − m_j·β` with `m_j` the feature means over the
/// rows that target was present on -- its own, not the Gram's.
#[test]
fn a_pairwise_intercept_is_centred_on_the_targets_own_rows() {
    let mut c = cfg(2, 2);
    c.target_gaps = TargetGaps::Pairwise;
    c.min_weight = 0.0;
    let (m, rows) = fitted(c, 60, 59);
    let beta = &m.coefficients().unwrap()[1];
    let own: Vec<_> = rows.iter().filter(|r| r.1[1].is_some()).collect();
    let w: f64 = own.iter().map(|r| r.2).sum();
    let my = own.iter().map(|r| r.2 * r.1[1].unwrap()).sum::<f64>() / w;
    let mx: Vec<f64> = (0..2)
        .map(|i| own.iter().map(|r| r.2 * r.0[i]).sum::<f64>() / w)
        .collect();
    let want = my - mx[0] * beta[1] - mx[1] * beta[2];
    assert!(
        (beta[0] - want).abs() <= 1e-10 * (1.0 + want.abs()),
        "{} against {want}",
        beta[0]
    );
    let all: f64 =
        rows.iter().map(|r| r.2 * r.0[0]).sum::<f64>() / rows.iter().map(|r| r.2).sum::<f64>();
    assert!(
        (all - mx[0]).abs() > 1e-3,
        "the target's rows are not every row's mean"
    );
}

/// A zero-weight copy of every row changes nothing, to the bit: the
/// targets' own means are pairs, and a pair given a step of zero could
/// round afresh (`crate::comp::add`), so a row of weight 0 takes none.
#[test]
fn a_zero_weight_copy_of_every_row_changes_nothing() {
    for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
        let mut c = cfg(2, 2);
        c.target_gaps = gaps;
        c.decay = Decay::Halflife(15.0);
        c.min_weight = 0.0;
        let (mut plain, mut doubled) = (EwRidge::new(c.clone()).unwrap(), EwRidge::new(c).unwrap());
        let mut s = 61u64;
        for i in 0..200 {
            // Features centred near zero, where a step can exceed the mean.
            let x = [0.01 * lcg(&mut s), 3.0 * lcg(&mut s)];
            let y = [Some(x[0] - x[1]), (i % 3 != 0).then(|| 0.5 * x[1])];
            let d = if i == 0 { 0.0 } else { 1.0 };
            let a = plain.step(&x, &y, d, 1.0).pred;
            let b = doubled.step(&x, &y, d, 1.0).pred;
            let copy = doubled.step(&x, &y, 0.0, 0.0).pred;
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b), "{gaps:?} row {i}");
            let after = plain.clone().step(&x, &[None, None], 0.0, 0.0).pred;
            assert_eq!(
                bits(&copy),
                bits(&after),
                "{gaps:?} row {i}: the copy's prediction"
            );
        }
        assert_eq!(plain.acc.cross, doubled.acc.cross, "{gaps:?}");
    }
}

/// Two identical features at no ridge: the system is singular, the
/// solve adds a jitter `μ`, and each copy's share of its coefficient is
/// `1 / (2 + μ)`, about a half -- the diagonal of `I − μ(A + μI)⁻¹` for
/// `A = [[1, 1], [1, 1]]`. A share below 1 at a ridge of 0 is itself the
/// evidence that the jitter was applied.
#[test]
fn a_duplicated_feature_shares_its_support_at_the_jitter() {
    let mut c = cfg(2, 1);
    c.standardize = true;
    c.ridge = vec![0.0];
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 67u64;
    for i in 0..40 {
        let v = lcg(&mut s);
        m.step(
            &[v, v],
            &[Some(2.0 * v + 0.1 * lcg(&mut s))],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let support = &m.support_coef().unwrap()[0];
    for &share in &support[1..] {
        assert!((share - 0.5).abs() < 1e-3, "{support:?}");
    }
}

/// The row leverage per slot, two targets on their own rows and three
/// ridges, against its definition (centred, plain): `h = (1 +
/// v'(C_j + λ_c I)⁻¹v) / n_j` over target `j`'s rows. With the floor
/// check at `<`, and nothing indexed past what the readiness holds.
#[test]
fn the_row_error_inflation_has_each_slots_own_leverage() {
    let ridges = [1e-6, 0.3, 2.0];
    let mut c = cfg(2, 2);
    c.ridge = ridges.to_vec();
    c.min_weight = 12.0;
    let mut m = EwRidge::new(c).unwrap();
    m.set_keep_factor(true);
    let mut s = 71u64;
    let mut rows: Vec<(Vec<f64>, [Option<f64>; 2])> = Vec::new();
    let x_new = [0.4, 1.9];
    let mut out = Vec::new();
    for i in 0..40 {
        m.row_error_inflation_into(&x_new, 1.0, &mut out);
        assert_eq!(out.len(), 6);
        if i < 12 {
            assert!(out.iter().all(|v| *v == f64::INFINITY), "row {i}: {out:?}");
        } else {
            assert!(out.iter().all(|v| v.is_finite()), "row {i}: {out:?}");
        }
        let x = vec![lcg(&mut s), 2.0 * lcg(&mut s) + 1.0];
        let y = [
            Some(x[0] + 0.3 * lcg(&mut s)),
            (i % 3 != 1).then(|| x[1] - x[0] + 0.3 * lcg(&mut s)),
        ];
        m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        rows.push((x, y));
    }
    m.row_error_inflation_into(&x_new, 1.0, &mut out);
    assert_eq!(m.support_coef().map(|v| v.len()), Some(6));
    for j in 0..2 {
        let own: Vec<&Vec<f64>> = rows
            .iter()
            .filter(|r| r.1[j].is_some())
            .map(|r| &r.0)
            .collect();
        let n = own.len() as f64;
        let mx: Vec<f64> = (0..2)
            .map(|i| own.iter().map(|x| x[i]).sum::<f64>() / n)
            .collect();
        let cov = |a: usize, b: usize| {
            own.iter()
                .map(|x| (x[a] - mx[a]) * (x[b] - mx[b]))
                .sum::<f64>()
                / n
        };
        for (ci, lam) in ridges.iter().enumerate() {
            let a: Vec<Vec<f64>> = (0..2)
                .map(|i| {
                    (0..2)
                        .map(|k| cov(i, k) + if i == k { *lam } else { 0.0 })
                        .collect()
                })
                .collect();
            let v: Vec<f64> = (0..2).map(|i| x_new[i] - mx[i]).collect();
            let q = oracle::quad_form(&a.concat(), &v);
            let want = (1.0 + (1.0 + q) / n).sqrt();
            let got = out[j * 3 + ci];
            assert!(
                (got - want).abs() <= 1e-9 * want,
                "target {j}, ridge {lam}: {got} against {want}"
            );
        }
    }
    m.ready.system_of.clear();
    m.row_error_inflation_into(&x_new, 1.0, &mut out);
    assert!(out.iter().all(|v| *v == f64::INFINITY), "{out:?}");
}

/// Under `ridge_scale` the system is on the sum scale, `W E[zz'] + λI`,
/// and a slope's share of its coefficient is `1 − λ [(W E[zz'] +
/// λI)⁻¹]_ii`: the penalty the readiness reads is `λ`, as solved.
#[test]
fn the_support_under_ridge_decay_is_the_sum_scale_systems() {
    let lam = 5.0;
    let mut c = cfg(2, 1);
    c.ridge = vec![lam];
    c.ridge_scale = true;
    c.min_weight = 0.0;
    let (m, rows) = fitted(c, 25, 73);
    let z = |x: &[f64], i: usize| if i == 0 { 1.0 } else { x[i - 1] };
    let a: Vec<Vec<f64>> = (0..3)
        .map(|i| {
            (0..3)
                .map(|j| {
                    rows.iter()
                        .map(|r| r.2 * z(&r.0, i) * z(&r.0, j))
                        .sum::<f64>()
                        + if i == j { lam } else { 0.0 }
                })
                .collect()
        })
        .collect();
    let support = &m.support_coef().unwrap()[0];
    let inv = oracle::inverse(&a.concat());
    for (i, &share) in support.iter().enumerate().skip(1) {
        let want = 1.0 - lam * inv[i * 3 + i];
        assert!(
            (share - want).abs() <= 1e-9,
            "slot {i}: {share} against {want}"
        );
    }
}

/// A combo with no ridge and no weight in its Gram is skipped, not
/// solved -- under `own_rows`, a target whose Gram a gap past the
/// underflow emptied and the next row did not refill -- and its slot has
/// no fit, NaN, where it kept the zeros of an empty matrix's fit and the
/// target's next rows were predicted as 0.0 (review round 4, CC1). It
/// reports no readiness shares either: the previous solve's `edf` and
/// data shares stood there, so a `coef` row read a share of data under
/// zero coefficients (review 2026-10-05, CA6). Ridge 0, a half-life of
/// one clock unit, a gap of 1100 of them (2^-1100 underflows to 0), then
/// the second target absent.
#[test]
fn a_combo_skipped_for_no_weight_reports_no_shares() {
    let mut c = cfg(1, 2);
    c.ridge = vec![0.0];
    c.decay = Decay::Halflife(1.0);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 41u64;
    for i in 0..40 {
        let x = [lcg(&mut s)];
        let y1 = (i % 3 != 1).then(|| 2.0 * x[0] + 0.1 * lcg(&mut s));
        m.step(
            &x,
            &[Some(1.0 - x[0]), y1],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let before = m.support_coef().expect("solved");
    assert!(
        before[1][1].is_finite(),
        "the case needs a share: {before:?}"
    );
    let x = [lcg(&mut s)];
    m.step(&x, &[Some(1.0 - x[0]), None], 1100.0, 1.0);
    let g1 = m.acc.grams.of[1];
    assert_eq!(
        m.acc.grams.grams[g1].n_eff(),
        0.0,
        "the gap emptied its Gram"
    );
    let beta = m.coefficients().unwrap();
    assert!(beta[1].iter().all(|v| v.is_nan()), "no fit: {beta:?}");
    let shares = m.support_coef().unwrap();
    assert!(
        shares[1].iter().all(|v| v.is_nan()),
        "no shares: {shares:?}"
    );
    assert!(shares[0][1].is_finite(), "the solved target keeps its own");
    let mut inflation = Vec::new();
    assert!(OnlineModel::error_inflation_into(&m, &mut inflation));
    assert_eq!(inflation[1], f64::INFINITY, "{inflation:?}");
}

/// The same with two combos, the one without a ridge second, so the slot
/// skipped is the second target's second, `j * nc + ci = 3` of four, and
/// not a slot one combo or the first target would name: it reports no fit
/// and no shares, and the first target's combo without a ridge keeps its
/// own. (The second target's other slot has neither either, as a target
/// with no row: CC1.)
#[test]
fn a_combo_skipped_for_no_weight_is_its_own_slot_among_several() {
    let mut c = cfg(1, 2);
    c.ridge = vec![0.5, 0.0];
    c.decay = Decay::Halflife(1.0);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 41u64;
    for i in 0..40 {
        let x = [lcg(&mut s)];
        let y1 = (i % 3 != 1).then(|| 2.0 * x[0] + 0.1 * lcg(&mut s));
        m.step(
            &x,
            &[Some(1.0 - x[0]), y1],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let before = m.support_coef().expect("solved");
    assert!(
        before[3][1].is_finite(),
        "the case needs a share: {before:?}"
    );
    let x = [lcg(&mut s)];
    m.step(&x, &[Some(1.0 - x[0]), None], 1100.0, 1.0);
    assert_eq!(m.acc.grams.grams[m.acc.grams.of[1]].n_eff(), 0.0, "emptied");
    let shares = m.support_coef().unwrap();
    assert!(shares[3].iter().all(|v| v.is_nan()), "{shares:?}");
    assert!(
        shares[1][1].is_finite(),
        "the first target's own: {shares:?}"
    );
    let beta = m.coefficients().unwrap();
    assert!(beta[3].iter().all(|v| v.is_nan()), "no fit: {beta:?}");
    assert!(beta[1].iter().all(|v| v.is_finite()), "{beta:?}");
}

/// A first solve that fails leaves no fit: every slot it could not solve
/// holds NaN, so it predicts nothing, and the slots it solved hold their
/// fit. The fit was the zeros `beta` starts at, so the slot predicted
/// 0.0 as though it had learned it (review 2026-10-05, CA5). A NaN in the
/// Gram, which no jitter factorizes and the next rows leave there, makes
/// it fail. A NaN feature on the first row put it there; every model
/// refuses one now (task 183), so it is written into the accumulator, as
/// `a_solve_that_fails_keeps_the_last_fit_or_says_nan` writes it. Before
/// the first solve there is no fit at all, as before.
#[test]
fn a_first_solve_that_fails_leaves_no_fit() {
    let mut c = cfg(2, 1);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    assert!(m.coefficients().is_none(), "no fit before the first solve");
    assert!(m.predict(&[0.5, 1.0], 0.0).pred[0].is_nan());
    let g = m.acc.grams.of[0];
    m.acc.grams.grams[g].update(&[f64::NAN; 3], 1.0, 1.0);
    m.step(&[0.5, 1.0], &[Some(2.0)], 0.0, 1.0);
    assert!(m.solve_failures > 0, "the first solve failed");
    let beta = m.coefficients().expect("a solve ran");
    assert!(beta[0].iter().all(|v| v.is_nan()), "{beta:?}");
    for i in 1..5 {
        let p = m.step(&[0.5, f64::from(i)], &[Some(2.0)], 1.0, 1.0).pred[0];
        assert!(
            p.is_nan(),
            "row {i}: a fit that failed predicts nothing, {p}"
        );
    }
    assert!(m.predict(&[0.5, 1.0], 1.0).pred[0].is_nan());
    // Two targets and two ridges: every one of the four slots, each
    // `j * nc + ci`, holds NaN, and none keeps the zeros it started at.
    let mut c = cfg(2, 2);
    c.ridge = vec![0.1, 1.0];
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let g = m.acc.grams.of[0];
    m.acc.grams.grams[g].update(&[f64::NAN; 3], 1.0, 1.0);
    m.step(&[0.5, 1.0], &[Some(2.0), Some(1.0)], 0.0, 1.0);
    assert!(m.solve_failures > 0, "the first solve failed");
    let beta = m.coefficients().expect("a solve ran");
    assert_eq!(beta.len(), 4);
    assert!(beta.iter().flatten().all(|v| v.is_nan()), "{beta:?}");
}

/// A system that cannot be factorized keeps the fit it had, slot by
/// slot, and is counted; with no fit before, the slot is NaN
/// (`a_first_solve_that_fails_leaves_no_fit`). The other target's fit is
/// solved as usual.
#[test]
fn a_solve_that_fails_keeps_the_last_fit_or_says_nan() {
    let mut c = cfg(2, 2);
    c.ridge = vec![1e-6, 0.5];
    c.min_weight = 0.0;
    let (mut m, _) = fitted(c.clone(), 30, 79);
    let before = m.coefficients().unwrap().to_vec();
    let failed = m.solve_failures;
    let g = m.acc.grams.of[1];
    m.acc.grams.grams[g].update(&[f64::NAN; 3], 1.0, 1.0);
    m.solve();
    assert_eq!(
        m.solve_failures,
        failed + 2,
        "both ridges of the second target failed"
    );
    let after = m.coefficients().unwrap();
    for ci in 0..2 {
        assert_eq!(
            after[2 + ci],
            before[2 + ci],
            "ridge {ci}: the second target keeps its fit"
        );
        assert!(after[ci].iter().all(|v| v.is_finite()));
    }
}

/// A window that holds no row of a target reports no fit for it, in
/// each of its ridges' slots, and only its (review 2026-09-12, C2).
#[test]
fn a_target_absent_from_the_window_has_no_fit() {
    let mut c = cfg(2, 2);
    c.ridge = vec![1e-6, 0.5];
    c.decay = Decay::Halflife(20.0);
    c.window = Some(10.0);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 97u64;
    for i in 0..40 {
        let x = [lcg(&mut s), lcg(&mut s)];
        // The second target stops at row 20; the window is 10 rows.
        let y1 = (i < 20).then_some(x[1]);
        m.step(&x, &[Some(x[0]), y1], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let beta = m.coefficients().unwrap();
    for ci in 0..2 {
        assert!(beta[ci].iter().all(|v| v.is_finite()), "{:?}", beta[ci]);
        assert!(
            beta[2 + ci].iter().all(|v| v.is_nan()),
            "{:?}",
            beta[2 + ci]
        );
    }
}

/// A target first seen after the last solve has weight but no statistic
/// yet: its inflation is infinite, never `sqrt(1 + NaN)`. Its own
/// `min_weight` is above its one row, so its own first solve (task 195,
/// S9b) has not come either.
#[test]
fn a_target_seen_since_the_last_solve_reads_infinite_inflation() {
    let mut c = cfg(2, 2);
    // With a ridge, a Gram with no weight is solved to the prior and
    // has a statistic; with none, it is skipped and has none.
    c.ridge = vec![0.0];
    c.min_weight = 0.0;
    c.solve_every = 1e9;
    c.max_rows_between_solves = 5;
    let mut m = EwRidge::new(c).unwrap();
    m.set_target_min_weight(vec![0.0, 3.0]).unwrap();
    let mut s = 89u64;
    for i in 0..7 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let y1 = (i == 6).then_some(x[1]);
        m.step(&x, &[Some(x[0]), y1], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let mut out = Vec::new();
    m.error_inflation_into(&mut out);
    assert!(out[0].is_finite(), "{out:?}");
    assert_eq!(out[1], f64::INFINITY, "{out:?}");
}

/// A target with no row at a solve has no fit: its slots are NaN, and
/// carry no readiness shares, so once its rows arrive it predicts nothing
/// until a solve has seen them. The solve read its cross-moments of zero
/// as data and left it zeros -- a fit of nothing -- so it was predicted as
/// exactly 0.0 until the next scheduled solve, 56 rows of 80 under
/// `solve_every = 1000` (review round 4, CC1). And it has a first solve of
/// its own, as a fresh model has: on the row its own weight first reaches
/// its `min_weight`, the fifth of its rows, row 14, where it waited for the
/// cadence's next solve, the row cap 25 rows after the first solve at row 4
/// (docs/PLAN.md task 195, S9b). Both target layouts, standardized and not.
#[test]
fn a_target_that_joins_after_the_first_solve_is_not_predicted_from_zeros() {
    for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
        for standardize in [false, true] {
            let case = format!("{gaps:?}, standardize {standardize}");
            let mut c = cfg(1, 2);
            c.ridge = vec![1e-6, 0.5];
            c.min_weight = 5.0;
            c.solve_every = f64::INFINITY;
            c.max_rows_between_solves = 25;
            c.target_gaps = gaps;
            c.standardize = standardize;
            let mut m = EwRidge::new(c).unwrap();
            let mut s = 17u64;
            for i in 0..40 {
                let x = [lcg(&mut s)];
                let b = (i >= 10).then(|| 1.0 - 2.0 * x[0] + 0.01 * lcg(&mut s));
                let p = m.step(
                    &x,
                    &[Some(0.5 + x[0]), b],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
                // Target 1's two slots, one per ridge.
                let late = &p.pred[2..];
                if i <= 14 {
                    assert!(
                        late.iter().all(|v| v.is_nan()),
                        "{case}, row {i}: predicted {late:?} from a fit nobody solved"
                    );
                } else if gaps == TargetGaps::OwnRows {
                    assert!(
                        (late[0] - (1.0 - 2.0 * x[0])).abs() < 0.2,
                        "{case}, row {i}: {late:?}"
                    );
                } else {
                    // Pairwise reads the Gram over every row beside the
                    // target's own cross-moments: a fit, not its five
                    // rows' least squares.
                    assert!(late.iter().all(|v| v.is_finite()), "{case}, row {i}");
                }
                // After the row: the target's own first solve at the end of
                // row 14 has seen it.
                if i == 14 {
                    let beta = m.coefficients().expect("solved at row 4");
                    assert!(
                        beta[2..].iter().flatten().all(|v| v.is_finite()),
                        "{case}, row {i}: its own first solve: {beta:?}"
                    );
                }
                if (4..=13).contains(&i) {
                    let beta = m.coefficients().expect("solved at row 4");
                    assert!(
                        beta[2..].iter().flatten().all(|v| v.is_nan()),
                        "{case}, row {i}: {beta:?}"
                    );
                    assert!(beta[..2].iter().flatten().all(|v| v.is_finite()), "{case}");
                    let shares = m.support_coef().expect("solved");
                    assert!(
                        shares[2..].iter().flatten().all(|v| v.is_nan()),
                        "{case}, row {i}: no shares beside no fit: {shares:?}"
                    );
                    let mut inflation = Vec::new();
                    assert!(OnlineModel::error_inflation_into(&m, &mut inflation));
                    assert_eq!(inflation[2..], [f64::INFINITY; 2], "{case}, row {i}");
                }
            }
        }
    }
}

/// A prior the caller gave is a fit before any row: a target with no row
/// and a `coef_prior` under a ridge reports the prior's fit, as
/// `coef0_with_ridge_decay_warms_the_start_then_fades` has the fresh model
/// do, where one without a prior reports none (CC1). Under `own_rows` the
/// late target's Gram holds no row, so the slope is the prior's; the
/// intercept is free in the mean form, the target's mean of no rows less
/// nothing.
#[test]
fn a_late_target_with_a_prior_reports_the_prior() {
    let mut c = cfg(1, 2);
    c.ridge = vec![0.5];
    c.min_weight = 2.0;
    c.solve_every = f64::INFINITY;
    c.max_rows_between_solves = 25;
    c.coef_prior = Some(vec![vec![0.0, 0.0], vec![1.0, -2.0]]);
    let mut m = EwRidge::new(c).unwrap();
    let mut s = 19u64;
    for i in 0..5 {
        let x = [lcg(&mut s)];
        m.step(
            &x,
            &[Some(0.5 + x[0]), None],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let beta = m.coefficients().expect("solved at row 1");
    assert_eq!(beta[1][0], 0.0, "{beta:?}");
    assert!((beta[1][1] + 2.0).abs() < 1e-12, "{beta:?}");
}

/// An empty window has no fit and no readiness shares: the early return
/// left the last solve's `edf` and data shares beside the NaN fit, so the
/// `support_coef` of the row after a gap was the row before's, bit for bit
/// (review round 4, CA1; the combo skipped for no weight is CA6's). A gap
/// of 101 past a window of 10, on a row of weight 0, leaves nothing in it.
/// That row does not solve -- a zero-weight row is clock alone, and the
/// fit stays the last solve's, as it would were the row not there (task
/// 214) -- so the solve on the empty window is called here.
#[test]
fn an_empty_window_reports_no_shares() {
    let mut c = cfg(1, 1);
    c.decay = Decay::Halflife(20.0);
    c.window = Some(10.0);
    c.min_weight = 0.0;
    let mut m = EwRidge::new(c).unwrap();
    m.set_keep_factor(true);
    let mut s = 43u64;
    for i in 0..40 {
        let x = [lcg(&mut s)];
        let y = 2.0 * x[0] + 0.1 * lcg(&mut s);
        m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
    }
    let before = m.support_coef().expect("solved");
    assert!(
        before[0][1].is_finite(),
        "the case needs a share: {before:?}"
    );
    let x = [lcg(&mut s)];
    let fit = m.coefficients().unwrap().to_vec();
    m.step(&x, &[Some(2.0 * x[0])], 101.0, 0.0);
    assert_eq!(m.n_eff(), 0.0, "the window is empty");
    assert_eq!(
        m.coefficients().unwrap(),
        &fit[..],
        "the zero-weight row solved"
    );
    m.solve();
    let beta = m.coefficients().unwrap();
    assert!(beta[0].iter().all(|v| v.is_nan()), "{beta:?}");
    let shares = m.support_coef().unwrap();
    assert!(
        shares[0].iter().all(|v| v.is_nan()),
        "no shares beside no fit: {shares:?}"
    );
    assert_eq!(
        m.pending_readiness(),
        0,
        "no solve's shares wait to be read"
    );
    let mut rows = Vec::new();
    assert!(OnlineModel::row_error_inflation_into(
        &m,
        &[0.3],
        1.0,
        &mut rows
    ));
    assert_eq!(rows, [f64::INFINITY], "no system to read a row against");
}

/// A kept system carries no Gram index: `System.gram` was written at every
/// solve and read nowhere (review round 4, CA11), so it left the state at
/// schema 38.
#[test]
fn a_kept_system_carries_only_what_is_read() {
    let mut m = EwRidge::new(cfg(2, 1)).unwrap();
    m.set_keep_factor(true);
    let mut s = 47u64;
    for i in 0..10 {
        let x = [lcg(&mut s), lcg(&mut s)];
        m.step(
            &x,
            &[Some(x[0] - x[1])],
            if i == 0 { 0.0 } else { 1.0 },
            1.0,
        );
    }
    let sys = m.ready.systems[0].as_ref().expect("kept");
    let json = serde_json::to_value(sys).unwrap();
    let mut keys: Vec<&str> = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["a", "centred", "mean", "s", "scale", "z"], "{json}");
}

// --- the weekly pass's survivors (docs/PLAN.md task 158) --------------

/// `solve_share` is a positive, finite fraction: 0, a negative, `inf`
/// and NaN are refused, and a fraction accepted.
#[test]
fn solve_share_must_be_finite_and_positive() {
    let with = |f: f64| {
        let mut c = cfg(2, 1);
        c.solve_share = Some(f);
        c.validate()
    };
    for bad in [0.0, -0.5, f64::INFINITY, f64::NAN] {
        let err = with(bad).unwrap_err();
        assert!(err.contains("solve_share"), "{bad}: {err}");
    }
    with(0.25).unwrap();
}

/// The share the model runs at is the one it is given, by its cfg or
/// set on it later.
#[test]
fn the_solve_share_is_the_one_set() {
    let mut c = cfg(2, 1);
    c.solve_share = Some(0.25);
    let mut m = EwRidge::new(c).unwrap();
    assert_eq!(m.solve_share(), Some(0.25));
    m.set_solve_share(Some(0.75));
    assert_eq!(m.solve_share(), Some(0.75));
    m.set_solve_share(None);
    assert_eq!(m.solve_share(), None);
}

/// Under `solve_share` a solve is due once the weight learned since the
/// last one reaches that share of `n_eff` (the row's own included), and
/// at the first row that meets `min_weight`; only a positive, finite
/// weight counts toward it, and a row of weight 0 never solves (task 214).
/// Held to that rule kept beside the model, at weights that move and with
/// a decay (`max_rows_between_solves` out of reach).
#[test]
fn solve_share_solves_when_the_weight_since_reaches_its_share() {
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(20.0);
    c.min_weight = 0.0;
    c.max_rows_between_solves = u32::MAX;
    c.solve_share = Some(0.3);
    let mut m = EwRidge::new(c).unwrap();
    let (mut w_sum, mut since, mut first) = (0.0f64, 0.0f64, true);
    let (mut s, mut solves) = (109u64, 0);
    for i in 0..200 {
        let x = [lcg(&mut s), lcg(&mut s)];
        let w = if i % 7 == 3 {
            0.0
        } else {
            0.25 + (lcg(&mut s) + 1.0)
        };
        let d = if i == 0 { 0.0 } else { 1.0 };
        m.step(&x, &[Some(x[0] - x[1])], d, w);
        w_sum = 0.5f64.powf(d / 20.0) * w_sum + w;
        since += w;
        let due = w > 0.0 && (since >= 0.3 * w_sum || first);
        assert_eq!(w > 0.0 && m.rows_since_solve == 0, due, "row {i}");
        if due {
            (since, first, solves) = (0.0, false, solves + 1);
        }
    }
    assert!((20..150).contains(&solves), "{solves} solves");
    // A weight that is not finite does not count toward the share.
    let before = m.weight_since_solve;
    m.step(&[0.1, 0.2], &[Some(0.0)], 1.0, f64::INFINITY);
    assert_eq!(m.weight_since_solve, before);
}

/// The slots whose solve still holds its factor: every slot after a
/// solve, none once the run has settled them (docs/PLAN.md task 140).
#[test]
fn pending_readiness_counts_the_slots_holding_a_factor() {
    let mut c = cfg(2, 2);
    c.min_weight = 0.0;
    let (mut m, _) = fitted(c, 20, 111);
    assert_eq!(m.pending_readiness(), 2);
    m.settle_readiness();
    assert_eq!(m.pending_readiness(), 0);
}

/// The noise gate's bound, read exactly: where a solve is unread and its
/// shares are sure to be numbers, a slot below the limit reads
/// `sqrt(1 + (1 + k) / n_kish)`, the ratio at the largest `edf` the
/// eliminated intercept and `k` slopes could make, `n_kish` the Kish
/// count of the weights (kept here from the definition); at a limit
/// exactly that ratio the bound is not below it and the exact ratio is
/// read; and once a read has taken the shares, the gate reads them,
/// never the bound.
#[test]
fn the_gate_bound_is_the_largest_edf_ratio_and_only_below_the_limit() {
    let mut c = cfg(3, 1);
    c.decay = Decay::Halflife(25.0);
    c.min_weight = 0.0;
    c.solve_every = 3.0;
    c.max_rows_between_solves = 100;
    c.ridge = vec![0.3];
    let (_, rows) = fitted(c.clone(), 60, 113);
    let mut m = EwRidge::new(c).unwrap();
    let (mut w_sum, mut q_sum) = (0.0f64, 0.0f64);
    let (mut gate, mut exact) = (Vec::new(), Vec::new());
    let (mut stood_in, mut read) = (0, 0);
    for (i, (x, y, w)) in rows.iter().enumerate() {
        if let Some(bound) = m.ready.edf_bound_at(0) {
            let n = w_sum * w_sum / q_sum;
            let want = (1.0 + 4.0 / n).sqrt();
            m.error_inflation_gate_into(&[], 1.0, &mut gate, 10.0);
            assert!(
                (gate[0] - want).abs() <= 1e-12 * want,
                "row {i}: {gate:?} against {want}"
            );
            assert_eq!(bound, 4.0);
            stood_in += 1;
            // At the bound's own ratio, the exact one.
            let n = m.gram_kish()[0].unwrap();
            let at = (1.0 + bound / n).sqrt();
            m.error_inflation_gate_into(&[], 1.0, &mut gate, at);
            m.error_inflation_into(&mut exact);
            assert!(exact[0] < at, "row {i}: the shares are under 1");
            assert_eq!(gate[0].to_bits(), exact[0].to_bits(), "row {i}");
        }
        if m.beta.is_some() {
            // The shares are taken: the gate reads them.
            m.error_inflation_into(&mut exact);
            m.error_inflation_gate_into(&[], 1.0, &mut gate, 10.0);
            assert_eq!(gate[0].to_bits(), exact[0].to_bits(), "row {i}");
            read += 1;
        }
        m.step(x, y, if i == 0 { 0.0 } else { 1.0 }, *w);
        let lam = if i == 0 { 1.0 } else { 0.5f64.powf(1.0 / 25.0) };
        w_sum = lam * w_sum + w;
        q_sum = lam * lam * q_sum + w * w;
    }
    assert!(
        stood_in > 5 && read > 20,
        "{stood_in} stood in, {read} read"
    );
}

/// A Gram whose Kish count is 0 -- its `Σ w²` held at the smallest
/// subnormal by rounding while `W²` underflows, over thousands of rows
/// of weight 0 -- with a fit of no degrees of freedom (no intercept, a
/// feature always 0, a ridge of 1) reads the infinite ratio, not
/// `sqrt(1 + 0/0)`.
#[test]
fn a_kish_count_of_zero_reads_an_infinite_ratio() {
    let mut c = cfg(1, 1);
    c.fit_intercept = false;
    c.ridge = vec![1.0];
    c.min_weight = 0.0;
    c.decay = Decay::Halflife(5.0);
    let mut m = EwRidge::new(c).unwrap();
    m.step(&[0.0], &[Some(1.0)], 0.0, 1.0);
    let mut i = 0;
    while m.gram_kish()[0] != Some(0.0) {
        m.step(&[0.0], &[Some(1.0)], 1.0, 0.0);
        i += 1;
        assert!(
            i < 20_000,
            "Kish's count never reached 0: {:?}",
            m.gram_kish()
        );
    }
    assert_eq!(m.ready.edf_at(0), 0.0, "a fit of no degrees of freedom");
    let mut out = Vec::new();
    m.error_inflation_into(&mut out);
    assert_eq!(out, vec![f64::INFINITY]);
}

/// The ring's shadow is the ring: learning the rows the model learns, it
/// is past a refusing budget exactly when the model's ring is (docs/PLAN.md
/// task 115 (d)); a model without a window has none.
#[test]
fn the_window_shadow_follows_the_ring() {
    assert!(EwRidge::new(cfg(2, 1)).unwrap().window_shadow().is_none());
    let mut c = cfg(2, 1);
    c.decay = Decay::Halflife(30.0);
    c.window = Some(40.0);
    c.max_rows_between_snapshots = Some(2);
    c.min_weight = 0.0;
    let (mut m, rows) = fitted(c, 10, 115);
    m.set_window_budget(Some(crate::WindowBudget::Refuse(0.002)));
    let mut shadow = m.window_shadow().expect("a windowed model has a shadow");
    let mut over = 0;
    for (x, y, w) in &rows {
        shadow.learn(1.0, None);
        m.step(x, y, 1.0, *w);
        assert_eq!(shadow.over_budget(), m.window_over_budget());
        over += usize::from(m.window_over_budget().is_some());
    }
    assert!(over > 0, "the ring went past its budget");
}

/// Hard rule 9 on a solve schedule (docs/PLAN.md task 214): a zero-weight
/// row is clock alone, so it never solves and is no row of the row cap. A
/// solve the clock brings due on one waits for the next row with weight,
/// which the clock has brought it due on too, and the solves fall on the
/// rows they fall on without it: the fit read after every row, and every
/// prediction, are the stream's without the zero-weight rows, their clock
/// carried into the next row. A clock-due solve fired on the zero-weight
/// row, before the next row's data, where the stream without it solved
/// after them: the contract test's fit moved until the next solve. Under a
/// row cap the zero-weight rows brought each solve a row early. By the
/// clock, by the row cap, and by the weight's share, the default.
#[test]
fn a_zero_weight_row_never_solves() {
    let base = {
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(12.0);
        c.min_weight = 3.0;
        c
    };
    let by_clock = EwRidgeCfg {
        solve_every: 3.0,
        max_rows_between_solves: 10_000,
        ..base.clone()
    };
    let by_rows = EwRidgeCfg {
        solve_every: 1e9,
        max_rows_between_solves: 4,
        ..base.clone()
    };
    let by_share = EwRidgeCfg {
        solve_every: 1e9,
        max_rows_between_solves: 10_000,
        solve_share: Some(0.2),
        ..base
    };
    for (what, c) in [("clock", by_clock), ("rows", by_rows), ("share", by_share)] {
        let (mut with, mut without) = (EwRidge::new(c.clone()).unwrap(), EwRidge::new(c).unwrap());
        let mut s = 214u64;
        let mut carried = 0.0;
        let mut compared = 0;
        for i in 0..160usize {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = [
                Some(0.3 + x[0] - 0.5 * x[1] + 0.05 * lcg(&mut s)),
                Some(2.0 * x[1]),
            ];
            let d = if i == 0 { 0.0 } else { 1.0 };
            if i > 4 && matches!(i % 9, 2 | 3) {
                with.step(&x, &y, d, 0.0);
                carried += d;
                continue;
            }
            let a = with.step(&x, &y, d, 1.0);
            let b = without.step(&x, &y, d + carried, 1.0);
            carried = 0.0;
            let close = |u: &[f64], v: &[f64]| {
                u.iter().zip(v).all(|(p, q)| {
                    (p.is_nan() && q.is_nan()) || (p - q).abs() <= 1e-12 * (1.0 + q.abs())
                })
            };
            assert!(
                close(&a.pred, &b.pred),
                "{what}, row {i}: {a:?} against {b:?}"
            );
            let (u, v) = (with.coefficients(), without.coefficients());
            assert_eq!(u.is_some(), v.is_some(), "{what}, row {i}");
            for (p, q) in u.unwrap_or(&[]).iter().zip(v.unwrap_or(&[])) {
                assert!(
                    close(p, q),
                    "{what}, row {i}: coefficients {p:?} against {q:?}"
                );
            }
            assert_eq!(
                with.rows_since_solve, without.rows_since_solve,
                "{what}, row {i}"
            );
            compared += usize::from(a.pred.iter().all(|p| p.is_finite()));
        }
        assert!(compared > 100, "{what}: {compared}");
    }
}

/// Task 180: `solve_every` reads the stamps its caller hands. On 1 ms rows
/// under 2 s it solves every 2,000th row after the first, as the stamps'
/// nanoseconds say; handed none, it reads the clock summed since the last
/// solve, which is a row late each time, as it always was.
#[test]
fn solve_every_measures_the_stamps_it_is_handed() {
    for (stamped, want) in [(true, [0, 2000, 4000]), (false, [0, 2001, 4002])] {
        let mut c = cfg(1, 1);
        c.min_weight = 0.0;
        c.solve_every = 2.0;
        c.max_rows_between_solves = u32::MAX;
        let mut m = EwRidge::new(c).unwrap();
        let got = crate::since::events_on_millisecond_rows(4_100, stamped, |i, stamp, d| {
            if let Some(s) = stamp {
                m.stamp_next(s);
            }
            let x = (i % 7) as f64;
            m.step(&[x], &[Some(1.0 + 2.0 * x)], d, 1.0);
            m.rows_since_solve == 0
        });
        assert_eq!(got, want, "stamped: {stamped}");
    }
}

/// Under a blocked Gram a row of weight 0 is held in the block as it is, to
/// be merged with the rest. A row with a value that is not usable is a row
/// of weight 0 by the rule every model keeps (`OnlineModel`, task 183), and
/// is held as a row of zeros instead, so nothing of its own is kept: the
/// one place a refused row's state is not the same row's at weight 0. While
/// the block holds it the two differ in the held values alone. The block
/// closes on the same row either way, and a held row of no weight merges to
/// nothing, so every prediction after the row is that row's at weight 0, to
/// the bit, and once the block has merged so is the state, byte for byte.
#[test]
fn a_blocked_gram_holds_a_refused_row_as_zeros() {
    // A solve merges the held block first, so the solves are spaced to
    // leave the refused row held after its own step: solves at rows 0, 6,
    // 12 and 18, the refused row the 10th, three rows into a block of 8.
    let mut c = cfg(2, 2);
    c.gram_block_rows = 8;
    c.solve_every = 6.0;
    c.max_rows_between_solves = 1000;
    c.min_weight = 0.0;
    let same = |u: &f64, v: &f64| u.to_bits() == v.to_bits() || (u.is_nan() && v.is_nan());
    let bytes = |m: &EwRidge| rmp_serde::to_vec(&m.state()).unwrap();
    let pending = |m: &EwRidge| m.acc.grams.grams.iter().any(EwCov::has_pending);
    let row = |s: &mut u64| {
        let x = [lcg(s), lcg(s)];
        (x, [Some(x[0] - 0.5 * x[1]), Some(2.0 * x[1])])
    };
    for (bad, w) in [
        (Some([f64::NAN, 0.5]), 1.0),
        (Some([0.5, f64::INFINITY]), 1.0),
        (None, f64::NAN),
    ] {
        let case = format!("{bad:?} at weight {w}");
        let mut refused = EwRidge::new(c.clone()).unwrap();
        let mut zero = EwRidge::new(c.clone()).unwrap();
        let mut s = 43u64;
        for i in 0..9 {
            let (x, y) = row(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            refused.step(&x, &y, d, 1.0);
            zero.step(&x, &y, d, 1.0);
        }
        let (x, y) = row(&mut s);
        refused.step(&bad.unwrap_or(x), &y, 1.0, w);
        zero.step(&x, &y, 1.0, 0.0);
        assert!(
            pending(&refused) && pending(&zero),
            "{case}: the fixture holds the row"
        );
        assert_ne!(
            bytes(&refused),
            bytes(&zero),
            "{case}: the held values differ"
        );
        for i in 0..12 {
            let (x, y) = row(&mut s);
            let (a, b) = (refused.step(&x, &y, 1.0, 1.0), zero.step(&x, &y, 1.0, 1.0));
            assert!(
                a.pred.iter().zip(&b.pred).all(|(u, v)| same(u, v)),
                "{case}, row {i} after: {a:?} against {b:?}"
            );
        }
        assert_eq!(bytes(&refused), bytes(&zero), "{case}: after the merge");
    }
}

/// A fit that keeps no column -- every feature of the slot constant, so the
/// standardiser drops each -- is the intercept alone, the mean of the
/// target, and reads as one (the docs of `coef_variance` and
/// `row_error_inflation_into`; docs/PLAN.md task 218): the intercept's
/// variance over the noise is `1 / n_kish`, every slope's NaN, and a row's
/// `h` is the mean's own `1 / n_kish`, `sqrt(1 + 1 / n)`, the summary's
/// `sqrt(1 + edf / n)` at `edf = 1`; through the origin, with every feature
/// 0, the fit is 0, `h` is 0 and the row reads 1. The live model read NaN
/// and `inf`, a kept system with no column having no factor until a load
/// factored its 0×0 matrix, so a resumed stream read `1 / n` and `sqrt(1 +
/// 1 / n)` where the uninterrupted one read NaN and `inf` until its next
/// solve. Live and resumed agree to the bit, row by row.
#[test]
fn a_fit_with_no_kept_column_reads_as_the_intercept_alone_live_and_resumed() {
    for fit_intercept in [true, false] {
        let mut c = cfg(1, 1);
        c.standardize = true;
        c.fit_intercept = fit_intercept;
        c.min_weight = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        m.set_keep_factor(true);
        let x = [if fit_intercept { 2.0 } else { 0.0 }];
        let mut s = 218u64;
        let mut checked = 0;
        for i in 0..12usize {
            let y = 1.0 + 0.5 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            let n = (i + 1) as f64;
            let resumed = EwRidge::restore(&m.state()).unwrap();
            let read = |m: &EwRidge| {
                let Some(crate::CoefVariance::PerNoise(v)) = m.coef_variance() else {
                    panic!("ewridge's variances are over the noise");
                };
                let mut row = Vec::new();
                assert!(m.row_error_inflation_into(&x, 1.0, &mut row));
                (v[0].clone(), row[0])
            };
            let ((v, row), (v_back, row_back)) = (read(&m), read(&resumed));
            if fit_intercept {
                assert!((v[0] - 1.0 / n).abs() <= 1e-12 / n, "row {i}: {v:?}");
                assert!(
                    ((row - (1.0 + 1.0 / n).sqrt()).abs()) <= 1e-12,
                    "row {i}: {row}"
                );
                let mut summary = Vec::new();
                assert!(OnlineModel::error_inflation_into(&m, &mut summary));
                assert_eq!(row.to_bits(), summary[0].to_bits(), "row {i}");
            } else {
                assert_eq!(row, 1.0, "row {i}");
            }
            let slope = usize::from(fit_intercept);
            assert!(v[slope].is_nan(), "row {i}: the dropped slope: {v:?}");
            // Resumed, the same numbers to the bit, and the next row too.
            let bits = |v: &[f64]| v.iter().map(|u| u.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&v), bits(&v_back), "row {i}");
            assert_eq!(row.to_bits(), row_back.to_bits(), "row {i}");
            let (mut a, mut b) = (m.clone(), resumed);
            let y = 1.0 + 0.5 * lcg(&mut { s });
            let (pa, pb) = (
                a.step(&x, &[Some(y)], 1.0, 1.0).pred,
                b.step(&x, &[Some(y)], 1.0, 1.0).pred,
            );
            assert_eq!(bits(&pa), bits(&pb), "row {i}");
            assert_eq!(bits(&read(&a).0), bits(&read(&b).0), "row {i}");
            checked += 1;
        }
        assert_eq!(checked, 12);
    }
}
