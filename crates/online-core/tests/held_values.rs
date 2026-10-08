//! A feature that stops moving (docs/PLAN.md task 101).
//!
//! A running mean steps toward each value it is given, and toward one value
//! given row after row it converges until the step rounds to nothing, a few
//! rounding steps short of it. Everything the mean centred was then off by
//! that gap: a variance settled on its square instead of decaying with the
//! history, a co-moment on the gap times the other side's noise, and a model
//! that divides one by the other read the ratio of two rounding artefacts.
//! The feature's level decides when, since the gap is counted in its
//! rounding steps: measured, a lasso's slope on a feature at 1e8 went from
//! the 0.5 it had learned to -4.7e3 within 40 half-lives of it stopping.
//!
//! Every running mean is now a pair, `hi + lo`, whose steps no rounding
//! drops (`crates/online-core/src/comp.rs`). These hold every
//! model that centres a feature to what exact arithmetic gives, for 150
//! half-lives after it stops, at levels from 0 to 1e12: the slope learned
//! while it moved, a spread that keeps decaying, and fits that do not
//! depend on the level the feature sits at; and the same of a target that
//! stops. Each fails with the means plain.

use online_core::*;

const H: f64 = 20.0;
const MOVING: usize = 300;
const HELD_HALFLIVES: usize = 150;
/// The levels a stopped feature sits at: `-0.37` holds it at exactly zero,
/// which a mean reaches by decaying geometrically, and stalls on only among
/// the subnormals.
const LEVELS: [f64; 6] = [0.5, -0.37, 1e3, 1e8, -1e8, 1e12];

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

/// One row per clock unit. `x0` and `x1` move on every row; `x2` moves around
/// `level` for 300 rows and then holds `level + 0.37`. The target is `1 + 2 x0
/// - x1 + 0.5 (x2 - level)` and noise, the same numbers at every level.
fn stream(level: f64) -> Vec<([f64; 3], f64)> {
    stream_of(level, HELD_HALFLIVES * H as usize)
}

/// [`stream`] with `held` rows after the feature stops.
fn stream_of(level: f64, held: usize) -> Vec<([f64; 3], f64)> {
    let mut s = 7u64;
    (0..MOVING + held)
        .map(|i| {
            let (x0, x1) = (lcg(&mut s), lcg(&mut s));
            let u = if i < MOVING { lcg(&mut s) } else { 0.37 };
            let y = 1.0 + 2.0 * x0 - x1 + 0.5 * u + 0.3 * lcg(&mut s);
            ([x0, x1, level + u], y)
        })
        .collect()
}

fn d(i: usize) -> f64 {
    if i == 0 { 0.0 } else { 1.0 }
}

/// A model's predictions over the stream, and its slope on `x2` after each
/// row where it has one (NaN where it has none).
fn run<M: OnlineModel>(m: M, level: f64, slope: impl Fn(&M) -> f64) -> (Vec<f64>, Vec<f64>) {
    run_on(m, &stream(level), slope)
}

fn run_on<M: OnlineModel>(
    mut m: M,
    rows: &[([f64; 3], f64)],
    slope: impl Fn(&M) -> f64,
) -> (Vec<f64>, Vec<f64>) {
    let (mut pred, mut coef) = (Vec::new(), Vec::new());
    for (i, (x, y)) in rows.iter().enumerate() {
        let out = m.step(x, &[Some(*y)], d(i), 1.0);
        pred.push(out.pred[out.pred.len() - 1]);
        coef.push(slope(&m));
    }
    (pred, coef)
}

fn decay() -> Decay {
    Decay::Halflife(H)
}

/// What a fit at each level is held to: its predictions against the same
/// fit at 0.5, as a share of `1 + |pred|`, from `from` half-lives after the
/// stop, to `tol(level)`.
struct Invariance {
    from: usize,
    tol: fn(f64) -> f64,
}

/// A hundred rounding steps of the level. The features at a level carry its
/// rounding in every row they moved on, so a fit on them differs from the
/// fit at 0.5 by that much from the start: under one step, measured for the
/// lasso, both ridges and `huber` at 1e3, 1e8 and 1e12, and under a
/// hundredth of one for `sgd`. The stall took
/// each of them 4e-2 to 2e-1 from the fit at 0.5, at every level; at 1e12,
/// where a step is 2.2e-4, that is 2.4 times this bound, and the slope is
/// what catches it first.
fn steps_of(level: f64) -> f64 {
    1e2 * (level.abs() * f64::EPSILON).max(f64::EPSILON)
}

/// The slope on the stopped feature stays near the one learned while it
/// moved, on every row after it stops and at every level, and the fit at
/// each level predicts what the fit at 0.5 does (`inv`).
///
/// "Near" is 0.5 ± 0.3. Exact arithmetic moves the slope a little as the
/// feature's history decays to the scale of its last few rows, by about
/// `sqrt(b)` of it. Measured over the levels and five seeds, the slope
/// ranged over 0.41 to 0.61, all of it within 10 half-lives of the stop, and
/// within the same range in blocks of 4 to 64 rows. With the means plain it
/// went to -79 at 50 half-lives, -4.7e3 at 40, 1e10.
///
/// Every row after the stop is held: the fit at 0.5 predicts on each, the
/// fit at each level predicts on each of those, and where a model reports a
/// slope it is a number. A model that never predicted, or never solved,
/// passed: the slope's test took NaN and the comparison skipped any row the
/// fit at 0.5 had no prediction for (review 2026-10-05, CF6). `kalman` and
/// `sgd` report no slope on the stopped feature, and say so with `None`.
fn holds<M: OnlineModel>(
    name: &str,
    make: impl Fn() -> M,
    slope: Option<fn(&M) -> f64>,
    inv: Invariance,
) {
    let read = |m: &M| slope.map_or(f64::NAN, |f| f(m));
    let (base, _) = run(make(), 0.5, read);
    let predicted = (MOVING..base.len())
        .filter(|&i| base[i].is_finite())
        .count();
    assert_eq!(
        predicted,
        base.len() - MOVING,
        "{name}: the fit at 0.5 predicts on every row after the stop"
    );
    for level in LEVELS {
        let (pred, coef) = run(make(), level, read);
        for (i, &c) in coef.iter().enumerate().skip(MOVING) {
            assert!(
                slope.is_none() || (0.2..0.8).contains(&c),
                "{name} at level {level}: slope {c} on the stopped feature, {:.1} half_lives after it stopped",
                (i - MOVING) as f64 / H
            );
        }
        let from = MOVING + inv.from * H as usize;
        assert!(
            pred[from..].iter().all(|p| p.is_finite()),
            "{name} at level {level}: a row with no prediction after the stop"
        );
        let worst = (from..pred.len())
            .map(|i| (pred[i] - base[i]).abs() / (1.0 + base[i].abs()))
            .fold(0.0, f64::max);
        let tol = (inv.tol)(level);
        println!("{name} at level {level}: predictions {worst:.1e} from those at 0.5 ({tol:.1e})");
        assert!(
            worst <= tol,
            "{name} at level {level}: predictions {worst:.3e} from those at 0.5, past {tol:.1e}"
        );
    }
}

/// The lasso, both ridges, `huber` and `sgd`: from the stop on.
const FROM_THE_STOP: Invariance = Invariance {
    from: 0,
    tol: steps_of,
};

#[test]
fn the_lasso_keeps_the_slope_it_learned() {
    let cfg = LassoCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        lasso_path: vec![0.05, 0.0],
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: 10.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        max_iter: 1000,
        tol: 1e-12,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    holds(
        "lasso",
        || Lasso::new(cfg.clone()).unwrap(),
        Some(|m: &Lasso| m.coefficients().map_or(f64::NAN, |c| c[0][1][3])),
        FROM_THE_STOP,
    );
}

fn ridge(standardize: bool) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        ridge: vec![1e-6],
        feature_sets: vec![],
        standardize,
        ridge_scale: false,
        coef_prior: None,
        session_shrink: None,
        long_half_life: None,
        min_weight: 10.0,
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

#[test]
fn a_standardized_ridge_keeps_the_slope_it_learned() {
    holds(
        "ewridge",
        || EwRidge::new(ridge(true)).unwrap(),
        Some(|m: &EwRidge| m.coefficients().map_or(f64::NAN, |c| c[0][3])),
        FROM_THE_STOP,
    );
}

/// The blocked Gram update merges rows a block at a time, and its merged
/// step is a pair's too.
#[test]
fn a_blocked_ridge_keeps_the_slope_it_learned() {
    holds(
        "ewridge, blocked",
        || {
            let mut c = ridge(true);
            c.gram_block_rows = 16;
            c.solve_every = 16.0;
            c.max_rows_between_solves = 16;
            EwRidge::new(c).unwrap()
        },
        Some(|m: &EwRidge| m.coefficients().map_or(f64::NAN, |c| c[0][3])),
        FROM_THE_STOP,
    );
}

/// Under a window (review 2026-09-25): while the window still holds rows
/// on which the feature moved, the slope is the one learned; once every row
/// inside it is held, the run reads the feature as having no spread there
/// and the standardized solve drops it, at every level alike. The
/// predictions at a level are those at 0.5 throughout: every row after the
/// stop has a prediction and a slope, at 0.5 and at each level, where a fit
/// that never predicted or never solved passed (review 2026-10-05, CF6).
#[test]
fn a_windowed_ridge_keeps_the_slope_while_the_window_has_spread() {
    let make = || {
        let mut c = ridge(true);
        c.window = Some(3.0 * H);
        EwRidge::new(c).unwrap()
    };
    let slope = |m: &EwRidge| m.coefficients().map_or(f64::NAN, |c| c[0][3]);
    let (base, _) = run(make(), 0.5, slope);
    assert!(
        base[MOVING..].iter().all(|p| p.is_finite()),
        "the fit at 0.5 predicts on every row after the stop"
    );
    let covered = MOVING + 4 * H as usize;
    for level in LEVELS {
        let (pred, coef) = run(make(), level, slope);
        for (i, &c) in coef.iter().enumerate().skip(MOVING) {
            if i < MOVING + 2 * H as usize {
                assert!(
                    (0.2..0.8).contains(&c),
                    "level {level}, row {i}: slope {c} while the window still has spread"
                );
            } else if i >= covered {
                assert!(
                    c == 0.0,
                    "level {level}, row {i}: slope {c} on a feature without spread in the window"
                );
            }
        }
        assert!(
            pred[MOVING..].iter().all(|p| p.is_finite()),
            "level {level}: a row with no prediction after the stop"
        );
        let worst = (MOVING..pred.len())
            .map(|i| (pred[i] - base[i]).abs() / (1.0 + base[i].abs()))
            .fold(0.0, f64::max);
        let tol = steps_of(level);
        assert!(
            worst <= tol,
            "level {level}: predictions {worst:.3e} from those at 0.5, past {tol:.1e}"
        );
    }
}

/// A hold that ends (review 2026-09-25): the feature moves again after
/// fifty half-lives, and within five more the slope is back where it was and
/// the fit at a level predicts what the fit at 0.5 does.
#[test]
fn a_feature_that_moves_again_gets_its_slope_back() {
    let resumes = |level: f64| -> Vec<([f64; 3], f64)> {
        let mut s = 7u64;
        let held = 50 * H as usize;
        (0..MOVING + held + 10 * H as usize)
            .map(|i| {
                let (x0, x1) = (lcg(&mut s), lcg(&mut s));
                let u = if (MOVING..MOVING + held).contains(&i) {
                    0.37
                } else {
                    lcg(&mut s)
                };
                let y = 1.0 + 2.0 * x0 - x1 + 0.5 * u + 0.3 * lcg(&mut s);
                ([x0, x1, level + u], y)
            })
            .collect()
    };
    let slope = |m: &EwRidge| m.coefficients().map_or(f64::NAN, |c| c[0][3]);
    let (base, _) = run_on(EwRidge::new(ridge(true)).unwrap(), &resumes(0.5), slope);
    let back = MOVING + 55 * H as usize;
    for level in LEVELS {
        let (pred, coef) = run_on(EwRidge::new(ridge(true)).unwrap(), &resumes(level), slope);
        for (i, &c) in coef.iter().enumerate().skip(back) {
            assert!(
                (0.2..0.8).contains(&c),
                "level {level}, row {i}: slope {c} after the feature moved again"
            );
        }
        let worst = (back..pred.len())
            .map(|i| (pred[i] - base[i]).abs() / (1.0 + base[i].abs()))
            .fold(0.0, f64::max);
        let tol = steps_of(level);
        assert!(
            worst <= tol,
            "level {level}: predictions {worst:.3e} from those at 0.5, past {tol:.1e}"
        );
    }
}

/// The blocked Gram at 1e12 with a row of no weight every seventh, carrying
/// another value of the stopped feature (review 2026-09-25): the block's
/// residue and the zero-weight rule together, where the earlier blocked
/// case had unit weights at 1e8. The slope stays learned, and the pair
/// reaches the held value in the moments.
#[test]
fn a_blocked_ridge_keeps_its_slope_through_rows_of_no_weight() {
    let level = 1e12;
    let rows: Vec<([f64; 3], f64, f64)> = stream(level)
        .into_iter()
        .enumerate()
        .map(|(i, (mut x, y))| {
            let w = if i > 0 && i % 7 == 0 { 0.0 } else { 1.0 };
            if w == 0.0 && i >= MOVING {
                x[2] = level + 5.0;
            }
            (x, y, w)
        })
        .collect();
    let mut c = ridge(true);
    c.gram_block_rows = 16;
    c.solve_every = 16.0;
    c.max_rows_between_solves = 16;
    let mut m = EwRidge::new(c).unwrap();
    for (i, (x, y, w)) in rows.iter().enumerate() {
        m.step(x, &[Some(*y)], d(i), *w);
        if i >= MOVING + 10 * H as usize {
            let c = m.coefficients().map_or(f64::NAN, |c| c[0][3]);
            assert!(c.is_nan() || (0.2..0.8).contains(&c), "row {i}: slope {c}");
        }
    }
}

#[test]
fn a_standardized_huber_keeps_the_slope_it_learned() {
    let cfg = RobustCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        loss: RobustLoss::Huber { delta: 1.345 },
        ridge: 1e-6,
        standardize: true,
        min_weight: 10.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 0.05,
    };
    holds(
        "huber",
        || Robust::new(cfg.clone()).unwrap(),
        Some(|m: &Robust| m.coefficients().map_or(f64::NAN, |c| c[0][3])),
        FROM_THE_STOP,
    );
}

/// `kalman` reports its coefficients through scales that shrink with a
/// stopped feature's spread, in exact arithmetic too, so its slope is not
/// held here; what it predicts is. Its scaler starts from a mean of zero, so
/// its first rows see the level itself and the fit at a level differs from
/// the fit at 0.5 by a warm-up that decays with the coefficient half-life:
/// 1.1e-5 at every level from 100 half-lives. Read through the moments as
/// they stood, before task 206 re-mapped the filter through their moves, it
/// was 7.3e-2 at the stop and 4.9e-4 from 100 half-lives. The stall grew it
/// back to 6.9e-2 there.
#[test]
fn kalman_predicts_the_same_at_every_level() {
    let cfg = KalmanCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        half_life: vec![200.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 10.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    };
    holds(
        "kalman",
        || Kalman::new(cfg.clone()).unwrap(),
        None,
        Invariance {
            from: 100,
            tol: |_| 5e-3,
        },
    );
}

/// `sgd` standardizes each row with the moments that include it, so it reads
/// the stopped feature's deviation against a spread that decays with it --
/// the deviation from the pair, which keeps every bit as it shrinks.
/// Measured from the stop: 5.4e-13 at 1e3, 5.8e-8 at 1e8, 5.6e-4 at 1e12,
/// where the ridges read 1.3e-13, 1.3e-8 and 9.3e-5. Past its warm-up the fit
/// is held in the caller's units (docs/PLAN.md task 206), so its intercept
/// carries the level times the slope and rounds at that size; read through
/// the scaler it was 3.7e-15, 1.7e-9 and 3.2e-6. The stall left 6.4e-2 at
/// 40 half-lives and 0.12 at 100.
#[test]
fn sgd_predicts_the_same_at_every_level() {
    let cfg = SgdCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        loss: SgdLoss::Squared,
        learning_rate: 0.01,
        schedule: LearningRate::Constant,
        l2: 0.0,
        min_weight: 10.0,
        standardize: true,
        strict_binary: false,
        clip_gradient: f64::INFINITY,
        constraint: None,
    };
    holds(
        "sgd",
        || Sgd::new(cfg.clone()).unwrap(),
        None,
        FROM_THE_STOP,
    );
}

/// What `ew_cov` reports of a stopped feature: its spread keeps decaying
/// with its history, as a weighted variance of rows that no longer move
/// must, and its correlation with a feature that moves goes to zero with it.
/// Each settled on a rounding artefact instead.
#[test]
fn ew_cov_reports_a_stopped_feature_as_it_is() {
    for level in LEVELS {
        let mut m = EwCovModel::new(EwCovCfg {
            n_features: 3,
            decay: decay(),
            stats: vec![EwCovStat::Var, EwCovStat::Corr],
            min_weight: 0.0,
            precision_prior: None,
            mahal_quantiles: vec![],
            pca: 0,
            pca_every: 0.0,
            max_rows_between_pca: u32::MAX,
            lags: vec![],
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let mut at_stop = 0.0;
        for (i, (x, _)) in stream(level).iter().enumerate() {
            m.step(x, &[], d(i), 1.0);
            if i == MOVING - 1 {
                at_stop = m.cov().var(2);
            }
        }
        let (var, cov) = (m.cov().var(2), m.cov());
        // 150 half-lives take 2^-150 of it, about 7e-46; rounding may keep a
        // little more, never a floor.
        assert!(
            var <= 1e-30 * at_stop,
            "level {level}: variance {var:e} of {at_stop:e}"
        );
        let corr = cov.cov(0, 2) / (cov.var(0) * var).sqrt();
        assert!(
            corr.is_nan() || corr.abs() <= 1e-10,
            "level {level}: correlation {corr}"
        );
    }
}

/// The same for `marginal`'s pairs: the stopped feature's correlation with
/// the target decays with its spread.
#[test]
fn marginal_reports_a_stopped_feature_as_it_is() {
    for level in LEVELS {
        let mut m = Marginal::new(MarginalCfg {
            n_features: 3,
            n_targets: 1,
            decay: decay(),
            min_weight: vec![0.0],
            lags: vec![],
            serial_rule: None,
            cross_lags: None,
            bins: None,
            feature_moments: online_core::FeatureMomentLayout::PerTarget,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let mut at_stop = 0.0;
        for (i, (x, y)) in stream(level).iter().enumerate() {
            m.step(x, &[Some(*y)], d(i), 1.0);
            if i == MOVING - 1 {
                at_stop = m.pair(0, 2).var_x;
            }
        }
        let p = m.pair(0, 2);
        assert!(
            p.var_x <= 1e-30 * at_stop,
            "level {level}: variance {:e} of {at_stop:e}",
            p.var_x
        );
        let corr = p.cov / (p.var_x * p.var_y).sqrt();
        assert!(
            corr.is_nan() || corr.abs() <= 1e-10,
            "level {level}: correlation {corr}"
        );
    }
}

/// One row per clock unit. The three features move on every row; the target
/// is `level + 1 + 2 x0 - x1 + 0.5 x2` and noise for 300 rows, and then holds
/// `level + 0.37`.
fn held_target(level: f64) -> Vec<([f64; 3], f64)> {
    let mut s = 7u64;
    (0..MOVING + HELD_HALFLIVES * H as usize)
        .map(|i| {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let noise = 0.3 * lcg(&mut s);
            let y = if i < MOVING {
                level + 1.0 + 2.0 * x[0] - x[1] + 0.5 * x[2] + noise
            } else {
                level + 0.37
            };
            (x, y)
        })
        .collect()
}

/// A target that stops moving: its mean is a pair as a feature's is, so the
/// cross-moments with the features that still move decay with the history,
/// as exact arithmetic has them, and the slopes with them -- to 2^-150 of
/// the 2 learned on `x0`, 150 half-lives on: measured, 1.35e-45 at every
/// level (1.26e-45 for `huber`). A mean stopped short of the target fed each
/// cross-moment that gap times the features' motion, and the slopes stayed
/// on it: -1.5e-16 at a level of 0.5, -2e-8 at 1e8, -1.6e-4 at 1e12 (the
/// bound is 1e-30).
fn slopes_decay<M: OnlineModel>(name: &str, make: impl Fn() -> M, slope: impl Fn(&M) -> f64) {
    for level in LEVELS {
        let mut m = make();
        let mut at_stop = f64::NAN;
        for (i, (x, y)) in held_target(level).iter().enumerate() {
            m.step(x, &[Some(*y)], d(i), 1.0);
            if i == MOVING - 1 {
                at_stop = slope(&m);
            }
        }
        let end = slope(&m);
        assert!(
            at_stop > 1.5,
            "{name} at level {level}: the case needs a slope, {at_stop}"
        );
        assert!(
            end.abs() <= 1e-30,
            "{name} at level {level}: slope {end:e} on x0, 150 half_lives after the target stopped"
        );
    }
}

#[test]
#[ignore = "extended: a second or more (1.3 s)"]
fn a_held_target_leaves_no_slope_on_a_moving_feature() {
    slopes_decay(
        "ewridge",
        || EwRidge::new(ridge(true)).unwrap(),
        |m: &EwRidge| m.coefficients().map_or(f64::NAN, |c| c[0][1]),
    );
    slopes_decay(
        "ewridge, unstandardized",
        || EwRidge::new(ridge(false)).unwrap(),
        |m: &EwRidge| m.coefficients().map_or(f64::NAN, |c| c[0][1]),
    );
    slopes_decay(
        "ewridge, blocked",
        || {
            let mut c = ridge(true);
            c.gram_block_rows = 16;
            c.solve_every = 16.0;
            c.max_rows_between_solves = 16;
            EwRidge::new(c).unwrap()
        },
        |m: &EwRidge| m.coefficients().map_or(f64::NAN, |c| c[0][1]),
    );
    let lasso = LassoCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        lasso_path: vec![0.0],
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: 10.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        max_iter: 1000,
        tol: 1e-30,
        target_gaps: TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    slopes_decay(
        "lasso",
        || Lasso::new(lasso.clone()).unwrap(),
        |m: &Lasso| m.coefficients().map_or(f64::NAN, |c| c[0][0][1]),
    );
    let huber = RobustCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        loss: RobustLoss::Huber { delta: 1.345 },
        ridge: 1e-6,
        standardize: true,
        min_weight: 10.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 0.05,
    };
    slopes_decay(
        "huber",
        || Robust::new(huber.clone()).unwrap(),
        |m: &Robust| m.coefficients().map_or(f64::NAN, |c| c[0][1]),
    );
}

/// What the models report of a held target: its variance decays with its
/// history, in `ewridge`'s target moments (the half of a saved Gram that
/// `R²` and the standard errors read) and in `marginal`'s pairs, where its
/// correlation with a moving feature goes to zero with it: 1.9e-23 at every
/// level, where exact arithmetic has 2^-75 of it. Each settled on a rounding
/// artefact instead: a variance of 4.2e-14 at 1e8, and a correlation of
/// -6.05e-2 at every level.
#[test]
fn a_held_target_is_reported_as_it_is() {
    for level in LEVELS {
        let mut ridge = EwRidge::new(ridge(true)).unwrap();
        let mut pairs = Marginal::new(MarginalCfg {
            n_features: 3,
            n_targets: 1,
            decay: decay(),
            min_weight: vec![0.0],
            lags: vec![],
            serial_rule: None,
            cross_lags: None,
            bins: None,
            feature_moments: online_core::FeatureMomentLayout::PerTarget,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let (mut ridge_at_stop, mut pair_at_stop) = (0.0, 0.0);
        for (i, (x, y)) in held_target(level).iter().enumerate() {
            ridge.step(x, &[Some(*y)], d(i), 1.0);
            pairs.step(x, &[Some(*y)], d(i), 1.0);
            if i == MOVING - 1 {
                ridge_at_stop = ridge.target_moments().unwrap().vars()[0];
                pair_at_stop = pairs.pair(0, 0).var_y;
            }
        }
        let var = ridge.target_moments().unwrap().vars()[0];
        assert!(
            var <= 1e-30 * ridge_at_stop,
            "level {level}: ewridge's target variance {var:e} of {ridge_at_stop:e}"
        );
        let p = pairs.pair(0, 0);
        assert!(
            p.var_y <= 1e-30 * pair_at_stop,
            "level {level}: marginal's target variance {:e} of {pair_at_stop:e}",
            p.var_y
        );
        assert!(
            p.corr.is_nan() || p.corr.abs() <= 1e-10,
            "level {level}: correlation {}",
            p.corr
        );
    }
}

/// Without decay a mean steps by `1/n`, and a plain one stops `n/2` rounding
/// steps short of a held value at row `n`, a gap exact arithmetic closes
/// only as `1/n`. At 1e12, where a rounding step is 1.2e-4, the variance of
/// a feature stopped after 300 rows settled on the gap's square by row
/// 20,000, and the standardized ridge's slope went from the 0.5 it learned
/// to 0.04 by row 300,000; with the means as pairs it stays at 0.50 to 0.52.
/// `sgd`'s predictions, which the stall took 2.2e-2 from the fit at 0.5,
/// stay within ten rounding steps of the level of it from the stop: 6.6e-4
/// measured. Setting the mean to the value at the stall, the first fix built
/// for this, moved them by 6.9e-2 at once; stepping a gap to it, the second,
/// by 3.5e-3. At 1e8 a plain mean stalls at row 120,000. `sgd`'s were within
/// 7e-7 while its coefficients were read through the scaler; past the
/// warm-up they are held in the caller's units (docs/PLAN.md task 206), so
/// the intercept carries the level times the slope, 5e11 here, each step
/// rounds it by up to 6e-5 and the prediction's products by as much: a few
/// rounding steps of the level, as every fit in the caller's units pays
/// (`steps_of`).
#[test]
#[ignore = "extended: a second or more (2.1 s)"]
fn without_decay_a_stopped_feature_keeps_its_slope() {
    let held = 100_000;
    let no_decay = |mut c: EwRidgeCfg| {
        c.decay = Decay::Halflife(f64::INFINITY);
        c
    };
    let slope = |m: &EwRidge| m.coefficients().map_or(f64::NAN, |c| c[0][3]);
    let (_, coef) = run_on(
        EwRidge::new(no_decay(ridge(true))).unwrap(),
        &stream_of(1e12, held),
        slope,
    );
    let (lo, hi) = coef[MOVING..]
        .iter()
        .filter(|c| c.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &c| {
            (a.min(c), b.max(c))
        });
    assert!(
        (0.2..0.8).contains(&lo) && (0.2..0.8).contains(&hi),
        "slope from {lo} to {hi} on the stopped feature"
    );
    let cfg = SgdCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        loss: SgdLoss::Squared,
        learning_rate: 0.01,
        schedule: LearningRate::Constant,
        l2: 0.0,
        min_weight: 10.0,
        standardize: true,
        strict_binary: false,
        clip_gradient: f64::INFINITY,
        constraint: None,
    };
    let (base, _) = run_on(
        Sgd::new(cfg.clone()).unwrap(),
        &stream_of(0.5, held),
        |_: &Sgd| f64::NAN,
    );
    let (pred, _) = run_on(Sgd::new(cfg).unwrap(), &stream_of(1e12, held), |_: &Sgd| {
        f64::NAN
    });
    let worst = (MOVING..pred.len())
        .map(|i| (pred[i] - base[i]).abs() / (1.0 + base[i].abs()))
        .fold(0.0, f64::max);
    assert!(
        worst <= steps_of(1e12) / 10.0,
        "sgd: predictions {worst:.3e} from those at 0.5"
    );
}

/// A state saved while a feature holds a value carries its means' low
/// parts, and the model restored from it goes on as the one that never
/// stopped, to the bit. Saved every 5 half-lives from 10 to 55 after the
/// feature stops at a level of 1e8, which spans the row a plain mean would
/// stall on (near 30 half-lives), through named msgpack as the bank writes a
/// state file. With the low parts left out of the state it fails.
fn resumes<M: OnlineModel>(name: &str, make: impl Fn() -> M) {
    let rows = stream(1e8);
    let step = |m: &mut M, i: usize| {
        let (x, y) = &rows[i];
        m.step(x, &[Some(*y)], d(i), 1.0).pred
    };
    let mut whole = make();
    let want: Vec<Vec<f64>> = (0..rows.len()).map(|i| step(&mut whole, i)).collect();
    for hl in (10..60).step_by(5) {
        let split = MOVING + hl * H as usize;
        let mut first = make();
        for i in 0..split {
            step(&mut first, i);
        }
        let bytes = rmp_serde::to_vec_named(&first.state()).unwrap();
        let mut resumed = M::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (i, want) in want.iter().enumerate().skip(split) {
            let got = step(&mut resumed, i);
            let same = got.len() == want.len()
                && got
                    .iter()
                    .zip(want)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                same,
                "{name}, saved {hl} half_lives in: row {i}, {got:?} against {want:?}"
            );
        }
    }
}

#[test]
#[ignore = "extended: a second or more (1.9 s)"]
fn a_state_saved_mid_hold_resumes_to_the_bit() {
    resumes("ewridge", || EwRidge::new(ridge(true)).unwrap());
    resumes("ewridge, blocked", || {
        let mut c = ridge(true);
        c.gram_block_rows = 16;
        c.solve_every = 16.0;
        c.max_rows_between_solves = 16;
        EwRidge::new(c).unwrap()
    });
    resumes("lasso", || {
        Lasso::new(LassoCfg {
            n_features: 3,
            n_targets: 1,
            fit_intercept: true,
            decay: decay(),
            lasso_path: vec![0.05, 0.0],
            l1_ratio: 1.0,
            select_half_life: None,
            min_weight: 10.0,
            target_min_weight: Vec::new(),
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            max_iter: 1000,
            tol: 1e-12,
            target_gaps: TargetGaps::OwnRows,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap()
    });
    resumes("huber", || {
        Robust::new(RobustCfg {
            n_features: 3,
            n_targets: 1,
            fit_intercept: true,
            decay: decay(),
            loss: RobustLoss::Huber { delta: 1.345 },
            ridge: 1e-6,
            standardize: true,
            min_weight: 10.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            quantile_eps: 0.05,
        })
        .unwrap()
    });
    resumes("kalman", || {
        Kalman::new(KalmanCfg {
            n_features: 3,
            n_targets: 1,
            fit_intercept: true,
            decay: decay(),
            half_life: vec![200.0],
            q: None,
            obs_var: None,
            p0: 1.0,
            share_p: false,
            min_weight: 10.0,
            revert_half_life: vec![f64::INFINITY],
            standardize: true,
        })
        .unwrap()
    });
    resumes("sgd", || {
        Sgd::new(SgdCfg {
            n_features: 3,
            n_targets: 1,
            fit_intercept: true,
            decay: decay(),
            loss: SgdLoss::Squared,
            learning_rate: 0.01,
            schedule: LearningRate::Constant,
            l2: 0.0,
            min_weight: 10.0,
            standardize: true,
            strict_binary: false,
            clip_gradient: f64::INFINITY,
            constraint: None,
        })
        .unwrap()
    });
    resumes("marginal", || {
        Marginal::new(MarginalCfg {
            n_features: 3,
            n_targets: 1,
            decay: decay(),
            min_weight: vec![0.0],
            lags: vec![],
            serial_rule: None,
            cross_lags: None,
            bins: None,
            feature_moments: online_core::FeatureMomentLayout::PerTarget,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap()
    });
}

/// `marginal`'s bins over a target that stops: each bin's mean is a pair, as
/// the pairs' means are, so each bin's variance decays with its history,
/// and the split gain -- the between-bin share of the target's variance --
/// goes to zero with it. Measured: a gain of 0.52 at the stop, and 0 at every
/// level 150 half-lives on (3.4e-46 where the target holds zero), with the
/// bins' variances at 4.7e-45. A bin's mean that stopped short left its `m2`
/// fed that gap: the variances settled on 4.9e-32 at 0.5 and 6.0e-8 at
/// 1e12, and the gain on a ratio of rounding artefacts, 0.45 with a `t` of
/// 6.7 at every level.
#[test]
fn a_held_target_leaves_no_split_in_the_bins() {
    for level in LEVELS {
        let mut m = Marginal::new(MarginalCfg {
            n_features: 3,
            n_targets: 1,
            decay: decay(),
            min_weight: vec![0.0],
            lags: vec![],
            serial_rule: None,
            cross_lags: None,
            bins: Some(Box::new(BinCfg {
                n_bins: 4,
                edges: None,
                rule: BinRule::Quantile,
                warm_rows: 100,
                budget_mib: None,
            })),
            feature_moments: online_core::FeatureMomentLayout::PerTarget,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap();
        let (mut gain_at_stop, mut var_at_stop) = (0.0, 0.0);
        for (i, (x, y)) in held_target(level).iter().enumerate() {
            m.step(x, &[Some(*y)], d(i), 1.0);
            if i == MOVING - 1 {
                let p = m.pair(0, 0);
                gain_at_stop = p.split_gain;
                var_at_stop = p.bin_var_y.iter().copied().fold(0.0, f64::max);
            }
        }
        let p = m.pair(0, 0);
        assert!(
            gain_at_stop > 0.1,
            "level {level}: the case needs a split, {gain_at_stop}"
        );
        assert!(
            p.split_gain <= 1e-10,
            "level {level}: split gain {:e}, 150 half_lives after the target stopped",
            p.split_gain
        );
        for (b, v) in p.bin_var_y.iter().enumerate() {
            assert!(
                *v <= 1e-30 * var_at_stop,
                "level {level}: bin {b}'s variance {v:e} of {var_at_stop:e}"
            );
        }
    }
}

/// A row of weight 0 learns nothing (CLAUDE.md hard rule 9), so the values
/// it carries change nothing. Every model runs a hold twice: once with its
/// rows of no weight carrying the held feature's value and the stream's
/// target, and once carrying 5 more and 99 more. Its predictions on every
/// other row agree to the bit. The second fix built for task 101, which
/// stepped a gap toward a value it detected as held, broke this: a row of
/// no weight ended no run, so it found the slot held while carrying another
/// value, and moved the mean to that value less the gap for the next row to
/// read.
fn no_weight_moves_nothing<M: OnlineModel>(name: &str, make: impl Fn() -> M) {
    let zero = |i: usize| i >= MOVING && i % 7 == 3;
    for level in [1e3, 1e8, -1e8, 1e12] {
        for (what, rows) in [("feature", stream(level)), ("target", held_target(level))] {
            let run = |far: bool| -> Vec<Vec<f64>> {
                let mut m = make();
                let rows = rows.iter().enumerate();
                rows.map(|(i, (x, y))| {
                    let (mut x, mut y) = (*x, *y);
                    if far && zero(i) {
                        (x[2], y) = (x[2] + 5.0, y + 99.0);
                    }
                    let w = if zero(i) { 0.0 } else { 1.0 };
                    m.step(&x, &[Some(y)], d(i), w).pred
                })
                .collect()
            };
            let (near, far) = (run(false), run(true));
            for i in (0..rows.len()).filter(|&i| !zero(i)) {
                let same = near[i]
                    .iter()
                    .zip(&far[i])
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(
                    same,
                    "{name}, held {what}, level {level}: row {i}, {:?} against {:?}",
                    far[i], near[i]
                );
            }
        }
    }
}

#[test]
#[ignore = "extended: a second or more (3.4 s); model_contract's per-model tests keep hard rule 9"]
fn a_row_of_no_weight_moves_no_mean() {
    no_weight_moves_nothing("ewridge", || EwRidge::new(ridge(true)).unwrap());
    no_weight_moves_nothing("ewridge, blocked", || {
        let mut c = ridge(true);
        c.gram_block_rows = 16;
        c.solve_every = 16.0;
        c.max_rows_between_solves = 16;
        EwRidge::new(c).unwrap()
    });
    no_weight_moves_nothing("lasso", || {
        Lasso::new(LassoCfg {
            n_features: 3,
            n_targets: 1,
            fit_intercept: true,
            decay: decay(),
            lasso_path: vec![0.05, 0.0],
            l1_ratio: 1.0,
            select_half_life: None,
            min_weight: 10.0,
            target_min_weight: Vec::new(),
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            max_iter: 1000,
            tol: 1e-12,
            target_gaps: TargetGaps::OwnRows,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap()
    });
    no_weight_moves_nothing("huber", || {
        Robust::new(RobustCfg {
            n_features: 3,
            n_targets: 1,
            fit_intercept: true,
            decay: decay(),
            loss: RobustLoss::Huber { delta: 1.345 },
            ridge: 1e-6,
            standardize: true,
            min_weight: 10.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            quantile_eps: 0.05,
        })
        .unwrap()
    });
    no_weight_moves_nothing("kalman", || {
        Kalman::new(KalmanCfg {
            n_features: 3,
            n_targets: 1,
            fit_intercept: true,
            decay: decay(),
            half_life: vec![200.0],
            q: None,
            obs_var: None,
            p0: 1.0,
            share_p: false,
            min_weight: 10.0,
            revert_half_life: vec![f64::INFINITY],
            standardize: true,
        })
        .unwrap()
    });
    no_weight_moves_nothing("sgd", || {
        Sgd::new(SgdCfg {
            n_features: 3,
            n_targets: 1,
            fit_intercept: true,
            decay: decay(),
            loss: SgdLoss::Squared,
            learning_rate: 0.01,
            schedule: LearningRate::Constant,
            l2: 0.0,
            min_weight: 10.0,
            standardize: true,
            strict_binary: false,
            clip_gradient: f64::INFINITY,
            constraint: None,
        })
        .unwrap()
    });
    no_weight_moves_nothing("marginal", || {
        Marginal::new(MarginalCfg {
            n_features: 3,
            n_targets: 1,
            decay: decay(),
            min_weight: vec![0.0],
            lags: vec![],
            serial_rule: None,
            cross_lags: None,
            bins: None,
            feature_moments: online_core::FeatureMomentLayout::PerTarget,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap()
    });
    no_weight_moves_nothing("ew_cov", || {
        EwCovModel::new(EwCovCfg {
            n_features: 3,
            decay: decay(),
            stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
            min_weight: 0.0,
            precision_prior: None,
            mahal_quantiles: vec![],
            pca: 0,
            pca_every: 0.0,
            max_rows_between_pca: u32::MAX,
            lags: vec![],
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap()
    });
    // Task 209 (b): the regressions the harness left out.
    no_weight_moves_nothing("quantile", || Robust::new(quantile_cfg()).unwrap());
    no_weight_moves_nothing("pa", || Pa::new(pa_cfg()).unwrap());
    no_weight_moves_nothing("rls", || Rls::new(rls_cfg()).unwrap());
    no_weight_moves_nothing("ftrl", || Ftrl::new(ftrl_cfg(FtrlLoss::Squared)).unwrap());
}

// --- task 110: the models the week's review left out -----------------------

/// A label from the stream's target: which side of its middle it fell.
fn label_of(y: f64) -> f64 {
    f64::from(y > 1.0)
}

/// `ew_class`: the class posteriors at every level are those at 0.5, and the
/// stopped feature's spread inside each class decays with its history rather
/// than settling on a floor (docs/PLAN.md task 110). A Gaussian classifier's
/// posterior does not move when a feature is shifted by a constant, so the
/// level is invisible in exact arithmetic.
#[test]
fn ew_class_classifies_the_same_at_every_level() {
    let make = || {
        EwClass::new(EwClassCfg {
            n_features: 3,
            n_classes: 2,
            decay: decay(),
            min_weight: 10.0,
            covariance: Covariance::Full,
            precision_prior: 1e-3,
            window: None,
            window_every: None,
            max_rows_between_snapshots: None,
        })
        .unwrap()
    };
    let run = |level: f64| {
        let mut m = make();
        let (mut post, mut at_stop) = (Vec::new(), [0.0; 2]);
        for (i, (x, y)) in stream(level).iter().enumerate() {
            let out = m.step(x, &[Some(label_of(*y))], d(i), 1.0);
            post.push(out.pred[1]);
            if i == MOVING - 1 {
                at_stop = [m.class_cov(0).var(2), m.class_cov(1).var(2)];
            }
        }
        (
            post,
            at_stop,
            [m.class_cov(0).var(2), m.class_cov(1).var(2)],
        )
    };
    let (base, _, _) = run(0.5);
    for level in LEVELS {
        let (post, at_stop, end) = run(level);
        for c in 0..2 {
            assert!(
                end[c] <= 1e-30 * at_stop[c],
                "level {level}, class {c}: variance {:e} of {:e}",
                end[c],
                at_stop[c]
            );
        }
        // Every row after the stop is compared, and both sides are numbers:
        // a filter on the base's alone counted nothing, and `f64::max`
        // drops a NaN, so a model that stopped reporting passed (review
        // 2026-10-06, CF7, as `holds` since CF6).
        let compared = (MOVING..post.len())
            .filter(|&i| base[i].is_finite() && post[i].is_finite())
            .count();
        assert_eq!(
            compared,
            post.len() - MOVING,
            "level {level}: a posterior on every row after the stop, at 0.5 and here"
        );
        let worst = (MOVING..post.len())
            .map(|i| (post[i] - base[i]).abs())
            .fold(0.0, f64::max);
        // Measured: 9.5e-15 at 1e3, 1.8e-9 at 1e8, 1.5e-5 at 1e12, about a
        // thousandth of `steps_of`.
        assert!(
            worst <= steps_of(level),
            "level {level}: posteriors {worst:e} from 0.5's"
        );
    }
}

/// `hmm`, seeded from its own warm-up rows: the filtered state probabilities
/// and the row's log-likelihood at every level are those at 0.5, and each
/// state's spread in the stopped feature decays (task 110). A Gaussian
/// emission shifted with its data is the same density.
#[test]
fn hmm_filters_the_same_at_every_level() {
    let make = || {
        Hmm::new(HmmCfg {
            n_features: 3,
            k: 2,
            decay: decay(),
            covariance: Covariance::Full,
            precision_prior: 1e-3,
            min_weight: 10.0,
            learn: true,
            transition_prior: 1.0,
            transition: None,
            means: None,
            covs: None,
            warm_rows: 40,
            seed_rule: SeedRule::First,
            seed: 3,
            tvtp: None,
        })
        .unwrap()
    };
    let run = |level: f64| {
        let mut m = make();
        let (mut out, mut at_stop) = (Vec::new(), [0.0; 2]);
        for (i, (x, _)) in stream(level).iter().enumerate() {
            let s = m.step(x, &[], d(i), 1.0);
            out.push((s.pred[0], s.pred[s.pred.len() - 1]));
            if i == MOVING - 1 {
                at_stop = [m.state_cov(0).var(2), m.state_cov(1).var(2)];
            }
        }
        (out, at_stop, [m.state_cov(0).var(2), m.state_cov(1).var(2)])
    };
    // Each state's weight is its share of the rows, so its spread decays at
    // that share of the rate: 2.7e-22 of its value at the stop, measured at
    // 0.5. What a stall would leave is a floor that grows with the level
    // (the gap squared: 5e-8 at 1e12), so each level is held to 0.5's.
    let (base, _, base_end) = run(0.5);
    for level in LEVELS {
        let (out, at_stop, end) = run(level);
        for s in 0..2 {
            assert!(
                end[s] <= 1e-15 * at_stop[s] && end[s] <= 4.0 * base_end[s] + 1e-300,
                "level {level}, state {s}: variance {:e} of {:e}, against {:e} at 0.5",
                end[s],
                at_stop[s],
                base_end[s]
            );
        }
        // Every row after the stop is compared, and both sides are numbers
        // (review 2026-10-06, CF7): a skip of the base's NaN counted
        // nothing, and `f64::max` drops a NaN on this side.
        let (mut worst_p, mut worst_ll) = (0.0f64, 0.0f64);
        for i in MOVING..out.len() {
            let ((p, ll), (bp, bll)) = (out[i], base[i]);
            assert!(
                [p, ll, bp, bll].iter().all(|v| v.is_finite()),
                "level {level}, row {i}: p {p} and loglik {ll} against {bp} and {bll} at 0.5"
            );
            worst_p = worst_p.max((p - bp).abs());
            worst_ll = worst_ll.max((ll - bll).abs() / (1.0 + bll.abs()));
        }
        // Measured: p 3.7e-14 at -0.37, 1.0e-12 at 1e3, 1.1e-7 at 1e8,
        // 5.9e-4 at 1e12 -- a twentieth of `steps_of` and less, but for the
        // levels near zero, where `steps_of` floors at 100 eps and the
        // filter's softmax lifts the data's own last bits past it; a stall
        // moved these fits by 4e-2 and more at 1e12.
        let tol = steps_of(level).max(1e-12);
        assert!(worst_p <= tol, "level {level}: p {worst_p:e}");
        assert!(worst_ll <= tol, "level {level}: loglik {worst_ll:e}");
    }
}

/// [`no_weight_moves_nothing`] for a model with no target: a row of weight 0
/// whose features are elsewhere changes nothing any later row reports, to
/// the bit (hard rule 9; task 110).
fn no_weight_moves_nothing_unsupervised<M: OnlineModel>(name: &str, make: impl Fn() -> M) {
    let zero = |i: usize| i >= MOVING && i % 7 == 3;
    for level in [0.5, 1e3, 1e8, -1e8] {
        let rows = stream_of(level, 400);
        let run = |far: bool| -> Vec<Vec<f64>> {
            let mut m = make();
            rows.iter()
                .enumerate()
                .map(|(i, (x, _))| {
                    let mut x = *x;
                    if far && zero(i) {
                        x[2] += 5.0;
                        x[0] -= 3.0;
                    }
                    let w = if zero(i) { 0.0 } else { 1.0 };
                    let s = m.step(&x, &[], d(i), w);
                    s.pred.iter().copied().chain([s.n_eff]).collect()
                })
                .collect()
        };
        let (near, far) = (run(false), run(true));
        for i in (0..rows.len()).filter(|&i| !zero(i)) {
            let same = near[i]
                .iter()
                .zip(&far[i])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                same,
                "{name}, level {level}: row {i}, {:?} against {:?}",
                far[i], near[i]
            );
        }
    }
}

/// The `bocpd` both checks below run: a per-row hazard of 50 rows, or `τ =
/// 30` clock units between changepoints on the clock (docs/PLAN.md task
/// 179).
fn bocpd_held(on_clock: bool) -> Bocpd {
    Bocpd::new(BocpdCfg {
        n_features: 3,
        hazard: if on_clock { 30.0 } else { 50.0 },
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: Some(vec![0.0; 3]),
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-6,
        max_run: 200,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: on_clock,
    })
    .unwrap()
}

/// **The exception, named: `bocpd` with its hazard on the clock**
/// (docs/PLAN.md task 179). Hard rule 9 says a row of weight 0 advances the
/// clock and learns nothing. Under a per-row hazard the clock's advance
/// means nothing to the run-length posterior, and such a row moves nothing,
/// to the bit. On the clock the clock's advance *is* the step's chance of a
/// break, so the row moves the posterior by exactly that, as decay moves a
/// decaying model's sums, and grows no run. What it carries still moves
/// nothing (the test above). `bocpd.rs`'s
/// `two_steps_with_nothing_learned_between_compose_into_one` holds such a
/// row to the step it folds into, to rounding.
#[test]
fn a_row_of_no_weight_on_the_clock_moves_the_run_lengths_by_its_step() {
    let (gap, tau) = (4.0f64, 30.0f64);
    for on_clock in [false, true] {
        let mut m = bocpd_held(on_clock);
        for (i, (x, _)) in stream_of(0.5, 60).iter().take(80).enumerate() {
            m.step(x, &[], d(i), 1.0);
        }
        let (before, runs, n_eff) = (
            m.run_posterior(),
            serde_json::to_value(&m).unwrap()["runs"].clone(),
            m.n_eff(),
        );
        m.step(&[9.0, -9.0, 9.0], &[], gap, 0.0);
        let after = m.run_posterior();
        assert_eq!(
            serde_json::to_value(&m).unwrap()["runs"],
            runs,
            "{on_clock}"
        );
        assert_eq!(m.n_eff().to_bits(), n_eff.to_bits(), "{on_clock}");
        if !on_clock {
            let bits = |p: &[f64]| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&after),
                bits(&before),
                "a per-row hazard moves nothing"
            );
            continue;
        }
        // From the definition: each run keeps `exp(-d/τ)` of its mass, and
        // the empty run takes the rest.
        let keep = (-gap / tau).exp();
        let want0 = before[0] + (1.0 - keep) * (1.0 - before[0]);
        assert!(
            (after[0] - want0).abs() < 1e-14,
            "{} against {want0}",
            after[0]
        );
        for (j, (a, b)) in after.iter().zip(&before).enumerate().skip(1) {
            assert!(
                (a - keep * b).abs() < 1e-14,
                "run {j}: {a} against {}",
                keep * b
            );
        }
        assert!(after[0] > before[0] + 0.1, "the time passed");
    }
}

#[test]
fn a_row_of_no_weight_moves_nothing_without_a_target() {
    no_weight_moves_nothing_unsupervised("bocpd", || bocpd_held(false));
    // On the clock a row of weight 0 still applies its step's chance of a
    // break (the exception named below); what it carries moves nothing, to
    // the bit, all the same.
    no_weight_moves_nothing_unsupervised("bocpd on the clock", || bocpd_held(true));
    no_weight_moves_nothing_unsupervised("deco", || {
        Deco::new(DecoCfg {
            n_features: 3,
            decay: decay(),
            dynamics: DecoDynamics::Ew,
            alpha: None,
            beta: None,
            blocks: Vec::new(),
            min_weight: 3.0,
        })
        .unwrap()
    });
    no_weight_moves_nothing_unsupervised("hmm", || {
        Hmm::new(HmmCfg {
            n_features: 3,
            k: 2,
            decay: decay(),
            covariance: Covariance::Full,
            precision_prior: 1e-3,
            min_weight: 10.0,
            learn: true,
            transition_prior: 1.0,
            transition: None,
            means: None,
            covs: None,
            warm_rows: 40,
            seed_rule: SeedRule::First,
            seed: 3,
            tvtp: None,
        })
        .unwrap()
    });
    // Task 209 (b): the kinds the harness left out.
    no_weight_moves_nothing_unsupervised("kmeans", || KMeans::new(kmeans_cfg()).unwrap());
    no_weight_moves_nothing_unsupervised("micro", || Micro::new(micro_cfg()).unwrap());
    no_weight_moves_nothing_unsupervised("corrchange", || {
        CorrChange::new(corrchange_cfg()).unwrap()
    });
}

/// Two targets under `pairwise` gaps, the second absent on every third row,
/// and a feature that stops: each target keeps the slope it learned on the
/// stopped feature, and predicts at every level what it does at 0.5 (task
/// 101's review; task 110). Under `pairwise` the Gram is over every row and
/// each target's own feature mean is kept beside it (`gaps.rs`, `Cross`),
/// the arrangement the stall would have hit twice.
#[test]
fn two_targets_under_pairwise_gaps_keep_their_slopes() {
    let make = || {
        let mut c = ridge(true);
        c.n_targets = 2;
        c.target_gaps = TargetGaps::Pairwise;
        EwRidge::new(c).unwrap()
    };
    let run = |level: f64| {
        let mut m = make();
        let (mut pred, mut slope) = (Vec::new(), Vec::new());
        for (i, (x, y)) in stream(level).iter().enumerate() {
            let second = (i % 3 != 1).then_some(0.5 * y + 0.2);
            let out = m.step(x, &[Some(*y), second], d(i), 1.0);
            pred.push([out.pred[0], out.pred[1]]);
            slope.push(
                m.coefficients()
                    .map_or([f64::NAN; 2], |b| [b[0][3], b[1][3]]),
            );
        }
        (pred, slope)
    };
    // The first target, on every row, keeps the slope it learned. The
    // second's slope under `pairwise` wanders at every level, 0.5 included,
    // from before the feature stops: the Gram is over every row and its
    // cross-moments over a third fewer, and the two samples' mismatch leaks
    // between slopes (under `own_rows` it holds 0.19 to 0.29). What a stall
    // would do is move it with the level, so it is held to 0.5's.
    // Every row after the stop is held, both targets, both slopes, and both
    // sides of each comparison are numbers: `s[0].is_nan() || ...` passed a
    // model with no coefficients at all, and the second target's checks
    // skipped every NaN (review 2026-10-06, CF7).
    let (base, base_slope) = run(0.5);
    for level in LEVELS {
        let (pred, slope) = run(level);
        let mut worst_slope = 0.0f64;
        for (i, s) in slope.iter().enumerate().skip(MOVING) {
            assert!(
                (0.2..0.8).contains(&s[0]),
                "level {level}, row {i}: slope {}",
                s[0]
            );
            assert!(
                s[1].is_finite() && base_slope[i][1].is_finite(),
                "level {level}, row {i}: second slope {} against {} at 0.5",
                s[1],
                base_slope[i][1]
            );
            worst_slope = worst_slope.max((s[1] - base_slope[i][1]).abs());
        }
        // Measured: 4.8e-6 at 1e12, 9.7e-10 at 1e8.
        assert!(
            worst_slope <= steps_of(level),
            "level {level}: second slope {worst_slope:e}"
        );
        let rows = (MOVING..pred.len()).flat_map(|i| (0..2).map(move |t| (i, t)));
        let compared = rows
            .clone()
            .filter(|&(i, t)| base[i][t].is_finite() && pred[i][t].is_finite())
            .count();
        assert_eq!(
            compared,
            2 * (pred.len() - MOVING),
            "level {level}: both targets predicted on every row after the stop"
        );
        let worst = rows
            .map(|(i, t)| (pred[i][t] - base[i][t]).abs() / (1.0 + base[i][t].abs()))
            .fold(0.0, f64::max);
        assert!(
            worst <= steps_of(level),
            "level {level}: {worst:e} from 0.5's"
        );
    }
}

// --- task 209 (b): the models the harnesses left out -----------------------

/// `quantile` at the median, standardized as `huber` is above, with the
/// bank's band (`quantile_eps` 0.2).
fn quantile_cfg() -> RobustCfg {
    RobustCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        loss: RobustLoss::Quantile { tau: 0.5 },
        ridge: 1e-6,
        standardize: true,
        min_weight: 10.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 0.2,
    }
}

/// `pa` at the bank's defaults: PA-I, `c` 1, the band 1% of the target's
/// spread, standardized.
fn pa_cfg() -> PaCfg {
    PaCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        mode: PaMode::Pa1,
        c: 1.0,
        eps: 0.01,
        min_weight: 10.0,
        constraint: None,
        standardize: true,
    }
}

fn rls_cfg() -> RlsCfg {
    RlsCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        delta: 1.0,
        coef_prior: None,
        min_weight: 10.0,
    }
}

fn ftrl_cfg(loss: FtrlLoss) -> FtrlCfg {
    FtrlCfg {
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        alpha: 0.1,
        beta: 1.0,
        l1: 0.0,
        l2: 1.0,
        min_weight: 10.0,
        strict_binary: false,
        loss,
    }
}

/// `quantile` keeps the slope it learned on the stopped feature, 0.31 to
/// 0.61 measured at every level, and the fit at each level predicts what
/// the fit at 0.5 does from the stop on: 1.8e-15 at -0.37, 1.5e-13 at 1e3,
/// 1.4e-8 at ±1e8 and 1.4e-4 at 1e12, a hundredth of [`steps_of`].
#[test]
fn a_standardized_quantile_keeps_the_slope_it_learned() {
    holds(
        "quantile",
        || Robust::new(quantile_cfg()).unwrap(),
        Some(|m: &Robust| m.coefficients().map_or(f64::NAN, |c| c[0][3])),
        FROM_THE_STOP,
    );
}

/// `pa` standardizes each row as `sgd` does and holds its fit in the
/// caller's units past the warm-up (docs/PLAN.md task 206), so, as `sgd`'s,
/// its slope on the stopped feature moves with the steps it takes (-0.05 to
/// 0.74 measured, at 0.5 as at every level) and is not held. What it
/// predicts is, from the stop on: 9.4e-16 at -0.37, 2.2e-13 at 1e3, 2.7e-8
/// at ±1e8 and 3.2e-4 at 1e12.
#[test]
fn pa_predicts_the_same_at_every_level() {
    holds("pa", || Pa::new(pa_cfg()).unwrap(), None, FROM_THE_STOP);
}

/// The root mean square of `pred - y` over the rows from `from` on.
fn rmse_from(pred: &[f64], rows: &[([f64; 3], f64)], from: usize) -> f64 {
    let n = (pred.len() - from) as f64;
    let sum: f64 = (from..pred.len())
        .map(|i| (pred[i] - rows[i].1).powi(2))
        .sum();
    (sum / n).sqrt()
}

/// **`rls` is held to its accuracy, not to the fit at 0.5.** Its prior,
/// `delta I`, is on the sums and fades with them, the intercept's included
/// (`rls.rs`), so at a level the intercept the level puts on the fit is
/// penalized until the rows outweigh the prior. Over rows 200 to 300 its
/// error is 0.32 to 0.34 at 1e3 and beyond, where it is 0.18 at 0.5. And
/// once a feature holds a value other than 0, the information that tells its
/// coefficient from the intercept decays with nothing to renew it, as the
/// prior's does: some 50 half-lives in it is under a rounding step, and the
/// slope wanders, to ±1e13 at 0.5, ±5e9 at 1e3, ±3e5 at 1e8 and ±7 at 1e12
/// within the 150 measured, while held at exactly 0 (-0.37) it stays 0.45 to
/// 0.50 (docs/PLAN.md task 209's report raises it). The rows that carry the
/// held value read the sum of the two, which the rows keep in view: every
/// level's error from the stop on is 0.178 to 0.180, against 0.179 at 0.5,
/// and here within 2% of it.
#[test]
fn rls_predicts_as_well_at_every_level_once_the_feature_stops() {
    let rows = stream(0.5);
    let (base, _) = run(Rls::new(rls_cfg()).unwrap(), 0.5, |_: &Rls| f64::NAN);
    let want = rmse_from(&base, &rows, MOVING);
    assert!((0.15..0.2).contains(&want), "the case: {want}");
    for level in LEVELS {
        let rows = stream(level);
        let (pred, _) = run(Rls::new(rls_cfg()).unwrap(), level, |_: &Rls| f64::NAN);
        assert!(
            pred[MOVING..].iter().all(|p| p.is_finite()),
            "rls at level {level}: a row with no prediction after the stop"
        );
        let got = rmse_from(&pred, &rows, MOVING);
        println!("rls at level {level}: error {got:.4} from the stop, {want:.4} at 0.5");
        assert!(
            (got / want - 1.0).abs() <= 0.02,
            "rls at level {level}: error {got} from the stop, {want} at 0.5"
        );
    }
}

/// **`ftrl` does not standardize, and its fit depends on a feature's
/// level** (docs/PLAN.md task 209's report raises it): with the stream's
/// third feature at a level `L`, the squared loss's error over the rows
/// after the stop is about `0.018 L`, 18 at 1e3 and 1.8e6 at 1e8, where it
/// is 0.72 at 0.5. What holds at every level is the sign: each coordinate's
/// step is odd in its feature, so the fit with the feature at `-L`, the
/// stream mirrored, `-(L + u)`, predicts to the bit what the fit at `L`
/// does, under either loss.
#[test]
fn ftrl_at_a_level_of_either_sign_is_the_others_mirror() {
    for loss in [FtrlLoss::Squared, FtrlLoss::Logistic] {
        for level in [0.5, 1e3, 1e8, 1e12] {
            let fit = |sign: f64| -> Vec<u64> {
                let mut m = Ftrl::new(ftrl_cfg(loss)).unwrap();
                stream(level)
                    .iter()
                    .enumerate()
                    .map(|(i, (x, y))| {
                        let y = match loss {
                            FtrlLoss::Squared => *y,
                            FtrlLoss::Logistic => label_of(*y),
                        };
                        let x = [x[0], x[1], sign * x[2]];
                        let p = m.step(&x, &[Some(y)], d(i), 1.0).pred[0];
                        assert!(
                            i < MOVING || p.is_finite(),
                            "{loss:?} at {}: row {i} predicts {p}",
                            sign * level
                        );
                        p.to_bits()
                    })
                    .collect()
            };
            assert_eq!(fit(1.0), fit(-1.0), "{loss:?} at ±{level}");
        }
    }
}

/// What a model with no target reports at each of `levels` is what it
/// reports at 0.5, slot by slot and on every row: both a number within
/// `steps_of(level)` (at least 1e-12) of `1 + |at 0.5|`, or both NaN. A slot
/// in `shifted` reports in the stopped feature's own units, and is held less
/// the level, to `shifted_steps` rounding steps of the level.
fn reports_the_same_at_every_level<M: OnlineModel>(
    name: &str,
    make: impl Fn() -> M,
    (shifted, shifted_steps): (&[usize], f64),
    levels: &[f64],
) {
    let run = |level: f64| -> Vec<Vec<f64>> {
        let mut m = make();
        stream(level)
            .iter()
            .enumerate()
            .map(|(i, (x, _))| m.step(x, &[], d(i), 1.0).pred)
            .collect()
    };
    let base = run(0.5);
    for &level in levels {
        let out = run(level);
        let tol = steps_of(level).max(1e-12);
        let shifted_tol = (shifted_steps * level.abs() * f64::EPSILON).max(tol);
        let (mut worst, mut at, mut numbers) = (0.0f64, (0, 0), 0usize);
        let mut worst_shifted = 0.0f64;
        for (i, (got, want)) in out.iter().zip(&base).enumerate() {
            for (s, (&a, &b)) in got.iter().zip(want).enumerate() {
                assert_eq!(
                    a.is_nan(),
                    b.is_nan(),
                    "{name} at level {level}, row {i}, slot {s}: {a} against {b} at 0.5"
                );
                if a.is_nan() {
                    continue;
                }
                numbers += 1;
                if shifted.contains(&s) {
                    let off = ((a - level) - (b - 0.5)).abs();
                    worst_shifted = worst_shifted.max(off);
                    assert!(
                        off <= shifted_tol,
                        "{name} at level {level}, row {i}, slot {s}: {a} less the level is \
                         {off:.3e} from 0.5's, past {shifted_tol:.1e}"
                    );
                    continue;
                }
                let off = (a - b).abs() / (1.0 + b.abs());
                if off > worst {
                    (worst, at) = (off, (i, s));
                }
            }
        }
        println!(
            "{name} at level {level}: {worst:.1e} from 0.5's (row {}, slot {}), {worst_shifted:.1e} \
             in the shifted slots, over {numbers} numbers ({tol:.1e}, {shifted_tol:.1e})",
            at.0, at.1
        );
        assert!(
            numbers > 500,
            "{name} at level {level}: {numbers} numbers compared"
        );
        assert!(
            worst <= tol,
            "{name} at level {level}: {worst:.3e} from 0.5's at row {}, slot {}, past {tol:.1e}",
            at.0,
            at.1
        );
    }
}

fn kmeans_cfg() -> KMeansCfg {
    KMeansCfg {
        n_features: 3,
        k: 3,
        decay: decay(),
        min_weight: 10.0,
        warm_rows: 40,
        seed_rule: SeedRule::Lloyd,
        seed: 0,
        update_every_rows: 1,
        split_merge: 0.5,
        split_merge_every_rows: 50,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.1,
    }
}

fn micro_cfg() -> MicroCfg {
    MicroCfg {
        n_features: 3,
        decay: decay(),
        min_weight: 3.0,
        eps: 0.6,
        beta_mu: 2.0,
        max_clusters: 50,
        prune_every: f64::INFINITY,
        max_rows_between_prunes: 10,
        macro_link: None,
        standardize: true,
        scale_floor: 0.1,
    }
}

fn corrchange_cfg() -> CorrChangeCfg {
    CorrChangeCfg {
        n_features: 3,
        kind: CorrChangeKind::Monitor,
        span_rows: 20,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: decay(),
        crit: None,
        n_perm: 20,
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
    }
}

/// [`LEVELS`] and the held value's mirror at 1e12, where a centre's rounding
/// step is largest.
const BOTH_SIGNS: [f64; 7] = [0.5, -0.37, 1e3, 1e8, -1e8, 1e12, -1e12];

/// **The clusters' centres are compensated means** (`crate::comp`; docs/PLAN.md
/// task 215, D2). Plain, a centre at 1e12 stalled short of the held value by
/// up to the rounding step over its step's share, 1.8e-3, a gap the metric's
/// floor counts `2^(Q/8)` times more as the feature stays quiet for `Q`
/// half-lives (`scale_floor`, task 102); in exact arithmetic the gap closes
/// as `2^-Q`. Stalled, it grew into the distances, and `kmeans` read a
/// different cluster than at 0.5 from 60 half-lives after the stop (row
/// 1504), `micro` from 40 (row 1105), on 1315 and 1023 of the 3000 held rows
/// (task 209's report), so only the rows through 30 half-lives were held.
/// Now every row of the 150 half-lives is, at both signs of 1e12.
#[test]
fn kmeans_assigns_the_same_at_every_level() {
    reports_the_same_at_every_level(
        "kmeans",
        || KMeans::new(kmeans_cfg()).unwrap(),
        (&[], 0.0),
        &BOTH_SIGNS,
    );
}

#[test]
fn micro_assigns_the_same_at_every_level() {
    reports_the_same_at_every_level(
        "micro",
        || Micro::new(micro_cfg()).unwrap(),
        (&[], 0.0),
        &BOTH_SIGNS,
    );
}

/// A value held for 100 half-lives is reached: the centre the last row is
/// assigned to sits on the held value of the stopped feature, to the bit, at
/// 0.5 and at either sign of 1e12 (docs/PLAN.md task 215, D2). Exact
/// arithmetic leaves `2^-100` of the gap, under a rounding step of the
/// level; a plain mean stalled 1.8e-3 short at 1e12, about 15 rounding
/// steps.
#[test]
fn a_centre_reaches_a_value_held_at_a_level() {
    for level in [0.5, 1e12, -1e12] {
        let rows = stream_of(level, 100 * H as usize);
        let held = rows[rows.len() - 1].0[2];
        assert_eq!(held, level + 0.37);
        let mut km = KMeans::new(kmeans_cfg()).unwrap();
        let mut mc = Micro::new(micro_cfg()).unwrap();
        let (mut cluster, mut micro_id) = (f64::NAN, f64::NAN);
        for (i, (x, _)) in rows.iter().enumerate() {
            cluster = km.step(x, &[], d(i), 1.0).pred[0];
            micro_id = mc.step(x, &[], d(i), 1.0).pred[2];
        }
        let centres = km.coefficients().unwrap();
        let c = centres[cluster as usize][2];
        assert_eq!(
            c,
            held,
            "kmeans at level {level}: the centre is {:.3e} short of the held value",
            held - c
        );
        let summaries = mc.coefficients().unwrap();
        let row = summaries
            .iter()
            .find(|r| r[0] == micro_id)
            .unwrap_or_else(|| panic!("micro at level {level}: summary {micro_id} not potential"));
        let c = row[4 + 2];
        assert_eq!(
            c,
            held,
            "micro at level {level}: the centre is {:.3e} short of the held value",
            held - c
        );
    }
}

/// An equicorrelation reads the rows standardized, which a level leaves
/// alone: 2.1e-13 at 1e3, 1.6e-8 at ±1e8, 1.6e-4 at 1e12 measured.
#[test]
fn deco_reports_the_same_at_every_level() {
    reports_the_same_at_every_level(
        "deco",
        || {
            Deco::new(DecoCfg {
                n_features: 3,
                decay: decay(),
                dynamics: DecoDynamics::Ew,
                alpha: None,
                beta: None,
                blocks: Vec::new(),
                min_weight: 3.0,
            })
            .unwrap()
        },
        (&[], 0.0),
        &LEVELS,
    );
}

/// `bocpd` with its prior read from its own warm-up rows, so that the prior
/// sits at the level as the data do (a prior mean given at 0 does not, and
/// a level then is a surprise by design): the run-length posterior and the
/// row's log density at each level are those at 0.5, 4.3e-8 at ±1e8 and
/// 3.6e-4 at 1e12 measured. **The predictive mean of the stopped feature
/// (slot 5), less the level, is 1,040 rounding steps of the level from
/// 0.5's**, 2.3e-10 at 1e3, 2.3e-5 at 1e8 and 0.23 at 1e12, late in the
/// hold, and is held to 2,000. It is `Σ pᵣ mᵣ` over the runs, each `mᵣ` at
/// the level, and the run posterior sums to 1 only to 2.3e-13 there,
/// measured: that times the level is the whole of it. `Σ pᵣ (mᵣ − m₀) +
/// m₀` would keep it at the spread's scale (docs/PLAN.md task 209's report
/// raises it).
#[test]
fn bocpd_reports_the_same_at_every_level() {
    reports_the_same_at_every_level(
        "bocpd",
        || {
            Bocpd::new(BocpdCfg {
                n_features: 3,
                hazard: 50.0,
                hazard_from_row: false,
                emission: BocpdEmission::Diag,
                prior_mean: None,
                prior_kappa: 1.0,
                prior_nu: None,
                prior_scale: None,
                robust_beta: 0.0,
                prune_below: 1e-6,
                max_run: 200,
                min_weight: 0.0,
                warm_rows: None,
                hazard_on_clock: false,
            })
            .unwrap()
        },
        (&[5], 2e3),
        &LEVELS,
    );
}

/// A test of a change in correlation reads the rows centred: its statistic
/// at each level is the one at 0.5 (3.9e-9 at ±1e8, 3.1e-5 at 1e12
/// measured), and its flags and counts are the same.
#[test]
fn corrchange_tests_the_same_at_every_level() {
    reports_the_same_at_every_level(
        "corrchange",
        || CorrChange::new(corrchange_cfg()).unwrap(),
        (&[], 0.0),
        &LEVELS,
    );
}
