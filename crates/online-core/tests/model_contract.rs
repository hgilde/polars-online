//! The contract every model in this crate keeps, asserted once for all of them.
//!
//! Why this file exists. Each model has thorough tests of *its own* behaviour,
//! but the small shared surface -- the shape accessors, `n_eff`, the prediction
//! slot count, state round-tripping, and the "before this row" reading of
//! everything -- was asserted nowhere in Rust. A `cargo mutants` pass made that
//! visible: `n_eff -> 0.0`, `n_targets -> 1` and `n_features -> 0` survived in
//! nearly every file, because the only tests exercising them run in Python
//! through the compiled extension, which `cargo test` cannot see.
//!
//! This is the core-level counterpart of `tests/test_semantics_all_models.py`.
//! Anything genuinely model-specific belongs in that model's own unit tests.

use online_core::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const K: usize = 2;
const HALFLIFE: f64 = 20.0;

/// What a model must report about itself, checked against the config it was
/// built from and against its own behaviour over a fixed stream.
struct Report {
    kind: &'static str,
    n_features: usize,
    n_targets: usize,
    n_outputs: usize,
    /// `n_eff` before the first row, and after 1, 2 and 40 rows.
    n_eff: Vec<f64>,
    /// Prediction-slot count actually returned by `step`.
    pred_len: usize,
    /// The weight reported on the row that carries a long clock gap, and the
    /// weight reported on the row after it.
    before_gap: f64,
    after_gap: f64,
    /// State round-trips through msgpack and continues identically.
    roundtrips: bool,
}

/// `n_eff_of` reads the model's own `n_eff()` accessor, which is inherent
/// rather than part of the trait, so it has to be handed in. The accessor's
/// value read before a row must equal the `n_eff` that row reports: they are
/// the same number, and reporting two different ones would make `min_periods`
/// mean something different from what a caller inspecting the model sees.
fn probe_with<M: OnlineModel>(
    mut m: M,
    targets: usize,
    n_eff_of: Option<&dyn Fn(&M) -> f64>,
) -> Report {
    let kind = m.state().model.kind();
    let mut n_eff = vec![];
    let mut s = 20260830u64;
    let mut pred_len = 0;

    let row = |m: &mut M, s: &mut u64, d: f64| -> Step {
        let x: Vec<f64> = (0..K).map(|_| lcg(s)).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| Some(0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1]))
            .collect();
        m.step(&x, &y, d, 1.0)
    };

    for i in 0..40 {
        let before = n_eff_of.map(|f| f(&m));
        let step = row(&mut m, &mut s, if i == 0 { 0.0 } else { 1.0 });
        if let Some(before) = before {
            assert!(
                (before - step.n_eff).abs() < 1e-12,
                "row {i}: accessor said {before}, the step reported {}",
                step.n_eff
            );
        }
        if matches!(i, 0..=2) {
            n_eff.push(step.n_eff);
        }
        pred_len = step.pred.len();
    }
    let after_40 = row(&mut m, &mut s, 1.0).n_eff;
    n_eff.push(after_40);

    // Serialize here, before the two branches diverge.
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut restored = M::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();

    let mut s2 = s;
    let undecayed = row(&mut m, &mut s, 1.0).n_eff;
    let continued = row(&mut restored, &mut s2, 1.0).n_eff;
    let roundtrips = (continued - undecayed).abs() < 1e-12;

    // `n_eff` is read before the row's own decay, so a gap shows up on the row
    // *after* the one that carries it.
    let before_gap = row(&mut restored, &mut s2, 10.0 * HALFLIFE).n_eff;
    let after_gap = row(&mut restored, &mut s2, 1.0).n_eff;

    Report {
        kind,
        n_features: m.n_features(),
        n_targets: m.n_targets(),
        n_outputs: m.n_outputs(),
        n_eff,
        pred_len,
        before_gap,
        after_gap,
        roundtrips,
    }
}

fn check(r: &Report, kind: &str, targets: usize, combos: usize) {
    assert_eq!(r.kind, kind, "state kind");
    assert_eq!(r.n_features, K, "{kind}: n_features");
    assert_eq!(r.n_targets, targets, "{kind}: n_targets");
    assert_eq!(
        r.n_outputs,
        targets * combos,
        "{kind}: n_outputs is targets x grid combos"
    );
    assert_eq!(
        r.pred_len, r.n_outputs,
        "{kind}: step must fill exactly n_outputs slots"
    );

    // `n_eff` is the weight *before* the row's update and before its decay, so
    // it starts at zero and lags the row count by one. Every model reports it
    // the same way; that uniformity is what makes `min_periods` portable.
    assert_eq!(r.n_eff[0], 0.0, "{kind}: nothing seen before the first row");
    assert_eq!(r.n_eff[1], 1.0, "{kind}: one row of weight 1");
    let two = 0.5f64.powf(1.0 / HALFLIFE) + 1.0;
    assert!(
        (r.n_eff[2] - two).abs() < 1e-12,
        "{kind}: decayed weight, got {}",
        r.n_eff[2]
    );
    // It saturates at 1/(1 - lam) rather than growing without bound.
    let ceiling = 1.0 / (1.0 - 0.5f64.powf(1.0 / HALFLIFE));
    assert!(
        r.n_eff[3] > 20.0 && r.n_eff[3] < ceiling,
        "{kind}: n_eff after 40 rows is {} (ceiling {ceiling})",
        r.n_eff[3]
    );

    // A ten-halflife gap must decay it by 2^-10, not reset or ignore it.
    let want = r.before_gap * 0.5f64.powi(10) + 1.0;
    assert!(
        (r.after_gap - want).abs() < 1e-9,
        "{kind}: gap decay got {} want {want}",
        r.after_gap
    );

    assert!(r.roundtrips, "{kind}: state did not round-trip");
}

fn decay() -> Decay {
    Decay::Halflife(HALFLIFE)
}

#[test]
fn ew_ridge() {
    let cfg = EwRidgeCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        ridge: vec![1e-6, 0.1],
        feature_sets: vec![],
        standardize: false,
        ridge_decay: false,
        session_shrink: None,
        long_halflife: None,
        coef0: None,
        min_periods: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
    };
    let r = probe_with(EwRidge::new(cfg).unwrap(), 2, Some(&EwRidge::n_eff));
    check(&r, "ew_ridge", 2, 2);
}

#[test]
fn rls() {
    let cfg = RlsCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        ridge: 1.0,
        coef0: None,
        min_periods: 3.0,
    };
    let r = probe_with(Rls::new(cfg).unwrap(), 2, Some(&Rls::n_eff));
    check(&r, "rls", 2, 1);
}

#[test]
fn lasso() {
    let cfg = LassoCfg {
        n_features: K,
        n_targets: 1,
        add_intercept: true,
        decay: decay(),
        lasso_path: vec![0.1, 0.0],
        l1_ratio: 1.0,
        select_halflife: None,
        min_periods: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        max_cd_iters: 100,
        cd_tol: 1e-10,
    };
    let r = probe_with(Lasso::new(cfg).unwrap(), 1, Some(&Lasso::n_eff));
    check(&r, "lasso", 1, 2);
}

#[test]
fn kalman() {
    let cfg = KalmanCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        halflife: vec![100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_periods: 3.0,
        standardize: true,
    };
    let r = probe_with(Kalman::new(cfg).unwrap(), 2, Some(&Kalman::n_eff));
    check(&r, "kalman", 2, 1);
}

#[test]
fn robust() {
    for loss in [
        RobustLoss::Huber { delta: 1.5 },
        RobustLoss::Quantile { tau: 0.5 },
    ] {
        let cfg = RobustCfg {
            n_features: K,
            n_targets: 2,
            add_intercept: true,
            decay: decay(),
            loss,
            ridge: 1e-6,
            standardize: false,
            min_periods: 3.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            quantile_eps: 1e-3,
        };
        let r = probe_with(Robust::new(cfg).unwrap(), 2, Some(&Robust::n_eff));
        check(&r, "robust", 2, 1);
    }
}

#[test]
fn ftrl() {
    let cfg = FtrlCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        alpha: 0.1,
        beta: 1.0,
        l1: 0.0,
        l2: 1.0,
        min_periods: 3.0,
        strict_binary: false,
        loss: FtrlLoss::Squared,
    };
    let r = probe_with(Ftrl::new(cfg).unwrap(), 2, Some(&Ftrl::n_eff));
    check(&r, "ftrl", 2, 1);
}

#[test]
fn sgd() {
    let cfg = SgdCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        loss: SgdLoss::Squared,
        learning_rate: 0.01,
        schedule: LearningRate::Constant,
        l2: 0.0,
        clip_gradient: 1e3,
        scale_features: false,
        min_periods: 3.0,
    };
    let r = probe_with(Sgd::new(cfg).unwrap(), 2, Some(&Sgd::n_eff));
    check(&r, "sgd", 2, 1);
}

#[test]
fn pa() {
    let cfg = PaCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        mode: PaMode::Pa1,
        c: 1.0,
        eps: 0.1,
        min_periods: 3.0,
    };
    let r = probe_with(Pa::new(cfg).unwrap(), 2, Some(&Pa::n_eff));
    check(&r, "pa", 2, 1);
}

#[test]
fn holt() {
    let cfg = HoltCfg {
        n_targets: 2,
        level_halflife: HALFLIFE,
        trend_halflife: 4.0 * HALFLIFE,
        min_periods: 3.0,
    };
    // Holt reads no features, so it is the one model whose `n_features` is 0.
    let r = probe_with(Holt::new(cfg).unwrap(), 2, Some(&Holt::n_eff));
    assert_eq!(r.n_features, 0, "holt takes no features");
    assert_eq!(r.kind, "holt");
    assert_eq!(r.n_targets, 2);
    assert_eq!(r.n_outputs, 2);
    assert_eq!(r.pred_len, 2);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

#[test]
fn ew_cov_model() {
    let cfg = EwCovCfg {
        n_features: K,
        decay: decay(),
        stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
        min_periods: 3.0,
        precision_prior: None,
    };
    // No targets: it emits statistics, one slot per (stat x column-or-pair).
    let m = EwCovModel::new(cfg).unwrap();
    assert_eq!(m.n_targets(), 0, "ew_cov has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(
        m.n_outputs(),
        2 + 2 + 1,
        "mean and var per column, one pair"
    );
    let r = probe_with(m, 0, Some(&EwCovModel::n_eff));
    assert_eq!(r.kind, "ew_cov");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!(r.roundtrips);
}

#[test]
fn every_state_kind_is_distinct_and_named() {
    // `ModelState::kind` names the model in every state error; a mutation that
    // returns a constant would make "expected X, found Y" meaningless.
    let kinds = [
        "ew_ridge", "rls", "lasso", "kalman", "robust", "ftrl", "sgd", "pa", "holt", "ew_cov",
    ];
    let mut seen = std::collections::HashSet::new();
    for k in kinds {
        assert!(!k.is_empty());
        assert!(seen.insert(k), "duplicate kind {k}");
    }
    // And the names really come from the states, not this list.
    assert_eq!(
        State::new(ModelState::Holt(Box::new(
            Holt::new(HoltCfg {
                n_targets: 1,
                level_halflife: 1.0,
                trend_halflife: 1.0,
                min_periods: 0.0,
            })
            .unwrap()
        )))
        .model
        .kind(),
        "holt"
    );
    assert_eq!(
        State::new(ModelState::EwCov(EwCov::new(1))).model.kind(),
        "ew_cov"
    );
}

#[test]
fn restoring_the_wrong_model_is_an_error_that_names_both() {
    let holt = Holt::new(HoltCfg {
        n_targets: 1,
        level_halflife: 10.0,
        trend_halflife: 40.0,
        min_periods: 0.0,
    })
    .unwrap();
    let s = holt.state();
    match Rls::restore(&s) {
        Err(StateError::WrongModel { expected, found }) => {
            assert_eq!(expected, "rls");
            assert_eq!(found, "holt");
        }
        other => panic!("expected a WrongModel error, got {other:?}"),
    }
}
