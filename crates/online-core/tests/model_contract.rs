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
    /// `clear_lags` on a warm model left every byte of the state where it
    /// was. True for every model that keeps nothing indexed by rows back,
    /// which is the default; a model with a ring says so in `KEEPS_LAGS`.
    lags_are_a_no_op: bool,
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
    let undecayed = row(&mut m, &mut s, 1.0);
    let continued = row(&mut restored, &mut s2, 1.0);
    // Every slot, not only `n_eff`: a state that restores its weights but
    // loses a ring reports the same `n_eff` and a different row (the
    // `ew_cov` lag ring did exactly that, docs/REVIEW-E54-E64.md L1).
    let roundtrips = (continued.n_eff - undecayed.n_eff).abs() < 1e-12
        && continued.pred.len() == undecayed.pred.len()
        && continued
            .pred
            .iter()
            .zip(&undecayed.pred)
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));

    // `n_eff` is read before the row's own decay, so a gap shows up on the row
    // *after* the one that carries it.
    let before_gap = row(&mut restored, &mut s2, 10.0 * HALFLIFE).n_eff;
    let after_gap = row(&mut restored, &mut s2, 1.0).n_eff;

    // Task 47: `clear_lags` drops what is indexed by rows back and *nothing
    // else*. For a model with no such state that is the whole of it, so the
    // bytes must not move; for one with a ring, only the ring may.
    let before_clear = rmp_serde::to_vec(&m.state()).unwrap();
    m.clear_lags();
    let lags_are_a_no_op = rmp_serde::to_vec(&m.state()).unwrap() == before_clear;
    // Asserted here rather than in `check`, so that the models with outputs
    // of their own -- which assert individually instead of calling it -- are
    // held to it too.
    assert_eq!(
        lags_are_a_no_op,
        !KEEPS_LAGS.contains(&kind),
        "{kind}: clear_lags moved state that is not a lag ring (or a model \
         with a ring is missing from KEEPS_LAGS)"
    );

    // A state saved with an *empty* ring must resume like the model it was
    // copied from. Read a ring's depth from a container's capacity instead
    // of from the configuration and it never refills after this copy, which
    // is what `ew_cov`'s lags did (docs/REVIEW-E54-E64.md L1).
    let mut after_clear =
        M::restore(&rmp_serde::from_slice(&rmp_serde::to_vec(&m.state()).unwrap()).unwrap())
            .unwrap();
    let mut s3 = s;
    for i in 0..8 {
        let a = row(&mut m, &mut s, 1.0);
        let b = row(&mut after_clear, &mut s3, 1.0);
        assert!(
            a.pred.len() == b.pred.len()
                && a.pred
                    .iter()
                    .zip(&b.pred)
                    .all(|(x, y)| x == y || (x.is_nan() && y.is_nan())),
            "{kind}: a state saved just after clear_lags diverged by row {i}"
        );
    }

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
        lags_are_a_no_op,
    }
}

/// The models that keep state indexed by *rows back*, and so may legally
/// move under `clear_lags`. Everything else must not: `clear_lags` is not a
/// reset, and a model that quietly threw away a mean here would look like a
/// decay bug three chunks later (docs/PLAN.md task 47).
const KEEPS_LAGS: &[&str] = &["rcov", "corrchange", "ew_cov", "marginal"];

/// Models that report on some rows and not others by design -- a span-based
/// test writes its statistic where the span closes -- so the predict-parity
/// helper cannot ask for 300 rows with every slot ready.
const SPARSE_OUTPUT: &[&str] = &["corrchange"];

fn check(r: &Report, kind: &str, targets: usize, combos: usize) {
    assert_eq!(r.kind, kind, "state kind");
    assert!(r.lags_are_a_no_op || KEEPS_LAGS.contains(&kind));
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

fn ew_ridge_cfg() -> EwRidgeCfg {
    EwRidgeCfg {
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
        coef_prior: None,
        min_periods: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        gram_block_rows: 0,
        target_gaps: online_core::TargetGaps::OwnRows,
        window: None,
        window_every: None,
    }
}

#[test]
fn ew_ridge() {
    let cfg = ew_ridge_cfg();
    let r = probe_with(EwRidge::new(cfg).unwrap(), 2, Some(&EwRidge::n_eff));
    check(&r, "ew_ridge", 2, 2);
}

fn rls_cfg() -> RlsCfg {
    RlsCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        ridge: 1.0,
        coef_prior: None,
        min_periods: 3.0,
    }
}

#[test]
fn rls() {
    let cfg = rls_cfg();
    let r = probe_with(Rls::new(cfg).unwrap(), 2, Some(&Rls::n_eff));
    check(&r, "rls", 2, 1);
}

fn lasso_cfg() -> LassoCfg {
    LassoCfg {
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
        window: None,
        window_every: None,
        max_cd_iters: 100,
        cd_tol: 1e-10,
        target_gaps: online_core::TargetGaps::OwnRows,
    }
}

#[test]
fn lasso() {
    let cfg = lasso_cfg();
    let r = probe_with(Lasso::new(cfg).unwrap(), 1, Some(&Lasso::n_eff));
    check(&r, "lasso", 1, 2);
}

fn kalman_cfg() -> KalmanCfg {
    KalmanCfg {
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
        revert_halflife: vec![f64::INFINITY],
        standardize: true,
    }
}

#[test]
fn kalman() {
    let cfg = kalman_cfg();
    let r = probe_with(Kalman::new(cfg).unwrap(), 2, Some(&Kalman::n_eff));
    check(&r, "kalman", 2, 1);
}

/// E41: a reverting filter is held to the same contract as the random walk.
fn kalman_revert_cfg() -> KalmanCfg {
    KalmanCfg {
        revert_halflife: vec![f64::INFINITY, 25.0, 6.0],
        ..kalman_cfg()
    }
}

#[test]
fn kalman_reverting() {
    let r = probe_with(
        Kalman::new(kalman_revert_cfg()).unwrap(),
        2,
        Some(&Kalman::n_eff),
    );
    check(&r, "kalman", 2, 1);
}

fn robust_cfg(loss: RobustLoss) -> RobustCfg {
    RobustCfg {
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
    }
}

const ROBUST_LOSSES: [RobustLoss; 2] = [
    RobustLoss::Huber { delta: 1.5 },
    RobustLoss::Quantile { tau: 0.5 },
];

#[test]
fn robust() {
    for loss in ROBUST_LOSSES {
        let cfg = robust_cfg(loss);
        let r = probe_with(Robust::new(cfg).unwrap(), 2, Some(&Robust::n_eff));
        check(&r, "robust", 2, 1);
    }
}

fn ftrl_cfg() -> FtrlCfg {
    FtrlCfg {
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
    }
}

#[test]
fn ftrl() {
    let cfg = ftrl_cfg();
    let r = probe_with(Ftrl::new(cfg).unwrap(), 2, Some(&Ftrl::n_eff));
    check(&r, "ftrl", 2, 1);
}

fn sgd_cfg() -> SgdCfg {
    SgdCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        loss: SgdLoss::Squared,
        learning_rate: 0.01,
        schedule: LearningRate::Constant,
        l2: 0.0,
        clip_gradient: 1e3,
        constraint: None,
        scale_features: false,
        min_periods: 3.0,
    }
}

#[test]
fn sgd() {
    let cfg = sgd_cfg();
    let r = probe_with(Sgd::new(cfg).unwrap(), 2, Some(&Sgd::n_eff));
    check(&r, "sgd", 2, 1);
}

fn pa_cfg() -> PaCfg {
    PaCfg {
        n_features: K,
        n_targets: 2,
        add_intercept: true,
        decay: decay(),
        mode: PaMode::Pa1,
        c: 1.0,
        eps: 0.1,
        min_periods: 3.0,
        constraint: None,
    }
}

#[test]
fn pa() {
    let cfg = pa_cfg();
    let r = probe_with(Pa::new(cfg).unwrap(), 2, Some(&Pa::n_eff));
    check(&r, "pa", 2, 1);
}

fn holt_cfg() -> HoltCfg {
    HoltCfg {
        n_targets: 2,
        level_halflife: HALFLIFE,
        trend_halflife: 4.0 * HALFLIFE,
        min_periods: 3.0,
    }
}

#[test]
fn holt() {
    let cfg = holt_cfg();
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

fn ew_cov_model_cfg() -> EwCovCfg {
    EwCovCfg {
        n_features: K,
        decay: decay(),
        stats: vec![
            EwCovStat::Mean,
            EwCovStat::Var,
            EwCovStat::Corr,
            EwCovStat::LagCorr,
        ],
        min_periods: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0,
        // The probe carries lags so that the save/restore and `clear_lags`
        // arms above see a model with a ring in them (L1).
        lags: vec![1, 3],
        window: None,
        window_every: None,
    }
}

#[test]
fn ew_cov_model() {
    let cfg = ew_cov_model_cfg();
    // No targets: it emits statistics, one slot per (stat x column-or-pair).
    let m = EwCovModel::new(cfg).unwrap();
    assert_eq!(m.n_targets(), 0, "ew_cov has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(
        m.n_outputs(),
        2 + 2 + 1 + 2 * K * K,
        "mean and var per column, one pair, and a k x k block per lag"
    );
    let r = probe_with(m, 0, Some(&EwCovModel::n_eff));
    assert_eq!(r.kind, "ew_cov");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!(r.roundtrips);
}

fn kmeans_cfg() -> KMeansCfg {
    KMeansCfg {
        n_features: K,
        k: 3,
        decay: decay(),
        min_periods: 3.0,
        warm_rows: 10,
        seed_rule: SeedRule::Lloyd,
        seed: 0,
        update_every: 1,
        split_merge: 0.5,
        split_merge_every: 50,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.1,
    }
}

#[test]
fn kmeans() {
    let cfg = kmeans_cfg();
    // No targets: it emits an assignment and two distances.
    let m = KMeans::new(cfg).unwrap();
    assert_eq!(m.n_targets(), 0, "kmeans has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 3, "cluster, dist, dist2");
    let r = probe_with(m, 0, Some(&KMeans::n_eff));
    assert_eq!(r.kind, "kmeans");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

fn micro_cfg() -> MicroCfg {
    // eps and beta_mu sized for a 20-halflife window over the harness's
    // uniform rows: a summary must reach beta_mu before it decays back.
    MicroCfg {
        n_features: K,
        decay: decay(),
        min_periods: 3.0,
        eps: 0.6,
        beta_mu: 2.0,
        max_clusters: 50,
        prune_every: 10,
        macro_link: None,
        standardize: true,
        scale_floor: 0.1,
    }
}

#[test]
fn micro() {
    let cfg = micro_cfg();
    // No targets: a label, a distance, a micro-cluster id, an outlier flag
    // and two counts.
    let m = Micro::new(cfg).unwrap();
    assert_eq!(m.n_targets(), 0, "micro has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(
        m.n_outputs(),
        6,
        "cluster, dist, micro, outlier, n_clusters, n_micro"
    );
    let r = probe_with(m, 0, Some(&Micro::n_eff));
    assert_eq!(r.kind, "micro");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

#[test]
fn every_state_kind_is_distinct_and_named() {
    // `ModelState::kind` names the model in every state error; a mutation that
    // returns a constant would make "expected X, found Y" meaningless.
    let kinds = [
        "ew_ridge",
        "rls",
        "lasso",
        "kalman",
        "robust",
        "ftrl",
        "sgd",
        "pa",
        "holt",
        "ew_cov",
        "kmeans",
        "micro",
        "ew_class",
        "seqtest",
        "marginal",
        "deco",
        "rcov",
        "hmm",
        "corrchange",
        "bocpd",
    ];
    let mut seen = std::collections::HashSet::new();
    for k in kinds {
        assert!(!k.is_empty());
        assert!(seen.insert(k), "duplicate kind {k}");
    }
    // And the names really come from the states, not this list: every one
    // read off a real model's state, where four of them were (review
    // 2026-09-18, T2; docs/PLAN.md task 112). `robust` covers both of its
    // losses, and the two `kalman` configurations are one kind.
    let from_states = [
        EwRidge::new(ew_ridge_cfg()).unwrap().state().model.kind(),
        Rls::new(rls_cfg()).unwrap().state().model.kind(),
        Lasso::new(lasso_cfg()).unwrap().state().model.kind(),
        Kalman::new(kalman_cfg()).unwrap().state().model.kind(),
        Robust::new(robust_cfg(ROBUST_LOSSES[0]))
            .unwrap()
            .state()
            .model
            .kind(),
        Ftrl::new(ftrl_cfg()).unwrap().state().model.kind(),
        Sgd::new(sgd_cfg()).unwrap().state().model.kind(),
        Pa::new(pa_cfg()).unwrap().state().model.kind(),
        Holt::new(holt_cfg()).unwrap().state().model.kind(),
        EwCovModel::new(ew_cov_model_cfg())
            .unwrap()
            .state()
            .model
            .kind(),
        KMeans::new(kmeans_cfg()).unwrap().state().model.kind(),
        Micro::new(micro_cfg()).unwrap().state().model.kind(),
        EwClass::new(ew_class_cfg()).unwrap().state().model.kind(),
        SeqTest::new(seqtest_cfg()).unwrap().state().model.kind(),
        Marginal::new(marginal_cfg()).unwrap().state().model.kind(),
        Deco::new(deco_cfg()).unwrap().state().model.kind(),
        Rcov::new(rcov_cfg()).unwrap().state().model.kind(),
        Hmm::new(hmm_cfg()).unwrap().state().model.kind(),
        CorrChange::new(corrchange_cfg())
            .unwrap()
            .state()
            .model
            .kind(),
        Bocpd::new(bocpd_cfg()).unwrap().state().model.kind(),
    ];
    assert_eq!(from_states, kinds);
    assert_eq!(
        Robust::new(robust_cfg(ROBUST_LOSSES[1]))
            .unwrap()
            .state()
            .model
            .kind(),
        "robust"
    );
    assert_eq!(
        Kalman::new(kalman_revert_cfg())
            .unwrap()
            .state()
            .model
            .kind(),
        "kalman"
    );
    // The accumulator an `ew_cov` state nests has a kind of its own.
    assert_eq!(
        State::new(ModelState::EwCov(Box::new(EwCov::new(1))))
            .model
            .kind(),
        "ew_cov_accumulator"
    );
}

fn ew_class_cfg() -> EwClassCfg {
    EwClassCfg {
        n_features: K,
        n_classes: 2,
        decay: decay(),
        min_periods: 3.0,
        covariance: Covariance::Full,
        precision_prior: 0.1,
        window: None,
        window_every: None,
    }
}

#[test]
fn ew_class() {
    for covariance in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
        let cfg = EwClassCfg {
            covariance,
            ..ew_class_cfg()
        };
        // The label is learned from, not predicted as a number: no targets,
        // and the outputs are the class and one posterior per class.
        let m = EwClass::new(cfg).unwrap();
        assert_eq!(m.n_targets(), 0, "ew_class has no targets");
        assert_eq!(m.n_features(), K);
        assert_eq!(m.n_outputs(), 3, "class, p_0, p_1");
        // Probed without labels: n_eff counts every accepted row, so
        // `min_periods` means the same number of rows as everywhere else.
        let r = probe_with(m, 0, Some(&EwClass::n_eff));
        assert_eq!(r.kind, "ew_class");
        assert_eq!(r.pred_len, r.n_outputs);
        assert_eq!(r.n_eff[0], 0.0);
        assert_eq!(r.n_eff[1], 1.0);
        assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
        assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
        assert!(r.roundtrips);
    }
}

fn seqtest_cfg() -> SeqTestCfg {
    SeqTestCfg {
        n_targets: 2,
        min_periods: 3.0,
    }
}

#[test]
fn seqtest() {
    // A test of the sign of each target: no features, four slots per target
    // (the two log e-values and the two counts), and no decay at all -- an
    // e-process that forgot would not be one -- so `n_eff` is the plain
    // weight sum and a clock gap changes nothing.
    let m = SeqTest::new(seqtest_cfg()).unwrap();
    assert_eq!(m.n_targets(), 2);
    assert_eq!(m.n_features(), 0, "seqtest reads no features");
    assert_eq!(m.n_outputs(), 2 * SEQTEST_SLOTS);
    let r = probe_with(m, 2, Some(&SeqTest::n_eff));
    assert_eq!(r.kind, "seqtest");
    assert_eq!(r.pred_len, 2 * SEQTEST_SLOTS);
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 40.0]);
    assert_eq!(r.after_gap, r.before_gap + 1.0, "no decay over a gap");
    assert!(r.roundtrips);
}

fn marginal_cfg() -> MarginalCfg {
    MarginalCfg {
        n_features: K,
        n_targets: 2,
        decay: decay(),
        min_periods: vec![3.0; 2],
        // A lag, so the contract exercises marginal's `clear_lags` arm (it
        // keeps a lag ring; review 2026-09-18, T2) -- which is why marginal is
        // in KEEPS_LAGS above.
        lags: vec![1],
        serial_rule: None,
        cross_lags: None,
        bins: None,
        window: None,
        window_every: None,
    }
}

#[test]
fn marginal() {
    // Per-pair moments read from the state, nothing per row: targets and
    // features both counted, zero output slots, and `n_eff` -- every learned
    // row's weight -- on the same recursion as every other model.
    let m = Marginal::new(marginal_cfg()).unwrap();
    assert_eq!(m.n_targets(), 2);
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 0, "marginal predicts nothing per row");
    let r = probe_with(m, 2, Some(&Marginal::n_eff));
    check(&r, "marginal", 2, 0);
    assert_eq!(r.pred_len, 0);
}

fn deco_cfg() -> DecoCfg {
    DecoCfg {
        n_features: K,
        decay: decay(),
        dynamics: DecoDynamics::Ew,
        alpha: None,
        beta: None,
        blocks: Vec::new(),
        min_periods: 3.0,
    }
}

#[test]
fn deco() {
    // One number for the whole correlation matrix: no targets, three slots
    // (`u`, `rho`, `loglik`), and `n_eff` -- the standardiser's accumulated
    // weight -- on the same recursion as every other model.
    let m = Deco::new(deco_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0, "deco has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 3, "u, rho, loglik");
    let r = probe_with(m, 0, Some(&Deco::n_eff));
    assert_eq!(r.kind, "deco");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

fn rcov_cfg() -> RcovCfg {
    RcovCfg {
        n_features: K,
        kind: RcovKind::Kernel,
        kernel: "parzen".into(),
        bandwidth: Some(3),
        jitter: 2,
        theta: 1.0,
        psd: false,
        block_rows: Some(100),
        max_bandwidth: None,
        preavg_rows: None,
        noise_stride: 1,
        iv_stride: 20,
    }
}

#[test]
fn rcov() {
    // A block estimator: no targets, no output slots, and no decay -- the
    // block is the value, read at the group's close. `n_eff` is the plain
    // count of returns, so a clock gap changes nothing.
    let m = Rcov::new(rcov_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0);
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 0, "rcov reports nothing per row");
    let r = probe_with(m, 0, Some(&Rcov::n_eff));
    assert_eq!(r.kind, "rcov");
    assert_eq!(r.pred_len, 0);
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 40.0]);
    assert_eq!(r.after_gap, r.before_gap + 1.0, "no decay over a gap");
    assert!(r.roundtrips);
}

fn hmm_cfg() -> HmmCfg {
    HmmCfg {
        n_features: K,
        k: 2,
        decay: decay(),
        covariance: Covariance::Full,
        precision_prior: 1e-2,
        min_periods: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: Some(vec![-1.0, -1.0, 1.0, 1.0]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 1,
        tvtp: None,
    }
}

#[test]
fn hmm() {
    // A hidden state, not a label: no targets, `2K + 2` slots (the filtered
    // and predicted posteriors, the state and the row's log-likelihood),
    // and `n_eff` on the shared recursion -- the responsibilities sum to
    // the row's weight, so it is unchanged.
    let m = Hmm::new(hmm_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0, "hmm has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 6, "p_0, p_1, p1_0, p1_1, state, loglik");
    let r = probe_with(m, 0, Some(&Hmm::n_eff));
    assert_eq!(r.kind, "hmm");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

fn corrchange_cfg() -> CorrChangeCfg {
    CorrChangeCfg {
        n_features: K,
        kind: CorrChangeKind::Monitor,
        span_rows: 20,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: decay(),
        crit: None,
        n_perm: 20,
        permute_every: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
    }
}

#[test]
fn corrchange() {
    // A test, not a model of the data: no targets, four slots (the
    // statistic, its critical value, the flag and the rows since the last
    // one), and `n_eff` on the shared recursion.
    let m = CorrChange::new(corrchange_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0);
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 4, "stat, crit, flag, since_flag");
    let r = probe_with(m, 0, Some(&CorrChange::n_eff));
    assert_eq!(r.kind, "corrchange");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff[0], 0.0);
    assert_eq!(r.n_eff[1], 1.0);
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

fn bocpd_cfg() -> BocpdCfg {
    BocpdCfg {
        n_features: K,
        hazard: 50.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: None,
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-6,
        max_run: 200,
        min_periods: 0.0,
    }
}

#[test]
fn bocpd() {
    // A posterior over run lengths, not a fit: no targets, `4 + d` slots,
    // and `n_eff` the plain weight sum -- the recursion does not decay, so
    // a clock gap changes nothing.
    let m = Bocpd::new(bocpd_cfg()).unwrap();
    assert_eq!(m.n_targets(), 0);
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 4 + K);
    let r = probe_with(m, 0, Some(&Bocpd::n_eff));
    assert_eq!(r.kind, "bocpd");
    assert_eq!(r.pred_len, r.n_outputs);
    assert_eq!(r.n_eff, vec![0.0, 1.0, 2.0, 40.0]);
    assert_eq!(r.after_gap, r.before_gap + 1.0, "no decay over a gap");
    assert!(r.roundtrips);
}

/// The variants of `ModelState` this file probes. A model added to the enum
/// and not to this list fails here, which is the reminder to write its
/// `*_cfg()` and probe above (docs/EXTENDING.md).
const PROBED: &[&str] = &[
    "EwCov",
    "EwRidge",
    "Rls",
    "Lasso",
    "Kalman",
    "Robust",
    "Ftrl",
    "EwCovModel",
    "Sgd",
    "Pa",
    "Holt",
    "KMeans",
    "Micro",
    "EwClass",
    "SeqTest",
    "Marginal",
    "Deco",
    "Rcov",
    "Hmm",
    "CorrChange",
    "Bocpd",
];

#[test]
fn every_model_state_variant_is_probed_here() {
    // serde's unknown-variant error names every variant the enum has -- the
    // one place that list exists outside the enum itself.
    let err = serde_json::from_str::<ModelState>(r#"{"Nope": null}"#)
        .unwrap_err()
        .to_string();
    let quoted: Vec<&str> = err.split('`').skip(1).step_by(2).collect();
    assert_eq!(quoted[0], "Nope", "{err}");
    assert_eq!(
        &quoted[1..],
        PROBED,
        "a ModelState variant has no contract probe"
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

// ---------------------------------------------------------------------------
// Bounded input (docs/IMPROVEMENTS.md C2, T4)
// ---------------------------------------------------------------------------

/// The largest magnitude the stream layer lets through for a feature, target
/// or weight; beyond it a value is treated as missing.
const BOUND: f64 = online_core::INPUT_BOUND;

struct Row {
    x: Vec<f64>,
    y: Vec<Option<f64>>,
    w: f64,
    /// Rows the clean twin never sees.
    extreme: bool,
}

/// One row of the well-behaved process, at `scale`.
fn nice(s: &mut u64, targets: usize, scale: f64) -> Row {
    let x: Vec<f64> = (0..K).map(|_| lcg(s) * scale).collect();
    let y = (0..targets)
        .map(|j| Some(0.5 * (j as f64 + 1.0) * scale + x[0] - 0.5 * x[1]))
        .collect();
    Row {
        x,
        y,
        w: 1.0,
        extreme: false,
    }
}

fn extreme(rows: &mut Vec<Row>, s: &mut u64, targets: usize, edit: impl FnOnce(&mut Row)) {
    let mut r = nice(s, targets, 1.0);
    edit(&mut r);
    r.extreme = true;
    rows.push(r);
}

/// A warm start, then the bound in every position and sign, a run at a tiny
/// scale followed by the bound again (a standardized regressor is then
/// `(1e100 - mean) / 1e-100`), and a long well-behaved tail.
fn bounded_script(targets: usize) -> Vec<Row> {
    let mut s = 20260901u64;
    let mut rows = Vec::new();
    // At the bound before anything else: the first observation is the `a = 0`
    // branch of every accumulator (CLAUDE.md rule 9).
    extreme(&mut rows, &mut s, targets, |r| r.x[0] = BOUND);
    extreme(&mut rows, &mut s, targets, |r| {
        r.y.iter_mut().for_each(|y| *y = Some(BOUND))
    });
    for _ in 0..300 {
        rows.push(nice(&mut s, targets, 1.0));
    }
    for sign in [1.0, -1.0] {
        let b = sign * BOUND;
        extreme(&mut rows, &mut s, targets, |r| r.x[0] = b);
        extreme(&mut rows, &mut s, targets, |r| {
            r.x.iter_mut().for_each(|x| *x = b)
        });
        extreme(&mut rows, &mut s, targets, |r| {
            r.y.iter_mut().for_each(|y| *y = Some(b))
        });
        extreme(&mut rows, &mut s, targets, |r| r.w = BOUND);
        extreme(&mut rows, &mut s, targets, |r| {
            r.x[0] = b;
            r.w = BOUND;
        });
        extreme(&mut rows, &mut s, targets, |r| {
            r.y.iter_mut().for_each(|y| *y = Some(b));
            r.w = BOUND;
        });
        extreme(&mut rows, &mut s, targets, |r| {
            r.x.iter_mut().for_each(|x| *x = b);
            r.y.iter_mut().for_each(|y| *y = Some(b));
            r.w = BOUND;
        });
        extreme(&mut rows, &mut s, targets, |r| r.w = 1.0 / BOUND);
    }
    for _ in 0..200 {
        let mut r = nice(&mut s, targets, 1.0 / BOUND);
        r.extreme = true;
        rows.push(r);
    }
    extreme(&mut rows, &mut s, targets, |r| r.x[0] = BOUND);
    extreme(&mut rows, &mut s, targets, |r| {
        r.y.iter_mut().for_each(|y| *y = Some(BOUND))
    });
    // 1500 halflives. A row at the bound with weight at the bound leaves a
    // moment of 1e300 on the sum scale, which needs 1000 halflives to fall
    // below 1e-6; the rest is margin (measured: ew_ridge agrees with its twin
    // to 3e-5 after 1000 halflives and to rounding after 1100).
    for _ in 0..30_000 {
        rows.push(nice(&mut s, targets, 1.0));
    }
    rows
}

/// What "recovered" means for a model over the tail of the script; both are
/// relative errors, `|a - b| / (1 + |b|)`.
#[derive(Clone, Copy)]
enum Recovery {
    /// Every prediction agrees with the clean twin's to this tolerance. The
    /// right criterion for a model that converges to one answer on clean data.
    Twin(f64),
    /// Every prediction of the model *and* of its twin is within this distance
    /// of the target. For models that stop learning inside a tolerance band
    /// (`pa` inside its epsilon tube, the quantile loss at its residual floor),
    /// two copies with different histories legitimately settle at different
    /// points of the band, so agreement with the twin is not a property they
    /// have; being as accurate as the twin is.
    Tube(f64),
    /// Every output is finite, and the mean squared distance to the assigned
    /// centre (slot 1, squared) over the tail is within this relative
    /// tolerance of the twin's. For a clustering model: two histories settle
    /// on two labelings of the same data, so neither the label nor the
    /// distance of one row is a property two copies share, but the fit is.
    Fit(f64),
}

/// The stream layer accepts any finite value with `|v| <= BOUND`, so a model
/// must keep a finite state -- and go on learning -- after any such row. Two
/// copies of the model: one sees the whole script, the twin only its
/// well-behaved rows. Over the last thousand rows every prediction of the
/// first must be finite and satisfy `how`: the extreme rows perturbed the
/// model as its equations say they should, and then washed out, rather than
/// leaving an `inf` or NaN that never decays.
fn recovers_from_bounded_extremes<M: OnlineModel>(
    build: impl Fn() -> M,
    targets: usize,
    how: Recovery,
) {
    recovers_over(&bounded_script(targets), build, how);
}

/// The same, over a script the caller has prepared (a classifier wants the
/// targets turned into labels).
fn recovers_over<M: OnlineModel>(rows: &[Row], build: impl Fn() -> M, how: Recovery) {
    let mut model = build();
    let mut twin = build();
    let kind = model.state().model.kind();
    let n = rows.len();
    let mut seen = [false, false];
    let mut worst = 0.0f64;
    let mut fit = [0.0f64, 0.0];
    for (i, r) in rows.iter().enumerate() {
        let d = if seen[0] { 1.0 } else { 0.0 };
        seen[0] = true;
        let a = model.step(&r.x, &r.y, d, r.w);
        assert!(
            a.n_eff.is_finite(),
            "{kind}: n_eff is {} at row {i}",
            a.n_eff
        );
        if r.extreme {
            continue;
        }
        let d = if seen[1] { 1.0 } else { 0.0 };
        seen[1] = true;
        let b = twin.step(&r.x, &r.y, d, r.w);
        if i + 1000 < n {
            continue;
        }
        for (slot, (pa, pb)) in a.pred.iter().zip(&b.pred).enumerate() {
            assert!(
                pa.is_finite(),
                "{kind}: slot {slot} predicts {pa} at row {i} (twin: {pb})"
            );
            let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
            match how {
                Recovery::Twin(tol) => {
                    let err = rel(*pa, *pb);
                    assert!(
                        err <= tol,
                        "{kind}: slot {slot} at row {i}: {pa} vs the twin's {pb} (tol {tol})"
                    );
                    worst = worst.max(err);
                }
                Recovery::Tube(tol) => {
                    let y = r.y[slot].unwrap();
                    for (who, p) in [("model", *pa), ("twin", *pb)] {
                        let err = rel(p, y);
                        assert!(
                            err <= tol,
                            "{kind}: slot {slot} at row {i}: the {who} predicts {p} for {y} (tol {tol})"
                        );
                        worst = worst.max(err);
                    }
                }
                Recovery::Fit(_) => {
                    assert!(
                        pb.is_finite(),
                        "{kind}: the twin's slot {slot} is {pb} at row {i}"
                    );
                    if slot == 1 {
                        fit[0] += pa * pa;
                        fit[1] += pb * pb;
                    }
                }
            }
        }
    }
    match how {
        Recovery::Twin(_) => {
            eprintln!("{kind}: worst relative disagreement with the twin {worst:.2e}")
        }
        Recovery::Tube(_) => eprintln!("{kind}: worst relative error of model or twin {worst:.2e}"),
        Recovery::Fit(tol) => {
            let err = (fit[0] - fit[1]).abs() / fit[1];
            eprintln!(
                "{kind}: tail mean squared distance {} vs the twin's {} ({err:.2e})",
                fit[0] / 1000.0,
                fit[1] / 1000.0
            );
            assert!(
                err <= tol,
                "{kind}: tail fit {} vs the twin's {} (tol {tol})",
                fit[0],
                fit[1]
            );
        }
    }
}

#[test]
fn ew_ridge_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || EwRidge::new(ew_ridge_cfg()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
    let mut cfg = ew_ridge_cfg();
    cfg.standardize = true;
    recovers_from_bounded_extremes(
        move || EwRidge::new(cfg.clone()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
}

#[test]
fn rls_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Rls::new(rls_cfg()).unwrap(), 2, Recovery::Twin(1e-9));
}

#[test]
fn lasso_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Lasso::new(lasso_cfg()).unwrap(), 1, Recovery::Twin(1e-9));
}

#[test]
fn kalman_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || Kalman::new(kalman_cfg()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
    let mut cfg = kalman_cfg();
    cfg.standardize = false;
    recovers_from_bounded_extremes(
        move || Kalman::new(cfg.clone()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
    recovers_from_bounded_extremes(
        || Kalman::new(kalman_revert_cfg()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
}

#[test]
fn robust_recovers_from_bounded_extremes() {
    for loss in ROBUST_LOSSES {
        // Huber is least squares near the solution and converges; the
        // quantile loss reweights by `s / |r|`, which never settles closer than
        // its residual floor, so two histories agree only to about that.
        let how = match loss {
            RobustLoss::Huber { .. } => Recovery::Twin(1e-9),
            RobustLoss::Quantile { .. } => Recovery::Tube(1e-3),
        };
        for standardize in [false, true] {
            let mut cfg = robust_cfg(loss);
            cfg.standardize = standardize;
            recovers_from_bounded_extremes(move || Robust::new(cfg.clone()).unwrap(), 2, how);
        }
    }
}

#[test]
fn ftrl_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(|| Ftrl::new(ftrl_cfg()).unwrap(), 2, Recovery::Twin(1e-9));
    let mut cfg = ftrl_cfg();
    cfg.loss = FtrlLoss::Logistic;
    recovers_from_bounded_extremes(
        move || Ftrl::new(cfg.clone()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
}

#[test]
fn sgd_recovers_from_bounded_extremes() {
    for scale_features in [false, true] {
        let mut cfg = sgd_cfg();
        cfg.scale_features = scale_features;
        recovers_from_bounded_extremes(
            move || Sgd::new(cfg.clone()).unwrap(),
            2,
            Recovery::Twin(1e-9),
        );
    }
}

#[test]
fn pa_recovers_from_bounded_extremes() {
    // PA-I with `epsilon = 0.1` stops updating inside its tube, so it is
    // accurate to the tube, not to the twin.
    recovers_from_bounded_extremes(|| Pa::new(pa_cfg()).unwrap(), 2, Recovery::Tube(1e-1));
}

#[test]
fn holt_recovers_from_bounded_extremes() {
    // The tail is 1500 halflives of HALFLIFE: what a row at the bound with
    // weight at the bound needs to wash out of a mean-form accumulator. Since
    // the code review's S29 `holt`'s level and trend are weighted means, so a
    // row's weight counts, and the trend forgets such a row on its own
    // halflife -- four times the level's by default, which would want a tail
    // four times as long. So the probe runs both at HALFLIFE. (The textbook
    // form it replaced ignored the weight, and recovered on the level's rate.)
    let cfg = HoltCfg {
        trend_halflife: HALFLIFE,
        ..holt_cfg()
    };
    recovers_from_bounded_extremes(|| Holt::new(cfg.clone()).unwrap(), 2, Recovery::Twin(1e-9));
}

#[test]
fn seqtest_is_indifferent_to_bounded_extremes() {
    // The e-process reads only the sign of each target, so a row at the
    // bound is one more trial like any other: no accumulator holds the
    // magnitude, and the state after the script equals a twin's that was
    // fed the signs alone. (The twin criteria above do not apply: a test
    // never forgets, so a copy that saw more rows is a different test.)
    let rows = bounded_script(2);
    let mut model = SeqTest::new(seqtest_cfg()).unwrap();
    let mut twin = SeqTest::new(seqtest_cfg()).unwrap();
    for (i, r) in rows.iter().enumerate() {
        let d = if i == 0 { 0.0 } else { 1.0 };
        let a = model.step(&r.x, &r.y, d, r.w);
        let signs: Vec<Option<f64>> = r.y.iter().map(|y| y.map(f64::signum)).collect();
        let b = twin.step(&[], &signs, d, r.w);
        assert!(a.n_eff.is_finite(), "n_eff is {} at row {i}", a.n_eff);
        assert!(
            a.pred.iter().all(|p| p.is_finite() || a.n_eff < 3.0),
            "row {i}: {:?}",
            a.pred
        );
        same_step("seqtest", i, &a, &b);
    }
    assert_eq!(model, twin);
    // A weight at the bound counts itself, once; the rows after it are
    // lost in its rounding but the counts and the wealth go on moving.
    assert!(model.n_eff() >= BOUND);
    assert!(model.n_pos()[0] + model.n_neg()[0] > 30_000.0);
}

#[test]
fn kmeans_recovers_from_bounded_extremes() {
    // The extreme rows drag a centre to the bound and blow the metric up;
    // the split–merge check re-places the emptied centre and the moments
    // decay back, so the tail is fitted as well as the twin fits it
    // (measured: the two agree to 1e-14; the tolerance is the margin).
    for standardize in [true, false] {
        for rule in [SeedRule::First, SeedRule::Lloyd] {
            let cfg = KMeansCfg {
                standardize,
                seed_rule: rule,
                ..kmeans_cfg()
            };
            recovers_from_bounded_extremes(
                move || KMeans::new(cfg.clone()).unwrap(),
                0,
                Recovery::Fit(1e-6),
            );
        }
    }
}

#[test]
fn ew_class_recovers_from_bounded_extremes() {
    // The script's target becomes the label: class 1 where y > 0. A row at
    // the bound with weight at the bound leaves one class's mean at 1e100
    // and its co-moments at 1e200; both decay back through the mean-form
    // update as the weight does, and the ridge scale, once driven to zero
    // by that row, never matters again (measured: the two states are
    // bitwise equal over the tail; the tolerance is the margin).
    let mut rows = bounded_script(1);
    for r in &mut rows {
        r.y = vec![Some(f64::from(r.y[0].unwrap() > 0.0))];
    }
    for covariance in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
        let cfg = EwClassCfg {
            covariance,
            ..ew_class_cfg()
        };
        recovers_over(
            &rows,
            move || EwClass::new(cfg.clone()).unwrap(),
            Recovery::Twin(1e-9),
        );
    }
}

#[test]
fn micro_recovers_from_bounded_extremes() {
    // A row of weight 1e100 makes a summary nothing moves until it has
    // decayed below `beta_mu` (332 halflives) and is pruned; a row at the
    // bound opens a summary there that the xi rule prunes at the next
    // checkpoint. By the tail both histories tile the unit square afresh,
    // and two tilings agree on the mean squared distance to the nearest
    // potential summary only loosely (measured: to about a tenth).
    for standardize in [true, false] {
        for macro_link in [None, Some(0.0)] {
            let cfg = MicroCfg {
                standardize,
                macro_link,
                ..micro_cfg()
            };
            recovers_from_bounded_extremes(
                move || Micro::new(cfg.clone()).unwrap(),
                0,
                Recovery::Fit(0.5),
            );
        }
    }
}

#[test]
fn deco_recovers_from_bounded_extremes() {
    // A 1e100 row makes every standardised value huge, so `u` is at its
    // bounds for a while and `loglik` is very negative; the state must stay
    // finite and the outputs must return to a clean twin's once the row has
    // decayed away.
    recovers_from_bounded_extremes(|| Deco::new(deco_cfg()).unwrap(), 0, Recovery::Twin(1e-9));
}

#[test]
fn ew_cov_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || EwCovModel::new(ew_cov_model_cfg()).unwrap(),
        0,
        Recovery::Twin(1e-9),
    );
    let mut cfg = ew_cov_model_cfg();
    cfg.stats.push(EwCovStat::PartialCorr);
    cfg.precision_prior = Some(1e-4);
    recovers_from_bounded_extremes(
        move || EwCovModel::new(cfg.clone()).unwrap(),
        0,
        Recovery::Twin(1e-9),
    );
}

#[test]
fn marginal_recovers_from_bounded_extremes() {
    // `marginal` predicts nothing, so `recovers_over` would have nothing to
    // compare: read the pairs from the state instead. Over the last thousand
    // rows every statistic of every pair must be finite and agree with the
    // twin's to rounding -- the extreme rows moved the moments as the
    // equations say and then washed out.
    let rows = bounded_script(2);
    let mut model = Marginal::new(marginal_cfg()).unwrap();
    let mut twin = Marginal::new(marginal_cfg()).unwrap();
    let n = rows.len();
    let mut seen = [false, false];
    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + b.abs());
    for (i, r) in rows.iter().enumerate() {
        let d = if seen[0] { 1.0 } else { 0.0 };
        seen[0] = true;
        let a = model.step(&r.x, &r.y, d, r.w);
        assert!(
            a.n_eff.is_finite(),
            "marginal: n_eff is {} at row {i}",
            a.n_eff
        );
        assert!(a.pred.is_empty());
        if r.extreme {
            continue;
        }
        let d = if seen[1] { 1.0 } else { 0.0 };
        seen[1] = true;
        twin.step(&r.x, &r.y, d, r.w);
        if i + 1000 < n {
            continue;
        }
        for t in 0..2 {
            for j in 0..K {
                let (pa, pb) = (model.pair(t, j), twin.pair(t, j));
                for (what, va, vb) in [
                    ("n_eff", pa.n_eff, pb.n_eff),
                    ("n_kish", pa.n_kish, pb.n_kish),
                    ("mean_x", pa.mean_x, pb.mean_x),
                    ("var_x", pa.var_x, pb.var_x),
                    ("mean_y", pa.mean_y, pb.mean_y),
                    ("var_y", pa.var_y, pb.var_y),
                    ("cov", pa.cov, pb.cov),
                    ("corr", pa.corr, pb.corr),
                    ("beta", pa.beta, pb.beta),
                    ("t", pa.t, pb.t),
                ] {
                    assert!(
                        va.is_finite(),
                        "marginal: {what}[{t},{j}] is {va} at row {i} (twin: {vb})"
                    );
                    assert!(
                        rel(va, vb) <= 1e-9,
                        "marginal: {what}[{t},{j}] at row {i}: {va} vs the twin's {vb}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `predict` is the step without the step (docs/ENHANCEMENTS.md E31).
// ---------------------------------------------------------------------------

/// Equal slot by slot, with NaN equal to NaN: the "not ready" marker has to
/// agree too.
fn same_step(kind: &str, i: usize, p: &Step, s: &Step) {
    assert_eq!(p.pred.len(), s.pred.len(), "{kind}: slot count at row {i}");
    for (slot, (a, b)) in p.pred.iter().zip(&s.pred).enumerate() {
        assert!(
            a == b || (a.is_nan() && b.is_nan()),
            "{kind}: slot {slot} at row {i}: predict said {a}, step said {b}"
        );
    }
    assert!(
        p.n_eff == s.n_eff,
        "{kind}: n_eff at row {i}: predict said {}, step said {}",
        p.n_eff,
        s.n_eff
    );
    assert_eq!(p.extra, s.extra, "{kind}: extra at row {i}");
}

/// Over a stream with missing targets, zero-weight rows, uneven weights and
/// clock gaps, `predict(x, d)` called before each `step(x, y, d, w)` must
/// return exactly what the step returns -- the same numbers, not close ones --
/// and, being `&self`, it cannot have moved the state. That equality is the
/// whole definition of `predict`; a model that computes its prediction any
/// other way in one of the two places fails here.
fn predict_is_the_step_without_the_step<M: OnlineModel>(
    build: impl Fn() -> M,
    targets: usize,
    binary: bool,
) {
    let mut m = build();
    let kind = m.state().model.kind();
    let mut s = 20260902u64;
    let mut ready = 0usize;
    for i in 0..400 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                if lcg(&mut s) > 0.8 {
                    return None;
                }
                let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + 0.1 * lcg(&mut s);
                Some(if binary { f64::from(lin > 0.5) } else { lin })
            })
            .collect();
        let u = lcg(&mut s);
        let w = if u < -0.9 {
            0.0
        } else if u < -0.5 {
            0.5
        } else if u > 0.7 {
            2.0
        } else {
            1.0
        };
        let d = match (i, lcg(&mut s)) {
            (0, _) => 0.0,
            (_, v) if v > 0.95 => 25.0,
            (_, v) if v > 0.8 => 3.0,
            _ => 1.0,
        };
        // `predict_with`, not `predict`: two models read a number out of
        // the targets slot, and the parity that matters is against the
        // answer the step gives for the same row (C1).
        let p = m.predict_with(&x, &y, d);
        let step = m.step(&x, &y, d, w);
        same_step(kind, i, &p, &step);
        if step.pred.iter().all(|v| v.is_finite()) {
            ready += 1;
        }
    }
    // The stream must actually have exercised the ready path, or the test
    // would pass on NaN == NaN alone. A model that reports only where a
    // span closes has far fewer such rows, and that is the point of it:
    // `SPARSE_OUTPUT` says how many are enough.
    let want = if SPARSE_OUTPUT.contains(&kind) {
        10
    } else {
        300
    };
    assert!(
        ready > want,
        "{kind}: only {ready} rows had every slot ready"
    );
    zero_weight_rows_only_advance_the_clock(&build, targets, binary, kind);
}

/// Hard rules 8 and 9, together and without naming a decay: a zero-weight
/// row advances the clock and teaches nothing, so the `n_eff` a stream
/// reports after one must equal the `n_eff` of the same stream with that row
/// *left out* and its clock delta carried into the next row. For an
/// exponential decay `lam(a)·lam(b) = lam(a + b)`, so the two agree exactly
/// whatever the halflife -- and a model that forgets to decay `n_eff` on the
/// row it learns nothing from does not (docs/REVIEW-E54-E64.md H3/C2).
fn zero_weight_rows_only_advance_the_clock<M: OnlineModel>(
    build: &impl Fn() -> M,
    targets: usize,
    binary: bool,
    kind: &'static str,
) {
    let row = |s: &mut u64, i: usize| {
        let x: Vec<f64> = (0..K).map(|_| lcg(s) * 3.0).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1];
                Some(if binary { f64::from(lin > 0.5) } else { lin })
            })
            .collect();
        let d = match i {
            0 => 0.0,
            _ if i % 5 == 0 => 3.0,
            _ => 1.0,
        };
        (x, y, d)
    };
    // Every seventh row has weight 0, and so does the first row with a
    // prediction -- where a weight a model keeps beside `n_eff` can still be
    // 0, and a zero-weight row forms `0/0` in any division left unguarded
    // (review 2026-09-12, C7: `lasso`'s selection). That row is found on a
    // probe of the same stream.
    let zero = |i: usize| i % 7 == 3;
    let first = {
        let (mut probe, mut s) = (build(), 20260906u64);
        (0..80).find(|&i| {
            let (x, y, d) = row(&mut s, i);
            let w = if zero(i) { 0.0 } else { 1.0 };
            probe.step(&x, &y, d, w).pred.iter().any(|p| p.is_finite())
        })
    };
    let (mut with, mut without) = (build(), build());
    let mut s = 20260906u64;
    let mut carried = 0.0;
    for i in 0..80 {
        let (x, y, d) = row(&mut s, i);
        if zero(i) || first == Some(i) {
            with.step(&x, &y, d, 0.0);
            carried += d;
            continue;
        }
        let a = with.step(&x, &y, d, 1.0);
        let b = without.step(&x, &y, d + carried, 1.0);
        // `n_eff` is read *before* the row's own decay, so on the row that
        // carries the skipped one's delta the two are one decay apart by
        // construction; they must agree on every other row, and the row
        // after the carry is the one that says the streams re-converged.
        let compare = carried == 0.0;
        carried = 0.0;
        assert!(
            !compare || (a.n_eff - b.n_eff).abs() <= 1e-9 * b.n_eff.abs().max(1.0),
            "{kind}: row {i}: n_eff {} in the stream with a zero-weight row, {} in the one \
             without it -- a zero-weight row must advance the clock and nothing else",
            a.n_eff,
            b.n_eff
        );
        // Nor may it leave a NaN behind: a prediction the stream without it
        // makes, the stream with it makes too (hard rule 9).
        let finite = |p: &[f64]| p.iter().map(|v| v.is_finite()).collect::<Vec<_>>();
        assert!(
            !compare || finite(&a.pred) == finite(&b.pred),
            "{kind}: row {i}: predictions {:?} in the stream with a zero-weight row, {:?} \
             in the one without it",
            a.pred,
            b.pred
        );
    }
}

#[test]
fn ew_ridge_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| EwRidge::new(ew_ridge_cfg()).unwrap(), 2, false);
    let mut cfg = ew_ridge_cfg();
    cfg.standardize = true;
    cfg.session_shrink = Some(0.5);
    cfg.long_halflife = Some(4.0 * HALFLIFE);
    predict_is_the_step_without_the_step(move || EwRidge::new(cfg.clone()).unwrap(), 2, false);
    // A lazily refreshed solve: both read the cached coefficients.
    let mut cfg = ew_ridge_cfg();
    cfg.solve_every = 5.0;
    cfg.max_rows_between_solves = 10_000;
    predict_is_the_step_without_the_step(move || EwRidge::new(cfg.clone()).unwrap(), 2, false);
}

#[test]
fn rls_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Rls::new(rls_cfg()).unwrap(), 2, false);
}

#[test]
fn lasso_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Lasso::new(lasso_cfg()).unwrap(), 1, false);
    // With a selection halflife `lam_selected` moves; `extra` must match too.
    let mut cfg = lasso_cfg();
    cfg.lasso_path = vec![1.0, 0.1, 0.01, 0.0];
    cfg.select_halflife = Some(HALFLIFE);
    predict_is_the_step_without_the_step(move || Lasso::new(cfg.clone()).unwrap(), 1, false);
}

#[test]
fn kalman_predict_is_the_step() {
    for standardize in [true, false] {
        for reverting in [false, true] {
            // Under reversion the prediction depends on `d`: `predict(x, d)`
            // has to propagate the mean by the same `phi_i(d)` the step does.
            let mut cfg = if reverting {
                kalman_revert_cfg()
            } else {
                kalman_cfg()
            };
            cfg.standardize = standardize;
            predict_is_the_step_without_the_step(
                move || Kalman::new(cfg.clone()).unwrap(),
                2,
                false,
            );
        }
    }
}

#[test]
fn robust_predict_is_the_step() {
    for loss in ROBUST_LOSSES {
        for standardize in [false, true] {
            let mut cfg = robust_cfg(loss);
            cfg.standardize = standardize;
            predict_is_the_step_without_the_step(
                move || Robust::new(cfg.clone()).unwrap(),
                2,
                false,
            );
        }
    }
}

#[test]
fn ftrl_predict_is_the_step() {
    // `l1 > 0` so some proximal weights sit at exactly zero.
    let mut cfg = ftrl_cfg();
    cfg.l1 = 0.05;
    predict_is_the_step_without_the_step(move || Ftrl::new(cfg.clone()).unwrap(), 2, false);
    let mut cfg = ftrl_cfg();
    cfg.loss = FtrlLoss::Logistic;
    predict_is_the_step_without_the_step(move || Ftrl::new(cfg.clone()).unwrap(), 2, true);
}

#[test]
fn sgd_predict_is_the_step() {
    for scale_features in [false, true] {
        let mut cfg = sgd_cfg();
        cfg.scale_features = scale_features;
        predict_is_the_step_without_the_step(move || Sgd::new(cfg.clone()).unwrap(), 2, false);
    }
    let mut cfg = sgd_cfg();
    cfg.loss = SgdLoss::Logistic;
    predict_is_the_step_without_the_step(move || Sgd::new(cfg.clone()).unwrap(), 2, true);
}

#[test]
fn pa_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Pa::new(pa_cfg()).unwrap(), 2, false);
}

#[test]
fn holt_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Holt::new(holt_cfg()).unwrap(), 2, false);
}

#[test]
fn seqtest_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| SeqTest::new(seqtest_cfg()).unwrap(), 2, false);
}

#[test]
fn kmeans_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| KMeans::new(kmeans_cfg()).unwrap(), 0, false);
    let cfg = KMeansCfg {
        update_every: 7,
        seed_rule: SeedRule::Farthest,
        standardize: false,
        ..kmeans_cfg()
    };
    predict_is_the_step_without_the_step(move || KMeans::new(cfg.clone()).unwrap(), 0, false);
}

#[test]
fn micro_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Micro::new(micro_cfg()).unwrap(), 0, false);
    let cfg = MicroCfg {
        macro_link: Some(2.5),
        standardize: false,
        max_clusters: 6,
        ..micro_cfg()
    };
    predict_is_the_step_without_the_step(move || Micro::new(cfg.clone()).unwrap(), 0, false);
}

#[test]
fn ew_class_predict_is_the_step() {
    for covariance in [Covariance::Full, Covariance::Shared, Covariance::Diagonal] {
        let cfg = EwClassCfg {
            covariance,
            ..ew_class_cfg()
        };
        predict_is_the_step_without_the_step(move || EwClass::new(cfg.clone()).unwrap(), 1, true);
    }
}

#[test]
fn ew_cov_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| EwCovModel::new(ew_cov_model_cfg()).unwrap(), 0, false);
}

#[test]
fn marginal_predict_is_the_step() {
    // No slots to compare, so this holds `n_eff` and `extra` alone -- and
    // that `predict` did not move the state.
    predict_is_the_step_without_the_step(|| Marginal::new(marginal_cfg()).unwrap(), 2, false);
}

#[test]
fn bocpd_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Bocpd::new(bocpd_cfg()).unwrap(), 0, true);
}

#[test]
fn bocpd_recovers_from_bounded_extremes() {
    // A 1e100 row is a changepoint by any reading, and the posterior is
    // path-dependent, so a clean twin is not the criterion. What must hold
    // is a finite state, a run posterior that stays a distribution, and a
    // model that goes on reporting.
    let rows = bounded_script(0);
    let mut m = Bocpd::new(bocpd_cfg()).unwrap();
    for r in &rows {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite());
        let p = m.run_posterior();
        assert!(p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
    assert!(m.run_posterior().len() <= 200, "max_run holds");
}

#[test]
fn corrchange_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| CorrChange::new(corrchange_cfg()).unwrap(), 0, true);
}

#[test]
fn corrchange_recovers_from_bounded_extremes() {
    // A test of the data, not a model of it: what must hold is that the
    // state stays finite and it goes on testing. A statistic computed over
    // a span containing a 1e100 row is whatever it is; the twin, which
    // never saw that row, is testing a different span.
    let rows = bounded_script(0);
    let mut m = CorrChange::new(corrchange_cfg()).unwrap();
    let mut reported = 0;
    for r in &rows {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite());
        reported += usize::from(step.pred[0].is_finite());
    }
    assert!(reported > 100, "spans go on closing");
    assert!(m.n_eff().is_finite());
}

#[test]
fn hmm_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Hmm::new(hmm_cfg()).unwrap(), 0, true);
}

/// `bocpd`'s hazard and `hmm`'s exogenous value ride in the **targets**
/// slot, which `predict` does not see: it answered from the configured
/// default instead and disagreed with the step on every row where the
/// column differed from it. `predict_with` is the one the plumbing calls
/// (docs/REVIEW-E54-E64.md C1, B3, H2).
#[test]
fn a_value_in_the_targets_slot_reaches_predict() {
    let mut s = 20260906u64;
    let mut bo = Bocpd::new(BocpdCfg {
        hazard_from_row: true,
        ..bocpd_cfg()
    })
    .unwrap();
    let a = vec![2.0, -2.0, -2.0, 2.0];
    let b = vec![1.5, -1.5, -1.5, 1.5];
    let mut hm = Hmm::new(HmmCfg {
        tvtp: Some((a, b)),
        ..hmm_cfg()
    })
    .unwrap();
    // The row's value has to *matter*, or the parity below is vacuous:
    // count the rows where ignoring it gives a different answer.
    let (mut bo_moved, mut hm_moved) = (0usize, 0usize);
    for i in 0..300 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
        // A hazard between 2 and 1000, never the configured 50; an
        // exogenous value swinging either side of the default 0.
        let hz = [Some(2.0 + 499.0 * (1.0 + lcg(&mut s)))];
        let z = [Some(3.0 * lcg(&mut s))];
        let d = if i == 0 { 0.0 } else { 1.0 };

        let p = bo.predict_with(&x, &hz, d);
        bo_moved += usize::from(!same_pred(&p, &bo.predict(&x, d)));
        same_step("bocpd", i, &p, &bo.step(&x, &hz, d, 1.0));

        let p = hm.predict_with(&x, &z, d);
        hm_moved += usize::from(!same_pred(&p, &hm.predict(&x, d)));
        same_step("hmm", i, &p, &hm.step(&x, &z, d, 1.0));
    }
    assert!(
        bo_moved > 200,
        "bocpd: the hazard moved only {bo_moved} rows"
    );
    assert!(
        hm_moved > 200,
        "hmm: the exogenous value moved only {hm_moved} rows"
    );
}

/// Whether two steps report the same slots; `same_step` asserts it.
fn same_pred(a: &Step, b: &Step) -> bool {
    a.pred.len() == b.pred.len()
        && a.pred
            .iter()
            .zip(&b.pred)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

#[test]
fn hmm_recovers_from_bounded_extremes() {
    // Agreement with a clean twin is not a property a *filter* has: `p` is
    // path-dependent by construction, and one extra row -- even one the
    // filter rejects, because both densities underflowed -- leaves the two
    // copies on different paths through a sticky chain. What the contract
    // asks of every model is that the state stays finite and it goes on
    // learning, and what is a property here is the **fit**: the state means
    // return to the twin's once the extreme row has decayed away.
    let rows = bounded_script(0);
    let mut m = Hmm::new(hmm_cfg()).unwrap();
    for r in &rows {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite());
        for v in &step.pred {
            assert!(!v.is_nan() || step.pred.iter().all(|p| p.is_nan()));
        }
        // `p` is a distribution on every row it is reported.
        let p = &step.pred[..2];
        if p.iter().all(|v| v.is_finite()) {
            assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
    // A clean twin was stepped through this whole script and then discarded:
    // agreement is not a property a filter has (the comment above), so the
    // twin proved nothing and ran a full-covariance two-state filter for
    // ~30 500 rows on every `cargo test` for nothing (review 2026-09-18, T2).
    // Every state's moments are finite, and the model goes on filtering.
    for s in 0..2 {
        assert!(m.state_cov(s).means().iter().all(|v| v.is_finite()));
        assert!(m.state_cov(s).comoments().iter().all(|v| v.is_finite()));
    }
    // **A known limitation, and the reason this is not a twin test.** A
    // bounded-extreme row is captured by whichever state wins it, and that
    // state's mean moves to ~1e98. In mean form a state with zero
    // responsibility keeps its moments -- `EwCov::update` at weight 0 moves
    // nothing -- so a state that stops winning never forgets, and the
    // mixture is left with one live state. `docs/PLAN.md` §11a records it
    // with the mitigations. What must still hold is that the survivor
    // tracks the data: fed a clean two-blob stream, the filter's `state`
    // output follows the blob it is in.
    let mut right = 0;
    let rows = 2000;
    let mut seed = 991u64;
    for i in 0..rows {
        let g = (i / 50) % 2;
        let c = if g == 0 { -6.0 } else { 6.0 };
        let x: Vec<f64> = (0..K).map(|_| c + lcg(&mut seed)).collect();
        let step = m.step(&x, &[], 1.0, 1.0);
        let state = step.pred[2 * 2];
        if state.is_finite() && i > rows / 2 {
            // Either labelling of the two blobs is correct; count the one
            // that is consistent.
            right += usize::from((state as usize == g) == (m.filtered()[g] >= 0.5));
        }
    }
    assert!(right > 0, "the filter reports a state on a clean stream");
    assert!(
        m.state_cov(0).means().iter().all(|v| v.is_finite())
            && m.state_cov(1).means().iter().all(|v| v.is_finite())
    );
}

#[test]
fn rcov_predict_is_the_step() {
    // No slots, so this holds `n_eff` and that `predict` moved nothing.
    predict_is_the_step_without_the_step(|| Rcov::new(rcov_cfg()).unwrap(), 0, false);
}

#[test]
fn rcov_recovers_from_bounded_extremes() {
    // A 1e100 return dominates the sums for good -- there is no decay to
    // wash it out -- so the twin comparison is not the contract here; what
    // is, is that the state stays finite and the model goes on accepting
    // rows.
    let rows = bounded_script(0);
    let mut m = Rcov::new(rcov_cfg()).unwrap();
    for r in &rows {
        let step = m.step(&r.x, &r.y, 1.0, r.w);
        assert!(step.n_eff.is_finite());
    }
    let e = m.estimate();
    assert!(e.rcov.expect("a block").iter().all(|v| !v.is_nan()));
}

#[test]
fn deco_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Deco::new(deco_cfg()).unwrap(), 0, true);
}

#[test]
fn every_model_with_a_recovery_test_has_a_predict_parity_test() {
    // The recovery tests above are the per-model roll call of this file; each
    // `<model>_recovers_from_bounded_extremes` must have its
    // `<model>_predict_is_the_step` twin.
    let src = include_str!("model_contract.rs");
    let models: Vec<&str> = src
        .lines()
        .filter_map(|l| {
            l.strip_prefix("fn ")?
                .strip_suffix("_recovers_from_bounded_extremes() {")
        })
        .collect();
    assert!(models.len() >= 10, "found only {models:?}");
    for model in models {
        let name = format!("fn {model}_predict_is_the_step()");
        assert!(
            src.contains(&name),
            "{model}: add `{name}` to the predict parity tests"
        );
    }
}

// ---------------------------------------------------------------------------
// The contract over generated streams (T-D2 in Rust, docs/PLAN.md task 121).
// ---------------------------------------------------------------------------

// --- states written before a representation grew ---------------------------

/// A state as a msgpack value, edited, and read back: in the named encoding a
/// bank writes, where an infinite parameter stays infinite (serde_json would
/// write it as null and refuse it on the way back).
fn edit_state(state: &State, edit: impl FnOnce(&mut rmpv::Value)) -> State {
    let bytes = rmp_serde::to_vec_named(state).unwrap();
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    edit(&mut v);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    rmp_serde::from_slice(&out).unwrap()
}

/// Every map entry whose key ends in `_lo`, removed, at any depth: the
/// means' low parts (docs/PLAN.md task 101), which a state written before
/// them does not carry. The count removed.
fn strip_low_parts(v: &mut rmpv::Value) -> usize {
    match v {
        rmpv::Value::Map(entries) => {
            let before = entries.len();
            entries.retain(|(k, _)| !k.as_str().is_some_and(|k| k.ends_with("_lo")));
            let mut n = before - entries.len();
            for (_, x) in entries.iter_mut() {
                n += strip_low_parts(x);
            }
            n
        }
        rmpv::Value::Array(a) => a.iter_mut().map(strip_low_parts).sum(),
        _ => 0,
    }
}

/// A stream at a level, so the means carry something below their doubles,
/// with a target now and then absent and weights that move.
fn leveled(targets: usize, binary: bool) -> Vec<Row> {
    let mut s = 20260927u64;
    (0..160)
        .map(|i| {
            let x: Vec<f64> = (0..K)
                .map(|j| 1e3 * (j as f64 + 1.0) + lcg(&mut s))
                .collect();
            let y = (0..targets)
                .map(|j| {
                    let v = 3.0 + x[0] - 0.5 * x[1] + 0.3 * lcg(&mut s);
                    (i % 9 != 4 + j).then_some(if binary { f64::from(v > 3.0) } else { v })
                })
                .collect();
            Row {
                x,
                y,
                w: 0.5 + (lcg(&mut s) + 1.0) / 2.0,
                extreme: false,
            }
        })
        .collect()
}

/// A model restored at row 80 from its state with every low part removed --
/// a state written before them -- next to the model restored whole: the
/// number of low parts the state had, and the worst gap between the two
/// afterwards, relative to `1 + |v|`. Each low part starts again at zero, so
/// the two part by the few rounding steps the dropped parts held.
fn resumes_without_low_parts<M: OnlineModel>(make: impl Fn() -> M, rows: &[Row]) -> (usize, f64) {
    let mut m = make();
    for (i, r) in rows.iter().enumerate().take(80) {
        m.step(&r.x, &r.y, if i == 0 { 0.0 } else { 1.0 }, r.w);
    }
    let state = m.state();
    let mut stripped = 0;
    let old = edit_state(&state, |v| stripped = strip_low_parts(v));
    let (mut whole, mut before) = (M::restore(&state).unwrap(), M::restore(&old).unwrap());
    let mut worst = 0.0f64;
    for r in &rows[80..] {
        let (a, b) = (
            whole.step(&r.x, &r.y, 1.0, r.w),
            before.step(&r.x, &r.y, 1.0, r.w),
        );
        let (u_all, v_all): (Vec<f64>, Vec<f64>) = (
            a.pred.iter().copied().chain([a.n_eff]).collect(),
            b.pred.iter().copied().chain([b.n_eff]).collect(),
        );
        for (u, v) in u_all.iter().zip(&v_all) {
            assert_eq!(u.is_nan(), v.is_nan(), "{u} against {v}");
            if u.is_finite() {
                worst = worst.max((u - v).abs() / (1.0 + u.abs()));
            }
        }
    }
    (stripped, worst)
}

/// Every model loads a state written before the means' low parts and goes on
/// as it would have (docs/PLAN.md task 110).
#[test]
fn every_model_resumes_from_a_state_without_the_low_parts() {
    let two = leveled(2, false);
    let one = leveled(1, false);
    let label = leveled(1, true);
    let label2 = leveled(2, true);
    let none = leveled(0, false);
    let results: Vec<(&str, (usize, f64))> = vec![
        (
            "ew_ridge",
            resumes_without_low_parts(|| EwRidge::new(ew_ridge_cfg()).unwrap(), &two),
        ),
        (
            "rls",
            resumes_without_low_parts(|| Rls::new(rls_cfg()).unwrap(), &two),
        ),
        (
            "lasso",
            resumes_without_low_parts(|| Lasso::new(lasso_cfg()).unwrap(), &one),
        ),
        (
            "kalman",
            resumes_without_low_parts(|| Kalman::new(kalman_cfg()).unwrap(), &two),
        ),
        (
            "kalman_revert",
            resumes_without_low_parts(|| Kalman::new(kalman_revert_cfg()).unwrap(), &two),
        ),
        (
            "huber",
            resumes_without_low_parts(|| Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap(), &two),
        ),
        (
            "quantile",
            resumes_without_low_parts(|| Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap(), &two),
        ),
        (
            "ftrl",
            resumes_without_low_parts(|| Ftrl::new(ftrl_cfg()).unwrap(), &label2),
        ),
        (
            "sgd",
            resumes_without_low_parts(|| Sgd::new(sgd_cfg()).unwrap(), &two),
        ),
        (
            "pa",
            resumes_without_low_parts(|| Pa::new(pa_cfg()).unwrap(), &two),
        ),
        (
            "holt",
            resumes_without_low_parts(|| Holt::new(holt_cfg()).unwrap(), &two),
        ),
        (
            "ew_cov",
            resumes_without_low_parts(|| EwCovModel::new(ew_cov_model_cfg()).unwrap(), &none),
        ),
        (
            "kmeans",
            resumes_without_low_parts(|| KMeans::new(kmeans_cfg()).unwrap(), &none),
        ),
        (
            "micro",
            resumes_without_low_parts(|| Micro::new(micro_cfg()).unwrap(), &none),
        ),
        (
            "ew_class",
            resumes_without_low_parts(|| EwClass::new(ew_class_cfg()).unwrap(), &label),
        ),
        (
            "seqtest",
            resumes_without_low_parts(|| SeqTest::new(seqtest_cfg()).unwrap(), &two),
        ),
        (
            "marginal",
            resumes_without_low_parts(|| Marginal::new(marginal_cfg()).unwrap(), &two),
        ),
        (
            "deco",
            resumes_without_low_parts(|| Deco::new(deco_cfg()).unwrap(), &none),
        ),
        (
            "rcov",
            resumes_without_low_parts(|| Rcov::new(rcov_cfg()).unwrap(), &none),
        ),
        (
            "hmm",
            resumes_without_low_parts(|| Hmm::new(hmm_cfg()).unwrap(), &none),
        ),
        (
            "corrchange",
            resumes_without_low_parts(|| CorrChange::new(corrchange_cfg()).unwrap(), &none),
        ),
        (
            "bocpd",
            resumes_without_low_parts(|| Bocpd::new(bocpd_cfg()).unwrap(), &none),
        ),
    ];
    for (name, (stripped, worst)) in &results {
        eprintln!("{name:>14}: {stripped:>3} low parts dropped, worst gap {worst:.2e}");
    }
    // Measured 2026-09-27: 3.7e-13 at worst (`deco`), 1.6e-13 for the rest.
    for (name, (_, worst)) in &results {
        assert!(
            *worst <= 1e-11,
            "{name}: {worst:e} from the model restored whole"
        );
    }
    // Each model that keeps a mean had parts to drop, so its load was a real
    // one: a model that lost its means, or a renamed field, shows here.
    let keeps_means = [
        "ew_ridge",
        "lasso",
        "kalman",
        "kalman_revert",
        "huber",
        "quantile",
        "ew_cov",
        "kmeans",
        "micro",
        "ew_class",
        "marginal",
        "deco",
        "hmm",
        "corrchange",
        "bocpd",
    ];
    for (name, (stripped, _)) in &results {
        if keeps_means.contains(name) {
            assert!(*stripped > 0, "{name}: no low part in its state");
        }
    }
}

/// A windowed `ew_ridge` or `lasso` saved at schema 16, when each target's
/// own feature mean was kept as its offset `d` from the all-row mean,
/// loads at 17 with the offsets made means once, in the live accumulator
/// and in every window snapshot, and goes on as the model restored whole
/// does (review 2026-09-27, G3). The state is rewritten to 16's shape: the
/// field named `d`, holding `mj - m`, and no `mj_lo`.
#[test]
fn a_schema_16_state_of_offsets_loads_as_own_means() {
    fn to_offsets(v: &mut rmpv::Value, n: &mut usize) {
        match v {
            rmpv::Value::Map(entries) => {
                let m: Option<Vec<f64>> = entries
                    .iter()
                    .find(|(k, _)| k.as_str() == Some("m"))
                    .and_then(|(_, v)| v.as_array())
                    .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect());
                entries.retain(|(k, _)| k.as_str() != Some("mj_lo"));
                for (k, x) in entries.iter_mut() {
                    if k.as_str() == Some("mj") {
                        let m = m.as_ref().expect("mj sits beside m");
                        let rmpv::Value::Array(rows) = x else {
                            panic!("mj is a list of lists")
                        };
                        for row in rows {
                            let rmpv::Value::Array(row) = row else {
                                panic!("mj is a list of lists")
                            };
                            for (i, e) in row.iter_mut().enumerate() {
                                *e = rmpv::Value::F64(e.as_f64().unwrap() - m[i]);
                            }
                        }
                        *k = rmpv::Value::from("d");
                        *n += 1;
                    } else {
                        to_offsets(x, n);
                    }
                }
            }
            rmpv::Value::Array(a) => a.iter_mut().for_each(|x| to_offsets(x, n)),
            _ => {}
        }
    }
    fn schema_16(state: &State) -> (State, usize) {
        let mut n = 0;
        let old = edit_state(state, |v| {
            if let rmpv::Value::Map(entries) = v {
                for (k, x) in entries.iter_mut() {
                    if k.as_str() == Some("schema_version") {
                        *x = rmpv::Value::from(16u32);
                    }
                }
            }
            to_offsets(v, &mut n);
        });
        (old, n)
    }
    fn check<M: OnlineModel>(name: &str, make: impl Fn() -> M, rows: &[Row]) {
        let mut m = make();
        for (i, r) in rows.iter().enumerate().take(80) {
            m.step(&r.x, &r.y, if i == 0 { 0.0 } else { 1.0 }, r.w);
        }
        let state = m.state();
        let (old, converted) = schema_16(&state);
        // The live accumulator, and the window's snapshots besides.
        assert!(converted >= 2, "{name}: {converted} offset vectors written");
        let (mut whole, mut loaded) = (M::restore(&state).unwrap(), M::restore(&old).unwrap());
        for r in &rows[80..] {
            let (a, b) = (
                whole.step(&r.x, &r.y, 1.0, r.w),
                loaded.step(&r.x, &r.y, 1.0, r.w),
            );
            for (u, v) in a.pred.iter().zip(&b.pred) {
                assert_eq!(u.is_nan(), v.is_nan(), "{name}: {u} against {v}");
                if u.is_finite() {
                    let gap = (u - v).abs() / (1.0 + u.abs());
                    assert!(gap <= 1e-9, "{name}: {u} against {v}, {gap:e}");
                }
            }
        }
    }
    let rows = leveled(2, false);
    for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
        let ridge = || {
            let mut c = ew_ridge_cfg();
            c.window = Some(30.0);
            c.window_every = Some(4);
            c.target_gaps = gaps;
            EwRidge::new(c).unwrap()
        };
        check(&format!("ew_ridge {gaps:?}"), ridge, &rows);
    }
    let one = leveled(1, false);
    let lasso = || {
        let mut c = lasso_cfg();
        c.window = Some(30.0);
        c.window_every = Some(4);
        c.target_gaps = TargetGaps::Pairwise;
        Lasso::new(c).unwrap()
    };
    check("lasso", lasso, &one);
}

/// `tests/test_properties.py` asserts invariants on the bank over streams
/// Hypothesis generates. This asserts three clauses of the contract above on
/// every model directly, over streams proptest generates and, on a failure,
/// shrinks to a short one: `predict_with` is the step without the update; a
/// state saved and restored at any row continues exactly as the model that
/// was not; and nothing a model reports is infinite, whatever the inputs
/// inside the bound. Each is checked above on one fixed stream; here the
/// stream is the variable. 128 streams a model in the suite; 2,000 a model,
/// run once when this was written (docs/PLAN.md task 121), found nothing.
/// The streams are drawn afresh each run (proptest seeds from the OS): a
/// failure prints the shrunk stream, which is what to keep as a fixed case,
/// and `PROPTEST_RNG_SEED=<n>` repeats a run (review 2026-09-26, C8).
mod generated {
    use super::*;
    use proptest::prelude::*;

    /// One row: the features, each target or its absence, the clock since
    /// the previous row, and the weight.
    #[derive(Debug, Clone)]
    struct GenRow {
        x: Vec<f64>,
        y: Vec<Option<f64>>,
        d: f64,
        w: f64,
    }

    /// A value a stream may carry: mostly ordinary, often a repeat that holds
    /// a column still, sometimes wide, sometimes at the input bound itself.
    fn value() -> impl Strategy<Value = f64> {
        prop_oneof![
            6 => -3.0..3.0f64,
            2 => Just(0.5),
            1 => -1e6..1e6f64,
            1 => (-1.0..1.0f64).prop_map(|u| u * 1e50),
            1 => prop_oneof![
                Just(INPUT_BOUND),
                Just(-INPUT_BOUND),
                (-1.0..1.0f64).prop_map(|u| u * INPUT_BOUND)
            ],
        ]
    }

    /// A row of `targets` targets; `binary` targets are 0 or 1, the label a
    /// classifier and a logistic loss read.
    fn row(targets: usize, binary: bool) -> impl Strategy<Value = GenRow> {
        (
            prop::collection::vec(value(), K),
            prop::collection::vec(prop::option::weighted(0.85, value()), targets),
            prop_oneof![5 => Just(1.0), 1 => Just(0.0), 1 => 0.0..30.0f64],
            prop_oneof![
                5 => Just(1.0),
                1 => Just(0.0),
                2 => 0.01..4.0f64,
                1 => Just(1e-100),
                1 => Just(1e100)
            ],
        )
            .prop_map(move |(x, y, d, w)| GenRow {
                x,
                y: if binary {
                    y.into_iter()
                        .map(|v| v.map(|v| f64::from(v > 0.0)))
                        .collect()
                } else {
                    y
                },
                d,
                w,
            })
    }

    fn stream(targets: usize, binary: bool) -> impl Strategy<Value = Vec<GenRow>> {
        prop::collection::vec(row(targets, binary), 1..60)
    }

    fn equal(a: &Step, b: &Step) -> bool {
        let same = |u: &f64, v: &f64| u.to_bits() == v.to_bits() || (u.is_nan() && v.is_nan());
        a.pred.len() == b.pred.len()
            && a.pred.iter().zip(&b.pred).all(|(u, v)| same(u, v))
            && same(&a.n_eff, &b.n_eff)
            && a.extra == b.extra
    }

    /// The three clauses over one stream, with the save and restore at row
    /// `split` (past the end: never).
    fn contract<M: OnlineModel>(
        build: impl Fn() -> M,
        rows: &[GenRow],
        split: usize,
    ) -> Result<(), TestCaseError> {
        let (mut whole, mut parted) = (build(), build());
        for (i, r) in rows.iter().enumerate() {
            if i == split {
                let bytes = rmp_serde::to_vec(&parted.state()).unwrap();
                parted = M::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            }
            let d = if i == 0 { 0.0 } else { r.d };
            let p = whole.predict_with(&r.x, &r.y, d);
            let a = whole.step(&r.x, &r.y, d, r.w);
            let b = parted.step(&r.x, &r.y, d, r.w);
            prop_assert!(equal(&p, &a), "row {}: predict {:?}, step {:?}", i, p, a);
            prop_assert!(
                equal(&a, &b),
                "row {}, restored at {}: {:?} against {:?}",
                i,
                split,
                b,
                a
            );
            prop_assert!(
                a.pred
                    .iter()
                    .chain(std::iter::once(&a.n_eff))
                    .all(|v| !v.is_infinite()),
                "row {}: {:?}",
                i,
                a
            );
        }
        Ok(())
    }

    /// The stream the quantile `robust` failed the contract on once the
    /// values reached the bound (review 2026-09-26, G2): a target of `1e100`,
    /// then features of `1e100` at weights from `1e-100`, and the prediction
    /// at `1e100` read `-inf` (`robust.rs` has the mechanism and the fix).
    #[test]
    fn robust_quantile_through_the_bound() {
        let row = |x0: f64, y0: Option<f64>, w: f64| GenRow {
            x: vec![x0, 0.0],
            y: vec![y0, None],
            d: 1.0,
            w,
        };
        let rows = vec![
            row(0.0, None, 1.0),
            row(0.0, Some(0.0), 1.0),
            row(0.0, Some(0.0), 1.0),
            row(0.0, Some(0.0), 1.0),
            row(0.0, Some(0.0), 1.1785268524789025),
            row(0.0, Some(0.0), 1.0),
            row(0.0, Some(1e100), 2.0699847121199655),
            row(0.0, Some(0.0), 1.4511913482607992),
            row(0.0, Some(0.0), 1.0),
            row(1.4589775263789706, Some(0.0), 1.0),
            row(1e100, Some(0.0), 1e-100),
            row(0.0, Some(0.0), 3.8874731502776667),
            row(1e100, Some(0.0), 1.0),
            row(1e100, None, 1.0),
        ];
        contract(
            || Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap(),
            &rows,
            0,
        )
        .unwrap();
    }

    /// The stream the windowed `lasso` failed the contract on once the
    /// values reached the bound (review 2026-09-26, G3): a target absent on
    /// a row whose feature stood at `-2.6e99`, then rows at `1e100`, and the
    /// prediction read `-inf` -- windowed or not, and through `ew_ridge`,
    /// which shares the cross accumulator (`gaps.rs` has the mechanism).
    #[test]
    fn a_target_absent_on_a_row_at_the_bound_through_the_bound() {
        let row = |x: [f64; 2], y0: Option<f64>, w: f64| GenRow {
            x: x.to_vec(),
            y: vec![y0],
            d: 1.0,
            w,
        };
        let rows = vec![
            row([0.0, -2.570413849510098e99], None, 1.0),
            row([0.0, 0.0], None, 1.0),
            row([0.0, 0.0], Some(0.0), 1.0),
            row(
                [0.0, -0.39761046383362925],
                Some(-5.567947375697575e49),
                1.0,
            ),
            row([1.4616143340058023, 1e100], Some(0.0), 0.01),
            row([1e100, 0.0], None, 1.0),
        ];
        for window in [None, Some(7.0)] {
            contract(
                || {
                    let mut c = lasso_cfg();
                    c.window = window;
                    Lasso::new(c).unwrap()
                },
                &rows,
                0,
            )
            .unwrap();
        }
        let two: Vec<GenRow> = rows
            .iter()
            .map(|r| GenRow {
                y: vec![r.y[0], r.y[0]],
                ..r.clone()
            })
            .collect();
        contract(|| EwRidge::new(ew_ridge_cfg()).unwrap(), &two, 0).unwrap();
    }

    /// The stream the windowed `lasso` failed the contract on (review
    /// 2026-09-27, G5): rows at `1e100` before the window, one of weight
    /// `1e100` inside it, and the prediction at `x1 = 1e100` read `-inf`
    /// (`crate::truncated` has the mechanism). Through `ew_ridge` too.
    #[test]
    fn a_window_the_live_state_cannot_resolve_through_the_bound() {
        let row = |x: [f64; 2], y0: Option<f64>, d: f64, w: f64| GenRow {
            x: x.to_vec(),
            y: vec![y0],
            d,
            w,
        };
        let rows = vec![
            row([0.0, 0.0], Some(0.0), 1.0, 1.0),
            row([0.0, 1e100], Some(1e100), 0.0, 1.0),
            row([0.0, 0.0], Some(0.0), 1.0, 0.714463635904142),
            row([0.0, 0.0], Some(0.0), 1.0, 1.0),
            row([0.0, 0.0], None, 1.0, 1.0),
            row([0.0, 0.0], None, 1.0, 1.0),
            row([1e100, -7.726806526869417e99], Some(0.0), 1.0, 1.0),
            row([0.0, 0.0], Some(0.0), 12.875843092015453, 0.01),
            row([0.0, 0.0], Some(0.0), 1.0, 1.0),
            row([0.0, 0.0], Some(0.0), 1.0, 1.0),
            row(
                [-2.9807652788918353, 1.021121247753338],
                Some(0.0),
                1.0,
                1e100,
            ),
            row([0.0, 0.0], None, 1.0, 1.0),
            row([0.0, 0.0], None, 0.0, 1.0),
            row([0.0, 0.0], None, 1.0, 1.0),
            row([0.0, 1e100], None, 1.0, 1.0),
        ];
        let build = || {
            let mut c = lasso_cfg();
            c.window = Some(7.0);
            Lasso::new(c).unwrap()
        };
        contract(build, &rows, 0).unwrap();
        let two: Vec<GenRow> = rows
            .iter()
            .map(|r| GenRow {
                y: vec![r.y[0], r.y[0]],
                ..r.clone()
            })
            .collect();
        contract(
            || {
                let mut c = ew_ridge_cfg();
                c.window = Some(7.0);
                EwRidge::new(c).unwrap()
            },
            &two,
            0,
        )
        .unwrap();
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

        #[test]
        fn ew_ridge(rows in stream(2, false), split in 0usize..60) {
            contract(|| EwRidge::new(ew_ridge_cfg()).unwrap(), &rows, split)?;
        }

        /// The window's ring, its snapshots and their serde, under both
        /// cadences; and the blocked Gram with pairwise gaps, whose held
        /// rows the state carries (review 2026-09-26, C8).
        #[test]
        fn ew_ridge_windowed(rows in stream(2, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = ew_ridge_cfg();
                c.window = Some(7.0);
                c.window_every = [None, Some(3)][every];
                EwRidge::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn ew_ridge_blocked_pairwise(rows in stream(2, false), split in 0usize..60) {
            contract(|| {
                let mut c = ew_ridge_cfg();
                // A block needs a solve cadence, or it never holds a second row.
                c.gram_block_rows = 4;
                c.solve_every = 3.0;
                c.max_rows_between_solves = 8;
                c.target_gaps = online_core::TargetGaps::Pairwise;
                EwRidge::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn rls(rows in stream(2, false), split in 0usize..60) {
            contract(|| Rls::new(rls_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn lasso(rows in stream(1, false), split in 0usize..60) {
            contract(|| Lasso::new(lasso_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn lasso_windowed(rows in stream(1, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = lasso_cfg();
                c.window = Some(7.0);
                c.window_every = [None, Some(3)][every];
                Lasso::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn kalman(rows in stream(2, false), split in 0usize..60) {
            contract(|| Kalman::new(kalman_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn kalman_reverting(rows in stream(2, false), split in 0usize..60) {
            contract(|| Kalman::new(kalman_revert_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn robust(rows in stream(2, false), split in 0usize..60, which in 0usize..ROBUST_LOSSES.len()) {
            contract(|| Robust::new(robust_cfg(ROBUST_LOSSES[which])).unwrap(), &rows, split)?;
        }

        #[test]
        fn ftrl(rows in stream(2, false), split in 0usize..60) {
            contract(|| Ftrl::new(ftrl_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn sgd(rows in stream(2, false), split in 0usize..60) {
            contract(|| Sgd::new(sgd_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn pa(rows in stream(2, false), split in 0usize..60) {
            contract(|| Pa::new(pa_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn holt(rows in stream(2, false), split in 0usize..60) {
            contract(|| Holt::new(holt_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn ew_cov(rows in stream(0, false), split in 0usize..60) {
            contract(|| EwCovModel::new(ew_cov_model_cfg()).unwrap(), &rows, split)?;
        }

        /// A window keeps no lags (the two rings do not combine), so the
        /// windowed `ew_cov` is the moments alone.
        #[test]
        fn ew_cov_windowed(rows in stream(0, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = ew_cov_model_cfg();
                c.lags = vec![];
                c.stats.retain(|s| *s != EwCovStat::LagCorr);
                c.window = Some(7.0);
                c.window_every = [None, Some(3)][every];
                EwCovModel::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn marginal(rows in stream(2, false), split in 0usize..60) {
            contract(|| Marginal::new(marginal_cfg()).unwrap(), &rows, split)?;
        }

        /// A window keeps no lags, so the windowed marginal is the moments
        /// alone.
        #[test]
        fn marginal_windowed(rows in stream(2, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = marginal_cfg();
                c.lags = vec![];
                c.window = Some(7.0);
                c.window_every = [None, Some(3)][every];
                Marginal::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn kmeans(rows in stream(0, false), split in 0usize..60) {
            contract(|| KMeans::new(kmeans_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn micro(rows in stream(0, false), split in 0usize..60) {
            contract(|| Micro::new(micro_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn ew_class(rows in stream(1, true), split in 0usize..60) {
            contract(|| EwClass::new(ew_class_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn ew_class_windowed(rows in stream(1, true), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = ew_class_cfg();
                c.window = Some(7.0);
                c.window_every = [None, Some(3)][every];
                EwClass::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn seqtest(rows in stream(2, false), split in 0usize..60) {
            contract(|| SeqTest::new(seqtest_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn bocpd(rows in stream(0, false), split in 0usize..60) {
            contract(|| Bocpd::new(bocpd_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn corrchange(rows in stream(0, false), split in 0usize..60) {
            contract(|| CorrChange::new(corrchange_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn hmm(rows in stream(0, false), split in 0usize..60) {
            contract(|| Hmm::new(hmm_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn rcov(rows in stream(0, false), split in 0usize..60) {
            contract(|| Rcov::new(rcov_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn deco(rows in stream(0, false), split in 0usize..60) {
            contract(|| Deco::new(deco_cfg()).unwrap(), &rows, split)?;
        }
    }
}
