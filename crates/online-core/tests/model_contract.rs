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
/// the same number, and reporting two different ones would make `min_weight`
/// mean something different from what a caller inspecting the model sees.
fn probe_with<M: OnlineModel + Clone>(
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

    // A value that is not usable, on the warm model, with the model's own
    // accessor for the `n_eff` the refused row reports (task 183). On
    // copies: `m` goes on below as it was.
    let mut s4 = s;
    let rows: Vec<Row3> = (0..13)
        .map(|_| {
            let x: Vec<f64> = (0..K).map(|_| lcg(&mut s4)).collect();
            let y = (0..targets)
                .map(|j| Some(0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1]))
                .collect();
            (x, y, 1.0)
        })
        .collect();
    refuses_unusable_values(&m, &rows[0], &rows[1..], kind, n_eff_of);

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

/// Slots null by design except on a flag, which the ready count leaves out:
/// `corrchange`'s `since_change` dates a change only where one is flagged
/// (docs/PLAN.md task 114).
const ONLY_ON_A_FLAG: &[(&str, usize)] = &[("corrchange", 4)];

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
    // the same way; that uniformity is what makes `min_weight` portable.
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

    // A ten-half-life gap must decay it by 2^-10, not reset or ignore it.
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

fn ewridge_cfg() -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        ridge: vec![1e-6, 0.1],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: online_core::TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

#[test]
fn ewridge() {
    let cfg = ewridge_cfg();
    let r = probe_with(EwRidge::new(cfg).unwrap(), 2, Some(&EwRidge::n_eff));
    check(&r, "ewridge", 2, 2);
}

fn rls_cfg() -> RlsCfg {
    RlsCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        delta: 1.0,
        coef_prior: None,
        min_weight: 3.0,
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
        fit_intercept: true,
        decay: decay(),
        lasso_path: vec![0.1, 0.0],
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: 3.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        max_iter: 100,
        tol: 1e-10,
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
        fit_intercept: true,
        decay: decay(),
        half_life: vec![100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
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
        revert_half_life: vec![f64::INFINITY, 25.0, 6.0],
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
        fit_intercept: true,
        decay: decay(),
        loss,
        ridge: 1e-6,
        standardize: false,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
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
        fit_intercept: true,
        decay: decay(),
        alpha: 0.1,
        beta: 1.0,
        l1: 0.0,
        l2: 1.0,
        min_weight: 3.0,
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
        fit_intercept: true,
        decay: decay(),
        loss: SgdLoss::Squared,
        learning_rate: 0.01,
        schedule: LearningRate::Constant,
        l2: 0.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize: false,
        strict_binary: false,
        min_weight: 3.0,
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
        fit_intercept: true,
        decay: decay(),
        mode: PaMode::Pa1,
        c: 1.0,
        eps: 0.1,
        min_weight: 3.0,
        constraint: None,
        standardize: false,
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
        level_half_life: HALFLIFE,
        trend_half_life: 4.0 * HALFLIFE,
        min_weight: 3.0,
        trend: true,
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
        min_weight: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        // The probe carries lags so that the save/restore and `clear_lags`
        // arms above see a model with a ring in them (L1).
        lags: vec![1, 3],
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
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
    // The decayed row and the gap, as the sibling probes hold them (review
    // 2026-10-05, CF1: this probe read rows 0 and 1 alone).
    assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
    assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
    assert!(r.roundtrips);
}

fn kmeans_cfg() -> KMeansCfg {
    KMeansCfg {
        n_features: K,
        k: 3,
        decay: decay(),
        min_weight: 3.0,
        warm_rows: 10,
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

#[test]
fn kmeans() {
    let cfg = kmeans_cfg();
    // No targets: it emits an assignment and two distances.
    let m = KMeans::new(cfg).unwrap();
    assert_eq!(m.n_targets(), 0, "kmeans has no targets");
    assert_eq!(m.n_features(), K);
    assert_eq!(m.n_outputs(), 3, "cluster, dist, dist_second");
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
    // eps and beta_mu sized for a 20-half-life window over the harness's
    // uniform rows: a summary must reach beta_mu before it decays back.
    MicroCfg {
        n_features: K,
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
        "cluster, dist, micro_id, outlier, n_clusters, n_micro"
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
        "ewridge",
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
    let states = [
        EwRidge::new(ewridge_cfg()).unwrap().state(),
        Rls::new(rls_cfg()).unwrap().state(),
        Lasso::new(lasso_cfg()).unwrap().state(),
        Kalman::new(kalman_cfg()).unwrap().state(),
        Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap().state(),
        Ftrl::new(ftrl_cfg()).unwrap().state(),
        Sgd::new(sgd_cfg()).unwrap().state(),
        Pa::new(pa_cfg()).unwrap().state(),
        Holt::new(holt_cfg()).unwrap().state(),
        EwCovModel::new(ew_cov_model_cfg()).unwrap().state(),
        KMeans::new(kmeans_cfg()).unwrap().state(),
        Micro::new(micro_cfg()).unwrap().state(),
        EwClass::new(ew_class_cfg()).unwrap().state(),
        SeqTest::new(seqtest_cfg()).unwrap().state(),
        Marginal::new(marginal_cfg()).unwrap().state(),
        Deco::new(deco_cfg()).unwrap().state(),
        Rcov::new(rcov_cfg()).unwrap().state(),
        Hmm::new(hmm_cfg()).unwrap().state(),
        CorrChange::new(corrchange_cfg()).unwrap().state(),
        Bocpd::new(bocpd_cfg()).unwrap().state(),
    ];
    let from_states: Vec<&str> = states.iter().map(|s| s.model.kind()).collect();
    assert_eq!(from_states, kinds);
    // Each state's variant beside its kind is a pair `PROBED` lists, so the
    // probe found by the kind is the variant's (review 2026-10-06, CF12).
    let bare = State::new(ModelState::EwCov(Box::new(EwCov::new(1))));
    for s in states.iter().chain([&bare]) {
        let debug = format!("{:?}", s.model);
        let variant = debug.split('(').next().unwrap();
        assert!(
            PROBED.contains(&(variant, s.model.kind())),
            "{variant} names {} and PROBED pairs it otherwise",
            s.model.kind()
        );
    }
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
        min_weight: 3.0,
        covariance: Covariance::Full,
        precision_prior: 0.1,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
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
        // `min_weight` means the same number of rows as everywhere else.
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
        min_weight: 3.0,
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
        min_weight: vec![3.0; 2],
        // A lag, so the contract exercises marginal's `clear_lags` arm (it
        // keeps a lag ring; review 2026-09-18, T2) -- which is why marginal is
        // in KEEPS_LAGS above.
        lags: vec![1],
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: online_core::FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
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
        min_weight: 3.0,
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
        min_weight: 0.0,
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
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
    }
}

#[test]
fn corrchange() {
    // A test, not a model of the data: no targets, five slots (the
    // statistic, its critical value, the flag, the rows since the last one
    // and since the change it dates), and `n_eff` on the shared recursion.
    // The sequential kind (Wied & Galeano) keeps the same contract.
    for kind in [CorrChangeKind::Monitor, CorrChangeKind::Sequential] {
        let m = CorrChange::new(CorrChangeCfg {
            kind,
            monitor_rows: 15,
            ..corrchange_cfg()
        })
        .unwrap();
        assert_eq!(m.n_targets(), 0);
        assert_eq!(m.n_features(), K);
        assert_eq!(
            m.n_outputs(),
            5,
            "stat, crit, flag, since_flag, since_change"
        );
        let r = probe_with(m, 0, Some(&CorrChange::n_eff));
        assert_eq!(r.kind, "corrchange");
        assert_eq!(r.pred_len, r.n_outputs);
        assert_eq!(r.n_eff[0], 0.0);
        assert_eq!(r.n_eff[1], 1.0);
        assert!((r.n_eff[2] - (0.5f64.powf(1.0 / HALFLIFE) + 1.0)).abs() < 1e-12);
        assert!((r.after_gap - (r.before_gap * 0.5f64.powi(10) + 1.0)).abs() < 1e-9);
        assert!(r.roundtrips, "{kind:?}");
    }
}

fn bocpd_cfg() -> BocpdCfg {
    BocpdCfg {
        n_features: K,
        hazard: 50.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: Some(vec![0.0; K]),
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-6,
        max_run: 200,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: false,
    }
}

/// [`bocpd_cfg`] with its hazard on the clock: `τ = 20` clock units between
/// changepoints, each step's chance of a break applied before the row it
/// leads into (docs/PLAN.md task 179).
fn bocpd_on_the_clock_cfg() -> BocpdCfg {
    BocpdCfg {
        hazard: 20.0,
        hazard_on_clock: true,
        ..bocpd_cfg()
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

/// Every model with a decay of its own refuses, through its own `new`, a
/// decay it cannot run on -- a half-life of 0, below 0 or NaN, a factor of
/// 0, below 0, above 1 or NaN -- naming itself and the parameter, and runs
/// on no decay (`Halflife(inf)`, `Lam(1)`) and an ordinary one. Only the
/// bank's spec checked (`Spec::decays`), so the Rust API built each on any
/// of them, and `Halflife(0)` gave NaN from the first row (review
/// 2026-10-05, CF5; docs/EXTENDING.md: every parameter check belongs in
/// `new`). `holt`, `seqtest`, `rcov` and `bocpd` take no `Decay`.
#[test]
fn every_model_refuses_a_decay_it_cannot_run_on() {
    type Build<'a> = (&'a str, &'a dyn Fn(Decay) -> Result<(), String>);
    let builds: [Build; 17] = [
        ("ewridge", &|decay| {
            EwRidge::new(EwRidgeCfg {
                decay,
                ..ewridge_cfg()
            })
            .map(drop)
        }),
        ("rls", &|decay| {
            Rls::new(RlsCfg { decay, ..rls_cfg() }).map(drop)
        }),
        ("lasso", &|decay| {
            Lasso::new(LassoCfg {
                decay,
                ..lasso_cfg()
            })
            .map(drop)
        }),
        ("kalman", &|decay| {
            Kalman::new(KalmanCfg {
                decay,
                ..kalman_cfg()
            })
            .map(drop)
        }),
        ("huber", &|decay| {
            Robust::new(RobustCfg {
                decay,
                ..robust_cfg(ROBUST_LOSSES[0])
            })
            .map(drop)
        }),
        ("quantile", &|decay| {
            Robust::new(RobustCfg {
                decay,
                ..robust_cfg(ROBUST_LOSSES[1])
            })
            .map(drop)
        }),
        ("ftrl", &|decay| {
            Ftrl::new(FtrlCfg {
                decay,
                ..ftrl_cfg()
            })
            .map(drop)
        }),
        ("sgd", &|decay| {
            Sgd::new(SgdCfg { decay, ..sgd_cfg() }).map(drop)
        }),
        ("pa", &|decay| {
            Pa::new(PaCfg { decay, ..pa_cfg() }).map(drop)
        }),
        ("ew_cov", &|decay| {
            EwCovModel::new(EwCovCfg {
                decay,
                ..ew_cov_model_cfg()
            })
            .map(drop)
        }),
        ("kmeans", &|decay| {
            KMeans::new(KMeansCfg {
                decay,
                ..kmeans_cfg()
            })
            .map(drop)
        }),
        ("micro", &|decay| {
            Micro::new(MicroCfg {
                decay,
                ..micro_cfg()
            })
            .map(drop)
        }),
        ("ew_class", &|decay| {
            EwClass::new(EwClassCfg {
                decay,
                ..ew_class_cfg()
            })
            .map(drop)
        }),
        ("marginal", &|decay| {
            Marginal::new(MarginalCfg {
                decay,
                ..marginal_cfg()
            })
            .map(drop)
        }),
        ("deco", &|decay| {
            Deco::new(DecoCfg {
                decay,
                ..deco_cfg()
            })
            .map(drop)
        }),
        ("hmm", &|decay| {
            Hmm::new(HmmCfg { decay, ..hmm_cfg() }).map(drop)
        }),
        ("corrchange", &|decay| {
            CorrChange::new(CorrChangeCfg {
                decay,
                ..corrchange_cfg()
            })
            .map(drop)
        }),
    ];
    let bad = [
        Decay::Halflife(0.0),
        Decay::Halflife(-1.0),
        Decay::Halflife(f64::NAN),
        Decay::Lam(1.5),
        Decay::Lam(0.0),
        Decay::Lam(-0.5),
        Decay::Lam(f64::NAN),
    ];
    for (name, build) in builds {
        for decay in bad {
            let e = build(decay).expect_err(&format!("{name}: {decay:?} was accepted"));
            let says = match decay {
                Decay::Halflife(_) => "half_life must be > 0",
                Decay::Lam(_) => "lam must be in (0, 1]",
            };
            assert!(
                e.starts_with(&format!("{name}: {says}")),
                "{name}, {decay:?}: {e}"
            );
        }
        for decay in [
            Decay::Halflife(f64::INFINITY),
            Decay::Halflife(HALFLIFE),
            Decay::Lam(1.0),
            Decay::Lam(0.97),
        ] {
            build(decay).unwrap_or_else(|e| panic!("{name}, {decay:?}: {e}"));
        }
    }
}

/// `model.<variant>.cfg.<key>` of a state's msgpack, set to `value`.
fn set_cfg(v: &mut rmpv::Value, key: &str, value: rmpv::Value) {
    fn entry<'a>(v: &'a mut rmpv::Value, key: &str) -> &'a mut rmpv::Value {
        let rmpv::Value::Map(entries) = v else {
            panic!("a map holds {key}")
        };
        &mut entries
            .iter_mut()
            .find(|(k, _)| k.as_str() == Some(key))
            .unwrap_or_else(|| panic!("no {key}"))
            .1
    }
    let rmpv::Value::Map(variant) = entry(v, "model") else {
        panic!("a model")
    };
    *entry(entry(&mut variant[0].1, "cfg"), key) = value;
}

/// Every model's `restore` holds the configuration its state carries to
/// what its `new` holds a fresh one to: no `restore` ran `validate`, so a
/// state whose cfg `new` refuses -- `kmeans` with `k = 0`, `micro` with
/// `max_clusters = 0`, a half-life of 0 -- loaded and failed at a later
/// row, as a panic or a NaN for good (review 2026-10-06, CF4, CA6, CD14).
/// Each model's state with one cfg field set, in the named msgpack a bank
/// writes, to a value its `validate` refuses: a half-life of 0 for the
/// seventeen with a `Decay`, a field of its own for the four without.
#[test]
fn every_model_refuses_at_restore_a_configuration_its_new_refuses() {
    fn refused<M: OnlineModel>(m: M, key: &str, value: rmpv::Value, says: &str) -> String {
        let state = m.state();
        let kind = state.model.kind();
        let damaged = edit_state(&state, |v| set_cfg(v, key, value));
        match M::restore(&damaged) {
            Err(StateError::Invalid(e)) => {
                // `robust`'s messages name the spec's model, `huber` or
                // `quantile` by its loss (docs/PLAN.md task 196, N1).
                let named =
                    |k: &str| e.starts_with(&format!("{k}: the state's configuration is refused"));
                let head =
                    named(kind) || (kind == "robust" && (named("huber") || named("quantile")));
                assert!(head && e.contains(says), "{kind}: {e}");
            }
            Err(e) => panic!("{kind}: {e}"),
            Ok(_) => panic!("{kind}: a {key} its `new` refuses was restored"),
        }
        assert!(M::restore(&state).is_ok(), "{kind}: the state as saved");
        kind.to_string()
    }
    let zero = || rmpv::Value::Map(vec![("Halflife".into(), 0.0.into())]);
    let h = "half_life must be > 0";
    let kinds = [
        refused(EwRidge::new(ewridge_cfg()).unwrap(), "decay", zero(), h),
        refused(Rls::new(rls_cfg()).unwrap(), "decay", zero(), h),
        refused(Lasso::new(lasso_cfg()).unwrap(), "decay", zero(), h),
        refused(Kalman::new(kalman_cfg()).unwrap(), "decay", zero(), h),
        refused(
            Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(
            Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(Ftrl::new(ftrl_cfg()).unwrap(), "decay", zero(), h),
        refused(Sgd::new(sgd_cfg()).unwrap(), "decay", zero(), h),
        refused(Pa::new(pa_cfg()).unwrap(), "decay", zero(), h),
        refused(
            Holt::new(holt_cfg()).unwrap(),
            "level_half_life",
            0.0.into(),
            "level_half_life must be > 0",
        ),
        refused(
            EwCovModel::new(ew_cov_model_cfg()).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(KMeans::new(kmeans_cfg()).unwrap(), "decay", zero(), h),
        refused(Micro::new(micro_cfg()).unwrap(), "decay", zero(), h),
        refused(EwClass::new(ew_class_cfg()).unwrap(), "decay", zero(), h),
        refused(
            SeqTest::new(seqtest_cfg()).unwrap(),
            "min_weight",
            f64::NAN.into(),
            "min_weight must be >= 0",
        ),
        refused(Marginal::new(marginal_cfg()).unwrap(), "decay", zero(), h),
        refused(Deco::new(deco_cfg()).unwrap(), "decay", zero(), h),
        refused(
            Rcov::new(rcov_cfg()).unwrap(),
            "theta",
            0.0.into(),
            "theta must be finite and > 0",
        ),
        refused(Hmm::new(hmm_cfg()).unwrap(), "decay", zero(), h),
        refused(
            CorrChange::new(corrchange_cfg()).unwrap(),
            "decay",
            zero(),
            h,
        ),
        refused(
            Bocpd::new(bocpd_cfg()).unwrap(),
            "hazard",
            0.5.into(),
            "must be finite and > 1",
        ),
    ];
    // Every model's `restore`: the twenty kinds, `robust` under both losses.
    let mut seen: Vec<&str> = kinds.iter().map(String::as_str).collect();
    seen.dedup();
    let models = PROBED.len() - 1;
    assert_eq!(seen.len(), models, "{seen:?}");
}

/// Each core `validate` refuses what the spec layer refuses (`spec.rs`),
/// naming the model, the parameter and the value, so the Rust API and a
/// state file are held to the rule a spec is. Each of these built a model
/// that learned nothing, predicted 0 or poisoned its state with a NaN that
/// never washes out, with no error: a NaN or negative `ridge`, a `tol` that
/// is not a positive number, a `select_half_life` of 0 (the selection
/// weight `2^(-0/0)`), a path point that is not a number, an infinite `rls`
/// ridge, a NaN `min_weight`, a `quantile_eps` of NaN or infinity, an
/// infinite `p0`, `obs_var` or `q` (review 2026-10-06, CA6 and CC7). The
/// legal neighbours -- `inf` where it means something -- still build.
#[test]
fn every_core_validate_refuses_what_the_spec_refuses() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    let ewridge = |c: EwRidgeCfg| EwRidge::new(c).map(drop);
    let lasso = |c: LassoCfg| Lasso::new(c).map(drop);
    let rls = |c: RlsCfg| Rls::new(c).map(drop);
    let robust = |c: RobustCfg| Robust::new(c).map(drop);
    let kalman = |c: KalmanCfg| Kalman::new(c).map(drop);
    let huber = || robust_cfg(ROBUST_LOSSES[0]);
    let quantile = || robust_cfg(ROBUST_LOSSES[1]);
    let refused: Vec<(Result<(), String>, &str)> = vec![
        (
            ewridge(EwRidgeCfg {
                ridge: vec![1e-6, nan],
                ..ewridge_cfg()
            }),
            "ewridge: ridge must be finite and >= 0, got NaN",
        ),
        (
            ewridge(EwRidgeCfg {
                ridge: vec![-1.0],
                ..ewridge_cfg()
            }),
            "ewridge: ridge must be finite and >= 0, got -1",
        ),
        (
            ewridge(EwRidgeCfg {
                ridge: vec![inf],
                ..ewridge_cfg()
            }),
            "ewridge: ridge must be finite and >= 0, got inf",
        ),
        (
            ewridge(EwRidgeCfg {
                min_weight: nan,
                ..ewridge_cfg()
            }),
            "ewridge: min_weight must be >= 0, got NaN",
        ),
        (
            lasso(LassoCfg {
                tol: nan,
                ..lasso_cfg()
            }),
            "lasso: tol must be finite and > 0, got NaN",
        ),
        (
            lasso(LassoCfg {
                tol: 0.0,
                ..lasso_cfg()
            }),
            "lasso: tol must be finite and > 0, got 0",
        ),
        (
            lasso(LassoCfg {
                tol: inf,
                ..lasso_cfg()
            }),
            "lasso: tol must be finite and > 0, got inf",
        ),
        (
            lasso(LassoCfg {
                select_half_life: Some(0.0),
                ..lasso_cfg()
            }),
            "lasso: select_half_life must be > 0, got 0",
        ),
        (
            lasso(LassoCfg {
                select_half_life: Some(nan),
                ..lasso_cfg()
            }),
            "lasso: select_half_life must be > 0, got NaN",
        ),
        (
            lasso(LassoCfg {
                lasso_path: vec![nan],
                ..lasso_cfg()
            }),
            "lasso: lasso_path values must be finite and >= 0, got NaN",
        ),
        (
            lasso(LassoCfg {
                lasso_path: vec![inf, 0.1],
                ..lasso_cfg()
            }),
            "lasso: lasso_path values must be finite and >= 0, got inf",
        ),
        (
            lasso(LassoCfg {
                min_weight: nan,
                ..lasso_cfg()
            }),
            "lasso: min_weight must be >= 0, got NaN",
        ),
        (
            rls(RlsCfg {
                delta: inf,
                ..rls_cfg()
            }),
            "rls: delta must be finite and > 0 (it sets P0 = I / delta), got inf",
        ),
        (
            rls(RlsCfg {
                delta: nan,
                ..rls_cfg()
            }),
            "rls: delta must be finite and > 0 (it sets P0 = I / delta), got NaN",
        ),
        (
            rls(RlsCfg {
                min_weight: nan,
                ..rls_cfg()
            }),
            "rls: min_weight must be >= 0, got NaN",
        ),
        // Review round 5 (C6): a NaN `min_weight` loaded through the Rust
        // API for the three models that did not check it, and a NaN
        // `solve_every` for the three that solve on a schedule.
        (
            Sgd::new(SgdCfg {
                min_weight: nan,
                ..sgd_cfg()
            })
            .map(drop),
            "sgd: min_weight must be >= 0, got NaN",
        ),
        (
            Sgd::new(SgdCfg {
                min_weight: -1.0,
                ..sgd_cfg()
            })
            .map(drop),
            "sgd: min_weight must be >= 0, got -1",
        ),
        (
            Holt::new(HoltCfg {
                min_weight: nan,
                ..holt_cfg()
            })
            .map(drop),
            "holt: min_weight must be >= 0, got NaN",
        ),
        (
            EwCovModel::new(EwCovCfg {
                min_weight: nan,
                ..ew_cov_model_cfg()
            })
            .map(drop),
            "ew_cov: min_weight must be >= 0, got NaN",
        ),
        (
            ewridge(EwRidgeCfg {
                solve_every: nan,
                ..ewridge_cfg()
            }),
            "ewridge: solve_every must not be NaN",
        ),
        (
            lasso(LassoCfg {
                solve_every: nan,
                ..lasso_cfg()
            }),
            "lasso: solve_every must not be NaN",
        ),
        (
            robust(RobustCfg {
                solve_every: nan,
                ..quantile()
            }),
            "quantile: solve_every must not be NaN",
        ),
        (
            robust(RobustCfg {
                ridge: nan,
                ..huber()
            }),
            "huber: ridge must be finite and >= 0, got NaN",
        ),
        (
            robust(RobustCfg {
                ridge: inf,
                ..quantile()
            }),
            "quantile: ridge must be finite and >= 0, got inf",
        ),
        (
            robust(RobustCfg {
                quantile_eps: nan,
                ..quantile()
            }),
            "quantile: quantile_eps must be finite and > 0, got NaN",
        ),
        (
            robust(RobustCfg {
                quantile_eps: inf,
                ..quantile()
            }),
            "quantile: quantile_eps must be finite and > 0, got inf",
        ),
        (
            robust(RobustCfg {
                min_weight: nan,
                ..huber()
            }),
            "huber: min_weight must be >= 0, got NaN",
        ),
        (
            robust(RobustCfg {
                loss: RobustLoss::Huber { delta: nan },
                ..huber()
            }),
            "huber: huber_delta must be > 0, got NaN",
        ),
        (
            robust(RobustCfg {
                loss: RobustLoss::Quantile { tau: 1.5 },
                ..quantile()
            }),
            "quantile: quantile must be in (0, 1), got 1.5",
        ),
        (
            kalman(KalmanCfg {
                p0: inf,
                ..kalman_cfg()
            }),
            "kalman: p0 must be finite and > 0, got inf",
        ),
        (
            kalman(KalmanCfg {
                obs_var: Some(inf),
                ..kalman_cfg()
            }),
            "kalman: obs_var must be finite and > 0, got inf",
        ),
        (
            kalman(KalmanCfg {
                q: Some(vec![0.0, inf, 0.0]),
                ..kalman_cfg()
            }),
            "kalman: q values must be finite and >= 0, got inf",
        ),
        (
            kalman(KalmanCfg {
                min_weight: nan,
                ..kalman_cfg()
            }),
            "kalman: min_weight must be >= 0, got NaN",
        ),
        (
            kalman(KalmanCfg {
                half_life: vec![nan],
                ..kalman_cfg()
            }),
            "kalman: half_life values must be > 0 (inf pins), got NaN",
        ),
        (
            kalman(KalmanCfg {
                revert_half_life: vec![0.0],
                ..kalman_cfg()
            }),
            "kalman: revert_half_life values must be > 0 (inf = random walk), got 0",
        ),
    ];
    let wrong: Vec<String> = refused
        .into_iter()
        .enumerate()
        .filter_map(|(i, (got, msg))| match got {
            Ok(()) => Some(format!("case {i} accepted, wanted: {msg}")),
            Err(e) if !e.contains(msg) => Some(format!("case {i} refused as: {e}")),
            Err(_) => None,
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    let accepted: Vec<Result<(), String>> = vec![
        ewridge(EwRidgeCfg {
            ridge: vec![0.0],
            min_weight: inf,
            ..ewridge_cfg()
        }),
        lasso(LassoCfg {
            select_half_life: Some(inf),
            lasso_path: vec![0.0],
            min_weight: inf,
            ..lasso_cfg()
        }),
        rls(RlsCfg {
            delta: 1e300,
            min_weight: inf,
            ..rls_cfg()
        }),
        robust(RobustCfg {
            ridge: 0.0,
            loss: RobustLoss::Huber { delta: inf },
            min_weight: inf,
            ..huber()
        }),
        robust(RobustCfg {
            quantile_eps: 1e-9,
            ..quantile()
        }),
        kalman(KalmanCfg {
            q: Some(vec![0.0; 3]),
            obs_var: Some(1e300),
            half_life: vec![inf],
            min_weight: inf,
            ..kalman_cfg()
        }),
        // An infinite `min_weight` never predicts, and an infinite
        // `solve_every` leaves the cadence to the row cap: both legal, as
        // the models above take them (C6).
        Sgd::new(SgdCfg {
            min_weight: inf,
            ..sgd_cfg()
        })
        .map(drop),
        Holt::new(HoltCfg {
            min_weight: inf,
            ..holt_cfg()
        })
        .map(drop),
        EwCovModel::new(EwCovCfg {
            min_weight: inf,
            ..ew_cov_model_cfg()
        })
        .map(drop),
        ewridge(EwRidgeCfg {
            solve_every: inf,
            ..ewridge_cfg()
        }),
        lasso(LassoCfg {
            solve_every: inf,
            ..lasso_cfg()
        }),
        robust(RobustCfg {
            solve_every: inf,
            ..huber()
        }),
    ];
    for (i, got) in accepted.into_iter().enumerate() {
        got.unwrap_or_else(|e| panic!("legal neighbour {i}: {e}"));
    }
}

/// Exactly the models that solve on a schedule report a solve share --
/// `ewridge`, `lasso`, and `robust` under both losses -- each the share it
/// is given; every other model keeps the trait's `None` whatever it is set
/// to (docs/EXTENDING.md's table of hooks; review 2026-10-05, CF7). The
/// stream sets the spec's share on every model it builds or restores, so
/// these are the models whose cadence it reaches.
#[test]
fn exactly_the_scheduled_solvers_report_a_solve_share() {
    fn share<M: OnlineModel>(mut m: M) -> Option<f64> {
        m.set_solve_share(Some(0.3));
        m.solve_share()
    }
    let shares = [
        ("ewridge", share(EwRidge::new(ewridge_cfg()).unwrap())),
        ("rls", share(Rls::new(rls_cfg()).unwrap())),
        ("lasso", share(Lasso::new(lasso_cfg()).unwrap())),
        ("kalman", share(Kalman::new(kalman_cfg()).unwrap())),
        (
            "huber",
            share(Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap()),
        ),
        (
            "quantile",
            share(Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap()),
        ),
        ("ftrl", share(Ftrl::new(ftrl_cfg()).unwrap())),
        ("sgd", share(Sgd::new(sgd_cfg()).unwrap())),
        ("pa", share(Pa::new(pa_cfg()).unwrap())),
        ("holt", share(Holt::new(holt_cfg()).unwrap())),
        (
            "ew_cov",
            share(EwCovModel::new(ew_cov_model_cfg()).unwrap()),
        ),
        ("kmeans", share(KMeans::new(kmeans_cfg()).unwrap())),
        ("micro", share(Micro::new(micro_cfg()).unwrap())),
        ("ew_class", share(EwClass::new(ew_class_cfg()).unwrap())),
        ("seqtest", share(SeqTest::new(seqtest_cfg()).unwrap())),
        ("marginal", share(Marginal::new(marginal_cfg()).unwrap())),
        ("deco", share(Deco::new(deco_cfg()).unwrap())),
        ("rcov", share(Rcov::new(rcov_cfg()).unwrap())),
        ("hmm", share(Hmm::new(hmm_cfg()).unwrap())),
        (
            "corrchange",
            share(CorrChange::new(corrchange_cfg()).unwrap()),
        ),
        ("bocpd", share(Bocpd::new(bocpd_cfg()).unwrap())),
    ];
    let reporting: Vec<&str> = shares
        .iter()
        .filter(|(_, s)| s.is_some())
        .map(|(name, _)| *name)
        .collect();
    assert_eq!(reporting, ["ewridge", "lasso", "huber", "quantile"]);
    for (name, s) in shares {
        assert!(s.is_none() || s == Some(0.3), "{name}: {s:?}");
    }
}

/// The variants of `ModelState` this file probes, each beside the kind its
/// states name (`ModelState::kind`), which names its probe: `fn
/// <kind>_predict_is_the_step()` above. A model added to the enum and not to
/// this list fails here, and so does an entry without its probe, which is
/// the reminder to write its `*_cfg()` and probe (docs/EXTENDING.md): the
/// list was held to the enum alone, so an entry with no probe passed
/// (review 2026-10-06, CF12). The bare accumulator is no model and has no
/// probe; `every_state_kind_is_distinct_and_named` reads its kind, and
/// every model's, off a real state and holds each pair here to it.
const PROBED: &[(&str, &str)] = &[
    ("EwCov", "ew_cov_accumulator"),
    ("EwRidge", "ewridge"),
    ("Rls", "rls"),
    ("Lasso", "lasso"),
    ("Kalman", "kalman"),
    ("Robust", "robust"),
    ("Ftrl", "ftrl"),
    ("EwCovModel", "ew_cov"),
    ("Sgd", "sgd"),
    ("Pa", "pa"),
    ("Holt", "holt"),
    ("KMeans", "kmeans"),
    ("Micro", "micro"),
    ("EwClass", "ew_class"),
    ("SeqTest", "seqtest"),
    ("Marginal", "marginal"),
    ("Deco", "deco"),
    ("Rcov", "rcov"),
    ("Hmm", "hmm"),
    ("CorrChange", "corrchange"),
    ("Bocpd", "bocpd"),
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
    let variants: Vec<&str> = PROBED.iter().map(|(variant, _)| *variant).collect();
    assert_eq!(
        &quoted[1..],
        variants,
        "a ModelState variant has no contract probe"
    );
    // Each model's probe is in this file, found by its kind.
    let src = include_str!("model_contract.rs");
    for (variant, kind) in PROBED {
        if *kind == "ew_cov_accumulator" {
            continue;
        }
        let probe = format!("fn {kind}_predict_is_the_step()");
        assert!(
            src.contains(&probe),
            "{variant}: add `{probe}` to this file"
        );
    }
}

#[test]
fn restoring_the_wrong_model_is_an_error_that_names_both() {
    let holt = Holt::new(HoltCfg {
        n_targets: 1,
        level_half_life: 10.0,
        trend_half_life: 40.0,
        min_weight: 0.0,
        trend: true,
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
    // 1500 half-lives. A row at the bound with weight at the bound leaves a
    // moment of 1e300 on the sum scale, which needs 1000 half-lives to fall
    // below 1e-6; the rest is margin (measured: ewridge agrees with its twin
    // to 3e-5 after 1000 half-lives and to rounding after 1100).
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
fn ewridge_recovers_from_bounded_extremes() {
    recovers_from_bounded_extremes(
        || EwRidge::new(ewridge_cfg()).unwrap(),
        2,
        Recovery::Twin(1e-9),
    );
    let mut cfg = ewridge_cfg();
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
    for standardize in [false, true] {
        let mut cfg = sgd_cfg();
        cfg.standardize = standardize;
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
    // The tail is 1500 half-lives of HALFLIFE: what a row at the bound with
    // weight at the bound needs to wash out of a mean-form accumulator. Since
    // the code review's S29 `holt`'s level and trend are weighted means, so a
    // row's weight counts, and the trend forgets such a row on its own
    // half-life -- four times the level's by default, which would want a tail
    // four times as long. So the probe runs both at HALFLIFE. (The textbook
    // form it replaced ignored the weight, and recovered on the level's rate.)
    let cfg = HoltCfg {
        trend_half_life: HALFLIFE,
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
    // decayed below `beta_mu` (332 half-lives) and is pruned; a row at the
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
                    ("weight_sum", pa.n_eff, pb.n_eff),
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
fn predict_is_the_step_without_the_step<M: OnlineModel + Clone>(
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
        let flagged_only = |slot: usize| ONLY_ON_A_FLAG.contains(&(kind, slot));
        if step
            .pred
            .iter()
            .enumerate()
            .all(|(slot, v)| v.is_finite() || flagged_only(slot))
        {
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
    a_zero_weight_row_past_the_underflow_forgets(&build, targets, binary, kind);
    unusable_values_are_refused(&build, targets, binary, kind);
}

/// Task 115 (c), decided 2026-09-28: a zero-weight row whose decay
/// underflows to exactly 0 -- `2^-1075` rounds to 0, so from 1075 half-lives
/// on -- forgets the history, as the decay does. It used to keep it: the
/// mean-form update is `0/0` there, the guard skipped the row whole, and the
/// next row saw the old count (PLAN §12). "As the decay does" is the same row
/// one half-life short, whose factor `2^-1074` is the smallest double: it
/// ages the history to nothing, and the two streams must agree from there
/// on. The row after the one that forgot reports no weight, unless the
/// model's weight does not decay at all.
///
/// Predictions are compared too, except where the reference is not "the
/// history aged to nothing": `kalman` adds `Q·d` to its covariance and
/// `holt` carries its level along its trend, so a half-life more of gap is a
/// different state for both; `ew_class` counts a class of subnormal weight
/// as present and one of weight 0 as absent, a threshold rather than an age;
/// and `hmm` one half-life short is not a sane state -- a subnormal history
/// share leaves its co-moments and precision prior a few bits, and its
/// densities NaN for 39 of the next 40 rows, which do not count towards its
/// `n_eff` either (from about 1025 half-lives; PLAN task 115 (c), measured
/// 2026-09-28 and raised, not fixed). `hmm` keeps the first check alone.
fn a_zero_weight_row_past_the_underflow_forgets<M: OnlineModel>(
    build: &impl Fn() -> M,
    targets: usize,
    binary: bool,
    kind: &'static str,
) {
    const NOT_COMPARED: [&str; 4] = ["kalman", "holt", "ew_class", "hmm"];
    let row = |s: &mut u64| {
        let x: Vec<f64> = (0..K).map(|_| lcg(s) * 3.0).collect();
        let y: Vec<Option<f64>> = (0..targets)
            .map(|j| {
                let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1] + 0.1 * lcg(s);
                Some(if binary { f64::from(lin > 0.5) } else { lin })
            })
            .collect();
        (x, y)
    };
    let run = |gap: f64| -> Vec<Step> {
        let mut m = build();
        let mut s = 20260928u64;
        for i in 0..40 {
            let (x, y) = row(&mut s);
            m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let (x, y) = row(&mut s);
        m.step(&x, &y, gap, 0.0);
        (0..40)
            .map(|_| {
                let (x, y) = row(&mut s);
                m.step(&x, &y, 1.0, 1.0)
            })
            .collect()
    };
    assert_eq!(decay().factor(1075.0 * HALFLIFE), 0.0);
    assert!(decay().factor(1074.0 * HALFLIFE) > 0.0);
    let forgot = run(1075.0 * HALFLIFE);
    let aged = run(1074.0 * HALFLIFE);
    assert!(
        forgot[0].n_eff == 0.0 || forgot[0].n_eff == aged[0].n_eff,
        "{kind}: the row after a zero-weight row 1075 half_lives on reports n_eff {}, a \
         history the decay forgot",
        forgot[0].n_eff
    );
    let close = |a: f64, b: f64| {
        a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * (1.0 + b.abs())
    };
    if kind == "hmm" {
        return;
    }
    for (i, (a, b)) in forgot.iter().zip(&aged).enumerate() {
        let preds = NOT_COMPARED.contains(&kind)
            || a.pred.len() == b.pred.len()
                && a.pred.iter().zip(&b.pred).all(|(p, q)| close(*p, *q));
        assert!(
            close(a.n_eff, b.n_eff) && preds,
            "{kind}: row {i} after the zero-weight row: n_eff {} and {:?} where its decay \
             underflowed, n_eff {} and {:?} one half_life short of that",
            a.n_eff,
            a.pred,
            b.n_eff,
            b.pred
        );
    }
}

/// The models whose fitted function a zero-weight row inside a gap moves,
/// found when [`zero_weight_rows_only_advance_the_clock`] began comparing
/// the numbers (docs/PLAN.md task 211) and raised there, not fixed: each is a
/// design of its own, not a slip. Measured on that test's stream.
const PARTS_ON_A_SPLIT_GAP: [(&str, &str); 2] = [
    (
        "ftrl",
        "the penalties' scale `W/W*` reads `W*` on a clock that runs only on the rows \
         that teach the target, each such row aging it by its own delta: a zero-weight row \
         takes its delta off that clock, and the fit at a fixed row moved by 1.9% \
         (-0.027757 against -0.027249) on the row after the first one",
    ),
    (
        "hmm",
        "a row reads its posterior before its own decay, from states whose absolute \
         precision prior weighs more as their weight ages, and learns from that posterior: \
         a zero-weight row ages the states before the next row reads them, and the fit at a \
         fixed row moved by 1.2e-5 (0.997912 against 0.997900) on the row after the first \
         one",
    ),
];

/// A fit solved on a clock schedule (`ewridge`'s `solve_every`), which a
/// zero-weight row at the clock a solve is due triggers before the next row's
/// data where the stream without it solves after them: the reported fit
/// differs until the next solve, by 1.8e-4 on the rows between, though both
/// hold the same statistics (docs/PLAN.md task 211, raised).
fn solves_on_a_clock<M: OnlineModel>(m: &M) -> bool {
    match m.state().model {
        ModelState::EwRidge(e) => e.cfg().solve_every > 0.0,
        _ => false,
    }
}

/// Hard rules 8 and 9, together and without naming a decay: a zero-weight
/// row advances the clock and teaches nothing, so the `n_eff` a stream
/// reports after one must equal the `n_eff` of the same stream with that row
/// *left out* and its clock delta carried into the next row. For an
/// exponential decay `lam(a)·lam(b) = lam(a + b)`, so the two agree exactly
/// whatever the half-life -- and a model that forgets to decay `n_eff` on the
/// row it learns nothing from does not (docs/REVIEW-E54-E64.md H3/C2).
///
/// And every number the real rows report agrees too, to rounding (within
/// 1e-9 of `1 + |value|`): each prediction, the fitted function -- what the
/// model predicts at two fixed rows after each real row, which for a linear
/// fit is its coefficients -- and the coefficients' variances where a model
/// keeps them. `kalman` charged its process noise per row, `Q d²`, which is
/// not additive over a split gap, and moved every prediction after a
/// zero-weight row by up to 0.197 at `coef_half_life` 20 while this test,
/// which compared `n_eff` and which predictions were null, passed
/// (docs/PLAN.md task 211).
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
            _ if i.is_multiple_of(5) => 3.0,
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
        // And the numbers themselves (docs/PLAN.md task 211): the
        // predictions on the rows `n_eff` is compared on, and after every
        // real row the fitted function at two fixed rows and the
        // coefficients' variances. On the row that carries a skipped row's
        // delta the prediction may differ by construction, as `n_eff` does:
        // the stream with the zero-weight row has aged its moments by that
        // row's delta before predicting, and a fit solved after that, with
        // a penalty that does not age with them, is a fit solved after the
        // decay; the other ages them inside the row. The models whose fit a
        // zero-weight row inside a gap moves, each a decision of its own,
        // are named in `PARTS_ON_A_SPLIT_GAP`.
        if PARTS_ON_A_SPLIT_GAP.iter().any(|(k, _)| *k == kind) || solves_on_a_clock(&with) {
            continue;
        }
        let probes: [Vec<f64>; 2] = [vec![0.7; K], (0..K).map(|f| f as f64 - 0.4).collect()];
        let mut theirs = Vec::new();
        if compare {
            theirs.push((String::from("pred"), a.pred.clone(), b.pred.clone()));
        }
        for (n, x) in probes.iter().enumerate() {
            theirs.push((
                format!("the fit at probe {n}"),
                with.predict(x, 0.0).pred,
                without.predict(x, 0.0).pred,
            ));
        }
        let variances = |m: &M| match m.coef_variance() {
            Some(CoefVariance::PerNoise(v) | CoefVariance::Absolute(v)) => {
                v.into_iter().flatten().collect()
            }
            None => vec![],
        };
        theirs.push((
            String::from("coef variance"),
            variances(&with),
            variances(&without),
        ));
        for (what, u, v) in theirs {
            assert_eq!(u.len(), v.len(), "{kind}: row {i}: {what}");
            for (slot, (p, q)) in u.iter().zip(&v).enumerate() {
                assert!(
                    !(p.is_finite() && q.is_finite()) || (p - q).abs() <= 1e-9 * (1.0 + q.abs()),
                    "{kind}: row {i}: {what}, slot {slot}: {p} in the stream with zero-weight \
                     rows, {q} in the one without them -- a zero-weight row must advance the \
                     clock and teach nothing"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A value that is not usable, refused by every model by one rule (the
// `OnlineModel` docs; docs/PLAN.md task 183).
// ---------------------------------------------------------------------------

/// The values a model reads as missing (`online_core::usable`): not a
/// number, either infinity, and either sign past the input bound.
const UNUSABLE: [f64; 5] = [
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    2.0 * INPUT_BOUND,
    -2.0 * INPUT_BOUND,
];

/// A row: the features, the targets and the clock since the row before.
type Row3 = (Vec<f64>, Vec<Option<f64>>, f64);

fn state_bytes<M: OnlineModel>(m: &M) -> Vec<u8> {
    rmp_serde::to_vec(&m.state()).unwrap()
}

/// The same numbers to the bit, NaN for NaN.
fn same_number(u: f64, v: f64) -> bool {
    u.to_bits() == v.to_bits() || (u.is_nan() && v.is_nan())
}

/// Two steps that report the same thing to the bit, NaN for NaN: every
/// slot, `n_eff` and `extra`.
fn same_bits(a: &Step, b: &Step) -> bool {
    let extra = match (&a.extra, &b.extra) {
        (Some(Extra::Lasso { lam_selected: u }), Some(Extra::Lasso { lam_selected: v })) => {
            u.len() == v.len() && u.iter().zip(v).all(|(p, q)| same_number(*p, *q))
        }
        (u, v) => u == v,
    };
    a.pred.len() == b.pred.len()
        && a.pred.iter().zip(&b.pred).all(|(u, v)| same_number(*u, *v))
        && same_number(a.n_eff, b.n_eff)
        && extra
}

/// Nothing reported: NaN in every slot, and in every number of `extra`.
fn reports_nothing(s: &Step) -> bool {
    s.pred.iter().all(|v| v.is_nan())
        && match &s.extra {
            None => true,
            Some(Extra::Lasso { lam_selected }) => lam_selected.iter().all(|v| v.is_nan()),
            Some(_) => false,
        }
}

/// The rule for a value that is not usable, checked on `m` as it stands
/// before `row`, with `after` the rows that follow it:
///
/// - **A feature**, in each position, as each of [`UNUSABLE`]: `step`,
///   `predict` and `predict_with` report nothing, and `n_eff` as on any row
///   (the accessor's, where `n_eff_of` is given); the state is the one the
///   same row leaves at weight 0 with that feature as it was -- byte for
///   byte, and so the same whatever the value and wherever it stands -- and
///   the rows after it report, to the bit, what they report after that row.
/// - **A target** is an absent one: `Some(bad)` and `None` leave the same
///   state and report the same step, `predict_with`'s included.
/// - **A weight**: the row reports what `predict_with` does, and leaves the
///   state a row with a feature that is not usable leaves.
///
/// No model is excepted: no model's zero-weight row takes anything from the
/// row's features into its state -- no ring slot, warm-up row or likelihood
/// -- so the refused row, which takes none of it, leaves that row's state.
/// A blocked `ewridge` Gram, which holds a zero-weight row in its block, is
/// the one place one does, and is tested in `ewridge`'s own file.
fn refuses_unusable_values<M: OnlineModel + Clone>(
    m: &M,
    (x, y, d): &Row3,
    after: &[Row3],
    kind: &str,
    n_eff_of: Option<&dyn Fn(&M) -> f64>,
) {
    let d = *d;
    // The same row at weight 0, and the rows after it from there.
    let mut reference = m.clone();
    let zero = reference.step(x, y, d, 0.0);
    let zero_state = state_bytes(&reference);
    let goes_on_as_the_reference = |mut a: M, what: &str| {
        let mut b = reference.clone();
        for (i, (x, y, d)) in after.iter().enumerate() {
            let (sa, sb) = (a.step(x, y, *d, 1.0), b.step(x, y, *d, 1.0));
            assert!(
                same_bits(&sa, &sb),
                "{kind}: {what}: row {i} after it reports {sa:?}, and {sb:?} after the row at \
                 weight 0"
            );
        }
    };

    let mut refused_state: Option<Vec<u8>> = None;
    for pos in 0..m.n_features() {
        for bad in UNUSABLE {
            let what = format!("feature {pos} = {bad:e}");
            let mut xb = x.clone();
            xb[pos] = bad;
            let mut c = m.clone();
            let s = c.step(&xb, y, d, 1.0);
            assert!(
                reports_nothing(&s),
                "{kind}: {what}: the step reported {s:?}"
            );
            for (how, p) in [
                ("predict", m.predict(&xb, d)),
                ("predict_with", m.predict_with(&xb, y, d)),
            ] {
                assert!(
                    same_bits(&p, &s),
                    "{kind}: {what}: {how} reported {p:?}, the step {s:?}"
                );
            }
            assert!(
                same_number(s.n_eff, zero.n_eff),
                "{kind}: {what}: n_eff {}, and {} on any row",
                s.n_eff,
                zero.n_eff
            );
            if let Some(n_eff) = n_eff_of {
                assert_eq!(
                    s.n_eff,
                    n_eff(m),
                    "{kind}: {what}: n_eff against the accessor"
                );
            }
            let state = state_bytes(&c);
            assert!(
                state == zero_state,
                "{kind}: {what}: the state is not the one the row leaves at weight 0"
            );
            match &refused_state {
                Some(first) => assert!(
                    *first == state,
                    "{kind}: {what}: the state depends on the value or where it stands"
                ),
                None => refused_state = Some(state),
            }
            goes_on_as_the_reference(c, &what);
        }
    }

    for j in 0..y.len() {
        for bad in UNUSABLE {
            let what = format!("target {j} = Some({bad:e})");
            let (mut yb, mut yn) = (y.clone(), y.clone());
            yb[j] = Some(bad);
            yn[j] = None;
            let (pb, pn) = (m.predict_with(x, &yb, d), m.predict_with(x, &yn, d));
            assert!(
                same_bits(&pb, &pn),
                "{kind}: {what}: predict_with reported {pb:?}, and {pn:?} for None"
            );
            let (mut b, mut n) = (m.clone(), m.clone());
            let (sb, sn) = (b.step(x, &yb, d, 1.0), n.step(x, &yn, d, 1.0));
            assert!(
                same_bits(&sb, &sn),
                "{kind}: {what}: the step reported {sb:?}, and {sn:?} for None"
            );
            assert!(
                state_bytes(&b) == state_bytes(&n),
                "{kind}: {what}: the state is not the one None leaves"
            );
        }
    }

    // A model that reads no features is held to the row at weight 0.
    let refused_state = refused_state.unwrap_or(zero_state);
    for bad in UNUSABLE {
        let what = format!("weight {bad:e}");
        let p = m.predict_with(x, y, d);
        let mut c = m.clone();
        let s = c.step(x, y, d, bad);
        assert!(
            same_bits(&p, &s),
            "{kind}: {what}: the step reported {s:?}, predict_with {p:?}"
        );
        assert!(
            state_bytes(&c) == refused_state,
            "{kind}: {what}: the state is not the one a row with a feature that is not \
             usable leaves"
        );
        goes_on_as_the_reference(c, &what);
    }
}

/// [`refuses_unusable_values`] at the head of a stream, at its fourth row
/// (inside every warm-up the configurations here keep) and on a warm model
/// at row 60: the stream of [`zero_weight_rows_only_advance_the_clock`],
/// with twelve rows after each checked one.
fn unusable_values_are_refused<M: OnlineModel + Clone>(
    build: &impl Fn() -> M,
    targets: usize,
    binary: bool,
    kind: &'static str,
) {
    let mut s = 20261006u64;
    let rows: Vec<Row3> = (0..80usize)
        .map(|i| {
            let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
            let y = (0..targets)
                .map(|j| {
                    let lin = 0.5 * (j as f64 + 1.0) + x[0] - 0.5 * x[1];
                    Some(if binary { f64::from(lin > 0.5) } else { lin })
                })
                .collect();
            let d = match i {
                0 => 0.0,
                _ if i.is_multiple_of(5) => 3.0,
                _ => 1.0,
            };
            (x, y, d)
        })
        .collect();
    let mut m = build();
    for (i, (x, y, d)) in rows.iter().enumerate() {
        if matches!(i, 0 | 3 | 60) {
            refuses_unusable_values(&m, &rows[i], &rows[i + 1..i + 13], kind, None);
        }
        m.step(x, y, *d, 1.0);
    }
}

#[test]
fn ewridge_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| EwRidge::new(ewridge_cfg()).unwrap(), 2, false);
    let mut cfg = ewridge_cfg();
    cfg.standardize = true;
    cfg.session_shrink = Some(0.5);
    cfg.long_half_life = Some(4.0 * HALFLIFE);
    predict_is_the_step_without_the_step(move || EwRidge::new(cfg.clone()).unwrap(), 2, false);
    // A lazily refreshed solve: both read the cached coefficients.
    let mut cfg = ewridge_cfg();
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
    // With a selection half-life `lam_selected` moves; `extra` must match too.
    let mut cfg = lasso_cfg();
    cfg.lasso_path = vec![1.0, 0.1, 0.01, 0.0];
    cfg.select_half_life = Some(HALFLIFE);
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
    for standardize in [false, true] {
        let mut cfg = sgd_cfg();
        cfg.standardize = standardize;
        predict_is_the_step_without_the_step(move || Sgd::new(cfg.clone()).unwrap(), 2, false);
    }
    let mut cfg = sgd_cfg();
    cfg.loss = SgdLoss::Logistic;
    predict_is_the_step_without_the_step(move || Sgd::new(cfg.clone()).unwrap(), 2, true);
}

#[test]
fn pa_predict_is_the_step() {
    // Standardized too, the default since task 195: before the scaler's
    // warm-up ends and after it, when the fit is held in the caller's units
    // (docs/PLAN.md task 206).
    for standardize in [false, true] {
        let mut cfg = pa_cfg();
        cfg.standardize = standardize;
        predict_is_the_step_without_the_step(move || Pa::new(cfg.clone()).unwrap(), 2, false);
    }
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
        update_every_rows: 7,
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

/// [`bocpd_cfg`] with its prior left to the first rows (docs/PLAN.md task
/// 195, U4): the warm-up holds `d + 2` learned rows, reporting nothing, then
/// reads them as rows of the prior they set.
fn bocpd_from_the_data_cfg() -> BocpdCfg {
    BocpdCfg {
        prior_mean: None,
        prior_scale: None,
        prior_nu: None,
        ..bocpd_cfg()
    }
}

/// Inside the warm-up `predict` reports nothing, as the step does; after
/// it, the step's number.
#[test]
fn bocpd_from_the_data_predict_is_the_step() {
    predict_is_the_step_without_the_step(
        || Bocpd::new(bocpd_from_the_data_cfg()).unwrap(),
        0,
        true,
    );
}

/// On the clock `predict` applies the step's chance before it reads the
/// row, as the step does; and the zero-weight contracts this runs hold a
/// row of weight 0 to the stream without it, its step carried into the
/// next row, which on the clock is a transition that composes.
#[test]
fn bocpd_on_the_clock_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| Bocpd::new(bocpd_on_the_clock_cfg()).unwrap(), 0, true);
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
    // A hazard between 2 and 1000, never the configured 50; an exogenous
    // value swinging either side of the default 0.
    let (mut hazards, mut exogenous) = (Vec::new(), Vec::new());
    for i in 0..300 {
        let x: Vec<f64> = (0..K).map(|_| lcg(&mut s) * 3.0).collect();
        let hz = vec![Some(2.0 + 499.0 * (1.0 + lcg(&mut s)))];
        let z = vec![Some(3.0 * lcg(&mut s))];
        let d = if i == 0 { 0.0 } else { 1.0 };
        hazards.push((x.clone(), hz, d));
        exogenous.push((x, z, d));
    }
    for i in 0..300 {
        let ((x, hz, d), (_, z, _)) = (&hazards[i], &exogenous[i]);
        let d = *d;
        // A value there that is not usable is an absent one, as a target
        // that is not usable is (task 183): the configured hazard, the
        // default exogenous value. At the head of the stream, and warm.
        if matches!(i, 0 | 150) {
            let next = i + 1..i + 13;
            refuses_unusable_values(&bo, &hazards[i], &hazards[next.clone()], "bocpd", None);
            refuses_unusable_values(&hm, &exogenous[i], &exogenous[next], "hmm", None);
        }

        let p = bo.predict_with(x, hz, d);
        bo_moved += usize::from(!same_pred(&p, &bo.predict(x, d)));
        same_step("bocpd", i, &p, &bo.step(x, hz, d, 1.0));

        let p = hm.predict_with(x, z, d);
        hm_moved += usize::from(!same_pred(&p, &hm.predict(x, d)));
        same_step("hmm", i, &p, &hm.step(x, z, d, 1.0));
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

/// A model's state at row 80 with every low part removed -- a state written
/// before them -- read back: the number of low parts the state had, and
/// whether the model refused it, by the decoding or by its restore.
fn refuses_without_low_parts<M: OnlineModel>(make: impl Fn() -> M, rows: &[Row]) -> (usize, bool) {
    let mut m = make();
    for (i, r) in rows.iter().enumerate().take(80) {
        m.step(&r.x, &r.y, if i == 0 { 0.0 } else { 1.0 }, r.w);
    }
    let state = m.state();
    let bytes = rmp_serde::to_vec_named(&state).unwrap();
    let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    let stripped = strip_low_parts(&mut v);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    let refused = match rmp_serde::from_slice::<State>(&out) {
        Err(_) => true,
        Ok(old) => M::restore(&old).is_err(),
    };
    (stripped, refused)
}

/// Every model refuses a state without the means' low parts, where it
/// loaded one and went on with each part at zero (docs/PLAN.md task 110):
/// such a state is of a layout before this build's schema, which the
/// version refuses (docs/PLAN.md task 198), and what the repair reached
/// was a damaged file. `tests/state_repairs.rs` damages each low part
/// alone, a vector at a time.
#[test]
fn every_model_refuses_a_state_without_the_low_parts() {
    let two = leveled(2, false);
    let one = leveled(1, false);
    let label = leveled(1, true);
    let label2 = leveled(2, true);
    let none = leveled(0, false);
    let results: Vec<(&str, (usize, bool))> = vec![
        (
            "ewridge",
            refuses_without_low_parts(|| EwRidge::new(ewridge_cfg()).unwrap(), &two),
        ),
        (
            "rls",
            refuses_without_low_parts(|| Rls::new(rls_cfg()).unwrap(), &two),
        ),
        (
            "lasso",
            refuses_without_low_parts(|| Lasso::new(lasso_cfg()).unwrap(), &one),
        ),
        (
            "kalman",
            refuses_without_low_parts(|| Kalman::new(kalman_cfg()).unwrap(), &two),
        ),
        (
            "kalman_revert",
            refuses_without_low_parts(|| Kalman::new(kalman_revert_cfg()).unwrap(), &two),
        ),
        (
            "huber",
            refuses_without_low_parts(|| Robust::new(robust_cfg(ROBUST_LOSSES[0])).unwrap(), &two),
        ),
        (
            "quantile",
            refuses_without_low_parts(|| Robust::new(robust_cfg(ROBUST_LOSSES[1])).unwrap(), &two),
        ),
        (
            "ftrl",
            refuses_without_low_parts(|| Ftrl::new(ftrl_cfg()).unwrap(), &label2),
        ),
        (
            "sgd",
            refuses_without_low_parts(|| Sgd::new(sgd_cfg()).unwrap(), &two),
        ),
        (
            "pa",
            refuses_without_low_parts(|| Pa::new(pa_cfg()).unwrap(), &two),
        ),
        (
            "holt",
            refuses_without_low_parts(|| Holt::new(holt_cfg()).unwrap(), &two),
        ),
        (
            "ew_cov",
            refuses_without_low_parts(|| EwCovModel::new(ew_cov_model_cfg()).unwrap(), &none),
        ),
        (
            "kmeans",
            refuses_without_low_parts(|| KMeans::new(kmeans_cfg()).unwrap(), &none),
        ),
        (
            "micro",
            refuses_without_low_parts(|| Micro::new(micro_cfg()).unwrap(), &none),
        ),
        (
            "ew_class",
            refuses_without_low_parts(|| EwClass::new(ew_class_cfg()).unwrap(), &label),
        ),
        (
            "seqtest",
            refuses_without_low_parts(|| SeqTest::new(seqtest_cfg()).unwrap(), &two),
        ),
        (
            "marginal",
            refuses_without_low_parts(|| Marginal::new(marginal_cfg()).unwrap(), &two),
        ),
        (
            "deco",
            refuses_without_low_parts(|| Deco::new(deco_cfg()).unwrap(), &none),
        ),
        (
            "rcov",
            refuses_without_low_parts(|| Rcov::new(rcov_cfg()).unwrap(), &none),
        ),
        (
            "hmm",
            refuses_without_low_parts(|| Hmm::new(hmm_cfg()).unwrap(), &none),
        ),
        (
            "corrchange",
            refuses_without_low_parts(|| CorrChange::new(corrchange_cfg()).unwrap(), &none),
        ),
        (
            "bocpd",
            refuses_without_low_parts(|| Bocpd::new(bocpd_cfg()).unwrap(), &none),
        ),
    ];
    // Each model that keeps a mean had parts to drop, so its refusal is of
    // a real state: a model that lost its means, or a renamed field, shows
    // here. The rest keep no low part, and their state is whole.
    let keeps_means = [
        "ewridge",
        "lasso",
        "kalman",
        "kalman_revert",
        "huber",
        "quantile",
        "pa",
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
    for (name, (stripped, refused)) in &results {
        if keeps_means.contains(name) {
            assert!(*stripped > 0, "{name}: no low part in its state");
            assert!(*refused, "{name}: a state without its low parts loaded");
        } else {
            assert_eq!(*stripped, 0, "{name}: list it among the models with means");
        }
    }
}

/// A windowed `ewridge` or `lasso` saved at schema 16, when each target's
/// own feature mean was kept as its offset `d` from the all-row mean, is
/// refused: by its version, and by its layout, which names no `mj`. It
/// loaded at 17 with the offsets made means once (review 2026-09-27, G3),
/// a loader the floor at the schema shipped retired (docs/PLAN.md task
/// 198).
#[test]
fn a_schema_16_state_of_offsets_is_refused() {
    fn to_offsets(v: &mut rmpv::Value, n: &mut usize) {
        match v {
            rmpv::Value::Map(entries) => {
                entries.retain(|(k, _)| k.as_str() != Some("mj_lo"));
                for (k, x) in entries.iter_mut() {
                    if k.as_str() == Some("mj") {
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
    fn check<M: OnlineModel>(name: &str, make: impl Fn() -> M, rows: &[Row]) {
        let mut m = make();
        for (i, r) in rows.iter().enumerate().take(80) {
            m.step(&r.x, &r.y, if i == 0 { 0.0 } else { 1.0 }, r.w);
        }
        let state = m.state();
        let mut old = state.clone();
        old.schema_version = 16;
        assert!(
            matches!(
                M::restore(&old),
                Err(StateError::SchemaVersion { found: 16, .. })
            ),
            "{name}: refused by its version"
        );
        let bytes = rmp_serde::to_vec_named(&state).unwrap();
        let mut v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
        let mut n = 0;
        to_offsets(&mut v, &mut n);
        assert!(n >= 2, "{name}: {n} offset vectors written");
        let mut out = Vec::new();
        rmpv::encode::write_value(&mut out, &v).unwrap();
        assert!(
            rmp_serde::from_slice::<State>(&out).is_err(),
            "{name}: refused by its layout"
        );
    }
    let rows = leveled(2, false);
    for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
        let ridge = || {
            let mut c = ewridge_cfg();
            c.window = Some(30.0);
            c.max_rows_between_snapshots = Some(4);
            c.target_gaps = gaps;
            EwRidge::new(c).unwrap()
        };
        check(&format!("ewridge {gaps:?}"), ridge, &rows);
    }
    let one = leveled(1, false);
    let lasso = || {
        let mut c = lasso_cfg();
        c.window = Some(30.0);
        c.max_rows_between_snapshots = Some(4);
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
    /// prediction read `-inf` -- windowed or not, and through `ewridge`,
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
        contract(|| EwRidge::new(ewridge_cfg()).unwrap(), &two, 0).unwrap();
    }

    /// The stream the windowed `lasso` failed the contract on (review
    /// 2026-09-27, G5): rows at `1e100` before the window, one of weight
    /// `1e100` inside it, and the prediction at `x1 = 1e100` read `-inf`
    /// (`crate::truncated` has the mechanism). Through `ewridge` too.
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
                let mut c = ewridge_cfg();
                c.window = Some(7.0);
                EwRidge::new(c).unwrap()
            },
            &two,
            0,
        )
        .unwrap();
    }

    /// The case that failed the scheduled mutation pass of 2026-10-04 at its
    /// baseline (run 37194203887, shard 13), as proptest shrank it: on row
    /// 7 a feature at the input bound, standardized against a scale the
    /// earlier rows set, times its coefficient overflowed, and `kalman`
    /// predicted `inf` for the second target. Kept as a named test, since
    /// proptest cannot find this file's source to save its own regression
    /// file (docs/PLAN.md task 158).
    #[test]
    fn kalman_at_the_input_bound_predicts_a_number_or_nothing() {
        let row = |x: [f64; 2], y: Option<f64>, d: f64, w: f64| GenRow {
            x: x.to_vec(),
            y: vec![None, y],
            d,
            w,
        };
        let rows = [
            row([-1.197488167584902, 0.0], Some(0.0), 1.0, 2.785109721135609),
            row([0.0, 0.0], None, 4.697822926878844, 0.0),
            row(
                [-2.982070606445263, 4.622005473946053e49],
                Some(0.0),
                1.0,
                1.0,
            ),
            row([-0.6358232128145591, 0.0], Some(INPUT_BOUND), 1.0, 1e100),
            row([0.0, 0.0], None, 1.0, 0.0),
            row([0.0, 0.2675076319235289], Some(0.0), 1.0, 1.0),
            row([-INPUT_BOUND, 0.5], Some(0.0), 0.0, 1.0),
            row([0.0, INPUT_BOUND], None, 1.0, 1.0),
        ];
        contract(|| Kalman::new(kalman_cfg()).unwrap(), &rows, 0).unwrap();
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

        #[test]
        fn ewridge(rows in stream(2, false), split in 0usize..60) {
            contract(|| EwRidge::new(ewridge_cfg()).unwrap(), &rows, split)?;
        }

        /// The window's ring, its snapshots and their serde, under both
        /// cadences; and the blocked Gram with pairwise gaps, whose held
        /// rows the state carries (review 2026-09-26, C8).
        #[test]
        fn ewridge_windowed(rows in stream(2, false), split in 0usize..60, every in 0usize..2) {
            contract(|| {
                let mut c = ewridge_cfg();
                c.window = Some(7.0);
                c.max_rows_between_snapshots = [None, Some(3)][every];
                EwRidge::new(c).unwrap()
            }, &rows, split)?;
        }

        #[test]
        fn ewridge_blocked_pairwise(rows in stream(2, false), split in 0usize..60) {
            contract(|| {
                let mut c = ewridge_cfg();
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
                c.max_rows_between_snapshots = [None, Some(3)][every];
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

        /// `pa` and `sgd` standardize by default since docs/PLAN.md task 195
        /// (U2), so the contract holds the scaled step too.
        #[test]
        fn pa_standardized(rows in stream(2, false), split in 0usize..60) {
            contract(|| Pa::new(PaCfg { standardize: true, ..pa_cfg() }).unwrap(), &rows, split)?;
        }

        #[test]
        fn sgd_standardized(rows in stream(2, false), split in 0usize..60) {
            contract(
                || Sgd::new(SgdCfg { standardize: true, ..sgd_cfg() }).unwrap(),
                &rows,
                split,
            )?;
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
                c.max_rows_between_snapshots = [None, Some(3)][every];
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
                c.max_rows_between_snapshots = [None, Some(3)][every];
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
                c.max_rows_between_snapshots = [None, Some(3)][every];
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
        fn bocpd_on_the_clock(rows in stream(0, false), split in 0usize..60) {
            contract(|| Bocpd::new(bocpd_on_the_clock_cfg()).unwrap(), &rows, split)?;
        }

        #[test]
        fn bocpd_from_the_data(rows in stream(0, false), split in 0usize..60) {
            contract(|| Bocpd::new(bocpd_from_the_data_cfg()).unwrap(), &rows, split)?;
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
