//! Golden-value regression tests: one fixed stream per model, with the exact
//! expected outputs embedded.
//!
//! Why these exist. `online-core` is meant to be exhaustively unit-tested
//! (CLAUDE.md), but a `cargo mutants` pass showed 517 of 1645 mutations
//! surviving — the arithmetic inside the recursions is largely pinned by the
//! *Python* oracle suite (`tests/reference.py`, agreement to ~1e-13), which
//! `cargo test` cannot see. A single golden run per model closes that gap
//! cheaply: any change to a coefficient, a decay factor, an accumulator update
//! or a solve moves these numbers.
//!
//! The constants are not arbitrary. They are the current implementation's
//! output, and that implementation is independently verified elsewhere: the
//! numpy references in `tests/reference.py` for `ewridge`, `rls`, `kalman`,
//! `huber`/`quantile` and `ftrl`; the lasso's KKT conditions in
//! `tests/test_oracles.py`; and, for `sgd`, `pa`, `holt` and `ew_cov`, the
//! recursion written out longhand in each module's own unit tests. This file
//! locks in numbers that have already been checked against something else.
//!
//! Every model has a signature here, and
//! `tests/test_model_registry.py::test_the_core_golden_file_pins_every_model`
//! fails when one is missing.
//!
//! Regenerate with `PRINT_GOLDEN=1 cargo test -p online-core --test golden --
//! --nocapture`, and only after confirming a change is intended.

use online_core::*;

/// Deterministic pseudo-random stream, no external rng dependency.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// One row of the fixed stream: features, targets, clock delta, row weight.
type Row = ([f64; 2], [Option<f64>; 1], f64, f64);

/// 60 rows: two features, one null target and one clock gap, so the null
/// policy and the decay both participate.
fn stream() -> Vec<Row> {
    let mut s = 20240830u64;
    (0..60)
        .map(|i| {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.75 * x[1] + 0.25 + 0.1 * lcg(&mut s);
            let target = if i == 31 { None } else { Some(y) };
            let d = if i == 0 {
                0.0
            } else if i == 40 {
                25.0
            } else {
                1.0
            };
            (x, [target], d, 0.5 + 0.5 * (i % 3) as f64)
        })
        .collect()
}

/// Predictions at rows 20, 45 and 59 for one output slot.
fn signature<M: OnlineModel>(model: &mut M, pick: usize) -> Vec<f64> {
    signature_of(model, pick, true)
}

/// The same three rows, with the features withheld for the model that
/// takes none (`holt`).
fn signature_of<M: OnlineModel>(model: &mut M, pick: usize, features: bool) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        let x: &[f64] = if features { &x } else { &[] };
        let step = model.step(x, &y, d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[pick]);
        }
    }
    out
}

fn check(name: &str, got: &[f64], want: &[f64]) {
    // `1` and nothing else: any value, the empty string included, turned
    // every golden off with only stdout to say so (review 2026-10-06, CF9).
    if std::env::var("PRINT_GOLDEN").is_ok_and(|v| v == "1") {
        // `{v:?}` prints the shortest representation that round-trips, which
        // avoids clippy's excessive-precision lint on the embedded constants.
        let vals: Vec<String> = got.iter().map(|v| format!("{v:?}")).collect();
        println!("GOLDEN {name}: &[{}];", vals.join(", "));
        return;
    }
    assert_eq!(got.len(), want.len(), "{name}: wrong signature length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
            "{name}[{i}]: got {g:.17e}, want {w:.17e}"
        );
    }
}

fn ewridge_cfg(standardize: bool, ridge: f64) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        ridge: vec![ridge],
        feature_sets: vec![],
        standardize,
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

fn robust_cfg(loss: RobustLoss, standardize: bool) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        loss,
        ridge: 1e-4,
        standardize,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 1e-3,
    }
}

#[test]
fn ewridge_golden() {
    let mut m = EwRidge::new(ewridge_cfg(false, 1e-4)).unwrap();
    check("ewridge", &signature(&mut m, 0), GOLDEN_EW_RIDGE);
}

#[test]
fn ewridge_standardized_golden() {
    let mut m = EwRidge::new(ewridge_cfg(true, 0.01)).unwrap();
    check("ewridge_std", &signature(&mut m, 0), GOLDEN_EW_RIDGE_STD);
}

#[test]
fn ewridge_windowed_golden() {
    // A hard window with `window_every = 3`: the fit reads only the rows
    // inside the window, and a snapshot is kept every third row. This pins it
    // so P1's move of the snapshot build into the `offer` closure -- built
    // only when it is kept, not on every row -- cannot move a number (review
    // 2026-09-18, P1).
    let mut c = ewridge_cfg(false, 1e-4);
    c.window = Some(30.0);
    c.max_rows_between_snapshots = Some(3);
    let mut m = EwRidge::new(c).unwrap();
    // Under `closed = "both"`, as `lasso_windowed_golden` says why (this
    // stream's cadence puts no snapshot exactly one window old, so the two
    // edges agree here).
    m.set_window_closed(WindowClosed::Both);
    check(
        "ewridge_windowed",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_WINDOWED,
    );
}

/// The through-origin fits move no number above (every signature there has
/// an intercept), so the branches that only they take -- `ewridge`'s
/// standardized solve through the origin, with the warm prior it dropped
/// until the review of 2026-09-18 (B2, T1), `robust`'s raw-scaled one, and
/// `kalman` without the standardizer -- are pinned here.
#[test]
fn ewridge_origin_standardized_golden() {
    let mut c = ewridge_cfg(true, 0.05);
    c.fit_intercept = false;
    c.coef_prior = Some(vec![vec![1.0, -0.5]]);
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_origin_std",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_ORIGIN_STD,
    );
}

#[test]
fn huber_origin_standardized_golden() {
    let mut c = robust_cfg(RobustLoss::Huber { delta: 1.5 }, true);
    c.fit_intercept = false;
    let mut m = Robust::new(c).unwrap();
    check(
        "huber_origin_std",
        &signature(&mut m, 0),
        GOLDEN_HUBER_ORIGIN_STD,
    );
}

#[test]
fn kalman_plain_golden() {
    let mut m = Kalman::new(KalmanCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    })
    .unwrap();
    check("kalman_plain", &signature(&mut m, 0), GOLDEN_KALMAN_PLAIN);
}

#[test]
fn rls_golden() {
    let mut m = Rls::new(RlsCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        delta: 0.5,
        coef_prior: None,
        min_weight: 3.0,
    })
    .unwrap();
    check("rls", &signature(&mut m, 0), GOLDEN_RLS);
}

#[test]
fn kalman_golden() {
    let mut m = Kalman::new(KalmanCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    })
    .unwrap();
    check("kalman", &signature(&mut m, 0), GOLDEN_KALMAN);
}

/// E41: the intercept a random walk, both slopes reverting, one of them
/// fast, on the same stream as `kalman_golden`.
#[test]
fn kalman_revert_golden() {
    let mut m = Kalman::new(KalmanCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY, 40.0, 8.0],
        standardize: true,
    })
    .unwrap();
    check("kalman_revert", &signature(&mut m, 0), GOLDEN_KALMAN_REVERT);
}

#[test]
fn lasso_golden() {
    let mut m = Lasso::new(LassoCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        lasso_path: vec![0.2, 0.02, 0.0],
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
        max_iter: 200,
        tol: 1e-12,
        target_gaps: online_core::TargetGaps::OwnRows,
    })
    .unwrap();
    check("lasso", &signature(&mut m, 1), GOLDEN_LASSO);
}

#[test]
fn huber_golden() {
    let mut m = Robust::new(robust_cfg(RobustLoss::Huber { delta: 1.5 }, false)).unwrap();
    check("huber", &signature(&mut m, 0), GOLDEN_HUBER);
}

#[test]
fn quantile_golden() {
    let mut m = Robust::new(robust_cfg(RobustLoss::Quantile { tau: 0.7 }, true)).unwrap();
    check("quantile", &signature(&mut m, 0), GOLDEN_QUANTILE);
}

#[test]
fn ftrl_golden() {
    let mut m = Ftrl::new(FtrlCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(40.0),
        alpha: 0.1,
        beta: 1.0,
        l1: 0.05,
        l2: 1.0,
        min_weight: 3.0,
        strict_binary: false,
        loss: FtrlLoss::Logistic,
    })
    .unwrap();
    check("ftrl", &signature(&mut m, 0), GOLDEN_FTRL);
}

#[test]
fn ftrl_squared_golden() {
    let mut m = Ftrl::new(FtrlCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(40.0),
        alpha: 0.5,
        beta: 1.0,
        l1: 0.05,
        l2: 0.01,
        min_weight: 3.0,
        strict_binary: false,
        loss: FtrlLoss::Squared,
    })
    .unwrap();
    check("ftrl_squared", &signature(&mut m, 0), GOLDEN_FTRL_SQUARED);
}

/// The busier path: a robust loss, an annealed rate, a penalty and the
/// running standardization all take part, so each has a number to move.
#[test]
fn sgd_golden() {
    let mut m = Sgd::new(SgdCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        loss: SgdLoss::Huber { delta: 0.5 },
        learning_rate: 0.05,
        schedule: LearningRate::InvScaling { power: 0.25 },
        l2: 0.01,
        clip_gradient: 1e3,
        constraint: None,
        standardize: true,
        strict_binary: false,
        min_weight: 3.0,
    })
    .unwrap();
    check("sgd", &signature(&mut m, 0), GOLDEN_SGD);
}

/// The plain path: squared loss, constant rate, no penalty, raw features.
#[test]
fn sgd_squared_golden() {
    let mut m = Sgd::new(SgdCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        loss: SgdLoss::Squared,
        learning_rate: 0.05,
        schedule: LearningRate::Constant,
        l2: 0.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize: false,
        strict_binary: false,
        min_weight: 3.0,
    })
    .unwrap();
    check("sgd_squared", &signature(&mut m, 0), GOLDEN_SGD_SQUARED);
}

#[test]
fn pa_golden() {
    let mut m = Pa::new(PaCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        mode: PaMode::Pa2,
        c: 0.5,
        eps: 0.05,
        min_weight: 3.0,
        constraint: None,
        standardize: false,
    })
    .unwrap();
    check("pa", &signature(&mut m, 0), GOLDEN_PA);
}

/// The constrained path (ENHANCEMENTS E40): slopes on the simplex, so the
/// projection with a sum and a wall both take part; the truth (1.5, -0.75)
/// lies outside the set.
#[test]
fn sgd_simplex_golden() {
    let mut m = Sgd::new(SgdCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        loss: SgdLoss::Squared,
        learning_rate: 0.05,
        schedule: LearningRate::Constant,
        l2: 0.0,
        clip_gradient: 1e3,
        constraint: Some(Constraint {
            lo: vec![0.0, 0.0],
            hi: vec![f64::INFINITY, f64::INFINITY],
            sum: Some(1.0),
        }),
        standardize: false,
        strict_binary: false,
        min_weight: 3.0,
    })
    .unwrap();
    check("sgd_simplex", &signature(&mut m, 0), GOLDEN_SGD_SIMPLEX);
}

/// A box with one wall the truth crosses (`x1`'s slope capped at 0) and a
/// sum below what the free fit would give.
#[test]
fn pa_box_golden() {
    let mut m = Pa::new(PaCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        mode: PaMode::Pa1,
        c: 0.5,
        eps: 0.05,
        min_weight: 3.0,
        constraint: Some(Constraint {
            lo: vec![-1.0, -1.0],
            hi: vec![1.0, 0.0],
            sum: Some(0.5),
        }),
        standardize: false,
    })
    .unwrap();
    check("pa_box", &signature(&mut m, 0), GOLDEN_PA_BOX);
}

#[test]
fn holt_golden() {
    let mut m = Holt::new(HoltCfg {
        n_targets: 1,
        level_half_life: 10.0,
        trend_half_life: 40.0,
        min_weight: 3.0,
        trend: true,
    })
    .unwrap();
    check("holt", &signature_of(&mut m, 0, false), GOLDEN_HOLT);
}

#[test]
fn seqtest_golden() {
    // Slot 0 is the log e-value for "the target is positive", slot 1 the one
    // for "negative": together they pin the stake, the wealth update and the
    // counts (a wrong count moves the next stake).
    let mut m = SeqTest::new(SeqTestCfg {
        n_targets: 1,
        min_weight: 0.0,
    })
    .unwrap();
    check("seqtest", &signature_of(&mut m, 0, false), GOLDEN_SEQTEST);
    let mut m = SeqTest::new(SeqTestCfg {
        n_targets: 1,
        min_weight: 0.0,
    })
    .unwrap();
    check(
        "seqtest_neg",
        &signature_of(&mut m, 1, false),
        GOLDEN_SEQTEST_NEG,
    );
}

/// Slot 0 is `p_change`, which reads the whole run-length posterior and
/// every run's predictive at once.
#[test]
fn bocpd_golden() {
    let mut m = Bocpd::new(BocpdCfg {
        n_features: 2,
        hazard: 30.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: Some(vec![0.0, 0.0]),
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-8,
        max_run: 100,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: false,
    })
    .unwrap();
    check("bocpd", &signature(&mut m, 0), GOLDEN_BOCPD);
}

/// `bocpd_golden` with the hazard on the clock, `τ = 30` clock units
/// between changepoints (docs/PLAN.md task 179): the stream's steps of 1
/// carry a chance of 0.033 each and its gap of 25 one of 0.565, applied
/// before the row each leads into. Checked against the definition by the
/// enumeration in `tests/test_bocpd.py`, and against the per-row hazard on
/// regular steps by `bocpd.rs`'s tests.
#[test]
fn bocpd_on_the_clock_golden() {
    let mut m = Bocpd::new(BocpdCfg {
        n_features: 2,
        hazard: 30.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: Some(vec![0.0, 0.0]),
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-8,
        max_run: 100,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: true,
    })
    .unwrap();
    check(
        "bocpd_on_the_clock",
        &signature(&mut m, 0),
        GOLDEN_BOCPD_ON_THE_CLOCK,
    );
}

/// `corrchange` reports only where a span closes, so its signature is the
/// statistic at the three span ends a 60-row stream at `horizon = 20` has.
#[test]
fn corrchange_golden() {
    let mut m = CorrChange::new(CorrChangeCfg {
        n_features: 2,
        kind: CorrChangeKind::Monitor,
        span_rows: 20,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(20.0),
        crit: None,
        n_perm: 20,
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
    })
    .unwrap();
    let mut out = Vec::new();
    for (x, y, d, w) in stream() {
        let step = m.step(&x, &y, d, w);
        if step.pred[0].is_finite() {
            out.push(step.pred[0]);
        }
    }
    check("corrchange", &out, GOLDEN_CORRCHANGE);
}

/// Slot 5 is `loglik`, which reads the transition matrix, both densities
/// and the softmax at once -- the whole filter in one number.
#[test]
fn hmm_golden() {
    let mut m = Hmm::new(HmmCfg {
        n_features: 2,
        k: 2,
        decay: Decay::Halflife(20.0),
        covariance: Covariance::Full,
        precision_prior: 1e-2,
        min_weight: 3.0,
        learn: true,
        transition_prior: 1.0,
        transition: Some(vec![0.9, 0.1, 0.2, 0.8]),
        means: Some(vec![-0.5, -0.5, 0.5, 0.5]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 1,
        tvtp: None,
    })
    .unwrap();
    check("hmm", &signature(&mut m, 5), GOLDEN_HMM);
}

/// `rcov` reports nothing per row, so its signature is the block: the
/// kernel estimate's three distinct entries after the whole stream.
#[test]
fn rcov_golden() {
    let mut m = Rcov::new(RcovCfg {
        n_features: 2,
        kind: RcovKind::Kernel,
        kernel: "parzen".into(),
        bandwidth: Some(3),
        jitter: 2,
        theta: 1.0,
        psd: false,
        block_rows: Some(60),
        max_bandwidth: None,
        preavg_rows: None,
        noise_stride: 1,
        iv_stride: 20,
    })
    .unwrap();
    for (x, y, d, w) in stream() {
        m.step(&x, &y, d, w);
    }
    let cov = m.estimate().rcov.expect("a block");
    check("rcov", &[cov[0], cov[1], cov[3]], GOLDEN_RCOV);
}

/// The pre-averaged estimate on the same stream: the other arithmetic.
#[test]
fn rcov_preavg_golden() {
    let mut m = Rcov::new(RcovCfg {
        n_features: 2,
        kind: RcovKind::Preavg,
        kernel: "parzen".into(),
        bandwidth: None,
        jitter: 2,
        theta: 1.0,
        psd: false,
        block_rows: Some(60),
        max_bandwidth: None,
        preavg_rows: Some(6),
        noise_stride: 1,
        iv_stride: 20,
    })
    .unwrap();
    for (x, y, d, w) in stream() {
        m.step(&x, &y, d, w);
    }
    let cov = m.estimate().rcov.expect("a block");
    check("rcov_preavg", &[cov[0], cov[1], cov[3]], GOLDEN_RCOV_PREAVG);
}

#[test]
fn deco_golden() {
    // Slot 1 is `rho`, the level the whole model is for: it reads the
    // standardiser, the row estimate and the recursion at once. The
    // `"linear"` dynamics is the busier path -- it runs the `"ew"` recursion
    // alongside as its target -- so that is the one pinned.
    let mut m = Deco::new(DecoCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        dynamics: DecoDynamics::Linear,
        alpha: Some(0.05),
        beta: Some(0.9),
        blocks: Vec::new(),
        min_weight: 3.0,
    })
    .unwrap();
    check("deco", &signature(&mut m, 1), GOLDEN_DECO);
}

#[test]
fn deco_loglik_golden() {
    // Slot 2 is `loglik`, which is the only consumer of the block algebra.
    let mut m = Deco::new(DecoCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        dynamics: DecoDynamics::Ew,
        alpha: None,
        beta: None,
        blocks: Vec::new(),
        min_weight: 3.0,
    })
    .unwrap();
    check("deco_loglik", &signature(&mut m, 2), GOLDEN_DECO_LOGLIK);
}

#[test]
fn ew_cov_golden() {
    // Slots in emission order: mean x0, mean x1, var x0, var x1, corr x0x1.
    // The correlation is the one that reads every accumulator at once.
    let mut m = EwCovModel::new(EwCovCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
        min_weight: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    check("ew_cov", &signature(&mut m, 4), GOLDEN_EW_COV);
}

/// `marginal` predicts nothing per row, so its signature is read from the
/// state: the pair `(x1, y)`'s correlation, slope and Kish size after rows
/// 20, 45 and 59 -- together they touch every accumulator (both means, both
/// variances, the covariance, `W_t` and `Q_t`).
fn marginal_signature(model: &mut Marginal) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        let step = model.step(&x, &y, d, w);
        assert!(step.pred.is_empty());
        if matches!(i, 20 | 45 | 59) {
            let p = model.pair(0, 1);
            out.extend([p.corr, p.beta, p.n_kish]);
        }
    }
    out
}

#[test]
fn marginal_golden() {
    let mut m = Marginal::new(MarginalCfg {
        n_features: 2,
        n_targets: 1,
        decay: Decay::Halflife(20.0),
        min_weight: vec![3.0],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: online_core::FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    check("marginal", &marginal_signature(&mut m), GOLDEN_MARGINAL);
}

fn kmeans_cfg(rule: SeedRule) -> KMeansCfg {
    KMeansCfg {
        n_features: 2,
        k: 3,
        decay: Decay::Halflife(20.0),
        min_weight: 3.0,
        warm_rows: 12,
        seed_rule: rule,
        seed: 0,
        update_every_rows: 1,
        split_merge: 0.5,
        split_merge_every_rows: 10,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.0,
    }
}

#[test]
fn kmeans_golden() {
    // Slot 1 is the distance to the assigned centre under the standardized
    // metric: it reads the centres, the feature moments and the assignment
    // at once. Slot 0 pins the assignment itself, seeded by the generator.
    let mut m = KMeans::new(kmeans_cfg(SeedRule::Lloyd)).unwrap();
    check("kmeans", &signature(&mut m, 1), GOLDEN_KMEANS);
    let mut m = KMeans::new(kmeans_cfg(SeedRule::Lloyd)).unwrap();
    check(
        "kmeans_cluster",
        &signature(&mut m, 0),
        GOLDEN_KMEANS_CLUSTER,
    );
    // The `first` rule with a checkpoint every seven rows: the batch path.
    let mut m = KMeans::new(KMeansCfg {
        update_every_rows: 7,
        ..kmeans_cfg(SeedRule::First)
    })
    .unwrap();
    check("kmeans_first", &signature(&mut m, 2), GOLDEN_KMEANS_FIRST);
}

fn micro_cfg() -> MicroCfg {
    MicroCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        min_weight: 3.0,
        eps: 0.35,
        beta_mu: 2.0,
        max_clusters: 8,
        prune_every: f64::INFINITY,
        max_rows_between_prunes: 7,
        macro_link: None,
        standardize: true,
        scale_floor: 0.0,
    }
}

#[test]
fn micro_golden() {
    // Slot 1 is the distance to the nearest potential summary under the
    // standardized metric: it reads the centres, the moments and the
    // promotion rule at once. Slot 0 pins the labels the linkage assigns
    // with the derived threshold; slot 2 the ids, which read the cap and
    // the pruning.
    let mut m = Micro::new(micro_cfg()).unwrap();
    check("micro", &signature(&mut m, 1), GOLDEN_MICRO);
    let mut m = Micro::new(micro_cfg()).unwrap();
    check("micro_cluster", &signature(&mut m, 0), GOLDEN_MICRO_CLUSTER);
    let mut m = Micro::new(MicroCfg {
        macro_link: Some(0.0),
        standardize: false,
        ..micro_cfg()
    })
    .unwrap();
    check("micro_id", &signature(&mut m, 2), GOLDEN_MICRO_ID);
}

fn ew_class_cfg(covariance: Covariance) -> EwClassCfg {
    EwClassCfg {
        n_features: 2,
        n_classes: 2,
        decay: Decay::Halflife(20.0),
        min_weight: 3.0,
        covariance,
        precision_prior: 0.1,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// The three rows' outputs with the stream's target turned into a label:
/// class 1 where `y > 0.25`, else class 0; the null stays null.
fn labelled_signature<M: OnlineModel>(model: &mut M, pick: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        let label = y[0].map(|v| if v > 0.25 { 1.0 } else { 0.0 });
        let step = model.step(&x, &[label], d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[pick]);
        }
    }
    out
}

#[test]
fn ew_class_golden() {
    // Slot 1 is the posterior of class 0: it reads both classes' means,
    // co-moments, decaying ridges and priors at once, through the solve.
    // Slot 0 pins the assignment.
    let mut m = EwClass::new(ew_class_cfg(Covariance::Full)).unwrap();
    check("ew_class", &labelled_signature(&mut m, 1), GOLDEN_EW_CLASS);
    let mut m = EwClass::new(ew_class_cfg(Covariance::Full)).unwrap();
    check(
        "ew_class_class",
        &labelled_signature(&mut m, 0),
        GOLDEN_EW_CLASS_CLASS,
    );
    let mut m = EwClass::new(ew_class_cfg(Covariance::Shared)).unwrap();
    check(
        "ew_class_shared",
        &labelled_signature(&mut m, 2),
        GOLDEN_EW_CLASS_SHARED,
    );
    let mut m = EwClass::new(ew_class_cfg(Covariance::Diagonal)).unwrap();
    check(
        "ew_class_diagonal",
        &labelled_signature(&mut m, 1),
        GOLDEN_EW_CLASS_DIAGONAL,
    );
}

// --- second signatures: the settings the first never runs ---
//
// One per model, each over what its golden above leaves at a default, so a
// mutant in those branches moves a number `cargo mutants` sees: the Python
// golden pipeline pins most of them, which `cargo test` cannot see (review
// 2026-10-06, CF10). Clear of what task 186 moves: no target that starts
// late in a solving model, no windowed `ew_class` coefficient (its
// posteriors only), no `share_p`, no `rcov` across a break.

/// The fixed stream with `x1` held still from row 30 on: a feature gone
/// quiet, where a clusterer's `scale_floor` binds.
fn quiet_stream() -> Vec<Row> {
    stream()
        .into_iter()
        .enumerate()
        .map(|(i, (x, y, d, w))| (if i >= 30 { [x[0], 0.3] } else { x }, y, d, w))
        .collect()
}

/// [`signature`] on `rows`.
fn signature_on<M: OnlineModel>(model: &mut M, pick: usize, rows: Vec<Row>) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in rows.into_iter().enumerate() {
        let step = model.step(&x, &y, d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[pick]);
        }
    }
    out
}

/// [`signature`] with a call between rows: `between(model, i)` runs before
/// row `i`.
fn signature_between<M: OnlineModel>(
    model: &mut M,
    pick: usize,
    mut between: impl FnMut(&mut M, usize),
) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        between(model, i);
        let step = model.step(&x, &y, d, w);
        if matches!(i, 20 | 45 | 59) {
            out.push(step.pred[pick]);
        }
    }
    out
}

/// `kmeans` seeded by farthest-first and by k-means++, on the raw metric,
/// and on the standardized one with a scale floor, where a feature has gone
/// quiet (`quiet_stream`): the floor moves the distances there.
#[test]
fn kmeans_seeding_and_metric_golden() {
    let mut m = KMeans::new(kmeans_cfg(SeedRule::Farthest)).unwrap();
    check(
        "kmeans_farthest",
        &signature(&mut m, 1),
        GOLDEN_KMEANS_FARTHEST,
    );
    let mut m = KMeans::new(kmeans_cfg(SeedRule::Kmeanspp)).unwrap();
    check(
        "kmeans_kmeanspp",
        &signature(&mut m, 1),
        GOLDEN_KMEANS_KMEANSPP,
    );
    let mut m = KMeans::new(KMeansCfg {
        standardize: false,
        ..kmeans_cfg(SeedRule::Lloyd)
    })
    .unwrap();
    check("kmeans_raw", &signature(&mut m, 1), GOLDEN_KMEANS_RAW);
    let floored = |scale_floor| {
        let mut m = KMeans::new(KMeansCfg {
            scale_floor,
            ..kmeans_cfg(SeedRule::Lloyd)
        })
        .unwrap();
        signature_on(&mut m, 1, quiet_stream())
    };
    let got = floored(0.5);
    assert_ne!(got, floored(0.0), "the floor binds on the quiet feature");
    check("kmeans_floor", &got, GOLDEN_KMEANS_FLOOR);
}

/// `micro` pruned on the clock, every 5 units (the gap of 25 is one prune),
/// and with a scale floor where a feature has gone quiet.
#[test]
fn micro_on_the_clock_golden() {
    let mut m = Micro::new(MicroCfg {
        prune_every: 5.0,
        max_rows_between_prunes: u32::MAX,
        ..micro_cfg()
    })
    .unwrap();
    check("micro_clock", &signature(&mut m, 1), GOLDEN_MICRO_CLOCK);
    let floored = |scale_floor| {
        let mut m = Micro::new(MicroCfg {
            scale_floor,
            ..micro_cfg()
        })
        .unwrap();
        signature_on(&mut m, 1, quiet_stream())
    };
    let got = floored(0.5);
    assert_ne!(got, floored(0.0), "the floor binds on the quiet feature");
    check("micro_floor", &got, GOLDEN_MICRO_FLOOR);
}

/// `ewridge` over feature sets, under `session_shrink` with a session
/// boundary at row 30, with pairwise gaps, with the Gram blocked, and on
/// the solve schedules: the clock, and the default share of the weight.
#[test]
fn ewridge_paths_golden() {
    let mut c = ewridge_cfg(false, 1e-4);
    c.feature_sets = vec![("one".into(), vec![0]), ("both".into(), vec![0, 1])];
    let mut m = EwRidge::new(c.clone()).unwrap();
    check(
        "ewridge_set_one",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_SET_ONE,
    );
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_set_both",
        &signature(&mut m, 1),
        GOLDEN_EW_RIDGE_SET_BOTH,
    );

    let mut c = ewridge_cfg(false, 1e-4);
    c.session_shrink = Some(0.5);
    c.long_half_life = Some(80.0);
    let mut m = EwRidge::new(c).unwrap();
    let got = signature_between(&mut m, 0, |m, i| {
        if i == 30 {
            m.blend_toward_long_run();
        }
    });
    check(
        "ewridge_session_shrink",
        &got,
        GOLDEN_EW_RIDGE_SESSION_SHRINK,
    );

    let mut c = ewridge_cfg(false, 1e-4);
    c.target_gaps = online_core::TargetGaps::Pairwise;
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_pairwise",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_PAIRWISE,
    );

    let mut c = ewridge_cfg(false, 1e-4);
    c.gram_block_rows = 5;
    c.solve_every = 4.0;
    c.max_rows_between_solves = 8;
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_blocked",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_BLOCKED,
    );

    let mut c = ewridge_cfg(false, 1e-4);
    c.solve_every = 3.0;
    c.max_rows_between_solves = u32::MAX;
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_on_the_clock",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_ON_THE_CLOCK,
    );

    // A fifth of the fit's weight between solves: the default share, ln 2 /
    // 50, solves after every row on a stream this short.
    let mut c = ewridge_cfg(false, 1e-4);
    c.max_rows_between_solves = u32::MAX;
    c.solve_share = Some(0.2);
    let mut m = EwRidge::new(c).unwrap();
    check(
        "ewridge_by_weight",
        &signature(&mut m, 0),
        GOLDEN_EW_RIDGE_BY_WEIGHT,
    );
}

/// The lasso under a window, and selecting on its own half-life.
#[test]
fn lasso_windowed_golden() {
    let cfg = || LassoCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(20.0),
        lasso_path: vec![0.2, 0.02, 0.0],
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
        max_iter: 200,
        tol: 1e-12,
        target_gaps: online_core::TargetGaps::OwnRows,
    };
    let mut m = Lasso::new(LassoCfg {
        window: Some(30.0),
        ..cfg()
    })
    .unwrap();
    // Pinned under `closed = "both"`, the edge every window had before task
    // 196, so the numbers are the old ones to the bit; `"right"`, the
    // default, is held to the definition in each model's unit tests.
    m.set_window_closed(WindowClosed::Both);
    check(
        "lasso_windowed",
        &signature(&mut m, 1),
        GOLDEN_LASSO_WINDOWED,
    );
    // The selection, read every fifth row from row 10: the penalty whose EW
    // mean squared error, at `select_half_life`, is the least.
    let selected = |select_half_life| {
        let mut m = Lasso::new(LassoCfg {
            select_half_life,
            ..cfg()
        })
        .unwrap();
        let mut out = Vec::new();
        for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
            m.step(&x, &y, d, w);
            if i >= 10 && i % 5 == 0 {
                out.push(m.lam_selected()[0]);
            }
        }
        out
    };
    let got = selected(Some(3.0));
    assert_ne!(
        got,
        selected(None),
        "the selection's own half-life moves it"
    );
    check("lasso_select", &got, GOLDEN_LASSO_SELECT);
}

/// `kalman` with an explicit process noise, and with a fixed observation
/// noise and its prior.
#[test]
fn kalman_noise_golden() {
    let cfg = || KalmanCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(50.0),
        half_life: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    };
    let mut m = Kalman::new(KalmanCfg {
        q: Some(vec![0.0, 0.01, 0.02]),
        ..cfg()
    })
    .unwrap();
    check("kalman_q", &signature(&mut m, 0), GOLDEN_KALMAN_Q);
    let mut m = Kalman::new(KalmanCfg {
        obs_var: Some(0.25),
        p0: 4.0,
        ..cfg()
    })
    .unwrap();
    check(
        "kalman_obs_var",
        &signature(&mut m, 0),
        GOLDEN_KALMAN_OBS_VAR,
    );
}

/// `hmm` with a transition that moves with an exogenous column, the
/// stream's target in the target slot.
#[test]
fn hmm_tvtp_golden() {
    let mut m = Hmm::new(HmmCfg {
        n_features: 2,
        k: 2,
        decay: Decay::Halflife(20.0),
        covariance: Covariance::Full,
        precision_prior: 1e-2,
        min_weight: 3.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: Some(vec![-0.5, -0.5, 0.5, 0.5]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 1,
        tvtp: Some((vec![2.0, 0.0, 0.0, 2.0], vec![0.0, 1.5, -1.0, 0.0])),
    })
    .unwrap();
    check("hmm_tvtp", &signature(&mut m, 5), GOLDEN_HMM_TVTP);
}

/// `bocpd` with the full Gaussian emission, and the robust one at a
/// `robust_beta` above 0.
#[test]
fn bocpd_emissions_golden() {
    let cfg = |emission, robust_beta| BocpdCfg {
        n_features: 2,
        hazard: 30.0,
        hazard_from_row: false,
        emission,
        prior_mean: Some(vec![0.0, 0.0]),
        prior_kappa: 1.0,
        prior_nu: Some(4.0),
        prior_scale: Some(vec![1.0]),
        robust_beta,
        prune_below: 1e-8,
        max_run: 100,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: false,
    };
    let mut m = Bocpd::new(cfg(BocpdEmission::Gaussian, 0.0)).unwrap();
    check(
        "bocpd_gaussian",
        &signature(&mut m, 0),
        GOLDEN_BOCPD_GAUSSIAN,
    );
    let mut m = Bocpd::new(cfg(BocpdEmission::Robust, 0.2)).unwrap();
    check("bocpd_robust", &signature(&mut m, 0), GOLDEN_BOCPD_ROBUST);
}

/// `corrchange`'s sequential monitor: a history of 20 rows, then 20
/// monitored against W&G's boundary at `γ = 0.25`. Its signature is the
/// statistic on every row it reports.
#[test]
fn corrchange_sequential_golden() {
    let mut m = CorrChange::new(CorrChangeCfg {
        n_features: 2,
        kind: CorrChangeKind::Sequential,
        span_rows: 20,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(20.0),
        crit: None,
        n_perm: 20,
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows: 20,
        boundary_gamma: 0.25,
    })
    .unwrap();
    let mut out = Vec::new();
    for (x, y, d, w) in stream() {
        let step = m.step(&x, &y, d, w);
        if step.pred[0].is_finite() {
            out.push(step.pred[0]);
        }
    }
    assert!(out.len() >= 10, "the monitor reports: {}", out.len());
    check("corrchange_sequential", &out, GOLDEN_CORRCHANGE_SEQUENTIAL);
}

/// `rcov`'s pre-averaged estimate in its PSD form: the longer window, and
/// no bias term. (The kernel's PSD form clips a negative eigenvalue, which
/// this stream's estimate does not have: it repeats `GOLDEN_RCOV`. The clip
/// is held in `tests/test_rcov.py`.)
#[test]
fn rcov_psd_golden() {
    let cfg = |kind, bandwidth, preavg_rows| RcovCfg {
        n_features: 2,
        kind,
        kernel: "parzen".into(),
        bandwidth,
        jitter: 2,
        theta: 1.0,
        psd: true,
        block_rows: Some(60),
        max_bandwidth: None,
        preavg_rows,
        noise_stride: 1,
        iv_stride: 20,
    };
    for (name, c, want) in [(
        "rcov_preavg_psd",
        cfg(RcovKind::Preavg, None, None),
        GOLDEN_RCOV_PREAVG_PSD,
    )] {
        let mut m = Rcov::new(c).unwrap();
        for (x, y, d, w) in stream() {
            m.step(&x, &y, d, w);
        }
        let cov = m.estimate().rcov.expect("a block");
        check(name, &[cov[0], cov[1], cov[3]], want);
    }
}

/// `ew_cov` with lagged moments, with a principal component refreshed on
/// every row, and under a window.
#[test]
fn ew_cov_paths_golden() {
    let cfg = || EwCovCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        stats: vec![EwCovStat::Corr],
        min_weight: 3.0,
        precision_prior: None,
        mahal_quantiles: Vec::new(),
        pca: 0,
        pca_every: 0.0,
        max_rows_between_pca: u32::MAX,
        lags: Vec::new(),
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    // Slots: corr, then lag_corr per lag and ordered pair; slot 3 is lag 1's
    // (x0, x1).
    let mut m = EwCovModel::new(EwCovCfg {
        stats: vec![EwCovStat::Corr, EwCovStat::LagCorr],
        lags: vec![1, 2],
        ..cfg()
    })
    .unwrap();
    check("ew_cov_lags", &signature(&mut m, 2), GOLDEN_EW_COV_LAGS);
    // Slots: corr, then pc0's variance, share, two loadings and score; the
    // score reads the eigenvectors and the means at once.
    let mut m = EwCovModel::new(EwCovCfg { pca: 1, ..cfg() }).unwrap();
    check("ew_cov_pca", &signature(&mut m, 5), GOLDEN_EW_COV_PCA);
    let mut m = EwCovModel::new(EwCovCfg {
        window: Some(30.0),
        ..cfg()
    })
    .unwrap();
    // Under `closed = "both"`, as `lasso_windowed_golden` says why.
    m.set_window_closed(WindowClosed::Both);
    check(
        "ew_cov_windowed",
        &signature(&mut m, 0),
        GOLDEN_EW_COV_WINDOWED,
    );
}

/// `marginal` with lags and the serial count they feed, with bins and the
/// best split, and under a window.
#[test]
fn marginal_paths_golden() {
    let cfg = || MarginalCfg {
        n_features: 2,
        n_targets: 1,
        decay: Decay::Halflife(20.0),
        min_weight: vec![3.0],
        lags: Vec::new(),
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: online_core::FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    };
    let read = |m: &mut Marginal, f: &dyn Fn(&MarginalPair) -> [f64; 3]| {
        let mut out = Vec::new();
        for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
            m.step(&x, &y, d, w);
            if matches!(i, 20 | 45 | 59) {
                out.extend(f(&m.pair(0, 1)));
            }
        }
        out
    };
    let mut m = Marginal::new(MarginalCfg {
        lags: vec![1, 2],
        serial_rule: Some(online_core::SerialRule::Truncated),
        ..cfg()
    })
    .unwrap();
    let got = read(&mut m, &|p| [p.n_serial, p.t_serial, p.lag_corr_xy[0]]);
    check("marginal_lags", &got, GOLDEN_MARGINAL_LAGS);
    let mut m = Marginal::new(MarginalCfg {
        bins: Some(Box::new(online_core::BinCfg {
            n_bins: 4,
            edges: None,
            rule: online_core::BinRule::Quantile,
            warm_rows: 10,
            budget_mib: None,
        })),
        ..cfg()
    })
    .unwrap();
    let got = read(&mut m, &|p| [p.split_gain, p.split_at, p.bin_mean_y[1]]);
    check("marginal_bins", &got, GOLDEN_MARGINAL_BINS);
    let mut m = Marginal::new(MarginalCfg {
        window: Some(30.0),
        ..cfg()
    })
    .unwrap();
    // Under `closed = "both"`, as `lasso_windowed_golden` says why.
    m.set_window_closed(WindowClosed::Both);
    let got = read(&mut m, &|p| [p.corr, p.beta, p.n_kish]);
    check("marginal_windowed", &got, GOLDEN_MARGINAL_WINDOWED);
}

/// `ew_class` under a window: the posterior, which reads the window's
/// class moments (its `coef` is task 186's).
#[test]
fn ew_class_windowed_golden() {
    let mut m = EwClass::new(EwClassCfg {
        window: Some(30.0),
        ..ew_class_cfg(Covariance::Full)
    })
    .unwrap();
    // Under `closed = "both"`, as `lasso_windowed_golden` says why.
    m.set_window_closed(WindowClosed::Both);
    check(
        "ew_class_windowed",
        &labelled_signature(&mut m, 1),
        GOLDEN_EW_CLASS_WINDOWED,
    );
}

// --- generated; see the module docs ---
const GOLDEN_BOCPD: &[f64] = &[
    0.04617202802926437,
    0.05977980779859338,
    0.04295639028471272,
];
// Frozen 2026-10-06 (task 179), after a longhand of the clock form in plain
// probabilities -- weighted runs two-pass, the step's chance moved to the
// empty run before each row, pruned at 1e-8 -- gave the same three numbers
// to 4e-14.
const GOLDEN_BOCPD_ON_THE_CLOCK: &[f64] = &[
    0.01304148203335747,
    0.026769829077477554,
    0.009878006501232724,
];
// Re-frozen 2026-09-19 for the review's S3: the delta-method gradient of `ρ`
// carried the wrong powers of `σ_x` and `σ_y`, so `D̂` and every `Q` moved
// where the two columns' variances differ, as this stream's do. The
// finite-difference check in `corrchange.rs` is what the new numbers rest on.
// Regenerated 2026-09-28 (docs/PLAN.md task 114): the long-run variance's
// kernel is WKD's `1 − l/γ`, where it was Newey–West's `1 − l/(γ+1)`
// (0.8418929529846794, 0.7933802263147072, 0.787701333637405).
const GOLDEN_CORRCHANGE: &[f64] = &[0.8103851544234802, 0.764942538158755, 0.6621041576121629];
// Re-frozen 2026-09-06: `hmm`'s `min_weight` used to withhold a row from
// the *update* as well as from the report, so the first rows of this stream
// (`min_weight = 3`) never reached the filter. It now gates the report
// alone, as it does in every other model (docs/REVIEW-E54-E64.md H1).
// Re-pinned 2026-10-08 (docs/PLAN.md task 214): a row reads the transition
// matrix from counts its clock has aged, `(λA + τ)/Σ`, where it read `(A +
// τ)/Σ`, so a zero-weight row is clock alone (hard rule 9). A replica written
// from the module docs gives these to 2.5e-16, and the old ones, under the old
// reading, to 1.2e-16 (-1.1511065244639265, -2.6218190023816343,
// -0.875789289803333).
const GOLDEN_HMM: &[f64] = &[
    -1.1509510476558056,
    -2.6217067703865613,
    -0.8756632595908331,
];
const GOLDEN_RCOV: &[f64] = &[15.118271471980519, -2.2219191583655915, 22.721761773534745];
// Re-pinned 2026-10-05 (docs/PLAN.md task 158): the stream runs `preavg_rows
// = 6` at the default `theta`, and the bias term now reads θ from that window,
// `k_n/√n`, as CKP's Eq. 7 defines it, where it read the configured `theta`.
// Re-pinned 2026-10-05 (task 159, D1): the pre-averaged estimate now forms
// CKP's first term, `Ȳ₀`, so a block has `n − k_n + 2` terms and the scale
// counts the terms summed; the entries moved by 4e-4 to 3e-3.
// Re-pinned 2026-10-06 (review CE1): the estimate is rescaled by `1/(1 −
// ψ₁/(2ψ₂k_n²))`, CKP's footnote 1, which at `k_n = 6` is exactly 19/16
// (`ψ₁ = 1`, `ψ₂ = 19/216`); each entry is the last pin times 19/16, to an ulp
// (7.978660961271573, -1.6354544255766184, 20.275069787152503).
const GOLDEN_RCOV_PREAVG: &[f64] = &[9.474659891509994, -1.9421021303722343, 24.0766453722436];
const GOLDEN_DECO: &[f64] = &[
    -0.05328065158114557,
    -0.10464551302436545,
    -0.036211528292942816,
];
const GOLDEN_DECO_LOGLIK: &[f64] = &[-2.2831901919538913, -3.7586606055778677, -1.896095909765855];
// `ewridge`, `ewridge_std` and `lasso` re-frozen 2026-09-14 (docs/PLAN.md
// task 81): the target is null on row 31, and under `target_gaps =
// "own_rows"` its Gram no longer learns that row's features, so rows 45 and
// 59 moved; row 20, before it, moved in the last bits only (`lasso` keeps
// its cross-moments centred now, N2).
const GOLDEN_EW_RIDGE: &[f64] = &[0.23958810892448573, 2.20363868480897, -0.06755913936964057];
const GOLDEN_EW_RIDGE_WINDOWED: &[f64] =
    &[0.23958810892448573, 2.192008777608849, -0.07602329017942382];
const GOLDEN_EW_RIDGE_STD: &[f64] = &[
    0.24074332641726603,
    2.1866449169182474,
    -0.06533729995816817,
];
// The three through-origin signatures, frozen 2026-09-19 (the review's T1)
// on the build with B2 fixed: `ewridge_origin_std` reads the prior.
const GOLDEN_EW_RIDGE_ORIGIN_STD: &[f64] = &[
    -0.06496193224777927,
    1.900894045241888,
    -0.31464368574465446,
];
const GOLDEN_HUBER_ORIGIN_STD: &[f64] = &[
    -0.06544791008639866,
    1.924195894463372,
    -0.31922560537078165,
];
// The three `kalman` signatures, regenerated 2026-10-06 (review 2026-10-05,
// CC4): before a target's first residual its noise is the row's innovation
// squared, where it was the literal 1, and its prior variance is `p0` times
// that first noise, where it was `p0` in the target's units. Each matches
// `tests/reference.py`'s `kalman_ref`, with the same rules, on this stream to
// 1.3e-14; the old values matched the old `kalman_ref` to 6.3e-16.
//
// Every `kalman` signature again for task 211 (2026-10-08): the process noise
// is charged once on each row that observes the target, for the whole clock
// since the last, `Q D²`, so the null target on row 31 takes its clock into
// row 32's charge (`Q 2²`, where two rows charged `Q 1²` each); and a
// standardizing filter sizes each coefficient's prior once its feature's
// scale is usable, from the mean squared innovation over three rows, and
// follows the moments from its first row, with no warm-up. Rows 45 and 59
// moved for all five; row 20, before the null, for the standardizing two
// alone. Before: `kalman_plain` 2.226426987212936 and -0.05944106082324331,
// `kalman_q` 2.1635957406955013 and -0.08050613330500465, `kalman_obs_var`
// 2.2123547676208344 and -0.06081942698886093 (rows 45 and 59). The new
// values match `tests/reference.py`'s `kalman_ref`, rewritten from the module
// doc with the state re-mapped on every row, to 1.8e-15 (the scratchpad's
// `replica_rust_pins.py`).
const GOLDEN_KALMAN_PLAIN: &[f64] = &[0.2534302289666238, 2.2264285943696005, -0.0594422280502202];
const GOLDEN_RLS: &[f64] = &[
    0.24355619170018697,
    2.1844587322364037,
    -0.06708586579330882,
];
// `kalman` and `kalman_revert` again for task 206 (2026-10-08): past the
// standardizer's warm-up, 22 rows by Kish's count of the weights, `b` and `P`
// follow every move of the moments (review round 5, G1). Row 20 is inside the
// warm-up and kept its bits; rows 45 and 59 were 1.9517466851899055 and
// 0.026207278510925018 (`kalman`), 1.153918160782199 and 0.028098479936384025
// (`kalman_revert`). `tests/reference.py`'s `kalman_ref`, with the warm-up and
// the change of coordinates by matrix products, gives the new ones to 1.4e-14.
// Task 211 (above): `kalman` was 0.08051708612823805, 1.9812568625118254 and
// -0.06615560201475972, `kalman_revert` 0.3098186122738434, 1.1519113095475324
// and 0.009601088217273672. `kalman_revert` again for task 214 (2026-10-08):
// a reverting slot's process noise for a gap `D` is `q ((1 − 2^(−D/r)) /
// θ)²`, bounded, where it was `q D²`; row 40's gap of 25 is three of slope
// 2's half-lives of 8. `kalman_ref`, rewritten from the module doc, gives
// the new ones to 8.4e-16, and the old ones, 0.4810076582323497,
// 1.2730031757113542 and 0.13848333209268554, under the old charge to 1.8e-15.
const GOLDEN_KALMAN: &[f64] = &[
    0.22526048935963278,
    2.2330040098824835,
    -0.06612869061558382,
];
const GOLDEN_KALMAN_REVERT: &[f64] = &[0.4810352559516703, 1.1575926629367428, 0.16244017811259087];
const GOLDEN_LASSO: &[f64] = &[
    0.25359037757905656,
    2.1511829817060866,
    -0.06380197762074491,
];
const GOLDEN_HUBER: &[f64] = &[
    0.24786900553362573,
    2.2047028651324143,
    -0.06446675716780813,
];
// Regenerated 2026-09-26 (review, G2): the quantile nudge is bounded by the
// row's leverage, which binds on this stream's early rows; the three values
// moved by 1.4e-3, 1.3e-3 and 4e-4 of themselves, and the QuantReg oracles
// in `robust.rs` hold. Regenerated again 2026-10-05 (review, TC1b): the
// leverage is the row's full one against the band system, where it was the
// Gram's diagonal; the early rows' band Gram is not diagonal, so the bound
// they bind moved, and the three values by 5.1e-4, 4.6e-4 and 1.5e-3 of
// themselves. The QuantReg oracles hold.
const GOLDEN_QUANTILE: &[f64] = &[0.25710455741995175, 2.226788224665792, -0.02104718549249386];
// Regenerated for the code review's C24 (2026-09-15), this and the next:
// under a half-life the proximal term is a decayed sum of its own, where `n`
// was decayed inside its square root and every coefficient shrank.
// Both moved on 2026-09-29 (docs/PLAN.md task 115 (d)): under a half-life the
// penalties now age with the sums, so the fit no longer shrinks between rows.
const GOLDEN_FTRL_SQUARED: &[f64] = &[
    0.31755217793601365,
    1.8450090309450071,
    -0.05909108310584128,
];
const GOLDEN_FTRL: &[f64] = &[0.4937166166955374, 0.5899321334553916, 0.45251642778855805];
// Regenerated for docs/PLAN.md task 74 (2026-09-08): `standardize`
// standardises against the moments with the row admitted, so every
// prediction of this scaled fit moved. Again for task 195 (2026-10-07):
// `huber_delta` is in units of the residual's EW std (U1), so the cut of 0.5
// is 0.5·s where it was 0.5 of the target; a replica of the docstring's
// recursion -- the scaler with the row admitted, `s²` the EW mean of the
// squared out-of-sample residuals -- gave those three to 1e-15. Again for
// task 206 (2026-10-08): past the scaler's warm-up, 22 rows by Kish's count of
// the weights, the coefficients are held in the caller's units and each step
// mapped into them by its own row's scaler (review round 5, G1). Row 20 is
// inside the warm-up and kept its bits; rows 45 and 59 were
// 0.8740905120546256 and -0.026418737010923815. `tests/reference_paths.py`'s
// `sgd_ref` under `standardize`, the moments by their definition, gives the
// new ones to 1.3e-17.
const GOLDEN_SGD: &[f64] = &[
    -9.140859901712872e-5,
    0.9626546015542387,
    -0.09404195296589056,
];
const GOLDEN_SGD_SQUARED: &[f64] = &[
    0.31727038792368356,
    1.429415537069254,
    -0.007974524370910653,
];
// Regenerated for task 195 (2026-10-07): `eps` is in units of the residual's
// EW std (U1), the tube 0.05·σ where it was 0.05 of the target. Again for
// task 202 (the same day): `eps` is in units of the target's own EW std, the
// tube 0.05·σ_y. `tests/reference_paths.py::pa_ref`, written from the
// docstring with `σ_y²` the EW variance of `y` from its definition, gives
// these to the bit on this stream, and the same recursion with the box and
// the sum projected gives `GOLDEN_PA_BOX`'s to 8e-17.
const GOLDEN_PA: &[f64] = &[0.3512828551618081, 2.136865004465535, -0.062000906653408974];
const GOLDEN_SGD_SIMPLEX: &[f64] = &[0.5169094734826561, 1.109247996359838, -0.020146022626397198];
const GOLDEN_PA_BOX: &[f64] = &[0.5262123538733023, 1.6796740252199436, 0.1566262229378625];
// Re-frozen 2026-09-13: a null target is transparent to `holt` now -- the
// next observed row forecasts over the clock since the last one -- where the
// level stood still across it (review 2026-09-12, C22). Row 20 comes before
// the stream's null at row 31 and did not move.
// Regenerated for the code review's S29/S30 (2026-09-15): level and trend
// are weighted means, whose gains start at 1 where the textbook's were fixed.
const GOLDEN_HOLT: &[f64] = &[
    -1.5909733825311503,
    0.07655111549786203,
    0.17534943347133736,
];
const GOLDEN_SEQTEST: &[f64] = &[-0.6443570163905132, -1.086393303225433, -0.9577414997686987];
const GOLDEN_SEQTEST_NEG: &[f64] = &[
    -0.9917118216489557,
    -0.9917118216489557,
    -0.9917118216489557,
];
const GOLDEN_EW_COV: &[f64] = &[
    -0.3469363807058677,
    -0.1331528573613194,
    -0.013968574548668105,
];
const GOLDEN_MARGINAL: &[f64] = &[
    -0.658828097450141,
    -1.2763707481988686,
    16.974118022314695,
    -0.575404661163831,
    -1.0280783998000982,
    21.444816431035854,
    -0.4441310076916231,
    -0.7754079187908046,
    26.59144641744387,
];
const GOLDEN_KMEANS: &[f64] = &[0.5834101526098997, 0.8292779994915372, 0.923506490724854];
const GOLDEN_KMEANS_CLUSTER: &[f64] = &[2.0, 0.0, 2.0];
const GOLDEN_KMEANS_FIRST: &[f64] = &[1.8013837727258295, 2.935845007414875, 1.351429916256865];
const GOLDEN_MICRO: &[f64] = &[0.3004267420269594, 0.8769688913202295, 1.4153400286799591];
const GOLDEN_MICRO_CLUSTER: &[f64] = &[1.0, 6.0, 6.0];
const GOLDEN_MICRO_ID: &[f64] = &[0.0, 3.0, 0.0];
const GOLDEN_EW_CLASS: &[f64] = &[
    0.7905217777831283,
    2.3962695799139553e-7,
    0.8870452595331249,
];
const GOLDEN_EW_CLASS_CLASS: &[f64] = &[0.0, 1.0, 0.0];
const GOLDEN_EW_CLASS_SHARED: &[f64] =
    &[0.10145341892546339, 0.9999995909585125, 0.22525421810089402];
const GOLDEN_EW_CLASS_DIAGONAL: &[f64] =
    &[0.9347724076192444, 0.004084196343630897, 0.8672014742687666];
const GOLDEN_BOCPD_GAUSSIAN: &[f64] = &[
    0.056219033365587236,
    0.05468567337937875,
    0.050622861481265845,
];
const GOLDEN_BOCPD_ROBUST: &[f64] = &[0.05647515197223256, 0.05942243315748, 0.05285901703185968];
const GOLDEN_CORRCHANGE_SEQUENTIAL: &[f64] = &[
    0.8316281190223411,
    0.6909945046453919,
    0.8717864537157914,
    1.088107497209456,
    1.122362938000111,
    0.8341873826345729,
    0.8805008348308133,
    0.9496344256708152,
    0.32833736423817417,
    0.07921916384339617,
    0.2074817753847168,
    0.30015612602690156,
    0.49240780002347756,
    0.5101683171994119,
    0.5007703521180233,
    0.755536012901163,
    0.8179783419406013,
    1.073767937120612,
    0.7304393221230613,
];
const GOLDEN_EW_CLASS_WINDOWED: &[f64] = &[0.7905217777831282, 1.0, 0.8827845147955854];
const GOLDEN_EW_COV_LAGS: &[f64] = &[
    0.42675674146655357,
    -0.2941126260468523,
    -0.23312910489114877,
];
const GOLDEN_EW_COV_PCA: &[f64] = &[
    -0.31399435832348843,
    1.2149616979919209,
    -0.17449357703751237,
];
const GOLDEN_EW_COV_WINDOWED: &[f64] =
    &[-0.3469363807058677, 0.06020754802385295, 0.0899414616598929];
const GOLDEN_EW_RIDGE_BLOCKED: &[f64] =
    &[0.23958810892448573, 2.20363868480897, -0.07344251215567098];
const GOLDEN_EW_RIDGE_BY_WEIGHT: &[f64] =
    &[0.23728591098765225, 2.20363868480897, -0.06946714940555972];
const GOLDEN_EW_RIDGE_ON_THE_CLOCK: &[f64] =
    &[0.2424004251925553, 2.2241093236471907, -0.06755913936964053];
const GOLDEN_EW_RIDGE_PAIRWISE: &[f64] =
    &[0.23958810892448573, 2.1830039922943563, -0.0678454894530192];
const GOLDEN_EW_RIDGE_SESSION_SHRINK: &[f64] = &[
    0.23958810892448573,
    2.2036034970611253,
    -0.06773477106376563,
];
const GOLDEN_EW_RIDGE_SET_BOTH: &[f64] =
    &[0.23958810892448573, 2.20363868480897, -0.06755913936964053];
const GOLDEN_EW_RIDGE_SET_ONE: &[f64] =
    &[0.6226036377649254, 1.6401144565737151, -0.12394041898406416];
const GOLDEN_HMM_TVTP: &[f64] = &[
    -1.0643298399245595,
    -1.8091334719187875,
    -1.0356264682715979,
];
// Both again for task 211 (`GOLDEN_KALMAN_PLAIN`'s note).
const GOLDEN_KALMAN_OBS_VAR: &[f64] =
    &[0.24438391241807816, 2.212385812063775, -0.06082198940989911];
const GOLDEN_KALMAN_Q: &[f64] = &[0.2624940862221372, 2.16361003592116, -0.08052213955730514];
const GOLDEN_KMEANS_FARTHEST: &[f64] =
    &[0.5834101526098997, 0.8292779994915372, 1.0040104663998375];
const GOLDEN_KMEANS_FLOOR: &[f64] = &[0.5834101526098997, 1.1754072514037952, 0.7869506375655098];
const GOLDEN_KMEANS_KMEANSPP: &[f64] =
    &[0.5781848335106177, 0.8471048961306411, 0.9945561598807307];
const GOLDEN_KMEANS_RAW: &[f64] = &[0.3137395740537139, 0.48742765492297646, 0.5466642991217668];
const GOLDEN_LASSO_SELECT: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0];
const GOLDEN_LASSO_WINDOWED: &[f64] = &[
    0.25359037757905656,
    2.0887627806236106,
    -0.07423130166085738,
];
const GOLDEN_MARGINAL_BINS: &[f64] = &[
    0.2835419144684436,
    -0.14863035120403767,
    0.8850728014130631,
    0.27125897159512596,
    -0.5516946583900888,
    0.37497034342253555,
    0.2335678702944847,
    -0.5516946583900888,
    0.21387162829564357,
];
const GOLDEN_MARGINAL_LAGS: &[f64] = &[
    16.268396176259806,
    -3.308052338372048,
    -0.043937176942903604,
    19.5919219948599,
    -2.9508446537440984,
    0.10989778139661917,
    26.18686633083888,
    -2.437874678623021,
    -0.012455077577307155,
];
const GOLDEN_MARGINAL_WINDOWED: &[f64] = &[
    -0.658828097450141,
    -1.2763707481988686,
    16.974118022314695,
    -0.4279581919553364,
    -0.6817858147667967,
    5.45319450047077,
    -0.3564749745323196,
    -0.6156300333085452,
    16.500104235927783,
];
const GOLDEN_MICRO_CLOCK: &[f64] = &[0.3004267420269594, 0.8769688913202294, 1.4507685900691734];
const GOLDEN_MICRO_FLOOR: &[f64] = &[0.3004267420269594, 1.161118351411367, 0.8126582237424175];
const GOLDEN_RCOV_PREAVG_PSD: &[f64] =
    &[11.307541539663788, -0.7389071190718174, 18.691806669414845];
