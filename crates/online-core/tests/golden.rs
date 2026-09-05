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
//! numpy references in `tests/reference.py` for `ew_ridge`, `rls`, `kalman`,
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
    if std::env::var("PRINT_GOLDEN").is_ok() {
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
        add_intercept: true,
        decay: Decay::Halflife(20.0),
        ridge: vec![ridge],
        feature_sets: vec![],
        standardize,
        ridge_decay: false,
        session_shrink: None,
        long_halflife: None,
        coef0: None,
        min_periods: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
    }
}

fn robust_cfg(loss: RobustLoss, standardize: bool) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(20.0),
        loss,
        ridge: 1e-4,
        standardize,
        min_periods: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        quantile_eps: 1e-3,
    }
}

#[test]
fn ew_ridge_golden() {
    let mut m = EwRidge::new(ewridge_cfg(false, 1e-4)).unwrap();
    check("ew_ridge", &signature(&mut m, 0), GOLDEN_EW_RIDGE);
}

#[test]
fn ew_ridge_standardized_golden() {
    let mut m = EwRidge::new(ewridge_cfg(true, 0.01)).unwrap();
    check("ew_ridge_std", &signature(&mut m, 0), GOLDEN_EW_RIDGE_STD);
}

#[test]
fn rls_golden() {
    let mut m = Rls::new(RlsCfg {
        n_features: 2,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(20.0),
        ridge: 0.5,
        coef0: None,
        min_periods: 3.0,
    })
    .unwrap();
    check("rls", &signature(&mut m, 0), GOLDEN_RLS);
}

#[test]
fn kalman_golden() {
    let mut m = Kalman::new(KalmanCfg {
        n_features: 2,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(50.0),
        halflife: vec![f64::INFINITY, 30.0, 100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_periods: 3.0,
        standardize: true,
    })
    .unwrap();
    check("kalman", &signature(&mut m, 0), GOLDEN_KALMAN);
}

#[test]
fn lasso_golden() {
    let mut m = Lasso::new(LassoCfg {
        n_features: 2,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(20.0),
        lasso_path: vec![0.2, 0.02, 0.0],
        l1_ratio: 1.0,
        select_halflife: None,
        min_periods: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        max_cd_iters: 200,
        cd_tol: 1e-12,
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
        add_intercept: true,
        decay: Decay::Halflife(40.0),
        alpha: 0.1,
        beta: 1.0,
        l1: 0.05,
        l2: 1.0,
        min_periods: 3.0,
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
        add_intercept: true,
        decay: Decay::Halflife(40.0),
        alpha: 0.5,
        beta: 1.0,
        l1: 0.05,
        l2: 0.01,
        min_periods: 3.0,
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
        add_intercept: true,
        decay: Decay::Halflife(20.0),
        loss: SgdLoss::Huber { delta: 0.5 },
        learning_rate: 0.05,
        schedule: LearningRate::InvScaling { power: 0.25 },
        l2: 0.01,
        clip_gradient: 1e3,
        scale_features: true,
        min_periods: 3.0,
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
        add_intercept: true,
        decay: Decay::Halflife(20.0),
        loss: SgdLoss::Squared,
        learning_rate: 0.05,
        schedule: LearningRate::Constant,
        l2: 0.0,
        clip_gradient: 1e3,
        scale_features: false,
        min_periods: 3.0,
    })
    .unwrap();
    check("sgd_squared", &signature(&mut m, 0), GOLDEN_SGD_SQUARED);
}

#[test]
fn pa_golden() {
    let mut m = Pa::new(PaCfg {
        n_features: 2,
        n_targets: 1,
        add_intercept: true,
        decay: Decay::Halflife(20.0),
        mode: PaMode::Pa2,
        c: 0.5,
        eps: 0.05,
        min_periods: 3.0,
    })
    .unwrap();
    check("pa", &signature(&mut m, 0), GOLDEN_PA);
}

#[test]
fn holt_golden() {
    let mut m = Holt::new(HoltCfg {
        n_targets: 1,
        level_halflife: 10.0,
        trend_halflife: 40.0,
        min_periods: 3.0,
    })
    .unwrap();
    check("holt", &signature_of(&mut m, 0, false), GOLDEN_HOLT);
}

#[test]
fn ew_cov_golden() {
    // Slots in emission order: mean x0, mean x1, var x0, var x1, corr x0x1.
    // The correlation is the one that reads every accumulator at once.
    let mut m = EwCovModel::new(EwCovCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
        min_periods: 3.0,
        precision_prior: None,
    })
    .unwrap();
    check("ew_cov", &signature(&mut m, 4), GOLDEN_EW_COV);
}

fn kmeans_cfg(rule: SeedRule) -> KMeansCfg {
    KMeansCfg {
        n_features: 2,
        k: 3,
        decay: Decay::Halflife(20.0),
        min_periods: 3.0,
        warm_rows: 12,
        seed_rule: rule,
        seed: 0,
        update_every: 1,
        split_merge: 0.5,
        sm_every: 10,
        dead_frac: 0.05,
        standardize: true,
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
        update_every: 7,
        ..kmeans_cfg(SeedRule::First)
    })
    .unwrap();
    check("kmeans_first", &signature(&mut m, 2), GOLDEN_KMEANS_FIRST);
}

// --- generated; see the module docs ---
const GOLDEN_EW_RIDGE: &[f64] = &[
    0.23958810892448523,
    2.1453177394610767,
    -0.07945442055200784,
];
const GOLDEN_EW_RIDGE_STD: &[f64] = &[0.24074332641726595, 2.129047638104942, -0.07712184691335192];
const GOLDEN_RLS: &[f64] = &[
    0.24355619170018697,
    2.1844587322364037,
    -0.06708586579330882,
];
const GOLDEN_KALMAN: &[f64] = &[-0.07791204926408574, 2.037785637299904, 0.01654269850933259];
const GOLDEN_LASSO: &[f64] = &[0.25359037757905634, 2.094156488785958, -0.07537065924040198];
const GOLDEN_HUBER: &[f64] = &[
    0.24786900553362573,
    2.2047028651324143,
    -0.06446675716780813,
];
const GOLDEN_QUANTILE: &[f64] = &[0.2951076807653745, 2.232303167920436, -0.03927577244973528];
const GOLDEN_FTRL_SQUARED: &[f64] = &[
    0.31690964540626376,
    1.6128031168738046,
    -0.053001920184771734,
];
const GOLDEN_FTRL: &[f64] = &[0.4944157427243535, 0.5720078133207852, 0.4641448801691502];
const GOLDEN_SGD: &[f64] = &[-0.11253411046976192, 1.168224926369749, -0.0631738574598267];
const GOLDEN_SGD_SQUARED: &[f64] = &[
    0.31727038792368356,
    1.429415537069254,
    -0.007974524370910653,
];
const GOLDEN_PA: &[f64] = &[
    0.35403211576284754,
    2.1379339164214444,
    -0.061839814133866494,
];
const GOLDEN_HOLT: &[f64] = &[0.6940554404209057, 0.5781242794831807, 0.2548083372371531];
const GOLDEN_EW_COV: &[f64] = &[
    -0.3469363807058677,
    -0.1331528573613194,
    -0.013968574548668105,
];
const GOLDEN_KMEANS: &[f64] = &[0.5834101526098997, 0.8292779994915372, 0.923506490724854];
const GOLDEN_KMEANS_CLUSTER: &[f64] = &[2.0, 0.0, 2.0];
const GOLDEN_KMEANS_FIRST: &[f64] = &[1.8013837727258295, 2.935845007414875, 1.351429916256865];
