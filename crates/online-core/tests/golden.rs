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
//! output, and that implementation is independently verified against numpy
//! references for every model in `tests/test_oracles.py`. This file locks in
//! numbers that have already been checked against something else.
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
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in stream().into_iter().enumerate() {
        let step = model.step(&x, &y, d, w);
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
