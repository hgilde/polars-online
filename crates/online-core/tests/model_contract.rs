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
        coef0: None,
        min_periods: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
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
        coef0: None,
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
        max_cd_iters: 100,
        cd_tol: 1e-10,
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
        standardize: true,
    }
}

#[test]
fn kalman() {
    let cfg = kalman_cfg();
    let r = probe_with(Kalman::new(cfg).unwrap(), 2, Some(&Kalman::n_eff));
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
        stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
        min_periods: 3.0,
        precision_prior: None,
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
    let rows = bounded_script(targets);
    let mut model = build();
    let mut twin = build();
    let kind = model.state().model.kind();
    let n = rows.len();
    let mut seen = [false, false];
    let mut worst = 0.0f64;
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
            }
        }
    }
    match how {
        Recovery::Twin(_) => {
            eprintln!("{kind}: worst relative disagreement with the twin {worst:.2e}")
        }
        Recovery::Tube(_) => eprintln!("{kind}: worst relative error of model or twin {worst:.2e}"),
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
    recovers_from_bounded_extremes(|| Holt::new(holt_cfg()).unwrap(), 2, Recovery::Twin(1e-9));
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
        let p = m.predict(&x, d);
        let step = m.step(&x, &y, d, w);
        same_step(kind, i, &p, &step);
        if step.pred.iter().all(|v| v.is_finite()) {
            ready += 1;
        }
    }
    // The stream must actually have exercised the ready path, or the test
    // would pass on NaN == NaN alone.
    assert!(
        ready > 300,
        "{kind}: only {ready} rows had every slot ready"
    );
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
        let mut cfg = kalman_cfg();
        cfg.standardize = standardize;
        predict_is_the_step_without_the_step(move || Kalman::new(cfg.clone()).unwrap(), 2, false);
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
fn ew_cov_predict_is_the_step() {
    predict_is_the_step_without_the_step(|| EwCovModel::new(ew_cov_model_cfg()).unwrap(), 0, false);
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
