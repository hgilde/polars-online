//! An explicit solve cadence's first period on the default weight-share
//! schedule ([`super::FirstPeriod`]; task 278, the user's decision of
//! 2026-10-26, "Default cadence first"), held on each model with a solve
//! cadence -- `ewridge`, `lasso`, `huber`, `quantile` -- against the
//! definition written out here.
//!
//! The rows are unit-weight and undecayed (`half_life = inf`), stepped with
//! no stamps, the first a step of 0 and the rest of 1, so the clock since
//! the model's start is the row's index and the weight held is its count:
//! the default schedule solves a row once the rows since the last solve
//! reach `ln 2 / 50` of the rows held, exactly. A solve is read from the
//! state: it leaves no weight learned since it.

use crate::{
    Decay, EwRidge, EwRidgeCfg, HuberScale, Lasso, LassoCfg, ModelState, OnlineModel, Robust,
    RobustCfg, RobustLoss, TargetGaps,
};

const K: usize = 2;
const N: usize = 140;

/// The three models with a solve cadence, as one type (`OnlineModel` is
/// not object safe).
enum Model {
    Ridge(EwRidge),
    Lasso(Lasso),
    Robust(Robust),
}

macro_rules! each {
    ($m:expr, $v:ident => $body:expr) => {
        match $m {
            Model::Ridge($v) => $body,
            Model::Lasso($v) => $body,
            Model::Robust($v) => $body,
        }
    };
}

/// A model with a solve cadence, at `min_weight`, under `solve_every` and
/// a row cap, `min_weight` withholding the output alone (the form a spec
/// builds).
fn model(kind: &str, min_weight: f64, every: f64, cap: u32) -> Model {
    let decay = Decay::Halflife(f64::INFINITY);
    match kind {
        "ewridge" => {
            let mut m = EwRidge::new(EwRidgeCfg {
                n_features: K,
                n_targets: 1,
                fit_intercept: true,
                decay,
                ridge: vec![1e-8],
                feature_sets: vec![],
                standardize: false,
                ridge_scale: false,
                coef_prior: None,
                session_shrink: None,
                long_half_life: None,
                min_weight,
                solve_every: every,
                max_rows_between_solves: cap,
                solve_share: None,
                gram_block_rows: 0,
                target_gaps: TargetGaps::OwnRows,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
            })
            .unwrap();
            m.set_min_weight_output_only(true);
            Model::Ridge(m)
        }
        "lasso" => Model::Lasso(
            Lasso::new(LassoCfg {
                min_weight_holds_learning: false,
                n_features: K,
                n_targets: 1,
                fit_intercept: true,
                decay,
                lasso_path: vec![0.01, 0.0],
                l1_ratio: 1.0,
                select_half_life: None,
                min_weight,
                target_min_weight: Vec::new(),
                solve_every: every,
                max_rows_between_solves: cap,
                solve_share: None,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
                target_gaps: TargetGaps::OwnRows,
                max_iter: 200,
                tol: 1e-12,
            })
            .unwrap(),
        ),
        _ => Model::Robust(
            Robust::new(RobustCfg {
                n_features: K,
                n_targets: 1,
                fit_intercept: true,
                decay,
                loss: if kind == "huber" {
                    RobustLoss::Huber { delta: 1.5 }
                } else {
                    RobustLoss::Quantile { tau: 0.5 }
                },
                ridge: 1e-8,
                standardize: false,
                min_weight,
                solve_every: every,
                max_rows_between_solves: cap,
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
            })
            .unwrap(),
        ),
    }
}

const KINDS: [&str; 4] = ["ewridge", "lasso", "huber", "quantile"];

/// Row `i`: two features and a target from integer arithmetic alone.
fn row(i: usize) -> ([f64; K], f64) {
    let a = ((i * 37) % 101) as f64 / 101.0 - 0.5;
    let b = ((i * 59) % 97) as f64 / 97.0 - 0.5;
    let noise = ((i * 13) % 17) as f64 / 170.0;
    ([a, b], 1.5 * a - 0.75 * b + 0.25 + noise)
}

/// The model's state as JSON, and the value of `key` in it.
fn read(m: &Model, key: &str) -> serde_json::Value {
    let s = each!(m, m => m.state());
    let v = match &s.model {
        ModelState::EwRidge(inner) => serde_json::to_value(inner),
        ModelState::Lasso(inner) => serde_json::to_value(inner),
        ModelState::Robust(inner) => serde_json::to_value(inner),
        _ => unreachable!("a model with a solve cadence"),
    }
    .unwrap();
    v[key].clone()
}

/// The rows the model solved on, each read after the row from what the
/// solve leaves (no weight learned since it), the predictions it made,
/// and whether the mark was still set after each row.
fn run(m: &mut Model) -> (Vec<usize>, Vec<u64>, Vec<bool>) {
    let (mut solved, mut preds, mut marks) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..N {
        let (x, y) = row(i);
        let d = if i == 0 { 0.0 } else { 1.0 };
        // The fit's own prediction, whatever `min_weight` withholds.
        let own = each!(&mut *m, m => m.predict_with_floor(&x, &[Some(y)], d, 0.0));
        preds.extend(own.pred.iter().map(|p| p.to_bits()));
        each!(&mut *m, m => m.step(&x, &[Some(y)], d, 1.0));
        if read(m, "weight_since_solve").as_f64() == Some(0.0) {
            solved.push(i);
        }
        marks.push(read(m, "first_period_by_share").as_bool() == Some(true));
    }
    (solved, preds, marks)
}

/// The definition: the default schedule's rows before the period's `end`
/// (a solve once the rows since the last reach `ln 2 / 50` of the rows
/// held), then `end` and every `step` rows after it.
fn definition(end: usize, step: usize) -> Vec<usize> {
    let share = std::f64::consts::LN_2 / 50.0;
    let (mut since, mut rows) = (0.0_f64, Vec::new());
    for i in 0..end {
        since += 1.0;
        if since >= share * (i + 1) as f64 {
            rows.push(i);
            since = 0.0;
        }
    }
    rows.extend((end..N).step_by(step));
    rows
}

/// Under `solve_every = 50` the period ends on row 50, the first whose
/// clock from the start (its index) reaches 50; under a row cap of 50 on
/// the 50th row, 49. Before it the model solves on the default schedule
/// -- every one of the first 72 rows here -- and the cadence's clock and
/// count run from the model's start; on it the mark clears, and from it
/// the cadence alone solves. Without the mark, the state a spec built
/// before the rule, the model solves at its floor and then on the cadence
/// alone: three rows or fewer before row 49.
#[test]
fn an_explicit_cadences_first_period_solves_on_the_default_schedule() {
    for kind in KINDS {
        for (every, cap, end) in [(50.0, u32::MAX, 50), (1e9, 50, 49)] {
            let what = format!("{kind} under solve_every {every}, a cap of {cap}");
            let mut m = model(kind, 3.0, every, cap);
            each!(&mut m, m => m.set_first_period_by_share(true));
            let (solved, _, marks) = run(&mut m);
            assert_eq!(solved, definition(end, 50), "{what}");
            assert!(solved.len() > 50, "{what}: {solved:?}");
            let cleared = marks.iter().position(|&on| !on);
            assert_eq!(
                cleared,
                Some(end),
                "{what}: the mark clears on the period's end"
            );
            let mut old = model(kind, 3.0, every, cap);
            each!(&mut old, m => m.set_first_period_by_share(false));
            let (before, _, _) = run(&mut old);
            let early = before.iter().filter(|&&i| i < 49).count();
            assert!((1..=3).contains(&early), "{what}, unmarked: {before:?}");
        }
    }
}

/// The period's schedule reads the weight learned and held, nothing of
/// `min_weight`: at `min_weight` 3, 30 and 1e9 every solve falls on the
/// same row and the fit's own prediction is the same on every row, to the
/// bit.
#[test]
fn the_first_period_is_the_same_at_any_min_weight() {
    for kind in KINDS {
        let runs: Vec<_> = [3.0, 30.0, 1e9]
            .into_iter()
            .map(|floor| {
                let mut m = model(kind, floor, 50.0, u32::MAX);
                each!(&mut m, m => m.set_first_period_by_share(true));
                run(&mut m)
            })
            .collect();
        for r in &runs[1..] {
            assert_eq!(r.0, runs[0].0, "{kind}: the solve rows");
            assert_eq!(r.1, runs[0].1, "{kind}: the fit's predictions");
        }
    }
}

/// The mark is taken only where the schedule is an explicit clock: not
/// under a weight share (the default cadence), whose output it leaves to
/// the bit, nor under `solve_every <= 0`, which solves every row.
#[test]
fn the_mark_is_taken_only_by_an_explicit_clock() {
    assert!(super::explicit_cadence(None, 50.0));
    assert!(!super::explicit_cadence(Some(0.01), 50.0));
    assert!(!super::explicit_cadence(None, 0.0));
    assert!(!super::explicit_cadence(None, -1.0));
    for kind in KINDS {
        for every in [0.0, -1.0] {
            let mut m = model(kind, 3.0, every, u32::MAX);
            each!(&mut m, m => m.set_first_period_by_share(true));
            assert_eq!(read(&m, "first_period_by_share"), false, "{kind}");
        }
        let mut shared = model(kind, 3.0, 50.0, u32::MAX);
        each!(&mut shared, m => m.set_solve_share(Some(crate::DEFAULT_SOLVE_SHARE)));
        let plain = run(&mut shared);
        let mut marked = model(kind, 3.0, 50.0, u32::MAX);
        each!(&mut marked, m => m.set_solve_share(Some(crate::DEFAULT_SOLVE_SHARE)));
        each!(&mut marked, m => m.set_first_period_by_share(true));
        assert_eq!(read(&marked, "first_period_by_share"), false);
        assert_eq!(
            run(&mut marked),
            plain,
            "{kind}: the default schedule moved"
        );
    }
}

/// A cadence that never comes due ([`crate::SOLVE_NEVER`]): `inf`, and any
/// finite value at or past the threshold, `1e18` among them, ask for no
/// solves, so the mark is not taken and there is no first period. The model
/// solves only where its own first solves and its row cap say, the same
/// rows and predictions, to the bit, at `inf` and `1e18`, with the mark set
/// or not; a row cap beside it still solves, as it did before the rule.
/// The largest value below the threshold is a cadence: it takes the mark,
/// and its first period, which this stream never ends, solves on the
/// default schedule throughout.
#[test]
fn a_cadence_past_the_threshold_never_solves_and_has_no_first_period() {
    assert!(crate::solve_never(f64::INFINITY) && crate::solve_never(1e18));
    let below = f64::from_bits(crate::SOLVE_NEVER.to_bits() - 1);
    assert!(!crate::solve_never(below));
    assert!(!super::explicit_cadence(None, f64::INFINITY));
    assert!(!super::explicit_cadence(None, 1e18));
    assert!(super::explicit_cadence(None, below));
    for kind in KINDS {
        for cap in [u32::MAX, 50] {
            let runs: Vec<_> = [(f64::INFINITY, true), (1e18, true), (f64::INFINITY, false)]
                .into_iter()
                .map(|(every, mark)| {
                    let mut m = model(kind, 3.0, every, cap);
                    each!(&mut m, m => m.set_first_period_by_share(mark));
                    assert_eq!(read(&m, "first_period_by_share"), false, "{kind}");
                    run(&mut m)
                })
                .collect();
            assert_eq!(runs[1], runs[0], "{kind}, cap {cap}: 1e18 is inf");
            assert_eq!(runs[2], runs[0], "{kind}, cap {cap}: the mark moved inf");
            let solved = &runs[0].0;
            let early = solved.iter().filter(|&&i| i < 49).count();
            assert!((1..=3).contains(&early), "{kind}, cap {cap}: {solved:?}");
            if cap == 50 {
                // The cap's solves, 50 rows apart from the model's first.
                let last = *solved.last().unwrap();
                assert!(solved.contains(&(last - 50)), "{kind}: {solved:?}");
            } else {
                assert!(solved.iter().all(|&i| i < 49), "{kind}: {solved:?}");
            }
        }
        let mut m = model(kind, 3.0, below, u32::MAX);
        each!(&mut m, m => m.set_first_period_by_share(true));
        let (solved, _, marks) = run(&mut m);
        assert!(
            marks.iter().all(|&on| on),
            "{kind}: the period never ends here"
        );
        assert_eq!(
            solved,
            definition(N, 1),
            "{kind}: the default schedule throughout"
        );
    }
}
