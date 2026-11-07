//! `window_could_refuse` answers what the window's shadow answers, for every
//! model with a window, without copying the ring: the bank asks it of every
//! chunk and forms a shadow only where it says yes, so a wrong "no" would let
//! a chunk past a refusing budget that the shadow would have refused whole.

use online_core::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

const K: usize = 3;
const WINDOW: f64 = 40.0;

fn decay() -> Decay {
    Decay::Halflife(30.0)
}

/// A model's name and how to build it fresh.
type Make = (&'static str, Box<dyn Fn() -> Box<dyn Probe>>);

fn models() -> Vec<Make> {
    vec![
        (
            "ewridge",
            Box::new(|| {
                Box::new(
                    EwRidge::new(EwRidgeCfg {
                        n_features: K,
                        n_targets: 2,
                        fit_intercept: true,
                        decay: decay(),
                        ridge: vec![1e-6],
                        feature_sets: vec![],
                        standardize: false,
                        ridge_scale: false,
                        session_shrink: None,
                        long_half_life: None,
                        coef_prior: None,
                        min_weight: 4.0,
                        solve_every: 0.0,
                        max_rows_between_solves: 1,
                        solve_share: None,
                        gram_block_rows: 0,
                        target_gaps: TargetGaps::OwnRows,
                        window: Some(WINDOW),
                        window_every: None,
                        max_rows_between_snapshots: None,
                    })
                    .unwrap(),
                ) as Box<dyn Probe>
            }),
        ),
        (
            "lasso",
            Box::new(|| {
                Box::new(
                    Lasso::new(LassoCfg {
                        min_weight_holds_learning: false,
                        n_features: K,
                        n_targets: 2,
                        fit_intercept: true,
                        decay: decay(),
                        lasso_path: vec![0.1, 0.0],
                        l1_ratio: 1.0,
                        select_half_life: None,
                        min_weight: 4.0,
                        target_min_weight: Vec::new(),
                        solve_every: 0.0,
                        max_rows_between_solves: 1,
                        solve_share: None,
                        window: Some(WINDOW),
                        window_every: None,
                        max_rows_between_snapshots: None,
                        max_iter: 100,
                        tol: 1e-10,
                        target_gaps: TargetGaps::OwnRows,
                    })
                    .unwrap(),
                )
            }),
        ),
        (
            "marginal",
            Box::new(|| {
                Box::new(
                    Marginal::new(MarginalCfg {
                        n_features: K,
                        n_targets: 2,
                        decay: decay(),
                        min_weight: vec![3.0; 2],
                        lags: vec![],
                        serial_rule: None,
                        cross_lags: None,
                        bins: None,
                        feature_moments: FeatureMomentLayout::PerTarget,
                        window: Some(WINDOW),
                        window_every: None,
                        max_rows_between_snapshots: None,
                    })
                    .unwrap(),
                )
            }),
        ),
        (
            "ew_cov",
            Box::new(|| {
                Box::new(
                    EwCovModel::new(EwCovCfg {
                        bias: false,
                        n_features: K,
                        decay: decay(),
                        stats: vec![EwCovStat::Mean, EwCovStat::Var],
                        min_weight: 3.0,
                        precision_prior: None,
                        mahal_quantiles: Vec::new(),
                        pca: 0,
                        pca_every: 0.0,
                        max_rows_between_pca: u32::MAX,
                        lags: vec![],
                        window: Some(WINDOW),
                        window_every: None,
                        max_rows_between_snapshots: None,
                    })
                    .unwrap(),
                )
            }),
        ),
        (
            "ew_class",
            Box::new(|| {
                Box::new(
                    EwClass::new(EwClassCfg {
                        n_features: K,
                        n_classes: 2,
                        decay: decay(),
                        min_weight: 3.0,
                        covariance: Covariance::Diagonal,
                        precision_prior: 0.1,
                        window: Some(WINDOW),
                        window_every: None,
                        max_rows_between_snapshots: None,
                    })
                    .unwrap(),
                )
            }),
        ),
    ]
}

/// What the test reads of a model, as a trait object.
trait Probe {
    fn targets(&self) -> usize;
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d: f64);
    fn budget(&mut self, b: Option<WindowBudget>);
    fn cheap(&self, rows: usize) -> bool;
    fn shadow(&self, rows: usize) -> bool;
}

impl<M: OnlineModel> Probe for M {
    fn targets(&self) -> usize {
        // `ew_class` reads its label from the first target value.
        self.n_targets()
            .max(usize::from(self.state().model.kind() == "ew_class"))
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d: f64) {
        OnlineModel::step(self, x, y, d, 1.0);
    }

    fn budget(&mut self, b: Option<WindowBudget>) {
        self.set_window_budget(b);
    }

    fn cheap(&self, rows: usize) -> bool {
        self.window_could_refuse(rows)
    }

    fn shadow(&self, rows: usize) -> bool {
        self.window_shadow()
            .expect("a windowed model")
            .could_refuse(rows)
    }
}

#[test]
fn window_could_refuse_is_the_shadows_answer() {
    // Budgets from a few snapshots to many, refusing and thinning, and none.
    let budgets = [
        None,
        Some(WindowBudget::Thin(0.004)),
        Some(WindowBudget::Refuse(0.0005)),
        Some(WindowBudget::Refuse(0.004)),
        Some(WindowBudget::Refuse(0.05)),
    ];
    let (mut yes, mut no) = (0, 0);
    for (name, make) in models() {
        for budget in budgets {
            let mut m = make();
            m.budget(budget);
            let mut s = 349u64;
            for i in 0..120 {
                // An empty ring too: asked before the first row.
                for rows in [0, 1, 7, 60, 5_000] {
                    let (cheap, shadow) = (m.cheap(rows), m.shadow(rows));
                    assert_eq!(cheap, shadow, "{name} {budget:?} row {i} rows {rows}");
                    if cheap {
                        yes += 1;
                    } else {
                        no += 1;
                    }
                }
                let x: Vec<f64> = (0..K).map(|_| lcg(&mut s)).collect();
                let y: Vec<Option<f64>> =
                    [Some((i % 2) as f64), Some(x[0] + 0.1 * lcg(&mut s))][..m.targets()].to_vec();
                m.step(&x, &y, if i == 0 { 0.0 } else { 1.0 });
            }
        }
    }
    assert!(
        yes > 1_000 && no > 1_000,
        "both answers seen: {yes} yes, {no} no"
    );
}

/// A model with no window has no ring to refuse a row: it answers no for
/// any number of rows, before its first row and after many, whether it is a
/// kind that never keeps a window or one configured without one, and under
/// a refusing budget too.
#[test]
fn a_model_without_a_window_could_refuse_nothing() {
    fn check<M: OnlineModel>(name: &str, mut m: M, targets: usize) {
        m.set_window_budget(Some(WindowBudget::Refuse(0.0005)));
        assert!(m.window_shadow().is_none(), "{name}: no shadow");
        let mut s = 351u64;
        for i in 0..60 {
            for rows in [0, 1, 7, 5_000] {
                assert!(!m.window_could_refuse(rows), "{name} row {i} rows {rows}");
            }
            let x: Vec<f64> = (0..K).map(|_| lcg(&mut s)).collect();
            let y: Vec<Option<f64>> = (0..targets).map(|_| Some(lcg(&mut s))).collect();
            m.step(if name == "holt" { &[] } else { &x }, &y, 1.0, 1.0);
        }
    }
    check(
        "pa",
        Pa::new(PaCfg {
            n_features: K,
            n_targets: 2,
            fit_intercept: true,
            decay: decay(),
            mode: PaMode::Pa1,
            c: 1.0,
            eps: 0.1,
            min_weight: 3.0,
            constraint: None,
            standardize: true,
        })
        .unwrap(),
        2,
    );
    check(
        "holt",
        Holt::new(HoltCfg {
            n_targets: 1,
            level_half_life: 2.0,
            trend_half_life: 4.0,
            min_weight: 2.0,
            trend: true,
        })
        .unwrap(),
        1,
    );
}
