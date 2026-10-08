//! The models the frozen state fixtures (`tests/state_fixtures.rs`) and the
//! layout checks (`tests/state_repairs.rs`) build: at least one per
//! [`ModelState`] variant, each configured to keep what its variant can keep
//! -- a window, lags, bins, a warm-up buffer, far rows, held moments -- and
//! a stream for each, read in two parts: the rows before the save, and the
//! rows its continuation is checked on (docs/PLAN.md task 198).
//!
//! The streams come from an integer generator and arithmetic alone (no libm:
//! a stream made with one would differ by platform), and every number of a
//! continuation is frozen beside the state, so the generator could change
//! without moving a fixture.

#![allow(dead_code)]

use online_core::*;

/// One row: features, targets, clock step, weight.
#[derive(Debug, Clone)]
pub struct Row {
    pub x: Vec<f64>,
    pub y: Vec<Option<f64>>,
    pub d: f64,
    pub w: f64,
}

/// What a row reports: the prediction slots, `n_eff`, and the numbers of
/// the step's extra (lasso's selected penalties), in that order.
#[derive(Debug, Clone, PartialEq)]
pub struct Out {
    pub pred: Vec<f64>,
    pub n_eff: f64,
    pub extra: Vec<f64>,
}

/// A model the harness runs, whatever its type.
pub trait Run {
    fn step(&mut self, r: &Row) -> Out;
    fn state(&self) -> State;
}

struct Model<M>(M);

impl<M: OnlineModel> Run for Model<M> {
    fn step(&mut self, r: &Row) -> Out {
        let s = self.0.step(&r.x, &r.y, r.d, r.w);
        let extra = match s.extra {
            Some(Extra::Lasso { lam_selected }) => lam_selected,
            Some(other) => panic!("an extra the harness does not know: {other:?}"),
            None => Vec::new(),
        };
        Out {
            pred: s.pred,
            n_eff: s.n_eff,
            extra,
        }
    }

    fn state(&self) -> State {
        self.0.state()
    }
}

/// The bare accumulator, which no model's state is and which a variant
/// still names: stepped by its own `update`, its means and co-moments
/// reported as the slots.
struct Accumulator(EwCov);

const ACC_DECAY: Decay = Decay::Halflife(8.0);

impl Run for Accumulator {
    fn step(&mut self, r: &Row) -> Out {
        let n_eff = self.0.n_eff();
        self.0.update(&r.x, ACC_DECAY.factor(r.d), r.w);
        let k = self.0.k();
        let mut pred: Vec<f64> = self.0.means().to_vec();
        pred.extend(
            (0..k)
                .flat_map(|i| (0..k).map(move |j| (i, j)))
                .map(|(i, j)| self.0.cov(i, j)),
        );
        Out {
            pred,
            n_eff,
            extra: Vec::new(),
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::EwCov(Box::new(self.0.clone())))
    }
}

fn restore_accumulator(s: &State) -> Result<Box<dyn Run>, String> {
    check_schema(s).map_err(|e| e.to_string())?;
    match &s.model {
        ModelState::EwCov(a) if a.has_shape(a.k()) => Ok(Box::new(Accumulator((**a).clone()))),
        ModelState::EwCov(_) => Err("the accumulator has the wrong shape".into()),
        other => Err(format!("expected the accumulator, found {}", other.kind())),
    }
}

fn restore<M: OnlineModel + 'static>(s: &State) -> Result<Box<dyn Run>, String> {
    M::restore(s)
        .map(|m| Box::new(Model(m)) as Box<dyn Run>)
        .map_err(|e| e.to_string())
}

/// What the targets of a case's stream are.
#[derive(Debug, Clone, Copy)]
pub enum Targets {
    /// `n` linear targets of the features, with noise.
    Linear(usize),
    /// One 0/1 target.
    Binary,
    /// One class label in `0..n`.
    Class(usize),
    /// None: the model reads only its features.
    None,
}

/// One case: a model, configured, and its stream.
pub struct Case {
    /// The fixture's file name.
    pub name: &'static str,
    /// The [`ModelState`] variant its state is.
    pub variant: &'static str,
    pub build: fn() -> Box<dyn Run>,
    pub restore: fn(&State) -> Result<Box<dyn Run>, String>,
    pub n_features: usize,
    pub targets: Targets,
    /// Rows before the save; the continuation is [`CONTINUATION`] rows.
    pub before: usize,
}

/// Rows a continuation is checked on.
pub const CONTINUATION: usize = 16;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

impl Case {
    /// The case's whole stream: `before` rows, then the continuation. A clock
    /// gap, a null target, a zero weight and weights other than 1 among
    /// them, on both sides of the save; two regimes, so the detectors and
    /// clusterers have something to find.
    pub fn rows(&self) -> Vec<Row> {
        let mut s = 0x5eed_u64 ^ (self.name.len() as u64) << 32;
        for b in self.name.bytes() {
            s = s.wrapping_mul(31).wrapping_add(u64::from(b));
        }
        let n = self.before + CONTINUATION;
        (0..n)
            .map(|i| {
                let shift = if (i / 12) % 2 == 1 { 2.0 } else { 0.0 };
                let x: Vec<f64> = (0..self.n_features).map(|_| shift + lcg(&mut s)).collect();
                let x0 = x.first().copied().unwrap_or(0.0);
                let x1 = x.get(1).copied().unwrap_or(0.0);
                let signal = 1.5 * x0 - 0.75 * x1 + 0.25;
                let noise = 0.1 * lcg(&mut s);
                let null = i % 11 == 7;
                let y = match self.targets {
                    Targets::Linear(t) => (0..t)
                        .map(|j| (!null).then_some(signal * (1.0 + j as f64) + noise - j as f64))
                        .collect(),
                    Targets::Binary => {
                        vec![(!null).then_some(f64::from(u8::from(signal + noise > 1.0)))]
                    }
                    Targets::Class(c) => {
                        let label = ((signal + noise + 2.0).max(0.0) as usize).min(c - 1);
                        vec![(!null).then_some(label as f64)]
                    }
                    Targets::None => Vec::new(),
                };
                let d = match i {
                    0 => 0.0,
                    _ if i % 23 == 13 => 9.0,
                    _ => 1.0,
                };
                let w = match i % 9 {
                    4 => 0.0,
                    2 => 2.5,
                    6 => 0.5,
                    _ => 1.0,
                };
                Row { x, y, d, w }
            })
            .collect()
    }
}

const K: usize = 2;
const H: f64 = 10.0;

fn decay() -> Decay {
    Decay::Halflife(H)
}

fn ewridge_cfg() -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        ridge: vec![1e-6, 0.1],
        feature_sets: vec![],
        standardize: true,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::Pairwise,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

fn lasso_cfg() -> LassoCfg {
    LassoCfg {
        n_features: K,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        lasso_path: vec![0.1, 0.01, 0.0],
        l1_ratio: 1.0,
        select_half_life: Some(5.0),
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
        target_gaps: TargetGaps::OwnRows,
    }
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
        revert_half_life: vec![f64::INFINITY, 25.0, 6.0],
        standardize: true,
    }
}

fn robust_cfg(loss: RobustLoss) -> RobustCfg {
    RobustCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        loss,
        ridge: 1e-6,
        standardize: true,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 1e-3,
    }
}

fn ew_cov_cfg() -> EwCovCfg {
    EwCovCfg {
        n_features: 3,
        decay: decay(),
        stats: vec![
            EwCovStat::Mean,
            EwCovStat::Var,
            EwCovStat::Corr,
            EwCovStat::Mahal,
            EwCovStat::LagCorr,
        ],
        min_weight: 3.0,
        precision_prior: Some(0.01),
        mahal_quantiles: vec![0.5, 0.9],
        pca: 1,
        pca_every: 4.0,
        max_rows_between_pca: 8,
        lags: vec![1, 3],
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

fn ew_class_cfg() -> EwClassCfg {
    EwClassCfg {
        n_features: K,
        n_classes: 3,
        decay: decay(),
        min_weight: 3.0,
        covariance: Covariance::Full,
        precision_prior: 0.1,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

fn marginal_cfg() -> MarginalCfg {
    MarginalCfg {
        n_features: K,
        n_targets: 2,
        decay: decay(),
        min_weight: vec![3.0; 2],
        lags: vec![],
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
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
        split_merge_every_rows: 20,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.1,
    }
}

fn corrchange_cfg(kind: CorrChangeKind) -> CorrChangeCfg {
    CorrChangeCfg {
        n_features: K,
        kind,
        span_rows: 10,
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
        monitor_rows: 15,
        boundary_gamma: 0.0,
    }
}

/// The window every windowed case runs: twelve clock units, a snapshot
/// every three.
const WINDOW: (Option<f64>, Option<f64>, Option<usize>) = (Some(12.0), Some(3.0), None);

macro_rules! case {
    ($name:literal, $variant:literal, $ty:ty, $build:expr, $k:expr, $targets:expr, $before:expr) => {
        Case {
            name: $name,
            variant: $variant,
            build: || Box::new(Model::<$ty>($build)),
            restore: restore::<$ty>,
            n_features: $k,
            targets: $targets,
            before: $before,
        }
    };
}

/// Every case, in file order.
pub fn all() -> Vec<Case> {
    vec![
        Case {
            name: "ew_cov_accumulator",
            variant: "EwCov",
            build: || Box::new(Accumulator(EwCov::new(K))),
            restore: restore_accumulator,
            n_features: K,
            targets: Targets::None,
            before: 40,
        },
        case!(
            "ewridge",
            "EwRidge",
            EwRidge,
            EwRidge::new(ewridge_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "ewridge_window",
            "EwRidge",
            EwRidge,
            EwRidge::new(EwRidgeCfg {
                target_gaps: TargetGaps::OwnRows,
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..ewridge_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "rls",
            "Rls",
            Rls,
            Rls::new(RlsCfg {
                n_features: K,
                n_targets: 2,
                fit_intercept: true,
                decay: decay(),
                delta: 1.0,
                coef_prior: None,
                min_weight: 3.0,
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "lasso",
            "Lasso",
            Lasso,
            Lasso::new(lasso_cfg()).unwrap(),
            K,
            Targets::Linear(1),
            40
        ),
        case!(
            "lasso_window",
            "Lasso",
            Lasso,
            Lasso::new(LassoCfg {
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..lasso_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(1),
            40
        ),
        case!(
            "kalman",
            "Kalman",
            Kalman,
            Kalman::new(kalman_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "huber",
            "Robust",
            Robust,
            Robust::new(robust_cfg(RobustLoss::Huber { delta: 1.5 })).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "quantile",
            "Robust",
            Robust,
            Robust::new(robust_cfg(RobustLoss::Quantile { tau: 0.75 })).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "ftrl",
            "Ftrl",
            Ftrl,
            Ftrl::new(FtrlCfg {
                n_features: K,
                n_targets: 1,
                fit_intercept: true,
                decay: decay(),
                alpha: 0.1,
                beta: 1.0,
                l1: 0.01,
                l2: 1.0,
                min_weight: 3.0,
                strict_binary: false,
                loss: FtrlLoss::Logistic,
            })
            .unwrap(),
            K,
            Targets::Binary,
            40
        ),
        case!(
            "ew_cov",
            "EwCovModel",
            EwCovModel,
            EwCovModel::new(ew_cov_cfg()).unwrap(),
            3,
            Targets::None,
            40
        ),
        case!(
            "ew_cov_window",
            "EwCovModel",
            EwCovModel,
            EwCovModel::new(EwCovCfg {
                stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
                mahal_quantiles: Vec::new(),
                pca: 0,
                pca_every: 0.0,
                max_rows_between_pca: u32::MAX,
                lags: Vec::new(),
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..ew_cov_cfg()
            })
            .unwrap(),
            3,
            Targets::None,
            40
        ),
        case!(
            "sgd",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                n_features: K,
                n_targets: 2,
                fit_intercept: true,
                decay: decay(),
                loss: SgdLoss::Squared,
                learning_rate: 0.05,
                schedule: LearningRate::AdaGrad,
                l2: 0.01,
                clip_gradient: 1e3,
                constraint: None,
                standardize: true,
                min_weight: 3.0,
                strict_binary: false,
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "pa",
            "Pa",
            Pa,
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
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "holt",
            "Holt",
            Holt,
            Holt::new(HoltCfg {
                n_targets: 2,
                level_half_life: H,
                trend_half_life: 4.0 * H,
                min_weight: 3.0,
                trend: true,
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // Saved with its warm-up buffer still filling: 10 rows wanted, 6
        // held.
        case!(
            "kmeans_warming",
            "KMeans",
            KMeans,
            KMeans::new(kmeans_cfg()).unwrap(),
            K,
            Targets::None,
            6
        ),
        case!(
            "kmeans",
            "KMeans",
            KMeans,
            KMeans::new(kmeans_cfg()).unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "micro",
            "Micro",
            Micro,
            Micro::new(MicroCfg {
                n_features: K,
                decay: decay(),
                min_weight: 3.0,
                eps: 0.6,
                beta_mu: 2.0,
                max_clusters: 50,
                prune_every: 6.0,
                max_rows_between_prunes: 10,
                macro_link: None,
                standardize: true,
                scale_floor: 0.1,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "ew_class",
            "EwClass",
            EwClass,
            EwClass::new(ew_class_cfg()).unwrap(),
            K,
            Targets::Class(3),
            40
        ),
        case!(
            "ew_class_window",
            "EwClass",
            EwClass,
            EwClass::new(EwClassCfg {
                covariance: Covariance::Diagonal,
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..ew_class_cfg()
            })
            .unwrap(),
            K,
            Targets::Class(3),
            40
        ),
        case!(
            "seqtest",
            "SeqTest",
            SeqTest,
            SeqTest::new(SeqTestCfg {
                n_targets: 2,
                min_weight: 3.0,
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        case!(
            "marginal_lags",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                lags: vec![1, 2],
                cross_lags: Some(vec![2]),
                feature_moments: FeatureMomentLayout::Shared,
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Saved while the bins hold their warm-up rows: 30 wanted.
        case!(
            "marginal_bins",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                bins: Some(Box::new(BinCfg {
                    n_bins: 4,
                    edges: None,
                    rule: BinRule::Quantile,
                    warm_rows: 30,
                    budget_mib: None,
                })),
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            24
        ),
        case!(
            "marginal_window",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "deco",
            "Deco",
            Deco,
            Deco::new(DecoCfg {
                n_features: 3,
                decay: decay(),
                dynamics: DecoDynamics::Ew,
                alpha: None,
                beta: None,
                blocks: Vec::new(),
                min_weight: 3.0,
            })
            .unwrap(),
            3,
            Targets::None,
            40
        ),
        case!(
            "rcov",
            "Rcov",
            Rcov,
            Rcov::new(RcovCfg {
                n_features: K,
                kind: RcovKind::Kernel,
                kernel: "parzen".into(),
                bandwidth: Some(3),
                jitter: 2,
                theta: 1.0,
                psd: true,
                block_rows: Some(8),
                max_bandwidth: None,
                preavg_rows: None,
                noise_stride: 1,
                iv_stride: 20,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "hmm",
            "Hmm",
            Hmm,
            Hmm::new(HmmCfg {
                n_features: K,
                k: 2,
                decay: decay(),
                covariance: Covariance::Full,
                precision_prior: 1e-2,
                min_weight: 0.0,
                learn: true,
                transition_prior: 1.0,
                transition: None,
                means: None,
                covs: None,
                warm_rows: 10,
                seed_rule: SeedRule::First,
                seed: 1,
                tvtp: None,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "corrchange_monitor",
            "CorrChange",
            CorrChange,
            CorrChange::new(corrchange_cfg(CorrChangeKind::Monitor)).unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "corrchange_sequential",
            "CorrChange",
            CorrChange,
            CorrChange::new(corrchange_cfg(CorrChangeKind::Sequential)).unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "bocpd",
            "Bocpd",
            Bocpd,
            Bocpd::new(BocpdCfg {
                n_features: K,
                hazard: 20.0,
                hazard_from_row: false,
                emission: BocpdEmission::Diag,
                prior_mean: None,
                prior_kappa: 1.0,
                prior_nu: Some(4.0),
                prior_scale: Some(vec![1.0]),
                robust_beta: 0.0,
                prune_below: 1e-6,
                max_run: 60,
                min_weight: 0.0,
                warm_rows: None,
                hazard_on_clock: true,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
    ]
}

/// The case's model after its first part, and its rows.
pub fn saved(case: &Case) -> (Box<dyn Run>, Vec<Row>) {
    let rows = case.rows();
    let mut m = (case.build)();
    for r in &rows[..case.before] {
        m.step(r);
    }
    (m, rows)
}

/// A state as the bank writes one: named msgpack.
pub fn named(s: &State) -> Vec<u8> {
    rmp_serde::to_vec_named(s).expect("a state encodes")
}

/// `bytes` decoded and restored as `case`'s model, or why not.
pub fn load(case: &Case, bytes: &[u8]) -> Result<Box<dyn Run>, String> {
    let s: State = rmp_serde::from_slice(bytes).map_err(|e| format!("decoding: {e}"))?;
    (case.restore)(&s)
}

/// A frozen fixture, as `tests/state_fixtures/<name>.rs` holds it (written
/// by `PRINT_STATE_FIXTURES=1`): the case's state after its first part, as
/// named msgpack in hex, the schema and the platform it was written at, and
/// its continuation -- every input of each row and every number it reported
/// -- as the bits of each double, so nothing is read back through a decimal.
pub struct Fixture {
    pub name: &'static str,
    pub variant: &'static str,
    pub schema: u32,
    /// `<os>-<arch>` of the build that wrote it ([`writer`]).
    pub writer: &'static str,
    pub state: &'static str,
    /// Per row, `n_features` doubles.
    pub x: &'static [u64],
    /// Per row, one entry a target.
    pub y: &'static [Option<u64>],
    pub d: &'static [u64],
    pub w: &'static [u64],
    /// Per row, `n_pred` doubles.
    pub pred: &'static [u64],
    pub n_eff: &'static [u64],
    /// Per row, `n_extra` doubles.
    pub extra: &'static [u64],
    pub n_features: usize,
    pub n_targets: usize,
    pub n_pred: usize,
    pub n_extra: usize,
}

impl Fixture {
    pub fn bytes(&self) -> Vec<u8> {
        unhex(self.state)
    }

    /// The continuation's rows, as the fixture holds them.
    pub fn rows(&self) -> Vec<Row> {
        let f = f64::from_bits;
        (0..self.d.len())
            .map(|i| Row {
                x: self.x[i * self.n_features..(i + 1) * self.n_features]
                    .iter()
                    .map(|&b| f(b))
                    .collect(),
                y: self.y[i * self.n_targets..(i + 1) * self.n_targets]
                    .iter()
                    .map(|b| b.map(f))
                    .collect(),
                d: f(self.d[i]),
                w: f(self.w[i]),
            })
            .collect()
    }

    /// What each continuation row reported when the fixture was written.
    pub fn outs(&self) -> Vec<Out> {
        let f = f64::from_bits;
        (0..self.d.len())
            .map(|i| Out {
                pred: self.pred[i * self.n_pred..(i + 1) * self.n_pred]
                    .iter()
                    .map(|&b| f(b))
                    .collect(),
                n_eff: f(self.n_eff[i]),
                extra: self.extra[i * self.n_extra..(i + 1) * self.n_extra]
                    .iter()
                    .map(|&b| f(b))
                    .collect(),
            })
            .collect()
    }
}

/// The platform a build runs on, as a fixture records its writer.
pub fn writer() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(s: &str) -> Vec<u8> {
    let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    s.chunks(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}

/// Whether two numbers a row reported agree: both NaN -- whatever their
/// bits, which an x86 and an ARM build make differently -- or, at
/// `tol = 0`, the same bits, and otherwise within `tol` relative to
/// `1 + |want|`.
pub fn agree(got: f64, want: f64, tol: f64) -> bool {
    if got.is_nan() || want.is_nan() {
        return got.is_nan() && want.is_nan();
    }
    if tol == 0.0 {
        return got.to_bits() == want.to_bits();
    }
    (got - want).abs() <= tol * (1.0 + want.abs())
}
