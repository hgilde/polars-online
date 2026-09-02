//! One (spec, group) stream: clock state + model instances (one per halflife
//! grid entry), row-by-row processing with the docs/PLAN.md §3 null policy.

use online_core::{
    ClockState, Decay, EwAutoCorr, EwCovCfg, EwCovModel, EwCovStat, EwRidge, EwRidgeCfg, Ftrl,
    FtrlCfg, FtrlLoss, Holt, HoltCfg, INPUT_BOUND, Kalman, KalmanCfg, Lasso, LassoCfg,
    LearningRate, ModelState, OnlineModel, P2Quantile, Pa, PaCfg, PaMode, PageHinkley, Rls, RlsCfg,
    Robust, RobustCfg, RobustLoss, Sgd, SgdCfg, SgdLoss, SlotMetrics, State, StateError,
};
use serde::{Deserialize, Serialize};

use crate::spec::{FloatOrList, ModelKind, Spec};

/// Enum dispatch over the models the bank can run (serde-friendly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnyModel {
    EwRidge(Box<EwRidge>),
    Rls(Box<Rls>),
    Lasso(Box<Lasso>),
    Kalman(Box<Kalman>),
    Robust(Box<Robust>),
    Ftrl(Box<Ftrl>),
    EwCov(Box<EwCovModel>),
    Sgd(Box<Sgd>),
    Pa(Box<Pa>),
    Holt(Box<Holt>),
}

/// Bind the boxed model of whichever variant `$self` is, then run `$body`.
///
/// Only for the methods that are the *same* call on every variant. Three of
/// `AnyModel`'s six are not — `solve_failures` groups the models that never
/// factorize, `coefficients` reshapes per model, and `restore` matches on
/// `ModelState` rather than on `Self` — and those stay written out, because a
/// macro that needed a per-variant escape hatch would be harder to read than
/// the match it replaced (docs/SIMPLIFICATION.md S3).
macro_rules! dispatch {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            AnyModel::EwRidge($m) => $body,
            AnyModel::Rls($m) => $body,
            AnyModel::Lasso($m) => $body,
            AnyModel::Kalman($m) => $body,
            AnyModel::Robust($m) => $body,
            AnyModel::Ftrl($m) => $body,
            AnyModel::EwCov($m) => $body,
            AnyModel::Sgd($m) => $body,
            AnyModel::Pa($m) => $body,
            AnyModel::Holt($m) => $body,
        }
    };
}

impl AnyModel {
    /// Mix the main accumulators toward a long-run twin, where the model has
    /// one (`session_shrink`). A no-op elsewhere.
    pub fn blend_toward_long_run(&mut self) {
        if let AnyModel::EwRidge(m) = self {
            m.blend_toward_long_run();
        }
    }

    pub fn step(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
    ) -> online_core::Step {
        dispatch!(self, m => m.step(x, y, d_clock, weight))
    }

    /// Cumulative count of jittered or failed factorizations (docs/PLAN.md §7).
    /// Models that do not factorize (rls, kalman, ftrl) report 0.
    pub fn solve_failures(&self) -> u64 {
        match self {
            AnyModel::EwRidge(m) => m.solve_failures,
            AnyModel::Lasso(m) => m.solve_failures,
            AnyModel::Robust(m) => m.solve_failures,
            AnyModel::Rls(_)
            | AnyModel::Kalman(_)
            | AnyModel::Ftrl(_)
            | AnyModel::EwCov(_)
            | AnyModel::Sgd(_)
            | AnyModel::Pa(_)
            | AnyModel::Holt(_) => 0,
        }
    }

    pub fn n_outputs(&self) -> usize {
        dispatch!(self, m => m.n_outputs())
    }

    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        match self {
            AnyModel::EwRidge(m) => m.coefficients().map(|b| b.to_vec()),
            AnyModel::Rls(m) => Some(m.coefficients().to_vec()),
            // Flattened to (target x path point) rows, matching the pred slots.
            AnyModel::Lasso(m) => m
                .coefficients()
                .map(|b| b.iter().flat_map(|per_t| per_t.iter().cloned()).collect()),
            AnyModel::Kalman(m) => Some(m.coefficients()),
            AnyModel::Robust(m) => m.coefficients().map(|b| b.to_vec()),
            AnyModel::Ftrl(m) => Some(m.coefficients()),
            // ew_cov has no coefficients: its outputs are the statistics.
            AnyModel::EwCov(_) => None,
            AnyModel::Sgd(m) => Some(m.coefficients()),
            AnyModel::Pa(m) => Some(m.coefficients().to_vec()),
            AnyModel::Holt(m) => Some(m.coefficients()),
        }
    }

    pub fn state(&self) -> State {
        dispatch!(self, m => m.state())
    }

    pub fn restore(s: &State) -> Result<Self, StateError> {
        match &s.model {
            ModelState::EwRidge(_) => Ok(AnyModel::EwRidge(Box::new(EwRidge::restore(s)?))),
            ModelState::Rls(_) => Ok(AnyModel::Rls(Box::new(Rls::restore(s)?))),
            ModelState::Lasso(_) => Ok(AnyModel::Lasso(Box::new(Lasso::restore(s)?))),
            ModelState::Kalman(_) => Ok(AnyModel::Kalman(Box::new(Kalman::restore(s)?))),
            ModelState::Robust(_) => Ok(AnyModel::Robust(Box::new(Robust::restore(s)?))),
            ModelState::Ftrl(_) => Ok(AnyModel::Ftrl(Box::new(Ftrl::restore(s)?))),
            ModelState::EwCovModel(_) => Ok(AnyModel::EwCov(Box::new(EwCovModel::restore(s)?))),
            ModelState::Sgd(_) => Ok(AnyModel::Sgd(Box::new(Sgd::restore(s)?))),
            ModelState::Pa(_) => Ok(AnyModel::Pa(Box::new(Pa::restore(s)?))),
            ModelState::Holt(_) => Ok(AnyModel::Holt(Box::new(Holt::restore(s)?))),
            other => Err(StateError::WrongModel {
                expected: "a bank-supported model",
                found: other.kind(),
            }),
        }
    }
}

/// Build the model instances for a spec: one per halflife grid entry.
pub fn build_models(spec: &Spec) -> Result<Vec<(String, AnyModel)>, String> {
    let decays = spec.decays()?;
    decays
        .into_iter()
        .map(|(suffix, decay)| {
            let m = build_one(spec, decay)?;
            Ok((suffix, m))
        })
        .collect()
}

fn build_one(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    match &spec.model {
        ModelKind::EwRidge {
            ridge,
            feature_sets,
            standardize,
            ridge_decay,
            coef0,
            session_shrink,
            long_halflife,
            solve_every,
            max_rows_between_solves,
        } => {
            let fs = feature_sets
                .as_ref()
                .map(|sets| {
                    sets.iter()
                        .map(|(name, cols)| {
                            let idx = cols
                                .iter()
                                .map(|c| spec.features.iter().position(|f| f == c).unwrap())
                                .collect();
                            (name.clone(), idx)
                        })
                        .collect()
                })
                .unwrap_or_default();
            let cfg = EwRidgeCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                ridge: ridge
                    .as_ref()
                    .map(FloatOrList::to_vec)
                    .unwrap_or_else(|| vec![1e-6]),
                feature_sets: fs,
                standardize: *standardize,
                ridge_decay: *ridge_decay,
                session_shrink: *session_shrink,
                long_halflife: *long_halflife,
                coef0: coef0.clone(),
                min_periods: spec.min_periods_or_default(),
                solve_every: solve_every.unwrap_or_else(|| spec.solve_every_default(decay)),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
            };
            Ok(AnyModel::EwRidge(Box::new(EwRidge::new(cfg)?)))
        }
        ModelKind::Rls { ridge, coef0 } => {
            let cfg = RlsCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                ridge: ridge.unwrap_or(1.0),
                coef0: coef0.clone(),
                min_periods: spec.min_periods_or_default(),
            };
            Ok(AnyModel::Rls(Box::new(Rls::new(cfg)?)))
        }
        ModelKind::Lasso {
            lasso_path,
            l1_ratio,
            select_halflife,
            solve_every,
            max_rows_between_solves,
            max_cd_iters,
            cd_tol,
        } => {
            let cfg = LassoCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                lasso_path: lasso_path.clone(),
                l1_ratio: l1_ratio.unwrap_or(1.0),
                select_halflife: *select_halflife,
                min_periods: spec.min_periods_or_default(),
                solve_every: solve_every.unwrap_or_else(|| spec.solve_every_default(decay)),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                max_cd_iters: max_cd_iters.unwrap_or(100),
                cd_tol: cd_tol.unwrap_or(1e-10),
            };
            Ok(AnyModel::Lasso(Box::new(Lasso::new(cfg)?)))
        }
        ModelKind::Kalman {
            coef_halflife,
            q,
            obs_var,
            p0,
            share_p,
            standardize,
        } => {
            let cfg = KalmanCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                halflife: coef_halflife.to_vec(),
                q: q.as_ref().map(|v| v.iter().map(|n| n.0).collect()),
                obs_var: *obs_var,
                p0: p0.unwrap_or(1.0),
                share_p: *share_p,
                min_periods: spec.min_periods_or_default(),
                standardize: *standardize,
            };
            Ok(AnyModel::Kalman(Box::new(Kalman::new(cfg)?)))
        }
        ModelKind::Huber {
            huber_delta,
            ridge,
            standardize,
            solve_every,
            max_rows_between_solves,
        } => {
            let cfg = RobustCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                loss: RobustLoss::Huber {
                    delta: huber_delta.unwrap_or(1.5),
                },
                ridge: ridge.unwrap_or(1e-6),
                standardize: *standardize,
                min_periods: spec.min_periods_or_default(),
                solve_every: solve_every.unwrap_or_else(|| spec.solve_every_default(decay)),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                quantile_eps: 1e-3,
            };
            Ok(AnyModel::Robust(Box::new(Robust::new(cfg)?)))
        }
        ModelKind::Quantile {
            quantile,
            ridge,
            standardize,
            solve_every,
            max_rows_between_solves,
            quantile_eps,
        } => {
            let cfg = RobustCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                loss: RobustLoss::Quantile { tau: *quantile },
                ridge: ridge.unwrap_or(1e-6),
                standardize: *standardize,
                min_periods: spec.min_periods_or_default(),
                solve_every: solve_every.unwrap_or_else(|| spec.solve_every_default(decay)),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                quantile_eps: quantile_eps.unwrap_or(1e-3),
            };
            Ok(AnyModel::Robust(Box::new(Robust::new(cfg)?)))
        }
        ModelKind::Ftrl {
            alpha,
            beta,
            l1,
            l2,
            strict_binary,
            loss,
        } => {
            let loss = match loss.as_deref() {
                None | Some("logistic") => FtrlLoss::Logistic,
                Some("squared") => FtrlLoss::Squared,
                Some(other) => {
                    return Err(format!(
                        "unknown ftrl loss {other:?}; expected \"logistic\" or \"squared\""
                    ));
                }
            };
            let cfg = FtrlCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                alpha: alpha.unwrap_or(0.1),
                beta: beta.unwrap_or(1.0),
                l1: l1.unwrap_or(0.0),
                l2: l2.unwrap_or(1.0),
                min_periods: spec.min_periods_or_default(),
                strict_binary: *strict_binary,
                loss,
            };
            Ok(AnyModel::Ftrl(Box::new(Ftrl::new(cfg)?)))
        }
        ModelKind::EwCov {
            stats,
            precision_prior,
        } => {
            let names = stats
                .clone()
                .unwrap_or_else(|| vec!["mean".into(), "std".into(), "corr".into()]);
            let stats = names
                .iter()
                .map(|s| match s.as_str() {
                    "mean" => Ok(EwCovStat::Mean),
                    "var" => Ok(EwCovStat::Var),
                    "std" => Ok(EwCovStat::Std),
                    "cov" => Ok(EwCovStat::Cov),
                    "corr" => Ok(EwCovStat::Corr),
                    "partial_corr" => Ok(EwCovStat::PartialCorr),
                    other => Err(format!("unknown ew_cov statistic {other:?}")),
                })
                .collect::<Result<Vec<_>, String>>()?;
            let cfg = EwCovCfg {
                n_features: spec.k(),
                decay,
                stats,
                min_periods: spec.min_periods_per_target()[0].max(2.0),
                precision_prior: *precision_prior,
            };
            Ok(AnyModel::EwCov(Box::new(EwCovModel::new(cfg)?)))
        }
        ModelKind::Sgd {
            loss,
            huber_delta,
            quantile,
            eps,
            learning_rate,
            schedule,
            power,
            l2,
            clip_gradient,
            scale_features,
        } => {
            let loss = match loss.as_deref().unwrap_or("squared") {
                "squared" => SgdLoss::Squared,
                "huber" => SgdLoss::Huber {
                    delta: huber_delta.unwrap_or(1.0),
                },
                "quantile" => SgdLoss::Quantile {
                    tau: quantile.ok_or("sgd: loss \"quantile\" needs a `quantile` level")?,
                },
                "epsilon_insensitive" => SgdLoss::EpsilonInsensitive {
                    eps: eps.unwrap_or(0.1),
                },
                "poisson" => SgdLoss::Poisson,
                "logistic" => SgdLoss::Logistic,
                other => return Err(format!("unknown sgd loss {other:?}")),
            };
            let sched = match schedule.as_deref().unwrap_or("constant") {
                "constant" => LearningRate::Constant,
                "inv_scaling" => LearningRate::InvScaling {
                    power: power.unwrap_or(0.5),
                },
                "adagrad" => LearningRate::AdaGrad,
                other => return Err(format!("unknown sgd schedule {other:?}")),
            };
            let cfg = SgdCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                loss,
                learning_rate: learning_rate.unwrap_or(0.01),
                schedule: sched,
                l2: l2.unwrap_or(0.0),
                min_periods: spec.min_periods_or_default(),
                // Finite by default: see SgdCfg::clip_gradient.
                clip_gradient: clip_gradient.map_or(1e3, |n| n.0),
                scale_features: *scale_features,
            };
            Ok(AnyModel::Sgd(Box::new(Sgd::new(cfg)?)))
        }
        ModelKind::Pa { mode, c, eps } => {
            let mode = match mode.as_deref().unwrap_or("pa1") {
                "pa" => PaMode::Pa,
                "pa1" => PaMode::Pa1,
                "pa2" => PaMode::Pa2,
                other => return Err(format!("unknown pa mode {other:?}")),
            };
            let cfg = PaCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                add_intercept: spec.add_intercept,
                decay,
                mode,
                c: c.unwrap_or(1.0),
                eps: eps.unwrap_or(0.1),
                min_periods: spec.min_periods_or_default(),
            };
            Ok(AnyModel::Pa(Box::new(Pa::new(cfg)?)))
        }
        ModelKind::Holt {
            level_halflife,
            trend_halflife,
        } => {
            // Default the level to the spec's own halflife, so `halflife` means
            // the same thing here as it does for every other model.
            let level = level_halflife.unwrap_or(match decay {
                Decay::Halflife(h) => h,
                Decay::Lam(l) => -std::f64::consts::LN_2 / l.ln(),
            });
            let cfg = HoltCfg {
                n_targets: spec.m(),
                level_halflife: level,
                trend_halflife: trend_halflife.map_or(level * 4.0, |n| n.0),
                min_periods: spec.min_periods_or_default(),
            };
            Ok(AnyModel::Holt(Box::new(Holt::new(cfg)?)))
        }
    }
}

/// One grid combo: the rendered label plus the machine values it encodes, so
/// metadata can never drift from the string (docs/RELEASE-READINESS.md).
#[derive(Debug, Clone, Default)]
pub struct Combo {
    /// Rendered suffix ("" when there is only one combo).
    pub label: String,
    pub ridge: Option<f64>,
    pub feature_set: Option<String>,
    /// Lasso path point.
    pub lambda: Option<f64>,
}

/// Combo labels per model instance ("" when there is only one combo).
pub fn combo_labels(spec: &Spec) -> Vec<String> {
    combos(spec).into_iter().map(|c| c.label).collect()
}

/// The combos with their machine values.
pub fn combos(spec: &Spec) -> Vec<Combo> {
    match &spec.model {
        ModelKind::EwRidge {
            ridge,
            feature_sets,
            ..
        } => {
            let nr = ridge.as_ref().map(|r| r.to_vec().len()).unwrap_or(1);
            let nf = feature_sets.as_ref().map(|f| f.len()).unwrap_or(0).max(1);
            if nr * nf == 1 {
                return vec![Combo::default()];
            }
            let ridges = ridge
                .as_ref()
                .map(FloatOrList::to_vec)
                .unwrap_or(vec![1e-6]);
            let fs_names: Vec<String> = feature_sets
                .as_ref()
                .map(|f| f.iter().map(|(n, _)| n.clone()).collect())
                .unwrap_or_else(|| vec!["all".to_string()]);
            let mut out = Vec::new();
            for f in &fs_names {
                for r in &ridges {
                    let label = if nf == 1 {
                        format!("__r{}", crate::spec::num_label(*r))
                    } else if nr == 1 {
                        format!("__{f}")
                    } else {
                        format!("__{f}_r{}", crate::spec::num_label(*r))
                    };
                    out.push(Combo {
                        label,
                        ridge: Some(*r),
                        feature_set: (nf > 1).then(|| f.clone()),
                        lambda: None,
                    });
                }
            }
            out
        }
        ModelKind::Rls { .. }
        | ModelKind::Kalman { .. }
        | ModelKind::Huber { .. }
        | ModelKind::Quantile { .. }
        | ModelKind::Ftrl { .. }
        | ModelKind::EwCov { .. }
        | ModelKind::Sgd { .. }
        | ModelKind::Pa { .. }
        | ModelKind::Holt { .. } => vec![Combo::default()],
        ModelKind::Lasso { lasso_path, .. } => lasso_path
            .iter()
            .map(|l| Combo {
                label: format!("__l{}", crate::spec::num_label(*l)),
                ridge: None,
                feature_set: None,
                lambda: Some(*l),
            })
            .collect(),
    }
}

/// Serialized per-stream state: the clock plus each halflife's model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamState {
    pub clock: ClockState,
    pub models: Vec<State>,
    pub rows_seen: u64,
    /// EW mean squared out-of-sample residual, per model instance and output
    /// slot. `#[serde(default)]` so state files written before this existed
    /// still load (they simply restart the estimate).
    #[serde(default)]
    pub resid_var: Vec<Vec<f64>>,
    #[serde(default)]
    pub resid_w: Vec<Vec<f64>>,
    #[serde(default)]
    pub drift: Vec<Vec<PageHinkley>>,
    #[serde(default)]
    pub resid_q: Vec<Vec<Vec<P2Quantile>>>,
    #[serde(default)]
    pub autocorr: Vec<Vec<EwAutoCorr>>,
    #[serde(default)]
    pub metrics: Vec<Vec<SlotMetrics>>,
}

/// Live per-stream state.
pub struct Stream {
    pub clock: ClockState,
    pub models: Vec<(String, AnyModel)>,
    pub rows_seen: u64,
    /// Decay of each model instance, needed to age `resid_var` on the same
    /// clock the model itself uses.
    decays: Vec<Decay>,
    /// EW mean squared out-of-sample residual per instance and slot, and its
    /// weight sum. Kept here rather than in the models so that every model gets
    /// the same definition: the models' own `sigma2` fields serve different
    /// internal purposes (robust scaling, Kalman observation noise) and are not
    /// all present or comparable.
    resid_var: Vec<Vec<f64>>,
    resid_w: Vec<Vec<f64>>,
    /// Page-Hinkley detectors per instance and slot, when `emit_drift` is on.
    drift: Vec<Vec<PageHinkley>>,
    /// Warmup threshold per target (ENHANCEMENTS E7).
    min_periods: Vec<f64>,
    /// P² estimators per instance, slot and requested level (ENHANCEMENTS E23).
    resid_q: Vec<Vec<Vec<P2Quantile>>>,
    /// EW residual autocorrelation per instance and slot.
    autocorr: Vec<Vec<EwAutoCorr>>,
    /// Evaluation metrics per instance and slot (ENHANCEMENTS E22).
    metrics: Vec<Vec<SlotMetrics>>,
    /// Row scratch, reused for the life of the stream so the chunk loop
    /// itself allocates nothing (docs/PERFORMANCE.md P1). The `pred` `Vec`
    /// inside each `Step` is the one per-row allocation left, and it is cheap
    /// (docs/IMPROVEMENTS.md P2).
    scratch: Vec<Scratch>,
}

impl Stream {
    /// Summed over this stream's model instances (one per halflife).
    pub fn solve_failures(&self) -> u64 {
        self.models.iter().map(|(_, m)| m.solve_failures()).sum()
    }

    /// Model instances in this stream (one per halflife).
    pub fn n_models(&self) -> usize {
        self.models.len()
    }

    /// Output slots per instance. Instances of one spec differ only in decay,
    /// so they all report the same count; the max is that count.
    pub fn n_slots(&self) -> usize {
        self.models
            .iter()
            .map(|(_, m)| m.n_outputs())
            .max()
            .unwrap_or(0)
    }
}

/// Is a feature, target or weight value one the models are asked to learn
/// from? Null (extracted as NaN), NaN, infinities and magnitudes beyond
/// [`INPUT_BOUND`] are all "missing": a feature or weight skips the row, a
/// target makes it predict-only (docs/PLAN.md §3, docs/IMPROVEMENTS.md C2).
#[inline]
pub fn usable(v: f64) -> bool {
    v.is_finite() && v.abs() <= INPUT_BOUND
}

/// True when this target has not reached its own warmup threshold yet.
#[inline]
fn step_n_eff_below(n_eff: f64, min_periods: &[f64], target: usize) -> bool {
    min_periods.get(target).is_some_and(|t| n_eff < *t)
}

/// Flat output buffers for one (stream, chunk) task (docs/PERFORMANCE.md P1).
///
/// One allocation per output *column* for the whole chunk, rather than the
/// ~11 `Vec`s per row the previous `RowOut` needed. Every numeric buffer is
/// `n_slots * n_rows`, slot-major, so a slot's values are contiguous and the
/// scatter into the final column is a straight walk. NaN is null.
///
/// `processed` distinguishes "this row was skipped" from "this row produced a
/// NaN", which matters only for `drift` (a bool, with no NaN to spare) and for
/// `n_eff`, which is otherwise always finite.
pub struct ChunkOut {
    /// Absolute row indices this task wrote, in order.
    pub rows: Vec<usize>,
    /// Per row: false = skipped, every output is null.
    pub processed: Vec<bool>,
    /// `n_models * n_slots * n_rows` unless noted; NaN = null.
    pub pred: Vec<f64>,
    pub resid: Vec<f64>,
    pub sigma: Vec<f64>,
    pub resid_z: Vec<f64>,
    pub autocorr: Vec<f64>,
    /// `(ic, r2, hit_rate)`, model-major: `n_models * 3 * n_slots * n_rows`.
    pub metrics: Vec<f64>,
    /// Model-major: `n_models * n_levels * n_slots * n_rows`.
    pub resid_q: Vec<f64>,
    pub drift: Vec<bool>,
    /// `n_models * n_rows`.
    pub n_eff: Vec<f64>,
    /// `n_models * n_targets * n_rows`, lasso only.
    pub lam_selected: Vec<f64>,
    /// Emitted on a cadence rather than every row, so it stays boxed:
    /// `[model][row]`.
    pub coef: Vec<Vec<Option<Vec<f64>>>>,
    /// Slot counts this layout was built for.
    pub n_models: usize,
    pub n_slots: usize,
    pub n_levels: usize,
}

impl ChunkOut {
    /// Buffers for `n_rows` rows of one stream, all null until written.
    pub fn new(spec: &Spec, n_models: usize, n_slots: usize, n_rows: usize) -> Self {
        let per = n_models * n_slots * n_rows;
        let on = |flag: bool| if flag { per } else { 0 };
        let n_levels = spec.resid_quantiles.as_ref().map_or(0, Vec::len);
        let is_lasso = matches!(spec.model, crate::ModelKind::Lasso { .. });
        // `sigma` is also the loss that `emit_selected` and `emit_averaged`
        // rank slots by (E13/E14 reuse E12's tracked error), so it has to be
        // materialized for them even when it is not itself an output field.
        let extras =
            spec.emit_sigma || spec.emit_resid_z || spec.emit_selected || spec.emit_averaged;
        Self {
            rows: Vec::with_capacity(n_rows),
            processed: vec![false; n_rows],
            pred: vec![f64::NAN; per],
            resid: vec![f64::NAN; per],
            sigma: vec![f64::NAN; on(extras)],
            resid_z: vec![f64::NAN; on(extras)],
            autocorr: vec![f64::NAN; on(spec.emit_autocorr)],
            metrics: vec![f64::NAN; 3 * on(spec.emit_metrics)],
            resid_q: vec![f64::NAN; n_levels * per],
            drift: vec![false; on(spec.emit_drift)],
            n_eff: vec![f64::NAN; n_models * n_rows],
            lam_selected: vec![
                f64::NAN;
                if is_lasso {
                    n_models * spec.m() * n_rows
                } else {
                    0
                }
            ],
            coef: vec![vec![None; n_rows]; n_models],
            n_models,
            n_slots,
            n_levels,
        }
    }

    /// Offset of `(model, slot)` at row `ri`, in any `n_models * n_slots *
    /// n_rows` buffer. The single place the layout is spelled out; the writer
    /// in `process_one` and the reader in `assemble` both go through it.
    #[inline]
    pub fn at(n_slots: usize, n_rows: usize, mi: usize, slot: usize, ri: usize) -> usize {
        (mi * n_slots + slot) * n_rows + ri
    }
}

impl Stream {
    pub fn new(spec: &Spec) -> Result<Self, String> {
        let models = build_models(spec)?;
        let decays: Vec<Decay> = spec.decays()?.into_iter().map(|(_, d)| d).collect();
        let slots: Vec<usize> = models.iter().map(|(_, m)| m.n_outputs()).collect();
        let drift = if spec.emit_drift {
            let d = PageHinkley::new(
                spec.drift_delta.unwrap_or(0.5),
                spec.drift_threshold.unwrap_or(20.0),
            );
            slots.iter().map(|&n| vec![d.clone(); n]).collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            clock: ClockState::new(),
            resid_var: slots.iter().map(|&n| vec![0.0; n]).collect(),
            resid_w: slots.iter().map(|&n| vec![0.0; n]).collect(),
            drift,
            resid_q: match &spec.resid_quantiles {
                Some(levels) => {
                    let protos: Vec<P2Quantile> = levels
                        .iter()
                        .map(|q| P2Quantile::new(*q))
                        .collect::<Result<_, _>>()?;
                    slots.iter().map(|&n| vec![protos.clone(); n]).collect()
                }
                None => Vec::new(),
            },
            autocorr: if spec.emit_autocorr {
                let proto = EwAutoCorr::new(spec.resid_autocorr_lag.unwrap_or(1))?;
                slots.iter().map(|&n| vec![proto.clone(); n]).collect()
            } else {
                Vec::new()
            },
            metrics: if spec.emit_metrics {
                slots.iter().map(|&n| vec![SlotMetrics::new(); n]).collect()
            } else {
                Vec::new()
            },
            min_periods: spec.min_periods_per_target(),
            models,
            decays,
            rows_seen: 0,
            scratch: slots.iter().map(|_| Scratch::default()).collect(),
        })
    }

    pub fn save(&self) -> StreamState {
        StreamState {
            clock: self.clock.clone(),
            models: self.models.iter().map(|(_, m)| m.state()).collect(),
            rows_seen: self.rows_seen,
            resid_var: self.resid_var.clone(),
            resid_w: self.resid_w.clone(),
            drift: self.drift.clone(),
            resid_q: self.resid_q.clone(),
            autocorr: self.autocorr.clone(),
            metrics: self.metrics.clone(),
        }
    }

    pub fn restore(spec: &Spec, saved: &StreamState) -> Result<Self, String> {
        let mut stream = Stream::new(spec)?;
        if stream.models.len() != saved.models.len() {
            return Err("saved state has a different number of model instances".into());
        }
        let models = stream
            .models
            .drain(..)
            .zip(&saved.models)
            .map(|((suffix, _), st)| {
                Ok((suffix, AnyModel::restore(st).map_err(|e| e.to_string())?))
            })
            .collect::<Result<Vec<_>, String>>()?;
        stream.models = models;
        stream.clock = saved.clock.clone();
        stream.rows_seen = saved.rows_seen;
        // Written before these fields existed => start the estimate over.
        if saved.resid_var.len() == stream.resid_var.len() {
            stream.resid_var = saved.resid_var.clone();
            stream.resid_w = saved.resid_w.clone();
        }
        if saved.drift.len() == stream.drift.len() {
            stream.drift = saved.drift.clone();
        }
        if saved.resid_q.len() == stream.resid_q.len() {
            stream.resid_q = saved.resid_q.clone();
        }
        if saved.autocorr.len() == stream.autocorr.len() {
            stream.autocorr = saved.autocorr.clone();
        }
        if saved.metrics.len() == stream.metrics.len() {
            stream.metrics = saved.metrics.clone();
        }
        Ok(stream)
    }

    /// Under `on_clock_reset = "error"`: the first row of the chunk at which
    /// this stream's clock would go backwards, exactly as [`Stream::process_chunk`]
    /// would report it, found on a copy of the clock so nothing is touched.
    ///
    /// The bank runs this over every stream of a chunk before it runs any of
    /// them (docs/IMPROVEMENTS.md C3). Without it a backwards clock in one
    /// group refused that group's rows but left every other group updated, so
    /// the corrected chunk could not be re-fed and only a `load` recovered the
    /// bank.
    pub fn check_clock(
        &self,
        cfg: &online_core::ClockCfg,
        clock: Option<&[f64]>,
        session: Option<&[u64]>,
        idx: &[usize],
    ) -> Result<(), (f64, usize)> {
        // A row-count clock cannot go backwards, and no other policy refuses a
        // row, so this costs nothing unless it can fail.
        let (Some(clock), online_core::OnClockReset::Error) = (clock, cfg.on_clock_reset) else {
            return Ok(());
        };
        let mut state = self.clock.clone();
        for &i in idx {
            // `accept` only routes the delta into `pending`; whether the row
            // is refused depends on the clock and session alone.
            let adv = state.advance(cfg, Some(clock[i]), session.map(|s| s[i]), true);
            if let Some(raw) = adv.backwards {
                return Err((raw, i));
            }
        }
        Ok(())
    }

    /// Process this stream's rows of one chunk, writing into flat per-slot
    /// buffers (docs/PERFORMANCE.md P1, P2).
    ///
    /// Two passes. The first walks the rows advancing the clock and deciding
    /// which are accepted -- that depends only on the clock and the input
    /// columns, never on the models. The second runs each model instance over
    /// the whole chunk, and because instances share nothing but that schedule,
    /// they run **in parallel**: a five-halflife grid on a single stream is
    /// five independent recursions rather than one serial loop.
    ///
    /// The one exception is `drift_action = "reset"`, where a break detected by
    /// any instance resets all of them, so instances are coupled *within a
    /// row*. That case keeps row-major order. Both paths call the same
    /// [`run_instance`], so there is one implementation of the arithmetic.
    ///
    /// On a backwards clock under `on_clock_reset = "error"`, returns the raw
    /// delta and the absolute row it happened at, for the caller to name, and
    /// leaves the stream untouched (pass 1 runs on a copy of the clock). The
    /// bank runs [`Stream::check_clock`] over every stream before it runs
    /// this on any, so the refusal is per chunk, not per stream.
    #[allow(clippy::too_many_arguments)]
    pub fn process_chunk(
        &mut self,
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        features: &[Vec<f64>],
        targets: &[Vec<f64>],
        clock: Option<&[f64]>,
        session: Option<&[u64]>,
        weight: Option<&[f64]>,
        idx: &[usize],
        out: &mut ChunkOut,
    ) -> Result<(), (f64, usize)> {
        let n_rows = idx.len();
        out.rows.extend_from_slice(idx);

        // ---- pass 1: the clock schedule, models untouched ----
        // On a copy of the clock, committed below, so a refused row leaves
        // the stream exactly as it was.
        let mut clock_state = self.clock.clone();
        let mut rows_seen = self.rows_seen;
        let last = idx.last().copied();
        let mut plans: Vec<RowPlan> = Vec::with_capacity(n_rows);
        for (ri, &i) in idx.iter().enumerate() {
            // Null arrives as NaN from extraction, so one `usable` covers
            // null, NaN, infinity and the bound.
            let w = weight.map(|w| w[i]);
            let accept = features.iter().all(|f| usable(f[i])) && w.map(usable).unwrap_or(true);
            let adv = clock_state.advance(cfg, clock.map(|c| c[i]), session.map(|s| s[i]), accept);
            // `on_clock_reset = "error"`: hand the offending delta back so the
            // caller can name the row and column.
            if let Some(raw) = adv.backwards {
                return Err((raw, i));
            }
            if accept {
                rows_seen += 1;
                out.processed[ri] = true;
            }
            let want_coef = accept
                && (Some(i) == last
                    || (spec.coef_every > 0 && rows_seen % u64::from(spec.coef_every) == 0));
            plans.push(RowPlan {
                ri,
                i,
                d_clock: adv.d_clock,
                reset: adv.reset,
                blend: !adv.reset && adv.session_changed,
                accept,
                want_coef,
                w: w.unwrap_or(1.0),
            });
        }

        self.clock = clock_state;
        self.rows_seen = rows_seen;

        // ---- pass 2: the instances ----
        let drift_resets = spec.drift_action.as_deref() == Some("reset");
        let coupled = !self.drift.is_empty() && drift_resets;
        // `min_periods` is read by every instance; move it out so the split
        // below can borrow the rest of `self` mutably.
        let min_periods = std::mem::take(&mut self.min_periods);
        let mut insts = self.split_instances(spec, out, n_rows);
        if coupled && insts.len() > 1 {
            for pi in 0..plans.len() {
                let mut seen = false;
                for inst in insts.iter_mut() {
                    // false: the caller resets every instance below, once all
                    // of them have seen this row.
                    seen |= run_instance(
                        inst,
                        &plans[pi..=pi],
                        features,
                        targets,
                        &min_periods,
                        false,
                    );
                }
                if seen {
                    insts.iter_mut().for_each(Instance::reset);
                }
            }
        } else if insts.len() > 1 {
            use rayon::prelude::*;
            // Only reached when instances are independent, which (given
            // `coupled` above) means `drift_action` is not "reset".
            insts.par_iter_mut().for_each(|inst| {
                run_instance(inst, &plans, features, targets, &min_periods, false);
            });
        } else if let Some(inst) = insts.first_mut() {
            // One instance: a drift reset has nothing to coordinate with, so
            // it resets itself inline at the row that fired.
            run_instance(inst, &plans, features, targets, &min_periods, drift_resets);
        }
        drop(insts);
        self.min_periods = min_periods;
        Ok(())
    }

    /// Split per-instance state and output into disjoint pieces, so instances
    /// can run concurrently. Every `ChunkOut` buffer is laid out model-major,
    /// which is what makes an instance's region one contiguous slice; the
    /// state vectors are already `[mi]`-indexed.
    fn split_instances<'a>(
        &'a mut self,
        spec: &'a Spec,
        out: &'a mut ChunkOut,
        n_rows: usize,
    ) -> Vec<Instance<'a>> {
        let n = self.models.len();
        let block = out.n_slots * n_rows;
        let n_targets = if n == 0 || n_rows == 0 || out.lam_selected.is_empty() {
            0
        } else {
            out.lam_selected.len() / (n * n_rows)
        };

        let mut drift = self.drift.iter_mut();
        let mut resid_q = self.resid_q.iter_mut();
        let mut autocorr = self.autocorr.iter_mut();
        let mut metrics = self.metrics.iter_mut();
        let mut o_pred = out.pred.chunks_mut(block.max(1));
        let mut o_resid = out.resid.chunks_mut(block.max(1));
        let mut o_sigma = out.sigma.chunks_mut(block.max(1));
        let mut o_resid_z = out.resid_z.chunks_mut(block.max(1));
        let mut o_autocorr = out.autocorr.chunks_mut(block.max(1));
        let mut o_metrics = out.metrics.chunks_mut((3 * block).max(1));
        let mut o_resid_q = out.resid_q.chunks_mut((out.n_levels * block).max(1));
        let mut o_drift = out.drift.chunks_mut(block.max(1));
        let mut o_n_eff = out.n_eff.chunks_mut(n_rows.max(1));
        let mut o_lam = out.lam_selected.chunks_mut((n_targets * n_rows).max(1));
        let mut o_coef = out.coef.iter_mut();

        let n_slots = out.n_slots;
        let mut models = self.models.iter_mut();
        let mut decays = self.decays.iter();
        let mut resid_var = self.resid_var.iter_mut();
        let mut resid_w = self.resid_w.iter_mut();
        let mut scratch = self.scratch.iter_mut();

        // Pulled in lockstep: each iterator yields disjoint `&mut`s, so every
        // Instance owns its own piece of everything.
        (0..n)
            .enumerate()
            .map(|(mi, _)| Instance {
                spec,
                mi,
                model: &mut models.next().expect("one per instance").1,
                decay: *decays.next().expect("one per instance"),
                resid_var: resid_var.next().expect("one per instance"),
                resid_w: resid_w.next().expect("one per instance"),
                drift: drift.next(),
                resid_q: resid_q.next(),
                autocorr: autocorr.next(),
                metrics: metrics.next(),
                scratch: scratch.next().expect("one per instance"),
                n_slots,
                n_rows,
                o_pred: o_pred.next().unwrap_or_default(),
                o_resid: o_resid.next().unwrap_or_default(),
                o_sigma: o_sigma.next().unwrap_or_default(),
                o_resid_z: o_resid_z.next().unwrap_or_default(),
                o_autocorr: o_autocorr.next().unwrap_or_default(),
                o_metrics: o_metrics.next().unwrap_or_default(),
                o_resid_q: o_resid_q.next().unwrap_or_default(),
                o_drift: o_drift.next().unwrap_or_default(),
                o_n_eff: o_n_eff.next().unwrap_or_default(),
                o_lam: o_lam.next().unwrap_or_default(),
                o_coef: o_coef.next().expect("one per instance"),
            })
            .collect()
    }
}

/// What pass 1 decided about one row, so pass 2 can replay it per instance
/// without touching the clock again.
struct RowPlan {
    /// Position within the chunk (index into the output buffers).
    ri: usize,
    /// Absolute row in the DataFrame (index into the input columns).
    i: usize,
    d_clock: f64,
    reset: bool,
    blend: bool,
    accept: bool,
    want_coef: bool,
    w: f64,
}

/// Per-row scratch, one set per model instance so instances can run
/// concurrently without sharing buffers.
#[derive(Default)]
pub struct Scratch {
    xs: Vec<f64>,
    ys: Vec<Option<f64>>,
    r: Vec<f64>,
    sig: Vec<f64>,
    zs: Vec<f64>,
}

/// One model instance's state and its disjoint slice of the chunk output.
struct Instance<'a> {
    /// For rebuilding this instance on a reset -- the spec is the only
    /// description of a pristine model, and `mi` picks this one out of it.
    spec: &'a Spec,
    mi: usize,
    model: &'a mut AnyModel,
    decay: Decay,
    resid_var: &'a mut Vec<f64>,
    resid_w: &'a mut Vec<f64>,
    drift: Option<&'a mut Vec<PageHinkley>>,
    resid_q: Option<&'a mut Vec<Vec<P2Quantile>>>,
    autocorr: Option<&'a mut Vec<EwAutoCorr>>,
    metrics: Option<&'a mut Vec<SlotMetrics>>,
    scratch: &'a mut Scratch,
    n_slots: usize,
    n_rows: usize,
    o_pred: &'a mut [f64],
    o_resid: &'a mut [f64],
    o_sigma: &'a mut [f64],
    o_resid_z: &'a mut [f64],
    o_autocorr: &'a mut [f64],
    o_metrics: &'a mut [f64],
    o_resid_q: &'a mut [f64],
    o_drift: &'a mut [bool],
    o_n_eff: &'a mut [f64],
    o_lam: &'a mut [f64],
    o_coef: &'a mut Vec<Option<Vec<f64>>>,
}

impl Instance<'_> {
    /// Restart this instance: the same thing a clock reset or a drift break
    /// does, applied to one instance rather than the whole stream.
    fn reset(&mut self) {
        let spec = self.spec;
        *self.model = build_models(spec)
            .expect("spec was already validated")
            .swap_remove(self.mi)
            .1;
        self.resid_var.iter_mut().for_each(|v| *v = 0.0);
        self.resid_w.iter_mut().for_each(|v| *v = 0.0);
        if let Some(d) = self.drift.as_deref_mut() {
            d.iter_mut().for_each(PageHinkley::reset);
        }
        // Residual diagnostics restart with the model they describe.
        if let (Some(q), Some(levels)) = (self.resid_q.as_deref_mut(), &spec.resid_quantiles) {
            for per_level in q.iter_mut() {
                for (est, lvl) in per_level.iter_mut().zip(levels) {
                    *est = P2Quantile::new(*lvl).expect("validated");
                }
            }
        }
        if let Some(a) = self.autocorr.as_deref_mut() {
            let lag = spec.resid_autocorr_lag.unwrap_or(1);
            a.iter_mut()
                .for_each(|e| *e = EwAutoCorr::new(lag).expect("validated"));
        }
        if let Some(m) = self.metrics.as_deref_mut() {
            m.iter_mut().for_each(|s| *s = SlotMetrics::new());
        }
    }
}

/// Run one model instance over a run of rows. Returns whether any drift
/// detector fired, which is all the caller needs to decide about a reset.
///
/// This is the whole per-row arithmetic, and the only copy of it: the parallel
/// path calls it once per instance with every row, the drift-coupled path once
/// per instance per row.
fn run_instance(
    inst: &mut Instance<'_>,
    plans: &[RowPlan],
    features: &[Vec<f64>],
    targets: &[Vec<f64>],
    min_periods: &[f64],
    // `drift_action = "reset"` with nothing else to coordinate with: restart
    // *at the row that fired*, not at the end of the chunk, or the rest of the
    // chunk keeps learning from the regime the detector just rejected.
    reset_on_drift: bool,
) -> bool {
    let mut drift_seen = false;
    let n_rows = inst.n_rows;
    let block = inst.n_slots * n_rows;
    for plan in plans {
        if plan.reset {
            inst.reset();
        } else if plan.blend {
            // A gentler alternative to resetting: revert partway toward the
            // long-run relationship (ENHANCEMENTS E6).
            inst.model.blend_toward_long_run();
        }
        if !plan.accept {
            continue;
        }
        let (i, ri, w) = (plan.i, plan.ri, plan.w);
        let sc = &mut *inst.scratch;
        sc.xs.clear();
        sc.xs.extend(features.iter().map(|f| f[i]));
        sc.ys.clear();
        sc.ys
            .extend(targets.iter().map(|t| Some(t[i]).filter(|f| usable(*f))));
        let m_targets = sc.ys.len();

        let mut step = inst.model.step(&sc.xs, &sc.ys, plan.d_clock, w);
        let n_slots = step.pred.len();
        // `ew_cov` has no targets; its slots are statistics, so every one
        // maps to "target 0" for the warmup check.
        let nc = n_slots.checked_div(m_targets).unwrap_or(1);

        // Per-target warmup (ENHANCEMENTS E7). The model itself predicts once
        // the *smallest* threshold is met; a slot whose own target is not ready
        // is withheld here, before it can reach the residual, sigma, resid_z,
        // drift or selection. Warmup gates output, not learning -- the model
        // has already updated from this row.
        for (slot, p) in step.pred.iter_mut().enumerate() {
            if step_n_eff_below(step.n_eff, min_periods, slot / nc) {
                *p = f64::NAN;
            }
        }

        sc.r.clear();
        sc.r.extend(step.pred.iter().enumerate().map(|(slot, p)| {
            match sc.ys.get(slot / nc).copied().flatten() {
                Some(yj) if p.is_finite() => yj - p,
                _ => f64::NAN,
            }
        }));
        sc.sig.clear();
        sc.sig.resize(n_slots, f64::NAN);
        sc.zs.clear();
        sc.zs.resize(n_slots, f64::NAN);

        // sigma is read from the state BEFORE this row's residual is folded in,
        // so `resid_z` is out-of-sample like the prediction it scales.
        let lam = inst.decay.factor(plan.d_clock);
        for (slot, &rv) in sc.r.iter().enumerate() {
            if inst.resid_w[slot] > 0.0 {
                let sd = inst.resid_var[slot].max(0.0).sqrt();
                sc.sig[slot] = sd;
                if rv.is_finite() && sd > 0.0 {
                    sc.zs[slot] = rv / sd;
                }
            }
            if rv.is_finite() {
                let w_new = lam * inst.resid_w[slot] + w;
                if w_new > 0.0 {
                    inst.resid_var[slot] =
                        (lam * inst.resid_w[slot] * inst.resid_var[slot] + w * rv * rv) / w_new;
                    inst.resid_w[slot] = w_new;
                }
            } else {
                inst.resid_w[slot] *= lam;
            }
        }

        for (slot, (&p, &rv)) in step.pred.iter().zip(sc.r.iter()).enumerate() {
            let at = slot * n_rows + ri;
            inst.o_pred[at] = p;
            inst.o_resid[at] = rv;
        }
        if !inst.o_sigma.is_empty() {
            for (slot, (&s, &z)) in sc.sig.iter().zip(sc.zs.iter()).enumerate() {
                let at = slot * n_rows + ri;
                inst.o_sigma[at] = s;
                inst.o_resid_z[at] = z;
            }
        }

        // Drift is monitored on |resid| scaled by the slot's own EW residual
        // std, so `drift_delta` means the same thing whatever the target's
        // units. Rows with no residual are skipped, not treated as zero error.
        let mut row_drift = false;
        if let Some(dets) = inst.drift.as_deref_mut() {
            for (slot, &rv) in sc.r.iter().enumerate() {
                let scale = sc.sig[slot];
                if rv.is_finite() && scale.is_finite() && scale > 0.0 {
                    let flag = dets[slot].update(rv.abs() / scale);
                    inst.o_drift[slot * n_rows + ri] = flag;
                    row_drift |= flag;
                }
            }
            drift_seen |= row_drift;
        }

        // Residual diagnostics (ENHANCEMENTS E23), all read before the row's
        // own residual is folded in, like sigma.
        if let Some(ests) = inst.resid_q.as_deref_mut() {
            for (slot, per_level) in ests.iter_mut().enumerate() {
                for (li, est) in per_level.iter_mut().enumerate() {
                    inst.o_resid_q[li * block + slot * n_rows + ri] = est.get().unwrap_or(f64::NAN);
                    if sc.r[slot].is_finite() {
                        est.update(sc.r[slot].abs());
                    }
                }
            }
        }
        if let Some(ests) = inst.autocorr.as_deref_mut() {
            for (slot, est) in ests.iter_mut().enumerate() {
                inst.o_autocorr[slot * n_rows + ri] = est.get().unwrap_or(f64::NAN);
                if sc.r[slot].is_finite() {
                    est.update(sc.r[slot], lam);
                }
            }
        }

        // Metrics are read before this row is scored, like every other
        // diagnostic here, so they never include the row they describe.
        if let Some(ms) = inst.metrics.as_deref_mut() {
            for (slot, met) in ms.iter_mut().enumerate() {
                let at = slot * n_rows + ri;
                inst.o_metrics[at] = met.ic().unwrap_or(f64::NAN);
                inst.o_metrics[block + at] = met.r2().unwrap_or(f64::NAN);
                inst.o_metrics[2 * block + at] = met.hit_rate().unwrap_or(f64::NAN);
                let yj = sc.ys.get(slot / nc).copied().flatten().unwrap_or(f64::NAN);
                met.update(step.pred[slot], yj, lam, w);
            }
        }

        inst.o_n_eff[ri] = step.n_eff;
        if let Some(online_core::Extra::Lasso { lam_selected }) = &step.extra {
            for (t_i, l) in lam_selected.iter().enumerate() {
                inst.o_lam[t_i * n_rows + ri] = *l;
            }
        }
        if plan.want_coef {
            // A model that has not solved yet has nothing to report, and
            // `null` is how every other output spells that. This used to be
            // an empty list, so `coef.list.get(i)` -- the documented way to
            // read one coefficient -- raised "index out of bounds" on the
            // warmup rows instead of returning null (IMPROVEMENTS U7).
            inst.o_coef[ri] = inst
                .model
                .coefficients()
                .map(|c| c.into_iter().flatten().collect());
        }
        if reset_on_drift && row_drift {
            inst.reset();
        }
    }
    drift_seen
}
