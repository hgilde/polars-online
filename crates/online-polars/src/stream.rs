//! One (spec, group) stream: clock state + model instances (one per halflife
//! grid entry), row-by-row processing with the docs/PLAN.md §3 null policy.

use online_core::{
    ClockState, Decay, EwAutoCorr, EwCovCfg, EwCovModel, EwCovStat, EwRidge, EwRidgeCfg, Ftrl,
    FtrlCfg, FtrlLoss, Kalman, KalmanCfg, Lasso, LassoCfg, LearningRate, ModelState, OnlineModel,
    P2Quantile, Pa, PaCfg, PaMode, PageHinkley, Rls, RlsCfg, Robust, RobustCfg, RobustLoss, Sgd,
    SgdCfg, SgdLoss, SlotMetrics, State, StateError,
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
        match self {
            AnyModel::EwRidge(m) => m.step(x, y, d_clock, weight),
            AnyModel::Rls(m) => m.step(x, y, d_clock, weight),
            AnyModel::Lasso(m) => m.step(x, y, d_clock, weight),
            AnyModel::Kalman(m) => m.step(x, y, d_clock, weight),
            AnyModel::Robust(m) => m.step(x, y, d_clock, weight),
            AnyModel::Ftrl(m) => m.step(x, y, d_clock, weight),
            AnyModel::EwCov(m) => m.step(x, y, d_clock, weight),
            AnyModel::Sgd(m) => m.step(x, y, d_clock, weight),
            AnyModel::Pa(m) => m.step(x, y, d_clock, weight),
        }
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
            | AnyModel::Pa(_) => 0,
        }
    }

    pub fn n_outputs(&self) -> usize {
        match self {
            AnyModel::EwRidge(m) => m.n_outputs(),
            AnyModel::Rls(m) => m.n_outputs(),
            AnyModel::Lasso(m) => m.n_outputs(),
            AnyModel::Kalman(m) => m.n_outputs(),
            AnyModel::Robust(m) => m.n_outputs(),
            AnyModel::Ftrl(m) => m.n_outputs(),
            AnyModel::EwCov(m) => m.n_outputs(),
            AnyModel::Sgd(m) => m.n_outputs(),
            AnyModel::Pa(m) => m.n_outputs(),
        }
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
        }
    }

    pub fn state(&self) -> State {
        match self {
            AnyModel::EwRidge(m) => m.state(),
            AnyModel::Rls(m) => m.state(),
            AnyModel::Lasso(m) => m.state(),
            AnyModel::Kalman(m) => m.state(),
            AnyModel::Robust(m) => m.state(),
            AnyModel::Ftrl(m) => m.state(),
            AnyModel::EwCov(m) => m.state(),
            AnyModel::Sgd(m) => m.state(),
            AnyModel::Pa(m) => m.state(),
        }
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
                clip_gradient: clip_gradient.unwrap_or(1e3),
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
    }
}

/// Combo labels per model instance ("" when there is only one combo).
pub fn combo_labels(spec: &Spec) -> Vec<String> {
    match &spec.model {
        ModelKind::EwRidge {
            ridge,
            feature_sets,
            ..
        } => {
            let nr = ridge.as_ref().map(|r| r.to_vec().len()).unwrap_or(1);
            let nf = feature_sets.as_ref().map(|f| f.len()).unwrap_or(0).max(1);
            if nr * nf == 1 {
                return vec![String::new()];
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
                    out.push(if nf == 1 {
                        format!("__r{r}")
                    } else if nr == 1 {
                        format!("__{f}")
                    } else {
                        format!("__{f}_r{r}")
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
        | ModelKind::Pa { .. } => vec![String::new()],
        ModelKind::Lasso { lasso_path, .. } => {
            lasso_path.iter().map(|l| format!("__l{l}")).collect()
        }
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
}

impl Stream {
    /// Summed over this stream's model instances (one per halflife).
    pub fn solve_failures(&self) -> u64 {
        self.models.iter().map(|(_, m)| m.solve_failures()).sum()
    }
}

/// True when this target has not reached its own warmup threshold yet.
#[inline]
fn step_n_eff_below(n_eff: f64, min_periods: &[f64], target: usize) -> bool {
    min_periods.get(target).is_some_and(|t| n_eff < *t)
}

/// Output of one row for one stream: `None` = skipped/emit-all-null.
pub struct RowOut {
    /// Per model instance, per output slot (NaN = null).
    pub pred: Vec<Vec<f64>>,
    /// `y_j - pred_slot`; NaN when the target is null or the pred not ready.
    pub resid: Vec<Vec<f64>>,
    /// EW residual standard deviation per slot, from the state *before* this
    /// row (so it is out-of-sample like everything else). NaN until there is
    /// at least one residual.
    pub sigma: Vec<Vec<f64>>,
    /// `resid / sigma`, NaN wherever either is.
    pub resid_z: Vec<Vec<f64>>,
    /// Page-Hinkley drift flag per slot (empty when `emit_drift` is off).
    pub drift: Vec<Vec<bool>>,
    /// `|resid|` quantiles per instance and slot, one entry per level.
    pub resid_q: Vec<Vec<Vec<f64>>>,
    /// EW residual autocorrelation per instance and slot.
    pub autocorr: Vec<Vec<f64>>,
    /// `(ic, r2, hit_rate)` per slot, NaN where not yet available.
    pub metrics: Vec<Vec<(f64, f64, f64)>>,
    pub n_eff: Vec<f64>,
    pub coef: Option<Vec<Vec<Vec<f64>>>>,
    pub extra: Vec<Option<online_core::Extra>>,
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

    fn reset_models(&mut self, spec: &Spec) {
        self.models = build_models(spec).expect("spec was already validated");
        for v in self.resid_var.iter_mut().chain(self.resid_w.iter_mut()) {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
        for d in self.drift.iter_mut() {
            d.iter_mut().for_each(PageHinkley::reset);
        }
        // Residual diagnostics restart with the model they describe.
        let (levels, lag) = (spec.resid_quantiles.clone(), spec.resid_autocorr_lag);
        if let Some(levels) = levels {
            for per_slot in self.resid_q.iter_mut() {
                for per_level in per_slot.iter_mut() {
                    for (est, q) in per_level.iter_mut().zip(&levels) {
                        *est = P2Quantile::new(*q).expect("validated");
                    }
                }
            }
        }
        for per_slot in self.autocorr.iter_mut() {
            for est in per_slot.iter_mut() {
                *est = EwAutoCorr::new(lag.unwrap_or(1)).expect("validated");
            }
        }
        for per_slot in self.metrics.iter_mut() {
            per_slot.iter_mut().for_each(|m| *m = SlotMetrics::new());
        }
    }

    /// Process one row. `emit_coef` forces a coefficient snapshot (last row of
    /// chunk); `coef_every` counts accepted rows.
    #[allow(clippy::too_many_arguments)]
    pub fn process_row(
        &mut self,
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        x: &[Option<f64>],
        y: &[Option<f64>],
        clock: Option<f64>,
        session: Option<u64>,
        // weight: None = no weight column; Some(NaN) = null value (skips the row)
        weight: Option<f64>,
        emit_coef: bool,
    ) -> Result<Option<RowOut>, f64> {
        let accept = x.iter().all(|v| v.is_some_and(f64::is_finite))
            && weight.map(|w| w.is_finite()).unwrap_or(true);
        let adv = self.clock.advance(cfg, clock, session, accept);
        // `on_clock_reset = "error"`: hand the offending delta back so the
        // caller can name the row and column.
        if let Some(raw) = adv.backwards {
            return Err(raw);
        }
        if adv.reset {
            self.reset_models(spec);
        } else if adv.session_changed {
            // A gentler alternative to resetting: revert partway toward the
            // long-run relationship (ENHANCEMENTS E6).
            for (_, m) in self.models.iter_mut() {
                m.blend_toward_long_run();
            }
        }
        if !accept {
            return Ok(None);
        }
        self.rows_seen += 1;
        let xs: Vec<f64> = x.iter().map(|v| v.unwrap()).collect();
        let ys: Vec<Option<f64>> = y.iter().map(|v| v.filter(|f| f.is_finite())).collect();
        let w = weight.unwrap_or(1.0);

        let want_coef =
            emit_coef || (spec.coef_every > 0 && self.rows_seen % u64::from(spec.coef_every) == 0);
        let mut pred = Vec::with_capacity(self.models.len());
        let mut resid = Vec::with_capacity(self.models.len());
        let mut n_eff = Vec::with_capacity(self.models.len());
        let mut extra = Vec::with_capacity(self.models.len());
        let mut coef = if want_coef { Some(Vec::new()) } else { None };
        let m_targets = ys.len();
        let mut sigma = Vec::with_capacity(self.models.len());
        let mut resid_z = Vec::with_capacity(self.models.len());
        let mut drift = Vec::with_capacity(self.drift.len());
        let mut drift_seen = false;
        let mut resid_q = Vec::with_capacity(self.resid_q.len());
        let mut autocorr = Vec::with_capacity(self.autocorr.len());
        let mut metrics = Vec::with_capacity(self.metrics.len());
        for (mi, (_, m)) in self.models.iter_mut().enumerate() {
            let mut step = m.step(&xs, &ys, adv.d_clock, w);
            let nc = step.pred.len() / m_targets;

            // Per-target warmup (ENHANCEMENTS E7). The model itself predicts
            // once the *smallest* threshold is met; a slot whose own target is
            // not ready is withheld here, before it can reach the residual,
            // sigma, resid_z, drift or selection. Warmup gates output, not
            // learning -- the model has already updated from this row.
            for (slot, p) in step.pred.iter_mut().enumerate() {
                if step_n_eff_below(step.n_eff, &self.min_periods, slot / nc) {
                    *p = f64::NAN;
                }
            }

            let r: Vec<f64> = step
                .pred
                .iter()
                .enumerate()
                .map(|(slot, p)| match ys[slot / nc] {
                    Some(yj) if p.is_finite() => yj - p,
                    _ => f64::NAN,
                })
                .collect();

            // sigma is read from the state BEFORE this row's residual is folded
            // in, so `resid_z` is out-of-sample like the prediction it scales.
            let lam = self.decays[mi].factor(adv.d_clock);
            let vars = &mut self.resid_var[mi];
            let wsum = &mut self.resid_w[mi];
            let mut sig = vec![f64::NAN; r.len()];
            let mut zs = vec![f64::NAN; r.len()];
            for (slot, &ri) in r.iter().enumerate() {
                if wsum[slot] > 0.0 {
                    let sd = vars[slot].max(0.0).sqrt();
                    sig[slot] = sd;
                    if ri.is_finite() && sd > 0.0 {
                        zs[slot] = ri / sd;
                    }
                }
                if ri.is_finite() {
                    let w_new = lam * wsum[slot] + w;
                    if w_new > 0.0 {
                        vars[slot] = (lam * wsum[slot] * vars[slot] + w * ri * ri) / w_new;
                        wsum[slot] = w_new;
                    }
                } else {
                    wsum[slot] *= lam;
                }
            }

            // Drift is monitored on |resid| scaled by the slot's own EW
            // residual std, so `drift_delta` means the same thing whatever the
            // target's units. Rows with no residual are skipped, not treated as
            // zero error.
            if !self.drift.is_empty() {
                let dets = &mut self.drift[mi];
                let mut flags = vec![false; r.len()];
                for (slot, &ri) in r.iter().enumerate() {
                    let scale = sig[slot];
                    if ri.is_finite() && scale.is_finite() && scale > 0.0 {
                        flags[slot] = dets[slot].update(ri.abs() / scale);
                        drift_seen |= flags[slot];
                    }
                }
                drift.push(flags);
            }

            // Residual diagnostics (ENHANCEMENTS E23), all read before the
            // row's own residual is folded in, like sigma.
            if !self.resid_q.is_empty() {
                let ests = &mut self.resid_q[mi];
                let mut row = Vec::with_capacity(ests.len());
                for (slot, per_level) in ests.iter_mut().enumerate() {
                    let mut vals = Vec::with_capacity(per_level.len());
                    for est in per_level.iter_mut() {
                        vals.push(est.get().unwrap_or(f64::NAN));
                        if r[slot].is_finite() {
                            est.update(r[slot].abs());
                        }
                    }
                    row.push(vals);
                }
                resid_q.push(row);
            }
            if !self.autocorr.is_empty() {
                let ests = &mut self.autocorr[mi];
                let mut row = vec![f64::NAN; r.len()];
                for (slot, est) in ests.iter_mut().enumerate() {
                    row[slot] = est.get().unwrap_or(f64::NAN);
                    if r[slot].is_finite() {
                        est.update(r[slot], lam);
                    }
                }
                autocorr.push(row);
            }

            // Metrics are read before this row is scored, like every other
            // diagnostic here, so they never include the row they describe.
            if !self.metrics.is_empty() {
                let ms = &mut self.metrics[mi];
                let mut row = Vec::with_capacity(ms.len());
                for (slot, met) in ms.iter_mut().enumerate() {
                    row.push((
                        met.ic().unwrap_or(f64::NAN),
                        met.r2().unwrap_or(f64::NAN),
                        met.hit_rate().unwrap_or(f64::NAN),
                    ));
                    let yj = ys[slot / nc].unwrap_or(f64::NAN);
                    met.update(step.pred[slot], yj, lam, w);
                }
                metrics.push(row);
            }

            pred.push(step.pred);
            resid.push(r);
            sigma.push(sig);
            resid_z.push(zs);
            n_eff.push(step.n_eff);
            extra.push(step.extra);
            if let Some(c) = &mut coef {
                c.push(m.coefficients().unwrap_or_default());
            }
        }
        // `drift_action = "reset"`: a detected break restarts this stream's
        // models, the same path a clock reset takes. The flags for this row are
        // still reported, so the reset is visible rather than silent.
        if drift_seen && spec.drift_action.as_deref() == Some("reset") {
            self.reset_models(spec);
        }

        Ok(Some(RowOut {
            pred,
            resid,
            sigma,
            resid_z,
            drift,
            resid_q,
            autocorr,
            metrics,
            n_eff,
            coef,
            extra,
        }))
    }
}
