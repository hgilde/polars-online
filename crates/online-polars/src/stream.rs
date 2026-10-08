//! One (spec, group) stream: clock state + model instances (one per half-life
//! grid entry), row-by-row processing with the docs/PLAN.md §3 null policy.

use online_core::{
    Bocpd, BocpdCfg, BocpdEmission, ChangeNorm, ClockState, Conformal, Constraint, CorrChange,
    CorrChangeCfg, CorrChangeKind, Covariance, Decay, Deco, DecoCfg, DecoDynamics, Disorder,
    EwAutoCorr, EwClass, EwClassCfg, EwCovCfg, EwCovModel, EwCovStat, EwQuantile, EwRidge,
    EwRidgeCfg, Ftrl, FtrlCfg, FtrlLoss, HitTest, Hmm, HmmCfg, Holt, HoltCfg, KMeans, KMeansCfg,
    Kalman, KalmanCfg, Lasso, LassoCfg, LearningRate, Marginal, MarginalCfg, Micro, MicroCfg,
    ModelState, OnlineModel, Pa, PaCfg, PaMode, PageHinkley, Rcov, RcovCfg, RcovKind, Rls, RlsCfg,
    Robust, RobustCfg, RobustLoss, SeedRule, SeqTest, SeqTestCfg, Sgd, SgdCfg, SgdLoss,
    SlotMetrics, State, StateError, WindowShadow,
};
use online_core::{ClockValue, ExactCaps, Stamp};
use serde::{Deserialize, Serialize};

use crate::arrow::ClockCol;
use crate::resid_window::ResidWindow;
use crate::rows::FeatureRows;
use crate::span::{Span, SpanList};
use crate::spec::{FloatOrList, ModelKind, ShardSpec, Spec};
use crate::summary::DataSummary;

/// A refused backwards clock, for the bank to turn into the error naming it.
#[derive(Debug, Clone, Copy)]
pub struct ClockRefusal {
    /// The raw delta, negative.
    pub raw: f64,
    /// The absolute row it happened at, in the chunk.
    pub row: usize,
    /// Under `"reset_state"`, the late-row minimum that refused it; `None`
    /// under `"error"`, which refuses every step back.
    pub disorder: Option<Disorder>,
    /// On a temporal clock, the step back in integer nanoseconds: `raw` is a
    /// double of seconds, which above about 104 days resolves more than a
    /// nanosecond, and the message names the step exactly (review
    /// 2026-09-28).
    pub back_ns: Option<i128>,
    /// On an integer clock, the step back in the column's own units,
    /// exactly (docs/PLAN.md task 200).
    pub back_int: Option<i128>,
}

impl ClockRefusal {
    fn new(
        raw: f64,
        row: usize,
        disorder: Option<Disorder>,
        clock: Option<online_core::ClockValue>,
        prev: Option<online_core::ClockValue>,
    ) -> Self {
        use online_core::ClockValue::Ns;
        let back_ns = match (clock, prev) {
            (Some(Ns(c)), Some(Ns(p))) => Some(i128::from(p) - i128::from(c)),
            _ => None,
        };
        let back_int = match (clock, prev) {
            (Some(c), Some(p)) => c.int_step(p).map(|step| -step),
            _ => None,
        };
        Self {
            raw,
            row,
            disorder,
            back_ns,
            back_int,
        }
    }
}

/// A refused clock step on a number clock: two values whose difference is
/// past the largest double, for the bank to turn into the error naming it
/// (review 2026-10-06, PB7).
#[derive(Debug, Clone, Copy)]
pub struct StepRefusal {
    /// The absolute row it happened at, in the chunk.
    pub row: usize,
    /// The clock value of the row before it in the stream, and its own.
    pub prev: f64,
    pub now: f64,
}

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
    KMeans(Box<KMeans>),
    Micro(Box<Micro>),
    EwClass(Box<EwClass>),
    SeqTest(Box<SeqTest>),
    Marginal(Box<Marginal>),
    Deco(Box<Deco>),
    Rcov(Box<Rcov>),
    Hmm(Box<Hmm>),
    CorrChange(Box<CorrChange>),
    Bocpd(Box<Bocpd>),
}

/// A flush's shards, run on whichever pool the caller is in: the bank's
/// own (pool.rs), since every learning path runs inside it.
fn on_the_pool(count: usize) -> online_core::Shards<'static> {
    fn run(shards: &mut [online_core::MarginalShard<'_>]) {
        use rayon::prelude::*;
        shards
            .par_iter_mut()
            .for_each(online_core::MarginalShard::run);
    }
    online_core::Shards { count, run: &run }
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
            AnyModel::KMeans($m) => $body,
            AnyModel::Micro($m) => $body,
            AnyModel::EwClass($m) => $body,
            AnyModel::SeqTest($m) => $body,
            AnyModel::Marginal($m) => $body,
            AnyModel::Deco($m) => $body,
            AnyModel::Rcov($m) => $body,
            AnyModel::Hmm($m) => $body,
            AnyModel::CorrChange($m) => $body,
            AnyModel::Bocpd($m) => $body,
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

    /// [`Self::step`], with a `marginal`'s pair work split into `shards`
    /// ranges of features and run on the pool a batch of rows at a time
    /// ([`online_core::Marginal::step_sharded`], docs/PLAN.md task 126).
    /// Every other model, and one shard, steps as `step` does. The rows a
    /// sharded step holds are learned by [`Self::flush`], which every run
    /// of rows ends with; the numbers are `step`'s to the bit.
    pub fn step_sharded(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        shards: usize,
    ) -> online_core::Step {
        match self {
            AnyModel::Marginal(m) if shards > 1 => {
                m.step_sharded(x, y, d_clock, weight, &on_the_pool(shards))
            }
            _ => self.step(x, y, d_clock, weight),
        }
    }

    /// End a run of rows: learn every row a sharded step holds
    /// ([`Self::step_sharded`]), and take the readiness shares of every
    /// solve an `ewridge` left unread, dropping the factors they wait on
    /// ([`online_core::EwRidge::settle_readiness`], docs/PLAN.md task 140).
    pub fn flush(&mut self, shards: usize) {
        match self {
            AnyModel::Marginal(m) => m.flush(&on_the_pool(shards)),
            AnyModel::EwRidge(m) => m.settle_readiness(),
            _ => {}
        }
    }

    /// The step's answer without the step
    /// ([`OnlineModel::predict_with`]). `y` is passed because two models
    /// read a number out of the targets slot -- `bocpd`'s hazard column and
    /// `hmm`'s exogenous column -- and their answer depends on it; every
    /// other model ignores it, which is what makes `predict` out of sample
    /// (docs/REVIEW-E54-E64.md C1).
    pub fn predict(&self, x: &[f64], y: &[Option<f64>], d_clock: f64) -> online_core::Step {
        dispatch!(self, m => m.predict_with(x, y, d_clock))
    }

    /// Drop the row-lagged state ([`OnlineModel::clear_lags`]); a no-op for
    /// a model that keeps none.
    pub fn clear_lags(&mut self) {
        dispatch!(self, m => m.clear_lags())
    }

    /// Hand the model the next learned row's stamp, its decayed clock held
    /// exactly ([`OnlineModel::stamp_next`], docs/PLAN.md task 175); a no-op
    /// for a model without a window.
    pub fn stamp_next(&mut self, stamp: Stamp) {
        dispatch!(self, m => m.stamp_next(stamp))
    }

    /// Bound the model's window ([`OnlineModel::set_window_budget`]).
    pub fn set_window_budget(&mut self, budget: Option<online_core::WindowBudget>) {
        dispatch!(self, m => m.set_window_budget(budget))
    }

    /// Set the model's window edge ([`OnlineModel::set_window_closed`]).
    pub fn set_window_closed(&mut self, closed: online_core::WindowClosed) {
        dispatch!(self, m => m.set_window_closed(closed))
    }

    /// The model's window ring as a shadow ([`OnlineModel::window_shadow`]).
    pub fn window_shadow(&self) -> Option<WindowShadow> {
        dispatch!(self, m => m.window_shadow())
    }

    /// The weight-share cadence ([`OnlineModel::set_solve_share`]).
    pub fn set_solve_share(&mut self, share: Option<f64>) {
        dispatch!(self, m => m.set_solve_share(share))
    }

    /// The weight-share cadence the model runs by ([`OnlineModel::solve_share`]).
    pub fn solve_share(&self) -> Option<f64> {
        dispatch!(self, m => m.solve_share())
    }

    /// A refusing budget's overrun ([`OnlineModel::window_over_budget`]).
    pub fn window_over_budget(&self) -> Option<(usize, online_core::Cadence)> {
        dispatch!(self, m => m.window_over_budget())
    }

    /// Each target's own weight, for the per-target warmup
    /// ([`OnlineModel::target_n_eff_into`]).
    pub fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        dispatch!(self, m => m.target_n_eff_into(out))
    }

    /// The noise gate's statistic per slot
    /// ([`OnlineModel::error_inflation_into`]); `false` for a model
    /// without one.
    pub fn error_inflation_into(&self, out: &mut Vec<f64>) -> bool {
        dispatch!(self, m => m.error_inflation_into(out))
    }

    /// The same for the noise gate at `limit`, where a bound may stand in
    /// below it ([`OnlineModel::error_inflation_gate_into`]).
    pub fn error_inflation_gate_into(&self, out: &mut Vec<f64>, limit: f64) -> bool {
        dispatch!(self, m => m.error_inflation_gate_into(out, limit))
    }

    /// The same for one row's features
    /// ([`OnlineModel::row_error_inflation_into`]).
    pub fn row_error_inflation_into(&self, x: &[f64], out: &mut Vec<f64>) -> bool {
        dispatch!(self, m => m.row_error_inflation_into(x, out))
    }

    /// Each coefficient's data share ([`OnlineModel::support_coef`]), laid
    /// out like [`Self::coefficients`].
    pub fn support_coef(&self) -> Option<Vec<Vec<f64>>> {
        dispatch!(self, m => m.support_coef())
    }

    /// What went wrong in a model's solves, counted (docs/PLAN.md §7):
    /// `ewridge` and `robust` count their jittered or failed factorizations,
    /// `lasso` its coordinate descents that ran out of sweeps, `ew_class` the
    /// rows whose scoring met a failed factorization, `hmm` the rows its
    /// filter could not evaluate, and `bocpd` the rows whose predictive could
    /// not be. Every other model reports 0 because it counts nothing, not
    /// because nothing can fail: `kalman` inverts an innovation variance, and
    /// `ew_cov` solves for `mahal` and `partial_corr` (review 2026-09-12, S8).
    pub fn solve_failures(&self) -> u64 {
        match self {
            AnyModel::EwRidge(m) => m.solve_failures,
            AnyModel::Lasso(m) => m.solve_failures,
            AnyModel::Robust(m) => m.solve_failures,
            AnyModel::EwClass(m) => m.solve_failures,
            AnyModel::Hmm(m) => m.solve_failures,
            AnyModel::Bocpd(m) => m.solve_failures,
            AnyModel::Rls(_)
            | AnyModel::Kalman(_)
            | AnyModel::Ftrl(_)
            | AnyModel::EwCov(_)
            | AnyModel::Sgd(_)
            | AnyModel::Pa(_)
            | AnyModel::Holt(_)
            | AnyModel::KMeans(_)
            | AnyModel::Micro(_)
            | AnyModel::SeqTest(_)
            | AnyModel::Marginal(_)
            | AnyModel::Deco(_)
            | AnyModel::Rcov(_)
            | AnyModel::CorrChange(_) => 0,
        }
    }

    pub fn n_outputs(&self) -> usize {
        dispatch!(self, m => m.n_outputs())
    }

    pub fn n_features(&self) -> usize {
        dispatch!(self, m => m.n_features())
    }

    pub fn n_targets(&self) -> usize {
        dispatch!(self, m => m.n_targets())
    }

    /// The accumulated weight behind the model as it stands: what the next
    /// row's `n_eff` field reports (before that row's update).
    pub fn n_eff(&self) -> f64 {
        dispatch!(self, m => m.n_eff())
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
            // The centres, k rows of p; absent until seeded.
            AnyModel::KMeans(m) => m.coefficients(),
            // micro: one row per potential summary -- ragged, and absent
            // until there is one.
            AnyModel::Micro(m) => m.coefficients(),
            // The class means, one row per class (NaN for a class not seen).
            AnyModel::EwClass(m) => Some(m.coefficients()),
            // seqtest has no coefficients: its outputs are the e-values and
            // the counts they are staked on.
            AnyModel::SeqTest(_) => None,
            // marginal has no coefficients and no outputs: its pairs are
            // read from the state by `Bank::marginal`.
            AnyModel::Marginal(_) => None,
            // The levels themselves, one row: `coef_fields` names one slot
            // per correlation value, term `rho`.
            AnyModel::Deco(m) => Some(vec![m.rho().to_vec()]),
            // rcov has no coefficients: its value is the block it emits at
            // the group's close.
            AnyModel::Rcov(_) => None,
            // The state means, one row per state, as `ew_class` reports its
            // class means.
            AnyModel::Hmm(m) => Some(
                (0..m.cfg().k)
                    .map(|s| m.state_cov(s).means().to_vec())
                    .collect(),
            ),
            // corrchange has no coefficients: its outputs are the test.
            AnyModel::CorrChange(_) => None,
            // bocpd has none either: its value is the run-length posterior.
            AnyModel::Bocpd(_) => None,
        }
    }

    pub fn state(&self) -> State {
        dispatch!(self, m => m.state())
    }

    /// The model's configuration as built, as JSON: what
    /// [`crate::resolved_defaults`] reads every model default from. Refused
    /// where the JSON would not read back as the configuration, so an
    /// infinity written as `null` cannot be pinned as one (`faithful_json`).
    pub fn cfg_json(&self) -> Result<serde_json::Value, String> {
        dispatch!(self, m => crate::defaults::faithful_json(m.cfg()))
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
            ModelState::KMeans(_) => Ok(AnyModel::KMeans(Box::new(KMeans::restore(s)?))),
            ModelState::Micro(_) => Ok(AnyModel::Micro(Box::new(Micro::restore(s)?))),
            ModelState::EwClass(_) => Ok(AnyModel::EwClass(Box::new(EwClass::restore(s)?))),
            ModelState::SeqTest(_) => Ok(AnyModel::SeqTest(Box::new(SeqTest::restore(s)?))),
            ModelState::Marginal(_) => Ok(AnyModel::Marginal(Box::new(Marginal::restore(s)?))),
            ModelState::Deco(_) => Ok(AnyModel::Deco(Box::new(Deco::restore(s)?))),
            ModelState::Rcov(_) => Ok(AnyModel::Rcov(Box::new(Rcov::restore(s)?))),
            ModelState::Hmm(_) => Ok(AnyModel::Hmm(Box::new(Hmm::restore(s)?))),
            ModelState::CorrChange(_) => {
                Ok(AnyModel::CorrChange(Box::new(CorrChange::restore(s)?)))
            }
            ModelState::Bocpd(_) => Ok(AnyModel::Bocpd(Box::new(Bocpd::restore(s)?))),
            other => Err(StateError::WrongModel {
                expected: "a bank-supported model",
                found: other.kind(),
            }),
        }
    }
}

/// The default `bins` the Python layer leaves to us: enough resolution to see
/// a threshold or a V, and small enough that the histogram stays a rounding
/// error next to the pair moments it rides along with. With `bin_edges` the
/// count is the edges' and this is not read.
const DEFAULT_BINS: usize = 16;

/// Learned rows held before the bin edges are fixed. A thousand rows put
/// roughly sixty in each of sixteen quantile bins -- enough for edges that do
/// not move much -- and hold 8 KB per feature while they wait.
const DEFAULT_BIN_WARM_ROWS: usize = 1_000;

/// Build the model instances for a spec: one per half-life grid entry. A
/// model's own refusal names the spec, as `Spec::validate` names it for
/// `bocpd`'s and the other cfg builders': in a bank of several specs
/// `sgd: clip_gradient must be > 0` did not say which (review 2026-10-06,
/// YA4). `decays` names it already.
pub fn build_models(spec: &Spec) -> Result<Vec<(String, AnyModel)>, String> {
    let decays = spec.decays()?;
    decays
        .into_iter()
        .map(|(suffix, decay)| {
            let m = build_one(spec, decay).map_err(|e| format!("spec {:?}: {e}", spec.name))?;
            Ok((suffix, m))
        })
        .collect()
}

/// The slope constraint of `sgd` / `pa` from the spec's `coef_min`,
/// `coef_max` and `coef_sum`: a scalar bound is broadcast to every feature,
/// a list is taken as given (its length is checked by the model's config,
/// which names the offence). `None` when nothing is constrained.
fn constraint(
    k: usize,
    coef_min: &Option<FloatOrList>,
    coef_max: &Option<FloatOrList>,
    coef_sum: Option<f64>,
) -> Option<Constraint> {
    if coef_min.is_none() && coef_max.is_none() && coef_sum.is_none() {
        return None;
    }
    let bounds = |b: &Option<FloatOrList>, none: f64| match b {
        None => vec![none; k],
        Some(FloatOrList::Float(v)) => vec![v.0; k],
        Some(list) => list.to_vec(),
    };
    let c = Constraint {
        lo: bounds(coef_min, f64::NEG_INFINITY),
        hi: bounds(coef_max, f64::INFINITY),
        sum: coef_sum,
    };
    // A list of the wrong length goes to the model's check, which names
    // it, whatever its bounds: dropped as trivial first, `coef_min =
    // ["-inf"]` with two features, or `[]`, passed (task 160, PB4).
    if c.lo.len() != k || c.hi.len() != k {
        return Some(c);
    }
    (!c.is_trivial()).then_some(c)
}

/// One model instance, its window bounded as its spec says
/// (`ModelKind::window_budget`). The budget is configuration, which a state
/// does not carry, so every model built -- a stream's, a reset's -- gets it
/// here, and every model restored gets it in [`Stream::restore`].
fn build_one(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    let mut m = build_bare(spec, decay)?;
    m.set_window_budget(spec.model.window_budget());
    // The window's edge travels with the state, so it is set once, here,
    // before the first row (docs/PLAN.md task 196, N17).
    m.set_window_closed(spec.model.window_edge());
    Ok(m)
}

/// The ring a windowed spread is cut from (review 2026-09-12, S1), for a
/// windowed model that predicts a target, when anything reads the spread:
/// the output fields, the ranking of `emit_selected` and `emit_averaged`,
/// the drift detector or the conformal band. Without a reader it would be
/// memory, and a budget to fail, for nothing.
fn resid_window(spec: &Spec) -> Result<Option<ResidWindow>, String> {
    let Some((window, cadence)) = spec.model.window_and_cadence() else {
        return Ok(None);
    };
    let read = Buffers::of(spec).extras || spec.emit_drift || spec.conformal.is_some();
    if spec.model.predicts_no_target() || !read {
        return Ok(None);
    }
    ResidWindow::new(
        window,
        cadence,
        spec.model.window_edge(),
        spec.model.window_budget(),
    )
    .map(Some)
}

fn build_bare(spec: &Spec, decay: Decay) -> Result<AnyModel, String> {
    match &spec.model {
        ModelKind::EwRidge {
            ridge,
            feature_sets,
            standardize,
            ridge_scale,
            coef_prior,
            session_shrink,
            long_half_life,
            solve_every,
            max_rows_between_solves,
            gram_block_rows,
            target_gaps,
            window_size: window,
            window_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
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
                fit_intercept: spec.fit_intercept,
                decay,
                ridge: ridge
                    .as_ref()
                    .map(FloatOrList::to_vec)
                    .unwrap_or_else(|| vec![1e-6]),
                feature_sets: fs,
                standardize: *standardize,
                ridge_scale: *ridge_scale == crate::spec::RidgeScale::Sum,
                session_shrink: *session_shrink,
                long_half_life: long_half_life.as_ref().map(Span::value),
                coef_prior: coef_prior.clone(),
                // The spec's default is 0, the noise gate being this model's
                // (docs/WARMUP-AND-CONVERGENCE.md §2.1); the model's own
                // floor -- its first solve, and its own gate -- is then a
                // row per unknown, as the old default was. An explicit value
                // is the user's, above or below that.
                min_weight: match spec.min_weight {
                    Some(_) => spec.min_periods_or_default(),
                    None => (spec.k() + usize::from(spec.fit_intercept)) as f64,
                },
                solve_every: solve_every
                    .as_ref()
                    .map_or_else(|| spec.solve_every_default(decay), Span::value),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
                gram_block_rows: gram_block_rows.unwrap_or(0),
                target_gaps: *target_gaps,
                window: window.as_ref().map(Span::value),
                window_every: window_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
            };
            let mut m = EwRidge::new(cfg)?;
            // The per-row leverage needs the factors kept (§2.1).
            m.set_keep_factor(spec.emit_error_inflation);
            // Each target's own threshold, for its own first solve (task
            // 195, S9b); left out, every target's is the model's.
            if spec.min_weight.is_some() {
                m.set_target_min_weight(spec.min_periods_per_target())?;
            }
            Ok(AnyModel::EwRidge(Box::new(m)))
        }
        ModelKind::Rls { delta, coef_prior } => {
            let cfg = RlsCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                delta: delta.unwrap_or(1.0),
                coef_prior: coef_prior.clone(),
                min_weight: spec.min_periods_or_default(),
            };
            Ok(AnyModel::Rls(Box::new(Rls::new(cfg)?)))
        }
        ModelKind::Lasso {
            lasso_path,
            l1_ratio,
            select_half_life,
            solve_every,
            max_rows_between_solves,
            max_iter,
            tol,
            target_gaps,
            window_size: window,
            window_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
        } => {
            let cfg = LassoCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                lasso_path: lasso_path.clone(),
                l1_ratio: l1_ratio.unwrap_or(1.0),
                select_half_life: select_half_life.as_ref().map(Span::value),
                min_weight: spec.min_periods_or_default(),
                // Each target's own, which its selection counts errors
                // from, as this layer gates its output (CA3).
                target_min_weight: spec.min_periods_per_target(),
                solve_every: solve_every
                    .as_ref()
                    .map_or_else(|| spec.solve_every_default(decay), Span::value),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
                window: window.as_ref().map(Span::value),
                window_every: window_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
                max_iter: max_iter.unwrap_or(100),
                tol: tol.unwrap_or(1e-10),
                target_gaps: *target_gaps,
            };
            Ok(AnyModel::Lasso(Box::new(Lasso::new(cfg)?)))
        }
        ModelKind::Kalman {
            coef_half_life,
            q,
            obs_var,
            p0,
            share_p,
            revert_half_life,
            standardize,
        } => {
            let cfg = KalmanCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                // None beside a `q`, which the model reads instead (review
                // 2026-10-06, PC6).
                half_life: coef_half_life
                    .as_ref()
                    .map_or_else(Vec::new, SpanList::to_vec),
                q: q.as_ref().map(|v| v.iter().map(|n| n.0).collect()),
                obs_var: *obs_var,
                p0: p0.unwrap_or(1.0),
                share_p: *share_p,
                min_weight: spec.min_periods_or_default(),
                revert_half_life: revert_half_life
                    .as_ref()
                    .map_or_else(|| vec![f64::INFINITY], SpanList::to_vec),
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
                fit_intercept: spec.fit_intercept,
                decay,
                // The 95%-efficiency constant (Huber 1981; statsmodels'
                // `HuberT`, scikit-learn's 1.35): task 195, U3.
                loss: RobustLoss::Huber {
                    delta: huber_delta.map_or(1.345, |n| n.0),
                },
                ridge: ridge.unwrap_or(1e-6),
                standardize: *standardize,
                min_weight: spec.min_periods_or_default(),
                solve_every: solve_every
                    .as_ref()
                    .map_or_else(|| spec.solve_every_default(decay), Span::value),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
                quantile_eps: 1e-3,
            };
            let mut m = Robust::new(cfg)?;
            // Each target's own threshold, for its own first solve (S9b).
            m.set_target_min_weight(spec.min_periods_per_target())?;
            Ok(AnyModel::Robust(Box::new(m)))
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
                fit_intercept: spec.fit_intercept,
                decay,
                loss: RobustLoss::Quantile { tau: *quantile },
                ridge: ridge.unwrap_or(1e-6),
                standardize: *standardize,
                min_weight: spec.min_periods_or_default(),
                solve_every: solve_every
                    .as_ref()
                    .map_or_else(|| spec.solve_every_default(decay), Span::value),
                max_rows_between_solves: max_rows_between_solves.unwrap_or(u32::MAX),
                solve_share: spec.solve_share_default(solve_every.as_ref(), decay),
                quantile_eps: quantile_eps.unwrap_or(0.2),
            };
            let mut m = Robust::new(cfg)?;
            // Each target's own threshold, for its own first solve (S9b).
            m.set_target_min_weight(spec.min_periods_per_target())?;
            Ok(AnyModel::Robust(Box::new(m)))
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
                fit_intercept: spec.fit_intercept,
                decay,
                alpha: alpha.unwrap_or(0.1),
                beta: beta.unwrap_or(1.0),
                l1: l1.unwrap_or(0.0),
                l2: l2.unwrap_or(1.0),
                min_weight: spec.min_periods_or_default(),
                strict_binary: *strict_binary,
                loss,
            };
            Ok(AnyModel::Ftrl(Box::new(Ftrl::new(cfg)?)))
        }
        ModelKind::EwCov {
            stats,
            precision_prior,
            mahal_quantiles,
            pca,
            pca_every,
            max_rows_between_pca,
            lags,
            window_size: window,
            window_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
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
                    "mahal" => Ok(EwCovStat::Mahal),
                    "lag_corr" => Ok(EwCovStat::LagCorr),
                    other => Err(format!("unknown ew_cov statistic {other:?}")),
                })
                .collect::<Result<Vec<_>, String>>()?;
            let cfg = EwCovCfg {
                n_features: spec.k(),
                decay,
                stats,
                // Floored at 2: a variance needs two rows, so a spec asking
                // for fewer (including 0) is quietly raised rather than
                // refused -- a covariance below two rows has nothing to
                // report (review 2026-09-18, minor).
                min_weight: spec.min_periods_per_target()[0].max(2.0),
                precision_prior: *precision_prior,
                mahal_quantiles: mahal_quantiles.clone().unwrap_or_default(),
                pca: pca.unwrap_or(0),
                // The regressions' schedule (task 161): the clock, the rows,
                // or every row when neither is given.
                pca_every: match (pca_every, max_rows_between_pca) {
                    (Some(e), _) => e.value(),
                    (None, Some(_)) => f64::INFINITY,
                    (None, None) => 0.0,
                },
                max_rows_between_pca: max_rows_between_pca.unwrap_or(u32::MAX),
                lags: lags.clone().unwrap_or_default(),
                window: window.as_ref().map(Span::value),
                window_every: window_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
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
            standardize,
            coef_min,
            coef_max,
            strict_binary,
            coef_sum,
        } => {
            let loss = match loss.as_deref().unwrap_or("squared") {
                "squared" => SgdLoss::Squared,
                // In units of the target's residual std, `huber`'s constant
                // under `huber`'s name (task 195, U1 and U3).
                "huber" => SgdLoss::Huber {
                    delta: huber_delta.map_or(1.345, |n| n.0),
                },
                "quantile" => SgdLoss::Quantile {
                    tau: quantile.ok_or("sgd: loss \"quantile\" needs a `quantile` level")?,
                },
                // 1% of the target's own spread: a band that does not
                // shrink must sit inside a good fit's errors (task 203).
                "epsilon_insensitive" => SgdLoss::EpsilonInsensitive {
                    eps: eps.unwrap_or(0.01),
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
                fit_intercept: spec.fit_intercept,
                decay,
                loss,
                learning_rate: learning_rate.unwrap_or(0.01),
                schedule: sched,
                l2: l2.unwrap_or(0.0),
                min_weight: spec.min_periods_or_default(),
                // Finite by default: see SgdCfg::clip_gradient.
                clip_gradient: clip_gradient.map_or(1e3, |n| n.0),
                standardize: *standardize,
                constraint: constraint(spec.k(), coef_min, coef_max, *coef_sum),
                strict_binary: *strict_binary,
            };
            Ok(AnyModel::Sgd(Box::new(Sgd::new(cfg)?)))
        }
        ModelKind::Pa {
            mode,
            c,
            eps,
            coef_min,
            coef_max,
            coef_sum,
            standardize,
        } => {
            let mode = match mode.as_deref().unwrap_or("pa1") {
                "pa" => PaMode::Pa,
                "pa1" => PaMode::Pa1,
                "pa2" => PaMode::Pa2,
                other => return Err(format!("unknown pa mode {other:?}")),
            };
            let cfg = PaCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                fit_intercept: spec.fit_intercept,
                decay,
                mode,
                c: c.map_or(1.0, |n| n.0),
                // As `sgd`'s tube: 1% of the target's own spread (task 203).
                eps: eps.unwrap_or(0.01),
                min_weight: spec.min_periods_or_default(),
                constraint: constraint(spec.k(), coef_min, coef_max, *coef_sum),
                standardize: *standardize,
            };
            Ok(AnyModel::Pa(Box::new(Pa::new(cfg)?)))
        }
        ModelKind::Holt {
            trend_half_life,
            trend,
        } => {
            // The level's half-life is the spec's own, so `half_life` means
            // the same thing here as it does for every other model; it had a
            // second name, `level_half_life`, until task 196.
            let level = match decay {
                Decay::Halflife(h) => h,
                // `lam = 1` forgets nothing, as for every other model; its
                // log is 0, and the division made it `-inf` (review
                // 2026-09-12, S30).
                Decay::Lam(1.0) => f64::INFINITY,
                Decay::Lam(l) => -std::f64::consts::LN_2 / l.ln(),
            };
            let cfg = HoltCfg {
                n_targets: spec.m(),
                level_half_life: level,
                trend_half_life: trend_half_life.as_ref().map_or(level * 4.0, Span::value),
                min_weight: spec.min_periods_or_default(),
                trend: trend.unwrap_or(true),
            };
            Ok(AnyModel::Holt(Box::new(Holt::new(cfg)?)))
        }
        ModelKind::KMeans {
            k,
            warm_rows,
            seed_rule,
            seed,
            update_every_rows,
            split_merge,
            split_merge_every_rows,
            dead_frac,
            standardize,
            scale_floor,
        } => {
            let seed_rule = match seed_rule.as_deref() {
                None | Some("lloyd") => SeedRule::Lloyd,
                Some("kmeanspp") => SeedRule::Kmeanspp,
                Some("farthest") => SeedRule::Farthest,
                Some("first") => SeedRule::First,
                // Unreachable after `Spec::validate`; a stale string is a bug
                // in the caller, not a runtime condition to swallow.
                Some(other) => return Err(format!("unknown kmeans seed_rule {other:?}")),
            };
            let cfg = KMeansCfg {
                n_features: spec.k(),
                k: *k,
                decay,
                // One `min_weight` for the whole model: the spec's first
                // (and only) entry, defaulting like the regressions do.
                min_weight: spec.min_periods_per_target()[0],
                // At least `k`: a smaller one given is refused, and left out
                // it is the buffer `max(warm_rows, k)` always was (review
                // 2026-10-06, PC8).
                warm_rows: warm_rows.unwrap_or_else(|| (*k).max(500)),
                seed_rule,
                seed: seed.unwrap_or(0),
                update_every_rows: update_every_rows.unwrap_or(1),
                split_merge: split_merge.unwrap_or(0.5),
                split_merge_every_rows: split_merge_every_rows.unwrap_or(100),
                // The dead rule runs at a split-merge check, so with none it
                // is 0, where a value above 0 is refused (CF6).
                dead_frac: dead_frac.unwrap_or(if *split_merge == Some(0.0) { 0.0 } else { 0.05 }),
                standardize: standardize.unwrap_or(true),
                scale_floor: scale_floor.unwrap_or(0.1),
            };
            Ok(AnyModel::KMeans(Box::new(KMeans::new(cfg)?)))
        }
        ModelKind::Micro {
            eps,
            beta_mu,
            max_clusters,
            prune_every,
            max_rows_between_prunes,
            macro_link,
            standardize,
            scale_floor,
        } => {
            let cfg = MicroCfg {
                n_features: spec.k(),
                decay,
                min_weight: spec.min_periods_per_target()[0],
                eps: *eps,
                beta_mu: beta_mu.unwrap_or(3.0),
                max_clusters: max_clusters.unwrap_or(200),
                // The regressions' schedule (task 163): the clock, the
                // learned rows, or, with neither, every 100 learned rows.
                prune_every: prune_every.as_ref().map_or(f64::INFINITY, Span::value),
                max_rows_between_prunes: match (prune_every, max_rows_between_prunes) {
                    (_, Some(r)) => *r,
                    (None, None) => 100,
                    (Some(_), None) => u32::MAX,
                },
                macro_link: *macro_link,
                standardize: standardize.unwrap_or(true),
                scale_floor: scale_floor.unwrap_or(0.1),
            };
            Ok(AnyModel::Micro(Box::new(Micro::new(cfg)?)))
        }
        ModelKind::EwClass {
            classes,
            covariance,
            precision_prior,
            window_size: window,
            window_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
        } => {
            let cfg = EwClassCfg {
                n_features: spec.k(),
                n_classes: classes.len(),
                decay,
                min_weight: spec.min_periods_per_target()[0],
                covariance: match covariance {
                    Some(c) => Covariance::parse("ew_class", c)?,
                    None => Covariance::Full,
                },
                precision_prior: *precision_prior,
                window: window.as_ref().map(Span::value),
                window_every: window_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
            };
            Ok(AnyModel::EwClass(Box::new(EwClass::new(cfg)?)))
        }
        // No decay to build with: `decays()` gives one undecayed instance.
        ModelKind::SeqTest { .. } => {
            let cfg = SeqTestCfg {
                n_targets: spec.m(),
                min_weight: spec.min_periods_or_default(),
            };
            Ok(AnyModel::SeqTest(Box::new(SeqTest::new(cfg)?)))
        }
        ModelKind::Marginal {
            window_size: window,
            window_every,
            max_rows_between_snapshots,
            window_budget: _,
            closed: _,
            // A spec-level acceptance of the price, checked in `validate`.
            window_lags: _,
            feature_moments,
            lags,
            serial_rule,
            cross_lags,
            bins,
            bin_rule,
            bin_warm_rows,
            bin_edges,
            bin_budget,
            // How the pairs are run, not what they are (`marginal_shards`).
            shards: _,
        } => {
            // The per-target thresholds go to the model whole: it is the
            // reader of its own state, so it gates each target's pairs
            // itself where a regression's stream layer would.
            let cfg = MarginalCfg {
                n_features: spec.k(),
                n_targets: spec.m(),
                decay,
                min_weight: spec.min_periods_per_target(),
                lags: lags.clone().unwrap_or_default(),
                cross_lags: cross_lags.clone(),
                serial_rule: match serial_rule.as_deref() {
                    None => None,
                    Some("truncated") => Some(online_core::SerialRule::Truncated),
                    Some("geometric") => Some(online_core::SerialRule::Geometric),
                    Some("bartlett") => Some(online_core::SerialRule::Bartlett),
                    Some(other) => {
                        return Err(format!(
                            "marginal: unknown serial_rule {other:?}; expected \"truncated\", \
                             \"bartlett\" or \"geometric\""
                        ));
                    }
                },
                bins: match (bins, bin_edges) {
                    (None, None) => None,
                    (_, edges) => Some(Box::new(online_core::BinCfg {
                        n_bins: bins.unwrap_or(DEFAULT_BINS),
                        edges: edges.clone(),
                        rule: match bin_rule.as_deref() {
                            None | Some("quantile") => online_core::BinRule::Quantile,
                            Some("fixed") => online_core::BinRule::Fixed,
                            Some(other) => {
                                return Err(format!(
                                    "marginal: unknown bin_rule {other:?}; expected \"quantile\" \
                                     or \"fixed\""
                                ));
                            }
                        },
                        warm_rows: bin_warm_rows.unwrap_or(DEFAULT_BIN_WARM_ROWS),
                        budget_mib: bin_budget.map(|n| n.0),
                    })),
                },
                feature_moments: match feature_moments.as_deref() {
                    None | Some("per_target") => online_core::FeatureMomentLayout::PerTarget,
                    Some("shared") => online_core::FeatureMomentLayout::Shared,
                    Some(other) => {
                        return Err(format!(
                            "marginal: unknown feature_moments {other:?}; expected \"per_target\" \
                             or \"shared\""
                        ));
                    }
                },
                window: window.as_ref().map(Span::value),
                window_every: window_every.as_ref().map(Span::value),
                max_rows_between_snapshots: max_rows_between_snapshots.map(|r| r as usize),
            };
            Ok(AnyModel::Marginal(Box::new(Marginal::new(cfg)?)))
        }
        ModelKind::Deco { .. } => {
            let mut cfg = deco_cfg(spec)?;
            cfg.decay = decay;
            Ok(AnyModel::Deco(Box::new(Deco::new(cfg)?)))
        }
        // No decay to build with, so the one undecayed instance `decays()`
        // gives is the only one.
        ModelKind::Rcov { .. } => Ok(AnyModel::Rcov(Box::new(Rcov::new(rcov_cfg(spec)?)?))),
        ModelKind::Hmm { .. } => {
            let mut cfg = hmm_cfg(spec)?;
            cfg.decay = decay;
            Ok(AnyModel::Hmm(Box::new(Hmm::new(cfg)?)))
        }
        ModelKind::CorrChange { .. } => {
            let mut cfg = corrchange_cfg(spec)?;
            cfg.decay = decay;
            Ok(AnyModel::CorrChange(Box::new(CorrChange::new(cfg)?)))
        }
        // No decay: the run-length posterior is what forgets.
        ModelKind::Bocpd { .. } => Ok(AnyModel::Bocpd(Box::new(Bocpd::new(bocpd_cfg(spec)?)?))),
    }
}

/// A `bocpd` spec's [`BocpdCfg`]; every parameter check is the model's.
pub fn bocpd_cfg(spec: &Spec) -> Result<BocpdCfg, String> {
    let ModelKind::Bocpd {
        hazard,
        hazard_col,
        emission,
        prior_mean,
        prior_kappa,
        prior_nu,
        prior_scale,
        robust_beta,
        prune_below,
        max_run,
        warm_rows,
    } = &spec.model
    else {
        return Err("not a bocpd spec".into());
    };
    Ok(BocpdCfg {
        n_features: spec.k(),
        // A number is the expected rows between changepoints; a duration,
        // which only a temporal clock reads (`Spec::clock_scale`), the
        // expected time between them, in the clock's seconds (task 179).
        hazard: hazard.as_ref().map_or(250.0, Span::value),
        hazard_from_row: hazard_col.is_some(),
        emission: match emission.as_deref() {
            None | Some("diag") => BocpdEmission::Diag,
            Some("gaussian") => BocpdEmission::Gaussian,
            Some("robust") => BocpdEmission::Robust,
            Some(other) => {
                return Err(format!(
                    "unknown bocpd emission {other:?}; expected \"gaussian\", \"diag\" or \
                     \"robust\""
                ));
            }
        },
        prior_mean: prior_mean.clone(),
        prior_kappa: prior_kappa.unwrap_or(1.0),
        prior_nu: *prior_nu,
        prior_scale: prior_scale.clone(),
        // Measured (`bocpd.rs`, `a_sustained_shift_survives_a_small_beta_
        // _and_not_a_large_one`): 0.1 ignores a 20-sigma row outright and
        // still finds a real shift within five rows; above ~0.2 nothing is
        // ever detected.
        robust_beta: robust_beta.unwrap_or(match emission.as_deref() {
            Some("robust") => 0.1,
            _ => 0.0,
        }),
        prune_below: prune_below.unwrap_or(1e-6),
        max_run: max_run.unwrap_or(10_000),
        min_weight: spec.min_periods_or_default(),
        warm_rows: *warm_rows,
        hazard_on_clock: hazard.as_ref().is_some_and(Span::is_duration),
    })
}

/// A `corrchange` spec's [`CorrChangeCfg`]; every parameter check is the
/// model's.
pub fn corrchange_cfg(spec: &Spec) -> Result<CorrChangeCfg, String> {
    let ModelKind::CorrChange {
        kind,
        span_rows,
        alpha,
        alpha_adjust,
        bandwidth,
        scalar,
        crit,
        n_perm,
        permute_every_rows,
        perm_block,
        norm,
        seed,
        reset_on_flag: reset,
        monitor_rows,
        boundary_gamma,
    } = &spec.model
    else {
        return Err("not a corrchange spec".into());
    };
    let (kind, name) = match kind.as_deref() {
        None | Some("monitor") => (CorrChangeKind::Monitor, "monitor"),
        Some("window") => (CorrChangeKind::Window, "window"),
        Some("sequential") => (CorrChangeKind::Sequential, "sequential"),
        Some(other) => {
            return Err(format!(
                "unknown corrchange kind {other:?}; expected \"monitor\", \"sequential\" or \
                 \"window\""
            ));
        }
    };
    let Some(span_rows) = *span_rows else {
        return Err(format!(
            "corrchange: kind = {name:?} needs `span_rows`, {}",
            match kind {
                CorrChangeKind::Sequential =>
                    "the rows of history each monitoring period is \
                                               tested against",
                _ => "the rows per comparison block",
            }
        ));
    };
    // A parameter that belongs to another kind is refused, not ignored: a
    // `crit` given to `"monitor"` changed nothing, in silence. A value equal
    // to the builders' default (`norm = "l1"`, `reset = false`) is taken as
    // unset, since the Python builder writes those whether or not asked.
    let refuse = |set: bool, param: &str, kinds: &str| -> Result<(), String> {
        if set {
            return Err(format!(
                "corrchange: {param} applies to kind = {kinds}, not {name:?}"
            ));
        }
        Ok(())
    };
    let (monitor, window, sequential) = (
        kind == CorrChangeKind::Monitor,
        kind == CorrChangeKind::Window,
        kind == CorrChangeKind::Sequential,
    );
    if !window {
        refuse(n_perm.is_some(), "n_perm", "\"window\"")?;
        refuse(
            permute_every_rows.is_some(),
            "permute_every_rows",
            "\"window\"",
        )?;
        refuse(perm_block.is_some(), "perm_block", "\"window\"")?;
        refuse(seed.is_some(), "seed", "\"window\"")?;
        refuse(
            norm.as_deref().is_some_and(|n| n != "l1"),
            "norm",
            "\"window\"",
        )?;
        refuse(
            *reset == Some(true),
            "reset_on_flag",
            "\"window\" (a \"monitor\" span and a \"sequential\" cycle end at their flag \
             already)",
        )?;
    }
    refuse(
        monitor && crit.is_some(),
        "crit",
        "\"window\" or \"sequential\" (\"monitor\"'s is the Kolmogorov quantile)",
    )?;
    refuse(
        window && bandwidth.is_some(),
        "bandwidth",
        "\"monitor\" or \"sequential\"",
    )?;
    // `"window"` takes its permutation quantile at `alpha` itself, so a
    // spread over the pairs changed nothing there (task 160, CD4). The
    // builders' default, "bonferroni", is taken as unset.
    refuse(
        window && alpha_adjust.as_deref().is_some_and(|a| a != "bonferroni"),
        "alpha_adjust",
        "\"monitor\" or \"sequential\" (\"window\" takes its permutation quantile at alpha)",
    )?;
    if !sequential {
        refuse(monitor_rows.is_some(), "monitor_rows", "\"sequential\"")?;
        refuse(boundary_gamma.is_some(), "boundary_gamma", "\"sequential\"")?;
    }
    Ok(CorrChangeCfg {
        n_features: spec.k(),
        kind,
        span_rows,
        alpha: alpha.unwrap_or(0.05),
        alpha_adjust: alpha_adjust.clone().unwrap_or_else(|| "bonferroni".into()),
        bandwidth: *bandwidth,
        scalar: scalar.unwrap_or(false),
        decay: online_core::Decay::Lam(1.0),
        crit: *crit,
        n_perm: n_perm.unwrap_or(200),
        permute_every_rows: permute_every_rows.unwrap_or(50),
        perm_block: perm_block.unwrap_or(1),
        norm: match norm.as_deref() {
            None | Some("l1") => ChangeNorm::L1,
            Some("linf") => ChangeNorm::LInf,
            Some(other) => {
                return Err(format!(
                    "unknown corrchange norm {other:?}; expected \"l1\" or \"linf\""
                ));
            }
        },
        seed: seed.unwrap_or(0),
        reset: reset.unwrap_or(false),
        // W&G's `T = 1`: as many rows monitored as the history has.
        monitor_rows: if sequential {
            monitor_rows.unwrap_or(span_rows)
        } else {
            0
        },
        boundary_gamma: boundary_gamma.unwrap_or(0.0),
    })
}

/// An `hmm` spec's [`HmmCfg`]. The decay is the caller's; every other check
/// is the model's, so `Spec::validate` gets the same messages.
pub fn hmm_cfg(spec: &Spec) -> Result<HmmCfg, String> {
    let ModelKind::Hmm {
        k,
        covariance,
        precision_prior,
        learn,
        transition_prior,
        transition,
        means,
        covs,
        warm_rows,
        seed_rule,
        seed,
        exog_tvtp,
        tvtp_coef,
    } = &spec.model
    else {
        return Err("not an hmm spec".into());
    };
    let tvtp = match (exog_tvtp, tvtp_coef) {
        (Some(_), Some(ab)) if ab.len() == 2 => Some((ab[0].clone(), ab[1].clone())),
        (Some(_), _) => {
            return Err(
                "hmm exog_tvtp needs tvtp_coef = [A, B], each a K x K matrix flattened row-major"
                    .into(),
            );
        }
        (None, Some(_)) => {
            return Err("hmm tvtp_coef needs exog_tvtp (the column it reads)".into());
        }
        (None, None) => None,
    };
    // A parameter its mode does not read is refused, not ignored (review
    // 2026-10-06, CE4): under `tvtp_coef` no transition count is learned,
    // and with the states given nothing is seeded. A value given is told
    // from the default here, where both are `None` until filled.
    for (param, given) in [
        ("transition", transition.is_some()),
        ("transition_prior", transition_prior.is_some()),
    ] {
        if given && tvtp.is_some() {
            return Err(format!(
                "hmm: {param} does not apply with tvtp_coef: the matrix is softmax(A + B·z) and \
                 no transition count is learned, so there is no prior to spread"
            ));
        }
    }
    for (param, given) in [
        ("warm_rows", warm_rows.is_some()),
        ("seed_rule", seed_rule.is_some()),
        ("seed", seed.is_some()),
    ] {
        if given && means.is_some() && covs.is_some() {
            return Err(format!(
                "hmm: {param} does not apply with means and covs given: the states are not \
                 seeded from the rows, so there is no warm-up"
            ));
        }
    }
    Ok(HmmCfg {
        n_features: spec.k(),
        k: *k,
        decay: online_core::Decay::Lam(1.0),
        covariance: match covariance {
            Some(c) => Covariance::parse("hmm", c)?,
            None => Covariance::Full,
        },
        precision_prior: *precision_prior,
        min_weight: spec.min_periods_or_default(),
        learn: learn.unwrap_or(true),
        transition_prior: transition_prior.unwrap_or(1.0),
        transition: transition.clone(),
        means: means.clone(),
        covs: covs.clone(),
        warm_rows: warm_rows.unwrap_or(50),
        seed_rule: match seed_rule.as_deref() {
            None | Some("lloyd") => SeedRule::Lloyd,
            Some("first") => SeedRule::First,
            Some("farthest") => SeedRule::Farthest,
            Some("kmeanspp") => SeedRule::Kmeanspp,
            Some(other) => {
                return Err(format!(
                    "unknown hmm seed_rule {other:?}; expected first, farthest, kmeanspp or lloyd"
                ));
            }
        },
        seed: seed.unwrap_or(0),
        tvtp,
    })
}

/// An `rcov` spec's [`RcovCfg`]. Every parameter check is the model's, so
/// `Spec::validate` and the CLI get one set of messages.
pub fn rcov_cfg(spec: &Spec) -> Result<RcovCfg, String> {
    let ModelKind::Rcov {
        kind,
        kernel,
        bandwidth,
        jitter,
        theta,
        psd,
        block_rows,
        max_bandwidth,
        preavg_rows,
        noise_stride,
        iv_stride,
    } = &spec.model
    else {
        return Err("not an rcov spec".into());
    };
    let kind = match kind.as_deref() {
        None | Some("kernel") => RcovKind::Kernel,
        Some("preavg") => RcovKind::Preavg,
        Some("plain") => RcovKind::Plain,
        Some(other) => {
            return Err(format!(
                "unknown rcov kind {other:?}; expected \"kernel\", \"preavg\" or \"plain\""
            ));
        }
    };
    // A parameter its kind does not read is refused, not ignored, as
    // `bandwidth` and `preavg_rows` are (review 2026-10-06, CE4): the jitter
    // and the ring are the kernel's, `theta` sets the pre-averaging window.
    // The builders leave both `None` unless given.
    for (param, given, owner) in [
        ("jitter", jitter.is_some(), RcovKind::Kernel),
        ("max_bandwidth", max_bandwidth.is_some(), RcovKind::Kernel),
        ("theta", theta.is_some(), RcovKind::Preavg),
    ] {
        if given && kind != owner {
            return Err(format!(
                "rcov: {param} applies to kind = {:?}, not {:?}",
                owner.as_str(),
                kind.as_str()
            ));
        }
    }
    Ok(RcovCfg {
        n_features: spec.k(),
        kind,
        kernel: kernel.clone().unwrap_or_else(|| "parzen".into()),
        bandwidth: *bandwidth,
        jitter: jitter.unwrap_or(2),
        theta: theta.unwrap_or(1.0),
        psd: psd.unwrap_or(true),
        block_rows: *block_rows,
        max_bandwidth: *max_bandwidth,
        preavg_rows: *preavg_rows,
        noise_stride: noise_stride.unwrap_or(1),
        iv_stride: iv_stride.unwrap_or(20),
    })
}

/// The named blocks of a `deco` spec, if it has any.
fn blocks_named(model: &ModelKind) -> Option<&Vec<(String, Vec<String>)>> {
    match model {
        ModelKind::Deco { blocks, .. } => blocks.as_ref(),
        _ => None,
    }
}

/// A `deco` spec's [`DecoCfg`], with the block *names* resolved to feature
/// positions. The decay is the caller's (one instance per half-life); every
/// other check is `DecoCfg::validate`'s, so `Spec::validate` gets the same
/// messages the model would give.
pub fn deco_cfg(spec: &Spec) -> Result<DecoCfg, String> {
    let ModelKind::Deco {
        dynamics,
        alpha,
        beta,
        blocks,
    } = &spec.model
    else {
        return Err("not a deco spec".into());
    };
    let dynamics = match dynamics.as_deref() {
        None | Some("ew") => DecoDynamics::Ew,
        Some("linear") => DecoDynamics::Linear,
        Some(other) => {
            return Err(format!(
                "unknown deco dynamics {other:?}; expected \"ew\" or \"linear\""
            ));
        }
    };
    let blocks = match blocks {
        None => Vec::new(),
        Some(named) if named.is_empty() => {
            // An empty list is not "one block of everything" and not the
            // unblocked form either; whichever was meant, say which
            // (docs/REVIEW-E54-E64.md D1).
            // One sentence: continued across the line without a `\`, the
            // literal carried a run of 18 spaces (task 160, PB8).
            return Err("deco blocks is empty; leave it out for the unblocked \
                        equicorrelation, or name at least one block"
                .into());
        }
        Some(named) => named
            .iter()
            .map(|(name, cols)| {
                cols.iter()
                    .map(|c| {
                        spec.features.iter().position(|f| f == c).ok_or_else(|| {
                            format!("deco block {name:?} names {c:?}, which is not a feature")
                        })
                    })
                    .collect::<Result<Vec<usize>, String>>()
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    // A JSON or TOML spec writes `blocks` as an array of pairs, so two of
    // them can carry the same name; the collision then surfaces as two
    // output fields called `u_u` rather than as the block list's problem.
    if let Some(named) = blocks_named(&spec.model) {
        let mut seen = std::collections::HashSet::new();
        if let Some((dup, _)) = named.iter().find(|(n, _)| !seen.insert(n.as_str())) {
            return Err(format!(
                "deco block {dup:?} is named twice; block names are the output labels, so \
                 they have to be distinct"
            ));
        }
    }
    Ok(DecoCfg {
        n_features: spec.k(),
        decay: online_core::Decay::Lam(1.0),
        dynamics,
        alpha: *alpha,
        beta: *beta,
        blocks,
        min_weight: spec.min_periods_or_default(),
    })
}

/// The block names of a `deco` spec, in emission order; empty when it has
/// none, which is the unblocked model.
pub fn deco_block_names(spec: &Spec) -> Vec<String> {
    match &spec.model {
        ModelKind::Deco {
            blocks: Some(named),
            ..
        } => named.iter().map(|(n, _)| n.clone()).collect(),
        _ => Vec::new(),
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
                    // The label names what varies; the machine values say what
                    // the slot is, whatever the count. A single combo carried
                    // none, and a single named set was dropped from a ridge
                    // grid's (review 2026-09-12, S5).
                    let label = if nr * nf == 1 {
                        String::new()
                    } else if nf == 1 {
                        format!("__r{}", crate::spec::num_label(*r))
                    } else if nr == 1 {
                        format!("__{f}")
                    } else {
                        format!("__{f}_r{}", crate::spec::num_label(*r))
                    };
                    out.push(Combo {
                        label,
                        ridge: Some(*r),
                        feature_set: feature_sets.is_some().then(|| f.clone()),
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
        | ModelKind::Holt { .. }
        | ModelKind::KMeans { .. }
        | ModelKind::Micro { .. }
        | ModelKind::EwClass { .. }
        | ModelKind::SeqTest { .. }
        | ModelKind::Marginal { .. }
        | ModelKind::Deco { .. }
        | ModelKind::Rcov { .. }
        | ModelKind::Hmm { .. }
        | ModelKind::CorrChange { .. }
        | ModelKind::Bocpd { .. } => vec![Combo::default()],
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

/// The clock column's type, kept per spec so that `scored_clock` and
/// `learned_clock` (docs/PLAN.md task 152) come out as the column came in:
/// a `Datetime` in its own unit and zone, a `Date`, a `Duration`, an
/// integer in its own width (task 200), a float for any other number clock,
/// and the group's row index as an integer with none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClockDtype {
    Rows,
    Numeric,
    Date,
    Datetime {
        unit: ClockUnit,
        tz: Option<String>,
    },
    Duration {
        unit: ClockUnit,
    },
    /// An integer clock up to 64 bits wide, held in integers (task 200).
    Int(IntWidth),
}

/// An integer clock column's dtype (docs/PLAN.md task 200).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntWidth {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ClockUnit {
    Ms,
    Us,
    Ns,
}

impl ClockDtype {
    /// From the spec's clock column as the chunk holds it: `dtype` is the
    /// column's when it is temporal, `None` otherwise.
    pub fn of(has_clock: bool, dtype: Option<&polars::prelude::DataType>) -> Self {
        use polars::prelude::{DataType as D, TimeUnit as T};
        let unit = |u: &T| match u {
            T::Milliseconds => ClockUnit::Ms,
            T::Microseconds => ClockUnit::Us,
            T::Nanoseconds => ClockUnit::Ns,
        };
        let width = |d: &D| {
            Some(match d {
                D::Int8 => IntWidth::I8,
                D::Int16 => IntWidth::I16,
                D::Int32 => IntWidth::I32,
                D::Int64 => IntWidth::I64,
                D::UInt8 => IntWidth::U8,
                D::UInt16 => IntWidth::U16,
                D::UInt32 => IntWidth::U32,
                D::UInt64 => IntWidth::U64,
                _ => return None,
            })
        };
        match (has_clock, dtype) {
            (false, _) => Self::Rows,
            (true, Some(d)) if width(d).is_some() => Self::Int(width(d).expect("matched")),
            (true, Some(D::Date)) => Self::Date,
            (true, Some(D::Datetime(u, tz))) => Self::Datetime {
                unit: unit(u),
                tz: tz.as_ref().map(|z| z.to_string()),
            },
            (true, Some(D::Duration(u))) => Self::Duration { unit: unit(u) },
            (true, _) => Self::Numeric,
        }
    }

    /// Nanoseconds in one unit of the column.
    fn ns_per_unit(&self) -> i64 {
        match self {
            Self::Date => 86_400 * 1_000_000_000,
            Self::Datetime { unit, .. } | Self::Duration { unit } => match unit {
                ClockUnit::Ms => 1_000_000,
                ClockUnit::Us => 1_000,
                ClockUnit::Ns => 1,
            },
            Self::Rows | Self::Numeric | Self::Int(_) => 1,
        }
    }

    /// Whether a chunk whose clock column is `now` reads its steps another
    /// way than this dtype, which the bank kept from its first chunk (task
    /// 200): an integer clock's steps are taken in integers and a float
    /// clock's in doubles, so a stream reads one or the other for its life;
    /// and an integer clock's fields come out in its width, which a wider
    /// column would not fit. A temporal clock is read in nanoseconds
    /// whatever its unit, and a spec's clock parameters keep it temporal.
    pub fn conflicts(&self, now: &ClockDtype) -> bool {
        match (self, now) {
            (Self::Int(a), Self::Int(b)) => a != b,
            (Self::Int(_), Self::Numeric) | (Self::Numeric, Self::Int(_)) => true,
            _ => false,
        }
    }

    /// The column's type as polars names it: what [`Self::array`] makes.
    pub fn dtype(&self) -> polars::prelude::DataType {
        use polars::prelude::{DataType as D, TimeUnit as T};
        let unit = |u: &ClockUnit| match u {
            ClockUnit::Ms => T::Milliseconds,
            ClockUnit::Us => T::Microseconds,
            ClockUnit::Ns => T::Nanoseconds,
        };
        match self {
            Self::Rows => D::Int64,
            Self::Numeric => D::Float64,
            Self::Int(w) => match w {
                IntWidth::I8 => D::Int8,
                IntWidth::I16 => D::Int16,
                IntWidth::I32 => D::Int32,
                IntWidth::I64 => D::Int64,
                IntWidth::U8 => D::UInt8,
                IntWidth::U16 => D::UInt16,
                IntWidth::U32 => D::UInt32,
                IntWidth::U64 => D::UInt64,
            },
            Self::Date => D::Date,
            // A zone polars gave the column is one it reads back.
            Self::Datetime { unit: u, tz } => D::Datetime(
                unit(u),
                polars::prelude::TimeZone::opt_try_new(tz.clone())
                    .ok()
                    .flatten(),
            ),
            Self::Duration { unit: u } => D::Duration(unit(u)),
        }
    }

    /// A frame's clock column of `values` (review round 4, N18): in the
    /// clock column's own dtype, exactly, as `scored_clock` is -- a
    /// `Datetime` in its unit and zone, a `Date`, a `Duration`, an integer
    /// in its own width (task 200), a `Float64` for any other number clock
    /// -- and a `Float64` of nulls where there is no clock column (`Rows`)
    /// or no chunk has said what it is (`None`).
    pub fn column(
        dtype: Option<&ClockDtype>,
        name: &str,
        values: &[Option<ClockValue>],
    ) -> polars::prelude::PolarsResult<polars::prelude::Column> {
        use polars::prelude::{Column, Series};
        match dtype {
            Some(d) if *d != Self::Rows => Ok(Column::from(Series::from_arrow(
                name.into(),
                d.array(values),
            )?)),
            _ => Ok(Column::new(
                name.into(),
                values
                    .iter()
                    .map(|c| c.map(ClockValue::seconds))
                    .collect::<Vec<Option<f64>>>(),
            )),
        }
    }

    /// The values as an Arrow array of the column's type, null where `None`.
    pub fn array(&self, values: &[Option<ClockValue>]) -> Box<dyn polars_arrow::array::Array> {
        use polars_arrow::array::PrimitiveArray;
        use polars_arrow::datatypes::{ArrowDataType, TimeUnit as AT};
        let au = |u: &ClockUnit| match u {
            ClockUnit::Ms => AT::Millisecond,
            ClockUnit::Us => AT::Microsecond,
            ClockUnit::Ns => AT::Nanosecond,
        };
        let per = self.ns_per_unit();
        let ns = |c: Option<ClockValue>| match c {
            Some(ClockValue::Ns(n)) => Some(n / per),
            _ => None,
        };
        // An integer clock's values in its width: each came from a column
        // of it, the bank refusing another width (`Self::conflicts`), so
        // none is out of its range; one that were would be null, not wrapped.
        fn ints<T>(values: &[Option<ClockValue>]) -> Box<dyn polars_arrow::array::Array>
        where
            T: polars_arrow::types::NativeType + TryFrom<i64>,
        {
            let v: Vec<Option<T>> = values
                .iter()
                .map(|c| match c {
                    Some(ClockValue::I64(x)) => T::try_from(*x).ok(),
                    _ => None,
                })
                .collect();
            Box::new(PrimitiveArray::<T>::from(v))
        }
        match self {
            Self::Numeric => {
                let v: Vec<Option<f64>> = values
                    .iter()
                    .map(|c| match c {
                        Some(ClockValue::F64(x)) => Some(*x),
                        _ => None,
                    })
                    .collect();
                Box::new(PrimitiveArray::<f64>::from(v))
            }
            Self::Int(w) => match w {
                IntWidth::I8 => ints::<i8>(values),
                IntWidth::I16 => ints::<i16>(values),
                IntWidth::I32 => ints::<i32>(values),
                IntWidth::I64 => ints::<i64>(values),
                IntWidth::U8 => ints::<u8>(values),
                IntWidth::U16 => ints::<u16>(values),
                IntWidth::U32 => ints::<u32>(values),
                IntWidth::U64 => ints::<u64>(values),
            },
            Self::Rows => {
                let v: Vec<Option<i64>> = values
                    .iter()
                    .map(|c| match c {
                        Some(ClockValue::F64(x)) => Some(*x as i64),
                        _ => None,
                    })
                    .collect();
                Box::new(PrimitiveArray::<i64>::from(v))
            }
            Self::Date => {
                let v: Vec<Option<i32>> = values.iter().map(|c| ns(*c).map(|d| d as i32)).collect();
                Box::new(PrimitiveArray::<i32>::from(v).to(ArrowDataType::Date32))
            }
            Self::Datetime { unit, tz } => {
                let v: Vec<Option<i64>> = values.iter().map(|c| ns(*c)).collect();
                let dt = ArrowDataType::Timestamp(au(unit), tz.clone().map(Into::into));
                Box::new(PrimitiveArray::<i64>::from(v).to(dt))
            }
            Self::Duration { unit } => {
                let v: Vec<Option<i64>> = values.iter().map(|c| ns(*c)).collect();
                Box::new(PrimitiveArray::<i64>::from(v).to(ArrowDataType::Duration(au(unit))))
            }
        }
    }
}

/// A break's events while they wait in the `embargo` buffer: a session
/// change, the long-run blend it asks for, and a gap past `gap_cap`
/// (docs/PLAN.md task 153). A skipped row's wait with the next accepted row,
/// across a chunk boundary too.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct HeldBreak {
    pub session_changed: bool,
    pub blend: bool,
    pub capped: bool,
}

impl HeldBreak {
    fn is_none(&self) -> bool {
        *self == Self::default()
    }
}

/// One accepted row waiting for its label to mature (`embargo`,
/// docs/ENHANCEMENTS.md E47).
///
/// It carries everything the models need to be stepped with later: the row
/// itself, the clock delta it arrived with -- so replaying the buffer in
/// order gives the models exactly the gap sequence they would have seen
/// without the delay -- its place on the elapsed clock, which its release
/// is measured from, and the break it arrived after, whose events run when
/// it is learned (docs/PLAN.md task 153).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingRow {
    /// The row's place on the elapsed clock, held exactly
    /// ([`online_core::ClockAdvance::elapsed_stamp`]): it is learned from
    /// once an accepted row's place is at least `embargo` past it. The
    /// elapsed clock counts the clock column's own steps rather than the
    /// capped delta, so a break never releases a row before its delay has
    /// passed (task 153), and two places compare exactly -- in integer
    /// nanoseconds on a temporal clock, by one subtraction of raw values on
    /// a number clock -- where an embargo counted down in doubles released
    /// rows a row late or early (task 176). `None` only in a damaged state,
    /// which [`Stream::restore`] refuses.
    #[serde(default)]
    pub arrived: Option<Stamp>,
    /// The delta the row arrived with, replayed when it is released.
    pub d_clock: f64,
    /// The row's stamp, its decayed clock held exactly, which a window keys
    /// the row's snapshot by when it is released (docs/PLAN.md task 175).
    #[serde(default)]
    pub stamp: Option<Stamp>,
    pub w: f64,
    pub xs: Vec<f64>,
    /// `None` for a target that was null on the row.
    pub ys: Vec<Option<f64>>,
    /// The break before this row -- a session change (with the long-run
    /// blend `session_shrink` asks for) or a gap past `gap_cap`, on this
    /// row or on rows skipped since the last accepted one -- applied to the
    /// models when the row is learned, after every row before the break.
    /// Applied when the row arrived, it forced every held row out first.
    #[serde(default)]
    pub session_changed: bool,
    #[serde(default)]
    pub blend: bool,
    #[serde(default)]
    pub capped: bool,
    /// The row's clock as the fields show it (task 152).
    #[serde(default)]
    pub clock: Option<ClockValue>,
    /// The row's number in the stream the spec's window core was fed
    /// (docs/PLAN.md task 104): what a resolution names. 0 for a spec
    /// without a formula target, where nothing reads it.
    #[serde(default)]
    pub seq: u64,
    /// Whether every formula target of the row is known: true from the
    /// start for a spec without one, and once the window core has resolved
    /// the row otherwise. A row is released only when it is.
    #[serde(default = "yes")]
    pub resolved: bool,
}

fn yes() -> bool {
    true
}

/// A spec's `embargo` as the release compares the elapsed clock with it
/// (docs/PLAN.md task 176): its clock units, and its integer nanoseconds
/// where it is a duration, so two temporal places compare as integers at
/// any length ([`Stamp::cmp_span_ns`]), as a window operator's edge does.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Embargo {
    units: f64,
    ns: Option<i128>,
}

impl Embargo {
    fn of(span: &Span) -> Self {
        Embargo {
            units: span.value(),
            ns: match span {
                Span::Duration(d) => Some(i128::from(d.nanos)),
                Span::Units(_) => None,
            },
        }
    }

    /// Whether a row that arrived at `then` on the elapsed clock has waited
    /// the embargo out at `now`: inclusive, so a row exactly one embargo
    /// back is learned, as the countdown's `<= 0` learned it. A place
    /// missing on either side, which no stream hands over, waits.
    fn passed(self, now: Option<Stamp>, then: Option<Stamp>) -> bool {
        match (now, then) {
            (Some(now), Some(then)) => {
                now.cmp_span_ns(then, self.units, self.ns) != std::cmp::Ordering::Less
            }
            _ => false,
        }
    }
}

/// What `coef_every` and `max_rows_between_coefs` ask of the `coef` field
/// (docs/PLAN.md task 178).
#[derive(Debug, Clone, Copy, PartialEq)]
enum CoefPlan {
    /// Neither given: each group's last accepted row in each chunk
    /// ([`last_accepted`]).
    ChunkEnds,
    /// A cadence, as `solve_every` and `max_rows_between_solves` schedule a
    /// solve: a `coef` row once the clock has moved `clock` units since the
    /// last, `0` every row, or at the `rows`-th accepted row since,
    /// whichever comes first; `INFINITY` and `u64::MAX` for the one not
    /// given.
    Cadence { clock: f64, rows: u64 },
}

impl CoefPlan {
    fn of(spec: &Spec) -> Self {
        match (&spec.coef_every, spec.max_rows_between_coefs) {
            (None, None) => CoefPlan::ChunkEnds,
            (every, rows) => CoefPlan::Cadence {
                clock: every.as_ref().map_or(f64::INFINITY, Span::value),
                rows: rows.map_or(u64::MAX, u64::from),
            },
        }
    }
}

/// Where a stream's `coef` cadence stands (docs/PLAN.md task 178): the
/// stamp its clock is measured from -- the last `coef` row's, or before the
/// first the group's start or the clock's last reset ([`coef_origin`]) --
/// and the accepted rows since. The clock is a [`Stamp`], the decayed
/// clock held exactly, as a window spaces its snapshots by it (task 175):
/// summed in doubles, two thousand steps of a millisecond come to less than
/// two seconds, and `coef_every = "2s"` would write a row late.
///
/// Moved only under a cadence, so a spec without one writes the state it
/// did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CoefCadence {
    /// `None` before the group's first row.
    pub from: Option<Stamp>,
    /// Accepted rows since `from`, rows of weight zero and rows with a null
    /// target included.
    pub rows: u64,
}

impl CoefCadence {
    fn is_unset(&self) -> bool {
        *self == Self::default()
    }

    /// One row under a cadence of `clock` units or `rows` accepted rows
    /// ([`CoefPlan::Cadence`]), whether the stream accepts the row or not,
    /// `now` its clock value and `adv` what the clock made of it: whether it
    /// is a `coef` row. The group's first row, and a row the clock resets
    /// at, start the cadence over, as they start the models over; a skipped
    /// row is never one, and moves the clock only as the next accepted
    /// row's stamp shows.
    fn step(
        &mut self,
        clock: f64,
        rows: u64,
        now: Option<ClockValue>,
        adv: &online_core::ClockAdvance,
        accept: bool,
    ) -> bool {
        if adv.reset || self.from.is_none() {
            *self = CoefCadence {
                from: Some(coef_origin(now)),
                rows: 0,
            };
        }
        // `advance_stamped` stamps every accepted row.
        debug_assert!(!accept || adv.stamp.is_some(), "an accepted row unstamped");
        let (true, Some(stamp), Some(from)) = (accept, adv.stamp, self.from) else {
            return false;
        };
        self.rows = self.rows.saturating_add(1);
        let due = self.rows >= rows
            || clock == 0.0
            || (clock.is_finite() && stamp.cmp_span(from, clock) != std::cmp::Ordering::Less);
        if due {
            *self = CoefCadence {
                from: Some(stamp),
                rows: 0,
            };
        }
        due
    }
}

/// The stamp a stream's `coef` clock is measured from at the group's first
/// row, or at a row the clock resets at (task 178). On a clock column, the
/// stamp that row takes ([`ClockState::advance_stamped`]): 0 on a temporal
/// clock, and the row's own value with nothing removed on a number clock,
/// whether the row is accepted or skipped. Without one, the place before
/// the row's: the clock is then the row's number, the first row being 1,
/// so `coef_every = N` writes the `N`-th, `2N`-th, ... row, as the count of
/// rows it read before task 178 did.
fn coef_origin(now: Option<ClockValue>) -> Stamp {
    match now {
        Some(ClockValue::Ns(_)) => Stamp::Ns(0),
        Some(ClockValue::I64(v)) => Stamp::Int(i128::from(v), 0.0),
        Some(v) => Stamp::Raw(v.seconds(), 0.0),
        None => Stamp::Raw(-1.0, 0.0),
    }
}

/// What a spec's window core resolved in one chunk for one group
/// (docs/PLAN.md task 104): per resolved row, in row order, its number,
/// the number of the row that resolved it -- the row that closed, cut or
/// discarded its windows -- and each formula target's value, NaN where the
/// window gave none, row-major. Flat, so a chunk of rows costs no
/// allocation per row. Applied in row order, so a resolution reaches the
/// buffer at the row that made it and never earlier; see the review under
/// task 104.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Resolutions {
    pub seqs: Vec<u64>,
    pub ats: Vec<u64>,
    pub values: Vec<f64>,
    pub width: usize,
}

impl Resolutions {
    /// The resolution of row `seq`, if the chunk made one: the number of
    /// the row that made it, and the values.
    pub fn find(&self, seq: u64) -> Option<(u64, &[f64])> {
        let i = self.seqs.binary_search(&seq).ok()?;
        Some((
            self.ats[i],
            &self.values[i * self.width..(i + 1) * self.width],
        ))
    }
}

/// How a stream's formula targets reach it: which targets are formulas,
/// each row's number in the bank's stream, in the laid-out row order the
/// other columns have, and what the chunk resolved for this stream's group.
#[derive(Clone, Copy)]
pub struct FormulaTargets<'a> {
    pub slots: &'a [usize],
    pub seqs: &'a [u64],
    pub resolved: Option<&'a Resolutions>,
}

/// Serialized per-stream state: each half-life's model, and the rest of the
/// stream as it runs ([`Persisted`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamState {
    pub models: Vec<State>,
    pub persisted: Persisted,
}

/// Everything a stream keeps across a save but its models: the fields
/// [`Stream`] runs on, held there as one value, so [`Stream::save`] is one
/// clone of it and a field added here is saved by construction, where
/// `save` once copied two dozen by hand and a forgotten one was silently
/// not persisted (docs/SIMPLIFICATION.md S6, review round 4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Persisted {
    /// The clock the stream's rows are measured on.
    pub clock: ClockState,
    /// Rows the null policy accepted: what `rows_processed` reports.
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
    pub resid_q: Vec<Vec<EwQuantile>>,
    #[serde(default)]
    pub autocorr: Vec<Vec<EwAutoCorr>>,
    #[serde(default)]
    pub metrics: Vec<Vec<SlotMetrics>>,
    #[serde(default)]
    pub conformal: Vec<Vec<Conformal>>,
    /// The output row of the last row learned from (docs/PLAN.md task 34).
    /// `None` before the first, and in files written before it existed.
    #[serde(default)]
    pub last_row: Option<LastRow>,
    /// What the stream was fed (docs/PLAN.md task 35). `None` in files
    /// written before it existed, and then for the life of the stream: a
    /// count that began partway would read as a count.
    #[serde(default)]
    pub summary: Option<DataSummary>,
    /// Per model instance, the decay time it has seen: every `d_clock` the
    /// models decayed by, a capped gap at the cap, which `settled_frac` is
    /// a fraction of (docs/WARMUP-AND-CONVERGENCE.md §2). Zero in a file
    /// written before it existed.
    #[serde(default)]
    pub decay_time: Vec<f64>,
    /// Per model instance, the clock the rows held under `embargo` have
    /// covered and the model has not decayed by yet, which `settled_frac`
    /// adds (`Instance::pending_clock`). Kept, not rebuilt at each chunk:
    /// a running `+=`/`-=` and a fresh sum of the held rows round
    /// differently, so rebuilding it made `settled_frac` depend on where a
    /// chunk ended (hard rule 3; found by a property test, 2026-09-24).
    /// Kept only by a stream with an `embargo`, one per model instance, and
    /// empty without one, so a spec without one writes what it always did.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_clock: Vec<f64>,
    /// Per model instance, which readiness notices it has raised (§3), so a
    /// resumed stream does not raise them again.
    #[serde(default)]
    pub notified: Vec<Notified>,
    /// Rows accepted but not yet learned from (`embargo`, E47). Skipped
    /// when empty, so a spec without a delay writes the same bytes it always
    /// did.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending: Vec<PendingRow>,
    /// The break skipped rows raised since the last accepted row, waiting
    /// for the next one (docs/PLAN.md task 153). Skipped when there is none.
    #[serde(default, skip_serializing_if = "HeldBreak::is_none")]
    pub held_break: HeldBreak,
    /// The clock of the newest row the models have learned from, and the
    /// rows fed to this group: what `learned_clock` and a clockless
    /// `scored_clock` show (docs/PLAN.md task 152).
    #[serde(default)]
    pub last_learned: Option<ClockValue>,
    #[serde(default)]
    pub fed: u64,
    /// Per model instance, the prediction each row in `pending` was *scored*
    /// with, in the same order (review 2026-09-12, C21): the one thing a
    /// replay cannot recompute, and what the residual diagnostics fold when
    /// the row's label matures. With a conformal interval, each record goes
    /// on with the radius every slot's interval was shown with (schema 18),
    /// which the release scores the row against. Skipped when there is none, so a spec
    /// without a delay writes the same bytes it always did.
    #[serde(default, skip_serializing_if = "no_score_preds")]
    pub score_pred: Vec<std::collections::VecDeque<Vec<f64>>>,
    /// Per model instance, the ring a windowed spread is cut from (review
    /// 2026-09-12, S1): `None` for an instance with no window, or whose spec
    /// reads no spread. Skipped when every entry is `None`, so no file
    /// without one moves.
    #[serde(default, skip_serializing_if = "no_resid_windows")]
    pub resid_win: Vec<Option<ResidWindow>>,
    /// The session value of the span this stream is in, under `group_close =
    /// "session"` (docs/ENHANCEMENTS.md E54): the closed row reports it, and
    /// the clock keeps only a hash. Written only by a spec that closes on
    /// session, so no other spec's bytes move.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_session: Option<String>,
    /// Where the `coef` cadence stands (docs/PLAN.md task 178): the clock
    /// since the last `coef` row and the rows since, which a resumed stream
    /// goes on counting. Written only once a cadence has moved it, so a
    /// spec without one writes the bytes it did.
    #[serde(default, skip_serializing_if = "CoefCadence::is_unset")]
    pub coef_cadence: CoefCadence,
}

/// [`Persisted::score_pred`] holds nothing worth writing.
fn no_score_preds(v: &[std::collections::VecDeque<Vec<f64>>]) -> bool {
    v.iter().all(std::collections::VecDeque::is_empty)
}

/// Which readiness notices one model instance has raised
/// (docs/WARMUP-AND-CONVERGENCE.md §3), each once per stream, and the
/// messages raised since the bank last drained them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Notified {
    /// A coefficient more ridge than data was named.
    pub support: bool,
    /// A readiness floor was found unreachable at steady state: the noise
    /// gate, or a target's `min_weight` (docs/PLAN.md task 198). One flag for
    /// both: the floor is checked first, so while it withholds a row the
    /// noise gate's reason is never the row's, and a stream says the one
    /// that holds it back.
    pub unreachable: bool,
    /// Raised and not yet drained; not state.
    #[serde(skip)]
    pub pending: Vec<String>,
}

/// How far the decay window has filled toward steady state after `t`
/// clock units of decay: `1 − 2^(−t/h)`, or `1 − λ^t`; NaN where nothing
/// decays, there being no steady state to settle toward
/// (docs/WARMUP-AND-CONVERGENCE.md §1). Rate-independent by construction:
/// it reads the clock the decay has covered, not a count of rows.
pub fn settled_frac(decay: Decay, t: f64) -> f64 {
    match decay {
        Decay::Halflife(h) if h.is_finite() => 1.0 - (-(t / h)).exp2(),
        Decay::Lam(l) if l < 1.0 => 1.0 - l.powf(t),
        _ => f64::NAN,
    }
}

/// The weight a stream settles at, read from where it stands: `weight`, the
/// accumulated weight before the next row (`weight_sum`'s meaning), less what
/// is left of the first row's, over the settled fraction:
///
/// ```text
/// weight_sum_settled = (W − w₁·(1 − s)) / s,    s = settled_frac
/// ```
///
/// On a regular stream -- rows `d` apart, each of weight `w` -- `W` after
/// `n` rows is `w·(1 − λⁿ)/(1 − λ)` and `s` is `1 − λⁿ⁻¹` (the first row
/// brings no clock), so this is `w/(1 − λ)`, the ceiling `1/(1 − λ^d)` in
/// rows (docs/WARMUP-AND-CONVERGENCE.md §5.1), exactly, from the second
/// row; on the first `s` is 0 and there is none (NaN). `w₁` is the first
/// row's weight, which on such a stream is every row's: the stream keeps no
/// first weight, so a caller hands the weight a row has there (docs/PLAN.md
/// task 198, D8).
pub fn settled_weight(weight: f64, first: f64, settled: f64) -> f64 {
    if settled > 0.0 && settled.is_finite() {
        (weight - first * (1.0 - settled)) / settled
    } else {
        f64::NAN
    }
}

/// The reasons a row's predictions are withheld, as `withheld_reason`
/// spells them: the code the row buffer carries (0 for none) and the name
/// the output shows. The order is the precedence when several apply.
pub const WITHHELD_REASONS: [&str; 3] = [
    "below_min_settled_frac",
    "below_min_weight",
    "above_max_error_inflation",
];
const REASON_SETTLED: u8 = 1;
const REASON_MIN_PERIODS: u8 = 2;
const REASON_INFLATION: u8 = 3;

/// Where a stream stands on the readiness statistics
/// (docs/WARMUP-AND-CONVERGENCE.md §3), as `Bank::summary` reports it: NaN
/// where a statistic does not exist for the model.
#[derive(Debug, Clone, PartialEq)]
pub struct Readiness {
    pub settled_frac: f64,
    /// The weight the stream settles at, read from where it stands
    /// ([`settled_weight`]); NaN where it has not one: no decay, a window,
    /// or a single row.
    pub weight_sum_settled: f64,
    /// The largest over the instance's slots.
    pub error_inflation: f64,
    pub min_support_coef: f64,
    pub min_support_coef_feature: Option<String>,
    pub n_coef: u64,
}

/// Where the noise gate's notice reads the stream: the gate's largest ratio,
/// its limit, the fraction the learned rows have settled -- the rows held
/// under an embargo left out, as the weight has them (review round 5, C1)
/// -- the weight before the row and the row's own (docs/PLAN.md task 198,
/// D8).
struct Unmet {
    worst: f64,
    max: f64,
    settled: f64,
    weight: f64,
    row_weight: f64,
}

impl Unmet {
    /// Whether the gate stays shut at steady state, which the notice says:
    /// the ratio's projection there -- `1 + (worst² − 1)·s/(2 − s)`, Kish's
    /// size growing from `s/(2 − s)` of its ceiling to all of it -- at or
    /// above the limit, or, where the model has not solved, its weight's
    /// ceiling below the floor it needs. The notice said so of the ratio
    /// read at 95% settled, which is still falling there: a gate a little
    /// above the limit then opened later, after a notice that it never
    /// would (docs/PLAN.md task 198, found by its half-life figure's test).
    fn for_good(&self, spec: &Spec) -> bool {
        if self.worst.is_finite() {
            let s = self.settled;
            1.0 + (self.worst * self.worst - 1.0) * s / (2.0 - s) >= self.max * self.max
        } else {
            settled_weight(self.weight, self.row_weight, self.settled) < solve_floor(spec)
        }
    }
}

/// The weight a model with a noise gate needs before it solves at all: a
/// row per coefficient, or the spec's own `min_weight` (`build_bare`). Below
/// it the ratio is infinite.
fn solve_floor(spec: &Spec) -> f64 {
    match spec.min_weight {
        Some(_) => spec.min_periods_or_default(),
        None => (spec.k() + usize::from(spec.fit_intercept)) as f64,
    }
}

/// What the noise gate's notice asks of the decay: the half-life above which
/// the gate opens at steady state on a regular stream (docs/PLAN.md task
/// 198, D8; WARMUP-AND-CONVERGENCE §7).
///
/// The gate reads `sqrt(1 + edf/n)`, `n` Kish's effective sample size, and
/// opens below `max`. Where it reads `worst`, finite, `n` must grow by
/// `r = (worst² − 1)/(max² − 1)`. At steady state `n` is `(1 + λ)/(1 − λ)`,
/// in proportion to the half-life where it is many rows, and at a settled
/// fraction `s` it is `s/(2 − s)` of that -- `(1 − λᵐ)/(1 + λᵐ)` of the
/// ceiling with `λᵐ = 1 − s` -- so the half-life must pass `h·r·s/(2 − s)`.
///
/// Where the ratio is infinite the model never solved: its weight tops out
/// at `C = w/(1 − 2^(−d/h))` ([`settled_weight`]) below its floor `N`
/// ([`solve_floor`]). From `C` the spacing is `d/h = −log2(1 − w/C)`, and
/// the half-life must pass both `d/−log2(1 − w/N)`, for the weight to reach
/// `N`, and `d/−log2((n − 1)/(n + 1))` with `n = k/(max² − 1)`, for Kish's
/// size to carry `k` degrees of freedom, the most a ridge leaves.
///
/// "Above", since the gate withholds at equality. In the spec's own terms: a
/// duration on a temporal clock, `lam` where the spec gives one.
fn half_life_figure(spec: &Spec, decay: Decay, at: &Unmet) -> String {
    let h = match decay {
        Decay::Halflife(h) => h,
        Decay::Lam(l) => -1.0 / l.log2(),
    };
    let (max, s, w) = (at.max, at.settled, at.row_weight);
    let need = if at.worst.is_finite() {
        h * (at.worst * at.worst - 1.0) / (max * max - 1.0) * s / (2.0 - s)
    } else {
        let d = h * -(1.0 - w / settled_weight(at.weight, w, s)).log2();
        let k = (spec.k() + usize::from(spec.fit_intercept)) as f64;
        let n = k / (max * max - 1.0);
        let floor = d / -(1.0 - w / solve_floor(spec)).log2();
        let kish = d / -((n - 1.0) / (n + 1.0)).log2();
        // A bound that cannot bind -- a floor of one row, a Kish size of
        // one -- is NaN, and `max` takes the other.
        floor.max(kish)
    };
    if !(need.is_finite() && need > 0.0) {
        return "raise the half_life".into();
    }
    match decay {
        Decay::Lam(_) => format!("raise lam above {:.6}", (-1.0 / need).exp2()),
        Decay::Halflife(_) => format!(
            "raise the half_life above {}",
            crate::bank::clock_amount(spec, need)
        ),
    }
}

/// [`Persisted::resid_win`] holds no ring.
fn no_resid_windows(v: &[Option<ResidWindow>]) -> bool {
    v.iter().all(Option::is_none)
}

/// Live per-stream state: the models, what a save keeps beside them
/// ([`Persisted`]), and what the spec configures.
pub struct Stream {
    pub models: Vec<(String, AnyModel)>,
    /// Everything else a save keeps, as it keeps it.
    pub persisted: Persisted,
    /// Decay of each model instance, needed to age `resid_var` on the same
    /// clock the model itself uses.
    decays: Vec<Decay>,
    /// Warmup threshold per target (ENHANCEMENTS E7).
    min_weight: Vec<f64>,
    /// Row scratch, reused for the life of the stream so the chunk loop
    /// itself allocates nothing (docs/PERFORMANCE.md P1). The `pred` `Vec`
    /// inside each `Step` is the one per-row allocation left, and it is cheap
    /// (docs/IMPROVEMENTS.md P2).
    scratch: Vec<Scratch>,
    /// The time a row waits on the elapsed clock before the models learn
    /// from it (E47, docs/PLAN.md task 176); `None` for the ordinary "learn
    /// where it sits".
    embargo: Option<Embargo>,
    /// The spec's `gap_cap` and `session_gap` in integer nanoseconds, where
    /// they are durations, for the rows' stamps (docs/PLAN.md task 175):
    /// configuration from the spec, like `embargo`.
    exact_caps: ExactCaps,
}

impl Stream {
    /// The first of this stream's windows to pass a refusing budget -- a
    /// model instance's, or a spread's ring (S1): its ring's bytes and its
    /// cadence, `window_every` and `max_rows_between_snapshots` (review
    /// 2026-09-12, P4; docs/PLAN.md task 162).
    pub fn window_over_budget(&self) -> Option<(usize, online_core::Cadence)> {
        self.models
            .iter()
            .find_map(|(_, m)| m.window_over_budget())
            .or_else(|| {
                self.persisted
                    .resid_win
                    .iter()
                    .flatten()
                    .find_map(ResidWindow::over_budget)
            })
    }

    /// Summed over this stream's model instances (one per half-life).
    pub fn solve_failures(&self) -> u64 {
        self.models.iter().map(|(_, m)| m.solve_failures()).sum()
    }

    /// The readiness notices raised since the last drain, in instance
    /// order (docs/WARMUP-AND-CONVERGENCE.md §3); each is raised once per
    /// instance for the life of the stream.
    pub fn take_notices(&mut self) -> Vec<String> {
        self.persisted
            .notified
            .iter_mut()
            .flat_map(|n| std::mem::take(&mut n.pending))
            .collect()
    }

    /// Where this stream stands (§3), for `Bank::summary`: read from its
    /// first instance, as it stands after the last row.
    pub fn readiness(&self, spec: &Spec) -> Readiness {
        let decay = self.decays.first().copied();
        let settled = decay.map_or(f64::NAN, |d| {
            settled_frac(d, self.persisted.decay_time.first().copied().unwrap_or(0.0))
        });
        let model = self.models.first().map(|(_, m)| m);
        // The weight a row has: 1 without a weight column, and the rows'
        // mean weight with one, which on the regular stream the estimate is
        // exact for is every row's. A window's weight does not settle as the
        // decay does, so it has no estimate.
        let windowed = spec
            .model
            .window_parts()
            .is_some_and(|(window, _)| window.is_some());
        let row_weight = match (&spec.weight, self.summary()) {
            (None, _) => 1.0,
            (Some(_), Some(d)) if self.persisted.rows_seen > 0 => {
                d.weight_sum / self.persisted.rows_seen as f64
            }
            _ => f64::NAN,
        };
        let weight_sum_settled = match model {
            Some(m) if !windowed => settled_weight(m.n_eff(), row_weight, settled),
            _ => f64::NAN,
        };
        let mut infl = Vec::new();
        let error_inflation = match model {
            Some(m) if m.error_inflation_into(&mut infl) => {
                infl.iter().cloned().fold(f64::NAN, f64::max)
            }
            _ => f64::NAN,
        };
        let k_total = spec.k() + usize::from(spec.fit_intercept);
        let (mut min_support, mut feature) = (f64::NAN, None);
        if let Some(s) = model.and_then(|m| m.support_coef()) {
            let worst = s
                .iter()
                .flat_map(|slot| slot.iter().enumerate())
                .filter(|(_, v)| v.is_finite())
                .min_by(|a, b| a.1.total_cmp(b.1));
            if let Some((i, &share)) = worst {
                min_support = share;
                feature = spec
                    .features
                    .get(i % k_total - usize::from(spec.fit_intercept))
                    .cloned();
            }
        }
        Readiness {
            settled_frac: settled,
            weight_sum_settled,
            error_inflation,
            min_support_coef: min_support,
            min_support_coef_feature: feature,
            n_coef: k_total as u64,
        }
    }

    /// Model instances in this stream (one per half-life).
    pub fn n_models(&self) -> usize {
        self.models.len()
    }

    /// The warm-up threshold per target this stream gates its outputs on
    /// (ENHANCEMENTS E7): the spec's `min_weight`, or its kind's default.
    pub fn min_weight(&self) -> &[f64] {
        &self.min_weight
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

    /// The output row of the last row this stream learned from, if any
    /// (docs/PLAN.md task 34): what [`Stream::process_chunk`] wrote for it,
    /// kept across [`Stream::save`] and [`Stream::restore`].
    pub fn last_row(&self) -> Option<&LastRow> {
        self.persisted.last_row.as_ref()
    }

    /// What this stream has been fed (docs/PLAN.md task 35), accumulated
    /// by [`Stream::process_chunk`] and kept across [`Stream::save`] and
    /// [`Stream::restore`]. `None` for a stream restored from a file
    /// written before the summary existed.
    pub fn summary(&self) -> Option<&DataSummary> {
        self.persisted.summary.as_ref()
    }

    /// Keep the last learned row of a run's buffers as [`Self::last_row`]:
    /// the last row `processed` marks, so a chunk that ends in skipped rows
    /// keeps the row before them. A run with no learned row changes nothing.
    /// The vectors are reused, so after the first run this allocates only
    /// for a `coef` the row carried.
    pub fn remember_last(&mut self, out: &ChunkOut) {
        if let Some(ri) = out.processed.iter().rposition(|&p| p) {
            self.persisted
                .last_row
                .get_or_insert_with(Default::default)
                .take(out, ri);
        }
    }
}

/// The core's one definition of a value a model learns from, which the
/// stream reads its rows by: a feature or weight that is not usable skips
/// the row, a target that is not makes it predict-only (docs/PLAN.md §3,
/// task 183).
pub use online_core::{all_usable, usable};

/// Whether the stream accepts input row `i`: every feature and the weight
/// usable. The one test [`Stream::process_chunk`]'s schedule and
/// [`last_accepted`] share, so the row `coef` is promised on is a row the
/// schedule accepts.
#[inline]
fn accepts(features: &FeatureRows, weight: Option<&[f64]>, i: usize) -> bool {
    all_usable(features.row(i)) && weight.map(|w| usable(w[i])).unwrap_or(true)
}

/// The last of a group's input rows `base..base + n` in the chunk that the
/// stream accepts: where `coef` is written without a cadence (review round
/// 4, PB2). `None` when every one is skipped, and the chunk writes no
/// `coef` for the group.
pub fn last_accepted(
    features: &FeatureRows,
    weight: Option<&[f64]>,
    base: usize,
    n: usize,
) -> Option<usize> {
    (base..base + n)
        .rev()
        .find(|&i| accepts(features, weight, i))
}

/// True when this target has not reached its own warmup threshold yet.
#[inline]
fn step_n_eff_below(n_eff: f64, min_weight: &[f64], target: usize) -> bool {
    min_weight.get(target).is_some_and(|t| n_eff < *t)
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
    pub zscore: Vec<f64>,
    pub autocorr: Vec<f64>,
    /// `(ic, r2, hit_rate)`, model-major: `n_models * 3 * n_slots * n_rows`.
    pub metrics: Vec<f64>,
    /// `(pred_lo, pred_hi, coverage)`, laid out like `metrics`.
    pub conformal: Vec<f64>,
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
    /// `n_models * n_rows`: how settled the instance was before the row
    /// (docs/WARMUP-AND-CONVERGENCE.md §3); NaN = null, where nothing decays.
    pub settled: Vec<f64>,
    /// `n_models * n_rows`: why the row's predictions were withheld, as an
    /// index into [`WITHHELD_REASONS`] plus one; 0 = none (null).
    pub reason: Vec<u8>,
    /// `n_models * n_slots * n_rows` under `emit_error_inflation`, else
    /// empty: the row's own error inflation per slot.
    pub inflation: Vec<f64>,
    /// Each coefficient's data share, on `coef`'s cadence: `[model][row]`.
    pub support_coef: Vec<Vec<Option<Vec<f64>>>>,
    /// `n_rows` each under `emit_clocks`, else empty: the row's own clock,
    /// and the newest learned row's when it was scored (task 152).
    pub scored_clock: Vec<Option<ClockValue>>,
    pub learned_clock: Vec<Option<ClockValue>>,
    /// Slot counts this layout was built for.
    pub n_models: usize,
    pub n_slots: usize,
    pub n_levels: usize,
}

/// Which optional buffers a spec fills: read by `ChunkOut::new` to size them
/// and by `ChunkOut::run_rows` to count the values a row writes.
struct Buffers {
    residuals: bool,
    extras: bool,
    n_levels: usize,
    is_lasso: bool,
}

impl Buffers {
    fn of(spec: &Spec) -> Self {
        Self {
            // A model with no target to predict has no residual field, so its
            // `resid` buffer would be filled, written once per slot per row
            // and never read: for a 230-statistic `ew_cov` that was a fifth
            // of the row (docs/PERFORMANCE.md §13).
            residuals: !spec.model.predicts_no_target(),
            // `sigma` is also the loss that `emit_selected` and
            // `emit_averaged` rank slots by (E13/E14 reuse E12's tracked
            // error), so it has to be materialized for them even when it is
            // not itself an output field.
            extras: spec.emit_sigma || spec.emit_zscore || spec.emit_selected || spec.emit_averaged,
            n_levels: spec.resid_quantiles.as_ref().map_or(0, Vec::len),
            is_lasso: matches!(spec.model, crate::ModelKind::Lasso { .. }),
        }
    }

    /// `f64` values one row writes across every buffer, for `n_models`
    /// instances of `n_slots` slots.
    fn per_row(&self, spec: &Spec, n_models: usize, n_slots: usize) -> usize {
        let per = n_models * n_slots;
        let on = |flag: bool| if flag { per } else { 0 };
        per + on(self.residuals)
            + 2 * on(self.extras)
            + on(spec.emit_autocorr)
            + 3 * on(spec.emit_metrics)
            + 3 * on(spec.conformal.is_some())
            + self.n_levels * per
            + n_models
            + n_models
            + on(spec.emit_error_inflation)
            + if self.is_lasso {
                n_models * spec.m()
            } else {
                0
            }
    }
}

impl ChunkOut {
    /// Buffers for `n_rows` rows of one stream, all null until written.
    pub fn new(spec: &Spec, n_models: usize, n_slots: usize, n_rows: usize) -> Self {
        let per = n_models * n_slots * n_rows;
        let on = |flag: bool| if flag { per } else { 0 };
        let Buffers {
            residuals,
            extras,
            n_levels,
            is_lasso,
        } = Buffers::of(spec);
        Self {
            rows: Vec::with_capacity(n_rows),
            processed: vec![false; n_rows],
            pred: vec![f64::NAN; per],
            resid: vec![f64::NAN; on(residuals)],
            sigma: vec![f64::NAN; on(extras)],
            zscore: vec![f64::NAN; on(extras)],
            autocorr: vec![f64::NAN; on(spec.emit_autocorr)],
            metrics: vec![f64::NAN; 3 * on(spec.emit_metrics)],
            conformal: vec![f64::NAN; 3 * on(spec.conformal.is_some())],
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
            settled: vec![f64::NAN; n_models * n_rows],
            reason: vec![0; n_models * n_rows],
            inflation: vec![f64::NAN; on(spec.emit_error_inflation)],
            support_coef: vec![vec![None; n_rows]; n_models],
            scored_clock: vec![None; if spec.emit_clocks { n_rows } else { 0 }],
            learned_clock: vec![None; if spec.emit_clocks { n_rows } else { 0 }],
            n_models,
            n_slots,
            n_levels,
        }
    }

    /// How many rows one learning run covers (docs/PERFORMANCE.md §13).
    ///
    /// The buffers are slot-major, so a row is one store per slot, each a
    /// slot's stride apart. Two things about that stride decide how fast a
    /// wide model runs, and both are set here rather than by whoever chose
    /// the chunk size:
    ///
    /// - The run's buffers stay within `BUDGET` bytes, so the lines a wide
    ///   model's row touches are still in cache when the next rows complete
    ///   them; over a 400k-row chunk, a 230-statistic `ew_cov` spent a third
    ///   of its time on stores to lines that had already been evicted.
    /// - A slot's stride is an odd number of 128-byte lines, so no two slots
    ///   of a row share a cache set. A power-of-two chunk (65 536 rows, the
    ///   natural streaming batch) put every slot of that `ew_cov` in the
    ///   same set and ran it 2-3x slower than a chunk 16 rows longer.
    ///
    /// A narrow model writes a few values per row and gets runs of a hundred
    /// thousand rows: a chunk is one or two, and nothing changes for it. A
    /// model too wide for the budget gets the floor, five lines. Chunk
    /// invariance is what makes the split invisible.
    pub fn run_rows(spec: &Spec, n_models: usize, n_slots: usize) -> usize {
        const BUDGET: usize = 2 << 20;
        /// `f64`s in a 128-byte line, the longest of the two OSes' lines.
        const LINE: usize = 16;
        let width = Buffers::of(spec).per_row(spec, n_models, n_slots).max(1);
        let lines = BUDGET / (8 * width * LINE);
        let lines = if lines.is_multiple_of(2) {
            lines.saturating_sub(1)
        } else {
            lines
        };
        lines.max(5) * LINE
    }

    /// Offset of `(model, slot)` at row `ri`, in any `n_models * n_slots *
    /// n_rows` buffer. The single place the layout is spelled out: each
    /// instance's block is written by `run_instance` at `slot * n_rows + ri`,
    /// and `assemble` in `bank.rs` reads the whole buffer through this.
    #[inline]
    pub fn at(n_slots: usize, n_rows: usize, mi: usize, slot: usize, ri: usize) -> usize {
        (mi * n_slots + slot) * n_rows + ri
    }

    /// The first row when this task's rows are one unbroken run `base..base +
    /// n`: a single group, or a group that arrives in blocks. `rows` is
    /// strictly increasing by construction (`group_indices` walks the chunk
    /// in order and each row belongs to one group), so the two ends decide.
    pub fn contiguous(&self) -> Option<usize> {
        let (&first, &last) = (self.rows.first()?, self.rows.last()?);
        debug_assert!(
            self.rows.windows(2).all(|w| w[0] < w[1]),
            "rows out of order"
        );
        (last - first + 1 == self.rows.len()).then_some(first)
    }
}

/// One row of a [`ChunkOut`], as a value: the output-struct fields of the
/// last row a stream learned from, kept with its state so that a bank loaded
/// from a file reports the diagnostics as they stood -- `pred`, `resid`,
/// `sigma`, the metrics, the residual quantiles, `n_eff`, `coef`, ... --
/// without the output frame (docs/PLAN.md task 34). Every field is the
/// chunk's buffer with `n_rows = 1`, so the row goes out through
/// [`LastRow::take`] and back in through [`LastRow::to_chunk`] by the one
/// layout `assemble` reads, and a value it holds is the value the output
/// row had, to the bit.
///
/// Every float vector here is annotated for the JSON export
/// ([`online_core::humanfloat`]): these are *diagnostics*, and `NaN` is an
/// ordinary value among them -- an unsupervised model has no `pred` or
/// `resid`, and `sigma` is `NaN` until the warm-up ends -- so this is the
/// struct that loses the most to a naive encoding. The msgpack the state
/// file uses is unchanged.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LastRow {
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub pred: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub resid: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub sigma: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub zscore: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub autocorr: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub metrics: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub conformal: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub resid_q: Vec<f64>,
    pub drift: Vec<bool>,
    /// One each under `emit_clocks`, else empty (task 152).
    #[serde(default)]
    pub scored_clock: Vec<Option<ClockValue>>,
    #[serde(default)]
    pub learned_clock: Vec<Option<ClockValue>>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub n_eff: Vec<f64>,
    #[serde(with = "online_core::humanfloat::vec_f64_or_tag")]
    pub lam_selected: Vec<f64>,
    /// A slot no solve has fit is NaN (review round 4, CC1), as is the mean
    /// of an `ew_class` class no row has carried, so the export tags them.
    #[serde(with = "online_core::humanfloat::vec_opt_vec_f64_or_tag")]
    pub coef: Vec<Option<Vec<f64>>>,
    #[serde(default, with = "online_core::humanfloat::vec_f64_or_tag")]
    pub settled: Vec<f64>,
    #[serde(default)]
    pub reason: Vec<u8>,
    #[serde(default, with = "online_core::humanfloat::vec_f64_or_tag")]
    pub inflation: Vec<f64>,
    /// The intercept's share is NaN by definition, so the export tags it.
    #[serde(default, with = "online_core::humanfloat::vec_opt_vec_f64_or_tag")]
    pub support_coef: Vec<Option<Vec<f64>>>,
}

/// Row `ri` of a buffer of `n_rows` rows: in every `ChunkOut` buffer the row
/// is the innermost index, whatever the outer product (`ChunkOut::at`, and
/// the model-major blocks of `metrics`, `conformal` and `resid_q`), so the
/// row's values are the elements `g * n_rows + ri`.
fn take_row<T: Copy>(dst: &mut Vec<T>, src: &[T], n_rows: usize, ri: usize) {
    dst.clear();
    dst.extend(src.iter().skip(ri).step_by(n_rows.max(1)));
}

impl LastRow {
    /// Overwrite with row `ri` of `out`, reusing the vectors.
    pub fn take(&mut self, out: &ChunkOut, ri: usize) {
        let n = out.processed.len();
        debug_assert!(ri < n && out.processed[ri]);
        take_row(&mut self.pred, &out.pred, n, ri);
        take_row(&mut self.resid, &out.resid, n, ri);
        take_row(&mut self.sigma, &out.sigma, n, ri);
        take_row(&mut self.zscore, &out.zscore, n, ri);
        take_row(&mut self.autocorr, &out.autocorr, n, ri);
        take_row(&mut self.metrics, &out.metrics, n, ri);
        take_row(&mut self.conformal, &out.conformal, n, ri);
        take_row(&mut self.resid_q, &out.resid_q, n, ri);
        take_row(&mut self.drift, &out.drift, n, ri);
        take_row(&mut self.n_eff, &out.n_eff, n, ri);
        take_row(&mut self.scored_clock, &out.scored_clock, n, ri);
        take_row(&mut self.learned_clock, &out.learned_clock, n, ri);
        take_row(&mut self.lam_selected, &out.lam_selected, n, ri);
        self.coef.clear();
        self.coef.extend(out.coef.iter().map(|c| c[ri].clone()));
        take_row(&mut self.settled, &out.settled, n, ri);
        take_row(&mut self.reason, &out.reason, n, ri);
        take_row(&mut self.inflation, &out.inflation, n, ri);
        self.support_coef.clear();
        self.support_coef
            .extend(out.support_coef.iter().map(|c| c[ri].clone()));
    }

    /// A one-row chunk at absolute row `row`, marked processed, for
    /// `assemble`. `Err` when the row is not the shape `spec` writes with
    /// `n_models` instances of `n_slots` slots -- a state file that does not
    /// belong to its spec.
    pub fn to_chunk(
        &self,
        spec: &Spec,
        n_models: usize,
        n_slots: usize,
        row: usize,
    ) -> Result<ChunkOut, String> {
        let mut out = ChunkOut::new(spec, n_models, n_slots, 1);
        let same = [
            out.pred.len() == self.pred.len(),
            out.resid.len() == self.resid.len(),
            out.sigma.len() == self.sigma.len(),
            out.zscore.len() == self.zscore.len(),
            out.autocorr.len() == self.autocorr.len(),
            out.metrics.len() == self.metrics.len(),
            out.conformal.len() == self.conformal.len(),
            out.resid_q.len() == self.resid_q.len(),
            out.drift.len() == self.drift.len(),
            out.n_eff.len() == self.n_eff.len(),
            out.lam_selected.len() == self.lam_selected.len(),
            out.coef.len() == self.coef.len(),
            out.settled.len() == self.settled.len(),
            out.reason.len() == self.reason.len(),
            out.inflation.len() == self.inflation.len(),
            out.support_coef.len() == self.support_coef.len(),
            out.scored_clock.len() == self.scored_clock.len(),
            out.learned_clock.len() == self.learned_clock.len(),
        ];
        if same.contains(&false) {
            return Err(format!(
                "saved last row of spec {:?} has the wrong shape for its spec",
                spec.name
            ));
        }
        out.rows.push(row);
        out.processed[0] = true;
        out.pred.clone_from(&self.pred);
        out.resid.clone_from(&self.resid);
        out.sigma.clone_from(&self.sigma);
        out.zscore.clone_from(&self.zscore);
        out.autocorr.clone_from(&self.autocorr);
        out.metrics.clone_from(&self.metrics);
        out.conformal.clone_from(&self.conformal);
        out.resid_q.clone_from(&self.resid_q);
        out.drift.clone_from(&self.drift);
        out.n_eff.clone_from(&self.n_eff);
        out.lam_selected.clone_from(&self.lam_selected);
        out.scored_clock.clone_from(&self.scored_clock);
        out.learned_clock.clone_from(&self.learned_clock);
        for (dst, src) in out.coef.iter_mut().zip(&self.coef) {
            dst[0].clone_from(src);
        }
        out.settled.clone_from(&self.settled);
        out.reason.clone_from(&self.reason);
        out.inflation.clone_from(&self.inflation);
        for (dst, src) in out.support_coef.iter_mut().zip(&self.support_coef) {
            dst[0].clone_from(src);
        }
        Ok(out)
    }
}

impl Stream {
    pub fn new(spec: &Spec) -> Result<Self, String> {
        let models = build_models(spec)?;
        let decays: Vec<Decay> = spec.decays()?.into_iter().map(|(_, d)| d).collect();
        let slots: Vec<usize> = models.iter().map(|(_, m)| m.n_outputs()).collect();
        let drift = if spec.emit_drift {
            // 20 only without a clock, where a row is one unit: the spec
            // requires a threshold with a clock (task 168).
            let d = PageHinkley::new(
                spec.drift_delta_or_default(),
                spec.drift_threshold_or_default(),
            );
            slots.iter().map(|&n| vec![d.clone(); n]).collect()
        } else {
            Vec::new()
        };
        let resid_win = models
            .iter()
            .map(|_| resid_window(spec))
            .collect::<Result<Vec<_>, String>>()?;
        let instances = slots.len();
        let embargo = spec.embargo.as_ref().map(Embargo::of);
        Ok(Self {
            persisted: Persisted {
                clock: ClockState::new(),
                rows_seen: 0,
                resid_var: slots.iter().map(|&n| vec![0.0; n]).collect(),
                resid_w: slots.iter().map(|&n| vec![0.0; n]).collect(),
                resid_win,
                drift,
                resid_q: match &spec.resid_quantiles {
                    Some(levels) => {
                        let proto = EwQuantile::new(levels)?;
                        slots.iter().map(|&n| vec![proto.clone(); n]).collect()
                    }
                    None => Vec::new(),
                },
                autocorr: if spec.emit_autocorr {
                    let proto = EwAutoCorr::new(spec.resid_autocorr_lag_or_default())?;
                    slots.iter().map(|&n| vec![proto.clone(); n]).collect()
                } else {
                    Vec::new()
                },
                metrics: if spec.emit_metrics {
                    slots.iter().map(|&n| vec![SlotMetrics::new(); n]).collect()
                } else {
                    Vec::new()
                },
                conformal: match spec.conformal {
                    Some(level) => {
                        let proto = Conformal::new(level, spec.conformal_rate_or_default())?;
                        slots.iter().map(|&n| vec![proto.clone(); n]).collect()
                    }
                    None => Vec::new(),
                },
                last_row: None,
                summary: Some(DataSummary::new(spec)),
                decay_time: vec![0.0; instances],
                // Only a delay ever holds rows: without one there is no
                // clock to keep, and none is written.
                pending_clock: if embargo.is_some() {
                    vec![0.0; instances]
                } else {
                    Vec::new()
                },
                notified: vec![Notified::default(); instances],
                pending: Vec::new(),
                held_break: HeldBreak::default(),
                last_learned: None,
                fed: 0,
                score_pred: (0..instances)
                    .map(|_| std::collections::VecDeque::new())
                    .collect(),
                last_session: None,
                coef_cadence: CoefCadence::default(),
            },
            min_weight: spec.min_periods_per_target(),
            models,
            decays,
            scratch: slots.iter().map(|_| Scratch::default()).collect(),
            embargo,
            exact_caps: spec.exact_caps(),
        })
    }

    /// The stream as a state file keeps it: each model's state, and the rest
    /// as it stands ([`Persisted`]).
    pub fn save(&self) -> StreamState {
        StreamState {
            models: self.models.iter().map(|(_, m)| m.state()).collect(),
            persisted: self.persisted.clone(),
        }
    }

    /// [`Self::save`]'s inverse, for `spec`: everything the state holds is
    /// taken, once it is checked to be this spec's -- a field the spec has
    /// since gained or lost the option for starts afresh, and one shaped for
    /// another spec, which would index past this one's slots, is refused.
    pub fn restore(spec: &Spec, saved: &StreamState) -> Result<Self, String> {
        let mut stream = Stream::new(spec)?;
        if stream.models.len() != saved.models.len() {
            return Err("saved state has a different number of model instances".into());
        }
        let models = stream
            .models
            .drain(..)
            .zip(&saved.models)
            .map(|((suffix, fresh), st)| {
                let mut m = AnyModel::restore(st).map_err(|e| e.to_string())?;
                // The cadence is the spec's, as the budget is: a state saved
                // before the weight-share rule takes it on load (task 115 (b)).
                m.set_solve_share(fresh.solve_share());
                // The state's cfg is the model's, but the stream feeds it the
                // spec's columns and reads the spec's slots: a state whose
                // cfg is another width would index past both (review
                // 2026-09-18, B3).
                if m.n_features() != fresh.n_features()
                    || m.n_targets() != fresh.n_targets()
                    || m.n_outputs() != fresh.n_outputs()
                {
                    return Err(format!(
                        "saved state's model {suffix:?} is not this spec's shape"
                    ));
                }
                Ok((suffix, m))
            })
            .collect::<Result<Vec<_>, String>>()?;
        stream.models = models;
        // The window's budget is configuration, which the state does not
        // carry (`build_one`).
        let budget = spec.model.window_budget();
        for (_, m) in stream.models.iter_mut() {
            m.set_window_budget(budget);
        }
        let fresh = &stream.persisted;
        let mut p = saved.persisted.clone();
        // The decay time and the notices, per instance; a file written
        // before either existed leaves the fresh zeros, so for such a file
        // `settled_frac` counts from the load.
        if p.decay_time.len() != fresh.decay_time.len() {
            p.decay_time = fresh.decay_time.clone();
        }
        if p.notified.len() != fresh.notified.len() {
            p.notified = fresh.notified.clone();
        }
        // A saved per-instance, per-slot diagnostic is taken when it is
        // shaped as this spec's; one absent, or sized for another spec
        // (written before the field existed, or under a spec that has since
        // gained or lost the option), starts over. One with this spec's
        // number of instances but another width is corrupt and is refused:
        // it reached `build_instances`'s "one per instance" and the slot
        // loops as a panic (review 2026-09-18, B3).
        fn take_diag<T: Clone>(
            name: &str,
            kept: &mut Vec<Vec<T>>,
            fresh: &[Vec<T>],
            same: impl Fn(&T, &T) -> bool,
        ) -> Result<(), String> {
            if kept.len() != fresh.len() {
                *kept = fresh.to_vec();
                return Ok(());
            }
            let fits = kept
                .iter()
                .zip(fresh)
                .all(|(s, l)| s.len() == l.len() && s.iter().zip(l).all(|(a, b)| same(a, b)));
            if !fits {
                return Err(format!("saved state's {name} do not fit this spec"));
            }
            Ok(())
        }
        // The variance and its weight are one estimate: both or neither.
        let n = fresh.resid_var.len();
        if p.resid_var.len() == n || p.resid_w.len() == n {
            let fits = p.resid_var.len() == n
                && p.resid_w.len() == n
                && p.resid_var
                    .iter()
                    .zip(&fresh.resid_var)
                    .chain(p.resid_w.iter().zip(&fresh.resid_w))
                    .all(|(s, l)| s.len() == l.len());
            if !fits {
                return Err("saved state's residual variances do not fit this spec".into());
            }
        } else {
            p.resid_var = fresh.resid_var.clone();
            p.resid_w = fresh.resid_w.clone();
        }
        // A spread's ring, where the spec still keeps one. A file written
        // before it existed (S1) leaves the fresh one, so for one window the
        // spread counts only the rows after the load.
        let mut rings = fresh.resid_win.clone();
        for ((live, saved), slots) in rings
            .iter_mut()
            .zip(&p.resid_win)
            .zip(fresh.resid_var.iter().map(Vec::len))
        {
            if let (Some(live), Some(saved)) = (live.as_mut(), saved) {
                if !saved.fits(slots) {
                    return Err("saved state's residual window does not fit this spec".into());
                }
                *live = saved.clone();
                live.set_budget(budget);
            }
        }
        p.resid_win = rings;
        take_diag("drift detectors", &mut p.drift, &fresh.drift, |_, _| true)?;
        // A sketch is held to its own shape too, as `ew_cov`'s is: the next
        // residual indexes its buckets by its pointers (review 2026-10-06,
        // CB2's class).
        take_diag(
            "residual quantiles",
            &mut p.resid_q,
            &fresh.resid_q,
            |s, l| s.levels() == l.levels() && s.has_shape(),
        )?;
        take_diag(
            "autocorrelations",
            &mut p.autocorr,
            &fresh.autocorr,
            |s, l| s.same_shape(l),
        )?;
        take_diag("metrics", &mut p.metrics, &fresh.metrics, |_, _| true)?;
        take_diag(
            "conformal intervals",
            &mut p.conformal,
            &fresh.conformal,
            |_, _| true,
        )?;
        // A file written by a spec with a `embargo` carries the rows it
        // had not learned from yet; one written without a delay has none, and
        // one restored under a spec that has since gained or lost a delay
        // gets whatever the file holds, which is what "resume this stream"
        // means (E47).
        // Each waiting row is replayed into `step` with the spec's columns,
        // so it must carry exactly them (review 2026-09-18, B3), and its
        // place on the elapsed clock, which its release is measured from
        // (task 176): every row a stream holds has one.
        let (nf, nt) = (spec.features.len(), spec.targets.len());
        if p.pending
            .iter()
            .any(|r| r.xs.len() != nf || r.ys.len() != nt || r.arrived.is_none())
        {
            return Err("saved state's pending rows do not fit this spec".into());
        }
        // And values the bank would send a model: a held row is learned at
        // its release, where none of the bank's checks on a row run, so a
        // feature or target that is not usable, a weight that is not usable
        // or is below 0, or a step outside `[0, gap_cap]` -- where every
        // step `ClockState` hands a model is -- is refused here (review
        // 2026-10-06, PB4).
        let cap = spec.clock_cfg()?.gap_cap;
        for (i, r) in p.pending.iter().enumerate() {
            let what = if !all_usable(&r.xs) {
                "a feature that is not a usable number"
            } else if !r.ys.iter().flatten().all(|&y| usable(y)) {
                "a target that is not a usable number"
            } else if !(usable(r.w) && r.w >= 0.0) {
                "a weight that is not a usable number >= 0"
            } else if !(r.d_clock.is_finite() && r.d_clock >= 0.0 && r.d_clock <= cap) {
                "a step outside [0, gap_cap]"
            } else {
                continue;
            };
            return Err(format!("saved state's held row {i} is damaged: {what}"));
        }
        // The held rows' clock per instance, kept where a delay holds rows
        // and nowhere else. (A schema-14 file had none, and its loader
        // rebuilt it from the rows still held, until the minimum schema
        // passed it: docs/PLAN.md tasks 198 and 194-202.)
        let instances = fresh.decay_time.len();
        if p.pending_clock.len() != fresh.pending_clock.len() {
            return Err(format!(
                "saved state's held-rows clock has {} entries where this spec keeps {}",
                p.pending_clock.len(),
                fresh.pending_clock.len()
            ));
        }
        // The score-time predictions ride with the waiting rows (C21). A file
        // written before they were kept has none: each of its waiting rows
        // gets an empty record, at the front where those rows sit, so every
        // queue stays in step with `pending` and those rows fold nothing when
        // they mature -- there being no honest prediction to fold.
        let held = p.pending.len();
        p.score_pred = (0..instances)
            .map(|mi| {
                let saved_q = saved.persisted.score_pred.get(mi);
                let kept = saved_q.map_or(0, |q| q.len().min(held));
                let mut q: std::collections::VecDeque<Vec<f64>> =
                    std::iter::repeat_n(Vec::new(), held - kept).collect();
                if let Some(saved_q) = saved_q {
                    q.extend(saved_q.iter().skip(saved_q.len() - kept).cloned());
                }
                q
            })
            .collect();
        // Checked here, where a file that is not its spec's is refused with
        // the models, rather than at the first read.
        if let Some(last) = &p.last_row {
            last.to_chunk(spec, stream.n_models(), stream.n_slots(), 0)?;
        }
        // A file from before the summary existed leaves it `None` for good:
        // the alternative, counting from here on, would report a number
        // that looks like the whole history and is not.
        if let Some(summary) = &p.summary {
            summary.validate(spec, p.rows_seen)?;
        }
        stream.persisted = p;
        Ok(stream)
    }

    /// The first row of the chunk whose step from the row before it in this
    /// stream -- the stream's last row, for the first -- is not a finite
    /// number of clock units: two values of a number clock whose difference
    /// is past the largest double, `1e308 − (−1e308)`. Such a step left the
    /// decayed clock's removed time infinite for good, and every later stamp
    /// compared equal to any span (review 2026-10-06, PB7). A temporal
    /// clock's steps are integer nanoseconds and always finite. The bank runs
    /// it beside [`Self::check_clock`], before any stream is touched, so the
    /// refusal is the chunk's, as a clock value that is not a number is.
    pub fn check_steps(
        &self,
        clock: Option<&ClockCol>,
        rows: &[usize],
        base: usize,
    ) -> Result<(), StepRefusal> {
        let Some(ClockCol::F64(values)) = clock else {
            return Ok(());
        };
        let mut prev = match self.persisted.clock.last_clock() {
            Some(ClockValue::F64(p)) => Some(p),
            _ => None,
        };
        for (ri, &row) in rows.iter().enumerate() {
            let now = values[base + ri];
            if let Some(p) = prev
                && !(now - p).is_finite()
            {
                return Err(StepRefusal { row, prev: p, now });
            }
            prev = Some(now);
        }
        Ok(())
    }

    /// The first row of the chunk at which this stream's clock would step back
    /// in a way its policy refuses -- any step back under `"error"`, a late
    /// row under `"reset_state"` -- exactly as [`Stream::process_chunk`] would
    /// report it, found on a copy of the clock so nothing is touched.
    ///
    /// The bank runs this over every stream of a chunk before it runs any of
    /// them (docs/IMPROVEMENTS.md C3). Without it a backwards clock in one
    /// group refused that group's rows but left every other group updated, so
    /// the corrected chunk could not be re-fed and only a `load` recovered the
    /// bank.
    pub fn check_clock(
        &self,
        cfg: &online_core::ClockCfg,
        clock: Option<&ClockCol>,
        session: Option<&[u64]>,
        rows: &[usize],
        base: usize,
    ) -> Result<(), ClockRefusal> {
        // A row-count clock cannot go backwards, so this costs nothing there.
        // With a clock it can fail under `"error"`, the default, which
        // refuses every step back, or under `"reset_state"` with a
        // `min_backwards_jump` above 0, which refuses a late row; the pass
        // runs whenever one of those can refuse a row: that is what keeps
        // the refusal chunk-level and the bank untouched.
        let Some(clock) = clock else {
            return Ok(());
        };
        let can_refuse = matches!(cfg.on_clock_reset, online_core::OnClockReset::Error)
            || cfg.min_backwards_jump > 0.0;
        if !can_refuse {
            return Ok(());
        }
        let mut state = self.persisted.clock.clone();
        for (ri, &row) in rows.iter().enumerate() {
            let at = base + ri;
            // `accept` only routes the delta into `pending`; whether the row
            // is refused depends on the clock and session alone.
            let prev = state.last_clock();
            let adv = state.advance(cfg, Some(clock.at(at)), session.map(|s| s[at]), true);
            if let Some(raw) = adv.backwards {
                return Err(ClockRefusal::new(
                    raw,
                    row,
                    adv.disorder,
                    Some(clock.at(at)),
                    prev,
                ));
            }
        }
        Ok(())
    }

    /// Pass 1 of [`Self::process_chunk`]: the clock schedule of `rows`, on
    /// a copy of the clock, with what it leaves: the clock, the count of
    /// accepted rows and of rows fed, and where the `coef` cadence stands.
    /// What it decides depends on the clock and the input columns alone,
    /// never on the models, so the window budget's pre-pass
    /// ([`Self::window_prepass`]) replays the schedule the run will follow. A
    /// step back the policy refuses is handed back for the caller to name.
    #[allow(clippy::too_many_arguments)]
    fn schedule(
        &self,
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        features: &FeatureRows,
        clock: Option<&ClockCol>,
        session: Option<&[u64]>,
        weight: Option<&[f64]>,
        rows: &[usize],
        base: usize,
        coef_at: Option<usize>,
    ) -> Result<Scheduled, ClockRefusal> {
        let n_rows = rows.len();
        let mut clock_state = self.persisted.clock.clone();
        let mut rows_seen = self.persisted.rows_seen;
        let mut fed = self.persisted.fed;
        let coef_plan = CoefPlan::of(spec);
        let mut coef_cadence = self.persisted.coef_cadence;
        let mut plans: Vec<RowPlan> = Vec::with_capacity(n_rows);
        for (ri, &row) in rows.iter().enumerate() {
            let i = base + ri;
            // Null arrives as NaN from extraction, so one `usable` covers
            // null, NaN, infinity and the bound.
            let w = weight.map(|w| w[i]);
            let accept = accepts(features, weight, i);
            let c = clock.map(|c| c.at(i));
            // A clock below the previous row's, before the schedule decides
            // what to do about it; the summary counts them (task 35).
            let prev = clock_state.last_clock();
            let below = matches!((c, prev), (Some(c), Some(p)) if c.is_before(p));
            // Stamped: every row this stream commits is, so the exact clock
            // a window's edge reads never misses a step (task 175).
            let adv = clock_state.advance_stamped(
                cfg,
                &self.exact_caps,
                c,
                session.map(|s| s[i]),
                accept,
            );
            // A step back the policy refuses: hand the offending delta (and,
            // for a late row, the minimum) back so the caller can name the
            // row and column.
            if let Some(raw) = adv.backwards {
                return Err(ClockRefusal::new(raw, row, adv.disorder, c, prev));
            }
            // Saturating, as every count the bank keeps (task 160, PA2).
            if accept {
                rows_seen = rows_seen.saturating_add(1);
            }
            // The clock the fields show (task 152): the column's value, or
            // the row's index in the group, every row counted.
            let shown = c.or(Some(ClockValue::F64(fed as f64)));
            fed = fed.saturating_add(1);
            // With no cadence the group's last accepted row in the chunk
            // reports the coefficients, `coef_at`, which may sit in an earlier
            // run than the chunk's last (`ChunkOut::run_rows`); a chunk whose
            // last row of the group was skipped carried none (review round 4,
            // PB2). A cadence reads the clock and the rows alone, so its rows
            // do not move with the chunking (task 178).
            let want_coef = match coef_plan {
                CoefPlan::ChunkEnds => accept && coef_at == Some(i),
                CoefPlan::Cadence { clock: every, rows } => {
                    coef_cadence.step(every, rows, c, &adv, accept)
                }
            };
            plans.push(RowPlan {
                ri,
                i,
                pending: usize::MAX,
                d_clock: adv.d_clock,
                stamp: adv.stamp,
                elapsed: adv.elapsed_stamp,
                clock: shown,
                reset: adv.reset,
                blend: !adv.reset && adv.session_changed,
                session_changed: adv.session_changed,
                backwards: below && !adv.session_changed,
                capped: adv.capped,
                accept,
                want_coef,
                emit: true,
                drift_ri: ri,
                learn: true,
                buffered: false,
                w: w.unwrap_or(1.0),
            });
        }
        Ok(Scheduled {
            clock: clock_state,
            rows_seen,
            fed,
            coef_cadence,
            plans,
        })
    }

    /// The first overrun a chunk's rows would give a window's ring past a
    /// refusing budget -- the bytes and the cadence, as the ring would report
    /// them -- found by replaying the chunk's schedule on shadows of every
    /// ring ([`online_core::WindowShadow`]) before any model is touched
    /// (docs/PLAN.md task 115 (d)). The bank refuses such a chunk whole; the
    /// ring used to find the overrun with the chunk half learned, and the
    /// bank then refused every call after it. `None` when no ring would
    /// cross. `None` as well where the replay cannot see: under
    /// `drift_action = "reset"`, whose resets depend on the residuals, and
    /// for a snapshot that grows within the chunk. There the ring's own
    /// refusal still stops the run.
    #[allow(clippy::too_many_arguments)]
    pub fn window_prepass(
        &self,
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        features: &FeatureRows,
        targets: &[Vec<f64>],
        clock: Option<&ClockCol>,
        session: Option<&[u64]>,
        weight: Option<&[f64]>,
        rows: &[usize],
        base: usize,
        formulas: Option<FormulaTargets<'_>>,
    ) -> Option<(usize, online_core::Cadence)> {
        let mut live = self.window_shadows();
        // A learned row adds at most one snapshot to a ring, and the rows a
        // `embargo` holds are learned in this chunk at the most.
        let most = rows.len() + self.persisted.pending.len();
        if !live.iter().any(|s| s.could_refuse(most)) {
            return None;
        }
        if !self.persisted.drift.is_empty() && spec.drift_action.as_deref() == Some("reset") {
            return None;
        }
        let mut plans = self
            .schedule(
                spec, cfg, features, clock, session, weight, rows, base, None,
            )
            .ok()?
            .plans;
        let mut pending = self.persisted.pending.clone();
        let mut held_break = self.persisted.held_break;
        Self::apply_label_delay(
            self.embargo,
            &mut pending,
            &mut held_break,
            &mut plans,
            features,
            targets,
            formulas,
        );
        // A reset rebuilds each instance's model and its residual ring, as
        // `Instance::reset` does, so their shadows start empty.
        let mut fresh: Option<Vec<WindowShadow>> = None;
        for plan in &plans {
            if plan.reset {
                live = fresh
                    .get_or_insert_with(|| {
                        Stream::new(spec)
                            .map(|s| s.window_shadows())
                            .unwrap_or_default()
                    })
                    .clone();
            }
            // The rows `run_instance` steps the models on, with the delta and
            // the stamp it steps them with.
            if !(plan.accept && plan.learn) {
                continue;
            }
            for shadow in &mut live {
                shadow.learn(plan.d_clock, plan.stamp);
                if let Some(over) = shadow.over_budget() {
                    return Some(over);
                }
            }
        }
        None
    }

    /// A shadow of every window ring: each model's, then each instance's
    /// residual ring.
    /// Whether a ring of this stream could refuse a chunk of `rows` rows
    /// past its budget: what the window pre-pass replays for.
    pub(crate) fn could_refuse_window(&self, rows: usize) -> bool {
        let most = rows + self.persisted.pending.len();
        self.window_shadows().iter().any(|s| s.could_refuse(most))
    }

    fn window_shadows(&self) -> Vec<WindowShadow> {
        let n_slots = self.n_slots();
        self.models
            .iter()
            .filter_map(|(_, m)| m.window_shadow())
            .chain(
                self.persisted
                    .resid_win
                    .iter()
                    .flatten()
                    .map(|r| r.shadow(n_slots)),
            )
            .collect()
    }

    /// Process this stream's rows of one chunk, writing into flat per-slot
    /// buffers (docs/PERFORMANCE.md P1, P2).
    ///
    /// Two passes. The first walks the rows advancing the clock and deciding
    /// which are accepted -- that depends only on the clock and the input
    /// columns, never on the models. The second runs each model instance over
    /// the whole chunk, and because instances share nothing but that schedule,
    /// they run **in parallel**: a five-half-life grid on a single stream is
    /// five independent recursions rather than one serial loop.
    ///
    /// The one exception is `drift_action = "reset"`, where a break detected by
    /// any instance resets all of them, so instances are coupled *within a
    /// row*. That case keeps row-major order. Both paths call the same
    /// `run_instance`, so there is one implementation of the arithmetic.
    ///
    /// On a step back the policy refuses, returns the raw delta and the
    /// absolute row it happened at, for the caller to name, and
    /// leaves the stream untouched (pass 1 runs on a copy of the clock). The
    /// bank runs [`Stream::check_clock`] over every stream before it runs
    /// this on any, so the refusal is per chunk, not per stream.
    ///
    /// `rows` are the absolute rows of the chunk this stream owns, in order;
    /// they name rows in the output and in errors. The input columns are read
    /// at `base + ri` for the `ri`-th of them: the bank lays every spec's
    /// columns out group after group (docs/PERFORMANCE.md P9), so a stream's
    /// rows are one contiguous run whatever order the frame interleaves its
    /// groups in.
    ///
    /// The bank feeds a chunk's rows in runs of [`ChunkOut::run_rows`], each
    /// into its own buffers, and says with `coef_at` which input row is the
    /// group's last accepted one in the chunk ([`last_accepted`]): that row
    /// is where the coefficients are reported without a cadence. Chunk
    /// invariance makes the runs the same computation as one.
    #[allow(clippy::too_many_arguments)]
    pub fn process_chunk(
        &mut self,
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        features: &FeatureRows,
        targets: &[Vec<f64>],
        clock: Option<&ClockCol>,
        session: Option<&[u64]>,
        weight: Option<&[f64]>,
        rows: &[usize],
        base: usize,
        out: &mut ChunkOut,
        coef_at: Option<usize>,
        formulas: Option<FormulaTargets<'_>>,
    ) -> Result<(), ClockRefusal> {
        let n_rows = rows.len();
        out.rows.extend_from_slice(rows);

        // ---- pass 1: the clock schedule, models untouched ----
        // On a copy of the clock, committed below, so a refused row leaves
        // the stream exactly as it was.
        let Scheduled {
            clock: clock_state,
            rows_seen,
            fed,
            coef_cadence,
            mut plans,
        } = self.schedule(
            spec, cfg, features, clock, session, weight, rows, base, coef_at,
        )?;
        for plan in plans.iter().filter(|p| p.accept) {
            out.processed[plan.ri] = true;
        }

        self.persisted.clock = clock_state;
        self.persisted.rows_seen = rows_seen;
        self.persisted.fed = fed;
        self.persisted.coef_cadence = coef_cadence;

        // ---- the data summary (docs/PLAN.md task 35) ----
        // After the clock is committed, so a refused row above has fed
        // nothing; per row in row order, so chunking cannot move a bit; and
        // before `embargo` moves a break's events onto the row that
        // learns them, so a break counts where it arrived.
        if let Some(summary) = self.persisted.summary.as_mut() {
            for plan in plans.iter().filter(|p| p.direct()) {
                let i = plan.i;
                summary.feed_row(
                    features.row(i),
                    targets,
                    weight.map(|w| w[i]),
                    clock.map(|c| c.at(i)),
                    i,
                );
                summary.events(plan.session_changed, plan.backwards, plan.reset);
                if plan.accept {
                    // A plain target counts at its row; a formula target
                    // counts when the row is released with a value (review
                    // R1, D3: a window cut null taught nothing).
                    let has_target = targets.is_empty() || targets.iter().any(|t| usable(t[i]));
                    summary.accepted(plan.w, has_target);
                }
            }
        }

        // ---- embargo: hold each row back until its label matures ----
        // Rewrites the plan list into (release, ..., score this row) order,
        // and hands the released rows' values out beside it. Release depends
        // on the rows alone, so chunking cannot move a single one.
        let released = Self::apply_label_delay(
            self.embargo,
            &mut self.persisted.pending,
            &mut self.persisted.held_break,
            &mut plans,
            features,
            targets,
            formulas,
        );

        // A released row whose formula target resolved with a value, and
        // whose plain targets gave none at its row, is learned from now
        // (review R1, D3); counted here, after the release, in row order.
        // The values themselves join the targets' statistics here too, at
        // any weight, as a plain target's do at its row: the row was fed,
        // and counted a null, before its window closed (review round 4,
        // PA9).
        if let (Some(summary), Some(f)) = (self.persisted.summary.as_mut(), formulas) {
            let n_features = spec.features.len();
            for plan in plans.iter().filter(|p| !p.direct()) {
                let ys = &released[plan.pending].ys;
                for &k in f.slots {
                    if let Some(y) = ys[k] {
                        summary.resolved(n_features, k, y);
                    }
                }
                let plain = ys
                    .iter()
                    .enumerate()
                    .any(|(k, y)| !f.slots.contains(&k) && y.is_some());
                let formula = f.slots.iter().any(|&k| ys[k].is_some());
                if plan.w > 0.0 && !plain && formula {
                    summary.learned_late();
                }
            }
        }

        // ---- the clocks a row shows (docs/PLAN.md task 152) ----
        // Its own, and the newest row the models had learned from, at a
        // positive weight, when it was scored; in plan order, so a row
        // released just before this one counts, and a reset clears it.
        for plan in &plans {
            if plan.reset {
                self.persisted.last_learned = None;
            }
            if spec.emit_clocks && plan.emit && plan.accept {
                out.scored_clock[plan.ri] = plan.clock;
                out.learned_clock[plan.ri] = self.persisted.last_learned;
            }
            if plan.accept && plan.learn && plan.w > 0.0 {
                self.persisted.last_learned = if plan.direct() {
                    plan.clock
                } else {
                    released[plan.pending].clock
                };
            }
        }

        // ---- pass 2: the instances ----
        let drift_resets = spec.drift_action.as_deref() == Some("reset");
        let coupled = !self.persisted.drift.is_empty() && drift_resets;
        // `min_weight` is read by every instance; move it out so the split
        // below can borrow the rest of `self` mutably.
        let min_weight = std::mem::take(&mut self.min_weight);
        let models = self.models.iter_mut().map(|(_, m)| ModelRef::Learn(m));
        let rings = self
            .persisted
            .resid_win
            .iter_mut()
            .map(|r| r.as_mut().map(SpreadRef::Learn));
        let diag = Diagnostics {
            resid_var: &mut self.persisted.resid_var,
            resid_w: &mut self.persisted.resid_w,
            drift: &mut self.persisted.drift,
            resid_q: &mut self.persisted.resid_q,
            autocorr: &mut self.persisted.autocorr,
            metrics: &mut self.persisted.metrics,
            conformal: &mut self.persisted.conformal,
            scratch: &mut self.scratch,
            score_pred: &mut self.persisted.score_pred,
            decay_time: &mut self.persisted.decay_time,
            pending_clock: &mut self.persisted.pending_clock,
            notified: &mut self.persisted.notified,
        };
        let mut insts = build_instances(spec, models, rings, &self.decays, diag, out, n_rows);
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
                        &released,
                        &min_weight,
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
                run_instance(
                    inst,
                    &plans,
                    features,
                    targets,
                    &released,
                    &min_weight,
                    false,
                );
            });
        } else if let Some(inst) = insts.first_mut() {
            // One instance: a drift reset has nothing to coordinate with, so
            // it resets itself inline at the row that fired.
            run_instance(
                inst,
                &plans,
                features,
                targets,
                &released,
                &min_weight,
                drift_resets,
            );
        }
        drop(insts);
        self.min_weight = min_weight;
        Ok(())
    }

    /// Rewrite a chunk's plan list so each accepted row is *scored* where it
    /// sits and *learned from* only once `embargo` has passed
    /// (docs/ENHANCEMENTS.md E47). A no-op, and one `Option` test, for a
    /// spec without a delay.
    ///
    /// The rule is one line: at each accepted row, every buffered row whose
    /// wait has run out is released -- in arrival order, each replaying the
    /// clock delta it arrived with, so the models see exactly the sequence of
    /// gaps they would have seen without the delay -- and then the row itself
    /// is scored and buffered. The wait counts the time that passed, the
    /// clock column's own steps ([`online_core::ClockAdvance::elapsed`]), not
    /// the capped delta: `gap_cap` and `session_gap` say how much a model
    /// forgets across a break, not how long it lasted. Release therefore
    /// depends on the rows alone, which is what makes it chunk-invariant.
    ///
    /// A row's wait has run out when the accepted row's place on the
    /// elapsed clock is at least the embargo past the row's own
    /// ([`online_core::ClockAdvance::elapsed_stamp`], docs/PLAN.md task
    /// 176): two integers of nanoseconds on a temporal clock, one
    /// subtraction of raw values on a number clock, two row counts without a
    /// clock. Each held row counted its embargo down in doubles until then,
    /// and the rounded steps drifted: on rows 1 ms apart under `"2s"` every
    /// row was learned a row late.
    ///
    /// A break releases nothing early (docs/PLAN.md task 153). Its events --
    /// the lag rings' clear at a gap past `gap_cap` or a session change,
    /// and `session_shrink`'s blend -- wait in the buffer with the row that
    /// raised them, or with the next accepted row when a skipped row raised
    /// them, and run when that row is learned: after every row before the
    /// break, before every row after it, so no ring pairs rows across it
    /// (docs/REVIEW-E54-E64.md L2). Applied when the row arrived, they had
    /// forced every held row out first, and a label shorter-lived than the
    /// delay was learned before it was known. The models run one delay
    /// behind, events included. Only a clock **reset** acts on arrival: it
    /// drops the buffer, since the state those rows would teach is gone. A
    /// drift a released row trips is flagged on the row releasing it
    /// ([`RowPlan::drift_ri`]), and a drift reset restarts the models there,
    /// keeping the buffer: the rows it holds arrived after the one that
    /// tripped it.
    ///
    /// A formula target (docs/PLAN.md task 104) is not known at its row:
    /// the row waits, with no embargo as well, until the spec's window core
    /// has resolved it -- at the row that closed, cut or discarded its
    /// windows, which `formulas` carries by sequence number -- and its wait
    /// has run out, whichever is later. Resolutions are applied in row
    /// order, at the row that made them, so a coarse chunking releases
    /// nothing a fine one holds (the review under task 104: with an embargo
    /// equal to the window, a row exactly one window later has the wait run
    /// out while the window is still open).
    ///
    /// Returns the released rows' values, indexed by `RowPlan::pending`.
    #[allow(clippy::too_many_arguments)]
    fn apply_label_delay(
        embargo: Option<Embargo>,
        pending: &mut Vec<PendingRow>,
        held_break: &mut HeldBreak,
        plans: &mut Vec<RowPlan>,
        features: &FeatureRows,
        targets: &[Vec<f64>],
        formulas: Option<FormulaTargets<'_>>,
    ) -> Vec<PendingRow> {
        if embargo.is_none() && formulas.is_none() {
            return Vec::new();
        }
        let mut released: Vec<PendingRow> = Vec::new();
        let mut out: Vec<RowPlan> = Vec::with_capacity(plans.len());
        // One template for every replayed row: only `pending` and the
        // break's events differ, and a replay never emits, never wants
        // coefficients and never counts as a row of the frame. A drift it
        // detects is flagged on `at`, the row releasing it (task 160, PB1).
        let replay = |slot: usize, row: &PendingRow, at: usize| RowPlan {
            ri: usize::MAX,
            i: usize::MAX,
            pending: slot,
            d_clock: row.d_clock,
            stamp: row.stamp,
            elapsed: None,
            clock: row.clock,
            reset: false,
            blend: row.blend,
            session_changed: row.session_changed,
            backwards: false,
            capped: row.capped,
            accept: true,
            want_coef: false,
            emit: false,
            drift_ri: at,
            learn: true,
            buffered: false,
            w: row.w,
        };
        // The events of rows skipped since the last accepted one fall before
        // the next accepted row, so they wait with it, across a chunk
        // boundary too (`held_break` is the stream's).
        for plan in plans.drain(..) {
            if plan.reset {
                // The models restart at this row -- and the reset is applied
                // here, not a delay later, because the point of it is that
                // the state is no longer about this stream. The rows waiting
                // to teach that state go with it.
                pending.clear();
                *held_break = HeldBreak::default();
            }
            // The break's events run when its row is learned, not here.
            let quiet = RowPlan {
                session_changed: false,
                blend: false,
                capped: false,
                ..plan
            };
            // The resolutions made at or before this row, in order, at
            // every row of the group -- a skipped row resolves windows as
            // any row does, and the next accepted row may be chunks away.
            // The resolved rows are a prefix of the buffer (windows close in
            // clock order, a cut or a discard takes every open one), so the
            // first unresolved row still waiting ends the pass. A resolution
            // is made by a row of this group in this chunk, so it is never
            // left for a later one.
            if !plan.reset
                && let Some(res) = formulas.and_then(|f| f.resolved)
            {
                let f = formulas.expect("checked");
                let now = f.seqs[plan.i];
                for row in pending.iter_mut().skip_while(|r| r.resolved) {
                    match res.find(row.seq) {
                        Some((at, ys)) if at <= now => {
                            for (&slot, &y) in f.slots.iter().zip(ys) {
                                row.ys[slot] = Some(y).filter(|v| usable(*v));
                            }
                            row.resolved = true;
                        }
                        _ => break,
                    }
                }
            }
            if !plan.accept {
                // A skipped row teaches nothing and waits for nothing; its
                // time passes on the elapsed clock, which the next accepted
                // row's place carries, and its events wait with that row.
                held_break.session_changed |= plan.session_changed;
                held_break.blend |= plan.blend;
                held_break.capped |= plan.capped;
                out.push(quiet);
                continue;
            }
            if !plan.reset {
                // With no embargo a formula row waits for its window alone.
                let now = plan.elapsed;
                let ready = pending
                    .iter()
                    .take_while(|r| r.resolved && embargo.is_none_or(|e| e.passed(now, r.arrived)))
                    .count();
                for row in pending.drain(..ready) {
                    released.push(row);
                    out.push(replay(
                        released.len() - 1,
                        released.last().unwrap(),
                        plan.ri,
                    ));
                }
            }
            // The row itself: scored from the state as it now stands, and
            // buffered with the delta it arrived with, its place on the
            // elapsed clock and the break before it.
            let i = plan.i;
            pending.push(PendingRow {
                arrived: plan.elapsed,
                d_clock: plan.d_clock,
                stamp: plan.stamp,
                w: plan.w,
                xs: features.row(i).to_vec(),
                ys: targets
                    .iter()
                    .map(|t| Some(t[i]).filter(|v| usable(*v)))
                    .collect(),
                session_changed: held_break.session_changed || plan.session_changed,
                blend: held_break.blend || plan.blend,
                capped: held_break.capped || plan.capped,
                clock: plan.clock,
                seq: formulas.map_or(0, |f| f.seqs[i]),
                resolved: formulas.is_none(),
            });
            *held_break = HeldBreak::default();
            out.push(RowPlan {
                learn: false,
                buffered: true,
                ..quiet
            });
        }
        *plans = out;
        released
    }

    /// Score this stream's rows of one chunk against the state as it stands
    /// (docs/ENHANCEMENTS.md E31): every output [`Self::process_chunk`]
    /// would write for the row as the stream's next accepted row, and no
    /// update -- not to the models, the clock, the diagnostics or
    /// `rows_seen`. Row order is immaterial, since every row is scored from
    /// the same state; the clock distance each prediction extrapolates over
    /// is the row's own distance from the last learned row, with the cap,
    /// the pending time of skipped rows and the session policy applied as
    /// the learning path would apply them.
    ///
    /// The session policy holds too, because it is part of what "the next
    /// row" means: a row that would start a fresh stream (`session_gap =
    /// "reset"`) is scored by a fresh one -- null throughout, as it would be
    /// -- and a row that would blend toward the long run (`session_shrink`)
    /// is scored by a blended copy. A row before the last learned clock is
    /// scored against the state as it stands, a step of 0, whatever
    /// `restart_after_step_back` says: scoring learns nothing, so it neither refuses nor
    /// starts over (task 120). The scoring for each of the three is the one
    /// `run_instance`, with its updates switched off.
    ///
    /// Per field: `pred` and `lam_selected` as the model would report;
    /// `resid` where the row carries a usable target; `sigma`, `zscore`,
    /// the residual quantiles, autocorrelation and metrics from the
    /// diagnostics as they stand; `n_eff` frozen; `coef` on the last
    /// accepted row of the chunk (the same coefficients score every row);
    /// `drift` never fires. A weight column is not read, so a row
    /// `fit_predict` would skip for an unusable weight is scored anyway.
    /// `&self`, so a stream can be scored from several threads at once.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_chunk(
        &self,
        spec: &Spec,
        cfg: &online_core::ClockCfg,
        features: &FeatureRows,
        targets: &[Vec<f64>],
        clock: Option<&ClockCol>,
        session: Option<&[u64]>,
        rows: &[usize],
        base: usize,
        out: &mut ChunkOut,
    ) {
        let n_rows = rows.len();
        out.rows.extend_from_slice(rows);
        // Three classes of row, by what the clock says the row would do to
        // the state it is scored against: nothing, reset it, or blend it.
        let mut classes: [Vec<RowPlan>; 3] = Default::default();
        let [as_is, fresh, blended] = [0, 1, 2];
        // (class, position in it) of the last accepted row, which carries
        // the coefficients for the chunk.
        let mut last_accepted: Option<(usize, usize)> = None;
        // Scoring learns nothing, so nothing that guards what the bank
        // *learns* applies: a fresh frame scored against the bank as it
        // stands may well sit before the last learned clock, and such a row
        // is scored against the state as it stands -- a step of 0 -- under
        // either policy, never refused and never a reset (task 120). A
        // session change still takes its rule.
        for ri in 0..rows.len() {
            let i = base + ri;
            let accept = all_usable(features.row(i));
            // Every row is the first after the last learned one.
            let adv = self.persisted.clock.clone().advance_scoring(
                cfg,
                clock.map(|c| c.at(i)),
                session.map(|s| s[i]),
            );
            if accept {
                out.processed[ri] = true;
            }
            let plan = RowPlan {
                ri,
                i,
                pending: usize::MAX,
                d_clock: adv.d_clock,
                // Scored, never learned: no window moves, no row is held.
                stamp: None,
                elapsed: None,
                // Without a clock column, the row's index in its group: the
                // rows learned, then this call's rows in order, as
                // `fit_predict` counts them (task 159, B2: every scored row
                // showed the count of rows fed).
                clock: clock.map(|c| c.at(i)).or(Some(ClockValue::F64(
                    (self.persisted.fed + ri as u64) as f64,
                ))),
                reset: false,
                blend: false,
                session_changed: false,
                backwards: false,
                // Scoring moves nothing, the lag ring included: the classes
                // below give the row a *copy* of the model in the state a
                // learning stream would have reached, and a copy that
                // cleared its lags would answer for a stream that had
                // learned the row. `predict` never mutates.
                capped: false,
                accept,
                want_coef: false,
                emit: true,
                drift_ri: ri,
                learn: false,
                buffered: false,
                w: 1.0,
            };
            // Scored against the state as it stands, so the learned clock is
            // the fit's last learned row on every row (task 152).
            if spec.emit_clocks && accept {
                out.scored_clock[ri] = plan.clock;
                out.learned_clock[ri] = self.persisted.last_learned;
            }
            // A new session's row under `group_close = "session"` is a fresh
            // stream's first row, as `fit_predict` restarts the stream at the
            // change; it was scored by the closed session's fit (review
            // 2026-09-12, C19).
            let class = if adv.reset || (adv.session_changed && spec.closes_on_session()) {
                fresh
            } else if adv.session_changed {
                blended
            } else {
                as_is
            };
            classes[class].push(plan);
            if accept {
                last_accepted = Some((class, classes[class].len() - 1));
            }
        }
        if let Some((class, at)) = last_accepted {
            classes[class][at].want_coef = true;
        }

        self.score(
            spec,
            &classes[as_is],
            &self.models,
            features,
            targets,
            out,
            n_rows,
        );
        if !classes[fresh].is_empty() {
            let stream = Stream::new(spec).expect("spec was already validated");
            stream.score(
                spec,
                &classes[fresh],
                &stream.models,
                features,
                targets,
                out,
                n_rows,
            );
        }
        if !classes[blended].is_empty() {
            let mut models = self.models.clone();
            models
                .iter_mut()
                .for_each(|(_, m)| m.blend_toward_long_run());
            self.score(
                spec,
                &classes[blended],
                &models,
                features,
                targets,
                out,
                n_rows,
            );
        }
    }

    /// Run these plans through every instance with `learn = false`, scoring
    /// with `models` (this stream's own, or a variant of them) against this
    /// stream's diagnostics.
    #[allow(clippy::too_many_arguments)]
    fn score(
        &self,
        spec: &Spec,
        plans: &[RowPlan],
        models: &[(String, AnyModel)],
        features: &FeatureRows,
        targets: &[Vec<f64>],
        out: &mut ChunkOut,
        n_rows: usize,
    ) {
        if plans.is_empty() {
            return;
        }
        // The diagnostics are read, never written, on this path; a copy is
        // how a `&self` stream lends them to the same `run_instance` that
        // updates them when learning. They are per-slot summaries, so the
        // copy is a few hundred bytes per slot, and a quantile sketch's
        // buckets, a few kilobytes.
        let n = self.models.len();
        let (mut resid_var, mut resid_w) = (
            self.persisted.resid_var.clone(),
            self.persisted.resid_w.clone(),
        );
        let (mut drift, mut resid_q) =
            (self.persisted.drift.clone(), self.persisted.resid_q.clone());
        let (mut autocorr, mut metrics) = (
            self.persisted.autocorr.clone(),
            self.persisted.metrics.clone(),
        );
        let mut conformal = self.persisted.conformal.clone();
        let mut scratch: Vec<Scratch> = (0..n).map(|_| Scratch::default()).collect();
        // Scoring buffers nothing and replays nothing, so its queues stay empty.
        let mut score_pred: Vec<std::collections::VecDeque<Vec<f64>>> =
            (0..n).map(|_| std::collections::VecDeque::new()).collect();
        // Scoring learns nothing, so the decay time does not move and no
        // notice a scoring row might raise is kept.
        let mut decay_time = self.persisted.decay_time.clone();
        let mut pending_clock = self.persisted.pending_clock.clone();
        let mut notified = self.persisted.notified.clone();
        let diag = Diagnostics {
            resid_var: &mut resid_var,
            resid_w: &mut resid_w,
            drift: &mut drift,
            resid_q: &mut resid_q,
            autocorr: &mut autocorr,
            metrics: &mut metrics,
            conformal: &mut conformal,
            scratch: &mut scratch,
            score_pred: &mut score_pred,
            decay_time: &mut decay_time,
            pending_clock: &mut pending_clock,
            notified: &mut notified,
        };
        let models = models.iter().map(|(_, m)| ModelRef::Score(m));
        let rings = self
            .persisted
            .resid_win
            .iter()
            .map(|r| r.as_ref().map(SpreadRef::Score));
        let mut insts = build_instances(spec, models, rings, &self.decays, diag, out, n_rows);
        if insts.len() > 1 {
            use rayon::prelude::*;
            insts.par_iter_mut().for_each(|inst| {
                run_instance(inst, plans, features, targets, &[], &self.min_weight, false);
            });
        } else if let Some(inst) = insts.first_mut() {
            run_instance(inst, plans, features, targets, &[], &self.min_weight, false);
        }
    }
}

/// How an instance reaches its model: exclusively, to learn, or shared, to
/// score. [`Stream::predict_chunk`] runs on `&Stream`, so it can only hand
/// out the latter -- which is what lets a bank be scored from several
/// threads at once.
enum ModelRef<'a> {
    Learn(&'a mut AnyModel),
    Score(&'a AnyModel),
}

impl ModelRef<'_> {
    fn get(&self) -> &AnyModel {
        match self {
            ModelRef::Learn(m) => m,
            ModelRef::Score(m) => m,
        }
    }

    /// Only a learning instance is ever stepped, blended or reset; the
    /// scoring plans never ask for any of the three.
    fn get_mut(&mut self) -> &mut AnyModel {
        match self {
            ModelRef::Learn(m) => m,
            ModelRef::Score(_) => unreachable!("a scoring instance never updates its model"),
        }
    }
}

/// How an instance reaches its spread's ring (S1): exclusively, to learn,
/// or shared, to score, as [`ModelRef`] reaches the model.
enum SpreadRef<'a> {
    Learn(&'a mut ResidWindow),
    Score(&'a ResidWindow),
}

impl SpreadRef<'_> {
    fn get(&self) -> &ResidWindow {
        match self {
            SpreadRef::Learn(r) => r,
            SpreadRef::Score(r) => r,
        }
    }

    fn get_mut(&mut self) -> &mut ResidWindow {
        match self {
            SpreadRef::Learn(r) => r,
            SpreadRef::Score(_) => unreachable!("a scoring instance never updates its spread"),
        }
    }
}

/// Everything an instance touches besides its model and its output slice,
/// split per instance below. `process_chunk` lends the stream's own;
/// `predict_chunk` lends a copy it drops afterwards.
struct Diagnostics<'a> {
    resid_var: &'a mut [Vec<f64>],
    resid_w: &'a mut [Vec<f64>],
    drift: &'a mut [Vec<PageHinkley>],
    resid_q: &'a mut [Vec<EwQuantile>],
    autocorr: &'a mut [Vec<EwAutoCorr>],
    metrics: &'a mut [Vec<SlotMetrics>],
    conformal: &'a mut [Vec<Conformal>],
    scratch: &'a mut [Scratch],
    score_pred: &'a mut [std::collections::VecDeque<Vec<f64>>],
    decay_time: &'a mut [f64],
    pending_clock: &'a mut [f64],
    notified: &'a mut [Notified],
}

/// Split per-instance state and output into disjoint pieces, so instances
/// can run concurrently. Every `ChunkOut` buffer is laid out model-major,
/// which is what makes an instance's region one contiguous slice; the
/// state vectors are already `[mi]`-indexed.
#[allow(clippy::too_many_arguments)]
fn build_instances<'a>(
    spec: &'a Spec,
    models: impl Iterator<Item = ModelRef<'a>>,
    mut rings: impl Iterator<Item = Option<SpreadRef<'a>>>,
    decays: &[Decay],
    diag: Diagnostics<'a>,
    out: &'a mut ChunkOut,
    n_rows: usize,
) -> Vec<Instance<'a>> {
    let n = decays.len();
    let block = out.n_slots * n_rows;
    let n_targets = if n == 0 || n_rows == 0 || out.lam_selected.is_empty() {
        0
    } else {
        out.lam_selected.len() / (n * n_rows)
    };

    let mut drift = diag.drift.iter_mut();
    let mut resid_q = diag.resid_q.iter_mut();
    let mut autocorr = diag.autocorr.iter_mut();
    let mut metrics = diag.metrics.iter_mut();
    let mut conformal = diag.conformal.iter_mut();
    let mut o_pred = out.pred.chunks_mut(block.max(1));
    let mut o_resid = out.resid.chunks_mut(block.max(1));
    let mut o_sigma = out.sigma.chunks_mut(block.max(1));
    let mut o_resid_z = out.zscore.chunks_mut(block.max(1));
    let mut o_autocorr = out.autocorr.chunks_mut(block.max(1));
    let mut o_metrics = out.metrics.chunks_mut((3 * block).max(1));
    let mut o_conformal = out.conformal.chunks_mut((3 * block).max(1));
    let mut o_resid_q = out.resid_q.chunks_mut((out.n_levels * block).max(1));
    let mut o_drift = out.drift.chunks_mut(block.max(1));
    let mut o_n_eff = out.n_eff.chunks_mut(n_rows.max(1));
    let mut o_lam = out.lam_selected.chunks_mut((n_targets * n_rows).max(1));
    let mut o_coef = out.coef.iter_mut();
    let mut o_settled = out.settled.chunks_mut(n_rows.max(1));
    let mut o_reason = out.reason.chunks_mut(n_rows.max(1));
    let mut o_inflation = out.inflation.chunks_mut(block.max(1));
    let mut o_support_coef = out.support_coef.iter_mut();

    let n_slots = out.n_slots;
    let mut decays = decays.iter();
    let mut resid_var = diag.resid_var.iter_mut();
    let mut resid_w = diag.resid_w.iter_mut();
    let mut scratch = diag.scratch.iter_mut();
    let mut score_pred = diag.score_pred.iter_mut();
    let mut decay_time = diag.decay_time.iter_mut();
    let mut pending_clock = diag.pending_clock.iter_mut();
    let mut notified = diag.notified.iter_mut();

    // Pulled in lockstep: each iterator yields disjoint `&mut`s, so every
    // Instance owns its own piece of everything.
    models
        .map(|model| Instance {
            spec,
            shards: marginal_shards(spec, model.get()),
            model,
            resid_win: rings.next().expect("one per instance"),
            decay: *decays.next().expect("one per instance"),
            residuals: !spec.model.predicts_no_target(),
            resid_var: resid_var.next().expect("one per instance"),
            resid_w: resid_w.next().expect("one per instance"),
            drift: drift.next(),
            resid_q: resid_q.next(),
            autocorr: autocorr.next(),
            metrics: metrics.next(),
            conformal: conformal.next(),
            scratch: scratch.next().expect("one per instance"),
            score_pred: score_pred.next().expect("one per instance"),
            n_slots,
            n_rows,
            o_pred: o_pred.next().unwrap_or_default(),
            o_resid: o_resid.next().unwrap_or_default(),
            o_sigma: o_sigma.next().unwrap_or_default(),
            o_resid_z: o_resid_z.next().unwrap_or_default(),
            o_autocorr: o_autocorr.next().unwrap_or_default(),
            o_metrics: o_metrics.next().unwrap_or_default(),
            o_conformal: o_conformal.next().unwrap_or_default(),
            o_resid_q: o_resid_q.next().unwrap_or_default(),
            o_drift: o_drift.next().unwrap_or_default(),
            o_n_eff: o_n_eff.next().unwrap_or_default(),
            o_lam: o_lam.next().unwrap_or_default(),
            o_coef: o_coef.next().expect("one per instance"),
            o_settled: o_settled.next().unwrap_or_default(),
            o_reason: o_reason.next().unwrap_or_default(),
            o_inflation: o_inflation.next().unwrap_or_default(),
            o_support_coef: o_support_coef.next().expect("one per instance"),
            decay_time: decay_time.next().expect("one per instance"),
            notified: notified.next().expect("one per instance"),
            // None without an embargo, which keeps no held rows' clock.
            pending_clock: pending_clock.next(),
        })
        .collect()
}

/// The ranges of features a `marginal` splits its pair work into
/// (docs/PLAN.md task 126): its spec's `shards` count as given, or under
/// `"auto"` as many as the model's width keeps busy on the pool the caller
/// is in ([`online_core::MarginalCfg::auto_shards`]); one for any other
/// model. Read afresh for each run, so a bank loaded on another machine
/// sizes itself to that machine's pool.
pub fn marginal_shards(spec: &Spec, model: &AnyModel) -> usize {
    match (&spec.model, model) {
        (
            ModelKind::Marginal {
                shards: Some(ShardSpec::Count(n)),
                ..
            },
            _,
        ) => *n,
        (
            ModelKind::Marginal {
                shards: Some(ShardSpec::Auto),
                ..
            },
            AnyModel::Marginal(m),
        ) => m.cfg().auto_shards(rayon::current_num_threads()),
        _ => 1,
    }
}

/// What pass 1 ([`Stream::schedule`]) leaves: the stream's clock and counts
/// after the rows, for the caller to commit, and the plan of each row.
struct Scheduled {
    clock: ClockState,
    rows_seen: u64,
    fed: u64,
    coef_cadence: CoefCadence,
    plans: Vec<RowPlan>,
}

/// What pass 1 decided about one row, so pass 2 can replay it per instance
/// without touching the clock again.
struct RowPlan {
    /// Position within the stream's run (index into the output buffers).
    ri: usize,
    /// Position in the input columns: the run's base plus `ri`, since the
    /// columns are laid out group after group. [`usize::MAX`] for a row
    /// replayed out of the `embargo` buffer, whose values are in
    /// `pending` instead (E47).
    i: usize,
    /// Index into the released-rows slice for a replayed row; [`usize::MAX`]
    /// for a row read from the columns.
    pending: usize,
    d_clock: f64,
    /// The row's place on the decayed clock, held exactly, which a window
    /// decides its edge from ([`online_core::Stamp`], docs/PLAN.md task
    /// 175): `None` for a skipped row and a row only scored.
    stamp: Option<Stamp>,
    /// The row's place on the elapsed clock, held exactly: every step of the
    /// clock column since the stream began, uncapped (docs/PLAN.md task
    /// 153), from which the `embargo` buffer measures each held row's wait
    /// (task 176). `None` for a skipped row, a row only scored and a row
    /// replayed out of the buffer.
    elapsed: Option<Stamp>,
    /// The row's clock as the clock fields show it (task 152): the column's
    /// value, or the group's row index with no column.
    clock: Option<ClockValue>,
    reset: bool,
    blend: bool,
    /// The session id differed from the previous row's.
    session_changed: bool,
    /// The clock fell below the previous row's within a session.
    backwards: bool,
    /// The clock jumped further than `gap_cap`, so the delta the models
    /// see is the ceiling. Anything lagged by *rows* is stale
    /// (`OnlineModel::clear_lags`, docs/PLAN.md task 47).
    capped: bool,
    accept: bool,
    want_coef: bool,
    /// Write this row's outputs at `ri`. False for a replayed row: it is a
    /// lesson, not a row of the frame.
    emit: bool,
    /// The output row a drift detection at this row is flagged on: `ri`
    /// for a row learned where it sits; for a row replayed out of the
    /// `embargo` buffer, the row whose clock released it, the one being
    /// output as the release happens -- the replayed row's own is out
    /// already. A detection on a replay set no flag, so under an embargo
    /// `drift_<t>` was never true, and `drift_action = "reset"` restarted
    /// the model unannounced (task 160, PB1).
    drift_ri: usize,
    /// Step the models and fold the diagnostics. False for a row whose label
    /// has not matured: it is scored here and learned from later.
    learn: bool,
    /// Pushed into the `embargo` buffer as it was scored: its prediction
    /// is kept for the diagnostics to fold when it matures (C21).
    buffered: bool,
    w: f64,
}

impl RowPlan {
    /// A row of the chunk, read from the columns, learned from where it sits.
    #[inline]
    fn direct(&self) -> bool {
        self.pending == usize::MAX
    }
}

/// Per-row scratch, one set per model instance so instances can run
/// concurrently without sharing buffers.
#[derive(Default)]
pub struct Scratch {
    ys: Vec<Option<f64>>,
    r: Vec<f64>,
    sig: Vec<f64>,
    zs: Vec<f64>,
    /// Each target's own weight before the row, when the model keeps one
    /// (review 2026-09-12, S2).
    tn: Vec<f64>,
    /// The noise gate's statistic per slot before the row, when the model
    /// has one, and the row's own (docs/WARMUP-AND-CONVERGENCE.md §2.1).
    infl: Vec<f64>,
    row_infl: Vec<f64>,
}

/// One model instance's state and its disjoint slice of the chunk output.
struct Instance<'a> {
    /// For rebuilding this instance on a reset: the spec is the only
    /// description of a pristine model, and `decay` says which of its
    /// instances this is.
    spec: &'a Spec,
    model: ModelRef<'a>,
    /// The ring that cuts this instance's spread at the fit's window
    /// boundary (S1); `None` without a window, or when nothing reads it.
    resid_win: Option<SpreadRef<'a>>,
    decay: Decay,
    /// False for a model that predicts no target (`ModelKind::
    /// predicts_no_target`): no residual is formed, tracked or written for
    /// it, and `ChunkOut` gives it no `resid` buffer.
    residuals: bool,
    resid_var: &'a mut Vec<f64>,
    resid_w: &'a mut Vec<f64>,
    drift: Option<&'a mut Vec<PageHinkley>>,
    resid_q: Option<&'a mut Vec<EwQuantile>>,
    autocorr: Option<&'a mut Vec<EwAutoCorr>>,
    metrics: Option<&'a mut Vec<SlotMetrics>>,
    conformal: Option<&'a mut Vec<Conformal>>,
    scratch: &'a mut Scratch,
    /// This instance's score-time predictions for the rows still waiting
    /// under `embargo`, oldest first (C21).
    score_pred: &'a mut std::collections::VecDeque<Vec<f64>>,
    n_slots: usize,
    n_rows: usize,
    o_pred: &'a mut [f64],
    o_resid: &'a mut [f64],
    o_sigma: &'a mut [f64],
    o_resid_z: &'a mut [f64],
    o_autocorr: &'a mut [f64],
    o_metrics: &'a mut [f64],
    o_conformal: &'a mut [f64],
    o_resid_q: &'a mut [f64],
    o_drift: &'a mut [bool],
    o_n_eff: &'a mut [f64],
    o_lam: &'a mut [f64],
    o_coef: &'a mut Vec<Option<Vec<f64>>>,
    o_settled: &'a mut [f64],
    o_reason: &'a mut [u8],
    o_inflation: &'a mut [f64],
    o_support_coef: &'a mut Vec<Option<Vec<f64>>>,
    /// The decay time this instance has seen (docs/WARMUP-AND-CONVERGENCE.md
    /// §2), read before each row and advanced by the rows it learns from.
    decay_time: &'a mut f64,
    /// Ranges of features a `marginal` splits its pair work into
    /// ([`marginal_shards`]); one for every other model.
    shards: usize,
    /// The readiness notices this instance has raised, and the ones pending.
    notified: &'a mut Notified,
    /// Clock the rows held under `embargo` have covered and the models
    /// have not yet decayed by: added as a row is buffered, taken back as it
    /// is released. `settled_frac` counts it, so a scored row reads what the
    /// doubled stream (E47's oracle) reads for it, where every held row has
    /// already decayed the model as a weight-0 row (§8). The stream keeps
    /// it across chunks ([`Persisted::pending_clock`]), and keeps none
    /// without an embargo, where no row is ever held.
    pending_clock: Option<&'a mut f64>,
}

impl Instance<'_> {
    /// Restart this instance: the same thing a clock reset or a drift break
    /// does, applied to one instance rather than the whole stream.
    fn reset(&mut self) {
        let spec = self.spec;
        // This instance alone, at its own decay: `build_models` built every
        // instance of the grid to keep one (review 2026-09-12, P2).
        *self.model.get_mut() = build_one(spec, self.decay).expect("spec was already validated");
        self.resid_var.iter_mut().for_each(|v| *v = 0.0);
        self.resid_w.iter_mut().for_each(|v| *v = 0.0);
        // A rebuilt model has seen no decay: it settles from here. The
        // clock of the rows held under `embargo` is the held rows' own: a
        // drift reset keeps them, and each teaches the rebuilt model as it
        // is released, its delta moving from the held clock to the decay
        // time as any held row's does, so the held clock stays as it stands
        // -- the kept rows' summed `d_clock`, which the rebuilt model will
        // decay by. Zeroed here, each release took back a delta the reset
        // had never added, and `settled_frac` left the held rows out for
        // the rest of the stream (review round 4, PB1). A clock reset drops
        // the rows, and the held clock with them (`run_instance`).
        *self.decay_time = 0.0;
        if let Some(ring) = self.resid_win.as_mut() {
            *ring.get_mut() = resid_window(spec)
                .expect("spec was already validated")
                .expect("the spec that built the ring keeps one");
        }
        if let Some(d) = self.drift.as_deref_mut() {
            d.iter_mut().for_each(PageHinkley::reset);
        }
        // Residual diagnostics restart with the model they describe.
        if let Some(q) = self.resid_q.as_deref_mut() {
            q.iter_mut().for_each(EwQuantile::reset);
        }
        if let Some(a) = self.autocorr.as_deref_mut() {
            let lag = spec.resid_autocorr_lag_or_default();
            a.iter_mut()
                .for_each(|e| *e = EwAutoCorr::new(lag).expect("validated"));
        }
        if let Some(m) = self.metrics.as_deref_mut() {
            m.iter_mut().for_each(|s| *s = SlotMetrics::new());
        }
        if let (Some(c), Some(level)) = (self.conformal.as_deref_mut(), spec.conformal) {
            let rate = spec.conformal_rate_or_default();
            c.iter_mut()
                .for_each(|e| *e = Conformal::new(level, rate).expect("validated"));
        }
    }
}

/// Run one model instance over a run of rows. Returns whether any drift
/// detector fired, which is all the caller needs to decide about a reset.
///
/// This is the whole per-row arithmetic, and the only copy of it: the parallel
/// path calls it once per instance with every row, the drift-coupled path once
/// per instance per row, and [`Stream::predict_chunk`] with `learn = false`,
/// which reads every state the row would be scored against and writes none
/// of them back. Scoring is the learning path with its updates switched off
/// rather than a second loop, so the two cannot drift apart
/// (docs/ENHANCEMENTS.md E31).
#[allow(clippy::too_many_arguments)]
fn run_instance(
    inst: &mut Instance<'_>,
    plans: &[RowPlan],
    features: &FeatureRows,
    targets: &[Vec<f64>],
    // Rows released from the `embargo` buffer, indexed by
    // `RowPlan::pending`; empty for every spec without a delay (E47).
    released: &[PendingRow],
    min_weight: &[f64],
    // `drift_action = "reset"` with nothing else to coordinate with: restart
    // *at the row that fired*, not at the end of the chunk, or the rest of the
    // chunk keeps learning from the regime the detector just rejected.
    reset_on_drift: bool,
) -> bool {
    let mut drift_seen = false;
    let n_rows = inst.n_rows;
    let block = inst.n_slots * n_rows;
    // Whether `pred` is a probability against a 0/1 label rather than a
    // signed regression target -- the model's *declared* loss, not
    // something read off a row's value, so a slot's metrics mean the same
    // thing whatever data a chunk happens to carry (docs/PLAN.md task 76).
    // One flag per instance: `loss` is one setting for the whole model, not
    // per target.
    let binary_loss = match &inst.spec.model {
        ModelKind::Sgd { loss, .. } => loss.as_deref() == Some("logistic"),
        ModelKind::Ftrl { loss, .. } => loss.as_deref().unwrap_or("logistic") == "logistic",
        _ => false,
    };
    // A Poisson fit has no sign to hit: a rate is positive and a count is
    // never negative, so the sign test about zero agreed on every row and
    // `hit_rate` read 1.0 whatever the fit. It is null, as `po.eval` nulls a
    // metric that is not defined (review round 4, CC5; docs/PLAN.md task
    // 195, S5). Every other target's test is about zero: task 201 removed
    // the ratio target, whose test was about 1.
    let poisson = matches!(&inst.spec.model, ModelKind::Sgd { loss, .. }
        if loss.as_deref() == Some("poisson"));
    let hit_test = if poisson {
        HitTest::Undefined
    } else if binary_loss {
        HitTest::Threshold
    } else {
        HitTest::Sign
    };
    for plan in plans {
        if plan.reset {
            inst.reset();
            // The stream dropped its waiting rows at this row, skipped or
            // not (C5); the record of what they were scored with goes too,
            // and so does the clock they covered (E47).
            inst.score_pred.clear();
            if let Some(c) = inst.pending_clock.as_deref_mut() {
                *c = 0.0;
            }
        } else {
            if plan.blend {
                // A gentler alternative to resetting: revert partway toward
                // the long-run relationship (ENHANCEMENTS E6).
                inst.model.get_mut().blend_toward_long_run();
            }
            // The rows behind this one are no longer adjacent to it: drop
            // whatever is indexed by rows back, and nothing else
            // (docs/PLAN.md task 47). Before the `accept` test, because a
            // skipped row's gap breaks adjacency just as much; not under a
            // reset, which rebuilds the model whole.
            if plan.session_changed || plan.capped {
                inst.model.get_mut().clear_lags();
                // The residual autocorrelation pairs rows back too
                // (docs/PLAN.md task 146).
                if let Some(a) = inst.autocorr.as_deref_mut() {
                    a.iter_mut().for_each(EwAutoCorr::clear_lags);
                }
            }
        }
        if !plan.accept {
            continue;
        }
        let (i, ri, w) = (plan.i, plan.ri, plan.w);
        let (emit, learn) = (plan.emit, plan.learn);
        let sc = &mut *inst.scratch;
        sc.ys.clear();
        // The features go to the model as the chunk holds them: one
        // contiguous row, not a gather (docs/PERFORMANCE.md §20).
        let xs: &[f64] = if plan.direct() {
            sc.ys
                .extend(targets.iter().map(|t| Some(t[i]).filter(|f| usable(*f))));
            features.row(i)
        } else {
            // A row released from the `embargo` buffer: the same values,
            // stepped now instead of where they arrived (E47).
            let row = &released[plan.pending];
            sc.ys.extend_from_slice(&row.ys);
            &row.xs
        };
        let m_targets = sc.ys.len();
        // Each target's own weight, read before the step: the rows it was
        // present on, as `n_eff` is the rows the model saw (review 2026-09-12,
        // S2). `false` for a model that keeps only the shared one.
        let own_weights = inst.model.get().target_n_eff_into(&mut sc.tn);
        // The readiness statistics, read before the row like everything
        // else a row is gated on (docs/WARMUP-AND-CONVERGENCE.md §2): how
        // settled the instance is, and the noise gate's ratio per slot where
        // the model has one -- plus the row's own, when the field is asked
        // for. The gate and its warning read only which side of
        // `max_error_inflation` a slot is on, and the largest ratio when one
        // is at or above it, so a model may put a bound below the limit in
        // place of a ratio that costs it an `O(k³)` read (docs/PLAN.md task
        // 140); `summary` reads the exact one.
        let held = inst.pending_clock.as_deref().copied().unwrap_or(0.0);
        let settled = settled_frac(inst.decay, *inst.decay_time + held);
        // The two "cannot be met" notices below read the learned rows'
        // clock alone, as `Stream::readiness` does: the weight they
        // project a ceiling from has neither decayed by nor accumulated
        // the held rows, so paired with the row's fraction it read as
        // settled before anything was learned, and the ceiling came out
        // negative -- a floor the stream then met, after a notice that it
        // never would (review round 5, C1). The gate and the row's field
        // keep the fraction above, the held rows' clock counted (§8).
        let learned = settled_frac(inst.decay, *inst.decay_time);
        let max_infl = inst.spec.max_error_inflation_or_default();
        let has_infl = inst
            .model
            .get()
            .error_inflation_gate_into(&mut sc.infl, max_infl);
        let has_row_infl = inst.spec.emit_error_inflation
            && inst
                .model
                .get()
                .row_error_inflation_into(xs, &mut sc.row_infl);

        let mut step = if learn {
            // The row's stamp, for a window to key its snapshot by and
            // decide its edge from (task 175); decay reads `d_clock`.
            if let Some(stamp) = plan.stamp {
                inst.model.get_mut().stamp_next(stamp);
            }
            inst.model
                .get_mut()
                .step_sharded(xs, &sc.ys, plan.d_clock, w, inst.shards)
        } else {
            inst.model.get().predict(xs, &sc.ys, plan.d_clock)
        };
        let n_slots = step.pred.len();
        // `ew_cov` has no targets; its slots are statistics, so every one
        // maps to "target 0" for the warmup check. Slot `s` belongs to target
        // `s / nc`; the passes below walk the slots in groups of `nc` rather
        // than dividing per slot -- two integer divisions a slot were most of
        // a 230-slot `ew_cov` row (docs/PERFORMANCE.md §13).
        let nc = n_slots.checked_div(m_targets).unwrap_or(1).max(1);

        // The readiness gates (docs/WARMUP-AND-CONVERGENCE.md §2), each
        // withholding a prediction before it can reach the residual, sigma,
        // zscore, drift or selection, and each gating output, not learning
        // -- the model has already updated from this row. In precedence: the
        // settled fraction, for the whole instance; the per-target
        // `min_weight` (ENHANCEMENTS E7; the model itself predicts once the
        // *smallest* threshold is met), by each target's own weight where the
        // model keeps one, else the shared `n_eff` (review 2026-09-12, S2),
        // which is the user's explicit floor and so is named before the
        // noise gate, per slot, whose ratio is infinite until the model has
        // solved. The reason recorded is the first gate that withheld
        // anything.
        let mut reason = 0u8;
        let min_settled = inst.spec.min_settled_frac_or_default();
        if min_settled > 0.0 && settled.is_finite() && settled < min_settled {
            step.pred.fill(f64::NAN);
            reason = REASON_SETTLED;
        }
        // The first target the floor withholds, and the weight it read.
        let mut short: Option<(usize, f64)> = None;
        for (tj, group) in step.pred.chunks_mut(nc).enumerate() {
            let weight = match sc.tn.get(tj) {
                Some(&w) if own_weights => w,
                _ => step.n_eff,
            };
            if step_n_eff_below(weight, min_weight, tj) {
                group.fill(f64::NAN);
                short.get_or_insert((tj, weight));
                if reason == 0 {
                    reason = REASON_MIN_PERIODS;
                }
            }
        }
        // The floor cannot be met: the stream has all but settled, and the
        // weight the target reads tops out below it -- the ceiling
        // `1/(1 − λ^d)` at the rows' spacing and weight
        // (docs/WARMUP-AND-CONVERGENCE.md §5.1), which with a clock column
        // no spec can say in advance. Said once per instance, as the noise
        // gate's is, with the way out (docs/PLAN.md task 198, D8).
        if let Some((tj, weight)) = short
            && !inst.notified.unreachable
            && learned >= 0.95
        {
            let ceiling = settled_weight(weight, w, learned);
            let floor = min_weight.get(tj).copied().unwrap_or(f64::NAN);
            if ceiling < floor {
                inst.notified.unreachable = true;
                let target = inst
                    .spec
                    .targets
                    .get(tj)
                    .map_or_else(String::new, |t| format!(" for target {t:?}"));
                inst.notified.pending.push(format!(
                    "min_weight = {floor} cannot be met{target}: the stream is {:.0}% settled and \
                     the weight it reads tops out near {ceiling:.4}, the ceiling 1/(1 - lam^d) \
                     at its rows' spacing d and weight, so every prediction is withheld for \
                     good. Lower min_weight below {ceiling:.4}, or raise the half_life \
                     (docs/WARMUP-AND-CONVERGENCE.md).",
                    100.0 * learned
                ));
            }
        }
        // At or above the ratio: at equality the estimation variance is the
        // noise, and one observation's mean -- `edf = n_kish = 1`, exactly
        // `sqrt(2)` -- must not pass. A clean design opens where `k + 1`
        // did, the ridge keeping `edf` a hair under it.
        // `inf` is off: an infinite ratio -- the model unsolved -- is then
        // the model's own null, not this gate's.
        if has_infl && max_infl.is_finite() {
            for (slot, p) in step.pred.iter_mut().enumerate() {
                if sc.infl.get(slot).is_some_and(|&r| r >= max_infl) {
                    *p = f64::NAN;
                    if reason == 0 {
                        reason = REASON_INFLATION;
                    }
                }
            }
        }
        // The noise gate cannot be met: the stream has all but settled and
        // the ratio is still above the threshold, so nothing will change
        // it. Said once per instance, with the way out (§3).
        let unmet = (reason == REASON_INFLATION && !inst.notified.unreachable && learned >= 0.95)
            .then(|| Unmet {
                worst: sc.infl.iter().cloned().fold(0.0, f64::max),
                max: max_infl,
                settled: learned,
                weight: step.n_eff,
                row_weight: w,
            })
            .filter(|u| u.for_good(inst.spec));
        if let Some(unmet) = unmet {
            let worst = unmet.worst;
            inst.notified.unreachable = true;
            let figure = half_life_figure(inst.spec, inst.decay, &unmet);
            let k = inst.spec.k() + usize::from(inst.spec.fit_intercept);
            inst.notified.pending.push(if worst.is_finite() {
                format!(
                    "max_error_inflation = {max_infl:.3} cannot be met: the stream is {:.0}% \
                     settled and error_inflation is still {worst:.3}, so every prediction is \
                     withheld for good. Kish's effective sample size tops out near 2.9 \
                     half_lives of rows; {figure} so it can carry the {k} coefficients, or \
                     raise max_error_inflation to at least {worst:.3} to accept this much \
                     estimation noise (docs/WARMUP-AND-CONVERGENCE.md).",
                    100.0 * learned
                )
            } else {
                // Infinite: the model has not solved, its weight short of the
                // floor its first solve needs, which no ratio can loosen.
                format!(
                    "max_error_inflation = {max_infl:.3} cannot be met: the stream is {:.0}% \
                     settled and its weight tops out near {:.4}, below the {} the model needs \
                     before it solves (a row per coefficient, or min_weight), so \
                     error_inflation stays infinite and every prediction is withheld for good. \
                     {}{} so it can carry the {k} coefficients \
                     (docs/WARMUP-AND-CONVERGENCE.md).",
                    100.0 * learned,
                    settled_weight(step.n_eff, w, learned),
                    solve_floor(inst.spec),
                    figure[..1].to_uppercase(),
                    &figure[1..]
                )
            });
        }

        // Under `embargo` the residual diagnostics fold the prediction a
        // row was *scored* with, not the one the model gives it at the
        // replay: by then the model has learned every row released before
        // this one, so its prediction has seen the labels the delay says were
        // not yet available, and `sigma`, `zscore`, the quantiles, the
        // metrics, the conformal interval and the drift detector all
        // described a prediction nobody was shown (review 2026-09-12, C21).
        // The score keeps its prediction; the replay takes it back. The
        // model's own update above stands.
        //
        // With a conformal interval the record also keeps the radius each
        // slot's interval was shown with, after the predictions, so the
        // release scores the interval the row was shown rather than the one
        // the radius has reached since (C21's other half; docs/PLAN.md task
        // 112). A record without it -- a state saved before it was kept --
        // is scored against the radius at release, as it was.
        let mut shown_radius: Option<Vec<f64>> = None;
        if plan.buffered {
            let mut record = step.pred.clone();
            if let Some(cs) = inst.conformal.as_deref() {
                record.extend(cs.iter().map(|c| c.radius().unwrap_or(f64::NAN)));
            }
            inst.score_pred.push_back(record);
        } else if !plan.direct() {
            let n = step.pred.len();
            match inst.score_pred.pop_front() {
                Some(p) if p.len() == n => step.pred = p,
                Some(p) if p.len() == 2 * n && inst.conformal.is_some() => {
                    shown_radius = Some(p[n..].to_vec());
                    step.pred.copy_from_slice(&p[..n]);
                }
                // No record of the score -- a state saved before it was kept:
                // fold nothing rather than the prediction that peeks.
                _ => step.pred.fill(f64::NAN),
            }
        }

        // A model that predicts no target has no residual: its slots are
        // statistics, and `validate` refuses every residual-based option for
        // it, so `r`, `sig` and `zs` stay empty and every loop over them
        // below is a no-op. Tracking a residual for such a model was a
        // division per slot per row, and filling the three buffers a
        // `memset` per row, that no field ever read (docs/PERFORMANCE.md §13).
        sc.r.clear();
        sc.sig.clear();
        sc.zs.clear();
        let lam = inst.decay.factor(plan.d_clock);
        if inst.residuals {
            sc.sig.resize(n_slots, f64::NAN);
            sc.zs.resize(n_slots, f64::NAN);
            for (tj, group) in step.pred.chunks(nc).enumerate() {
                let yj = sc.ys.get(tj).copied().flatten();
                sc.r.extend(group.iter().map(|p| match yj {
                    Some(yj) if p.is_finite() => yj - p,
                    _ => f64::NAN,
                }));
            }

            // sigma is read from the state BEFORE this row's residual is
            // folded in, so `zscore` is out-of-sample like the prediction it
            // scales -- under a window, inside it, where the model reads its
            // fit (review 2026-09-12, S1).
            let ring = inst.resid_win.as_ref().map(SpreadRef::get);
            for (slot, &rv) in sc.r.iter().enumerate() {
                if inst.resid_w[slot] <= 0.0 {
                    continue;
                }
                let var = match ring {
                    Some(ring) => {
                        ring.inside(inst.decay, slot, inst.resid_w[slot], inst.resid_var[slot])
                    }
                    None => Some(inst.resid_var[slot]),
                };
                if let Some(var) = var {
                    let sd = var.max(0.0).sqrt();
                    sc.sig[slot] = sd;
                    if rv.is_finite() && sd > 0.0 {
                        sc.zs[slot] = rv / sd;
                    }
                }
            }
            if learn {
                // The spread as the row finds it, before its own residual,
                // keyed by the stamp the model's window was (task 175).
                if let Some(ring) = inst.resid_win.as_mut() {
                    ring.get_mut().learn(
                        plan.d_clock,
                        plan.stamp,
                        lam,
                        inst.resid_w.as_slice(),
                        inst.resid_var.as_slice(),
                    );
                }
                for (slot, &rv) in sc.r.iter().enumerate() {
                    if rv.is_finite() {
                        let w_new = lam * inst.resid_w[slot] + w;
                        if w_new > 0.0 {
                            inst.resid_var[slot] =
                                (lam * inst.resid_w[slot] * inst.resid_var[slot] + w * rv * rv)
                                    / w_new;
                            inst.resid_w[slot] = w_new;
                        }
                    } else {
                        inst.resid_w[slot] *= lam;
                    }
                }
            }
        }

        if emit {
            for (slot, &p) in step.pred.iter().enumerate() {
                inst.o_pred[slot * n_rows + ri] = p;
            }
            if inst.residuals {
                for (slot, &rv) in sc.r.iter().enumerate() {
                    inst.o_resid[slot * n_rows + ri] = rv;
                }
            }
            if !inst.o_sigma.is_empty() {
                for (slot, (&s, &z)) in sc.sig.iter().zip(sc.zs.iter()).enumerate() {
                    let at = slot * n_rows + ri;
                    inst.o_sigma[at] = s;
                    inst.o_resid_z[at] = z;
                }
            }
        }

        // Drift is monitored on |resid| scaled by the slot's own EW residual
        // std, so `drift_delta` means the same thing whatever the target's
        // units, and integrated over the row's clock (docs/PLAN.md task 146).
        // Rows with no residual are not scored, not treated as zero error.
        // Nor is a zero-weight row: the user weighted it out, `sigma` already
        // treats it as unseen, and a row that fired the detector could restart
        // the model under `drift_action = "reset"` (review 2026-09-12, S28).
        // Either still ages the detector's mean: it is clock.
        let mut row_drift = false;
        if let (true, Some(dets)) = (learn, inst.drift.as_deref_mut()) {
            for (slot, &rv) in sc.r.iter().enumerate() {
                let scale = sc.sig[slot];
                if w > 0.0 && rv.is_finite() && scale.is_finite() && scale > 0.0 {
                    let flag = dets[slot].update(rv.abs() / scale, plan.d_clock, lam);
                    // On the row being output: this one, or under an
                    // embargo the row releasing this one (task 160, PB1).
                    // Set, never cleared: one row may release several.
                    if flag && plan.drift_ri != usize::MAX {
                        inst.o_drift[slot * n_rows + plan.drift_ri] = true;
                    }
                    row_drift |= flag;
                } else {
                    dets[slot].age(lam);
                }
            }
            drift_seen |= row_drift;
        }

        // Residual diagnostics (ENHANCEMENTS E23), all read before the row's
        // own residual is folded in, like sigma.
        // The quantiles weigh each |resid| by the row's weight, as `sigma`
        // does, and decay on the clock at the model's rate (task 146).
        if let Some(ests) = inst.resid_q.as_deref_mut() {
            for (slot, est) in ests.iter_mut().enumerate() {
                if emit {
                    for li in 0..est.levels().len() {
                        inst.o_resid_q[li * block + slot * n_rows + ri] =
                            est.get(li).unwrap_or(f64::NAN);
                    }
                }
                if learn {
                    est.age(lam);
                    // A zero-weight row's residual stays out, as it does of
                    // `sigma` (S28).
                    if w > 0.0 && sc.r[slot].is_finite() {
                        est.add(sc.r[slot].abs(), w);
                    }
                }
            }
        }
        if let Some(ests) = inst.autocorr.as_deref_mut() {
            for (slot, est) in ests.iter_mut().enumerate() {
                if emit {
                    inst.o_autocorr[slot * n_rows + ri] = est.get().unwrap_or(f64::NAN);
                }
                // Likewise the autocorrelation (S28); a row it does not take
                // still ages it (task 146).
                if learn {
                    if w > 0.0 && sc.r[slot].is_finite() {
                        est.update(sc.r[slot], lam);
                    } else {
                        est.age(lam);
                    }
                }
            }
        }

        // Metrics are read before this row is scored, like every other
        // diagnostic here, so they never include the row they describe.
        if let Some(ms) = inst.metrics.as_deref_mut() {
            for (slot, met) in ms.iter_mut().enumerate() {
                if emit {
                    let at = slot * n_rows + ri;
                    inst.o_metrics[at] = met.ic().unwrap_or(f64::NAN);
                    inst.o_metrics[block + at] = met.r2().unwrap_or(f64::NAN);
                    inst.o_metrics[2 * block + at] = met.hit_rate().unwrap_or(f64::NAN);
                }
                if learn {
                    let yj = sc.ys.get(slot / nc).copied().flatten().unwrap_or(f64::NAN);
                    met.update_with(step.pred[slot], yj, lam, w, hit_test);
                }
            }
        }

        // The conformal interval is `pred ± q` with `q` read before the row,
        // then the row's own score moves `q` (ENHANCEMENTS E36). `sigma` is
        // the pre-row value too: it sets the step and the warm start.
        if let Some(cs) = inst.conformal.as_deref_mut() {
            for (slot, c) in cs.iter_mut().enumerate() {
                if emit {
                    let at = slot * n_rows + ri;
                    let p = step.pred[slot];
                    let q = c.radius().unwrap_or(f64::NAN);
                    inst.o_conformal[at] = p - q;
                    inst.o_conformal[block + at] = p + q;
                    inst.o_conformal[2 * block + at] = c.coverage().unwrap_or(f64::NAN);
                }
                if learn {
                    let shown = shown_radius.as_ref().map(|q| q[slot]);
                    c.update_against(sc.r[slot], sc.sig[slot], lam, w, shown);
                }
            }
        }

        if emit {
            inst.o_n_eff[ri] = step.n_eff;
            inst.o_settled[ri] = settled;
            inst.o_reason[ri] = reason;
            if has_row_infl {
                for (slot, r) in sc.row_infl.iter().enumerate().take(n_slots) {
                    inst.o_inflation[slot * n_rows + ri] = *r;
                }
            }
            if let Some(online_core::Extra::Lasso { lam_selected }) = &step.extra {
                for (t_i, l) in lam_selected.iter().enumerate() {
                    inst.o_lam[t_i * n_rows + ri] = *l;
                }
            }
        }
        // The decay time advances with the rows the model learns from, by
        // the delta it decayed by: a capped gap counts as the cap (§8). A
        // row held under `embargo` parks its delta until its release
        // replays it.
        if learn {
            *inst.decay_time += plan.d_clock;
        }
        if let Some(c) = inst.pending_clock.as_deref_mut() {
            if plan.buffered {
                *c += plan.d_clock;
            } else if !plan.direct() {
                *c -= plan.d_clock;
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
                .get()
                .coefficients()
                .map(|c| c.into_iter().flatten().collect());
            // Each coefficient's data share rides on the same cadence
            // (§2.2), and a coefficient more ridge than data is named once
            // per instance, here, where the shares are read anyway -- but
            // only on a row whose prediction the gates let through
            // (`reason == 0`). The first solve of any spec is
            // under-determined by construction, one row against `k` slopes,
            // and the warning cannot be retracted; judging a fit the model
            // is itself withholding as noise made the message false by the
            // row after it. Found verifying the shipped 0.9.0 wheel: an
            // ordinary two-feature fit warned at `n_eff = 1.00` and read
            // `support_coef = 1.00` from the next row to the end of the
            // stream.
            let support = inst.model.get().support_coef();
            if let (Some(s), false, 0) = (&support, inst.notified.support, reason) {
                let k_total = inst.spec.k() + usize::from(inst.spec.fit_intercept);
                let worst = s
                    .iter()
                    .flat_map(|slot| slot.iter().enumerate())
                    .filter(|(_, v)| v.is_finite())
                    .min_by(|a, b| a.1.total_cmp(b.1));
                if let Some((i, &share)) = worst
                    && share < 0.5
                {
                    let feature = inst
                        .spec
                        .features
                        .get(i % k_total - usize::from(inst.spec.fit_intercept))
                        .map_or("?", String::as_str);
                    inst.notified.support = true;
                    inst.notified.pending.push(format!(
                        "the coefficient of {feature:?} is {share:.2} data and {:.2} \
                             ridge (support_coef < 0.5): the design does not determine it \
                             -- a duplicated or constant column, or a ridge as large as the \
                             feature's variance. Predictions are unaffected in sample; the \
                             split among such columns is arbitrary and moves the moment the \
                             collinearity breaks (docs/WARMUP-AND-CONVERGENCE.md §2.2).",
                        1.0 - share
                    ));
                }
            }
            inst.o_support_coef[ri] = support.map(|s| s.into_iter().flatten().collect());
        }
        if reset_on_drift && row_drift {
            inst.reset();
        }
    }
    // Rows a sharded `marginal` held are learned before the run ends:
    // nothing outside a run reads a model with rows held.
    if matches!(inst.model, ModelRef::Learn(_)) {
        inst.model.get_mut().flush(inst.shards);
    }
    drift_seen
}
