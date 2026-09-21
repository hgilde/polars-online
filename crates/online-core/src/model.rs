//! The core contract every model implements (docs/PLAN.md §2).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{MIN_SCHEMA_VERSION, SCHEMA_VERSION};

/// Output of one [`OnlineModel::step`].
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// Per output slot (targets, or targets x grid combos); NaN when not ready.
    pub pred: Vec<f64>,
    /// EW count of observations (with the model's decay).
    pub n_eff: f64,
    /// Model-specific extras.
    pub extra: Option<Extra>,
}

/// Model-specific step extras (docs/PLAN.md §4).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Extra {
    /// Lasso path: selected lambda per target (docs/PLAN.md §4.3).
    Lasso { lam_selected: Vec<f64> },
}

/// Versioned, serializable model state. The bank wraps this with its own header
/// (spec, package version) before writing msgpack (docs/PLAN.md §5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub schema_version: u32,
    pub model: ModelState,
}

impl State {
    pub fn new(model: ModelState) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            model,
        }
    }
}

/// One variant per model; grows as models land (tasks 4-14).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelState {
    EwCov(Box<crate::EwCov>),
    EwRidge(Box<crate::EwRidge>),
    Rls(Box<crate::Rls>),
    Lasso(Box<crate::Lasso>),
    Kalman(Box<crate::Kalman>),
    Robust(Box<crate::Robust>),
    Ftrl(Box<crate::Ftrl>),
    EwCovModel(Box<crate::EwCovModel>),
    Sgd(Box<crate::Sgd>),
    Pa(Box<crate::Pa>),
    Holt(Box<crate::Holt>),
    KMeans(Box<crate::KMeans>),
    Micro(Box<crate::Micro>),
    EwClass(Box<crate::EwClass>),
    SeqTest(Box<crate::SeqTest>),
    Marginal(Box<crate::Marginal>),
    Deco(Box<crate::Deco>),
    Rcov(Box<crate::Rcov>),
    Hmm(Box<crate::Hmm>),
    CorrChange(Box<crate::CorrChange>),
    Bocpd(Box<crate::Bocpd>),
}

#[derive(Debug, Error)]
pub enum StateError {
    #[error("state schema version {found} not supported (current: {current})")]
    SchemaVersion { found: u32, current: u32 },
    #[error("state is for a different model: expected {expected}, found {found}")]
    WrongModel {
        expected: &'static str,
        found: &'static str,
    },
    #[error("invalid state: {0}")]
    Invalid(String),
}

impl ModelState {
    pub fn kind(&self) -> &'static str {
        match self {
            // The bare accumulator, which the bank does not run; "ew_cov" is
            // the model, and a `WrongModel` must say which it found (review
            // 2026-09-12, S4).
            ModelState::EwCov(_) => "ew_cov_accumulator",
            ModelState::EwRidge(_) => "ew_ridge",
            ModelState::Rls(_) => "rls",
            ModelState::Lasso(_) => "lasso",
            ModelState::Kalman(_) => "kalman",
            ModelState::Robust(_) => "robust",
            ModelState::Ftrl(_) => "ftrl",
            ModelState::EwCovModel(_) => "ew_cov",
            ModelState::Sgd(_) => "sgd",
            ModelState::Pa(_) => "pa",
            ModelState::Holt(_) => "holt",
            ModelState::KMeans(_) => "kmeans",
            ModelState::Micro(_) => "micro",
            ModelState::EwClass(_) => "ew_class",
            ModelState::SeqTest(_) => "seqtest",
            ModelState::Marginal(_) => "marginal",
            ModelState::Deco(_) => "deco",
            ModelState::Rcov(_) => "rcov",
            ModelState::Hmm(_) => "hmm",
            ModelState::CorrChange(_) => "corrchange",
            ModelState::Bocpd(_) => "bocpd",
        }
    }
}

/// Check a state's schema version before dispatching to a model's `restore`.
///
/// Layout migrations do not live here: a model whose layout changed accepts
/// every version it can convert in its own `Deserialize` (see `rls`), so by
/// the time a `State` exists the migration has already happened. This gate
/// only rejects versions no model can convert.
pub fn check_schema(state: &State) -> Result<(), StateError> {
    if !(MIN_SCHEMA_VERSION..=SCHEMA_VERSION).contains(&state.schema_version) {
        return Err(StateError::SchemaVersion {
            found: state.schema_version,
            current: SCHEMA_VERSION,
        });
    }
    Ok(())
}

/// The largest magnitude of a feature, target or weight a model has to cope
/// with. The plumbing (`online-polars`) treats any value beyond it as missing,
/// like a null or a NaN, so a model never sees one.
///
/// Every model must keep a finite state, and go on learning, through any row
/// within the bound -- including a weight of `1e100` and a feature of `1e100`
/// on the same row -- and its predictions must return to a clean copy's once
/// such a row has decayed (`tests/model_contract.rs`, docs/IMPROVEMENTS.md
/// C2). The bound is where that is provable with `f64`: squares of `1e100`
/// still fit, products of a weight and a square (`1e300`) still fit.
pub const INPUT_BOUND: f64 = 1e100;

/// One row in, one [`Step`] out (docs/PLAN.md §2).
///
/// Invariants:
/// - `pred` uses state *before* the update with this row (out-of-sample by
///   construction);
/// - deterministic given input order;
/// - no allocation in the hot path after warmup (buffers preallocated);
/// - every input is finite and within [`INPUT_BOUND`]; the state stays finite
///   and the model keeps learning after any such row.
///
/// `x` excludes the intercept (the model adds it if configured); `y[j] = None`
/// means predict-only for target j; `d_clock` is already capped/gap-adjusted
/// (see [`crate::ClockState`]); `weight >= 0` scales the row, and `0` means
/// "advance the clock, learn nothing".
///
/// ```
/// use online_core::{Holt, HoltCfg, OnlineModel};
///
/// // Holt's linear trend has no features, so `x` is empty.
/// let mut model = Holt::new(HoltCfg {
///     n_targets: 1,
///     level_halflife: 2.0,
///     trend_halflife: 4.0,
///     min_periods: 2.0,
/// })?;
/// for t in 0..60 {
///     model.step(&[], &[Some(t as f64)], 1.0, 1.0);
/// }
///
/// // A missing target is predict-only: the forecast extrapolates one clock
/// // unit ahead and nothing is learned from the row.
/// let step = model.step(&[], &[None], 1.0, 1.0);
/// assert!((step.pred[0] - 60.0).abs() < 1e-3);
///
/// // A zero weight advances the clock and learns nothing, however wild the
/// // target: the coefficients (here `[level, trend]`) do not move.
/// let before = model.coefficients();
/// model.step(&[], &[Some(1e9)], 1.0, 0.0);
/// assert_eq!(model.coefficients(), before);
///
/// // `predict` is the step's answer without the step: the forecast three
/// // clock units past the last row. Neither row above was learned from, so
/// // `holt` extrapolates from its last observation (level 59, trend 1) over
/// // their two clock units as well, `59 + 5·1`. Nothing -- not even the
/// // clock -- has moved.
/// let ahead = model.predict(&[], 3.0);
/// assert!((ahead.pred[0] - 64.0).abs() < 1e-3);
/// assert_eq!(model.predict(&[], 3.0), ahead);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait OnlineModel: Sized {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step;
    /// The [`Step`] that [`Self::step`] would return for this row, without
    /// the update: the same `pred`, the same `n_eff`, the same `extra`, and
    /// the state untouched. `d_clock` is the clock elapsed since the last
    /// learned row -- a trend model (`holt`) extrapolates over it, on top of
    /// the clock since its target was last observed, and a proximal model
    /// (`ftrl`) sees its accumulators decayed by it; the
    /// coefficient models ignore it, since decay alone never moves a mean.
    ///
    /// `tests/model_contract.rs` holds every model to the equality
    /// `predict(x, d) == step(x, y, d, w)` on `pred`, `n_eff` and `extra`,
    /// row by row (docs/ENHANCEMENTS.md E31).
    fn predict(&self, x: &[f64], d_clock: f64) -> Step;

    /// [`Self::predict`] for a model that reads a number out of the
    /// **targets** slot rather than regressing it: `bocpd`'s hazard column
    /// and `hmm`'s exogenous column ride there, the way a label does, and
    /// `predict` alone cannot see them -- so it answered from the
    /// configured default and disagreed with the step on every row where
    /// the column differed from it (docs/REVIEW-E54-E64.md C1).
    ///
    /// The default ignores `y`, which is right for every model that only
    /// regresses its targets: `predict` is out-of-sample precisely because
    /// it does not read the row's answer.
    fn predict_with(&self, x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
        let _ = y;
        self.predict(x, d_clock)
    }

    /// Drop whatever this model keeps that is indexed by *rows back* -- a
    /// ring of past feature vectors, a partially filled window -- because
    /// the rows behind it are no longer adjacent to the next one
    /// (docs/PLAN.md task 47).
    ///
    /// The stream calls it on a **session change** and on a **capped clock
    /// gap** ([`crate::ClockAdvance::capped`]), and on nothing else: a reset
    /// already rebuilds the model, and an ordinary gap is what decay is for.
    /// A model with no row-lagged state does nothing, which is every model
    /// but the ones that keep one -- hence the default.
    ///
    /// It is **not** a reset: means, co-moments, coefficients and `n_eff`
    /// are untouched. Only the lags go.
    fn clear_lags(&mut self) {}

    /// Bound this model's window, if it has one ([`crate::WindowBudget`]).
    /// Configuration, not state: a caller sets it after building or
    /// restoring the model, and a model without a window ignores it --
    /// hence the default.
    fn set_window_budget(&mut self, _budget: Option<crate::WindowBudget>) {}

    /// What a refusing budget saw the window reach: its snapshots' bytes,
    /// and its spacing (`window_every`, doubled by any thinning). `None`
    /// while under budget, and for a model without a window.
    fn window_over_budget(&self) -> Option<(usize, usize)> {
        None
    }

    /// Each target's own accumulated weight, with `n_eff`'s meaning --
    /// before this row's update and before its own decay, and inside the
    /// window under one -- for the per-target `min_periods` gate: the stream
    /// checks each target's threshold against its own weight, where the
    /// shared `n_eff` is the feature side's, the same for every target
    /// (review 2026-09-12, S2). Clears and fills `out`, an entry a target,
    /// and says whether it did; a model that keeps no weight per target
    /// leaves every target on the shared `n_eff` -- hence the default.
    fn target_n_eff_into(&self, _out: &mut Vec<f64>) -> bool {
        false
    }

    /// Per output slot, how much estimation error is expected to inflate a
    /// prediction's error over the noise floor, read from the state before
    /// the row: `sqrt(1 + edf / n_kish)`, the effective degrees of freedom
    /// the last solve used over Kish's effective sample size behind the
    /// slot's Gram (docs/WARMUP-AND-CONVERGENCE.md §2.1). Infinite before
    /// the first solve, and where the Gram has no weight. Clears and fills
    /// `out`, an entry a slot, and says whether it did; a model with no
    /// linear fit to read it from -- `sgd`, `pa`, `ftrl`, and every model
    /// that is not a regression -- has none, hence the default, and the
    /// stream's `max_error_inflation` gate leaves such a model alone.
    fn error_inflation_into(&self, _out: &mut Vec<f64>) -> bool {
        false
    }

    /// The same for *this* row's features: `sqrt(1 + h(x))` per slot with
    /// `h(x) = x' Σ̂⁻¹ x / n_kish`, the leverage of the row against the
    /// factor the fit came from, so a row whose `x` leans on a direction the
    /// data never showed reads large where the stream average cannot see it.
    /// One triangular solve per slot, `O(k²)`, which is why it rides on an
    /// opt-in field (`emit_error_inflation`) and the model keeps the factor
    /// only when asked to. `false` where [`Self::error_inflation_into`] is.
    fn row_error_inflation_into(&self, _x: &[f64], _out: &mut Vec<f64>) -> bool {
        false
    }

    /// Per output slot, each coefficient's data share
    /// (docs/WARMUP-AND-CONVERGENCE.md §2.2): `1 − λ (Σ̂⁻¹)_jj`, the fraction
    /// of the coefficient the data determined rather than the ridge, in
    /// `[0, 1]`, laid out like the coefficients (`k_total` per slot). NaN in
    /// the intercept slot, which is not a share, and for a coefficient
    /// outside the slot's feature set; 0 for a column the standardiser
    /// dropped. `None` before the first solve, and for a model with no
    /// ridge system to read it from.
    fn support_coef(&self) -> Option<Vec<Vec<f64>>> {
        None
    }

    fn state(&self) -> State;
    fn restore(s: &State) -> Result<Self, StateError>;
    fn n_targets(&self) -> usize;
    fn n_features(&self) -> usize;
    /// Number of prediction slots (`n_targets * grid combos`; usually `n_targets`).
    fn n_outputs(&self) -> usize {
        self.n_targets()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each state names its own kind, so a `WrongModel` says which one it
    /// found: the bare accumulator answered "ew_cov", the name of the bank
    /// model it is not (review 2026-09-12, S4).
    #[test]
    fn the_bare_accumulator_is_not_named_as_the_ew_cov_model() {
        let bare = State::new(ModelState::EwCov(Box::new(crate::EwCov::new(1))));
        assert_ne!(bare.model.kind(), "ew_cov");
        match crate::EwCovModel::restore(&bare) {
            Err(StateError::WrongModel { expected, found }) => {
                assert_eq!(expected, "ew_cov");
                assert_ne!(found, expected, "the error must say what it found");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn schema_check() {
        let s = State::new(ModelState::EwCov(Box::new(crate::EwCov::new(1))));
        assert!(check_schema(&s).is_ok());
        let old = State {
            schema_version: MIN_SCHEMA_VERSION,
            ..s.clone()
        };
        assert!(check_schema(&old).is_ok());
        for v in [MIN_SCHEMA_VERSION - 1, SCHEMA_VERSION + 1] {
            let bad = State {
                schema_version: v,
                ..s.clone()
            };
            assert!(matches!(
                check_schema(&bad),
                Err(StateError::SchemaVersion { .. })
            ));
        }
    }
}
