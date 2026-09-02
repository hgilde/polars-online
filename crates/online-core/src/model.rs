//! The core contract every model implements (docs/PLAN.md §2).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{MIN_SCHEMA_VERSION, SCHEMA_VERSION};

/// Output of one [`OnlineModel::step`].
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// Per output slot (targets, or targets x grid combos); NaN when not ready.
    pub pred: Vec<f64>,
    /// Per output slot, only on `coef_every` rows / the last row of a chunk.
    pub coef: Option<Vec<Vec<f64>>>,
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
    EwCov(crate::EwCov),
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
            ModelState::EwCov(_) => "ew_cov",
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
///     min_periods: 3.0,
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
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait OnlineModel: Sized {
    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step;
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

    #[test]
    fn schema_check() {
        let s = State::new(ModelState::EwCov(crate::EwCov::new(1)));
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
