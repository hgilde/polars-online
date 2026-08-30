//! The core contract every model implements (docs/PLAN.md §2).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::SCHEMA_VERSION;

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
/// When old versions gain migration paths they are handled here.
pub fn check_schema(state: &State) -> Result<(), StateError> {
    if state.schema_version != SCHEMA_VERSION {
        return Err(StateError::SchemaVersion {
            found: state.schema_version,
            current: SCHEMA_VERSION,
        });
    }
    Ok(())
}

/// One row in, one [`Step`] out (docs/PLAN.md §2).
///
/// Invariants:
/// - `pred` uses state *before* the update with this row (out-of-sample by
///   construction);
/// - deterministic given input order;
/// - no allocation in the hot path after warmup (buffers preallocated).
///
/// `x` excludes the intercept (the model adds it if configured); `y[j] = None`
/// means predict-only for target j; `d_clock` is already capped/gap-adjusted
/// (see [`crate::ClockState`]); `weight` scales the row.
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
        let bad = State {
            schema_version: SCHEMA_VERSION + 1,
            ..s
        };
        assert!(matches!(
            check_schema(&bad),
            Err(StateError::SchemaVersion { .. })
        ));
    }
}
