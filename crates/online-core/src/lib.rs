//! Pure-Rust online (streaming) regression models.
//!
//! This crate knows nothing about Polars, Python, or clocks-as-columns: it consumes
//! one row at a time (`&[f64]` features, `&[Option<f64>]` targets, a clock delta and a
//! weight) and produces a [`Step`]. All plumbing lives in `online-polars` / `online-py`.
//!
//! See `docs/PLAN.md` §2 and §4.

mod clock;
mod drift;
mod ewcov;
mod ewridge;
mod ftrl;
mod holt;
mod kalman;
mod lasso;
mod model;
mod pa;
mod rls;
mod robust;
mod sgd;
mod solve;
mod stats;

pub use clock::{ClockAdvance, ClockCfg, ClockState, Decay, OnClockReset, SessionGap};
pub use drift::PageHinkley;
pub use ewcov::{EwCov, EwCovCfg, EwCovModel, EwCovStat, variance_is_usable};
pub use ewridge::{EwRidge, EwRidgeCfg};
pub use ftrl::{Ftrl, FtrlCfg, FtrlLoss};
pub use holt::{Holt, HoltCfg};
pub use kalman::{Kalman, KalmanCfg};
pub use lasso::{Lasso, LassoCfg};
pub use model::{Extra, ModelState, OnlineModel, State, StateError, Step, check_schema};
pub use pa::{Pa, PaCfg, PaMode};
pub use rls::{Rls, RlsCfg};
pub use robust::{Robust, RobustCfg, RobustLoss};
pub use sgd::{LearningRate, Sgd, SgdCfg, SgdLoss};
pub use solve::solve_spd;
pub use stats::{EwAutoCorr, P2Quantile, SlotMetrics};

/// Version of the serialized model-state layout.
///
/// Bump on any state layout change and keep a loader for the previous version
/// (`docs/PLAN.md`, hard rule 5).
pub const SCHEMA_VERSION: u32 = 1;
