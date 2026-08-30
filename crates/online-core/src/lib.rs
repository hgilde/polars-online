//! Pure-Rust online (streaming) regression models.
//!
//! This crate knows nothing about Polars, Python, or clocks-as-columns: it consumes
//! one row at a time (`&[f64]` features, `&[Option<f64>]` targets, a clock delta and a
//! weight) and produces a [`Step`]. All plumbing lives in `online-polars` / `online-py`.
//!
//! See `docs/PLAN.md` §2 and §4.

mod clock;
mod ewcov;
mod ewridge;
mod kalman;
mod lasso;
mod model;
mod rls;
mod solve;

pub use clock::{ClockAdvance, ClockCfg, ClockState, Decay, OnClockReset, SessionGap};
pub use ewcov::EwCov;
pub use ewridge::{EwRidge, EwRidgeCfg};
pub use kalman::{Kalman, KalmanCfg};
pub use lasso::{Lasso, LassoCfg};
pub use model::{Extra, ModelState, OnlineModel, State, StateError, Step, check_schema};
pub use rls::{Rls, RlsCfg};
pub use solve::solve_spd;

/// Version of the serialized model-state layout.
///
/// Bump on any state layout change and keep a loader for the previous version
/// (`docs/PLAN.md`, hard rule 5).
pub const SCHEMA_VERSION: u32 = 1;
