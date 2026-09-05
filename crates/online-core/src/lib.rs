//! Pure-Rust online (streaming) regression models.
//!
//! This crate knows nothing about Polars, Python, or clocks-as-columns: it consumes
//! one row at a time (`&[f64]` features, `&[Option<f64>]` targets, a clock delta and a
//! weight) and produces a [`Step`]. All plumbing lives in `online-polars` / `online-py`.
//!
//! ```
//! use online_core::{Decay, EwRidge, EwRidgeCfg, OnlineModel};
//!
//! // y = 1 + 2x, fitted by exponentially weighted ridge with a 50-row halflife.
//! let mut model = EwRidge::new(EwRidgeCfg {
//!     n_features: 1,
//!     n_targets: 1,
//!     add_intercept: true,
//!     decay: Decay::Halflife(50.0),
//!     ridge: vec![1e-8],
//!     feature_sets: vec![],
//!     standardize: false,
//!     ridge_decay: false,
//!     coef0: None,
//!     session_shrink: None,
//!     long_halflife: None,
//!     min_periods: 5.0,
//!     solve_every: 0.0,
//!     max_rows_between_solves: 1,
//! })?;
//!
//! let mut pred = f64::NAN;
//! for i in 0..100 {
//!     let x = (i % 7) as f64;
//!     // `d_clock = 1.0` on every row is a row-count clock; `weight = 1.0`.
//!     let step = model.step(&[x], &[Some(1.0 + 2.0 * x)], 1.0, 1.0);
//!     // `pred` comes from the state *before* this row's target is learned
//!     // (out of sample by construction), and is NaN until `n_eff` -- the
//!     // accumulated weight before the row -- reaches `min_periods`.
//!     assert_eq!(step.pred[0].is_nan(), step.n_eff < 5.0);
//!     pred = step.pred[0];
//! }
//! assert!((pred - 3.0).abs() < 1e-6, "x = 99 % 7 = 1, so y = 3");
//! let coef = model.coefficients().unwrap();
//! assert!((coef[0][0] - 1.0).abs() < 1e-6 && (coef[0][1] - 2.0).abs() < 1e-6);
//!
//! // The state is a versioned, serializable value: save it, restore it later,
//! // and the restored model continues exactly where this one stopped.
//! let saved = model.state();
//! let mut restored = EwRidge::restore(&saved)?;
//! let next = (&[4.0], &[Some(9.0)]);
//! assert_eq!(restored.step(next.0, next.1, 1.0, 1.0), model.step(next.0, next.1, 1.0, 1.0));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The clock delta is the caller's job: [`ClockState`] turns raw clock values
//! into capped, session-aware deltas, and every model decays by them.
//! See `docs/PLAN.md` §2 and §4.

mod clock;
mod cluster;
mod conformal;
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
pub use cluster::{
    ClusterSummary, FeatureMoments, KMeans, KMeansCfg, LINK_FACTOR, LINK_FLOOR, LINK_QUANTILE,
    Micro, MicroCfg, MicroCluster, SeedRule, SplitMix64, dist2, merged_radius2,
};
pub use conformal::{Conformal, norm_ppf};
pub use drift::PageHinkley;
pub use ewcov::{EwCov, EwCovCfg, EwCovModel, EwCovStat, partial_corr, variance_is_usable};
pub use ewridge::{EwRidge, EwRidgeCfg};
pub use ftrl::{Ftrl, FtrlCfg, FtrlLoss};
pub use holt::{Holt, HoltCfg};
pub use kalman::{Kalman, KalmanCfg};
pub use lasso::{Lasso, LassoCfg};
pub use model::{
    Extra, INPUT_BOUND, ModelState, OnlineModel, State, StateError, Step, check_schema,
};
pub use pa::{Pa, PaCfg, PaMode};
pub use rls::{Rls, RlsCfg};
pub use robust::{Robust, RobustCfg, RobustLoss};
pub use sgd::{LearningRate, Sgd, SgdCfg, SgdLoss};
pub use solve::solve_spd;
pub use stats::{EwAutoCorr, P2Quantile, SlotMetrics};

/// Version of the serialized model-state layout.
///
/// Bump on any state layout change and keep a loader for the previous version
/// (`docs/PLAN.md`, hard rule 5). History:
///
/// - 1: initial layout.
/// - 2: `rls` stores the information factor `R` and `u = R^-T b` instead of
///   the covariance `P` (docs/IMPROVEMENTS.md C5). Schema-1 `rls` states are
///   converted on load; every other model is unchanged. `kmeans` and `micro`
///   (0.2.0) added `ModelState` variants without a bump: no existing layout moved,
///   and a 0.1 build meets a bank holding one as an unknown variant at
///   deserialization rather than as a version it refuses.
pub const SCHEMA_VERSION: u32 = 2;

/// Oldest state layout this build still loads.
pub const MIN_SCHEMA_VERSION: u32 = 1;
