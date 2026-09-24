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
//!     coef_prior: None,
//!     session_shrink: None,
//!     long_halflife: None,
//!     min_periods: 5.0,
//!     solve_every: 0.0,
//!     max_rows_between_solves: 1,
//!     gram_block_rows: 0,
//!     target_gaps: online_core::TargetGaps::OwnRows,
//!     window: None,
//!     window_every: None,
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

mod bocpd;
mod clock;
mod cluster;
mod conformal;
mod constraint;
mod corrchange;
mod deco;
mod drift;
mod ewclass;
mod ewcov;
mod ewdiag;
mod ewlagcov;
mod ewridge;
mod ftrl;
mod gaps;
mod hmm;
mod holt;
pub mod humanfloat;
mod kalman;
mod lasso;
mod margbins;
mod marginal;
mod marglag;
mod model;
mod pa;
mod rcov;
mod rls;
mod robust;
mod seqtest;
mod sgd;
mod solve;
mod stats;
mod window;

pub use bocpd::{Bocpd, BocpdCfg, BocpdEmission};
pub use clock::{
    ClockAdvance, ClockCfg, ClockState, ClockValue, Decay, Disorder, OnClockReset, SessionGap,
    seconds_of_ns,
};
pub use cluster::{
    ClusterSummary, FeatureMoments, KMeans, KMeansCfg, LINK_FACTOR, LINK_FLOOR, LINK_QUANTILE,
    Micro, MicroCfg, MicroCluster, SeedRule, SplitMix64, dist2, merged_radius2,
};
pub use conformal::{Conformal, norm_ppf};
pub use constraint::Constraint;
pub use corrchange::{
    ChangeNorm, CorrChange, CorrChangeCfg, CorrChangeKind, kolmogorov_cdf, kolmogorov_quantile,
};
pub use deco::{Deco, DecoCfg, DecoDynamics};
pub use drift::PageHinkley;
pub use ewclass::{Covariance, EwClass, EwClassCfg};
pub use ewcov::{
    EwCov, EwCovCfg, EwCovModel, EwCovStat, Pca, TargetMoments, partial_corr, variance_is_usable,
};
pub use ewdiag::{EwDiag, Including};
pub use ewlagcov::EwLagCov;
pub use ewridge::{EwRidge, EwRidgeCfg};
pub use ftrl::{Ftrl, FtrlCfg, FtrlLoss};
pub use gaps::{GramPart, TargetGaps};
pub use hmm::{Hmm, HmmCfg};
pub use holt::{Holt, HoltCfg};
pub use kalman::{Kalman, KalmanCfg};
pub use lasso::{Lasso, LassoCfg};
pub use margbins::{BinCfg, BinRule};
pub(crate) use margbins::{MarginalBins, edges_from};
pub use marginal::{Marginal, MarginalCfg, Pair as MarginalPair, SerialRule};
pub(crate) use marglag::{MarginalLags, PairMix};
pub use model::{
    Extra, INPUT_BOUND, ModelState, OnlineModel, State, StateError, Step, check_schema,
};
pub use pa::{Pa, PaCfg, PaMode};
pub use rcov::{
    Rcov, RcovCfg, RcovEstimate, RcovKind, parzen, parzen_c_star, phi_11, phi_12, phi_22, preavg_g,
    psi1, psi2,
};
pub use rls::{Rls, RlsCfg};
pub use robust::{Robust, RobustCfg, RobustLoss};
pub use seqtest::{SLOTS as SEQTEST_SLOTS, SeqTest, SeqTestCfg};
pub use sgd::{LearningRate, Sgd, SgdCfg, SgdLoss};
pub use solve::{SpdFactor, quad_forms_logdet, solve_spd};
pub use stats::{EwAutoCorr, P2Quantile, SlotMetrics};
pub use window::{
    Footprint, Moments, Snapshots, WindowBudget, truncated, truncated_mean, truncated_scalar,
};

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
///   deserialization rather than as a version it refuses. `ew_class` and
///   `seqtest` (0.2.0) likewise.
/// - 3: `kalman` and `sgd` standardize with an [`EwDiag`] (means and
///   variances, O(k) a row) instead of a full [`EwCov`] they only ever read
///   the diagonal of (docs/PERFORMANCE.md §13). Schema-2 states of both are
///   converted on load by taking that diagonal, which is the same numbers;
///   every other model is unchanged.
/// - 4: the spec every bank file carries gained `label_delay`
///   (docs/ENHANCEMENTS.md E47), and a stream carries the rows it has
///   accepted but not yet learned from. Both are additive with defaults, so
///   a schema-3 file loads, continues to the bit and re-saves in the new
///   layout; the bump is because every spec's bytes moved, which is what
///   hard rule 5 asks to be told about. Nothing in a model's own state
///   changed. Task 38's `Sum w^2` and target moments rode into schema 3
///   without a bump: they are skipped when absent, so a schema-3 file's
///   bytes did not move.
/// - 5: the spec gained `group_close` (docs/ENHANCEMENTS.md E54), and a
///   stream that closes on session keeps the session value of the span it is
///   in. Additive with defaults again, so a schema-4 file loads, continues to
///   the bit and re-saves as 5; the bump is because every spec's bytes moved.
///   The rest of the E54-E64 batch rides on 5 without a further bump:
///   appended `ModelState` variants, `Option` fields that skip when absent,
///   and new `BankFile` map keys with defaults are all additive.
/// - 6: the `ew_cov` spec gained `window` and `window_every` (docs/PLAN.md
///   §13), and a windowed accumulator carries the ring of snapshots the
///   cutoff is computed from. The model's own fields skip when absent, so an
///   unwindowed *model* is unchanged, but spec fields serialize their nulls
///   like every other spec key, so every spec's bytes moved -- which is the
///   same reason 4 and 5 bumped. A schema-5 file loads, continues to the bit
///   and re-saves as 6. What an older build would do with a *windowed* file
///   is why this is a bump rather than a silent addition: it would ignore the
///   ring and report untruncated statistics.
///
///   The code review of 2026-09-12 added one field and rides on 6 without a
///   bump, as task 38's rode on 3: a stream's record of the prediction each
///   row waiting under `label_delay` was scored with (C21), skipped when
///   empty, so no file without a delay moves. An older build reading a file
///   that has it ignores the record and folds the replay's prediction,
///   which is what it always did.
/// - 7: three models moved to centred or clock-aware state (the code review
///   of 2026-09-12). `ew_ridge` keeps each target's cross-moments centred --
///   the target's mean, `E[(z − m_z)(y − ȳ)]` and the offset of `z`'s mean
///   over the target's rows from its mean over all of them -- where it kept
///   `E[z·y]` raw: in the live accumulators, the `session_shrink` twin and a
///   window's snapshots (N1). `bocpd`'s runs keep a Welford mean and scatter
///   where they kept `Σw·x` and `Σw·x x'` (C23). `holt` keeps each target's
///   clock since its last observation (C22). A schema-6 file loads: the raw
///   moments are split into centred ones as it is read (`EwRidgeWire`,
///   `RunWire`), which kept the numbers the file carried and not the bits,
///   and `holt`'s clock started at zero, the one thing a schema-6 file could
///   not say -- until 8 raised the minimum and the conversions went.
/// - 8: `ew_ridge` and `lasso` gained `target_gaps` (docs/PLAN.md task 81).
///   Their accumulators hold one Gram per set of targets present on the
///   same rows and the Gram each target reads, the weight over every row
///   moved into the cross-moments, and `lasso` keeps its cross-moments
///   centred, as `ew_ridge` does since 7 (the review's N2). A closed row's
///   Gram names its targets and carries their column means. No loader for
///   7: see [`MIN_SCHEMA_VERSION`].
/// - 9: `holt` keeps the weight its level and its trend have gathered per
///   target, which its weighted means need (the code review of 2026-09-12,
///   S29/S30), and `ftrl` the discounted sum of its proximal steps (C24). No
///   loader for 8: see [`MIN_SCHEMA_VERSION`].
/// - 10: `robust` keeps each target's observation weight beside its Gram's
///   (the second review of 2026-09-15, F1): the rows the target was present
///   on, which the per-target `min_periods` gate and the quantile fit's
///   warm-up read, where both read the Gram's weight -- for the quantile the
///   band's, which a halflife caps. No loader for 9: see
///   [`MIN_SCHEMA_VERSION`].
///
///   The ring a stream cuts a windowed spread with (the code review's S1)
///   rides on 9 without a bump, as C21's record rode on 6: skipped when
///   absent, so no file without one moves, and an older build reading one
///   ignores it and reports the whole history's spread, as it always did. A
///   file saved by an earlier build of 9 restarts the ring on load, so for
///   one window its spread counts only the rows after the load.
/// - 11: `robust` keeps each target's cross-moments centred, `c_j = E[(z −
///   m_j)(y − ȳ_j)]` beside `ȳ_j`, where it kept them raw, `E[z·y]`, and
///   solved the raw normal equations or centred them by subtraction -- the
///   `level²·ε` loss `ew_ridge` and `lasso` were cured of in 7 and 8, left
///   behind here (the review of 2026-09-18, S2). No loader for 10: see
///   [`MIN_SCHEMA_VERSION`].
/// - 12: the clock state drops the three fields the two 0.8.x disorder rules
///   kept -- the inferred session's start and the typical-step estimate with
///   its weight -- now that one rule, `min_backwards_jump` against
///   `max_dclock`, needs no state (2026-09-20). A field removed from a
///   positional layout is a layout change; no loader for 11.
/// - 13: the readiness statistics (docs/WARMUP-AND-CONVERGENCE.md, 2026-09-21).
///   `ew_ridge` keeps what its last solve left for them -- the effective
///   degrees of freedom and each coefficient's data share per slot, and,
///   when the per-row leverage is asked for, the systems it is read
///   against -- and whether it keeps them, ahead of its window; the stream
///   keeps the decay time each instance has seen (what `settled_frac` is
///   read from) and which of its readiness notices it has raised. No loader
///   for 12.
/// - 14: the clock state keeps its previous row's value in the form the
///   source had it -- a number, or a temporal clock's nanoseconds -- so the
///   gap between two instants is taken in integers and is exact whatever
///   the stream's age ([`ClockValue`], 2026-09-24). A temporal clock was a
///   double of seconds from a per-column origin the bank kept beside the
///   streams, which resolved 2^-52 of the time since it; the origin is gone
///   with it. No loader for 13.
pub const SCHEMA_VERSION: u32 = 14;

/// Oldest state layout this build still loads.
///
/// **14 since 2026-09-24**: a schema-13 clock state holds a double where the
/// nanoseconds now are, and pre-1.0 no loader is written for one.
///
/// **13 since 2026-09-21** (task 87): a schema-12 state holds none of the
/// readiness statistics -- the decay time each instance has seen, and what
/// `ew_ridge`'s last solve left for them -- and pre-1.0 no loader is written
/// for one.
///
/// **12 since 2026-09-20**, on the same rule as the entries below: a
/// schema-10 `robust` state's raw cross-moments could be centred on load,
/// but only by the subtraction the change exists to remove, and pre-1.0 no
/// loader is written for one.
///
/// **10 since 2026-09-15**, later the same day as 9: `robust`'s observation
/// weights cannot be recovered from a schema-9 state, and pre-1.0 no loader
/// is written for one.
///
/// **9 since 2026-09-15**, where it had been 8, on the same rule as the entry
/// below: `holt`'s weights and `ftrl`'s proximal sum cannot be recovered
/// from a schema-8 state, and pre-1.0 no loader is written for one.
///
/// **8 since 2026-09-14**, where it had been 6. The user, on `target_gaps`
/// (docs/PLAN.md task 81): "Do not worry about state saved before the next
/// version release, we are pre 1.0 and we can change things now". A
/// schema-6 or 7 file is refused by its version with the message
/// [`check_schema`] gives; the conversions 7 made for 6 -- `ew_ridge`'s raw
/// cross-moments, `bocpd`'s run sums -- went with their frozen fixtures
/// (`state_schema6.rs`, `state_schema6_ridge.rs`). The same exception to hard
/// rule 5 as the one below, for the same reason: pre-1.0, the layout is
/// worth more than the compatibility.
///
/// **6 from 2026-09-07**, where it had been 1 since the beginning. The
/// naming pass that day renamed six spec keys with no aliases, and a spec
/// denies unknown fields, so no file written before it can be read: a
/// schema-1..5 state names fields no builder has. Rejecting it on the
/// version, with the message [`check_schema`] gives, is kinder than failing
/// later on a field name the user never chose.
///
/// This is a deliberate exception to hard rule 5 ("keep a loader for the
/// previous version"), taken while the library is days old and pre-1.0
/// because getting the names right was judged worth more than the
/// compatibility. Schema 7's conversions were held to schema-6 fixtures
/// until 8 raised the minimum again.
pub const MIN_SCHEMA_VERSION: u32 = 14;
