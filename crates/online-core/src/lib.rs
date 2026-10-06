//! Pure-Rust online (streaming) regression models.
//!
//! This crate knows nothing about Polars, Python, or clocks-as-columns: it consumes
//! one row at a time (`&[f64]` features, `&[Option<f64>]` targets, a clock delta and a
//! weight) and produces a [`Step`]. All plumbing lives in `online-polars` / `online-py`.
//!
//! ```
//! use online_core::{Decay, EwRidge, EwRidgeCfg, OnlineModel};
//!
//! // y = 1 + 2x, fitted by exponentially weighted ridge with a 50-row half_life.
//! let mut model = EwRidge::new(EwRidgeCfg {
//!     n_features: 1,
//!     n_targets: 1,
//!     fit_intercept: true,
//!     decay: Decay::Halflife(50.0),
//!     ridge: vec![1e-8],
//!     feature_sets: vec![],
//!     standardize: false,
//!     ridge_scale: false,
//!     coef_prior: None,
//!     session_shrink: None,
//!     long_half_life: None,
//!     min_weight: 5.0,
//!     solve_every: 0.0,
//!     solve_share: None,
//!     max_rows_between_solves: 1,
//!     gram_block_rows: 0,
//!     target_gaps: online_core::TargetGaps::OwnRows,
//!     window: None,
//!     window_every: None,
//!     max_rows_between_snapshots: None,
//! })?;
//!
//! let mut pred = f64::NAN;
//! for i in 0..100 {
//!     let x = (i % 7) as f64;
//!     // `d_clock = 1.0` on every row is a row-count clock; `weight = 1.0`.
//!     let step = model.step(&[x], &[Some(1.0 + 2.0 * x)], 1.0, 1.0);
//!     // `pred` comes from the state *before* this row's target is learned
//!     // (out of sample by construction), and is NaN until `n_eff` -- the
//!     // accumulated weight before the row -- reaches `min_weight`.
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
mod boundary;
mod clock;
mod cluster;
mod comp;
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
mod runs;
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
pub use marginal::{
    FeatureMomentLayout, Marginal, MarginalCfg, MarginalShard, Pair as MarginalPair, SerialRule,
    ShardRunner, Shards, run_in_order,
};
pub(crate) use marglag::{LagMoments, MarginalLags, PairMix, TargetLag};
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
pub use runs::Runs;
pub use seqtest::{SLOTS as SEQTEST_SLOTS, SeqTest, SeqTestCfg};
pub use sgd::{LearningRate, Sgd, SgdCfg, SgdLoss};
pub use solve::{SpdFactor, quad_forms_logdet, solve_spd};
pub use stats::{EW_QUANTILE_ALPHA, EwAutoCorr, EwQuantile, SlotMetrics};
pub use window::{
    Bytes, Cadence, Footprint, Moments, Snapshots, WindowBudget, WindowShadow, truncated,
    truncated_mean, truncated_scalar,
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
/// - 4: the spec every bank file carries gained `embargo`
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
///   row waiting under `embargo` was scored with (C21), skipped when
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
///   on, which the per-target `min_weight` gate and the quantile fit's
///   warm-up read, where both read the Gram's weight -- for the quantile the
///   band's, which a half-life caps. No loader for 9: see
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
///   `gap_cap`, needs no state (2026-09-20). A field removed from a
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
/// - 15: each stream keeps, per model instance, the clock its rows held under
///   `embargo` have covered (`StreamState::pending_clock` in
///   online-polars), where 14 rebuilt it at every chunk as a fresh sum. The
///   two round differently, so `settled_frac` depended on where a chunk
///   ended, against hard rule 3 (a property test found it, 2026-09-24). A
///   schema-14 file loads: its loader rebuilds the value as 14 did at a chunk
///   boundary.
/// - 16: every `EwCov`, and `marginal` for each of its pairs, keeps per slot
///   the value it has held since it last changed and the learned row that
///   started that run ([`Runs`]), counting its learned rows, and a window's
///   snapshot records the count, so a window can say exactly that a slot
///   held one value over it, where the subtraction left a remainder the
///   lasso standardized by (docs/PLAN.md task 94; by row index since the
///   review of 2026-09-25, where a decayed weight drifted). Every running
///   mean keeps what its double leaves out (`m_lo`, `mean_lo` and their
///   kin): the mean is a pair whose steps no rounding drops, where a plain
///   mean stopped short of a value held and left every variance centred on
///   it on the gap (task 101, 2026-09-24). And `kmeans` and `micro` keep,
///   in `FeatureMoments`, each feature's long-run reference for the
///   metric's floor, and `scale_floor` in their cfgs (task 102). A schema-14
///   or 15 file loads: the named encoding defaults all of it to empty or
///   zero, the runs start at the next learned row, a mean starts as the
///   double it was saved as, the references start at the next five rows,
///   and `scale_floor` is 0, the metric the state had.
/// - 17: the cross accumulator behind `ew_ridge` and `lasso` keeps each
///   target's own feature mean as a pair of its own, where it kept the
///   offset from the all-row mean and reconstructed the mean from two
///   level-sized numbers (review 2026-09-26, G3: a target absent on a row
///   whose features stood at the input bound left the fit infinite). A
///   schema-14, 15 or 16 file loads: the offsets are turned into means
///   once, at the precision they had, in the live accumulators and in
///   every window snapshot.
/// - 18: a stream under `embargo` with a conformal interval keeps, beside
///   each held row's score-time prediction, the radius every slot's interval
///   was shown with, and the release scores the row against it (docs/PLAN.md
///   task 112, C21's other half). The file's types do not change, what a
///   record holds does, so a 17 build would misread it and is refused. A
///   17 file loads: its records carry no radius, and those rows are scored
///   against the radius at release, as 17 scored them.
/// - 19: no model state changes; a bank file does. `on_clock_reset` keeps
///   `"error"`, now its default, and `"reset_state"`; `"max"` and `"zero"`
///   are gone, and `min_backwards_jump` is required with `"reset_state"`
///   and refused with `"error"` (docs/PLAN.md task 120, decided
///   2026-09-28). A bank file names every spec's policy, and every one
///   written before names one that no longer exists, so the bank loads none
///   older than 19 (`online_polars`' `MIN_BANK_SCHEMA_VERSION`); a model's
///   own state from 14 on still loads.
/// - 20: `ewridge`, `lasso`, `huber` and `quantile` keep the weight learned
///   since their last solve, and their configuration the share of the fit's
///   weight that makes a solve due, the default cadence under a finite
///   half-life (docs/PLAN.md task 115 (b), [`DEFAULT_SOLVE_SHARE`]); `pa`,
///   `sgd`, `ftrl` and `rls` keep each target's own weight, which its
///   `min_weight` reads, and `ftrl` each target's penalty scale, its weight
///   on the clock of the rows that teach it and the decay it is owed (task
///   115 (d)), a 19 file loading with the scale at 1; `corrchange` keeps a `sequential`
///   monitoring period, and its cfg `monitor_rows` and `boundary_gamma`
///   (task 114); a window's snapshot in `ewridge` and `lasso` keeps the
///   target moments (task 136), and in `marginal` the lag moments (task
///   137); `deco` keeps one weight per correlation value (task 115 (h)),
///   where a 19 file holds one for all, read as that weight on each; and a
///   `marginal`'s cfg names where its feature moments are kept (task 125).
///   The rest are `#[serde(default)]`, so a 19 file loads: the
///   counter starts at 0, the bank sets the share from the spec, each
///   target's weight starts at the shared one its gate read, and a
///   `corrchange` has no monitoring period. A 19 build would do none of
///   that, so a 20 file is refused there by its version.
/// - 21: the stream's diagnostics run on the clock (docs/PLAN.md task 146).
///   `resid_quantiles` and `ew_cov`'s `mahal_quantiles` keep an
///   exponentially weighted sketch ([`EwQuantile`]) where they kept P²
///   markers, one per slot for every level; a Page-Hinkley detector keeps
///   its mean's decaying weight and an excess integrated over the clock;
///   a residual autocorrelation keeps its pairs' weight. A 20 file's P²
///   markers say nothing about a decayed distribution and its excess is
///   in rows, so no loader is written. The stream's diagnostics live in the
///   bank file, which refuses one older than 21 (`online_polars`'
///   `MIN_BANK_SCHEMA_VERSION`); of the models' own states only an `ew_cov`
///   with `mahal_quantiles` changed, and it refuses one older than 21 by
///   name. Every other model's state from 14 on still loads.
/// - 22 (2026-10-02, task 144): the public names changed, and a bank file
///   stores its specs under them, so the bank refuses one older than 22;
///   the models' own states are unchanged, and still load from 14.
/// - 23 (2026-10-03, task 104): a row held under `embargo` carries its
///   number in the spec's window core and whether that core has resolved
///   its formula targets, and a bank file carries each such core with the
///   rows it holds. Both are additive with defaults, so a 22 file -- which
///   has no formula target -- loads as it was, and the bank's minimum stays
///   22.
/// - 24 (2026-10-03, review R4): the window core a bank file carries per
///   formula target is in the windows state's version 3 form (a row's raw
///   clock in its queues, the rows a resume skips), so a 23 file holding
///   one would fail at a group's first chunk; the bank refuses one older
///   than 24 by number (pre-1.0, no loader). The models' own states are
///   unchanged, and still load from 14.
/// - 25 (2026-10-03, review R6): the window core a bank file carries per
///   formula target is in the windows state's version 5 form (the last row
///   read, beside the rows held, as a sliced state's identity of its
///   input). Round five moved the windows version alone, to 4, so a 24
///   file holding a version-4 core failed at a group's first chunk; the
///   bank refuses one older than 25 by number, and a test pairs the two
///   numbers. The models' own states are unchanged, and still load from
///   14.
/// - 26 (2026-10-05, task 159): the window core a bank file carries per
///   formula target is in the windows state's version 6 form (a number
///   clock's raw value rides in a row's `off`, and a window's edge is
///   decided from it); the bank refuses one older than 26 by number. The
///   models' own states are unchanged, and still load from 14.
/// - 27 (2026-10-05, task 159): the windows state's version 7 form, which
///   carries the group and session columns' dtypes; the bank refuses one
///   older than 27 by number. The models' own states are unchanged.
/// - 28 (2026-10-06, task 161): `ew_cov`'s configuration carries
///   `pca_every` in clock units and `max_rows_between_pca`, and its state
///   the clock since the last refresh. The compact encoding is positional,
///   so an `ew_cov` state from before 28 does not decode; the bank refuses
///   a file older than 28 by number, and pre-1.0 no loader is written.
///   Every other model's state from 14 on still loads.
/// - 29 (2026-10-06, task 163): `micro`'s configuration carries
///   `prune_every` in clock units and `max_rows_between_prunes`, and its
///   state the clock since the last checkpoint; a `micro` state from before
///   29 does not decode, and the bank refuses a file older than 29 by
///   number. Every other model's state from 14 on still loads, `ew_cov`'s
///   from 28.
/// - 30 (2026-10-06, task 162): a window's snapshots are spaced on the clock
///   ([`Cadence`]): the ring keeps its clock spacing beside its row cap, and
///   the five windowed models' configurations carry `window_every` in clock
///   units and `max_rows_between_snapshots`, with `window` and
///   `window_every` written as nil where absent so that only the row cap,
///   the last, skips. A windowed state from before 30 does not decode
///   compactly; named, as a bank file is, it decodes with no spacing and
///   its rows, but its `window_every` counted rows, so the bank refuses a
///   file older than 30 by number, and pre-1.0 no loader is written.
pub const SCHEMA_VERSION: u32 = 30;

/// The default solve cadence of `ewridge`, `lasso`, `huber` and `quantile`
/// (docs/PLAN.md task 115 (b)): a solve once the weight learned since the last
/// reaches `ln 2 / 50` of the weight the fit holds, which is the share that
/// `half_life / 50` of clock brings in steady state.
pub const DEFAULT_SOLVE_SHARE: f64 = std::f64::consts::LN_2 / 50.0;

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

#[cfg(test)]
mod tests {
    /// The default solve share is what its doc says it is: the share of the
    /// steady-state weight that `half_life / 50` of clock brings. A row per
    /// clock unit at a half-life of 500 holds `1/(1 − 2^(−1/500))` in steady
    /// state, and ten rows bring ten of it; the two agree to `O(1/h)`.
    #[test]
    fn the_default_solve_share_is_a_fiftieth_of_a_half_life() {
        let h = 500.0_f64;
        let steady = 1.0 / (1.0 - (-1.0 / h).exp2());
        let share = (h / 50.0) / steady;
        let rel = (super::DEFAULT_SOLVE_SHARE - share).abs() / share;
        assert!(rel < 1e-3, "{} vs {share}", super::DEFAULT_SOLVE_SHARE);
    }
}
