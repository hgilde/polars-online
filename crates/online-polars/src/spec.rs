//! Model bank specs (docs/PLAN.md §3, §5): serde-deserializable from JSON
//! (Python) and TOML (CLI), with common-parameter validation and the output
//! struct layout.

use online_core::{ClockCfg, Decay, ExactCaps, TargetGaps};
use serde::{Deserialize, Serialize};

use crate::span::{Span, SpanList};
use crate::targets::Targets;

fn default_true() -> bool {
    true
}

mod checks;
pub(crate) mod diagnostics;
mod limits;
pub(crate) use checks::render_refusal;
use checks::*;
mod names;
mod readiness;

pub use names::*;
mod values;
pub use values::*;
mod clock;
pub use clock::*;
mod label_clock;
pub(crate) use clock::{ClockPolicy, clock_cfg_of};
pub use label_clock::*;

#[cfg(test)]
mod kinds_tests;

#[cfg(test)]
mod fill_tests;

#[cfg(test)]
mod num_label_tests;

/// A number rendered for a **field name** (`__r{ridge}`, `@h{half_life}`,
/// `abs_resid_q{level}`, `__p{penalty}`).
///
/// Rust's `Display` for `f64` never uses scientific notation, so a perfectly
/// legal `ridge = 1e-300` produced a **311-character field name** (three
/// hundred zeros). Values outside [1e-6, 1e7) render as compact scientific
/// (`1e-300`, `2.5e8`) instead; everything inside renders exactly as before,
/// so no existing name changes.
///
/// These strings are **public API** — users index the output struct by them —
/// and this function is deliberately the only place the rendering lives.
/// `tests/test_api_surface.py` pins the results, so a change here (or a change
/// in rustc's float formatting, which has happened historically) fails a test
/// instead of silently renaming users' columns.
pub fn num_label(v: f64) -> String {
    // `-0.0` is `0.0` (`==`), so one name: a ridge grid of the two was two
    // instances under `__r0` and `__r-0` (review 2026-10-06, PC10).
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if v.is_finite() && !(1e-6..1e7).contains(&a) {
        // `{:e}` gives `1e-300` / `2.5e8`; normalize the `e0` exponent Rust
        // emits for values that just crossed the threshold.
        let s = format!("{v:e}");
        match s.strip_suffix("e0") {
            Some(t) => t.to_string(),
            None => s,
        }
    } else {
        format!("{v}")
    }
}

/// The words a Huber scale takes, the default first (docs/PLAN.md task 271).
pub const HUBER_SCALES: [&str; 2] = ["mad", "rms"];

/// The MiB a spec's groups' Huber refit buffers may reserve by default
/// (docs/PLAN.md task 277): 256 at first, `window_budget`'s default, and
/// 320 since task 331 (the user's, 2026-11-05), when each held row's
/// one-pass `|e|` (8 bytes) and its node in the median's tree (task 323,
/// 40 bytes), 48 bytes a row in all, raised a target's reservation at 5
/// features and 500 rows from 100 KB to 124 KB: 256 MiB then refit 2,164
/// such targets where it had refit 2,684, and 320 refits 2,706.
pub const REFIT_BUDGET_MIB: f64 = 320.0;

/// Model choice + model-specific params (docs/PLAN.md §4).
///
/// A key no variant has is an error, not ignored: a spec is typed by hand in
/// TOML, and a misspelt parameter that silently kept its default would change
/// the model without a word.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
/// **State files written before 2026-09-07 do not load.** The naming pass
/// that day renamed six spec keys with no aliases, and a spec is
/// deserialized with unknown fields denied, so a file naming `coef0` or
/// `n_max` is refused rather than silently losing the field.
///
/// That is a deliberate exception to hard rule 5, taken while the library is
/// days old and pre-1.0, on the grounds that the names are worth more than
/// the compatibility: `MIN_SCHEMA_VERSION` became 6, the fixtures that
/// proved older files load went, and a state saved by 0.2.0 has to be refit.
/// It was taken again on 2026-09-14 (`MIN_SCHEMA_VERSION` 8, docs/PLAN.md
/// task 81): a state saved before `target_gaps` has to be refit too. And on
/// 2026-09-15 (`MIN_SCHEMA_VERSION` 9, the code review's S29/S30 and C24): a
/// state saved before `holt`'s weighted means and `ftrl`'s proximal sum.
pub enum ModelKind {
    /// `type = "ewridge"`, the builder's, the README's and the core's
    /// spelling; `"ew_ridge"` is refused naming it ([`RENAMED_VALUES`],
    /// docs/PLAN.md task 196, N1).
    #[serde(rename = "ewridge")]
    EwRidge {
        #[serde(default)]
        ridge: Option<FloatOrList>,
        #[serde(default)]
        feature_sets: Option<Vec<(String, Vec<String>)>>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        ridge_scale: RidgeScale,
        /// Shrink toward these coefficients instead of toward zero, one vector
        /// per target of length `n_features + intercept`, in original units.
        #[serde(default)]
        coef_prior: Option<Vec<Vec<f64>>>,
        /// On a session change, fit on this share of a slow-moving twin's
        /// moments, the rest today's, at today's weight: 0 keeps today's
        /// fit, 1 takes the long run's. Needs `long_half_life`.
        #[serde(default)]
        session_shrink: Option<f64>,
        /// Half-life of that twin; `"inf"` makes the long run the whole
        /// history (review 2026-09-12, S27).
        #[serde(default)]
        long_half_life: Option<Span>,
        #[serde(default)]
        solve_every: Option<Span>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        /// Rows of the Gram update held back and merged as one block
        /// (docs/ENHANCEMENTS.md E51): `0` or absent updates the `k×k`
        /// matrix on every row; `256` is the measured setting, `6.6×` faster
        /// at a thousand features. Needs a solve cadence (`solve_every > 0`
        /// and `max_rows_between_solves > 1`), does not combine with
        /// `window_size`, and the merged sum is not bit-identical to the per-row
        /// one. Chunk invariance holds either way.
        #[serde(default)]
        gram_block_rows: Option<usize>,
        /// Threads of the bank's pool the Gram's `k²` work may run on
        /// (docs/PLAN.md tasks 225 and 253): a block's merge under
        /// `gram_block_rows`, the per-row update otherwise. A count is
        /// capped at the pool's size, `"auto"` is the whole pool, and absent
        /// is one. The same bits at every count. Skipped when absent, so
        /// only a spec that sets it writes it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gram_threads: Option<GramThreads>,
        /// Which rows a target's fit is read from where the target is null
        /// on some (docs/PLAN.md task 81): `"own_rows"`, the default, fits it
        /// on exactly its rows; `"pairwise"` reads the Gram over every row
        /// and its cross-moments over its own.
        #[serde(default)]
        target_gaps: TargetGaps,
        /// Clock units of history the fit sees, with a **hard** cutoff: a row
        /// older than this is not in the Gram at all, where the exponential
        /// weight alone would leave `0.5^(age/half_life)` of it. Inside the
        /// window the weights are still exponential (docs/PLAN.md §13). A
        /// half-life grid is one instance per entry, each with its own ring.
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the snapshots the window is computed from, as
        /// `solve_every` is between solves: a number of the clock column's
        /// units, or a duration on a temporal clock, `0` every row
        /// (docs/PLAN.md task 162). With `max_rows_between_snapshots` too,
        /// whichever comes first; with neither, every row.
        #[serde(default)]
        snapshot_every: Option<Span>,
        /// At most this many rows between the window's snapshots, as
        /// `max_rows_between_solves` is between solves; `1` is every row,
        /// and `0`, no schedule, is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
    },
    Lasso {
        /// Decreasing penalties on standardized stats; required.
        penalties: Vec<f64>,
        /// 1.0 = lasso, < 1.0 = elastic net.
        #[serde(default)]
        l1_ratio: Option<f64>,
        /// Half-life of the EW squared error used to select lambda; `"inf"`
        /// selects on the plain mean over every row so far.
        #[serde(default)]
        select_half_life: Option<Span>,
        #[serde(default)]
        solve_every: Option<Span>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        #[serde(default)]
        max_iter: Option<u32>,
        #[serde(default)]
        tol: Option<f64>,
        /// Which rows a target's path is fitted from where the target is
        /// null on some, as for `EwRidge` (docs/PLAN.md task 81).
        #[serde(default)]
        target_gaps: TargetGaps,
        /// Clock units of history the path is fitted from, with a **hard**
        /// cutoff (docs/PLAN.md §13). The selection error follows the same
        /// window, so the chosen `lambda` fits the rows the model reports on.
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the window's snapshots, `0` every row, as
        /// for `EwRidge` (docs/PLAN.md task 162).
        #[serde(default)]
        snapshot_every: Option<Span>,
        /// At most this many rows between the window's snapshots; `1` is
        /// every row, and `0` is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
    },
    Kalman {
        /// Per-factor coefficient half-life (scalar or one per slot, intercept
        /// first). `inf` pins a coefficient. Note this is the COEFFICIENT
        /// half-life; the spec-level `half_life` drives the standardization and
        /// residual-variance statistics. Exactly one of it and `q` is given:
        /// it was required, and ignored beside a `q` (review 2026-10-06,
        /// PC6). Optional, so a state written when it was required reads.
        #[serde(default)]
        coef_half_life: Option<SpanList>,
        /// The process noise per slot, given outright rather than derived from
        /// `coef_half_life`, which is then refused -- as `_spec.py`'s
        /// `kalman` docstring says (review 2026-09-12, S22 and V16; review
        /// 2026-10-06, PC6).
        #[serde(default)]
        q: Option<Vec<Num>>,
        #[serde(default)]
        obs_var: Option<f64>,
        #[serde(default)]
        p0: Option<f64>,
        #[serde(default)]
        share_p: bool,
        /// Per-slot reversion half-life (scalar or one per slot, intercept
        /// first): the coefficient mean shrinks toward zero by `2^(-d/r_i)`
        /// per row. `inf` (the default) is the random walk
        /// (docs/ENHANCEMENTS.md E41).
        #[serde(default)]
        revert_half_life: Option<SpanList>,
        /// Standardize features internally (default true). Off makes the filter
        /// a plain Bayesian linear regression on the features' own scale.
        #[serde(default = "default_true")]
        standardize: bool,
    },
    /// Huber regression (docs/PLAN.md §4.5).
    Huber {
        /// Cut point in units of the residual scale. Default 1.345, the
        /// constant of 95% efficiency at the normal (Huber 1981), filled
        /// where the model is built ([`crate::build_models`]); `"inf"` cuts
        /// nothing, which is least squares.
        #[serde(default)]
        huber_delta: Option<Num>,
        /// The scale the cut is drawn in, per target: `"mad"` (the
        /// default), the EW median of `|e|` over `Φ⁻¹(3/4)`, or `"rms"`,
        /// the EW root mean square, which one wild row sets
        /// ([`online_core::HuberScale`]; docs/PLAN.md task 271). Filled
        /// where the model is built. Skipped when absent, so a spec written
        /// before it serializes as it did.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        huber_scale: Option<String>,
        /// The scale's own half-life, in clock units; default the spec's
        /// `half_life` (or `lam`). `"inf"` forgets no residual.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        huber_scale_half_life: Option<Span>,
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        solve_every: Option<Span>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        /// Named subsets of `features`, each a fit of its own from the
        /// target's sums, as `EwRidge`'s are; a row's Huber weight is read
        /// from the largest set's residual (docs/PLAN.md task 272 (a)).
        /// Skipped when absent, so a spec written before it serializes as
        /// it did.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        feature_sets: Option<Vec<(String, Vec<String>)>>,
        /// `"mean"` (the default), the ridge on the means, or `"sum"`,
        /// RLS's decaying prior on the sum scale, as `EwRidge`'s: at
        /// `huber_delta = inf` the fit is `rls`'s (docs/PLAN.md task 272
        /// (a)). Skipped at its default, so a spec written before it
        /// serializes as it did.
        #[serde(default, skip_serializing_if = "RidgeScale::is_mean")]
        ridge_scale: RidgeScale,
        /// Emit `huber_scale_<t>`: each target's residual scale as the row's
        /// cut reads it, before the row's own residual (docs/PLAN.md task
        /// 272 (a)). Skipped when off, so a spec written before it
        /// serializes as it did.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        emit_huber_scale: bool,
        /// Each target's first rows refit to convergence, once
        /// (`RobustCfg::refit_rows`, docs/PLAN.md task 277): absent is
        /// `default_refit_rows(k)`, 0 is off. Skipped when absent, so a spec
        /// written before it serializes as it did.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        huber_refit_rows: Option<u32>,
        /// The MiB the refit buffers of all this spec's groups may reserve
        /// (`Bank`, docs/PLAN.md task 277): absent is 256, `inf` no bound.
        /// Skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refit_budget: Option<Num>,
    },
    /// Quantile regression at level `quantile` (docs/PLAN.md §4.5).
    Quantile {
        quantile: f64,
        #[serde(default)]
        ridge: Option<f64>,
        #[serde(default)]
        standardize: bool,
        #[serde(default)]
        solve_every: Option<Span>,
        #[serde(default)]
        max_rows_between_solves: Option<u32>,
        /// Half-width of the band the fit takes its Newton step in, in units
        /// of the EW residual std; default 0.2 (review 2026-09-12, N9), and
        /// never narrower than `(k/n)^(2/5)` for the target's effective sample
        /// `n` (the second review's F3). A band holding under one row per
        /// coefficient takes least-squares rows until it holds rows again,
        /// which rebuilds a fit a row at the input bound left behind
        /// (`online_core::robust`'s module docs).
        #[serde(default)]
        quantile_eps: Option<f64>,
        /// Each target's first rows refit to the converged smoothed
        /// check-loss fit, once (`RobustCfg::refit_rows`, docs/PLAN.md task
        /// 278, D-Q2): absent is `default_refit_rows(k)`, 0 is off. Skipped
        /// when absent, so a spec written before it serializes as it did.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        quantile_refit_rows: Option<u32>,
        /// `huber`'s `refit_budget`, for the quantile refit's buffers.
        /// Skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refit_budget: Option<Num>,
    },
    /// Online logistic regression via FTRL-proximal (docs/PLAN.md §4.6).
    /// `pred` is a probability; `resid = y - p`.
    Ftrl {
        #[serde(default)]
        alpha: Option<f64>,
        #[serde(default)]
        beta: Option<f64>,
        #[serde(default)]
        l1: Option<f64>,
        #[serde(default)]
        l2: Option<f64>,
        /// Refuse a chunk whose target is not 0 or 1, naming the row, rather
        /// than clamp it into [0, 1] (review 2026-09-12, S31). Logistic loss
        /// only.
        #[serde(default)]
        strict_binary: bool,
        /// "logistic" (default, binary targets, `pred` is a probability) or
        /// "squared" (continuous targets, sparse linear regression).
        #[serde(default)]
        loss: Option<String>,
    },
    /// EW moments of the feature columns, no regression (docs/PLAN.md §4.7).
    /// `targets` is ignored; every column of interest goes in `features`.
    EwCov {
        /// Any of "mean", "var", "std", "cov", "corr", "partial_corr", "mahal",
        /// "lag_corr" (the last with `lags`).
        /// Default: mean + std + corr.
        #[serde(default)]
        stats: Option<Vec<String>>,
        /// Polars' `bias`, with its default `false`: every second moment is
        /// multiplied by `W² / (W² − Σw²)`, the correction Polars'
        /// `ewm_var(bias=False)` applies; `true` reports the weighted means
        /// as kept ([`online_core::EwCovCfg::bias`]).
        #[serde(default)]
        bias: bool,
        /// Prior for the precision matrix, required by "partial_corr" and "mahal".
        #[serde(default)]
        precision_prior: Option<f64>,
        /// Quantile levels of the past Mahalanobis scores, exponentially
        /// weighted at the model's half-life, one `mahal_q<p>` field each;
        /// needs "mahal" in `stats` (E37, docs/PLAN.md task 146).
        #[serde(default)]
        mahal_quantiles: Option<Vec<f64>>,
        /// Principal components to track, each `pc<j>_var`, `pc<j>_share`,
        /// `pc<j>_loading_<feature>` per feature and `pc<j>_score` (E38).
        #[serde(default)]
        pca: Option<usize>,
        /// Clock units between refreshes of the components, as `solve_every`
        /// is the regressions': a number of the clock column's units, or a
        /// duration on a temporal clock, `0` every row (docs/PLAN.md task
        /// 161). With `max_rows_between_pcas` too, whichever comes first;
        /// with neither, every row.
        #[serde(default)]
        pca_every: Option<Span>,
        /// At most this many rows between refreshes, as
        /// `max_rows_between_solves` is the regressions'; `1` is every row,
        /// and `0` is refused (tasks 161 and 196).
        #[serde(default)]
        max_rows_between_pcas: Option<u32>,
        /// Lags to accumulate cross-moments at, in output order
        /// (docs/ENHANCEMENTS.md E56): strictly increasing, `>= 1` and at
        /// most 2^20 (`online_core::MAX_LAG`, the ring being sized before the
        /// first row), counted in *learned rows within the group*. Read from `gram()`
        /// as `lags`/`lag_comoments`, or emitted as `lag_corr_l<l>_<a>_<b>`
        /// by adding `"lag_corr"` to `stats`.
        #[serde(default)]
        lags: Option<Vec<usize>>,
        /// Threads of the bank's pool the per-row update of the `k × k`
        /// co-moments may run on (docs/PLAN.md tasks 225 and 253), as
        /// `EwRidge`'s: the same bits at every count, absent is one,
        /// skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gram_threads: Option<GramThreads>,
        /// Clock units of history the statistics see, with a **hard** cutoff:
        /// a row older than this contributes nothing, where the exponential
        /// weight alone would leave `0.5^(age/half_life)` of it. Inside the
        /// window the weights are still exponential -- it is not a flat
        /// window (docs/PLAN.md §13).
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the snapshots the window is computed from, as
        /// for `EwRidge`, `0` every row (docs/PLAN.md task 162). Every row,
        /// the default, is the tightest boundary; a coarser cadence divides
        /// the memory and shortens the effective window by at most one
        /// spacing -- never lengthens it.
        #[serde(default)]
        snapshot_every: Option<Span>,
        /// At most this many rows between the window's snapshots; `1` is
        /// every row, and `0` is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
    },
    /// Stochastic gradient descent with pluggable losses (ENHANCEMENTS E16).
    /// O(k) per row, no solves, and the only model here that takes count
    /// targets (via `loss = "poisson"`).
    Sgd {
        /// "squared" (default), "huber", "quantile", "epsilon_insensitive",
        /// "poisson" or "logistic".
        #[serde(default)]
        loss: Option<String>,
        /// Huber cut, in units of the target's residual scale, as
        /// `huber`'s; default 1.345. `"inf"` clips nothing, which is the
        /// squared loss. The residual's std, where `eps` is in the target's
        /// own: a clipped gradient keeps learning while early residuals
        /// widen the cut, and a zero one inside a band does not
        /// (docs/PLAN.md task 202).
        #[serde(default)]
        huber_delta: Option<Num>,
        /// The scale the Huber cut is drawn in, `huber`'s: `"mad"` (the
        /// default under `loss = "huber"`) or `"rms"`. Refused under any
        /// other loss.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        huber_scale: Option<String>,
        /// The Huber scale's own half-life, `huber`'s; default the spec's
        /// `half_life`. Refused under any other loss.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        huber_scale_half_life: Option<Span>,
        /// `huber`'s refit of each target's first rows (docs/PLAN.md task
        /// 277): absent is `default_refit_rows(k)` under `loss = "huber"`, 0
        /// is off. Refused under any other loss. Skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        huber_refit_rows: Option<u32>,
        /// `huber`'s `refit_budget`, under `loss = "huber"`; refused under
        /// any other loss. Skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refit_budget: Option<Num>,
        #[serde(default)]
        quantile: Option<f64>,
        /// Half-width of the insensitive tube, in units of the target's
        /// own EW std, the spread of `y` around its EW mean (docs/PLAN.md
        /// task 202). Default 0.01: errors under 1% of the target's own
        /// spread do not move the fit. In units of the noise the tube is
        /// `eps / √(1 − R²)` wide, `R²` the fit's: 0.067 noise stds at R²
        /// 0.978, 2.2 at 0.99998, where a tube of 0.1, 22 noise stds, held
        /// the fit wherever it first landed inside it (out-of-sample error
        /// 24 noise variances above the noise against 0.20, `inv_scaling`;
        /// 0.010 against 0.015 at R² 0.978). Task 207 kept 0.01.
        #[serde(default)]
        eps: Option<f64>,
        #[serde(default)]
        learning_rate: Option<f64>,
        /// "constant" (default), "inv_scaling" or "adagrad".
        #[serde(default)]
        schedule: Option<String>,
        /// Exponent for `inv_scaling`.
        #[serde(default)]
        power: Option<f64>,
        #[serde(default)]
        l2: Option<f64>,
        #[serde(default)]
        /// `Num`, not `f64`: the documented way to disable clipping is
        /// `inf`, and JSON cannot carry Infinity as a number -- `Num` accepts
        /// the string "inf", exactly as the half-life fields do.
        clip_gradient: Option<Num>,
        /// Standardize features against their running moments before the
        /// gradient step, unscaling the coefficients on the way out.
        /// Default true, as `kalman`'s: a gradient step's one learning rate
        /// has to suit every feature (docs/PLAN.md task 195, U2).
        #[serde(default = "default_true")]
        standardize: bool,
        /// Lower bound per slope (ENHANCEMENTS E40): one number for every
        /// feature or a list with one entry per feature; "-inf" for none.
        /// The intercept is never bounded. Imposed by Euclidean projection
        /// after each update, in the space the step is taken in.
        #[serde(default)]
        coef_min: Option<FloatOrList>,
        /// Upper bound per slope, as `coef_min`; "inf" for none.
        #[serde(default)]
        coef_max: Option<FloatOrList>,
        /// The slopes sum to this, in the caller's units. With `coef_min =
        /// 0` and `coef_sum = 1` the slopes are weights on the simplex.
        #[serde(default)]
        coef_sum: Option<f64>,
        /// Refuse a chunk whose target is not 0 or 1, naming the row, rather
        /// than clamp it into [0, 1]: `ftrl`'s rule. Logistic loss only
        /// (docs/PLAN.md task 195, S4).
        #[serde(default)]
        strict_binary: bool,
    },
    /// Passive-aggressive regression (ENHANCEMENTS E17). No learning rate:
    /// each row's update is the smallest change that satisfies it.
    Pa {
        /// "pa1" (default), "pa" (unbounded) or "pa2" (damped).
        #[serde(default)]
        mode: Option<String>,
        /// Aggressiveness: under "pa1" a cap on the step, in the target's
        /// units over `‖z‖²`'s, so the same `c` binds by the target's scale
        /// (review round 5, G5); under "pa2" the damping `1/(2c)` beside
        /// `‖z‖²`, free of the target's units. Default 1. Refused under
        /// "pa", which reads none; `"inf"` caps nothing, so either bounded
        /// mode is "pa".
        #[serde(default)]
        c: Option<Num>,
        /// Insensitive tube, in units of the target's own EW std, the
        /// spread of `y` around its EW mean: rows already this close leave
        /// the fit alone (docs/PLAN.md task 202). Default 0.01: errors under
        /// 1% of the target's own spread do not move the fit. In units of
        /// the noise it is `eps / √(1 − R²)` wide, `R²` the fit's: 0.067
        /// noise stds at R² 0.978, where a binding cap damps and a wider
        /// tube would too (out-of-sample error 1.15 noise variances above
        /// the noise at the defaults, 0.33 at `c = 0.1`, 0.57 at `eps =
        /// 0.1`, on a target of spread about 2); 2.2 at 0.99998, where a
        /// tube of 0.1, 22 noise stds, holds the fit wherever it first lands
        /// inside it (0.14 at the defaults, 65 at `eps = 0.1`). Task 207
        /// kept 0.01.
        #[serde(default)]
        eps: Option<f64>,
        /// Bounds and sum on the slopes, as for `sgd` (ENHANCEMENTS E40).
        #[serde(default)]
        coef_min: Option<FloatOrList>,
        #[serde(default)]
        coef_max: Option<FloatOrList>,
        #[serde(default)]
        coef_sum: Option<f64>,
        /// Standardize features against `sgd`'s running moments. Default
        /// true (docs/PLAN.md task 195, U2).
        #[serde(default = "default_true")]
        standardize: bool,
    },
    /// Holt's linear trend method (ENHANCEMENTS E25): level plus slope, no
    /// features. The baseline a feature-based model should have to beat.
    /// The level's half-life is the spec's own `half_life` (or `lam`),
    /// `"inf"` included: no forgetting. It had a second name,
    /// `level_half_life`, which built a second dict shape and is refused
    /// naming `half_life` ([`RENAMED`]; review 2026-10-06, TA6; docs/PLAN.md
    /// task 196, N16).
    Holt {
        /// Half-life of the trend; `"inf"` forgets no slope, so the trend is
        /// the whole history's drift. Defaults to four times the level's
        /// half-life.
        #[serde(default)]
        trend_half_life: Option<Span>,
        /// `false` fits the level alone: the trend is held at zero and the
        /// forecast is flat, simple exponential smoothing (docs/PLAN.md task
        /// 115, S30). `trend_half_life` is refused beside it. Skipped when
        /// absent: the trend is on.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trend: Option<bool>,
    },
    Rls {
        /// Prior strength: `A0 = delta I`, i.e. `P0 = I / delta`, the
        /// classic RLS name; `ew_ridge`'s `ridge` under `ridge_scale =
        /// "sum"`, not its `ridge` (docs/PLAN.md task 195, N11). Default 1.
        /// Scalar only (baked into the state).
        #[serde(default)]
        delta: Option<f64>,
        #[serde(default)]
        coef_prior: Option<Vec<Vec<f64>>>,
    },
    /// Exponentially weighted k-means (docs/CLUSTERING.md §6.2; PLAN §11a,
    /// task 23). No targets: every column of interest goes in `features`, and
    /// the outputs are the nearest centre's index and two distances, read
    /// before the row is learned.
    #[serde(rename = "kmeans")]
    KMeans {
        /// Number of clusters, `1 ..= 2^16` (`online_core::KMeans::MAX_K`):
        /// `n_clusters`, scikit-learn's `KMeans` name (task 244, P-9), kept
        /// `k` here as the core's configuration calls it.
        #[serde(rename = "n_clusters")]
        k: usize,
        /// Learned rows buffered before seeding, at least `k`: fewer is
        /// refused by name, as `hmm` refuses it, where it was floored to `k`
        /// in silence (review 2026-10-06, PC8). Default 500, or `k` where
        /// that is more. The buffer is held to 256 MiB
        /// (`online_core::KMeans::WARM_BUDGET_MIB`).
        #[serde(default)]
        warm_rows: Option<usize>,
        /// "lloyd" (default), "k-means++", "farthest" or "first": how the
        /// first centres are placed, scikit-learn's `init` and its spelling
        /// of k-means++ (task 244).
        #[serde(default)]
        init: Option<String>,
        /// Seed of the generator behind "k-means++" and "lloyd". Default 0.
        #[serde(default)]
        seed: Option<u64>,
        /// Learned rows between centre updates. Default 1 (every row).
        #[serde(default)]
        update_every_rows: Option<u32>,
        /// Merge the two closest clusters when their centres are closer than
        /// this many summed radii, re-placing the freed centre at the
        /// farthest row seen. Default 0.5; `0` disables split–merge.
        #[serde(default)]
        split_merge: Option<f64>,
        /// Learned rows between split–merge checks. Default 100.
        #[serde(default)]
        split_merge_every_rows: Option<u32>,
        /// A cluster lighter than `dead_frac · n_eff / k` at a check is
        /// re-placed. Default 0.05; `0` disables the dead rule; below 0.5
        /// (`limits.rs`). The rule runs
        /// at a split–merge check, so under `split_merge = 0` the default is
        /// 0 and a value above it is refused (review 2026-10-06, CF6).
        #[serde(default)]
        dead_frac: Option<f64>,
        /// Measure distances in units of each feature's EW standard
        /// deviation. Default true.
        #[serde(default)]
        standardize: Option<bool>,
        /// Floor the metric's variance at this fraction of the feature's
        /// long-run (undecayed) variance. Default 0.1; `0` is the EW
        /// variance alone.
        #[serde(default)]
        scale_floor: Option<f64>,
    },
    /// DenStream-style micro-clusters with a linkage macro step
    /// (docs/CLUSTERING.md §6.5; PLAN §11a, task 24). No targets: the
    /// outputs are the nearest cluster's label and distance, the
    /// micro-cluster id the row goes to, an outlier flag and two counts,
    /// read before the row is learned; the potential summaries ride in
    /// `coef`, one row each.
    #[serde(rename = "micro")]
    Micro {
        /// Bound on a summary's RMS radius per standardized coordinate:
        /// `eps √p` in the metric. Required, finite, `> 0`.
        eps: f64,
        /// Weight at which a summary becomes potential (DenStream's βµ).
        /// Default 3.
        #[serde(default)]
        beta_mu: Option<f64>,
        /// Cap on live summaries; at the cap the lightest outlier summary is
        /// evicted, else the lightest potential one. Default 200.
        #[serde(default)]
        max_clusters: Option<usize>,
        /// Clock units between checkpoints (pruning, then linkage), as
        /// `solve_every` is the regressions': a number of the clock column's
        /// units, or a duration on a temporal clock, `0` every row
        /// (docs/PLAN.md task 163). With `max_rows_between_prunes` too,
        /// whichever comes first; with neither, every 100 learned rows.
        #[serde(default)]
        prune_every: Option<Span>,
        /// At most this many learned rows between checkpoints (task 163).
        #[serde(default)]
        max_rows_between_prunes: Option<u32>,
        /// Single-linkage threshold in units of `eps √p`; `0` links nothing.
        /// Default: derived from the spacing of the potential summaries at
        /// each checkpoint.
        #[serde(default)]
        macro_link: Option<f64>,
        /// Measure distances in units of each feature's EW standard
        /// deviation. Default true.
        #[serde(default)]
        standardize: Option<bool>,
        /// Floor the metric's variance at this fraction of the feature's
        /// long-run (undecayed) variance. Default 0.1; `0` is the EW
        /// variance alone.
        #[serde(default)]
        scale_floor: Option<f64>,
    },
    /// Class-conditional Gaussian classifier on exponentially weighted
    /// class moments (docs/ENHANCEMENTS.md E39; PLAN §11a, task 27). The
    /// one target is the label column, which names its class by value; the
    /// outputs are the most probable class and one posterior per class,
    /// read before the row is learned, and `coef` holds the class means.
    #[serde(rename = "ew_class")]
    EwClass {
        /// The classes, in output order; a label value not listed here is
        /// an error, a null label scores the row without learning from it.
        classes: Vec<String>,
        /// "full" (default: one covariance per class, QDA), "tied" (one
        /// pooled covariance, LDA) or "diag" (naive Bayes): scikit-learn's
        /// `GaussianMixture` words (task 244, P-2).
        #[serde(default)]
        covariance_type: Option<String>,
        /// Ridge on every class covariance, finite and `> 0`; it decays as
        /// the class accumulates data, like `ew_cov`'s `precision_prior`.
        precision_prior: f64,
        /// Clock units of history each class's moments are computed from,
        /// with a **hard** cutoff (docs/PLAN.md §13). `covariance_type = "full"`
        /// pays one factorization per class per row under a window, because
        /// the truncated covariance moves every row where a decayed one does
        /// not; the other shapes are unaffected.
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the window's snapshots, `0` every row, as
        /// for `EwRidge` (docs/PLAN.md task 162).
        #[serde(default)]
        snapshot_every: Option<Span>,
        /// At most this many rows between the window's snapshots; `1` is
        /// every row, and `0` is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
    },
    /// Sequential test of a sign by betting (docs/ENHANCEMENTS.md E42;
    /// PLAN §11a, task 30): per target, two e-processes, one for "positive"
    /// and one for "negative", each a Kelly bettor with a
    /// Krichevsky–Trofimov stake on the sign counts so far. The outputs are
    /// the two log e-values and the two counts, read before the row is
    /// learned, so `exp(log_e) >= 1/alpha` at any row is evidence at level
    /// `alpha` -- anytime-valid, under no assumption but that the signs are
    /// not predictable from the past. No features, no decay, no `weight`
    /// (every learned row is one trial); a null or zero target bets nothing.
    ///
    /// Two specs of the same bank are compared by naming them as `a` and
    /// `b`: each target `t` then names a residual field both specs carry
    /// (`resid_<t>`, plus the side's grid suffix when it has one), the sign
    /// tested is that of `|resid_b| - |resid_a|` -- positive when `a` was
    /// closer on the row -- and the fields are `log_e_a_<t>`, `log_e_b_<t>`,
    /// `wins_a_<t>`, `wins_b_<t>`. The bank runs `a` and `b` first, so the
    /// comparison reads the same out-of-sample residuals the two structs
    /// report.
    #[serde(rename = "seqtest")]
    SeqTest {
        /// The spec whose predictions are hoped to be better.
        #[serde(default)]
        a: Option<String>,
        /// The spec it is compared with.
        #[serde(default)]
        b: Option<String>,
        /// The grid suffix of `a`'s residual field, when `a` is a grid:
        /// `resid_<t><a_suffix>`, e.g. `__r0.1` or `@h50`. Default none.
        #[serde(default)]
        a_suffix: Option<String>,
        /// The same for `b`.
        #[serde(default)]
        b_suffix: Option<String>,
    },
    /// Exponentially weighted moments of every feature against every target,
    /// one pair at a time (docs/ENHANCEMENTS.md E44; PLAN §11a, task 37):
    /// per (target, feature) the two means, the two variances, the
    /// covariance, and the target's `Σw` and `Σw²` -- `O(p·T)` state where
    /// an `ew_cov` over the same columns keeps `O((p+T)²)`. It emits nothing
    /// per row but `n_eff`; the pairs are read from the bank as a long frame
    /// (`ModelBank.marginal`), each with its correlation, the slope of the
    /// target on the feature, Kish's effective sample size and the
    /// t-statistic of the correlation at that size. A pair's moments are
    /// an `ew_cov`'s over the two columns: the means and variances to the
    /// bit, the covariance and correlation within a few roundings. A null
    /// target ages that target's pairs and moves nothing else; a
    /// null feature drops the row for every pair, as everywhere.
    /// `half_life`/`lam`, `weight`, `clock` and `min_weight` are the spec's;
    /// the lags, the bins, the window and the shards below are its own.
    /// `min_weight` (default 3) is the
    /// weight a target needs before its pairs' `corr`, `beta` and `t` are
    /// reported -- a correlation of two rows is ±1 whatever the data.
    #[serde(rename = "marginal")]
    Marginal {
        /// Lags to accumulate pair moments at (docs/ENHANCEMENTS.md E66):
        /// strictly increasing, `>= 1` and at most 2^20
        /// (`online_core::MAX_LAG`), counted in **learned rows within the
        /// group**. Gives the two autocorrelations and both
        /// cross-correlations per pair, and the serial-dependence-corrected
        /// `n_serial` when `serial_rule` asks for it.
        ///
        /// **Skipped when absent**, unlike the spec keys that predate
        /// 2026-09-07: a key that writes `null` moves every spec's bytes and
        /// costs a `SCHEMA_VERSION` bump for a model nobody using it has
        /// heard of. New optional keys skip, so adding one to a model
        /// changes only the states that use it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lags: Option<Vec<usize>>,
        /// `"truncated"`, `"bartlett"` or `"geometric"`: how `n_serial` is
        /// formed from the lags. Needs `lags`. Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        serial_rule: Option<String>,
        /// The lags to keep the cross-correlations at (E70, docs/PLAN.md
        /// task 123): strictly increasing, each one of `lags`, empty for
        /// none. Absent keeps them at every lag. `n_serial` reads the
        /// autocorrelations alone, which every lag keeps. Needs `lags`.
        /// Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cross_lags: Option<Vec<usize>>,
        /// Bins per feature for the binned target moments
        /// (docs/ENHANCEMENTS.md E67): the feature's response curve, and the
        /// best single split of it, which is what a threshold or a V shows
        /// up in when `corr` cannot see it. Skipped when absent, as `lags`
        /// is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bins: Option<usize>,
        /// `"quantile"` (equal weights, the default) or `"fixed"` (equal
        /// widths): how `bins` becomes edges. Refused with `bin_edges`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_rule: Option<String>,
        /// Learned rows to hold before fixing the edges, default 1,000. The
        /// rows are held and replayed, not spent. Refused with `bin_edges`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_warm_rows: Option<usize>,
        /// Explicit interior edges, one strictly increasing list per feature
        /// in feature order. Exact and comparable across runs, and skips the
        /// warm-up entirely; `bins`, `bin_rule` and `bin_warm_rows` are the
        /// learned kind's and refused beside it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_edges: Option<Vec<Vec<f64>>>,
        /// The MiB the bins' warm-up hold and histogram may each take, per
        /// model instance -- every group keeps its own, and so does every
        /// half-life of a grid -- before the model refuses to build
        /// (docs/PLAN.md task 131): 256 when absent, `"inf"` no bound. Needs
        /// `bins` or `bin_edges`. Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_budget: Option<Num>,
        /// What each bin's location is: `"mean"`, the default, or
        /// `"huber"`, Huber's M-estimate of location learned in one pass,
        /// with the cut at `bin_huber_delta` times the target's robust scale
        /// (`online_core::BinStat`; docs/PLAN.md task 272 (c)). Needs `bins`
        /// or `bin_edges`. Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_stat: Option<String>,
        /// The cut in units of the target's robust scale under
        /// `bin_stat = "huber"`: above 0, default 1.345 (`huber`'s), and
        /// `"inf"` clips nothing, which is the mean to the bit. Refused
        /// under any other `bin_stat`. Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_huber_delta: Option<Num>,
        /// The bins' refit count (`online_core::BinCfg::refit_rows`, task
        /// 278): absent is `max(500, 30 · bins)`. Skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bin_refit_rows: Option<u32>,
        /// `huber`'s `refit_budget`, for the bins' refit holds (task 278).
        /// Skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refit_budget: Option<Num>,
        /// Ranges of features to split the pair work into, run on the bank's
        /// pool a batch of rows at a time (docs/PLAN.md task 126): a count,
        /// or `"auto"` for as many as the width can keep busy. The numbers
        /// are the same to the bit whatever it says, so it is a setting,
        /// not state. Absent is one, the pairs row by row on the group's
        /// own thread. Skipped when absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shards: Option<ShardSpec>,
        /// Clock units of history the pairs are computed from, with a
        /// **hard** cutoff: a row older than this contributes nothing
        /// (docs/PLAN.md §13). Inside the window the weights are still
        /// exponential.
        #[serde(default)]
        window_size: Option<Span>,
        /// Which edge holds a row exactly `window_size` old, Polars'
        /// `closed` ([`Closed`], docs/PLAN.md task 196): `"right"`, the
        /// default, or `"both"`, which needs `window_size`.
        #[serde(default)]
        closed: ModelClosed,
        /// Clock units between the window's snapshots, `0` every row, as
        /// for `EwRidge` (docs/PLAN.md task 162).
        #[serde(default)]
        snapshot_every: Option<Span>,
        /// At most this many rows between the window's snapshots; `1` is
        /// every row, and `0` is refused (tasks 162 and 196).
        #[serde(default)]
        max_rows_between_snapshots: Option<u32>,
        /// Past this many MiB the window's snapshots thin, or refuse the run
        /// ([`WindowBudgetSpec`]); a spec that names none refuses past 256
        /// MiB (review 2026-09-12, P4).
        #[serde(default)]
        window_budget: Option<WindowBudgetSpec>,
        /// Accept `lags` under a `window_size` at its price (docs/PLAN.md task
        /// 137): each window snapshot then also holds the lag moments, `L·T
        /// + (L + 2C)·p·T` doubles beside the `(3p + 5)·T` it holds without
        /// them, adding about `(L + 2C)/3` times as many. Refused without both a
        /// window and lags; the pair is refused without it. Skipped when
        /// absent, as `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        window_lags: Option<bool>,
        /// Where the feature moments are kept (docs/PLAN.md task 125):
        /// `"per_target"`, the default, keeps each pair's mean and variance of
        /// the feature over its target's rows; `"shared"` keeps one per
        /// feature over every learned row, `p` where the default keeps `p·T`,
        /// and under `lags` the feature's autocovariance with them. Refused
        /// with a window, as the core says. Skipped when absent, as
        /// `window_lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        feature_moments: Option<String>,
    },
    /// Dynamic equicorrelation (Engle & Kelly 2012; docs/ENHANCEMENTS.md
    /// E55): one number for the whole correlation matrix, `O(m)` a row where
    /// a full one is `O(m²)`.
    ///
    /// The row is standardised against an [`online_core::EwDiag`]'s pre-row
    /// means and variances, and the average pairwise correlation of the
    /// standardised row is a closed form in two sums. `blocks` estimates one
    /// number per named block and one per pair of blocks instead of a single
    /// one, which is the useful middle between "one correlation" and "all of
    /// them". Outputs `u` (the row's own estimate), `rho` (the level before
    /// the row), `loglik` (the row's Gaussian density under it) and `n_eff`;
    /// with blocks, one `u_*` and one `rho_*` per value and a single
    /// `loglik`. `half_life`/`lam` are the spec's and are required, since
    /// both the standardiser and the level decay on them.
    #[serde(rename = "deco")]
    Deco {
        /// `"ew"` (default): `rho' = a·rho + b·u`, the exponentially
        /// weighted mean of the row estimates. `"linear"`: Engle–Kelly
        /// eq. 21 with correlation targeting, `rho' = (1 − α − β)·rho_bar' +
        /// α·u + β·rho`, which needs `alpha` and `beta`.
        #[serde(default)]
        dynamics: Option<String>,
        #[serde(default)]
        alpha: Option<f64>,
        #[serde(default)]
        beta: Option<f64>,
        /// Named groups of features, in emission order; every feature must
        /// be in exactly one, and a block needs at least two. Omitted, the
        /// model estimates one number over every feature.
        /// `feature_sets`' shape: a Python dict, serialised as a list of
        /// `[name, columns]` pairs so the order the caller wrote is the
        /// order the outputs come in.
        #[serde(default)]
        blocks: Option<Vec<(String, Vec<String>)>>,
    },
    /// A block's realised covariance, robust to microstructure noise
    /// (docs/ENHANCEMENTS.md E57). Rows are **returns**; there is no decay
    /// and no per-row output but `n_eff`, because the model's value is its
    /// state at the group's close -- so it requires `group` and
    /// `group_close`, and the block rides in that row.
    #[serde(rename = "rcov")]
    Rcov {
        /// `"kernel"` (default), `"preavg"` or `"plain"`.
        #[serde(default)]
        kind: Option<String>,
        /// Only `"parzen"`.
        #[serde(default)]
        kernel: Option<String>,
        /// A fixed `H`, or omitted for BNHLS's automatic rule (which needs
        /// `block_rows`).
        #[serde(default)]
        bandwidth: Option<usize>,
        /// Observations averaged at each end; default 2, `1` is none.
        /// `"kernel"` only: the other kinds read none, and a value given to
        /// them is refused (review 2026-10-06, CE4).
        #[serde(default)]
        jitter: Option<usize>,
        /// Pre-averaging window scale, `kₙ = ⌈θ·block_rows^0.6⌉` under `psd`
        /// (the default) and `⌊θ√block_rows⌋` without it; default 1.
        /// `"preavg"` only, refused under the other kinds (CE4).
        #[serde(default)]
        theta: Option<f64>,
        /// Clip negative eigenvalues at close; default true.
        #[serde(default)]
        psd: Option<bool>,
        /// The block's expected length, which sizes the ring before the
        /// first row.
        #[serde(default)]
        block_rows: Option<usize>,
        /// Ring depth, if not the default from `block_rows`. `"kernel"`
        /// only, refused under the other kinds (CE4); the ring's lagged
        /// products, `(ring + 1)·k²` doubles, are held to 256 MiB (CE9).
        #[serde(default)]
        max_bandwidth: Option<usize>,
        /// A fixed pre-averaging length, in rows, instead of `⌊θ√block_rows⌋`.
        #[serde(default)]
        preavg_rows: Option<usize>,
        /// Subsampling stride for the noise estimate; default 1.
        #[serde(default)]
        noise_stride: Option<usize>,
        /// Subsampling stride for the sparse variance; default 20.
        #[serde(default)]
        iv_stride: Option<usize>,
    },
    /// A Gaussian hidden Markov model, filtered online
    /// (docs/ENHANCEMENTS.md E60). `ew_class` without the labels: the state
    /// is hidden, and a transition matrix carries information from one row
    /// to the next.
    #[serde(rename = "hmm")]
    Hmm {
        /// Hidden states, `2 ..= 1024` (`online_core::Hmm::MAX_K`):
        /// `n_components`, scikit-learn's `GaussianMixture` and hmmlearn's
        /// name (task 244, P-9), kept `k` here as the core calls it.
        #[serde(rename = "n_components")]
        k: usize,
        /// `"full"` (default), `"tied"` or `"diag"`, as `ew_class`.
        #[serde(default)]
        covariance_type: Option<String>,
        /// Ridge on every state covariance; **required**, since a state's
        /// centred co-moments start at zero.
        precision_prior: f64,
        /// Update the states and the transition counts; default true.
        #[serde(default)]
        learn: Option<bool>,
        /// Dirichlet pseudo-count per cell of the transition matrix
        /// (default 1). With `transition` it is spread over that matrix, so
        /// the given one is the prior mean. Refused beside `tvtp_coef`,
        /// which learns no count (review 2026-10-06, CE4).
        #[serde(default)]
        transition_prior: Option<f64>,
        /// A `K x K` row-stochastic matrix, flattened row-major. Refused
        /// beside `tvtp_coef`, as `transition_prior` is.
        #[serde(default)]
        transition: Option<Vec<f64>>,
        /// State means, `K x d` row-major; with `covs`, there is no warm-up.
        #[serde(default)]
        means: Option<Vec<f64>>,
        /// State covariances, `K` matrices of `d x d`, row-major.
        #[serde(default)]
        covs: Option<Vec<f64>>,
        /// Learned rows buffered before the states are seeded (default
        /// 1,000, or `n_components` where that is more).
        /// With `means` and `covs` given nothing is seeded, and it, like
        /// `init` and `seed`, is refused (review 2026-10-06, CE4).
        #[serde(default)]
        warm_rows: Option<usize>,
        /// `"first"`, `"farthest"`, `"k-means++"` or `"lloyd"` (default), as
        /// `kmeans`.
        #[serde(default)]
        init: Option<String>,
        #[serde(default)]
        seed: Option<u64>,
        /// A column whose value drives the transition matrix through
        /// `tvtp_coef`; declared like `weight`, not a feature.
        #[serde(default)]
        exog_tvtp: Option<String>,
        /// `[A, B]`, each `K x K` row-major: `Π(t) = softmax(A + B·z)`.
        #[serde(default)]
        tvtp_coef: Option<Vec<Vec<f64>>>,
    },
    /// Has the correlation structure changed? (docs/ENHANCEMENTS.md E59)
    ///
    /// `"monitor"` is Wied, Krämer & Dehling's closed-sample constancy
    /// test, run over consecutive spans of `window_rows` rows; `"sequential"`
    /// is Wied & Galeano's detector, a history of `window_rows` rows and then
    /// every row of a monitoring period tested against it; `"window"` is
    /// the size of the change between two adjacent blocks of `window_rows`,
    /// against a fixed threshold or a permutation critical value. A
    /// parameter that belongs to another kind is refused.
    #[serde(rename = "corrchange")]
    CorrChange {
        /// `"monitor"` (default), `"sequential"` or `"window"`.
        #[serde(default)]
        kind: Option<String>,
        /// Rows per comparison block, required by every kind: the span
        /// `"monitor"` tests for constancy (at least 8), the history
        /// `"sequential"` monitors against (at least 8), or the length of
        /// each of the two adjacent blocks `"window"` compares (at least 3).
        #[serde(default)]
        window_rows: Option<usize>,
        /// Nominal level; default 0.05.
        #[serde(default)]
        alpha: Option<f64>,
        /// `"bonferroni"` (default) over the pairs, or `"none"`.
        #[serde(default)]
        alpha_adjust: Option<String>,
        /// `"monitor"` and `"sequential"`: the Bartlett bandwidth, `⌊ln T⌋`
        /// (or `⌊ln window_rows⌋`) when unset.
        #[serde(default)]
        bandwidth: Option<usize>,
        /// `"monitor"` and `"sequential"`: test the equicorrelation of the
        /// standardised row instead of every pair.
        #[serde(default)]
        scalar: Option<bool>,
        /// `"window"`: a fixed critical value; unset draws a permutation
        /// one. `"sequential"`: replaces Wied & Galeano's.
        #[serde(default)]
        crit: Option<f64>,
        /// `"window"`: permutation draws per critical value; default 200, at
        /// least 20 and at most 2^20 (`online_core::MAX_PERM`).
        #[serde(default)]
        n_perm: Option<usize>,
        #[serde(default)]
        permute_every_rows: Option<usize>,
        /// Permute blocks of this many consecutive rows (default 1), so
        /// serial dependence does not make the null too liberal.
        #[serde(default)]
        perm_block: Option<usize>,
        /// `"l1"` (default) or `"linf"` over the strict upper triangle.
        #[serde(default)]
        norm: Option<String>,
        #[serde(default)]
        seed: Option<u64>,
        /// `"reset"` empties the rings at a flag and starts over;
        /// `"flag"`, the default, flags and carries on: the words
        /// `drift_action` takes (task 244). `"window"` only: `"monitor"`'s
        /// spans are disjoint already, and a `"sequential"` cycle ends at its
        /// flag.
        #[serde(default)]
        change_action: Option<String>,
        /// `"sequential"`: the rows monitored after each history, Wied &
        /// Galeano's `⌊mT⌋`; default `window_rows` (`T = 1`).
        #[serde(default)]
        monitor_rows: Option<usize>,
        /// `"sequential"`: the boundary's exponent `γ`, `0 ≤ γ < 1/2`;
        /// default 0.
        #[serde(default)]
        boundary_gamma: Option<f64>,
    },
    /// Bayesian online changepoint detection (docs/ENHANCEMENTS.md E61): a
    /// posterior over how long the current run has lasted, updated one row
    /// at a time.
    #[serde(rename = "bocpd")]
    Bocpd {
        /// A number: the expected rows between changepoints, `H = 1/hazard`
        /// the chance of a break after each row, on any clock. A duration,
        /// which only a temporal clock reads: the expected time between
        /// them, the step of `d` into each row carrying the chance `1 −
        /// exp(−d/hazard)`, applied before the row is read (task 179).
        /// Default 250 rows. In [`DURATION_OR_UNIT_FREE_FIELDS`].
        #[serde(default)]
        hazard: Option<Span>,
        /// A column carrying a per-row hazard instead of one number: the
        /// chance of a break after its row. Refused beside a duration
        /// `hazard`.
        #[serde(default)]
        hazard_col: Option<String>,
        /// `"full"`, a normal-inverse-Wishart over all the features, or
        /// `"diag"` (default), a normal-inverse-gamma per feature:
        /// scikit-learn's covariance words (task 244, P-2), where `emission`
        /// said `"gaussian"` and `"diag"`.
        #[serde(default)]
        covariance_type: Option<String>,
        /// `μ₀`; absent, the first `warm_rows` learned rows' sample mean.
        /// scikit-learn's `BayesianGaussianMixture` names the four priors
        /// (task 244, P-3).
        #[serde(default)]
        mean_prior: Option<Vec<f64>>,
        /// `κ₀`, the mean's weight in rows.
        #[serde(default)]
        mean_precision_prior: Option<f64>,
        /// `ν₀`; absent, `d + 2` under `"full"` and 3 under `"diag"`
        /// (docs/PLAN.md task 195, U5).
        #[serde(default)]
        degrees_of_freedom_prior: Option<f64>,
        /// A scalar for `sI`, or a `d x d` matrix, in the data's units;
        /// absent, the first `warm_rows` learned rows' sample covariance
        /// (its diagonal under `"diag"`; task 195, U4).
        #[serde(default)]
        covariance_prior: Option<Vec<f64>>,
        /// Temper each row by `(π(x) / π(mode)) ** robust_beta`, the robust
        /// predictive, under `"diag"`; absent, the predictive untempered.
        /// It was `emission = "robust"`, whose default β was 0.1.
        #[serde(default)]
        robust_beta: Option<f64>,
        #[serde(default)]
        prune_below: Option<f64>,
        #[serde(default)]
        max_run: Option<usize>,
        /// Learned rows held, reporting nothing, to set the priors left out
        /// from; default `d + 2`. Refused beside both `mean_prior` and
        /// `covariance_prior` (task 195, U4).
        #[serde(default)]
        warm_rows: Option<usize>,
    },
    /// What a stream's columns hold, read in one pass (docs/PLAN.md task
    /// 223 (b)): per column the nulls, NaNs, infinities and values past
    /// the input bound counted apart, the moments, the distinct values and
    /// the most repeated one, the longest run of one value, the lag-1
    /// autocorrelation and a robust z; per pair of columns, when `pairs`,
    /// the correlation and the equal rows; and the clock's duplicate stamps,
    /// gaps and spread of steps. It learns nothing, emits `weight_sum`
    /// alone, and is read from the bank as frames (`ModelBank.audit`). Its
    /// columns are its `features`; it reads every row, a null in any column
    /// included. No decay, no `weight` (a value is a value), no `embargo`.
    #[serde(rename = "audit")]
    Audit {
        /// Keep each pair of columns' correlation and equal rows, `O(k²)`.
        /// Default false. Skipped when absent, as `marginal`'s `lags` is.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pairs: Option<bool>,
        /// Counters per column: the distinct count is exact up to this many
        /// values, the most repeated value's count within `n / (cap + 1)`
        /// past it. Default 256. Skipped when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        distinct_cap: Option<usize>,
    },
}

/// The two sides of a `seqtest` comparison, as [`ModelKind::compares`]
/// reads them off the spec: each side's spec name and the grid suffix its
/// residual fields carry (`""` for a single instance).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compare<'a> {
    pub a: &'a str,
    pub b: &'a str,
    pub a_suffix: &'a str,
    pub b_suffix: &'a str,
}

impl Compare<'_> {
    /// The residual field target `t` names on each side.
    pub fn fields(&self, t: &str) -> (String, String) {
        (
            format!("resid_{t}{}", self.a_suffix),
            format!("resid_{t}{}", self.b_suffix),
        )
    }
}

impl ModelKind {
    /// Every model the bank can build, by the `type` a spec names it with,
    /// in declaration order. The registry the Python side and the tests
    /// check themselves against (docs/EXTENDING.md); `kinds_tests` holds it
    /// to the enum, so a new variant fails a test until it is listed here.
    pub const KINDS: &'static [&'static str] = &[
        "ewridge",
        "lasso",
        "kalman",
        "huber",
        "quantile",
        "ftrl",
        "ew_cov",
        "sgd",
        "pa",
        "holt",
        "rls",
        "kmeans",
        "micro",
        "ew_class",
        "seqtest",
        "marginal",
        "deco",
        "rcov",
        "hmm",
        "corrchange",
        "bocpd",
        "audit",
    ];

    pub fn kind_name(&self) -> &'static str {
        match self {
            ModelKind::EwRidge { .. } => "ewridge",
            ModelKind::Rls { .. } => "rls",
            ModelKind::Lasso { .. } => "lasso",
            ModelKind::Kalman { .. } => "kalman",
            ModelKind::Huber { .. } => "huber",
            ModelKind::Quantile { .. } => "quantile",
            ModelKind::Ftrl { .. } => "ftrl",
            ModelKind::EwCov { .. } => "ew_cov",
            ModelKind::Sgd { .. } => "sgd",
            ModelKind::Pa { .. } => "pa",
            ModelKind::Holt { .. } => "holt",
            ModelKind::KMeans { .. } => "kmeans",
            ModelKind::Micro { .. } => "micro",
            ModelKind::EwClass { .. } => "ew_class",
            ModelKind::SeqTest { .. } => "seqtest",
            ModelKind::Marginal { .. } => "marginal",
            ModelKind::Deco { .. } => "deco",
            ModelKind::Rcov { .. } => "rcov",
            ModelKind::Hmm { .. } => "hmm",
            ModelKind::CorrChange { .. } => "corrchange",
            ModelKind::Bocpd { .. } => "bocpd",
            ModelKind::Audit { .. } => "audit",
        }
    }

    /// True for the model that reads every row whatever its columns hold:
    /// `audit`, which counts what the others skip (docs/PLAN.md task 223
    /// (b)). Every other model's row is skipped at a feature that is not
    /// usable.
    pub fn reads_every_row(&self) -> bool {
        matches!(self, ModelKind::Audit { .. })
    }

    /// The column a model with no target reads from the targets slot, when
    /// the spec names one: `bocpd`'s `hazard_col` and `hmm`'s `exog_tvtp`.
    /// It fills `targets` ([`Spec::fill_defaults`]) so the column is read
    /// from the targets slot (review 2026-09-12, C20, S24).
    pub fn targets_slot_column(&self) -> Option<&str> {
        match self {
            ModelKind::Bocpd { hazard_col, .. } => hazard_col.as_deref(),
            ModelKind::Hmm { exog_tvtp, .. } => exog_tvtp.as_deref(),
            _ => None,
        }
    }

    /// False for the models that report no coefficients -- no `coef` field
    /// in their output: `ew_cov`, `seqtest`, `marginal`, `rcov`, `corrchange`,
    /// `bocpd` and `audit`. `tests/test_model_registry.py` holds this to the field
    /// names, which are rendered elsewhere.
    pub fn has_coef(&self) -> bool {
        !matches!(
            self,
            ModelKind::EwCov { .. }
                | ModelKind::SeqTest { .. }
                | ModelKind::Marginal { .. }
                | ModelKind::Rcov { .. }
                | ModelKind::CorrChange { .. }
                | ModelKind::Bocpd { .. }
                | ModelKind::Audit { .. }
        )
    }

    /// `gram_threads` as the spec gives it, for the two kinds that take it
    /// (docs/PLAN.md task 225); `None` for every other kind.
    pub fn gram_threads_given(&self) -> Option<GramThreads> {
        match self {
            ModelKind::EwRidge { gram_threads, .. } | ModelKind::EwCov { gram_threads, .. } => {
                *gram_threads
            }
            _ => None,
        }
    }

    /// The threads a model's Gram update may run on, as the model takes
    /// it ([`online_core::OnlineModel::set_gram_threads`]), before the bank
    /// shares its pool out: the spec's count, or [`usize::MAX`], every
    /// thread of the pool, for `"auto"` and for none given (task 253: the
    /// default is `"auto"`). The model caps it at the pool it runs on, and
    /// the bank sets [`Self::gram_threads_sharing`] on each chunk.
    /// Configuration, not state: the stream sets it on every model it
    /// builds or restores.
    pub fn gram_threads(&self) -> usize {
        self.gram_threads_sharing(usize::MAX)
    }

    /// [`Self::gram_threads`] when the bank's pool is shared: `share` is
    /// [`crate::pool::gram_share`]`(T, P)` for the chunk's `T` parallel
    /// tasks on a pool of `P`. `"auto"`, and none given, take the share, so
    /// a model splits its rows only where the pool has idle threads; a count
    /// is the caller's, capped at the pool by the model as before.
    pub fn gram_threads_sharing(&self, share: usize) -> usize {
        match self.gram_threads_given() {
            Some(CountOrAuto::Count(n)) => n,
            Some(CountOrAuto::Auto) | None => share,
        }
    }

    /// Whether a stream of this kind keeps a window's ring: a model's, or a
    /// spread's cut from it, both of which need a `window_size`. Without one
    /// the window pre-pass has nothing to replay (task 270, G-1).
    pub fn keeps_window_ring(&self) -> bool {
        self.window_parts()
            .is_some_and(|(window, _)| window.is_some())
    }

    /// A windowed kind's `window_size` and `window_budget`; `None` for a kind with
    /// no window to bound.
    pub fn window_parts(&self) -> Option<(Option<f64>, Option<WindowBudgetSpec>)> {
        match self {
            ModelKind::EwRidge {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::Lasso {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::EwCov {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::EwClass {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::Marginal {
                window_size: window,
                window_budget,
                ..
            } => Some((window.as_ref().map(Span::value), *window_budget)),
            _ => None,
        }
    }

    /// The budget a model's window runs under: the spec's, or refuse past
    /// 256 MiB where it names none (`online_core::WindowBudget::DEFAULT`).
    /// `None` without a window.
    /// The MiB a spec's groups' Huber refit buffers may reserve in a bank
    /// (docs/PLAN.md task 277): `refit_budget`, [`REFIT_BUDGET_MIB`] by default, `None` for
    /// no bound (`inf`) or a kind without the refit.
    pub fn refit_budget_mib(&self) -> Option<f64> {
        let given = match self {
            ModelKind::Huber { refit_budget, .. } | ModelKind::Quantile { refit_budget, .. } => {
                refit_budget
            }
            ModelKind::Sgd {
                refit_budget, loss, ..
            } if loss.as_deref() == Some("huber") => refit_budget,
            ModelKind::Marginal {
                refit_budget,
                bin_stat,
                ..
            } if bin_stat.as_deref() == Some("huber") => refit_budget,
            _ => return None,
        };
        let mib = given.map_or(REFIT_BUDGET_MIB, |n| n.0);
        mib.is_finite().then_some(mib)
    }

    pub fn window_budget(&self) -> Option<online_core::WindowBudget> {
        let (window, budget) = self.window_parts()?;
        window.map(|_| {
            budget.map_or(
                online_core::WindowBudget::DEFAULT,
                WindowBudgetSpec::to_core,
            )
        })
    }

    /// A windowed kind's `snapshot_every` and `max_rows_between_snapshots`, as
    /// the spec gives them; `None` for a kind with no window (docs/PLAN.md
    /// task 162).
    pub fn window_cadence(&self) -> Option<(Option<&Span>, Option<u32>)> {
        match self {
            ModelKind::EwRidge {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::Lasso {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::EwCov {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::EwClass {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::Marginal {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            } => Some((snapshot_every.as_ref(), *max_rows_between_snapshots)),
            _ => None,
        }
    }

    /// A windowed kind's `closed`, as the spec gives it; `None` for a kind
    /// with no window (docs/PLAN.md task 196, N17).
    pub fn window_closed(&self) -> Option<ModelClosed> {
        match self {
            ModelKind::EwRidge { closed, .. }
            | ModelKind::Lasso { closed, .. }
            | ModelKind::EwCov { closed, .. }
            | ModelKind::EwClass { closed, .. }
            | ModelKind::Marginal { closed, .. } => Some(*closed),
            _ => None,
        }
    }

    /// The edge a windowed model's ring keeps ([`online_core::WindowClosed`]):
    /// the spec's `closed`, `Right` by default and for a kind with no window,
    /// which ignores it. A `closed` a model refuses never gets here: it is
    /// refused as the spec is read ([`ModelClosed`]).
    pub fn window_edge(&self) -> online_core::WindowClosed {
        self.window_closed().map(model_edge).unwrap_or_default()
    }

    /// Every `max_rows_between_*` the model takes, as given, beside the
    /// clock parameter it caps: what [`Spec::validate`] holds to `>= 1`
    /// (docs/PLAN.md task 196, U7). The spec's own `max_rows_between_coefs`
    /// is checked there.
    pub fn row_caps(&self) -> Vec<(&'static str, Option<u32>, &'static str)> {
        let mut caps = Vec::new();
        match self {
            ModelKind::EwRidge {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Lasso {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Huber {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Quantile {
                max_rows_between_solves,
                ..
            } => caps.push((
                "max_rows_between_solves",
                *max_rows_between_solves,
                "solve_every",
            )),
            ModelKind::EwCov {
                max_rows_between_pcas,
                ..
            } => caps.push(("max_rows_between_pcas", *max_rows_between_pcas, "pca_every")),
            ModelKind::Micro {
                max_rows_between_prunes,
                ..
            } => caps.push((
                "max_rows_between_prunes",
                *max_rows_between_prunes,
                "prune_every",
            )),
            _ => {}
        }
        if let Some((_, rows)) = self.window_cadence() {
            caps.push(("max_rows_between_snapshots", rows, "snapshot_every"));
        }
        caps
    }

    /// The cadence a windowed model's ring takes its snapshots on, as the
    /// model maps its configuration ([`online_core::Cadence::of`]): every
    /// row unless given (docs/PLAN.md task 162).
    pub fn snapshot_cadence(&self) -> Option<online_core::Cadence> {
        let (every, rows) = self.window_cadence()?;
        Some(online_core::Cadence::of(
            every.map(Span::value),
            rows.map(|r| r as usize),
        ))
    }

    /// The `window_size` and snapshot cadence of a windowed model that
    /// predicts a target: what the stream cuts its residual spread with
    /// (review 2026-09-12, S1), the model's own cadence, so the spread's
    /// boundary is the fit's. The other windowed models predict none, so
    /// the stream keeps no spread for them.
    pub fn window_and_cadence(&self) -> Option<(f64, online_core::Cadence)> {
        match self {
            ModelKind::EwRidge {
                window_size: Some(w),
                ..
            }
            | ModelKind::Lasso {
                window_size: Some(w),
                ..
            } => Some((w.value(), self.snapshot_cadence()?)),
            _ => None,
        }
    }

    /// True for the models that learn from no target column: `ew_cov`,
    /// `kmeans`, `micro`, `deco`, `rcov`, `hmm`, `corrchange`, `bocpd` and
    /// `audit`,
    /// the list `_spec.py`'s `UNSUPERVISED` keeps too. Their `targets` mirror
    /// `features[0]` for plumbing, so a target that is also a feature is not
    /// a leak for them -- except the column [`Self::targets_slot_column`]
    /// names, which is read as a target is.
    pub fn is_unsupervised(&self) -> bool {
        matches!(
            self,
            ModelKind::EwCov { .. }
                | ModelKind::KMeans { .. }
                | ModelKind::Micro { .. }
                | ModelKind::Deco { .. }
                | ModelKind::Rcov { .. }
                | ModelKind::Hmm { .. }
                | ModelKind::CorrChange { .. }
                | ModelKind::Bocpd { .. }
                | ModelKind::Audit { .. }
        )
    }

    /// True for the models that predict no target as a number: the
    /// unsupervised ones, `ew_class`, whose target is a label it
    /// classifies, `seqtest`, whose targets are the signs it tests, and
    /// `marginal`, whose targets are the columns it correlates the features
    /// with. Their outputs are statistics, assignments, posteriors or
    /// e-values read from the state *before* each row (or, for `marginal`,
    /// nothing at all), and nothing residual-based (`sigma`, `zscore`,
    /// metrics, quantiles, conformal, autocorrelation, drift, selection,
    /// averaging) applies to them; their slots are whatever rides in
    /// `pred`, not targets × combos.
    pub fn predicts_no_target(&self) -> bool {
        self.is_unsupervised()
            || matches!(
                self,
                ModelKind::EwClass { .. } | ModelKind::SeqTest { .. } | ModelKind::Marginal { .. }
            )
    }

    /// The two specs a `seqtest` compares, when it compares rather than
    /// tests columns. Such a spec's targets name residual fields, read from
    /// the two specs' own output in the bank, not columns of the frame; the
    /// bank runs it after them.
    pub fn compares(&self) -> Option<Compare<'_>> {
        match self {
            ModelKind::SeqTest {
                a: Some(a),
                b: Some(b),
                a_suffix,
                b_suffix,
            } => Some(Compare {
                a,
                b,
                a_suffix: a_suffix.as_deref().unwrap_or(""),
                b_suffix: b_suffix.as_deref().unwrap_or(""),
            }),
            _ => None,
        }
    }
}

/// The parameters task 144 renamed (docs/PLAN.md): a spec dict, a TOML file
/// or a windows config naming the old one is refused naming the new one, in
/// place of a bare "unknown field". No alias.
pub const RENAMED: &[(&str, &str)] = &[
    ("halflife", "half_life"),
    ("long_halflife", "long_half_life"),
    ("coef_halflife", "coef_half_life"),
    ("revert_halflife", "revert_half_life"),
    ("select_halflife", "select_half_life"),
    // Straight to the spec's `half_life`: `level_half_life` is refused too
    // (task 196).
    ("level_halflife", "half_life"),
    ("trend_halflife", "trend_half_life"),
    ("label_delay", "embargo"),
    ("max_dclock", "gap_cap"),
    ("window", "window_size"),
    ("min_periods", "min_weight"),
    ("emit_resid_z", "emit_zscore"),
    ("scale_features", "standardize"),
    ("on_clock_reset", "restart_after_step_back"),
    ("min_backwards_jump", "restart_after_step_back"),
    ("ridge_decay", "ridge_scale"),
    ("add_intercept", "fit_intercept"),
    ("max_cd_iters", "max_iter"),
    ("cd_tol", "tol"),
    ("reset", "change_action"),
    // Task 196 (docs/PLAN.md §18, N14): a count of rows says so, so that a
    // clock form can come later under the plain name.
    ("update_every", "update_every_rows"),
    ("split_merge_every", "split_merge_every_rows"),
    ("permute_every", "permute_every_rows"),
    // N16: holt's level takes the spec's `half_life`, one knob under one
    // name.
    ("level_half_life", "half_life"),
];

/// The parameters renamed in one model whose old name another model keeps:
/// `rls`'s `ridge` is `delta` (docs/PLAN.md task 195, N11), and `ridge` is
/// still `ewridge`'s, `huber`'s and `quantile`'s. Named only where the
/// refused field's model takes the new name -- where serde's list of the
/// fields it expected holds it -- so a model with neither is told nothing
/// about another's rename.
pub const RENAMED_WHERE_EXPECTED: &[(&str, &str)] = &[("ridge", "delta")];

/// The values task 196 renamed (docs/PLAN.md §18): the model `type` tag
/// (N1, the builder's, the README's and the core's spelling) and an
/// `ew_cov` statistic (N10, beside `partial_corr`). A spec naming the old
/// one is refused naming the new one, as [`RENAMED`] does for a key.
pub const RENAMED_VALUES: &[(&str, &str)] = &[("ew_ridge", "ewridge"), ("lagcorr", "lag_corr")];
/// The parameters renamed after 1.0, `(old, new)`: a spec dict, a TOML file
/// or a windows config naming the old one is read as naming the new one,
/// with a deprecation notice ([`deprecation_notice`]; in Python a
/// `polars_online.PolarsOnlineDeprecationWarning`, on the command line a
/// line on stderr), until the next major version removes the entry and the
/// name is refused through [`RENAMED`] (docs/PLAN.md task 198; review round
/// 4, D2). Empty: every rename so far was made before 1.0, and stays
/// refused by name, task 144's rule. A rename after 1.0 goes here, beside
/// its twin in `python/polars_online/_warnings.py`'s `_DEPRECATED`.
pub const DEPRECATED: &[(&str, &str)] = &[];

/// `msg`, a deserialization error, with the rename named when the field it
/// refuses as unknown is an old name, or the model `type` it refuses as an
/// unknown variant is an old tag, and the table's key named when the field
/// is a builder keyword a table spells otherwise ([`BUILDER_KEYWORDS`]).
pub fn name_renamed(msg: &str) -> String {
    for (keyword, builder, key) in BUILDER_KEYWORDS {
        if msg.contains(&format!("unknown field `{keyword}`")) {
            return format!(
                "{msg}; `{keyword}` is {builder}'s builder keyword; a spec table says `{key}`"
            );
        }
    }
    // No citation: a wheel's user has no docs/PLAN.md (review 2026-10-06,
    // PC11).
    for (old, new) in RENAMED {
        if msg.contains(&format!("unknown field `{old}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    for (old, new) in RENAMED_WHERE_EXPECTED {
        if msg.contains(&format!("unknown field `{old}`")) && msg.contains(&format!("`{new}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    for (old, new) in RENAMED_VALUES {
        if msg.contains(&format!("unknown variant `{old}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    msg.to_string()
}

/// One model spec: common parameters (docs/PLAN.md §3) + the model. An
/// unknown key is refused, naming the keys there are (see [`ModelKind`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// Output struct column name.
    pub name: String,
    pub model: ModelKind,
    /// Columns to learn against, each a name, a table naming a column
    /// under another name, or a table holding a formula of the row's future
    /// ([`Targets`]); reads as the names. Optional for a model that learns from no
    /// target (`ModelKind::is_unsupervised`), where
    /// [`Self::fill_defaults`] mirrors `features[0]` the way the Python
    /// builders do (docs/ENHANCEMENTS.md E53); required otherwise.
    #[serde(default)]
    pub targets: Targets,
    pub features: Vec<String>,
    #[serde(default = "default_true")]
    pub fit_intercept: bool,
    #[serde(default)]
    pub clock: Option<String>,
    #[serde(default)]
    pub half_life: Option<SpanList>,
    #[serde(default)]
    pub lam: Option<f64>,
    /// Ceiling on the clock delta, in clock units: finite and positive,
    /// required with a clock and refused without one (task 160, PB2), where
    /// every row is one step. It caps the step between two rows a model
    /// learns from, so a run of skipped rows hands the row after it at most
    /// this much clock (review 2026-09-12, S3); a step the ceiling cut is a
    /// break, which clears `ew_cov`'s and `marginal`'s lagged co-moments,
    /// adjacency being broken -- under `embargo`, when the row after the
    /// break is learned (docs/PLAN.md task 153).
    #[serde(default)]
    pub gap_cap: Option<Span>,
    /// The clock stepping back within a group. Unset, every step back is
    /// refused, naming the row, the bank untouched. Given, in clock units:
    /// a step back no larger than this is a late row and is refused the
    /// same way; a larger one starts the model over. `0` starts over at
    /// every step back. Finite and `>= 0`; needs `clock` (task 120, decided
    /// 2026-09-28; one name since task 144).
    #[serde(default)]
    pub restart_after_step_back: Option<Span>,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub session_gap: Option<SessionGapSpec>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    /// Warmup in `n_eff` units. A scalar applies to every target; a list gives
    /// one threshold per target, in `targets` order (ENHANCEMENTS E7) — a
    /// 5-minute-ahead target and a 1-day-ahead target rarely deserve the same
    /// warmup. Warmup gates *output*, not learning: the model still updates
    /// from rows whose predictions are withheld. Each target's threshold is
    /// checked against that target's own weight -- the rows it was present
    /// on, inside the window under one -- where the model keeps one
    /// (`ewridge`, `lasso`, `kalman`, `huber`, `quantile`, `holt`, and since
    /// task 158 `sgd`, `pa`, `ftrl` and `rls`), and against the shared
    /// `n_eff` otherwise; the emitted `n_eff` is the shared weight either way
    /// (review 2026-09-12, S2).
    pub min_weight: Option<FloatOrList>,
    /// Withhold predictions until the decay window has filled this far
    /// toward steady state: `settled_frac = 1 − 2^(−T/h)`, `T` the decay
    /// time the models have seen (docs/WARMUP-AND-CONVERGENCE.md §2). A
    /// fraction in `[0, 1)`, so it cannot be set unreachably; `0`, the
    /// default, is off. Off because under a stationary process the
    /// mean-form fit is unbiased from its first row and its variance is
    /// `max_error_inflation`'s to gate; what this guards is a history that
    /// does not represent the process -- regimes or seasons the half-life
    /// was chosen to average across -- which only the user can judge. Needs
    /// a decay to settle toward.
    #[serde(default)]
    pub min_settled_frac: Option<Num>,
    /// Withhold predictions while the estimation error is expected to
    /// inflate the prediction error over the noise floor by more than this:
    /// `error_inflation = sqrt(1 + estimation variance / noise)` (§2.1).
    /// `ewridge`: `sqrt(1 + edf / n_kish)`, the effective degrees of freedom
    /// the last solve used over Kish's effective sample size behind the fit,
    /// default `sqrt(2)`, the estimation variance no larger than the noise
    /// being fitted. Off (`inf`) unless set on the others (docs/PLAN.md task
    /// 116): `rls`, `sqrt(1 + k_total / n_kish)`, its edf's bound under the
    /// fading prior; `lasso`, `sqrt(1 + df / n_kish)` per path point, `df`
    /// the active count plus the intercept; `kalman`, per row and exact,
    /// `sqrt(1 + z' P⁻ z / R)`, the prior predictive variance over the noise.
    /// Tracks the model, so adding a feature moves the gate with it, and
    /// reads Kish's `n` rather than the weight, so uneven weights withhold
    /// for longer. A ratio above 1; `inf` is off. Refused on every other
    /// model, which is left to `min_weight`.
    #[serde(default)]
    pub max_error_inflation: Option<Num>,
    /// Emit `error_inflation_<slot>`: the same ratio for *this* row's
    /// features, `sqrt(1 + h(x))`, so a row leaning on a direction the data
    /// never showed reads large where the stream average cannot see it.
    /// `ewridge`: `h = x' Σ̂⁻¹ x / n_kish`, the row's leverage against the
    /// factor its fit came from; `rls`: `‖R⁻ᵀ z‖² s₂ / s₁`, the same in sum
    /// form against its kept factor; `kalman`: `z' P⁻ z / R`, the gate's own
    /// value. One triangular solve or quadratic form a row, `O(k²)`, and
    /// `ewridge` keeps its factors -- which is why it is opt-in. Refused
    /// elsewhere, `lasso` included: it keeps no factor.
    #[serde(default)]
    pub emit_error_inflation: bool,
    /// Emit `se_coef`: each coefficient's standard error, on `coef`'s rows
    /// and laid out like it, in `coef`'s own units, the intercept's included
    /// -- `T Cov Tᵀ`'s diagonal, `T` the map `coef` is read out by
    /// (docs/PLAN.md task 116, F). `ewridge`: `Cov = σ̂² M`,
    /// `M = Σ̂⁻¹ / n_kish` (WARMUP §7.1), which leaves out the ridge's
    /// sandwich and so errs large; `rls`: `Cov = σ̂² (s₂ / s₁) A⁻¹`, the same
    /// in sum form, `O(k³)` on the `coef` schedule; `kalman`: `Cov = P`, its
    /// posterior, exact. `σ̂` is the slot's EW out-of-sample residual std as
    /// the row reads it (`sigma`), which carries the estimation error too,
    /// `sqrt(1 + h)` larger than the noise in warm-up, so the error errs
    /// large there; null before there is one. A report, not a gate. Refused
    /// for `lasso` (post-selection), `huber` and `quantile` (an
    /// M-estimator's covariance is a sandwich) and the gradient models (no
    /// second moment).
    #[serde(default)]
    pub emit_se_coef: bool,
    /// Emit `scored_clock` and `learned_clock` on every scored row: the row's
    /// own clock, and the clock of the newest row the models had learned
    /// from, at a positive weight, when the row was scored (docs/PLAN.md
    /// task 152). In the clock column's own type; the group's row index
    /// with no clock. One pair per spec, since every instance learns the
    /// same rows at the same time. Default false.
    #[serde(default)]
    pub emit_clocks: bool,
    /// When the `coef` field is filled, as `solve_every` schedules a solve
    /// (docs/PLAN.md task 178): a `coef` row once the clock has moved this
    /// far since the group's last one -- a number of the clock column's
    /// units, or a duration on a temporal clock, the clock measured as the
    /// models are stepped on it, a gap capped at `gap_cap` counting as the
    /// cap -- and `0` every row. With `max_rows_between_coefs` too,
    /// whichever comes first. Without a clock column the clock is the row's
    /// number, the first row being 1, so `N` is every `N` rows. With neither
    /// given, **each group's** last row within every chunk: one row per
    /// group per chunk, the one schedule that follows the chunking, while
    /// every other field, and `coef` under a cadence, is chunk-invariant.
    #[serde(default)]
    pub coef_every: Option<Span>,
    /// At most this many accepted rows between `coef` rows, rows of weight
    /// zero and rows with a null target included, as `coef_every` counted
    /// them before it read the clock; at least 1 (task 178).
    #[serde(default)]
    pub max_rows_between_coefs: Option<u32>,
    /// Emit `sigma_<slot>`: the EW standard deviation of this slot's
    /// out-of-sample residuals, read from the state *before* each row. Off by
    /// default because it widens the output struct. Its weight ages on every
    /// row, a row with no prediction or with weight 0 included (review
    /// 2026-09-12, N6), and under a `window_size` it is the window's, as the fit
    /// is (S1).
    #[serde(default)]
    pub emit_sigma: bool,
    /// Emit `zscore_<slot>` = `resid / sigma`: how surprising this row was, in
    /// units of the model's own recent error. Off by default.
    #[serde(default)]
    pub emit_zscore: bool,
    /// Emit `ic_<slot>`, `r2_<slot>` and `hit_rate_<slot>`: exponentially
    /// weighted evaluation metrics kept beside the model (ENHANCEMENTS E22).
    /// `polars_online.eval` computes the same things in Polars over collected
    /// output, which needs the whole frame; this is the O(state) version, so a
    /// long-running stream or the CLI can report how the fit is doing without
    /// keeping the rows.
    ///
    /// On a `sgd` or `ftrl` fit with `loss = "logistic"`, `pred` is a
    /// probability and `y` a 0/1 label rather than a signed regression
    /// target, and two of the three read differently (docs/PLAN.md task
    /// 76): `hit_rate` is accuracy at a 0.5 threshold instead of sign
    /// agreement (the sign test always agrees on two positive numbers, and
    /// read 1.0 for every such fit before this), `r2` is the Brier skill
    /// score against the running base rate, and `ic` is the point-biserial
    /// correlation between the probability and the label -- both under
    /// their usual names, since the formula does not change. There is no
    /// log loss here; `polars_online.eval.metrics(..., binary=True)` adds
    /// it over the collected frame (a streaming version would put a `ln`
    /// result into the state, which `docs/PLAN.md` §11a's B4 rule forbids).
    ///
    /// The sign test is about zero for every target, so a target that sits
    /// about 1, such as a plain ratio of two prices, reads 1.0 whatever the
    /// fit: two positive numbers always agree. A return is better a
    /// difference or a log ratio, which sit about zero.
    #[serde(default)]
    pub emit_metrics: bool,
    /// Emit `lo_<slot>`, `hi_<slot>` and `coverage_<slot>`: an
    /// adaptive conformal interval `pred ± q` at this coverage level, with
    /// the realized coverage beside it (ENHANCEMENTS E36). `q` is a tracked
    /// quantile of `|resid|` — `q ← max(0, q + rate·sigma·(w/w̄)·(miss −
    /// α))`, `miss = 1{|resid| > q}`, `α = 1 − coverage`, `w̄` the EW mean
    /// weight — so the long-run coverage weighted by `w/w̄` is the target
    /// whatever the residual distribution is and however it moves, where
    /// `sigma` gives a Gaussian interval (review round 4, CA5). Read before
    /// the row, like every other diagnostic. A coverage level strictly
    /// between 0 and 1.
    #[serde(default)]
    pub conformal: Option<f64>,
    /// Step of the conformal radius per unit of the slot's `sigma`. Default
    /// 0.05: a miss widens the interval by `0.05·sigma·(1 − α)`, a hit
    /// narrows it by `0.05·sigma·α`, each times the row's weight over the
    /// scored rows' EW mean weight (docs/PLAN.md task 147). `"auto"`
    /// divides 0.05 by `1 + 0.05·g(α)·n̄/c`, `n̄` the EW mean of the rows
    /// waiting for their labels at each release (task 342b): 0.05 exactly
    /// where nothing waits.
    #[serde(default)]
    pub conformal_rate: Option<ConformalRate>,
    /// Emit `abs_resid_q<p>_<slot>` for each level in `resid_quantiles`: the
    /// exponentially weighted quantile of `|resid|` at the model's half-life
    /// and each row's weight (ENHANCEMENTS E23, docs/PLAN.md task 146), from
    /// one decaying sketch per slot within `tanh(1/128)` of the exact one --
    /// a distribution-free interval where `sigma` only gives a Gaussian one.
    #[serde(default)]
    pub resid_quantiles: Option<Vec<f64>>,
    /// Emit `autocorr_<slot>`: EW lag-`resid_autocorr_lag` autocorrelation of
    /// the out-of-sample residuals. A residual stream should look like noise;
    /// autocorrelation is the classic sign that it does not.
    #[serde(default)]
    pub emit_autocorr: bool,
    /// Lag for `emit_autocorr`. Default 1.
    #[serde(default)]
    pub resid_autocorr_lag: Option<usize>,
    /// Emit `drift_<slot>`: a Page-Hinkley detector on each slot's absolute
    /// out-of-sample residual, true on the row where a break is detected.
    /// Under `embargo` a residual reaches the detector when its label is
    /// released, so the flag is on the row whose clock released the label
    /// that tripped it; that label's own row is out already (task 160,
    /// PB1). Complements the half-life: decay forgets smoothly and always,
    /// drift detection notices a break and says so.
    #[serde(default)]
    pub emit_drift: bool,
    /// Change magnitude the drift detector tolerates before accumulating,
    /// in units of the slot's own EW residual std. Default 0.5.
    #[serde(default)]
    pub drift_delta: Option<f64>,
    /// Accumulated excess that counts as drift, in `sigma` times clock units:
    /// the detector sums each row's excess times its clock step. A number of
    /// the clock column's units, or a duration on a temporal clock (`"20m"`
    /// is one `sigma` of excess held for twenty minutes). Default 20 without
    /// a clock column, where a row is one unit and it is the classic test;
    /// required with one, as `gap_cap` is, since no number means the same
    /// evidence on every clock (task 168; review 2026-10-05, CC1).
    #[serde(default)]
    pub drift_threshold: Option<Span>,
    /// What a detection does besides setting the flag: `"flag"` (default) or
    /// `"reset"`, which restarts this stream's models and their residual
    /// diagnostics on the flagged row, before it is scored under `embargo`
    /// and after it is learned from otherwise. Unlike a clock reset it
    /// keeps the rows `embargo` holds: they teach the restarted models as
    /// they are released.
    #[serde(default)]
    pub drift_action: Option<String>,
    /// Emit `calibration_slope_<slot>`, `calibration_intercept_<slot>` and
    /// `calibration_wald_<slot>`: Mincer and Zarnowitz's (1969) regression
    /// of the outcome on its out-of-sample prediction, `y = a + b·pred`, and
    /// Wald's statistic for `a = 0, b = 1` at Kish's size (docs/PLAN.md task
    /// 221 (a); [`online_core::Calibration`] has the update). The slope is
    /// the multiplier to put on a prediction. Read before the row, like
    /// every residual diagnostic, at the memory `calibration_half_life`.
    #[serde(default)]
    pub emit_calibration: bool,
    /// The calibration's memory, in clock units: four times the fit's own
    /// memory unless set ([`Spec::fit_memory`]; [`Spec::memory_multiple`]
    /// has the measurement), `inf` the run-once form, which forgets nothing.
    #[serde(default)]
    pub calibration_half_life: Option<Span>,
    /// Emit `studentized_<slot>`, `cusum_<slot>`, `cusum_sq_<slot>` and
    /// `break_wald_<slot>` (docs/PLAN.md task 221 (b); the update is
    /// [`online_core::Breaks`]'s and [`online_core::TwinFit`]'s): the row's
    /// predictive studentized residual `resid / (sigma · error_inflation)`
    /// -- 1 in place of the inflation on a model without one -- the
    /// standardized CUSUM and CUSUM of squares of those before it at
    /// `breaks_half_life` (each over its null: the CUSUM of squares by the
    /// measured `E[z⁴] − 1`, a windowed one by its null's share; task 232
    /// (4)), run once the CUSUM the path of Brown, Durbin and Evans (1975)
    /// and with a memory moving sums, and the Wald distance between
    /// two least-squares fits of the target, one at that memory and one at
    /// four times it, null run once, from `2k` learned rows on for `k`
    /// coefficients, below which its variance is singular.
    #[serde(default)]
    pub emit_breaks: bool,
    /// The breaks' memory, in clock units: the fit's own memory unless set
    /// ([`Spec::fit_memory`]), `inf` the run-once form.
    #[serde(default)]
    pub breaks_half_life: Option<Span>,
    /// Emit `se_coef_hc0` and, with a lag, `se_coef_hac` on `coef`'s rows,
    /// laid out like it (docs/PLAN.md task 221 (c); the update is
    /// [`online_core::Sandwich`]'s): each coefficient's standard error from
    /// the least-squares sandwich `B⁻¹ M B⁻¹` at `robust_se_half_life`, the
    /// meat `Σ ω² e² z z'` (White's HC0) and, under `robust_se_lags`, Newey
    /// and West's Bartlett-weighted lag products, `e` the row's
    /// out-of-sample residual. `ewridge` and `rls`, the least-squares fits;
    /// the ridge is left out, as `se_coef` leaves it out.
    #[serde(default)]
    pub emit_robust_se: bool,
    /// The sandwich's memory, in clock units: the fit's own memory unless set
    /// ([`Spec::fit_memory`]), `inf` the run-once form.
    #[serde(default)]
    pub robust_se_half_life: Option<Span>,
    /// Newey and West's lags, in rows: twice the target's horizon unless
    /// set -- `embargo` in rows on a spec with no clock column -- and `0`,
    /// HC0 alone, otherwise ([`Spec::robust_se_lags_or_default`] has the
    /// measurement behind twice).
    #[serde(default)]
    pub robust_se_lags: Option<usize>,
    /// Emit `ljung_box_<slot>`, `breusch_pagan_<slot>` and `reset_<slot>`
    /// (docs/PLAN.md task 221 (d); [`online_core::Specification`] has the
    /// update): Ljung and Box's `Q` over the residuals' lags past the
    /// target's horizon, Breusch and Pagan's `n R²` of `e²` on the slot's
    /// features, and RESET's Lagrange multiplier for `p²` and `p³`, each at
    /// `specification_half_life` and Kish's size.
    #[serde(default)]
    pub emit_specification: bool,
    /// The specification tests' memory, in clock units (the default is
    /// [`Spec::memory_multiple`]'s), `inf` the run-once form.
    #[serde(default)]
    pub specification_half_life: Option<Span>,
    /// Ljung and Box's lags, `L`, counted past the horizon: 10 unless set.
    #[serde(default)]
    pub ljung_box_lags: Option<usize>,
    /// The rows a look-ahead target's residuals overlap, its horizon `h`
    /// ([`Spec::horizon`]; task 232 (3)): `embargo` in rows unless set on a
    /// spec with no clock column, and 0 unless set on one with a clock.
    /// Under it the calibration's, Breusch and Pagan's and RESET's tests
    /// take Newey and West's variance at `2h` lags, the CUSUMs and
    /// `break_wald` the long-run variance, Ljung and Box skip the lags
    /// inside it, and `se_coef_hac`'s lags default to `2h`. A count of
    /// rows, named `*_rows` as `warm_rows` and `window_rows` are: a row
    /// count is what the residuals overlap by, whatever the clock.
    #[serde(default)]
    pub horizon_rows: Option<usize>,
    /// Emit `skew_<slot>`, `kurtosis_<slot>` and `jarque_bera_<slot>`
    /// (docs/PLAN.md task 221 (e); [`online_core::Tails`] has the update):
    /// the EW skewness, excess kurtosis and Jarque and Bera's statistic of
    /// the recursive residuals, `resid / error_inflation`, at
    /// `tails_half_life` and Kish's size.
    #[serde(default)]
    pub emit_tails: bool,
    /// The tails' memory, in clock units (the default is
    /// [`Spec::memory_multiple`]'s), `inf` the run-once form.
    #[serde(default)]
    pub tails_half_life: Option<Span>,
    /// Emit `influence_<slot>` (docs/PLAN.md task 221 (f);
    /// [`online_core::Influence`] has the update): the row's move of the
    /// fit in the fit's own metric, an online DFFITS, `(v / s)·sqrt(h)` with
    /// `h` the row's leverage, `v` its recursive residual and `s` the spread
    /// of those before it at `influence_half_life`. `ewridge`, `rls` and
    /// `kalman`, which read a row's leverage.
    #[serde(default)]
    pub emit_influence: bool,
    /// The influence's scale's memory, in clock units (the default is
    /// [`Spec::memory_multiple`]'s), `inf` the run-once form.
    #[serde(default)]
    pub influence_half_life: Option<Span>,
    /// Emit `spread_ratio_<feature>` and `mean_shift_<feature>`, per
    /// instance (docs/PLAN.md task 221 (g); [`online_core::FeatureHealth`]
    /// has the update): each feature's EW spread at
    /// `feature_health_half_life` over its spread at four times it, and the
    /// two means' difference in units of the longer one's spread -- a
    /// feature gone quiet, or moved.
    #[serde(default)]
    pub emit_feature_health: bool,
    /// The feature health's fast memory, in clock units (the default is
    /// [`Spec::memory_multiple`]'s); its slow memory is four times it.
    #[serde(default)]
    pub feature_health_half_life: Option<Span>,
    /// Emit `pred_<target>__averaged`: an exponentially weighted average of
    /// every slot's prediction, with weights `softmax(−eta · σ²/σ²_best)`,
    /// each slot's EW squared error as a ratio to the best slot's
    /// (ENHANCEMENTS E14; a ratio since the code review's S23, so `eta` does
    /// not depend on the target's units). The soft counterpart of
    /// `emit_selected`: averaging hedges where selection commits, which is
    /// usually the better trade when several slots are close. The weights
    /// come from each slot's EW *mean* squared error, so they stay bounded
    /// and do not sharpen as rows accumulate, as a weighting by summed
    /// losses (river's `EWARegressor`) does; and a slot with no prediction
    /// or no `sigma` on the row is left out of that row's average, which is
    /// null only when every slot is (code review of 2026-09-12, S23).
    #[serde(default)]
    pub emit_averaged: bool,
    /// Sharpness of the averaging weights, `exp(−eta · (σ²/σ²_best − 1))`:
    /// at 1, a slot whose error is twice the best's weighs `e⁻¹` of it. Large
    /// values approach `emit_selected`'s argmin, and `"inf"` is it, a tie
    /// shared; small values approach an equal-weight mean. Default 1.
    #[serde(default)]
    pub average_eta: Option<Num>,
    /// Emit `selected_<target>` and `pred_<target>__selected`: online model
    /// selection across every grid slot for that target (ridge values, feature
    /// sets and half-lives), by lowest EW out-of-sample error. Generalizes the
    /// lasso's `lam_selected`. Requires more than one slot per target.
    #[serde(default)]
    pub emit_selected: bool,
    /// Hold each row back from *learning* until the model's clock has moved
    /// `embargo` further on, in the same units the decay uses
    /// (docs/ENHANCEMENTS.md E47). `None` (the default) learns from a row
    /// where it sits.
    ///
    /// A target that is a forward quantity over `h` clock units is not known
    /// at the row it sits on. Learning it there hands the model `h` of the
    /// future before it predicts the rows in between, and every
    /// "out-of-sample" number after that is contaminated -- with an
    /// autocorrelated feature even a pure noise column then shows a
    /// correlation with its target. With a delay the row is buffered, and
    /// released into the model, the residual, `sigma`, the metrics, drift,
    /// the conformal interval, `n_eff` and `min_weight` only once its label
    /// would really have been known.
    ///
    /// The delay counts the time that passed: the clock column's own steps,
    /// skipped rows' time included, not the capped delta the models decay by
    /// (`gap_cap` and `session_gap` say how much a model forgets across a
    /// break, not how long it lasted), and the session's gap where a session
    /// change restarts the clock (docs/PLAN.md task 153). A row exactly the
    /// delay later releases a row, and the time is held exactly: in integer
    /// nanoseconds on a temporal clock, by one subtraction of the two rows'
    /// raw values on a number clock (task 176). Release depends on the rows
    /// alone, which makes it chunk-invariant. With no `clock`
    /// column that is one unit per row of the group, a skipped row included,
    /// so `embargo = 20` is twenty rows. A reset drops the buffer (the
    /// state it would teach is gone). A break releases nothing early: its
    /// events -- the lag rings' clear, `session_shrink`'s blend -- wait with
    /// the row after it and run when that row is learned, so the models run
    /// one delay behind, events included.
    #[serde(default)]
    pub embargo: Option<Span>,
    /// The column holding each row's label time, on the clock's own type
    /// (docs/PLAN.md task 341): the row is scored where it sits, from the
    /// labels known by then, and held until the first row whose clock is
    /// at or past its label time, which releases it -- its label scored
    /// against the prediction, interval and spread it was served with --
    /// to be learned at its label time. `embargo` is the case where every
    /// row's label time is its clock plus one delay. A list gives one
    /// column per target; this version refuses a list whose entries
    /// differ. Not written while `None`, so no other spec's bytes move.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_clock: Option<LabelClock>,
    /// The most rows a group may hold for their label times; `None` holds
    /// any number. Past it, `held_overflow` decides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_held_rows: Option<u64>,
    /// `"refuse"` (the default) or `"drop_oldest"` ([`HELD_OVERFLOW`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub held_overflow: Option<String>,
    /// One state per key.
    #[serde(default)]
    pub group: Option<String>,
    /// Emit a group's accumulators when the stream can prove no further row
    /// will join it, and drop the stream (docs/ENHANCEMENTS.md E54).
    ///
    /// `"monotone"`: the group column is non-decreasing, so a key smaller
    /// than the largest key of the chunk just fed is finished. Requires
    /// `group`, and refuses a chunk whose keys are out of order, naming the
    /// row.
    ///
    /// `"session"`: a group's span ends where its `session` value changes.
    /// Requires `session`; refused with `session_gap`, which is a second
    /// prescription for the same event (the close *is* the reset, with an
    /// emission).
    ///
    /// `None` (the default) keeps every group for the life of the bank,
    /// which is what makes a bank over an unbounded key space grow without
    /// bound. The closed rows queue in the bank and are read with
    /// `closed_groups()`, written to the `closed_groups` sidecar by a run,
    /// or both.
    ///
    /// Refused with `embargo`: a closed group cannot release the rows it
    /// is holding, so its row would differ from the `gram()` a driver reads
    /// at the same point -- which is the one thing the closed row promises.
    #[serde(default)]
    pub group_close: Option<String>,
}

impl Spec {
    /// Fill in what a spec is allowed to leave out
    /// (docs/ENHANCEMENTS.md E53), so that everything downstream sees a spec
    /// with every field set.
    ///
    /// Two rules. `ew_cov`, `kmeans` and `micro` learn from no target,
    /// and their `targets` mirror `features[0]` for the plumbing's sake. The
    /// Python builders write that line; a TOML author should not have to
    /// invent a target for a model that has none, and a spec that left it
    /// out used to be refused with "targets must be non-empty". Filling it
    /// with the same value the builders use is what makes a spec written in
    /// TOML byte-identical to the same spec written in Python -- so a state
    /// saved from one resumes under the other.
    ///
    /// And `drift_action` is spelled out as `"flag"`, which is what it means
    /// when it is absent. It is the one other field the Python builders write
    /// and a TOML author would not, and leaving it unfilled would make the
    /// two surfaces' specs differ in a byte for no reason at all.
    ///
    /// Idempotent, and a no-op for a spec that already says both.
    pub fn fill_defaults(&mut self) {
        if self.targets.is_empty() && self.model.is_unsupervised() {
            // A column the model reads from the slot fills it, as the
            // builders write it. `features[0]` went there, and was then read
            // as the hazard or the exogenous series (review 2026-09-12, C20).
            if let Some(col) = self.model.targets_slot_column() {
                self.targets = vec![col.to_string()].into();
            } else if let Some(first) = self.features.first() {
                self.targets = vec![first.clone()].into();
            }
        }
        if self.drift_action.is_none() {
            self.drift_action = Some("flag".into());
        }
    }

    /// Does this spec close a group when its `session` value changes
    /// (docs/ENHANCEMENTS.md E54)?
    pub fn closes_on_session(&self) -> bool {
        self.group_close.as_deref() == Some("session")
    }

    /// Does this spec close a group once a larger key has been seen?
    pub fn closes_monotone(&self) -> bool {
        self.group_close.as_deref() == Some("monotone")
    }

    pub fn k(&self) -> usize {
        self.features.len()
    }

    pub fn m(&self) -> usize {
        self.targets.len()
    }

    /// Decay values, one per half-life grid entry (one model instance each).
    pub fn decays(&self) -> Result<Vec<(String, Decay)>, String> {
        // A model that takes no decay has one undecayed instance whatever
        // `half_life` or `lam` hold: its own check refuses either given, and
        // a second refusal here, of their values, would ask the reader to
        // fix a setting the first tells them to leave out.
        let decays_nothing = match &self.model {
            ModelKind::SeqTest { .. }
            | ModelKind::Rcov { .. }
            | ModelKind::Bocpd { .. }
            | ModelKind::Audit { .. } => true,
            ModelKind::CorrChange { scalar, .. } => !scalar.unwrap_or(false),
            _ => false,
        };
        if decays_nothing {
            return Ok(vec![(String::new(), Decay::Halflife(f64::INFINITY))]);
        }
        match (&self.half_life, self.lam) {
            (Some(_), Some(_)) => Err(format!(
                "spec {:?}: half_life and lam cannot both be set: each sets the decay. Leave \
                 out half_life, or lam.",
                self.name
            )),
            (None, None) => match &self.model {
                // An e-process does not forget: its validity is the product
                // of every bet made, so there is nothing a decay could apply
                // to. One undecayed instance, and `half_life`/`lam` refused
                // below rather than ignored.
                // A realised covariance is a sum over a block, not a
                // decayed mean: the block boundary is `group_close`'s, and
                // `half_life`/`lam` are refused below rather than ignored.
                ModelKind::SeqTest { .. }
                | ModelKind::Rcov { .. }
                | ModelKind::CorrChange { .. }
                | ModelKind::Bocpd { .. }
                | ModelKind::Audit { .. } => {
                    Ok(vec![(String::new(), Decay::Halflife(f64::INFINITY))])
                }
                // A window with no half-life is the common case of a flat
                // window, so it is named: `half_life` "inf" weighs each row
                // in the window alike.
                _ if self.model.keeps_window_ring() => Err(format!(
                    "spec {:?}: a hard window (window_size) needs half_life, but half_life is not \
                     set. Set half_life to \"inf\" (in Python, float(\"inf\")) to weigh every \
                     row in the hard window alike, or to a finite value to decay the older rows \
                     within it.",
                    self.name
                )),
                _ => Err(format!(
                    "spec {:?}: this model needs a half_life, and none is given. Set half_life \
                     to a value in clock units, or to \"inf\" (in Python, float(\"inf\")) for \
                     no decay.",
                    self.name
                )),
            },
            (None, Some(l)) => {
                if !(0.0 < l && l <= 1.0) {
                    return Err(format!(
                        "spec {:?}: lam must be above 0 and at most 1, but it is {l}. Set lam to \
                         a value above 0 and below 1, or to 1 for no decay.",
                        self.name
                    ));
                }
                Ok(vec![(String::new(), Decay::Lam(l))])
            }
            (Some(h), None) => {
                let hs = h.to_vec();
                // An empty grid built no model instance at all, and the
                // output struct had no field (task 160, PB3).
                if hs.is_empty() {
                    return Err(format!(
                        "spec {:?}: half_life must list at least one half-life, but it is empty. \
                         Set half_life to one value, or to a list of them for a grid.",
                        self.name
                    ));
                }
                // `!(h > 0)` rather than `h <= 0` so NaN is refused too: it
                // decays every accumulator to NaN and nothing washes it out.
                let bad: Vec<String> = h
                    .spans()
                    .iter()
                    .filter(|s| !positive(s.value()))
                    .map(Span::label)
                    .collect();
                if !bad.is_empty() {
                    let (is, set) = match (hs.len(), bad.as_slice()) {
                        (1, [one]) => (format!("it is {one}"), "half_life"),
                        (_, [one]) => (format!("one of its values is {one}"), "that value"),
                        (_, [rest @ .., last]) => (
                            format!("its values {} and {last} are not", rest.join(", ")),
                            "each",
                        ),
                        (_, []) => unreachable!("bad is not empty"),
                    };
                    return Err(format!(
                        "spec {:?}: half_life must be above 0, but {is}. Set {set} to a positive \
                         number of clock units, or to \"inf\" (in Python, float(\"inf\")) for \
                         no decay.",
                        self.name
                    ));
                }
                if let Some(dup) = first_duplicate(&hs) {
                    // "5m" and "300s" are one half-life written two ways:
                    // each spelling is named, so the reader finds both.
                    let mut labels: Vec<String> = Vec::new();
                    for s in h.spans().iter().filter(|s| s.value() == dup) {
                        let l = s.label();
                        if !labels.contains(&l) {
                            labels.push(l);
                        }
                    }
                    let lists = match labels.as_slice() {
                        [one] => format!("lists {one} more than once"),
                        [rest @ .., last] => format!(
                            "lists {} and {last}, which are the same half-life",
                            rest.join(", ")
                        ),
                        [] => format!("lists {} more than once", num_label(dup)),
                    };
                    return Err(format!(
                        "spec {:?}: half_life {lists}, and each half-life is one instance (the \
                         part of the model that fits that half-life), so its copies would be one \
                         instance. List each half-life once.",
                        self.name
                    ));
                }
                if hs.len() == 1 {
                    Ok(vec![(String::new(), Decay::Halflife(hs[0]))])
                } else {
                    // A grid names each instance by its half-life as written:
                    // `@h600` for a number, `@h10m` for a duration.
                    Ok(h.spans()
                        .iter()
                        .map(|s| (format!("@h{}", s.label()), Decay::Halflife(s.value())))
                        .collect())
                }
            }
        }
    }

    /// `gap_cap` and `session_gap` in integer nanoseconds where they are
    /// durations, which only a temporal clock reads: what a row's stamp, its
    /// decayed clock held exactly, caps a step at ([`ExactCaps`],
    /// docs/PLAN.md task 175).
    pub fn exact_caps(&self) -> ExactCaps {
        let nanos = |s: &Span| match s {
            Span::Duration(d) => Some(d.nanos),
            Span::Units(_) => None,
        };
        ExactCaps {
            gap_cap_ns: self.gap_cap.as_ref().and_then(nanos),
            session_gap_ns: match &self.session_gap {
                Some(SessionGapSpec::Gap(g)) => nanos(g),
                _ => None,
            },
        }
    }

    pub fn clock_cfg(&self) -> Result<ClockCfg, String> {
        clock_cfg_of(&ClockPolicy {
            who: &format!("spec {:?}", self.name),
            clock: self.clock.as_deref(),
            gap_cap: self.gap_cap.as_ref(),
            restart_after_step_back: self.restart_after_step_back.as_ref(),
            session: self.session.as_deref(),
            session_gap: self.session_gap.as_ref(),
            spec_closes_on_session: Some(self.closes_on_session()),
            group: self.group.as_deref(),
            looks_ahead: self.targets.defs().iter().any(|t| {
                t.formula
                    .as_ref()
                    .is_some_and(crate::formula::Node::has_forward_operator)
            }),
        })
    }

    /// Threshold per target, in `targets` order.
    pub fn min_periods_per_target(&self) -> Vec<f64> {
        match &self.min_weight {
            None => vec![self.default_min_periods(); self.m()],
            Some(FloatOrList::Float(v)) => vec![v.0; self.m()],
            Some(FloatOrList::List(v)) => v.iter().map(|n| n.0).collect(),
        }
    }

    fn default_min_periods(&self) -> f64 {
        match self.model {
            // An e-value is valid from the first row (it is 1 before it).
            ModelKind::SeqTest { .. } => 0.0,
            // A pair's statistics are over two columns whatever the feature
            // count: two rows give a correlation of ±1, three the first one
            // with any content.
            ModelKind::Marginal { .. } => 3.0,
            // The standardiser needs a variance per feature before the row
            // can be standardised at all; three rows is where it has one.
            ModelKind::Deco { .. } => 3.0,
            // Nothing is gated: `rcov` reports nothing per row.
            ModelKind::Rcov { .. } => 0.0,
            // The states are seeded from `warm_rows`, which is the real
            // gate; `min_weight` on top of it would be a second one.
            ModelKind::Hmm { .. } => 0.0,
            // The span or the windows are the gate.
            ModelKind::CorrChange { .. } => 0.0,
            // On row one the run-length posterior has no run older than
            // one, so `P(r <= 1)` is 1 whatever the row: one row of
            // warm-up, and the prior is the gate after it.
            ModelKind::Bocpd { .. } => 1.0,
            // Nothing is gated: `audit` reports nothing per row.
            ModelKind::Audit { .. } => 0.0,
            // The noise gate is this model's readiness gate
            // (docs/WARMUP-AND-CONVERGENCE.md §2.1): `max_error_inflation`
            // tracks the model where a count of weight cannot, and the
            // `k + 1` rows its first solve needs are the model's own floor
            // (`build_one`), not a setting. An explicit value still floors.
            ModelKind::EwRidge { .. } => 0.0,
            // No intercept to count: `fit_intercept` has nothing to act on in
            // these, and counted, it moved the first reported row (review
            // 2026-09-12, S22). `k + 1` is what the builders' default gave.
            ModelKind::EwCov { .. }
            | ModelKind::KMeans { .. }
            | ModelKind::Micro { .. }
            | ModelKind::EwClass { .. }
            | ModelKind::Holt { .. } => (self.k() + 1) as f64,
            _ => (self.k() + usize::from(self.fit_intercept)) as f64,
        }
    }

    /// The threshold the *model* uses: the smallest across targets, so a model
    /// starts predicting as soon as any target is ready. Per-target gating of
    /// the reported values happens in the stream layer.
    pub fn min_periods_or_default(&self) -> f64 {
        self.min_periods_per_target()
            .into_iter()
            .fold(f64::INFINITY, f64::min)
    }

    /// The settled-fraction gate, `0` (off) unless set
    /// (docs/WARMUP-AND-CONVERGENCE.md §4.1.1).
    pub fn min_settled_frac_or_default(&self) -> f64 {
        self.min_settled_frac.map_or(0.0, |v| v.0)
    }

    /// The noise gate: on `ewridge`, `sqrt(2)` unless set -- the estimation
    /// variance no larger than the noise (§2), the gate that replaced its
    /// `k + 1` floor -- and on every other model off, infinite, unless set:
    /// those keep their `min_weight` defaults, and no default moves
    /// (docs/PLAN.md task 116, D8).
    pub fn max_error_inflation_or_default(&self) -> f64 {
        let default = if matches!(self.model, ModelKind::EwRidge { .. }) {
            2f64.sqrt()
        } else {
            f64::INFINITY
        };
        self.max_error_inflation.map_or(default, |v| v.0)
    }

    // The diagnostics' defaults, each read here and nowhere else, so the
    // stream that builds a diagnostic and `crate::defaults`, which the API
    // snapshot pins them through, read one value (review 2026-10-06, AP1,
    // DB4).

    /// The drift detector's tolerance, `0.5` of the slot's residual std
    /// unless set.
    pub fn drift_delta_or_default(&self) -> f64 {
        self.drift_delta.unwrap_or(0.5)
    }

    /// The drift detector's threshold in `sigma` times clock units: `20`
    /// unless set, which only a spec without a clock column can leave it
    /// (task 168).
    pub fn drift_threshold_or_default(&self) -> f64 {
        self.drift_threshold.as_ref().map_or(20.0, Span::value)
    }

    /// The conformal radius' step per unit of `sigma` with nothing waiting,
    /// `0.05` unless set (and under `"auto"`).
    pub fn conformal_rate_or_default(&self) -> f64 {
        self.conformal_rate
            .map_or(ConformalRate::DEFAULT, |r| r.base())
    }

    /// Whether the conformal step reads the rows waiting for their labels
    /// (`conformal_rate = "auto"`).
    pub fn conformal_rate_auto(&self) -> bool {
        self.conformal_rate.is_some_and(|r| r.is_auto())
    }

    /// The lag of `emit_autocorr`, `1` unless set.
    pub fn resid_autocorr_lag_or_default(&self) -> usize {
        self.resid_autocorr_lag.unwrap_or(1)
    }

    /// The sharpness of `emit_averaged`'s weights, `1` unless set.
    pub fn average_eta_or_default(&self) -> f64 {
        self.average_eta.map_or(1.0, |n| n.0)
    }

    /// Whether any of this spec's model instances forgets: a finite
    /// half-life, or a `lam` below 1. What `settled_frac` needs to be a
    /// fraction of.
    pub fn has_decay(&self) -> bool {
        self.decays().is_ok_and(|ds| {
            ds.iter().any(|(_, d)| match d {
                Decay::Halflife(h) => h.is_finite(),
                Decay::Lam(l) => *l < 1.0,
            })
        })
    }

    /// The weight-share cadence (docs/PLAN.md task 115 (b)), the default
    /// where `solve_every` is left out under a finite half-life: a solve once
    /// the weight learned since the last reaches `ln 2 / 50` of the weight
    /// the fit holds. `None` with an explicit `solve_every`, which keeps its
    /// clock, and under `lam` or an infinite half-life, which solve every row.
    pub fn solve_share_default(&self, solve_every: Option<&Span>, decay: Decay) -> Option<f64> {
        match (solve_every, decay) {
            (None, Decay::Halflife(h)) if h.is_finite() => Some(online_core::DEFAULT_SOLVE_SHARE),
            _ => None,
        }
    }

    /// The clock cadence a spec without `solve_every` carries: half-life/50, or
    /// 0 (every row) under `lam` or an infinite half-life. Under a finite
    /// half-life the solves go by weight instead ([`Self::solve_share_default`],
    /// task 115 (b)); this value stays in the cfg, where `gram_block_rows`
    /// reads whether there is a cadence at all.
    pub fn solve_every_default(&self, decay: Decay) -> f64 {
        match decay {
            Decay::Halflife(h) if h.is_finite() => h / 50.0,
            _ => 0.0, // lam decay / infinite half_life: solve every row
        }
    }

    /// Everything a spec meets before a bank runs it, in one place: the
    /// defaults it may leave out ([`Self::fill_defaults`]), its own rules
    /// ([`Self::validate`]) and its models' (`build_models`, which builds each
    /// instance and drops it). Every door a spec comes in by -- the bank, a
    /// run config, the builders' check, `output_fields`, `output_index` and
    /// `coef_fields` -- goes through this, so a spec is accepted or refused
    /// alike at each. They met one, two or all three of the steps, by door
    /// (review 2026-09-12, S25).
    pub fn check(&mut self) -> Result<(), String> {
        self.fill_defaults();
        self.validate()?;
        crate::stream::build_models(self)?;
        // Two outputs rendered to one field name, named by their inputs
        // (task 144); the bank's tripwire for a spec built some other way.
        match crate::bank::duplicate_field(self) {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Every problem the spec's own checks find, in one refusal
    /// ([`render_refusal`]). The checks of the model's own settings, which
    /// the model makes as it is built (`build_models`), run once these pass:
    /// building reads what these check.
    pub fn validate(&self) -> Result<(), String> {
        let mut no = online_core::Refusals::new();
        if let Err(e) = self.validate_each(&mut no)
            && !no.stopped()
        {
            // A check that still returns at its problem stops the rest.
            no.take(Err(e));
            let _ = no.stop_here();
        }
        render_refusal(&self.name, self.model.kind_name(), &no)
    }

    /// The spec's checks, each problem added to `no`; an `Err` only where a
    /// check stops (`no.stop_here()`), since the checks after it read what
    /// it refuses.
    fn validate_each(&self, no: &mut online_core::Refusals) -> Result<(), String> {
        let name = &self.name;
        // A parameter that applies only with a switch on, given with it off,
        // does nothing: refused rather than ignored (review 2026-09-12, S22).
        let needs = |key: &str, switch: &str| {
            format!(
                "spec {name:?}: {key} applies only when {switch} is on, but {switch} is off. \
                 Turn on {switch}, or leave out {key}."
            )
        };
        // Clock parameters given as durations and as numbers at once, or
        // durations with no clock to measure them (task 88). Every check
        // after this reads a clock parameter's value, which means nothing
        // until the clock that measures it is settled: a stop.
        if !no.take(self.clock_scale().map(|_| ())) {
            no.stop_here()?;
        }
        // The bank's tables put `spec` and `group` columns beside a struct
        // named after the spec (`ModelBank.last_row`), and an empty name names
        // no struct at all (review 2026-09-12, S7).
        if self.name.is_empty() {
            no.refuse(format!(
                "spec {name:?}: the spec's name must not be empty, since it names the spec's \
                 output column. Give the spec a name."
            ));
        } else if self.name == "spec" || self.name == "group" {
            no.refuse(format!(
                "spec {name:?}: the spec's name must not be \"spec\" or \"group\", which the \
                 bank's tables use for their own columns, but it is {name:?}. Give the spec \
                 another name."
            ));
        }
        // The columns: a spec with no target, or a model with no feature to
        // fit on, is a stop, since the checks after these read the columns
        // (a list per target, a leak, a diagnostic of the features).
        let mut columns_missing = false;
        if self.targets.is_empty() {
            // An unsupervised spec is allowed to omit them, and
            // `fill_defaults` mirrors `features[0]` (E53) -- so an empty list
            // here means either a model that needs targets, or one that would
            // have had them filled but names no features either.
            columns_missing = true;
            no.refuse(if self.model.is_unsupervised() {
                format!(
                    "spec {name:?}: features must name at least one column, but it is empty: {} \
                     learns from no target, and fills its targets from its first feature. Name \
                     at least one column in features.",
                    self.model.kind_name()
                )
            } else {
                format!(
                    "spec {name:?}: targets must name at least one column, but it is empty. Name \
                     the columns to predict in targets."
                )
            });
        }
        // An empty name names no column and writes fields called `pred_` and
        // `resid_`; `po.target("")` and an empty table name were refused, a
        // plain `""` was not (review 2026-10-06, YA5).
        if self.targets.defs().iter().any(|t| t.name.is_empty()) {
            no.refuse(format!(
                "spec {name:?}: every target needs a name, but one of them is \"\", which would \
                 name its fields pred_ and resid_. Give each target a name."
            ));
        }
        if let Some(gc) = &self.group_close {
            const GROUP_CLOSE: [&str; 2] = ["monotone", "session"];
            if !GROUP_CLOSE.contains(&gc.as_str()) {
                no.refuse(spec_words_refusal(
                    &self.name,
                    "group_close",
                    &GROUP_CLOSE,
                    gc,
                ));
            }
            // rcov needs group_close and audit refuses it: each model's own
            // check says what to do there.
            let own = matches!(self.model, ModelKind::Rcov { .. } | ModelKind::Audit { .. });
            if self.group.is_none() && !own {
                no.refuse(format!(
                    "spec {name:?}: group_close needs a group column, but group is not set, so \
                     every row belongs to one group. Set group, or leave out group_close."
                ));
            }
            if self.embargo.is_some() {
                no.refuse(format!(
                    "spec {name:?}: group_close and embargo cannot both be set: a closed group \
                     cannot release the rows it still holds for the embargo. Leave out \
                     group_close, or embargo."
                ));
            }
            if gc == "session" {
                if self.session.is_none() {
                    no.refuse(format!(
                        "spec {name:?}: group_close=\"session\" closes each group at a session \
                         change, but session is not set. Set session to the column of sessions, \
                         or set group_close=\"monotone\"."
                    ));
                }
                if self.session_gap.is_some() {
                    no.refuse(format!(
                        "spec {name:?}: group_close=\"session\" and session_gap cannot both be \
                         set: both say what happens at a session change, and under \
                         group_close=\"session\" the group closes there. Leave out session_gap, \
                         or group_close."
                    ));
                }
            }
        }
        no.take(self.check_label_clock());
        if let Some(e) = self.embargo.as_ref().filter(|e| {
            let d = e.value();
            d.is_nan() || !d.is_finite() || d <= 0.0
        }) {
            no.refuse(format!(
                "spec {name:?}: embargo must be finite and above 0, but it is {e}. Set embargo \
                 to the delay before a row's target is known, in clock units, or leave it out \
                 for no delay."
            ));
        }
        // Holt has no features by construction, and neither has seqtest;
        // every other model needs at least one to regress on. Both
        // directions are errors: silently ignoring features passed to Holt
        // would look like they were used.
        if matches!(self.model, ModelKind::Holt { .. }) {
            if !self.features.is_empty() {
                no.refuse(format!(
                    "spec {name:?}: holt takes no features, but features lists {}: it \
                     extrapolates each target's own level and trend. Leave out features, or use \
                     a regression model such as ewridge to fit on them.",
                    columns_counted(self.features.len())
                ));
            }
        } else if let ModelKind::SeqTest {
            a,
            b,
            a_suffix,
            b_suffix,
        } = &self.model
        {
            if !self.features.is_empty() {
                no.refuse(if a.is_some() || b.is_some() {
                    format!(
                        "spec {name:?}: seqtest takes no features, but features lists {}: it \
                         compares the residuals of the specs a and b name. Leave out features.",
                        columns_counted(self.features.len())
                    )
                } else {
                    format!(
                        "spec {name:?}: seqtest takes no features, but features lists {}: it \
                         tests the sign of each target. Put the columns to test in targets, and \
                         leave out features.",
                        columns_counted(self.features.len())
                    )
                });
            }
            // The two are one switch: a comparison needs both sides.
            match (a, b) {
                (Some(_), None) | (None, Some(_)) => {
                    no.refuse(format!(
                        "spec {name:?}: seqtest needs a and b together, but only {} is set. Name \
                         both specs to compare them, or neither to test the sign of each target.",
                        if a.is_some() { "a" } else { "b" }
                    ));
                }
                // One spec against itself is a tie on every row -- unless
                // the suffixes pick two instances of its grid, which is a
                // comparison like any other.
                (Some(a), Some(b)) if a == b && a_suffix == b_suffix => {
                    no.refuse(format!(
                        "spec {name:?}: seqtest a and b both name {a:?}, with no suffix or the \
                         same one, so the spec would be compared with itself and tie on every row. Name two \
                         different specs, or give a_suffix and b_suffix to compare two of its \
                         slots."
                    ));
                }
                (Some(a), Some(b)) if a == &self.name || b == &self.name => {
                    no.refuse(format!(
                        "spec {name:?}: seqtest a and b must name other specs, but {} names this \
                         spec, which has no residuals of its own. Name two other specs in a and b.",
                        if a == &self.name { "a" } else { "b" }
                    ));
                }
                (None, None) if a_suffix.is_some() || b_suffix.is_some() => {
                    no.refuse(format!(
                        "spec {name:?}: a_suffix and b_suffix pick which slots of a and b to \
                         compare, but a and b are not set. Set a and b, or leave out the \
                         suffixes."
                    ));
                }
                _ => {}
            }
            // The stake is a function of the trial counts, so a trial is a
            // row, not a weight, and the wealth is a product that no decay
            // can apply to. Refused rather than ignored, as elsewhere.
            if self.weight.is_some() {
                no.refuse(format!(
                    "spec {name:?}: weight does not apply to seqtest, which counts every learned \
                     row as one trial, but weight is set. Leave out weight, and make a target \
                     null on a row to leave that row out of the test."
                ));
            }
            if self.half_life.is_some() || self.lam.is_some() {
                let key = if self.half_life.is_some() {
                    "half_life"
                } else {
                    "lam"
                };
                no.refuse(format!(
                    "spec {name:?}: {key} does not apply to seqtest, whose test does not forget, \
                     but it is set. Leave out {key}, and set session with session_gap=\"reset\", \
                     or restart_after_step_back, to reset the test."
                ));
            }
        } else if self.features.is_empty() && !(columns_missing && self.model.is_unsupervised()) {
            columns_missing = true;
            no.refuse(format!(
                "spec {name:?}: features must name at least one column, but it is empty. Name \
                 the columns to fit on in features."
            ));
        }
        if columns_missing {
            no.stop_here()?;
        }
        // A duplicated column silently splits its coefficient across identical
        // slots on an exactly singular system (the jitter fallback rescues the
        // solve, so nothing else complains), and a duplicated target collides
        // in the output struct. Both are always mistakes.
        for (label, cols) in [("features", &self.features), ("targets", &self.targets)] {
            let mut seen = std::collections::HashSet::new();
            let mut dups: Vec<&str> = Vec::new();
            for c in cols.iter() {
                if !seen.insert(c.as_str()) && !dups.contains(&c.as_str()) {
                    dups.push(c.as_str());
                }
            }
            if !dups.is_empty() {
                let listed = dups
                    .iter()
                    .map(|d| format!("{d:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                no.refuse(format!(
                    "spec {name:?}: {label} must name each column once, but it lists {listed} \
                     more than once. Remove the repeats from {label}."
                ));
            }
        }
        // A target used as a feature reads the *current row's* target to
        // predict that same row: perfect leakage, measured as corr(pred, y)
        // = 1.0. Hard rule 2 (out-of-sample by construction) protects the
        // target as target; this is the other door, and it must be locked --
        // it is exactly the accident a long column list invites, and the
        // resulting backtest looks wonderful right up until deployment.
        //
        // The unsupervised models are exempt *by design*, not oversight: they
        // predict no target. Their "targets" mirror their columns for
        // plumbing, and their outputs are read from the state BEFORE each
        // row, which is what makes an ew_cov statistic or a kmeans
        // assignment safe to use as a same-row feature (E1).
        let unsupervised = self.model.is_unsupervised();
        // The leak is a feature that is a target's *column* (a renamed
        // target's as much as a plain one's); a target merely named like a
        // feature reads another column (review 2026-09-26, D7: the names
        // were matched too, and refused that).
        // A formula target's columns may be features (docs/PLAN.md task 104,
        // reviewed): a window over `mid` looking ahead is not the row's
        // `mid`, and `closed` decides whether the row's own value is in it.
        let is_target = |f: &String| {
            self.targets
                .defs()
                .iter()
                .any(|t| t.value_column() == Some(f.as_str()))
        };
        if !unsupervised {
            // `ew_class` names its target with `label` (task 278, batch
            // 29); the glossary keeps the word for that name alone.
            let role = if matches!(self.model, ModelKind::EwClass { .. }) {
                "the column of classes"
            } else {
                "a target"
            };
            // Under a group, a shift crosses from one group's rows to the
            // next unless it is taken over the group.
            let shift = match &self.group {
                Some(g) => format!("shift(1).over({g:?})"),
                None => "shift(1)".to_string(),
            };
            for leak in self.features.iter().filter(|f| is_target(f)) {
                no.refuse(format!(
                    "spec {name:?}: {leak:?} is both {role} and a feature, so the model would \
                     predict it from itself on the same row. Remove {leak:?} from features, or \
                     make its previous value a column with Polars' {shift} in the query before \
                     the bank, and use that as the feature."
                ));
            }
        }
        // A formula target (docs/PLAN.md task 104) is a regression target.
        // Where the targets slot holds something else -- a column an
        // unsupervised model mirrors, a label, a sign, a 0/1 a probability is
        // fitted to -- it is refused by name rather than turned into a number
        // that means nothing.
        if self.targets.any_formula() {
            let why = match &self.model {
                m if m.is_unsupervised() => Some(("learns from no target", "")),
                ModelKind::EwClass { .. } => Some((
                    "classifies a column of classes",
                    ", and give that column in its place",
                )),
                // A comparison's own refusal below says it.
                m if m.compares().is_some() => None,
                ModelKind::SeqTest { .. } => Some((
                    "tests the signs of its targets",
                    ", and name the columns whose signs are tested in targets",
                )),
                ModelKind::Ftrl { loss, .. }
                    if loss.as_deref().unwrap_or("logistic") == "logistic" =>
                {
                    Some((
                        "fits a probability to a 0/1 target under loss=\"logistic\"",
                        ", and name a 0/1 column in targets, or set loss=\"squared\"",
                    ))
                }
                // `sgd` fits the same probability under the same loss, and a
                // log rate to a count under "poisson" (review 2026-09-26, D2).
                ModelKind::Sgd { loss, .. } if loss.as_deref() == Some("logistic") => Some((
                    "fits a probability to a 0/1 target under loss=\"logistic\"",
                    ", and name a 0/1 column in targets, or set loss=\"squared\"",
                )),
                ModelKind::Sgd { loss, .. } if loss.as_deref() == Some("poisson") => Some((
                    "fits a log rate to a count under loss=\"poisson\"",
                    ", and name a column of counts in targets, or set loss=\"squared\"",
                )),
                _ => None,
            };
            if let Some((why, also)) = why {
                no.refuse(format!(
                    "spec {name:?}: a formula target (a target holding a window operator with \
                     reverse=True) does not apply to {}, which {why}. Remove the formula \
                     target{also}.",
                    self.model.kind_name()
                ));
            }
        }
        // A formula target is resolved by the bank's window core, which
        // needs the stream's clock and runs the spec's clock policy
        // (docs/PLAN.md task 104). A closed group cannot release the rows
        // it holds, as under `embargo`.
        if self.targets.any_formula() {
            if self.group_close.is_some() {
                no.refuse(format!(
                    "spec {name:?}: group_close cannot be set beside a formula target: a closed \
                     group cannot release the rows whose windows are still open. Leave out \
                     group_close, or replace the formula target with a column."
                ));
            }
            if self.model.compares().is_some() {
                no.refuse(format!(
                    "spec {name:?}: a formula target does not apply to a spec that compares two \
                     other specs' residuals. Leave out the formula target."
                ));
            }
            // The bank adds the target beside the columns its formula reads,
            // so it cannot be one of them. The builder has said so since task
            // 159 (P1); a spec written as a dict or in TOML was refused only
            // at its first chunk (task 160, PB5).
            for t in self.targets.defs().iter().filter(|t| {
                t.formula
                    .as_ref()
                    .is_some_and(|tree| tree.columns().contains(&t.name))
            }) {
                no.refuse(format!(
                    "spec {name:?}: target {:?} is named after a column its formula reads, so \
                     the bank could not add it beside that column. Give the target a name of \
                     its own.",
                    t.name
                ));
            }
        }
        let decays = no.value(self.decays());
        no.take(self.clock_cfg().map(|_| ()));
        let mp = self.min_periods_per_target();
        if mp.len() != self.m() {
            no.refuse(format!(
                "spec {name:?}: min_weight must list one value per target ({}), but it lists {}. \
                 Give a list of {}, or a single number for every target.",
                self.m(),
                mp.len(),
                counted(self.m(), "value", "values"),
            ));
        }
        let bad: Vec<String> = mp
            .iter()
            .filter(|v| **v < 0.0 || v.is_nan())
            .map(|v| num_label(*v))
            .collect();
        if !bad.is_empty() {
            let got = match &self.min_weight {
                Some(FloatOrList::List(_)) => Got::Of(bad),
                _ => Got::One(bad.into_iter().next().unwrap_or_default()),
            };
            no.refuse(at_least_zero(
                &format!("spec {name:?}"),
                "min_weight",
                "",
                &got,
                Some("to predict from the first row"),
                None,
            ));
        }
        // The values that mean nothing (`limits.rs`).
        no.take(self.refuse_values_that_mean_nothing());
        if let Some(Num(f)) = self.min_settled_frac {
            if !(0.0..1.0).contains(&f) {
                no.refuse(format!(
                    "spec {name:?}: min_settled_frac must be at least 0 and below 1, but it is \
                     {f}. Set min_settled_frac to 0 to withhold nothing, or to a fraction such as \
                     0.5, which waits one half-life."
                ));
            } else if f > 0.0 && decays.is_some() && !self.has_decay() {
                // Where the decay itself is refused, its own refusal speaks.
                // Name only what the reader set, and offer decay only to a
                // model that takes it.
                // As `decays()` reads it: corrchange decays under scalar=True.
                let decays_by_kind = match &self.model {
                    ModelKind::SeqTest { .. }
                    | ModelKind::Rcov { .. }
                    | ModelKind::Bocpd { .. }
                    | ModelKind::Audit { .. } => false,
                    ModelKind::CorrChange { scalar, .. } => scalar.unwrap_or(false),
                    _ => true,
                };
                no.refuse(match (decays_by_kind, self.lam.is_some()) {
                    (true, true) => format!(
                        "spec {name:?}: min_settled_frac waits for the model to come toward its \
                         steady state under decay, but lam is 1, so it would withhold nothing. \
                         Set lam below 1, or leave out min_settled_frac."
                    ),
                    (true, false) => format!(
                        "spec {name:?}: min_settled_frac waits for the model to come toward its \
                         steady state under decay, but half_life is inf, so it would withhold \
                         nothing. Set half_life to a finite value, or leave out min_settled_frac."
                    ),
                    _ => format!(
                        "spec {name:?}: min_settled_frac waits for the model to come toward its \
                         steady state under decay, but {} does not decay, so it would withhold \
                         nothing. Leave out min_settled_frac.",
                        self.model.kind_name()
                    ),
                });
            }
        }
        if let Some(Num(r)) = self.max_error_inflation {
            if r.is_nan() || r <= 1.0 {
                no.refuse(format!(
                    "spec {name:?}: max_error_inflation must be above 1, but it is {r}: it bounds \
                     how far the estimation error may inflate a prediction's error over the \
                     noise. Set max_error_inflation above 1, such as 1.41, or to inf to withhold \
                     nothing."
                ));
            }
            // Range-checked and then dropped on every other model before
            // release 0.11.1 (docs/PLAN.md task 109): refused by name, as
            // `emit_error_inflation` is, with the reason (task 116).
            if !self.has_error_inflation() {
                no.refuse(format!(
                    "spec {name:?}: max_error_inflation needs a model that tracks how uncertain \
                     each prediction is, as ewridge, rls, kalman and lasso do, but {} does not: \
                     {}. Leave out max_error_inflation, and use min_weight to withhold the first \
                     predictions.",
                    self.model.kind_name(),
                    self.no_noise_statistic()
                ));
            }
        }
        if self.emit_se_coef && !self.has_se_coef() {
            let why = match self.model {
                ModelKind::Lasso { .. } => {
                    "it chooses its coefficients from the same rows, and a covariance that \
                     ignores the choice understates their error"
                }
                ModelKind::Huber { .. } | ModelKind::Quantile { .. } => {
                    "a robust fit's covariance needs a different formula from a least-squares \
                     fit's"
                }
                ModelKind::Sgd { .. } | ModelKind::Pa { .. } | ModelKind::Ftrl { .. } => {
                    "a gradient fit keeps no second moments of the features to read a covariance \
                     from"
                }
                _ => "it keeps no covariance of its coefficients",
            };
            no.refuse(format!(
                "spec {name:?}: emit_se_coef needs a model that keeps its coefficients' \
                 covariance, as ewridge, rls and kalman do, but {} does not: {why}. Leave out \
                 emit_se_coef, or use one of those models.",
                self.model.kind_name()
            ));
        }
        // A model that predicts no target is refused these below, once, with
        // every residual switch it was given.
        let predicts = !self.model.predicts_no_target();
        if self.emit_influence && predicts && !self.has_row_error_inflation() {
            no.refuse(format!(
                "spec {name:?}: emit_influence needs a model that reads one row's leverage, as \
                 ewridge, rls and kalman do, but {} does not: {}. Leave out emit_influence, or \
                 use one of those models.",
                self.model.kind_name(),
                self.no_leverage()
            ));
        }
        if self.emit_robust_se && predicts && !self.has_robust_se() {
            let why = match self.model {
                ModelKind::Kalman { .. } => {
                    "its coefficients' covariance is its own posterior, which emit_se_coef \
                     reports"
                }
                ModelKind::Lasso { .. } => {
                    "it chooses its coefficients from the same rows, and a covariance that \
                     ignores the choice understates their error"
                }
                ModelKind::Huber { .. } | ModelKind::Quantile { .. } => {
                    "a robust fit weighs each row by its loss, not as least squares does"
                }
                ModelKind::Sgd { .. } | ModelKind::Pa { .. } | ModelKind::Ftrl { .. } => {
                    "a gradient fit is not a weighted least-squares fit"
                }
                _ => "it has no least-squares coefficients",
            };
            no.refuse(format!(
                "spec {name:?}: emit_robust_se needs a least-squares fit, as ewridge and rls \
                 have, but {} has none: {why}. Leave out emit_robust_se, or use one of those \
                 models.",
                self.model.kind_name()
            ));
        }
        if self.ljung_box_lags.is_some() && !self.emit_specification {
            no.refuse(needs("ljung_box_lags", "emit_specification"));
        }
        if let Some(l) = self.ljung_box_lags {
            // The test reads its lags past the ones a horizon skips, and the
            // rows of them all are set aside before the first row: the
            // bound on the reader's own number is what is left of the
            // ceiling (the check `online_core::check_lag_ceiling` makes).
            let skip = self.ljung_box_skip();
            let most = online_core::MAX_LAG.saturating_sub(skip);
            if l == 0 {
                no.refuse(format!(
                    "spec {name:?}: ljung_box_lags must be at least 1, but it is 0. Set \
                     ljung_box_lags to 1 or more, such as the default 10, or leave it out."
                ));
            } else if l > most {
                no.refuse(if skip == 0 {
                    format!(
                        "spec {name:?}: ljung_box_lags must be at most {most}, but it is {l}: \
                         memory for every lag is set aside before the first row. Lower \
                         ljung_box_lags to {most} or below."
                    )
                } else {
                    format!(
                        "spec {name:?}: ljung_box_lags must be at most {most}, since \
                         horizon_rows={} skips the first {skip} lags and memory for every lag \
                         is set aside before the first row, but it is {l}. Lower \
                         ljung_box_lags to {most} or below, or lower horizon_rows.",
                        self.horizon()
                    )
                });
            }
        }
        if self.horizon_rows.is_some() && self.horizon_readers().is_empty() {
            no.refuse(format!(
                "spec {name:?}: horizon_rows applies only when emit_calibration, emit_breaks, \
                 emit_specification or emit_robust_se is on, but none is. Turn on the \
                 diagnostic that should read it, or leave out horizon_rows."
            ));
        }
        if let Some(h) = self.horizon_rows.filter(|h| *h > online_core::MAX_LAG / 2) {
            no.refuse(format!(
                "spec {name:?}: horizon_rows must be at most {}, but it is {h}: the diagnostics \
                 read twice as many lags, and memory for every lag is set aside before the \
                 first row. Lower horizon_rows to {} or below.",
                online_core::MAX_LAG / 2,
                online_core::MAX_LAG / 2
            ));
        }
        if self.robust_se_lags.is_some() && !self.emit_robust_se {
            no.refuse(needs("robust_se_lags", "emit_robust_se"));
        }
        if let Some(l) = self.robust_se_lags {
            no.take(
                online_core::check_lag_ceiling("robust_se_lags", l)
                    .map_err(|e| format!("spec {name:?}: {e}")),
            );
        }
        if self.emit_error_inflation && !self.has_row_error_inflation() {
            no.refuse(format!(
                "spec {name:?}: emit_error_inflation needs a model that reads one row's \
                 leverage, as ewridge, rls and kalman do, but {} does not: {}. Leave out \
                 emit_error_inflation, or use one of those models.",
                self.model.kind_name(),
                self.no_leverage()
            ));
        }
        const DRIFT_ACTION: [&str; 2] = ["flag", "reset"];
        if let Some(a) = &self.drift_action
            && !DRIFT_ACTION.contains(&a.as_str())
        {
            no.refuse(spec_words_refusal(
                &self.name,
                "drift_action",
                &DRIFT_ACTION,
                a,
            ));
        }
        // A tolerance or a threshold of `inf` is a detector that never
        // fires: no setting (review 2026-09-12, S27).
        if let Some(v) = self.drift_delta.filter(|v| *v < 0.0 || !v.is_finite()) {
            no.refuse(at_least_zero(
                &format!("spec {name:?}"),
                "drift_delta",
                "",
                &Got::One(num_label(v)),
                None,
                None,
            ));
        }
        if let Some(v) = self
            .drift_threshold
            .as_ref()
            .filter(|v| v.value() <= 0.0 || !v.value().is_finite())
        {
            no.refuse(format!(
                "spec {name:?}: drift_threshold must be finite and above 0, but it is {v}. Set \
                 drift_threshold to the summed excess that flags a drift, such as 20 without a \
                 clock, or a duration such as \"20m\" on a clock that holds times."
            ));
        }
        if let Some(qs) = &self.resid_quantiles {
            if qs.is_empty() {
                no.refuse(format!(
                    "spec {name:?}: resid_quantiles must list at least one level, but it is \
                     empty. List levels above 0 and below 1, such as [0.05, 0.95], or leave out \
                     resid_quantiles."
                ));
            }
            let bad: Vec<String> = qs
                .iter()
                .filter(|q| !(**q > 0.0 && **q < 1.0))
                .map(|q| num_label(*q))
                .collect();
            if !bad.is_empty() {
                no.refuse(format!(
                    "spec {name:?}: resid_quantiles must each be above 0 and below 1, but {} {}. \
                     Set each level above 0 and below 1.",
                    if bad.len() == 1 {
                        "one of them is"
                    } else {
                        "these are not:"
                    },
                    bad.join(", ")
                ));
            }
        }
        if let Some(c) = self.conformal.filter(|c| !(*c > 0.0 && *c < 1.0)) {
            no.refuse(format!(
                "spec {name:?}: conformal must be above 0 and below 1, but it is {c}. Set \
                 conformal to the share of rows whose target the interval should cover, such \
                 as 0.9."
            ));
        }
        if let Some(ConformalRate::Fixed(r)) = self
            .conformal_rate
            .filter(|r| !(r.base() > 0.0 && r.base().is_finite()))
        {
            no.refuse(format!(
                "spec {name:?}: conformal_rate must be finite and above 0, but it is {r}. Set \
                 conformal_rate to how fast the interval's width adapts, such as the default \
                 0.05."
            ));
        }
        if self.conformal_rate.is_some() && self.conformal.is_none() {
            no.refuse(format!(
                "spec {name:?}: conformal_rate sets how fast the interval adapts, but conformal, \
                 its coverage, is not set. Set conformal, such as 0.9, or leave out \
                 conformal_rate."
            ));
        }
        if self.resid_autocorr_lag.is_some_and(|l| l == 0) {
            no.refuse(format!(
                "spec {name:?}: resid_autocorr_lag must be at least 1, but it is 0. Set \
                 resid_autocorr_lag to the lag of the autocorrelation to report, such as 1."
            ));
        }
        // The tracker's buffer is sized from the lag before the first row,
        // as the models' lag rings are (review 2026-10-06, CD10).
        if let Some(l) = self.resid_autocorr_lag.filter(|l| *l > 0) {
            no.take(
                online_core::check_lag_ceiling("resid_autocorr_lag", l)
                    .map_err(|e| format!("spec {name:?}: {e}")),
            );
        }
        if let Some(v) = self.average_eta.filter(|v| v.0 <= 0.0 || v.0.is_nan()) {
            no.refuse(format!(
                "spec {name:?}: average_eta must be above 0, but it is {}. Set average_eta to a \
                 positive value, such as the default 1, or to inf to take the best slot alone, \
                 as emit_selected does.",
                v.0
            ));
        }
        // A knob whose switch is off does nothing: refused rather than
        // ignored, as elsewhere here (review 2026-09-12, S22). `drift_action
        // = "flag"` is what `fill_defaults` writes on every spec, so only
        // `"reset"` asks for anything.
        if !self.emit_drift {
            if self.drift_action.as_deref() == Some("reset") {
                no.refuse(format!(
                    "spec {name:?}: drift_action=\"reset\" applies only when emit_drift is on, \
                     but emit_drift is off. Turn on emit_drift, or leave out drift_action."
                ));
            }
            for (knob, set) in [
                ("drift_delta", self.drift_delta.is_some()),
                ("drift_threshold", self.drift_threshold.is_some()),
            ] {
                if set {
                    no.refuse(needs(knob, "emit_drift"));
                }
            }
        } else if self.clock.is_some() && self.drift_threshold.is_none() {
            no.refuse(format!(
                "spec {name:?}: drift_threshold must be set when the spec has a clock, but it is \
                 not: the detector sums each row's excess times its clock step, so no default \
                 means the same on every clock. Set drift_threshold in clock units, or as a \
                 duration such as \"20m\" on a clock that holds times."
            ));
        }
        if self.average_eta.is_some() && !self.emit_averaged {
            no.refuse(needs("average_eta", "emit_averaged"));
        }
        if self.resid_autocorr_lag.is_some() && !self.emit_autocorr {
            no.refuse(needs("resid_autocorr_lag", "emit_autocorr"));
        }
        // A diagnostic's own memory (docs/PLAN.md task 221): a half-life
        // above 0, `inf` the run-once form that forgets nothing; given with
        // its switch off it does nothing, and is refused as any such knob.
        for (key, switch, on, memory) in self.diagnostic_memories() {
            let Some(h) = memory else { continue };
            if !on {
                no.refuse(needs(key, switch));
            }
            if h.value().is_nan() || h.value() <= 0.0 {
                // Feature health compares two memories, which an infinite
                // one makes the same (refused below).
                no.refuse(if key == "feature_health_half_life" {
                    format!(
                        "spec {name:?}: {key} must be above 0, but it is {h}. Set {key} to a \
                         finite positive number of clock units."
                    )
                } else {
                    format!(
                        "spec {name:?}: {key} must be above 0, but it is {h}. Set {key} to a \
                         positive number of clock units, or to inf for a diagnostic that \
                         forgets nothing."
                    )
                });
            }
        }
        if self.session_gap.is_some() && self.session.is_none() {
            no.refuse(format!(
                "spec {name:?}: session_gap applies only with a session column, but session is \
                 not set. Set session, or leave out session_gap."
            ));
        }
        // A step back is read off the clock; given without one it would be
        // silently ignored, and a key that does nothing is refused here.
        if self.restart_after_step_back.is_some() && self.clock.is_none() {
            no.refuse(format!(
                "spec {name:?}: restart_after_step_back applies only with a clock, but clock is \
                 not set. Set clock, or leave out restart_after_step_back."
            ));
        }
        // The snapshot cadence spaces a window's snapshots, so each part of
        // it needs a window; the clock spacing is clock units, finite and
        // `>= 0` as `solve_every` is, `0` for every row (docs/PLAN.md task
        // 162). A negative value would have meant every row unsaid, NaN never.
        if let (Some((window, _)), Some((every, rows))) =
            (self.model.window_parts(), self.model.window_cadence())
        {
            for (key, given) in [
                ("snapshot_every", every.is_some()),
                ("max_rows_between_snapshots", rows.is_some()),
            ] {
                if given && window.is_none() {
                    no.refuse(format!(
                        "spec {name:?}: {key} applies only under a hard window, but window_size \
                         is not set. Set window_size, or leave out {key}."
                    ));
                }
            }
            if let Some(e) = every.filter(|e| !non_negative(e.value()) || !e.value().is_finite()) {
                no.refuse(at_least_zero(
                    &format!("spec {name:?}"),
                    "snapshot_every",
                    " clock units",
                    &Got::One(e.to_string()),
                    Some("to take a snapshot on every row"),
                    None,
                ));
            }
        }
        // The window's edge, Polars' `closed` (docs/PLAN.md task 196, N17):
        // of its four values the two that leave out the window's last row
        // cannot apply, and are refused as the spec is read
        // ([`ModelClosed`]); `"both"` with no window is a key that does
        // nothing (review S10's rule).
        if let (Some((window, _)), Some(closed)) =
            (self.model.window_parts(), self.model.window_closed())
            && closed == ModelClosed::Both
            && window.is_none()
        {
            no.refuse(format!(
                "spec {name:?}: closed applies only under a hard window, where it says whether a \
                 row exactly window_size old keeps its weight, but window_size is not set. Set \
                 window_size, or leave out closed."
            ));
        }
        // A row cap of no rows is no schedule, in every `max_rows_between_*`,
        // as `max_rows_between_coefs`'s is since task 178; the clock form's
        // `0` is every row. `0` was every row for four of them and refused
        // for the fifth (review 2026-10-06, PC3 CB4 YA2; docs/PLAN.md task
        // 196, U7).
        for (key, cap, clock) in self.model.row_caps() {
            if cap == Some(0) {
                no.refuse(format!(
                    "spec {name:?}: {key} must be at least 1, but it is 0. Set {key} to 1 or \
                     more, or leave it out and set {clock} to 0 to act on every row."
                ));
            }
        }
        // A count of no threads (docs/PLAN.md task 225): the reader refuses
        // it, and a spec built in Rust is held to the same.
        if self.model.gram_threads_given() == Some(CountOrAuto::Count(0)) {
            no.refuse(format!(
                "spec {name:?}: gram_threads must be at least 1 or \"auto\", but it is 0. Set \
                 gram_threads to the number of the bank's threads to update the Gram on, or to \
                 \"auto\" for all of them."
            ));
        }
        // A budget bounds a window's snapshots, so it needs a window, and a
        // budget of no bytes bounds nothing (review 2026-09-12, P4).
        if let Some((window, Some(budget))) = self.model.window_parts() {
            if window.is_none() {
                no.refuse(format!(
                    "spec {name:?}: window_budget applies only under a hard window, but \
                     window_size is not set. Set window_size, or leave out window_budget."
                ));
            }
            let mib = budget.mib();
            if mib <= 0.0 || mib.is_nan() {
                // The mode the reader gave, as they wrote it.
                let mode = match budget {
                    WindowBudgetSpec::Thin(_) => "thin",
                    WindowBudgetSpec::Refuse(_) => "refuse",
                };
                no.refuse(format!(
                    "spec {name:?}: window_budget's \"{mode}\" must be above 0 MiB, but it is \
                     {}. Set it to a positive number of MiB, such as 256, or to inf for no \
                     bound.",
                    num_label(mib)
                ));
            }
        }
        // `coef_every` and `max_rows_between_coefs` schedule the `coef`
        // field, which a model with no coefficients does not have (review
        // 2026-09-12, S22). `0` was the default and passed there; since task
        // 178 it is every row, and refused there as any other value is.
        for (key, given) in [
            ("coef_every", self.coef_every.is_some()),
            (
                "max_rows_between_coefs",
                self.max_rows_between_coefs.is_some(),
            ),
        ] {
            if given && !self.model.has_coef() {
                no.refuse(format!(
                    "spec {name:?}: {key} applies only to a model that reports coefficients, but \
                     {} reports none. Leave out {key}.",
                    self.model.kind_name()
                ));
            }
        }
        if let Some(e) = self
            .coef_every
            .as_ref()
            .filter(|e| !(e.value().is_finite() && non_negative(e.value())))
        {
            no.refuse(at_least_zero(
                &format!("spec {name:?}"),
                "coef_every",
                " clock units",
                &Got::One(e.to_string()),
                Some("to fill the coef field on every row"),
                None,
            ));
        }
        // A cap of no rows is no schedule: every row is `coef_every = 0`.
        if self.max_rows_between_coefs == Some(0) {
            no.refuse(format!(
                "spec {name:?}: max_rows_between_coefs must be at least 1, but it is 0. Set \
                 max_rows_between_coefs to 1 or more, or leave it out and set coef_every to 0 \
                 to fill the coef field on every row."
            ));
        }
        // Nothing residual-based applies to a model that predicts no target.
        // Refused rather than ignored: a flag that silently emits nothing
        // looks like a bug in the output, not in the spec.
        let mut health_refused = false;
        if self.model.predicts_no_target() {
            let asked = [
                ("emit_sigma", self.emit_sigma),
                ("emit_zscore", self.emit_zscore),
                ("emit_metrics", self.emit_metrics),
                ("resid_quantiles", self.resid_quantiles.is_some()),
                ("conformal", self.conformal.is_some()),
                ("emit_autocorr", self.emit_autocorr),
                ("emit_drift", self.emit_drift),
                ("emit_calibration", self.emit_calibration),
                ("emit_breaks", self.emit_breaks),
                ("emit_robust_se", self.emit_robust_se),
                ("emit_specification", self.emit_specification),
                ("emit_tails", self.emit_tails),
                ("emit_influence", self.emit_influence),
                ("emit_averaged", self.emit_averaged),
                ("emit_selected", self.emit_selected),
            ];
            let on: Vec<&str> = asked
                .iter()
                .filter(|(_, on)| *on)
                .map(|(f, _)| *f)
                .collect();
            if let Some((last, rest)) = on.split_last() {
                let listed = if rest.is_empty() {
                    (*last).to_string()
                } else {
                    format!("{} and {last}", rest.join(", "))
                };
                no.refuse(format!(
                    "spec {name:?}: {listed} {} not apply to {}, which predicts no number for a \
                     target, so has no residuals. Leave out {listed}.",
                    if rest.is_empty() { "does" } else { "do" },
                    self.model.kind_name()
                ));
            }
            // Feature health reads no residual: it watches the features
            // a fit's coefficients lean on (task 232 (14); review round
            // 6, F-8).
            if self.emit_feature_health {
                health_refused = true;
                no.refuse(format!(
                    "spec {name:?}: emit_feature_health does not apply to {}, which fits no \
                     coefficients of its features: it watches the features a fit's coefficients \
                     lean on. Leave out emit_feature_health, and use ModelBank.check() or \
                     po.spec.audit to watch the features themselves.",
                    self.model.kind_name()
                ));
            }
        }
        // Feature health reads the features alone: a model with none would
        // write no field, so the switch is refused rather than ignored
        // (review round 6, F6).
        if self.emit_feature_health && !health_refused && self.k() == 0 {
            health_refused = true;
            no.refuse(format!(
                "spec {name:?}: emit_feature_health watches each feature's spread and mean, but \
                 {} has no features. Leave out emit_feature_health.",
                self.model.kind_name()
            ));
        }
        // Run once the fast and the slow memory are both the whole stream,
        // and every field would be null: refused rather than written null
        // (task 232 (2); review round 6, B-4).
        if self.emit_feature_health
            && !health_refused
            && let Some(decays) = &decays
        {
            let forgets = |d: Decay| match d {
                Decay::Halflife(h) => h.is_finite(),
                Decay::Lam(l) => l < 1.0,
            };
            let memory = self.feature_health_half_life.as_ref();
            if decays
                .iter()
                .any(|(_, d)| !forgets(self.diagnostic_decay_of(memory, *d, 1)))
            {
                let why = if memory.is_some() {
                    "feature_health_half_life is inf"
                } else if self.lam.is_some() {
                    "its memory is the fit's, which forgets nothing (lam is 1, with no \
                     window_size)"
                } else {
                    "its memory is the fit's, which forgets nothing (half_life is inf, with no \
                     window_size)"
                };
                no.refuse(format!(
                    "spec {name:?}: emit_feature_health compares each feature at its memory with \
                     four times that memory, but {why}, so both read the whole stream and every \
                     field would be null. Set feature_health_half_life to a finite value."
                ));
            }
        }
        // Both choose across a target's slots, and one slot leaves nothing to
        // choose: `emit_averaged` there was the slot's own prediction under
        // another name, null on the first scored row (task 278).
        // A model that predicts no target has no slots to choose among: its
        // own refusal of the flag speaks.
        if let Some(decays) = decays.as_ref().filter(|_| !self.model.predicts_no_target()) {
            for (flag, on) in [
                ("emit_selected", self.emit_selected),
                ("emit_averaged", self.emit_averaged),
            ] {
                if !on {
                    continue;
                }
                let n_slots = decays.len() * crate::combo_labels(self).len();
                if n_slots < 2 {
                    // Only the grids this model takes: every model takes a
                    // half_life grid (`combos`, stream.rs).
                    let grids = match self.model {
                        ModelKind::EwRidge { .. } => "half_life, ridge or feature_sets",
                        ModelKind::Huber { .. } => "half_life or feature_sets",
                        ModelKind::Lasso { .. } => "half_life or penalties",
                        _ => "half_life",
                    };
                    no.refuse(format!(
                        "spec {name:?}: {flag} chooses among each target's slots, but this spec \
                         has one slot per target. Give {grids} several values, or leave out \
                         {flag}."
                    ));
                }
            }
        }
        self.validate_model(no);
        Ok(())
    }
}

mod models;

/// `1 column`, `3 columns`: a count with its noun, in agreement.
fn counted(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

fn columns_counted(n: usize) -> String {
    counted(n, "column", "columns")
}

#[cfg(test)]
mod clock_tests;

#[cfg(test)]
mod sgd_tests;

#[cfg(test)]
mod huber_scale_tests;

#[cfg(test)]
mod pa_tests;
