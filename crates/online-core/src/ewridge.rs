//! EW-ridge on sufficient statistics (docs/PLAN.md §4.1) — the workhorse.
//!
//! Math (per row, decay factor `lam = decay.factor(d_clock)`, weight `w`,
//! `z = [1, x]` when the intercept is configured):
//!
//! ```text
//! W'   = lam W + w                          (n_eff, over every row)
//! S'   = (lam W_S S + w z z^T) / W_S'       (EW mean of z z^T over its Gram's rows)
//! W_j' = lam W_j + w                        (only when y_j present)
//! r_j' = (lam W_j r_j + w z y_j) / W_j'     (per target: EW mean of z y_j)
//! ```
//!
//! Which rows a Gram `S` learns is `target_gaps` (docs/PLAN.md task 81,
//! [`TargetGaps`]). Under `own_rows`, the default, a target's Gram learns
//! exactly the rows the target is present on and ages over the rest, so its
//! fit is the fit of its rows; targets present on the same rows share one.
//! Under `pairwise` one Gram learns every row. A target present on every row
//! reads the same Gram either way (`crate::gaps` has the bookkeeping).
//!
//! `S` is kept centred, as [`EwCov`] keeps it, and so is each `r_j` (review
//! 2026-09-12, N1): over the rows target `j` was present on, the EW mean
//! `ȳ_j`, the mean `m_j` of `z` there, and `c_j = E[(z − m_j)(y_j − ȳ_j)]`,
//! each by the weighted Welford step with `a = lam W_j / W_j'` and
//! `b = w / W_j'`, so that `r_j = c_j + m_j ȳ_j` (`crate::gaps::Cross`).
//!
//! Solve (per grid combo = feature set x ridge value), scheduled by
//! `solve_every` clock units / `max_rows_between_solves`, one system per
//! Gram for the targets that read it:
//! - plain:        `(S + ridge D) beta = r_j`, D = I minus the intercept slot;
//! - standardized: the same system scaled to correlation form, solved,
//!   unscaled; ~zero-variance features dropped;
//! - ridge_scale:  `(W_S S + prior_scale * ridge I) beta = W_S r_j` — a
//!   decaying prior on the sum scale, penalizing the intercept: exactly
//!   classic RLS regularization (used by the RLS agreement test, task 9).
//!
//! With an intercept, and without `ridge_scale`, the unpenalized intercept is
//! eliminated and the slopes are solved on the centred system
//!
//! ```text
//! (C + ridge I) beta = c_j        beta_0 = ȳ_j − m_j · beta
//! ```
//!
//! `C` the centred co-moments of the target's Gram. Under `own_rows` that is
//! the plain system exactly, with nothing in it the size of `level²`: the raw
//! normal equations, or a right-hand side formed as `E[z y] − m ȳ`, lose
//! `level²·ε`, which at `1e8` is the whole fit. Under `pairwise` it is the
//! same formula on the Gram of every row. Through the origin, and under
//! `ridge_scale`, there is no intercept to eliminate and the raw system is
//! solved: the Gram's `E[z zᵀ]` against the target's `E[z·y_j]`.
//!
//! Predictions use the last solved coefficients (out-of-sample by construction:
//! the solve happens after the row's update, the pred before it).

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use crate::gaps::{Acc, AccSnap, AccView, Cross, gram_parts};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::{SpdFactor, dot_aug};
use crate::{Decay, EwCov, GramPart, TargetGaps, TargetMoments};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwRidgeCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    pub decay: Decay,
    /// Ridge grid, expanded at solve time; length >= 1.
    pub ridge: Vec<f64>,
    /// Named subsets of feature indices (0-based, excluding the intercept).
    /// Empty means one full set named "all".
    pub feature_sets: Vec<(String, Vec<usize>)>,
    pub standardize: bool,
    /// Decaying sum-scale prior (classic RLS regularization). Incompatible with
    /// `standardize` and with grids; the intercept is penalized.
    pub ridge_scale: bool,
    /// Shrink toward these coefficients instead of toward zero
    /// (ENHANCEMENTS E15): the solve becomes `(S + ridge·D)β = r + ridge·D·β₀`.
    /// One vector per target, each `k_total` long, in the features' original
    /// units. The intercept slot is read only under `ridge_scale`, the one
    /// solve that penalizes the intercept; elsewhere it is unpenalized and
    /// the slot is ignored.
    ///
    /// **Whether the prior fades depends on `ridge_scale`, and the difference
    /// matters.** `S` here is a weighted *mean*, not a sum, so it does not grow
    /// with the sample: a plain `ridge` is a fixed per-observation penalty and
    /// its pull toward `coef_prior` is **permanent** — "always stay near this
    /// belief". With `ridge_scale` the prior sits on the sum scale and its
    /// weight decays with the data, which is the usual warm start: "begin at
    /// yesterday's fit and let evidence take over".
    #[serde(default)]
    pub coef_prior: Option<Vec<Vec<f64>>>,
    /// Blend toward a slow-moving twin on a session change, instead of the
    /// all-or-nothing choice between `session_gap` and a full reset
    /// (ENHANCEMENTS E6, PLAN §12 open question 1).
    ///
    /// A second accumulator runs alongside the main one with `long_half_life`,
    /// representing the long-run relationship. On a session boundary the main
    /// accumulators' moments become a mixture of the two data sets, `1 − f` of
    /// today's and `f` of the long run's, with `f = session_shrink`:
    ///
    /// ```text
    /// m' = (1−f)·m_fast + f·m_slow
    /// C' = (1−f)·C_fast + f·C_slow + f·(1−f)·(m_fast − m_slow)(m_fast − m_slow)ᵀ
    /// ```
    ///
    /// for the means and the centred moments, the cross-moments with the
    /// targets alike. The weight, the Kish sums and the `ridge_scale` prior
    /// scale stay today's, so `n_eff`, the warm-up gates and the solve
    /// schedule do not move: `0` keeps today's fit, `1` takes the long run's
    /// moments at today's weight, and `f` between fits on that share of the
    /// long run (docs/PLAN.md task 145). Unlike `session_gap` this changes
    /// *what the model believes*, not how confident it is.
    #[serde(default)]
    pub session_shrink: Option<f64>,
    /// Half-life of the slow twin. Required by `session_shrink`.
    #[serde(default)]
    pub long_half_life: Option<f64>,
    /// Outputs are null until `n_eff` (before the row's update) reaches this.
    pub min_weight: f64,
    /// Solve cadence in clock units; <= 0 solves every row.
    pub solve_every: f64,
    /// Row cap between solves; 1 solves every row.
    pub max_rows_between_solves: u32,
    /// The default cadence (docs/PLAN.md task 115 (b)): solve once the weight
    /// learned since the last solve reaches this share of the weight the fit
    /// holds, in place of `solve_every`'s clock. In steady state that is the
    /// clock's own `half_life / 50` at a share of `ln 2 / 50`; where they part
    /// -- warm-up, after a gap, a half-life far longer than the stream -- it
    /// keeps the fit that close to its data, where the clock solved once and
    /// never again. `None` keeps the clock.
    #[serde(default)]
    pub solve_share: Option<f64>,
    /// Rows of the Gram update held back and merged as one block
    /// (docs/ENHANCEMENTS.md E51, docs/PLAN.md task 71). `0`, the default,
    /// updates the `k×k` matrix on every row. With `B` here, a row's `z` is
    /// buffered and the matrix is brought up to date once per `B` rows by a
    /// `k×B` times `B×k` product, `6.6×` faster than the rank-one updates at
    /// `k = 1,000` with `B = 256`; the four scalars still run per row, so
    /// `n_eff` and `min_weight` are unchanged to the bit.
    ///
    /// The matrix is also brought up to date before every solve, so the
    /// effective block is `min(B, rows between solves)` and the option pays
    /// only with a solve cadence: it is refused with `solve_every <= 0` or
    /// `max_rows_between_solves <= 1`, and with `window`, which reads the
    /// matrix on every row. A row a Gram skips under `target_gaps =
    /// "own_rows"` is held as a zero-weight row rather than flushing the
    /// block. Memory: `B × k_total` floats per Gram -- one, or under
    /// `own_rows` one per pattern of missing targets -- twice with
    /// `session_shrink`'s twin, held in the state file mid-block, and refused
    /// over 256 MiB a Gram. The block's product is a floating-point sum in a
    /// different order from the per-row one, so a blocked fit is not
    /// bit-identical to an unblocked one and its last bits may differ across
    /// CPUs; it is identical whichever way the stream is chunked.
    #[serde(default)]
    pub gram_block_rows: usize,
    /// Which rows a target's Gram is taken over where the target is null on
    /// some (docs/PLAN.md task 81; [`TargetGaps`]): its own, the default, or
    /// every row.
    #[serde(default)]
    pub target_gaps: TargetGaps,
    /// Clock units of history the fit is computed from, with a *hard* cutoff:
    /// a row older than this contributes nothing to the Gram, where the
    /// exponential weight alone would leave `0.5^(age/half_life)` of it
    /// (docs/PLAN.md §13). Inside the window the weights are still
    /// exponential. A half-life grid is one model instance per entry, so each
    /// carries its own ring.
    ///
    /// **Last, with `window_every` and `max_rows_between_snapshots`, and
    /// they must stay last.** The compact msgpack encoding writes a struct as
    /// an *array*, so a `skip_serializing_if` field anywhere but the end
    /// shifts every field after it when absent: only the row cap skips, and
    /// `window` and `window_every` are written as nil.
    #[serde(default)]
    pub window: Option<f64>,
    /// Clock units between the snapshots the window is computed from, as
    /// `solve_every` is between solves (docs/PLAN.md task 162): `0` is every
    /// row. With `max_rows_between_snapshots` too, whichever comes first;
    /// with neither, every row, the tightest boundary. A coarser cadence
    /// divides the memory and only ever shortens the effective window, to
    /// no less than `window - window_every`. The clock is the one the model
    /// is stepped on, so a gap capped at `gap_cap` counts as the cap.
    #[serde(default)]
    pub window_every: Option<f64>,
    /// At most this many rows between the window's snapshots, as
    /// `max_rows_between_solves` is between solves, counted on every row the
    /// model is stepped with, rows of weight zero included; `0` or `1` is
    /// every row (docs/PLAN.md task 162).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rows_between_snapshots: Option<usize>,
}

/// The most a held Gram block may take before the model refuses to build,
/// on the pattern of `marginal`'s bins: `k` can be 10,000 here, and silently
/// allocating gigabytes is a worse outcome than an error that names the
/// number.
const GRAM_BLOCK_BUDGET: usize = 256 << 20;

impl EwRidgeCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    /// (feature-set index, ridge index) pairs in output order.
    fn combos(&self) -> Vec<(usize, usize)> {
        let nf = self.feature_sets.len().max(1);
        let nr = self.ridge.len();
        (0..nf).flat_map(|f| (0..nr).map(move |r| (f, r))).collect()
    }

    pub fn n_combos(&self) -> usize {
        self.feature_sets.len().max(1) * self.ridge.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        self.decay.check().map_err(|e| format!("ewridge: {e}"))?;
        if self
            .solve_share
            .is_some_and(|f| !(f.is_finite() && f > 0.0))
        {
            return Err("solve_share must be finite and > 0".into());
        }
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        if self.ridge.is_empty() {
            return Err("ridge grid must have at least one value".into());
        }
        if let Some(w) = self.window {
            if !w.is_finite() || w <= 0.0 {
                return Err(format!(
                    "ewridge: window_size must be finite and > 0 (got {w}); it is clock units of \
                     history to keep"
                ));
            }
            if self.ridge_scale {
                return Err(
                    "ewridge: window_size and ridge_scale do not combine; the decaying prior's scale \
                     is the product of every decay factor the stream has applied, which a \
                     window truncates the data of but not the prior"
                        .into(),
                );
            }
            if self.session_shrink.is_some() {
                return Err(
                    "ewridge: window_size and session_shrink do not combine; the slow twin is a \
                     second accumulator under a longer half_life, and truncating one and not \
                     the other would blend two different histories"
                        .into(),
                );
            }
        }
        crate::window::check_cadence(
            "ewridge",
            self.window,
            self.window_every,
            self.max_rows_between_snapshots,
        )?;
        if self.gram_block_rows > 0 {
            if self.window.is_some() {
                return Err(
                    "ewridge: gram_block_rows and window do not combine; the window snapshots \
                     the Gram on every row, so there would be nothing to hold back"
                        .into(),
                );
            }
            if self.solve_every <= 0.0 || self.max_rows_between_solves <= 1 {
                return Err(format!(
                    "ewridge: gram_block_rows needs a solve cadence; the Gram is brought up to \
                     date before every solve, and with solve_every = {} and \
                     max_rows_between_solves = {} that is every row, so a block would never \
                     hold more than one. Set solve_every > 0 (the default is by weight under a \
                     finite half_life, and 0 for `lam` and an infinite half_life) and \
                     max_rows_between_solves > 1",
                    self.solve_every, self.max_rows_between_solves
                ));
            }
            // Held rows: `B × k_total` per accumulator, the slow twin included.
            let twins = 1 + usize::from(self.session_shrink.is_some());
            let cells = self
                .gram_block_rows
                .saturating_mul(self.k_total())
                .saturating_mul(twins);
            let bytes = cells.saturating_mul(std::mem::size_of::<f64>());
            if bytes > GRAM_BLOCK_BUDGET {
                return Err(format!(
                    "ewridge: gram_block_rows = {} would hold {:.1} GiB of rows ({} × {} \
                     features{}), over the {} MiB budget; reduce it, or narrow the features",
                    self.gram_block_rows,
                    bytes as f64 / (1u64 << 30) as f64,
                    self.gram_block_rows,
                    self.k_total(),
                    if twins == 2 {
                        ", twice for the slow twin"
                    } else {
                        ""
                    },
                    GRAM_BLOCK_BUDGET >> 20,
                ));
            }
        }
        if self.ridge_scale && (self.standardize || self.n_combos() > 1) {
            return Err("ridge_scale is incompatible with standardize and grids".into());
        }
        match (self.session_shrink, self.long_half_life) {
            (Some(f), _) if !(0.0..=1.0).contains(&f) => {
                return Err("session_shrink must be in [0, 1]".into());
            }
            (Some(_), None) => {
                return Err("session_shrink needs long_half_life (the slow twin's decay)".into());
            }
            (None, Some(_)) => {
                return Err("long_half_life has no effect without session_shrink".into());
            }
            (Some(_), Some(h)) if h <= 0.0 || h.is_nan() => {
                return Err("long_half_life must be > 0".into());
            }
            _ => {}
        }
        if let Some(c) = &self.coef_prior {
            if c.len() != self.n_targets || c.iter().any(|v| v.len() != self.k_total()) {
                return Err(format!(
                    "coef_prior must be {} vector{} of length {}",
                    self.n_targets,
                    if self.n_targets == 1 { "" } else { "s" },
                    self.k_total()
                ));
            }
            if c.iter().flatten().any(|v| !v.is_finite()) {
                return Err("coef_prior values must be finite".into());
            }
        }
        for (name, idx) in &self.feature_sets {
            // Empty is its own message: an empty set has no index to be out
            // of range (review 2026-09-12, S7).
            if idx.is_empty() {
                return Err(format!("feature set {name:?} is empty"));
            }
            if idx.iter().any(|&i| i >= self.n_features) {
                return Err(format!("feature set {name:?} has out-of-range indices"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwRidge {
    cfg: EwRidgeCfg,
    /// The Grams, and per target its weight, centred cross-moments and
    /// moments (see `crate::gaps::Acc`).
    acc: Acc,
    /// Per-target EW residual variance and its weight sum.
    wsig: Vec<f64>,
    sig2: Vec<f64>,
    /// Last solved coefficients per output slot (target-major, then combo),
    /// each of length `k_total` (zeros outside a combo's feature set).
    beta: Option<Vec<Vec<f64>>>,
    /// Slow-moving twin for `session_shrink`: the same accumulators under
    /// `long_half_life`, representing the long-run relationship.
    #[serde(default)]
    slow: Option<Box<Acc>>,
    clock_since_solve: f64,
    rows_since_solve: u32,
    /// Weight learned since the last solve, for `solve_share`.
    #[serde(default)]
    weight_since_solve: f64,
    pub solve_failures: u64,
    /// What the last solve left for the readiness statistics
    /// (docs/WARMUP-AND-CONVERGENCE.md §2): the effective degrees of freedom
    /// and each coefficient's data share per slot, and -- under
    /// `keep_factor` -- the systems the per-row leverage is read against.
    /// A solve's shares wait until something reads them, and a save writes
    /// them as a read would (`Pending`, `serialize_settled`).
    #[serde(default, serialize_with = "serialize_settled")]
    ready: Readiness,
    /// Keep each solve's system so the per-row leverage can be answered
    /// ([`EwRidge::set_keep_factor`]). Configuration that travels with the
    /// state, since the systems it keeps do.
    #[serde(default)]
    keep_factor: bool,
    /// The hard-cutoff window, when the spec asks for one (docs/PLAN.md §13).
    /// Absent otherwise, so an ordinary fit writes no ring.
    ///
    /// **Last, and it must stay last.** The compact msgpack encoding writes a
    /// struct as an *array*, so a `skip_serializing_if` field anywhere but the
    /// end shifts every field after it when it is absent, and the state loads
    /// as garbage -- "invalid type: boolean, expected f64" is what that looks
    /// like. `state_roundtrip_continues_identically` catches it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    win: Option<Windowed>,
    // scratch buffers (serialized for simplicity; tiny)
    #[serde(skip)]
    zbuf: Vec<f64>,
    /// The factors of `ready.systems`, rebuilt from them on load.
    #[serde(skip)]
    factors: Factors,
}

/// What a window needs beyond the accumulators: where the clock has got to,
/// and the snapshots to subtract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Windowed {
    clock: f64,
    snaps: crate::Snapshots<RidgeMoments>,
}

/// Every accumulator a fit is read from, as it stood before a row and decayed
/// to that row's clock. The Grams and the per-target cross-moments are what
/// the solve reads; the residual variance is what `sigma` and `zscore` read,
/// and truncating one without the other would report a windowed fit beside an
/// unwindowed spread.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RidgeMoments {
    acc: AccSnap,
    wsig: Vec<f64>,
    sig2: Vec<f64>,
}

impl crate::Footprint for RidgeMoments {
    fn footprint(&self) -> usize {
        crate::Footprint::footprint(&self.acc)
            + crate::window::floats(&self.wsig)
            + crate::window::floats(&self.sig2)
    }
}

/// The accumulators a windowed fit reads, with everything older than the
/// window subtracted off.
struct RidgeView {
    /// A target with no row left in the window has weight 0 here, and then
    /// reports nothing (review 2026-09-12, C2).
    acc: AccView,
    sig2: Vec<f64>,
}

impl EwRidge {
    pub fn new(cfg: EwRidgeCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k_total = cfg.k_total();
        let m = cfg.n_targets;
        // Blocked or not, the twin runs the same accumulators as the main
        // ones: its update is the same cost, and its matrices are read only at
        // a blend.
        let block = cfg.gram_block_rows;
        let windowed = cfg.window.is_some();
        let acc = move || Acc::new(m, k_total, block, windowed);
        let slow = cfg.session_shrink.map(|_| Box::new(acc()));
        let cadence = crate::Cadence::of(cfg.window_every, cfg.max_rows_between_snapshots);
        let win = match cfg.window {
            Some(w) => Some(Windowed {
                clock: 0.0,
                snaps: crate::Snapshots::with_cadence(w, cadence)?,
            }),
            None => None,
        };
        Ok(Self {
            acc: acc(),
            slow,
            wsig: vec![0.0; m],
            sig2: vec![0.0; m],
            beta: None,
            clock_since_solve: 0.0,
            rows_since_solve: 0,
            weight_since_solve: 0.0,
            solve_failures: 0,
            win,
            zbuf: vec![0.0; k_total],
            ready: Readiness::default(),
            keep_factor: false,
            factors: Factors::default(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &EwRidgeCfg {
        &self.cfg
    }

    /// Per-target EW residual variance of the first slot's prediction; under
    /// a `window`, the variance *inside* it. The bank's `sigma` and `zscore`
    /// are not this: they are the stream's spread per slot, cut at the same
    /// boundary by a ring of its own (`online-polars`' `resid_window`; review
    /// 2026-09-12, S1).
    pub fn sigma2(&self) -> std::borrow::Cow<'_, [f64]> {
        match self.view() {
            Some(v) => std::borrow::Cow::Owned(v.sig2),
            None => std::borrow::Cow::Borrowed(&self.sig2),
        }
    }

    /// The Grams the fit is read from, one entry per Gram with the targets
    /// that read it ([`GramPart`]), and the target moments: what `Bank::gram`
    /// and a closed row report. Under a `window` that has truncated the
    /// accumulators, every number is the window's, so each Gram solves to the
    /// fit `coef` reports (review 2026-09-12, S19), and the target moments
    /// are `None`: the window's snapshots do not carry them, and the export
    /// says it cannot give them rather than giving the whole history's.
    pub fn gram_parts(&self) -> (Vec<GramPart>, Option<TargetMoments>) {
        let (of, gaps) = (&self.acc.grams.of, self.cfg.target_gaps);
        match self.view() {
            Some(v) => (
                gram_parts(&v.acc.grams, of, &v.acc.cross, &v.acc.wj, gaps),
                v.acc.tm,
            ),
            None => (
                gram_parts(
                    &self.acc.grams.grams,
                    of,
                    &self.acc.cross,
                    &self.acc.wj,
                    gaps,
                ),
                Some(self.acc.tm.clone()),
            ),
        }
    }

    /// The accumulated weight over every row, whatever the targets (hard rule
    /// 8): under a `window`, the weight *inside* it, which stops growing once
    /// the window fills. That is what `min_weight` then gates on.
    pub fn n_eff(&self) -> f64 {
        self.window_weights().map_or(self.acc.cross.w, |(w, _)| w)
    }

    /// Per-target **uncentered** cross-moments `r[t]`, each `k_total` long:
    /// the EW mean of `z·y_t` over the rows target `t` was present on, where
    /// `z` is the feature row with the intercept slot as a constant 1. Formed
    /// from the centred moments the model keeps, `r = c + m_t·ȳ` (see
    /// `crate::gaps::Cross`), so at a level `L` it carries the `L²·ε` of
    /// rounding that the model's own solves with an intercept do not see
    /// (review 2026-09-12, N1). [`Self::gram_parts`] pairs each with the Gram
    /// it is solved against.
    pub fn cross_moments(&self) -> Vec<Vec<f64>> {
        (0..self.cfg.n_targets)
            .map(|j| self.acc.cross.raw(j))
            .collect()
    }

    /// Per-target accumulated weight, the denominator behind
    /// [`Self::cross_moments`]. This is `n_eff` *per target*, which differs
    /// from the shared [`Self::n_eff`] when targets have different null
    /// patterns.
    pub fn target_weights(&self) -> &[f64] {
        &self.acc.wj
    }

    /// Per-target mean, variance and `Sum w^2` -- the other half of what a
    /// saved Gram needs (docs/ENHANCEMENTS.md E45).
    pub fn target_moments(&self) -> Option<&TargetMoments> {
        Some(&self.acc.tm)
    }

    /// Current coefficients per output slot, if solved.
    pub fn coefficients(&self) -> Option<&[Vec<f64>]> {
        self.beta.as_deref()
    }

    /// Gram `g` of the live accumulators. Test helper.
    #[cfg(test)]
    pub(crate) fn gram(&self, g: usize) -> &EwCov {
        &self.acc.grams.grams[g]
    }

    /// Re-solve and return the first target's first slope. Test helper. A
    /// blend now re-solves on its own (review 2026-09-12, C3), so the solve
    /// here repeats one on the same accumulators and changes nothing.
    #[cfg(test)]
    pub(crate) fn coefficients_after_blend(&mut self) -> f64 {
        self.solve();
        self.coefficients().unwrap()[0][1]
    }

    /// Mix the main accumulators toward the slow twin, as a session boundary
    /// asks for (see [`EwRidgeCfg::session_shrink`]). A no-op when the twin is
    /// not configured, or before it has seen anything.
    pub fn blend_toward_long_run(&mut self) {
        let Some(f) = self.cfg.session_shrink else {
            return;
        };
        if f <= 0.0 {
            return;
        }
        let Some(slow) = self.slow.as_mut() else {
            return;
        };
        // Every matrix is read in full; a held block goes in first.
        slow.grams.flush();
        self.acc.grams.flush();
        let Some(slow) = self.slow.as_deref() else {
            return;
        };
        // The blend moved the accumulators the fit is read from, so the fit
        // moves with it: re-solve now, when anything was mixed and there is a
        // fit to replace. Left to the schedule, the first rows of the new
        // session -- the rows the blend exists for -- were predicted with the
        // coefficients from before it, and so was every row `predict` scored
        // on a blended copy (review 2026-09-12, C3). A blend with no weight on
        // either side stays the no-op the doc promises, re-solve included.
        if self.acc.blend(slow, f) && self.beta.is_some() {
            self.solve();
        }
    }

    fn z(&mut self, x: &[f64]) {
        if self.cfg.fit_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }
    }

    /// Indices into z for a combo's feature set (intercept first if configured).
    fn combo_z_indices(&self, fs_idx: usize) -> Vec<usize> {
        let off = usize::from(self.cfg.fit_intercept);
        let mut idx: Vec<usize> = if self.cfg.feature_sets.is_empty() {
            (0..self.cfg.n_features).map(|i| i + off).collect()
        } else {
            self.cfg.feature_sets[fs_idx]
                .1
                .iter()
                .map(|&i| i + off)
                .collect()
        };
        if self.cfg.fit_intercept {
            idx.insert(0, 0);
        }
        idx
    }

    /// The accumulators a fit is read from: the live ones, or -- with a
    /// `window` -- the same ones with everything older than the window
    /// subtracted off (docs/PLAN.md §13).
    ///
    /// The Grams, the per-target cross-moments and the residual variance are
    /// all sums of per-row contributions, so each is truncated by the same
    /// identity, `A(t) - lam^(t-u)·A(u)` for the boundary snapshot at `u` --
    /// the first two in the centred form they are kept in, by the pooling
    /// identity (`crate::truncated`, `Acc::window`). The solve then runs on
    /// Grams that provably contain no row older than the window, which is the
    /// whole point -- a rolling regression with a guarantee rather than a
    /// decay. `None` when there is no window or nothing has aged out of it;
    /// nothing inside the window at all -- a clock gap longer than it -- is an
    /// *empty* view, never the live state (review 2026-09-12, C2).
    fn view(&self) -> Option<RidgeView> {
        let win = self.win.as_ref()?;
        let (u, old) = win.snaps.boundary()?;
        let f = self.cfg.decay.factor(win.clock - u);
        let acc = self.acc.window(&old.acc, f)?;
        let m = self.cfg.n_targets;
        let sig2 = if acc.cross.w > 0.0 {
            // The residual variance is one number per target, so the same
            // subtraction on a one-element mean. A target whose residual
            // history is entirely outside the window reports no spread rather
            // than a stale one.
            (0..m)
                .map(|j| {
                    crate::truncated_mean(
                        self.wsig[j],
                        &[self.sig2[j]],
                        old.wsig[j],
                        &[old.sig2[j]],
                        f,
                    )
                    .map_or(0.0, |(_w, s2)| s2[0].max(0.0))
                })
                .collect()
        } else {
            vec![0.0; m]
        };
        Some(RidgeView { acc, sig2 })
    }

    /// The window's weights alone, over every row and per target, as
    /// [`Self::view`] has them: what `predict` and `n_eff` read on every row,
    /// without the view's O(k²) truncation (review 2026-09-12, P1). `None`
    /// where the live accumulators are the answer.
    fn window_weights(&self) -> Option<(f64, Vec<f64>)> {
        let win = self.win.as_ref()?;
        let (u, old) = win.snaps.boundary()?;
        let f = self.cfg.decay.factor(win.clock - u);
        self.acc.window_weights(&old.acc, f)
    }

    fn solve(&mut self) {
        // A held Gram block goes in before the matrix is read. Solves are
        // scheduled by the clock and the row count, never by a chunk end, so
        // the block boundary this moves is the same whichever way the stream
        // was chunked.
        self.acc.grams.flush();
        let k_total = self.cfg.k_total();
        let m = self.cfg.n_targets;
        let combos = self.cfg.combos();
        let nc = combos.len();
        let mut beta = vec![vec![0.0; k_total]; m * nc];
        // The readiness statistics of the last solve, carried forward for a
        // combo whose solve fails at every jitter (it keeps its previous
        // coefficients, and so its previous shares), and resized on the
        // first. Taken out first: the loop below borrows the accumulators.
        let mut ready = std::mem::take(&mut self.ready);
        ready.edf.resize(m * nc, f64::NAN);
        ready.support.resize(m * nc, vec![f64::NAN; k_total]);
        ready.system_of.resize(m * nc, usize::MAX);
        ready.pending.resize(m * nc, None);
        let mut factors = std::mem::take(&mut self.factors).0;
        let keep_factor = self.keep_factor;
        // With a `window`, the fit is solved from the truncated accumulators:
        // no row older than the window is in a Gram at all. Without one, and
        // before anything has aged out, these borrow the live state.
        let view = self.view();
        if view.as_ref().is_some_and(|v| v.acc.cross.w <= 0.0) {
            // An empty window has nothing to solve and no fit to report (C2).
            self.beta = Some(vec![vec![f64::NAN; k_total]; m * nc]);
            self.ready = ready;
            self.factors = Factors(factors);
            self.clock_since_solve = 0.0;
            self.rows_since_solve = 0;
            self.weight_since_solve = 0.0;
            return;
        }
        let mut failures = 0u64;
        let (grams, cross) = match view.as_ref() {
            Some(v) => (v.acc.grams.as_slice(), &v.acc.cross),
            None => (self.acc.grams.grams.as_slice(), &self.acc.cross),
        };
        let n_systems = grams.len() * nc;
        ready.systems.resize(n_systems, None);
        factors.resize(n_systems, None);
        // With an intercept the slopes are solved on the centred system (see
        // the module docs). The rest read the raw normal equations and need
        // the uncentred cross-moments: `ridge_scale`, whose intercept is
        // penalized and so part of the system rather than eliminated from it,
        // and the two solves through the origin, which centre nothing.
        let centred = self.cfg.fit_intercept && !self.cfg.ridge_scale;

        // One system per Gram, for the targets that read it.
        for (g, cov) in grams.iter().enumerate() {
            let readers = self.acc.grams.readers(g);
            let mr = readers.len();
            let r: Vec<Vec<f64>> = if centred {
                Vec::new()
            } else {
                readers.iter().map(|&j| cross.raw(j)).collect()
            };
            for (ci, &(fs_idx, r_idx)) in combos.iter().enumerate() {
                let ridge = self.cfg.ridge[r_idx];
                // A Gram with no weight and no penalty has no system to
                // solve: under `own_rows` a target not seen yet is alone in
                // one, and its coefficients stay zero -- what the jittered
                // solve of an empty matrix gave, at one counted failure a
                // solve. With a penalty the fit is the prior, `coef_prior` or
                // zero, and is solved. No system, so no readiness shares
                // either: the last solve's stood beside the zeros, a share of
                // data under no fit (review 2026-10-05, CA6).
                if ridge == 0.0 && cov.n_eff() <= 0.0 {
                    for &j in &readers {
                        let slot = j * nc + ci;
                        ready.edf[slot] = f64::NAN;
                        ready.support[slot].fill(f64::NAN);
                        ready.pending[slot] = None;
                        ready.system_of[slot] = usize::MAX;
                    }
                    continue;
                }
                let zidx = self.combo_z_indices(fs_idx);
                let kc = zidx.len();

                let solved: Option<Solved> = if centred {
                    self.solve_centred(cov, cross, &readers, &mut failures, &zidx, ridge)
                } else {
                    // Gather the sub-block of S and the per-target rhs.
                    let mut a = vec![0.0; kc * kc];
                    for (ai, &zi) in zidx.iter().enumerate() {
                        for (aj, &zj) in zidx.iter().enumerate() {
                            a[ai * kc + aj] = cov.raw(zi, zj);
                        }
                    }
                    let mut b = vec![0.0; kc * mr];
                    for (jj, rj) in r.iter().enumerate() {
                        for (ai, &zi) in zidx.iter().enumerate() {
                            b[jj * kc + ai] = rj[zi];
                        }
                    }
                    if self.cfg.ridge_scale {
                        // (W S + prior_scale * ridge I) beta = W r  — intercept
                        // penalized, `W` the Gram's weight: under `own_rows`
                        // the target's own, so both sides are sums over its
                        // rows.
                        let w = cov.n_eff();
                        let ps = cov.prior_scale();
                        for i in 0..kc {
                            for j in 0..kc {
                                a[i * kc + j] *= w;
                            }
                            a[i * kc + i] += ps * ridge;
                        }
                        for v in b.iter_mut() {
                            *v *= w;
                        }
                        // Warm start: the prior enters on the same decaying sum
                        // scale, so its weight falls away as data accumulates.
                        if let Some(c0) = &self.cfg.coef_prior {
                            for (jj, &j) in readers.iter().enumerate() {
                                for (ai, &zi) in zidx.iter().enumerate() {
                                    b[jj * kc + ai] += ps * ridge * c0[j][zi];
                                }
                            }
                        }
                        // The system is on the sum scale, so a row form
                        // against it is scaled back by `W` (see `System`).
                        Self::run_solve(&mut failures, &a, &b, kc, mr)
                            .map(|(sol, factor)| Solved::raw(sol, factor, a, kc, ps * ridge, w))
                    } else if !self.cfg.standardize {
                        // Through the origin every slot is a slope, and every
                        // slot is penalized.
                        for i in 0..kc {
                            a[i * kc + i] += ridge;
                        }
                        // Warm prior: shrink toward coef_prior rather than
                        // toward zero, by moving the penalty's target into the
                        // right-hand side.
                        if let Some(c0) = &self.cfg.coef_prior {
                            for (jj, &j) in readers.iter().enumerate() {
                                for (ai, &zi) in zidx.iter().enumerate() {
                                    b[jj * kc + ai] += ridge * c0[j][zi];
                                }
                            }
                        }
                        Self::run_solve(&mut failures, &a, &b, kc, mr)
                            .map(|(sol, factor)| Solved::raw(sol, factor, a, kc, ridge, 1.0))
                    } else {
                        Self::solve_scaled_through_origin(
                            cov,
                            &mut failures,
                            &zidx,
                            &b,
                            &readers,
                            self.cfg.coef_prior.as_deref(),
                            ridge,
                        )
                    }
                };

                if let Some(mut solved) = solved {
                    for (jj, &j) in readers.iter().enumerate() {
                        for (ai, &zi) in zidx.iter().enumerate() {
                            beta[j * nc + ci][zi] = solved.sol[jj * kc + ai];
                        }
                    }
                    // What the readiness statistics read from this system
                    // (docs/WARMUP-AND-CONVERGENCE.md §2.2): each kept
                    // column's data share `1 − λ (A⁻¹)_jj`, with `λ` the
                    // penalty on the diagonal plus the jitter the factor
                    // needed, since that is what it carries; a column the
                    // standardiser dropped has share 0; the intercept is not
                    // a share. The effective degrees of freedom are the
                    // shares summed, plus one for an eliminated intercept,
                    // which the data alone determines. `A⁻¹`'s diagonal is
                    // `O(k³)`, and a solve runs every few rows, so it is
                    // taken when something reads it: a `coef` row, the
                    // noise gate where its bound does not decide, `summary`,
                    // a save, or the end of the run (`Pending`; docs/PLAN.md
                    // task 140).
                    let is_intercept = |pos: usize| self.cfg.fit_intercept && pos == 0;
                    let mut sup0 = vec![f64::NAN; k_total];
                    for (pos, &zi) in zidx.iter().enumerate() {
                        if !is_intercept(pos) {
                            sup0[zi] = 0.0;
                        }
                    }
                    let edf0 = if solved.centred { 1.0 } else { 0.0 };
                    let factor = solved.factor.take();
                    let kept = if keep_factor { factor.clone() } else { None };
                    let pending = factor.map(|factor| {
                        let lam = solved.shift + factor.jitter();
                        Arc::new(Pending {
                            numbers: lam.is_finite() && factor.inverse_is_finite(),
                            lam,
                            to: solved
                                .keep
                                .iter()
                                .map(|&pos| (!is_intercept(pos)).then(|| zidx[pos]))
                                .collect(),
                            factor,
                            edf0,
                            sup0: sup0.clone(),
                            shares: OnceLock::new(),
                        })
                    });
                    let at = g * nc + ci;
                    for &j in &readers {
                        let slot = j * nc + ci;
                        match &pending {
                            Some(p) => ready.pending[slot] = Some(Arc::clone(p)),
                            None => {
                                ready.edf[slot] = edf0;
                                ready.support[slot].clone_from(&sup0);
                                ready.pending[slot] = None;
                            }
                        }
                        ready.system_of[slot] = at;
                    }
                    if keep_factor {
                        ready.systems[at] = Some(System {
                            a: solved.a,
                            z: solved.keep.iter().map(|&p| zidx[p]).collect(),
                            s: solved.s,
                            mean: solved.mean,
                            centred: solved.centred,
                            scale: solved.scale,
                            gram: g,
                        });
                        factors[at] = kept;
                    }
                } else if let Some(prev) = &self.beta {
                    // Total failure even with jitter: keep the previous
                    // coefficients.
                    for &j in &readers {
                        beta[j * nc + ci] = prev[j * nc + ci].clone();
                    }
                } else {
                    // And with no fit before it, no fit: the zeros `beta`
                    // starts at predicted 0.0 as though they had been
                    // learned (review 2026-10-05, CA5).
                    for &j in &readers {
                        beta[j * nc + ci].fill(f64::NAN);
                    }
                }
            }
        }
        // A target the window holds no row of has no fit to report (C2).
        if let Some(v) = view.as_ref() {
            for (j, &w) in v.acc.wj.iter().enumerate() {
                if w <= 0.0 {
                    for ci in 0..nc {
                        beta[j * nc + ci].fill(f64::NAN);
                    }
                }
            }
        }
        if !keep_factor {
            ready.systems.clear();
            factors.clear();
        }
        self.solve_failures += failures;
        self.beta = Some(beta);
        self.ready = ready;
        self.factors = Factors(factors);
        self.clock_since_solve = 0.0;
        self.rows_since_solve = 0;
        self.weight_since_solve = 0.0;
    }

    /// Factorize and solve, counting the jitter it took; the factor comes
    /// back with the solution so the readiness statistics can read it
    /// (docs/WARMUP-AND-CONVERGENCE.md §2.1). The solution is
    /// `crate::solve_spd`'s to the bit: both are the same factor's `solve`.
    fn run_solve(
        failures: &mut u64,
        a: &[f64],
        b: &[f64],
        k: usize,
        m: usize,
    ) -> Option<(Vec<f64>, SpdFactor)> {
        match SpdFactor::of(a, k) {
            Some(f) => {
                *failures += u64::from(f.attempts());
                Some((f.solve(b, k, m), f))
            }
            None => {
                *failures += 1;
                None
            }
        }
    }

    /// The standardized solve through the origin, on one combo's raw
    /// sub-block: scaled by the raw second-moment diagonals -- there is no
    /// centering here, so no cancellation either -- solved and unscaled. `b`
    /// is the reader-major uncentred right-hand side, `readers` the targets
    /// it holds, and `zidx` maps the combo's slots to accumulator indices.
    /// A warm prior shrinks toward `coef_prior` rather than toward zero,
    /// entering the right-hand side on the standardized scale as
    /// `ridge · c0 · s` (a coefficient there is `beta · s`), the same form
    /// as [`EwRidge::solve_centred`]'s. This branch alone never read the
    /// prior, so a standardized fit through the origin was un-warmed with
    /// no error (review 2026-09-18, B2).
    #[allow(clippy::too_many_arguments)]
    fn solve_scaled_through_origin(
        cov: &EwCov,
        failures: &mut u64,
        zidx: &[usize],
        b: &[f64],
        readers: &[usize],
        coef_prior: Option<&[Vec<f64>]>,
        ridge: f64,
    ) -> Option<Solved> {
        let kc = zidx.len();
        let m = readers.len();
        let s: Vec<f64> = (0..kc)
            .map(|i| cov.raw(zidx[i], zidx[i]).max(0.0).sqrt())
            .collect();
        // No centering here, so no cancellation: any strictly positive raw
        // moment is usable.
        let keep: Vec<usize> = (0..kc).filter(|&i| s[i] > 0.0).collect();
        let kk = keep.len();
        if kk == 0 {
            return Some(Solved {
                sol: vec![0.0; kc * m],
                factor: None,
                a: Vec::new(),
                keep,
                s: Vec::new(),
                mean: Vec::new(),
                centred: false,
                shift: ridge,
                scale: 1.0,
            });
        }
        let mut asub = vec![0.0; kk * kk];
        for (i2, &i) in keep.iter().enumerate() {
            for (j2, &j) in keep.iter().enumerate() {
                asub[i2 * kk + j2] = cov.raw(zidx[i], zidx[j]) / (s[i] * s[j]);
            }
            asub[i2 * kk + i2] += ridge;
        }
        let mut bsub = vec![0.0; kk * m];
        for (jj, &j) in readers.iter().enumerate() {
            for (i2, &i) in keep.iter().enumerate() {
                bsub[jj * kk + i2] = b[jj * kc + i] / s[i];
                if let Some(c0) = coef_prior {
                    bsub[jj * kk + i2] += ridge * c0[j][zidx[i]] * s[i];
                }
            }
        }
        let (sol, factor) = Self::run_solve(failures, &asub, &bsub, kk, m)?;
        let mut out = vec![0.0; kc * m];
        for j in 0..m {
            for (i2, &i) in keep.iter().enumerate() {
                out[j * kc + i] = sol[j * kk + i2] / s[i];
            }
        }
        let s_kept = keep.iter().map(|&i| s[i]).collect();
        Some(Solved {
            sol: out,
            factor: Some(factor),
            a: asub,
            keep,
            s: s_kept,
            mean: vec![0.0; kk],
            centred: false,
            shift: ridge,
            scale: 1.0,
        })
    }

    /// The two solves with an intercept, plain and standardized, on the
    /// centred system (see the module docs) for the targets `readers` of one
    /// Gram: the slopes from `C`, the Gram's centred co-moments over the
    /// combo's features, and each target's centred cross-moment `c_j`; then
    /// `beta_0 = ȳ_j − m_j·beta`, `m_j` the target's own column means.
    /// Standardized, `C` is scaled to correlation form and ~zero-variance
    /// features are dropped (coefficient 0); plain, the scale is 1 and
    /// nothing is dropped, as the raw system had it. The result is
    /// reader-major, `k_c` slots each.
    ///
    /// Every statistic comes from `cov` and `cross`, the accumulators `solve`
    /// handed in: the truncated view under a `window`, the live state
    /// otherwise. The standardized solve once read the means and the centred
    /// Gram from the live accumulator while its right-hand side came from the
    /// view, so a windowed fit mixed two histories and the window's guarantee
    /// failed silently (review 2026-09-12, C1).
    fn solve_centred(
        &self,
        cov: &EwCov,
        cross: &Cross,
        readers: &[usize],
        failures: &mut u64,
        zidx: &[usize],
        ridge: f64,
    ) -> Option<Solved> {
        // Feature slots are 1..kc; slot 0 is the intercept.
        let kc = zidx.len();
        let kf = kc - 1;
        let mr = readers.len();
        // The system is read straight from the Gram, `C`'s entries as they
        // are needed: the scales and the kept columns need only its
        // diagonal. Unstandardized, every scale is 1 and every column is
        // kept, and a division or product by 1 gives back the same bits, so
        // none is made: at a solve a row the `k²` copy and divisions were a
        // sixth of the solve (docs/PERFORMANCE.md §29).
        let standardize = self.cfg.standardize;
        let var = |i: usize| cov.cov(zidx[i + 1], zidx[i + 1]);
        let (s, keep): (Vec<f64>, Vec<usize>) = if standardize {
            let s = (0..kf).map(|i| var(i).max(0.0).sqrt()).collect();
            // A genuinely constant feature is dropped (coefficient 0) rather
            // than blowing up; with centered accumulators its variance is
            // exactly zero.
            let keep = (0..kf)
                .filter(|&i| crate::variance_is_usable(var(i), cov.raw(zidx[i + 1], zidx[i + 1])))
                .collect();
            (s, keep)
        } else {
            (vec![1.0; kf], (0..kf).collect())
        };
        let kk = keep.len();
        let mut out = vec![0.0; kc * mr];
        let mut system = None;
        if kk > 0 {
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                let row = &mut asub[i2 * kk..(i2 + 1) * kk];
                if standardize {
                    for (a, &j) in row.iter_mut().zip(&keep) {
                        *a = cov.cov(zidx[i + 1], zidx[j + 1]) / (s[i] * s[j]);
                    }
                } else {
                    for (a, &j) in row.iter_mut().zip(&keep) {
                        *a = cov.cov(zidx[i + 1], zidx[j + 1]);
                    }
                }
                row[i2] += ridge;
            }
            let mut bsub = vec![0.0; kk * mr];
            for (jj, &j) in readers.iter().enumerate() {
                for (i2, &i) in keep.iter().enumerate() {
                    let z = zidx[i + 1];
                    bsub[jj * kk + i2] = if standardize {
                        cross.c[j][z] / s[i]
                    } else {
                        cross.c[j][z]
                    };
                    // Warm prior: shrink toward coef_prior rather than toward
                    // zero. It lives in original units, and on the
                    // standardized scale a coefficient is beta * sd.
                    if let Some(c0) = &self.cfg.coef_prior {
                        bsub[jj * kk + i2] += if standardize {
                            ridge * c0[j][z] * s[i]
                        } else {
                            ridge * c0[j][z]
                        };
                    }
                }
            }
            let (sol, factor) = Self::run_solve(failures, &asub, &bsub, kk, mr)?;
            for jj in 0..mr {
                for (i2, &i) in keep.iter().enumerate() {
                    let v = sol[jj * kk + i2];
                    out[jj * kc + i + 1] = if standardize { v / s[i] } else { v };
                }
            }
            system = Some((asub, factor));
        }
        // `m_j` is the Gram's mean, which under `own_rows` is over exactly the
        // target's rows (and the cross-moments' own copy of it to the bit),
        // and the target's own mean under `pairwise`, where the Gram is over
        // every row (`crate::gaps::Cross`).
        let pairwise = self.cfg.target_gaps == TargetGaps::Pairwise;
        for (jj, &j) in readers.iter().enumerate() {
            let mut b0 = cross.my[j];
            for i in 0..kf {
                let z = zidx[i + 1];
                let m_j = if pairwise {
                    cross.mj[j][z]
                } else {
                    cov.mean(z)
                };
                b0 -= m_j * out[jj * kc + i + 1];
            }
            out[jj * kc] = b0;
        }
        // The kept columns as positions in `zidx`, the centred system's
        // scale and the mean it subtracted, for the row form.
        let (a, factor) = match system {
            Some((a, f)) => (a, Some(f)),
            None => (Vec::new(), None),
        };
        let mean = keep.iter().map(|&i| cov.mean(zidx[i + 1])).collect();
        let s_kept = keep.iter().map(|&i| s[i]).collect();
        Some(Solved {
            sol: out,
            factor,
            a,
            keep: keep.iter().map(|&i| i + 1).collect(),
            s: s_kept,
            mean,
            centred: true,
            shift: ridge,
            scale: 1.0,
        })
    }

    /// Make the per-row leverage answerable: keep each solve's system and
    /// factor (docs/WARMUP-AND-CONVERGENCE.md §2.1). Configuration that
    /// travels with the state, set by the stream when `emit_error_inflation`
    /// asks for the field; a model without it keeps nothing and answers
    /// [`OnlineModel::row_error_inflation_into`] with infinity.
    pub fn set_keep_factor(&mut self, on: bool) {
        self.keep_factor = on;
        if !on {
            self.ready.systems.clear();
            self.factors.0.clear();
        }
    }

    /// Whether the fit and what the last solve left for the readiness
    /// statistics are this cfg's shape (`restore`'s check; review
    /// 2026-10-05, CA1): a slot per target and combo, each `k_total` long,
    /// and per slot an `edf`, a `support` of `k_total` and a `system_of`
    /// naming a system or none -- all of it empty before the first solve,
    /// and only then. A Gram split off since the last solve adds systems
    /// that solve has not sized, so a system's index is held to the Grams
    /// there are now.
    fn fit_has_shape(&self) -> bool {
        let (k, nc) = (self.cfg.k_total(), self.cfg.n_combos());
        let slots = self.cfg.n_targets * nc;
        let n_systems = self.acc.grams.grams.len() * nc;
        let r = &self.ready;
        match &self.beta {
            None => r.edf.is_empty() && r.support.is_empty() && r.system_of.is_empty(),
            Some(beta) => {
                beta.len() == slots
                    && beta.iter().all(|b| b.len() == k)
                    && r.edf.len() == slots
                    && r.support.len() == slots
                    && r.support.iter().all(|s| s.len() == k)
                    && r.system_of.len() == slots
                    && r.system_of
                        .iter()
                        .all(|&at| at == usize::MAX || at < n_systems)
                    && r.systems.len() <= n_systems
            }
        }
    }

    /// Rebuild the kept factors from the systems the state carries: the
    /// factorization is deterministic, so a loaded model reads the same
    /// leverage the saved one did.
    fn refactor(&mut self) {
        self.factors = Factors(
            self.ready
                .systems
                .iter()
                .map(|s| {
                    s.as_ref().and_then(|s| {
                        let kk = s.z.len();
                        (s.a.len() == kk * kk)
                            .then(|| SpdFactor::of(&s.a, kk))
                            .flatten()
                    })
                })
                .collect(),
        );
    }

    /// Kish's effective sample size behind each Gram, as it stands before
    /// the row: the window's where there is one, the live accumulator's
    /// otherwise. `None` for a Gram with no weight, or with no Kish sum.
    fn gram_kish(&self) -> Vec<Option<f64>> {
        let windowed = self.win.as_ref().and_then(|win| {
            let (u, old) = win.snaps.boundary()?;
            let f = self.cfg.decay.factor(win.clock - u);
            self.acc.window_kish(&old.acc, f)
        });
        windowed.unwrap_or_else(|| self.acc.grams.grams.iter().map(EwCov::n_kish).collect())
    }

    /// `sqrt(1 + edf / n_kish)` per slot, as it stands before the row
    /// ([`OnlineModel::error_inflation_into`]): exact wherever the ratio
    /// could reach `limit`. Where a solve's shares have not been taken and
    /// are sure to be numbers, the ratio of the `edf` bound stands in if it
    /// is already below `limit`: the computed ratio is no larger, since
    /// each step to it rounds monotonically, so a gate at `limit` decides
    /// the same either way. `f64::NEG_INFINITY` asks for every value exact.
    fn inflation_into(&self, out: &mut Vec<f64>, limit: f64) -> bool {
        let (m, nc) = (self.cfg.n_targets, self.cfg.n_combos());
        out.clear();
        out.resize(m * nc, f64::INFINITY);
        // Below the model's own floor -- fewer effective observations than
        // its first solve needs -- the estimation variance is unbounded, and
        // the solves the schedule ran before it read degenerate Grams: the
        // ratio is infinite there, as `predict` withholds there.
        if self.beta.is_none()
            || self.ready.edf.len() != m * nc
            || self.n_eff() < self.cfg.min_weight
        {
            return true;
        }
        let kish = self.gram_kish();
        for j in 0..m {
            let Some(n) = kish.get(self.acc.grams.of[j]).copied().flatten() else {
                continue;
            };
            for c in 0..nc {
                let slot = j * nc + c;
                if n > 0.0
                    && let Some(bound) = self.ready.edf_bound_at(slot)
                {
                    let ratio = (1.0 + bound / n).sqrt();
                    if ratio < limit {
                        out[slot] = ratio;
                        continue;
                    }
                }
                let edf = self.ready.edf_at(slot);
                if n > 0.0 && edf.is_finite() {
                    out[slot] = (1.0 + edf / n).sqrt();
                }
            }
        }
        true
    }

    /// Take the shares of every solve nothing has read yet, and drop the
    /// factors they were read from (docs/PLAN.md task 140): what the stream
    /// calls at the end of each run, so a factor is held only while its
    /// stream runs. The values are the ones a read would have given.
    pub fn settle_readiness(&mut self) {
        self.ready.settle();
    }

    /// Slots whose readiness shares still hold the factor of the solve they
    /// come from, read or not (docs/PLAN.md task 140): 0 once
    /// [`Self::settle_readiness`] has run, as it does at the end of every
    /// run of a stream.
    pub fn pending_readiness(&self) -> usize {
        self.ready.pending.iter().flatten().count()
    }

    /// The `z` slot `zi` of a feature row, without the augmentation buffer:
    /// the intercept's constant 1, else the feature.
    #[inline]
    fn z_at(&self, x: &[f64], zi: usize) -> f64 {
        if self.cfg.fit_intercept {
            if zi == 0 { 1.0 } else { x[zi - 1] }
        } else {
            x[zi]
        }
    }
}

/// One combo's solve, and what the readiness statistics read from it
/// (docs/WARMUP-AND-CONVERGENCE.md §2.1, §2.2).
struct Solved {
    /// Reader-major coefficients, `kc` slots each, in original units.
    sol: Vec<f64>,
    /// The factor of `a`; `None` when nothing was factorized (every column
    /// dropped).
    factor: Option<SpdFactor>,
    /// The ridged, normalised system as factorized, row-major over the kept
    /// columns, the jitter excluded (the factor knows it).
    a: Vec<f64>,
    /// Positions in the combo's `zidx` of the kept columns, in the system's
    /// order.
    keep: Vec<usize>,
    /// The scale each kept column was divided by, and the mean subtracted
    /// from it (1 and 0 where the system is raw).
    s: Vec<f64>,
    mean: Vec<f64>,
    /// Centred, with the intercept eliminated.
    centred: bool,
    /// The penalty on the system's diagonal, before jitter.
    shift: f64,
    /// What a row form against `a` is scaled by to be in the mean-form
    /// Gram's units: the Gram's weight under `ridge_scale`, else 1.
    scale: f64,
}

impl Solved {
    /// A raw, uncentred system over every column of the combo.
    fn raw(
        sol: Vec<f64>,
        factor: SpdFactor,
        a: Vec<f64>,
        kc: usize,
        shift: f64,
        scale: f64,
    ) -> Self {
        Self {
            sol,
            factor: Some(factor),
            a,
            keep: (0..kc).collect(),
            s: vec![1.0; kc],
            mean: vec![0.0; kc],
            centred: false,
            shift,
            scale,
        }
    }
}

/// One solved system, kept so a row's leverage can be read against the
/// factor its fit came from (docs/WARMUP-AND-CONVERGENCE.md §2.1): the
/// ridged, normalised sub-block the solve factorized, and how a feature row
/// is mapped into its space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct System {
    /// Row-major `kk × kk`, the jitter excluded; the factor is rebuilt from
    /// it on load.
    a: Vec<f64>,
    /// The accumulator index (`z` slot) of each kept column.
    z: Vec<usize>,
    /// The scale each kept column is divided by, and the mean subtracted.
    s: Vec<f64>,
    mean: Vec<f64>,
    /// Centred, the intercept eliminated: its share of the leverage is then
    /// the 1 of the mean.
    centred: bool,
    /// See [`Solved::scale`].
    scale: f64,
    /// The Gram this system was solved from.
    gram: usize,
}

/// What the last solve left for the readiness statistics
/// (docs/WARMUP-AND-CONVERGENCE.md §2): per output slot (target-major, then
/// combo), the effective degrees of freedom the fit used and each
/// coefficient's data share; and, under `keep_factor`, the systems the
/// per-row leverage is read against.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Readiness {
    /// Per slot: an eliminated intercept counts 1, each kept slope its share.
    #[serde(default, with = "crate::humanfloat::vec_f64_or_tag")]
    edf: Vec<f64>,
    /// Per slot, `k_total` long: NaN in the intercept slot and outside the
    /// slot's feature set, 0 for a column the standardiser dropped. Tagged
    /// in the human-readable export, since the intercept's NaN is the rule.
    #[serde(default, with = "crate::humanfloat::vec_vec_f64_or_tag")]
    support: Vec<Vec<f64>>,
    /// Per `(gram, combo)`, `gram * n_combos + combo`; empty unless kept.
    systems: Vec<Option<System>>,
    /// Per slot, its index into `systems`.
    system_of: Vec<usize>,
    /// Per slot, the last solve, until the end of the run stores its
    /// shares: while one is here, it is the slot's `edf` and `support`,
    /// computed on the first read, and the two stored above are an earlier
    /// solve's. Not state: a save writes the shares it would give
    /// (`serialize_settled`), and the end of a run stores them and drops
    /// the factor ([`Readiness::settle`]), so the memory lasts only while a
    /// stream runs (docs/PLAN.md task 140).
    #[serde(skip)]
    pending: Vec<Option<Arc<Pending>>>,
}

/// One solve's shares, taken when something reads them (docs/PLAN.md task
/// 140). `A⁻¹`'s diagonal is `O(k³)` and a solve runs every few rows:
/// taken at every solve, it cost `ewridge` up to 21% of its throughput at
/// `k = 20` and 46% at `k = 50` (docs/PERFORMANCE.md §29). Something reads
/// it far less often: a `coef` row, by default one per group per chunk,
/// and the noise gate only while its bound cannot decide. The arithmetic is
/// the solve's, in its order, so the shares are the same bits whenever they
/// are taken.
#[derive(Debug)]
struct Pending {
    factor: SpdFactor,
    /// The penalty on the diagonal plus the jitter the factor needed.
    lam: f64,
    /// Per kept column, in the factor's order: the accumulator slot its
    /// share goes to, or `None` for an intercept, which is not a share.
    to: Vec<Option<usize>>,
    /// The degrees of freedom before any share: 1 for an eliminated
    /// intercept.
    edf0: f64,
    /// `support` before any share: NaN outside the combo and in the
    /// intercept, 0 for a column the standardiser dropped.
    sup0: Vec<f64>,
    /// Every share is sure to be a number: `lam` is finite and so is `A⁻¹`'s
    /// diagonal ([`SpdFactor::inverse_is_finite`]). A share can be NaN --
    /// no ridge and an inverse that overflowed make `1 − 0·∞` -- and then
    /// the exact ratio is infinite, which the bound would not say.
    numbers: bool,
    shares: OnceLock<(f64, Vec<f64>)>,
}

impl Pending {
    /// The effective degrees of freedom and every coefficient's data share,
    /// computed on the first read.
    fn shares(&self) -> &(f64, Vec<f64>) {
        self.shares.get_or_init(|| {
            let inv = self.factor.inverse_diagonal(self.to.len());
            let mut edf = self.edf0;
            let mut sup = self.sup0.clone();
            for (i2, to) in self.to.iter().enumerate() {
                let share = (1.0 - self.lam * inv[i2]).clamp(0.0, 1.0);
                edf += share;
                if let Some(zi) = *to {
                    sup[zi] = share;
                }
            }
            (edf, sup)
        })
    }

    /// Above or at the `edf` the shares would give, without taking them,
    /// where they are sure to be numbers: each is then in `[0, 1]`, and
    /// adding numbers no larger rounds to nothing larger, so `edf0 + k`
    /// bounds the computed sum too. `None` where a share might be NaN.
    fn edf_bound(&self) -> Option<f64> {
        self.numbers.then_some(self.edf0 + self.to.len() as f64)
    }
}

impl Readiness {
    /// Slot `s`'s effective degrees of freedom.
    fn edf_at(&self, s: usize) -> f64 {
        match self.pending.get(s) {
            Some(Some(p)) => p.shares().0,
            _ => self.edf[s],
        }
    }

    /// Slot `s`'s data shares, `k_total` long.
    fn support_at(&self, s: usize) -> &[f64] {
        match self.pending.get(s) {
            Some(Some(p)) => &p.shares().1,
            _ => &self.support[s],
        }
    }

    /// A bound on slot `s`'s `edf` when its shares have not been taken yet
    /// and are sure to be numbers; `None` when the exact value costs
    /// nothing more to read, or a bound could not stand in for it.
    fn edf_bound_at(&self, s: usize) -> Option<f64> {
        match self.pending.get(s) {
            Some(Some(p)) if p.shares.get().is_none() => p.edf_bound(),
            _ => None,
        }
    }

    fn has_pending(&self) -> bool {
        self.pending.iter().any(Option::is_some)
    }

    /// Take every pending solve's shares into the stored ones and drop the
    /// factors they were read from.
    fn settle(&mut self) {
        for (s, slot) in self.pending.iter_mut().enumerate() {
            if let Some(p) = slot.take() {
                let (edf, sup) = p.shares();
                self.edf[s] = *edf;
                self.support[s].clone_from(sup);
            }
        }
    }

    /// A copy with every pending solve's shares taken: what the state holds.
    fn settled(&self) -> Self {
        let mut r = self.clone();
        r.settle();
        r
    }
}

/// `Readiness` as the state carries it: every pending solve's shares taken,
/// so a save writes the values a read would give, whenever it comes.
fn serialize_settled<S: serde::Serializer>(r: &Readiness, s: S) -> Result<S::Ok, S::Error> {
    if r.has_pending() {
        r.settled().serialize(s)
    } else {
        r.serialize(s)
    }
}

/// Two floats are the same value when their bits are: the intercept's share
/// is NaN by definition, and `NaN != NaN` would make a model unequal to its
/// own clone.
fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

impl PartialEq for Readiness {
    /// On the values a read gives: a pending solve's shares, not the stale
    /// ones stored beneath them.
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (settled_view(self), settled_view(other));
        same_bits(&a.edf, &b.edf)
            && a.support.len() == b.support.len()
            && a.support
                .iter()
                .zip(&b.support)
                .all(|(x, y)| same_bits(x, y))
            && a.systems == b.systems
            && a.system_of == b.system_of
    }
}

/// A `Readiness` as its values stand, borrowed where nothing is pending.
fn settled_view(r: &Readiness) -> Cow<'_, Readiness> {
    if r.has_pending() {
        Cow::Owned(r.settled())
    } else {
        Cow::Borrowed(r)
    }
}

/// The kept factors, one per entry of `Readiness::systems`. Not state: a
/// deterministic function of the systems, rebuilt on load, and no part of
/// what two models are compared on.
#[derive(Debug, Clone, Default)]
struct Factors(Vec<Option<SpdFactor>>);

impl PartialEq for Factors {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl OnlineModel for EwRidge {
    fn set_solve_share(&mut self, share: Option<f64>) {
        self.cfg.solve_share = share;
    }

    fn solve_share(&self) -> Option<f64> {
        self.cfg.solve_share
    }

    fn stamp_next(&mut self, stamp: crate::Stamp) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.stamp_next(stamp);
        }
    }

    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.set_budget(budget);
        }
    }

    fn window_over_budget(&self) -> Option<(usize, crate::Cadence)> {
        self.win.as_ref().and_then(|win| win.snaps.over_budget())
    }

    /// The ring's shadow, its snapshots formed as the step forms them
    /// (docs/PLAN.md task 115 (d)).
    fn window_shadow(&self) -> Option<crate::WindowShadow> {
        let win = self.win.as_ref()?;
        Some(crate::WindowShadow::new(win.clock, &win.snaps, || {
            RidgeMoments {
                acc: self.acc.snapshot(1.0),
                wsig: self.wsig.clone(),
                sig2: self.sig2.clone(),
            }
        }))
    }
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        match self.window_weights() {
            Some((_, wj)) => out.extend_from_slice(&wj),
            None => out.extend_from_slice(&self.acc.wj),
        }
        true
    }

    /// `sqrt(1 + edf / n_kish)` per slot: the last solve's effective degrees
    /// of freedom over Kish's sample size behind the Gram the target reads
    /// now, as it stands before the row (docs/WARMUP-AND-CONVERGENCE.md
    /// §2.1). Infinite before the first solve and where the Gram has no
    /// weight.
    fn error_inflation_into(&self, out: &mut Vec<f64>) -> bool {
        self.inflation_into(out, f64::NEG_INFINITY)
    }

    /// The gate needs only which side of `limit` a slot is on, so where a
    /// solve's shares have not been taken and the `edf` bound already puts
    /// the ratio below it, the bound stands in and the `O(k³)` read waits
    /// (docs/PLAN.md task 140).
    fn error_inflation_gate_into(&self, out: &mut Vec<f64>, limit: f64) -> bool {
        self.inflation_into(out, limit)
    }

    /// `sqrt(1 + h(x))` per slot, `h(x)` the row's leverage against the
    /// factor its fit came from over Kish's sample size: for a centred
    /// system the mean's own `1` plus the centred, scaled row's quadratic
    /// form; for a raw one the row's form alone, scaled back to the
    /// mean-form Gram's units under `ridge_scale`. Infinite before the first
    /// solve, and where no system was kept ([`EwRidge::set_keep_factor`]).
    fn row_error_inflation_into(&self, x: &[f64], out: &mut Vec<f64>) -> bool {
        let (m, nc) = (self.cfg.n_targets, self.cfg.n_combos());
        out.clear();
        out.resize(m * nc, f64::INFINITY);
        if self.beta.is_none()
            || self.ready.system_of.len() != m * nc
            || self.n_eff() < self.cfg.min_weight
        {
            return true;
        }
        let kish = self.gram_kish();
        let mut v = Vec::new();
        for j in 0..m {
            let Some(n) = kish.get(self.acc.grams.of[j]).copied().flatten() else {
                continue;
            };
            if n <= 0.0 {
                continue;
            }
            for c in 0..nc {
                let slot = j * nc + c;
                let at = self.ready.system_of[slot];
                let (Some(Some(sys)), Some(Some(factor))) =
                    (self.ready.systems.get(at), self.factors.0.get(at))
                else {
                    continue;
                };
                let kk = sys.z.len();
                v.clear();
                v.extend(
                    sys.z
                        .iter()
                        .zip(&sys.s)
                        .zip(&sys.mean)
                        .map(|((&zi, &s), &mu)| (self.z_at(x, zi) - mu) / s),
                );
                let q = if kk > 0 {
                    factor.quad_forms(&v, kk, 1)[0] * sys.scale
                } else {
                    0.0
                };
                let h = (q + if sys.centred { 1.0 } else { 0.0 }) / n;
                out[slot] = (1.0 + h).sqrt();
            }
        }
        true
    }

    fn support_coef(&self) -> Option<Vec<Vec<f64>>> {
        self.beta.as_ref()?;
        let (m, nc) = (self.cfg.n_targets, self.cfg.n_combos());
        (self.ready.support.len() == m * nc).then(|| {
            (0..m * nc)
                .map(|s| self.ready.support_at(s).to_vec())
                .collect()
        })
    }

    fn step(&mut self, x: &[f64], y: &[Option<f64>], d_clock: f64, weight: f64) -> Step {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        debug_assert_eq!(y.len(), self.cfg.n_targets);
        let m = self.cfg.n_targets;
        let nc = self.cfg.n_combos();
        let lam = self.cfg.decay.factor(d_clock);
        let gaps = self.cfg.target_gaps;
        self.z(x);

        // ---- predict (state before this row's update) ----
        let out = self.predict(x, d_clock);
        let pred = &out.pred;

        // ---- update ----
        // The slow twin sees the same rows under its own, longer half-life.
        if let (Some(slow), Some(h)) = (self.slow.as_mut(), self.cfg.long_half_life) {
            let slow_lam = Decay::Halflife(h).factor(d_clock);
            slow.learn(&self.zbuf, y, slow_lam, weight, gaps);
        }
        // The snapshot is every accumulator as it stands *before* this row,
        // decayed to this row's clock, so subtracting it later retains this
        // row and everything after. Keyed by the row's stamp (task 175)
        // beside a clock the model accumulates itself, so the boundary cannot
        // depend on the chunking.
        if let Some(win) = self.win.as_mut() {
            // Built inside the closure, so the O(k²) snapshot is only formed
            // on the rows `offer` actually keeps -- the cadence's --
            // rather than on every row and then dropped (review 2026-09-18,
            // P1). The closure reads `acc`/`wsig`/`sig2`, disjoint fields from
            // `win`, so the borrows do not collide.
            win.snaps.learn(&mut win.clock, d_clock, || RidgeMoments {
                acc: self.acc.snapshot(lam),
                wsig: self.wsig.iter().map(|w| w * lam).collect(),
                sig2: self.sig2.clone(),
            });
        }
        // EW residual variance from the primary (first-combo) pred. Its
        // weight ages on every row, and a row with a target, a weight and a
        // prediction adds its squared residual. A row with a target and no
        // prediction -- `min_weight` unmet after a clock gap, say -- ages it
        // as a null row does; it aged nothing, so `σ²` forgot less across
        // such rows than the clock says (N6).
        for j in 0..m {
            let aged = lam * self.wsig[j];
            match y[j] {
                Some(yj) if weight > 0.0 && pred[j * nc].is_finite() => {
                    let resid = yj - pred[j * nc];
                    let ws_new = aged + weight;
                    self.sig2[j] = (aged * self.sig2[j] + weight * resid * resid) / ws_new;
                    self.wsig[j] = ws_new;
                }
                _ => self.wsig[j] = aged,
            }
        }
        self.acc.learn(&self.zbuf, y, lam, weight, gaps);

        // ---- solve schedule ----
        self.clock_since_solve += d_clock;
        self.rows_since_solve += 1;
        if weight.is_finite() && weight > 0.0 {
            self.weight_since_solve += weight;
        }
        let by_cadence = match self.cfg.solve_share {
            Some(share) => self.weight_since_solve >= share * self.n_eff(),
            None => self.cfg.solve_every <= 0.0 || self.clock_since_solve >= self.cfg.solve_every,
        };
        let due = by_cadence
            || self.rows_since_solve >= self.cfg.max_rows_between_solves
            || (self.beta.is_none() && self.n_eff() >= self.cfg.min_weight);
        if due {
            self.solve();
        }
        out
    }

    fn predict(&self, x: &[f64], _d_clock: f64) -> Step {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        let (m, nc) = (self.cfg.n_targets, self.cfg.n_combos());
        // The gate and the per-target test read the window's weights, so a
        // target with nothing in the window reports nothing (C2) -- the
        // weights alone, not the O(k²) view, which is the solve's and was
        // built here every row for these two numbers (review 2026-09-12, P1).
        let weights = self.window_weights();
        let (n_eff, wj) = match weights.as_ref() {
            Some((w, wj)) => (*w, wj.as_slice()),
            None => (self.acc.cross.w, self.acc.wj.as_slice()),
        };
        let mut pred = vec![f64::NAN; m * nc];
        if let (true, Some(beta)) = (n_eff >= self.cfg.min_weight, &self.beta) {
            for j in 0..m {
                if wj[j] > 0.0 {
                    for c in 0..nc {
                        pred[j * nc + c] = dot_aug(&beta[j * nc + c], x, self.cfg.fit_intercept);
                    }
                }
            }
        }
        Step {
            pred,
            n_eff,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::EwRidge(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::EwRidge(m) => {
                let mut m = (**m).clone();
                let (n, k) = (m.cfg.n_targets, m.cfg.k_total());
                // A state written before schema 17 kept each target's own
                // mean as an offset (`crate::gaps::Cross`).
                if s.schema_version < 17 {
                    m.acc.offsets_to_means();
                    if let Some(slow) = m.slow.as_mut() {
                        slow.offsets_to_means();
                    }
                    if let Some(win) = m.win.as_mut() {
                        win.snaps.iter_mut().for_each(|s| s.acc.offsets_to_means());
                    }
                }
                // The slow twin exactly when `session_shrink` asks for one:
                // without it the blend was a silent no-op (review 2026-10-05,
                // CA1).
                let twin = m.slow.is_some() == m.cfg.session_shrink.is_some()
                    && m.slow.as_ref().is_none_or(|s| s.has_shape(n, k));
                // The fit and the readiness statistics at the cfg's shape, as
                // `lasso` and `rls` check theirs: an empty `beta` loaded and
                // `predict` panicked, a short slot predicted a constant, and
                // an empty `edf` withheld every row (review 2026-10-05, CA1).
                let fit = m.fit_has_shape();
                // A ring exactly when the cfg has a window (review 2026-09-26,
                // C5: `ew_cov`, `lasso` and `ew_class` checked this, this one
                // did not, and ran unwindowed without a word).
                if !m.acc.has_shape(n, k)
                    || !twin
                    || !fit
                    || m.wsig.len() != n
                    || m.sig2.len() != n
                    || m.win.is_some() != m.cfg.window.is_some()
                {
                    return Err(StateError::Invalid(
                        "ew_ridge: the accumulators have the wrong shape".into(),
                    ));
                }
                // The runs follow the window, not the file (review 2026-09-26,
                // C4; `EwCovModel::restore` says why).
                match m.cfg.window {
                    None => {
                        m.acc.set_runs_off();
                        if let Some(slow) = m.slow.as_mut() {
                            slow.set_runs_off();
                        }
                    }
                    Some(_) if !m.acc.keeps_runs() => {
                        return Err(StateError::Invalid(
                            "ew_ridge: a windowed state whose runs are off".into(),
                        ));
                    }
                    Some(_) => {}
                }
                m.zbuf = vec![0.0; k];
                m.refactor();
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "ew_ridge",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * self.cfg.n_combos()
    }
}

#[cfg(test)]
mod tests;
