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
//! - ridge_decay:  `(W_S S + prior_scale * ridge I) beta = W_S r_j` — a
//!   decaying prior on the sum scale, penalizing the intercept: exactly
//!   classic RLS regularization (used by the RLS agreement test, task 9).
//!
//! With an intercept, and without `ridge_decay`, the unpenalized intercept is
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
//! `ridge_decay`, there is no intercept to eliminate and the raw system is
//! solved: the Gram's `E[z zᵀ]` against the target's `E[z·y_j]`.
//!
//! Predictions use the last solved coefficients (out-of-sample by construction:
//! the solve happens after the row's update, the pred before it).

use serde::{Deserialize, Serialize};

use crate::gaps::{Acc, AccSnap, AccView, Cross, gram_parts};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::{dot_aug, solve_spd};
use crate::{Decay, EwCov, GramPart, TargetGaps, TargetMoments};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EwRidgeCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub add_intercept: bool,
    pub decay: Decay,
    /// Ridge grid, expanded at solve time; length >= 1.
    pub ridge: Vec<f64>,
    /// Named subsets of feature indices (0-based, excluding the intercept).
    /// Empty means one full set named "all".
    pub feature_sets: Vec<(String, Vec<usize>)>,
    pub standardize: bool,
    /// Decaying sum-scale prior (classic RLS regularization). Incompatible with
    /// `standardize` and with grids; the intercept is penalized.
    pub ridge_decay: bool,
    /// Shrink toward these coefficients instead of toward zero
    /// (ENHANCEMENTS E15): the solve becomes `(S + ridge·D)β = r + ridge·D·β₀`.
    /// One vector per target, each `k_total` long, in the features' original
    /// units; the intercept slot is unpenalized and therefore ignored.
    ///
    /// **Whether the prior fades depends on `ridge_decay`, and the difference
    /// matters.** `S` here is a weighted *mean*, not a sum, so it does not grow
    /// with the sample: a plain `ridge` is a fixed per-observation penalty and
    /// its pull toward `coef_prior` is **permanent** — "always stay near this
    /// belief". With `ridge_decay` the prior sits on the sum scale and its
    /// weight decays with the data, which is the usual warm start: "begin at
    /// yesterday's fit and let evidence take over".
    #[serde(default)]
    pub coef_prior: Option<Vec<Vec<f64>>>,
    /// Blend toward a slow-moving twin on a session change, instead of the
    /// all-or-nothing choice between `session_gap` and a full reset
    /// (ENHANCEMENTS E6, PLAN §12 open question 1).
    ///
    /// A second accumulator runs alongside the main one with `long_halflife`,
    /// representing the long-run relationship. On a session boundary the two
    /// are mixed, weight-respectingly, with the slow one taking share
    /// `session_shrink`:
    ///
    /// ```text
    /// W'  = (1−f)·W_fast + f·W_slow
    /// S'  = ((1−f)·W_fast·S_fast + f·W_slow·S_slow) / W'
    /// ```
    ///
    /// so `0` keeps today's fit, `1` reverts fully to the long run, and
    /// anything between says "overnight, drift partway back". Unlike
    /// `session_gap` this changes *what the model believes*, not merely how
    /// confident it is.
    #[serde(default)]
    pub session_shrink: Option<f64>,
    /// Halflife of the slow twin. Required by `session_shrink`.
    #[serde(default)]
    pub long_halflife: Option<f64>,
    /// Outputs are null until `n_eff` (before the row's update) reaches this.
    pub min_periods: f64,
    /// Solve cadence in clock units; <= 0 solves every row.
    pub solve_every: f64,
    /// Row cap between solves; 1 solves every row.
    pub max_rows_between_solves: u32,
    /// Rows of the Gram update held back and merged as one block
    /// (docs/ENHANCEMENTS.md E51, docs/PLAN.md task 71). `0`, the default,
    /// updates the `k×k` matrix on every row. With `B` here, a row's `z` is
    /// buffered and the matrix is brought up to date once per `B` rows by a
    /// `k×B` times `B×k` product, `6.6×` faster than the rank-one updates at
    /// `k = 1,000` with `B = 256`; the four scalars still run per row, so
    /// `n_eff` and `min_periods` are unchanged to the bit.
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
    /// exponential weight alone would leave `0.5^(age/halflife)` of it
    /// (docs/PLAN.md §13). Inside the window the weights are still
    /// exponential. A halflife grid is one model instance per entry, so each
    /// carries its own ring.
    ///
    /// **Last, with `window_every`, and they must stay last.** The compact
    /// msgpack encoding writes a struct as an *array*, so a
    /// `skip_serializing_if` field anywhere but the end shifts every field
    /// after it when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<f64>,
    /// Learned rows between the snapshots the window is computed from; `1`
    /// (the default) is the tightest boundary, larger divides the memory and
    /// only ever shortens the effective window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_every: Option<usize>,
}

/// The most a held Gram block may take before the model refuses to build,
/// on the pattern of `marginal`'s bins: `k` can be 10,000 here, and silently
/// allocating gigabytes is a worse outcome than an error that names the
/// number.
const GRAM_BLOCK_BUDGET: usize = 256 << 20;

impl EwRidgeCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.add_intercept)
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
        if self.n_features == 0 || self.n_targets == 0 {
            return Err("n_features and n_targets must be >= 1".into());
        }
        if self.ridge.is_empty() {
            return Err("ridge grid must have at least one value".into());
        }
        if let Some(w) = self.window {
            if !w.is_finite() || w <= 0.0 {
                return Err(format!(
                    "ewridge: window must be finite and > 0 (got {w}); it is clock units of \
                     history to keep"
                ));
            }
            if self.window_every.is_some_and(|e| e == 0) {
                return Err("ewridge: window_every must be >= 1".into());
            }
            if self.ridge_decay {
                return Err(
                    "ewridge: window and ridge_decay do not combine; the decaying prior's scale \
                     is the product of every decay factor the stream has applied, which a \
                     window truncates the data of but not the prior"
                        .into(),
                );
            }
            if self.session_shrink.is_some() {
                return Err(
                    "ewridge: window and session_shrink do not combine; the slow twin is a \
                     second accumulator under a longer halflife, and truncating one and not \
                     the other would blend two different histories"
                        .into(),
                );
            }
        } else if self.window_every.is_some() {
            return Err("ewridge: window_every needs `window`".into());
        }
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
                     hold more than one. Set solve_every > 0 (the default is halflife / 50, \
                     or 0 for `lam` and an infinite halflife) and max_rows_between_solves > 1",
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
        if self.ridge_decay && (self.standardize || self.n_combos() > 1) {
            return Err("ridge_decay is incompatible with standardize and grids".into());
        }
        match (self.session_shrink, self.long_halflife) {
            (Some(f), _) if !(0.0..=1.0).contains(&f) => {
                return Err("session_shrink must be in [0, 1]".into());
            }
            (Some(_), None) => {
                return Err("session_shrink needs long_halflife (the slow twin's decay)".into());
            }
            (None, Some(_)) => {
                return Err("long_halflife has no effect without session_shrink".into());
            }
            (Some(_), Some(h)) if h <= 0.0 || h.is_nan() => {
                return Err("long_halflife must be > 0".into());
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
    /// `long_halflife`, representing the long-run relationship.
    #[serde(default)]
    slow: Option<Box<Acc>>,
    clock_since_solve: f64,
    rows_since_solve: u32,
    pub solve_failures: u64,
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
/// the solve reads; the residual variance is what `sigma` and `resid_z` read,
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
        let acc = move || Acc::new(m, k_total, block);
        let slow = cfg.session_shrink.map(|_| Box::new(acc()));
        let win = match cfg.window {
            Some(w) => Some(Windowed {
                clock: 0.0,
                snaps: crate::Snapshots::new(w, cfg.window_every.unwrap_or(1))?,
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
            solve_failures: 0,
            win,
            zbuf: vec![0.0; k_total],
            cfg,
        })
    }

    pub fn cfg(&self) -> &EwRidgeCfg {
        &self.cfg
    }

    /// Per-target EW residual variance of the first slot's prediction; under
    /// a `window`, the variance *inside* it. The bank's `sigma` and `resid_z`
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
    pub fn gram_parts(&self) -> (Vec<GramPart>, Option<&TargetMoments>) {
        let (of, gaps) = (&self.acc.grams.of, self.cfg.target_gaps);
        match self.view() {
            Some(v) => (
                gram_parts(&v.acc.grams, of, &v.acc.cross, &v.acc.wj, gaps),
                None,
            ),
            None => (
                gram_parts(
                    &self.acc.grams.grams,
                    of,
                    &self.acc.cross,
                    &self.acc.wj,
                    gaps,
                ),
                Some(&self.acc.tm),
            ),
        }
    }

    /// The accumulated weight over every row, whatever the targets (hard rule
    /// 8): under a `window`, the weight *inside* it, which stops growing once
    /// the window fills. That is what `min_periods` then gates on.
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
        if self.cfg.add_intercept {
            self.zbuf[0] = 1.0;
            self.zbuf[1..].copy_from_slice(x);
        } else {
            self.zbuf.copy_from_slice(x);
        }
    }

    /// Indices into z for a combo's feature set (intercept first if configured).
    fn combo_z_indices(&self, fs_idx: usize) -> Vec<usize> {
        let off = usize::from(self.cfg.add_intercept);
        let mut idx: Vec<usize> = if self.cfg.feature_sets.is_empty() {
            (0..self.cfg.n_features).map(|i| i + off).collect()
        } else {
            self.cfg.feature_sets[fs_idx]
                .1
                .iter()
                .map(|&i| i + off)
                .collect()
        };
        if self.cfg.add_intercept {
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
        // With a `window`, the fit is solved from the truncated accumulators:
        // no row older than the window is in a Gram at all. Without one, and
        // before anything has aged out, these borrow the live state.
        let view = self.view();
        if view.as_ref().is_some_and(|v| v.acc.cross.w <= 0.0) {
            // An empty window has nothing to solve and no fit to report (C2).
            self.beta = Some(vec![vec![f64::NAN; k_total]; m * nc]);
            self.clock_since_solve = 0.0;
            self.rows_since_solve = 0;
            return;
        }
        let mut failures = 0u64;
        let (grams, cross) = match view.as_ref() {
            Some(v) => (v.acc.grams.as_slice(), &v.acc.cross),
            None => (self.acc.grams.grams.as_slice(), &self.acc.cross),
        };
        // With an intercept the slopes are solved on the centred system (see
        // the module docs). The rest read the raw normal equations and need
        // the uncentred cross-moments: `ridge_decay`, whose intercept is
        // penalized and so part of the system rather than eliminated from it,
        // and the two solves through the origin, which centre nothing.
        let centred = self.cfg.add_intercept && !self.cfg.ridge_decay;

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
                // zero, and is solved.
                if ridge == 0.0 && cov.n_eff() <= 0.0 {
                    continue;
                }
                let zidx = self.combo_z_indices(fs_idx);
                let kc = zidx.len();

                let solved: Option<Vec<f64>> = if centred {
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
                    if self.cfg.ridge_decay {
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
                        Self::run_solve(&mut failures, &a, &b, kc, mr)
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
                    } else {
                        Self::solve_scaled_through_origin(
                            cov,
                            &mut failures,
                            &zidx,
                            &b,
                            kc,
                            mr,
                            ridge,
                        )
                    }
                };

                if let Some(sol) = solved {
                    for (jj, &j) in readers.iter().enumerate() {
                        for (ai, &zi) in zidx.iter().enumerate() {
                            beta[j * nc + ci][zi] = sol[jj * kc + ai];
                        }
                    }
                } else if let Some(prev) = &self.beta {
                    // Total failure even with jitter: keep the previous
                    // coefficients.
                    for &j in &readers {
                        beta[j * nc + ci] = prev[j * nc + ci].clone();
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
        self.solve_failures += failures;
        self.beta = Some(beta);
        self.clock_since_solve = 0.0;
        self.rows_since_solve = 0;
    }

    fn run_solve(failures: &mut u64, a: &[f64], b: &[f64], k: usize, m: usize) -> Option<Vec<f64>> {
        match solve_spd(a, b, k, m) {
            Some((x, jit)) => {
                *failures += u64::from(jit);
                Some(x)
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
    /// is the per-target uncentred right-hand side, and `zidx` maps the
    /// combo's slots to accumulator indices.
    fn solve_scaled_through_origin(
        cov: &EwCov,
        failures: &mut u64,
        zidx: &[usize],
        b: &[f64],
        kc: usize,
        m: usize,
        ridge: f64,
    ) -> Option<Vec<f64>> {
        let s: Vec<f64> = (0..kc)
            .map(|i| cov.raw(zidx[i], zidx[i]).max(0.0).sqrt())
            .collect();
        // No centering here, so no cancellation: any strictly positive raw
        // moment is usable.
        let keep: Vec<usize> = (0..kc).filter(|&i| s[i] > 0.0).collect();
        let kk = keep.len();
        if kk == 0 {
            return Some(vec![0.0; kc * m]);
        }
        let mut asub = vec![0.0; kk * kk];
        for (i2, &i) in keep.iter().enumerate() {
            for (j2, &j) in keep.iter().enumerate() {
                asub[i2 * kk + j2] = cov.raw(zidx[i], zidx[j]) / (s[i] * s[j]);
            }
            asub[i2 * kk + i2] += ridge;
        }
        let mut bsub = vec![0.0; kk * m];
        for j in 0..m {
            for (i2, &i) in keep.iter().enumerate() {
                bsub[j * kk + i2] = b[j * kc + i] / s[i];
            }
        }
        let sol = Self::run_solve(failures, &asub, &bsub, kk, m)?;
        let mut out = vec![0.0; kc * m];
        for j in 0..m {
            for (i2, &i) in keep.iter().enumerate() {
                out[j * kc + i] = sol[j * kk + i2] / s[i];
            }
        }
        Some(out)
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
    ) -> Option<Vec<f64>> {
        // Feature slots are 1..kc; slot 0 is the intercept.
        let kc = zidx.len();
        let kf = kc - 1;
        let mr = readers.len();
        let mut c = vec![0.0; kf * kf];
        for i in 0..kf {
            for j in 0..kf {
                c[i * kf + j] = cov.cov(zidx[i + 1], zidx[j + 1]);
            }
        }
        let (s, keep): (Vec<f64>, Vec<usize>) = if self.cfg.standardize {
            let s = (0..kf).map(|i| c[i * kf + i].max(0.0).sqrt()).collect();
            // A genuinely constant feature is dropped (coefficient 0) rather
            // than blowing up; with centered accumulators its variance is
            // exactly zero.
            let keep = (0..kf)
                .filter(|&i| {
                    crate::variance_is_usable(c[i * kf + i], cov.raw(zidx[i + 1], zidx[i + 1]))
                })
                .collect();
            (s, keep)
        } else {
            (vec![1.0; kf], (0..kf).collect())
        };
        let kk = keep.len();
        let mut out = vec![0.0; kc * mr];
        if kk > 0 {
            let mut asub = vec![0.0; kk * kk];
            for (i2, &i) in keep.iter().enumerate() {
                for (j2, &j) in keep.iter().enumerate() {
                    asub[i2 * kk + j2] = c[i * kf + j] / (s[i] * s[j]);
                }
                asub[i2 * kk + i2] += ridge;
            }
            let mut bsub = vec![0.0; kk * mr];
            for (jj, &j) in readers.iter().enumerate() {
                for (i2, &i) in keep.iter().enumerate() {
                    bsub[jj * kk + i2] = cross.c[j][zidx[i + 1]] / s[i];
                    // Warm prior: shrink toward coef_prior rather than toward
                    // zero. It lives in original units, and on the
                    // standardized scale a coefficient is beta * sd.
                    if let Some(c0) = &self.cfg.coef_prior {
                        bsub[jj * kk + i2] += ridge * c0[j][zidx[i + 1]] * s[i];
                    }
                }
            }
            let sol = Self::run_solve(failures, &asub, &bsub, kk, mr)?;
            for jj in 0..mr {
                for (i2, &i) in keep.iter().enumerate() {
                    out[jj * kc + i + 1] = sol[jj * kk + i2] / s[i];
                }
            }
        }
        // `m_j` is the Gram's mean, which under `own_rows` is over exactly the
        // target's rows, plus the target's offset under `pairwise`, where the
        // Gram is over every row (`crate::gaps::Cross`). Not the offset form
        // under `own_rows` too, so that a row the target misses leaves its
        // fit where it was to the bit: there the offset takes back the step
        // the all-row mean took, which is exact only in exact arithmetic.
        let pairwise = self.cfg.target_gaps == TargetGaps::Pairwise;
        for (jj, &j) in readers.iter().enumerate() {
            let mut b0 = cross.my[j];
            for i in 0..kf {
                let z = zidx[i + 1];
                let offset = if pairwise { cross.d[j][z] } else { 0.0 };
                b0 -= (cov.mean(z) + offset) * out[jj * kc + i + 1];
            }
            out[jj * kc] = b0;
        }
        Some(out)
    }
}

impl OnlineModel for EwRidge {
    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.set_budget(budget);
        }
    }

    fn window_over_budget(&self) -> Option<(usize, usize)> {
        self.win.as_ref().and_then(|win| win.snaps.over_budget())
    }
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        match self.window_weights() {
            Some((_, wj)) => out.extend_from_slice(&wj),
            None => out.extend_from_slice(&self.acc.wj),
        }
        true
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
        // The slow twin sees the same rows under its own, longer halflife.
        if let (Some(slow), Some(h)) = (self.slow.as_mut(), self.cfg.long_halflife) {
            let slow_lam = Decay::Halflife(h).factor(d_clock);
            slow.learn(&self.zbuf, y, slow_lam, weight, gaps);
        }
        // The snapshot is every accumulator as it stands *before* this row,
        // decayed to this row's clock, so subtracting it later retains this
        // row and everything after. Keyed by a clock the model accumulates
        // itself, so the boundary cannot depend on the chunking.
        if let Some(win) = self.win.as_mut() {
            let t = win.clock + d_clock;
            let snap = RidgeMoments {
                acc: self.acc.snapshot(lam),
                wsig: self.wsig.iter().map(|w| w * lam).collect(),
                sig2: self.sig2.clone(),
            };
            win.snaps.offer(t, || snap);
            win.clock = t;
            win.snaps.trim(t);
        }
        // EW residual variance from the primary (first-combo) pred. Its
        // weight ages on every row, and a row with a target, a weight and a
        // prediction adds its squared residual. A row with a target and no
        // prediction -- `min_periods` unmet after a clock gap, say -- ages it
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
        let due = self.cfg.solve_every <= 0.0
            || self.clock_since_solve >= self.cfg.solve_every
            || self.rows_since_solve >= self.cfg.max_rows_between_solves
            || (self.beta.is_none() && self.n_eff() >= self.cfg.min_periods);
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
        if let (true, Some(beta)) = (n_eff >= self.cfg.min_periods, &self.beta) {
            for j in 0..m {
                if wj[j] > 0.0 {
                    for c in 0..nc {
                        pred[j * nc + c] = dot_aug(&beta[j * nc + c], x, self.cfg.add_intercept);
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
                let twin = m.slow.as_ref().is_none_or(|s| s.has_shape(n, k));
                if !m.acc.has_shape(n, k) || !twin || m.wsig.len() != n || m.sig2.len() != n {
                    return Err(StateError::Invalid(
                        "ew_ridge: the accumulators have the wrong shape".into(),
                    ));
                }
                m.zbuf = vec![0.0; k];
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
mod tests {
    use super::*;

    fn cfg(k: usize, m: usize) -> EwRidgeCfg {
        EwRidgeCfg {
            n_features: k,
            n_targets: m,
            add_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: vec![1e-8],
            feature_sets: vec![],
            standardize: false,
            ridge_decay: false,
            coef_prior: None,
            session_shrink: None,
            long_halflife: None,
            min_periods: (k + 1) as f64,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            gram_block_rows: 0,
            target_gaps: TargetGaps::OwnRows,
            window: None,
            window_every: None,
        }
    }

    /// With `ridge_decay` the prior sits on the sum scale, so it starts the
    /// fit and then fades: the usual warm start.
    #[test]
    fn coef0_with_ridge_decay_warms_the_start_then_fades() {
        let mut c = cfg(2, 1);
        c.ridge = vec![10.0];
        c.ridge_decay = true;
        c.standardize = false;
        c.coef_prior = Some(vec![vec![0.0, 5.0, -5.0]]);
        c.min_periods = 0.0;
        let mut m = EwRidge::new(c).unwrap();

        // With no target seen yet the fit is essentially the prior. (Not
        // exactly: the feature row itself has already entered S, which pulls a
        // little even with no target.)
        let mut s = 71u64;
        let x0 = [lcg(&mut s), lcg(&mut s)];
        m.step(&x0, &[None], 0.0, 1.0);
        let early = m.coefficients().unwrap()[0].clone();
        assert!(
            (early[1] - 5.0).abs() < 0.5 && (early[2] + 5.0).abs() < 0.5,
            "cold start should sit at the prior: {early:?}"
        );

        // With enough contradicting evidence it moves to the truth.
        for i in 0..20000 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.5 * x[0] - 0.5 * x[1];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let late = m.coefficients().unwrap()[0].clone();
        assert!(
            (late[1] - 1.5).abs() < 0.05,
            "prior did not wash out: {late:?}"
        );
        assert!((late[2] + 0.5).abs() < 0.05);
    }

    /// Without `ridge_decay` the pull is permanent, because `S` is a weighted
    /// mean and never outgrows a fixed `ridge`. Worth pinning: it is the
    /// opposite of the usual "the prior washes out" intuition.
    #[test]
    fn coef0_without_ridge_decay_pulls_forever() {
        let mut c = cfg(1, 1);
        c.ridge = vec![10.0];
        c.coef_prior = Some(vec![vec![0.0, 5.0]]);
        c.min_periods = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 72u64;
        for i in 0..50000 {
            let x = [lcg(&mut s)];
            m.step(&x, &[Some(1.5 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = m.coefficients().unwrap()[0][1];
        assert!(
            b > 3.0,
            "a fixed ridge should keep pulling toward the prior forever, got {b}"
        );
    }

    #[test]
    fn coef0_shrinks_toward_the_prior_not_zero() {
        // Same ridge, same data, different priors => the fits differ, and each
        // sits between the data's answer and its own prior.
        let run = |prior: Option<Vec<Vec<f64>>>| {
            let mut c = cfg(1, 1);
            c.ridge = vec![50.0];
            c.coef_prior = prior;
            c.min_periods = 0.0;
            let mut m = EwRidge::new(c).unwrap();
            let mut s = 73u64;
            for i in 0..200 {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(2.0 * x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.coefficients().unwrap()[0][1]
        };
        let toward_zero = run(None);
        let toward_ten = run(Some(vec![vec![0.0, 10.0]]));
        assert!(
            toward_zero < 2.0,
            "no prior should shrink toward 0: {toward_zero}"
        );
        assert!(
            toward_ten > 2.0,
            "a prior of 10 should pull up: {toward_ten}"
        );
    }

    #[test]
    fn coef0_works_with_standardization() {
        // The prior is stated in original units; on badly scaled features the
        // standardized path must still honour it.
        let mut c = cfg(1, 1);
        c.ridge = vec![1e6];
        c.standardize = true;
        c.coef_prior = Some(vec![vec![0.0, 0.02]]);
        c.min_periods = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 79u64;
        for i in 0..500 {
            let x = [100.0 * lcg(&mut s)];
            let y = 0.05 * x[0];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        // An overwhelming ridge pins the fit at the prior, in original units.
        let b = m.coefficients().unwrap()[0][1];
        assert!(
            (b - 0.02).abs() < 5e-3,
            "expected ~0.02 in original units, got {b}"
        );
    }

    /// A session break should be able to revert partway toward the long run,
    /// rather than only choosing between "carry on" and "start over".
    #[test]
    fn session_shrink_reverts_toward_the_long_run() {
        // Long run: slope 1. Today: slope -1 for a while. After a session
        // break with shrink f, the fit should sit between the two.
        let build = |f: Option<f64>| {
            let mut c = cfg(1, 1);
            c.decay = Decay::Halflife(50.0);
            c.session_shrink = f;
            c.long_halflife = f.map(|_| 100_000.0);
            c.min_periods = 0.0;
            EwRidge::new(c).unwrap()
        };
        let run = |m: &mut EwRidge| {
            let mut s = 91u64;
            // a long history at slope +1
            for i in 0..4000 {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            // then a shorter stretch at slope -1
            for _ in 0..300 {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(-x[0])], 1.0, 1.0);
            }
            m.coefficients().unwrap()[0][1]
        };

        let mut plain = build(None);
        let before_plain = run(&mut plain);
        let mut shrunk = build(Some(0.9));
        let before_shrunk = run(&mut shrunk);
        // Both have been dragged to roughly -1 by the recent regime.
        assert!(before_plain < -0.5 && before_shrunk < -0.5);

        // The session boundary reverts the shrinking one toward +1.
        shrunk.blend_toward_long_run();
        plain.blend_toward_long_run(); // no twin configured: a no-op
        let after_shrunk = shrunk.coefficients_after_blend();
        let after_plain = plain.coefficients_after_blend();

        assert!(
            (after_plain - before_plain).abs() < 1e-9,
            "no twin configured should mean no change: {before_plain} -> {after_plain}"
        );
        assert!(
            after_shrunk > before_shrunk + 0.5,
            "shrink should pull back toward the long run: {before_shrunk} -> {after_shrunk}"
        );
    }

    /// E45: the target moments must survive a blend as a mixture, not go
    /// stale, and `f = 0` (or a twin identical to the model) must leave them
    /// exactly where they were -- the invariant the means, co-moments and
    /// weights already hold.
    #[test]
    fn a_blend_mixes_the_target_moments() {
        let build = |f: f64| {
            let mut c = cfg(1, 1);
            c.decay = Decay::Halflife(50.0);
            c.session_shrink = Some(f);
            c.long_halflife = Some(50.0); // the same halflife: an identical twin
            c.min_periods = 0.0;
            EwRidge::new(c).unwrap()
        };
        let run = |m: &mut EwRidge| {
            let mut s = 7u64;
            for i in 0..500 {
                let x = [lcg(&mut s)];
                m.step(
                    &x,
                    &[Some(3.0 + 2.0 * x[0])],
                    if i == 0 { 0.0 } else { 1.0 },
                    1.0,
                );
            }
        };
        let mut twin = build(0.5);
        run(&mut twin);
        let (m0, v0, q0) = {
            let tm = twin.target_moments().unwrap();
            (tm.means()[0], tm.vars()[0], tm.q()[0])
        };
        twin.blend_toward_long_run();
        let tm = twin.target_moments().unwrap();
        assert!(
            (tm.means()[0] - m0).abs() < 1e-12,
            "{} vs {m0}",
            tm.means()[0]
        );
        assert!((tm.vars()[0] - v0).abs() < 1e-9, "{} vs {v0}", tm.vars()[0]);
        assert!((tm.q()[0] - q0).abs() < 1e-9, "{} vs {q0}", tm.q()[0]);

        // A genuinely slower twin moves them toward the long run. The level
        // sits at 0 for a long stretch and jumps to 10 for a short one, so
        // the fast window's mean is ~10 and the twin's is much lower.
        let mut slow = {
            let mut c = cfg(1, 1);
            c.decay = Decay::Halflife(20.0);
            c.session_shrink = Some(0.9);
            c.long_halflife = Some(5000.0);
            c.min_periods = 0.0;
            EwRidge::new(c).unwrap()
        };
        let mut s = 11u64;
        for i in 0..2000 {
            let x = [lcg(&mut s)];
            let level = if i < 1900 { 0.0 } else { 10.0 };
            slow.step(
                &x,
                &[Some(level + x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let before = slow.target_moments().unwrap().means()[0];
        assert!(
            before > 9.0,
            "the fast window should be at the new level: {before}"
        );
        slow.blend_toward_long_run();
        let tm = slow.target_moments().unwrap();
        assert!(
            tm.means()[0] < before - 5.0,
            "the blend should pull the mean toward the long run: {before} -> {}",
            tm.means()[0]
        );
        // And the mixture is still a valid set of moments.
        assert!(tm.vars()[0] > 0.0);
        assert!(tm.q()[0] > 0.0 && tm.n_kish(slow.target_weights())[0].unwrap() > 1.0);
    }

    #[test]
    fn session_shrink_config_is_validated() {
        let mut c = cfg(1, 1);
        c.session_shrink = Some(0.5);
        assert!(EwRidge::new(c).is_err(), "shrink without long_halflife");
        let mut c = cfg(1, 1);
        c.long_halflife = Some(1000.0);
        assert!(EwRidge::new(c).is_err(), "long_halflife without shrink");
        let mut c = cfg(1, 1);
        c.session_shrink = Some(1.5);
        c.long_halflife = Some(1000.0);
        assert!(EwRidge::new(c).is_err(), "shrink out of range");
    }

    #[test]
    fn coef0_shape_is_validated() {
        let mut c = cfg(2, 1);
        c.coef_prior = Some(vec![vec![0.0, 1.0]]); // too short
        assert!(EwRidge::new(c).is_err());
        let mut c = cfg(2, 1);
        c.coef_prior = Some(vec![vec![0.0, 1.0, f64::NAN]]);
        assert!(EwRidge::new(c).is_err());
        // coef_prior *is* allowed with ridge_decay -- that combination is the
        // fading warm start -- so only the shape rules above are enforced.
        let mut c = cfg(2, 1);
        c.ridge_decay = true;
        c.standardize = false;
        c.coef_prior = Some(vec![vec![0.0, 1.0, 2.0]]);
        assert!(EwRidge::new(c).is_ok());
    }

    /// Deterministic pseudo-random stream (no external rng dependency).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn recovers_static_beta() {
        let beta = [0.5, -1.0, 2.0];
        let mut m = EwRidge::new(cfg(3, 1)).unwrap();
        let mut s = 42u64;
        let mut last = Step {
            pred: vec![],
            n_eff: 0.0,
            extra: None,
        };
        for i in 0..500 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>() + 3.0;
            let d = if i == 0 { 0.0 } else { 1.0 };
            last = m.step(&x, &[Some(y)], d, 1.0);
        }
        let c = &m.coefficients().unwrap()[0];
        assert!((c[0] - 3.0).abs() < 1e-6, "intercept {}", c[0]);
        for i in 0..3 {
            assert!((c[i + 1] - beta[i]).abs() < 1e-6, "beta[{i}] {}", c[i + 1]);
        }
        assert!(last.pred[0].is_finite());
    }

    #[test]
    fn pred_is_out_of_sample_and_warmup_nan() {
        let mut m = EwRidge::new(cfg(2, 1)).unwrap();
        let mut s = 7u64;
        for i in 0..3 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let st = m.step(
                &x,
                &[Some(lcg(&mut s))],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
            assert!(st.pred[0].is_nan(), "warmup row {i} must be NaN");
        }
        let st = m.step(&[0.1, 0.2], &[Some(0.3)], 1.0, 1.0);
        assert!(st.pred[0].is_finite());
    }

    #[test]
    fn solve_schedule_staleness() {
        // With a large solve_every, coefficients stay fixed between solves.
        let mut c = cfg(1, 1);
        c.solve_every = 1e9;
        c.max_rows_between_solves = 10;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 9u64;
        let mut snapshots = vec![];
        for i in 0..40 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.01 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            snapshots.push(m.coefficients().map(|b| b[0].clone()));
        }
        // Between solve rows the coefficients are bit-identical.
        let mut changes = 0;
        for w in snapshots.windows(2) {
            if w[0] != w[1] {
                changes += 1;
            }
        }
        assert!(
            changes <= 5,
            "expected sparse solves, got {changes} changes"
        );
    }

    #[test]
    fn multi_target_and_grids() {
        let mut c = cfg(3, 2);
        c.ridge = vec![1e-8, 10.0];
        c.feature_sets = vec![("a".into(), vec![0, 1]), ("b".into(), vec![2])];
        let m = EwRidge::new(c.clone()).unwrap();
        assert_eq!(m.n_outputs(), 2 * 4);

        let mut m = EwRidge::new(c).unwrap();
        let mut s = 3u64;
        let mut last = None;
        for i in 0..200 {
            let x = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
            let y0 = x[0] - x[1];
            let y1 = 3.0 * x[2];
            last = Some(m.step(
                &x,
                &[Some(y0), Some(y1)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            ));
        }
        let st = last.unwrap();
        assert_eq!(st.pred.len(), 8);
        assert!(st.pred.iter().all(|p| p.is_finite()));
        // combo "b" (only x2) predicts y1 well, y0 badly; heavy ridge shrinks.
        let b = m.coefficients().unwrap();
        // target 1 (y1), combo index 2 = fs "b", small ridge: coef on x2 ~ 3
        assert!((b[4 + 2][3] - 3.0).abs() < 0.05, "{:?}", b[6]);
        // heavy-ridge combo shrinks toward zero
        assert!(b[4 + 3][3].abs() < b[4 + 2][3].abs());
        // feature set "b" never touches x0/x1
        assert_eq!(b[4 + 2][1], 0.0);
        assert_eq!(b[4 + 2][2], 0.0);
    }

    #[test]
    fn standardize_matches_plain_when_ridge_tiny() {
        // On well-conditioned data the centered/standardized solve and the raw
        // solve are algebraically identical (ridge ~ 0), so they must agree
        // tightly. (On badly scaled data they diverge because the raw normal
        // equations are ill-conditioned -- which is why standardize exists.)
        // Ridge is dropped to 1e-12 and the feature scales are put ~1e3 apart,
        // so the centering and scaling matrices are far from the identity and
        // an operation applied in the wrong direction cannot hide inside the
        // tolerance. The invariance is exact only at ridge = 0, because the
        // penalty lands on the raw scale in one path and the standardized
        // scale in the other.
        let mut ca = cfg(2, 1);
        ca.ridge = vec![1e-12];
        let mut cb = ca.clone();
        cb.standardize = true;
        let mut ma = EwRidge::new(ca).unwrap();
        let mut mb = EwRidge::new(cb).unwrap();
        let mut s = 11u64;
        for i in 0..300 {
            let x = [400.0 + 3.0 * lcg(&mut s), 0.002 * lcg(&mut s)];
            let y = 7.0 + 0.5 * x[0] - 300.0 * x[1] + 0.001 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            ma.step(&x, &[Some(y)], d, 1.0);
            mb.step(&x, &[Some(y)], d, 1.0);
        }
        let a = &ma.coefficients().unwrap()[0];
        let b = &mb.coefficients().unwrap()[0];
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < 1e-6 * (1.0 + a[i].abs()),
                "coef {i}: plain {} vs standardized {}",
                a[i],
                b[i]
            );
        }
        // And the standardized path recovered the generating relationship, so
        // the agreement is not two paths failing the same way. The intercept is
        // reconstructed from the centered fit, which is the step most easily
        // lost.
        assert!((b[0] - 7.0).abs() < 0.1, "intercept {}", b[0]);
        assert!((b[1] - 0.5).abs() < 1e-3, "slope 0 {}", b[1]);
        assert!((b[2] + 300.0).abs() < 5.0, "slope 1 {}", b[2]);
    }

    /// A model with a slow twin, fed `n` rows of a deterministic stream.
    fn blended_pair(shrink: f64) -> EwRidge {
        let mut c = cfg(2, 1);
        c.session_shrink = Some(shrink);
        c.long_halflife = Some(400.0);
        c.decay = Decay::Halflife(20.0);
        c.min_periods = 3.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 23u64;
        for i in 0..200 {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            // A relationship that flips halfway, so fast and slow genuinely
            // disagree by the time the session boundary arrives.
            let sign = if i < 100 { 1.0 } else { -1.0 };
            let y = sign * (2.0 * x[0] - x[1]);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        m
    }

    #[test]
    fn coef0_solves_the_ridge_problem_it_claims_to() {
        // With `coef_prior = c`, the penalty shrinks toward `c` rather than zero:
        //     beta = (C + rI)^-1 (d + r c)
        // on the centered accumulators, with the intercept recovered after.
        // The existing coef_prior tests check the *direction* of the pull; this
        // pins the closed form, computed by hand from the model's own state.
        let (r, c0) = (0.7, vec![vec![0.0, 3.0, -2.0]]);
        let mut cfg_ = cfg(2, 1);
        cfg_.ridge = vec![r];
        cfg_.coef_prior = Some(c0.clone());
        cfg_.min_periods = 3.0;
        let mut m = EwRidge::new(cfg_).unwrap();
        let mut s = 127u64;
        for i in 0..200 {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            let y = 1.0 + 2.0 * x[0] - x[1] + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }

        // The 2x2 penalized normal equations on the centered moments.
        let (c00, c01, c11) = (
            m.gram(0).cov(1, 1),
            m.gram(0).cov(1, 2),
            m.gram(0).cov(2, 2),
        );
        // d_i = E[x_i y] - E[x_i] E[y], from the tracked cross-moment means.
        let raw = m.cross_moments();
        let d0 = raw[0][1] - m.gram(0).mean(1) * raw[0][0];
        let d1 = raw[0][2] - m.gram(0).mean(2) * raw[0][0];
        let (a00, a11) = (c00 + r, c11 + r);
        let (rhs0, rhs1) = (d0 + r * c0[0][1], d1 + r * c0[0][2]);
        let det = a00 * a11 - c01 * c01;
        let want = [
            (rhs0 * a11 - c01 * rhs1) / det,
            (a00 * rhs1 - c01 * rhs0) / det,
        ];

        let got = &m.coefficients().unwrap()[0];
        for i in 0..2 {
            assert!(
                (got[i + 1] - want[i]).abs() < 1e-9 * (1.0 + want[i].abs()),
                "slope {i}: {} vs {}",
                got[i + 1],
                want[i]
            );
        }
        // The intercept is reconstructed, not fitted: mean(y) - b'mean(x).
        let want0 = raw[0][0] - got[1] * m.gram(0).mean(1) - got[2] * m.gram(0).mean(2);
        assert!((got[0] - want0).abs() < 1e-9, "{} vs {want0}", got[0]);

        // An overwhelming penalty must land on coef_prior exactly.
        let mut cfg_ = cfg(2, 1);
        cfg_.ridge = vec![1e12];
        cfg_.coef_prior = Some(c0.clone());
        cfg_.min_periods = 3.0;
        let mut m = EwRidge::new(cfg_).unwrap();
        let mut s = 131u64;
        for i in 0..100 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let got = &m.coefficients().unwrap()[0];
        assert!(
            (got[1] - 3.0).abs() < 1e-3,
            "slope 0 -> coef_prior: {}",
            got[1]
        );
        assert!(
            (got[2] + 2.0).abs() < 1e-3,
            "slope 1 -> coef_prior: {}",
            got[2]
        );
    }

    #[test]
    fn a_singular_solve_is_counted_and_the_previous_fit_is_kept() {
        // `run_solve` records both outcomes: a solve rescued by jitter and a
        // total failure. Two perfectly collinear features with no ridge give
        // a rank-deficient system; the model must never emit NaN and must say
        // that something went wrong.
        let mut c = cfg(2, 1);
        c.ridge = vec![0.0];
        c.min_periods = 2.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 137u64;
        for i in 0..40 {
            let a = lcg(&mut s);
            m.step(
                &[a, a],
                &[Some(2.0 * a + 1.0)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        assert!(m.solve_failures > 0, "a singular solve must be recorded");
        let beta = &m.coefficients().unwrap()[0];
        assert!(beta.iter().all(|v| v.is_finite()), "never NaN: {beta:?}");

        // A well-conditioned stream records nothing.
        let mut c = cfg(2, 1);
        c.min_periods = 2.0;
        let mut ok = EwRidge::new(c).unwrap();
        let mut s = 139u64;
        for i in 0..40 {
            let x = [lcg(&mut s), lcg(&mut s)];
            ok.step(
                &x,
                &[Some(x[0] - x[1])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        assert_eq!(ok.solve_failures, 0, "a healthy fit must record no failure");
    }

    #[test]
    fn the_slow_twin_is_the_same_model_at_the_long_halflife() {
        // The twin's accumulators are updated by a second copy of the update
        // block inside `step`, which nothing else reaches. The oracle is the
        // obvious one: a standalone model configured at `long_halflife` and
        // fed the same rows must end up with identical statistics.
        let mut c = cfg(2, 1);
        c.session_shrink = Some(0.4);
        c.long_halflife = Some(300.0);
        c.decay = Decay::Halflife(15.0);
        c.min_periods = 3.0;

        let mut twin_cfg = cfg(2, 1);
        twin_cfg.decay = Decay::Halflife(300.0);
        twin_cfg.min_periods = 3.0;

        let mut m = EwRidge::new(c).unwrap();
        let mut reference = EwRidge::new(twin_cfg).unwrap();
        let mut s = 43u64;
        for i in 0..150 {
            let x = [lcg(&mut s), 2.0 + lcg(&mut s)];
            // Every fourth row has a null target and an irregular gap, so the
            // twin's null branch (`*wj *= slow_lam`) is exercised too.
            let y = if i % 4 == 3 {
                None
            } else {
                Some(1.5 * x[0] - 0.5 * x[1])
            };
            let d = if i == 0 { 0.0 } else { 1.0 + (i % 3) as f64 };
            m.step(&x, &[y], d, 1.0);
            reference.step(&x, &[y], d, 1.0);
        }

        let slow = m.slow.as_ref().unwrap();
        let k = m.cfg.k_total();
        assert!((slow.grams.grams[0].n_eff() - reference.gram(0).n_eff()).abs() < 1e-9);
        for i in 0..k {
            assert!(
                (slow.grams.grams[0].mean(i) - reference.gram(0).mean(i)).abs() < 1e-9,
                "mean {i}"
            );
            for j in 0..k {
                assert!((slow.grams.grams[0].cov(i, j) - reference.gram(0).cov(i, j)).abs() < 1e-9);
            }
        }
        assert!((slow.wj[0] - reference.acc.wj[0]).abs() < 1e-9);
        let (got, want) = (slow.cross.raw(0), reference.acc.cross.raw(0));
        for i in 0..k {
            assert!((got[i] - want[i]).abs() < 1e-9, "r[{i}]");
        }
        // And it is genuinely slower than the fast side, or the test would
        // pass with the twin wired to the wrong decay.
        assert!(
            slow.grams.grams[0].n_eff() > 3.0 * m.gram(0).n_eff(),
            "{} vs {}",
            slow.grams.grams[0].n_eff(),
            m.gram(0).n_eff()
        );
    }

    #[test]
    fn residual_variance_is_the_ew_mean_of_squared_out_of_sample_errors() {
        // `sigma2` is only surfaced through the Polars layer, so nothing in
        // this crate pinned its recursion. It is an EW mean on the model's own
        // clock, over the *predicted* residual -- and rows before the first
        // prediction contribute nothing.
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(25.0);
        c.min_periods = 3.0;
        let mut m = EwRidge::new(c).unwrap();

        let (mut want, mut wsig) = (0.0, 0.0);
        let mut s = 47u64;
        for i in 0..120 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.3 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let lam = 0.5f64.powf(d / 25.0);
            let p = m.step(&x, &[Some(y)], d, 1.0).pred[0];
            if p.is_finite() {
                let resid = y - p;
                let ws_new = lam * wsig + 1.0;
                want = (lam * wsig * want + resid * resid) / ws_new;
                wsig = ws_new;
            }
            assert!(
                (m.sigma2()[0] - want).abs() < 1e-12,
                "row {i}: {} vs {want}",
                m.sigma2()[0]
            );
        }
        // The weight saturates at 1/(1 - lam) ~ 36.6 for this halflife.
        assert!(
            wsig > 30.0,
            "the recursion should have run, not been skipped"
        );
        assert!(
            want > 0.0 && want < 1.0,
            "plausible residual variance: {want}"
        );
    }

    #[test]
    fn null_targets_decay_the_residual_variance_weight_without_adding_to_it() {
        // The `None` arm decays `wj` and `wsig` but must not fold a residual
        // in -- otherwise a gap in the target inflates or freezes sigma.
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(10.0);
        c.min_periods = 2.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 53u64;
        for i in 0..60 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(2.0 * x[0] + 0.1)],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (sig, w, wj) = (m.sigma2()[0], m.wsig[0], m.acc.wj[0]);
        assert!(sig > 0.0 && w > 0.0);

        let lam = 0.5f64.powf(3.0 / 10.0);
        m.step(&[0.5], &[None], 3.0, 1.0);
        assert_eq!(m.sigma2()[0], sig, "a null target must not move sigma2");
        assert!((m.wsig[0] - w * lam).abs() < 1e-12, "but its weight decays");
        assert!((m.acc.wj[0] - wj * lam).abs() < 1e-12);
    }

    /// `σ²` is the EW mean of the squared out-of-sample errors: every row
    /// ages its weight by the row's `lam`, and a row with a target, a
    /// weight and a prediction adds `w·r²`. A row with a target and no
    /// prediction -- here the rows after a clock gap has taken `n_eff` under
    /// `min_periods` -- rightly added nothing, and aged nothing either, so
    /// `σ²` forgot less across them than the clock says (N6, found beside
    /// review 2026-09-12 S13).
    #[test]
    fn the_residual_variance_ages_on_every_row() {
        let hl = 10.0;
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(hl);
        c.min_periods = 3.0;
        let mut m = EwRidge::new(c).unwrap();
        let (mut want, mut wsig, mut unpredicted) = (0.0f64, 0.0f64, 0);
        let mut s = 61u64;
        for i in 0..160 {
            let x = [lcg(&mut s)];
            let y = 2.0 * x[0] + 0.1 + 0.2 * lcg(&mut s);
            let d = match i {
                0 => 0.0,
                80 => 200.0,
                _ => 1.0,
            };
            let (y, w) = match i % 9 {
                4 => (None, 1.0),
                7 => (Some(y), 0.0),
                _ => (Some(y), 1.0),
            };
            let p = m.step(&x, &[y], d, w).pred[0];
            wsig *= 0.5f64.powf(d / hl);
            match y {
                Some(y) if w > 0.0 && p.is_finite() => {
                    let r = y - p;
                    let ws_new = wsig + w;
                    want = (wsig * want + w * r * r) / ws_new;
                    wsig = ws_new;
                }
                Some(_) if w > 0.0 && wsig > 0.0 => unpredicted += 1,
                _ => {}
            }
            let got = m.sigma2()[0];
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "row {i}: sigma2 {got}, the EW mean of the squared errors {want}"
            );
        }
        assert!(
            unpredicted >= 2,
            "the gap must leave rows with no prediction"
        );
    }

    /// Under `ridge_decay` the penalty is a pseudo-observation of the
    /// history, `prior_scale · ridge · I` on the sum scale, and it decays
    /// with the history; a blend mixes the history, so it mixes the prior
    /// by the same coefficients as the moments -- `(1 − f)` of the fast
    /// side's and `f` of the twin's. The blend rebuilt the Gram from
    /// `EwCov::new`, which put the prior back at full strength on every
    /// session boundary (review 2026-09-12, C6). At `f = 1` the blend is the
    /// twin, fit included: the fit RLS gives at the twin's halflife.
    #[test]
    fn a_blend_mixes_the_decaying_prior_as_it_mixes_the_moments() {
        let (hl, long, ridge) = (20.0, 400.0, 5.0);
        let build = |f: f64| {
            let mut c = cfg(2, 1);
            c.ridge = vec![ridge];
            c.ridge_decay = true;
            c.decay = Decay::Halflife(hl);
            c.long_halflife = Some(long);
            c.session_shrink = Some(f);
            c.min_periods = 0.0;
            EwRidge::new(c).unwrap()
        };
        let mut rls = crate::Rls::new(crate::RlsCfg {
            n_features: 2,
            n_targets: 1,
            add_intercept: true,
            decay: Decay::Halflife(long),
            ridge,
            coef_prior: None,
            min_periods: 0.0,
        })
        .unwrap();
        let (mut part, mut full) = (build(0.3), build(1.0));
        let mut s = 31u64;
        for i in 0..250 {
            let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
            let y = 1.0 + 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            part.step(&x, &[Some(y)], d, 1.0);
            full.step(&x, &[Some(y)], d, 1.0);
            rls.step(&x, &[Some(y)], d, 1.0);
        }
        let fast = part.gram(0).prior_scale();
        let slow = part.slow.as_ref().unwrap().grams.grams[0].prior_scale();
        assert!(fast < 1e-3 && slow > 0.5, "fast {fast}, slow {slow}");
        part.blend_toward_long_run();
        let want = (1.0 - 0.3) * fast + 0.3 * slow;
        assert!(
            (part.gram(0).prior_scale() - want).abs() <= 1e-15 * want,
            "prior_scale {} after the blend, the mixture {want}",
            part.gram(0).prior_scale()
        );

        full.blend_toward_long_run();
        let got = full.coefficients().unwrap()[0].clone();
        let wanted = rls.coefficients()[0].clone();
        for i in 0..3 {
            assert!(
                (got[i] - wanted[i]).abs() < 1e-9,
                "coef {i}: {} after a full blend, {} from RLS at the twin's halflife",
                got[i],
                wanted[i]
            );
        }
    }

    /// The window's boundary across a clock gap longer than the window,
    /// through the model: the row after the gap is the only one inside, so
    /// the window's weight is that row's, at every `window_every`. With a
    /// cadence of 5 the boundary stayed at the last snapshot before the gap,
    /// and the rows since it stayed in (found testing review 2026-09-12, S6).
    #[test]
    fn a_window_holds_nothing_from_before_a_gap_longer_than_it() {
        for every in [1, 5] {
            let mut c = cfg(1, 1);
            c.window = Some(20.0);
            c.window_every = Some(every);
            c.min_periods = 0.0;
            let mut m = EwRidge::new(c).unwrap();
            let mut s = 67u64;
            for i in 0..101 {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            for (rows, d) in [(1.0, 100.0), (2.0, 1.0)] {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(x[0])], d, 1.0);
                assert!(
                    (m.n_eff() - rows).abs() < 1e-9,
                    "window_every {every}: a weight of {} in the window, {rows} rows inside it",
                    m.n_eff()
                );
            }
        }
    }

    /// `predict` and `n_eff` read the window's weights without the O(k²)
    /// view (review 2026-09-12, P1), and must read the view's numbers, to
    /// the bit, on every row: before the window has aged anything out, while
    /// it does, across a gap that empties it, and per target with gaps.
    #[test]
    fn the_window_weights_are_the_views_to_the_bit() {
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(15.0);
        c.window = Some(12.0);
        c.window_every = Some(3);
        c.min_periods = 0.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 71u64;
        for i in 0..120 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y0 = (i % 5 != 2).then_some(x[0] - x[1]);
            let y1 = (i % 3 != 1).then_some(x[0] + 0.5);
            let d = match i {
                0 => 0.0,
                60 => 40.0,
                _ => 1.0,
            };
            m.step(&x, &[y0, y1], d, if i % 11 == 4 { 0.0 } else { 1.0 });
            let live = (m.acc.cross.w, m.acc.wj.clone());
            let cheap = m.window_weights().unwrap_or_else(|| live.clone());
            let full = m
                .view()
                .map_or_else(|| live.clone(), |v| (v.acc.cross.w, v.acc.wj.clone()));
            assert_eq!(cheap.0.to_bits(), full.0.to_bits(), "row {i}");
            assert_eq!(cheap.1, full.1, "row {i}");
        }
    }

    #[test]
    fn the_solve_schedule_controls_when_coefficients_move() {
        // Almost every test here solves on every row, which masks the three
        // clauses of the `due` condition. Each is checked on its own.
        let coefs = |c: EwRidgeCfg, ds: &[f64]| {
            let mut m = EwRidge::new(c).unwrap();
            let mut s = 59u64;
            let mut out = Vec::new();
            for (i, &d) in ds.iter().enumerate() {
                let x = [lcg(&mut s)];
                m.step(&x, &[Some(3.0 * x[0])], if i == 0 { 0.0 } else { d }, 1.0);
                out.push(m.coefficients().map(|b| b[0][1]));
            }
            out
        };
        let ones = vec![1.0; 24];

        // Row-counted. `solve_every = 0` means "every row" and short-circuits
        // the rest of the condition, so the row cap is only visible with a
        // clock schedule that will not fire.
        let mut c = cfg(1, 1);
        c.min_periods = 2.0;
        c.solve_every = 1e9;
        c.max_rows_between_solves = 5;
        let by_rows = coefs(c, &ones);
        let changes: Vec<usize> = by_rows
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0] != w[1])
            .map(|(i, _)| i + 1)
            .collect();
        // First solve as soon as min_periods is met, then strictly every 5
        // accepted rows: nothing in between, and it does not stop.
        assert_eq!(changes, vec![1, 6, 11, 16, 21], "24 rows, cap of 5");

        // Clock-counted: the same stream on a clock that advances 2 per row
        // must solve half as often when `solve_every` is 4.
        let mut c = cfg(1, 1);
        c.min_periods = 2.0;
        c.max_rows_between_solves = u32::MAX;
        c.solve_every = 4.0;
        let slow = coefs(c, &[2.0; 24]);
        let n_slow = slow.windows(2).filter(|w| w[0] != w[1]).count();

        let mut c = cfg(1, 1);
        c.min_periods = 2.0;
        c.max_rows_between_solves = u32::MAX;
        c.solve_every = 0.0; // 0 means "every row"
        let every = coefs(c, &ones);
        let n_every = every.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(
            n_slow < n_every,
            "solve_every should throttle: {n_slow} vs {n_every}"
        );

        // The first solve is not throttled: it happens as soon as min_periods
        // is met, however long the schedule says to wait.
        let mut c = cfg(1, 1);
        c.min_periods = 3.0;
        c.max_rows_between_solves = u32::MAX;
        c.solve_every = 1e9;
        let first = coefs(c, &ones);
        assert!(
            first.iter().take(6).any(|b| b.is_some()),
            "the first solve must not wait for the schedule"
        );
    }

    #[test]
    fn cfg_validation_rejects_each_bad_field() {
        // One case per rejection in `EwRidgeCfg::validate`, each matched on the
        // message so a mutation that reports the wrong reason is caught too --
        // and each accompanied by the nearest *valid* config, so a validator
        // that rejects everything cannot pass either.
        let bad = |f: &dyn Fn(&mut EwRidgeCfg), want: &str| {
            let mut c = cfg(2, 1);
            f(&mut c);
            match c.validate() {
                Err(e) => assert!(e.contains(want), "wanted {want:?}, got {e:?}"),
                Ok(()) => panic!("expected rejection mentioning {want:?}"),
            }
        };
        let good = |f: &dyn Fn(&mut EwRidgeCfg)| {
            let mut c = cfg(2, 1);
            f(&mut c);
            c.validate().expect("should be accepted");
        };

        bad(&|c| c.n_features = 0, "must be >= 1");
        bad(&|c| c.n_targets = 0, "must be >= 1");
        bad(&|c| c.ridge = vec![], "at least one value");

        // ridge_decay alone is fine; it is the combination that is refused.
        good(&|c| c.ridge_decay = true);
        bad(
            &|c| {
                c.ridge_decay = true;
                c.standardize = true;
            },
            "incompatible",
        );
        bad(
            &|c| {
                c.ridge_decay = true;
                c.ridge = vec![1e-6, 1.0];
            },
            "incompatible",
        );

        bad(&|c| c.session_shrink = Some(-0.1), "in [0, 1]");
        bad(&|c| c.session_shrink = Some(1.5), "in [0, 1]");
        bad(&|c| c.session_shrink = Some(0.5), "needs long_halflife");
        bad(&|c| c.long_halflife = Some(100.0), "no effect without");
        bad(
            &|c| {
                c.session_shrink = Some(0.5);
                c.long_halflife = Some(0.0);
            },
            "must be > 0",
        );
        bad(
            &|c| {
                c.session_shrink = Some(0.5);
                c.long_halflife = Some(f64::NAN);
            },
            "must be > 0",
        );
        good(&|c| {
            c.session_shrink = Some(0.0);
            c.long_halflife = Some(100.0);
        });
        good(&|c| {
            c.session_shrink = Some(1.0);
            c.long_halflife = Some(100.0);
        });

        // coef_prior is one vector per target, each of length k_total (2 + intercept).
        bad(
            &|c| c.coef_prior = Some(vec![vec![0.0; 3], vec![0.0; 3]]),
            "1 vector of",
        );
        bad(&|c| c.coef_prior = Some(vec![vec![0.0; 2]]), "length 3");
        bad(
            &|c| c.coef_prior = Some(vec![vec![0.0, 0.0, f64::NAN]]),
            "finite",
        );
        bad(
            &|c| c.coef_prior = Some(vec![vec![0.0, 0.0, f64::INFINITY]]),
            "finite",
        );
        good(&|c| c.coef_prior = Some(vec![vec![1.0, 2.0, 3.0]]));

        // Empty is its own message: it named "out-of-range indices", which
        // an empty set has not got (review 2026-09-12, S7).
        bad(&|c| c.feature_sets = vec![("a".into(), vec![])], "is empty");
        bad(
            &|c| c.feature_sets = vec![("a".into(), vec![2])],
            "out-of-range",
        );
        good(&|c| c.feature_sets = vec![("a".into(), vec![0]), ("b".into(), vec![0, 1])]);

        cfg(2, 1).validate().expect("the baseline config is valid");
    }

    #[test]
    fn blend_is_the_weight_respecting_mixture() {
        // Every arithmetic step of `blend_toward_long_run` is checked against
        // the same quantities recomputed by hand from the pre-blend state, so
        // a factor applied to the wrong side, a missing re-centering, or a
        // swapped index all show up.
        let f = 0.3;
        let mut m = blended_pair(f);
        let before = m.clone();
        let slow = before.slow.as_ref().unwrap();
        let k = m.cfg.k_total();

        let (wf, ws) = (before.gram(0).n_eff(), slow.grams.grams[0].n_eff());
        assert!(wf > 0.0 && ws > wf, "the slow twin should hold more weight");
        let w_new = (1.0 - f) * wf + f * ws;
        let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);

        m.blend_toward_long_run();

        assert!((m.gram(0).n_eff() - w_new).abs() < 1e-12);
        for i in 0..k {
            let want = af * before.gram(0).mean(i) + as_ * slow.grams.grams[0].mean(i);
            assert!(
                (m.gram(0).mean(i) - want).abs() < 1e-12,
                "mean {i}: {} vs {want}",
                m.gram(0).mean(i)
            );
        }
        for i in 0..k {
            for j in 0..k {
                // Raw moments mix linearly; the centered moment must then be
                // re-derived against the *mixed* mean, not either input's.
                let raw = af * before.gram(0).raw(i, j) + as_ * slow.grams.grams[0].raw(i, j);
                let want = raw - m.gram(0).mean(i) * m.gram(0).mean(j);
                assert!(
                    (m.gram(0).cov(i, j) - want).abs() < 1e-10,
                    "cov {i},{j}: {} vs {want}",
                    m.gram(0).cov(i, j)
                );
            }
        }
        // The raw cross-moments mix linearly, whatever split the centred ones
        // were mixed in.
        for (j, got) in m.cross_moments().iter().enumerate() {
            let (wf, ws) = (before.acc.wj[j], slow.wj[j]);
            let w_new = (1.0 - f) * wf + f * ws;
            let (af, as_) = ((1.0 - f) * wf / w_new, f * ws / w_new);
            assert!((m.acc.wj[j] - w_new).abs() < 1e-12);
            let (fast_r, slow_r) = (before.acc.cross.raw(j), slow.cross.raw(j));
            for i in 0..k {
                let want = af * fast_r[i] + as_ * slow_r[i];
                assert!((got[i] - want).abs() < 1e-12, "r[{j}][{i}]");
            }
        }
    }

    #[test]
    fn blend_endpoints_are_identity_and_full_replacement() {
        // f = 0 must not touch the state; f = 1 must land exactly on the twin.
        let mut zero = blended_pair(0.0);
        let before = zero.clone();
        zero.blend_toward_long_run();
        assert_eq!(zero, before, "session_shrink = 0 must be a no-op");

        let mut one = blended_pair(1.0);
        let slow = one.slow.clone().unwrap();
        one.blend_toward_long_run();
        let k = one.cfg.k_total();
        assert!((one.gram(0).n_eff() - slow.grams.grams[0].n_eff()).abs() < 1e-12);
        for i in 0..k {
            assert!(
                (one.gram(0).mean(i) - slow.grams.grams[0].mean(i)).abs() < 1e-12,
                "mean {i}"
            );
        }
        for (j, got) in one.cross_moments().iter().enumerate() {
            assert!((one.acc.wj[j] - slow.wj[j]).abs() < 1e-12);
            let want = slow.cross.raw(j);
            for i in 0..k {
                assert!((got[i] - want[i]).abs() < 1e-12, "r[{j}][{i}]");
            }
        }
    }

    #[test]
    fn blend_before_any_data_is_a_no_op() {
        // The other half of the doc comment's promise: a no-op when the twin
        // is not configured, *or before it has seen anything*. With both sides
        // at zero weight the mixture's denominator is zero, and the guard has
        // to catch that rather than divide.
        let mut c = cfg(2, 1);
        c.session_shrink = Some(0.5);
        c.long_halflife = Some(200.0);
        let mut m = EwRidge::new(c).unwrap();
        assert!(m.slow.is_some());
        let before = m.clone();
        m.blend_toward_long_run();
        assert_eq!(m, before, "nothing seen yet: nothing to blend");
        for i in 0..m.cfg.k_total() {
            assert!(m.gram(0).mean(i).is_finite(), "and no NaN got in");
        }

        // Still safe on a session boundary that arrives with a session's worth
        // of zero-weight rows behind it.
        for _ in 0..5 {
            m.step(&[1.0, 2.0], &[Some(3.0)], 1.0, 0.0);
        }
        let before = m.clone();
        m.blend_toward_long_run();
        assert_eq!(m, before, "zero-weight rows carry no weight to blend");
    }

    #[test]
    fn blend_without_a_twin_is_a_no_op() {
        // No `session_shrink` at all: there is no twin to blend with, and the
        // method must return before touching anything.
        let mut c = cfg(2, 1);
        c.min_periods = 3.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 29u64;
        for i in 0..50 {
            let x = [lcg(&mut s), lcg(&mut s)];
            m.step(&x, &[Some(x[0])], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        assert!(m.slow.is_none());
        let before = m.clone();
        m.blend_toward_long_run();
        assert_eq!(m, before);
    }

    #[test]
    fn blend_moves_the_fit_toward_the_long_run_relationship() {
        // The point of the feature, not just its arithmetic: the last session
        // ran at the opposite sign to the long run, and blending must pull the
        // fit back -- monotonically in the shrink parameter.
        let slope = |f: f64| {
            let mut m = blended_pair(f);
            m.blend_toward_long_run();
            m.coefficients_after_blend()
        };
        let (none, half, full) = (slope(0.0), slope(0.5), slope(1.0));
        assert!(none < 0.0, "the last session was negative: {none}");
        assert!(
            full > none,
            "the long run should pull it up: {full} vs {none}"
        );
        assert!(
            none < half && half < full,
            "reversion should be monotone: {none} < {half} < {full}"
        );
    }

    #[test]
    fn standardize_without_intercept_matches_plain_when_ridge_tiny() {
        // `solve_standardized` has a second, quite different branch for
        // `add_intercept = false`: it scales by the *raw* second-moment
        // diagonals rather than centering first. The invariance is the same --
        // a diagonal rescale of the normal equations cannot move the solution
        // when the penalty is negligible -- so a scaling applied in the wrong
        // direction, or to only one of A, b and the unscaled result, shows up
        // here. Features are deliberately on very different scales (~4 and
        // ~0.003) so the scaling matrix is far from the identity.
        let mut ca = cfg(2, 1);
        ca.add_intercept = false;
        ca.min_periods = 2.0;
        // The invariance is exact only at ridge = 0: a penalty applies on the
        // raw scale in one path and the standardized scale in the other, and
        // the two feature scales here differ by ~1e3, so 1e-8 is enough to
        // separate them at 1e-6.
        ca.ridge = vec![1e-12];
        let mut cb = ca.clone();
        cb.standardize = true;
        let mut ma = EwRidge::new(ca).unwrap();
        let mut mb = EwRidge::new(cb).unwrap();
        let mut s = 17u64;
        for i in 0..300 {
            let x = [4.0 + lcg(&mut s), 0.003 * lcg(&mut s)];
            let y = 0.25 * x[0] - 40.0 * x[1] + 0.001 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            ma.step(&x, &[Some(y)], d, 1.0);
            mb.step(&x, &[Some(y)], d, 1.0);
        }
        let a = &ma.coefficients().unwrap()[0];
        let b = &mb.coefficients().unwrap()[0];
        assert_eq!(a.len(), 2, "no intercept slot when add_intercept is false");
        for i in 0..2 {
            assert!(
                (a[i] - b[i]).abs() < 1e-6 * (1.0 + a[i].abs()),
                "coef {i}: plain {} vs standardized {}",
                a[i],
                b[i]
            );
        }
        // And it actually recovered the generating coefficients, so the
        // agreement is not two paths being wrong the same way.
        assert!((b[0] - 0.25).abs() < 1e-3, "{}", b[0]);
        assert!((b[1] + 40.0).abs() < 1.0, "{}", b[1]);
    }

    #[test]
    fn standardize_without_intercept_drops_a_zero_column() {
        // The `s[i] > 0.0` guard in the no-intercept branch: a feature that is
        // identically zero has zero raw second moment, so it cannot be scaled
        // and must come out with a zero coefficient rather than a NaN.
        let mut c = cfg(2, 1);
        c.add_intercept = false;
        c.standardize = true;
        c.min_periods = 2.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 19u64;
        for i in 0..100 {
            let x = [1.0 + lcg(&mut s), 0.0];
            let y = 2.0 * x[0];
            m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let b = &m.coefficients().unwrap()[0];
        assert_eq!(b[1], 0.0, "the all-zero feature must be dropped, not NaN");
        assert!((b[0] - 2.0).abs() < 1e-6, "{}", b[0]);
    }

    #[test]
    fn zero_variance_feature_dropped_in_standardized_solve() {
        let mut c = cfg(2, 1);
        c.standardize = true;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 13u64;
        for i in 0..100 {
            let x = [lcg(&mut s), 5.0]; // constant second feature
            let y = 2.0 * x[0];
            let st = m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            if i > 10 {
                assert!(st.pred[0].is_finite(), "row {i}");
            }
        }
        let b = &m.coefficients().unwrap()[0];
        assert_eq!(b[2], 0.0, "full coef {b:?}"); // dropped, not blown up
        assert!((b[1] - 2.0).abs() < 1e-6); // 1e-8 ridge itself shifts this by ~2e-8
    }

    /// A windowed ridge fit, solved directly from the normal equations over
    /// exactly the rows inside the window. Written from the definition:
    /// `(Z'WZ + ridge·I) beta = Z'Wy` with `W = diag(lam^age)` over the rows
    /// whose age is under the window, and nothing else.
    fn direct_window_fit(
        xs: &[[f64; 1]],
        ys: &[f64],
        t: &[f64],
        halflife: f64,
        window: f64,
        ridge: f64,
        upto: usize,
    ) -> [f64; 2] {
        let now = t[upto - 1];
        // z = [1, x]: a 2x2 normal system, solved in closed form.
        let (mut s11, mut s1x, mut sxx, mut s1y, mut sxy, mut wsum) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..upto {
            if now - t[i] >= window {
                continue;
            }
            let w = 0.5_f64.powf((now - t[i]) / halflife);
            let x = xs[i][0];
            wsum += w;
            s11 += w;
            s1x += w * x;
            sxx += w * x * x;
            s1y += w * ys[i];
            sxy += w * x * ys[i];
        }
        // The model accumulates *means*, so the ridge sits on the mean scale,
        // and on the slope alone: the intercept is not penalized.
        let (a11, a12, a22) = (s11 / wsum, s1x / wsum, sxx / wsum + ridge);
        let (b1, b2) = (s1y / wsum, sxy / wsum);
        let det = a11 * a22 - a12 * a12;
        [(b1 * a22 - b2 * a12) / det, (b2 * a11 - b1 * a12) / det]
    }

    /// PLAN §13.4 (1), for the Gram: a windowed fit is the fit of the rows in
    /// the window, and nothing older reaches the coefficients.
    #[test]
    fn a_windowed_fit_is_the_fit_of_the_rows_inside_the_window() {
        let (halflife, window, ridge) = (30.0, 80.0, 1e-8);
        let mut c = cfg(1, 1);
        c.decay = Decay::Halflife(halflife);
        c.ridge = vec![ridge];
        c.min_periods = 0.0;
        c.solve_every = 0.0;
        c.max_rows_between_solves = 1; // solve every row, so beta is never stale
        c.window = Some(window);
        let mut m = EwRidge::new(c).unwrap();

        let mut s = 11u64;
        let (mut xs, mut ys, mut t, mut clock) = (vec![], vec![], vec![], 0.0);
        for i in 0..150 {
            // `lcg` is in [-1, 1), so take its magnitude: a clock must not go
            // backwards, and one that does makes the window meaningless.
            let d = if i == 0 {
                0.0
            } else {
                0.5 + 2.0 * lcg(&mut s).abs()
            };
            clock += d;
            let x = [lcg(&mut s) * 4.0 - 2.0];
            // A relationship that changes halfway, so a window that forgets
            // the old one reports something the full history could not.
            let y = if i < 75 {
                3.0 * x[0] + 1.0
            } else {
                -2.0 * x[0] + 5.0
            } + 0.05 * lcg(&mut s);
            m.step(&x, &[Some(y)], d, 1.0);
            xs.push(x);
            ys.push(y);
            t.push(clock);
            if i >= 8 {
                // Membership first: if the truncated weight matches the sum
                // over the rows the definition keeps, the two agree on *which*
                // rows are in, and any difference left is arithmetic.
                let now = t[i];
                let wsum: f64 = (0..=i)
                    .filter(|&j| now - t[j] < window)
                    .map(|j| 0.5_f64.powf((now - t[j]) / halflife))
                    .sum();
                assert!(
                    (m.n_eff() - wsum).abs() < 1e-9 * wsum,
                    "row {i}: n_eff {} vs {wsum} -- the window holds different rows",
                    m.n_eff()
                );
                let want = direct_window_fit(&xs, &ys, &t, halflife, window, ridge, i + 1);
                let got = m.coefficients().unwrap();
                for (slot, wanted) in want.iter().enumerate() {
                    // To rounding: the window is a subtraction, but at 2.7
                    // halflives it discards a sixth of the weight and loses
                    // next to nothing (1.2e-14 measured). The "eight
                    // significant figures" this once allowed was the oracle's
                    // own ridge on the intercept, which the model leaves free
                    // (review 2026-09-12, D6).
                    assert!(
                        (got[0][slot] - wanted).abs() < 1e-12 * wanted.abs().max(1.0),
                        "row {i} slot {slot}: {} vs {wanted}",
                        got[0][slot]
                    );
                }
            }
        }
    }

    /// PLAN §13.4 (2): the guarantee, on the fit rather than a moment. A
    /// relationship that held before the window cannot bend the coefficients
    /// inside it.
    #[test]
    fn a_fit_cannot_be_moved_by_rows_older_than_its_window() {
        let run = |old_slope: f64| {
            let mut c = cfg(1, 1);
            c.decay = Decay::Halflife(25.0);
            c.ridge = vec![1e-8];
            c.min_periods = 0.0;
            c.max_rows_between_solves = 1;
            c.window = Some(60.0);
            let mut m = EwRidge::new(c).unwrap();
            let mut s = 3u64;
            for i in 0..200 {
                let x = [lcg(&mut s) * 2.0 - 1.0];
                // Everything before row 130 follows `old_slope`; the last 70
                // rows are the same in both runs and all lie inside a
                // 60-unit window at the end.
                let y = if i < 130 {
                    old_slope * x[0]
                } else {
                    4.0 * x[0]
                };
                m.step(&x, &[Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
            }
            m.coefficients().unwrap()[0].clone()
        };
        let a = run(1.0);
        let b = run(-500.0);
        for slot in 0..2 {
            assert!(
                (a[slot] - b[slot]).abs() < 1e-6,
                "slot {slot}: a slope of -500 outside the window moved the fit: {} vs {}",
                a[slot],
                b[slot]
            );
        }
        assert!(
            (a[1] - 4.0).abs() < 0.01,
            "the in-window slope is 4: {}",
            a[1]
        );
    }

    /// PLAN §13.4 (5): a window no stream reaches changes nothing.
    #[test]
    fn a_ridge_window_no_stream_reaches_is_the_untruncated_fit() {
        let mk = |window: Option<f64>| {
            let mut c = cfg(2, 1);
            c.decay = Decay::Halflife(40.0);
            c.min_periods = 0.0;
            c.max_rows_between_solves = 1;
            c.window = window;
            EwRidge::new(c).unwrap()
        };
        let (mut plain, mut windowed) = (mk(None), mk(Some(1e9)));
        let mut s = 7u64;
        for i in 0..80 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] - 2.0 * x[1] + 0.1 * lcg(&mut s);
            let d = if i == 0 { 0.0 } else { 1.0 };
            let a = plain.step(&x, &[Some(y)], d, 1.0);
            let b = windowed.step(&x, &[Some(y)], d, 1.0);
            assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits(), "row {i}");
            assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "row {i} n_eff");
        }
    }

    #[test]
    fn state_roundtrip_continues_identically() {
        let mut m1 = EwRidge::new(cfg(2, 1)).unwrap();
        let mut s = 5u64;
        let mut rows = vec![];
        for _ in 0..60 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = x[0] + 0.1 * lcg(&mut s);
            rows.push((x, y));
        }
        for (i, (x, y)) in rows[..30].iter().enumerate() {
            m1.step(x, &[Some(*y)], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m1.state()).unwrap();
        let restored: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut m2 = EwRidge::restore(&restored).unwrap();
        for (x, y) in &rows[30..] {
            let a = m1.step(x, &[Some(*y)], 1.0, 1.0);
            let b = m2.step(x, &[Some(*y)], 1.0, 1.0);
            assert_eq!(a.pred, b.pred);
            assert_eq!(a.n_eff, b.n_eff);
        }
    }

    #[test]
    fn null_target_is_predict_only() {
        let mut m = EwRidge::new(cfg(1, 2)).unwrap();
        let mut s = 17u64;
        for i in 0..50 {
            let x = [lcg(&mut s)];
            m.step(
                &x,
                &[Some(2.0 * x[0]), Some(-x[0])],
                if i == 0 { 0.0 } else { 1.0 },
                1.0,
            );
        }
        let (c1, my1) = (m.acc.cross.c[1].clone(), m.acc.cross.my[1]);
        let mean_z1: Vec<f64> = (0..2)
            .map(|i| m.acc.cross.m[i] + m.acc.cross.d[1][i])
            .collect();
        let st = m.step(&[0.5], &[Some(1.0), None], 1.0, 1.0);
        assert!(st.pred[1].is_finite()); // pred still emitted
        // Target 1's own moments do not move: no data added (mean form). The
        // mean of `z` over its rows is the all-row mean plus its offset, and
        // the all-row mean did move, so the offset took the step back.
        assert_eq!(m.acc.cross.c[1], c1);
        assert_eq!(m.acc.cross.my[1], my1);
        for (i, want) in mean_z1.iter().enumerate() {
            let got = m.acc.cross.m[i] + m.acc.cross.d[1][i];
            assert!(
                (got - want).abs() < 1e-15,
                "the mean of z[{i}] moved: {got} vs {want}"
            );
        }
    }

    /// A level costs the fit nothing (review 2026-09-12, N1). The same stream
    /// at the origin and shifted by `1e8` -- features and target alike, a
    /// price regressed on prices -- must give the same slopes, and
    /// predictions that differ by the shift. The cross-moments were raw, and
    /// the solves formed `E[z·y] − m·ȳ` or read the raw normal equations,
    /// both of which lose `L²·ε`: at `1e8` the prediction was off by about
    /// ten. Blocked too, where the Gram's own mean moves only at a flush and
    /// the cross-moments keep theirs. The tolerances are the data's own
    /// resolution at `1e8`, `ulp ≈ 1.5e-8`.
    #[test]
    fn a_level_costs_the_fit_nothing() {
        for (standardize, block) in [(false, 0), (true, 0), (false, 8), (true, 8)] {
            let run = |level: f64| {
                let mut c = cfg(2, 1);
                c.standardize = standardize;
                c.ridge = vec![1e-6];
                c.decay = Decay::Halflife(200.0);
                c.min_periods = 10.0;
                if block > 0 {
                    c.gram_block_rows = block;
                    c.solve_every = 5.0;
                    c.max_rows_between_solves = 10;
                }
                let mut m = EwRidge::new(c).unwrap();
                let mut s = 29u64;
                let mut preds = Vec::new();
                for i in 0..600 {
                    let u = [lcg(&mut s), lcg(&mut s)];
                    let x = [level + u[0], level + u[1]];
                    let y = level + 2.0 * u[0] - u[1] + 0.1 * lcg(&mut s);
                    let d = if i == 0 { 0.0 } else { 1.0 };
                    preds.push(m.step(&x, &[Some(y)], d, 1.0).pred[0] - level);
                }
                (preds, m.coefficients().unwrap()[0].clone())
            };
            let what = format!("standardize {standardize}, block {block}");
            let ((p0, b0), (p8, b8)) = (run(0.0), run(1e8));
            for i in 1..3 {
                assert!(
                    (b0[i] - b8[i]).abs() < 1e-6,
                    "{what}: slope {i}: {} vs {}",
                    b0[i],
                    b8[i]
                );
            }
            let mut worst = 0.0f64;
            for (t, (a, b)) in p0.iter().zip(&p8).enumerate() {
                assert_eq!(a.is_finite(), b.is_finite(), "{what}: row {t}: {a} vs {b}");
                if a.is_finite() {
                    worst = worst.max((a - b).abs());
                }
            }
            assert!(worst < 1e-5, "{what}: the predictions part by {worst}");
        }
    }

    /// A target's offset from the all-row mean is exactly 0 while it has been
    /// present on every row, blocked or not -- what the centred right-hand
    /// side rests on, since `c + δ·ȳ` is then `c` and nothing level-sized
    /// enters it -- and a genuine number for a target with gaps (N1).
    #[test]
    fn a_target_present_on_every_row_has_no_offset() {
        for block in [0, 8] {
            let mut c = cfg(2, 2);
            c.decay = Decay::Halflife(50.0);
            if block > 0 {
                c.gram_block_rows = block;
                c.solve_every = 5.0;
                c.max_rows_between_solves = 10;
            }
            let mut m = EwRidge::new(c).unwrap();
            let mut s = 31u64;
            for i in 0..100 {
                let x = [1e8 + lcg(&mut s), 3.0 + lcg(&mut s)];
                // Target 1 is null every fifth row; target 0 never is.
                let y1 = (i % 5 != 2).then_some(x[1]);
                let d = if i == 0 { 0.0 } else { 1.0 };
                m.step(&x, &[Some(x[0] + x[1]), y1], d, 1.0);
            }
            assert!(
                m.acc.cross.d[0].iter().all(|&v| v == 0.0),
                "block {block}: {:?}",
                m.acc.cross.d[0]
            );
            assert!(
                m.acc.cross.d[1][2] != 0.0,
                "block {block}: a target with gaps"
            );
            // Unblocked, the cross-moments' copy of the Gram's mean is the
            // Gram's own, to the bit.
            if block == 0 {
                assert_eq!(m.acc.cross.m.as_slice(), m.gram(0).means());
            }
        }
    }

    // ---- gram_block_rows (docs/ENHANCEMENTS.md E51, docs/PLAN.md task 71) ----

    /// A halflife, a solve cadence in both clock and rows, and a small ridge:
    /// the setting the option is for. `block` is the only difference between
    /// the two sides of every comparison below.
    fn blocked_cfg(k: usize, block: usize) -> EwRidgeCfg {
        let mut c = cfg(k, 1);
        c.decay = Decay::Halflife(200.0);
        c.ridge = vec![1e-3];
        c.solve_every = 10.0;
        c.max_rows_between_solves = 50;
        c.gram_block_rows = block;
        c
    }

    /// `(x, y, d_clock, weight)`: a missing target every 17th row, a zero
    /// weight every 23rd, a clock gap of 5 halflives every 41st and one
    /// infinite gap (`lam == 0`, the history discarded) -- everything a held
    /// row has to carry into the merge.
    fn ridge_rows(n: usize, k: usize, seed: u64) -> Vec<(Vec<f64>, Option<f64>, f64, f64)> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let x: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
                let y: f64 = x
                    .iter()
                    .enumerate()
                    .map(|(j, v)| (j as f64 + 1.0) * v)
                    .sum::<f64>()
                    + 0.5
                    + 0.1 * lcg(&mut s);
                let y = if i % 17 == 5 { None } else { Some(y) };
                let d = match i {
                    0 => 0.0,
                    301 => f64::INFINITY,
                    _ if i % 41 == 0 => 1000.0,
                    _ => 1.0,
                };
                let w = if i % 23 == 7 {
                    0.0
                } else {
                    0.5 + lcg(&mut s).abs()
                };
                (x, y, d, w)
            })
            .collect()
    }

    fn assert_pred_close(a: &[f64], b: &[f64], what: &str) {
        for (i, (p, q)) in a.iter().zip(b).enumerate() {
            let same = (p.is_nan() && q.is_nan()) || (p - q).abs() <= 1e-9 * (1.0 + p.abs());
            assert!(same, "{what}: pred[{i}] {p} vs {q}");
        }
    }

    /// The merge is the per-row recursion to rounding: the same predictions
    /// at every row, the same `n_eff` to the bit (the scalars never left the
    /// per-row path), and the same coefficients at the end. Checked with
    /// blocks smaller than, equal to and larger than the solve cadence, and
    /// with the path actually taken: a block was pending at some point.
    #[test]
    fn a_blocked_fit_is_the_per_row_fit_to_rounding() {
        for k in [1usize, 3, 12] {
            for block in [1usize, 7, 50, 64] {
                let mut plain = EwRidge::new(blocked_cfg(k, 0)).unwrap();
                let mut blocked = EwRidge::new(blocked_cfg(k, block)).unwrap();
                let mut held = false;
                for (i, (x, y, d, w)) in ridge_rows(600, k, 9 + k as u64).iter().enumerate() {
                    let a = plain.step(x, &[*y], *d, *w);
                    let b = blocked.step(x, &[*y], *d, *w);
                    assert_pred_close(&a.pred, &b.pred, &format!("k={k} block={block} row {i}"));
                    assert_eq!(a.n_eff, b.n_eff, "k={k} block={block} row {i}: n_eff");
                    held |= blocked.gram(0).has_pending();
                }
                // A block of one fills on the row that opens it, so nothing
                // is ever seen pending; every larger block is.
                assert_eq!(
                    held,
                    block > 1,
                    "k={k} block={block}: rows held between steps"
                );
                let (ca, cb) = (
                    plain.coefficients().unwrap(),
                    blocked.coefficients().unwrap(),
                );
                for (i, (p, q)) in ca[0].iter().zip(&cb[0]).enumerate() {
                    assert!(
                        (p - q).abs() <= 1e-9 * (1.0 + p.abs()),
                        "k={k} block={block}: coef[{i}] {p} vs {q}"
                    );
                }
            }
        }
    }

    /// A solve reads the matrix, so it merges the block first: after every
    /// solve nothing is pending, whichever of the two schedules fired it.
    #[test]
    fn a_solve_merges_the_block_first() {
        let mut m = EwRidge::new(blocked_cfg(3, 64)).unwrap();
        let mut solves = 0;
        for (x, y, d, w) in ridge_rows(400, 3, 4) {
            m.step(&x, &[y], d, w);
            if m.rows_since_solve == 0 {
                solves += 1;
                assert!(!m.gram(0).has_pending(), "a solve left rows pending");
            }
        }
        assert!(solves > 10, "the schedule fired {solves} times");
    }

    /// A blend reads both matrices in full, so it merges both blocks first,
    /// and the blended accumulator keeps the block size: the option is a
    /// property of the model, not of one accumulator's lifetime.
    #[test]
    fn a_blend_merges_the_held_blocks_first_and_keeps_the_block_size() {
        let shrink = |block: usize| {
            let mut c = blocked_cfg(2, block);
            c.session_shrink = Some(0.5);
            c.long_halflife = Some(1e4);
            EwRidge::new(c).unwrap()
        };
        let (mut plain, mut blocked) = (shrink(0), shrink(16));
        for (x, y, d, w) in ridge_rows(37, 2, 5) {
            plain.step(&x, &[y], d, w);
            blocked.step(&x, &[y], d, w);
        }
        assert!(blocked.gram(0).has_pending(), "row 37 should sit mid-block");
        assert!(blocked.slow.as_ref().unwrap().grams.grams[0].has_pending());
        plain.blend_toward_long_run();
        blocked.blend_toward_long_run();
        assert!(!blocked.gram(0).has_pending());
        assert!(!blocked.slow.as_ref().unwrap().grams.grams[0].has_pending());
        assert_eq!(blocked.gram(0).block_rows(), 16);
        assert_eq!(
            blocked.slow.as_ref().unwrap().grams.grams[0].block_rows(),
            16
        );
        let (a, b) = (
            plain.coefficients_after_blend(),
            blocked.coefficients_after_blend(),
        );
        assert!((a - b).abs() <= 1e-9 * (1.0 + a.abs()), "{a} vs {b}");
        // And the blended accumulator goes on holding rows.
        for (x, y, d, w) in ridge_rows(5, 2, 6) {
            blocked.step(&x, &[y], d, w);
        }
        assert!(blocked.gram(0).has_pending());
    }

    /// The held rows are in the state, so a save mid-block resumes the same
    /// merge on the same rows: bit for bit, both encodings, and the restored
    /// model keeps holding rows rather than flushing on load.
    #[test]
    fn a_state_saved_mid_block_resumes_the_blocked_fit_bit_for_bit() {
        let rows = ridge_rows(120, 3, 8);
        for named in [false, true] {
            let mut one = EwRidge::new(blocked_cfg(3, 32)).unwrap();
            for (x, y, d, w) in &rows[..23] {
                one.step(x, &[*y], *d, *w);
            }
            assert!(one.gram(0).has_pending(), "row 23 should sit mid-block");
            let st = one.state();
            let bytes = if named {
                rmp_serde::to_vec_named(&st).unwrap()
            } else {
                rmp_serde::to_vec(&st).unwrap()
            };
            let restored: State = rmp_serde::from_slice(&bytes).unwrap();
            let mut two = EwRidge::restore(&restored).unwrap();
            assert!(
                two.gram(0).has_pending(),
                "the held rows did not survive the save"
            );
            assert_eq!(two.gram(0).block_rows(), 32);
            for (x, y, d, w) in &rows[23..] {
                let a = one.step(x, &[*y], *d, *w);
                let b = two.step(x, &[*y], *d, *w);
                let bits = |p: &[f64]| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a.pred), bits(&b.pred), "named={named}");
                assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits(), "named={named}");
            }
            assert_eq!(one.coefficients(), two.coefficients());
        }
    }

    /// The settings under which a block cannot pay, or cannot be read,
    /// are refused with the reason; `0` is the untouched path whatever the
    /// schedule.
    #[test]
    fn gram_block_rows_is_refused_where_it_cannot_pay() {
        let err = |c: EwRidgeCfg| EwRidge::new(c).err().unwrap_or_default();

        let mut c = blocked_cfg(3, 64);
        c.window = Some(50.0);
        assert!(err(c).contains("gram_block_rows and window do not combine"));

        let mut c = blocked_cfg(3, 64);
        c.solve_every = 0.0;
        let e = err(c);
        assert!(
            e.contains("needs a solve cadence") && e.contains("solve_every = 0"),
            "{e}"
        );

        let mut c = blocked_cfg(3, 64);
        c.max_rows_between_solves = 1;
        let e = err(c);
        assert!(e.contains("max_rows_between_solves = 1"), "{e}");

        // 2^30 rows of 4 floats is 32 GiB; the twin doubles it.
        let mut c = blocked_cfg(3, 1 << 30);
        c.session_shrink = Some(0.5);
        c.long_halflife = Some(1e4);
        let e = err(c);
        assert!(
            e.contains("over the 256 MiB budget") && e.contains("64.0 GiB") && e.contains("twin"),
            "{e}"
        );
        // Just inside the budget builds.
        let mut c = blocked_cfg(3, (256 << 20) / (4 * 8));
        assert!(EwRidge::new(c.clone()).is_ok());
        c.gram_block_rows += 1;
        assert!(err(c).contains("over the 256 MiB budget"));

        let mut c = blocked_cfg(3, 0);
        c.solve_every = 0.0;
        c.max_rows_between_solves = 1;
        assert!(EwRidge::new(c).is_ok(), "0 is off, whatever the schedule");
    }

    // ---- target_gaps (docs/PLAN.md task 81) ----

    /// Features, three targets, clock delta, weight.
    type GappyRow = ([f64; 2], [Option<f64>; 3], f64, f64);

    /// Three targets with three patterns of missing rows -- present on every
    /// row, on a schedule, and where a feature is low -- beside a model of
    /// each target alone, fed the same rows. Zero-weight rows are in the
    /// stream too, some with targets missing.
    fn gappy_rows(n: usize) -> Vec<GappyRow> {
        let mut s = 61u64;
        (0..n)
            .map(|i| {
                let x = [lcg(&mut s), 2.0 + lcg(&mut s)];
                let base = 1.5 * x[0] - 0.5 * x[1] + 0.05 * lcg(&mut s);
                let y = [
                    Some(3.0 + base),
                    (i % 3 != 1).then_some(-1.0 + base),
                    (x[0] < 0.3).then_some(7.0 + 0.5 * base),
                ];
                let d = if i == 0 { 0.0 } else { 1.0 };
                let w = if i % 11 == 5 {
                    0.0
                } else {
                    0.5 + (i % 3) as f64
                };
                (x, y, d, w)
            })
            .collect()
    }

    /// Under `own_rows` each target of a bank is fitted on exactly its rows:
    /// its Gram is, to the bit, the Gram a model of that target alone keeps
    /// -- whichever targets it shared one with and wherever it split from
    /// them, blocked or not -- and so are its cross-moments, its weight and
    /// its fit. Every row counts toward `n_eff` whatever the targets, and the
    /// split Grams survive a save in both encodings.
    #[test]
    fn under_own_rows_each_target_is_the_model_of_that_target_alone() {
        for block in [0usize, 8] {
            let mut c = cfg(2, 3);
            c.decay = Decay::Halflife(30.0);
            if block > 0 {
                c.gram_block_rows = block;
                c.solve_every = 3.0;
                c.max_rows_between_solves = 5;
            }
            let mut bank = EwRidge::new(c.clone()).unwrap();
            let one = EwRidgeCfg {
                n_targets: 1,
                ..c.clone()
            };
            let mut alone: Vec<EwRidge> =
                (0..3).map(|_| EwRidge::new(one.clone()).unwrap()).collect();
            let rows = gappy_rows(300);
            for (x, y, d, w) in &rows[..200] {
                bank.step(x, y, *d, *w);
                for (j, a) in alone.iter_mut().enumerate() {
                    a.step(x, &[y[j]], *d, *w);
                }
            }
            // Three patterns of missing rows, three Grams.
            assert_eq!(bank.acc.grams.grams.len(), 3, "block {block}");
            for (j, a) in alone.iter().enumerate() {
                let g = &bank.acc.grams.grams[bank.acc.grams.of[j]];
                assert_eq!(g, a.gram(0), "block {block}: target {j}'s Gram");
                assert_eq!(
                    g.n_eff(),
                    bank.acc.wj[j],
                    "block {block}: target {j}'s weight"
                );
                assert_eq!(
                    bank.acc.cross.c[j], a.acc.cross.c[0],
                    "block {block}: c[{j}]"
                );
                assert_eq!(bank.acc.cross.my[j], a.acc.cross.my[0], "block {block}");
                assert_eq!(bank.n_eff(), a.n_eff(), "block {block}: every row counts");
                let (b, want) = (
                    &bank.coefficients().unwrap()[j],
                    &a.coefficients().unwrap()[0],
                );
                for (p, q) in b.iter().zip(want) {
                    assert!(
                        (p - q).abs() <= 1e-12 * (1.0 + q.abs()),
                        "block {block}: target {j}: {b:?} vs {want:?}"
                    );
                }
            }
            for named in [false, true] {
                let st = bank.state();
                let bytes = if named {
                    rmp_serde::to_vec_named(&st).unwrap()
                } else {
                    rmp_serde::to_vec(&st).unwrap()
                };
                let mut back = EwRidge::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
                let mut live = bank.clone();
                for (x, y, d, w) in &rows[200..] {
                    let (p, q) = (live.step(x, y, *d, *w), back.step(x, y, *d, *w));
                    let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&p.pred), bits(&q.pred), "block {block} named={named}");
                }
                assert_eq!(live, back, "block {block} named={named}");
            }
        }
    }

    /// N3 (docs/PLAN.md task 81): a slope no longer moves with the target's
    /// level. The target is present only where `x0 < 0.2`, so its rows' mean
    /// of `x0` is not the all-row mean, and the old right-hand side added
    /// `(m_j − m)·ȳ_j / Var(x)` to the slope. Under either reading the
    /// slopes at a level of `1e3` are the slopes at 0, and the intercept
    /// takes the level.
    #[test]
    fn a_target_level_moves_no_slope_under_either_reading() {
        for gaps in [TargetGaps::OwnRows, TargetGaps::Pairwise] {
            for standardize in [false, true] {
                let run = |level: f64| {
                    let mut c = cfg(2, 1);
                    c.decay = Decay::Halflife(60.0);
                    c.standardize = standardize;
                    c.target_gaps = gaps;
                    c.ridge = vec![1e-6];
                    let mut m = EwRidge::new(c).unwrap();
                    let mut s = 67u64;
                    for i in 0..400 {
                        let x = [lcg(&mut s), lcg(&mut s)];
                        let y = level + 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s);
                        let d = if i == 0 { 0.0 } else { 1.0 };
                        m.step(&x, &[(x[0] < 0.2).then_some(y)], d, 1.0);
                    }
                    m.coefficients().unwrap()[0].clone()
                };
                let (b0, b1) = (run(0.0), run(1e3));
                let what = format!("{gaps:?}, standardize {standardize}");
                for i in 1..3 {
                    assert!(
                        (b0[i] - b1[i]).abs() < 1e-9,
                        "{what}: slope {i}: {} vs {}",
                        b0[i],
                        b1[i]
                    );
                }
                assert!(
                    (b1[0] - b0[0] - 1e3).abs() < 1e-8,
                    "{what}: the intercept takes the level"
                );
            }
        }
    }

    /// A blend under `own_rows` mixes each Gram with the twin's Gram of the
    /// same targets. The twin sees the same rows and targets, so it splits
    /// where the fast side does, and each target's blended accumulators are,
    /// to the bit, those of a model of that target alone blended at the same
    /// boundary.
    #[test]
    fn a_blend_under_own_rows_is_each_targets_own_blend() {
        let mut c = cfg(2, 3);
        c.decay = Decay::Halflife(15.0);
        c.session_shrink = Some(0.4);
        c.long_halflife = Some(200.0);
        c.min_periods = 3.0;
        let mut bank = EwRidge::new(c.clone()).unwrap();
        let one = EwRidgeCfg {
            n_targets: 1,
            ..c.clone()
        };
        let mut alone: Vec<EwRidge> = (0..3).map(|_| EwRidge::new(one.clone()).unwrap()).collect();
        for (x, y, d, w) in gappy_rows(200) {
            bank.step(&x, &y, d, w);
            for (j, a) in alone.iter_mut().enumerate() {
                a.step(&x, &[y[j]], d, w);
            }
        }
        let twin = bank.slow.as_ref().unwrap();
        assert_eq!(
            twin.grams.of, bank.acc.grams.of,
            "the twin splits where the fast side does"
        );
        bank.blend_toward_long_run();
        for a in &mut alone {
            a.blend_toward_long_run();
        }
        for (j, a) in alone.iter().enumerate() {
            assert_eq!(
                bank.acc.grams.grams[bank.acc.grams.of[j]],
                *a.gram(0),
                "target {j}"
            );
            assert_eq!(bank.acc.wj[j], a.acc.wj[0], "target {j}");
            assert_eq!(bank.acc.cross.c[j], a.acc.cross.c[0], "target {j}");
            let (b, want) = (
                &bank.coefficients().unwrap()[j],
                &a.coefficients().unwrap()[0],
            );
            for (p, q) in b.iter().zip(want) {
                assert!(
                    (p - q).abs() <= 1e-12 * (1.0 + q.abs()),
                    "target {j}: {b:?} vs {want:?}"
                );
            }
        }
    }

    /// `pairwise` keeps one Gram over every row whatever the gaps, and its
    /// slopes are pairwise-complete moments: the Gram's centred co-moments
    /// against each target's own centred cross-moments.
    #[test]
    fn pairwise_keeps_one_gram_over_every_row() {
        let mut c = cfg(2, 3);
        c.decay = Decay::Halflife(30.0);
        c.target_gaps = TargetGaps::Pairwise;
        let mut m = EwRidge::new(c).unwrap();
        for (x, y, d, w) in gappy_rows(200) {
            m.step(&x, &y, d, w);
        }
        assert_eq!(m.acc.grams.grams.len(), 1);
        assert_eq!(m.gram(0).n_eff(), m.n_eff(), "the Gram is over every row");
        assert!(
            m.acc.wj[2] < m.acc.wj[0],
            "a target with gaps has less weight"
        );
    }

    /// A row with no weight learns nothing, so it parts no targets: one
    /// that is null on a zero-weight row keeps its Gram, which is still, to
    /// the bit, the Gram of a model of that target alone. Only a zero-weight
    /// row that also takes all of the weight -- `lam = 0`, an infinite clock
    /// gap -- splits them, since there taking the row and skipping it age
    /// the Gram differently.
    #[test]
    fn a_zero_weight_row_splits_no_gram() {
        let mut c = cfg(2, 2);
        c.decay = Decay::Halflife(20.0);
        let mut bank = EwRidge::new(c.clone()).unwrap();
        let one = EwRidgeCfg { n_targets: 1, ..c };
        let mut alone = [
            EwRidge::new(one.clone()).unwrap(),
            EwRidge::new(one).unwrap(),
        ];
        let mut s = 71u64;
        for i in 0..120 {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.0 + x[0] - 2.0 * x[1];
            // Target 1 is null on exactly the zero-weight rows.
            let w = if i % 10 == 9 { 0.0 } else { 1.0 };
            let ys = [Some(y), (w > 0.0).then_some(2.0 * y)];
            let d = if i == 0 { 0.0 } else { 1.0 };
            bank.step(&x, &ys, d, w);
            for (j, a) in alone.iter_mut().enumerate() {
                a.step(&x, &[ys[j]], d, w);
            }
        }
        assert_eq!(
            bank.acc.grams.grams.len(),
            1,
            "a zero-weight row split them"
        );
        for (j, a) in alone.iter().enumerate() {
            assert_eq!(bank.gram(0), a.gram(0), "target {j}");
            assert_eq!(bank.acc.wj[j], a.acc.wj[0], "target {j}");
        }
        bank.step(&[0.1, 0.2], &[Some(1.0), None], f64::INFINITY, 0.0);
        assert_eq!(
            bank.acc.grams.grams.len(),
            2,
            "an infinite gap does split them"
        );
        let of = &bank.acc.grams.of;
        assert_eq!(bank.acc.grams.grams[of[1]].n_eff(), 0.0);
        assert_eq!(bank.acc.grams.grams[of[0]].n_eff(), bank.acc.wj[0]);
    }

    /// Under `own_rows` a target not seen yet is alone in a Gram with no
    /// weight. With `ridge = 0` there is no system there, so its
    /// coefficients stay zero and no failure is counted; a solve of the empty
    /// Gram was a jittered one, counted, on every row until the target came.
    #[test]
    fn a_target_not_seen_yet_costs_no_solve_failure() {
        let mut c = cfg(2, 2);
        c.ridge = vec![0.0];
        c.min_periods = 3.0;
        let mut m = EwRidge::new(c).unwrap();
        let mut s = 73u64;
        let mut row = |m: &mut EwRidge, i: usize| {
            let x = [lcg(&mut s), lcg(&mut s)];
            let y = 1.0 + x[0] - 2.0 * x[1] + 0.01 * lcg(&mut s);
            m.step(&x, &[Some(y), None], if i == 0 { 0.0 } else { 1.0 }, 1.0);
        };
        // The first rows are fewer than the terms: target 0's own solves are
        // singular there, and counted, as least squares on them would be.
        for i in 0..10 {
            row(&mut m, i);
        }
        let warm = m.solve_failures;
        for i in 10..60 {
            row(&mut m, i);
        }
        assert_eq!(m.acc.grams.grams.len(), 2);
        assert_eq!(m.solve_failures, warm, "the empty Gram was solved");
        assert!(m.coefficients().unwrap()[1].iter().all(|&b| b == 0.0));
        assert!(m.predict(&[0.1, 0.2], 1.0).pred[1].is_nan());
    }
}
